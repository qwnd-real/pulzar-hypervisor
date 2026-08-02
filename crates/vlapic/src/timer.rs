//! The guest's timer, which is the real timer.
//!
//! The local controller's timer is the one part of it that is not emulated at
//! all. The guest's vector, divide, count and mode are programmed straight onto
//! the hardware and the hardware counts them, because there is nothing to be
//! gained by counting them again in software: a timer is a decrementing
//! register and a comparison, and this hypervisor has no reason to lie about
//! either.
//!
//! Nothing else on the machine uses this processor's timer, which is what makes
//! handing it over wholesale possible — and what makes the hardware registers
//! the honest place to read the guest's timer state back from.
//!
//! # Configuring a timer is not starting one
//!
//! The distinction runs through this whole module and it is the architecture's,
//! not a refinement of it. Writing the entry says what the timer delivers and
//! in which mode; writing the divide says how fast it counts; and neither
//! starts anything. A counting timer starts when the initial count is written,
//! and a deadline timer arms when the deadline is written.
//!
//! Conflating them breaks a guest in ways that are hard to see and impossible
//! to work around. A guest that masks its timer, or changes its vector, or
//! writes the same divide back, has not asked for the count to restart — but a
//! hypervisor that reprogrammed everything on every touch would move the phase
//! of a periodic tick each time, resurrect a one-shot that had already fired,
//! and re-arm a deadline the hardware had already cleared. Operating systems
//! touch these registers constantly and rely on the ones they did not write
//! standing still.
//!
//! # Masking is not cancelling
//!
//! The mask bit suppresses the interrupt and nothing else. The count still runs
//! and software can still read it, and that is not a corner worth being
//! approximate about: masking the entry, writing a count and watching it fall
//! against a clock it already trusts is exactly how an operating system
//! measures what its timer's rate is, and it is one of the first things it
//! does.
//!
//! A masked deadline is the same rule seen from the other side. The deadline
//! keeps running towards a moment that is fixed in absolute time; masking says
//! only that nothing is delivered if it arrives while masked. A hypervisor that
//! cleared the deadline register instead would lose an appointment the guest
//! cannot re-derive.
//!
//! # Deadline mode reads and writes through hardware
//!
//! The deadline is not mirrored in this crate. Hardware clears the register
//! when the deadline fires, and nothing about that expiry is visible anywhere
//! else — the interrupt arrives as an ordinary vector carrying no hint of where
//! it came from. A software copy would therefore go stale exactly once per
//! expiry and would then be re-armed by the next reconfiguration, firing
//! immediately from a moment already in the past. Reading the register is both
//! simpler and the only thing that is correct.

use apic::{Divisor, LocalApic, TimerMode as HardwareMode};
use log::warn;

use crate::{
    lvt::{Entry, TimerMode},
    state::Vlapic,
};

/// What the guest's timer has left to count.
///
/// Read from the hardware, because the hardware is what is counting. In
/// deadline mode the architecture defines this as reading zero.
pub(crate) fn remaining(vlapic: &Vlapic) -> u32 {
    if mode_of(vlapic) == Some(TimerMode::Deadline) {
        return 0;
    }
    apic::local()
        .and_then(|local| local.timer().remaining())
        .unwrap_or(0)
}

/// The deadline the guest's timer is counting towards.
///
/// Zero outside deadline mode, which is what the architecture requires of the
/// read: the register means nothing in the counting modes and a guest reading
/// it there must not be handed a value it could act on. Zero, too, once the
/// deadline has fired, because that is what hardware leaves behind.
pub(crate) fn deadline(vlapic: &Vlapic) -> u64 {
    if mode_of(vlapic) != Some(TimerMode::Deadline) {
        return 0;
    }
    apic::local()
        .and_then(|local| local.timer().deadline())
        .unwrap_or(0)
}

/// Brings the real timer's configuration into agreement with the guest's,
/// starting nothing.
///
/// Called whenever the guest writes the timer's entry or its divide. What is
/// deliberately absent is any write to the count or to the deadline: those are
/// what start a timer, and neither of the registers this answers for is a
/// request to start one.
///
/// The one thing it does put down is a deadline the guest has left behind by
/// changing mode. A deadline belongs to deadline mode; carried across a mode
/// change it would be a moment in the past waiting to fire the instant the
/// guest came back.
///
/// Answers whether the hardware agrees with the guest's entry afterwards.
pub(crate) fn reprogram(vlapic: &Vlapic) -> bool {
    let Ok(timer) = apic::local().map(LocalApic::timer) else {
        return false;
    };
    let entry = vlapic.lvt(Entry::Timer);
    let asked = mode_of(vlapic);
    let was = timer.mode().unwrap_or(None);
    let mode = match asked {
        Some(TimerMode::OneShot) => HardwareMode::OneShot,
        Some(TimerMode::Periodic) => HardwareMode::Periodic,
        Some(TimerMode::Deadline) => HardwareMode::Deadline,
        // The encoding the architecture reserves. A controller given it does
        // nothing defined, so the timer is stopped and left delivering nothing.
        None => return disarm(vlapic),
    };
    // Entering deadline mode leaves the timer disarmed until the guest writes a
    // deadline, and leaving it puts down whatever was outstanding. Both are the
    // same rule: a deadline only means something while the mode it belongs to is
    // selected.
    if (mode == HardwareMode::Deadline) != (was == Some(HardwareMode::Deadline))
        && let Err(error) = timer.set_deadline(0)
    {
        warn!(
            "vlapic: {} could not put down its timer deadline: {error}",
            vlapic.index()
        );
    }
    let delivery = (!entry.masked())
        .then(|| entry.vector())
        .filter(|vector| crate::sources::arms(*vector));
    match timer.configure(delivery, mode, divisor(vlapic)) {
        Ok(()) => true,
        Err(error) => {
            warn!(
                "vlapic: {} could not program its timer: {error}",
                vlapic.index()
            );
            false
        }
    }
}

/// Starts the guest's timer counting from what it last wrote.
///
/// The one operation that starts a counting timer, and it is reached only from
/// a write to the initial count — which is the write the architecture defines
/// as starting one. A count of zero stops the timer rather than firing it,
/// which is the architecture's own spelling and needs no special case here.
pub(crate) fn reload(vlapic: &Vlapic) {
    let Ok(timer) = apic::local().map(LocalApic::timer) else {
        return;
    };
    if let Err(error) = timer.reload(vlapic.timer_initial()) {
        warn!(
            "vlapic: {} could not start its timer: {error}",
            vlapic.index()
        );
    }
}

/// Arms the guest's timer at the deadline it just wrote.
pub(crate) fn arm_deadline(vlapic: &Vlapic, deadline: u64) {
    let Ok(timer) = apic::local().map(LocalApic::timer) else {
        return;
    };
    if let Err(error) = timer.set_deadline(deadline) {
        warn!(
            "vlapic: {} could not arm its timer deadline: {error}",
            vlapic.index()
        );
    }
}

/// Stops the guest's timer and stops it delivering.
///
/// What a controller transition needs: an old timer left running would deliver
/// into a guest that has been reset, or after the controller it belonged to was
/// switched off.
pub(crate) fn disarm(vlapic: &Vlapic) -> bool {
    let Ok(timer) = apic::local().map(LocalApic::timer) else {
        return false;
    };
    match timer.disarm() {
        Ok(()) => true,
        Err(error) => {
            warn!(
                "vlapic: {} could not disarm its timer: {error}",
                vlapic.index()
            );
            false
        }
    }
}

/// Which mode the guest's entry selects, or `None` for the encoding the
/// architecture reserves.
pub(crate) fn mode_of(vlapic: &Vlapic) -> Option<TimerMode> {
    TimerMode::from_bits(vlapic.lvt(Entry::Timer).timer_mode())
}

/// How far the guest asked for the clock to be divided.
///
/// The encoding is three bits that are not adjacent — bit two sits between them
/// and is reserved — and every one of the eight it can name is a divisor the
/// hardware offers, so this cannot fail.
fn divisor(vlapic: &Vlapic) -> Divisor {
    let value = vlapic.timer_divide();
    match (value & 0b11) | ((value & 0b1000) >> 1) {
        0b000 => Divisor::By2,
        0b001 => Divisor::By4,
        0b010 => Divisor::By8,
        0b011 => Divisor::By16,
        0b100 => Divisor::By32,
        0b101 => Divisor::By64,
        0b110 => Divisor::By128,
        _ => Divisor::By1,
    }
}
