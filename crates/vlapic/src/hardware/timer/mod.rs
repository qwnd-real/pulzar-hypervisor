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
//!
//! The guest's timestamp counter is offset from the host's by its control
//! block. The model-specific-register face translates guest deadlines into the
//! physical counter's domain before they reach this module and translates
//! readback the other way. An offset change rebases an armed deadline here
//! before the new offset is published, so an absolute appointment remains the
//! same guest-visible number in the adjusted timestamp domain.

mod clamp;
mod deadline;
mod inherit;

use apic::{Divisor, LocalApic, Source, TimerMode as HardwareMode};
use log::{trace, warn};

pub use crate::hardware::timer::deadline::adjust_deadline;
pub(crate) use crate::hardware::timer::{
    deadline::arm_deadline,
    inherit::{calibrate, inherit},
};
use crate::{
    hardware::{
        sources,
        timer::clamp::{clamped_count, reconfigure, scale_remaining},
    },
    registers::{
        Vlapic,
        lvt::{Entry, TimerMode},
    },
};

/// What the guest's timer has left to count.
///
/// Read from the hardware, because the hardware is what is counting. In
/// deadline mode the architecture defines this as reading zero.
pub(crate) fn remaining(vlapic: &Vlapic) -> u32 {
    if vlapic.timer_mode() == Some(TimerMode::Deadline) {
        return 0;
    }
    let remaining = apic::local().map_or(0, |local| local.timer().remaining());
    scale_remaining(remaining, vlapic.timer_initial(), vlapic.timer_clamp())
}

/// The deadline the guest's timer is counting towards.
///
/// Zero outside deadline mode, which is what the architecture requires of the
/// read: the register means nothing in the counting modes and a guest reading
/// it there must not be handed a value it could act on. Zero, too, once the
/// deadline has fired, because that is what hardware leaves behind.
pub(crate) fn deadline(vlapic: &Vlapic) -> u64 {
    if vlapic.timer_mode() != Some(TimerMode::Deadline) {
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
/// request to start one. A deadline the guest has left behind by changing mode
/// is put down by the configuration itself, which is where both sides of the
/// change are known.
///
/// Answers whether the hardware agrees with the guest's entry afterwards.
pub(crate) fn reprogram(vlapic: &Vlapic) -> bool {
    let Ok(timer) = apic::local().map(LocalApic::timer) else {
        return false;
    };
    let entry = vlapic.lvt(Entry::Timer);
    let mode = match vlapic.timer_mode() {
        Some(TimerMode::OneShot) => HardwareMode::OneShot,
        Some(TimerMode::Periodic) => HardwareMode::Periodic,
        Some(TimerMode::Deadline) => HardwareMode::Deadline,
        // The encoding the architecture reserves. A controller given it does
        // nothing defined, so the timer is stopped and left delivering nothing.
        None => return disarm(vlapic),
    };
    let delivery = (!entry.masked())
        .then(|| entry.vector())
        .and_then(|vector| sources::armable(vlapic, vector));
    let divisor = divisor(vlapic);
    match reconfigure(vlapic, &timer, delivery, mode, divisor) {
        Ok(()) => {
            report(vlapic, "configured");
            true
        }
        Err(error) => {
            warn!(
                "vlapic: {} could not program its timer, so it is stopped: {error}",
                vlapic.index()
            );
            // A configuration hardware refused is one it never took, so the
            // entry still holds whatever it held before — quite possibly a
            // vector the guest has stopped asking for, unmasked. Stopping the
            // timer is what keeps a refusal from leaving a source delivering.
            disarm(vlapic);
            false
        }
    }
}

/// Says what the guest asked its timer for and what the real timer holds
/// afterwards.
///
/// The two are the same registers seen from either side, and the whole of what
/// makes a guest's timer work is that they agree: a mode the guest selected and
/// the hardware did not take, a vector masked on one side and not the other, or
/// a count the hardware refused to start are each invisible from the guest's
/// own reads and each stop its ticks.
fn report(vlapic: &Vlapic, what: &str) {
    let Ok(local) = apic::local() else {
        return;
    };
    let guest = vlapic.lvt(Entry::Timer);
    let real = local.source(Source::Timer);
    trace!(
        "vlapic: {} {what} its timer: guest {:?} vector {} {}, divide {:#x}, count {:#x}; real \
         mode {:?}, {}, remaining {:#x}",
        vlapic.index(),
        vlapic.timer_mode(),
        guest.vector(),
        if guest.masked() { "masked" } else { "armed" },
        vlapic.timer_divide(),
        vlapic.timer_initial(),
        local.timer().mode(),
        match real {
            Ok(entry) if entry.is_masked() => "masked",
            Ok(_) => "armed",
            Err(_) => "unreachable",
        },
        local.timer().remaining(),
    );
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
    let guest_count = vlapic.timer_initial();
    let physical_count = clamped_count(vlapic, guest_count).unwrap_or(guest_count);
    if let Err(error) = timer.reload(physical_count) {
        warn!(
            "vlapic: {} could not start its timer counting: {error}",
            vlapic.index()
        );
        vlapic.clear_timer_clamp();
        vlapic.set_timer_periodic_running(false);
        timer.disarm();
        return;
    }
    vlapic.set_timer_periodic_running(
        physical_count != 0 && vlapic.timer_mode() == Some(TimerMode::Periodic),
    );
    let clamp = if physical_count == guest_count {
        0
    } else {
        physical_count
    };
    vlapic.set_timer_clamp(clamp);
    if physical_count != guest_count && vlapic.report_timer_clamp_once() {
        warn!(
            "vlapic: {} limited a periodic timer count from {guest_count:#x} to {physical_count:#x}",
            vlapic.index()
        );
    }
    report(vlapic, "started");
}

/// Stops the guest's timer and stops it delivering.
///
/// What a controller transition needs: an old timer left running would deliver
/// into a guest that has been reset, or after the controller it belonged to was
/// switched off.
pub(crate) fn disarm(vlapic: &Vlapic) -> bool {
    let Ok(timer) = apic::local().map(LocalApic::timer) else {
        warn!(
            "vlapic: {} could not reach its controller to disarm its timer",
            vlapic.index()
        );
        return false;
    };
    timer.disarm();
    vlapic.clear_timer_clamp();
    vlapic.set_timer_periodic_running(false);
    trace!("vlapic: {} disarmed its timer", vlapic.index());
    true
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
