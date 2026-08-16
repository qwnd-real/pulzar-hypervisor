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
//! Every function here is about the timer of the processor that calls it: each
//! pairs the [`Vlapic`] it is handed with [`apic::local`], which answers for
//! the processor doing the reaching and not for the one whose row it was given.
//! A caller holding another processor's row would stop its own timer while
//! reporting that it had stopped that one's, and nothing local could notice.
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
//! # The one thing the guest asks for and does not get
//!
//! An unmasked periodic period shorter than the floor in [`clamp`] is
//! lengthened to it, which is the only place the real timer deliberately
//! differs from what the guest programmed. Nothing about it is concealed: the
//! count a guest reads out of its current-count register is hardware's own, so
//! a guest measuring its timer's rate measures the rate it is being given
//! rather than the one it asked for.
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

use apic::{ApicError, Divisor, LocalApic, Source, TimerMode as HardwareMode};
use descriptors::Vector;
use log::{Level, log_enabled, trace, warn};

pub use crate::hardware::timer::deadline::adjust_deadline;
pub(crate) use crate::hardware::timer::{
    deadline::arm_deadline,
    inherit::{calibrate, inherit},
};
use crate::{
    hardware::{
        sources::{self, Refusal},
        timer::clamp::{Reconfigured, floor, reconfigure},
    },
    machine::diagnostics::Report,
    registers::{
        Vlapic,
        lvt::{Entry, TimerMode},
    },
};

/// What the guest's timer has left to count.
///
/// Read from the hardware, because the hardware is what is counting — including
/// where hardware is counting a longer period than the guest asked for, which
/// is the whole of how a guest can tell that it is.
///
/// Zero in the two modes that count nothing: deadline mode, where the
/// architecture defines this as reading zero, and the encoding the architecture
/// reserves, where the timer is stopped and a physical count would be one for a
/// timer the guest never started.
pub(crate) fn remaining(vlapic: &Vlapic) -> u32 {
    if !matches!(
        vlapic.timer_mode(),
        Some(TimerMode::OneShot | TimerMode::Periodic)
    ) {
        return 0;
    }
    apic::local().map_or(0, |local| local.timer().remaining())
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
#[must_use]
pub(crate) fn reprogram(vlapic: &Vlapic) -> bool {
    let Ok(timer) = apic::local().map(LocalApic::timer) else {
        warn!(
            "vlapic: {} could not reach its controller to program its timer",
            vlapic.index()
        );
        return false;
    };
    let Some(mode) = hardware_mode(vlapic) else {
        // The encoding the architecture reserves. A controller given it does
        // nothing defined, so the timer is stopped and left delivering nothing.
        return disarm(vlapic);
    };
    let divisor = divisor(vlapic);
    let delivery = armed(vlapic);
    match reconfigure(
        &timer,
        delivery,
        mode,
        divisor,
        floor(vlapic.timer_frequency(), divisor),
    ) {
        Ok(Reconfigured::Applied) => {
            report(vlapic, "configured");
            true
        }
        Ok(Reconfigured::Raised { from, to }) => {
            report_floor(vlapic, from, to);
            report(vlapic, "configured");
            true
        }
        // Stopped rather than armed, and so not in agreement with the guest's
        // entry: the guest's timer has stopped and its own registers say it
        // should be running.
        Ok(Reconfigured::Stopped) => {
            sources::refused(
                vlapic,
                Entry::Timer,
                Refusal::Rate(vlapic.timer_frequency()),
            );
            report(vlapic, "stopped");
            false
        }
        Err(error) => {
            sources::refused(vlapic, Entry::Timer, Refusal::Hardware(error));
            // A configuration hardware refused is one it never took, so the
            // entry still holds whatever it held before — quite possibly a
            // vector the guest has stopped asking for, unmasked. Stopping the
            // timer is what keeps a refusal from leaving a source delivering,
            // and whether the stop itself succeeded is [`disarm`]'s to say: the
            // answer here is that hardware does not hold what the guest asked
            // for.
            let _stopped = disarm(vlapic);
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
///
/// A refusal is reported here, once per controller, because the guest's own
/// write has already succeeded and no register the architecture defines has a
/// bit for "your controller did not start the timer". The answer is returned as
/// well, for the one caller that has something of its own to say about it.
///
/// # Errors
///
/// [`ApicError::Calibration`] where the timer's own rate is not known well
/// enough to say whether the guest's period is one the machine can answer, in
/// which case the timer is stopped rather than started; otherwise whatever the
/// controller refused.
pub(crate) fn reload(vlapic: &Vlapic) -> Result<(), ApicError> {
    let timer = apic::local()?.timer();
    if hardware_mode(vlapic).is_none() {
        // The encoding the architecture reserves defines nothing for a count
        // either, so nothing is started and nothing has gone wrong. `reprogram`
        // stopped the timer when the mode was selected, and `remaining` reports
        // zero for it, so the guest sees a timer that is doing nothing — which
        // is all the architecture promises about this encoding.
        return Ok(());
    }
    let guest_count = vlapic.timer_initial();
    let Some(count) = physical_count(vlapic, guest_count) else {
        // Fail closed: a count that cannot be shown to be long enough is not put
        // on hardware at all.
        let _stopped = disarm(vlapic);
        sources::refused(
            vlapic,
            Entry::Timer,
            Refusal::Rate(vlapic.timer_frequency()),
        );
        return Err(ApicError::Calibration);
    };
    if let Err(error) = timer.reload(count) {
        sources::refused(vlapic, Entry::Timer, Refusal::Hardware(error));
        return Err(error);
    }
    if count != guest_count {
        report_floor(vlapic, guest_count, count);
    }
    report(vlapic, "started");
    Ok(())
}

/// Stops the guest's timer delivering, leaving what it is counting alone.
///
/// The masking half of a configuration, on its own, for the two moments the
/// ordering matters — a write that masks the entry, and a controller the guest
/// has software-disabled. Both reach hardware before the register file records
/// them, and the argument for that is in `face::dispatch`, where the
/// decision is made.
///
/// Masking is not cancelling: the mode, the count and any armed deadline are
/// left exactly where they were, so a guest that unmasks the entry again finds
/// the timer where it left it. That is what this has that [`disarm`] does not,
/// and it is the whole reason the timer cannot be masked the way the other
/// sources are — writing a masked entry over it would put its mode down with
/// it, and a mode change disarms the timer.
///
/// Answers whether it really was masked.
pub(crate) fn mask(vlapic: &Vlapic) -> bool {
    let Ok(timer) = apic::local().map(LocalApic::timer) else {
        warn!(
            "vlapic: {} could not reach its controller to mask its timer",
            vlapic.index()
        );
        return false;
    };
    let Some(mode) = hardware_mode(vlapic) else {
        // The reserved encoding, for which `reprogram` has already stopped the
        // timer: there is nothing to mask and nothing that can deliver.
        return true;
    };
    match timer.configure(None, mode, divisor(vlapic)) {
        Ok(()) => true,
        Err(error) => {
            sources::refused(vlapic, Entry::Timer, Refusal::Hardware(error));
            false
        }
    }
}

/// Stops the guest's timer and stops it delivering.
///
/// What a controller transition needs: an old timer left running would deliver
/// into a guest that has been reset, or after the controller it belonged to was
/// switched off.
#[must_use]
pub(crate) fn disarm(vlapic: &Vlapic) -> bool {
    let Ok(timer) = apic::local().map(LocalApic::timer) else {
        warn!(
            "vlapic: {} could not reach its controller to disarm its timer",
            vlapic.index()
        );
        return false;
    };
    timer.disarm();
    trace!("vlapic: {} disarmed its timer", vlapic.index());
    true
}

/// What the timer is to deliver, or `None` for an entry that delivers nothing.
///
/// A masked entry is one, and so is one naming a vector no source may be armed
/// with here — which is refused for the timer by the same rule and reported the
/// same way as for every other source, because it is the same restriction.
fn armed(vlapic: &Vlapic) -> Option<Vector> {
    let entry = vlapic.lvt(Entry::Timer);
    if entry.masked() {
        return None;
    }
    let vector = entry.vector();
    if !sources::arms(vector) {
        sources::refused(vlapic, Entry::Timer, Refusal::Vector(vector));
        return None;
    }
    Some(vector)
}

/// The physical count the guest's own count has to be started at, or `None`
/// where that cannot be said.
///
/// The floor applies to an unmasked periodic timer and to nothing else: a
/// masked entry delivers nothing however fast it counts — which is exactly what
/// an operating system measuring its timer's rate programs — a one-shot fires
/// once, and a count of zero is how the architecture spells a stopped timer.
fn physical_count(vlapic: &Vlapic, guest_count: u32) -> Option<u32> {
    let periodic = vlapic.timer_mode() == Some(TimerMode::Periodic)
        && !vlapic.lvt(Entry::Timer).masked()
        && guest_count != 0;
    if !periodic {
        return Some(guest_count);
    }
    floor(vlapic.timer_frequency(), divisor(vlapic)).map(|floor| guest_count.max(floor))
}

/// The mode the real timer counts in for the mode the guest selected, or `None`
/// for the encoding the architecture reserves.
fn hardware_mode(vlapic: &Vlapic) -> Option<HardwareMode> {
    match vlapic.timer_mode()? {
        TimerMode::OneShot => Some(HardwareMode::OneShot),
        TimerMode::Periodic => Some(HardwareMode::Periodic),
        TimerMode::Deadline => Some(HardwareMode::Deadline),
    }
}

/// Says once that this guest's periodic timer is being given a longer period
/// than it asked for, and counts every time it happens.
///
/// `programmed` is the count that was going to be reloaded and `given` is what
/// is reloaded instead.
///
/// The line is once per controller, because the guest can rewrite the count as
/// fast as it can take an exit and each of those would otherwise be a line of
/// serial output with a machine-wide lock held. The count is every time,
/// because that is the record a machine with no serial port leaves behind — and
/// the guest is not relying on either to find out: its current-count reads are
/// hardware's, so the period it is being given is one it can measure.
fn report_floor(vlapic: &Vlapic, programmed: u32, given: u32) {
    vlapic.diagnostics().clamped();
    if vlapic.diagnostics().say(Report::TimerFloor) {
        warn!(
            "vlapic: {} is running its guest's periodic timer from a count of {given:#x} where \
             {programmed:#x} was programmed, that being the shortest period this hypervisor puts \
             on real hardware; the guest's current-count reads report the count hardware is really \
             counting",
            vlapic.index()
        );
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
///
/// Behind the log level rather than inside the record, because reaching the
/// controller at all is an uncached register read and reading the entry back is
/// another — two of them on every guest write to the entry, the divide or the
/// count, for a record the configured level throws away.
fn report(vlapic: &Vlapic, what: &str) {
    if !log_enabled!(Level::Trace) {
        return;
    }
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
