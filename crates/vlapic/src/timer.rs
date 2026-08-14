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

use apic::{Divisor, LocalApic, LocalState, Source, TimerMode as HardwareMode};
use clock::{Frequency, Kind};
use log::{trace, warn};
use x86_64::instructions::interrupts;

use crate::{
    lvt::{Entry, TimerMode},
    sources,
    state::Vlapic,
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
    match reconfigure(vlapic, timer, delivery, mode, divisor) {
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

/// Rebases an armed physical deadline across a guest timestamp-offset change.
///
/// `adjustment` is the amount added to the old offset. The physical deadline
/// moves by the opposite amount so that adding the new offset still produces
/// the same guest-visible absolute deadline. A disarmed deadline and every
/// counting mode have nothing to rebase.
///
/// # Errors
///
/// The error returned by the physical controller if its timer cannot be read or
/// rewritten.
pub(crate) fn adjust_deadline(vlapic: &Vlapic, adjustment: u64) -> Result<(), apic::ApicError> {
    if adjustment == 0 || vlapic.timer_mode() != Some(TimerMode::Deadline) {
        return Ok(());
    }
    interrupts::without_interrupts(|| {
        let timer = apic::local()?.timer();
        let deadline = timer.deadline()?;
        if deadline == 0 {
            return Ok(());
        }
        timer.set_deadline(rebased_deadline(deadline, adjustment))
    })
}

/// Measures and records this processor's undivided local-timer frequency.
///
/// Called before the timer is handed to the guest. Calibration temporarily
/// uses the physical timer and leaves it stopped, so doing this later would
/// destroy an appointment the guest had already made.
///
/// # Errors
///
/// The error returned by the physical timer if its rate cannot be measured.
pub(crate) fn calibrate(vlapic: &Vlapic) -> Result<(), apic::ApicError> {
    if vlapic.timer_frequency() != 0 {
        return Ok(());
    }
    let frequency = apic::local()?.timer().calibrate(Divisor::By1)?;
    vlapic.set_timer_frequency(frequency.hz());
    vlapic.clear_timer_clamp();
    vlapic.set_timer_periodic_running(false);
    trace!(
        "vlapic: {} calibrated its undivided timer at {} Hz",
        vlapic.index(),
        frequency.hz()
    );
    Ok(())
}

/// Starts the guest's timer where firmware's was, having seeded the entry it
/// runs against.
///
/// The register file has already been seeded, so what the timer delivers and in
/// which mode is settled; what is left is the half no configuration write
/// performs, which is starting it. Firmware may have had a timer counting for
/// its own purposes, and the host stopped it during bring-up — so leaving it
/// stopped would be an appointment firmware made and can no longer re-derive.
///
/// The three modes are not restored alike, because only one of them holds
/// something that is still true:
///
/// - A **deadline** is absolute. It is written back exactly, and one that has
///   passed in the meantime fires at once, which is what firmware was owed.
/// - A **periodic** timer is reloaded from its initial count. The period is
///   exact and the phase moves by up to one period, which nothing could
///   reconstruct: the count firmware was at says where it is in *this* period,
///   not where the period began.
/// - A **one-shot** is aged. Its remaining count is a duration into the future,
///   so however long the capture has been sitting in the chunk has to come off
///   it, or firmware's appointment lands late by the whole of bring-up.
pub(crate) fn inherit(vlapic: &Vlapic, firmware: &LocalState, since: u64) {
    reprogram(vlapic);
    match vlapic.timer_mode() {
        Some(TimerMode::Deadline) if firmware.tsc_deadline != 0 => {
            arm_deadline(vlapic, firmware.tsc_deadline);
        }
        Some(TimerMode::Periodic) => reload(vlapic),
        Some(TimerMode::OneShot) if firmware.timer_current_count != 0 => {
            oneshot(vlapic, firmware, since);
        }
        // Three ways to have nothing to start, and `reprogram` has left the
        // timer stopped for all of them: a deadline mode firmware had not armed,
        // a one-shot that had already run out, and the encoding the architecture
        // reserves, which a controller given it does nothing defined with.
        _ => {}
    }
}

/// Restarts a one-shot firmware left counting, less however long it has been
/// since.
///
/// The count is in divided timer ticks and the elapsed span is in timestamp
/// counter ticks. The timer's undivided rate was measured once before guest
/// ownership and is divided by the guest's current configuration here; the
/// timestamp counter's rate comes from the timebase itself. A remainder that
/// has already run out is armed at one tick rather than zero, because zero is
/// how the architecture spells a stopped timer and would drop the appointment
/// entirely.
///
/// Where the timebase is not the timestamp counter, what firmware had left is
/// armed as it stands and the reason is logged. That fires late by the length
/// of bring-up, which is a worse answer than the aged one and a far better
/// answer than never.
fn oneshot(vlapic: &Vlapic, firmware: &LocalState, since: u64) {
    let left = firmware.timer_current_count;
    let count = match elapsed_ticks(vlapic, since) {
        Some(elapsed) => left.saturating_sub(elapsed).max(SOONEST),
        None => left,
    };
    let Ok(timer) = apic::local().map(LocalApic::timer) else {
        return;
    };
    if let Err(error) = timer.reload(count) {
        warn!(
            "vlapic: {} could not restart firmware's one-shot timer: {error}",
            vlapic.index()
        );
        return;
    }
    vlapic.set_timer_periodic_running(false);
    trace!(
        "vlapic: {} restarted firmware's one-shot timer at {count} of the {left} it had left",
        vlapic.index(),
    );
}

/// How many of this timer's ticks have passed since the capture was taken, or
/// `None` where that cannot be worked out.
fn elapsed_ticks(vlapic: &Vlapic, since: u64) -> Option<u32> {
    let source = clock::source()?;
    if source.kind() != Kind::Tsc {
        warn!(
            "vlapic: {} cannot age firmware's one-shot timer, the timebase is the {} rather than \
             the timestamp counter",
            vlapic.index(),
            source.kind()
        );
        return None;
    }
    let rate = Frequency::from_hz(vlapic.timer_frequency())?;
    // SAFETY: `RDTSC` is always permitted at privilege level zero, whatever
    // `CR4.TSD` says, and reading the counter does not disturb it.
    let now = unsafe { core::arch::x86_64::_rdtsc() };
    // Through nanoseconds rather than by a ratio of the two rates, because that
    // is the one conversion both clocks already offer and it needs no arithmetic
    // of its own to be right about.
    let nanos = source.frequency().nanos(now.saturating_sub(since));
    let elapsed = rate.ticks(nanos) / u64::from(divisor(vlapic).ratio());
    u32::try_from(elapsed).ok().or(Some(u32::MAX))
}

/// The fewest ticks a restarted timer is armed at.
///
/// One rather than zero, because the architecture reads a count of zero as a
/// stopped timer rather than as one due immediately — so a deadline that has
/// already passed has to be spelled as the soonest reachable one instead.
const SOONEST: u32 = 1;

/// The shortest unmasked periodic interval exposed to physical hardware.
const MINIMUM_PERIOD_NANOS: u64 = 200_000;

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

/// Reconfigures a timer, lengthening an already-running unsafe period without
/// exposing the short count on an unmasked physical entry.
fn reconfigure(
    vlapic: &Vlapic,
    timer: apic::Timer,
    delivery: Option<descriptors::Vector>,
    mode: HardwareMode,
    divisor: Divisor,
) -> Result<(), apic::ApicError> {
    let active = vlapic.timer_clamp();
    let periodic = delivery.is_some() && mode == HardwareMode::Periodic;
    let desired = if periodic {
        clamped_count(vlapic, vlapic.timer_initial()).unwrap_or(0)
    } else {
        active
    };
    let remaining = timer.remaining();
    if periodic && desired != active && (remaining != 0 || vlapic.timer_periodic_running()) {
        let physical_count = if desired == 0 {
            vlapic.timer_initial()
        } else {
            desired
        };
        timer.configure(None, mode, divisor)?;
        timer.reload(physical_count)?;
        timer.configure(delivery, mode, divisor)?;
        vlapic.set_timer_clamp(desired);
        vlapic.set_timer_periodic_running(physical_count != 0);
        if desired != 0 && vlapic.report_timer_clamp_once() {
            warn!(
                "vlapic: {} limited a running periodic timer count from {:#x} to {desired:#x}",
                vlapic.index(),
                vlapic.timer_initial()
            );
        }
        return Ok(());
    }
    timer.configure(delivery, mode, divisor)?;
    if mode != HardwareMode::Periodic {
        vlapic.set_timer_periodic_running(false);
    } else if remaining != 0 {
        vlapic.set_timer_periodic_running(true);
    }
    Ok(())
}

/// The physical count required to enforce the minimum periodic interval.
fn clamped_count(vlapic: &Vlapic, guest_count: u32) -> Option<u32> {
    if guest_count == 0
        || vlapic.timer_mode() != Some(TimerMode::Periodic)
        || vlapic.lvt(Entry::Timer).masked()
    {
        return None;
    }
    let minimum = minimum_period_count(vlapic.timer_frequency(), divisor(vlapic))?;
    (guest_count < minimum).then_some(minimum)
}

/// The least physical count that lasts the enforced periodic interval.
fn minimum_period_count(frequency: u64, divisor: Divisor) -> Option<u32> {
    let frequency = Frequency::from_hz(frequency)?;
    let undivided = frequency.ticks_ceil(MINIMUM_PERIOD_NANOS);
    let divided = undivided.div_ceil(u64::from(divisor.ratio())).max(1);
    Some(u32::try_from(divided).unwrap_or(u32::MAX))
}

/// Maps a physically lengthened countdown back into the guest's count range.
fn scale_remaining(remaining: u32, guest_initial: u32, physical_initial: u32) -> u32 {
    if physical_initial == 0 || guest_initial == 0 {
        return remaining;
    }
    let scaled =
        (u128::from(remaining) * u128::from(guest_initial)).div_ceil(u128::from(physical_initial));
    u32::try_from(scaled).unwrap_or(u32::MAX).min(guest_initial)
}

/// Moves a physical deadline opposite to a guest timestamp-offset change.
///
/// Physical zero disarms the deadline timer. The single wrapping combination
/// that would produce it is therefore represented by the earliest armed value
/// instead, one physical timestamp tick away.
const fn rebased_deadline(deadline: u64, adjustment: u64) -> u64 {
    let rebased = deadline.wrapping_sub(adjustment);
    if rebased == 0 { 1 } else { rebased }
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

#[cfg(test)]
mod tests {
    use apic::Divisor;

    use super::{minimum_period_count, rebased_deadline, scale_remaining};

    #[test]
    fn minimum_period_counts_round_up_after_division() {
        assert_eq!(
            minimum_period_count(100_000_000, Divisor::By1),
            Some(20_000)
        );
        assert_eq!(
            minimum_period_count(100_000_001, Divisor::By2),
            Some(10_001)
        );
        assert_eq!(minimum_period_count(1, Divisor::By128), Some(1));
        assert_eq!(minimum_period_count(0, Divisor::By1), None);
    }

    #[test]
    fn clamped_current_counts_stay_in_the_guest_range() {
        assert_eq!(scale_remaining(20_000, 1_000, 20_000), 1_000);
        assert_eq!(scale_remaining(10_000, 1_000, 20_000), 500);
        assert_eq!(scale_remaining(1, 1_000, 20_000), 1);
        assert_eq!(scale_remaining(0, 1_000, 20_000), 0);
    }

    #[test]
    fn unclamped_current_counts_are_unchanged() {
        assert_eq!(scale_remaining(123, 456, 0), 123);
        assert_eq!(scale_remaining(123, 0, 456), 123);
    }

    #[test]
    fn deadline_rebasing_moves_opposite_to_the_offset() {
        assert_eq!(rebased_deadline(0x2000, 0x300), 0x1D00);
        assert_eq!(rebased_deadline(0x100, u64::MAX), 0x101);
        assert_eq!(rebased_deadline(0x1234, 0x1234), 1);
    }
}
