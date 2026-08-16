//! Taking the timer over: measuring what it counts at, and restarting whatever
//! firmware had running on it.
//!
//! Both are done once, before the guest owns the timer, and the order is
//! forced. Calibration uses the physical timer and leaves it stopped, so
//! measuring after the guest had programmed something would destroy an
//! appointment it had already made — and pulzar does not boot a fresh guest, so
//! there is very often such an appointment to keep.

use apic::{Divisor, LocalState};
use clock::{Frequency, Kind};
use log::{trace, warn};

use crate::{
    hardware::timer::{arm_deadline, divisor, reload, reprogram},
    registers::{Vlapic, lvt::TimerMode},
};

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
    if !reprogram(vlapic) {
        warn!(
            "vlapic: {} could not put firmware's timer configuration on real hardware, so \
             firmware's own appointment is not restarted",
            vlapic.index()
        );
        return;
    }
    let restarted = match vlapic.timer_mode() {
        Some(TimerMode::Deadline) if firmware.tsc_deadline != 0 => {
            arm_deadline(vlapic, firmware.tsc_deadline)
        }
        Some(TimerMode::Periodic) => reload(vlapic),
        Some(TimerMode::OneShot) if firmware.timer_current_count != 0 => {
            oneshot(vlapic, firmware, since)
        }
        // Three ways to have nothing to start, and `reprogram` has left the
        // timer stopped for all of them: a deadline mode firmware had not armed,
        // a one-shot that had already run out, and the encoding the architecture
        // reserves, which a controller given it does nothing defined with.
        _ => Ok(()),
    };
    if let Err(error) = restarted {
        warn!(
            "vlapic: {} could not restart the timer firmware left running, so firmware's next \
             appointment is one nobody is keeping: {error}",
            vlapic.index()
        );
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
///
/// # Errors
///
/// Whatever the controller refused the count for.
fn oneshot(vlapic: &Vlapic, firmware: &LocalState, since: u64) -> Result<(), apic::ApicError> {
    let left = firmware.timer_current_count;
    let count = match elapsed_ticks(vlapic, since) {
        Some(elapsed) => left.saturating_sub(elapsed).max(SOONEST),
        None => left,
    };
    apic::local()?.timer().reload(count)?;
    trace!(
        "vlapic: {} restarted firmware's one-shot timer at {count} of the {left} it had left",
        vlapic.index(),
    );
    Ok(())
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
    // Read through the same wrapper every other reading of this counter in the
    // workspace goes through, including the one the capture took and the one
    // being subtracted here.
    let now = processor::timestamp();
    // Through nanoseconds rather than by a ratio of the two rates, because that
    // is the one conversion both clocks already offer and it needs no arithmetic
    // of its own to be right about.
    let nanos = source.frequency().nanos(now.saturating_sub(since));
    let elapsed = rate.ticks(nanos) / u64::from(divisor(vlapic).ratio());
    // Saturating rather than fallible: a span longer than the timer can count is
    // one whose appointment is due at once, which is what the largest count the
    // register holds becomes after the subtraction above.
    Some(u32::try_from(elapsed).unwrap_or(u32::MAX))
}

/// The fewest ticks a restarted timer is armed at.
///
/// One rather than zero, because the architecture reads a count of zero as a
/// stopped timer rather than as one due immediately — so a deadline that has
/// already passed has to be spelled as the soonest reachable one instead.
const SOONEST: u32 = 1;
