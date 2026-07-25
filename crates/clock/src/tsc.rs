//! Measuring the timestamp counter against a counter of known rate.
//!
//! The timestamp counter is the only counter worth reading often — one
//! instruction, no bus cycle, no lock — and the only one whose rate nothing
//! reliably reports. So it is measured: run both counters over the same span,
//! and the ratio of what they counted is the ratio of their rates.
//!
//! # The window
//!
//! Ten milliseconds, which is a compromise with one term on each side. Longer
//! costs boot time and buys accuracy the reference's own crystal does not have.
//! Shorter runs into the reference's resolution: the power management timer
//! ticks 3.58 million times a second, so ten milliseconds is some 35 800 ticks
//! and one tick of quantization is 28 parts per million — an order of magnitude
//! below the tolerance those crystals are sold to.
//!
//! # What can happen during it and does not matter
//!
//! An interrupt, or a system management interrupt firmware takes without
//! asking: both counters keep counting through either, and the result is
//! computed from the ticks actually observed rather than from the window that
//! was intended, so a span that came out longer than asked for is simply a
//! longer measurement. What would matter is a wrap of the reference, and that
//! is ruled out before the measurement rather than detected after it.

use core::hint::spin_loop;

use crate::{
    ClockError, Frequency, Kind,
    counter::{Counter, Register},
};

/// Nanoseconds the measurement runs for.
const WINDOW_NANOS: u64 = 10_000_000;

/// What a calibration observed.
///
/// Kept so the log can carry the measurement and not just its conclusion: a
/// frequency that looks wrong is a great deal easier to argue with when the
/// ticks it was divided out of are on the record beside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Calibration {
    /// The counter the timestamp counter was measured against.
    pub reference: Kind,
    /// That counter's own rate, in hertz.
    pub reference_hz: u64,
    /// Ticks of it the measurement covered.
    pub reference_ticks: u64,
    /// Ticks of the timestamp counter over the same span.
    pub tsc_ticks: u64,
    /// What that span worked out to in nanoseconds.
    pub nanos: u64,
}

/// Measures the timestamp counter against `reference`.
///
/// # Errors
///
/// [`ClockError::ReferenceTooNarrow`] if the reference would wrap inside the
/// window, which makes a difference across it ambiguous, or
/// [`ClockError::Calibration`] if the measurement produced a rate no counter
/// could have — a reference that did not advance, or a timestamp counter that
/// did not.
pub(crate) fn calibrate(reference: &Counter) -> Result<(Frequency, Calibration), ClockError> {
    let window = reference.frequency().ticks(WINDOW_NANOS);
    if !reference.can_span(window) {
        return Err(ClockError::ReferenceTooNarrow {
            reference: reference.kind(),
            bits: reference.bits(),
        });
    }

    // The two counters are read as a pair at both ends, in the same order, so
    // whatever it costs to read the reference falls inside both spans and out of
    // the ratio.
    let first_reference = reference.read();
    let first_tsc = timestamp();
    let (reference_ticks, tsc_ticks) = loop {
        let now_reference = reference.read();
        let now_tsc = timestamp();
        let elapsed = reference.difference(first_reference, now_reference);
        if elapsed >= window {
            break (elapsed, now_tsc.wrapping_sub(first_tsc));
        }
        spin_loop();
    };

    let nanos = reference.frequency().nanos(reference_ticks);
    let frequency = Frequency::from_measurement(tsc_ticks, nanos).ok_or(ClockError::Calibration)?;
    Ok((
        frequency,
        Calibration {
            reference: reference.kind(),
            reference_hz: reference.frequency().hz(),
            reference_ticks,
            tsc_ticks,
            nanos,
        },
    ))
}

/// The calibrated timestamp counter, as a counter in its own right.
pub(crate) fn counter(frequency: Frequency) -> Counter {
    // SAFETY: reading the timestamp counter needs nothing mapped and has no
    // effect on the machine, so both conditions `Counter::new` asks about hold
    // by construction.
    unsafe { Counter::new(Kind::Tsc, Register::Timestamp, frequency, u64::BITS) }
}

/// The timestamp counter, with the reads before it already done.
///
/// `rdtsc` is not ordered against its neighbours, so a bare read may be
/// executed before the reference read it is meant to pair with — which would
/// put the error into the measurement rather than into the noise. An `lfence`
/// costs a few tens of cycles and settles the question; against ten
/// milliseconds it is nothing.
fn timestamp() -> u64 {
    // SAFETY: `lfence` is implemented by every processor that can run 64-bit
    // code, takes no operands, touches no memory, and cannot fault.
    unsafe { core::arch::x86_64::_mm_lfence() };
    processor::timestamp()
}
