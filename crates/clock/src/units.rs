//! Ticks, nanoseconds, and the one place the two are converted.
//!
//! Every counter counts at its own rate and nothing above this module wants to
//! think in that rate, so each counter's rate becomes a [`Frequency`] and every
//! conversion goes through one. Having exactly one implementation of the
//! arithmetic is the point: a hypervisor that converted ticks in three places
//! would round differently in three places.
//!
//! The arithmetic is done in 128 bits and narrowed once at the end, which keeps
//! every conversion exact for any counter and any uptime a machine can reach —
//! a 64-bit tick count at 5 GHz is 116 years, and multiplying it by a billion
//! needs 96 bits. The narrowing saturates rather than wrapping, so an input
//! that could not have come from real hardware produces an obviously wrong
//! answer instead of a plausible one.

use core::{num::NonZeroU64, ops::Sub, time::Duration};

/// Nanoseconds in a second.
pub(crate) const NANOS_PER_SECOND: u64 = 1_000_000_000;

/// Nanoseconds in a microsecond.
pub(crate) const NANOS_PER_MICRO: u64 = 1_000;

/// Femtoseconds in a second, which is the unit the HPET reports its tick period
/// in.
const FEMTOS_PER_SECOND: u64 = 1_000_000_000_000_000;

/// The rate a counter ticks at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Frequency(NonZeroU64);

impl Frequency {
    /// The rate of a counter that ticks this many times a second.
    #[must_use]
    pub const fn new(hz: NonZeroU64) -> Self {
        Self(hz)
    }

    /// The same from a plain count, or `None` for a counter that does not tick
    /// at all — a rate nothing could convert with, and hardware describing
    /// itself impossibly.
    #[must_use]
    pub const fn from_hz(hz: u64) -> Option<Self> {
        match NonZeroU64::new(hz) {
            Some(hz) => Some(Self(hz)),
            None => None,
        }
    }

    /// The rate of a counter whose tick period is `femtos` femtoseconds, which
    /// is how the HPET describes itself.
    ///
    /// Truncated to whole hertz, which loses at most one part in the counter's
    /// own frequency: 0.07 parts per million for a 14 MHz counter, three orders
    /// of magnitude below the tolerance of the crystal driving it.
    #[must_use]
    pub const fn from_period_femtos(femtos: u64) -> Option<Self> {
        match femtos {
            0 => None,
            femtos => Self::from_hz(FEMTOS_PER_SECOND / femtos),
        }
    }

    /// The rate of a counter observed to advance `ticks` in `nanos`
    /// nanoseconds, which is what a calibration measures.
    #[must_use]
    pub fn from_measurement(ticks: u64, nanos: u64) -> Option<Self> {
        match nanos {
            0 => None,
            nanos => Self::from_hz(narrow(
                u128::from(ticks) * u128::from(NANOS_PER_SECOND) / u128::from(nanos),
            )),
        }
    }

    /// Ticks a second.
    #[must_use]
    pub const fn hz(self) -> u64 {
        self.0.get()
    }

    /// Nanoseconds that `ticks` of this counter take, rounded down.
    #[must_use]
    pub fn nanos(self, ticks: u64) -> u64 {
        narrow(u128::from(ticks) * u128::from(NANOS_PER_SECOND) / u128::from(self.hz()))
    }

    /// Ticks of this counter that `nanos` nanoseconds take, rounded down.
    #[must_use]
    pub fn ticks(self, nanos: u64) -> u64 {
        narrow(u128::from(nanos) * u128::from(self.hz()) / u128::from(NANOS_PER_SECOND))
    }
}

// Converted at compile time, so that a change to the arithmetic cannot quietly
// move a rate a machine actually reports.
const _: () = {
    // The event timer every PC chipset derives from the original colour burst
    // clock. Its nominal rate is 14 318 180 Hz and the period it reports
    // divides out one hertz short of that: the truncation
    // `from_period_femtos` describes, at 0.07 parts per million.
    assert!(
        Frequency::from_period_femtos(69_841_279).unwrap().hz() == 14_318_179,
        "a 69.841279 ns tick period is a 14.318 MHz counter"
    );
    assert!(
        Frequency::from_period_femtos(FEMTOS_PER_SECOND)
            .unwrap()
            .hz()
            == 1,
        "a tick period of one second is a counter that ticks once a second"
    );
    assert!(
        Frequency::from_period_femtos(0).is_none(),
        "a counter with no tick period is not a counter"
    );
    assert!(
        Frequency::from_hz(0).is_none(),
        "a counter that does not tick is not a counter"
    );
};

/// A point on the monotonic clock, as nanoseconds since it was installed.
///
/// It says nothing about what time it is — that is [`crate::Wall`] — only how
/// much time has passed, which is the question a hypervisor asks far more often
/// and the only one a counter can answer on its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Instant(u64);

impl Instant {
    /// The instant `nanos` nanoseconds after the clock was installed.
    pub(crate) const fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }

    /// Nanoseconds since the clock was installed.
    #[must_use]
    pub const fn nanos(self) -> u64 {
        self.0
    }
}

impl Sub for Instant {
    type Output = Duration;

    /// The span from `earlier` to this instant, or zero if the two are the
    /// wrong way round.
    ///
    /// Saturating rather than wrapping because the reversed order is a mistake
    /// in the caller, and answering it with an eighteen-year span would hide
    /// that mistake behind an entirely believable number.
    fn sub(self, earlier: Self) -> Duration {
        Duration::from_nanos(self.0.saturating_sub(earlier.0))
    }
}

/// A 128-bit intermediate as a `u64`, saturating at the top of the range.
fn narrow(value: u128) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}
