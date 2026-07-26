//! What time it is, how long since, and how to wait.
//!
//! A hypervisor needs time for three different things, and this crate answers
//! all three from one counter: [`Clock::wall_clock`] says what time it is,
//! [`Clock::now`] says how much has passed, and [`Clock::sleep_micros`] waits
//! for a fixed amount of it. Nothing above this crate reads a counter, converts
//! a tick, or knows which timer a given machine turned out to have.
//!
//! # Where a nanosecond comes from
//!
//! Two counters are involved and they do different jobs.
//!
//! The *reference* is a counter whose rate is known without measuring it: the
//! event timer, which reports its own tick period, or failing that the power
//! management timer, whose rate ACPI fixes. It is slow to read — every read is
//! a bus cycle to the chipset — but it is honest about how fast it counts.
//!
//! The *timebase* is what the clock reads afterwards. Where the processor has
//! an invariant timestamp counter, that is the timebase: one instruction, no
//! bus cycle, and a rate that does not move with power state or core. The one
//! thing it will not tell anyone is how fast it counts, so it is measured
//! against the reference once during bring-up, and the reference is handed
//! straight back to the machine.
//!
//! Where the processor has no invariant timestamp counter, its rate is not a
//! constant at all and no measurement of it would stay true, so the event timer
//! keeps the time itself and its mapping is kept for good. That needs a 64-bit
//! main counter: a 32-bit one wraps every few minutes, and something has to
//! read it more often than that for a difference to mean anything. A machine
//! with neither an invariant timestamp counter nor a 64-bit event timer has no
//! timebase pulzar can build, and is refused rather than run with a clock that
//! is quietly wrong.
//!
//! The power management timer is never a timebase, only ever a reference, for
//! the same wrapping reason at 24 or 32 bits.
//!
//! # Wall-clock time
//!
//! The absolute time comes from firmware, once, before the loader jumps: after
//! that there is no real-time clock driver in the picture, only counters. So
//! wall-clock time is that reading plus however much the timebase says has
//! elapsed since the clock was installed. It therefore lags real time by the
//! part of bring-up between the two — milliseconds — and does not drift
//! afterwards beyond the timebase's own error.
//!
//! # Reaching it from anywhere
//!
//! [`Clock::install`] publishes the clock so that [`now`], [`wall_clock`] and
//! [`sleep_micros`] work without a handle, the way serial logging does: the
//! code that will need a delay most — starting the other processors, and later
//! the interrupt paths — is exactly the code with no address space or subsystem
//! handle to hand. Before the clock is installed those functions say so rather
//! than answer from a guess.
//!
//! # Which processor is asking
//!
//! Every function here answers from whichever processor calls it, and two of
//! them measure from a reading the *installing* processor took. An invariant
//! timestamp counter is invariant in rate rather than in origin: the
//! architecture promises that every core counts at the same speed, not that
//! they all started from the same value. Firmware synchronizes them on the
//! machines pulzar runs on, and no processor can confirm that on its own.
//!
//! So the arithmetic is arranged to fail small rather than spectacularly. A
//! processor whose counter started from a different value than the installing
//! one's reads an uptime off by that difference, floored at zero — not one five
//! hundred years long, because a full-width counter reading below what it is
//! compared against is taken as a counter that went backwards rather than one
//! that came round. Delays are unaffected whoever asks: [`Clock::sleep_micros`]
//! measures from a reading it takes itself, so only the rate matters, and the
//! rate is the property the architecture actually guarantees.

#![no_std]

mod counter;
mod hpet;
mod pm_timer;
mod reference;
mod tsc;
mod units;
mod wall;

use core::sync::atomic::{AtomicBool, Ordering};

use acpi::{Acpi, Space};
use log::{info, warn};
use paging::{AddressSpace, PagingError};
use processor::Features;
use spin::Once;
use thiserror::Error;

pub use crate::{
    counter::{Counter, Kind},
    tsc::Calibration,
    units::{Frequency, Instant},
    wall::{Civil, Wall},
};
use crate::{reference::Borrowed, units::NANOS_PER_MICRO};

/// The timebase this image keeps time with.
#[derive(Clone, Copy, Debug)]
pub struct Clock {
    source: Counter,
    origin: u64,
    boot: Option<Wall>,
    calibration: Option<Calibration>,
}

impl Clock {
    /// Establishes the timebase and publishes it.
    ///
    /// `boot` is the wall-clock time firmware reported, which the clock counts
    /// forward from; a machine whose firmware would not say has a monotonic
    /// clock and no wall clock, which is worth having on its own.
    ///
    /// One-shot, like the serial logger and for the same reason: a second call
    /// would take over hardware that the first is already reading, so it is
    /// refused rather than allowed to reprogram a running clock. That holds
    /// even for a call that failed — retrying cannot cure a machine that
    /// has no counter.
    ///
    /// # Errors
    ///
    /// [`ClockError::AlreadyInstalled`] for a second call;
    /// [`ClockError::NoReference`] if the machine describes no counter of known
    /// rate; [`ClockError::NoTimebase`] if nothing here can keep time on it;
    /// and otherwise whichever check the hardware or the measurement
    /// failed. The machine is left as it was found in every case but the
    /// last of those.
    pub fn install(
        space: &mut AddressSpace,
        acpi: &Acpi,
        boot: Option<Wall>,
    ) -> Result<Self, ClockError> {
        // Claimed before any hardware is touched, so a second caller cannot
        // start an event timer that the first is already reading.
        if CLAIMED
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return Err(ClockError::AlreadyInstalled);
        }
        let borrowed = Borrowed::open(space, reference::choose(acpi)?)?;
        let reference = borrowed.counter();

        // With an invariant timestamp counter the reference's whole job is to
        // say how fast that counter runs, so the hardware goes back to the
        // machine as soon as it has.
        if processor::features().contains(Features::INVARIANT_TSC) {
            let measured = tsc::calibrate(&reference);
            borrowed.release(space)?;
            let (frequency, calibration) = measured?;
            return Ok(Self::publish(
                tsc::counter(frequency),
                boot,
                Some(calibration),
            ));
        }

        warn!(
            "clock: the timestamp counter is not invariant, so the {} has to keep the time itself",
            reference.kind()
        );
        if reference.kind() == Kind::Hpet && reference.bits() >= u64::BITS {
            return Ok(Self::publish(borrowed.keep(), boot, None));
        }
        let refused = ClockError::NoTimebase {
            reference: reference.kind(),
            bits: reference.bits(),
        };
        borrowed.release(space)?;
        Err(refused)
    }

    /// How long the clock has been running.
    ///
    /// Measured from the reading the installing processor took, so a processor
    /// whose timestamp counter started from a different value reads an uptime
    /// off by that difference, floored at zero. See the crate documentation.
    #[must_use]
    pub fn now(&self) -> Instant {
        Instant::from_nanos(self.source.nanos_since(self.origin))
    }

    /// What time it is, or `None` if firmware reported no time to count from.
    ///
    /// Firmware's reading plus [`Clock::now`], and so carries that reading's
    /// dependence on which processor is asking.
    #[must_use]
    pub fn wall_clock(&self) -> Option<Wall> {
        self.boot.map(|boot| boot.after(self.now().nanos()))
    }

    /// Waits for `micros` microseconds.
    ///
    /// A spin and not a halt. This exists for the delays hardware bring-up is
    /// specified in — the pauses an INIT-SIPI sequence requires between its
    /// steps, and little else — where the wait is microseconds long and there
    /// is nothing else for the processor to be doing. It is not a
    /// scheduling primitive and must not become one.
    pub fn sleep_micros(&self, micros: u64) {
        let ticks = self
            .source
            .frequency()
            .ticks(micros.saturating_mul(NANOS_PER_MICRO));
        let started = self.source.read();
        while self.source.difference(started, self.source.read()) < ticks {
            core::hint::spin_loop();
        }
    }

    /// The counter the clock reads.
    #[must_use]
    pub const fn source(&self) -> Counter {
        self.source
    }

    /// What the calibration measured, where the timestamp counter had to be
    /// calibrated at all.
    #[must_use]
    pub const fn calibration(&self) -> Option<Calibration> {
        self.calibration
    }

    /// Logs the timebase, the measurement behind it, and the wall clock.
    pub fn describe(&self, who: &str) {
        info!(
            "{who}: clock keeps time from the {} at {} Hz",
            self.source.kind(),
            self.source.frequency().hz(),
        );
        match self.calibration {
            Some(measured) => info!(
                "{who}: clock calibrated it against the {} at {} Hz: {} reference ticks and {} \
                 timestamp ticks over {} ns",
                measured.reference,
                measured.reference_hz,
                measured.reference_ticks,
                measured.tsc_ticks,
                measured.nanos,
            ),
            None => info!("{who}: clock reads that counter directly, so nothing was calibrated"),
        }
        match self.wall_clock() {
            Some(wall) => info!(
                "{who}: clock says it is {wall}, {} ns after the reading firmware gave the loader",
                self.now().nanos(),
            ),
            None => warn!("{who}: clock has no wall time; firmware reported none"),
        }
    }

    /// Starts the clock on `source` and publishes it.
    fn publish(source: Counter, boot: Option<Wall>, calibration: Option<Calibration>) -> Self {
        let clock = Self {
            origin: source.read(),
            source,
            boot,
            calibration,
        };
        // The claim at the top of `install` elected this caller, so the closure
        // is the one that runs and the value that comes back is this clock.
        *CLOCK.call_once(|| clock)
    }
}

/// The installed clock's reading, or `None` before one is installed.
#[must_use]
pub fn now() -> Option<Instant> {
    CLOCK.get().map(Clock::now)
}

/// What time it is, or `None` if there is no clock or firmware reported no
/// time.
#[must_use]
pub fn wall_clock() -> Option<Wall> {
    CLOCK.get().and_then(Clock::wall_clock)
}

/// Waits for `micros` microseconds on the installed clock.
///
/// # Errors
///
/// [`NotInstalled`] if there is no clock yet, in which case nothing was waited
/// for. Saying so beats spinning on a guessed rate: a delay that a caller asked
/// for and did not get is a fault that caller has to be able to see.
pub fn sleep_micros(micros: u64) -> Result<(), NotInstalled> {
    CLOCK
        .get()
        .ok_or(NotInstalled)
        .map(|clock| clock.sleep_micros(micros))
}

/// Why no clock could be established.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum ClockError {
    /// A register could not be mapped, or a mapping could not be released.
    #[error(transparent)]
    Paging(#[from] PagingError),
    /// A clock was already installed on this image.
    #[error("a clock is already installed")]
    AlreadyInstalled,
    /// The machine describes neither an event timer nor a power management
    /// timer, so there is nothing of known rate to measure against.
    #[error("the machine describes no counter of known rate")]
    NoReference,
    /// Firmware put a timer's register somewhere pulzar cannot reach it.
    #[error("a timer register in {space} cannot be reached")]
    UnreachableRegister {
        /// Where firmware said it was.
        space: Space,
    },
    /// An address firmware gave for a timer is not one this processor can form,
    /// or not one an I/O port number fits in.
    #[error("{address:#x} is not a usable timer register address")]
    BadAddress {
        /// The offending value.
        address: u64,
    },
    /// A register is not aligned for the width it has to be read at.
    #[error("a timer register at {address:#x} is not {align}-byte aligned")]
    Misaligned {
        /// Where the register is.
        address: u64,
        /// The alignment its reads need.
        align: u64,
    },
    /// The event timer reports a tick period no timer could have: zero, or
    /// slower than the specification's own limit.
    #[error("the hpet reports a tick period of {femtos} fs, which no timer has")]
    HpetPeriod {
        /// What it reported.
        femtos: u64,
    },
    /// The reference would wrap inside the calibration window, so no difference
    /// across it could be believed.
    #[error("the {reference} is {bits} bits wide, too narrow to calibrate against")]
    ReferenceTooNarrow {
        /// The counter in question.
        reference: Kind,
        /// How wide it turned out to be.
        bits: u32,
    },
    /// The measurement produced a rate no counter could have, which means one
    /// of the two counters did not advance at all.
    #[error("the timestamp counter calibration measured nothing")]
    Calibration,
    /// The processor has no invariant timestamp counter and the machine's own
    /// counter cannot keep time either, so nothing here can.
    #[error("the timestamp counter is not invariant and the {reference} is only {bits} bits wide")]
    NoTimebase {
        /// The counter that would have had to keep the time.
        reference: Kind,
        /// How wide it turned out to be.
        bits: u32,
    },
}

/// Something asked the clock for time before there was one.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[error("no clock is installed")]
pub struct NotInstalled;

/// The clock every free function in this crate answers from.
static CLOCK: Once<Clock> = Once::new();

/// Claimed by the first [`Clock::install`], so a second cannot take over
/// hardware the first is already reading.
static CLAIMED: AtomicBool = AtomicBool::new(false);
