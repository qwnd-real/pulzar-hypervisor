//! The local APIC's timer: the one timer each processor has entirely to itself.
//!
//! Three modes, and the third is not like the other two. One-shot and periodic
//! both count a register down to zero at a divided bus clock; the difference is
//! only whether the count reloads. Deadline mode counts nothing — the timer
//! fires when the timestamp counter passes a value written to a model-specific
//! register — which makes it the only one of the three that is not quantized to
//! a divided tick, and the only one whose deadline is absolute rather than
//! relative.
//!
//! # Why the rate has to be measured
//!
//! Because nothing reports it. The counting modes tick at the bus or core
//! crystal clock divided by the divisor, and no register says what that clock
//! is. So it is measured once against the timebase, the same way the timestamp
//! counter's rate is, and everything afterwards converts through the
//! measurement. Deadline mode needs no such thing: it is denominated in
//! timestamp counter ticks, whose rate the clock subsystem already knows.
//!
//! # Arming, and who handles the arrival
//!
//! Arming programs the local vector table entry and the count, and does not
//! register a handler. Which vector the timer uses and what happens when it
//! fires are decisions about the hypervisor's interrupt structure, not about
//! the timer, and they belong to whoever is arming it.

use clock::Frequency;
use descriptors::Vector;
use processor::Features;
use x86_64::registers::model_specific::Msr;

use crate::{
    ApicError, LocalApic,
    lvt::{Delivery, Entry},
    register::Register,
};

/// How the timer decides when to fire.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Count down once from the initial count and stop.
    OneShot,
    /// Count down from the initial count and reload, firing every time.
    Periodic,
    /// Fire when the timestamp counter passes a written deadline.
    ///
    /// Needs [`Features::TSC_DEADLINE`], and the deadline is set with
    /// [`Timer::arm_deadline`] rather than by a count.
    Deadline,
}

impl Mode {
    /// The mode field's encoding in the local vector table entry.
    const fn bits(self) -> u32 {
        match self {
            Self::OneShot => 0b00 << MODE_SHIFT,
            Self::Periodic => 0b01 << MODE_SHIFT,
            Self::Deadline => 0b10 << MODE_SHIFT,
        }
    }
}

/// Bits the timer's mode field is shifted by in its local vector table entry.
const MODE_SHIFT: u32 = 17;

/// The model-specific register holding the timestamp counter deadline.
const IA32_TSC_DEADLINE: u32 = 0x6E0;

/// How far the input clock is divided before the timer counts it.
///
/// The encoding is the awkward part and the reason this is an enum rather than
/// a number: the three-bit field is split, with its middle bit skipped, so that
/// the value written is not the ratio and is not even contiguous with it. It is
/// written out once here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Divisor {
    /// Count every input tick.
    By1,
    /// Count one input tick in two.
    By2,
    /// Count one input tick in four.
    By4,
    /// Count one input tick in eight.
    By8,
    /// Count one input tick in sixteen.
    By16,
    /// Count one input tick in thirty-two.
    By32,
    /// Count one input tick in sixty-four.
    By64,
    /// Count one input tick in a hundred and twenty-eight.
    By128,
}

impl Divisor {
    /// Every divisor, so a caller can search them without repeating the list.
    pub const ALL: [Self; 8] = [
        Self::By1,
        Self::By2,
        Self::By4,
        Self::By8,
        Self::By16,
        Self::By32,
        Self::By64,
        Self::By128,
    ];

    /// The divide configuration register's encoding.
    ///
    /// Bit 2 of the field is not used, so the three meaningful bits are 0, 1
    /// and 3 — which is why the values below look nothing like the ratios.
    const fn bits(self) -> u32 {
        match self {
            Self::By1 => 0b1011,
            Self::By2 => 0b0000,
            Self::By4 => 0b0001,
            Self::By8 => 0b0010,
            Self::By16 => 0b0011,
            Self::By32 => 0b1000,
            Self::By64 => 0b1001,
            Self::By128 => 0b1010,
        }
    }

    /// How many input ticks make one the timer counts.
    #[must_use]
    pub const fn ratio(self) -> u32 {
        match self {
            Self::By1 => 1,
            Self::By2 => 2,
            Self::By4 => 4,
            Self::By8 => 8,
            Self::By16 => 16,
            Self::By32 => 32,
            Self::By64 => 64,
            Self::By128 => 128,
        }
    }
}

/// The timer of whichever processor is holding this.
///
/// Zero-sized for the same reason [`LocalApic`] is: every register it drives is
/// reached the same way on every processor and answers about the processor
/// doing the reaching, so there is nothing for a handle to carry and a handle
/// that named a processor would be one that could be wrong.
#[derive(Clone, Copy, Debug)]
pub struct Timer;

impl Timer {
    /// This processor's timer.
    pub(crate) const fn new(_: LocalApic) -> Self {
        Self
    }

    /// Stops the timer and stops it delivering.
    ///
    /// Both, in that order, because either alone leaves something behind: a
    /// masked entry over a running count still reloads, and a zero count in
    /// periodic mode is how the architecture spells "stopped" but leaves the
    /// entry armed for whoever writes a count next.
    ///
    /// # Errors
    ///
    /// [`ApicError::NotInstalled`] if this processor's controller is not up.
    pub fn disarm(self) -> Result<(), ApicError> {
        let access = crate::register::access()?;
        // SAFETY: zero is the architectural way to stop a counting timer, and a
        // masked entry with a valid vector delivers nothing. Neither can produce
        // an interrupt, which is the whole point of doing them.
        unsafe {
            access.write(Register::TIMER_INITIAL_COUNT, 0);
            access.write(Register::LVT_TIMER, Entry::masked().bits());
        }
        Ok(())
    }

    /// Arms the timer to fire on `vector` after `count` of its own ticks, once
    /// or over and over.
    ///
    /// The entry is written before the count, because writing the count is what
    /// starts it: the other order leaves a window in which the timer is running
    /// towards an entry that still says whatever it said before.
    ///
    /// # Errors
    ///
    /// [`ApicError::ZeroCount`] for a count of zero, which the architecture
    /// reads as "stopped" rather than as "immediately" and which would leave a
    /// caller waiting for an interrupt that never comes;
    /// [`ApicError::WrongTimerMode`] if [`Mode::Deadline`] was asked for, which
    /// takes a deadline rather than a count; or [`ApicError::NotInstalled`] if
    /// this processor's controller is not up.
    pub fn arm(
        self,
        vector: Vector,
        mode: Mode,
        divisor: Divisor,
        count: u32,
    ) -> Result<(), ApicError> {
        if matches!(mode, Mode::Deadline) {
            return Err(ApicError::WrongTimerMode);
        }
        if count == 0 {
            return Err(ApicError::ZeroCount);
        }
        let access = crate::register::access()?;
        // SAFETY: the divisor's encoding comes from the architecture's own
        // table, the entry names a vector that has a gate like every other, and
        // the count is a plain 32-bit value the timer counts down. The order is
        // what keeps the timer from running against a stale entry.
        unsafe {
            access.write(Register::TIMER_DIVIDE, divisor.bits());
            access.write(
                Register::LVT_TIMER,
                Entry::new(Delivery::Fixed(vector)).bits() | mode.bits(),
            );
            access.write(Register::TIMER_INITIAL_COUNT, count);
        }
        Ok(())
    }

    /// Arms the timer to fire on `vector` when the timestamp counter passes
    /// `deadline`.
    ///
    /// # Errors
    ///
    /// [`ApicError::NoTscDeadline`] if the processor does not implement the
    /// mode, or [`ApicError::NotInstalled`] if this processor's controller is
    /// not up.
    pub fn arm_deadline(self, vector: Vector, deadline: u64) -> Result<(), ApicError> {
        if !processor::features().contains(Features::TSC_DEADLINE) {
            return Err(ApicError::NoTscDeadline);
        }
        let access = crate::register::access()?;
        // SAFETY: the entry names a vector with a gate, in a mode this processor
        // reports implementing.
        unsafe {
            access.write(
                Register::LVT_TIMER,
                Entry::new(Delivery::Fixed(vector)).bits() | Mode::Deadline.bits(),
            );
        }
        // Written after the entry, and it is what arms the timer: the
        // architecture starts the deadline running on this write. A fence first,
        // because the entry has to be in place before the deadline can be
        // reached, and nothing but the ordering of these two guarantees it.
        core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
        // SAFETY: any value is a valid deadline; one already in the past fires
        // at once, which is a defined outcome and the caller's to intend.
        unsafe { Msr::new(IA32_TSC_DEADLINE).write(deadline) };
        Ok(())
    }

    /// What the timer has left to count.
    ///
    /// Zero in deadline mode, where nothing is counting, and zero after a
    /// one-shot has fired.
    ///
    /// # Errors
    ///
    /// [`ApicError::NotInstalled`] if this processor's controller is not up.
    pub fn remaining(self) -> Result<u32, ApicError> {
        crate::register::access().map(|access| access.read(Register::TIMER_CURRENT_COUNT))
    }

    /// Measures how fast the timer counts at `divisor`.
    ///
    /// A one-shot from the top of the count is started, the timebase is used to
    /// wait a fixed span, and what the timer got through in that span is the
    /// rate. Left disarmed afterwards, because a measurement is not an arming
    /// and leaving a timer running that a caller did not ask for would deliver
    /// an interrupt nothing expects.
    ///
    /// The entry is masked throughout, so the count reaching zero during a
    /// measurement — which it should not, but a machine with a very fast bus
    /// clock and a slow caller could — delivers nothing.
    ///
    /// # Errors
    ///
    /// [`ApicError::Clock`] if no timebase is installed to measure against,
    /// [`ApicError::Calibration`] if the timer did not move or moved so far it
    /// wrapped, or [`ApicError::NotInstalled`] if this processor's controller
    /// is not up.
    pub fn calibrate(self, divisor: Divisor) -> Result<Frequency, ApicError> {
        let access = crate::register::access()?;
        // SAFETY: a masked entry delivers nothing whatever the count does, and
        // the count is a plain value. Nothing is armed by this.
        unsafe {
            access.write(Register::TIMER_DIVIDE, divisor.bits());
            access.write(Register::LVT_TIMER, Entry::masked().bits());
            access.write(Register::TIMER_INITIAL_COUNT, u32::MAX);
        }
        let started = clock::now().ok_or(ApicError::Clock)?;
        clock::sleep_micros(CALIBRATION_MICROS).map_err(|_| ApicError::Clock)?;
        let remaining = access.read(Register::TIMER_CURRENT_COUNT);
        let elapsed = (clock::now().ok_or(ApicError::Clock)? - started).as_nanos();
        self.disarm()?;

        // Reaching zero means the count wrapped, so what is left says nothing
        // about how far the timer got; not moving at all means the timer is not
        // running. Neither is a measurement.
        let ticks = u64::from(u32::MAX - remaining);
        if ticks == 0 || remaining == 0 {
            return Err(ApicError::Calibration);
        }
        let nanos = u64::try_from(elapsed).map_err(|_| ApicError::Calibration)?;
        Frequency::from_measurement(ticks, nanos).ok_or(ApicError::Calibration)
    }
}

/// How long a rate measurement watches the timer for.
///
/// The same order as the timestamp counter's own calibration window: long
/// enough that the cost of reading the timebase at each end is lost in it,
/// short enough that a timer counting an undivided gigahertz bus clock does not
/// wrap its 32-bit count.
const CALIBRATION_MICROS: u64 = 10_000;
