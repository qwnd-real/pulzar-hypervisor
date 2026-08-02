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
    /// The mode an entry the controller already holds is in, or `None` where
    /// the field holds the encoding the architecture reserves.
    pub(crate) const fn of(entry: u32) -> Option<Self> {
        match (entry >> MODE_SHIFT) & MODE_MASK {
            0b00 => Some(Self::OneShot),
            0b01 => Some(Self::Periodic),
            0b10 => Some(Self::Deadline),
            _ => None,
        }
    }

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

/// The timer's mode field, once shifted down.
const MODE_MASK: u32 = 0b11;

/// The model-specific register holding the timestamp counter deadline.
pub(crate) const IA32_TSC_DEADLINE: u32 = 0x6E0;

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
}

/// The timer of whichever processor is holding this.
///
/// Carries nothing for the same reason [`LocalApic`] carries nothing: every
/// register it drives is reached the same way on every processor and answers
/// about the processor doing the reaching, so a handle that named one would be
/// a handle that could be wrong. Like that one, it is made only from a
/// controller that is up, and only inside this crate.
#[derive(Clone, Copy, Debug)]
pub struct Timer(());

impl Timer {
    /// This processor's timer.
    pub(crate) const fn new(_: LocalApic) -> Self {
        Self(())
    }

    /// Stops the timer and stops it delivering.
    ///
    /// Both, in that order, because either alone leaves something behind: a
    /// masked entry over a running count still reloads, and a zero count in
    /// periodic mode is how the architecture spells "stopped" but leaves the
    /// entry armed for whoever writes a count next.
    ///
    /// A deadline is a third thing to put down. Changing the mode disarms the
    /// timer, so a masked entry is already enough to stop the interrupt — but
    /// the deadline itself sits in a register of its own and would still be
    /// there, describing a moment in the past, for whoever reads it next.
    ///
    /// # Errors
    ///
    /// [`ApicError::NotInstalled`] if this processor's controller is not up.
    pub fn disarm(self) -> Result<(), ApicError> {
        let access = crate::register::access()?;
        let was = Mode::of(access.read(Register::LVT_TIMER));
        // SAFETY: zero is the architectural way to stop a counting timer, and a
        // masked entry with a valid vector delivers nothing. Neither can produce
        // an interrupt, which is the whole point of doing them.
        unsafe {
            access.write(Register::TIMER_INITIAL_COUNT, 0);
            access.write(Register::LVT_TIMER, Entry::masked().bits());
        }
        if matches!(was, Some(Mode::Deadline)) {
            // SAFETY: the entry the timer was holding says it was counting
            // against a deadline, which only a processor that implements the
            // register can be doing, and zero is what the architecture defines
            // as no deadline at all.
            unsafe { Msr::new(IA32_TSC_DEADLINE).write(0) };
        }
        Ok(())
    }

    /// Says what the timer delivers, in which mode, at which rate — and starts
    /// nothing.
    ///
    /// Configuration and starting are separate operations because the
    /// architecture makes them separate, and conflating them is not a shortcut
    /// but a different timer. Writing the initial count is what starts a
    /// counting timer, and writing the deadline is what arms a deadline one;
    /// the entry and the divide say only what happens when whichever of
    /// those is reached. Software that changes a vector, masks a source, or
    /// selects a mode has not asked for the timer to restart, and a caller
    /// that restarted it would move the phase of a periodic tick every time
    /// the guest touched an unrelated field.
    ///
    /// `delivery` says both whether the entry delivers and what it delivers on.
    /// `None` masks it, which suppresses the interrupt and nothing else: the
    /// count goes on running and software can still read it, which is exactly
    /// how software measures what the timer's rate is. A masked entry still
    /// carries a vector, because a masked entry naming vector zero is a
    /// configuration some processors report as an error, so the lowest vector
    /// the platform may assign stands in and nothing is ever delivered on it.
    ///
    /// # Errors
    ///
    /// [`ApicError::IllegalVector`] for a vector no controller may deliver,
    /// [`ApicError::NoTscDeadline`] if [`Mode::Deadline`] was asked for on a
    /// processor that does not implement it, or [`ApicError::NotInstalled`] if
    /// this processor's controller is not up.
    pub fn configure(
        self,
        delivery: Option<Vector>,
        mode: Mode,
        divisor: Divisor,
    ) -> Result<(), ApicError> {
        if let Some(vector) = delivery
            && !crate::deliverable(vector)
        {
            return Err(ApicError::IllegalVector { vector });
        }
        if matches!(mode, Mode::Deadline) && !processor::features().contains(Features::TSC_DEADLINE)
        {
            return Err(ApicError::NoTscDeadline);
        }
        let entry = match delivery {
            Some(vector) => Entry::new(Delivery::Fixed(vector)),
            None => Entry::masked(),
        };
        let access = crate::register::access()?;
        // SAFETY: the divisor's encoding comes from the architecture's own
        // table, and the entry either names a vector that has a gate like every
        // other or is masked and delivers nothing. Neither write starts
        // anything: the count and the deadline are left exactly as they were.
        unsafe {
            access.write(Register::TIMER_DIVIDE, divisor.bits());
            access.write(Register::LVT_TIMER, entry.bits() | mode.bits());
        }
        Ok(())
    }

    /// Starts a counting timer from `count`, against whatever
    /// [`Timer::configure`] last said.
    ///
    /// This is the write the architecture defines as starting the timer, which
    /// is why it is the only thing here that does. A count of zero stops it,
    /// which is the architecture's own spelling and not an error.
    ///
    /// # Errors
    ///
    /// [`ApicError::NotInstalled`] if this processor's controller is not up.
    pub fn reload(self, count: u32) -> Result<(), ApicError> {
        let access = crate::register::access()?;
        // SAFETY: the count is a plain 32-bit value the timer counts down, and
        // zero is the architectural way to say stopped. What it delivers when it
        // gets there was settled by the entry, which this does not touch.
        unsafe { access.write(Register::TIMER_INITIAL_COUNT, count) };
        Ok(())
    }

    /// Arms the deadline the timer fires at, against whatever
    /// [`Timer::configure`] last said.
    ///
    /// Zero is the architecture's way of spelling no deadline at all, and
    /// disarms without disturbing the entry. Any other value is a moment; one
    /// already past fires at once, which is defined and is the caller's to
    /// intend.
    ///
    /// # Errors
    ///
    /// [`ApicError::NoTscDeadline`] if the processor does not implement the
    /// mode.
    pub fn set_deadline(self, deadline: u64) -> Result<(), ApicError> {
        if !processor::features().contains(Features::TSC_DEADLINE) {
            return Err(ApicError::NoTscDeadline);
        }
        // The entry has to be in place before the deadline can be reached, and
        // nothing but the ordering of the two guarantees it: this write is what
        // starts the deadline running.
        core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
        // SAFETY: any value is a valid deadline, and the processor was just
        // established to implement the register.
        unsafe { Msr::new(IA32_TSC_DEADLINE).write(deadline) };
        Ok(())
    }

    /// The deadline the timer is counting towards, or zero if it is counting
    /// towards none.
    ///
    /// Hardware clears the register when the deadline fires, so this is also
    /// how to ask whether a deadline that was armed has since expired —
    /// which is a question nothing else can answer, since the expiry
    /// arrives as an ordinary interrupt carrying nothing about where it
    /// came from.
    ///
    /// # Errors
    ///
    /// [`ApicError::NoTscDeadline`] if the processor does not implement the
    /// mode.
    pub fn deadline(self) -> Result<u64, ApicError> {
        if !processor::features().contains(Features::TSC_DEADLINE) {
            return Err(ApicError::NoTscDeadline);
        }
        // SAFETY: the processor was just established to implement the register,
        // and reading it has no effect on what the timer is doing.
        Ok(unsafe { Msr::new(IA32_TSC_DEADLINE).read() })
    }

    /// Which mode the timer is currently in, or `None` for the encoding the
    /// architecture reserves.
    ///
    /// Read back rather than remembered because the answer decides whether a
    /// deadline still means anything: changing the mode disarms the timer, so a
    /// caller reconfiguring it has to know whether it is leaving deadline mode
    /// in order to put the deadline register down as well.
    ///
    /// # Errors
    ///
    /// [`ApicError::NotInstalled`] if this processor's controller is not up.
    pub fn mode(self) -> Result<Option<Mode>, ApicError> {
        crate::register::access().map(|access| Mode::of(access.read(Register::LVT_TIMER)))
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
    /// The span is bracketed by two timebase readings with the count read
    /// between them, so it covers one register read more than it counted ticks
    /// for. Over ten milliseconds that is around a hundred-thousandth of the
    /// answer, which is orders of magnitude inside the tolerance of the crystal
    /// being measured.
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
