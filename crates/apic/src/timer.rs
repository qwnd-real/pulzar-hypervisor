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
//! Which mode the timer is in decides which of those two ways of arming it
//! means anything: the architecture ignores a count written in deadline mode
//! and ignores a deadline written outside it. So both are checked against the
//! mode the entry currently selects rather than written and hoped for, because
//! a write hardware ignores is an appointment nobody is keeping.
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
//!
//! # What none of this can undo
//!
//! Stopping the timer stops it *generating* interrupts. One it has already
//! generated is somewhere else by then — in the controller's request register,
//! or accepted and in service — and no write here retracts it. Anything that
//! quiets a timer in order to change what its vector means has to expect one
//! more arrival on the old one and deal with it.

use clock::Frequency;
use descriptors::Vector;
use processor::Features;
use x86_64::registers::model_specific::Msr;

use crate::{
    ApicError, LocalApic,
    lvt::{self, Delivery, Entry},
    register::{Access, Register},
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
    /// [`Timer::set_deadline`] rather than by a count.
    Deadline,
}

impl Mode {
    /// The mode an entry the controller already holds is in, or `None` where
    /// the field holds the encoding the architecture reserves.
    pub(crate) const fn of(entry: u32) -> Option<Self> {
        match (entry >> lvt::MODE_SHIFT) & lvt::MODE_FIELD {
            0b00 => Some(Self::OneShot),
            0b01 => Some(Self::Periodic),
            0b10 => Some(Self::Deadline),
            _ => None,
        }
    }

    /// The mode field's encoding in the local vector table entry.
    const fn bits(self) -> u32 {
        match self {
            Self::OneShot => 0b00 << lvt::MODE_SHIFT,
            Self::Periodic => 0b01 << lvt::MODE_SHIFT,
            Self::Deadline => 0b10 << lvt::MODE_SHIFT,
        }
    }

    /// Whether this mode arms by counting rather than by a deadline.
    const fn counts(self) -> bool {
        matches!(self, Self::OneShot | Self::Periodic)
    }
}

/// The model-specific register holding the timestamp counter deadline.
///
/// Public because it is not only this crate's to write: a hypervisor
/// virtualizing the timer has to intercept the register the guest arms it
/// through, and naming the same index twice is how the two would drift apart.
pub const IA32_TSC_DEADLINE: u32 = 0x6E0;

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
    /// The numeric ratio between input ticks and timer ticks.
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
/// Carries that processor's controller, for the reason [`LocalApic`] is what it
/// is: every register the timer drives answers about the processor doing the
/// reaching, and how to reach it is that processor's own and is worked out at
/// each access rather than remembered.
#[derive(Clone, Copy, Debug)]
pub struct Timer(LocalApic);

impl Timer {
    /// This processor's timer.
    pub(crate) const fn new(local: LocalApic) -> Self {
        Self(local)
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
    /// Stops *future* interrupts, and only those. One the timer has already
    /// generated has left the timer, and this cannot reach it; a caller
    /// changing what the old vector means has to expect it.
    pub fn disarm(self) {
        let access = self.0.access();
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
    /// Two registers are written, and nothing makes the pair indivisible. A
    /// timer already counting spends the gap between them running at the new
    /// rate under the old entry, so an expiry that lands there is delivered on
    /// the old vector. The timer is one processor's and this crate takes it to
    /// be that processor's alone; a caller that shares it has to hold whatever
    /// it shares it with off itself.
    ///
    /// The one thing this does put down is a deadline the mode change leaves
    /// behind. Crossing into or out of deadline mode disarms the timer, but the
    /// deadline sits in a register of its own and would still be there: a
    /// moment already past waiting to fire the instant the mode is
    /// selected, or an appointment the mode no longer keeps. It belongs
    /// here because this is the only place both sides of the crossing are
    /// known — [`Timer::set_deadline`] answers for the mode the timer is
    /// in, and so cannot write the register from either side of a change.
    ///
    /// # Errors
    ///
    /// [`ApicError::IllegalVector`] for a vector no controller may deliver, or
    /// [`ApicError::NoTscDeadline`] if [`Mode::Deadline`] was asked for on a
    /// processor that does not implement it.
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
        if matches!(mode, Mode::Deadline) && !has_deadline() {
            return Err(ApicError::NoTscDeadline);
        }
        let entry = match delivery {
            Some(vector) => Entry::new(Delivery::Fixed(vector)),
            None => Entry::masked(),
        };
        let access = self.0.access();
        let crossing =
            matches!(mode, Mode::Deadline) != matches!(self.mode(), Some(Mode::Deadline));
        if crossing {
            // Before the entry rather than after, so that a stale deadline is
            // gone by the time the mode that would honour it is selected.
            //
            // SAFETY: zero is what the architecture defines as no deadline at
            // all, and the processor implements the register either way round:
            // going in, the check above established the feature; coming out, the
            // timer was in a mode only a processor that has it can be in.
            unsafe { Msr::new(IA32_TSC_DEADLINE).write(0) };
        }
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
    /// [`ApicError::WrongTimerMode`] if the timer is not counting — in deadline
    /// mode the architecture ignores this write, and reporting an appointment
    /// as made when hardware discarded it is worse than refusing to make
    /// it.
    pub fn reload(self, count: u32) -> Result<(), ApicError> {
        let access = self.0.access();
        if !counting(access) {
            return Err(ApicError::WrongTimerMode);
        }
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
    /// mode, or [`ApicError::WrongTimerMode`] if the timer is counting instead
    /// — where the architecture ignores this write.
    pub fn set_deadline(self, deadline: u64) -> Result<(), ApicError> {
        if !has_deadline() {
            return Err(ApicError::NoTscDeadline);
        }
        if self.mode() != Some(Mode::Deadline) {
            return Err(ApicError::WrongTimerMode);
        }
        // The entry has to be in place before the deadline can be reached, and
        // nothing but the ordering of the two guarantees it: this write is what
        // starts the deadline running. A full barrier rather than a release
        // fence, because what has to land first is an uncached register write
        // rather than an ordinary store.
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
        if !has_deadline() {
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
    #[must_use]
    pub fn mode(self) -> Option<Mode> {
        Mode::of(self.0.access().read(Register::LVT_TIMER))
    }

    /// What the timer has left to count.
    ///
    /// Zero in deadline mode, where nothing is counting, and zero after a
    /// one-shot has fired.
    #[must_use]
    pub fn remaining(self) -> u32 {
        self.0.access().read(Register::TIMER_CURRENT_COUNT)
    }

    /// What the timer was last started from, and what a periodic one reloads at
    /// every zero crossing.
    ///
    /// The register that decides how short a periodic period actually is, which
    /// is why it is readable here rather than remembered by whoever wrote it: a
    /// count written while the entry was masked, or in another mode, is still
    /// the count hardware will reload when the entry is unmasked in periodic
    /// mode, and no software copy of it can be relied on to say so.
    #[must_use]
    pub fn initial(self) -> u32 {
        self.0.access().read(Register::TIMER_INITIAL_COUNT)
    }

    /// Measures how fast the timer counts at `divisor`.
    ///
    /// A one-shot from the top of the count is started, the timebase is used to
    /// wait a fixed span, and what the timer got through in that span is the
    /// rate.
    ///
    /// The timer is left stopped and configured as it was found. That is not
    /// tidiness: the one caller a measurement has is something reconstructing
    /// an appointment, which has already said what the timer is to deliver
    /// and would have no way to know this had thrown it away. What cannot
    /// be put back is the count — a measurement runs it down, and where it
    /// was is a moment that has passed — so the timer is stopped rather
    /// than left pretending, and whoever measured arms it again.
    ///
    /// The entry is masked throughout, so the count reaching zero during a
    /// measurement — which it should not, but a machine with a very fast bus
    /// clock and a slow caller could — delivers nothing.
    ///
    /// Both spans are bracketed the same way round: the timebase is read, then
    /// the count, then at the far end the count, then the timebase. So the
    /// elapsed span encloses the counted one by one register read at each end,
    /// which over ten milliseconds is around a hundred-thousandth of the answer
    /// and biases it low by that much — orders of magnitude inside the
    /// tolerance of the crystal being measured.
    ///
    /// # Errors
    ///
    /// [`ApicError::Clock`] if no timebase is installed to measure against —
    /// checked before the timer is touched at all — or
    /// [`ApicError::Calibration`] if the timer did not move or ran out of count
    /// before the span was over.
    pub fn calibrate(self, divisor: Divisor) -> Result<Frequency, ApicError> {
        // Before anything is written, so that a machine with no timebase is one
        // this leaves alone rather than one it leaves with a measurement
        // running. The span itself is opened later, where it is only the count
        // that separates its ends.
        if clock::now().is_none() {
            return Err(ApicError::Clock);
        }
        let access = self.0.access();
        let entry = access.read(Register::LVT_TIMER);
        let divide = access.read(Register::TIMER_DIVIDE);
        let measurement = measure(access, divisor);
        // Whatever came of it. The entry goes back before the count is put down,
        // so nothing is armed at any point by a restored entry.
        // SAFETY: both values came out of these same registers a moment ago, so
        // each is one the register accepts; and zero is the architectural way to
        // leave a counting timer stopped.
        unsafe {
            access.write(Register::TIMER_DIVIDE, divide);
            access.write(Register::LVT_TIMER, entry);
            access.write(Register::TIMER_INITIAL_COUNT, 0);
        }
        measurement
    }
}

/// Runs one measurement, leaving the timer however it ended up.
///
/// Split out so that [`Timer::calibrate`] restores the timer on every path out
/// of it, including the ones that fail part-way through.
fn measure(access: Access, divisor: Divisor) -> Result<Frequency, ApicError> {
    // SAFETY: a masked entry delivers nothing whatever the count does, and
    // the count is a plain value. Nothing is armed by this.
    unsafe {
        access.write(Register::TIMER_DIVIDE, divisor.bits());
        access.write(Register::LVT_TIMER, Entry::masked().bits());
        access.write(Register::TIMER_INITIAL_COUNT, u32::MAX);
    }
    // Opened here and not before the timer was set up, which is what keeps the
    // two spans a single register read apart at each end: the writes above are
    // uncached and there are three of them.
    let started = clock::now().ok_or(ApicError::Clock)?;
    let first = access.read(Register::TIMER_CURRENT_COUNT);
    clock::sleep_micros(CALIBRATION_MICROS).map_err(|_| ApicError::Clock)?;
    let last = access.read(Register::TIMER_CURRENT_COUNT);
    let elapsed = (clock::now().ok_or(ApicError::Clock)? - started).as_nanos();

    // Running out of count means the span was longer than the timer could
    // measure, so what is left says nothing about how far it got; not moving at
    // all means the timer is not running. Neither is a measurement.
    if last == 0 || last == first {
        return Err(ApicError::Calibration);
    }
    let nanos = u64::try_from(elapsed).map_err(|_| ApicError::Calibration)?;
    Frequency::from_measurement(u64::from(first - last), nanos).ok_or(ApicError::Calibration)
}

/// Whether this processor implements the timestamp counter deadline.
fn has_deadline() -> bool {
    processor::features().contains(Features::TSC_DEADLINE)
}

/// Whether the timer is in a mode that arms by counting.
///
/// The reserved encoding is not one of them: a controller given it does nothing
/// the architecture defines, and a count written into that is a count nothing
/// promises to act on.
fn counting(access: Access) -> bool {
    Mode::of(access.read(Register::LVT_TIMER)).is_some_and(Mode::counts)
}

/// How long a rate measurement watches the timer for.
///
/// The same order as the timestamp counter's own calibration window: long
/// enough that the cost of reading the timebase at each end is lost in it,
/// short enough that a timer counting an undivided gigahertz bus clock does not
/// run out of its 32-bit count.
const CALIBRATION_MICROS: u64 = 10_000;

#[cfg(test)]
mod tests {
    use super::{Divisor, Mode};
    use crate::lvt::{Entry, MODE_SHIFT};

    #[test]
    fn every_mode_encodes_and_decodes_where_the_entry_holds_it() {
        for mode in [Mode::OneShot, Mode::Periodic, Mode::Deadline] {
            assert_eq!(Mode::of(mode.bits()), Some(mode));
            assert_eq!(mode.bits() & !(0b11 << MODE_SHIFT), 0, "{mode:?}");
        }
        assert_eq!(Mode::of(0b11 << MODE_SHIFT), None);
    }

    #[test]
    fn the_mode_is_read_out_of_a_whole_entry_and_not_out_of_the_rest_of_it() {
        let periodic = Entry::masked().bits() | Mode::Periodic.bits();
        assert_eq!(Mode::of(periodic), Some(Mode::Periodic));
        assert_eq!(Mode::of(!(0b11 << MODE_SHIFT)), Some(Mode::OneShot));
    }

    #[test]
    fn only_the_two_counting_modes_are_armed_by_a_count() {
        assert!(Mode::OneShot.counts());
        assert!(Mode::Periodic.counts());
        assert!(!Mode::Deadline.counts());
    }

    #[test]
    fn the_divide_register_s_encoding_skips_the_bit_the_field_does_not_use() {
        let expected = [
            (Divisor::By1, 0b1011),
            (Divisor::By2, 0b0000),
            (Divisor::By4, 0b0001),
            (Divisor::By8, 0b0010),
            (Divisor::By16, 0b0011),
            (Divisor::By32, 0b1000),
            (Divisor::By64, 0b1001),
            (Divisor::By128, 0b1010),
        ];
        for (divisor, bits) in expected {
            assert_eq!(divisor.bits(), bits, "{divisor:?}");
            assert_eq!(
                divisor.bits() & 0b100,
                0,
                "bit two is not part of the field"
            );
            assert_eq!(divisor.bits() & !0b1011, 0, "{divisor:?}");
        }
    }

    #[test]
    fn every_divisor_reports_its_numeric_ratio() {
        let expected = [
            (Divisor::By1, 1),
            (Divisor::By2, 2),
            (Divisor::By4, 4),
            (Divisor::By8, 8),
            (Divisor::By16, 16),
            (Divisor::By32, 32),
            (Divisor::By64, 64),
            (Divisor::By128, 128),
        ];
        for (divisor, ratio) in expected {
            assert_eq!(divisor.ratio(), ratio, "{divisor:?}");
        }
    }
}
