//! Keeping a pathological period off physical hardware, and hiding nothing
//! about having done it.
//!
//! A guest may ask its periodic timer for an interval shorter than the machine
//! can answer without spending all of its time answering. The count put on
//! hardware is then lengthened, which is the one place the real timer
//! deliberately disagrees with the guest's registers.
//!
//! Nothing conceals that. What the guest reads out of its current-count
//! register is the count hardware is really counting, so a guest that measures
//! its timer's rate against a clock it already trusts measures the rate it is
//! actually being given — above the count it asked for, which is exactly how it
//! can tell. Mapping the count back into the range the guest asked for would
//! make the floor invisible and every interval the guest later derived from the
//! measurement wrong by the same factor.
//!
//! # The floor, and what happens when it cannot be worked out
//!
//! The floor is a duration, and a count is a duration only once the timer's
//! rate is known — which is measured, because no register reports it. So a rate
//! that is missing, or too low to be a local timer's, makes the floor
//! unanswerable, and an unanswerable floor stops the timer rather than arming
//! what the guest asked for. That direction is the whole point: the alternative
//! is a guest programming a count of one into an unmasked periodic timer and
//! taking a `#VMEXIT` per divided bus tick, which no amount of reporting
//! afterwards recovers from.

use apic::{ApicError, Divisor, TimerMode as HardwareMode};
use clock::Frequency;
use descriptors::Vector;

/// What lengthening a period needs of the physical timer.
///
/// Four operations, and the reason they are a trait rather than [`apic::Timer`]
/// itself is that the sequence they are performed in is the whole of this
/// module's difficulty: a short count must never be visible on an unmasked
/// entry, so the order is mask, reload, unmask, and getting it wrong is a storm
/// of interrupts on a physical processor. Taking the timer as a parameter is
/// what lets that order be checked without one.
pub(super) trait Physical {
    /// What the timer has left to count, which is zero when it is stopped.
    fn remaining(&self) -> u32;

    /// What it was last started from, and what a periodic timer reloads at
    /// every zero crossing.
    fn initial(&self) -> u32;

    /// Says what the timer delivers, in which mode, at what rate. Delivering
    /// `None` is what masks it.
    ///
    /// # Errors
    ///
    /// Whatever the controller refused.
    fn configure(
        &self,
        delivery: Option<Vector>,
        mode: HardwareMode,
        divisor: Divisor,
    ) -> Result<(), ApicError>;

    /// Starts it counting down from `count`, or stops it for a count of zero.
    ///
    /// # Errors
    ///
    /// Whatever the controller refused.
    fn reload(&self, count: u32) -> Result<(), ApicError>;

    /// Runs `sequence` with nothing else on this processor able to reach the
    /// controller, and answers what it answered.
    ///
    /// Part of the seam rather than of the arithmetic, because it is a property
    /// of the controller: the three writes that raise a period leave the entry
    /// masked in the middle, and a host handler interposing there — one that
    /// logs, above all, since a line of serial output costs milliseconds —
    /// holds it masked for longer than the period it is raising.
    fn exclusively<T>(&self, sequence: impl FnOnce() -> T) -> T;
}

impl Physical for apic::Timer {
    fn remaining(&self) -> u32 {
        apic::Timer::remaining(*self)
    }

    fn initial(&self) -> u32 {
        apic::Timer::initial(*self)
    }

    fn configure(
        &self,
        delivery: Option<Vector>,
        mode: HardwareMode,
        divisor: Divisor,
    ) -> Result<(), ApicError> {
        apic::Timer::configure(*self, delivery, mode, divisor)
    }

    fn reload(&self, count: u32) -> Result<(), ApicError> {
        apic::Timer::reload(*self, count)
    }

    fn exclusively<T>(&self, sequence: impl FnOnce() -> T) -> T {
        x86_64::instructions::interrupts::without_interrupts(sequence)
    }
}

/// Applies a configuration without leaving an unmasked periodic timer reloading
/// from a count shorter than the floor.
///
/// `floor` is the least physical count such a timer may run at, or `None` where
/// the machine has not said enough for one to be worked out.
///
/// The count hardware would reload from is read from hardware rather than
/// remembered, because that register is what decides the question: a count
/// written while the entry was masked, or in another mode, is still the count a
/// periodic timer reloads once the entry is unmasked. Whether hardware is going
/// to reload it at all is the other half, and the current count answers that —
/// a periodic timer reloads at the zero crossing, so a current count of zero is
/// a timer that has already crossed and stopped.
///
/// # Errors
///
/// Whatever the controller refused, in which case how much of the sequence was
/// applied is the controller's answer and not this one's: a caller that cannot
/// leave the timer as it found it stops it.
pub(super) fn reconfigure(
    timer: &impl Physical,
    delivery: Option<Vector>,
    mode: HardwareMode,
    divisor: Divisor,
    floor: Option<u32>,
) -> Result<Reconfigured, ApicError> {
    let counting = delivery.is_some() && mode == HardwareMode::Periodic && timer.remaining() != 0;
    if !counting {
        timer.configure(delivery, mode, divisor)?;
        return Ok(Reconfigured::Applied);
    }
    let Some(floor) = floor else {
        // Fail closed. Nothing here can say how long the guest's count lasts, so
        // nothing here can say it is not a storm — and the guest's own timer
        // stopping is a failure it can see, where a storm is one the machine
        // does not come back from.
        timer.configure(None, mode, divisor)?;
        timer.reload(0)?;
        return Ok(Reconfigured::Stopped);
    };
    let from = timer.initial();
    if from >= floor {
        timer.configure(delivery, mode, divisor)?;
        return Ok(Reconfigured::Applied);
    }
    // Masked across the reload, because the short count is exactly what must not
    // be visible on an entry that delivers. The cost is one tick: an expiry
    // already due when the entry was masked and not yet in the request register
    // delivers nothing, which against a period of at least the floor is a small
    // fraction of one period — and the window is bounded by holding host
    // interrupts off, without which a handler could hold the entry masked for
    // longer than the whole period.
    timer.exclusively(|| {
        timer.configure(None, mode, divisor)?;
        timer.reload(floor)?;
        timer.configure(delivery, mode, divisor)
    })?;
    Ok(Reconfigured::Raised { from, to: floor })
}

/// What a reconfiguration did to the count the timer runs at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Reconfigured {
    /// The configuration was applied and the count was left where it was.
    Applied,
    /// The count an unmasked periodic entry would have reloaded was shorter
    /// than the machine can answer, and was raised.
    Raised {
        /// What hardware would have reloaded from.
        from: u32,
        /// What it reloads from instead.
        to: u32,
    },
    /// The timer was stopped, because how long its count lasts cannot be worked
    /// out and so cannot be ruled a storm.
    Stopped,
}

/// The least physical count an unmasked periodic timer may run at on a timer of
/// this rate and divide.
///
/// `None` where it cannot be worked out at all, which is two things. A rate of
/// zero is a calibration that never happened. A rate so low that a whole
/// floor's worth of time is one divided tick or less is a floor the clamp could
/// never engage at — every count a guest can write is already at or above one
/// tick — and no local timer counts that slowly: the slowest input clock any of
/// them has ever been driven by is megahertz, so such a measurement is evidence
/// about the measurement rather than about the timer. Which matters because the
/// direction it is wrong in is the dangerous one: a gigahertz timer measured as
/// kilohertz is one whose real period at the count a guest writes is
/// nanoseconds.
pub(super) fn floor(frequency: u64, divisor: Divisor) -> Option<u32> {
    let undivided = Frequency::from_hz(frequency)?.ticks_ceil(MINIMUM_PERIOD_NANOS);
    let divided = undivided.div_ceil(u64::from(divisor.ratio()));
    (divided > 1).then(|| u32::try_from(divided).unwrap_or(u32::MAX))
}

/// The shortest unmasked periodic interval exposed to physical hardware, and so
/// the fastest rate a guest's periodic timer is given.
///
/// Chosen against the cost of the `#VMEXIT` each expiry takes rather than
/// against a software timer's granularity, which is what the same number guards
/// in a hypervisor that emulates the timer. Moving it is a measurement — one
/// exit's cost on the machine in question — and not a preference.
const MINIMUM_PERIOD_NANOS: u64 = 200_000;

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use core::cell::RefCell;

    use apic::{ApicError, Divisor, TimerMode as HardwareMode};
    use descriptors::Vector;

    use super::{Physical, Reconfigured, floor, reconfigure};

    /// A vector the guest's timer is armed with, which nothing here delivers.
    const VECTOR: Vector = Vector::new(0x30);

    /// An undivided rate a local timer plausibly counts at.
    const PLAUSIBLE: u64 = 1_000_000_000;

    /// What a controller was told, in the order it was told it.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Told {
        /// The entry was written, delivering or masked.
        Configured(Option<Vector>),
        /// The initial count was written.
        Reloaded(u32),
    }

    /// A timer that remembers what it was told rather than counting.
    struct Fake {
        remaining: u32,
        initial: RefCell<u32>,
        told: RefCell<Vec<Told>>,
    }

    impl Fake {
        /// One counting from `initial`, with `remaining` left to go.
        fn counting(initial: u32, remaining: u32) -> Self {
            Self {
                remaining,
                initial: RefCell::new(initial),
                told: RefCell::new(Vec::new()),
            }
        }

        /// What it was told, in order.
        fn told(&self) -> Vec<Told> {
            self.told.borrow().clone()
        }
    }

    impl Physical for Fake {
        fn remaining(&self) -> u32 {
            self.remaining
        }

        fn initial(&self) -> u32 {
            *self.initial.borrow()
        }

        fn configure(
            &self,
            delivery: Option<Vector>,
            _: HardwareMode,
            _: Divisor,
        ) -> Result<(), ApicError> {
            self.told.borrow_mut().push(Told::Configured(delivery));
            Ok(())
        }

        fn reload(&self, count: u32) -> Result<(), ApicError> {
            *self.initial.borrow_mut() = count;
            self.told.borrow_mut().push(Told::Reloaded(count));
            Ok(())
        }

        fn exclusively<T>(&self, sequence: impl FnOnce() -> T) -> T {
            // A host test may not execute `cli`, and what the real timer holds
            // off is host interrupts rather than anything this can observe.
            sequence()
        }
    }

    #[test]
    fn a_plausible_rate_gives_a_floor_the_divide_scales() {
        // Two hundred microseconds of the undivided rate, divided, rounding up
        // at both steps.
        assert_eq!(floor(PLAUSIBLE, Divisor::By1), Some(200_000));
        assert_eq!(floor(PLAUSIBLE, Divisor::By16), Some(12_500));
        assert_eq!(floor(100_000_001, Divisor::By2), Some(10_001));
    }

    #[test]
    fn a_rate_that_cannot_produce_a_floor_produces_none_rather_than_a_floor_of_one() {
        // A calibration that never happened, and one whose answer no local timer
        // could have given: a floor of a single divided tick is no floor at all,
        // because every count a guest can write is already at or above it.
        assert_eq!(floor(0, Divisor::By1), None);
        assert_eq!(floor(4_999, Divisor::By1), None);
        assert_eq!(floor(5_000, Divisor::By1), None);
        assert_eq!(floor(640_000, Divisor::By128), None);
        // The boundary: two divided ticks is the shortest floor a guest can write
        // a count below, which is the least that is a floor.
        assert_eq!(floor(10_000, Divisor::By1), Some(2));
        assert_eq!(floor(640_001, Divisor::By128), Some(2));
    }

    #[test]
    fn a_period_long_enough_is_applied_and_the_count_left_alone() {
        let timer = Fake::counting(20_000, 15_000);
        assert_eq!(
            reconfigure(
                &timer,
                Some(VECTOR),
                HardwareMode::Periodic,
                Divisor::By16,
                floor(PLAUSIBLE, Divisor::By16)
            ),
            Ok(Reconfigured::Applied)
        );
        assert_eq!(timer.told(), [Told::Configured(Some(VECTOR))]);
        assert_eq!(timer.initial(), 20_000);
    }

    #[test]
    fn a_period_too_short_is_raised_behind_a_masked_entry() {
        let timer = Fake::counting(1, 1);
        assert_eq!(
            reconfigure(
                &timer,
                Some(VECTOR),
                HardwareMode::Periodic,
                Divisor::By1,
                floor(PLAUSIBLE, Divisor::By1)
            ),
            Ok(Reconfigured::Raised {
                from: 1,
                to: 200_000
            })
        );
        // The order is the whole of it: the short count is never on an entry
        // that delivers.
        assert_eq!(
            timer.told(),
            [
                Told::Configured(None),
                Told::Reloaded(200_000),
                Told::Configured(Some(VECTOR)),
            ]
        );
    }

    #[test]
    fn an_unanswerable_floor_stops_the_timer_rather_than_arming_it() {
        // Fail closed, for both ways the floor can be unanswerable and for the
        // count that would otherwise be a `#VMEXIT` per divided bus tick.
        for frequency in [0, 4_999] {
            let timer = Fake::counting(1, 1);
            assert_eq!(
                reconfigure(
                    &timer,
                    Some(VECTOR),
                    HardwareMode::Periodic,
                    Divisor::By1,
                    floor(frequency, Divisor::By1)
                ),
                Ok(Reconfigured::Stopped),
                "{frequency} Hz"
            );
            assert_eq!(
                timer.told(),
                [Told::Configured(None), Told::Reloaded(0)],
                "{frequency} Hz"
            );
        }
    }

    #[test]
    fn nothing_a_masked_or_stopped_timer_holds_is_ever_raised() {
        // The floor applies to a timer that is going to deliver, and to one that
        // is going to reload. A masked entry delivers nothing, a one-shot
        // reloads nothing, and a current count of zero has already crossed —
        // raising any of them would start a timer the guest did not start.
        for (delivery, mode, remaining) in [
            (None, HardwareMode::Periodic, 1),
            (Some(VECTOR), HardwareMode::OneShot, 1),
            (Some(VECTOR), HardwareMode::Deadline, 1),
            (Some(VECTOR), HardwareMode::Periodic, 0),
        ] {
            let timer = Fake::counting(1, remaining);
            assert_eq!(
                reconfigure(
                    &timer,
                    delivery,
                    mode,
                    Divisor::By1,
                    floor(PLAUSIBLE, Divisor::By1)
                ),
                Ok(Reconfigured::Applied),
                "{delivery:?} {mode:?} with {remaining} left"
            );
            assert_eq!(timer.told(), [Told::Configured(delivery)]);
            assert_eq!(timer.initial(), 1);
        }
    }

    #[test]
    fn a_floor_that_falls_below_the_running_count_leaves_the_phase_alone() {
        // A divide write that lowers the floor under a count already at or above
        // it is pass-through, exactly as it is for a timer the floor never
        // touched: reloading would restart the period, and guest-visible phase
        // behaviour must not depend on whether the floor happens to be engaged.
        let timer = Fake::counting(12_500, 6_000);
        assert_eq!(
            reconfigure(
                &timer,
                Some(VECTOR),
                HardwareMode::Periodic,
                Divisor::By128,
                floor(PLAUSIBLE, Divisor::By128)
            ),
            Ok(Reconfigured::Applied)
        );
        assert_eq!(timer.told(), [Told::Configured(Some(VECTOR))]);
        assert_eq!(timer.initial(), 12_500);
    }
}
