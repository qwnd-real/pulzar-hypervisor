//! Keeping a pathological period off physical hardware, and hiding nothing
//! about having done it.
//!
//! A guest may ask its periodic timer for an interval shorter than the machine
//! can answer without spending all of its time answering. The count put on
//! hardware is then lengthened, which is the one place the real timer
//! deliberately disagrees with the guest's registers — so everything that reads
//! a count back has to map between the two, and every path that changes the
//! configuration has to keep the lengthened count and the guest's in step.

use apic::{Divisor, TimerMode as HardwareMode};
use clock::Frequency;
use descriptors::Vector;
use log::warn;

use crate::{
    hardware::timer::divisor,
    registers::{
        Vlapic,
        lvt::{Entry, TimerMode},
    },
};

/// What lengthening a period needs of the physical timer.
///
/// Three operations, and the reason they are a trait rather than
/// [`apic::Timer`] itself is that the sequence they are performed in is the
/// whole of this module's difficulty: a short count must never be visible on an
/// unmasked entry, so the order is mask, reload, unmask, and getting it wrong
/// is a storm of interrupts on a physical processor. Taking the timer as a
/// parameter is what lets that order be checked without one.
pub(super) trait Physical {
    /// What the timer has left to count.
    fn remaining(&self) -> u32;

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
    ) -> Result<(), apic::ApicError>;

    /// Starts it counting down from `count`, or stops it for a count of zero.
    ///
    /// # Errors
    ///
    /// Whatever the controller refused.
    fn reload(&self, count: u32) -> Result<(), apic::ApicError>;
}

impl Physical for apic::Timer {
    fn remaining(&self) -> u32 {
        apic::Timer::remaining(*self)
    }

    fn configure(
        &self,
        delivery: Option<Vector>,
        mode: HardwareMode,
        divisor: Divisor,
    ) -> Result<(), apic::ApicError> {
        apic::Timer::configure(*self, delivery, mode, divisor)
    }

    fn reload(&self, count: u32) -> Result<(), apic::ApicError> {
        apic::Timer::reload(*self, count)
    }
}

/// Reconfigures a timer, lengthening an already-running unsafe period without
/// exposing the short count on an unmasked physical entry.
pub(super) fn reconfigure(
    vlapic: &Vlapic,
    timer: &impl Physical,
    delivery: Option<Vector>,
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
pub(super) fn clamped_count(vlapic: &Vlapic, guest_count: u32) -> Option<u32> {
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
pub(super) fn scale_remaining(remaining: u32, guest_initial: u32, physical_initial: u32) -> u32 {
    if physical_initial == 0 || guest_initial == 0 {
        return remaining;
    }
    let scaled =
        (u128::from(remaining) * u128::from(guest_initial)).div_ceil(u128::from(physical_initial));
    u32::try_from(scaled).unwrap_or(u32::MAX).min(guest_initial)
}

/// The shortest unmasked periodic interval exposed to physical hardware.
const MINIMUM_PERIOD_NANOS: u64 = 200_000;

#[cfg(test)]
mod tests {
    use apic::Divisor;

    use super::{minimum_period_count, scale_remaining};

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
}
