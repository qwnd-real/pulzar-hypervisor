//! What the controller remembers about its timer.
//!
//! The registers only: the divide, the initial count, and what this hypervisor
//! has to keep beside them — the measured rate of the real timer, and the
//! physical count standing in for a period too short to put on hardware. What
//! any of it does to the real timer is [`crate::hardware::timer`]'s, and the
//! division of labour is the architecture's own: writing the configuration and
//! starting a count are different operations, so storing a value here starts
//! nothing.

use core::sync::atomic::Ordering;

use crate::registers::{
    Vlapic,
    lvt::{Entry, TimerMode},
};

impl Vlapic {
    /// How far the bus clock is divided before the timer counts it.
    pub(crate) fn timer_divide(&self) -> u32 {
        self.timer_divide.load(Ordering::Acquire)
    }

    /// Sets the timer's divide configuration.
    pub(crate) fn set_timer_divide(&self, value: u32) {
        self.timer_divide
            .store(value & TIMER_DIVIDE_MASK, Ordering::Release);
    }

    /// What the timer counts down from.
    pub(crate) fn timer_initial(&self) -> u32 {
        self.timer_initial.load(Ordering::Acquire)
    }

    /// Sets what the timer counts down from, and says whether the write took.
    ///
    /// Refused in deadline mode, where the architecture has the count registers
    /// stop meaning anything and ignores writes to them. Ignored rather than
    /// faulted, and ignored completely: the value is not stored either, so a
    /// guest that writes a count in deadline mode and later selects a counting
    /// mode does not find the count it wrote waiting to start a timer it never
    /// asked for.
    pub(crate) fn set_timer_initial(&self, value: u32) -> bool {
        if self.timer_mode() == Some(TimerMode::Deadline) {
            return false;
        }
        self.timer_initial.store(value, Ordering::Release);
        true
    }

    /// The calibrated rate of the timer before division, in ticks per second.
    pub(crate) fn timer_frequency(&self) -> u64 {
        self.timer_frequency.load(Ordering::Acquire)
    }

    /// Records the timer's undivided calibrated rate.
    pub(crate) fn set_timer_frequency(&self, frequency: u64) {
        self.timer_frequency.store(frequency, Ordering::Release);
    }

    /// The physical initial count used to lengthen a pathological period.
    pub(crate) fn timer_clamp(&self) -> u32 {
        self.timer_clamp.load(Ordering::Acquire)
    }

    /// Records the physical initial count backing the guest's periodic timer.
    pub(crate) fn set_timer_clamp(&self, count: u32) {
        self.timer_clamp.store(count, Ordering::Release);
    }

    /// Stops scaling current-count reads for a physically lengthened period.
    pub(crate) fn clear_timer_clamp(&self) {
        self.set_timer_clamp(0);
    }

    /// Whether this controller has already reported period clamping.
    pub(crate) fn report_timer_clamp_once(&self) -> bool {
        self.timer_clamp_reported
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// Whether a nonzero periodic count was successfully loaded on hardware.
    pub(crate) fn timer_periodic_running(&self) -> bool {
        self.timer_periodic_running.load(Ordering::Acquire)
    }

    /// Records whether the physical timer is running periodically.
    pub(crate) fn set_timer_periodic_running(&self, running: bool) {
        self.timer_periodic_running
            .store(running, Ordering::Release);
    }

    /// Which mode the timer's entry selects, or `None` for the encoding the
    /// architecture reserves.
    pub(crate) fn timer_mode(&self) -> Option<TimerMode> {
        TimerMode::from_bits(self.lvt(Entry::Timer).timer_mode())
    }
}

/// The timer's divide configuration is three bits, and not three adjacent
/// ones: bit two is reserved and sits in the middle of them.
///
/// Reached by the model-specific-register face as well, which has to fault on
/// exactly the bits this drops — the reserved bit between them included.
pub(crate) const TIMER_DIVIDE_MASK: u32 = 0b1011;
