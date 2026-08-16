//! What the controller remembers about its timer.
//!
//! The registers only: the divide, the initial count, and the one thing this
//! hypervisor has to keep beside them, which is the measured rate of the real
//! timer — nothing reports it, and the shortest period a periodic timer may run
//! at is a duration rather than a count. What any of it does to the real timer
//! is [`crate::hardware::timer`]'s, and the division of labour is the
//! architecture's own: writing the configuration and starting a count are
//! different operations, so storing a value here starts nothing.
//!
//! What is deliberately *not* kept is anything about what the real timer is
//! doing. The count it reloads from and the count it has left are readable
//! registers, and reading them is exact where remembering them is a copy that
//! goes stale the first time hardware moves underneath it.

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

    /// Whether this controller has already said that its guest's periodic timer
    /// is being given a longer period than it asked for.
    ///
    /// Latched rather than counted: the guest can rewrite the count as fast as
    /// it can take an exit, and what an operator needs is the fact rather than
    /// one line per write.
    pub(crate) fn report_timer_clamp_once(&self) -> bool {
        self.timer_clamp_reported
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
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
