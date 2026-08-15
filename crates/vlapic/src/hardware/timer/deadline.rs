//! The timestamp-counter deadline, which is an appointment rather than a count.
//!
//! Nothing about it is mirrored in this crate. Hardware clears the register
//! when the deadline fires, and nothing about that expiry is visible anywhere
//! else, so a software copy would go stale exactly once per expiry and would
//! then be re-armed by the next reconfiguration — firing immediately, from a
//! moment already in the past.
//!
//! The guest's timestamp counter is offset from the host's by its control
//! block, so a deadline is translated on the way in and out by the face that
//! carries it. An offset change rebases the armed deadline here, before the new
//! offset is published, so an absolute appointment stays the same guest-visible
//! number.

use apic::LocalApic;
use log::warn;
use x86_64::instructions::interrupts;

use crate::{
    VlapicError,
    machine::current,
    registers::{Vlapic, lvt::TimerMode},
};

/// Keeps an armed timestamp-counter deadline fixed while the guest's timestamp
/// offset changes.
///
/// `adjustment` is the wrapping difference between the new offset and the old
/// one. A deadline in physical timestamp space moves by the opposite amount;
/// counting timers and disarmed deadline timers are unchanged.
///
/// # Errors
///
/// [`VlapicError::NotInstalled`] before [`crate::install`],
/// [`VlapicError::NoLapic`] on a processor with no controller, or
/// [`VlapicError::Apic`] if the physical deadline cannot be read or rewritten.
pub fn adjust_deadline(adjustment: u64) -> Result<(), VlapicError> {
    if adjustment == 0 {
        return Ok(());
    }
    Ok(rebase(current()?, adjustment)?)
}

/// Arms the guest's timer at the deadline it just wrote.
pub(crate) fn arm_deadline(vlapic: &Vlapic, deadline: u64) {
    let Ok(timer) = apic::local().map(LocalApic::timer) else {
        return;
    };
    if let Err(error) = timer.set_deadline(deadline) {
        warn!(
            "vlapic: {} could not arm its timer deadline: {error}",
            vlapic.index()
        );
    }
}

/// Rebases an armed physical deadline across a guest timestamp-offset change.
///
/// `adjustment` is the amount added to the old offset. The physical deadline
/// moves by the opposite amount so that adding the new offset still produces
/// the same guest-visible absolute deadline. A disarmed deadline and every
/// counting mode have nothing to rebase.
///
/// # Errors
///
/// The error returned by the physical controller if its timer cannot be read or
/// rewritten.
fn rebase(vlapic: &Vlapic, adjustment: u64) -> Result<(), apic::ApicError> {
    if adjustment == 0 || vlapic.timer_mode() != Some(TimerMode::Deadline) {
        return Ok(());
    }
    interrupts::without_interrupts(|| {
        let timer = apic::local()?.timer();
        let deadline = timer.deadline()?;
        if deadline == 0 {
            return Ok(());
        }
        timer.set_deadline(rebased_deadline(deadline, adjustment))
    })
}

/// Moves a physical deadline opposite to a guest timestamp-offset change.
///
/// Physical zero disarms the deadline timer. The single wrapping combination
/// that would produce it is therefore represented by the earliest armed value
/// instead, one physical timestamp tick away.
const fn rebased_deadline(deadline: u64, adjustment: u64) -> u64 {
    let rebased = deadline.wrapping_sub(adjustment);
    if rebased == 0 { 1 } else { rebased }
}

#[cfg(test)]
mod tests {
    use super::rebased_deadline;

    #[test]
    fn deadline_rebasing_moves_opposite_to_the_offset() {
        assert_eq!(rebased_deadline(0x2000, 0x300), 0x1D00);
        assert_eq!(rebased_deadline(0x100, u64::MAX), 0x101);
        assert_eq!(rebased_deadline(0x1234, 0x1234), 1);
    }
}
