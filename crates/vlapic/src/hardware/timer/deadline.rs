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
//!
//! # A deadline that is no longer in the future is not rebased
//!
//! Because rebasing it would re-arm an appointment hardware has already kept.
//! The register is read, modified and written, and hardware clears it in the
//! middle of that of its own accord — masking host interrupts delays the
//! *handler*, not the expiry — so an appointment that has come due between the
//! read and the write would be written back as a moment in the past, which
//! hardware answers by firing again at once. The guest would take two
//! interrupts for one deadline.
//!
//! So the timestamp is read as well, and a deadline at or before it is left
//! exactly as hardware has it: zero if the expiry has already happened, and the
//! moment it is about to happen at otherwise. Either way the guest is owed one
//! interrupt and gets one. What is left is the few dozen cycles between that
//! comparison and the write, which no ordering of a read-modify-write against a
//! register hardware also writes can close.

use x86_64::instructions::interrupts;

use crate::{
    VlapicError,
    hardware::sources::{self, Refusal},
    machine::current,
    registers::{
        Vlapic,
        lvt::{Entry, TimerMode},
    },
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

/// Arms this processor's timer at the deadline the guest just wrote.
///
/// The deadline is already in the physical timestamp domain, translated by the
/// face that carried it, and which timer it belongs to is the timer of
/// whichever processor is executing this — which is the processor whose guest
/// wrote it, and so the processor whose controller `vlapic` is.
///
/// A refusal is reported here, once per controller, because the guest's own
/// write has already succeeded and the architecture has no bit for an
/// appointment its controller declined to keep. The answer is returned as well,
/// for the one caller with something of its own to say about it.
///
/// # Errors
///
/// Whatever the controller refused: a processor without the mode, or a timer
/// that is counting rather than waiting for a deadline — where the architecture
/// ignores the write, so an appointment reported as made would be one nobody is
/// keeping.
pub(crate) fn arm_deadline(vlapic: &Vlapic, deadline: u64) -> Result<(), apic::ApicError> {
    let armed = apic::local().and_then(|local| local.timer().set_deadline(deadline));
    if let Err(error) = armed {
        sources::refused(vlapic, Entry::Timer, Refusal::Hardware(error));
    }
    armed
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
        if !rebasable(deadline, processor::timestamp()) {
            return Ok(());
        }
        timer.set_deadline(rebased_deadline(deadline, adjustment))
    })
}

/// Whether an armed deadline is one that may still be rewritten.
///
/// Two things it may not be. Zero is not a deadline at all: it is how the
/// architecture spells a disarmed timer, and it is what hardware leaves behind
/// an expiry. And a deadline at or before the counter's current value is an
/// appointment hardware is keeping or has kept — writing it back a fixed amount
/// earlier would re-arm one that has already been kept, and hardware answers a
/// deadline in the past by firing at once, so the guest would take two
/// interrupts for one appointment.
///
/// What is left is the few dozen cycles between this comparison and the write,
/// which no ordering of a read-modify-write against a register hardware also
/// writes can close.
const fn rebasable(deadline: u64, now: u64) -> bool {
    deadline != 0 && deadline > now
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
    use super::{rebasable, rebased_deadline};

    #[test]
    fn deadline_rebasing_moves_opposite_to_the_offset() {
        assert_eq!(rebased_deadline(0x2000, 0x300), 0x1D00);
        assert_eq!(rebased_deadline(0x100, u64::MAX), 0x101);
        assert_eq!(rebased_deadline(0x1234, 0x1234), 1);
    }

    #[test]
    fn a_deadline_that_is_no_longer_in_the_future_is_not_rebased() {
        // The race the comparison exists for: hardware clears the register when
        // the deadline fires, and it does that between the read and the write of
        // a read-modify-write however host interrupts are masked. An appointment
        // that has come due is one hardware has kept or is keeping, and the guest
        // is owed exactly one interrupt for it.
        const NOW: u64 = 0x1_0000;
        assert!(rebasable(NOW + 1, NOW), "still in the future");
        assert!(!rebasable(NOW, NOW), "due exactly now");
        assert!(!rebasable(NOW - 1, NOW), "already due");
        assert!(!rebasable(0, NOW), "not armed at all");
        assert!(!rebasable(0, 0), "nor at the beginning of time");
    }
}
