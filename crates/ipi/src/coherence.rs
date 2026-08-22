//! Making a processor leave the guest it is running.
//!
//! Reducing what a guest's nested page tables permit is not enough on its own:
//! every processor that walked them may still be acting on what they said, and
//! the field that discards a guest's translations belongs to a control block,
//! which is read on the way into a guest and by nothing else. So the only
//! processor that can act on it is one that is *not* inside the guest, and this
//! is the interprocessor interrupt that makes that true.
//!
//! # The handler does nothing, on purpose
//!
//! There is no work to do here. A physical interrupt arriving while a guest
//! runs forces the processor out of guest mode — with interrupt masking
//! virtualized, the host's flag at entry governs physical interrupts — and the
//! acknowledgement is the proof that the target is in host context. What
//! happens next is the target's own entry path's: it compares a counter against
//! what it last discarded for and discards again where the two differ. So the
//! handler has nothing to do and nothing to take, which is also what makes it
//! safe to run while another processor is blocked waiting for it.
//!
//! # There is no payload
//!
//! The counter is shared memory and is already the payload. Two sends that
//! coalesce therefore fold to either of them, because both mean the same thing:
//! leave the guest.

use core::num::NonZeroU64;

use cpu::CpuIndex;
use log::error;
use spin::Once;

use crate::{Ipi, IpiError, Request};

/// How long every processor is given to leave the guest and say so.
///
/// Long by the standards of what it takes — one world switch and an empty
/// handler — because what it really bounds is a processor that is not answering
/// interrupts at all, and for that the only wrong answer is waiting forever.
const ACKNOWLEDGE_MICROS: u64 = 100_000;

/// What is sent, which nothing reads.
///
/// A payload is not optional in the transport, and this is the smallest one
/// that cannot be mistaken for nothing outstanding.
const LEAVE_THE_GUEST: NonZeroU64 = NonZeroU64::MIN;

/// Registers the interrupt and hands the nested page tables the way to send it.
///
/// # Errors
///
/// Whatever registering the interrupt reported, or [`IpiError::Coherence`] if
/// the nested page tables already have a way to reach the processors inside a
/// guest.
pub(crate) fn install() -> Result<(), IpiError> {
    let ipi = crate::register(arrived, either)?;
    KICK.call_once(|| ipi);
    // The nested page tables are underneath this subsystem and have no way to
    // reach another processor. This is the whole of what they are given: one
    // function, pointing downwards.
    npt::coherence::install(kick)?;
    Ok(())
}

/// Makes every processor of `targets` leave the guest, and reports whether
/// every one of them did.
///
/// A `false` answer is not a reason to retry — the tables have already been
/// changed — but it does mean some processor may still be inside the guest
/// acting on a translation the change invalidated.
fn kick(targets: &[CpuIndex]) -> bool {
    let Some(ipi) = KICK.get() else {
        // Unreachable: the tables are only handed this function once the interrupt
        // behind it exists. Refusing is still the right answer to it, being the
        // one that cannot report a processor made to leave the guest when nothing
        // asked it to.
        error!("ipi: a processor cannot be made to leave the guest before the interrupt exists");
        return false;
    };
    match ipi.each_and_wait(
        || targets.iter().copied(),
        LEAVE_THE_GUEST,
        ACKNOWLEDGE_MICROS,
    ) {
        Ok(_) => true,
        Err(error) => {
            error!("ipi: a processor did not leave the guest: {error}");
            false
        }
    }
}

/// Nothing, which is the whole of what the target has to do here.
///
/// Arriving is the work: it is what put this processor in host context, and
/// returning is what says so. Discarding the guest's translations belongs to
/// the entry that follows, because the field that does it is only read there.
fn arrived(_request: Request) {}

/// Folds two outstanding sends into the one that answers both, which is either
/// of them.
///
/// There is nothing to fold. The counter the entry path compares against is
/// shared memory, so both sends already say everything they mean.
fn either(held: NonZeroU64, _sent: NonZeroU64) -> NonZeroU64 {
    held
}

/// The interrupt that makes a processor leave the guest, once it has been
/// registered.
static KICK: Once<Ipi> = Once::new();
