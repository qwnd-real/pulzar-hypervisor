//! Telling every other processor that a translation it may hold is gone.
//!
//! Invalidating a page table entry evicts the stale translation from the
//! processor that wrote it and from no other. This is the interprocessor
//! interrupt that reaches the rest, and the function the address space
//! subsystem calls to send it.
//!
//! # What the handler is told
//!
//! Everything it needs and nothing more: a [`Flush`], packed into the one word
//! a request carries. It says which pages stopped being described, so the
//! handler invalidates those and leaves every other translation this processor
//! has cached alone.
//!
//! Coalescing is what makes that harder than it sounds, and is why the payload
//! is merged rather than replaced. Several requests to one processor collapse
//! into a single delivery, so a handler that invalidated only the last one's
//! range would be right exactly when nothing had been folded together — and
//! being right most of the time is worse than being slow, because what it
//! leaves behind is a translation to memory that has been given to something
//! else. [`Flush::hull`] is what makes the single delivery answer for all of
//! them.
//!
//! Past a certain length the arithmetic stops paying and a request degrades to
//! [`Flush::EVERYTHING`], which reloads the page table root. That is the old
//! behaviour, now reached only where it is the cheaper answer rather than
//! always.
//!
//! # What it must not do
//!
//! Ask for the address space lock. The processor that sent this is holding it
//! and is waiting for this handler to finish; a handler that waited for the
//! lock would be waiting for the processor that is waiting for it.
//! Invalidating needs nothing, which is the other reason it is the right
//! answer.

use core::num::NonZeroU64;

use paging::shootdown::Flush;
use spin::Once;

use crate::{Ipi, IpiError, Request};

/// How long every other processor is given to acknowledge.
///
/// Long by the standards of what the handler does, which is a bounded run of
/// invalidations and no memory access. What it really bounds is a processor
/// that is not answering interrupts at all, and for that the only wrong answer
/// is waiting forever.
const ACKNOWLEDGE_MICROS: u64 = 100_000;

/// Registers the shootdown interrupt and hands the address space subsystem the
/// way to send it.
///
/// # Errors
///
/// Whatever registering the interrupt reported, or
/// [`IpiError::Shootdown`] if the address space subsystem already has a way to
/// reach the other processors.
pub(crate) fn install() -> Result<(), IpiError> {
    let ipi = crate::register(invalidate, merge)?;
    SHOOTDOWN.call_once(|| ipi);
    // The address space subsystem is underneath this one and has no way to reach
    // another processor. This is the whole of what it is given: one function,
    // pointing downwards.
    paging::shootdown::install(broadcast)?;
    Ok(())
}

/// What the address space subsystem calls after it has invalidated something.
///
/// Reports whether every other processor acknowledged. A `false` answer is not
/// a reason to retry — the invalidation has already happened — but it does mean
/// some processor may still be holding a translation to memory that no longer
/// describes what it did, which is something the caller has to be told.
fn broadcast(flush: Flush) -> bool {
    let Some(ipi) = SHOOTDOWN.get() else {
        // Nothing has been installed, so nothing else can be running, so there
        // is no processor that could be holding anything.
        return true;
    };
    match ipi.broadcast_and_wait(flush.bits(), ACKNOWLEDGE_MICROS) {
        Ok(_) => true,
        Err(error) => {
            log::error!("ipi: translation shootdown incomplete: {error}");
            false
        }
    }
}

/// Drops the translations one request describes from this processor.
///
/// Nothing to do where the payload is gone: an earlier run of this handler
/// already took a request this delivery's send had been folded into, and the
/// invalidation it asked for has happened. Returning is what settles the debt.
fn invalidate(request: Request) {
    if let Some(payload) = request.payload() {
        Flush::from_word(payload.get()).apply();
    }
}

/// Folds two outstanding requests into the one that answers both.
fn merge(held: NonZeroU64, sent: NonZeroU64) -> NonZeroU64 {
    Flush::from_word(held.get())
        .hull(Flush::from_word(sent.get()))
        .bits()
}

/// The shootdown interrupt, once it has been registered.
static SHOOTDOWN: Once<Ipi> = Once::new();
