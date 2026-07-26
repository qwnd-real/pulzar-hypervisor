//! Telling every other processor that a translation it may hold is gone.
//!
//! Invalidating a page table entry evicts the stale translation from the
//! processor that wrote it and from no other. This is the interprocessor
//! interrupt that reaches the rest, and the function the address space
//! subsystem calls to send it.
//!
//! # Why the handler flushes everything
//!
//! Because it has no way to know what to flush. Several requests to one
//! processor coalesce into a single arrival, so a handler that invalidated one
//! address would be right only when nothing had been folded together — and
//! being right most of the time is worse than being slow, because what it
//! leaves behind is a translation to memory that has been given to something
//! else.
//!
//! Reloading the page table root evicts every translation that is not marked
//! global, and nothing pulzar maps is: so it evicts all of them. That costs the
//! processor its cached translations, which it will fault back in. Unmapping is
//! rare and correctness is not.
//!
//! # What it must not do
//!
//! Ask for the address space lock. The processor that sent this is holding it
//! and is waiting for this handler to finish; a handler that waited for the
//! lock would be waiting for the processor that is waiting for it. Reloading a
//! control register needs nothing, which is the other reason it is the right
//! answer.

use spin::Once;
use x86_64::registers::control::Cr3;

use crate::{Ipi, IpiError, Request};

/// How long every other processor is given to acknowledge.
///
/// Long by the standards of what the handler does, which is two instructions
/// and no memory access. What it really bounds is a processor that is not
/// answering interrupts at all, and for that the only wrong answer is waiting
/// forever.
const ACKNOWLEDGE_MICROS: u64 = 100_000;

/// Registers the shootdown interrupt and hands the address space subsystem the
/// way to send it.
///
/// # Errors
///
/// Whatever registering the interrupt reported. A second call is refused by the
/// address space subsystem, which takes the hook once.
pub(crate) fn install() -> Result<Ipi, IpiError> {
    let ipi = crate::register(flush)?;
    SHOOTDOWN.call_once(|| ipi);
    // The address space subsystem is underneath this one and has no way to reach
    // another processor. This is the whole of what it is given: one function,
    // pointing downwards.
    paging::shootdown::install(broadcast).map_err(|_| IpiError::AlreadyInstalled)?;
    Ok(ipi)
}

/// What the address space subsystem calls after it has invalidated something.
///
/// Reports whether every other processor acknowledged. A `false` answer is not
/// a reason to retry — the invalidation has already happened — but it does mean
/// some processor may still be holding a translation to memory that no longer
/// describes what it did, which is something the caller has to be told.
fn broadcast() -> bool {
    let Some(ipi) = SHOOTDOWN.get() else {
        // Nothing has been installed, so nothing else can be running, so there
        // is no processor that could be holding anything.
        return true;
    };
    match ipi.broadcast_and_wait(ACKNOWLEDGE_MICROS) {
        Ok(_) => true,
        Err(error) => {
            log::error!("ipi: translation shootdown incomplete: {error}");
            false
        }
    }
}

/// Drops every translation this processor has cached.
///
/// Writing the page table root back is what does it: the architecture defines
/// that as invalidating everything not marked global, and nothing pulzar maps
/// is marked global.
fn flush(_: Request) {
    let (root, flags) = Cr3::read();
    // SAFETY: this writes back the value the register already holds, so the
    // address space this processor is running in does not change and every
    // address it is using stays mapped. What it does change is the translations
    // cached for them, which is the point.
    unsafe { Cr3::write(root, flags) };
}

/// The shootdown interrupt, once it has been registered.
static SHOOTDOWN: Once<Ipi> = Once::new();
