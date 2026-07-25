//! Telling the other processors that a translation they may hold is gone.
//!
//! Invalidating a page table entry evicts the stale translation from the
//! processor that wrote it and from no other. Every other processor keeps
//! whatever its own translation lookaside buffer cached until something tells
//! it otherwise, and the only thing that can tell it is an interprocessor
//! interrupt.
//!
//! Sending one is not this crate's job and must not become it: the subsystem
//! that owns interprocessor interrupts is built on the one that starts the
//! other processors, which is built on this one. So the direction is inverted.
//! This module holds a slot; whoever owns interprocessor interrupts
//! [`install`]s a function into it, and the address space calls whatever is
//! there after it has invalidated something.
//!
//! # Why an empty slot is an answer and not a gap
//!
//! Before any other processor has been started there is no other translation
//! lookaside buffer in the machine, so "tell everyone else" is already true
//! when nobody has been told. An empty slot therefore reports success rather
//! than failing or refusing, which is what lets the whole address space
//! subsystem work unchanged on a machine with one processor — including one
//! where starting the others was deliberately left out.
//!
//! # What the installed function may not do
//!
//! It runs while the address space lock is held, and it runs to completion
//! before the invalidating operation returns. It must not take that lock, on
//! either the sending or the receiving side: a handler that waited for a lock
//! the processor it is answering already holds would stop the machine.

use core::{
    mem,
    ptr::{NonNull, null_mut},
    sync::atomic::{AtomicPtr, Ordering},
};

/// Makes every other processor drop the translations it has cached, and reports
/// whether every one of them acknowledged doing so.
///
/// A `false` answer means some processor did not respond in the time it was
/// given. It is not a reason to retry — the invalidation has already happened —
/// but it does mean the machine is in a state the caller has to be told about,
/// so it becomes [`PagingError::ShootdownIncomplete`](crate::PagingError).
pub type Shootdown = fn() -> bool;

/// Records how this address space reaches the other processors.
///
/// One-shot: the second caller is refused rather than allowed to replace a hook
/// that invalidations may already be going through.
///
/// # Errors
///
/// [`AlreadyInstalled`] if something already installed one.
pub fn install(hook: Shootdown) -> Result<(), AlreadyInstalled> {
    HOOK.compare_exchange(
        null_mut(),
        hook as *mut (),
        Ordering::Release,
        Ordering::Relaxed,
    )
    .map(drop)
    .map_err(|_| AlreadyInstalled)
}

/// A second attempt to say how the other processors are reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("a translation shootdown hook is already installed")]
pub struct AlreadyInstalled;

/// Runs the installed hook, or reports success if there is none because there
/// is then no other processor that could be holding anything.
pub(crate) fn broadcast() -> bool {
    // SAFETY: the only value `install` ever stores is a `Shootdown` cast to a
    // raw pointer, and null — filtered out first — is the only other value the
    // slot can hold. `transmute` checks that the two types are the same size,
    // which makes this exactly the inverse of the cast that stored it.
    NonNull::new(HOOK.load(Ordering::Acquire))
        .map(|hook| unsafe { mem::transmute::<NonNull<()>, Shootdown>(hook) })
        .is_none_or(|hook| hook())
}

/// How the other processors are reached, or null while there are none.
static HOOK: AtomicPtr<()> = AtomicPtr::new(null_mut());
