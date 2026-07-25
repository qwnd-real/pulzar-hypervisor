//! The one address space, reachable from any processor.
//!
//! An [`AddressSpace`] is a value, and until the other processors are started
//! passing `&mut` to it is the whole of what keeps its mutations serialized.
//! That stops working the moment a second processor needs a stack of its own:
//! there is no reference to hand it, and two `AddressSpace` values over one
//! chunk would be two `&mut` to the same buddy allocator state — the very thing
//! [`AddressSpace::adopt`](crate::AddressSpace::adopt)'s safety contract has
//! always asked callers not to do.
//!
//! So the value is handed over once, with [`adopt`], and afterwards the only
//! way to reach it is [`with`], which is also the only place the lock is taken.
//! The contract stops being prose and becomes a lock.
//!
//! # Why the lock does not mask interrupts
//!
//! Because a processor waiting for it must still be able to answer a
//! translation shootdown. The processor holding the lock is the one that sends
//! those, and it waits for every other processor to acknowledge; if waiting for
//! the lock meant not answering, the two would wait for each other for good.
//!
//! The obligation this creates is stated in [`crate::shootdown`] and is the
//! only thing that makes it safe: a shootdown handler must never call [`with`].

use core::sync::atomic::{AtomicBool, Ordering};

use spin::{Mutex, Once};

use crate::{AddressSpace, PagingError};

/// Hands the address space over, so that every processor reaches the same one.
///
/// Called once, by the processor that built or adopted it, at the point where
/// it stops being one function's value and starts being the machine's. The
/// loader never calls this: it is the only thing running, it owns its space
/// from beginning to end, and it never returns from the jump.
///
/// # Errors
///
/// [`PagingError::AlreadyAdopted`] for a second call. Replacing the address
/// space every processor is running in, while they are running in it, is never
/// what a second caller wants.
pub fn adopt(space: AddressSpace) -> Result<(), PagingError> {
    // Claimed before the space is stored, so the loser of a race is refused
    // rather than silently dropping the space it brought.
    if CLAIMED
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        return Err(PagingError::AlreadyAdopted);
    }
    SPACE.call_once(|| Mutex::new(space));
    Ok(())
}

/// Runs `action` on the address space, holding the lock for exactly that long.
///
/// # Errors
///
/// [`PagingError::NotAdopted`] if nothing has been handed over yet. That means
/// the caller is running before the address space became the machine's, and
/// whatever it wanted belongs on one side of that point or the other.
pub fn with<T>(action: impl FnOnce(&mut AddressSpace) -> T) -> Result<T, PagingError> {
    let space = SPACE.get().ok_or(PagingError::NotAdopted)?;
    Ok(action(&mut space.lock()))
}

/// Whether the address space has been handed over yet.
#[must_use]
pub fn adopted() -> bool {
    SPACE.get().is_some()
}

/// The address space, once it stops belonging to one function.
static SPACE: Once<Mutex<AddressSpace>> = Once::new();

/// Claimed by the first caller of [`adopt`], so the second is refused.
static CLAIMED: AtomicBool = AtomicBool::new(false);
