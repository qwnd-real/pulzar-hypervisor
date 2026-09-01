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
//! # What that costs, and the rules it buys
//!
//! A spin lock held with interrupts enabled, over a closure the caller chose,
//! is the shape a self-deadlock takes. Three rules keep it from being one, and
//! all three are the caller's to honour:
//!
//! 1. **No reentry.** A closure passed to [`with`] must not call [`with`], on
//!    any path, however deep. The lock is not reentrant and the second
//!    acquisition would never complete.
//! 2. **Nothing that preempts a holder may take it.** Any interrupt, fault or
//!    non-maskable interrupt handler that can arrive while a processor holds
//!    this lock must not call [`with`]. The translation shootdown handler is
//!    the one that arrives by construction, and it needs nothing from the
//!    address space; the same rule binds every handler added later.
//! 3. **It is the innermost lock of the machine.** Nothing is acquired under it
//!    that is also acquired without it, so no other lock can be waited for by
//!    the processor holding this one while its own holder waits for this. In
//!    practice that means the closure does address-space work and nothing else:
//!    no nested-paging structure, no partition, no device. The one exception is
//!    taking a guest's storage controllers over at the moment its firmware
//!    services end — asking their configuration registers how far they decode,
//!    mapping their doorbell arrays, and registering the regions the hypervisor
//!    answers for — which is a whole subsystem's bring-up under the lock, and
//!    safe only because of when it happens: the guest is stopped mid-call and
//!    the other processors do not exist yet, so with no second processor to
//!    hold the locks it takes, the wait this rule forbids cannot happen. That
//!    exception is as wide as that one moment and no wider.
//!
//! Rule 1 is checked rather than merely stated: a debug build records the
//! processor that holds the lock and refuses a second acquisition from the same
//! one instead of spinning forever, which turns the deadlock into a reported
//! error at the point that caused it. [`try_with`] is the same refusal made
//! available to release builds and to callers that would rather not wait at
//! all.
//!
//! The closure is also the unit of hold time. Whatever is inside it is what
//! every other processor waits behind, so long-running work belongs outside:
//! take what is needed, drop the lock, then do the work.

use core::sync::atomic::{AtomicU64, Ordering};

use spin::{Mutex, Once};

use crate::{AddressSpace, PagingError};

/// Hands the address space over, so that every processor reaches the same one.
///
/// Called once, by the processor that built or adopted it, at the point where
/// it stops being one function's value and starts being the machine's. The
/// loader never calls this: it is the only thing running, it owns its space
/// from beginning to end, and it never returns from the jump.
///
/// The value is published by the same operation that claims the right to
/// publish it, so there is no state in which a second caller is refused while
/// [`with`] still reports that nothing has been adopted. A caller that is
/// refused here knows the space is already reachable.
///
/// # Errors
///
/// [`PagingError::AlreadyAdopted`] for a second call. Replacing the address
/// space every processor is running in, while they are running in it, is never
/// what a second caller wants.
pub fn adopt(space: AddressSpace) -> Result<(), PagingError> {
    // The cell runs the closure for exactly one caller and publishes what it
    // returns before any other caller observes the cell as filled. Whether the
    // closure ran is therefore both the claim and the publication, in that
    // order and with no gap between them.
    let mut claimed = false;
    SPACE.call_once(|| {
        claimed = true;
        Mutex::new(space)
    });
    claimed.then_some(()).ok_or(PagingError::AlreadyAdopted)
}

/// Runs `action` on the address space, holding the lock for exactly that long.
///
/// # Errors
///
/// [`PagingError::NotAdopted`] if nothing has been handed over yet. That means
/// the caller is running before the address space became the machine's, and
/// whatever it wanted belongs on one side of that point or the other.
///
/// [`PagingError::InUse`] if this processor is detected to be inside [`with`]
/// already — the reentry rule refused at the call that would have deadlocked
/// rather than spun on. See [`Held`] for exactly what that detection covers.
pub fn with<T>(action: impl FnOnce(&mut AddressSpace) -> T) -> Result<T, PagingError> {
    let space = SPACE.get().ok_or(PagingError::NotAdopted)?;
    let me = stack_page();
    Held::refuse_reentry(me)?;
    let mut space = space.lock();
    // Declared after the guard so it is dropped before it: the holder stops
    // being recorded while the lock is still held, never after.
    let _held = Held::mark(me);
    Ok(action(&mut space))
}

/// As [`with`], but refuses rather than waits if the lock is held.
///
/// For callers that have something else to do — a diagnostic, a periodic
/// report, anything on a path that must not stall behind an unrelated mapping —
/// and the reliable form of the reentry refusal, since a lock this processor
/// already holds is a lock this cannot take.
///
/// # Errors
///
/// [`PagingError::NotAdopted`] as [`with`], or [`PagingError::InUse`] if the
/// lock is held, by this processor or another.
pub fn try_with<T>(action: impl FnOnce(&mut AddressSpace) -> T) -> Result<T, PagingError> {
    let space = SPACE.get().ok_or(PagingError::NotAdopted)?;
    let mut space = space.try_lock().ok_or(PagingError::InUse)?;
    let _held = Held::mark(stack_page());
    Ok(action(&mut space))
}

/// Whether the address space has been handed over yet.
#[must_use]
pub fn adopted() -> bool {
    SPACE.get().is_some()
}

/// Records which processor is inside the lock, so that a call from inside it is
/// refused instead of spinning against itself.
///
/// A processor is named by the page its [`with`] stack frame lives on. There is
/// no processor identifier to use: this crate sits underneath the one that
/// hands those out, and an application processor calls [`with`] before it has
/// one. A stack is the one thing a processor already has that no other
/// processor shares, and the stacks here are disjoint runs of the mapping
/// window, so two processors can never present the same page.
///
/// That makes the detection one-sided, deliberately:
///
/// - It never refuses wrongly. Equality of pages means the same stack, and the
///   same stack means the same processor.
/// - It can miss. A nested call whose frame has moved onto another page is not
///   recognized, and spins the way it would have without any of this.
///
/// One-sided is the only useful direction. A missed detection leaves behaviour
/// exactly as it was; a wrong refusal would break a boot that was correct. A
/// caller that wants the reliable answer has [`try_with`], which cannot be
/// fooled because it asks the lock itself.
struct Held;

impl Held {
    /// Refuses if the processor whose stack page is `me` is already inside the
    /// lock.
    fn refuse_reentry(me: u64) -> Result<(), PagingError> {
        (HOLDER.load(Ordering::Relaxed) != me)
            .then_some(())
            .ok_or(PagingError::InUse)
    }

    /// Records `me` as the holder until the returned value is dropped. Only
    /// ever called with the lock held, so nothing races with the store.
    fn mark(me: u64) -> Self {
        HOLDER.store(me, Ordering::Relaxed);
        Self
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        HOLDER.store(NOBODY, Ordering::Relaxed);
    }
}

/// Something no stack page can be mistaken for: nothing is mapped at zero.
const NOBODY: u64 = 0;

/// The page this call's own frame lives on.
fn stack_page() -> u64 {
    let local = 0_u8;
    core::ptr::from_ref(&local) as u64 & !(crate::chunk::FRAME_SIZE - 1)
}

/// The address space, once it stops belonging to one function.
static SPACE: Once<Mutex<AddressSpace>> = Once::new();

/// Which processor is inside the lock, or [`NOBODY`].
static HOLDER: AtomicU64 = AtomicU64::new(NOBODY);
