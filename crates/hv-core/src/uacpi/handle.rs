//! Naming a host object with the opaque handle uACPI passes around.
//!
//! Several of the answers uACPI needs are objects rather than values: a mutex,
//! an event, a claimed range of ports. uACPI holds each one as a `void *` it
//! never looks inside, creates and destroys them through the host, and hands
//! the same pointer back for every operation on one.
//!
//! So a handle here is the address of one heap allocation holding one object,
//! and the three functions below are its whole life. They are shared rather
//! than written per kind because getting any of them subtly wrong — a layout
//! that does not match between allocation and release, a null that is
//! dereferenced — is the same bug whatever the object is.
//!
//! Every object reached this way is borrowed as shared, never as unique,
//! because uACPI may hold one handle in two places at once and a `&mut` derived
//! from a second borrow would alias. That is why each of them keeps all of its
//! mutable state in atomics.

use alloc::alloc::{Layout, alloc, dealloc};
use core::ptr;

use log::warn;
use uacpi_sys::raw;

/// Puts `value` on the heap and names it with a handle, or reports null.
///
/// The heap directly rather than through a `Box`, because a `Box` answers a
/// full heap by panicking and there is no reason to end a boot over an
/// allocation uACPI is prepared to be refused: every one of its create calls
/// treats null as "no memory" and reports that upwards.
pub fn place<T>(value: T) -> raw::uacpi_handle {
    const {
        assert!(
            size_of::<T>() > 0,
            "an object named by a handle needs an address of its own"
        );
    }
    let layout = Layout::new::<T>();
    // SAFETY: the layout is for one `T`, whose size is asserted non-zero above,
    // which is the whole of what the global allocator asks of its caller.
    let memory = unsafe { alloc(layout) }.cast::<T>();
    if memory.is_null() {
        warn!("core: uacpi asked for an object and the heap had no room for it");
        return ptr::null_mut();
    }
    // SAFETY: the allocation is for exactly one `T` and aligned for it, and holds
    // no previous value that would need dropping first.
    unsafe { memory.write(value) };
    memory.cast()
}

/// Borrows what a handle names, or nothing if it names nothing.
///
/// # Safety
///
/// The handle must be one [`place`] returned for a `T`, and the object it names
/// must not have been released.
pub unsafe fn borrow<T>(handle: raw::uacpi_handle) -> Option<&'static T> {
    // SAFETY: the caller vouches that a non-null handle points at a live `T` from
    // `place`, which lives until it is released.
    unsafe { handle.cast::<T>().as_ref() }
}

/// Releases what a handle names, reporting whether there was anything to
/// release.
///
/// # Safety
///
/// The handle must be one [`place`] returned for a `T`, and nothing may use the
/// object afterwards.
pub unsafe fn displace<T>(handle: raw::uacpi_handle) -> bool {
    if handle.is_null() {
        return false;
    }
    let memory = handle.cast::<T>();
    // SAFETY: the caller vouches that the handle names a live `T` from `place`
    // and that nothing uses it again, so dropping it where it is and giving the
    // allocation back with the layout it was made with is sound.
    unsafe {
        memory.drop_in_place();
        dealloc(memory.cast(), Layout::new::<T>());
    }
    true
}
