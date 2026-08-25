//! The locks, the events, and the interrupt flag.
//!
//! uACPI asks for three kinds of mutual exclusion and tells them apart by where
//! they may be used. A mutex may block and may not be taken from an interrupt
//! handler. A spinlock may be taken from one, and must mask interrupts to be
//! safe there. An event is not exclusion at all but a counter, signalled from
//! one place and waited for in another.
//!
//! # What one processor makes of them
//!
//! Only one processor ever runs ACPI work in this hypervisor: bring-up brings
//! uACPI up before any other processor is started, and nothing after bring-up
//! evaluates bytecode. So a mutex here is never contended in practice, and the
//! implementations still take real atomics rather than pretending — because
//! "never contended" is a fact about the current bring-up order and not a
//! property anything checks, and an uncontended atomic costs one instruction.
//!
//! Waiting is where being alone shows through and cannot be papered over. A
//! wait for an event only ends if something signals it, and on one processor
//! the only things that could are an interrupt handler and a queued work item —
//! so a wait with nothing to wake it spins to its deadline and reports the
//! timeout. That is the honest answer rather than a hang, and it is the answer
//! uACPI is written to expect from a host that cannot deliver.
//!
//! # Naming the thread
//!
//! uACPI tracks which thread owns a mutex, so it needs a stable identifier that
//! is never its reserved "no thread" value. A processor is the unit of
//! execution here, and each one already has a block of its own at an address no
//! other processor shares — so the block's address is the name. Before this
//! processor has attached to one there is no block, and a value that no address
//! can be is used instead.

use core::{
    ptr,
    sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering},
};

use log::info;
use uacpi_sys::{Status, raw};
use x86_64::instructions::interrupts;

use crate::uacpi::{
    handle::{borrow, displace, place},
    time,
};

/// The timeout uACPI spells "wait for as long as it takes".
const FOREVER: raw::uacpi_u16 = 0xFFFF;

/// What names this processor before it has a block of its own.
///
/// Nothing is mapped at the first page of the address space, so no block can
/// ever be here, and it is not the value uACPI reserves to mean that a mutex is
/// unowned.
const UNATTACHED: raw::uacpi_thread_id = 1 as raw::uacpi_thread_id;

/// Logs how many of each kind of object uACPI is holding.
pub fn describe(who: &str) {
    info!(
        "{who}: uacpi holds {} mutexes, {} events and {} spinlocks",
        MUTEXES.load(Ordering::Relaxed),
        EVENTS.load(Ordering::Relaxed),
        SPINLOCKS.load(Ordering::Relaxed),
    );
}

/// A mutex, which may block and may not be taken from an interrupt handler.
struct Mutex {
    held: AtomicBool,
}

/// A counter, added to from one place and taken from in another.
struct Event {
    count: AtomicU32,
}

/// A lock that masks interrupts, so that it may be taken from a handler.
struct Spinlock {
    held: AtomicBool,
}

/// Creates a mutex, or reports none by returning null.
///
/// # Safety
///
/// Called by uACPI.
pub(super) unsafe extern "C" fn uacpi_kernel_create_mutex() -> raw::uacpi_handle {
    let handle = place(Mutex {
        held: AtomicBool::new(false),
    });
    if !handle.is_null() {
        MUTEXES.fetch_add(1, Ordering::Relaxed);
    }
    handle
}

/// Destroys a mutex.
///
/// # Safety
///
/// Called by uACPI with a handle [`uacpi_kernel_create_mutex`] returned, which
/// it no longer holds and will not use again.
pub(super) unsafe extern "C" fn uacpi_kernel_free_mutex(handle: raw::uacpi_handle) {
    // SAFETY: the caller vouches that the handle named a mutex from this module
    // and is not used again.
    if unsafe { displace::<Mutex>(handle) } {
        MUTEXES.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Takes a mutex, waiting up to `timeout` milliseconds for it.
///
/// # Safety
///
/// Called by uACPI with a handle [`uacpi_kernel_create_mutex`] returned.
pub(super) unsafe extern "C" fn uacpi_kernel_acquire_mutex(
    handle: raw::uacpi_handle,
    timeout: raw::uacpi_u16,
) -> raw::uacpi_status {
    // SAFETY: the caller vouches that the handle names a live mutex.
    let Some(mutex) = (unsafe { borrow::<Mutex>(handle) }) else {
        return Status::INVALID_ARGUMENT.code();
    };
    if time::wait_until(timeout, || claim(&mutex.held)) {
        Status::OK.code()
    } else {
        Status::TIMEOUT.code()
    }
}

/// Gives a mutex back.
///
/// # Safety
///
/// Called by uACPI with a handle it holds the mutex of.
pub(super) unsafe extern "C" fn uacpi_kernel_release_mutex(handle: raw::uacpi_handle) {
    // SAFETY: the caller vouches that the handle names a live mutex.
    let Some(mutex) = (unsafe { borrow::<Mutex>(handle) }) else {
        return;
    };
    mutex.held.store(false, Ordering::Release);
}

/// Creates an event, or reports none by returning null.
///
/// # Safety
///
/// Called by uACPI.
pub(super) unsafe extern "C" fn uacpi_kernel_create_event() -> raw::uacpi_handle {
    let handle = place(Event {
        count: AtomicU32::new(0),
    });
    if !handle.is_null() {
        EVENTS.fetch_add(1, Ordering::Relaxed);
    }
    handle
}

/// Destroys an event.
///
/// # Safety
///
/// Called by uACPI with a handle [`uacpi_kernel_create_event`] returned, which
/// it will not use again.
pub(super) unsafe extern "C" fn uacpi_kernel_free_event(handle: raw::uacpi_handle) {
    // SAFETY: the caller vouches that the handle named an event from this module
    // and is not used again.
    if unsafe { displace::<Event>(handle) } {
        EVENTS.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Waits for an event's counter to be positive and takes one from it.
///
/// # Safety
///
/// Called by uACPI with a handle [`uacpi_kernel_create_event`] returned.
pub(super) unsafe extern "C" fn uacpi_kernel_wait_for_event(
    handle: raw::uacpi_handle,
    timeout: raw::uacpi_u16,
) -> raw::uacpi_bool {
    // SAFETY: the caller vouches that the handle names a live event.
    let Some(event) = (unsafe { borrow::<Event>(handle) }) else {
        return false;
    };
    time::wait_until(timeout, || take(&event.count))
}

/// Adds one to an event's counter.
///
/// # Safety
///
/// Called by uACPI, possibly from an interrupt handler, with a handle
/// [`uacpi_kernel_create_event`] returned.
pub(super) unsafe extern "C" fn uacpi_kernel_signal_event(handle: raw::uacpi_handle) {
    // SAFETY: the caller vouches that the handle names a live event.
    let Some(event) = (unsafe { borrow::<Event>(handle) }) else {
        return;
    };
    event.count.fetch_add(1, Ordering::Release);
}

/// Puts an event's counter back to zero.
///
/// # Safety
///
/// Called by uACPI with a handle [`uacpi_kernel_create_event`] returned.
pub(super) unsafe extern "C" fn uacpi_kernel_reset_event(handle: raw::uacpi_handle) {
    // SAFETY: the caller vouches that the handle names a live event.
    let Some(event) = (unsafe { borrow::<Event>(handle) }) else {
        return;
    };
    event.count.store(0, Ordering::Release);
}

/// Creates a spinlock, or reports none by returning null.
///
/// # Safety
///
/// Called by uACPI.
pub(super) unsafe extern "C" fn uacpi_kernel_create_spinlock() -> raw::uacpi_handle {
    let handle = place(Spinlock {
        held: AtomicBool::new(false),
    });
    if !handle.is_null() {
        SPINLOCKS.fetch_add(1, Ordering::Relaxed);
    }
    handle
}

/// Destroys a spinlock.
///
/// # Safety
///
/// Called by uACPI with a handle [`uacpi_kernel_create_spinlock`] returned,
/// which it will not use again.
pub(super) unsafe extern "C" fn uacpi_kernel_free_spinlock(handle: raw::uacpi_handle) {
    // SAFETY: the caller vouches that the handle named a spinlock from this
    // module and is not used again.
    if unsafe { displace::<Spinlock>(handle) } {
        SPINLOCKS.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Takes a spinlock with interrupts masked, reporting the state to restore.
///
/// Cannot fail and cannot wait on a clock: a handle that names nothing still
/// has to return flags, and it returns the ones a matching unlock will restore.
///
/// # Safety
///
/// Called by uACPI, possibly from an interrupt handler, with a handle
/// [`uacpi_kernel_create_spinlock`] returned.
pub(super) unsafe extern "C" fn uacpi_kernel_lock_spinlock(
    handle: raw::uacpi_handle,
) -> raw::uacpi_cpu_flags {
    let flags = mask();
    // SAFETY: the caller vouches that the handle names a live spinlock.
    if let Some(lock) = unsafe { borrow::<Spinlock>(handle) } {
        while !claim(&lock.held) {
            core::hint::spin_loop();
        }
    }
    flags
}

/// Releases a spinlock and restores the interrupt state.
///
/// # Safety
///
/// Called by uACPI with a handle it holds the lock of, and the flags the
/// matching lock reported.
pub(super) unsafe extern "C" fn uacpi_kernel_unlock_spinlock(
    handle: raw::uacpi_handle,
    flags: raw::uacpi_cpu_flags,
) {
    // SAFETY: the caller vouches that the handle names a live spinlock.
    if let Some(lock) = unsafe { borrow::<Spinlock>(handle) } {
        lock.held.store(false, Ordering::Release);
    }
    // Last, so the window in which an interrupt could arrive is one where the
    // lock is already free.
    restore(flags);
}

/// Masks every interrupt on this processor, reporting the state to restore.
///
/// # Safety
///
/// Called by uACPI.
pub(super) unsafe extern "C" fn uacpi_kernel_disable_interrupts() -> raw::uacpi_interrupt_state {
    mask()
}

/// Restores what [`uacpi_kernel_disable_interrupts`] reported.
///
/// # Safety
///
/// Called by uACPI with a value one of its own calls returned.
pub(super) unsafe extern "C" fn uacpi_kernel_restore_interrupts(state: raw::uacpi_interrupt_state) {
    restore(state);
}

/// Names the processor that is running, never with uACPI's reserved value.
///
/// # Safety
///
/// Called by uACPI.
pub(super) unsafe extern "C" fn uacpi_kernel_get_thread_id() -> raw::uacpi_thread_id {
    if !cpu::attached() {
        return UNATTACHED;
    }
    // SAFETY: this processor has attached, which is exactly the condition
    // `cpu::current` asks for, and the block it returns lives as long as the
    // image does.
    let block = unsafe { cpu::current() };
    ptr::from_ref(block).cast_mut().cast()
}

/// Masks interrupts and reports whether they were on.
fn mask() -> raw::uacpi_interrupt_state {
    let enabled = interrupts::are_enabled();
    interrupts::disable();
    u64::from(enabled)
}

/// Puts interrupts back the way [`mask`] found them.
fn restore(state: raw::uacpi_interrupt_state) {
    if state != 0 {
        interrupts::enable();
    }
}

/// Takes a flag if it is clear, reporting whether it was taken.
fn claim(flag: &AtomicBool) -> bool {
    flag.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_ok()
}

/// Takes one from a counter if it is positive, reporting whether it was taken.
fn take(counter: &AtomicU32) -> bool {
    counter
        .try_update(Ordering::Acquire, Ordering::Relaxed, |count| {
            count.checked_sub(1)
        })
        .is_ok()
}

/// How many mutexes uACPI is holding.
static MUTEXES: AtomicUsize = AtomicUsize::new(0);

/// How many events uACPI is holding.
static EVENTS: AtomicUsize = AtomicUsize::new(0);

/// How many spinlocks uACPI is holding.
static SPINLOCKS: AtomicUsize = AtomicUsize::new(0);

/// `FOREVER` is what uACPI documents as an infinite wait, and
/// [`time::wait_until`] reads it as such. Stated here because this is where the
/// timeouts come from.
const _: () = assert!(FOREVER == raw::uacpi_u16::MAX);
