//! Work uACPI defers, and where it actually runs.
//!
//! uACPI hands a host two kinds of work to run later rather than in place: a
//! general purpose event's bytecode, and a `Notify` a device raised. Both are
//! deferred for the same reason — the code that produces them can be an
//! interrupt handler, and neither can be run there — and a host with threads
//! runs them on one.
//!
//! This one has none. uACPI is brought up before any other processor is
//! started, and nothing after bring-up evaluates bytecode, so there is no
//! second thread of execution to hand an item to. What there is instead is an
//! order: an item is queued where it is produced, and it runs when uACPI next
//! asks for the queue to be empty. That is deferral without concurrency, which
//! is exactly the property the two kinds of work need — neither of them may run
//! in the middle of the bytecode that produced it — and it is the property
//! uACPI's own barrier is written to wait for.
//!
//! # Why the queue does not allocate
//!
//! Because the call that fills it may be an interrupt handler, and an interrupt
//! handler that allocates can arrive on a processor already inside the
//! allocator. So the queue is a fixed run of slots in this image's own memory,
//! and a full one is a refusal uACPI reports rather than a wait. That bounds
//! how many notifications one stretch of bytecode may raise before the queue is
//! drained; a machine that exceeds it says so, and says how many were lost.
//!
//! # Draining
//!
//! Items are taken one at a time and run with the lock released, because an
//! item is bytecode and bytecode raises notifications — an item that queued
//! another while the queue was locked would deadlock against itself. Draining
//! therefore continues until the queue is empty rather than for the number of
//! items that were in it when the drain began.

use core::sync::atomic::{AtomicUsize, Ordering};

use log::{info, warn};
use spin::Mutex;
use uacpi_sys::{Status, raw};

/// How many deferred items may be outstanding at once.
///
/// Sized for the notifications one stretch of firmware bytecode can raise
/// before the interpreter returns and the queue is drained, which is a handful
/// on the machines that raise any at all. Generous rather than tuned: the slots
/// cost one pointer pair each and live in this image's own memory for its whole
/// life.
const CAPACITY: usize = 128;

/// One deferred item.
#[derive(Clone, Copy)]
struct Item {
    /// What to call.
    handler: raw::uacpi_work_handler,
    /// The argument to call it with, which is uACPI's and opaque here.
    ctx: raw::uacpi_handle,
}

// SAFETY: the context is a pointer uACPI owns and this module only ever carries
// from the call that queued the item to the call that runs it, without reading
// through it. Both calls happen on the one processor that runs ACPI work, so
// carrying it across the queue moves it nowhere the producer could not already
// reach.
unsafe impl Send for Item {}

/// The outstanding items, oldest first.
struct Queue {
    /// The slots, used as a ring.
    slots: [Option<Item>; CAPACITY],
    /// Where the next item to run is.
    next: usize,
    /// How many slots hold an item.
    held: usize,
}

impl Queue {
    /// An empty queue.
    const fn new() -> Self {
        Self {
            slots: [const { None }; CAPACITY],
            next: 0,
            held: 0,
        }
    }

    /// Adds an item, or reports that there was no room.
    fn push(&mut self, item: Item) -> bool {
        if self.held == CAPACITY {
            return false;
        }
        self.slots[(self.next + self.held) % CAPACITY] = Some(item);
        self.held += 1;
        true
    }

    /// Takes the oldest item, if there is one.
    fn pop(&mut self) -> Option<Item> {
        let item = self.slots[self.next].take()?;
        self.next = (self.next + 1) % CAPACITY;
        self.held -= 1;
        Some(item)
    }
}

/// Logs what the queue has carried.
pub fn describe(who: &str) {
    let held = QUEUE.lock().held;
    let ran = RAN.load(Ordering::Relaxed);
    let refused = REFUSED.load(Ordering::Relaxed);
    info!("{who}: uacpi has run {ran} deferred items, {held} outstanding");
    if refused != 0 {
        warn!(
            "{who}: uacpi had {refused} deferred items refused for want of a queue slot; the \
             machine raised more than {CAPACITY} before one drain"
        );
    }
}

/// Queues an item to run when the queue is next drained.
///
/// # Safety
///
/// Called by uACPI, possibly from an interrupt handler, with a handler it owns
/// and a context that belongs to that handler.
pub(super) unsafe extern "C" fn uacpi_kernel_schedule_work(
    kind: raw::uacpi_work_type,
    handler: raw::uacpi_work_handler,
    ctx: raw::uacpi_handle,
) -> raw::uacpi_status {
    if handler.is_none() {
        return Status::INVALID_ARGUMENT.code();
    }
    // Both kinds run in the same place. uACPI asks for event bytecode on the boot
    // processor to keep clear of firmware bugs around system management interrupts,
    // and the boot processor is the only one that runs any of this — so honouring
    // the request and ignoring the distinction are the same thing here.
    let _ = kind;
    if QUEUE.lock().push(Item { handler, ctx }) {
        return Status::OK.code();
    }
    REFUSED.fetch_add(1, Ordering::Relaxed);
    Status::OUT_OF_MEMORY.code()
}

/// Runs every queued item, and every item they queue, until none is left.
///
/// uACPI's own barrier: it waits for in-flight interrupt handlers first and
/// then for deferred work. There are no interrupt handlers to wait for, because
/// installing one is refused — see [`super::interrupts`] — so the whole of the
/// wait is the drain.
///
/// # Safety
///
/// Called by uACPI outside any interrupt handler, which is what makes running
/// bytecode from here allowed.
pub(super) unsafe extern "C" fn uacpi_kernel_wait_for_work_completion() -> raw::uacpi_status {
    // The lock is taken and released once per item rather than held for the drain,
    // because an item may queue another and would otherwise wait for a lock its own
    // caller holds.
    while let Some(item) = QUEUE.lock().pop() {
        let Some(handler) = item.handler else {
            continue;
        };
        // SAFETY: the handler and its context are the pair uACPI queued together,
        // carried here unchanged, and this is the ordinary call path uACPI asks for
        // them to be run from rather than an interrupt handler.
        unsafe { handler(item.ctx) };
        RAN.fetch_add(1, Ordering::Relaxed);
    }
    Status::OK.code()
}

/// The outstanding items.
static QUEUE: Mutex<Queue> = Mutex::new(Queue::new());

/// How many items have run.
static RAN: AtomicUsize = AtomicUsize::new(0);

/// How many items were refused because the queue was full.
static REFUSED: AtomicUsize = AtomicUsize::new(0);
