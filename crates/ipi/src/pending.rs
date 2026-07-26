//! What lets a processor tell an interrupt it was sent from one it merely
//! received.
//!
//! Nothing in the arrival says who caused it. A vector is a number, and the
//! same number can be delivered by another processor, by a device, or by
//! whatever was using the machine before this hypervisor was. The only thing
//! that knows an interrupt is ours is the code that sent it, and by the time it
//! arrives that code has moved on. So it leaves a note.
//!
//! Each interprocessor interrupt has one counter per processor. Sending
//! increments the target's; the target's handler takes the whole count and, if
//! there was one, the arrival was ours.
//!
//! # Why the count is drained rather than decremented
//!
//! Not as an optimization — for correctness. The controller holds one request
//! bit per vector, so several sends to one processor that have not been
//! serviced yet coalesce into a single delivery. A counter decremented by one
//! would leave a residue behind, and the next arrival on that vector — which
//! might be a device's, or a guest's — would find it and be claimed as ours.
//!
//! Draining also makes the count worth handing to the handler: it is how many
//! requests were folded into the one delivery, which is something a handler may
//! legitimately want to know and can otherwise never find out.
//!
//! # The other counter
//!
//! Sending is not the same as being served, and a sender that has to know the
//! work happened needs to wait for something. So the handler advances a second
//! counter after it runs, and a sender that waits watches that one. Two
//! counters rather than one because they answer different questions: what is
//! owed, and what has been done.

use core::sync::atomic::{AtomicU64, Ordering};

/// One processor's state for one interprocessor interrupt.
///
/// Aligned to a cache line and padded out to one, so that two processors
/// counting their own arrivals never write to the same line. Without it every
/// send would pull the line away from the processor about to read it, which is
/// the one thing a mechanism this small must not do.
#[derive(Debug)]
#[repr(C, align(64))]
pub(crate) struct Slot {
    /// Requests sent to this processor that it has not taken yet.
    pending: AtomicU64,
    /// Requests it has taken and run the handler for.
    served: AtomicU64,
    /// Enough to fill the line, so that the next slot starts on a new one.
    _padding: [u64; 6],
}

impl Slot {
    /// A slot nothing has been sent to.
    pub(crate) const fn new() -> Self {
        Self {
            pending: AtomicU64::new(0),
            served: AtomicU64::new(0),
            _padding: [0; 6],
        }
    }

    /// Records that one more request is owed to this processor.
    ///
    /// Release ordering, and called before the command is written, so that a
    /// processor which sees the interrupt also sees the count that explains it.
    pub(crate) fn owe(&self) {
        self.pending.fetch_add(1, Ordering::Release);
    }

    /// Takes everything owed, and reports how much that was.
    ///
    /// Zero means the arrival was not ours.
    pub(crate) fn take(&self) -> u64 {
        self.pending.swap(0, Ordering::Acquire)
    }

    /// Records that `count` requests have been run.
    pub(crate) fn done(&self, count: u64) {
        self.served.fetch_add(count, Ordering::Release);
    }

    /// How much this processor has run so far.
    ///
    /// What a sender waiting for its request to be served watches, by comparing
    /// against what it read before sending.
    pub(crate) fn progress(&self) -> u64 {
        self.served.load(Ordering::Acquire)
    }
}
