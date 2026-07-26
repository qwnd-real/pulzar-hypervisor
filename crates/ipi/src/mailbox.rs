//! What lets a processor tell an interrupt it was sent from one it merely
//! received, and what lets a sender know the work it asked for has happened.
//!
//! Nothing in the arrival says who caused it. A vector is a number, and the
//! same number can be delivered by another processor, by a device, or by
//! whatever was using the machine before this hypervisor was. The only thing
//! that knows an interrupt is ours is the code that sent it, and by the time it
//! arrives that code has moved on. So it leaves a note.
//!
//! Each interprocessor interrupt has one of these per processor. Sending
//! records what is owed; the handler compares what is owed against what it has
//! already run, and an arrival with nothing between the two was not ours.
//!
//! # Why two running totals rather than one count
//!
//! The controller holds one request bit per vector, so several sends to one
//! processor that have not been serviced yet coalesce into a single delivery.
//! A count that were drained to zero on arrival would answer "was this ours"
//! destructively: the delivery that found the count already taken would be
//! reported as somebody else's, and a sender waiting on it would have to be
//! told about it through some third thing.
//!
//! Two totals that only ever climb answer both questions without touching each
//! other. What is owed minus what has been served is what is outstanding, zero
//! means the arrival belongs to someone else, and a sender waits by watching
//! the second climb past a figure it read from the first. Neither is a state
//! that a reader can consume out from under a writer.
//!
//! # The request word
//!
//! What is owed is a number, and a number is not enough to act on: a shootdown
//! that knows an invalidation happened but not where has no choice but to drop
//! everything. So a payload rides alongside, one word wide, merged by the
//! interrupt's own [`Merge`] as sends fold together.
//!
//! It is merged before the debt is recorded, which is the ordering that
//! matters: a handler that observes the debt has, by that fact, already been
//! able to observe the payload explaining it. The reverse order would let a
//! handler run for a request whose payload had not landed yet.
//!
//! # Why each of these fills a cache line
//!
//! Two processors counting their own arrivals must never write to the same
//! line. Without the padding every send would pull the line away from the
//! processor about to read it, which is the one thing a mechanism this small
//! must not do. The counters that would otherwise be one set of shared globals
//! live in that padding, so the accounting costs no line of its own.

use core::{
    num::NonZeroU64,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::Merge;

/// One processor's state for one interprocessor interrupt.
#[derive(Debug)]
#[repr(C, align(64))]
pub(crate) struct Mailbox {
    /// Requests addressed to this processor, ever.
    owed: AtomicU64,
    /// Requests it has run the handler for, ever. Never above
    /// [`Mailbox::owed`].
    served: AtomicU64,
    /// What the outstanding requests add up to, or zero once taken.
    request: AtomicU64,
    /// Deliveries that turned out to be ours.
    arrivals: AtomicU64,
    /// Deliveries on this vector that turned out not to be.
    foreign: AtomicU64,
    /// Enough to fill the line, so that the next mailbox starts on a new one.
    _padding: [u64; 3],
}

impl Mailbox {
    /// A mailbox nothing has been sent to.
    pub(crate) const fn new() -> Self {
        Self {
            owed: AtomicU64::new(0),
            served: AtomicU64::new(0),
            request: AtomicU64::new(0),
            arrivals: AtomicU64::new(0),
            foreign: AtomicU64::new(0),
            _padding: [0; 3],
        }
    }

    /// Merges `payload` into what is outstanding and records that one more
    /// request is owed.
    ///
    /// `merge` is only ever handed two payloads that were sent, never the empty
    /// word a taken request leaves behind: the first payload after a handler
    /// has drained one stands on its own.
    pub(crate) fn owe(&self, payload: NonZeroU64, merge: Merge) {
        let _ = self
            .request
            .try_update(Ordering::Release, Ordering::Relaxed, |held| {
                Some(
                    match NonZeroU64::new(held) {
                        Some(held) => merge(held, payload),
                        None => payload,
                    }
                    .get(),
                )
            });
        // Released after the payload, so that a handler which sees this debt
        // sees what explains it. This is what the arrival path acquires against.
        self.owed.fetch_add(1, Ordering::Release);
    }

    /// What this processor has been asked for, or `None` if the arrival was not
    /// ours.
    ///
    /// Taking the request word does not settle the debt: the handler has to run
    /// first, and [`Mailbox::done`] is what says it did.
    pub(crate) fn claim(&self) -> Option<Claim> {
        let upto = self.owed.load(Ordering::Acquire);
        if upto == self.served.load(Ordering::Relaxed) {
            self.foreign.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        self.arrivals.fetch_add(1, Ordering::Relaxed);
        Some(Claim {
            payload: self.request.swap(0, Ordering::Acquire),
            upto,
        })
    }

    /// Records that everything `claim` covered has now run.
    ///
    /// A plain store rather than an addition because only this processor ever
    /// writes here, and it writes with its own interrupts masked: no two
    /// handlers for one mailbox can be in flight at once, and each of them read
    /// a figure at least as high as the last one wrote.
    pub(crate) fn done(&self, claim: Claim) {
        self.served.store(claim.upto, Ordering::Release);
    }

    /// Everything owed as of now, which is what [`Mailbox::caught_up`] is asked
    /// about.
    pub(crate) fn outstanding(&self) -> u64 {
        self.owed.load(Ordering::Acquire)
    }

    /// Whether the handler has run everything that was owed as of `upto`.
    pub(crate) fn caught_up(&self, upto: u64) -> bool {
        self.served.load(Ordering::Acquire) >= upto
    }

    /// What this mailbox has seen, for the one place it is reported.
    pub(crate) fn counts(&self) -> Counts {
        Counts {
            owed: self.owed.load(Ordering::Relaxed),
            served: self.served.load(Ordering::Relaxed),
            arrivals: self.arrivals.load(Ordering::Relaxed),
            foreign: self.foreign.load(Ordering::Relaxed),
        }
    }
}

/// An arrival that was ours, and everything it is answering for.
///
/// Carried from [`Mailbox::claim`] to [`Mailbox::done`] rather than re-read in
/// between, because what the handler is answering for has to be the figure that
/// was true when the request was taken — anything sent since is a debt the next
/// delivery settles.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Claim {
    payload: u64,
    upto: u64,
}

impl Claim {
    /// Everything the outstanding requests merged to.
    ///
    /// Zero where a previous handler had already taken a payload this
    /// delivery's send was folded into, which is the honest answer: the
    /// work it describes has been done.
    pub(crate) const fn payload(self) -> u64 {
        self.payload
    }
}

/// What one mailbox has seen, or every mailbox once they are added up.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Counts {
    /// Requests sent here.
    pub(crate) owed: u64,
    /// Requests answered.
    pub(crate) served: u64,
    /// Deliveries that were ours.
    pub(crate) arrivals: u64,
    /// Deliveries on one of our vectors that were not.
    pub(crate) foreign: u64,
}

impl Counts {
    /// Both of these together, which is how the machine's total is reached from
    /// the mailboxes it is spread across.
    pub(crate) const fn and(self, other: Self) -> Self {
        Self {
            owed: self.owed + other.owed,
            served: self.served + other.served,
            arrivals: self.arrivals + other.arrivals,
            foreign: self.foreign + other.foreign,
        }
    }
}
