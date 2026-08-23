//! Changing what a guest's memory means while its processors are inside it.
//!
//! Filling an entry that described nothing, and granting permission an entry
//! withheld, need no cooperation from anybody: the page walker notices a
//! constraint being lifted by itself. Everything else does. An entry that
//! described one frame and now describes another, or that permitted a write and
//! now does not, is an entry every processor which walked it may still be
//! acting on — a processor caches the translation it derived, and it caches the
//! entries above a leaf as well as the leaf.
//!
//! # The unit is the whole guest, because the architecture gives nothing finer
//!
//! There is no instruction that invalidates a nested translation by guest
//! physical address. The one that invalidates by an alternate address space
//! identifier takes a guest *virtual* address, and the manual says outright
//! that it cannot reach a nested translation that way. So the unit is the
//! identifier: a mutation reports the range it made stricter, and that range is
//! what a barrier coalesces and what a failure names — never an argument to
//! hardware.
//!
//! Discarding an identifier's translations is a field of a control block, which
//! is per processor and read on the way into a guest. The only processor that
//! can act on it is therefore the one about to enter, and that is what shapes
//! everything here: a barrier cannot invalidate anything on another processor.
//! It can only make that processor leave the guest, and make its next entry
//! flush.
//!
//! # The handshake, and why neither side can miss the other
//!
//! Two things are shared. An epoch, one per set of tables, advanced by every
//! barrier. And one word per processor saying whether it is inside the guest,
//! which that processor publishes on the way in and clears on the way out.
//!
//! ```text
//! barrier (initiator)                   entry (each processor)
//! ----------------------------------    ----------------------------------
//! 1. epoch.fetch_add(1, Release)        A. inside.store(true, Relaxed)
//! 2. fence(SeqCst)                      B. fence(SeqCst)
//! 3. kick every processor whose         C. seen = epoch.load(Acquire)
//!    word reads inside                  D. if seen != flushed, discard this
//! 4. wait for every kick to answer         guest's translations and record it
//!                                       E. enter the guest
//! ```
//!
//! Each side stores its own word, fences, and then loads the other's, and that
//! is the whole of the argument. Take a processor which entered the guest
//! holding a translation the mutation invalidated. Its load at C answered with
//! the epoch as it stood before step 1 — that is what holding a stale
//! translation means — so C is ordered before step 1, and the fence on each
//! side carries that back to A being ordered before step 3. Step 3 therefore
//! reads a word saying inside, unless the processor has meanwhile left; and a
//! processor that left reads the epoch again when it next enters, finds it
//! advanced, and flushes there. So either the processor sees the new epoch
//! before it enters, or the barrier sees it inside and makes it leave. Both may
//! happen at once. Neither can be skipped.
//!
//! Kicking a processor that did not need it is harmless by construction, which
//! is why the two windows where that happens are left open rather than closed.
//! A processor kicked just after it left the guest does one redundant flush on
//! its next entry, and a processor kicked between publishing its word and
//! entering does the same. Closing either would need both sides to agree about
//! more than one word each, and what not closing them costs is one flush.
//!
//! # What a barrier guarantees, exactly
//!
//! When it returns, no processor can *begin* a guest access using a translation
//! that predates the mutation. It does not promise that no such access happened
//! while it was running: an access already in flight completes on the old
//! translation, and so does an access by a processor that has not been kicked
//! yet. So a caller taking memory back must not repurpose it until the barrier
//! has returned, and a caller arming a trap has to expect a bounded tail of
//! accesses that were not trapped.
//!
//! It says nothing about caches, either. A range whose memory type changed
//! would need a writeback-invalidate as well; nothing here does one, and no
//! operation on these tables changes a memory type.
//!
//! # The kick takes no lock of any kind
//!
//! It runs in interrupt context on a processor that another processor is
//! waiting for, so a handler waiting for anything the waiting processor might
//! hold would stop the machine. There is nothing for it to take: the work
//! belongs to the target's own entry path, and the interrupt exists only
//! because a physical interrupt is what forces a processor out of guest mode.
//! The acknowledgement is the proof that it is in host context.
//!
//! # Reaching the other processors from underneath them
//!
//! Sending an interprocessor interrupt is not this crate's to do and must not
//! become it: the subsystem that owns them is built on the one that starts the
//! processors, which is above these tables. So the direction is inverted the
//! way the address space subsystem already inverts it, and this module holds
//! two slots. [`install`] takes the way to make a processor leave the guest,
//! and [`watch`] takes the way to ask how many processors are running.
//!
//! # Why an empty slot is sometimes an answer and sometimes a refusal
//!
//! Before any other processor has been started nothing else can be inside a
//! guest, so "every processor was made to leave" is already true when nobody
//! has been told. An empty slot therefore reports success — but only while this
//! really is the only processor running. Once the machine has more than one, an
//! empty slot means a barrier has no way to reach the others, and reporting
//! success would be reporting that a stale translation had been discarded when
//! nothing had asked anybody to discard one.
//!
//! So the answer is conditioned on the online count rather than assumed, which
//! is what lets these tables work unchanged on a machine with one processor —
//! including one where starting the others was deliberately left out — without
//! the same code silently lying on a machine where they were started and
//! nothing was ever installed.

use alloc::{boxed::Box, vec::Vec};
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering, fence};

use cpu::CpuIndex;
use log::{error, info};
use spin::{Mutex, Once};
use thiserror::Error;
use x86_64::PhysAddr;

use crate::NptError;

/// How a barrier makes every processor of a set leave the guest it is inside,
/// and whether every one of them did.
///
/// A `false` answer is not a reason to retry — the tables have already been
/// changed — but it does mean some processor may still be inside the guest
/// acting on a translation the change invalidated, which is something the
/// caller has to be told.
pub type Kick = fn(&[CpuIndex]) -> bool;

/// How many processors are running.
///
/// Answered by whoever counts the machine's processors, and asked on the one
/// path that has to tell "nobody to reach" from "no way to reach anybody".
pub type OnlineCount = fn() -> usize;

/// Records how a barrier makes a processor leave the guest.
///
/// One-shot: a second caller is refused rather than allowed to replace a way
/// that barriers may already be going through.
///
/// # Errors
///
/// [`AlreadyInstalled`] if something already installed one.
pub fn install(kick: Kick) -> Result<(), AlreadyInstalled> {
    // The cell runs the closure for the one caller that fills it and for no
    // other, so whether it ran is exactly whether this call installed the hook.
    let mut installed = false;
    KICK.call_once(|| {
        installed = true;
        kick
    });
    installed.then_some(()).ok_or(AlreadyInstalled)
}

/// Records how the number of running processors is asked for.
///
/// Separate from [`install`] and installable before it, because the two answer
/// different questions and become available at different points in bring-up:
/// the machine's processors are surveyed before anything can reach them. Until
/// this is set these tables are entitled to believe they are alone, which is
/// true of the boot processor before it has looked.
///
/// # Errors
///
/// [`AlreadyInstalled`] if something already installed one.
pub fn watch(online: OnlineCount) -> Result<(), AlreadyInstalled> {
    let mut installed = false;
    ONLINE.call_once(|| {
        installed = true;
        online
    });
    installed.then_some(()).ok_or(AlreadyInstalled)
}

/// A second attempt to say how the processors inside a guest are reached.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[error("a way to make a processor leave the guest is already installed")]
pub struct AlreadyInstalled;

/// Everything one set of tables needs to be changed while a guest runs on them.
#[derive(Debug)]
pub(crate) struct Coherence {
    /// Advanced by every barrier, and compared against by every entry.
    epoch: AtomicU64,
    /// One per processor the roster described.
    posts: Box<[Post]>,
    /// The processors a barrier has gathered to kick.
    ///
    /// Kept rather than built, because a barrier may not allocate: the register
    /// page's description follows a guest's own writes to its interrupt
    /// controller, so a guest can drive one of these. As long as the roster,
    /// since every processor may be inside the guest at once, which is what
    /// keeps a gather from ever reaching for more room.
    ///
    /// The lock is also what keeps two barriers from overlapping. That costs
    /// nothing worth having: both would kick the same processors for the same
    /// reason, and the second would wait for acknowledgements the first is
    /// already waiting for.
    kicked: Mutex<Vec<CpuIndex>>,
    /// What the barriers taken over these tables have come to.
    barriers: Barriers,
}

/// What the barriers taken over one set of tables have come to.
///
/// Relaxed on every access and read by nothing but a report. What they are for
/// is the two failures this layer has that nothing else can see: a barrier that
/// stopped reaching a processor, and a guest driving one often enough that the
/// interrupts are the cost of running it.
#[derive(Debug)]
struct Barriers {
    /// Changes that owed nothing, so nothing was sent and nobody had to leave
    /// the guest.
    free: AtomicU64,
    /// Tightenings whose barrier found nobody inside the guest.
    alone: AtomicU64,
    /// Tightenings whose barrier had processors to make leave.
    sent: AtomicU64,
    /// How many processors those made leave the guest, in all.
    kicked: AtomicU64,
    /// Barriers a processor did not answer in the time it was given, each of
    /// which is a processor that may still have been acting on what the
    /// mutation replaced.
    incomplete: AtomicU64,
}

impl Barriers {
    /// None taken yet.
    const fn new() -> Self {
        Self {
            free: AtomicU64::new(0),
            alone: AtomicU64::new(0),
            sent: AtomicU64::new(0),
            kicked: AtomicU64::new(0),
            incomplete: AtomicU64::new(0),
        }
    }
}

impl Coherence {
    /// One post per processor the roster describes, with nobody inside the
    /// guest and nothing owed.
    ///
    /// Sized by the roster, so it is built after the machine has been surveyed
    /// and before any processor enters a guest — a processor is reached by its
    /// position in that roster, and tables built before the roster existed have
    /// nowhere for one to publish anything.
    pub(crate) fn new(processors: &[cpu::Entry]) -> Self {
        Self {
            epoch: AtomicU64::new(0),
            posts: processors
                .iter()
                .map(|entry| Post::new(entry.index()))
                .collect(),
            kicked: Mutex::new(Vec::with_capacity(processors.len())),
            barriers: Barriers::new(),
        }
    }

    /// Records a change that owed no barrier at all, which is every change that
    /// granted permission or wrote what was already written.
    ///
    /// Counted here rather than where the change was made, so that what a
    /// report says about the barriers this guest has taken includes the ones it
    /// did not have to.
    pub(crate) fn free(&self) {
        self.barriers.free.fetch_add(1, Ordering::Relaxed);
    }

    /// Makes every processor inside the guest leave it, so that none of them
    /// can begin an access with a translation older than the mutation
    /// asking for this.
    ///
    /// `first` and `bytes` are what that mutation made stricter. They are not
    /// an argument to anything the hardware does — there is no such
    /// argument — and serve to say what a barrier that could not finish was
    /// for.
    ///
    /// # Errors
    ///
    /// [`NptError::BarrierIncomplete`] if a processor that had to leave the
    /// guest did not answer, or if there is no way to reach one at all on a
    /// machine where others are running.
    pub(crate) fn barrier(&self, first: PhysAddr, bytes: u64) -> Result<(), NptError> {
        // Advanced before any processor's word is read, and releasing so that one
        // which finds the new value also sees every entry the mutation wrote.
        self.epoch.fetch_add(1, Ordering::Release);
        // This side's store, then the fence, then the other side's words.
        fence(Ordering::SeqCst);
        let mut kicked = self.kicked.lock();
        kicked.clear();
        kicked.extend(
            self.posts
                .iter()
                .filter(|post| post.station.inside())
                .map(Post::who),
        );
        if kicked.is_empty() {
            // Nobody to make leave. Every processor's next entry reads the epoch
            // this call has already advanced.
            self.barriers.alone.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
        self.barriers.sent.fetch_add(1, Ordering::Relaxed);
        self.barriers
            .kicked
            .fetch_add(kicked.len() as u64, Ordering::Relaxed);
        if evicted(&kicked) {
            return Ok(());
        }
        let unanswered = kicked.len();
        self.barriers.incomplete.fetch_add(1, Ordering::Relaxed);
        error!(
            "npt: {unanswered} processors did not leave the guest after guest physical \
             {:#x}..{:#x} was made stricter",
            first.as_u64(),
            first.as_u64().saturating_add(bytes),
        );
        Err(NptError::BarrierIncomplete { unanswered })
    }

    /// Publishes that a processor is entering the guest, and answers whether it
    /// has to discard what it cached of these tables first.
    pub(crate) fn entering(&self, who: CpuIndex) -> bool {
        match self.posts.get(who.get()) {
            Some(post) => post.station.entering(&self.epoch),
            // A processor the roster did not describe when these tables were
            // built has nowhere to publish anything, so no barrier can ever see
            // it inside the guest. Discarding on every entry instead is what
            // makes never being seen harmless.
            None => true,
        }
    }

    /// Publishes that a processor has left the guest.
    pub(crate) fn left(&self, who: CpuIndex) {
        if let Some(post) = self.posts.get(who.get()) {
            post.station.left();
        }
    }

    /// How many times these tables have been made stricter.
    ///
    /// What every entry compares against, and the only thing here a mutation
    /// changes: one number for the whole guest rather than a range per
    /// processor, because a range is not something the hardware could be
    /// asked to act on.
    pub(crate) fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Relaxed)
    }

    /// Logs how often these tables have been made stricter and who is inside
    /// the guest now, which is the whole of what this layer holds.
    pub(crate) fn describe(&self, who: &str) {
        info!(
            "{who}: npt has been made stricter {} times, {} of {} processors inside the guest",
            self.epoch(),
            self.posts
                .iter()
                .filter(|post| post.station.inside())
                .count(),
            self.posts.len(),
        );
        info!(
            "{who}: npt barriers: {} owed nothing, {} found nobody inside the guest, {} made {} \
             processors leave it, {} were not answered in time",
            self.barriers.free.load(Ordering::Relaxed),
            self.barriers.alone.load(Ordering::Relaxed),
            self.barriers.sent.load(Ordering::Relaxed),
            self.barriers.kicked.load(Ordering::Relaxed),
            self.barriers.incomplete.load(Ordering::Relaxed),
        );
    }
}

/// One processor's post: what it publishes about itself, and the name the
/// interrupt that makes it leave the guest is addressed to.
///
/// A cache line each, because a station is written by its own processor on
/// every entry and every exit — a barrier reading one must not take the line
/// away from the processor that owns it.
#[derive(Debug)]
#[repr(align(64))]
struct Post {
    /// Which processor this is.
    who: CpuIndex,
    /// Where it is, and what it has already answered for.
    station: Station,
}

impl Post {
    /// The post of the processor at `who`, with nobody inside the guest.
    const fn new(who: CpuIndex) -> Self {
        Self {
            who,
            station: Station::new(),
        }
    }

    /// Which processor this is.
    const fn who(&self) -> CpuIndex {
        self.who
    }
}

/// What one processor publishes about itself, and what it has already answered
/// for.
///
/// Nothing here names the processor it belongs to. The handshake is the same
/// whichever processor performs it, and keeping the name out of it is what
/// makes the whole of it decidable without a machine to run it on.
#[derive(Debug)]
struct Station {
    /// Whether this processor is inside the guest, as far as anything else can
    /// tell.
    inside: AtomicBool,
    /// The epoch this processor's last entry discarded the guest's translations
    /// for.
    ///
    /// Read and written by that processor alone, so it carries no ordering of
    /// its own.
    flushed: AtomicU64,
}

impl Station {
    /// Not inside the guest, and owing nothing: the epoch a set of tables
    /// starts at is the one no mutation has advanced.
    const fn new() -> Self {
        Self {
            inside: AtomicBool::new(false),
            flushed: AtomicU64::new(0),
        }
    }

    /// Publishes that this processor is entering the guest, and answers whether
    /// it has to discard what it cached of these tables first.
    ///
    /// The store, the fence and the load are this side of the handshake, in
    /// that order and for the reason the module doc gives: the store has to
    /// be visible to a barrier which reads it after this load returns a
    /// stale epoch, and the fence is what keeps the processor from
    /// performing the two the other way round.
    fn entering(&self, epoch: &AtomicU64) -> bool {
        self.inside.store(true, Ordering::Relaxed);
        fence(Ordering::SeqCst);
        // Acquiring, and paired with the releasing advance a barrier makes, so
        // that a processor which finds the epoch moved also sees every entry the
        // mutation that moved it wrote.
        let seen = epoch.load(Ordering::Acquire);
        if self.flushed.load(Ordering::Relaxed) == seen {
            return false;
        }
        // Recorded before the discard it asks for has happened, which is safe in
        // the one direction that matters: a barrier advancing the epoch between
        // here and the entry finds this processor inside and makes it leave, so
        // the discard is arranged again rather than lost.
        self.flushed.store(seen, Ordering::Relaxed);
        true
    }

    /// Publishes that this processor has left the guest.
    ///
    /// Relaxed and unfenced, because being seen inside a moment after leaving
    /// costs one redundant discard and nothing else.
    fn left(&self) {
        self.inside.store(false, Ordering::Relaxed);
    }

    /// Whether a barrier has to make this processor leave the guest.
    fn inside(&self) -> bool {
        self.inside.load(Ordering::Relaxed)
    }
}

/// Whether every processor of `targets` was made to leave the guest.
///
/// Two ways: whatever was installed to reach them, and — where nothing was —
/// the observation that a machine which has started nobody has nobody to reach.
fn evicted(targets: &[CpuIndex]) -> bool {
    match KICK.get() {
        Some(kick) => kick(targets),
        // No way to reach anybody. That is a complete answer only while there is
        // nobody to reach; otherwise these processors have not left the guest, and
        // saying they have is the one answer that leaves a stale translation in
        // use believing it is not.
        None => alone(),
    }
}

/// Whether this is the only processor running, and so whether having no way to
/// reach the others leaves anything undone.
///
/// An empty count is taken as alone, which is true of the boot processor before
/// anything has surveyed the machine.
fn alone() -> bool {
    ONLINE.get().is_none_or(|online| online() <= 1)
}

/// How a processor inside the guest is made to leave it, or empty while nothing
/// can reach one.
static KICK: Once<Kick> = Once::new();

/// How many processors are running, or empty while nothing has counted them.
static ONLINE: Once<OnlineCount> = Once::new();

#[cfg(test)]
mod tests {
    //! The handshake as the state machine it is: one station, one epoch, and
    //! every order the two can be touched in.
    //!
    //! None of it needs a processor, a control block or a set of tables. What a
    //! station publishes and what it owes is decided by two words, and keeping
    //! the name of a processor out of a station is what makes that
    //! decidable here.

    use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    use super::{Station, alone, watch};

    /// How many processors the one test that asks about them says are running.
    static RUNNING: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn a_processor_that_entered_before_a_barrier_is_made_to_leave_the_guest() {
        let (station, epoch) = quiet();
        assert!(
            !station.entering(&epoch),
            "an entry no mutation preceded discards nothing"
        );

        epoch.fetch_add(1, Ordering::Release);

        assert!(
            station.inside(),
            "and the barrier that mutation owes finds the processor inside the guest, \
             which is the only way its translations can be got rid of"
        );
    }

    #[test]
    fn a_processor_that_enters_after_a_barrier_discards_and_is_not_kicked() {
        let (station, epoch) = quiet();

        epoch.fetch_add(1, Ordering::Release);

        assert!(
            !station.inside(),
            "a processor that has not entered yet is nobody's to kick"
        );
        assert!(
            station.entering(&epoch),
            "and its own entry is what discards what it might otherwise carry in"
        );
    }

    #[test]
    fn a_processor_in_the_host_is_neither_kicked_nor_made_to_discard() {
        let (station, epoch) = quiet();
        assert!(!station.entering(&epoch), "one entry, nothing owed");

        station.left();

        assert!(
            !station.inside(),
            "leaving the guest is what takes a processor out of a barrier's reach"
        );
        assert!(
            !station.entering(&epoch),
            "and with no mutation in between, entering again costs nothing"
        );
    }

    #[test]
    fn a_processor_kicked_after_it_left_the_guest_discards_once_and_no_more() {
        let (station, epoch) = quiet();
        assert!(!station.entering(&epoch), "one entry, nothing owed");
        epoch.fetch_add(1, Ordering::Release);

        // Kicked while the barrier read it as inside, having already left: the
        // window the handshake leaves open on purpose.
        station.left();

        assert!(
            station.entering(&epoch),
            "the entry after a barrier discards, whether the kick reached the guest \
             or a processor that had already left it"
        );
        station.left();
        assert!(
            !station.entering(&epoch),
            "and one redundant discard is the whole of what being kicked late costs"
        );
    }

    #[test]
    fn two_mutations_before_one_entry_cost_that_entry_one_discard() {
        let (station, epoch) = quiet();

        epoch.fetch_add(1, Ordering::Release);
        epoch.fetch_add(1, Ordering::Release);

        assert!(
            station.entering(&epoch),
            "the entry after them discards once"
        );
        station.left();
        assert!(
            !station.entering(&epoch),
            "which answered for both, the unit being the whole guest rather than a range"
        );
    }

    #[test]
    fn with_no_way_to_reach_the_others_a_barrier_answers_only_while_there_are_none() {
        watch(running).expect("nothing else here fills the count slot");

        RUNNING.store(1, Ordering::Relaxed);
        assert!(
            alone(),
            "with one processor running, every processor that had to leave the guest \
             has left it before anybody was told"
        );

        RUNNING.store(2, Ordering::Relaxed);
        assert!(
            !alone(),
            "with two, an empty slot is a barrier that reached nobody, and reporting \
             success would report a translation discarded that nothing asked for"
        );
    }

    /// A station of a processor that has never entered the guest, and the epoch
    /// of tables no mutation has made stricter.
    fn quiet() -> (Station, AtomicU64) {
        (Station::new(), AtomicU64::new(0))
    }

    /// What the count slot is filled with: a number the one test that cares
    /// moves.
    fn running() -> usize {
        RUNNING.load(Ordering::Relaxed)
    }
}
