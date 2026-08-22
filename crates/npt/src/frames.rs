//! Where a table frame comes from, and where one goes back to.
//!
//! Describing a region a guest has just touched costs a page table now and
//! then, and the chunk's allocator is the wrong place to ask for one on that
//! path: it is reached through the lock every processor's mapping work shares,
//! so asking there would serialize every processor's first touch of every page
//! against every mapping operation in the hypervisor.
//!
//! So each processor keeps a short list of frames of its own and fills from
//! that. The allocator is asked for anything only when a list runs low, a batch
//! at a time, and a guest whose memory has been described once never reaches
//! even that.
//!
//! # Why a list is made of atomics rather than owned
//!
//! Not because of contention: a list belongs to one processor and nothing takes
//! frames from another's. It is because the cache is reached through a shared
//! reference — that is the whole point of it, filling taking no lock — and a
//! shared reference is enough to exchange an atomic and not enough to move a
//! value out of an array.
//!
//! What the exchange buys on top of that is worth having anyway: a frame cannot
//! be handed out twice however many processors ask for one, where a load
//! followed by a store could hand one frame to two of them. The guarantee holds
//! by construction rather than by a claim about who asks.
//!
//! # When a refill happens, and when it must not
//!
//! A refill is the one operation here that reaches outside these tables, for
//! the address space the chunk's allocator lives in. So it is asked for before
//! the map is read and never while the map is held: holding two locks at once
//! is an ordering to keep, and there is no reason here to have one.

use alloc::boxed::Box;
use core::sync::atomic::{AtomicU64, Ordering};

use log::{error, info};
use paging::{Frames, chunk};
use x86_64::{PhysAddr, structures::paging::PhysFrame};

use crate::{NptError, tree::walk::Level};

/// The frames each processor of a machine fills tables from.
#[derive(Debug)]
pub(crate) struct FrameCache {
    /// One list per processor the roster describes, or one list where no roster
    /// had been taken when these tables were built.
    lists: Box<[List]>,
    /// Frames the tree has given up, which may not be handed out again until a
    /// barrier has passed.
    ///
    /// A processor caches the entries above a leaf as well as the leaf, so a
    /// frame that stopped being a table could still be under a walk in
    /// progress. Held here rather than in the list a fill takes from,
    /// because those are exactly the same thing to a fill and it must not
    /// take one of these.
    detached: List,
}

impl FrameCache {
    /// An empty list for each of `processors`.
    ///
    /// Sized by the roster, so it is built after the machine has been surveyed
    /// and before any processor enters a guest — a processor is reached by its
    /// position in that roster, and a cache built before the roster existed has
    /// one list for whoever asks.
    pub(crate) fn new(processors: usize) -> Self {
        Self {
            lists: (0..processors.max(1)).map(|_| List::new()).collect(),
            detached: List::new(),
        }
    }

    /// One frame for a table.
    ///
    /// # Errors
    ///
    /// [`NptError::OutOfFrames`] if this processor's list is empty, which means
    /// the chunk could not fill it — this is the one place a frame that could
    /// not be had stops an operation, and it is the point at which one is
    /// really needed.
    pub(crate) fn take(&self) -> Result<PhysAddr, NptError> {
        self.mine().take().ok_or(NptError::OutOfFrames)
    }

    /// Puts a frame back, for the loser of a race to install a table.
    ///
    /// A list has room for whatever a fill hands back to it, the fill having
    /// taken that frame out of the same list a moment earlier. A refusal is
    /// therefore not a state to handle, but losing a frame silently is not a
    /// way to find out that it happened.
    pub(crate) fn give(&self, frame: PhysAddr) {
        if !self.mine().give(frame) {
            error!("npt: the table frame at {frame:#x} does not fit its own list and is lost");
        }
    }

    /// Brings this processor's list up to its full depth out of an allocator
    /// the caller holds.
    ///
    /// Silent about a chunk that cannot fill it. What a short list costs is
    /// discovered by [`FrameCache::take`], which is where a missing frame stops
    /// something; reporting it here as well would make every caller answer for
    /// a failure that may not matter, a fault on a region the hypervisor
    /// answers for needing no table at all.
    pub(crate) fn stock(&self, frames: &mut Frames) {
        let list = self.mine();
        for _ in list.held()..DEPTH {
            let Ok(frame) = frames.allocate(0) else {
                break;
            };
            if !list.give(frame.start_address()) {
                release(frames, frame.start_address());
                break;
            }
        }
    }

    /// Brings this processor's list up to what one fill can need, through the
    /// address space every processor shares.
    ///
    /// One acquisition of that lock for a batch of frames rather than one per
    /// table, and none at all for a list that already holds what a fill can
    /// need — which is why describing a page a guest has touched before takes
    /// no lock outside these tables.
    ///
    /// Called before the map is read and never while it is held.
    pub(crate) fn replenish(&self) {
        if self.mine().held() >= RESERVE {
            return;
        }
        if let Err(cause) = paging::with(|space| self.stock(space.frames())) {
            error!("npt: this processor's table frames could not be replenished: {cause}");
        }
    }

    /// How many frames the cache is holding aside.
    pub(crate) fn held(&self) -> usize {
        self.lists.iter().map(List::held).sum()
    }

    /// Whether there is room to hold back another frame the tree has given up.
    ///
    /// Asked before a compaction detaches anything, because a frame with
    /// nowhere to wait would have to be either handed out early or lost.
    /// Added to only under the map's write lock, so a vacancy seen here is
    /// a vacancy still there when the frame arrives — and taken from only
    /// by a barrier, which makes room and never uses it.
    pub(crate) fn detainable(&self) -> bool {
        self.detained() < DEPTH
    }

    /// How many frames the tree has given up that no barrier has passed for
    /// yet.
    pub(crate) fn detained(&self) -> usize {
        self.detached.held()
    }

    /// Holds back a frame the tree has stopped naming, until a barrier has
    /// passed.
    pub(crate) fn detain(&self, frame: PhysAddr) {
        if !self.detached.give(frame) {
            error!("npt: the detached table frame at {frame:#x} has nowhere to wait and is lost");
        }
    }

    /// Hands back every frame the tree gave up, now that a barrier has passed
    /// and no walk can still be reading one.
    ///
    /// Into this processor's own list where it fits, because a frame that was a
    /// table is a frame for a table and the list is where a fill looks for one;
    /// to the chunk otherwise, so that a full list does not turn a frame the
    /// tree released into a frame nothing holds.
    pub(crate) fn restore(&self) {
        let list = self.mine();
        while let Some(frame) = self.detached.take() {
            if list.give(frame) {
                continue;
            }
            if let Err(cause) = paging::with(|space| release(space.frames(), frame)) {
                // Nowhere to put it: this processor's list is full and there is no
                // address space to reach the chunk through. Held where it was, for
                // whichever barrier comes next, rather than lost.
                error!(
                    "npt: the detached table frame at {frame:#x} could not be handed back: {cause}"
                );
                self.detain(frame);
                return;
            }
        }
    }

    /// Logs what the cache is keeping, which is the part of these tables'
    /// footprint that is not in the tree.
    pub(crate) fn describe(&self, who: &str) {
        info!(
            "{who}: npt holds {} table frames of {} across {} processors, {} waiting for a barrier",
            self.held(),
            self.lists.len() * DEPTH,
            self.lists.len(),
            self.detained(),
        );
    }

    /// The list belonging to the processor asking.
    ///
    /// Indexed rather than searched, and indexed without a fallback: the array
    /// is as long as the roster and a processor's index is its position in that
    /// roster, so a position is always a list.
    fn mine(&self) -> &List {
        &self.lists[self.here()]
    }

    /// Which list that is.
    fn here(&self) -> usize {
        if self.lists.len() == 1 {
            // One list is every list, so there is nothing to ask — and asking
            // would read a `GS` base that a processor which has not attached
            // does not have. A cache this size is either a machine with one
            // processor or one whose roster was not taken before these tables
            // were built.
            return 0;
        }
        // SAFETY: a cache with a list per processor was sized by a roster,
        // which is taken before any processor attaches and long before any of
        // them enters a guest — and every path that fills a table is either a
        // guest's exit or the bring-up of the processor doing it, both of which
        // are past its own attach. So this processor's `GS` base points at its
        // own block.
        unsafe { cpu::current() }.index().get()
    }
}

/// Hands a frame back to the chunk, saying so if it is refused.
///
/// A refusal loses the frame, which is worth a line: the allocator only refuses
/// a frame it never handed out, so a refusal here means the frame came from
/// somewhere this crate should not have taken it from.
pub(crate) fn release(frames: &mut Frames, frame: PhysAddr) {
    if let Err(cause) = frames.release(PhysFrame::containing_address(frame), 0) {
        error!("npt: the frame at {frame:#x} could not be handed back: {cause}");
    }
}

/// A fixed set of table frames, reachable and changeable through a shared
/// reference.
///
/// What one processor may fill tables from without asking the chunk, and — for
/// the one set that is not a processor's — what the tree has given up and may
/// not hand out yet.
///
/// Cache-line aligned, so that two processors taking frames at the same moment
/// never contend for one line.
#[derive(Debug)]
#[repr(align(64))]
struct List {
    /// The frames it holds, [`EMPTY`] where a slot holds none.
    slots: [AtomicU64; DEPTH],
}

impl List {
    /// A list holding nothing.
    fn new() -> Self {
        Self {
            slots: [const { AtomicU64::new(EMPTY) }; DEPTH],
        }
    }

    /// One frame, taken out of the list, or `None` if it holds none.
    ///
    /// The exchange is what takes the frame: a load telling something else
    /// where a frame is would let two processors sharing a list carry the same
    /// one away. A slot is read before it is exchanged so that a pass over an
    /// empty list costs loads of this processor's own cache lines rather than
    /// locked exchanges of them.
    fn take(&self) -> Option<PhysAddr> {
        self.slots.iter().find_map(|slot| {
            (slot.load(Ordering::Relaxed) != EMPTY)
                .then(|| slot.swap(EMPTY, Ordering::Acquire))
                .filter(|frame| *frame != EMPTY)
                .map(PhysAddr::new_truncate)
        })
    }

    /// Puts a frame in the list, or answers `false` if every slot is taken.
    ///
    /// Releasing, and paired with the acquiring exchange that takes one out, so
    /// that whoever fills a table out of this frame sees the zeroes the
    /// allocator wrote into it.
    fn give(&self, frame: PhysAddr) -> bool {
        self.slots.iter().any(|slot| {
            slot.compare_exchange(EMPTY, frame.as_u64(), Ordering::Release, Ordering::Relaxed)
                .is_ok()
        })
    }

    /// How many frames it holds.
    ///
    /// Counted rather than kept, because a count kept beside the slots is a
    /// second account of what they hold and the two would have to be updated
    /// together to agree. Sixteen loads of lines this processor already owns is
    /// nothing against the exit that led here.
    fn held(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| slot.load(Ordering::Relaxed) != EMPTY)
            .count()
    }
}

/// What a slot holding no frame says.
///
/// Physical zero rather than a flag beside the address, because the chunk never
/// hands out the frame at physical zero: the frames at its front hold the
/// allocator's own state and are never allocatable, so the lowest address it
/// can hand out is above its base.
const EMPTY: u64 = 0;

/// Frames one list holds when it is full.
///
/// Several fills' worth, so that a run of faults describing memory the guest
/// has not touched before refills once rather than once per table, and small
/// enough that what a processor keeps aside is a page or two of the chunk.
const DEPTH: usize = 16;

/// Frames one fill can need, which is one table for every level that can name
/// one.
///
/// A list holding this many is not refilled, which is what keeps the address
/// space out of the steady state.
const RESERVE: usize = Level::TABLES.len();

const _: () = assert!(
    chunk::METADATA_FRAMES > 0,
    "the frame at the chunk's base must never be handed out, since a list holding \
     no frame says so with physical zero",
);
const _: () = assert!(
    RESERVE < DEPTH,
    "a list must hold more than one fill can take, or every fill would refill it",
);

#[cfg(test)]
mod tests {
    //! What a refill costs, what an empty list answers, and that a frame handed
    //! back is handed out again — over a run of host memory standing in for the
    //! reserved chunk.

    use paging::Frames;

    use super::{DEPTH, FrameCache, NptError};

    #[test]
    fn a_refill_fills_the_list_and_takes_exactly_that_many_frames() {
        let (mut frames, _) = crate::tests::reserved();
        let cache = FrameCache::new(1);
        let free = frames.free();

        cache.stock(&mut frames);

        assert_eq!(cache.held(), DEPTH, "a refill brings the list to its depth");
        assert_eq!(
            frames.free(),
            free - DEPTH,
            "and costs the chunk exactly what the list now holds"
        );

        cache.stock(&mut frames);
        assert_eq!(
            frames.free(),
            free - DEPTH,
            "a list that is already full is not refilled"
        );
    }

    #[test]
    fn a_list_with_nothing_left_reports_it_rather_than_reaching_for_more() {
        let (mut frames, _) = crate::tests::reserved();
        let cache = FrameCache::new(1);
        cache.stock(&mut frames);

        for _ in 0..DEPTH {
            cache
                .take()
                .expect("a full list hands out every frame it holds");
        }

        assert_eq!(cache.held(), 0);
        assert_eq!(
            cache.take(),
            Err(NptError::OutOfFrames),
            "a fault that cannot have a table fails rather than panicking, and \
             nothing here reaches the chunk to avoid it"
        );
    }

    #[test]
    fn a_frame_handed_back_is_handed_out_again() {
        let (mut frames, _) = crate::tests::reserved();
        let cache = FrameCache::new(1);
        cache.stock(&mut frames);
        let taken = cache.take().expect("a full list has a frame");

        cache.give(taken);

        assert_eq!(cache.held(), DEPTH, "the frame is back in the list");
        assert_eq!(
            cache.take(),
            Ok(taken),
            "and is handed out again rather than lost, which is what keeps the \
             loser of a race from leaking one"
        );
    }

    #[test]
    fn a_chunk_with_nothing_left_leaves_the_list_empty_rather_than_failing_here() {
        let (mut frames, _) = crate::tests::reserved();
        let cache = FrameCache::new(1);
        drained(&mut frames);

        cache.stock(&mut frames);

        assert_eq!(cache.held(), 0);
        assert_eq!(
            cache.take(),
            Err(NptError::OutOfFrames),
            "the chunk being empty is reported where a frame is needed"
        );
    }

    /// Takes everything the chunk has left, for the paths that must answer for
    /// a list they cannot fill.
    fn drained(frames: &mut Frames) {
        while frames.allocate(0).is_ok() {}
    }
}
