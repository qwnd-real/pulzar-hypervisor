//! The second set of page tables: guest physical addresses to system physical
//! addresses.
//!
//! Under nested paging a guest walks its own page tables freely and the
//! addresses it believes are physical are translated again by these. That is
//! what makes a guest fast — no shadow tables, no intercept on every change the
//! guest makes to its own mapping — and it is what lets a guest be given a view
//! of physical memory that is not the machine's.
//!
//! Here that view is almost the machine's own. Pulzar passes the hardware
//! through, so a guest physical address maps to the identical system physical
//! address, and these tables exist to say two things: *yes, identically* for
//! everything the machine has, and *no* for the regions that are not the
//! guest's.
//!
//! # Two jobs, kept apart
//!
//! What an address *means* and how that meaning is written into four levels of
//! hardware table are separate. [`map`] answers the first, for a *run* of
//! addresses at a time — the largest run the same answer covers — and [`tree`]
//! answers the second, taking the granularity to describe a region at from the
//! run rather than probing for it. [`frames`] is where the tables themselves
//! come from. What is left here is the handle the three are reached through and
//! the operations that need more than one of them.
//!
//! # More than one processor, at the same time
//!
//! Describing a page a guest has just touched happens on every processor of the
//! machine at once, so [`Npt::fault`] takes a shared reference and nothing it
//! does serializes one processor against another doing the same. Two different
//! things make that safe.
//!
//! The tree needs no lock at all. Every entry is an atomic, a table is
//! installed with a compare-exchange, and the loser of that exchange follows
//! the winner's table — which is sound *because* the granularity a fill chooses
//! is a function of the map and the address and of nothing else, so two
//! processors describing one region compute the same entry.
//!
//! The map needs one, because the operations that change what an address means
//! rewrite it. Faults share it, and those operations take it exclusively and
//! hold it across the tree as well, so a fault can never describe a region the
//! map has stopped agreeing with. Reading a translation shares it too, for a
//! second reason: a mutation that gives a table back takes part of the tree out
//! from under a walk, and the lock is what keeps that from happening between
//! one entry of a walk and the next.
//!
//! Table frames come from a list per processor, so the chunk's allocator is off
//! the path too. What is left of a fault on memory the guest has touched before
//! is one read-lock acquisition, one search of the map and a walk down three
//! entries: no allocation, and no lock outside these tables.
//!
//! Filling one of those lists is the one thing here that reaches outside these
//! tables, for the allocator the address space owns, and it is done before the
//! map is taken and never under it. So no processor ever holds one of these two
//! locks while it waits for the other, and there is no ordering between them to
//! state.
//!
//! # Built as it is asked for
//!
//! Nothing is described until a guest touches it. A guest physical address with
//! no translation raises `#VMEXIT(NPF)`, the handler calls [`Npt::fault`], and
//! that describes the region containing the address. A machine with a terabyte
//! of memory therefore costs a handful of tables rather than one per gigabyte
//! it never touched, and a device aperture far above the last byte of RAM is
//! described the moment it is used and not before.
//!
//! The region a fault describes is the coarsest page whose whole extent the
//! answer covers: 1 GiB where the processor has 1 GiB pages, otherwise 2 MiB,
//! otherwise one page. Which of the three it is takes no searching, because the
//! answer already says how far it reaches — a page of real memory with nothing
//! of the hypervisor's or of a device's within a gigabyte of it is answered for
//! by the gigabyte, and one that sits a page below a device is answered for by
//! the page.
//!
//! # What the guest sees instead of the hypervisor
//!
//! Every guest physical page of the reserved chunk — pulzar's image, its heap,
//! its stacks, its descriptor tables, its control blocks and these tables
//! themselves — is mapped to one shared frame of zeroes, read-only. A guest
//! reading there sees zeroes rather than either the hypervisor or a fault,
//! which is what keeps a guest that merely walks physical memory out of
//! trouble.
//!
//! A guest *writing* there is a different matter. The write faults and
//! [`Npt::fault`] reports [`Outcome::Refused`], which is as far as these tables
//! can take it: there is no page to accept the write and there never will be.
//! Resuming the guest unchanged re-executes the instruction and faults again,
//! so the caller has to step over it instead — emulate the instruction, discard
//! the write, and resume past it. That is not something the tables can do, and
//! it is stated here so that a live-lock is not diagnosed as a bug in them.
//!
//! A guest whose *own page tables* are in such memory is the one case where
//! stepping over the instruction does not help, because what faulted is the
//! walk rather than the access: the instruction would fault again on the same
//! walk for ever. It is reported apart, as [`Outcome::WalkRefused`], so that a
//! caller answers it with the exception the guest's own architecture owes it.
//!
//! [`Npt::create`] checks that the chunk is 2 MiB aligned and a whole number of
//! 2 MiB regions rather than assuming it, because that is what makes the 2 MiB
//! region containing any page of the chunk entirely the chunk's — so the shadow
//! is written a whole page table at a time, and the only pages of one it leaves
//! alone are the few the guest is shown on purpose.
//!
//! # Regions the hardware does not answer for
//!
//! [`Npt::protect`] marks a range as one whose accesses belong to something
//! other than the memory or device behind it — a device the hypervisor
//! interposes on, presenting the guest a view that is not the hardware's.
//! Either writes alone fault or every access does, and [`Npt::fault`] reports
//! [`Outcome::Interposed`] for an address inside one rather than describing it.
//!
//! Trapping a range is the one thing here that needs finer granularity than the
//! fill rule would otherwise produce, so it splits whatever larger page covers
//! the range into pages first. Afterwards it needs no special case: the answer
//! for the memory around a trapped region stops at the region's edge, so the
//! 511 pages either side of one trapped page are still described 2 MiB at a
//! time and reached at full speed.
//!
//! # Pages given rather than trapped
//!
//! [`Npt::sink`] is the opposite of [`Npt::protect`]: where trapping keeps a
//! page from being described so that every access exits, sinking describes one
//! page as a frame of its own, writable, that nothing reads — a place for the
//! guest's accesses to go without anywhere to arrive. The use for one is the
//! register page of an interrupt controller the processor drives itself, which
//! the acceleration requires to translate to memory the guest may write while
//! redirecting every access away from it: an exit is no longer the expected
//! shape of an access, and the page still has to translate to something.
//!
//! # Every region has a name, whichever way it is described
//!
//! Both operations hand out a [`RegionTag`], and [`Npt::region`] answers with
//! the name and extent of the region containing an address. That is how
//! something above these tables keeps a device per region without keeping a
//! second copy of where the region is — and there is a reason the two
//! descriptions share one set of names rather than only the trapped one having
//! them. A processor serving a sunk page itself still reports back the accesses
//! it declines to serve, and performing one of those means performing it
//! against whatever answers for the page. So which description a region is in
//! says how an access arrives and nothing about what answers for it, and the
//! name is what stays the same across the two.
//!
//! # Why the memory type is write-back everywhere
//!
//! Under nested paging the effective memory type of a guest access is the
//! guest's type combined with the nested one. A nested type of write-back
//! leaves the guest's choice in force for every type but write-combining, which
//! becomes write-combining-with-snooping — uncacheable either way, and coherent
//! as well. So write-back at this level is not a claim that the memory is
//! cacheable; it is the identity element that lets a guest mark its own device
//! apertures uncacheable and be obeyed.
//!
//! # Changing what an address means while a guest is running
//!
//! Filling only ever turns a not-present entry present, and the architecture
//! requires no invalidation for that — the walker detects a constraint being
//! removed on its own. Splitting replaces one entry with a table describing the
//! same memory the same way, which removes no constraint either. So answering a
//! fault owes nothing to anybody, on any processor.
//!
//! Every other operation here answers with what it did to what a processor may
//! already believe, as a [`Change`], and a [`Change::Tightened`] is discharged
//! by handing it to [`Npt::barrier`]. That is the whole of the contract: every
//! operation is legal at any time, whatever the guest is doing on whichever
//! processor, and the barrier is what stops every processor using what the
//! operation invalidated. It works by making a processor leave the guest and
//! making its next entry discard this guest's translations, which is why
//! [`Npt::before_entry`] and [`Npt::after_exit`] sit on the world switch.
//! [`coherence`] carries the ordering argument that makes it sound.
//!
//! Two things a barrier does not do, because a caller has to reason about both.
//! It does not stop an access already in flight, so memory taken back may not
//! be repurposed until the barrier has returned. And it does not reach
//! backwards, so a trap just armed has a bounded tail of accesses that were not
//! trapped.

#![no_std]

extern crate alloc;

pub mod coherence;
mod frames;
mod map;
mod tree;

use cpu::CpuIndex;
use paging::{DirectMap, Frames, chunk};
use processor::Features;
use spin::RwLock;
use svm::exit::NestedPageFault;
use thiserror::Error;
use vcpu::Vcpu;
use x86_64::PhysAddr;

use crate::{
    coherence::Coherence,
    frames::FrameCache,
    map::Map,
    tree::{Tree, Written},
};
pub use crate::{
    map::{Access, Answered, Kind, MapError, Range, RegionTag, Trap, Verdict},
    tree::walk::Level,
};

/// One guest's nested page tables.
///
/// Holds no allocator of its own: the frames its tables are built out of are
/// kept aside per processor, taken from the chunk in batches by whoever has the
/// chunk's allocator in hand or through the address space every processor
/// shares. The one lock is the map's, and it is inside here rather than around
/// the whole of it — describing a page is shared work, and only changing what
/// an address means is exclusive.
#[derive(Debug)]
pub struct Npt {
    /// What each of the guest's physical addresses means.
    map: RwLock<Map>,
    /// Where those meanings are written for the hardware to walk.
    tree: Tree,
    /// The frames the tables are built out of, a list per processor.
    frames: FrameCache,
    /// What makes a change to any of it safe while a guest is running on it.
    coherence: Coherence,
}

impl Npt {
    /// Builds an empty set of tables for a guest whose physical memory is the
    /// machine's, less the hypervisor's own.
    ///
    /// Empty is the correct starting point rather than a stub: with no entry
    /// present, the guest's first access to any address faults, and
    /// [`Npt::fault`] is what turns that into a translation.
    ///
    /// The processor doing this takes its own table frames here, out of the
    /// allocator it already holds, so describing the guest's memory during
    /// bring-up needs nothing further. Every other processor's list is filled
    /// the first time that processor describes anything.
    ///
    /// # Errors
    ///
    /// [`NptError::OutOfFrames`] if the chunk cannot spare the root table and
    /// the shared page of zeroes, [`NptError::Unreachable`] if the window does
    /// not reach a frame it just handed out, or [`NptError::ChunkGeometry`] if
    /// the reserved chunk is not 2 MiB aligned and a whole number of 2 MiB
    /// regions — the property that makes the shadow a whole page table at a
    /// time.
    pub fn create(frames: &mut Frames, window: DirectMap) -> Result<Self, NptError> {
        let base = frames.chunk_base().as_u64();
        let large = Level::Directory.span();
        if !base.is_multiple_of(large) || !chunk::CHUNK_SIZE.is_multiple_of(large) {
            return Err(NptError::ChunkGeometry {
                base,
                size: chunk::CHUNK_SIZE,
            });
        }
        let root = frame(frames, window)?;
        let zero = frame(frames, window)?;
        let roster = cpu::roster().ok();
        let cache = FrameCache::new(roster.map_or(1, cpu::Roster::count));
        cache.stock(frames);
        Ok(Self {
            map: RwLock::new(Map::new(
                Range::new(frames.chunk_base(), chunk::CHUNK_SIZE)?,
                processor::physical_address_bits(),
            )),
            tree: Tree::new(
                root,
                zero,
                window,
                processor::features().contains(Features::GIB_PAGES),
            ),
            frames: cache,
            coherence: Coherence::new(roster.map_or(&[][..], |roster| roster.entries())),
        })
    }

    /// The value a guest's control block names these tables by, in its nested
    /// table root field.
    #[must_use]
    pub const fn root(&self) -> PhysAddr {
        self.tree.root()
    }

    /// The window these tables are reached through, which is the same window
    /// whatever they translate to has to be reached through.
    ///
    /// Handed out so that anything reading a guest's memory uses the window
    /// this was built with rather than one it found for itself. Two windows
    /// onto physical memory would be two chances to disagree about what is
    /// reachable.
    #[must_use]
    pub const fn window(&self) -> DirectMap {
        self.tree.window()
    }

    /// Describes the region containing a guest physical address that had no
    /// translation.
    ///
    /// `cause` is the first exit-information field of the nested page fault,
    /// decoded. Only its direction is read: what a fault means here does not
    /// otherwise depend on why the guest was touching the address.
    ///
    /// One question is asked of the map and its answer serves twice — for what
    /// the address is, and for how much of memory around it is the same thing.
    ///
    /// Runs on every processor at once. The map is read rather than held
    /// exclusively, so faults do not serialize against each other, and the
    /// answer is acted on while that read is still held — an answer let go of
    /// first could be written into the tables after something had changed it.
    ///
    /// What the fill made of an entry is not read here. Describing a region the
    /// guest has just touched either fills an entry that said nothing or leaves
    /// one that already said this, and neither is anything another processor
    /// has to be told about.
    ///
    /// # Errors
    ///
    /// [`NptError::OutOfFrames`] if this processor has no frame left for a
    /// table, [`NptError::Unreachable`] if the window does not reach one, or
    /// [`NptError::Coarser`] if a larger page already covers the address, which
    /// means something described this region at a granularity the fill rule
    /// never produces.
    ///
    /// An address the machine does not have is not one of them: it is an
    /// [`Outcome::Unaddressable`] rather than a failure here, because nothing
    /// about these tables went wrong and what the caller owes the guest is the
    /// same kind of decision as for every other outcome.
    pub fn fault(&self, gpa: PhysAddr, cause: NestedPageFault) -> Result<Outcome, NptError> {
        // Before the map is read, because filling this processor's list reaches
        // for the address space the chunk's allocator lives in, and nothing here
        // may hold one lock while it takes another. Nothing is asked of it in the
        // steady state, a list that still holds frames being left alone.
        self.frames.replenish();
        let map = self.map.read();
        let verdict = map.resolve(gpa);
        match verdict.kind {
            // Before the tables are touched at all, because an address inside
            // one of these faults on purpose and describing it is exactly what
            // must not happen.
            Kind::Interposed { tag, .. } => Ok(Outcome::Interposed { tag }),
            // Nothing can describe an address the machine does not have, and
            // narrowing one into an address that exists would describe the
            // wrong page.
            Kind::Unaddressable => Ok(Outcome::Unaddressable),
            Kind::Ram { .. } | Kind::Sink { .. } => {
                self.tree.fill(&self.frames, verdict, gpa)?;
                Ok(Outcome::Filled)
            }
            // A write to either faults however it is described — no page behind
            // them can take one — so what the caller is owed is what the access
            // came to rather than whether anything was built. The fill is asked
            // for all the same: an entry that already says this is left alone,
            // and a page the guest has not touched before is described.
            Kind::Shadow | Kind::Exposed { .. } => {
                self.tree.fill(&self.frames, verdict, gpa)?;
                Ok(hypervisors(cause))
            }
        }
    }

    /// Where a guest physical address really is, or `None` if nothing describes
    /// it yet.
    ///
    /// This is how anything outside the guest reaches the guest's memory: the
    /// address a guest calls physical means nothing to the machine until these
    /// tables have said what it is.
    ///
    /// `None` is not a failure. A guest's memory is described as it is touched,
    /// so an address it has not touched has no translation and the answer is to
    /// describe it — which needs the frame allocator, and so belongs to whoever
    /// owns that rather than here.
    ///
    /// The map is read for the length of the walk without being asked anything.
    /// What that buys is the tree standing still: a mutation that gives a table
    /// back holds the map exclusively, so it cannot take one out from under
    /// this walk between one entry of it and the next.
    ///
    /// # Errors
    ///
    /// [`NptError::Unreachable`] if the window does not reach one of these
    /// tables, which is a broken window rather than anything about `gpa`.
    pub fn translate(&self, gpa: PhysAddr) -> Result<Option<Translation>, NptError> {
        let _held = self.map.read();
        self.tree.translate(gpa)
    }

    /// Discharges what a mutation made stricter, so that no processor can begin
    /// a guest access with a translation the mutation invalidated.
    ///
    /// The only way to be rid of a [`Change::Tightened`], and nothing at all
    /// for the other two: neither granting permission nor writing what was
    /// already written leaves a processor holding anything these tables
    /// have stopped justifying.
    ///
    /// What it guarantees, and the two things it deliberately does not, are
    /// [`coherence`]'s to state. The short of it is that a caller may repurpose
    /// memory once this has returned and not before.
    ///
    /// # Not while the address space is held
    ///
    /// A table a compaction gave back goes into this processor's own list of
    /// frames, and to the chunk where that list is full — which means the lock
    /// every processor's mapping work shares. So a barrier is taken with that
    /// lock let go, exactly as filling a list of frames is.
    ///
    /// # Errors
    ///
    /// [`NptError::BarrierIncomplete`] if a processor that had to leave the
    /// guest did not answer in the time it was given, or if there is no way
    /// to reach one at all on a machine where others are running. The
    /// mutation has already happened either way; what the caller is being
    /// told is that some processor may still be acting on what it replaced.
    pub fn barrier(&self, change: Change) -> Result<(), NptError> {
        let Change::Tightened { first, bytes } = change else {
            return Ok(());
        };
        self.coherence.barrier(first, bytes)?;
        // Only now, and only where the barrier finished: a table the tree gave up
        // could have been handed out for something else while a walk in progress
        // was still reading it.
        self.frames.restore();
        Ok(())
    }

    /// Publishes that this processor is entering the guest, and arms the
    /// discard of this guest's translations where a mutation has happened
    /// since its last entry.
    ///
    /// One relaxed store, one fence, one relaxed load and a comparison, on the
    /// world switch. That is the whole price of being able to change a guest's
    /// memory while it runs, and it is noise against the switch itself.
    pub fn before_entry(&self, vcpu: &mut Vcpu, who: CpuIndex) {
        if self.coherence.entering(who) {
            vcpu.flush();
        }
    }

    /// Publishes that this processor has left the guest, so that a barrier
    /// stops having to make it leave.
    pub fn after_exit(&self, who: CpuIndex) {
        self.coherence.left(who);
    }

    /// Marks a range as one whose accesses are not the hardware's to answer,
    /// and answers by what name.
    ///
    /// The range is described one page at a time, splitting whatever larger
    /// page covers it, because permissions belong to an entry and
    /// neighbouring pages must keep theirs. What the guest may still do for
    /// itself is [`Trap`]'s to say.
    ///
    /// The range is recorded as well as described, because describing it is not
    /// enough on its own: [`Npt::fault`] would otherwise fill a trapped page
    /// back in the first time the guest touched it, and would describe a 2 MiB
    /// region straight over one.
    ///
    /// The name is what something else keys whatever answers for the region by.
    /// It is handed out here rather than taken in because these tables are the
    /// one record of where a region is, so they are also the only place that
    /// can say which names are free.
    ///
    /// Answers with what it made stricter, which is [`Npt::barrier`]'s to
    /// discharge. A region the map already records by exactly this range and
    /// exactly this trap is [`Change::None`] — nothing was written, and a
    /// caller driving the description of a page back and forth pays nothing
    /// for the transition it is already in.
    ///
    /// # Errors
    ///
    /// [`NptError::Map`] if the map will not record the region — the range is
    /// not a whole number of pages on a page boundary the processor can
    /// address, something already describes part of it another way, or
    /// there is no room for another — [`NptError::OutOfFrames`] if the
    /// chunk cannot spare the tables it takes, [`NptError::Unreachable`] if the
    /// window does not reach one, or [`NptError::Coarser`] if a leaf turns up
    /// at a level the architecture has no large page at.
    pub fn protect(
        &self,
        frames: &mut Frames,
        gpa: PhysAddr,
        bytes: u64,
        trap: Trap,
    ) -> Result<(RegionTag, Change), NptError> {
        let range = Range::new(gpa, bytes)?;
        // The map exclusively, and held across the tree as well, so that a fault
        // on another processor sees either all of this or none of it: one that
        // read the map before the region was recorded and described a page after
        // it was would describe a page nothing may describe.
        let mut map = self.map.write();
        if let Some(tag) = map.interposed(range, trap) {
            return Ok((tag, Change::None));
        }
        // Recorded before it is described, so that a failure part-way through
        // leaves a region that still traps everything it should. The reverse
        // order would leave pages described as untouchable that nothing knows to
        // trap, which is a guest faulting for ever on an address the tables have
        // no answer for.
        let tag = map.interpose(range, trap)?;
        let described = range.pages().try_fold(Written::Same, |written, page| {
            self.interpose(&map, frames, page.base())
                .map(|entry| written.and(entry))
        });
        match described {
            Ok(written) => Ok((tag, owed(written, range))),
            // The region is not described and so must not go on being recorded:
            // a record naming pages that were never trapped would refuse to let
            // `fault` describe them, and the guest would fault on them for ever
            // with nothing to answer. The pages that *were* described are left
            // as they are — they trap, which is safe — and the caller undoing
            // this registration removes them. The region was recorded a moment
            // ago, so nothing can refuse to give it back, and what the caller
            // needs to hear about is the failure to describe it.
            Err(refused) => {
                let _ = map.release(range);
                Err(refused)
            }
        }
    }

    /// Stops trapping a region, leaving its pages to be described on demand
    /// again.
    ///
    /// The counterpart [`Npt::protect`] needs, for two situations that both
    /// otherwise leave the tables describing something nothing answers for: a
    /// registration that failed after the region was trapped, and a guest being
    /// taken apart.
    ///
    /// The entries are cleared rather than filled in. A trapped page is
    /// described as read-only or as nothing at all, and what it *should* be
    /// depends on what the map says of it once the region has gone — which is
    /// exactly the question [`Npt::fault`] already asks. So the pages are left
    /// with no translation, which is where every page of a guest starts, and
    /// the first access to one describes it correctly.
    ///
    /// This is also where the tree shrinks. A region that stops needing to be
    /// described a page at a time leaves the larger page containing it
    /// describable by one entry again, and the tables it no longer needs
    /// are handed back — which is what a [`Change::Tightened`] from here is
    /// for, however little the entries themselves changed. Untrapping
    /// grants permission rather than reducing it, and a processor still
    /// holding the trapping description merely faults once more than it has
    /// to; a table given back while a walk is still reading it is another
    /// matter entirely.
    ///
    /// # Errors
    ///
    /// [`NptError::Map`] if the range is not a whole number of pages on a page
    /// boundary or no region was recorded at exactly that range, or
    /// [`NptError::Unreachable`] if the window does not reach one of these
    /// tables.
    pub fn release(&self, gpa: PhysAddr, bytes: u64) -> Result<Change, NptError> {
        let range = Range::new(gpa, bytes)?;
        let mut map = self.map.write();
        // Forgotten first, so that a failure part-way through leaves pages that
        // `fault` is willing to describe rather than pages it refuses to touch
        // and nothing answers for.
        map.release(range)?;
        for page in range.pages() {
            self.tree.abandon(page.base())?;
        }
        Ok(Change::Loosened.and(self.coarsen(&map, range)?))
    }

    /// Maps immutable pages of the owned chunk for guest access.
    ///
    /// The regular owned-memory rule maps every chunk page to shared zeroes.
    /// This is the narrow exception for immutable entry code and its
    /// parameters, whose only job is to transfer from the captured firmware
    /// state into the preloaded guest. The range remains read-only, so it
    /// cannot become writable guest-controlled hypervisor memory.
    ///
    /// Answers with what it made stricter, which for a page nothing had
    /// described is nothing at all: filling an absent entry takes nothing
    /// away.
    ///
    /// # Errors
    ///
    /// [`NptError::OutsideOwned`] if the range leaves the chunk,
    /// [`NptError::Map`] if the map will not record it — the range is not a
    /// whole number of pages on a page boundary, something already describes
    /// part of it another way, or there is no room for another — or an error
    /// from building the required nested tables.
    pub fn expose(
        &self,
        frames: &mut Frames,
        gpa: PhysAddr,
        bytes: u64,
        exposure: Exposure,
    ) -> Result<Change, NptError> {
        let mut map = self.map.write();
        let range = owned(&map, gpa, bytes)?;
        map.expose(range, exposure.access())?;
        let written = range.pages().try_fold(Written::Same, |written, page| {
            self.describe_page(&map, frames, page.base())
                .map(|entry| written.and(entry))
        })?;
        Ok(owed(written, range))
    }

    /// Takes an exposed range back, leaving it as every other page of the
    /// hypervisor's memory already is: the shared page of zeroes, read-only.
    ///
    /// The counterpart of [`Npt::expose`], for entry code whose work is done.
    /// Afterwards the range is indistinguishable from the rest of the chunk — a
    /// guest reading it sees zeroes, and a guest writing it is reported as
    /// [`Outcome::Refused`] like any other write to hypervisor memory.
    ///
    /// # What it makes stricter
    ///
    /// Where the range was described, this replaces a page of the chunk the
    /// guest could read with the shared frame of zeroes, so a processor
    /// that entered this guest may hold a translation that no longer says
    /// what it says now. That is a [`Change::Tightened`], and
    /// [`Npt::barrier`] is what discharges it — a caller that means to
    /// reuse the pages must wait for it to return.
    ///
    /// # Where its tables come from
    ///
    /// From this processor's own list, the way a fault's do, and not from an
    /// allocator handed in — which is the one thing that distinguishes this
    /// from [`Npt::expose`] and it follows from when the two are called.
    /// Everything that describes a guest's memory before it has run does so
    /// with the address space still one function's value, so it has the
    /// chunk's allocator in hand; this runs afterwards, when reaching that
    /// allocator means the lock every processor shares, and the list is
    /// what keeps that lock from ever being wanted by a processor already
    /// holding one of these tables' own.
    ///
    /// # Errors
    ///
    /// [`NptError::OutsideOwned`] if the range leaves the chunk,
    /// [`NptError::Map`] if the range is not a whole number of pages on a page
    /// boundary or a page of it was not being shown to the guest, or an error
    /// from building the required nested tables.
    pub fn conceal(&self, gpa: PhysAddr, bytes: u64) -> Result<Change, NptError> {
        // Before the map is taken, for the reason `fault` does it there.
        self.frames.replenish();
        let mut map = self.map.write();
        let range = owned(&map, gpa, bytes)?;
        map.conceal(range)?;
        let written = range.pages().try_fold(Written::Same, |written, page| {
            self.fill_page(&map, page.base())
                .map(|entry| written.and(entry))
        })?;
        // Nothing to coarsen: this range is the hypervisor's own memory, which is
        // described one page at a time whatever else is true of it, so the region
        // holding it needed a table of its own before this and needs one still.
        Ok(owed(written, range))
    }

    /// Describes one page as a place the guest may touch without anything
    /// answering: reads see what the frame holds and writes are kept by it,
    /// but nothing reads it back. Answers by what name.
    ///
    /// The page gets a frame of its own rather than the shared page of zeroes
    /// the hypervisor's memory shadows onto, because that one is read-only for
    /// every address that shadows it at once — a writable sink has to be
    /// writable for the guest without becoming writable for anything else.
    ///
    /// The page is recorded as well as described, for the reason a trapped
    /// region is: [`Npt::fault`] would otherwise describe a large page straight
    /// over it the first time the guest touched a neighbour.
    ///
    /// It is named for the reason a trapped region is, too. A page given this
    /// way is one whose accesses a processor performs itself and reports back
    /// the ones it declines, and performing one of those means performing it
    /// against whatever answers for the page — which is found by name, the same
    /// name a trapped region would have been found by.
    ///
    /// Answers with what it made stricter, which for a page nothing described
    /// is nothing at all — filling an absent entry is the cheap direction,
    /// and it is the direction this goes in whenever the page was trapped a
    /// moment ago.
    ///
    /// # Errors
    ///
    /// [`NptError::Map`] if the map will not record the page — it is not page
    /// aligned or addressable, something already describes it another way, a
    /// page already sunk included, or there is no room for another — or an
    /// error from allocating or reaching the frame behind it.
    pub fn sink(
        &self,
        frames: &mut Frames,
        gpa: PhysAddr,
    ) -> Result<(RegionTag, Change), NptError> {
        // Refused before a frame is taken for it, so that a page the map would
        // not describe this way costs the chunk nothing.
        let page = Range::new(gpa, chunk::FRAME_SIZE)?;
        let frame = frame(frames, self.window())?;
        let mut map = self.map.write();
        // Recorded before it is described, because the map is what `fault`
        // consults: a page the map calls sunk is described that way whenever it
        // is next touched, while a page described as a sink that the map did not
        // record would be filled back over as ordinary memory.
        let tag = match map.sink(gpa, frame) {
            Ok(tag) => tag,
            // The frame was handed out for a page the map will not describe that
            // way, so it goes back rather than being held by nothing.
            Err(refused) => {
                frames::release(frames, frame);
                return Err(refused.into());
            }
        };
        let written = self.describe_page(&map, frames, gpa)?;
        Ok((tag, owed(written, page)))
    }

    /// Stops giving one page to the guest, leaving it with no translation and
    /// its frame to be handed back once a barrier has passed.
    ///
    /// The counterpart of [`Npt::sink`], for a page whose accesses stop being
    /// somebody else's to perform. The entry is cleared rather than replaced,
    /// for the reason [`Npt::release`] clears one: what the page *should* be
    /// once it is no longer sunk is the question [`Npt::fault`] already
    /// answers, and a page with no translation is where every page of a guest
    /// starts.
    ///
    /// Always owes a barrier, and the frame is what owes it rather than the
    /// entry. A processor that has entered this guest may hold a translation of
    /// the page to a frame this hands back, so the frame is held aside until a
    /// barrier has passed exactly as a table a compaction gave up is — and
    /// unlike a trapped region, there is no state this is already in for the
    /// transition to be free from, a page that is not sunk being refused
    /// outright.
    ///
    /// # Errors
    ///
    /// [`NptError::Map`] if the page is not page aligned or was not being sunk,
    /// or [`NptError::Unreachable`] if the window does not reach one of these
    /// tables.
    pub fn unsink(&self, gpa: PhysAddr) -> Result<Change, NptError> {
        let page = Range::new(gpa, chunk::FRAME_SIZE)?;
        let mut map = self.map.write();
        // Forgotten first, so that a failure part-way through leaves a page that
        // `fault` is willing to describe rather than one it describes as a sink
        // the map no longer holds a frame for.
        let frame = map.unsink(gpa)?;
        // Before the compaction below, which stops detaining tables once there is
        // nowhere left to keep one: the frame this page was given must be the
        // first thing to get a place, since handing it out again while a
        // processor still writes to it is the one failure that corrupts memory
        // rather than costing a table.
        self.frames.detain(frame);
        self.tree.abandon(gpa)?;
        Ok(Change::Tightened {
            first: page.base(),
            bytes: page.bytes(),
        }
        .and(self.coarsen(&map, page)?))
    }

    /// The region something other than the hardware answers for that `gpa` is
    /// in, or `None` if the hardware answers for it.
    ///
    /// How anything above these tables finds what answers for an address
    /// without keeping its own copy of where the region is. Both
    /// descriptions of a region answer: one whose accesses fault, and one
    /// given to the guest over a frame nothing reads because a processor
    /// performs them itself. Which of the two says how an access arrives,
    /// not what answers for it.
    #[must_use]
    pub fn region(&self, gpa: PhysAddr) -> Option<Answered> {
        self.map.read().region(gpa)
    }

    /// Logs the shape of the translation, which is the whole of what a guest's
    /// view of memory is.
    pub fn describe(&self, who: &str) {
        self.tree.describe(who);
        self.map.read().describe(who);
        self.frames.describe(who);
        self.coherence.describe(who);
    }

    /// Describes every region of `range` that has stopped needing to be written
    /// down finely with one entry again, and holds back the tables they no
    /// longer need until a barrier has passed.
    ///
    /// The only reason the tree ever shrinks. Under the map's write lock and
    /// never on the fault path: what decides whether a region may be
    /// described coarsely is the map, and the answer is only stable while
    /// nothing is changing it.
    ///
    /// Finest level first, because coarsening a 2 MiB region is what leaves the
    /// gigabyte holding it describable by one entry — the two levels in the
    /// other order would coarsen the smaller regions of a gigabyte that had
    /// just stopped being coarsenable.
    fn coarsen(&self, map: &Map, range: Range) -> Result<Change, NptError> {
        let mut change = Change::None;
        for level in [Level::Directory, Level::Pointer] {
            for base in range.aligned(level.span()) {
                // A frame the tree gives up may not be handed out again until a
                // barrier has passed, so a compaction with nowhere to keep one
                // stops instead of freeing it early. What it leaves behind is a
                // region described more finely than it has to be, which is
                // correct and costs a table.
                if !self.frames.detainable() {
                    return Ok(change);
                }
                if let Some(frame) = self.tree.compact(map.resolve(base), level, base)? {
                    self.frames.detain(frame);
                    change = change.and(Change::Tightened {
                        first: base,
                        bytes: level.span(),
                    });
                }
            }
        }
        Ok(change)
    }

    /// Describes one page of a trapped region as the map now says it is.
    ///
    /// A write-trapped page is described as the memory really there and
    /// read-only, so that reads reach the hardware without an exit. A page
    /// where every access is trapped is described as nothing at all: a
    /// present entry has no bit that denies a read, so not present is the
    /// only encoding that faults on one.
    fn interpose(
        &self,
        map: &Map,
        frames: &mut Frames,
        gpa: PhysAddr,
    ) -> Result<Written, NptError> {
        // Broken up first, because the page has to say something its neighbours
        // do not and permissions belong to an entry. This is the one place that
        // needs finer granularity than the fill rule would otherwise produce,
        // and it is why answering a fault never has to.
        self.frames.stock(frames);
        self.tree.split(&self.frames, gpa)?;
        self.describe_page(map, frames, gpa)
    }

    /// As [`Npt::fill_page`], with this processor's list brought up out of an
    /// allocator the caller holds rather than through the address space every
    /// processor shares.
    ///
    /// Which is what every operation handed one wants: a caller that has the
    /// chunk's allocator has it because the address space is still one
    /// function's value, and taking the lock instead would be taking it
    /// while the map is held.
    fn describe_page(
        &self,
        map: &Map,
        frames: &mut Frames,
        gpa: PhysAddr,
    ) -> Result<Written, NptError> {
        self.frames.stock(frames);
        self.fill_page(map, gpa)
    }

    /// Describes the one page at `gpa` as the map says it is, out of the frames
    /// this processor has already set aside.
    ///
    /// Every mutation here works a page at a time — a page shown to the guest,
    /// a page sunk, a page of a trapped region — so each of them asks the map
    /// afresh rather than saying for itself what it just recorded. Two accounts
    /// of one page could disagree; one cannot.
    fn fill_page(&self, map: &Map, gpa: PhysAddr) -> Result<Written, NptError> {
        let verdict = map.resolve(gpa);
        self.tree.fill(&self.frames, verdict, gpa)
    }
}

/// What a mutation over `range` owes, given what writing its entries came to.
///
/// The range travels only so that a barrier can say what it was for and
/// coalesce several of them; there is no instruction it could be an argument
/// to.
fn owed(written: Written, range: Range) -> Change {
    match written {
        Written::Same => Change::None,
        Written::Filled => Change::Loosened,
        Written::Replaced => Change::Tightened {
            first: range.base(),
            bytes: range.bytes(),
        },
    }
}

/// The range `bytes` from `gpa` names, refused unless every page of it is the
/// hypervisor's own memory.
///
/// One check for both directions of a guest-visible chunk mapping, because what
/// the two have to ask is identical and a range that could be exposed but not
/// concealed — or the reverse — would be a way for the two to disagree about
/// what a valid range is.
fn owned(map: &Map, gpa: PhysAddr, bytes: u64) -> Result<Range, NptError> {
    let range = Range::new(gpa, bytes)?;
    if map.chunk().covers(range) {
        Ok(range)
    } else {
        Err(NptError::OutsideOwned {
            gpa: gpa.as_u64(),
            bytes,
        })
    }
}

/// What a fault on a page of the hypervisor's own memory came to.
///
/// A read is satisfied by whatever the page was described as, so the guest
/// merely retries it. A write is not and never will be: neither the shared page
/// of zeroes nor the entry code the guest is shown has a page behind it that
/// may take one.
///
/// Which of the two answers a write is owed depends on what was writing. The
/// guest's own instruction is answered by being stepped over. A walk of the
/// guest's page tables is not: the processor was reading a table the guest put
/// in this memory and writing the bit that records the access, and stepping the
/// instruction leaves the walk unsatisfied — so the same instruction faults
/// again, for ever.
fn hypervisors(cause: NestedPageFault) -> Outcome {
    match (cause.write(), cause.page_table_walk()) {
        (false, _) => Outcome::Filled,
        (true, false) => Outcome::Refused,
        (true, true) => Outcome::WalkRefused,
    }
}

/// What became of a guest physical address that faulted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// A translation exists now, and the access will succeed when the guest
    /// retries it.
    Filled,
    /// The guest tried to write memory the hypervisor owns. Reads there see
    /// zeroes; a write cannot be satisfied, so resuming the guest unchanged
    /// re-executes it and faults again. The caller has to emulate the
    /// instruction, discard the write, and resume past it.
    Refused,
    /// A walk of the guest's own page tables tried to write memory the
    /// hypervisor owns, which is a write these tables can no more satisfy than
    /// any other — and one the guest cannot be stepped past, because what
    /// faulted is not the instruction but the translation it needs. The caller
    /// owes the guest the exception its own architecture answers an
    /// unsatisfiable translation with.
    WalkRefused,
    /// The address is inside a region the hardware does not answer for, named
    /// by the region's tag. Nothing was described and nothing will be: what the
    /// access means is the caller's to decide, and stepping the guest past it
    /// is the caller's to do.
    Interposed {
        /// The name whatever answers for the region is known by.
        tag: RegionTag,
    },
    /// Above the processor's physical address width, so not an address this
    /// machine has at all. Nothing describes it and nothing may: a guest that
    /// reached it has been given an answer no real machine would give, and
    /// folding it into an address that exists would answer for a different
    /// page.
    Unaddressable,
}

/// What a mutation did to what a processor may already believe.
///
/// Every operation that changes these tables answers with one, and the only way
/// to be rid of a [`Change::Tightened`] is to hand it to [`Npt::barrier`] — so
/// a caller cannot forget, and the compiler says so.
#[must_use]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Change {
    /// The tables already said this. Nothing was written and nothing is owed.
    ///
    /// A distinct answer rather than a successful no-op, because it is what
    /// keeps a description a guest can drive from sending interprocessor
    /// interrupts for a transition that did not happen.
    None,
    /// Something was written, and only to grant permission or to describe what
    /// nothing described. Nothing is owed: no processor can hold a translation
    /// these tables have stopped justifying, and the walker notices a
    /// constraint being lifted by itself.
    Loosened,
    /// Something a processor may hold is no longer justified. A barrier is owed
    /// before the memory behind it may be repurposed or the trap relied on.
    ///
    /// The range is *not* a hardware argument. There is no way to invalidate a
    /// nested translation by guest physical address — the instruction that
    /// invalidates by an alternate address space identifier takes a guest
    /// *virtual* address, and the manual says outright that it cannot do this —
    /// so the unit is the identifier, and this is what a barrier coalesces
    /// and what a barrier that could not finish names.
    Tightened {
        /// The first address made stricter.
        first: PhysAddr,
        /// How many bytes from there.
        bytes: u64,
    },
}

impl Change {
    /// The one change that answers for both, which is whichever owes more.
    ///
    /// What a mutation touching several regions comes to. Two tightenings
    /// become the run that spans both, which claims more than either did
    /// and never less — the direction an error here has to fall, and free,
    /// the range being for the record rather than for the hardware.
    pub fn and(self, other: Self) -> Self {
        match (self, other) {
            (
                Self::Tightened { first, bytes },
                Self::Tightened {
                    first: other,
                    bytes: others,
                },
            ) => {
                let base = first.as_u64().min(other.as_u64());
                let end = first
                    .as_u64()
                    .saturating_add(bytes)
                    .max(other.as_u64().saturating_add(others));
                Self::Tightened {
                    first: PhysAddr::new_truncate(base),
                    bytes: end - base,
                }
            }
            (tightened @ Self::Tightened { .. }, _) | (_, tightened @ Self::Tightened { .. }) => {
                tightened
            }
            (Self::Loosened, _) | (_, Self::Loosened) => Self::Loosened,
            (Self::None, Self::None) => Self::None,
        }
    }
}

/// Access permitted to a guest-visible page owned by the hypervisor.
///
/// Writable is intentionally not representable: exposed pages are a narrow
/// transfer mechanism, never guest-owned memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exposure {
    /// Readable and non-executable.
    ReadOnly,
    /// Readable and executable.
    ReadExecute,
}

impl Exposure {
    /// What a guest may do with a page shown this way.
    const fn access(self) -> Access {
        match self {
            Self::ReadOnly => Access::empty(),
            Self::ReadExecute => Access::EXECUTE,
        }
    }
}

/// Where a guest physical address really is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Translation {
    /// The system physical address it translates to.
    pub spa: PhysAddr,
    /// Whether the guest may write there.
    ///
    /// Clear for every page of the hypervisor's own memory, which all
    /// translates to one shared page of zeroes — so anything writing on a
    /// guest's behalf has to consult this rather than assume, or that one page
    /// stops being zeroes for every address that shadows it at once.
    pub writable: bool,
    /// Bytes from [`Translation::spa`] that the same entry describes.
    ///
    /// What makes a copy across a large page cost one translation instead of
    /// one per 4 KiB: the caller can move this many bytes before it has to ask
    /// again.
    pub span: u64,
}

/// Why a guest physical address could not be described.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum NptError {
    /// The chunk has no frame left for a table.
    #[error("the reserved chunk has no frame left for a nested page table")]
    OutOfFrames,
    /// The window onto physical memory does not reach a table.
    #[error("nested page table at physical {phys:#x} is outside the window onto physical memory")]
    Unreachable {
        /// The table that could not be reached.
        phys: u64,
    },
    /// A leaf describes the address at a coarser level than the map now allows,
    /// which no fill produces and no fault repairs.
    #[error(
        "guest physical {gpa:#x} is described by one entry spanning {:#x} bytes, coarser than the \
         memory map allows",
        level.span()
    )]
    Coarser {
        /// The address in question.
        gpa: u64,
        /// The level the leaf was found at.
        level: Level,
    },
    /// The reserved chunk is not aligned or sized as the shadow requires.
    #[error(
        "the reserved chunk at {base:#x}, {size:#x} bytes, is not a whole number of 2 MiB regions"
    )]
    ChunkGeometry {
        /// Where the chunk begins.
        base: u64,
        /// How long it is.
        size: u64,
    },
    /// A guest-visible chunk mapping was requested outside the chunk.
    #[error("guest exposure at {gpa:#x} of {bytes:#x} bytes leaves the hypervisor chunk")]
    OutsideOwned {
        /// Guest physical base requested.
        gpa: u64,
        /// Bytes requested.
        bytes: u64,
    },
    /// A processor inside the guest was not made to leave it, so it may still
    /// be acting on a translation the mutation invalidated.
    ///
    /// The mutation itself has happened. What has not is every processor being
    /// made to stop using what it replaced, which is why a caller repurposing
    /// memory may not go on and a caller arming a trap may not rely on it.
    #[error("{unanswered} processors did not leave the guest when they were asked to")]
    BarrierIncomplete {
        /// How many were asked and not confirmed to have left.
        ///
        /// The whole batch, because what crosses the slot to the subsystem that
        /// owns interprocessor interrupts is whether every one of them
        /// answered.
        unanswered: usize,
    },
    /// What an address means could not be recorded or answered for.
    #[error(transparent)]
    Map(#[from] MapError),
}

/// A zeroed frame of the chunk, and proof that the window reaches it.
///
/// Reaching it is checked here rather than at first use because a frame the
/// window cannot describe is a broken window, and finding that out while
/// building the tables beats finding it out inside a fault handler.
fn frame(frames: &mut Frames, window: DirectMap) -> Result<PhysAddr, NptError> {
    let frame = frames
        .allocate(0)
        .map_err(|_| NptError::OutOfFrames)?
        .start_address();
    if window.virt(frame).is_none() {
        return Err(NptError::Unreachable {
            phys: frame.as_u64(),
        });
    }
    Ok(frame)
}

const _: () = assert!(
    chunk::CHUNK_SIZE.is_multiple_of(Level::Directory.span()),
    "the shadow describes the chunk in whole 2 MiB regions",
);
const _: () = assert!(
    chunk::CHUNK_ALIGN.is_multiple_of(Level::Directory.span()),
    "a 2 MiB region overlapping the chunk must lie entirely inside it",
);
const _: () = assert!(
    chunk::FRAME_SIZE == Level::Page.span(),
    "the page a range is measured in must be the page an entry describes",
);
const _: () = {
    /// Refuses a type that more than one processor cannot share.
    const fn shared<T: Send + Sync>() {}
    // Derived, never asserted: what makes these tables safe to describe a page
    // through from every processor at once is that every field of them is, so a
    // field that stopped being one has to fail the build rather than be noticed.
    shared::<Npt>();
};

#[cfg(test)]
pub(crate) mod tests {
    //! The two ways one page of a guest can be described, what describing one
    //! that way costs its neighbours, what a description that stops being
    //! needed gives back, what two processors describing memory at the same
    //! moment cost between them, and a run of host memory standing in for
    //! the reserved chunk for the rest of the crate to build tables in.
    //!
    //! Only the chunk has to be real. Every table these tables build is a frame
    //! of it, reached through the window, and nothing here ever dereferences a
    //! guest physical address — so a run of memory with a window pointed at it
    //! is the whole of the machine this needs.
    //!
    //! Two host threads stand in for two processors, and they share one list of
    //! table frames because nothing has surveyed a roster here. That is the
    //! harder case rather than an easier one: on a machine each processor has a
    //! list of its own and nothing contends for one.
    //!
    //! A list is also all the frames a test has. Nothing here can refill one —
    //! the frames for that come through the address space every processor
    //! shares, and there is no address space in a test — so the racing tests
    //! describe as much memory as one list of frames can describe and no more.
    //!
    //! Nothing has a station here either, for the same reason: a processor
    //! publishes what it is doing at the position the roster gave it. So a
    //! barrier finds nobody inside the guest and answers for everybody, which
    //! is what a boot describing a guest's memory sees — and what a barrier
    //! does when it finds somebody is [`crate::coherence`]'s to establish.

    extern crate std;

    use alloc::alloc::{Layout, alloc_zeroed};
    use std::{sync::Barrier, vec::Vec};

    use paging::{DirectMap, Frames, chunk::FRAME_SIZE};
    use svm::exit::NestedPageFault;
    use x86_64::{PhysAddr, VirtAddr};

    use super::{
        Answered, Change, Exposure, MapError, Npt, NptError, Outcome, Range, Translation, Trap,
        chunk, hypervisors,
    };

    /// Where the interrupt controllers' register page is, which is the one page
    /// a boot chooses between these two descriptions for.
    const REGISTER_PAGE: u64 = 0xFEE0_0000;

    /// What one entry of the level above a page describes, which is the
    /// granularity ordinary memory is described in wherever nothing is in the
    /// way.
    const LARGE: u64 = 2 << 20;

    /// Where the machine's memory begins for the tests that race two threads at
    /// it: far above the run standing in for the chunk, so that nothing else is
    /// described anywhere near it.
    const RAM: u64 = 8 << 30;

    /// Regions of the hypervisor's own memory the racing threads describe.
    ///
    /// The hypervisor's own memory is described one page at a time whatever
    /// page sizes the processor reports, so each of these regions is a page
    /// table to be built and five hundred and twelve entries to write —
    /// which is what gives two threads long enough at it to arrive at one
    /// entry together. Twelve of them, so that the tables above them fit in
    /// one list of frames as well.
    const REGIONS: u64 = 12;

    /// Trees the racing test describes memory in.
    ///
    /// One round is a tree with nothing in it, because a race can only be run
    /// against tables that do not exist yet. Enough of them that a fill which
    /// stopped telling the loser of a race that it lost fails this rather than
    /// passing most of the time.
    const ROUNDS: usize = 64;

    #[test]
    fn a_trapped_page_has_no_translation_and_every_access_to_it_is_reported() {
        let (mut frames, window) = reserved();
        let npt = Npt::create(&mut frames, window).expect("tables over the chunk");
        let page = PhysAddr::new(REGISTER_PAGE);

        let named = npt
            .protect(&mut frames, page, FRAME_SIZE, Trap::Everything)
            .map(|(tag, change)| {
                discharge(&npt, change);
                tag
            })
            .expect("the page can be trapped");

        assert_eq!(
            npt.translate(page).expect("the tables can be walked"),
            None,
            "a page every access to which faults must have no translation at all"
        );
        for write in [false, true] {
            assert_eq!(
                npt.fault(page, fault(write))
                    .expect("the fault can be answered"),
                Outcome::Interposed { tag: named },
                "a {} of a trapped page belongs to whatever answers for it, by the \
                 name the tables gave the region",
                if write { "write" } else { "read" }
            );
        }
        assert_eq!(
            npt.translate(page).expect("the tables can be walked"),
            None,
            "answering the fault must not have described the page"
        );
    }

    #[test]
    fn a_sunk_page_translates_to_a_frame_of_its_own_the_guest_may_write() {
        let (mut frames, window) = reserved();
        let npt = Npt::create(&mut frames, window).expect("tables over the chunk");
        let page = PhysAddr::new(REGISTER_PAGE);

        npt.sink(&mut frames, page)
            .map(|(_, change)| discharge(&npt, change))
            .expect("the page can be sunk");

        let translation = npt
            .translate(page)
            .expect("the tables can be walked")
            .expect("a sunk page is described");
        assert!(
            translation.writable,
            "the acceleration requires the register page to be writable memory"
        );
        assert_ne!(
            translation.spa, page,
            "the sink is a frame of the chunk and never the hardware behind the page"
        );
        assert_eq!(
            npt.fault(page, fault(true))
                .expect("the fault can be answered"),
            Outcome::Filled,
            "a described page that faults anyway is described rather than reported"
        );
    }

    #[test]
    fn neither_description_performs_the_other() {
        let (mut frames, window) = reserved();
        let page = PhysAddr::new(REGISTER_PAGE);

        let trapped = Npt::create(&mut frames, window).expect("tables over the chunk");
        trapped
            .protect(&mut frames, page, FRAME_SIZE, Trap::Everything)
            .map(|(_, change)| discharge(&trapped, change))
            .expect("the page can be trapped");
        assert_eq!(
            trapped.sink(&mut frames, page),
            Err(NptError::Map(MapError::Overlaps {
                base: REGISTER_PAGE,
                other: REGISTER_PAGE,
            })),
            "trapping leaves the page undescribed, so sinking it would untrap it"
        );

        let sunk = Npt::create(&mut frames, window).expect("tables over the chunk");
        sunk.sink(&mut frames, page)
            .map(|(_, change)| discharge(&sunk, change))
            .expect("the page can be sunk");
        assert_eq!(
            sunk.release(page, FRAME_SIZE),
            Err(NptError::Map(MapError::NoRegion {
                base: REGISTER_PAGE,
                bytes: FRAME_SIZE,
            })),
            "sinking records no trapped region, so there is none to give back"
        );
    }

    #[test]
    fn either_description_of_a_page_gives_it_a_name_something_else_can_find_it_by() {
        let (mut frames, window) = reserved();
        let page = PhysAddr::new(REGISTER_PAGE);
        let whole = Range::new(page, FRAME_SIZE).expect("one page is a range");

        let trapped = Npt::create(&mut frames, window).expect("tables over the chunk");
        let named = trapped
            .protect(&mut frames, page, FRAME_SIZE, Trap::Everything)
            .map(|(tag, change)| {
                discharge(&trapped, change);
                tag
            })
            .expect("the page can be trapped");
        assert_eq!(
            trapped.region(page),
            Some(Answered {
                tag: named,
                range: whole,
            }),
            "a trapped region is found by the name the tables handed out"
        );

        let sunk = Npt::create(&mut frames, window).expect("tables over the chunk");
        let given = sunk
            .sink(&mut frames, page)
            .map(|(tag, change)| {
                discharge(&sunk, change);
                tag
            })
            .expect("the page can be sunk");
        assert_eq!(
            sunk.region(page),
            Some(Answered {
                tag: given,
                range: whole,
            }),
            "and so is a page given to the guest instead, because what answers for a \
             region does not depend on how its accesses arrive"
        );
        assert_eq!(
            sunk.region(page + FRAME_SIZE),
            None,
            "while the page beside it is the hardware's to answer for"
        );
    }

    #[test]
    fn what_a_write_to_the_hypervisors_own_memory_comes_to_tells_a_walk_from_an_access() {
        // The whole of the decision, as the table it is. A read is satisfied by
        // whatever the page was described as, so the guest merely retries it; a
        // write is not, and which answer it gets turns on what was writing —
        // the guest's own instruction, which can be stepped over, or a walk of
        // its page tables, which cannot, because what waits is the translation
        // rather than the instruction.
        for (write, walk, owed) in [
            (false, false, Outcome::Filled),
            (false, true, Outcome::Filled),
            (true, false, Outcome::Refused),
            (true, true, Outcome::WalkRefused),
        ] {
            assert_eq!(
                hypervisors(
                    NestedPageFault::new()
                        .with_write(write)
                        .with_page_table_walk(walk)
                ),
                owed,
                "write {write}, page table walk {walk}"
            );
            assert_eq!(
                hypervisors(
                    NestedPageFault::new()
                        .with_present(true)
                        .with_write(write)
                        .with_page_table_walk(walk)
                ),
                owed,
                "and whether a translation was present says nothing about it: a page \
                 of the hypervisor's own memory is present and readable, and it is the \
                 write that has nowhere to go"
            );
        }
    }

    #[test]
    fn every_kind_of_address_a_guest_can_fault_on_has_an_outcome_that_makes_progress() {
        let (mut frames, window) = reserved();
        let npt = Npt::create(&mut frames, window).expect("tables over the chunk");
        // One page of the hypervisor's own memory shown to the guest on purpose,
        // and one of the register page's two descriptions each, so that every
        // arm of the answer is reached over an address really described that way.
        let shown = PhysAddr::new(LARGE + 3 * FRAME_SIZE);
        npt.expose(&mut frames, shown, FRAME_SIZE, Exposure::ReadOnly)
            .map(|change| discharge(&npt, change))
            .expect("a page of the chunk can be shown to the guest");
        let sunk = PhysAddr::new(REGISTER_PAGE);
        npt.sink(&mut frames, sunk)
            .map(|(_, change)| discharge(&npt, change))
            .expect("the register page can be sunk");
        let trapped = PhysAddr::new(REGISTER_PAGE + LARGE);
        let named = npt
            .protect(&mut frames, trapped, FRAME_SIZE, Trap::Everything)
            .map(|(tag, change)| {
                discharge(&npt, change);
                tag
            })
            .expect("a device aperture can be taken over");

        for (gpa, what, reads, writes, walks) in [
            (
                PhysAddr::new(RAM),
                "ordinary memory",
                Outcome::Filled,
                Outcome::Filled,
                Outcome::Filled,
            ),
            (
                PhysAddr::new(LARGE),
                "the hypervisor's own memory",
                Outcome::Filled,
                Outcome::Refused,
                Outcome::WalkRefused,
            ),
            (
                shown,
                "a page of it the guest is shown",
                Outcome::Filled,
                Outcome::Refused,
                Outcome::WalkRefused,
            ),
            (
                sunk,
                "a page given to the guest over a frame nothing reads",
                Outcome::Filled,
                Outcome::Filled,
                Outcome::Filled,
            ),
            (
                trapped,
                "a region this hypervisor answers for",
                Outcome::Interposed { tag: named },
                Outcome::Interposed { tag: named },
                Outcome::Interposed { tag: named },
            ),
        ] {
            for (cause, owed, direction) in [
                (fault(false), reads, "a read"),
                (fault(true), writes, "a write"),
                (walk(), walks, "a walk of the guest's own page tables"),
            ] {
                assert_eq!(
                    npt.fault(gpa, cause).expect("the fault can be answered"),
                    owed,
                    "{direction} of {what} at {gpa:#x}"
                );
            }
        }
    }

    #[test]
    fn giving_a_sunk_page_up_holds_its_frame_back_until_a_barrier_has_passed() {
        let (mut frames, window) = reserved();
        let npt = Npt::create(&mut frames, window).expect("tables over the chunk");
        let page = PhysAddr::new(REGISTER_PAGE);
        npt.sink(&mut frames, page)
            .map(|(_, change)| discharge(&npt, change))
            .expect("the page can be sunk");
        assert_ne!(
            translated(&npt, page).spa,
            page,
            "a sunk page translates to a frame of the chunk rather than to itself"
        );

        let change = npt.unsink(page).expect("the page can be given up");

        assert!(
            matches!(
                change,
                Change::Tightened { first, bytes }
                    if first.as_u64() <= REGISTER_PAGE
                        && REGISTER_PAGE + FRAME_SIZE <= first.as_u64() + bytes
            ),
            "a processor may hold a translation to the frame this hands back, and \
             there is no state a page that was not sunk could be coming from — the \
             run is wider than the page because the region around it stopped \
             needing to be written down finely at the same moment"
        );
        let detained = npt.frames.detained();
        assert!(
            detained > 0,
            "so the frame waits for the barrier rather than being handed out again"
        );
        // Room for them to come back to, as the compaction test needs: a barrier
        // hands detained frames to this processor's list, and reaches the chunk
        // only for what the list will not hold.
        for _ in 0..detained {
            npt.frames
                .take()
                .expect("a stocked list has frames to spare");
        }
        let held = npt.frames.held();

        discharge(&npt, change);

        assert_eq!(
            (npt.frames.held(), npt.frames.detained()),
            (held + detained, 0),
            "and the barrier is what hands it back"
        );
        assert_eq!(
            translated(&npt, page).spa,
            page,
            "after which the page is the machine's own memory at the same address"
        );
        assert_eq!(
            npt.unsink(page),
            Err(NptError::Map(MapError::NoRegion {
                base: REGISTER_PAGE,
                bytes: FRAME_SIZE,
            })),
            "with nothing left to give up and no name still held for it"
        );
        assert_eq!(npt.region(page), None);
    }

    #[test]
    fn a_trapped_page_narrows_its_own_two_megabytes_and_nothing_further() {
        let (mut frames, window) = reserved();
        let npt = Npt::create(&mut frames, window).expect("tables over the chunk");
        // A page inside a 2 MiB region of ordinary memory, so that the region
        // holding it has to be described more finely than the fill rule would
        // otherwise choose and the regions around it do not.
        let trapped = PhysAddr::new(REGISTER_PAGE + FRAME_SIZE);

        npt.protect(&mut frames, trapped, FRAME_SIZE, Trap::Everything)
            .map(|(_, change)| discharge(&npt, change))
            .expect("the page can be trapped");

        for (gpa, span) in [
            (PhysAddr::new(REGISTER_PAGE), FRAME_SIZE),
            (PhysAddr::new(REGISTER_PAGE + LARGE), LARGE),
        ] {
            assert_eq!(
                npt.fault(gpa, fault(false))
                    .expect("the fault can be answered"),
                Outcome::Filled,
                "ordinary memory around a trapped page is still the guest's"
            );
            let translation = npt
                .translate(gpa)
                .expect("the tables can be walked")
                .expect("the address is described");
            assert_eq!(
                translation.spa, gpa,
                "and is the machine's own memory at the same address"
            );
            assert_eq!(
                translation.span, span,
                "how far one entry reaches from {gpa:#x}"
            );
        }
        assert_eq!(
            npt.translate(trapped).expect("the tables can be walked"),
            None,
            "while the trapped page itself is described by neither of them"
        );
    }

    #[test]
    fn describing_a_region_the_map_already_holds_writes_nothing_and_owes_nothing() {
        let (mut frames, window) = reserved();
        let npt = Npt::create(&mut frames, window).expect("tables over the chunk");
        let page = PhysAddr::new(REGISTER_PAGE);
        let named = npt
            .protect(&mut frames, page, FRAME_SIZE, Trap::Everything)
            .map(|(tag, change)| {
                discharge(&npt, change);
                tag
            })
            .expect("the page can be trapped");

        assert_eq!(
            npt.protect(&mut frames, page, FRAME_SIZE, Trap::Everything),
            Ok((named, Change::None)),
            "a description the map already holds is a transition that did not \
             happen, and nothing is told about one — and it answers with the name \
             the region already had"
        );
    }

    #[test]
    fn trapping_a_page_the_guest_has_had_described_owes_a_barrier_over_that_page() {
        let (mut frames, window) = reserved();
        let npt = Npt::create(&mut frames, window).expect("tables over the chunk");
        let page = PhysAddr::new(REGISTER_PAGE);
        npt.fault(page, fault(false))
            .expect("the fault can be answered");

        let (_, change) = npt
            .protect(&mut frames, page, FRAME_SIZE, Trap::Everything)
            .expect("the page can be trapped while a guest is running");

        assert_eq!(
            change,
            Change::Tightened {
                first: page,
                bytes: FRAME_SIZE,
            },
            "an entry that described the hardware and now describes nothing is one \
             a processor may still be acting on"
        );
        discharge(&npt, change);
    }

    #[test]
    fn describing_a_page_nothing_described_owes_nothing() {
        let (mut frames, window) = reserved();
        let npt = Npt::create(&mut frames, window).expect("tables over the chunk");

        let (_, change) = npt
            .sink(&mut frames, PhysAddr::new(REGISTER_PAGE))
            .expect("the page can be sunk");

        assert_eq!(
            change,
            Change::Loosened,
            "filling an absent entry takes nothing away, and a not-present entry \
             left nothing cached that could disagree"
        );
        discharge(&npt, change);
    }

    #[test]
    fn a_barrier_is_owed_for_a_tightening_and_for_nothing_else() {
        let (mut frames, window) = reserved();
        let npt = Npt::create(&mut frames, window).expect("tables over the chunk");
        let quiet = npt.coherence.epoch();

        for change in [Change::None, Change::Loosened] {
            npt.barrier(change).expect("neither owes anything");
        }

        assert_eq!(
            npt.coherence.epoch(),
            quiet,
            "neither is a change any processor has to be told about, so neither \
             advances what an entry compares against"
        );
        npt.barrier(Change::Tightened {
            first: PhysAddr::new(REGISTER_PAGE),
            bytes: FRAME_SIZE,
        })
        .expect("a barrier with nobody inside the guest has nobody to make leave");
        assert_ne!(
            npt.coherence.epoch(),
            quiet,
            "while a tightening is exactly what one is for"
        );
    }

    #[test]
    fn a_region_that_stops_needing_pages_is_described_by_one_entry_again() {
        let (mut frames, window) = reserved();
        let npt = Npt::create(&mut frames, window).expect("tables over the chunk");
        // A page inside a 2 MiB region of ordinary memory, so that trapping it has
        // to break up whatever describes the region — and giving it back leaves
        // that region uniform again.
        let page = PhysAddr::new(RAM + LARGE + 8 * FRAME_SIZE);
        let probes = [
            PhysAddr::new(RAM + LARGE),
            page,
            PhysAddr::new(RAM + 2 * LARGE - FRAME_SIZE),
        ];
        npt.fault(page, fault(false))
            .expect("the fault can be answered");
        let before = probes.map(|gpa| translated(&npt, gpa));
        npt.protect(&mut frames, page, FRAME_SIZE, Trap::Everything)
            .map(|(_, change)| discharge(&npt, change))
            .expect("the page can be trapped");

        let change = npt
            .release(page, FRAME_SIZE)
            .expect("the region can be given back");

        let detained = npt.frames.detained();
        assert!(
            detained > 0,
            "a region that has stopped needing to be written down finely gives up \
             the tables that wrote it down that way"
        );
        // Room for them to come back to: a fill takes frames out of this list and
        // a barrier puts them back into it, reaching the chunk only for what the
        // list will not hold — and there is no address space to reach a chunk
        // through here.
        for _ in 0..detained {
            npt.frames
                .take()
                .expect("a stocked list has frames to spare");
        }
        let held = npt.frames.held();

        discharge(&npt, change);

        assert_eq!(
            (npt.frames.held(), npt.frames.detained()),
            (held + detained, 0),
            "and the barrier is what hands them back, never the compaction itself: \
             a processor caches the entries above a leaf as well"
        );
        for (gpa, was) in probes.into_iter().zip(before) {
            assert_eq!(
                translated(&npt, gpa),
                was,
                "{gpa:#x} means exactly what it meant before the page was ever trapped"
            );
        }
    }

    #[test]
    fn two_processors_describing_memory_at_once_build_one_set_of_tables() {
        let (mut frames, window) = reserved();
        // The same addresses, which race on the entry describing a page, and
        // addresses one page apart, which race on the table above them.
        for apart in [0, FRAME_SIZE] {
            let (mine, theirs) = (described(0), described(apart));
            let cost = alone(&mut frames, window, &[&mine, &theirs]);
            let trees = (0..ROUNDS)
                .map(|_| Npt::create(&mut frames, window).expect("tables over the chunk"))
                .collect::<Vec<_>>();
            let held = trees
                .iter()
                .map(|npt| npt.frames.held())
                .collect::<Vec<_>>();
            let (raced, together) = (&trees, &Barrier::new(2));

            std::thread::scope(|threads| {
                for addresses in [&mine, &theirs] {
                    threads.spawn(move || {
                        for npt in raced {
                            // Both threads start each round together, so that
                            // the two are inside one tree at the same moment
                            // rather than one of them having finished before the
                            // other began.
                            together.wait();
                            for at in addresses {
                                npt.fault(*at, fault(false))
                                    .expect("a fault answered while another is being answered");
                            }
                        }
                    });
                }
            });

            for (npt, held) in trees.iter().zip(held) {
                assert_eq!(
                    held - npt.frames.held(),
                    cost,
                    "two processors describing addresses {apart:#x} apart must build \
                     the tables one of them would have built alone, the loser of \
                     every race putting its frame back"
                );
                // Every page of the hypervisor's own memory reads as one frame,
                // and which frame that is belongs to the tree rather than to the
                // page — so the first of them says what all of them must say.
                let zero = behind(npt, PhysAddr::new(0));
                for at in mine.iter().chain(&theirs) {
                    let translation = npt
                        .translate(*at)
                        .expect("the tables can be walked")
                        .expect("the address is described");
                    let expected = if at.as_u64() < chunk::CHUNK_SIZE {
                        (zero, false)
                    } else {
                        (*at, true)
                    };
                    assert_eq!(
                        (translation.spa, translation.writable),
                        expected,
                        "and {at:#x} means what one processor describing it alone \
                         would have meant by it"
                    );
                }
            }
        }
    }

    #[test]
    fn describing_a_page_a_second_time_costs_no_frame_at_all() {
        let (mut frames, window) = reserved();
        let npt = Npt::create(&mut frames, window).expect("tables over the chunk");
        let gpa = PhysAddr::new(RAM);
        npt.fault(gpa, fault(false))
            .expect("the fault can be answered");
        let (held, free) = (npt.frames.held(), frames.free());

        npt.fault(gpa, fault(false))
            .expect("and answering it again is allowed");

        assert_eq!(
            (npt.frames.held(), frames.free()),
            (held, free),
            "the steady state takes no frame from this processor's list and \
             nothing at all from the chunk"
        );
    }

    #[test]
    fn a_region_taken_over_while_a_guest_faults_is_never_described_as_memory() {
        let (mut frames, window) = reserved();
        let npt = Npt::create(&mut frames, window).expect("tables over the chunk");
        // A page in the middle of a 2 MiB region of ordinary memory, so that
        // taking it over has to break up whatever describes the region and the
        // faults racing it are ones that would otherwise describe all of it.
        let inside = PhysAddr::new(RAM + LARGE + 8 * FRAME_SIZE);
        let together = Barrier::new(2);

        std::thread::scope(|threads| {
            threads.spawn(|| {
                together.wait();
                for page in 0..LARGE / FRAME_SIZE {
                    let at = PhysAddr::new(RAM + LARGE + page * FRAME_SIZE);
                    npt.fault(at, fault(false))
                        .expect("a fault answered while a region is taken over");
                    let described = npt.translate(inside).expect("the tables can be walked");
                    assert!(
                        described.is_none_or(|translation| translation.spa == inside),
                        "a page being taken over is either the memory behind it or \
                         nothing at all, and never something else"
                    );
                }
            });
            together.wait();
            let named = npt
                .protect(&mut frames, inside, FRAME_SIZE, Trap::Everything)
                .map(|(tag, change)| {
                    discharge(&npt, change);
                    tag
                })
                .expect("the page can be taken over while another processor faults");
            assert_eq!(
                npt.fault(inside, fault(false))
                    .expect("the fault can be answered"),
                Outcome::Interposed { tag: named },
                "and a fault on it names the region it was taken over as"
            );
        });

        assert_eq!(
            npt.translate(inside).expect("the tables can be walked"),
            None,
            "a page every access to which faults must end up described by nothing"
        );
    }

    /// The addresses one racing thread describes: a page of ordinary memory
    /// with nothing near it, and one page in each of the first [`REGIONS`]
    /// regions of the hypervisor's own memory, all offset by `apart`.
    fn described(apart: u64) -> Vec<PhysAddr> {
        core::iter::once(RAM)
            .chain((0..REGIONS).map(|region| region * LARGE))
            .map(|base| PhysAddr::new(base + apart))
            .collect()
    }

    /// What describing every one of those addresses costs on a tree of its own,
    /// with nothing racing it.
    ///
    /// The yardstick the racing test measures against, so that what a fill
    /// costs is not written out here: it depends on the page sizes the
    /// processor running the test reports.
    fn alone(frames: &mut Frames, window: DirectMap, threads: &[&[PhysAddr]]) -> usize {
        let npt = Npt::create(frames, window).expect("tables over the chunk");
        let held = npt.frames.held();
        for at in threads.iter().flat_map(|addresses| addresses.iter()) {
            npt.fault(*at, fault(false))
                .expect("the fault can be answered");
        }
        held - npt.frames.held()
    }

    /// Where an address one of the racing trees describes really is.
    fn behind(npt: &Npt, gpa: PhysAddr) -> PhysAddr {
        translated(npt, gpa).spa
    }

    /// What the tables say about an address they must already describe.
    fn translated(npt: &Npt, gpa: PhysAddr) -> Translation {
        npt.translate(gpa)
            .expect("the tables can be walked")
            .expect("the address is described")
    }

    /// Discharges what a mutation reported, the way every caller of one does.
    fn discharge(npt: &Npt, change: Change) {
        npt.barrier(change)
            .expect("a barrier with nobody inside the guest has nobody to make leave");
    }

    /// A nested page fault of the direction alone, which is all [`Npt::fault`]
    /// reads of one for an access the guest itself made.
    fn fault(write: bool) -> NestedPageFault {
        NestedPageFault::new()
            .with_write(write)
            .with_final_address(true)
    }

    /// A nested page fault of a walk of the guest's own page tables, which is
    /// the one access whose direction is not the guest instruction's.
    ///
    /// Always a write and never against the final address: the processor is
    /// reading one of the guest's tables and recording that it did, and the
    /// address the guest was after has not been arrived at.
    fn walk() -> NestedPageFault {
        NestedPageFault::new()
            .with_write(true)
            .with_page_table_walk(true)
    }

    /// An allocator over a run of host memory standing in for the reserved
    /// chunk, and the window that reaches it.
    ///
    /// Physical zero is the run's first byte, so a frame the allocator hands
    /// out is an address inside the run and every table is written where a
    /// table really would be. Leaked deliberately: the run stands in for
    /// memory firmware reserved and nothing ever frees, and a window is a
    /// raw pointer with no lifetime attached — a run that could be dropped
    /// while a window still named it would differ from the machine in the
    /// direction that hides mistakes.
    pub(crate) fn reserved() -> (Frames, DirectMap) {
        let size = usize::try_from(chunk::CHUNK_SIZE).expect("a test chunk fits a host pointer");
        let align =
            usize::try_from(chunk::CHUNK_ALIGN).expect("a chunk's alignment fits a host pointer");
        let layout = Layout::from_size_align(size, align).expect("the chunk describes a layout");
        // SAFETY: the layout is of a non-zero size, which is the whole of what
        // this asks of a caller.
        let base = unsafe { alloc_zeroed(layout) };
        assert!(!base.is_null(), "the test chunk could not be allocated");
        let window = DirectMap::new(VirtAddr::from_ptr(base), chunk::CHUNK_SIZE)
            .expect("a window over the test chunk");
        // SAFETY: the run was just allocated, is a whole chunk long and chunk
        // aligned, is reached through a window that begins at its first byte,
        // and is never freed or handed to anything else — which is what the
        // reservation this stands in for guarantees on a machine.
        let frames = unsafe { Frames::create(PhysAddr::new(0), window) }
            .expect("an allocator over the test chunk");
        (frames, window)
    }
}
