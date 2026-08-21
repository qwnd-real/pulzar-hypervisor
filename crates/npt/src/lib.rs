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
//! run rather than probing for it. What is left here is the handle the two are
//! reached through and the operations that need both.
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
//! [`Npt::fault`] reports [`Resolution::Shadowed`], which is as far as these
//! tables can take it: there is no page to accept the write and there never
//! will be. Resuming the guest unchanged re-executes the instruction and faults
//! again, so the caller has to step over it instead — emulate the instruction,
//! discard the write, and resume past it. That is not something the tables can
//! do, and it is stated here so that a live-lock is not diagnosed as a bug in
//! them.
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
//! [`Resolution::Trapped`] for an address inside one rather than describing it.
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
//! # Nothing here invalidates a translation
//!
//! Filling only ever turns a not-present entry present, and the architecture
//! requires no invalidation for that — the walker detects a constraint being
//! removed on its own. Splitting replaces one entry with a table describing the
//! same memory the same way, which removes no constraint either.
//!
//! Three operations here change what a processor may already have cached, and
//! none of them flushes anything itself. [`Npt::protect`] takes permission away
//! and [`Npt::sink`] moves where a page translates to; both may only be called
//! before the guest has ever run, so nothing has walked these tables at that
//! point and no processor holds a translation they have stopped justifying.
//! [`Npt::conceal`] is the one that may be called while a guest is running, and
//! it states in as many words that discarding what the guest cached is the
//! caller's — the caller is what knows which processors ran the guest and what
//! makes the next entry, and `TLB_CONTROL` is a field of a control block this
//! crate does not have.

#![no_std]

#[cfg(test)]
extern crate alloc;

mod map;
mod tree;

use log::error;
use paging::{DirectMap, Frames, chunk};
use processor::Features;
use svm::exit::NestedPageFault;
use thiserror::Error;
use x86_64::{PhysAddr, structures::paging::PhysFrame};

use crate::{map::Map, tree::Tree};
pub use crate::{
    map::{Access, Kind, MapError, Range, RegionTag, Trap, Verdict},
    tree::walk::Level,
};

/// One guest's nested page tables.
///
/// Holds no allocator and no lock. Frames are handed in per call by whoever
/// owns the chunk's allocator, and exclusion is the caller's — which keeps this
/// a description of a translation rather than a second owner of the machine's
/// memory.
#[derive(Debug)]
pub struct Npt {
    /// What each of the guest's physical addresses means.
    map: Map,
    /// Where those meanings are written for the hardware to walk.
    tree: Tree,
}

impl Npt {
    /// Builds an empty set of tables for a guest whose physical memory is the
    /// machine's, less the hypervisor's own.
    ///
    /// Empty is the correct starting point rather than a stub: with no entry
    /// present, the guest's first access to any address faults, and
    /// [`Npt::fault`] is what turns that into a translation.
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
        Ok(Self {
            map: Map::new(
                Range::new(frames.chunk_base(), chunk::CHUNK_SIZE)?,
                processor::physical_address_bits(),
            ),
            tree: Tree::new(
                root,
                zero,
                window,
                processor::features().contains(Features::GIB_PAGES),
            ),
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
    /// # Errors
    ///
    /// [`NptError::Map`] carrying [`MapError::Unaddressable`] if the address is
    /// above the processor's physical address width, [`NptError::OutOfFrames`]
    /// if the chunk cannot spare a table, [`NptError::Unreachable`] if the
    /// window does not reach one, or [`NptError::Coarser`] if a larger page
    /// already covers the address, which means something described this region
    /// at a granularity the fill rule never produces.
    pub fn fault(
        &mut self,
        frames: &mut Frames,
        gpa: PhysAddr,
        cause: NestedPageFault,
    ) -> Result<Resolution, NptError> {
        let verdict = self.map.resolve(gpa);
        match verdict.kind {
            // Before the tables are touched at all, because an address inside
            // one of these faults on purpose and describing it is exactly what
            // must not happen.
            Kind::Interposed { .. } => Ok(Resolution::Trapped),
            // Nothing can describe an address the machine does not have, and
            // narrowing one into an address that exists would describe the
            // wrong page.
            Kind::Unaddressable => Err(MapError::Unaddressable { gpa: gpa.as_u64() }.into()),
            Kind::Ram { .. } | Kind::Sink { .. } => {
                self.tree.fill(frames, verdict, gpa)?;
                Ok(Resolution::Mapped)
            }
            // A write to either faults however it is described — no page behind
            // them can take one — so what the caller is owed is what the access
            // came to rather than whether anything was built. The fill is asked
            // for all the same: an entry that already says this is left alone,
            // and a page the guest has not touched before is described.
            Kind::Shadow | Kind::Exposed { .. } => {
                self.tree.fill(frames, verdict, gpa)?;
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
    /// # Errors
    ///
    /// [`NptError::Unreachable`] if the window does not reach one of these
    /// tables, which is a broken window rather than anything about `gpa`.
    pub fn translate(&self, gpa: PhysAddr) -> Result<Option<Translation>, NptError> {
        self.tree.translate(gpa)
    }

    /// Marks a range as one whose accesses are not the hardware's to answer.
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
    /// # Only before a guest has run
    ///
    /// This reduces what an entry permits, and it may only be called before
    /// these tables have ever been entered. Every processor's cached
    /// translations would otherwise have to be discarded, and nothing here does
    /// that — precisely because nothing can have cached one yet. Trapping a
    /// region while a guest is running would need that machinery first, and
    /// would need it before this call rather than after.
    ///
    /// # Errors
    ///
    /// [`NptError::Map`] if the map will not record the region — the range is
    /// not a whole number of pages on a page boundary the processor can
    /// address, something already describes part of it another way, or
    /// there is no room for another — [`NptError::OutOfFrames`] if the
    /// chunk cannot spare a table, [`NptError::Unreachable`] if the window
    /// does not reach one, or [`NptError::Coarser`] if a leaf turns up at a
    /// level the architecture has no large page at.
    pub fn protect(
        &mut self,
        frames: &mut Frames,
        gpa: PhysAddr,
        bytes: u64,
        trap: Trap,
    ) -> Result<(), NptError> {
        let range = Range::new(gpa, bytes)?;
        // Recorded before it is described, so that a failure part-way through
        // leaves a region that still traps everything it should. The reverse
        // order would leave pages described as untouchable that nothing knows to
        // trap, which is a guest faulting for ever on an address the tables have
        // no answer for.
        self.map.interpose(range, trap)?;
        let described = range
            .pages()
            .try_for_each(|page| self.interpose(frames, page.base()));
        if described.is_err() {
            // The region is not described and so must not go on being recorded:
            // a record naming pages that were never trapped would refuse to let
            // `fault` describe them, and the guest would fault on them for ever
            // with nothing to answer. The pages that *were* described are left
            // as they are — they trap, which is safe — and the caller undoing
            // this registration removes them. The region was recorded a moment
            // ago, so nothing can refuse to give it back, and what the caller
            // needs to hear about is the failure to describe it.
            let _ = self.map.release(range);
        }
        described
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
    /// # The caller discards what the guest cached
    ///
    /// This grants permission rather than reducing it, so no processor can hold
    /// a translation that these tables no longer justify — a page with no
    /// entry had nothing to cache. A processor may hold the *trapping*
    /// description, which is more restrictive than what replaces it, so the
    /// guest merely faults once more than it needs to and is then
    /// described. Removing a region while a guest runs is still the
    /// caller's to make safe, which is why nothing here may be called at
    /// that point.
    ///
    /// # Errors
    ///
    /// [`NptError::Map`] if the range is not a whole number of pages on a page
    /// boundary or no region was recorded at exactly that range, or
    /// [`NptError::Unreachable`] if the window does not reach one of these
    /// tables.
    pub fn release(&mut self, gpa: PhysAddr, bytes: u64) -> Result<(), NptError> {
        let range = Range::new(gpa, bytes)?;
        // Forgotten first, so that a failure part-way through leaves pages that
        // `fault` is willing to describe rather than pages it refuses to touch
        // and nothing answers for.
        self.map.release(range)?;
        range
            .pages()
            .try_for_each(|page| self.tree.abandon(page.base()))
    }

    /// Maps immutable pages of the owned chunk for guest access before first
    /// entry.
    ///
    /// The regular owned-memory rule maps every chunk page to shared zeroes.
    /// This is the narrow exception for immutable entry code and its
    /// parameters, whose only job is to transfer from the captured firmware
    /// state into the preloaded guest. The range remains read-only, so it
    /// cannot become writable guest-controlled hypervisor memory.
    ///
    /// # Errors
    ///
    /// [`NptError::OutsideOwned`] if the range leaves the chunk,
    /// [`NptError::Map`] if the map will not record it — the range is not a
    /// whole number of pages on a page boundary, something already describes
    /// part of it another way, or there is no room for another — or an error
    /// from building the required nested tables.
    pub fn expose(
        &mut self,
        frames: &mut Frames,
        gpa: PhysAddr,
        bytes: u64,
        exposure: Exposure,
    ) -> Result<(), NptError> {
        let range = self.owned(gpa, bytes)?;
        self.map.expose(range, exposure.access())?;
        self.overlay(frames, range)
    }

    /// Takes an exposed range back, leaving it as every other page of the
    /// hypervisor's memory already is: the shared page of zeroes, read-only.
    ///
    /// The counterpart of [`Npt::expose`], for entry code whose work is done.
    /// Afterwards the range is indistinguishable from the rest of the chunk — a
    /// guest reading it sees zeroes, and a guest writing it is reported as
    /// [`Resolution::Shadowed`] like any other write to hypervisor memory.
    ///
    /// # The caller discards what the guest cached
    ///
    /// This is the one operation here that may be called after a guest has run,
    /// and it takes permission away rather than granting it — so a processor
    /// that has entered this guest may hold a translation these tables no
    /// longer justify. Getting rid of it is the caller's, because it is the
    /// caller that knows which processors have run the guest and it is the
    /// caller that makes the next entry.
    ///
    /// # Errors
    ///
    /// [`NptError::OutsideOwned`] if the range leaves the chunk,
    /// [`NptError::Map`] if the range is not a whole number of pages on a page
    /// boundary or a page of it was not being shown to the guest, or an error
    /// from building the required nested tables.
    pub fn conceal(
        &mut self,
        frames: &mut Frames,
        gpa: PhysAddr,
        bytes: u64,
    ) -> Result<(), NptError> {
        let range = self.owned(gpa, bytes)?;
        self.map.conceal(range)?;
        self.overlay(frames, range)
    }

    /// Describes one page as a place the guest may touch without anything
    /// answering: reads see what the frame holds and writes are kept by it,
    /// but nothing reads it back.
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
    /// # Only before a guest has run
    ///
    /// Like [`Npt::protect`], this changes what an entry says while there must
    /// be no cached translation to disagree with: no processor may have walked
    /// these tables yet.
    ///
    /// # Errors
    ///
    /// [`NptError::Map`] if the map will not record the page — it is not page
    /// aligned or addressable, something already describes it another way, a
    /// page already sunk included, or there is no room for another — or an
    /// error from allocating or reaching the frame behind it.
    pub fn sink(&mut self, frames: &mut Frames, gpa: PhysAddr) -> Result<(), NptError> {
        let frame = frame(frames, self.window())?;
        // Recorded before it is described, because the map is what `fault`
        // consults: a page the map calls sunk is described that way whenever it
        // is next touched, while a page described as a sink that the map did not
        // record would be filled back over as ordinary memory.
        if let Err(refused) = self.map.sink(gpa, frame) {
            // The frame was handed out for a page the map will not describe that
            // way, so it goes back rather than being held by nothing.
            if let Err(cause) = frames.release(PhysFrame::containing_address(frame), 0) {
                error!("npt: the frame at {frame:#x} could not be handed back: {cause}");
            }
            return Err(refused.into());
        }
        self.describe_page(frames, gpa)
    }

    /// Logs the shape of the translation, which is the whole of what a guest's
    /// view of memory is.
    pub fn describe(&self, who: &str) {
        self.tree.describe(who);
        self.map.describe(who);
    }

    /// The range `bytes` from `gpa` names, refused unless every page of it is
    /// the hypervisor's own memory.
    ///
    /// One check for both directions of a guest-visible chunk mapping, because
    /// what the two have to ask is identical and a range that could be exposed
    /// but not concealed — or the reverse — would be a way for the two to
    /// disagree about what a valid range is.
    fn owned(&self, gpa: PhysAddr, bytes: u64) -> Result<Range, NptError> {
        let range = Range::new(gpa, bytes)?;
        if self.map.chunk().covers(range) {
            Ok(range)
        } else {
            Err(NptError::OutsideOwned {
                gpa: gpa.as_u64(),
                bytes,
            })
        }
    }

    /// Describes every page of a range of the hypervisor's own memory as the
    /// map now says it is.
    ///
    /// One helper for both directions of showing a range to the guest, because
    /// once the map has been told, what those pages are is the map's to say:
    /// shown on purpose, or the shared page of zeroes every other page of the
    /// chunk reads as.
    fn overlay(&mut self, frames: &mut Frames, range: Range) -> Result<(), NptError> {
        range
            .pages()
            .try_for_each(|page| self.describe_page(frames, page.base()))
    }

    /// Describes one page of a trapped region as the map now says it is.
    ///
    /// A write-trapped page is described as the memory really there and
    /// read-only, so that reads reach the hardware without an exit. A page
    /// where every access is trapped is described as nothing at all: a
    /// present entry has no bit that denies a read, so not present is the
    /// only encoding that faults on one.
    fn interpose(&mut self, frames: &mut Frames, gpa: PhysAddr) -> Result<(), NptError> {
        // Broken up first, because the page has to say something its neighbours
        // do not and permissions belong to an entry. This is the one place that
        // needs finer granularity than the fill rule would otherwise produce,
        // and it is why answering a fault never has to.
        self.tree.split(frames, gpa)?;
        self.describe_page(frames, gpa)
    }

    /// Describes the one page at `gpa` as the map says it is.
    ///
    /// Every mutation here works a page at a time — a page shown to the guest,
    /// a page sunk, a page of a trapped region — so each of them asks the map
    /// afresh rather than saying for itself what it just recorded. Two accounts
    /// of one page could disagree; one cannot.
    fn describe_page(&mut self, frames: &mut Frames, gpa: PhysAddr) -> Result<(), NptError> {
        let verdict = self.map.resolve(gpa);
        self.tree.fill(frames, verdict, gpa)
    }
}

/// What a fault on a page of the hypervisor's own memory came to.
///
/// A read is satisfied by whatever the page was described as, so the guest
/// merely retries it. A write is not and never will be: neither the shared page
/// of zeroes nor the entry code the guest is shown has a page behind it that
/// may take one, so the caller has to step the guest past the instruction
/// rather than resume it.
fn hypervisors(cause: NestedPageFault) -> Resolution {
    if cause.write() {
        Resolution::Shadowed
    } else {
        Resolution::Mapped
    }
}

/// What became of a guest physical address that faulted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resolution {
    /// A translation exists now, and the access will succeed when the guest
    /// retries it.
    Mapped,
    /// The guest tried to write memory the hypervisor owns. Reads there see
    /// zeroes; a write cannot be satisfied, so resuming the guest unchanged
    /// re-executes it and faults again. The caller has to emulate the
    /// instruction, discard the write, and resume past it.
    Shadowed,
    /// The address is inside a region the hardware does not answer for.
    /// Nothing was described and nothing will be: what the access means is the
    /// caller's to decide, and stepping the guest past it is the caller's to
    /// do.
    Trapped,
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

#[cfg(test)]
pub(crate) mod tests {
    //! The two ways one page of a guest can be described, what describing one
    //! that way costs its neighbours, and a run of host memory standing in for
    //! the reserved chunk for the rest of the crate to build tables in.
    //!
    //! Only the chunk has to be real. Every table these tables build is a frame
    //! of it, reached through the window, and nothing here ever dereferences a
    //! guest physical address — so a run of memory with a window pointed at it
    //! is the whole of the machine this needs.

    use alloc::alloc::{Layout, alloc_zeroed};

    use paging::{DirectMap, Frames, chunk::FRAME_SIZE};
    use svm::exit::NestedPageFault;
    use x86_64::{PhysAddr, VirtAddr};

    use super::{MapError, Npt, NptError, Resolution, Trap, chunk};

    /// Where the interrupt controllers' register page is, which is the one page
    /// a boot chooses between these two descriptions for.
    const REGISTER_PAGE: u64 = 0xFEE0_0000;

    /// What one entry of the level above a page describes, which is the
    /// granularity ordinary memory is described in wherever nothing is in the
    /// way.
    const LARGE: u64 = 2 << 20;

    #[test]
    fn a_trapped_page_has_no_translation_and_every_access_to_it_is_reported() {
        let (mut frames, window) = reserved();
        let mut npt = Npt::create(&mut frames, window).expect("tables over the chunk");
        let page = PhysAddr::new(REGISTER_PAGE);

        npt.protect(&mut frames, page, FRAME_SIZE, Trap::Everything)
            .expect("the page can be trapped before a guest runs");

        assert_eq!(
            npt.translate(page).expect("the tables can be walked"),
            None,
            "a page every access to which faults must have no translation at all"
        );
        for write in [false, true] {
            assert_eq!(
                npt.fault(&mut frames, page, fault(write))
                    .expect("the fault can be answered"),
                Resolution::Trapped,
                "a {} of a trapped page belongs to whatever answers for it",
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
        let mut npt = Npt::create(&mut frames, window).expect("tables over the chunk");
        let page = PhysAddr::new(REGISTER_PAGE);

        npt.sink(&mut frames, page)
            .expect("the page can be sunk before a guest runs");

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
            npt.fault(&mut frames, page, fault(true))
                .expect("the fault can be answered"),
            Resolution::Mapped,
            "a described page that faults anyway is described rather than reported"
        );
    }

    #[test]
    fn neither_description_performs_the_other() {
        let (mut frames, window) = reserved();
        let page = PhysAddr::new(REGISTER_PAGE);

        let mut trapped = Npt::create(&mut frames, window).expect("tables over the chunk");
        trapped
            .protect(&mut frames, page, FRAME_SIZE, Trap::Everything)
            .expect("the page can be trapped");
        assert_eq!(
            trapped.sink(&mut frames, page),
            Err(NptError::Map(MapError::Overlaps {
                base: REGISTER_PAGE,
                other: REGISTER_PAGE,
            })),
            "trapping leaves the page undescribed, so sinking it would untrap it"
        );

        let mut sunk = Npt::create(&mut frames, window).expect("tables over the chunk");
        sunk.sink(&mut frames, page).expect("the page can be sunk");
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
    fn a_trapped_page_narrows_its_own_two_megabytes_and_nothing_further() {
        let (mut frames, window) = reserved();
        let mut npt = Npt::create(&mut frames, window).expect("tables over the chunk");
        // A page inside a 2 MiB region of ordinary memory, so that the region
        // holding it has to be described more finely than the fill rule would
        // otherwise choose and the regions around it do not.
        let trapped = PhysAddr::new(REGISTER_PAGE + FRAME_SIZE);

        npt.protect(&mut frames, trapped, FRAME_SIZE, Trap::Everything)
            .expect("the page can be trapped before a guest runs");

        for (gpa, span) in [
            (PhysAddr::new(REGISTER_PAGE), FRAME_SIZE),
            (PhysAddr::new(REGISTER_PAGE + LARGE), LARGE),
        ] {
            assert_eq!(
                npt.fault(&mut frames, gpa, fault(false))
                    .expect("the fault can be answered"),
                Resolution::Mapped,
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

    /// A nested page fault of the direction alone, which is all [`Npt::fault`]
    /// reads of one.
    fn fault(write: bool) -> NestedPageFault {
        NestedPageFault::new()
            .with_write(write)
            .with_final_address(true)
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
