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
//! everything the machine has, and *no* for the one region that is the
//! hypervisor's.
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
//! The region a fault describes is the largest one whose whole extent can be
//! answered the same way:
//!
//! - the 1 GiB containing the address, where the processor has 1 GiB pages and
//!   the hypervisor's own memory is nowhere inside it;
//! - otherwise the 2 MiB containing it, on the same condition;
//! - otherwise the address is inside the hypervisor's own memory, and the 2 MiB
//!   containing it is shadowed (below).
//!
//! Those three are exhaustive because the reserved chunk is 2 MiB aligned and a
//! whole number of 2 MiB regions, which makes *overlapping* the chunk and
//! *lying inside* it the same thing at 2 MiB granularity. [`Npt::create`]
//! checks that rather than assuming it, because it is the one property the case
//! analysis rests on.
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
//! # Regions the hardware does not answer for
//!
//! [`Npt::protect`] marks a range as one whose accesses belong to something
//! other than the memory or device behind it — a device the hypervisor
//! interposes on, presenting the guest a view that is not the hardware's.
//! Either writes alone fault or every access does, and [`Npt::fault`] reports
//! [`Resolution::Trapped`] for an address inside one rather than describing it.
//!
//! Trapping a range is the one thing here that needs finer granularity than the
//! fill rule produces, so it splits whatever large page covers the range into
//! 4 KiB entries first. It is also why the fill rule has a third case: a 2 MiB
//! region holding a trapped page cannot be described all at once, so the region
//! containing an address is narrowed until it holds no trapped page, down to a
//! single page if it must be.
//!
//! # One page given rather than trapped
//!
//! [`Npt::sink`] is the opposite of [`Npt::protect`]: where trapping keeps a
//! page from being described so that every access exits, sinking describes one
//! page as a frame of its own, writable, that nothing reads — a place for the
//! guest's accesses to go without anywhere to arrive. The use for one is a
//! device the hardware itself takes over the answering of, where an exit is no
//! longer the expected shape of an access and the page still has to translate
//! to something. Like trapping, it is for before a guest has run, and the fill
//! rule narrows around it the same way: a region holding a sunk page is not
//! described all at once.
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
//! Two operations here reduce what an entry permits, and neither of them
//! flushes anything itself. [`Npt::protect`] needs no flush at all, because it
//! may only be called before the guest has ever run: nothing has walked these
//! tables at that point, so no processor holds a translation the tables have
//! stopped justifying. [`Npt::conceal`] is the one that may be called while a
//! guest is running, and it states in as many words that discarding what the
//! guest cached is the caller's — the caller is what knows which processors ran
//! the guest and what makes the next entry, and `TLB_CONTROL` is a field of a
//! control block this crate does not have.

#![no_std]

#[cfg(test)]
extern crate alloc;

mod walk;

use log::info;
use paging::{DirectMap, Frames, chunk};
use processor::Features;
use svm::exit::NestedPageFault;
use thiserror::Error;
use x86_64::{PhysAddr, structures::paging::PageTableFlags};

use crate::walk::{Level, Meeting, PARENT};

/// One guest's nested page tables.
///
/// Holds no allocator and no lock. Frames are handed in per call by whoever
/// owns the chunk's allocator, and exclusion is the caller's — which keeps this
/// a description of a translation rather than a second owner of the machine's
/// memory.
#[derive(Debug)]
pub struct Npt {
    root: PhysAddr,
    zero: PhysAddr,
    owned: Span,
    trapped: [Option<Interposed>; TRAPPED_REGIONS],
    sink: Option<Sink>,
    window: DirectMap,
    large: bool,
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
    /// regions — the property that makes the fill rule's cases exhaustive.
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
            root,
            zero,
            owned: Span::new(frames.chunk_base(), chunk::CHUNK_SIZE),
            trapped: [None; TRAPPED_REGIONS],
            sink: None,
            window,
            large: processor::features().contains(Features::GIB_PAGES),
        })
    }

    /// The value a guest's control block names these tables by, in its nested
    /// table root field.
    #[must_use]
    pub const fn root(&self) -> PhysAddr {
        self.root
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
        self.window
    }

    /// Describes the region containing a guest physical address that had no
    /// translation.
    ///
    /// `cause` is the first exit-information field of the nested page fault,
    /// decoded. Only its direction is read: what a fault means here does not
    /// otherwise depend on why the guest was touching the address.
    ///
    /// # Errors
    ///
    /// [`NptError::OutOfFrames`] if the chunk cannot spare a table,
    /// [`NptError::Unreachable`] if the window does not reach one, or
    /// [`NptError::LargePage`] if a large page already covers the address,
    /// which means something described this region at a granularity the
    /// fill rule never produces.
    pub fn fault(
        &mut self,
        frames: &mut Frames,
        gpa: PhysAddr,
        cause: NestedPageFault,
    ) -> Result<Resolution, NptError> {
        // Before anything else, because a trapped address faults on purpose and
        // describing it is exactly what must not happen.
        if self.caught(gpa) {
            return Ok(Resolution::Trapped);
        }
        let ours = self.owned.contains(gpa);
        if walk::lookup(self.window, self.root, gpa)?.is_none() {
            if ours {
                self.shadow(frames, gpa)?;
            } else {
                self.identity(frames, gpa)?;
            }
        }
        Ok(if ours && cause.write() {
            Resolution::Shadowed
        } else {
            Resolution::Mapped
        })
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
        let Some(leaf) = walk::lookup(self.window, self.root, gpa)? else {
            return Ok(None);
        };
        // How far into the region the address falls, which is both how far into
        // the frame it lands and how much of the region is behind it.
        let offset = gpa.as_u64() - gpa.align_down(leaf.level.span()).as_u64();
        Ok(Some(Translation {
            spa: leaf.frame + offset,
            writable: leaf.flags.contains(PageTableFlags::WRITABLE),
            span: leaf.level.span() - offset,
        }))
    }

    /// Marks a range as one whose accesses are not the hardware's to answer.
    ///
    /// The range is described one page at a time, splitting whatever larger
    /// page covers it, because permissions belong to an entry and
    /// neighbouring pages must keep theirs. What the guest may still do for
    /// itself is [`Trap`]'s to say.
    ///
    /// The range is remembered as well as described, because describing it is
    /// not enough on its own: [`Npt::fault`] would otherwise fill a trapped
    /// page back in the first time the guest touched it, and would describe
    /// a 2 MiB region straight over one.
    ///
    /// # Only before a guest has run
    ///
    /// This is the one operation here that reduces what an entry permits, and
    /// it may only be called before these tables have ever been entered.
    /// Every processor's cached translations would otherwise have to be
    /// discarded, and nothing here does that — precisely because nothing
    /// can have cached one yet. Trapping a region while a guest is running
    /// would need that machinery first, and would need it before this call
    /// rather than after.
    ///
    /// # Errors
    ///
    /// [`NptError::TrapGeometry`] unless the range is a whole number of pages
    /// on a page boundary, [`NptError::TooManyTrapped`] if these tables
    /// have no room to remember another region, [`NptError::OutOfFrames`]
    /// if the chunk cannot spare a table, [`NptError::Unreachable`] if the
    /// window does not reach one, or [`NptError::LargePage`] if a large
    /// page turns up at a level the architecture has none at.
    pub fn protect(
        &mut self,
        frames: &mut Frames,
        gpa: PhysAddr,
        bytes: u64,
        trap: Trap,
    ) -> Result<(), NptError> {
        let page = Level::Page.span();
        if bytes == 0 || !gpa.as_u64().is_multiple_of(page) || !bytes.is_multiple_of(page) {
            return Err(NptError::TrapGeometry {
                gpa: gpa.as_u64(),
                bytes,
            });
        }
        let Some(slot) = self.trapped.iter().position(Option::is_none) else {
            return Err(NptError::TooManyTrapped {
                limit: TRAPPED_REGIONS,
            });
        };
        // Remembered before it is described, so that a failure part-way through
        // leaves a region that still traps everything it should. The reverse
        // order would leave pages described as untouchable that nothing knows to
        // trap, which is a guest faulting for ever on an address the tables have
        // no answer for.
        self.trapped[slot] = Some(Interposed {
            span: Span::new(gpa, bytes),
            trap,
        });

        let described = (0..bytes / page)
            .try_for_each(|index| self.interpose(frames, gpa + index * page, trap));
        if described.is_err() {
            // The region is not described and so must not go on being remembered:
            // a slot naming pages that were never trapped would refuse to let
            // `fault` describe them, and the guest would fault on them for ever
            // with nothing to answer. The pages that *were* described are left as
            // they are — they trap, which is safe — and the caller undoing this
            // registration removes them.
            self.trapped[slot] = None;
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
    /// depends on whether it is the hypervisor's own memory or the guest's
    /// — which is exactly the question [`Npt::fault`] already answers. So
    /// the pages are left with no translation, which is where every page of
    /// a guest starts, and the first access to one describes it correctly.
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
    /// [`NptError::TrapGeometry`] unless the range is a whole number of pages
    /// on a page boundary, [`NptError::NotTrapped`] if no region was
    /// recorded at exactly that range, or [`NptError::Unreachable`] if the
    /// window does not reach one of these tables.
    pub fn release(&mut self, gpa: PhysAddr, bytes: u64) -> Result<(), NptError> {
        let page = Level::Page.span();
        if bytes == 0 || !gpa.as_u64().is_multiple_of(page) || !bytes.is_multiple_of(page) {
            return Err(NptError::TrapGeometry {
                gpa: gpa.as_u64(),
                bytes,
            });
        }
        let span = Span::new(gpa, bytes);
        let slot = self
            .trapped
            .iter()
            .position(|region| region.is_some_and(|region| region.span == span))
            .ok_or(NptError::NotTrapped {
                gpa: gpa.as_u64(),
                bytes,
            })?;
        // Forgotten first, so that a failure part-way through leaves pages that
        // `fault` is willing to describe rather than pages it refuses to touch and
        // nothing answers for.
        self.trapped[slot] = None;
        (0..bytes / page).try_for_each(|index| self.abandon(gpa + index * page))
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
    /// [`NptError::TrapGeometry`] unless the range is a whole number of pages
    /// on page boundaries, [`NptError::OutsideOwned`] if it leaves the chunk,
    /// or an error from building the required nested tables.
    pub fn expose(
        &mut self,
        frames: &mut Frames,
        gpa: PhysAddr,
        bytes: u64,
        exposure: Exposure,
    ) -> Result<(), NptError> {
        self.overlay(frames, gpa, bytes, Some(exposure))
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
    /// [`NptError::TrapGeometry`] unless the range is a whole number of pages
    /// on page boundaries, [`NptError::OutsideOwned`] if it leaves the chunk,
    /// or an error from building the required nested tables.
    pub fn conceal(
        &mut self,
        frames: &mut Frames,
        gpa: PhysAddr,
        bytes: u64,
    ) -> Result<(), NptError> {
        self.overlay(frames, gpa, bytes, None)
    }

    /// Describes a range of the hypervisor's own memory one page at a time,
    /// either as itself or as the shared page of zeroes.
    ///
    /// One body for both directions, because what the two have to check is
    /// identical and a range that could be exposed but not concealed — or the
    /// reverse — would be a way for the two to disagree about what a valid
    /// range is.
    fn overlay(
        &mut self,
        frames: &mut Frames,
        gpa: PhysAddr,
        bytes: u64,
        exposure: Option<Exposure>,
    ) -> Result<(), NptError> {
        let page = Level::Page.span();
        if bytes == 0 || !gpa.as_u64().is_multiple_of(page) || !bytes.is_multiple_of(page) {
            return Err(NptError::TrapGeometry {
                gpa: gpa.as_u64(),
                bytes,
            });
        }
        if !self.owned.contains(gpa) || !self.owned.contains(gpa + bytes - 1) {
            return Err(NptError::OutsideOwned {
                gpa: gpa.as_u64(),
                bytes,
            });
        }
        (0..bytes / page).try_for_each(|index| {
            let page = gpa + index * page;
            let (frame, flags) = match exposure {
                Some(exposure) => (page, exposure.flags()),
                None => (self.zero, SHADOW),
            };
            walk::map(
                self.window,
                self.root,
                page,
                Level::Page,
                frame,
                flags,
                frames,
            )
        })
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
    /// The page is remembered as well as described, for the reason a trapped
    /// region is: [`Npt::fault`] would otherwise describe a large page
    /// straight over it the first time the guest touched a neighbour.
    ///
    /// # Only before a guest has run
    ///
    /// Like [`Npt::protect`], this changes what an entry says while there must
    /// be no cached translation to disagree with: no processor may have walked
    /// these tables yet.
    ///
    /// # Errors
    ///
    /// [`NptError::TrapGeometry`] unless the address is page aligned,
    /// [`NptError::AlreadySunk`] if a page is already sunk,
    /// [`NptError::Trapped`] if the address is inside a trapped region —
    /// describing it would overwrite the entry the trap is expressed in and
    /// quietly untrap it — or an error from allocating or reaching the frame.
    pub fn sink(&mut self, frames: &mut Frames, gpa: PhysAddr) -> Result<(), NptError> {
        let page = Level::Page.span();
        if !gpa.as_u64().is_multiple_of(page) {
            return Err(NptError::TrapGeometry {
                gpa: gpa.as_u64(),
                bytes: page,
            });
        }
        if self.sink.is_some() {
            return Err(NptError::AlreadySunk { gpa: gpa.as_u64() });
        }
        if self.caught(gpa) {
            return Err(NptError::Trapped { gpa: gpa.as_u64() });
        }
        let frame = frame(frames, self.window)?;
        walk::map(
            self.window,
            self.root,
            gpa,
            Level::Page,
            frame,
            LEAF,
            frames,
        )?;
        self.sink = Some(Sink {
            span: Span::new(gpa, page),
            frame,
        });
        Ok(())
    }

    /// Logs the shape of the translation, which is the whole of what a guest's
    /// view of memory is.
    pub fn describe(&self, who: &str) {
        info!(
            "{who}: npt rooted at {:#x}, identity mapped in {} pages",
            self.root,
            if self.large { "1 GiB" } else { "2 MiB" },
        );
        info!(
            "{who}: npt shadows physical {:#x}..{:#x} onto {:#x}, read only",
            self.owned.base, self.owned.end, self.zero,
        );
        for region in self.trapped.iter().flatten() {
            info!(
                "{who}: npt traps physical {:#x}..{:#x}, {}",
                region.span.base,
                region.span.end,
                match region.trap {
                    Trap::Writes => "writes only",
                    Trap::Everything => "every access",
                },
            );
        }
        if let Some(sink) = self.sink {
            info!(
                "{who}: npt sinks guest physical {:#x} onto frame {:#x}",
                sink.span.base, sink.frame,
            );
        }
    }

    /// Maps the largest region containing `gpa` that can be answered the same
    /// way throughout, to itself.
    ///
    /// Three cases rather than two, because a trapped page can sit anywhere. A
    /// 2 MiB region overlapping the chunk lies entirely inside it, so an
    /// address outside the chunk is never narrowed on the chunk's account —
    /// but a 2 MiB region holding one trapped page is otherwise ordinary
    /// memory, and the 511 pages around it are still the guest's to reach
    /// at full speed.
    ///
    /// Narrowing to a single page is therefore the floor, and it is always
    /// right when it is reached: [`Npt::fault`] has already answered for an
    /// address that is the hypervisor's or trapped, so what is left is one page
    /// of real memory that is neither.
    fn identity(&mut self, frames: &mut Frames, gpa: PhysAddr) -> Result<(), NptError> {
        if self.large {
            let base = gpa.align_down(Level::Pointer.span());
            if self.describable(base, Level::Pointer) {
                return self.itself(frames, base, Level::Pointer);
            }
        }
        let base = gpa.align_down(Level::Directory.span());
        if self.describable(base, Level::Directory) {
            return self.itself(frames, base, Level::Directory);
        }
        self.itself(frames, gpa.align_down(Level::Page.span()), Level::Page)
    }

    /// Whether the whole `level`-sized region at `base` can be described as
    /// itself: none of it the hypervisor's own, none of it trapped, and none
    /// of it sunk.
    fn describable(&self, base: PhysAddr, level: Level) -> bool {
        !self.owned.overlaps(base, level.span())
            && self
                .sink
                .is_none_or(|sink| !sink.span.overlaps(base, level.span()))
            && !self
                .trapped
                .iter()
                .flatten()
                .any(|region| region.span.overlaps(base, level.span()))
    }

    /// Whether this address is inside a region the hardware does not answer
    /// for.
    fn caught(&self, gpa: PhysAddr) -> bool {
        self.trapped
            .iter()
            .flatten()
            .any(|region| region.span.contains(gpa))
    }

    /// Describes one page of a trapped region.
    ///
    /// A write-trapped page is still described, as itself and read-only, so
    /// that reads reach the hardware without an exit. A page where every
    /// access is trapped is described as nothing at all: a present entry
    /// has no bit that denies a read, so not present is the only encoding
    /// that faults on one.
    fn interpose(
        &mut self,
        frames: &mut Frames,
        gpa: PhysAddr,
        trap: Trap,
    ) -> Result<(), NptError> {
        let mut table = walk::descend(
            self.window,
            self.root,
            gpa,
            Level::Page,
            frames,
            Meeting::Split,
        )?;
        // SAFETY: `descend` returns a table of these nested tables, reached
        // through the window, and `&mut self` is the only handle to them.
        let table = unsafe { table.as_mut() };
        let entry = &mut table[Level::Page.index(gpa)];
        match trap {
            Trap::Writes => entry.set_addr(gpa, TRAPPED),
            Trap::Everything => entry.set_unused(),
        }
        Ok(())
    }

    /// Leaves one page of a released region with no translation at all.
    ///
    /// The opposite of [`Npt::interpose`], and deliberately not its mirror
    /// image: it does not put back whatever the page was described as
    /// before, because that is not knowable here and is not worth
    /// remembering. A page with no entry is where every page of a guest
    /// starts, and the first access to one goes through [`Npt::fault`],
    /// which decides correctly whether it is the guest's memory or the
    /// hypervisor's.
    ///
    /// A page that has no table under it needs nothing done: there is no entry
    /// to clear, which is already the state this is trying to reach.
    fn abandon(&mut self, gpa: PhysAddr) -> Result<(), NptError> {
        let Some(mut table) = walk::table_of(self.window, self.root, gpa, Level::Page)? else {
            return Ok(());
        };
        // SAFETY: `table_of` returns a table of these nested tables, reached
        // through the window, and `&mut self` is the only handle to them.
        let table = unsafe { table.as_mut() };
        table[Level::Page.index(gpa)].set_unused();
        Ok(())
    }

    /// Points every 4 KiB of the 2 MiB region containing `gpa` at the shared
    /// page of zeroes.
    ///
    /// The whole 2 MiB at once, because by the geometry [`Npt::create`]
    /// checked, all of it is the hypervisor's: describing it page by page
    /// as each one is touched would cost a fault per page and arrive at the
    /// same table.
    fn shadow(&mut self, frames: &mut Frames, gpa: PhysAddr) -> Result<(), NptError> {
        let base = gpa.align_down(Level::Directory.span());
        let mut table = walk::descend(
            self.window,
            self.root,
            base,
            Level::Page,
            frames,
            Meeting::Refuse,
        )?;
        // SAFETY: `descend` returns a table of these nested tables, reached
        // through the window, and `&mut self` is the only handle to them.
        let table = unsafe { table.as_mut() };
        for entry in table.iter_mut() {
            entry.set_addr(self.zero, SHADOW);
        }
        Ok(())
    }

    /// Describes the `level`-sized region beginning at `base` as itself.
    ///
    /// `base` is both the guest physical address being described and the system
    /// physical address it is described as, which is the whole of what makes
    /// this an identity map — and the reason the two are one argument
    /// rather than two that could be passed the wrong way round.
    fn itself(
        &mut self,
        frames: &mut Frames,
        base: PhysAddr,
        level: Level,
    ) -> Result<(), NptError> {
        walk::map(self.window, self.root, base, level, base, LEAF, frames)
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
    /// Nested page-table flags implementing this access.
    const fn flags(self) -> PageTableFlags {
        let present = PageTableFlags::PRESENT.union(PageTableFlags::USER_ACCESSIBLE);
        match self {
            Self::ReadOnly => present.union(PageTableFlags::NO_EXECUTE),
            Self::ReadExecute => present,
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

/// What a guest may still do for itself in a trapped region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Trap {
    /// Reads reach the hardware directly and cost nothing; writes fault.
    ///
    /// The choice for a device whose registers read back what they are, and
    /// where only what the guest writes has to be interfered with.
    Writes,
    /// Every access faults.
    ///
    /// The choice for a device the guest must be shown something other than
    /// the truth about. It costs an exit per read as well as per write, which
    /// is the price of the read never reaching the hardware.
    Everything,
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
    /// A large page already covers the address, and nothing here splits one.
    #[error("guest physical {gpa:#x} is already covered by a large page")]
    LargePage {
        /// The address in question.
        gpa: u64,
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
    /// A region to be trapped was not a whole number of pages on a page
    /// boundary, which is the granularity permissions come in.
    #[error("a trapped region at {gpa:#x} of {bytes:#x} bytes is not a whole number of pages")]
    TrapGeometry {
        /// Where the region begins.
        gpa: u64,
        /// How long it is.
        bytes: u64,
    },
    /// A region was released that was never trapped, or was trapped as part of
    /// a differently shaped one.
    ///
    /// Released by exactly the range it was trapped by, deliberately: releasing
    /// half of a region would leave the other half trapped by a record that no
    /// longer describes it.
    #[error("no trapped region covers exactly {gpa:#x} for {bytes:#x} bytes")]
    NotTrapped {
        /// Where the region was said to begin.
        gpa: u64,
        /// How long it was said to be.
        bytes: u64,
    },
    /// A guest-visible chunk mapping was requested outside the chunk.
    #[error("guest exposure at {gpa:#x} of {bytes:#x} bytes leaves the hypervisor chunk")]
    OutsideOwned {
        /// Guest physical base requested.
        gpa: u64,
        /// Bytes requested.
        bytes: u64,
    },
    /// More regions were trapped than these tables have room to remember.
    #[error("no room to trap another region; {limit} is the most these tables remember")]
    TooManyTrapped {
        /// How many they remember.
        limit: usize,
    },
    /// A page was asked to be sunk when one already is; these tables remember
    /// at most one.
    #[error("guest physical {gpa:#x} cannot be sunk: a page already is")]
    AlreadySunk {
        /// The address that was asked.
        gpa: u64,
    },
    /// A page inside a trapped region was asked to be sunk, which would
    /// describe what must stay undescribed.
    #[error("guest physical {gpa:#x} is inside a trapped region and cannot be sunk")]
    Trapped {
        /// The address that was asked.
        gpa: u64,
    },
}

/// How many regions of a guest's physical memory may be trapped at once.
///
/// Sized for the devices a pass-through hypervisor interposes on rather than
/// for a machine's aperture space: each one is a device's registers, and every
/// fault consults the whole list, so this is a number that wants to stay small
/// enough to scan. A hypervisor that needed hundreds would want a different
/// structure rather than a larger array.
const TRAPPED_REGIONS: usize = 16;

/// One region whose accesses are not the hardware's to answer.
#[derive(Clone, Copy, Debug)]
struct Interposed {
    span: Span,
    trap: Trap,
}

/// One page the guest may touch freely, and the frame its accesses are kept
/// by.
#[derive(Clone, Copy, Debug)]
struct Sink {
    span: Span,
    frame: PhysAddr,
}

/// A run of guest physical addresses.
///
/// Two quite different things are described by one of these — the memory that
/// is the hypervisor's own, and a region something interposes on — but the
/// questions asked of them are the same two, so they are written once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Span {
    base: u64,
    end: u64,
}

impl Span {
    /// The run of `bytes` beginning at `base`.
    fn new(base: PhysAddr, bytes: u64) -> Self {
        let base = base.as_u64();
        Self {
            base,
            end: base.saturating_add(bytes),
        }
    }

    /// Whether this address is inside the run.
    fn contains(self, phys: PhysAddr) -> bool {
        (self.base..self.end).contains(&phys.as_u64())
    }

    /// Whether any of `span` bytes from `base` is inside the run.
    fn overlaps(self, base: PhysAddr, span: u64) -> bool {
        let base = base.as_u64();
        base < self.end && self.base < base.saturating_add(span)
    }
}

/// Flags on a leaf that describes real memory the guest may use freely.
///
/// Write-back, which is the absence of both cache bits rather than a bit of its
/// own: it is what leaves the guest's own memory type in force.
const LEAF: PageTableFlags = PARENT;

/// Flags on a leaf that stands in for the hypervisor's own memory: readable and
/// executable so that a guest walking memory is not surprised, and not writable
/// so that the shared page of zeroes stays zero.
const SHADOW: PageTableFlags = PageTableFlags::PRESENT.union(PageTableFlags::USER_ACCESSIBLE);

/// Flags on a leaf of a region whose writes are trapped: describing the memory
/// that is really there, readable and executable, and not writable — so a read
/// costs nothing and a write faults.
///
/// The same bits as [`SHADOW`] and for an entirely different reason. They are
/// named apart so that changing what one of them means cannot quietly change
/// the other.
const TRAPPED: PageTableFlags = PageTableFlags::PRESENT.union(PageTableFlags::USER_ACCESSIBLE);

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
    !Exposure::ReadOnly
        .flags()
        .contains(PageTableFlags::WRITABLE)
        && !Exposure::ReadExecute
            .flags()
            .contains(PageTableFlags::WRITABLE),
    "guest-callable portal pages must remain immutable",
);
const _: () = assert!(
    chunk::CHUNK_ALIGN.is_multiple_of(Level::Directory.span()),
    "a 2 MiB region overlapping the chunk must lie entirely inside it",
);
const _: () = assert!(
    !SHADOW.contains(PageTableFlags::WRITABLE),
    "the page of zeroes standing in for the hypervisor must not be writable",
);
const _: () = assert!(
    TRAPPED.contains(PageTableFlags::PRESENT) && !TRAPPED.contains(PageTableFlags::WRITABLE),
    "a write-trapped page must be present, so that reads cost nothing, and not \
     writable, so that writes fault",
);
const _: () = assert!(
    LEAF.contains(PageTableFlags::USER_ACCESSIBLE)
        && !LEAF.contains(PageTableFlags::NO_EXECUTE)
        && !LEAF.contains(PageTableFlags::WRITE_THROUGH)
        && !LEAF.contains(PageTableFlags::NO_CACHE),
    "a guest reaches nested memory only through a user page, executes only \
     without no-execute, and keeps its own memory type only under write-back",
);

#[cfg(test)]
mod tests {
    //! The two ways one page of a guest can be described, over a run of host
    //! memory standing in for the reserved chunk.
    //!
    //! Only the chunk has to be real. Every table these tables build is a frame
    //! of it, reached through the window, and nothing here ever dereferences a
    //! guest physical address — so a run of memory with a window pointed at it
    //! is the whole of the machine this needs.

    use alloc::alloc::{Layout, alloc_zeroed};

    use paging::{DirectMap, Frames, chunk::FRAME_SIZE};
    use svm::exit::NestedPageFault;
    use x86_64::{PhysAddr, VirtAddr};

    use super::{Npt, NptError, Resolution, Trap, chunk};

    /// Where the interrupt controllers' register page is, which is the one page
    /// a boot chooses between these two descriptions for.
    const REGISTER_PAGE: u64 = 0xFEE0_0000;

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
            Err(NptError::Trapped { gpa: REGISTER_PAGE }),
            "trapping leaves the page undescribed, so sinking it would untrap it"
        );

        let mut sunk = Npt::create(&mut frames, window).expect("tables over the chunk");
        sunk.sink(&mut frames, page).expect("the page can be sunk");
        assert_eq!(
            sunk.release(page, FRAME_SIZE),
            Err(NptError::NotTrapped {
                gpa: REGISTER_PAGE,
                bytes: FRAME_SIZE,
            }),
            "sinking records no trapped region, so there is none to give back"
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
    fn reserved() -> (Frames, DirectMap) {
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
