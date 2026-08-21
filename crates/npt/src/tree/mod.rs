//! Four levels of hardware page table, and the rule that decides how coarsely a
//! run of guest physical addresses is written into them.
//!
//! What an address *means* is [`crate::map`]'s to answer. Everything here is
//! the machinery that writes such an answer where the processor's page walker
//! will find it, and it knows nothing about why an address means what it does.
//!
//! # The level a fill chooses is a function of the answer and the address
//!
//! A region is described at the largest level `L` such that the `L`-aligned
//! region containing the address lies wholly inside the run the map answered
//! for, the processor has pages at `L`, and no table already exists at `L` on
//! the path to it. So the granularity falls out of the answer rather than out
//! of a search of the tree, and two things follow that are worth stating
//! because they are what keep the fault path simple.
//!
//! **A table where a leaf was wanted means something narrowed this region
//! deliberately.** The rule re-applies one level down, where it can only choose
//! a finer level than it just did. A table is never turned back into a large
//! page: nothing here coarsens a description, and doing it while a guest was
//! running would mean making every processor forget the finer one first.
//!
//! **A leaf where a table was wanted is a broken invariant.** Every mutation
//! that narrows a region breaks the larger page covering it into pages *before*
//! describing the region more finely, so meeting one means that order was not
//! kept. It is reported as [`NptError::Coarser`] and never repaired:
//! [`Tree::split`] is the operation that narrows the tree, and answering a
//! fault is not allowed to perform one.

mod entry;
pub(crate) mod walk;

use log::info;
use paging::{DirectMap, Frames, as_usize};
use x86_64::{PhysAddr, structures::paging::PageTableIndex};

use crate::{
    NptError, Translation,
    map::{Kind, Verdict},
    tree::{
        entry::{Entry, encode},
        walk::Level,
    },
};

/// One guest's nested page tables, as the hardware walks them.
///
/// Holds no allocator: frames for the tables are handed in per call by whoever
/// owns the chunk's, which keeps this a description of a translation rather
/// than a second owner of the machine's memory.
#[derive(Debug)]
pub(crate) struct Tree {
    /// The table `nCR3` names.
    root: PhysAddr,
    /// The one frame every page of the hypervisor's own memory reads as.
    zero: PhysAddr,
    /// The window onto physical memory the tables themselves are reached
    /// through.
    window: DirectMap,
    /// Whether the processor has pages of a gigabyte.
    large: bool,
}

impl Tree {
    /// An empty tree rooted at `root`, describing nothing yet.
    ///
    /// Empty is the correct starting point rather than a stub: with no entry
    /// present, a guest's first access to any address faults, and
    /// [`Tree::fill`] is what turns that into a translation.
    pub(crate) const fn new(
        root: PhysAddr,
        zero: PhysAddr,
        window: DirectMap,
        large: bool,
    ) -> Self {
        Self {
            root,
            zero,
            window,
            large,
        }
    }

    /// The value a guest's control block names these tables by.
    pub(crate) const fn root(&self) -> PhysAddr {
        self.root
    }

    /// The window the tables are reached through, which is the window whatever
    /// they translate to has to be reached through as well.
    pub(crate) const fn window(&self) -> DirectMap {
        self.window
    }

    /// Describes the coarsest region containing `gpa` whose whole extent the
    /// verdict covers, as the memory the verdict says is behind it.
    ///
    /// Idempotent, and cheaply so: an entry already saying this is left alone,
    /// which is what answering a fault on an address that is already described
    /// comes to — a write to memory the guest may only read faults every time
    /// it is retried.
    ///
    /// # Errors
    ///
    /// [`NptError::Coarser`] if a leaf already covers the address at a coarser
    /// level than the verdict allows, [`NptError::OutOfFrames`] if the chunk
    /// cannot spare a table, or [`NptError::Unreachable`] if the window does
    /// not reach one.
    pub(crate) fn fill(
        &mut self,
        frames: &mut Frames,
        verdict: Verdict,
        gpa: PhysAddr,
    ) -> Result<(), NptError> {
        match verdict.kind {
            // The hypervisor's own memory is described a whole table at a time.
            // Every page of it reads as the same frame, so there is one entry to
            // write five hundred and twelve times, and writing them one fault at
            // a time would cost five hundred and twelve faults to arrive at the
            // same table.
            Kind::Shadow => self.shadow(frames, verdict, gpa),
            Kind::Ram { .. }
            | Kind::Exposed { .. }
            | Kind::Sink { .. }
            | Kind::Interposed { .. }
            | Kind::Unaddressable => {
                let at = self.descend(frames, gpa, self.coarsest(verdict, gpa))?;
                self.install(at, self.describes(verdict, at.level, gpa))
            }
        }
    }

    /// Where a guest physical address really is, or `None` if nothing describes
    /// it yet.
    ///
    /// `None` is not a failure: a guest's memory is described as it is touched,
    /// so an address it has not touched has no translation.
    ///
    /// # Errors
    ///
    /// [`NptError::Unreachable`] if the window does not reach one of these
    /// tables, which is a broken window rather than anything about `gpa`.
    pub(crate) fn translate(&self, gpa: PhysAddr) -> Result<Option<Translation>, NptError> {
        let at = self.find(gpa)?;
        if !entry::present(at.value) {
            return Ok(None);
        }
        // How far into the region the address falls, which is both how far into
        // the memory behind it that lands and how much of the region is left.
        let offset = gpa.as_u64() - gpa.align_down(at.level.span()).as_u64();
        Ok(Some(Translation {
            spa: entry::frame(at.value) + offset,
            writable: entry::writable(at.value),
            span: at.level.span() - offset,
        }))
    }

    /// Describes whatever covers `gpa` in pages rather than in one larger page.
    ///
    /// What a mutation narrowing a region calls before it describes the region:
    /// permissions belong to an entry, so a page cannot be given permissions of
    /// its own while one entry above it describes all five hundred and twelve
    /// of its neighbours. Nothing about the translation changes, only how
    /// finely it is written down.
    ///
    /// An address nothing describes needs nothing done. There is no larger page
    /// to break up, and whatever describes it next is written at the
    /// granularity it asks for.
    ///
    /// # Errors
    ///
    /// [`NptError::OutOfFrames`] if the chunk cannot spare a table,
    /// [`NptError::Unreachable`] if the window does not reach one, or
    /// [`NptError::Coarser`] if a leaf turns up at a level the architecture has
    /// no large page at, which is an entry this crate did not write.
    pub(crate) fn split(&mut self, frames: &mut Frames, gpa: PhysAddr) -> Result<(), NptError> {
        let mut table = self.root;
        for level in Level::TABLES {
            let index = level.index(gpa);
            let value = walk::load(self.window, table, index)?;
            if !entry::present(value) {
                return Ok(());
            }
            let at = Reached {
                table,
                index,
                level,
                value,
            };
            table = if entry::leaf(value, level) {
                self.divide(frames, at, gpa)?
            } else {
                entry::frame(value)
            };
        }
        Ok(())
    }

    /// Leaves one page with no translation at all.
    ///
    /// Deliberately not the mirror image of describing one: it does not put
    /// back whatever the page was before, because that is not knowable here
    /// and is not worth remembering. A page with no entry is where every
    /// page of a guest starts, and the first access to one is answered by
    /// the map.
    ///
    /// A page under a larger page, or under no table at all, needs nothing
    /// done. Nothing here describes one page of a trapped region coarsely,
    /// so there is no entry of its own to clear, and clearing the larger
    /// one would abandon its neighbours too.
    ///
    /// # Errors
    ///
    /// [`NptError::Unreachable`] if the window does not reach one of these
    /// tables.
    pub(crate) fn abandon(&mut self, gpa: PhysAddr) -> Result<(), NptError> {
        let at = self.find(gpa)?;
        if !matches!(at.level, Level::Page) {
            return Ok(());
        }
        self.install(at, encode(Entry::Absent))
    }

    /// Logs the shape of the translation, which is the whole of what a guest's
    /// view of memory is that the machine's own memory map does not say.
    pub(crate) fn describe(&self, who: &str) {
        info!(
            "{who}: npt rooted at {:#x}, the machine's memory identically in {} pages, the \
             hypervisor's own reading as frame {:#x}",
            self.root,
            if self.large { "1 GiB" } else { "2 MiB" },
            self.zero,
        );
    }

    /// Points every page of the hypervisor's own memory around `gpa` that the
    /// verdict covers at the shared frame of zeroes.
    ///
    /// A whole page table's worth at once, because by the geometry the chunk is
    /// checked against, all of that region is the hypervisor's. The pages the
    /// verdict does not cover are the few of the chunk described some other way
    /// — shown to the guest on purpose, or handed to something else to answer
    /// for — and putting zeroes over one of those would take back what it was
    /// given.
    fn shadow(
        &mut self,
        frames: &mut Frames,
        verdict: Verdict,
        gpa: PhysAddr,
    ) -> Result<(), NptError> {
        let at = self.descend(frames, gpa, SHADOW)?;
        // One entry serves every page of the region, all of them reading as the
        // same frame. So the entry for the page this fault was for already
        // saying it means the whole region does — and a write to the
        // hypervisor's own memory, which faults every time the guest retries it,
        // must not rewrite five hundred and twelve entries each time.
        let wanted = self.describes(verdict, SHADOW, gpa);
        if entry::unchanged(at.value, wanted) {
            return Ok(());
        }
        let first = gpa.align_down(SHADOWED);
        walk::store_each(self.window, at.table, |slot| {
            verdict
                .covers(first + slot * SHADOW.span(), SHADOW.span())
                .then_some(wanted)
        })
    }

    /// The coarsest level whose region containing `gpa` the verdict covers
    /// whole.
    ///
    /// A run reaching a gigabyte past the address in both directions is written
    /// down as one 1 GiB entry, and one that stops a page away is written down
    /// as one page. The floor is one page and reaching it is always right,
    /// because a verdict covers at least the page containing the address it
    /// answered for.
    fn coarsest(&self, verdict: Verdict, gpa: PhysAddr) -> Level {
        match verdict.kind {
            // Only the machine's own memory is a run of frames that one entry
            // can describe.
            Kind::Ram { .. } => [Level::Pointer, Level::Directory]
                .into_iter()
                // The coarser of the two exists only on a processor that reports
                // it.
                .skip(usize::from(!self.large))
                .find(|level| verdict.covers(gpa.align_down(level.span()), level.span()))
                .unwrap_or(Level::Page),
            Kind::Shadow => SHADOW,
            // A page sunk, shown or trapped is one page by construction, and an
            // address the machine does not have is not described at all. A
            // larger entry would describe a run of frames beginning at one of
            // them, which is not what any of them is.
            Kind::Sink { .. }
            | Kind::Exposed { .. }
            | Kind::Interposed { .. }
            | Kind::Unaddressable => Level::Page,
        }
    }

    /// The entry describing the `level` region containing `gpa`, as the verdict
    /// says that region is.
    fn describes(&self, verdict: Verdict, level: Level, gpa: PhysAddr) -> u64 {
        let base = gpa.align_down(level.span());
        encode(Entry::Leaf {
            kind: verdict.kind,
            level,
            frame: self.behind(verdict, base),
        })
    }

    /// Where the run the verdict answers for really is, from `base` onwards.
    fn behind(&self, verdict: Verdict, base: PhysAddr) -> PhysAddr {
        match verdict.kind {
            // Described from the run's own base onwards, so the region being
            // described begins that far into whatever is behind the run.
            Kind::Ram { spa, .. } => spa + (base - verdict.base),
            // One frame each, however far the run around them reaches.
            Kind::Sink { spa } | Kind::Exposed { spa, .. } => spa,
            // Every page of the hypervisor's own memory reads as one shared
            // frame.
            Kind::Shadow => self.zero,
            // The hardware at the same address, for a region only whose writes
            // are trapped. For one where every access is trapped, and for an
            // address the machine does not have, nothing is described at all and
            // the answer is not read.
            Kind::Interposed { .. } | Kind::Unaddressable => base,
        }
    }

    /// Writes an entry unless it already says exactly this.
    ///
    /// Which is not merely an economy: storing an identical value would discard
    /// the accessed and dirty bits the processor has recorded there since, and
    /// answering a fault on a page something already described is exactly when
    /// that would happen.
    fn install(&self, at: Reached, wanted: u64) -> Result<(), NptError> {
        if entry::unchanged(at.value, wanted) {
            return Ok(());
        }
        walk::store(self.window, at.table, at.index, wanted)
    }

    /// The entry that describes `gpa` at the coarsest level at or below `level`
    /// the tree can hold it at, building every table above that entry which
    /// does not exist yet.
    ///
    /// *At or below*, because a table already on the path is a region something
    /// narrowed deliberately: the descent follows it and answers with the finer
    /// entry rather than writing over it.
    ///
    /// # Errors
    ///
    /// [`NptError::Coarser`] if a leaf covers the address above `level`,
    /// [`NptError::OutOfFrames`] if the chunk cannot spare a table, or
    /// [`NptError::Unreachable`] if the window does not reach one.
    fn descend(
        &self,
        frames: &mut Frames,
        gpa: PhysAddr,
        level: Level,
    ) -> Result<Reached, NptError> {
        let mut table = self.root;
        for above in Level::TABLES {
            let index = above.index(gpa);
            let value = walk::load(self.window, table, index)?;
            if entry::present(value) && !entry::leaf(value, above) {
                table = entry::frame(value);
                continue;
            }
            if above <= level {
                return Ok(Reached {
                    table,
                    index,
                    level: above,
                    value,
                });
            }
            if entry::present(value) {
                return Err(NptError::Coarser {
                    gpa: gpa.as_u64(),
                    level: above,
                });
            }
            table = self.link(frames, table, index)?;
        }
        // Every entry of a page table describes memory, so a descent that got
        // this far has arrived.
        self.entry_at(table, Level::Page, gpa)
    }

    /// The entry that describes `gpa` at whatever level the tree already holds
    /// it at, building nothing.
    ///
    /// The counterpart of [`Tree::descend`] for reading and for undoing: it
    /// allocates nothing and answers with the absent entry it stopped at, which
    /// is what makes it usable on a path that is giving something back. A call
    /// that had to allocate in order to clear an entry could fail for want of
    /// memory while releasing memory.
    ///
    /// # Errors
    ///
    /// [`NptError::Unreachable`] if the window does not reach one of these
    /// tables.
    fn find(&self, gpa: PhysAddr) -> Result<Reached, NptError> {
        let mut table = self.root;
        for above in Level::TABLES {
            let index = above.index(gpa);
            let value = walk::load(self.window, table, index)?;
            if !entry::present(value) || entry::leaf(value, above) {
                return Ok(Reached {
                    table,
                    index,
                    level: above,
                    value,
                });
            }
            table = entry::frame(value);
        }
        self.entry_at(table, Level::Page, gpa)
    }

    /// The entry of `table` that `gpa` indexes at this level, as a walk that
    /// stopped there answers with it.
    fn entry_at(&self, table: PhysAddr, level: Level, gpa: PhysAddr) -> Result<Reached, NptError> {
        let index = level.index(gpa);
        Ok(Reached {
            table,
            index,
            level,
            value: walk::load(self.window, table, index)?,
        })
    }

    /// The table below an entry that had none, allocated and linked.
    ///
    /// Every table this creates is zeroed by the allocator that handed out its
    /// frame, so an entry nothing has written yet reads as not present rather
    /// than as whatever the frame last held.
    fn link(
        &self,
        frames: &mut Frames,
        table: PhysAddr,
        index: PageTableIndex,
    ) -> Result<PhysAddr, NptError> {
        let frame = taken(frames)?;
        walk::store(self.window, table, index, encode(Entry::Table { frame }))?;
        Ok(frame)
    }

    /// Replaces a leaf with a table of the level below describing the same
    /// memory, and answers where that table is.
    ///
    /// # Why a walk in progress cannot see this half done
    ///
    /// The new table is filled completely before the entry above it is
    /// replaced, and that entry is then replaced by a single store of an
    /// aligned quadword, which the hardware page walker reads atomically.
    /// So a walk on another processor sees either the leaf or the finished
    /// table and never a table with entries still to be written: the store
    /// is a releasing one, which is what keeps the fill from being
    /// reordered after it, and stores here become visible in the order they
    /// are made in any case.
    fn divide(
        &self,
        frames: &mut Frames,
        at: Reached,
        gpa: PhysAddr,
    ) -> Result<PhysAddr, NptError> {
        let Some(below) = at.level.below() else {
            // A leaf at a level the architecture has no large page at is not
            // something this crate wrote, and is reported rather than split.
            return Err(NptError::Coarser {
                gpa: gpa.as_u64(),
                level: at.level,
            });
        };
        let frame = taken(frames)?;
        walk::store_each(self.window, frame, |slot| {
            Some(entry::narrowed(at.value, below, slot))
        })?;
        walk::store(
            self.window,
            at.table,
            at.index,
            encode(Entry::Table { frame }),
        )?;
        Ok(frame)
    }
}

/// The entry a walk stopped at.
#[derive(Clone, Copy, Debug)]
struct Reached {
    /// The table holding it.
    table: PhysAddr,
    /// Where in that table it is.
    index: PageTableIndex,
    /// What one entry of that table describes, which says how much of memory
    /// this one speaks for.
    level: Level,
    /// What it said when the walk read it.
    value: u64,
}

/// One frame of the chunk, for a table.
fn taken(frames: &mut Frames) -> Result<PhysAddr, NptError> {
    let frame = frames.allocate(0).map_err(|_| NptError::OutOfFrames)?;
    Ok(frame.start_address())
}

/// The level every page of the hypervisor's own memory is described at.
///
/// One page, because all of the chunk reads as a single shared frame and one
/// entry describes a *run* of frames rather than one repeated: at 2 MiB the
/// entry would describe 2 MiB of the chunk itself. So the chunk costs one page
/// table per 2 MiB of it — thirty-two of them, 128 KiB, for a 64 MiB chunk.
///
/// The alternative is a whole 2 MiB frame of zeroes and one large-page entry
/// per 2 MiB of the chunk: no page tables at all, and 2 MiB of the chunk spent
/// once for every tree that shares that frame. A loss at one tree and a win
/// past about sixteen of them, which is why the choice is written here rather
/// than assumed anywhere else.
const SHADOW: Level = Level::Page;

/// How much of the hypervisor's own memory one fill describes: everything one
/// table of [`SHADOW`] entries covers.
const SHADOWED: u64 = Level::Directory.span();

const _: () = assert!(
    as_usize(SHADOWED / SHADOW.span()) == walk::ENTRIES,
    "the hypervisor's own memory is described one whole table at a time, so what \
     one fill covers must be exactly what one table of its entries does",
);

#[cfg(test)]
mod tests {
    //! The level a fill chooses, what a split preserves, and what a second fill
    //! of the same address costs — over a run of host memory standing in for
    //! the reserved chunk.
    //!
    //! The verdicts are written out rather than resolved from a map, because
    //! what is under test is what the tree does with an answer and nothing
    //! about how that answer was arrived at.

    use paging::{Frames, chunk};
    use x86_64::PhysAddr;

    use super::{SHADOWED, Tree};
    use crate::{Access, Kind, Level, NptError, Translation, Verdict};

    /// One page, which is the floor of every description here.
    const PAGE: u64 = chunk::FRAME_SIZE;

    /// What one entry of the level above a page describes.
    const LARGE: u64 = 2 << 20;

    /// What one entry of the level above that describes.
    const HUGE: u64 = 1 << 30;

    /// Where the machine's memory begins for these tests: far enough above the
    /// run standing in for the chunk that a gigabyte of it holds nothing else.
    const RAM: u64 = 8 << 30;

    #[test]
    fn a_run_of_gigabytes_is_one_entry_where_the_processor_has_pages_that_large() {
        let (mut tree, mut frames, _) = empty(true);
        let gpa = PhysAddr::new(RAM + LARGE);

        tree.fill(&mut frames, open(RAM, 4 * HUGE), gpa)
            .expect("a run of gigabytes can be described");

        let translation = described(&tree, gpa);
        assert_eq!(
            translation.spa, gpa,
            "the guest's memory is the machine's own at the same address"
        );
        assert!(translation.writable, "and the guest may use it freely");
        assert_eq!(
            translation.span,
            HUGE - LARGE,
            "one entry describes the whole gigabyte, so the rest of it is behind the address"
        );
    }

    #[test]
    fn without_pages_that_large_the_same_run_is_described_two_megabytes_at_a_time() {
        let (mut tree, mut frames, _) = empty(false);
        let gpa = PhysAddr::new(RAM + LARGE);

        tree.fill(&mut frames, open(RAM, 4 * HUGE), gpa)
            .expect("a run of gigabytes can be described");

        assert_eq!(
            described(&tree, gpa).span,
            LARGE,
            "the coarsest page the processor reports is the coarsest one used"
        );
    }

    #[test]
    fn the_granularity_follows_the_run_rather_than_the_address() {
        let (mut tree, mut frames, _) = empty(true);
        // A run ending one page into the second 2 MiB of a gigabyte, which is the
        // shape of the map's answer either side of a page something else answers
        // for.
        let run = open(RAM, LARGE + PAGE);
        let whole = PhysAddr::new(RAM + PAGE);
        let part = PhysAddr::new(RAM + LARGE);

        tree.fill(&mut frames, run, whole)
            .expect("the 2 MiB the run covers is described at once");
        tree.fill(&mut frames, run, part)
            .expect("and the page past it on its own");

        assert_eq!(
            described(&tree, whole).span,
            LARGE - PAGE,
            "the 2 MiB the run covers whole is one entry"
        );
        assert_eq!(
            described(&tree, part).span,
            PAGE,
            "while the page the run stops after is an entry of its own"
        );
        assert_eq!(
            tree.translate(PhysAddr::new(RAM + LARGE + PAGE))
                .expect("the tables can be walked"),
            None,
            "and neither fill described anything past the run's end"
        );
    }

    #[test]
    fn splitting_a_large_page_changes_no_translation_it_replaces() {
        let (mut tree, mut frames, _) = empty(true);
        let inside = PhysAddr::new(RAM + LARGE + PAGE);
        tree.fill(&mut frames, open(RAM, HUGE), inside)
            .expect("a gigabyte of ordinary memory is one entry");
        let probes = [
            PhysAddr::new(RAM),
            PhysAddr::new(RAM + HUGE / 2),
            inside,
            PhysAddr::new(RAM + HUGE - PAGE),
        ];
        let before = probes.map(|gpa| described(&tree, gpa));

        tree.split(&mut frames, inside)
            .expect("the page can be broken out of the gigabyte");

        for (gpa, was) in probes.into_iter().zip(before) {
            let now = described(&tree, gpa);
            assert_eq!(
                (now.spa, now.writable),
                (was.spa, was.writable),
                "a split must change neither where {gpa:#x} is nor what may be done there"
            );
            // Only the 2 MiB holding the address the split was asked about ends up
            // described page by page. The rest of the gigabyte is one entry per
            // 2 MiB, and how far either kind of entry reaches is what is left of
            // its own region.
            let region = if gpa.align_down(LARGE) == inside.align_down(LARGE) {
                PAGE
            } else {
                LARGE
            };
            assert_eq!(
                now.span,
                region - gpa.as_u64() % region,
                "how far one entry reaches from {gpa:#x} once the split has been made"
            );
        }
    }

    #[test]
    fn a_fill_meeting_a_coarser_leaf_reports_it_rather_than_narrowing_the_tree() {
        let (mut tree, mut frames, _) = empty(true);
        let gpa = PhysAddr::new(RAM + PAGE);
        tree.fill(&mut frames, open(RAM, HUGE), gpa)
            .expect("a gigabyte of ordinary memory is one entry");
        let frame = spare(&mut frames);

        assert_eq!(
            tree.fill(
                &mut frames,
                Verdict {
                    base: gpa,
                    span: PAGE,
                    kind: Kind::Sink { spa: frame },
                },
                gpa
            ),
            Err(NptError::Coarser {
                gpa: gpa.as_u64(),
                level: Level::Pointer,
            }),
            "answering a fault may not break up a region: whatever narrowed it \
             was to have done that first"
        );
    }

    #[test]
    fn filling_the_same_address_twice_writes_the_same_entry_and_allocates_nothing() {
        let (mut tree, mut frames, _) = empty(true);
        let gpa = PhysAddr::new(RAM);
        let run = open(RAM, HUGE);
        tree.fill(&mut frames, run, gpa)
            .expect("a gigabyte of ordinary memory is one entry");
        let (was, free) = (described(&tree, gpa), frames.free());

        tree.fill(&mut frames, run, gpa)
            .expect("and describing it again is allowed");

        assert_eq!(
            frames.free(),
            free,
            "a fill that changes nothing allocates nothing"
        );
        assert_eq!(
            described(&tree, gpa),
            was,
            "and describes the same memory the same way"
        );
    }

    #[test]
    fn the_hypervisors_own_memory_is_described_a_whole_table_at_a_time() {
        let (mut tree, mut frames, zero) = empty(true);
        // The run standing in for the chunk begins at physical zero, so an
        // address inside it is an address of the hypervisor's own memory.
        let own = Verdict {
            base: PhysAddr::new(0),
            span: chunk::CHUNK_SIZE,
            kind: Kind::Shadow,
        };

        tree.fill(&mut frames, own, PhysAddr::new(SHADOWED + PAGE))
            .expect("the hypervisor's own memory can be described");

        for page in [SHADOWED, SHADOWED + PAGE, 2 * SHADOWED - PAGE] {
            let translation = described(&tree, PhysAddr::new(page));
            assert_eq!(
                translation.spa, zero,
                "every page of it reads as the one shared frame"
            );
            assert!(!translation.writable, "and none of them may be written");
            assert_eq!(
                translation.span, PAGE,
                "one frame repeated is one entry per page"
            );
        }
        assert_eq!(
            tree.translate(PhysAddr::new(2 * SHADOWED))
                .expect("the tables can be walked"),
            None,
            "and only the region the fault was in was described"
        );

        let free = frames.free();
        tree.fill(&mut frames, own, PhysAddr::new(SHADOWED + 2 * PAGE))
            .expect("a second fault in the same region can be answered");
        assert_eq!(
            frames.free(),
            free,
            "which costs nothing, the region being described already"
        );
    }

    #[test]
    fn a_page_given_a_frame_of_its_own_is_described_alone() {
        let (mut tree, mut frames, _) = empty(true);
        let page = PhysAddr::new(RAM + LARGE);
        let frame = spare(&mut frames);

        tree.fill(
            &mut frames,
            Verdict {
                base: page,
                span: PAGE,
                kind: Kind::Sink { spa: frame },
            },
            page,
        )
        .expect("a page the guest may write and nothing reads can be described");

        let translation = described(&tree, page);
        assert_eq!(
            (translation.spa, translation.writable, translation.span),
            (frame, true, PAGE),
            "the frame its writes are kept by, writable, and one page of it"
        );
        assert_eq!(
            tree.translate(page + PAGE)
                .expect("the tables can be walked"),
            None,
            "and its neighbour is left for the map to answer for"
        );
    }

    /// An empty tree over a run of host memory standing in for the chunk, the
    /// allocator its tables come from, and the frame the hypervisor's own
    /// memory reads as.
    fn empty(large: bool) -> (Tree, Frames, PhysAddr) {
        let (mut frames, window) = crate::tests::reserved();
        let root = spare(&mut frames);
        let zero = spare(&mut frames);
        (Tree::new(root, zero, window, large), frames, zero)
    }

    /// One frame of the run standing in for the chunk.
    fn spare(frames: &mut Frames) -> PhysAddr {
        super::taken(frames).expect("the run standing in for the chunk has frames")
    }

    /// The verdict for a run of the machine's own memory, which a guest sees at
    /// the addresses it really has.
    fn open(base: u64, bytes: u64) -> Verdict {
        Verdict {
            base: PhysAddr::new(base),
            span: bytes,
            kind: Kind::Ram {
                spa: PhysAddr::new(base),
                access: Access::all(),
            },
        }
    }

    /// What the tree says about an address it must already describe.
    fn described(tree: &Tree, gpa: PhysAddr) -> Translation {
        tree.translate(gpa)
            .expect("the tables can be walked")
            .expect("the address is described")
    }
}
