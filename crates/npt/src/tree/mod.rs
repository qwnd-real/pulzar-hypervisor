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
//!
//! # Two processors filling at once need no lock, and only for that reason
//!
//! Every entry is an atomic and a table is installed with a compare-exchange,
//! so the loser of a race is told so and follows the winner's table instead of
//! writing over it. That is sound *because* the level a fill chooses is a
//! function of the map and the address and of nothing else: two processors
//! describing the same region compute the same level, so they never disagree
//! about whether an entry is to be a table or a leaf, and the leaves they then
//! store are the same quadword. Take away the level rule and nothing here would
//! be safe to run twice at once.
//!
//! Nothing in this module takes an exclusive reference to anything, and what
//! keeps the operations that *narrow* the tree from racing a fill is the map's
//! lock, one layer up: a fill reads the map and a mutation writes it, so the
//! two cannot overlap.

mod entry;
pub(crate) mod walk;

use log::info;
use paging::{DirectMap, as_usize};
use x86_64::{PhysAddr, structures::paging::PageTableIndex};

use crate::{
    NptError, Translation,
    frames::FrameCache,
    map::{Kind, Verdict},
    tree::{
        entry::{Entry, encode},
        walk::Level,
    },
};

/// One guest's nested page tables, as the hardware walks them.
///
/// Holds no allocator and no lock. Table frames come from the cache handed in
/// per call, whose frames belong to the processor asking, and every entry is an
/// atomic — so describing a region needs nothing to be exclusive.
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
    /// Answers with what writing the entry came to, which is what a mutation
    /// needs to know whether it owes anything: an entry that already said this,
    /// or that described nothing at all, leaves no processor holding a
    /// translation these tables have stopped justifying.
    ///
    /// Idempotent, and cheaply so: an entry already saying this is left alone,
    /// which is what answering a fault on an address that is already described
    /// comes to — a write to memory the guest may only read faults every time
    /// it is retried.
    ///
    /// # Errors
    ///
    /// [`NptError::Coarser`] if a leaf already covers the address at a coarser
    /// level than the verdict allows, [`NptError::OutOfFrames`] if this
    /// processor has no frame left for a table, or [`NptError::Unreachable`] if
    /// the window does not reach one.
    pub(crate) fn fill(
        &self,
        frames: &FrameCache,
        verdict: Verdict,
        gpa: PhysAddr,
    ) -> Result<Written, NptError> {
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
    /// [`NptError::OutOfFrames`] if this processor has no frame left for a
    /// table, [`NptError::Unreachable`] if the window does not reach one, or
    /// [`NptError::Coarser`] if a leaf turns up at a level the architecture has
    /// no large page at, which is an entry this crate did not write.
    pub(crate) fn split(&self, frames: &FrameCache, gpa: PhysAddr) -> Result<(), NptError> {
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
    pub(crate) fn abandon(&self, gpa: PhysAddr) -> Result<Written, NptError> {
        let at = self.find(gpa)?;
        if !matches!(at.level, Level::Page) {
            return Ok(Written::Same);
        }
        self.install(at, encode(Entry::Absent))
    }

    /// Describes the `level` region containing `base` with one entry again,
    /// where the map has stopped asking for anything finer, and answers with
    /// the table that region no longer needs.
    ///
    /// `None` where nothing was done: the answer does not reach across the
    /// whole region, or the level is one the tree may not hold a leaf at
    /// here, or the region is already one entry, or a table below the one
    /// that would go still describes a region of its own more finely than a
    /// single leaf could.
    ///
    /// The leaf is stored and the table is only answered with — it is not
    /// freed. A processor caches the entries above a leaf as well as the
    /// leaf, so a frame handed back before the barrier this owes could be
    /// handed out for something else while a walk in progress is still
    /// reading it.
    ///
    /// One aligned quadword replaces the entry, which the hardware page walker
    /// reads atomically, so a walk sees either the table or the leaf — and both
    /// describe the same memory the same way, which is what makes the moment
    /// between them uninteresting.
    ///
    /// # Errors
    ///
    /// [`NptError::Unreachable`] if the window does not reach one of these
    /// tables.
    pub(crate) fn compact(
        &self,
        verdict: Verdict,
        level: Level,
        base: PhysAddr,
    ) -> Result<Option<PhysAddr>, NptError> {
        // The level rule is the authority on how coarsely a region may be
        // described, and a region it would describe more finely than this is one
        // whose table is still doing something.
        if level > self.coarsest(verdict, base) {
            return Ok(None);
        }
        let Some(at) = self.reached(level, base)? else {
            return Ok(None);
        };
        if !entry::present(at.value) || entry::leaf(at.value, level) {
            return Ok(None);
        }
        let table = entry::frame(at.value);
        if self.branching(table, level)? {
            return Ok(None);
        }
        walk::store(
            self.window,
            at.table,
            at.index,
            self.describes(verdict, level, base),
        )?;
        Ok(Some(table))
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
        &self,
        frames: &FrameCache,
        verdict: Verdict,
        gpa: PhysAddr,
    ) -> Result<Written, NptError> {
        let at = self.descend(frames, gpa, SHADOW)?;
        // One entry serves every page of the region, all of them reading as the
        // same frame. So the entry for the page this fault was for already
        // saying it means the whole region does — and a write to the
        // hypervisor's own memory, which faults every time the guest retries it,
        // must not rewrite five hundred and twelve entries each time.
        let wanted = self.describes(verdict, SHADOW, gpa);
        if entry::unchanged(at.value, wanted) {
            return Ok(Written::Same);
        }
        let first = gpa.align_down(SHADOWED);
        walk::store_each(self.window, at.table, |slot| {
            verdict
                .covers(first + slot * SHADOW.span(), SHADOW.span())
                .then_some(wanted)
        })?;
        Ok(written(at.value, wanted))
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

    /// Writes an entry unless it already says exactly this, and answers with
    /// what that came to.
    ///
    /// Leaving an identical value alone is not merely an economy: storing one
    /// would discard the accessed and dirty bits the processor has recorded
    /// there since, and answering a fault on a page something already
    /// described is exactly when that would happen.
    fn install(&self, at: Reached, wanted: u64) -> Result<Written, NptError> {
        if entry::unchanged(at.value, wanted) {
            return Ok(Written::Same);
        }
        walk::store(self.window, at.table, at.index, wanted)?;
        Ok(written(at.value, wanted))
    }

    /// Whether any entry of `table` names a table of its own rather than
    /// describing memory.
    ///
    /// What a compaction has to ask before it detaches a table: an entry naming
    /// another table below would be a whole subtree left behind, held by
    /// nothing and describing nothing. A page table can never contain one —
    /// every entry of one describes memory — so this only ever has anything
    /// to find one level up.
    fn branching(&self, table: PhysAddr, level: Level) -> Result<bool, NptError> {
        let Some(below) = level.below() else {
            // A level the architecture has no large page at is one this never
            // reaches: nothing above asks to describe a region at the root, and
            // the page table is where a walk arrives rather than descends from.
            return Ok(true);
        };
        walk::any(self.window, table, |value| {
            entry::present(value) && !entry::leaf(value, below)
        })
    }

    /// The entry that describes `gpa` at the coarsest level at or below `level`
    /// the tree can hold it at, building every table above that entry which
    /// does not exist yet.
    ///
    /// *At or below*, because a table already on the path is a region something
    /// narrowed deliberately: the descent follows it and answers with the finer
    /// entry rather than writing over it. A table another processor installs
    /// while this descent is on its way down is followed for the same reason
    /// and is not distinguishable from one that was always there.
    ///
    /// # Errors
    ///
    /// [`NptError::Coarser`] if a leaf covers the address above `level`,
    /// [`NptError::OutOfFrames`] if this processor has no frame left for a
    /// table, or [`NptError::Unreachable`] if the window does not reach one.
    fn descend(
        &self,
        frames: &FrameCache,
        gpa: PhysAddr,
        level: Level,
    ) -> Result<Reached, NptError> {
        let mut table = self.root;
        for above in Level::TABLES {
            let index = above.index(gpa);
            let mut value = walk::load(self.window, table, index)?;
            if !entry::present(value) && above > level {
                value = self.link(frames, table, index, value)?;
            }
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
            // A leaf where the descent asked for a table, which is either an
            // entry this crate did not write or a region narrowed without being
            // broken up first. Both are reported rather than repaired.
            return Err(NptError::Coarser {
                gpa: gpa.as_u64(),
                level: above,
            });
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

    /// The entry that describes the `level` region containing `gpa`, building
    /// nothing and descending no further.
    ///
    /// `None` where the walk cannot get that far, which is one answer for two
    /// shapes: an absent entry above `level`, and a leaf above it. Both mean
    /// the tree holds no table at `level` for this address, and so that
    /// there is nothing there to be replaced by one entry.
    ///
    /// # Errors
    ///
    /// [`NptError::Unreachable`] if the window does not reach one of these
    /// tables.
    fn reached(&self, level: Level, gpa: PhysAddr) -> Result<Option<Reached>, NptError> {
        let mut table = self.root;
        for above in Level::TABLES {
            let index = above.index(gpa);
            let value = walk::load(self.window, table, index)?;
            if above == level {
                return Ok(Some(Reached {
                    table,
                    index,
                    level: above,
                    value,
                }));
            }
            if !entry::present(value) || entry::leaf(value, above) {
                return Ok(None);
            }
            table = entry::frame(value);
        }
        // Every level a leaf may be coarsened to is one a table can name, so a
        // walk that got past all of them was asked about the page table, which no
        // entry names a table of.
        Ok(None)
    }

    /// The entry naming the table below one that named none.
    ///
    /// One of this processor's frames if the exchange is made, and whatever
    /// another processor put there if it got in first — in which case ours goes
    /// back to the list it came from and the descent follows the winner's
    /// table. Neither processor can tell afterwards which of them it was,
    /// and neither has to.
    ///
    /// Every table this creates is zeroed by the allocator that handed out its
    /// frame, so an entry nothing has written yet reads as not present rather
    /// than as whatever the frame last held.
    fn link(
        &self,
        frames: &FrameCache,
        table: PhysAddr,
        index: PageTableIndex,
        absent: u64,
    ) -> Result<u64, NptError> {
        let frame = frames.take()?;
        let wanted = encode(Entry::Table { frame });
        match walk::exchange(self.window, table, index, absent, wanted)? {
            None => Ok(wanted),
            Some(installed) => {
                frames.give(frame);
                Ok(installed)
            }
        }
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
    ///
    /// A store rather than an exchange, unlike the tables a fill installs: this
    /// replaces an entry that says something, and the only operations that
    /// replace one hold the map exclusively, so nothing else is writing it.
    fn divide(
        &self,
        frames: &FrameCache,
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
        let frame = frames.take()?;
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

/// What writing one entry came to.
///
/// The whole of what a mutation needs in order to know what it owes: only an
/// entry that described something else can have left a processor holding a
/// translation these tables have stopped justifying.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Written {
    /// The entry already said this, so nothing was stored.
    Same,
    /// It described nothing, so nothing was cached that could disagree with
    /// what it says now.
    Filled,
    /// It described something else, which a processor may still be acting on.
    Replaced,
}

impl Written {
    /// The one answer for both, which is whichever owes more.
    ///
    /// What a mutation over a run of pages comes to: the run owes whatever its
    /// most demanding page owes, and the order the variants are declared in is
    /// that ranking.
    pub(crate) fn and(self, other: Self) -> Self {
        self.max(other)
    }
}

/// What replacing `was` with `wanted` came to, for an entry that is about to be
/// or has just been stored.
fn written(was: u64, wanted: u64) -> Written {
    if entry::unchanged(was, wanted) {
        Written::Same
    } else if entry::present(was) {
        Written::Replaced
    } else {
        Written::Filled
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

    use super::{FrameCache, SHADOWED, Tree};
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
        let (tree, cache, ..) = empty(true);
        let gpa = PhysAddr::new(RAM + LARGE);

        tree.fill(&cache, open(RAM, 4 * HUGE), gpa)
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
        let (tree, cache, ..) = empty(false);
        let gpa = PhysAddr::new(RAM + LARGE);

        tree.fill(&cache, open(RAM, 4 * HUGE), gpa)
            .expect("a run of gigabytes can be described");

        assert_eq!(
            described(&tree, gpa).span,
            LARGE,
            "the coarsest page the processor reports is the coarsest one used"
        );
    }

    #[test]
    fn the_granularity_follows_the_run_rather_than_the_address() {
        let (tree, cache, ..) = empty(true);
        // A run ending one page into the second 2 MiB of a gigabyte, which is the
        // shape of the map's answer either side of a page something else answers
        // for.
        let run = open(RAM, LARGE + PAGE);
        let whole = PhysAddr::new(RAM + PAGE);
        let part = PhysAddr::new(RAM + LARGE);

        tree.fill(&cache, run, whole)
            .expect("the 2 MiB the run covers is described at once");
        tree.fill(&cache, run, part)
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
        let (tree, cache, ..) = empty(true);
        let inside = PhysAddr::new(RAM + LARGE + PAGE);
        tree.fill(&cache, open(RAM, HUGE), inside)
            .expect("a gigabyte of ordinary memory is one entry");
        let probes = [
            PhysAddr::new(RAM),
            PhysAddr::new(RAM + HUGE / 2),
            inside,
            PhysAddr::new(RAM + HUGE - PAGE),
        ];
        let before = probes.map(|gpa| described(&tree, gpa));

        tree.split(&cache, inside)
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
        let (tree, cache, mut frames, _) = empty(true);
        let gpa = PhysAddr::new(RAM + PAGE);
        tree.fill(&cache, open(RAM, HUGE), gpa)
            .expect("a gigabyte of ordinary memory is one entry");
        let frame = spare(&mut frames);

        assert_eq!(
            tree.fill(
                &cache,
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
        let (tree, cache, ..) = empty(true);
        let gpa = PhysAddr::new(RAM);
        let run = open(RAM, HUGE);
        tree.fill(&cache, run, gpa)
            .expect("a gigabyte of ordinary memory is one entry");
        let (was, held) = (described(&tree, gpa), cache.held());

        tree.fill(&cache, run, gpa)
            .expect("and describing it again is allowed");

        assert_eq!(
            cache.held(),
            held,
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
        let (tree, cache, _, zero) = empty(true);
        // The run standing in for the chunk begins at physical zero, so an
        // address inside it is an address of the hypervisor's own memory.
        let own = Verdict {
            base: PhysAddr::new(0),
            span: chunk::CHUNK_SIZE,
            kind: Kind::Shadow,
        };

        tree.fill(&cache, own, PhysAddr::new(SHADOWED + PAGE))
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

        let held = cache.held();
        tree.fill(&cache, own, PhysAddr::new(SHADOWED + 2 * PAGE))
            .expect("a second fault in the same region can be answered");
        assert_eq!(
            cache.held(),
            held,
            "which costs nothing, the region being described already"
        );
    }

    #[test]
    fn a_page_given_a_frame_of_its_own_is_described_alone() {
        let (tree, cache, mut frames, _) = empty(true);
        let page = PhysAddr::new(RAM + LARGE);
        let frame = spare(&mut frames);

        tree.fill(
            &cache,
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
    /// frames its tables come from, the allocator behind those, and the frame
    /// the hypervisor's own memory reads as.
    ///
    /// One list, which is what a machine nothing has surveyed has — and what
    /// makes every list here reachable without asking the processor which of
    /// them is its own.
    fn empty(large: bool) -> (Tree, FrameCache, Frames, PhysAddr) {
        let (mut frames, window) = crate::tests::reserved();
        let root = spare(&mut frames);
        let zero = spare(&mut frames);
        let cache = FrameCache::new(1);
        cache.stock(&mut frames);
        (Tree::new(root, zero, window, large), cache, frames, zero)
    }

    /// One frame of the run standing in for the chunk.
    fn spare(frames: &mut Frames) -> PhysAddr {
        frames
            .allocate(0)
            .expect("the run standing in for the chunk has frames")
            .start_address()
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
