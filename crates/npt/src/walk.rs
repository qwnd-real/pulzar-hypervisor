//! Descending a guest physical address through four levels of table.
//!
//! The entries are ordinary long-mode page table entries — the second
//! translation uses whatever paging mode the hypervisor was running in when it
//! entered the guest, which here is always 4-level long mode — so the entry
//! format is [`x86_64`]'s and is reused rather than restated.
//!
//! What is not reused is the walker. Everything in [`x86_64`] that walks a
//! table is typed in [`VirtAddr`](x86_64::VirtAddr) and
//! [`Page`](x86_64::structures::paging::Page), and a guest physical address is
//! neither: it is bounded by the processor's physical address width rather than
//! by canonicality, so an address at or above the canonical boundary makes
//! `VirtAddr::new` panic. A panic is not an acceptable answer to a guest
//! touching a high address, so the descent is written here instead — a few
//! dozen lines that index tables straight out of the address.
//!
//! # A large page is split only when it is asked for
//!
//! A descent says what it wants done with a large page it meets where it
//! expected a table, and the two answers exist for two different callers.
//!
//! [`Meeting::Refuse`] reports [`NptError::LargePage`]. The fill rule this
//! crate follows never asks for a granularity finer than the one it already
//! described a region with, so a filling descent that meets one means the rule
//! was broken — which is worth hearing about, and is not something to paper
//! over by rewriting a mapping the guest may already be running on.
//!
//! [`Meeting::Split`] breaks it into a table of the next level down describing
//! the same memory the same way, which is what trapping part of a region
//! requires: a 4 KiB entry cannot be given different permissions from its
//! neighbours while a single entry above them describes all five hundred and
//! twelve. Nothing about the translation changes, only how finely it is written
//! down.

use core::{
    ptr::NonNull,
    sync::atomic::{Ordering, compiler_fence},
};

use paging::{DirectMap, Frames};
use x86_64::{
    PhysAddr,
    structures::paging::{PageTable, PageTableFlags, PageTableIndex, page_table::PageTableEntry},
};

use crate::NptError;

/// Flags every table above a leaf carries.
///
/// User access is not a choice: a guest's own page-table walks are performed as
/// *user writes* at the nested level, so a table that is not both
/// user-accessible and writable turns every such walk into a fault. Nothing
/// here is ever marked no-execute, because the host has `EFER.NXE` set and the
/// bit would then deny the guest execution of everything below it.
pub(crate) const PARENT: PageTableFlags = PageTableFlags::PRESENT
    .union(PageTableFlags::WRITABLE)
    .union(PageTableFlags::USER_ACCESSIBLE);

/// One of the four levels a guest physical address is walked through.
///
/// Named for the table rather than numbered, because a number only means
/// something together with a convention about which end it counts from. The
/// shift and the span are derived from the level rather than written out once
/// per level, so the three can never disagree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Level {
    /// The page table. Each entry describes 4 KiB, and no entry of one is ever
    /// a large page.
    Page = 1,
    /// The page directory. Each entry describes 2 MiB.
    Directory = 2,
    /// The page-directory-pointer table. Each entry describes 1 GiB.
    Pointer = 3,
    /// The table `nCR3` names.
    Root = 4,
}

impl Level {
    /// Every level, from the root downwards, which is the order a walk visits
    /// them in.
    const ALL: [Self; 4] = [Self::Root, Self::Pointer, Self::Directory, Self::Page];

    /// Bits of a guest physical address below this level's index.
    const fn shift(self) -> u32 {
        /// Bits of an address that are an offset within the smallest page.
        const PAGE: u32 = 12;
        /// Bits of an address that index one table.
        const INDEX: u32 = 9;
        PAGE + INDEX * (self as u32 - 1)
    }

    /// Bytes one entry of this level describes.
    pub(crate) const fn span(self) -> u64 {
        1 << self.shift()
    }

    /// The level whose entries describe a slice of what one entry of this level
    /// describes, for the two levels a large page can appear at.
    ///
    /// `None` at the other two, and for opposite reasons: a page table's
    /// entries are already the smallest the architecture has, and the root
    /// has no large page at all. Both answers mean the same thing to a
    /// caller holding an entry it believed was one — that the entry is not
    /// something this crate wrote.
    const fn below(self) -> Option<Self> {
        Some(match self {
            Self::Pointer => Self::Directory,
            Self::Directory => Self::Page,
            Self::Root | Self::Page => return None,
        })
    }

    /// Where `gpa` indexes a table at this level.
    pub(crate) fn index(self, gpa: PhysAddr) -> PageTableIndex {
        /// The nine bits of an address that are one table index.
        const MASK: u64 = 0x1FF;
        // Masked before it is narrowed, so nothing is truncated: nine bits fit
        // an index with room to spare, and `new_truncate` has nothing left to
        // truncate either.
        PageTableIndex::new_truncate(((gpa.as_u64() >> self.shift()) & MASK) as u16)
    }
}

/// Describes the `level`-sized region containing `gpa` as `frame`, building
/// every table above it that does not exist yet.
///
/// `flags` is the leaf's own; the large-page bit is added here rather than by
/// the caller, because whether an entry is a leaf is what `level` already says
/// and stating it twice is how the two come to disagree.
pub(crate) fn map(
    window: DirectMap,
    root: PhysAddr,
    gpa: PhysAddr,
    level: Level,
    frame: PhysAddr,
    flags: PageTableFlags,
    frames: &mut Frames,
) -> Result<(), NptError> {
    let leaf = if level == Level::Page {
        flags
    } else {
        flags.union(PageTableFlags::HUGE_PAGE)
    };
    let mut table = descend(window, root, gpa, level, frames, Meeting::Refuse)?;
    // SAFETY: `descend` returns a table of these nested tables, reached through
    // the window, and the caller holds the only handle to them.
    let table = unsafe { table.as_mut() };
    table[level.index(gpa)].set_addr(frame, leaf);
    Ok(())
}

/// What a descent does with a large page it meets where it wanted a table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Meeting {
    /// Report it, because meeting one means the fill rule was broken.
    Refuse,
    /// Break it into a table of the next level down describing the same memory.
    Split,
}

/// The entry that describes a guest physical address.
///
/// The level is part of the answer rather than an implementation detail: it is
/// what says how much of physical memory this one entry speaks for, and so how
/// far past the address the same translation continues.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Leaf {
    /// Where the region this entry describes begins in system physical memory.
    pub(crate) frame: PhysAddr,
    /// What the entry permits.
    pub(crate) flags: PageTableFlags,
    /// The level the entry was found at.
    pub(crate) level: Level,
}

/// The entry describing `gpa`, or `None` if nothing does.
///
/// A guest physical address can fault while a translation exists — a write to a
/// read-only page is exactly that — so this answers what describes the address,
/// not whether the access that faulted would now succeed.
pub(crate) fn lookup(
    window: DirectMap,
    root: PhysAddr,
    gpa: PhysAddr,
) -> Result<Option<Leaf>, NptError> {
    let mut table = root;
    let mut found = None;
    for level in Level::ALL {
        let (flags, frame) = read(window, table, level.index(gpa))?;
        if !flags.contains(PageTableFlags::PRESENT) {
            return Ok(None);
        }
        // A large page is a leaf wherever it appears, and every entry of the
        // bottom table is one whether or not it carries the bit that says so.
        if flags.contains(PageTableFlags::HUGE_PAGE) || level == Level::Page {
            found = Some(Leaf {
                frame,
                flags,
                level,
            });
            break;
        }
        table = frame;
    }
    Ok(found)
}

/// The table whose entries describe `level`-sized regions on the path to `gpa`,
/// or `None` if no such table exists.
///
/// The counterpart of [`descend`] for undoing rather than building: it
/// allocates nothing, splits nothing, and answers `None` where `descend` would
/// have made a table. That is what makes it usable on a path that is giving
/// something back — there is nothing to give back where nothing was ever built,
/// and a call that allocated in order to clear an entry could fail for want of
/// memory while releasing memory.
///
/// A large page on the way down is reported as `None` for the same reason: this
/// crate only ever traps at 4 KiB, so an address covered by a large page was
/// never trapped page by page, and splitting one here would be building rather
/// than undoing.
pub(crate) fn table_of(
    window: DirectMap,
    root: PhysAddr,
    gpa: PhysAddr,
    level: Level,
) -> Result<Option<NonNull<PageTable>>, NptError> {
    let mut table = root;
    for above in Level::ALL {
        if above <= level {
            break;
        }
        let (flags, frame) = read(window, table, above.index(gpa))?;
        if !flags.contains(PageTableFlags::PRESENT) || flags.contains(PageTableFlags::HUGE_PAGE) {
            return Ok(None);
        }
        table = frame;
    }
    reach(window, table).map(Some)
}

/// The table whose entries describe `level`-sized regions on the path to `gpa`,
/// building every table above it that does not exist yet.
///
/// Every table this creates is zeroed by the allocator that handed out its
/// frame, so an entry nothing has written yet reads as not present rather than
/// as whatever the frame last held.
pub(crate) fn descend(
    window: DirectMap,
    root: PhysAddr,
    gpa: PhysAddr,
    level: Level,
    frames: &mut Frames,
    meeting: Meeting,
) -> Result<NonNull<PageTable>, NptError> {
    let mut table = root;
    for above in Level::ALL {
        if above <= level {
            break;
        }
        table = child(window, table, above, gpa, frames, meeting)?;
    }
    reach(window, table)
}

/// The table below the entry `gpa` indexes at this level, allocating and
/// linking one if the entry is empty and splitting it if it is a large page.
fn child(
    window: DirectMap,
    table: PhysAddr,
    level: Level,
    gpa: PhysAddr,
    frames: &mut Frames,
    meeting: Meeting,
) -> Result<PhysAddr, NptError> {
    let mut table = reach(window, table)?;
    // SAFETY: `reach` returns a table of these nested tables, reached through
    // the window, and the caller holds the only handle to them.
    let entry = &mut unsafe { table.as_mut() }[level.index(gpa)];
    if entry.flags().contains(PageTableFlags::PRESENT) {
        if entry.flags().contains(PageTableFlags::HUGE_PAGE) {
            return match meeting {
                Meeting::Refuse => Err(NptError::LargePage { gpa: gpa.as_u64() }),
                Meeting::Split => split(window, entry, level, gpa, frames),
            };
        }
        return Ok(entry.addr());
    }
    let frame = frames
        .allocate(0)
        .map_err(|_| NptError::OutOfFrames)?
        .start_address();
    entry.set_addr(frame, PARENT);
    Ok(frame)
}

/// Replaces a large page with a table of the next level down describing the
/// same memory, and answers where that table is.
///
/// Nothing about the translation changes. Every entry of the new table
/// describes its slice of what the one entry described, with the same
/// permissions and the same memory type, so an address translated before the
/// split translates to the same place after it.
///
/// # Why a walk in progress cannot see this half done
///
/// The new table is filled completely before the entry above it is touched, and
/// that entry is then replaced by a single store of an aligned quadword, which
/// the hardware page walker reads atomically. So another processor walking this
/// path sees either the large page or the finished table and never a table with
/// entries still to be written. The fence is what stops the compiler from
/// hoisting that store above the fill; the processor will not reorder the two
/// on its own, because stores here become visible in the order they are made.
fn split(
    window: DirectMap,
    entry: &mut PageTableEntry,
    level: Level,
    gpa: PhysAddr,
    frames: &mut Frames,
) -> Result<PhysAddr, NptError> {
    let Some(below) = level.below() else {
        // A large page at a level the architecture has none at is not something
        // this crate wrote, and is reported as what it is rather than split.
        return Err(NptError::LargePage { gpa: gpa.as_u64() });
    };
    let leaf = if below == Level::Page {
        entry.flags().difference(PageTableFlags::HUGE_PAGE)
    } else {
        entry.flags()
    };
    let frame = frames
        .allocate(0)
        .map_err(|_| NptError::OutOfFrames)?
        .start_address();
    let mut table = reach(window, frame)?;
    // SAFETY: the frame was just handed out by the chunk's allocator, so nothing
    // else holds it, and `reach` proved the window describes it.
    let table = unsafe { table.as_mut() };
    let mut describes = entry.addr();
    for slot in table.iter_mut() {
        slot.set_addr(describes, leaf);
        describes += below.span();
    }
    compiler_fence(Ordering::Release);
    entry.set_addr(frame, PARENT);
    Ok(frame)
}

/// The flags and address of one entry.
fn read(
    window: DirectMap,
    table: PhysAddr,
    index: PageTableIndex,
) -> Result<(PageTableFlags, PhysAddr), NptError> {
    let table = reach(window, table)?;
    // SAFETY: as in `child`. Reading an entry cannot observe a partly written
    // one: an entry is an aligned eight-byte value and every writer here stores
    // it in a single instruction.
    let entry = &unsafe { table.as_ref() }[index];
    Ok((entry.flags(), entry.addr()))
}

/// Where a table is readable, through the window onto physical memory.
fn reach(window: DirectMap, table: PhysAddr) -> Result<NonNull<PageTable>, NptError> {
    window
        .ptr::<PageTable>(table)
        .map_err(|_| NptError::Unreachable {
            phys: table.as_u64(),
        })
}

const _: () = assert!(
    Level::Page.span() == 4096
        && Level::Directory.span() == 2 << 20
        && Level::Pointer.span() == 1 << 30,
    "each level must describe the page size the architecture gives it",
);
const _: () = assert!(
    Level::Root.span() == 512 << 30,
    "the root table's entries must each describe five hundred and twelve gigabytes",
);
