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
//! # Nothing here splits a large page
//!
//! A descent that meets a large page where it expected a table reports
//! [`NptError::LargePage`] rather than breaking the page up. The fill rule this
//! crate follows never asks for a granularity finer than the one it already
//! described a region with, so reaching that case means the rule was broken —
//! which is worth hearing about, and is not something to paper over by
//! rewriting a mapping the guest may already be running on.

use core::ptr::NonNull;

use paging::{DirectMap, Frames};
use x86_64::{
    PhysAddr,
    structures::paging::{PageTable, PageTableFlags, PageTableIndex},
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

    /// Where `gpa` indexes a table at this level.
    fn index(self, gpa: PhysAddr) -> PageTableIndex {
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
    let mut table = descend(window, root, gpa, level, frames)?;
    // SAFETY: `descend` returns a table of these nested tables, reached through
    // the window, and the caller holds the only handle to them.
    let table = unsafe { table.as_mut() };
    table[level.index(gpa)].set_addr(frame, leaf);
    Ok(())
}

/// Whether `gpa` already translates to anything.
///
/// A guest physical address can fault while a translation exists — a write to a
/// read-only page is exactly that — so this answers whether one exists, not
/// whether the access that faulted would now succeed.
pub(crate) fn translated(
    window: DirectMap,
    root: PhysAddr,
    gpa: PhysAddr,
) -> Result<bool, NptError> {
    let mut table = root;
    for level in Level::ALL {
        let (flags, addr) = read(window, table, level.index(gpa))?;
        if !flags.contains(PageTableFlags::PRESENT) {
            return Ok(false);
        }
        if flags.contains(PageTableFlags::HUGE_PAGE) {
            return Ok(true);
        }
        table = addr;
    }
    // Every level was present and none of them was a large page, so the last
    // one read was an entry of the bottom table: a 4 KiB translation.
    Ok(true)
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
) -> Result<NonNull<PageTable>, NptError> {
    let mut table = root;
    for above in Level::ALL {
        if above <= level {
            break;
        }
        table = child(window, table, above.index(gpa), gpa, frames)?;
    }
    reach(window, table)
}

/// The table below this entry, allocating and linking one if the entry is
/// empty.
fn child(
    window: DirectMap,
    table: PhysAddr,
    index: PageTableIndex,
    gpa: PhysAddr,
    frames: &mut Frames,
) -> Result<PhysAddr, NptError> {
    let mut table = reach(window, table)?;
    // SAFETY: `reach` returns a table of these nested tables, reached through
    // the window, and the caller holds the only handle to them.
    let entry = &mut unsafe { table.as_mut() }[index];
    if entry.flags().contains(PageTableFlags::PRESENT) {
        if entry.flags().contains(PageTableFlags::HUGE_PAGE) {
            return Err(NptError::LargePage { gpa: gpa.as_u64() });
        }
        return Ok(entry.addr());
    }
    let frame = frames
        .allocate(0)
        .ok_or(NptError::OutOfFrames)?
        .start_address();
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
    window.ptr::<PageTable>(table).ok_or(NptError::Unreachable {
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
