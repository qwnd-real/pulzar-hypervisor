//! Reaching one entry of one nested page table, and the four levels a guest
//! physical address is walked down through.
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
//! # An entry is not this crate's to own
//!
//! The processor writes the accessed and dirty bits into these entries itself:
//! the architecture says so of every nested entry touched while walking a
//! guest's own page tables. Software is therefore not the only writer of one,
//! and the shape of this file follows from that.
//!
//! An entry is an [`AtomicU64`]: read by [`load`], written by [`store`], and
//! replaced-if-unchanged by [`exchange`]. Reading or writing one through an
//! ordinary reference would be a data race with the hardware, and a
//! read-modify-write through one could lose what the hardware recorded between
//! the read and the write — which is why the only operation here that both
//! reads and writes is one the processor performs indivisibly.
//!
//! One aligned eight-byte store is also what the hardware page walker reads
//! atomically, so a walk in progress sees the whole of the old entry or the
//! whole of the new one and never half of each. That is what lets a leaf be
//! replaced, and a table be published, without stopping anything.
//!
//! Nothing here takes an exclusive reference to a table, and a shared one is
//! enough to write an entry, precisely because every entry is an atomic. What
//! the hardware's two bits do cost is that no comparison of two entries may
//! include them.

use core::sync::atomic::{AtomicU64, Ordering};

use paging::{DirectMap, as_usize, chunk};
use x86_64::{PhysAddr, structures::paging::PageTableIndex};

use crate::NptError;

/// One of the four levels a guest physical address is walked through.
///
/// Named for the table rather than numbered, because a number only means
/// something together with a convention about which end it counts from. The
/// shift and the span are derived from the level rather than written out once
/// per level, so the three can never disagree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
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
    /// Every level whose entries can name a table, from the root downwards,
    /// which is the order a walk visits them in.
    ///
    /// The page table is the one that cannot: its entries describe memory
    /// whether or not they carry the bit that says so, which is why a walk
    /// reaching one has arrived rather than having somewhere further to go.
    pub(crate) const TABLES: [Self; 3] = [Self::Root, Self::Pointer, Self::Directory];

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
    pub(crate) const fn below(self) -> Option<Self> {
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

/// What one entry of the table at `at` says.
///
/// Acquiring, so that a walk which follows this entry to a table also sees
/// every entry of that table: the ordering pairs with the release store that
/// published it.
pub(crate) fn load(
    window: DirectMap,
    at: PhysAddr,
    index: PageTableIndex,
) -> Result<u64, NptError> {
    Ok(reach(window, at)?.entries[usize::from(index)].load(Ordering::Acquire))
}

/// Makes one entry of the table at `at` say `value`.
///
/// Releasing, so that whatever was written before it — the entries of a table
/// this one is about to name — is visible to any walk that follows it.
pub(crate) fn store(
    window: DirectMap,
    at: PhysAddr,
    index: PageTableIndex,
    value: u64,
) -> Result<(), NptError> {
    reach(window, at)?.entries[usize::from(index)].store(value, Ordering::Release);
    Ok(())
}

/// Makes one entry of the table at `at` say `value`, but only while it still
/// says `was`.
///
/// Answers with what the entry says instead, or `None` if the exchange was
/// made. This is the one operation two processors can be performing on one
/// entry at once — both describing an address whose region has no table yet —
/// and it is what tells the loser that the table it built is not the one below
/// this entry.
///
/// Acquiring on failure and releasing on success, for the reasons [`load`] and
/// [`store`] have: the loser goes on to read the winner's table, and the winner
/// has already written its own.
pub(crate) fn exchange(
    window: DirectMap,
    at: PhysAddr,
    index: PageTableIndex,
    was: u64,
    value: u64,
) -> Result<Option<u64>, NptError> {
    Ok(reach(window, at)?.entries[usize::from(index)]
        .compare_exchange(was, value, Ordering::AcqRel, Ordering::Acquire)
        .err())
}

/// Makes each entry of the table at `at` say what `value` answers for its slot,
/// leaving the entries it declines to answer for as they are.
///
/// One reach through the window for a whole table rather than five hundred and
/// twelve of them, which is what the two callers that describe a table at a
/// time need: the shadow over the hypervisor's own memory, and the table a
/// split fills before it publishes it. The slot a value is asked for is the
/// index of the entry, so a caller can turn it into the addresses that entry
/// describes.
pub(crate) fn store_each(
    window: DirectMap,
    at: PhysAddr,
    value: impl Fn(u64) -> Option<u64>,
) -> Result<(), NptError> {
    for (entry, slot) in reach(window, at)?.entries.iter().zip(0u64..) {
        if let Some(value) = value(slot) {
            entry.store(value, Ordering::Release);
        }
    }
    Ok(())
}

/// One nested page table.
///
/// Page aligned and page sized, as the frame it occupies is and as the hardware
/// requires: the alignment is what the window checks before it hands out a
/// pointer to one, so a table reached through the window is a table the
/// processor could have reached itself.
#[repr(C, align(4096))]
struct Table {
    /// The entries, indexed by the nine bits of an address that name one.
    entries: [AtomicU64; ENTRIES],
}

/// Entries in one table, which is what nine bits of an address can name.
pub(crate) const ENTRIES: usize = 512;

/// The table at `at`, through the window onto physical memory.
///
/// Shared rather than exclusive even where an entry is about to be written,
/// because the processor writes into the same quadwords: an exclusive reference
/// would be a claim that nothing else touches them, and that claim is false of
/// every table here.
fn reach<'a>(window: DirectMap, at: PhysAddr) -> Result<&'a Table, NptError> {
    let table = window
        .ptr::<Table>(at)
        .map_err(|_| NptError::Unreachable { phys: at.as_u64() })?;
    // SAFETY: `ptr` proved that the window reaches all four kilobytes of the
    // table and that `at` is aligned for one, so the pointer is valid for a
    // whole `Table`. The frame behind it was handed out zeroed by the chunk's
    // allocator and has held nothing but entries since, so every quadword is
    // initialised. The other writer of those quadwords is the processor's own
    // page walker recording accessed and dirty bits, which a shared reference
    // tolerates precisely because each entry is an `AtomicU64` — and the
    // reference outlives nothing, every caller being in this module and done
    // with it before it returns.
    Ok(unsafe { table.as_ref() })
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
const _: () = assert!(
    ENTRIES == as_usize(Level::Directory.span() / Level::Page.span()),
    "a table must hold one entry per region of the level below that its own \
     entries describe",
);
const _: () = assert!(
    size_of::<Table>() == as_usize(chunk::FRAME_SIZE),
    "a table must be exactly the frame it occupies, which is what makes the \
     window's own alignment check the hardware's requirement",
);
