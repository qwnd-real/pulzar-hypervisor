//! Two levels of four-byte entries.
//!
//! The oldest translation there is, and still where a processor lands when a
//! guest turns paging on before it turns anything else on. Ten bits of the
//! address index each of the two levels and twelve are an offset within a page.
//!
//! # The four-megabyte page is where this goes wrong
//!
//! A page directory entry may describe its whole four megabytes, but only if
//! the guest asked for that ability — without the size extension the bit that
//! would say so is reserved, and a guest that set it anyway gets a walk that
//! carries on to the table below, which is what the hardware does with it.
//!
//! Its base is not one field. Four bytes cannot hold a forty-bit address beside
//! everything else in an entry, so the address is split: bits 31:22 lie where
//! they would in any entry, and bits 39:32 sit eight positions further down
//! with a reserved bit between the two runs. Reading the entry's upper bits
//! across as though they were a single field gives a page in the right place on
//! a machine with four gigabytes of memory and the wrong place on every larger
//! one, which is a bug that hides until the machine grows.

use crate::{
    Addressing, MemoryError, Physical,
    walk::{Mapped, PAGE, PAGE_SHIFT, entry32, index10, region},
};

/// Bits of a four-byte entry that are the address of a page or a table.
const ADDRESS: u32 = 0xFFFF_F000;

/// Bit of a page directory entry that says it describes four megabytes.
const LARGE: u32 = 1 << 7;

/// Where the ten bits indexing the page directory sit, and the bytes an entry
/// of it describes when it describes a region rather than a table.
const DIRECTORY_SHIFT: u32 = 22;

/// Where a linear address lands, walking from this guest's root.
pub(super) fn walk(
    physical: Physical<'_>,
    addressing: &Addressing,
    linear: u64,
) -> Result<Mapped, MemoryError> {
    let entry = entry32(
        physical,
        addressing.root() & u64::from(ADDRESS),
        index10(linear, DIRECTORY_SHIFT),
        linear,
    )?;
    if addressing.large_pages() && entry & LARGE != 0 {
        return Ok(region(wide(entry), linear, DIRECTORY_SHIFT));
    }
    let entry = entry32(
        physical,
        u64::from(entry & ADDRESS),
        index10(linear, PAGE_SHIFT),
        linear,
    )?;
    Ok(region(u64::from(entry & ADDRESS), linear, PAGE_SHIFT))
}

/// The base of the four megabytes a page directory entry describes, out of the
/// two separated runs of bits it is kept in.
fn wide(entry: u32) -> u64 {
    /// Bits of the entry that are physical address bits 31:22, where they lie.
    const LOW: u32 = 0xFFC0_0000;
    /// Where physical address bits 39:32 are kept instead.
    const HIGH: u32 = 13;
    /// How many of them there are.
    const HIGH_BITS: u32 = 0xFF;
    u64::from(entry & LOW) | (u64::from((entry >> HIGH) & HIGH_BITS) << 32)
}

const _: () = assert!(
    1 << DIRECTORY_SHIFT == 4 * (1 << 20) && PAGE == 4096,
    "a page directory entry describes four megabytes and a page table entry four kilobytes",
);
