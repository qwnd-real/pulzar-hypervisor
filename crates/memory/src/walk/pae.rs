//! Four pointers, then two levels of eight-byte entries.
//!
//! The page-address extension is what let a thirty-two-bit guest reach physical
//! memory above four gigabytes, and it did it by widening every entry to eight
//! bytes — which halves how many fit a table, which is why a third level
//! appears above the two that were there before.
//!
//! That third level is the odd one. It has four entries rather than five
//! hundred and twelve, it is thirty-two bytes rather than a page, and the
//! register that points at it is aligned to those thirty-two bytes rather than
//! to a page. Its entries hold a present bit and an address and nothing else:
//! there is no region for one of them to describe, so nothing here tests them
//! for one.

use crate::{
    MemoryError, Physical,
    walk::{ADDRESS, LARGE, Mapped, PAGE_SHIFT, entry64, index9, leaf},
};

/// Bits of the address of the four pointers, which are aligned to their own
/// thirty-two bytes rather than to a page.
const POINTERS: u64 = 0xFFFF_FFE0;

/// Where the two bits selecting one of the four pointers sit.
const POINTER_SHIFT: u32 = 30;

/// Where the nine bits indexing the page directory sit, and the bytes an entry
/// of it describes when it describes a region rather than a table.
const DIRECTORY_SHIFT: u32 = 21;

/// Where a linear address lands, walking from this root.
pub(super) fn walk(physical: Physical<'_>, cr3: u64, linear: u64) -> Result<Mapped, MemoryError> {
    let pointer = entry64(
        physical,
        cr3 & POINTERS,
        (linear >> POINTER_SHIFT) & 0x3,
        linear,
    )?;
    let entry = entry64(
        physical,
        pointer & ADDRESS,
        index9(linear, DIRECTORY_SHIFT),
        linear,
    )?;
    if entry & LARGE != 0 {
        return Ok(leaf(entry, linear, DIRECTORY_SHIFT));
    }
    let entry = entry64(
        physical,
        entry & ADDRESS,
        index9(linear, PAGE_SHIFT),
        linear,
    )?;
    Ok(leaf(entry, linear, PAGE_SHIFT))
}
