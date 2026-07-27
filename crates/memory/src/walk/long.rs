//! Four levels of eight-byte entries.
//!
//! What a 64-bit guest translates through, and by far the common case. Nine
//! bits of the address index each level, twelve are an offset within a page,
//! and the seventeen above those are the sign extension the walk never looks at
//! — the caller has already refused an address whose upper bits are not that.
//!
//! An entry of either middle level may describe its whole region instead of the
//! table below it, which is what a two-megabyte and a one-gigabyte page are.
//! The top level may not: the architecture has no
//! five-hundred-and-twelve-gigabyte page, and the bit that would say so is
//! reserved there.

use crate::{
    MemoryError, Physical,
    walk::{LARGE, Mapped, PAGE_SHIFT, entry64, index9, leaf},
};

/// The levels above the bottom one: where each indexes the address, and whether
/// an entry there may describe its whole region.
const ABOVE: [(u32, bool); 3] = [(39, false), (30, true), (21, true)];

/// Where a linear address lands, walking from this root.
pub(super) fn walk(physical: Physical<'_>, cr3: u64, linear: u64) -> Result<Mapped, MemoryError> {
    let mut table = cr3 & super::ADDRESS;
    for (shift, large) in ABOVE {
        let entry = entry64(physical, table, index9(linear, shift), linear)?;
        if large && entry & LARGE != 0 {
            return Ok(leaf(entry, linear, shift));
        }
        table = entry & super::ADDRESS;
    }
    // The bottom table's entries describe one page each and nothing else, so
    // reaching one is the end of the walk whatever it says.
    let entry = entry64(physical, table, index9(linear, PAGE_SHIFT), linear)?;
    Ok(leaf(entry, linear, PAGE_SHIFT))
}
