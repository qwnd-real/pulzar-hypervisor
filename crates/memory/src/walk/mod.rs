//! Walking a guest's own page tables.
//!
//! Four shapes of translation, one answer. Which shape applies is
//! [`Mode`]'s to say and the modules below are one apiece, sharing the pieces
//! that really are shared — reading an entry, and what an entry that describes
//! a region says about an address inside it — and nothing that is not.
//!
//! # A guest's tables are read like all its other memory
//!
//! Every entry read here goes through [`Physical`], so a guest's page tables
//! are translated by the nested tables exactly like everything else the guest
//! owns. Nothing assumes the identity relation those tables happen to describe
//! today: a guest page table is at a guest physical address, and what that
//! means is not this crate's to decide.
//!
//! # Nothing here checks a permission
//!
//! An entry is read for where it points and for whether it is present, and its
//! writability, privilege and no-execute bits are not consulted. The reasoning
//! is in the crate's own documentation; the short of it is that the processor
//! has already walked these same tables for this same access, and a second
//! opinion formed here could only contradict it.

mod legacy;
mod long;
mod pae;

use x86_64::PhysAddr;

use crate::{Addressing, MemoryError, Mode, Physical};

/// Where a guest linear address lands, and how far the entry that put it there
/// reaches past it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Mapped {
    /// The guest physical address.
    pub(crate) gpa: PhysAddr,
    /// Bytes from `gpa` still inside the same guest page.
    pub(crate) span: u64,
}

/// Where a guest linear address lands.
///
/// # Errors
///
/// [`MemoryError::NonCanonical`] for an address no 64-bit walk is defined for,
/// [`MemoryError::FiveLevelGuest`] for a guest translating through five levels,
/// [`MemoryError::Untranslated`] if the guest's own tables do not describe the
/// address, or anything [`Physical::read`] reports while the tables are read.
pub(crate) fn walk(
    physical: Physical<'_>,
    addressing: &Addressing,
    linear: u64,
) -> Result<Mapped, MemoryError> {
    match addressing.mode() {
        Mode::Unpaged => Ok(untranslated(linear)),
        // A guest not in long mode computes addresses thirty-two bits wide, and
        // anything above that in a value handed here is not part of the address.
        Mode::Legacy => legacy::walk(physical, addressing, linear & LOW),
        Mode::Pae => pae::walk(physical, addressing.root(), linear & LOW),
        Mode::Long if !canonical(linear) => Err(MemoryError::NonCanonical { linear }),
        Mode::Long => long::walk(physical, addressing.root(), linear),
        Mode::FiveLevel => Err(MemoryError::FiveLevelGuest),
    }
}

/// Bit of an entry that says it describes anything at all.
pub(super) const PRESENT: u64 = 1;

/// Bit of an entry above the bottom level that says it describes a region
/// rather than the table below it.
///
/// Only above the bottom level. In an entry of the lowest table the same
/// position is a memory-type bit, so an entry read there and tested for this
/// would be a page mistaken for a region on the strength of its cacheability.
pub(super) const LARGE: u64 = 1 << 7;

/// Bits of an eight-byte entry that are a physical address.
pub(super) const ADDRESS: u64 = 0x000F_FFFF_FFFF_F000;

/// Bits of a linear address that are an offset within the smallest page.
pub(super) const PAGE_SHIFT: u32 = 12;

/// Bytes in the smallest page.
pub(super) const PAGE: u64 = 1 << PAGE_SHIFT;

/// Bits of a linear address that survive when the guest is not in long mode.
const LOW: u64 = 0xFFFF_FFFF;

/// One eight-byte entry of a guest's tables, provided it describes anything.
pub(super) fn entry64(
    physical: Physical<'_>,
    table: u64,
    index: u64,
    linear: u64,
) -> Result<u64, MemoryError> {
    let mut bytes = [0; 8];
    physical.read(PhysAddr::new_truncate(table + index * 8), &mut bytes)?;
    let entry = u64::from_le_bytes(bytes);
    (entry & PRESENT != 0)
        .then_some(entry)
        .ok_or(MemoryError::Untranslated { linear })
}

/// One four-byte entry of a guest's tables, provided it describes anything.
pub(super) fn entry32(
    physical: Physical<'_>,
    table: u64,
    index: u64,
    linear: u64,
) -> Result<u32, MemoryError> {
    let mut bytes = [0; 4];
    physical.read(PhysAddr::new_truncate(table + index * 4), &mut bytes)?;
    let entry = u32::from_le_bytes(bytes);
    (u64::from(entry) & PRESENT != 0)
        .then_some(entry)
        .ok_or(MemoryError::Untranslated { linear })
}

/// Where an address falls inside the region an entry describes.
///
/// `shift` is how many bits of the address the region spans, which is the whole
/// of what distinguishes a gigabyte page from a four-kilobyte one here.
pub(super) fn region(base: u64, linear: u64, shift: u32) -> Mapped {
    let size = 1 << shift;
    let offset = linear & (size - 1);
    Mapped {
        gpa: PhysAddr::new_truncate((base & !(size - 1)) | offset),
        span: size - offset,
    }
}

/// Where an address falls inside the region an eight-byte entry describes.
pub(super) fn leaf(entry: u64, linear: u64, shift: u32) -> Mapped {
    region(entry & ADDRESS, linear, shift)
}

/// The nine bits of a linear address that index one table of eight-byte
/// entries.
pub(super) const fn index9(linear: u64, shift: u32) -> u64 {
    (linear >> shift) & 0x1FF
}

/// The ten bits of a linear address that index one table of four-byte entries.
pub(super) const fn index10(linear: u64, shift: u32) -> u64 {
    (linear >> shift) & 0x3FF
}

/// What a guest that does not translate reaches.
///
/// The address is already physical, so there is nothing to walk. It is still
/// answered a page at a time rather than all at once, because a caller copying
/// across it has to ask the nested tables page by page regardless — and one
/// shape of answer for every mode is worth more than saving a translation in
/// the one mode a processor spends its first few instructions in.
fn untranslated(linear: u64) -> Mapped {
    region(linear, linear, PAGE_SHIFT)
}

/// Whether the upper bits of an address are the copy of its topmost meaningful
/// bit that a 64-bit walk requires.
fn canonical(linear: u64) -> bool {
    /// Bits above the widest linear address four levels of table describe.
    const SIGN: u32 = 47;
    matches!(linear >> SIGN, 0 | 0x1_FFFF)
}
