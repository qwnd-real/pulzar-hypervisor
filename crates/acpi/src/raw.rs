//! Reading what firmware wrote, trusting neither its alignment nor its length
//! fields.
//!
//! Two hazards, both answered here so that no parser has to think about
//! either.
//!
//! Alignment: ACPI aligns its structures to four bytes and then puts eight-byte
//! fields in them — the XSDT's table addresses are the standard example — so a
//! `repr(C)` view of a table would produce unaligned loads. Every scalar in
//! this crate is therefore assembled from bytes instead, which costs nothing a
//! compiler cannot fold away and makes the whole question disappear.
//!
//! Length: a structure's extent comes from a field inside it, written by
//! firmware. Every read is checked against the bytes the structure actually
//! occupies, so a length that promises more than it holds is reported rather
//! than read past — and the report names the physical address, which is the
//! one thing that identifies a firmware structure for certain.

use core::slice;

use paging::DirectMap;
use x86_64::PhysAddr;

use crate::{AcpiError, as_u64};

/// Read access to physical memory through the direct map.
#[derive(Clone, Copy, Debug)]
pub struct Physical {
    map: DirectMap,
}

impl Physical {
    /// Reads physical memory through `map`.
    ///
    /// # Safety
    ///
    /// `map` must be the direct map of the active address space, so that
    /// adding its base to a physical address really does yield somewhere that
    /// address is mapped.
    pub const unsafe fn new(map: DirectMap) -> Self {
        Self { map }
    }

    /// The `len` bytes at `phys`.
    ///
    /// # Errors
    ///
    /// [`AcpiError::Unreachable`] unless the direct map covers the whole range.
    /// For a table address that means firmware pointed at something the memory
    /// map does not describe as memory, which is a table this crate must not
    /// touch rather than one to reach for anyway.
    pub fn bytes(&self, phys: PhysAddr, len: usize) -> Result<&[u8], AcpiError> {
        let covered = phys
            .as_u64()
            .checked_add(as_u64(len))
            .is_some_and(|end| end <= self.map.size());
        let virt = self
            .map
            .virt(phys)
            .filter(|_| covered)
            .ok_or(AcpiError::Unreachable {
                phys: phys.as_u64(),
                len,
            })?;
        // SAFETY: `new`'s caller guarantees the direct map is the live one and the
        // check above puts the whole range inside it, so every byte is mapped
        // readable. The range is ordinary RAM that firmware or a previous owner
        // wrote, so each byte holds a valid `u8`, and `u8` needs no alignment.
        // Nothing hands out a mutable path to a firmware table, so the shared
        // borrow is exclusive of writers, and its lifetime is this `Physical`'s,
        // which the caller of `new` keeps within the life of the address space.
        Ok(unsafe { slice::from_raw_parts(virt.as_ptr::<u8>(), len) })
    }
}

/// One firmware structure's bytes, addressed by field offset.
///
/// Carries the physical address it was read from, so that a failure names the
/// structure without any parser having to pass a description along beside it.
#[derive(Clone, Copy, Debug)]
pub struct Fields<'a> {
    phys: PhysAddr,
    bytes: &'a [u8],
}

impl<'a> Fields<'a> {
    /// A view of the structure at `phys` occupying `bytes`.
    pub const fn new(phys: PhysAddr, bytes: &'a [u8]) -> Self {
        Self { phys, bytes }
    }

    /// Where the structure lives.
    pub const fn phys(&self) -> PhysAddr {
        self.phys
    }

    /// Bytes the structure occupies.
    pub const fn size(&self) -> usize {
        self.bytes.len()
    }

    /// Whether the bytes sum to zero, which is how ACPI marks every structure
    /// it checksums.
    pub fn sums_to_zero(&self) -> bool {
        self.bytes
            .iter()
            .fold(0_u8, |sum, byte| sum.wrapping_add(*byte))
            == 0
    }

    /// The byte at `at`.
    ///
    /// # Errors
    ///
    /// [`AcpiError::Truncated`] if the structure is not that long.
    pub fn u8(&self, at: usize) -> Result<u8, AcpiError> {
        self.array::<1>(at).map(u8::from_le_bytes)
    }

    /// The little-endian `u16` at `at`.
    ///
    /// # Errors
    ///
    /// As [`Fields::u8`].
    pub fn u16(&self, at: usize) -> Result<u16, AcpiError> {
        self.array::<2>(at).map(u16::from_le_bytes)
    }

    /// The little-endian `u32` at `at`.
    ///
    /// # Errors
    ///
    /// As [`Fields::u8`].
    pub fn u32(&self, at: usize) -> Result<u32, AcpiError> {
        self.array::<4>(at).map(u32::from_le_bytes)
    }

    /// The little-endian `u64` at `at`.
    ///
    /// # Errors
    ///
    /// As [`Fields::u8`].
    pub fn u64(&self, at: usize) -> Result<u64, AcpiError> {
        self.array::<8>(at).map(u64::from_le_bytes)
    }

    /// The `N` bytes at `at`.
    ///
    /// # Errors
    ///
    /// As [`Fields::u8`].
    pub fn array<const N: usize>(&self, at: usize) -> Result<[u8; N], AcpiError> {
        self.bytes
            .get(at..)
            .and_then(<[u8]>::first_chunk::<N>)
            .copied()
            .ok_or(AcpiError::Truncated {
                phys: self.phys.as_u64(),
                len: self.bytes.len(),
                offset: at,
                wanted: N,
            })
    }

    /// The `len` bytes at `at`, as a structure in their own right.
    ///
    /// The result carries its own physical address, so offsets inside a nested
    /// structure stay relative to it and errors still name where it is.
    ///
    /// # Errors
    ///
    /// As [`Fields::u8`].
    pub fn nested(&self, at: usize, len: usize) -> Result<Self, AcpiError> {
        let bytes = at
            .checked_add(len)
            .and_then(|end| self.bytes.get(at..end))
            .ok_or(AcpiError::Truncated {
                phys: self.phys.as_u64(),
                len: self.bytes.len(),
                offset: at,
                wanted: len,
            })?;
        Ok(Self {
            phys: self.phys + as_u64(at),
            bytes,
        })
    }
}
