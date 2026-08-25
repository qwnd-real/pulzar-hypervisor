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
//! than read past — and the report names the address the structure was read at,
//! which is what identifies it in a log beside everything else that was said
//! about it.

use crate::{AcpiError, as_u64};

/// One firmware structure's bytes, addressed by field offset.
///
/// Carries the address it was read at, so that a failure names the structure
/// without any parser having to pass a description along beside it.
#[derive(Clone, Copy, Debug)]
pub struct Fields<'a> {
    at: u64,
    bytes: &'a [u8],
}

impl<'a> Fields<'a> {
    /// A view of the structure readable at `at` and occupying `bytes`.
    pub const fn new(at: u64, bytes: &'a [u8]) -> Self {
        Self { at, bytes }
    }

    /// Where the structure was read.
    pub const fn at(&self) -> u64 {
        self.at
    }

    /// Bytes the structure occupies.
    pub const fn size(&self) -> usize {
        self.bytes.len()
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
                at: self.at,
                len: self.bytes.len(),
                offset: at,
                wanted: N,
            })
    }

    /// The `len` bytes at `at`, as a structure in their own right.
    ///
    /// The result carries its own address, so offsets inside a nested structure
    /// stay relative to it and errors still name where it is.
    ///
    /// # Errors
    ///
    /// As [`Fields::u8`].
    pub fn nested(&self, at: usize, len: usize) -> Result<Self, AcpiError> {
        let bytes = at
            .checked_add(len)
            .and_then(|end| self.bytes.get(at..end))
            .ok_or(AcpiError::Truncated {
                at: self.at,
                len: self.bytes.len(),
                offset: at,
                wanted: len,
            })?;
        Ok(Self {
            at: self.at.wrapping_add(as_u64(at)),
            bytes,
        })
    }
}
