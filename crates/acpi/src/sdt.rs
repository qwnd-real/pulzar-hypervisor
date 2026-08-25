//! The header every ACPI table begins with, and what a checked table is.
//!
//! Thirty-six bytes naming the table, giving its total length, and carrying a
//! checksum over the whole of it. Nothing here reads a table out of memory or
//! verifies one: uACPI walks the root pointer's directory, checks each header
//! and each checksum, and hands over tables that have already passed. What is
//! left is the two things this crate says about a table it has been handed —
//! which table it is, and where to find it again — and that is what [`Table`]
//! is.

use core::fmt::{self, Display, Formatter, Write};

use crate::{AcpiError, raw::Fields};

/// Bytes in the header every table begins with.
pub const HEADER_BYTES: usize = 36;

/// Offset of the four characters naming the table.
const SIGNATURE: usize = 0;

/// Offset of the table's total length, header included.
pub const LENGTH: usize = 4;

/// Offset of the table's revision.
const REVISION: usize = 8;

/// The four characters that name an ACPI table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Signature([u8; 4]);

impl Signature {
    /// Multiple APIC Description Table: the processors of the machine and the
    /// interrupt controllers that serve them. Signed `APIC`, from before it
    /// described anything else.
    pub const MADT: Self = Self(*b"APIC");

    /// Fixed ACPI Description Table: the platform's own registers, sleep states
    /// and feature flags. Signed `FACP`, from before the table was named after
    /// what it describes.
    pub const FADT: Self = Self(*b"FACP");

    /// High Precision Event Timer: where one event timer's register block is.
    pub const HPET: Self = Self(*b"HPET");

    /// Memory Mapped Configuration Space: where PCI Express configuration
    /// space is mapped, per segment group.
    pub const MCFG: Self = Self(*b"MCFG");

    /// Root System Description Table: the table directory of ACPI 1.0, whose
    /// entries are 32-bit addresses.
    pub const RSDT: Self = Self(*b"RSDT");

    /// Extended System Description Table: the table directory of ACPI 2.0 and
    /// later, whose entries are 64-bit addresses.
    pub const XSDT: Self = Self(*b"XSDT");

    /// The signature these four characters spell.
    #[must_use]
    pub const fn new(characters: [u8; 4]) -> Self {
        Self(characters)
    }

    /// The four characters and a terminator, which is how uACPI is asked for a
    /// table by signature.
    pub(crate) const fn terminated(self) -> [u8; 5] {
        let [a, b, c, d] = self.0;
        [a, b, c, d, 0]
    }
}

impl Display for Signature {
    /// Anything outside printable ASCII becomes `?`. A signature is firmware's
    /// to write and a corrupt one is exactly what gets logged, so it must not
    /// be able to put control characters into the log it is being reported in.
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        self.0.iter().try_for_each(|byte| {
            formatter.write_char(match byte {
                0x20..=0x7e => char::from(*byte),
                _ => '?',
            })
        })
    }
}

/// A table that has been found, named and checked, but not yet parsed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Table {
    signature: Signature,
    index: usize,
    at: u64,
    length: u32,
    revision: u8,
}

impl Table {
    /// Reads what a table's header says about it.
    ///
    /// `index` is the position uACPI keeps the table at, which is the only name
    /// the table can be asked for again by.
    ///
    /// # Errors
    ///
    /// [`AcpiError::Truncated`] if the bytes do not reach the end of a header,
    /// which for a table uACPI has already checked cannot happen.
    pub fn read(index: usize, header: &Fields<'_>) -> Result<Self, AcpiError> {
        Ok(Self {
            signature: Signature(header.array::<4>(SIGNATURE)?),
            index,
            at: header.at(),
            length: header.u32(LENGTH)?,
            revision: header.u8(REVISION)?,
        })
    }

    /// The four characters naming the table.
    #[must_use]
    pub const fn signature(&self) -> Signature {
        self.signature
    }

    /// Where uACPI keeps the table, which is how it is asked for again.
    #[must_use]
    pub const fn index(&self) -> usize {
        self.index
    }

    /// The address the table was readable at when it was described.
    ///
    /// A virtual address, and not one to read through afterwards: the mapping
    /// behind it belongs to uACPI, which gives it back once nothing holds a
    /// reference to the table. It is kept because it is what identifies the
    /// table in a log line beside everything else that was said about it.
    #[must_use]
    pub const fn at(&self) -> u64 {
        self.at
    }

    /// Bytes the table occupies, header included.
    #[must_use]
    pub const fn length(&self) -> u32 {
        self.length
    }

    /// The revision of this table's own definition, which says which of its
    /// fields are meaningful.
    #[must_use]
    pub const fn revision(&self) -> u8 {
        self.revision
    }
}
