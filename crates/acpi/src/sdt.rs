//! The header every ACPI table begins with, and what can be trusted after it
//! has been checked.
//!
//! Thirty-six bytes naming the table, giving its total length, and carrying a
//! checksum over the whole of it. Locating a table means reading that header,
//! believing its length only far enough to read the bytes it claims, and then
//! requiring those bytes to sum to zero. Only then does the table become a
//! [`Table`], which is this crate's evidence that a signature, an address and a
//! length belong together.

use core::fmt::{self, Display, Formatter, Write};

use x86_64::PhysAddr;

use crate::{
    AcpiError, as_usize,
    raw::{Fields, Physical},
};

/// Bytes in the header every table begins with.
pub const HEADER_BYTES: usize = 36;

/// Offset of the four characters naming the table.
const SIGNATURE: usize = 0;

/// Offset of the table's total length, header included.
const LENGTH: usize = 4;

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
    phys: PhysAddr,
    length: u32,
    revision: u8,
}

impl Table {
    /// The four characters naming the table.
    #[must_use]
    pub const fn signature(&self) -> Signature {
        self.signature
    }

    /// Where the table lives in physical memory.
    #[must_use]
    pub const fn phys(&self) -> PhysAddr {
        self.phys
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

/// Reads and checks the header of the table at `phys`.
///
/// # Errors
///
/// [`AcpiError::Unreachable`] if the direct map does not reach the table,
/// [`AcpiError::Truncated`] if it claims to be shorter than a header, or
/// [`AcpiError::BadChecksum`] if its bytes do not sum to zero.
pub fn locate(memory: &Physical, phys: PhysAddr) -> Result<Table, AcpiError> {
    let header = Fields::new(phys, memory.bytes(phys, HEADER_BYTES)?);
    let signature = Signature(header.array::<4>(SIGNATURE)?);
    let length = header.u32(LENGTH)?;
    let size = as_usize(u64::from(length));
    if size < HEADER_BYTES {
        return Err(AcpiError::Truncated {
            phys: phys.as_u64(),
            len: size,
            offset: 0,
            wanted: HEADER_BYTES,
        });
    }
    if !Fields::new(phys, memory.bytes(phys, size)?).sums_to_zero() {
        return Err(AcpiError::BadChecksum {
            phys: phys.as_u64(),
            len: size,
        });
    }
    Ok(Table {
        signature,
        phys,
        length,
        revision: header.u8(REVISION)?,
    })
}

/// The whole of `table`, header included, ready for a parser.
///
/// # Errors
///
/// [`AcpiError::Unreachable`] if the direct map no longer reaches the table,
/// which cannot happen for a table [`locate`] returned.
pub fn contents<'a>(memory: &'a Physical, table: &Table) -> Result<Fields<'a>, AcpiError> {
    let size = as_usize(u64::from(table.length));
    Ok(Fields::new(table.phys, memory.bytes(table.phys, size)?))
}
