//! The Root System Description Pointer, and the table directories it names.
//!
//! Everything ACPI describes hangs off one structure that is not itself a
//! table: no header, no signature of the usual four characters, and two
//! different layouts depending on the revision. The 1.0 layout is twenty bytes
//! ending in a 32-bit address of the RSDT. The 2.0 layout extends it with a
//! length, a 64-bit address of the XSDT, and a second checksum covering the
//! whole structure — and firmware that supports the later revision fills in
//! both directories.
//!
//! Which directory to believe is a decision this module only prepares: it
//! reports the candidates in order of preference and lets the caller find out
//! whether the preferred one actually reads.

use core::fmt::{self, Display, Formatter};

use x86_64::PhysAddr;

use crate::{
    AcpiError,
    raw::{Fields, Physical},
    sdt::Signature,
};

/// What the structure spells at its front, padding included.
const MAGIC: [u8; 8] = *b"RSD PTR ";

/// Bytes in the part of the structure every revision defines, and the extent
/// the first checksum covers.
const LEGACY_BYTES: usize = 20;

/// Bytes in the smallest structure revision 2 and later may present, and so
/// the smallest length its own length field may claim.
const EXTENDED_BYTES: usize = 36;

/// The revision from which the structure carries a 64-bit directory.
const EXTENDED_REVISION: u8 = 2;

/// Offset of [`MAGIC`].
const SIGNATURE: usize = 0;

/// Offset of the revision.
const REVISION: usize = 15;

/// Offset of the 32-bit address of the RSDT.
const RSDT: usize = 16;

/// Offset of the structure's total length, present from revision 2.
const LENGTH: usize = 20;

/// Offset of the 64-bit address of the XSDT, present from revision 2.
const XSDT: usize = 24;

/// A table directory, and which of the two kinds it is.
///
/// The two differ only in the width of their entries, which is exactly the
/// difference between the ACPI revision that introduced each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Directory {
    /// The 32-bit directory, all that ACPI 1.0 had.
    Rsdt(PhysAddr),
    /// The 64-bit directory, which every revision since 2.0 provides.
    Xsdt(PhysAddr),
}

impl Directory {
    /// Where the directory lives.
    #[must_use]
    pub const fn phys(self) -> PhysAddr {
        match self {
            Self::Rsdt(phys) | Self::Xsdt(phys) => phys,
        }
    }

    /// The signature the directory's own header must carry.
    #[must_use]
    pub const fn signature(self) -> Signature {
        match self {
            Self::Rsdt(_) => Signature::RSDT,
            Self::Xsdt(_) => Signature::XSDT,
        }
    }

    /// Bytes in one of the directory's entries.
    #[must_use]
    pub const fn stride(self) -> usize {
        match self {
            Self::Rsdt(_) => size_of::<u32>(),
            Self::Xsdt(_) => size_of::<u64>(),
        }
    }

    /// The table address the entry at `at` holds.
    ///
    /// # Errors
    ///
    /// [`AcpiError::Truncated`] if the directory ends inside the entry.
    pub fn entry(self, directory: &Fields<'_>, at: usize) -> Result<u64, AcpiError> {
        match self {
            Self::Rsdt(_) => directory.u32(at).map(u64::from),
            Self::Xsdt(_) => directory.u64(at),
        }
    }
}

impl Display for Directory {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} at {:#x}", self.signature(), self.phys())
    }
}

/// The root pointer, checked, and the directories it names.
#[derive(Clone, Copy, Debug)]
pub struct RootPointer {
    revision: u8,
    rsdt: Option<PhysAddr>,
    xsdt: Option<PhysAddr>,
}

impl RootPointer {
    /// Reads and checks the root pointer at `phys`.
    ///
    /// The revision decides how much of the structure exists, so the extended
    /// half is read only once the revision has said it is there. That is not
    /// caution for its own sake: on an ACPI 1.0 machine the twenty-first byte
    /// belongs to something else entirely.
    ///
    /// # Errors
    ///
    /// [`AcpiError::NoRootPointer`] if `phys` is zero, meaning firmware
    /// published none; [`AcpiError::BadAddress`] for an address this processor
    /// cannot form; [`AcpiError::NotARootPointer`] if the signature does not
    /// match; [`AcpiError::BadChecksum`] if either checksum does not hold; or
    /// [`AcpiError::Truncated`] if a revision 2 structure claims to be shorter
    /// than revision 2 defines.
    pub fn read(memory: &Physical, phys: u64) -> Result<Self, AcpiError> {
        if phys == 0 {
            return Err(AcpiError::NoRootPointer);
        }
        let phys = crate::address(phys)?;
        let legacy = Fields::new(phys, memory.bytes(phys, LEGACY_BYTES)?);
        if legacy.array::<8>(SIGNATURE)? != MAGIC {
            return Err(AcpiError::NotARootPointer {
                phys: phys.as_u64(),
            });
        }
        if !legacy.sums_to_zero() {
            return Err(AcpiError::BadChecksum {
                phys: phys.as_u64(),
                len: LEGACY_BYTES,
            });
        }
        let revision = legacy.u8(REVISION)?;
        let rsdt = optional(u64::from(legacy.u32(RSDT)?))?;
        if revision < EXTENDED_REVISION {
            return Ok(Self {
                revision,
                rsdt,
                xsdt: None,
            });
        }

        let extended = Fields::new(phys, memory.bytes(phys, EXTENDED_BYTES)?);
        let length = crate::as_usize(u64::from(extended.u32(LENGTH)?));
        if length < EXTENDED_BYTES {
            return Err(AcpiError::Truncated {
                phys: phys.as_u64(),
                len: length,
                offset: 0,
                wanted: EXTENDED_BYTES,
            });
        }
        if !Fields::new(phys, memory.bytes(phys, length)?).sums_to_zero() {
            return Err(AcpiError::BadChecksum {
                phys: phys.as_u64(),
                len: length,
            });
        }
        Ok(Self {
            revision,
            rsdt,
            xsdt: optional(extended.u64(XSDT)?)?,
        })
    }

    /// The ACPI revision the root pointer claims.
    #[must_use]
    pub const fn revision(&self) -> u8 {
        self.revision
    }

    /// The directories to try, in the order to try them.
    ///
    /// The 64-bit directory comes first where it exists: it is the one later
    /// revisions define, and the only one that can name a table above 4 GiB.
    /// The 32-bit one is offered afterwards rather than discarded, because
    /// firmware that fills in a broken extended directory beside a sound legacy
    /// one is a thing that ships, and a machine like that is still a machine
    /// pulzar should run on.
    pub fn directories(&self) -> impl Iterator<Item = Directory> {
        [
            self.xsdt.map(Directory::Xsdt),
            self.rsdt.map(Directory::Rsdt),
        ]
        .into_iter()
        .flatten()
    }
}

/// A table address out of the root pointer, where zero means the directory it
/// would name does not exist.
fn optional(value: u64) -> Result<Option<PhysAddr>, AcpiError> {
    match value {
        0 => Ok(None),
        address => crate::address(address).map(Some),
    }
}
