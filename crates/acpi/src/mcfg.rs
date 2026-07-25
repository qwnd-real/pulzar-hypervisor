//! The Memory Mapped Configuration table: where PCI Express configuration
//! space is mapped.
//!
//! PCI Express replaces the two I/O ports that used to reach configuration
//! space with a plain memory aperture, and this table is the only thing that
//! says where that aperture is. One entry per segment group, each covering a
//! contiguous run of bus numbers, laid out so that a bus, device, function and
//! register decode into an offset from the entry's base.
//!
//! The table is absent on a machine with no PCI Express at all, which is why
//! nothing here treats its absence as a failure.

use alloc::vec::Vec;

use log::{info, warn};
use x86_64::PhysAddr;

use crate::{AcpiError, raw::Fields};

/// Offset of the first allocation, after the header and the reserved field that
/// follows it.
const FIRST_ENTRY: usize = 44;

/// Bytes in one allocation.
const ENTRY_BYTES: usize = 16;

/// Bytes of configuration space one bus occupies: thirty-two devices of eight
/// functions each, four kilobytes apiece.
const BUS_BYTES: u64 = 32 * 8 * 4096;

/// Every configuration space aperture firmware described.
#[derive(Debug)]
pub struct Mcfg {
    spaces: Vec<ConfigSpace>,
}

impl Mcfg {
    /// The apertures, in the order firmware listed them.
    #[must_use]
    pub fn spaces(&self) -> &[ConfigSpace] {
        &self.spaces
    }

    /// Logs every aperture.
    pub fn describe(&self, who: &str) {
        for space in &self.spaces {
            info!(
                "{who}: mcfg segment {} buses {}..={} at {:#x}, {:#x} bytes",
                space.segment,
                space.first_bus,
                space.last_bus,
                space.base,
                space.bytes(),
            );
        }
    }

    /// Parses the table.
    ///
    /// A ragged tail is tolerated: firmware whose length field does not land on
    /// an allocation boundary exists, and the entries before the remainder are
    /// still sound. So is an entry whose bus range runs backwards, which
    /// describes nothing and is dropped rather than turned into an aperture of
    /// nonsensical size. Both are logged.
    ///
    /// # Errors
    ///
    /// [`AcpiError::Truncated`] if an allocation runs past the end of the
    /// table, or [`AcpiError::BadAddress`] if one names a base this processor
    /// cannot form.
    pub(crate) fn parse(table: &Fields<'_>) -> Result<Self, AcpiError> {
        let listed = table.size().saturating_sub(FIRST_ENTRY);
        if !listed.is_multiple_of(ENTRY_BYTES) {
            warn!(
                "acpi: the mcfg at {:#x} ends {} bytes into an allocation; ignoring the remainder",
                table.phys(),
                listed % ENTRY_BYTES,
            );
        }
        let mut spaces = Vec::new();
        for index in 0..listed / ENTRY_BYTES {
            // Base address, segment group, first and last bus, four reserved
            // bytes.
            let entry = table.nested(FIRST_ENTRY + index * ENTRY_BYTES, ENTRY_BYTES)?;
            let space = ConfigSpace {
                base: crate::address(entry.u64(0)?)?,
                segment: entry.u16(8)?,
                first_bus: entry.u8(10)?,
                last_bus: entry.u8(11)?,
            };
            if space.last_bus < space.first_bus {
                warn!(
                    "acpi: ignoring mcfg segment {} at {:#x}, whose bus range {}..={} is empty",
                    space.segment, space.base, space.first_bus, space.last_bus,
                );
                continue;
            }
            spaces.push(space);
        }
        Ok(Self { spaces })
    }
}

/// One segment group's configuration space aperture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConfigSpace {
    base: PhysAddr,
    segment: u16,
    first_bus: u8,
    last_bus: u8,
}

impl ConfigSpace {
    /// Physical address the aperture starts at, which is the configuration
    /// space of function zero of device zero on [`ConfigSpace::first_bus`].
    #[must_use]
    pub const fn base(&self) -> PhysAddr {
        self.base
    }

    /// The PCI segment group this aperture serves.
    #[must_use]
    pub const fn segment(&self) -> u16 {
        self.segment
    }

    /// Lowest bus number the aperture covers.
    #[must_use]
    pub const fn first_bus(&self) -> u8 {
        self.first_bus
    }

    /// Highest bus number the aperture covers.
    #[must_use]
    pub const fn last_bus(&self) -> u8 {
        self.last_bus
    }

    /// Bytes the aperture occupies, which is what has to be mapped to reach all
    /// of it.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        (u64::from(self.last_bus) - u64::from(self.first_bus) + 1) * BUS_BYTES
    }
}
