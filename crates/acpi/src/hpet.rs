//! The HPET table: where the high precision event timer's registers are.
//!
//! Twenty bytes past the header, and almost all of it is an address. That is
//! the point of the table: the timer describes itself — its tick period, its
//! width, whether it is running — in the register block, and the only thing
//! firmware has to say is where that block is. So this parse ends at the
//! address, and what the timer does is a question for whoever maps it.
//!
//! Pulzar reads it because the HPET is the one counter on a PC whose frequency
//! is knowable without measuring it against something else. Everything else
//! either has to be calibrated first, like the timestamp counter, or is too
//! narrow to keep time in, like the power management timer — so this table is
//! where a nanosecond ultimately comes from.
//!
//! A machine may describe several blocks, one table each, distinguished by
//! [`Hpet::number`]. The absence of the table is not a failure here: a virtual
//! machine can be configured without an HPET at all, and what to do about that
//! is the caller's decision rather than this crate's.

use log::{info, warn};
use x86_64::PhysAddr;

use crate::{
    AcpiError,
    gas::{GenericAddress, Space},
    raw::Fields,
};

/// Offset of the block's revision, the first byte of the event timer block
/// identifier. The specification requires it to be nonzero.
const REVISION: usize = 36;

/// Offset of the byte of the block identifier holding the comparator count and
/// the width of the main counter.
const CAPABILITIES: usize = 37;

/// Offset of the PCI vendor identifier, the top half of the block identifier.
const VENDOR: usize = 38;

/// Offset of the register block's address.
const ADDRESS: usize = 40;

/// Offset of the number distinguishing this block from the machine's others.
const NUMBER: usize = 52;

/// Capabilities: comparators in the block, less one.
const COMPARATORS: u8 = 0x1F;

/// Capabilities: the main counter is 64 bits wide rather than 32.
const COUNTER_64BIT: u8 = 1 << 5;

/// One high precision event timer's register block, as firmware describes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hpet {
    register_block: GenericAddress,
    number: u8,
    revision: u8,
    comparators: u8,
    counter_64bit: bool,
    vendor: u16,
}

impl Hpet {
    /// Where the register block is, as firmware described it.
    #[must_use]
    pub const fn register_block(&self) -> GenericAddress {
        self.register_block
    }

    /// Physical address of the register block, or `None` if firmware put it
    /// somewhere nothing can map.
    ///
    /// The specification allows the block in system memory only, so an I/O
    /// port, an address with bits above the physical address space, or a null
    /// address all describe a timer no software could reach. Such a block is
    /// reported as absent rather than interpreted.
    #[must_use]
    pub fn memory_base(&self) -> Option<PhysAddr> {
        (self.register_block.space() == Space::Memory && self.register_block.address() != 0)
            .then(|| crate::address(self.register_block.address()).ok())
            .flatten()
    }

    /// Which of the machine's register blocks this is.
    #[must_use]
    pub const fn number(&self) -> u8 {
        self.number
    }

    /// Comparators in the block, which is how many independent timers it can
    /// arm.
    #[must_use]
    pub const fn comparators(&self) -> u8 {
        self.comparators
    }

    /// Whether the main counter is 64 bits wide rather than 32.
    ///
    /// A 32-bit counter at the rates these run at wraps every few minutes,
    /// which anything keeping time from it has to account for — and which a
    /// calibration window measured in milliseconds never runs into.
    #[must_use]
    pub const fn counter_64bit(&self) -> bool {
        self.counter_64bit
    }

    /// The PCI vendor identifier of whoever implemented the block.
    #[must_use]
    pub const fn vendor(&self) -> u16 {
        self.vendor
    }

    /// Logs the block, and whether it is one that can be reached.
    pub fn describe(&self, who: &str) {
        info!(
            "{who}: hpet {} at {}, revision {}, {} comparators, {}-bit counter, vendor {:#06x}",
            self.number,
            self.register_block,
            self.revision,
            self.comparators,
            if self.counter_64bit { 64 } else { 32 },
            self.vendor,
        );
        if self.memory_base().is_none() {
            warn!("{who}: that hpet register block is not mappable memory, so nothing can read it");
        }
    }

    /// Parses the table.
    ///
    /// Each field of the event timer block identifier is read at its own
    /// offset, so the parse needs neither a shift nor a narrowing cast.
    ///
    /// # Errors
    ///
    /// [`AcpiError::Truncated`] if the table is shorter than the fields it has
    /// to hold.
    pub(crate) fn parse(table: &Fields<'_>) -> Result<Self, AcpiError> {
        let capabilities = table.u8(CAPABILITIES)?;
        let hpet = Self {
            register_block: GenericAddress::parse(table, ADDRESS)?,
            number: table.u8(NUMBER)?,
            revision: table.u8(REVISION)?,
            comparators: (capabilities & COMPARATORS) + 1,
            counter_64bit: capabilities & COUNTER_64BIT != 0,
            vendor: table.u16(VENDOR)?,
        };
        if hpet.revision == 0 {
            warn!(
                "acpi: the hpet at {:#x} reports revision 0, which the specification forbids",
                table.phys()
            );
        }
        Ok(hpet)
    }
}
