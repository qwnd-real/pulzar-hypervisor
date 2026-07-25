//! The Generic Address Structure: how ACPI points at a register instead of
//! describing a value.
//!
//! Twelve bytes — an address space, a width, an offset inside the register, an
//! access size, and a 64-bit address — used wherever a table has to say where a
//! register is rather than what is in it. The timers pulzar reads arrive this
//! way: the HPET's register block, and the power management timer.
//!
//! Two address spaces are modelled, because two are all a timer can be in on a
//! machine pulzar runs on: physical memory and the processor's I/O ports. Every
//! other space ACPI defines — embedded controller, system management bus, PCI
//! configuration —
//! is kept as the raw identifier firmware wrote, so that a consumer refuses it
//! by name rather than mistaking it for one of the two it can reach.
//!
//! The access size is not modelled. The registers this crate describes state
//! their width in the same table, and the field was reserved before ACPI 2.0,
//! so firmware that predates it writes a zero that means nothing at all.

use core::fmt::{self, Display, Formatter};

use crate::{AcpiError, raw::Fields};

/// Bytes in a generic address structure.
pub const ADDRESS_BYTES: usize = 12;

/// Offset of the address space identifier.
const SPACE: usize = 0;

/// Offset of the register's width in bits.
const BIT_WIDTH: usize = 1;

/// Offset of the register's offset in bits.
const BIT_OFFSET: usize = 2;

/// Offset of the address itself.
const ADDRESS: usize = 4;

/// Address space identifier for physical memory.
const SYSTEM_MEMORY: u8 = 0;

/// Address space identifier for the processor's I/O ports.
const SYSTEM_IO: u8 = 1;

/// Where a register is and how wide firmware says it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GenericAddress {
    space: Space,
    bit_width: u8,
    bit_offset: u8,
    address: u64,
}

impl GenericAddress {
    /// The address space the register is reached through.
    #[must_use]
    pub const fn space(&self) -> Space {
        self.space
    }

    /// Bits of the register that are meaningful, counted from
    /// [`GenericAddress::bit_offset`].
    #[must_use]
    pub const fn bit_width(&self) -> u8 {
        self.bit_width
    }

    /// Bit the register starts at within the addressed unit.
    #[must_use]
    pub const fn bit_offset(&self) -> u8 {
        self.bit_offset
    }

    /// The address, whose meaning depends on [`GenericAddress::space`]: a
    /// physical address in memory, a port number in I/O space.
    #[must_use]
    pub const fn address(&self) -> u64 {
        self.address
    }

    /// Reads the structure at `at`.
    ///
    /// # Errors
    ///
    /// [`AcpiError::Truncated`] if the table ends inside it.
    pub(crate) fn parse(table: &Fields<'_>, at: usize) -> Result<Self, AcpiError> {
        let fields = table.nested(at, ADDRESS_BYTES)?;
        Ok(Self {
            space: Space::from(fields.u8(SPACE)?),
            bit_width: fields.u8(BIT_WIDTH)?,
            bit_offset: fields.u8(BIT_OFFSET)?,
            address: fields.u64(ADDRESS)?,
        })
    }
}

impl Display for GenericAddress {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} {:#x}, {} bits at bit {}",
            self.space, self.address, self.bit_width, self.bit_offset
        )
    }
}

/// The address space a register is reached through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Space {
    /// Physical memory, so the register is reached by mapping it.
    Memory,
    /// The processor's I/O ports, so the register is reached by `in` and `out`.
    Io,
    /// A space this crate does not model, kept by its identifier so that a
    /// consumer can say what it refused.
    Other(u8),
}

impl From<u8> for Space {
    fn from(identifier: u8) -> Self {
        match identifier {
            SYSTEM_MEMORY => Self::Memory,
            SYSTEM_IO => Self::Io,
            other => Self::Other(other),
        }
    }
}

impl Display for Space {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Memory => formatter.write_str("system memory"),
            Self::Io => formatter.write_str("i/o port"),
            Self::Other(identifier) => write!(formatter, "address space {identifier:#04x}"),
        }
    }
}
