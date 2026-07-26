//! Base address registers: where a function's own registers answer.
//!
//! Six of them on an ordinary function, two on a bridge, and another six inside
//! the single-root virtualization capability for the functions it can conjure.
//! All three sets have the same encoding, so this module decodes a run of them
//! wherever it starts.
//!
//! What a register says is its bottom four bits: whether it answers in I/O or
//! memory space, whether it is one register or the low half of two, and whether
//! reads of it may be prefetched. Everything above those bits is the address
//! firmware assigned.
//!
//! # Nothing here writes, so nothing here knows a size
//!
//! A base address register reports its size only by being written: all ones go
//! in, and what stays set says which bits the device decodes. That write makes
//! the register decode somewhere else for as long as it lasts, which is not
//! something to do to a device that firmware is still driving — and firmware is
//! still driving them, because pulzar leaves the boot environment running. So
//! the survey reads, and a [`Bar`] carries a base and no extent. Sizing is
//! [`crate::size`], which the caller invokes deliberately once the devices are
//! its own.
//!
//! # The last slot of a run cannot hold a wide register
//!
//! A 64-bit register is two consecutive slots, and the second one is not a
//! register in its own right. A function reporting the wide encoding in the
//! last slot of its run is therefore describing something that does not fit:
//! reading the next four bytes as its upper half would take a bridge's bus
//! numbers, or an endpoint's `CardBus` pointer, and produce an address in the
//! terabytes. That is [`Bar::Malformed`], and the run stops treating it as an
//! address at all.

use core::fmt::{self, Display, Formatter};

use bitfield_struct::bitfield;
use log::warn;
use x86_64::PhysAddr;

use crate::{Offset, PciError, access::Config};

/// Base address registers one function can have.
pub const SLOTS: usize = 6;

/// What one base address register describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bar {
    /// The register is not implemented, or firmware assigned it nothing.
    ///
    /// The two are indistinguishable without writing to it: an unimplemented
    /// register reads as zero and so does an unassigned one. On a machine whose
    /// firmware has already configured its devices — which is every machine
    /// pulzar runs on — this mostly means unimplemented.
    Unset,
    /// The function answers I/O accesses in this range.
    Port {
        /// First port the function answers.
        base: u16,
    },
    /// The function answers memory accesses in this range.
    Memory {
        /// First address the function answers.
        base: PhysAddr,
        /// How wide the register that holds it is.
        width: Width,
        /// Whether reads may be prefetched, which is to say whether they have
        /// no side effects and can be merged.
        prefetchable: bool,
    },
    /// The upper half of the 64-bit register in the previous slot, which is not
    /// a register of its own.
    Upper,
    /// The slot holds an encoding no function should produce.
    Malformed,
}

impl Bar {
    /// Where the function answers, if it answers in memory space.
    ///
    /// The upper half of a wide register is deliberately not an address: a
    /// capability naming it as the register holding a table is naming the wrong
    /// half, and returning something plausible would hide that.
    #[must_use]
    pub const fn memory_base(self) -> Option<PhysAddr> {
        match self {
            Self::Memory { base, .. } => Some(base),
            _ => None,
        }
    }

    /// Whether this slot describes a range the function answers.
    #[must_use]
    pub const fn is_set(self) -> bool {
        matches!(self, Self::Port { .. } | Self::Memory { .. })
    }
}

impl Display for Bar {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unset => formatter.write_str("unset"),
            Self::Port { base } => write!(formatter, "io {base:#06x}"),
            Self::Memory {
                base,
                width,
                prefetchable,
            } => {
                write!(formatter, "mem {base:#x} {width}")?;
                if *prefetchable {
                    formatter.write_str(" prefetchable")?;
                }
                Ok(())
            }
            Self::Upper => formatter.write_str("upper half"),
            Self::Malformed => formatter.write_str("malformed"),
        }
    }
}

/// How wide the register holding an address is, and where it may point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Width {
    /// Thirty-two bits, anywhere below four gigabytes.
    Bits32,
    /// Thirty-two bits, and the specification requires the range to lie below
    /// one megabyte. Deprecated for decades and still legal, so it is decoded
    /// rather than treated as reserved — reading it as a wide register would
    /// consume the following slot and shift every register after it.
    Bits32Low,
    /// Sixty-four bits, formed with the slot that follows.
    Bits64,
    /// An encoding the specification does not define, which describes no range
    /// this crate is willing to name an address.
    Reserved(u8),
}

impl Display for Width {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bits32 => formatter.write_str("32-bit"),
            Self::Bits32Low => formatter.write_str("32-bit below 1M"),
            Self::Bits64 => formatter.write_str("64-bit"),
            Self::Reserved(value) => write!(formatter, "reserved width {value:#b}"),
        }
    }
}

/// A function's expansion ROM, which is a memory range like any other except
/// that it has an enable bit of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rom {
    base: PhysAddr,
    enabled: bool,
}

impl Rom {
    /// Where the ROM answers.
    #[must_use]
    pub const fn base(&self) -> PhysAddr {
        self.base
    }

    /// Whether the function decodes the range at all.
    ///
    /// Memory space must also be enabled in the command register for the range
    /// to answer; this bit alone is not enough.
    #[must_use]
    pub const fn enabled(&self) -> bool {
        self.enabled
    }
}

impl Display for Rom {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "rom {:#x}", self.base)?;
        if !self.enabled {
            formatter.write_str(" disabled")?;
        }
        Ok(())
    }
}

/// Decodes the `count` base address registers starting at `first`.
///
/// Slots past `count` stay [`Bar::Unset`], so the result is the same shape
/// whatever layout it came from and a caller indexing it by a register number
/// out of a capability cannot read one run's registers as another's.
///
/// # Errors
///
/// Whatever the configuration mechanism reported.
pub(crate) fn decode(
    config: &Config,
    first: Offset,
    count: usize,
) -> Result<[Bar; SLOTS], PciError> {
    let mut bars = [Bar::Unset; SLOTS];
    let count = count.min(SLOTS);
    let mut slot = 0;
    while slot < count {
        let raw = config.u32(first.plus(stride(slot)))?;
        if raw == 0 {
            slot += 1;
            continue;
        }
        let register = Register::from_bits(raw);
        if register.is_port() {
            bars[slot] = port(config, PortRegister::from_bits(raw));
            slot += 1;
            continue;
        }
        let low = u64::from(register.address()) << ADDRESS_SHIFT;
        match register.width() {
            Width::Bits64 if slot + 1 < count => {
                let upper = config.u32(first.plus(stride(slot + 1)))?;
                let base = (u64::from(upper) << u32::BITS) | low;
                bars[slot] = memory(config, base, Width::Bits64, register.prefetchable());
                bars[slot + 1] = Bar::Upper;
                slot += 2;
            }
            Width::Bits64 => {
                warn!(
                    "pci: {} register {slot} claims 64 bits with no slot to hold the other half",
                    config.address()
                );
                bars[slot] = Bar::Malformed;
                slot += 1;
            }
            Width::Reserved(kind) => {
                warn!(
                    "pci: {} register {slot} uses the reserved memory type {kind:#b}",
                    config.address()
                );
                bars[slot] = Bar::Malformed;
                slot += 1;
            }
            width => {
                bars[slot] = memory(config, low, width, register.prefetchable());
                slot += 1;
            }
        }
    }
    Ok(bars)
}

/// Decodes an expansion ROM base address register.
///
/// # Errors
///
/// Whatever the configuration mechanism reported.
pub(crate) fn rom(config: &Config, offset: Offset) -> Result<Option<Rom>, PciError> {
    let register = RomRegister::from_bits(config.u32(offset)?);
    let base = u64::from(register.address()) << ROM_ADDRESS_SHIFT;
    if base == 0 {
        return Ok(None);
    }
    Ok(PhysAddr::try_new(base).ok().map(|base| Rom {
        base,
        enabled: register.enabled(),
    }))
}

/// One I/O register's contents.
///
/// An I/O address is sixteen bits on this architecture, so anything above them
/// is a register that cannot describe a range the processor could reach.
fn port(config: &Config, register: PortRegister) -> Bar {
    let wide = register.address() << PORT_ADDRESS_SHIFT;
    let Ok(base) = u16::try_from(wide) else {
        warn!(
            "pci: {} holds the I/O address {wide:#x}, which is outside I/O space",
            config.address()
        );
        return Bar::Malformed;
    };
    Bar::Port { base }
}

/// One memory register's contents.
///
/// A base of zero is an unassigned register rather than a range at address
/// zero: nothing is ever assigned there, and a wide register whose halves are
/// both zero still reads as the wide encoding because those bits are hardwired.
fn memory(config: &Config, base: u64, width: Width, prefetchable: bool) -> Bar {
    if base == 0 {
        return Bar::Unset;
    }
    let Ok(base) = PhysAddr::try_new(base) else {
        warn!(
            "pci: {} holds {base:#x}, which is not a usable physical address",
            config.address()
        );
        return Bar::Malformed;
    };
    if matches!(width, Width::Bits32Low) && base.as_u64() >= ONE_MEGABYTE {
        warn!(
            "pci: {} is a below-1M register holding {base:#x}",
            config.address()
        );
    }
    Bar::Memory {
        base,
        width,
        prefetchable,
    }
}

/// Bytes from the start of a run to the register in `slot`.
fn stride(slot: usize) -> u16 {
    // The caller bounds `slot` by `SLOTS`, so this is at most twenty.
    u16::try_from(slot).unwrap_or(u16::MAX).saturating_mul(4)
}

#[bitfield(u32)]
#[derive(PartialEq, Eq)]
/// One memory base address register, as the hardware lays it out.
struct Register {
    /// Whether the register answers in I/O space. Its whole layout differs
    /// when it does, which is why this is read before anything else.
    pub is_port: bool,
    /// How wide the register is and where it may point.
    #[bits(2)]
    pub width: Width,
    /// Whether reads of the range have no side effects and may be merged.
    pub prefetchable: bool,
    /// Address bits 31 to 4 of where the function answers.
    #[bits(28)]
    pub address: u32,
}

#[bitfield(u32)]
#[derive(PartialEq, Eq)]
/// One I/O base address register.
struct PortRegister {
    /// Always set, which is what makes this an I/O register.
    pub is_port: bool,
    /// Reserved.
    __: bool,
    /// Address bits 31 to 2 of the first port the function answers.
    #[bits(30)]
    pub address: u32,
}

#[bitfield(u32)]
#[derive(PartialEq, Eq)]
/// An expansion ROM base address register, which is a memory range with an
/// enable bit of its own.
struct RomRegister {
    /// Whether the function decodes the range. Memory space must be enabled in
    /// the command register as well for it to answer.
    pub enabled: bool,
    /// Reserved.
    #[bits(10)]
    __: u16,
    /// Address bits 31 to 11 of where the ROM answers.
    #[bits(21)]
    pub address: u32,
}

impl Width {
    /// The width this encoding names.
    const fn from_bits(value: u8) -> Self {
        match value {
            0b00 => Self::Bits32,
            0b01 => Self::Bits32Low,
            0b10 => Self::Bits64,
            other => Self::Reserved(other),
        }
    }

    /// The encoding itself.
    const fn into_bits(self) -> u8 {
        match self {
            Self::Bits32 => 0b00,
            Self::Bits32Low => 0b01,
            Self::Bits64 => 0b10,
            Self::Reserved(value) => value,
        }
    }
}

/// Bits a memory register's address field is shifted by.
const ADDRESS_SHIFT: u8 = 4;

/// Bits an I/O register's address field is shifted by.
const PORT_ADDRESS_SHIFT: u8 = 2;

/// Bits an expansion ROM register's address field is shifted by.
const ROM_ADDRESS_SHIFT: u8 = 11;

/// Where a below-1M register stops being one.
const ONE_MEGABYTE: u64 = 1 << 20;
