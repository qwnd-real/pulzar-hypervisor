//! Naming one function of one device on one bus of one segment group.
//!
//! Every access to configuration space is addressed by the same four numbers,
//! and all four are bounded by the architecture rather than by anything a
//! machine chooses: at most 65536 segment groups, 256 buses in each, 32 devices
//! on each bus, 8 functions in each device. Because the bounds are fixed, an
//! [`Address`] can hold the two narrow numbers already reduced to their field
//! widths, and every offset derived from one is then inside the aperture it
//! belongs to by construction rather than by a check at the point of use.
//!
//! That property is what the memory-mapped read path leans on. A function's
//! registers are found by adding [`Address::within_bus`] to the base of a
//! mapped bus, and the sum cannot leave that megabyte because the two fields it
//! is built from cannot exceed five and three bits.
//!
//! # Why an offset is not validated when it is made
//!
//! How far into a function's configuration space an access may reach is not a
//! property of the offset. It is a property of the mechanism carrying the
//! access: the memory aperture reaches all 4096 bytes, the legacy ports reach
//! the first 256. So [`Offset`] is a plain name for a byte position, and the
//! reach is checked once, in [`crate::access`], where the mechanism is known —
//! rather than twice, in two places that could disagree.

use core::fmt::{self, Display, Formatter};

/// Bytes of configuration space one function occupies.
pub(crate) const FUNCTION_BYTES: u64 = 4096;

/// Devices on one bus.
pub(crate) const DEVICES: u8 = 1 << DEVICE_BITS;

/// Functions in one device.
pub(crate) const FUNCTIONS: u8 = 1 << FUNCTION_BITS;

/// Bytes of configuration space one bus occupies.
pub(crate) const BUS_BYTES: u64 = FUNCTION_BYTES << (DEVICE_BITS + FUNCTION_BITS);

/// Bits of an address the device number occupies.
const DEVICE_BITS: u8 = 5;

/// Bits of an address the function number occupies.
const FUNCTION_BITS: u8 = 3;

/// Bits of an address the register offset occupies, which is the log of the
/// space one function has.
const FUNCTION_SHIFT: u8 = 12;

/// Widest device number that fits the field.
const DEVICE_MASK: u8 = DEVICES - 1;

/// Widest function number that fits the field.
const FUNCTION_MASK: u8 = FUNCTIONS - 1;

/// A PCI segment group.
///
/// One machine may have several, each with a configuration space aperture of
/// its own and a bus numbering that starts again from zero. Only group zero is
/// reachable through the legacy ports, which predate the concept.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Segment(u16);

impl Segment {
    /// The group every machine has, and the only one without PCI Express.
    pub const ZERO: Self = Self(0);

    /// The group with this number.
    #[must_use]
    pub const fn new(number: u16) -> Self {
        Self(number)
    }

    /// The number itself.
    #[must_use]
    pub const fn get(self) -> u16 {
        self.0
    }
}

impl Display for Segment {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:04x}", self.0)
    }
}

/// A bus number within a segment group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Bus(u8);

impl Bus {
    /// The bus a segment group's hierarchy starts at.
    pub const ZERO: Self = Self(0);

    /// The highest bus number a segment group can hold.
    pub const LAST: Self = Self(u8::MAX);

    /// The bus with this number.
    #[must_use]
    pub const fn new(number: u8) -> Self {
        Self(number)
    }

    /// The number itself.
    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
    }
}

impl Display for Bus {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:02x}", self.0)
    }
}

/// A byte position in a function's configuration space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Offset(u16);

impl Offset {
    /// Bytes of configuration space a PCI Express function has.
    pub const EXTENDED_BYTES: u16 = 4096;

    /// Bytes of configuration space that predate PCI Express, and all the
    /// legacy ports can reach.
    pub const LEGACY_BYTES: u16 = 256;

    /// Where extended configuration space begins.
    pub const EXTENDED: Self = Self(Self::LEGACY_BYTES);

    /// The position this many bytes in.
    #[must_use]
    pub const fn new(value: u16) -> Self {
        Self(value)
    }

    /// The position itself.
    #[must_use]
    pub const fn get(self) -> u16 {
        self.0
    }

    /// Whether this lies in extended configuration space, which only the
    /// memory aperture reaches.
    #[must_use]
    pub const fn is_extended(self) -> bool {
        self.0 >= Self::LEGACY_BYTES
    }

    /// The position `delta` bytes further in.
    ///
    /// Saturating, so that a walk of a malformed capability list produces an
    /// offset the reach check refuses rather than one that wrapped around onto
    /// a register that exists.
    pub(crate) const fn plus(self, delta: u16) -> Self {
        Self(self.0.saturating_add(delta))
    }
}

impl Display for Offset {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:#05x}", self.0)
    }
}

/// One function of one device on one bus of one segment group.
///
/// Ordered as it is written, so that sorting a collection of these produces the
/// order a machine is conventionally listed in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Address {
    segment: Segment,
    bus: Bus,
    device: u8,
    function: u8,
}

impl Address {
    /// The address of `function` of `device` on `bus`.
    ///
    /// `None` unless both numbers fit their fields, so that an address which
    /// exists always names something that could exist.
    #[must_use]
    pub const fn new(segment: Segment, bus: Bus, device: u8, function: u8) -> Option<Self> {
        if device >= DEVICES || function >= FUNCTIONS {
            return None;
        }
        Some(Self::at(segment, bus, device, function))
    }

    /// The same address, narrowed to the field widths rather than refused.
    ///
    /// For the enumeration loops, which iterate the architectural ranges and so
    /// cannot produce a number out of range. Narrowing rather than refusing is
    /// what keeps the invariant that every `Address` in existence names
    /// something addressable, and that invariant is what makes
    /// [`Address::within_bus`] sound to add to the base of a mapped bus.
    pub(crate) const fn at(segment: Segment, bus: Bus, device: u8, function: u8) -> Self {
        Self {
            segment,
            bus,
            device: device & DEVICE_MASK,
            function: function & FUNCTION_MASK,
        }
    }

    /// The segment group.
    #[must_use]
    pub const fn segment(self) -> Segment {
        self.segment
    }

    /// The bus.
    #[must_use]
    pub const fn bus(self) -> Bus {
        self.bus
    }

    /// The device on that bus.
    #[must_use]
    pub const fn device(self) -> u8 {
        self.device
    }

    /// The function of that device.
    #[must_use]
    pub const fn function(self) -> u8 {
        self.function
    }

    /// Another function of the same device.
    #[must_use]
    pub const fn with_function(self, function: u8) -> Self {
        Self::at(self.segment, self.bus, self.device, function)
    }

    /// How everything upstream of this function identifies it.
    ///
    /// The same sixteen bits that appear in a requester identifier, which is
    /// how interrupt remapping, address translation and access control all name
    /// a function. The segment group is outside it, because a routing
    /// identifier only has to be unique within one bus hierarchy.
    #[must_use]
    pub fn routing_id(self) -> u16 {
        (u16::from(self.bus.get()) << u8::BITS)
            | (u16::from(self.device) << FUNCTION_BITS)
            | u16::from(self.function)
    }

    /// Bytes from the start of this bus's configuration space to this
    /// function's.
    ///
    /// At most `BUS_BYTES - FUNCTION_BYTES`, because both fields were narrowed
    /// when the address was made.
    pub(crate) fn within_bus(self) -> u64 {
        (u64::from(self.device) << (FUNCTION_BITS + FUNCTION_SHIFT))
            | (u64::from(self.function) << FUNCTION_SHIFT)
    }
}

impl Display for Address {
    /// The conventional spelling, which is what every other tool that talks
    /// about a PCI function prints.
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "pci {}:{}:{:02x}.{}",
            self.segment, self.bus, self.device, self.function
        )
    }
}

const _: () = assert!(
    FUNCTION_BYTES == 1 << FUNCTION_SHIFT,
    "the function stride and the shift that derives it must agree"
);
