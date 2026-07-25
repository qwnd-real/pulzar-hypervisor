//! The local vector table: what each of the controller's own interrupt sources
//! does when it fires.
//!
//! Six sources — the timer, the thermal sensor, the performance counters, the
//! two interrupt pins, and the controller's own errors — and one register each,
//! all with the same layout. What differs is which fields a given source
//! honours: only the two pins have a polarity and a trigger mode, because only
//! they are wired to anything outside the processor.
//!
//! The register is built here rather than assembled at each use site, so that
//! the reserved bits stay zero and a field cannot be written into the wrong
//! position. Every entry has a mask bit, and an entry this crate does not
//! deliberately arm is masked: a source left in whatever state firmware wanted
//! is a source that can deliver an interrupt to a vector chosen by something
//! that is no longer running.

use bitflags::bitflags;
use descriptors::Vector;

/// How an interrupt from one of the controller's own sources is delivered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Delivery {
    /// As an ordinary interrupt on the entry's vector.
    Fixed(Vector),
    /// As a non-maskable interrupt, which ignores the vector entirely and
    /// always arrives on vector 2.
    NonMaskable,
    /// As a system management interrupt, which the vector must be zero for.
    SystemManagement,
    /// As an external interrupt, meaning the processor fetches the vector from
    /// a legacy controller rather than from this entry.
    External,
}

impl Delivery {
    /// The delivery mode field's encoding.
    const fn mode(self) -> u32 {
        match self {
            Self::Fixed(_) => 0b000,
            Self::SystemManagement => 0b010,
            Self::NonMaskable => 0b100,
            Self::External => 0b111,
        }
    }

    /// The vector field's contents, which every mode but the first ignores and
    /// two of them require to be zero.
    const fn vector(self) -> u32 {
        match self {
            Self::Fixed(vector) => vector.number() as u32,
            _ => 0,
        }
    }
}

/// How a source that is wired to something asserts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Polarity {
    /// Asserted high, which is the default for everything on the ISA bus.
    ActiveHigh,
    /// Asserted low.
    ActiveLow,
}

/// When a source that is wired to something asserts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Trigger {
    /// On the transition.
    Edge,
    /// For as long as the level holds.
    Level,
}

bitflags! {
    /// The bits of a local vector table entry that are not a field.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Bits: u32 {
        /// The source asserts low rather than high. Only the two pins have it.
        const ACTIVE_LOW = 1 << 13;
        /// The source is level triggered rather than edge triggered. Only the
        /// two pins have it.
        const LEVEL = 1 << 15;
        /// Nothing is delivered from this source.
        const MASKED = 1 << 16;
    }
}

/// Bits the delivery mode field is shifted by.
const DELIVERY_SHIFT: u32 = 8;

/// One local vector table entry, ready to be written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Entry(u32);

impl Entry {
    /// An entry that delivers nothing.
    ///
    /// The state every source this crate does not arm is left in. The vector
    /// still has to be a real one — a masked entry with vector zero is a
    /// configuration some processors report as an error — so the lowest vector
    /// the platform may assign is used, and nothing is ever delivered on it.
    #[must_use]
    pub const fn masked() -> Self {
        Self(Bits::MASKED.bits() | Vector::FIRST_EXTERNAL.number() as u32)
    }

    /// An entry that delivers `delivery`, edge triggered and asserted high,
    /// which is what every source that is not one of the two pins is.
    #[must_use]
    pub const fn new(delivery: Delivery) -> Self {
        Self(delivery.vector() | (delivery.mode() << DELIVERY_SHIFT))
    }

    /// The same entry with the polarity and trigger mode firmware described,
    /// for the two pins where those mean something.
    #[must_use]
    pub const fn wired(self, polarity: Polarity, trigger: Trigger) -> Self {
        let mut bits = self.0;
        if matches!(polarity, Polarity::ActiveLow) {
            bits |= Bits::ACTIVE_LOW.bits();
        }
        if matches!(trigger, Trigger::Level) {
            bits |= Bits::LEVEL.bits();
        }
        Self(bits)
    }

    /// The same entry, masked.
    #[must_use]
    pub const fn mask(self) -> Self {
        Self(self.0 | Bits::MASKED.bits())
    }

    /// Whether this entry delivers anything.
    #[must_use]
    pub const fn is_masked(self) -> bool {
        self.0 & Bits::MASKED.bits() != 0
    }

    /// The entry as the register holds it.
    pub(crate) const fn bits(self) -> u32 {
        self.0
    }

    /// An entry read back out of a register.
    pub(crate) const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }
}
