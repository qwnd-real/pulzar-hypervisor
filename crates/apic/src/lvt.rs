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
//! position.
//!
//! # Every entry is written, and that is not belt and braces
//!
//! These registers come out of reset masked, so a processor being started for
//! the first time needs none of this. The boot processor is the exception, and
//! it is not a small one: firmware has been running on it, and firmware arms
//! the local timer for its own use. Left alone, that entry goes on delivering —
//! on whatever vector firmware chose — the moment this hypervisor unmasks
//! interrupts, which is an interrupt arriving from something that no longer
//! exists.
//!
//! So every entry is written on every processor: masked, except the two pins,
//! which get whatever firmware's tables said they are wired to.

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
        /// A delivery from this source has been accepted and has not completed.
        /// The controller's to set and clear; software cannot write it.
        const SEND_PENDING = 1 << 12;
        /// The source asserts low rather than high. Only the two pins have it.
        const ACTIVE_LOW = 1 << 13;
        /// A level-triggered interrupt from this pin is accepted and not yet
        /// acknowledged. Only the two pins have it, and the controller
        /// maintains it.
        const REMOTE_IRR = 1 << 14;
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

    /// The same entry, delivering or not.
    ///
    /// Masking this way rather than by writing [`Entry::masked`] is what keeps
    /// the rest of the entry — the delivery mode, the vector, and for the timer
    /// the counting mode — in place across the change. A caller that only wants
    /// a source to stop delivering must not also relinquish how it was
    /// configured, because putting it back would then be a reconstruction
    /// rather than a restoration.
    #[must_use]
    pub const fn delivering(self, delivering: bool) -> Self {
        if delivering {
            Self(self.0 & !Bits::MASKED.bits())
        } else {
            Self(self.0 | Bits::MASKED.bits())
        }
    }

    /// Whether a delivery from this source is still in flight.
    ///
    /// The controller sets this from the moment it accepts the interrupt until
    /// delivery completes, and software cannot write it. Anything reporting an
    /// entry's state to somebody else has to read it from here rather than
    /// remember it, because nothing tells software when it clears.
    #[must_use]
    pub const fn pending(self) -> bool {
        self.0 & Bits::SEND_PENDING.bits() != 0
    }

    /// Whether a level-triggered interrupt from this pin has been accepted and
    /// not yet acknowledged.
    ///
    /// Meaningless outside the two pins, and the controller's to maintain in
    /// both: it is set when the interrupt is accepted and cleared by the
    /// acknowledgement.
    #[must_use]
    pub const fn remote_irr(self) -> bool {
        self.0 & Bits::REMOTE_IRR.bits() != 0
    }

    /// Whether this entry delivers nothing.
    ///
    /// Worth asking of an entry read back rather than one built here: hardware
    /// sets this bit itself on the performance-counter source when the counter
    /// it watches overflows, so an entry that was armed can be found masked
    /// without anything having written it.
    #[must_use]
    pub const fn is_masked(self) -> bool {
        self.0 & Bits::MASKED.bits() != 0
    }

    /// The entry a register holds.
    pub(crate) const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// The entry as the register holds it.
    pub(crate) const fn bits(self) -> u32 {
        self.0
    }
}
