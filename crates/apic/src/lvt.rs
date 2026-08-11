//! The local vector table: what each of the controller's own interrupt sources
//! does when it fires.
//!
//! Six sources — the timer, the thermal sensor, the performance counters, the
//! two interrupt pins, and the controller's own errors — and one register each.
//! The fields sit in the same positions in all of them, and that is as far as
//! the resemblance goes: which fields a given register *has* differs, and a
//! field written where the register reserves it is a reserved bit set. Only the
//! two pins have a polarity and a trigger mode, because only they are wired to
//! anything outside the processor; only the timer has a counting mode; and only
//! the pins may hand the vector over to a legacy controller.
//!
//! So an entry is built here rather than assembled at each use site — which
//! keeps the reserved bits zero and a field out of the wrong position — and
//! [`Shape`] says which of those fields a particular source honours, so that
//! writing one it does not have is refused rather than encoded.
//!
//! # A register read back is not a value to write
//!
//! Two bits of every entry are the controller's: whether a delivery from this
//! source is still in flight, and whether a level-triggered interrupt from this
//! pin has been accepted. Software cannot set either. An entry read from
//! hardware carries both, along with whatever the model leaves reserved, so
//! everything on its way *to* a register goes through [`Entry::writable`] first
//! — the alternative is asking hardware to set bits it owns and relying on it
//! to ignore the request.
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
    /// the system-management mode requires to be zero.
    const fn vector(self) -> u32 {
        match self {
            Self::Fixed(vector) => vector.number() as u32,
            _ => 0,
        }
    }

    /// Which delivery an entry the controller already holds describes, or
    /// `None` where the field holds one of the three encodings the
    /// architecture reserves.
    const fn of(bits: u32) -> Option<Self> {
        match (bits >> DELIVERY_SHIFT) & DELIVERY_FIELD {
            0b000 => Some(Self::Fixed(Vector::new(vector_of(bits)))),
            0b010 => Some(Self::SystemManagement),
            0b100 => Some(Self::NonMaskable),
            0b111 => Some(Self::External),
            _ => None,
        }
    }
}

/// Which of the fields every entry shares a position for a given source
/// actually has.
///
/// The reason writing an entry needs to know which source it is for: a polarity
/// on the thermal sensor or a counting mode on an interrupt pin is a reserved
/// bit, and a reserved bit in an entry reached through the model-specific
/// registers is a general protection fault rather than a bit hardware ignores.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Shape {
    /// The controller's own timer: a vector, a mask, and the two bits that
    /// choose how it counts.
    Timer,
    /// One of the two interrupt pins: everything an entry can hold, because a
    /// pin is the only source wired to something outside the processor and the
    /// only one that can defer to a legacy controller.
    Pin,
    /// A source inside the processor: a vector, a delivery mode and a mask.
    Internal,
}

impl Shape {
    /// Whether a source of this shape can deliver this way.
    pub(crate) const fn allows(self, delivery: Delivery) -> bool {
        match self {
            // The timer counts; what it delivers when it gets there is an
            // ordinary interrupt and nothing else.
            Self::Timer => matches!(delivery, Delivery::Fixed(_)),
            Self::Pin => true,
            // An external interrupt means the processor runs an acknowledge
            // cycle to a legacy controller, which only a pin is wired to.
            Self::Internal => !matches!(delivery, Delivery::External),
        }
    }

    /// The bits software may set in a source of this shape.
    const fn writable(self) -> u32 {
        let common = VECTOR_FIELD | (DELIVERY_FIELD << DELIVERY_SHIFT) | Bits::MASKED.bits();
        match self {
            Self::Timer => common | (MODE_FIELD << MODE_SHIFT),
            Self::Pin => common | Bits::ACTIVE_LOW.bits() | Bits::LEVEL.bits(),
            Self::Internal => common,
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
        /// A delivery from this source has been accepted by the controller and
        /// has not yet reached the processor's interrupt request register. The
        /// controller's to set and clear; software cannot write it.
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

/// The vector field: the low byte.
const VECTOR_FIELD: u32 = 0xFF;

/// Bits the delivery mode field is shifted by.
const DELIVERY_SHIFT: u32 = 8;

/// The delivery mode field, once shifted down: three bits.
const DELIVERY_FIELD: u32 = 0b111;

/// Bits the timer's counting mode field is shifted by.
pub(crate) const MODE_SHIFT: u32 = 17;

/// The timer's counting mode field, once shifted down.
pub(crate) const MODE_FIELD: u32 = 0b11;

/// The vector an entry names.
const fn vector_of(bits: u32) -> u8 {
    (bits & VECTOR_FIELD) as u8
}

/// One local vector table entry.
///
/// Both a value on its way to a register and one read out of it, which are not
/// the same thing: a register carries bits the controller owns, and
/// [`Entry::writable`] is what separates the two.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Entry(u32);

impl Entry {
    /// An entry that delivers nothing.
    ///
    /// The state every source this crate does not arm is left in. The vector
    /// still has to be a real one — a masked entry with vector zero is a
    /// configuration some processors report as an error — so the lowest vector
    /// the platform may assign is used, and nothing is ever delivered on it.
    ///
    /// Everything else is given up: this is the entry to write when nothing
    /// about the old one is worth keeping. [`Entry::delivering`] is the one to
    /// use when it is.
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
    ///
    /// Both fields are replaced rather than merely set, so that this says what
    /// the wiring *is* whatever the entry said before — including on an entry
    /// read back from a controller firmware had programmed the other way.
    #[must_use]
    pub const fn wired(self, polarity: Polarity, trigger: Trigger) -> Self {
        let mut bits = self.0 & !(Bits::ACTIVE_LOW.bits() | Bits::LEVEL.bits());
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

    /// Whether the controller has accepted a delivery from this source and not
    /// yet handed it to the processor.
    ///
    /// Send status, and no more than that: it clears once the interrupt reaches
    /// the processor's request register, which is before anything accepts the
    /// vector into service and long before anything acknowledges it. It says
    /// nothing about whether the interrupt is still owed.
    ///
    /// The controller's to maintain, and software cannot write it — so anything
    /// reporting an entry's state to somebody else has to read it from here
    /// rather than remember it.
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

    /// Which delivery this entry describes, or `None` for one of the encodings
    /// the architecture reserves.
    pub(crate) const fn delivery(self) -> Option<Delivery> {
        Delivery::of(self.0)
    }

    /// This entry with every bit a source of this shape does not have taken
    /// out.
    ///
    /// Which is the two the controller owns, the fields the shape does not
    /// include, and everything the architecture reserves — all of which arrive
    /// here together, because the usual way to get one is to read a register.
    pub(crate) const fn writable(self, shape: Shape) -> Self {
        Self(self.0 & shape.writable())
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

#[cfg(test)]
mod tests {
    use descriptors::Vector;

    use super::{Delivery, Entry, Polarity, Shape, Trigger};

    #[test]
    fn every_delivery_encodes_where_the_architecture_puts_it() {
        assert_eq!(Entry::new(Delivery::Fixed(Vector::new(0x20))).bits(), 0x20);
        assert_eq!(Entry::new(Delivery::SystemManagement).bits(), 0x200);
        assert_eq!(Entry::new(Delivery::NonMaskable).bits(), 0x400);
        assert_eq!(Entry::new(Delivery::External).bits(), 0x700);
    }

    #[test]
    fn a_masked_entry_names_the_lowest_vector_the_platform_may_assign() {
        assert_eq!(Entry::masked().bits(), 0x0001_0020);
        assert!(Entry::masked().is_masked());
        assert_eq!(
            Entry::masked().delivery(),
            Some(Delivery::Fixed(Vector::FIRST_EXTERNAL))
        );
    }

    #[test]
    fn wiring_replaces_both_fields_rather_than_only_setting_them() {
        let entry = Entry::new(Delivery::Fixed(Vector::new(0x20)));
        assert_eq!(
            entry.wired(Polarity::ActiveHigh, Trigger::Edge).bits(),
            0x20
        );
        assert_eq!(
            entry.wired(Polarity::ActiveLow, Trigger::Edge).bits(),
            0x2020
        );
        assert_eq!(
            entry.wired(Polarity::ActiveHigh, Trigger::Level).bits(),
            0x8020
        );
        assert_eq!(
            entry.wired(Polarity::ActiveLow, Trigger::Level).bits(),
            0xA020
        );
        // The case a one-way set gets wrong: an entry read back from a
        // controller firmware wired the other way.
        let firmware = Entry::from_bits(0xA020);
        assert_eq!(
            firmware.wired(Polarity::ActiveHigh, Trigger::Edge).bits(),
            0x20
        );
    }

    #[test]
    fn masking_keeps_everything_else_and_unmasking_gives_it_back() {
        let armed = Entry::from_bits(0x0002_A0EF);
        assert_eq!(armed.delivering(false).bits(), 0x0003_A0EF);
        assert_eq!(
            armed.delivering(false).delivering(true).bits(),
            armed.bits()
        );
    }

    #[test]
    fn the_bits_the_controller_owns_are_read_where_the_architecture_puts_them() {
        assert!(Entry::from_bits(1 << 12).pending());
        assert!(!Entry::from_bits(!(1 << 12)).pending());
        assert!(Entry::from_bits(1 << 14).remote_irr());
        assert!(Entry::from_bits(1 << 16).is_masked());
    }

    #[test]
    fn a_reserved_delivery_encoding_is_no_delivery_at_all() {
        for mode in [0b001, 0b011, 0b101, 0b110] {
            assert_eq!(Entry::from_bits(mode << 8).delivery(), None, "{mode:#b}");
        }
    }

    #[test]
    fn only_a_pin_may_defer_to_a_legacy_controller() {
        assert!(Shape::Pin.allows(Delivery::External));
        assert!(!Shape::Internal.allows(Delivery::External));
        assert!(!Shape::Timer.allows(Delivery::External));
    }

    #[test]
    fn the_timer_delivers_ordinary_interrupts_and_nothing_else() {
        assert!(Shape::Timer.allows(Delivery::Fixed(Vector::FIRST_EXTERNAL)));
        assert!(!Shape::Timer.allows(Delivery::NonMaskable));
        assert!(!Shape::Timer.allows(Delivery::SystemManagement));
    }

    #[test]
    fn writing_an_entry_drops_every_bit_the_source_does_not_have() {
        // Everything set: both bits the controller owns, both wiring bits, the
        // timer's counting mode, and every reserved bit above them.
        let everything = Entry::from_bits(u32::MAX);
        assert_eq!(everything.writable(Shape::Internal).bits(), 0x0001_07FF);
        assert_eq!(everything.writable(Shape::Pin).bits(), 0x0001_A7FF);
        assert_eq!(everything.writable(Shape::Timer).bits(), 0x0007_07FF);
    }

    #[test]
    fn an_entry_built_here_is_already_writable_by_every_shape() {
        for shape in [Shape::Timer, Shape::Pin, Shape::Internal] {
            let masked = Entry::masked();
            assert_eq!(masked.writable(shape), masked, "{shape:?}");
        }
        let fixed = Entry::new(Delivery::Fixed(Vector::new(0xFE)));
        assert_eq!(fixed.writable(Shape::Timer), fixed);
        let wired = Entry::new(Delivery::NonMaskable).wired(Polarity::ActiveLow, Trigger::Level);
        assert_eq!(wired.writable(Shape::Pin), wired);
    }
}
