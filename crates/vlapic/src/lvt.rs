//! The local vector table: how the controller's own sources reach the
//! processor.
//!
//! Everything the controller raises itself rather than receives from elsewhere
//! — its timer expiring, either of its two local interrupt pins being
//! asserted, an error it noticed in its own operation, and on processors that
//! have them the corrected-machine-check, thermal and performance-monitoring
//! sources — is delivered through one of seven registers of the same shape.
//! Each gives its source a vector and, in most of them, a delivery mode, and
//! each carries a mask bit that reset leaves set: nothing here can deliver
//! anything until software has said what it should deliver.
//!
//! The seven are not interchangeable, and that is the reason this module exists
//! as more than a bit layout. A field defined in one entry is reserved in
//! another: pin polarity and trigger mode describe a wire and so exist only in
//! the two pin entries, the timer-mode field exists only in the timer entry,
//! and the timer and error entries have no delivery-mode field at all because
//! they are always fixed. [`Entry::writable`] is where that per-entry shape
//! lives, so that a write which sets a bit reserved in the entry it names has
//! that bit dropped instead of stored and a later read never reports state the
//! architecture says cannot exist there.
//!
//! [`Entry::ALL`] is in the order the architecture counts these in, which is
//! what the version register reports the highest index of. It is deliberately
//! not the order the registers sit at — the corrected-machine-check entry is
//! last in the count and first in memory — so nothing may derive one order from
//! the other.

use bitfield_struct::bitfield;
use descriptors::Vector;

use crate::register::Register;

#[bitfield(u32)]
#[derive(PartialEq, Eq)]
/// One local vector table entry, in the layout all seven share.
///
/// Which of these fields mean anything depends on which entry a value came
/// from, and [`Entry::writable`] is what answers that. A field reserved in the
/// entry holding it reads back zero, because the write that would have set it
/// had the bit removed first.
pub(crate) struct Lvt {
    /// The vector this source is delivered on.
    ///
    /// Read only for fixed delivery. The other modes are events the processor
    /// takes by their own architectural entry point, and the vector is ignored
    /// for them.
    #[bits(8, from = Vector::new, into = Vector::number)]
    pub(crate) vector: Vector,
    /// How the interrupt is delivered, in the encoding [`Delivery::from_bits`]
    /// decodes. Reserved in the timer and error entries, which deliver fixed
    /// and nothing else.
    #[bits(3)]
    pub(crate) delivery: u8,
    /// Reserved.
    __: bool,
    /// Whether a delivery from this source is still in flight: clear while the
    /// controller is idle with respect to it, set from the moment the interrupt
    /// is accepted for delivery until delivery completes. The controller writes
    /// this; software cannot.
    pub(crate) send_pending: bool,
    /// Whether the pin is asserted low rather than high. Reserved outside the
    /// two pin entries, since only they describe a wire.
    pub(crate) active_low: bool,
    /// Whether a level-triggered interrupt from this pin has been accepted and
    /// not yet acknowledged. Set when the controller accepts the interrupt and
    /// cleared by the guest's end-of-interrupt, and meaningless for an
    /// edge-triggered one. Reserved outside the two pin entries, and written by
    /// the controller rather than by software.
    pub(crate) remote_irr: bool,
    /// Whether the pin is level triggered rather than edge triggered, which is
    /// what decides whether an acknowledgement is owed for it. Reserved outside
    /// the two pin entries.
    pub(crate) level_triggered: bool,
    /// Whether this source is stopped from delivering anything. Set at reset in
    /// every entry, so that a source cannot fire on a vector nobody chose.
    pub(crate) masked: bool,
    /// How the timer counts, in the encoding [`TimerMode::from_bits`] decodes.
    /// Reserved outside the timer entry.
    #[bits(2)]
    pub(crate) timer_mode: u8,
    /// Reserved.
    #[bits(13)]
    __: u32,
}

/// How an entry's interrupt is delivered to the processor.
///
/// Three of the eight encodings a three-bit field can hold are reserved, and no
/// entry accepts one; a further two are accepted only by the entries that
/// describe a pin, because they are how an external controller's own signalling
/// is carried in over a wire rather than a mode a local source may choose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum Delivery {
    /// Delivered on the vector in the entry, which is the only mode that reads
    /// it.
    Fixed = 0b000,
    /// Delivered as a system-management interrupt, which the processor takes
    /// through its own entry point.
    SystemManagement = 0b010,
    /// Delivered as a non-maskable interrupt, which no masking holds off.
    NonMaskable = 0b100,
    /// Delivered as an INIT, resetting the processor's state without a start-up
    /// message.
    Init = 0b101,
    /// Delivered as though from an external interrupt controller, whose
    /// acknowledgement cycle supplies the vector.
    External = 0b111,
}

impl Delivery {
    /// The mode this encoding names, or `None` for one no entry accepts.
    ///
    /// Refusing rather than substituting is the point: a reserved delivery mode
    /// written into a real controller has undefined behaviour, so a guest that
    /// writes one is told nothing was stored rather than quietly given a mode
    /// it did not ask for.
    pub(crate) const fn from_bits(bits: u8) -> Option<Self> {
        match bits {
            0b000 => Some(Self::Fixed),
            0b010 => Some(Self::SystemManagement),
            0b100 => Some(Self::NonMaskable),
            0b101 => Some(Self::Init),
            0b111 => Some(Self::External),
            _ => None,
        }
    }
}

/// How the timer counts, which the timer entry alone carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum TimerMode {
    /// Counts the initial count down once and stops.
    OneShot = 0b00,
    /// Counts it down and reloads it, so the interrupt repeats at a fixed
    /// period.
    Periodic = 0b01,
    /// Ignores the counters entirely and fires when the time-stamp counter
    /// reaches the value in the deadline register.
    Deadline = 0b10,
}

impl TimerMode {
    /// The mode this encoding names, or `None` for the reserved one.
    pub(crate) const fn from_bits(bits: u8) -> Option<Self> {
        match bits {
            0b00 => Some(Self::OneShot),
            0b01 => Some(Self::Periodic),
            0b10 => Some(Self::Deadline),
            _ => None,
        }
    }
}

/// Which of the seven entries a value belongs to.
///
/// The distinction is not cosmetic: it is what says which fields of an [`Lvt`]
/// exist and which delivery modes may be written into it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Entry {
    /// The controller's own timer.
    Timer,
    /// The first local interrupt pin, which firmware conventionally wires the
    /// legacy 8259 through.
    Lint0,
    /// The second local interrupt pin, conventionally the non-maskable
    /// interrupt.
    Lint1,
    /// Errors the controller noticed in its own operation.
    Error,
    /// Performance-monitoring counter overflow.
    Performance,
    /// The thermal sensor.
    Thermal,
    /// Corrected machine-check errors, which are reported rather than fatal.
    CorrectedMachineCheck,
}

impl Entry {
    /// How many entries the local vector table has.
    pub(crate) const COUNT: usize = 7;

    /// One less than [`Entry::COUNT`], which is the form the version register
    /// reports it in — so that a controller always has at least one entry and
    /// the field cannot wrap. Written as a byte because that is the width of
    /// the field, and checked against the count below.
    pub(crate) const MAX_INDEX: u8 = 6;

    /// Every entry, in the order the architecture counts them.
    ///
    /// The version register reports one less than this length, so the position
    /// of an entry here is architectural rather than an implementation detail.
    pub(crate) const ALL: [Self; Self::COUNT] = [
        Self::Timer,
        Self::Lint0,
        Self::Lint1,
        Self::Error,
        Self::Performance,
        Self::Thermal,
        Self::CorrectedMachineCheck,
    ];

    /// What reset leaves in every entry: masked, and nothing else set.
    ///
    /// Uniform across all seven, so a controller coming out of reset delivers
    /// nothing at all until software unmasks a source it has given a vector.
    pub(crate) const RESET: u32 = Lvt::new().with_masked(true).into_bits();

    /// Its position in [`Entry::ALL`].
    pub(crate) const fn index(self) -> usize {
        self as usize
    }

    /// The register this entry is read and written through.
    pub(crate) const fn register(self) -> Register {
        match self {
            Self::Timer => Register::LVT_TIMER,
            Self::Lint0 => Register::LVT_LINT0,
            Self::Lint1 => Register::LVT_LINT1,
            Self::Error => Register::LVT_ERROR,
            Self::Performance => Register::LVT_PERFORMANCE,
            Self::Thermal => Register::LVT_THERMAL,
            Self::CorrectedMachineCheck => Register::LVT_CORRECTED_MACHINE_CHECK,
        }
    }

    /// The entry a register holds, or `None` for a register that is not one.
    ///
    /// Searched rather than tabulated so that [`Entry::register`] stays the one
    /// place an offset is attached to an entry.
    pub(crate) fn of(register: Register) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|entry| entry.register() == register)
    }

    /// Which bits of this entry software may set. Everything else is reserved
    /// in it and is dropped from a write rather than stored.
    ///
    /// The delivery-status and remote-IRR bits are absent from every answer:
    /// both are the controller's to report, and letting a guest write either
    /// would let it claim a delivery that never happened or retire one that is
    /// still outstanding.
    pub(crate) const fn writable(self) -> u32 {
        WRITABLE_COMMON
            | match self {
                Self::Timer => WRITABLE_TIMER_MODE,
                Self::Lint0 | Self::Lint1 => WRITABLE_DELIVERY | WRITABLE_PIN,
                Self::Error => 0,
                Self::Performance | Self::Thermal | Self::CorrectedMachineCheck => {
                    WRITABLE_DELIVERY
                }
            }
    }

    /// Whether this entry accepts that delivery mode.
    ///
    /// [`Delivery::Init`] and [`Delivery::External`] are refused everywhere but
    /// the two pins. Both describe something arriving over a wire from outside
    /// the processor, and neither means anything for a source the controller
    /// raises itself.
    pub(crate) const fn allows(self, delivery: Delivery) -> bool {
        match delivery {
            Delivery::Fixed => true,
            Delivery::SystemManagement | Delivery::NonMaskable => self.has_delivery(),
            Delivery::Init | Delivery::External => matches!(self, Self::Lint0 | Self::Lint1),
        }
    }

    /// Whether this entry has a delivery-mode field at all.
    ///
    /// The timer and error entries do not: bits 10:8 are reserved in them and
    /// their delivery is fixed by the architecture.
    pub(crate) const fn has_delivery(self) -> bool {
        !matches!(self, Self::Timer | Self::Error)
    }
}

/// The vector and the mask bit, which every entry lets software set.
const WRITABLE_COMMON: u32 = Lvt::new()
    .with_vector(Vector::new(u8::MAX))
    .with_masked(true)
    .into_bits();

/// The delivery-mode field, in the five entries that have one.
const WRITABLE_DELIVERY: u32 = Lvt::new().with_delivery(DELIVERY_FIELD).into_bits();

/// Pin polarity and trigger mode, in the two entries that describe a wire.
const WRITABLE_PIN: u32 = Lvt::new()
    .with_active_low(true)
    .with_level_triggered(true)
    .into_bits();

/// The timer-mode field, in the timer entry alone.
const WRITABLE_TIMER_MODE: u32 = Lvt::new().with_timer_mode(TIMER_MODE_FIELD).into_bits();

/// A three-bit delivery-mode field with every bit set, which is what marks
/// where the field sits rather than a mode any entry accepts.
const DELIVERY_FIELD: u8 = 0b111;

/// The same for the two-bit timer-mode field.
const TIMER_MODE_FIELD: u8 = 0b11;

/// Checks the parts of the layout that are silent when wrong: an entry that
/// comes out of reset unmasked, a reserved encoding accepted as a delivery
/// mode, or a reserved field left writable all produce a working controller
/// that delivers the wrong thing.
/// The reported entry count has to be the real one.
const _: () = assert!(
    Entry::MAX_INDEX as usize + 1 == Entry::COUNT,
    "the version register must report the number of entries the table has"
);

#[cfg(test)]
mod tests {
    use descriptors::Vector;

    use super::{Delivery, Entry, Lvt, TimerMode, WRITABLE_DELIVERY, WRITABLE_TIMER_MODE};
    use crate::register::Register;

    #[test]
    fn reset_is_masked_and_nothing_more() {
        assert_eq!(Entry::RESET, 0x0001_0000);
        let entry = Lvt::from_bits(Entry::RESET);
        assert!(entry.masked());
        assert_eq!(entry.vector(), Vector::new(0));
        assert_eq!(Delivery::from_bits(entry.delivery()), Some(Delivery::Fixed));
        assert_eq!(
            TimerMode::from_bits(entry.timer_mode()),
            Some(TimerMode::OneShot)
        );
        assert!(!entry.send_pending());
        assert!(!entry.active_low());
        assert!(!entry.remote_irr());
        assert!(!entry.level_triggered());
        assert_eq!(entry.into_bits(), Entry::RESET);
    }

    #[test]
    fn every_entry_is_found_by_its_own_register() {
        for (index, entry) in Entry::ALL.into_iter().enumerate() {
            assert_eq!(entry.index(), index);
            assert_eq!(Entry::of(entry.register()), Some(entry));
        }
        assert_eq!(Entry::of(Register::SPURIOUS), None);
    }

    #[test]
    fn reserved_delivery_encodings_name_nothing() {
        for bits in [0b001, 0b011, 0b110] {
            assert_eq!(Delivery::from_bits(bits), None);
        }
    }

    #[test]
    fn corrected_machine_check_refuses_init_and_external() {
        let entry = Entry::CorrectedMachineCheck;
        assert!(entry.allows(Delivery::Fixed));
        assert!(entry.allows(Delivery::SystemManagement));
        assert!(entry.allows(Delivery::NonMaskable));
        assert!(!entry.allows(Delivery::Init));
        assert!(!entry.allows(Delivery::External));
    }

    #[test]
    fn the_timer_carries_no_delivery_mode() {
        assert!(!Entry::Timer.has_delivery());
        assert!(!Entry::Error.has_delivery());
        assert_eq!(Entry::Timer.writable() & WRITABLE_DELIVERY, 0);
        assert!(Entry::Timer.allows(Delivery::Fixed));
        assert!(!Entry::Timer.allows(Delivery::NonMaskable));
    }

    #[test]
    fn only_the_timer_may_set_the_timer_mode() {
        for entry in Entry::ALL {
            let held = entry.writable() & WRITABLE_TIMER_MODE;
            assert_eq!(held == WRITABLE_TIMER_MODE, matches!(entry, Entry::Timer));
        }
    }
}
