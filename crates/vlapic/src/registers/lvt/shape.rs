//! The shape all seven entries share: the bit layout, and the two fields whose
//! encodings mean something.
//!
//! Which of these fields exist in a given entry is
//! [`Entry::writable`](super::Entry::writable)'s to say — a field reserved in
//! the entry holding it reads back zero, because the write that would have set
//! it had the bit removed first — and which delivery modes an entry accepts is
//! [`crate::hardware::model`]'s.

use bitfield_struct::bitfield;
use descriptors::Vector;

#[bitfield(u32)]
#[derive(PartialEq, Eq)]
/// One local vector table entry, in the layout all seven share.
///
/// Which of these fields mean anything depends on which entry a value came
/// from, and [`Entry::writable`](super::Entry::writable) is what answers that.
/// A field reserved in the entry holding it reads back zero, because the write
/// that would have set it had the bit removed first.
pub(crate) struct Lvt {
    /// The vector this source is delivered on.
    ///
    /// Consulted only for fixed delivery. The other modes are events the
    /// processor takes by their own architectural entry point, and the field is
    /// not read for them.
    #[bits(8, from = Vector::new, into = Vector::number)]
    pub(crate) vector: Vector,
    /// How the interrupt is delivered, in the encoding [`Delivery::from_bits`]
    /// decodes. Reserved in the timer and error entries, which deliver fixed
    /// and nothing else.
    #[bits(3)]
    pub(crate) delivery: u8,
    /// Reserved.
    __: bool,
    /// Whether a delivery from this source has been accepted by the controller
    /// and not yet handed to the processor: set from the moment the interrupt
    /// is accepted for delivery until it reaches the request register, and
    /// clear while the controller is idle with respect to the source. The
    /// controller writes this; software cannot, and a write of it is
    /// ignored rather than refused.
    pub(crate) send_pending: bool,
    /// Whether the pin is asserted low rather than high. Reserved outside the
    /// two pin entries, since only they describe a wire.
    pub(crate) active_low: bool,
    /// Whether a level-triggered interrupt from this pin has been accepted and
    /// not yet acknowledged. Set when the controller accepts the interrupt and
    /// cleared by the guest's end-of-interrupt, and meaningless for an
    /// edge-triggered one. Reserved outside the two pin entries, and written by
    /// the controller rather than by software — a write of it inside them is
    /// ignored rather than refused.
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
    /// A reserved encoding is retained rather than rejected: the field is
    /// writable in every entry that has one, so a guest that writes a reserved
    /// mode reads it back, exactly as it would from hardware. What refuses it
    /// is everything downstream — nothing decodes it to a mode, and a
    /// source holding one is programmed masked rather than programmed with
    /// a delivery mode real hardware calls undefined. Every caller of this
    /// therefore has to handle `None`, and handling it means delivering
    /// nothing.
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

#[cfg(test)]
mod tests {
    use super::Delivery;

    #[test]
    fn reserved_delivery_encodings_name_nothing() {
        for bits in [0b001, 0b011, 0b110] {
            assert_eq!(Delivery::from_bits(bits), None);
        }
    }
}
