//! The register as sixty-four bits: which of them software owns, and how the
//! two faces divide them up.
//!
//! Nothing here reads a field for its meaning; [`super::decode`] does that.
//! What is here is the layout — stated once, as [`Field`]s — and the three
//! writable masks built out of it, so that a mask cannot drift away from the
//! fields it is made of.

use crate::registers::icr::Command;

impl Command {
    /// The destination every processor answers to in x2APIC, in the logical
    /// destination mode as much as the physical one.
    ///
    /// The memory-mapped face spells the same thing with all eight bits of its
    /// narrower field set, which is [`BROADCAST_XAPIC`] and is a different
    /// number. Which of the two a command means is decided by the face the
    /// *sender* wrote it through, because that is the face whose width the
    /// field has — see [`Command::is_broadcast`].
    pub(crate) const BROADCAST: u32 = u32::MAX;

    /// Which bits of the low half software may set.
    ///
    /// Every field a guest owns and nothing else: the delivery-status bit at 12
    /// is the controller's own report of whether a command is still going out,
    /// and bits 13, 17:16 and 31:20 are reserved. A write through the
    /// memory-mapped face is masked with this, which is what keeps a guest's
    /// stray bits from being read back; the wide face, which permits the same
    /// fields and is built from this, faults on a bit outside it rather than
    /// dropping it.
    pub(crate) const WRITABLE_LOW: u32 = VECTOR.mask()
        | DELIVERY.mask()
        | DESTINATION_MODE.mask()
        | LEVEL.mask()
        | TRIGGER.mask()
        | SHORTHAND.mask();

    /// Which bits of the high half software may set through the memory-mapped
    /// face.
    ///
    /// The destination is the top byte and everything below it is reserved. A
    /// guest that writes one of those must read it back as zero, and the
    /// destination arithmetic reads only the top byte anyway — so storing the
    /// rest would be state that is wrong without being consulted, which is the
    /// kind that survives until something starts consulting it.
    pub(crate) const WRITABLE_HIGH: u32 = DESTINATION_XAPIC.mask();

    /// Which bits the whole register may hold in x2APIC.
    ///
    /// The same fields of the low half as the older face, because software
    /// forms a command the same way whichever face it writes it through, and
    /// the whole of the upper half instead of one byte of it, because that is
    /// where the wider destination went.
    ///
    /// Reserved here means `RsvdZ`: writing a non-zero value into one is a
    /// general protection fault rather than something quietly dropped, so what
    /// this permits is what conforming software is able to send at all. That is
    /// why level and trigger mode are in it. Neither describes anything any
    /// more — the bus they asserted over is gone, and hardware reads past them
    /// — but the architecture tells software the level bit must be set for
    /// every delivery mode except the INIT de-assert, and software does exactly
    /// that: an operating system's start-up sequence writes the level bit with
    /// its INIT, and firmware sets it on every interrupt it sends. A controller
    /// that faulted on it could not be given a command by conforming software
    /// at all.
    ///
    /// The delivery-status bit is not in it, and that is the one place this
    /// face is genuinely narrower: an x2APIC write does not return until
    /// the command has been accepted, so there is no delivery in flight for
    /// software to be told about, and this vendor requires the bit to be
    /// written as zero. Bit 13, bits 17:16 and bits 31:20 are reserved on
    /// both faces.
    pub(crate) const WRITABLE_X2APIC: u64 = Self::WRITABLE_LOW as u64 | (u32::MAX as u64) << HALF;

    /// The command sixty-four bits describe, as x2APIC presents them.
    pub(crate) const fn from_bits(bits: u64) -> Self {
        Self(bits)
    }

    /// The command a pair of memory-mapped halves describes, the destination
    /// being the top byte of `high`.
    pub(crate) const fn from_halves(low: u32, high: u32) -> Self {
        Self::from_bits(((high as u64) << HALF) | low as u64)
    }

    /// The whole register, as x2APIC reads and writes it.
    pub(crate) const fn bits(self) -> u64 {
        self.0
    }

    /// The half a guest writes to send the command.
    pub(crate) const fn low(self) -> u32 {
        truncate(self.bits())
    }

    /// The half that carries the destination, in whichever width the face gives
    /// it.
    pub(crate) const fn high(self) -> u32 {
        truncate(self.bits() >> HALF)
    }
}

/// A run of adjacent bits in the register, named by where it starts and how
/// wide it is.
///
/// The layout is stated once, here and in the constants below, so that no
/// accessor spells out a shift or a mask of its own and the writable mask
/// cannot drift away from the fields it is made of.
#[derive(Clone, Copy)]
pub(super) struct Field {
    /// How far above bit zero the field starts.
    shift: u32,
    /// How many bits it spans.
    width: u32,
}

impl Field {
    /// The field of `width` bits starting at `shift`.
    pub(super) const fn new(shift: u32, width: u32) -> Self {
        Self { shift, width }
    }

    /// The field's value, brought down to bit zero.
    pub(super) const fn get(self, bits: u32) -> u32 {
        (bits & self.mask()) >> self.shift
    }

    /// Whether any of the field's bits are set, which for a one-bit field is
    /// the field itself.
    pub(super) const fn test(self, bits: u32) -> bool {
        bits & self.mask() != 0
    }

    /// The field's bits, where they sit.
    pub(super) const fn mask(self) -> u32 {
        ((1 << self.width) - 1) << self.shift
    }
}

/// The low thirty-two bits of a quadword.
///
/// One place where the upper half is dropped, so that splitting the register
/// into the halves the memory-mapped face presents is the only thing that ever
/// narrows it.
#[expect(
    clippy::cast_possible_truncation,
    reason = "discarding the upper half is the whole of what this does"
)]
const fn truncate(bits: u64) -> u32 {
    bits as u32
}

/// How many bits one half of the register spans, and so how far above the low
/// half the high one sits.
const HALF: u32 = 32;

/// The vector, which is the interrupt itself for the delivery modes that carry
/// one and reserved for the rest.
pub(super) const VECTOR: Field = Field::new(0, 8);

/// Which of the eight delivery modes the command names.
pub(super) const DELIVERY: Field = Field::new(8, 3);

/// Whether the destination is an identifier or a logical mask.
pub(super) const DESTINATION_MODE: Field = Field::new(11, 1);

/// Assert or de-assert, read only for the INIT de-assert.
pub(super) const LEVEL: Field = Field::new(14, 1);

/// Edge or level, read only for the INIT de-assert.
pub(super) const TRIGGER: Field = Field::new(15, 1);

/// Which shorthand, if any, names the targets in place of the destination.
pub(super) const SHORTHAND: Field = Field::new(18, 2);

/// The destination the memory-mapped face names every processor with: all eight
/// bits of its narrower field.
///
/// Derived from the field rather than written out, so that it cannot drift away
/// from the width the field actually has.
pub(super) const BROADCAST_XAPIC: u32 = DESTINATION_XAPIC.get(u32::MAX);

/// The destination within the high half of the memory-mapped register, where it
/// is a byte at the top rather than the whole word.
pub(super) const DESTINATION_XAPIC: Field = Field::new(24, 8);

#[cfg(test)]
mod tests {
    //! The masks are written out as the values a guest may write, so that a
    //! test fails when a field moves rather than moving with it.

    use crate::registers::icr::Command;

    #[test]
    fn the_wide_face_keeps_the_fields_software_is_told_to_write() {
        // Level and trigger mode. Hardware reads past both in this face, and
        // software sets them anyway because the architecture tells it to — so a
        // controller that treated them as reserved would fault every conforming
        // interprocessor interrupt and the whole start-up sequence with them.
        assert_eq!(
            Command::WRITABLE_X2APIC & (1 << 14 | 1 << 15),
            1 << 14 | 1 << 15
        );
        // The delivery-status bit, which this face does not have and this
        // vendor requires to be written as zero.
        assert_eq!(Command::WRITABLE_X2APIC & (1 << 12), 0);
        // What is left: the same low-half fields as the older face, and the
        // whole of the upper half for the wider destination.
        assert_eq!(Command::WRITABLE_X2APIC, 0xFFFF_FFFF_000C_CFFF);
        assert_eq!(
            Command::WRITABLE_X2APIC & u64::from(u32::MAX),
            u64::from(Command::WRITABLE_LOW),
            "the two faces permit the same fields of the half that sends the command"
        );
    }

    #[test]
    fn the_high_half_keeps_only_the_destination() {
        assert_eq!(Command::WRITABLE_HIGH, 0xFF00_0000);
    }

    #[test]
    fn only_the_fields_a_guest_owns_are_writable() {
        // Vector, delivery mode and destination mode; level and trigger mode;
        // the shorthand. Not the delivery-status bit at 12, and not bit 13,
        // bits 17:16 or bits 31:20.
        assert_eq!(Command::WRITABLE_LOW, 0x000C_CFFF);
    }
}
