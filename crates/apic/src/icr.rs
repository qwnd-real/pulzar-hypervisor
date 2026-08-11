//! The interrupt command register: how one processor interrupts another.
//!
//! Writing it is the only way to send anything to another processor, and the
//! three things this crate needs of it are quite different from each other. An
//! ordinary interprocessor interrupt delivers a vector. The startup sequence
//! delivers no vector at all — `INIT` holds a processor in reset and `SIPI`
//! tells it where to begin — and is the reason the register has modes that look
//! nothing like an interrupt.
//!
//! So a [`Command`] is built by naming what it is rather than by setting bits,
//! which is what keeps a field out of the wrong position and the reserved bits
//! at zero. Two things it cannot keep out are checked instead, because they are
//! combinations of fields that are each individually legal:
//!
//! - `NMI`, `INIT` and `SIPI` may not be addressed with a shorthand that
//!   includes the processor sending them. One vendor calls the combination
//!   invalid and the other leaves it undefined, and what it would mean is a
//!   processor resetting out from under the hypervisor.
//! - The all-ones identifier is not a processor. In both interfaces it is the
//!   architecture's physical broadcast, so a command meant for one processor
//!   addressed to it reaches every processor on the machine.
//!
//! `INIT` is always the level-assert form modern processors expect; the
//! deassert pulse that discrete controllers needed is not offered, because no
//! processor this runs on has one.
//!
//! # Addressing
//!
//! A destination is either one processor, named by its identifier, or a
//! shorthand. The shorthands exist because "everyone but me" is both the common
//! case and the one that cannot be expressed as an identifier. Under x2APIC the
//! identifier field is the whole upper word; under the older interface it is
//! the top eight bits of it, which is why an identifier that does not fit is
//! refused rather than silently truncated into somebody else's.

use cpu::ApicId;
use descriptors::Vector;

use crate::{ApicError, Mode};

/// Who a command goes to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    /// The one processor with this identifier.
    One(ApicId),
    /// Every processor including the one sending.
    All,
    /// Every processor except the one sending.
    Others,
    /// The processor sending, and nobody else.
    Myself,
}

impl Target {
    /// The destination shorthand field's encoding.
    const fn shorthand(self) -> u64 {
        match self {
            Self::One(_) => 0b00,
            Self::Myself => 0b01,
            Self::All => 0b10,
            Self::Others => 0b11,
        }
    }

    /// Whether the processor sending is one of the processors addressed.
    ///
    /// True of the two shorthands that include the sender, and of neither of
    /// the forms that cannot: a command to everyone else excludes it by
    /// definition, and one addressed to an identifier is refused for the
    /// sender's own by whoever chose the identifier rather than here — this
    /// module does not know which processor it is running on.
    const fn includes_sender(self) -> bool {
        matches!(self, Self::Myself | Self::All)
    }

    /// The destination field's contents, which every shorthand but the first
    /// leaves unused.
    ///
    /// # Errors
    ///
    /// [`ApicError::IdTooWide`] if the older interface is in use and the
    /// identifier does not fit its eight-bit field — truncating would deliver
    /// the interrupt to a different processor, which is worse than refusing —
    /// or [`ApicError::BroadcastId`] for the all-ones identifier of either
    /// interface, which is not a processor but the architecture's way of
    /// addressing all of them.
    const fn destination(self, mode: Mode) -> Result<u64, ApicError> {
        let Self::One(id) = self else {
            return Ok(0);
        };
        match mode {
            Mode::X2Apic if id.get() == u32::MAX => Err(ApicError::BroadcastId { apic_id: id }),
            Mode::X2Apic => Ok((id.get() as u64) << u32::BITS),
            Mode::XApic if id.get() == XAPIC_BROADCAST => {
                Err(ApicError::BroadcastId { apic_id: id })
            }
            Mode::XApic if id.fits_xapic() => Ok((id.get() as u64) << XAPIC_DESTINATION_SHIFT),
            Mode::XApic => Err(ApicError::IdTooWide { apic_id: id }),
        }
    }
}

/// The identifier the older interface reads as "every processor" rather than as
/// one of them, in the width the destination field is compared in. x2APIC's is
/// all ones of its wider field.
const XAPIC_BROADCAST: u32 = 0xFF;

/// Bits the older interface's eight-bit destination field is shifted by: the
/// top eight of the register's upper word.
const XAPIC_DESTINATION_SHIFT: u32 = 56;

/// Bits the destination shorthand field is shifted by.
const SHORTHAND_SHIFT: u32 = 18;

/// Bits the delivery mode field is shifted by.
const DELIVERY_SHIFT: u32 = 8;

/// Set for every command but the deassert form of `INIT`, which this crate does
/// not send.
const LEVEL_ASSERT: u64 = 1 << 14;

/// What a command asks the receiving processor to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Delivery {
    /// Deliver an ordinary interrupt on this vector.
    Fixed(Vector),
    /// Deliver a non-maskable interrupt, whatever the receiver is doing.
    NonMaskable,
    /// Reset the processor and hold it waiting for a startup command.
    Init,
    /// Start the processor executing in real mode at `vector << 12`.
    Startup(u8),
}

impl Delivery {
    /// The delivery mode field's encoding.
    const fn mode(self) -> u64 {
        match self {
            Self::Fixed(_) => 0b000,
            Self::NonMaskable => 0b100,
            Self::Init => 0b101,
            Self::Startup(_) => 0b110,
        }
    }

    /// The vector field's contents, which means a vector for one mode, a page
    /// of physical memory for another, and nothing for the rest.
    const fn vector(self) -> u64 {
        match self {
            Self::Fixed(vector) => vector.number() as u64,
            Self::Startup(page) => page as u64,
            _ => 0,
        }
    }

    /// Whether this delivery may be addressed to the processor sending it.
    ///
    /// Only an ordinary interrupt may. The other three are asked of a processor
    /// from outside it: two vendors between them call self-addressed `NMI`,
    /// `INIT` and `SIPI` invalid and undefined, and the one thing that is
    /// certain about a processor being reset by the code running on it is that
    /// the code does not continue.
    const fn may_reach_sender(self) -> bool {
        matches!(self, Self::Fixed(_))
    }
}

/// One interrupt command, ready to be written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Command {
    delivery: Delivery,
    target: Target,
}

impl Command {
    /// A command asking `target` to do `delivery`.
    #[must_use]
    pub const fn new(delivery: Delivery, target: Target) -> Self {
        Self { delivery, target }
    }

    /// The command as the register holds it.
    ///
    /// The destination mode field stays zero throughout: physical addressing,
    /// where a destination is an identifier rather than a bitmask of them.
    /// Logical addressing exists to interrupt a set of processors at once, and
    /// nothing here wants a set that the shorthands do not already name.
    ///
    /// # Errors
    ///
    /// [`ApicError::IllegalVector`] if the command names a vector no controller
    /// may deliver — which a controller answers by delivering nothing and
    /// latching an error, so that the interrupt a caller is waiting for simply
    /// never arrives; [`ApicError::SelfAddressed`] for a reset, startup or
    /// non-maskable interrupt addressed to a set that includes the processor
    /// sending it; or [`ApicError::IdTooWide`] or [`ApicError::BroadcastId`] if
    /// the target's identifier is not one this interface can name one processor
    /// by.
    pub(crate) const fn bits(self, mode: Mode) -> Result<u64, ApicError> {
        if let Delivery::Fixed(vector) = self.delivery
            && !crate::deliverable(vector)
        {
            return Err(ApicError::IllegalVector { vector });
        }
        if self.target.includes_sender() && !self.delivery.may_reach_sender() {
            return Err(ApicError::SelfAddressed);
        }
        let destination = match self.target.destination(mode) {
            Ok(destination) => destination,
            Err(error) => return Err(error),
        };
        Ok(destination
            | (self.target.shorthand() << SHORTHAND_SHIFT)
            | LEVEL_ASSERT
            | (self.delivery.mode() << DELIVERY_SHIFT)
            | self.delivery.vector())
    }
}

#[cfg(test)]
mod tests {
    use cpu::ApicId;
    use descriptors::Vector;

    use super::{Command, Delivery, Target};
    use crate::{ApicError, Mode};

    /// An ordinary processor identifier, and a vector and a startup page that
    /// are distinguishable from every field around them.
    const ID: ApicId = ApicId::new(0x2A);
    const VECTOR: Vector = Vector::new(0xF0);
    const PAGE: u8 = 0x08;

    #[test]
    fn a_command_to_one_processor_encodes_the_way_both_interfaces_hold_it() {
        let cases = [
            (
                Delivery::Fixed(VECTOR),
                0x2A00_0000_0000_40F0,
                0x0000_002A_0000_40F0,
            ),
            (
                Delivery::NonMaskable,
                0x2A00_0000_0000_4400,
                0x0000_002A_0000_4400,
            ),
            (Delivery::Init, 0x2A00_0000_0000_4500, 0x0000_002A_0000_4500),
            (
                Delivery::Startup(PAGE),
                0x2A00_0000_0000_4608,
                0x0000_002A_0000_4608,
            ),
        ];
        for (delivery, xapic, x2apic) in cases {
            let command = Command::new(delivery, Target::One(ID));
            assert_eq!(command.bits(Mode::XApic), Ok(xapic), "{delivery:?}");
            assert_eq!(command.bits(Mode::X2Apic), Ok(x2apic), "{delivery:?}");
        }
    }

    #[test]
    fn init_is_the_assert_form_and_carries_no_vector() {
        let bits = Command::new(Delivery::Init, Target::One(ID))
            .bits(Mode::X2Apic)
            .expect("an ordinary identifier can be named in x2apic");
        assert_eq!(bits & 0xFF, 0, "init's vector field is zero");
        assert_eq!(bits & (1 << 14), 1 << 14, "level is assert");
        assert_eq!(bits & (1 << 15), 0, "trigger is edge");
        assert_eq!(bits & (1 << 11), 0, "the destination is physical");
    }

    #[test]
    fn a_shorthand_leaves_the_destination_field_alone() {
        for (target, shorthand) in [
            (Target::Myself, 0b01),
            (Target::All, 0b10),
            (Target::Others, 0b11),
        ] {
            let bits = Command::new(Delivery::Fixed(VECTOR), target)
                .bits(Mode::XApic)
                .expect("an ordinary vector may be delivered to any shorthand");
            assert_eq!(bits, (shorthand << 18) | 0x4000 | 0xF0);
        }
    }

    #[test]
    fn a_reset_or_startup_addressed_to_the_sender_is_refused() {
        for delivery in [
            Delivery::NonMaskable,
            Delivery::Init,
            Delivery::Startup(PAGE),
        ] {
            for target in [Target::Myself, Target::All] {
                for mode in [Mode::XApic, Mode::X2Apic] {
                    assert_eq!(
                        Command::new(delivery, target).bits(mode),
                        Err(ApicError::SelfAddressed),
                        "{delivery:?} to {target:?} in {mode}"
                    );
                }
            }
            // Everyone else is the one shorthand that cannot include the sender,
            // and the architecture allows all three there.
            assert!(
                Command::new(delivery, Target::Others)
                    .bits(Mode::XApic)
                    .is_ok()
            );
        }
    }

    #[test]
    fn the_all_ones_identifier_is_a_broadcast_rather_than_a_processor() {
        let eight_bit = ApicId::new(0xFF);
        assert_eq!(
            Command::new(Delivery::Init, Target::One(eight_bit)).bits(Mode::XApic),
            Err(ApicError::BroadcastId { apic_id: eight_bit })
        );
        let whole_word = ApicId::new(u32::MAX);
        assert_eq!(
            Command::new(Delivery::Init, Target::One(whole_word)).bits(Mode::X2Apic),
            Err(ApicError::BroadcastId {
                apic_id: whole_word
            })
        );
        // The same number is merely too wide for the older interface, which is a
        // different fault and a different message.
        assert_eq!(
            Command::new(Delivery::Init, Target::One(whole_word)).bits(Mode::XApic),
            Err(ApicError::IdTooWide {
                apic_id: whole_word
            })
        );
    }

    #[test]
    fn the_widest_identifier_each_interface_can_name_one_processor_by() {
        let last_xapic = ApicId::new(0xFE);
        assert_eq!(
            Command::new(Delivery::Fixed(VECTOR), Target::One(last_xapic)).bits(Mode::XApic),
            Ok(0xFE00_0000_0000_40F0)
        );
        let too_wide = ApicId::new(0x100);
        assert_eq!(
            Command::new(Delivery::Fixed(VECTOR), Target::One(too_wide)).bits(Mode::XApic),
            Err(ApicError::IdTooWide { apic_id: too_wide })
        );
        assert_eq!(
            Command::new(Delivery::Fixed(VECTOR), Target::One(too_wide)).bits(Mode::X2Apic),
            Ok(0x0000_0100_0000_40F0)
        );
    }

    #[test]
    fn a_vector_no_controller_may_deliver_is_refused_before_anything_is_encoded() {
        for number in [0x00, 0x0F, 0x10, 0x1F] {
            let vector = Vector::new(number);
            assert_eq!(
                Command::new(Delivery::Fixed(vector), Target::One(ID)).bits(Mode::X2Apic),
                Err(ApicError::IllegalVector { vector })
            );
        }
        for number in [0x20, 0xFF] {
            assert!(
                Command::new(Delivery::Fixed(Vector::new(number)), Target::One(ID))
                    .bits(Mode::X2Apic)
                    .is_ok()
            );
        }
    }

    #[test]
    fn a_startup_page_is_a_page_number_and_reaches_the_whole_first_megabyte() {
        for (page, low) in [(0x00, 0x4600), (0xFF, 0x46FF)] {
            let bits = Command::new(Delivery::Startup(page), Target::One(ID))
                .bits(Mode::X2Apic)
                .expect("every page number below one megabyte is nameable");
            assert_eq!(bits & 0xFFFF, low);
        }
    }

    #[test]
    fn nothing_reserved_is_ever_set() {
        /// Bit 13, the shorthand's neighbours at 17:16, and 31:20.
        const RESERVED: u64 = (1 << 13) | (0b11 << 16) | (0xFFF << 20);
        for delivery in [
            Delivery::Fixed(VECTOR),
            Delivery::NonMaskable,
            Delivery::Init,
            Delivery::Startup(PAGE),
        ] {
            for mode in [Mode::XApic, Mode::X2Apic] {
                let bits = Command::new(delivery, Target::One(ID))
                    .bits(mode)
                    .expect("an ordinary identifier and vector encode in both interfaces");
                assert_eq!(bits & RESERVED, 0, "{delivery:?} in {mode}");
            }
        }
    }
}
