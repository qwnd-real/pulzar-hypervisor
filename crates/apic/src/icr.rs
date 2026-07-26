//! The interrupt command register: how one processor interrupts another.
//!
//! Writing it is the only way to send anything to another processor, and the
//! three things this crate needs of it are quite different from each other. An
//! ordinary interprocessor interrupt delivers a vector. The startup sequence
//! delivers no vector at all — `INIT` holds a processor in reset and `SIPI`
//! tells it where to begin — and is the reason the register has modes that look
//! nothing like an interrupt.
//!
//! So a [`Command`] is built by naming what it is, not by setting bits, and the
//! combinations the architecture does not define cannot be written down. `INIT`
//! is always the level-assert form modern processors expect; the deassert pulse
//! that discrete controllers needed is not offered, because no processor this
//! runs on has one.
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

    /// The destination field's contents, which every shorthand but the first
    /// leaves unused.
    ///
    /// # Errors
    ///
    /// [`ApicError::IdTooWide`] if the older interface is in use and the
    /// identifier does not fit its eight-bit field. Truncating would deliver
    /// the interrupt to a different processor, which is worse than
    /// refusing.
    const fn destination(self, mode: Mode) -> Result<u64, ApicError> {
        let Self::One(id) = self else {
            return Ok(0);
        };
        match mode {
            Mode::X2Apic => Ok((id.get() as u64) << u32::BITS),
            Mode::XApic if id.fits_xapic() => Ok((id.get() as u64) << XAPIC_DESTINATION_SHIFT),
            Mode::XApic => Err(ApicError::IdTooWide { apic_id: id }),
        }
    }
}

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
    /// [`ApicError::IdTooWide`] if the target's identifier does not fit the
    /// interface in use.
    pub(crate) const fn bits(self, mode: Mode) -> Result<u64, ApicError> {
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
