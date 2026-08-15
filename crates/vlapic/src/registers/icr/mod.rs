//! The interrupt command register, which is how a guest asks for an interrupt
//! to be sent.
//!
//! One register with two shapes. Through the memory-mapped page it is two
//! 32-bit registers: the high one holds nothing but the destination, in its top
//! byte, and is written first, because writing the low one is what sends the
//! command. Through x2APIC it is a single 64-bit model-specific register
//! written in one instruction, with the destination widened to the whole of the
//! upper half. [`Command`] is both — sixty-four bits assembled from whichever
//! face the guest used — and the fields whose meaning depends on the face take
//! the mode as an argument rather than being decoded twice.
//!
//! # What the architecture no longer sends
//!
//! Level and trigger mode are left over from the external APIC bus, where a
//! command could be asserted and de-asserted like a wire. Nothing since
//! delivers a level-triggered interprocessor interrupt: whatever software
//! writes, the command goes out edge triggered. One encoding survives, and it
//! is not an interrupt at all — INIT with the level bit clear and the
//! trigger-mode bit set is a synchronisation message that resets every target's
//! arbitration identifier and delivers nothing else.
//!
//! # Deciding here rather than at every caller
//!
//! A guest may write any sixty-four bits it likes, and most of what the
//! architecture says about them is of the form "this combination is not a
//! command". Those rules are applied here rather than at every caller, and they
//! are in two places because they are two different kinds of rule.
//!
//! [`Command::delivery`] decodes the field: the two reserved encodings and
//! lowest priority in x2APIC answer with nothing, because no mode is named.
//! [`Command::legal`] judges the whole command — which shorthands a mode may be
//! addressed with, which fields it must leave clear — and is asked before a
//! single target is worked out. That order is the point of it: a command
//! resolved first and judged afterwards has already reset or started some of
//! the processors it named.
//!
//! Fields the hardware reads past are answered the same way.
//! [`Command::trigger`] reports [`Trigger::Edge`] for everything but the one
//! message above, [`Command::vector`] reports zero for the delivery modes that
//! carry no vector, and [`Command::destination_mode`] reports
//! [`DestinationMode::Physical`] whenever a shorthand has already named the
//! targets.

mod command;
mod decode;
mod legal;

pub(crate) use crate::registers::icr::decode::{Delivery, DestinationMode, Shorthand, Trigger};

use core::sync::atomic::Ordering;

use crate::registers::Vlapic;

/// One interrupt command, in the sixty-four bits both faces describe it with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Command(u64);

impl Vlapic {
    /// The interrupt command register as it stands.
    ///
    /// Kept because the older face writes it in two halves and the destination
    /// has to survive between them. A guest reading it back gets what it wrote,
    /// which the architecture does not promise but does not forbid, and which
    /// costs nothing to be honest about.
    pub(crate) fn command(&self) -> Command {
        Command::from_bits(self.command.load(Ordering::Acquire))
    }

    /// Sets the destination half, which sends nothing on its own.
    ///
    /// Everything below the top byte is reserved in this half, and is dropped
    /// rather than stored: a guest reading the register back must not find bits
    /// the architecture says read as zero.
    pub(crate) fn set_command_high(&self, value: u32) {
        let low = self.command().low();
        self.command.store(
            Command::from_halves(low, value & Command::WRITABLE_HIGH).bits(),
            Ordering::Release,
        );
    }

    /// Sets the half whose write sends the command, and answers with the whole
    /// of what is to be sent.
    pub(crate) fn set_command_low(&self, value: u32) -> Command {
        let high = self.command().high();
        let command = Command::from_halves(value & Command::WRITABLE_LOW, high);
        self.command.store(command.bits(), Ordering::Release);
        command
    }

    /// Sets the whole register at once, which is how x2APIC writes it.
    pub(crate) fn set_command(&self, value: u64) -> Command {
        let command = Command::from_bits(value);
        self.command.store(command.bits(), Ordering::Release);
        command
    }
}
