//! Whether a command is one the architecture defines at all.
//!
//! Kept apart from [`super::decode`] because it is a different kind of rule.
//! Decoding answers what a field says; this answers whether the combination of
//! them is a command, and it is asked of the whole command before a single
//! target is worked out — which is the difference between rejecting a command
//! and half-performing one.

use crate::registers::{
    base::Mode,
    icr::{
        Command,
        command::VECTOR,
        decode::{Delivery, Shorthand},
    },
};

impl Command {
    /// Whether this is a command the architecture defines at all.
    ///
    /// Applied to the whole command before any target is worked out, which is
    /// the difference between rejecting a command and half-performing one.
    /// Every combination below is one the architecture either forbids
    /// outright or leaves undefined, and a controller that resolved targets
    /// for it first would have already reset, started or interrupted some
    /// of them by the time it noticed.
    pub(crate) const fn legal(self, mode: Mode) -> bool {
        let Some(delivery) = self.delivery(mode) else {
            return false;
        };
        match delivery {
            // Both are events with no vector, and the architecture requires the
            // field to be written as zero rather than merely ignoring it.
            Delivery::SystemManagement | Delivery::Init
                if VECTOR.get(self.low()) != 0 && !self.is_init_deassert() =>
            {
                false
            }
            // The synchronisation message is defined only as a broadcast to
            // every processor including the sender. Addressed anywhere else it
            // is not that message and is not anything else either.
            Delivery::Init if self.is_init_deassert() => {
                matches!(self.shorthand(), Shorthand::All)
            }
            // A start-up cannot be addressed to the processor that would have to
            // send it, and the shorthands that include the sender are how that
            // is expressed.
            Delivery::Startup => !matches!(self.shorthand(), Shorthand::Myself | Shorthand::All),
            _ => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::registers::{
        base::Mode,
        icr::{Command, Delivery},
    };

    #[test]
    fn a_start_up_may_not_be_addressed_to_whoever_sends_it() {
        for bits in [0x0004_0630, 0x0008_0630] {
            let command = Command::from_bits(bits);
            // Still decodes: what it asks for is a start-up either way, and it
            // is the whole command that is refused rather than the field.
            assert_eq!(command.delivery(Mode::XApic), Some(Delivery::Startup));
            assert!(!command.legal(Mode::XApic));
        }

        let rest = Command::from_bits(0x000C_0630);
        assert_eq!(rest.delivery(Mode::XApic), Some(Delivery::Startup));
        assert!(rest.legal(Mode::XApic));
    }

    #[test]
    fn the_vectorless_modes_must_be_sent_with_the_field_clear() {
        // A system-management interrupt and an INIT, each carrying a vector the
        // architecture requires to be zero.
        for bits in [0x0000_0230_u64, 0x0000_0530] {
            assert!(!Command::from_bits(bits).legal(Mode::XApic));
        }
        // The same two with the field clear.
        for bits in [0x0000_0200_u64, 0x0000_0500] {
            assert!(Command::from_bits(bits).legal(Mode::XApic));
        }
        // A non-maskable interrupt reads past the field rather than requiring
        // it clear, so one carrying a number is still a command.
        assert!(Command::from_bits(0x0000_0430).legal(Mode::XApic));
    }

    #[test]
    fn the_synchronisation_message_is_only_ever_a_broadcast() {
        // INIT de-assert addressed to everyone, which is the one form it has.
        let all = Command::from_bits(0x0008_8500);
        assert!(all.is_init_deassert());
        assert!(all.legal(Mode::XApic));

        // The same message addressed any other way is not that message.
        for bits in [0x0000_8500_u64, 0x0004_8500, 0x000C_8500] {
            let command = Command::from_bits(bits);
            assert!(command.is_init_deassert());
            assert!(!command.legal(Mode::XApic));
        }
    }
}
