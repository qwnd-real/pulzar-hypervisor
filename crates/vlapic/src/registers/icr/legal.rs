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
        decode::{Delivery, Shorthand},
    },
};

impl Command {
    /// Whether this is a command the architecture defines at all.
    ///
    /// Applied to the whole command before any target is worked out, which is
    /// the difference between rejecting a command and half-performing one: a
    /// controller that resolved targets first would have already reset, started
    /// or interrupted some of them by the time it noticed.
    ///
    /// What is judged is what a *controller* refuses, which is much less than
    /// what the architecture tells software to write. Its tables of valid
    /// combinations forbid a good deal that hardware performs anyway, and every
    /// one of those rows is a rule for the sender rather than a licence to
    /// discard the message: the vector field of a delivery mode that carries no
    /// vector is not read at all, so an INIT that leaves something in it is
    /// delivered with the field ignored, and a processor may interrupt or
    /// initialise itself. Refusing those would drop messages hardware delivers
    /// and leave the sender waiting for something it will never be told about.
    ///
    /// One combination has nowhere to go, and it is the only one here: a
    /// start-up cannot be addressed to the processor that would have to send
    /// it, because a processor waiting for a start-up is not executing the
    /// instruction that sends one.
    pub(crate) const fn legal(self, mode: Mode) -> bool {
        let Some(delivery) = self.delivery(mode) else {
            return false;
        };
        match delivery {
            // The shorthands that include the sender are how software addresses
            // a start-up to itself.
            Delivery::Startup => !matches!(self.shorthand(), Shorthand::Myself | Shorthand::All),
            _ => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use descriptors::Vector;

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
    fn a_vectorless_mode_carrying_a_vector_is_still_a_command() {
        // A system-management interrupt and an INIT, each carrying a number in
        // a field the architecture requires software to clear and hardware
        // never reads. Dropping these is what would leave a processor that
        // reuses one command word for both a start-up and an INIT — writing the
        // start-up page and then the INIT through it — never reset at all, and
        // its sender waiting forever for a processor it never initialised.
        for bits in [0x0000_0230_u64, 0x0000_0530] {
            let command = Command::from_bits(bits);
            assert!(command.legal(Mode::XApic));
            assert_eq!(
                command.vector(),
                Vector::new(0),
                "the field is not read, so nothing downstream can act on it"
            );
        }
        // The same two with the field clear, which is how software is told to
        // write them.
        for bits in [0x0000_0200_u64, 0x0000_0500] {
            assert!(Command::from_bits(bits).legal(Mode::XApic));
        }
    }

    #[test]
    fn the_synchronisation_message_is_a_command_however_it_is_addressed() {
        // The architecture sends this one to every processor whatever the
        // destination or the shorthand says, and merely asks software to
        // address it to all including self. An operating system does not: its
        // start-up sequence sends the de-assert to one processor, physically
        // addressed, immediately after the INIT it follows. Refusing that would
        // make every bring-up an illegal command.
        for bits in [0x0000_8500_u64, 0x0004_8500, 0x0008_8500, 0x000C_8500] {
            let command = Command::from_bits(bits);
            assert!(command.is_init_deassert());
            assert!(command.legal(Mode::XApic));
        }
    }

    #[test]
    fn a_reserved_delivery_mode_is_not_a_command_in_either_face() {
        // The two encodings no mode is named by, and lowest priority, which is
        // a mode in the older face alone.
        for bits in [0x0000_0330_u64, 0x0000_0730] {
            let command = Command::from_bits(bits);
            assert!(!command.legal(Mode::XApic));
            assert!(!command.legal(Mode::X2Apic));
        }
        let redirectable = Command::from_bits(0x0000_0130);
        assert!(redirectable.legal(Mode::XApic));
        assert!(!redirectable.legal(Mode::X2Apic));
    }
}
