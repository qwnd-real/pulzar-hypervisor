//! What a command asks for, and whether the architecture defines it at all.
//!
//! Every field whose meaning depends on the face it arrived through takes the
//! mode as an argument rather than being decoded twice, and every field the
//! hardware reads past is answered here rather than at each caller: a caller
//! given the bits software happened to leave in a field nothing reads would
//! have to know which fields those are.

use descriptors::Vector;

use crate::registers::{
    base::Mode,
    icr::{
        Command,
        command::{
            DELIVERY, DESTINATION_MODE, DESTINATION_XAPIC, LEVEL, SHORTHAND, TRIGGER, VECTOR,
        },
    },
};

impl Command {
    /// Whether the command asks for a redirectable interrupt, whether or not
    /// this face can send one.
    ///
    /// Asked separately from [`Command::delivery`] because the interesting case
    /// is exactly the one that decodes to nothing: a controller in x2APIC has
    /// no lowest-priority delivery, and the architecture has it record an
    /// error of its own rather than treat the request as an unrecognised
    /// encoding.
    pub(crate) const fn wants_lowest_priority(self) -> bool {
        DELIVERY.get(self.low()) == LOWEST_PRIORITY
    }

    /// The vector the command carries.
    ///
    /// Zero for the three delivery modes that carry none, whatever software
    /// left in the field: a non-maskable interrupt arrives on the vector the
    /// architecture fixes for it, and a system-management interrupt or an INIT
    /// must be sent with the field clear. Reporting what was written there
    /// would hand a caller a number nothing is ever delivered on.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the field is eight bits wide, which is the whole of a vector's range"
    )]
    pub(crate) const fn vector(self) -> Vector {
        match DELIVERY.get(self.low()) {
            SYSTEM_MANAGEMENT | NON_MASKABLE | INIT => Vector::new(0),
            _ => Vector::new(VECTOR.get(self.low()) as u8),
        }
    }

    /// What the command asks to be delivered, or nothing if it asks for
    /// something this mode cannot send.
    ///
    /// Two encodings are reserved outright and lowest priority is reserved in
    /// x2APIC as well. Everything else about whether the command makes sense —
    /// which shorthands a mode may be addressed with, which fields it must
    /// leave clear — belongs to [`Command::legal`], so that this stays a
    /// decoding of the field rather than half of a rule stated in two
    /// places.
    pub(crate) const fn delivery(self, mode: Mode) -> Option<Delivery> {
        Delivery::decode(DELIVERY.get(self.low()), mode)
    }

    /// Which processors the command names without naming any.
    pub(crate) const fn shorthand(self) -> Shorthand {
        match SHORTHAND.get(self.low()) {
            MYSELF => Shorthand::Myself,
            ALL_INCLUDING_SELF => Shorthand::All,
            ALL_EXCLUDING_SELF => Shorthand::Others,
            // The field is two bits, so what is left is the encoding that names
            // no shorthand and leaves the destination to say who.
            _ => Shorthand::None,
        }
    }

    /// How the destination field names its targets.
    ///
    /// Physical whenever a shorthand is in use, because the shorthand has named
    /// the targets itself and the architecture reads neither this field nor the
    /// destination beside it. Reporting what software wrote would send a caller
    /// matching logical identifiers that nothing was addressed by.
    pub(crate) const fn destination_mode(self) -> DestinationMode {
        if !matches!(self.shorthand(), Shorthand::None) {
            return DestinationMode::Physical;
        }
        if DESTINATION_MODE.test(self.low()) {
            DestinationMode::Logical
        } else {
            DestinationMode::Physical
        }
    }

    /// Who the command is addressed to, widened to the wider of the two faces.
    ///
    /// Eight bits through the memory-mapped page, where the field is the top
    /// byte of the high half, and thirty-two in x2APIC, where it is the whole
    /// of it. Means nothing while a shorthand is in use.
    pub(crate) const fn destination(self, mode: Mode) -> u32 {
        if matches!(mode, Mode::X2Apic) {
            self.high()
        } else {
            DESTINATION_XAPIC.get(self.high())
        }
    }

    /// Whether the level bit is set, which on the bus this outlived meant
    /// assert rather than de-assert.
    ///
    /// Private, because nothing reads it any more except the one command that
    /// is recognised by it being clear: [`Command::is_init_deassert`] is what a
    /// caller wants, and a caller given the raw bit would be given a field
    /// hardware reads past.
    const fn level(self) -> bool {
        LEVEL.test(self.low())
    }

    /// The trigger mode the command is actually issued with.
    ///
    /// Edge for everything the architecture still sends. A guest that asks for
    /// a level-triggered interprocessor interrupt gets an edge-triggered one,
    /// silently, exactly as it would on hardware — the one exception being the
    /// INIT de-assert, which is level triggered by definition.
    pub(crate) const fn trigger(self) -> Trigger {
        if self.is_init_deassert() {
            Trigger::Level
        } else {
            Trigger::Edge
        }
    }

    /// Whether this is the INIT de-assert: delivery mode INIT with the level
    /// bit clear and the trigger-mode bit set.
    ///
    /// It resets the arbitration identifiers of everything it reaches and does
    /// nothing else — no processor is initialised by it and no vector is
    /// delivered. Expressible through both faces, because both keep the two
    /// bits that name it, and this is what tells it apart from the INIT that
    /// starts a processor: an operating system's start-up sequence writes the
    /// same delivery mode twice, once with the level bit set and once without,
    /// and a controller that could not tell them apart would reset every target
    /// a second time.
    pub(crate) const fn is_init_deassert(self) -> bool {
        DELIVERY.get(self.low()) == INIT && !self.level() && TRIGGER.test(self.low())
    }
}

/// What the controller is being asked to deliver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Delivery {
    /// The vector, to every processor the command names.
    Fixed,
    /// The vector, to whichever of them is running at the lowest priority.
    /// Reserved in x2APIC, and arbitrated by the chipset rather than by the
    /// architecture where it does exist.
    LowestPriority,
    /// A system-management interrupt, which carries no vector and takes its
    /// target into system-management mode.
    SystemManagement,
    /// A non-maskable interrupt, which arrives whatever the target has masked
    /// and on the vector the architecture fixes for it rather than on any the
    /// command names.
    NonMaskable,
    /// INIT, which resets its target and leaves it waiting for a start-up.
    Init,
    /// Start-up, which releases a target from that wait at the address the
    /// vector field gives the page number of.
    Startup,
}

impl Delivery {
    /// What a three-bit encoding names, given the face it arrived through.
    ///
    /// Reserved encodings answer with nothing rather than with a mode that
    /// happens to be nearby: a command a processor would not send is one this
    /// controller has no business delivering either.
    const fn decode(encoding: u32, mode: Mode) -> Option<Self> {
        match encoding {
            FIXED => Some(Self::Fixed),
            LOWEST_PRIORITY if !matches!(mode, Mode::X2Apic) => Some(Self::LowestPriority),
            SYSTEM_MANAGEMENT => Some(Self::SystemManagement),
            NON_MASKABLE => Some(Self::NonMaskable),
            INIT => Some(Self::Init),
            STARTUP => Some(Self::Startup),
            _ => None,
        }
    }
}

/// Which processors a command names, when it names them without addressing any.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Shorthand {
    /// No shorthand: the destination field says who.
    None,
    /// The processor sending the command, and only it.
    Myself,
    /// Every processor on the machine, the sender included.
    All,
    /// Every processor but the sender.
    Others,
}

/// How the destination field picks the processors it names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DestinationMode {
    /// The destination is one processor's identifier, or the broadcast value.
    Physical,
    /// The destination is matched against each processor's logical identifier,
    /// so one command may reach several.
    Logical,
}

/// How a delivered interrupt is signalled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Trigger {
    /// A single event, complete when it is accepted.
    Edge,
    /// A held assertion, which only the INIT de-assert still expresses.
    Level,
}

/// Deliver the vector to everything addressed.
const FIXED: u32 = 0b000;

/// Deliver it to whichever of them is least busy.
const LOWEST_PRIORITY: u32 = 0b001;

/// Deliver a system-management interrupt.
const SYSTEM_MANAGEMENT: u32 = 0b010;

/// Deliver a non-maskable interrupt.
const NON_MASKABLE: u32 = 0b100;

/// Reset the target and leave it waiting.
const INIT: u32 = 0b101;

/// Release a waiting target at the address the vector gives the page of.
const STARTUP: u32 = 0b110;

/// Address the sender alone.
const MYSELF: u32 = 0b01;

/// Address every processor, sender included.
const ALL_INCLUDING_SELF: u32 = 0b10;

/// Address every processor but the sender.
const ALL_EXCLUDING_SELF: u32 = 0b11;

#[cfg(test)]
mod tests {
    //! Commands are written out as the raw halves a guest would write, so that
    //! a test fails when the layout moves rather than moving with it.

    use super::{Command, Delivery, DestinationMode, Mode, Shorthand, Trigger, Vector};

    /// An ordinary vector, well clear of the exceptions.
    const VECTOR: Vector = Vector::new(0x30);

    /// The processor the tests address by identifier.
    const TARGET: u32 = 0x05;

    #[test]
    fn a_fixed_interrupt_decodes_the_same_through_both_faces() {
        let mapped = Command::from_halves(0x0000_0030, 0x0500_0000);
        assert_eq!(mapped.low(), 0x0000_0030);
        assert_eq!(mapped.high(), 0x0500_0000);
        assert_eq!(mapped.vector(), VECTOR);
        assert_eq!(mapped.delivery(Mode::XApic), Some(Delivery::Fixed));
        assert_eq!(mapped.shorthand(), Shorthand::None);
        assert_eq!(mapped.destination_mode(), DestinationMode::Physical);
        assert_eq!(mapped.destination(Mode::XApic), TARGET);
        assert_eq!(mapped.trigger(), Trigger::Edge);

        let wide = Command::from_bits(0x0000_0005_0000_0030);
        assert_eq!(wide.vector(), VECTOR);
        assert_eq!(wide.delivery(Mode::X2Apic), Some(Delivery::Fixed));
        assert_eq!(wide.shorthand(), Shorthand::None);
        assert_eq!(wide.destination_mode(), DestinationMode::Physical);
        assert_eq!(wide.destination(Mode::X2Apic), TARGET);
        assert_eq!(wide.trigger(), Trigger::Edge);
    }

    #[test]
    fn a_level_triggered_command_is_issued_as_an_edge_triggered_one() {
        // A fixed interrupt asserted level triggered: bits a guest may write
        // and the architecture then reads past.
        let fixed = Command::from_bits(0x0000_C030);
        assert!(fixed.level());
        assert!(!fixed.is_init_deassert());
        assert_eq!(fixed.trigger(), Trigger::Edge);

        // An INIT asserted level triggered is not the de-assert: the level bit
        // is what tells the two apart.
        let init = Command::from_bits(0x0000_C500);
        assert_eq!(init.delivery(Mode::XApic), Some(Delivery::Init));
        assert!(!init.is_init_deassert());
        assert_eq!(init.trigger(), Trigger::Edge);
    }

    #[test]
    fn the_init_de_assert_is_the_only_level_triggered_command() {
        let deassert = Command::from_bits(0x0000_8500);
        assert!(deassert.is_init_deassert());
        assert!(!deassert.level());
        assert_eq!(deassert.trigger(), Trigger::Level);
        assert_eq!(deassert.delivery(Mode::XApic), Some(Delivery::Init));

        // The same trigger-mode bit under another delivery mode, an INIT
        // without it, and a start-up carrying it.
        for bits in [0x0000_8030, 0x0000_0500, 0x0000_8630] {
            let command = Command::from_bits(bits);
            assert!(!command.is_init_deassert());
            assert_eq!(command.trigger(), Trigger::Edge);
        }
    }

    #[test]
    fn the_reserved_delivery_modes_name_nothing() {
        for bits in [0x0000_0330, 0x0000_0730] {
            let command = Command::from_bits(bits);
            assert_eq!(command.delivery(Mode::XApic), None);
            assert_eq!(command.delivery(Mode::X2Apic), None);
        }
    }

    #[test]
    fn lowest_priority_is_reserved_in_the_wide_face_alone() {
        let command = Command::from_bits(0x0000_0130);
        assert_eq!(
            command.delivery(Mode::XApic),
            Some(Delivery::LowestPriority)
        );
        assert_eq!(command.delivery(Mode::X2Apic), None);
    }

    #[test]
    fn the_destination_is_a_byte_in_one_face_and_a_word_in_the_other() {
        let broadcast = Command::from_bits(0xFFFF_FFFF_0000_0030);
        assert_eq!(broadcast.destination(Mode::XApic), 0xFF);
        assert_eq!(broadcast.destination(Mode::X2Apic), Command::BROADCAST);

        // Through the page, everything below the top byte of the high half is
        // reserved and no part of the destination.
        let mapped = Command::from_halves(0x0000_0030, 0x0500_00FF);
        assert_eq!(mapped.destination(Mode::XApic), TARGET);
    }

    #[test]
    fn the_delivery_modes_that_carry_no_vector_report_none() {
        // A system-management interrupt and an INIT, both of which must be sent
        // with the field clear, and a non-maskable interrupt, whose vector the
        // architecture reads past.
        for bits in [0x0000_0230, 0x0000_0530, 0x0000_0430] {
            assert_eq!(Command::from_bits(bits).vector(), Vector::new(0));
        }

        // A start-up keeps its own: the field is the page it starts at.
        assert_eq!(Command::from_bits(0x0000_0630).vector(), VECTOR);
    }

    #[test]
    fn a_shorthand_answers_for_the_fields_it_makes_meaningless() {
        // Fixed, logical destination mode, addressed to every other processor.
        let others = Command::from_bits(0x000C_0830);
        assert_eq!(others.shorthand(), Shorthand::Others);
        assert_eq!(others.destination_mode(), DestinationMode::Physical);
        assert_eq!(others.delivery(Mode::XApic), Some(Delivery::Fixed));
    }

    #[test]
    fn the_sequence_a_processor_is_started_with_survives_the_wide_face() {
        // The four quadwords an operating system and this machine's firmware
        // actually write to bring up an application processor, addressed to
        // identifier five: INIT asserted, INIT de-asserted, and a start-up at
        // page 0x08 — the last of them twice, once as firmware forms it with
        // the level bit set and once as an operating system forms it without.
        //
        // Written out as raw bits because that is what a guest writes. Every
        // one of them has to be accepted: a bit outside the writable mask is a
        // general protection fault at the `WRMSR`, and a fault here is a
        // machine whose second processor never starts.
        const DESTINATION: u64 = 5 << 32;
        let init = Command::from_bits(DESTINATION | 0xC500);
        let deassert = Command::from_bits(DESTINATION | 0x8500);
        let startup = Command::from_bits(DESTINATION | 0x0608);
        let firmware_startup = Command::from_bits(DESTINATION | 0x4608);

        for command in [init, deassert, startup, firmware_startup] {
            assert_eq!(
                command.bits() & !Command::WRITABLE_X2APIC,
                0,
                "{:#018x} sets a bit the wide face refuses",
                command.bits()
            );
            assert!(command.legal(Mode::X2Apic), "{:#018x}", command.bits());
            assert_eq!(command.destination(Mode::X2Apic), 5);
            assert_eq!(command.shorthand(), Shorthand::None);
            assert_eq!(command.destination_mode(), DestinationMode::Physical);
        }

        // The INIT that resets the target, and the message that follows it and
        // must not: the level bit is the whole of the difference between them.
        assert_eq!(init.delivery(Mode::X2Apic), Some(Delivery::Init));
        assert!(!init.is_init_deassert());
        assert_eq!(init.trigger(), Trigger::Edge);
        assert_eq!(deassert.delivery(Mode::X2Apic), Some(Delivery::Init));
        assert!(deassert.is_init_deassert());
        assert_eq!(deassert.trigger(), Trigger::Level);

        // Both spellings of the start-up name the same page, and the level bit
        // firmware sets changes nothing about it.
        for command in [startup, firmware_startup] {
            assert_eq!(command.delivery(Mode::X2Apic), Some(Delivery::Startup));
            assert_eq!(command.vector(), Vector::new(0x08));
            assert_eq!(command.trigger(), Trigger::Edge);
        }
    }
}
