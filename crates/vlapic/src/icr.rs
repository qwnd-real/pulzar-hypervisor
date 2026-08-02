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

use descriptors::Vector;

use crate::base::Mode;

/// One interrupt command, in the sixty-four bits both faces describe it with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Command(u64);

impl Command {
    /// The destination every processor answers to in x2APIC, in the logical
    /// destination mode as much as the physical one.
    ///
    /// The memory-mapped face spells the same thing with all eight bits of its
    /// narrower field set, which is what this value truncates to.
    pub(crate) const BROADCAST: u32 = u32::MAX;

    /// Which bits of the low half software may set.
    ///
    /// Every field a guest owns and nothing else: the delivery-status bit at 12
    /// is the controller's own report of whether a command is still going out,
    /// and bits 13, 17:16 and 31:20 are reserved. A write through the
    /// memory-mapped face is masked with this, which is what keeps a guest's
    /// stray bits from being read back; the wide face has no delivery-status
    /// bit at all and faults on a reserved bit rather than dropping it.
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
    /// Narrower than the memory-mapped face in exactly the places where the
    /// older one kept bus-era fields. The delivery-status bit is gone, because
    /// an x2APIC write does not return until the command has been accepted and
    /// there is nothing to report; and level and trigger mode are gone with the
    /// bus they described, which is why the INIT de-assert cannot be expressed
    /// here at all.
    ///
    /// Reserved here means `RsvdZ`: writing a non-zero value into one is a
    /// general protection fault rather than something quietly dropped.
    pub(crate) const WRITABLE_X2APIC: u64 =
        (VECTOR.mask() | DELIVERY.mask() | DESTINATION_MODE.mask() | SHORTHAND.mask()) as u64
            | (u32::MAX as u64) << HALF;

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
    /// Nothing reads it any more except the one command that is recognised by
    /// it being clear, which is why [`Command::is_init_deassert`] rather than
    /// this is what a caller normally wants.
    pub(crate) const fn level(self) -> bool {
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
    /// delivered. Only the memory-mapped face can express it, since x2APIC
    /// reserves both bits and faults a guest that sets them.
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

/// A run of adjacent bits in the register, named by where it starts and how
/// wide it is.
///
/// The layout is stated once, here and in the constants below, so that no
/// accessor spells out a shift or a mask of its own and the writable mask
/// cannot drift away from the fields it is made of.
#[derive(Clone, Copy)]
struct Field {
    /// How far above bit zero the field starts.
    shift: u32,
    /// How many bits it spans.
    width: u32,
}

impl Field {
    /// The field of `width` bits starting at `shift`.
    const fn new(shift: u32, width: u32) -> Self {
        Self { shift, width }
    }

    /// The field's value, brought down to bit zero.
    const fn get(self, bits: u32) -> u32 {
        (bits & self.mask()) >> self.shift
    }

    /// Whether any of the field's bits are set, which for a one-bit field is
    /// the field itself.
    const fn test(self, bits: u32) -> bool {
        bits & self.mask() != 0
    }

    /// The field's bits, where they sit.
    const fn mask(self) -> u32 {
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
const VECTOR: Field = Field::new(0, 8);

/// Which of the eight delivery modes the command names.
const DELIVERY: Field = Field::new(8, 3);

/// Whether the destination is an identifier or a logical mask.
const DESTINATION_MODE: Field = Field::new(11, 1);

/// Assert or de-assert, read only for the INIT de-assert.
const LEVEL: Field = Field::new(14, 1);

/// Edge or level, read only for the INIT de-assert.
const TRIGGER: Field = Field::new(15, 1);

/// Which shorthand, if any, names the targets in place of the destination.
const SHORTHAND: Field = Field::new(18, 2);

/// The destination within the high half of the memory-mapped register, where it
/// is a byte at the top rather than the whole word.
const DESTINATION_XAPIC: Field = Field::new(24, 8);

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

    #[test]
    fn the_wide_face_reserves_the_bus_era_fields() {
        // Level and trigger mode, which x2APIC does not have.
        assert_eq!(Command::WRITABLE_X2APIC & (1 << 14 | 1 << 15), 0);
        // The delivery-status bit, which it does not have either.
        assert_eq!(Command::WRITABLE_X2APIC & (1 << 12), 0);
        // What it does have: vector, delivery mode, destination mode,
        // shorthand, and the whole of the upper half for the destination.
        assert_eq!(Command::WRITABLE_X2APIC, 0xFFFF_FFFF_000C_0FFF);
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
