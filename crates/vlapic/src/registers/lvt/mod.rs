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
//! What software may *set* and what it may *put a one in* are two masks and not
//! one, because two of the bits belong to neither category. Delivery status and
//! remote IRR are the controller's own reports: software cannot set them, and
//! writing them is ignored rather than refused, so [`Entry::writable`] drops
//! them and [`Entry::reserved`] does not fault on them.
//!
//! [`Entry::ALL`] is in the order the architecture counts these in, which is
//! what the version register reports the highest index of. It is deliberately
//! not the order the registers sit at — the corrected-machine-check entry is
//! last in the count and first in memory — so nothing may derive one order from
//! the other.

mod shape;
mod table;

use descriptors::Vector;

pub(crate) use crate::registers::lvt::shape::{Delivery, Lvt, TimerMode};
use crate::{face::table::Register, hardware::model::Model};

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
    /// How many entries the local vector table has at most.
    ///
    /// How many a particular controller has is [`Model::version`]'s to report,
    /// and is usually fewer: three of these are optional. This is the size of
    /// the array they are stored in and the largest a model may claim.
    pub(crate) const COUNT: usize = 7;

    /// How few a controller may have.
    ///
    /// The timer, both interrupt pins and the error entry: the smallest table
    /// any controller has ever reported, and the smallest the version register
    /// describes. Nothing has fewer, and a model built with fewer would tell a
    /// guest its non-maskable interrupt pin and its error entry are not there —
    /// which every operating system's controller bring-up writes.
    pub(crate) const FEWEST: usize = 4;

    /// Every entry, in the order the architecture counts them.
    ///
    /// A controller has exactly the first however-many of these, so the
    /// position of an entry here is architectural rather than an
    /// implementation detail — it is what decides whether a given
    /// controller has the entry at all.
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

    /// Whether this entry has a delivery-mode field software may write.
    ///
    /// Only the timer does not: its delivery is fixed by the architecture and
    /// the bits the field would occupy are reserved. Every other entry has one,
    /// the error entry included — which is where the controller this crate
    /// presents differs from Intel's, whose error entry reserves those bits as
    /// the timer's does. [`crate::hardware::model`] is where the choice of
    /// manual is argued.
    pub(crate) const fn has_delivery(self) -> bool {
        !matches!(self, Self::Timer)
    }

    /// Whether this entry accepts a delivery mode.
    ///
    /// Four modes exist in an entry: a fixed interrupt on the vector it names,
    /// a system-management interrupt, a non-maskable one, and an external
    /// interrupt signalled over a wire. INIT is *not* among them — the
    /// architecture defines that encoding for the interrupt command register
    /// and not for these entries — and an external interrupt means a wire,
    /// so it belongs to the two entries that describe one.
    ///
    /// The exception to all of that is a system-management interrupt, which no
    /// entry accepts, and that is this hypervisor's own departure rather than
    /// the architecture's. These entries are programmed onto the machine's
    /// own controller, so a guest that chose that mode would take the
    /// *host* into system-management mode over host state, running
    /// firmware's handler against a context it was not written for — and
    /// the guest would see nothing of it either way. The interrupt command
    /// register refuses to send one for exactly the same reason, and the
    /// decision belongs in one place rather than in both.
    ///
    /// An external interrupt reaches a pin's entry from here and is refused
    /// where the entry is turned into a physical one, because what is wrong
    /// with it is not the shape of the entry;
    /// [`crate::hardware::sources`] is where that is stated.
    pub(crate) const fn allows(self, delivery: Delivery) -> bool {
        match delivery {
            Delivery::Fixed => true,
            Delivery::NonMaskable => self.has_delivery(),
            Delivery::SystemManagement | Delivery::Init => false,
            Delivery::External => matches!(self, Self::Lint0 | Self::Lint1),
        }
    }

    /// Whether this entry, holding `lvt`, would actually deliver a vector.
    ///
    /// Which is the only condition under which its vector field means anything.
    /// Every other delivery mode is an event the processor takes by its own
    /// architectural entry point and reads no vector for, so a number left in
    /// the field is not a vector at all and reporting it as an illegal one
    /// would be reporting an error about a field nothing reads.
    ///
    /// Asked of a value rather than of the stored register, so that a caller
    /// deciding this about a write it has just performed decides it about what
    /// it wrote.
    pub(crate) const fn delivers_a_vector(self, lvt: Lvt) -> bool {
        !self.has_delivery() || matches!(Delivery::from_bits(lvt.delivery()), Some(Delivery::Fixed))
    }

    /// Which bits of this entry software may set, in this model. Everything
    /// else is reserved in it and is dropped from a write rather than
    /// stored.
    ///
    /// The delivery-status and remote-IRR bits are absent from every answer:
    /// both are the controller's to report, and letting a guest write either
    /// would let it claim a delivery that never happened or retire one that is
    /// still outstanding. That makes this the mask a write is *stored* through
    /// and not the one it is judged against — [`Entry::reserved`] is that one,
    /// and the two differ by exactly those bits.
    ///
    /// Two entries have a shape the model decides rather than the entry. The
    /// error entry's message type is writable on AMD and reserved on Intel. And
    /// the timer's mode field is only two bits wide on a processor that
    /// implements the timestamp-counter deadline: without it there are two
    /// modes rather than three, the bit that would select the third is
    /// reserved, and a guest allowed to set it would select a mode its own
    /// `CPUID` denies and real hardware would refuse to be programmed for.
    pub(crate) const fn writable(self, model: Model) -> u32 {
        let delivery = if self.has_delivery() {
            WRITABLE_DELIVERY
        } else {
            0
        };
        WRITABLE_COMMON
            | delivery
            | match self {
                Self::Timer if model.deadline() => WRITABLE_TIMER_MODE,
                Self::Timer => WRITABLE_COUNTING_MODE,
                Self::Lint0 | Self::Lint1 => WRITABLE_PIN,
                _ => 0,
            }
    }

    /// Which bits of this entry no software may put a one in, in this model.
    ///
    /// Not the complement of [`Entry::writable`], and the difference is the
    /// point: *reserved* and *read-only* are two things, and only the first is
    /// a general protection fault through the model-specific registers. The
    /// two bits the controller reports through — delivery status in every
    /// entry, and remote IRR in the two that describe a wire — are
    /// read-only. Hardware ignores a write to them, and it has to: reading
    /// an entry, changing one field and writing the whole of it back is how
    /// software touches these registers, and what it read back holds
    /// whatever the controller had put in those bits. A controller that
    /// faulted on them would fault a guest for writing back a value it had
    /// just been given.
    ///
    /// So a write of one of those bits is dropped, by [`Entry::writable`], and
    /// not refused. Everything outside both masks is reserved and is refused.
    pub(crate) const fn reserved(self, model: Model) -> u32 {
        let pin = if self.is_pin() { REMOTE_IRR } else { 0 };
        !(self.writable(model) | SEND_PENDING | pin)
    }

    /// Whether a register names a local vector table entry this model does not
    /// have.
    ///
    /// Asked of a register rather than of an entry because it is what decides
    /// whether the register exists at all, and the answer has to be `false` for
    /// every register that is not one of these to begin with.
    pub(crate) fn absent(register: Register, model: Model) -> bool {
        Self::of(register).is_some_and(|entry| !model.has(entry))
    }

    /// Whether this entry describes a wire into the processor.
    ///
    /// The two pins do, and nothing else does, which is what decides whether
    /// the polarity and trigger-mode fields mean anything — and whether a
    /// guest's trigger mode may be carried onto real hardware at all.
    pub(crate) const fn is_pin(self) -> bool {
        matches!(self, Self::Lint0 | Self::Lint1)
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

/// As much of it as a processor without the timestamp-counter deadline has: the
/// bit that chooses between the two counting modes, and not the one that would
/// select the third.
const WRITABLE_COUNTING_MODE: u32 = Lvt::new().with_timer_mode(COUNTING_MODE_FIELD).into_bits();

/// The bit the controller reports a delivery from this source still being in
/// flight through. Present in every entry, and read-only in all of them.
const SEND_PENDING: u32 = Lvt::new().with_send_pending(true).into_bits();

/// The bit the controller reports an accepted, unacknowledged level-triggered
/// interrupt through, which is why it exists in the two pin entries alone and
/// is reserved in the rest.
const REMOTE_IRR: u32 = Lvt::new().with_remote_irr(true).into_bits();

/// A three-bit delivery-mode field with every bit set, which is what marks
/// where the field sits rather than a mode any entry accepts.
const DELIVERY_FIELD: u8 = 0b111;

/// The same for the two-bit timer-mode field.
const TIMER_MODE_FIELD: u8 = 0b11;

/// The part of that field that selects between one-shot and periodic, which are
/// the two modes every controller's timer has.
const COUNTING_MODE_FIELD: u8 = 0b01;

/// The bit that masks a local-vector-table entry.
pub(super) const MASKED: u32 = 1 << 16;

/// A controller has exactly the first however-many of [`Entry::ALL`], and
/// [`Model::has`] answers that from an entry's own position — so the list has
/// to be in the order the discriminants are, or a guest would be handed an
/// entry its controller does not have and refused one it does.
const _: () = {
    let mut index = 0;
    while index < Entry::COUNT {
        assert!(
            Entry::ALL[index].index() == index,
            "every entry has to sit at the position its own index names"
        );
        index += 1;
    }
};

/// The table the guest programs has to be as long as the one the real
/// controller has entries in, because a controller is seeded from a capture of
/// those entries and every one of them is programmed back onto real hardware. A
/// shorter table here would drop an entry firmware left armed; a longer one
/// would offer the guest a register with no source behind it.
const _: () = assert!(
    Entry::COUNT == apic::LVT_ENTRIES,
    "the guest's table and the real controller's have to have the same entries"
);

#[cfg(test)]
mod tests {
    use descriptors::Vector;

    use super::{
        Delivery, Entry, Lvt, TimerMode, WRITABLE_COUNTING_MODE, WRITABLE_DELIVERY, WRITABLE_PIN,
        WRITABLE_TIMER_MODE,
    };
    use crate::{face::table::Register, hardware::model};

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
    fn only_the_pins_describe_a_wire() {
        for entry in Entry::ALL {
            assert_eq!(
                entry.is_pin(),
                matches!(entry, Entry::Lint0 | Entry::Lint1),
                "{entry:?}"
            );
            let held = entry.writable(model::tests::AMD) & WRITABLE_PIN;
            assert_eq!(held == WRITABLE_PIN, entry.is_pin(), "{entry:?}");
        }
    }

    #[test]
    fn only_the_timer_may_set_the_timer_mode() {
        for entry in Entry::ALL {
            let held = entry.writable(model::tests::AMD) & WRITABLE_TIMER_MODE;
            assert_eq!(held == WRITABLE_TIMER_MODE, matches!(entry, Entry::Timer));
        }
    }

    #[test]
    fn the_deadline_mode_cannot_be_selected_without_the_processor_feature() {
        let sparse = model::tests::SPARSE;
        assert!(!sparse.deadline());
        // The counting modes stay reachable and the third does not: what is left
        // of the field is the one bit that tells one-shot from periodic.
        let held = Entry::Timer.writable(sparse) & WRITABLE_TIMER_MODE;
        assert_eq!(held, WRITABLE_COUNTING_MODE);
        assert_eq!(
            Lvt::from_bits(held).timer_mode(),
            TimerMode::Periodic as u8,
            "the bit that survives is the one periodic mode needs"
        );
        assert_eq!(
            Entry::Timer.writable(model::tests::AMD) & WRITABLE_TIMER_MODE,
            WRITABLE_TIMER_MODE
        );
    }

    #[test]
    fn the_timer_is_the_one_entry_with_no_delivery_mode() {
        assert!(!Entry::Timer.has_delivery());
        assert_eq!(
            Entry::Timer.writable(model::tests::AMD) & WRITABLE_DELIVERY,
            0
        );
        assert!(Entry::Timer.allows(Delivery::Fixed));
        assert!(!Entry::Timer.allows(Delivery::NonMaskable));
        for entry in Entry::ALL
            .into_iter()
            .filter(|entry| *entry != Entry::Timer)
        {
            assert!(entry.has_delivery(), "{entry:?}");
            assert_eq!(
                entry.writable(model::tests::AMD) & WRITABLE_DELIVERY,
                WRITABLE_DELIVERY,
                "{entry:?}"
            );
        }
    }

    #[test]
    fn the_error_entry_takes_a_message_type() {
        // The one entry the two vendors' manuals shape differently: AMD gives it
        // a message type in bits 10:8 and Intel reserves the whole of 11:8. This
        // crate presents AMD's controller unconditionally, so the field is here —
        // and a vector left in the entry is an illegal one only while the mode is
        // fixed, which is what makes the difference observable at all.
        assert!(Entry::Error.has_delivery());
        assert_eq!(
            Entry::Error.writable(model::tests::AMD) & WRITABLE_DELIVERY,
            WRITABLE_DELIVERY
        );
        assert!(Entry::Error.allows(Delivery::NonMaskable));
        // It still has no wire, so neither of the modes that describe one.
        assert!(!Entry::Error.allows(Delivery::External));
        assert!(!Entry::Error.allows(Delivery::Init));
    }

    #[test]
    fn corrected_machine_check_refuses_everything_but_a_vector_and_a_pin_signal() {
        let entry = Entry::CorrectedMachineCheck;
        assert!(entry.allows(Delivery::Fixed));
        assert!(entry.allows(Delivery::NonMaskable));
        // Not the architecture's rule for this entry: a system-management
        // interrupt would take the host into system-management mode, which is
        // refused in every entry rather than in this one.
        assert!(!entry.allows(Delivery::SystemManagement));
        assert!(!entry.allows(Delivery::Init));
        assert!(!entry.allows(Delivery::External));
    }

    #[test]
    fn no_entry_delivers_a_system_management_interrupt_or_an_init() {
        // Two modes refused everywhere. A system-management interrupt is refused
        // by this hypervisor rather than by the architecture: taking it would put
        // the host into system-management mode over host state, and the interrupt
        // command register refuses to send one for the same reason. INIT is
        // refused by the architecture, which defines that encoding for the
        // interrupt command register and not for these entries.
        for entry in Entry::ALL {
            assert!(!entry.allows(Delivery::SystemManagement), "{entry:?}");
            assert!(!entry.allows(Delivery::Init), "{entry:?}");
        }
    }

    #[test]
    fn only_a_pin_takes_an_external_interrupt() {
        for entry in Entry::ALL {
            assert_eq!(
                entry.allows(Delivery::External),
                entry.is_pin(),
                "{entry:?}"
            );
        }
    }

    /// Bit 12: the controller's report that a delivery from the source is still
    /// in flight.
    const DELIVERY_STATUS: u32 = 1 << 12;

    /// Bit 14: the controller's report that a level-triggered interrupt from
    /// the pin has been accepted and not acknowledged.
    const REMOTE_IRR: u32 = 1 << 14;

    #[test]
    fn what_software_may_store_in_each_entry() {
        // Written out as literals rather than composed from the same constants
        // the masks are, so that a field moving fails a test instead of moving
        // with it. Vector 7:0 and mask 16 everywhere; delivery mode 10:8 in
        // every entry but the timer; polarity 13 and trigger mode 15 in the two
        // pins; timer mode 18:17 in the timer.
        for (entry, writable) in [
            (Entry::Timer, 0x0007_00FF),
            (Entry::Lint0, 0x0001_A7FF),
            (Entry::Lint1, 0x0001_A7FF),
            (Entry::Error, 0x0001_07FF),
            (Entry::Performance, 0x0001_07FF),
            (Entry::Thermal, 0x0001_07FF),
            (Entry::CorrectedMachineCheck, 0x0001_07FF),
        ] {
            assert_eq!(entry.writable(model::tests::AMD), writable, "{entry:?}");
        }
        // A processor without the timestamp-counter deadline has one bit of the
        // timer's mode field rather than two.
        assert_eq!(Entry::Timer.writable(model::tests::SPARSE), 0x0003_00FF);
    }

    #[test]
    fn what_no_software_may_put_a_one_in() {
        // The complement of the mask above, less the two bits the controller
        // reports through: those are read-only rather than reserved, so a write
        // of one is dropped and not refused.
        for (entry, reserved) in [
            (Entry::Timer, 0xFFF8_EF00),
            (Entry::Lint0, 0xFFFE_0800),
            (Entry::Lint1, 0xFFFE_0800),
            (Entry::Error, 0xFFFE_E800),
            (Entry::Performance, 0xFFFE_E800),
            (Entry::Thermal, 0xFFFE_E800),
            (Entry::CorrectedMachineCheck, 0xFFFE_E800),
        ] {
            assert_eq!(entry.reserved(model::tests::AMD), reserved, "{entry:?}");
        }
        assert_eq!(Entry::Timer.reserved(model::tests::SPARSE), 0xFFFC_EF00);
    }

    #[test]
    fn the_bits_the_controller_reports_through_are_read_only_and_not_reserved() {
        // The distinction one mask serving both purposes would lose, and the
        // reason it matters: a guest reads an entry, sets the mask bit in what
        // it read and writes the whole of it back, which is how every operating
        // system stops one of these sources. If the controller had a delivery in
        // flight, or held an unacknowledged level-triggered interrupt from a
        // pin, that value has the bit set — and refusing it would be a general
        // protection fault for writing back what the guest was just given.
        for model in [model::tests::AMD, model::tests::SPARSE] {
            for entry in Entry::ALL {
                assert_eq!(
                    entry.reserved(model) & DELIVERY_STATUS,
                    0,
                    "{entry:?} must not fault on the delivery-status bit"
                );
                assert_eq!(
                    entry.writable(model) & DELIVERY_STATUS,
                    0,
                    "{entry:?} must not store the delivery-status bit"
                );
                // Remote IRR exists in the two entries that describe a wire and
                // is reserved in the rest, so unlike delivery status it does
                // fault where the architecture has no field for it.
                assert_eq!(
                    entry.reserved(model) & REMOTE_IRR == 0,
                    entry.is_pin(),
                    "{entry:?}"
                );
                assert_eq!(entry.writable(model) & REMOTE_IRR, 0, "{entry:?}");
            }
        }
    }

    #[test]
    fn a_controller_has_at_least_the_timer_the_two_pins_and_the_error_entry() {
        // The smallest table the version register describes. A model built with
        // fewer would answer that a guest's own non-maskable interrupt pin is
        // not a register.
        assert_eq!(Entry::FEWEST, 4);
        for entry in [Entry::Timer, Entry::Lint0, Entry::Lint1, Entry::Error] {
            assert!(entry.index() < Entry::FEWEST, "{entry:?}");
        }
    }
}
