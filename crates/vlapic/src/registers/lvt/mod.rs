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
//! [`Entry::ALL`] is in the order the architecture counts these in, which is
//! what the version register reports the highest index of. It is deliberately
//! not the order the registers sit at — the corrected-machine-check entry is
//! last in the count and first in memory — so nothing may derive one order from
//! the other.

mod shape;
mod table;

pub(crate) use crate::registers::lvt::shape::{Delivery, Lvt, TimerMode};
use descriptors::Vector;

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
    /// How many a particular controller has is [`Model::max_lvt`]'s to say, and
    /// is usually fewer: three of these are optional. This is the size of the
    /// array they are stored in and the largest a model may claim.
    pub(crate) const COUNT: usize = 7;

    /// How few a controller may have.
    ///
    /// The version register reports one less than the count, so a controller
    /// with none could not describe itself. Every controller has at least the
    /// timer.
    pub(crate) const FEWEST: usize = 1;

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

    /// Which bits of this entry software may set, in this model. Everything
    /// else is reserved in it and is dropped from a write rather than
    /// stored.
    ///
    /// The delivery-status and remote-IRR bits are absent from every answer:
    /// both are the controller's to report, and letting a guest write either
    /// would let it claim a delivery that never happened or retire one that is
    /// still outstanding.
    ///
    /// Two entries have a shape the model decides rather than the entry. The
    /// error entry's message type is writable on AMD and reserved on Intel. And
    /// the timer's mode field is only two bits wide on a processor that
    /// implements the timestamp-counter deadline: without it there are two
    /// modes rather than three, the bit that would select the third is
    /// reserved, and a guest allowed to set it would select a mode its own
    /// `CPUID` denies and real hardware would refuse to be programmed for.
    pub(crate) const fn writable(self, model: Model) -> u32 {
        let delivery = if model.has_delivery(self) {
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
    fn the_timer_carries_no_delivery_mode_on_either_vendor() {
        for model in [model::tests::AMD, model::tests::INTEL] {
            assert!(!model.has_delivery(Entry::Timer));
            assert_eq!(Entry::Timer.writable(model) & WRITABLE_DELIVERY, 0);
            assert!(model.allows(Entry::Timer, Delivery::Fixed));
            assert!(!model.allows(Entry::Timer, Delivery::NonMaskable));
        }
    }

    #[test]
    fn the_error_entry_takes_a_message_type_on_amd_alone() {
        let amd = model::tests::AMD;
        let intel = model::tests::INTEL;
        assert_eq!(Entry::Error.writable(amd) & WRITABLE_DELIVERY, {
            WRITABLE_DELIVERY
        });
        assert_eq!(Entry::Error.writable(intel) & WRITABLE_DELIVERY, 0);
        assert!(amd.allows(Entry::Error, Delivery::NonMaskable));
        assert!(!intel.allows(Entry::Error, Delivery::NonMaskable));
        // Neither vendor lets the error entry reach for a wire's modes.
        assert!(!amd.allows(Entry::Error, Delivery::External));
        assert!(!amd.allows(Entry::Error, Delivery::Init));
    }

    #[test]
    fn corrected_machine_check_refuses_init_and_external() {
        let model = model::tests::AMD;
        let entry = Entry::CorrectedMachineCheck;
        assert!(model.allows(entry, Delivery::Fixed));
        assert!(model.allows(entry, Delivery::SystemManagement));
        assert!(model.allows(entry, Delivery::NonMaskable));
        assert!(!model.allows(entry, Delivery::Init));
        assert!(!model.allows(entry, Delivery::External));
    }
}
