//! Handing the guest the controller's own interrupt sources.
//!
//! The local vector table is where a controller's own hardware reaches the
//! processor: its timer, its two interrupt pins, its thermal sensor, its
//! performance counters, its machine-check reporting. None of it is emulated,
//! because none of it can usefully be — a thermal sensor is a thermal sensor,
//! and a hypervisor that synthesised one would be inventing readings for
//! hardware the guest is otherwise being given directly.
//!
//! So each entry the guest programs is programmed onto the real controller,
//! with the guest's own mask, trigger mode and polarity.
//!
//! # The vector is the one thing that is not the guest's
//!
//! The real entry carries a vector this crate owns, one per source, and the
//! guest's own vector is what gets injected when one arrives.
//!
//! That is what makes an arrival unambiguous. Vectors on this machine are
//! shared between the guest's devices, the hypervisor's interprocessor
//! interrupts, and the controller's spurious and error vectors — an interrupt
//! arriving on a number the guest chose could have come from any of them. An
//! interrupt arriving on a number only this module ever programmed came from
//! exactly one place.
//!
//! # Two entries are not passed through
//!
//! The error entry is the host's and stays the host's. A controller reporting
//! its own errors is reporting them to whoever has to act on them, and the
//! guest cannot: the errors are the *real* controller's, produced by commands
//! this crate issued on the guest's behalf. The guest's own error reporting
//! comes from its emulated error status register instead.
//!
//! And an entry the guest programs to deliver something other than a vector —
//! a non-maskable interrupt, a system-management interrupt, INIT, or an
//! external interrupt — carries no vector to remap. Those are programmed in the
//! delivery mode the guest asked for and arrive on their own paths rather than
//! as a vector this module would recognise.

use apic::{Entry as HardwareEntry, LvtDelivery, Polarity, Source, Trigger as HardwareTrigger};
use descriptors::Vector;
use log::{trace, warn};

use crate::{
    lvt::{Delivery, Entry},
    state::Vlapic,
};

/// Brings every source the guest can reach into agreement with what it has
/// programmed.
///
/// Called whenever the guest writes a local vector table entry, or does
/// anything else that changes what those entries mean — software-disabling the
/// controller masks all of them, and the hardware has to follow.
///
/// Reprogramming all of them rather than the one that changed is deliberate:
/// the table is seven entries of one register each, the cost is a handful of
/// uncached writes on a path the guest takes rarely, and it is immune to a
/// guest that writes the registers in an order nothing anticipated.
pub(crate) fn reprogram(vlapic: &Vlapic) {
    for source in Source::ALL {
        // The timer is programmed by the timer module, which has the count and
        // the divide to go with the entry.
        if source == Source::Timer {
            continue;
        }
        program(vlapic, source);
    }
}

/// Brings one source into agreement with the guest's entry for it.
pub(crate) fn program(vlapic: &Vlapic, source: Source) {
    let entry = of(source);
    let Ok(local) = apic::local() else {
        return;
    };
    let programmed = describe(vlapic, entry, hardware_vector(entry));
    match local.program(source, programmed) {
        Ok(()) => trace!("vlapic: {} programmed its {source} source", vlapic.index()),
        // A controller that does not have the entry is not a failure: three of
        // the seven are optional, and a guest writing one this machine does not
        // have has written to a register that answers nothing.
        Err(apic::ApicError::NoSuchLvt { .. }) => {}
        Err(error) => warn!(
            "vlapic: {} could not program its {source} source: {error}",
            vlapic.index()
        ),
    }
}

/// What the real entry should say, given what the guest's says.
///
/// A masked entry is programmed masked, which is the whole of what masking
/// means: the hardware delivers nothing and there is nothing to inject.
fn describe(vlapic: &Vlapic, entry: Entry, vector: Vector) -> HardwareEntry {
    let guest = vlapic.lvt(entry);
    if guest.masked() {
        return HardwareEntry::masked();
    }
    // An entry that does not accept the mode the guest asked for is one real
    // hardware would refuse too. Three of the seven accept neither INIT nor an
    // external interrupt, and the timer and error entries accept nothing but a
    // fixed vector.
    let asked = Delivery::from_bits(guest.delivery());
    if asked.is_some_and(|delivery| !entry.allows(delivery)) {
        return HardwareEntry::masked();
    }
    let delivery = match asked {
        // The only case where the vector is substituted, because it is the only
        // case where the entry carries one.
        Some(Delivery::Fixed) => LvtDelivery::Fixed(vector),
        // These carry no vector, so there is nothing to substitute and the
        // guest's chosen delivery is programmed as it stands.
        Some(Delivery::NonMaskable) => LvtDelivery::NonMaskable,
        Some(Delivery::SystemManagement) => LvtDelivery::SystemManagement,
        Some(Delivery::External) => LvtDelivery::External,
        // INIT through a local vector table entry would reset this processor,
        // which is the host — and no entry the guest can reach is allowed to
        // deliver it anyway. A reserved encoding is the other thing a controller
        // does nothing with. Both are programmed masked.
        Some(Delivery::Init) | None => return HardwareEntry::masked(),
    };
    // Only the two pins have a polarity and a trigger mode; everything else is
    // edge triggered and active high by definition.
    if !matches!(entry, Entry::Lint0 | Entry::Lint1) {
        return HardwareEntry::new(delivery);
    }
    HardwareEntry::new(delivery).wired(
        if guest.active_low() {
            Polarity::ActiveLow
        } else {
            Polarity::ActiveHigh
        },
        if guest.level_triggered() {
            HardwareTrigger::Level
        } else {
            HardwareTrigger::Edge
        },
    )
}

/// Which of the guest's entries a source corresponds to.
const fn of(source: Source) -> Entry {
    match source {
        Source::Timer => Entry::Timer,
        Source::Lint0 => Entry::Lint0,
        Source::Lint1 => Entry::Lint1,
        Source::Thermal => Entry::Thermal,
        Source::Performance => Entry::Performance,
        Source::CorrectedMachineCheck => Entry::CorrectedMachineCheck,
    }
}

/// Which source an arrival on one of this crate's own vectors came from.
pub(crate) fn arrived_on(vector: Vector) -> Option<Entry> {
    let number = vector.number();
    let first = FIRST_HARDWARE_VECTOR.number();
    if number < first {
        return None;
    }
    Entry::ALL
        .into_iter()
        .find(|entry| first + entry_index(*entry) == number)
}

/// Which vector this crate programmed for an entry.
pub(crate) const fn hardware_vector(entry: Entry) -> Vector {
    Vector::new(FIRST_HARDWARE_VECTOR.number() + entry_index(entry))
}

/// An entry's position in the table, as a byte, so it can be added to a vector.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the local vector table has seven entries, so a position in it fits a byte"
)]
const fn entry_index(entry: Entry) -> u8 {
    entry.index() as u8
}

/// The first of the vectors this crate programs onto real hardware for the
/// controller's own sources.
///
/// High, and below the block the interprocessor interrupts take. High because a
/// level-triggered interrupt the guest has not finished with holds the real
/// controller's in-service bit and blocks everything of lower priority — and
/// the hypervisor's own sources must not be among the things a slow guest
/// driver can block.
pub(crate) const FIRST_HARDWARE_VECTOR: Vector = Vector::new(0xE0);

/// The vectors this crate takes must not collide with the two the controller
/// keeps for itself or with the block the interprocessor interrupts take.
const _: () = assert!(
    FIRST_HARDWARE_VECTOR.number() as usize + Entry::COUNT <= ipi::FIRST.number() as usize,
    "the local vector table's own vectors must sit below the interprocessor interrupt block"
);
