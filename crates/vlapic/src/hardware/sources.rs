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
//! with the guest's own vector, mask, trigger mode and polarity.
//!
//! # The vector is the guest's too
//!
//! It has to be, and the reason is that the alternative does not work. A
//! hypervisor could claim a private vector per source, program that onto the
//! hardware, and translate back to the guest's vector on arrival — which makes
//! an arrival on one of those numbers unambiguous, but only by assuming nothing
//! else can send one. On a machine whose I/O controllers are passed through,
//! something else can: the guest programs its own I/O APIC and its own devices'
//! messages, directly, with whatever vectors it likes. A device pointed at a
//! claimed number would arrive looking exactly like a timer, and be translated
//! through an unrelated entry.
//!
//! Reserving numbers away from the guest is not available either, for the same
//! reason — nothing mediates the writes that would have to be checked.
//!
//! Programming the guest's own vector removes the question rather than
//! answering it. Two sources sharing a vector is then not a hypervisor's
//! confusion but the guest's own configuration, reproduced exactly: on real
//! hardware a guest that points its timer and a device at one number gets both
//! on that number, and its handler is what has to cope. Nothing here has to
//! know which source an arrival came from, because nothing here has to
//! translate it.
//!
//! # What the host keeps, and why it is only two things
//!
//! The error entry is the host's. A controller reporting its own errors is
//! reporting them to whoever has to act on them, and the guest cannot: the
//! errors are the *real* controller's, produced in part by commands this crate
//! issued on the guest's behalf. The guest's error reporting is delivered from
//! its own emulated error status register instead, which reports exactly the
//! errors the guest caused.
//!
//! The spurious vector register is the host's for a blunter reason: it holds
//! the bit that software-enables the real controller. A guest that cleared it
//! would stop the machine's own timer, interprocessor interrupts and devices.
//!
//! Both of those, and the block the interprocessor interrupts take, are numbers
//! the guest can still point a device at. Each is told apart from a guest's
//! interrupt on the same number by its own handler and passed on when it is not
//! the host's — a mailbox for the interprocessor interrupts, the error status
//! register for the error vector, and the in-service bit for the spurious one,
//! which a withdrawn interrupt never sets.
//!
//! # Vectors that cannot be armed
//!
//! Three ranges, and one of them is a hazard rather than a rule. The controller
//! itself refuses vectors 0 through 15. Above those, 16 through 31 are the
//! architecture's remaining exception vectors: a controller *will* deliver on
//! them, and delivering one here would enter a host exception handler with no
//! exception having occurred, which the host answers by stopping. And the
//! host's own error and spurious vectors would be consumed by the handlers that
//! own them before anything could pass them on, because for those two the
//! pass-through test is about the arrival rather than about the vector.
//!
//! A source whose guest vector falls in any of them is programmed masked and
//! the attempt is recorded in the guest's error status register. For the first
//! range that is exactly what hardware does. For the other two it is a
//! deliberate departure — hardware would deliver — taken because the
//! alternative is a guest halting the host by writing a register.

use apic::{Entry as HardwareEntry, LvtDelivery, Polarity, Source, Trigger as HardwareTrigger};
use descriptors::Vector;
use log::{trace, warn};

use crate::registers::{
    Vlapic,
    error::Errors,
    lvt::{Delivery, Entry},
};

/// Brings every source the guest can reach into agreement with what it has
/// programmed.
///
/// Called whenever the guest writes a local vector table entry, or does
/// anything else that changes what those entries mean.
///
/// Reprogramming all of them rather than the one that changed is deliberate:
/// the table is a handful of entries of one register each, the cost is a few
/// uncached writes on a path the guest takes rarely, and it is immune to a
/// guest that writes the registers in an order nothing anticipated.
///
/// Answers whether every source the model says exists was brought into
/// agreement. A caller making a transition the guest must not observe half of —
/// disabling the controller, applying a reset — needs to know that, because a
/// source left as it was is one that can still deliver.
pub(crate) fn reprogram(vlapic: &Vlapic) -> bool {
    Source::ALL
        .into_iter()
        // The timer is programmed by the timer module, which has the count, the
        // divide and the deadline to go with the entry, and which must not
        // restart it merely because something else changed.
        .filter(|source| *source != Source::Timer)
        // Folded rather than `all`, which stops at the first failure: every
        // source has to be programmed whatever the others did, and the answer is
        // whether all of them were.
        .fold(true, |settled, source| program(vlapic, source) & settled)
}

/// Stops every source the guest can reach from delivering, leaving what they
/// were programmed with in place.
///
/// What a controller transition needs before it throws virtual state away.
/// Masking rather than rewriting is the point: a source is quieted without
/// losing the configuration behind it, so a transition that has to put things
/// back is restoring rather than reconstructing.
///
/// Answers whether everything really was quieted.
pub(crate) fn quiesce(vlapic: &Vlapic) -> bool {
    let Ok(local) = apic::local() else {
        return false;
    };
    Source::ALL
        .into_iter()
        .filter(|source| *source != Source::Timer)
        .fold(true, |quiet, source| {
            let outcome = local
                .source(source)
                .and_then(|entry| local.program(source, entry.delivering(false)));
            match outcome {
                // A source the controller does not have is a source that cannot
                // deliver, which is what was being asked for.
                Ok(()) | Err(apic::ApicError::NoSuchLvt { .. }) => quiet,
                Err(error) => {
                    warn!(
                        "vlapic: {} could not quiesce its {source} source: {error}",
                        vlapic.index()
                    );
                    false
                }
            }
        })
}

/// Brings one source into agreement with the guest's entry for it.
///
/// Answers whether it was brought into agreement.
pub(crate) fn program(vlapic: &Vlapic, source: Source) -> bool {
    let entry = of(source);
    let Ok(local) = apic::local() else {
        return false;
    };
    match local.program(source, describe(vlapic, entry)) {
        Ok(()) => {
            trace!("vlapic: {} programmed its {source} source", vlapic.index());
            true
        }
        // A controller that does not have the entry is not a failure so long as
        // the guest was not told it had one. The model takes its entry count
        // from this same controller, so a guest can only reach an entry the
        // hardware has — and a register the guest cannot reach cannot disagree
        // with hardware.
        Err(apic::ApicError::NoSuchLvt { .. }) => !vlapic.model().has(entry),
        Err(error) => {
            warn!(
                "vlapic: {} could not program its {source} source: {error}",
                vlapic.index()
            );
            false
        }
    }
}

/// What the real entry should say, given what the guest's says.
///
/// A masked entry is programmed masked, which is the whole of what masking
/// means: the hardware delivers nothing and there is nothing to inject.
fn describe(vlapic: &Vlapic, entry: Entry) -> HardwareEntry {
    let guest = vlapic.lvt(entry);
    if guest.masked() {
        return HardwareEntry::masked();
    }
    let model = vlapic.model();
    // An entry that does not accept the mode the guest asked for is one real
    // hardware would refuse too, and a reserved encoding is something a
    // controller given it does nothing defined with.
    let Some(asked) = Delivery::from_bits(guest.delivery()).filter(|d| model.allows(entry, *d))
    else {
        return HardwareEntry::masked();
    };
    let delivery = match asked {
        Delivery::Fixed => match armable(vlapic, guest.vector()) {
            Some(vector) => LvtDelivery::Fixed(vector),
            None => return HardwareEntry::masked(),
        },
        // These carry no vector, so there is nothing to check and the guest's
        // chosen delivery is programmed as it stands.
        Delivery::NonMaskable => LvtDelivery::NonMaskable,
        Delivery::SystemManagement => LvtDelivery::SystemManagement,
        // INIT through a local vector table entry would reset this processor,
        // which is the host — and no entry the guest can reach is allowed to
        // deliver it anyway.
        //
        // An external interrupt is refused for a subtler reason. It means the
        // processor runs an acknowledge cycle to a legacy controller and takes
        // whatever vector that returns, bypassing the local controller's
        // in-service register entirely — so nothing is accepted, nothing is
        // owed, and an arrival taking the ordinary path here would issue an
        // acknowledgement that retires an unrelated interrupt. There is nothing
        // to mediate in any case: this hypervisor masks every input of both
        // legacy controllers during bring-up, so the pin can never assert.
        Delivery::Init | Delivery::External => return HardwareEntry::masked(),
    };
    // Only the two pins have a polarity and a trigger mode. Everything else is
    // edge triggered and active high by definition, and forwarding a guest's
    // bits for a source that has no wire would write meaningless state into real
    // hardware.
    if !entry.is_pin() {
        return HardwareEntry::new(delivery);
    }
    // A pin's trigger mode is the wire's, and only a fixed delivery reads it.
    // Every other mode is an event with its own signalling — a non-maskable
    // interrupt or a system-management interrupt is edge triggered by
    // definition, and programming one level triggered is a configuration the
    // architecture does not define and hardware need not honour.
    let trigger = if matches!(asked, Delivery::Fixed) && guest.level_triggered() {
        HardwareTrigger::Level
    } else {
        HardwareTrigger::Edge
    };
    HardwareEntry::new(delivery).wired(polarity(guest.active_low()), trigger)
}

/// The vector a source may actually be armed with, or `None` if it may not be
/// armed at all.
///
/// Recording the refusal here rather than at the write is what makes it match
/// hardware: the error belongs to the moment a source would have delivered on
/// an impossible vector, and a guest that programs one into a masked entry and
/// never unmasks it has not caused an interrupt to go missing.
///
/// Shared with the timer, which is programmed elsewhere and would otherwise
/// mask itself over an impossible vector without telling the guest why.
pub(crate) fn armable(vlapic: &Vlapic, vector: Vector) -> Option<Vector> {
    if arms(vector) {
        return Some(vector);
    }
    vlapic.errors().record(Errors::RECEIVE_ILLEGAL_VECTOR);
    None
}

/// Whether real hardware may be told to deliver a source on this vector.
fn arms(vector: Vector) -> bool {
    !vector.is_exception() && vector != apic::ERROR && vector != apic::SPURIOUS
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

/// Which source one of the guest's entries is, for the entries that are one.
///
/// The error entry is not: it is the host's, and [`Source`] deliberately has no
/// name for it.
pub(crate) const fn source_of(entry: Entry) -> Option<Source> {
    match entry {
        Entry::Timer => Some(Source::Timer),
        Entry::Lint0 => Some(Source::Lint0),
        Entry::Lint1 => Some(Source::Lint1),
        Entry::Thermal => Some(Source::Thermal),
        Entry::Performance => Some(Source::Performance),
        Entry::CorrectedMachineCheck => Some(Source::CorrectedMachineCheck),
        Entry::Error => None,
    }
}

/// How a pin asserts.
const fn polarity(active_low: bool) -> Polarity {
    if active_low {
        Polarity::ActiveLow
    } else {
        Polarity::ActiveHigh
    }
}
