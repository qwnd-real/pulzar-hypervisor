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
//! Every function here is about the controller of the processor that calls it:
//! each pairs the [`Vlapic`] it is handed with [`apic::local`], which is this
//! processor's controller and not that controller's. A caller holding another
//! processor's row would program its own hardware from another guest's table,
//! and nothing local could notice. [`crate::delivery`] walks other processors'
//! rows; nothing here may be reached from there.
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
//! # Who owns which vector
//!
//! Every row of this is asked rather than named. Two of the host's vectors are
//! constants and the rest are handed out at run time, so [`arms`] consults
//! [`descriptors::is_claimed`] and [`Vector::is_exception`] — a rule written as
//! numbers would be wrong the moment one more interprocessor interrupt existed.
//!
//! | vectors | whose | what makes them so |
//! | --- | --- | --- |
//! | below [`Vector::FIRST_EXTERNAL`] | the architecture's exceptions | the host has a gate for every one it can return from, and an arrival on one nothing claimed stops the machine |
//! | [`Vector::NON_MASKABLE`] | the guest's, by way of the host | no handler claims it: it is counted against the guest's own controller and given to the guest on the way back in |
//! | [`Vector::FIRST_EXTERNAL`] up to [`ipi::FIRST`] | the guest's, entirely | nothing in the host claims one of these, which is the whole of what makes them the guest's |
//! | [`ipi::FIRST`] through [`ipi::LAST`] | the host's interprocessor interrupts | taken by [`descriptors::claim`] as each subsystem needs one, highest first, so which of them are claimed is not a compile-time fact |
//! | [`apic::ERROR`] | the host | the real controller's report of its own errors, which the guest cannot act on because they are the real controller's |
//! | [`apic::SPURIOUS`] | the host | the low byte of the register whose enable bit software-enables the machine's own controller |
//!
//! # Vectors that cannot be armed
//!
//! [`arms`] is the whole rule, and none of it is the architecture's. The
//! architectural rule is [`crate::priority::legal`] — the sixteen vectors of
//! class 0, which no controller delivers — and a guest that writes one of those
//! into an entry is told through its error status register at the moment of the
//! write, which is where hardware tells it.
//!
//! This rule is narrower, and both of the things it adds are this hypervisor's
//! own restriction on hardware that would have obliged:
//!
//! - The architecture's remaining exception vectors, the sixteen above class 0.
//!   A controller *will* deliver on them, and delivering one here would enter a
//!   host exception handler with no exception having occurred, which the host
//!   answers by stopping.
//! - Every vector the host has claimed a handler on, because such a vector's
//!   arrivals are adjudicated by that handler before anything else sees them.
//!   For the controller's own error and spurious vectors the handler *consumes*
//!   an arrival it decides is the host's, so a guest source on one of those
//!   loses every interrupt that coincides with the condition the handler looks
//!   for — and no read of any register can tell the two apart. For the
//!   interprocessor interrupts the same coincidence holds, and one more thing
//!   does: those vectors are the top priority class, where an acknowledgement
//!   cannot be withheld at all, so a level-triggered source there would be one
//!   this crate could not defer.
//!
//! Because neither is architectural, a refused vector sets no bit in the
//! guest's error status register: hardware raises no error for a vector it
//! would have delivered, and a guest told otherwise would be told its
//! controller found an illegal vector where the vector is perfectly legal. What
//! happens instead is what happens to every configuration refused here — the
//! source is programmed masked and the refusal is stated once, in the log; see
//! [`Refusal`].
//!
//! # Limitations
//!
//! External-interrupt delivery through LINT0 is not available: no legacy
//! interrupt controller is presented to the guest, and both of the machine's
//! own are masked before the guest runs, so no interrupt-acknowledge cycle can
//! be answered. A guest that configures virtual-wire mode sees the entry
//! refused rather than armed. Supporting it would require presenting an
//! interrupt controller the guest can drive and answering the acknowledge cycle
//! from it.
//!
//! What such a guest reads back is what it wrote, unmasked and external,
//! because the mask bit is the only field a readback could say a refusal
//! through and it is a bit the guest stores again on its next read-modify-write
//! — after which the entry is masked with the guest's own register saying it
//! asked for that. So the refusal is in the log and not in the register file,
//! and a guest calibrating on the legacy timer's line waits for an interrupt
//! that cannot arrive.

use core::fmt::{self, Display, Formatter};

use apic::{
    ApicError, Entry as HardwareEntry, LocalApic, LvtDelivery, Polarity, Source,
    Trigger as HardwareTrigger,
};
use descriptors::Vector;
use log::{trace, warn};

use crate::registers::{
    Vlapic,
    lvt::{Delivery, Entry, Lvt},
};

/// Brings every source the guest can reach into agreement with what it has
/// programmed.
///
/// Called whenever the guest writes a local vector table entry, or does
/// anything else that changes what those entries mean — including a processor
/// joining the guest, whose reset table says every source of its is masked
/// while firmware's wiring has left two of them armed.
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
#[must_use]
pub(crate) fn reprogram(vlapic: &Vlapic) -> bool {
    let Some(local) = controller(vlapic) else {
        return false;
    };
    let settled = Source::ALL
        .into_iter()
        // The timer is programmed by the timer module, which has the count, the
        // divide and the deadline to go with the entry, and which must not
        // restart it merely because something else changed.
        .filter(|source| *source != Source::Timer)
        // Folded rather than `all`, which stops at the first failure: every
        // source has to be programmed whatever the others did, and the answer is
        // whether all of them were.
        .fold(true, |settled, source| {
            program(vlapic, local, source) & settled
        });
    // One record for the pass rather than one per source. A guest's single
    // register write reaches five sources, and five lines of serial output per
    // write is how a guest spinning on one register stops a machine that has its
    // logging turned up.
    trace!(
        "vlapic: {} brought its sources into agreement with its guest's table",
        vlapic.index()
    );
    settled
}

/// Stops every source the guest can reach from delivering.
///
/// What a controller transition needs before it throws virtual state away, and
/// what a guest that software-disables its controller needs before its register
/// file says the sources are off: one left armed goes on delivering into a
/// controller that is no longer accepting anything.
///
/// Nothing of the old configuration is kept, because nothing restores it. Every
/// caller either resets the register file — after which each entry is at its
/// masked reset value — or brings hardware back into agreement with the guest's
/// own table through [`reprogram`], which builds each entry from that table and
/// reads no hardware at all. Masking each entry in place instead would be a
/// read-modify-write of a register the controller owns three bits of, in
/// exchange for a value nothing ever looks at again.
///
/// Answers whether everything really was quieted.
#[must_use]
pub(crate) fn quiesce(vlapic: &Vlapic) -> bool {
    let Some(local) = controller(vlapic) else {
        return false;
    };
    Source::ALL
        .into_iter()
        .filter(|source| *source != Source::Timer)
        .fold(true, |quiet, source| stop(vlapic, local, source) & quiet)
}

/// Stops one source delivering.
///
/// What a write that masks an entry needs before the guest's own register says
/// the source is off — the ordering is argued where the decision is made, in
/// `face::dispatch`.
///
/// [`Source::Timer`] is not one of these. Masking the timer has to leave its
/// mode, its count and any armed deadline where they are, which is
/// [`crate::hardware::timer::mask`]'s.
///
/// Answers whether it really was stopped.
pub(crate) fn mask(vlapic: &Vlapic, source: Source) -> bool {
    controller(vlapic).is_some_and(|local| stop(vlapic, local, source))
}

/// Brings one source into agreement with the guest's entry for it.
///
/// A configuration this hypervisor will not put on hardware is programmed
/// masked rather than left alone, because leaving it alone is leaving the
/// source delivering whatever it was last given.
///
/// Answers whether it was brought into agreement.
fn program(vlapic: &Vlapic, local: LocalApic, source: Source) -> bool {
    let entry = of(source);
    let described = match describe(entry, vlapic.lvt(entry)) {
        Ok(described) => described,
        Err(refusal) => {
            refused(vlapic, entry, refusal);
            HardwareEntry::masked()
        }
    };
    written(vlapic, local, source, described)
}

/// Stops one source delivering, whatever it was programmed with.
fn stop(vlapic: &Vlapic, local: LocalApic, source: Source) -> bool {
    written(vlapic, local, source, HardwareEntry::masked())
}

/// Puts one entry on real hardware and says whether hardware took it.
///
/// A controller that does not have the entry is not a failure so long as the
/// guest was not told it had one. The model takes its entry count from this
/// same controller, so a guest can only reach an entry the hardware has — and a
/// register the guest cannot reach cannot disagree with hardware.
fn written(vlapic: &Vlapic, local: LocalApic, source: Source, entry: HardwareEntry) -> bool {
    match local.program(source, entry) {
        Ok(()) => true,
        Err(ApicError::NoSuchLvt { .. }) => !vlapic.model().has(of(source)),
        Err(error) => {
            refused(vlapic, of(source), Refusal::Hardware(error));
            false
        }
    }
}

/// What the real entry should say, given what the guest's says.
///
/// Decided from two things and nothing else — which entry it is, and what the
/// guest wrote in it — so that every shape a guest can write is answerable
/// without a controller.
///
/// A masked entry is programmed masked, which is the whole of what masking
/// means: the hardware delivers nothing and there is nothing to inject.
///
/// # Errors
///
/// The [`Refusal`] for a configuration that will not be put on real hardware.
/// Programming such a source masked instead is the caller's.
fn describe(entry: Entry, guest: Lvt) -> Result<HardwareEntry, Refusal> {
    if guest.masked() {
        return Ok(HardwareEntry::masked());
    }
    // An entry that does not accept the mode the guest asked for is one the
    // entry itself has already answered for — including the two modes no entry
    // accepts — and a reserved encoding is something a controller given it does
    // nothing defined with.
    let asked = Delivery::from_bits(guest.delivery())
        .filter(|delivery| entry.allows(*delivery))
        .ok_or(Refusal::Delivery(guest.delivery()))?;
    let delivery = match asked {
        Delivery::Fixed => {
            let vector = guest.vector();
            if !arms(vector) {
                return Err(Refusal::Vector(vector));
            }
            LvtDelivery::Fixed(vector)
        }
        // This one carries no vector, so there is nothing to admit and the
        // guest's chosen delivery is programmed as it stands.
        Delivery::NonMaskable => LvtDelivery::NonMaskable,
        // A system-management interrupt and an INIT are refused for every entry
        // before they get here, and the arm is written out rather than answered
        // by a wildcard so that a mode the entries start allowing has to be
        // decided here as well.
        //
        // An external interrupt is refused for a subtler reason. It means the
        // processor runs an acknowledge cycle to a legacy controller and takes
        // whatever vector that returns, bypassing the local controller's
        // in-service register entirely — so nothing is accepted, nothing is
        // owed, and an arrival taking the ordinary path here would issue an
        // acknowledgement that retires an unrelated interrupt. There is nothing
        // to mediate in any case: this hypervisor masks every input of both
        // legacy controllers during bring-up, so the pin can never assert.
        Delivery::SystemManagement | Delivery::Init | Delivery::External => {
            return Err(Refusal::Delivery(guest.delivery()));
        }
    };
    // Only the two pins have a polarity and a trigger mode. Everything else is
    // edge triggered and active high by definition, and forwarding a guest's
    // bits for a source that has no wire would write meaningless state into real
    // hardware.
    if !entry.is_pin() {
        return Ok(HardwareEntry::new(delivery));
    }
    // A pin's trigger mode is the wire's, and only a fixed delivery reads it.
    // Every other mode is an event with its own signalling — a non-maskable
    // interrupt is edge triggered by definition, and programming one level
    // triggered is a configuration the architecture does not define and hardware
    // need not honour.
    let trigger = if matches!(asked, Delivery::Fixed) && guest.level_triggered() {
        HardwareTrigger::Level
    } else {
        HardwareTrigger::Edge
    };
    Ok(HardwareEntry::new(delivery).wired(polarity(guest.active_low()), trigger))
}

/// Says once that a source's configuration did not reach real hardware.
///
/// Once per entry per kind, and not once per attempt: every one of these is
/// re-derived on every reprogram, so a guest that leaves a refused entry in
/// place would otherwise have a line of serial output per unrelated register
/// write — with a machine-wide lock held and interrupts off for each of them.
pub(crate) fn refused(vlapic: &Vlapic, entry: Entry, refusal: Refusal) {
    if vlapic.diagnostics().say_refusal(entry, refusal) {
        warn!(
            "vlapic: {} is not giving its {entry:?} source what its guest asked of it: {refusal}",
            vlapic.index()
        );
    }
}

/// Why a source's configuration did not reach real hardware.
///
/// None of these is an error the architecture has a bit for, so none of them
/// records one: hardware would either have honoured the configuration or have
/// refused it silently. What the guest gets is a source that does not deliver,
/// and what the operator gets is one line per entry per kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Refusal {
    /// The delivery mode the guest chose is one this hypervisor will not put on
    /// the machine's own controller, or one the architecture reserves. Carries
    /// the field as the guest wrote it, because a reserved encoding has no
    /// name.
    Delivery(u8),
    /// The vector is not one a source may be armed with here; see [`arms`].
    Vector(Vector),
    /// The timer's own rate — which is measured, because no register reports it
    /// — is not one a shortest period can be worked out from, so no periodic
    /// count can be shown to be one the machine can answer. Carries the
    /// measurement.
    Rate(u64),
    /// The real controller refused the entry, or could not be reached to be
    /// given it.
    Hardware(ApicError),
}

impl Refusal {
    /// How many kinds of refusal there are, which is what the latch that
    /// reports each of them once per entry is sized by.
    pub(crate) const COUNT: usize = 4;

    /// Which kind this is, as a position in that latch.
    ///
    /// The payload takes no part: two vectors refused in the same entry are the
    /// same thing said twice, and the first of them is the one worth saying.
    pub(crate) const fn kind(self) -> usize {
        match self {
            Self::Delivery(_) => 0,
            Self::Vector(_) => 1,
            Self::Rate(_) => 2,
            Self::Hardware(_) => 3,
        }
    }
}

impl Display for Refusal {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Delivery(mode) => write!(
                formatter,
                "delivery mode {mode:#05b} is not one this hypervisor puts on real hardware"
            ),
            Self::Vector(vector) => {
                write!(formatter, "{vector} is not one a source may be armed with")
            }
            Self::Rate(hertz) => write!(
                formatter,
                "a measured rate of {hertz} Hz is not one the shortest period a periodic timer may \
                 run at can be worked out from"
            ),
            Self::Hardware(error) => write!(formatter, "the real controller refused it: {error}"),
        }
    }
}

/// Whether real hardware may be told to deliver a source on this vector.
///
/// The exceptions, and whatever the host has claimed. Both are asked of the
/// authority that owns the answer rather than restated here: the architecture
/// fixes which vectors are exceptions, and the descriptor tables know which
/// vectors have a handler, including the ones handed out at run time.
///
/// Shared with the timer, which is programmed elsewhere and would otherwise
/// arm a source on a vector this refuses.
pub(crate) fn arms(vector: Vector) -> bool {
    !vector.is_exception() && !descriptors::is_claimed(vector)
}

/// This processor's controller, or nothing having said why.
///
/// Unreachable once [`crate::install`] has run: the real base register is the
/// host's and no guest can write it, so a controller that answered once answers
/// always. It is said unlatched for that reason — reaching this is a broken
/// invariant rather than something a guest can drive, and it leaves every
/// source exactly as it was, which for a controller being switched off means
/// armed.
fn controller(vlapic: &Vlapic) -> Option<LocalApic> {
    match apic::local() {
        Ok(local) => Some(local),
        Err(error) => {
            warn!(
                "vlapic: {} could not reach its controller to program its sources: {error}",
                vlapic.index()
            );
            None
        }
    }
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

/// The two mappings above are written by hand in opposite directions, and
/// nothing but this says they agree: one that had drifted would program a
/// source from another source's entry, and would report a guest's read of one
/// entry out of another's real register — invisibly, and only on the hardware
/// that has the entries in question.
const _: () = {
    let mut index = 0;
    while index < Source::ALL.len() {
        let source = Source::ALL[index];
        assert!(
            matches!(source_of(of(source)), Some(same) if same as usize == source as usize),
            "every source has to be the source of the entry it is the entry of"
        );
        index += 1;
    }
};

/// How a pin asserts.
const fn polarity(active_low: bool) -> Polarity {
    if active_low {
        Polarity::ActiveLow
    } else {
        Polarity::ActiveHigh
    }
}

#[cfg(test)]
mod tests {
    //! Two decisions here need no controller, and they are the two a mistake in
    //! costs the *machine* rather than the guest: which vectors a source may be
    //! armed with, and what a guest's entry becomes on real hardware.

    use apic::{Entry as HardwareEntry, LvtDelivery, Trigger as HardwareTrigger};
    use descriptors::{Disposition, Interrupt, Vector};

    use super::{Refusal, arms, describe, polarity};
    use crate::registers::lvt::{Delivery, Entry, Lvt};

    /// A vector nothing else in this crate's tests uses, standing in for one
    /// the host has taken.
    const CLAIMED: Vector = Vector::new(0x47);

    /// One in the middle of the range the host claims nothing in, standing in
    /// for the guest's own.
    const GUEST: Vector = Vector::new(0x33);

    /// What a claimed vector's handler would answer, which nothing here runs.
    fn handler(_: &Interrupt) -> Disposition {
        Disposition::Passed
    }

    #[test]
    fn no_source_is_armed_on_a_vector_the_host_has_claimed() {
        // Whatever the host claimed, and not a range: two of its vectors are
        // constants and the rest are handed out at run time, so the test claims
        // one the same way the host does.
        assert!(arms(CLAIMED), "nothing has claimed it yet");
        descriptors::register(CLAIMED, handler).expect("the vector is free");
        assert!(!arms(CLAIMED));
    }

    #[test]
    fn no_source_is_armed_on_an_exception_vector() {
        // The architecture's own, and the reason is not the host's claim: a
        // controller delivering one of these enters a host exception handler with
        // no exception having occurred.
        for number in 0..Vector::FIRST_EXTERNAL.number() {
            assert!(!arms(Vector::new(number)), "vector {number:#x}");
        }
        assert!(arms(Vector::FIRST_EXTERNAL));
    }

    #[test]
    fn the_vectors_the_host_owns_are_refused_wherever_the_host_took_them() {
        // The ownership table in the module doc, asserted from the same
        // constants it names rather than from the numbers they happen to be.
        // Each of these is claimed here as the host claims it, because a host
        // test has claimed nothing.
        for vector in [apic::ERROR, apic::SPURIOUS, ipi::FIRST, ipi::LAST] {
            assert!(arms(vector), "{vector} beforehand");
            descriptors::register(vector, handler).expect("the vector is free");
            assert!(!arms(vector), "{vector}");
        }
        // And the range between the exceptions and the host's own pool is the
        // guest's entirely, which is what makes the refusals above narrow.
        let below = Vector::new(ipi::FIRST.number() - 1);
        assert!(arms(below) && arms(Vector::FIRST_EXTERNAL));
    }

    #[test]
    fn a_masked_entry_is_programmed_masked_whatever_else_it_says() {
        // Including the shapes that would be refused unmasked: masking is what
        // the guest asked for, and there is nothing to refuse about it.
        for entry in Entry::ALL {
            for delivery in 0..8 {
                let guest = Lvt::new()
                    .with_masked(true)
                    .with_delivery(delivery)
                    .with_vector(Vector::new(0x08))
                    .with_level_triggered(true);
                assert_eq!(
                    describe(entry, guest),
                    Ok(HardwareEntry::masked()),
                    "{entry:?} with delivery {delivery:#05b}"
                );
            }
        }
    }

    #[test]
    fn a_fixed_source_is_armed_with_the_guests_own_vector() {
        for entry in Entry::ALL {
            let guest = Lvt::new().with_vector(GUEST);
            let armed = HardwareEntry::new(LvtDelivery::Fixed(GUEST));
            let expected = if entry.is_pin() {
                armed.wired(polarity(false), HardwareTrigger::Edge)
            } else {
                armed
            };
            assert_eq!(describe(entry, guest), Ok(expected), "{entry:?}");
        }
    }

    #[test]
    fn a_vector_no_source_may_be_armed_with_is_refused_rather_than_masked_quietly() {
        // The refusal is the value, because it is what the diagnostic is said
        // from: an entry masked with nothing said is the defect this closes.
        let exception = Lvt::new().with_vector(Vector::new(0x1F));
        assert_eq!(
            describe(Entry::Thermal, exception),
            Err(Refusal::Vector(Vector::new(0x1F)))
        );
        assert_eq!(
            describe(Entry::Timer, exception),
            Err(Refusal::Vector(Vector::new(0x1F)))
        );
    }

    #[test]
    fn a_reserved_delivery_encoding_is_refused_in_every_entry() {
        for entry in Entry::ALL {
            for delivery in [0b001, 0b011, 0b110] {
                let guest = Lvt::new().with_delivery(delivery).with_vector(GUEST);
                assert_eq!(
                    describe(entry, guest),
                    Err(Refusal::Delivery(delivery)),
                    "{entry:?}"
                );
            }
        }
    }

    #[test]
    fn the_three_modes_no_entry_delivers_are_refused_in_all_of_them() {
        // A system-management interrupt would take the *host* into
        // system-management mode; an INIT would reset the host processor; and an
        // external interrupt would run an acknowledge cycle to a legacy
        // controller this hypervisor has masked, taking whatever vector it
        // returned. The two pins are the interesting rows: the architecture
        // defines all three for them.
        for entry in Entry::ALL {
            for delivery in [
                Delivery::SystemManagement,
                Delivery::Init,
                Delivery::External,
            ] {
                let bits = delivery as u8;
                let guest = Lvt::new().with_delivery(bits).with_vector(GUEST);
                assert_eq!(
                    describe(entry, guest),
                    Err(Refusal::Delivery(bits)),
                    "{entry:?} with {delivery:?}"
                );
            }
        }
    }

    #[test]
    fn a_non_maskable_source_carries_no_vector_and_is_never_level_triggered() {
        // Both pins and the three internal sources that have a delivery field:
        // the vector is not read for this mode, and the architecture delivers it
        // edge triggered whatever the entry's trigger bit says.
        let guest = Lvt::new()
            .with_delivery(Delivery::NonMaskable as u8)
            .with_vector(Vector::new(0x08))
            .with_level_triggered(true)
            .with_active_low(true);
        for entry in [
            Entry::Lint0,
            Entry::Lint1,
            Entry::Performance,
            Entry::Thermal,
            Entry::CorrectedMachineCheck,
        ] {
            let armed = HardwareEntry::new(LvtDelivery::NonMaskable);
            let expected = if entry.is_pin() {
                armed.wired(polarity(true), HardwareTrigger::Edge)
            } else {
                armed
            };
            assert_eq!(describe(entry, guest), Ok(expected), "{entry:?}");
        }
        // And the one entry whose delivery is the architecture's rather than
        // the guest's refuses it: the timer, which has no message-type field at
        // all.
        assert_eq!(
            describe(Entry::Timer, guest),
            Err(Refusal::Delivery(Delivery::NonMaskable as u8))
        );
    }

    #[test]
    fn only_a_pin_carries_a_wires_polarity_and_trigger_mode() {
        let guest = Lvt::new()
            .with_vector(GUEST)
            .with_level_triggered(true)
            .with_active_low(true);
        for entry in Entry::ALL {
            let armed = HardwareEntry::new(LvtDelivery::Fixed(GUEST));
            let expected = if entry.is_pin() {
                armed.wired(polarity(true), HardwareTrigger::Level)
            } else {
                armed
            };
            assert_eq!(describe(entry, guest), Ok(expected), "{entry:?}");
        }
    }

    #[test]
    fn nothing_a_guest_can_write_arms_more_than_it_asked_for() {
        // Every shape: every entry, every delivery encoding the field can hold,
        // masked and not, both wiring bits, and vectors from each part of the
        // range. What is asserted is the property the whole module exists for —
        // that what reaches hardware is either nothing at all or exactly what
        // the guest asked for, on the vector the guest chose.
        for entry in Entry::ALL {
            for delivery in 0..8 {
                for wiring in 0..8 {
                    for vector in [
                        Vector::new(0),
                        Vector::new(0x1F),
                        Vector::FIRST_EXTERNAL,
                        GUEST,
                        apic::SPURIOUS,
                    ] {
                        let guest = Lvt::new()
                            .with_delivery(delivery)
                            .with_vector(vector)
                            .with_masked(wiring & 0b100 != 0)
                            .with_level_triggered(wiring & 0b01 != 0)
                            .with_active_low(wiring & 0b10 != 0);
                        let described = describe(entry, guest).unwrap_or(HardwareEntry::masked());
                        assert!(
                            permitted(entry, guest).contains(&described),
                            "{entry:?} armed something else from delivery {delivery:#05b} {vector}"
                        );
                        // The two the module refuses outright, whatever else is
                        // set: an exception vector, and a masked entry.
                        if vector.is_exception() && delivery == Delivery::Fixed as u8
                            || guest.masked()
                        {
                            assert_eq!(
                                described,
                                HardwareEntry::masked(),
                                "{entry:?} with {vector}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Every entry a guest's write may legitimately become: one that delivers
    /// nothing, or one that delivers what the guest asked on the vector the
    /// guest named, wired the way the guest wired it.
    fn permitted(entry: Entry, guest: Lvt) -> [HardwareEntry; 3] {
        let fixed = HardwareEntry::new(LvtDelivery::Fixed(guest.vector()));
        let non_maskable = HardwareEntry::new(LvtDelivery::NonMaskable);
        if !entry.is_pin() {
            return [HardwareEntry::masked(), fixed, non_maskable];
        }
        let polarity = polarity(guest.active_low());
        let trigger = if guest.level_triggered() {
            HardwareTrigger::Level
        } else {
            HardwareTrigger::Edge
        };
        [
            HardwareEntry::masked(),
            fixed.wired(polarity, trigger),
            non_maskable.wired(polarity, HardwareTrigger::Edge),
        ]
    }
}
