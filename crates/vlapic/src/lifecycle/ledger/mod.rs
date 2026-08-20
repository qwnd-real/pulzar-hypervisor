//! What real hardware is still holding on the guest's behalf.
//!
//! A level-triggered interrupt is asserted until whatever raised it is dealt
//! with, so the real controller's acknowledgement cannot be issued when the
//! interrupt arrives: doing so would deliver it again immediately, forever. It
//! is withheld until the guest's own driver has finished, which the guest
//! announces by acknowledging its emulated controller. Between those two
//! moments the real controller is holding a vector in service for a guest that
//! has not finished with it, and that is a debt this module keeps.
//!
//! # Two ledgers, because two machines
//!
//! What a debt costs to settle turns on one fact about the hardware: whether
//! its acknowledgement register takes a vector.
//!
//! Ordinarily it does not. Writing it retires whichever vector the controller
//! holds highest rather than one the writer names — so a debt may only be paid
//! at the moment its own vector *is* that highest one, every payment is a read
//! of the in-service bank followed by a write that depends on what it said, and
//! a debt nothing will ever acknowledge cannot be retired at all. [`deferred`]
//! is that machine, and each of those three is a reason it is as long as it is.
//!
//! The extended register space AMD puts above the architectural registers takes
//! the fact away. Its acknowledgement names its vector, and its
//! interrupt-enable bank decides per vector whether the controller accepts an
//! interrupt at all. Between them every one of those difficulties goes: a debt
//! is paid the moment the guest gives its acknowledgement, whatever else is in
//! service, and a debt nothing will acknowledge is paid as well — once the
//! vector it arrived on has been stopped from being accepted again.
//! [`immediate`] is that machine, and it is shorter because there is less to
//! say.
//!
//! Which arm a controller gets is decided once, from [`Extended`], and it is
//! the same question [`apic::LocalApic::specific`] answers — so a ledger that
//! retires by name is never built for a controller that cannot.
//!
//! # What does not differ
//!
//! Both arms are told the same four things — a debt was taken on, the guest has
//! discharged one, nothing will discharge one, and the guest is gone — and both
//! answer the same question about what is left.
//!
//! Both are handed the controller rather than reaching for it, because *which*
//! controller a debt is settled through is the one thing that must not be
//! assumed: an acknowledgement goes to whichever controller the processor
//! issuing it is running on, and settling through the wrong one retires an
//! interrupt belonging to somebody else. Passing it in is also what makes every
//! interleaving in either arm a test, since nothing then has to be on a
//! processor that has a controller at all.

pub(crate) mod deferred;
pub(crate) mod immediate;

use core::fmt::{self, Display, Formatter};

use apic::{Extended, LocalApic};
use descriptors::Vector;
use x86_64::instructions::interrupts;

use crate::lifecycle::ledger::{deferred::Deferred, immediate::Immediate};

/// The real controller, as a ledger needs it.
///
/// Six operations, and which of them an arm uses is the whole difference
/// between the two: [`deferred`] reads the in-service bank and issues the
/// acknowledgement that takes no vector, [`immediate`] retires a vector by name
/// and decides which vectors are accepted, and both ask whether the controller
/// can be reached and both need a window nothing else on this processor can
/// reach it in.
pub(crate) trait Controller {
    /// The highest-priority vector the controller is holding in service, or
    /// `None` if it is holding nothing.
    fn in_service_top(&self) -> Option<Vector>;

    /// Retires whichever vector is highest, which is the only thing an ordinary
    /// acknowledgement register can be told to do.
    fn end_of_interrupt(&self);

    /// Retires `vector`, whatever else the controller is holding.
    ///
    /// A vector the controller is not holding retires nothing rather than
    /// retiring something else, which is what makes this safe to issue against
    /// a debt that turns out to have been discharged already.
    fn retire(&self, vector: Vector);

    /// Decides whether the controller accepts an interrupt on `vector` at all.
    ///
    /// A cleared bit holds an arrival in the request register instead of
    /// accepting it into service, so nothing is lost and no priority class is
    /// blocked; setting the bit again delivers whatever was held.
    fn set_enabled(&self, vector: Vector, enabled: bool);

    /// Whether the controller can be reached to be asked or told anything at
    /// all.
    ///
    /// Told apart from a controller holding nothing because the two mean
    /// opposite things to a debt. A controller that answers "nothing" is one
    /// that is demonstrably not holding this debt either, which is a
    /// bookkeeping error to correct; a controller that cannot be reached
    /// has said nothing at all, and every debt must be left exactly where
    /// it was.
    fn reachable(&self) -> bool;

    /// Runs `settling` with nothing else on this processor able to reach the
    /// controller, and answers what it answered.
    ///
    /// Part of the seam rather than of the bookkeeping, because it is a
    /// property of the *controller* and not of the debts.
    fn exclusively<T>(&self, settling: impl FnOnce() -> T) -> T;
}

impl Controller for LocalApic {
    fn in_service_top(&self) -> Option<Vector> {
        LocalApic::in_service_top(*self)
    }

    fn end_of_interrupt(&self) {
        LocalApic::end_of_interrupt(*self);
    }

    /// Reaches the extended register space, which is the one thing here a
    /// controller may not have.
    ///
    /// A controller without it does nothing rather than falling back to the
    /// acknowledgement that takes no vector, because that one would retire
    /// whatever is highest and this is called precisely where that is the wrong
    /// interrupt. Nothing is lost by the silence: the arm that calls this is
    /// chosen from the same [`Extended`] that decides whether the space is
    /// there, so a controller answering `None` here is one no [`Immediate`]
    /// ledger was ever built for.
    fn retire(&self, vector: Vector) {
        if let Some(specific) = self.specific() {
            specific.end_of_interrupt(vector);
        }
    }

    /// Reaches the extended register space, for the reason and with the
    /// guarantee [`Controller::retire`] gives.
    fn set_enabled(&self, vector: Vector, enabled: bool) {
        if let Some(specific) = self.specific() {
            specific.set_enabled(vector, enabled);
        }
    }

    fn reachable(&self) -> bool {
        true
    }

    fn exclusively<T>(&self, settling: impl FnOnce() -> T) -> T {
        interrupts::without_interrupts(settling)
    }
}

/// A controller that may not have been reachable when it was asked for.
///
/// Answering as though it were holding nothing is what leaves every debt where
/// it was: a debt that could not be settled is a real in-service entry with
/// nothing left that would retire it, and inventing an acknowledgement would
/// retire whatever the controller does hold instead. Which is why the absence
/// is reported as such rather than as an empty controller — the two answers
/// lead to opposite conclusions about a debt hardware is not holding.
impl Controller for Option<LocalApic> {
    fn in_service_top(&self) -> Option<Vector> {
        self.as_ref().and_then(Controller::in_service_top)
    }

    fn end_of_interrupt(&self) {
        if let Some(local) = self {
            Controller::end_of_interrupt(local);
        }
    }

    fn retire(&self, vector: Vector) {
        if let Some(local) = self {
            Controller::retire(local, vector);
        }
    }

    fn set_enabled(&self, vector: Vector, enabled: bool) {
        if let Some(local) = self {
            Controller::set_enabled(local, vector, enabled);
        }
    }

    fn reachable(&self) -> bool {
        self.is_some()
    }

    fn exclusively<T>(&self, settling: impl FnOnce() -> T) -> T {
        interrupts::without_interrupts(settling)
    }
}

/// The debts one processor's real controller is holding for its guest.
///
/// Which arm this is was decided when the controller was built and never
/// changes, so nothing downstream asks: every operation below is the same
/// whichever machine is underneath.
#[derive(Debug)]
pub(crate) enum Ledger {
    /// A controller whose acknowledgement takes no vector, so a payment waits
    /// for its vector to reach the top and a debt nobody will discharge is
    /// never paid at all.
    Deferred(Deferred),
    /// A controller that retires a named vector and can stop one being
    /// accepted, so every debt is settled the moment there is licence to
    /// settle it.
    Immediate(Immediate),
}

impl Ledger {
    /// The ledger a controller offering `extended` needs, owing nothing.
    ///
    /// The one place the two arms are chosen between, and the condition is
    /// [`Extended::usable`] — the same one [`apic::LocalApic::specific`] uses,
    /// so the arm that retires by name is only ever built where the
    /// operations it is built around exist.
    pub(crate) const fn new(extended: Extended) -> Self {
        if extended.usable() {
            Self::Immediate(Immediate::new())
        } else {
            Self::Deferred(Deferred::new())
        }
    }

    /// Records that real hardware holds `vector` in service for this guest.
    ///
    /// Recorded before the guest is given the interrupt, so that a guest which
    /// acknowledges immediately finds the debt already there.
    ///
    /// At most one debt per vector can exist, because the real controller has
    /// one in-service bit per vector and cannot accept a second interrupt on a
    /// vector it is already holding. A repeat is therefore not a second debt.
    pub(crate) fn owe(&self, vector: Vector) {
        match self {
            Self::Deferred(deferred) => deferred.owe(vector),
            Self::Immediate(immediate) => immediate.owe(vector),
        }
    }

    /// Records that the guest has finished with `vector`, and settles whatever
    /// that makes settleable.
    ///
    /// Any debt on the vector is discharged by this, written-off ones included:
    /// the guest acknowledging a vector is the licence to retire it, whatever
    /// had been concluded about whether that acknowledgement would come.
    pub(crate) fn release(&self, vector: Vector, controller: &impl Controller) {
        match self {
            Self::Deferred(deferred) => deferred.release(vector, controller),
            Self::Immediate(immediate) => immediate.release(vector, controller),
        }
    }

    /// Records that nothing is expected to acknowledge `vector`.
    ///
    /// Reached from three places: an arrival the guest refused, which is the
    /// one exit a single interrupt makes on its own; the sweep in
    /// [`Ledger::settle`]; and a request bit deleted from a backing page
    /// whose guest has stopped existing. What becomes of the debt is the
    /// sharpest difference between the two arms — one keeps it and leaves a
    /// priority class blocked for the life of the machine, the other pays
    /// it and stops the vector arriving again until the guest is reset.
    ///
    /// Only ever called for a vector [`Ledger::owes`] answers for. Writing off
    /// a debt that does not exist would report one the machine is not
    /// holding, and on a controller that retires by name it would stop a
    /// line the guest is still using from being accepted at all.
    pub(crate) fn abandon(&self, vector: Vector, controller: &impl Controller) {
        match self {
            Self::Deferred(deferred) => deferred.abandon(vector),
            Self::Immediate(immediate) => immediate.abandon(vector, controller),
        }
    }

    /// Whether real hardware is holding `vector` in service for this guest and
    /// may still be given an acknowledgement for it.
    ///
    /// What a caller about to delete the request a debt is waiting on has to
    /// ask. A debt already written off is not one — the expectation of an
    /// acknowledgement is gone, and on the arm that can act it has been retired
    /// as well — so both arms answer about what is outstanding rather than
    /// about what they have ever held.
    pub(crate) fn owes(&self, vector: Vector) -> bool {
        match self {
            Self::Deferred(deferred) => deferred.owes(vector),
            Self::Immediate(immediate) => immediate.owes(vector),
        }
    }

    /// Settles everything real hardware is owed for a guest that will not be
    /// acknowledging any of it, and says what is left.
    ///
    /// The lifecycle boundaries: a guest that has been reset, and one that has
    /// switched its controller off. Neither will announce that it has finished
    /// with what it was given.
    #[must_use]
    pub(crate) fn settle(&self, controller: &impl Controller) -> Debts {
        match self {
            Self::Deferred(deferred) => Debts::Deferred(deferred.settle(controller)),
            Self::Immediate(immediate) => Debts::Immediate(immediate.settle(controller)),
        }
    }

    /// What real hardware is holding for this guest, and what has become of
    /// what it held before.
    ///
    /// Several independent reads and so not one instant's truth, which is all a
    /// diagnostic needs. The one caller that needs it exact takes it from
    /// inside [`Ledger::settle`], where this processor's interrupts are
    /// held off and nothing else can be mutating.
    #[must_use]
    pub(crate) fn debts(&self) -> Debts {
        match self {
            Self::Deferred(deferred) => Debts::Deferred(deferred.owing()),
            Self::Immediate(immediate) => Debts::Immediate(immediate.owing()),
        }
    }
}

/// What real hardware is holding for one guest, and what has become of what it
/// held before.
///
/// Two shapes because the two arms leave different things behind, and
/// flattening them into one would mean a count that is structurally always zero
/// on half the machines — or worse, one name for two states that are not the
/// same state. A written-off debt on a deferred ledger is a priority class lost
/// for good; on an immediate one it is one vector held quiet until the guest is
/// reset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Debts {
    /// From a controller whose acknowledgement takes no vector.
    Deferred(deferred::Owing),
    /// From a controller that retires a named vector.
    Immediate(immediate::Owing),
}

impl Debts {
    /// Whether real hardware is holding nothing at all for this guest.
    ///
    /// About what is outstanding now and not about what has happened: a
    /// controller that had a debt written off and honoured afterwards is
    /// holding nothing and says so, while its counts still say what it did.
    #[must_use]
    pub(crate) const fn is_empty(&self) -> bool {
        match self {
            Self::Deferred(owing) => owing.is_empty(),
            Self::Immediate(owing) => owing.is_empty(),
        }
    }
}

impl Display for Debts {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Deferred(owing) => owing.fmt(formatter),
            Self::Immediate(owing) => owing.fmt(formatter),
        }
    }
}

/// Writes the counts that are not zero, and says so when none of them is.
///
/// Both arms have the same shape of report — a handful of counts of which most
/// are zero on a machine that is behaving — and a line stating all of them says
/// nothing. Written once here because it is the same sentence either way, and
/// this is the record a machine with no serial port is read by afterwards.
fn describe(formatter: &mut Formatter<'_>, counts: &[(u32, &str)]) -> fmt::Result {
    let mut said = false;
    for (count, what) in counts.iter().filter(|(count, _)| *count != 0) {
        if said {
            formatter.write_str(", ")?;
        }
        write!(formatter, "{count} {what}")?;
        said = true;
    }
    if !said {
        formatter.write_str("owed nothing")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    //! What each arm does with a debt is asserted in its own module. What is
    //! here is the one decision this file makes — which arm a controller gets —
    //! and that the four words reach it, because a machine that got the other
    //! arm would settle every debt through registers it does not have.

    use alloc::{vec, vec::Vec};
    use core::cell::RefCell;

    use apic::Extended;
    use descriptors::Vector;

    use super::{Controller, Debts, Ledger};

    /// A vector of a middling priority class, which is every vector these need.
    const VECTOR: Vector = Vector::new(0x42);

    /// A controller that records which of the six operations it was asked for.
    ///
    /// Which is the whole of what these tests are about: the two arms reach for
    /// different registers, and asking the wrong ones of a real controller is
    /// either a write that does nothing or an acknowledgement that retires
    /// somebody else's interrupt.
    #[derive(Default)]
    struct Asked {
        /// The names of the operations, in the order they came.
        acts: RefCell<Vec<&'static str>>,
    }

    impl Asked {
        /// What it was asked for.
        fn acts(&self) -> Vec<&'static str> {
            self.acts.borrow().clone()
        }

        /// Records one operation.
        fn asked(&self, what: &'static str) {
            self.acts.borrow_mut().push(what);
        }
    }

    impl Controller for Asked {
        fn in_service_top(&self) -> Option<Vector> {
            self.asked("in_service_top");
            Some(VECTOR)
        }

        fn end_of_interrupt(&self) {
            self.asked("end_of_interrupt");
        }

        fn retire(&self, _vector: Vector) {
            self.asked("retire");
        }

        fn set_enabled(&self, _vector: Vector, _enabled: bool) {
            self.asked("set_enabled");
        }

        fn reachable(&self) -> bool {
            true
        }

        fn exclusively<T>(&self, settling: impl FnOnce() -> T) -> T {
            settling()
        }
    }

    #[test]
    fn a_controller_that_can_retire_by_name_gets_the_arm_that_does() {
        assert!(matches!(Ledger::new(Extended::all()), Ledger::Immediate(_)));
        assert!(matches!(
            Ledger::new(Extended::all().difference(Extended::SWITCHED_ON)),
            Ledger::Deferred(_)
        ));
        assert!(matches!(
            Ledger::new(Extended::all().difference(Extended::SPECIFIC_EOI)),
            Ledger::Deferred(_)
        ));
        assert!(matches!(
            Ledger::new(Extended::all().difference(Extended::INTERRUPT_ENABLE)),
            Ledger::Deferred(_)
        ));
        assert!(matches!(
            Ledger::new(Extended::empty()),
            Ledger::Deferred(_)
        ));
    }

    #[test]
    fn each_arm_reaches_only_for_the_registers_its_machine_has() {
        // A deferred payment asks what is highest and then acknowledges without
        // naming anything; an immediate one names its vector and asks nothing.
        let controller = Asked::default();
        let deferred = Ledger::new(Extended::empty());
        deferred.owe(VECTOR);
        deferred.release(VECTOR, &controller);
        assert_eq!(controller.acts(), ["in_service_top", "end_of_interrupt"]);

        let controller = Asked::default();
        let immediate = Ledger::new(Extended::all());
        immediate.owe(VECTOR);
        immediate.release(VECTOR, &controller);
        assert_eq!(controller.acts(), ["retire"]);
    }

    #[test]
    fn a_refusal_is_written_off_on_one_machine_and_settled_on_the_other() {
        // The sharpest difference between the arms, at the one call site that
        // exposes it: the deferred arm touches no register at all and leaves the
        // vector in service for good, while the immediate arm blocks the vector
        // and then retires it.
        let controller = Asked::default();
        let deferred = Ledger::new(Extended::empty());
        deferred.owe(VECTOR);
        deferred.abandon(VECTOR, &controller);
        assert!(controller.acts().is_empty());
        assert!(!deferred.debts().is_empty(), "hardware is still holding it");

        let controller = Asked::default();
        let immediate = Ledger::new(Extended::all());
        immediate.owe(VECTOR);
        immediate.abandon(VECTOR, &controller);
        assert_eq!(controller.acts(), ["set_enabled", "retire"]);
    }

    #[test]
    fn a_ledger_says_which_vectors_it_is_still_owed_an_acknowledgement_for() {
        // What a caller about to delete the request a debt is waiting on has to
        // ask, and the answer is about what is outstanding rather than about what
        // the controller has ever held. A debt already written off is not owed:
        // writing one off twice would count a second stranding for one interrupt
        // on the arm that keeps it, and on the arm that can act it would stop a
        // vector the guest is using from being accepted at all.
        for extended in [Extended::empty(), Extended::all()] {
            let controller = Asked::default();
            let ledger = Ledger::new(extended);
            assert!(!ledger.owes(VECTOR));

            ledger.owe(VECTOR);
            assert!(ledger.owes(VECTOR));
            ledger.abandon(VECTOR, &controller);
            assert!(!ledger.owes(VECTOR), "nothing more is expected for it");

            // And a debt the guest discharged is not owed either, which is the
            // ordinary way one ends.
            ledger.owe(VECTOR);
            ledger.release(VECTOR, &controller);
            assert!(!ledger.owes(VECTOR));
        }
    }

    #[test]
    fn what_is_reported_says_which_machine_reported_it() {
        // The two are not one shape, because a written-off debt means opposite
        // things on them: a priority class lost for good, or one vector held
        // quiet until the guest is reset.
        assert!(matches!(
            Ledger::new(Extended::empty()).debts(),
            Debts::Deferred(_)
        ));
        assert!(matches!(
            Ledger::new(Extended::all()).debts(),
            Debts::Immediate(_)
        ));
        assert!(Ledger::new(Extended::all()).debts().is_empty());
        assert_eq!(
            vec![Ledger::new(Extended::empty()).settle(&Asked::default())],
            vec![Ledger::new(Extended::empty()).debts()],
            "a settlement of nothing leaves nothing"
        );
    }
}
