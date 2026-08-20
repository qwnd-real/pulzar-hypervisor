//! Settling debts on a controller that can retire a vector by name.
//!
//! Two registers of AMD's extended space are the whole difference. One
//! acknowledgement names the vector it retires; one bank of bits decides, per
//! vector, whether this controller accepts an interrupt on it at all. What they
//! remove between them is every reason [`super::deferred`] is complicated.
//!
//! # Two states, because a debt now ends two ways
//!
//! A debt is *owed* from the moment the interrupt arrives, and it is settled
//! the moment there is licence to settle it — which is the guest acknowledging
//! its own controller, and nothing else. There is no state for a payment
//! waiting its turn, because none waits: an acknowledgement that names its
//! vector retires that vector whatever else the controller holds, so the moment
//! the guest gives its own is the moment the real one is issued.
//!
//! Nor is there a state for a debt hardware turns out not to hold. Naming a
//! vector the controller is not holding retires nothing rather than retiring
//! something else, so there is nothing to detect and nothing to count: the
//! bookkeeping is corrected by the same write that would have paid.
//!
//! The second state is *blocked*, and it is what the extended space adds rather
//! than what it removes. A debt nothing will ever acknowledge — a guest that
//! refused the interrupt, or was reset while holding it — is retired like any
//! other, and then the vector it arrived on is stopped from being accepted
//! again.
//!
//! # Why the block has to come first
//!
//! Retiring a level-triggered vector also broadcasts an end-of-interrupt to the
//! I/O controllers, which clears the remote request bit of whichever one sent
//! it. That is the point — it is what lets the line be re-armed — but the line
//! is still asserted by a device nobody has serviced, so the interrupt is sent
//! again at once. Blocking the vector first is what stops that arrival being
//! accepted into a guest that has already refused it; blocking it afterwards
//! leaves a window the re-arrival lands in, and what it lands in is this same
//! function, one frame down.
//!
//! Where it settles is quiet and stable. The re-sent interrupt is latched in
//! the request register and not accepted, so no acknowledgement is owed for it
//! and the I/O controller's remote request bit stays set — which is exactly
//! what stops that controller sending a third. One vector held, one line
//! waiting, and nothing spinning.
//!
//! # What a blocked vector costs, and what it does not
//!
//! That vector, on that processor, until the guest is reset. A request held in
//! the request register is not held in service, so the controller's processor
//! priority is untouched and every other vector — including every other vector
//! of the same interrupt-priority class — goes on arriving. Which is the whole
//! of what this arm is for: [`super::deferred`] pays for the same refusal with
//! the vector's entire priority class and everything below it, for the life of
//! the machine.
//!
//! Two things follow that are worth stating because the other arm's reasoning
//! says the opposite. More than one debt of a class can be blocked at once,
//! since nothing is left in service to stop the class being delivered. And a
//! blocked vector is recoverable: the guest acknowledging it unblocks it, and
//! so does every lifecycle boundary.
//!
//! # Nothing is left behind
//!
//! [`Immediate::settle`] retires everything owed and unblocks everything
//! blocked, so a guest that is reset hands the next one a controller holding
//! nothing at all. Unblocking is not a hope that the line has gone quiet: a
//! line still asserted is delivered once more, refused by whatever the new
//! guest is, and blocked again by the ordinary path — one interrupt per
//! boundary, and the alternative is a vector the next guest could never use.
//!
//! # Where an exclusion is still needed, and where it is not
//!
//! Only [`Immediate::settle`] holds this processor's interrupts off, and only
//! because its sweep must not miss a debt an arrival records underneath it:
//! such a debt would be one this reports as settled and nothing ever retires.
//!
//! Nothing else needs it. No operation here reads a hardware register and then
//! writes one that depends on what it said, which is what the other arm's
//! masking exists for. And the one interleaving left — this processor's own
//! interrupt handler recording a debt while one of these runs — cannot touch
//! the same vector: the controller will not deliver a vector it is already
//! holding in service, and it will not deliver one whose bit this arm has
//! cleared.

use core::{
    fmt::{self, Display, Formatter},
    sync::atomic::{AtomicU32, Ordering},
};

use descriptors::Vector;

use crate::{lifecycle::ledger::Controller, registers::bitmap::Bitmap};

/// The debts one processor's real controller is holding, on a machine where an
/// acknowledgement names the vector it retires.
#[derive(Debug)]
pub(crate) struct Immediate {
    /// What real hardware holds and the guest may still acknowledge.
    owed: Bitmap,
    /// Which vectors this controller has been told not to accept, because
    /// nothing was left to service what arrived on them.
    ///
    /// Every one of them has already been retired, so this is not a debt: it is
    /// one line held quiet, and the record of which bits have to be given back
    /// at the next lifecycle boundary.
    blocked: Bitmap,
    /// How many vectors have been blocked since the controller was built, which
    /// the map above cannot say: a vector blocked, unblocked and blocked again
    /// is one bit and two events.
    ///
    /// Kept for the reason the other arm keeps its own count — this hypervisor
    /// runs on machines with no serial port, and the controllers outlive every
    /// guest — but it counts something recoverable rather than something lost.
    blockings: AtomicU32,
}

impl Immediate {
    /// Nothing owed and nothing blocked, which is what a controller is built
    /// with.
    pub(crate) const fn new() -> Self {
        Self {
            owed: Bitmap::new(),
            blocked: Bitmap::new(),
            blockings: AtomicU32::new(0),
        }
    }

    /// Records that real hardware holds `vector` in service for this guest.
    ///
    /// One store, and the caller is an interrupt handler on the processor whose
    /// hardware is owed.
    pub(crate) fn owe(&self, vector: Vector) {
        self.owed.set(vector);
    }

    /// Records that the guest has finished with `vector`, and settles it.
    ///
    /// Two things a guest's acknowledgement can mean here, and they are
    /// exclusive because the states are: a debt real hardware is holding,
    /// which is retired at once by name; or a vector this arm had blocked,
    /// which the guest is evidently servicing again — so the block is given
    /// back and the line, if it is still asserted, is delivered to a guest
    /// that will now take it.
    ///
    /// The bit is cleared only once hardware has been reached. A controller
    /// that could not be reached has done nothing, and a debt whose bit was
    /// cleared against a write that never happened is a real in-service
    /// entry with nothing left that names it.
    pub(crate) fn release(&self, vector: Vector, controller: &impl Controller) {
        if self.owed.get(vector) {
            if controller.reachable() {
                controller.retire(vector);
                self.owed.clear(vector);
            }
        } else if self.blocked.get(vector) && controller.reachable() {
            controller.set_enabled(vector, true);
            self.blocked.clear(vector);
        }
    }

    /// Retires `vector` and stops it being accepted, for a debt nothing is
    /// expected to acknowledge.
    ///
    /// The order is load-bearing and is argued in this module's documentation:
    /// the block goes on before the acknowledgement, because the
    /// acknowledgement is what makes the still-asserted line arrive again.
    ///
    /// A controller that cannot be reached leaves the debt exactly where it
    /// was, owed. Retiring without blocking is the one thing that must not
    /// happen, so a half of this is not attempted.
    pub(crate) fn abandon(&self, vector: Vector, controller: &impl Controller) {
        if !controller.reachable() {
            return;
        }
        controller.set_enabled(vector, false);
        self.blocked.set(vector);
        self.blockings.fetch_add(1, Ordering::Relaxed);
        controller.retire(vector);
        self.owed.clear(vector);
    }

    /// Settles everything for a guest that will not be acknowledging any of it,
    /// gives back every block it left behind, and says what is left.
    ///
    /// The answer is exact rather than a snapshot, because the whole of it runs
    /// with this processor's interrupts held off: a debt recorded by an arrival
    /// landing underneath the sweep would otherwise be one this reports as
    /// settled and nothing ever retires.
    ///
    /// A controller that cannot be reached leaves every debt and every block
    /// exactly where they were, and the answer says so — which is the whole of
    /// what a caller can do with it, since the alternative is bookkeeping
    /// cleared against writes that never happened.
    ///
    /// Both sweeps are bounded by the number of vectors, because each pass that
    /// continues clears one bit and nothing inside either can set one.
    #[must_use]
    pub(crate) fn settle(&self, controller: &impl Controller) -> Owing {
        controller.exclusively(|| {
            if !controller.reachable() {
                return self.owing();
            }
            for _ in 0..Bitmap::CAPACITY {
                let Some(vector) = self.owed.highest() else {
                    break;
                };
                self.abandon(vector, controller);
            }
            for _ in 0..Bitmap::CAPACITY {
                let Some(vector) = self.blocked.highest() else {
                    break;
                };
                // Given back to whatever guest comes next, which is the one that
                // will have to decide about the line rather than inheriting a
                // vector it could never receive on.
                controller.set_enabled(vector, true);
                self.blocked.clear(vector);
            }
            self.owing()
        })
    }

    /// What real hardware is holding for this guest, and what it has been told
    /// not to accept.
    #[must_use]
    pub(crate) fn owing(&self) -> Owing {
        Owing {
            owed: self.owed.count(),
            blocked: self.blocked.count(),
            blockings: self.blockings.load(Ordering::Relaxed),
        }
    }

    /// Whether real hardware is holding `vector` and an acknowledgement for it
    /// is still expected.
    ///
    /// A blocked vector is not one: it has already been retired, so there is
    /// nothing left to settle and nothing a deleted request could strand.
    pub(crate) fn owes(&self, vector: Vector) -> bool {
        self.owed.get(vector)
    }
}

/// What real hardware is holding for one guest on this kind of controller, and
/// what it has been told not to accept.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Owing {
    /// How many vectors the guest may still acknowledge.
    pub(super) owed: u32,
    /// How many vectors this controller has been told not to accept.
    pub(super) blocked: u32,
    /// How many have been blocked since the controller was built.
    pub(super) blockings: u32,
}

impl Owing {
    /// Whether real hardware is holding nothing and refusing nothing for this
    /// guest.
    #[must_use]
    pub(crate) const fn is_empty(&self) -> bool {
        self.owed == 0 && self.blocked == 0
    }
}

impl Display for Owing {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        super::describe(
            formatter,
            &[
                (self.owed, "owed"),
                (self.blocked, "blocked"),
                (self.blockings, "blocked in all"),
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    //! The controller is a parameter, so a test is a controller — and this one
    //! records the order it was told things in, because one of the two
    //! operations here is only correct before the other.
    //!
    //! The double refuses the two operations this arm must never reach for. An
    //! acknowledgement that takes no vector would retire whichever interrupt is
    //! highest, which is the wrong one in every place this arm acts.

    use alloc::{format, vec::Vec};
    use core::cell::RefCell;

    use descriptors::Vector;

    use super::{Controller, Immediate};

    /// Something a controller was told to do.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Act {
        /// Retire one named vector.
        Retired(Vector),
        /// Accept this vector again.
        Enabled(Vector),
        /// Stop accepting this vector.
        Disabled(Vector),
    }

    /// A controller with the extended space: it retires the vector it is given
    /// and holds a bit per vector deciding what it accepts.
    struct Hardware {
        /// What it is holding in service. Unordered as far as this arm is
        /// concerned, which is the point: nothing here depends on which is
        /// highest.
        held: RefCell<Vec<Vector>>,
        /// Which vectors it has been told not to accept.
        refusing: RefCell<Vec<Vector>>,
        /// Everything it was told, in the order it was told.
        acts: RefCell<Vec<Act>>,
    }

    impl Hardware {
        /// A controller holding these vectors and refusing none.
        fn new(held: &[Vector]) -> Self {
            Self {
                held: RefCell::new(held.to_vec()),
                refusing: RefCell::new(Vec::new()),
                acts: RefCell::new(Vec::new()),
            }
        }

        /// What it is still holding in service.
        fn held(&self) -> Vec<Vector> {
            self.held.borrow().clone()
        }

        /// Which vectors it is refusing.
        fn refusing(&self) -> Vec<Vector> {
            self.refusing.borrow().clone()
        }

        /// Everything it was told, in order.
        fn acts(&self) -> Vec<Act> {
            self.acts.borrow().clone()
        }
    }

    impl Controller for Hardware {
        fn in_service_top(&self) -> Option<Vector> {
            unreachable!("this arm never asks what is highest; it names its vector");
        }

        fn end_of_interrupt(&self) {
            unreachable!("an acknowledgement taking no vector would retire the wrong one");
        }

        fn retire(&self, vector: Vector) {
            self.held.borrow_mut().retain(|held| *held != vector);
            self.acts.borrow_mut().push(Act::Retired(vector));
        }

        fn set_enabled(&self, vector: Vector, enabled: bool) {
            let mut refusing = self.refusing.borrow_mut();
            if enabled {
                refusing.retain(|refused| *refused != vector);
            } else if !refusing.contains(&vector) {
                refusing.push(vector);
            }
            self.acts.borrow_mut().push(if enabled {
                Act::Enabled(vector)
            } else {
                Act::Disabled(vector)
            });
        }

        fn reachable(&self) -> bool {
            true
        }

        /// Nothing else reaches this controller: it is one test's own, and a
        /// test process may not execute the instruction that holds a
        /// real processor's interrupts off.
        fn exclusively<T>(&self, settling: impl FnOnce() -> T) -> T {
            settling()
        }
    }

    /// A controller that cannot be reached, which is what a failure to find
    /// this processor's own leaves a caller holding.
    struct Unreachable;

    impl Controller for Unreachable {
        fn in_service_top(&self) -> Option<Vector> {
            unreachable!("this arm never asks what is highest; it names its vector");
        }

        fn end_of_interrupt(&self) {
            unreachable!("an acknowledgement taking no vector would retire the wrong one");
        }

        fn retire(&self, _vector: Vector) {
            unreachable!("a controller that cannot be reached is never written");
        }

        fn set_enabled(&self, _vector: Vector, _enabled: bool) {
            unreachable!("a controller that cannot be reached is never written");
        }

        fn reachable(&self) -> bool {
            false
        }

        fn exclusively<T>(&self, settling: impl FnOnce() -> T) -> T {
            settling()
        }
    }

    /// Three vectors of three priority classes, in ascending order.
    const LOW: Vector = Vector::new(0x31);
    const HIGH: Vector = Vector::new(0x52);
    const HIGHER: Vector = Vector::new(0x84);

    #[test]
    fn a_vector_that_was_never_owed_is_never_named() {
        // An edge-triggered interrupt was acknowledged when it arrived, so there
        // is no debt — and this arm must not issue an acknowledgement anyway,
        // because a vector the controller is holding for somebody else would then
        // be retired by name.
        let ledger = Immediate::new();
        let controller = Hardware::new(&[LOW]);

        ledger.release(LOW, &controller);
        assert!(controller.acts().is_empty());
        assert_eq!(controller.held(), [LOW]);
        assert!(ledger.owing().is_empty());
    }

    #[test]
    fn a_debt_is_retired_by_name_while_a_higher_vector_stays_in_service() {
        // The whole difference from the other arm. There the guest acknowledging
        // the lower vector could not be honoured at all until the higher one was
        // retired by whoever owned it; here it is honoured at once, and the higher
        // one is left exactly where it was.
        let ledger = Immediate::new();
        let controller = Hardware::new(&[LOW, HIGH]);
        ledger.owe(LOW);

        ledger.release(LOW, &controller);
        assert_eq!(controller.acts(), [Act::Retired(LOW)]);
        assert_eq!(controller.held(), [HIGH], "HIGH is somebody else's");
        assert!(
            ledger.owing().is_empty(),
            "nothing is left waiting its turn"
        );
    }

    #[test]
    fn a_debt_is_retired_once_however_often_the_guest_acknowledges() {
        // Every write past the first has no debt to find. A second acknowledgement
        // would name a vector the controller is no longer holding, which retires
        // nothing — but it would also be a write this arm cannot justify.
        let ledger = Immediate::new();
        let controller = Hardware::new(&[LOW]);
        ledger.owe(LOW);

        for _ in 0..8 {
            ledger.release(LOW, &controller);
        }
        assert_eq!(controller.acts(), [Act::Retired(LOW)]);
        assert!(ledger.owing().is_empty());
    }

    #[test]
    fn a_repeated_arrival_is_not_a_second_debt() {
        // One in-service bit per vector on real hardware, so one debt and one
        // acknowledgement.
        let ledger = Immediate::new();
        let controller = Hardware::new(&[LOW]);
        ledger.owe(LOW);
        ledger.owe(LOW);

        ledger.release(LOW, &controller);
        assert_eq!(controller.acts(), [Act::Retired(LOW)]);
        assert!(ledger.owing().is_empty());
    }

    #[test]
    fn a_debt_hardware_is_not_holding_is_corrected_by_the_write_that_would_have_paid() {
        // The other arm has to detect this and count it, because there a write
        // against the wrong state retires an interrupt belonging to somebody else.
        // Here the write names its vector, so a vector the controller is not
        // holding retires nothing and the bookkeeping is simply right afterwards.
        let ledger = Immediate::new();
        let controller = Hardware::new(&[HIGH]);
        ledger.owe(HIGHER);

        ledger.release(HIGHER, &controller);
        assert_eq!(controller.acts(), [Act::Retired(HIGHER)]);
        assert_eq!(controller.held(), [HIGH], "nothing else was retired");
        assert!(ledger.owing().is_empty());
    }

    #[test]
    fn a_refused_arrival_is_blocked_before_it_is_retired() {
        // The one ordering in this module that is not a preference. Retiring first
        // broadcasts an end-of-interrupt to the I/O controllers, and the line is
        // still asserted — so the interrupt arrives again in the window before the
        // block goes on, into a guest that has already refused it.
        let ledger = Immediate::new();
        let controller = Hardware::new(&[LOW]);
        ledger.owe(LOW);

        ledger.abandon(LOW, &controller);
        assert_eq!(
            controller.acts(),
            [Act::Disabled(LOW), Act::Retired(LOW)],
            "the block has to be on before the acknowledgement that re-arms the line"
        );
        assert!(
            controller.held().is_empty(),
            "nothing is left in service, so no priority class is blocked"
        );
        let owing = ledger.owing();
        assert_eq!((owing.owed, owing.blocked, owing.blockings), (0, 1, 1));
        assert!(!owing.is_empty(), "the block is still outstanding");
    }

    #[test]
    fn more_than_one_vector_of_a_class_can_be_blocked_at_once() {
        // True here and false on the other arm, and the reason is the assertion
        // above: nothing is left in service, so the class goes on being delivered
        // and a second vector of it can arrive and be refused in turn.
        let ledger = Immediate::new();
        let controller = Hardware::new(&[Vector::new(0x31), Vector::new(0x3F)]);
        for vector in [Vector::new(0x31), Vector::new(0x3F)] {
            ledger.owe(vector);
            ledger.abandon(vector, &controller);
        }
        assert_eq!(
            controller.refusing(),
            [Vector::new(0x31), Vector::new(0x3F)]
        );
        assert!(controller.held().is_empty());
        assert_eq!(ledger.owing().blocked, 2);
    }

    #[test]
    fn the_guest_acknowledging_a_blocked_vector_gives_the_block_back() {
        // A guest acknowledging a vector is servicing it, whatever it was doing
        // when the arrival was refused — so the line is allowed to reach it again.
        // The other arm can only honour such an acknowledgement by retiring
        // something; here there is nothing left in service and the recovery is the
        // block coming off.
        let ledger = Immediate::new();
        let controller = Hardware::new(&[LOW]);
        ledger.owe(LOW);
        ledger.abandon(LOW, &controller);

        ledger.release(LOW, &controller);
        assert_eq!(
            controller.acts(),
            [Act::Disabled(LOW), Act::Retired(LOW), Act::Enabled(LOW)]
        );
        assert!(controller.refusing().is_empty());
        assert!(ledger.owing().is_empty());
        assert_eq!(
            ledger.owing().blockings,
            1,
            "the block happened, and is counted whether or not it was given back"
        );
    }

    #[test]
    fn a_reset_leaves_the_controller_holding_and_refusing_nothing() {
        // What the other arm cannot do. There a reset writes off every debt and the
        // real controller goes on holding those vectors for the life of the
        // machine; here everything owed is retired by name and every block is
        // given back, so the next guest finds a controller with nothing on it.
        let ledger = Immediate::new();
        let controller = Hardware::new(&[LOW, HIGH, HIGHER]);
        ledger.owe(LOW);
        ledger.owe(HIGH);
        ledger.abandon(HIGH, &controller);

        let owing = ledger.settle(&controller);
        assert_eq!(
            controller.held(),
            [HIGHER],
            "everything this guest was given is retired, and nothing else is"
        );
        assert!(
            controller.refusing().is_empty(),
            "every block is given back"
        );
        assert!(owing.is_empty());
        assert_eq!(owing.blockings, 2, "both refusals are still on the record");
    }

    #[test]
    fn a_settlement_answers_about_a_debt_recorded_underneath_it() {
        // Masking makes this impossible on real hardware, which is why the answer
        // is what is asserted rather than the interleaving: a debt the sweep did
        // not see would be one this reports as settled and nothing ever retires.
        let ledger = Immediate::new();
        let controller = Hardware::new(&[LOW]);

        let owing = ledger.settle(&controller);
        assert!(owing.is_empty());
        assert_eq!(
            controller.held(),
            [LOW],
            "nothing was owed, so nothing went"
        );

        ledger.owe(LOW);
        assert_eq!(ledger.owing().owed, 1);
        let owing = ledger.settle(&controller);
        assert!(owing.is_empty());
        assert!(controller.held().is_empty());
    }

    #[test]
    fn a_controller_that_cannot_be_reached_leaves_every_debt_where_it_was() {
        // A debt whose bit was cleared against a write that never happened is a
        // real in-service entry with nothing left that names it. So nothing is
        // cleared until hardware has been reached, and a refusal that could not be
        // blocked is not retired either — retiring without the block is the one
        // thing that must not happen.
        let ledger = Immediate::new();
        let controller = Hardware::new(&[LOW]);
        ledger.owe(LOW);

        ledger.release(LOW, &Unreachable);
        ledger.abandon(LOW, &Unreachable);
        let owing = ledger.owing();
        assert_eq!((owing.owed, owing.blocked, owing.blockings), (1, 0, 0));

        // Reachable again, and the debt is settled rather than lost.
        ledger.release(LOW, &controller);
        assert_eq!(controller.acts(), [Act::Retired(LOW)]);
        assert!(ledger.owing().is_empty());
    }

    #[test]
    fn a_settlement_against_an_unreachable_controller_changes_nothing() {
        // Every block and every debt is left where it was, because clearing a bit
        // against a write that never happened is what leaves real hardware holding
        // a vector with nothing left that names it.
        let ledger = Immediate::new();
        let controller = Hardware::new(&[LOW, HIGH]);
        ledger.owe(LOW);
        ledger.owe(HIGH);
        ledger.abandon(HIGH, &controller);

        let owing = ledger.settle(&Unreachable);
        assert_eq!((owing.owed, owing.blocked), (1, 1));
        assert!(!owing.is_empty());

        // Reachable again, and the same settlement finishes the job.
        let owing = ledger.settle(&controller);
        assert!(owing.is_empty());
        assert!(controller.held().is_empty() && controller.refusing().is_empty());
    }

    #[test]
    fn debts_report_only_what_is_not_zero() {
        // The record a machine with no serial port is read by, so what it says has
        // to be legible: three counts of which two are usually zero.
        let ledger = Immediate::new();
        let controller = Hardware::new(&[LOW]);
        assert_eq!(format!("{}", ledger.owing()), "owed nothing");

        ledger.owe(LOW);
        assert_eq!(format!("{}", ledger.owing()), "1 owed");

        ledger.abandon(LOW, &controller);
        assert_eq!(format!("{}", ledger.owing()), "1 blocked, 1 blocked in all");
    }
}
