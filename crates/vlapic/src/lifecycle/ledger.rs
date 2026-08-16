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
//! # The acknowledgement register carries no vector
//!
//! Which is the whole difficulty. Writing it retires whichever vector the real
//! controller currently holds highest, not one named by the writer — so a debt
//! may only be paid at the moment its own vector *is* that highest one. Paying
//! at any other moment retires an interrupt belonging to somebody else, and the
//! interrupt it should have retired stays in service forever, blocking
//! everything of its priority or lower on that processor for the life of the
//! machine.
//!
//! So nothing here ever writes the acknowledgement register without first
//! asking the controller what it is holding, and comparing.
//!
//! # Three states, because a debt ends three ways
//!
//! A debt is *owed* from the moment the interrupt arrives, and becomes
//! *released* when the guest acknowledges its own controller. Only a released
//! debt may be paid, and the distinction is not bookkeeping: paying a debt the
//! guest is still servicing would let the line re-assert underneath a driver
//! that is halfway through quieting it, which is the exact failure withholding
//! the acknowledgement exists to prevent.
//!
//! The two are needed together because the guest does not acknowledge in the
//! order hardware retires. A guest servicing a low vector while a higher one is
//! merely requested acknowledges the low one first; its debt is then released
//! but not payable, because the higher vector is what the real controller holds
//! at the top. It becomes payable when the higher one is retired, and the next
//! [`Ledger::release`] is what notices.
//!
//! The third state is what the other two cannot express: *abandoned*, for a
//! debt whose acknowledgement is not coming. A guest that is reset stops
//! existing; one that switches its controller off stops taking interrupts
//! through it; one that refuses an interrupt never sees the vector at all. In
//! each of those the one event that would have discharged the debt cannot
//! happen, and this says so rather than guessing.
//!
//! An abandoned debt is not paid — see "Limitations", which is where that
//! decision is argued. It is also not forgotten: an acknowledgement that
//! arrives after all is honoured, because a guest acknowledging a vector is
//! itself the licence the architecture gives for retiring it, whatever had been
//! concluded about whether one would come.
//!
//! # What makes a payment one step
//!
//! A payment is a read of the real in-service register followed by a write that
//! depends on what it said, and the two have to refer to the same hardware
//! state. They do, because the controller moves a vector into that register
//! only when this processor accepts an interrupt, and it cannot accept one
//! while this processor's interrupts are masked. No other processor can write
//! that register, and a non-maskable interrupt sets nothing in it. So
//! everything here that pays runs inside [`InService::exclusively`], where the
//! register is frozen for as long as the payment takes.
//!
//! The same window is what makes the bookkeeping safe. Each transition below
//! moves a vector between two of the maps, which is two separate atomics, and
//! this processor's own interrupt handler is the only other thing that touches
//! them. Masking is what keeps a handler from interleaving with half of a move,
//! and it is why nothing here depends on which memory ordering another module
//! chose for a bitmap.
//!
//! # Every way a debt ends
//!
//! | how it ends | who says so | what becomes of it |
//! | --- | --- | --- |
//! | the guest acknowledges it | the guest's own acknowledgement register | released, and paid once its vector is what real hardware holds highest |
//! | the guest is reset | the processor applying an `INIT` to itself | abandoned |
//! | the guest switches its controller off | a write to the base register, or to the enable bit of the spurious-vector register | abandoned |
//! | the guest refuses the interrupt | its controller, at the moment of arrival | abandoned |
//! | real hardware turns out not to hold it | the in-service register, when the payment comes due | dropped, and counted |
//!
//! There is no sixth. A guest that simply never takes the vector — one holding
//! its task priority high, or with no driver for it — leaves the debt owed,
//! which is the honest answer: that acknowledgement is still possible.
//!
//! # Limitations
//!
//! An abandoned debt is never paid, so its vector stays in service on the real
//! controller for the life of the machine, and that controller refuses
//! everything of the vector's own interrupt-priority class or lower for as
//! long. How much that costs is the vector's own class: a debt near the bottom
//! of the range costs the guest the classes below it, and one just under the
//! class the host keeps for itself costs that processor every interrupt the
//! guest has. Nothing is ever abandoned *in* the host's class, because an
//! acknowledgement is never withheld there at all — and at most one debt can
//! exist per class, because the real controller will not deliver a second
//! vector of a class it is already holding one of.
//!
//! What a guest observes is one processor that stops taking interrupts at or
//! below one priority. The device whose line was abandoned never fires again,
//! nor does anything the guest points at that class or lower on that processor;
//! everything above it keeps working, and so does every other processor. There
//! is no recovery from inside the guest: the class the debt holds is exactly
//! the class the vector would have to arrive in to be acknowledged again.
//!
//! Paying instead is worse, which is why this is the choice. The real
//! acknowledgement of a level-triggered interrupt also clears the remote
//! in-service state of the I/O controller that sent it — hardware this
//! hypervisor passes through and cannot quiet — so the still-asserted line is
//! delivered again at once, into a guest that has already refused it or been
//! reset. That is a processor making no progress at all, rather than one that
//! runs with a priority missing.
//!
//! Removing the limitation needs awareness of the passed-through I/O
//! controller: with the redirection entry a vector came from in reach, an
//! abandoned debt could be paid safely by masking that entry first and leaving
//! it masked until the guest programs it again.

use core::{
    fmt::{self, Display, Formatter},
    sync::atomic::{AtomicU32, Ordering},
};

use apic::LocalApic;
use descriptors::Vector;
use x86_64::instructions::interrupts;

use crate::registers::bitmap::Bitmap;

/// What the ledger needs of the controller holding its debts.
///
/// Taken as a parameter rather than reached for through [`apic::local`],
/// because *which* controller a debt is paid through is the one thing that must
/// not be assumed — the acknowledgement register carries no vector, so paying
/// through the wrong controller retires an interrupt belonging to somebody else
/// and strands the one that should have been retired for the life of the
/// machine.
///
/// Passing it in is also what makes every interleaving in this module a test:
/// nothing here has to be on a processor that has a controller at all.
pub(crate) trait InService {
    /// The highest-priority vector the controller is holding in service, or
    /// `None` if it is holding nothing.
    fn in_service_top(&self) -> Option<Vector>;

    /// Whether the controller can be reached to be asked at all.
    ///
    /// Told apart from a controller holding nothing because the two mean
    /// opposite things to a debt. A controller that answers "nothing" is one
    /// that is demonstrably not holding this debt either, which is a
    /// bookkeeping error to correct; a controller that cannot be asked has said
    /// nothing at all, and every debt must be left exactly where it was.
    fn reachable(&self) -> bool;

    /// Retires whichever vector is highest, which is the only thing the
    /// acknowledgement register can be told to do.
    fn end_of_interrupt(&self);

    /// Runs `paying` with nothing else on this processor able to reach the
    /// controller, and answers what it answered.
    ///
    /// Part of the seam rather than of the bookkeeping, because it is a
    /// property of the *controller* and not of the debts: every payment is
    /// a read of a real register followed by a write that depends on what
    /// it said, and a handler interposing between the two would be answered
    /// about one vector and paid for another.
    fn exclusively<T>(&self, paying: impl FnOnce() -> T) -> T;
}

impl InService for LocalApic {
    fn in_service_top(&self) -> Option<Vector> {
        LocalApic::in_service_top(*self)
    }

    fn reachable(&self) -> bool {
        true
    }

    fn end_of_interrupt(&self) {
        LocalApic::end_of_interrupt(*self);
    }

    fn exclusively<T>(&self, paying: impl FnOnce() -> T) -> T {
        interrupts::without_interrupts(paying)
    }
}

/// A controller that may not have been reachable when it was asked for.
///
/// Answering as though it were holding nothing is what leaves every debt where
/// it was: a debt that could not be paid is a real in-service entry with
/// nothing left that would retire it, and inventing an acknowledgement would
/// retire whatever the controller does hold instead. Which is why the absence
/// is reported as such rather than as an empty controller — the two answers
/// lead to opposite conclusions about a debt hardware is not holding.
impl InService for Option<LocalApic> {
    fn in_service_top(&self) -> Option<Vector> {
        self.as_ref().and_then(InService::in_service_top)
    }

    fn reachable(&self) -> bool {
        self.is_some()
    }

    fn end_of_interrupt(&self) {
        if let Some(local) = self {
            InService::end_of_interrupt(local);
        }
    }

    fn exclusively<T>(&self, paying: impl FnOnce() -> T) -> T {
        interrupts::without_interrupts(paying)
    }
}

/// The debts one processor's real controller is holding for its guest.
#[derive(Debug)]
pub(crate) struct Ledger {
    /// What real hardware holds and the guest may still acknowledge.
    owed: Bitmap,
    /// What the guest has acknowledged, waiting for its turn at the top.
    released: Bitmap,
    /// What real hardware holds and nothing is expected to acknowledge.
    abandoned: Bitmap,
    /// How many debts have been abandoned on this controller, which the map
    /// above cannot say: a vector abandoned, honoured after all and abandoned
    /// again is one bit and two events.
    ///
    /// The record a strand leaves behind, and the reason it is a count in the
    /// controller rather than a log line: this hypervisor runs on machines with
    /// no serial port, where an abandoned debt is otherwise invisible, and the
    /// controllers outlive every guest.
    strandings: AtomicU32,
    /// How many debts came due against a controller that turned out not to be
    /// holding them.
    ///
    /// Zero on a machine that is behaving. Anything else means an
    /// acknowledgement this crate did not issue retired a vector it was
    /// withholding, and the count is what tells that apart from a debt the
    /// guest is merely slow to discharge.
    phantoms: AtomicU32,
}

impl Ledger {
    /// Nothing owed, which is what a controller is built with.
    pub(crate) const fn new() -> Self {
        Self {
            owed: Bitmap::new(),
            released: Bitmap::new(),
            abandoned: Bitmap::new(),
            strandings: AtomicU32::new(0),
            phantoms: AtomicU32::new(0),
        }
    }

    /// Records that real hardware holds `vector` in service for this guest.
    ///
    /// Recorded before the guest is given the interrupt, so that a guest which
    /// acknowledges immediately finds the debt already there.
    ///
    /// At most one debt per vector can exist, because the real controller has
    /// one in-service bit per vector and cannot accept a second interrupt
    /// on a vector it is already holding. A repeat is therefore not a
    /// second debt and is not counted as one.
    ///
    /// One store, so there is nothing for an exclusion to protect: no vector is
    /// being moved between two maps, and the caller is an interrupt handler on
    /// the processor whose hardware is owed.
    pub(crate) fn owe(&self, vector: Vector) {
        self.owed.set(vector);
    }

    /// Records that the guest has finished with `vector`, and pays whatever
    /// that makes payable.
    ///
    /// Any debt on the vector is discharged by this, abandoned ones included:
    /// the guest acknowledging a vector is the licence to retire it, and a debt
    /// written off because no acknowledgement was expected is still owed the
    /// payment if one arrives.
    ///
    /// Every acknowledgement retries what an earlier one had to defer, which is
    /// what keeps a debt waiting on a vector that has since come to the top
    /// from waiting forever. The retry costs the eight loads below on a
    /// path with nothing to pay, which is the overwhelmingly common one: a
    /// guest acknowledging an edge-triggered interrupt that owed nothing at
    /// all.
    pub(crate) fn release(&self, vector: Vector, controller: &impl InService) {
        if !self.holds(vector) && self.released.is_empty() {
            return;
        }
        controller.exclusively(|| {
            let owed = self.owed.clear(vector);
            let abandoned = self.abandoned.clear(vector);
            if owed || abandoned {
                self.released.set(vector);
            }
            self.pay(controller);
        });
    }

    /// Records that nothing is expected to acknowledge `vector`.
    ///
    /// Reached from two places: an arrival the guest refused, which is the one
    /// exit a single interrupt makes on its own, and the sweep in
    /// [`Ledger::settle`]. The debt is kept rather than paid — see this
    /// module's limitations — and the real controller goes on holding the
    /// vector.
    ///
    /// Needs no exclusion of its own. The two operations that move a vector out
    /// of the owed map hold this processor's interrupts off for as long as they
    /// take, and the refused arrival is an interrupt handler, which is
    /// therefore never inside either of them.
    pub(crate) fn abandon(&self, vector: Vector) {
        self.owed.clear(vector);
        self.abandoned.set(vector);
        self.strandings.fetch_add(1, Ordering::Relaxed);
    }

    /// Settles everything real hardware is owed for a guest that will not be
    /// acknowledging any of it, and says what is left.
    ///
    /// The two lifecycle boundaries: a guest that has been reset, and one that
    /// has switched its controller off. Neither will announce that it has
    /// finished with what it was given, so every debt it still owed is
    /// abandoned. What it had already acknowledged is *not* — that payment is
    /// licensed and still due, and is made here if its vector has come to the
    /// top.
    ///
    /// The whole of it runs with this processor's interrupts held off, which is
    /// what makes the answer exact: a debt recorded by an arrival landing
    /// underneath the sweep would otherwise be one this reports as settled and
    /// nothing ever pays.
    #[must_use]
    pub(crate) fn settle(&self, controller: &impl InService) -> Debts {
        controller.exclusively(|| {
            // Bounded like the payment below, and for the same reason: each pass
            // that continues clears one bit and nothing inside can set one.
            for _ in 0..Bitmap::CAPACITY {
                let Some(vector) = self.owed.highest() else {
                    break;
                };
                self.abandon(vector);
            }
            self.pay(controller);
            self.debts()
        })
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
        Debts {
            owed: self.owed.count(),
            released: self.released.count(),
            abandoned: self.abandoned.count(),
            strandings: self.strandings.load(Ordering::Relaxed),
            phantoms: self.phantoms.load(Ordering::Relaxed),
        }
    }

    /// Whether real hardware is holding `vector` for this guest, whether or not
    /// anything is expected to acknowledge it.
    fn holds(&self, vector: Vector) -> bool {
        self.owed.get(vector) || self.abandoned.get(vector)
    }

    /// Pays every debt the guest has released that real hardware is holding at
    /// the top, and drops any it is not holding at all.
    ///
    /// Each pass compares the highest debt due against what the controller says
    /// it is holding highest, which is the one write that cannot retire
    /// somebody else's interrupt, and there are exactly three answers:
    ///
    /// - the same vector, so the debt is what an acknowledgement would retire,
    ///   and it is paid;
    /// - something higher, which belongs to whoever owns it — everything below
    ///   is unreachable until that one is retired, and there is nothing useful
    ///   to do about it now;
    /// - something lower, or nothing at all, which means the controller is not
    ///   holding the debt: no acknowledgement could ever retire it, so the bit
    ///   is dropped instead of waiting for a moment that cannot arrive.
    ///
    /// Bounded by the number of vectors, because every pass that continues has
    /// cleared one bit of the released map and nothing inside can set one.
    fn pay(&self, controller: &impl InService) {
        for _ in 0..Bitmap::CAPACITY {
            let Some(due) = self.released.highest() else {
                return;
            };
            let holding = controller.in_service_top();
            if holding == Some(due) {
                controller.end_of_interrupt();
                self.released.clear(due);
                continue;
            }
            // A controller that could not be asked has said nothing, and a debt
            // is never dropped on the strength of an answer nobody gave.
            if holding.is_some_and(|top| top > due) || !controller.reachable() {
                return;
            }
            self.forget(due);
        }
    }

    /// Drops a debt real hardware is not holding, and counts it.
    ///
    /// Such a bit is unpayable by construction — it can never be what an
    /// acknowledgement would retire — so keeping it would leave the ledger
    /// permanently reporting a debt the machine does not have, and would pay
    /// for the vector's *next* arrival the moment that one came to the top,
    /// while the guest was still servicing it.
    fn forget(&self, vector: Vector) {
        self.released.clear(vector);
        self.phantoms.fetch_add(1, Ordering::Relaxed);
    }
}

/// What real hardware is holding for one guest, and what has become of what it
/// held before.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Debts {
    /// How many vectors the guest may still acknowledge.
    owed: u32,
    /// How many it has acknowledged that are waiting for their turn at the top.
    released: u32,
    /// How many are held with no acknowledgement expected.
    abandoned: u32,
    /// How many have been abandoned since the controller was built.
    strandings: u32,
    /// How many came due against a controller that was not holding them.
    phantoms: u32,
}

impl Debts {
    /// Whether real hardware is holding nothing at all for this guest.
    ///
    /// About what is outstanding now and not about what has happened: a
    /// controller that had a debt written off and honoured afterwards is
    /// holding nothing and says so, while its counts still say what it did.
    #[must_use]
    pub(crate) const fn is_empty(&self) -> bool {
        self.owed == 0 && self.released == 0 && self.abandoned == 0
    }
}

impl Display for Debts {
    /// Only what is not zero, because a line reporting five of them says
    /// nothing.
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        let mut said = false;
        for (count, what) in [
            (self.owed, "owed"),
            (self.released, "released"),
            (self.abandoned, "abandoned"),
            (self.strandings, "abandoned in all"),
            (self.phantoms, "never held"),
        ] {
            if count == 0 {
                continue;
            }
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
}

#[cfg(test)]
mod tests {
    //! Every interleaving these assert is one the module's own documentation
    //! describes: the controller is a parameter, so a test is a controller.

    use alloc::{format, vec::Vec};
    use core::cell::{Cell, RefCell};

    use descriptors::Vector;

    use super::{InService, Ledger};

    /// A controller holding a stack of in-service vectors, retiring the top one
    /// when it is acknowledged.
    ///
    /// Which is what the real one does: interrupts nest by priority, the
    /// highest is the one being serviced, and the acknowledgement register
    /// takes no vector — so a test that let one be named would be testing
    /// something the hardware cannot do.
    struct Holding {
        /// Highest last, so the top of the stack is the vector in service.
        held: RefCell<Vec<Vector>>,
        /// Every acknowledgement issued, in order, so that paying somebody
        /// else's interrupt is visible rather than merely wrong.
        paid: RefCell<Vec<Vector>>,
    }

    impl Holding {
        /// A controller holding these vectors, lowest priority first.
        fn new(held: &[Vector]) -> Self {
            Self {
                held: RefCell::new(held.to_vec()),
                paid: RefCell::new(Vec::new()),
            }
        }

        /// What has been acknowledged, in the order it was.
        fn paid(&self) -> Vec<Vector> {
            self.paid.borrow().clone()
        }

        /// Takes another interrupt, which nests above what is already held.
        fn accepted(&self, vector: Vector) {
            self.held.borrow_mut().push(vector);
        }
    }

    impl InService for Holding {
        fn in_service_top(&self) -> Option<Vector> {
            self.held.borrow().last().copied()
        }

        fn reachable(&self) -> bool {
            true
        }

        fn end_of_interrupt(&self) {
            if let Some(vector) = self.held.borrow_mut().pop() {
                self.paid.borrow_mut().push(vector);
            }
        }

        /// Nothing else reaches this controller: it is one test's own, and a
        /// test process may not execute the instruction that holds a
        /// real processor's interrupts off.
        fn exclusively<T>(&self, paying: impl FnOnce() -> T) -> T {
            paying()
        }
    }

    /// A controller that cannot be reached, which is what a failure to find
    /// this processor's own leaves a caller holding.
    struct Unreachable;

    impl InService for Unreachable {
        fn in_service_top(&self) -> Option<Vector> {
            None
        }

        fn reachable(&self) -> bool {
            false
        }

        fn end_of_interrupt(&self) {
            unreachable!("a controller that cannot be asked is never acknowledged");
        }

        fn exclusively<T>(&self, paying: impl FnOnce() -> T) -> T {
            paying()
        }
    }

    /// A controller that takes a fresh interrupt in the instant after an
    /// acknowledgement, recording the debt for it.
    ///
    /// A real one cannot do that to a settlement, because a settlement holds
    /// this processor's interrupts off and an arrival is a maskable
    /// interrupt. The interleaving is here so that what a settlement
    /// *answers* can be pinned for the case masking makes impossible: a
    /// debt recorded underneath it has to be in the answer rather than
    /// reported as settled.
    struct Interposing<'ledger> {
        /// The controller proper.
        controller: Holding,
        /// The ledger the arrival records its debt in.
        ledger: &'ledger Ledger,
        /// What arrives, once.
        arrival: Cell<Option<Vector>>,
    }

    impl InService for Interposing<'_> {
        fn in_service_top(&self) -> Option<Vector> {
            self.controller.in_service_top()
        }

        fn reachable(&self) -> bool {
            true
        }

        fn end_of_interrupt(&self) {
            self.controller.end_of_interrupt();
            if let Some(vector) = self.arrival.take() {
                self.controller.accepted(vector);
                self.ledger.owe(vector);
            }
        }

        fn exclusively<T>(&self, paying: impl FnOnce() -> T) -> T {
            paying()
        }
    }

    /// Three vectors of three priority classes, in ascending order.
    const LOW: Vector = Vector::new(0x31);
    const HIGH: Vector = Vector::new(0x52);
    const HIGHER: Vector = Vector::new(0x84);

    #[test]
    fn a_vector_that_was_never_owed_releases_nothing() {
        let ledger = Ledger::new();
        let controller = Holding::new(&[LOW]);

        ledger.release(LOW, &controller);
        assert!(ledger.debts().is_empty());
        assert!(
            controller.paid().is_empty(),
            "an edge-triggered vector was acknowledged when it arrived, and paying \
             again would retire somebody else's interrupt"
        );
    }

    #[test]
    fn the_guest_acknowledging_pays_the_debt_it_makes_payable() {
        let ledger = Ledger::new();
        let controller = Holding::new(&[LOW]);
        ledger.owe(LOW);

        ledger.release(LOW, &controller);
        assert_eq!(controller.paid(), [LOW]);
        assert!(ledger.debts().is_empty());
    }

    #[test]
    fn a_debt_is_paid_once_however_often_the_guest_acknowledges() {
        // A guest may write its acknowledgement register as often as it likes, and
        // every write past the first has no debt to find: a second payment would
        // retire whatever the controller had taken since.
        let ledger = Ledger::new();
        let controller = Holding::new(&[LOW, HIGH]);
        ledger.owe(LOW);

        for _ in 0..8 {
            ledger.release(LOW, &controller);
        }
        assert!(controller.paid().is_empty(), "HIGH is what is at the top");

        controller.end_of_interrupt();
        ledger.release(HIGH, &controller);
        assert_eq!(controller.paid(), [HIGH, LOW]);
        assert!(ledger.debts().is_empty());
    }

    #[test]
    fn a_debt_is_not_paid_while_a_higher_vector_is_in_service() {
        // The interleaving the two payable states exist for: the guest services a
        // low vector while a higher one is still held, and acknowledges the low one
        // first. Its debt is released and not payable, because the real controller
        // would retire the higher one instead.
        let ledger = Ledger::new();
        let controller = Holding::new(&[LOW, HIGH]);
        ledger.owe(LOW);

        ledger.release(LOW, &controller);
        assert!(controller.paid().is_empty());
        let debts = ledger.debts();
        assert_eq!(debts.released, 1, "the debt is kept rather than forgotten");
        assert_eq!(debts.owed, 0);
    }

    #[test]
    fn a_payment_deferred_is_retried_at_the_next_acknowledgement() {
        // Nothing else would ever retry it. The vector blocking the top is retired
        // by whoever owns it, and the guest's next acknowledgement — of any vector,
        // debt or no debt — is what notices that the debt below has come up.
        let ledger = Ledger::new();
        let controller = Holding::new(&[LOW, HIGH]);
        ledger.owe(LOW);
        ledger.release(LOW, &controller);

        controller.end_of_interrupt();
        ledger.release(HIGHER, &controller);
        assert_eq!(controller.paid(), [HIGH, LOW]);
        assert!(ledger.debts().is_empty());
    }

    #[test]
    fn debts_are_paid_from_the_top_down() {
        let ledger = Ledger::new();
        let controller = Holding::new(&[LOW, HIGH]);
        ledger.owe(LOW);
        ledger.owe(HIGH);

        ledger.release(HIGH, &controller);
        assert_eq!(
            controller.paid(),
            [HIGH],
            "only the debt at the top is payable; the lower one is still the guest's"
        );

        ledger.release(LOW, &controller);
        assert_eq!(controller.paid(), [HIGH, LOW]);
        assert!(ledger.debts().is_empty());
    }

    #[test]
    fn a_repeated_arrival_is_not_a_second_debt() {
        // The real controller has one in-service bit per vector and cannot accept a
        // second interrupt on a vector it is already holding, so a repeat is one
        // debt and one acknowledgement.
        let ledger = Ledger::new();
        let controller = Holding::new(&[LOW]);
        ledger.owe(LOW);
        ledger.owe(LOW);

        ledger.release(LOW, &controller);
        assert_eq!(controller.paid(), [LOW]);
        assert!(ledger.debts().is_empty());
    }

    #[test]
    fn a_refused_arrival_keeps_its_debt_and_acknowledges_nothing() {
        // The guest was not given the interrupt and will never acknowledge it, so
        // nothing will discharge the debt. Paying it would clear the remote
        // in-service state of an I/O controller whose line is still asserted, and
        // the interrupt would arrive again at once, into a guest that has already
        // refused it.
        let ledger = Ledger::new();
        let controller = Holding::new(&[LOW]);
        ledger.owe(LOW);

        ledger.abandon(LOW);
        assert!(controller.paid().is_empty());
        let debts = ledger.debts();
        assert_eq!((debts.owed, debts.abandoned, debts.strandings), (0, 1, 1));
        assert!(!debts.is_empty());
    }

    #[test]
    fn a_reset_abandons_a_debt_rather_than_acknowledging_a_line_nobody_has_quieted() {
        // A guest that has been reset will never acknowledge anything, and the
        // source it was servicing is still asserting: the acknowledgement is
        // withheld for good rather than issued into a processor that is being reset
        // and cannot service the re-arrival.
        let ledger = Ledger::new();
        let controller = Holding::new(&[LOW]);
        ledger.owe(LOW);

        let debts = ledger.settle(&controller);
        assert!(controller.paid().is_empty());
        assert_eq!((debts.owed, debts.abandoned, debts.strandings), (0, 1, 1));
        assert!(!debts.is_empty(), "the answer says what was left behind");
    }

    #[test]
    fn a_reset_still_pays_what_the_guest_had_already_acknowledged() {
        // Released is not abandoned. The guest said it had finished with this one
        // before it stopped existing, and that acknowledgement is the licence to
        // issue the real one; only the debts it never got to are written off.
        let ledger = Ledger::new();
        let controller = Holding::new(&[LOW, HIGH]);
        ledger.owe(LOW);
        ledger.owe(HIGH);
        ledger.release(HIGH, &controller);
        assert_eq!(controller.paid(), [HIGH]);

        let debts = ledger.settle(&controller);
        assert_eq!(controller.paid(), [HIGH], "LOW was never acknowledged");
        assert_eq!((debts.owed, debts.abandoned), (0, 1));
    }

    #[test]
    fn a_software_disable_abandons_a_debt_the_guest_may_yet_come_back_for() {
        // A guest that switches its controller off keeps whatever it had already
        // taken and may switch it on again, so the write-off is not a refusal to
        // honour the acknowledgement — it is the absence of an expectation of one.
        // If it arrives after all it is paid, because a guest acknowledging a vector
        // is itself the licence to retire it.
        let ledger = Ledger::new();
        let controller = Holding::new(&[LOW]);
        ledger.owe(LOW);

        let debts = ledger.settle(&controller);
        assert!(controller.paid().is_empty());
        assert_eq!(debts.abandoned, 1);

        ledger.release(LOW, &controller);
        assert_eq!(controller.paid(), [LOW]);
        let debts = ledger.debts();
        assert!(debts.is_empty());
        assert_eq!(
            debts.strandings, 1,
            "the write-off happened, and is counted whether or not it was honoured"
        );
    }

    #[test]
    fn a_settlement_answers_about_every_debt_and_not_only_the_payable_ones() {
        // The answer decides whether the reset is reported as clean, and a
        // settlement that looked only at what it could pay would call this one
        // clean while real hardware went on holding two vectors.
        let ledger = Ledger::new();
        let controller = Holding::new(&[LOW, HIGH, HIGHER]);
        ledger.owe(LOW);
        ledger.owe(HIGH);
        ledger.release(LOW, &controller);

        let debts = ledger.settle(&controller);
        assert!(controller.paid().is_empty(), "HIGHER is somebody else's");
        assert_eq!((debts.released, debts.abandoned), (1, 1));
        assert!(!debts.is_empty());
    }

    #[test]
    fn a_debt_re_owed_underneath_a_settlement_is_in_the_answer() {
        // The interleaving masking now makes impossible, played out anyway: an
        // arrival lands in the instant after the settlement's own acknowledgement,
        // when its sweep has already been past. Its debt is unpayable — nothing
        // pays an owed bit — so an answer that did not mention it would be a
        // settlement reported as complete with a real in-service bit held for good.
        let ledger = Ledger::new();
        let controller = Interposing {
            controller: Holding::new(&[LOW, HIGH]),
            ledger: &ledger,
            arrival: Cell::new(Some(HIGHER)),
        };
        // The guest acknowledges LOW while HIGH is still in service, so the payment
        // is deferred; HIGH is then retired by whoever owns it, and the debt below
        // becomes payable.
        ledger.owe(LOW);
        ledger.release(LOW, &controller);
        assert!(controller.controller.paid().is_empty());
        InService::end_of_interrupt(&controller.controller);

        let debts = ledger.settle(&controller);
        assert_eq!(controller.controller.paid(), [HIGH, LOW]);
        assert_eq!(debts.owed, 1, "the arrival's debt is the answer's");
        assert!(!debts.is_empty());
    }

    #[test]
    fn a_debt_hardware_is_not_holding_is_dropped_and_counted() {
        // Something else acknowledged the vector — the one unowed acknowledgement in
        // this image is the controller's own error handler's — so the bit can never
        // be what an acknowledgement would retire. Keeping it would report a debt
        // the machine does not have forever, and would pay for the vector's *next*
        // arrival while the guest was still servicing it.
        let ledger = Ledger::new();
        let controller = Holding::new(&[HIGH]);
        ledger.owe(HIGHER);

        ledger.release(HIGHER, &controller);
        assert!(controller.paid().is_empty(), "HIGH is not the debt");
        let debts = ledger.debts();
        assert!(debts.is_empty(), "the phantom is gone rather than stuck");
        assert_eq!(debts.phantoms, 1);
    }

    #[test]
    fn a_controller_that_cannot_be_reached_leaves_every_debt_where_it_was() {
        // It answers nothing about what it holds, and a debt is never dropped on the
        // strength of an answer nobody gave: an unreachable controller and one
        // holding nothing would otherwise be the same thing, and they are opposites.
        let ledger = Ledger::new();
        let controller = Holding::new(&[LOW]);
        ledger.owe(LOW);
        ledger.release(LOW, &Unreachable);

        let debts = ledger.debts();
        assert_eq!((debts.released, debts.phantoms), (1, 0));

        // Reachable again, and the debt is paid rather than lost.
        ledger.release(HIGHER, &controller);
        assert_eq!(controller.paid(), [LOW]);
        assert!(ledger.debts().is_empty());
    }

    #[test]
    fn debts_report_only_what_is_not_zero() {
        // The record a machine with no serial port is read by, so what it says has
        // to be legible: five counts of which four are usually zero.
        let ledger = Ledger::new();
        assert_eq!(format!("{}", ledger.debts()), "owed nothing");

        ledger.owe(LOW);
        assert_eq!(format!("{}", ledger.debts()), "1 owed");

        ledger.abandon(LOW);
        assert_eq!(
            format!("{}", ledger.debts()),
            "1 abandoned, 1 abandoned in all"
        );
    }
}
