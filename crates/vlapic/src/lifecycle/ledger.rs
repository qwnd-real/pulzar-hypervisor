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
//! # Two states, not one
//!
//! A debt is *owed* from the moment the interrupt arrives, and becomes
//! *released* when the guest acknowledges its own controller. Only a released
//! debt may be paid, and the distinction is not bookkeeping: paying a debt the
//! guest is still servicing would let the line re-assert underneath a driver
//! that is halfway through quieting it, which is the exact failure withholding
//! the acknowledgement exists to prevent.
//!
//! The two states are needed together because the guest does not acknowledge in
//! the order hardware retires. A guest servicing a low vector while a higher
//! one is merely requested acknowledges the low one first; its debt is then
//! released but not payable, because the higher vector is what the real
//! controller holds at the top. It becomes payable later, when the higher one
//! is retired, and [`Ledger::drain`] is what notices.
//!
//! # Nothing is forgotten because paying failed
//!
//! Every operation that could not pay leaves the debt where it was. A debt
//! dropped because the controller could not be reached, or because its turn had
//! not come, is a real in-service entry with nothing left that would ever
//! retire it.

use apic::LocalApic;
use descriptors::Vector;
use x86_64::instructions::interrupts;

use crate::registers::bitmap::Bitmap;

/// What the ledger needs of the controller holding its debts.
///
/// Two operations, which are the whole of what paying a debt is: ask what the
/// controller is holding highest, and retire it. Taken as a parameter rather than
/// reached for through [`apic::local`], because *which* controller a debt is paid
/// through is the one thing that must not be assumed — the acknowledgement
/// register carries no vector, so paying through the wrong controller retires an
/// interrupt belonging to somebody else and strands the one that should have been
/// retired for the life of the machine.
///
/// Passing it in is also what makes every interleaving in this module a test:
/// nothing here has to be on a processor that has a controller at all.
pub(crate) trait InService {
    /// The highest-priority vector the controller is holding in service, or
    /// `None` if it is holding nothing — or cannot be reached to be asked.
    fn in_service_top(&self) -> Option<Vector>;

    /// Retires whichever vector that is, which is the only thing the
    /// acknowledgement register can be told to do.
    fn end_of_interrupt(&self);

    /// Runs `paying` with nothing else on this processor able to reach the
    /// controller.
    ///
    /// Part of the seam rather than of the bookkeeping, because it is a property
    /// of the *controller* and not of the debts: every pass of a payment is a
    /// read of a real register followed by a write that depends on what it said,
    /// and a handler interposing between the two would be answered about one
    /// vector and paid for another.
    fn exclusively(&self, paying: impl FnOnce());
}

impl InService for LocalApic {
    fn in_service_top(&self) -> Option<Vector> {
        LocalApic::in_service_top(*self)
    }

    fn end_of_interrupt(&self) {
        LocalApic::end_of_interrupt(*self);
    }

    fn exclusively(&self, paying: impl FnOnce()) {
        interrupts::without_interrupts(paying);
    }
}

/// A controller that may not have been reachable when it was asked for.
///
/// Answering as though it were holding nothing is what leaves every debt where
/// it was: a debt that could not be paid is a real in-service entry with nothing
/// left that would retire it, and inventing an acknowledgement would retire
/// whatever the controller does hold instead.
impl InService for Option<LocalApic> {
    fn in_service_top(&self) -> Option<Vector> {
        self.as_ref().and_then(InService::in_service_top)
    }

    fn end_of_interrupt(&self) {
        if let Some(local) = self {
            InService::end_of_interrupt(local);
        }
    }

    fn exclusively(&self, paying: impl FnOnce()) {
        interrupts::without_interrupts(paying);
    }
}

/// The debts one processor's real controller is holding for its guest.
#[derive(Debug)]
pub(crate) struct Ledger {
    owed: Bitmap,
    released: Bitmap,
}

impl Ledger {
    /// Nothing owed, which is what reset leaves this.
    pub(crate) const fn new() -> Self {
        Self {
            owed: Bitmap::new(),
            released: Bitmap::new(),
        }
    }

    /// Records that real hardware holds `vector` in service for this guest.
    ///
    /// Recorded before the guest is given the interrupt, so that a guest which
    /// acknowledges immediately finds the debt already there.
    ///
    /// At most one debt per vector can exist, because the real controller has
    /// one in-service bit per vector and cannot accept a second interrupt on a
    /// vector it is already holding. A repeat is therefore not a second debt
    /// and is not counted as one.
    pub(crate) fn owe(&self, vector: Vector) {
        self.owed.set(vector);
    }

    /// Records that the guest has finished with `vector`, and pays whatever
    /// that makes payable.
    ///
    /// Answers whether there was a debt to release at all. A vector with none
    /// was edge triggered and was acknowledged when it arrived, which is the
    /// common case and costs nothing here.
    pub(crate) fn release(&self, vector: Vector, controller: &impl InService) -> bool {
        if !self.owed.clear(vector) {
            return false;
        }
        self.released.set(vector);
        self.drain(controller);
        true
    }

    /// Pays every debt that has come to the top, in the order hardware retires
    /// them.
    pub(crate) fn drain(&self, controller: &impl InService) {
        // Asked before the controller is held, because the overwhelmingly common
        // case is a guest acknowledging an edge-triggered interrupt that owed
        // nothing: there is no debt to pay and no reason to hold anything off to
        // establish that.
        if self.released.is_empty() {
            return;
        }
        controller.exclusively(|| self.pay(controller));
    }

    /// The paying itself, with nothing said about interrupts.
    ///
    /// Each pass asks the controller what it is holding highest and pays only
    /// that, which is the one write that cannot retire somebody else's
    /// interrupt. It stops at the first vector that is not a released debt:
    /// anything below it is unreachable until that one is retired by whoever
    /// owns it, and there is nothing useful to do about it now.
    ///
    /// Bounded by the number of vectors, because each pass that continues has
    /// retired one in-service bit and nothing here sets one.
    fn pay(&self, controller: &impl InService) {
        for _ in 0..Bitmap::CAPACITY {
            if self.released.is_empty() {
                return;
            }
            let Some(top) = controller.in_service_top() else {
                return;
            };
            if !self.released.get(top) {
                return;
            }
            controller.end_of_interrupt();
            self.released.clear(top);
        }
    }

    /// Pays everything outstanding, because the guest that owed it is about to
    /// stop existing.
    ///
    /// A guest that has been reset will never acknowledge anything, so every
    /// debt it left — released or not — has to be settled by somebody, and this
    /// is the one place it is known that nobody else will. Debts the guest was
    /// still servicing are released first for exactly that reason: there is no
    /// longer a driver whose progress the withholding was protecting.
    ///
    /// Answers whether everything was settled. What could not be paid is
    /// retained, and stays retained across the reset — a real in-service entry
    /// outlives the guest that caused it, and the processor may still be able
    /// to retire it later.
    pub(crate) fn settle(&self, controller: &impl InService) -> bool {
        while let Some(vector) = self.owed.take_highest() {
            self.released.set(vector);
        }
        self.drain(controller);
        self.released.is_empty()
    }

    /// Whether real hardware is owed anything at all.
    pub(crate) fn is_empty(&self) -> bool {
        self.owed.is_empty() && self.released.is_empty()
    }
}

#[cfg(test)]
mod tests {
    //! Every interleaving these assert is one the module's own documentation
    //! describes and nothing could previously exercise: the controller is a
    //! parameter, so a test is a controller.

    use alloc::vec::Vec;
    use core::cell::RefCell;

    use descriptors::Vector;

    use super::{InService, Ledger};

    /// A controller holding a stack of in-service vectors, retiring the top one
    /// when it is acknowledged.
    ///
    /// Which is what the real one does: interrupts nest by priority, the highest
    /// is the one being serviced, and the acknowledgement register takes no
    /// vector — so a test that let one be named would be testing something the
    /// hardware cannot do.
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
    }

    impl InService for Holding {
        fn in_service_top(&self) -> Option<Vector> {
            self.held.borrow().last().copied()
        }

        fn end_of_interrupt(&self) {
            if let Some(vector) = self.held.borrow_mut().pop() {
                self.paid.borrow_mut().push(vector);
            }
        }

        /// Nothing else reaches this controller: it is one test's own, and a test
        /// process may not execute the instruction that holds a real processor's
        /// interrupts off.
        fn exclusively(&self, paying: impl FnOnce()) {
            paying();
        }
    }

    /// A controller that cannot be reached, which is what a failure to find this
    /// processor's own leaves a caller holding.
    struct Unreachable;

    impl InService for Unreachable {
        fn in_service_top(&self) -> Option<Vector> {
            None
        }

        fn end_of_interrupt(&self) {
            unreachable!("a controller that holds nothing is never acknowledged");
        }

        fn exclusively(&self, paying: impl FnOnce()) {
            paying();
        }
    }

    /// Two vectors of different priority classes, the second the higher.
    const LOW: Vector = Vector::new(0x31);
    const HIGH: Vector = Vector::new(0x52);

    #[test]
    fn a_vector_that_was_never_owed_releases_nothing() {
        let ledger = Ledger::new();
        let controller = Holding::new(&[LOW]);

        assert!(!ledger.release(LOW, &controller));
        assert!(ledger.is_empty());
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

        assert!(ledger.release(LOW, &controller));
        assert_eq!(controller.paid(), [LOW]);
        assert!(ledger.is_empty());
    }

    #[test]
    fn a_debt_is_not_paid_while_a_higher_vector_is_in_service() {
        // The interleaving the two states exist for: the guest services a low
        // vector while a higher one is still held, and acknowledges the low one
        // first. Its debt is released and not payable, because the real
        // controller would retire the higher one instead.
        let ledger = Ledger::new();
        let controller = Holding::new(&[LOW, HIGH]);
        ledger.owe(LOW);

        assert!(ledger.release(LOW, &controller));
        assert!(controller.paid().is_empty());
        assert!(!ledger.is_empty(), "the debt is kept rather than forgotten");

        // The higher vector is retired by whoever owns it, and the debt below it
        // becomes payable — which is what draining again is for.
        controller.end_of_interrupt();
        ledger.drain(&controller);
        assert_eq!(controller.paid(), [HIGH, LOW]);
        assert!(ledger.is_empty());
    }

    #[test]
    fn debts_are_paid_from_the_top_down_in_one_drain() {
        let ledger = Ledger::new();
        let controller = Holding::new(&[LOW, HIGH]);
        ledger.owe(LOW);
        ledger.owe(HIGH);

        assert!(ledger.release(HIGH, &controller));
        assert_eq!(
            controller.paid(),
            [HIGH],
            "only the debt at the top is payable; the lower one is still the guest's"
        );

        assert!(ledger.release(LOW, &controller));
        assert_eq!(controller.paid(), [HIGH, LOW]);
        assert!(ledger.is_empty());
    }

    #[test]
    fn settling_pays_a_debt_the_guest_never_acknowledged() {
        // A guest that has been reset will never acknowledge anything, so a debt
        // it was still servicing has to be paid by somebody.
        let ledger = Ledger::new();
        let controller = Holding::new(&[LOW]);
        ledger.owe(LOW);

        assert!(ledger.settle(&controller));
        assert_eq!(controller.paid(), [LOW]);
        assert!(ledger.is_empty());
    }

    #[test]
    fn settling_keeps_what_it_could_not_pay_and_says_so() {
        // The vector the guest owed is not what the controller is holding
        // highest, and nothing here may retire the one that is: the debt stays,
        // and the answer says the machine is still holding something.
        let ledger = Ledger::new();
        let controller = Holding::new(&[LOW, HIGH]);
        ledger.owe(LOW);

        assert!(!ledger.settle(&controller));
        assert!(controller.paid().is_empty());
        assert!(!ledger.is_empty());
    }

    #[test]
    fn a_controller_that_cannot_be_reached_leaves_every_debt_where_it_was() {
        let ledger = Ledger::new();
        ledger.owe(LOW);

        assert!(!ledger.settle(&Unreachable));
        assert!(
            !ledger.is_empty(),
            "a debt that could not be paid is a real in-service entry with nothing \
             left that would retire it"
        );
    }

    #[test]
    fn a_repeated_arrival_is_not_a_second_debt() {
        // The real controller has one in-service bit per vector and cannot accept
        // a second interrupt on a vector it is already holding, so a repeat is
        // one debt and one acknowledgement.
        let ledger = Ledger::new();
        let controller = Holding::new(&[LOW]);
        ledger.owe(LOW);
        ledger.owe(LOW);

        assert!(ledger.release(LOW, &controller));
        assert_eq!(controller.paid(), [LOW]);
        assert!(ledger.is_empty());
    }
}
