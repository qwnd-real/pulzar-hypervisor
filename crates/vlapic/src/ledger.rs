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
//! dropped because the controller refused a write, or because its turn had not
//! come, is a real in-service entry with nothing left that would ever retire
//! it.

use descriptors::Vector;
use log::warn;

use crate::vectors::Bitmap;

/// The debts one processor's real controller is holding for its guest.
#[derive(Debug, Default)]
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
    pub(crate) fn release(&self, vector: Vector) -> bool {
        if !self.owed.clear(vector) {
            return false;
        }
        self.released.set(vector);
        self.drain();
        true
    }

    /// Pays every debt that has come to the top, in the order hardware retires
    /// them.
    ///
    /// Each pass asks the controller what it is holding highest and pays only
    /// that, which is the one write that cannot retire somebody else's
    /// interrupt. It stops at the first vector that is not a released debt:
    /// anything below it is unreachable until that one is retired by whoever
    /// owns it, and there is nothing useful to do about it now.
    ///
    /// Bounded by the number of vectors, because each pass that continues has
    /// retired one in-service bit and nothing here sets one.
    pub(crate) fn drain(&self) {
        let Ok(local) = apic::local() else {
            return;
        };
        for _ in 0..Bitmap::CAPACITY {
            if self.released.is_empty() {
                return;
            }
            let Ok(Some(top)) = local.in_service_top() else {
                return;
            };
            if !self.released.get(top) {
                return;
            }
            if let Err(error) = local.end_of_interrupt() {
                // Deliberately kept. The real controller is still holding the
                // vector, so the debt is still real, and forgetting it here
                // would leave nothing that could ever retire it.
                warn!("vlapic: could not acknowledge {top} on real hardware: {error}");
                return;
            }
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
    pub(crate) fn settle(&self) -> bool {
        while let Some(vector) = self.owed.take_highest() {
            self.released.set(vector);
        }
        self.drain();
        self.released.is_empty()
    }

    /// Whether real hardware is owed anything at all.
    pub(crate) fn is_empty(&self) -> bool {
        self.owed.is_empty() && self.released.is_empty()
    }
}
