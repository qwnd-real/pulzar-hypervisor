//! Raising a controller's own error interrupt.
//!
//! The error status register records what went wrong; this is what tells the
//! guest. Every other interrupt this crate delivers came from somewhere else —
//! a command another processor wrote, a source on real hardware — and this one
//! has no source at all: the entry is the only one in the local vector table
//! with nothing behind it, so the controller itself is what raises it.
//!
//! # Why it is not simply a boolean at the recorder
//!
//! Because the recorder is not always the processor that has to take the
//! interrupt. An interrupt aimed at a controller is examined by the *sender*,
//! so a badly formed one is recorded on the target from another processor's
//! context — and raising it there means publishing into the target's register
//! file against its reset count and then making the target look, which is the
//! whole of ordinary delivery. A caller handed a boolean would raise it on
//! itself.
//!
//! So a record answers with a [`Raise`], which names the controller the
//! interrupt is owed to, and this module is the one thing that can discharge
//! one.
//!
//! # What the entry has to say before anything is delivered
//!
//! Three things, and a guest can put all three wrong. The entry is writable, so
//! it can hold a masked source, a vector no controller may deliver, or a
//! delivery mode this hypervisor does not raise — and the interrupt being
//! raised is itself an error report, so a controller that raised an error while
//! raising an error would be a controller that recursed. Xen's `vlapic_error`
//! deadlocked on exactly that shape.
//!
//! Nothing here recurses, and not by luck. The entry is examined *before* the
//! vector is accepted, so the acceptance can no longer record an error of its
//! own; and a record that did happen underneath this one would find the pending
//! set non-empty and arm nothing, because the set is accumulated before it is
//! tested.

use descriptors::Vector;
use log::warn;

use crate::{
    delivery::doorbell::nudge,
    machine::diagnostics::Report,
    priority,
    registers::{
        Accepted, Raise, Vlapic,
        error::Errors,
        icr::Trigger,
        lvt::{Delivery, Entry, Lvt},
    },
};

/// Records an error one controller noticed about another, and raises that
/// controller's error interrupt if the record armed one.
///
/// `from` is the controller of the processor that noticed the error and `on` is
/// the controller the error belongs to. The two differ for exactly one error:
/// an interrupt carrying a vector no controller may deliver is noticed by the
/// sender and is the *receiver's* to report.
pub(crate) fn record(from: &Vlapic, on: &Vlapic, errors: Errors) {
    if let Some(raise) = on.record_error(errors) {
        deliver(from, raise);
    }
}

/// The same, for an error a controller noticed about itself.
///
/// Which is every error but one. A guest naming a reserved register, sending a
/// command with an illegal vector, or sending one nobody accepted is a guest
/// whose own controller is the one with something to report, and the processor
/// running it is the one that will take the interrupt.
pub(crate) fn noticed(vlapic: &Vlapic, errors: Errors) {
    record(vlapic, vlapic, errors);
}

/// Gives a controller the error interrupt its error status has armed.
///
/// [`raised`] is the whole of the decision; what is left here is performing it,
/// which is the acceptance and the doorbell an ordinary interrupt takes.
fn deliver(from: &Vlapic, raise: Raise<'_>) {
    let on = raise.owed();
    let vector = match raised(on.lvt(Entry::Error)) {
        Raised::On(vector) => vector,
        // Not worth a line: masking the entry is how software asks for exactly
        // this, and the guest reads the register instead.
        Raised::Masked => return,
        Raised::Delivery => {
            report(on, "a delivery mode this controller does not raise it as");
            return;
        }
        Raised::Vector => {
            report(on, "a vector no controller may deliver");
            return;
        }
    };
    match on.accept(vector, Trigger::Edge) {
        Accepted::Requested | Accepted::Coalesced => nudge(from, on),
        // Nothing is owed for one of these — it came from the controller itself
        // rather than from a real source — so a controller that is not accepting
        // simply does not get its own error interrupt, which is what a
        // software-disabled one does with everything.
        declined => {
            on.diagnostics().declined();
            if matches!(declined, Accepted::Resetting) {
                on.diagnostics().dropped();
            }
        }
    }
}

/// What a controller's error entry says should become of its error interrupt.
///
/// Three of the four answers are no, and a guest can reach all three by writing
/// one register. Each of them leaves the error recorded and latched: masking an
/// entry stops the interrupt and not the recording, which is what the
/// architecture says and what lets a guest poll the register instead.
///
/// Written out as a decision over the entry so that it can be checked without a
/// controller, because it is the decision that has to be right for this crate
/// to deliver an error interrupt at all — and the one an error raised while
/// raising an error would go through.
const fn raised(entry: Lvt) -> Raised {
    if entry.masked() {
        return Raised::Masked;
    }
    // The entry has a message-type field on the controller this crate presents,
    // and only a fixed delivery reads the vector the entry exists to name. The
    // others are events the processor takes by their own entry point, and this
    // crate does not synthesise one of those on a guest's behalf: a
    // system-management interrupt is refused everywhere, and a non-maskable one
    // would be indistinguishable to the guest from the platform's own.
    if !matches!(Delivery::from_bits(entry.delivery()), Some(Delivery::Fixed)) {
        return Raised::Delivery;
    }
    // Already recorded as an illegal vector when the guest wrote it. Delivering
    // here would set a request bit in the range the controller never sets one in,
    // and recording it again would be the recursion this module exists not to
    // have.
    if priority::legal(entry.vector()) {
        Raised::On(entry.vector())
    } else {
        Raised::Vector
    }
}

/// What [`raised`] answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Raised {
    /// Deliver the error interrupt on this vector.
    On(Vector),
    /// The entry is masked, so nothing is delivered and nothing is worth
    /// saying.
    Masked,
    /// The entry names a delivery mode this controller does not raise an error
    /// as.
    Delivery,
    /// The entry names a vector no controller may deliver.
    Vector,
}

/// Says once that a controller's error entry does not name an interrupt it can
/// deliver, so its errors are recorded and never raised.
fn report(on: &Vlapic, why: &str) {
    if on.diagnostics().say(Report::ErrorEntry) {
        warn!(
            "vlapic: {} has {why} in its error entry, so the errors it records are latched for the \
             guest to read and no error interrupt is raised",
            on.index()
        );
    }
}

#[cfg(test)]
mod tests {
    //! What the entry says is the whole of what is decided here, and it is the
    //! one part that needs no controller. What performing it does — the
    //! acceptance and the doorbell — is the same pair every other delivery
    //! takes.

    use descriptors::Vector;

    use super::{Raised, raised};
    use crate::registers::lvt::{Delivery, Lvt};

    /// One in the middle of the range the host claims nothing in, standing in
    /// for the vector a guest would point its error interrupt at.
    const GUEST: Vector = Vector::new(0x33);

    #[test]
    fn an_armed_fixed_entry_raises_the_error_interrupt_on_the_vector_it_names() {
        let entry = Lvt::new().with_vector(GUEST);
        assert_eq!(raised(entry), Raised::On(GUEST));
    }

    #[test]
    fn a_masked_entry_raises_nothing_and_is_not_a_complaint() {
        // Masking is how software asks for the errors to be recorded and not
        // delivered, so this is the one refusal that says nothing in the log.
        let entry = Lvt::new().with_vector(GUEST).with_masked(true);
        assert_eq!(raised(entry), Raised::Masked);
    }

    #[test]
    fn a_mode_that_carries_no_vector_raises_nothing() {
        // Every encoding but fixed, including the three the field reserves: the
        // vector is the whole of what the entry names, and a mode that does not
        // read it is one this crate does not raise an error as.
        for delivery in 1..8 {
            let entry = Lvt::new().with_vector(GUEST).with_delivery(delivery);
            assert_eq!(raised(entry), Raised::Delivery, "delivery {delivery:#05b}");
        }
        assert_eq!(Delivery::Fixed as u8, 0, "fixed is the encoding left out");
    }

    #[test]
    fn a_vector_no_controller_may_deliver_raises_nothing() {
        // The shape Xen deadlocked on: raising an error interrupt on an illegal
        // vector is itself an error, so a controller that delivered here would
        // record one while recording one. The vector was already reported when
        // the guest wrote it.
        for number in 0..16 {
            let entry = Lvt::new().with_vector(Vector::new(number));
            assert_eq!(raised(entry), Raised::Vector, "vector {number:#x}");
        }
        assert_eq!(raised(Lvt::new().with_vector(Vector::new(16))), {
            Raised::On(Vector::new(16))
        });
    }
}
