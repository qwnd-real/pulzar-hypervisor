//! The seam every interrupt that arrived on real hardware and belongs to a
//! guest reaches.
//!
//! What arrives is a physical vector nothing in the hypervisor claimed, which
//! on a machine whose I/O controllers are passed through means it was meant for
//! the guest. Nothing is translated, and nothing has to be: every source the
//! guest can reach is programmed onto real hardware with the guest's own
//! vector.
//!
//! Whether real hardware may be acknowledged now is the whole of what is
//! decided here, and two things decide it: the controller itself, which
//! recorded as it accepted the interrupt whether the interrupt arrived level
//! triggered, and the vector, because an acknowledgement withheld in the
//! priority class the host keeps for itself would hold the host's own
//! interrupts off with it.

use apic::LocalApic;
use descriptors::Vector;
use log::{trace, warn};

use crate::{
    VlapicError,
    machine::current,
    priority::Priority,
    registers::{Accepted, Vlapic, icr::Trigger},
};

/// Gives this processor's guest an interrupt that arrived on real hardware.
///
/// The seam every unclaimed interrupt reaches. What arrives is a physical
/// vector that nothing in the hypervisor claimed, which on a machine whose I/O
/// controllers are passed through means it was meant for the guest.
///
/// Whether real hardware may be acknowledged now is the whole of what is
/// decided here, and the controller itself is asked: it recorded, as it
/// accepted the interrupt, whether the interrupt arrived level triggered.
///
/// An edge-triggered interrupt is finished with once taken, so it is
/// acknowledged immediately and the guest is given it. A level-triggered one is
/// asserted until the guest's own driver deals with whatever raised it, so
/// acknowledging now would deliver it again at once — the acknowledgement is
/// withheld and becomes owed, and is issued when the guest acknowledges its
/// own.
///
/// With one exception, which is the priority class the host keeps for its own
/// interprocessor interrupts. A withheld acknowledgement there would hold every
/// one of those off as well, for as long as the guest took to give its own, so
/// such an arrival is acknowledged at once and handed over as the
/// edge-triggered interrupt it then is.
///
/// # Errors
///
/// [`VlapicError::NotInstalled`] before [`crate::install`], or
/// [`VlapicError::Apic`] if the real controller could not be asked or
/// acknowledged.
pub fn arrived(vector: Vector) -> Result<(), VlapicError> {
    let vlapic = current()?;
    let local = apic::local()?;
    // Nothing is translated, and nothing has to be. Every source the guest can
    // reach is programmed onto real hardware with the guest's own vector, so
    // the number an interrupt arrived on is already the number the guest is
    // owed — and a source the guest has masked was programmed masked and did
    // not deliver at all.
    let level = local.arrived_level(vector);
    // Counted in the controller rather than in a machine-wide word, and counted
    // unconditionally: what it answers is "did the source fire once, or is it
    // firing and nothing is taking it", and on a machine with no serial port the
    // count is the only place that answer can be read from afterwards. The line
    // it replaces was a machine-global read-modify-write on the hottest path in
    // the crate, executed on every arrival for a trace the configured level
    // discards — and its plain increment panicked in a debug build once the
    // machine had seen `u32::MAX` of them, inside an interrupt handler.
    vlapic.diagnostics().arrived(level);
    // The real controller's own in-service bank is what says whether a source
    // can go on delivering: it refuses everything of a held vector's priority or
    // lower, so a vector left in service there is a source that has stopped for
    // good, and the acknowledgement below is the only thing that ever clears it.
    trace!(
        "vlapic: {} took {vector}, {} triggered: real in service {:?}, guest requested {:?}, {} \
         in service, task priority {}, hardware {}",
        vlapic.index(),
        if level { "level" } else { "edge" },
        local.in_service_top(),
        vlapic.requested(),
        vlapic.in_service_count(),
        vlapic.task_priority(),
        vlapic.ledger().debts()
    );
    if withholdable(vector, level) {
        withhold(vlapic, local, vector);
    } else {
        acknowledge(vlapic, local, vector, level);
    }
    Ok(())
}

/// Whether real hardware's acknowledgement for an arrival on `vector` may be
/// withheld until the guest gives its own.
///
/// Only for a level-triggered arrival — an edge-triggered one is finished with
/// the moment it is taken, and there is nothing to withhold — and not even then
/// in the priority class the host keeps for itself.
///
/// That exception is the whole of what this answers, and it is about what a
/// withheld acknowledgement costs. It leaves the vector in service on the real
/// controller, which then refuses everything of the vector's own
/// interrupt-priority class or lower — and the top class is the host's: its
/// interprocessor interrupts, the doorbell that fetches a processor out of the
/// guest, and the translation shootdowns another processor blocks waiting for.
/// A debt anywhere in that class holds all of them off for as long as the guest
/// takes to acknowledge, which for a guest that never does is the life of the
/// machine.
///
/// So an arrival there is acknowledged at once and given to the guest as the
/// edge-triggered interrupt it now is: nothing is owed for it. The cost is that
/// a level line the guest points at one of those vectors is delivered again
/// until its driver quiets it, and one the guest is not accepting at all is
/// delivered again until it is. That still leaves the processor answering the
/// host between arrivals, which is exactly what withholding would not.
fn withholdable(vector: Vector, level: bool) -> bool {
    level && Priority::of(vector).class() < Priority::of(ipi::FIRST).class()
}

/// Gives the guest an interrupt and acknowledges real hardware at once.
///
/// For everything nothing is owed for: an edge-triggered arrival, which is
/// finished with the moment it is taken, and a level-triggered one on a vector
/// whose acknowledgement may not be withheld.
///
/// The acknowledgement is owed to hardware however the guest's controller
/// answered, so it is issued before the answer is looked at.
fn acknowledge(vlapic: &Vlapic, local: LocalApic, vector: Vector, level: bool) {
    let accepted = vlapic.accept(vector, Trigger::Edge);
    local.end_of_interrupt();
    match accepted {
        // A controller that was mid-reset has dropped an interrupt that reached
        // this machine, which is this hypervisor losing one rather than a guest
        // declining it.
        Accepted::Resetting => {
            vlapic.diagnostics().dropped();
            warn!(
                "vlapic: {} dropped real {vector}, which arrived while its register file was being \
                 reset",
                vlapic.index()
            );
        }
        // Level, and so retired before the guest has done anything with it. Worth
        // its own line: it is the one arrival handed over already acknowledged,
        // and the reason is nothing about the interrupt itself.
        _ if level => trace!(
            "vlapic: {} acknowledged level {vector} at once and gave it to its guest as an edge, \
             because withholding it would hold the host's own interrupts off with it",
            vlapic.index()
        ),
        Accepted::Requested | Accepted::Coalesced | Accepted::Illegal | Accepted::Refused => {
            if matches!(accepted, Accepted::Illegal | Accepted::Refused) {
                vlapic.diagnostics().declined();
            }
            trace!(
                "vlapic: {} acknowledged real {vector} at once, leaving real in service {:?}",
                vlapic.index(),
                local.in_service_top()
            );
        }
    }
}

/// Gives the guest a level-triggered interrupt and keeps real hardware's
/// acknowledgement until the guest gives its own.
fn withhold(vlapic: &Vlapic, local: LocalApic, vector: Vector) {
    // The debt is recorded before the guest is given the interrupt, so that a
    // guest which acknowledges immediately finds the debt already there.
    vlapic.ledger().owe(vector);
    match vlapic.accept(vector, Trigger::Level) {
        Accepted::Requested | Accepted::Coalesced => trace!(
            "vlapic: {} withheld real {vector}'s acknowledgement until its guest gives one, real \
             in service {:?}",
            vlapic.index(),
            local.in_service_top()
        ),
        // Not a refusal the guest made, so the debt is not one nothing will ever
        // discharge: the register file is between one guest and the next, and
        // whichever guest comes out of the reset may still acknowledge this
        // vector. Writing it off here is what would leave a debt the guest that
        // comes out of the reset could have paid.
        Accepted::Resetting => warn!(
            "vlapic: {} kept real {vector}'s acknowledgement owed: it arrived while the register \
             file was being reset",
            vlapic.index()
        ),
        refused => {
            // The guest was not given it and will therefore never acknowledge
            // it, so the only thing that could ever have discharged the debt
            // does not exist. What becomes of it is the machine's answer rather
            // than this crate's: a controller that can retire a named vector
            // retires it and stops the vector arriving again until the guest is
            // reset, and one that cannot keeps the debt — because acknowledging
            // a level line nobody has quieted clears the remote in-service state
            // of the I/O controller that sent it, and the still-asserted line
            // arrives again at once, into a controller that has just refused it.
            vlapic.ledger().abandon(vector, &local);
            vlapic.diagnostics().declined();
            // Not latched, and bounded without one on either machine: the debt
            // leaves the vector either in service on the real controller, which
            // then refuses everything of its class or lower, or blocked there —
            // so the line that produced this cannot produce another.
            warn!(
                "vlapic: {} received level {vector} but is not accepting it: {refused:?}, so real \
                 hardware goes on holding it — {}",
                vlapic.index(),
                vlapic.ledger().debts()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    //! Which arrivals may have their acknowledgement withheld is the one
    //! decision here that needs no controller, and it is the one that costs
    //! the *machine* rather than the guest if it is wrong.

    use descriptors::Vector;

    use super::withholdable;

    #[test]
    fn an_edge_arrival_owes_nothing_whatever_its_vector() {
        for number in 0x10..=u8::MAX {
            assert!(!withholdable(Vector::new(number), false));
        }
    }

    #[test]
    fn a_level_arrival_below_the_hosts_own_class_is_withheld() {
        for number in 0x10..ipi::FIRST.number() {
            assert!(
                withholdable(Vector::new(number), true),
                "vector {number:#x}"
            );
        }
    }

    #[test]
    fn a_level_arrival_in_the_hosts_own_class_is_not() {
        // Every vector of the top class, not merely the ones a handler has taken:
        // a debt at any of them holds the whole class in service, and the host's
        // interprocessor interrupts, its error vector and its spurious vector are
        // all in it.
        for number in ipi::FIRST.number()..=u8::MAX {
            assert!(
                !withholdable(Vector::new(number), true),
                "vector {number:#x}"
            );
        }
        assert!(
            !withholdable(apic::ERROR, true) && !withholdable(apic::SPURIOUS, true),
            "the controller's own two vectors are in that class as well"
        );
    }
}
