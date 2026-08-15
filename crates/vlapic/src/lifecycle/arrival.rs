//! The seam every interrupt that arrived on real hardware and belongs to a guest
//! reaches.
//!
//! What arrives is a physical vector nothing in the hypervisor claimed, which on
//! a machine whose I/O controllers are passed through means it was meant for the
//! guest. Nothing is translated, and nothing has to be: every source the guest
//! can reach is programmed onto real hardware with the guest's own vector.
//!
//! Whether real hardware may be acknowledged now is the whole of what is decided
//! here, and the controller itself is asked — it recorded, as it accepted the
//! interrupt, whether the interrupt arrived level triggered.

use core::sync::atomic::{AtomicU32, Ordering};

use descriptors::Vector;
use log::{trace, warn};

use crate::{VlapicError, machine::current, registers::{Accepted, icr::Trigger}};

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
/// # Errors
///
/// [`VlapicError::NotInstalled`] before [`install`], or [`VlapicError::Apic`]
/// if the real controller could not be asked or acknowledged.
pub fn arrived(vector: Vector) -> Result<(), VlapicError> {
    let vlapic = current()?;
    let local = apic::local()?;
    // Nothing is translated, and nothing has to be. Every source the guest can
    // reach is programmed onto real hardware with the guest's own vector, so
    // the number an interrupt arrived on is already the number the guest is
    // owed — and a source the guest has masked was programmed masked and did
    // not deliver at all.
    let level = local.arrived_level(vector);
    // The real controller's own in-service bank is what says whether a source
    // can go on delivering: it refuses everything of a held vector's priority or
    // lower, so a vector left in service there is a source that has stopped for
    // good, and the acknowledgement below is the only thing that ever clears it.
    let seen = ARRIVALS.fetch_add(1, Ordering::Relaxed) + 1;
    trace!(
        "vlapic: {} arrival {seen} of {vector}, {} triggered: real in service {:?}, guest requested \
         {:?}, {} in service, task priority {}, hardware {}",
        vlapic.index(),
        if level { "level" } else { "edge" },
        local.in_service_top(),
        vlapic.requested(),
        vlapic.in_service_count(),
        vlapic.task_priority(),
        if vlapic.ledger().is_empty() {
            "owed nothing"
        } else {
            "still owed an acknowledgement"
        }
    );
    if !level {
        vlapic.accept(vector, Trigger::Edge);
        local.end_of_interrupt();
        trace!(
            "vlapic: {} acknowledged real {vector} at once, leaving real in service {:?}",
            vlapic.index(),
            local.in_service_top()
        );
        return Ok(());
    }
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
        refused => {
            // The guest was not given it and will therefore never acknowledge
            // it, so the only thing that could ever have discharged the debt
            // does not exist. Settling here is what stops a refused interrupt
            // occupying a real in-service slot for the life of the machine,
            // blocking everything of its priority or lower on this processor.
            vlapic.ledger().release(vector);
            warn!(
                "vlapic: {} received level {vector} but is not accepting it: {refused:?}, real in \
                 service {:?}",
                vlapic.index(),
                local.in_service_top()
            );
        }
    }
    Ok(())
}


/// How many interrupts have reached [`arrived`] on this machine.
///
/// One counter for every processor rather than one each, because what it is for
/// is telling "the source fired once" from "the source is firing and nothing is
/// taking it", and a single number answers that whichever processor the
/// arrivals landed on.
static ARRIVALS: AtomicU32 = AtomicU32::new(0);
