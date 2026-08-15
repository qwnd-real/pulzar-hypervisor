//! Bringing the real controller into agreement with the guest's, when the guest
//! changes which face it uses.
//!
//! Two registers of the real controller are not the host's to choose, because the
//! I/O controllers are passed through: the guest programs them with logical
//! destinations directly, and hardware matches those against the *real* register.
//! A disagreement is an interrupt delivered to the wrong processor or to none,
//! which is how a guest ends up unable to find its own disk.
//!
//! How agreement is reached differs between the faces, and in x2APIC it is
//! reached by moving the real controller rather than by writing anything — which
//! is the whole reason [`promote`] exists.

use log::{info, trace, warn};

use crate::{
    hardware::{sources, timer},
    registers::{Vlapic, base::Mode, base::Transition},
};

/// Brings real hardware across a change of face, and says so.
///
/// The virtual half of the transition has already happened: whatever the
/// architecture does not preserve was reset, and where it does not preserve
/// anything the sources were quieted and the debts settled first. What is left
/// is to bring the machine across after it — which means taking the real
/// controller into the same face the guest just entered, and then programming
/// it from whatever the emulated controller now holds. The order is not a
/// preference: every register written below is written through whichever face
/// the real controller presents, and the logical destination in particular is
/// only reachable in one of them.
///
/// None of it disturbs a running timer. Reprogramming says what the timer
/// delivers and how fast it counts, and the count and the deadline are left
/// where they were — which is what carries a guest's armed timer across the one
/// transition the architecture preserves it across.
pub(crate) fn entered(vlapic: &Vlapic, transition: Transition) {
    match transition {
        Transition::Unchanged => return,
        // Worth a line rather than a trace: real hardware was left holding
        // something across a boundary the guest believes cleared it, and that is
        // a state nothing later in the guest's life will explain.
        Transition::Changed { quiet, settled } if !quiet || !settled => warn!(
            "vlapic: {} changed face without fully settling hardware: sources {}, \
             acknowledgements {}",
            vlapic.index(),
            if quiet { "quiet" } else { "still armed" },
            if settled { "settled" } else { "still owed" },
        ),
        Transition::Preserved | Transition::Changed { .. } => {}
    }
    promote(vlapic);
    mirror_logical_destination(vlapic);
    sources::reprogram(vlapic);
    timer::reprogram(vlapic);
    info!("vlapic: {} entered {}", vlapic.index(), vlapic.mode());
}

/// Brings the real controller's logical destination into agreement with the
/// guest's.
///
/// Necessary because the I/O controllers are passed through: the guest programs
/// them directly with logical destinations, and hardware matches those against
/// the *real* register. A disagreement is an interrupt delivered to the wrong
/// processor or to none, which is how a guest ends up unable to find its own
/// disk.
///
/// How agreement is reached is not the same in the two faces, and neither is a
/// failure. In the older face the registers are writable and the guest's values
/// are written. In x2APIC they are not writable at all — hardware derives the
/// identifier from this processor's own, and there is only the one destination
/// model — so nothing is written and nothing needs to be: the emulated
/// controller derives its answer by the architecture's rule from the *real*
/// identifier, which is the same rule applied to the same number. That is the
/// whole reason [`promote`] exists, and it is checked here rather than trusted,
/// because a silent disagreement in this register is exactly the fault this
/// function is for.
pub(crate) fn mirror_logical_destination(vlapic: &Vlapic) {
    let Ok(local) = apic::local() else {
        warn!(
            "vlapic: {} could not reach its controller to mirror its logical destination",
            vlapic.index()
        );
        return;
    };
    let wanted = vlapic.logical_destination();
    if local.set_logical_routing(vlapic.destination_format(), wanted) {
        return;
    }
    // The register is hardware's in this face. Reading it back is the only way
    // to know the two really do agree, and a mismatch means passed-through
    // interrupts are being matched against something the guest never asked for.
    let real = local.logical_destination();
    if real == wanted {
        trace!(
            "vlapic: {} answers logical destination {real:#x}, which its guest derives too",
            vlapic.index()
        );
        return;
    }
    warn!(
        "vlapic: {} answers logical destination {real:#x} and its guest believes {wanted:#x}; \
         interrupts addressed logically will not reach it",
        vlapic.index()
    );
}

/// Takes the real controller into x2APIC behind a guest that has just gone
/// there.
///
/// The one thing that makes logical destinations pass through at all. A guest
/// in x2APIC addresses interrupts by an identifier the architecture *derives*
/// from the processor's own, and programs that identifier straight into
/// passed-through I/O controllers and device messages — none of which this
/// hypervisor intercepts. Hardware then matches those against the real
/// controller's own register, which in the older face holds something written
/// by the host and spelled differently. There is no value the host could write
/// that would agree: the two faces encode a logical identifier differently, and
/// the x2APIC one is read-only. So the real controller is moved into the same
/// face, where hardware derives the identifier from the same number the guest
/// derived it from, and the two agree because they are the same computation.
///
/// Only ever into x2APIC, and only when the guest is already there. A guest
/// that switches its controller off is not followed: the host needs its own
/// controller for the doorbells and shootdowns that keep the machine running,
/// and an emulated controller that is off already refuses everything offered to
/// it.
fn promote(vlapic: &Vlapic) {
    if vlapic.mode() != Mode::X2Apic {
        return;
    }
    let Ok(local) = apic::local() else {
        return;
    };
    if local.mode() == apic::Mode::X2Apic {
        return;
    }
    match local.enter_x2apic() {
        Ok(()) => info!(
            "vlapic: {} took its real controller into x2apic behind its guest",
            vlapic.index()
        ),
        // Worth a line of its own rather than being folded into the mirror
        // warning below it: this is the reason the two will disagree, and it says
        // the machine cannot do what the guest asked rather than that something
        // went wrong doing it.
        Err(error) => warn!(
            "vlapic: {} could not take its real controller into x2apic: {error}",
            vlapic.index()
        ),
    }
}
