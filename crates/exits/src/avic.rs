//! The two exits a guest raises while the hardware drives its interrupt
//! controller.
//!
//! One says the hardware started delivering an interrupt between the guest's
//! own processors and could not finish; the other says the guest touched a
//! controller register the hardware does not implement. Neither can arrive
//! while the controller is driven in software — which is how it is driven
//! until this host decides otherwise — so both are answered defensively:
//! loudly, and in whichever direction the architecture says is safe for the
//! guest.

use inject::Pending;
use log::warn;
use svm::avic::{IncompleteIpiExit, UnacceleratedAccessExit};
use vcpu::{Flow, Vcpu};

/// Answers an interrupt the hardware could not finish delivering between the
/// guest's processors.
///
/// The exit is trap-like — the processor has already stepped the guest past
/// the request — so the answer is only to record what the hardware could not
/// do and go back in. Until delivery in hardware exists, this is unreachable;
/// when it is reached, the record is what says so.
pub(crate) fn incomplete_ipi(vcpu: &Vcpu) -> Flow {
    let control = vcpu.control();
    let exit = IncompleteIpiExit::from_exit_info(control.exit_info_1, control.exit_info_2);
    warn!(
        "exits: an interrupt the hardware delivers stopped: {:?}, icr {:#018x}, index {:#x}",
        exit.cause(),
        exit.icr(),
        exit.index()
    );
    Flow::Resume
}

/// Answers an access to a controller register the hardware does not
/// accelerate.
///
/// Refused with the fault the architecture owes for one. The unaccelerated
/// accesses are allow-or-fault rather than traps — the hardware never steps
/// the guest over one — so a fault, taken at the instruction that made it,
/// is the one answer that is correct for all of them.
pub(crate) fn unaccelerated_access(vcpu: &mut Vcpu, interrupts: &mut Pending) -> Flow {
    let control = vcpu.control();
    let exit = UnacceleratedAccessExit::from_exit_info(control.exit_info_1, control.exit_info_2);
    warn!(
        "exits: the guest touched a controller register the hardware does not accelerate: \
         offset {:#x}, {}",
        exit.offset(),
        if exit.is_write() { "write" } else { "read" }
    );
    crate::msr::refuse(vcpu, interrupts)
}
