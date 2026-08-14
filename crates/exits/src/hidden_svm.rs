//! Refusing SVM instructions the guest has been told do not exist.

use descriptors::Vector;
use inject::Pending;
use svm::Event;
use vcpu::{Flow, Vcpu};

/// Gives the guest the invalid-opcode exception for an intercepted SVM
/// instruction.
///
/// The guest-visible `EFER.SVME` bit is clear, so every SVM instruction faults
/// before executing. The host keeps the real bit set only because VMRUN
/// requires it for the outer guest.
pub(crate) fn refuse(vcpu: &mut Vcpu, interrupts: &mut Pending) -> Flow {
    interrupts.raise_exception(vcpu, Event::exception(Vector::INVALID_OPCODE));
    Flow::Resume
}
