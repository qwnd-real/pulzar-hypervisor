//! The model-specific registers the guest is answered for rather than allowed
//! to reach.
//!
//! Two kinds, and they fail differently. The register saying whether the
//! virtualization extension exists is answered with a lie the guest is entitled
//! to believe. The interrupt controller's registers are answered by the guest's
//! own emulated controller, and an access the architecture does not allow is a
//! general protection fault the guest is given rather than an error the host
//! reports.

use log::error;
use svm::{
    Event,
    msr::{VM_CR, VmCr},
    permissions::MsrAccess,
};
use vcpu::{Flow, Vcpu};

use crate::advance;

/// The exception a guest takes for a model-specific register access the
/// architecture does not allow.
const GENERAL_PROTECTION: descriptors::Vector = descriptors::Vector::new(13);

/// Bytes in `RDMSR` and in `WRMSR`, for a processor that does not report the
/// address after an intercepted instruction. The same length for both, which is
/// why one constant answers for either direction.
const BYTES: u64 = 2;

/// Tells the guest the virtualization extension is disabled, without touching
/// what the host's own register says.
///
/// The register is the one place a guest can look after `CPUID` has denied the
/// extension exists, and firmware does look. It is answered as a machine whose
/// firmware turned virtualization off, which is a state guests already know how
/// to be told about.
pub(crate) fn exit(vcpu: &mut Vcpu) -> Flow {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a model-specific register index is the low half of RCX; the architecture ignores the rest"
    )]
    let msr = vcpu.registers().rcx as u32;
    if vlapic::claims(msr) {
        return controller(vcpu, msr);
    }
    if u64::from(msr) != u64::from(VM_CR) {
        // Nothing else is intercepted, so this is the permission bitmap
        // disagreeing with this handler rather than anything the guest did.
        error!("exits: unexpected intercepted MSR {msr:#x}");
        return Flow::Leave;
    }
    match MsrAccess::from_exit_info(vcpu.control().exit_info_1) {
        MsrAccess::Read => {
            let value = VmCr::new().with_svm_disabled(true).into_bits();
            vcpu.save_mut().rax = value & u64::from(u32::MAX);
            vcpu.registers_mut().rdx = value >> u32::BITS;
        }
        // Dropped rather than refused. What the guest would be writing is the
        // host's own register, and a guest that has been told the extension is
        // disabled has nothing to write that this answer does not already
        // reflect.
        MsrAccess::Write => {}
    }
    advance(vcpu, BYTES);
    Flow::Resume
}

/// Answers an access to the guest's own interrupt controller.
///
/// The guest is given a general protection fault for anything the architecture
/// refuses — a reserved index, a read of a write-only register, a reserved bit
/// written non-zero — because that is what the access would have raised on real
/// hardware, and a guest probing its controller relies on being told no.
///
/// The instruction pointer is deliberately not advanced for a fault: the guest
/// takes the exception at the instruction that caused it, which is where its
/// handler expects to find it.
fn controller(vcpu: &mut Vcpu, msr: u32) -> Flow {
    let outcome = match MsrAccess::from_exit_info(vcpu.control().exit_info_1) {
        MsrAccess::Read => vlapic::read_msr(msr).map(|value| {
            vcpu.save_mut().rax = value & u64::from(u32::MAX);
            vcpu.registers_mut().rdx = value >> u32::BITS;
        }),
        MsrAccess::Write => {
            let value =
                (vcpu.registers().rdx << u32::BITS) | (vcpu.save().rax & u64::from(u32::MAX));
            vlapic::write_msr(msr, value)
        }
    };
    match outcome {
        Ok(()) => {
            advance(vcpu, BYTES);
            Flow::Resume
        }
        Err(vlapic::VlapicError::Fault) => {
            vcpu.control_mut().event_injection = Event::exception_with_code(GENERAL_PROTECTION, 0);
            Flow::Resume
        }
        Err(error) => {
            error!("exits: the guest's controller could not answer for {msr:#x}: {error}");
            Flow::Leave
        }
    }
}
