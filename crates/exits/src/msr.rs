//! The one model-specific register the guest is answered for rather than
//! allowed to reach.

use log::error;
use svm::{
    msr::{VM_CR, VmCr},
    permissions::MsrAccess,
};
use vcpu::{Flow, Vcpu};

use crate::advance;

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
    let msr = vcpu.registers().rcx;
    if msr != u64::from(VM_CR) {
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
