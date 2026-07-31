//! What the guest is told about the processor it runs on.

use vcpu::{Flow, Vcpu};

use crate::advance;

/// Bytes in `CPUID`, for a processor that does not report the address after an
/// intercepted instruction.
const BYTES: u64 = 2;

/// The extended feature leaf, whose ECX word advertises the virtualization
/// extension.
const EXTENDED_FEATURES: u32 = 0x8000_0001;

/// The virtualization extension's bit in that word.
const SVM: u32 = 1 << 2;

/// Answers the guest with the machine's own answer, less the virtualization
/// extension.
///
/// Hiding the extension is not concealment for its own sake. Every attempt the
/// guest makes to use it is intercepted, because a guest running a guest of its
/// own would be running one on a control block this hypervisor never inspected
/// — so a guest that was told the extension exists would be told a thing it
/// cannot act on, and would fail somewhere further away than here.
pub(crate) fn exit(vcpu: &mut Vcpu) -> Flow {
    let leaf = low(vcpu.save().rax);
    let subleaf = low(vcpu.registers().rcx);
    let mut result = processor::cpuid(leaf, subleaf);
    if leaf == EXTENDED_FEATURES {
        result.ecx &= !SVM;
    }
    vcpu.save_mut().rax = u64::from(result.eax);
    let registers = vcpu.registers_mut();
    registers.rbx = u64::from(result.ebx);
    registers.rcx = u64::from(result.ecx);
    registers.rdx = u64::from(result.edx);
    advance(vcpu, BYTES);
    Flow::Resume
}

/// The low half of a register, which is the whole of what a leaf number is.
///
/// Taken apart into bytes rather than truncated with a cast, so there is
/// nothing to suppress a lint about and no fallback for a case that cannot
/// happen.
fn low(value: u64) -> u32 {
    let [a, b, c, d, ..] = value.to_le_bytes();
    u32::from_le_bytes([a, b, c, d])
}
