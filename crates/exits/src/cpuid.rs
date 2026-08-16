//! What the guest is told about the processor it runs on.

use vcpu::{Flow, Vcpu};

use crate::advance;

/// Bytes in `CPUID`, for a processor that does not report the address after an
/// intercepted instruction.
const BYTES: u64 = 2;

/// The standard feature leaf, whose ECX word advertises hypervisor presence
/// to software that checks before it goes looking for the hypervisor leaves.
const STANDARD_FEATURES: u32 = 0x1;

/// The hypervisor-present bit in that word.
const HYPERVISOR_PRESENT: u32 = 1 << 31;

/// The extended feature leaf, whose ECX word advertises the virtualization
/// extension.
const EXTENDED_FEATURES: u32 = 0x8000_0001;

/// The virtualization extension's bit in that word.
const SVM: u32 = 1 << 2;

/// The extended APIC register space's bit in the same word.
const EXTENDED_APIC_SPACE: u32 = 1 << 3;

/// The first leaf of the range reserved for hypervisor use. Neither vendor
/// assigns architectural meaning here; it exists so a hypervisor has
/// somewhere to answer without colliding with real leaves.
const HYPERVISOR_LEAF_BASE: u32 = 0x4000_0000;

/// The last leaf of that range a guest might plausibly probe. Nothing
/// requires a hypervisor stop sooner, so the whole block is covered rather
/// than just the one or two leaves any particular hypervisor happens to use.
const HYPERVISOR_LEAF_LIMIT: u32 = 0x4000_00FF;

/// Answers the guest with the machine's own answer, less the virtualization
/// extension, the extended APIC register space, the hypervisor-present bit,
/// and anything in the hypervisor leaf range.
///
/// Hiding these is not concealment for its own sake. Every attempt the guest
/// makes to use the virtualization extension is intercepted, because a guest
/// running a guest of its own would be running one on a control block this
/// hypervisor never inspected — so a guest that was told the extension
/// exists, or went looking for a hypervisor to cooperate with, would be told
/// a thing it cannot act on, and would fail somewhere further away than
/// here. The hypervisor leaves get the same treatment as the bit that
/// announces them: a guest that skips the check and probes the range
/// directly should find it as empty as one that trusted the bit would have
/// expected.
///
/// The extended APIC register space is the same argument about a different
/// capability. The guest's interrupt controller is emulated and does not model
/// the registers that space holds, so a guest told the space is there finds
/// nothing in it — and the one thing worse than a missing capability is a
/// capability whose registers read as though something else had already claimed
/// them, which is how an operating system reads a zero out of an extended local
/// vector table entry.
pub(crate) fn exit(vcpu: &mut Vcpu) -> Flow {
    let leaf = low(vcpu.save().rax);
    let subleaf = low(vcpu.registers().rcx);
    let mut result = processor::cpuid(leaf, subleaf);
    if leaf == EXTENDED_FEATURES {
        result.ecx &= !(SVM | EXTENDED_APIC_SPACE);
    }
    if leaf == STANDARD_FEATURES {
        result.ecx &= !HYPERVISOR_PRESENT;
    }
    if (HYPERVISOR_LEAF_BASE..=HYPERVISOR_LEAF_LIMIT).contains(&leaf) {
        result.eax = 0;
        result.ebx = 0;
        result.ecx = 0;
        result.edx = 0x67;
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
