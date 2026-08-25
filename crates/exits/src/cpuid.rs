//! What the guest is told about the processor it runs on.

use log::error;
use vcpu::{Flow, Vcpu};
use x86_64::registers::control::Cr4Flags;

use crate::advance;

/// Bytes in `CPUID`, for a processor that does not report the address after an
/// intercepted instruction.
const BYTES: u64 = 2;

/// The standard feature leaf, whose ECX word advertises hypervisor presence
/// to software that checks before it goes looking for the hypervisor leaves.
const STANDARD_FEATURES: u32 = 0x1;

/// The structured extended feature leaf, whose ECX word reports protection-key
/// support and whether the operating system enabled it.
const STRUCTURED_EXTENDED_FEATURES: u32 = 0x7;

/// The extended state enumeration leaf. Its size fields are evaluated against
/// the processor's current `XCR0`, which remains the guest's value across an
/// SVM exit in this hypervisor.
const EXTENDED_STATE: u32 = 0xD;

/// The subleaf that advertises which user state components exist and the size
/// required by the components currently enabled in `XCR0`.
const EXTENDED_STATE_INFO: u32 = 0;

/// The subleaf that advertises XSAVE instruction support and the size of the
/// state enabled in `XCR0` and `IA32_XSS`.
const EXTENDED_STATE_INSTRUCTIONS: u32 = 1;

/// The hypervisor-present bit in that word.
const HYPERVISOR_PRESENT: u32 = 1 << 31;

/// `CPUID.01H:ECX[26]`: the static XSAVE capability required by OSXSAVE.
const XSAVE: u32 = 1 << 26;

/// `CPUID.01H:ECX[27]`: the current `CR4.OSXSAVE` state.
const OSXSAVE: u32 = 1 << 27;

/// `CPUID.01H:EDX[9]`: the local APIC's current enablement state.
const APIC: u32 = 1 << 9;

/// `CPUID.01H:ECX[21]`: the controller's wider face, reached through
/// model-specific registers.
const X2APIC: u32 = 1 << 21;

/// `CPUID.07H:ECX[3]`: static protection-key support.
const PKU: u32 = 1 << 3;

/// `CPUID.07H:ECX[4]`: the current `CR4.PKE` state.
const OSPKE: u32 = 1 << 4;

/// The extended feature leaf, whose ECX word advertises the virtualization
/// extension.
const EXTENDED_FEATURES: u32 = 0x8000_0001;

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
/// The extended APIC register space is hidden for a sharper reason than the
/// other two. Two of its registers are how the hypervisor settles what real
/// hardware is still holding on the guest's behalf: one retires a named vector,
/// and one decides which vectors the controller accepts at all. Both are the
/// host's own bookkeeping, on the real controller rather than the emulated one,
/// so a guest that could reach them could retire an interrupt the host is
/// accounting for or silence a vector the host has just given back. The guest's
/// emulated controller answers the whole range as reserved, and clearing this
/// bit is what stops software looking there in the first place.
pub(crate) fn exit(vcpu: &mut Vcpu) -> Flow {
    let leaf = low(vcpu.save().rax);
    let subleaf = low(vcpu.registers().rcx);
    let mut result = processor::cpuid(leaf, subleaf);
    if leaf == STANDARD_FEATURES {
        result.ecx &= !OSXSAVE;
        if Cr4Flags::from_bits_retain(vcpu.save().cr4).contains(Cr4Flags::OSXSAVE)
            && result.ecx & XSAVE != 0
        {
            result.ecx |= OSXSAVE;
        }
        // The wider controller face is withheld wherever the interrupt
        // acceleration exists but cannot drive it: a guest offered the face
        // would enter a mode whose deliveries fall back to software at the
        // very moments it believes them accelerated, so the machine presents
        // the narrower machine it can actually run. A machine with no
        // acceleration emulates the face as it emulates everything else, and
        // keeps offering it.
        if !vlapic::x2apic_offered() {
            result.ecx &= !X2APIC;
        }
        result.edx &= !APIC;
        if vlapic::apic_enabled().unwrap_or_else(|error| {
            error!("exits: could not read the guest APIC enable state: {error}");
            false
        }) {
            result.edx |= APIC;
        }
    }
    if leaf == STRUCTURED_EXTENDED_FEATURES && subleaf == 0 {
        result.ecx &= !OSPKE;
        if Cr4Flags::from_bits_retain(vcpu.save().cr4).contains(Cr4Flags::PROTECTION_KEY_USER)
            && result.ecx & PKU != 0
        {
            result.ecx |= OSPKE;
        }
    }
    // Leaf 0DH is deliberately passed through unchanged. SVM does not save
    // `XCR0` in the VMCB, so this processor is still running with the guest's
    // `XCR0` when the intercepted `CPUID` executes; the hardware therefore
    // supplies the enabled-state sizes in EBX and the related fields directly.
    if leaf == EXTENDED_STATE
        && matches!(subleaf, EXTENDED_STATE_INFO | EXTENDED_STATE_INSTRUCTIONS)
    {
        result = processor::cpuid(leaf, subleaf);
    }
    if leaf == EXTENDED_FEATURES {
        result.ecx &= !EXTENDED_APIC_SPACE;
    }
    if leaf == STANDARD_FEATURES {
        result.ecx &= !HYPERVISOR_PRESENT;
    }
    if (HYPERVISOR_LEAF_BASE..=HYPERVISOR_LEAF_LIMIT).contains(&leaf) {
        result.eax = 0;
        result.ebx = 0;
        result.ecx = 0;
        result.edx = 0;
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
