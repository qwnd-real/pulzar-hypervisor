//! The entry checks, performed before the processor performs them.
//!
//! `VMRUN` inspects the state it has just loaded and, if any of it is a
//! combination the architecture forbids, exits immediately with
//! `VMEXIT_INVALID` — no guest instruction executed, and no indication of which
//! rule was broken. That is a hard thing to debug from: the report is the same
//! whether a control block names no address space, forgot the one mandatory
//! intercept, or has a single reserved bit set in a control register.
//!
//! So the rules are written out here and checked in software, and what comes
//! back is the rule rather than the verdict.
//!
//! # It under-reports rather than over-reports
//!
//! Several of the architecture's rules are about "any must-be-zero bit" of a
//! register whose must-be-zero bits depend on the guest's paging mode, on the
//! processor's physical address width, or on which features the silicon has.
//! Guessing wide there would refuse control blocks that are perfectly legal,
//! which is a much worse failure than missing one: a refusal stops a guest that
//! would have run, where a miss merely leaves the processor to report what it
//! was going to report anyway.
//!
//! Every check below is therefore one that holds in every mode and on every
//! processor. Where a rule has a mode-dependent part, the part that always
//! holds is checked and the rest is left alone — the address-width rules are
//! checked because a bit above the width is reserved whatever the mode, while
//! the low-order bits of a control register are not, because what they mean
//! changes with paging mode.
//!
//! [`Invalid::Unexplained`] is what remains: the processor refused a control
//! block that satisfies every rule stated here. It is a real answer and worth
//! reporting as one.

use svm::{
    ControlArea, EventKind, SaveArea,
    intercept::Intercepts2Flags,
    permissions::{IOPM_BYTES, MSRPM_BYTES},
};
use thiserror::Error;
use x86_64::registers::{
    control::{Cr0Flags, Cr4Flags},
    model_specific::EferFlags,
};

/// A rule of the architecture's that this control block breaks.
///
/// One variant per rule, named for what is wrong rather than for where it was
/// found, because the point of the whole module is to answer "why would this
/// not run" in one line.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum Invalid {
    /// The guest's extended feature register does not enable the extension.
    /// A guest that writes that register without preserving the bit disables
    /// itself, which is why this is the first rule and the most common one.
    #[error("the guest's EFER.SVME is clear")]
    SvmeClear,
    /// Caching is on while write-through is off, which is a combination the
    /// processor refuses to run in.
    #[error("the guest's CR0 has NW set with CD clear")]
    CacheDisableWithoutWriteThrough,
    /// The upper half of the first control register is reserved.
    #[error("the guest's CR0 has bits set above 31")]
    Cr0Upper,
    /// The third control register names an address wider than the processor
    /// implements.
    #[error("the guest's CR3 has bits set above physical bit {bits}")]
    Cr3TooWide {
        /// How many bits of physical address this processor implements.
        bits: u8,
    },
    /// The upper half of the fourth control register is reserved.
    #[error("the guest's CR4 has bits set above 31")]
    Cr4Upper,
    /// The upper half of the debug status register is reserved.
    #[error("the guest's DR6 has bits set above 31")]
    Dr6Upper,
    /// The upper half of the debug control register is reserved.
    #[error("the guest's DR7 has bits set above 31")]
    Dr7Upper,
    /// A reserved bit of the extended feature register is set.
    #[error("the guest's EFER has a reserved bit set")]
    EferReserved,
    /// Long mode with paging needs the physical address extension.
    #[error("the guest enables long mode and paging without CR4.PAE")]
    LongModeWithoutPae,
    /// Long mode with paging needs protected mode.
    #[error("the guest enables long mode and paging without CR0.PE")]
    LongModeWithoutProtection,
    /// A code segment cannot be both 64-bit and 32-bit at once.
    #[error("the guest's CS is both long and default-size in long mode")]
    LongAndDefaultSize,
    /// A guest allowed to enter a guest of its own could run one this
    /// hypervisor never saw, so the architecture refuses to start it.
    #[error("the VMRUN intercept is clear")]
    VmrunNotIntercepted,
    /// The port permission bitmap runs past the end of physical memory.
    #[error("the port permission bitmap ends above physical bit {bits}")]
    IoMapTooHigh {
        /// How many bits of physical address this processor implements.
        bits: u8,
    },
    /// The model-specific register permission bitmap runs past the end of
    /// physical memory.
    #[error("the register permission bitmap ends above physical bit {bits}")]
    MsrMapTooHigh {
        /// How many bits of physical address this processor implements.
        bits: u8,
    },
    /// The event to be injected uses a type the architecture reserves.
    #[error("the event to inject has a reserved type")]
    EventKindReserved,
    /// The event to be injected is an exception on a vector that names none —
    /// which includes vector 2, the non-maskable interrupt, since that is a
    /// type of its own here.
    #[error("the event to inject is an exception on vector {vector}, which names no exception")]
    EventNotAnException {
        /// The vector in question.
        vector: u8,
    },
    /// Address space zero is the host's, and no guest may be tagged with it.
    #[error("the guest's ASID is zero, which belongs to the host")]
    AsidZero,
    /// Control-flow enforcement requires write protection.
    #[error("the guest enables CR4.CET without CR0.WP")]
    ControlFlowWithoutWriteProtect,
    /// The nested table root names an address wider than the processor
    /// implements.
    #[error("the nested table root has bits set above physical bit {bits}")]
    NestedCr3TooWide {
        /// How many bits of physical address this processor implements.
        bits: u8,
    },
    /// A field of the guest's page-attribute table holds a type encoding the
    /// architecture does not define, or a reserved bit.
    #[error("the guest's PAT field {field} holds {encoding:#x}, which is not a memory type")]
    PatEncoding {
        /// Which of the eight fields.
        field: u8,
        /// What it held.
        encoding: u8,
    },
    /// Every rule above holds and the processor refused the block anyway.
    ///
    /// Not a failure of this module so much as its honest edge: the checks here
    /// are the ones that hold in every mode and on every processor, and the
    /// architecture has rules that do not. Reporting this rather than nothing
    /// is what says the block was examined and came back clean.
    #[error("the processor refused a control block that breaks no rule pulzar checks")]
    Unexplained,
}

/// The first rule this control block breaks, or `None` if it breaks none.
///
/// `bits` is how many bits of physical address the processor the guest would
/// run on implements. It is a property of the machine rather than of the block,
/// which is why it is handed in rather than read here.
///
/// One of the architecture's rules is missing on purpose: a guest may not
/// enable long mode on a processor that has none, and no processor without long
/// mode can execute this image, so there is nothing here that could be false.
#[must_use]
pub fn check(control: &ControlArea, save: &SaveArea, bits: u8) -> Option<Invalid> {
    let efer = EferFlags::from_bits_retain(save.efer);
    let cr0 = Cr0Flags::from_bits_retain(save.cr0);
    let cr4 = Cr4Flags::from_bits_retain(save.cr4);
    let paging = cr0.contains(Cr0Flags::PAGING);
    let long = efer.contains(EferFlags::LONG_MODE_ENABLE);

    if !efer.contains(EferFlags::SECURE_VIRTUAL_MACHINE_ENABLE) {
        return Some(Invalid::SvmeClear);
    }
    if cr0.contains(Cr0Flags::NOT_WRITE_THROUGH) && !cr0.contains(Cr0Flags::CACHE_DISABLE) {
        return Some(Invalid::CacheDisableWithoutWriteThrough);
    }
    if save.cr0 & UPPER_HALF != 0 {
        return Some(Invalid::Cr0Upper);
    }
    if save.cr4 & UPPER_HALF != 0 {
        return Some(Invalid::Cr4Upper);
    }
    if save.dr6 & UPPER_HALF != 0 {
        return Some(Invalid::Dr6Upper);
    }
    if save.dr7 & UPPER_HALF != 0 {
        return Some(Invalid::Dr7Upper);
    }
    if save.efer & EFER_RESERVED != 0 {
        return Some(Invalid::EferReserved);
    }
    if long && paging {
        if !cr4.contains(Cr4Flags::PHYSICAL_ADDRESS_EXTENSION) {
            return Some(Invalid::LongModeWithoutPae);
        }
        if !cr0.contains(Cr0Flags::PROTECTED_MODE_ENABLE) {
            return Some(Invalid::LongModeWithoutProtection);
        }
        if cr4.contains(Cr4Flags::PHYSICAL_ADDRESS_EXTENSION)
            && save.cs.attributes.long()
            && save.cs.attributes.default_size()
        {
            return Some(Invalid::LongAndDefaultSize);
        }
    }
    if cr4.contains(Cr4Flags::CONTROL_FLOW_ENFORCEMENT) && !cr0.contains(Cr0Flags::WRITE_PROTECT) {
        return Some(Invalid::ControlFlowWithoutWriteProtect);
    }
    if above(save.cr3, bits) {
        return Some(Invalid::Cr3TooWide { bits });
    }
    if !control
        .intercept_2
        .flags()
        .contains(Intercepts2Flags::VMRUN)
    {
        return Some(Invalid::VmrunNotIntercepted);
    }
    if control.asid == 0 {
        return Some(Invalid::AsidZero);
    }
    if let Some(invalid) = maps(control, bits) {
        return Some(invalid);
    }
    if let Some(invalid) = injection(control) {
        return Some(invalid);
    }
    if control.nested_paging.enabled() {
        if above(control.nested_cr3, bits) {
            return Some(Invalid::NestedCr3TooWide { bits });
        }
        if let Some(invalid) = pat(save.g_pat) {
            return Some(invalid);
        }
    }
    None
}

/// Whether either permission bitmap reaches an address the processor cannot
/// form.
///
/// A map is only consulted when the matching intercept is set, but the rule is
/// not conditional on that: the architecture checks where the tables *extend
/// to*, so a stale address left in a block whose intercept is clear is still a
/// refused entry.
fn maps(control: &ControlArea, bits: u8) -> Option<Invalid> {
    let ends = |base: u64, len: usize| base.saturating_add(len as u64).saturating_sub(1);
    if above(ends(control.io_permissions, IOPM_BYTES), bits) {
        return Some(Invalid::IoMapTooHigh { bits });
    }
    if above(ends(control.msr_permissions, MSRPM_BYTES), bits) {
        return Some(Invalid::MsrMapTooHigh { bits });
    }
    None
}

/// Whether the event waiting to be injected is one the guest could take.
///
/// Only two things make an injection illegal in a way that is knowable without
/// the guest's mode: a reserved type, and an exception on a vector that names
/// no exception. The third — an event impossible in the mode the guest is about
/// to run in — depends on that mode and is left to the processor.
fn injection(control: &ControlArea) -> Option<Invalid> {
    let event = control.event_injection;
    if !event.valid() {
        return None;
    }
    match event.kind() {
        EventKind::Reserved => Some(Invalid::EventKindReserved),
        EventKind::Exception if !event.vector().is_exception() => {
            Some(Invalid::EventNotAnException {
                vector: event.vector().number(),
            })
        }
        _ => None,
    }
}

/// Whether every field of a page-attribute table holds a memory type.
///
/// Only read when nested paging is on, which is the only time the processor
/// loads this register from the control block at all.
fn pat(g_pat: u64) -> Option<Invalid> {
    for field in 0..PAT_FIELDS {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "one field of the register is one byte by construction"
        )]
        let encoding = (g_pat >> (u32::from(field) * u8::BITS)) as u8;
        if !matches!(encoding, 0 | 1 | 4 | 5 | 6 | 7) {
            return Some(Invalid::PatEncoding { field, encoding });
        }
    }
    None
}

/// Whether an address has a bit set above what the processor implements.
///
/// The one rule about a physical address that holds in every paging mode: the
/// low-order bits of a control register mean different things from mode to
/// mode, but a bit above the implemented width is reserved in all of them.
fn above(address: u64, bits: u8) -> bool {
    // A processor implementing all sixty-four would leave nothing above, and the
    // shift itself would be undefined.
    u32::from(bits) < u64::BITS && address >> bits != 0
}

/// Bits of a value that lie above the low thirty-two.
const UPPER_HALF: u64 = !0 << 32;

/// The bits of the extended feature register that are reserved on every
/// processor.
///
/// Bits 63:22, 19, 16 and 9. The two most recent flags — automatic indirect
/// branch restriction and upper address ignore — are deliberately outside this
/// mask even though they are reserved on a processor that lacks them, because
/// which processor that is takes a feature query and a wrong answer here
/// refuses a legal guest.
const EFER_RESERVED: u64 = (!0 << 22) | (1 << 19) | (1 << 16) | (1 << 9);

/// How many memory-type fields a page-attribute table has.
const PAT_FIELDS: u8 = 8;

const _: () = assert!(
    EFER_RESERVED & (1 << 12) == 0 && EFER_RESERVED & (1 << 8) == 0,
    "the virtualization-enable and long-mode-enable bits are not reserved",
);
