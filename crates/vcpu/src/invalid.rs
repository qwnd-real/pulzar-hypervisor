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
//! changes with paging mode. The interrupt controller's rules are the same
//! kind: every one of them is conditional on the controller being enabled, and
//! they need to know which controller modes the machine has and how far each
//! reaches — [`AvicLimits`], handed in for the same reason the address width
//! is.
//!
//! [`Invalid::Unexplained`] is what remains: the processor refused a control
//! block that satisfies every rule stated here. It is a real answer and worth
//! reporting as one.
//!
//! # Diagnosis after the fact, and prevention before it
//!
//! [`check`] answers about a block as it stands, which is what a refused entry
//! is explained with. [`arming`] answers about one edit that has not been made
//! yet: turning the interrupt acceleration on. That one is worth asking early
//! because the enable bits are the only part of a control block whose legality
//! turns on fields written long before them — so a block armed and then refused
//! stops a guest that could have gone on being delivered for in software.

use processor::SvmFeatures;
use svm::{
    ControlArea, EventKind, SaveArea,
    avic::{AvicPhysicalTable, X2_EXTENDED_MAX_PHYSICAL_ID, X2_MAX_PHYSICAL_ID},
    control::InterruptControl,
    intercept::Intercepts2Flags,
    msr::EFER_RESERVED,
    permissions::{IOPM_BYTES, MSRPM_BYTES},
};
use thiserror::Error;
use x86_64::registers::{
    control::{Cr0Flags, Cr4Flags},
    model_specific::EferFlags,
};

/// A rule this control block breaks.
///
/// One variant per rule, named for what is wrong rather than for where it was
/// found, because the point of the whole module is to answer "why would this
/// not run" in one line.
///
/// Most of these are the architecture's own consistency rules. Two are not:
/// [`Invalid::AvicCr8Intercepted`] and [`Invalid::AvicPointerZero`] are
/// defence-in-depth about the interrupt controller, where the architecture
/// leaves a combination legal that this hypervisor could only have produced by
/// mistake and whose consequence is a write into memory that belongs to
/// somebody else. Each says so in its own documentation; nothing else here is
/// anything but a rule of the architecture's.
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
    /// The hardware drives a guest's controller in 32-bit mode only by
    /// driving it at all: the wider mode's bit beside the enable bit and not
    /// with it names a configuration the architecture does not have.
    #[error("x2AVIC is enabled without AVIC")]
    X2AvicWithoutAvic,
    /// The hardware is asked to drive a guest's controller on a processor
    /// whose virtualization extension cannot.
    #[error("AVIC is enabled on a processor without the extension")]
    AvicUnsupported,
    /// The same for the wider mode, which is a feature of its own: the enable
    /// bit is reserved on a processor that has only the eight-bit mode, and a
    /// block setting it is refused with no rule of the architecture's broken.
    #[error("x2AVIC is enabled on a processor without it")]
    X2AvicUnsupported,
    /// The hardware delivers into a guest's controller through the guest's
    /// own second-level translation, so it cannot be asked to while that is
    /// off.
    #[error("AVIC is enabled without nested paging")]
    AvicWithoutNestedPaging,
    /// The hardware answers a guest's task-priority changes itself while it
    /// drives the controller; an intercept on the register that changes it
    /// would race the hardware.
    ///
    /// Not one of the architecture's own consistency rules: the processor
    /// refuses the combination, and this names it before the refusal.
    #[error("AVIC is enabled while writes of CR8 are intercepted")]
    AvicCr8Intercepted,
    /// The table of virtual processors names more of them than the mode can
    /// address.
    #[error("AVIC's max index {index:#x} is above the mode's limit of {limit:#x}")]
    AvicMaxIndex {
        /// What the table held.
        index: u16,
        /// What the mode allows.
        limit: u16,
    },
    /// One of the controller's pointer fields is not page aligned, or names an
    /// address the processor cannot form.
    #[error(
        "AVIC's {field} pointer {address:#x} is not a page-aligned address within physical bit {bits}"
    )]
    AvicPointerAlignment {
        /// Which of the four fields.
        field: AvicField,
        /// What it held.
        address: u64,
        /// How many bits of physical address this processor implements.
        bits: u8,
    },
    /// One of the two host-owned pointer fields is zero while the hardware is
    /// asked to drive the controller.
    ///
    /// Not one of the architecture's rules, and it cannot become one: physical
    /// page zero is a legal, page-aligned address inside every implemented
    /// width, so nothing about the value is wrong to the processor. What is
    /// wrong is whose page it is. The hardware serves the guest's controller
    /// registers out of the backing page and recomputes priorities in it, so a
    /// block entered with a zero pointer has the guest writing, and the
    /// processor recomputing, in a host page nothing gave it — the one AVIC
    /// field combination whose consequence is somebody else's memory rather
    /// than a refused entry.
    #[error("AVIC's {field} pointer is zero while the hardware is asked to drive the controller")]
    AvicPointerZero {
        /// Which of the two fields.
        field: AvicField,
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

/// Which of the interrupt controller's pointer fields a rule is about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AvicField {
    /// The base the guest's controller is reached at.
    ApicBar,
    /// The page one virtual processor's registers live in.
    BackingPage,
    /// The table translating logical destinations.
    LogicalTable,
    /// The table of virtual processors.
    PhysicalTable,
}

impl core::fmt::Display for AvicField {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::ApicBar => "APIC bar",
            Self::BackingPage => "backing page",
            Self::LogicalTable => "logical table",
            Self::PhysicalTable => "physical table",
        })
    }
}

/// How far this processor's hardware interrupt delivery reaches.
///
/// Which of the two controller modes the extension implements, and — for the
/// wider one, whose table may span more than one page of entries — the highest
/// index it can name. Both are properties of the machine rather than of a
/// control block, and a rule about the controller's fields cannot be decided
/// without them: judging a block against a number alone accepts one that
/// enables a mode the silicon does not have, where the processor sees a
/// reserved bit set and refuses the entry with nothing to say about why.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AvicLimits {
    /// Whether the extension can drive a guest's controller at all.
    pub avic: bool,
    /// The highest table index its 32-bit mode may name, or `None` on a
    /// processor that has no 32-bit mode.
    pub x2avic: Option<u16>,
}

impl AvicLimits {
    /// What the extension's own feature words say.
    ///
    /// The wider mode's limit is asked only of a processor that has the mode,
    /// which is what keeps the extended table's bit from answering on a
    /// machine with nothing for it to extend.
    #[must_use]
    pub fn of(features: SvmFeatures) -> Self {
        Self {
            avic: features.contains(SvmFeatures::AVIC),
            x2avic: features.contains(SvmFeatures::X2AVIC).then(|| {
                if features.contains(SvmFeatures::X2AVIC_EXT) {
                    X2_EXTENDED_MAX_PHYSICAL_ID
                } else {
                    X2_MAX_PHYSICAL_ID
                }
            }),
        }
    }
}

/// The first rule this control block breaks, or `None` if it breaks none.
///
/// `bits` is how many bits of physical address the processor the guest would
/// run on implements, and `limits` is how far that processor's hardware
/// interrupt delivery reaches. Both are properties of the machine rather than
/// of the block, which is why they are handed in rather than read here.
///
/// One of the architecture's rules is missing on purpose: a guest may not
/// enable long mode on a processor that has none, and no processor without long
/// mode can execute this image, so there is nothing here that could be false.
#[must_use]
pub fn check(
    control: &ControlArea,
    save: &SaveArea,
    bits: u8,
    limits: AvicLimits,
) -> Option<Invalid> {
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
    avic(
        control,
        control.interrupt_control,
        control.avic_physical_table,
        bits,
        limits,
    )
}

/// The rule that would make turning the interrupt acceleration on refuse this
/// block, or `None` if none would.
///
/// The same rules [`check`] applies, asked of an arming that has not happened
/// yet: `x2avic` says which of the two modes the acceleration would be armed
/// in, and `max_index` the extent the table would be published with. Every
/// other field is the block's own, because every other field was written
/// before this decision was reached.
///
/// Answering before the bits are set is the whole point. The processor's own
/// verdict arrives as `VMEXIT_INVALID`, which executes no guest instruction and
/// cannot be resumed from — so a block armed and then refused ends its guest,
/// where one refused here is one the host goes on delivering interrupts for in
/// software.
#[must_use]
pub fn arming(
    control: &ControlArea,
    x2avic: bool,
    max_index: u16,
    bits: u8,
    limits: AvicLimits,
) -> Option<Invalid> {
    avic(
        control,
        control
            .interrupt_control
            .with_avic_enable(true)
            .with_x2avic_enable(x2avic),
        control.avic_physical_table.with_max_index(max_index),
        bits,
        limits,
    )
}

/// Whether one of the rules about the interrupt controller's fields is
/// broken.
///
/// Consulted only when the controller is enabled in one mode or the other: a
/// block that leaves it off may hold anything in those fields, because the
/// processor reads none of them.
///
/// `interrupts` and `table` are passed rather than read out of `control` so
/// that the same rules answer for an arming that is being considered as for one
/// that has happened; every caller in this module supplies either the block's
/// own or exactly one prospective edit of it.
fn avic(
    control: &ControlArea,
    interrupts: InterruptControl,
    table: AvicPhysicalTable,
    bits: u8,
    limits: AvicLimits,
) -> Option<Invalid> {
    if !interrupts.avic_enable() && !interrupts.x2avic_enable() {
        return None;
    }
    if interrupts.x2avic_enable() && !interrupts.avic_enable() {
        return Some(Invalid::X2AvicWithoutAvic);
    }
    if !limits.avic {
        return Some(Invalid::AvicUnsupported);
    }
    if !control.nested_paging.enabled() {
        return Some(Invalid::AvicWithoutNestedPaging);
    }
    if control.intercept_control_registers.intercepts_write(CR8) {
        return Some(Invalid::AvicCr8Intercepted);
    }
    // Asked in this order because the wider mode's limit is only a number on a
    // machine that has the mode: a block enabling it anywhere else is refused
    // for the mode rather than judged against a limit that does not exist.
    let limit = if interrupts.x2avic_enable() {
        match limits.x2avic {
            Some(limit) => limit,
            None => return Some(Invalid::X2AvicUnsupported),
        }
    } else {
        XAVIC_INDEX_LIMIT
    };
    if table.max_index() > limit {
        return Some(Invalid::AvicMaxIndex {
            index: table.max_index(),
            limit,
        });
    }
    for (field, address) in [
        (AvicField::ApicBar, control.avic_apic_bar),
        (AvicField::BackingPage, control.avic_backing_page),
        (AvicField::LogicalTable, control.avic_logical_table),
        // The table's own pointer, whose low twelve bits hold the maximum
        // index and are read separately above.
        (AvicField::PhysicalTable, table.address().as_u64()),
    ] {
        if address & PAGE_MASK != 0 || above(address, bits) {
            return Some(Invalid::AvicPointerAlignment {
                field,
                address,
                bits,
            });
        }
    }
    // The two the host owns and the hardware writes through. The bar is the
    // guest's own physical address and zero is a legal one for it; the logical
    // table is legitimately zero under the wider mode, where the hardware does
    // not read it at all.
    for (field, address) in [
        (AvicField::BackingPage, control.avic_backing_page),
        (AvicField::PhysicalTable, table.address().as_u64()),
    ] {
        if address == 0 {
            return Some(Invalid::AvicPointerZero { field });
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
/// loads this register from the control block at all — and asked of a guest's
/// write of the register before it is stored, so that a value the entry check
/// would refuse becomes the fault the guest is owed instead.
pub(crate) fn pat(g_pat: u64) -> Option<Invalid> {
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

/// How many memory-type fields a page-attribute table has.
const PAT_FIELDS: u8 = 8;

/// The task-priority control register, whose writes the architecture refuses
/// to intercept while the hardware drives the guest's interrupt controller.
const CR8: u8 = 8;

/// The highest table index an eight-bit controller may name.
///
/// One more than a table entry can usefully hold, because the all-ones
/// identifier means "every processor" — the architecture still permits it, so
/// this is the entry check's limit rather than the useful one.
const XAVIC_INDEX_LIMIT: u16 = 0xFF;

/// Bits of an address below page alignment.
const PAGE_MASK: u64 = 0xFFF;

const _: () = assert!(
    EFER_RESERVED & EferFlags::SECURE_VIRTUAL_MACHINE_ENABLE.bits() == 0
        && EFER_RESERVED & EferFlags::LONG_MODE_ENABLE.bits() == 0,
    "the virtualization-enable and long-mode-enable bits are not reserved",
);

#[cfg(test)]
mod tests {
    use svm::{
        Vmcb,
        avic::{AvicPhysicalTable, X2_EXTENDED_MAX_PHYSICAL_ID, X2_MAX_PHYSICAL_ID},
        control::NestedPagingControl,
        intercept::{ControlRegisterIntercepts, Intercepts2},
    };
    use x86_64::PhysAddr;

    use super::*;

    /// How many bits of physical address these tests assume.
    const BITS: u8 = 48;

    /// A machine with both controller modes and the extended table, which is
    /// what most of these tests want out of the way.
    const BOTH_MODES: AvicLimits = AvicLimits {
        avic: true,
        x2avic: Some(X2_EXTENDED_MAX_PHYSICAL_ID),
    };

    /// Where the fixture's backing page is. Any non-zero page-aligned address
    /// inside the width will do; what matters is that it is not zero, which is
    /// a rule of its own.
    const BACKING: u64 = 0x0001_0000;

    /// Where the fixture's table of virtual processors is.
    const TABLE: u64 = 0x0002_0000;

    /// A control block that satisfies every rule, with the controller driving
    /// interrupts in eight-bit mode.
    fn avic_enabled() -> Vmcb {
        let mut vmcb = Vmcb::zeroed();
        vmcb.save.efer = EferFlags::SECURE_VIRTUAL_MACHINE_ENABLE.bits();
        vmcb.control.intercept_2 = Intercepts2::from_flags(Intercepts2Flags::VMRUN);
        vmcb.control.asid = 1;
        vmcb.control.nested_paging = NestedPagingControl::new().with_enabled(true);
        vmcb.control.interrupt_control = vmcb.control.interrupt_control.with_avic_enable(true);
        vmcb.control.avic_backing_page = BACKING;
        vmcb.control.avic_physical_table = table(0);
        vmcb
    }

    /// The fixture's table of virtual processors, with this largest valid
    /// index.
    fn table(max_index: u16) -> AvicPhysicalTable {
        AvicPhysicalTable::new()
            .with_max_index(max_index)
            .with_address(PhysAddr::new(TABLE))
    }

    /// The first rule this block breaks on a machine with both modes.
    fn verdict(vmcb: &Vmcb) -> Option<Invalid> {
        verdict_on(vmcb, BOTH_MODES)
    }

    /// The same, judged on this machine.
    fn verdict_on(vmcb: &Vmcb, limits: AvicLimits) -> Option<Invalid> {
        super::check(&vmcb.control, &vmcb.save, BITS, limits)
    }

    #[test]
    fn a_block_the_controller_drives_through_is_accepted() {
        assert_eq!(verdict(&avic_enabled()), None);
    }

    #[test]
    fn x2avic_needs_avic_beside_it() {
        let mut vmcb = avic_enabled();
        vmcb.control.interrupt_control = vmcb
            .control
            .interrupt_control
            .with_avic_enable(false)
            .with_x2avic_enable(true);
        assert_eq!(verdict(&vmcb), Some(Invalid::X2AvicWithoutAvic));
    }

    #[test]
    fn x2avic_with_both_bits_is_a_mode_not_a_mistake() {
        let mut vmcb = avic_enabled();
        vmcb.control.interrupt_control = vmcb.control.interrupt_control.with_x2avic_enable(true);
        assert_eq!(verdict(&vmcb), None);
    }

    #[test]
    fn the_controller_needs_the_processor_to_have_one() {
        let vmcb = avic_enabled();
        assert_eq!(
            verdict_on(
                &vmcb,
                AvicLimits {
                    avic: false,
                    x2avic: None,
                }
            ),
            Some(Invalid::AvicUnsupported),
            "a machine whose extension cannot drive a controller at all"
        );
    }

    #[test]
    fn the_wider_mode_needs_the_processor_to_have_it() {
        let mut vmcb = avic_enabled();
        vmcb.control.interrupt_control = vmcb.control.interrupt_control.with_x2avic_enable(true);
        // The eight-bit mode alone: the wider mode's enable bit is reserved
        // here, and judging its table against a limit that does not exist is
        // exactly what accepted a block the processor refuses.
        assert_eq!(
            verdict_on(
                &vmcb,
                AvicLimits {
                    avic: true,
                    x2avic: None,
                }
            ),
            Some(Invalid::X2AvicUnsupported),
        );
    }

    #[test]
    fn the_controller_needs_nested_paging() {
        let mut vmcb = avic_enabled();
        vmcb.control.nested_paging = NestedPagingControl::new();
        assert_eq!(verdict(&vmcb), Some(Invalid::AvicWithoutNestedPaging));
    }

    #[test]
    fn the_controller_races_an_intercepted_task_priority() {
        let mut vmcb = avic_enabled();
        vmcb.control.intercept_control_registers = ControlRegisterIntercepts::EMPTY.with_write(CR8);
        assert_eq!(verdict(&vmcb), Some(Invalid::AvicCr8Intercepted));
    }

    #[test]
    fn an_eight_bit_table_cannot_name_two_hundred_and_fifty_six_processors() {
        let mut vmcb = avic_enabled();
        vmcb.control.avic_physical_table = table(XAVIC_INDEX_LIMIT + 1);
        assert_eq!(
            verdict(&vmcb),
            Some(Invalid::AvicMaxIndex {
                index: XAVIC_INDEX_LIMIT + 1,
                limit: XAVIC_INDEX_LIMIT,
            })
        );
    }

    #[test]
    fn every_modes_index_limit_is_the_boundary_it_says_it_is() {
        // Pinned as literals, because a wrong hex digit here is a machine that
        // either refuses a block the processor would have run or arms one it
        // will not: 0xFF is the eight-bit mode's broadcast identifier, which
        // the architecture still permits in this field, 0x1FF one page of
        // entries, and 0xFFF the eight the extended table may span. Nothing is
        // above the last of them — the field is twelve bits wide, so the
        // extended mode's largest index is also the largest a control block can
        // express.
        for (x2avic, limit, over) in [
            (false, 0x0FF_u16, Some(0x100_u16)),
            (true, 0x1FF, Some(0x200)),
            (true, 0xFFF, None),
        ] {
            let limits = AvicLimits {
                avic: true,
                x2avic: x2avic.then_some(limit),
            };
            let mut vmcb = avic_enabled();
            vmcb.control.interrupt_control =
                vmcb.control.interrupt_control.with_x2avic_enable(x2avic);
            vmcb.control.avic_physical_table = table(limit);
            assert_eq!(verdict_on(&vmcb, limits), None, "{limit:#x}");
            if let Some(over) = over {
                vmcb.control.avic_physical_table = table(over);
                assert_eq!(
                    verdict_on(&vmcb, limits),
                    Some(Invalid::AvicMaxIndex { index: over, limit }),
                    "{over:#x}"
                );
            }
        }
    }

    #[test]
    fn a_thirty_two_bit_table_is_judged_against_the_machines_limit() {
        let mut vmcb = avic_enabled();
        vmcb.control.interrupt_control = vmcb.control.interrupt_control.with_x2avic_enable(true);
        vmcb.control.avic_physical_table = table(0x200);
        assert_eq!(
            verdict_on(
                &vmcb,
                AvicLimits {
                    avic: true,
                    x2avic: Some(X2_MAX_PHYSICAL_ID),
                }
            ),
            Some(Invalid::AvicMaxIndex {
                index: 0x200,
                limit: X2_MAX_PHYSICAL_ID,
            })
        );
        assert_eq!(verdict(&vmcb), None);
    }

    #[test]
    fn a_pointer_must_be_page_aligned() {
        let mut vmcb = avic_enabled();
        vmcb.control.avic_apic_bar = 0x1001;
        assert_eq!(
            verdict(&vmcb),
            Some(Invalid::AvicPointerAlignment {
                field: AvicField::ApicBar,
                address: 0x1001,
                bits: BITS,
            })
        );
    }

    #[test]
    fn a_pointer_must_not_run_past_physical_memory() {
        let mut vmcb = avic_enabled();
        vmcb.control.avic_backing_page = 1 << BITS;
        assert_eq!(
            verdict(&vmcb),
            Some(Invalid::AvicPointerAlignment {
                field: AvicField::BackingPage,
                address: 1 << BITS,
                bits: BITS,
            })
        );
    }

    #[test]
    fn the_two_pointers_the_hardware_writes_through_may_not_be_zero() {
        let mut vmcb = avic_enabled();
        vmcb.control.avic_backing_page = 0;
        assert_eq!(
            verdict(&vmcb),
            Some(Invalid::AvicPointerZero {
                field: AvicField::BackingPage,
            })
        );

        // The index bits are not part of the address, so a table pointer of
        // nothing is one whose page number is zero however many processors it
        // claims to describe.
        let mut vmcb = avic_enabled();
        vmcb.control.avic_physical_table = AvicPhysicalTable::new().with_max_index(7);
        assert_eq!(
            verdict(&vmcb),
            Some(Invalid::AvicPointerZero {
                field: AvicField::PhysicalTable,
            })
        );
    }

    #[test]
    fn the_bar_and_the_logical_table_are_legitimately_zero() {
        // The bar is a guest physical address, and the logical table is a
        // structure the wider mode does not read — so neither is a host page
        // the hardware would write through.
        let mut vmcb = avic_enabled();
        vmcb.control.avic_apic_bar = 0;
        vmcb.control.avic_logical_table = 0;
        assert_eq!(verdict(&vmcb), None);
    }

    #[test]
    fn a_block_with_the_controller_off_may_hold_a_zero_pointer() {
        let mut vmcb = avic_enabled();
        vmcb.control.interrupt_control = vmcb.control.interrupt_control.with_avic_enable(false);
        vmcb.control.avic_backing_page = 0;
        vmcb.control.avic_physical_table = AvicPhysicalTable::new();
        assert_eq!(verdict(&vmcb), None);
    }

    #[test]
    fn the_table_pointer_is_judged_without_its_index_bits() {
        let mut vmcb = avic_enabled();
        vmcb.control.avic_physical_table = AvicPhysicalTable::new()
            .with_max_index(3)
            .with_address(PhysAddr::new(0x2000));
        assert_eq!(verdict(&vmcb), None);

        vmcb.control.avic_physical_table = AvicPhysicalTable::new()
            .with_max_index(3)
            .with_address(PhysAddr::new(1 << BITS));
        assert_eq!(
            verdict(&vmcb),
            Some(Invalid::AvicPointerAlignment {
                field: AvicField::PhysicalTable,
                address: 1 << BITS,
                bits: BITS,
            })
        );
    }

    #[test]
    fn a_block_that_leaves_the_controller_off_is_not_judged_on_its_fields() {
        let mut vmcb = avic_enabled();
        vmcb.control.interrupt_control = vmcb.control.interrupt_control.with_avic_enable(false);
        vmcb.control.avic_apic_bar = 1;
        // An unaligned bar, an index past the eight-bit mode's limit, and a
        // racing intercept: each would be a verdict of its own if the
        // controller were on.
        vmcb.control.avic_physical_table = table(0xFFF);
        vmcb.control.intercept_control_registers = ControlRegisterIntercepts::EMPTY.with_write(CR8);
        assert_eq!(verdict(&vmcb), None);
    }

    #[test]
    fn an_arming_is_judged_before_it_happens() {
        // The block carries the acceleration off, which is what every block
        // does until an entry boundary turns it on — so the rules are inert
        // against it and answer only about the arming being considered.
        let mut vmcb = avic_enabled();
        vmcb.control.interrupt_control = vmcb.control.interrupt_control.with_avic_enable(false);
        assert_eq!(verdict(&vmcb), None);
        assert_eq!(
            super::arming(&vmcb.control, false, 0, BITS, BOTH_MODES),
            None
        );
        assert_eq!(
            super::arming(&vmcb.control, false, 0x100, BITS, BOTH_MODES),
            Some(Invalid::AvicMaxIndex {
                index: 0x100,
                limit: XAVIC_INDEX_LIMIT,
            }),
            "the arming's own extent, not the one the block still carries"
        );
        assert_eq!(
            super::arming(&vmcb.control, true, 0x100, BITS, BOTH_MODES),
            None,
            "the same extent is inside the wider mode's limit"
        );
        assert_eq!(
            super::arming(
                &vmcb.control,
                true,
                0,
                BITS,
                AvicLimits {
                    avic: true,
                    x2avic: None,
                }
            ),
            Some(Invalid::X2AvicUnsupported),
        );
    }

    #[test]
    fn an_arming_sees_the_fields_the_block_already_carries() {
        let mut vmcb = avic_enabled();
        vmcb.control.interrupt_control = vmcb.control.interrupt_control.with_avic_enable(false);
        vmcb.control.avic_backing_page = 0;
        assert_eq!(
            super::arming(&vmcb.control, false, 0, BITS, BOTH_MODES),
            Some(Invalid::AvicPointerZero {
                field: AvicField::BackingPage,
            }),
        );
    }

    #[test]
    fn a_machine_with_only_the_narrow_mode_has_no_wider_limit() {
        let narrow = AvicLimits::of(SvmFeatures::AVIC);
        assert_eq!(
            narrow,
            AvicLimits {
                avic: true,
                x2avic: None,
            }
        );
        // The extended table's bit without the mode it extends is not a limit
        // of any kind: the mode is what the bit is about.
        assert_eq!(
            AvicLimits::of(SvmFeatures::AVIC | SvmFeatures::X2AVIC_EXT),
            narrow
        );
    }

    #[test]
    fn the_wider_modes_limit_follows_the_extended_table() {
        assert_eq!(
            AvicLimits::of(SvmFeatures::AVIC | SvmFeatures::X2AVIC).x2avic,
            Some(X2_MAX_PHYSICAL_ID)
        );
        assert_eq!(
            AvicLimits::of(SvmFeatures::AVIC | SvmFeatures::X2AVIC | SvmFeatures::X2AVIC_EXT)
                .x2avic,
            Some(X2_EXTENDED_MAX_PHYSICAL_ID)
        );
        assert!(
            !AvicLimits::of(SvmFeatures::empty()).avic,
            "a processor whose extension cannot drive a controller at all"
        );
    }
}
