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
//! one of them needs a limit only the machine knows, which is handed in for
//! the same reason the address width is.
//!
//! [`Invalid::Unexplained`] is what remains: the processor refused a control
//! block that satisfies every rule stated here. It is a real answer and worth
//! reporting as one.

use svm::{
    ControlArea, EventKind, SaveArea,
    intercept::Intercepts2Flags,
    msr::EFER_RESERVED,
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
    /// The hardware drives a guest's controller in 32-bit mode only by
    /// driving it at all: the wider mode's bit beside the enable bit and not
    /// with it names a configuration the architecture does not have.
    #[error("x2AVIC is enabled without AVIC")]
    X2AvicWithoutAvic,
    /// The hardware delivers into a guest's controller through the guest's
    /// own second-level translation, so it cannot be asked to while that is
    /// off.
    #[error("AVIC is enabled without nested paging")]
    AvicWithoutNestedPaging,
    /// The hardware answers a guest's task-priority changes itself while it
    /// drives the controller; an intercept on the register that changes it
    /// would race the hardware.
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

/// The first rule this control block breaks, or `None` if it breaks none.
///
/// `bits` is how many bits of physical address the processor the guest would
/// run on implements, and `x2avic_limit` is the highest table index that
/// processor's 32-bit controller mode can name. Both are properties of the
/// machine rather than of the block, which is why they are handed in rather
/// than read here.
///
/// One of the architecture's rules is missing on purpose: a guest may not
/// enable long mode on a processor that has none, and no processor without long
/// mode can execute this image, so there is nothing here that could be false.
#[must_use]
pub fn check(
    control: &ControlArea,
    save: &SaveArea,
    bits: u8,
    x2avic_limit: u16,
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
    avic(control, bits, x2avic_limit)
}

/// Whether one of the rules about the interrupt controller's fields is
/// broken.
///
/// Consulted only when the controller is enabled in one mode or the other: a
/// block that leaves it off may hold anything in those fields, because the
/// processor reads none of them.
fn avic(control: &ControlArea, bits: u8, x2avic_limit: u16) -> Option<Invalid> {
    let interrupts = control.interrupt_control;
    if !interrupts.avic_enable() && !interrupts.x2avic_enable() {
        return None;
    }
    if interrupts.x2avic_enable() && !interrupts.avic_enable() {
        return Some(Invalid::X2AvicWithoutAvic);
    }
    if !control.nested_paging.enabled() {
        return Some(Invalid::AvicWithoutNestedPaging);
    }
    if control.intercept_control_registers.intercepts_write(CR8) {
        return Some(Invalid::AvicCr8Intercepted);
    }
    let table = control.avic_physical_table;
    let limit = if interrupts.x2avic_enable() {
        x2avic_limit
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

    /// The 32-bit mode's limit these tests hand in.
    const X2_LIMIT: u16 = X2_EXTENDED_MAX_PHYSICAL_ID;

    /// A control block that satisfies every rule, with the controller driving
    /// interrupts in eight-bit mode.
    fn avic_enabled() -> Vmcb {
        let mut vmcb = Vmcb::zeroed();
        vmcb.save.efer = EferFlags::SECURE_VIRTUAL_MACHINE_ENABLE.bits();
        vmcb.control.intercept_2 = Intercepts2::from_flags(Intercepts2Flags::VMRUN);
        vmcb.control.asid = 1;
        vmcb.control.nested_paging = NestedPagingControl::new().with_enabled(true);
        vmcb.control.interrupt_control = vmcb.control.interrupt_control.with_avic_enable(true);
        vmcb
    }

    /// The first rule this block breaks, judged against this module's own
    /// constants.
    fn verdict(vmcb: &Vmcb) -> Option<Invalid> {
        super::check(&vmcb.control, &vmcb.save, BITS, X2_LIMIT)
    }

    /// The same, judged against this 32-bit limit.
    fn verdict_with_limit(vmcb: &Vmcb, x2avic_limit: u16) -> Option<Invalid> {
        super::check(&vmcb.control, &vmcb.save, BITS, x2avic_limit)
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
        vmcb.control.avic_physical_table =
            AvicPhysicalTable::new().with_max_index(XAVIC_INDEX_LIMIT + 1);
        assert_eq!(
            verdict(&vmcb),
            Some(Invalid::AvicMaxIndex {
                index: XAVIC_INDEX_LIMIT + 1,
                limit: XAVIC_INDEX_LIMIT,
            })
        );
    }

    #[test]
    fn a_thirty_two_bit_table_is_judged_against_the_machines_limit() {
        let mut vmcb = avic_enabled();
        vmcb.control.interrupt_control = vmcb.control.interrupt_control.with_x2avic_enable(true);
        vmcb.control.avic_physical_table = AvicPhysicalTable::new().with_max_index(0x200);
        assert_eq!(
            verdict_with_limit(&vmcb, X2_MAX_PHYSICAL_ID),
            Some(Invalid::AvicMaxIndex {
                index: 0x200,
                limit: X2_MAX_PHYSICAL_ID,
            })
        );
        assert_eq!(verdict_with_limit(&vmcb, X2_LIMIT), None);
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
        vmcb.control.avic_physical_table = AvicPhysicalTable::new().with_max_index(0xFFF);
        vmcb.control.intercept_control_registers = ControlRegisterIntercepts::EMPTY.with_write(CR8);
        assert_eq!(verdict(&vmcb), None);
    }
}
