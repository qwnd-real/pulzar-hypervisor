//! What the processor is told about a guest before it runs, and what it says
//! back when the guest stops.
//!
//! The first kibibyte of a guest's control block, and the half a hypervisor
//! spends nearly all of its time in. It answers three questions the processor
//! asks on entry — which of the guest's actions must come back to us, which
//! address space and interrupt state to run it in, and how much of this it may
//! take from its own cache — and carries three answers back on exit: why the
//! guest stopped, what it was doing, and what it was in the middle of
//! delivering when it stopped.
//!
//! # Not every field is read every time
//!
//! Some of these are ours to write and the processor only reads: the intercept
//! vectors, the address-space identifier, the permission map addresses. Some
//! are the processor's to write and we only read: the exit code and the two
//! exit-information fields. And a few go both ways — the virtual interrupt
//! state is loaded on entry and written back on exit, which is how a guest's
//! own view of its task priority survives a round trip.
//!
//! The distinction matters because it decides what is safe to leave alone.
//! A field the processor writes is meaningless until it has run a guest; a
//! field it only reads keeps whatever we last put there.
//!
//! # The extension that is not modelled here
//!
//! Several fields belong to encrypted virtualization, which pulzar does not
//! implement. They are present as plain values because the layout is not the
//! architecture's layout without them, and leaving a hole would put every
//! field after it at the wrong offset. They are documented as what they are
//! and are meant to stay zero.

use crate::{
    CleanBits, Event,
    avic::AvicPhysicalTable,
    event::InterruptState,
    exit::ExitCode,
    intercept::{
        ControlRegisterIntercepts, DebugRegisterIntercepts, ErapControl, ExceptionIntercepts,
        Intercepts1, Intercepts2, Intercepts3, TlbControl,
    },
};

/// Bytes the control area occupies, which the architecture fixes regardless of
/// how much of it is defined.
pub const CONTROL_AREA_BYTES: usize = 0x400;

/// How many bytes at the end of the control area are left to the hypervisor.
///
/// The processor will never use these, which makes them the one place in the
/// block where software may keep something of its own.
pub const HOST_BYTES: usize = 0x20;

/// How the guest runs, and how it stopped.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct ControlArea {
    /// Which of the guest's reads and writes of a control register come back
    /// to us.
    pub intercept_control_registers: ControlRegisterIntercepts,
    /// Which of the guest's reads and writes of a debug register come back to
    /// us.
    pub intercept_debug_registers: DebugRegisterIntercepts,
    /// Which exceptions raised in the guest come back to us instead of being
    /// delivered through its own descriptor table.
    pub intercept_exceptions: ExceptionIntercepts,
    /// Interrupts, descriptor-table access, and the older instruction
    /// intercepts.
    pub intercept_1: Intercepts1,
    /// The virtualization instructions, and the write traps that report a
    /// register change after the guest's instruction has already taken effect.
    pub intercept_2: Intercepts2,
    /// The newer intercepts: broadcast invalidation, bus locks, and the halt
    /// that only traps when nothing is pending.
    pub intercept_3: Intercepts3,
    reserved_0x018: [u8; 0x24],
    /// How many cycles apart two `PAUSE` instructions may be before the filter
    /// stops treating them as a spin.
    pub pause_filter_threshold: u16,
    /// How many filtered `PAUSE` instructions the guest may execute before the
    /// intercept fires.
    pub pause_filter_count: u16,
    /// Physical address of the port permission bitmap. The low twelve bits are
    /// ignored, so the map must be page aligned.
    pub io_permissions: u64,
    /// Physical address of the model-specific register permission bitmap, with
    /// the same alignment rule.
    pub msr_permissions: u64,
    /// Added to the timestamp counter the guest reads, so a guest can be given
    /// a clock that does not jump when it is moved or resumed.
    pub tsc_offset: u64,
    /// Which address space the guest's translations are tagged with. Zero is
    /// the host's own and must not be given to a guest.
    pub asid: u32,
    /// What to flush on the way in.
    pub tlb_control: TlbControl,
    /// What to do with the return-address predictor on the way in.
    pub erap_control: ErapControl,
    reserved_0x05e: [u8; 2],
    /// The guest's virtual interrupt state: its task priority, whether an
    /// interrupt is waiting for it, and which parts of interrupt handling the
    /// processor should virtualize. Loaded on entry and written back on exit.
    pub interrupt_control: InterruptControl,
    /// Whether the guest is between an instruction that blocks interrupts and
    /// the one after it, where an interrupt must not be delivered.
    pub interrupt_state: InterruptState,
    /// Why the guest stopped. Written by the processor.
    pub exit_code: ExitCode,
    /// What the guest was doing, decoded differently for each exit code.
    /// Written by the processor.
    pub exit_info_1: u64,
    /// The rest of what the guest was doing — a faulting address for a nested
    /// page fault, the address after the instruction for a port access.
    /// Written by the processor.
    pub exit_info_2: u64,
    /// What the guest was in the middle of delivering through its own
    /// descriptor table when it stopped, if anything. Written by the processor,
    /// and the only record of an event that is otherwise lost.
    pub exit_interrupt_info: Event,
    /// Whether the guest's own page tables are translated by a second set of
    /// tables, and the other translation behaviour that travels with that.
    pub nested_paging: NestedPagingControl,
    /// Physical address the guest's interrupt controller registers appear at
    /// when the hardware drives them on its behalf.
    pub avic_apic_bar: u64,
    /// Guest physical address of the communication block used by the
    /// encrypted-virtualization extension, which pulzar does not implement.
    pub ghcb_gpa: u64,
    /// An exception or interrupt to deliver to the guest before its first
    /// instruction, bypassing every intercept check.
    pub event_injection: Event,
    /// Root of the second set of page tables, translating what the guest
    /// believes are physical addresses into real ones.
    pub nested_cr3: u64,
    /// Which pieces of state the processor should virtualize for the guest
    /// rather than trapping.
    pub virtualization_control: VirtualizationControl,
    /// Which groups of this block the processor may take from its own cache
    /// instead of reading again.
    pub clean: CleanBits,
    reserved_0x0c4: u32,
    /// Address of the instruction after the one that caused the exit, where the
    /// processor can supply it. It is what makes stepping a guest past an
    /// intercepted instruction possible without decoding the instruction.
    pub next_rip: u64,
    /// How many bytes of the guest's instruction stream the processor managed
    /// to fetch, or zero when it fetched none.
    pub fetched_bytes: u8,
    /// The guest instruction bytes themselves, filled in only for faults on
    /// data access. An instruction is at most fifteen bytes, which is why there
    /// are fifteen of them.
    pub instruction_bytes: [u8; 15],
    /// Physical address of the page the guest's interrupt controller registers
    /// are backed by.
    pub avic_backing_page: u64,
    reserved_0x0e8: [u8; 8],
    /// Physical address of the table mapping a logical interrupt controller
    /// identifier to a physical one.
    pub avic_logical_table: u64,
    /// Physical address of the table of the guest's virtual processors, and the
    /// highest index in it that is valid.
    pub avic_physical_table: AvicPhysicalTable,
    reserved_0x100: [u8; 8],
    /// Physical address of the encrypted state area, part of an extension
    /// pulzar does not implement.
    pub vmsa_pointer: u64,
    /// Accumulator value handed across an encrypted guest's explicit exit, part
    /// of an extension pulzar does not implement.
    pub vmgexit_rax: u64,
    /// Privilege level of an encrypted guest's explicit exit, part of an
    /// extension pulzar does not implement.
    pub vmgexit_cpl: u8,
    reserved_0x119: [u8; 7],
    /// How many bus locks the guest may take before the intercept fires. The
    /// processor writes back what is left of it on exit.
    pub bus_lock_threshold: u16,
    reserved_0x122: [u8; 0x12],
    /// Whether the processor should merge requested interrupts into an
    /// encrypted guest's own pending set, part of an extension pulzar does not
    /// implement.
    pub update_irr: u32,
    /// Which features an encrypted guest is permitted, part of an extension
    /// pulzar does not implement.
    pub allowed_sev_features: u64,
    /// Which features an encrypted guest is running with, part of an extension
    /// pulzar does not implement.
    pub guest_sev_features: u64,
    reserved_0x148: [u8; 8],
    /// Interrupts to merge into an encrypted guest's pending set, part of an
    /// extension pulzar does not implement.
    pub requested_irr: [u32; 8],
    reserved_0x170: [u8; 0x270],
    /// Bytes at the end of the control area the processor will never use, left
    /// for the hypervisor to keep whatever it likes in.
    pub host: [u8; HOST_BYTES],
}

impl ControlArea {
    /// A control area with every byte zero.
    ///
    /// Every reserved byte is zero as the architecture requires, no intercept
    /// is set, and no clean bit is set — which is exactly what a block the
    /// processor has not seen before must say, since it has nothing cached that
    /// could be reused.
    #[must_use]
    pub const fn zeroed() -> Self {
        Self {
            intercept_control_registers: ControlRegisterIntercepts::EMPTY,
            intercept_debug_registers: DebugRegisterIntercepts::EMPTY,
            intercept_exceptions: ExceptionIntercepts::empty(),
            intercept_1: Intercepts1::empty(),
            intercept_2: Intercepts2::EMPTY,
            intercept_3: Intercepts3::empty(),
            reserved_0x018: [0; 0x24],
            pause_filter_threshold: 0,
            pause_filter_count: 0,
            io_permissions: 0,
            msr_permissions: 0,
            tsc_offset: 0,
            asid: 0,
            tlb_control: TlbControl::DoNothing,
            erap_control: ErapControl::empty(),
            reserved_0x05e: [0; 2],
            interrupt_control: InterruptControl::new(),
            interrupt_state: InterruptState::new(),
            exit_code: ExitCode::from_bits(0),
            exit_info_1: 0,
            exit_info_2: 0,
            exit_interrupt_info: Event::none(),
            nested_paging: NestedPagingControl::new(),
            avic_apic_bar: 0,
            ghcb_gpa: 0,
            event_injection: Event::none(),
            nested_cr3: 0,
            virtualization_control: VirtualizationControl::new(),
            clean: CleanBits::nothing_cached(),
            reserved_0x0c4: 0,
            next_rip: 0,
            fetched_bytes: 0,
            instruction_bytes: [0; 15],
            avic_backing_page: 0,
            reserved_0x0e8: [0; 8],
            avic_logical_table: 0,
            avic_physical_table: AvicPhysicalTable::new(),
            reserved_0x100: [0; 8],
            vmsa_pointer: 0,
            vmgexit_rax: 0,
            vmgexit_cpl: 0,
            reserved_0x119: [0; 7],
            bus_lock_threshold: 0,
            reserved_0x122: [0; 0x12],
            update_irr: 0,
            allowed_sev_features: 0,
            guest_sev_features: 0,
            reserved_0x148: [0; 8],
            requested_irr: [0; 8],
            reserved_0x170: [0; 0x270],
            host: [0; HOST_BYTES],
        }
    }

    /// The instruction bytes the processor managed to fetch.
    ///
    /// Only meaningful after an exit that fills them in — a nested page fault
    /// or an intercepted page fault on a data access. Every other exit clears
    /// the count to zero, which this reports as an empty slice rather than as
    /// stale bytes from a previous exit.
    #[must_use]
    pub fn fetched_instruction(&self) -> &[u8] {
        let fetched = self.fetched_bytes as usize;
        let len = if fetched > self.instruction_bytes.len() {
            self.instruction_bytes.len()
        } else {
            fetched
        };
        &self.instruction_bytes[..len]
    }
}

layout! {
    ControlArea, size = CONTROL_AREA_BYTES,
    0x000 => intercept_control_registers,
    0x004 => intercept_debug_registers,
    0x008 => intercept_exceptions,
    0x00C => intercept_1,
    0x010 => intercept_2,
    0x014 => intercept_3,
    0x018 => reserved_0x018,
    0x03C => pause_filter_threshold,
    0x03E => pause_filter_count,
    0x040 => io_permissions,
    0x048 => msr_permissions,
    0x050 => tsc_offset,
    0x058 => asid,
    0x05C => tlb_control,
    0x05D => erap_control,
    0x05E => reserved_0x05e,
    0x060 => interrupt_control,
    0x068 => interrupt_state,
    0x070 => exit_code,
    0x078 => exit_info_1,
    0x080 => exit_info_2,
    0x088 => exit_interrupt_info,
    0x090 => nested_paging,
    0x098 => avic_apic_bar,
    0x0A0 => ghcb_gpa,
    0x0A8 => event_injection,
    0x0B0 => nested_cr3,
    0x0B8 => virtualization_control,
    0x0C0 => clean,
    0x0C4 => reserved_0x0c4,
    0x0C8 => next_rip,
    0x0D0 => fetched_bytes,
    0x0D1 => instruction_bytes,
    0x0E0 => avic_backing_page,
    0x0E8 => reserved_0x0e8,
    0x0F0 => avic_logical_table,
    0x0F8 => avic_physical_table,
    0x100 => reserved_0x100,
    0x108 => vmsa_pointer,
    0x110 => vmgexit_rax,
    0x118 => vmgexit_cpl,
    0x119 => reserved_0x119,
    0x120 => bus_lock_threshold,
    0x122 => reserved_0x122,
    0x134 => update_irr,
    0x138 => allowed_sev_features,
    0x140 => guest_sev_features,
    0x148 => reserved_0x148,
    0x150 => requested_irr,
    0x170 => reserved_0x170,
    0x3E0 => host,
}

/// The guest's virtual interrupt state.
///
/// Two quite different things share this register. The lower half is a
/// *pending interrupt*: one the hypervisor wants the guest to take, described
/// by priority and vector, which the guest will accept when its own flags and
/// task priority allow — the same rules a real interrupt obeys. The upper half
/// is *policy*: which parts of interrupt handling the processor should
/// virtualize rather than let the guest control directly.
///
/// The virtualized-masking bit is the one with the widest consequences. With it
/// clear, the guest's interrupt flag controls both its own interrupts and the
/// physical ones, so a guest that disables interrupts disables them for the
/// machine. With it set, the host's flag at the moment of entry governs
/// physical interrupts and the guest's flag governs only its own — and reads
/// and writes of the task priority through the control register go to the
/// virtual copy here instead of to the real interrupt controller.
///
/// When the hardware drives the guest's interrupt controller directly, the
/// pending-interrupt fields here are ignored on entry: the controller's own
/// registers are where a pending interrupt lives instead.
///
/// The two bits that turn that driving on live here as well, and they are the
/// one thing about this register worth knowing before editing it: the whole
/// quadword is a single clean group — the interrupt one — so a hypervisor that
/// arms or disarms the acceleration and clears the acceleration's own clean bit
/// has cleared the wrong one. [`crate::CleanBits::INTERRUPT`] is where both
/// halves of that are stated.
#[bitfield_struct::bitfield(u64)]
#[derive(PartialEq, Eq)]
pub struct InterruptControl {
    /// The guest's task priority: the interrupt priority it will not accept
    /// below. Only the low four bits are used, and the processor writes this
    /// back on exit because the guest may have changed it.
    #[bits(4)]
    pub virtual_tpr: u8,
    #[bits(4)]
    __: u8,
    /// An interrupt is waiting for the guest. Written back on exit, and
    /// ignored on entry when the hardware drives the guest's controller.
    pub virtual_irq_pending: bool,
    /// The guest's own global interrupt flag: clear masks its virtual
    /// interrupts, set unmasks them.
    pub virtual_gif: bool,
    __: bool,
    /// A non-maskable interrupt is waiting for the guest.
    pub virtual_nmi_pending: bool,
    /// The guest is inside a non-maskable interrupt handler and will not take
    /// another until it returns.
    pub virtual_nmi_masked: bool,
    #[bits(3)]
    __: u8,
    /// Priority of the waiting interrupt, compared against the guest's task
    /// priority unless that comparison is being ignored.
    #[bits(4)]
    pub virtual_priority: u8,
    /// Deliver the waiting interrupt whatever the guest's task priority says.
    pub ignore_virtual_tpr: bool,
    #[bits(3)]
    __: u8,
    /// Virtualize interrupt masking: the guest's interrupt flag stops
    /// controlling physical interrupts, and its task priority accesses are
    /// redirected to the virtual copy above.
    pub intercept_interrupt_masking: bool,
    /// Give the guest its own global interrupt flag, so that the instructions
    /// which clear and set it need not be intercepted.
    pub virtual_gif_enable: bool,
    /// Virtualize non-maskable interrupt masking. Requires that non-maskable
    /// interrupts also be intercepted; entering a guest without that fails.
    pub virtual_nmi_enable: bool,
    #[bits(3)]
    __: u8,
    /// Drive the guest's interrupt controller in hardware with 32-bit
    /// identifiers rather than eight-bit ones.
    pub x2avic_enable: bool,
    /// Drive the guest's interrupt controller in hardware.
    pub avic_enable: bool,
    /// Vector of the waiting interrupt.
    #[bits(8)]
    pub virtual_vector: u8,
    #[bits(24)]
    __: u32,
}

/// Whether the guest's own page tables are themselves translated, and what
/// else travels with that decision.
///
/// Without nested paging a hypervisor must maintain shadow page tables and
/// intercept every change the guest makes to its own. With it, the guest walks
/// its tables freely and a second set translates the result — which is both far
/// faster and the reason a guest can be given memory that simply is not there.
///
/// The remaining bits either belong to the encrypted-virtualization extension
/// or refine how the second set of tables is walked.
#[bitfield_struct::bitfield(u64)]
#[derive(PartialEq, Eq)]
pub struct NestedPagingControl {
    /// Translate the guest's physical addresses through a second set of page
    /// tables rooted at this block's nested table root.
    pub enabled: bool,
    /// Encrypt the guest's memory, part of an extension pulzar does not
    /// implement.
    pub encrypted_memory: bool,
    /// Encrypt the guest's register state too, part of an extension pulzar does
    /// not implement.
    pub encrypted_state: bool,
    /// Trap a guest that executes from a page its own tables call user memory,
    /// which lets a hypervisor tell the two kinds of execution apart.
    pub guest_mode_execute_trap: bool,
    /// Restrict which pages a guest may use for a supervisor shadow stack.
    /// Requires nested paging, and requires the host to have no-execute
    /// translation enabled.
    pub supervisor_shadow_stack_check: bool,
    /// Encrypt every guest access whatever the guest's own tables say, part of
    /// an extension pulzar does not implement.
    pub transparent_encryption: bool,
    /// Treat the guest's own page tables as read-only, so that the processor
    /// does not write access and dirty bits into them.
    pub read_only_guest_page_tables: bool,
    /// Let the guest execute the broadcast invalidation instructions rather
    /// than raising an invalid-opcode exception.
    pub invlpgb_enable: bool,
    #[bits(56)]
    __: u64,
}

/// Which pieces of processor state the hardware maintains for the guest
/// instead of trapping.
///
/// Each of these turns a set of model-specific registers from something a
/// hypervisor must intercept and emulate into something the guest may touch
/// directly, with the processor swapping the values around the guest's
/// execution. Every one depends on the processor supporting it.
#[bitfield_struct::bitfield(u64)]
#[derive(PartialEq, Eq)]
pub struct VirtualizationControl {
    /// Maintain the guest's own branch-record registers, so a guest may profile
    /// its own branches without every access coming back to us.
    pub last_branch_record: bool,
    /// Let the guest save and restore the processor state those two
    /// instructions cover without being intercepted, which is what makes a
    /// hypervisor running inside this one affordable.
    pub vmsave_vmload: bool,
    /// Maintain the guest's instruction-sampling state.
    pub instruction_based_sampling: bool,
    /// Maintain the guest's performance counters.
    pub performance_counters: bool,
    #[bits(60)]
    __: u64,
}

#[cfg(test)]
mod tests {
    //! The word at 060h is the most-written word in the block, and the two bits
    //! of it that choose how a guest's interrupts are delivered are the ones
    //! whose positions nothing else can catch: the layout macro fixes the *sum*
    //! of the field widths to the size of the word, so resizing or reordering
    //! anything below them moves them and still compiles.

    use super::InterruptControl;

    /// The two enable bits, at bits 31 and 30 of the interrupt-control word
    /// (APM Table B-1).
    ///
    /// Which of the two is which decides whether the hardware drives a guest's
    /// controller at all and whether it addresses the guest's processors with
    /// eight-bit or 32-bit identifiers — and a control block that names the
    /// wider mode without the narrower one is refused outright, with no guest
    /// instruction executed.
    #[test]
    fn the_avic_enables_are_the_two_bits_above_the_vector() {
        assert_eq!(
            InterruptControl::new().with_avic_enable(true).into_bits(),
            1 << 31
        );
        assert_eq!(
            InterruptControl::new().with_x2avic_enable(true).into_bits(),
            1 << 30
        );
        assert_eq!(
            InterruptControl::new()
                .with_avic_enable(true)
                .with_x2avic_enable(true)
                .into_bits(),
            (1 << 31) | (1 << 30)
        );
    }

    /// The fields either side of them, so that a bit moved into or out of the
    /// enables fails here rather than at an entry the processor refuses.
    #[test]
    fn the_neighbouring_fields_are_where_the_architecture_puts_them() {
        // The task priority is the low four bits with the four above it
        // reserved, and the waiting vector is the byte above the enables.
        assert_eq!(
            InterruptControl::new().with_virtual_tpr(0xF).into_bits(),
            0xF
        );
        assert_eq!(
            InterruptControl::new()
                .with_virtual_vector(0xFF)
                .into_bits(),
            0xFF << 32
        );
        // The three bits below the enables are reserved, so the field before
        // them ends where it does.
        assert_eq!(
            InterruptControl::new()
                .with_virtual_nmi_enable(true)
                .into_bits(),
            1 << 26
        );
    }
}
