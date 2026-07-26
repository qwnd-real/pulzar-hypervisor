//! The state firmware was running with, read before pulzar overwrites any of
//! it.
//!
//! Pulzar hands the machine on rather than keeping it. After bring-up it means
//! to enter a guest that *is* the firmware environment as it was found, and go
//! on booting whatever firmware would have booted. Describing that guest to the
//! processor takes the registers firmware was using — and every one of them is
//! a register pulzar overwrites on its way up: the extended feature register to
//! enable no-execute, the page-attribute table to fix what cache selection
//! means, the descriptor tables to own its own interrupts, the local
//! controller's timer to own its own clock. Not one can be read back
//! afterwards.
//!
//! So they are read at the one instant they are still firmware's: the first
//! statement of the loader, above the line that reprograms a serial port and
//! before a single boot service has been called.
//!
//! # Why it cannot report anything
//!
//! Because there is no serial port yet — bringing one up is the next thing that
//! happens. [`capture`] therefore has nowhere to log and no way to fail
//! usefully, and is written to need neither: every read that could fault is
//! gated on the bit that says the register exists, and a register that is not
//! there leaves its field zero. What was captured is logged by
//! [`FirmwareContext::describe`] once there is somewhere to log it to.
//!
//! # Why the processor state is a save area
//!
//! Because that is the shape it will be used in. [`SaveArea`] is the layout the
//! processor loads a guest from, every offset in it asserted against the
//! architecture by the crate that defines it. Filling one in here means the
//! eventual entry assigns it into a control block whole, instead of copying
//! forty fields out of a structure that says the same thing in a different
//! order.
//!
//! Three of its fields are deliberately left as [`SaveArea::zeroed`] left them,
//! because firmware's values are not the ones a guest wants:
//!
//! - `rip` and `rax`, which would point inside this capture. The guest starts
//!   on a stub of pulzar's own, so whoever enters it fills these in.
//! - the virtualization-enable bit of `efer`, which the processor requires set
//!   to enter a guest at all and which firmware naturally does not have set.
//!   The field holds firmware's register exactly; adding that bit belongs to
//!   the entry.
//!
//! `rsp` is *not* among them. The guest goes on using the UEFI stack rather
//! than one of pulzar's, so firmware's stack pointer is captured like
//! everything else.
//!
//! Nor are the fields the processor only swaps when a feature is switched on
//! elsewhere — the branch records, the speculation control, the performance
//! counters, the sampling block, the shadow-stack registers. Pulzar writes none
//! of the underlying registers, so firmware's values are still in the hardware,
//! and a control area that leaves those features off never reads these fields.
//! The page-attribute table is the exception and the reason it is captured: the
//! address-space subsystem does overwrite it, and the processor does load it
//! under nested paging.

#![no_std]

mod segments;

use apic::{Controller, FirmwareState};
use log::info;
use paging::DirectMap;
use svm::SaveArea;
use x86_64::registers::{
    control::{Cr0, Cr2, Cr3, Cr4, Cr4Flags},
    debug::{Dr6, Dr7},
    model_specific::{Efer, KernelGsBase, Msr},
    rflags,
    xcontrol::XCr0,
};

/// Everything firmware was running with.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct FirmwareContext {
    /// Firmware's processor state, in the layout a guest is loaded from.
    pub cpu: SaveArea,
    /// Firmware's interrupt controllers, local and legacy.
    pub interrupts: FirmwareState,
    /// The extended control register, or zero on a processor with none — which
    /// is the same answer, since a processor without it has no extended state
    /// enabled either.
    pub xcr0: u64,
    /// The timestamp counter, read immediately after the local timer's counts
    /// so that how stale they are can be worked out from something that
    /// also counts.
    pub tsc: u64,
}

/// Reads the machine as firmware left it.
///
/// # Safety
///
/// `window` must describe a live mapping of physical memory in the active
/// address space, and that address space must be firmware's: the descriptor
/// tables are read through the bases the processor's own registers name, and
/// the local controller's register page through the window.
///
/// Nothing checks that this is called before pulzar has modified anything —
/// that is the caller's to arrange, and it is the whole value of the result.
#[must_use]
pub unsafe fn capture(window: DirectMap) -> FirmwareContext {
    let mut cpu = SaveArea::zeroed();
    control_registers(&mut cpu);
    model_specific(&mut cpu);
    // SAFETY: the caller guarantees the active address space is the one
    // firmware's descriptor-table registers were loaded under.
    unsafe { segments::capture(&mut cpu) };

    // SAFETY: the caller guarantees the window reaches physical memory, which is
    // how the register page is read where the controller presents one.
    let interrupts = unsafe { apic::capture(window) };
    // Immediately afterwards, so that the timer's counts and the counter they go
    // stale against are as close together as this code can put them.
    let tsc = timestamp();

    FirmwareContext {
        cpu,
        interrupts,
        xcr0: extended_control(),
        tsc,
    }
}

impl FirmwareContext {
    /// Logs what firmware was running with.
    ///
    /// Every value that anything later restores or virtualizes, so that a
    /// serial log records the machine as it was found and the two images
    /// can be checked against each other: the loader captures this, and the
    /// hypervisor prints the same numbers back out of the chunk once
    /// firmware is gone.
    pub fn describe(&self, who: &str) {
        let cpu = &self.cpu;
        info!(
            "{who}: firmware cr0 {:#x}, cr2 {:#x}, cr3 {:#x}, cr4 {:#x}",
            cpu.cr0, cpu.cr2, cpu.cr3, cpu.cr4
        );
        info!(
            "{who}: firmware efer {:#x}, pat {:#x}, xcr0 {:#x}, rflags {:#x}",
            cpu.efer, cpu.g_pat, self.xcr0, cpu.rflags
        );
        info!(
            "{who}: firmware rsp {:#x}, cpl {}, dr6 {:#x}, dr7 {:#x}",
            cpu.rsp, cpu.cpl, cpu.dr6, cpu.dr7
        );
        info!(
            "{who}: firmware gdtr {:#x}+{:#x}, idtr {:#x}+{:#x}",
            cpu.gdtr.base, cpu.gdtr.limit, cpu.idtr.base, cpu.idtr.limit
        );
        for (name, segment) in [
            ("cs", &cpu.cs),
            ("ss", &cpu.ss),
            ("ds", &cpu.ds),
            ("es", &cpu.es),
            ("fs", &cpu.fs),
            ("gs", &cpu.gs),
            ("ldtr", &cpu.ldtr),
            ("tr", &cpu.tr),
        ] {
            info!(
                "{who}: firmware {name} {:#06x}, base {:#x}, limit {:#x}, attributes {:#05x}",
                segment.selector,
                segment.base,
                segment.limit,
                segment.attributes.into_bits(),
            );
        }
        info!(
            "{who}: firmware kernel_gs_base {:#x}, star {:#x}, lstar {:#x}, cstar {:#x}, sfmask {:#x}",
            cpu.kernel_gs_base, cpu.star, cpu.lstar, cpu.cstar, cpu.sfmask
        );
        info!(
            "{who}: firmware sysenter cs {:#x}, esp {:#x}, eip {:#x}",
            cpu.sysenter_cs, cpu.sysenter_esp, cpu.sysenter_eip
        );
        self.describe_interrupts(who);
    }

    /// Logs the interrupt controllers, which is the half of the capture a
    /// virtual local controller is later built from.
    fn describe_interrupts(&self, who: &str) {
        let apic = &self.interrupts;
        let local = &apic.local;
        // Said first, because everything below it is zero when the controller
        // was not read and zeros that mean "not read" look exactly like zeros
        // that were read.
        if apic.controller != Controller::Read {
            info!("{who}: firmware apic not read: {:?}", apic.controller);
        }
        info!(
            "{who}: firmware apic base {:#x}, id {:#x}, version {:#x}",
            apic.base, local.id, local.version
        );
        info!(
            "{who}: firmware apic spurious {:#x}, task priority {:#x}, processor priority {:#x}, errors {:#x}",
            local.spurious, local.task_priority, local.processor_priority, local.error_status
        );
        info!(
            "{who}: firmware apic destination {:#x}, format {:#x}, command {:#x}",
            local.logical_destination, local.destination_format, local.command
        );
        info!(
            "{who}: firmware apic timer entry {:#x}, divide {:#x}, initial {:#x}, current {:#x}, deadline {:#x}, tsc {:#x}",
            local.lvt_timer,
            local.timer_divide,
            local.timer_initial_count,
            local.timer_current_count,
            local.tsc_deadline,
            self.tsc,
        );
        info!(
            "{who}: firmware apic lvt lint0 {:#x}, lint1 {:#x}, error {:#x}, thermal {:#x}, performance {:#x}, machine check {:#x}",
            local.lvt_lint0,
            local.lvt_lint1,
            local.lvt_error,
            local.lvt_thermal,
            local.lvt_performance,
            local.lvt_corrected_machine_check,
        );
        for (name, words) in [
            ("in service", &local.in_service),
            ("pending", &local.interrupt_request),
            ("level triggered", &local.trigger_mode),
        ] {
            if words.iter().any(|word| *word != 0) {
                info!("{who}: firmware apic {name} {words:#010x?}");
            }
        }
        info!(
            "{who}: firmware 8259 masks {:#04x} and {:#04x}",
            apic.legacy_masks[0], apic.legacy_masks[1]
        );
    }
}

/// Firmware's control, flag, debug and stack registers.
fn control_registers(cpu: &mut SaveArea) {
    let (root, flags) = Cr3::read_raw();
    cpu.cr0 = Cr0::read_raw();
    cpu.cr2 = Cr2::read_raw();
    cpu.cr3 = root.start_address().as_u64() | u64::from(flags);
    cpu.cr4 = Cr4::read_raw();
    cpu.efer = Efer::read_raw();
    cpu.rflags = rflags::read_raw();
    cpu.dr6 = Dr6::read_raw();
    cpu.dr7 = Dr7::read_raw();
    cpu.rsp = stack_pointer();
}

/// Firmware's model-specific registers.
///
/// Read raw, through [`Msr`], rather than through the typed wrappers for them:
/// the save area holds what the register holds, and a wrapper that decodes
/// `IA32_STAR` into the two selectors long mode uses would drop the half of it
/// a guest is still entered with.
fn model_specific(cpu: &mut SaveArea) {
    cpu.star = read(IA32_STAR);
    cpu.lstar = read(IA32_LSTAR);
    cpu.cstar = read(IA32_CSTAR);
    cpu.sfmask = read(IA32_SFMASK);
    cpu.kernel_gs_base = KernelGsBase::read().as_u64();
    cpu.sysenter_cs = read(IA32_SYSENTER_CS);
    cpu.sysenter_esp = read(IA32_SYSENTER_ESP);
    cpu.sysenter_eip = read(IA32_SYSENTER_EIP);
    // The guest's page-attribute table, which the processor loads under nested
    // paging — and the one register here the loader is about to overwrite, since
    // the address-space subsystem programs the architectural default so that
    // cache selection needs no `PAT` bit of its own.
    cpu.g_pat = read(IA32_PAT);
}

/// Reads a model-specific register.
fn read(index: u32) -> u64 {
    // SAFETY: every index this is called with is architectural on any processor
    // running in long mode, which firmware is, and reading a model-specific
    // register has no side effect.
    unsafe { Msr::new(index).read() }
}

/// Firmware's stack pointer.
///
/// What it names is a point in firmware's own stack below every frame that was
/// live when firmware handed control over, which is all a stack pointer has to
/// be: everything above it stays firmware's and everything below it is free.
/// The guest goes on using this stack, so how deep it is matters and where
/// exactly does not.
fn stack_pointer() -> u64 {
    let rsp: u64;
    // SAFETY: reading the stack pointer into a register touches no memory,
    // disturbs no flags, and has no effect on anything at all.
    unsafe {
        core::arch::asm!("mov {}, rsp", out(reg) rsp, options(nomem, nostack, preserves_flags));
    }
    rsp
}

/// The timestamp counter.
fn timestamp() -> u64 {
    // SAFETY: `RDTSC` is always permitted at privilege level zero, whatever
    // `CR4.TSD` says, and reading the counter does not disturb it.
    unsafe { core::arch::x86_64::_rdtsc() }
}

/// The extended control register, where the processor has one.
///
/// Reading it faults where the operating-system support bit is clear, so that
/// bit is what decides whether it is read at all rather than something to find
/// out by trying.
fn extended_control() -> u64 {
    if Cr4::read().contains(Cr4Flags::OSXSAVE) {
        XCr0::read_raw()
    } else {
        0
    }
}

/// The selectors and entry point the fast system-call instructions use.
const IA32_STAR: u32 = 0xC000_0081;

/// Where a fast system call from 64-bit code lands.
const IA32_LSTAR: u32 = 0xC000_0082;

/// Where a fast system call from compatibility mode lands.
const IA32_CSTAR: u32 = 0xC000_0083;

/// Which flags a fast system call clears on entry.
const IA32_SFMASK: u32 = 0xC000_0084;

/// The code selector the older fast system-call mechanism enters through.
const IA32_SYSENTER_CS: u32 = 0x174;

/// The stack pointer that mechanism enters with.
const IA32_SYSENTER_ESP: u32 = 0x175;

/// The instruction pointer that mechanism enters at.
const IA32_SYSENTER_EIP: u32 = 0x176;

/// What each `PAT:PCD:PWT` combination in a page-table entry means.
const IA32_PAT: u32 = 0x277;
