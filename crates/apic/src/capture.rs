//! The interrupt controllers as firmware left them.
//!
//! Everything [`crate::Apic::install`] is about to overwrite, read before it
//! does. The boot processor's controller has been firmware's for the whole of
//! the boot: its timer armed for firmware's own purposes, its pins wired the
//! way firmware's tables describe, whatever it was delivering still latched in
//! its in-service and request registers. A hypervisor that means to hand the
//! machine back has to know all of that, and none of it can be read once the
//! controller has been claimed.
//!
//! # Reading changes nothing
//!
//! Not one register here is written, which is what separates a snapshot from a
//! bring-up. The error status register is the case worth naming: the
//! architecture only refreshes it on a write, so reading it alone reports
//! whatever was latched rather than what has happened since. That is the honest
//! answer to ask for — the alternative reports more by destroying it.
//!
//! # Which registers exist depends on the controller
//!
//! Three gates, each the difference between a value and a fault. The controller
//! may be switched off, in which case it has no registers to read. It may
//! already be in x2APIC, where the destination format register and the upper
//! half of the interrupt command are not merely unused but absent, and reading
//! the model-specific registers they would have had is a general protection
//! fault. And the corrected machine-check entry exists only where the version
//! register counts far enough to reach it.

use paging::DirectMap;
use x86_64::{PhysAddr, registers::model_specific::Msr};

use crate::{
    LVT_COUNT_SHIFT, VERSION_MASK, base, pic,
    register::{Access, Register, X2APIC_COMMAND_MSR},
    timer::{self, Mode},
};

/// How many registers it takes to describe two hundred and fifty-six vectors
/// one bit at a time.
pub const VECTOR_WORDS: usize = 8;

/// The interrupt controllers as firmware left them.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct FirmwareState {
    /// `IA32_APIC_BASE`: where the register page is, whether the controller is
    /// switched on, which of the two interfaces it presents, and whether this
    /// is the processor the machine started on.
    pub base: u64,
    /// This processor's local controller, or every field zero where firmware
    /// had it switched off.
    pub local: LocalState,
    /// The interrupt masks of the two legacy controllers, the primary's first.
    pub legacy_masks: [u8; 2],
}

/// One local controller's registers.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct LocalState {
    /// The identifier interrupts to this processor are addressed to, in the
    /// register's own form: the top eight bits under the older interface, the
    /// whole of it under x2APIC.
    pub id: u32,
    /// The controller's version, and one less than its number of local vector
    /// table entries in the third byte.
    pub version: u32,
    /// Which interrupt priorities firmware was willing to accept.
    pub task_priority: u32,
    /// The priority it was actually servicing at.
    pub processor_priority: u32,
    /// Which logical destinations this processor answered to.
    pub logical_destination: u32,
    /// How that logical destination was matched. Zero under x2APIC, which has
    /// one model and no register to choose it with.
    pub destination_format: u32,
    /// The spurious vector, and the bit that software-enables the controller.
    pub spurious: u32,
    /// Which vectors firmware had accepted and not yet acknowledged.
    pub in_service: [u32; VECTOR_WORDS],
    /// Which of those arrived level triggered.
    pub trigger_mode: [u32; VECTOR_WORDS],
    /// Which vectors had been delivered and not yet accepted.
    pub interrupt_request: [u32; VECTOR_WORDS],
    /// What the controller had latched about itself, read without clearing it.
    pub error_status: u32,
    /// The interrupt command, as one value whichever interface holds it in two.
    pub command: u64,
    /// The corrected machine-check entry, or zero on a controller too old to
    /// have one.
    pub lvt_corrected_machine_check: u32,
    /// The timer's entry: its vector, its mask, and which of the three modes it
    /// was counting in.
    pub lvt_timer: u32,
    /// The thermal sensor's entry.
    pub lvt_thermal: u32,
    /// The performance counters' entry.
    pub lvt_performance: u32,
    /// The first interrupt pin's entry, including the polarity and trigger mode
    /// only a wired source has.
    pub lvt_lint0: u32,
    /// The second interrupt pin's entry.
    pub lvt_lint1: u32,
    /// The entry the controller reports its own errors through.
    pub lvt_error: u32,
    /// How far the input clock was divided before the timer counted it.
    pub timer_divide: u32,
    /// What the timer was counting down from.
    pub timer_initial_count: u32,
    /// What it had left at the instant this was read, and stale from that
    /// instant onwards: a running timer goes on counting. Whoever needs to know
    /// by how much has to pair it with a reading of something that also counts.
    pub timer_current_count: u32,
    /// The deadline the timer was waiting for, where it was in the mode that
    /// has one. Zero otherwise, and the register is not read at all: it exists
    /// only on a processor that implements the mode.
    pub tsc_deadline: u64,
}

/// Reads the interrupt controllers as firmware left them, writing to none of
/// them.
///
/// # Safety
///
/// `window` must reach physical memory in the active address space. Where the
/// controller presents the older interface its register page is read through
/// it, and a window that does not cover the page would read whatever else lies
/// at that address.
#[must_use]
pub unsafe fn capture(window: DirectMap) -> FirmwareState {
    let base = base::read();
    let legacy_masks = pic::masks();
    // SAFETY: the caller guarantees the window describes the active address
    // space, which is what makes the register page reachable through it.
    let local = unsafe { reached(base, window) }.map_or_else(LocalState::default, local);
    FirmwareState {
        base,
        local,
        legacy_masks,
    }
}

/// How the controller firmware left is reached, or `None` if firmware had it
/// switched off or its register page lies outside the window.
///
/// # Safety
///
/// As [`capture`].
unsafe fn reached(base: u64, window: DirectMap) -> Option<Access> {
    if base & base::GLOBAL_ENABLE == 0 {
        return None;
    }
    if base & base::X2APIC_ENABLE != 0 {
        return Some(Access::Msr);
    }
    window
        .virt(PhysAddr::new(base & base::ADDRESS_MASK))
        .map(Access::Mapped)
}

/// Every register the controller reached through `access` has.
fn local(access: Access) -> LocalState {
    let version = access.read(Register::VERSION);
    let lvt_timer = access.read(Register::LVT_TIMER);
    LocalState {
        id: access.read(Register::ID),
        version,
        task_priority: access.read(Register::TASK_PRIORITY),
        processor_priority: access.read(Register::PROCESSOR_PRIORITY),
        logical_destination: access.read(Register::LOGICAL_DESTINATION),
        // The register the newer interface does not have. Reading its
        // model-specific register there is a fault, not a zero.
        destination_format: match access {
            Access::Mapped(_) => access.read(Register::DESTINATION_FORMAT),
            Access::Msr => 0,
        },
        spurious: access.read(Register::SPURIOUS),
        in_service: bank(access, Register::IN_SERVICE),
        trigger_mode: bank(access, Register::TRIGGER_MODE),
        interrupt_request: bank(access, Register::INTERRUPT_REQUEST),
        error_status: access.read(Register::ERROR_STATUS),
        command: command(access),
        lvt_corrected_machine_check: if entries(version) > CORRECTED_MACHINE_CHECK_ENTRY {
            access.read(Register::LVT_CORRECTED_MACHINE_CHECK)
        } else {
            0
        },
        lvt_timer,
        lvt_thermal: access.read(Register::LVT_THERMAL),
        lvt_performance: access.read(Register::LVT_PERFORMANCE),
        lvt_lint0: access.read(Register::LVT_LINT0),
        lvt_lint1: access.read(Register::LVT_LINT1),
        lvt_error: access.read(Register::LVT_ERROR),
        timer_divide: access.read(Register::TIMER_DIVIDE),
        timer_initial_count: access.read(Register::TIMER_INITIAL_COUNT),
        timer_current_count: access.read(Register::TIMER_CURRENT_COUNT),
        // A timer already in the mode is proof the processor implements it,
        // which is the only thing that makes the register safe to read.
        tsc_deadline: match Mode::of(lvt_timer) {
            Some(Mode::Deadline) => {
                // SAFETY: the entry says the timer is counting against a
                // deadline, so the register holding that deadline exists, and
                // reading a model-specific register has no side effect.
                unsafe { Msr::new(timer::IA32_TSC_DEADLINE).read() }
            }
            _ => 0,
        },
    }
}

/// One of the three banks of eight registers that describe every vector.
fn bank(access: Access, first: Register) -> [u32; VECTOR_WORDS] {
    let mut words = [0; VECTOR_WORDS];
    // Counted alongside rather than by index, so the slot number is already the
    // width the register offset is computed in.
    for (word, slot) in words.iter_mut().zip(0..) {
        *word = access.read(first.offset_by(slot));
    }
    words
}

/// The interrupt command, which the older interface splits across two registers
/// and the newer one holds in a single wide model-specific register.
fn command(access: Access) -> u64 {
    match access {
        Access::Mapped(_) => {
            u64::from(access.read(Register::COMMAND_HIGH)) << u32::BITS
                | u64::from(access.read(Register::COMMAND_LOW))
        }
        // SAFETY: the register is architectural on a controller in x2APIC mode,
        // which is the only way this variant is reached, and reading it neither
        // sends a command nor disturbs one.
        Access::Msr => unsafe { Msr::new(X2APIC_COMMAND_MSR).read() },
    }
}

/// How many local vector table entries the controller reports having.
const fn entries(version: u32) -> u32 {
    ((version >> LVT_COUNT_SHIFT) & VERSION_MASK) + 1
}

/// Which entry the corrected machine-check register is, counting from one.
///
/// The last one the architecture added, so a controller has it exactly when it
/// counts more entries than the six that came before.
const CORRECTED_MACHINE_CHECK_ENTRY: u32 = 6;
