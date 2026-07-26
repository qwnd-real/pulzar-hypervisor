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
//! Four gates, each the difference between a value and a fault. The processor
//! may have no local controller at all, in which case even the register saying
//! where the others are does not exist. The controller may be switched off, in
//! which case it has no registers to answer with. It may already be in x2APIC,
//! where the destination format register and the upper half of the interrupt
//! command are not merely unused but absent, and reading the model-specific
//! registers they would have had is a general protection fault. And the entries
//! of the local vector table exist only as far as the version register counts,
//! so the performance, thermal and machine-check entries are all questions a
//! controller may have no answer to.
//!
//! Every one of those is reported rather than papered over: a zeroed
//! [`LocalState`] always comes with the [`Controller`] saying why, because
//! zeros that mean "not read" and zeros that mean "read as zero" are not the
//! same answer and nothing downstream could tell them apart.

use paging::DirectMap;
use x86_64::registers::model_specific::Msr;

use crate::{
    base, pic, register,
    register::{Access, MappedRegister, Page, Register},
    timer::{self, Mode},
};

/// How many registers it takes to describe two hundred and fifty-six vectors
/// one bit at a time.
pub const VECTOR_WORDS: usize = register::VECTOR_SLOTS;

/// The interrupt controllers as firmware left them.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct FirmwareState {
    /// `IA32_APIC_BASE`: where the register page is, whether the controller is
    /// switched on, which of the two interfaces it presents, and whether this
    /// is the processor the machine started on. Zero on a processor with no
    /// local controller, which has no such register.
    pub base: u64,
    /// Whether the local controller below was read, and why not where it was
    /// not.
    pub controller: Controller,
    /// This processor's local controller, or every field zero where
    /// [`FirmwareState::controller`] says it could not be read.
    pub local: LocalState,
    /// The interrupt masks of the two legacy controllers, the primary's first.
    pub legacy_masks: [u8; 2],
}

/// Whether the local controller could be read, and why not where it could not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Controller {
    /// Read. Every field of [`LocalState`] is what the controller held.
    Read = 0,
    /// The processor has no local controller, so there was nothing to read and
    /// no register saying where to read it from.
    Absent = 1,
    /// Firmware had switched the controller off. A controller that is off does
    /// not answer, and what its registers held is already gone.
    Disabled = 2,
    /// The controller presents the memory-mapped interface and its register
    /// page is not inside the window this was given, so reading it would have
    /// meant reading something else instead.
    Unreachable = 3,
}

/// One local controller's registers.
///
/// Every field is zero where the controller does not have that register: the
/// local vector table stops where the version register says it stops, and the
/// destination format register and the deadline exist only in one interface and
/// one timer mode respectively.
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
    /// The corrected machine-check entry, or zero on a controller that does not
    /// count far enough to have one.
    pub lvt_corrected_machine_check: u32,
    /// The timer's entry: its vector, its mask, and which of the three modes it
    /// was counting in.
    pub lvt_timer: u32,
    /// The thermal sensor's entry, or zero on a controller without one.
    pub lvt_thermal: u32,
    /// The performance counters' entry, or zero on a controller without one.
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
/// `window` must reach physical memory in the active address space, and its
/// mapping of the controller's register page — where the controller presents
/// one — must be *uncached*. The architecture requires those registers to be
/// reached strongly uncacheable, and through a write-back mapping a read may be
/// answered out of a cache line that says nothing about what the controller
/// holds. Firmware's own identity map satisfies both; a window built to reach
/// ordinary memory quickly does not, and passing one would read plausible
/// rubbish rather than fail.
#[must_use]
pub unsafe fn capture(window: DirectMap) -> FirmwareState {
    let legacy_masks = pic::masks();
    let base = base::read();
    // SAFETY: the caller guarantees the window describes the active address
    // space and maps the register page uncached, which is what makes the page
    // both reachable and worth reading through it.
    let reached = unsafe { reached(base, window) };
    let (controller, local) = match reached {
        Ok(access) => (Controller::Read, read(access)),
        Err(controller) => (controller, LocalState::default()),
    };
    FirmwareState {
        base: base.unwrap_or_default(),
        controller,
        local,
        legacy_masks,
    }
}

/// How the controller firmware left is reached, or why it cannot be.
///
/// # Safety
///
/// As [`capture`].
unsafe fn reached(base: Option<u64>, window: DirectMap) -> Result<Access, Controller> {
    let base = base.ok_or(Controller::Absent)?;
    if base & base::GLOBAL_ENABLE == 0 {
        return Err(Controller::Disabled);
    }
    if base & base::X2APIC_ENABLE != 0 {
        return Ok(Access::Msr);
    }
    let virt = window
        .virt(base::page_of(base))
        .ok_or(Controller::Unreachable)?;
    // SAFETY: the window maps whole frames and the caller guarantees this one is
    // mapped, and mapped uncached — which is the whole of what a page needs.
    Ok(Access::Mapped(unsafe { Page::new(virt) }))
}

/// Every register the controller reached through `access` has.
fn read(access: Access) -> LocalState {
    let version = access.read(Register::VERSION);
    let entries = register::lvt_entries(version);
    let lvt_timer = entry(access, Register::LVT_TIMER, entries);
    LocalState {
        id: access.read(Register::ID),
        version,
        task_priority: access.read(Register::TASK_PRIORITY),
        processor_priority: access.read(Register::PROCESSOR_PRIORITY),
        logical_destination: access.read(Register::LOGICAL_DESTINATION),
        // The register the newer interface does not have, which is why naming
        // it needs the page the older interface is.
        destination_format: match access {
            Access::Mapped(page) => page.read(MappedRegister::DESTINATION_FORMAT),
            Access::Msr => 0,
        },
        spurious: access.read(Register::SPURIOUS),
        in_service: bank(access, Register::IN_SERVICE),
        trigger_mode: bank(access, Register::TRIGGER_MODE),
        interrupt_request: bank(access, Register::INTERRUPT_REQUEST),
        error_status: access.read(Register::ERROR_STATUS),
        command: access.command(),
        lvt_corrected_machine_check: entry(access, Register::LVT_CORRECTED_MACHINE_CHECK, entries),
        lvt_timer,
        lvt_thermal: entry(access, Register::LVT_THERMAL, entries),
        lvt_performance: entry(access, Register::LVT_PERFORMANCE, entries),
        lvt_lint0: entry(access, Register::LVT_LINT0, entries),
        lvt_lint1: entry(access, Register::LVT_LINT1, entries),
        lvt_error: entry(access, Register::LVT_ERROR, entries),
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

/// One local vector table entry, or zero on a controller that does not count
/// far enough to have it.
fn entry(access: Access, register: Register, entries: u32) -> u32 {
    if register::has_lvt(register, entries) {
        access.read(register)
    } else {
        0
    }
}

/// One of the three banks of eight registers that describe every vector.
fn bank(access: Access, first: Register) -> [u32; VECTOR_WORDS] {
    let mut words = [0; VECTOR_WORDS];
    for (word, register) in words.iter_mut().zip(register::bank(first)) {
        *word = access.read(register);
    }
    words
}
