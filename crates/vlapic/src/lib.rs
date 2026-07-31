//! The local interrupt controller a guest sees, in place of the machine's own.
//!
//! Pulzar passes the platform through. It does not pass the interrupt
//! controller through, and the difference is the point: a guest that reached
//! the real controller would be a guest that could mask the host's interrupts,
//! acknowledge the host's, send the host's processors a reset, and rename the
//! processor an interrupt is addressed by. Every one of those is something a
//! hypervisor has to keep.
//!
//! So the controller is emulated and the hardware behind it stays the host's.
//! Every interrupt on the machine arrives in a host handler first, and reaches
//! the guest only as a decision this crate made.
//!
//! # What is emulated and what is not
//!
//! The register file is emulated in full: the request, in-service and
//! trigger-mode banks, the priorities, the error status and its write-then-read
//! protocol, the interrupt command register, both destination models, and the
//! whole of the base register's state machine.
//!
//! The *sources* are not. The timer is the real timer, programmed with the
//! guest's own divide, count and mode, because there is nothing to be gained by
//! counting the same numbers twice — and the same will hold for the rest of the
//! local vector table. What is never passed through is a vector: the real entry
//! carries a vector this crate claimed, so that an arrival is unambiguous, and
//! the guest's own vector is what gets injected.
//!
//! # One page, every processor
//!
//! The memory-mapped face is one page of guest physical memory at the same
//! address on every processor, each seeing its own controller through it. So
//! the device registered for that page is a single device holding one
//! controller per processor, picking out the row belonging to whichever
//! processor took the exit — which is exactly the shape of the hardware it
//! stands for.

#![no_std]

extern crate alloc;

mod access;
mod base;
mod delivery;
mod error;
mod icr;
mod lvt;
mod mmio;
mod msr;
mod priority;
mod register;
mod sources;
mod state;
mod timer;
mod vectors;

use alloc::boxed::Box;
use core::num::NonZeroU64;

use cpu::{CpuError, CpuIndex};
use descriptors::{DescriptorError, Vector};
use emulate::{Commit, Data, Device, Read, Region, Trap, Write};
use log::{info, warn};
use spin::Once;
use thiserror::Error;
use x86_64::PhysAddr;

use crate::{
    access::Written,
    base::ApicBase,
    icr::Trigger,
    lvt::Entry,
    mmio::Page,
    state::{Retired, Vlapic},
};
pub use crate::{
    delivery::Resumption,
    msr::{TSC_DEADLINE_MSR, claims},
};

/// Builds one controller per processor the machine has.
///
/// Called once, on the boot processor, after the roster is taken and before any
/// processor is started — a controller has to exist before anything can deliver
/// to it, and an application processor's controller has to exist before that
/// processor does.
///
/// # Errors
///
/// [`VlapicError::AlreadyInstalled`] for a second call, [`VlapicError::Cpu`] if
/// the roster has not been taken, or [`VlapicError::Descriptors`] if no vector
/// is free for the sources this crate programs onto real hardware.
pub fn install() -> Result<(), VlapicError> {
    let roster = cpu::roster()?;
    // Firmware lists the boot processor first, and the bootstrap flag in the
    // base register records which processor the machine came up on. Nothing
    // else in the roster distinguishes it.
    let bootstrap = roster.entries().first().map(cpu::Entry::apic_id);
    let lapics = roster
        .entries()
        .iter()
        .map(|entry| {
            Vlapic::new(
                entry.index(),
                entry.apic_id(),
                Some(entry.apic_id()) == bootstrap,
            )
        })
        .collect();

    let mut built = false;
    let page = LAPICS.call_once(|| {
        built = true;
        // Leaked rather than owned by this cell, because the device registered
        // for the guest's page needs the same controllers this crate reaches
        // for delivery, and a device is handed over as a boxed trait object.
        // One allocation for the life of the machine is the honest cost of
        // that.
        &*Box::leak(Box::new(Page::new(lapics)))
    });
    if !built {
        return Err(VlapicError::AlreadyInstalled);
    }
    let doorbell = ipi::register(rung, merge)?;
    DOORBELL.call_once(|| doorbell);
    info!(
        "vlapic: {} emulated controllers, reported as version {:#x}",
        page.all().len(),
        Vlapic::version()
    );
    Ok(())
}

/// The region of the guest's memory this crate answers for.
///
/// Handed to whatever traps regions before the guest runs. Every access is
/// trapped, reads included: the values a guest reads out of its controller are
/// this crate's answers and never the hardware's.
///
/// # Errors
///
/// [`VlapicError::NotInstalled`] before [`install`].
pub fn region() -> Result<Region, VlapicError> {
    let page = lapics()?;
    Ok(Region {
        gpa: PhysAddr::new(ApicBase::DEFAULT_PAGE),
        bytes: PAGE,
        trap: Trap::Everything,
        device: Box::new(Aperture(page)),
    })
}

/// What the guest reads from one of the controller's model-specific registers.
///
/// # Errors
///
/// [`VlapicError::NotInstalled`] before [`install`], [`VlapicError::NoLapic`]
/// on a processor with no controller, or [`VlapicError::Fault`] if the guest
/// should take a general protection fault for the access.
pub fn read_msr(index: u32) -> Result<u64, VlapicError> {
    let vlapic = current()?;
    msr::read(vlapic, index).map_err(|fault| {
        warn!("vlapic: refusing a read of {index:#x}: {fault:?}");
        VlapicError::Fault
    })
}

/// What a write to one of the controller's model-specific registers does.
///
/// # Errors
///
/// As [`read_msr`].
pub fn write_msr(index: u32, value: u64) -> Result<(), VlapicError> {
    let vlapic = current()?;
    let written = msr::write(vlapic, index, value).map_err(|fault| {
        warn!("vlapic: refusing a write of {value:#x} to {index:#x}: {fault:?}");
        VlapicError::Fault
    })?;
    acted(vlapic, written);
    Ok(())
}

/// Gives this processor's guest an interrupt that arrived on real hardware.
///
/// The seam every unclaimed interrupt reaches. What arrives is a physical
/// vector that nothing in the hypervisor claimed, which on a machine whose I/O
/// controllers are passed through means it was meant for the guest.
///
/// Whether real hardware may be acknowledged now is the whole of what is
/// decided here, and the controller itself is asked: it recorded, as it
/// accepted the interrupt, whether the interrupt arrived level triggered.
///
/// An edge-triggered interrupt is finished with once taken, so it is
/// acknowledged immediately and the guest is given it. A level-triggered one is
/// asserted until the guest's own driver deals with whatever raised it, so
/// acknowledging now would deliver it again at once — the acknowledgement is
/// withheld and becomes owed, and is issued when the guest acknowledges its
/// own.
///
/// # Errors
///
/// [`VlapicError::NotInstalled`] before [`install`], or [`VlapicError::Apic`]
/// if the real controller could not be asked or acknowledged.
pub fn arrived(vector: Vector) -> Result<(), VlapicError> {
    let vlapic = current()?;
    let local = apic::local()?;
    // One of the controller's own sources, which this crate programmed with a
    // vector of its own precisely so that this test is possible. What the guest
    // is owed is its vector for that entry, not the one it arrived on — and
    // nothing at all if the guest has the entry masked, since it asked not to
    // be told.
    let guest = match sources::arrived_on(vector) {
        Some(entry) => {
            let programmed = vlapic.lvt(entry);
            if programmed.masked() {
                local.end_of_interrupt()?;
                return Ok(());
            }
            programmed.vector()
        }
        None => vector,
    };
    let level = local.arrived_level(vector)?;
    if level {
        // The debt is recorded against the vector the *guest* will acknowledge,
        // because that acknowledgement is what discharges it — and recorded
        // before the guest is given the interrupt, so that a guest which
        // acknowledges immediately finds the debt already there.
        vlapic.defer_acknowledgement(guest);
        vlapic.accept(guest, Trigger::Level);
    } else {
        vlapic.accept(guest, Trigger::Edge);
        local.end_of_interrupt()?;
    }
    Ok(())
}

/// Logs what each processor's controller is doing.
///
/// What is worth having here is the state that says something is stuck: a
/// controller with interrupts requested and never taken, or one still owing
/// real hardware an acknowledgement, is the shape both an interrupt storm and a
/// lost wakeup show up as.
pub fn describe(who: &str) {
    let Ok(page) = lapics() else {
        info!("{who}: the emulated controllers have not been installed");
        return;
    };
    for vlapic in page.all() {
        info!(
            "{who}: {} {} in {}{}, task priority {}, {} requested, {} in service{}",
            vlapic.index(),
            vlapic.apic_id(),
            vlapic.mode(),
            if vlapic.base().bootstrap() {
                " as the bootstrap processor"
            } else {
                ""
            },
            vlapic.task_priority(),
            vlapic.requested_count(),
            vlapic.in_service_count(),
            if vlapic.owes_acknowledgement() {
                ", owing hardware an acknowledgement"
            } else {
                ""
            },
        );
        if let Some(vector) = vlapic.requested() {
            info!(
                "{who}: {} has {vector} requested at processor priority {}",
                vlapic.index(),
                vlapic.processor_priority()
            );
        }
    }
}

/// Records that this hypervisor now runs this processor, so that a startup
/// message aimed at it is emulated rather than forwarded to real hardware.
///
/// Called by each processor as it comes up, after it has a controller and
/// before anything can be sent to it.
///
/// # Errors
///
/// As [`read_msr`].
pub fn claim_processor() -> Result<(), VlapicError> {
    current().map(Vlapic::take_ownership)
}

/// What this processor should do before entering the guest again, having
/// applied whatever startup message arrived for it.
///
/// # Errors
///
/// As [`read_msr`].
pub fn settle() -> Result<Resumption, VlapicError> {
    current().map(delivery::settle)
}

/// The highest-priority interrupt this processor's guest should take now, moved
/// from requested to in service.
///
/// Answers `None` when nothing is requested, or when what is requested does not
/// outrank what the guest is already servicing. The caller must actually
/// deliver whatever it is given: a vector taken here has already been moved.
///
/// # Errors
///
/// As [`read_msr`].
pub fn take_deliverable() -> Result<Option<Vector>, VlapicError> {
    current().map(Vlapic::take_deliverable)
}

/// Whether this processor's guest is owed a non-maskable interrupt, taking it
/// if so.
///
/// # Errors
///
/// As [`read_msr`].
pub fn take_nmi() -> Result<bool, VlapicError> {
    current().map(Vlapic::take_nmi)
}

/// Every model-specific register the guest's controller answers for.
///
/// Handed to whatever programs the permission map. The whole of the range the
/// architecture reserves for the controller is named, not merely the indices
/// that hold a register: reaching an unassigned one is a fault the guest is
/// entitled to, and it cannot be given one by code that never sees the access.
pub fn intercepted() -> impl Iterator<Item = u32> {
    (register::X2APIC_BASE_MSR..=register::X2APIC_LAST_MSR).chain([ApicBase::MSR, TSC_DEADLINE_MSR])
}

/// Records that this processor's guest is owed a non-maskable interrupt.
///
/// Called from the host's own handler for one that arrived while the host was
/// running — in the window between a world switch restoring host state and the
/// next entry — which the host takes and the guest is still owed.
///
/// # Errors
///
/// As [`read_msr`].
pub fn raise_nmi() -> Result<(), VlapicError> {
    current().map(Vlapic::raise_nmi)
}

/// Records the task priority the guest set while it was running.
///
/// With interrupt masking virtualized, a guest's writes to its task priority
/// through the control register do not exit — the processor keeps them in the
/// control block instead. So the value is read back out of it at every exit,
/// and this is where it lands.
///
/// A failure is deliberately not reported: this is called on the exit path
/// before anything has been decided, and a processor with no controller has
/// nothing that could want the value.
pub fn observe_task_priority(priority: u8) {
    if let Ok(vlapic) = current() {
        vlapic.observe_task_priority(priority);
    }
}

/// Records whether this processor is inside the guest.
///
/// The second half of the protocol that stops an interrupt being lost to a
/// processor that was entering the guest as it arrived. A caller must store
/// `true` and then consult [`take_deliverable`] once more before it actually
/// enters, abandoning the entry if something appeared in between.
///
/// # Errors
///
/// As [`read_msr`].
pub fn set_in_guest(inside: bool) -> Result<(), VlapicError> {
    current().map(|vlapic| vlapic.set_in_guest(inside))
}

/// Interrupts a processor that is inside the guest, so that it looks at its
/// controller.
///
/// # Errors
///
/// [`VlapicError::NotInstalled`] if the doorbell has not been registered, or
/// whatever sending it reported.
fn doorbell(target: CpuIndex) -> Result<(), VlapicError> {
    let doorbell = DOORBELL.get().ok_or(VlapicError::NotInstalled)?;
    doorbell.send(target, RING)?;
    Ok(())
}

/// Whether this hypervisor runs the processor a controller belongs to.
fn owns(vlapic: &Vlapic) -> bool {
    vlapic.owned()
}

/// This processor's controller.
///
/// # Errors
///
/// [`VlapicError::NotInstalled`] before [`install`], or
/// [`VlapicError::NoLapic`] if the roster does not describe this processor.
fn current() -> Result<&'static Vlapic, VlapicError> {
    // SAFETY: nothing calls this before the processor has attached — a guest
    // cannot be entered until bring-up is past that point, and the interrupt
    // path is reached only from a processor that has — so this processor's `GS`
    // base points at its own block.
    let index = unsafe { cpu::current() }.index();
    lapics()?.all().get(index.get()).ok_or(VlapicError::NoLapic)
}

/// Performs what a guest's write asked for beyond the value being stored.
///
/// Shared by both faces deliberately: a guest that sent an interrupt through
/// the memory-mapped command register and one that sent it through a
/// model-specific register have asked for exactly the same thing, and this is
/// where that stops being two code paths.
pub(crate) fn acted(vlapic: &Vlapic, written: Written) {
    match written {
        Written::Nothing => {}
        Written::EndOfInterrupt => retire(vlapic),
        Written::Timer => timer::reprogram(vlapic, sources::hardware_vector(Entry::Timer)),
        Written::LocalVectorTable => sources::reprogram(vlapic),
        Written::LogicalDestination => mirror_logical_destination(vlapic),
        Written::ModeChanged => {
            info!("vlapic: {} entered {}", vlapic.index(), vlapic.mode());
        }
        Written::Command(command) => match lapics() {
            Ok(page) => delivery::send(vlapic, page.all(), command),
            Err(error) => warn!("vlapic: a command could not be delivered: {error}"),
        },
        Written::SelfIpi(vector) => {
            vlapic.accept(vector, Trigger::Edge);
        }
    }
}

/// Retires the interrupt the guest says it has finished with, acknowledging
/// real hardware if it was waiting for exactly this.
fn retire(vlapic: &Vlapic) {
    let Some(Retired {
        vector,
        acknowledge_hardware,
    }) = vlapic.end_of_interrupt()
    else {
        // The architecture defines acknowledging nothing as doing nothing.
        return;
    };
    if !acknowledge_hardware {
        return;
    }
    if let Err(error) = apic::local().and_then(apic::LocalApic::end_of_interrupt) {
        warn!(
            "vlapic: {} could not acknowledge {vector}: {error}",
            vlapic.index()
        );
    }
}

/// Tells the real controller which logical destinations this processor answers
/// to.
///
/// Necessary because the I/O controllers are passed through: the guest programs
/// them directly with logical destinations, and hardware matches those against
/// the *real* register. A disagreement is an interrupt delivered to the wrong
/// processor or to none, which is how a guest ends up unable to find its own
/// disk.
fn mirror_logical_destination(vlapic: &Vlapic) {
    let Ok(local) = apic::local() else {
        return;
    };
    let outcome = local
        .set_destination_format(vlapic.destination_format())
        .and_then(|()| local.set_logical_destination(vlapic.logical_destination()));
    if let Err(error) = outcome {
        warn!(
            "vlapic: {} could not mirror its logical destination: {error}",
            vlapic.index()
        );
    }
}

/// The controllers, once they exist.
fn lapics() -> Result<&'static Page, VlapicError> {
    LAPICS.get().copied().ok_or(VlapicError::NotInstalled)
}

/// What answers for the guest's register page.
///
/// A thin wrapper rather than the controllers themselves, because a device is
/// handed over as an owned trait object and the same controllers are reached
/// from elsewhere for delivery.
#[derive(Debug)]
struct Aperture(&'static Page);

impl Device for Aperture {
    fn read(&self, access: Read) -> Data {
        self.0.read(access)
    }

    fn write(&self, access: Write) -> Commit {
        self.0.write(access)
    }
}

/// Built once, by the boot processor, before any other processor is started.
static LAPICS: Once<&'static Page> = Once::new();

/// How a processor inside the guest is made to leave it.
static DOORBELL: Once<ipi::Ipi> = Once::new();

/// What a doorbell carries, which is nothing.
///
/// The arrival is the whole message: it forces the target out of the guest, and
/// what to do about that is decided by reading the controller, not by reading a
/// payload. A constant is needed only because a send must carry something.
const RING: NonZeroU64 = NonZeroU64::new(1).unwrap();

/// What runs on a processor a doorbell was sent to.
///
/// Deliberately empty. Ringing it has already done the only thing it was for —
/// the guest has left, and the exit loop consults the controller on its way
/// back in.
fn rung(_: ipi::Request) {}

/// How two outstanding doorbells become one.
///
/// They carry nothing, so there is nothing to combine, and a processor that
/// left the guest once has left it for both.
fn merge(first: NonZeroU64, _: NonZeroU64) -> NonZeroU64 {
    first
}

/// How long the memory-mapped register page is.
const PAGE: u64 = 4096;

/// Why the emulated controllers could not be set up or driven.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum VlapicError {
    /// Nothing has built the controllers yet.
    #[error("the emulated controllers have not been installed")]
    NotInstalled,
    /// They have, and there is one set for the machine.
    #[error("the emulated controllers have already been installed")]
    AlreadyInstalled,
    /// The roster does not describe the processor asking.
    #[error("this processor has no emulated controller")]
    NoLapic,
    /// The guest should take a general protection fault for what it asked.
    #[error("the guest's access to its controller is not one the architecture allows")]
    Fault,
    /// The real controller refused something.
    #[error(transparent)]
    Apic(#[from] apic::ApicError),
    /// The processor roster refused something.
    #[error(transparent)]
    Cpu(#[from] CpuError),
    /// No vector was free for a source this crate programs onto real hardware.
    #[error(transparent)]
    Descriptors(#[from] DescriptorError),
    /// A processor inside the guest could not be interrupted.
    #[error(transparent)]
    Ipi(#[from] ipi::IpiError),
}
