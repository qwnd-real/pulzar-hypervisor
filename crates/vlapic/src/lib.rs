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
mod ledger;
mod lvt;
mod mmio;
mod model;
mod msr;
mod priority;
mod register;
mod sources;
mod state;
mod timer;
mod vectors;

use alloc::boxed::Box;
use core::num::NonZeroU64;

use apic::{Controller, IA32_TSC_DEADLINE};
use cpu::{CpuError, CpuIndex};
use descriptors::{DescriptorError, Vector};
use emulate::{Capability, Commit, Data, Device, Read, Region, Trap, Write};
use log::{info, trace, warn};
use snapshot::FirmwareContext;
use spin::Once;
use thiserror::Error;
use x86_64::PhysAddr;

use crate::{
    access::Written,
    base::{ApicBase, Mode},
    error::Errors,
    icr::Trigger,
    mmio::Page,
    model::Model,
    state::{Accepted, Transition, Vlapic},
};
pub use crate::{delivery::Resumption, msr::claims};

/// Builds one controller per processor the machine has, and puts this
/// processor's into the state firmware left the real one in.
///
/// Called once, on the boot processor, after the roster is taken and before any
/// processor is started — a controller has to exist before anything can deliver
/// to it, and an application processor's controller has to exist before that
/// processor does.
///
/// `firmware` is the capture taken before anything had overwritten it. Only
/// this processor's controller is seeded from it, because it is the only
/// processor the capture describes and the only one whose guest has ever run:
/// every other processor joins the guest held, so the guest believes it was
/// never started, and a processor that was never started has a controller at
/// reset.
///
/// The whole context is taken rather than its interrupt half alone because the
/// two fields read here belong together: the timer's counts are stale from the
/// instant they were read, and the timestamp counter beside them is what says
/// by how much.
///
/// # Errors
///
/// [`VlapicError::AlreadyInstalled`] for a second call, [`VlapicError::Cpu`] if
/// the roster has not been taken, [`VlapicError::Descriptors`] if no vector is
/// free for the sources this crate programs onto real hardware, or
/// [`VlapicError::Apic`] if this processor's own controller cannot be reached
/// to be asked which one it is.
pub fn install(firmware: &FirmwareContext) -> Result<(), VlapicError> {
    // Asked before anything is acquired so that a second call is cheap and
    // leaves nothing behind. It is not what makes this safe against two callers
    // at once — nothing is, and nothing needs to be: this runs on the boot
    // processor before any other processor exists.
    if LAPICS.is_completed() {
        return Err(VlapicError::AlreadyInstalled);
    }
    let here = apic::local()?.id();
    let roster = cpu::roster()?;
    // Firmware lists the boot processor first, and the bootstrap flag in the
    // base register records which processor the machine came up on. Nothing
    // else in the roster distinguishes it.
    let bootstrap = roster.entries().first().map(cpu::Entry::apic_id);
    let model = Model::of_machine();
    let lapics = roster
        .entries()
        .iter()
        .map(|entry| {
            Vlapic::new(
                entry.index(),
                entry.apic_id(),
                Some(entry.apic_id()) == bootstrap,
                model,
            )
        })
        .collect();

    // Everything that can fail happens before either cell is published. A page
    // installed without a doorbell is a machine that can deliver an interrupt
    // to a processor inside the guest and has no way to make it look — and
    // because both cells are written once and never cleared, a failure between
    // them would be permanent and a retry would find the page already there.
    let doorbell = ipi::register(rung, merge)?;

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
    DOORBELL.call_once(|| doorbell);
    info!(
        "vlapic: {} emulated controllers, reported as version {:#x}",
        page.all().len(),
        page.all().first().map_or(0, Vlapic::version)
    );

    // Last, because it programs real hardware from what it seeds and so needs
    // everything that reaches hardware to be in place — and because a controller
    // that could not be seeded is still a working controller, so a machine whose
    // firmware left nothing to inherit is one this leaves at reset rather than
    // one it refuses.
    if let Some(vlapic) = page.all().iter().find(|vlapic| vlapic.apic_id() == here) {
        inherit(vlapic, firmware);
    } else {
        warn!("vlapic: {here} is not in the roster, so nothing inherited firmware's controller");
    }
    Ok(())
}

/// Puts one controller into the state firmware left the real one in, and brings
/// real hardware into agreement with it.
///
/// Both halves are necessary and the second is the one easily forgotten. The
/// host's own bring-up masked every source, stopped the timer, zeroed the task
/// priority and re-vectored the spurious and error entries — so a controller
/// seeded with firmware's registers and left there would describe a machine
/// that no longer exists. Programming hardware from the seeded values is what
/// makes firmware's timer tick again and its pins deliver again.
fn inherit(vlapic: &Vlapic, firmware: &FirmwareContext) {
    let interrupts = &firmware.interrupts;
    if interrupts.controller != Controller::Read {
        // Not a failure, and worth saying rather than passing over: a controller
        // firmware had switched off, or that this processor does not have, left
        // nothing to inherit, and the reset state the guest gets instead is the
        // honest answer for it.
        info!(
            "vlapic: {} has nothing to inherit, firmware's controller was {:?}",
            vlapic.index(),
            interrupts.controller
        );
        return;
    }
    if ApicBase::relocated(interrupts.base) {
        // The page is trapped once, before any guest runs, and nothing here can
        // re-trap a range while processors are executing. Firmware that had
        // moved its register page therefore resumes to find it back at the
        // default address, which is a way in which this machine is narrower than
        // the one it describes and is not something to discover from a symptom.
        warn!(
            "vlapic: {} inherits firmware's register page from {:#x} moved to {:#x}, which is the \
             only address this hypervisor traps",
            vlapic.index(),
            ApicBase::page_of(interrupts.base),
            ApicBase::DEFAULT_PAGE
        );
    }
    vlapic.seed(&interrupts.local, interrupts.base);
    mirror_logical_destination(vlapic);
    sources::reprogram(vlapic);
    timer::inherit(vlapic, &interrupts.local, firmware.tsc);
    info!(
        "vlapic: {} inherited firmware's controller in {}, spurious {:#x}, task priority {}",
        vlapic.index(),
        vlapic.mode(),
        vlapic.spurious(),
        vlapic.task_priority(),
    );
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
    // Nothing is translated, and nothing has to be. Every source the guest can
    // reach is programmed onto real hardware with the guest's own vector, so
    // the number an interrupt arrived on is already the number the guest is
    // owed — and a source the guest has masked was programmed masked and did
    // not deliver at all.
    if !local.arrived_level(vector) {
        vlapic.accept(vector, Trigger::Edge);
        local.end_of_interrupt();
        trace!(
            "vlapic: {} received edge {vector}, acknowledged and requested it",
            vlapic.index()
        );
        return Ok(());
    }
    // The debt is recorded before the guest is given the interrupt, so that a
    // guest which acknowledges immediately finds the debt already there.
    vlapic.ledger().owe(vector);
    match vlapic.accept(vector, Trigger::Level) {
        Accepted::Requested | Accepted::Coalesced => trace!(
            "vlapic: {} received level {vector}, owing real hardware an acknowledgement",
            vlapic.index()
        ),
        refused => {
            // The guest was not given it and will therefore never acknowledge
            // it, so the only thing that could ever have discharged the debt
            // does not exist. Settling here is what stops a refused interrupt
            // occupying a real in-service slot for the life of the machine,
            // blocking everything of its priority or lower on this processor.
            vlapic.ledger().release(vector);
            trace!(
                "vlapic: {} received level {vector} but is not accepting it: {refused:?}",
                vlapic.index()
            );
        }
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
            "{who}: {} {} {} in {}{}, task priority {}, {} requested, {} in service{}",
            vlapic.index(),
            vlapic.apic_id(),
            if vlapic.running() {
                "running"
            } else {
                "waiting to be started"
            },
            vlapic.mode(),
            if vlapic.base().bootstrap() {
                " as the bootstrap processor"
            } else {
                ""
            },
            vlapic.task_priority(),
            vlapic.requested_count(),
            vlapic.in_service_count(),
            if vlapic.ledger().is_empty() {
                ""
            } else {
                ", owing hardware an acknowledgement"
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
/// Called by each processor as it comes up, as early as it can be: until it has
/// been called, a startup message the guest aims at this processor is sent to
/// real hardware, and real hardware would reset the host out from under it.
///
/// `joining` is where the guest's own view of this processor stands, which is
/// not the same question as whether the processor is running. Every processor
/// but the one the guest was entered on has never been started *by the guest*,
/// however long it has been executing the hypervisor's own code.
///
/// # Errors
///
/// As [`read_msr`].
pub fn claim_processor(joining: Joining) -> Result<(), VlapicError> {
    current().map(|vlapic| {
        vlapic.set_startup(joining.startup());
        vlapic.take_ownership();
    })
}

/// Where a processor's guest stands at the moment the hypervisor takes the
/// processor over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Joining {
    /// Already executing the guest, which is true of exactly one processor: the
    /// one the guest was entered on.
    Running,
    /// Never started by the guest, so it holds until the guest starts it —
    /// exactly as a processor still in reset would.
    WaitingForSipi,
}

impl Joining {
    /// The startup state this is.
    const fn startup(self) -> state::Startup {
        match self {
            Self::Running => state::Startup::Running,
            Self::WaitingForSipi => state::Startup::WaitingForSipi,
        }
    }
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

/// Waits, without running the guest, until a startup message arrives for this
/// processor.
///
/// What a processor whose guest has been reset does instead of spinning: there
/// is nothing to run, and there will be nothing to run until another processor
/// starts this one, which may never happen. The processor halts, and the
/// interrupt that wakes it is either the doorbell that says a message arrived
/// or something unrelated — so a caller consults [`settle`] again rather than
/// assuming the wait ended for the reason it was entered.
///
/// # Errors
///
/// As [`read_msr`].
pub fn hold() -> Result<(), VlapicError> {
    let vlapic = current()?;
    // Said before the message is looked for, and the reason is the whole of why
    // this is not a bare halt: a sender that misses this flag is one whose
    // message the test below finds, and a test that misses the message is one
    // the sender's doorbell wakes.
    vlapic.set_away(true);
    descriptors::wait_until(|| vlapic.signalled());
    vlapic.set_away(false);
    Ok(())
}

/// Whether this processor's guest is running, rather than reset and waiting to
/// be started again.
///
/// Consulted on the exit path, where a startup message that arrived while the
/// guest was running is a reason to stop running it.
///
/// # Errors
///
/// As [`read_msr`].
pub fn running() -> Result<bool, VlapicError> {
    current().map(Vlapic::running)
}

/// The highest-priority interrupt this processor's guest should take now, left
/// where it is.
///
/// Answers `None` when nothing is requested, when what is requested does not
/// outrank what the guest is already servicing, or when the controller is not
/// in a state that delivers anything.
///
/// Nothing is consumed. Whether the guest can actually be given this is not the
/// controller's to know — the processor may already have an event part-way
/// through delivery, or a non-maskable interrupt that goes first, or an
/// interrupt window that is shut — so the caller decides, and reports back
/// through [`committed`]. A controller that moved a vector out of the request
/// register for an injection that then did not happen would have thrown the
/// interrupt away, and for a level-triggered one would have stranded the real
/// acknowledgement owed for it as well.
///
/// # Errors
///
/// As [`read_msr`].
pub fn select() -> Result<Option<Vector>, VlapicError> {
    current().map(Vlapic::select)
}

/// Records that the guest really has been given `vector`, moving it from
/// requested to in service.
///
/// The other half of [`select`], and the only thing that consumes a request.
/// Called once an injection is known to have happened.
///
/// # Errors
///
/// As [`read_msr`].
pub fn committed(vector: Vector) -> Result<(), VlapicError> {
    current().map(|vlapic| {
        if !vlapic.committed(vector) {
            // The request was withdrawn between the two halves, which a reset
            // arriving in that window does. Nothing is put in service: the
            // interrupt belonged to a guest that no longer exists.
            trace!(
                "vlapic: {} was given {vector}, which its controller no longer had requested",
                vlapic.index()
            );
        }
    })
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
    (register::X2APIC_BASE_MSR..=register::X2APIC_LAST_MSR)
        .chain([ApicBase::MSR, IA32_TSC_DEADLINE])
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

/// The task priority this processor's guest has set.
///
/// The value the exit loop pushes into the control block's virtual task
/// priority, so that a guest which wrote the emulated register through the
/// page or a model-specific register finds its `CR8` answering the same
/// number — the two are one register on real hardware and must stay one.
///
/// # Errors
///
/// As [`read_msr`].
pub fn task_priority() -> Result<u8, VlapicError> {
    current().map(|vlapic| vlapic.task_priority().get())
}

/// Records whether this processor has stopped looking at its controller.
///
/// The second half of the protocol that stops an interrupt being lost to a
/// processor that was entering the guest as it arrived. A caller must store
/// `true` and then consult [`take_deliverable`] once more before it actually
/// enters, abandoning the entry if something appeared in between — and store
/// `false` on the way out, because a processor answering an exit will consult
/// its controller again on its own and needs nothing to remind it.
///
/// # Errors
///
/// As [`read_msr`].
pub fn set_away(away: bool) -> Result<(), VlapicError> {
    current().map(|vlapic| vlapic.set_away(away))
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
        // The architecture defines acknowledging nothing as doing nothing, and
        // discharging whatever real hardware was owed for it is part of the
        // acknowledgement rather than something done after it.
        Written::EndOfInterrupt => {
            vlapic.end_of_interrupt();
        }
        Written::Timer => {
            timer::reprogram(vlapic);
        }
        // The one write that starts a counting timer. What it delivers and in
        // which mode was settled when those registers were written, so this
        // does not reconfigure anything — reconfiguring here is what would move
        // the phase of a periodic tick on every unrelated write.
        Written::TimerStarted => timer::reload(vlapic),
        Written::TimerDeadline(deadline) => timer::arm_deadline(vlapic, deadline),
        Written::LocalVectorTable => {
            sources::reprogram(vlapic);
        }
        // Software-disabling masked every stored entry, and the timer is
        // programmed from its own entry rather than with the rest, so both have
        // to follow. Neither is disarmed: masking suppresses delivery and does
        // not stop a count the guest may still be reading.
        Written::Disabled => {
            sources::reprogram(vlapic);
            timer::reprogram(vlapic);
        }
        Written::LogicalDestination => mirror_logical_destination(vlapic),
        Written::ModeChanged(transition) => entered(vlapic, transition),
        Written::Command(command) => match lapics() {
            Ok(page) => delivery::send(vlapic, page.all(), command),
            Err(error) => warn!("vlapic: a command could not be delivered: {error}"),
        },
        // A guest sending itself an interrupt is the sender, so a vector no
        // controller may deliver is its error to be told about rather than the
        // receiver's — even though the two are the same controller here.
        Written::SelfIpi(vector) => {
            if priority::legal(vector) {
                vlapic.accept(vector, Trigger::Edge);
            } else {
                vlapic.errors().record(Errors::SEND_ILLEGAL_VECTOR);
            }
        }
    }
}

/// Brings real hardware across a change of face, and says so.
///
/// The virtual half of the transition has already happened: whatever the
/// architecture does not preserve was reset, and where it does not preserve
/// anything the sources were quieted and the debts settled first. What is left
/// is to bring the machine across after it — which means taking the real
/// controller into the same face the guest just entered, and then programming
/// it from whatever the emulated controller now holds. The order is not a
/// preference: every register written below is written through whichever face
/// the real controller presents, and the logical destination in particular is
/// only reachable in one of them.
///
/// None of it disturbs a running timer. Reprogramming says what the timer
/// delivers and how fast it counts, and the count and the deadline are left
/// where they were — which is what carries a guest's armed timer across the one
/// transition the architecture preserves it across.
fn entered(vlapic: &Vlapic, transition: Transition) {
    match transition {
        Transition::Unchanged => return,
        // Worth a line rather than a trace: real hardware was left holding
        // something across a boundary the guest believes cleared it, and that is
        // a state nothing later in the guest's life will explain.
        Transition::Changed { quiet, settled } if !quiet || !settled => warn!(
            "vlapic: {} changed face without fully settling hardware: sources {}, \
             acknowledgements {}",
            vlapic.index(),
            if quiet { "quiet" } else { "still armed" },
            if settled { "settled" } else { "still owed" },
        ),
        Transition::Preserved | Transition::Changed { .. } => {}
    }
    promote(vlapic);
    mirror_logical_destination(vlapic);
    sources::reprogram(vlapic);
    timer::reprogram(vlapic);
    info!("vlapic: {} entered {}", vlapic.index(), vlapic.mode());
}

/// Brings the real controller's logical destination into agreement with the
/// guest's.
///
/// Necessary because the I/O controllers are passed through: the guest programs
/// them directly with logical destinations, and hardware matches those against
/// the *real* register. A disagreement is an interrupt delivered to the wrong
/// processor or to none, which is how a guest ends up unable to find its own
/// disk.
///
/// How agreement is reached is not the same in the two faces, and neither is a
/// failure. In the older face the registers are writable and the guest's values
/// are written. In x2APIC they are not writable at all — hardware derives the
/// identifier from this processor's own, and there is only the one destination
/// model — so nothing is written and nothing needs to be: the emulated
/// controller derives its answer by the architecture's rule from the *real*
/// identifier, which is the same rule applied to the same number. That is the
/// whole reason [`promote`] exists, and it is checked here rather than trusted,
/// because a silent disagreement in this register is exactly the fault this
/// function is for.
fn mirror_logical_destination(vlapic: &Vlapic) {
    let Ok(local) = apic::local() else {
        warn!(
            "vlapic: {} could not reach its controller to mirror its logical destination",
            vlapic.index()
        );
        return;
    };
    let wanted = vlapic.logical_destination();
    if local.set_logical_routing(vlapic.destination_format(), wanted) {
        return;
    }
    // The register is hardware's in this face. Reading it back is the only way
    // to know the two really do agree, and a mismatch means passed-through
    // interrupts are being matched against something the guest never asked for.
    let real = local.logical_destination();
    if real == wanted {
        trace!(
            "vlapic: {} answers logical destination {real:#x}, which its guest derives too",
            vlapic.index()
        );
        return;
    }
    warn!(
        "vlapic: {} answers logical destination {real:#x} and its guest believes {wanted:#x}; \
         interrupts addressed logically will not reach it",
        vlapic.index()
    );
}

/// Takes the real controller into x2APIC behind a guest that has just gone
/// there.
///
/// The one thing that makes logical destinations pass through at all. A guest
/// in x2APIC addresses interrupts by an identifier the architecture *derives*
/// from the processor's own, and programs that identifier straight into
/// passed-through I/O controllers and device messages — none of which this
/// hypervisor intercepts. Hardware then matches those against the real
/// controller's own register, which in the older face holds something written
/// by the host and spelled differently. There is no value the host could write
/// that would agree: the two faces encode a logical identifier differently, and
/// the x2APIC one is read-only. So the real controller is moved into the same
/// face, where hardware derives the identifier from the same number the guest
/// derived it from, and the two agree because they are the same computation.
///
/// Only ever into x2APIC, and only when the guest is already there. A guest
/// that switches its controller off is not followed: the host needs its own
/// controller for the doorbells and shootdowns that keep the machine running,
/// and an emulated controller that is off already refuses everything offered to
/// it.
fn promote(vlapic: &Vlapic) {
    if vlapic.mode() != Mode::X2Apic {
        return;
    }
    let Ok(local) = apic::local() else {
        return;
    };
    if local.mode() == apic::Mode::X2Apic {
        return;
    }
    match local.enter_x2apic() {
        Ok(()) => info!(
            "vlapic: {} took its real controller into x2apic behind its guest",
            vlapic.index()
        ),
        // Worth a line of its own rather than being folded into the mirror
        // warning below it: this is the reason the two will disagree, and it says
        // the machine cannot do what the guest asked rather than that something
        // went wrong doing it.
        Err(error) => warn!(
            "vlapic: {} could not take its real controller into x2apic: {error}",
            vlapic.index()
        ),
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
    fn capability(&self) -> Capability {
        self.0.capability()
    }

    fn read(&self, access: Read<'_>) -> Data {
        self.0.read(access)
    }

    fn write(&self, access: Write<'_>) -> Commit {
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
