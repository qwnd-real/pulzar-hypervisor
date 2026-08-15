//! Building every controller the machine has, and putting this processor's into
//! the state firmware left the real one in.
//!
//! Once, on the boot processor, after the roster is taken and before any
//! processor is started: a controller has to exist before anything can deliver to
//! it, and an application processor's controller has to exist before that
//! processor does.
//!
//! # Nothing is published until everything that can fail has succeeded
//!
//! Both cells this crate keeps are written once and never cleared, so a failure
//! between them would be permanent and a retry would find half a machine. A page
//! installed without a doorbell is the worst of those halves: it can deliver an
//! interrupt to a processor inside the guest and has no way to make it look.

use alloc::boxed::Box;

use apic::Controller;
use log::{info, warn};
use snapshot::FirmwareContext;

use crate::{
    VlapicError,
    delivery::doorbell,
    hardware::{mirror::mirror_logical_destination, model::Model, sources, timer},
    machine::registry,
    registers::{Vlapic, base::ApicBase},
};

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
/// [`VlapicError::AlreadyInstalled`] for a second call,
/// [`VlapicError::MisplacedPage`] on a machine whose firmware put the register
/// page anywhere but where this hypervisor traps one, [`VlapicError::Cpu`] if
/// the roster has not been taken, [`VlapicError::Descriptors`] if no vector is
/// free for the sources this crate programs onto real hardware, or
/// [`VlapicError::Apic`] if this processor's own controller cannot be reached
/// to be asked which one it is.
pub fn install(firmware: &FirmwareContext) -> Result<(), VlapicError> {
    // Asked before anything is acquired so that a second call is cheap and
    // leaves nothing behind. It is not what makes this safe against two callers
    // at once — nothing is, and nothing needs to be: this runs on the boot
    // processor before any other processor exists.
    if registry::installed() {
        return Err(VlapicError::AlreadyInstalled);
    }
    // Also before anything is acquired, because it is a refusal of the machine
    // rather than of this call. One page is trapped, at one address, and it is
    // the only thing standing between a guest and the real local APIC — so a
    // machine whose firmware put the page somewhere else is one no guest may be
    // entered on at all.
    if let Some(page) =
        ApicBase::misplaced(firmware.interrupts.controller, firmware.interrupts.base)
    {
        return Err(VlapicError::MisplacedPage { page });
    }
    let here = apic::local()?.id();
    let roster = cpu::roster()?;
    // Firmware lists the boot processor first, and the bootstrap flag in the
    // base register records which processor the machine came up on. Nothing
    // else in the roster distinguishes it.
    let bootstrap = roster.entries().first().map(cpu::Entry::apic_id);
    let model = Model::of_machine();
    let lapics: Box<[_]> = roster
        .entries()
        .iter()
        .map(|entry| {
            Vlapic::new(
                entry.index(),
                entry.apic_id(),
                Some(entry.apic_id()) == bootstrap,
                entry.startable(),
                model,
            )
        })
        .collect();
    if let Some(vlapic) = lapics.iter().find(|vlapic| vlapic.apic_id() == here) {
        timer::calibrate(vlapic)?;
    }

    // Everything that can fail happens before either cell is published. A page
    // installed without a doorbell is a machine that can deliver an interrupt
    // to a processor inside the guest and has no way to make it look — and
    // because both cells are written once and never cleared, a failure between
    // them would be permanent and a retry would find the page already there.
    let doorbell = doorbell::acquire()?;

    let (page, built) = registry::publish(lapics);
    if !built {
        return Err(VlapicError::AlreadyInstalled);
    }
    doorbell::publish(doorbell);
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
