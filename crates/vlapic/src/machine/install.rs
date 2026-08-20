//! Building every controller the machine has, and putting this processor's into
//! the state firmware left the real one in.
//!
//! Once, on the boot processor, after the roster is taken and before any
//! processor is started: a controller has to exist before anything can deliver
//! to it, and an application processor's controller has to exist before that
//! processor does.
//!
//! # Nothing is published until everything that can fail has succeeded
//!
//! Both cells this crate keeps are written once and never cleared, so a failure
//! between them would be permanent and a retry would find half a machine. A
//! page installed without a doorbell is the worst of those halves: it can
//! deliver an interrupt to a processor inside the guest and has no way to make
//! it look.

use alloc::boxed::Box;

use apic::Controller;
use log::{info, warn};
use snapshot::FirmwareContext;

use crate::{
    VlapicError, avic,
    delivery::doorbell,
    hardware::{
        mirror::mirror_logical_destination,
        model::{self, Model},
        sources, timer,
    },
    machine::registry,
    registers::{
        Vlapic,
        base::{ApicBase, Mode},
    },
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
/// `x2apic_offered` is whether a guest on this machine may be given the
/// controller face its identifiers are reached through in model-specific
/// registers. It arrives here rather than with the structures hardware-driven
/// delivery runs on, which are built later, because the controller seeded below
/// may already *be* in that face — firmware routinely leaves it there — and
/// whether it may stay is this answer.
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
pub fn install(firmware: &FirmwareContext, x2apic_offered: bool) -> Result<(), VlapicError> {
    // Asked before anything is acquired so that a second call is cheap and
    // leaves nothing behind. It is also the whole of the check: `install` runs on
    // the boot processor before any other processor exists, so nothing can be
    // between this and the publication below — and a second check after the
    // publication would be an early return with a vector already acquired, which
    // is the one thing this ordering exists to make impossible.
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
    // Recorded before any controller exists, which is what makes it answerable
    // for the one seeded below and for every later question about the wider
    // face — the transition a guest writes, and the feature bit `CPUID`
    // reports.
    avic::activation::permit_x2apic(x2apic_offered);
    let local = apic::local()?;
    let here = local.id();
    let roster = cpu::roster()?;
    let model = Model::of_machine(local);
    model::describe("vlapic", model);
    // What the real controllers offer above their architectural registers, which
    // decides how a withheld acknowledgement is settled and nothing else. Read
    // from this processor's controller for the reason the model is, and given to
    // every controller rather than reached for later: it cannot change while the
    // machine runs, and a controller that answered the question differently at
    // two moments would be one that had settled some of its debts one way and
    // the rest the other.
    let extended = local.extended();
    let lapics: Box<[_]> = roster
        .entries()
        .iter()
        .map(|entry| {
            Vlapic::new(
                entry.index(),
                entry.apic_id(),
                // The processor running this is the one the machine came up on,
                // and its own controller is what says which processor that is.
                // Firmware lists processors in whatever order it likes, so taking
                // the flag from the roster's first entry gave the real bootstrap
                // processor a guest that believed it was an application one.
                entry.apic_id() == here,
                entry.startable(),
                model,
                extended,
            )
        })
        .collect();
    // Found once. Both things this processor's own controller needs — a
    // calibrated timer before anything is published, and firmware's registers
    // after everything is — are about the same row, and searching for it twice
    // was two answers to a question with one.
    let mine = lapics.iter().position(|vlapic| vlapic.apic_id() == here);
    if let Some(vlapic) = mine.and_then(|index| lapics.get(index)) {
        timer::calibrate(vlapic)?;
    }

    // Everything that can fail happens before either cell is published. A page
    // installed without a doorbell is a machine that can deliver an interrupt
    // to a processor inside the guest and has no way to make it look — and
    // because both cells are written once and never cleared, a failure between
    // them would be permanent and a retry would find the page already there.
    let doorbell = doorbell::acquire()?;

    let page = registry::publish(lapics);
    doorbell::publish(doorbell);
    info!(
        "vlapic: {} emulated controllers, reported as version {:#x}",
        page.all().len(),
        page.all().first().map_or(0, Vlapic::version)
    );
    // Said once, on the machine where it matters. The older face's identifier
    // field is eight bits, so on a machine with a processor whose identifier
    // needs more than that, a guest in that face reads two processors' registers
    // answering with the same number and can address only one of them — and the
    // one it reaches is whichever the roster happens to name first. Nothing here
    // can widen a field the architecture defines, and the guest is entitled to
    // use the face; what it is not entitled to is silence about it.
    if roster.needs_x2apic() {
        warn!(
            "vlapic: this machine has a processor whose identifier does not fit the older \
             interface's destination field, so a guest that uses that interface will find \
             identifiers aliased to their low eight bits"
        );
    }

    // Last, because it programs real hardware from what it seeds and so needs
    // everything that reaches hardware to be in place — and because a controller
    // that could not be seeded is still a working controller, so a machine whose
    // firmware left nothing to inherit is one this leaves at reset rather than
    // one it refuses.
    if let Some(vlapic) = mine.and_then(|index| page.all().get(index)) {
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
    // Said where it happens, because the guest cannot see it and nothing else
    // would: firmware had taken the real controller into the wider face and this
    // machine may not offer a guest that face, so the emulated one starts in the
    // older one — which is the face this guest's own `CPUID` describes. A machine
    // that reaches this is one whose delivery policy should have declined
    // hardware delivery on the same grounds, so the line is also how a policy
    // that did not would be found.
    if ApicBase::from_bits(interrupts.base).mode() == Mode::X2Apic && vlapic.mode() != Mode::X2Apic
    {
        warn!(
            "vlapic: {} starts in the older interface although firmware had left the real \
             controller in x2apic, because this machine may not offer a guest that face",
            vlapic.index()
        );
    }
    mirror_logical_destination(vlapic);
    if !sources::reprogram(vlapic) {
        warn!(
            "vlapic: {} could not put firmware's own sources back on real hardware, so a source \
             firmware was using may be left masked or armed with the wrong vector",
            vlapic.index()
        );
    }
    timer::inherit(vlapic, &interrupts.local, firmware.tsc);
    info!(
        "vlapic: {} inherited firmware's controller in {}, spurious {:#x}, task priority {}",
        vlapic.index(),
        vlapic.mode(),
        vlapic.spurious(),
        vlapic.task_priority(),
    );
}
