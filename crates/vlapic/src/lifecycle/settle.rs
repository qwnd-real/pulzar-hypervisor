//! Applying a startup message, and quieting the machine behind a guest that has
//! stopped existing.
//!
//! Called by a processor about itself, at an exit boundary, which is what makes
//! the reset safe: nothing else is looking at this controller's registers, and
//! this processor is not part-way through injecting anything.

use log::{info, trace, warn};

use crate::{
    hardware::{sources, timer},
    registers::{Phase, StartupPage, Vlapic},
};

/// Applies whatever startup message arrived for this processor, and says what
/// it should do now.
///
/// Called by a processor about itself, at an exit boundary, which is what makes
/// the reset safe: nothing else is looking at this controller's registers, and
/// this processor is not part-way through injecting anything.
///
/// The two transitions are separate steps deliberately. Applying an `INIT`
/// leaves an application processor waiting, and it stays waiting across however
/// many exits it takes for a start-up message to arrive — including none at
/// all, which is what an operating system that never uses a processor leaves it
/// doing for the rest of its life. What it does not do is *refuse* a start-up
/// message that arrives before the reset has been applied: the two are sent
/// microseconds apart, so the message is kept where the INIT can be seen to
/// precede it and is applied on the way out of this same function.
pub(crate) fn applied(vlapic: &Vlapic) -> Resumption {
    if matches!(vlapic.startup().phase(), Phase::InitRequested(_)) {
        discharge(vlapic);
        // Exactly what hardware leaves behind: the identifier and the face
        // survive, everything else is as it was at reset.
        vlapic.reset_registers();
        // Not part of the register file, and so not part of that: this is a count
        // of interrupts the processor has already been given, and an INIT is one
        // of the two transitions that discard one.
        vlapic.discard_nmi();
        let bootstrap = vlapic.base().bootstrap();
        vlapic.startup().initialized(bootstrap);
        info!(
            "vlapic: {} reset by an init and {}",
            vlapic.index(),
            if bootstrap { "restarting" } else { "waiting" }
        );
        if bootstrap {
            // Only application processors are held. The processor the machine
            // came up on is the one that sends the start-up messages, so a
            // bootstrap processor waiting for one would be waiting for itself —
            // and the guest would be down the processor every other one waits
            // on. It resumes at the reset vector instead, which is where the
            // architecture puts a processor an INIT has reset.
            return Resumption::Restart;
        }
    }
    match vlapic.startup().started() {
        Some(page) => {
            // The other transition that discards them, and for a sharper reason
            // than the INIT's: this processor is about to begin executing in real
            // mode with no interrupt descriptor table, and a non-maskable
            // interrupt the old guest was owed would be the first thing it took.
            vlapic.discard_nmi();
            info!(
                "vlapic: {} started at page {:#x}",
                vlapic.index(),
                page.number()
            );
            Resumption::StartAt(page)
        }
        None if vlapic.startup().running() => Resumption::Carry,
        None => Resumption::Wait,
    }
}

/// Quiets the machine behind a controller whose guest is being reset, and
/// settles everything real hardware is owed.
///
/// Both halves matter and they are separate failures. A source left armed goes
/// on delivering into a virtual processor that is reset and held — arrivals
/// nothing will ever take, against a register file that has been cleared. And a
/// level-triggered interrupt's real acknowledgement is deliberately withheld
/// until the guest acknowledges its own, so a guest that has just been reset
/// leaves debts nobody else will ever pay; the real controller would go on
/// holding those vectors in service, refusing everything of their priority or
/// lower on this processor for the rest of the machine's life.
///
/// Called by the processor about itself, which is what makes acknowledging
/// legitimate: an acknowledgement is to whichever controller the processor
/// issuing it is running on.
fn discharge(vlapic: &Vlapic) {
    let quiet = sources::quiesce(vlapic) & timer::disarm(vlapic);
    let settled = vlapic.ledger().settle(&apic::local().ok());
    if quiet && settled {
        trace!(
            "vlapic: {} quieted its sources and settled its debts for a guest that was reset",
            vlapic.index()
        );
        return;
    }
    warn!(
        "vlapic: {} was reset with sources {} and acknowledgements {}",
        vlapic.index(),
        if quiet { "quiet" } else { "still armed" },
        if settled { "settled" } else { "still owed" },
    );
}

/// What a processor should do after its startup state has been settled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resumption {
    /// Carry on running the guest.
    Carry,
    /// Reset and held, with no start-up message yet. Do not enter the guest.
    Wait,
    /// Begin executing the guest in real mode at the start of this page, which
    /// is what a start-up message names.
    StartAt(StartupPage),
    /// Reset, and running again from the machine's reset vector.
    ///
    /// What an INIT does to the processor the machine came up on: only
    /// application processors are held waiting to be started, because the
    /// bootstrap processor is the one that starts them.
    Restart,
}
