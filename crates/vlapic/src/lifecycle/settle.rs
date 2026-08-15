//! Applying a startup message, and quieting the machine behind a guest that has
//! stopped existing.
//!
//! Called by a processor about itself, at an exit boundary, which is what makes
//! the reset safe: nothing else is looking at this controller's registers, and
//! this processor is not part-way through injecting anything.

use log::{info, trace, warn};

use crate::{
    hardware::{sources, timer},
    registers::{Startup, StartupPage, Vlapic},
};

/// Applies whatever startup message arrived for this processor, and says what
/// it should do now.
///
/// Called by a processor about itself, at an exit boundary, which is what makes
/// the reset safe: nothing else is looking at this controller's registers, and
/// this processor is not part-way through injecting anything.
///
/// The two transitions are separate steps deliberately. Applying an `INIT`
/// leaves the processor waiting, and it stays waiting across however many exits
/// it takes for a start-up message to arrive — including none at all, which is
/// what an operating system that never uses a processor leaves it doing for the
/// rest of its life.
pub(crate) fn applied(vlapic: &Vlapic) -> Resumption {
    if vlapic.startup() == Startup::InitRequested {
        discharge(vlapic);
        // Exactly what hardware leaves behind: the identifier and the face
        // survive, everything else is as it was at reset.
        vlapic.reset_registers();
        vlapic.set_startup(Startup::WaitingForSipi);
        info!("vlapic: {} reset by an init and waiting", vlapic.index());
    }
    if vlapic.startup() != Startup::WaitingForSipi {
        return Resumption::Carry;
    }
    match vlapic.take_sipi() {
        Some(page) => {
            vlapic.set_startup(Startup::Running);
            info!(
                "vlapic: {} started at page {:#x}",
                vlapic.index(),
                page.number()
            );
            Resumption::StartAt(page)
        }
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
}
