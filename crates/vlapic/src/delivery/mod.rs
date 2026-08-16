//! Working out which processors a command names, and giving it to them.
//!
//! This is the one place a guest's interrupt command register becomes an
//! interrupt somewhere else. What a command asks for is
//! [`crate::registers::icr`]'s to decode and to judge; what is here is what
//! becomes of a command the architecture defines — which is one of five quite
//! different things, only two of which are an interrupt at all.
//!
//! # Delivering is setting a bit, and then telling somebody
//!
//! Both halves are needed and the second is the one easily forgotten: the
//! target may not be looking at its controller. [`doorbell`] is that half, and
//! the ordering that stops a wakeup being lost is stated there.
//!
//! # Nothing here fails in a way the guest can see
//!
//! A command naming a mode the architecture reserves, or a processor that does
//! not exist, is one real hardware would also do nothing useful with — so it is
//! recorded in the sender's error status and dropped, which is what a
//! controller does with a message nobody accepts.
//!
//! # Limitations
//!
//! A system-management interrupt directed through the interrupt command
//! register is not delivered: the guest's system-management mode is not
//! virtualised, and there is no state in which the target could execute an SMI
//! handler. Chipset-initiated system management is unaffected, since it
//! bypasses this controller entirely. Software that uses an interrupt-command
//! SMI to rendezvous its processors will wait indefinitely for the ones this
//! controller owns.

pub(crate) mod doorbell;

mod arbitration;
mod destination;
mod startup;

use log::{trace, warn};

use crate::{
    delivery::{
        arbitration::least_busy,
        destination::targets,
        doorbell::nudge,
        startup::{initialize, start},
    },
    machine::ownership,
    priority,
    registers::{
        Accepted, StartupPage, Vlapic,
        error::Errors,
        icr::{Command, Delivery},
    },
};

/// Delivers a command the guest wrote to its interrupt command register.
///
/// Nothing here fails in a way the guest can see. A command naming a mode the
/// architecture reserves, or a processor that does not exist, is one real
/// hardware would also do nothing useful with — so it is recorded and dropped,
/// which is what a controller does with a message nobody accepts.
pub(crate) fn send(from: &Vlapic, lapics: &[Vlapic], command: Command) {
    let mode = from.mode();
    let Some(delivery) = command.delivery(mode) else {
        // Lowest priority is the one reserved encoding with an error of its own:
        // the architecture has a controller that cannot send a redirectable
        // interrupt say so, rather than merely doing nothing.
        if command.wants_lowest_priority() {
            from.errors().record(Errors::REDIRECTABLE_IPI);
        }
        warn!(
            "vlapic: {} sent a command with a reserved delivery mode: {:#x}",
            from.index(),
            command.bits()
        );
        return;
    };
    // Judged whole, before a single processor is named. A command the
    // architecture does not define must not have reset, started or interrupted
    // half the machine by the time that is noticed.
    if !command.legal(mode) {
        warn!(
            "vlapic: {} sent a command no processor would send: {:#x}",
            from.index(),
            command.bits()
        );
        return;
    }
    // A synchronisation message that reloads arbitration identifiers and does
    // nothing else. No processor this hypervisor runs on arbitrates over a bus,
    // so there is nothing for it to reload.
    if command.is_init_deassert() {
        trace!(
            "vlapic: {} sent an init de-assert, which does nothing",
            from.index()
        );
        return;
    }
    // A vector no controller may deliver is the *sender's* error, and is caught
    // before the targets are worked out. Recording it on the receivers instead
    // would put the error on the wrong controllers — and on every one of them
    // for a single malformed broadcast, so that one guest mistake contaminated
    // the error status of the whole machine.
    if matches!(delivery, Delivery::Fixed | Delivery::LowestPriority)
        && !priority::legal(command.vector())
    {
        from.errors().record(Errors::SEND_ILLEGAL_VECTOR);
        return;
    }

    match delivery {
        // The chipset picks, and it picks by priority. Which priority, and how a
        // tie is broken, is the guest's own processor's rule rather than a
        // universal one.
        Delivery::LowestPriority => {
            if let Some(target) = least_busy(from, targets(from, lapics, command)) {
                accept(from, target, delivery, command);
            }
        }
        Delivery::Fixed => {
            for target in targets(from, lapics, command) {
                accept(from, target, delivery, command);
            }
        }
        // A non-maskable interrupt reaches a controller that is switched off or
        // software-disabled, which is the whole of what makes it non-maskable —
        // so it deliberately does not go through the acceptance that would
        // refuse it.
        Delivery::NonMaskable => {
            for target in targets(from, lapics, command) {
                raise(from, target);
            }
        }
        Delivery::Init => {
            for target in targets(from, lapics, command) {
                initialize(from, target);
            }
        }
        Delivery::Startup => {
            for target in targets(from, lapics, command) {
                start(from, target, StartupPage::new(command.vector().number()));
            }
        }
        // Deliberately not delivered, and this is a limitation of the machine
        // Pulzar presents rather than an oversight.
        //
        // Forwarding one to the real processor would take the *host* into
        // system-management mode over host state, running firmware's handler
        // against a context it was not written for, and the guest would see
        // nothing of it either way. Emulating one would need virtual
        // system-management machinery — a separate mode, its own save state, its
        // own memory aperture — that nothing in this hypervisor has.
        //
        // So the guest's interrupt command register cannot send this one thing,
        // and a guest whose firmware or operating system relies on an SMI
        // rendezvous will not get one.
        Delivery::SystemManagement => warn!(
            "vlapic: {} sent a system-management interrupt, which this machine does not deliver",
            from.index()
        ),
    }
}

/// Gives one processor an interrupt, and makes sure it notices.
///
/// A controller that is switched off or software-disabled refuses it, which is
/// what a real one does — and it is refused at the target rather than filtered
/// here, because whether a controller is accepting is the target's own state
/// and may change between the two.
fn accept(from: &Vlapic, target: &Vlapic, delivery: Delivery, command: Command) {
    if !ownership::owns(target) {
        refused(from, target, delivery);
        return;
    }
    let vector = command.vector();
    match target.accept(vector, command.trigger()) {
        // It is in the target's register file now, and the target may not be
        // looking at it.
        Accepted::Requested | Accepted::Coalesced => nudge(from, target),
        // The refusals the architecture defines, and both are the target's own
        // state to answer with rather than anything that went wrong.
        Accepted::Illegal | Accepted::Refused => trace!(
            "vlapic: {} offered {vector} to {}, which did not take it",
            from.index(),
            target.index()
        ),
        // Not one of those: the interrupt was not recorded anywhere, so nothing
        // will deliver it and nothing will report it but this.
        Accepted::Resetting => warn!(
            "vlapic: {} offered {vector} to {}, whose register file was being reset, and it was \
             not delivered",
            from.index(),
            target.index()
        ),
    }
}

/// Gives one processor a non-maskable interrupt, and makes sure it notices.
fn raise(from: &Vlapic, target: &Vlapic) {
    if !ownership::owns(target) {
        refused(from, target, Delivery::NonMaskable);
        return;
    }
    target.raise_nmi();
    nudge(from, target);
}

/// Records that a message could not be given to a processor this hypervisor
/// does not run, and says so once.
///
/// Such a processor is executing firmware's own code on real hardware and is
/// not looking at its emulated controller: a request bit set in it is one
/// nothing will ever consume, and a non-maskable interrupt counted there is one
/// the processor is handed as the first event it sees if the guest ever starts
/// it — in real mode, before it has an interrupt descriptor table to take it
/// through. INIT and start-up are the two messages that have somewhere else to
/// go in that window, and [`startup`] is where they go; there is nowhere to put
/// these.
///
/// Recorded as a message no processor accepted, which is exactly what happened,
/// and said once per re-armed error status rather than once per message: a
/// guest can send these as fast as it can write a register.
fn refused(from: &Vlapic, target: &Vlapic, delivery: Delivery) {
    if from.errors().record(Errors::SEND_ACCEPT) {
        warn!(
            "vlapic: {} sent {delivery:?} to {}, which this hypervisor does not run",
            from.index(),
            target.index()
        );
    }
}
