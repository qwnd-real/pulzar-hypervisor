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
//! The bit goes into the target's own register file even where the hardware is
//! driving that controller, so the second half is the same signal in both
//! cases: what the hardware delivers from is a backing page this crate writes
//! only on the processor it belongs to, and the exit a host interrupt forces is
//! what carries the request into it. [`avic`] is where that is argued, along
//! with everything else the acceleration leaves for the software to finish.
//!
//! # Nothing here fails in a way the guest can see
//!
//! A command naming a mode the architecture reserves, or a processor that does
//! not exist, is one real hardware would also do nothing useful with — so it is
//! dropped, and said once. A message no processor accepted is the one of those
//! the architecture gives a bit for, and it is recorded in the sender's error
//! status as well.
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

pub(crate) mod avic;
pub(crate) mod doorbell;
pub(crate) mod error;

mod arbitration;
mod destination;
mod startup;

use log::{trace, warn};

use crate::{
    delivery::{
        arbitration::candidates,
        destination::targets,
        doorbell::nudge,
        startup::{initialize, start},
    },
    machine::{diagnostics::Report, ownership},
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
/// hardware would also do nothing useful with — so it is dropped, and a message
/// no processor accepted is recorded as well, which is what a controller does
/// with one.
pub(crate) fn send(from: &Vlapic, lapics: &[Vlapic], command: Command) {
    let mode = from.mode();
    let Some(delivery) = command.delivery(mode) else {
        // Lowest priority is the one reserved encoding worth telling apart in the
        // log: x2APIC removed it, so a guest that asks for one in that face has
        // asked for something the older face would have delivered.
        if from.diagnostics().say(Report::ReservedCommand) {
            warn!(
                "vlapic: {} sent a command with {}: {:#x}",
                from.index(),
                if command.wants_lowest_priority() {
                    "a lowest-priority delivery, which the wide face does not have"
                } else {
                    "a reserved delivery mode"
                },
                command.bits()
            );
        }
        return;
    };
    // Judged whole, before a single processor is named. A command the
    // architecture does not define must not have reset, started or interrupted
    // half the machine by the time that is noticed.
    if !command.legal(mode) {
        if from.diagnostics().say(Report::IllegalCommand) {
            warn!(
                "vlapic: {} sent a command no processor would send: {:#x}",
                from.index(),
                command.bits()
            );
        }
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
        error::noticed(from, Errors::SEND_ILLEGAL_VECTOR);
        return;
    }

    match delivery {
        // One of the named processors takes it, and which one is the chipset's
        // choice on real hardware rather than the architecture's. The set is
        // walked in the order arbitration puts it in until a processor takes it,
        // because a processor that refuses has not consumed the interrupt — it is
        // meant for one of the set, and the next one is entitled to it.
        Delivery::LowestPriority => {
            let named = targets(from, lapics, command);
            if !candidates(named, command.vector())
                .any(|target| accept(from, target, delivery, command))
            {
                // Every processor the command named refused it, or it named none
                // that was accepting. Nothing has it, and the architecture's
                // nearest report is the one for a message nobody accepted.
                refused_by_all(from, delivery, command);
            }
        }
        Delivery::Fixed => {
            // Each target answers for itself, so what any one of them made of it
            // is nothing the others or the sender depend on: a fixed interrupt
            // names the processors it names, and one that refuses has refused.
            for target in targets(from, lapics, command) {
                accept(from, target, delivery, command);
            }
        }
        // A non-maskable interrupt is delivered to the processor rather than to
        // the register file, so it deliberately does not go through the
        // acceptance that would refuse it. That is right for a controller its
        // guest has *software*-disabled, which the architecture says still takes
        // one; it is wider than the architecture allows for one whose guest has
        // switched it off through the base register, which takes nothing at all.
        // So is INIT and so is a start-up message, and the deviation is stated
        // where the modes are: [`crate::registers::base`].
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
        Delivery::SystemManagement => {
            if from.diagnostics().say(Report::SystemManagement) {
                warn!(
                    "vlapic: {} sent a system-management interrupt, which this machine does not \
                     deliver",
                    from.index()
                );
            }
        }
    }
}

/// Gives one processor an interrupt, and makes sure it notices.
///
/// Answers whether the interrupt is now recorded somewhere that will deliver
/// it. A refusal is not a failure — a controller that is switched off or
/// software-disabled refuses interrupts, which is what a real one does — but it
/// is the difference between a fixed interrupt, where each target answers for
/// itself, and a redirectable one, where a refusal means the next processor in
/// the set has to be offered it.
///
/// Whether a controller is accepting is checked at the target rather than
/// filtered here, because it is the target's own state and may change between
/// the two.
fn accept(from: &Vlapic, target: &Vlapic, delivery: Delivery, command: Command) -> bool {
    if !ownership::owns(target) {
        refused(from, target, delivery);
        return false;
    }
    let vector = command.vector();
    match target.accept(vector, command.trigger()) {
        // It is in the target's register file now, and the target may not be
        // looking at it. The host interrupt is what tells it whether or not the
        // hardware is driving that controller: the vector is in the model, which
        // the hardware does not read, and the exit this forces is what carries it
        // into the page the hardware does.
        Accepted::Requested | Accepted::Coalesced => {
            nudge(from, target);
            true
        }
        // A vector no controller may deliver is the receiver's to report, and is
        // reported on the receiver: the sender has already been refused for
        // exactly this on every path that reaches here, so nothing but a
        // mis-decode gets this far. The record is made here rather than inside
        // the acceptance because raising the interrupt it arms needs both
        // controllers — the target's to publish into, and this one's to ring the
        // doorbell with.
        Accepted::Illegal => {
            target.diagnostics().declined();
            error::record(from, target, Errors::RECEIVE_ILLEGAL_VECTOR);
            false
        }
        // The refusal the architecture defines: a controller its guest switched
        // off or software-disabled, saying no to something it is entitled to say
        // no to.
        Accepted::Refused => {
            target.diagnostics().declined();
            trace!(
                "vlapic: {} offered {vector} to {}, which did not take it",
                from.index(),
                target.index()
            );
            false
        }
        // Not one of those: the interrupt was not recorded anywhere, so nothing
        // will deliver it and nothing will report it but this.
        Accepted::Resetting => {
            target.diagnostics().dropped();
            if from.diagnostics().say(Report::Resetting) {
                warn!(
                    "vlapic: {} offered {vector} to {}, whose register file was being reset, and \
                     it was not delivered",
                    from.index(),
                    target.index()
                );
            }
            false
        }
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
/// and said once per controller rather than once per message: a guest can send
/// these as fast as it can write a register.
fn refused(from: &Vlapic, target: &Vlapic, delivery: Delivery) {
    target.diagnostics().dropped();
    error::noticed(from, Errors::SEND_ACCEPT);
    if from.diagnostics().say(Report::Unrun) {
        warn!(
            "vlapic: {} sent {delivery:?} to {}, which this hypervisor does not run",
            from.index(),
            target.index()
        );
    }
}

/// Records that a redirectable interrupt was offered to every processor the
/// command named and taken by none of them.
///
/// Which is not nothing having happened: the guest asked for the interrupt to
/// be delivered to one of a set, and it has been delivered to none. The set can
/// be empty — a command naming processors whose controllers are all
/// software-disabled — or every one of them can refuse it in the window between
/// being found accepting and being offered it, which is the target's own
/// guest's state changing and cannot be closed from here.
///
/// Recorded as a message no processor accepted, which is what happened, and
/// said once per controller rather than once per message: a guest that offlines
/// a processor with a redirectable interrupt in flight can produce these as
/// fast as it can write a register. The loss is counted against the sender,
/// because a command nobody took had no one target to count it against.
fn refused_by_all(from: &Vlapic, delivery: Delivery, command: Command) {
    from.diagnostics().dropped();
    error::noticed(from, Errors::SEND_ACCEPT);
    if from.diagnostics().say(Report::Unaccepted) {
        warn!(
            "vlapic: {} sent {delivery:?} {} to processors none of which took it",
            from.index(),
            command.vector()
        );
    }
}
