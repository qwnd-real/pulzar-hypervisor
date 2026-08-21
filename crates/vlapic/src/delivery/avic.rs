//! Completing in software what the hardware started delivering between the
//! guest's own processors.
//!
//! Fixed edge-triggered IPIs go through the hardware without an exit; what
//! reaches here is everything else — the modes the hardware does not
//! implement, the targets it could not reach, the failures it reports. Each
//! of those is answered by the failure's own rule, and the one invariant all
//! of them keep is that exactly one of the hardware and the software delivers
//! each interrupt: the hardware reports which of them already acted before it
//! exited, and nothing here acts twice.
//!
//! # A request and the signal that announces it name one authority
//!
//! Two authorities can be holding an interrupt for a guest, and each has a
//! signal of its own. A request in the software model is announced with the
//! host interrupt [`super::doorbell`] sends, which makes the target leave the
//! guest and consult that model on its way back in. A request in a backing page
//! would be announced with the hardware doorbell, a write of the target's
//! physical identifier that makes its processor re-evaluate the page without
//! leaving the guest at all.
//!
//! Only the first of the two is ever sent from here, and that is not a
//! preference: nothing on the host side writes another processor's page. What
//! the software path accepts, it accepts into the target's model, and a backing
//! page is written by the processor it belongs to — at the entry that hands
//! that model's interrupts to the hardware. A doorbell rung for one of those
//! requests would name a page the vector is not in: the target's hardware would
//! answer it by finding nothing, no exit would be raised, and nothing would be
//! left that could raise one.
//!
//! What the *hardware* deposits in a page it announces itself, and the one case
//! it cannot — a target that was not in the guest — is the only wake these
//! paths owe: a host interrupt each, counted, because how many of them a guest
//! costs is the measure of how often the acceleration cannot finish what it
//! started.

use core::sync::atomic::{AtomicU64, Ordering};

use log::{error, trace, warn};
use svm::avic::{IncompleteIpiExit, IpiFailure};

use crate::{
    VlapicError,
    avic::activation,
    delivery::{doorbell, targets},
    machine::{current, ownership, registry},
    registers::{Vlapic, icr::Command},
};

/// Host-interrupt kicks sent by these paths, cumulative.
static KICKS: AtomicU64 = AtomicU64::new(0);

/// How many host-interrupt kicks these paths have sent.
pub(crate) fn kicks() -> u64 {
    KICKS.load(Ordering::Relaxed)
}

/// Answers an interrupt the hardware could not finish delivering between the
/// guest's own processors.
///
/// The exit is trap-like — the guest is past the write that asked for it —
/// so the answer is only to finish the delivery, in whichever way the
/// failure's rule says:
///
/// - The modes the hardware does not implement, and the targets it cannot name,
///   are sent by the software path end to end. The hardware attempted nothing
///   for these; the software owns the delivery whole.
/// - A target that was not running was already given the interrupt by the
///   hardware — the request bit is set — and needs only the wake.
/// - An invalid backing page is this hypervisor's own mistake, the one the
///   guest must not pay for: the vCPU steps back to software delivery and the
///   interrupt goes through the software path.
/// - A vector the architecture will not deliver is discarded, as it is on real
///   hardware.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub(crate) fn incomplete_ipi(exit: IncompleteIpiExit) -> Result<(), VlapicError> {
    // Refusal guard: the exit cannot legitimately arrive while the software
    // delivers. One that does is an enable bit or a table out of step with
    // the path the processor is on, and the answer is the safe one in both
    // directions: the vCPU stops trusting the acceleration, and the command
    // goes through the path that needs none of it.
    let sender = current()?;
    if !activation::active_for(sender) {
        warn!(
            "vlapic: an incomplete-IPI exit arrived while software delivery is active, command \
             {:#018x}; the vCPU stays on the software path",
            exit.icr()
        );
        sender.inhibit_avic();
        return activation::complete_command(exit.icr());
    }
    match exit.cause() {
        IpiFailure::InvalidInterruptType | IpiFailure::InvalidTarget => {
            trace!(
                "vlapic: completing in software an IPI the hardware does not deliver: {:?}, \
                 command {:#018x}",
                exit.cause(),
                exit.icr()
            );
            activation::complete_command(exit.icr())?;
        }
        IpiFailure::TargetNotRunning => {
            // The hardware set the request bits before it exited; what is
            // owed is only the wake, and re-emulating the command would
            // deliver every interrupt twice.
            wake_targets(exit.icr())?;
        }
        IpiFailure::InvalidBackingPage => {
            // A table this hypervisor itself wrote naming a page it did not
            // provide: a broken invariant, and one the guest must not be
            // stopped for. The vCPU steps back to software delivery, and the
            // interrupt goes through the path that does not need a table.
            let vlapic = current()?;
            vlapic.inhibit_avic();
            // The destination mode goes with the index because it is what says
            // which table the index is into: the hardware resolves a directed
            // interprocessor interrupt through the logical table where the
            // command names a logical destination and through the physical one
            // otherwise, and which of the two named a page that is not there is
            // the whole diagnostic value of the number.
            error!(
                "vlapic: {} reported an invalid backing page for an IPI, command {:#018x}, \
                 index {:#x}, destination mode {:?}; this vCPU returns to software delivery",
                vlapic.index(),
                exit.icr(),
                exit.index(),
                Command::from_bits(exit.icr()).destination_mode(),
            );
            activation::complete_command(exit.icr())?;
        }
        IpiFailure::InvalidIpiVector => {
            // The architecture refuses vectors below sixteen; real hardware
            // discards the command, and the only bookkeeping is the
            // delivery-status bit the guest may be watching.
            trace!(
                "vlapic: discarding an IPI of a vector the architecture does not deliver: \
                 {:#018x}",
                exit.icr()
            );
            let vlapic = current()?;
            activation::clear_command_busy(vlapic)?;
        }
        IpiFailure::UnacceleratedIpi => {
            // A failure the architecture reserves for secure delivery, which
            // this machine does not run: something below is not what it
            // claims, and the machine steps off the acceleration for good
            // rather than trusting it further.
            activation::inhibit_machine("the hardware reported a secure-delivery IPI failure");
            activation::complete_command(exit.icr())?;
        }
    }
    Ok(())
}

/// Wakes every target of the command the hardware could not announce a delivery
/// to.
///
/// The request bits are the hardware's already; a kick is a wakeup and not a
/// delivery, so a redundant one is only an exit, never a duplicate interrupt.
/// The set of targets is worked out the way the software delivery works it
/// out — against each controller's own state — which is what the exit's one
/// index cannot say for a command naming several.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub(crate) fn wake_targets(command_bits: u64) -> Result<(), VlapicError> {
    let from = current()?;
    let page = registry::lapics()?;
    let command = Command::from_bits(command_bits);
    for target in targets(from, page.all(), command) {
        if owed(
            target.index() == from.index(),
            ownership::owns(target),
            activation::is_running(target.apic_id()).unwrap_or(false),
        ) {
            kick(from, target);
        }
    }
    Ok(())
}

/// Whether a target the hardware delivered to is owed the host interrupt that
/// makes it look at what was left in its backing page.
///
/// Three facts, and each excludes the wake for a reason of its own. Written as
/// a decision over values because a wake this refuses is a request bit standing
/// in a page with nothing left to make its processor read it, and because two
/// of the three are answers about a machine that can change underneath the
/// walk.
///
/// - The sender is not one of them, whatever the command named. It is outside
///   the guest — it is executing this — and consults its own controller on the
///   way back in, so an interrupt sent here would be one this processor answers
///   with an empty handler and nothing else. Both the all-inclusive shorthand
///   and a broadcast destination name it, which is how a great deal of firmware
///   and some kernels send.
/// - A processor this hypervisor does not run has no guest to be woken into.
///   The software path reports such a message as one no processor accepted, and
///   there is nothing here to add to that.
/// - A target that is in the guest now entered it after the hardware set the
///   request bit, and an entry re-evaluates the page it is entered with — so
///   the look this would ask for has already happened. Its own away flag is
///   deliberately not consulted, unlike [`doorbell::nudge`]'s: the hardware has
///   just reported the target as not running, and reading the flag would only
///   race that answer.
const fn owed(sender: bool, owned: bool, running: bool) -> bool {
    !sender && owned && !running
}

/// Wakes a target with the host doorbell interrupt, whether or not it said it
/// was away.
///
/// What this adds to [`doorbell::nudge`] is the count; what it leaves out is
/// the away flag, and [`owed`] is where that and every other term of the
/// decision are argued. Nothing is re-examined here.
fn kick(from: &Vlapic, target: &Vlapic) {
    doorbell::interrupt(from, target);
    KICKS.fetch_add(1, Ordering::Relaxed);
}

/// Gives one processor an interrupt that arrived for its guest through real
/// hardware, while its controller is driven by the acceleration.
///
/// The request is left in the backing page, where the hardware reads it, and
/// the target is told about it by whichever signal its state calls for: the
/// hardware doorbell for one in the guest, the host interrupt for one that is
/// not. What real hardware is owed is decided exactly as the software path
/// decides it — a level arrival below the host's own class keeps its
/// acknowledgement until the guest's, and everything else is acknowledged at
/// once.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub(crate) fn arrive(
    vlapic: &Vlapic,
    local: apic::LocalApic,
    vector: descriptors::Vector,
    level: bool,
) -> Result<(), VlapicError> {
    if !vlapic.accepting() {
        vlapic.diagnostics().declined();
        trace!(
            "vlapic: {} received {vector} but is not accepting it; the hardware will not be \
             told it arrived",
            vlapic.index()
        );
        return Ok(());
    }
    if crate::lifecycle::arrival::withholdable(vector, level) {
        // The debt is recorded before the request is published, so that a
        // guest which acknowledges immediately finds the debt already there.
        vlapic.ledger().owe(vector);
    } else {
        local.end_of_interrupt();
    }
    let requested = activation::request(vector)?;
    if !requested {
        trace!(
            "vlapic: {vector} coalesced into {}'s backing page",
            vlapic.index()
        );
    }
    // Nothing else is owed. The target is this processor — an arrival runs
    // on the processor the interrupt was addressed to — and this arrival is
    // itself the wake: it interrupted whichever state the processor was in,
    // and every path from here re-enters the guest through the loop that
    // evaluates the backing state on its way in.
    Ok(())
}

/// Warns once per machine about an arrival shape the acceleration cannot
/// represent faithfully, for the defensive corner that should never be
/// reached.
pub(crate) fn warn_external_once() {
    if WARNED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
    {
        warn!(
            "vlapic: an arrival the real controller never accepted reached the accelerated \
             path; it is delivered through the backing page, and its acknowledgement is the \
             guest's own legacy controller's"
        );
    }
}

/// Whether the external-arrival warning has been said.
static WARNED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

#[cfg(test)]
mod tests {
    //! Which of a command's targets a delivery the hardware could not finish
    //! owes a wake to, which is the whole of what these paths decide and the
    //! one part of them that needs no machine. What performing it does — a
    //! host interrupt, retried the once a retry can cure — belongs to the
    //! doorbell module and is argued there.

    use super::owed;

    #[test]
    fn a_target_that_is_not_in_the_guest_is_owed_the_wake() {
        // The case the exit exists for: the hardware set the request bit in a
        // page whose processor is not looking at it, and nothing but this makes
        // it look.
        assert!(owed(false, true, false));
    }

    #[test]
    fn the_sender_is_never_woken_by_its_own_command() {
        // Every state the rest of the machine can be in, against the one fact
        // that decides it: a broadcast and the all-inclusive shorthand both name
        // the sender, and the sender is out of the guest and about to consult its
        // own controller. Its running bit reads clear at this point — the exit
        // withdrew it — so nothing but this excludes it.
        for owned in [false, true] {
            for running in [false, true] {
                assert!(
                    !owed(true, owned, running),
                    "owned {owned}, running {running}"
                );
            }
        }
    }

    #[test]
    fn nothing_is_owed_a_target_the_hardware_told_or_a_processor_nothing_runs() {
        // In the guest: its own entry re-evaluated the page after the request bit
        // was set. Not this hypervisor's: there is no guest on it to wake, and
        // the software path has already reported the message as accepted by
        // nobody.
        assert!(!owed(false, true, true));
        assert!(!owed(false, false, false));
        assert!(!owed(false, false, true));
    }
}
