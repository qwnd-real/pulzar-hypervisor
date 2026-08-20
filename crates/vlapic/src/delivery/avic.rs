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
//! The running half of wakeup is the hardware doorbell, a write of the
//! target's physical identifier to a register of this processor; the
//! not-running half is the host interrupt [`super::doorbell`] has always
//! been. The two are counted apart, because the proportion of one to the
//! other is the measure of whether the acceleration is doing its job.

use core::sync::atomic::{AtomicU64, Ordering};

use log::{error, trace, warn};
use svm::avic::{AVIC_DOORBELL, Doorbell, IncompleteIpiExit, IpiFailure};
use x86_64::registers::model_specific::Msr;

use crate::{
    VlapicError,
    avic::activation,
    delivery::{doorbell, targets},
    machine::{current, ownership, registry},
    registers::{Vlapic, icr::Command},
};

/// Hardware doorbells rung by these paths, cumulative.
static RINGS: AtomicU64 = AtomicU64::new(0);

/// Host-interrupt kicks sent by these paths, cumulative.
static KICKS: AtomicU64 = AtomicU64::new(0);

/// How many hardware doorbells the machine has rung through these paths.
pub(crate) fn rings() -> u64 {
    RINGS.load(Ordering::Relaxed)
}

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
            error!(
                "vlapic: {} reported an invalid backing page for an IPI, command {:#018x}, \
                 index {:#x}; this vCPU returns to software delivery",
                vlapic.index(),
                exit.icr(),
                exit.index()
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

/// Wakes every target of the command that is not in the guest.
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
        if !ownership::owns(target) {
            continue;
        }
        // A target the hardware reports as running was woken by the doorbell
        // it never needed; everything else is owed the host interrupt.
        if activation::is_running(target.apic_id()).unwrap_or(false) {
            continue;
        }
        kick(from, target);
    }
    Ok(())
}

/// Rings the hardware doorbell of the processor a target runs on.
///
/// The write tells this processor to deliver an interrupt to the physical
/// processor named in it — which is the target's, because a virtual
/// processor here runs on the physical processor that owns it. The request
/// the ring announces was stored before it, with the ordering the protocol
/// states; and a ring is never read back, because the register faults a read.
///
/// Never rung at the processor doing the ringing: it is outside the guest
/// already, and whatever it left for itself its own next entry evaluates.
pub(crate) fn ring(target: &Vlapic) {
    let Ok(self_vlapic) = current() else {
        return;
    };
    if target.index() == self_vlapic.index() {
        return;
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the policy accepts no machine whose identifiers do not fit the doorbell's field"
    )]
    let doorbell = Doorbell::new().with_host_apic_id(target.apic_id().get() as u16);
    // SAFETY: the register exists on every processor the policy accepted the
    // acceleration for — its feature bit was checked at boot — and writing it
    // does nothing but interrupt the named physical processor, which is the
    // machine's own and owes the target a look. The write is serializing in
    // the sense the protocol needs: the request it announces was published
    // before it, in program order, by the caller.
    unsafe { Msr::new(AVIC_DOORBELL).write(doorbell.into_bits()) };
    RINGS.fetch_add(1, Ordering::Relaxed);
}

/// Makes a target notice an interrupt a software delivery left for it,
/// under whichever transport its state calls for.
///
/// While the acceleration drives the target's controller and the target is
/// in the guest, that is the hardware doorbell: one write, and the
/// processor evaluates its backing state. Otherwise it is the host
/// interrupt [`super::doorbell`] has always been — for a target not in the
/// guest is one the doorbell would ring to no purpose.
pub(crate) fn wake(from: &Vlapic, target: &Vlapic) {
    if activation::active_for(target) && activation::is_running(target.apic_id()).unwrap_or(false) {
        ring(target);
    } else {
        doorbell::nudge(from, target);
    }
}

/// Wakes a target with the host doorbell interrupt, whether or not it said
/// it was away.
///
/// What the kick paths add to [`doorbell::nudge`]: the nudge is for a target
/// whose state the sender consulted, and a kick is for one the hardware has
/// already said is not running — where consulting the flag again would only
/// race the answer the exit just gave.
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
