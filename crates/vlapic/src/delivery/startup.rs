//! Startup messages: resetting a processor, releasing it, and the one window in
//! which either belongs on real hardware.
//!
//! A guest may reset and start processors before this hypervisor has taken them
//! over. In that window the target really is executing firmware's own code on
//! real hardware, so INIT and start-up are forwarded to the real controller and
//! the processor starts exactly as it would have bare metal.
//!
//! Once a processor is running the hypervisor's own code, forwarding would
//! reset the *host*. From then on both are emulated, and neither is applied by
//! the processor that sent it: an INIT and a start-up page are left where the
//! target will find them, and the target applies them to itself at an exit
//! boundary — which is what makes resetting a controller's whole register file
//! safe without a lock.
//!
//! What closes the window is not the target's own claim, because a claim the
//! target makes cannot say anything about a processor that never makes one.
//! [`permits_forwarding`] is where the machine-wide fence and the per-target
//! claim meet, and it is the whole of the isolation this file provides.

use apic::{Command as HardwareCommand, Delivery as HardwareDelivery, Target};
use log::{info, warn};

use crate::{
    delivery::doorbell::nudge,
    machine::ownership,
    registers::{StartupPage, Vlapic},
};

/// Resets a processor and leaves it waiting to be started.
///
/// Including the processor that sent it, which is not a special case and must
/// not be one: a guest resetting its machine does it with a single broadcast
/// INIT, and refusing the sender's own copy left that guest carrying on in the
/// old image with every other processor held. Nothing here resets anything —
/// the INIT is recorded and the target applies it to itself at its next exit
/// boundary — so applying one to this processor is the same deferral as
/// applying one to any other, and the exit loop refuses to re-enter a guest
/// that has been reset.
pub(super) fn initialize(from: &Vlapic, target: &Vlapic) {
    if forwardable(target) {
        forward(from, target, HardwareDelivery::Init);
        return;
    }
    info!(
        "vlapic: {} sent an init to {}",
        from.index(),
        target.index()
    );
    target.startup().requested_init();
    nudge(from, target);
}

/// Releases a processor from waiting, at the address the vector gives the page
/// number of.
///
/// The sender included, and nothing is lost by it. What keeps a processor from
/// releasing itself is not a comparison here but [`Command::legal`]: a start-up
/// addressed to the sender alone, or to all including the sender, is refused as
/// a whole command, because a processor waiting for a start-up is not executing
/// the instruction that sends one. What reaches this with the sender among its
/// targets is a start-up whose *destination* names it — and if it has reset
/// itself with an INIT, releasing it is what the message asks for and what
/// hardware would do.
///
/// [`Command::legal`]: crate::registers::icr::Command::legal
pub(super) fn start(from: &Vlapic, target: &Vlapic, page: StartupPage) {
    if forwardable(target) {
        forward(from, target, HardwareDelivery::Startup(page.number()));
        return;
    }
    // Refused unless the target has been reset, which is what makes the second
    // of the pair a guest sends harmless: the first starts the processor, and
    // the second finds it already running.
    let taken = target.startup().offered(page);
    info!(
        "vlapic: {} sent a startup at page {:#x} to {}, {}",
        from.index(),
        page.number(),
        target.index(),
        if taken {
            "which was waiting for one"
        } else {
            "which was not waiting for one"
        }
    );
    if taken {
        nudge(from, target);
    }
}

/// Whether a startup message aimed at this processor belongs on real hardware.
///
/// True only inside the one window in which forwarding is what a guest is
/// entitled to: the processor is still running firmware's own code, and it is
/// still going to be taken over. Both halves are needed and neither is
/// sufficient.
///
/// The target's own claim says the first. It cannot say the second, because it
/// is made by the target itself — so "not claimed yet" and "never going to be
/// claimed" are the same answer, and taking the second for the first is what
/// would put a real start-up on the wire for a processor parked in firmware's
/// wait-for-start-up state. That processor would begin executing in real mode
/// at a page the guest chose, with no nested tables, no intercepts and no
/// address space tag: the guest, on bare metal.
///
/// So [`ownership::brought_up`] closes the window for the machine once the host
/// has finished starting processors, and a processor firmware described as
/// unstartable never has one at all — the host was never going to start it, so
/// there is no moment at which a real start-up aimed at it is anything but a
/// processor the guest has taken.
fn forwardable(target: &Vlapic) -> bool {
    permits_forwarding(
        ownership::owns(target),
        target.startable(),
        ownership::brought_up(),
    )
}

/// The rule [`forwardable`] applies, over the three facts it is made of.
///
/// Separated so that it can be checked on its own: none of the three states can
/// be reached in a test on a host, and the rule is the whole of the isolation
/// this file provides.
const fn permits_forwarding(owned: bool, startable: bool, brought_up: bool) -> bool {
    !owned && startable && !brought_up
}

/// Sends a startup message to real hardware, for a processor this hypervisor
/// has not taken over yet.
fn forward(from: &Vlapic, target: &Vlapic, delivery: HardwareDelivery) {
    let outcome = apic::local().and_then(|local| {
        local.send(HardwareCommand::new(
            delivery,
            Target::One(target.apic_id()),
        ))
    });
    match outcome {
        Ok(()) => info!(
            "vlapic: {} forwarded {delivery:?} to {} on real hardware",
            from.index(),
            target.apic_id()
        ),
        Err(error) => warn!(
            "vlapic: {} could not forward {delivery:?} to {}: {error}",
            from.index(),
            target.apic_id()
        ),
    }
}

#[cfg(test)]
mod tests {
    //! The one thing in this file that can be checked without a machine, and
    //! the one that decides whether a guest can reach real silicon: none of the
    //! three states it is made of exists on a host, so the rule is exercised
    //! over the facts rather than through a controller.

    use super::permits_forwarding;

    #[test]
    fn only_an_unclaimed_startable_processor_during_bring_up_is_forwarded_to() {
        for owned in [false, true] {
            for startable in [false, true] {
                for brought_up in [false, true] {
                    assert_eq!(
                        permits_forwarding(owned, startable, brought_up),
                        !owned && startable && !brought_up,
                        "owned {owned}, startable {startable}, brought up {brought_up}"
                    );
                }
            }
        }
    }

    #[test]
    fn nothing_is_forwarded_once_the_host_has_finished_starting_processors() {
        // The case the per-processor claim cannot answer, and the escape it
        // leaves open: a processor firmware said may be started, that the host
        // tried to start and failed, is unclaimed and startable for the rest of
        // the machine's life.
        assert!(permits_forwarding(false, true, false));
        assert!(!permits_forwarding(false, true, true));
    }

    #[test]
    fn a_processor_firmware_will_not_have_started_is_never_forwarded_to() {
        // It has no bring-up window at all: the host was never going to start
        // it, so a real start-up aimed at it can only ever be the guest's.
        assert!(!permits_forwarding(false, false, false));
    }
}
