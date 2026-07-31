//! The first guest's progress out of firmware, and what the host owes it at
//! each step.
//!
//! Three things happen once and in one order: the boot manager is started
//! through the portal, `ExitBootServices` succeeds, and the guest leaves the
//! portal for good. The first is the loader's and the hypervisor's bring-up;
//! the other two arrive here, and each of them is the moment something the host
//! was waiting to do becomes possible.

use log::{error, info};
use partition::Partition;
use portal::{Notification, Portal};
use vcpu::{Flow, Vcpu};
use x86_64::PhysAddr;

use crate::advance;

/// Bytes in `VMMCALL`, for a processor that does not report the address after
/// an intercepted instruction.
const VMMCALL_BYTES: u64 = 3;

/// What the host does once firmware's services are gone.
///
/// The other processors are started here rather than during bring-up because
/// starting one takes memory below a megabyte away from firmware, and firmware
/// still owns every byte of the machine until `ExitBootServices` returns.
#[derive(Clone, Copy, Debug)]
pub struct Boot {
    /// Where the trampoline the other processors start on has been placed.
    pub trampoline: PhysAddr,
    /// What each of them runs once it is up. It never returns: a processor that
    /// finished this would have nowhere to go back to.
    pub attach: fn() -> !,
}

/// How far the guest has got out of firmware.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    /// Inside the portal, with firmware's boot services live and the
    /// `ExitBootServices` wrapper installed in them.
    Portal,
    /// The wrapper reported success, so firmware's services are gone and the
    /// other processors have been started — but the guest is still returning
    /// through the portal, which therefore has to stay where it is.
    Handed,
    /// The guest has left the portal and the portal has been taken back. No
    /// hypervisor memory is visible to the guest any more.
    Gone,
}

/// The guest's way out of firmware: the pages it goes through, and how far
/// along them it is.
#[derive(Debug)]
pub(crate) struct Firmware {
    portal: Portal,
    boot: Boot,
    stage: Stage,
}

impl Firmware {
    /// A guest that has not been entered yet.
    pub(crate) const fn new(portal: Portal, boot: Boot) -> Self {
        Self {
            portal,
            boot,
            stage: Stage::Portal,
        }
    }

    /// Acts on one of the portal's two notifications.
    pub(crate) fn notified(&mut self, vcpu: &mut Vcpu) -> Flow {
        let marker = vcpu.registers().rdx;
        match Notification::from_bits(marker) {
            Some(Notification::ExitSucceeded) => self.handed(vcpu),
            Some(Notification::StartReturned) => {
                error!(
                    "exits: firmware StartImage returned status {:#x}",
                    vcpu.save().rax
                );
                Flow::Leave
            }
            // Nothing else in the guest has any business issuing this: the
            // instruction is intercepted for the portal's sake alone, and a
            // marker the portal never writes means something else made the
            // call.
            None => {
                error!("exits: the guest issued VMMCALL with unknown marker {marker:#x}");
                Flow::Leave
            }
        }
    }

    /// Takes the portal back, once the guest has finished with it.
    ///
    /// Two conditions, and both are needed. `ExitBootServices` must have
    /// returned, because until it has, the wrapper can still be called and the
    /// pages have to be callable. And the guest must be executing somewhere
    /// other than the portal, because the wrapper returns through the portal
    /// after it notifies the host — concealing the pages at the notification
    /// itself would take the ground out from under the instruction after it.
    ///
    /// After both hold, nothing can reach the portal again: firmware's
    /// boot-services table is gone along with the pointer the wrapper was
    /// installed in, and nothing else in the guest knows the address.
    pub(crate) fn retire(&mut self, vcpu: &mut Vcpu, partition: &Partition) {
        if self.stage != Stage::Handed || self.portal.holds(vcpu.save().rip) {
            return;
        }
        // Moved whatever comes of the attempt. A portal that could not be taken
        // back now cannot be taken back at the next exit either, and retrying
        // would report the same failure for the rest of the guest's life.
        self.stage = Stage::Gone;
        match partition.conceal(self.portal.entry(), self.portal.bytes()) {
            Ok(()) => {
                // The tables now permit less than they did, and this processor
                // has been running on them — so what it cached from them has to
                // go before it enters the guest again.
                vcpu.flush();
                info!("exits: the portal is behind the guest and reads as zeroes again");
            }
            // Not fatal to the guest: the pages stay visible and read-only, so
            // the guest carries on correctly and the hypervisor keeps one page
            // of its own on show. Worth saying so loudly rather than stopping a
            // boot that would otherwise finish.
            Err(error) => error!("exits: the portal could not be taken back: {error}"),
        }
    }

    /// Starts the other processors, now that firmware no longer owns the
    /// machine.
    fn handed(&mut self, vcpu: &mut Vcpu) -> Flow {
        if self.stage == Stage::Portal {
            match apic::start(self.boot.trampoline, self.boot.attach) {
                Ok(started) => {
                    self.stage = Stage::Handed;
                    info!(
                        "exits: ExitBootServices succeeded; {} of {} processors online",
                        started.online, started.startable
                    );
                }
                Err(error) => {
                    error!("exits: application processors could not start: {error}");
                    return Flow::Leave;
                }
            }
        }
        advance(vcpu, VMMCALL_BYTES);
        Flow::Resume
    }
}
