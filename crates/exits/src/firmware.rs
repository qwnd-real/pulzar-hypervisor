//! The first guest's progress out of firmware, and what the host owes it at
//! each step.
//!
//! The boot manager is started through the portal, the loader is handled on the
//! first `ExitBootServices` hook, EBS eventually succeeds, and the guest leaves
//! the portal for good. Each transition arrives here as a notification, and
//! each is the moment something the host was waiting to do becomes possible.

use log::{error, info, warn};
use partition::Partition;
use portal::{Notification, Portal};
use vcpu::{Flow, Vcpu};
use x86_64::PhysAddr;

use crate::advance;

/// Bytes in `VMMCALL`, for a processor that does not report the address after
/// an intercepted instruction.
///
/// Both users of the instruction step over it with this: the portal's
/// notifications here, and the hypercalls [`crate::hypercall`] answers.
pub(crate) const VMMCALL_BYTES: u64 = 3;

/// What the host does once firmware's services are gone.
///
/// The other processors are started here rather than during bring-up because
/// starting one uses the temporary low page only after firmware has returned
/// successfully from `ExitBootServices`.
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
    /// The first EBS hook has handled the loader, either by wiping it after a
    /// successful unload or by recording that firmware rejected the unload.
    LoaderHandled,
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
///
/// One of these exists on one processor: the one the guest is entered on, which
/// is the only one that ever sees the portal. Every other processor joins a
/// guest that is already past firmware — it exists *because* `ExitBootServices`
/// succeeded — and has nothing here to keep track of.
///
/// That is why this is not shared state despite describing the guest rather
/// than a processor. Both things a stage decides are questions only the boot
/// processor can be asked: whether the instruction pointer *it* stopped at is
/// still inside the portal, and whether *it* has run the wrapper yet.
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

    /// Acts on one of the portal's notifications.
    pub(crate) fn notified(&mut self, vcpu: &mut Vcpu) -> Flow {
        let marker = vcpu.registers().rdx;
        match Notification::from_bits(marker) {
            Some(Notification::LoaderUnloaded) => self.loader(vcpu, true),
            Some(Notification::LoaderSkipped) => self.loader(vcpu, false),
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

    /// Records the first EBS hook and removes the loader's contents after
    /// firmware has released its image pages.
    fn loader(&mut self, vcpu: &mut Vcpu, unloaded: bool) -> Flow {
        if self.stage != Stage::Portal {
            error!("exits: duplicate or out-of-order loader notification");
            return Flow::Leave;
        }
        let status = vcpu.save().rax;
        let wiped = if unloaded {
            match self.portal.wipe_loader() {
                Ok(()) => {
                    info!("exits: hv-loader image unloaded and wiped");
                    true
                }
                Err(error) => {
                    warn!("exits: hv-loader was unloaded but could not be wiped: {error}");
                    false
                }
            }
        } else {
            warn!("exits: firmware rejected hv-loader UnloadImage with status {status:#x}");
            false
        };
        let transition = if wiped {
            self.portal.mark_loader_handled()
        } else {
            self.portal.mark_loader_skipped()
        };
        if let Err(error) = transition {
            error!("exits: could not publish the loader state: {error}");
            return Flow::Leave;
        }
        self.stage = Stage::LoaderHandled;
        advance(vcpu, VMMCALL_BYTES);
        Flow::Resume
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
        match partition
            .conceal(self.portal.entry(), self.portal.bytes())
            .and_then(|change| partition.barrier(change))
        {
            Ok(()) => {
                // The tables now permit less than they did, and every processor
                // that has run this guest may have cached what they used to say
                // — so the barrier above is what makes each of them discard it,
                // this one at its own next entry.
                info!("exits: the portal is behind the guest and reads as zeroes again");
            }
            // Not fatal to the guest: the pages stay visible and read-only, so
            // the guest carries on correctly and the hypervisor keeps one page
            // of its own on show. Worth saying so loudly rather than stopping a
            // boot that would otherwise finish.
            Err(error) => error!("exits: the portal could not be taken back: {error}"),
        }
    }

    /// Restores firmware's table, takes the legacy interrupt controllers back,
    /// and starts the other processors now that firmware no longer owns the
    /// machine.
    ///
    /// The controllers go first, and before anything else here, because they
    /// are the one piece of the machine that was handed *to* the guest
    /// rather than kept from it. Firmware needed them: its own periodic
    /// timer arrives through them, on vectors it chose and only it had
    /// handlers for, and it programmed them long before this hypervisor
    /// existed. Past this point nothing is coming back for that
    /// configuration — every operating system reinitializes
    /// both controllers from scratch before it uses either — while an input
    /// left open is an interrupt arriving on a number no guest has claimed
    /// and nothing here can name, because the vector base is write-only and
    /// cannot be read back.
    fn handed(&mut self, vcpu: &mut Vcpu) -> Flow {
        if self.stage == Stage::Portal {
            error!("exits: ExitBootServices succeeded before the loader transition");
            return Flow::Leave;
        }
        if self.stage == Stage::LoaderHandled {
            if let Err(error) = self.portal.restore_boot_services() {
                error!("exits: the original BootServices table could not be restored: {error}");
                return Flow::Leave;
            }
            info!("exits: restored the original BootServices table");
            match apic::mask_legacy() {
                Ok(true) => {
                    info!("exits: legacy controllers masked again, firmware is done with them");
                }
                Ok(false) => {}
                // Not fatal to the guest, and not a reason to stop a boot that
                // will otherwise finish: what is left is firmware's own masks,
                // which is the state the machine was in a moment ago.
                Err(error) => warn!("exits: the legacy controllers could not be masked: {error}"),
            }
            info!("exits: About to boot APICs");
            let started = apic::start(self.boot.trampoline, self.boot.attach);
            // Whatever came of it, and before the guest is let go: every
            // processor that was going to become this hypervisor's has done so,
            // so from here on a startup message the guest sends can only be one
            // aimed at a processor the guest itself owns. One that failed to
            // start is the sharpest case — it sits in firmware's wait-for-startup
            // state, where a forwarded message would give the guest a processor
            // with no nested tables at all.
            vlapic::bring_up_finished();
            match started {
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
        } else {
            error!("exits: duplicate or out-of-order successful ExitBootServices notification");
            return Flow::Leave;
        }
        advance(vcpu, VMMCALL_BYTES);
        Flow::Resume
    }
}
