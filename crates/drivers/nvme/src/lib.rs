//! The storage controller a guest reads its disks' identities from, answered
//! for so that what it reads names nothing.
//!
//! An `NVMe` controller keeps two kinds of thing a guest can ask it for. Data,
//! moved by thousands of commands a second through queues the guest builds
//! itself, and identity — the serial number of the drive, the identifiers of
//! each volume on it — asked for a handful of times a boot. This driver lets
//! the data through untouched and answers for the identity: each identifying
//! field of an identify response is replaced with what `spoof` makes of it
//! under the machine's seed, so the same drive presents the same name on
//! every boot, in the same shape, with no link left to the hardware's real
//! one.
//!
//! # It starts the moment the devices are this hypervisor's
//!
//! A controller cannot be taken over while anything else may be driving it,
//! and pulzar keeps the firmware environment it booted from running until
//! the guest's own `ExitBootServices`. That interception is the earliest
//! moment a base address register may be asked how far it decodes — which is
//! what [`adopt`] needs it for — and [`adopt`] is meant to be called from
//! it, on the one processor that exists at that point, before the others
//! are started: the barriers registering regions owe cost nothing while
//! there is nobody to send them to.
//!
//! # What each access costs
//!
//! Two regions per controller, both trapping writes only. Reads of the
//! register file and the doorbell array never fault, because the hardware
//! answers them honestly and this driver has nothing to add. Writes to the
//! register file are rare — a guest writes them while configuring the
//! controller, at boot — and cost one exit each. Doorbell writes cost one
//! exit each, always, and there is no way around it: the admin doorbells
//! share their page with every I/O queue's doorbells, a page is the finest
//! the tables can trap, and watching what a guest submits is the whole of
//! what this driver does. So an I/O doorbell write is forwarded by one
//! comparison and one volatile write, with no lock held and nothing
//! allocated, and that is the entire running cost the storage path pays.
//!
//! # Why the driver waits for the answer itself
//!
//! A spoof has to be in place before the guest reads the response, and a
//! guest taking its answers by interrupt reads its identify buffer in the
//! interrupt, before it writes any doorbell again. So the moment to spoof is
//! not the next doorbell write but the one that submitted the command: the
//! handler rings the submission doorbell itself and then waits, inside the
//! exit, for the hardware to post the completion, which it does in
//! microseconds. Identify commands are asked for a handful of times a boot,
//! which is what makes waiting affordable; nothing else this driver does
//! waits for anything.
//!
//! # Failing open
//!
//! Every way this driver can lose its place — a queue it cannot read, a
//! command that never answers, a controller whose registers make no sense —
//! ends the same way: a loud warning, and the response passing through as
//! the hardware wrote it. A guest that boots reading one real serial number
//! is a failure of the spoofing; a guest that does not boot is a failure of
//! the hypervisor, which is the worse one.
//!
//! # What this driver deliberately does not do
//!
//! A guest that moves a controller's first base address register after
//! takeover — which no operating system does to a boot controller, and which
//! the specification gives it no reason to — leaves the traps and the
//! doorbell mapping aimed at the old address. Watching for that means
//! watching configuration space, which is a feature of its own.
//!
//! The spoof is not atomic with the completion's visibility. The processor
//! that submitted an identify is stopped for the whole of the wait, but a
//! guest may take the controller's completion interrupt on another
//! processor, and that processor has a window — the microseconds between the
//! hardware posting the completion and this driver's spin reading it — in
//! which the buffer holds what the hardware wrote. Closing it means holding
//! the completion's interrupt back until the spoof is in place, which is
//! interrupt-delivery machinery for a window no driver in practice reaches:
//! the submitting processor is the one the interrupt wakes, and it cannot be
//! woken while it is stopped in here.
//!
//! # Allocation
//!
//! Controllers are allocated when they are taken over, which happens once,
//! on the boot processor, inside the firmware handoff. Nothing on the exit
//! path allocates: the pending identifies are a fixed slab, the queues are
//! plain fields, and the entries are read into stack buffers.
//!
//! # More than one processor
//!
//! One device per controller, reached from every processor the guest runs
//! on. The admin-queue shadow is behind a lock, taken and released per queue
//! read and per poll of a waited-for completion — never held across a spin,
//! so no processor waits on another's answer. The I/O doorbell fast path
//! takes no lock at all.

#![no_std]

extern crate alloc;

mod command;
mod controller;
mod identify;
mod regs;

use alloc::vec::Vec;

use log::{info, warn};
use npt::Change;
use paging::PagingError;
use partition::{Partition, PartitionError};
use pci::PciError;
use spin::Once;
use thiserror::Error;

use crate::controller::Controller;

/// The controllers this driver took over, so that a second takeover can be
/// refused rather than leave two drivers answering for one register file.
static ADOPTED: Once<Vec<&'static Controller>> = Once::new();

/// Interposes every `NVMe` controller the machine has.
///
/// Called once, on the boot processor, after the guest's `ExitBootServices`
/// has been intercepted and before the guest's other processors are started.
/// A controller that cannot be taken over — one whose first base address
/// register decodes nothing, or too little, or off a page boundary — is left
/// alone with a warning, because a guest that boots reading one real serial
/// number is the better failure.
///
/// # Errors
///
/// [`NvmeError::AlreadyAdopted`] for a second call, [`NvmeError::Pci`] if
/// the machine was never surveyed, [`NvmeError::Paging`] if the address
/// space is not the machine's yet, or [`NvmeError::Partition`] if the
/// barrier owed for the trapped regions could not be paid.
pub fn adopt(partition: &'static Partition) -> Result<(), NvmeError> {
    if ADOPTED.is_completed() {
        return Err(NvmeError::AlreadyAdopted);
    }
    let topology = pci::topology()?;
    // Taking a controller over asks its configuration space, maps its
    // doorbells, and registers its regions — all of it under the address
    // space's lock, which is the one exception the lock's rules allow, and
    // only because nothing else is running: the guest is stopped mid-call
    // and the other processors have not started.
    let (controllers, owed) = paging::with(|space| {
        let mut controllers = Vec::new();
        let mut owed = Change::None;
        for function in topology
            .functions()
            .iter()
            .filter(|function| function.class().is_nvme())
        {
            let controller = Controller::take(partition, space, function);
            let controller = match controller {
                Ok(controller) => controller,
                Err(error) => {
                    warn!("nvme: {} was left alone: {error}", function.address());
                    continue;
                }
            };
            // One registration per controller, so that a controller which
            // cannot be registered costs itself and nothing else — and so
            // that the changes the successful ones owe stay in hand to be
            // paid even then. What a registration that failed partway
            // tightened is lost inside it, which is a seam in the
            // registration interface rather than a decision made here.
            match partition.interpose(space, Controller::regions(controller)) {
                Ok(change) => {
                    owed = owed.and(change);
                    controller.describe();
                    controllers.push(controller);
                }
                Err(error) => {
                    warn!(
                        "nvme: {} was left alone: its regions could not be trapped: {error}",
                        function.address()
                    );
                }
            }
        }
        Ok::<_, NvmeError>((controllers, owed))
    })??;
    partition.barrier(owed)?;
    if controllers.is_empty() {
        info!("nvme: no controller was taken over");
    }
    ADOPTED.call_once(|| controllers);
    Ok(())
}

/// Why the machine's `NVMe` controllers are not being answered for.
#[derive(Debug, Error)]
pub enum NvmeError {
    /// They were taken over already, and a second takeover would leave two
    /// drivers answering for one register file.
    #[error("the machine's nvme controllers were already taken over")]
    AlreadyAdopted,
    /// The machine was never surveyed, so there is nothing to find
    /// controllers in.
    #[error(transparent)]
    Pci(#[from] PciError),
    /// The address space is not the machine's, so there is nowhere to map a
    /// doorbell array.
    #[error(transparent)]
    Paging(#[from] PagingError),
    /// The regions could not be trapped, or the barrier owed for trapping
    /// them not paid.
    #[error(transparent)]
    Partition(#[from] PartitionError),
    /// The controller's first base address register decodes nothing, or is
    /// not implemented at all.
    #[error("bar0 decodes nothing")]
    Unaddressed,
    /// The register decodes less than a register page and a doorbell page,
    /// which leaves nothing to watch a submission with.
    #[error("bar0 is {bytes:#x} bytes, too small for a register page and a doorbell page")]
    TooSmall {
        /// How many bytes the register decodes.
        bytes: u64,
    },
    /// The register decodes from an address the tables cannot trap a page
    /// at.
    #[error("bar0 starts at {base:#x}, which is not a page boundary")]
    Misaligned {
        /// Where the register decodes from.
        base: u64,
    },
}
