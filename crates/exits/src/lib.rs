//! What a guest's exits mean.
//!
//! [`vcpu`] runs a guest and hands back every exit without interpreting one,
//! deliberately: what an exit means is policy. This is that policy — the loop
//! that answers each exit and decides whether the guest carries on.
//!
//! # What is answered, and how
//!
//! - `CPUID`, answered with the machine's own answer less the virtualization
//!   extension, so the guest does not discover the hypervisor underneath it.
//! - The one model-specific register that says whether the extension is
//!   available, answered consistently with that.
//! - Nested page faults, which are how a guest's memory comes to be described
//!   at all, and how a write to hypervisor memory is stepped over.
//! - The two notifications the [`portal`] makes, which are the only two things
//!   the guest ever deliberately tells the host.
//!
//! Anything else stops the guest. A hypervisor that resumed an exit it did not
//! understand would be resuming a guest whose state it did not fix, and the
//! same exit would arrive again forever.
//!
//! # Firmware's guest passes through three stages
//!
//! The first guest is firmware, entered at the portal, and the host owes it
//! something different at each stage of its life. It starts inside the portal
//! with boot services live and the wrapper installed; it is handed the machine
//! when the wrapper reports `ExitBootServices` succeeded, which is where the
//! other processors are started; and the portal is gone from it at the first
//! exit taken outside those pages, which is where they are taken back and the
//! guest stops being able to see any hypervisor memory at all.

#![no_std]

mod cpuid;
mod firmware;
mod msr;
mod nested;

use core::convert::Infallible;

use log::{error, info};
use partition::Partition;
use portal::Portal;
use svm::Reason;
use thiserror::Error;
use vcpu::{Flow, Vcpu, VcpuError};

pub use crate::firmware::Boot;
use crate::firmware::Firmware;

/// The guest's exits, and everything the host needs to answer one.
///
/// One of these per guest rather than per processor: what it holds is the
/// guest's memory and the guest's progress through firmware, and both are the
/// same for every processor that runs it.
#[derive(Debug)]
pub struct Exits<'a> {
    partition: &'a Partition,
    firmware: Firmware,
}

impl<'a> Exits<'a> {
    /// What will answer for a guest that is entered at `portal`.
    #[must_use]
    pub const fn new(partition: &'a Partition, portal: Portal, boot: Boot) -> Self {
        Self {
            partition,
            firmware: Firmware::new(portal, boot),
        }
    }

    /// Runs the guest until an exit nothing here answers for.
    ///
    /// Returns only by failing: a guest that keeps running never leaves this,
    /// and a guest that stops has nothing left for the caller to resume.
    ///
    /// # Errors
    ///
    /// [`ExitError::Vcpu`] if the processor refuses the control block, naming
    /// the rule it breaks, or [`ExitError::Stopped`] if an exit was one this
    /// crate could not answer for. Nothing is rolled back: the guest is not
    /// entered again either way.
    ///
    /// # Safety
    ///
    /// As [`Vcpu::run`]: the control block must not have moved or been entered
    /// on another processor since this processor last entered it, and the guest
    /// state in it must describe a guest this hypervisor is entitled to run.
    pub unsafe fn run(&mut self, vcpu: &mut Vcpu) -> Result<Infallible, ExitError> {
        // SAFETY: forwarded to the caller, whose obligations are `Vcpu::run`'s
        // in full.
        unsafe { vcpu.run(|vcpu| self.exit(vcpu)) }?;
        Err(ExitError::Stopped)
    }

    /// Answers one exit.
    fn exit(&mut self, vcpu: &mut Vcpu) -> Flow {
        let reason = vcpu.reason();
        info!("exits: {reason:?} at rip {:#x}", vcpu.save().rip);
        // Before the exit is answered rather than after, because answering one
        // can resume the guest and the portal must be gone by the time it runs
        // again.
        self.firmware.retire(vcpu, self.partition);
        match reason {
            Some(Reason::Cpuid) => cpuid::exit(vcpu),
            Some(Reason::MsrAccess) => msr::exit(vcpu),
            Some(Reason::NestedPageFault) => nested::exit(vcpu, self.partition),
            Some(Reason::Vmmcall) => self.firmware.notified(vcpu),
            _ => {
                error!("exits: unhandled {:?}", vcpu.control().exit_code);
                Flow::Leave
            }
        }
    }
}

/// Why the guest stopped.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum ExitError {
    /// The processor would not enter the control block, or would not enter it
    /// again.
    #[error(transparent)]
    Vcpu(#[from] VcpuError),
    /// An exit was one nothing here could answer for, so the guest was left
    /// rather than resumed on state the host had not fixed.
    #[error("guest execution stopped after an unhandled VM exit")]
    Stopped,
}

/// Moves the guest past an intercepted instruction the host has completed.
///
/// The processor reports the address of the following instruction where it can,
/// which costs nothing and is right for every encoding. `fallback` is the
/// instruction's length on a processor that does not — every instruction
/// intercepted here has exactly one encoding, so its length is a constant
/// rather than something that has to be decoded.
fn advance(vcpu: &mut Vcpu, fallback: u64) {
    let next = vcpu.control().next_rip;
    vcpu.save_mut().rip = if next == 0 {
        vcpu.save().rip.wrapping_add(fallback)
    } else {
        next
    };
}
