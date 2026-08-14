//! Running a guest on one processor.
//!
//! The `svm` crate says what the processor's virtualization structures are;
//! this is the first code that drives them. It turns the extension on,
//! allocates a guest a control block, enters the guest, and hands back what the
//! guest did.
//!
//! # Mechanism, not policy
//!
//! Nothing here decides anything about a guest. It does not choose what to
//! intercept beyond the one intercept the architecture makes mandatory, it does
//! not write a single register the guest will run with, and it does not act on
//! any exit. [`Vcpu::run`] enters the guest and hands each exit to a closure,
//! and what that closure does is somebody else's crate.
//!
//! That line is where it is because the interesting decisions — what a guest
//! may do freely, what its memory is, what happens when it stops — all belong
//! to a guest as a whole rather than to one of its processors, and this crate
//! is about one of its processors.
//!
//! # Shape
//!
//! - [`Host`] is what a processor needs before any guest runs on it: the
//!   extension enabled, a page for the processor to swap host state through,
//!   and a snapshot of the host state a world switch does not restore by
//!   itself. Once per processor.
//! - [`Vcpu`] is one virtual processor: its control block, its registers, and
//!   the loop.
//! - [`Registers`] is the fourteen general-purpose registers the architecture
//!   leaves to the hypervisor, and the complete list of what a world switch has
//!   to move by hand.
//! - [`Invalid`] is the architecture's entry rules, checked in software so that
//!   a refused control block reports the rule rather than the refusal.
//!
//! # Order
//!
//! On every processor that is to run a guest, in this order:
//!
//! 1. Descriptor tables, and then attach — both before [`Host::install`], which
//!    snapshots the state those two establish.
//! 2. [`Host::install`], once.
//! 3. [`Vcpu::create`], once per virtual processor this processor will run.
//!
//! # What is deliberately absent
//!
//! Nothing here saves floating-point or vector state across a world switch. The
//! architecture does not swap it, so a guest's is still live in the processor
//! while the hypervisor runs — and nothing the hypervisor runs can touch it,
//! because this image is built for a target whose feature string disables MMX
//! and SSE and uses software floating point. Should that ever change, the world
//! switch is where the swap would have to go.
//!
//! One rule of the architecture's is also absent from [`Invalid`]: a guest
//! cannot enable long mode on a processor without it, and no processor without
//! long mode can execute this image, so there is nothing to check.

#![no_std]

extern crate alloc;

mod host;
mod invalid;
mod registers;
mod switch;
mod vcpu;

use thiserror::Error;

pub use crate::{
    host::Host,
    invalid::Invalid,
    registers::{RAX, RSP, Registers},
    vcpu::{Flow, Guest, RunPhase, Vcpu},
};

/// Why a processor could not be prepared to run a guest, or why a guest could
/// not be run.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum VcpuError {
    /// The processor has no virtualization extension. Every machine pulzar runs
    /// on must have one.
    #[error("this processor has no virtualization extension")]
    NoSvm,
    /// Firmware disabled the extension on a processor with no key mechanism, so
    /// nothing software can do will re-enable it and a firmware setting has to
    /// be changed by hand.
    #[error("firmware disabled the virtualization extension and left no way to re-enable it")]
    SvmDisabled,
    /// Firmware disabled the extension and locked it, but this processor has
    /// the key mechanism, so the lock could in principle be lifted by
    /// whoever holds the key.
    #[error("firmware locked the virtualization extension off")]
    SvmLocked,
    /// The processor can tag translations with too few address spaces to run
    /// any guest: identifier zero is the host's, so at least two are
    /// needed.
    #[error("this processor supports {asids} address space identifiers, too few for a guest")]
    TooFewAsids {
        /// How many it reported.
        asids: u32,
    },
    /// The reserved chunk has no frame left for a control block.
    #[error("the reserved chunk has no frame left for a control block")]
    OutOfFrames,
    /// The window onto physical memory does not reach a page the processor was
    /// to be given.
    #[error("physical {phys:#x} is outside the window onto physical memory")]
    Unreachable {
        /// The address that could not be reached.
        phys: u64,
    },
    /// The control block breaks one of the architecture's entry rules.
    #[error("the control block would not be entered: {0}")]
    Invalid(#[from] Invalid),
    /// The chunk's allocator would not take a page back.
    #[error(transparent)]
    Paging(#[from] paging::PagingError),
}
