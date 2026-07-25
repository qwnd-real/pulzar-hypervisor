//! The descriptor tables the hypervisor runs on, and the path every interrupt
//! takes through them.
//!
//! Firmware's tables live in the lower half of the address space and stop
//! existing the moment it is dropped, so having tables of our own is a
//! precondition for that step rather than an improvement on it. But the
//! interesting part is not that they exist — it is what the interrupt
//! descriptor table is for in a hypervisor that does not own the machine.
//!
//! # Sharing vectors with something else
//!
//! Pulzar passes the platform through. It will arm an APIC timer and send
//! interprocessor interrupts, and those arrive on the same vectors as
//! everything else that was already using the machine. Two things follow. Every
//! vector needs an entry point, because any of them may turn out to carry
//! something the hypervisor caused. And an entry point cannot decide on its own
//! what it is looking at, because whether a given arrival was the hypervisor's
//! own doing is visible only to the subsystem that armed it.
//!
//! So [`dispatch`] splits the two apart: a subsystem [`register`]s a
//! [`Handler`] on its vector and answers with a [`Disposition`], and anything
//! not claimed reaches the [`Unclaimed`] callback the hypervisor supplied. That
//! callback is where giving an interrupt back to a guest will happen.
//!
//! # Shape
//!
//! - [`vector`] states what the architecture fixes about each of the 256
//!   vectors: error code, returnability, name, and which stack it switches to.
//!   Everything else derives its behaviour from it rather than repeating it.
//! - [`gdt`] builds the segments and the task state segment, whose only real
//!   content is the seven stacks the most dangerous exceptions run on.
//! - [`idt`] gives every vector a gate and an entry point that knows its own
//!   number.
//! - [`dispatch`] is where they all arrive and where the hypervisor's own
//!   interrupts are separated from everyone else's.
//!
//! # One processor
//!
//! These are the boot processor's tables. The interrupt descriptor table is the
//! same for every processor and can be shared as it stands, but a task state
//! segment cannot: each processor needs its own stacks and its own descriptor
//! for them. Starting the others is what will introduce that, and it is not
//! pretended at here.

#![feature(abi_x86_interrupt)]
#![no_std]

mod dispatch;
mod gdt;
mod idt;
mod vector;

use core::sync::atomic::{AtomicBool, Ordering};

use log::info;
use paging::{AddressSpace, PagingError};
use thiserror::Error;
use x86_64::instructions::{hlt, interrupts};

pub use crate::{
    dispatch::{Disposition, Handler, Interrupt, Unclaimed, register},
    gdt::Selectors,
    vector::{InterruptStack, Vector},
};

/// The descriptor tables this processor is running on.
#[derive(Clone, Copy, Debug)]
pub struct Descriptors {
    selectors: Selectors,
}

impl Descriptors {
    /// Builds the tables, switches the processor onto them, and makes every
    /// vector deliverable.
    ///
    /// The order inside is forced by the hardware and cannot be rearranged. A
    /// gate descriptor records the code selector its handler is entered with,
    /// taken from whatever `CS` holds when the gate is written, so the global
    /// descriptor table has to be loaded and `CS` reloaded before any gate
    /// exists. Interrupts must stay masked across the whole of it: between
    /// those two steps the live interrupt descriptor table is still firmware's,
    /// and its gates name selectors in a table that is no longer loaded.
    ///
    /// # Errors
    ///
    /// [`DescriptorError::AlreadyInstalled`] if this processor already has
    /// tables — replacing them underneath itself is never what a second caller
    /// wants — or [`DescriptorError::Paging`] if the interrupt stacks cannot be
    /// backed.
    pub fn install(
        space: &mut AddressSpace,
        unclaimed: Unclaimed,
    ) -> Result<Self, DescriptorError> {
        if INSTALLED
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return Err(DescriptorError::AlreadyInstalled);
        }
        let selectors = interrupts::without_interrupts(|| gdt::install(space))?;
        // Before the table that names the entry points is loaded, so that no
        // delivery can find no answer.
        dispatch::adopt(unclaimed);
        idt::install();
        Ok(Self { selectors })
    }

    /// The selectors into the global descriptor table.
    #[must_use]
    pub const fn selectors(&self) -> Selectors {
        self.selectors
    }

    /// Logs what was installed.
    pub fn describe(&self, who: &str) {
        info!(
            "{who}: gdt loaded, cs {:#06x}, ds {:#06x}, tss {:#06x}",
            self.selectors.code.0, self.selectors.data.0, self.selectors.task.0
        );
        info!(
            "{who}: idt loaded, {} vectors, {} on stacks of their own",
            Vector::COUNT,
            InterruptStack::COUNT
        );
    }
}

/// Stops this processor for good.
///
/// Interrupts are masked before each halt, so the only thing that can wake the
/// processor is a non-maskable interrupt, and the loop puts it straight back to
/// sleep if one does.
///
/// It lives here because this is the crate that owns what happens when the
/// processor cannot continue: an interrupt no one will ever claim is the case
/// this crate must answer on its own, and having one way to stop rather than
/// one per caller is what keeps that answer the same everywhere.
pub fn halt() -> ! {
    loop {
        interrupts::disable();
        hlt();
    }
}

/// Why the descriptor tables could not be set up or added to.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum DescriptorError {
    /// The interrupt stacks could not be allocated or mapped.
    #[error(transparent)]
    Paging(#[from] PagingError),
    /// This processor already has descriptor tables.
    #[error("this processor already has descriptor tables")]
    AlreadyInstalled,
    /// Something already claimed the vector, and a vector holds one handler.
    #[error("{vector} already has a handler")]
    VectorTaken {
        /// The vector in question.
        vector: Vector,
    },
    /// The architecture gives no defined way back from this vector, so nothing
    /// may claim it: consuming it would mean resuming a machine whose state the
    /// processor has already declared lost.
    #[error("{vector} cannot be returned from, so nothing may claim it")]
    NotReturnable {
        /// The vector in question.
        vector: Vector,
    },
}

/// Claimed by the first install, so a second cannot replace the tables the
/// processor is already running on while it is running on them.
static INSTALLED: AtomicBool = AtomicBool::new(false);
