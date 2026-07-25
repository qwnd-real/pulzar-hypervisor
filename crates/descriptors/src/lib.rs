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
//! # Every processor
//!
//! Two of the three tables are per processor and one is not, and which is which
//! follows from what each of them holds.
//!
//! A task state segment holds stacks, and a stack cannot be shared: two
//! processors taking a double fault at once would take it on the same one. The
//! global descriptor table holds the descriptor for that task state segment, so
//! it cannot be shared either. Both are therefore built by each processor for
//! itself, in [`Descriptors::install`].
//!
//! The interrupt descriptor table holds only gates, and a gate names an entry
//! point and a selector — the same entry point on every processor, and the same
//! selector, because every processor's descriptor table puts its code segment at
//! the same index. So one table is built and every processor is pointed at it.
//!
//! What becomes of an unclaimed interrupt is not a per-processor fact at all: it
//! is what this hypervisor does. So it is said once, with [`adopt`], and saying
//! it is a precondition of any processor installing tables — an interrupt must
//! never arrive to find no answer.

#![feature(abi_x86_interrupt)]
#![no_std]

extern crate alloc;

mod dispatch;
mod gdt;
mod idt;
mod vector;

use log::info;
use paging::{AddressSpace, PagingError};
use thiserror::Error;
use x86_64::instructions::{hlt, interrupts};

pub use crate::{
    dispatch::{Disposition, Handler, Interrupt, Unclaimed, adopt, claim, register},
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
    /// Called once per processor, by that processor. Calling it twice on one
    /// processor would leave it running on a second set of tables and leak the
    /// first, which nothing here can detect and nothing has reason to do.
    ///
    /// # Errors
    ///
    /// [`DescriptorError::Unadopted`] if nothing has said yet what becomes of an
    /// unclaimed interrupt, since loading a table of gates before then would
    /// make a delivery possible that has no answer; or
    /// [`DescriptorError::Paging`] if the interrupt stacks cannot be backed.
    pub fn install(space: &mut AddressSpace) -> Result<Self, DescriptorError> {
        if !dispatch::adopted() {
            return Err(DescriptorError::Unadopted);
        }
        let selectors = interrupts::without_interrupts(|| gdt::install(space))?;
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
    /// Nothing has said what becomes of an interrupt no handler claims, so no
    /// table of gates may be loaded yet.
    #[error("nothing has adopted the unclaimed interrupts yet")]
    Unadopted,
    /// Something already said what becomes of an unclaimed interrupt, and it is
    /// one answer for the whole machine.
    #[error("the unclaimed interrupts have already been adopted")]
    AlreadyAdopted,
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
    /// Every vector in the range asked for already has a handler.
    #[error("no vector between {first} and {last} is free")]
    NoVectorFree {
        /// Low end of the range searched.
        first: Vector,
        /// High end of the range searched.
        last: Vector,
    },
}
