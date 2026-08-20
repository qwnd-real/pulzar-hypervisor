//! The local interrupt controller a guest sees, in place of the machine's own.
//!
//! Pulzar passes the platform through. It does not pass the interrupt
//! controller through, and the difference is the point: a guest that reached
//! the real controller would be a guest that could mask the host's interrupts,
//! acknowledge the host's, send the host's processors a reset, and rename the
//! processor an interrupt is addressed by. Every one of those is something a
//! hypervisor has to keep.
//!
//! So the controller is emulated and the hardware behind it stays the host's.
//! Every interrupt on the machine arrives in a host handler first, and reaches
//! the guest only as a decision this crate made.
//!
//! # What is emulated and what is not
//!
//! The register file is emulated in full: the request, in-service and
//! trigger-mode banks, the priorities, the error status and its write-then-read
//! protocol, the interrupt command register, both destination models, and the
//! whole of the base register's state machine.
//!
//! The *sources* are not. The timer is the real timer, programmed with the
//! guest's own divide, count and mode, because there is nothing to be gained by
//! counting the same numbers twice — and the same will hold for the rest of the
//! local vector table. What is never passed through is a vector: the real entry
//! carries a vector this crate claimed, so that an arrival is unambiguous, and
//! the guest's own vector is what gets injected.
//!
//! # One page, every processor
//!
//! The memory-mapped face is one page of guest physical memory at the same
//! address on every processor, each seeing its own controller through it. So
//! the device registered for that page is a single device holding one
//! controller per processor, picking out the row belonging to whichever
//! processor took the exit — which is exactly the shape of the hardware it
//! stands for.
//!
//! # How to read this crate
//!
//! Seven directories, each answering one question about the controller:
//!
//! - `machine` — one controller per processor, and everything true of the
//!   machine rather than of any one of them: installation, the registry, which
//!   processors this hypervisor owns, and what each exit loop asks of its own.
//! - `face` — how a guest reaches its controller: the page, the model-specific
//!   registers, and the one statement of what each register means that both of
//!   them share.
//! - `registers` — the register file itself, one file per register family.
//! - `delivery` — which processors a command names, and how they are told.
//! - `lifecycle` — what becomes of a controller between one guest and the next,
//!   and what real hardware is owed across it.
//! - `hardware` — the surface where the guest's registers become physical ones.
//! - `avic` — the structures hardware-driven delivery runs on, for a processor
//!   that delivers a guest's interrupts without an exit: the tables and the
//!   backing pages, built once before any guest runs.
//!
//! `priority` is on its own because it is the one rule everything else
//! compares against, and it holds no state at all. It is also the one thing
//! here that is not only this crate's: the deliverability rule is evaluated in
//! halves, one of them by the processor against fields another crate writes
//! into a control block, so [`Priority`] is public and is the workspace's
//! single statement of what an interrupt-priority class is.
//!
//! # What is tested, and what cannot be
//!
//! Every decision this crate makes is tested where it can be reached without a
//! machine, which is why so much of it is written as a function of values
//! rather than as a method on a controller: which register an access names,
//! what a command asks for, which processors a destination names, what an entry
//! becomes on real hardware, which writes are ordered against hardware, what a
//! debt becomes, and the whole of the priority arithmetic and the startup state
//! machine.
//!
//! What has no test is what needs a controller, and no host
//! test can have one: a controller is built from a roster entry,
//! `cpu::CpuIndex` has no public constructor, and construction resets the
//! register file — which holds this processor's interrupts off with an
//! instruction a test process may not execute. So the modules that only pair a
//! controller with the machine behind it — `machine/{install, ownership,
//! registry, exits}`, `face/dispatch`, `delivery/{mod, doorbell, error}`,
//! `hardware/mirror`, `hardware/timer/{mod, inherit}`, `lifecycle/{mod,
//! settle}` and the `registers` files that are accessors over its fields — are
//! covered through the pure decisions they call and by inspection of the
//! sequencing, and the module roots hold documentation rather than code. Making
//! the rest reachable needs a constructor for a processor index and a
//! controller that can be built without masking interrupts, neither of which is
//! this crate's to add.

#![no_std]

extern crate alloc;

mod avic;
mod delivery;
mod face;
mod hardware;
mod lifecycle;
mod machine;
mod priority;
mod registers;

use cpu::CpuError;
use descriptors::DescriptorError;
use thiserror::Error;

pub use crate::{
    avic::{
        active as avic_active, apic_page, backing_page, deliverable as avic_deliverable,
        doorbells as avic_doorbells, incomplete_ipi as avic_incomplete_ipi, kicks as avic_kicks,
        provision, publish_running as avic_publish_running, reconcile as avic_reconcile,
        trap_access as avic_trap_access, unaccelerated_trap as avic_unaccelerated_trap,
        unpublish_running as avic_unpublish_running, wake_targets as avic_wake_targets,
        x2apic_offered,
    },
    face::{
        mmio::region,
        msr::{apic_enabled, claims, intercepted, read_msr, write_msr},
    },
    hardware::timer::adjust_deadline,
    lifecycle::{arrival::arrived, hold, running, settle, settle::Resumption},
    machine::{
        diagnostics::describe,
        exits::{
            committed, nominate, observe_task_priority, raise_nmi, set_away, take_nmi,
            task_priority,
        },
        install::install,
        ownership::{Joining, bring_up_finished, claim_processor},
    },
    priority::Priority,
    registers::{Nomination, StartupPage},
};

/// Why the emulated controllers could not be set up or driven.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum VlapicError {
    /// Nothing has built the controllers yet.
    #[error("the emulated controllers have not been installed")]
    NotInstalled,
    /// They have, and there is one set for the machine.
    #[error("the emulated controllers have already been installed")]
    AlreadyInstalled,
    /// Firmware left the memory-mapped register page somewhere other than the
    /// address this hypervisor traps.
    ///
    /// Refused rather than accommodated. Everything but the one trapped page is
    /// an identity map of machine physical memory, so a guest on such a machine
    /// would reach the real local APIC untrapped at firmware's address — which
    /// is the whole of what this crate exists to prevent.
    #[error("firmware put the apic register page at {page:#x}, which this hypervisor cannot trap")]
    MisplacedPage {
        /// Where firmware put it.
        page: u64,
    },
    /// The roster does not describe the processor asking.
    #[error("this processor has no emulated controller")]
    NoLapic,
    /// The guest should take a general protection fault for what it asked.
    #[error("the guest's access to its controller is not one the architecture allows")]
    Fault,
    /// The real controller refused something.
    #[error(transparent)]
    Apic(#[from] apic::ApicError),
    /// The processor roster refused something.
    #[error(transparent)]
    Cpu(#[from] CpuError),
    /// No vector was free for a source this crate programs onto real hardware.
    #[error(transparent)]
    Descriptors(#[from] DescriptorError),
    /// A processor inside the guest could not be interrupted.
    #[error(transparent)]
    Ipi(#[from] ipi::IpiError),
    /// The structures hardware-driven delivery runs on were built twice; there
    /// is one set for the machine.
    #[error("the interrupt-acceleration structures have already been provisioned")]
    AlreadyProvisioned,
    /// They were asked of before they were built.
    #[error("the interrupt-acceleration structures have not been provisioned")]
    NotProvisioned,
    /// The physical table was asked to hold more entries than one page of
    /// them, which is the most a control block can name.
    #[error("a physical interrupt table of {entries} entries does not fit one page")]
    TableTooLarge {
        /// How many entries were asked for.
        entries: usize,
    },
    /// A startable processor's identifier is beyond the largest index the
    /// table was sized for, which the table cannot express.
    #[error("apic id {id} is beyond the physical interrupt table's largest index {max_index}")]
    IdBeyondTable {
        /// The identifier that does not fit.
        id: u32,
        /// The largest index the table was sized for.
        max_index: u16,
    },
    /// The reserved chunk had no frame left, or the window did not reach one
    /// it just handed out.
    #[error(transparent)]
    Paging(#[from] paging::PagingError),
    /// A control block refused a change the acceleration asked of it, which
    /// is the permission map it carries: the structures were being moved at
    /// an entry boundary, and the block said no.
    #[error(transparent)]
    Vcpu(#[from] vcpu::VcpuError),
}
