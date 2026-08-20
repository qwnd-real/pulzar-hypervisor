//! One processor's emulated controller, and everything it remembers.
//!
//! # Everything here is atomic, and there is no lock
//!
//! A local controller is per-processor state that other processors write. That
//! is not an implementation choice — it is what an interprocessor interrupt
//! *is*: one processor reaching into another's controller and setting a bit.
//! So the register file cannot be owned by the processor it describes, and
//! putting it behind a lock would mean taking that lock on the path an
//! interrupt is delivered on, from inside an interrupt handler, on every
//! processor at once.
//!
//! Instead every field is an atomic and there is no lock at all. That is
//! affordable because almost nothing here is a read-modify-write of more than
//! one field: a guest's register access is a load or a store, delivery is a
//! bit set, and the one genuinely compound operation — moving a vector from
//! requested to in service — is performed only by the processor that owns the
//! controller, which is the only one that ever clears a request bit.
//!
//! The division of labour that makes that true is worth stating outright:
//!
//! - **Any processor** may set a bit in the request and trigger-mode registers,
//!   record an error, and change the startup state. Those are what delivery is.
//! - **Only the owning processor** clears a request bit, touches the in-service
//!   register, or writes any of the registers its guest programs — because
//!   those writes come out of that guest, which runs nowhere else.
//!
//! # Reset is the one thing that is not a single field
//!
//! Clearing the register file touches four bitmaps and a dozen registers, and
//! it happens while other processors may be delivering into it. Without
//! something to order them against each other, a level-triggered interrupt
//! accepted half-way through could end up with its request bit surviving and
//! its trigger-mode bit cleared — which is a level interrupt that will be
//! treated as an edge one, and so a real acknowledgement that is never issued
//! and a line that never fires again.
//!
//! [`Vlapic::epoch`] is what orders them. It counts resets, and is odd exactly
//! while one is in progress. A deliverer publishes into the register file and
//! then checks that the count did not move underneath it; if it did, it
//! publishes again into the state the reset left. That is deliberately a retry
//! rather than a withdrawal: an interrupt racing a reset arrived at a moment
//! nothing distinguishes from just after it, and just after it is when the new
//! guest is entitled to see it.
//!
//! # What the guest may not change
//!
//! The identifier is read-only, and not merely because recent processors made
//! it so. Every interrupt this hypervisor passes through — from an I/O
//! controller, from a device's message — is routed by hardware using the *real*
//! identifier. A guest that renamed its controller would be describing a
//! machine whose interrupts could no longer be delivered to it.
//!
//! # Where the operations are
//!
//! One file per register family, each of them an `impl Vlapic` block over the
//! fields declared here. The fields are private to this module and its
//! descendants, so everything above reaches them through an operation named for
//! what the architecture calls it — and an operation cannot be written anywhere
//! a reader would not think to look for it.

pub(crate) mod base;
pub(crate) mod bitmap;
pub(crate) mod error;
pub(crate) mod icr;
pub(crate) mod lvt;

mod file;
mod identity;
mod interrupts;
mod spurious;
mod startup;
mod task_priority;
mod timer;

use core::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64};

use cpu::{ApicId, CpuIndex};

pub(crate) use crate::registers::{
    error::Raise,
    identity::{FLAT_DESTINATION_FORMAT, xapic_word},
    interrupts::Accepted,
    spurious::{SPURIOUS_RESET, SPURIOUS_WRITABLE},
    startup::{Phase, Startup},
    task_priority::TASK_PRIORITY_MASK,
    timer::TIMER_DIVIDE_MASK,
};
pub use crate::registers::{interrupts::Nomination, startup::StartupPage};
use crate::{
    hardware::model::Model,
    lifecycle::ledger::Ledger,
    machine::diagnostics::Diagnostics,
    registers::{bitmap::Bitmap, error::ErrorStatus, lvt::Entry},
};

/// One processor's emulated local interrupt controller.
///
/// Aligned to a cache line, and sized to whole ones, because these are held in
/// one array for the machine and written by every processor: two controllers
/// sharing a line would have an interrupt delivered to one processor
/// invalidating the line another is reading its own state out of.
#[derive(Debug)]
#[repr(align(64))]
pub(crate) struct Vlapic {
    index: CpuIndex,
    apic_id: ApicId,
    startable: bool,
    model: Model,
    base: AtomicU64,
    request: Bitmap,
    in_service: Bitmap,
    trigger_mode: Bitmap,
    /// Which requested vectors reached this guest through the pin that bypasses
    /// its controller, and so must not be held in service when they are given
    /// to it.
    ///
    /// An external interrupt is answered by an acknowledge cycle to a legacy
    /// controller, and neither controller's in-service register takes any part
    /// in it — so the guest acknowledges the legacy controller and nothing
    /// else. An in-service bit set for one of these would be a bit the
    /// guest has no reason to clear, holding its whole interrupt-priority
    /// class for good.
    external: Bitmap,
    task_priority: AtomicU32,
    logical_destination: AtomicU32,
    destination_format: AtomicU32,
    spurious: AtomicU32,
    lvt: [AtomicU32; Entry::COUNT],
    timer_divide: AtomicU32,
    timer_initial: AtomicU32,
    timer_frequency: AtomicU64,
    command: AtomicU64,
    errors: ErrorStatus,
    ledger: Ledger,
    epoch: AtomicU64,
    startup: Startup,
    away: AtomicBool,
    nmi: AtomicU8,
    owned: AtomicBool,
    /// What has happened to this controller and what it has already said about
    /// it. Diagnostic only: nothing above reads it back except the report that
    /// writes it and [`crate::describe`].
    diagnostics: Diagnostics,
    /// The last selection state `Vlapic::report` logged, packed into one word
    /// by that function, so a controller whose answer has not changed stays
    /// quiet. Diagnostic only: nothing reads it back but the report.
    reported: AtomicU64,
}
