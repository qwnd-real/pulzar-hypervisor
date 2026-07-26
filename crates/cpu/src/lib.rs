//! Which processors this machine has, and which one is running.
//!
//! Everything above this crate that does something per processor needs two
//! answers, and neither is available from anywhere else. What is the set of
//! processors — which is firmware's to say, in a table whose identifiers are
//! not indices and are not promised to be ordered or dense. And which one am I
//! — which the processor itself has to answer, cheaply, from an interrupt
//! handler, with nothing passed in.
//!
//! So this crate does two things and nothing else. [`survey`] reads the set out
//! of the multiple APIC description table into a [`Roster`], which hands out
//! the dense [`CpuIndex`] that other subsystems may size arrays by. And
//! [`attach`] gives the calling processor a [`Block`] of its own, reached
//! through its `GS` base, so that [`current`] is one load.
//!
//! It deliberately owns no hardware. The local APIC identifier a processor
//! attaches with is passed in, by the subsystem that just brought that
//! processor's interrupt controller up and read it — which keeps the layering
//! one way round and this crate free of register access.
//!
//! # Order
//!
//! On every processor, in this order and for reasons that are not
//! interchangeable:
//!
//! 1. Install descriptor tables. Loading a selector into `GS` zeroes its base,
//!    so this cannot come after step 3.
//! 2. Bring the local interrupt controller up, which is where the identifier
//!    comes from.
//! 3. [`attach`], which publishes the block and makes the processor a thing
//!    other processors may send interrupts to and wait on.
//! 4. Unmask interrupts. Not before 3: a handler that ran first would call
//!    [`current`] with no base set.
//!
//! [`survey`] happens once, on the boot processor, before any of it.

#![no_std]

extern crate alloc;

mod block;
mod roster;

use alloc::{boxed::Box, vec::Vec};
use core::{
    ptr::null_mut,
    sync::atomic::{AtomicPtr, Ordering},
};

use acpi::Processor;
use log::info;
use spin::Once;
use thiserror::Error;

pub use crate::{
    block::{Block, attached, current},
    roster::{ApicId, CpuIndex, Entry, Roster},
};

/// Records what firmware said about the machine's processors.
///
/// One-shot, on the boot processor, before any processor attaches: everything
/// afterwards indexes arrays sized by [`Roster::count`], and a second survey
/// could change that length underneath them.
///
/// # Errors
///
/// [`CpuError::AlreadySurveyed`] for a second call, or
/// [`CpuError::NoProcessors`] if firmware described none — a machine whose own
/// tables do not mention the processor reading them is one nothing here can
/// reason about.
pub fn survey(processors: &[Processor]) -> Result<(), CpuError> {
    if processors.is_empty() {
        return Err(CpuError::NoProcessors);
    }
    if MACHINE.is_completed() {
        return Err(CpuError::AlreadySurveyed);
    }
    let roster = Roster::new(processors);
    let blocks = (0..roster.count())
        .map(|_| AtomicPtr::new(null_mut()))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    MACHINE.call_once(|| Machine { roster, blocks });
    Ok(())
}

/// Gives the calling processor a block of its own and makes it reachable from
/// the others.
///
/// `apic_id` is what this processor read out of its own interrupt controller,
/// which is what ties it to the entry firmware described — never to a position,
/// because identifiers are not positions.
///
/// # Errors
///
/// [`CpuError::NotSurveyed`] before [`survey`], [`CpuError::Unknown`] if no
/// entry has this identifier, which means the processor running is one firmware
/// did not describe, or [`CpuError::AlreadyAttached`] if the entry already has
/// a block.
pub fn attach(apic_id: ApicId) -> Result<&'static Block, CpuError> {
    let machine = machine()?;
    let index = machine
        .roster
        .find(apic_id)
        .ok_or(CpuError::Unknown { apic_id })?
        .index();
    let slot = &machine.blocks[index.get()];
    if !slot.load(Ordering::Acquire).is_null() {
        return Err(CpuError::AlreadyAttached { index });
    }
    // This processor's own, because this processor is the one that read
    // `apic_id` out of its own controller.
    let block = Block::activate(Box::leak(Box::new(Block::new(index, apic_id))));
    // Published last, with release ordering, so that a processor which finds
    // this one online also sees a block that is fully written and a base that is
    // already set.
    slot.store(core::ptr::from_ref(block).cast_mut(), Ordering::Release);
    Ok(block)
}

/// The block of the processor at `index`, or `None` if it has not attached.
///
/// This is how one processor reaches another's state, and how "is it online"
/// is answered: a processor is online exactly when it has a published block,
/// because publishing it is the last thing attaching does.
#[must_use]
pub fn by_index(index: CpuIndex) -> Option<&'static Block> {
    let slot = MACHINE.get()?.blocks.get(index.get())?;
    // SAFETY: a slot holds either null — filtered out by `as_ref` — or the
    // address of a leaked `Block` stored by `attach`, which can never dangle.
    unsafe { slot.load(Ordering::Acquire).as_ref() }
}

/// Every processor that has attached.
pub fn online() -> impl Iterator<Item = &'static Block> {
    MACHINE
        .get()
        .map(|machine| machine.blocks.as_ref())
        .unwrap_or_default()
        .iter()
        // SAFETY: as in `by_index`.
        .filter_map(|slot| unsafe { slot.load(Ordering::Acquire).as_ref() })
}

/// How many processors have attached.
#[must_use]
pub fn online_count() -> usize {
    online().count()
}

/// What firmware said about the machine's processors.
///
/// # Errors
///
/// [`CpuError::NotSurveyed`] before [`survey`].
pub fn roster() -> Result<&'static Roster, CpuError> {
    machine().map(|machine| &machine.roster)
}

/// Logs the roster and who is up.
///
/// One line per processor, because which processors exist, which may be
/// started, and which are running is the whole of what this crate knows.
pub fn describe(who: &str) {
    let Ok(machine) = machine() else {
        info!("{who}: cpu roster not taken yet");
        return;
    };
    info!(
        "{who}: cpu {} processors described, {} online, identifiers {}",
        machine.roster.count(),
        online_count(),
        if machine.roster.needs_x2apic() {
            "beyond xapic's eight bits"
        } else {
            "within xapic's eight bits"
        },
    );
    for entry in machine.roster.entries() {
        info!(
            "{who}: {} is {}, uid {}, {:?}, {}",
            entry.index(),
            entry.apic_id(),
            entry.uid(),
            entry.state(),
            match by_index(entry.index()) {
                Some(_) => "online",
                None if entry.startable() => "offline",
                None => "not to be started",
            },
        );
    }
}

/// Why a processor could not be described or attached.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum CpuError {
    /// Firmware's table described no processors at all.
    #[error("firmware described no processors")]
    NoProcessors,
    /// The roster has already been taken, and its length is what everything
    /// above has sized its arrays by.
    #[error("the processor roster has already been taken")]
    AlreadySurveyed,
    /// Nothing has read firmware's table yet.
    #[error("the processor roster has not been taken yet")]
    NotSurveyed,
    /// A processor is running that firmware's own table does not mention.
    #[error("no processor firmware described has {apic_id}")]
    Unknown {
        /// The identifier that matched nothing.
        apic_id: ApicId,
    },
    /// The processor already has a block, so something attached twice.
    #[error("{index} has already attached")]
    AlreadyAttached {
        /// The processor in question.
        index: CpuIndex,
    },
}

/// The roster, and one slot per processor for the block it publishes.
#[derive(Debug)]
struct Machine {
    roster: Roster,
    blocks: Box<[AtomicPtr<Block>]>,
}

/// What firmware said, read once.
static MACHINE: Once<Machine> = Once::new();

/// The surveyed machine.
fn machine() -> Result<&'static Machine, CpuError> {
    MACHINE.get().ok_or(CpuError::NotSurveyed)
}
