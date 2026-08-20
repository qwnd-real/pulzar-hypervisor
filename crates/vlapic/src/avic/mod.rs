//! The structures hardware-driven interrupt delivery runs on.
//!
//! When the processor delivers a guest's interrupts without an exit, it does
//! not guess at anything: it reads three tables and one page per processor,
//! all physically addressed, all laid out by the architecture. This module
//! builds them, once, before any guest runs, and keeps them for the life of
//! the machine.
//!
//! # What is built here and what is not
//!
//! Here are the *provisioning* halves only: the physical and logical tables,
//! one backing page per startable processor holding the register file in its
//! reset state, and the answers a control block asks for when it is composed.
//! What is not here is any enable bit: the structures are initialized while
//! the acceleration is still off, because the architecture asks for exactly
//! that order, and turning the acceleration on is a later decision made
//! elsewhere.
//!
//! # Why the backing page is a copy of the reset state
//!
//! The hardware serves a number of the controller's registers out of the
//! backing page without any exit — the identifier and version among them,
//! which no write of the guest's ever changes. So the page must already hold
//! what those registers answer with before the first entry, and that is
//! exactly the state the software model comes out of reset in. The image is
//! therefore built from the same constants the model's own reset uses, and
//! the two are tested against each other: a byte that disagrees is a register
//! the guest would read differently depending on whether the hardware or the
//! emulator answered it.
//!
//! # One set for the machine
//!
//! The tables describe the guest as a whole and there is one guest, so
//! [`provision`] runs once on the boot processor and refuses a second call.
//! What is per-processor is the backing page, and [`backing_page`] answers
//! which one belongs to the processor asking.

mod backing;
mod tables;

use alloc::{boxed::Box, vec, vec::Vec};

use spin::Once;
use x86_64::PhysAddr;

use crate::{
    VlapicError,
    avic::{backing::ResetImage, tables::PhysicalTable},
    machine::registry,
    registers::base::ApicBase,
};

/// Builds the structures hardware-driven interrupt delivery runs on, and
/// publishes them.
///
/// One backing page per startable processor, holding the register file in its
/// reset state; the physical table, with an entry per processor naming its
/// page; and the logical table, a page of zeroes — allocated even though no
/// logical destination is described in it yet, because a control block is
/// asked for the address either way and must be given a real one.
///
/// `max_index` is the highest identifier any startable processor answers to,
/// and sizes the physical table: the hardware walks no further than it.
///
/// # Errors
///
/// [`VlapicError::AlreadyProvisioned`] on a second call,
/// [`VlapicError::NotInstalled`] if the emulated controllers do not exist
/// yet, [`VlapicError::TableTooLarge`] if the table would need more than one
/// page, [`VlapicError::IdBeyondTable`] if a startable processor's identifier
/// is beyond `max_index`, or [`VlapicError::Paging`] if the chunk cannot spare
/// a frame or the window does not reach one it just handed out.
pub fn provision(
    space: &mut paging::AddressSpace,
    max_index: u16,
) -> Result<vcpu::AvicTables, VlapicError> {
    if PROVISIONED.is_completed() {
        return Err(VlapicError::AlreadyProvisioned);
    }
    let mut physical = PhysicalTable::new(max_index)?;
    let window = space.direct_map();
    let lapics = registry::lapics()?;
    let mut backing: Vec<Option<PhysAddr>> = vec![None; lapics.all().len()];
    for vlapic in lapics.all() {
        if !vlapic.startable() {
            continue;
        }
        let page = space.frames().allocate(0)?.start_address();
        let image = ResetImage::new(vlapic.apic_id(), vlapic.version());
        // SAFETY: the frame was just allocated out of the reserved chunk, so it
        // is RAM and not a device aperture, it is zeroed and nothing else holds
        // a reference to it or will until this call publishes it.
        unsafe { window.write(page, image.bytes())? };
        physical.describe(vlapic.apic_id(), page)?;
        backing[vlapic.index().get()] = Some(page);
    }
    let logical_table = space.frames().allocate(0)?.start_address();
    let table_frame = space.frames().allocate(physical.order())?;
    physical.write(window, table_frame.start_address())?;
    let tables = vcpu::AvicTables {
        apic_bar: apic_page(),
        logical_table,
        physical_table: svm::avic::AvicPhysicalTable::new()
            .with_max_index(max_index)
            .with_address(table_frame.start_address()),
    };
    let state = Box::new(Provisioned {
        backing: backing.into_boxed_slice(),
    });
    PROVISIONED.call_once(|| &*Box::leak(state));
    Ok(tables)
}

/// The page the processor asking has its controller registers backed by.
///
/// What a control block's backing-page field is composed out of: each
/// processor carries its own page, and the one asking is the one a control
/// block is being built for.
///
/// # Errors
///
/// [`VlapicError::NotProvisioned`] before [`provision`], or
/// [`VlapicError::NoLapic`] if the processor asking has no page — either the
/// roster does not describe it or firmware said it may not be started.
pub fn backing_page() -> Result<PhysAddr, VlapicError> {
    let state = PROVISIONED.get().ok_or(VlapicError::NotProvisioned)?;
    // SAFETY: nothing reaches this before the processor has attached — a
    // guest cannot be entered until bring-up is past that point — so this
    // processor's `GS` base points at its own block.
    let index = unsafe { cpu::current() }.index();
    state
        .backing
        .get(index.get())
        .copied()
        .flatten()
        .ok_or(VlapicError::NoLapic)
}

/// The guest physical address the controllers' register page appears at.
///
/// Both the hardware's match register and the nested page tables' one
/// exception name it, so it is stated once here.
#[must_use]
pub const fn apic_page() -> PhysAddr {
    PhysAddr::new(ApicBase::DEFAULT_PAGE)
}

/// The per-processor pages, once built.
///
/// They are indexed by roster position, which is what a processor is turned
/// into when it asks for its own; a processor firmware described as
/// unstartable has no page and is recorded as having none.
struct Provisioned {
    backing: Box<[Option<PhysAddr>]>,
}

/// Built once, by the boot processor, before any guest has run.
static PROVISIONED: Once<&'static Provisioned> = Once::new();
