//! The structures hardware-driven interrupt delivery runs on.
//!
//! When the processor delivers a guest's interrupts without an exit, it does
//! not guess at anything: it reads three tables and one page per processor,
//! all physically addressed, all laid out by the architecture. This module
//! builds them, once, before any guest runs, and keeps them for the life of
//! the machine; [`activation`] is everything that changes about them
//! afterwards — the running bits, the request bits other processors set, the
//! logical table a guest reprograms, and the transitions between the
//! hardware's driving and the software's.
//!
//! # What is built here and what is not
//!
//! Here are the *provisioning* halves only: the physical and logical tables,
//! one backing page per startable processor holding the register file in its
//! reset state, and the answers a control block asks for when it is composed.
//! What is not here is any enable bit: the structures are initialized while
//! the acceleration is still off, because the architecture asks for exactly
//! that order, and turning the acceleration on is a later decision made at an
//! entry boundary — [`activation::reconcile`].
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

pub(crate) mod activation;

mod backing;
mod tables;

use alloc::{vec, vec::Vec};

use svm::avic::{IncompleteIpiExit, MAX_PHYSICAL_ID, UnacceleratedAccessExit};
use vcpu::Vcpu;
use x86_64::PhysAddr;

use crate::{
    VlapicError,
    avic::{backing::ResetImage, tables::PhysicalTable},
    face,
    machine::{current, registry},
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
/// and sizes the physical table: there is an entry for every identifier up to
/// it and none past it. How much of that the hardware is told to walk is a
/// second question, asked again at every transition, because the face being
/// driven is what answers it.
/// `ipi_virtual` is whether the silicon's reading of the running bits is
/// trustworthy — the boot-time policy's answer — and decides whether the
/// running bits are ever published at all. `x2avic` is the highest index the
/// controller face a guest reaches through model-specific registers may name
/// here, or `None` where the acceleration may not drive that face at all. It is
/// the widest face the table may be indexed in, and so the limit the table's
/// own extent is judged against. Whether a guest may be *in* that face is not
/// asked here: it is recorded when the controllers are built, because a
/// controller seeded from firmware's own register can already be in it — see
/// [`crate::install`].
///
/// # What a failure leaves behind
///
/// Nothing published, and its frames not returned. The ordering is what makes
/// the first half true: every fallible step runs before anything is recorded,
/// so there is no half-described table, no processor that can be given a page
/// that does not exist, and no window in which [`backing_page`] answers out of
/// an attempt that failed. The second half is a decision rather than an
/// omission — the frames come out of the reserved chunk, which nothing else
/// allocates from before a guest runs, and the only caller of this halts the
/// machine on the error. A retry is therefore not a case that exists: it would
/// find nothing recorded and build a second set, which is why nothing here
/// pretends to be idempotent.
///
/// # Errors
///
/// [`VlapicError::AlreadyProvisioned`] on a second call,
/// [`VlapicError::NotInstalled`] if the emulated controllers do not exist
/// yet, [`VlapicError::TableTooLarge`] if the table would need more entries
/// than a control block can name, [`VlapicError::IndexBeyondMode`] if the
/// widest face the machine may drive cannot name `max_index`,
/// [`VlapicError::IdBeyondTable`] if a startable processor's identifier is
/// beyond `max_index`, [`VlapicError::IdDescribedTwice`] if two of them answer
/// to one identifier, or [`VlapicError::Paging`] if the chunk cannot spare a
/// frame or the window does not reach one it just handed out.
pub fn provision(
    space: &mut paging::AddressSpace,
    max_index: u16,
    ipi_virtual: bool,
    x2avic: Option<u16>,
) -> Result<vcpu::AvicTables, VlapicError> {
    if activation::provisioned() {
        return Err(VlapicError::AlreadyProvisioned);
    }
    // Sized for the widest face the machine may ever drive the table in, which
    // is the 32-bit one wherever the policy permits it and the eight-bit one
    // otherwise; each face's own limit is applied where the acceleration is
    // armed in it.
    let mut physical = PhysicalTable::new(max_index, x2avic.unwrap_or(MAX_PHYSICAL_ID))?;
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
    activation::establish(
        backing.into_boxed_slice(),
        table_frame.start_address(),
        logical_table,
        max_index,
        window,
        ipi_virtual,
    );
    Ok(tables)
}

/// Whether the guest may be told about the controller face its identifiers
/// are reached through in model-specific registers.
///
/// The question the machine's `CPUID` answer is judged against: a guest
/// offered a face the acceleration cannot drive is one whose mode
/// transitions would leave it delivered in software at exactly the moments
/// it believes it is accelerated, so a machine provisioned without the
/// capability withholds the bit. A machine with no acceleration at all
/// emulates the face as it emulates everything else, and offers it.
///
/// The same answer decides the face a controller may be seeded into, so that
/// what a guest is told about the feature and the state it finds its own
/// controller in cannot disagree.
#[must_use]
pub fn x2apic_offered() -> bool {
    activation::x2avic_permitted()
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
    activation::own_page()
}

/// The guest physical address the controllers' register page appears at.
///
/// Both the hardware's match register and the nested page tables' one
/// exception name it, so it is stated once here.
#[must_use]
pub const fn apic_page() -> PhysAddr {
    PhysAddr::new(ApicBase::DEFAULT_PAGE)
}

/// Brings the control block's acceleration into agreement with the guest it
/// describes, on the way into the guest.
///
/// The entry seam of every transition: the bit is set or cleared here, the
/// backing page rebuilt or carried back, and the flush asked for — and an
/// entry that changes nothing costs one comparison.
///
/// # Errors
///
/// [`VlapicError::NotProvisioned`] answers as "no acceleration", which is
/// the state of every machine the policy left on the software path; anything
/// else names a frame or a processor the caller cannot be given.
pub fn reconcile(vcpu: &mut Vcpu) -> Result<(), VlapicError> {
    activation::reconcile(vcpu)
}

/// Whether this processor's controller is being driven in hardware at this
/// moment.
///
/// The question the exit loop asks to decide what the entry may carry: a
/// processor whose controller the hardware drives takes no pending-interrupt
/// fields, and mirrors no task priority.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub fn active() -> Result<bool, VlapicError> {
    current().map(activation::active_for)
}

/// Says this processor is in the guest, so another processor delivering an
/// IPI may ring it rather than exit.
///
/// The entry half of the publication protocol; [`unpublish_running`] is the
/// exit half, and the park path clears before it waits and looks again.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub fn publish_running() -> Result<(), VlapicError> {
    activation::publish_running()
}

/// Says this processor is no longer in the guest.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub fn unpublish_running() -> Result<(), VlapicError> {
    activation::unpublish_running()
}

/// Whether this processor's backing page holds anything its guest could take
/// at this instant.
///
/// The park path's second look, and the one the software nomination cannot
/// answer while the hardware owns the request bits: a halted processor is
/// woken by what the page holds, not by what the model holds.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub fn deliverable() -> Result<bool, VlapicError> {
    activation::deliverable()
}

/// Answers an interrupt the hardware started delivering between the guest's
/// own processors and could not finish.
///
/// The whole of the exit's meaning lives here, keyed by the failure the
/// hardware reported — see [`crate::delivery::avic`] for what each one
/// becomes. Always resumes: every arm either completes the interrupt in
/// software or wakes whoever the hardware already delivered to, and neither
/// can fail in a way the guest should stop for.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub fn incomplete_ipi(exit: IncompleteIpiExit) -> Result<(), VlapicError> {
    crate::delivery::avic::incomplete_ipi(exit)
}

/// Performs the host's half of a register access the hardware completed into
/// the backing page before it exited.
///
/// The trap half of the unaccelerated access: the guest's own write already
/// landed, the processor is past it, and what is owed is the bookkeeping the
/// register asks for beyond the store.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub fn unaccelerated_trap(exit: UnacceleratedAccessExit) -> Result<(), VlapicError> {
    let Some(register) = face_register(exit.offset()) else {
        // An offset the register table does not name cannot be a trap the
        // hardware completed: it is an exit nothing here can describe, and
        // the processor's own inhibition is the honest answer.
        let vlapic = current()?;
        vlapic.inhibit_avic();
        return Ok(());
    };
    activation::trap_write(register, exit.eoi_vector())
}

/// Wakes every target of a command the hardware already delivered to but
/// could not finish, because the targets were not running.
///
/// The request bits are the hardware's already; what the targets need is
/// only to be told, which is a host interrupt each.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub fn wake_targets(command_bits: u64) -> Result<(), VlapicError> {
    crate::delivery::avic::wake_targets(command_bits)
}

/// How many hardware doorbells the machine has rung, cumulative.
///
/// One of the census's two numbers for how interrupts reached running
/// processors; the other is [`kicks`].
#[must_use]
pub fn doorbells() -> u64 {
    activation::doorbell_count()
}

/// How many host-interrupt kicks the acceleration's paths have sent,
/// cumulative.
#[must_use]
pub fn kicks() -> u64 {
    activation::kick_count()
}

/// Whether an access the hardware reported as unaccelerated is one it
/// completed before it exited.
///
/// The trap half of the classification the architecture's table gives each
/// register access; the other half is everything else, and is owed the
/// instruction rather than the bookkeeping. Stated here, against the
/// register table, so that the exit path asks one question and never
/// transcribes the table a second time.
#[must_use]
pub fn trap_access(offset: u16, write: bool) -> bool {
    face_register(offset).is_some_and(|register| {
        matches!(register.avic_access(write), face::table::AvicAccess::Trap)
    })
}

/// The register a page offset names, as the faces name it.
///
/// Kept here rather than reached into [`crate::face`] from the exit crate,
/// because the offset table is this crate's and an exit handler should not
/// need the face's internals to ask which register an offset was.
fn face_register(offset: u16) -> Option<crate::face::table::Register> {
    crate::face::table::Register::at(u64::from(offset))
}
