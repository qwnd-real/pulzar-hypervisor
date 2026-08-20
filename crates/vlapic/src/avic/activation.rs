//! The runtime halves of hardware-driven delivery.
//!
//! Provisioning builds the tables and the backing pages once, before any guest
//! runs; everything that changes afterwards lives here. The running bit each
//! processor toggles as it enters and leaves the guest, the request bits any
//! processor may set in another's backing page, the logical table a guest
//! reprograms through its own registers, the inhibits that demote one
//! processor or the whole machine back to software delivery, and the
//! transitions that move a control block between the two ways of being.
//!
//! # Every mutable datum has one writer
//!
//! That is what replaces the locks a migrating hypervisor would need:
//!
//! - Entry *i* of the physical table is written only by pCPU *i*, and is
//!   toggled with a read-modify-write: Release on the set that publishes it,
//!   Release on the clear that withdraws it, and an Acquire load is what every
//!   decision made from it reads.
//! - A backing page is written by its own processor alone — by these functions
//!   at transition boundaries, and by the hardware while the guest runs —
//!   except for the interrupt-request words, which any processor may atomically
//!   OR into. The OR is Release: it publishes the request before the doorbell
//!   or the host interrupt that tells the target to look.
//! - The logical table is rebuilt under [`Activation::logical_lock`], held only
//!   in the LDR/DFR handlers and never across an exit: entries move as whole
//!   aligned words, and a processor's old entry is invalidated before its new
//!   one is written.
//! - The inhibits are single atomic booleans, Release on the set.
//!
//! # How the pages and entries are reached
//!
//! Everything below goes through [`Activation::word`] and its siblings, which
//! form references out of the direct map. The frames were allocated once out
//! of the reserved chunk and are never freed or aliased by any other
//! reference — provisioning's byte copies finished before any of this was
//! published — and every concurrent access, this crate's and the hardware's,
//! is an aligned single-copy word access. So an atomic view of a slot is the
//! one kind of view that is always legitimate, and nothing here ever forms a
//! plain reference into one of these frames.

use alloc::boxed::Box;
use core::sync::atomic::{AtomicBool, AtomicU16, AtomicU32, AtomicU64, Ordering};

use apic::REGISTER_STRIDE;
use cpu::ApicId;
use descriptors::Vector;
use log::{info, warn};
use paging::DirectMap;
use spin::{Mutex, Once};
use svm::{
    CleanBits,
    avic::{LogicalApicEntry, PhysicalApicEntry},
};
use vcpu::Vcpu;
use x86_64::PhysAddr;

use crate::{
    VlapicError,
    avic::backing::ResetImage,
    face::{dispatch, table::Register},
    machine::{current, registry},
    priority::{self, Priority},
    registers::{FLAT_DESTINATION_FORMAT, Vlapic, base::Mode, icr::Command, lvt::Entry},
};

/// The bit of a physical-table entry the owning processor toggles.
const IS_RUNNING: u64 = PhysicalApicEntry::new().with_is_running(true).into_bits();

/// The bit of a logical-table entry that says it names a processor.
const LOGICAL_VALID: u32 = LogicalApicEntry::new().with_valid(true).into_bits();

/// The delivery-status bit of the interrupt command's low half.
const COMMAND_BUSY: u32 = 1 << 12;

/// The bit that software-enables a controller, in its spurious-vector
/// register.
const SOFTWARE_ENABLE: u32 = 1 << 8;

/// How many slots each of the three vector banks has.
const BANK_SLOTS: u32 = 8;

/// How many entries the logical table holds: fifteen clusters of four.
const LOGICAL_ENTRIES: usize = 60;

/// The logical-table slot a processor has no entry in.
const NO_SLOT: u16 = u16::MAX;

/// The machine-wide runtime state of hardware-driven delivery.
///
/// Built once by provisioning and never freed. The fields that change say
/// who changes them in their own documentation; everything else was written
/// before the first activation and is read-only after.
struct Activation {
    /// The per-processor backing pages, indexed by roster position; a
    /// processor firmware will not start has none.
    backing: Box<[Option<PhysAddr>]>,
    /// The frame the physical table was written to.
    physical_table: PhysAddr,
    /// The frame the logical table lives in.
    logical_table: PhysAddr,
    /// The largest valid index of the physical table.
    max_index: u16,
    /// The window every access below reaches its frame through.
    window: DirectMap,
    /// Whether the silicon's reading of the running bits is trustworthy.
    ///
    /// Clear on the families erratum #1235 afflicts: the publish step is
    /// skipped there, so the bit is never set and can never be read stale,
    /// and every directed IPI takes the exit-and-kick path instead.
    ipi_virtual: bool,
    /// The machine has been told something is wrong with the acceleration
    /// itself, and stays on the software path for the rest of its life.
    ///
    /// Sticky rather than re-enabled automatically: the conditions that set
    /// it are either broken hypervisor state or a guest the hardware cannot
    /// be trusted to resolve destinations for, and neither announces when it
    /// has gone away.
    machine_inhibited: AtomicBool,
    /// Held while the logical table is rebuilt, and never across an exit.
    logical_lock: Mutex<()>,
    /// The logical-table slot each processor's identity was last published
    /// in, indexed by roster position; [`NO_SLOT`] while it has none.
    ///
    /// Written only by the processor's own pCPU, inside the rebuild.
    logical_slots: Box<[AtomicU16]>,
    /// The reset count each backing page was last rebuilt at, indexed by
    /// roster position: a model reset under an active controller is a page
    /// the hardware must be shown again.
    rebuilt_at: Box<[AtomicU64]>,
}

/// The runtime state, once provisioning has built it.
static ACTIVATED: Once<&'static Activation> = Once::new();

/// Records the structures provisioning just built, for everything that
/// changes about them afterwards.
pub(super) fn establish(
    backing: Box<[Option<PhysAddr>]>,
    physical_table: PhysAddr,
    logical_table: PhysAddr,
    max_index: u16,
    window: DirectMap,
    ipi_virtual: bool,
) {
    let processors = backing.len();
    let state = Box::new(Activation {
        backing,
        physical_table,
        logical_table,
        max_index,
        window,
        ipi_virtual,
        machine_inhibited: AtomicBool::new(false),
        logical_lock: Mutex::new(()),
        logical_slots: (0..processors).map(|_| AtomicU16::new(NO_SLOT)).collect(),
        rebuilt_at: (0..processors).map(|_| AtomicU64::new(0)).collect(),
    });
    ACTIVATED.call_once(|| &*Box::leak(state));
}

/// Whether the structures exist at all, which is whether the policy chose
/// hardware delivery for this machine.
pub(super) fn provisioned() -> bool {
    ACTIVATED.is_completed()
}

/// Whether this processor's controller is being driven in hardware at this
/// moment.
///
/// The question every delivery path asks before taking the hardware's: the
/// structures must exist, the machine must not have been demoted, the guest
/// must be in the face the hardware drives, and the processor must not have
/// been demoted on its own.
pub(crate) fn active_for(vlapic: &Vlapic) -> bool {
    let Some(activation) = ACTIVATED.get() else {
        return false;
    };
    !activation.machine_inhibited.load(Ordering::Acquire)
        && vlapic.mode() == Mode::XApic
        && vlapic.software_enabled()
        && !vlapic.avic_inhibited()
}

/// The backing page of the processor asking.
///
/// # Errors
///
/// [`VlapicError::NotProvisioned`] before provisioning, or
/// [`VlapicError::NoLapic`] if the processor asking has no page.
pub(crate) fn own_page() -> Result<PhysAddr, VlapicError> {
    let activation = ACTIVATED.get().ok_or(VlapicError::NotProvisioned)?;
    let vlapic = current()?;
    activation.page(vlapic.index().get())
}

/// Brings the control block's acceleration into agreement with the guest it
/// describes, on the way in.
///
/// Cheap in steady state: a comparison, and nothing else. Where the guest's
/// face and the control block's bit disagree, the transition is performed
/// here — the backing page rebuilt from the model on the way up, the
/// hardware's state carried back into the model on the way down — because
/// the rule every transition keeps is that the software model moves first
/// and the acceleration follows it.
///
/// # Errors
///
/// [`VlapicError::NotProvisioned`] is answered as "no acceleration" rather
/// than reported: a machine the policy left on the software path reaches
/// here on every entry, and that is not an error. Anything else names a
/// frame the window does not reach or a processor the roster does not
/// describe, and the caller degrades rather than refusing the entry.
pub(crate) fn reconcile(vcpu: &mut Vcpu) -> Result<(), VlapicError> {
    let Some(activation) = ACTIVATED.get() else {
        return Ok(());
    };
    let vlapic = current()?;
    let index = vlapic.index().get();
    let want = active_for(vlapic);
    let have = vcpu.control().interrupt_control.avic_enable();
    match (have, want) {
        (true, false) => deactivate(activation, vcpu, vlapic),
        (false, true) => activate(activation, vcpu, vlapic),
        // The steady state. One thing can still have moved: a reset rebuilt
        // the model underneath an active controller, and the page the
        // hardware serves must be rebuilt after it.
        (true, true) => {
            if activation.rebuilt_at[index].load(Ordering::Relaxed) != vlapic.epoch() {
                rebuild_backing(vlapic)?;
                activation.rebuilt_at[index].store(vlapic.epoch(), Ordering::Relaxed);
            }
            // The task priority the guest sets without exiting has to be the
            // model's before anything consults the model about what is
            // deliverable.
            sync_task_priority(activation, vlapic)?;
            Ok(())
        }
        (false, false) => Ok(()),
    }
}

/// Turns the acceleration on for this processor, at the entry that asked.
fn activate(activation: &Activation, vcpu: &mut Vcpu, vlapic: &Vlapic) -> Result<(), VlapicError> {
    rebuild_backing(vlapic)?;
    activation.rebuilt_at[vlapic.index().get()].store(vlapic.epoch(), Ordering::Relaxed);
    let control = vcpu.control_mut();
    control.interrupt_control = control.interrupt_control.with_avic_enable(true);
    // The enable bit is one clean group and the pointers beside it are
    // another, and both were touched by the life the guest lived since the
    // acceleration was last on.
    vcpu.soil(CleanBits::INTERRUPT.union(CleanBits::AVIC));
    // The processor may have cached a translation of the register page from
    // before the redirection existed; nothing but a flush gets rid of it.
    vcpu.flush();
    info!(
        "vlapic: {} turned hardware delivery on for its guest",
        vlapic.index()
    );
    Ok(())
}

/// Turns the acceleration off for this processor, carrying the hardware's
/// state back into the model first.
fn deactivate(
    activation: &Activation,
    vcpu: &mut Vcpu,
    vlapic: &Vlapic,
) -> Result<(), VlapicError> {
    sync_into_model(activation, vlapic)?;
    let control = vcpu.control_mut();
    control.interrupt_control = control.interrupt_control.with_avic_enable(false);
    vcpu.soil(CleanBits::INTERRUPT.union(CleanBits::AVIC));
    vcpu.flush();
    // Defensive: the unpublish belongs to the exit and the park boundaries,
    // and a demotion arriving anywhere else must not leave the bit behind.
    let _ = unpublish_running();
    info!(
        "vlapic: {} turned hardware delivery off for its guest",
        vlapic.index()
    );
    Ok(())
}

/// Says this processor is in the guest, at the entry boundary.
///
/// The running bit is what another processor's IPI resolution reads before
/// it decides between the doorbell and the exit, so the publish happens
/// after everything the entry prepared and before the guest is entered; a
/// request that lands between the two is one the entry's own look, or the
/// VMRUN's re-evaluation, still finds.
///
/// Nothing at all on the erratum families: the bit stays clear for the life
/// of the machine there, and every directed IPI takes the exit it then
/// cannot avoid.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub(crate) fn publish_running() -> Result<(), VlapicError> {
    let activation = ACTIVATED.get().ok_or(VlapicError::NotProvisioned)?;
    if !activation.ipi_virtual {
        return Ok(());
    }
    let vlapic = current()?;
    activation
        .entry(vlapic.apic_id())?
        .fetch_or(IS_RUNNING, Ordering::Release);
    Ok(())
}

/// Says this processor is no longer in the guest, at the exit boundary.
///
/// The clear precedes every consultation of what has been left for it: a
/// sender that still reads the bit set rings a doorbell the processor
/// answers in host code, which is harmless; a sender that reads it clear
/// takes the kick path, and the rescan after this is what finds whatever
/// the kick is for.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub(crate) fn unpublish_running() -> Result<(), VlapicError> {
    let activation = ACTIVATED.get().ok_or(VlapicError::NotProvisioned)?;
    if !activation.ipi_virtual {
        return Ok(());
    }
    let vlapic = current()?;
    activation
        .entry(vlapic.apic_id())?
        .fetch_and(!IS_RUNNING, Ordering::Release);
    Ok(())
}

/// Whether the processor `id` is in the guest at this instant.
///
/// An Acquire load, pairing with the publisher's Release: whatever the
/// publisher stored before withdrawing is visible to whatever this decides.
/// A processor the table does not hold is not running anywhere.
///
/// # Errors
///
/// [`VlapicError::NotProvisioned`] before provisioning, or
/// [`VlapicError::Paging`] if the window does not reach the table — neither
/// of which a caller treats as running.
pub(crate) fn is_running(id: ApicId) -> Result<bool, VlapicError> {
    let activation = ACTIVATED.get().ok_or(VlapicError::NotProvisioned)?;
    Ok(activation.entry(id)?.load(Ordering::Acquire) & IS_RUNNING != 0)
}

/// Sets a vector in this processor's backing request bits: the device path's
/// delivery under hardware-driven mode.
///
/// The OR is Release, so the request is published before the doorbell or the
/// host interrupt that follows it, and idempotent, so an arrival that races
/// itself coalesces exactly as the software path's does. Answers whether
/// the bit was newly set.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub(crate) fn request(vector: Vector) -> Result<bool, VlapicError> {
    let activation = ACTIVATED.get().ok_or(VlapicError::NotProvisioned)?;
    let vlapic = current()?;
    let page = activation.page(vlapic.index().get())?;
    let word = activation.word(page, bank_offset(Register::INTERRUPT_REQUEST, vector))?;
    let bit = 1u32 << (vector.number() % 32);
    Ok(word.fetch_or(bit, Ordering::Release) & bit == 0)
}

/// Whether this processor's backing page holds anything its guest could take
/// at this instant.
///
/// The second half of the park protocol: asked after the running bit was
/// withdrawn and before the processor parks, with Acquire loads so that a
/// request published before the withdrawal is one this scan sees. The
/// comparison is the architecture's own — the highest request against the
/// processor priority the backing TPR and in-service state impose — and a
/// software-disabled controller takes nothing, here as in the model.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub(crate) fn deliverable() -> Result<bool, VlapicError> {
    let activation = ACTIVATED.get().ok_or(VlapicError::NotProvisioned)?;
    let vlapic = current()?;
    if !active_for(vlapic) {
        return Ok(false);
    }
    let page = activation.page(vlapic.index().get())?;
    let spurious = activation
        .word(page, Register::SPURIOUS.offset())?
        .load(Ordering::Acquire);
    if spurious & SOFTWARE_ENABLE == 0 {
        return Ok(false);
    }
    let task = activation
        .word(page, Register::TASK_PRIORITY.offset())?
        .load(Ordering::Acquire);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "only the low byte of the task priority register carries meaning"
    )]
    let task = task as u8;
    let servicing = activation.highest(page, Register::IN_SERVICE, Ordering::Acquire)?;
    let processor = priority::processor_priority(Priority::new(task), servicing);
    Ok(activation
        .highest(page, Register::INTERRUPT_REQUEST, Ordering::Acquire)?
        .is_some_and(|vector| priority::deliverable(vector, processor)))
}

/// Completes an interrupt a guest-to-guest IPI could not be: the command is
/// given to the software path end to end, and the delivery-status bit the
/// hardware may have left in the backing command register is cleared so the
/// guest does not wait on it.
///
/// The command comes from the exit rather than from the register: the
/// register is the backing page's now, and the exit carries what the guest
/// wrote.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub(crate) fn complete_command(command_bits: u64) -> Result<(), VlapicError> {
    let vlapic = current()?;
    let page = registry::lapics()?;
    crate::delivery::send(vlapic, page.all(), Command::from_bits(command_bits));
    clear_command_busy(vlapic)
}

/// What the guest wrote to a register the hardware completed into the
/// backing page before it exited: the value is read back out of the page and
/// given the same meaning the software path would have given it.
///
/// The guest's own write already landed — that is what a trap is — so what
/// runs here is only the bookkeeping a register write asks for beyond the
/// store: the sources reprogrammed by an LVT write, the real hardware quieted
/// by a disable, the logical table moved by an LDR.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub(crate) fn trap_write(register: Register, exit_vector: Option<u8>) -> Result<(), VlapicError> {
    let vlapic = current()?;
    match register {
        // The one trap whose value is not in the page: the hardware retired
        // the interrupt before it exited, and what is owed is the release of
        // whatever real hardware is holding for it.
        Register::END_OF_INTERRUPT => end_of_interrupt(vlapic, exit_vector),
        // An IPI the hardware attempted is completed by the exit it raised;
        // what can be left behind is only the delivery-status bit.
        Register::COMMAND_LOW => clear_command_busy(vlapic),
        Register::ERROR_STATUS => {
            // The write latched the page's copy clear, as it latches the
            // model's: both are what the guest reads back, depending on
            // which of them is serving the register at the time.
            dispatch::acted(vlapic, dispatch::write(vlapic, register, 0));
            let activation = ACTIVATED.get().ok_or(VlapicError::NotProvisioned)?;
            let page = activation.page(vlapic.index().get())?;
            activation
                .word(page, register.offset())?
                .store(0, Ordering::Release);
            Ok(())
        }
        other => {
            let activation = ACTIVATED.get().ok_or(VlapicError::NotProvisioned)?;
            let page = activation.page(vlapic.index().get())?;
            let value = activation
                .word(page, other.offset())?
                .load(Ordering::Acquire);
            dispatch::acted(vlapic, dispatch::write(vlapic, other, value));
            if matches!(
                other,
                Register::LOGICAL_DESTINATION | Register::DESTINATION_FORMAT
            ) {
                mirror_logical(vlapic)?;
            }
            Ok(())
        }
    }
}

/// Reads one of the registers the hardware owns out of this processor's
/// backing page: what the faces answer a guest's access with while the page,
/// and not the model, is where that register lives.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub(crate) fn backing_read(register: Register) -> Result<u32, VlapicError> {
    let activation = ACTIVATED.get().ok_or(VlapicError::NotProvisioned)?;
    let vlapic = current()?;
    let page = activation.page(vlapic.index().get())?;
    Ok(activation
        .word(page, register.offset())?
        .load(Ordering::Acquire))
}

/// Stores the guest's task priority where the hardware reads it while the
/// acceleration is on.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub(crate) fn backing_write_task_priority(value: u32) -> Result<(), VlapicError> {
    let activation = ACTIVATED.get().ok_or(VlapicError::NotProvisioned)?;
    let vlapic = current()?;
    let page = activation.page(vlapic.index().get())?;
    activation
        .word(page, Register::TASK_PRIORITY.offset())?
        .store(value, Ordering::Release);
    Ok(())
}

/// Demotes the whole machine, for the rest of its life.
///
/// For the reports that say the acceleration itself cannot be trusted:
/// destinations resolving somewhere the guest did not name them, or an exit
/// the architecture has no reason to raise.
pub(crate) fn inhibit_machine(reason: &str) {
    let Some(activation) = ACTIVATED.get() else {
        return;
    };
    if !activation.machine_inhibited.swap(true, Ordering::AcqRel) {
        warn!("vlapic: hardware delivery inhibited for the machine: {reason}");
    }
}

/// How many hardware doorbells this machine has rung, for the exit census.
pub(crate) fn doorbell_count() -> u64 {
    crate::delivery::avic::rings()
}

/// How many host-interrupt kicks the AVIC paths have sent, for the exit
/// census.
pub(crate) fn kick_count() -> u64 {
    crate::delivery::avic::kicks()
}

impl Activation {
    /// The backing page of the processor at roster position `index`.
    fn page(&self, index: usize) -> Result<PhysAddr, VlapicError> {
        self.backing
            .get(index)
            .copied()
            .flatten()
            .ok_or(VlapicError::NoLapic)
    }

    /// One register slot of a backing page, as an atomic word.
    fn word(&self, page: PhysAddr, offset: u32) -> Result<&AtomicU32, VlapicError> {
        let at = PhysAddr::new(page.as_u64() + u64::from(offset));
        let pointer = self.window.ptr::<AtomicU32>(at)?;
        // SAFETY: the frame is host RAM out of the reserved chunk, allocated
        // for this page and never freed, and no other code holds a reference
        // into it — provisioning's byte copies finished before this state was
        // published, and every access since is one of these atomics or the
        // hardware's own aligned word accesses. The reference is valid for as
        // long as the machine runs, which is as long as it is used.
        Ok(unsafe { pointer.as_ref() })
    }

    /// The physical-table entry of the processor `id`, as an atomic word.
    fn entry(&self, id: ApicId) -> Result<&AtomicU64, VlapicError> {
        let index = id.get();
        if index > u32::from(self.max_index) {
            return Err(VlapicError::IdBeyondTable {
                id: index,
                max_index: self.max_index,
            });
        }
        let at = PhysAddr::new(
            self.physical_table.as_u64() + u64::from(index) * size_of::<PhysicalApicEntry>() as u64,
        );
        let pointer = self.window.ptr::<AtomicU64>(at)?;
        // SAFETY: as [`Activation::word`]: one frame, reserved and never
        // freed, reached only through these atomics.
        Ok(unsafe { pointer.as_ref() })
    }

    /// One entry of the logical table, as an atomic word.
    fn logical(&self, slot: usize) -> Result<&AtomicU32, VlapicError> {
        let at = PhysAddr::new(
            self.logical_table.as_u64() + (slot * size_of::<LogicalApicEntry>()) as u64,
        );
        let pointer = self.window.ptr::<AtomicU32>(at)?;
        // SAFETY: as [`Activation::word`].
        Ok(unsafe { pointer.as_ref() })
    }

    /// The highest vector with a bit set in one of a page's three banks.
    fn highest(
        &self,
        page: PhysAddr,
        bank: Register,
        ordering: Ordering,
    ) -> Result<Option<Vector>, VlapicError> {
        for slot in (0..BANK_SLOTS).rev() {
            let bits = self
                .word(page, bank.offset() + slot * REGISTER_STRIDE)?
                .load(ordering);
            if bits != 0 {
                let number = slot * u32::BITS + (u32::BITS - 1 - bits.leading_zeros());
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "eight slots of thirty-two bits is exactly the vector space"
                )]
                let vector = Vector::new(number as u8);
                return Ok(Some(vector));
            }
        }
        Ok(None)
    }
}

/// Rebuilds this processor's backing page out of the model.
///
/// The reset image, with the registers the architecture preserves written
/// over it from the live model — the spurious vector and its enable bit, the
/// destination registers, and whichever local vector entries the controller
/// behind the model has. The guest's own writes reach the page directly
/// afterwards; this is only what has to be there before the first of them.
fn rebuild_backing(vlapic: &Vlapic) -> Result<(), VlapicError> {
    let activation = ACTIVATED.get().ok_or(VlapicError::NotProvisioned)?;
    let page = activation.page(vlapic.index().get())?;
    let mut image = ResetImage::new(vlapic.apic_id(), vlapic.version());
    image.overlay(Register::SPURIOUS, vlapic.spurious());
    image.overlay(
        Register::LOGICAL_DESTINATION,
        vlapic.logical_destination(Mode::XApic),
    );
    image.overlay(Register::DESTINATION_FORMAT, vlapic.destination_format());
    for entry in Entry::ALL {
        if vlapic.model().has(entry) {
            image.overlay(entry.register(), vlapic.lvt_readback(entry).into_bits());
        }
    }
    // SAFETY: the frame is this processor's backing page, host RAM out of
    // the reserved chunk: no device aperture, and nothing holds a reference
    // to it while the guest is not running — which it is not, since this
    // runs at an entry boundary.
    unsafe { activation.window.write(page, image.bytes())? };
    mirror_logical(vlapic)?;
    Ok(())
}

/// Carries the hardware's state back into the model.
///
/// The direction a deactivation goes: the task priority the guest set
/// without exiting, and whatever the hardware accepted, took into service
/// and recorded as level, so that software-only delivery continues from
/// exactly where the hardware left it. Runs at an exit boundary with the
/// guest stopped, so the loads need no ordering stronger than the exit's
/// own serialization.
fn sync_into_model(activation: &Activation, vlapic: &Vlapic) -> Result<(), VlapicError> {
    let page = activation.page(vlapic.index().get())?;
    sync_task_priority(activation, vlapic)?;
    for slot in 0..BANK_SLOTS {
        let offset = |bank: Register| bank.offset() + slot * REGISTER_STRIDE;
        let irr = activation
            .word(page, offset(Register::INTERRUPT_REQUEST))?
            .load(Ordering::Relaxed);
        let isr = activation
            .word(page, offset(Register::IN_SERVICE))?
            .load(Ordering::Relaxed);
        let tmr = activation
            .word(page, offset(Register::TRIGGER_MODE))?
            .load(Ordering::Relaxed);
        for bit in 0..u32::BITS {
            let mask = 1 << bit;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "eight slots of thirty-two bits is exactly the vector space"
            )]
            let vector = Vector::new((slot * u32::BITS + bit) as u8);
            if tmr & mask != 0 {
                vlapic.force_trigger_mode(vector);
            }
            if isr & mask != 0 {
                vlapic.force_in_service(vector);
            }
            // A request the model already holds in service is a level
            // arrival the software path is already tracking: requesting it
            // again would deliver it twice.
            if irr & mask != 0 && !vlapic.holds_in_service(vector) {
                vlapic.force_request(vector);
            }
        }
    }
    Ok(())
}

/// Copies the backing task priority into the model.
fn sync_task_priority(activation: &Activation, vlapic: &Vlapic) -> Result<(), VlapicError> {
    let page = activation.page(vlapic.index().get())?;
    let value = activation
        .word(page, Register::TASK_PRIORITY.offset())?
        .load(Ordering::Relaxed);
    vlapic.set_task_priority(value);
    Ok(())
}

/// Clears the delivery-status bit of the backing command register.
///
/// Whatever the hardware left it set for, the software completion is the
/// completion, and a guest watching the bit would otherwise wait on a
/// delivery nothing is performing.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub(crate) fn clear_command_busy(vlapic: &Vlapic) -> Result<(), VlapicError> {
    let activation = ACTIVATED.get().ok_or(VlapicError::NotProvisioned)?;
    let page = activation.page(vlapic.index().get())?;
    activation
        .word(page, Register::COMMAND_LOW.offset())?
        .fetch_and(!COMMAND_BUSY, Ordering::Release);
    Ok(())
}

/// Completes the host's half of an EOI the hardware trapped.
///
/// The hardware raised the exit because the interrupt being acknowledged is
/// level triggered, and what a level acknowledgement owes is the release of
/// whatever real hardware is holding for the vector. The vector comes from
/// the backing in-service bank where the delivery recorded it, with the
/// exit's own report consulted where the two disagree — see [`eoi_vector`].
///
/// The in-service bit itself is cleared idempotently: whichever of the
/// hardware and this cleared it first, the second clear is a no-op, and
/// nothing here retires anything twice.
fn end_of_interrupt(vlapic: &Vlapic, exit_vector: Option<u8>) -> Result<(), VlapicError> {
    let activation = ACTIVATED.get().ok_or(VlapicError::NotProvisioned)?;
    let page = activation.page(vlapic.index().get())?;
    let top = activation.highest(page, Register::IN_SERVICE, Ordering::Acquire)?;
    let Some(vector) = eoi_vector(top, exit_vector) else {
        // An EOI with nothing in service is the architectural no-op, and a
        // guest is entitled to make one.
        return Ok(());
    };
    if top == Some(vector) {
        let word = activation.word(page, bank_offset(Register::IN_SERVICE, vector))?;
        word.fetch_and(!(1u32 << (vector.number() % 32)), Ordering::AcqRel);
    }
    let local = apic::local()?;
    vlapic.ledger().release(vector, &local);
    Ok(())
}

/// Which vector an EOI retired, given the in-service bank and the exit's
/// own report.
///
/// The two agree wherever the hardware exited before it retired the
/// interrupt: the bank's top is the vector. Where they disagree, the
/// hardware had already retired the top — cleared the bit — before the exit,
/// and the report is the only place the vector still is. A report of nothing
/// usable falls back to the bank whatever it holds.
fn eoi_vector(in_service_top: Option<Vector>, exit_vector: Option<u8>) -> Option<Vector> {
    let reported = exit_vector
        .map(Vector::new)
        .filter(|vector| priority::legal(*vector));
    match reported {
        // A report that disagrees with the bank is the hardware having
        // retired the bank's top before it exited; a report where the bank
        // holds nothing is the only place the vector is.
        Some(reported) if in_service_top != Some(reported) => Some(reported),
        // The two agree, or there is no usable report: the bank's top is the
        // answer, which is nothing at all for an empty bank.
        _ => in_service_top,
    }
}

/// Publishes this processor's logical identity in the table, withdrawing
/// whatever its previous identity was first, and demotes the machine if two
/// processors now claim one identity.
///
/// Called under [`Activation::logical_lock`] semantics — the lock is taken
/// here — and whole aligned words are what move, so a hardware resolution
/// racing the rebuild sees either the old entry or the new one, and either
/// answer is a delivery, or an exit the handlers complete.
fn mirror_logical(vlapic: &Vlapic) -> Result<(), VlapicError> {
    let activation = ACTIVATED.get().ok_or(VlapicError::NotProvisioned)?;
    let index = vlapic.index().get();
    let guard = activation.logical_lock.lock();
    // Withdraw the old entry before publishing a new one, so the window in
    // between names the processor nowhere rather than twice.
    let previous = activation.logical_slots[index].swap(NO_SLOT, Ordering::Relaxed);
    if previous != NO_SLOT {
        let entry = activation.logical(usize::from(previous))?;
        entry.fetch_and(!LOGICAL_VALID, Ordering::Relaxed);
    }
    let wanted = logical_slot(
        vlapic.logical_destination(Mode::XApic),
        vlapic.destination_format(),
    );
    if let Some(slot) = wanted {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the logical table's identity field is eight bits wide, and xAVIC identifiers are"
        )]
        let entry = LogicalApicEntry::new()
            .with_guest_apic_id(vlapic.apic_id().get() as u8)
            .with_valid(true);
        activation
            .logical(slot)?
            .store(entry.into_bits(), Ordering::Relaxed);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a slot is below sixty, which sixteen bits hold with room to spare"
        )]
        let stored = slot as u16;
        activation.logical_slots[index].store(stored, Ordering::Relaxed);
    }
    // Two processors claiming one logical identity is a resolution the
    // hardware cannot be trusted with: the machine steps back to software
    // delivery, where destinations are matched against each controller's own
    // state.
    let aliased = {
        let mut seen = [false; LOGICAL_ENTRIES];
        activation.logical_slots.iter().any(|slot| {
            let claimed = slot.load(Ordering::Relaxed);
            claimed != NO_SLOT && {
                let taken = seen[usize::from(claimed)];
                seen[usize::from(claimed)] = true;
                taken
            }
        })
    };
    drop(guard);
    if aliased {
        inhibit_machine("two of the guest's processors claim one logical identity");
    }
    Ok(())
}

/// The logical-table slot a logical destination names in a destination
/// format, if it names one at all.
///
/// In the flat model the identifier byte is a bit per processor and must be
/// one bit exactly; in the cluster model the top nibble is the cluster — of
/// which the table holds fifteen — and the bottom one must be a bit exactly.
/// Every other format is one the architecture leaves undefined, and a table
/// entry for it would be a destination the guest never asked for.
fn logical_slot(ldr: u32, dfr: u32) -> Option<usize> {
    let identifier = ldr >> 24;
    if dfr == FLAT_DESTINATION_FORMAT {
        return (identifier != 0 && identifier.is_power_of_two())
            .then(|| identifier.trailing_zeros() as usize);
    }
    let cluster = (identifier >> 4) as usize;
    let members = identifier & 0x0F;
    (cluster < LOGICAL_ENTRIES / 4 && members != 0 && members.is_power_of_two())
        .then(|| cluster * 4 + members.trailing_zeros() as usize)
}

/// The offset of the bank slot a vector sits in.
const fn bank_offset(bank: Register, vector: Vector) -> u32 {
    bank.offset() + (vector.number() as u32 / u32::BITS) * REGISTER_STRIDE
}

#[cfg(test)]
mod tests {
    //! The decisions here that need no machine: which vector an EOI retired,
    //! and which logical destinations name a table entry.

    use descriptors::Vector;

    use super::{eoi_vector, logical_slot};

    #[test]
    fn an_eoi_retires_the_in_service_top_when_the_exit_agrees_or_says_nothing() {
        let top = Some(Vector::new(0x42));
        assert_eq!(eoi_vector(top, Some(0x42)), top);
        assert_eq!(eoi_vector(top, None), top);
        // A report of an illegal vector is a report of nothing.
        assert_eq!(eoi_vector(top, Some(0x05)), top);
        assert_eq!(eoi_vector(None, None), None);
    }

    #[test]
    fn an_eoi_whose_report_disagrees_retires_the_report() {
        // The bank's top has moved on, which is the hardware having retired
        // the interrupt before it exited; the exit's own word is the only
        // place the vector still is.
        assert_eq!(
            eoi_vector(Some(Vector::new(0x31)), Some(0x42)),
            Some(Vector::new(0x42))
        );
        assert_eq!(eoi_vector(None, Some(0x42)), Some(Vector::new(0x42)));
    }

    #[test]
    fn a_flat_destination_is_one_bit_per_entry() {
        assert_eq!(logical_slot(0x0100_0000, 0xFFFF_FFFF), Some(0));
        assert_eq!(logical_slot(0x8000_0000, 0xFFFF_FFFF), Some(7));
        // Two bits, and no bits, name nothing.
        assert_eq!(logical_slot(0x0300_0000, 0xFFFF_FFFF), None);
        assert_eq!(logical_slot(0, 0xFFFF_FFFF), None);
    }

    #[test]
    fn a_cluster_destination_is_an_address_and_one_bit() {
        // Cluster 2, member bit 0: slot 8.
        assert_eq!(logical_slot(0x2100_0000, 0x0FFF_FFFF), Some(8));
        // Cluster 14 is the last the table holds.
        assert_eq!(logical_slot(0xE100_0000, 0x0FFF_FFFF), Some(56));
        assert_eq!(logical_slot(0xF100_0000, 0x0FFF_FFFF), None);
        // Two members, and none, name nothing.
        assert_eq!(logical_slot(0x2300_0000, 0x0FFF_FFFF), None);
        assert_eq!(logical_slot(0x2000_0000, 0x0FFF_FFFF), None);
    }

    #[test]
    fn a_format_that_is_not_flat_follows_the_cluster_model() {
        // The flat encoding is the only one that is flat: every other value
        // of the format register is matched the cluster way, which is the
        // convention the kernel's AVIC uses and the one the plan follows.
        assert_eq!(logical_slot(0x2100_0000, 0x1FFF_FFFF), Some(8));
        assert_eq!(logical_slot(0x2100_0000, 0xEFFF_FFFF), Some(8));
    }
}
