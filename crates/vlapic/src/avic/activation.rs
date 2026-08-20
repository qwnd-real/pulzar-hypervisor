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
//!   at transition boundaries and at the exits its own guest's register writes
//!   raise, and by the hardware while the guest runs — except for the
//!   interrupt-request words, which any processor may atomically OR into. The
//!   OR is Release: it publishes the request before the doorbell or the host
//!   interrupt that tells the target to look.
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
use core::{
    cmp::min,
    iter::once,
    sync::atomic::{AtomicBool, AtomicU16, AtomicU32, AtomicU64, Ordering},
};

use apic::REGISTER_STRIDE;
use cpu::ApicId;
use descriptors::Vector;
use log::{info, warn};
use paging::DirectMap;
use spin::{Mutex, Once};
use svm::{
    CleanBits,
    avic::{LogicalApicEntry, MAX_PHYSICAL_ID, PhysicalApicEntry},
};
use vcpu::Vcpu;
use x86_64::PhysAddr;

use crate::{
    VlapicError,
    avic::backing::{Life, Projection, ResetImage},
    delivery::error,
    face::{dispatch, dispatch::Written, table::Register},
    machine::{current, registry},
    priority::{self, Priority},
    registers::{
        FLAT_DESTINATION_FORMAT, Vlapic, base::Mode, error::Errors, icr::Command, lvt::Entry,
    },
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
    ///
    /// What the table was built for, which is the widest face the machine may
    /// drive it in; how far the hardware is told to walk it is narrower
    /// wherever the face being driven is — see [`face_limit`].
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

/// Whether the wider controller face may be used at all, recorded when the
/// controllers are built.
static X2APIC_PERMITTED: Once<bool> = Once::new();

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
pub(crate) fn provisioned() -> bool {
    ACTIVATED.is_completed()
}

/// Records whether a guest on this machine may be given the controller face
/// its identifiers are reached through in model-specific registers.
///
/// The boot-time policy's answer, and it has to be taken before the first
/// controller is built rather than with the structures the acceleration runs
/// on: a controller seeded from firmware's own register can already *be* in
/// that face, and whether it may stay there is this question — asked at a
/// moment when nothing has been provisioned yet.
pub(crate) fn permit_x2apic(permitted: bool) {
    X2APIC_PERMITTED.call_once(|| permitted);
}

/// Whether a guest may be given the controller face its identifiers are
/// reached through in model-specific registers.
///
/// The one statement of it, consulted wherever the answer can be observed: the
/// state a controller is seeded into, the transition a guest writes, and the
/// feature bit `CPUID` reports. True until [`permit_x2apic`] says otherwise,
/// which is a window with no controller in it at all — building the controllers
/// is what records the answer.
pub(crate) fn x2avic_permitted() -> bool {
    X2APIC_PERMITTED.get().copied().unwrap_or(true)
}

/// Whether this processor's controller is being driven in hardware at this
/// moment.
///
/// The question every delivery path asks before taking the hardware's, and
/// [`active_face`] is the whole of it: the structures must exist, neither
/// demotion may have fired, and the face the guest is in must be one the
/// acceleration drives.
///
/// # The software-enable bit is not one of the terms
///
/// A controller its guest has software-disabled goes on being driven by the
/// hardware, and that is deliberate. The bit gates *delivery* rather than
/// access: the hardware evaluates it out of the backing page itself, which is
/// where [`deliverable`] asks it and the only place it is asked. Registers of a
/// software-disabled controller are architecturally still readable and
/// writable, and the backing page is exactly where they should be read and
/// written.
///
/// Making it a term here would be worse than redundant on a machine the policy
/// provisioned. Reset leaves it clear, so every processor the guest starts
/// would come up unaccelerated — and the register page a controller then falls
/// back to is not the emulator's, because the acceleration's redirection is
/// arranged once before the guest runs and the page translates to a frame
/// nothing reads for the rest of the machine's life. The guest's write of the
/// very register that would switch the controller on would land there and be
/// lost, and no processor but the first would ever have an interrupt
/// controller.
pub(crate) fn active_for(vlapic: &Vlapic) -> bool {
    active_face(vlapic).is_some()
}

/// The face the acceleration drives this controller in at this instant, or
/// nothing where it does not drive it at all.
///
/// One derivation of what used to be two: whether the acceleration is on for
/// this processor, and which face it is on in. A caller that asked the first
/// and then computed the second was deriving one decision twice out of state
/// that can move between the two questions.
fn active_face(vlapic: &Vlapic) -> Option<Face> {
    let activation = ACTIVATED.get()?;
    driven_face(
        activation.machine_inhibited.load(Ordering::Acquire),
        vlapic.mode(),
        x2avic_permitted(),
        vlapic.avic_inhibited(),
    )
}

/// The face hardware delivery drives a controller in, out of the state the
/// decision is made from.
///
/// Pure rather than four loads inside [`active_face`], because which terms it
/// has *is* the activation gate, and a decision of values can be read against
/// the states it has to cover.
///
/// The older face is what a provisioned machine was built for wherever it was
/// built at all; the wider one is driven only where the policy says it can be,
/// which is what [`x2avic_permitted`] answers. A controller its
/// guest has globally disabled is driven in neither: with the enable bit of the
/// base register clear there is no controller to drive, and the guest's only
/// way back is that register, which is a model-specific register this
/// hypervisor never stops intercepting.
const fn driven_face(
    machine_inhibited: bool,
    mode: Mode,
    x2avic: bool,
    inhibited: bool,
) -> Option<Face> {
    if machine_inhibited || inhibited {
        return None;
    }
    match mode {
        Mode::XApic => Some(Face::XAvic),
        Mode::X2Apic if x2avic => Some(Face::X2Avic),
        Mode::X2Apic | Mode::Disabled => None,
    }
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

/// The face the control block's acceleration drives, which is the whole of
/// what its two enable bits mean.
///
/// The bits and the face are one decision rather than two because the
/// architecture defines no state with the wider bit set and the narrower one
/// clear, and the transitions below never produce one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Face {
    /// Eight-bit identifiers, reached through the register page.
    XAvic,
    /// Thirty-two-bit identifiers, reached through model-specific registers.
    X2Avic,
}

/// What an entry owes the acceleration, given the face the control block
/// carries and the face the guest is in.
///
/// Pure rather than folded into [`reconcile`], because the decision is the
/// part of the transition that must not be wrong, and a decision of values
/// is the kind of thing that can be read against the table it implements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Move {
    /// Both agree the software delivers; nothing is owed.
    Idle,
    /// Both agree the hardware delivers, in the same face; only the
    /// steady-state upkeep is owed.
    Steady,
    /// The acceleration is turned on, in this face.
    Enable(Face),
    /// The acceleration is turned off; it was on in this face.
    Disable(Face),
    /// The acceleration stays on and changes face.
    Switch { from: Face, to: Face },
}

/// The move an entry owes, as a table rather than a chain of comparisons.
fn move_for(have: Option<Face>, want: Option<Face>) -> Move {
    match (have, want) {
        (None, None) => Move::Idle,
        (None, Some(face)) => Move::Enable(face),
        (Some(face), None) => Move::Disable(face),
        (Some(from), Some(to)) if from != to => Move::Switch { from, to },
        (Some(_), Some(_)) => Move::Steady,
    }
}

/// The highest table index the hardware can address while it drives a
/// controller in this face, out of the highest one the machine provisioned.
///
/// A property of the face rather than of the machine, which is why it is asked
/// again at every transition instead of once at provisioning. The older face
/// resolves a destination out of an eight-bit field whose all-ones encoding
/// means "every processor", so it reaches no further than [`MAX_PHYSICAL_ID`]
/// however many processors the machine has — and a control block entered in
/// that face over a wider table is one the processor refuses outright, with no
/// guest instruction executed.
///
/// Above the limit the guest's processors are unaddressable while it is in that
/// face, which is the eight-bit field's own behaviour and what the warning
/// [`crate::install`] says out loud on such a machine is about. The wider face
/// is what the table was sized for wherever the machine has it, so there its
/// own extent is the answer.
fn face_limit(face: Face, provisioned: u16) -> u16 {
    match face {
        Face::XAvic => min(provisioned, MAX_PHYSICAL_ID),
        Face::X2Avic => provisioned,
    }
}

/// Brings the control block's acceleration into agreement with the guest it
/// describes, on the way in.
///
/// Cheap in steady state: a comparison, and nothing else. Where the guest's
/// face and the control block's bits disagree, the transition is performed
/// here — the backing page rebuilt from the model on the way up, the
/// hardware's state carried back into the model on the way down, the
/// permission map's pass-through granted before the wider face's enable bit
/// and withdrawn after it, and the extent the table may be walked to published
/// with the bit that selects the face it belongs to — because the rule every
/// transition keeps is that the software model moves first and the acceleration
/// follows it.
///
/// # Errors
///
/// [`VlapicError::NotProvisioned`] is answered as "no acceleration" rather
/// than reported: a machine the policy left on the software path reaches
/// here on every entry, and that is not an error.
/// [`VlapicError::AvicRefused`] is a transition the entry rules would not have
/// survived, reported with the acceleration left exactly as it was. Anything
/// else names a frame the window does not reach or a processor the roster does
/// not describe. In every case the caller degrades rather than refusing the
/// entry, and what the guest is entered with is what the control block says
/// rather than what the model asked for.
pub(crate) fn reconcile(vcpu: &mut Vcpu) -> Result<(), VlapicError> {
    let Some(activation) = ACTIVATED.get() else {
        return Ok(());
    };
    let vlapic = current()?;
    let have = face_of(vcpu);
    let want = active_face(vlapic);
    // One reading above the match rather than one inside an arm, because which
    // life the backing page belongs to is a term of every transition and not of
    // one: it decides whether a deactivation may carry the page into the model,
    // and whether the steady state has a page to rebuild at all. Taken once, so
    // that two arms cannot straddle a reset and disagree about it.
    let standing = Standing::of(
        activation.rebuilt_at[vlapic.index().get()].load(Ordering::Relaxed),
        vlapic.epoch(),
    );
    match move_for(have, want) {
        Move::Idle => Ok(()),
        Move::Enable(face) => enable(activation, vcpu, vlapic, face, standing),
        Move::Disable(face) => disable(activation, vcpu, vlapic, face, standing),
        Move::Switch { from, to } => switch(activation, vcpu, vlapic, from, to),
        // The steady state. One thing can still have moved: a reset cleared the
        // model underneath an active controller, and the page the hardware
        // serves holds the state that reset was required to destroy.
        Move::Steady => {
            if !standing.life.carried() {
                rebuild_backing(activation, vlapic, standing)?;
            }
            // The task priority the guest sets without exiting has to be the
            // model's before anything consults the model about what is
            // deliverable.
            sync_task_priority(activation, vlapic)?;
            Ok(())
        }
    }
}

/// What an entry's transition stands on: the model's reset count, and what that
/// count makes of the backing page.
///
/// One value rather than two arguments because the two are one reading. A
/// transition that recorded a count other than the one it decided against would
/// leave the next entry believing a page of the wrong life.
#[derive(Clone, Copy, Debug)]
struct Standing {
    /// The model's reset count as the entry read it.
    epoch: u64,
    /// Whose state the page holds.
    life: Life,
}

impl Standing {
    /// Where an entry stands, out of the count the page was last built at and
    /// the count the model is on now.
    ///
    /// The one test of staleness on this path. The count moves whenever the
    /// register file is cleared or seeded, and only the processor the
    /// controller belongs to moves it — at an exit boundary, which is where
    /// a transition is not. So a mismatch read here says exactly one thing:
    /// the model the page was built from has been replaced since.
    const fn of(rebuilt_at: u64, epoch: u64) -> Self {
        Self {
            epoch,
            life: if rebuilt_at == epoch {
                Life::Same
            } else {
                Life::Ended
            },
        }
    }
}

/// The face a control block's enable bits name, or nothing while they are
/// clear.
fn face_of(vcpu: &Vcpu) -> Option<Face> {
    let interrupts = vcpu.control().interrupt_control;
    interrupts.avic_enable().then(|| {
        if interrupts.x2avic_enable() {
            Face::X2Avic
        } else {
            Face::XAvic
        }
    })
}

/// Turns the acceleration on for this processor, at the entry that asked.
///
/// What it is about to arm is examined before anything is armed, because the
/// processor's own verdict on an illegal combination is an exit that executed
/// no guest instruction and cannot be resumed from — so a block armed and then
/// refused would end the guest, where a refusal here leaves this processor's
/// interrupts the software's to deliver. Nothing has been edited when the
/// refusal is reported.
///
/// The permission map's pass-through goes first wherever the wider face is
/// what turns on: granting an access after its enable bit is set would leave
/// the guest a window of unguarded registers, and the order that cannot be
/// wrong is the order that cannot be observed.
fn enable(
    activation: &Activation,
    vcpu: &mut Vcpu,
    vlapic: &Vlapic,
    face: Face,
    standing: Standing,
) -> Result<(), VlapicError> {
    let limit = face_limit(face, activation.max_index);
    if let Some(invalid) = vcpu.avic_refusal(face == Face::X2Avic, limit) {
        return Err(VlapicError::AvicRefused(invalid));
    }
    rebuild_backing(activation, vlapic, standing)?;
    let mut soil = CleanBits::INTERRUPT.union(CleanBits::AVIC);
    if face == Face::X2Avic {
        vcpu.passthrough_msrs(activation.window, crate::face::msr::passthrough())?;
        soil = soil.union(CleanBits::PERMISSION_MAPS);
    }
    let control = vcpu.control_mut();
    // The table's extent is published with the enable bits rather than at
    // provisioning: how far the hardware may walk it is the face's answer, and
    // this is where the face is decided.
    control.avic_physical_table = control.avic_physical_table.with_max_index(limit);
    control.interrupt_control = control
        .interrupt_control
        .with_avic_enable(true)
        .with_x2avic_enable(face == Face::X2Avic);
    // The enable bit is one clean group and the pointers beside it are
    // another, and both were touched by the life the guest lived since the
    // acceleration was last on.
    vcpu.soil(soil);
    // The processor may have cached a translation of the register page from
    // before the redirection existed; nothing but a flush gets rid of it.
    vcpu.flush();
    info!(
        "vlapic: {} turned hardware delivery on for its guest, in the {face:?} face, over {} \
         addressable entries",
        vlapic.index(),
        usize::from(limit) + 1
    );
    Ok(())
}

/// Turns the acceleration off for this processor, carrying the hardware's
/// state back into the model first.
///
/// Where the wider face is what turns off, full interception comes back
/// before the enable bit goes: the one state the architecture must never
/// see is a guest that believes it owns its controller's registers while
/// nothing guards them, and restoring first is the order that cannot make
/// one.
///
/// # A page of a life that has ended is not carried
///
/// The carry-back runs only where the model is still the one the page was built
/// from, and the reset count is what says so. Every model reset is a lifecycle
/// boundary the architecture requires to clear the task priority, the
/// in-service bank, the requests and the trigger modes — so a deactivation
/// reached one entry later would put all four back, and two of them are not
/// merely wrong: a forced in-service bit is one nothing will ever acknowledge,
/// which floors the new guest's processor priority for as long as it lives, and
/// a forced trigger mode claims an acknowledgement is owed after the reset's
/// own settlement has already been made.
///
/// The reset that gets here is the one the guest's face did not survive — its
/// base register's enable bit cleared, which resets the file and leaves no
/// controller to drive. An `INIT` and the start-up message after it leave the
/// face where it was, so they arrive at the steady arm instead, and it rebuilds
/// the page from the same reading this refuses to carry.
///
/// The count is recorded only where the carry-back ran, because that is what
/// empties the page: a page still holding a dead guest's requests is one the
/// activation that next rebuilds it must still reconcile, and recording here
/// would tell that activation the page was this life's.
fn disable(
    activation: &Activation,
    vcpu: &mut Vcpu,
    vlapic: &Vlapic,
    face: Face,
    standing: Standing,
) -> Result<(), VlapicError> {
    let mut soil = CleanBits::INTERRUPT.union(CleanBits::AVIC);
    if face == Face::X2Avic {
        // Setting a permission bit already set changes nothing, so the whole
        // claimed range is restored rather than the handful that was given
        // back: the result is the map the block was created with, and no
        // count of what moved in between.
        vcpu.intercept_msrs(activation.window, crate::intercepted())?;
        soil = soil.union(CleanBits::PERMISSION_MAPS);
    }
    if standing.life.carried() {
        sync_into_model(activation, vlapic)?;
        activation.rebuilt_at[vlapic.index().get()].store(standing.epoch, Ordering::Relaxed);
    }
    let control = vcpu.control_mut();
    control.interrupt_control = control
        .interrupt_control
        .with_avic_enable(false)
        .with_x2avic_enable(false);
    vcpu.soil(soil);
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

/// Keeps the acceleration on and moves it between faces, at the entry that
/// asked.
///
/// The backing page is the guest's rather than the face's and survives the
/// move whole — both faces read and write the same registers in it — which
/// is what makes this a bit, a permission map and one slot rather than a
/// deactivation and an activation. The slot is the identifier, whose shape the
/// face decides: see [`rewrite_identifier`]. The logical table is likewise left
/// alone: the wider face does not consult it, and the narrower one never stops
/// finding it populated.
///
/// What does not survive the move is how far the table may be walked, because
/// that is the one thing about the acceleration the two faces disagree on. The
/// new face's answer is published with the bit that selects it, and examined
/// before either is written for the same reason [`enable`] examines its own: a
/// block found illegal here is left in the face it was already in, which the
/// processor has been entering all along.
///
/// The reset count is deliberately not recorded here, which is what leaves a
/// stale page still stale: a face change is not a rebuild, so a model reset the
/// page has not caught up with is one the next entry's steady arm sees and
/// answers. Recording the count would be this arm claiming a rebuild it did not
/// perform.
fn switch(
    activation: &Activation,
    vcpu: &mut Vcpu,
    vlapic: &Vlapic,
    from: Face,
    to: Face,
) -> Result<(), VlapicError> {
    let limit = face_limit(to, activation.max_index);
    if let Some(invalid) = vcpu.avic_refusal(to == Face::X2Avic, limit) {
        return Err(VlapicError::AvicRefused(invalid));
    }
    rewrite_identifier(activation, vlapic)?;
    // The face being moved to is the whole of what the permission map owes:
    // the same face twice is the steady state and [`move_for`] does not call
    // it a switch, so `from` is here to be said in the log.
    match to {
        Face::X2Avic => {
            vcpu.passthrough_msrs(activation.window, crate::face::msr::passthrough())?;
        }
        // A guest cannot step from the wider face back to the narrower one —
        // the base register's state machine refuses the move — so this arm is
        // the defensive one, written anyway because a block found in it must
        // come out of it guarded.
        Face::XAvic => {
            vcpu.intercept_msrs(activation.window, crate::intercepted())?;
        }
    }
    let control = vcpu.control_mut();
    control.avic_physical_table = control.avic_physical_table.with_max_index(limit);
    control.interrupt_control = control
        .interrupt_control
        .with_x2avic_enable(to == Face::X2Avic);
    // The enable word moved and so did the permission map, and a translation
    // cached under the old face's redirection is one the flush alone retires.
    vcpu.soil(
        CleanBits::INTERRUPT
            .union(CleanBits::AVIC)
            .union(CleanBits::PERMISSION_MAPS),
    );
    vcpu.flush();
    info!(
        "vlapic: {} moved hardware delivery from the {from:?} face to the {to:?} face, over {} \
         addressable entries",
        vlapic.index(),
        usize::from(limit) + 1
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
/// Nothing at all on the erratum families: the bit stays clear for the life of
/// the machine there, and every directed IPI takes the exit it then cannot
/// avoid. Nothing either for a processor whose identifier the face being driven
/// cannot address: its entry is one the hardware never walks, so a running bit
/// set there would be a promise nothing reads, and a sender that read it back
/// would skip the wake the target does need. [`unpublish_running`] is
/// deliberately not gated the same way — see there.
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
    let Some(face) = active_face(vlapic) else {
        return Ok(());
    };
    if vlapic.apic_id().get() > u32::from(face_limit(face, activation.max_index)) {
        return Ok(());
    }
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
/// Judged against the table's own extent rather than the face's limit, which is
/// the asymmetry with [`publish_running`] and is deliberate: a bit is only ever
/// set where the driving face could address it, but it must be clearable
/// wherever it may have been set. The face can change between the publish and
/// the withdrawal — a machine-wide demotion is a store any processor may make —
/// and a withdrawal that declined on that account would leave the bit standing,
/// which is a sender told this processor is in the guest for the rest of the
/// machine's life.
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
/// processor priority the backing TPR and in-service state impose.
///
/// The page's software-enable bit is the only test of it anywhere on this
/// path, and load-bearing rather than a cross-check: the model gates
/// *activation* and does not consult the bit at all — see [`active_for`] —
/// while the page gates *delivery*, which is the question being asked here. A
/// software-disabled controller takes nothing, and this is where that is
/// decided.
///
/// # Whether the page may hold anything at all is the caller's question
///
/// Nothing here asks whether the acceleration is *permitted*, and that is the
/// point. A page goes on holding requests for as long as the control block's
/// own enable bit is set, which outlives the permission by exactly one entry: a
/// machine-wide demotion is a store any processor may make, while the processor
/// whose page it is may be parked and reach no entry until this answers yes.
/// Refusing on the strength of the permission would leave the only copy of a
/// vector in a page its own processor had stopped consulting, with the kick
/// that announced it already spent — and the page is harvested into the model
/// at an entry, which is the very thing that would then never happen.
///
/// So the caller asks the control block instead, and what is asked here is only
/// the architecture's comparison. The entry that answer produces settles the
/// page one way or another: while the enable bit is set the transition either
/// leaves the hardware driving the page or takes its state into the model, and
/// once the bit is clear the caller stops asking — so a request cannot make
/// this answer yes forever with nothing draining it.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub(crate) fn deliverable() -> Result<bool, VlapicError> {
    let activation = ACTIVATED.get().ok_or(VlapicError::NotProvisioned)?;
    let vlapic = current()?;
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
/// # And then the page is given what the model made of the value
///
/// The word the hardware stored is the guest's raw one; the model's is the
/// narrowed one, and sometimes the refused one — a reserved bit dropped, a
/// read-only register left as it was, an entry this controller does not have
/// not written at all. Every read of these registers is served out of the page
/// with no exit, so a page left holding the raw word is a guest reading back
/// what its own controller refused: an identifier it appears to have renamed,
/// out of which the hardware then resolves a destination the guest cannot
/// address, or a reserved bit that survives until the next rebuild silently
/// takes it away. So every slot the write moved is rewritten from the model,
/// which is [`mirror_slot`], and the page goes back to being a projection of
/// it. The store costs nothing where the hardware narrowed the value itself.
///
/// The error status is that rule rather than an exception to it. Its protocol
/// is write-then-read — the write latches whatever the controller has noticed
/// since the last one and a read answers the latched word — so the page is
/// given the word this write just latched and not the zero the guest wrote.
/// Latching is the only thing that moves that word, which is why one store here
/// is the whole of keeping the two in step.
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
        other => {
            let activation = ACTIVATED.get().ok_or(VlapicError::NotProvisioned)?;
            let page = activation.page(vlapic.index().get())?;
            let value = activation
                .word(page, other.offset())?
                .load(Ordering::Acquire);
            let written = dispatch::write(vlapic, other, value);
            dispatch::acted(vlapic, written);
            // After the bookkeeping rather than before it: two of the bits a
            // local vector entry reads back are the real controller's own
            // reports, and what has just reprogrammed the source they come from
            // is the bookkeeping.
            for slot in mirrored(other, written) {
                mirror_slot(activation, page, vlapic, slot)?;
            }
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

/// Which slots of the page a trapped write leaves owing the model's answer.
///
/// The register the guest named, always. And every local vector table entry
/// where the write software-disabled the controller, because that is the one
/// write the architecture defines as changing registers the guest did not name:
/// it masks all of them, in the stored entries a guest reads back, and the
/// hardware serves those slots without an exit — so a page left as it was is a
/// controller whose sources still read as live after its guest switched it off.
fn mirrored(register: Register, written: Written) -> impl Iterator<Item = Register> {
    // The whole table or none of it: the architecture masks every entry, and the
    // model has already done so by the time this is asked.
    let masked: &'static [Entry] = if matches!(written, Written::Disabled) {
        &Entry::ALL
    } else {
        &[]
    };
    once(register).chain(masked.iter().copied().map(Entry::register))
}

/// Rewrites one slot of a backing page with what the model answers a read of
/// that register with.
///
/// [`dispatch::read`] is the one statement of what each register answers with,
/// so nothing here knows which register it has been handed — which is what
/// keeps the identifier's face-dependent shape, the error status's latched word
/// and an absent entry's masked reset value from becoming three special cases
/// of one store.
///
/// # Errors
///
/// As [`crate::read_msr`].
fn mirror_slot(
    activation: &Activation,
    page: PhysAddr,
    vlapic: &Vlapic,
    register: Register,
) -> Result<(), VlapicError> {
    activation
        .word(page, register.offset())?
        .store(dispatch::read(vlapic, register), Ordering::Release);
    Ok(())
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

/// What the guest reads out of one of the registers the hardware owns while
/// it drives the controller, where that is not what the model holds.
///
/// `None` answers for a register the model is still the truth of — the
/// identifier, the version, the local vector entries and everything else the
/// guest programs through exits — and `Some` for the registers that move
/// without one: the priorities, the vector banks, and the command the
/// hardware carries out between the page's two halves.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub(crate) fn msr_read(register: Register) -> Result<Option<u64>, VlapicError> {
    let activation = ACTIVATED.get().ok_or(VlapicError::NotProvisioned)?;
    let vlapic = current()?;
    let page = activation.page(vlapic.index().get())?;
    let word = |slot: Register| -> Result<u32, VlapicError> {
        Ok(activation
            .word(page, slot.offset())?
            .load(Ordering::Acquire))
    };
    let value = match register {
        Register::TASK_PRIORITY | Register::PROCESSOR_PRIORITY => u64::from(word(register)?),
        // The one register the wide face reads whole: the page holds it as
        // the older face does, in two halves, and a reader of it is owed the
        // halves as one.
        Register::COMMAND_LOW => {
            u64::from(word(Register::COMMAND_LOW)?)
                | (u64::from(word(Register::COMMAND_HIGH)?) << u32::BITS)
        }
        _ if register.bank().is_some() => u64::from(word(register)?),
        _ => return Ok(None),
    };
    Ok(Some(value))
}

/// Performs a write the permission map handed back while the hardware drives
/// the controller.
///
/// The model is not the truth of these registers while it does — the page
/// is — so the value goes where the hardware reads it and whatever the
/// register owes beyond the store is performed here rather than left to the
/// model's path: the acknowledgement an end-of-interrupt owes real hardware,
/// the delivery a command asks for, the request a self-interrupt names.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub(crate) fn msr_write(register: Register, value: u64) -> Result<Written, VlapicError> {
    let activation = ACTIVATED.get().ok_or(VlapicError::NotProvisioned)?;
    let vlapic = current()?;
    let page = activation.page(vlapic.index().get())?;
    match register {
        // The hardware reads the task priority out of the page while it
        // drives; the model takes it back at whichever entry next consults
        // it.
        Register::TASK_PRIORITY => {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the face refused every bit above the low half before this was asked"
            )]
            let narrow = value as u32;
            activation
                .word(page, register.offset())?
                .store(narrow, Ordering::Release);
            Ok(Written::Nothing)
        }
        // The interception means the hardware never performed the write, so
        // what is owed is the acknowledgement whole: the in-service bit the
        // page holds and the release of whatever real hardware is holding.
        Register::END_OF_INTERRUPT => {
            end_of_interrupt(vlapic, None)?;
            Ok(Written::Nothing)
        }
        Register::ERROR_STATUS => {
            // The write latches whatever the controller has noticed since the
            // last one, and a read of the register answers the latched word — so
            // the page is given that word rather than the zero the guest wrote,
            // exactly as a trapped write gives it. Nothing reads it through this
            // face, where the read is intercepted and the model answers it; the
            // page holds it because the page is a projection of the model, and a
            // slot that is not is one that changes at the next transition.
            let written = dispatch::write(vlapic, register, 0);
            mirror_slot(activation, page, vlapic, register)?;
            Ok(written)
        }
        // A command the map handed back is one the hardware never attempted,
        // and is completed by the software path end to end, exactly as an
        // incomplete delivery the hardware reported would be.
        //
        // Both authorities are given the value first, because the hardware
        // performed no part of this write and neither of them otherwise has it:
        // the page is what a guest's read is answered out of while the
        // acceleration drives, and the model is what answers it after a
        // deactivation. The store leaves the delivery status clear as well,
        // which is how this face requires a guest to write it, so the
        // completion's own clear of that bit finds nothing left to correct
        // here — it is there for the command the hardware did attempt.
        Register::COMMAND_LOW => {
            let command = Command::from_bits(value);
            activation
                .word(page, Register::COMMAND_LOW.offset())?
                .store(command.low(), Ordering::Release);
            activation
                .word(page, Register::COMMAND_HIGH.offset())?
                .store(command.high(), Ordering::Release);
            let _stored = vlapic.set_command(value);
            complete_command(value)?;
            Ok(Written::Nothing)
        }
        // A self-interrupt is a request against one's own page, with the
        // architecture's check of the vector performed first.
        Register::SELF_IPI => {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "a vector is the low eight bits of the register it is written in"
            )]
            let vector = Vector::new(value as u8);
            if priority::legal(vector) {
                request(vector)?;
            } else {
                error::noticed(
                    vlapic,
                    Errors::SEND_ILLEGAL_VECTOR | Errors::RECEIVE_ILLEGAL_VECTOR,
                );
            }
            Ok(Written::Nothing)
        }
        // Everything else is a register the guest programs through exits —
        // the spurious vector, the local vector entries, the timer's
        // configuration — and the page must hold what the model is told, or
        // the hardware evaluates state the guest never set.
        other => {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the face refused every bit above the low half before this was asked"
            )]
            let narrow = value as u32;
            activation
                .word(page, other.offset())?
                .store(narrow, Ordering::Release);
            Ok(dispatch::write(vlapic, other, narrow))
        }
    }
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
/// The reset image with the whole projection written over it. An activation is
/// invisible to the guest, so every register the page answers a read with has
/// to hold what the model holds rather than what a reset would have left there
/// — the task priority and the three banks as much as the spurious vector and
/// the local vector table. The guest's own writes reach the page directly
/// afterwards; this is only what has to be there before the first of them.
///
/// The mode is taken once and threaded into the projection, because the shape
/// of two of those registers is the face's and a page built from two loads of
/// it would be a page in neither face.
///
/// # Slot by slot, because the frame has other writers
///
/// A whole-page copy over this frame is the one operation that cannot be made
/// correct here, whatever it is entered with: another processor's hardware sets
/// a request bit in this page whenever its guest sends an interrupt to this
/// one, and this processor's own interrupt handler does the same for a device
/// arrival taken in the host window — the window between the exit that made the
/// controller eligible and this entry, in which host interrupts are enabled
/// throughout. Both writers would be erased, and the second one costs more than
/// the interrupt: an arrival whose real acknowledgement is being withheld
/// records the debt for it before it publishes the request, so erasing the
/// request leaves a debt with nothing left that could ever discharge it.
///
/// So the image is written through one atomic per register slot, and the
/// request bank is reconciled rather than overwritten — see
/// [`ResetImage::publish`], which is where what becomes of each displaced bit
/// is decided.
fn rebuild_backing(
    activation: &Activation,
    vlapic: &Vlapic,
    standing: Standing,
) -> Result<(), VlapicError> {
    let page = activation.page(vlapic.index().get())?;
    let mut image = ResetImage::new(vlapic.apic_id(), vlapic.version());
    Projection::of(vlapic, vlapic.mode()).overlay(&mut image);
    let local = apic::local().ok();
    image.publish(
        standing.life,
        |offset| activation.word(page, offset),
        |vector| {
            // A request bit the page held that this life's model does not is one
            // the guest it was meant for never took, and that guest has stopped
            // existing. Real hardware may still be holding the vector in service
            // on its behalf, and a debt whose request has just been deleted is
            // one nothing can ever discharge — the acknowledgement it waits for
            // is the guest's, for an interrupt this bit was the only record of.
            // So it is written off here rather than left waiting.
            //
            // Only a debt this ledger really has. Writing one off that does not
            // exist would report a debt the machine is not holding, and on a
            // controller that retires by name it would stop a line the guest is
            // still using from being accepted at all.
            if vlapic.ledger().owes(vector) {
                vlapic.ledger().abandon(vector, &local);
            }
        },
    )?;
    activation.rebuilt_at[vlapic.index().get()].store(standing.epoch, Ordering::Relaxed);
    mirror_logical(vlapic)?;
    Ok(())
}

/// Carries the hardware's state back into the model.
///
/// The direction a deactivation goes: the projection is taken back out of the
/// page and the model takes what the hardware moved while it drove — the task
/// priority the guest set without exiting, the command it sent, and whatever
/// the hardware accepted, took into service and recorded as level, so that
/// software-only delivery continues from exactly where the hardware left it.
/// Which registers those are, and which of them the page merely holds a copy
/// of, is [`Projection::into_model`]'s.
///
/// Runs at an *entry* boundary, because that is where a transition is
/// performed: this processor's guest is stopped and its own hardware is reading
/// nothing. What is not stopped is a peer whose control block still has the
/// acceleration armed, which goes on setting request bits in this page until
/// its own next entry — so the three banks are taken with a swap apiece rather
/// than read, and [`Projection::take`] is where that is argued.
fn sync_into_model(activation: &Activation, vlapic: &Vlapic) -> Result<(), VlapicError> {
    let page = activation.page(vlapic.index().get())?;
    Projection::take(|offset| activation.word(page, offset))?.into_model(vlapic);
    Ok(())
}

/// Copies the backing task priority into the model.
///
/// The one register the steady state carries back, because it is the one the
/// guest changes without exiting: anything that consults the model about what
/// is deliverable has to be looking at the number the hardware is looking at.
/// A deactivation carries it with everything else.
fn sync_task_priority(activation: &Activation, vlapic: &Vlapic) -> Result<(), VlapicError> {
    let page = activation.page(vlapic.index().get())?;
    let value = activation
        .word(page, Register::TASK_PRIORITY.offset())?
        .load(Ordering::Relaxed);
    vlapic.set_task_priority(value);
    Ok(())
}

/// Rewrites the one slot of the backing page whose shape a face change moves.
///
/// The page survives a move between the faces whole — both of them read and
/// write the same registers in it — with the identifier as the exception: the
/// older face keeps it in the top byte and the wider one uses the whole word,
/// and the hardware derives the logical destination it matches an
/// interprocessor interrupt against from that slot. A page left in the shape
/// the face before it used is a processor the guest can no longer address.
///
/// Which shape that is comes from the model, through [`mirror_slot`], as it
/// does for the trap that answers a guest's own write of the register.
fn rewrite_identifier(activation: &Activation, vlapic: &Vlapic) -> Result<(), VlapicError> {
    let page = activation.page(vlapic.index().get())?;
    mirror_slot(activation, page, vlapic, Register::ID)
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
    //! The decisions here that need no machine: which face the acceleration
    //! drives a controller in, how far it reaches in each, which vector an EOI
    //! retired, which logical destinations name a table entry, which move an
    //! entry owes the acceleration, which life the page it holds belongs to,
    //! and which of its slots a trapped write leaves owing the model's answer.

    use alloc::vec::Vec;
    use core::iter::once;

    use descriptors::Vector;
    use svm::avic::MAX_PHYSICAL_ID;

    use super::{
        Entry, Face, Life, Mode, Move, Register, Standing, Written, driven_face, eoi_vector,
        face_limit, logical_slot, mirrored, move_for,
    };

    /// Every mode a controller can be in, which is the one term of the
    /// activation gate that is not a boolean.
    const MODES: [Mode; 3] = [Mode::Disabled, Mode::XApic, Mode::X2Apic];

    #[test]
    fn the_acceleration_drives_the_face_of_a_mode_it_has_one_for() {
        for x2avic in [false, true] {
            assert_eq!(
                driven_face(false, Mode::XApic, x2avic, false),
                Some(Face::XAvic),
                "the older face is what a provisioned machine was built for"
            );
            // A controller its guest has globally disabled has none to drive,
            // and the way back is a register this hypervisor never stops
            // intercepting.
            assert_eq!(driven_face(false, Mode::Disabled, x2avic, false), None);
        }
        assert_eq!(
            driven_face(false, Mode::X2Apic, true, false),
            Some(Face::X2Avic)
        );
        // The wider face on a machine the policy left without it: a mode the
        // acceleration cannot drive, so the software path serves it.
        assert_eq!(driven_face(false, Mode::X2Apic, false, false), None);
    }

    #[test]
    fn the_gate_is_exactly_no_demotion_and_a_face_the_policy_drives() {
        // The whole truth table, so that a term added to or taken out of the
        // gate has to be argued for here as well as written there. The
        // software-enable bit is deliberately not among the terms: a
        // controller its guest has switched off stays driven, because the
        // register that switches it back on is served out of the backing page.
        //
        // Whether the structures exist at all is the one term that is not a
        // value here: it is the `ACTIVATED.get()?` above the call, and cannot
        // be false while there is an `x2avic` answer to pass.
        for mode in MODES {
            for machine_inhibited in [false, true] {
                for x2avic in [false, true] {
                    for inhibited in [false, true] {
                        let driven = !machine_inhibited
                            && !inhibited
                            && match mode {
                                Mode::XApic => true,
                                Mode::X2Apic => x2avic,
                                Mode::Disabled => false,
                            };
                        assert_eq!(
                            driven_face(machine_inhibited, mode, x2avic, inhibited).is_some(),
                            driven,
                            "{mode:?}, machine inhibited {machine_inhibited}, x2avic \
                             {x2avic}, inhibited {inhibited}"
                        );
                    }
                }
            }
        }
    }

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
        // The flat encoding is the only one that is flat. The architecture
        // defines two values for the format register and leaves every other
        // undefined, so anything that is not the flat one is matched the
        // cluster way — which is the shape of the table the hardware resolves
        // a logical destination through, and what the kernel's own
        // acceleration does with the same register.
        assert_eq!(logical_slot(0x2100_0000, 0x1FFF_FFFF), Some(8));
        assert_eq!(logical_slot(0x2100_0000, 0xEFFF_FFFF), Some(8));
    }

    #[test]
    fn an_entry_owes_nothing_where_both_sides_agree() {
        assert_eq!(move_for(None, None), Move::Idle);
        for face in [Face::XAvic, Face::X2Avic] {
            assert_eq!(move_for(Some(face), Some(face)), Move::Steady);
        }
    }

    #[test]
    fn an_entry_turns_the_acceleration_on_and_off_in_the_face_it_is_asked_for() {
        for face in [Face::XAvic, Face::X2Avic] {
            assert_eq!(move_for(None, Some(face)), Move::Enable(face));
            assert_eq!(move_for(Some(face), None), Move::Disable(face));
        }
    }

    #[test]
    fn an_entry_moves_a_live_acceleration_between_faces() {
        assert_eq!(
            move_for(Some(Face::XAvic), Some(Face::X2Avic)),
            Move::Switch {
                from: Face::XAvic,
                to: Face::X2Avic,
            }
        );
        // The guest's own state machine cannot ask for the other direction,
        // but a block found in it is one the table still describes.
        assert_eq!(
            move_for(Some(Face::X2Avic), Some(Face::XAvic)),
            Move::Switch {
                from: Face::X2Avic,
                to: Face::XAvic,
            }
        );
    }

    #[test]
    fn the_older_face_reaches_no_further_than_the_broadcast_identifier() {
        // Pinned as literals on both sides of the boundary, because this cap is
        // the difference between a machine that boots and one whose guest dies
        // at its first accelerated entry: 0xFE is the highest identifier an
        // eight-bit destination field can name, 0xFF being the encoding that
        // means every processor.
        assert_eq!(face_limit(Face::XAvic, 0x0FE), 0x0FE);
        for provisioned in [0x0FF_u16, 0x100, 0x1FF, 0x200, 0xFFF] {
            assert_eq!(
                face_limit(Face::XAvic, provisioned),
                0x0FE,
                "{provisioned:#x}"
            );
        }
        // And it is the constant the architecture's own reason is written
        // against, not a second spelling of the same number.
        assert_eq!(face_limit(Face::XAvic, 0xFFF), MAX_PHYSICAL_ID);
    }

    #[test]
    fn the_wider_face_reaches_every_entry_the_table_was_sized_for() {
        // The table is provisioned for the widest face the machine may drive it
        // in, so in that face nothing is clamped — including the identifier the
        // older face has to give up.
        for provisioned in [0x0FE_u16, 0x0FF, 0x100, 0x1FF, 0x200, 0xFFF] {
            assert_eq!(
                face_limit(Face::X2Avic, provisioned),
                provisioned,
                "{provisioned:#x}"
            );
        }
    }

    #[test]
    fn a_machine_no_wider_than_the_older_face_is_clamped_by_neither() {
        // The common machine: every identifier already inside the eight-bit
        // field, so the two faces agree and a mode change costs no change of
        // extent at all.
        for provisioned in [0_u16, 1, 0x0FE] {
            assert_eq!(face_limit(Face::XAvic, provisioned), provisioned);
            assert_eq!(face_limit(Face::X2Avic, provisioned), provisioned);
        }
    }

    #[test]
    fn the_page_belongs_to_the_life_the_model_was_on_when_it_was_built() {
        // The counts a real machine compares. A reset moves the count by two —
        // it is odd for as long as the stores run — so an entry that finds the
        // page built two behind is one reset late, and a guest that resets a
        // processor and starts it again reaches the next transition four behind.
        // A page built before either reset is no less dead for the second one.
        assert_eq!(Standing::of(4, 4).life, Life::Same);
        assert_eq!(Standing::of(4, 6).life, Life::Ended);
        assert_eq!(Standing::of(4, 8).life, Life::Ended);
        // The first activation every processor makes: the page holds the image
        // provisioning wrote, and the model has been reset — and on one processor
        // seeded from firmware's own registers — since.
        assert_eq!(Standing::of(0, 2).life, Life::Ended);
        // What the transition then records is the count it decided against and not
        // a second reading of it: anything else would leave the next entry judging
        // the page against a life no transition ever saw.
        assert_eq!(Standing::of(4, 6).epoch, 6);
    }

    #[test]
    fn the_page_is_carried_into_the_model_only_where_the_life_continues() {
        // The decision the carry-back is gated on. Both arms a reset is followed
        // by consult it: the steady one, where the model was cleared and the face
        // survived, and the deactivation, where the guest cleared the face as
        // well — and the harm the two prevent is the same, a model given back the
        // task priority, the in-service bank and the requests its reset had just
        // been required to clear.
        for (built, now) in [(0_u64, 2_u64), (4, 6), (4, 8), (2, u64::MAX)] {
            assert!(
                !Standing::of(built, now).life.carried(),
                "built at {built}, now {now}"
            );
        }
        for count in [0_u64, 2, 4, u64::MAX] {
            assert!(Standing::of(count, count).life.carried(), "{count}");
        }
    }

    #[test]
    fn a_trapped_write_answers_for_the_slot_the_guest_named() {
        // The whole of what an ordinary trapped write owes the page: the register
        // it was made to, holding what the model made of it rather than the raw
        // word the hardware stored.
        for register in [
            Register::ID,
            Register::ERROR_STATUS,
            Register::REMOTE_READ,
            Register::SPURIOUS,
            Register::LVT_TIMER,
            Register::TIMER_DIVIDE,
        ] {
            assert!(
                mirrored(register, Written::Nothing).eq([register]),
                "{register:?}"
            );
        }
    }

    #[test]
    fn a_software_disable_answers_for_every_entry_it_masked() {
        // The one write that changes registers the guest did not name. The
        // architecture has it mask the whole local vector table, and those slots
        // are ones the hardware answers a read of with no exit at all — so a
        // disable that rewrote only the register the guest wrote would leave a
        // controller reading its own sources back live.
        let owed: Vec<Register> = mirrored(Register::SPURIOUS, Written::Disabled).collect();
        let expected: Vec<Register> = once(Register::SPURIOUS)
            .chain(Entry::ALL.map(Entry::register))
            .collect();
        assert_eq!(owed, expected);
        assert_eq!(owed.len(), 1 + Entry::COUNT);
    }
}
