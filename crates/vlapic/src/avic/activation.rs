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
//! - Entry *i* of the physical table is written only by pCPU *i*, and its two
//!   mutable bits are each toggled with a read-modify-write. The bit that says
//!   a peer's hardware may resolve an interrupt here at all is Release on the
//!   set that publishes it and Release on the clear that withdraws it, and an
//!   Acquire load is what every decision made from it reads.
//! - The bit that says the processor is in the guest is sequentially consistent
//!   on both, and so is the rescan the withdrawal precedes. That is not
//!   caution: the two halves of the park protocol are a store followed by a
//!   load of a *different* location on each side — the target withdraws the bit
//!   and then rescans its page, while a sender's hardware sets a request bit in
//!   that page and then reads the bit — and the argument that neither half can
//!   miss the other needs every access in it to be in the single total order
//!   that only sequentially consistent operations join. `delivery::doorbell`
//!   makes that argument in full for the software half of the same protocol,
//!   spells out why a release read-modify-write and an acquire load do not give
//!   it, and shows that on this processor it costs no instructions at all. What
//!   this hypervisor cannot spell is the hardware's half, and what it assumes
//!   of it is that the hardware's read of an entry is coherent with a locked
//!   write from the core that entry describes: nothing in the design covers
//!   that read answering from before the withdrawal.
//! - A backing page is written by its own processor alone — by these functions
//!   at transition boundaries and at the exits its own guest's register writes
//!   raise, and by the hardware while the guest runs — except for the
//!   interrupt-request words, which any processor may atomically OR into. The
//!   OR is Release: it publishes the request before the doorbell or the host
//!   interrupt that tells the target to look.
//! - The trigger-mode words go with them. A request published into a page
//!   carries the trigger mode the hardware classifies it by, written first and
//!   also Release, because a processor that observed the request without it
//!   would read an acknowledgement's fate out of the record of an earlier
//!   arrival — see [`request`].
//! - The logical table is rebuilt under [`Activation::logical_lock`], which is
//!   taken wherever an entry moves — the handlers that answer a guest's write
//!   of its logical identity, and the transitions that change the face the
//!   acceleration drives a controller in — and never held across an exit.
//!   Entries move as whole aligned words, and a processor's old entry is
//!   invalidated before its new one is published, the publishing store being
//!   Release so that nothing in the model permits the two being reordered. The
//!   lock does not establish that, because the reader it has to hold against is
//!   not a thread and takes no lock: it is the AVIC hardware of any core. So a
//!   resolution racing a rebuild reads the old entry, the new one, or no entry
//!   — a delivery, a delivery, or an exit the software path completes — and
//!   never two entries naming one processor.
//! - The inhibits are single atomic booleans, and Relaxed on every access. A
//!   boolean is the whole of what either publishes: nothing is written before
//!   one that a reader of it must see, and every reader acts on the boolean
//!   alone.
//!
//! # And one authority
//!
//! Which of the two copies of a register is the truth is a different question
//! from which processor may write it, and while a control block carries the
//! acceleration the answer is the page for four of them: the three vector banks
//! and the task priority. So the model is emptied of everything the hardware
//! can deliver at the entry ([`hand_over`]), is not written from the control
//! block's copy of the task priority at the exit ([`TaskPriority`]), and is
//! given the page's state back whole when the acceleration comes off
//! ([`sync_into_model`]) or when this processor stops trusting it
//! ([`hand_back`]).
//!
//! One kind of interrupt is the exception, and it is the one no controller
//! holds in service: an arrival that came in through the pin that bypasses the
//! controller stays in the model and is injected from there.
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
    fmt,
    iter::once,
    sync::atomic::{AtomicBool, AtomicU16, AtomicU32, AtomicU64, Ordering},
};

use apic::REGISTER_STRIDE;
use cpu::{ApicId, CpuIndex};
use descriptors::Vector;
use log::{info, trace, warn};
use paging::DirectMap;
use spin::{Mutex, Once};
use svm::{
    CleanBits,
    avic::{LogicalApicEntry, MAX_PHYSICAL_ID, PhysicalApicEntry},
};
use vcpu::Vcpu;
use x86_64::{PhysAddr, instructions::interrupts};

use crate::{
    VlapicError,
    avic::backing::{Handover, Life, Projection, ResetImage, bank_bit, publish_request},
    delivery::error,
    face::{dispatch, dispatch::Written, table::Register},
    machine::{current, diagnostics::Report, registry},
    priority::{self, Priority},
    registers::{
        FLAT_DESTINATION_FORMAT, Vlapic,
        base::Mode,
        bitmap::SLOTS,
        error::Errors,
        icr::{Command, Trigger},
        lvt::Entry,
    },
};

/// The bit of a physical-table entry that says the processor it describes is in
/// the guest.
const IS_RUNNING: u64 = PhysicalApicEntry::new().with_is_running(true).into_bits();

/// The bit of a physical-table entry that says an interrupt may resolve to it,
/// which is whether the hardware is the authority for the controller it
/// describes.
const IS_VALID: u64 = PhysicalApicEntry::new().with_valid(true).into_bits();

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
    ///
    /// Written by any processor and read by all of them, Relaxed on both:
    /// nothing is published with it, and a reader that acted on it a moment
    /// late is a processor taking one more accelerated entry, which is what
    /// every reading of it is one entry away from anyway.
    machine_inhibited: AtomicBool,
    /// Held while the logical table and the record of what is in it move.
    ///
    /// Taken with host interrupts off on both of the paths that take it, which
    /// is what bounds the wait for it. The transitions reach it from inside the
    /// entry callback, where both interrupt flags are already clear — a
    /// processor spinning there answers no doorbell, no translation shootdown
    /// and no non-maskable interrupt while it waits — and the handlers that
    /// answer a guest's write of its logical identity reach it from an exit,
    /// where interrupts are enabled and a holder could otherwise be interrupted
    /// for orders of magnitude longer than the critical section itself. Three
    /// rules keep that safe, and all three are the caller's to honour:
    ///
    /// 1. **Nothing under it waits for another processor.** What it guards is a
    ///    bounded run of atomic loads and stores over one frame and one array:
    ///    no second lock, no serial output, nothing that can fault. So a holder
    ///    always finishes, and a spinner waits exactly that long.
    /// 2. **No reentry.** Nothing reached under it settles the table again.
    /// 3. **Nothing that reports is under it.** A line of serial output is
    ///    taken with a machine-wide lock and is far longer than what this
    ///    guards, so [`mirror_logical`] drops the guard and lets interrupts
    ///    back in before it says anything.
    logical_lock: Mutex<()>,
    /// The logical-table slot each processor's identity is published in,
    /// indexed by roster position; [`NO_SLOT`] while it has none.
    ///
    /// Written by the processor's own pCPU, and by any processor that takes a
    /// contested entry out of the table — both under
    /// [`Activation::logical_lock`], which is what makes a record and the entry
    /// it names one fact rather than two.
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
/// # This is the want, and [`accelerated`] is the have
///
/// Every term here is a statement about what the acceleration is *permitted* to
/// do, and none of them says whether the control block was brought to it. So a
/// decision about the run that is happening — what the entry prepared, which
/// authority answered a register access, which of the two copies of the task
/// priority the guest was running under — is [`accelerated`]'s to make, and
/// this one belongs to decisions about what should happen next.
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
        activation.machine_inhibited.load(Ordering::Relaxed),
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
/// not describe, and takes the acceleration off this controller as it is
/// reported — see [`degraded`].
///
/// In every case the caller degrades rather than refusing the entry, and what
/// the guest is entered with is what the control block says rather than what
/// the model asked for: [`accelerated`] is the predicate that makes that true.
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
    let performed = match move_for(have, want) {
        // Both sides agree the software delivers, so nothing is owed but the one
        // encoding of the enable bits the architecture rejects — which this is
        // the only arm that can be left holding.
        Move::Idle => {
            normalize(vcpu, vlapic);
            Ok(())
        }
        Move::Enable(face) => enable(activation, vcpu, vlapic, face, standing),
        Move::Disable(face) => disable(activation, vcpu, vlapic, face, standing),
        Move::Switch { from, to } => switch(activation, vcpu, vlapic, from, to),
        // The steady state. One thing can still have moved: a reset cleared the
        // model underneath an active controller, and the page the hardware
        // serves holds the state that reset was required to destroy.
        //
        // The logical table is settled with the page and for the same reason:
        // the reset cleared the logical identity the entry was derived from, so
        // an entry left behind names this processor for a destination its model
        // no longer answers to.
        //
        // The task priority is deliberately not carried here. It is taken at the
        // exit instead — see [`TaskPriority`] — because the model is consulted
        // between the two, and an entry is what that consultation decides on.
        Move::Steady => {
            if standing.life.carried() {
                Ok(())
            } else {
                rebuild_backing(activation, vlapic, standing)
                    .and_then(|()| mirror_logical(vlapic, want))
            }
        }
    };
    degraded(vlapic, performed)
}

/// Takes the acceleration off this controller where a transition could not be
/// performed, so that a failure degrades rather than being attempted again on
/// every entry for the rest of the guest's life.
///
/// The entry itself is already correct without this, because what the processor
/// is entered with is what the block carries and a transition that failed left
/// it carrying the software path. What this adds is that the *model* stops
/// asking for a transition nothing can perform. Every failure it acts on names
/// hypervisor state a guest cannot move — a frame the window does not reach, a
/// processor the roster does not describe, a permission map the block refused —
/// so the next entry would attempt the same transition, fail the same way, and
/// say so again, once per entry, through a serial port taken with a
/// machine-wide lock. One demotion is one attempt and one line.
///
/// The cost is that a failure whose cause goes away costs this controller its
/// acceleration until the guest's own next change of face, which is where
/// [`Vlapic::permit_avic`] remakes the decision, or until a reset. None of the
/// causes announces when it has gone away, which is why the sticky answer is
/// the honest one.
///
/// A refusal is deliberately not one of them. [`VlapicError::AvicRefused`] is
/// reported with nothing edited — the block is exactly as the processor has
/// been entering it all along — and the rules behind it are re-asked at the
/// next entry for the price of reading fields that are already in cache.
fn degraded(vlapic: &Vlapic, performed: Result<(), VlapicError>) -> Result<(), VlapicError> {
    match performed {
        Ok(()) | Err(VlapicError::AvicRefused(_)) => {}
        Err(_) => vlapic.inhibit_avic(),
    }
    performed
}

/// Takes away the one encoding of the enable bits the architecture defines no
/// meaning for, at the entry that found a block holding it.
///
/// The base bit turns the acceleration on and the wider bit only selects
/// between the faces, so a block carrying the wider bit alone describes no
/// state: the processor refuses it outright, with no guest instruction executed
/// and nothing to resume from. Nothing here produces one — every transition
/// that writes those bits writes both — so this is defence against a block
/// written from somewhere else, and it belongs on this arm because this is the
/// only arm that would not repair it anyway. A controller whose acceleration is
/// wanted reaches [`enable`], which rewrites both bits; one whose acceleration
/// is not wanted used to reach an arm that did nothing at all, and the entry
/// after it was refused, revalidated, reported, and the guest stopped. So the
/// only architecturally invalid encoding was also the only state this state
/// machine could not leave.
///
/// One branch on a word the caller has already loaded is the whole cost, and
/// the arm it sits on is the one every entry of an unaccelerated guest takes.
fn normalize(vcpu: &mut Vcpu, vlapic: &Vlapic) {
    if !vcpu.control().interrupt_control.x2avic_enable() {
        return;
    }
    let control = vcpu.control_mut();
    control.interrupt_control = control.interrupt_control.with_x2avic_enable(false);
    vcpu.soil(CleanBits::INTERRUPT);
    // Unlatched deliberately, and it cannot repeat: the store above is what the
    // line reports, so a second one means a second writer of that field, which
    // is the thing worth hearing about every time it happens.
    warn!(
        "vlapic: {}'s control block carried the wider face's enable bit without the base bit, \
         which is not a state the architecture defines; the bit is taken away",
        vlapic.index()
    );
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
    face_bits(interrupts.avic_enable(), interrupts.x2avic_enable())
}

/// The same, out of the two bits themselves.
///
/// Pure rather than folded into [`face_of`], because these two bits are what
/// every decision about whether the hardware *is* driving is made from — the
/// shape of an entry, the trap-or-fault classification of an unaccelerated
/// access, the transition a reconciliation owes — and one of their four
/// encodings is one the architecture defines no meaning for.
///
/// The base bit is the whole of the answer and the wider bit only chooses
/// between the faces, so a block carrying the wider bit alone is driven in
/// neither: the processor refuses such a block outright, so it names no face
/// this could honestly report. [`normalize`] is what takes the stray bit away.
const fn face_bits(avic_enable: bool, x2avic_enable: bool) -> Option<Face> {
    if !avic_enable {
        return None;
    }
    Some(if x2avic_enable {
        Face::X2Avic
    } else {
        Face::XAvic
    })
}

/// Whether the control block this processor was entered with, or is about to be
/// entered with, carries hardware-driven delivery.
///
/// The *have* rather than the want, and that difference is the whole of what
/// this exists for. [`active_for`] answers whether the acceleration is
/// permitted — the policy, the guest's face, the two inhibits — which is what
/// [`reconcile`] tries to bring the block to and says nothing about whether it
/// got there. The block's enable bits are what the processor is entered with,
/// so every decision about which authority delivers for the run that is
/// *happening* is made from them: whether the model's interrupts crossed into
/// the backing page, whether the task priority the processor honours was
/// mirrored, whether an interrupt window was armed, whether the running bit was
/// published, and whether an access the hardware reported was one it completed.
///
/// The two can differ without anything having failed — a machine-wide demotion
/// is a store any processor may make, and the boot processor reaches its first
/// entry with the activation state already published and no block naming a page
/// yet — and they differ until the next entry wherever a transition could not
/// be performed.
pub(crate) fn accelerated(vcpu: &Vcpu) -> bool {
    face_of(vcpu).is_some()
}

/// Whether the block carries that acceleration in the face a guest reaches its
/// controller through model-specific registers.
///
/// Which face the block carries decides what an unaccelerated access
/// *describes* rather than merely which registers it covers: the exit reports a
/// register offset either way, and under this face no memory access took place
/// at all — the guest executed `RDMSR` or `WRMSR`, and the offset is the one
/// the architecture derives that register's index from.
pub(crate) fn wider_face(vcpu: &Vcpu) -> bool {
    matches!(face_of(vcpu), Some(Face::X2Avic))
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
/// what turns on, and the table entry is published last of the steps that can
/// fail and before the enable bit, so that no failure leaves a peer's hardware
/// delivering into a page this processor is not driving: whichever step reports
/// one, the acceleration is left off and the entry invalid, and the next entry
/// performs the whole transition again.
///
/// # The pass-through and the enable bit are one fact
///
/// The registers the wider face hands the guest reach the *host's* controller
/// with no exit at all, and what makes that safe is the hardware answering them
/// out of this processor's backing page — which it does only while the block
/// carries the acceleration. So the grant and the bit move together: the grant
/// goes first, because granting an access after its enable bit is set would
/// leave the guest a window of unguarded registers, and it is taken back on a
/// failure between the two, because a grant left standing with the bit never
/// set is that window with nothing at all behind it.
///
/// Restoring is all-or-nothing and reaches the same map the grant just reached,
/// so the only way it can fail is a window that stopped reaching a page it
/// reached a moment ago; that failure is reported rather than swallowed, being
/// strictly worse news than the one it was answering.
///
/// The logical table is published before both, and may survive a failure that
/// the physical entry does not, which costs nothing: a logically addressed
/// interrupt resolves through it into a physical entry that is still invalid,
/// which is an exit the software path completes.
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
    mirror_logical(vlapic, Some(face))?;
    let mut soil = CleanBits::INTERRUPT.union(CleanBits::AVIC);
    if face == Face::X2Avic {
        vcpu.passthrough_msrs(activation.window, crate::face::msr::passthrough())?;
        soil = soil.union(CleanBits::PERMISSION_MAPS);
    }
    if let Err(error) = publish_entry(activation, vlapic) {
        // The grant goes back with the failure: the enable bit below is the whole
        // of what justifies it, and it is not going to be set.
        if face == Face::X2Avic {
            vcpu.intercept_msrs(activation.window, crate::intercepted())?;
        }
        return Err(error);
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
    transitioned(
        vlapic,
        Report::Accelerated,
        format_args!(
            "turned hardware delivery on for its guest, in the {face:?} face, over {} addressable \
             entries",
            usize::from(limit) + 1
        ),
    );
    Ok(())
}

/// Turns the acceleration off for this processor, giving the hardware's state
/// back to the model as the last thing it does.
///
/// Where the wider face is what turns off, full interception comes back
/// before the enable bit goes: the one state the architecture must never
/// see is a guest that believes it owns its controller's registers while
/// nothing guards them, and restoring first is the order that cannot make
/// one.
///
/// The table entry is withdrawn before both, and before the carry-back below,
/// which is what makes the carry-back the end of the page's authority rather
/// than a moment in the middle of it: an entry still valid is a peer's hardware
/// still resolving interrupts into a page whose state has already been given
/// back to the model, where a request would wait for the next activation to
/// notice it.
///
/// The logical table's entry goes with it, and for the same reason rather than
/// a weaker one: that table is walked by whichever processor is resolving a
/// logically addressed interrupt, so an entry left behind is one a peer
/// resolves through to a controller the software is now delivering for.
/// Withdrawn second, because the physical entry is what gates delivery and a
/// failure reaching one table must not leave the other standing.
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
///
/// # The enable bits go before the carry-back, which is the fallible half
///
/// Everything that can fail is either before the bits or after them, and the
/// division is what makes a failure describable. Up to and including the
/// interception, a failure leaves the block exactly as the processor has been
/// entering it — accelerated, with the pass-through the face it is in justifies
/// — and the next entry performs the whole transition again. From the bits
/// onward the block is guarded and unaccelerated, and stays so whatever the
/// carry-back reports. There is no third state, and in particular not the one
/// this ordering replaces: full interception restored with the enable bits
/// still set, where the guest runs accelerated with every register of the wider
/// face trapping and the intercepted handlers answer out of the model while the
/// hardware owns the page.
///
/// Moving the bits earlier costs the carry-back nothing. This processor's guest
/// is stopped, its own hardware is reading nothing, and a bit of the control
/// block takes effect at the next `VMRUN` — so what the page holds and what
/// [`Projection::take`] makes of it are the same either way.
fn disable(
    activation: &Activation,
    vcpu: &mut Vcpu,
    vlapic: &Vlapic,
    face: Face,
    standing: Standing,
) -> Result<(), VlapicError> {
    unpublish_entry(activation, vlapic)?;
    mirror_logical(vlapic, None)?;
    let mut soil = CleanBits::INTERRUPT.union(CleanBits::AVIC);
    if face == Face::X2Avic {
        // Setting a permission bit already set changes nothing, so the whole
        // claimed range is restored rather than the handful that was given
        // back: the result is the map the block was created with, and no
        // count of what moved in between.
        vcpu.intercept_msrs(activation.window, crate::intercepted())?;
        soil = soil.union(CleanBits::PERMISSION_MAPS);
    }
    let control = vcpu.control_mut();
    // Everything that can fail is on one side of this or the other, which is what
    // makes each failure describable: before it the block is still the one the
    // processor has been entering, and from here on it is guarded and
    // unaccelerated whatever the carry-back below reports.
    control.interrupt_control = control
        .interrupt_control
        .with_avic_enable(false)
        .with_x2avic_enable(false);
    vcpu.soil(soil);
    vcpu.flush();
    // Defensive: the unpublish belongs to the exit and the park boundaries,
    // and a demotion arriving anywhere else must not leave the bit behind.
    let _ = unpublish_running();
    transitioned(
        vlapic,
        Report::Unaccelerated,
        format_args!("turned hardware delivery off for its guest"),
    );
    if standing.life.carried() {
        sync_into_model(activation, vlapic)?;
        activation.rebuilt_at[vlapic.index().get()].store(standing.epoch, Ordering::Relaxed);
    }
    Ok(())
}

/// Keeps the acceleration on and moves it between faces, at the entry that
/// asked.
///
/// The backing page is the guest's rather than the face's and survives the
/// move whole — both faces read and write the same registers in it — which
/// is what makes this a bit, a permission map and one slot rather than a
/// deactivation and an activation. The slot is the identifier, whose shape the
/// face decides: see [`rewrite_identifier`]. So is this processor's entry in
/// the physical table, which says a peer's hardware may deliver here — true
/// across the move, and in both faces.
///
/// The logical table does not survive it, and that is the one thing about this
/// move the table's own shape decides. Only the older face is resolved through
/// it: the wider one derives a logical identifier from the processor's own
/// identifier and never reads the table at all. But the table is read by
/// whichever processor is *resolving*, whatever face the controller it resolves
/// to is in — so a controller that has moved into the wider face and kept its
/// entry goes on answering, through a peer still in the older face, to an
/// eight-bit logical identifier the architecture says it no longer has. The
/// entry is therefore settled for the face being moved to, before anything else
/// the move touches, so that no peer resolves to an identity this controller
/// has already given up.
///
/// What else does not survive the move is how far the table may be walked,
/// because that is the one thing about the acceleration the two faces disagree
/// on. The new face's answer is published with the bit that selects it, and
/// examined before either is written for the same reason [`enable`] examines
/// its own: a block found illegal here is left in the face it was already in,
/// which the processor has been entering all along.
///
/// The reset count is deliberately not recorded here, which is what leaves a
/// stale page still stale: a face change is not a rebuild, so a model reset the
/// page has not caught up with is one the next entry's steady arm sees and
/// answers. Recording the count would be this arm claiming a rebuild it did not
/// perform.
///
/// # The permission map is the last thing that can fail
///
/// Which is what makes this arm need no undo of its own, where [`enable`] does.
/// The map moves all or not at all — the one failure it has is a window that
/// cannot reach it, discovered before a bit of it is touched — so a failure
/// here leaves the block in the face it arrived in, with the pass-through that
/// face justifies, and the next entry performs the whole move again. Every step
/// before it moves a structure rather than a permission, and a failure in one
/// of those leaves the acceleration where it was: an identity withdrawn from
/// the logical table costs one exit per interrupt addressed there, which the
/// software path completes for both.
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
    mirror_logical(vlapic, Some(to))?;
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
    transitioned(
        vlapic,
        Report::FaceMoved,
        format_args!(
            "moved hardware delivery from the {from:?} face to the {to:?} face, over {} \
             addressable entries",
            usize::from(limit) + 1
        ),
    );
    Ok(())
}

/// Says what a transition did, once per controller for each kind of transition,
/// and traces every one after it.
///
/// How often these happen is the guest's own choice. The register that decides
/// which face its controller answers through is a model-specific register this
/// hypervisor never stops intercepting, so a write of it that moves the face is
/// a legal exit a guest may take in a loop — and each of these lines leaves
/// through a polled serial register taken with a machine-wide lock, from inside
/// the entry callback with both interrupt flags clear, which costs orders of
/// magnitude more than the exit that asked for it. Unlatched, that is a denial
/// of service rather than a diagnostic, and it is the shape every other
/// repeated report in this crate has already been given.
///
/// One line per controller per kind of transition is also the whole of what the
/// line is for: a bring-up reads as one activation per processor, and what the
/// line says needs no repeating to stay true.
fn transitioned(vlapic: &Vlapic, report: Report, what: fmt::Arguments) {
    if vlapic.diagnostics().say(report) {
        info!(
            "vlapic: {} {what}; later transitions of this kind on this controller are traced \
             rather than reported",
            vlapic.index()
        );
        return;
    }
    trace!("vlapic: {} {what}", vlapic.index());
}

/// Says this processor is in the guest, at the entry boundary.
///
/// The running bit is what another processor's IPI resolution reads before
/// it decides between the doorbell and the exit, so the publish happens
/// after everything the entry prepared and before the guest is entered; a
/// request that lands between the two is one the entry's own look, or the
/// VMRUN's re-evaluation, still finds.
///
/// Sequentially consistent, as the withdrawal and the rescan after it are: the
/// module header is where that ordering is argued, and it is the same argument
/// on both boundaries because the bit is one word two halves of one protocol
/// agree about.
///
/// Nothing at all on the erratum families: the bit stays clear for the life of
/// the machine there, and every directed IPI takes the exit it then cannot
/// avoid. Nothing either for a processor whose identifier the face being driven
/// cannot address: its entry is one the hardware never walks, so a running bit
/// set there would be a promise nothing reads, and a sender that read it back
/// would skip the wake the target does need. [`unpublish_running`] is
/// deliberately not gated the same way — see there.
///
/// # Every entry, where only the park boundary needs it
///
/// Correctness needs the park: for every other exit the processor is entered
/// again, and an entry re-evaluates the page — so a doorbell rung into a
/// host-side window is dropped and nothing waits on it. What the pair costs is
/// one locked read-modify-write each way, on a line the first eight entries of
/// the table share and every sender's hardware reads while it resolves a
/// destination; and it costs interrupts as well, because a peer that reads this
/// bit clear during a brief host window takes the exit-and-kick path rather
/// than writing a doorbell. What publishing once and clearing only at the park
/// would buy is those two prices, and what it would spend is a target reported
/// as running while it is in host code.
///
/// So which is cheaper is a measurement rather than an argument: how many kicks
/// a guest costs against how many exits it takes, which is what the exit census
/// now reports per processor. It stays as it is until that measurement exists,
/// because the state it would move to is only correct on the strength of the
/// re-evaluation above — and a wrong guess there is a parked processor with a
/// pending interrupt and nothing to wake it.
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
        .fetch_or(IS_RUNNING, Ordering::SeqCst);
    Ok(())
}

/// Says this processor is no longer in the guest, at the exit boundary.
///
/// The clear precedes every consultation of what has been left for it. A sender
/// whose hardware still reads the bit set deposits the request in this
/// processor's backing page and announces it to the physical processor named
/// beside it, which is no longer in a guest the acceleration is armed for: the
/// announcement is dropped there, the request bit survives in the page, and the
/// entry after this exit re-evaluates it. A sender that reads it clear has the
/// target reported as not running instead, and the rescan after this is what
/// finds whatever the kick that follows is for.
///
/// Sequentially consistent, and this is the half of that pairing the park
/// protocol turns on: this store and the first load of the rescan that follows
/// it are a store-then-load against a hardware reader doing the same to the
/// same two locations in the other order. The module header is where the
/// argument is, and [`deliverable`] is the rescan.
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
        .fetch_and(!IS_RUNNING, Ordering::SeqCst);
    Ok(())
}

/// Says a peer's hardware may resolve an interrupt to this processor, at the
/// transition that made the hardware the authority for its controller.
///
/// Delivery between the guest's processors is performed by the *sender's*
/// hardware: it indexes this table by the destination's guest identifier, and a
/// valid entry is what it deposits the request in the page named beside.
/// Nothing of the destination's own is consulted in that decision — so while
/// this processor's controller is the software's, an interrupt addressed to it
/// must not resolve here at all, and the bit follows the acceleration rather
/// than the table's construction. A sender's hardware then reports the
/// destination as invalid, which is the exit the software path completes the
/// command from, and no request is left in a page nothing reads.
///
/// Not gated on how far this processor's own face may walk the table, which is
/// the asymmetry with [`publish_running`]: the extent is the *sender's* control
/// block's, so a processor whose identifier its own face could not address is
/// still one a peer driven in the wider face resolves — and where the entry is
/// beyond every face's reach, a valid bit in it is read by nobody.
///
/// # Errors
///
/// [`VlapicError::Paging`] if the window does not reach the table, or
/// [`VlapicError::IdBeyondTable`] for an identifier the table does not hold —
/// which no processor with a backing page has, since describing one refused it.
fn publish_entry(activation: &Activation, vlapic: &Vlapic) -> Result<(), VlapicError> {
    activation
        .entry(vlapic.apic_id())?
        .fetch_or(IS_VALID, Ordering::Release);
    Ok(())
}

/// Says a peer's hardware may no longer resolve an interrupt to this processor,
/// at the transition that took the acceleration off its controller.
///
/// The withdrawal half of [`publish_entry`], which is where what the bit
/// promises is argued. Never gated on anything, for the reason
/// [`unpublish_running`] is not: a promise must be retractable wherever it may
/// have been made.
///
/// # Errors
///
/// As [`publish_entry`].
fn unpublish_entry(activation: &Activation, vlapic: &Vlapic) -> Result<(), VlapicError> {
    activation
        .entry(vlapic.apic_id())?
        .fetch_and(!IS_VALID, Ordering::Release);
    Ok(())
}

/// Whether the processor `id` is in the guest at this instant.
///
/// An Acquire load, which is all a decision made on the host side needs: the
/// publisher's read-modify-write is sequentially consistent and so includes the
/// release this pairs with, and nothing here is one half of the store-then-load
/// the module header argues about — the caller has just been told by the
/// hardware that the target was not running, and this narrows a wake rather
/// than deciding whether one is owed at all.
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

/// What this hypervisor believes the physical table holds at `index`.
///
/// For the one report that has to be readable against the table it accuses. The
/// hardware reporting an entry that names an unusable backing page is reporting
/// something about a structure nothing but this hypervisor writes, and the
/// value is what tells the two possibilities apart: an entry this hypervisor
/// never described, which is a table walked further than it was built for, or
/// one it described with a page the processor then refused.
///
/// # Errors
///
/// [`VlapicError::NotProvisioned`] before provisioning,
/// [`VlapicError::IdBeyondTable`] for an index the table does not hold, or
/// [`VlapicError::Paging`] if the window does not reach the table.
pub(crate) fn physical_entry(index: u16) -> Result<PhysicalApicEntry, VlapicError> {
    let activation = ACTIVATED.get().ok_or(VlapicError::NotProvisioned)?;
    let bits = activation
        .entry(ApicId::new(u32::from(index)))?
        .load(Ordering::Acquire);
    Ok(PhysicalApicEntry::from_bits(bits))
}

/// Sets a vector in this processor's backing request bits, with the trigger
/// mode the hardware classifies it by: the device path's delivery under
/// hardware-driven mode.
///
/// Both are Release and the request is published second, so that a processor
/// which observes the request bit cannot then read a trigger mode from before
/// it was set. The request is idempotent, so an arrival that races itself
/// coalesces exactly as the software path's does; answers whether the bit was
/// newly set.
///
/// What the trigger mode is for, and why the bank is written rather than added
/// to, is [`publish_request`]'s — which is also what keeps this and
/// [`Vlapic::accept`] from becoming two accounts of the same bank.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub(crate) fn request(vector: Vector, trigger: Trigger) -> Result<bool, VlapicError> {
    let activation = ACTIVATED.get().ok_or(VlapicError::NotProvisioned)?;
    let vlapic = current()?;
    let page = activation.page(vlapic.index().get())?;
    publish_request(vector, trigger, |offset| activation.word(page, offset))
}

/// Hands this processor's own interrupts to the hardware that is about to
/// deliver them, at an entry the acceleration is armed for.
///
/// The three vector banks are the backing page's alone while the hardware
/// drives, and this is what makes that true of the interrupts the *software*
/// path accepted: a request a peer's software delivery left in the model, an
/// error interrupt this controller raised on itself, whatever firmware's own
/// register file was seeded with. Each of them crosses into the page, and the
/// model stops holding it, so the hardware both delivers the vector and retires
/// it — out of one bank, against the priority it computes from that same page.
///
/// An injection could be neither. An interrupt put into the guest through the
/// control block's event field while the acceleration is on is one the hardware
/// never sees: it does not raise the priority the hardware arbitrates with, so
/// a lower-priority vector can be delivered on top of its handler, and the
/// guest's acknowledgement of it is performed against the page, where its bit
/// was never set — which leaves the model's own in-service bank holding it for
/// as long as the guest lives, refusing everything of that class or below in
/// every nomination afterwards.
///
/// What cannot cross stays and is injected, and it is one thing: an arrival
/// that came in through the pin that bypasses the controller. See
/// [`Handover::from_words`], which is where that is decided.
///
/// Cheap where there is nothing to do, which is every entry of a guest whose
/// interrupts are the hardware's already: four loads of this processor's own
/// register file per bank slot, and no store at all.
///
/// # Errors
///
/// As [`crate::read_msr`]. A failure leaves the interrupt in the model, where
/// the entry's own nomination is what delivers it.
pub(crate) fn hand_over() -> Result<(), VlapicError> {
    let activation = ACTIVATED.get().ok_or(VlapicError::NotProvisioned)?;
    let vlapic = current()?;
    let page = activation.page(vlapic.index().get())?;
    for (slot, within) in (0..)
        .step_by(REGISTER_STRIDE as usize)
        .enumerate()
        .take(SLOTS)
    {
        let handover = Handover::of(vlapic, slot);
        if handover.is_empty() {
            continue;
        }
        handover.publish(within, |offset| activation.word(page, offset))?;
        handover.retire(vlapic, slot);
    }
    Ok(())
}

/// Whether this processor's backing page holds anything its guest could take
/// at this instant.
///
/// The second half of the park protocol: asked after the running bit was
/// withdrawn and before the processor parks, with sequentially consistent loads
/// so that a request published before the withdrawal is one this scan sees. The
/// module header is where that ordering is argued; every load of this scan
/// takes part in it rather than only the first, because one ordering for the
/// whole scan is one policy where an exception would be a second. On this
/// processor they are the plain moves an Acquire load already was. The
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
        .load(Ordering::SeqCst);
    if spurious & SOFTWARE_ENABLE == 0 {
        return Ok(false);
    }
    let task = activation
        .word(page, Register::TASK_PRIORITY.offset())?
        .load(Ordering::SeqCst);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "only the low byte of the task priority register carries meaning"
    )]
    let task = task as u8;
    let servicing = activation.highest(page, Register::IN_SERVICE, Ordering::SeqCst)?;
    let processor = priority::processor_priority(Priority::new(task), servicing);
    Ok(activation
        .highest(page, Register::INTERRUPT_REQUEST, Ordering::SeqCst)?
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
        // The one trap whose value is not in the page: what the guest wrote is
        // not a value at all, and what is owed is the release of whatever real
        // hardware is holding for the vector the hardware named.
        Register::END_OF_INTERRUPT => end_of_interrupt(vlapic, Acknowledged::Trapped(exit_vector)),
        // A command reported through this exit is a command the hardware did
        // *not* accelerate — the trap is raised where the acceleration ran out —
        // so the delivery is the software path's whole business and there is no
        // second attempt to collide with. The value is the page's two halves,
        // which is where the trap left the guest's own write and the only place
        // it is: this exit reports an offset rather than a value.
        //
        // Which is what the intercepted write of the same register does with it,
        // and the two doors to it have to agree. A trap answered with the
        // delivery-status bit alone drops the interrupt outright, and clearing
        // that bit is what stops the guest even waiting for it.
        Register::COMMAND_LOW => {
            let activation = ACTIVATED.get().ok_or(VlapicError::NotProvisioned)?;
            let page = activation.page(vlapic.index().get())?;
            let half = |register: Register| -> Result<u32, VlapicError> {
                Ok(activation
                    .word(page, register.offset())?
                    .load(Ordering::Acquire))
            };
            let command =
                Command::from_halves(half(Register::COMMAND_LOW)?, half(Register::COMMAND_HIGH)?);
            complete_command(command.bits())
        }
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
            end_of_interrupt(vlapic, Acknowledged::Intercepted)?;
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
        // architecture's check of the vector performed first. It is an edge
        // whatever else the vector has been: this face's register carries no
        // trigger mode, and the interrupt is complete once taken.
        Register::SELF_IPI => {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "a vector is the low eight bits of the register it is written in"
            )]
            let vector = Vector::new(value as u8);
            if priority::legal(vector) {
                request(vector, Trigger::Edge)?;
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

/// Stops driving this controller in hardware from inside an exit, handing the
/// page's state back to the model first.
///
/// What [`disable`] performs at an entry, performed at the moment a caller
/// stops trusting the acceleration: the access that discovered the trouble is
/// about to be answered out of the model, and the model is the nearest true
/// state only once it holds what the page does. Without the carry it is not the
/// nearest anything for two of its registers — a task priority the guest set
/// through the page is not in it, so a write the model then accepts is reverted
/// by the next entry's own carry-back; and an acknowledgement pops an
/// in-service bank that has not been the guest's since the acceleration was
/// turned on.
///
/// A page belonging to a life that has ended is not carried, for the reason
/// [`disable`] gives: a model a reset has just cleared is already the nearest
/// true state, and putting the page back would restore exactly what the reset
/// was required to destroy.
///
/// The demotion itself happens whether or not the carry could, because a claim
/// that cannot reach the page is one the processor stops making either way; the
/// control block's enable bits follow at the next entry.
///
/// # Errors
///
/// As [`crate::read_msr`], from the carry alone.
pub(crate) fn hand_back(vlapic: &Vlapic) -> Result<(), VlapicError> {
    let carried = ACTIVATED
        .get()
        .ok_or(VlapicError::NotProvisioned)
        .and_then(|activation| {
            let standing = Standing::of(
                activation.rebuilt_at[vlapic.index().get()].load(Ordering::Relaxed),
                vlapic.epoch(),
            );
            if standing.life.carried() {
                sync_into_model(activation, vlapic)?;
            }
            Ok(())
        });
    vlapic.inhibit_avic();
    carried
}

/// Demotes the whole machine, for the rest of its life.
///
/// For the reports that say the acceleration itself cannot be trusted:
/// destinations resolving somewhere the guest did not name them, or an exit
/// the architecture has no reason to raise.
///
/// The exchange is Relaxed, as every access to the flag is: what it orders is
/// nothing but the line below, and the flag itself is the whole of what any
/// other processor reads.
pub(crate) fn inhibit_machine(reason: &str) {
    let Some(activation) = ACTIVATED.get() else {
        return;
    };
    if !activation.machine_inhibited.swap(true, Ordering::Relaxed) {
        warn!("vlapic: hardware delivery inhibited for the machine: {reason}");
    }
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

/// Which of the two copies of the guest's task priority an exit owes the model.
///
/// The register is one the guest reaches through two doors, and only one of
/// them is open at a time. While the hardware drives the controller the guest
/// writes the backing page with no exit at all, and the whole byte it wrote is
/// there; otherwise the processor keeps the guest's writes to its task-priority
/// control register in the control block, four bits of them, and that is the
/// only place they are.
///
/// Never both. The model is consulted between the exit and the next entry — the
/// park decision asks it whether this processor may sleep — so a task priority
/// written from one authority and then overwritten from the other is right only
/// until whichever of the two runs next. Too low, and the processor wakes for a
/// vector the guest's real priority masks, re-enters, finds the truth restored
/// and halts again, without bound. Too high, and it sleeps through one the
/// guest would have taken.
///
/// Pure rather than a branch inside the caller, because which authority owns a
/// register is exactly the kind of thing that has to be readable against the
/// states it covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TaskPriority {
    /// The backing page's word, which the guest wrote without exiting.
    Backing,
    /// The control block's class, which is what the processor kept for it.
    Block,
}

impl TaskPriority {
    /// Which of the two the guest was running under.
    ///
    /// The control block's own enable bit is the term rather than whether the
    /// acceleration is permitted, because what owned the register is whatever
    /// the processor was entered with — which after a reconciliation that could
    /// not finish is not the same answer.
    pub(crate) const fn owner(accelerated: bool) -> Self {
        if accelerated {
            Self::Backing
        } else {
            Self::Block
        }
    }
}

/// Copies the backing task priority into the model.
///
/// The one register the guest moves without exiting, so nothing that consults
/// the model about what is deliverable is looking at the number the hardware is
/// looking at until this has run.
///
/// Called at the *exit* boundary — [`crate::observe_task_priority`] is where
/// the choice between this and the control block's copy is made — so the model
/// is right for the whole host-side window rather than only after the next
/// entry's reconciliation. The park decision falls in that window, and an entry
/// is precisely what it is deciding whether to make.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub(crate) fn sync_task_priority(vlapic: &Vlapic) -> Result<(), VlapicError> {
    let activation = ACTIVATED.get().ok_or(VlapicError::NotProvisioned)?;
    let page = activation.page(vlapic.index().get())?;
    let value = activation
        .word(page, Register::TASK_PRIORITY.offset())?
        .load(Ordering::Relaxed);
    vlapic.set_task_priority(value);
    Ok(())
}

/// Brings the logical table into agreement with this controller's model, at the
/// exit that moved what the table is a projection of.
///
/// The second door to [`mirror_logical`], and the one the *software* path uses.
/// Two writes reach it: the logical identity itself, and the face that decides
/// whether this controller has an entry in that table at all. Both are answered
/// here whether the hardware trapped the write or the permission map kept it,
/// which is what makes the table follow the model rather than the acceleration
/// — without it a controller whose interrupts are the software's reprograms its
/// identity with no entry moved, and the table goes on naming it for an
/// identifier it has given up, which is also what makes a second processor
/// legitimately taking that identifier look like an alias.
///
/// The face is read here rather than given, because this is not a transition:
/// nothing above has decided anything about the acceleration, so the face the
/// hardware drives this controller in is whatever it is at the instant the
/// write is answered.
///
/// Answers "nothing to do" where the structures do not exist, which is every
/// machine the policy left on the software path.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub(crate) fn observe_logical_identity(vlapic: &Vlapic) -> Result<(), VlapicError> {
    if !provisioned() {
        return Ok(());
    }
    mirror_logical(vlapic, active_face(vlapic))
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

/// Completes the host's half of an acknowledgement the guest gave its
/// controller while the hardware was driving it.
///
/// What is owed is the release of whatever real hardware is holding for the
/// vector, and — where the hardware performed no part of the write — the
/// retirement of the in-service bit the page holds. Which vector that is comes
/// from the backing in-service bank where the delivery recorded it, reconciled
/// with what the hardware says it already did: see [`eoi_vector`].
///
/// The controller is resolved before anything is retired. The guest's own bit
/// is the record that an acknowledgement is still to come, and consuming it and
/// then failing to reach the controller would leave the withheld physical
/// acknowledgement owed with nothing left that could ever produce another one —
/// the line stays asserted for the life of the machine.
///
/// The in-service bit itself is cleared idempotently: whichever of the hardware
/// and this cleared it first, the second clear is a no-op, and nothing here
/// retires anything twice.
fn end_of_interrupt(vlapic: &Vlapic, acknowledged: Acknowledged) -> Result<(), VlapicError> {
    let activation = ACTIVATED.get().ok_or(VlapicError::NotProvisioned)?;
    let page = activation.page(vlapic.index().get())?;
    let top = activation.highest(page, Register::IN_SERVICE, Ordering::Acquire)?;
    let owed = top.is_some_and(|vector| vlapic.ledger().owes(vector));
    let Some(vector) = eoi_vector(top, acknowledged, owed) else {
        // An EOI with nothing in service is the architectural no-op, and a
        // guest is entitled to make one.
        return Ok(());
    };
    let local = apic::local()?;
    if top == Some(vector) {
        let (within, bit) = bank_bit(vector);
        activation
            .word(page, Register::IN_SERVICE.offset() + within)?
            .fetch_and(!bit, Ordering::AcqRel);
    }
    vlapic.ledger().release(vector, &local);
    Ok(())
}

/// What the hardware had already done with the guest's acknowledgement by the
/// time the host got it.
///
/// The two doors an acknowledgement reaches the host through, and they say
/// different things about the same in-service bank — which is why the vector is
/// decided from this rather than from the presence of a reported one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Acknowledged {
    /// The permission map kept the access, so the hardware performed no part of
    /// the write: the bank is exactly as the delivery left it.
    Intercepted,
    /// The hardware trapped the write because the vector being retired is level
    /// triggered, reporting the vector it saw in service — or, on hardware that
    /// leaves the field alone, reporting nothing usable.
    Trapped(Option<u8>),
}

/// Which vector an acknowledgement retired, given the in-service bank, what the
/// hardware had already done and whether the bank's top is a vector real
/// hardware is owed an acknowledgement for.
///
/// An intercepted write is the unambiguous one: nothing has happened yet, so
/// the bank's top is the vector the guest is acknowledging, exactly as it is
/// for [`Vlapic::end_of_interrupt`] on the software path — and an empty bank is
/// the architectural no-op.
///
/// A trapped write is the write the hardware already performed some part of,
/// and what it reports is the vector it saw in service. Where that disagrees
/// with the bank, the hardware retired the bank's top before it exited and the
/// report is the only place the vector still is.
///
/// # A trap that reports nothing is the one ambiguous case, and the ledger
/// decides it
///
/// The architecture defines the field, and hardware that leaves it alone is not
/// ruled out; a trap carrying nothing usable is then either a vector the
/// hardware has not retired — the bank's top, which is what should be retired —
/// or one it has, in which case the bank's top is the *next* interrupt down and
/// the guest is still servicing it. Nothing in the exit tells the two apart.
///
/// The ledger does, because a trap happens only where a trigger-mode bit is set
/// and this hypervisor sets one only for an arrival whose physical
/// acknowledgement it is withholding. So a top the ledger is holding a debt for
/// is the acknowledged vector, and one it is not is a vector nothing is owed
/// for — where acting would retire an interrupt underneath a guest that is
/// still servicing it, and declining costs nothing that was owed.
fn eoi_vector(
    in_service_top: Option<Vector>,
    acknowledged: Acknowledged,
    owed: bool,
) -> Option<Vector> {
    match acknowledged {
        Acknowledged::Intercepted => in_service_top,
        Acknowledged::Trapped(reported) => reported
            .map(Vector::new)
            .filter(|vector| priority::legal(*vector))
            .or_else(|| in_service_top.filter(|_| owed)),
    }
}

/// Brings the logical table into agreement with what this controller's model
/// says its logical identity is.
///
/// The table is a projection of the model exactly as the backing page is, with
/// one difference that decides its whole lifecycle: a *peer's* hardware reads
/// it. Whichever processor is resolving a logically addressed interrupt walks
/// it, whatever face the controller it resolves to is in — so it may not be
/// left describing a model that has moved on, and it is settled wherever either
/// of its two terms moves. The identity moves at the exit that answers a
/// guest's write of it, whichever door that write came through; the face moves
/// at the transition that arms, disarms or changes the acceleration.
///
/// `face` is the face the acceleration drives this controller in, given rather
/// than read again: a caller part-way through a transition has already decided
/// it, and a second reading would be a second answer to the question the
/// transition is performing.
///
/// # Withdrawn first, and published only where the slot is this controller's
/// alone
///
/// The withdrawal is unconditional, so the window in between names this
/// processor nowhere rather than twice, and so a controller that has left the
/// older face or lost the acceleration stops being named at all.
///
/// The entry is then published only where no other controller's model answers
/// to the same slot. One entry names one processor, so two controllers claiming
/// one logical identity is a destination this table cannot express — and rather
/// than let the later writer overwrite the earlier one, which resolves that
/// destination to whichever wrote last and silently strands the other, the slot
/// is taken *out* of the table. An entry the hardware finds invalid is an exit,
/// and the software path then completes the command against each controller's
/// own registers, which match both of them. That costs one exit per interrupt
/// addressed there for as long as both claim it, and nothing else: the rest of
/// the machine goes on being delivered by the hardware.
///
/// Whole aligned words are what move, so a resolution racing this reads the old
/// entry, the new one, or no entry — a delivery, a delivery, or an exit the
/// handlers complete. Which of the three it cannot be is two entries naming
/// this processor, and [`publish_logical`] is where the ordering that rules
/// that out is argued.
///
/// # Errors
///
/// [`VlapicError::NotProvisioned`] before provisioning,
/// [`VlapicError::NotInstalled`] before the controllers exist, or
/// [`VlapicError::Paging`] if the window does not reach the table.
fn mirror_logical(vlapic: &Vlapic, face: Option<Face>) -> Result<(), VlapicError> {
    let activation = ACTIVATED.get().ok_or(VlapicError::NotProvisioned)?;
    let record = &activation.logical_slots[vlapic.index().get()];
    let wanted = entry_slot(
        face,
        vlapic.logical_destination(Mode::XApic),
        vlapic.destination_format(),
    );
    let entry = |slot: usize| activation.logical(slot);
    // Host interrupts off for the whole of the critical section, on this path as
    // on the transitions that reach it with both flags already clear: a holder
    // that could be interrupted is every spinner stalled for as long as the
    // handler runs, and one of those spinners is inside an entry callback and
    // answering nothing. What the section may contain for that to be safe is
    // three rules, and they are stated where the lock is declared.
    let settled = interrupts::without_interrupts(
        || -> Result<(Publish, Option<&'static Vlapic>), VlapicError> {
            let _guard = activation.logical_lock.lock();
            withdraw(record, entry)?;
            // Asked of the other controllers' models rather than of their
            // published slots, because a controller the hardware does not drive
            // still answers to its logical identifier — the software delivers to
            // it — and an entry published for the one of them the hardware can
            // reach is one the hardware resolves with no exit at all, leaving
            // the other with nothing.
            let claimant = match wanted {
                None => None,
                Some(slot) => registry::lapics()?
                    .all()
                    .iter()
                    .find(|peer| peer.index() != vlapic.index() && claims(peer, slot)),
            };
            let published = publication(wanted, claimant.is_some());
            match published {
                Publish::Nothing => {}
                Publish::Take(slot) => publish_logical(record, slot, vlapic.apic_id(), entry)?,
                Publish::Contested(slot) => disclaim(&activation.logical_slots, slot, entry)?,
            }
            Ok((published, claimant))
        },
    )?;
    // Outside the lock, and outside the window the interrupts were held off for:
    // a byte of serial output leaves through a polled register and every other
    // processor that logs waits behind the same lock while it goes out.
    if let (Publish::Contested(slot), Some(peer)) = settled {
        say_aliased(peer.index(), vlapic.index(), slot);
    }
    Ok(())
}

/// The logical-table slot a controller has an entry in, out of the face the
/// acceleration drives it in and the identity its model holds.
///
/// The whole of the table's lifecycle as one decision, which is why it is a
/// function of values: only the older face is resolved through this table at
/// all — the wider one derives a logical identifier from the processor's own
/// identifier and never reads the table — so a controller driven in the wider
/// face, or driven in neither, has no entry whatever identity it holds. Which
/// makes a deactivation, a demotion, a guest disabling its controller and a
/// move into the wider face one answer rather than four.
///
/// `ldr` and `dfr` are the older face's own registers, that face being the only
/// one whose identity this table can express.
fn entry_slot(face: Option<Face>, ldr: u32, dfr: u32) -> Option<usize> {
    match face {
        Some(Face::XAvic) => logical_slot(ldr, dfr),
        Some(Face::X2Avic) | None => None,
    }
}

/// Whether one controller's model answers to the logical-table slot `slot`.
///
/// The acceleration is deliberately not a term, and that asymmetry with
/// [`entry_slot`] is the point: whether a controller has an *entry* is a
/// question about what the hardware may deliver to it, and whether it *claims*
/// a slot is a question about what the guest addressed — which a controller
/// answers whether its interrupts are the hardware's or the software's. A slot
/// published for the accelerated one of two claimants is a slot the hardware
/// resolves without an exit, and the software claimant never hears about the
/// interrupt at all.
fn claims(vlapic: &Vlapic, slot: usize) -> bool {
    matches!(vlapic.mode(), Mode::XApic)
        && logical_slot(
            vlapic.logical_destination(Mode::XApic),
            vlapic.destination_format(),
        ) == Some(slot)
}

/// What a controller's identity owes the logical table, beyond the withdrawal
/// every settle begins with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Publish {
    /// Nothing. The controller answers to no logical destination this table can
    /// express, or the hardware does not drive it in the face the table is read
    /// in.
    Nothing,
    /// The slot is this controller's alone, and the entry names this processor.
    Take(usize),
    /// Another controller's model answers to the same slot. One entry names one
    /// processor, so the slot leaves the table instead, and every interrupt
    /// addressed there is resolved by the software path — which matches both.
    Contested(usize),
}

/// What the table owes, out of the slot a controller wants and whether another
/// controller answers to that slot too.
///
/// Pure rather than a pair of branches inside the settle, because refusing to
/// publish is what keeps a live entry from being overwritten, and a decision of
/// values can be read against the three states it covers.
const fn publication(wanted: Option<usize>, claimed_by_another: bool) -> Publish {
    match wanted {
        None => Publish::Nothing,
        Some(slot) if claimed_by_another => Publish::Contested(slot),
        Some(slot) => Publish::Take(slot),
    }
}

/// Withdraws whatever entry a processor's record names, and stops recording
/// one.
///
/// `entry` reaches one slot of the table, for the reason
/// [`ResetImage::publish`] takes the same shape: the sequence is then
/// exercisable over an array of words rather than over a frame a window has to
/// reach.
///
/// The entry is invalidated *before* the record of it is cleared, and that
/// order is what makes a failure safe. A table the window cannot reach leaves
/// the entry valid — and leaves the record still naming it, so the next
/// withdrawal tries again and another controller taking that identity is still
/// seen taking it. Clearing the record first would leave a valid entry nothing
/// knows about and nothing will ever clear.
///
/// # Errors
///
/// [`VlapicError::Paging`] if the window does not reach the table.
fn withdraw<'a>(
    record: &AtomicU16,
    entry: impl Fn(usize) -> Result<&'a AtomicU32, VlapicError>,
) -> Result<(), VlapicError> {
    let previous = record.load(Ordering::Relaxed);
    if previous == NO_SLOT {
        return Ok(());
    }
    entry(usize::from(previous))?.fetch_and(!LOGICAL_VALID, Ordering::Relaxed);
    record.store(NO_SLOT, Ordering::Relaxed);
    Ok(())
}

/// Publishes `slot` as the entry naming the processor `id`, and records it.
///
/// Recorded after the store for the reason [`withdraw`] clears its record after
/// the invalidate: the record and the entry are one fact, and a failure must
/// leave the record describing the table rather than an intention.
///
/// # The store is Release, for a reader that is not a thread
///
/// The invariant it buys is the table's own: a processor's old entry is
/// invalidated before its new one appears, so nothing resolving a logical
/// destination ever finds two entries naming one processor. Those are stores to
/// two different words, and Relaxed constrains only each word's own
/// modification order — so nothing in the model would keep them in that order,
/// and the lock above does not help. It orders this processor against other
/// *processors*, and the reader here is the AVIC hardware of any core, which
/// takes no lock and participates in no happens-before. A Release store is what
/// forbids the earlier invalidate being moved after this one, and it is the
/// whole of what is needed: the record store below is read only under the lock.
///
/// # Errors
///
/// As [`withdraw`].
fn publish_logical<'a>(
    record: &AtomicU16,
    slot: usize,
    id: ApicId,
    entry: impl Fn(usize) -> Result<&'a AtomicU32, VlapicError>,
) -> Result<(), VlapicError> {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the logical table's identity field is eight bits wide, and xAVIC identifiers are"
    )]
    let named = LogicalApicEntry::new()
        .with_guest_apic_id(id.get() as u8)
        .with_valid(true);
    entry(slot)?.store(named.into_bits(), Ordering::Release);
    record.store(recorded(slot), Ordering::Relaxed);
    Ok(())
}

/// Takes a slot out of the table that two controllers' models both answer to,
/// and stops whichever processor held it being recorded as holding it.
///
/// The entry goes rather than either claim, because both claims are legitimate:
/// the two controllers really do answer to that logical identifier, and the
/// table can name one of them. An invalid entry is what makes the hardware exit
/// and the software path deliver to both.
///
/// # Errors
///
/// As [`withdraw`].
fn disclaim<'a>(
    records: &[AtomicU16],
    slot: usize,
    entry: impl Fn(usize) -> Result<&'a AtomicU32, VlapicError>,
) -> Result<(), VlapicError> {
    entry(slot)?.fetch_and(!LOGICAL_VALID, Ordering::Relaxed);
    let held = recorded(slot);
    for record in records {
        if record.load(Ordering::Relaxed) == held {
            record.store(NO_SLOT, Ordering::Relaxed);
        }
    }
    Ok(())
}

/// A slot as the record of one holds it.
///
/// One statement of the narrowing, because two would be two spellings of the
/// same number.
#[expect(
    clippy::cast_possible_truncation,
    reason = "a slot is below sixty, which sixteen bits hold with room to spare"
)]
const fn recorded(slot: usize) -> u16 {
    slot as u16
}

/// Says once per machine that two of the guest's processors answer to one
/// logical destination, naming both and the entry they claim.
///
/// Once for the machine rather than once per controller, because the table is
/// the machine's: a guest that alternates two identity writes would otherwise
/// produce a line per pair of stores, each taken with a machine-wide lock held.
/// What it reports needs no repeating to stay true — the slot is out of the
/// table for as long as both claim it, and every interrupt addressed there is
/// delivered by the software path.
fn say_aliased(claimant: CpuIndex, refused: CpuIndex, slot: usize) {
    if SAID_ALIASED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
    {
        warn!(
            "vlapic: {refused} answers to the same logical destination as {claimant}, which one \
             table entry cannot name: logical entry {slot} is taken out of the table, and \
             interrupts addressed there are delivered by the software path"
        );
    }
}

/// Whether two processors claiming one logical destination has been reported.
static SAID_ALIASED: AtomicBool = AtomicBool::new(false);

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

#[cfg(test)]
mod tests {
    //! The decisions here that need no machine: which face the acceleration
    //! drives a controller in, which face a control block's own enable bits say
    //! it is driving, how far it reaches in each, which vector an
    //! acknowledgement retires and which of the two doors it came through,
    //! which logical destinations name a table entry, whether a controller has
    //! one at all and what a slot two of them claim becomes, which move an
    //! entry owes the acceleration, which life the page it holds belongs
    //! to, which of its slots a trapped write leaves owing the model's
    //! answer, which authority an exit owes the guest's task priority to,
    //! and what the two read-modify-writes over a physical-table entry
    //! leave of the rest of it.
    //!
    //! The logical table's own three operations are here too, and they are the
    //! one thing in this file that touches a structure rather than deciding
    //! something. They reach it through a closure, so a test supplies an array
    //! of words in place of the frame — which is what makes the order of a
    //! withdrawal's two steps, and what a failure between them leaves behind,
    //! something a test can observe.

    use alloc::vec::Vec;
    use core::{
        array::from_fn,
        iter::once,
        sync::atomic::{AtomicU16, AtomicU32, Ordering},
    };

    use descriptors::Vector;
    use svm::avic::{MAX_PHYSICAL_ID, PhysicalApicEntry};
    use x86_64::PhysAddr;

    use super::{
        Acknowledged, ApicId, Entry, FLAT_DESTINATION_FORMAT, Face, IS_RUNNING, IS_VALID,
        LOGICAL_ENTRIES, LOGICAL_VALID, Life, Mode, Move, NO_SLOT, Publish, Register, Standing,
        TaskPriority, VlapicError, Written, disclaim, driven_face, entry_slot, eoi_vector,
        face_bits, face_limit, logical_slot, mirrored, move_for, publication, publish_logical,
        withdraw,
    };

    /// Every mode a controller can be in, which is the one term of the
    /// activation gate that is not a boolean.
    const MODES: [Mode; 3] = [Mode::Disabled, Mode::XApic, Mode::X2Apic];

    /// A flat logical identifier naming the third entry of the table.
    const FLAT_THIRD: u32 = 0x0400_0000;

    /// A table a test can hold: the same aligned words the window would have
    /// reached, in an array.
    fn table() -> [AtomicU32; LOGICAL_ENTRIES] {
        from_fn(|_| AtomicU32::new(0))
    }

    /// A window that does not reach the table, which is the one failure these
    /// operations have.
    fn unreached(_: usize) -> Result<&'static AtomicU32, VlapicError> {
        Err(VlapicError::NotProvisioned)
    }

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
    fn an_intercepted_acknowledgement_retires_what_the_bank_holds() {
        // The hardware performed no part of the write, so the bank is exactly as
        // the delivery left it and its top is what the guest is acknowledging —
        // the same answer the software path's own acknowledgement gives, and it
        // does not depend on a debt: every guest EOI in this face arrives here
        // wherever the permission map is keeping the register.
        let top = Some(Vector::new(0x42));
        for owed in [false, true] {
            assert_eq!(eoi_vector(top, Acknowledged::Intercepted, owed), top);
            assert_eq!(eoi_vector(None, Acknowledged::Intercepted, owed), None);
        }
    }

    #[test]
    fn a_trapped_acknowledgement_retires_the_vector_the_hardware_named() {
        // Whatever the bank says. Where the two agree the report is the top; where
        // they disagree the hardware retired the top before it exited and the
        // report is the only place the vector still is. The ledger is not a term
        // of either: the hardware has named the vector.
        let top = Some(Vector::new(0x42));
        for owed in [false, true] {
            assert_eq!(
                eoi_vector(top, Acknowledged::Trapped(Some(0x42)), owed),
                top
            );
            assert_eq!(
                eoi_vector(
                    Some(Vector::new(0x31)),
                    Acknowledged::Trapped(Some(0x42)),
                    owed
                ),
                Some(Vector::new(0x42))
            );
            assert_eq!(
                eoi_vector(None, Acknowledged::Trapped(Some(0x42)), owed),
                Some(Vector::new(0x42))
            );
        }
    }

    #[test]
    fn a_trap_that_reports_nothing_retires_only_a_vector_hardware_is_owed_for() {
        // The one ambiguous case, and the ledger is what decides it. A report of
        // an illegal vector is a report of nothing, exactly as an absent one is:
        // both are hardware leaving the field alone rather than naming a vector.
        //
        // With a debt, the bank's top is the vector the trap was raised for — a
        // trigger-mode bit is set only where an acknowledgement is being withheld.
        // Without one, the top is either an interrupt the guest is still
        // servicing, which acting on would retire underneath it, or a vector
        // nothing is owed for, where declining costs nothing.
        let top = Some(Vector::new(0x42));
        for reported in [None, Some(0x00), Some(0x05)] {
            assert_eq!(eoi_vector(top, Acknowledged::Trapped(reported), true), top);
            assert_eq!(
                eoi_vector(top, Acknowledged::Trapped(reported), false),
                None
            );
            assert_eq!(
                eoi_vector(None, Acknowledged::Trapped(reported), true),
                None
            );
        }
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
    fn a_controller_has_a_table_entry_only_where_the_hardware_drives_it_in_the_older_face() {
        // The table's whole lifecycle, read against the state it is decided from:
        // the mode, the two demotions, the policy's answer about the wider face,
        // and the identity the model holds. Only the older face is resolved
        // through this table, so a deactivation, a demotion of this processor, a
        // demotion of the machine, a guest switching its controller off and a
        // move into the wider face are one answer rather than five — no entry.
        for mode in MODES {
            for machine_inhibited in [false, true] {
                for x2avic in [false, true] {
                    for inhibited in [false, true] {
                        let face = driven_face(machine_inhibited, mode, x2avic, inhibited);
                        assert_eq!(
                            entry_slot(face, FLAT_THIRD, FLAT_DESTINATION_FORMAT),
                            (face == Some(Face::XAvic)).then_some(2),
                            "{mode:?}, machine inhibited {machine_inhibited}, x2avic {x2avic}, \
                             inhibited {inhibited}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn an_entry_in_the_older_face_is_whichever_slot_the_identity_names() {
        // The other half of the same decision: the face admits an entry and the
        // identity decides which one, or that there is none. Leaving the older
        // face withdraws it whatever the identity is, and returning publishes it
        // again from the same two registers — which is why the face is a term
        // here rather than a reason to keep a second copy of the slot anywhere.
        for (ldr, dfr, slot) in [
            (0x0100_0000, FLAT_DESTINATION_FORMAT, Some(0)),
            (0x8000_0000, FLAT_DESTINATION_FORMAT, Some(7)),
            (0x2100_0000, 0x0FFF_FFFF, Some(8)),
            // Two bits, and none: an identity the table cannot express, in a face
            // that would otherwise have an entry.
            (0x0300_0000, FLAT_DESTINATION_FORMAT, None),
            (0, FLAT_DESTINATION_FORMAT, None),
        ] {
            assert_eq!(
                entry_slot(Some(Face::XAvic), ldr, dfr),
                slot,
                "{ldr:#x}, {dfr:#x}"
            );
            assert_eq!(entry_slot(Some(Face::X2Avic), ldr, dfr), None);
            assert_eq!(entry_slot(None, ldr, dfr), None);
        }
    }

    #[test]
    fn a_slot_another_controller_answers_to_is_taken_out_rather_than_taken_over() {
        // The decision that keeps a live entry from being overwritten. One entry
        // names one processor, so the later writer neither wins nor loses the
        // slot: it leaves the table, the hardware exits on it, and the software
        // path completes the command against both controllers' own registers.
        // Overwriting instead resolved the destination to whichever wrote last
        // and left the other with nothing.
        assert_eq!(publication(Some(3), false), Publish::Take(3));
        assert_eq!(publication(Some(3), true), Publish::Contested(3));
        // Nothing to publish is nothing to contest.
        for claimed in [false, true] {
            assert_eq!(publication(None, claimed), Publish::Nothing);
        }
    }

    #[test]
    fn a_withdrawal_that_cannot_reach_the_table_leaves_its_entry_accounted_for() {
        // The invalidate and the record of it are two steps, and their order is
        // what a failure between them is judged by. A record cleared first would
        // leave a valid entry nothing knows about, nothing will ever clear, and
        // no check of who claims a slot can see — so a second processor taking
        // that identity would not be noticed.
        let entries = table();
        let reach = |slot: usize| entries.get(slot).ok_or(VlapicError::NotProvisioned);
        let record = AtomicU16::new(5);
        entries[5].store(LOGICAL_VALID | 0x07, Ordering::Relaxed);

        assert!(withdraw(&record, unreached).is_err());
        assert_eq!(record.load(Ordering::Relaxed), 5);
        assert_eq!(entries[5].load(Ordering::Relaxed), LOGICAL_VALID | 0x07);

        // Reached, both steps happen: the entry stops being valid and the record
        // stops naming it. The identifier beside the bit is left alone, because
        // an invalid entry is not a slot that has to be blanked.
        assert!(withdraw(&record, reach).is_ok());
        assert_eq!(record.load(Ordering::Relaxed), NO_SLOT);
        assert_eq!(entries[5].load(Ordering::Relaxed), 0x07);

        // And a processor with no entry has nothing to withdraw, so it cannot
        // fail even where the table is out of reach.
        assert!(withdraw(&record, unreached).is_ok());
    }

    #[test]
    fn publishing_names_this_processor_and_records_where_it_was_named() {
        let entries = table();
        let reach = |slot: usize| entries.get(slot).ok_or(VlapicError::NotProvisioned);
        let record = AtomicU16::new(NO_SLOT);

        assert!(publish_logical(&record, 9, ApicId::new(0x21), reach).is_ok());
        assert_eq!(entries[9].load(Ordering::Relaxed), LOGICAL_VALID | 0x21);
        assert_eq!(record.load(Ordering::Relaxed), 9);

        // The record follows the store for the reason the withdrawal's clear
        // follows its invalidate: a failure leaves the record describing the
        // table rather than an intention.
        assert!(publish_logical(&record, 9, ApicId::new(0x21), unreached).is_err());
        assert_eq!(record.load(Ordering::Relaxed), 9);
    }

    #[test]
    fn a_contested_slot_leaves_the_table_and_nobody_is_recorded_as_holding_it() {
        // What two controllers claiming one logical destination becomes. The
        // incumbent's entry is not replaced and the newcomer's is not published:
        // the slot is invalid, which is an exit rather than a delivery to one of
        // them, and no record claims an entry that is no longer there.
        let entries = table();
        let reach = |slot: usize| entries.get(slot).ok_or(VlapicError::NotProvisioned);
        let records = [
            AtomicU16::new(3),
            AtomicU16::new(NO_SLOT),
            AtomicU16::new(7),
        ];
        entries[3].store(LOGICAL_VALID | 0x01, Ordering::Relaxed);
        entries[7].store(LOGICAL_VALID | 0x02, Ordering::Relaxed);

        assert!(disclaim(&records, 3, reach).is_ok());
        assert_eq!(entries[3].load(Ordering::Relaxed), 0x01);
        assert_eq!(records[0].load(Ordering::Relaxed), NO_SLOT);
        // Nobody else is disturbed: one slot leaves the table, not the table.
        assert_eq!(records[1].load(Ordering::Relaxed), NO_SLOT);
        assert_eq!(records[2].load(Ordering::Relaxed), 7);
        assert_eq!(entries[7].load(Ordering::Relaxed), LOGICAL_VALID | 0x02);
    }

    #[test]
    fn an_entry_owes_nothing_where_both_sides_agree() {
        assert_eq!(move_for(None, None), Move::Idle);
        for face in [Face::XAvic, Face::X2Avic] {
            assert_eq!(move_for(Some(face), Some(face)), Move::Steady);
        }
    }

    #[test]
    fn a_blocks_two_enable_bits_name_the_face_the_hardware_is_driving() {
        // The *have* predicate, which every decision about the run that is
        // happening is made from: what the entry prepared, which authority
        // answered a register access, which copy of the task priority the guest
        // was running under. The base bit is the whole of the answer and the wider
        // bit only selects between the faces.
        assert_eq!(face_bits(true, false), Some(Face::XAvic));
        assert_eq!(face_bits(true, true), Some(Face::X2Avic));
        assert_eq!(face_bits(false, false), None);
    }

    #[test]
    fn the_one_encoding_the_architecture_rejects_is_driven_in_neither_face() {
        // The wider bit without the base bit: a block the processor refuses
        // outright, with no guest instruction executed and nothing to resume from.
        // Reported as no face, because that is what it is — and the move it
        // composes into against a controller whose acceleration is not wanted is
        // the idle one, which is why that arm is where the bit is taken away. An
        // arm that did nothing at all left this the one state the state machine
        // could not leave.
        assert_eq!(face_bits(false, true), None);
        assert_eq!(move_for(face_bits(false, true), None), Move::Idle);
        // And where the acceleration *is* wanted the transition rewrites both
        // bits, so the encoding is repaired by the arm that arms it.
        for face in [Face::XAvic, Face::X2Avic] {
            assert_eq!(
                move_for(face_bits(false, true), Some(face)),
                Move::Enable(face)
            );
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

    #[test]
    fn the_task_priority_has_one_authority_per_exit() {
        // The whole of the decision, and the reason it is one: the control block's
        // copy is taken only where the software was delivering, so the model is
        // never written from that field and then overwritten from the backing page
        // at the following entry — with the park decision, which reads the model,
        // falling between the two.
        assert_eq!(TaskPriority::owner(false), TaskPriority::Block);
        assert_eq!(TaskPriority::owner(true), TaskPriority::Backing);
        assert_ne!(TaskPriority::owner(true), TaskPriority::owner(false));
    }

    #[test]
    fn a_physical_entry_keeps_its_page_and_host_identifier_through_both_bits() {
        // An entry is described once, with a page and a host identifier, and
        // every write to it afterwards is one of these four read-modify-writes.
        // Each state they can leave it in is asserted as a whole value, because a
        // mask that reached a field beside its own bit would point a peer's
        // hardware at another processor's page, or at another physical processor,
        // and nothing on the machine would report either.
        let described = PhysicalApicEntry::new()
            .with_backing_page_address(PhysAddr::new(0x0012_3000))
            .with_host_apic_id(0x123);
        let driven = described.into_bits() | IS_VALID;
        let running = driven | IS_RUNNING;
        for (bits, expected) in [
            (driven, described.with_valid(true)),
            (running, described.with_valid(true).with_is_running(true)),
            (running & !IS_RUNNING, described.with_valid(true)),
            (running & !IS_VALID, described.with_is_running(true)),
            (running & !IS_VALID & !IS_RUNNING, described),
        ] {
            assert_eq!(PhysicalApicEntry::from_bits(bits), expected, "{bits:#018x}");
        }
    }
}
