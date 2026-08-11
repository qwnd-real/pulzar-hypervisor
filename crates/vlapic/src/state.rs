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

use core::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};

use apic::LocalState;
use cpu::{ApicId, CpuIndex};
use descriptors::Vector;
use log::info;

use crate::{
    base::{ApicBase, BaseFault, Mode},
    error::{ErrorStatus, Errors},
    icr::{Command, Trigger},
    ledger::Ledger,
    lvt::{Delivery, Entry, Lvt, TimerMode},
    model::Model,
    priority::{self, Priority},
    sources,
    vectors::Bitmap,
};

/// One processor's emulated local interrupt controller.
#[derive(Debug)]
pub struct Vlapic {
    index: CpuIndex,
    apic_id: ApicId,
    model: Model,
    base: AtomicU64,
    request: Bitmap,
    in_service: Bitmap,
    trigger_mode: Bitmap,
    task_priority: AtomicU32,
    logical_destination: AtomicU32,
    destination_format: AtomicU32,
    spurious: AtomicU32,
    lvt: [AtomicU32; Entry::COUNT],
    timer_divide: AtomicU32,
    timer_initial: AtomicU32,
    command: AtomicU64,
    errors: ErrorStatus,
    ledger: Ledger,
    epoch: AtomicU64,
    startup: AtomicU8,
    sipi_vector: AtomicU32,
    away: AtomicBool,
    nmi: AtomicBool,
    owned: AtomicBool,
    /// The last selection state [`Vlapic::report_selection`] logged, packed by
    /// [`SelectionState::bits`], so a controller whose answer has not changed
    /// stays quiet. Diagnostic only: nothing reads it back but the report.
    reported: AtomicU64,
}

impl Vlapic {
    /// A controller as a processor finds it coming out of reset.
    pub(crate) fn new(index: CpuIndex, apic_id: ApicId, bootstrap: bool, model: Model) -> Self {
        let this = Self {
            index,
            apic_id,
            model,
            base: AtomicU64::new(ApicBase::reset(bootstrap).bits()),
            request: Bitmap::new(),
            in_service: Bitmap::new(),
            trigger_mode: Bitmap::new(),
            task_priority: AtomicU32::new(0),
            logical_destination: AtomicU32::new(0),
            destination_format: AtomicU32::new(FLAT_DESTINATION_FORMAT),
            spurious: AtomicU32::new(SPURIOUS_RESET),
            lvt: [const { AtomicU32::new(Entry::RESET) }; Entry::COUNT],
            timer_divide: AtomicU32::new(0),
            timer_initial: AtomicU32::new(0),
            command: AtomicU64::new(0),
            errors: ErrorStatus::new(),
            ledger: Ledger::new(),
            epoch: AtomicU64::new(0),
            startup: AtomicU8::new(Startup::Running as u8),
            sipi_vector: AtomicU32::new(NO_SIPI),
            away: AtomicBool::new(false),
            nmi: AtomicBool::new(false),
            owned: AtomicBool::new(false),
            reported: AtomicU64::new(NOTHING_REPORTED),
        };
        this.reset_registers();
        this
    }

    /// Puts this controller into the state firmware left the real one in.
    ///
    /// Called once, before the guest has run, on the controller belonging to
    /// the processor the capture was taken on. What it is for is that
    /// pulzar does not boot a fresh guest: it re-enters the firmware it
    /// found, and firmware expects its controller to hold what it left
    /// there. Every register below was read before anything overwrote it
    /// and would otherwise be gone — the real controller has since been
    /// masked, re-vectored and acknowledged out from under firmware by the
    /// host's own bring-up.
    ///
    /// # What is not seeded, and why
    ///
    /// The in-service bank. The host has already retired every bit of it on
    /// real hardware, and more decisively the guest does not resume inside
    /// firmware's interrupt handler — it resumes at a stub that calls back
    /// into firmware from the top. So nothing would ever acknowledge a
    /// seeded in-service bit, and the controller would refuse everything of
    /// that priority or lower for the rest of the machine's life.
    ///
    /// The request and trigger-mode banks *are* seeded, and cannot double up:
    /// the host retired what was in service and left what was merely
    /// requested where it was, so those vectors are still latched in the
    /// real controller too. When hardware delivers one, [`Vlapic::accept`]
    /// folds it into the bit already set exactly as hardware folds a
    /// repeated interrupt.
    ///
    /// The identifier and the version are not seeded either, and are not
    /// firmware's to give: the first is the real one this controller was built
    /// with, because every passed-through interrupt is routed by it, and the
    /// second describes the hardware behind this controller rather than what
    /// firmware saw.
    pub(crate) fn seed(&self, firmware: &LocalState, base: u64) {
        self.base.store(
            ApicBase::seeded(base, self.base().bootstrap()).bits(),
            Ordering::Release,
        );
        self.task_priority.store(
            firmware.task_priority & TASK_PRIORITY_MASK,
            Ordering::Release,
        );
        // Stored raw rather than through the setters, which mask to what a guest
        // may write: these came out of hardware, so what they hold is by
        // definition what the register holds, and a guest reading one back has to
        // find firmware's value rather than a narrowed one.
        self.logical_destination.store(
            firmware.logical_destination & LOGICAL_DESTINATION_MASK,
            Ordering::Release,
        );
        self.destination_format.store(
            (firmware.destination_format & DESTINATION_FORMAT_MASK) | !DESTINATION_FORMAT_MASK,
            Ordering::Release,
        );
        self.spurious
            .store(firmware.spurious & SPURIOUS_WRITABLE, Ordering::Release);
        for (entry, value) in Entry::ALL.into_iter().zip(firmware.lvt()) {
            self.lvt[entry.index()].store(value & entry.writable(self.model), Ordering::Release);
        }
        self.timer_divide
            .store(firmware.timer_divide & TIMER_DIVIDE_MASK, Ordering::Release);
        self.timer_initial
            .store(firmware.timer_initial_count, Ordering::Release);
        self.command.store(firmware.command, Ordering::Release);
        self.errors.seed(firmware.error_status);
        self.request.seed(&firmware.interrupt_request);
        self.trigger_mode.seed(&firmware.trigger_mode);
    }

    /// Where this controller sits in the roster.
    pub(crate) const fn index(&self) -> CpuIndex {
        self.index
    }

    /// The identifier interrupts to this processor are addressed by, which is
    /// the real one.
    pub(crate) const fn apic_id(&self) -> ApicId {
        self.apic_id
    }

    /// The controller this guest was told it has.
    pub(crate) const fn model(&self) -> Model {
        self.model
    }

    /// Which face the guest is reaching this controller through.
    pub(crate) fn mode(&self) -> Mode {
        self.base().mode()
    }

    /// The base register as it stands.
    pub(crate) fn base(&self) -> ApicBase {
        ApicBase::from_bits(self.base.load(Ordering::Acquire))
    }

    /// Takes a write to the base register, and says what it became.
    ///
    /// A change of face is a lifecycle boundary rather than a different way of
    /// naming the same registers, and which of the two boundaries it is decides
    /// everything below.
    ///
    /// Entering x2APIC from the older face preserves everything the
    /// architecture says it preserves — the priorities, what is requested and
    /// in service, the table, the errors — because a guest doing it is
    /// changing how it addresses its controller and not asking for a new
    /// one. So the machine behind those registers is preserved too, and
    /// deliberately: a timer that is counting goes on counting, an armed
    /// deadline stays armed, and every acknowledgement real hardware is
    /// owed stays owed, because the entries and the in-service bits that
    /// make sense of all three survive the write. Quieting them here would
    /// take away a timer the guest is entitled to keep and an appointment
    /// it cannot re-derive.
    ///
    /// Switching the controller off is the other thing entirely, and leaves the
    /// register file as reset leaves it. There the physical hardware has to be
    /// brought across first: an entry left armed goes on delivering into a
    /// controller the guest believes is switched off, and every acknowledgement
    /// owed has to be settled before the tokens that would have discharged it
    /// are deleted.
    ///
    /// # Errors
    ///
    /// Whatever the transition refused: a reserved bit, a state that cannot be
    /// reached from this one, or an attempt to move the register page.
    pub(crate) fn write_base(&self, value: u64) -> Result<Transition, BaseFault> {
        let current = self.base();
        let next = current.written(value, self.model)?;
        if next.mode() == current.mode() {
            // Not a transition. Software that reads the register, changes a
            // field it is entitled to and writes it back has asked for nothing
            // to happen, and nothing does.
            self.base.store(next.bits(), Ordering::Release);
            return Ok(Transition::Unchanged);
        }
        if matches!(
            (current.mode(), next.mode()),
            (Mode::XApic, Mode::X2Apic) | (Mode::X2Apic, Mode::XApic)
        ) {
            self.base.store(next.bits(), Ordering::Release);
            // The two exceptions the architecture names. The logical destination
            // stops being stored at all — x2APIC derives it from the identifier
            // — and the destination half of the command register has no
            // equivalent to carry over.
            self.logical_destination.store(0, Ordering::Release);
            self.command
                .store(self.command().low().into(), Ordering::Release);
            return Ok(Transition::Preserved);
        }
        // Before anything virtual moves, and in this order: a source that is
        // still armed can deliver into whatever comes next, and a debt that is
        // still outstanding needs the register file that records it.
        let quiet = sources::quiesce(self) & crate::timer::disarm(self);
        let settled = self.ledger.settle();

        self.base.store(next.bits(), Ordering::Release);
        self.reset_registers();
        Ok(Transition::Changed { quiet, settled })
    }

    /// The identifier register, in whichever shape the face in use gives it.
    ///
    /// The older interface keeps it in the top eight bits; x2APIC uses the
    /// whole register.
    pub(crate) fn id_register(&self) -> u32 {
        match self.mode() {
            Mode::X2Apic => self.apic_id.get(),
            _ => self.apic_id.get() << XAPIC_ID_SHIFT,
        }
    }

    /// The version register.
    ///
    /// The entry count is the real controller's, because the sources behind
    /// those entries are the real ones. A guest told it has an entry its
    /// hardware does not would be told about a source that can never fire and
    /// handed a register that cannot be programmed.
    ///
    /// End-of-interrupt broadcast suppression is deliberately reported as
    /// unsupported. The bit would let a guest ask that acknowledging a
    /// level-triggered interrupt not be broadcast to the I/O controllers — but
    /// this hypervisor passes those controllers through, so the broadcast is
    /// performed by real hardware when the real acknowledgement is issued, and
    /// nothing here can suppress it. Reporting it unsupported is what stops a
    /// guest asking for something that would then silently not happen.
    pub(crate) const fn version(&self) -> u32 {
        self.model.max_lvt() << MAX_LVT_SHIFT | VERSION_NUMBER
    }

    /// Which logical destinations this controller answers to.
    ///
    /// In x2APIC this is not stored at all: the architecture derives it from
    /// the identifier and makes it read-only, so it is computed here for the
    /// same reason hardware computes it, and a guest cannot get the two out of
    /// step.
    pub(crate) fn logical_destination(&self) -> u32 {
        match self.mode() {
            Mode::X2Apic => {
                let id = self.apic_id.get();
                ((id >> X2APIC_CLUSTER_SHIFT) << CLUSTER_SHIFT) | (1 << (id & X2APIC_LOGICAL_MASK))
            }
            _ => self.logical_destination.load(Ordering::Acquire),
        }
    }

    /// Sets which logical destinations this controller answers to. Reachable
    /// only in the older face, where the register is writable.
    pub(crate) fn set_logical_destination(&self, value: u32) {
        self.logical_destination
            .store(value & LOGICAL_DESTINATION_MASK, Ordering::Release);
    }

    /// How a logical destination is matched.
    pub(crate) fn destination_format(&self) -> u32 {
        self.destination_format.load(Ordering::Acquire)
    }

    /// Sets how a logical destination is matched. The reserved remainder reads
    /// as ones, which is its reset value and what the architecture requires.
    pub(crate) fn set_destination_format(&self, value: u32) {
        self.destination_format.store(
            (value & DESTINATION_FORMAT_MASK) | !DESTINATION_FORMAT_MASK,
            Ordering::Release,
        );
    }

    /// The spurious-interrupt vector register, which also holds the bit that
    /// software-enables the controller.
    pub(crate) fn spurious(&self) -> u32 {
        self.spurious.load(Ordering::Acquire)
    }

    /// Takes a write to the spurious-interrupt vector register, and says
    /// whether it switched the controller off.
    ///
    /// Software-disabling a controller is a transition and not a flag. The
    /// architecture has it mask every local vector table entry, and it means
    /// the stored entries themselves: a guest that disables its controller
    /// and reads an entry back must see the mask bit set, and a guest that
    /// re-enables it must have to unmask what it wants rather than finding
    /// its old sources live again. Doing it to the stored values is also
    /// what keeps real hardware honest, since that is what every source is
    /// programmed from.
    ///
    /// What is deliberately kept is everything already requested or in service.
    /// A disabled controller stops accepting; it does not retract what it has
    /// already taken.
    pub(crate) fn set_spurious(&self, value: u32) -> bool {
        let was = self.software_enabled();
        self.spurious
            .store(value & SPURIOUS_WRITABLE, Ordering::Release);
        let disabled = was && !self.software_enabled();
        if disabled {
            for entry in &self.lvt {
                entry.fetch_or(MASKED, Ordering::AcqRel);
            }
        }
        disabled
    }

    /// Whether the guest has software-enabled its controller.
    ///
    /// A software-disabled controller holds every local-vector-table entry
    /// masked and refuses to unmask one, and stops accepting anything new —
    /// while keeping whatever is already requested or in service.
    pub(crate) fn software_enabled(&self) -> bool {
        self.spurious() & SOFTWARE_ENABLE != 0
    }

    /// Whether this controller is in a state that accepts interrupts at all.
    ///
    /// Both switches have to be on. A controller whose guest has cleared the
    /// global enable has no controller as far as its guest is concerned, and
    /// one that is merely software-disabled has stopped accepting — in both
    /// cases an interrupt offered to it is one it must refuse rather than
    /// hold.
    pub(crate) fn accepting(&self) -> bool {
        self.mode() != Mode::Disabled && self.software_enabled()
    }

    /// The task priority the guest has set.
    pub(crate) fn task_priority(&self) -> Priority {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "only the low byte of the task priority register holds anything; the rest is reserved"
        )]
        Priority::new(self.task_priority.load(Ordering::Acquire) as u8)
    }

    /// Sets the task priority.
    pub(crate) fn set_task_priority(&self, value: u32) {
        self.task_priority
            .store(value & TASK_PRIORITY_MASK, Ordering::Release);
    }

    /// Records the priority class the guest set through its control register
    /// without exiting.
    ///
    /// The control block carries only the four bits of the class, because that
    /// is all the control register carries — and a write to that register
    /// clears the subclass, which is why a class differing from the stored
    /// one is stored as a whole byte with nothing beneath it.
    ///
    /// A class that agrees with the stored one is left alone, and that is the
    /// substance of this rather than an optimisation. This is called at every
    /// exit with a value that is only ever as wide as a class, so storing
    /// unconditionally would erase the subclass of any priority the guest wrote
    /// through the register file instead: a guest that asked for `0x21` would
    /// read `0x20` back at its next exit, having been told its write did not
    /// happen.
    pub(crate) fn observe_task_priority(&self, class: u8) {
        if self.task_priority().class() == class {
            return;
        }
        self.task_priority
            .store(u32::from(class) << PRIORITY_CLASS_SHIFT, Ordering::Release);
    }

    /// The priority this controller is actually servicing at.
    pub(crate) fn processor_priority(&self) -> Priority {
        priority::processor_priority(self.task_priority(), self.in_service.highest())
    }

    /// The priority the interrupts this controller has already accepted impose,
    /// with the guest's task priority left out of it.
    ///
    /// Half of [`Vlapic::processor_priority`], and the half this hypervisor can
    /// see change. The other half moves without any exit at all — the guest's
    /// control register writes land in the control block — which is why the two
    /// are separable at all and why [`Vlapic::pending`] needs this one alone.
    fn servicing(&self) -> Priority {
        priority::processor_priority(Priority::NONE, self.in_service.highest())
    }

    /// The arbitration priority, which exists only in the older face.
    ///
    /// Computed by the rule the guest's own processor follows, which is not the
    /// same rule on both vendors and is visible to a guest that reads it.
    pub(crate) fn arbitration_priority(&self) -> Priority {
        self.model.arbitration_priority(
            self.task_priority(),
            self.in_service.highest(),
            self.request.highest(),
        )
    }

    /// One local-vector-table entry as the guest last wrote it.
    ///
    /// What the guest programmed, which is what every source is programmed onto
    /// real hardware from. A guest *reading* the register gets
    /// [`Vlapic::lvt_readback`] instead, because three of the bits in it are
    /// hardware's to report rather than software's to set.
    pub(crate) fn lvt(&self, entry: Entry) -> Lvt {
        Lvt::from_bits(self.lvt[entry.index()].load(Ordering::Acquire))
    }

    /// One local-vector-table entry as the guest reads it.
    ///
    /// Three bits of an entry are the controller's and not software's, and all
    /// three are answered from the real entry rather than from anything stored
    /// here — because the source behind the entry is the real one, and the real
    /// controller is what maintains them.
    ///
    /// The delivery-status bit says a delivery from this source is still in
    /// flight. The remote-IRR bit says a level-triggered interrupt from this
    /// pin has been accepted and not yet acknowledged. And the mask bit is
    /// not purely software's either: hardware sets it itself on the
    /// performance-counter entry when the counter overflows, so a guest that
    /// armed that source and reads it back unmasked would be told a source is
    /// live that hardware has already stopped.
    pub(crate) fn lvt_readback(&self, entry: Entry) -> Lvt {
        let stored = self.lvt(entry);
        let Some(source) = sources::source_of(entry) else {
            return stored;
        };
        let Ok(real) = apic::local().and_then(|local| local.source(source)) else {
            return stored;
        };
        stored
            .with_send_pending(real.pending())
            .with_remote_irr(entry.is_pin() && real.remote_irr())
            .with_masked(stored.masked() || real.is_masked())
    }

    /// Takes a write to a local-vector-table entry, and answers with what the
    /// entry became.
    ///
    /// Three rules the architecture states about writes here, all enforced:
    /// bits the entry reserves are dropped rather than stored; while the
    /// controller is software-disabled the mask bit cannot be cleared; and a
    /// timer entry that crosses into or out of deadline mode leaves the initial
    /// count behind, because the count registers stop meaning anything there
    /// and hardware clears them as the mode changes. A guest coming back to
    /// a counting mode must not find an old count waiting to start a timer
    /// it never asked for.
    pub(crate) fn write_lvt(&self, entry: Entry, value: u32) -> Lvt {
        let mut kept = value & entry.writable(self.model);
        if !self.software_enabled() {
            kept |= MASKED;
        }
        let was = self.lvt[entry.index()].swap(kept, Ordering::AcqRel);
        if entry == Entry::Timer && waits_for_a_deadline(was) != waits_for_a_deadline(kept) {
            self.timer_initial.store(0, Ordering::Release);
        }
        Lvt::from_bits(kept)
    }

    /// Whether an entry, as it stands, would actually deliver a vector.
    ///
    /// Which is the only condition under which its vector field means anything.
    /// Every other delivery mode is an event the processor takes by its own
    /// architectural entry point and reads no vector for, so a number left in
    /// the field is not a vector at all and reporting it as an illegal one
    /// would be reporting an error about a field nothing reads.
    pub(crate) fn delivers_a_vector(&self, entry: Entry) -> bool {
        !self.model.has_delivery(entry)
            || Delivery::from_bits(self.lvt(entry).delivery()) == Some(Delivery::Fixed)
    }

    /// How far the bus clock is divided before the timer counts it.
    pub(crate) fn timer_divide(&self) -> u32 {
        self.timer_divide.load(Ordering::Acquire)
    }

    /// Sets the timer's divide configuration.
    pub(crate) fn set_timer_divide(&self, value: u32) {
        self.timer_divide
            .store(value & TIMER_DIVIDE_MASK, Ordering::Release);
    }

    /// What the timer counts down from.
    pub(crate) fn timer_initial(&self) -> u32 {
        self.timer_initial.load(Ordering::Acquire)
    }

    /// Sets what the timer counts down from, and says whether the write took.
    ///
    /// Refused in deadline mode, where the architecture has the count registers
    /// stop meaning anything and ignores writes to them. Ignored rather than
    /// faulted, and ignored completely: the value is not stored either, so a
    /// guest that writes a count in deadline mode and later selects a counting
    /// mode does not find the count it wrote waiting to start a timer it never
    /// asked for.
    pub(crate) fn set_timer_initial(&self, value: u32) -> bool {
        if self.timer_mode() == Some(TimerMode::Deadline) {
            return false;
        }
        self.timer_initial.store(value, Ordering::Release);
        true
    }

    /// Which mode the timer's entry selects, or `None` for the encoding the
    /// architecture reserves.
    pub(crate) fn timer_mode(&self) -> Option<TimerMode> {
        TimerMode::from_bits(self.lvt(Entry::Timer).timer_mode())
    }

    /// The interrupt command register as it stands.
    ///
    /// Kept because the older face writes it in two halves and the destination
    /// has to survive between them. A guest reading it back gets what it wrote,
    /// which the architecture does not promise but does not forbid, and which
    /// costs nothing to be honest about.
    pub(crate) fn command(&self) -> Command {
        Command::from_bits(self.command.load(Ordering::Acquire))
    }

    /// Sets the destination half, which sends nothing on its own.
    ///
    /// Everything below the top byte is reserved in this half, and is dropped
    /// rather than stored: a guest reading the register back must not find bits
    /// the architecture says read as zero.
    pub(crate) fn set_command_high(&self, value: u32) {
        let low = self.command().low();
        self.command.store(
            Command::from_halves(low, value & Command::WRITABLE_HIGH).bits(),
            Ordering::Release,
        );
    }

    /// Sets the half whose write sends the command, and answers with the whole
    /// of what is to be sent.
    pub(crate) fn set_command_low(&self, value: u32) -> Command {
        let high = self.command().high();
        let command = Command::from_halves(value & Command::WRITABLE_LOW, high);
        self.command.store(command.bits(), Ordering::Release);
        command
    }

    /// Sets the whole register at once, which is how x2APIC writes it.
    pub(crate) fn set_command(&self, value: u64) -> Command {
        let command = Command::from_bits(value);
        self.command.store(command.bits(), Ordering::Release);
        command
    }

    /// The error status register and its write-then-read protocol.
    pub(crate) const fn errors(&self) -> &ErrorStatus {
        &self.errors
    }

    /// What real hardware is holding in service on this guest's behalf.
    pub(crate) const fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    /// One slot of the interrupt-request register, as the guest reads it.
    pub(crate) fn request_slot(&self, slot: usize) -> u32 {
        self.request.slot(slot)
    }

    /// One slot of the in-service register.
    pub(crate) fn in_service_slot(&self, slot: usize) -> u32 {
        self.in_service.slot(slot)
    }

    /// One slot of the trigger-mode register.
    pub(crate) fn trigger_mode_slot(&self, slot: usize) -> u32 {
        self.trigger_mode.slot(slot)
    }

    /// Accepts an interrupt into this controller, from anywhere.
    ///
    /// This is what delivery means, and it is the one operation any processor
    /// may perform on any controller. The trigger mode is recorded before the
    /// request, so that a processor which observes the request bit cannot then
    /// read a trigger mode from before it was set — the difference between the
    /// two decides whether an acknowledgement is owed to real hardware, and
    /// getting it wrong is a line that never fires again.
    ///
    /// Both are published against the reset count, and republished if a reset
    /// moved it. An interrupt that raced a reset is one whose moment is
    /// indistinguishable from just after it, and just after it is when the
    /// guest the reset produced is entitled to see it — so the retry
    /// converges on the new state rather than leaving a half-written event
    /// in the old one.
    ///
    /// Answers what became of it. A vector already requested and not yet
    /// accepted collapses into the one bit, exactly as hardware does, and is
    /// not a second interrupt.
    pub(crate) fn accept(&self, vector: Vector, trigger: Trigger) -> Accepted {
        // The controller never sets a request bit in the illegal range, and
        // records that it was asked to.
        if !priority::legal(vector) {
            self.errors.record(Errors::RECEIVE_ILLEGAL_VECTOR);
            return Accepted::Illegal;
        }
        // A controller that is switched off or software-disabled does not accept
        // interrupts. The special messages that reach a disabled controller
        // anyway — INIT, start-up, a non-maskable interrupt — do not come
        // through here.
        if !self.accepting() {
            return Accepted::Refused;
        }
        loop {
            let Some(epoch) = self.settled_epoch() else {
                // A reset that never finishes is a broken invariant rather than
                // contention, and spinning in a delivery path would take the
                // sender down with it.
                return Accepted::Refused;
            };
            match trigger {
                Trigger::Level => self.trigger_mode.set(vector),
                Trigger::Edge => self.trigger_mode.clear(vector),
            };
            let coalesced = self.request.set(vector);
            if self.epoch.load(Ordering::SeqCst) == epoch {
                return if coalesced {
                    Accepted::Coalesced
                } else {
                    Accepted::Requested
                };
            }
        }
    }

    /// The highest-priority interrupt the guest should take now, left where it
    /// is.
    ///
    /// Deliberately does not consume anything. Whether the guest can actually
    /// be given an interrupt is not this controller's to know — the
    /// processor may have an event already being delivered, or a
    /// non-maskable interrupt that outranks this, or a closed interrupt
    /// window — and a controller that moved a vector out of the request
    /// register for an injection that then did not happen would have lost
    /// it. So this only nominates, and [`Vlapic::committed`] is what
    /// accounts for one that went in.
    ///
    /// Answers `None` when nothing is requested, or when what is requested does
    /// not outrank what the guest is already servicing — in which case the
    /// request stays pending, which is what makes a task priority a filter
    /// rather than a discard.
    pub(crate) fn select(&self) -> Option<Vector> {
        let selected = self.select_inner();
        self.report_selection(selected);
        selected
    }

    /// The highest-priority interrupt nothing but the guest's task priority may
    /// be holding back.
    ///
    /// A superset of [`Vlapic::select`], and the difference between them is the
    /// one comparison this hypervisor must not be the one to make. A guest
    /// changes its task priority through its control register without exiting —
    /// the processor keeps the value in the control block — so a vector this
    /// crate ruled out on a task priority it read at the last exit would stay
    /// ruled out however far the guest lowered that priority afterwards, and
    /// nothing would ever ask again.
    ///
    /// So the two halves of the processor priority are split. What is already
    /// in service is applied here, because it moves only when the guest
    /// acknowledges an interrupt and that always exits. The task priority is
    /// left to the hardware that owns it: this is what an interrupt window is
    /// armed for, and the processor raises one exactly when the guest's own
    /// priority admits the vector.
    ///
    /// Applying the in-service half here rather than leaving both to hardware
    /// is what keeps that arrangement from spinning. The control block
    /// carries only the task priority, so a vector armed while an interrupt
    /// of its own class or higher is still in service would have the
    /// processor report a window the guest cannot actually take anything
    /// through, and every exit would arm it again.
    pub(crate) fn pending(&self) -> Option<Vector> {
        if !self.accepting() {
            return None;
        }
        let vector = self.request.highest()?;
        priority::deliverable(vector, self.servicing()).then_some(vector)
    }

    /// [`Vlapic::select`] proper, with nothing said about what it decided.
    fn select_inner(&self) -> Option<Vector> {
        if !self.accepting() {
            return None;
        }
        let vector = self.request.highest()?;
        priority::deliverable(vector, self.processor_priority()).then_some(vector)
    }

    /// Says why this controller is nominating what it is, the first time it
    /// reaches any given answer.
    ///
    /// A controller that has stopped delivering says so once and then goes
    /// quiet, so this costs nothing on the path it sits on: what makes an
    /// interrupt undeliverable is state that has to change before it becomes
    /// deliverable again, and the change is what gets reported. The three ways
    /// a nomination comes to nothing are indistinguishable to the caller, and
    /// they are three different faults — a controller its guest switched off,
    /// a controller with nothing to give, and a controller holding something
    /// back behind a priority that never falls.
    fn report_selection(&self, selected: Option<Vector>) {
        let accepting = self.accepting();
        let requested = self.request.highest();
        let in_service = self.in_service.highest();
        let task = self.task_priority();
        let processor = self.processor_priority();
        // Everything the report below names, packed into one word so that
        // "has this changed" is a single comparison rather than a lock.
        let bits = u64::from(accepting)
            | u64::from(number(requested)) << 8
            | u64::from(number(in_service)) << 24
            | u64::from(task.get()) << 40
            | u64::from(processor.get()) << 48
            | u64::from(number(selected)) << 56;
        if self.reported.swap(bits, Ordering::Relaxed) == bits {
            return;
        }
        match (accepting, requested, selected) {
            (false, _, _) => info!(
                "vlapic: {} is not accepting interrupts: mode {:?}, software {}",
                self.index(),
                self.mode(),
                if self.software_enabled() {
                    "enabled"
                } else {
                    "disabled"
                }
            ),
            (true, None, _) => info!(
                "vlapic: {} has nothing requested; in service {in_service:?}, task priority {:#x}",
                self.index(),
                task.get()
            ),
            (true, Some(vector), None) => info!(
                "vlapic: {} is holding {vector} back: processor priority {:#x} from task {:#x} \
                 and in service {in_service:?}",
                self.index(),
                processor.get(),
                task.get()
            ),
            (true, Some(_), Some(vector)) => info!(
                "vlapic: {} nominates {vector}, processor priority {:#x}, in service \
                 {in_service:?}",
                self.index(),
                processor.get()
            ),
        }
    }

    /// Records that the guest really has been given `vector`, moving it from
    /// requested to in service.
    ///
    /// Called only after the injection has been established to have happened,
    /// and only by the processor this controller belongs to — which is the only
    /// one that ever clears a request bit.
    ///
    /// The request bit is cleared before the in-service bit is set, so that a
    /// reader never sees the vector in neither register. Answers whether the
    /// vector really was still requested: a reset between the selection and the
    /// commitment leaves nothing to move, and nothing is then put in service.
    pub(crate) fn committed(&self, vector: Vector) -> bool {
        if !self.request.clear(vector) {
            return false;
        }
        self.in_service.set(vector);
        true
    }

    /// Whether anything is requested at all, whatever its priority.
    pub(crate) fn requested(&self) -> Option<Vector> {
        self.request.highest()
    }

    /// Acknowledges the interrupt the guest is servicing, and says which it
    /// was.
    ///
    /// Retires the highest in-service bit, which is the one the guest must have
    /// been handling: interrupts nest by priority, so the most recently taken
    /// is always the highest.
    ///
    /// Releasing the debt is part of the same operation rather than something a
    /// caller does afterwards, because the two must not come apart: a guest's
    /// acknowledgement is exactly the event that makes an acknowledgement to
    /// real hardware permissible, and nothing else ever will be.
    pub(crate) fn end_of_interrupt(&self) -> Option<Vector> {
        let vector = self.in_service.take_highest()?;
        self.trigger_mode.clear(vector);
        self.ledger.release(vector);
        Some(vector)
    }

    /// How many interrupts are requested and not yet taken.
    pub(crate) fn requested_count(&self) -> u32 {
        self.request.count()
    }

    /// How many the guest has taken and not yet acknowledged.
    pub(crate) fn in_service_count(&self) -> u32 {
        self.in_service.count()
    }

    /// Whether this processor has stopped looking at this controller.
    ///
    /// True while it is inside the guest, and true while it is halted waiting
    /// to be started — the two states in which nothing it does will notice a
    /// bit being set here until something interrupts it. False while it is
    /// answering an exit, because it consults this controller before it goes
    /// back in.
    ///
    /// Read by a processor about to deliver something here, to decide whether
    /// the target has to be interrupted to notice.
    pub(crate) fn away(&self) -> bool {
        self.away.load(Ordering::SeqCst)
    }

    /// Records whether this processor has stopped looking at this controller.
    ///
    /// Sequentially consistent, and it has to be. This store and the load of a
    /// request bit that follows it must not be reordered against a deliverer's
    /// store of that request bit and its load of this flag — if both were
    /// allowed to be seen stale, the deliverer would decide no interruption was
    /// needed while this processor decided nothing was pending, and the
    /// interrupt would be lost until something unrelated happened to cause an
    /// exit.
    pub(crate) fn set_away(&self, away: bool) {
        self.away.store(away, Ordering::SeqCst);
    }

    /// Records that this processor's guest is owed a non-maskable interrupt.
    ///
    /// Set by whichever processor sent it, and drained by this one at its next
    /// exit — which is why it lives here and not with the rest of what that
    /// processor's exit loop owns.
    ///
    /// Sequentially consistent for the reason [`Vlapic::signalled`] gives: this
    /// store and the sender's read of [`Vlapic::away`] pair with the target's
    /// store of that flag and its read of this one, and a weaker ordering lets
    /// both sides miss.
    pub(crate) fn raise_nmi(&self) {
        self.nmi.store(true, Ordering::SeqCst);
    }

    /// Takes the outstanding non-maskable interrupt, if there is one.
    pub(crate) fn take_nmi(&self) -> bool {
        self.nmi.swap(false, Ordering::AcqRel)
    }

    /// Whether this hypervisor runs the processor this controller belongs to.
    ///
    /// Until it does, the processor is running firmware's own code on real
    /// hardware and a startup message aimed at it belongs on the real
    /// controller. Once it does, the same message must be emulated, because
    /// forwarding it would reset the host.
    pub(crate) fn owned(&self) -> bool {
        self.owned.load(Ordering::Acquire)
    }

    /// Records that this hypervisor now runs this processor.
    pub(crate) fn take_ownership(&self) {
        self.owned.store(true, Ordering::Release);
    }

    /// What this processor is doing about being started and stopped.
    pub(crate) fn startup(&self) -> Startup {
        Startup::from_bits(self.startup.load(Ordering::Acquire))
    }

    /// Whether this processor's guest is running rather than reset.
    pub(crate) fn running(&self) -> bool {
        self.startup() == Startup::Running
    }

    /// Whether a startup message has arrived that this processor has not
    /// applied yet.
    ///
    /// What a processor halted with nothing to run tests before it halts again.
    /// Sequentially consistent, and paired with [`Vlapic::set_away`]: the
    /// sender stores the message and then reads whether this processor is away,
    /// this processor stores that it is away and then reads for a message, and
    /// the total order over those four is what guarantees at least one of them
    /// sees the other. Without it a message could arrive between the test and
    /// the halt and be answered by nobody.
    pub(crate) fn signalled(&self) -> bool {
        self.startup.load(Ordering::SeqCst) != Startup::WaitingForSipi as u8
            || self.sipi_vector.load(Ordering::SeqCst) != NO_SIPI
    }

    /// Puts this processor into the state an INIT leaves it in, from any state.
    ///
    /// The vector a start-up message would have carried is cleared as part of
    /// the same transition, so that an INIT always wins a race against a
    /// start-up message already on its way: the target applies the reset and
    /// then finds nothing to start with.
    ///
    /// Both stores are sequentially consistent for the reason
    /// [`Vlapic::signalled`] gives.
    pub(crate) fn request_init(&self) {
        self.sipi_vector.store(NO_SIPI, Ordering::SeqCst);
        self.startup
            .store(Startup::InitRequested as u8, Ordering::SeqCst);
    }

    /// Offers a start-up vector, which takes only if this processor is waiting
    /// for one.
    pub(crate) fn request_sipi(&self, vector: u8) -> bool {
        if self.startup() != Startup::WaitingForSipi {
            return false;
        }
        self.sipi_vector.store(u32::from(vector), Ordering::SeqCst);
        true
    }

    /// Takes the start-up vector this processor was given, if it has one.
    pub(crate) fn take_sipi(&self) -> Option<u8> {
        let vector = self.sipi_vector.swap(NO_SIPI, Ordering::AcqRel);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a start-up vector is eight bits, and the sentinel is the only wider value stored"
        )]
        (vector != NO_SIPI).then_some(vector as u8)
    }

    /// Moves this processor to a startup state it has decided to be in.
    pub(crate) fn set_startup(&self, startup: Startup) {
        self.startup.store(startup as u8, Ordering::Release);
    }

    /// Everything an INIT leaves behind, which is everything but the
    /// identifier, the version and the face in use.
    ///
    /// Bracketed by the reset count so that everything cleared here is cleared
    /// as one step as far as any deliverer is concerned. A processor delivering
    /// into this controller while it runs publishes against the count, sees it
    /// move, and publishes again — so no interrupt is left with its request bit
    /// set and its trigger mode cleared, which is the state that costs a real
    /// acknowledgement and kills a line.
    pub(crate) fn reset_registers(&self) {
        self.epoch.fetch_add(1, Ordering::SeqCst);
        self.request.reset();
        self.in_service.reset();
        self.trigger_mode.reset();
        self.task_priority.store(0, Ordering::Release);
        self.logical_destination.store(0, Ordering::Release);
        self.destination_format
            .store(FLAT_DESTINATION_FORMAT, Ordering::Release);
        self.spurious.store(SPURIOUS_RESET, Ordering::Release);
        for entry in &self.lvt {
            entry.store(Entry::RESET, Ordering::Release);
        }
        self.timer_divide.store(0, Ordering::Release);
        self.timer_initial.store(0, Ordering::Release);
        self.command.store(0, Ordering::Release);
        self.errors.reset();
        self.epoch.fetch_add(1, Ordering::SeqCst);
    }

    /// The reset count, once no reset is in progress.
    ///
    /// Odd means one is running. `None` says one has been running for longer
    /// than a reset can take, which is a broken invariant rather than
    /// contention: a reset is a few dozen stores by a processor that is not
    /// part-way through anything else.
    fn settled_epoch(&self) -> Option<u64> {
        (0..EPOCH_SPINS).find_map(|_| {
            let epoch = self.epoch.load(Ordering::SeqCst);
            if epoch.is_multiple_of(2) {
                return Some(epoch);
            }
            core::hint::spin_loop();
            None
        })
    }
}

/// How many times a deliverer re-reads a reset count that says a reset is
/// running before giving up on it.
const EPOCH_SPINS: u32 = 100_000;

/// Whether a timer entry selects the mode that counts nothing, and so the mode
/// the count registers mean nothing in.
fn waits_for_a_deadline(entry: u32) -> bool {
    TimerMode::from_bits(Lvt::from_bits(entry).timer_mode()) == Some(TimerMode::Deadline)
}

/// What a write to the base register did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Transition {
    /// The write named the state the register was already in, so nothing
    /// happened and nothing has to be reconciled.
    Unchanged,
    /// The controller changed face and kept its register file, which is what
    /// the architecture preserves across that one move — so nothing behind
    /// it was quieted and there was nothing to settle.
    Preserved,
    /// The controller changed face and its register file was reset, so the
    /// physical hardware behind it was brought across first.
    Changed {
        /// Whether every source really was quieted first.
        quiet: bool,
        /// Whether every acknowledgement real hardware was owed really was
        /// settled first.
        settled: bool,
    },
}

/// What became of an interrupt offered to a controller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Accepted {
    /// Newly requested.
    Requested,
    /// Already requested and not yet accepted, so it folded into the one bit.
    Coalesced,
    /// Named a vector no controller may deliver, and was refused.
    Illegal,
    /// Offered to a controller that is not accepting interrupts.
    Refused,
}

/// What a processor is doing about being started and stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum Startup {
    /// Running the guest, or ready to.
    Running = 0,
    /// Another processor has sent an INIT that this one has not applied yet.
    InitRequested = 1,
    /// Reset and held, doing nothing until a start-up message arrives.
    WaitingForSipi = 2,
}

impl Startup {
    /// The state an encoding names, with anything undefined reading as running
    /// — the state every processor is in unless something put it elsewhere.
    const fn from_bits(bits: u8) -> Self {
        match bits {
            1 => Self::InitRequested,
            2 => Self::WaitingForSipi,
            _ => Self::Running,
        }
    }
}

/// The value stored where a start-up vector would be when there is none.
///
/// Outside the eight bits a vector occupies, so it cannot collide with one.
const NO_SIPI: u32 = u32::MAX;

/// What [`Vlapic::reported`] holds before any selection has been reported.
///
/// Not a state any real packing produces, so the first selection always reports
/// however trivial it is.
const NOTHING_REPORTED: u64 = u64::MAX;

/// A vector's number, or a value outside the eight bits one occupies when there
/// is no vector.
///
/// Used to pack an optional vector into the reported selection state, where
/// "nothing" has to be as distinguishable as any vector is.
const fn number(vector: Option<Vector>) -> u16 {
    match vector {
        Some(vector) => vector.number() as u16,
        None => u16::MAX,
    }
}

/// Bits the older interface's identifier is shifted by.
const XAPIC_ID_SHIFT: u32 = 24;

/// The version this controller reports: an integrated one, which is what every
/// processor since the discrete controller reports.
const VERSION_NUMBER: u32 = 0x10;

/// Bits the local-vector-table entry count is shifted by in the version
/// register.
const MAX_LVT_SHIFT: u32 = 16;

/// The part of the logical destination register that holds anything.
const LOGICAL_DESTINATION_MASK: u32 = 0xFF00_0000;

/// The part of the destination format register that selects anything.
const DESTINATION_FORMAT_MASK: u32 = 0xF000_0000;

/// The destination format register's reset value: the flat model, with every
/// reserved bit set.
const FLAT_DESTINATION_FORMAT: u32 = u32::MAX;

/// Bits an x2APIC logical identifier's cluster is shifted by.
const CLUSTER_SHIFT: u32 = 16;

/// Bits an identifier is shifted by to leave the cluster it names.
const X2APIC_CLUSTER_SHIFT: u32 = 4;

/// The part of an identifier that selects one processor within its cluster.
const X2APIC_LOGICAL_MASK: u32 = 0xF;

/// Bits a priority class is shifted by within the byte that holds it.
const PRIORITY_CLASS_SHIFT: u32 = 4;

/// The task priority register's reserved bits are everything above the low
/// byte.
const TASK_PRIORITY_MASK: u32 = 0xFF;

/// The timer's divide configuration is three bits, and not three adjacent
/// ones: bit two is reserved and sits in the middle of them.
const TIMER_DIVIDE_MASK: u32 = 0b1011;

/// The bit that makes a controller deliver anything at all.
const SOFTWARE_ENABLE: u32 = 1 << 8;

/// The spurious-interrupt vector register's reset value: every vector bit set
/// and the controller software-disabled.
const SPURIOUS_RESET: u32 = 0xFF;

/// What software may set in the spurious-interrupt vector register.
///
/// The vector and the software-enable bit. Focus-processor checking and
/// end-of-interrupt broadcast suppression are both refused: the first has no
/// meaning on any processor this runs on, and the second is reported
/// unsupported in the version register because the broadcast is performed by
/// hardware this hypervisor passes through.
const SPURIOUS_WRITABLE: u32 = 0x1FF;

/// The bit that masks a local-vector-table entry.
const MASKED: u32 = 1 << 16;
