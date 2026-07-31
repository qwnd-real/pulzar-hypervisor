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
//! # What the guest may not change
//!
//! The identifier is read-only, and not merely because recent processors made
//! it so. Every interrupt this hypervisor passes through — from an I/O
//! controller, from a device's message — is routed by hardware using the *real*
//! identifier. A guest that renamed its controller would be describing a
//! machine whose interrupts could no longer be delivered to it.

use core::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};

use cpu::{ApicId, CpuIndex};
use descriptors::Vector;

use crate::{
    base::{ApicBase, BaseFault, Mode},
    error::{ErrorStatus, Errors},
    icr::{Command, Trigger},
    lvt::{Entry, Lvt, TimerMode},
    priority::{self, Priority},
    vectors::Bitmap,
};

/// One processor's emulated local interrupt controller.
#[derive(Debug)]
pub struct Vlapic {
    index: CpuIndex,
    apic_id: ApicId,
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
    timer_deadline: AtomicU64,
    command: AtomicU64,
    errors: ErrorStatus,
    deferred: Bitmap,
    startup: AtomicU8,
    sipi_vector: AtomicU32,
    in_guest: AtomicBool,
    nmi: AtomicBool,
    owned: AtomicBool,
}

impl Vlapic {
    /// A controller as a processor finds it coming out of reset.
    pub(crate) fn new(index: CpuIndex, apic_id: ApicId, bootstrap: bool) -> Self {
        let this = Self {
            index,
            apic_id,
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
            timer_deadline: AtomicU64::new(0),
            command: AtomicU64::new(0),
            errors: ErrorStatus::new(),
            deferred: Bitmap::new(),
            startup: AtomicU8::new(Startup::Running as u8),
            sipi_vector: AtomicU32::new(NO_SIPI),
            in_guest: AtomicBool::new(false),
            nmi: AtomicBool::new(false),
            owned: AtomicBool::new(false),
        };
        this.reset_registers();
        this
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

    /// Which face the guest is reaching this controller through.
    pub(crate) fn mode(&self) -> Mode {
        self.base().mode()
    }

    /// The base register as it stands.
    pub(crate) fn base(&self) -> ApicBase {
        ApicBase::from_bits(self.base.load(Ordering::Acquire))
    }

    /// Takes a write to the base register, and says what changed.
    ///
    /// A change of face is not merely a different way of naming the same
    /// registers: the architecture keeps only the identifier across it and
    /// leaves everything else to be programmed again. So the register file is
    /// reset here, exactly as hardware would.
    ///
    /// # Errors
    ///
    /// Whatever the transition refused: a reserved bit, a state that cannot be
    /// reached from this one, or an attempt to move the register page.
    pub(crate) fn write_base(&self, value: u64) -> Result<Mode, BaseFault> {
        let current = self.base();
        let next = current.written(value)?;
        self.base.store(next.bits(), Ordering::Release);
        if next.mode() != current.mode() {
            self.reset_registers();
        }
        Ok(next.mode())
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
    /// End-of-interrupt broadcast suppression is deliberately reported as
    /// unsupported. The bit would let a guest ask that acknowledging a
    /// level-triggered interrupt not be broadcast to the I/O controllers — but
    /// this hypervisor passes those controllers through, so the broadcast is
    /// performed by real hardware when the real acknowledgement is issued, and
    /// nothing here can suppress it. Reporting it unsupported is what stops a
    /// guest asking for something that would then silently not happen.
    pub(crate) const fn version() -> u32 {
        (Entry::MAX_INDEX as u32) << MAX_LVT_SHIFT | VERSION_NUMBER
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

    /// Takes a write to the spurious-interrupt vector register.
    pub(crate) fn set_spurious(&self, value: u32) {
        self.spurious
            .store(value & SPURIOUS_WRITABLE, Ordering::Release);
    }

    /// Whether the guest has software-enabled its controller.
    ///
    /// A software-disabled controller holds every local-vector-table entry
    /// masked and refuses to unmask one, but keeps whatever is already
    /// requested or in service and goes on answering interprocessor
    /// interrupts.
    pub(crate) fn software_enabled(&self) -> bool {
        self.spurious() & SOFTWARE_ENABLE != 0
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

    /// Records the task priority the guest set without exiting.
    ///
    /// The control block holds only the four bits of the priority class, which
    /// is what the control register a guest sets it through holds too — so the
    /// subclass is zero, and that is the value rather than a truncation of one.
    pub(crate) fn observe_task_priority(&self, priority: u8) {
        self.task_priority.store(
            u32::from(priority) << PRIORITY_CLASS_SHIFT,
            Ordering::Release,
        );
    }

    /// The priority this controller is actually servicing at.
    pub(crate) fn processor_priority(&self) -> Priority {
        priority::processor_priority(self.task_priority(), self.in_service.highest())
    }

    /// The arbitration priority, which exists only in the older face.
    pub(crate) fn arbitration_priority(&self) -> Priority {
        priority::arbitration_priority(
            self.task_priority(),
            self.in_service.highest(),
            self.request.highest(),
        )
    }

    /// One local-vector-table entry as the guest last wrote it.
    pub(crate) fn lvt(&self, entry: Entry) -> Lvt {
        Lvt::from_bits(self.lvt[entry.index()].load(Ordering::Acquire))
    }

    /// Takes a write to a local-vector-table entry, and answers with what the
    /// entry became.
    ///
    /// Two rules the architecture states about writes here, both enforced:
    /// bits the entry reserves are dropped rather than stored, and while the
    /// controller is software-disabled the mask bit cannot be cleared.
    pub(crate) fn write_lvt(&self, entry: Entry, value: u32) -> Lvt {
        let mut kept = value & entry.writable();
        if !self.software_enabled() {
            kept |= MASKED;
        }
        // An illegal vector in an entry that delivers one is an error the
        // architecture reports whether or not the entry is masked and whether
        // or not anything ever arrives on it.
        let written = Lvt::from_bits(kept);
        if entry.has_delivery() && !priority::legal(written.vector()) {
            self.errors.record(Errors::RECEIVE_ILLEGAL_VECTOR);
        }
        self.lvt[entry.index()].store(kept, Ordering::Release);
        written
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

    /// Sets what the timer counts down from, which is also what starts it.
    pub(crate) fn set_timer_initial(&self, value: u32) {
        self.timer_initial.store(value, Ordering::Release);
    }

    /// The timestamp the guest asked its timer to fire at.
    ///
    /// Meaningful only while the timer's entry selects deadline mode. In the
    /// other two modes the architecture has this read as zero and ignores
    /// writes, which is what [`Vlapic::set_timer_deadline`] enforces.
    pub(crate) fn timer_deadline(&self) -> u64 {
        self.timer_deadline.load(Ordering::Acquire)
    }

    /// Sets the timestamp the timer fires at, and says whether the write took.
    ///
    /// Refused unless the timer's entry selects deadline mode. Hardware ignores
    /// the write in that case rather than faulting, so this is not an error.
    pub(crate) fn set_timer_deadline(&self, value: u64) -> bool {
        let deadline =
            TimerMode::from_bits(self.lvt(Entry::Timer).timer_mode()) == Some(TimerMode::Deadline);
        if deadline {
            self.timer_deadline.store(value, Ordering::Release);
        }
        deadline
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
    pub(crate) fn set_command_high(&self, value: u32) {
        let low = self.command().low();
        self.command
            .store(Command::from_halves(low, value).bits(), Ordering::Release);
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
    /// Answers whether the interrupt was newly requested. A vector already
    /// requested and not yet accepted collapses into the one bit, exactly as
    /// hardware does, and is not a second interrupt.
    pub(crate) fn accept(&self, vector: Vector, trigger: Trigger) -> Accepted {
        // The controller never sets a request bit in the illegal range, and
        // records that it was asked to.
        if !priority::legal(vector) {
            self.errors.record(Errors::RECEIVE_ILLEGAL_VECTOR);
            return Accepted::Illegal;
        }
        match trigger {
            Trigger::Level => self.trigger_mode.set(vector),
            Trigger::Edge => self.trigger_mode.clear(vector),
        };
        if self.request.set(vector) {
            Accepted::Coalesced
        } else {
            Accepted::Requested
        }
    }

    /// The highest-priority interrupt the guest should take now, moved from
    /// requested to in service.
    ///
    /// Answers `None` when nothing is requested, or when what is requested does
    /// not outrank what the guest is already servicing — in which case the
    /// request stays pending, which is what makes a task priority a filter
    /// rather than a discard.
    ///
    /// Called only by the processor this controller belongs to, which is the
    /// only one that ever clears a request bit.
    pub(crate) fn take_deliverable(&self) -> Option<Vector> {
        let vector = self.request.highest()?;
        if !priority::deliverable(vector, self.processor_priority()) {
            return None;
        }
        self.request.clear(vector);
        self.in_service.set(vector);
        Some(vector)
    }

    /// Whether anything is requested at all, whatever its priority.
    pub(crate) fn requested(&self) -> Option<Vector> {
        self.request.highest()
    }

    /// Acknowledges the interrupt the guest is servicing, and says whether real
    /// hardware is still owed an acknowledgement for it.
    ///
    /// Retires the highest in-service bit, which is the one the guest must have
    /// been handling: interrupts nest by priority, so the most recently taken
    /// is always the highest.
    ///
    /// The answer is the whole of why level-triggered interrupts need care. An
    /// edge-triggered interrupt was finished with the moment it was taken, and
    /// the real controller was acknowledged then. A level-triggered one is
    /// asserted until the guest's driver deals with whatever raised it, so the
    /// real acknowledgement was deliberately withheld until now — and now is
    /// when it is owed.
    pub(crate) fn end_of_interrupt(&self) -> Option<Retired> {
        let vector = self.in_service.take_highest()?;
        Some(Retired {
            vector,
            acknowledge_hardware: self.deferred.clear(vector),
        })
    }

    /// Records that real hardware still holds `vector` in service, waiting for
    /// this guest to finish with it.
    pub(crate) fn defer_acknowledgement(&self, vector: Vector) {
        self.deferred.set(vector);
    }

    /// How many interrupts are requested and not yet taken.
    pub(crate) fn requested_count(&self) -> u32 {
        self.request.count()
    }

    /// How many the guest has taken and not yet acknowledged.
    pub(crate) fn in_service_count(&self) -> u32 {
        self.in_service.count()
    }

    /// Whether real hardware is owed an acknowledgement for anything.
    pub(crate) fn owes_acknowledgement(&self) -> bool {
        !self.deferred.is_empty()
    }

    /// Whether this processor is inside the guest.
    ///
    /// Read by a processor about to deliver an interrupt here, to decide
    /// whether the target has to be interrupted to notice it.
    pub(crate) fn in_guest(&self) -> bool {
        self.in_guest.load(Ordering::SeqCst)
    }

    /// Records whether this processor is inside the guest.
    ///
    /// Sequentially consistent, and it has to be. This store and the load of a
    /// request bit that follows it must not be reordered against a deliverer's
    /// store of that request bit and its load of this flag — if both were
    /// allowed to be seen stale, the deliverer would decide no interruption was
    /// needed while this processor decided nothing was pending, and the
    /// interrupt would be lost until something unrelated happened to cause an
    /// exit.
    pub(crate) fn set_in_guest(&self, inside: bool) {
        self.in_guest.store(inside, Ordering::SeqCst);
    }

    /// Records that this processor's guest is owed a non-maskable interrupt.
    ///
    /// Set by whichever processor sent it, and drained by this one at its next
    /// exit — which is why it lives here and not with the rest of what that
    /// processor's exit loop owns.
    pub(crate) fn raise_nmi(&self) {
        self.nmi.store(true, Ordering::Release);
    }

    /// Takes the outstanding non-maskable interrupt, if there is one.
    pub(crate) fn take_nmi(&self) -> bool {
        self.nmi.swap(false, Ordering::AcqRel)
    }

    /// Whether this hypervisor has taken this processor over.
    ///
    /// Until it has, the processor is running firmware's own code on real
    /// hardware and a startup message aimed at it belongs on the real
    /// controller. Once it has, the same message must be emulated, because
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

    /// Puts this processor into the state an INIT leaves it in, from any state.
    ///
    /// The vector a start-up message would have carried is cleared as part of
    /// the same transition, so that an INIT always wins a race against a
    /// start-up message already on its way: the target applies the reset and
    /// then finds nothing to start with.
    pub(crate) fn request_init(&self) {
        self.sipi_vector.store(NO_SIPI, Ordering::Release);
        self.startup
            .store(Startup::InitRequested as u8, Ordering::Release);
    }

    /// Offers a start-up vector, which takes only if this processor is waiting
    /// for one.
    pub(crate) fn request_sipi(&self, vector: u8) -> bool {
        if self.startup() != Startup::WaitingForSipi {
            return false;
        }
        self.sipi_vector.store(u32::from(vector), Ordering::Release);
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
    /// Also what a change of face leaves behind, for the same reason: the
    /// architecture preserves the identifier across one and requires software
    /// to program the rest again.
    pub(crate) fn reset_registers(&self) {
        self.request.reset();
        self.in_service.reset();
        self.trigger_mode.reset();
        self.deferred.reset();
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
        self.timer_deadline.store(0, Ordering::Release);
        self.command.store(0, Ordering::Release);
        self.errors.reset();
    }
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
}

/// An interrupt the guest has finished with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Retired {
    /// Which vector it was.
    pub(crate) vector: Vector,
    /// Whether the real controller is still holding it in service and is now
    /// owed an acknowledgement.
    pub(crate) acknowledge_hardware: bool,
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
