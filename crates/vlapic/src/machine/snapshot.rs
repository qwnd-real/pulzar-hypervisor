//! Everything one processor's emulated controller holds, read out at once.
//!
//! No part of driving a controller reads this. It exists for the debugging
//! facility outside the crate that has to *report* one — the register file is
//! this crate's and the reader is not — and every field below is taken through
//! the same named operation the rest of the crate uses, so a snapshot cannot
//! report a register in a shape nothing else would answer with.
//!
//! # A snapshot, and not an instant
//!
//! Several dozen independent atomic loads, some of them of registers another
//! processor is entitled to be changing while they happen: a bank read early
//! and a count read late can disagree, and an interrupt that arrived in between
//! appears in one and not the other. Nothing here takes a lock and nothing here
//! waits for the reset count, deliberately — a debugging read must not be able
//! to hold up interrupt delivery, and it is not a read anything decides
//! anything from.
//!
//! # What it deliberately does not carry
//!
//! Two things a reader might expect. The count of non-maskable interrupts this
//! processor has been delivered is not readable without consuming one, and a
//! read that consumed one would be a debugging tool swallowing an interrupt.
//! And the one-shot log latches are not here either: what they record is which
//! *lines* have already been printed, which is a fact about the log rather than
//! about the controller.

use apic::{LVT_ENTRIES, VECTOR_WORDS};
use bitflags::bitflags;
use descriptors::Vector;
use paging::{DirectMap, chunk};
use svm::avic::{LogicalApicEntry, PhysicalApicEntry};
use x86_64::PhysAddr;

use crate::{
    VlapicError,
    avic::activation,
    hardware::timer,
    lifecycle::ledger::Debts,
    machine::{current, registry},
    priority::Priority,
    registers::{
        Nomination, Phase, StartupPage, Vlapic,
        base::Mode,
        lvt::{Entry, TimerMode},
    },
};

/// Everything the calling processor's emulated controller holds.
///
/// # Errors
///
/// [`VlapicError::NotInstalled`] before [`crate::install`], or
/// [`VlapicError::NoLapic`] on a processor the roster does not describe.
pub fn snapshot() -> Result<Snapshot, VlapicError> {
    let vlapic = current()?;
    let processors = registry::lapics()?.all().len();
    let base = vlapic.base();
    let mode = base.mode();
    let (startup, startup_page) = StartupPhase::of(vlapic.startup().phase());
    Ok(Snapshot {
        index: vlapic.index().get(),
        processors,
        apic_id: vlapic.apic_id().get(),
        state: State::of(vlapic),
        base: base.bits(),
        face: Face::of(mode),
        id: vlapic.id_register(mode),
        xapic_id: vlapic.xapic_id(),
        version: vlapic.version(),
        task_priority: vlapic.task_priority(),
        processor_priority: vlapic.processor_priority(),
        arbitration_priority: vlapic.arbitration_priority(),
        logical_destination: vlapic.logical_destination(mode),
        destination_format: vlapic.destination_format(),
        spurious: vlapic.spurious(),
        error_status: vlapic.errors().read(),
        command: vlapic.command().bits(),
        lvt: core::array::from_fn(|entry| vlapic.lvt_readback(Entry::ALL[entry]).into_bits()),
        request: bank(|slot| vlapic.request_slot(slot)),
        in_service: bank(|slot| vlapic.in_service_slot(slot)),
        trigger_mode: bank(|slot| vlapic.trigger_mode_slot(slot)),
        external: bank(|slot| vlapic.external_slot(slot)),
        requested: vlapic.requested(),
        requested_count: vlapic.requested_count(),
        in_service_count: vlapic.in_service_count(),
        nomination: vlapic.nominate(),
        timer_divide: vlapic.timer_divide(),
        timer_initial: vlapic.timer_initial(),
        timer_remaining: timer::remaining(vlapic),
        timer_frequency: vlapic.timer_frequency(),
        timer_deadline: timer::deadline(vlapic),
        timing: vlapic.timer_mode().map(Timing::of),
        startup,
        startup_page,
        epoch: vlapic.epoch(),
        ledger: Ledger::of(vlapic.ledger().debts()),
        counted: Counted::of(vlapic),
        acceleration: Acceleration::of(),
    })
}

/// One bank's slots, read one at a time.
///
/// The bank answers `None` for a slot outside it, and the range walked here is
/// its own — so the zero is unreachable rather than a value a reader could be
/// handed for a slot that exists.
fn bank(slot: impl Fn(usize) -> Option<u32>) -> [u32; VECTOR_WORDS] {
    core::array::from_fn(|index| slot(index).unwrap_or(0))
}

/// Copies as much of the calling processor's backing page as `into` holds, and
/// answers how many bytes that was.
///
/// The register file the hardware serves the guest out of while it drives the
/// controller, which is a different set of registers from the model's whenever
/// the hardware is driving — that being the whole reason a reader wants both.
///
/// Not one instant's truth: the processor's own hardware writes this page
/// without telling anybody, so a bank read early and one read late can
/// disagree, exactly as they can in the model beside it.
///
/// # Errors
///
/// [`VlapicError::NotProvisioned`] on a machine that built no acceleration,
/// [`VlapicError::NoLapic`] on a processor that was given no page, or
/// [`VlapicError::Paging`] if the window does not reach it.
pub fn read_backing_page(into: &mut [u8]) -> Result<usize, VlapicError> {
    let policy = activation::policy().ok_or(VlapicError::NotProvisioned)?;
    let page = activation::own_page()?;
    let bytes = into.len().min(paging::as_usize(chunk::FRAME_SIZE));
    let into = &mut into[..bytes];
    // SAFETY: the page is a frame provisioning allocated out of the reserved
    // chunk for this processor and never freed, so it is RAM rather than a
    // device aperture and it outlives this call. Nothing in this crate holds a
    // Rust reference to it — the accesses that maintain it go through the same
    // window, a word at a time — and a byte the hardware is writing while this
    // runs yields what a non-atomic copy of it yields, which is what a
    // diagnostic asks for and what nothing here decides anything from.
    unsafe { policy.window.read(page, into) }?;
    Ok(bytes)
}

/// Copies as many whole entries of the physical table as `into` holds, and
/// answers how many bytes that was.
///
/// One entry per identifier an interrupt may name, each a quadword in the
/// machine's own order, and the table describes [`Acceleration::max_index`]
/// plus one of them — so a caller whose buffer is shorter has been given the
/// first of them and can say so.
///
/// # Errors
///
/// As [`read_backing_page`], less the processor's own page: this table is the
/// machine's rather than any one processor's.
pub fn read_physical_table(into: &mut [u8]) -> Result<usize, VlapicError> {
    let policy = activation::policy().ok_or(VlapicError::NotProvisioned)?;
    let entries = (usize::from(policy.max_index) + 1) * size_of::<PhysicalApicEntry>();
    read_table(
        policy.window,
        policy.physical_table,
        into,
        entries,
        size_of::<PhysicalApicEntry>(),
    )
}

/// Copies as many whole entries of the logical table as `into` holds, and
/// answers how many bytes that was.
///
/// Each is a doubleword in the machine's own order.
///
/// # Errors
///
/// As [`read_physical_table`].
pub fn read_logical_table(into: &mut [u8]) -> Result<usize, VlapicError> {
    let policy = activation::policy().ok_or(VlapicError::NotProvisioned)?;
    let entries = policy.logical_entries * size_of::<LogicalApicEntry>();
    read_table(
        policy.window,
        policy.logical_table,
        into,
        entries,
        size_of::<LogicalApicEntry>(),
    )
}

/// Copies the first `held` bytes of a table into `into`, rounded down to whole
/// entries of `entry` bytes, and answers how many bytes that was.
///
/// Rounded down because a reader given part of an entry would be given a word
/// that stands for nothing: the tables are runs of fixed-width entries, and
/// half of one is not a shorter table.
fn read_table(
    window: DirectMap,
    table: PhysAddr,
    into: &mut [u8],
    held: usize,
    entry: usize,
) -> Result<usize, VlapicError> {
    let bytes = into.len().min(held) / entry * entry;
    // SAFETY: the table is frames provisioning allocated out of the reserved
    // chunk and never freed, so the range is RAM rather than a device aperture
    // and it outlives this call, and it is reached through the window it was
    // written through. Nothing in this crate holds a Rust reference to it — the
    // accesses that maintain it go through the same window, a word at a time —
    // and an entry another processor is writing while this runs yields what a
    // non-atomic copy of it yields, which is what a diagnostic asks for and what
    // nothing here decides anything from. The destination is a distinct borrow of
    // the caller's own buffer.
    unsafe { window.read(table, &mut into[..bytes]) }?;
    Ok(bytes)
}

/// One processor's emulated controller, as it stood.
///
/// Plain data, with every register in the shape the guest's own read would
/// answer with. The two that are computed rather than stored — the processor
/// and arbitration priorities — are the numbers the guest's processor would
/// have produced from the rest of this.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Snapshot {
    /// Where this processor sits in the roster firmware published.
    pub index: usize,
    /// How many processors that roster describes.
    pub processors: usize,
    /// The identifier interrupts to this processor are addressed by, which is
    /// the real one.
    pub apic_id: u32,
    /// What is true of the controller that is not one of its registers.
    pub state: State,
    /// The base register, whole.
    pub base: u64,
    /// Which face the guest reaches this controller through.
    pub face: Face,
    /// The identifier register in the shape that face gives it.
    pub id: u32,
    /// The part of the identifier the older face can hold, which is what a
    /// physical destination is matched against there.
    pub xapic_id: u32,
    /// The version register: the version, and how many local vector table
    /// entries this controller has.
    pub version: u32,
    /// The task priority the guest set.
    pub task_priority: Priority,
    /// The priority it is servicing at.
    pub processor_priority: Priority,
    /// The arbitration priority, which exists only in the older face.
    pub arbitration_priority: Priority,
    /// Which logical destinations this controller answers to, in the shape the
    /// face gives that register.
    pub logical_destination: u32,
    /// How a logical destination is matched.
    pub destination_format: u32,
    /// The spurious-interrupt vector register.
    pub spurious: u32,
    /// The error status register as a guest's read answers it, which is what
    /// the last write to it latched.
    pub error_status: u32,
    /// The interrupt command register, both halves.
    pub command: u64,
    /// The local vector table, in the order the architecture counts the
    /// entries, each as a guest's read would answer it.
    pub lvt: [u32; LVT_ENTRIES],
    /// The interrupt request bank.
    pub request: [u32; VECTOR_WORDS],
    /// The in-service bank.
    pub in_service: [u32; VECTOR_WORDS],
    /// The trigger-mode bank.
    pub trigger_mode: [u32; VECTOR_WORDS],
    /// Which requested vectors reached the guest through the pin that bypasses
    /// its controller, and so must not be held in service.
    pub external: [u32; VECTOR_WORDS],
    /// The highest requested vector, whatever its priority.
    pub requested: Option<Vector>,
    /// How many vectors are requested and not yet taken.
    pub requested_count: u32,
    /// How many the guest has taken and not yet acknowledged.
    pub in_service_count: u32,
    /// What this controller has for its guest now, and what only the guest's
    /// own task priority is holding back.
    ///
    /// Taken through the controller's own nomination, which is the same read an
    /// entry makes: it consumes nothing and moves no request, and the only mark
    /// it leaves anywhere is on the word that keeps its own trace line from
    /// repeating.
    pub nomination: Nomination,
    /// How far the bus clock is divided before the timer counts it.
    pub timer_divide: u32,
    /// What the timer counts down from.
    pub timer_initial: u32,
    /// What the real timer has left of the count.
    pub timer_remaining: u32,
    /// The calibrated rate of the timer before division, in ticks per second,
    /// or zero if nothing has measured it.
    pub timer_frequency: u64,
    /// The timestamp the real timer will fire at in deadline mode, in the
    /// machine's own timestamp domain, or zero for a disarmed one.
    pub timer_deadline: u64,
    /// How the timer counts, or `None` for the encoding the architecture
    /// reserves — which a guest can nonetheless write into its timer entry.
    pub timing: Option<Timing>,
    /// Where this processor is in the sequence that starts it.
    pub startup: StartupPhase,
    /// The page a start-up message left it to begin at, if one has.
    pub startup_page: Option<StartupPage>,
    /// How many times the register file has been reset, which is odd while one
    /// is in progress.
    pub epoch: u64,
    /// What real hardware is holding in service for this guest.
    pub ledger: Ledger,
    /// What has happened to this controller since the machine came up.
    pub counted: Counted,
    /// What the machine built hardware-driven delivery on, and what this
    /// controller's own part of it is.
    pub acceleration: Option<Acceleration>,
}

bitflags! {
    /// What is true of one controller that is not one of its registers.
    ///
    /// A flag set rather than a dozen booleans, because that is what it is: each
    /// of these is one bit of the controller's condition, and a reader wants them
    /// together — a processor that is away and not running is a different animal
    /// from one that is neither.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct State: u32 {
        /// This is the machine's bootstrap processor.
        const BOOTSTRAP = 1 << 0;
        /// Firmware described the processor as one that may be started.
        const STARTABLE = 1 << 1;
        /// The guest has software-enabled the controller through its
        /// spurious-vector register.
        const SOFTWARE_ENABLED = 1 << 2;
        /// The controller is in a state that accepts interrupts at all, which
        /// needs both the global enable and the software enable.
        const ACCEPTING = 1 << 3;
        /// Its processor is running the guest rather than waiting to be started.
        const RUNNING = 1 << 4;
        /// Its processor has stopped watching the controller, so a sender must
        /// interrupt it rather than leave a request bit and walk away.
        const AWAY = 1 << 5;
        /// Its processor has claimed the controller, which is what says the host
        /// is running a guest on it.
        const OWNED = 1 << 6;
        /// The controller has been demoted back to software delivery by
        /// something the hardware-driven path reported.
        const AVIC_INHIBITED = 1 << 7;
        /// The controller this guest was told it has offers the timestamp
        /// deadline timer.
        const DEADLINE_OFFERED = 1 << 8;
        /// It offers the face reached through model-specific registers.
        const X2APIC_OFFERED = 1 << 9;
    }
}

impl State {
    /// The condition one controller is in.
    ///
    /// What is deliberately not among these is whether anything has arrived for
    /// the processor that it has not applied yet: the controller answers that
    /// question for the halt path, where it is true of a running processor as
    /// well, so as a line in a report it would say nothing. What a reader wants
    /// is the phase and the page beside it, and [`Snapshot::startup`] carries
    /// both.
    fn of(vlapic: &Vlapic) -> Self {
        let startup = vlapic.startup();
        let model = vlapic.model();
        let mut state = Self::empty();
        state.set(Self::BOOTSTRAP, vlapic.base().bootstrap());
        state.set(Self::STARTABLE, vlapic.startable());
        state.set(Self::SOFTWARE_ENABLED, vlapic.software_enabled());
        state.set(Self::ACCEPTING, vlapic.accepting());
        state.set(Self::RUNNING, startup.running());
        state.set(Self::AWAY, vlapic.away());
        state.set(Self::OWNED, vlapic.owned());
        state.set(Self::AVIC_INHIBITED, vlapic.avic_inhibited());
        state.set(Self::DEADLINE_OFFERED, model.deadline());
        state.set(Self::X2APIC_OFFERED, model.x2apic());
        state
    }
}

/// Which face a guest reaches its controller through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Face {
    /// Switched off: the register page decodes to nothing and the
    /// model-specific registers are not there.
    Disabled,
    /// The page of memory-mapped registers, with eight-bit identifiers.
    XApic,
    /// Model-specific registers, with 32-bit identifiers.
    X2Apic,
}

impl Face {
    /// The face a controller's mode is.
    const fn of(mode: Mode) -> Self {
        match mode {
            Mode::Disabled => Self::Disabled,
            Mode::XApic => Self::XApic,
            Mode::X2Apic => Self::X2Apic,
        }
    }
}

/// How a controller's timer counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Timing {
    /// Counts the initial count down once and stops.
    OneShot,
    /// Counts it down and reloads it, so the interrupt repeats.
    Periodic,
    /// Ignores the counters and fires when the timestamp counter reaches the
    /// deadline register.
    Deadline,
}

impl Timing {
    /// The timing a timer entry's mode is.
    const fn of(mode: TimerMode) -> Self {
        match mode {
            TimerMode::OneShot => Self::OneShot,
            TimerMode::Periodic => Self::Periodic,
            TimerMode::Deadline => Self::Deadline,
        }
    }
}

/// Where a processor is in the sequence that starts it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StartupPhase {
    /// Running the guest, or ready to.
    Running,
    /// Another processor has sent an INIT this one has not applied yet.
    InitRequested,
    /// Reset and held, doing nothing until a start-up message arrives.
    WaitingForSipi,
}

impl StartupPhase {
    /// The phase a startup state is in, and the page a message has left it.
    ///
    /// Both out of one reading, because they are only meaningful together: a
    /// processor that is running has nothing to be started at.
    const fn of(phase: Phase) -> (Self, Option<StartupPage>) {
        match phase {
            Phase::Running => (Self::Running, None),
            Phase::InitRequested(page) => (Self::InitRequested, page),
            Phase::WaitingForSipi(page) => (Self::WaitingForSipi, page),
        }
    }
}

/// What real hardware is holding in service for a guest, and what has become of
/// what it held before.
///
/// Which arm a controller has was decided by what the machine's own controller
/// can do, and it decides what becomes of a debt nobody will acknowledge: one
/// keeps it and leaves a priority class blocked for the life of the machine,
/// the other pays it and stops the vector arriving again until the guest is
/// reset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ledger {
    /// From a controller whose acknowledgement takes no vector.
    Deferred {
        /// Vectors the guest may still acknowledge.
        owed: u32,
        /// Vectors it has acknowledged that are waiting for their turn at the
        /// top of the in-service bank.
        released: u32,
        /// Vectors held with no acknowledgement expected, and so for good.
        abandoned: u32,
        /// How many have been abandoned since the controller was built.
        strandings: u32,
        /// How many came due against a controller that was not holding them.
        phantoms: u32,
    },
    /// From a controller that retires a named vector.
    Immediate {
        /// Vectors the guest may still acknowledge.
        owed: u32,
        /// Vectors this controller has told real hardware not to accept.
        blocked: u32,
        /// How many have been blocked since the controller was built.
        blockings: u32,
    },
}

impl Ledger {
    /// The debts a ledger reported.
    const fn of(debts: Debts) -> Self {
        match debts {
            Debts::Deferred(owing) => Self::Deferred {
                owed: owing.owed,
                released: owing.released,
                abandoned: owing.abandoned,
                strandings: owing.strandings,
                phantoms: owing.phantoms,
            },
            Debts::Immediate(owing) => Self::Immediate {
                owed: owing.owed,
                blocked: owing.blocked,
                blockings: owing.blockings,
            },
        }
    }
}

/// What has happened to one controller since the machine came up.
///
/// Every one of these saturates rather than wrapping, so a count that has
/// stopped moving says "at least this many" instead of reporting a small number
/// for a machine that has seen four billion of something.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counted {
    /// Interrupts that arrived on real hardware and were handed to this guest.
    pub arrivals: u32,
    /// How many of those arrived level triggered.
    pub level: u32,
    /// Interrupts this controller declined, which is its guest's own state
    /// saying no.
    pub declined: u32,
    /// Interrupts this hypervisor was given and recorded in no controller at
    /// all, which is an interrupt lost.
    pub dropped: u32,
    /// Periodic timer counts raised to the shortest period the host will put on
    /// real hardware.
    pub clamped: u32,
    /// Host interrupts this processor sent to make a target look at a request
    /// the hardware left in its backing page.
    pub kicks: u32,
    /// Host interrupts it sent to make a target look at a request the software
    /// path accepted into a target's model.
    pub nudges: u32,
}

impl Counted {
    /// What one controller has counted.
    fn of(vlapic: &Vlapic) -> Self {
        let counts = vlapic.diagnostics().counts();
        Self {
            arrivals: counts.arrivals,
            level: counts.level,
            declined: counts.declined,
            dropped: counts.dropped,
            clamped: counts.clamped,
            kicks: counts.wakes.kicks,
            nudges: counts.wakes.nudges,
        }
    }
}

/// What the machine built hardware-driven delivery on.
///
/// The policy's side of the acceleration and nothing about any control block:
/// what a processor was actually entered with is the block's own to say, and
/// the two differ wherever a transition could not be performed. A machine that
/// never provisioned any of this has no [`Acceleration`] at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Acceleration {
    /// The backing page this processor's controller registers are served out of
    /// while the hardware drives it, or `None` where firmware said the
    /// processor may not be started and it was given none.
    pub own_page: Option<PhysAddr>,
    /// The table of the guest's processors.
    pub physical_table: PhysAddr,
    /// The table resolving a logical destination to one of them.
    pub logical_table: PhysAddr,
    /// The largest valid index the physical table was sized for.
    pub max_index: u16,
    /// Whether the silicon's reading of the running bits is trusted, without
    /// which the bit is never published and every directed interprocessor
    /// interrupt takes the exit-and-kick path.
    pub ipi_virtual: bool,
    /// Whether the machine has been taken off the accelerated path for the rest
    /// of its life.
    pub machine_inhibited: bool,
    /// Whether a guest on this machine may be given the wider controller face
    /// at all, which is what the acceleration can drive rather than what
    /// any one controller is in.
    pub x2apic_permitted: bool,
}

impl Acceleration {
    /// What the machine provisioned, or `None` where it provisioned nothing.
    fn of() -> Option<Self> {
        let policy = activation::policy()?;
        Some(Self {
            own_page: activation::own_page().ok(),
            physical_table: policy.physical_table,
            logical_table: policy.logical_table,
            max_index: policy.max_index,
            ipi_virtual: policy.ipi_virtual,
            machine_inhibited: policy.machine_inhibited,
            x2apic_permitted: activation::x2avic_permitted(),
        })
    }
}
