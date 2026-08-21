//! Everything one processor's interrupt controllers hold, as it travels to a
//! reader inside the guest.
//!
//! Three controllers, really, and the point of the dump is that they are
//! different and can disagree: the emulated one the guest programs, the
//! machine's own one underneath it that every interrupt actually arrives at,
//! and — where the processor is delivering the guest's interrupts itself — the
//! page and tables the hardware reads instead of either. A lost interrupt is
//! almost always one of them holding something the other two do not.
//!
//! # Absence is a value here
//!
//! A machine that never provisioned hardware delivery has no backing page, a
//! controller in the older face has no logical destination of the wider one's
//! shape, and a register the host could not read is not a register that holds
//! zero. So every part of this that may be missing says whether it is there:
//! [`Present`] for whole sections, the flag words for the questions that are
//! yes-or-no, and [`ABSENT`] for the byte-wide fields that name a vector or a
//! page. Nothing is zero-filled and left to look like state.
//!
//! # Nothing here is decoded
//!
//! Every register is carried as the word it is, and the enumerations below are
//! how a reader turns one into a name. That is deliberate: a dump is evidence,
//! and a host that decoded a register into a meaning would be putting its own
//! reading of the hardware between the reader and what the hardware said.

use bitflags::bitflags;
use thiserror::Error;

use crate::{VERSION, Wire};

/// Everything the calling processor's interrupt controllers hold.
///
/// The answer to [`Command::APIC_DUMP`](crate::Command::APIC_DUMP), written
/// into the buffer the guest named. The header comes first in the layout and is
/// written *last* by the host, so a reader that finds the magic knows every
/// section the header claims is really there.
#[derive(Debug)]
#[repr(C)]
pub struct ApicDump {
    /// What this is, and which of the sections below hold anything.
    pub header: Header,
    /// The controller the guest programs.
    pub emulated: Emulated,
    /// The machine's own controller behind it.
    pub real: Real,
    /// The structures hardware-driven delivery runs on.
    pub avic: Avic,
    /// The whole of this processor's backing page, exactly as it stands: the
    /// register file the hardware serves the guest out of while it drives the
    /// controller, with the request, in-service and trigger-mode banks at their
    /// architectural offsets inside it.
    pub backing: [u8; PAGE_BYTES],
    /// The physical table's entries, one per identifier an interrupt may name,
    /// as many of them as the table describes and this buffer can hold.
    ///
    /// Each is a `PhysicalApicEntry`: whether the entry names a processor at
    /// all, whether that processor is in the guest right now, which physical
    /// processor is running it, and where its controller registers are backed.
    pub physical: [u64; PHYSICAL_ENTRIES],
    /// The logical table's entries, each naming the processor one logical
    /// destination resolves to, or naming none.
    pub logical: [u32; LOGICAL_ENTRIES],
}

impl ApicDump {
    /// A buffer with nothing in it, which is what a caller passes in.
    ///
    /// Zeroed rather than uninitialized for two reasons. The caller reads it
    /// back whatever the host did with it — a refused call leaves every
    /// byte of it as it was, and zeroes are a buffer that fails
    /// [`ApicDump::validate`] rather than one that reads as a controller.
    /// And writing every page of it is what makes the host able to reach it
    /// at all: a guest's memory is described to the hypervisor as the guest
    /// touches it, and a buffer nothing has touched is answered with
    /// [`Status::Unreachable`](crate::Status::Unreachable).
    #[must_use]
    pub fn zeroed() -> Self {
        Self {
            header: Header::default(),
            emulated: Emulated::default(),
            real: Real::default(),
            avic: Avic::default(),
            backing: [0; PAGE_BYTES],
            physical: [0; PHYSICAL_ENTRIES],
            logical: [0; LOGICAL_ENTRIES],
        }
    }

    /// Whether this holds a dump this build of the ABI can read.
    ///
    /// Asked after a served call and before anything else is read out of the
    /// buffer. The three questions are the three ways a buffer can hold
    /// something that is not this: nothing at all, a dump from a host built
    /// against another version of the interface, and one whose structure is a
    /// different size than the one this reader compiled against.
    ///
    /// # Errors
    ///
    /// [`Mismatch::Magic`] for a buffer no host wrote a dump into,
    /// [`Mismatch::Version`] for a version this build does not read, or
    /// [`Mismatch::Size`] for the same version at a different size — which is
    /// one of the two sides having been rebuilt without the version being
    /// moved.
    pub fn validate(&self) -> Result<(), Mismatch> {
        if self.header.magic != Header::MAGIC {
            return Err(Mismatch::Magic {
                magic: self.header.magic,
            });
        }
        if self.header.version != u32::from(VERSION) {
            return Err(Mismatch::Version {
                found: self.header.version,
            });
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the assertions below hold this structure to a few kilobytes"
        )]
        let expected = size_of::<Self>() as u32;
        if self.header.bytes != expected {
            return Err(Mismatch::Size {
                found: self.header.bytes,
                expected,
            });
        }
        Ok(())
    }
}

/// Why a buffer does not hold a dump this build can read.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum Mismatch {
    /// No host wrote a dump here. The commonest cause is a call that was
    /// refused, which writes nothing at all.
    #[error("no dump was written here (magic {magic:#018x})")]
    Magic {
        /// What stood where the magic should have been.
        magic: u64,
    },
    /// A host built against another version of this interface.
    #[error("the dump is version {found}, and this reads version {expected}", expected = VERSION)]
    Version {
        /// The version the host wrote.
        found: u32,
    },
    /// The same version at a different size: one side was rebuilt without
    /// [`VERSION`] being moved.
    #[error("the dump is {found} bytes, expected {expected}")]
    Size {
        /// The size the host wrote.
        found: u32,
        /// The size this reader was compiled against.
        expected: u32,
    },
}

/// What a buffer holds, and which of the sections after it hold anything.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Header {
    /// [`Header::MAGIC`]. First field so that a reader can tell a written
    /// buffer from its own zeroes before trusting anything else in it.
    pub magic: u64,
    /// Which of the sections below the host filled in.
    pub present: Present,
    /// The [`VERSION`] of the interface the host that wrote this implements.
    pub version: u32,
    /// `size_of::<ApicDump>()` as the host saw it.
    pub bytes: u32,
    /// Where the processor this describes sits in the roster firmware
    /// published.
    pub cpu_index: u32,
    /// The identifier its real controller answers to, which is the one every
    /// interrupt on the machine is routed by.
    pub apic_id: u32,
    /// The privilege level the caller made the call at.
    ///
    /// Carried because it is worth knowing that it did not have to be zero:
    /// the instruction has no privilege restriction, so an ordinary process
    /// gets the same answer a kernel would.
    pub cpl: u32,
    /// How many processors the machine's roster describes, this one included.
    pub processors: u32,
}

impl Header {
    /// Identifies a written dump. `"PULZAPIC"`, chosen to be legible in a hex
    /// dump of a buffer whose other possible contents are whatever the guest
    /// had there.
    pub const MAGIC: u64 = u64::from_le_bytes(*b"PULZAPIC");

    /// The header for a dump of this processor, naming the sections that were
    /// filled in.
    #[must_use]
    pub fn new(present: Present, cpu_index: u32, apic_id: u32, cpl: u32, processors: u32) -> Self {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the assertions below hold the dump to a few kilobytes"
        )]
        let bytes = size_of::<ApicDump>() as u32;
        Self {
            magic: Self::MAGIC,
            present,
            version: u32::from(VERSION),
            bytes,
            cpu_index,
            apic_id,
            cpl,
            processors,
        }
    }
}

impl Wire for Header {}

/// The controller the guest programs, as the emulated register file holds it.
///
/// Every register here is what a guest's own read would answer with, which for
/// the two that are computed rather than stored — the processor and arbitration
/// priorities — means the number its processor would have produced from the
/// rest of this state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Emulated {
    /// The controller's base register, whole: the face in use, the global
    /// enable, whether this is the bootstrap processor, and where the register
    /// page is.
    pub base: u64,
    /// The interrupt command register, both halves, as the guest last wrote it.
    pub command: u64,
    /// The timestamp the real timer will fire at where the guest's timer is in
    /// deadline mode, in the machine's own timestamp domain rather than the
    /// guest's, or zero for a disarmed one.
    ///
    /// The machine's, because that is the register the host actually armed: the
    /// guest's own domain is that number plus this processor's timestamp
    /// offset, and a dump reports what the hardware holds rather than a
    /// translation of it.
    pub timer_deadline: u64,
    /// The calibrated rate of the timer before division, in ticks per second,
    /// or zero if nothing has measured it yet.
    pub timer_frequency: u64,
    /// How many times this controller's register file has been reset, which is
    /// odd while one is in progress.
    pub epoch: u64,
    /// The boolean state of the controller that is not a register.
    pub flags: EmulatedFlags,
    /// Which face the controller is in, as [`Mode`] encodes it.
    pub mode: u32,
    /// The identifier register in that face: the whole identifier in the wider
    /// one, and the top byte of the register in the older one.
    pub id: u32,
    /// The part of the identifier the older face can hold, which is what a
    /// physical destination is matched against there. It differs from
    /// [`Emulated::id`] on a machine whose identifiers do not fit a byte.
    pub xapic_id: u32,
    /// The version register: the controller's version, and how many local
    /// vector table entries it has.
    pub version: u32,
    /// The task priority the guest set.
    pub task_priority: u32,
    /// The priority the controller is servicing at, computed from the task
    /// priority and what is in service.
    pub processor_priority: u32,
    /// The arbitration priority, which exists only in the older face and is
    /// computed from the task priority, what is in service and what is
    /// requested.
    pub arbitration_priority: u32,
    /// Which logical destinations this controller answers to, in the shape the
    /// face in use gives that register.
    pub logical_destination: u32,
    /// How a logical destination is matched, which the wider face does not
    /// have.
    pub destination_format: u32,
    /// The spurious-interrupt vector register, which also carries the bit that
    /// software-enables the controller.
    pub spurious: u32,
    /// The error status register as a guest's read would answer it: what the
    /// last write to it latched, rather than what has been noticed since.
    pub error_status: u32,
    /// How far the bus clock is divided before the timer counts it.
    pub timer_divide: u32,
    /// What the timer counts down from.
    pub timer_initial: u32,
    /// What the real timer has left of the count, which is read from the
    /// hardware rather than remembered.
    pub timer_remaining: u32,
    /// How the timer counts, as [`TimerMode`] encodes it.
    pub timer_mode: u32,
    /// How many vectors are requested and not yet taken.
    pub requested_count: u32,
    /// How many the guest has taken and not yet acknowledged.
    pub in_service_count: u32,
    /// The highest requested vector whatever its priority, or [`ABSENT`].
    pub requested: u32,
    /// The vector this controller would give the guest now, or [`ABSENT`] where
    /// nothing is requested, what is requested does not outrank what is in
    /// service, or the guest's own task priority holds it back.
    pub deliverable: u32,
    /// The vector nothing but the guest's own task priority is holding back,
    /// which is what an interrupt window is armed for, or [`ABSENT`].
    pub blocked: u32,
    /// Where this processor is in the startup sequence, as [`Startup`] encodes
    /// it.
    pub startup: u32,
    /// The page a start-up message left this processor to begin at, or
    /// [`ABSENT`] where it is not waiting for one or none has arrived.
    pub startup_page: u32,
    /// Interrupts that arrived on real hardware and were handed to this guest.
    pub arrivals: u32,
    /// How many of those arrived level triggered, and so had an
    /// acknowledgement withheld or were deliberately acknowledged at once.
    pub level: u32,
    /// Interrupts this controller declined, which is its guest's own state
    /// saying no rather than anything going wrong.
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
    /// Which ledger this controller settles real hardware's acknowledgements
    /// through, as [`DebtKind`] encodes it — and therefore which of the debt
    /// counts below mean anything.
    pub debt_kind: u32,
    /// Vectors real hardware is holding in service for this guest that the
    /// guest may still acknowledge. Both ledgers count this.
    pub debts_owed: u32,
    /// Vectors the guest has acknowledged that are waiting for their turn at
    /// the top of the in-service bank. [`DebtKind::Deferred`] only.
    pub debts_released: u32,
    /// Vectors held with no acknowledgement expected, and so held for good.
    /// [`DebtKind::Deferred`] only.
    pub debts_abandoned: u32,
    /// How many have been abandoned since the controller was built.
    /// [`DebtKind::Deferred`] only.
    pub debts_strandings: u32,
    /// How many came due against a controller that was not holding them.
    /// [`DebtKind::Deferred`] only.
    pub debts_phantoms: u32,
    /// Vectors this controller has told real hardware not to accept.
    /// [`DebtKind::Immediate`] only.
    pub debts_blocked: u32,
    /// How many have been blocked since the controller was built.
    /// [`DebtKind::Immediate`] only.
    pub debts_blockings: u32,
    /// The local vector table, in the order the architecture counts the
    /// entries: timer, both interrupt pins, error, performance, thermal,
    /// corrected machine check. Each as a guest's read would answer it, so the
    /// two bits the controller owns are the real controller's own report.
    pub lvt: [u32; LVT_ENTRIES],
    /// The interrupt request bank: one bit per vector, thirty-two to a slot,
    /// for interrupts accepted and not yet taken.
    pub request: [u32; BANK_SLOTS],
    /// The in-service bank: what the guest has taken and not acknowledged.
    pub in_service: [u32; BANK_SLOTS],
    /// The trigger-mode bank: which of them arrived level triggered.
    pub trigger_mode: [u32; BANK_SLOTS],
    /// Which requested vectors reached this guest through the pin that bypasses
    /// its controller, and so must not be held in service when it takes them.
    /// Not an architectural register — the host's own record.
    pub external: [u32; BANK_SLOTS],
}

impl Wire for Emulated {}

/// The machine's own controller behind the emulated one.
///
/// Not everything a controller has: only what this hypervisor's own driver can
/// read back. The registers a guest would find on real hardware and that are
/// missing here — its priorities, its spurious vector, its error status, its
/// request bank — are ones nothing in the host reads today, and a dump cannot
/// invent a way to read them.
///
/// What it does carry is the part that decides where a guest's interrupts
/// really come from: which of them this controller is holding, how they
/// arrived, and what each of its own sources is programmed to do.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Real {
    /// What is true of this controller that is not a register.
    pub flags: RealFlags,
    /// The timestamp its timer will fire at, where the timer is in deadline
    /// mode and the register exists, in the machine's own timestamp domain.
    pub timer_deadline: u64,
    /// Which interface it is being driven through, as [`Mode`] encodes it.
    pub mode: u32,
    /// The identifier it answers to.
    pub id: u32,
    /// Its version register.
    pub version: u32,
    /// How many local vector table entries it reports.
    pub entries: u32,
    /// Which logical destinations it answers to.
    pub logical_destination: u32,
    /// What its extended register space offers, as this hypervisor found it.
    pub extended: u32,
    /// How its timer counts, as [`TimerMode`] encodes it.
    pub timer_mode: u32,
    /// What its timer counts down from.
    pub timer_initial: u32,
    /// What its timer has left.
    pub timer_remaining: u32,
    /// The highest-priority vector it is holding in service, which is the one
    /// an acknowledgement would retire, or [`ABSENT`].
    pub in_service_top: u32,
    /// Its in-service bank: one bit per vector, thirty-two to a slot.
    pub in_service: [u32; BANK_SLOTS],
    /// Its trigger-mode bank, which is the machine's own record of how each
    /// interrupt it accepted arrived.
    pub trigger_mode: [u32; BANK_SLOTS],
    /// Each of its own sources, in the order [`SourceState`] documents, as far
    /// as the driver can report one: the raw entry word is not readable through
    /// it, so what is carried is the state of the bits that are.
    pub sources: [SourceState; SOURCES],
}

impl Wire for Real {}

/// The structures hardware-driven delivery runs on, and whether it is running.
///
/// Two authorities are described here and they are worth keeping apart. The
/// *control block* is what the processor was entered with, and what it says is
/// what actually happened on the run that has just ended. The *policy* is what
/// the machine built at boot and would like to be driving: they differ wherever
/// a transition could not be performed, wherever another processor demoted the
/// machine, and before the first entry that arms anything.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Avic {
    /// Which of the questions about hardware delivery hold.
    pub flags: AvicFlags,
    /// The guest physical address the control block has the controller's
    /// register page appearing at.
    pub apic_bar: u64,
    /// The backing page the control block names.
    pub backing_page: u64,
    /// The backing page provisioning gave this processor, which is what the
    /// next armed entry would name.
    pub own_backing_page: u64,
    /// The logical table the control block names.
    pub logical_table: u64,
    /// The physical table the control block names.
    pub physical_table: u64,
    /// The physical table provisioning built.
    pub policy_physical_table: u64,
    /// The logical table provisioning built.
    pub policy_logical_table: u64,
    /// The largest valid index the control block publishes beside its physical
    /// table, which is how far the hardware walks it.
    pub max_index: u32,
    /// The largest index provisioning sized that table for, which is the widest
    /// face the machine may drive it in.
    pub policy_max_index: u32,
    /// How many entries of [`ApicDump::physical`] the host filled in.
    pub physical_entries: u32,
    /// How many entries of [`ApicDump::logical`] it filled in.
    pub logical_entries: u32,
}

impl Wire for Avic {}

bitflags! {
    /// Which sections of a dump hold anything.
    ///
    /// A clear bit is not a section of zeroes: it says the host could not read
    /// that part of the machine at all, and a reader must say so rather than
    /// report the zeroes as state.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    #[repr(transparent)]
    pub struct Present: u64 {
        /// The emulated controller's register file was read.
        const EMULATED = 1 << 0;
        /// The machine's own controller was reached.
        const REAL = 1 << 1;
        /// The state of hardware-driven delivery was read.
        const AVIC = 1 << 2;
        /// The backing page's bytes were copied.
        const BACKING = 1 << 3;
        /// The physical table's entries were copied.
        const PHYSICAL = 1 << 4;
        /// The logical table's entries were copied.
        const LOGICAL = 1 << 5;
    }

    /// What is true of the emulated controller that is not one of its
    /// registers.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    #[repr(transparent)]
    pub struct EmulatedFlags: u64 {
        /// This is the machine's bootstrap processor.
        const BOOTSTRAP = 1 << 0;
        /// The guest has software-enabled the controller through its
        /// spurious-vector register.
        const SOFTWARE_ENABLED = 1 << 1;
        /// The controller is in a state that accepts interrupts at all, which
        /// needs both the global enable and the software enable.
        const ACCEPTING = 1 << 2;
        /// Its processor is running the guest rather than waiting to be
        /// started.
        const RUNNING = 1 << 3;
        /// Its processor has stopped watching the controller, so a sender must
        /// interrupt it rather than leave a request bit and walk away.
        const AWAY = 1 << 4;
        /// Its processor has claimed the controller, which is what says the
        /// host is running a guest on it.
        const OWNED = 1 << 5;
        /// Firmware described the processor as one that may be started.
        const STARTABLE = 1 << 6;
        /// The controller has been demoted back to software delivery by
        /// something the hardware-driven path reported.
        const AVIC_INHIBITED = 1 << 7;
        /// The controller this guest was told it has offers the timestamp
        /// deadline timer.
        const DEADLINE_OFFERED = 1 << 8;
        /// It reports the face reached through model-specific registers.
        const X2APIC_OFFERED = 1 << 9;
    }

    /// What is true of the machine's own controller that is not one of its
    /// registers.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    #[repr(transparent)]
    pub struct RealFlags: u64 {
        /// Its timestamp-deadline register exists and [`Real::timer_deadline`]
        /// was read out of it.
        const DEADLINE = 1 << 0;
        /// Its extended register space can retire a named vector and stop
        /// another being accepted, which is what decides how the host settles
        /// what it owes.
        const EXTENDED = 1 << 1;
    }

    /// What is true of hardware-driven delivery on this processor.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    #[repr(transparent)]
    pub struct AvicFlags: u64 {
        /// The machine built the structures at boot, which is the policy having
        /// chosen hardware delivery for it.
        const PROVISIONED = 1 << 0;
        /// This processor has a backing page of its own.
        const OWN_PAGE = 1 << 1;
        /// The control block this processor was entered with carries hardware
        /// delivery.
        const ACCELERATED = 1 << 2;
        /// It carries it in the face a guest reaches its controller through
        /// model-specific registers.
        const WIDER_FACE = 1 << 3;
        /// The block's own enable bit.
        const BLOCK_ENABLE = 1 << 4;
        /// The block's enable bit for 32-bit identifiers.
        const BLOCK_X2_ENABLE = 1 << 5;
        /// The backing page holds something this guest could take at this
        /// instant.
        const DELIVERABLE = 1 << 6;
        /// That question could be answered at all, which needs the page to have
        /// been reachable.
        const DELIVERABLE_KNOWN = 1 << 7;
        /// A guest on this machine may be given the wider controller face.
        const X2APIC_OFFERED = 1 << 8;
        /// The silicon's reading of the running bits is trusted, without which
        /// the running bit is never published and every directed interprocessor
        /// interrupt takes the exit-and-kick path.
        const IPI_VIRTUAL = 1 << 9;
        /// The machine has been taken off the accelerated path for the rest of
        /// its life.
        const MACHINE_INHIBITED = 1 << 10;
        /// The physical table describes more entries than a dump carries, so
        /// [`ApicDump::physical`] holds the first [`PHYSICAL_ENTRIES`] of them.
        const PHYSICAL_TRUNCATED = 1 << 11;
    }

    /// What the host can report about one of the real controller's own sources.
    ///
    /// The raw entry word is not among them: the driver exposes the three bits
    /// below and no way to read the register whole.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    #[repr(transparent)]
    pub struct SourceState: u32 {
        /// The entry was readable, so the rest of this means anything.
        const READ = 1 << 0;
        /// The source delivers nothing.
        const MASKED = 1 << 1;
        /// The controller has accepted a delivery from it and not yet handed it
        /// to the processor.
        const PENDING = 1 << 2;
        /// A level-triggered interrupt from this pin has been accepted and not
        /// yet acknowledged. Meaningless outside the two pins.
        const REMOTE_IRR = 1 << 3;
    }
}

/// Which interface a controller is reached through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Mode {
    /// The controller is switched off entirely, and its guest has none.
    Disabled = 0,
    /// The older face: one page of memory-mapped registers, eight-bit
    /// identifiers.
    XApic = 1,
    /// The wider face: the same registers as model-specific ones, 32-bit
    /// identifiers.
    X2Apic = 2,
}

impl Mode {
    /// The face a word names, or `None` for an encoding this ABI does not
    /// define.
    #[must_use]
    pub const fn from_word(word: u32) -> Option<Self> {
        Some(match word {
            0 => Self::Disabled,
            1 => Self::XApic,
            2 => Self::X2Apic,
            _ => return None,
        })
    }

    /// The word this face travels as.
    #[must_use]
    pub const fn word(self) -> u32 {
        self as u32
    }

    /// What to call it in a report.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::XApic => "xapic",
            Self::X2Apic => "x2apic",
        }
    }
}

/// How a controller's timer counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum TimerMode {
    /// Counts the initial count down once and stops.
    OneShot = 0,
    /// Counts it down and reloads it, so the interrupt repeats.
    Periodic = 1,
    /// Ignores the counters and fires when the timestamp counter reaches the
    /// deadline register.
    Deadline = 2,
    /// The encoding the architecture reserves, which a guest can nonetheless
    /// write into its timer entry.
    Reserved = 3,
}

impl TimerMode {
    /// The mode a word names, or `None` for an encoding this ABI does not
    /// define.
    #[must_use]
    pub const fn from_word(word: u32) -> Option<Self> {
        Some(match word {
            0 => Self::OneShot,
            1 => Self::Periodic,
            2 => Self::Deadline,
            3 => Self::Reserved,
            _ => return None,
        })
    }

    /// The word this mode travels as.
    #[must_use]
    pub const fn word(self) -> u32 {
        self as u32
    }

    /// What to call it in a report.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::OneShot => "one-shot",
            Self::Periodic => "periodic",
            Self::Deadline => "deadline",
            Self::Reserved => "reserved",
        }
    }
}

/// Where a processor is in the sequence that starts it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Startup {
    /// Running the guest, or ready to.
    Running = 0,
    /// Another processor has sent an INIT this one has not applied yet.
    InitRequested = 1,
    /// Reset and held, doing nothing until a start-up message arrives.
    WaitingForSipi = 2,
}

impl Startup {
    /// The phase a word names, or `None` for an encoding this ABI does not
    /// define.
    #[must_use]
    pub const fn from_word(word: u32) -> Option<Self> {
        Some(match word {
            0 => Self::Running,
            1 => Self::InitRequested,
            2 => Self::WaitingForSipi,
            _ => return None,
        })
    }

    /// The word this phase travels as.
    #[must_use]
    pub const fn word(self) -> u32 {
        self as u32
    }

    /// What to call it in a report.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::InitRequested => "init requested",
            Self::WaitingForSipi => "waiting for a start-up message",
        }
    }
}

/// Which ledger a controller settles real hardware's acknowledgements through.
///
/// The two are not variants of a preference: which one a controller has was
/// decided by what its machine's own controller can do, and it decides what
/// becomes of a debt nobody will acknowledge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum DebtKind {
    /// A controller whose acknowledgement takes no vector, so a payment waits
    /// for its vector to reach the top of the in-service bank and a debt nobody
    /// will discharge is never paid at all.
    Deferred = 0,
    /// A controller that retires a named vector and can stop one being
    /// accepted, so every debt is settled the moment there is licence to.
    Immediate = 1,
}

impl DebtKind {
    /// The ledger a word names, or `None` for an encoding this ABI does not
    /// define.
    #[must_use]
    pub const fn from_word(word: u32) -> Option<Self> {
        Some(match word {
            0 => Self::Deferred,
            1 => Self::Immediate,
            _ => return None,
        })
    }

    /// The word this ledger travels as.
    #[must_use]
    pub const fn word(self) -> u32 {
        self as u32
    }

    /// What to call it in a report.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Deferred => "deferred",
            Self::Immediate => "immediate",
        }
    }
}

/// What a field that names a vector or a page holds when there is none.
///
/// One past the widest value any of them can carry, so it cannot be mistaken
/// for one — where zero would be vector zero and page zero, both of which a
/// machine can really be holding.
pub const ABSENT: u32 = 0x100;

/// How many bytes of a backing page a dump carries, which is all of it.
pub const PAGE_BYTES: usize = 4096;

/// How many slots each of the vector banks has: thirty-two vectors to a slot,
/// and two hundred and fifty-six vectors.
pub const BANK_SLOTS: usize = 8;

/// How many local vector table entries a controller has at most, which is how
/// many a dump carries.
pub const LVT_ENTRIES: usize = 7;

/// How many of the real controller's own sources the host can report.
///
/// The error entry is not among them: it belongs to the host, which is the only
/// thing that can act on a controller reporting its own errors, and the driver
/// does not offer it. The order is the timer, both interrupt pins, the thermal
/// sensor, the performance counters, and corrected machine-check errors.
pub const SOURCES: usize = 6;

/// How many physical-table entries a dump carries.
///
/// The table itself can describe up to four thousand and ninety-six, which is
/// as far as the twelve-bit index beside its address reaches, and dumping all
/// of them would be thirty-two kilobytes of mostly nothing. This is the widest
/// a machine using 32-bit identifiers without the extended table can index, and
/// a table longer than it is reported as truncated rather than silently cut.
pub const PHYSICAL_ENTRIES: usize = 512;

/// How many logical-table entries a dump carries: fifteen clusters of four,
/// which is what the acceleration publishes a guest's logical identities into.
pub const LOGICAL_ENTRIES: usize = 60;

/// Every structure of this ABI is a padding-free run of unsigned integers, and
/// the sizes below are what says so: each is exactly the sum of its fields, so
/// every byte of one is a byte of a field. That is what makes
/// [`Wire`](crate::Wire) sound, and it is checked here rather than trusted,
/// because a field inserted in the wrong place would otherwise introduce
/// padding whose bytes a host would hand a guest uninitialized.
const _: () = assert!(
    size_of::<Header>() == 40
        && size_of::<Emulated>() == 352
        && size_of::<Real>() == 144
        && size_of::<Avic>() == 80,
    "a section of this ABI has changed size, so its fields no longer pack without padding",
);

/// And the dump is its sections and its three copied regions, with nothing
/// between them: every section's size is a multiple of a quadword, so nothing
/// after one is realigned.
const _: () = assert!(
    size_of::<ApicDump>()
        == size_of::<Header>()
            + size_of::<Emulated>()
            + size_of::<Real>()
            + size_of::<Avic>()
            + PAGE_BYTES
            + PHYSICAL_ENTRIES * size_of::<u64>()
            + LOGICAL_ENTRIES * size_of::<u32>(),
    "the dump has padding between its sections",
);

/// A quadword is what every structure here is aligned to, which is what
/// [`ALIGNMENT`](crate::ALIGNMENT) requires of a buffer: a reader could not
/// name the fields of a dump that arrived anywhere else.
const _: () = assert!(
    align_of::<ApicDump>() as u64 == crate::ALIGNMENT,
    "a dump's alignment is what a buffer is required to have",
);

/// The banks are the whole of a vector's range, and the marker for a field that
/// names none is outside it: a reader that could not tell the two apart would
/// report vector zero for a controller holding nothing.
const _: () = assert!(
    BANK_SLOTS * u32::BITS as usize == ABSENT as usize && ABSENT > u8::MAX as u32,
    "a bank does not cover a vector's range, or the absent marker is one",
);

#[cfg(test)]
mod tests {
    //! What a reader depends on: a header it can tell from anything else in its
    //! own buffer, an encoding for every name it prints, and bytes that are the
    //! structure they claim to be.

    use super::{
        ABSENT, ApicDump, DebtKind, Emulated, Header, Mismatch, Mode, Present, Real, Startup,
        TimerMode,
    };
    use crate::{VERSION, Wire};

    /// What a header says a dump is, as the tests spell it rather than as the
    /// header's own constructor computes it.
    fn bytes() -> u32 {
        u32::try_from(size_of::<ApicDump>()).expect("a dump is a few kilobytes")
    }

    #[test]
    fn a_written_header_is_the_one_this_build_reads() {
        let mut dump = ApicDump::zeroed();
        dump.header = Header::new(Present::all(), 3, 0x21, 3, 8);

        assert_eq!(dump.validate(), Ok(()));
        assert_eq!(dump.header.version, u32::from(VERSION));
        assert_eq!(dump.header.bytes, bytes());
        assert_eq!(
            dump.header.cpl, 3,
            "a dump is served to any privilege level"
        );
    }

    #[test]
    fn an_untouched_buffer_is_not_read_as_a_controller() {
        // The whole point of the magic: a refused call writes nothing, so what a
        // reader has in its hands is its own zeroes, and every field of them
        // would otherwise read as a controller holding nothing.
        assert_eq!(
            ApicDump::zeroed().validate(),
            Err(Mismatch::Magic { magic: 0 })
        );
    }

    #[test]
    fn a_dump_from_another_build_is_refused_rather_than_misread() {
        let mut dump = ApicDump::zeroed();
        dump.header = Header::new(Present::empty(), 0, 0, 0, 1);
        dump.header.version = u32::from(VERSION) + 1;
        assert_eq!(
            dump.validate(),
            Err(Mismatch::Version {
                found: u32::from(VERSION) + 1
            })
        );

        dump.header.version = u32::from(VERSION);
        dump.header.bytes -= 8;
        assert_eq!(
            dump.validate(),
            Err(Mismatch::Size {
                found: bytes() - 8,
                expected: bytes(),
            })
        );
    }

    #[test]
    fn every_name_a_reader_prints_has_an_encoding_that_round_trips() {
        for mode in [Mode::Disabled, Mode::XApic, Mode::X2Apic] {
            assert_eq!(Mode::from_word(mode.word()), Some(mode));
            assert!(!mode.name().is_empty());
        }
        for mode in [
            TimerMode::OneShot,
            TimerMode::Periodic,
            TimerMode::Deadline,
            TimerMode::Reserved,
        ] {
            assert_eq!(TimerMode::from_word(mode.word()), Some(mode));
        }
        for phase in [
            Startup::Running,
            Startup::InitRequested,
            Startup::WaitingForSipi,
        ] {
            assert_eq!(Startup::from_word(phase.word()), Some(phase));
        }
        for kind in [DebtKind::Deferred, DebtKind::Immediate] {
            assert_eq!(DebtKind::from_word(kind.word()), Some(kind));
        }
        // And a word from a host that reported something this build does not
        // know is not quietly read as the first variant.
        assert_eq!(Mode::from_word(ABSENT), None);
        assert_eq!(TimerMode::from_word(ABSENT), None);
        assert_eq!(Startup::from_word(ABSENT), None);
        assert_eq!(DebtKind::from_word(ABSENT), None);
    }

    #[test]
    fn a_section_travels_as_exactly_its_own_bytes() {
        // The property the copy-out rests on: what the host writes for a section
        // is the section's whole layout and nothing else, so a reader finds every
        // field where its own build puts it.
        assert_eq!(Header::default().wire().len(), size_of::<Header>());
        assert_eq!(Emulated::default().wire().len(), size_of::<Emulated>());
        assert_eq!(Real::default().wire().len(), size_of::<Real>());

        let header = Header::new(Present::EMULATED, 1, 2, 0, 4);
        let bytes = header.wire();
        assert_eq!(
            u64::from_le_bytes(
                bytes[..8]
                    .try_into()
                    .expect("the magic is the first quadword")
            ),
            Header::MAGIC
        );
    }
}
