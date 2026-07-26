//! The local interrupt controller each processor has, and starting the ones
//! that are not running yet.
//!
//! Every processor has a local APIC, and nothing reaches another processor
//! except through it. That makes this crate the floor under two quite different
//! things: the interrupts a processor sends and receives, and the sequence that
//! takes a processor from reset to running hypervisor code.
//!
//! # Two interfaces, one register file
//!
//! The controller answers through a page of memory-mapped registers, or — where
//! the processor implements it — through model-specific registers, which is
//! faster and is the only way to address a processor whose identifier does not
//! fit eight bits. [`register`] is where the two become one: there is a single
//! list of registers and the model-specific index is derived from the
//! memory-mapped offset, so neither interface can drift away from the other.
//!
//! The choice is the machine's rather than a processor's. Every processor is
//! put into the same mode, because a mode is also a destination width, and a
//! machine where some processors could be addressed and others could not is not
//! one anything above here should have to reason about.
//!
//! # What this crate does not do
//!
//! It does not decide what happens when an interrupt arrives. Vectors, handlers
//! and what an unclaimed interrupt means belong to [`descriptors`] and to
//! whoever is using the machine; [`Timer::arm`] programs a timer and registers
//! nothing.
//!
//! Two vectors are the exception, and they are the two that are the
//! controller's own rather than the platform's: the spurious vector, which is
//! what a withdrawn interrupt arrives on, and the vector the controller reports
//! its own errors through. Nothing else can know those are ours, and a spurious
//! interrupt reaching the unclaimed callback would stop the machine over an
//! event the architecture describes as normal.
//!
//! # Starting the others
//!
//! [`start`] is the whole of it, and what it needs is a page below one megabyte
//! to put a trampoline in — reserved by the loader, because firmware still owns
//! low memory and parks its own idle processors in it. It is deliberately not
//! tied to bring-up: nothing it uses is alive only then, so processors can be
//! started at any later point, and leaving it uncalled leaves a working machine
//! with one processor.

#![no_std]

extern crate alloc;

mod base;
mod capture;
mod icr;
mod lvt;
mod pic;
mod register;
mod smp;
mod timer;
mod trampoline;

use alloc::vec::Vec;

use acpi::{Madt, NmiTarget};
use cpu::{ApicId, CpuError};
use descriptors::{DescriptorError, Disposition, Interrupt, Vector};
use log::{info, warn};
use paging::{AddressSpace, CacheType, PagingError, Protection};
use processor::Features;
use spin::Once;
use thiserror::Error;

use crate::register::{Access, Register};
pub use crate::{
    capture::{FirmwareState, LocalState, VECTOR_WORDS, capture},
    icr::{Command, Delivery, Target},
    lvt::{Delivery as LvtDelivery, Entry, Polarity, Trigger},
    smp::{Started, start},
    timer::{Divisor, Mode as TimerMode, Timer},
};

/// The vector a withdrawn interrupt arrives on.
///
/// The highest there is, which is the convention and is also the choice with no
/// cost: nothing else wants it, and what arrives on it is by definition
/// something that has already been given up on.
pub const SPURIOUS: Vector = Vector::new(0xFF);

/// The vector the controller reports its own errors on.
pub const ERROR: Vector = Vector::new(0xFE);

/// Which of the two interfaces the machine's controllers present.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// The page of memory-mapped registers, with eight-bit identifiers.
    XApic,
    /// Model-specific registers, with 32-bit identifiers.
    X2Apic,
}

/// The machine's local interrupt controllers.
///
/// Produced once, by the boot processor. What it holds is what is true of all
/// of them; a particular processor's controller is a [`LocalApic`], which is
/// nothing but a handle for whichever processor asks.
#[derive(Clone, Copy, Debug)]
pub struct Apic {
    mode: Mode,
    masked_8259: bool,
}

impl Apic {
    /// Chooses how the machine's controllers are reached, silences what would
    /// interfere, and brings the boot processor's own controller up.
    ///
    /// # Errors
    ///
    /// [`ApicError::NoApic`] on a processor with no local controller;
    /// [`ApicError::NoX2Apic`] if the machine has a processor whose identifier
    /// needs x2APIC and this processor does not implement it;
    /// [`ApicError::Paging`] if the register page cannot be mapped; or whatever
    /// bringing this processor's controller up reported.
    pub fn install(space: &mut AddressSpace, madt: &Madt) -> Result<Self, ApicError> {
        if !processor::features().contains(Features::APIC) {
            return Err(ApicError::NoApic);
        }
        let wide = cpu::roster()?.needs_x2apic();
        let mode = match (wide, processor::features().contains(Features::X2APIC)) {
            (true, false) => return Err(ApicError::NoX2Apic),
            (_, true) => Mode::X2Apic,
            (false, false) => Mode::XApic,
        };

        // Before anything is unmasked. The legacy controllers deliver onto
        // vectors 8 to 15 at reset, which are exceptions, and vector 8 is the
        // one nothing may claim.
        let masked_8259 = madt.pic_8259();
        if masked_8259 {
            pic::mask();
        }

        let access = match mode {
            Mode::X2Apic => Access::Msr,
            Mode::XApic => {
                // SAFETY: the address is the one firmware's own table gives for
                // the register page, and a controller's registers are not
                // memory — nothing else in this image maps them, and no
                // reference into the mapping outlives this crate.
                let mapping = unsafe {
                    space.map_physical(
                        madt.local_apic(),
                        REGISTER_PAGE,
                        Protection::ReadWrite,
                        CacheType::Uncached,
                    )
                }?;
                Access::Mapped(mapping.addr())
            }
        };
        register::establish(access);
        CONFIGURATION.call_once(|| Configuration {
            mode,
            local_nmis: madt.local_nmis().to_vec(),
        });

        // Registered before any controller is switched on, so that neither
        // vector can arrive to find nothing claiming it.
        descriptors::register(SPURIOUS, spurious)?;
        descriptors::register(ERROR, errors)?;
        LocalApic::enable()?;
        Ok(Self { mode, masked_8259 })
    }

    /// Which interface the machine's controllers present.
    #[must_use]
    pub const fn mode(&self) -> Mode {
        self.mode
    }

    /// Logs what was chosen and what was silenced.
    pub fn describe(&self, who: &str) {
        let mode = match self.mode {
            Mode::XApic => "xapic",
            Mode::X2Apic => "x2apic",
        };
        match (LocalApic.id(), LocalApic.version()) {
            (Ok(id), Ok(version)) => info!(
                "{who}: apic {mode}, boot processor is {id}, version {:#04x}, {} lvt entries",
                version & VERSION_MASK,
                ((version >> LVT_COUNT_SHIFT) & VERSION_MASK) + 1,
            ),
            _ => info!("{who}: apic {mode}, controller not readable"),
        }
        if self.masked_8259 {
            info!("{who}: apic masked both legacy 8259 controllers");
        }
        info!("{who}: apic spurious interrupts on {SPURIOUS}, controller errors on {ERROR}");
    }
}

/// Bytes the memory-mapped register page occupies.
const REGISTER_PAGE: u64 = 4096;

/// The version register's fields are one byte each.
pub(crate) const VERSION_MASK: u32 = 0xFF;

/// Bits the version register's local-vector-table count is shifted by. It holds
/// one less than the number of entries.
pub(crate) const LVT_COUNT_SHIFT: u32 = 16;

/// A handle to the controller of whichever processor is holding it.
///
/// Zero-sized on purpose. There is nothing to carry: every register is reached
/// the same way on every processor and answers about the processor doing the
/// reaching, so a handle that named one would be a handle that could be wrong.
#[derive(Clone, Copy, Debug)]
pub struct LocalApic;

impl LocalApic {
    /// Brings this processor's controller up.
    ///
    /// Called by the boot processor from [`Apic::install`], and by every other
    /// processor for itself once it is running: the mode is a one-way
    /// transition each processor has to make, and the local vector table is
    /// per processor.
    ///
    /// # Errors
    ///
    /// [`ApicError::NotInstalled`] before [`Apic::install`],
    /// [`ApicError::NoX2Apic`] or [`ApicError::ModeRegression`] if this
    /// processor cannot reach the mode the machine chose, or
    /// [`ApicError::Cpu`] if the roster does not describe this processor.
    pub fn enable() -> Result<Self, ApicError> {
        let configuration = CONFIGURATION.get().ok_or(ApicError::NotInstalled)?;
        base::enter(configuration.mode)?;
        let access = register::access()?;

        // The priority register is a filter this hypervisor has no use for: an
        // interrupt it declines is one that stays pending rather than one that
        // goes away. Zero accepts everything.
        //
        // SAFETY: the spurious vector register's low byte is a vector with a
        // gate like any other, and setting the enable bit is what makes the
        // controller deliver at all — done after both of this crate's vectors
        // have handlers.
        unsafe {
            access.write(Register::TASK_PRIORITY, 0);
            access.write(
                Register::SPURIOUS,
                u32::from(SPURIOUS.number()) | SOFTWARE_ENABLE,
            );
            access.write(
                Register::LVT_ERROR,
                Entry::new(LvtDelivery::Fixed(ERROR)).bits(),
            );
        }

        // Every source the controller has, not only the ones this crate uses.
        // The boot processor inherits a timer firmware armed for its own
        // purposes, and an entry left as firmware set it delivers on firmware's
        // chosen vector the moment interrupts are unmasked — an interrupt from
        // something that no longer exists.
        //
        // How many the controller has is its own to report, and the three below
        // are the first three it counts. A register past that count is one the
        // controller does not implement, and writing it is undefined rather than
        // ignored.
        let entries = (Self.version()? >> LVT_COUNT_SHIFT & VERSION_MASK) + 1;
        // SAFETY: a masked entry with a valid vector delivers nothing, which is
        // what every one of these is, and a zero count stops a timer that
        // firmware left counting rather than merely silencing it.
        unsafe {
            access.write(Register::TIMER_INITIAL_COUNT, 0);
            for (index, register) in [
                Register::LVT_TIMER,
                Register::LVT_THERMAL,
                Register::LVT_PERFORMANCE,
            ]
            .into_iter()
            .enumerate()
            {
                if u32::try_from(index).is_ok_and(|index| index < entries) {
                    access.write(register, Entry::masked().bits());
                }
            }
        }

        let this = Self.id()?;
        let uid = cpu::roster()?
            .find(this)
            .ok_or(CpuError::Unknown { apic_id: this })?
            .uid();
        for (input, register) in [(0, Register::LVT_LINT0), (1, Register::LVT_LINT1)] {
            // SAFETY: every entry is either masked or the non-maskable delivery
            // firmware itself described, and both are values the register
            // defines.
            unsafe { access.write(register, configuration.wiring(uid, input).bits()) };
        }

        // Last: the register latches whatever the controller noticed while it
        // was being set up, and none of that describes a running machine.
        Self::clear_errors();
        Ok(Self)
    }

    /// This processor's identifier.
    ///
    /// The older interface keeps it in the top eight bits of the register; the
    /// newer one uses the whole of it.
    ///
    /// # Errors
    ///
    /// [`ApicError::NotInstalled`] if the controller is not up.
    pub fn id(self) -> Result<ApicId, ApicError> {
        let access = register::access()?;
        let raw = access.read(Register::ID);
        Ok(ApicId::new(match access {
            Access::Msr => raw,
            Access::Mapped(_) => raw >> XAPIC_ID_SHIFT,
        }))
    }

    /// The controller's version register: its version in the low byte, and one
    /// less than its number of local vector table entries in the third.
    ///
    /// # Errors
    ///
    /// [`ApicError::NotInstalled`] if the controller is not up.
    pub fn version(self) -> Result<u32, ApicError> {
        register::access().map(|access| access.read(Register::VERSION))
    }

    /// This processor's timer.
    #[must_use]
    pub const fn timer(self) -> Timer {
        Timer::new(self)
    }

    /// Acknowledges the interrupt this processor is currently servicing.
    ///
    /// Owed for everything the controller delivered, and for nothing else: a
    /// spurious interrupt was never accepted, so acknowledging one would retire
    /// whatever really is in service instead.
    ///
    /// # Errors
    ///
    /// [`ApicError::NotInstalled`] if the controller is not up.
    pub fn end_of_interrupt(self) -> Result<(), ApicError> {
        let access = register::access()?;
        // SAFETY: the register takes zero and nothing else, and writing it is
        // what the architecture defines as acknowledging.
        unsafe { access.write(Register::END_OF_INTERRUPT, 0) };
        Ok(())
    }

    /// Sends `command`.
    ///
    /// # Errors
    ///
    /// [`ApicError::IdTooWide`] if the target cannot be named in the interface
    /// in use, [`ApicError::CommandStuck`] if a previous command has still not
    /// left, or [`ApicError::NotInstalled`] if the controller is not up.
    pub fn send(self, command: Command) -> Result<(), ApicError> {
        let configuration = CONFIGURATION.get().ok_or(ApicError::NotInstalled)?;
        let bits = command.bits(configuration.mode)?;
        // SAFETY: `bits` came from a `Command`, which can only describe
        // combinations the architecture defines: its reserved fields are zero
        // and its destination was checked against the width of the interface.
        unsafe { register::access()?.send(bits) }
    }

    /// What the controller has noticed going wrong, clearing it as it reads.
    ///
    /// The register latches, and the architecture requires a write before a
    /// read to make it report what has happened since it was last asked.
    fn take_errors() -> u32 {
        let Ok(access) = register::access() else {
            return 0;
        };
        // SAFETY: the register takes zero and nothing else; the write is what
        // makes the read report anything, and the second is what clears what was
        // just read.
        unsafe {
            access.write(Register::ERROR_STATUS, 0);
            let errors = access.read(Register::ERROR_STATUS);
            access.write(Register::ERROR_STATUS, 0);
            errors
        }
    }

    /// Throws away whatever the controller latched during bring-up.
    fn clear_errors() {
        let _ = Self::take_errors();
    }
}

/// Bits the older interface's identifier is shifted by: the top eight of the
/// register.
const XAPIC_ID_SHIFT: u32 = 24;

/// The bit that makes the controller deliver anything at all.
const SOFTWARE_ENABLE: u32 = 1 << 8;

/// This processor's controller, if the machine's have been installed.
///
/// # Errors
///
/// [`ApicError::NotInstalled`] before [`Apic::install`].
pub fn local() -> Result<LocalApic, ApicError> {
    register::access().map(|_| LocalApic)
}

/// Acknowledges the interrupt this processor is servicing.
///
/// A free function as well as a method, because every subsystem that consumes
/// an interrupt owes one and none of them should have to hold a handle to
/// something with no state in it.
///
/// # Errors
///
/// [`ApicError::NotInstalled`] if the controller is not up.
pub fn end_of_interrupt() -> Result<(), ApicError> {
    LocalApic.end_of_interrupt()
}

/// What is true of every processor's controller, decided once.
#[derive(Debug)]
struct Configuration {
    mode: Mode,
    local_nmis: Vec<acpi::LocalNmi>,
}

impl Configuration {
    /// What to program into one of the two interrupt pins on the processor with
    /// this ACPI identifier.
    ///
    /// Masked unless firmware said the pin is wired as a non-maskable
    /// interrupt. A pin left as firmware had it is a pin that can deliver to a
    /// vector chosen by something no longer running, and on a modern machine
    /// both pins are either unconnected or exactly this.
    fn wiring(&self, uid: u32, input: u8) -> Entry {
        self.local_nmis
            .iter()
            .find(|nmi| {
                nmi.input() == input
                    && match nmi.target() {
                        NmiTarget::All => true,
                        NmiTarget::Processor(target) => target == uid,
                    }
            })
            .map_or_else(Entry::masked, |nmi| {
                Entry::new(LvtDelivery::NonMaskable).wired(
                    match nmi.polarity() {
                        acpi::Polarity::ActiveLow => Polarity::ActiveLow,
                        _ => Polarity::ActiveHigh,
                    },
                    match nmi.trigger() {
                        acpi::Trigger::Level => Trigger::Level,
                        _ => Trigger::Edge,
                    },
                )
            })
    }
}

/// Decided by the boot processor, read by every processor bringing its own
/// controller up.
static CONFIGURATION: Once<Configuration> = Once::new();

/// What arrives when the controller withdraws an interrupt it had already begun
/// to deliver.
///
/// Always ours, whatever else the machine is doing: this vector is in the
/// controller's own spurious vector register and nothing else can be told to
/// deliver on it. It is deliberately not acknowledged, because it was never
/// accepted — an acknowledgement here would retire whatever really is in
/// service.
fn spurious(_: &Interrupt) -> Disposition {
    warn!("apic: spurious interrupt");
    Disposition::Consumed
}

/// What arrives when the controller notices something wrong with itself.
///
/// Reported rather than acted on. Every condition it latches is either a
/// message this processor sent that could not be delivered or one it could not
/// accept, and neither is recoverable from here — but a machine dropping
/// interprocessor interrupts silently is exactly the kind of fault that is
/// impossible to find afterwards.
fn errors(_: &Interrupt) -> Disposition {
    warn!(
        "apic: controller reported errors {:#010b}",
        LocalApic::take_errors()
    );
    let _ = LocalApic.end_of_interrupt();
    Disposition::Consumed
}

/// Why the interrupt controllers could not be set up or driven.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum ApicError {
    /// The processor has no local interrupt controller, so nothing here works
    /// and nothing can reach another processor.
    #[error("the processor has no local apic")]
    NoApic,
    /// The machine needs x2APIC — it has a processor whose identifier does not
    /// fit eight bits — and the processor does not implement it.
    #[error("the machine needs x2apic and the processor does not have it")]
    NoX2Apic,
    /// The older interface was asked for on a controller already in x2APIC
    /// mode, which the architecture gives no way back from.
    #[error("a controller in x2apic mode cannot go back to xapic")]
    ModeRegression,
    /// Nothing has set the machine's controllers up yet.
    #[error("the interrupt controllers have not been installed")]
    NotInstalled,
    /// The identifier does not fit the interface in use, so an interrupt
    /// addressed to it would reach a different processor.
    #[error("{apic_id} does not fit xapic's eight-bit destination field")]
    IdTooWide {
        /// The identifier that could not be named.
        apic_id: ApicId,
    },
    /// A command the controller was given has still not left it, so sending
    /// another would overwrite something the machine is still waiting on.
    #[error("the interrupt command register is still busy with a previous command")]
    CommandStuck,
    /// A counting timer was given a count of zero, which the architecture reads
    /// as stopped rather than as immediately.
    #[error("a timer count of zero stops the timer rather than firing it")]
    ZeroCount,
    /// A deadline was given to a counting mode, or a count to the deadline
    /// mode.
    #[error("that timer mode does not take that kind of deadline")]
    WrongTimerMode,
    /// The processor does not implement the timestamp counter deadline.
    #[error("the processor does not have the tsc deadline timer")]
    NoTscDeadline,
    /// There is no timebase to measure against or to wait on.
    #[error("no timebase is installed")]
    Clock,
    /// The timer did not move, or moved so far that the measurement means
    /// nothing.
    #[error("the timer's rate could not be measured")]
    Calibration,
    /// The trampoline has grown into the parameters it reads.
    #[error("the trampoline is {bytes} bytes and has only {room} before its parameters")]
    TrampolineTooLarge {
        /// Bytes the blob occupies.
        bytes: usize,
        /// Bytes there were for it.
        room: usize,
    },
    /// The trampoline page is not somewhere a startup command can name.
    #[error("physical {phys:#x} is not a frame-aligned page below 1 MiB")]
    TrampolineUnreachable {
        /// The address that was offered.
        phys: u64,
    },
    /// The page tables are above four gigabytes, where the 32-bit instruction
    /// that loads them cannot name them.
    #[error("page tables at physical {phys:#x} cannot be loaded by a 32-bit cr3 write")]
    PageTableRootTooHigh {
        /// Where the tables are.
        phys: u64,
    },
    /// A processor answered a startup command with an identifier that is not
    /// the one the command was addressed to.
    #[error("{expected} was started and {found} answered")]
    WrongProcessor {
        /// Who was asked.
        expected: ApicId,
        /// Who arrived.
        found: ApicId,
    },
    /// The address space refused something.
    #[error(transparent)]
    Paging(#[from] PagingError),
    /// The descriptor tables refused something.
    #[error(transparent)]
    Descriptors(#[from] DescriptorError),
    /// The processor roster refused something.
    #[error(transparent)]
    Cpu(#[from] CpuError),
}
