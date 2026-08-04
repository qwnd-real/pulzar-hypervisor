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
//! memory-mapped offset, so neither interface can drift away from the other,
//! and the two registers only one of them has are a type only that one can
//! reach.
//!
//! The choice is not a preference. Pulzar hands the machine on to firmware and
//! goes on emulating a controller for it, and the emulated one's logical
//! destination register has to agree with the real one — because the I/O
//! controllers are passed through, and hardware matches a passed-through
//! interrupt against the *real* register. In x2APIC that register is read-only
//! and derived from the identifier, so the only way to make the two agree is
//! for the real controller to present the same interface the emulated one does.
//! So the mode is whichever one firmware was in, and each processor follows its
//! own guest from there. The trip into x2APIC is one-way and is made by each
//! processor for itself, so a handle to a controller carries how that processor
//! reaches it.
//!
//! # What the controller has
//!
//! Not the same thing on every machine. A controller reports how many local
//! vector table entries it has, and it has exactly the first that many of the
//! architecture's list — so the thermal and performance entries are simply
//! absent on a controller that counts four, and writing one that is absent is
//! undefined through the register page and a fault through the model-specific
//! registers. Nothing here touches an entry the controller has not claimed.
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
//! # Before any of it
//!
//! [`capture`] reads the controllers as firmware left them, and is the one
//! thing here that writes nothing at all. What it reads stops being readable
//! the moment [`Apic::install`] runs, and a hypervisor that means to hand the
//! machine back needs it.
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
use core::{
    fmt::{self, Display, Formatter},
    sync::atomic::{AtomicU64, Ordering},
};

use acpi::{Madt, NmiTarget};
use cpu::{ApicId, CpuError};
use descriptors::{DescriptorError, Disposition, Interrupt, Vector};
use log::{info, warn};
use paging::{AddressSpace, CacheType, PagingError, Protection};
use processor::Features;
use spin::Once;
use thiserror::Error;
use x86_64::PhysAddr;

use crate::register::{Access, MappedRegister, Page, Register};
pub use crate::{
    capture::{Controller, FirmwareState, LVT_ENTRIES, LocalState, VECTOR_WORDS, capture},
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

/// Both of this crate's own vectors have to be ones the platform may assign.
///
/// An exception vector cannot be delivered by a controller and cannot be
/// claimed from [`descriptors`], so a controller told to use one would be a
/// controller whose two most important interrupts never arrive.
const _: () = assert!(
    !SPURIOUS.is_exception() && !ERROR.is_exception(),
    "the apic's own vectors must be ones the platform may assign"
);

/// Whether a controller may be told to deliver on this vector.
///
/// Below the first the platform may assign are the architecture's own
/// exceptions, and a controller given one of those delivers nothing and latches
/// an illegal-vector error instead. From the caller's side that is an interrupt
/// which silently never arrives, so it is refused where it is asked for rather
/// than reported from the error handler afterwards.
pub(crate) const fn deliverable(vector: Vector) -> bool {
    !vector.is_exception()
}

/// Bytes in the page the architecture measures both the register page and a
/// startup command's vector in.
///
/// Its own constant rather than the paging crate's frame size: these are
/// architectural 4 KiB pages, and they would still be 4 KiB if this hypervisor
/// mapped memory in some other size.
pub(crate) const PAGE: u64 = 4096;

/// Which of the two interfaces a controller presents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// The page of memory-mapped registers, with eight-bit identifiers.
    XApic,
    /// Model-specific registers, with 32-bit identifiers.
    X2Apic,
}

impl Mode {
    /// What to call this mode in a log line.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::XApic => "xapic",
            Self::X2Apic => "x2apic",
        }
    }
}

impl Display for Mode {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// The machine's local interrupt controllers.
///
/// Produced once, by the boot processor. What it holds is what is true of all
/// of them; a particular processor's controller is a [`LocalApic`], which is a
/// handle for whichever processor asks.
#[derive(Clone, Copy, Debug)]
pub struct Apic {
    entered: Mode,
    masked_8259: bool,
}

impl Apic {
    /// Maps the register page, silences what would interfere, and brings the
    /// boot processor's own controller up in `mode`.
    ///
    /// The mode is the caller's rather than this crate's, and what it should be
    /// is whichever one firmware was already in — because the controller is
    /// handed back to firmware as an emulated one, and the emulated
    /// controller's logical destination register can only agree with the
    /// real one when both present the same interface.
    ///
    /// Everything that can be refused is refused before anything is recorded,
    /// so a machine this declines to run on is one nothing has been done to.
    /// After that point the register page is mapped for good: every processor
    /// reaches its own controller through that one mapping for as long as the
    /// image runs, and there is no later moment at which giving it back would
    /// be anything but a mistake.
    ///
    /// # Errors
    ///
    /// [`ApicError::AlreadyInstalled`] for a second call, which would leave the
    /// machine with two answers to a question that has one;
    /// [`ApicError::NoApic`] on a processor with no local controller;
    /// [`ApicError::NoRegisterPage`] if the processor reports its register page
    /// at address zero; [`ApicError::Descriptors`] if either of this crate's
    /// vectors is already claimed; [`ApicError::Paging`] if the register page
    /// cannot be mapped; or whatever bringing this processor's controller up
    /// reported.
    pub fn install(space: &mut AddressSpace, madt: &Madt, mode: Mode) -> Result<Self, ApicError> {
        if CONFIGURATION.is_completed() {
            return Err(ApicError::AlreadyInstalled);
        }
        if !processor::features().contains(Features::APIC) {
            return Err(ApicError::NoApic);
        }
        // Said rather than refused. A machine with a processor whose identifier
        // needs x2APIC, whose firmware left the controllers in xAPIC, is one
        // whose own firmware could not address those processors either — and the
        // ones that can be addressed are worth running. Each unaddressable
        // processor is refused individually, where it is named: `Command::bits`
        // answers `IdTooWide` and `smp::start_each` logs and skips it.
        if mode == Mode::XApic && cpu::roster()?.needs_x2apic() {
            warn!(
                "apic: firmware left the controllers in xapic and the machine has a processor \
                 whose identifier does not fit its destination field; those processors cannot be \
                 started"
            );
        }

        // Claimed before any controller is switched on, so that neither vector
        // can arrive to find nothing claiming it — and before anything at all is
        // established, so that a refusal here leaves the machine as it was.
        descriptors::register(SPURIOUS, spurious)?;
        descriptors::register(ERROR, errors)?;

        // Also before anything is unmasked. The legacy controllers deliver onto
        // vectors 8 to 15 at reset, which are exceptions, and vector 8 is the
        // one nothing may claim.
        let masked_8259 = madt.pic_8259();
        if masked_8259 {
            pic::mask();
        }

        // Mapped whichever mode was asked for, because which interface a
        // processor presents is that processor's own and changes while the
        // machine runs: a guest that enters x2APIC takes its processor with it,
        // and every processor that has not followed reaches its controller
        // through this page. There is no later moment at which it could be
        // mapped, either — the address space stops being a value once bring-up
        // is past it.
        let phys = register_page(madt)?;
        // SAFETY: the address is the processor's own answer for where its
        // register page is, and a controller's registers are not memory —
        // nothing else in this image maps them, and no reference into the
        // mapping outlives this crate.
        let mapping =
            unsafe { space.map_physical(phys, PAGE, Protection::ReadWrite, CacheType::Uncached) }?;
        // SAFETY: a whole register page, uncached as the architecture requires,
        // at an address nothing releases.
        register::establish_page(unsafe { Page::new(mapping.addr()) })?;
        CONFIGURATION.call_once(|| Configuration {
            local_nmis: madt.local_nmis().to_vec(),
        });
        let entered = LocalApic::enable(mode)?.mode();
        Ok(Self {
            entered,
            masked_8259,
        })
    }

    /// Which interface the boot processor's controller was brought up in.
    ///
    /// The machine's answer only for as long as no guest has moved a processor
    /// of its own; [`LocalApic::mode`] is what answers for a processor.
    #[must_use]
    pub const fn mode(self) -> Mode {
        self.entered
    }

    /// Logs what was entered, what was silenced, and what has gone wrong since.
    pub fn describe(self, who: &str) {
        let mode = self.entered.name();
        match Self::identity() {
            Ok((id, version)) => info!(
                "{who}: apic {mode}, boot processor is {id}, version {:#04x}, {} lvt entries",
                register::version_number(version),
                register::lvt_entries(version),
            ),
            Err(error) => info!("{who}: apic {mode}, controller not readable: {error}"),
        }
        if self.masked_8259 {
            info!("{who}: apic masked both legacy 8259 controllers");
        }
        info!(
            "{who}: apic spurious interrupts on {SPURIOUS} ({} so far), controller errors on {ERROR} ({} so far)",
            SPURIOUS_ARRIVALS.load(Ordering::Relaxed),
            CONTROLLER_ERRORS.load(Ordering::Relaxed),
        );
    }

    /// What this processor's controller calls itself, and what it says it is.
    fn identity() -> Result<(ApicId, u32), ApicError> {
        let local = local()?;
        Ok((local.id(), local.version()))
    }
}

/// Where the memory-mapped register page is.
///
/// The processor's own register is the answer, because it is the one the
/// processor decodes; firmware's table describes the same thing and is only a
/// description. They agree on any machine that was put together properly, and
/// where they do not, saying so is worth more than quietly preferring either.
///
/// # Errors
///
/// [`ApicError::NoApic`] if there is no controller to ask, or
/// [`ApicError::NoRegisterPage`] if the processor reports the page at address
/// zero, which is not somewhere a controller can be and is what a register
/// nothing ever filled in reads as.
fn register_page(madt: &Madt) -> Result<PhysAddr, ApicError> {
    let page = base::page().ok_or(ApicError::NoApic)?;
    if page.as_u64() == 0 {
        return Err(ApicError::NoRegisterPage);
    }
    let described = madt.local_apic();
    if page != described {
        warn!(
            "apic: the processor puts its register page at {page:#x} and firmware's tables say \
             {described:#x}; the processor decides"
        );
    }
    Ok(page)
}

/// A handle to the controller of whichever processor is holding it.
///
/// Carries how that processor reaches its controller, and nothing else: every
/// register answers about the processor doing the reaching, so a handle that
/// named one would be a handle that could be wrong. What the access *is* cannot
/// be a property of the machine, because a guest may take its own processor
/// into x2APIC and leave the others where they were — so it is derived, per
/// handle, from the one register that says which interface this controller
/// presents.
///
/// It is also proof: it cannot be made outside this crate, and inside it is
/// only made for a processor whose controller is switched on.
#[derive(Clone, Copy, Debug)]
pub struct LocalApic(Access);

impl LocalApic {
    /// Brings this processor's controller up in `mode`.
    ///
    /// Called by the boot processor from [`Apic::install`], and by every other
    /// processor for itself once it is running: entering a mode is something
    /// each processor does to its own controller, and the local vector table is
    /// per processor.
    ///
    /// The order inside is the architecture's rather than a preference. Every
    /// source the controller has is masked first, because a controller that
    /// firmware left running is a controller with a timer armed for something
    /// that no longer exists. The enable bit follows. Only then are the entries
    /// that are meant to deliver something written, because a software-disabled
    /// controller holds every entry masked and ignores an attempt to clear the
    /// bit.
    ///
    /// # Errors
    ///
    /// [`ApicError::NotInstalled`] before [`Apic::install`],
    /// [`ApicError::NoX2Apic`], [`ApicError::ModeRegression`] or
    /// [`ApicError::ModeNotEntered`] if this processor cannot reach `mode`, or
    /// [`ApicError::Cpu`] if the roster does not describe this processor.
    pub fn enable(mode: Mode) -> Result<Self, ApicError> {
        let configuration = CONFIGURATION.get().ok_or(ApicError::NotInstalled)?;
        base::enter(mode)?;
        let this = local()?;
        let access = this.0;
        let entries = this.entries();

        // SAFETY: a masked entry with a valid vector delivers nothing, which is
        // what every one of these is; a zero count stops a timer that firmware
        // left counting rather than merely silencing it; and every register
        // written is one the controller says it has.
        unsafe {
            for register in register::lvt_present(entries) {
                access.write(register, Entry::masked().bits());
            }
            access.write(Register::TIMER_INITIAL_COUNT, 0);
        }

        // The priority register is a filter this hypervisor has no use for: an
        // interrupt it declines is one that stays pending rather than one that
        // goes away. Zero accepts everything.
        //
        // SAFETY: the spurious vector register's low byte is a vector with a
        // gate like any other, and setting the enable bit is what makes the
        // controller deliver at all — done after both of this crate's vectors
        // have handlers and after every source of its own is masked.
        unsafe {
            access.write(Register::TASK_PRIORITY, 0);
            access.write(
                Register::SPURIOUS,
                u32::from(SPURIOUS.number()) | SOFTWARE_ENABLE,
            );
        }

        let id = this.id();
        let uid = cpu::roster()?
            .find(id)
            .ok_or(CpuError::Unknown { apic_id: id })?
            .uid();
        for (register, entry) in [
            (Register::LVT_LINT0, configuration.wiring(uid, 0)),
            (Register::LVT_LINT1, configuration.wiring(uid, 1)),
            (Register::LVT_ERROR, Entry::new(LvtDelivery::Fixed(ERROR))),
        ] {
            if register::has_lvt(register, entries) {
                // SAFETY: each of these is either the non-maskable delivery
                // firmware itself described, a masked pin, or this crate's own
                // error vector, and the register is one the controller has.
                unsafe { access.write(register, entry.bits()) };
            }
        }

        Self::retire_inherited(access);
        // Last: the register latches whatever the controller noticed while it
        // was being set up, and none of that describes a running machine.
        this.clear_errors();
        Ok(this)
    }

    /// Which interface this processor's controller presents.
    #[must_use]
    pub const fn mode(self) -> Mode {
        match self.0 {
            Access::Mapped(_) => Mode::XApic,
            Access::Msr => Mode::X2Apic,
        }
    }

    /// How this processor reaches its controller's registers.
    ///
    /// For the modules of this crate that drive a register file of their own —
    /// the timer — rather than for anything outside it.
    pub(crate) const fn access(self) -> Access {
        self.0
    }

    /// Takes this processor's controller into x2APIC, and answers with a handle
    /// that reaches it there.
    ///
    /// The whole of what a hypervisor needs this for is agreement. Its guest's
    /// emulated controller may enter x2APIC, and from that moment the guest
    /// addresses passed-through interrupts by an identifier the architecture
    /// derives rather than one software writes — so the real controller has to
    /// derive the same one, which it does only in the same mode.
    ///
    /// Nothing is reprogrammed, because nothing needs to be: the architecture
    /// carries the register file across this transition, so the spurious
    /// vector, the local vector table and the priorities are all still what
    /// they were.
    ///
    /// Returning a new handle rather than mutating this one is what stops a
    /// handle taken before the transition being used after it, which would
    /// reach registers that are no longer there.
    ///
    /// # Errors
    ///
    /// [`ApicError::NoX2Apic`] if this processor does not implement it, or
    /// [`ApicError::ModeNotEntered`] if it took the write and did not change.
    pub fn enter_x2apic(self) -> Result<Self, ApicError> {
        base::enter(Mode::X2Apic)?;
        local()
    }

    /// This processor's identifier.
    ///
    /// The older interface keeps it in the top eight bits of the register; the
    /// newer one uses the whole of it.
    #[must_use]
    pub fn id(self) -> ApicId {
        let raw = self.0.read(Register::ID);
        ApicId::new(match self.0 {
            Access::Msr => raw,
            Access::Mapped(_) => raw >> XAPIC_ID_SHIFT,
        })
    }

    /// The controller's version register: its version in the low byte, and one
    /// less than its number of local vector table entries in the third.
    #[must_use]
    pub fn version(self) -> u32 {
        self.0.read(Register::VERSION)
    }

    /// This processor's timer.
    #[must_use]
    pub const fn timer(self) -> Timer {
        Timer::new(self)
    }

    /// Whether the interrupt this processor accepted on `vector` arrived level
    /// triggered.
    ///
    /// The controller records this itself, one bit per vector, as it accepts
    /// each interrupt: set for level triggered and clear for edge. That makes
    /// it the one authority on the question that needs nothing else to be
    /// modelled — a hypervisor that passes the I/O controllers through does not
    /// know how a line was configured, but the local controller that took the
    /// interrupt does, and it was told by the same hardware that sent it.
    ///
    /// The distinction decides what is owed. An edge-triggered interrupt is
    /// finished with once it has been taken; a level-triggered one is asserted
    /// until whoever raised it is dealt with, so acknowledging it before that
    /// happens delivers it again immediately.
    #[must_use]
    pub fn arrived_level(self, vector: Vector) -> bool {
        let (slot, bit) = trigger_place(vector);
        self.0.read(Register::TRIGGER_MODE.offset_by(slot)) & bit != 0
    }

    /// Whether this processor has accepted `vector` and not yet acknowledged
    /// it.
    ///
    /// The in-service bank is the controller's own record of what it is
    /// holding, and it is the only authority on the question. Two callers
    /// need it and they need it for opposite reasons: one that has withheld
    /// an acknowledgement must know whether the vector it owes is still the
    /// highest one held, because the acknowledgement register carries no
    /// vector and retires whatever is; and a handler on a vector the
    /// controller also uses for something of its own can tell a genuinely
    /// accepted interrupt from a withdrawn one, since a withdrawn one is
    /// never accepted and sets no bit here.
    #[must_use]
    pub fn in_service(self, vector: Vector) -> bool {
        let (slot, bit) = trigger_place(vector);
        self.0.read(Register::IN_SERVICE.offset_by(slot)) & bit != 0
    }

    /// The highest-priority vector this processor is holding in service, if
    /// any.
    ///
    /// Which is the one an acknowledgement would retire: the register takes no
    /// vector, and the controller answers it by retiring the highest bit it
    /// holds. Anything withholding an acknowledgement has to compare against
    /// this before issuing one, or it retires an interrupt belonging to
    /// somebody else.
    ///
    /// Highest-numbered is highest-priority, so one descending scan answers it.
    #[must_use]
    pub fn in_service_top(self) -> Option<Vector> {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "eight slots of thirty-two bits is the whole of a vector's range, so neither the slot nor the bit can leave a byte"
        )]
        (0..register::VECTOR_SLOTS).rev().find_map(|slot| {
            let slot = slot as u32;
            let word = self.0.read(Register::IN_SERVICE.offset_by(slot));
            (word != 0).then(|| {
                let bit = u32::BITS - 1 - word.leading_zeros();
                Vector::new((slot * u32::BITS + bit) as u8)
            })
        })
    }

    /// One of the controller's own sources, as the controller currently holds
    /// it.
    ///
    /// # Errors
    ///
    /// [`ApicError::NoSuchLvt`] if this controller does not have the entry.
    pub fn source(self, source: Source) -> Result<Entry, ApicError> {
        let register = source.register();
        if !register::has_lvt(register, self.entries()) {
            return Err(ApicError::NoSuchLvt { which: source });
        }
        Ok(Entry::from_bits(self.0.read(register)))
    }

    /// How many local vector table entries this controller has.
    ///
    /// Between one and seven, and a controller with fewer than seven does not
    /// merely leave the rest unused: the registers are absent, and naming one
    /// is undefined through the page and a fault through the model-specific
    /// registers. Anything describing this controller to somebody else — a
    /// hypervisor handing a guest a local controller, above all — has to report
    /// this number rather than the largest the architecture allows.
    #[must_use]
    pub fn entries(self) -> u32 {
        register::lvt_entries(self.version())
    }

    /// Programs one of the controller's own sources.
    ///
    /// What a hypervisor passing the platform through needs in order to hand a
    /// guest the sources this processor really has: the guest's mask, trigger
    /// and polarity are its own, and only the vector is not — that is chosen by
    /// whoever calls this, so that an arrival can be told apart from every
    /// other interrupt on the machine.
    ///
    /// The error entry is deliberately absent from [`Source`]. The controller's
    /// errors are the host's to notice, and an entry handed to a guest would be
    /// one the host stopped hearing about.
    ///
    /// # Errors
    ///
    /// [`ApicError::NoSuchLvt`] if this controller does not have the entry —
    /// three of them are optional and a controller reports how many it has.
    pub fn program(self, source: Source, entry: Entry) -> Result<(), ApicError> {
        let register = source.register();
        if !register::has_lvt(register, self.entries()) {
            return Err(ApicError::NoSuchLvt { which: source });
        }
        // SAFETY: the value came from an `Entry`, which can only describe
        // combinations the architecture defines, and the register was just
        // established to be one this controller has.
        unsafe { self.0.write(register, entry.bits()) };
        Ok(())
    }

    /// Tells this controller how to match a logical destination, and which ones
    /// it answers to.
    ///
    /// The two registers are one operation, because a destination only means
    /// anything against the model that matches it: written apart there is a
    /// window in which the controller answers to a set of processors that is
    /// neither the old one nor the new one. The format is written first, which
    /// is the order that keeps that window from being a live one — a
    /// destination matched by the wrong model is a real interrupt going to
    /// the wrong processor, while a model with no destination yet matches
    /// nothing.
    ///
    /// The value belongs to whatever is deciding how interrupts are addressed
    /// on this machine. A hypervisor passing its I/O controllers through
    /// has to keep this equal to what its guest believes, because the guest
    /// programs those controllers directly and hardware matches the
    /// destination against *this* register — so a disagreement is an
    /// interrupt delivered to the wrong processor or to none.
    ///
    /// Answers whether the registers were written. In x2APIC they are not:
    /// hardware derives the identifier from this processor's own and there is
    /// only the one model, so neither register is writable and there is nothing
    /// to write. Nothing has gone wrong — a caller that needs the two to agree
    /// there agrees by deriving the same value, and
    /// [`LocalApic::logical_destination`] is what it checks against.
    #[must_use]
    pub fn set_logical_routing(self, format: u32, destination: u32) -> bool {
        let Access::Mapped(page) = self.0 else {
            return false;
        };
        // SAFETY: of the format register only the top nibble selects anything and
        // every encoding of it is one the architecture defines, with the reserved
        // remainder written as ones — its reset value and what the architecture
        // requires. Every value of the destination register's top byte is a legal
        // set of logical destinations, and its remainder is reserved and written
        // as zero.
        unsafe {
            page.write(
                MappedRegister::DESTINATION_FORMAT,
                format | !DESTINATION_FORMAT_MASK,
            );
            page.write(
                Register::LOGICAL_DESTINATION,
                destination & LOGICAL_DESTINATION_MASK,
            );
        }
        true
    }

    /// Which logical destinations this processor actually answers to.
    ///
    /// Read back rather than remembered, because in x2APIC nothing wrote it:
    /// hardware derives the value from the identifier and makes the register
    /// read-only. That makes this the one way to ask what passed-through
    /// interrupts are really matched against, in either mode.
    #[must_use]
    pub fn logical_destination(self) -> u32 {
        self.0.read(Register::LOGICAL_DESTINATION)
    }

    /// Acknowledges the interrupt this processor is currently servicing.
    ///
    /// Owed for everything the controller delivered, and for nothing else: a
    /// spurious interrupt was never accepted, so acknowledging one would retire
    /// whatever really is in service instead.
    pub fn end_of_interrupt(self) {
        // SAFETY: the register takes zero and nothing else, and writing it is
        // what the architecture defines as acknowledging.
        unsafe { self.0.write(Register::END_OF_INTERRUPT, 0) };
    }

    /// Sends `command`.
    ///
    /// # Errors
    ///
    /// [`ApicError::IdTooWide`] if the target cannot be named in the interface
    /// this processor presents, [`ApicError::IllegalVector`] if the command
    /// names a vector no controller may deliver, or
    /// [`ApicError::CommandStuck`] if a previous command has still not left.
    pub fn send(self, command: Command) -> Result<(), ApicError> {
        let bits = command.bits(self.mode())?;
        // SAFETY: `bits` came from a `Command`, which can only describe
        // combinations the architecture defines: its reserved fields are zero,
        // its vector is one a controller may deliver, and its destination was
        // checked against the width of the interface.
        unsafe { self.0.send(bits) }
    }

    /// Acknowledges whatever the controller was already holding in service, and
    /// says what it found.
    ///
    /// Masking a source stops the next interrupt, not one the processor has
    /// already accepted. The controller goes on refusing every interrupt of
    /// that priority or lower until it is told the one in service is finished,
    /// and firmware is under no obligation to have finished what it took — so a
    /// processor that inherited one and said nothing would quietly refuse a
    /// whole class of interrupts for the rest of its life.
    ///
    /// There can be at most one interrupt in service per vector, so the number
    /// of acknowledgements is bounded by the number of vectors and this cannot
    /// spin. What is merely requested cannot be retired at all — it has not
    /// been accepted yet — so it is counted and reported instead.
    fn retire_inherited(access: Access) {
        let in_service: u32 = register::bank(Register::IN_SERVICE)
            .map(|register| access.read(register).count_ones())
            .sum();
        for _ in 0..in_service {
            // SAFETY: the register takes zero and nothing else, and writing it
            // retires the highest-priority interrupt in service — which here is
            // something firmware accepted and this processor will never service.
            unsafe { access.write(Register::END_OF_INTERRUPT, 0) };
        }
        let requested: u32 = register::bank(Register::INTERRUPT_REQUEST)
            .map(|register| access.read(register).count_ones())
            .sum();
        if in_service != 0 || requested != 0 {
            info!(
                "apic: retired {in_service} interrupts firmware left in service, {requested} \
                 still requested"
            );
        }
    }

    /// What the controller has noticed going wrong, clearing it as it reads.
    ///
    /// The register latches, and the architecture requires a write before a
    /// read to make it report what has happened since it was last asked.
    fn take_errors(self) -> u32 {
        // SAFETY: the register takes zero and nothing else; the write is what
        // makes the read report anything, and the second is what clears what was
        // just read.
        unsafe {
            self.0.write(Register::ERROR_STATUS, 0);
            let errors = self.0.read(Register::ERROR_STATUS);
            self.0.write(Register::ERROR_STATUS, 0);
            errors
        }
    }

    /// Throws away whatever the controller latched during bring-up.
    fn clear_errors(self) {
        let _ = self.take_errors();
    }
}

/// One of the controller's own interrupt sources, as something outside this
/// crate may name it.
///
/// The error entry is missing on purpose: it belongs to the host, which is the
/// only thing that can act on a controller reporting its own errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// The controller's own timer.
    Timer,
    /// The first local interrupt pin.
    Lint0,
    /// The second local interrupt pin.
    Lint1,
    /// The thermal sensor.
    Thermal,
    /// The performance counters.
    Performance,
    /// Corrected machine-check errors.
    CorrectedMachineCheck,
}

impl core::fmt::Display for Source {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let name = match self {
            Self::Timer => "timer",
            Self::Lint0 => "lint0",
            Self::Lint1 => "lint1",
            Self::Thermal => "thermal",
            Self::Performance => "performance",
            Self::CorrectedMachineCheck => "corrected machine check",
        };
        formatter.write_str(name)
    }
}

impl Source {
    /// Every source that may be programmed from outside this crate.
    pub const ALL: [Self; 6] = [
        Self::Timer,
        Self::Lint0,
        Self::Lint1,
        Self::Thermal,
        Self::Performance,
        Self::CorrectedMachineCheck,
    ];

    /// The local vector table entry this source is programmed through.
    pub(crate) const fn register(self) -> Register {
        match self {
            Self::Timer => Register::LVT_TIMER,
            Self::Lint0 => Register::LVT_LINT0,
            Self::Lint1 => Register::LVT_LINT1,
            Self::Thermal => Register::LVT_THERMAL,
            Self::Performance => Register::LVT_PERFORMANCE,
            Self::CorrectedMachineCheck => Register::LVT_CORRECTED_MACHINE_CHECK,
        }
    }
}

/// Which of the trigger-mode registers a vector's bit is in, and which bit of
/// it.
const fn trigger_place(vector: Vector) -> (u32, u32) {
    let number = vector.number() as u32;
    (number / u32::BITS, 1 << (number % u32::BITS))
}

/// The part of the logical destination register that holds anything: the top
/// eight bits. The rest is reserved.
const LOGICAL_DESTINATION_MASK: u32 = 0xFF00_0000;

/// The part of the destination format register that selects anything: the top
/// four bits. The rest is reserved and reads as ones.
const DESTINATION_FORMAT_MASK: u32 = 0xF000_0000;

/// Bits the older interface's identifier is shifted by: the top eight of the
/// register.
const XAPIC_ID_SHIFT: u32 = 24;

/// The bit that makes the controller deliver anything at all.
const SOFTWARE_ENABLE: u32 = 1 << 8;

/// This processor's controller.
///
/// Answering means two things are true, and the second does not follow from the
/// first. The machine's controllers have been installed — and *this* processor
/// has switched its own on, which is something every processor does for itself
/// in [`LocalApic::enable`].
///
/// Which interface the handle reaches the controller through is decided here,
/// from the same register read that establishes the second of those. That is
/// what makes it this processor's answer rather than the machine's: a guest may
/// take its own processor into x2APIC and leave every other one where it was,
/// and this crate has to follow it there without disturbing the rest.
///
/// # Errors
///
/// [`ApicError::NoApic`] on a processor with no local controller,
/// [`ApicError::NotEnabled`] on one that has not yet switched its own on, or
/// [`ApicError::NotInstalled`] before [`Apic::install`] has mapped the register
/// page.
pub fn local() -> Result<LocalApic, ApicError> {
    let base = base::read().ok_or(ApicError::NoApic)?;
    if base & base::GLOBAL_ENABLE == 0 {
        return Err(ApicError::NotEnabled);
    }
    Access::of(base).map(LocalApic)
}

/// Acknowledges the interrupt this processor is servicing.
///
/// A free function as well as a method, because every subsystem that consumes
/// an interrupt owes one and none of them should have to hold a handle to say
/// so.
///
/// # Errors
///
/// As [`local`], which an interrupt being serviced already implies: it was
/// delivered by a controller, and a controller that delivers is one that is up.
pub fn end_of_interrupt() -> Result<(), ApicError> {
    local().map(LocalApic::end_of_interrupt)
}

/// What is true of every processor's controller, decided once.
#[derive(Debug)]
struct Configuration {
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

/// How many spurious interrupts have arrived, and how many times a controller
/// has reported an error.
///
/// Counted rather than logged one for one. Either can repeat as fast as a
/// controller can raise it, and a line of serial output costs milliseconds with
/// a lock on it that every processor's logging goes through — so a storm of
/// either would stop the machine more thoroughly than whatever caused it. The
/// first of each is worth a line, because it says something is happening that
/// was not before; the rest are worth a number, which [`Apic::describe`]
/// prints.
static SPURIOUS_ARRIVALS: AtomicU64 = AtomicU64::new(0);

/// How many times a controller has reported an error. See
/// [`SPURIOUS_ARRIVALS`].
static CONTROLLER_ERRORS: AtomicU64 = AtomicU64::new(0);

/// What arrives when the controller withdraws an interrupt it had already begun
/// to deliver.
///
/// Not necessarily ours, and that is the whole of the care taken here. This
/// vector is in the controller's own spurious vector register, but on a machine
/// whose I/O controllers are passed through it is also a number something else
/// may have been programmed to send — and the two are told apart by the one
/// thing that distinguishes them in hardware. A withdrawn interrupt was never
/// accepted, so it sets no in-service bit; anything that did set one was
/// genuinely accepted and belongs to whoever programmed it.
///
/// A withdrawn one is deliberately not acknowledged, because it was never
/// accepted — an acknowledgement here would retire whatever really is in
/// service. One that was accepted is passed on, and acknowledging it is part of
/// giving it to whoever it was for.
///
/// The two coinciding — a real withdrawal while an interrupt on the same vector
/// is in service — reads as accepted and is passed on. That misattributes one
/// arrival and cannot be told apart from the inside; nothing in the delivery
/// carries its origin.
fn spurious(interrupt: &Interrupt) -> Disposition {
    // A controller that cannot be reached cannot be asked, and the safe answer
    // is to pass the arrival on: consuming it would throw away an interrupt that
    // may have been genuinely accepted.
    let Ok(local) = local() else {
        return Disposition::Passed;
    };
    if local.in_service(interrupt.vector()) {
        return Disposition::Passed;
    }
    if SPURIOUS_ARRIVALS.fetch_add(1, Ordering::Relaxed) == 0 {
        warn!("apic: spurious interrupt, counting any others");
    }
    Disposition::Consumed
}

/// What arrives when the controller notices something wrong with itself.
///
/// Reported rather than acted on. Every condition it latches is either a
/// message this processor sent that could not be delivered or one it could not
/// accept, and neither is recoverable from here — but a machine dropping
/// interprocessor interrupts silently is exactly the kind of fault that is
/// impossible to find afterwards.
///
/// As with [`spurious`], the vector is not exclusively ours on a machine that
/// passes its I/O controllers through, and the discriminator is the error
/// status register: an error latches it before the interrupt it raises is
/// delivered, so an arrival finding nothing latched is one this controller did
/// not raise. The register is read on every arrival even where nothing is
/// logged, because reading is what makes it report the next fault rather than
/// the first one.
///
/// The same coincidence [`spurious`] describes applies here in the other
/// direction: a real error arriving alongside an interrupt something else sent
/// on this vector reads as ours, and that arrival is consumed.
fn errors(_: &Interrupt) -> Disposition {
    // As in `spurious`: a controller that cannot be asked what it latched has
    // said nothing that would make this arrival ours.
    let Ok(local) = local() else {
        return Disposition::Passed;
    };
    let errors = local.take_errors();
    if errors == 0 {
        return Disposition::Passed;
    }
    if CONTROLLER_ERRORS.fetch_add(1, Ordering::Relaxed) == 0 {
        warn!("apic: controller reported errors {errors:#010b}, counting any others");
    }
    local.end_of_interrupt();
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
    /// The controller took the write that switches it on and did not switch on,
    /// which is what a controller firmware disabled for good looks like.
    #[error("the controller did not enter the mode it was told to")]
    ModeNotEntered,
    /// Nothing has set the machine's controllers up yet.
    #[error("the interrupt controllers have not been installed")]
    NotInstalled,
    /// The machine's controllers have been set up already, and how they are
    /// reached is one answer for the whole machine.
    #[error("the interrupt controllers have already been installed")]
    AlreadyInstalled,
    /// This processor has not brought its own controller up, so it is not in
    /// the mode the machine's registers are named in.
    #[error("this processor's controller has not been enabled")]
    NotEnabled,
    /// This controller does not have that local vector table entry. Three of
    /// the seven are optional and a controller says how many it has; touching
    /// one it does not have is undefined through the page and a fault through
    /// the model-specific registers.
    #[error("this controller has no {which} entry")]
    NoSuchLvt {
        /// The source that was asked for.
        which: Source,
    },
    /// The register is one only the memory-mapped interface has, and the
    /// controller is reached through the model-specific registers, where the
    /// index it would occupy is reserved.
    #[error("that register exists only in xapic, and this controller is in x2apic mode")]
    NotMapped,
    /// The processor reports its register page at address zero, which is not
    /// somewhere a controller can be.
    #[error("the processor reports no address for its register page")]
    NoRegisterPage,
    /// The identifier does not fit the interface in use, so an interrupt
    /// addressed to it would reach a different processor.
    #[error("{apic_id} does not fit xapic's eight-bit destination field")]
    IdTooWide {
        /// The identifier that could not be named.
        apic_id: ApicId,
    },
    /// A vector below the first the platform may assign was given to a
    /// controller, which cannot deliver one and reports the attempt as an
    /// error.
    #[error("{vector} is not one a controller may deliver")]
    IllegalVector {
        /// The vector that was asked for.
        vector: Vector,
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
    /// A processor was sent every startup command the architecture prescribes
    /// and never executed the first instruction of the trampoline.
    #[error("{apic_id} did not answer a startup command")]
    NoStartupResponse {
        /// The processor that was asked.
        apic_id: ApicId,
    },
    /// A processor began executing and never reached the point of being one of
    /// the machine's, so something between the trampoline and attaching stopped
    /// it.
    #[error("{apic_id} started and did not finish coming up")]
    AttachTimeout {
        /// The processor that was asked.
        apic_id: ApicId,
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
