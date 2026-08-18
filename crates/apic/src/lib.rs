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
//! own guest from there.
//!
//! # Which interface is a question, never an answer
//!
//! The trip into x2APIC is one-way and is made by each processor for itself,
//! while the machine runs. A [`LocalApic`] therefore carries no interface: it
//! is proof that this processor's controller is up and that the machine's
//! controllers have been installed, and every operation on it asks
//! `IA32_APIC_BASE` which interface this processor presents *now*. Remembering
//! the answer is what a handle taken before a guest promoted its own processor
//! would be doing, and what it would be reaching is a page the architecture has
//! since made unavailable.
//!
//! For the same reason a handle belongs to the processor that obtained it and
//! cannot be sent to another: every register it names answers about whichever
//! processor is doing the reaching.
//!
//! # What the controller has
//!
//! Not the same thing on every machine. A controller reports how many local
//! vector table entries it has, and it has exactly the first that many of the
//! architecture's list — so the thermal and performance entries are simply
//! absent on a controller that counts four, and writing one that is absent is
//! undefined through the register page and a fault through the model-specific
//! registers. Nothing here touches an entry the controller has not claimed, and
//! nothing writes a field the entry it is writing does not have.
//!
//! # What this crate does not do
//!
//! It does not decide what happens when an interrupt arrives. Vectors, handlers
//! and what an unclaimed interrupt means belong to [`descriptors`] and to
//! whoever is using the machine; [`Timer::configure`] programs a timer and
//! registers nothing.
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
mod extended;
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
    marker::PhantomData,
    sync::atomic::{AtomicU64, Ordering},
};

use acpi::{LocalNmi, Madt, NmiTarget};
use cpu::{ApicId, CpuError};
use descriptors::{DescriptorError, Disposition, Interrupt, Vector};
use log::{info, warn};
use paging::{AddressSpace, CacheType, PagingError, Protection};
use processor::Features;
use spin::Once;
use thiserror::Error;
use x86_64::PhysAddr;

pub use crate::{
    capture::{Controller, FirmwareState, LVT_ENTRIES, LocalState, VECTOR_WORDS, capture},
    extended::{Extended, Specific},
    icr::{Command, Delivery, Target},
    lvt::{Delivery as LvtDelivery, Entry, Polarity, Trigger},
    register::{REGISTER_STRIDE, X2APIC_BASE_MSR, lvt_entries, version_number},
    smp::{Started, start},
    timer::{Divisor, IA32_TSC_DEADLINE, Mode as TimerMode, Timer},
};
use crate::{
    lvt::Shape,
    register::{Access, MappedRegister, Page, Register},
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
}

impl Apic {
    /// Maps the register page, silences what would interfere, and brings the
    /// boot processor's own controller up in at least `mode`.
    ///
    /// The mode is the caller's rather than this crate's, and what it should be
    /// is whichever one firmware was already in — because the controller is
    /// handed back to firmware as an emulated one, and the emulated
    /// controller's logical destination register can only agree with the
    /// real one when both present the same interface. At least, because the
    /// architecture allows no way back down: a controller firmware left in
    /// x2APIC stays there whatever is asked for, and [`Apic::mode`] is what it
    /// came to.
    ///
    /// Everything the machine can refuse is refused before anything permanent
    /// happens, and the register page is given back if a later step fails — so
    /// a machine this declines to run on is one nothing has been done to, save
    /// for the vectors below. Once it succeeds the page is mapped for good:
    /// every processor reaches its own controller through that one mapping for
    /// as long as the image runs, and there is no later moment at which giving
    /// it back would be anything but a mistake.
    ///
    /// Interrupts must be disabled on this processor. Switching a controller on
    /// before its handlers exist is only safe because nothing can be delivered
    /// while they do not.
    ///
    /// # Errors
    ///
    /// [`ApicError::AlreadyInstalled`] for a second call, which would leave the
    /// machine with two answers to a question that has one;
    /// [`ApicError::NoApic`] on a processor with no local controller;
    /// [`ApicError::NoRegisterPage`] if the processor reports its register page
    /// at address zero; [`ApicError::Paging`] if the register page cannot be
    /// mapped; [`ApicError::Cpu`] if the roster does not describe this
    /// processor; whatever bringing this processor's controller into `mode`
    /// reported; or [`ApicError::Descriptors`] if either of this crate's
    /// vectors is already claimed — which is the one refusal that leaves
    /// something behind, because a claimed vector cannot be given back and
    /// the first of the two may already be claimed when the second is
    /// refused.
    pub fn install(space: &mut AddressSpace, madt: &Madt, mode: Mode) -> Result<Self, ApicError> {
        if INSTALLED.is_completed() {
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
        for nmi in madt.local_nmis() {
            if nmi.input() > LAST_PIN {
                warn!(
                    "apic: firmware describes a non-maskable interrupt on local input {}, and a \
                     controller has only {}; it is ignored",
                    nmi.input(),
                    LAST_PIN + 1
                );
            }
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
        let page = unsafe { Page::new(mapping.addr()) };
        let claimed = Self::claim(page, madt, mode);
        if claimed.is_err() {
            // Nothing derived from the mapping outlives this function on this
            // path: the installation was never published, so no handle exists
            // that could name it.
            // SAFETY: as above — the page was reached only from `claim`, which
            // has returned, and nothing kept a copy.
            if let Err(error) = unsafe { space.unmap(mapping) } {
                warn!("apic: the register page could not be given back: {error}");
            }
        }
        claimed
    }

    /// Everything from the first permanent act to the last.
    ///
    /// Split out so that [`Apic::install`] can give the register page back on
    /// every path that does not get to the end of this, and so that the order
    /// inside is readable as one thing: the controller is switched on first,
    /// because it is the last step the machine itself can refuse; the vectors
    /// are claimed next, because a controller must not be told to deliver on a
    /// vector nothing is waiting for; and the rest cannot fail.
    fn claim(page: Page, madt: &Madt, mode: Mode) -> Result<Self, ApicError> {
        let entered = base::enter(mode)?;
        let access = access_through(entered, page);
        let id = identifier(access, entered);
        let uid = cpu::roster()?
            .find(id)
            .ok_or(CpuError::Unknown { apic_id: id })?
            .uid();
        let local_nmis = madt.local_nmis().to_vec();
        let wiring = wiring(&local_nmis, uid);

        descriptors::register(SPURIOUS, spurious)?;
        descriptors::register(ERROR, errors)?;

        // Also before anything is unmasked. The legacy controllers deliver onto
        // vectors 8 to 15 at reset, which are exceptions, and vector 8 is the
        // one nothing may claim.
        let legacy = madt.pic_8259();
        if legacy {
            pic::mask();
        }

        let extended = bring_up(access, wiring);
        INSTALLED.call_once(|| Installation {
            page,
            mode: entered,
            local_nmis,
            extended,
            legacy,
        });
        Ok(Self { entered })
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
        if INSTALLED.get().is_some_and(|installed| installed.legacy) {
            info!("{who}: apic masked both legacy 8259 controllers");
        }
        info!(
            "{who}: apic {}",
            INSTALLED
                .get()
                .map_or(Extended::empty(), |installed| installed.extended)
        );
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
/// Two things are true wherever one of these exists, and neither follows from
/// the other: the machine's controllers have been installed, and *this*
/// processor has switched its own on. It cannot be made outside this crate, and
/// inside it is only made where both hold.
///
/// What it does not carry is how the controller is reached. That is worked out
/// from `IA32_APIC_BASE` at each operation, because it is a per-processor fact
/// that changes while the machine runs — a guest may take its own processor
/// into x2APIC and leave the others where they were, and a handle that
/// remembered the old answer would go on addressing a page the architecture has
/// made unavailable. It is also why one of these may not be sent to another
/// processor: every register it names answers about whoever is doing the
/// reaching.
#[derive(Clone, Copy, Debug)]
pub struct LocalApic {
    installed: &'static Installation,
    /// Not shared, not sent: a handle belongs to the processor that obtained
    /// it.
    processor: PhantomData<*const ()>,
}

impl LocalApic {
    /// Brings this processor's controller up in the mode the machine was
    /// installed in.
    ///
    /// Called by every processor but the boot one, for itself, once it is
    /// running: entering a mode is something each processor does to its own
    /// controller, and the local vector table is per processor. Which mode is
    /// not the caller's to choose — it is the machine's, decided once by
    /// [`Apic::install`] — because a controller that presents a different
    /// interface from the rest is one whose logical destinations are spelled
    /// differently from every other processor's.
    ///
    /// A processor started with `INIT` arrives in whatever mode it was already
    /// in, because `INIT` preserves both of the bits that say. That is normally
    /// the machine's, and where it is already beyond it — x2APIC on a machine
    /// installed in xAPIC — it stays there rather than being refused, since the
    /// architecture offers no way back and losing the processor would be the
    /// only alternative.
    ///
    /// # Errors
    ///
    /// [`ApicError::NotInstalled`] before [`Apic::install`],
    /// [`ApicError::NoApic`] on a processor with no local controller,
    /// [`ApicError::UndefinedBase`] or [`ApicError::ModeNotEntered`] if this
    /// processor's controller cannot be brought up, [`ApicError::NoX2Apic`] if
    /// the machine is in x2APIC and this processor does not implement it, or
    /// [`ApicError::Cpu`] if the roster does not describe this processor.
    pub fn enable() -> Result<Self, ApicError> {
        let installed = INSTALLED.get().ok_or(ApicError::NotInstalled)?;
        let entered = base::enter(installed.mode)?;
        let access = access_through(entered, installed.page);
        let id = identifier(access, entered);
        let uid = cpu::roster()?
            .find(id)
            .ok_or(CpuError::Unknown { apic_id: id })?
            .uid();
        let extended = bring_up(access, wiring(&installed.local_nmis, uid));
        if extended != installed.extended {
            // Said rather than refused, because the processor is worth running
            // either way and there is nothing here that could change what was
            // decided from the boot processor's controller. What it costs is
            // stated where the decision is used: a controller that cannot retire
            // a named vector, asked to, retires nothing, and the withheld
            // acknowledgement stays withheld.
            warn!(
                "apic: {id} reports {extended} and the machine was installed with {}, so whichever \
                 of the two this processor does not have will not behave as the rest do",
                installed.extended
            );
        }
        Ok(Self::of(installed))
    }

    /// A handle to the controller of the processor calling this.
    const fn of(installed: &'static Installation) -> Self {
        Self {
            installed,
            processor: PhantomData,
        }
    }

    /// Which interface this processor's controller presents, right now.
    #[must_use]
    pub fn mode(self) -> Mode {
        if base::in_x2apic() {
            Mode::X2Apic
        } else {
            Mode::XApic
        }
    }

    /// How this processor reaches its controller's registers, right now.
    ///
    /// For the modules of this crate that drive a register file of their own —
    /// the timer — rather than for anything outside it.
    pub(crate) fn access(self) -> Access {
        access_through(self.mode(), self.installed.page)
    }

    /// Takes this processor's controller into x2APIC.
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
    /// Every handle to this controller — this one included, and any the caller
    /// took before — goes on working, because none of them remembers which
    /// interface the controller presents. The next operation on any of them
    /// asks again.
    ///
    /// # Errors
    ///
    /// [`ApicError::NoX2Apic`] if this processor does not implement it,
    /// [`ApicError::NoApic`] or [`ApicError::UndefinedBase`] if its base
    /// register says something unusable, or [`ApicError::ModeNotEntered`] if it
    /// took the write and did not change.
    pub fn enter_x2apic(self) -> Result<(), ApicError> {
        base::enter(Mode::X2Apic).map(drop)
    }

    /// This processor's identifier.
    ///
    /// The older interface keeps it in the top eight bits of the register; the
    /// newer one uses the whole of it.
    #[must_use]
    pub fn id(self) -> ApicId {
        let mode = self.mode();
        identifier(access_through(mode, self.installed.page), mode)
    }

    /// The controller's version register: its version in the low byte, and one
    /// less than its number of local vector table entries in the third.
    #[must_use]
    pub fn version(self) -> u32 {
        self.access().read(Register::VERSION)
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
        let (register, bit) = register::word(Register::TRIGGER_MODE, vector.number());
        self.access().read(register) & bit != 0
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
        let (register, bit) = register::word(Register::IN_SERVICE, vector.number());
        self.access().read(register) & bit != 0
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
    #[expect(
        clippy::cast_possible_truncation,
        reason = "eight slots of thirty-two bits is the whole of a vector's range, so neither the slot nor the sum can leave a byte"
    )]
    pub fn in_service_top(self) -> Option<Vector> {
        let access = self.access();
        register::bank(Register::IN_SERVICE)
            .enumerate()
            .rev()
            .find_map(|(slot, register)| {
                let word = access.read(register);
                (word != 0).then(|| {
                    let bit = u32::BITS - 1 - word.leading_zeros();
                    Vector::new((slot as u32 * u32::BITS + bit) as u8)
                })
            })
    }

    /// One of the controller's own sources, as the controller currently holds
    /// it.
    ///
    /// Everything the register holds, including the two bits the controller
    /// owns and any the model leaves reserved. Anything on its way back to
    /// a register loses them again; see [`LocalApic::program`].
    ///
    /// # Errors
    ///
    /// [`ApicError::NoSuchLvt`] if this controller does not have the entry.
    pub fn source(self, source: Source) -> Result<Entry, ApicError> {
        let access = self.access();
        let register = present(access, source)?;
        Ok(Entry::from_bits(access.read(register)))
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
    /// What reaches the register is the entry as this source can hold it.
    /// Fields the source does not have are dropped rather than written,
    /// which matters most for the ordinary case of reading an entry,
    /// changing one thing and writing it back: a register read carries two
    /// bits the controller owns and software may not set, and asking
    /// hardware to set them is not something to do and then rely on being
    /// ignored.
    ///
    /// The error entry is deliberately absent from [`Source`]. The controller's
    /// errors are the host's to notice, and an entry handed to a guest would be
    /// one the host stopped hearing about.
    ///
    /// # Errors
    ///
    /// [`ApicError::NoSuchLvt`] if this controller does not have the entry —
    /// three of them are optional and a controller reports how many it has;
    /// [`ApicError::WrongDelivery`] if the source cannot deliver the way the
    /// entry says, which includes the encodings the architecture reserves; or
    /// [`ApicError::IllegalVector`] for a vector no controller may deliver.
    pub fn program(self, source: Source, entry: Entry) -> Result<(), ApicError> {
        let access = self.access();
        let register = present(access, source)?;
        let shape = source.shape();
        let delivery = entry
            .delivery()
            .filter(|delivery| shape.allows(*delivery))
            .ok_or(ApicError::WrongDelivery { which: source })?;
        if let LvtDelivery::Fixed(vector) = delivery
            && !deliverable(vector)
        {
            return Err(ApicError::IllegalVector { vector });
        }
        // SAFETY: the register was just established to be one this controller
        // has, the delivery is one this source offers, its vector is one a
        // controller may deliver, and every bit the source does not have — the
        // two the controller owns and everything reserved — has been taken out.
        unsafe { access.write(register, entry.writable(shape).bits()) };
        Ok(())
    }

    /// Tells this controller how to match a logical destination, and which ones
    /// it answers to.
    ///
    /// The two registers are one operation, because a destination only means
    /// anything against the model that matches it. Written apart there is an
    /// interval in which the controller answers to a set of processors that is
    /// neither the old one nor the new one — and hardware goes on matching
    /// passed-through interrupts throughout it, so that set is not academic.
    ///
    /// So the destination is stood down first: a destination of zero matches
    /// nothing under either model, which turns the interval into one where this
    /// processor is not addressed rather than one where the wrong processor is.
    /// Then the model, then the destination the caller asked for. An interrupt
    /// that lands in the middle is one this processor does not take, which is
    /// the lesser of the two failures — a logically addressed interrupt has
    /// somewhere else to go, and one delivered to the wrong processor does not.
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
        let Access::Mapped(page) = self.access() else {
            return false;
        };
        // SAFETY: zero is a legal set of logical destinations — the empty one.
        // Of the format register only the top nibble selects anything and every
        // encoding of it is one the architecture defines, with the reserved
        // remainder written as ones, which is its reset value and what the
        // architecture requires. Every value of the destination register's top
        // byte is a legal set, and its remainder is reserved and written as zero.
        unsafe {
            page.write(Register::LOGICAL_DESTINATION, 0);
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
        self.access().read(Register::LOGICAL_DESTINATION)
    }

    /// Acknowledges the interrupt this processor is currently servicing.
    ///
    /// Owed for everything the controller delivered, and for nothing else: a
    /// spurious interrupt was never accepted, so acknowledging one would retire
    /// whatever really is in service instead.
    ///
    /// Retires whichever vector in service has the highest priority, because
    /// that is the only thing the register can be told to do. Anything that
    /// withheld an acknowledgement and means to issue it later has to establish
    /// that the vector it owes is still that one — or use
    /// [`LocalApic::specific`], where the question does not arise.
    pub fn end_of_interrupt(self) {
        self.access().acknowledge();
    }

    /// What the extended register space offers on this machine's controllers.
    ///
    /// Read once from the boot processor's own controller, for the reason
    /// [`Installation::extended`] gives.
    #[must_use]
    pub fn extended(self) -> Extended {
        self.installed.extended
    }

    /// Retiring one named vector and stopping one being accepted, on this
    /// processor's controller, or `None` where its controller cannot do both.
    ///
    /// The absence is the ordinary answer rather than a failure: it is every
    /// machine whose controllers have no extended register space, which is
    /// every Intel part and every emulator this hypervisor is developed
    /// against. Whoever asks has to have something to do without it.
    #[must_use]
    pub fn specific(self) -> Option<Specific> {
        self.extended().usable().then(|| Specific::new(self))
    }

    /// Sends `command`.
    ///
    /// # Errors
    ///
    /// [`ApicError::IdTooWide`] or [`ApicError::BroadcastId`] if the target
    /// cannot be named as one processor in the interface this processor
    /// presents, [`ApicError::IllegalVector`] if the command names a vector no
    /// controller may deliver, [`ApicError::SelfAddressed`] for a reset or
    /// startup addressed to a set including this processor, or
    /// [`ApicError::CommandStuck`] if a previous command has still not left.
    pub fn send(self, command: Command) -> Result<(), ApicError> {
        // Both from the same reading of the base register, so that the command
        // cannot be encoded for one interface and written through the other.
        let mode = self.mode();
        let access = access_through(mode, self.installed.page);
        let bits = command.bits(mode)?;
        // SAFETY: `bits` came from a `Command`, which can only describe
        // combinations the architecture defines: its reserved fields are zero,
        // its vector is one a controller may deliver, its delivery is one the
        // set it is addressed to may be asked for, and its destination was
        // checked against the width of this interface.
        unsafe { access.send(bits) }
    }
}

/// What the controller has noticed going wrong, and rearms it to report what
/// happens next.
///
/// The register latches, and the architecture requires a write before a read to
/// move what the controller has noticed into it. That write is also what clears
/// the previous report, so the pair is the whole of the operation: writing
/// again afterwards would throw away anything latched between the read and the
/// write, which is exactly the interval an error storm lands in.
fn take_errors(access: Access) -> u32 {
    // SAFETY: the register takes zero and nothing else — the only value x2APIC
    // permits — and the write is what makes the read report anything.
    unsafe { access.write(Register::ERROR_STATUS, 0) };
    access.read(Register::ERROR_STATUS)
}

/// How a controller in `mode` is reached, given where the register page was
/// mapped.
const fn access_through(mode: Mode, page: Page) -> Access {
    match mode {
        Mode::XApic => Access::Mapped(page),
        Mode::X2Apic => Access::Msr,
    }
}

/// The register a source is programmed through, if the controller reached
/// through `access` has it.
///
/// # Errors
///
/// [`ApicError::NoSuchLvt`] if it does not.
fn present(access: Access, source: Source) -> Result<Register, ApicError> {
    let register = source.register();
    let entries = register::lvt_entries(access.read(Register::VERSION));
    register::has_lvt(register, entries)
        .then_some(register)
        .ok_or(ApicError::NoSuchLvt { which: source })
}

/// What the controller reached through `access` calls the processor it belongs
/// to.
fn identifier(access: Access, mode: Mode) -> ApicId {
    let raw = access.read(Register::ID);
    ApicId::new(match mode {
        Mode::X2Apic => raw,
        Mode::XApic => raw >> XAPIC_ID_SHIFT,
    })
}

/// Brings one processor's controller up: nothing delivering, then delivering,
/// then the two sources that are meant to.
///
/// The order is the architecture's rather than a preference. Every source the
/// controller has is masked first, because a controller that firmware left
/// running is a controller with a timer armed for something that no longer
/// exists. The enable bit follows. Only then are the entries that are meant to
/// deliver something written, because a software-disabled controller holds
/// every entry masked and ignores an attempt to clear the bit.
///
/// Answers what the extended register space turned out to offer on this
/// processor, having switched on the parts of it this crate uses. It is done
/// here because it is per processor and belongs with everything else each
/// processor does to its own controller once, and it is done after the enable
/// bit for the same reason the sources are.
///
/// Nothing here can fail. Every register it writes is one the controller has
/// said it has, and every value is one the register accepts — which is what
/// lets installation put its last refusal before this and treat the rest as
/// done.
fn bring_up(access: Access, wiring: [Entry; 2]) -> Extended {
    let entries = register::lvt_entries(access.read(Register::VERSION));

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

    let [lint0, lint1] = wiring;
    for (register, entry, shape) in [
        (Register::LVT_LINT0, lint0, Shape::Pin),
        (Register::LVT_LINT1, lint1, Shape::Pin),
        (
            Register::LVT_ERROR,
            Entry::new(LvtDelivery::Fixed(ERROR)),
            Shape::Internal,
        ),
    ] {
        if register::has_lvt(register, entries) {
            // SAFETY: each of these is either the non-maskable delivery
            // firmware itself described, a masked pin, or this crate's own
            // error vector; the register is one the controller has; and the
            // value carries only fields the source it is written to holds.
            unsafe { access.write(register, entry.writable(shape).bits()) };
        }
    }

    retire_inherited(access);
    let extended = extended::install(access);
    // Last: the register latches whatever the controller noticed while it was
    // being set up, and none of that describes a running machine.
    let _ = take_errors(access);
    extended
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
        access.acknowledge();
    }
    let requested: u32 = register::bank(Register::INTERRUPT_REQUEST)
        .map(|register| access.read(register).count_ones())
        .sum();
    if in_service != 0 || requested != 0 {
        info!(
            "apic: retired {in_service} interrupts firmware left in service, {requested} still \
             requested"
        );
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

impl Display for Source {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
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

    /// Which of an entry's fields this source's register actually has.
    pub(crate) const fn shape(self) -> Shape {
        match self {
            Self::Timer => Shape::Timer,
            Self::Lint0 | Self::Lint1 => Shape::Pin,
            Self::Thermal | Self::Performance | Self::CorrectedMachineCheck => Shape::Internal,
        }
    }
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

/// The highest of the controller's own interrupt inputs, which it has two of.
const LAST_PIN: u8 = 1;

/// This processor's controller.
///
/// Answering means two things are true, and the second does not follow from the
/// first. The machine's controllers have been installed — and *this* processor
/// has switched its own on, which is something every processor does for itself
/// in [`LocalApic::enable`].
///
/// # Errors
///
/// [`ApicError::NotInstalled`] before [`Apic::install`], [`ApicError::NoApic`]
/// on a processor with no local controller, [`ApicError::UndefinedBase`] if its
/// base register names no interface at all, or [`ApicError::NotEnabled`] on a
/// processor that has not yet switched its own controller on.
pub fn local() -> Result<LocalApic, ApicError> {
    let installed = INSTALLED.get().ok_or(ApicError::NotInstalled)?;
    match base::state()? {
        base::State::Disabled => Err(ApicError::NotEnabled),
        base::State::XApic | base::State::X2Apic => Ok(LocalApic::of(installed)),
    }
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

/// Puts the legacy controllers' interrupt masks back to what firmware left
/// them, and says whether there were any to put back.
///
/// The one part of the host's own bring-up that is undone, and it is undone
/// because of what it costs to leave done. A guest that *is* firmware drives
/// its own periodic timer through these controllers and through the local
/// controller's first pin; it programmed both before this hypervisor existed
/// and never programs either again, so a mask laid over an input firmware had
/// left open is a firmware environment with no clock — no timer events, and
/// therefore no countdowns and no polled peripherals.
///
/// What the mask bought up to this point is kept, because this belongs at the
/// last moment before a guest is entered and nowhere earlier: an input firmware
/// had left open cannot assert into a host that is the only thing running and
/// has nowhere to hand an interrupt to, since it is still masked for the whole
/// of that. The interrupt that arrives afterwards arrives with somewhere to go.
///
/// `masks` is what [`capture`] read before anything had written them. Answers
/// `false` on a machine firmware said has no such controllers, where nothing
/// was masked and there is nothing to put back — writing to ports nothing
/// decodes is how such a machine gets a configuration it never had.
///
/// # Errors
///
/// [`ApicError::NotInstalled`] before [`Apic::install`], which is what decides
/// whether the machine has these controllers at all.
pub fn restore_legacy(masks: [u8; 2]) -> Result<bool, ApicError> {
    let installed = INSTALLED.get().ok_or(ApicError::NotInstalled)?;
    if installed.legacy {
        pic::restore(masks);
    }
    Ok(installed.legacy)
}

/// Masks every input of both legacy controllers again, and says whether there
/// were any to mask.
///
/// The counterpart of [`restore_legacy`], for the moment firmware's services
/// stop existing. Up to that point the guest is firmware and the legacy path is
/// firmware's own; past it, whatever the guest starts next programs both
/// controllers for itself before it uses either — every operating system
/// reinitializes them from scratch — so what firmware left behind is state
/// nothing is coming back for.
///
/// Leaving it behind is what costs. Those controllers deliver on vectors
/// firmware chose and only firmware had handlers for, and nothing can read back
/// where firmware put them: the vector base is write-only. So an input left
/// open past this point is an interrupt arriving on a number no guest has
/// claimed and this hypervisor cannot name, and masking is what makes that
/// impossible again.
///
/// # Errors
///
/// As [`restore_legacy`].
pub fn mask_legacy() -> Result<bool, ApicError> {
    let installed = INSTALLED.get().ok_or(ApicError::NotInstalled)?;
    if installed.legacy {
        pic::mask();
    }
    Ok(installed.legacy)
}

/// What is true of every processor's controller, decided once.
#[derive(Debug)]
struct Installation {
    /// Where the memory-mapped register file was mapped. The same physical page
    /// on every processor, each of which sees its own controller through it, so
    /// it is mapped once and shared — and it is kept even on a machine
    /// installed in x2APIC, because a processor that has not followed its
    /// guest across still reaches its controller here.
    page: Page,
    /// Which interface the machine's controllers were brought up in, and so the
    /// one every processor joining afterwards is brought up in.
    mode: Mode,
    /// What firmware said about the two interrupt pins of each processor.
    local_nmis: Vec<LocalNmi>,
    /// What the extended register space offers, as the boot processor's own
    /// controller reported it.
    ///
    /// Machine-wide because it is read once, and that is an assumption about
    /// the machine rather than about the architecture: the space and its
    /// parts are fixed at reset and uniform across a package, exactly as
    /// the features [`processor::features`] caches from one processor are.
    /// Every processor joining afterwards checks its own against this and
    /// says so if they differ, because what would follow otherwise is a
    /// vector-named acknowledgement issued to a controller that ignores it.
    extended: Extended,
    /// Whether firmware said the machine has the two legacy interrupt
    /// controllers, and so whether anything answers at the ports that mask
    /// them.
    ///
    /// Held here rather than beside the mode because it outlives the value
    /// [`Apic::install`] returns: the masks are put back for the firmware guest
    /// and taken away again once firmware's services stop existing, and neither
    /// of those moments has that value in reach.
    legacy: bool,
}

/// Decided by the boot processor, read by every processor bringing its own
/// controller up.
static INSTALLED: Once<Installation> = Once::new();

/// What to program into the two interrupt pins on the processor with this ACPI
/// identifier, the first pin first.
///
/// Masked unless firmware said the pin is wired as a non-maskable interrupt. A
/// pin left as firmware had it is a pin that can deliver to a vector chosen by
/// something no longer running, and on a modern machine both pins are either
/// unconnected or exactly this.
fn wiring(local_nmis: &[LocalNmi], uid: u32) -> [Entry; 2] {
    [0, 1].map(|input| {
        // A record naming this processor outranks one naming every processor,
        // whichever order firmware listed them in. ACPI describes connections
        // and states no precedence, so the only reading that makes both records
        // mean something is that the specific one is the exception the general
        // one is qualified by.
        let described = |specific: bool| {
            local_nmis.iter().find(|nmi| {
                nmi.input() == input
                    && match nmi.target() {
                        NmiTarget::Processor(target) => specific && target == uid,
                        NmiTarget::All => !specific,
                    }
            })
        };
        described(true)
            .or_else(|| described(false))
            .map_or_else(Entry::masked, |nmi| {
                Entry::new(LvtDelivery::NonMaskable)
                    .wired(polarity(nmi.polarity()), trigger(nmi.trigger()))
            })
    })
}

/// How a pin firmware described asserts.
///
/// The bus default is resolved here rather than left to mean whatever a reader
/// assumes: these pins are the ISA bus's inheritance, and that bus asserts
/// high.
const fn polarity(described: acpi::Polarity) -> Polarity {
    match described {
        acpi::Polarity::ActiveLow => Polarity::ActiveLow,
        acpi::Polarity::ActiveHigh | acpi::Polarity::BusDefault => Polarity::ActiveHigh,
    }
}

/// When a pin firmware described asserts.
///
/// The bus default is the ISA bus's again, which is edge triggered. It makes no
/// difference to delivery — the architecture delivers a non-maskable interrupt
/// edge triggered whatever this bit says — but it is what the register is asked
/// to hold, and a reader of it deserves the truth.
const fn trigger(described: acpi::Trigger) -> Trigger {
    match described {
        acpi::Trigger::Level => Trigger::Level,
        acpi::Trigger::Edge | acpi::Trigger::BusDefault => Trigger::Edge,
    }
}

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
/// direction, and worse: a real error arriving alongside an interrupt something
/// else sent on this vector reads as ours, and that arrival is consumed and
/// acknowledged. No read of any register can tell one arrival from another, so
/// the only real answer is not to let a guest program this vector into anything
/// — which is what [`vlapic`](https://docs.rs/vlapic) refusing it amounts to.
fn errors(_: &Interrupt) -> Disposition {
    // As in `spurious`: a controller that cannot be asked what it latched has
    // said nothing that would make this arrival ours.
    let Ok(local) = local() else {
        return Disposition::Passed;
    };
    let errors = take_errors(local.access());
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
    /// The machine needs x2APIC — because that is the interface its controllers
    /// were installed in, or because a processor's identifier does not fit
    /// eight bits — and this processor does not implement it.
    #[error("the machine needs x2apic and the processor does not have it")]
    NoX2Apic,
    /// The base register selects the model-specific interface with the
    /// controller switched off, which is a combination the architecture does
    /// not define and neither interface can be reached through.
    #[error("the controller's base register names neither interface")]
    UndefinedBase,
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
    /// This processor has not brought its own controller up.
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
    /// The source cannot deliver the way the entry says. Only the two pins can
    /// defer to a legacy controller, only an ordinary interrupt comes out of
    /// the timer, and three of the eight encodings are ones the
    /// architecture reserves.
    #[error("the {which} source cannot deliver that way")]
    WrongDelivery {
        /// The source that was asked for.
        which: Source,
    },
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
    /// The identifier is the one each interface reserves for addressing every
    /// processor at once, so a command meant for one would reach all of them.
    #[error("{apic_id} is this interface's broadcast identifier rather than a processor")]
    BroadcastId {
        /// The identifier that could not be named.
        apic_id: ApicId,
    },
    /// A reset, a startup or a non-maskable interrupt was addressed to a set of
    /// processors that includes the one sending it, which one vendor calls
    /// invalid and the other leaves undefined.
    #[error("that command cannot be addressed to the processor sending it")]
    SelfAddressed,
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
    /// A deadline was given to a counting timer, or a count to one waiting for
    /// a deadline. The architecture ignores the write either way.
    #[error("that timer mode does not take that kind of deadline")]
    WrongTimerMode,
    /// The processor does not implement the timestamp counter deadline.
    #[error("the processor does not have the tsc deadline timer")]
    NoTscDeadline,
    /// There is no timebase to measure against or to wait on.
    #[error("no timebase is installed")]
    Clock,
    /// The timer did not move, or ran out of count before the measurement was
    /// over.
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
    /// A processor began executing the trampoline and never got as far as
    /// taking the parameters out of it, so nothing else can be started: the
    /// next processor would have to overwrite parameters this one may still
    /// read.
    #[error("{apic_id} began starting and never took its parameters")]
    StartupUnresolved {
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
