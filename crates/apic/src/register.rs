//! The local APIC's registers, and the two ways a processor reaches them.
//!
//! There are two interfaces to the same controller. The older one puts its
//! registers in a 4 KiB page of memory-mapped space; x2APIC puts them in
//! model-specific registers instead, which is faster, needs no mapping, and is
//! the only way to address a processor whose identifier does not fit eight
//! bits.
//!
//! Nothing above this module should have to know which is in use, and nothing
//! here should have to say where a register is twice. It does not have to: the
//! model-specific register index is *derived* from the memory-mapped offset,
//! `0x800 + offset / 16`, which is how the architecture defined it. So there is
//! one list of registers, written once, and [`Access`] decides how it is
//! reached.
//!
//! # Where the two genuinely differ
//!
//! Four places, and the type system keeps three of them straight rather than a
//! convention doing it.
//!
//! The interrupt command register is two 32-bit registers in the older
//! interface and one 64-bit model-specific register in x2APIC, so it is offered
//! as [`Access::command`] and [`Access::send`] rather than as an offset, and no
//! caller assembles it twice. A write to the older one is posted, so its
//! delivery status has to be polled before the next command; an x2APIC write is
//! not, so there is nothing to poll and the bit does not exist.
//!
//! The upper half of that command register and the destination format register
//! exist only in the older interface, and the difference is not that x2APIC
//! leaves them unused: the model-specific registers they would occupy are
//! absent, and naming one is a general protection fault rather than a zero. So
//! those two are [`MappedRegister`]s rather than [`Register`]s, reachable only
//! through the [`Page`] that *is* the older interface — which means only code
//! that has already established which interface is in use can name them.
//!
//! The fourth is x2APIC's register for sending a processor an interrupt to
//! itself. Nothing here uses it: every command this crate sends is addressed to
//! another processor, and one addressed to this processor goes through the same
//! command register as the rest.
//!
//! # One mapping, every processor
//!
//! The memory-mapped page is at the same physical address on every processor,
//! and each one sees its own controller through it. So it is mapped once and
//! the address is shared, which is why this module holds a global rather than
//! handing out a mapping per processor.

use core::ptr;

use spin::Once;
use x86_64::{VirtAddr, registers::model_specific::Msr};

use crate::ApicError;

/// Something with an offset in the memory-mapped register page.
///
/// Implemented by both kinds of register, so that the page is read and written
/// through one pair of methods while [`Access`] — which may not be the page at
/// all — takes only the registers both interfaces have.
pub(crate) trait Offset: Copy {
    /// The byte offset into the page.
    fn offset(self) -> u64;
}

/// One of the local APIC's registers, named by its offset in the memory-mapped
/// page.
///
/// The offset is the canonical name even when x2APIC is in use, because it is
/// the one the model-specific register index is computed from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Register(u32);

impl Register {
    /// This processor's identifier.
    pub(crate) const ID: Self = Self(0x20);
    /// The controller's version, and how many local vector table entries it
    /// has.
    pub(crate) const VERSION: Self = Self(0x30);
    /// Task priority: which interrupt priorities this processor will accept.
    pub(crate) const TASK_PRIORITY: Self = Self(0x80);
    /// The priority this processor is actually servicing at, which is its task
    /// priority or the highest interrupt in service, whichever is higher.
    pub(crate) const PROCESSOR_PRIORITY: Self = Self(0xA0);
    /// Written to acknowledge the interrupt currently being serviced.
    pub(crate) const END_OF_INTERRUPT: Self = Self(0xB0);
    /// Which logical destinations this processor answers to.
    pub(crate) const LOGICAL_DESTINATION: Self = Self(0xD0);
    /// Spurious interrupt vector, and the bit that software-enables the
    /// controller.
    pub(crate) const SPURIOUS: Self = Self(0xF0);
    /// First of the eight registers saying which vectors this processor has
    /// accepted and not yet acknowledged.
    pub(crate) const IN_SERVICE: Self = Self(0x100);
    /// First of the eight registers saying which of the vectors in service
    /// arrived level triggered.
    pub(crate) const TRIGGER_MODE: Self = Self(0x180);
    /// First of the eight registers saying which vectors have been delivered to
    /// this processor and not yet accepted.
    pub(crate) const INTERRUPT_REQUEST: Self = Self(0x200);
    /// Errors the controller noticed, latched until written.
    pub(crate) const ERROR_STATUS: Self = Self(0x280);
    /// Local vector table entry for corrected machine-check errors.
    pub(crate) const LVT_CORRECTED_MACHINE_CHECK: Self = Self(0x2F0);
    /// The low half of the interrupt command register, and the whole of it
    /// under x2APIC.
    pub(crate) const COMMAND_LOW: Self = Self(0x300);
    /// Local vector table entry for the controller's own timer.
    pub(crate) const LVT_TIMER: Self = Self(0x320);
    /// Local vector table entry for the thermal sensor.
    pub(crate) const LVT_THERMAL: Self = Self(0x330);
    /// Local vector table entry for the performance counters.
    pub(crate) const LVT_PERFORMANCE: Self = Self(0x340);
    /// Local vector table entry for the first local interrupt pin.
    pub(crate) const LVT_LINT0: Self = Self(0x350);
    /// Local vector table entry for the second local interrupt pin.
    pub(crate) const LVT_LINT1: Self = Self(0x360);
    /// Local vector table entry for the controller's own errors.
    pub(crate) const LVT_ERROR: Self = Self(0x370);
    /// What the timer counts down from.
    pub(crate) const TIMER_INITIAL_COUNT: Self = Self(0x380);
    /// What the timer has left.
    pub(crate) const TIMER_CURRENT_COUNT: Self = Self(0x390);
    /// How far the bus clock is divided before the timer counts it.
    pub(crate) const TIMER_DIVIDE: Self = Self(0x3E0);

    /// The register `slots` slots past this one.
    ///
    /// Three of the controller's registers are really the first of eight
    /// consecutive ones, describing the two hundred and fifty-six vectors
    /// thirty-two at a time. Deriving the rest from the first is what keeps
    /// twenty-four offsets from being written out by hand.
    pub(crate) const fn offset_by(self, slots: u32) -> Self {
        Self(self.0 + slots * STRIDE)
    }

    /// The model-specific register x2APIC puts this register in.
    const fn msr(self) -> u32 {
        X2APIC_BASE_MSR + self.0 / STRIDE
    }
}

impl Offset for Register {
    fn offset(self) -> u64 {
        u64::from(self.0)
    }
}

/// One of the two registers only the memory-mapped interface has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MappedRegister(u32);

impl MappedRegister {
    /// The high half of the interrupt command register: the destination.
    /// x2APIC keeps the destination in the upper word of its single wide
    /// register instead.
    pub(crate) const COMMAND_HIGH: Self = Self(0x310);
    /// How a logical destination is matched.
    pub(crate) const DESTINATION_FORMAT: Self = Self(0xE0);
}

impl Offset for MappedRegister {
    fn offset(self) -> u64 {
        u64::from(self.0)
    }
}

/// Every local vector table entry, in the order the version register counts
/// them.
///
/// That order is the architecture's and is not the order the registers sit in.
/// A controller says how many entries it has, and it has exactly the first that
/// many of these — so one reporting four has the timer, the two pins and its
/// own error entry, and no performance, thermal or machine-check entry at all.
/// Touching one it does not have is undefined through the page and a general
/// protection fault through the model-specific registers, which is why nothing
/// here reaches for an entry without asking first.
const LVT_ENTRIES: [Register; 7] = [
    Register::LVT_TIMER,
    Register::LVT_LINT0,
    Register::LVT_LINT1,
    Register::LVT_ERROR,
    Register::LVT_PERFORMANCE,
    Register::LVT_THERMAL,
    Register::LVT_CORRECTED_MACHINE_CHECK,
];

/// How many of the controller's registers it takes to describe every vector one
/// bit at a time: two hundred and fifty-six vectors, thirty-two to a register.
pub(crate) const VECTOR_SLOTS: usize = 8;

/// The eight consecutive registers a bank of one bit per vector is spread
/// across, lowest vectors first.
pub(crate) fn bank(first: Register) -> impl Iterator<Item = Register> {
    (0..)
        .take(VECTOR_SLOTS)
        .map(move |slot| first.offset_by(slot))
}

/// Every local vector table entry a controller counting `entries` of them has,
/// in the order it counts them.
pub(crate) fn lvt_present(entries: u32) -> impl Iterator<Item = Register> {
    LVT_ENTRIES
        .into_iter()
        .zip(0..)
        .take_while(move |(_, index)| *index < entries)
        .map(|(register, _)| register)
}

/// Whether a controller counting `entries` local vector table entries has this
/// one.
pub(crate) fn has_lvt(register: Register, entries: u32) -> bool {
    lvt_present(entries).any(|candidate| candidate == register)
}

/// How many local vector table entries the version register reports.
///
/// The field holds one less than the count, so every controller has at least
/// one and the arithmetic cannot wrap.
pub(crate) const fn lvt_entries(version: u32) -> u32 {
    ((version >> LVT_COUNT_SHIFT) & VERSION_FIELD) + 1
}

/// The controller's version, out of the register that also carries the entry
/// count.
pub(crate) const fn version_number(version: u32) -> u32 {
    version & VERSION_FIELD
}

/// Index of the model-specific register the register at offset zero maps to.
const X2APIC_BASE_MSR: u32 = 0x800;

/// Bytes between one memory-mapped register and the next. Each is 32 bits wide
/// and each gets a 16-byte slot, which is why dividing by it turns an offset
/// into a model-specific register index.
const STRIDE: u32 = 16;

/// The single model-specific register x2APIC gives the interrupt command, in
/// place of the two the older interface splits it across.
const X2APIC_COMMAND_MSR: u32 = 0x830;

/// The version register's fields are one byte each.
const VERSION_FIELD: u32 = 0xFF;

/// Bits the version register's local-vector-table count is shifted by.
const LVT_COUNT_SHIFT: u32 = 16;

/// The bit the older interface sets in the command register while a command is
/// still being sent. x2APIC has no such bit: its write does not return until
/// the command has been accepted.
const DELIVERY_PENDING: u32 = 1 << 12;

/// The page of memory-mapped registers, at the address it was mapped to.
///
/// Holding one of these says the older interface is the one in use, which is
/// what makes it the only way to reach a [`MappedRegister`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Page(VirtAddr);

impl Page {
    /// The register page, mapped at `base`.
    ///
    /// # Safety
    ///
    /// `base` must be a whole 4 KiB local APIC register page, mapped for as
    /// long as this value is used, and mapped *uncached*. The architecture
    /// requires these registers to be reached strongly uncacheable: through a
    /// write-back mapping a read may be answered out of a cache line, which
    /// says nothing about what the controller currently holds, and a write may
    /// sit in a buffer past the point the machine depends on it having landed.
    pub(crate) const unsafe fn new(base: VirtAddr) -> Self {
        Self(base)
    }

    /// Reads a register.
    pub(crate) fn read(self, register: impl Offset) -> u32 {
        // SAFETY: a whole 4 KiB page is mapped uncached at this address, which
        // is what `new` requires, and every register is an offset inside it that
        // the architecture defines as a readable 32-bit slot. Volatile because
        // these are registers and the read is the point.
        unsafe { ptr::read_volatile(self.pointer(register)) }
    }

    /// Writes a register.
    ///
    /// # Safety
    ///
    /// The value must be one the register accepts. What a wrong one does ranges
    /// from an interrupt arriving somewhere nothing expects it to a processor
    /// held in reset.
    pub(crate) unsafe fn write(self, register: impl Offset, value: u32) {
        // SAFETY: as in `read`, and the caller vouches for the value.
        unsafe { ptr::write_volatile(self.pointer(register), value) };
    }

    /// Where a register sits in the page.
    fn pointer(self, register: impl Offset) -> *mut u32 {
        (self.0 + register.offset()).as_mut_ptr::<u32>()
    }
}

/// How the local APIC's registers are reached on this machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Access {
    /// Through the memory-mapped page.
    Mapped(Page),
    /// Through model-specific registers.
    Msr,
}

impl Access {
    /// Reads a register.
    pub(crate) fn read(self, register: Register) -> u32 {
        match self {
            Self::Mapped(page) => page.read(register),
            // SAFETY: the register exists because this variant is only chosen
            // once x2APIC has been enabled on this processor, and reading any of
            // the controller's registers has no side effect the architecture
            // does not document as one of its uses.
            Self::Msr => truncate(unsafe { Msr::new(register.msr()).read() }),
        }
    }

    /// Writes a register.
    ///
    /// # Safety
    ///
    /// The value must be one the register accepts. What a wrong one does ranges
    /// from a general protection fault, for a reserved bit in a model-specific
    /// register, to an interrupt arriving somewhere nothing expects it.
    pub(crate) unsafe fn write(self, register: Register, value: u32) {
        match self {
            // SAFETY: the caller vouches for the value.
            Self::Mapped(page) => unsafe { page.write(register, value) },
            // SAFETY: as in `read`, and the caller vouches for the value. The
            // upper half is zero, which every one of these registers requires.
            Self::Msr => unsafe { Msr::new(register.msr()).write(u64::from(value)) },
        }
    }

    /// The interrupt command as one value, whichever interface holds it and in
    /// however many pieces.
    pub(crate) fn command(self) -> u64 {
        match self {
            Self::Mapped(page) => {
                u64::from(page.read(MappedRegister::COMMAND_HIGH)) << u32::BITS
                    | u64::from(page.read(Register::COMMAND_LOW))
            }
            // SAFETY: the register is architectural on a controller in x2APIC
            // mode, which is the only way this variant is reached, and reading
            // it neither sends a command nor disturbs one.
            Self::Msr => unsafe { Msr::new(X2APIC_COMMAND_MSR).read() },
        }
    }

    /// Sends an interrupt command, waiting first for any previous one to have
    /// left.
    ///
    /// The two halves of the older interface are written destination first,
    /// because writing the low half is what sends the command.
    ///
    /// The wait is before the write and not after it. What it protects is the
    /// register's contents, which the previous command owns until it has left;
    /// once this one is written the register is this command's and there is
    /// nothing a caller could do with the news that it has gone. A command that
    /// never leaves is reported to whoever sends the next one, and not asking
    /// twice saves an uncached read on a path that sends one interrupt per
    /// processor.
    ///
    /// # Errors
    ///
    /// [`ApicError::CommandStuck`] if a previous command is still pending after
    /// the wait — which means the controller has not accepted something the
    /// processor gave it, and sending another would overwrite it.
    ///
    /// # Safety
    ///
    /// `command` must be a well-formed interrupt command. It is an instruction
    /// to interrupt a processor, and a malformed one can hold a processor in
    /// reset or deliver to a vector nothing is prepared for.
    pub(crate) unsafe fn send(self, command: u64) -> Result<(), ApicError> {
        match self {
            Self::Mapped(page) => {
                self.settle()?;
                // SAFETY: the destination half accepts any value in its top
                // eight bits, and writing it sends nothing on its own.
                unsafe { page.write(MappedRegister::COMMAND_HIGH, truncate(command >> u32::BITS)) };
                // SAFETY: the caller vouches for the command, and the previous
                // one has left.
                unsafe { page.write(Register::COMMAND_LOW, truncate(command)) };
                Ok(())
            }
            Self::Msr => {
                // SAFETY: the caller vouches for the command. The write does
                // not return until the controller has accepted it, so there is
                // nothing to wait for on either side of it.
                unsafe { Msr::new(X2APIC_COMMAND_MSR).write(command) };
                Ok(())
            }
        }
    }

    /// Waits for the older interface to finish sending whatever it was sending.
    ///
    /// A bounded spin rather than a delay: the wait is normally a handful of
    /// bus cycles, and a controller that has not finished after this many
    /// reads is not going to.
    fn settle(self) -> Result<(), ApicError> {
        let Self::Mapped(_) = self else {
            return Ok(());
        };
        for _ in 0..COMMAND_POLLS {
            if self.read(Register::COMMAND_LOW) & DELIVERY_PENDING == 0 {
                return Ok(());
            }
            core::hint::spin_loop();
        }
        Err(ApicError::CommandStuck)
    }
}

/// How many times the command register is read before a still-pending command
/// is called stuck. Each read goes to the controller and back, so this is
/// comfortably longer than any delivery and still a bounded wait on a broken
/// machine.
const COMMAND_POLLS: u32 = 1_000_000;

/// The low half of a 64-bit value.
///
/// Every register here is 32 bits wide, so narrowing is what reading one out of
/// a wider container means rather than a loss to guard against.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the local APIC's registers are 32 bits wide; the upper half is not part of the value"
)]
const fn truncate(value: u64) -> u32 {
    value as u32
}

/// How this machine's local APICs are reached, decided once by the boot
/// processor and the same for all of them.
static ACCESS: Once<Access> = Once::new();

/// Records how the registers are reached.
///
/// The choice belongs to the machine, not to a processor: every processor is
/// put into the same mode, so that one destination format and one register
/// width serve all of them.
///
/// # Errors
///
/// [`ApicError::AlreadyInstalled`] if something has already decided. The cell
/// runs the closure for the caller that fills it and for no other, so whether
/// it ran is exactly whether this call is the one that decided.
pub(crate) fn establish(access: Access) -> Result<(), ApicError> {
    let mut decided = false;
    ACCESS.call_once(|| {
        decided = true;
        access
    });
    decided.then_some(()).ok_or(ApicError::AlreadyInstalled)
}

/// How the registers are reached, once something has decided.
///
/// # Errors
///
/// [`ApicError::NotInstalled`] before the boot processor has chosen.
pub(crate) fn access() -> Result<Access, ApicError> {
    ACCESS.get().copied().ok_or(ApicError::NotInstalled)
}
