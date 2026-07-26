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
//! Four places, all handled by name rather than by a general mechanism, because
//! four is all there is.
//!
//! The interrupt command register is two 32-bit registers in the older
//! interface and one 64-bit model-specific register in x2APIC. A write to the
//! older one is posted, so its delivery status has to be polled before the next
//! command; an x2APIC write is not, so there is nothing to poll and the bit
//! does not exist. The destination format register exists only in the older
//! interface. And x2APIC adds a register for sending a processor an interrupt
//! to itself, which the older interface can only do the long way round.
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
    /// How the logical destination above is matched. The older interface only:
    /// x2APIC has one model and no register to choose it with.
    pub(crate) const DESTINATION_FORMAT: Self = Self(0xE0);
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
    /// Local vector table entry for corrected machine-check errors. The last
    /// entry the architecture added, so a controller only has it if its version
    /// register counts far enough to reach it.
    pub(crate) const LVT_CORRECTED_MACHINE_CHECK: Self = Self(0x2F0);
    /// The low half of the interrupt command register: everything but the
    /// destination.
    pub(crate) const COMMAND_LOW: Self = Self(0x300);
    /// The high half of the interrupt command register: the destination.
    pub(crate) const COMMAND_HIGH: Self = Self(0x310);
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

    /// The byte offset into the memory-mapped page.
    const fn offset(self) -> u64 {
        self.0 as u64
    }
}

/// Index of the model-specific register the register at offset zero maps to.
const X2APIC_BASE_MSR: u32 = 0x800;

/// Bytes between one memory-mapped register and the next. Each is 32 bits wide
/// and each gets a 16-byte slot, which is why dividing by it turns an offset
/// into a model-specific register index.
const STRIDE: u32 = 16;

/// The single model-specific register x2APIC gives the interrupt command, in
/// place of the two the older interface splits it across.
pub(crate) const X2APIC_COMMAND_MSR: u32 = 0x830;

/// The bit the older interface sets in the command register while a command is
/// still being sent. x2APIC has no such bit: its write does not return until
/// the command has been accepted.
const DELIVERY_PENDING: u32 = 1 << 12;

/// How the local APIC's registers are reached on this machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Access {
    /// Through the memory-mapped page, at the address it was mapped to.
    Mapped(VirtAddr),
    /// Through model-specific registers.
    Msr,
}

impl Access {
    /// Reads a register.
    pub(crate) fn read(self, register: Register) -> u32 {
        match self {
            Self::Mapped(base) => {
                // SAFETY: `base` is the address a whole 4 KiB register page was
                // mapped at, uncached, and every `Register` is an offset inside
                // it that the architecture defines as a readable 32-bit slot.
                // Volatile because these are registers and the read is the
                // point.
                unsafe { ptr::read_volatile(pointer(base, register)) }
            }
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
            Self::Mapped(base) => {
                // SAFETY: as in `read`, and the caller vouches for the value.
                unsafe { ptr::write_volatile(pointer(base, register), value) };
            }
            // SAFETY: as in `read`, and the caller vouches for the value. The
            // upper half is zero, which every one of these registers requires.
            Self::Msr => unsafe { Msr::new(register.msr()).write(u64::from(value)) },
        }
    }

    /// Sends an interrupt command, waiting first for any previous one to have
    /// left.
    ///
    /// The two halves of the older interface are written destination first,
    /// because writing the low half is what sends the command.
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
            Self::Mapped(_) => {
                self.settle()?;
                // SAFETY: the destination half accepts any value in its top
                // eight bits, and writing it sends nothing on its own.
                unsafe { self.write(Register::COMMAND_HIGH, truncate(command >> u32::BITS)) };
                // SAFETY: the caller vouches for the command, and the previous
                // one has left.
                unsafe { self.write(Register::COMMAND_LOW, truncate(command)) };
                self.settle()
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
/// is called stuck. Each read is a bus cycle, so this is comfortably longer
/// than any delivery and still a bounded wait on a broken machine.
const COMMAND_POLLS: u32 = 1_000_000;

/// Where a register sits in the mapped page.
fn pointer(base: VirtAddr, register: Register) -> *mut u32 {
    (base + register.offset()).as_mut_ptr::<u32>()
}

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
pub(crate) fn establish(access: Access) -> Access {
    *ACCESS.call_once(|| access)
}

/// How the registers are reached, once something has decided.
///
/// # Errors
///
/// [`ApicError::NotInstalled`] before the boot processor has chosen.
pub(crate) fn access() -> Result<Access, ApicError> {
    ACCESS.get().copied().ok_or(ApicError::NotInstalled)
}
