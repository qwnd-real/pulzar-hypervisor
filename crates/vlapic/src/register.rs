//! Which register an access names, and what may be done with it.
//!
//! A guest reaches the same controller two ways. Through the memory-mapped page
//! a register is a byte offset; through x2APIC it is a model-specific register
//! index, and the index is *derived* from the offset as `0x800 + offset / 16`.
//! So there is one list of registers here, named by offset, and both faces
//! decode into it.
//!
//! What differs between the two faces is not which registers exist but what
//! happens when software gets one wrong, and that difference is the reason this
//! module answers with an [`Access`] rather than a bare yes or no:
//!
//! - Through the page, naming a reserved offset cannot fault. It sets the
//!   illegal-register-address bit in the error status register and the access
//!   otherwise does nothing.
//! - Through the model-specific registers, naming a reserved index is a general
//!   protection fault, and so is writing a read-only register, reading a
//!   write-only one, or putting a non-zero value in any reserved bit.
//!
//! Three registers exist on one face and not the other, and that asymmetry is
//! in [`Access::of`] rather than in a convention: the destination format
//! register and the high half of the interrupt command are gone in x2APIC
//! because their indices are reserved, and the self-interrupt register exists
//! only in x2APIC because there is no offset it would sit at.

use crate::base::Mode;

/// One of the controller's registers, named by its offset in the memory-mapped
/// page.
///
/// The offset is the canonical name even for a guest using x2APIC, because it
/// is what the model-specific register index is computed from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Register(u32);

impl Register {
    /// This processor's identifier.
    pub(crate) const ID: Self = Self(0x20);
    /// The controller's version, and how many local vector table entries it
    /// reports.
    pub(crate) const VERSION: Self = Self(0x30);
    /// Task priority: which interrupt priorities the guest will accept.
    pub(crate) const TASK_PRIORITY: Self = Self(0x80);
    /// Arbitration priority. Absent in x2APIC.
    pub(crate) const ARBITRATION_PRIORITY: Self = Self(0x90);
    /// The priority the guest is actually servicing at.
    pub(crate) const PROCESSOR_PRIORITY: Self = Self(0xA0);
    /// Written to acknowledge the interrupt being serviced.
    pub(crate) const END_OF_INTERRUPT: Self = Self(0xB0);
    /// Remote read. Absent in x2APIC, and never implemented on anything this
    /// hypervisor runs on.
    pub(crate) const REMOTE_READ: Self = Self(0xC0);
    /// Which logical destinations this processor answers to.
    pub(crate) const LOGICAL_DESTINATION: Self = Self(0xD0);
    /// How a logical destination is matched. Absent in x2APIC, which has only
    /// the cluster model.
    pub(crate) const DESTINATION_FORMAT: Self = Self(0xE0);
    /// Spurious interrupt vector, and the bit that software-enables the
    /// controller.
    pub(crate) const SPURIOUS: Self = Self(0xF0);
    /// First of the eight registers saying which vectors are in service.
    pub(crate) const IN_SERVICE: Self = Self(0x100);
    /// First of the eight saying which of them arrived level triggered.
    pub(crate) const TRIGGER_MODE: Self = Self(0x180);
    /// First of the eight saying which vectors are requested but not accepted.
    pub(crate) const INTERRUPT_REQUEST: Self = Self(0x200);
    /// Errors the controller noticed, latched until written.
    pub(crate) const ERROR_STATUS: Self = Self(0x280);
    /// Local vector table entry for corrected machine-check errors.
    pub(crate) const LVT_CORRECTED_MACHINE_CHECK: Self = Self(0x2F0);
    /// The low half of the interrupt command register, and the whole of it in
    /// x2APIC.
    pub(crate) const COMMAND_LOW: Self = Self(0x300);
    /// The high half of the interrupt command: the destination. Absent in
    /// x2APIC, where the destination is the upper half of one wide register.
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
    /// Sending oneself an interrupt. Exists only in x2APIC: there is no
    /// memory-mapped register at this offset at all.
    pub(crate) const SELF_IPI: Self = Self(0x3F0);

    /// The register a byte offset into the page names, if any.
    ///
    /// Only offsets on a 128-bit boundary name anything. The architecture gives
    /// every 32-bit register a 16-byte slot and leaves the other twelve bytes
    /// undefined, so an access to one of them is an access to a reserved
    /// address rather than to part of a register.
    pub(crate) const fn at(offset: u64) -> Option<Self> {
        if offset >= PAGE || !offset.is_multiple_of(STRIDE as u64) {
            return None;
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the offset was just bounded by the size of a 4 KiB page"
        )]
        Some(Self(offset as u32))
    }

    /// The register a model-specific register index names, if any.
    pub(crate) const fn from_msr(index: u32) -> Option<Self> {
        if index < X2APIC_BASE_MSR || index > X2APIC_LAST_MSR {
            return None;
        }
        Some(Self((index - X2APIC_BASE_MSR) * STRIDE))
    }

    /// Its offset in the memory-mapped page.
    pub(crate) const fn offset(self) -> u32 {
        self.0
    }

    /// Which of a bank's eight slots this is, given the register the bank
    /// starts at.
    pub(crate) const fn slot_of(self, first: Self) -> Option<usize> {
        if self.0 < first.0 || self.0 >= first.0 + BANK_SLOTS * STRIDE {
            return None;
        }
        Some(((self.0 - first.0) / STRIDE) as usize)
    }

    /// Which bank of one-bit-per-vector registers this is in, and which of its
    /// eight slots, if it is in one at all.
    pub(crate) const fn bank(self) -> Option<(Bank, usize)> {
        if let Some(slot) = self.slot_of(Self::IN_SERVICE) {
            return Some((Bank::InService, slot));
        }
        if let Some(slot) = self.slot_of(Self::TRIGGER_MODE) {
            return Some((Bank::TriggerMode, slot));
        }
        if let Some(slot) = self.slot_of(Self::INTERRUPT_REQUEST) {
            return Some((Bank::InterruptRequest, slot));
        }
        None
    }
}

/// One of the three registers that give every vector a bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Bank {
    /// Accepted and not yet acknowledged.
    InService,
    /// Which of those arrived level triggered.
    TriggerMode,
    /// Delivered and not yet accepted.
    InterruptRequest,
}

/// What software may do with a register, in the mode it is reaching it through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Access {
    /// Readable and writable.
    ReadWrite,
    /// Readable; a write is refused.
    ReadOnly,
    /// Writable; a read is refused.
    WriteOnly,
    /// Not a register in this mode at all.
    Absent,
}

impl Access {
    /// What may be done with a register by a guest in this mode.
    ///
    /// The identifier register is read-only here on both faces. The
    /// architecture made it writable on the oldest processors and read-only
    /// from Nehalem onwards, software is told not to write it either way, and a
    /// guest that changed the identifier its interrupts are addressed by would
    /// be describing a machine this hypervisor cannot then deliver to — every
    /// passed-through interrupt is routed by the *real* identifier, which is
    /// not the guest's to move.
    pub(crate) const fn of(register: Register, mode: Mode) -> Self {
        let x2apic = matches!(mode, Mode::X2Apic);
        match register {
            Register::ID
            | Register::VERSION
            | Register::PROCESSOR_PRIORITY
            | Register::TIMER_CURRENT_COUNT => Self::ReadOnly,
            Register::END_OF_INTERRUPT => Self::WriteOnly,
            // Arbitration and remote read are gone in x2APIC, and neither has
            // ever done anything on a processor this hypervisor runs on.
            Register::ARBITRATION_PRIORITY | Register::REMOTE_READ => {
                if x2apic {
                    Self::Absent
                } else {
                    Self::ReadOnly
                }
            }
            // Derived by hardware from the identifier in x2APIC, and so not the
            // guest's to write.
            Register::LOGICAL_DESTINATION => {
                if x2apic {
                    Self::ReadOnly
                } else {
                    Self::ReadWrite
                }
            }
            // x2APIC has only the cluster model, so there is nothing left for
            // the format register to select between; and the destination half of
            // the command is reachable only as the upper half of the one wide
            // register. Both indices are reserved there.
            Register::DESTINATION_FORMAT | Register::COMMAND_HIGH => {
                if x2apic {
                    Self::Absent
                } else {
                    Self::ReadWrite
                }
            }
            // No memory-mapped register sits at this offset at all.
            Register::SELF_IPI => {
                if x2apic {
                    Self::WriteOnly
                } else {
                    Self::Absent
                }
            }
            Register::TASK_PRIORITY
            | Register::SPURIOUS
            | Register::ERROR_STATUS
            | Register::COMMAND_LOW
            | Register::LVT_CORRECTED_MACHINE_CHECK
            | Register::LVT_TIMER
            | Register::LVT_THERMAL
            | Register::LVT_PERFORMANCE
            | Register::LVT_LINT0
            | Register::LVT_LINT1
            | Register::LVT_ERROR
            | Register::TIMER_INITIAL_COUNT
            | Register::TIMER_DIVIDE => Self::ReadWrite,
            other => banked(other),
        }
    }
}

/// What may be done with a register that is one slot of a bank, or
/// [`Access::Absent`] for an offset in none of them.
const fn banked(register: Register) -> Access {
    match register.bank() {
        // All three banks are read-only on both faces.
        Some(_) => Access::ReadOnly,
        None => Access::Absent,
    }
}

/// How many 32-bit registers one bank of one-bit-per-vector state spans.
const BANK_SLOTS: u32 = 8;

/// Bytes between one memory-mapped register and the next.
///
/// Each register is 32 bits wide and each gets a 16-byte slot, which is why
/// dividing an offset by this turns it into a model-specific register index.
const STRIDE: u32 = 16;

/// Index of the model-specific register the register at offset zero maps to.
pub(crate) const X2APIC_BASE_MSR: u32 = 0x800;

/// The last index the architecture reserves for the controller.
pub(crate) const X2APIC_LAST_MSR: u32 = 0x8FF;

/// How long the memory-mapped register page is.
const PAGE: u64 = 4096;
