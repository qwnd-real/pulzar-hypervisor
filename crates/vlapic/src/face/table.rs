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
//! Five registers exist on one face and not the other, and that asymmetry is in
//! [`Access::of`] rather than in a convention. The destination format register
//! and the high half of the interrupt command are gone in x2APIC because their
//! indices are reserved there — the format register has nothing left to select
//! between, and the destination is the upper half of one wide register. The
//! arbitration priority and remote read registers are gone with the bus they
//! belonged to. And the self-interrupt register exists only in x2APIC, because
//! there is no offset it would sit at.
//!
//! Those last two are answered as read-only through the page rather than as
//! absent, and the distinction is the architecture's own: the register table
//! names exactly those two as the registers for which the
//! illegal-register-access error is *not* raised. So a guest reading either
//! through the page gets a value and no error, where a guest naming a genuinely
//! reserved offset gets zero and the error.
//!
//! # Limitations
//!
//! The extended APIC register space AMD defines at 0x400–0x530 is not offered
//! to the guest, and that is a requirement rather than an omission: two of its
//! registers are how this hypervisor settles what real hardware is holding on
//! the guest's behalf — see [`crate::lifecycle::ledger::immediate`] — so a
//! guest able to reach them could retire an interrupt the host is accounting
//! for, or stop a vector the host has just unblocked from arriving.
//!
//! Three things keep it out, and each would do on its own.
//! `CPUID Fn8000_0001_ECX[3]` is cleared, so software that honours the
//! capability bit never looks. The bit the version register sets to announce
//! the space is dropped from what the emulated one reports, so software that
//! checks there does not look either. And every offset of the range is answered
//! here as reserved: a guest that probes regardless reads zero and records an
//! illegal-register error, exactly as it would on a controller that has no such
//! space.
//!
//! Modelling it instead would mean emulating the extended interrupt-enable and
//! extended local vector table registers, and the machine-check and
//! instruction-based-sampling sources that use them.

use apic::{REGISTER_STRIDE, X2APIC_BASE_MSR};

use crate::{
    hardware::model::Model,
    registers::{base::Mode, bitmap::SLOTS, lvt::Entry},
};

/// One of the controller's registers, named by its offset in the memory-mapped
/// page.
///
/// The offset is the canonical name even for a guest using x2APIC, because it
/// is what the model-specific register index is computed from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
        if offset >= PAGE || !offset.is_multiple_of(REGISTER_STRIDE as u64) {
            return None;
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the offset was just bounded by the size of a 4 KiB page"
        )]
        Some(Self(offset as u32))
    }

    /// The register a model-specific register index names, if any.
    ///
    /// Bounded by the page as well as by the range x2APIC reserves, because the
    /// architecture reserves four times as many indices as the page has
    /// registers: the ones whose derived offset would be past its end name
    /// nothing at all, and are answered here rather than turned into a
    /// [`Register`] that is outside the page it is an offset into.
    pub(crate) const fn from_msr(index: u32) -> Option<Self> {
        if index < X2APIC_BASE_MSR || index > X2APIC_LAST_MSR {
            return None;
        }
        Self::at((index - X2APIC_BASE_MSR) as u64 * REGISTER_STRIDE as u64)
    }

    /// Its offset in the memory-mapped page.
    pub(crate) const fn offset(self) -> u32 {
        self.0
    }

    /// Which of a bank's eight slots this is, given the register the bank
    /// starts at.
    ///
    /// The count of slots is the bitmap's own rather than a second one, because
    /// the bank a guest reads *is* that bitmap: a disagreement would be a guest
    /// read of the register at the top of a bank answered out of nothing.
    ///
    /// Private, because unchecked offset arithmetic is how a register leaves
    /// the bank it was promised to be in: [`Register::bank`] is the answer
    /// every caller wants, and it names the bank the slot belongs to.
    const fn slot_of(self, first: Self) -> Option<usize> {
        if self.0 < first.0 {
            return None;
        }
        let slot = ((self.0 - first.0) / REGISTER_STRIDE) as usize;
        if slot >= SLOTS {
            return None;
        }
        Some(slot)
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
    pub(crate) fn of(register: Register, mode: Mode, model: Model) -> Self {
        // Matched exhaustively rather than compared against the one mode that
        // answers differently, so that a mode added later cannot silently be
        // answered for as though it were the older face.
        let x2apic = match mode {
            Mode::X2Apic => true,
            Mode::XApic => false,
            // A switched-off controller has no registers at all: the page
            // decodes to nothing and the indices are not assigned. Answered
            // here rather than left to the callers to know, so that the
            // function is total — both of them do establish the mode first, and
            // a third caller that did not would otherwise be handed the older
            // face's answers for a controller that has no face.
            Mode::Disabled => return Self::Absent,
        };
        match register {
            // An entry this controller does not have is not a register at all.
            // Three of the seven are optional, the model takes its count from
            // the hardware the sources actually live on, and a guest reaching
            // one it was never told about must find nothing there rather than a
            // register it can program and no source behind it.
            _ if Entry::absent(register, model) => Self::Absent,
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

/// The last index the architecture reserves for the controller.
///
/// Its own constant rather than the real controller's, because it bounds what
/// this crate *intercepts* rather than what it can reach: an index in the range
/// that names no register still has to be answered, with a fault, and one that
/// is never intercepted is one a guest executes against real hardware.
///
/// So it is the end of the range the architecture dedicates to the controller
/// and not the end of the quarter of it that holds registers. Three quarters of
/// these indices name nothing, and a guest reaching one of them is entitled to
/// a general protection fault — which it cannot be given by code that never
/// sees the access, because interception here is default-allow.
pub(crate) const X2APIC_LAST_MSR: u32 = 0xBFF;

/// How long the memory-mapped register page is.
///
/// Where the page ends is what makes an offset past it a reserved address
/// rather than a register, so this belongs to the register table and the
/// aperture that answers for the page takes its length from here.
pub(crate) const PAGE: u64 = 4096;

#[cfg(test)]
mod tests {
    //! A table is the one kind of code a wrong hex digit survives review in, so
    //! every offset, every index derived from one, every bank boundary and
    //! every access class is asserted here rather than read.

    use apic::{REGISTER_STRIDE, X2APIC_BASE_MSR};

    use super::{Access, Bank, PAGE, Register, X2APIC_LAST_MSR};
    use crate::{
        hardware::model,
        registers::{base::Mode, bitmap::SLOTS},
    };

    /// Every register the table names, with what may be done with it through
    /// the page and through the model-specific registers.
    ///
    /// Written out rather than derived, so that a register added without a
    /// decision about either face, or a cell that changes, fails a test rather
    /// than being answered for by a fall-through. Two cells are worth naming
    /// here because they are the ones a reader is likely to take for a mistake:
    /// the identifier is read-only through the page as well, because every
    /// interrupt this hypervisor passes through is routed by the *real*
    /// identifier and a guest that renamed its controller could no longer be
    /// delivered to; and the error status register is writable although the
    /// architecture's table calls it read only, because a write is what latches
    /// it and software has to perform one before a read answers anything.
    const MATRIX: [(Register, Access, Access); 27] = [
        (Register::ID, Access::ReadOnly, Access::ReadOnly),
        (Register::VERSION, Access::ReadOnly, Access::ReadOnly),
        (
            Register::TASK_PRIORITY,
            Access::ReadWrite,
            Access::ReadWrite,
        ),
        (
            Register::ARBITRATION_PRIORITY,
            Access::ReadOnly,
            Access::Absent,
        ),
        (
            Register::PROCESSOR_PRIORITY,
            Access::ReadOnly,
            Access::ReadOnly,
        ),
        (
            Register::END_OF_INTERRUPT,
            Access::WriteOnly,
            Access::WriteOnly,
        ),
        (Register::REMOTE_READ, Access::ReadOnly, Access::Absent),
        (
            Register::LOGICAL_DESTINATION,
            Access::ReadWrite,
            Access::ReadOnly,
        ),
        (
            Register::DESTINATION_FORMAT,
            Access::ReadWrite,
            Access::Absent,
        ),
        (Register::SPURIOUS, Access::ReadWrite, Access::ReadWrite),
        (Register::IN_SERVICE, Access::ReadOnly, Access::ReadOnly),
        (Register::TRIGGER_MODE, Access::ReadOnly, Access::ReadOnly),
        (
            Register::INTERRUPT_REQUEST,
            Access::ReadOnly,
            Access::ReadOnly,
        ),
        (Register::ERROR_STATUS, Access::ReadWrite, Access::ReadWrite),
        (
            Register::LVT_CORRECTED_MACHINE_CHECK,
            Access::ReadWrite,
            Access::ReadWrite,
        ),
        (Register::COMMAND_LOW, Access::ReadWrite, Access::ReadWrite),
        (Register::COMMAND_HIGH, Access::ReadWrite, Access::Absent),
        (Register::LVT_TIMER, Access::ReadWrite, Access::ReadWrite),
        (Register::LVT_THERMAL, Access::ReadWrite, Access::ReadWrite),
        (
            Register::LVT_PERFORMANCE,
            Access::ReadWrite,
            Access::ReadWrite,
        ),
        (Register::LVT_LINT0, Access::ReadWrite, Access::ReadWrite),
        (Register::LVT_LINT1, Access::ReadWrite, Access::ReadWrite),
        (Register::LVT_ERROR, Access::ReadWrite, Access::ReadWrite),
        (
            Register::TIMER_INITIAL_COUNT,
            Access::ReadWrite,
            Access::ReadWrite,
        ),
        (
            Register::TIMER_CURRENT_COUNT,
            Access::ReadOnly,
            Access::ReadOnly,
        ),
        (Register::TIMER_DIVIDE, Access::ReadWrite, Access::ReadWrite),
        (Register::SELF_IPI, Access::Absent, Access::WriteOnly),
    ];

    #[test]
    fn what_may_be_done_with_each_register_in_each_face() {
        for (register, xapic, x2apic) in MATRIX {
            assert_eq!(
                Access::of(register, Mode::XApic, model::tests::AMD),
                xapic,
                "{register:?} through the page"
            );
            assert_eq!(
                Access::of(register, Mode::X2Apic, model::tests::AMD),
                x2apic,
                "{register:?} through the model-specific registers"
            );
            // A controller its guest has switched off has no registers in either
            // face, and that is the whole of what switching one off means.
            assert_eq!(
                Access::of(register, Mode::Disabled, model::tests::AMD),
                Access::Absent,
                "{register:?} on a switched-off controller"
            );
        }
    }

    #[test]
    fn every_register_sits_on_a_128_bit_boundary_inside_the_page() {
        for (register, ..) in MATRIX {
            assert!(u64::from(register.offset()) < PAGE, "{register:?}");
            assert!(
                register.offset().is_multiple_of(REGISTER_STRIDE),
                "{register:?}"
            );
        }
    }

    #[test]
    fn an_index_and_an_offset_name_the_same_register() {
        // The whole of the range the architecture dedicates to the controller,
        // not merely the quarter of it that holds registers: the two
        // constructors have to agree about which offsets are registers, and
        // three quarters of these indices derive an offset past the end of the
        // page.
        for index in X2APIC_BASE_MSR..=X2APIC_LAST_MSR {
            let offset = u64::from(index - X2APIC_BASE_MSR) * u64::from(REGISTER_STRIDE);
            assert_eq!(
                Register::from_msr(index),
                Register::at(offset),
                "{index:#x}"
            );
        }
        // Either side of the range, and the first index whose offset is past the
        // end of the page — which is where the registers stop, long before the
        // indices do.
        assert_eq!(Register::from_msr(X2APIC_BASE_MSR - 1), None);
        assert_eq!(Register::from_msr(X2APIC_LAST_MSR + 1), None);
        assert_eq!(Register::from_msr(0x900), None);
        assert_eq!(Register::from_msr(0x83F), Some(Register::SELF_IPI));
    }

    #[test]
    fn a_bank_is_eight_consecutive_registers_and_stops_below_the_next() {
        // Eight slots of sixteen bytes, the stride being the one the boundary
        // test above pins to the architecture's.
        for (bank, first) in [
            (Bank::InService, 0x100_u64),
            (Bank::TriggerMode, 0x180),
            (Bank::InterruptRequest, 0x200),
        ] {
            let slots = (first..first + 0x80).step_by(0x10);
            assert_eq!(slots.clone().count(), SLOTS, "{bank:?}");
            for (slot, offset) in slots.enumerate() {
                let register = Register::at(offset).expect("a bank sits inside the page");
                assert_eq!(register.bank(), Some((bank, slot)), "{offset:#x}");
            }
        }
        // The register one slot past the last bank is the error status and not a
        // ninth word of the interrupt request register. One slot more would
        // answer a guest's read of its error status out of a bitmap; one fewer
        // would make the top thirty-two vectors a reserved address.
        assert_eq!(Register::at(0x280), Some(Register::ERROR_STATUS));
        assert_eq!(Register::ERROR_STATUS.bank(), None);
        // Nor is the slot below the first bank part of one.
        assert_eq!(Register::SPURIOUS.bank(), None);
    }

    #[test]
    fn only_an_offset_on_a_boundary_inside_the_page_names_a_register() {
        assert_eq!(Register::at(0x20), Some(Register::ID));
        // Four-byte aligned inside a register's own slot, which the
        // architecture leaves undefined and this table answers as a reserved
        // address rather than as part of the register.
        assert_eq!(Register::at(0x24), None);
        // The last aligned dword of the page, and the first offset past it.
        assert_eq!(Register::at(0xFFC), None);
        assert_eq!(Register::at(PAGE), None);
        assert!(Register::at(PAGE - u64::from(REGISTER_STRIDE)).is_some());
    }

    #[test]
    fn the_offsets_the_architecture_assigns_nothing_to_are_registers_in_neither_face() {
        // Every reserved slot of the page, including the extended space AMD
        // defines from 0x400 and this crate does not model.
        let reserved = [0x000_u64, 0x010]
            .into_iter()
            .chain((0x040..0x080).step_by(0x10))
            .chain((0x290..0x2F0).step_by(0x10))
            .chain((0x3A0..0x3E0).step_by(0x10))
            .chain((0x400..PAGE).step_by(0x10));
        for offset in reserved {
            let register = Register::at(offset).expect("an aligned offset inside the page");
            for mode in [Mode::XApic, Mode::X2Apic, Mode::Disabled] {
                assert_eq!(
                    Access::of(register, mode, model::tests::AMD),
                    Access::Absent,
                    "{offset:#x} in {mode}"
                );
            }
        }
    }

    #[test]
    fn an_entry_the_controller_does_not_have_is_not_a_register() {
        // The three optional local vector table entries, on a controller
        // reporting the fewest the architecture describes. A guest reaching one
        // of these finds a reserved address through the page and a fault through
        // the model-specific registers, rather than a register it can program
        // with no source behind it.
        for register in [
            Register::LVT_PERFORMANCE,
            Register::LVT_THERMAL,
            Register::LVT_CORRECTED_MACHINE_CHECK,
        ] {
            for mode in [Mode::XApic, Mode::X2Apic] {
                assert_eq!(
                    Access::of(register, mode, model::tests::SPARSE),
                    Access::Absent,
                    "{register:?}"
                );
            }
        }
    }
}
