//! What reading or writing each register means.
//!
//! Both faces of the controller end up here, and that is the point. The
//! architecture defines a register's behaviour once and offers it through two
//! interfaces; a hypervisor that implemented the behaviour twice would be one
//! whose guest could tell which interface it was using by the answers it got.
//!
//! So the faces above this module do only what genuinely differs between them —
//! decoding an address, deciding whether a malformed access faults or merely
//! records an error, splitting a 64-bit register into halves — and every
//! question about what a register *is* is answered here.
//!
//! # Some reads are not reads
//!
//! Two registers change when read or write in ways that make the plain words
//! misleading, and both are called out where they are implemented: writing the
//! interrupt command register is how an interrupt is sent, and writing the
//! error status register is what makes a subsequent read report anything at
//! all.

use descriptors::Vector;
use log::warn;

use crate::{
    error::Errors,
    icr::Command,
    lvt::Entry,
    register::{Bank, Register},
    state::Vlapic,
    timer,
};

/// What the guest sees when it reads a register.
///
/// Every register that is not simply stored is computed here rather than kept
/// up to date as things change, because the architecture defines most of them
/// as functions of other state and a stored copy is a copy that can be wrong.
pub(crate) fn read(vlapic: &Vlapic, register: Register) -> u32 {
    match register {
        Register::ID => vlapic.id_register(),
        Register::VERSION => Vlapic::version(),
        Register::TASK_PRIORITY => u32::from(vlapic.task_priority().get()),
        Register::ARBITRATION_PRIORITY => u32::from(vlapic.arbitration_priority().get()),
        Register::PROCESSOR_PRIORITY => u32::from(vlapic.processor_priority().get()),
        Register::LOGICAL_DESTINATION => vlapic.logical_destination(),
        Register::DESTINATION_FORMAT => vlapic.destination_format(),
        Register::SPURIOUS => vlapic.spurious(),
        Register::ERROR_STATUS => vlapic.errors().read(),
        Register::COMMAND_LOW => vlapic.command().low(),
        Register::COMMAND_HIGH => vlapic.command().high(),
        Register::TIMER_INITIAL_COUNT => vlapic.timer_initial(),
        // The one register whose value is not this hypervisor's at all: the
        // guest's timer is the real timer, so what it has left is what the real
        // one has left.
        Register::TIMER_CURRENT_COUNT => timer::remaining(vlapic),
        Register::TIMER_DIVIDE => vlapic.timer_divide(),
        // Three registers that answer nothing. The first two are write-only and
        // the faces above refuse a read of either before reaching here, so
        // answering zero rather than asserting keeps a mis-decode from becoming
        // a fault on the interrupt path. The third was never implemented on any
        // processor this runs on, and reading it has no side effect worth
        // inventing.
        Register::END_OF_INTERRUPT | Register::SELF_IPI | Register::REMOTE_READ => 0,
        other => read_indexed(vlapic, other),
    }
}

/// What a write to a register does.
///
/// The answer says whether anything downstream has to happen that could fail —
/// sending an interrupt, reprogramming real hardware — so that the faces above
/// can report it without knowing what any register means.
pub(crate) fn write(vlapic: &Vlapic, register: Register, value: u32) -> Written {
    match register {
        Register::TASK_PRIORITY => {
            vlapic.set_task_priority(value);
            Written::Nothing
        }
        Register::LOGICAL_DESTINATION => {
            vlapic.set_logical_destination(value);
            Written::LogicalDestination
        }
        Register::DESTINATION_FORMAT => {
            vlapic.set_destination_format(value);
            Written::LogicalDestination
        }
        Register::SPURIOUS => {
            vlapic.set_spurious(value);
            // Software-disabling a controller masks every entry, and the
            // entries are programmed into real hardware, so the two have to be
            // brought back into agreement.
            Written::LocalVectorTable
        }
        Register::END_OF_INTERRUPT => Written::EndOfInterrupt,
        Register::ERROR_STATUS => {
            vlapic.errors().written();
            Written::Nothing
        }
        // Writing the low half is what sends the command. Writing the high half
        // only says where the next one will go.
        Register::COMMAND_LOW => Written::Command(vlapic.set_command_low(value)),
        Register::COMMAND_HIGH => {
            vlapic.set_command_high(value);
            Written::Nothing
        }
        Register::SELF_IPI => Written::SelfIpi(Vector::new(vector_of(value))),
        Register::TIMER_INITIAL_COUNT => {
            vlapic.set_timer_initial(value);
            Written::Timer
        }
        Register::TIMER_DIVIDE => {
            vlapic.set_timer_divide(value);
            Written::Timer
        }
        other => write_indexed(vlapic, other, value),
    }
}

/// The registers that are one of several of a kind: a slot of a bank, or a
/// local-vector-table entry.
fn read_indexed(vlapic: &Vlapic, register: Register) -> u32 {
    if let Some((bank, slot)) = register.bank() {
        return match bank {
            Bank::InService => vlapic.in_service_slot(slot),
            Bank::TriggerMode => vlapic.trigger_mode_slot(slot),
            Bank::InterruptRequest => vlapic.request_slot(slot),
        };
    }
    Entry::of(register).map_or(0, |entry| vlapic.lvt(entry).into_bits())
}

/// As [`read_indexed`]. Only the local-vector-table entries are writable; the
/// three banks are read-only and the faces above refuse a write to them.
fn write_indexed(vlapic: &Vlapic, register: Register, value: u32) -> Written {
    match Entry::of(register) {
        Some(entry) => {
            vlapic.write_lvt(entry, value);
            match entry {
                // The timer's entry carries the mode it counts in, so writing
                // it can change what the real timer is doing — though not, on
                // its own, start it.
                Entry::Timer => Written::Timer,
                _ => Written::LocalVectorTable,
            }
        }
        None => Written::Nothing,
    }
}

/// The vector an eight-bit field of a wider value names.
#[expect(
    clippy::cast_possible_truncation,
    reason = "a vector is the low eight bits of the register it is written in"
)]
const fn vector_of(value: u32) -> u8 {
    value as u8
}

/// What a write asked for that the register file alone cannot finish.
///
/// A guest's write to a controller is very often not merely a store. It can
/// send an interrupt to another processor, retire one on this one, or change
/// what real hardware is doing — and all three can fail in ways worth
/// reporting, on paths the register file has no business reaching from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Written {
    /// The value was stored and that is all.
    Nothing,
    /// Send this.
    Command(Command),
    /// Send this processor an interrupt to itself, which is the same thing as a
    /// command with the self shorthand and a fixed delivery.
    SelfIpi(Vector),
    /// Retire the interrupt the guest is servicing, which may mean
    /// acknowledging real hardware.
    EndOfInterrupt,
    /// The local vector table has changed and real hardware has to be brought
    /// into agreement with it.
    LocalVectorTable,
    /// The timer's configuration has changed.
    Timer,
    /// Which logical destinations this processor answers to has changed, and
    /// the real controller has to be told, because hardware matches passed
    /// through interrupts against the real register rather than against this
    /// one.
    LogicalDestination,
    /// The guest changed which face it reaches its controller through, which
    /// reset every register and may mean the real controller has to follow.
    ModeChanged,
}

/// Records that the guest named a register the older face reserves.
///
/// Only that face: reaching a reserved index through the model-specific
/// registers is a general protection fault instead, and the architecture is
/// explicit that this bit is not set for one.
pub(crate) fn illegal_register(vlapic: &Vlapic, register: Option<Register>) {
    if vlapic.errors().record(Errors::ILLEGAL_REGISTER_ADDRESS) {
        // Only the first, because a guest probing its register page produces
        // one of these per probe and the error status register latches them all
        // into the same bit anyway.
        warn!(
            "vlapic: {} named a reserved register at offset {:#x}",
            vlapic.index(),
            register.map_or(0, Register::offset)
        );
    }
}
