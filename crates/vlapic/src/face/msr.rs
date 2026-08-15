//! The face a guest reaches its controller through in x2APIC mode.
//!
//! The same registers as the memory-mapped page, reached as model-specific
//! registers instead, with the index derived from the offset. What differs is
//! entirely in how software is told it got something wrong: through the page
//! nothing faults, and here almost everything does.
//!
//! The architecture's phrasing for the reserved bits is `RsvdZ` — writing a
//! non-zero value to one is a general protection fault, and reading one gives
//! zero. That applies to the upper half of every register except the interrupt
//! command, which is the one register that is genuinely 64 bits wide.
//!
//! # These are entry points, not a device
//!
//! There is no page to trap, so nothing calls into here through [`emulate`]. A
//! guest in x2APIC mode reaches its controller with `RDMSR` and `WRMSR`, which
//! are intercepted through the permission map, and the exit handler calls
//! [`read_msr`] and [`write_msr`] directly.

use apic::IA32_TSC_DEADLINE;
use descriptors::Vector;
use log::{trace, warn};

use crate::{
    VlapicError,
    face::{
        dispatch::{self, Written, acted},
        table::{Access, Register, X2APIC_BASE_MSR, X2APIC_LAST_MSR},
    },
    hardware::timer,
    machine::current,
    registers::{
        Vlapic,
        base::{ApicBase, BaseFault, Mode},
        icr::Command,
        lvt::{Entry, TimerMode},
    },
};

/// What the guest reads from one of the controller's model-specific registers.
///
/// `tsc_offset` is the offset applied to this processor's guest timestamp and
/// is used to translate timestamp-counter deadline readback into that domain.
///
/// # Errors
///
/// [`VlapicError::NotInstalled`] before [`crate::install`],
/// [`VlapicError::NoLapic`] on a processor with no controller, or
/// [`VlapicError::Fault`] if the guest should take a general protection fault
/// for the access.
pub fn read_msr(index: u32, tsc_offset: u64) -> Result<u64, VlapicError> {
    let vlapic = current()?;
    let value = read(vlapic, index, tsc_offset).map_err(|fault| {
        warn!("vlapic: refusing a read of {index:#x}: {fault:?}");
        VlapicError::Fault
    })?;
    trace!(
        "vlapic: {} read {index:#05x} to its {value:#018x} model-specific register",
        vlapic.index()
    );
    Ok(value)
}

/// What a write to one of the controller's model-specific registers does.
///
/// `tsc_offset` is the offset applied to this processor's guest timestamp and
/// is used to translate a timestamp-counter deadline onto physical hardware.
///
/// # Errors
///
/// As [`read_msr`].
pub fn write_msr(index: u32, value: u64, tsc_offset: u64) -> Result<(), VlapicError> {
    let vlapic = current()?;
    let written = write(vlapic, index, value, tsc_offset).map_err(|fault| {
        warn!("vlapic: refusing a write of {value:#x} to {index:#x}: {fault:?}");
        VlapicError::Fault
    })?;
    trace!(
        "vlapic: {} wrote {value:#018x} to its {index:#05x} model-specific register",
        vlapic.index()
    );
    acted(vlapic, written);
    Ok(())
}

/// Every model-specific register the guest's controller answers for.
///
/// Handed to whatever programs the permission map. The whole of the range the
/// architecture reserves for the controller is named, not merely the indices
/// that hold a register: reaching an unassigned one is a fault the guest is
/// entitled to, and it cannot be given one by code that never sees the access.
pub fn intercepted() -> impl Iterator<Item = u32> {
    (X2APIC_BASE_MSR..=X2APIC_LAST_MSR).chain([ApicBase::MSR, IA32_TSC_DEADLINE])
}

/// Whether an index is one this crate answers for.
///
/// The whole of the controller's reserved range counts, not merely the indices
/// that name a register: reaching an unassigned one is a fault the guest has to
/// be given, and it cannot be given one by code that never sees the access.
#[must_use]
pub const fn claims(index: u32) -> bool {
    matches!(index, X2APIC_BASE_MSR..=X2APIC_LAST_MSR)
        || index == ApicBase::MSR
        || index == IA32_TSC_DEADLINE
}

/// What the guest reads.
///
/// # Errors
///
/// [`Fault`] if the index names nothing, names a register that does not exist
/// in the mode the guest is in, or names one that may not be read — each of
/// which the guest takes as a general protection fault.
pub(crate) fn read(vlapic: &Vlapic, index: u32, tsc_offset: u64) -> Result<u64, Fault> {
    if index == ApicBase::MSR {
        return Ok(vlapic.base().bits());
    }
    if index == IA32_TSC_DEADLINE {
        // Absent unless the processor reports it, and `CPUID` is passed through
        // — so a guest told the feature does not exist finds the register does
        // not exist either.
        if !vlapic.model().deadline() {
            return Err(Fault::NoSuchRegister);
        }
        return Ok(guest_deadline(timer::deadline(vlapic), tsc_offset));
    }
    let register = addressable(vlapic, index)?;
    if !matches!(
        Access::of(register, vlapic.mode(), vlapic.model()),
        Access::ReadOnly | Access::ReadWrite
    ) {
        return Err(Fault::WriteOnly);
    }
    // The one register that is really 64 bits wide. Every other read has a zero
    // upper half, which is what `RsvdZ` requires.
    if register == Register::COMMAND_LOW {
        return Ok(vlapic.command().bits());
    }
    Ok(u64::from(dispatch::read(vlapic, register)))
}

/// What a write does.
///
/// # Errors
///
/// As [`read`], and additionally [`Fault::Reserved`] if a bit the register
/// reserves was written non-zero — which through this face is a fault rather
/// than something quietly dropped.
pub(crate) fn write(
    vlapic: &Vlapic,
    index: u32,
    value: u64,
    tsc_offset: u64,
) -> Result<Written, Fault> {
    if index == ApicBase::MSR {
        return vlapic
            .write_base(value)
            .map(Written::ModeChanged)
            .map_err(Fault::Base);
    }
    if index == IA32_TSC_DEADLINE {
        if !vlapic.model().deadline() {
            return Err(Fault::NoSuchRegister);
        }
        // Hardware ignores this write outside deadline mode rather than
        // faulting, so a refusal here is not an error — but it has to be a
        // refusal and nothing else. Reporting a timer action for an ignored
        // write is what would let a `WRMSR` the architecture discards go on to
        // re-arm a timer.
        if vlapic.timer_mode() != Some(TimerMode::Deadline) {
            return Ok(Written::Nothing);
        }
        return Ok(Written::TimerDeadline(physical_deadline(value, tsc_offset)));
    }
    let register = addressable(vlapic, index)?;
    if !matches!(
        Access::of(register, vlapic.mode(), vlapic.model()),
        Access::ReadWrite | Access::WriteOnly
    ) {
        return Err(Fault::ReadOnly);
    }
    // The one register that is genuinely 64 bits wide, and the one whose
    // reserved fields have to be judged before anything is stored: a write that
    // must raise a fault must not first have sent an interrupt, reset a
    // processor, or changed what a guest reads back.
    if register == Register::COMMAND_LOW {
        if value & !Command::WRITABLE_X2APIC != 0 {
            return Err(Fault::Reserved);
        }
        return Ok(Written::Command(vlapic.set_command(value)));
    }
    // Everything else is a 32-bit register whose upper half is reserved.
    if value > u64::from(u32::MAX) {
        return Err(Fault::Reserved);
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the upper half was just established to be zero"
    )]
    let narrow = value as u32;
    // Reserved bits within the low half fault here rather than being dropped,
    // which is the whole difference between the two faces: through the page a
    // stray bit is quietly discarded, and through a model-specific register it
    // is a general protection fault. Judged before the write, for the same
    // reason the command register is.
    if narrow & !writable(vlapic, register) != 0 {
        return Err(Fault::Reserved);
    }
    if register == Register::SELF_IPI {
        return Ok(Written::SelfIpi(Vector::new(vector_of(narrow))));
    }
    Ok(dispatch::write(vlapic, register, narrow))
}

/// Translates a physical deadline into the timestamp domain the guest reads.
///
/// Zero is not a timestamp: it is the architectural spelling of a disarmed
/// timer and must remain zero under every offset.
const fn guest_deadline(physical: u64, offset: u64) -> u64 {
    if physical == 0 {
        0
    } else {
        physical.wrapping_add(offset)
    }
}

/// Translates a guest deadline into the timestamp domain physical hardware
/// compares against.
const fn physical_deadline(guest: u64, offset: u64) -> u64 {
    if guest == 0 {
        0
    } else {
        let physical = guest.wrapping_sub(offset);
        if physical == 0 { 1 } else { physical }
    }
}

/// Which bits of a register's low half a guest in x2APIC may set.
///
/// Every one of these is `RsvdZ`, so this is the mask a write is judged against
/// rather than masked with. Where the older face silently drops what software
/// may not set, this face has to fault — and faulting requires knowing exactly
/// which bits those are, per register, rather than only checking the upper
/// half.
fn writable(vlapic: &Vlapic, register: Register) -> u32 {
    match register {
        // Acknowledging is not a value and neither is re-arming the error
        // register. The architecture spells both as a write of zero, so every
        // bit of them is reserved.
        Register::END_OF_INTERRUPT | Register::ERROR_STATUS => 0,
        // Only the priority byte; the rest of the register is reserved.
        Register::TASK_PRIORITY => TASK_PRIORITY,
        // The vector and the bit that software-enables the controller.
        Register::SPURIOUS => SPURIOUS,
        // A vector, and nothing else: the delivery mode is fixed, the
        // destination is this processor, and there is no shorthand to name.
        Register::SELF_IPI => VECTOR,
        // Three bits that are not adjacent — the middle one is reserved.
        Register::TIMER_DIVIDE => TIMER_DIVIDE,
        Register::TIMER_INITIAL_COUNT => u32::MAX,
        other => Entry::of(other).map_or(u32::MAX, |entry| entry.writable(vlapic.model())),
    }
}

/// The register an index names, if the guest may name it at all.
fn addressable(vlapic: &Vlapic, index: u32) -> Result<Register, Fault> {
    // Reaching the controller's registers at all is a fault outside x2APIC:
    // the indices are reserved until the guest has enabled the mode that
    // assigns them.
    if vlapic.mode() != Mode::X2Apic {
        return Err(Fault::NotX2Apic);
    }
    let register = Register::from_msr(index).ok_or(Fault::NoSuchRegister)?;
    match Access::of(register, vlapic.mode(), vlapic.model()) {
        Access::Absent => Err(Fault::NoSuchRegister),
        _ => Ok(register),
    }
}

/// Only the priority byte of the task priority register holds anything.
const TASK_PRIORITY: u32 = 0xFF;

/// The spurious vector register's vector and its software-enable bit. Focus
/// checking and end-of-interrupt broadcast suppression are both refused, and
/// the version register reports the second unsupported.
const SPURIOUS: u32 = 0x1FF;

/// A vector is the low eight bits of whatever register carries one.
const VECTOR: u32 = 0xFF;

/// The timer's divide configuration: three bits with a reserved one between
/// them.
const TIMER_DIVIDE: u32 = 0b1011;

/// The vector an eight-bit field of a wider value names.
#[expect(
    clippy::cast_possible_truncation,
    reason = "a vector is the low eight bits of the register it is written in"
)]
const fn vector_of(value: u32) -> u8 {
    value as u8
}

/// Why an access through this face is a general protection fault.
///
/// Every variant is the same exception as far as the guest is concerned. They
/// are kept apart because which one happened is worth logging: a guest reaching
/// the controller's registers in the wrong mode is doing something quite
/// different from one writing a reserved bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fault {
    /// The index names no register the architecture assigns.
    NoSuchRegister,
    /// The controller is not in x2APIC mode, so its registers have no indices.
    NotX2Apic,
    /// The register may not be written.
    ReadOnly,
    /// The register may not be read.
    WriteOnly,
    /// A bit the register reserves was written non-zero.
    Reserved,
    /// The base register refused the write.
    Base(BaseFault),
}

#[cfg(test)]
mod tests {
    use super::{guest_deadline, physical_deadline};

    #[test]
    fn deadline_translation_preserves_disarmed_timers() {
        assert_eq!(physical_deadline(0, 0x1234), 0);
        assert_eq!(guest_deadline(0, 0x1234), 0);
    }

    #[test]
    fn deadline_translation_round_trips_with_wrapping_offsets() {
        for (deadline, offset) in [
            (1, 0),
            (0x1234_5678_9ABC_DEF0, 0x1111_2222_3333_4444),
            (u64::MAX, u64::MAX - 7),
        ] {
            assert_eq!(
                guest_deadline(physical_deadline(deadline, offset), offset),
                deadline
            );
        }
    }

    #[test]
    fn an_armed_guest_deadline_never_becomes_the_physical_disarm_value() {
        assert_eq!(physical_deadline(0x1234, 0x1234), 1);
    }
}
