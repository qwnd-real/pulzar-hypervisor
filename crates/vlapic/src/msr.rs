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
//! There is no page to trap, so nothing calls into here through
//! [`emulate`](emulate). A guest in x2APIC mode reaches its controller with
//! `RDMSR` and `WRMSR`, which are intercepted through the permission map, and
//! the exit handler calls [`read`] and [`write`] directly.

use descriptors::Vector;

use crate::{
    access::{self, Written},
    base::ApicBase,
    register::{Access, Register, X2APIC_BASE_MSR, X2APIC_LAST_MSR},
    state::Vlapic,
};

/// The timestamp counter deadline the timer fires at, which the architecture
/// puts outside the controller's own range.
pub const TSC_DEADLINE_MSR: u32 = 0x6E0;

/// Whether an index is one this crate answers for.
///
/// The whole of the controller's reserved range counts, not merely the indices
/// that name a register: reaching an unassigned one is a fault the guest has to
/// be given, and it cannot be given one by code that never sees the access.
#[must_use]
pub const fn claims(index: u32) -> bool {
    matches!(index, X2APIC_BASE_MSR..=X2APIC_LAST_MSR)
        || index == ApicBase::MSR
        || index == TSC_DEADLINE_MSR
}

/// What the guest reads.
///
/// # Errors
///
/// [`Fault`] if the index names nothing, names a register that does not exist
/// in the mode the guest is in, or names one that may not be read — each of
/// which the guest takes as a general protection fault.
pub(crate) fn read(vlapic: &Vlapic, index: u32) -> Result<u64, Fault> {
    if index == ApicBase::MSR {
        return Ok(vlapic.base().bits());
    }
    if index == TSC_DEADLINE_MSR {
        return Ok(vlapic.timer_deadline());
    }
    let register = addressable(vlapic, index)?;
    if !matches!(
        Access::of(register, vlapic.mode()),
        Access::ReadOnly | Access::ReadWrite
    ) {
        return Err(Fault::WriteOnly);
    }
    // The one register that is really 64 bits wide. Every other read has a zero
    // upper half, which is what `RsvdZ` requires.
    if register == Register::COMMAND_LOW {
        return Ok(vlapic.command().bits());
    }
    Ok(u64::from(access::read(vlapic, register)))
}

/// What a write does.
///
/// # Errors
///
/// As [`read`], and additionally [`Fault::Reserved`] if a bit the register
/// reserves was written non-zero — which through this face is a fault rather
/// than something quietly dropped.
pub(crate) fn write(vlapic: &Vlapic, index: u32, value: u64) -> Result<Written, Fault> {
    if index == ApicBase::MSR {
        return vlapic
            .write_base(value)
            .map(|_| Written::ModeChanged)
            .map_err(Fault::Base);
    }
    if index == TSC_DEADLINE_MSR {
        // Hardware ignores this write outside deadline mode rather than
        // faulting, so a refusal here is not an error.
        vlapic.set_timer_deadline(value);
        return Ok(Written::Timer);
    }
    let register = addressable(vlapic, index)?;
    if !matches!(
        Access::of(register, vlapic.mode()),
        Access::ReadWrite | Access::WriteOnly
    ) {
        return Err(Fault::ReadOnly);
    }
    if register == Register::COMMAND_LOW {
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
    // Two registers take a value at all only if it is zero, and fault on
    // anything else. Acknowledging is not a value and neither is re-arming the
    // error register; the architecture spells both as a write of zero.
    if matches!(
        register,
        Register::END_OF_INTERRUPT | Register::ERROR_STATUS
    ) && narrow != 0
    {
        return Err(Fault::Reserved);
    }
    if register == Register::SELF_IPI {
        return Ok(Written::SelfIpi(Vector::new(vector_of(narrow))));
    }
    Ok(access::write(vlapic, register, narrow))
}

/// The register an index names, if the guest may name it at all.
fn addressable(vlapic: &Vlapic, index: u32) -> Result<Register, Fault> {
    // Reaching the controller's registers at all is a fault outside x2APIC:
    // the indices are reserved until the guest has enabled the mode that
    // assigns them.
    if vlapic.mode() != crate::base::Mode::X2Apic {
        return Err(Fault::NotX2Apic);
    }
    let register = Register::from_msr(index).ok_or(Fault::NoSuchRegister)?;
    match Access::of(register, vlapic.mode()) {
        Access::Absent => Err(Fault::NoSuchRegister),
        _ => Ok(register),
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
    Base(crate::base::BaseFault),
}
