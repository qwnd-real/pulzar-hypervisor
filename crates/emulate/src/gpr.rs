//! The guest's general-purpose registers, as a decoded instruction names them.
//!
//! A [`Vcpu`] already speaks the architecture's four-bit register numbers and
//! already knows that two of the sixteen live in the control block rather than
//! beside the other fourteen. So there is nothing to do here but turn what a
//! decoder calls a register into that number, and apply the one rule about
//! writing a register that is narrower than sixty-four bits.
//!
//! # The rule, which is not uniform
//!
//! A 64-bit write replaces the register. A **32-bit write clears the upper
//! half** — it is the only narrow write that does. A 16-bit or 8-bit write
//! leaves everything above it exactly as it was.
//!
//! # The four registers that are not where they look like they are
//!
//! `AH`, `CH`, `DH` and `BH` are bits 15:8 of the accumulator, count, data and
//! base registers, and they are the only registers that are not the bottom of
//! something. They are also only encodable without a REX prefix — with one, the
//! same four encodings mean `SPL`, `BPL`, `SIL` and `DIL`, which *are* the
//! bottoms of the stack pointer, frame pointer and the two index registers. A
//! decoder distinguishes all eight; anything that treats them as one family
//! writes the wrong half of the wrong register.

use iced_x86::Register;
use vcpu::Vcpu;

use crate::{
    EmulateError,
    value::{Data, Width},
};

/// Where in a register the four high-byte registers sit.
const HIGH: u32 = 8;

/// What the guest has in the register an instruction named.
///
/// # Errors
///
/// [`EmulateError::Register`] if the register is not a general-purpose one, or
/// is one of a width the architecture has no move for.
pub(crate) fn read(vcpu: &Vcpu, register: Register) -> Result<Data, EmulateError> {
    let (number, width) = about(register)?;
    let whole = vcpu.gpr(number);
    Ok(Data::from_u64(
        if high(register) { whole >> HIGH } else { whole },
        width,
    ))
}

/// Puts a value in the register an instruction named.
///
/// # Errors
///
/// As [`read`], and [`EmulateError::Register`] for a value too wide to fit a
/// general-purpose register at all.
pub(crate) fn write(vcpu: &mut Vcpu, register: Register, value: Data) -> Result<(), EmulateError> {
    let (number, _) = about(register)?;
    let whole = vcpu.gpr(number);
    let merged = if high(register) {
        let mask = Width::Byte.mask() << HIGH;
        (whole & !mask) | ((value.as_u64() & Width::Byte.mask()) << HIGH)
    } else {
        merge(whole, value)?
    };
    vcpu.set_gpr(number, merged);
    Ok(())
}

/// What a register of this width becomes when that value is written to the low
/// end of it.
///
/// Separate from [`write`] because the string instructions update their index
/// and count registers by the same rule while naming them by number rather
/// than as a decoded operand — and the rule is exactly the sort of thing that
/// drifts when it is written twice.
///
/// # Errors
///
/// [`EmulateError::Register`] if the value is sixteen bytes wide, which no
/// general-purpose register is.
pub(crate) fn merge(whole: u64, value: Data) -> Result<u64, EmulateError> {
    Ok(match value.width() {
        // The one width that does not preserve what is above it. Everything a
        // 32-bit instruction writes is zero-extended to the full register, and
        // a great deal of compiled code depends on it.
        Width::Long => value.as_u64() & Width::Long.mask(),
        Width::Quad => value.as_u64(),
        Width::Byte | Width::Word => {
            let mask = value.width().mask();
            (whole & !mask) | (value.as_u64() & mask)
        }
        Width::Vector => return Err(EmulateError::Register { width: 16 }),
    })
}

/// The number the architecture encodes this register as, and how wide it is.
fn about(register: Register) -> Result<(u8, Width), EmulateError> {
    let width = Width::from_bytes(register.size()).ok_or(EmulateError::Register {
        width: register.size(),
    })?;
    if !register.is_gpr() {
        return Err(EmulateError::Register {
            width: register.size(),
        });
    }
    // The full register a part belongs to is the 64-bit one, and its position
    // within the sixty-four-bit registers is the number the architecture encodes
    // — which is the same number a control block reports an operand as.
    let number =
        u8::try_from(register.full_register().number()).map_err(|_| EmulateError::Register {
            width: register.size(),
        })?;
    Ok((number, width))
}

/// Whether this register is bits 15:8 of one rather than the bottom of one.
fn high(register: Register) -> bool {
    matches!(
        register,
        Register::AH | Register::CH | Register::DH | Register::BH
    )
}

/// The number the architecture encodes the count register as, which the
/// repeated string instructions decrement.
pub(crate) const RCX: u8 = 1;
/// The number the architecture encodes the source index as.
pub(crate) const RSI: u8 = 6;
/// The number the architecture encodes the destination index as.
pub(crate) const RDI: u8 = 7;
