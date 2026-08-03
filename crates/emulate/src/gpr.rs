//! The guest's general-purpose registers, as a decoded instruction names them.
//!
//! A [`Cpu`] already speaks the architecture's four-bit register numbers and
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
//!
//! # A width is checked rather than inferred
//!
//! Every entry point here takes the width it is asked for and refuses a value
//! that is not exactly that wide. Merging by whatever width the *value* happens
//! to carry would make a decoding mistake somewhere above silently change the
//! architectural effect of the instruction: a byte value reaching a dword
//! destination would clear the upper half of a register the guest expected to
//! keep, and nothing would report it.

use iced_x86::Register;

use crate::{
    EmulateError,
    machine::Cpu,
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
pub(crate) fn read(cpu: &impl Cpu, register: Register) -> Result<Data, EmulateError> {
    let (number, width) = about(register)?;
    let whole = cpu.gpr(number);
    Ok(Data::from_u64(
        if high(register) { whole >> HIGH } else { whole },
        width,
    ))
}

/// Puts a value in the register an instruction named.
///
/// The value must be exactly as wide as the register: a mismatch is a decoding
/// fault above rather than something to resolve by merging, and it is reported
/// with both widths so that whoever reads the log can see which end was wrong.
///
/// # Errors
///
/// As [`read`], and [`EmulateError::WidthMismatch`] if the value is not the
/// width the register is.
pub(crate) fn write(
    cpu: &mut impl Cpu,
    register: Register,
    value: Data,
) -> Result<(), EmulateError> {
    let (number, width) = about(register)?;
    if value.width() != width {
        return Err(EmulateError::WidthMismatch {
            register: name(register),
            wanted: width,
            got: value.width(),
        });
    }
    let whole = cpu.gpr(number);
    let merged = if high(register) {
        let mask = Width::Byte.mask() << HIGH;
        (whole & !mask) | ((value.as_u64() & Width::Byte.mask()) << HIGH)
    } else {
        merge(whole, value)?
    };
    cpu.set_gpr(number, merged);
    Ok(())
}

/// What a register becomes when that value is written to the low end of it.
///
/// Separate from [`write`] because the string instructions update their index
/// and count registers by the same rule while naming them by number rather than
/// as a decoded operand — and the rule is exactly the sort of thing that drifts
/// when it is written twice.
///
/// # Errors
///
/// [`EmulateError::WidthMismatch`] if the value is sixteen bytes wide, which no
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
        Width::Vector => {
            return Err(EmulateError::WidthMismatch {
                register: "a general-purpose register",
                wanted: Width::Quad,
                got: Width::Vector,
            });
        }
    })
}

/// How wide the register an instruction named is.
///
/// # Errors
///
/// As [`read`].
pub(crate) fn width(register: Register) -> Result<Width, EmulateError> {
    about(register).map(|(_, width)| width)
}

/// The number the architecture encodes this register as, and how wide it is.
fn about(register: Register) -> Result<(u8, Width), EmulateError> {
    if !register.is_gpr() {
        return Err(EmulateError::Register {
            register: name(register),
            bytes: register.size(),
        });
    }
    let width = Width::from_bytes(register.size()).ok_or(EmulateError::Register {
        register: name(register),
        bytes: register.size(),
    })?;
    // The full register a part belongs to is the 64-bit one, and its position
    // within the sixty-four-bit registers is the number the architecture encodes
    // — which is the same number a control block reports an operand as.
    let number =
        u8::try_from(register.full_register().number()).map_err(|_| EmulateError::Register {
            register: name(register),
            bytes: register.size(),
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

/// What to call a register in a diagnostic.
///
/// iced-x86's own name would need its formatter, which this image does not
/// build — the decoder is the only part of that crate in the hypervisor. So the
/// registers this crate can name are named, and anything else is described by
/// the class it belongs to, which is the part that matters when the complaint
/// is that it is not a general-purpose register at all.
pub(crate) fn name(register: Register) -> &'static str {
    if register.is_gpr64() {
        return "a 64-bit register";
    }
    if register.is_gpr32() {
        return "a 32-bit register";
    }
    if register.is_gpr16() {
        return "a 16-bit register";
    }
    if register.is_gpr8() {
        return "an 8-bit register";
    }
    if register.is_xmm() || register.is_ymm() || register.is_zmm() {
        return "a vector register";
    }
    if register.is_segment_register() {
        return "a segment register";
    }
    if register.is_cr() {
        return "a control register";
    }
    if register.is_dr() {
        return "a debug register";
    }
    if register.is_tr() {
        return "a test register";
    }
    if register.is_mm() {
        return "an MMX register";
    }
    if register.is_k() {
        return "a mask register";
    }
    if register.is_bnd() {
        return "a bound register";
    }
    if register.is_tmm() {
        return "a tile register";
    }
    if register.is_ip() {
        return "the instruction pointer";
    }
    "a register no move names"
}

/// The number the architecture encodes the count register as, which the
/// repeated string instructions decrement.
pub(crate) const RCX: u8 = 1;
/// The number the architecture encodes the source index as.
pub(crate) const RSI: u8 = 6;
/// The number the architecture encodes the destination index as.
pub(crate) const RDI: u8 = 7;

#[cfg(test)]
mod tests {
    use iced_x86::Register;

    use super::{RCX, RDI, RSI, merge, read, width, write};
    use crate::{
        EmulateError,
        machine::{Cpu, tests::Machine},
        value::{Data, Width},
    };

    /// A pattern whose every byte differs, so that a write reaching the wrong
    /// half of the wrong register is visible in the value rather than
    /// plausible.
    const SEED: u64 = 0x1122_3344_5566_7788;

    /// One machine with every register holding [`SEED`].
    fn seeded() -> Machine {
        let mut machine = Machine::default();
        for number in 0..16 {
            machine.set_gpr(number, SEED);
        }
        machine
    }

    #[test]
    fn a_register_reads_as_wide_as_it_is() {
        let machine = seeded();
        for (register, expected) in [
            (Register::AL, Data::from_u64(0x88, Width::Byte)),
            (Register::AX, Data::from_u64(0x7788, Width::Word)),
            (Register::EAX, Data::from_u64(0x5566_7788, Width::Long)),
            (Register::RAX, Data::from_u64(SEED, Width::Quad)),
        ] {
            assert_eq!(read(&machine, register).expect("a GPR reads"), expected);
        }
    }

    #[test]
    fn the_four_high_bytes_are_bits_fifteen_to_eight() {
        let machine = seeded();
        for register in [Register::AH, Register::CH, Register::DH, Register::BH] {
            assert_eq!(
                read(&machine, register).expect("a high byte reads"),
                Data::from_u64(0x77, Width::Byte),
                "{register:?} must be bits 15:8 rather than the low byte"
            );
        }
    }

    #[test]
    fn writing_a_high_byte_leaves_every_other_bit_alone() {
        let mut machine = seeded();
        write(
            &mut machine,
            Register::AH,
            Data::from_u64(0xFF, Width::Byte),
        )
        .expect("a high byte is writable");
        assert_eq!(machine.gpr(0), 0x1122_3344_5566_FF88);
    }

    #[test]
    fn the_same_four_encodings_with_a_prefix_are_low_bytes_of_other_registers() {
        // SPL, BPL, SIL and DIL are the bottoms of registers 4 through 7, and
        // treating them as the high-byte family would write bits 15:8 of the
        // wrong four registers entirely.
        let mut machine = seeded();
        for (register, number) in [
            (Register::SPL, 4),
            (Register::BPL, 5),
            (Register::SIL, 6),
            (Register::DIL, 7),
        ] {
            write(&mut machine, register, Data::from_u64(0xFF, Width::Byte))
                .expect("a REX low byte is writable");
            assert_eq!(
                machine.gpr(number),
                0x1122_3344_5566_77FF,
                "{register:?} must be the low byte of register {number}"
            );
        }
    }

    #[test]
    fn only_a_dword_write_clears_the_upper_half() {
        let mut machine = seeded();

        write(
            &mut machine,
            Register::AL,
            Data::from_u64(0xFF, Width::Byte),
        )
        .expect("byte");
        assert_eq!(
            machine.gpr(0),
            0x1122_3344_5566_77FF,
            "a byte preserves 63:8"
        );

        write(
            &mut machine,
            Register::AX,
            Data::from_u64(0xEEEE, Width::Word),
        )
        .expect("word");
        assert_eq!(
            machine.gpr(0),
            0x1122_3344_5566_EEEE,
            "a word preserves 63:16"
        );

        write(
            &mut machine,
            Register::EAX,
            Data::from_u64(0xDDDD_DDDD, Width::Long),
        )
        .expect("dword");
        assert_eq!(machine.gpr(0), 0xDDDD_DDDD, "a dword clears 63:32");

        write(
            &mut machine,
            Register::RAX,
            Data::from_u64(u64::MAX, Width::Quad),
        )
        .expect("qword");
        assert_eq!(machine.gpr(0), u64::MAX, "a qword replaces the register");
    }

    #[test]
    fn the_rule_holds_for_the_extended_registers_too() {
        let mut machine = seeded();
        write(&mut machine, Register::R15D, Data::from_u64(1, Width::Long)).expect("dword");
        assert_eq!(
            machine.gpr(15),
            1,
            "R15D must clear the upper half like EAX"
        );

        write(
            &mut machine,
            Register::R8W,
            Data::from_u64(0xABCD, Width::Word),
        )
        .expect("word");
        assert_eq!(
            machine.gpr(8),
            0x1122_3344_5566_ABCD,
            "R8W must preserve 63:16"
        );

        write(
            &mut machine,
            Register::R9L,
            Data::from_u64(0xEF, Width::Byte),
        )
        .expect("byte");
        assert_eq!(
            machine.gpr(9),
            0x1122_3344_5566_77EF,
            "R9L must preserve 63:8"
        );
    }

    #[test]
    fn the_stack_pointer_obeys_the_same_rule_where_it_is_stored() {
        // RSP lives in the state-save area rather than the register block, and
        // the merge rule must not depend on which side of that split it is on.
        let mut machine = seeded();
        write(
            &mut machine,
            Register::ESP,
            Data::from_u64(0x1234, Width::Long),
        )
        .expect("dword");
        assert_eq!(machine.gpr(4), 0x1234);
        assert_eq!(
            machine.save().rsp,
            0x1234,
            "the write must reach the save area"
        );
    }

    #[test]
    fn a_value_of_the_wrong_width_is_refused_and_changes_nothing() {
        let mut machine = seeded();
        for (register, wanted, got) in [
            (Register::AL, Width::Byte, Width::Long),
            (Register::AX, Width::Word, Width::Quad),
            (Register::EAX, Width::Long, Width::Byte),
            (Register::RAX, Width::Quad, Width::Word),
            (Register::AH, Width::Byte, Width::Vector),
        ] {
            let error = write(&mut machine, register, Data::from_u64(1, got))
                .expect_err("a mismatched width must be refused");
            assert!(
                matches!(
                    error,
                    EmulateError::WidthMismatch { wanted: w, got: g, .. } if w == wanted && g == got
                ),
                "{register:?} wanted {wanted:?} and got {got:?}, reported as {error:?}"
            );
            assert_eq!(machine.gpr(0), SEED, "a refused write must change nothing");
        }
    }

    #[test]
    fn a_register_that_is_not_general_purpose_is_refused_by_class() {
        let machine = seeded();
        for (register, expected) in [
            (Register::XMM0, "a vector register"),
            (Register::YMM3, "a vector register"),
            (Register::ZMM31, "a vector register"),
            (Register::MM6, "an MMX register"),
            (Register::CS, "a segment register"),
            (Register::CR8, "a control register"),
            (Register::DR7, "a debug register"),
            (Register::K3, "a mask register"),
            (Register::BND1, "a bound register"),
            (Register::TMM7, "a tile register"),
            (Register::RIP, "the instruction pointer"),
            (Register::EIP, "the instruction pointer"),
            (Register::None, "a register no move names"),
        ] {
            let error = read(&machine, register).expect_err("not a GPR");
            assert!(
                matches!(error, EmulateError::Register { register: named, .. } if named == expected),
                "{register:?} should be described as {expected}, reported as {error:?}"
            );
        }
    }

    #[test]
    fn a_register_reports_its_own_width() {
        for (register, expected) in [
            (Register::AL, Width::Byte),
            (Register::AH, Width::Byte),
            (Register::DIL, Width::Byte),
            (Register::AX, Width::Word),
            (Register::R8W, Width::Word),
            (Register::EAX, Width::Long),
            (Register::R15D, Width::Long),
            (Register::RAX, Width::Quad),
            (Register::RSP, Width::Quad),
            (Register::R15, Width::Quad),
        ] {
            assert_eq!(width(register).expect("a GPR has a width"), expected);
        }
    }

    #[test]
    fn merging_a_vector_into_a_general_purpose_register_is_refused() {
        let error = merge(SEED, Data::vector_from([0xFF; 16])).expect_err("no GPR is that wide");
        assert!(matches!(
            error,
            EmulateError::WidthMismatch {
                got: Width::Vector,
                ..
            }
        ));
    }

    #[test]
    fn the_index_and_count_registers_are_the_numbers_the_architecture_encodes() {
        // The string instructions name these by number rather than as decoded
        // operands, so the numbers are asserted against the registers a decoder
        // reports for the same encodings.
        let mut machine = Machine::default();
        machine.set_gpr(RCX, 0xC);
        machine.set_gpr(RSI, 0x5);
        machine.set_gpr(RDI, 0xD);
        assert_eq!(read(&machine, Register::RCX).expect("rcx").as_u64(), 0xC);
        assert_eq!(read(&machine, Register::RSI).expect("rsi").as_u64(), 0x5);
        assert_eq!(read(&machine, Register::RDI).expect("rdi").as_u64(), 0xD);
    }
}
