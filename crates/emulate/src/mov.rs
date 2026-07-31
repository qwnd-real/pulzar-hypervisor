//! The moves.
//!
//! Every instruction here does the same thing: take a value from one place, put
//! it in another, change nothing else. No flags, no arithmetic, no
//! read-modify-write. That is what makes this family the one worth emulating
//! first, and it is why a single function performs nearly all of them.
//!
//! The three that differ do so only in what happens between reading and
//! writing: the two widening moves make the value wider, one of them copying
//! the sign bit and the other zeroes.
//!
//! # Everything outside the family is refused by name
//!
//! An instruction this crate does not perform is never approximated. Emulating
//! `or [rax], 1` as though it were a move would put the wrong value in a device
//! register and leave the guest's flags describing an operation that never
//! happened, and the guest would carry on from that quite happily and wrongly.
//! So an unrecognized instruction stops the emulation and names itself, and
//! whoever asked decides what to do about a guest that does something not
//! anticipated.

use iced_x86::{Instruction, Mnemonic};
use memory::Linear;
use vcpu::Vcpu;

use crate::{EmulateError, mmio::Mmio, operand, value::Width};

/// Which operand of a two-operand instruction is the destination.
const DESTINATION: u32 = 0;
/// Which is the source.
const SOURCE: u32 = 1;

/// Performs a move on the guest's behalf.
///
/// # Errors
///
/// [`EmulateError::Unsupported`] for an instruction outside the family, and
/// whatever reading or writing either end of the move reports.
pub(crate) fn perform(
    mmio: &Mmio,
    vcpu: &mut Vcpu,
    guest: Linear<'_>,
    instruction: &Instruction,
) -> Result<(), EmulateError> {
    let rip = vcpu.save().rip;
    let widen = match instruction.mnemonic() {
        // Plain moves, of every width and encoding: the general-purpose forms,
        // the vector ones aligned and unaligned, the non-temporal ones, and the
        // ones that move between a vector register and a general-purpose one.
        // They differ in what the processor does about alignment and caching and
        // not at all in what they move, and neither of those survives being
        // emulated in any case.
        Mnemonic::Mov
        | Mnemonic::Movaps
        | Mnemonic::Movapd
        | Mnemonic::Movups
        | Mnemonic::Movupd
        | Mnemonic::Movdqa
        | Mnemonic::Movdqu
        | Mnemonic::Movntdq
        | Mnemonic::Movntps
        | Mnemonic::Movntpd
        | Mnemonic::Movnti
        | Mnemonic::Movd
        | Mnemonic::Movq
        | Mnemonic::Movss
        | Mnemonic::Movsd => None,
        Mnemonic::Movzx => Some(false),
        Mnemonic::Movsx | Mnemonic::Movsxd => Some(true),
        mnemonic => {
            return Err(EmulateError::Unsupported {
                rip,
                mnemonic: name(mnemonic),
            });
        }
    };
    if instruction.op_count() != 2 {
        return Err(EmulateError::Operand { rip });
    }

    let (from, to) = (
        operand::place(mmio, vcpu, guest, instruction, SOURCE)?,
        operand::place(mmio, vcpu, guest, instruction, DESTINATION)?,
    );
    let source = operand::width(instruction, SOURCE).ok_or(EmulateError::Width { rip })?;
    let destination =
        operand::width(instruction, DESTINATION).ok_or(EmulateError::Width { rip })?;

    let value = operand::load(mmio, vcpu, guest, from, source)?;
    let value = match widen {
        Some(signed) => value.extended(destination, signed),
        // A plain move writes what it read, at the width it read it. Where the
        // two ends disagree — a four-byte move naming a vector register — the
        // narrower is the move, and `width` has already answered with it.
        None => value,
    };
    operand::store(mmio, vcpu, guest, to, value)
}

/// Whether this instruction is a move of a whole region rather than a value —
/// the string forms, which repeat.
pub(crate) fn repeats(instruction: &Instruction) -> bool {
    matches!(
        instruction.mnemonic(),
        Mnemonic::Movsb
            | Mnemonic::Movsw
            | Mnemonic::Movsd
            | Mnemonic::Movsq
            | Mnemonic::Stosb
            | Mnemonic::Stosw
            | Mnemonic::Stosd
            | Mnemonic::Stosq
            | Mnemonic::Lodsb
            | Mnemonic::Lodsw
            | Mnemonic::Lodsd
            | Mnemonic::Lodsq
    ) && instruction
        .op_kinds()
        .any(|kind| operand::index(kind).is_some())
}

/// How wide one repetition of a string instruction moves.
pub(crate) fn repetition(instruction: &Instruction) -> Option<Width> {
    operand::memory_width(instruction)
}

/// What to call an instruction this crate will not perform.
///
/// The mnemonic and not the whole instruction, because the mnemonic is what
/// decides: an operand that is unusual is still a move, and a mnemonic that is
/// not a move is not one however its operands read.
fn name(mnemonic: Mnemonic) -> &'static str {
    /// One entry per family this crate has been asked about and declined, so
    /// that a log says what the guest did rather than a number.
    const KNOWN: [(Mnemonic, &str); 12] = [
        (Mnemonic::And, "and"),
        (Mnemonic::Or, "or"),
        (Mnemonic::Xor, "xor"),
        (Mnemonic::Add, "add"),
        (Mnemonic::Sub, "sub"),
        (Mnemonic::Test, "test"),
        (Mnemonic::Cmp, "cmp"),
        (Mnemonic::Inc, "inc"),
        (Mnemonic::Dec, "dec"),
        (Mnemonic::Xchg, "xchg"),
        (Mnemonic::Cmpxchg, "cmpxchg"),
        (Mnemonic::Xadd, "xadd"),
    ];
    KNOWN
        .iter()
        .find_map(|(known, name)| (*known == mnemonic).then_some(*name))
        .unwrap_or("an instruction that is not a move")
}
