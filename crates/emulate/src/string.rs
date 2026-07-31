//! The moves that repeat.
//!
//! A guest copying to or from a device does not do it one value at a time. It
//! points the index registers at the two ends, puts a count in the count
//! register, and executes one instruction that moves the lot. Emulating that
//! one repetition per exit would turn a four-kilobyte copy into a thousand
//! world switches.
//!
//! So repetitions are performed in a batch, and the batch stops at whichever
//! comes first: the count reaching zero, either end reaching a page boundary,
//! or a fixed ceiling.
//!
//! # Why stopping early is safe, and why the page boundary is where
//!
//! A repeated string instruction is restartable by design. The architecture
//! keeps the whole of its progress in the index and count registers, so a
//! processor interrupted part way through one resumes by executing the same
//! instruction again with those registers where they were left.
//!
//! A batch that stops early does exactly that: the registers are updated, the
//! instruction pointer is *not* advanced, and the guest re-executes the
//! instruction. It faults again on the next page and the next batch carries on.
//! The guest can take an interrupt between the two, which is what keeps a long
//! copy from holding a processor inside one exit handler.
//!
//! The page boundary is the natural place to stop because it is the first point
//! at which nothing can be assumed any more: the next page may translate
//! somewhere unrelated, may be answered for by a different device or by none,
//! and may not be described at all.

use iced_x86::{Instruction, OpKind};
use memory::Linear;
use vcpu::Vcpu;

use crate::{
    EmulateError, Outcome, gpr,
    mmio::Mmio,
    operand::{self, Place},
    value::{Data, Width},
};

/// Which operand of a string instruction is the destination.
const DESTINATION: u32 = 0;
/// Which is the source.
const SOURCE: u32 = 1;

/// Bytes in the smallest page, which is where a batch stops and asks again.
const PAGE: u64 = 4096;

/// The most repetitions one exit performs.
///
/// A page of the narrowest repetition there is, which is more than can happen
/// before an end crosses a page boundary anyway. It is a backstop rather than a
/// target: it exists so that no guest can make one exit take unbounded time,
/// whatever it puts in the count register.
const CEILING: u64 = PAGE;

/// Bit of the guest's flags that says the index registers count downwards.
const DIRECTION: u64 = 1 << 10;

/// Performs as many repetitions as can be done in this exit.
///
/// # Errors
///
/// [`EmulateError::Width`] if the instruction has no implicit string operand to
/// take an address size from, and whatever reading or writing either end
/// reports.
pub(crate) fn perform(
    mmio: &Mmio,
    vcpu: &mut Vcpu,
    guest: Linear<'_>,
    instruction: &Instruction,
    width: Width,
) -> Result<Outcome, EmulateError> {
    let rip = vcpu.save().rip;
    if instruction.has_repne_prefix() {
        // The architecture gives this prefix a meaning on the comparing string
        // instructions and none on these. Performing it as though it were the
        // other prefix would repeat something the guest did not ask to repeat.
        return Err(EmulateError::Operand { rip });
    }
    let repeated = instruction.has_rep_prefix();
    let backwards = vcpu.save().rflags & DIRECTION != 0;
    let counting = address_width(instruction).ok_or(EmulateError::Width { rip })?;

    for _ in 0..CEILING {
        if repeated && left(vcpu, counting) == 0 {
            return Ok(Outcome::Stepped);
        }

        // Both ends are worked out again each time round rather than stepped,
        // because their addresses come out of the index registers and those are
        // what the previous repetition moved.
        let from = operand::place(mmio, vcpu, guest, instruction, SOURCE)?;
        let to = operand::place(mmio, vcpu, guest, instruction, DESTINATION)?;
        let value = operand::load(mmio, vcpu, guest, from, width)?;
        operand::store(mmio, vcpu, guest, to, value)?;
        advance(vcpu, instruction, width, backwards, counting)?;

        if !repeated {
            return Ok(Outcome::Stepped);
        }
        decrement(vcpu, counting)?;
        if left(vcpu, counting) == 0 {
            return Ok(Outcome::Stepped);
        }
        if leaves_page(&[from, to], width, backwards) {
            return Ok(Outcome::Repeating);
        }
    }
    Ok(Outcome::Repeating)
}

/// Whether the next repetition would touch a page other than this one did.
fn leaves_page(ends: &[Place], width: Width, backwards: bool) -> bool {
    let Ok(step) = u64::try_from(width.bytes()) else {
        return true;
    };
    ends.iter().filter_map(|place| place.linear()).any(|at| {
        let next = if backwards {
            at.wrapping_sub(step)
        } else {
            at.wrapping_add(step)
        };
        at / PAGE != next / PAGE
    })
}

/// Moves each index register the instruction walks on by one repetition.
fn advance(
    vcpu: &mut Vcpu,
    instruction: &Instruction,
    width: Width,
    backwards: bool,
    counting: Width,
) -> Result<(), EmulateError> {
    let by = i64::try_from(width.bytes()).unwrap_or_default();
    let by = if backwards { -by } else { by };
    for operand in 0..instruction.op_count() {
        if let Some(register) = operand::index(instruction.op_kind(operand)) {
            step(vcpu, register, by, counting)?;
        }
    }
    Ok(())
}

/// Takes one off the count register.
fn decrement(vcpu: &mut Vcpu, counting: Width) -> Result<(), EmulateError> {
    step(vcpu, gpr::RCX, -1, counting)
}

/// What is left of the count register, at the width the address size gives it.
fn left(vcpu: &Vcpu, counting: Width) -> u64 {
    vcpu.gpr(gpr::RCX) & counting.mask()
}

/// Adds to a register, at the width the address size gives it.
///
/// The width is not decoration. An instruction using 32-bit addresses walks
/// `ECX`, `ESI` and `EDI`, and a write to one of those clears the upper half of
/// the register rather than carrying into it — which is the rule
/// [`gpr::merge`] already knows, applied here rather than restated.
fn step(vcpu: &mut Vcpu, register: u8, by: i64, counting: Width) -> Result<(), EmulateError> {
    let whole = vcpu.gpr(register);
    let value = (whole & counting.mask()).wrapping_add_signed(by);
    vcpu.set_gpr(
        register,
        gpr::merge(whole, Data::from_u64(value, counting))?,
    );
    Ok(())
}

/// How wide the addresses this instruction walks are, out of the first implicit
/// operand that says so.
///
/// All of an instruction's implicit string operands use the same address size,
/// so the first one to name a width names all of them.
fn address_width(instruction: &Instruction) -> Option<Width> {
    (0..instruction.op_count()).find_map(|operand| match instruction.op_kind(operand) {
        OpKind::MemorySegSI | OpKind::MemoryESDI => Some(Width::Word),
        OpKind::MemorySegESI | OpKind::MemoryESEDI => Some(Width::Long),
        OpKind::MemorySegRSI | OpKind::MemoryESRDI => Some(Width::Quad),
        _ => None,
    })
}
