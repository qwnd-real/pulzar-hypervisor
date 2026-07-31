//! One end of a move: where it is, how wide it is, and how to get at it.
//!
//! Every instruction this crate emulates moves a value from somewhere to
//! somewhere else, and there are only five somewheres. Four of them the
//! hypervisor already has: a general-purpose register in the control block, a
//! vector register still sitting in the processor, an immediate encoded in the
//! instruction, and the guest's own memory. The fifth is a region something
//! interposes on, which is the only one that involves asking anybody anything.
//!
//! # Nothing here computes an address by hand
//!
//! An effective address is [`Instruction::virtual_address`]'s to work out, and
//! it is handed the guest's registers and segment bases to work it out from.
//! That one call covers instruction-relative operands, scaled indices,
//! address-size truncation, segment bases and every implicit operand the string
//! instructions have. Writing any of that out again here would be a second
//! implementation of the addressing modes, which is a great deal of code whose
//! only possible contribution is to disagree with the first.
//!
//! # A vector register's size is not a move's width
//!
//! Moving four bytes into a vector register still names a sixteen-byte
//! register, so the register's own size answers the wrong question. Where
//! memory is one end of the move — which for anything this crate is called
//! about, it is — the memory operand's size is the width of the move, and that
//! is what is used.

use iced_x86::{Instruction, OpKind, Register};
use memory::{Addressing, Linear, Segment};
use vcpu::Vcpu;
use x86_64::PhysAddr;

use crate::{
    EmulateError, gpr,
    mmio::Mmio,
    value::{Data, Width},
    xmm::Vector,
};

/// One end of a move.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Place {
    /// A general-purpose register.
    Gpr(Register),
    /// A vector register, which is still in the processor.
    Vector(Vector),
    /// A value encoded in the instruction, already widened however its encoding
    /// says. Only ever a source.
    Immediate(u64),
    /// The guest's own memory, at a linear address.
    Memory(u64),
    /// A region something answers for instead of the hardware.
    Device {
        /// Which region.
        index: usize,
        /// How far into it.
        offset: u64,
        /// Where, as the guest thinks of the address.
        gpa: PhysAddr,
        /// Where, as the instruction named it. Kept so that a repeated
        /// instruction can tell when the next repetition leaves the page.
        linear: u64,
    },
}

impl Place {
    /// The linear address this end of the move is at, if it is in memory at
    /// all.
    pub(crate) const fn linear(self) -> Option<u64> {
        match self {
            Self::Memory(linear) | Self::Device { linear, .. } => Some(linear),
            Self::Gpr(_) | Self::Vector(_) | Self::Immediate(_) => None,
        }
    }

    /// Whether this end of the move is a region something answers for.
    pub(crate) const fn interposed(self) -> bool {
        matches!(self, Self::Device { .. })
    }
}

/// Where one operand of an instruction is.
///
/// # Errors
///
/// [`EmulateError::Operand`] for an operand kind no move this crate performs
/// has, [`EmulateError::Address`] if the effective address cannot be computed
/// from the registers the guest stopped with, or [`EmulateError::Memory`] if
/// the guest's own tables do not translate it.
pub(crate) fn place(
    mmio: &Mmio,
    vcpu: &Vcpu,
    guest: Linear<'_>,
    instruction: &Instruction,
    operand: u32,
) -> Result<Place, EmulateError> {
    let rip = vcpu.save().rip;
    let kind = instruction.op_kind(operand);
    if kind == OpKind::Register {
        let register = instruction.op_register(operand);
        return match Vector::new(register) {
            Some(vector) => Ok(Place::Vector(vector)),
            // Not a vector register this crate can reach, so either a
            // general-purpose one or something a move of this family does not
            // have. `gpr` decides which, and says so.
            None => gpr::read(vcpu, register).map(|_| Place::Gpr(register)),
        };
    }
    if immediate(kind) {
        return Ok(Place::Immediate(instruction.immediate(operand)));
    }
    if !in_memory(kind) {
        return Err(EmulateError::Operand { rip });
    }

    let linear = address(instruction, operand, vcpu, guest.addressing())
        .ok_or(EmulateError::Address { rip })?;
    let gpa = guest.translate(linear)?;
    Ok(match mmio.find(gpa) {
        Some((index, offset)) => Place::Device {
            index,
            offset,
            gpa,
            linear,
        },
        None => Place::Memory(linear),
    })
}

/// What is at one end of a move now.
///
/// # Errors
///
/// [`EmulateError::Width`] for a slice of a vector register that is not a width
/// any move has, or whatever reading the guest's memory or asking a device
/// reports.
pub(crate) fn load(
    mmio: &Mmio,
    vcpu: &Vcpu,
    guest: Linear<'_>,
    place: Place,
    width: Width,
) -> Result<Data, EmulateError> {
    let rip = vcpu.save().rip;
    match place {
        Place::Gpr(register) => gpr::read(vcpu, register),
        // As much of the register as the move touches, from the low end. A move
        // of four bytes out of a vector register takes the low four.
        Place::Vector(vector) => {
            Data::from_bytes(&vector.read()[..width.bytes()]).ok_or(EmulateError::Width { rip })
        }
        Place::Immediate(value) => Ok(Data::from_u64(value, width)),
        Place::Memory(linear) => {
            let mut bytes = [0; Width::Vector.bytes()];
            let taken = &mut bytes[..width.bytes()];
            guest.read(linear, taken)?;
            Data::from_bytes(taken).ok_or(EmulateError::Width { rip })
        }
        Place::Device {
            index, offset, gpa, ..
        } => mmio.read(index, offset, gpa, width),
    }
}

/// Puts a value at one end of a move.
///
/// # Errors
///
/// [`EmulateError::ImmediateDestination`] if the destination decoded as an
/// immediate, which no move has, or whatever writing the guest's memory or
/// asking a device reports.
pub(crate) fn store(
    mmio: &Mmio,
    vcpu: &mut Vcpu,
    guest: Linear<'_>,
    place: Place,
    value: Data,
) -> Result<(), EmulateError> {
    match place {
        Place::Gpr(register) => gpr::write(vcpu, register, value),
        // Sixteen bytes, whatever the width was. Every move that puts fewer than
        // sixteen into a vector register with memory at the other end clears the
        // rest of it, and the zeroes above the value are that rather than
        // padding to be ignored.
        Place::Vector(vector) => {
            vector.write(value.vector());
            Ok(())
        }
        Place::Immediate(_) => Err(EmulateError::ImmediateDestination {
            rip: vcpu.save().rip,
        }),
        Place::Memory(linear) => {
            guest.write(linear, value.bytes())?;
            Ok(())
        }
        Place::Device {
            index, offset, gpa, ..
        } => mmio.write(index, offset, gpa, value),
    }
}

/// How wide one end of a move is.
pub(crate) fn width(instruction: &Instruction, operand: u32) -> Option<Width> {
    match instruction.op_kind(operand) {
        OpKind::Register => {
            let register = instruction.op_register(operand);
            if register.is_xmm() {
                // The move's width, not the register's, and the register's only
                // when there is no memory operand to take it from.
                return memory_width(instruction).or_else(|| Width::from_bytes(register.size()));
            }
            Width::from_bytes(register.size())
        }
        kind if in_memory(kind) => memory_width(instruction),
        _ => None,
    }
}

/// How wide this instruction's memory operand is, if it has one.
pub(crate) fn memory_width(instruction: &Instruction) -> Option<Width> {
    (0..instruction.op_count())
        .any(|operand| in_memory(instruction.op_kind(operand)))
        .then(|| Width::from_bytes(instruction.memory_size().size()))
        .flatten()
}

/// Whether an operand of this kind is in memory.
///
/// The six beside the ordinary one are the string instructions' implicit
/// operands, which name no registers in the encoding and take them from the
/// index registers instead.
pub(crate) const fn in_memory(kind: OpKind) -> bool {
    matches!(
        kind,
        OpKind::Memory
            | OpKind::MemorySegSI
            | OpKind::MemorySegESI
            | OpKind::MemorySegRSI
            | OpKind::MemoryESDI
            | OpKind::MemoryESEDI
            | OpKind::MemoryESRDI
    )
}

/// Which index register a string instruction's implicit operand walks, and so
/// which one a repetition has to advance.
pub(crate) const fn index(kind: OpKind) -> Option<u8> {
    Some(match kind {
        OpKind::MemorySegSI | OpKind::MemorySegESI | OpKind::MemorySegRSI => gpr::RSI,
        OpKind::MemoryESDI | OpKind::MemoryESEDI | OpKind::MemoryESRDI => gpr::RDI,
        _ => return None,
    })
}

/// Whether an operand of this kind is encoded in the instruction.
const fn immediate(kind: OpKind) -> bool {
    matches!(
        kind,
        OpKind::Immediate8
            | OpKind::Immediate8_2nd
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64
            | OpKind::Immediate32to64
    )
}

/// The linear address an operand names, out of the registers the guest stopped
/// with.
fn address(
    instruction: &Instruction,
    operand: u32,
    vcpu: &Vcpu,
    addressing: &Addressing,
) -> Option<u64> {
    instruction.virtual_address(operand, 0, |register, _, _| {
        // A segment register is asked about for its base rather than its value,
        // which is what a state-save area holds and what an offset is added to.
        if let Some(segment) = segment(register) {
            return Some(addressing.base(segment));
        }
        // Anything else is a general-purpose register. A vector one would be a
        // gathering or scattering instruction, which this crate does not perform
        // — and answering `None` is how it declines to compute an address for
        // one rather than computing a wrong one.
        gpr::read(vcpu, register).ok().map(|value| value.as_u64())
    })
}

/// Which segment a register names, or `None` if it is not one.
const fn segment(register: Register) -> Option<Segment> {
    Some(match register {
        Register::ES => Segment::Es,
        Register::CS => Segment::Cs,
        Register::SS => Segment::Ss,
        Register::DS => Segment::Ds,
        Register::FS => Segment::Fs,
        Register::GS => Segment::Gs,
        _ => return None,
    })
}
