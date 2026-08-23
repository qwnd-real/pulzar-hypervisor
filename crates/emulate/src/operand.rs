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
//! What is *not* delegated is which base a segment contributes. In 64-bit code
//! four of the six segments are architecturally ignored — their bases
//! contribute zero however the descriptor that loaded them read — and a
//! state-save area holds whatever the guest last put there. Handing those
//! values back would let a stale or hostile base move every address an
//! instruction computes.
//!
//! # A vector register's size is not a move's width
//!
//! Moving four bytes into a vector register still names a sixteen-byte
//! register, so the register's own size answers the wrong question. The width
//! of a move is the encoding's to state, which is why it arrives here as an
//! argument rather than being derived: [`crate::mov`] holds one entry per
//! accepted encoding, and the width in that entry is what the architecture says
//! that encoding moves.

use iced_x86::{Instruction, OpKind, Register};
use memory::{Addressing, Segment, Written};
use npt::RegionTag;
use x86_64::PhysAddr;

use crate::{
    EmulateError, gpr,
    machine::{Cpu, Guest},
    mmio::{self, Mmio},
    plan,
    value::{Data, Width},
    xmm::Vector,
};

/// One end of a move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Place {
    /// A general-purpose register.
    Gpr(Register),
    /// A vector register, which is still in the processor.
    Vector(Vector),
    /// A value encoded in the instruction, already widened however its encoding
    /// says.  Only ever a source.
    Immediate(u64),
    /// The guest's own memory, at a linear address whose whole span has been
    /// shown to translate contiguously.
    Memory(u64),
    /// A region something answers for instead of the hardware.
    Device {
        /// Which region, by the name the nested tables gave it.
        tag: RegionTag,
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

/// Where one operand of an instruction is, with every byte of it accounted for.
///
/// `width` is how much the encoding moves, and it is not optional: an operand
/// in memory is classified by its whole span, and a span needs a length.
/// Passing the register's own size would classify a four-byte move into a
/// vector register as sixteen bytes and consult four regions too many.
///
/// # Errors
///
/// [`EmulateError::Operand`] for an operand kind no move this crate performs
/// has, [`EmulateError::Address`] if the effective address cannot be computed
/// from the registers the guest stopped with, [`EmulateError::Span`] if the
/// operand's bytes do not lie in one contiguous piece of one region, or
/// [`EmulateError::Memory`] if the guest's own tables do not translate it.
pub(crate) fn place(
    cpu: &impl Cpu,
    guest: &impl Guest,
    instruction: &Instruction,
    operand: u32,
    width: Width,
) -> Result<Place, EmulateError> {
    let rip = cpu.save().rip;
    let kind = instruction.op_kind(operand);
    if kind == OpKind::Register {
        let register = instruction.op_register(operand);
        return match Vector::new(register) {
            Some(vector) => Ok(Place::Vector(vector)),
            // Not a vector register this crate can reach, so either a
            // general-purpose one or something a move of this family does not
            // have. `gpr` decides which, and says so.
            None => gpr::width(register).map(|_| Place::Gpr(register)),
        };
    }
    if immediate(kind) {
        return Ok(Place::Immediate(instruction.immediate(operand)));
    }
    if !in_memory(kind) {
        return Err(EmulateError::Operand { rip, operand });
    }

    let linear = address(instruction, operand, cpu, guest.addressing())
        .ok_or(EmulateError::Address { rip, operand })?;
    // The whole span, not its first byte: this is what establishes that one
    // access can describe the operand at all.
    let gpa = plan::contiguous(guest, linear, width)?;
    mmio::classify(guest, gpa, width, linear)
}

/// What is at one end of a move now.
///
/// # Errors
///
/// [`EmulateError::WidthMismatch`] if a register end is not as wide as the
/// move, or whatever reading the guest's memory or asking a device reports.
pub(crate) fn load(
    mmio: &Mmio,
    cpu: &impl Cpu,
    guest: &impl Guest,
    place: Place,
    width: Width,
) -> Result<Data, EmulateError> {
    match place {
        // A general-purpose end is exactly as wide as the move: the encoding
        // decided both, so a disagreement is a fault in the table above rather
        // than something to truncate.
        Place::Gpr(register) => {
            let value = gpr::read(cpu, register)?;
            if value.width() == width {
                Ok(value)
            } else {
                Err(EmulateError::WidthMismatch {
                    register: gpr::name(register),
                    wanted: width,
                    got: value.width(),
                })
            }
        }
        // As much of the register as the move touches, from the low end. A move
        // of four bytes out of a vector register takes the low four.
        Place::Vector(vector) => Data::from_bytes(&cpu.vector(vector)[..width.bytes()]).ok_or(
            EmulateError::WidthMismatch {
                register: "a vector register",
                wanted: width,
                got: Width::Vector,
            },
        ),
        Place::Immediate(value) => Ok(Data::from_u64(value, width)),
        Place::Memory(linear) => {
            let mut bytes = [0; Width::Vector.bytes()];
            let taken = &mut bytes[..width.bytes()];
            guest.read(linear, taken)?;
            Data::from_bytes(taken).ok_or(EmulateError::WidthMismatch {
                register: "the guest's memory",
                wanted: width,
                got: Width::Vector,
            })
        }
        Place::Device {
            tag, offset, gpa, ..
        } => mmio.read(tag, offset, gpa, width),
    }
}

/// Puts a value at one end of a move.
///
/// # Errors
///
/// [`EmulateError::Operand`] if the destination decoded as an immediate, which
/// no move has, [`EmulateError::Discarded`] if the guest's memory would not
/// accept the write, or whatever asking a device reports.
pub(crate) fn store(
    mmio: &Mmio,
    cpu: &mut impl Cpu,
    guest: &impl Guest,
    place: Place,
    value: Data,
) -> Result<(), EmulateError> {
    match place {
        Place::Gpr(register) => gpr::write(cpu, register, value),
        Place::Vector(vector) => {
            cpu.set_vector(vector, value.vector());
            Ok(())
        }
        Place::Immediate(_) => Err(EmulateError::Operand {
            rip: cpu.save().rip,
            operand: DESTINATION,
        }),
        // A discarded write is not a write. The guest's memory rejects one when
        // it is the hypervisor's own memory shadowed behind the guest's view, and
        // an instruction whose destination never changed has not been performed —
        // so it must not be retired as though it had.
        Place::Memory(linear) => match guest.write(linear, value.bytes())? {
            Written::Committed => Ok(()),
            Written::Discarded => Err(EmulateError::Discarded {
                linear,
                bytes: value.width().bytes(),
            }),
        },
        Place::Device {
            tag, offset, gpa, ..
        } => mmio.write(tag, offset, gpa, value),
    }
}

/// Whether a store to this end can fail once the value is in hand.
///
/// What lets a device read be the *last* fallible thing an instruction does. A
/// device answer cannot be taken back, so everything that could fail after it
/// has to be either ruled out beforehand or known to be incapable of failing —
/// registers are the latter, and everything else has to be checked.
pub(crate) const fn infallible(place: Place) -> bool {
    matches!(place, Place::Gpr(_) | Place::Vector(_))
}

/// Which operand of a two-operand instruction is the destination.
pub(crate) const DESTINATION: u32 = 0;
/// Which is the source.
pub(crate) const SOURCE: u32 = 1;

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
pub(crate) const fn immediate(kind: OpKind) -> bool {
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
    cpu: &impl Cpu,
    addressing: &Addressing,
) -> Option<u64> {
    instruction.virtual_address(operand, 0, |register, _, _| {
        // A segment register is asked about for its base rather than its value,
        // which is what a state-save area holds and what an offset is added to.
        if let Some(segment) = segment(register) {
            return Some(base(addressing, segment));
        }
        // Anything else is a general-purpose register. A vector one would be a
        // gathering or scattering instruction, which this crate does not perform
        // — and answering `None` is how it declines to compute an address for one
        // rather than computing a wrong one.
        gpr::read(cpu, register).ok().map(|value| value.as_u64())
    })
}

/// What a segment contributes to an address in the mode the guest is in.
///
/// In 64-bit code the four ordinary segments are ignored by the architecture:
/// an address is the offset, whatever their descriptors said. Only the two that
/// keep a full base — the ones an operating system points at per-processor and
/// per-thread storage — go on contributing. Handing back a saved base for the
/// other four would let a value the guest itself last loaded displace every
/// address the instruction computes.
fn base(addressing: &Addressing, segment: Segment) -> u64 {
    if addressing.long_mode() && !matches!(segment, Segment::Fs | Segment::Gs) {
        return 0;
    }
    addressing.base(segment)
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

#[cfg(test)]
mod tests {
    use iced_x86::{OpKind, Register};
    use memory::Segment;
    use npt::RegionTag;

    use super::{Place, base, immediate, in_memory, index, infallible};
    use crate::{
        gpr,
        machine::{Cpu, tests::Machine},
        xmm::Vector,
    };

    #[test]
    fn the_four_ignored_segments_contribute_nothing_in_sixty_four_bit_code() {
        // A guest can load whatever it likes into these and the architecture
        // ignores their bases in 64-bit code. Trusting the saved value would move
        // every address the instruction computes.
        let mut machine = Machine::long_mode();
        let save = machine.save_mut();
        save.es.base = 0xDEAD_0000;
        save.cs.base = 0xBEEF_0000;
        save.ss.base = 0xCAFE_0000;
        save.ds.base = 0xF00D_0000;
        save.fs.base = 0x1111_0000;
        save.gs.base = 0x2222_0000;
        let addressing = machine.addressing();

        for segment in [Segment::Es, Segment::Cs, Segment::Ss, Segment::Ds] {
            assert_eq!(
                base(&addressing, segment),
                0,
                "{segment:?} is architecturally ignored in 64-bit code"
            );
        }
        // These two are not ignored, and an operating system depends on them.
        assert_eq!(base(&addressing, Segment::Fs), 0x1111_0000);
        assert_eq!(base(&addressing, Segment::Gs), 0x2222_0000);
    }

    #[test]
    fn every_segment_contributes_its_base_outside_sixty_four_bit_code() {
        for mut machine in [Machine::protected(), Machine::real()] {
            let save = machine.save_mut();
            save.es.base = 0x1_0000;
            save.cs.base = 0x2_0000;
            save.ss.base = 0x3_0000;
            save.ds.base = 0x4_0000;
            save.fs.base = 0x5_0000;
            save.gs.base = 0x6_0000;
            let addressing = machine.addressing();
            for (segment, expected) in [
                (Segment::Es, 0x1_0000),
                (Segment::Cs, 0x2_0000),
                (Segment::Ss, 0x3_0000),
                (Segment::Ds, 0x4_0000),
                (Segment::Fs, 0x5_0000),
                (Segment::Gs, 0x6_0000),
            ] {
                assert_eq!(
                    base(&addressing, segment),
                    expected,
                    "{segment:?} is not ignored outside 64-bit code"
                );
            }
        }
    }

    #[test]
    fn the_implicit_string_operands_are_the_ones_in_memory() {
        for kind in [
            OpKind::Memory,
            OpKind::MemorySegSI,
            OpKind::MemorySegESI,
            OpKind::MemorySegRSI,
            OpKind::MemoryESDI,
            OpKind::MemoryESEDI,
            OpKind::MemoryESRDI,
        ] {
            assert!(in_memory(kind), "{kind:?} is in memory");
        }
        for kind in [
            OpKind::Register,
            OpKind::NearBranch16,
            OpKind::NearBranch32,
            OpKind::NearBranch64,
            OpKind::FarBranch16,
            OpKind::FarBranch32,
            OpKind::Immediate8,
            OpKind::Immediate64,
        ] {
            assert!(!in_memory(kind), "{kind:?} is not in memory");
        }
    }

    #[test]
    fn each_implicit_operand_names_the_index_register_it_walks() {
        for kind in [
            OpKind::MemorySegSI,
            OpKind::MemorySegESI,
            OpKind::MemorySegRSI,
        ] {
            assert_eq!(
                index(kind),
                Some(gpr::RSI),
                "{kind:?} walks the source index"
            );
        }
        for kind in [OpKind::MemoryESDI, OpKind::MemoryESEDI, OpKind::MemoryESRDI] {
            assert_eq!(
                index(kind),
                Some(gpr::RDI),
                "{kind:?} walks the destination index"
            );
        }
        // An ordinary memory operand walks nothing: its address is in the
        // encoding rather than in an index register.
        assert_eq!(index(OpKind::Memory), None);
        assert_eq!(index(OpKind::Register), None);
    }

    #[test]
    fn every_immediate_encoding_is_recognized_as_one() {
        for kind in [
            OpKind::Immediate8,
            OpKind::Immediate8_2nd,
            OpKind::Immediate16,
            OpKind::Immediate32,
            OpKind::Immediate64,
            OpKind::Immediate8to16,
            OpKind::Immediate8to32,
            OpKind::Immediate8to64,
            OpKind::Immediate32to64,
        ] {
            assert!(immediate(kind), "{kind:?} is an immediate");
        }
        for kind in [OpKind::Register, OpKind::Memory, OpKind::NearBranch64] {
            assert!(!immediate(kind), "{kind:?} is not an immediate");
        }
    }

    #[test]
    fn only_a_register_destination_cannot_fail() {
        // What lets a device read be the last fallible thing an instruction does.
        assert!(infallible(Place::Gpr(Register::RAX)));
        assert!(infallible(Place::Vector(Vector::Xmm0)));
        assert!(!infallible(Place::Memory(0x1000)));
        assert!(!infallible(Place::Immediate(1)));
        assert!(!infallible(Place::Device {
            tag: RegionTag::new(0),
            offset: 0,
            gpa: x86_64::PhysAddr::new(0x1000),
            linear: 0x1000,
        }));
    }

    #[test]
    fn only_an_end_in_memory_has_a_linear_address() {
        assert_eq!(Place::Memory(0x1234).linear(), Some(0x1234));
        assert_eq!(
            Place::Device {
                tag: RegionTag::new(0),
                offset: 8,
                gpa: x86_64::PhysAddr::new(0x2000),
                linear: 0x5678,
            }
            .linear(),
            Some(0x5678)
        );
        assert_eq!(Place::Gpr(Register::RAX).linear(), None);
        assert_eq!(Place::Vector(Vector::Xmm1).linear(), None);
        assert_eq!(Place::Immediate(7).linear(), None);
    }
}
