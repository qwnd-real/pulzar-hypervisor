//! The moves, one encoding at a time.
//!
//! # Why a mnemonic is not the unit
//!
//! A mnemonic names a family whose members do architecturally different things.
//! `MOVSD` is the clearest case: it is a scalar double-precision move *and* a
//! string move of a dword, two instructions that share nothing but four
//! letters. But it is not the only one. `MOVSS xmm, xmm` preserves the upper 96
//! bits of its destination while `MOVSS xmm, m32` zeroes them. `MOVD` names
//! both an XMM form this crate performs and an MMX form it cannot. `MOVAPS`
//! faults on a misaligned address where `MOVUPS`, the same move otherwise, does
//! not.
//!
//! Accepting a mnemonic and inferring the rest from the operands gets every one
//! of those wrong in the same direction: it performs something plausible
//! instead of what the instruction means. So the accepted set is a table of
//! exact encodings — [`Form`] — and an encoding that is not in it is refused by
//! name. Adding an instruction means adding a row, which means stating its
//! width, its destination policy and its alignment requirement, which is
//! exactly the decision that must not be made implicitly.
//!
//! # What a row says
//!
//! Three things the operands cannot answer:
//!
//! - **How much moves.** Not the register's size: `MOVD xmm, r32` names a
//!   sixteen-byte register and moves four bytes.
//! - **What happens to the rest of the destination.** A vector destination is
//!   either cleared above the value or left alone, and which one is a property
//!   of the encoding rather than of the width.
//! - **Whether the address must be aligned.** The aligned forms raise `#GP(0)`
//!   on a misaligned operand, before any access is made. That is a
//!   guest-visible fault this crate must reproduce: a device whose callback ran
//!   for an access the hardware would have refused has been told about
//!   something that never happened.

use iced_x86::{Code, Instruction, Mnemonic, OpKind};

use crate::{
    EmulateError, Outcome,
    machine::{Cpu, Guest},
    mmio::Mmio,
    operand::{self, DESTINATION, Place, SOURCE},
    plan::{Fault, Plan, Reported},
    value::{Data, Width},
};

/// What one accepted encoding moves, and how.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Form {
    /// How many bytes move.
    pub(crate) width: Width,
    /// What becomes of a destination wider than the value.
    pub(crate) destination: Destination,
    /// Whether the memory operand must be aligned to the transfer width.
    pub(crate) alignment: Alignment,
    /// Whether the value is widened on the way, and how.
    pub(crate) widen: Option<Widen>,
}

/// What happens to the part of a destination the value does not fill.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Destination {
    /// Exactly as wide as the value: a general-purpose register at its own
    /// width, or memory.
    Exact,
    /// A vector register, cleared above the value.
    ///
    /// What every move from memory or from a general-purpose register into a
    /// vector register does, and what the two register-to-register `MOVQ` forms
    /// do.
    ZeroFill,
    /// A vector register, with everything above the value left as it was.
    ///
    /// The legacy scalar register-to-register forms, and only those: `MOVSS
    /// xmm1, xmm2` merges 32 bits into the low lane and `MOVSD xmm1, xmm2`
    /// merges 64, in both cases leaving the rest of the destination alone.
    Merge,
}

/// Whether an encoding requires its memory operand to be aligned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Alignment {
    /// Any address will do, as it will for an ordinary scalar move.
    Any,
    /// The address must be a multiple of the transfer width, and a guest that
    /// breaks that takes `#GP(0)` with nothing else having happened.
    Required,
}

/// How a widening move fills what it adds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Widen {
    /// With zeroes.
    Zero,
    /// With copies of the source's top bit.
    Sign,
}

impl Form {
    /// An ordinary move of this width: exact destination, no alignment
    /// requirement, no widening.
    const fn plain(width: Width) -> Self {
        Self {
            width,
            destination: Destination::Exact,
            alignment: Alignment::Any,
            widen: None,
        }
    }

    /// A move into a vector register that clears what it does not write.
    const fn zeroing(width: Width) -> Self {
        Self {
            width,
            destination: Destination::ZeroFill,
            alignment: Alignment::Any,
            widen: None,
        }
    }

    /// A move into a vector register that leaves the rest of it alone.
    const fn merging(width: Width) -> Self {
        Self {
            width,
            destination: Destination::Merge,
            alignment: Alignment::Any,
            widen: None,
        }
    }

    /// The same, but requiring the memory operand to be aligned to its width.
    const fn aligned(self) -> Self {
        Self {
            alignment: Alignment::Required,
            ..self
        }
    }

    /// A widening move: the destination's width, filled as `widen` says.
    const fn widening(width: Width, widen: Widen) -> Self {
        Self {
            width,
            destination: Destination::Exact,
            alignment: Alignment::Any,
            widen: Some(widen),
        }
    }
}

/// What this encoding moves, or `None` if it is not one this crate performs.
///
/// The whole of the accepted set. Anything absent is refused, including
/// instructions whose mnemonic appears here in another form — which is the
/// point.
pub(crate) fn form(code: Code) -> Option<Form> {
    use Code as C;
    Some(match code {
        // An ordinary move of one byte. Three groups of encodings with one
        // behaviour between them: register to and from register or memory, the
        // accumulator form whose address is absolute, and the two immediate forms.
        C::Mov_rm8_r8
        | C::Mov_r8_rm8
        | C::Mov_AL_moffs8
        | C::Mov_moffs8_AL
        | C::Mov_r8_imm8
        | C::Mov_rm8_imm8 => Form::plain(Width::Byte),

        // The same at two bytes.
        C::Mov_rm16_r16
        | C::Mov_r16_rm16
        | C::Mov_AX_moffs16
        | C::Mov_moffs16_AX
        | C::Mov_r16_imm16
        | C::Mov_rm16_imm16 => Form::plain(Width::Word),

        // At four, with the non-temporal store alongside them: its hint is about
        // caching, which does not survive emulation against an uncached device
        // mapping and changes nothing about what moves.
        C::Mov_rm32_r32
        | C::Mov_r32_rm32
        | C::Mov_EAX_moffs32
        | C::Mov_moffs32_EAX
        | C::Mov_r32_imm32
        | C::Mov_rm32_imm32
        | C::Movnti_m32_r32 => Form::plain(Width::Long),

        // At eight. `Mov_rm64_imm32` belongs here rather than with the four-byte
        // forms: its immediate is encoded in four bytes and sign extended to
        // eight by the decoder, and eight is what the instruction writes.
        C::Mov_rm64_r64
        | C::Mov_r64_rm64
        | C::Mov_RAX_moffs64
        | C::Mov_moffs64_RAX
        | C::Mov_r64_imm64
        | C::Mov_rm64_imm32
        | C::Movnti_m64_r64 => Form::plain(Width::Quad),

        // The widening moves. The width is the destination's, because that is what
        // is written; how much is read comes from the source operand.
        C::Movzx_r16_rm8 | C::Movzx_r16_rm16 => Form::widening(Width::Word, Widen::Zero),
        C::Movzx_r32_rm8 | C::Movzx_r32_rm16 => Form::widening(Width::Long, Widen::Zero),
        C::Movzx_r64_rm8 | C::Movzx_r64_rm16 => Form::widening(Width::Quad, Widen::Zero),
        C::Movsx_r16_rm8 => Form::widening(Width::Word, Widen::Sign),
        C::Movsx_r32_rm8 | C::Movsx_r32_rm16 | C::Movsxd_r32_rm32 => {
            Form::widening(Width::Long, Widen::Sign)
        }
        C::Movsx_r64_rm8 | C::Movsx_r64_rm16 | C::Movsxd_r64_rm32 => {
            Form::widening(Width::Quad, Widen::Sign)
        }

        // The unaligned packed moves, both directions. Sixteen bytes, and a vector
        // destination is written whole, so there is nothing above the value to have
        // a policy about.
        C::Movups_xmm_xmmm128
        | C::Movupd_xmm_xmmm128
        | C::Movdqu_xmm_xmmm128
        | C::Movups_xmmm128_xmm
        | C::Movupd_xmmm128_xmm
        | C::Movdqu_xmmm128_xmm => Form::zeroing(Width::Vector),

        // The same moves with an alignment requirement the guest can break, and
        // the non-temporal packed stores, which carry the same requirement.
        C::Movaps_xmm_xmmm128
        | C::Movapd_xmm_xmmm128
        | C::Movdqa_xmm_xmmm128
        | C::Movaps_xmmm128_xmm
        | C::Movapd_xmmm128_xmm
        | C::Movdqa_xmmm128_xmm
        | C::Movntps_m128_xmm
        | C::Movntpd_m128_xmm
        | C::Movntdq_m128_xmm => Form::zeroing(Width::Vector).aligned(),

        // Four bytes into or out of a vector register, clearing the rest of a
        // vector destination: the scalar single-precision move and the two that
        // move between a vector register and a general-purpose one.
        //
        // The register-to-register scalar form is *not* this — it preserves what it
        // does not write — and `merging` is what tells the two apart.
        C::Movss_xmm_xmmm32 | C::Movss_xmmm32_xmm | C::Movd_xmm_rm32 | C::Movd_rm32_xmm => {
            Form::zeroing(Width::Long)
        }

        // Eight bytes, the same way: the scalar double-precision move, the two
        // that reach a general-purpose register, and the two legacy `MOVQ` forms
        // that move the low quadword between vector registers or memory.
        C::Movsd_xmm_xmmm64
        | C::Movsd_xmmm64_xmm
        | C::Movq_xmm_rm64
        | C::Movq_rm64_xmm
        | C::Movq_xmm_xmmm64
        | C::Movq_xmmm64_xmm => Form::zeroing(Width::Quad),
        _ => return None,
    })
}

/// The form of an instruction whose two register operands make it a different
/// instruction from the same encoding with memory.
///
/// Only the two legacy scalar moves need this, and they need it because the
/// architecture gives one encoding two destination policies: `MOVSS xmm1, xmm2`
/// merges into the low lane and leaves bits 127:32 alone, while `MOVSS xmm1,
/// m32` zeroes them. iced-x86 reports both as one `Code`, so the operands are
/// what distinguish them — which is exactly the sort of inference the rest of
/// this module refuses to make, and is written out here once rather than
/// happening implicitly everywhere.
fn merging(code: Code, instruction: &Instruction) -> Option<Form> {
    let both_registers =
        instruction.op0_kind() == OpKind::Register && instruction.op1_kind() == OpKind::Register;
    if !both_registers {
        return None;
    }
    Some(match code {
        Code::Movss_xmm_xmmm32 | Code::Movss_xmmm32_xmm => Form::merging(Width::Long),
        Code::Movsd_xmm_xmmm64 | Code::Movsd_xmmm64_xmm => Form::merging(Width::Quad),
        _ => return None,
    })
}

/// Performs a move on the guest's behalf.
///
/// Everything that can fail is made to fail before anything changes. The
/// encoding is looked up, the alignment the encoding requires is checked, both
/// ends are resolved, the fault the hardware reported is matched against them,
/// and the destination is proved able to accept a write — and only then is a
/// value read, which for a device is the one step that cannot be taken back.
///
/// # Errors
///
/// [`EmulateError::Unsupported`] for an encoding outside the table,
/// [`EmulateError::Provenance`] if the instruction does not account for the
/// reported fault, and whatever resolving, reading or writing either end
/// reports.
pub(crate) fn perform(
    mmio: &Mmio,
    cpu: &mut impl Cpu,
    guest: &impl Guest,
    instruction: &Instruction,
    fault: Reported,
) -> Result<Outcome, EmulateError> {
    let rip = cpu.save().rip;
    let code = instruction.code();
    let form = form(code).ok_or(EmulateError::Unsupported {
        rip,
        instruction: name(instruction),
    })?;
    // One encoding, two destination policies, distinguished by the operands.
    let form = merging(code, instruction).unwrap_or(form);
    if instruction.op_count() != 2 {
        return Err(EmulateError::Operand { rip, operand: 1 });
    }

    // How much is read is the source's own width for a widening move and the
    // transfer width otherwise. A `MOVZX r32, r/m8` reads one byte and writes
    // four.
    let source = if form.widen.is_some() {
        source_width(instruction).ok_or(EmulateError::Unsupported {
            rip,
            instruction: name(instruction),
        })?
    } else {
        form.width
    };

    let plan = Plan::moving(
        mmio,
        cpu,
        guest,
        instruction,
        (source, form.width),
        (SOURCE, DESTINATION),
    )?;
    // Before anything is read and before any callback runs: an instruction the
    // hardware would have refused must not reach a device at all.
    if let Some(fault) = misaligned(&plan, form) {
        return Ok(Outcome::Faulted(fault));
    }
    plan.authenticate(rip, fault.gpa, fault.cause)?;

    // The last thing that can fail before the source is consumed. A device read
    // cannot be taken back, so a destination that might refuse the write has to
    // be ruled out first rather than discovered afterwards.
    if plan.from.interposed() && !operand::infallible(plan.to.place) {
        preflight(mmio, guest, &plan)?;
    }

    let value = operand::load(mmio, cpu, guest, plan.from.place, source)?;
    let value = match form.widen {
        Some(Widen::Zero) => value.extended(form.width, false),
        Some(Widen::Sign) => value.extended(form.width, true),
        None => Some(value),
    }
    .ok_or(EmulateError::WidthMismatch {
        register: "the destination of a widening move",
        wanted: form.width,
        got: source,
    })?;
    let value = match form.destination {
        // Everything above the value stays as it was, which needs the old
        // contents of the destination to merge into.
        Destination::Merge => merge_into(cpu, &plan, value)?,
        Destination::Exact | Destination::ZeroFill => value,
    };
    commit(mmio, cpu, guest, &plan, value)?;
    Ok(Outcome::Stepped)
}

/// How much a widening move reads, which is its source operand's own width.
fn source_width(instruction: &Instruction) -> Option<Width> {
    match instruction.op1_kind() {
        OpKind::Register => Width::from_bytes(instruction.op1_register().size()),
        kind if operand::in_memory(kind) => Width::from_bytes(instruction.memory_size().size()),
        _ => None,
    }
}

/// The fault an aligned form owes when its memory operand is not aligned.
///
/// Checked against the linear address the instruction named, which is the
/// address the architecture applies the requirement to.
fn misaligned(plan: &Plan, form: Form) -> Option<Fault> {
    if form.alignment != Alignment::Required {
        return None;
    }
    [plan.from.place, plan.to.place]
        .into_iter()
        .filter_map(operand::Place::linear)
        .any(|linear| !linear.is_multiple_of(form.width.span()))
        .then(Fault::protection)
}

/// Proves the destination will accept a write, before the source is consumed.
///
/// This is what makes a device read the *last* fallible thing an instruction
/// does. A device that has answered cannot be asked to un-answer: a
/// clear-on-read register has already cleared, a FIFO has already popped. So an
/// error discovered after the read leaves the guest's state and the device's
/// state describing different histories, and the only honest report of that is
/// one that says so — which is worse for the caller than not getting into it.
///
/// Only properties that can be established without changing anything are
/// checked here. A device destination is asked whether it admits the access,
/// which is the same check the write itself makes and which touches no
/// hardware. A memory destination is asked whether the guest may write it,
/// which is a walk of the nested tables and no more.
fn preflight(mmio: &Mmio, guest: &impl Guest, plan: &Plan) -> Result<(), EmulateError> {
    match plan.to.place {
        Place::Device {
            index, offset, gpa, ..
        } => mmio.admits(index, offset, gpa, plan.to.width),
        // Writing nothing, to find out whether writing something would be
        // allowed. The guest's memory answers all-or-nothing, so a zero-length
        // probe cannot tell us anything — the range itself has to be the one that
        // will be written.
        Place::Memory(linear) => {
            guest
                .writable(linear, plan.to.width)?
                .then_some(())
                .ok_or(EmulateError::Discarded {
                    linear,
                    bytes: plan.to.width.bytes(),
                })
        }
        // A register cannot refuse a write of the width the encoding gives it,
        // and the encoding is what decided the width.
        Place::Gpr(_) | Place::Vector(_) | Place::Immediate(_) => Ok(()),
    }
}

/// The value to write, with the part of the destination it does not cover kept.
fn merge_into(cpu: &impl Cpu, plan: &Plan, value: Data) -> Result<Data, EmulateError> {
    let operand::Place::Vector(register) = plan.to.place else {
        // Only a vector destination has anything above the value to preserve, and
        // only the two legacy scalar forms ask for this — both of which name a
        // vector register on both sides.
        return Err(EmulateError::WidthMismatch {
            register: "a merging move's destination",
            wanted: Width::Vector,
            got: value.width(),
        });
    };
    let mut bytes = cpu.vector(register);
    bytes[..value.width().bytes()].copy_from_slice(value.bytes());
    Ok(Data::vector_from(bytes))
}

/// Puts the value at the destination, at the width the destination takes.
fn commit(
    mmio: &Mmio,
    cpu: &mut impl Cpu,
    guest: &impl Guest,
    plan: &Plan,
    value: Data,
) -> Result<(), EmulateError> {
    let value = match (plan.to.place, plan.to.width) {
        // A vector register is written whole: the zeroes above the value are the
        // clearing that the encoding specifies rather than padding to ignore.
        (Place::Vector(_), _) => Data::vector_from(value.vector()),
        _ => value,
    };
    operand::store(mmio, cpu, guest, plan.to.place, value)
}

/// What one repetition of a string instruction moves, or `None` if the encoding
/// is not one.
///
/// By exact encoding, like everything else here, and for the sharpest reason
/// there is: `Movsd_m32_m32` is a string move of four bytes and
/// `Movsd_xmm_xmmm64` is a scalar move of eight, and the two share a mnemonic.
pub(crate) fn string(code: Code) -> Option<Width> {
    use Code as C;
    Some(match code {
        C::Movsb_m8_m8 | C::Stosb_m8_AL | C::Lodsb_AL_m8 => Width::Byte,
        C::Movsw_m16_m16 | C::Stosw_m16_AX | C::Lodsw_AX_m16 => Width::Word,
        C::Movsd_m32_m32 | C::Stosd_m32_EAX | C::Lodsd_EAX_m32 => Width::Long,
        C::Movsq_m64_m64 | C::Stosq_m64_RAX | C::Lodsq_RAX_m64 => Width::Quad,
        _ => return None,
    })
}

/// What to call an instruction this crate will not perform.
///
/// The mnemonic where it is one of the families a driver plausibly aims at a
/// device register, so that a log says what the guest did rather than a number.
/// Everything else is described by what it is not, because the useful part of
/// the message is that it is outside the family rather than which of several
/// thousand encodings it happens to be.
fn name(instruction: &Instruction) -> &'static str {
    /// One entry per family this crate has been asked about and declined.
    const KNOWN: [(Mnemonic, &str); 18] = [
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
        (Mnemonic::Movbe, "movbe"),
        (Mnemonic::Movdiri, "movdiri"),
        (Mnemonic::Movdir64b, "movdir64b"),
        (Mnemonic::Movntdqa, "movntdqa"),
        (Mnemonic::Insb, "insb"),
        (Mnemonic::Outsb, "outsb"),
    ];
    let mnemonic = instruction.mnemonic();
    if let Some(name) = KNOWN
        .iter()
        .find_map(|(known, name)| (*known == mnemonic).then_some(*name))
    {
        return name;
    }
    // The families that are moves by name and are refused anyway, which are worth
    // distinguishing from an instruction that was never a move at all.
    if instruction.op0_kind() == OpKind::Register && instruction.op0_register().is_mm()
        || instruction.op1_kind() == OpKind::Register && instruction.op1_register().is_mm()
    {
        return "a move of an MMX register, whose state this hypervisor does not keep";
    }
    if mnemonic == Mnemonic::Mov {
        return "a move of a segment, control, debug or test register";
    }
    "an instruction that is not a move"
}

#[cfg(test)]
mod tests {
    use iced_x86::{Code, Decoder, DecoderOptions, Instruction};

    use super::{Alignment, Destination, Widen, form, merging, string};
    use crate::value::Width;

    /// What one encoding decodes to, in 64-bit mode.
    fn decode(bytes: &[u8]) -> Instruction {
        let mut decoder = Decoder::with_ip(64, bytes, 0x1000, DecoderOptions::NONE);
        decoder.decode()
    }

    #[test]
    fn every_immediate_move_has_a_width() {
        // The defect this replaces: width discovery answered `None` for every
        // immediate form, so no MMIO initialization or command write worked at
        // all. Each of these is a real encoding, checked by decoding it.
        for (bytes, code, width) in [
            (&[0xB0, 0x12][..], Code::Mov_r8_imm8, Width::Byte),
            (
                &[0x66, 0xB8, 0x34, 0x12][..],
                Code::Mov_r16_imm16,
                Width::Word,
            ),
            (
                &[0xB8, 0x78, 0x56, 0x34, 0x12][..],
                Code::Mov_r32_imm32,
                Width::Long,
            ),
            (
                &[0x48, 0xB8, 0, 0, 0, 0, 0, 0, 0, 0][..],
                Code::Mov_r64_imm64,
                Width::Quad,
            ),
            (&[0xC6, 0x00, 0x12][..], Code::Mov_rm8_imm8, Width::Byte),
            (
                &[0x66, 0xC7, 0x00, 0x34, 0x12][..],
                Code::Mov_rm16_imm16,
                Width::Word,
            ),
            (
                &[0xC7, 0x00, 0x78, 0x56, 0x34, 0x12][..],
                Code::Mov_rm32_imm32,
                Width::Long,
            ),
            (
                &[0x48, 0xC7, 0x00, 0xFF, 0xFF, 0xFF, 0xFF][..],
                Code::Mov_rm64_imm32,
                Width::Quad,
            ),
        ] {
            let decoded = decode(bytes);
            assert_eq!(
                decoded.code(),
                code,
                "the test's own encoding must be {code:?}"
            );
            let form = form(code).unwrap_or_else(|| panic!("{code:?} must be performed"));
            assert_eq!(form.width, width, "{code:?} moves {width:?}");
            assert_eq!(form.destination, Destination::Exact);
            assert_eq!(form.widen, None);
        }
    }

    #[test]
    fn a_sign_extended_immediate_moves_the_destination_width() {
        // `48 C7 00 FF FF FF FF` is one eight-byte write of all ones, not a
        // four-byte one: the encoding widens the immediate and the destination is
        // a qword.
        let decoded = decode(&[0x48, 0xC7, 0x00, 0xFF, 0xFF, 0xFF, 0xFF]);
        assert_eq!(decoded.code(), Code::Mov_rm64_imm32);
        assert_eq!(decoded.len(), 7);
        assert_eq!(form(decoded.code()).expect("performed").width, Width::Quad);
        // iced-x86 has already widened it, which is why nothing here re-does it.
        assert_eq!(decoded.immediate(1), u64::MAX);
    }

    #[test]
    fn the_aligned_packed_moves_require_alignment_and_the_unaligned_ones_do_not() {
        for code in [
            Code::Movaps_xmm_xmmm128,
            Code::Movapd_xmm_xmmm128,
            Code::Movdqa_xmm_xmmm128,
            Code::Movaps_xmmm128_xmm,
            Code::Movapd_xmmm128_xmm,
            Code::Movdqa_xmmm128_xmm,
            Code::Movntps_m128_xmm,
            Code::Movntpd_m128_xmm,
            Code::Movntdq_m128_xmm,
        ] {
            let form = form(code).unwrap_or_else(|| panic!("{code:?} must be performed"));
            assert_eq!(
                form.alignment,
                Alignment::Required,
                "{code:?} raises #GP(0) on a misaligned operand"
            );
            assert_eq!(form.width, Width::Vector);
        }
        for code in [
            Code::Movups_xmm_xmmm128,
            Code::Movupd_xmm_xmmm128,
            Code::Movdqu_xmm_xmmm128,
            Code::Movups_xmmm128_xmm,
            Code::Movupd_xmmm128_xmm,
            Code::Movdqu_xmmm128_xmm,
        ] {
            let form = form(code).unwrap_or_else(|| panic!("{code:?} must be performed"));
            assert_eq!(
                form.alignment,
                Alignment::Any,
                "{code:?} is the unaligned form and must not fault"
            );
        }
    }

    #[test]
    fn the_legacy_scalar_moves_merge_between_registers_and_zero_from_memory() {
        // One encoding, two architectural behaviours. `MOVSS xmm1, xmm2` leaves
        // bits 127:32 of the destination alone; `MOVSS xmm1, m32` clears them.
        let register = decode(&[0xF3, 0x0F, 0x10, 0xC1]);
        assert_eq!(register.code(), Code::Movss_xmm_xmmm32);
        let merged = merging(register.code(), &register).expect("both operands are registers");
        assert_eq!(merged.destination, Destination::Merge);
        assert_eq!(merged.width, Width::Long);

        let memory = decode(&[0xF3, 0x0F, 0x10, 0x00]);
        assert_eq!(memory.code(), Code::Movss_xmm_xmmm32);
        assert_eq!(
            merging(memory.code(), &memory),
            None,
            "a memory source zeroes the rest and does not merge"
        );
        assert_eq!(
            form(memory.code()).expect("performed").destination,
            Destination::ZeroFill
        );
    }

    #[test]
    fn the_double_precision_scalar_move_merges_sixty_four_bits() {
        let register = decode(&[0xF2, 0x0F, 0x10, 0xC1]);
        assert_eq!(register.code(), Code::Movsd_xmm_xmmm64);
        let merged = merging(register.code(), &register).expect("both operands are registers");
        assert_eq!(merged.destination, Destination::Merge);
        assert_eq!(merged.width, Width::Quad, "MOVSD merges the low 64 bits");
    }

    #[test]
    fn moving_between_a_vector_and_a_general_purpose_register_moves_four_or_eight_bytes() {
        // Not sixteen, which is what the register's own size would have said.
        for (code, width) in [
            (Code::Movd_xmm_rm32, Width::Long),
            (Code::Movd_rm32_xmm, Width::Long),
            (Code::Movq_xmm_rm64, Width::Quad),
            (Code::Movq_rm64_xmm, Width::Quad),
        ] {
            let form = form(code).unwrap_or_else(|| panic!("{code:?} must be performed"));
            assert_eq!(form.width, width);
            assert_eq!(
                form.destination,
                Destination::ZeroFill,
                "{code:?} clears the rest of a vector destination"
            );
        }
    }

    #[test]
    fn the_mmx_forms_are_not_performed() {
        // Their state is not something this hypervisor keeps, so they are refused
        // by name rather than performed against whatever is in an XMM register.
        for code in [
            Code::Movd_mm_rm32,
            Code::Movq_mm_rm64,
            Code::Movd_rm32_mm,
            Code::Movq_rm64_mm,
            Code::Movq_mm_mmm64,
            Code::Movq_mmm64_mm,
            Code::Movntq_m64_mm,
            Code::Movq2dq_xmm_mm,
            Code::Movdq2q_mm_xmm,
        ] {
            assert_eq!(form(code), None, "{code:?} needs MMX state and is refused");
        }
    }

    #[test]
    fn the_privileged_and_segment_moves_are_not_performed() {
        for code in [
            Code::Mov_r32_cr,
            Code::Mov_r64_cr,
            Code::Mov_cr_r32,
            Code::Mov_cr_r64,
            Code::Mov_r32_dr,
            Code::Mov_r64_dr,
            Code::Mov_dr_r32,
            Code::Mov_dr_r64,
            Code::Mov_r32_tr,
            Code::Mov_tr_r32,
            Code::Mov_rm16_Sreg,
            Code::Mov_Sreg_rm16,
            Code::Mov_r32m16_Sreg,
            Code::Mov_Sreg_r32m16,
            Code::Mov_r64m16_Sreg,
            Code::Mov_Sreg_r64m16,
        ] {
            assert_eq!(form(code), None, "{code:?} is not an ordinary move");
        }
    }

    #[test]
    fn the_partial_vector_moves_are_not_performed() {
        // These move one lane of a register and leave the other alone, which is
        // not what any policy in the table describes. Performing them as ordinary
        // moves would overwrite the half they must preserve.
        for code in [
            Code::Movlps_xmm_m64,
            Code::Movlps_m64_xmm,
            Code::Movhps_xmm_m64,
            Code::Movhps_m64_xmm,
            Code::Movlpd_xmm_m64,
            Code::Movhpd_m64_xmm,
            Code::Movhlps_xmm_xmm,
            Code::Movlhps_xmm_xmm,
            Code::Movddup_xmm_xmmm64,
            Code::Movsldup_xmm_xmmm128,
            Code::Movshdup_xmm_xmmm128,
            Code::Movmskps_r32_xmm,
            Code::Movmskpd_r32_xmm,
            Code::Movntdqa_xmm_m128,
        ] {
            assert_eq!(form(code), None, "{code:?} is not a whole-register move");
        }
    }

    #[test]
    fn every_widening_move_writes_its_destination_width() {
        for (code, width, widen) in [
            (Code::Movzx_r16_rm8, Width::Word, Widen::Zero),
            (Code::Movzx_r32_rm8, Width::Long, Widen::Zero),
            (Code::Movzx_r64_rm8, Width::Quad, Widen::Zero),
            (Code::Movzx_r32_rm16, Width::Long, Widen::Zero),
            (Code::Movzx_r64_rm16, Width::Quad, Widen::Zero),
            (Code::Movsx_r16_rm8, Width::Word, Widen::Sign),
            (Code::Movsx_r32_rm8, Width::Long, Widen::Sign),
            (Code::Movsx_r64_rm8, Width::Quad, Widen::Sign),
            (Code::Movsx_r32_rm16, Width::Long, Widen::Sign),
            (Code::Movsx_r64_rm16, Width::Quad, Widen::Sign),
            (Code::Movsxd_r32_rm32, Width::Long, Widen::Sign),
            (Code::Movsxd_r64_rm32, Width::Quad, Widen::Sign),
        ] {
            let form = form(code).unwrap_or_else(|| panic!("{code:?} must be performed"));
            assert_eq!(form.width, width, "{code:?} writes {width:?}");
            assert_eq!(form.widen, Some(widen));
        }
    }

    #[test]
    fn the_string_encodings_are_told_apart_from_the_scalar_ones_of_the_same_name() {
        // The sharpest case in the whole table: two instructions, one mnemonic.
        assert_eq!(string(Code::Movsd_m32_m32), Some(Width::Long));
        assert_eq!(
            string(Code::Movsd_xmm_xmmm64),
            None,
            "the scalar move is not a string instruction"
        );
        assert_eq!(
            form(Code::Movsd_m32_m32),
            None,
            "the string move is not performed as a scalar one"
        );
        assert_eq!(
            form(Code::Movsd_xmm_xmmm64).expect("performed").width,
            Width::Quad
        );
    }

    #[test]
    fn every_string_encoding_moves_its_own_width() {
        for (code, width) in [
            (Code::Movsb_m8_m8, Width::Byte),
            (Code::Movsw_m16_m16, Width::Word),
            (Code::Movsd_m32_m32, Width::Long),
            (Code::Movsq_m64_m64, Width::Quad),
            (Code::Stosb_m8_AL, Width::Byte),
            (Code::Stosw_m16_AX, Width::Word),
            (Code::Stosd_m32_EAX, Width::Long),
            (Code::Stosq_m64_RAX, Width::Quad),
            (Code::Lodsb_AL_m8, Width::Byte),
            (Code::Lodsw_AX_m16, Width::Word),
            (Code::Lodsd_EAX_m32, Width::Long),
            (Code::Lodsq_RAX_m64, Width::Quad),
        ] {
            assert_eq!(string(code), Some(width), "{code:?} repeats {width:?}");
        }
    }

    #[test]
    fn the_read_modify_write_families_are_not_moves() {
        // The whole reason this crate refuses by name: emulating one of these as
        // a move would put a wrong value in a device register and leave the
        // guest's flags describing an operation that never happened.
        for bytes in [
            &[0x01, 0x00][..],       // add [rax], eax
            &[0x09, 0x00][..],       // or [rax], eax
            &[0x21, 0x00][..],       // and [rax], eax
            &[0x31, 0x00][..],       // xor [rax], eax
            &[0x87, 0x00][..],       // xchg [rax], eax
            &[0x0F, 0xB1, 0x00][..], // cmpxchg [rax], eax
            &[0x0F, 0xC1, 0x00][..], // xadd [rax], eax
            &[0xFF, 0x00][..],       // inc dword [rax]
            &[0x85, 0x00][..],       // test [rax], eax
        ] {
            let decoded = decode(bytes);
            assert_eq!(
                form(decoded.code()),
                None,
                "{:?} is not a move and must not be performed as one",
                decoded.code()
            );
        }
    }

    #[test]
    fn the_plain_moves_cover_both_directions_at_every_width() {
        for (code, width) in [
            (Code::Mov_rm8_r8, Width::Byte),
            (Code::Mov_r8_rm8, Width::Byte),
            (Code::Mov_rm16_r16, Width::Word),
            (Code::Mov_r16_rm16, Width::Word),
            (Code::Mov_rm32_r32, Width::Long),
            (Code::Mov_r32_rm32, Width::Long),
            (Code::Mov_rm64_r64, Width::Quad),
            (Code::Mov_r64_rm64, Width::Quad),
            (Code::Mov_AL_moffs8, Width::Byte),
            (Code::Mov_moffs8_AL, Width::Byte),
            (Code::Mov_AX_moffs16, Width::Word),
            (Code::Mov_moffs16_AX, Width::Word),
            (Code::Mov_EAX_moffs32, Width::Long),
            (Code::Mov_moffs32_EAX, Width::Long),
            (Code::Mov_RAX_moffs64, Width::Quad),
            (Code::Mov_moffs64_RAX, Width::Quad),
            (Code::Movnti_m32_r32, Width::Long),
            (Code::Movnti_m64_r64, Width::Quad),
        ] {
            let form = form(code).unwrap_or_else(|| panic!("{code:?} must be performed"));
            assert_eq!(form.width, width);
            assert_eq!(form.destination, Destination::Exact);
            assert_eq!(form.alignment, Alignment::Any);
        }
    }

    #[test]
    fn nothing_outside_the_move_family_is_accepted_by_accident() {
        // A sweep of the whole encoding space, which is what stops a future
        // version of the decoder from quietly bringing new semantics into the
        // accepted set: anything accepted has to be a move of a shape the table
        // states, and the count is asserted so that additions are deliberate.
        let accepted = (0..=u16::MAX)
            .filter_map(|raw| {
                let code = Code::try_from(usize::from(raw)).ok()?;
                form(code).map(|_| code)
            })
            .count();
        assert_eq!(
            accepted, 64,
            "the accepted set changed; every addition must state its width, \
             destination policy and alignment requirement deliberately"
        );
    }
}
