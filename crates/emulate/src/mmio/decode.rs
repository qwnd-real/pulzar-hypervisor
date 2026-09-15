//! Getting the instruction the guest stopped on, and reading it.
//!
//! # Where the bytes come from
//!
//! Usually from the control block. On a nested page fault the processor has
//! already fetched the instruction and leaves it there, which is both free and
//! authoritative — it is what the processor decoded, not what is at that
//! address now.
//!
//! When it has not, the bytes come out of the guest at `CS.base + RIP`, which
//! is a guest linear address like any other and is read like any other. Two
//! things about that read are deliberate. It stops at a page boundary and then
//! tries to carry on, because an instruction may straddle one and the far side
//! may not be mapped — a guest can perfectly well have a page there and not the
//! next. And it reports what it got rather than insisting on fifteen bytes: an
//! instruction is at most fifteen bytes but is usually far fewer, and refusing
//! to decode two bytes because thirteen more were unreadable would refuse most
//! instructions near the top of a page.
//!
//! # What the fallback cannot promise
//!
//! Bytes read back out of the guest are not a snapshot of the instruction the
//! processor executed. Nothing here stops the other processors, and nothing
//! here stops a device writing the page — so between the fault and this read,
//! the bytes may have changed, and an instruction that straddles a page
//! boundary is read in two pieces that may not have coexisted.
//!
//! That is not something this module can close: doing so needs every other
//! processor stopped, which is a decision about the whole machine rather than
//! about one exit. What it can do is not pretend otherwise —
//! [`Provenance`](crate::Provenance) is checked against the decode afterwards,
//! so a decode that came out differently from what faulted is caught by
//! disagreeing with the fault rather than by being trusted. An instruction that
//! agrees with the reported address, direction and width is the instruction
//! that trapped, on any reading of those bytes.
//!
//! # Why the tables are built before any guest runs
//!
//! The decoder's tables are built once, on first use, and building them takes
//! several thousand heap allocations. On a guest's first intercepted access
//! that would be a long pause at the worst moment, and on a short heap it would
//! be an allocation failure inside an exit handler — which is a panic, on the
//! one path in this hypervisor that must not have one. So [`warm`] decodes
//! something during bring-up, where a failure is a boot that stops with a
//! reason.

use iced_x86::{Decoder, DecoderError, DecoderOptions, Instruction};
use memory::{Addressing, Segment};

use crate::{
    EmulateError, Undecodable, as_u64, as_usize,
    machine::{Cpu, Guest},
};

/// Bytes in the longest instruction the architecture allows.
const LONGEST: usize = 15;

/// Bytes in the smallest page, which is where a fetch from the guest stops and
/// asks again.
const PAGE: u64 = 4096;

/// An instruction the decoder is certain about, so that the tables behind it
/// exist before anything time-critical needs them.
///
/// Returns what it decoded, which is worth logging: it says the decoder works
/// on this machine, and it is the only thing that will say so until a guest
/// runs.
///
/// # Errors
///
/// [`EmulateError::Undecodable`] if the decoder cannot read an encoding this
/// crate chose itself, which would mean the crate was built wrong rather than
/// anything about the machine.
pub(crate) fn warm() -> Result<usize, EmulateError> {
    /// `movdqu [rdi], xmm3`: a vector move to memory, which is the shape of the
    /// hardest thing this crate emulates and reaches the widest part of the
    /// decode tables on the way in.
    const SAMPLE: [u8; 5] = [0xF3, 0x0F, 0x7F, 0x1F, 0x90];

    let mut decoder = Decoder::with_ip(64, &SAMPLE, 0, DecoderOptions::NONE);
    let instruction = decoder.decode();
    if decoder.last_error() != DecoderError::None {
        return Err(EmulateError::Undecodable {
            rip: 0,
            bytes: SAMPLE.len(),
            reason: reason(decoder.last_error()),
        });
    }
    Ok(instruction.len())
}

/// The instruction the guest stopped on.
///
/// # Errors
///
/// [`EmulateError::Undecodable`] if the bytes available are not a complete
/// instruction, or [`EmulateError::Memory`] if they could not be read out of
/// the guest at all.
pub(crate) fn instruction(cpu: &impl Cpu, guest: &impl Guest) -> Result<Instruction, EmulateError> {
    let mut buffer = [0; LONGEST];
    let bytes = fetch(cpu, guest, &mut buffer)?;
    let rip = cpu.save().rip;
    let mut decoder = Decoder::with_ip(
        bitness(guest.addressing()),
        bytes,
        rip,
        DecoderOptions::NONE,
    );
    let mut instruction = Instruction::default();
    decoder.decode_out(&mut instruction);
    if decoder.last_error() != DecoderError::None {
        return Err(EmulateError::Undecodable {
            rip,
            bytes: bytes.len(),
            reason: reason(decoder.last_error()),
        });
    }
    Ok(instruction)
}

/// What the decoder objected to, kept rather than flattened.
///
/// The difference matters to whoever reads the log. Running out of bytes is
/// ordinary — it is what a fetch stopped at a page boundary reports when the
/// instruction continues on the next page, and it says the fetch was too short.
/// An invalid encoding says the opposite: the bytes were there and they are not
/// an instruction, which is a guest executing something it should not or a
/// fetch that read the wrong address entirely.
///
/// The decoder has no third complaint. An encoding longer than the architecture
/// allows is reported as an invalid one, so [`Undecodable::Overlong`] comes
/// from the length check in [`crate::plan::after`] rather than from here.
const fn reason(error: DecoderError) -> Undecodable {
    match error {
        DecoderError::NoMoreBytes => Undecodable::Incomplete,
        // `None` cannot reach here — it is checked for before this is called —
        // and `InvalidInstruction` is the remaining case the decoder defines. The
        // catch-all is for a complaint a later version of the decoder might add,
        // which is by definition a decode that did not succeed.
        _ => Undecodable::Invalid,
    }
}

/// How wide the instruction's operands and addresses default to.
///
/// The mode alone does not answer this. A long-mode guest running compatibility
/// code has a code segment that says 32-bit or 16-bit, and decoding its
/// instructions as 64-bit would read a REX prefix out of an opcode.
fn bitness(addressing: &Addressing) -> u32 {
    if addressing.long_mode() {
        64
    } else if addressing.default_size() {
        32
    } else {
        16
    }
}

/// The instruction's bytes, from the control block if the processor left them
/// there and from the guest otherwise.
fn fetch<'a>(
    cpu: &impl Cpu,
    guest: &impl Guest,
    buffer: &'a mut [u8; LONGEST],
) -> Result<&'a [u8], EmulateError> {
    let fetched = cpu.fetched();
    if !fetched.is_empty() {
        let taken = fetched.len().min(LONGEST);
        buffer[..taken].copy_from_slice(&fetched[..taken]);
        return Ok(&buffer[..taken]);
    }

    let at = guest
        .addressing()
        .base(Segment::Cs)
        .wrapping_add(cpu.save().rip);
    // As far as the end of the page and no further, because that much has to be
    // there — the guest is executing out of it — and the next page need not be.
    let head = as_usize(PAGE - at % PAGE).min(LONGEST);
    guest.read(at, &mut buffer[..head])?;
    if head == LONGEST {
        return Ok(buffer);
    }
    // The rest, if the guest has it. An instruction that really does straddle the
    // boundary needs these bytes and one that does not will never look at them, so
    // failing here is only worth reporting if the decode also fails.
    let rest = at.wrapping_add(as_u64(head));
    if guest.read(rest, &mut buffer[head..]).is_ok() {
        return Ok(buffer);
    }
    Ok(&buffer[..head])
}

#[cfg(test)]
mod tests {
    use iced_x86::{Code, DecoderError, Register};

    use super::{instruction, reason, warm};
    use crate::{
        EmulateError, Undecodable,
        machine::{
            Cpu,
            tests::{Machine, Memory},
        },
    };

    /// `mov rax, [rdx]` — seven bytes with a REX prefix, so that decoding it in
    /// the wrong mode produces something visibly different rather than nothing.
    const MOV_RAX_MEM: [u8; 3] = [0x48, 0x8B, 0x02];

    #[test]
    fn the_decoder_reads_its_own_sample_before_a_guest_exists() {
        assert_eq!(warm().expect("the crate's own encoding must decode"), 4);
    }

    #[test]
    fn an_instruction_comes_out_of_the_control_block_when_the_processor_left_it_there() {
        let machine = Machine::long_mode().at(0x1000).with_fetched(&MOV_RAX_MEM);
        // Deliberately no guest memory described at all: if this read the guest
        // rather than the control block, it would fail to translate.
        let memory = Memory::new(&machine);
        let decoded = instruction(&machine, &memory).expect("the bytes are in the control block");
        assert_eq!(decoded.code(), Code::Mov_r64_rm64);
        assert_eq!(decoded.op0_register(), Register::RAX);
        assert_eq!(decoded.len(), MOV_RAX_MEM.len());
    }

    #[test]
    fn an_instruction_comes_out_of_the_guest_when_it_did_not() {
        let machine = Machine::long_mode().at(0x1000);
        let mut memory = Memory::new(&machine);
        memory.fill(0x1000, &MOV_RAX_MEM);
        let decoded = instruction(&machine, &memory).expect("the bytes are in the guest");
        assert_eq!(decoded.code(), Code::Mov_r64_rm64);
    }

    #[test]
    fn the_code_segment_base_is_added_to_the_pointer_outside_long_mode() {
        let mut machine = Machine::protected().at(0x1000);
        machine.save_mut().cs.base = 0x2_0000;
        let mut memory = Memory::new(&machine);
        // The instruction is at CS.base + RIP rather than at RIP.
        memory.fill(0x2_1000, &[0x8B, 0x02]);
        let decoded = instruction(&machine, &memory).expect("the fetch must add the base");
        assert_eq!(decoded.code(), Code::Mov_r32_rm32);
    }

    #[test]
    fn a_fetch_stops_at_the_end_of_a_page_and_carries_on_if_it_can() {
        let machine = Machine::long_mode().at(0xFFE);
        let mut memory = Memory::new(&machine);
        // Two bytes on one page and one on the next: an instruction that really
        // does straddle the boundary.
        memory.fill(0xFFE, &MOV_RAX_MEM);
        let decoded = instruction(&machine, &memory).expect("both pages are described");
        assert_eq!(decoded.code(), Code::Mov_r64_rm64);
        assert_eq!(decoded.len(), 3);
    }

    #[test]
    fn an_instruction_at_a_page_end_decodes_without_the_next_page() {
        // One byte is a whole instruction, so the unreadable next page must not
        // stop it — which is why the fetch reports what it got.
        let machine = Machine::long_mode().at(0xFFF);
        let mut memory = Memory::new(&machine);
        memory.map(0);
        memory.fill(0xFFF, &[0x90]);
        let decoded = instruction(&machine, &memory).expect("a one-byte instruction fits");
        assert_eq!(decoded.len(), 1);
        assert!(decoded.is_invalid() || decoded.code() == Code::Nopd || decoded.len() == 1);
    }

    #[test]
    fn an_instruction_running_off_the_readable_bytes_reports_incompleteness() {
        let machine = Machine::long_mode().at(0xFFF);
        let mut memory = Memory::new(&machine);
        memory.map(0);
        // A REX prefix at the last byte of the page, with the next page absent:
        // the instruction continues where nothing can be read.
        memory.fill(0xFFF, &[0x48]);
        let error = instruction(&machine, &memory).expect_err("the instruction is cut off");
        assert!(
            matches!(
                error,
                EmulateError::Undecodable {
                    reason: Undecodable::Incomplete,
                    ..
                }
            ),
            "a truncated instruction is incomplete input, not a bad encoding: {error:?}"
        );
    }

    #[test]
    fn bytes_that_are_not_an_instruction_are_reported_as_invalid_rather_than_short() {
        // The distinction the old code lost. This is not a fetch that was too
        // short — there are bytes and they were read — it is a guest executing
        // something that is not an instruction, which a log has to be able to
        // tell apart from an instruction continuing onto an absent page.
        let machine = Machine::long_mode()
            .at(0x1000)
            .with_fetched(&[0xFF, 0xFF, 0xFF]);
        let memory = Memory::new(&machine);
        let error = instruction(&machine, &memory).expect_err("not an encoding");
        assert!(
            matches!(
                error,
                EmulateError::Undecodable {
                    reason: Undecodable::Invalid,
                    ..
                }
            ),
            "an unencodable byte string is invalid, not incomplete: {error:?}"
        );
    }

    #[test]
    fn a_deliberately_undefined_instruction_decodes_as_itself() {
        // `UD2` is a valid encoding of an invalid instruction, which is not a
        // decode failure: the decoder's job is to say what the bytes are, and
        // refusing to perform it belongs to the accepted-code table instead.
        let machine = Machine::long_mode().at(0x1000).with_fetched(&[0x0F, 0x0B]);
        let memory = Memory::new(&machine);
        let decoded = instruction(&machine, &memory).expect("UD2 is a valid encoding");
        assert_eq!(decoded.code(), Code::Ud2);
        assert_eq!(
            crate::mov::form(decoded.code()),
            None,
            "it is still not a move"
        );
    }

    #[test]
    fn a_guest_whose_instruction_page_is_absent_reports_a_memory_failure() {
        let machine = Machine::long_mode().at(0x1000);
        let memory = Memory::new(&machine);
        let error = instruction(&machine, &memory).expect_err("nothing is described");
        assert!(
            matches!(error, EmulateError::Memory(_)),
            "an unreadable instruction page is a memory failure: {error:?}"
        );
    }

    #[test]
    fn each_decoder_complaint_keeps_its_own_identity() {
        assert_eq!(reason(DecoderError::NoMoreBytes), Undecodable::Incomplete);
        assert_eq!(
            reason(DecoderError::InvalidInstruction),
            Undecodable::Invalid
        );
    }

    #[test]
    fn each_mode_decodes_at_its_own_width() {
        // The same two bytes are a 32-bit move in protected mode and a 16-bit one
        // in real mode, which is why the mode is read from the code segment
        // rather than assumed.
        let protected = Machine::protected().at(0x100).with_fetched(&[0x8B, 0x02]);
        let decoded = instruction(&protected, &Memory::new(&protected)).expect("protected");
        assert_eq!(decoded.code(), Code::Mov_r32_rm32);

        let real = Machine::real().at(0x100).with_fetched(&[0x8B, 0x02]);
        let decoded = instruction(&real, &Memory::new(&real)).expect("real");
        assert_eq!(decoded.code(), Code::Mov_r16_rm16);

        let long = Machine::long_mode().at(0x100).with_fetched(&MOV_RAX_MEM);
        let decoded = instruction(&long, &Memory::new(&long)).expect("long");
        assert_eq!(decoded.code(), Code::Mov_r64_rm64);
    }
}
