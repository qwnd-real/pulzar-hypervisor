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
use memory::{Addressing, Linear, Segment};
use vcpu::Vcpu;

use crate::{EmulateError, as_u64, as_usize};

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
pub(crate) fn instruction(vcpu: &Vcpu, guest: Linear<'_>) -> Result<Instruction, EmulateError> {
    let mut buffer = [0; LONGEST];
    let bytes = fetch(vcpu, guest, &mut buffer)?;
    let rip = vcpu.save().rip;
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
        });
    }
    Ok(instruction)
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
    vcpu: &Vcpu,
    guest: Linear<'_>,
    buffer: &'a mut [u8; LONGEST],
) -> Result<&'a [u8], EmulateError> {
    let fetched = vcpu.control().fetched_instruction();
    if !fetched.is_empty() {
        let taken = fetched.len().min(LONGEST);
        buffer[..taken].copy_from_slice(&fetched[..taken]);
        return Ok(&buffer[..taken]);
    }

    let at = guest
        .addressing()
        .base(Segment::Cs)
        .wrapping_add(vcpu.save().rip);
    // As far as the end of the page and no further, because that much has to be
    // there — the guest is executing out of it — and the next page need not be.
    let head = as_usize(PAGE - at % PAGE).min(LONGEST);
    guest.read(at, &mut buffer[..head])?;
    if head == LONGEST {
        return Ok(buffer);
    }
    // The rest, if the guest has it. An instruction that really does straddle
    // the boundary needs these bytes and one that does not will never look at
    // them, so failing here is only worth reporting if the decode also fails.
    let rest = at.wrapping_add(as_u64(head));
    if guest.read(rest, &mut buffer[head..]).is_ok() {
        return Ok(buffer);
    }
    Ok(&buffer[..head])
}
