//! Performing a guest's instruction on its behalf.
//!
//! Two things a hypervisor needs from an instruction it intercepted, and they
//! are not the same thing. Usually it only needs to know how long the
//! instruction was, so that the guest can be resumed after it —
//! [`next_rip`] answers that, and answers it without decoding anything at all
//! where the processor already has. Sometimes it needs the instruction actually
//! carried out, but against something other than what the guest aimed it at —
//! [`Mmio::dispatch`] does that, and is what every device and shadow-device
//! emulator will be built on.
//!
//! # What is emulated, and what is refused
//!
//! Moves. Every width and encoding of them: the general-purpose forms, the two
//! widening ones, the vector ones, the ones that move between a vector register
//! and a general-purpose one, and the string forms with their repeat prefix.
//!
//! Nothing else, and nothing else is approximated either. A read-modify-write
//! instruction against a device register — `or dword [rax], 1` — is a perfectly
//! ordinary thing for a driver to do and is *not* a move, so it is refused by
//! name rather than performed as one. Emulating it as a move would put a wrong
//! value in a device register and leave the guest's flags describing an
//! operation that never happened, and a guest carries on from that quite
//! happily and quite wrongly. Refusing loudly is the only honest answer until
//! the family is implemented properly.
//!
//! Length decoding is unaffected by any of that: [`next_rip`] answers for every
//! instruction the machine can execute, including the ones this crate will not
//! perform.
//!
//! # This must be installed before a guest runs
//!
//! [`install`] is not optional and not merely tidy. The decoder builds its
//! tables on first use and building them takes several thousand heap
//! allocations — which inside an exit handler is a long pause at the worst
//! possible moment, and on a short heap is an allocation failure on the one
//! path in this hypervisor that must not panic. Doing it during bring-up turns
//! both of those into a boot that stops with a reason.

#![no_std]

extern crate alloc;

mod decode;
mod gpr;
mod mmio;
mod mov;
mod operand;
mod string;
mod value;
mod xmm;

use iced_x86::Instruction;
use log::info;
use memory::{Linear, MemoryError};
use processor::SvmFeatures;
use svm::{Reason, exit::NestedPageFault};
use thiserror::Error;
use vcpu::Vcpu;
use x86_64::PhysAddr;

pub use crate::{
    mmio::{Commit, Device, Mmio, MmioError, Read, Region, Registrar, Trap, Write},
    value::{Data, Width},
};

/// Counts and lengths are `usize` while addresses are `u64`, and the two meet
/// wherever an instruction's length becomes part of an address. That is
/// lossless exactly while they are the same width, which this crate's only
/// target guarantees.
const _: () = assert!(
    size_of::<usize>() == size_of::<u64>(),
    "this crate assumes 64-bit pointers"
);

/// A length as `u64`.
const fn as_u64(value: usize) -> u64 {
    value as u64
}

/// A length as `usize`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "usize is 64 bits wide on this crate's only target, asserted above"
)]
const fn as_usize(value: u64) -> usize {
    value as usize
}

/// Prepares this processor to emulate, and says what it found.
///
/// Two things, both of which have to happen before a guest exists. The
/// decoder's tables are built, so that no exit ever pays for building them. And
/// the processor is checked for the vector support a sixteen-byte move needs,
/// and given the one bit of it that is the hypervisor's to grant.
///
/// # Errors
///
/// [`EmulateError::NoVectors`] if the processor is set to trap vector
/// instructions rather than execute them, or [`EmulateError::Undecodable`] if
/// the decoder cannot read an encoding this crate chose itself — which would
/// mean the crate was built wrong rather than anything about the machine.
pub fn install() -> Result<(), EmulateError> {
    xmm::available()?;
    let bytes = decode::warm()?;
    info!("emulate: decoder ready, its own sample instruction read as {bytes} bytes");
    match processor::svm() {
        Some(svm) if svm.features.contains(SvmFeatures::NEXT_RIP) => {
            info!("emulate: the processor supplies the address after an intercepted instruction");
        }
        // Not a failure. It means every step over an instruction costs a decode
        // rather than a field read, which is exactly what this crate is for.
        _ => info!("emulate: the processor supplies no next instruction address; decoding instead"),
    }
    Ok(())
}

/// Where the guest resumes after the instruction that caused this exit.
///
/// The processor usually knows and says so, and where it does this is a field
/// read. Two guards decide whether to believe it: the processor has to report
/// the feature at all, and the field has to hold something that could be an
/// answer. For a nested page fault it does not — the processor does not fill it
/// in for those, so it holds either zero or whatever the previous exit left
/// there, and a hypervisor that trusted it would resume the guest at the
/// instruction before last.
///
/// Failing those, the instruction is decoded and its length added, which is the
/// answer for every exit and every instruction.
///
/// # Errors
///
/// [`EmulateError::Undecodable`] if the bytes available are not a whole
/// instruction, or [`EmulateError::Memory`] if they cannot be read out of the
/// guest.
pub fn next_rip(vcpu: &Vcpu, guest: Linear<'_>) -> Result<u64, EmulateError> {
    if let Some(supplied) = supplied(vcpu) {
        return Ok(supplied);
    }
    let instruction = decode::instruction(vcpu, guest)?;
    Ok(vcpu.save().rip.wrapping_add(as_u64(instruction.len())))
}

/// What became of the instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Done. The instruction pointer has been advanced past it.
    Stepped,
    /// A repeated instruction with repetitions left. The instruction pointer is
    /// deliberately unchanged, so the guest executes it again — with the index
    /// and count registers where this batch left them — and the next exit
    /// carries the copy on.
    Repeating,
}

impl Mmio {
    /// Performs the instruction behind a nested page fault in a trapped region.
    ///
    /// The instruction pointer is advanced here rather than by the caller,
    /// because whether it should be advanced at all is part of the answer: a
    /// repeated move that stopped at a page boundary has to be executed again.
    /// Nothing else about the control block is touched, and nothing needs
    /// [`Vcpu::soil`] — no clean-bit group covers the instruction pointer, the
    /// flags, the stack pointer or the accumulator, so the processor reloads
    /// them whatever this says.
    ///
    /// # Errors
    ///
    /// [`EmulateError::FetchFault`] if the fault was the instruction fetch
    /// itself, [`EmulateError::NotTrapped`] if no operand turned out to be in a
    /// region anything answers for, [`EmulateError::Unsupported`] for an
    /// instruction outside the move family, and whatever decoding the
    /// instruction or reaching either end of it reports.
    pub fn dispatch(
        &self,
        vcpu: &mut Vcpu,
        guest: Linear<'_>,
        gpa: PhysAddr,
        cause: NestedPageFault,
    ) -> Result<Outcome, EmulateError> {
        if cause.instruction_fetch() {
            // The fetch itself faulted, so there is no instruction to perform
            // and no length to compute. Whatever is wrong is wrong about the
            // mapping the guest is executing from, which is not this crate's to
            // put right.
            return Err(EmulateError::FetchFault { gpa: gpa.as_u64() });
        }
        let instruction = decode::instruction(vcpu, guest)?;
        let outcome = self.perform(vcpu, guest, &instruction, gpa)?;
        if outcome == Outcome::Stepped {
            let rip = vcpu.save().rip.wrapping_add(as_u64(instruction.len()));
            vcpu.save_mut().rip = rip;
        }
        Ok(outcome)
    }

    /// The instruction itself, once it is known what it is.
    fn perform(
        &self,
        vcpu: &mut Vcpu,
        guest: Linear<'_>,
        instruction: &Instruction,
        gpa: PhysAddr,
    ) -> Result<Outcome, EmulateError> {
        let rip = vcpu.save().rip;
        if mov::repeats(instruction) {
            let width = mov::repetition(instruction).ok_or(EmulateError::Width { rip })?;
            return string::perform(self, vcpu, guest, instruction, width);
        }
        mov::perform(self, vcpu, guest, instruction)?;
        // Checked after the fact rather than before, because which operand is
        // interposed on is only known once both have been worked out — and
        // working them out is most of performing the move. Reaching here having
        // touched no trapped region means this exit was not what it appeared to
        // be, which is worth saying rather than stepping quietly over.
        if !touched(self, vcpu, guest, instruction) {
            return Err(EmulateError::NotTrapped { gpa: gpa.as_u64() });
        }
        Ok(Outcome::Stepped)
    }
}

/// Whether either end of the move was a region something answers for.
fn touched(mmio: &Mmio, vcpu: &Vcpu, guest: Linear<'_>, instruction: &Instruction) -> bool {
    (0..instruction.op_count()).any(|operand| {
        operand::place(mmio, vcpu, guest, instruction, operand)
            .is_ok_and(operand::Place::interposed)
    })
}

/// The address after the instruction, where the processor supplied one.
fn supplied(vcpu: &Vcpu) -> Option<u64> {
    if vcpu.reason() == Some(Reason::NestedPageFault) {
        return None;
    }
    if !processor::svm()?.features.contains(SvmFeatures::NEXT_RIP) {
        return None;
    }
    let next = vcpu.control().next_rip;
    // Zero is what the field holds when nothing wrote it, and a copy of the
    // instruction pointer is what it holds when something wrote it for an
    // instruction that is not this one. Neither is an address after anything.
    (next != 0 && next != vcpu.save().rip).then_some(next)
}

/// Why an instruction could not be performed.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum EmulateError {
    /// The processor will not execute the vector instructions a sixteen-byte
    /// move needs.
    #[error("this processor is set to trap vector instructions rather than execute them")]
    NoVectors,
    /// The bytes available are not a whole instruction.
    #[error("the {bytes} bytes at {rip:#x} are not a complete instruction")]
    Undecodable {
        /// Where the guest stopped.
        rip: u64,
        /// How many bytes there were to read.
        bytes: usize,
    },
    /// The instruction is not one this crate performs.
    #[error("{mnemonic} at {rip:#x} is not a move, and is not emulated")]
    Unsupported {
        /// Where the guest stopped.
        rip: u64,
        /// What it was doing.
        mnemonic: &'static str,
    },
    /// The fault was the instruction fetch itself, so there is no instruction.
    #[error("the fault at guest physical {gpa:#x} was the instruction fetch")]
    FetchFault {
        /// The address that faulted.
        gpa: u64,
    },
    /// No operand of the instruction was in a region anything answers for.
    #[error("nothing interposes on guest physical {gpa:#x}, so there was nothing to emulate")]
    NotTrapped {
        /// The address that faulted.
        gpa: u64,
    },
    /// The effective address could not be worked out from the registers the
    /// guest stopped with.
    #[error("the address an operand of the instruction at {rip:#x} names cannot be computed")]
    Address {
        /// Where the guest stopped.
        rip: u64,
    },
    /// An operand is of a kind no move this crate performs has.
    #[error("the instruction at {rip:#x} has an operand no move has")]
    Operand {
        /// Where the guest stopped.
        rip: u64,
    },
    /// The destination decoded as an immediate, which no move has.
    #[error("the instruction at {rip:#x} decoded with an immediate destination")]
    ImmediateDestination {
        /// Where the guest stopped.
        rip: u64,
    },
    /// The instruction moves a number of bytes no move moves.
    #[error("the instruction at {rip:#x} moves a width no move has")]
    Width {
        /// Where the guest stopped.
        rip: u64,
    },
    /// An operand named a register this crate cannot reach, or one no move is
    /// that wide.
    #[error("an operand named a {width}-byte register that is not one a move reaches")]
    Register {
        /// How wide the register was.
        width: usize,
    },
    /// The access is not one a device can be asked to answer: it begins inside
    /// a region and ends past it, or it is not aligned to its own width.
    ///
    /// Neither is something a device would ever see from real hardware, so
    /// neither is something to pass on to one.
    #[error("a {bytes}-byte access at guest physical {gpa:#x} is not one a device can answer")]
    Inadmissible {
        /// Where the access begins.
        gpa: u64,
        /// How long it is.
        bytes: usize,
    },
    /// A region was named that no longer exists.
    #[error("there is no interposed region {index}")]
    NoSuchRegion {
        /// Which was named.
        index: usize,
    },
    /// The guest's memory could not be reached.
    #[error(transparent)]
    Memory(#[from] MemoryError),
    /// A region could not be taken over.
    #[error(transparent)]
    Mmio(#[from] MmioError),
}
