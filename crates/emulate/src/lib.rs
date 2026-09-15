//! Performing a guest's instruction on its behalf.
//!
//! Two things a hypervisor needs from an instruction it intercepted, and they
//! are not the same thing. Usually it only needs to know how long the
//! instruction was, so that the guest can be resumed after it — [`next_rip`]
//! answers that, and answers it without decoding anything at all where the
//! processor already has. Sometimes it needs the instruction actually carried
//! out, but against something other than what the guest aimed it at —
//! [`Mmio::dispatch`] does that, and is what every device and shadow-device
//! emulator is built on.
//!
//! # What is emulated, and what is refused
//!
//! Moves, named one encoding at a time. Not one mnemonic at a time: a mnemonic
//! covers forms whose architectural effects differ, and `MOVSD` covers two
//! instructions that are not even the same family. So the accepted set is a
//! table of exact instruction encodings, each with the transfer width and the
//! destination policy that encoding actually has — see [`mov`].
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
//! # An instruction is planned before any of it is performed
//!
//! Every dispatch resolves both ends of the move from the register state the
//! guest stopped with, validates the whole plan against what the hardware
//! reported, and only then touches anything. That order is the reason the crate
//! is shaped the way it is:
//!
//! - Operands are resolved **once**. A move writes one of the registers its own
//!   address was computed from — `mov rax, [rax]` — so anything re-derived
//!   after the fact is derived from state that is no longer the state the
//!   instruction ran with.
//! - The access is checked against the fault the hardware actually reported:
//!   its direction, its address, and that it was the final access rather than a
//!   walk of the guest's own tables. An emulator that skips this performs a
//!   different access from the one that trapped, which a guest can arrange
//!   deliberately.
//! - Every byte of the access is accounted for, not just the first. A four-byte
//!   operand two bytes from the end of a page is two translations, possibly two
//!   regions, possibly one region and one page of ordinary memory.
//! - Everything that can fail is made to fail *before* the first side effect. A
//!   device read cannot be taken back, so nothing fallible may follow one — and
//!   where something irreversible has happened anyway, the outcome says so
//!   rather than looking like an error that can be retried.
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

#[cfg(test)]
mod dispatch;
mod gpr;
mod machine;
mod mmio;
mod mov;
mod operand;
mod plan;
mod string;
mod value;
mod xmm;

use log::info;
use memory::{Linear, MemoryError};
use processor::SvmFeatures;
use svm::{Reason, exit::NestedPageFault};
use thiserror::Error;
use vcpu::Vcpu;
use x86_64::PhysAddr;

pub use crate::{
    mmio::{
        Capability, Commit, Device, Hardware, Mmio, MmioError, Read, Region, Retired, Trap, Write,
    },
    plan::Fault,
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
    let bytes = mmio::decode::warm()?;
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
/// the feature at all, and the exit has to be one of the classes the
/// architecture fills the field in for.
///
/// A nested page fault is not one of them — the architecture resets the field
/// to zero for that class rather than describing the faulting instruction — so
/// it is never believed there, and the instruction is decoded instead.
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
    let instruction = mmio::decode::instruction(vcpu, &guest)?;
    plan::after(vcpu.save(), instruction.len()).ok_or(EmulateError::Undecodable {
        rip: vcpu.save().rip,
        bytes: instruction.len(),
        reason: Undecodable::Overlong,
    })
}

/// What became of the instruction.
///
/// The distinction between the last two is the whole of what makes a failure
/// safe to act on. An emulation that has changed nothing can be retried, or
/// reported, or turned into a fault, freely. One that has already consumed a
/// device read cannot: retrying it would read the device twice, and a
/// clear-on-read register answers the second read differently from the first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use = "an outcome says whether the guest advanced and whether anything is already committed"]
pub enum Outcome {
    /// Done. The instruction pointer has been advanced past it.
    Stepped,
    /// A repeated instruction with repetitions left, and every one of them so
    /// far complete. The instruction pointer is deliberately unchanged, so the
    /// guest executes it again — with the index and count registers where this
    /// batch left them — and the next exit carries the copy on.
    Repeating,
    /// The guest owes an architectural fault, which the instruction did not
    /// complete. Whatever progress the architecture keeps in registers is
    /// already recorded, and the instruction pointer is unchanged, so
    /// delivering the fault and later resuming re-executes exactly what is
    /// left.
    Faulted(Fault),
}

impl Mmio {
    /// Performs the instruction behind a nested page fault in a trapped region.
    ///
    /// The instruction pointer is advanced here rather than by the caller,
    /// because whether it should be advanced at all is part of the answer: a
    /// repeated move that stopped at a page boundary has to be executed again,
    /// and an instruction that faulted must be re-executed after the fault is
    /// delivered. Nothing else about the control block is touched, and nothing
    /// needs [`Vcpu::soil`] — no clean-bit group covers the instruction
    /// pointer, the flags, the stack pointer or the accumulator, so the
    /// processor reloads them whatever this says.
    ///
    /// # Errors
    ///
    /// [`EmulateError::FetchFault`] if the fault was the instruction fetch
    /// itself, [`EmulateError::Provenance`] if the decoded instruction does not
    /// account for the fault the hardware reported,
    /// [`EmulateError::NotTrapped`] if no part of the access is in a region
    /// anything answers for, [`EmulateError::Unsupported`] for an
    /// instruction outside the move family, and whatever decoding the
    /// instruction or reaching either end of it reports.
    pub fn dispatch(
        &self,
        vcpu: &mut Vcpu,
        guest: Linear<'_>,
        gpa: PhysAddr,
        cause: NestedPageFault,
    ) -> Result<Outcome, EmulateError> {
        self.execute(vcpu, &guest, gpa, cause)
    }
}

/// The address after the instruction, where the processor supplied one.
fn supplied(vcpu: &Vcpu) -> Option<u64> {
    if !nrip_valid(vcpu.reason()?) {
        return None;
    }
    if !processor::svm()?.features.contains(SvmFeatures::NEXT_RIP) {
        return None;
    }
    let next = vcpu.control().next_rip;
    // A copy of the instruction pointer is what the field holds when something
    // wrote it for an instruction that is not this one, which is not an address
    // after anything. Zero is accepted: for the classes that fill the field in,
    // zero is where a legacy-mode instruction at the top of the address space
    // really does resume.
    (next != vcpu.save().rip).then_some(next)
}

/// Whether the architecture fills in the next-instruction field for this class
/// of exit.
///
/// The field is only meaningful for intercepts of instructions the processor
/// decoded — and is explicitly reset to zero for the fault-like classes, of
/// which a nested page fault is one. Believing it there would resume the guest
/// at zero, or at whatever a previous exit left behind on a processor that does
/// not reset it.
const fn nrip_valid(reason: Reason) -> bool {
    !matches!(
        reason,
        Reason::NestedPageFault
            | Reason::Exception(_)
            | Reason::Interrupt
            | Reason::Nmi
            | Reason::Smi
            | Reason::Init
            | Reason::VirtualInterrupt
            | Reason::Shutdown
            | Reason::Invalid
            | Reason::Busy
            | Reason::IdleRequired
            | Reason::InvalidPmc
    )
}

/// Why an instruction could not be performed.
///
/// Every variant carries what an operator needs to identify the case from a log
/// line alone: which instruction, which register, which address, and which two
/// widths disagreed. A diagnostic that collapses two different failures into
/// one message is a diagnostic that will be misread.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum EmulateError {
    /// The processor will not execute the vector instructions a sixteen-byte
    /// move needs.
    #[error("this processor is set to trap vector instructions rather than execute them")]
    NoVectors,
    /// The bytes available are not a whole instruction.
    #[error("the {bytes} bytes at {rip:#x} are not a complete instruction: {reason}")]
    Undecodable {
        /// Where the guest stopped.
        rip: u64,
        /// How many bytes there were to read.
        bytes: usize,
        /// What the decoder objected to, which distinguishes an instruction
        /// that ran off the end of what could be read from one that is
        /// not an instruction at all.
        reason: Undecodable,
    },
    /// The instruction is not one this crate performs.
    #[error("{instruction} at {rip:#x} is not a move this crate performs")]
    Unsupported {
        /// Where the guest stopped.
        rip: u64,
        /// What it was doing.
        instruction: &'static str,
    },
    /// The fault was the instruction fetch itself, so there is no instruction.
    #[error("the fault at guest physical {gpa:#x} was the instruction fetch")]
    FetchFault {
        /// The address that faulted.
        gpa: u64,
    },
    /// The instruction that was decoded does not account for the fault the
    /// hardware reported.
    ///
    /// Not a decoding failure and not a device failure: the two disagree, and
    /// performing either one of them would be performing an access the
    /// processor did not make.
    #[error("the instruction at {rip:#x} does not account for the fault at {gpa:#x}: {reason}")]
    Provenance {
        /// Where the guest stopped.
        rip: u64,
        /// The address the hardware reported.
        gpa: u64,
        /// How the two disagree.
        reason: Provenance,
    },
    /// No part of the access was in a region anything answers for.
    #[error("nothing interposes on guest physical {gpa:#x}, so there was nothing to emulate")]
    NotTrapped {
        /// The address that faulted.
        gpa: u64,
    },
    /// The effective address could not be worked out from the registers the
    /// guest stopped with.
    #[error(
        "the address operand {operand} of the instruction at {rip:#x} names cannot be computed"
    )]
    Address {
        /// Where the guest stopped.
        rip: u64,
        /// Which operand.
        operand: u32,
    },
    /// An operand is of a kind no move this crate performs has.
    #[error("operand {operand} of the instruction at {rip:#x} is of a kind no move has")]
    Operand {
        /// Where the guest stopped.
        rip: u64,
        /// Which operand.
        operand: u32,
    },
    /// An operand named a register this crate cannot reach.
    #[error("an operand named {register}, of {bytes} bytes, which is not one a move reaches")]
    Register {
        /// What class of register it was.
        register: &'static str,
        /// How wide it was.
        bytes: usize,
    },
    /// Two ends of a transfer disagree about how many bytes it moves.
    ///
    /// Always a fault in whatever produced the value rather than something to
    /// resolve by truncating or widening: the width an instruction moves is
    /// decided by its encoding, and a value of another width means something
    /// above has already gone wrong.
    #[error("{register} takes {wanted:?} and was given {got:?}")]
    WidthMismatch {
        /// Which end was being written.
        register: &'static str,
        /// What it takes.
        wanted: Width,
        /// What it was given.
        got: Width,
    },
    /// The access is not one the device behind it can be asked to answer.
    #[error(
        "a {bytes}-byte access at guest physical {gpa:#x} is not one this device answers: {reason}"
    )]
    Inadmissible {
        /// Where the access begins.
        gpa: u64,
        /// How long it is.
        bytes: usize,
        /// Why the device cannot be asked.
        reason: Inadmissible,
    },
    /// The access does not lie within one region in one contiguous piece.
    #[error("a {bytes}-byte access at {linear:#x} is not one contiguous span: {reason}")]
    Span {
        /// Where the access begins, as the instruction named it.
        linear: u64,
        /// How long it is.
        bytes: usize,
        /// How it is broken up.
        reason: Spanning,
    },
    /// The destination would not accept the write, so the instruction did not
    /// complete — and must not be retired as though it had.
    #[error("the write of {bytes} bytes at {linear:#x} was discarded, so nothing was moved")]
    Discarded {
        /// Where the write was aimed.
        linear: u64,
        /// How long it was.
        bytes: usize,
    },
    /// A device read had already happened when something later failed.
    ///
    /// The one error that must never be retried: the device has answered, and a
    /// register that changes as it is read will not answer the same way twice.
    #[error(
        "{consumed} bytes were already read from guest physical {gpa:#x} when {cause} followed"
    )]
    Committed {
        /// Where the device read happened.
        gpa: u64,
        /// How much was taken from it.
        consumed: usize,
        /// What failed afterwards.
        cause: &'static str,
    },
    /// A repeated instruction failed after some of its repetitions had already
    /// completed.
    ///
    /// Distinct from the same failure with nothing behind it, because the index
    /// and count registers already describe work that was done: the
    /// instruction is partly performed, and whoever handles this must not
    /// treat it as an instruction that never ran.
    #[error("{completed} repetitions had already completed when {cause}")]
    Partial {
        /// How many repetitions are already committed, which the guest's index
        /// and count registers also reflect.
        completed: u32,
        /// What stopped the rest.
        cause: &'static str,
    },
    /// Nothing answers for the region an access falls in.
    ///
    /// A region the nested tables trap and nothing was registered for. The two
    /// decisions are independent, so this is a state that can be reached — and
    /// reporting it is the only honest answer, there being no device to ask.
    #[error("no device answers for region {region}")]
    NoDevice {
        /// Which region was named.
        region: u16,
    },
    /// The guest's memory could not be reached.
    #[error(transparent)]
    Memory(#[from] MemoryError),
    /// A region could not be taken over.
    #[error(transparent)]
    Mmio(#[from] MmioError),
}

/// What a decoder objected to.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum Undecodable {
    /// The bytes ran out part-way through an instruction. Not necessarily a
    /// malformed instruction: it is what a fetch stopped at a page boundary
    /// reports when the instruction continues on the next page.
    #[error("the instruction continues past the bytes that could be read")]
    Incomplete,
    /// The bytes are not an instruction the architecture defines.
    #[error("the bytes are not a valid encoding")]
    Invalid,
    /// The encoding is longer than the architecture allows one to be.
    #[error("the encoding is longer than an instruction may be")]
    Overlong,
}

/// How a decoded instruction and a reported fault disagree.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum Provenance {
    /// The fault was raised walking the guest's own page tables rather than on
    /// the address the guest was after. There is no data access to perform: the
    /// walk itself has to be resolved.
    #[error("the fault was a walk of the guest's own page tables")]
    PageTableWalk,
    /// The fault was not reported against the final address of the access, so
    /// which access it belongs to is not established.
    #[error("the fault was not reported against the address the access was aimed at")]
    NotFinal,
    /// The hardware reported a write and the instruction reads, or the reverse.
    #[error("the hardware reported a {reported} and the instruction performs a {decoded}")]
    Direction {
        /// What the hardware said.
        reported: &'static str,
        /// What the instruction does.
        decoded: &'static str,
    },
    /// No byte of either end of the instruction lands on the address the
    /// hardware reported.
    #[error("no part of the access covers the address the fault was reported at")]
    Elsewhere,
}

/// Why a device cannot be asked to answer an access.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum Inadmissible {
    /// The access begins inside the region and ends past the end of it.
    #[error("it runs past the end of the region")]
    PastEnd,
    /// The device does not answer accesses of this width.
    #[error("this device does not answer accesses of that width")]
    Width,
    /// The device requires accesses of this width to be aligned to it, and this
    /// one is not.
    #[error("this device requires that width to be aligned to itself")]
    Alignment,
    /// The device would have to see one indivisible sixteen-byte transaction,
    /// which this hypervisor cannot make without abandoning guest vector state
    /// across a faultable access.
    #[error("this device requires a single sixteen-byte transaction, which cannot be made safely")]
    Indivisible,
    /// A handler asked for a write to reach the hardware behind its region,
    /// having declared that it never touches it — so there is no mapping to
    /// perform the write through.
    #[error("this device declared that it never reaches the hardware behind its region")]
    Untouched,
    /// A handler answered with a value of a different width from the access it
    /// was asked about.
    #[error("the device answered {got:?} for a {wanted:?} access")]
    Answer {
        /// The width the guest used.
        wanted: Width,
        /// The width the device answered with.
        got: Width,
    },
}

/// How an access fails to be one contiguous span within one thing.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum Spanning {
    /// The bytes of the access do not translate to consecutive guest physical
    /// addresses, so no single transaction covers them.
    #[error("its bytes do not translate contiguously")]
    Discontiguous,
    /// Part of the access is in a trapped region and part is not.
    #[error("it is partly in an interposed region and partly in ordinary memory")]
    Straddles,
    /// The access spans two different trapped regions, which two different
    /// devices answer for.
    #[error("it spans two interposed regions")]
    TwoRegions,
    /// The address arithmetic left the address space.
    #[error("it leaves the address space")]
    Wraps,
}

#[cfg(test)]
mod tests {
    use svm::Reason;

    use super::nrip_valid;

    #[test]
    fn the_fault_like_exit_classes_supply_no_next_instruction_address() {
        // The architecture resets the field for these rather than describing the
        // instruction, so a hypervisor that believed it would resume the guest
        // at zero or at whatever the last exit left there.
        for reason in [
            Reason::NestedPageFault,
            Reason::Exception(descriptors::Vector::new(14)),
            Reason::Interrupt,
            Reason::Nmi,
            Reason::Smi,
            Reason::Init,
            Reason::VirtualInterrupt,
            Reason::Shutdown,
            Reason::Invalid,
        ] {
            assert!(
                !nrip_valid(reason),
                "{reason:?} must not be trusted for a next-instruction address"
            );
        }
    }

    #[test]
    fn the_instruction_intercepts_do_supply_one() {
        for reason in [
            Reason::Cpuid,
            Reason::Hlt,
            Reason::MsrAccess,
            Reason::PortAccess,
            Reason::Vmmcall,
            Reason::Invlpg,
            Reason::ReadControlRegister(0),
            Reason::WriteControlRegister(4),
            Reason::Rdtsc,
            Reason::Wbinvd,
        ] {
            assert!(
                nrip_valid(reason),
                "{reason:?} is an instruction intercept and does supply one"
            );
        }
    }
}
