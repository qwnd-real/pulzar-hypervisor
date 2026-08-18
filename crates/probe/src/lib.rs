//! Reading and writing one of the machine's model-specific registers without
//! being stopped by one it does not have.
//!
//! Every access a guest makes to a register outside the three ranges its
//! permission map covers is intercepted whatever that map says — there is no
//! bit to clear for those — so the host answers them or the guest does not run.
//! Answering means reaching the machine's own register, and that is where the
//! problem this crate exists for starts: most of the four billion indices a
//! guest can name are not registers at all, and `RDMSR` on one of those is a
//! general protection fault rather than a value.
//!
//! Which indices exist cannot be decided in software. The set is per model and
//! in places undocumented, and a table of it written out by hand would be wrong
//! in both directions — refusing registers the machine has, forwarding accesses
//! to registers it has not. The processor is the only authority on the question
//! and the only way it answers is by faulting, so the access is attempted and
//! the fault is caught.
//!
//! # Catching one
//!
//! Each of the two instructions that can fault is written into a routine of its
//! own, with the instruction at an exported label and a second label just past
//! the routine's answer. A handler on the general protection vector compares
//! the address the processor reported against those two instructions: a match
//! means the fault is one this crate asked for, and the interrupted routine is
//! sent to its own recovery label instead of retrying what faulted.
//!
//! Nothing is remembered between the attempt and the fault. The pairing is two
//! addresses fixed when the image was linked, so there is no per-processor
//! state to keep, nothing to publish before the instruction or clear after it,
//! and two processors probing at once cannot see each other's attempts. The
//! recovery label sits inside the routine that faulted with nothing of its own
//! pushed below it, which is what lets recovery be an ordinary return: the
//! stack at the faulting instruction is still the stack the routine was called
//! on.
//!
//! # Why the assembly is written out
//!
//! [`x86_64`] wraps both instructions already, and a wrapper is the one thing
//! that cannot be used here: what the handler needs is the *address* of the
//! faulting instruction, as a symbol it can compare against. No wrapper offers
//! one. Writing the two routines out is what makes that address exist.
//!
//! # Installing comes first
//!
//! Until [`install`] has claimed the vector, a refused access is a fault
//! nothing recognises — it reaches whatever the hypervisor said becomes of an
//! unclaimed interrupt, which is a report and a stopped processor. Claiming it
//! belongs with the rest of bring-up, before any guest runs.

#![no_std]

use core::arch::naked_asm;

use descriptors::{DescriptorError, Disposition, Interrupt, Vector};
use log::info;
use thiserror::Error;
use x86_64::VirtAddr;

/// Claims the general protection vector, without which neither [`read`] nor
/// [`write`] may be used.
///
/// One claim for the machine rather than one per processor: what a fault at a
/// given address means is a fact about this image, and the registry it is
/// recorded in is shared. Every processor's own tables name the same entry
/// points and reach the same handler through them.
///
/// # Errors
///
/// [`DescriptorError::VectorTaken`] if something else has already claimed the
/// vector, which would mean two subsystems disagreeing about what a general
/// protection fault in this image means.
pub fn install() -> Result<(), DescriptorError> {
    descriptors::register(Vector::GENERAL_PROTECTION, recover)?;
    info!("probe: a register access the machine refuses will be reported rather than taken");
    Ok(())
}

/// What the machine's register `msr` holds.
///
/// # Errors
///
/// [`Fault`] if the processor refused the read, which for a read is the machine
/// saying it has no such register.
pub fn read(msr: u32) -> Result<u64, Fault> {
    let mut value = 0;
    // SAFETY: the routine reads the register its first argument names into the
    // `u64` its second points at, and the pointer is to a live, aligned local
    // nothing else can reach. A refused read faults on the instruction the
    // handler installed by `install` recognises, which is what makes this answer
    // `false` rather than end the processor.
    let read = unsafe { attempt_read(msr, &raw mut value) };
    read.then_some(value).ok_or(Fault { msr })
}

/// Writes `value` to the machine's register `msr`.
///
/// # Errors
///
/// [`Fault`] if the processor refused the write, which is either a register
/// this machine does not have or a value it does not allow in one it does.
pub fn write(msr: u32, value: u64) -> Result<(), Fault> {
    // SAFETY: as `read`, and the routine dereferences nothing at all: both of
    // its arguments are values.
    let written = unsafe { attempt_write(msr, value) };
    written.then_some(()).ok_or(Fault { msr })
}

/// An access the machine refused with a general protection fault.
///
/// Which of the two reasons it was — no such register, or a value that register
/// does not take — is not knowable from here, and does not matter to a caller
/// answering for a guest: the guest earns the same exception either way.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[error("the machine refused the access to register {msr:#x}")]
pub struct Fault {
    /// The index the refused access named.
    pub msr: u32,
}

/// Whether a general protection fault was one of this crate's own attempts.
fn recover(interrupt: &Interrupt) -> Disposition {
    match recovery(interrupt.frame().instruction_pointer) {
        Some(resume) => Disposition::Redirected(resume),
        // Every other general protection fault in this image is a real one, and
        // what becomes of it is not this crate's business.
        None => Disposition::Passed,
    }
}

/// Where the routine holding the instruction at `rip` carries on, if that
/// instruction is one of the two this crate is prepared to have fault.
///
/// The whole address rather than a range around it, which is what makes a match
/// mean something: the only instruction that can produce it is the one its
/// routine is written around.
fn recovery(rip: VirtAddr) -> Option<VirtAddr> {
    [
        (probe_read_fault as Label, probe_read_resume as Label),
        (probe_write_fault as Label, probe_write_resume as Label),
    ]
    .into_iter()
    .find(|&(fault, _)| address(fault) == rip)
    .map(|(_, resume)| address(resume))
}

/// Where a label is.
fn address(label: Label) -> VirtAddr {
    VirtAddr::from_ptr(label as *const ())
}

/// One of the labels the two routines export, taken as a symbol rather than as
/// something to call.
type Label = unsafe extern "sysv64" fn();

// The labels themselves, declared so that their addresses can be had. Calling
// one would be meaningless — the first is an instruction in the middle of a
// routine and the second is that routine's second answer — which is why nothing
// here does.
unsafe extern "sysv64" {
    /// The `RDMSR` a fault at is one this crate asked for.
    fn probe_read_fault();
    /// Where [`attempt_read`] carries on when that read faulted.
    fn probe_read_resume();
    /// The `WRMSR` a fault at is one this crate asked for.
    fn probe_write_fault();
    /// Where [`attempt_write`] carries on when that write faulted.
    fn probe_write_resume();
}

/// Reads the register `msr` names into `*value`, answering whether the machine
/// allowed it.
///
/// System V rather than the target's own C convention, which for the hypervisor
/// image is the Microsoft one: the index arrives in `EDI` and the destination
/// in `RSI` whichever platform this crate is built for, so the assembly is
/// written once and is right for the image and for a host build alike.
///
/// # Safety
///
/// `value` must point at a `u64` this call may write. The index needs no
/// vouching for: one the machine does not implement is the case this answers
/// `false` for rather than a way to go wrong.
#[unsafe(naked)]
unsafe extern "sysv64" fn attempt_read(msr: u32, value: *mut u64) -> bool {
    naked_asm!(
        "mov ecx, edi",

        // The read, at the address the handler recognises. `RDMSR` answers in
        // `EDX:EAX` with the upper half of each register cleared, so the value
        // is assembled out of the two halves rather than merely moved.
        ".globl probe_read_fault",
        "probe_read_fault:",
        "rdmsr",
        "shl rdx, 32",
        "or rax, rdx",
        "mov [rsi], rax",
        "mov eax, 1",
        "ret",

        // Where the handler sends this routine when the read faulted. Nothing of
        // this routine's own is on the stack, so the top of it is still the
        // caller's return address: answering is a zero and a return, and what
        // the caller passed a pointer to is left as it was.
        ".globl probe_read_resume",
        "probe_read_resume:",
        "xor eax, eax",
        "ret",
    )
}

/// Writes `value` to the register `msr` names, answering whether the machine
/// allowed it.
///
/// # Safety
///
/// As [`attempt_read`], less the pointer: both arguments are values, and
/// neither index nor value can make this do anything but answer. What the write
/// *means* is another matter — a register the machine has is really written,
/// and the caller owns that decision.
#[unsafe(naked)]
unsafe extern "sysv64" fn attempt_write(msr: u32, value: u64) -> bool {
    naked_asm!(
        // The index, and the value split the way `WRMSR` takes it: low half in
        // `EAX`, high half in `EDX`. Both come out of the one argument register,
        // which is copied twice rather than shifted in place.
        "mov ecx, edi",
        "mov eax, esi",
        "mov rdx, rsi",
        "shr rdx, 32",
        ".globl probe_write_fault",
        "probe_write_fault:",
        "wrmsr",
        "mov eax, 1",
        "ret",
        // As above: the recovery label is this routine's other answer, reached
        // with the stack exactly as the faulting instruction left it.
        ".globl probe_write_resume",
        "probe_write_resume:",
        "xor eax, eax",
        "ret",
    )
}

#[cfg(test)]
mod tests {
    use x86_64::VirtAddr;

    use super::{
        Label, address, attempt_read, attempt_write, probe_read_fault, probe_read_resume,
        probe_write_fault, probe_write_resume, recovery,
    };

    /// The pairing is what the whole mechanism rests on, and it is the one part
    /// of it a test can reach: the addresses are the linker's, and getting them
    /// crossed would send a faulted read into the write routine's answer.
    #[test]
    fn each_faulting_instruction_recovers_into_its_own_routine() {
        assert_eq!(
            recovery(address(probe_read_fault as Label)),
            Some(address(probe_read_resume as Label))
        );
        assert_eq!(
            recovery(address(probe_write_fault as Label)),
            Some(address(probe_write_resume as Label))
        );
    }

    /// A fault anywhere but at one of the two instructions is somebody else's —
    /// at a routine's first instruction, which cannot fault, or at a recovery
    /// label, which is not an attempt at anything. That is what makes the match
    /// an exact address rather than a routine to be anywhere inside of.
    #[test]
    fn an_address_no_attempt_can_fault_at_is_left_alone() {
        let read: unsafe extern "sysv64" fn(u32, *mut u64) -> bool = attempt_read;
        let write: unsafe extern "sysv64" fn(u32, u64) -> bool = attempt_write;
        assert_eq!(recovery(VirtAddr::from_ptr(read as *const ())), None);
        assert_eq!(recovery(VirtAddr::from_ptr(write as *const ())), None);
        assert_eq!(recovery(address(probe_read_resume as Label)), None);
        assert_eq!(recovery(address(probe_write_resume as Label)), None);
    }
}
