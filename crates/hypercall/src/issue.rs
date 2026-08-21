//! The guest's side of the call: the one instruction, and what it carries.
//!
//! `VMMCALL` is the whole mechanism. It has no privilege restriction, so this
//! is as usable from an ordinary process as from a kernel, and nothing here
//! needs to know which it is in.
//!
//! # What happens where there is no pulzar underneath
//!
//! The instruction is not one every machine has. A processor with no
//! virtualization extension enabled, and any hypervisor that does not intercept
//! it, raises an invalid-opcode exception — which reaches a hosted caller as
//! the signal its operating system delivers for one, and ends the process.
//! There is no way around it from inside the guest: pulzar deliberately answers
//! `CPUID` as though nothing were underneath the guest, so no feature bit
//! exists to ask first. A caller that might not be running under this
//! hypervisor has to be prepared for the fault, and every caller here is a
//! debugging tool run on purpose.
//!
//! A hypervisor that *does* intercept the instruction and is not this one sees
//! a command word whose top doubleword is [`SELECTOR`](crate::SELECTOR) — no
//! small number any other interface assigns — and answers with something this
//! crate reports as [`Answer::Unknown`] rather than reading as a status of its
//! own.

use core::arch::asm;

use crate::{ApicDump, Arguments, Command, Status};

/// Asks for everything the calling processor's interrupt controllers hold.
///
/// The safe way in, and the only one a reader needs: the buffer is a borrow of
/// a real structure, so its address and its capacity are the two things the
/// host has to be told and neither can be got wrong. What comes back says
/// whether the buffer was filled in; [`ApicDump::validate`] is what says it
/// holds a dump this build reads.
///
/// The caller's own bytes are left alone by every refusal. A host that will not
/// serve the call writes nothing at all.
pub fn apic_dump(into: &mut ApicDump) -> Answer {
    let arguments = Arguments::buffer(
        core::ptr::from_mut(into) as u64,
        size_of::<ApicDump>() as u64,
    );
    // SAFETY: the arguments name the borrowed structure itself, so the address
    // is memory the caller owns for exactly the capacity given, and it is
    // borrowed mutably for the whole of the call — nothing else may be reading
    // it while the host writes it. Every bit pattern of the structure is a
    // valid value of it, so whatever the host leaves behind is a value the
    // caller may go on to read.
    unsafe { issue(Command::APIC_DUMP, arguments) }
}

/// Issues `command` with `arguments`, and answers with what came back.
///
/// The register convention in one place: the command word in `RAX`, the two
/// argument words in `RDI` and `RSI`, the status back in `RAX`. Nothing else is
/// read and nothing else is written, and the assembly claims no more than that
/// — in particular it does not promise to leave the stack alone, because the
/// exception a machine with no hypervisor raises for this instruction is
/// delivered on the caller's own stack.
///
/// `RBX` would be the conventional third register of such a convention and is
/// deliberately not used: Rust's inline assembly reserves it, so no operand can
/// name it and a caller would have to move it by hand around every call.
///
/// # Safety
///
/// Both argument words describe a buffer the host will write: the first is its
/// address, in the caller's own linear addresses, and the second is how many
/// bytes of it the host may write. The address must name memory the caller owns
/// for at least that many bytes, and nothing else may be reading it for the
/// duration of the call — the host writes it while the caller is stopped inside
/// the instruction. The bytes written are the command's own structure, every
/// bit pattern of which must be a valid value of whatever the caller holds
/// there.
#[must_use]
pub unsafe fn issue(command: Command, arguments: Arguments) -> Answer {
    let answered: u64;
    // SAFETY: the caller guarantees the buffer the argument words name. The
    // block reads and writes memory as far as the compiler is concerned, which
    // is what keeps the buffer from being cached across it, and clobbers
    // nothing beyond the one output register.
    unsafe {
        asm!(
            "vmmcall",
            inlateout("rax") command.word() => answered,
            in("rdi") arguments.first,
            in("rsi") arguments.second,
        );
    }
    Answer::of(answered)
}

/// What a hypercall answered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Answer {
    /// A status this interface defines.
    Status(Status),
    /// A word this interface does not define, carried as it came back.
    ///
    /// Which is what a caller gets from a hypervisor that intercepts the
    /// instruction and is not this one, and from a pulzar built with statuses
    /// this reader does not have.
    Unknown(u64),
}

impl Answer {
    /// What a returned word says.
    #[must_use]
    pub const fn of(word: u64) -> Self {
        match Status::from_word(word) {
            Some(status) => Self::Status(status),
            None => Self::Unknown(word),
        }
    }

    /// Whether the call was served.
    #[must_use]
    pub const fn served(self) -> bool {
        matches!(self, Self::Status(Status::Ok))
    }
}
