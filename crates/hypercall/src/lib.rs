//! The call a guest makes into the pulzar hypervisor, and what comes back.
//!
//! One instruction, `VMMCALL`, which the hypervisor intercepts. The
//! architecture puts no privilege restriction on it at all: a guest executes it
//! at any privilege level, and the exit reaches the host the same way from ring
//! zero and from ring three. So a debugging tool inside the guest needs no
//! driver — it issues the call itself, from an ordinary process, and this crate
//! is the whole of what both sides have to agree on.
//!
//! # The registers
//!
//! - `RAX` carries the command: a selector naming this interface, the version
//!   of it the caller was built against, and which command is being made. All
//!   three in one register, because the first thing the host has to decide
//!   about an intercepted `VMMCALL` is whether it belongs to this interface at
//!   all — the instruction has other users inside the same guest, and a word
//!   that happens to be in `RAX` must not be read as a hypercall.
//! - `RDI` carries the address of the buffer the answer is written to, as the
//!   caller's own linear address: the address a process has for its own memory,
//!   which the host translates through the guest's page tables. A guest cannot
//!   learn where its memory is in physical terms without asking its operating
//!   system, and the point of this interface is that it need not ask anything.
//! - `RSI` carries how many bytes of that buffer the host may write.
//! - `RAX` carries the [`Status`] back.
//!
//! Nothing else is read and nothing else is written. Every other register the
//! guest had is the guest's, and the flags are left as the instruction leaves
//! them.
//!
//! # Versioning
//!
//! [`VERSION`] travels in every command word, and a host that does not
//! implement it refuses the call with [`Status::Version`] rather than writing a
//! structure the caller would read at the wrong offsets. The buffer the host
//! writes then repeats the version in its own header, so a reader that kept a
//! dump can still tell what it is holding.
//!
//! # What this crate is not
//!
//! It has no dependency on any other part of pulzar, and it must not grow one.
//! Both ends need it: the hypervisor's exit path, which is `no_std` firmware
//! code, and a tool inside the guest, which is an ordinary hosted program. A
//! crate that reached into the hypervisor would drag the whole of it into the
//! tool.

#![no_std]

mod apic;
mod issue;

use core::{
    fmt::{self, Display, Formatter},
    slice,
};

pub use crate::{
    apic::{
        ABSENT, ApicDump, Avic, AvicFlags, BANK_SLOTS, DebtKind, Emulated, EmulatedFlags, Header,
        LOGICAL_ENTRIES, LVT_ENTRIES, Mismatch, Mode, PAGE_BYTES, PHYSICAL_ENTRIES, Present, Real,
        RealFlags, SOURCES, SourceState, Startup, TimerMode,
    },
    issue::{Answer, apic_dump, issue},
};

/// Names this interface in the command register.
///
/// A recognizable byte string rather than a small integer, for the reason the
/// portal's own notifications are: `VMMCALL` has other users inside the same
/// guest, and a command word that could be arrived at by accident is one whose
/// arrival would be answered as a hypercall.
pub const SELECTOR: u32 = u32::from_be_bytes(*b"PULZ");

/// The version of this interface both sides must agree on.
///
/// Bump it whenever a structure below changes meaning, moves a field, or grows
/// one: the tool and the hypervisor are built together but need not be *run*
/// together, and a mismatched pair must be refused at the call rather than
/// misread afterwards.
pub const VERSION: u16 = 2;

/// How the command word is divided.
const CODE_BITS: u32 = u16::BITS;

/// Where the version sits in it.
const VERSION_SHIFT: u32 = CODE_BITS;

/// Where the selector sits in it.
const SELECTOR_SHIFT: u32 = u32::BITS;

/// The alignment every copy-out buffer must have.
///
/// Every structure in this ABI is a `#[repr(C)]` layout of quadwords and
/// doublewords, so a buffer that is not quadword aligned is one whose fields a
/// reader could not name without an unaligned access. Refusing it is also the
/// cheapest sanity check there is on an address the host is about to be pointed
/// at.
pub const ALIGNMENT: u64 = 8;

/// Which command a guest is making.
///
/// A newtype over the whole command word rather than an enum, because the word
/// carries three things and only one of them is the command: the code names the
/// operation, and the selector and version around it are what make a word from
/// another user of the instruction impossible to mistake for one of these.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Command(u16);

impl Command {
    /// Everything the calling processor's interrupt controllers hold, written
    /// into the caller's buffer.
    ///
    /// The emulated controller the guest sees, the machine's own controller
    /// underneath it, and the structures the hardware delivers through where
    /// the processor is driving that controller itself.
    pub const APIC_DUMP: Self = Self(1);

    /// The command a code names, or `None` for one this interface does not
    /// define.
    #[must_use]
    pub const fn from_code(code: u16) -> Option<Self> {
        match code {
            code if code == Self::APIC_DUMP.0 => Some(Self::APIC_DUMP),
            _ => None,
        }
    }

    /// The word this command travels in the command register as.
    #[must_use]
    pub const fn word(self) -> u64 {
        ((SELECTOR as u64) << SELECTOR_SHIFT) | ((VERSION as u64) << VERSION_SHIFT) | self.0 as u64
    }

    /// The code that names this command.
    #[must_use]
    pub const fn code(self) -> u16 {
        self.0
    }

    /// How many bytes of buffer this command needs before it will be served.
    #[must_use]
    pub const fn bytes(self) -> u64 {
        match self {
            Self::APIC_DUMP => size_of::<ApicDump>() as u64,
            // Every command this interface defines is answered above; a value
            // that reached here would be one `from_code` had accepted and this
            // had not, which is a mistake in this crate rather than in a caller.
            _ => u64::MAX,
        }
    }
}

/// The two argument words a command carries.
///
/// Both of them describe the buffer the answer goes in, for every command this
/// interface has: where it is, and how much of it the host may write. They are
/// carried as a pair rather than as two parameters so that the register
/// convention is stated in one place and the issuer cannot put them in the
/// wrong order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arguments {
    /// The word in `RDI`.
    pub first: u64,
    /// The word in `RSI`.
    pub second: u64,
}

impl Arguments {
    /// The arguments naming a copy-out buffer at `address`, of `capacity`
    /// bytes.
    #[must_use]
    pub const fn buffer(address: u64, capacity: u64) -> Self {
        Self {
            first: address,
            second: capacity,
        }
    }
}

/// What an intercepted `VMMCALL` turns out to be.
///
/// Three outcomes, and the first is why this is not a `Result`: an instruction
/// this interface has nothing to do with is not a failed hypercall, and a host
/// that answered one as if it were would be stealing an instruction from
/// whatever else in the guest issues it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decoded {
    /// Not this interface's. The command register carries no selector, so
    /// whatever the instruction was for, it was not this.
    Foreign,
    /// A request this interface answers.
    Request(Request),
    /// This interface's, and refused before anything was done: the status the
    /// caller is owed.
    Refused(Status),
}

/// A validated request, as the host is to serve it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Request {
    /// What is being asked for.
    pub command: Command,
    /// Where the answer goes, as the calling guest's own linear address.
    pub buffer: u64,
    /// How many bytes of it the host may write, which is at least
    /// [`Command::bytes`].
    pub capacity: u64,
}

/// What the three registers of an intercepted `VMMCALL` say.
///
/// Pure, and the whole of the request-side validation: the selector, the
/// version, the command code, and the two properties of a buffer that can be
/// judged without touching the guest's memory. What cannot be judged here is
/// whether the buffer is really the guest's to write, which needs the guest's
/// own page tables and is the host's to answer.
#[must_use]
pub const fn decode(command: u64, first: u64, second: u64) -> Decoded {
    // The selector is the top doubleword of the command word, which a shift
    // leaves nothing above.
    let selector = (command >> SELECTOR_SHIFT) as u32;
    if selector != SELECTOR {
        return Decoded::Foreign;
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the version and the code are the two words below the selector by definition"
    )]
    let (version, code) = ((command >> VERSION_SHIFT) as u16, command as u16);
    if version != VERSION {
        return Decoded::Refused(Status::Version);
    }
    let Some(command) = Command::from_code(code) else {
        return Decoded::Refused(Status::UnknownCommand);
    };
    if !first.is_multiple_of(ALIGNMENT) {
        return Decoded::Refused(Status::Misaligned);
    }
    if second < command.bytes() {
        return Decoded::Refused(Status::TooSmall);
    }
    Decoded::Request(Request {
        command,
        buffer: first,
        capacity: second,
    })
}

/// What became of a hypercall, as the caller reads it out of `RAX`.
///
/// Every refusal is a statement about the request rather than about the
/// machine, and none of them leaves the guest in a different state than it was
/// in: a call that was not served wrote nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u64)]
pub enum Status {
    /// Served. The buffer holds what was asked for.
    Ok = 0,
    /// The command word named a version of this interface the host does not
    /// implement.
    Version = 1,
    /// The selector and version were this interface's, and the code named no
    /// command it has.
    UnknownCommand = 2,
    /// The buffer address is not quadword aligned; see [`ALIGNMENT`].
    Misaligned = 3,
    /// The buffer is smaller than the command needs. Its required size is
    /// `size_of` the command's own structure, which the caller can ask this
    /// crate for rather than guess.
    TooSmall = 4,
    /// There is nowhere to put the answer: either the guest's own page tables
    /// do not translate the whole of the buffer, or they do and the host
    /// has not yet described the memory they translate to.
    ///
    /// The second is not a refusal a caller has to live with. A guest's memory
    /// is described to the hypervisor as the guest touches it, so a caller
    /// that has written every page of its buffer — which anything that
    /// zeroed it has — has already ruled it out.
    Unreachable = 5,
    /// The buffer translates to memory that is not the guest's to write — the
    /// hypervisor's own, which a guest sees as zeroes — so nothing was written.
    Unwritable = 6,
    /// The buffer was the guest's to write when it was checked and a write into
    /// it failed anyway, which is another processor of the same guest changing
    /// its tables while this one was being served. Part of the buffer may hold
    /// part of an answer, and its header is not written, so nothing can be read
    /// out of it.
    Faulted = 7,
}

impl Status {
    /// The status a word names, or `None` for one this interface does not
    /// define — which is what a caller reads if it is talking to something
    /// other than this hypervisor.
    #[must_use]
    pub const fn from_word(word: u64) -> Option<Self> {
        Some(match word {
            0 => Self::Ok,
            1 => Self::Version,
            2 => Self::UnknownCommand,
            3 => Self::Misaligned,
            4 => Self::TooSmall,
            5 => Self::Unreachable,
            6 => Self::Unwritable,
            7 => Self::Faulted,
            _ => return None,
        })
    }

    /// The word this status travels in `RAX` as.
    #[must_use]
    pub const fn word(self) -> u64 {
        self as u64
    }
}

impl Display for Status {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        let said = match self {
            Self::Ok => "the call was served",
            Self::Version => "the hypervisor does not implement this version of the interface",
            Self::UnknownCommand => "the hypervisor does not have the command that was asked for",
            Self::Misaligned => "the buffer address is not quadword aligned",
            Self::TooSmall => "the buffer is smaller than the command needs",
            Self::Unreachable => {
                "the guest's page tables do not translate the whole buffer, or the \
                                  hypervisor has not described what they translate to"
            }
            Self::Unwritable => "the buffer is not the guest's own memory to write",
            Self::Faulted => "the buffer stopped being writable while it was being written",
        };
        formatter.write_str(said)
    }
}

/// A structure of this interface, as the bytes it travels in a buffer as.
///
/// Implemented for the sections of a dump and for nothing else. Every one of
/// them is a `#[repr(C)]` layout of unsigned integers and arrays of them, with
/// no padding anywhere — which the assertions beside each of them enforce — so
/// every byte of one is an initialized byte of a field, and no arrangement of
/// those bytes is an invalid value of the type.
pub trait Wire: Sized {
    /// This value's bytes, in the order the ABI puts them in.
    fn wire(&self) -> &[u8] {
        // SAFETY: `Self` is one of the padding-free integer structures this
        // trait is implemented for, so all `size_of::<Self>()` bytes behind the
        // reference are initialized, and a shared borrow of them cannot outlive
        // the borrow of `self` they came from.
        unsafe { slice::from_raw_parts(core::ptr::from_ref(self).cast::<u8>(), size_of::<Self>()) }
    }
}

#[cfg(test)]
mod tests {
    //! What both sides depend on being the same statement: a command word that
    //! round-trips, a refusal for every request-side rule, and an instruction
    //! that is somebody else's left alone.

    use super::{
        ALIGNMENT, Arguments, Command, Decoded, Request, SELECTOR, Status, VERSION, decode,
    };
    use crate::ApicDump;

    /// A command word with the parts of it this suite varies spelled out, so
    /// that a test naming a wrong version or a wrong code is not building one
    /// through the encoder it is testing.
    fn word(selector: u32, version: u16, code: u16) -> u64 {
        (u64::from(selector) << 32) | (u64::from(version) << 16) | u64::from(code)
    }

    #[test]
    fn a_command_travels_as_the_selector_the_version_and_its_own_code() {
        let dump = Command::APIC_DUMP;
        assert_eq!(dump.word(), word(SELECTOR, VERSION, dump.code()));
        assert_eq!(Command::from_code(dump.code()), Some(dump));
    }

    #[test]
    fn a_request_decodes_to_the_command_and_the_buffer_it_named() {
        let arguments = Arguments::buffer(0x1_0000, size_of::<ApicDump>() as u64);
        assert_eq!(
            decode(Command::APIC_DUMP.word(), arguments.first, arguments.second),
            Decoded::Request(Request {
                command: Command::APIC_DUMP,
                buffer: arguments.first,
                capacity: arguments.second,
            })
        );
    }

    #[test]
    fn an_instruction_that_is_not_this_interfaces_is_left_alone() {
        // The portal's own notifications go through the same instruction, and so
        // may anything else in the guest. None of them may be answered as a
        // hypercall, whatever else about the registers looks plausible.
        for foreign in [
            0,
            u64::MAX,
            u64::from_le_bytes(*b"EBS DONE"),
            word(SELECTOR.wrapping_add(1), VERSION, Command::APIC_DUMP.code()),
        ] {
            assert_eq!(
                decode(foreign, 0x1_0000, u64::MAX),
                Decoded::Foreign,
                "{foreign:#x}"
            );
        }
    }

    #[test]
    fn a_version_this_interface_does_not_have_is_refused_before_the_command_is_read() {
        // In that order deliberately: a caller from another build may be naming a
        // command code that has since come to mean something else, so the version
        // has to be judged first.
        for version in [VERSION.wrapping_sub(1), VERSION.wrapping_add(1), u16::MAX] {
            assert_eq!(
                decode(word(SELECTOR, version, 0xFFFF), 0x1_0000, u64::MAX),
                Decoded::Refused(Status::Version),
                "version {version}"
            );
        }
    }

    #[test]
    fn a_code_this_interface_does_not_have_is_refused() {
        for code in [0, Command::APIC_DUMP.code() + 1, u16::MAX] {
            assert_eq!(
                decode(word(SELECTOR, VERSION, code), 0x1_0000, u64::MAX),
                Decoded::Refused(Status::UnknownCommand),
                "code {code}"
            );
        }
    }

    #[test]
    fn a_buffer_that_could_not_be_read_at_its_own_offsets_is_refused() {
        for misaligned in 1..ALIGNMENT {
            assert_eq!(
                decode(Command::APIC_DUMP.word(), 0x1_0000 + misaligned, u64::MAX),
                Decoded::Refused(Status::Misaligned),
                "{misaligned} past a quadword"
            );
        }
    }

    #[test]
    fn a_buffer_shorter_than_the_answer_is_refused_rather_than_partly_filled() {
        let needed = Command::APIC_DUMP.bytes();
        assert_eq!(needed, size_of::<ApicDump>() as u64);
        for capacity in [0, 1, needed - 1] {
            assert_eq!(
                decode(Command::APIC_DUMP.word(), 0x1_0000, capacity),
                Decoded::Refused(Status::TooSmall),
                "{capacity} bytes"
            );
        }
        assert!(matches!(
            decode(Command::APIC_DUMP.word(), 0x1_0000, needed),
            Decoded::Request(_)
        ));
    }

    #[test]
    fn every_status_round_trips_through_the_register_it_travels_in() {
        for status in [
            Status::Ok,
            Status::Version,
            Status::UnknownCommand,
            Status::Misaligned,
            Status::TooSmall,
            Status::Unreachable,
            Status::Unwritable,
            Status::Faulted,
        ] {
            assert_eq!(Status::from_word(status.word()), Some(status));
        }
        // And a word from something that is not this hypervisor is not silently
        // read as one of them.
        assert_eq!(Status::from_word(u64::MAX), None);
    }
}
