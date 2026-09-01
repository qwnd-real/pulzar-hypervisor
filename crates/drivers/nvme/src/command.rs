//! What one admin command says, decoded out of the queues the guest shares
//! with the hardware.
//!
//! The entries are sixty-four and sixteen bytes, laid out by the
//! specification (`NVMe` 2.0 §4.3 and §4.4.1), and this driver cares about
//! three fields of each: which command it is, which response buffer the
//! command named, and which command a completion belongs to. Everything else
//! stays bytes.

use x86_64::PhysAddr;

/// How many bytes one submission queue entry is.
pub(crate) const SUBMISSION: u64 = 64;

/// How many bytes one completion queue entry is.
pub(crate) const COMPLETION: u64 = 16;

/// The opcode of an identify command, the one admin command this driver
/// answers for.
const IDENTIFY: u8 = 0x06;

/// The identify value asking for the controller itself (CNS 0x01).
const CONTROLLER: u8 = 0x01;

/// The identify value asking for one namespace (CNS 0x00).
const NAMESPACE: u8 = 0x00;

/// Where a submission's command identifier is, in its second and third
/// bytes: the first is the opcode and the byte between holds flags.
const COMMAND: usize = 2;

/// Where a submission's first physical region pointer is, in dwords six and
/// seven: two dwords the common commands leave reserved come first, then the
/// metadata pointer, which an identify carries none of.
const FIRST_REGION: usize = 24;

/// Where a submission's second physical region pointer is, in dwords eight
/// and nine: a whole page, wherever the buffer the first pointer names is
/// not page aligned and runs past its page's end.
const SECOND_REGION: usize = 32;

/// Where a submission's command-specific dwords begin, at dword ten. An
/// identify's first one holds its `cns`, the byte that says what it asks
/// for.
const COMMAND_WORDS: usize = 40;

/// Where a completion's command identifier is, in dwords two's upper half
/// and three's lower: the completion names what it answers, above the
/// submission queue position it reports and below its status.
const ANSWERED: usize = 12;

/// Where a completion's status word is, in its last two bytes.
const STATUS: usize = 14;

/// The phase tag, in the status word's first bit: whether the entry was
/// written in the round of the queue the reader is on.
const PHASE: u16 = 0x0001;

/// Every bit of the status word that a success leaves clear: the status
/// code in bits ten to one, its type in thirteen to eleven, and the more and
/// do-not-retry flags above them.
const NOT_SUCCESS: u16 = 0x3ffe;

/// What one admin submission is, as far as this driver cares.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Admin {
    /// An identify whose response this driver will answer for.
    Identify(Identify),
    /// Anything else, which passes through without a second look.
    Through,
}

/// What an identify command asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Identified {
    /// The controller itself: its serial number names the drive.
    Controller,
    /// One namespace: its identifiers name the volume on the drive.
    Namespace,
}

/// An identify command this driver means to answer for.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Identify {
    /// The command's identifier, which the completion carries back.
    pub cid: u16,
    /// What the response identifies.
    pub what: Identified,
    /// Where the response's data buffer is, as the PRP pair the command
    /// carried: the first physical region pointer, and the second, which is
    /// a whole page wherever the buffer is not page aligned.
    pub prp1: PhysAddr,
    pub prp2: PhysAddr,
}

/// What one admin completion says, as far as this driver cares.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Completion {
    /// Which command completed.
    pub cid: u16,
    /// The entry's phase tag: whether the hardware wrote it in the round the
    /// driver is reading, or an earlier one whose shadow is still in memory.
    pub phase: bool,
    /// Whether the command succeeded: everything but a clean status leaves
    /// the response as the hardware wrote it, or leaves no response at all.
    pub succeeded: bool,
}

/// Decodes one admin submission queue entry.
#[must_use]
pub(crate) fn admin(entry: &[u8; 64]) -> Admin {
    if entry[0] != IDENTIFY {
        return Admin::Through;
    }
    let what = match entry[COMMAND_WORDS] {
        CONTROLLER => Identified::Controller,
        NAMESPACE => Identified::Namespace,
        // The other things an identify can ask for — namespace lists,
        // descriptors, secondary controllers — carry nothing this driver
        // replaces, and pass through like any other command.
        _ => return Admin::Through,
    };
    // A physical region pointer the guest built out of an address the
    // machine cannot hold is a command the hardware will fail anyway; this
    // driver lets it, rather than tracking a response nowhere.
    let (Ok(prp1), Ok(prp2)) = (
        PhysAddr::try_new(quad(entry, FIRST_REGION)),
        PhysAddr::try_new(quad(entry, SECOND_REGION)),
    ) else {
        return Admin::Through;
    };
    Admin::Identify(Identify {
        cid: word(entry, COMMAND),
        what,
        prp1,
        prp2,
    })
}

/// Decodes one admin completion queue entry.
#[must_use]
pub(crate) fn completion(entry: &[u8; 16]) -> Completion {
    let status = word(entry, STATUS);
    Completion {
        cid: word(entry, ANSWERED),
        phase: status & PHASE != 0,
        succeeded: status & NOT_SUCCESS == 0,
    }
}

/// The eight bytes at `at`, least significant first.
fn quad(entry: &[u8], at: usize) -> u64 {
    let mut value = 0;
    for position in 0..8 {
        value |= u64::from(entry[at + position]) << (8 * position);
    }
    value
}

/// The two bytes at `at`, least significant first.
fn word(entry: &[u8], at: usize) -> u16 {
    u16::from(entry[at]) | (u16::from(entry[at + 1]) << 8)
}

#[cfg(test)]
mod tests {
    use x86_64::PhysAddr;

    use super::{Admin, Identified, admin, completion};

    /// One identify command, saying it asks for the controller.
    ///
    /// The data pointer is at dwords six through nine and the `cns` at the
    /// tenth, past the reserved dwords and the metadata pointer the command
    /// does not carry.
    fn controller_identify() -> [u8; 64] {
        let mut entry = [0; 64];
        entry[0] = 0x06;
        entry[2] = 0x2a;
        entry[3] = 0x01;
        entry[24] = 0x00;
        entry[25] = 0x10;
        entry[32] = 0x00;
        entry[33] = 0x20;
        entry[40] = 0x01;
        entry
    }

    #[test]
    fn an_identify_of_the_controller_is_held() {
        let Admin::Identify(identify) = admin(&controller_identify()) else {
            panic!("an identify was decoded as something else");
        };
        assert_eq!(identify.cid, 0x012a);
        assert_eq!(identify.what, Identified::Controller);
        assert_eq!(identify.prp1.as_u64(), 0x1000);
        assert_eq!(identify.prp2.as_u64(), 0x2000);
    }

    #[test]
    fn an_identify_of_a_namespace_is_held() {
        let mut entry = controller_identify();
        entry[40] = 0x00;
        let Admin::Identify(identify) = admin(&entry) else {
            panic!("an identify was decoded as something else");
        };
        assert_eq!(identify.what, Identified::Namespace);
    }

    #[test]
    fn another_identify_passes_through() {
        let mut entry = controller_identify();
        entry[40] = 0x02;
        assert!(matches!(admin(&entry), Admin::Through));
    }

    #[test]
    fn another_command_passes_through() {
        let mut entry = controller_identify();
        entry[0] = 0x02;
        assert!(matches!(admin(&entry), Admin::Through));
    }

    #[test]
    fn an_impossible_buffer_passes_through() {
        let mut entry = controller_identify();
        entry[24] = 0xff;
        entry[25] = 0xff;
        entry[26] = 0xff;
        entry[27] = 0xff;
        entry[28] = 0xff;
        entry[29] = 0xff;
        entry[30] = 0xff;
        entry[31] = 0xff;
        assert!(matches!(admin(&entry), Admin::Through));
    }

    #[test]
    fn an_impossible_second_pointer_passes_through() {
        // A first pointer the machine can hold does not save a command whose
        // second one it cannot: the answer may land in either, and a
        // response tracked to nowhere is worse than none.
        let mut entry = controller_identify();
        entry[32] = 0xff;
        entry[33] = 0xff;
        entry[34] = 0xff;
        entry[35] = 0xff;
        entry[36] = 0xff;
        entry[37] = 0xff;
        entry[38] = 0xff;
        entry[39] = 0xff;
        assert!(matches!(admin(&entry), Admin::Through));
    }

    #[test]
    fn an_unused_second_pointer_of_zero_is_held() {
        // A page-aligned buffer needs no second pointer, and a command
        // carrying none is an ordinary one.
        let mut entry = controller_identify();
        entry[32] = 0x00;
        entry[33] = 0x00;
        let Admin::Identify(identify) = admin(&entry) else {
            panic!("an identify was decoded as something else");
        };
        assert_eq!(identify.prp2, PhysAddr::zero());
    }

    #[test]
    fn a_completion_carries_its_command_and_its_freshness() {
        let mut entry = [0; 16];
        entry[12] = 0x2a;
        entry[13] = 0x01;
        entry[14] = 0x01;
        let decoded = completion(&entry);
        assert_eq!(decoded.cid, 0x012a);
        assert!(decoded.phase);
        assert!(decoded.succeeded);
    }

    #[test]
    fn a_failed_completion_is_not_a_success() {
        let mut entry = [0; 16];
        entry[12] = 0x2a;
        entry[14] = 0x01;
        entry[15] = 0x04;
        let decoded = completion(&entry);
        assert!(decoded.phase);
        assert!(!decoded.succeeded);
    }

    #[test]
    fn a_stale_completion_is_not_fresh() {
        let mut entry = [0; 16];
        entry[12] = 0x2a;
        entry[14] = 0x00;
        let decoded = completion(&entry);
        assert!(!decoded.phase);
        // A clean status in a stale round is still stale, and still clean.
        assert!(decoded.succeeded);
    }

    #[test]
    fn a_clean_status_above_the_phase_bit_is_a_success() {
        // The status code's type, alone above the phase bit: not a success.
        let mut entry = [0; 16];
        entry[14] = 0x01;
        entry[15] = 0x08;
        assert!(!completion(&entry).succeeded);
        // A status code in the low bits: also not a success.
        entry[15] = 0x00;
        entry[14] = 0x41;
        assert!(!completion(&entry).succeeded);
        // The more and do-not-retry flags are information about an error
        // rather than the error itself: with the code and its type clean,
        // the command succeeded.
        entry[14] = 0x01;
        entry[15] = 0xc0;
        assert!(completion(&entry).succeeded);
    }
}
