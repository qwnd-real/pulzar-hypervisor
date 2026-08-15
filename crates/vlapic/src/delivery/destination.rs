//! Which processors a command names.
//!
//! Almost all of delivery is this: four ways of naming processors, three of
//! which can name several at once. Every one of them is judged against the
//! *target's* own state rather than the sender's, because that is where the
//! architecture puts it — the face a controller is in decides how wide its
//! identifier is and which logical model it matches by, and one command may
//! reach controllers that disagree about both.

use crate::{
    registers::{
        Vlapic,
        base::Mode,
        icr::{Command, DestinationMode, Shorthand},
    },
};

/// Which processors a command names.
///
/// Streamed rather than collected, which is what lets a command name every
/// processor on the machine however many there are. A fixed array would have to
/// be sized for the largest machine and would silently drop targets on anything
/// larger — a partial broadcast, which for a shootdown or a rendezvous is worse
/// than none at all because the sender has no way to know.
///
/// A shorthand answers without looking at the destination field at all, which
/// is the architecture's rule and not a shortcut: the destination *mode* is
/// ignored too whenever one is used.
pub(super) fn targets<'a>(
    from: &Vlapic,
    lapics: &'a [Vlapic],
    command: Command,
) -> impl Iterator<Item = &'a Vlapic> {
    let shorthand = command.shorthand();
    let here = from.index();
    let destination = command.destination(from.mode());
    let mode = command.destination_mode();
    lapics.iter().filter(move |target| match shorthand {
        Shorthand::Myself => target.index() == here,
        Shorthand::All => true,
        Shorthand::Others => target.index() != here,
        Shorthand::None => addresses(target, destination, mode),
    })
}

/// Whether a destination names this processor.
fn addresses(target: &Vlapic, destination: u32, mode: DestinationMode) -> bool {
    match mode {
        DestinationMode::Physical => physical(target, destination),
        DestinationMode::Logical => logical(target, destination),
    }
}

/// Whether a physical destination names this processor.
///
/// The all-ones value is a broadcast in both faces. In the older one the field
/// is only eight bits wide, so the value that broadcasts is the eight-bit one.
fn physical(target: &Vlapic, destination: u32) -> bool {
    if destination == broadcast(target) {
        return true;
    }
    target.apic_id().get() == destination
}

/// Whether a logical destination names this processor.
///
/// Three models, and which one applies is not a property of the command but of
/// the controller being matched against: the face it is in, and in the older
/// face the destination format register it was programmed with.
fn logical(target: &Vlapic, destination: u32) -> bool {
    if destination == broadcast(target) {
        return true;
    }
    let logical_id = target.logical_destination();
    match target.mode() {
        // Cluster in the high half, a bit per processor in the low half. The
        // only model x2APIC has.
        Mode::X2Apic => {
            destination >> CLUSTER_SHIFT == logical_id >> CLUSTER_SHIFT
                && destination & logical_id & CLUSTER_MEMBERS != 0
        }
        _ if flat(target) => {
            // Eight processors, a bit each, in the top byte of the register.
            (destination & (logical_id >> XAPIC_LOGICAL_SHIFT)) & u32::from(u8::MAX) != 0
        }
        // Four-bit cluster address, then a four-bit mask within it.
        _ => {
            let cluster = (logical_id >> XAPIC_LOGICAL_SHIFT) >> XAPIC_CLUSTER_SHIFT;
            let members = (logical_id >> XAPIC_LOGICAL_SHIFT) & XAPIC_CLUSTER_MASK;
            destination >> XAPIC_CLUSTER_SHIFT == cluster
                && destination & XAPIC_CLUSTER_MASK & members != 0
        }
    }
}

/// Whether this controller matches logical destinations the flat way.
fn flat(target: &Vlapic) -> bool {
    target.destination_format() >> DESTINATION_FORMAT_SHIFT == FLAT_MODEL
}

/// The destination value that names every processor, in the width the face in
/// use gives the field.
fn broadcast(target: &Vlapic) -> u32 {
    match target.mode() {
        Mode::X2Apic => Command::BROADCAST,
        _ => u32::from(u8::MAX),
    }
}

/// Bits an x2APIC logical identifier's cluster is shifted by.
const CLUSTER_SHIFT: u32 = 16;

/// The part of an x2APIC logical identifier that names processors within a
/// cluster.
const CLUSTER_MEMBERS: u32 = 0xFFFF;

/// Bits the older face's logical identifier is shifted by: the top byte.
const XAPIC_LOGICAL_SHIFT: u32 = 24;

/// Bits a cluster address is shifted by within that byte.
const XAPIC_CLUSTER_SHIFT: u32 = 4;

/// The part of that byte naming processors within a cluster.
const XAPIC_CLUSTER_MASK: u32 = 0xF;

/// Bits the destination format register's model selector is shifted by.
const DESTINATION_FORMAT_SHIFT: u32 = 28;

/// The encoding that selects the flat model.
const FLAT_MODEL: u32 = 0xF;
