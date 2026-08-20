//! Which processors a command names.
//!
//! Almost all of delivery is this: four ways of naming processors, three of
//! which can name several at once. Which of the three logical models applies,
//! and how wide an identifier is compared, is judged against the *target's* own
//! state rather than the sender's, because that is where the architecture puts
//! it — and one command may reach controllers that disagree about both.
//!
//! # One snapshot per target, and one per sender
//!
//! The two sides are asked exactly once each, and that is the whole of what
//! makes a decision here a decision about a machine that existed.
//!
//! A target's face, its identifier and its logical identifier are three answers
//! that only mean anything together: the face decides the *format* of the
//! logical identifier and the *rule* it is matched by, so a decision made from
//! two loads of a word the target's own guest is changing can match an
//! xAPIC-format identifier by the x2APIC cluster rule. That matches nothing,
//! and nothing reports it — the interrupt is simply gone, and a guest whose
//! processors enable x2APIC one at a time during bring-up runs that transition
//! once per processor. So [`Addressee`] is one snapshot of the target, taken
//! before anything is compared.
//!
//! The width the destination *field* has is the sender's, not the target's,
//! because it is the sender's register the value was written into. That is why
//! the broadcast question is asked of the command once, by
//! [`Command::is_broadcast`], and passed down as an answer rather than
//! re-derived per target.

use cpu::ApicId;

use crate::registers::{
    Vlapic,
    base::Mode,
    icr::{Command, DestinationMode, Shorthand},
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
pub(crate) fn targets<'a>(
    from: &Vlapic,
    lapics: &'a [Vlapic],
    command: Command,
) -> impl Iterator<Item = &'a Vlapic> + Clone {
    let shorthand = command.shorthand();
    let here = from.index();
    // Every one of these is the sending controller's, and the sender is the
    // processor executing this — so they are read once because there is nothing
    // to race, as much as because a command is one command.
    let sender = from.mode();
    let destination = command.destination(sender);
    let broadcast = command.is_broadcast(sender);
    let mode = command.destination_mode();
    lapics.iter().filter(move |target| match shorthand {
        Shorthand::Myself => target.index() == here,
        Shorthand::All => true,
        Shorthand::Others => target.index() != here,
        Shorthand::None => Addressee::of(target).addresses(destination, broadcast, mode),
    })
}

/// Everything about one controller that deciding whether a destination names it
/// depends on, read as one snapshot.
///
/// A separate type rather than three accessors because the three are only
/// meaningful together, and because it makes the rule a function of values —
/// which is what lets the whole destination matrix be checked without a
/// machine.
#[derive(Clone, Copy, Debug)]
struct Addressee {
    /// The face this controller answers through, which decides both the width
    /// of its identifier and which logical model applies.
    mode: Mode,
    /// The identifier interrupts to it are addressed by, which is the real one.
    apic_id: u32,
    /// The eight bits of that identifier the older face can hold, which is what
    /// its guest reads out of the register and so all it can address it by.
    xapic_id: u32,
    /// Which logical destinations it answers to, in the format `mode` gives.
    logical_id: u32,
    /// How a logical destination is matched, in the older face.
    format: u32,
}

impl Addressee {
    /// One snapshot of a controller, taken from one load of its base register.
    fn of(target: &Vlapic) -> Self {
        let mode = target.base().mode();
        Self {
            mode,
            apic_id: target.apic_id().get(),
            xapic_id: target.xapic_id(),
            logical_id: target.logical_destination(mode),
            format: target.destination_format(),
        }
    }

    /// Whether a destination names this controller.
    ///
    /// `broadcast` is whether the destination is the all-ones value in the
    /// width the *sending* face gives the field.
    const fn addresses(self, destination: u32, broadcast: bool, mode: DestinationMode) -> bool {
        if broadcast {
            return true;
        }
        match mode {
            DestinationMode::Physical => self.physical(destination),
            DestinationMode::Logical => self.logical(destination),
        }
    }

    /// Whether a physical destination names this controller.
    ///
    /// The comparison is as wide as the narrower of the two: a controller in
    /// the older face has an eight-bit identifier field, and that is the
    /// identifier its guest can read and therefore the only one it can
    /// address.
    ///
    /// Two consequences, and both are what hardware does. A machine with
    /// identifiers past 255 aliases them in the older face — two processors
    /// answering to one destination — which is the same aliasing
    /// [`Vlapic::id_register`] reports and is why the older face cannot address
    /// such a machine properly at all. And a destination that does not fit
    /// those eight bits is compared whole even against a controller in the
    /// older face, because there is no eight-bit value it could have meant:
    /// a sender in x2APIC naming identifier `0x100` reaches the processor
    /// whose identifier that is, whatever face that processor is in.
    const fn physical(self, destination: u32) -> bool {
        if matches!(self.mode, Mode::X2Apic) || !ApicId::new(destination).fits_xapic() {
            return destination == self.apic_id;
        }
        destination == self.xapic_id
    }

    /// Whether a logical destination names this controller.
    ///
    /// Three models, and which one applies is not a property of the command but
    /// of the controller being matched against: the face it is in, and in the
    /// older face the destination format register it was programmed with.
    const fn logical(self, destination: u32) -> bool {
        // Cluster in the high half, a bit per processor in the low half. The only
        // model x2APIC has, and the one that needs no format register.
        if matches!(self.mode, Mode::X2Apic) {
            return destination >> CLUSTER_SHIFT == self.logical_id >> CLUSTER_SHIFT
                && destination & self.logical_id & CLUSTER_MEMBERS != 0;
        }
        // The older face's identifier lives in the top byte, and only two of the
        // sixteen models the format register can name are models at all. The
        // other fourteen match nothing: what a controller does when programmed
        // with one is undefined, and naming no processor is the answer that
        // cannot deliver an interrupt somewhere the guest did not ask for.
        let identifier = self.logical_id >> XAPIC_LOGICAL_SHIFT;
        match self.format >> DESTINATION_FORMAT_SHIFT {
            // Eight processors, a bit each.
            FLAT_MODEL => destination & identifier & XAPIC_LOGICAL_MASK != 0,
            // A four-bit cluster address, then a four-bit mask within it.
            CLUSTER_MODEL => {
                destination >> XAPIC_CLUSTER_SHIFT == identifier >> XAPIC_CLUSTER_SHIFT
                    && destination & identifier & XAPIC_CLUSTER_MASK != 0
            }
            _ => false,
        }
    }
}

/// Bits an x2APIC logical identifier's cluster is shifted by.
const CLUSTER_SHIFT: u32 = 16;

/// The part of an x2APIC logical identifier that names processors within a
/// cluster.
const CLUSTER_MEMBERS: u32 = 0xFFFF;

/// Bits the older face's logical identifier is shifted by: the top byte.
const XAPIC_LOGICAL_SHIFT: u32 = 24;

/// The part of a destination the older face's logical identifier can name.
const XAPIC_LOGICAL_MASK: u32 = 0xFF;

/// Bits a cluster address is shifted by within that byte.
const XAPIC_CLUSTER_SHIFT: u32 = 4;

/// The part of that byte naming processors within a cluster.
const XAPIC_CLUSTER_MASK: u32 = 0xF;

/// Bits the destination format register's model selector is shifted by.
const DESTINATION_FORMAT_SHIFT: u32 = 28;

/// The encoding that selects the flat model.
const FLAT_MODEL: u32 = 0xF;

/// The encoding that selects the cluster model.
const CLUSTER_MODEL: u32 = 0x0;

#[cfg(test)]
mod tests {
    //! The whole destination matrix, over values rather than over controllers:
    //! physical and logical, flat and cluster, both faces, unicast and
    //! broadcast — including the pairs where the sender and the target disagree
    //! about which face they are in, which is the case a decision made from two
    //! loads of the target's base register gets wrong.

    use super::{Addressee, XAPIC_LOGICAL_SHIFT};
    use crate::registers::{
        base::Mode,
        icr::{Command, DestinationMode},
    };

    /// A controller in the older face, matching the flat model.
    fn flat(apic_id: u32, logical: u8) -> Addressee {
        Addressee {
            mode: Mode::XApic,
            apic_id,
            xapic_id: apic_id & 0xFF,
            logical_id: u32::from(logical) << XAPIC_LOGICAL_SHIFT,
            format: 0xFFFF_FFFF,
        }
    }

    /// The same controller, matching the cluster model: the top nibble of its
    /// logical identifier is a cluster address and the low one a mask within
    /// it.
    fn cluster(apic_id: u32, logical: u8) -> Addressee {
        Addressee {
            format: 0x0FFF_FFFF,
            ..flat(apic_id, logical)
        }
    }

    /// A controller in x2APIC, whose logical identifier the architecture
    /// derives from its own — cluster above, one bit within it below.
    ///
    /// The derivation is written out here rather than taken from the register
    /// layer, so that the matrix below is a check of the matching rule and does
    /// not agree with it merely by sharing an expression.
    fn wide(apic_id: u32) -> Addressee {
        Addressee {
            mode: Mode::X2Apic,
            logical_id: ((apic_id >> 4) << 16) | (1 << (apic_id & 0xF)),
            ..flat(apic_id, 0)
        }
    }

    /// Whether a physical destination written through `sender` names `target`.
    fn physical(target: Addressee, sender: Mode, destination: u32) -> bool {
        addresses(target, sender, destination, DestinationMode::Physical)
    }

    /// Whether a logical destination written through `sender` names `target`.
    fn logical(target: Addressee, sender: Mode, destination: u32) -> bool {
        addresses(target, sender, destination, DestinationMode::Logical)
    }

    /// The decision as `targets` makes it: the command decides the broadcast in
    /// the sender's width, and the target answers everything else.
    fn addresses(target: Addressee, sender: Mode, destination: u32, mode: DestinationMode) -> bool {
        let high = if matches!(sender, Mode::X2Apic) {
            destination
        } else {
            destination << XAPIC_LOGICAL_SHIFT
        };
        let command = Command::from_halves(0x0000_0030, high);
        target.addresses(
            command.destination(sender),
            command.is_broadcast(sender),
            mode,
        )
    }

    #[test]
    fn a_physical_destination_names_one_processor_in_either_face() {
        for face in [Mode::XApic, Mode::X2Apic] {
            assert!(physical(flat(0x05, 0), face, 0x05));
            assert!(!physical(flat(0x05, 0), face, 0x06));
            assert!(physical(wide(0x05), face, 0x05));
            assert!(!physical(wide(0x05), face, 0x06));
        }
    }

    #[test]
    fn a_physical_destination_is_eight_bits_wide_in_the_older_face() {
        // What a guest in the older face reads out of its identifier register is
        // the low eight bits, so that is what it can address — and the aliasing
        // that follows is hardware's own.
        let alias = flat(0x100, 0);
        assert!(physical(alias, Mode::XApic, 0x00));
        assert!(physical(flat(0x00, 0), Mode::XApic, 0x00));
        // A destination too wide for those eight bits cannot have meant any of
        // them, so it is compared whole — which is how a sender in x2APIC reaches
        // a processor still in the older face.
        assert!(physical(alias, Mode::X2Apic, 0x100));
        assert!(!physical(flat(0x00, 0), Mode::X2Apic, 0x100));
        // And a controller in x2APIC is never aliased: its field is as wide as
        // its identifier.
        assert!(!physical(wide(0x100), Mode::XApic, 0x00));
        assert!(physical(wide(0x100), Mode::X2Apic, 0x100));
    }

    #[test]
    fn a_broadcast_is_judged_in_the_senders_width_and_reaches_every_face() {
        // The mixed-mode pair: a broadcast written through one face reaching a
        // controller in the other. Judged in the target's width instead, the
        // first of these names nobody in the older face and the second is a
        // broadcast to nobody in x2APIC.
        for mode in [DestinationMode::Physical, DestinationMode::Logical] {
            for target in [flat(0x05, 0x01), cluster(0x05, 0x11), wide(0x05)] {
                assert!(
                    addresses(target, Mode::X2Apic, 0xFFFF_FFFF, mode),
                    "{target:?} under a wide broadcast"
                );
                assert!(
                    addresses(target, Mode::XApic, 0xFF, mode),
                    "{target:?} under a narrow broadcast"
                );
            }
        }
    }

    #[test]
    fn an_identifier_the_older_face_broadcasts_with_is_not_a_broadcast_from_the_wide_one() {
        // Identifier 255 is an ordinary processor, and a command from x2APIC
        // naming it must reach that one processor rather than the machine.
        assert!(physical(wide(0xFF), Mode::X2Apic, 0xFF));
        assert!(!physical(wide(0x05), Mode::X2Apic, 0xFF));
        assert!(!physical(flat(0x05, 0x01), Mode::X2Apic, 0xFF));
    }

    #[test]
    fn the_flat_model_is_a_bit_per_processor() {
        let target = flat(0x03, 0b0000_0100);
        assert!(logical(target, Mode::XApic, 0b0000_0100));
        assert!(logical(target, Mode::XApic, 0b0000_0110));
        assert!(!logical(target, Mode::XApic, 0b0000_0010));
        assert!(!logical(target, Mode::XApic, 0));
    }

    #[test]
    fn the_cluster_model_is_an_address_and_a_mask_within_it() {
        // Cluster 1, processor 2 of it.
        let target = cluster(0x03, 0x12);
        assert!(logical(target, Mode::XApic, 0x12));
        assert!(logical(target, Mode::XApic, 0x13));
        // The right mask in the wrong cluster, and the right cluster with the
        // wrong mask.
        assert!(!logical(target, Mode::XApic, 0x22));
        assert!(!logical(target, Mode::XApic, 0x11));
    }

    #[test]
    fn the_wide_face_has_one_logical_model() {
        // Identifier 0x11 is cluster 1, bit 1 within it.
        let target = wide(0x11);
        assert_eq!(target.logical_id, 0x0001_0002);
        assert!(logical(target, Mode::X2Apic, 0x0001_0002));
        assert!(logical(target, Mode::X2Apic, 0x0001_0003));
        // The right members of the wrong cluster.
        assert!(!logical(target, Mode::X2Apic, 0x0002_0002));
        assert!(!logical(target, Mode::X2Apic, 0x0001_0001));
    }

    #[test]
    fn the_fourteen_undefined_destination_formats_name_nobody() {
        // Only the flat and cluster encodings are models. A controller programmed
        // with any of the other fourteen matched as though it were a cluster,
        // which is an interrupt delivered somewhere the guest did not ask for.
        for model in 0x1..=0xE {
            let target = Addressee {
                format: model << 28,
                ..flat(0x03, 0xFF)
            };
            // Every destination but the one that names every processor, which is
            // not a logical identifier being matched at all.
            for destination in 0..0xFF {
                assert!(
                    !logical(target, Mode::XApic, destination),
                    "format {model:#x} matched {destination:#x}"
                );
            }
            assert!(logical(target, Mode::XApic, 0xFF), "format {model:#x}");
        }
    }

    #[test]
    fn a_logical_destination_of_nothing_names_nobody() {
        // Every model, against the one destination that selects no processor in
        // any of them.
        for target in [flat(0x03, 0xFF), cluster(0x03, 0x1F), wide(0x03)] {
            assert!(!logical(target, Mode::X2Apic, 0), "{target:?}");
        }
    }

    #[test]
    fn a_controller_that_is_switched_off_is_still_addressed_by_the_older_rules() {
        // Whether it accepts what arrives is its own question, asked later. What
        // is asked here is only whether the command named it, and a disabled
        // controller has the identifier and the registers it had before.
        let target = Addressee {
            mode: Mode::Disabled,
            ..flat(0x05, 0b0000_1000)
        };
        assert!(physical(target, Mode::XApic, 0x05));
        assert!(logical(target, Mode::XApic, 0b0000_1000));
        assert!(physical(target, Mode::X2Apic, 0x05));
    }
}
