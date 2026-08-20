//! Who this controller is, and which logical destinations it answers to.
//!
//! Two questions the guest may read and one it may not. The identifier is
//! read-only for the reason [`crate::registers`] gives, and the version
//! register describes the hardware behind this controller rather than anything
//! the guest chose. The logical destination is writable in the older face and
//! derived by the architecture in x2APIC, which is why it is computed here in
//! that face rather than stored: hardware computes the same thing from the same
//! number, and a guest cannot get the two out of step.
//!
//! # Both answers are asked in a face, and the face is the caller's to fix
//!
//! Which face the controller is in decides both of them, and the caller passes
//! it in rather than each of these reading it again. A remote processor working
//! out whether a command names this controller is reading a word the target's
//! own guest may be changing under it, and one decision made from two loads of
//! that word is a decision made about a controller in neither state — an
//! xAPIC-format logical identifier matched by the x2APIC cluster rule, which
//! matches nothing and drops the interrupt with no error anywhere. So the mode
//! is taken once, as a [`Mode`], and threaded down.

use core::sync::atomic::Ordering;

use cpu::ApicId;

use crate::registers::{Vlapic, base::Mode};

impl Vlapic {
    /// The identifier register, in whichever shape `mode` gives it.
    ///
    /// The older interface keeps it in the top eight bits; x2APIC uses the
    /// whole register.
    ///
    /// Only eight bits of it survive in the older face, and the mask is not
    /// decoration: an identifier of `0x100` shifted into the top byte loses
    /// every bit it has, so a guest would read the same zero from that
    /// processor as from the one whose identifier really is zero. Masking
    /// reports the identifier the older face can hold, which is the
    /// aliasing real hardware performs — and is the same eight bits a
    /// physical destination is matched against in this face, so what a
    /// guest reads here is what it can address. [`crate::install`] says
    /// once, on a machine with such a processor, that the aliasing is
    /// happening.
    pub(crate) fn id_register(&self, mode: Mode) -> u32 {
        match mode {
            Mode::X2Apic => self.apic_id.get(),
            _ => self.xapic_id() << XAPIC_ID_SHIFT,
        }
    }

    /// The identifier the older face can hold, which is the low eight bits of
    /// the real one.
    ///
    /// One statement of that narrowing, because two would be a guest reading an
    /// identifier out of its register that a physical destination then does not
    /// match: what [`Vlapic::id_register`] answers with in that face and what
    /// [`crate::delivery`] compares a destination against have to be the same
    /// eight bits.
    pub(crate) const fn xapic_id(&self) -> u32 {
        self.apic_id.get() & XAPIC_ID_MASK
    }

    /// The version register.
    ///
    /// Both of its fields are the real controller's, out of one read of the
    /// real register: the version this controller is, and how many local
    /// vector table entries it has. The entry count has to be the machine's
    /// because the sources behind those entries are the real ones — a guest
    /// told it has an entry its hardware does not would be told about a
    /// source that can never fire and handed a register that cannot be
    /// programmed — and the version has to come from the same place,
    /// because the architecture's own boundary between the discrete
    /// controller and this one is a version number and software draws
    /// conclusions about the rest of the register from it.
    /// [`crate::hardware::model`] is where both are read.
    ///
    /// End-of-interrupt broadcast suppression is deliberately reported as
    /// unsupported. The bit would let a guest ask that acknowledging a
    /// level-triggered interrupt not be broadcast to the I/O controllers — but
    /// this hypervisor passes those controllers through, so the broadcast is
    /// performed by real hardware when the real acknowledgement is issued, and
    /// nothing here can suppress it. Reporting it unsupported is what stops a
    /// guest asking for something that would then silently not happen.
    ///
    /// So is the bit that says the extended register space is present, for the
    /// same kind of reason: that space is not modelled, and
    /// [`crate::face::table`] carries the limitation.
    pub(crate) const fn version(&self) -> u32 {
        self.model.version()
    }

    /// Which logical destinations this controller answers to, in `mode`.
    ///
    /// In x2APIC this is not stored at all: the architecture derives it from
    /// the identifier and makes it read-only, so it is computed here for the
    /// same reason hardware computes it, and a guest cannot get the two out of
    /// step.
    pub(crate) fn logical_destination(&self, mode: Mode) -> u32 {
        match mode {
            Mode::X2Apic => {
                let id = self.apic_id.get();
                ((id >> X2APIC_CLUSTER_SHIFT) << CLUSTER_SHIFT) | (1 << (id & X2APIC_LOGICAL_MASK))
            }
            _ => self.logical_destination.load(Ordering::Acquire),
        }
    }

    /// Sets which logical destinations this controller answers to. Reachable
    /// only in the older face, where the register is writable.
    pub(crate) fn set_logical_destination(&self, value: u32) {
        self.logical_destination
            .store(value & LOGICAL_DESTINATION_MASK, Ordering::Release);
    }

    /// How a logical destination is matched.
    pub(crate) fn destination_format(&self) -> u32 {
        self.destination_format.load(Ordering::Acquire)
    }

    /// Sets how a logical destination is matched. The reserved remainder reads
    /// as ones, which is its reset value and what the architecture requires.
    pub(crate) fn set_destination_format(&self, value: u32) {
        self.destination_format.store(
            (value & DESTINATION_FORMAT_MASK) | !DESTINATION_FORMAT_MASK,
            Ordering::Release,
        );
    }
}

/// Bits the older interface's identifier is shifted by.
const XAPIC_ID_SHIFT: u32 = 24;

/// The part of an identifier the older interface's field can hold.
const XAPIC_ID_MASK: u32 = 0xFF;

/// The word the older face's identifier register holds for `id`: the part of
/// the identifier that face can keep, in the byte it keeps it in.
///
/// The one statement of that layout outside a controller, because a backing
/// page holds the same word and must not compute it a second way.
pub(crate) const fn xapic_word(id: ApicId) -> u32 {
    (id.get() & XAPIC_ID_MASK) << XAPIC_ID_SHIFT
}

/// The part of the logical destination register that holds anything.
pub(super) const LOGICAL_DESTINATION_MASK: u32 = 0xFF00_0000;

/// The part of the destination format register that selects anything.
pub(super) const DESTINATION_FORMAT_MASK: u32 = 0xF000_0000;

/// The destination format register's reset value: the flat model, with every
/// reserved bit set.
pub(crate) const FLAT_DESTINATION_FORMAT: u32 = u32::MAX;

/// Bits an x2APIC logical identifier's cluster is shifted by.
const CLUSTER_SHIFT: u32 = 16;

/// Bits an identifier is shifted by to leave the cluster it names.
const X2APIC_CLUSTER_SHIFT: u32 = 4;

/// The part of an identifier that selects one processor within its cluster.
const X2APIC_LOGICAL_MASK: u32 = 0xF;
