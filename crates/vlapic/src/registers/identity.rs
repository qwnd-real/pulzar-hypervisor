//! Who this controller is, and which logical destinations it answers to.
//!
//! Two questions the guest may read and one it may not. The identifier is
//! read-only for the reason [`crate::registers`] gives, and the version
//! register describes the hardware behind this controller rather than anything
//! the guest chose. The logical destination is writable in the older face and
//! derived by the architecture in x2APIC, which is why it is computed here in
//! that face rather than stored: hardware computes the same thing from the same
//! number, and a guest cannot get the two out of step.

use core::sync::atomic::Ordering;

use crate::registers::{Vlapic, base::Mode};

impl Vlapic {
    /// The identifier register, in whichever shape the face in use gives it.
    ///
    /// The older interface keeps it in the top eight bits; x2APIC uses the
    /// whole register.
    pub(crate) fn id_register(&self) -> u32 {
        match self.mode() {
            Mode::X2Apic => self.apic_id.get(),
            _ => self.apic_id.get() << XAPIC_ID_SHIFT,
        }
    }

    /// The version register.
    ///
    /// The entry count is the real controller's, because the sources behind
    /// those entries are the real ones. A guest told it has an entry its
    /// hardware does not would be told about a source that can never fire and
    /// handed a register that cannot be programmed.
    ///
    /// End-of-interrupt broadcast suppression is deliberately reported as
    /// unsupported. The bit would let a guest ask that acknowledging a
    /// level-triggered interrupt not be broadcast to the I/O controllers — but
    /// this hypervisor passes those controllers through, so the broadcast is
    /// performed by real hardware when the real acknowledgement is issued, and
    /// nothing here can suppress it. Reporting it unsupported is what stops a
    /// guest asking for something that would then silently not happen.
    pub(crate) const fn version(&self) -> u32 {
        self.model.max_lvt() << MAX_LVT_SHIFT | VERSION_NUMBER
    }

    /// Which logical destinations this controller answers to.
    ///
    /// In x2APIC this is not stored at all: the architecture derives it from
    /// the identifier and makes it read-only, so it is computed here for the
    /// same reason hardware computes it, and a guest cannot get the two out of
    /// step.
    pub(crate) fn logical_destination(&self) -> u32 {
        match self.mode() {
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

/// The version this controller reports: an integrated one, which is what every
/// processor since the discrete controller reports.
const VERSION_NUMBER: u32 = 0x10;

/// Bits the local-vector-table entry count is shifted by in the version
/// register.
const MAX_LVT_SHIFT: u32 = 16;

/// The part of the logical destination register that holds anything.
pub(super) const LOGICAL_DESTINATION_MASK: u32 = 0xFF00_0000;

/// The part of the destination format register that selects anything.
pub(super) const DESTINATION_FORMAT_MASK: u32 = 0xF000_0000;

/// The destination format register's reset value: the flat model, with every
/// reserved bit set.
pub(super) const FLAT_DESTINATION_FORMAT: u32 = u32::MAX;

/// Bits an x2APIC logical identifier's cluster is shifted by.
const CLUSTER_SHIFT: u32 = 16;

/// Bits an identifier is shifted by to leave the cluster it names.
const X2APIC_CLUSTER_SHIFT: u32 = 4;

/// The part of an identifier that selects one processor within its cluster.
const X2APIC_LOGICAL_MASK: u32 = 0xF;
