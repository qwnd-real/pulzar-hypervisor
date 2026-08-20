//! Which processors the machine has, and what each of them is called.
//!
//! A processor has two names and neither is an index. Firmware knows it by an
//! ACPI processor identifier, the interrupt controllers know it by a local APIC
//! identifier, and neither is promised to be small, dense, ordered, or to start
//! at zero — a machine with two sockets and threads disabled in firmware can
//! report identifiers 0, 2, 32 and 34, and a machine with one core can report
//! 12. Anything that indexes an array by an identifier is wrong on those
//! machines.
//!
//! So the roster gives out a third name that is an index. [`CpuIndex`] is
//! assigned by position as the table is read, is dense by construction, and is
//! the only one of the three that may be used to reach into an array. Turning
//! an [`ApicId`] into one is a search, and it is deliberately the only way.

use alloc::vec::Vec;
use core::fmt::{self, Display, Formatter};

use acpi::{Processor, ProcessorState};

/// A processor's local APIC identifier: the value an interrupt is addressed to.
///
/// Thirty-two bits because that is what x2APIC uses. Firmware describing a
/// processor the older way reports eight, which widens without losing anything.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ApicId(u32);

impl ApicId {
    /// The identifier `id`.
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// The identifier itself.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }

    /// Whether this identifier fits the eight-bit destination field the older
    /// interface has, and so whether it can be addressed without x2APIC.
    #[must_use]
    pub const fn fits_xapic(self) -> bool {
        self.0 <= u8::MAX as u32
    }
}

impl Display for ApicId {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "apic {}", self.0)
    }
}

/// A processor's position in the roster.
///
/// Dense, assigned in the order firmware described the processors, and the only
/// one of a processor's names that may be used as an array index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CpuIndex(usize);

impl CpuIndex {
    /// The index itself, for reaching into an array sized by
    /// [`Roster::count`].
    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }
}

impl Display for CpuIndex {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "cpu {}", self.0)
    }
}

/// One processor, as the roster records it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Entry {
    index: CpuIndex,
    apic_id: ApicId,
    uid: u32,
    state: ProcessorState,
}

impl Entry {
    /// Where this processor sits in the roster.
    #[must_use]
    pub const fn index(&self) -> CpuIndex {
        self.index
    }

    /// The local APIC identifier interrupts to it are addressed to.
    #[must_use]
    pub const fn apic_id(&self) -> ApicId {
        self.apic_id
    }

    /// The identifier the rest of firmware's tables know it by.
    #[must_use]
    pub const fn uid(&self) -> u32 {
        self.uid
    }

    /// What firmware says may be done with it.
    #[must_use]
    pub const fn state(&self) -> ProcessorState {
        self.state
    }

    /// Whether this processor may be brought up.
    ///
    /// Both states firmware offers count. `Enabled` means usable now, and
    /// `OnlineCapable` means not usable as it stands but permitted to be
    /// started, which is exactly what starting it is. Only `Disabled` is a
    /// refusal, and it is firmware's to make.
    #[must_use]
    pub const fn startable(&self) -> bool {
        matches!(
            self.state,
            ProcessorState::Enabled | ProcessorState::OnlineCapable
        )
    }
}

/// Every processor firmware described, in the order it described them.
#[derive(Debug)]
pub struct Roster {
    entries: Vec<Entry>,
}

impl Roster {
    /// Records what the multiple APIC description table said.
    pub(crate) fn new(processors: &[Processor]) -> Self {
        Self {
            entries: processors
                .iter()
                .enumerate()
                .map(|(position, processor)| Entry {
                    index: CpuIndex(position),
                    apic_id: ApicId::new(processor.apic_id()),
                    uid: processor.uid(),
                    state: processor.state(),
                })
                .collect(),
        }
    }

    /// How many processors there are, and so how long an array indexed by
    /// [`CpuIndex`] must be.
    #[must_use]
    pub fn count(&self) -> usize {
        self.entries.len()
    }

    /// Every processor.
    #[must_use]
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// The processor at `index`.
    #[must_use]
    pub fn at(&self, index: CpuIndex) -> Option<&Entry> {
        self.entries.get(index.get())
    }

    /// The processor with this local APIC identifier.
    ///
    /// A search rather than an index, because identifiers are not positions and
    /// treating them as such is the bug this whole type exists to prevent.
    #[must_use]
    pub fn find(&self, apic_id: ApicId) -> Option<&Entry> {
        self.entries.iter().find(|entry| entry.apic_id == apic_id)
    }

    /// Whether any processor that may be started needs an identifier wider than
    /// the older interface's eight-bit destination field.
    ///
    /// A machine that answers yes cannot be run in xAPIC mode at all: there
    /// would be processors in it that no interrupt could be addressed to.
    #[must_use]
    pub fn needs_x2apic(&self) -> bool {
        self.entries
            .iter()
            .any(|entry| entry.startable() && !entry.apic_id.fits_xapic())
    }

    /// The first identifier two entries share, if any two do.
    ///
    /// Firmware that describes one processor with both an eight-bit and a
    /// 32-bit structure yields two entries for it, and every consumer of this
    /// roster then has two positions for one processor while [`Roster::find`]
    /// can only ever answer with the earlier of them. What that costs depends
    /// on the consumer — a per-position array with one live half, a processor
    /// started twice, a page of interrupt-controller registers nothing reads —
    /// so the roster is refused rather than each consumer defending itself.
    ///
    /// Quadratic, and asked once on a table with as many entries as the machine
    /// has processors: sorting to do better would need a copy of the whole
    /// roster, and the count is bounded by what one machine's firmware
    /// describes.
    pub(crate) fn duplicated(&self) -> Option<ApicId> {
        self.entries
            .iter()
            .enumerate()
            .find_map(|(position, entry)| {
                self.entries[..position]
                    .iter()
                    .any(|earlier| earlier.apic_id == entry.apic_id)
                    .then_some(entry.apic_id)
            })
    }
}

#[cfg(test)]
mod tests {
    //! The roster is a list with no hardware behind it, so what it says about a
    //! table firmware handed it is decidable here.

    use alloc::vec;

    use acpi::ProcessorState;

    use super::{ApicId, CpuIndex, Entry, Roster};

    /// A roster of processors with these identifiers, described in this order
    /// and all startable.
    fn roster(ids: &[u32]) -> Roster {
        Roster {
            entries: ids
                .iter()
                .enumerate()
                .map(|(position, id)| Entry {
                    index: CpuIndex(position),
                    apic_id: ApicId::new(*id),
                    uid: 0,
                    state: ProcessorState::Enabled,
                })
                .collect(),
        }
    }

    #[test]
    fn identifiers_that_are_merely_sparse_are_not_duplicated() {
        // The machines the roster exists for: unordered, sparse, not starting at
        // zero, and none of them describing a processor twice.
        for ids in [vec![], vec![12], vec![0, 2, 32, 34], vec![34, 0, 32, 2]] {
            assert_eq!(roster(&ids).duplicated(), None, "{ids:?}");
        }
    }

    #[test]
    fn one_identifier_on_two_entries_is_reported() {
        // The firmware this catches: one processor described with both an
        // eight-bit and a 32-bit structure, which the table parser puts into
        // one list.
        assert_eq!(roster(&[0, 1, 0]).duplicated(), Some(ApicId::new(0)));
        // Reported at the second of the two, so a machine with several
        // duplicates names one of them rather than the last.
        assert_eq!(roster(&[4, 5, 5, 4]).duplicated(), Some(ApicId::new(5)));
    }
}
