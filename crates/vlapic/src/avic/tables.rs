//! The physical APIC ID table: one entry per identifier an interrupt may
//! name.
//!
//! An interrupt addressed to a processor is an index into this table, and the
//! entry says where that processor's controller registers are backed. That is
//! the whole of what is built here: the two bits beside the address — whether
//! the processor is running on a physical one, and whether an interrupt may
//! resolve to the entry at all — are promises about an instant rather than
//! facts about the machine, so the table is written with both clear and neither
//! is ever set here. Making one of them belongs to whoever puts the processor
//! on a physical core and to whoever hands its controller to the hardware.
//!
//! The table is a run of whole pages, because the pointer a control block
//! carries names its first page and the architecture gives it no length but
//! the largest valid index beside it.

use alloc::{boxed::Box, vec};

use cpu::ApicId;
use paging::{DirectMap, PagingError, chunk};
use svm::avic::PhysicalApicEntry;
use x86_64::PhysAddr;

use crate::VlapicError;

/// How many entries the table may hold.
///
/// The field beside the table's address names the largest valid index in
/// twelve bits, so the hardware walks at most this many entries — eight pages
/// of them — and a table asked to be larger is one a control block could not
/// describe anyway.
pub(crate) const MAX_ENTRIES: usize = 4096;

/// The table under construction: its entries, and the largest index in them
/// that is valid.
pub(crate) struct PhysicalTable {
    max_index: u16,
    entries: Box<[PhysicalApicEntry]>,
}

impl PhysicalTable {
    /// An empty table whose largest valid index will be `max_index`, for a
    /// controller mode that can name up to `limit`.
    ///
    /// The limit is the mode's rather than the machine's, and it is what makes
    /// this constructor able to enforce the rule its own documentation is
    /// about: how far a table may be indexed is a property of the mode the
    /// hardware will drive it in — the eight-bit mode reaches one byte of
    /// identifiers, the 32-bit mode a page of entries, and further than that
    /// only where the extension reports the extended table.
    ///
    /// # Errors
    ///
    /// [`VlapicError::TableTooLarge`] if the entries would not fit the run a
    /// control block can name, or [`VlapicError::IndexBeyondMode`] if the mode
    /// cannot name the largest index asked for.
    pub(super) fn new(max_index: u16, limit: u16) -> Result<Self, VlapicError> {
        let count = usize::from(max_index) + 1;
        if count > MAX_ENTRIES {
            return Err(VlapicError::TableTooLarge { entries: count });
        }
        if max_index > limit {
            return Err(VlapicError::IndexBeyondMode { max_index, limit });
        }
        Ok(Self {
            max_index,
            entries: vec![PhysicalApicEntry::new(); count].into_boxed_slice(),
        })
    }

    /// Describes the processor `id` as one whose controller registers are
    /// backed by `page`.
    ///
    /// Written neither running nor valid. Nothing is on a physical processor
    /// yet, and nothing is driving any controller in hardware yet, so either
    /// bit set here would be a promise that does not hold: the running bit
    /// would send an interrupt to a physical processor executing something
    /// else, and the valid bit would let a sender's hardware resolve an
    /// interrupt into a page the software model, and not that page, is the
    /// authority for.
    ///
    /// A slot may be described once. Two processors answering to one identifier
    /// have one entry between them, and the second description would name the
    /// second processor's page while the first processor goes on being served
    /// out of its own — so every interrupt addressed to that identifier would
    /// reach a page nothing reads, with the running bit of the entry saying it
    /// had arrived.
    ///
    /// # Errors
    ///
    /// [`VlapicError::IdBeyondTable`] if the identifier is past the largest
    /// index this table was sized for, which the table cannot express, or
    /// [`VlapicError::IdDescribedTwice`] if something has already described it.
    pub(super) fn describe(&mut self, id: ApicId, page: PhysAddr) -> Result<(), VlapicError> {
        let Some(slot) = self.entries.get_mut(id.get() as usize) else {
            return Err(VlapicError::IdBeyondTable {
                id: id.get(),
                max_index: self.max_index,
            });
        };
        // The whole entry is the record of having been described, rather than
        // one bit of it standing for that: an undescribed slot is the zero entry
        // the table was built with, and a described one names a backing page,
        // which is never physical page zero — every one of them comes out of the
        // reserved chunk.
        if *slot != PhysicalApicEntry::new() {
            return Err(VlapicError::IdDescribedTwice { id: id.get() });
        }
        *slot = PhysicalApicEntry::new()
            .with_backing_page_address(page)
            .with_host_apic_id(host_identifier(id));
        Ok(())
    }

    /// The allocation order of the frame run this table fits in.
    pub(super) fn order(&self) -> usize {
        let bytes = self.entries.len() * size_of::<PhysicalApicEntry>();
        let pages = bytes.div_ceil(paging::as_usize(chunk::FRAME_SIZE));
        pages.next_power_of_two().trailing_zeros() as usize
    }

    /// Stores the table at `at`, which must be the frame run
    /// [`PhysicalTable::order`] asked for.
    ///
    /// Written entry by entry into the run rather than assembled into a byte
    /// image first. The entries are already a copy of the whole table — up to
    /// 32 KiB of them — and a second buffer beside them would cost that much
    /// heap again and a copy of it that nothing reads.
    ///
    /// # Errors
    ///
    /// [`PagingError`] if the window does not reach the whole run.
    pub(super) fn write(&self, window: DirectMap, at: PhysAddr) -> Result<(), PagingError> {
        let run =
            window.bytes_ptr::<u64>(at, self.entries.len() * size_of::<PhysicalApicEntry>())?;
        for (index, entry) in self.entries.iter().enumerate() {
            // SAFETY: the run was allocated out of the reserved chunk for this
            // table immediately before this call, so it is RAM, and nothing
            // else holds a reference into it or will until the tables are
            // published. `bytes_ptr` proved the window reaches every byte of
            // the run and that its base is aligned for a quadword, and `index`
            // is below the entry count that length was computed from.
            unsafe { run.add(index).write(entry.into_bits()) };
        }
        Ok(())
    }
}

/// The identifier the host delivers to for this processor: the same one,
/// because a virtual processor here runs on the physical processor that owns
/// it.
///
/// Narrowed to the field's width, which is provably enough: an entry is
/// written only at a position the table holds, and a table that was accepted
/// holds no position beyond twelve bits.
#[expect(
    clippy::cast_possible_truncation,
    reason = "an entry is only written at an index the table holds, and no accepted table has one beyond twelve bits"
)]
fn host_identifier(id: ApicId) -> u16 {
    id.get() as u16
}

#[cfg(test)]
mod tests {
    //! The table is pure until it is written to memory, so its sizing, its
    //! refusals and the shape of an entry are all tested without a machine.

    use cpu::ApicId;
    use svm::avic::{
        MAX_PHYSICAL_ID, PhysicalApicEntry, X2_EXTENDED_MAX_PHYSICAL_ID, X2_MAX_PHYSICAL_ID,
    };
    use x86_64::PhysAddr;

    use super::PhysicalTable;
    use crate::VlapicError;

    /// A limit no mode is narrower than, for the tests that are about sizing
    /// rather than about which mode may name what.
    const WIDEST: u16 = X2_EXTENDED_MAX_PHYSICAL_ID;

    #[test]
    fn the_table_is_sized_by_its_largest_index() {
        // One entry per identifier up to and including the largest, in whole
        // pages of them; the order is the run those pages fit in.
        for (max_index, entries, order) in [
            (0x0FE_u16, 255_usize, 0_usize),
            (0x1FF, 512, 0),
            (0x7FF, 2048, 2),
            (0xFFF, 4096, 3),
        ] {
            let table = PhysicalTable::new(max_index, WIDEST).expect("a table the mode can name");
            assert_eq!(table.entries.len(), entries, "{max_index:#x}");
            assert_eq!(table.order(), order, "{max_index:#x}");
        }
    }

    #[test]
    fn a_table_larger_than_a_control_block_can_name_is_refused() {
        let Err(error) = PhysicalTable::new(0x1000, WIDEST) else {
            panic!("a table past the largest index must be refused");
        };
        assert_eq!(error, VlapicError::TableTooLarge { entries: 4097 });
    }

    #[test]
    fn each_mode_refuses_the_index_it_cannot_name() {
        // Every boundary as a literal, because the mode's limit is exactly the
        // thing a wrong hex digit here would move: one byte of identifiers for
        // the eight-bit mode, one page of entries for the 32-bit one, and eight
        // pages where the extension reports the extended table.
        for (limit, over) in [
            (MAX_PHYSICAL_ID, 0x0FF_u16),
            (X2_MAX_PHYSICAL_ID, 0x200),
            (X2_EXTENDED_MAX_PHYSICAL_ID, 0x1000),
        ] {
            assert!(
                PhysicalTable::new(limit, limit).is_ok(),
                "the mode's own limit is a table it can name: {limit:#x}"
            );
            let Err(error) = PhysicalTable::new(over, limit) else {
                panic!("a table the mode cannot name must be refused: {over:#x}");
            };
            // The widest mode's limit is also the widest table a control block
            // can name, so past it the sizing rule answers first — which is the
            // one that says the entries would not fit.
            let expected = if usize::from(over) < super::MAX_ENTRIES {
                VlapicError::IndexBeyondMode {
                    max_index: over,
                    limit,
                }
            } else {
                VlapicError::TableTooLarge {
                    entries: usize::from(over) + 1,
                }
            };
            assert_eq!(error, expected, "{over:#x}");
        }
    }

    #[test]
    fn an_entry_names_the_processors_page_and_promises_nothing_beside_it() {
        let mut table = PhysicalTable::new(0xFF, WIDEST).expect("a table the mode can name");
        let page = PhysAddr::new(0x0012_3000);
        table.describe(ApicId::new(7), page).unwrap();
        let entry = table.entries[7];
        assert_eq!(entry.backing_page_address(), page);
        assert_eq!(entry.host_apic_id(), 7);
        // Both promises are the activation's to make: an interrupt may resolve
        // here only while the hardware is the authority for this controller, and
        // may be sent to a physical processor only while this one is on it.
        assert!(!entry.valid());
        assert!(!entry.is_running());
        // An entry nobody described names nothing at all, which is what the
        // description is recognised by.
        assert_eq!(table.entries[6], PhysicalApicEntry::new());
    }

    #[test]
    fn an_identifier_beyond_the_largest_index_is_refused() {
        let mut table = PhysicalTable::new(0xFF, WIDEST).expect("a table the mode can name");
        assert_eq!(
            table
                .describe(ApicId::new(0x100), PhysAddr::new(0))
                .unwrap_err(),
            VlapicError::IdBeyondTable {
                id: 0x100,
                max_index: 0xFF,
            },
        );
    }

    #[test]
    fn an_identifier_described_twice_is_refused_rather_than_overwritten() {
        // Firmware that describes one processor with both structure kinds. The
        // second description would name the second page in the one entry the
        // two share, while the first processor goes on being served out of the
        // first — so every interrupt to that identifier would land in a page
        // nothing reads.
        let mut table = PhysicalTable::new(0xFF, WIDEST).expect("a table the mode can name");
        let first = PhysAddr::new(0x0012_3000);
        let second = PhysAddr::new(0x0045_6000);
        table.describe(ApicId::new(7), first).unwrap();
        assert_eq!(
            table.describe(ApicId::new(7), second).unwrap_err(),
            VlapicError::IdDescribedTwice { id: 7 },
        );
        assert_eq!(table.entries[7].backing_page_address(), first);
        // And a different identifier is still describable afterwards: the
        // refusal is about the slot rather than about the table.
        table.describe(ApicId::new(8), second).unwrap();
    }
}
