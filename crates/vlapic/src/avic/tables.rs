//! The physical APIC ID table: one entry per identifier an interrupt may
//! name.
//!
//! An interrupt addressed to a processor is an index into this table, and the
//! entry answers two questions the hardware cannot ask anybody: where that
//! processor's controller registers are backed, and whether it is running
//! anywhere at the moment. The second is the one that changes — Phase one
//! writes every entry not running, and nothing here ever sets the bit;
//! turning a processor's entry to running belongs to whoever puts it on one.
//!
//! The table is a run of whole pages, because the pointer a control block
//! carries names its first page and the architecture gives it no length but
//! the largest valid index beside it.

use alloc::{boxed::Box, vec, vec::Vec};

use cpu::ApicId;
use paging::{DirectMap, PagingError, chunk};
use svm::avic::PhysicalApicEntry;
use x86_64::PhysAddr;

use crate::VlapicError;

/// How many entries the table may hold.
///
/// The field beside the table's address names the largest valid index in
/// twelve bits, so the hardware can walk one page of entries and no more —
/// and a table asked to be larger is one a control block could not describe
/// anyway.
pub(crate) const MAX_ENTRIES: usize = 4096;

/// The table under construction: its entries, and the largest index in them
/// that is valid.
pub(crate) struct PhysicalTable {
    max_index: u16,
    entries: Box<[PhysicalApicEntry]>,
}

impl PhysicalTable {
    /// An empty table whose largest valid index will be `max_index`.
    ///
    /// # Errors
    ///
    /// [`VlapicError::TableTooLarge`] if the entries would not fit one page.
    pub(super) fn new(max_index: u16) -> Result<Self, VlapicError> {
        let count = usize::from(max_index) + 1;
        if count > MAX_ENTRIES {
            return Err(VlapicError::TableTooLarge { entries: count });
        }
        Ok(Self {
            max_index,
            entries: vec![PhysicalApicEntry::new(); count].into_boxed_slice(),
        })
    }

    /// Describes the processor `id` as one an interrupt may reach, with its
    /// registers backed by `page`.
    ///
    /// Written not running: nothing is on any physical processor yet, and a
    /// set running bit before the first entry would hand the hardware a
    /// promise that does not hold.
    ///
    /// # Errors
    ///
    /// [`VlapicError::IdBeyondTable`] if the identifier is past the largest
    /// index this table was sized for, which the table cannot express.
    pub(super) fn describe(&mut self, id: ApicId, page: PhysAddr) -> Result<(), VlapicError> {
        let Some(slot) = self.entries.get_mut(id.get() as usize) else {
            return Err(VlapicError::IdBeyondTable {
                id: id.get(),
                max_index: self.max_index,
            });
        };
        *slot = PhysicalApicEntry::new()
            .with_valid(true)
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
    /// # Errors
    ///
    /// [`PagingError`] if the window does not reach the whole run.
    pub(super) fn write(&self, window: DirectMap, at: PhysAddr) -> Result<(), PagingError> {
        let mut bytes = Vec::with_capacity(self.entries.len() * size_of::<PhysicalApicEntry>());
        for entry in &self.entries {
            bytes.extend_from_slice(&entry.into_bits().to_le_bytes());
        }
        // SAFETY: the run was allocated out of the reserved chunk for this
        // table immediately before this call, so it is RAM, it is zeroed, and
        // nothing else holds a reference to it or will until the tables are
        // published.
        unsafe { window.write(at, &bytes) }
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
    use x86_64::PhysAddr;

    use super::PhysicalTable;
    use crate::VlapicError;

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
            let table = PhysicalTable::new(max_index).expect("a table that fits one page");
            assert_eq!(table.entries.len(), entries, "{max_index:#x}");
            assert_eq!(table.order(), order, "{max_index:#x}");
        }
    }

    #[test]
    fn a_table_larger_than_one_page_is_refused() {
        let Err(error) = PhysicalTable::new(0x1000) else {
            panic!("a table past the largest index must be refused");
        };
        assert_eq!(error, VlapicError::TableTooLarge { entries: 4097 });
    }

    #[test]
    fn an_entry_names_the_processors_page_and_starts_not_running() {
        let mut table = PhysicalTable::new(0xFF).expect("a table that fits one page");
        let page = PhysAddr::new(0x0012_3000);
        table.describe(ApicId::new(7), page).unwrap();
        let entry = table.entries[7];
        assert!(entry.valid());
        assert!(!entry.is_running());
        assert_eq!(entry.backing_page_address(), page);
        assert_eq!(entry.host_apic_id(), 7);
        // An entry nobody described names nothing.
        assert!(!table.entries[6].valid());
    }

    #[test]
    fn an_identifier_beyond_the_largest_index_is_refused() {
        let mut table = PhysicalTable::new(0xFF).expect("a table that fits one page");
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
}
