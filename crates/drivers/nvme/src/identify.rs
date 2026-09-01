//! The two identify responses this driver answers for, and where in them the
//! identifying fields are.
//!
//! An identify response is four kilobytes the controller writes into a buffer
//! the guest chose, named by the physical region pointers the command
//! carried. The whole of it is the hardware's answer except three fields:
//! the controller's serial number (`NVMe` 2.0 §5.15.2, offset 4, twenty bytes)
//! and a namespace's NGUID and EUI64 (§5.15.4, offsets 104 and 120, sixteen
//! and eight bytes). Those are read out, replaced with what `spoof` makes of
//! them under the machine's seed, and written back, leaving every other byte
//! of the response exactly as the hardware wrote it.

use log::{info, warn};
use memory::{MemoryError, Physical, Written};
use x86_64::PhysAddr;

use crate::regs::PAGE;

/// Where the controller's serial number is in its identify response, and how
/// long it is.
const SERIAL: u64 = 4;
const SERIAL_BYTES: usize = 20;

/// Where a namespace's NGUID is in its identify response, and how long it is.
const NGUID: u64 = 104;
const NGUID_BYTES: usize = 16;

/// Where a namespace's EUI64 is, and how long it is.
const EUI64: u64 = 120;
const EUI64_BYTES: usize = 8;

/// An identify response's data buffer, in the guest's physical memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Response {
    /// Where the buffer begins. The guest chose it; it is page aligned or it
    /// is not, and if it is not the buffer runs to the end of that page and
    /// continues at the second pointer.
    prp1: PhysAddr,
    /// Where the buffer continues, once past the end of the first page.
    prp2: PhysAddr,
}

impl Response {
    /// A response buffer at the physical region pointers a command carried.
    #[must_use]
    pub(crate) const fn new(prp1: PhysAddr, prp2: PhysAddr) -> Self {
        Self { prp1, prp2 }
    }

    /// Replaces the controller's serial number with another of the same
    /// shape.
    ///
    /// A failure is warned about and that is all: the response reaches the
    /// guest as the hardware wrote it, which is the state of affairs this
    /// driver exists to end, so the guest still runs and the boot still
    /// finishes. The original is never logged — a serial number is what this
    /// whole crate exists to keep off the record.
    pub(crate) fn controller(&self, physical: &Physical) {
        let serial = self.field::<SERIAL_BYTES>(physical, SERIAL);
        let serial = match serial {
            Ok(serial) => serial,
            Err(error) => {
                warn!(
                    "nvme: the serial number could not be read at {:#x}: {error}",
                    self.prp1.as_u64() + SERIAL
                );
                return;
            }
        };
        let replaced = spoof::text(&serial, &config::serial_seed());
        info!(
            "nvme: the serial number is replaced with {:?}",
            shown(&replaced)
        );
        self.replaced(physical, SERIAL, &replaced, "serial number");
    }

    /// Replaces a namespace's two identifiers with others of the same shape.
    ///
    /// A field of zeroes means the namespace has no such identifier, and
    /// [`spoof::bytes`] leaves it exactly as it was: an absent identifier
    /// must not gain a value.
    pub(crate) fn namespace(&self, physical: &Physical) {
        let nguid = match self.field::<NGUID_BYTES>(physical, NGUID) {
            Ok(nguid) => nguid,
            Err(error) => {
                warn!(
                    "nvme: the NGUID could not be read at {:#x}: {error}",
                    self.prp1.as_u64() + NGUID
                );
                return;
            }
        };
        let eui64 = match self.field::<EUI64_BYTES>(physical, EUI64) {
            Ok(eui64) => eui64,
            Err(error) => {
                warn!(
                    "nvme: the EUI64 could not be read at {:#x}: {error}",
                    self.prp1.as_u64() + EUI64
                );
                return;
            }
        };
        let seed = config::serial_seed();
        let nguid_replaced = spoof::bytes(&nguid, &seed);
        if nguid_replaced != nguid {
            info!("nvme: the NGUID is replaced with {nguid_replaced:02x?}");
        }
        self.replaced(physical, NGUID, &nguid_replaced, "NGUID");
        let eui64_replaced = spoof::bytes(&eui64, &seed);
        if eui64_replaced != eui64 {
            info!("nvme: the EUI64 is replaced with {eui64_replaced:02x?}");
        }
        self.replaced(physical, EUI64, &eui64_replaced, "EUI64");
    }

    /// Writes one field's replacement back, warning about every way it could
    /// not have landed.
    fn replaced<const BYTES: usize>(
        &self,
        physical: &Physical,
        at: u64,
        with: &[u8; BYTES],
        named: &str,
    ) {
        match self.replace(physical, at, with) {
            Ok(Written::Committed) => {}
            Ok(Written::Discarded) => warn!(
                "nvme: the guest's memory refused the {named}'s replacement at {:#x}",
                self.prp1.as_u64() + at
            ),
            Err(error) => warn!(
                "nvme: the {named} could not be replaced at {:#x}: {error}",
                self.prp1.as_u64() + at
            ),
        }
    }

    /// Reads the field `BYTES` long at `at` out of the response.
    ///
    /// # Errors
    ///
    /// Whatever reaching the guest's memory reports, or
    /// [`MemoryError::Range`] if the command's pointers do not describe a
    /// range inside the physical address space.
    fn field<const BYTES: usize>(
        &self,
        physical: &Physical,
        at: u64,
    ) -> Result<[u8; BYTES], MemoryError> {
        let mut field = [0; BYTES];
        let mut filled = 0;
        for (where_, bytes) in pieces(self.prp1, self.prp2, at, BYTES as u64) {
            if bytes == 0 {
                continue;
            }
            let where_ =
                PhysAddr::try_new(where_).map_err(|_| MemoryError::Range { gpa: where_, bytes })?;
            let bytes = as_usize(bytes);
            physical.read(where_, &mut field[filled..][..bytes])?;
            filled += bytes;
        }
        Ok(field)
    }

    /// Writes a field back where it was read from.
    ///
    /// # Errors
    ///
    /// Whatever reaching the guest's memory reports. A range the guest's
    /// memory declines is not an error — [`Written::Discarded`] says so, and
    /// the field is left as the hardware wrote it.
    fn replace<const BYTES: usize>(
        &self,
        physical: &Physical,
        at: u64,
        with: &[u8; BYTES],
    ) -> Result<Written, MemoryError> {
        let mut written = 0;
        let mut reached = Written::Committed;
        for (where_, bytes) in pieces(self.prp1, self.prp2, at, BYTES as u64) {
            if bytes == 0 {
                continue;
            }
            let where_ =
                PhysAddr::try_new(where_).map_err(|_| MemoryError::Range { gpa: where_, bytes })?;
            let bytes = as_usize(bytes);
            if physical.write(where_, &with[written..][..bytes])? == Written::Discarded {
                reached = Written::Discarded;
            }
            written += bytes;
        }
        Ok(reached)
    }
}

/// A field's bytes as the text a serial number is, with anything
/// unprintable standing in for itself.
fn shown(field: &[u8]) -> alloc::string::String {
    field
        .iter()
        .map(|&byte| {
            if byte.is_ascii_graphic() || byte == b' ' {
                byte as char
            } else {
                '.'
            }
        })
        .collect()
}

/// A field's length as `usize`.
///
/// Every length here is one identify field's, at most twenty bytes, so the
/// cast loses nothing on any target a field could fit in memory on.
#[expect(
    clippy::cast_possible_truncation,
    reason = "a field's length is at most twenty bytes, which a usize holds on any target this crate runs on"
)]
const fn as_usize(bytes: u64) -> usize {
    bytes as usize
}

/// Where a range of the response's data buffer is, in one or two pieces.
///
/// The buffer starts at `prp1` and, once past the end of that page, continues
/// at `prp2` — which is how a physical region list describes a buffer that is
/// not page aligned. A range therefore lands in one of three shapes: inside
/// the first page, straddling the end of it, or — where the buffer began near
/// its page's end and the range starts past that end — entirely at `prp2`,
/// however far in. An empty piece's address is meaningless.
///
/// The buffer is one identify response's four kilobytes, and the ranges this
/// answers for start within the first hundred and twenty-eight bytes of it,
/// so no range can leave the second piece's page: the furthest in a range can
/// begin is a hundred and twenty-eight bytes, and the longest is twenty.
#[must_use]
fn pieces(prp1: PhysAddr, prp2: PhysAddr, at: u64, bytes: u64) -> [(u64, u64); 2] {
    let start = prp1.as_u64() + at;
    let page = (prp1.as_u64() & !(PAGE - 1)) + PAGE;
    if start >= page {
        // Entirely past the first page: at `prp2`, as far in as the range is
        // past the end. The subtraction is in range because the branch says
        // so.
        [(prp2.as_u64() + (start - page), bytes), (0, 0)]
    } else if start + bytes <= page {
        [(start, bytes), (0, 0)]
    } else {
        // Straddling: to the end of the first page, then the rest from the
        // start of the second. Both subtractions are in range, because the
        // two branches above say so.
        let first = page - start;
        [(start, first), (prp2.as_u64(), bytes - first)]
    }
}

#[cfg(test)]
mod tests {
    use x86_64::PhysAddr;

    use super::pieces;

    /// A page-aligned first pointer, whose buffer needs no second.
    const ALIGNED: PhysAddr = PhysAddr::new(0x5000);

    /// A first pointer mid-page, whose buffer runs to the page's end and
    /// continues elsewhere.
    const SPLIT: PhysAddr = PhysAddr::new(0x5830);

    /// Where a split buffer continues.
    const SECOND: PhysAddr = PhysAddr::new(0x9000);

    #[test]
    fn a_range_inside_the_first_page_is_one_piece() {
        assert_eq!(pieces(ALIGNED, SECOND, 4, 20), [(0x5004, 20), (0, 0)]);
        assert_eq!(pieces(SPLIT, SECOND, 8, 8), [(0x5838, 8), (0, 0)]);
    }

    #[test]
    fn a_range_reaching_the_pages_end_is_still_one_piece() {
        // Eight bytes at 0xff8 of the page: the last eight it holds.
        assert_eq!(pieces(ALIGNED, SECOND, 0xff8, 8), [(0x5ff8, 8), (0, 0)]);
    }

    #[test]
    fn a_range_past_the_pages_end_is_two() {
        assert_eq!(pieces(SPLIT, SECOND, 0x7d0, 20), [(0x9000, 20), (0, 0)]);
        assert_eq!(
            pieces(SPLIT, SECOND, 0x7c0, 32),
            [(0x5ff0, 16), (0x9000, 16)]
        );
    }

    #[test]
    fn a_range_wholly_past_the_first_page_is_at_the_second() {
        // A buffer beginning sixteen bytes before its page's end: the NGUID
        // of a namespace, a hundred and four bytes into the response, is
        // entirely past that end and into the second page. This is the shape
        // that once subtracted its way to a panic.
        let near = PhysAddr::new(0x5ff0);
        assert_eq!(pieces(near, SECOND, 104, 16), [(0x9058, 16), (0, 0)]);
        // The EUI64, eight bytes further on, likewise.
        assert_eq!(pieces(near, SECOND, 120, 8), [(0x9068, 8), (0, 0)]);
        // And a field straddling the first page's end still splits.
        assert_eq!(pieces(near, SECOND, 4, 20), [(0x5ff4, 12), (0x9000, 8)]);
    }
}
