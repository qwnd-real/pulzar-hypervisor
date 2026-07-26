//! Handing out the tags a guest's cached translations are kept apart by.
//!
//! Translations in the processor's buffers are tagged with an address space
//! identifier, which is what lets a guest's survive the hypervisor running and
//! another guest's coexist with them. Without that, every entry into a guest
//! and every exit from one would have to throw the buffers away.
//!
//! The tag is per address space rather than per virtual processor: under nested
//! paging a guest's identifier goes with its *physical* address space, so every
//! processor of one guest shares one, and a guest keeps it across all of its
//! own virtual address spaces. That is why this is allocated once per partition
//! and not once per virtual processor.
//!
//! # Zero is not available
//!
//! Identifier zero is the host's, and it is what the processor reverts to on
//! every exit. A control block naming it is refused outright, so the allocator
//! starts at one rather than treating zero as a valid answer that happens to
//! fail later.
//!
//! # How many there are is the processor's to say
//!
//! The count comes from the extension's own feature leaf and differs between
//! machines by more than an order of magnitude. A machine reporting fewer than
//! two can run no guest at all, which is checked where the extension is enabled
//! rather than here.

use core::sync::atomic::{AtomicU32, Ordering};

use crate::PartitionError;

/// The tag one guest's cached translations carry.
///
/// A newtype rather than a bare number because the value has a rule attached
/// that a number does not carry: zero is the host's. Nothing can construct one
/// holding zero, so a control block programmed from one of these cannot name
/// the host's address space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Asid(u32);

impl Asid {
    /// The identifier as a control block holds it.
    #[must_use]
    pub const fn number(self) -> u32 {
        self.0
    }
}

/// The machine's supply of identifiers.
///
/// One counter, handing out ascending values and never reusing one. Reuse is a
/// real requirement for a hypervisor that creates and destroys guests, because
/// an identifier handed back carries whatever the previous guest left in the
/// translation buffers — but nothing here destroys a guest, and an allocator
/// with a free list nothing ever puts anything on is a free list that has never
/// been tested.
#[derive(Debug)]
pub struct Asids {
    next: AtomicU32,
    count: u32,
}

impl Asids {
    /// The supply a processor reporting `count` identifiers has.
    ///
    /// `count` is what the extension's feature leaf reports, so it counts the
    /// host's own identifier along with the guests'.
    #[must_use]
    pub const fn new(count: u32) -> Self {
        Self {
            next: AtomicU32::new(FIRST_GUEST),
            count,
        }
    }

    /// An identifier no guest has been given.
    ///
    /// # Errors
    ///
    /// [`PartitionError::OutOfAsids`] once every identifier the processor
    /// supports has been handed out.
    pub fn take(&self) -> Result<Asid, PartitionError> {
        let asid = self.next.fetch_add(1, Ordering::Relaxed);
        if asid >= self.count {
            // Put it back, so that a machine that has run out reports running
            // out rather than eventually wrapping onto the host's identifier.
            self.next.store(self.count, Ordering::Relaxed);
            return Err(PartitionError::OutOfAsids { count: self.count });
        }
        Ok(Asid(asid))
    }
}

/// The lowest identifier a guest may be given, zero being the host's.
const FIRST_GUEST: u32 = 1;

const _: () = assert!(
    FIRST_GUEST != 0,
    "address space zero is the host's and no guest may be tagged with it",
);
