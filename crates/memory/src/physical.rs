//! A guest's physical memory.
//!
//! The lower of the two views: an address the guest believes is physical,
//! translated by the nested tables and reached through the window onto physical
//! memory. Everything above it — the guest's own page tables, and whatever
//! computed the address in the first place — ends up here.
//!
//! # Nothing is copied a page at a time unless it has to be
//!
//! A translation says how far the entry behind it reaches, so a copy inside a
//! 1 GiB identity page is one translation and one move rather than a quarter of
//! a million of each. The loops below advance by whatever the last translation
//! promised, which makes the common case — an operand that fits in one page —
//! exactly one translation.

use npt::Npt;
use paging::DirectMap;
use x86_64::PhysAddr;

use crate::{MemoryError, as_u64, reachable};

/// A guest's physical memory.
#[derive(Clone, Copy, Debug)]
pub struct Physical<'a> {
    npt: &'a Npt,
    window: DirectMap,
}

impl<'a> Physical<'a> {
    /// The physical memory of the guest those tables describe, reached through
    /// that window.
    #[must_use]
    pub const fn new(npt: &'a Npt, window: DirectMap) -> Self {
        Self { npt, window }
    }

    /// Where a guest physical address really is, and how far the same entry
    /// reaches past it.
    ///
    /// # Errors
    ///
    /// [`MemoryError::Undescribed`] if nothing describes the address yet, or
    /// [`MemoryError::Npt`] if the tables themselves could not be walked.
    pub fn translate(&self, gpa: PhysAddr) -> Result<npt::Translation, MemoryError> {
        self.npt
            .translate(gpa)?
            .ok_or(MemoryError::Undescribed { gpa: gpa.as_u64() })
    }

    /// The region something other than the hardware answers for that a guest
    /// physical address is in, or `None` if the hardware answers for it.
    ///
    /// Not a question about memory, and here because this is the handle onto
    /// the tables that record it: an emulated access has to know which
    /// device answers for the address it lands on, and the tables are the
    /// one place where a region's name and its extent are kept.
    #[must_use]
    pub fn region(&self, gpa: PhysAddr) -> Option<npt::Answered> {
        self.npt.region(gpa)
    }

    /// Copies `into.len()` bytes of the guest's physical memory.
    ///
    /// # Errors
    ///
    /// [`MemoryError::Undescribed`] naming the first address of the range that
    /// nothing describes, [`MemoryError::Range`] if the range leaves the
    /// physical address space, or [`MemoryError::Paging`] if the window does
    /// not reach where the range translates to.
    pub fn read(&self, gpa: PhysAddr, into: &mut [u8]) -> Result<(), MemoryError> {
        let mut rest = into;
        let mut at = gpa.as_u64();
        while !rest.is_empty() {
            let there = self.translate(address(gpa, at, as_u64(rest.len()))?)?;
            let (piece, tail) = rest.split_at_mut(reachable(there.span, rest.len()));
            // SAFETY: the address came from the nested tables, which describe a
            // guest's memory and the shared page of zeroes and nothing else — so
            // it cannot name anything the hypervisor holds a reference to, and
            // `piece` is a distinct borrow of the caller's buffer.
            unsafe { self.window.read(there.spa, piece) }?;
            at += as_u64(piece.len());
            rest = tail;
        }
        Ok(())
    }

    /// Copies `from.len()` bytes into the guest's physical memory, or discards
    /// the write if any of the range is not the guest's to write.
    ///
    /// # Why writability is checked before anything is written
    ///
    /// A range the guest may not write is not an error — it is the hypervisor's
    /// own memory, which the guest sees as zeroes and cannot change. The write
    /// simply does not happen, and saying so is the whole of the answer.
    ///
    /// Which is why the whole range is checked first. Writing until the refusal
    /// is met would leave the earlier bytes written and report that nothing
    /// was, and a caller acting on that answer would be acting on a lie about
    /// state it can no longer see.
    ///
    /// # Errors
    ///
    /// As [`Physical::read`].
    pub fn write(&self, gpa: PhysAddr, from: &[u8]) -> Result<Written, MemoryError> {
        if !self.writable(gpa, from.len())? {
            return Ok(Written::Discarded);
        }
        let mut rest = from;
        let mut at = gpa.as_u64();
        while !rest.is_empty() {
            let there = self.translate(address(gpa, at, as_u64(rest.len()))?)?;
            let (piece, tail) = rest.split_at(reachable(there.span, rest.len()));
            // SAFETY: as in `read`, and the range was just shown to be one the
            // nested tables let the guest write — so it is the guest's memory
            // rather than the shared page of zeroes standing in for ours.
            unsafe { self.window.write(there.spa, piece) }?;
            at += as_u64(piece.len());
            rest = tail;
        }
        Ok(Written::Committed)
    }

    /// Whether every byte of a range is the guest's to write.
    ///
    /// What [`Physical::write`] asks itself before writing anything, exposed
    /// because the view above this one has to ask the same question of each
    /// piece a range breaks into before it commits to any of them.
    ///
    /// # Errors
    ///
    /// As [`Physical::translate`], for the first address of the range that
    /// nothing describes.
    pub fn writable(&self, gpa: PhysAddr, bytes: usize) -> Result<bool, MemoryError> {
        let mut left = as_u64(bytes);
        let mut at = gpa.as_u64();
        while left != 0 {
            let there = self.translate(address(gpa, at, left)?)?;
            if !there.writable {
                return Ok(false);
            }
            let step = there.span.min(left);
            at += step;
            left -= step;
        }
        Ok(true)
    }
}

/// What became of a write to a guest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Written {
    /// The bytes are in the guest's memory.
    Committed,
    /// Some of the range was not the guest's to write, so none of it was
    /// written. The guest's own view of that memory is unchanged, which for the
    /// pages that stand in for the hypervisor's own means it goes on reading
    /// zeroes.
    Discarded,
}

/// One address of a range, as a [`PhysAddr`].
///
/// `start` and `bytes` are carried only so that a range running off the end of
/// the physical address space is reported as the range it was rather than as
/// whichever byte of it first failed to exist.
fn address(start: PhysAddr, at: u64, bytes: u64) -> Result<PhysAddr, MemoryError> {
    PhysAddr::try_new(at).map_err(|_| MemoryError::Range {
        gpa: start.as_u64(),
        bytes,
    })
}
