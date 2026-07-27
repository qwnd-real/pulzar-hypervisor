//! A guest's memory at the addresses the guest itself uses.
//!
//! The upper of the two views, and the one almost everything wants: an address
//! out of a guest's register is a linear address, and turning it into something
//! reachable takes the guest's own translation before the hypervisor's.
//!
//! A linear address is a `u64` here and not a
//! [`VirtAddr`](x86_64::VirtAddr), for the same reason the nested walker keeps
//! guest physical addresses in one: constructing a `VirtAddr` panics on a value
//! that is not canonical, a guest is perfectly capable of computing one, and a
//! panic is not an acceptable answer to a guest doing arithmetic. An address
//! that cannot be walked is reported as an address that cannot be walked.

use x86_64::PhysAddr;

use crate::{Addressing, MemoryError, Physical, Written, as_u64, as_usize, reachable, walk};

/// A guest's memory, at the addresses the guest uses for it.
#[derive(Clone, Copy, Debug)]
pub struct Linear<'a> {
    physical: Physical<'a>,
    addressing: Addressing,
}

impl<'a> Linear<'a> {
    /// That guest's memory, translated the way that guest translates.
    #[must_use]
    pub const fn new(physical: Physical<'a>, addressing: Addressing) -> Self {
        Self {
            physical,
            addressing,
        }
    }

    /// The same memory at the addresses the guest believes are physical.
    #[must_use]
    pub const fn physical(&self) -> Physical<'a> {
        self.physical
    }

    /// How this guest translates.
    #[must_use]
    pub const fn addressing(&self) -> &Addressing {
        &self.addressing
    }

    /// Where a linear address lands in the guest's physical memory.
    ///
    /// # Errors
    ///
    /// Whatever walking the guest's own tables reports: an address they do not
    /// describe, an address no walk is defined for, or a failure to read the
    /// tables themselves.
    pub fn translate(&self, linear: u64) -> Result<PhysAddr, MemoryError> {
        Ok(walk::walk(self.physical, &self.addressing, linear)?.gpa)
    }

    /// Copies `into.len()` bytes of the guest's memory.
    ///
    /// The range may cross as many of the guest's pages as it likes, and pages
    /// next to each other in the guest need not be anywhere near each other
    /// underneath — which is the whole reason this is a loop and not an address
    /// calculation.
    ///
    /// # Errors
    ///
    /// As [`Linear::translate`], and as [`Physical::read`] for the memory the
    /// walk arrives at.
    pub fn read(&self, linear: u64, into: &mut [u8]) -> Result<(), MemoryError> {
        let mut rest = into;
        let mut at = linear;
        while !rest.is_empty() {
            let there = walk::walk(self.physical, &self.addressing, at)?;
            let (piece, tail) = rest.split_at_mut(reachable(there.span, rest.len()));
            self.physical.read(there.gpa, piece)?;
            at = at.wrapping_add(as_u64(piece.len()));
            rest = tail;
        }
        Ok(())
    }

    /// Copies `from.len()` bytes into the guest's memory, or discards the write
    /// if any of the range is not the guest's to write.
    ///
    /// All of it or none of it, and for the reason [`Physical::write`] gives:
    /// a partly completed write reported as no write at all is worse than
    /// either honest answer.
    ///
    /// # Errors
    ///
    /// As [`Linear::read`].
    pub fn write(&self, linear: u64, from: &[u8]) -> Result<Written, MemoryError> {
        if !self.writable(linear, from.len())? {
            return Ok(Written::Discarded);
        }
        let mut rest = from;
        let mut at = linear;
        while !rest.is_empty() {
            let there = walk::walk(self.physical, &self.addressing, at)?;
            let (piece, tail) = rest.split_at(reachable(there.span, rest.len()));
            self.physical.write(there.gpa, piece)?;
            at = at.wrapping_add(as_u64(piece.len()));
            rest = tail;
        }
        Ok(Written::Committed)
    }

    /// Whether every byte of a range is the guest's to write.
    fn writable(&self, linear: u64, bytes: usize) -> Result<bool, MemoryError> {
        let mut left = as_u64(bytes);
        let mut at = linear;
        while left != 0 {
            let there = walk::walk(self.physical, &self.addressing, at)?;
            let step = there.span.min(left);
            if !self.physical.writable(there.gpa, as_usize(step))? {
                return Ok(false);
            }
            at = at.wrapping_add(step);
            left -= step;
        }
        Ok(true)
    }
}
