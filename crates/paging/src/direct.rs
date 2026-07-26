//! Linear window onto physical memory.
//!
//! One contiguous high-half range covers physical `0..top_of_ram`, so any
//! physical address can be read or written by adding a constant. That is what
//! the user asked it for — quick reads and in-place edits of arbitrary physical
//! memory — and it is deliberately *not* how memory that gets used repeatedly
//! should be reached: those want a real mapping with their own protection and
//! cache type, from `AddressSpace::map_physical`. The direct map is fixed at
//! read-write, no-execute, write-back for everything it covers.
//!
//! It covers RAM and nothing else. Device apertures are excluded on both
//! counts: they can sit terabytes above the last byte of RAM, so reaching them
//! would cost page tables proportional to the gap, and a cached, always-present
//! window is the wrong way to touch device registers in the first place.
//!
//! It is also load-bearing rather than a convenience. The page tables and the
//! allocator state live in the reserved chunk at whatever low physical address
//! firmware handed out, so once the firmware half of the address space is gone
//! the direct map is the only way left to reach them. This is why it is built
//! before anything else and why it never contains global entries: it has to
//! stay flushable.
//!
//! The same type also describes the *identity* relation firmware set up, which
//! is how the loader reaches page-table frames while it is still running under
//! firmware's address space and building ours.

use core::ptr::{self, NonNull};

use x86_64::{
    PhysAddr, VirtAddr,
    structures::paging::{PageTable, PhysFrame, mapper::PageTableFrameMapping},
};

use crate::{PagingError, as_u64};

/// A linear physical-to-virtual relation: `virt = base + phys`, valid for the
/// first `size` bytes of physical memory.
#[derive(Clone, Copy, Debug)]
pub struct DirectMap {
    base: VirtAddr,
    size: u64,
}

impl DirectMap {
    /// A window at `base` covering physical `0..size`.
    #[must_use]
    pub const fn new(base: VirtAddr, size: u64) -> Self {
        Self { base, size }
    }

    /// The relation firmware's identity map provides, where a physical address
    /// *is* its virtual address.
    ///
    /// The size is the whole lower canonical half rather than a claim about how
    /// much firmware actually mapped: it only has to admit the chunk, and every
    /// address the loader reaches through it is one firmware just handed out.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            base: VirtAddr::zero(),
            size: 1 << 47,
        }
    }

    /// Virtual base of the window.
    #[must_use]
    pub const fn base(&self) -> VirtAddr {
        self.base
    }

    /// Bytes of physical memory the window covers.
    #[must_use]
    pub const fn size(&self) -> u64 {
        self.size
    }

    /// Where `phys` is readable, or `None` if the window does not reach it.
    #[must_use]
    pub fn virt(&self, phys: PhysAddr) -> Option<VirtAddr> {
        (phys.as_u64() < self.size).then(|| self.base + phys.as_u64())
    }

    /// Where `phys` is readable, as a pointer to `T`.
    ///
    /// Returns `None` if the window does not reach `phys`, or if `phys` is not
    /// aligned for `T` — an unaligned pointer into physical memory is always a
    /// mistake in the caller rather than something to paper over.
    #[must_use]
    pub fn ptr<T>(&self, phys: PhysAddr) -> Option<NonNull<T>> {
        let virt = self.virt(phys)?;
        (virt.as_u64() % align_of::<T>() as u64 == 0)
            .then(|| NonNull::new(virt.as_mut_ptr::<T>()))
            .flatten()
    }

    /// Copies `into.len()` bytes of physical memory out of the window.
    ///
    /// # Errors
    ///
    /// [`PagingError::Unreachable`] if any byte of the range falls outside the
    /// window. The whole range is checked, not just where it starts.
    ///
    /// # Safety
    ///
    /// The range must be memory the caller is entitled to read, and none of it
    /// may alias anything the hypervisor holds a Rust reference to. A range
    /// something else is writing at the same time yields what a non-atomic copy
    /// of it would, which is the answer the hardware gives too.
    pub unsafe fn read(&self, phys: PhysAddr, into: &mut [u8]) -> Result<(), PagingError> {
        let from = self.reach(phys, into.len())?;
        // SAFETY: `reach` proved every byte of the source lies inside the window,
        // so it is valid for `into.len()` bytes. The caller guarantees the range
        // is theirs to read and aliases no live reference, which is what rules
        // out the destination overlapping it.
        unsafe { ptr::copy_nonoverlapping(from.as_ptr::<u8>(), into.as_mut_ptr(), into.len()) };
        Ok(())
    }

    /// Copies `from.len()` bytes of physical memory into the window.
    ///
    /// # Errors
    ///
    /// [`PagingError::Unreachable`] if any byte of the range falls outside the
    /// window.
    ///
    /// # Safety
    ///
    /// The range must be memory the caller owns or is entitled to overwrite,
    /// and none of it may alias anything the hypervisor holds a Rust
    /// reference to.
    pub unsafe fn write(&self, phys: PhysAddr, from: &[u8]) -> Result<(), PagingError> {
        let into = self.reach(phys, from.len())?;
        // SAFETY: as in `read`, with the roles reversed: the destination is the
        // range `reach` checked, and the caller guarantees it is theirs to write
        // and aliases nothing.
        unsafe { ptr::copy_nonoverlapping(from.as_ptr(), into.as_mut_ptr::<u8>(), from.len()) };
        Ok(())
    }

    /// Where a run of `len` bytes from `phys` begins, provided all of it is
    /// inside the window.
    ///
    /// [`DirectMap::virt`] answers for one address, which is not the same
    /// question: a range starting just below the top of the window can end
    /// above it, and a copy that trusted the start alone would run off the
    /// end of the mapping into whatever the next region is.
    fn reach(&self, phys: PhysAddr, len: usize) -> Result<VirtAddr, PagingError> {
        let unreachable = || PagingError::Unreachable {
            phys: phys.as_u64(),
        };
        let end = phys
            .as_u64()
            .checked_add(as_u64(len))
            .ok_or_else(unreachable)?;
        if end > self.size {
            return Err(unreachable());
        }
        self.virt(phys).ok_or_else(unreachable)
    }
}

// SAFETY: `frame_to_pointer` returns `base + frame`, which is exactly where the
// window maps that frame, and the caller of `MappedPageTable::new` is required
// to have built the window before handing it over. The window is read-write and
// covers whole frames, so a returned pointer is valid for a `PageTable`, whose
// alignment is a frame's.
unsafe impl PageTableFrameMapping for DirectMap {
    /// Page-table frames always come out of the reserved chunk, which lies
    /// below `top_of_ram` and so inside the window; the `None` case is
    /// unreachable. It is still handled rather than asserted, because the
    /// signature cannot report a failure and the hypervisor must not panic:
    /// a null pointer faults at a recognizable address and is reported by the
    /// page-fault handler, where a truncated address would silently corrupt
    /// whatever it landed on.
    fn frame_to_pointer(&self, frame: PhysFrame) -> *mut PageTable {
        self.virt(frame.start_address())
            .map_or_else(core::ptr::null_mut, VirtAddr::as_mut_ptr)
    }
}
