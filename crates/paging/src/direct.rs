//! Linear window onto physical memory.
//!
//! One contiguous high-half range covers physical `0..size`, so any physical
//! address inside it can be read or written by adding a constant. That is what
//! it is for — quick reads and in-place edits of arbitrary physical memory —
//! and it is deliberately *not* how memory that gets used repeatedly should be
//! reached: those want a real mapping with their own protection and cache type,
//! from `AddressSpace::map_physical`. The direct map is fixed at read-write,
//! no-execute, write-back for everything it covers.
//!
//! It covers RAM and nothing else. Device apertures are excluded on both
//! counts: they can sit terabytes above the last byte of RAM, so reaching them
//! would cost page tables proportional to the gap, and a cached, always-present
//! window is the wrong way to touch device registers in the first place.
//! Nothing here may be used to reach a device: a write-back alias of an
//! aperture is an architecturally conflicting memory type even if nothing ever
//! reads it.
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
//!
//! # Numeric reach is not membership
//!
//! [`DirectMap::size`] is how far the window's arithmetic goes, not a claim
//! that every address below it is RAM the window maps. The address space builds
//! the map from the memory ranges firmware described, so the holes between them
//! are absent from the page tables. An address inside a hole is one this type
//! will happily compute a virtual address for and the hardware will fault on,
//! which is the right order for those two things to happen in.
//!
//! # What the checks here do and do not cover
//!
//! Every accessor answers for a whole range rather than for where it starts: a
//! run beginning just below the top of the window ends above it, and a copy
//! that trusted the start alone would run off the end of the mapping into
//! whatever the next region is.
//!
//! None of them says anything about concurrency. The window is a shared,
//! always-present alias of all of physical memory; two processors reaching the
//! same physical address through it get exactly what two processors writing the
//! same memory get. Whoever forms a reference through it owns the exclusion.

use core::ptr::{self, NonNull};

use x86_64::{
    PhysAddr, VirtAddr,
    structures::paging::{PageSize, PageTable, PhysFrame, Size4KiB, mapper::PageTableFrameMapping},
};

use crate::{PagingError, as_u64, end_of};

/// A linear physical-to-virtual relation: `virt = base + phys`, valid for the
/// first `size` bytes of physical memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectMap {
    base: VirtAddr,
    size: u64,
}

impl DirectMap {
    /// A window at `base` covering physical `0..size`.
    ///
    /// # Errors
    ///
    /// [`PagingError::EmptyRegion`] for a window covering nothing, or
    /// [`PagingError::Arithmetic`] if the whole span `base..base + size` is not
    /// one run of canonical addresses. Only the base is given, so only checking
    /// the base is what would let a window be constructed whose far end is in
    /// the canonical hole — and every accessor below would then be validating
    /// against a top that does not exist.
    pub fn new(base: VirtAddr, size: u64) -> Result<Self, PagingError> {
        if size == 0 {
            return Err(PagingError::EmptyRegion);
        }
        crate::virt_at(base, size - 1, "the far end of a physical-memory window")?;
        Ok(Self { base, size })
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
    ///
    /// One address only. Anything with a length behind it wants
    /// [`DirectMap::reach`], which answers for the whole of it.
    #[must_use]
    pub fn virt(&self, phys: PhysAddr) -> Option<VirtAddr> {
        (phys.as_u64() < self.size).then(|| self.base + phys.as_u64())
    }

    /// Where a run of `len` bytes from `phys` begins, provided every byte of it
    /// is inside the window.
    ///
    /// # Errors
    ///
    /// [`PagingError::Unreachable`] if any byte of the range falls outside the
    /// window, including the case where the range's end is not a representable
    /// physical address.
    pub fn reach(&self, phys: PhysAddr, len: u64) -> Result<VirtAddr, PagingError> {
        let unreachable = PagingError::Unreachable {
            phys: phys.as_u64(),
            len,
        };
        // A zero-length run is inside any window that reaches its start, and is
        // the only case where the last byte is not a byte of the run.
        let end =
            end_of(phys.as_u64(), len, "the end of a physical range").map_err(|_| unreachable)?;
        if end > self.size {
            return Err(unreachable);
        }
        self.virt(phys).ok_or(unreachable)
    }

    /// Where `phys` is readable, as a pointer to `T`.
    ///
    /// Valid for a whole `T`, not merely for its first byte: a structure
    /// starting inside the window can end outside it, and a reference formed
    /// from such a pointer is invalid however it is used.
    ///
    /// # Errors
    ///
    /// [`PagingError::Unreachable`] if the window does not reach all of the
    /// value, or [`PagingError::BadPointer`] if `phys` is not aligned for `T`
    /// — an unaligned pointer into physical memory is always a mistake in the
    /// caller rather than something to paper over — or if `T` has no size, for
    /// which no address in physical memory means anything.
    pub fn ptr<T>(&self, phys: PhysAddr) -> Result<NonNull<T>, PagingError> {
        self.bytes_ptr::<T>(phys, size_of::<T>())
    }

    /// Where `phys` is readable, as a pointer aligned for `T` and valid for
    /// `bytes` bytes.
    ///
    /// What [`DirectMap::ptr`] is for a run whose length is not a type's size:
    /// an allocator's state region, a frame run, a copy destination.
    ///
    /// # Errors
    ///
    /// As [`DirectMap::ptr`], with `bytes` in place of the size of `T`. A zero
    /// `bytes` is refused for the same reason a zero-sized `T` is.
    pub fn bytes_ptr<T>(&self, phys: PhysAddr, bytes: usize) -> Result<NonNull<T>, PagingError> {
        let bad = PagingError::BadPointer {
            phys: phys.as_u64(),
            bytes,
        };
        if bytes == 0 || !phys.as_u64().is_multiple_of(align_of::<T>() as u64) {
            return Err(bad);
        }
        let virt = self.reach(phys, as_u64(bytes))?;
        NonNull::new(virt.as_mut_ptr::<T>()).ok_or(bad)
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
    /// The range must be RAM the caller is entitled to read — never a device
    /// aperture, which this window's write-back type would be a conflicting
    /// alias of — and none of it may alias anything the hypervisor holds a Rust
    /// reference to. A range something else is writing at the same time yields
    /// what a non-atomic copy of it would, which is the answer the hardware
    /// gives too.
    pub unsafe fn read(&self, phys: PhysAddr, into: &mut [u8]) -> Result<(), PagingError> {
        if into.is_empty() {
            return Ok(());
        }
        let from = self.reach(phys, as_u64(into.len()))?;
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
    /// As [`DirectMap::read`], and the range must be memory the caller owns or
    /// is entitled to overwrite.
    pub unsafe fn write(&self, phys: PhysAddr, from: &[u8]) -> Result<(), PagingError> {
        if from.is_empty() {
            return Ok(());
        }
        let into = self.reach(phys, as_u64(from.len()))?;
        // SAFETY: as in `read`, with the roles reversed: the destination is the
        // range `reach` checked, and the caller guarantees it is theirs to write
        // and aliases nothing.
        unsafe { ptr::copy_nonoverlapping(from.as_ptr(), into.as_mut_ptr::<u8>(), from.len()) };
        Ok(())
    }
}

/// The capability to walk a page-table hierarchy through a window onto physical
/// memory.
///
/// Separate from [`DirectMap`], which is a description and nothing more.
/// `x86_64`'s mapper takes a [`PageTableFrameMapping`] and forms `&PageTable`
/// and `&mut PageTable` from whatever pointer it is handed, for every frame any
/// entry of the hierarchy names, without checking it. Implementing that trait
/// is therefore a promise about a live address space and about every frame
/// reachable from its root — not about a base and a length — and it cannot be
/// made by a value anybody can construct with two numbers.
///
/// So the promise is made once, at an unsafe boundary, and the type that
/// carries it exists only to hold it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PageTables {
    window: DirectMap,
}

impl PageTables {
    /// Promises that `window` may be walked as a page-table hierarchy.
    ///
    /// # Safety
    ///
    /// `window` must be a mapping that is active on the calling processor,
    /// read-write, and must cover every frame that the hierarchy this
    /// capability will be used on can name — which for this crate means the
    /// whole of the reserved chunk, since every table frame is allocated from
    /// it. The caller must also hold exclusive access to that hierarchy for as
    /// long as the capability is used, because the mapper it is handed to forms
    /// `&mut PageTable` from it.
    pub(crate) const unsafe fn new(window: DirectMap) -> Self {
        Self { window }
    }
}

// SAFETY: `frame_to_pointer` returns the window's address of the whole frame,
// having proved that all `Size4KiB::SIZE` bytes of it lie inside the window and
// that the result is aligned as `PageTable` requires — which is automatic,
// since the window's base is 1 GiB aligned and frames are frame-aligned. The
// constructor's contract carries the rest: the window is active and read-write,
// and the caller has exclusive access to the hierarchy.
unsafe impl PageTableFrameMapping for PageTables {
    /// A page-table entry naming a frame this window does not cover is not a
    /// recoverable condition and cannot be reported: the signature has no way
    /// to say so, and the caller has already committed to the walk.
    ///
    /// Returning null would not contain it either. `x86_64` forms `&*pointer`
    /// from whatever comes back, and a reference to null is undefined behaviour
    /// before any hardware fault can occur — the fault at address zero that
    /// looks like containment happens strictly after the language has already
    /// been lied to.
    ///
    /// Every frame in this crate's hierarchies comes from the reserved chunk,
    /// which the constructor's contract requires the window to cover, so
    /// reaching this is a page table that has been corrupted underneath us.
    /// Stopping is the only honest response, and it happens with the offending
    /// frame on the record.
    fn frame_to_pointer(&self, frame: PhysFrame) -> *mut PageTable {
        let phys = frame.start_address();
        match self.window.reach(phys, Size4KiB::SIZE) {
            Ok(virt) => virt.as_mut_ptr(),
            Err(error) => panic!(
                "paging: a page table entry names frame {:#x}, which the physical-memory \
                 window does not cover ({error}); the hierarchy is corrupt",
                phys.as_u64()
            ),
        }
    }
}
