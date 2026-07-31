//! Physical frame allocator over the reserved chunk.
//!
//! A thin translation between the buddy allocator's block indices and physical
//! frame addresses, plus the two guarantees callers rely on: a frame handed out
//! is zeroed, and a frame handed out is not reachable from any other
//! allocation. Zeroing here rather than at each call site is what makes a
//! freshly allocated page table usable without a separate clearing step, and
//! what keeps whatever the previous owner left in a frame from leaking into its
//! next one.
//!
//! The allocator's state lives at a fixed offset inside the chunk, so
//! `hv-loader` creates it and the hypervisor image picks it up unchanged by
//! addressing the same bytes through the direct map.
//!
//! # The metadata frames
//!
//! The front of the chunk holds the handoff, the captured firmware state, both
//! allocators' own state, the memory map and the portal. Those frames are never
//! handed to the buddy allocator, so there is no way to allocate them.
//!
//! They are also refused by name on the release path, which is not redundant
//! with that. The buddy allocator authenticates a release against its record of
//! live allocations, and the metadata frames have no such record — so a release
//! of one is already refused. The explicit check is what makes that refusal
//! independent of the allocator's design rather than a consequence of it: a
//! frame holding the allocator's own bitmap must not become allocatable, and
//! zeroed, whatever else changes.

use core::ptr::NonNull;

use log::error;
use x86_64::{
    PhysAddr,
    structures::paging::{FrameAllocator, FrameDeallocator, PhysFrame, Size4KiB},
};

use crate::{
    DirectMap, PagingError, as_u64, as_usize,
    buddy::{self, Buddy, BuddyError},
    chunk::{self, FRAME_SIZE},
    phys_at, state_ptr,
};

/// Owner of the chunk's frames.
#[derive(Debug)]
pub struct Frames {
    buddy: Buddy<'static>,
    chunk_base: PhysAddr,
    window: DirectMap,
}

impl Frames {
    /// Takes ownership of the chunk, with the metadata at its front withheld.
    ///
    /// # Errors
    ///
    /// [`PagingError::Misaligned`] unless `chunk_base` is
    /// [`chunk::CHUNK_ALIGN`] aligned, [`PagingError::Unreachable`] if `window`
    /// does not cover the whole chunk, or a [`PagingError::Buddy`] for a chunk
    /// geometry the allocator cannot manage.
    ///
    /// # Safety
    ///
    /// `chunk_base` must be the base of a [`chunk::CHUNK_SIZE`]-byte region
    /// that firmware has reserved and nothing else uses, reachable through
    /// `window`.
    pub unsafe fn create(chunk_base: PhysAddr, window: DirectMap) -> Result<Self, PagingError> {
        let state = Self::state(chunk_base, window)?;
        // SAFETY: `state` is an eight-byte-aligned pointer proved to be valid
        // for the whole of `state_bytes(CHUNK_FRAMES)`, inside a region the
        // chunk layout reserves for this allocator alone. The chunk is never
        // freed, so the region outlives the `'static` borrow.
        let mut buddy = unsafe { Buddy::create(state, chunk::CHUNK_FRAMES) }?;
        let first = as_usize(chunk::METADATA_FRAMES);
        buddy.hand_over(first, as_usize(chunk::CHUNK_FRAMES) - first)?;
        Ok(Self {
            buddy,
            chunk_base,
            window,
        })
    }

    /// Picks up the state a previous [`Frames::create`] left in the chunk.
    ///
    /// # Errors
    ///
    /// As [`Frames::create`], plus [`PagingError::Buddy`] carrying
    /// [`BuddyError::NotInitialized`] if the chunk holds no allocator state of
    /// this build's geometry — which means the handoff pointed at the wrong
    /// memory, or at a chunk a differently built loader wrote.
    ///
    /// # Safety
    ///
    /// As [`Frames::create`], and no other `Frames` may be live for this chunk.
    pub unsafe fn adopt(chunk_base: PhysAddr, window: DirectMap) -> Result<Self, PagingError> {
        let state = Self::state(chunk_base, window)?;
        // SAFETY: as in `create`; the caller additionally guarantees that this
        // chunk is the one a matching `create` initialized.
        let buddy = unsafe { Buddy::adopt(state, chunk::CHUNK_FRAMES) }?;
        Ok(Self {
            buddy,
            chunk_base,
            window,
        })
    }

    /// Allocates `1 << order` contiguous, naturally aligned, zeroed frames.
    ///
    /// # Errors
    ///
    /// [`PagingError::OutOfFrames`] if the chunk has no run that large,
    /// [`PagingError::Buddy`] carrying [`BuddyError::InvalidOrder`] for an
    /// order the chunk is too small to have at all, or
    /// [`PagingError::Unreachable`] if the window does not reach the whole run
    /// — which cannot happen while the window covers the chunk, and is reported
    /// rather than assumed away because the run would otherwise be zeroed
    /// through a pointer that was never checked.
    pub fn allocate(&mut self, order: usize) -> Result<PhysFrame, PagingError> {
        let index = self.buddy.allocate(order).map_err(|error| match error {
            BuddyError::Exhausted { order } => PagingError::OutOfFrames { order },
            other => PagingError::Buddy(other),
        })?;
        // At most `FRAME_SIZE << 31` by the allocator's own order limit, which
        // is a thousandth of what a `usize` holds on this target.
        let bytes = as_usize(FRAME_SIZE) << order;
        match self.wipe(index, bytes) {
            Ok(phys) => Ok(PhysFrame::containing_address(phys)),
            Err(error) => {
                // Give the run straight back: a window that cannot reach the
                // chunk is a broken invariant, and leaking memory on top of it
                // helps nobody.
                if let Err(inner) = self.buddy.release(index, order) {
                    error!("paging: could not return an unreachable run: {inner}");
                }
                Err(error)
            }
        }
    }

    /// Returns a run obtained from [`Frames::allocate`] with the same `order`.
    ///
    /// # Errors
    ///
    /// [`PagingError::NotOurs`] if the frame is not a chunk frame or is one of
    /// the metadata frames the allocator never owned, or a
    /// [`PagingError::Buddy`] if it is not the start of a live run of this
    /// order.
    pub fn release(&mut self, frame: PhysFrame, order: usize) -> Result<(), PagingError> {
        let phys = frame.start_address().as_u64();
        let not_ours = PagingError::NotOurs { phys };
        let offset = phys
            .checked_sub(self.chunk_base.as_u64())
            .filter(|offset| offset.is_multiple_of(FRAME_SIZE) && *offset < chunk::CHUNK_SIZE)
            .ok_or(not_ours)?;
        let index = offset / FRAME_SIZE;
        // Independent of the allocator's own bookkeeping, and deliberately so:
        // the frames below this line hold the allocator's bitmaps, the handoff
        // and the portal, and a release that reached them would make them
        // allocatable and then zero them.
        if index < chunk::METADATA_FRAMES {
            return Err(not_ours);
        }
        self.buddy.release(as_usize(index), order)?;
        Ok(())
    }

    /// Physical base of the chunk.
    #[must_use]
    pub const fn chunk_base(&self) -> PhysAddr {
        self.chunk_base
    }

    /// Frames still available.
    #[must_use]
    pub fn free(&self) -> usize {
        self.buddy.free_blocks()
    }

    /// Frames under management, metadata excluded.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.buddy.capacity() - as_usize(chunk::METADATA_FRAMES)
    }

    /// The chunk's state region, proved reachable in full.
    fn state(chunk_base: PhysAddr, window: DirectMap) -> Result<NonNull<u8>, PagingError> {
        if !chunk_base.as_u64().is_multiple_of(chunk::CHUNK_ALIGN) {
            return Err(PagingError::Misaligned {
                value: chunk_base.as_u64(),
                align: chunk::CHUNK_ALIGN,
            });
        }
        // Everything below depends on the whole chunk being addressable, not
        // just the part the state happens to live in: `run` hands out pointers
        // into any of it, and the page-table walker follows entries into any of
        // it.
        window.reach(chunk_base, chunk::CHUNK_SIZE)?;
        state_ptr(
            chunk_base,
            window,
            chunk::FRAME_STATE_OFFSET,
            buddy::state_bytes(chunk::CHUNK_FRAMES),
        )
    }

    /// Zeroes `bytes` bytes from block `index` and answers where they start.
    ///
    /// The whole run is proved to be inside the window before a byte of it is
    /// written. Checking only the run's first address would leave a run that
    /// begins inside the window and ends past it being cleared through a
    /// pointer nothing vouched for.
    fn wipe(&mut self, index: usize, bytes: usize) -> Result<PhysAddr, PagingError> {
        let phys = phys_at(
            self.chunk_base,
            as_u64(index) * FRAME_SIZE,
            "the address of a frame run",
        )?;
        let run = self.window.bytes_ptr::<u8>(phys, bytes)?;
        // SAFETY: the run is this allocator's to write until it is released, and
        // `bytes_ptr` proved the window covers all `bytes` of it at a pointer
        // valid for that many bytes and trivially aligned for `u8`.
        unsafe { run.write_bytes(0, bytes) };
        Ok(phys)
    }
}

// SAFETY: every frame this yields comes from the chunk's buddy allocator and is
// not yielded again until it is released, which is exactly the exclusive
// ownership the trait requires. The allocator authenticates releases against
// its record of live allocations, so a frame cannot re-enter circulation except
// through the release of the allocation that produced it.
unsafe impl FrameAllocator<Size4KiB> for Frames {
    fn allocate_frame(&mut self) -> Option<PhysFrame<Size4KiB>> {
        self.allocate(0).ok()
    }
}

impl FrameDeallocator<Size4KiB> for Frames {
    /// # Safety
    ///
    /// `frame` must have come from this allocator and must no longer be mapped
    /// anywhere.
    unsafe fn deallocate_frame(&mut self, frame: PhysFrame<Size4KiB>) {
        // The trait cannot report a failure, and a rejected release means the
        // caller passed a frame this allocator never handed out. Logging it
        // loses the frame but keeps the bitmaps consistent, which is the better
        // of the two.
        if let Err(error) = self.release(frame, 0) {
            error!(
                "paging: refusing to release {:#x}: {error}",
                frame.start_address()
            );
        }
    }
}
