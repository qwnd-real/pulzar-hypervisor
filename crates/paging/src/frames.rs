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

use log::error;
use x86_64::{
    PhysAddr,
    structures::paging::{FrameAllocator, FrameDeallocator, PhysFrame, Size4KiB},
};

use crate::{
    DirectMap, PagingError, as_u64, as_usize,
    buddy::Buddy,
    chunk::{self, FRAME_SIZE},
    state_ptr,
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
    /// The metadata frames are simply never handed to the allocator, so there
    /// is no way to allocate them and no special case anywhere that has to
    /// remember not to.
    ///
    /// # Errors
    ///
    /// [`PagingError::Unreachable`] if `window` does not cover the chunk, or a
    /// [`PagingError::Buddy`] for a chunk geometry the allocator cannot manage.
    ///
    /// # Safety
    ///
    /// `chunk_base` must be the base of a [`chunk::CHUNK_SIZE`]-byte region
    /// that firmware has reserved and nothing else uses, reachable through
    /// `window`.
    pub unsafe fn create(chunk_base: PhysAddr, window: DirectMap) -> Result<Self, PagingError> {
        let state = state_ptr(chunk_base, window, chunk::FRAME_STATE_OFFSET)?;
        // SAFETY: `state_ptr` yields an eight-byte-aligned pointer into the
        // chunk, and the chunk layout reserves at least `state_bytes` there for
        // this allocator alone. The chunk is never freed, so the region outlives
        // the `'static` borrow.
        let mut buddy = unsafe { Buddy::create(state, chunk::CHUNK_FRAMES) }?;
        let first = as_usize(chunk::METADATA_FRAMES);
        buddy.release_range(first, as_usize(chunk::CHUNK_FRAMES) - first)?;
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
    /// [`crate::buddy::BuddyError::NotInitialized`] if the chunk holds no
    /// allocator state — which means the handoff pointed at the wrong memory.
    ///
    /// # Safety
    ///
    /// As [`Frames::create`], and no other `Frames` may be live for this chunk.
    pub unsafe fn adopt(chunk_base: PhysAddr, window: DirectMap) -> Result<Self, PagingError> {
        let state = state_ptr(chunk_base, window, chunk::FRAME_STATE_OFFSET)?;
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
    pub fn allocate(&mut self, order: usize) -> Option<PhysFrame> {
        let index = self.buddy.allocate(order)?;
        let phys = self.chunk_base + as_u64(index) * FRAME_SIZE;
        let Some(bytes) = self.window.ptr::<u64>(phys) else {
            // Unreachable while `window` covers the chunk, which both
            // constructors check. Returning the run keeps a broken window from
            // also leaking memory.
            let _ = self.buddy.release(index, order);
            return None;
        };
        // SAFETY: the run is this allocator's to write until it is released, it
        // is `FRAME_SIZE << order` bytes long by construction, and `window`
        // covers the whole chunk it lies in.
        unsafe {
            bytes
                .cast::<u8>()
                .write_bytes(0, as_usize(FRAME_SIZE) << order);
        }
        Some(PhysFrame::containing_address(phys))
    }

    /// Returns a run obtained from [`Frames::allocate`] with the same `order`.
    ///
    /// # Errors
    ///
    /// [`PagingError::NotOurs`] if the frame is not a chunk frame, or a
    /// [`PagingError::Buddy`] if it is not the start of a run of this order.
    pub fn release(&mut self, frame: PhysFrame, order: usize) -> Result<(), PagingError> {
        let phys = frame.start_address().as_u64();
        let offset = phys
            .checked_sub(self.chunk_base.as_u64())
            .filter(|offset| offset.is_multiple_of(FRAME_SIZE) && *offset < chunk::CHUNK_SIZE)
            .ok_or(PagingError::NotOurs { phys })?;
        self.buddy.release(as_usize(offset / FRAME_SIZE), order)?;
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
}

// SAFETY: every frame this yields comes from the chunk's buddy allocator and is
// not yielded again until it is released, which is exactly the exclusive
// ownership the trait requires.
unsafe impl FrameAllocator<Size4KiB> for Frames {
    fn allocate_frame(&mut self) -> Option<PhysFrame<Size4KiB>> {
        self.allocate(0)
    }
}

impl FrameDeallocator<Size4KiB> for Frames {
    /// # Safety
    ///
    /// `frame` must have come from this allocator and must no longer be mapped
    /// anywhere.
    unsafe fn deallocate_frame(&mut self, frame: PhysFrame<Size4KiB>) {
        // The trait cannot report a failure, and a rejected release means the
        // caller passed a frame this allocator never owned. Logging it loses the
        // frame but keeps the bitmap consistent, which is the better of the two.
        if let Err(error) = self.release(frame, 0) {
            error!(
                "paging: refusing to release {:#x}: {error}",
                frame.start_address()
            );
        }
    }
}
