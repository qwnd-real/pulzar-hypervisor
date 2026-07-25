//! Allocator for page-sized slots in the mapping window.
//!
//! Explicit physical mappings need somewhere to put them. Handing out virtual
//! addresses has exactly the same shape as handing out physical frames —
//! power-of-two runs, natural alignment, coalescing on release — so it runs on
//! the same buddy allocator over a second block space, where a block is one
//! page of the mapping window instead of one frame of the chunk. Nothing here
//! touches memory: a slot is an address and nothing more until something is
//! mapped into it.
//!
//! Because the window's base is randomized and its allocator state lives in the
//! chunk, mappings and stacks land at addresses that move between boots and
//! stay valid across the handoff.

use x86_64::{
    PhysAddr, VirtAddr,
    structures::paging::{Page, Size4KiB},
};

use crate::{
    DirectMap, PagingError, as_u64, as_usize,
    buddy::Buddy,
    chunk::{self, FRAME_SIZE},
    state_ptr,
};

/// Owner of the mapping window's addresses.
#[derive(Debug)]
pub struct Slots {
    buddy: Buddy<'static>,
    base: VirtAddr,
}

impl Slots {
    /// Takes ownership of the whole window at `base`.
    ///
    /// # Errors
    ///
    /// [`PagingError::Unreachable`] if `window` does not cover the chunk
    /// holding the allocator state, or a [`PagingError::Buddy`] for a
    /// window geometry the allocator cannot manage.
    ///
    /// # Safety
    ///
    /// As [`crate::Frames::create`]: `chunk_base` must name the reserved chunk,
    /// reachable through `window`.
    pub unsafe fn create(
        chunk_base: PhysAddr,
        window: DirectMap,
        base: VirtAddr,
    ) -> Result<Self, PagingError> {
        let state = state_ptr(chunk_base, window, chunk::WINDOW_STATE_OFFSET)?;
        // SAFETY: `state_ptr` yields an eight-byte-aligned pointer into the
        // chunk, and the chunk layout reserves at least `state_bytes` there for
        // this allocator alone; the chunk outlives the `'static` borrow.
        let mut buddy = unsafe { Buddy::create(state, chunk::MAPPING_WINDOW_SLOTS) }?;
        buddy.release_range(0, as_usize(chunk::MAPPING_WINDOW_SLOTS))?;
        Ok(Self { buddy, base })
    }

    /// Picks up the state a previous [`Slots::create`] left in the chunk.
    ///
    /// `base` must be the same address that `create` was given; it is carried
    /// in the handoff rather than stored in the chunk because the
    /// allocator's own bookkeeping is base-relative and so stays valid
    /// either way.
    ///
    /// # Errors
    ///
    /// As [`Slots::create`].
    ///
    /// # Safety
    ///
    /// As [`Slots::create`], and no other `Slots` may be live for this window.
    pub unsafe fn adopt(
        chunk_base: PhysAddr,
        window: DirectMap,
        base: VirtAddr,
    ) -> Result<Self, PagingError> {
        let state = state_ptr(chunk_base, window, chunk::WINDOW_STATE_OFFSET)?;
        // SAFETY: as in `create`; the caller additionally guarantees the chunk is
        // the one a matching `create` initialized.
        let buddy = unsafe { Buddy::adopt(state, chunk::MAPPING_WINDOW_SLOTS) }?;
        Ok(Self { buddy, base })
    }

    /// Reserves `1 << order` contiguous, naturally aligned pages of the window.
    pub fn allocate(&mut self, order: usize) -> Option<Page<Size4KiB>> {
        let index = self.buddy.allocate(order)?;
        Some(Page::containing_address(
            self.base + as_u64(index) * FRAME_SIZE,
        ))
    }

    /// Returns a run obtained from [`Slots::allocate`] with the same `order`.
    ///
    /// # Errors
    ///
    /// [`PagingError::OutsideWindow`] if the page is outside the window, or a
    /// [`PagingError::Buddy`] if it is not the start of a run of this order.
    pub fn release(&mut self, page: Page<Size4KiB>, order: usize) -> Result<(), PagingError> {
        let virt = page.start_address().as_u64();
        let offset = virt
            .checked_sub(self.base.as_u64())
            .filter(|offset| *offset < chunk::MAPPING_WINDOW_SIZE)
            .ok_or(PagingError::OutsideWindow { virt })?;
        self.buddy.release(as_usize(offset / FRAME_SIZE), order)?;
        Ok(())
    }

    /// Virtual base of the window.
    #[must_use]
    pub const fn base(&self) -> VirtAddr {
        self.base
    }

    /// Slots still available.
    #[must_use]
    pub fn free(&self) -> usize {
        self.buddy.free_blocks()
    }
}
