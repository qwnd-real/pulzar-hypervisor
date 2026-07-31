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
//!
//! # What the base has to satisfy
//!
//! The allocator's bookkeeping is base-relative: it counts slots from zero and
//! the base turns a slot number into an address. That is what lets the state
//! survive the handoff, and it is also what makes the base the one value the
//! two images must agree on exactly. A base a page off would leave every live
//! allocation naming a different page than the one that was reserved, with
//! nothing in the state to notice.
//!
//! So it is validated on the way in and then recorded. Validated:
//! frame-aligned, and with the whole window `base..base + MAPPING_WINDOW_SIZE`
//! a single run of canonical addresses — a base near the top of the address
//! space whose window wraps, or one in the lower half whose window crosses the
//! canonical hole, would otherwise be accepted here and panic at the first
//! allocation that formed an address near the end.
//!
//! Recorded, because validity is not identity. [`Slots::create`] writes the
//! base it was given into the chunk beside the allocator state, and
//! [`Slots::adopt`] refuses any base but that one. The value reaches the second
//! image through the handoff, which is a structure the first image wrote and
//! nothing has checked against the memory it describes; the copy in the chunk
//! is written by the same call that built the state it belongs to, so the two
//! cannot disagree without one of them having been corrupted. Every other
//! high-half address in the handoff can be checked against something the
//! machine knows — the direct map against a walk of the live tables, the root
//! against `CR3`. The window base maps nothing yet, so this is what it is
//! checked against instead.

use x86_64::{
    PhysAddr, VirtAddr,
    structures::paging::{Page, Size4KiB},
};

use crate::{
    DirectMap, PagingError, as_u64, as_usize,
    buddy::{self, Buddy, BuddyError},
    chunk::{self, FRAME_SIZE},
    state_ptr, virt_at,
};

/// Identifies a recorded window base as one a matching `create` wrote, so that
/// uninitialized memory or another image's bytes are not read as an address.
const RECORDED: u64 = u64::from_le_bytes(*b"PZWINDW1");

/// What [`Slots::create`] leaves in the chunk about the window it opened.
///
/// `repr(C)` because two separately compiled images read the same bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
struct Recorded {
    magic: u64,
    base: u64,
}

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
    /// [`PagingError::Misaligned`] if `base` is not frame-aligned,
    /// [`PagingError::Arithmetic`] if the window does not fit above it,
    /// [`PagingError::Unreachable`] if `window` does not cover the whole of the
    /// allocator state in the chunk, or a [`PagingError::Buddy`] for a window
    /// geometry the allocator cannot manage.
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
        let base = validate(base)?;
        let state = state(chunk_base, window)?;
        // SAFETY: `state` is an eight-byte-aligned pointer proved to be valid
        // for the whole of `state_bytes(MAPPING_WINDOW_SLOTS)`, inside a region
        // the chunk layout reserves for this allocator alone; the chunk outlives
        // the `'static` borrow.
        let mut buddy = unsafe { Buddy::create(state, chunk::MAPPING_WINDOW_SLOTS) }?;
        buddy.hand_over(0, as_usize(chunk::MAPPING_WINDOW_SLOTS))?;
        let recorded = recorded(chunk_base, window)?;
        // SAFETY: the record sits in the same reserved region as the allocator
        // state and after it, which the layout's assertions prove there is room
        // for; `recorded` proved the window reaches all of it and that it is
        // aligned for the type. Nothing else writes these bytes.
        unsafe {
            recorded.write(Recorded {
                magic: RECORDED,
                base: base.as_u64(),
            });
        }
        Ok(Self { buddy, base })
    }

    /// Picks up the state a previous [`Slots::create`] left in the chunk.
    ///
    /// `base` must be the address that `create` was given, which is checked
    /// against the copy `create` left beside the state rather than taken on
    /// trust: the bookkeeping is base-relative, so a base that is merely valid
    /// and not identical would leave every live allocation naming a page other
    /// than the one that was reserved.
    ///
    /// # Errors
    ///
    /// As [`Slots::create`], plus [`PagingError::LayoutMismatch`] if the chunk
    /// records a different base, or no base at all.
    ///
    /// # Safety
    ///
    /// As [`Slots::create`], and no other `Slots` may be live for this window.
    pub unsafe fn adopt(
        chunk_base: PhysAddr,
        window: DirectMap,
        base: VirtAddr,
    ) -> Result<Self, PagingError> {
        let base = validate(base)?;
        let state = state(chunk_base, window)?;
        // SAFETY: as in `create`; the caller additionally guarantees the chunk is
        // the one a matching `create` initialized.
        let buddy = unsafe { Buddy::adopt(state, chunk::MAPPING_WINDOW_SLOTS) }?;
        let recorded = recorded(chunk_base, window)?;
        // SAFETY: as in `create`, and the bytes hold what a matching `create`
        // wrote — which is what the magic below establishes rather than assumes.
        let found = unsafe { recorded.read() };
        if found.magic != RECORDED || found.base != base.as_u64() {
            return Err(PagingError::LayoutMismatch {
                field: "the mapping window's base",
                expected: found.base,
                found: base.as_u64(),
            });
        }
        Ok(Self { buddy, base })
    }

    /// Reserves `1 << order` contiguous, naturally aligned pages of the window.
    ///
    /// # Errors
    ///
    /// [`PagingError::OutOfWindow`] if the window has no run that large, or
    /// [`PagingError::Buddy`] carrying [`BuddyError::InvalidOrder`] for an
    /// order the window is too small to have at all.
    pub fn allocate(&mut self, order: usize) -> Result<Page<Size4KiB>, PagingError> {
        let index = self.buddy.allocate(order).map_err(|error| match error {
            BuddyError::Exhausted { order } => PagingError::OutOfWindow { order },
            other => PagingError::Buddy(other),
        })?;
        // The offset is below `MAPPING_WINDOW_SIZE` because the index is below
        // the slot count, and `validate` proved the whole window lies above the
        // base, so this address exists.
        let virt = virt_at(
            self.base,
            as_u64(index) * FRAME_SIZE,
            "the address of a window run",
        )?;
        Ok(Page::containing_address(virt))
    }

    /// Returns a run obtained from [`Slots::allocate`] with the same `order`.
    ///
    /// # Errors
    ///
    /// [`PagingError::OutsideWindow`] if the page is outside the window, or a
    /// [`PagingError::Buddy`] if it is not the start of a live run of this
    /// order.
    pub fn release(&mut self, page: Page<Size4KiB>, order: usize) -> Result<(), PagingError> {
        let virt = page.start_address().as_u64();
        let offset = virt
            .checked_sub(self.base.as_u64())
            .filter(|offset| {
                offset.is_multiple_of(FRAME_SIZE) && *offset < chunk::MAPPING_WINDOW_SIZE
            })
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

/// Proves `base` can carry the whole window.
fn validate(base: VirtAddr) -> Result<VirtAddr, PagingError> {
    if !base.as_u64().is_multiple_of(FRAME_SIZE) {
        return Err(PagingError::Misaligned {
            value: base.as_u64(),
            align: FRAME_SIZE,
        });
    }
    virt_at(
        base,
        chunk::MAPPING_WINDOW_SIZE - 1,
        "the far end of the mapping window",
    )?;
    Ok(base)
}

/// The window allocator's state region in the chunk, proved reachable in full.
fn state(chunk_base: PhysAddr, window: DirectMap) -> Result<core::ptr::NonNull<u8>, PagingError> {
    state_ptr(
        chunk_base,
        window,
        chunk::WINDOW_STATE_OFFSET,
        buddy::state_bytes(chunk::MAPPING_WINDOW_SLOTS),
    )
}

/// Where the window's base is recorded: immediately after the allocator state,
/// inside the same reserved region.
///
/// Placed by the state's own length rather than at an offset of its own, so
/// that a change to the allocator's state ABI moves the record with it instead
/// of leaving it overlapping the bitmap. The region has room for both, which
/// [`RECORD_FITS`] proves at compile time.
fn recorded(
    chunk_base: PhysAddr,
    window: DirectMap,
) -> Result<core::ptr::NonNull<Recorded>, PagingError> {
    let phys = crate::phys_at(
        chunk_base,
        chunk::WINDOW_STATE_OFFSET + RECORD_OFFSET,
        "the address of the window's recorded base",
    )?;
    window.ptr::<Recorded>(phys)
}

/// Where the record sits inside the window's state region, rounded up so that
/// it is aligned for its own type whatever the state's length happens to be.
const RECORD_OFFSET: u64 = crate::as_u64(buddy::state_bytes(chunk::MAPPING_WINDOW_SLOTS))
    .next_multiple_of(align_of::<Recorded>() as u64);

const _: () = assert!(
    RECORD_OFFSET + size_of::<Recorded>() as u64 <= chunk::WINDOW_STATE_SIZE,
    "WINDOW_STATE_SIZE must hold the allocator's state and the window base it \
     was created with, which adoption compares against the handoff"
);
