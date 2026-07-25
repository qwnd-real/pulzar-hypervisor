//! Buddy allocator over a fixed, power-of-two number of equal-sized blocks.
//!
//! Two things in this crate need the same allocator: physical frames inside the
//! reserved chunk, and page-aligned slots inside the virtual mapping window.
//! Both want power-of-two runs, natural alignment, and coalescing on release,
//! so both use this one implementation and differ only in what a block means.
//!
//! The allocator keeps no pointers and touches none of the memory it hands out.
//! Its entire state is a header plus a bitmap, living wherever the caller puts
//! it — which is what lets `hv-loader` create it and the hypervisor image adopt
//! it afterwards, addressing the same bytes through a different mapping. A
//! design that threaded free lists through the free blocks themselves would be
//! marginally faster to allocate from and unusable for virtual address space,
//! where there is no backing memory to thread anything through.
//!
//! One bit per block per order says whether that block is free *as a whole
//! block at that order*, which is what makes coalescing a single bit test of
//! the buddy rather than a search. Each order's bits start on a word boundary
//! so no operation ever has to mask a partial word.
//!
//! Allocation walks up from the requested order to the first non-empty one and
//! splits back down, so a request is served from the smallest suitable block
//! and large blocks stay intact for large requests. Finding a free block within
//! an order scans that order's bitmap from a rover left where the last
//! allocation landed; in the steady state that is one word read, and the worst
//! case is one pass over the order's bitmap — 256 words for the 64 MiB chunk.

use core::{ptr::NonNull, slice};

use thiserror::Error;

use crate::{as_u64, as_usize};

/// Ceiling on the number of orders, and so on `state_bytes`'s header.
/// Thirty-two orders is `2^31` blocks: four petabytes at 2 MiB granularity,
/// well past anything this allocator will be pointed at.
const MAX_ORDERS: usize = 32;

/// Bits in one bitmap word.
const WORD_BITS: usize = u64::BITS as usize;

/// Fixed-size head of the allocator's state, followed immediately in memory by
/// the bitmap.
///
/// `repr(C)` because two separately compiled images read the same bytes.
#[derive(Debug)]
#[repr(C)]
struct State {
    magic: u64,
    blocks: u64,
    orders: u64,
    /// First bitmap word of each order.
    word_base: [u64; MAX_ORDERS],
    /// Free blocks at each order, so allocation can skip empty orders without
    /// touching the bitmap.
    free: [u64; MAX_ORDERS],
    /// Word offset within each order where the next scan starts.
    rover: [u64; MAX_ORDERS],
}

impl State {
    const MAGIC: u64 = u64::from_le_bytes(*b"PZBUDDY1");
}

/// A buddy allocator over `blocks` blocks, borrowing its state in place.
#[derive(Debug)]
pub struct Buddy<'a> {
    state: &'a mut State,
    bitmap: &'a mut [u64],
}

/// Bytes of state needed to manage `blocks` blocks.
///
/// Callers reserve this much and pass its address to [`Buddy::create`] or
/// [`Buddy::adopt`]. A free function rather than an associated one so the chunk
/// layout can size its regions with it in constant expressions, where
/// `Buddy`'s lifetime parameter has nothing to be inferred from.
#[must_use]
pub const fn state_bytes(blocks: u64) -> usize {
    size_of::<State>() + words_for(blocks) * size_of::<u64>()
}

/// Smallest order whose run covers `blocks`.
#[must_use]
pub const fn order_for(blocks: usize) -> usize {
    match blocks {
        0 | 1 => 0,
        _ => (blocks - 1).ilog2() as usize + 1,
    }
}

impl Buddy<'_> {
    /// Initializes state at `state` with every block *allocated*.
    ///
    /// Starting empty and then handing over what is actually available with
    /// [`Buddy::release_range`] is how the caller reserves regions: whatever is
    /// never released is permanently unavailable, with no special case in the
    /// allocator and no way to hand it out by accident.
    ///
    /// # Errors
    ///
    /// [`BuddyError::Geometry`] unless `blocks` is a power of two of at least
    /// one block and few enough to fit [`MAX_ORDERS`] orders.
    ///
    /// # Safety
    ///
    /// `state` must be eight-byte aligned and point to at least
    /// [`state_bytes`] writable bytes that no other live borrow covers,
    /// and which outlive `'a`.
    pub unsafe fn create(state: NonNull<u8>, blocks: u64) -> Result<Self, BuddyError> {
        let orders = orders_for(blocks)?;
        // SAFETY: the caller guarantees an aligned, exclusively owned region of
        // at least `state_bytes(blocks)` bytes living for `'a`, and `orders_for`
        // has confirmed the geometry those bytes were sized for.
        let buddy = unsafe { Self::borrow(state, blocks) };
        *buddy.state = State {
            magic: State::MAGIC,
            blocks,
            orders: as_u64(orders),
            word_base: word_bases(blocks),
            free: [0; MAX_ORDERS],
            rover: [0; MAX_ORDERS],
        };
        buddy.bitmap.fill(0);
        Ok(buddy)
    }

    /// Borrows state a previous [`Buddy::create`] left behind, in this or in
    /// another image.
    ///
    /// # Errors
    ///
    /// [`BuddyError::Geometry`] for an unusable `blocks`, or
    /// [`BuddyError::NotInitialized`] if the state does not describe an
    /// allocator over exactly `blocks` blocks.
    ///
    /// # Safety
    ///
    /// As [`Buddy::create`], and the region must hold state a matching
    /// `create` wrote.
    pub unsafe fn adopt(state: NonNull<u8>, blocks: u64) -> Result<Self, BuddyError> {
        let orders = orders_for(blocks)?;
        // SAFETY: identical to `create`'s requirement, which the caller carries.
        let buddy = unsafe { Self::borrow(state, blocks) };
        if buddy.state.magic != State::MAGIC
            || buddy.state.blocks != blocks
            || buddy.state.orders != as_u64(orders)
        {
            return Err(BuddyError::NotInitialized);
        }
        Ok(buddy)
    }

    /// Allocates one naturally aligned run of `1 << order` blocks, returning
    /// its index in blocks, or `None` when no block that large is free.
    pub fn allocate(&mut self, order: usize) -> Option<usize> {
        let mut source = order;
        while source < self.orders() && self.state.free[source] == 0 {
            source += 1;
        }
        if source >= self.orders() {
            return None;
        }
        let mut index = self.take_any(source)?;
        // Split back down, releasing the upper half of each level and
        // descending into the lower one.
        while source > order {
            source -= 1;
            index *= 2;
            self.mark_free(source, index + 1);
        }
        Some(index << order)
    }

    /// Returns a run obtained from [`Buddy::allocate`] with the same `order`,
    /// coalescing with its buddy as far up as it will go.
    ///
    /// # Errors
    ///
    /// [`BuddyError::Misaligned`] if `index` is not a valid start for `order`,
    /// [`BuddyError::DoubleFree`] if the run is already free. Both mean a bug
    /// in the caller; reporting them beats corrupting the bitmap, and neither
    /// aborts the hypervisor.
    pub fn release(&mut self, index: usize, order: usize) -> Result<(), BuddyError> {
        if order >= self.orders() || !index.is_multiple_of(1 << order) || index >= self.capacity() {
            return Err(BuddyError::Misaligned { index, order });
        }
        let mut order = order;
        let mut index = index >> order;
        if self.is_free(order, index) {
            return Err(BuddyError::DoubleFree { index, order });
        }
        while order + 1 < self.orders() {
            let buddy = index ^ 1;
            if !self.is_free(order, buddy) {
                break;
            }
            self.take_exact(order, buddy);
            index >>= 1;
            order += 1;
        }
        self.mark_free(order, index);
        Ok(())
    }

    /// Hands `count` blocks starting at `first` to the allocator.
    ///
    /// Releasing block by block lets the coalescing in [`Buddy::release`] build
    /// the largest blocks the range's alignment allows, so a range added this
    /// way is indistinguishable from one that was allocated and freed.
    ///
    /// # Errors
    ///
    /// As [`Buddy::release`], plus [`BuddyError::OutOfRange`] if the range does
    /// not lie inside the allocator.
    pub fn release_range(&mut self, first: usize, count: usize) -> Result<(), BuddyError> {
        if first
            .checked_add(count)
            .is_none_or(|end| end > self.capacity())
        {
            return Err(BuddyError::OutOfRange { first, count });
        }
        (first..first + count).try_for_each(|index| self.release(index, 0))
    }

    /// Total blocks under management, free or not.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        as_usize(self.state.blocks)
    }

    /// Blocks currently free, counted at order zero.
    #[must_use]
    pub fn free_blocks(&self) -> usize {
        (0..self.orders())
            .map(|order| as_usize(self.state.free[order]) << order)
            .sum()
    }

    const fn orders(&self) -> usize {
        as_usize(self.state.orders)
    }

    /// Word range backing `order`, in bitmap words.
    const fn words(&self, order: usize) -> (usize, usize) {
        let lo = as_usize(self.state.word_base[order]);
        let hi = if order + 1 == self.orders() {
            self.bitmap.len()
        } else {
            as_usize(self.state.word_base[order + 1])
        };
        (lo, hi)
    }

    /// Takes any free block at `order`, scanning from that order's rover.
    fn take_any(&mut self, order: usize) -> Option<usize> {
        let (lo, hi) = self.words(order);
        let span = hi - lo;
        let start = as_usize(self.state.rover[order]) % span;
        (0..span).find_map(|step| {
            let offset = (start + step) % span;
            let word = self.bitmap[lo + offset];
            (word != 0).then(|| {
                let bit = word.trailing_zeros() as usize;
                self.bitmap[lo + offset] = word & !(1 << bit);
                self.state.free[order] -= 1;
                self.state.rover[order] = as_u64(offset);
                offset * WORD_BITS + bit
            })
        })
    }

    fn take_exact(&mut self, order: usize, index: usize) {
        let (word, bit) = self.locate(order, index);
        self.bitmap[word] &= !(1 << bit);
        self.state.free[order] -= 1;
    }

    fn mark_free(&mut self, order: usize, index: usize) {
        let (word, bit) = self.locate(order, index);
        self.bitmap[word] |= 1 << bit;
        self.state.free[order] += 1;
        self.state.rover[order] = as_u64(index / WORD_BITS);
    }

    fn is_free(&self, order: usize, index: usize) -> bool {
        let (word, bit) = self.locate(order, index);
        self.bitmap[word] & (1 << bit) != 0
    }

    const fn locate(&self, order: usize, index: usize) -> (usize, usize) {
        (
            as_usize(self.state.word_base[order]) + index / WORD_BITS,
            index % WORD_BITS,
        )
    }

    /// Reconstitutes the two borrows the state region is split into.
    ///
    /// # Safety
    ///
    /// As [`Buddy::create`]. The header and the bitmap are disjoint halves of
    /// one region, so the two `&mut` never alias.
    unsafe fn borrow(state: NonNull<u8>, blocks: u64) -> Self {
        let header = state.cast::<State>();
        // SAFETY: the caller guarantees `state_bytes(blocks)` writable bytes at
        // `state`, of which the header occupies the first `size_of::<State>()`
        // and the bitmap exactly the remainder.
        let bitmap = unsafe { state.add(size_of::<State>()) }.cast::<u64>();
        Self {
            // SAFETY: as above; the region is aligned for `State` and lives for
            // `'a`, and no other borrow of it exists.
            state: unsafe { &mut *header.as_ptr() },
            // SAFETY: as above, and disjoint from the header.
            bitmap: unsafe { slice::from_raw_parts_mut(bitmap.as_ptr(), words_for(blocks)) },
        }
    }
}

/// Why a buddy operation was refused.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum BuddyError {
    /// The block count is not a usable power of two.
    #[error("{blocks} blocks is not a power of two within {MAX_ORDERS} orders")]
    Geometry {
        /// The rejected block count.
        blocks: u64,
    },
    /// The state region holds no allocator over this many blocks.
    #[error("no initialized allocator of this geometry at the given state")]
    NotInitialized,
    /// A release that does not match any allocation.
    #[error("block {index} is not a valid start for order {order}")]
    Misaligned {
        /// Block index offered.
        index: usize,
        /// Order offered.
        order: usize,
    },
    /// The run was already free.
    #[error("block {index} at order {order} is already free")]
    DoubleFree {
        /// Block index offered.
        index: usize,
        /// Order offered.
        order: usize,
    },
    /// The range reaches past the end of the allocator.
    #[error("range of {count} blocks at {first} leaves the allocator")]
    OutOfRange {
        /// First block of the range.
        first: usize,
        /// Length of the range.
        count: usize,
    },
}

/// Orders needed for `blocks`, rejecting geometries this allocator cannot hold.
const fn orders_for(blocks: u64) -> Result<usize, BuddyError> {
    if !blocks.is_power_of_two() {
        return Err(BuddyError::Geometry { blocks });
    }
    let orders = blocks.trailing_zeros() as usize + 1;
    if orders > MAX_ORDERS {
        return Err(BuddyError::Geometry { blocks });
    }
    Ok(orders)
}

/// Bitmap words for `blocks`, with every order starting on a word boundary.
const fn words_for(blocks: u64) -> usize {
    let mut words = 0;
    let mut count = blocks;
    loop {
        words += as_usize(count).div_ceil(WORD_BITS);
        if count == 1 {
            break;
        }
        count >>= 1;
    }
    words
}

/// First bitmap word of each order; entries past the last order are unused.
const fn word_bases(blocks: u64) -> [u64; MAX_ORDERS] {
    let mut bases = [0; MAX_ORDERS];
    let mut count = blocks;
    let mut order = 0;
    let mut next = 0;
    loop {
        bases[order] = next;
        next += as_u64(as_usize(count).div_ceil(WORD_BITS));
        if count == 1 {
            break;
        }
        count >>= 1;
        order += 1;
    }
    bases
}
