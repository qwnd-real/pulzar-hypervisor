//! Buddy allocator over a fixed, power-of-two number of equal-sized blocks.
//!
//! Two things in this crate need the same allocator: physical frames inside the
//! reserved chunk, and page-aligned slots inside the virtual mapping window.
//! Both want power-of-two runs, natural alignment, and coalescing on release,
//! so both use this one implementation and differ only in what a block means.
//!
//! The allocator keeps no pointers and touches none of the memory it hands out.
//! Its entire state is a header plus two bitmaps, living wherever the caller
//! puts it — which is what lets `hv-loader` create it and the hypervisor image
//! adopt it afterwards, addressing the same bytes through a different mapping.
//! A design that threaded free lists through the free blocks themselves would
//! be marginally faster to allocate from and unusable for virtual address
//! space, where there is no backing memory to thread anything through.
//!
//! # Why there are two bitmaps and not one
//!
//! The free bitmap says, one bit per block per order, whether that block is
//! free *as a whole block at that order*, which is what makes coalescing a
//! single bit test of the buddy rather than a search.
//!
//! That bitmap alone cannot authenticate a release. A clear bit means "not free
//! at this order", which is equally true of the interior of a live larger run,
//! of a live run of exactly this size, and of a block that was freed and has
//! since coalesced upwards. A release checked against it therefore accepts
//! interior addresses, wrong orders, and repeats — each of which hands the same
//! memory to two owners.
//!
//! So a second bitmap of the same shape records where live allocations *start*
//! and at which order. [`Buddy::allocate`] sets exactly one bit in it and
//! [`Buddy::release`] requires exactly that bit, which makes a release either
//! the inverse of a particular allocation or an error. It costs one extra bit
//! per block per order — under 0.4 % of what it manages — and it is the whole
//! of what stops a duplicate release from becoming duplicate ownership.
//!
//! Each order's bits start on a word boundary so no operation ever has to mask
//! a partial word.
//!
//! # Where the memory comes from
//!
//! A freshly created allocator owns nothing: every block is unavailable and no
//! block is live, so nothing can be allocated and nothing can be released.
//! [`Buddy::hand_over`] is what gives it memory, and it is deliberately a
//! separate operation from [`Buddy::release`] rather than a loop over it —
//! handing the allocator a region it never allocated is not the inverse of an
//! allocation and must not be authenticated as one. Whatever is never handed
//! over is permanently unreachable, with no special case in the allocator and
//! no way to hand it out by accident.
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
/// the free bitmap and then the live bitmap.
///
/// `repr(C)` because two separately compiled images read the same bytes.
#[derive(Debug)]
#[repr(C)]
struct State {
    magic: u64,
    blocks: u64,
    orders: u64,
    /// First bitmap word of each order, in either bitmap.
    word_base: [u64; MAX_ORDERS],
    /// Free blocks at each order, so allocation can skip empty orders without
    /// touching the bitmap.
    free: [u64; MAX_ORDERS],
    /// Word offset within each order where the next scan starts.
    rover: [u64; MAX_ORDERS],
}

impl State {
    /// Identifies state this exact layout wrote. The trailing digit is the
    /// state ABI: it changes whenever the header or the bitmap geometry does,
    /// so an image built against one layout refuses state written by another
    /// instead of misreading it.
    const MAGIC: u64 = u64::from_le_bytes(*b"PZBUDDY2");
}

/// A buddy allocator over `blocks` blocks, borrowing its state in place.
#[derive(Debug)]
pub struct Buddy<'a> {
    state: &'a mut State,
    /// One bit per block per order: is this block free as a whole block here.
    free: &'a mut [u64],
    /// One bit per block per order: does a live allocation of exactly this
    /// order start at this block.
    live: &'a mut [u64],
}

/// Bytes of state needed to manage `blocks` blocks.
///
/// Callers reserve this much and pass its address to [`Buddy::create`] or
/// [`Buddy::adopt`]. A free function rather than an associated one so the chunk
/// layout can size its regions with it in constant expressions, where
/// `Buddy`'s lifetime parameter has nothing to be inferred from.
///
/// Defined for every input, `0` included, so that a layout computation can ask
/// about a geometry the allocator would refuse without the question itself
/// hanging. What it answers for such a geometry is the size of a state region
/// no constructor will ever accept.
#[must_use]
pub const fn state_bytes(blocks: u64) -> usize {
    size_of::<State>() + 2 * words_for(blocks) * size_of::<u64>()
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
    /// Initializes state at `state` owning nothing at all.
    ///
    /// Every block starts unavailable and no block starts live, so the
    /// allocator can neither hand anything out nor accept anything back until
    /// [`Buddy::hand_over`] gives it a region. That is how a caller reserves
    /// part of the block space: whatever is never handed over stays
    /// unreachable.
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
        buddy.free.fill(0);
        buddy.live.fill(0);
        Ok(buddy)
    }

    /// Borrows state a previous [`Buddy::create`] left behind, in this or in
    /// another image.
    ///
    /// # Errors
    ///
    /// [`BuddyError::Geometry`] for an unusable `blocks`, or
    /// [`BuddyError::NotInitialized`] if the state does not describe an
    /// allocator of this state ABI over exactly `blocks` blocks with the
    /// bitmap geometry this build computes.
    ///
    /// # Safety
    ///
    /// As [`Buddy::create`], and the region must hold state a matching
    /// `create` wrote.
    pub unsafe fn adopt(state: NonNull<u8>, blocks: u64) -> Result<Self, BuddyError> {
        let orders = orders_for(blocks)?;
        // SAFETY: identical to `create`'s requirement, which the caller carries.
        let buddy = unsafe { Self::borrow(state, blocks) };
        let expected = word_bases(blocks);
        if buddy.state.magic != State::MAGIC
            || buddy.state.blocks != blocks
            || buddy.state.orders != as_u64(orders)
            || buddy.state.word_base != expected
        {
            return Err(BuddyError::NotInitialized);
        }
        // A counter that claims more free blocks at an order than the order has
        // is state this build did not write, and every scan below trusts these.
        if (0..orders).any(|order| buddy.state.free[order] > blocks >> order) {
            return Err(BuddyError::NotInitialized);
        }
        Ok(buddy)
    }

    /// Allocates one naturally aligned run of `1 << order` blocks, returning
    /// its index in blocks.
    ///
    /// # Errors
    ///
    /// [`BuddyError::InvalidOrder`] if the allocator has no such order, which
    /// is a bug in the caller rather than a shortage, or
    /// [`BuddyError::Exhausted`] if no block that large is free. The two are
    /// separate because only the second is worth retrying differently.
    pub fn allocate(&mut self, order: usize) -> Result<usize, BuddyError> {
        if order >= self.orders() {
            return Err(BuddyError::InvalidOrder { order });
        }
        let mut source = order;
        while source < self.orders() && self.state.free[source] == 0 {
            source += 1;
        }
        if source >= self.orders() {
            return Err(BuddyError::Exhausted { order });
        }
        let mut index = self
            .take_any(source)
            .ok_or(BuddyError::Exhausted { order })?;
        // Split back down, releasing the upper half of each level and
        // descending into the lower one.
        while source > order {
            source -= 1;
            index *= 2;
            self.mark_free(source, index + 1);
        }
        self.mark_live(order, index);
        Ok(index << order)
    }

    /// Returns a run obtained from [`Buddy::allocate`] with the same `order`,
    /// coalescing with its buddy as far up as it will go.
    ///
    /// Only the exact inverse of a live allocation is accepted. An interior
    /// block of a larger run, a different order for the same start, a block the
    /// allocator never handed out, and a repeat of a release that has already
    /// happened are all refused with the allocator's state left untouched —
    /// each of them would otherwise end with the same memory owned twice.
    ///
    /// # Errors
    ///
    /// [`BuddyError::InvalidOrder`] for an order this allocator does not have,
    /// [`BuddyError::Misaligned`] if `index` is not a valid start for `order`
    /// or lies outside the allocator, or [`BuddyError::NotAllocated`] if no
    /// live allocation of exactly this order starts there. All three mean a bug
    /// in the caller; reporting them beats corrupting the bitmap, and none
    /// aborts the hypervisor.
    pub fn release(&mut self, index: usize, order: usize) -> Result<(), BuddyError> {
        if order >= self.orders() {
            return Err(BuddyError::InvalidOrder { order });
        }
        if !index.is_multiple_of(1 << order) || index >= self.capacity() {
            return Err(BuddyError::Misaligned { index, order });
        }
        let block = index >> order;
        if !self.is_live(order, block) {
            return Err(BuddyError::NotAllocated { index, order });
        }
        self.take_live(order, block);
        self.give_back(block, order);
        Ok(())
    }

    /// Hands `count` blocks starting at `first` to the allocator, as memory it
    /// did not allocate and so cannot authenticate a release of.
    ///
    /// This is how an allocator created owning nothing is given the region it
    /// manages. Adding block by block lets the coalescing in [`Buddy::release`]
    /// build the largest blocks the range's alignment allows, so a range added
    /// this way is indistinguishable afterwards from one that was allocated and
    /// freed.
    ///
    /// All or nothing: the whole range is checked against the current state
    /// before any of it is added, so a range that overlaps memory the allocator
    /// already has leaves the allocator exactly as it was rather than partly
    /// extended.
    ///
    /// # Errors
    ///
    /// [`BuddyError::OutOfRange`] if the range does not lie inside the
    /// allocator, or [`BuddyError::DoubleFree`] naming the first block that is
    /// already available or already inside a live allocation.
    pub fn hand_over(&mut self, first: usize, count: usize) -> Result<(), BuddyError> {
        if first
            .checked_add(count)
            .is_none_or(|end| end > self.capacity())
        {
            return Err(BuddyError::OutOfRange { first, count });
        }
        if let Some(index) = (first..first + count).find(|index| self.is_accounted(*index)) {
            return Err(BuddyError::DoubleFree { index, order: 0 });
        }
        // Nothing below can fail: every block was just proved to be neither
        // available nor part of a live allocation, which is the only way
        // `give_back` could reach an inconsistent state.
        for index in first..first + count {
            self.give_back(index, 0);
        }
        Ok(())
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

    /// Marks `block` free at `order` and coalesces upwards as far as it goes.
    ///
    /// The half of a release that touches no liveness, shared by the
    /// authenticated release and by handing the allocator memory it never
    /// allocated.
    fn give_back(&mut self, block: usize, order: usize) {
        let mut order = order;
        let mut block = block;
        while order + 1 < self.orders() {
            let buddy = block ^ 1;
            if !self.is_free(order, buddy) {
                break;
            }
            self.take_exact(order, buddy);
            block >>= 1;
            order += 1;
        }
        self.mark_free(order, block);
    }

    /// Whether the allocator already accounts for order-zero block `index`,
    /// either as memory it can hand out or as memory it has handed out.
    ///
    /// Both are answered by walking the ancestors: a block is available if any
    /// ancestor is free, and it is owned if any ancestor is the start of a live
    /// allocation covering it.
    fn is_accounted(&self, index: usize) -> bool {
        (0..self.orders()).any(|order| {
            let ancestor = index >> order;
            self.is_free(order, ancestor) || self.is_live(order, ancestor)
        })
    }

    /// Word range backing `order`, in bitmap words. The same range indexes
    /// either bitmap: the two have identical shape.
    const fn words(&self, order: usize) -> (usize, usize) {
        let lo = as_usize(self.state.word_base[order]);
        let hi = if order + 1 == self.orders() {
            self.free.len()
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
            let word = self.free[lo + offset];
            (word != 0).then(|| {
                let bit = word.trailing_zeros() as usize;
                let left = word & !(1 << bit);
                self.free[lo + offset] = left;
                self.state.free[order] -= 1;
                // Past an emptied word rather than back to it: the next scan
                // would otherwise read a word it has just proved to be zero.
                self.state.rover[order] = as_u64(if left == 0 {
                    (offset + 1) % span
                } else {
                    offset
                });
                offset * WORD_BITS + bit
            })
        })
    }

    fn take_exact(&mut self, order: usize, index: usize) {
        let (word, bit) = self.locate(order, index);
        self.free[word] &= !(1 << bit);
        self.state.free[order] -= 1;
    }

    fn mark_free(&mut self, order: usize, index: usize) {
        let (word, bit) = self.locate(order, index);
        self.free[word] |= 1 << bit;
        self.state.free[order] += 1;
        self.state.rover[order] = as_u64(index / WORD_BITS);
    }

    fn is_free(&self, order: usize, index: usize) -> bool {
        let (word, bit) = self.locate(order, index);
        self.free[word] & (1 << bit) != 0
    }

    fn mark_live(&mut self, order: usize, index: usize) {
        let (word, bit) = self.locate(order, index);
        self.live[word] |= 1 << bit;
    }

    fn take_live(&mut self, order: usize, index: usize) {
        let (word, bit) = self.locate(order, index);
        self.live[word] &= !(1 << bit);
    }

    fn is_live(&self, order: usize, index: usize) -> bool {
        let (word, bit) = self.locate(order, index);
        self.live[word] & (1 << bit) != 0
    }

    const fn locate(&self, order: usize, index: usize) -> (usize, usize) {
        (
            as_usize(self.state.word_base[order]) + index / WORD_BITS,
            index % WORD_BITS,
        )
    }

    /// Reconstitutes the three borrows the state region is split into.
    ///
    /// # Safety
    ///
    /// As [`Buddy::create`]. The header, the free bitmap and the live bitmap
    /// are disjoint consecutive pieces of one region, so the three `&mut` never
    /// alias.
    unsafe fn borrow(state: NonNull<u8>, blocks: u64) -> Self {
        let words = words_for(blocks);
        let header = state.cast::<State>();
        // SAFETY: the caller guarantees `state_bytes(blocks)` writable bytes at
        // `state`, of which the header occupies the first `size_of::<State>()`
        // and the two bitmaps `words` words each after it.
        let free = unsafe { state.add(size_of::<State>()) }.cast::<u64>();
        // SAFETY: as above; the live bitmap follows the free one, and together
        // they occupy exactly the remainder of the region.
        let live = unsafe { free.add(words) };
        Self {
            // SAFETY: as above; the region is aligned for `State` and lives for
            // `'a`, and no other borrow of it exists.
            state: unsafe { &mut *header.as_ptr() },
            // SAFETY: as above, and disjoint from the header.
            free: unsafe { slice::from_raw_parts_mut(free.as_ptr(), words) },
            // SAFETY: as above, and disjoint from both the header and the free
            // bitmap.
            live: unsafe { slice::from_raw_parts_mut(live.as_ptr(), words) },
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
    /// The state region holds no allocator of this state ABI and geometry.
    #[error("no initialized allocator of this geometry at the given state")]
    NotInitialized,
    /// An order this allocator does not have. Distinct from exhaustion: no
    /// amount of freeing would make the request satisfiable.
    #[error("order {order} is beyond this allocator's largest")]
    InvalidOrder {
        /// Order offered.
        order: usize,
    },
    /// No free run of the requested order.
    #[error("no free run of order {order} remains")]
    Exhausted {
        /// Order that could not be satisfied.
        order: usize,
    },
    /// A release whose start is not a valid one for its order.
    #[error("block {index} is not a valid start for order {order}")]
    Misaligned {
        /// Block index offered.
        index: usize,
        /// Order offered.
        order: usize,
    },
    /// A release that matches no live allocation: an interior block, the wrong
    /// order, a block never handed out, or a repeat.
    #[error("no live allocation of order {order} starts at block {index}")]
    NotAllocated {
        /// Block index offered.
        index: usize,
        /// Order offered.
        order: usize,
    },
    /// A block being handed to the allocator is one it already accounts for.
    #[error("block {index} at order {order} is already the allocator's")]
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
    if blocks == 0 || !blocks.is_power_of_two() {
        return Err(BuddyError::Geometry { blocks });
    }
    let orders = blocks.trailing_zeros() as usize + 1;
    if orders > MAX_ORDERS {
        return Err(BuddyError::Geometry { blocks });
    }
    Ok(orders)
}

/// Whether `blocks` is a geometry the allocator can manage, as a constant
/// expression — so a layout can assert its block counts rather than discover
/// them at boot.
#[must_use]
pub const fn supports(blocks: u64) -> bool {
    orders_for(blocks).is_ok()
}

/// Bitmap words for `blocks`, with every order starting on a word boundary.
///
/// Zero blocks need no words, which is what keeps this total rather than
/// looping forever on a geometry no constructor would accept anyway.
const fn words_for(blocks: u64) -> usize {
    if blocks == 0 {
        return 0;
    }
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
    if blocks == 0 {
        return bases;
    }
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
