//! The hypervisor's heap, and the global allocator that runs on it.
//!
//! The frame allocator hands out whole frames of the reserved chunk, naturally
//! aligned and in power-of-two runs, which is the right shape for page tables
//! and for the hardware structures a virtual machine needs — and the wrong
//! shape for everything else. This module covers the rest: arbitrary sizes,
//! arbitrary alignments, and the `alloc` collections built on them.
//!
//! # Where the memory comes from
//!
//! One contiguous run of frames, taken from the chunk once during bring-up and
//! addressed through the direct map, which is already read-write, no-execute
//! and write-back over all of physical memory. No page tables are built and no
//! mapping-window address space is spent: the heap is simply a span of bytes
//! inside memory the hypervisor already owns and can already reach.
//!
//! # Why it never grows
//!
//! Growing would mean the allocator calling the frame allocator, which lives
//! inside the [`AddressSpace`]. A caller holding the address space is exactly
//! the caller most likely to allocate, so that path would have to be sound
//! while the address space was already borrowed and while the allocator's own
//! lock was held. A heap of a size fixed at compile time gives that up and
//! gets determinism in return: the frame allocator is touched once, by one
//! caller, before anything can allocate, and never again. There is no
//! allocate-inside-allocate path to reason about because there is no code that
//! could form one.
//!
//! Exhausting it therefore fails the allocation, which Rust turns into a panic.
//! That is the correct outcome for the only phase that runs here: bring-up,
//! where a heap too small for the machine must stop the boot rather than be
//! papered over.
//!
//! # The allocator itself
//!
//! `talc` keeps its bookkeeping inside the span it manages, so the heap needs
//! no metadata region of its own, and it services a request out of a bucketed
//! free list rather than by walking one. Its [`Source`](talc::source::Source)
//! is [`Manual`]: the allocator never reaches outside itself for memory, which
//! is what makes the paragraph above a property of the type and not just of
//! this module's code.

use core::ptr::NonNull;

use log::info;
use paging::{AddressSpace, PagingError, buddy, chunk::FRAME_SIZE};
use talc::{TalcLock, source::Manual};
use x86_64::VirtAddr;

use crate::{bytes, error::CoreError};

/// Frames the heap reserves from the chunk.
///
/// Eight megabytes of the chunk's sixty-four. Generous for what allocates
/// today — the firmware tables the hypervisor caches are a few kilobytes — and
/// deliberately so, because the frames it leaves behind are still the majority
/// of the chunk and the ones a virtual machine's page tables will come out of.
const HEAP_FRAMES: u64 = 2048;

/// Bytes of heap, derived from the frame count so the two cannot disagree.
const HEAP_BYTES: u64 = HEAP_FRAMES * FRAME_SIZE;

/// The allocator every `alloc` type in this image goes through.
///
/// Locked rather than cell-based even though only one processor runs today:
/// application processors are started from the same image, and an allocator
/// that would have to be replaced to admit them is one that would be replaced
/// under load.
#[global_allocator]
static ALLOCATOR: TalcLock<spin::Mutex<()>, Manual> = TalcLock::new(Manual);

/// The span of memory [`ALLOCATOR`] hands out of.
///
/// Held by the caller of [`Heap::establish`] rather than in a static, so that
/// the one thing worth reporting about the heap — how much of it is spoken for
/// — can only be asked of a heap that was actually established.
#[derive(Clone, Copy, Debug)]
pub struct Heap {
    base: VirtAddr,
    end: NonNull<u8>,
}

impl Heap {
    /// Reserves the heap's frames and gives them to the global allocator.
    ///
    /// Nothing in this image may allocate before this returns. Calling it a
    /// second time would reserve a second run and give the allocator a second
    /// heap, which works but is not what any caller wants.
    ///
    /// # Errors
    ///
    /// [`CoreError::Paging`] if the chunk has no run this large or the direct
    /// map does not reach the one it produced, or
    /// [`CoreError::HeapRefused`] if the allocator will not take the span,
    /// which can only mean it is too small to hold the allocator's own
    /// bookkeeping.
    pub fn establish(space: &mut AddressSpace) -> Result<Self, CoreError> {
        let order = buddy::order_for(bytes(HEAP_FRAMES));
        let frames = space
            .frames()
            .allocate(order)
            .ok_or(PagingError::OutOfFrames { order })?;
        let phys = frames.start_address();
        let base = space
            .direct_map()
            .virt(phys)
            .ok_or(PagingError::Unreachable {
                phys: phys.as_u64(),
            })?;

        // SAFETY: the run was just allocated from the chunk, so this is its only
        // owner and nothing outside the allocator will write to it; it is
        // `HEAP_BYTES` of contiguous zeroed memory that the direct map keeps
        // readable and writable for as long as the hypervisor runs, and `Manual`
        // is a source that permits managing heaps by hand.
        let end = unsafe {
            ALLOCATOR
                .lock()
                .claim(base.as_mut_ptr::<u8>(), bytes(HEAP_BYTES))
        }
        .ok_or(CoreError::HeapRefused {
            base: base.as_u64(),
            bytes: HEAP_BYTES,
        })?;
        Ok(Self { base, end })
    }

    /// Logs the heap's extent and how much of it is spoken for.
    ///
    /// The reserved figure is the high-water mark of the heap rather than the
    /// sum of the live allocations: it covers the allocator's own bookkeeping
    /// and any hole a freed allocation left below it. That is the number that
    /// says whether the heap is the right size, which is the reason to log it.
    pub fn describe(&self, who: &str) {
        // SAFETY: `end` is the heap end the `claim` in `establish` returned, and
        // that heap is still one of this allocator's, since nothing ever
        // truncates it.
        let reserved = unsafe { ALLOCATOR.lock().reserved(self.end) };
        let used = if reserved.any {
            VirtAddr::from_ptr(reserved.up_to.as_ptr()) - self.base
        } else {
            0
        };
        info!(
            "{who}: heap at {:#x}, {HEAP_BYTES:#x} bytes, {used:#x} reserved",
            self.base
        );
    }
}
