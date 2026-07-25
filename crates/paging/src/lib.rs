//! Address space and physical memory management for pulzar.
//!
//! The hypervisor owns exactly one region of physical memory — the chunk
//! firmware reserved for it — and one half of one virtual address space. This
//! crate manages both, from the moment `hv-loader` builds the first page tables
//! to long after the firmware environment is gone.
//!
//! Nothing here depends on UEFI, by design. The address space outlives boot
//! services and eventually `ExitBootServices`, so the subsystem that manages it
//! cannot be built on anything that does not. The one concession to the boot
//! sequence is [`DirectMap::identity`], which lets the loader address page
//! tables through the identity map firmware happens to have set up.
//!
//! # Shape
//!
//! - [`buddy`] is the bookkeeping both allocators run on: a bitmap buddy whose
//!   state is pointer-free and lives at a fixed offset in the chunk, so one
//!   image can create it and another can adopt it.
//! - [`Frames`] hands out physical frames from the chunk; [`Slots`] hands out
//!   virtual addresses from the mapping window. Same algorithm, different
//!   block.
//! - [`DirectMap`] is the linear window onto physical memory: cheap access to
//!   any physical address, and the only way to reach the chunk once firmware's
//!   identity map is gone.
//! - [`AddressSpace`] owns a PML4 and drives all of the above, including the
//!   phase transitions.
//! - [`kaslr`] places the high-half regions; [`cpu`] establishes the processor
//!   state the rest of it assumes; [`chunk`] fixes the geometry both images
//!   agree on.
//!
//! # Two kinds of physical access
//!
//! [`DirectMap`] is for reading a value or editing one in place — it is always
//! read-write, no-execute, write-back, and it exists for convenience and for
//! reaching our own tables. Memory that will be used repeatedly, or that needs
//! a particular protection or cache type, gets a real mapping from
//! [`AddressSpace::map_physical`] in the randomized mapping window.

#![no_std]

pub mod buddy;
pub mod chunk;
pub mod cpu;
pub mod kaslr;

mod direct;
mod frames;
mod slots;
mod space;

use core::ptr::NonNull;

pub use direct::DirectMap;
pub use frames::Frames;
pub use slots::Slots;
pub use space::{AddressSpace, CacheType, Existing, Mapping, Protection, Stack};
use thiserror::Error;
use x86_64::PhysAddr;

use crate::buddy::BuddyError;

/// Block indices and bitmap words are counted in `usize` while addresses and
/// lengths are `u64`, so the two are converted constantly. That is lossless
/// exactly while they are the same width, which this crate's only target
/// guarantees — but silently would not be if that ever changed.
const _: () = assert!(
    size_of::<usize>() == size_of::<u64>(),
    "paging assumes 64-bit pointers"
);

/// A count or address as `usize`.
///
/// Every such conversion in the crate goes through here, so the width
/// assumption above is stated once instead of at each call site. `try_from` in
/// its place would be error handling for a state the assertion rules out, on
/// paths that must not panic.
#[expect(
    clippy::cast_possible_truncation,
    reason = "usize is 64 bits wide on this crate's only target, asserted above"
)]
pub(crate) const fn as_usize(value: u64) -> usize {
    value as usize
}

/// A count or index as `u64`.
pub(crate) const fn as_u64(value: usize) -> u64 {
    value as u64
}

/// Bytes the direct map must cover to reach every physical address below
/// `top_of_ram`.
///
/// Rounded up to a gigabyte so the map can be described in 1 GiB pages and so
/// its randomized base needs no size-dependent adjustment.
#[must_use]
pub const fn direct_map_size(top_of_ram: u64) -> u64 {
    const GIB: u64 = 1 << 30;
    top_of_ram.div_ceil(GIB) * GIB
}

/// Why an address-space or allocation operation was refused.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum PagingError {
    /// `CR4.LA57` is set. Under 5-level paging the root table is a PML5, and
    /// every walk this crate performs would be off by a level; refusing to boot
    /// is the only safe response.
    #[error("5-level paging is enabled, which pulzar does not support")]
    FiveLevelPaging,
    /// The processor has no `NX`, so no mapping could be made non-executable.
    #[error("the processor does not support the no-execute bit")]
    NoExecuteUnsupported,
    /// A region does not fit the slice of the high half reserved for it.
    #[error("a {size:#x}-byte region does not fit a {span:#x}-byte high-half slice")]
    RegionTooLarge {
        /// Bytes the region needs.
        size: u64,
        /// Bytes available.
        span: u64,
    },
    /// An address or length did not meet an alignment the operation requires.
    #[error("{value:#x} is not {align:#x}-aligned")]
    Misaligned {
        /// The offending value.
        value: u64,
        /// The alignment required.
        align: u64,
    },
    /// A zero-length region was asked for.
    #[error("a region must be at least one byte")]
    EmptyRegion,
    /// The window in use does not reach a physical address the operation needs.
    /// Before the direct map exists this means firmware's identity map falls
    /// short; afterwards it means the address is above `top_of_ram`.
    #[error("physical {phys:#x} is outside the current physical-access window")]
    Unreachable {
        /// The address that could not be reached.
        phys: u64,
    },
    /// A frame that did not come from the chunk was offered back to it.
    #[error("physical {phys:#x} is not a frame of the reserved chunk")]
    NotOurs {
        /// The offending address.
        phys: u64,
    },
    /// A page outside the mapping window was offered back to it.
    #[error("virtual {virt:#x} is not a page of the mapping window")]
    OutsideWindow {
        /// The offending address.
        virt: u64,
    },
    /// The chunk has no free run of this order left.
    #[error("the reserved chunk has no free run of order {order}")]
    OutOfFrames {
        /// Order that could not be satisfied.
        order: usize,
    },
    /// The mapping window has no free run of this order left.
    #[error("the mapping window has no free run of order {order}")]
    OutOfWindow {
        /// Order that could not be satisfied.
        order: usize,
    },
    /// Firmware already has a high-half mapping, so the halves cannot simply be
    /// combined.
    #[error("firmware already maps PML4 entry {index}, which pulzar needs")]
    HighHalfInUse {
        /// The occupied entry.
        index: usize,
    },
    /// Something is already mapped where a new mapping was to go.
    #[error("virtual {virt:#x} is already mapped")]
    AlreadyMapped {
        /// The address in question.
        virt: u64,
    },
    /// A large page covers the address, so it cannot be described at a finer
    /// granularity without splitting it first.
    #[error("virtual {virt:#x} is covered by a large page")]
    ParentHugePage {
        /// The address in question.
        virt: u64,
    },
    /// Nothing is mapped at the address.
    #[error("virtual {virt:#x} is not mapped")]
    NotMapped {
        /// The address in question.
        virt: u64,
    },
    /// A page table entry held an address that is not a valid frame base.
    #[error("page table entry holds the invalid frame address {phys:#x}")]
    InvalidFrame {
        /// The address that was found.
        phys: u64,
    },
    /// The buddy allocator underneath refused the operation.
    #[error(transparent)]
    Buddy(#[from] BuddyError),
}

/// Pointer to a fixed metadata region inside the chunk.
///
/// Requesting the pointer as `u64` is what enforces the eight-byte alignment
/// the buddy allocator's header needs; the chunk's own alignment and the
/// layout's frame-aligned offsets guarantee it holds.
fn state_ptr(
    chunk_base: PhysAddr,
    window: DirectMap,
    offset: u64,
) -> Result<NonNull<u8>, PagingError> {
    let phys = chunk_base + offset;
    window
        .ptr::<u64>(phys)
        .map(NonNull::cast::<u8>)
        .ok_or(PagingError::Unreachable {
            phys: phys.as_u64(),
        })
}
