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
//! - [`adopt`] and [`with`] are how a machine with more than one processor
//!   reaches the single [`AddressSpace`] all of them run in; [`shootdown`] is
//!   how the others are told a translation they may hold is gone.
//!
//! # More than one processor
//!
//! Two things change once the other processors are running, and only two.
//!
//! The address space stops being a value one function owns and becomes
//! something behind a lock, because there is no `&mut` to hand a processor that
//! was not there when the space was built. Everything below that is unchanged:
//! a page table entry is an aligned eight-byte store, which the hardware page
//! walker already reads atomically, so the lock is the whole of what is needed
//! and per-entry atomics would add nothing. Their absence is deliberate.
//!
//! And invalidating an entry stops being a local matter. [`shootdown`] is the
//! seam for that, empty until something fills it — which is exactly right on a
//! machine where nothing else is running.
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
pub mod shootdown;

mod direct;
mod frames;
mod global;
mod slots;
mod space;

use core::ptr::NonNull;

pub use direct::DirectMap;
pub use frames::Frames;
pub use global::{adopt, adopted, try_with, with};
pub use slots::Slots;
pub use space::{AddressSpace, CacheType, Existing, Mapping, Protection, Ram, Stack};
use thiserror::Error;
use x86_64::{PhysAddr, VirtAddr};

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
/// Every such conversion goes through here, so the width assumption above is
/// stated once instead of at each call site. `try_from` in its place would be
/// error handling for a state the assertion rules out, on paths that must not
/// panic.
///
/// Public because the same conversion is needed wherever a chunk offset or size
/// from this crate has to be a length in memory, and a second copy of it
/// elsewhere would be a second place the width assumption lives.
#[expect(
    clippy::cast_possible_truncation,
    reason = "usize is 64 bits wide on this crate's only target, asserted above"
)]
#[must_use]
pub const fn as_usize(value: u64) -> usize {
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
///
/// # Errors
///
/// [`PagingError::Arithmetic`] if rounding up would leave the range of a `u64`,
/// which means the memory map described an address space that cannot exist.
/// Rounding is checked rather than wrapping because the result sizes a mapping:
/// a `top_of_ram` within a gigabyte of `u64::MAX` would otherwise round to zero
/// and produce a direct map covering nothing while every caller believed it
/// covered everything.
pub const fn direct_map_size(top_of_ram: u64) -> Result<u64, PagingError> {
    const GIB: u64 = 1 << 30;
    match round_up(top_of_ram, GIB) {
        Some(size) => Ok(size),
        None => Err(PagingError::Arithmetic {
            what: "rounding the top of RAM up to a gigabyte",
        }),
    }
}

/// `value` rounded up to a multiple of `align`, or `None` if that is not
/// representable.
///
/// `u64::next_multiple_of` panics on overflow in a debug build and wraps in a
/// release one, and neither is an answer this crate can act on: every use of a
/// rounded size here goes on to size a mapping or a reservation.
pub(crate) const fn round_up(value: u64, align: u64) -> Option<u64> {
    value.checked_next_multiple_of(align)
}

/// One past the last byte of a `len`-byte run starting at `start`.
///
/// The single place a half-open range's end is formed, so that no caller
/// computes `start + len` and finds out at the wrap what it should have found
/// out before mutating anything.
pub(crate) const fn end_of(start: u64, len: u64, what: &'static str) -> Result<u64, PagingError> {
    match start.checked_add(len) {
        Some(end) => Ok(end),
        None => Err(PagingError::Arithmetic { what }),
    }
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
    /// An address or length calculation could not be carried out over the whole
    /// range it was asked about: it would have left the range of the type, left
    /// the canonical address space, or crossed the hole in the middle of it.
    ///
    /// Separate from the errors that describe a *reachable* but unsuitable
    /// address, because this one says the request could not even be formed —
    /// and because it is always reported before anything has been changed.
    #[error("{what} is not representable")]
    Arithmetic {
        /// What was being computed.
        what: &'static str,
    },
    /// The window in use does not reach every byte of a range the operation
    /// needs. Before the direct map exists this means firmware's identity map
    /// falls short; afterwards it means the range reaches above `top_of_ram`.
    ///
    /// The whole range is answered for, not just where it starts: a run
    /// beginning just below the top of a window ends above it.
    #[error("physical {phys:#x}+{len:#x} is outside the current physical-access window")]
    Unreachable {
        /// Where the range that could not be reached begins.
        phys: u64,
        /// How many bytes of it were asked for.
        len: u64,
    },
    /// A pointer was asked for at an address that is not aligned for the type,
    /// or for a type with no bytes to point at.
    ///
    /// Distinct from [`PagingError::Unreachable`]: the window does cover the
    /// address, and the request is the thing that is wrong.
    #[error("physical {phys:#x} is not a usable address for a {bytes}-byte value")]
    BadPointer {
        /// The address in question.
        phys: u64,
        /// Bytes the type occupies.
        bytes: usize,
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
    /// A page table in the hierarchy could not be reached through the window,
    /// so the walk could not be performed at all.
    ///
    /// Deliberately not the same answer as "nothing is mapped there": one says
    /// the address has no translation, the other says this subsystem can no
    /// longer read its own tables, which is a broken invariant rather than a
    /// fact about an address.
    #[error("the page table at physical {phys:#x} is not reachable through the window")]
    TableUnreachable {
        /// Physical address of the table that could not be reached.
        phys: u64,
    },
    /// The address space has already been handed over, and every processor is
    /// running in it.
    #[error("an address space has already been adopted")]
    AlreadyAdopted,
    /// No address space has been handed over yet, so there is not yet one the
    /// whole machine shares.
    #[error("no address space has been adopted")]
    NotAdopted,
    /// The address space lock is held by this processor already, or by another
    /// one for longer than a caller that would not wait is prepared to.
    #[error("the address space is in use")]
    InUse,
    /// Some processor did not acknowledge dropping a translation that no longer
    /// describes anything. What was unmapped is unmapped; what is not known is
    /// whether every processor has stopped believing otherwise.
    #[error("a processor did not acknowledge dropping a stale translation")]
    ShootdownIncomplete,
    /// An operation failed and undoing what it had already done failed too, so
    /// the resources involved could not be proved detached and were retired
    /// instead of returned.
    ///
    /// The address space is consistent — nothing is described that should not
    /// be — but some frames or window addresses are now permanently held. This
    /// is reported rather than logged because the alternative to retiring them
    /// is handing out memory that may still be mapped.
    #[error("cleanup after a failed operation could not complete at {virt:#x}")]
    CleanupFailed {
        /// Where cleanup stopped.
        virt: u64,
    },
    /// The processor has no page attribute table, so the cache type a mapping
    /// asks for cannot be established.
    #[error("the processor does not support the page attribute table")]
    PatUnsupported,
    /// `CR4.PCIDE` is set. Process-context identifiers change what invalidating
    /// a translation reaches, and this crate's shootdowns do not enumerate
    /// contexts; running under them would leave stale translations in every
    /// context but the current one.
    #[error("process-context identifiers are enabled, which pulzar does not support")]
    PcidEnabled,
    /// No source of entropy the placement of the high half may be drawn from.
    #[error("the processor offers no hardware entropy source")]
    NoSecureEntropy,
    /// The hardware entropy source stopped answering part-way through a draw.
    #[error("the hardware entropy source failed")]
    EntropyFailed,
    /// A value the loader recorded about the chunk's layout does not match what
    /// this image was built for, so nothing in the chunk can be trusted.
    #[error("{field} is {found:#x}, but this image was built for {expected:#x}")]
    LayoutMismatch {
        /// Which value disagreed.
        field: &'static str,
        /// What this image expects.
        expected: u64,
        /// What the loader recorded.
        found: u64,
    },
    /// The buddy allocator underneath refused the operation.
    #[error(transparent)]
    Buddy(#[from] BuddyError),
}

/// A physical address `offset` bytes past `base`.
///
/// `PhysAddr`'s own addition panics on a value the architecture cannot
/// represent, and a panic is not an answer any caller here can act on: these
/// offsets come from a handoff another image wrote.
pub(crate) fn phys_at(
    base: PhysAddr,
    offset: u64,
    what: &'static str,
) -> Result<PhysAddr, PagingError> {
    base.as_u64()
        .checked_add(offset)
        .and_then(|value| PhysAddr::try_new(value).ok())
        .ok_or(PagingError::Arithmetic { what })
}

/// A virtual address `offset` bytes past `base`.
///
/// As [`phys_at`], and with the canonical hole to answer for as well: adding to
/// a lower-half address can land in the hole, which is not an address at all.
pub(crate) fn virt_at(
    base: VirtAddr,
    offset: u64,
    what: &'static str,
) -> Result<VirtAddr, PagingError> {
    base.as_u64()
        .checked_add(offset)
        .and_then(|value| VirtAddr::try_new(value).ok())
        .ok_or(PagingError::Arithmetic { what })
}

/// Pointer to a fixed metadata region inside the chunk, valid for all `bytes`
/// of it.
///
/// Requesting the pointer as `u64` is what enforces the eight-byte alignment
/// the buddy allocator's header needs; the chunk's own alignment and the
/// layout's frame-aligned offsets guarantee it holds.
///
/// The length matters as much as the address. A window that reaches the start
/// of the frame allocator's state but not its last bitmap word would otherwise
/// produce a pointer that succeeds here and faults, or corrupts whatever
/// follows the window, at the first allocation.
fn state_ptr(
    chunk_base: PhysAddr,
    window: DirectMap,
    offset: u64,
    bytes: usize,
) -> Result<NonNull<u8>, PagingError> {
    let phys = phys_at(chunk_base, offset, "the address of an allocator's state")?;
    window
        .bytes_ptr::<u64>(phys, bytes)
        .map(NonNull::cast::<u8>)
}
