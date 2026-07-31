//! Geometry of the one physical region pulzar owns, and the fixed layout of the
//! bookkeeping at its front.
//!
//! Firmware hands the loader a single reserved region and that is all the
//! physical memory the hypervisor may touch as its own: everything else belongs
//! to the firmware or the guest. Both images therefore have to agree on where
//! inside it each piece of metadata lives, because the loader writes that
//! metadata and the hypervisor image reads it after the loader is gone. Fixed
//! offsets rather than pointers, so nothing in the chunk depends on where the
//! chunk itself landed.
//!
//! The metadata occupies the front of the chunk and everything past it is
//! allocatable. Nothing here is allowed to grow at runtime.
//!
//! # Why the assertions at the end are the interesting part
//!
//! Every unsafe operation in this crate that reaches into the chunk is sound
//! only because of a property of these numbers: that a region starts where the
//! previous one ends, that it is frame-aligned, that it is large enough for
//! what is written into it, that it lies inside the chunk, that the block
//! counts are geometries the buddy allocator can manage. Those properties are
//! not self-evident from a column of `+` expressions, and a single edited
//! constant can break one of them while leaving the crate compiling and the
//! failure to be discovered as a corrupted allocator at boot.
//!
//! So each of them is asserted, individually and by name, at the bottom of this
//! file. They cost nothing at run time and they are the only place the layout's
//! invariants are stated in a form that a change has to satisfy.
//!
//! # Why the layout carries a version
//!
//! The two images are separate files that can be staged independently. The
//! handoff's own version protects the *structure* the loader passes; it says
//! nothing about the bytes inside the chunk that structure points at. A loader
//! that reserved twenty pages for the window allocator's state and an image
//! that expects thirty-three would agree on every field of the handoff and
//! disagree about where the allocator's bitmap ends. [`LAYOUT`] is what that
//! pair disagrees on loudly instead.

use x86_64::structures::paging::{PageSize, Size2MiB, Size4KiB};

use crate::{as_usize, buddy};

/// Bytes in the smallest page this subsystem allocates or maps.
pub const FRAME_SIZE: u64 = 4096;

/// Bytes in the reserved chunk.
pub const CHUNK_SIZE: u64 = 64 << 20;

/// Alignment the loader must give the chunk.
///
/// Firmware only promises 4 KiB from `AllocatePages`. Two megabytes is what
/// lets the direct map describe the chunk in large pages and lets a 2
/// MiB-aligned image allocation be protected in the direct map without
/// splitting anything.
pub const CHUNK_ALIGN: u64 = 2 << 20;

/// Frames in the reserved chunk.
pub const CHUNK_FRAMES: u64 = CHUNK_SIZE / FRAME_SIZE;

/// Bytes of virtual address space reserved for explicit physical mappings.
///
/// A whole 1 GiB so the window is 1 GiB aligned and its randomized base needs
/// no size-dependent clamping.
pub const MAPPING_WINDOW_SIZE: u64 = 1 << 30;

/// Page-sized slots in the mapping window.
pub const MAPPING_WINDOW_SLOTS: u64 = MAPPING_WINDOW_SIZE / FRAME_SIZE;

/// Identifies this arrangement of the chunk's metadata.
///
/// Bumped whenever an offset moves, a region changes size, or the bytes written
/// into one change meaning — including a change to the buddy allocator's own
/// state ABI, since that decides how much of a state region is meaningful. The
/// hypervisor image compares the value the loader recorded against its own
/// before it adopts anything out of the chunk, so a mismatched pair of images
/// fails at the first check rather than at a misread bitmap.
pub const LAYOUT: u64 = 2;

/// Offset of the boot protocol structure. First, so a hex dump of the chunk
/// starts with the magic that identifies it.
pub const HANDOFF_OFFSET: u64 = 0;

/// Space set aside for the boot protocol structure.
pub const HANDOFF_SIZE: u64 = FRAME_SIZE;

/// Offset of the state firmware was running with, captured by the loader before
/// it modified any of it.
///
/// Beside the handoff rather than anywhere else, because it is the same kind of
/// thing: written once by the loader, read by the hypervisor image, and part of
/// what the two agree on rather than something either of them allocates.
pub const FIRMWARE_CONTEXT_OFFSET: u64 = HANDOFF_OFFSET + HANDOFF_SIZE;

/// Space set aside for the captured firmware state.
pub const FIRMWARE_CONTEXT_SIZE: u64 = FRAME_SIZE;

/// Offset of the physical frame allocator's state.
pub const FRAME_STATE_OFFSET: u64 = FIRMWARE_CONTEXT_OFFSET + FIRMWARE_CONTEXT_SIZE;

/// Space set aside for the physical frame allocator's state.
pub const FRAME_STATE_SIZE: u64 = 4 * FRAME_SIZE;

/// Offset of the mapping window allocator's state.
pub const WINDOW_STATE_OFFSET: u64 = FRAME_STATE_OFFSET + FRAME_STATE_SIZE;

/// Space set aside for the mapping window allocator's state.
///
/// The window is 262 144 slots, and the allocator keeps two bits per block per
/// order over nineteen orders: one saying whether a block is free, one
/// recording where a live allocation starts. The second is what makes a release
/// authenticated rather than assumed, and it is why this region is the largest
/// piece of metadata in the chunk.
pub const WINDOW_STATE_SIZE: u64 = 33 * FRAME_SIZE;

/// Offset of the loader's copy of the UEFI memory map.
pub const MEMORY_MAP_OFFSET: u64 = WINDOW_STATE_OFFSET + WINDOW_STATE_SIZE;

/// Space set aside for the copied UEFI memory map. Sixty-four kilobytes holds
/// several hundred descriptors, an order of magnitude more than firmware
/// reports on any machine this runs on; the loader refuses to boot rather than
/// truncate if a map ever exceeds it.
pub const MEMORY_MAP_SIZE: u64 = 16 * FRAME_SIZE;

/// Offset of the guest-callable firmware portal.
///
/// The first page is position-independent code and the second is immutable
/// arguments for it. They are metadata rather than allocations because the
/// guest must be able to find them before any guest allocator exists.
pub const PORTAL_OFFSET: u64 = MEMORY_MAP_OFFSET + MEMORY_MAP_SIZE;

/// Bytes reserved for the guest-callable firmware portal.
pub const PORTAL_SIZE: u64 = 2 * FRAME_SIZE;

/// Bytes of the chunk reserved for metadata, never handed out by the allocator.
pub const METADATA_SIZE: u64 = PORTAL_OFFSET + PORTAL_SIZE;

/// Frames of the chunk reserved for metadata.
pub const METADATA_FRAMES: u64 = METADATA_SIZE / FRAME_SIZE;

/// Every metadata region, as `(offset, size)`, in layout order: handoff,
/// firmware context, frame allocator state, window allocator state, memory map,
/// portal.
///
/// One list rather than a repeated pattern of assertions per region: the
/// properties below are the same for every region, and stating them once over a
/// table is what keeps adding a region from silently skipping a check.
const REGIONS: [(u64, u64); 6] = [
    (HANDOFF_OFFSET, HANDOFF_SIZE),
    (FIRMWARE_CONTEXT_OFFSET, FIRMWARE_CONTEXT_SIZE),
    (FRAME_STATE_OFFSET, FRAME_STATE_SIZE),
    (WINDOW_STATE_OFFSET, WINDOW_STATE_SIZE),
    (MEMORY_MAP_OFFSET, MEMORY_MAP_SIZE),
    (PORTAL_OFFSET, PORTAL_SIZE),
];

/// Whether every region starts on a frame boundary, is a whole number of
/// frames, is non-empty, begins exactly where the previous one ended, and ends
/// exactly at the end of the metadata area.
const fn regions_are_contiguous() -> bool {
    let mut index = 0;
    let mut expected = 0;
    while index < REGIONS.len() {
        let (offset, size) = REGIONS[index];
        if offset != expected
            || size == 0
            || !offset.is_multiple_of(FRAME_SIZE)
            || !size.is_multiple_of(FRAME_SIZE)
        {
            return false;
        }
        expected = offset + size;
        index += 1;
    }
    expected == METADATA_SIZE
}

const _: () = assert!(
    FRAME_SIZE == Size4KiB::SIZE,
    "a frame must be the hardware's smallest page: the crate maps runs of \
     frames with Size4KiB and indexes the chunk in FRAME_SIZE units"
);
const _: () = assert!(
    CHUNK_ALIGN == Size2MiB::SIZE,
    "the chunk's alignment must be the hardware's large page, so the direct map \
     can describe the chunk in large pages and tighten their protection without \
     splitting one"
);
const _: () = assert!(
    CHUNK_SIZE.is_multiple_of(CHUNK_ALIGN),
    "the chunk must be a whole number of large pages"
);
const _: () = assert!(
    CHUNK_SIZE.is_multiple_of(FRAME_SIZE) && CHUNK_FRAMES * FRAME_SIZE == CHUNK_SIZE,
    "the chunk must divide exactly into frames, so no byte of it is unaccounted for"
);
const _: () = assert!(
    MAPPING_WINDOW_SIZE.is_multiple_of(FRAME_SIZE)
        && MAPPING_WINDOW_SLOTS * FRAME_SIZE == MAPPING_WINDOW_SIZE,
    "the mapping window must divide exactly into page-sized slots"
);
const _: () = assert!(
    MAPPING_WINDOW_SIZE.is_multiple_of(Size2MiB::SIZE),
    "the mapping window must be a whole number of large pages, so its randomized \
     base needs no size-dependent adjustment"
);
const _: () = assert!(
    buddy::supports(CHUNK_FRAMES),
    "the chunk's frame count is not a geometry the buddy allocator can manage"
);
const _: () = assert!(
    buddy::supports(MAPPING_WINDOW_SLOTS),
    "the mapping window's slot count is not a geometry the buddy allocator can manage"
);
const _: () = assert!(
    regions_are_contiguous(),
    "the metadata regions must be non-empty, frame-aligned, frame-sized, \
     adjacent in the order declared, and exactly fill METADATA_SIZE"
);
const _: () = assert!(
    METADATA_SIZE.is_multiple_of(FRAME_SIZE) && METADATA_FRAMES * FRAME_SIZE == METADATA_SIZE,
    "metadata must end on a frame boundary so the allocator starts on one"
);
const _: () = assert!(
    buddy::state_bytes(CHUNK_FRAMES) <= as_usize(FRAME_STATE_SIZE),
    "FRAME_STATE_SIZE is too small for the chunk's frame count"
);
const _: () = assert!(
    buddy::state_bytes(MAPPING_WINDOW_SLOTS) <= as_usize(WINDOW_STATE_SIZE),
    "WINDOW_STATE_SIZE is too small for the mapping window's slot count"
);
const _: () = assert!(
    METADATA_SIZE < CHUNK_SIZE,
    "metadata must leave allocatable frames behind"
);
const _: () = assert!(
    METADATA_FRAMES < CHUNK_FRAMES,
    "the frame allocator must be handed at least one frame"
);
