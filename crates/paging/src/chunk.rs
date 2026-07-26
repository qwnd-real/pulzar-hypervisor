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
//! allocatable. Nothing here is allowed to grow at runtime — the sizes are
//! checked against what they must hold at compile time, below.

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
pub const WINDOW_STATE_SIZE: u64 = 20 * FRAME_SIZE;

/// Offset of the loader's copy of the UEFI memory map.
pub const MEMORY_MAP_OFFSET: u64 = WINDOW_STATE_OFFSET + WINDOW_STATE_SIZE;

/// Space set aside for the copied UEFI memory map. Sixty-four kilobytes holds
/// several hundred descriptors, an order of magnitude more than firmware
/// reports on any machine this runs on; the loader refuses to boot rather than
/// truncate if a map ever exceeds it.
pub const MEMORY_MAP_SIZE: u64 = 16 * FRAME_SIZE;

/// Bytes of the chunk reserved for metadata, never handed out by the allocator.
pub const METADATA_SIZE: u64 = MEMORY_MAP_OFFSET + MEMORY_MAP_SIZE;

/// Frames of the chunk reserved for metadata.
pub const METADATA_FRAMES: u64 = METADATA_SIZE / FRAME_SIZE;

const _: () = assert!(
    CHUNK_SIZE.is_multiple_of(CHUNK_ALIGN),
    "the chunk must be a whole number of large pages"
);
const _: () = assert!(
    METADATA_SIZE.is_multiple_of(FRAME_SIZE),
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
