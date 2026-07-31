//! The boot protocol `hv-loader` hands to the pulzar hypervisor image.
//!
//! The loader does not start the hypervisor image the way firmware starts a
//! UEFI application. It maps the image itself, at a randomized high-half
//! virtual address, and jumps to its entry point with a pointer to a
//! [`Handoff`] in the first argument register. Everything the hypervisor needs
//! to continue — where its memory is, where its address space came from, which
//! firmware objects are still alive, and what time firmware says it is —
//! arrives in that one structure.
//!
//! The structure lives in the loader's reserved memory chunk rather than in
//! either image, because the loader's image is wiped moments after the jump.
//! The pointer the hypervisor receives is a direct-map address, so it stays
//! valid after the firmware half of the address space is dropped.
//!
//! Reading it is a validated operation, not a blind dereference: the same
//! entry point is reachable by a user launching the image from the UEFI shell,
//! in which case the first argument is a firmware image handle and not a
//! handoff at all. [`Handoff::from_ptr`] tells the two apart by magic value.

#![no_std]

use core::ffi::c_void;

use thiserror::Error;
use uefi_raw::{Handle, table::system::SystemTable};

/// Layout the loader and the hypervisor image must agree on.
///
/// Bump [`Handoff::VERSION`] whenever a field changes meaning, moves, or is
/// added: the two images are built together but staged as separate files, and
/// a mismatched pair must fail loudly at the jump instead of misreading each
/// other's memory.
#[derive(Debug)]
#[repr(C)]
pub struct Handoff {
    /// [`Handoff::MAGIC`]. First field so validation can read it before
    /// trusting anything else about the pointer.
    pub magic: u64,
    /// [`Handoff::VERSION`].
    pub version: u32,
    /// `size_of::<Handoff>()` as the loader saw it.
    pub size: u32,

    /// UEFI system table, and through it boot services. Valid only until the
    /// hypervisor drops the firmware half of its address space.
    pub system_table: *mut SystemTable,
    /// Image handle of `hv-loader`, for the `UnloadImage` that evicts it. The
    /// hypervisor image has no handle of its own: firmware never loaded it.
    pub loader_image_handle: Handle,
    /// Physical base of the loader's image, to be wiped after it unloads.
    pub loader_image_base: u64,
    /// Byte length of the loader's image, page-aligned.
    pub loader_image_size: u64,
    /// Handle of the Windows Boot Manager image firmware loaded for the guest.
    pub guest_image_handle: Handle,

    /// Physical base of the reserved chunk, 2 MiB aligned. This is the only
    /// memory the hypervisor owns.
    pub chunk_base: u64,
    /// Byte length of the reserved chunk.
    pub chunk_size: u64,
    /// Which arrangement of the chunk's metadata the loader wrote, as
    /// `paging::chunk::LAYOUT`.
    ///
    /// The version above protects this structure; this protects the bytes
    /// inside the chunk that it points at. They are separate questions and they
    /// change for separate reasons: a field added here moves nothing in the
    /// chunk, and a metadata region that grows moves everything after it while
    /// leaving every field here where it was. A pair of images that agreed on
    /// one and not the other would read each other's allocator bitmaps at the
    /// wrong offsets and find nothing to complain about.
    pub chunk_layout: u64,

    /// Physical address of the PML4 the loader built and activated.
    pub page_table_root: u64,
    /// Virtual base of the direct map of physical memory.
    pub direct_map_base: u64,
    /// Byte length of the direct map, covering physical `0..top_of_ram`.
    pub direct_map_size: u64,
    /// Virtual base of the region explicit mappings are carved from.
    pub mapping_window_base: u64,
    /// Byte length of the mapping window.
    pub mapping_window_size: u64,

    /// Virtual base the hypervisor image was relocated to and mapped at.
    pub core_image_base: u64,
    /// Byte length of the hypervisor image's virtual span.
    pub core_image_size: u64,
    /// Virtual base of the mapped part of the initial stack, guard pages
    /// excluded.
    pub stack_base: u64,
    /// Byte length of the mapped part of the initial stack.
    pub stack_size: u64,

    /// Direct-map address of the UEFI memory map the loader copied into the
    /// chunk. Unlike the firmware's own copy, this survives phase 3.
    pub memory_map: u64,
    /// Number of descriptors at [`Handoff::memory_map`].
    pub memory_map_entries: u32,
    /// Stride between descriptors, which UEFI is free to make larger than the
    /// structure it documents.
    pub memory_map_entry_size: u32,
    /// One past the highest physical address the memory map describes as
    /// memory, and so the extent the direct map covers. Device apertures are
    /// excluded: they can sit far above the last byte of RAM and must be
    /// reached by an explicit mapping, not through the direct map.
    pub top_of_ram: u64,

    /// Physical address of the ACPI Root System Description Pointer, as
    /// firmware published it in the UEFI configuration table, or zero if it
    /// published none.
    ///
    /// Only the root pointer travels in the protocol. Everything it leads to
    /// lies in memory the direct map covers, so the hypervisor reads the
    /// tables themselves long after firmware is gone rather than having them
    /// copied for it.
    pub acpi_rsdp: u64,

    /// Nanoseconds since the Unix epoch, in UTC, as firmware's real-time clock
    /// read when the loader asked it — or zero if firmware would not say, or
    /// said something the calendar does not admit.
    ///
    /// This is the only absolute time pulzar is ever handed. The hypervisor
    /// runs on after boot services are gone, and nothing left in the machine
    /// then knows what year it is: the counters it keeps time with only count.
    /// So the reading is taken once, while there is still firmware to take it
    /// from, and everything after it is that number plus elapsed time.
    pub boot_wall_nanos: u64,

    /// Physical base of the page reserved for the trampoline the other
    /// processors start on, always below 1 MiB and always frame-aligned.
    ///
    /// A processor answering a startup interprocessor interrupt begins in real
    /// mode at `vector << 12`, and the vector is eight bits wide, so the first
    /// instruction it executes has to be somewhere in the first megabyte. That
    /// is firmware's memory, and firmware is still using it — its own idle
    /// processors are parked down there — so the page is asked for rather than
    /// picked. It is reserved memory, like the chunk, which is what lets the
    /// other processors be started long after the loader is gone.
    pub ap_trampoline_base: u64,

    /// Direct-map address of the state firmware was running with, captured
    /// before the loader had modified any of it.
    ///
    /// Everything a guest that continues the firmware environment has to be
    /// entered with, and everything a virtual interrupt controller has to be
    /// seeded from. It travels here rather than being re-read because there is
    /// nothing left to re-read it from: by the time the hypervisor wants it,
    /// every register it describes holds pulzar's value instead.
    ///
    /// A direct-map address, like [`Handoff::memory_map`], so it survives the
    /// firmware half of the address space being dropped.
    pub firmware_context: u64,
}

impl Handoff {
    /// Identifies a real handoff. `"PULZARH1"`, chosen to be recognizable in a
    /// hex dump and impossible to confuse with the firmware image handle that
    /// arrives in the same register when the image is launched as a UEFI
    /// application.
    pub const MAGIC: u64 = u64::from_le_bytes(*b"PULZARH1");

    /// Current protocol version.
    pub const VERSION: u32 = 7;

    /// Validates `ptr` and borrows the handoff behind it.
    ///
    /// The reference is `'static` because the handoff lives in the reserved
    /// chunk, which is never freed and never reused.
    ///
    /// # Errors
    ///
    /// [`HandoffError::NotAHandoff`] when the magic does not match, which is
    /// the expected outcome of being launched as a UEFI application rather
    /// than jumped to by the loader. [`HandoffError::Version`] or
    /// [`HandoffError::Size`] when a loader and a hypervisor image from
    /// different builds were staged together.
    ///
    /// # Safety
    ///
    /// `ptr` must be readable for eight bytes. Every value the loader can pass
    /// satisfies this, and so does a firmware image handle, which is what
    /// makes the magic check a sound way to distinguish them.
    pub unsafe fn from_ptr(ptr: *const c_void) -> Result<&'static Self, HandoffError> {
        if ptr.is_null() {
            return Err(HandoffError::NotAHandoff { magic: 0 });
        }
        // SAFETY: the caller guarantees eight readable bytes. The read is
        // unaligned because at this point the pointer is only a candidate: a
        // firmware handle carries no alignment promise for our layout.
        let magic = unsafe { ptr.cast::<u64>().read_unaligned() };
        if magic != Self::MAGIC {
            return Err(HandoffError::NotAHandoff { magic });
        }
        // SAFETY: the magic matched, so this is a `Handoff` the loader wrote
        // into the reserved chunk: correctly sized, aligned, and initialized,
        // in memory that outlives the hypervisor.
        let handoff = unsafe { &*ptr.cast::<Self>() };
        if handoff.version != Self::VERSION {
            return Err(HandoffError::Version {
                found: handoff.version,
            });
        }
        // Compared as `u64` so that neither side needs a fallible narrowing on a
        // path that must not panic.
        let found = u64::from(handoff.size);
        let expected = size_of::<Self>() as u64;
        if found != expected {
            return Err(HandoffError::Size { found, expected });
        }
        Ok(handoff)
    }
}

/// Why a handoff pointer was rejected.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum HandoffError {
    /// The magic did not match, so the pointer is not a handoff. Firmware
    /// starting the image as an application lands here.
    #[error("not a pulzar handoff (magic {magic:#018x})")]
    NotAHandoff {
        /// What was found where the magic should have been.
        magic: u64,
    },
    /// A loader and a hypervisor image from different builds.
    #[error("handoff version {found} is not the expected {expected}", expected = Handoff::VERSION)]
    Version {
        /// Version the loader wrote.
        found: u32,
    },
    /// Same protocol version, different structure size — one side was rebuilt
    /// without bumping [`Handoff::VERSION`].
    #[error("handoff is {found} bytes, expected {expected}")]
    Size {
        /// Size the loader wrote.
        found: u64,
        /// Size this image was compiled against.
        expected: u64,
    },
}
