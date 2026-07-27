//! The two guest-callable pages that transfer into Windows Boot Manager.
//!
//! The portal is copied from a position-independent assembly blob into fixed
//! chunk metadata. Its first page installs the boot-services wrapper and calls
//! `StartImage`; its second page holds immutable arguments and the original
//! `ExitBootServices` pointer the wrapper calls. No host virtual address is
//! embedded in it, because the guest runs under firmware's page tables.

use core::mem::offset_of;

use handoff::Handoff;
use paging::{DirectMap, chunk};
use thiserror::Error;
use uefi_raw::table::{Header, boot::BootServices, system::SystemTable};
use x86_64::PhysAddr;

/// Portal notification that the original `ExitBootServices` returned success.
pub(crate) const EXIT_SUCCEEDED: u64 = u64::from_le_bytes(*b"EBS DONE");

/// Portal notification that `StartImage` returned instead of transferring the
/// machine to the operating system.
pub(crate) const START_RETURNED: u64 = u64::from_le_bytes(*b"STARTRET");

/// The portal placed in the chunk for the initial guest processor.
#[derive(Debug)]
pub(crate) struct Portal {
    base: PhysAddr,
}

impl Portal {
    /// Copies the portal blob and fills the arguments its entry consumes.
    ///
    /// # Errors
    ///
    /// [`PortalError::Unreachable`] if the direct map cannot reach the fixed
    /// metadata pages, [`PortalError::InvalidLayout`] if the assembly no longer
    /// places its parameters at the second page, or [`PortalError::TooLarge`]
    /// if the assembly no longer fits in them.
    pub(crate) fn place(window: DirectMap, handoff: &Handoff) -> Result<Self, PortalError> {
        let base = PhysAddr::new(handoff.chunk_base + chunk::PORTAL_OFFSET);
        let at = window.ptr::<u8>(base).ok_or(PortalError::Unreachable {
            phys: base.as_u64(),
        })?;
        let blob = blob();
        let page = crate::bytes(chunk::FRAME_SIZE);
        let room = crate::bytes(chunk::PORTAL_SIZE);
        if data_offset() != page {
            return Err(PortalError::InvalidLayout);
        }
        if blob.len() > room {
            return Err(PortalError::TooLarge {
                bytes: blob.len(),
                room,
            });
        }
        // SAFETY: the portal pages are fixed chunk metadata, hence allocated to
        // no subsystem; `at` reaches their first byte and `blob` was bounded to
        // their size immediately above. Clearing the whole reservation prevents
        // bytes outside the assembled parameters from exposing prior contents.
        unsafe {
            at.as_ptr().write_bytes(0, room);
            at.as_ptr()
                .copy_from_nonoverlapping(blob.as_ptr(), blob.len());
        };
        let parameters =
            window
                .ptr::<Parameters>(base + chunk::FRAME_SIZE)
                .ok_or(PortalError::Unreachable {
                    phys: base.as_u64() + chunk::FRAME_SIZE,
                })?;
        let system = window
            .ptr::<SystemTable>(PhysAddr::new_truncate(handoff.system_table as u64))
            .ok_or(PortalError::Unreachable {
                phys: handoff.system_table as u64,
            })?;
        // SAFETY: the handoff pointer is firmware's live system table and the
        // direct-map pointer above reaches that physical table.
        let boot_services = unsafe { (*system.as_ptr()).boot_services };
        let boot_services = window
            .ptr::<BootServices>(PhysAddr::new_truncate(boot_services as u64))
            .ok_or(PortalError::MissingBootServices)?;
        // SAFETY: firmware's boot-services table remains initialized and
        // readable because the guest has not called ExitBootServices yet.
        let original = unsafe { (*boot_services.as_ptr()).exit_boot_services } as usize as u64;
        // SAFETY: the second portal page belongs only to this portal and has
        // `Parameters` alignment because every chunk page is frame-aligned.
        unsafe {
            parameters.as_ptr().write(Parameters {
                system_table: handoff.system_table as u64,
                guest_image_handle: handoff.guest_image_handle as u64,
                original_exit_boot_services: original,
            });
        }
        Ok(Self { base })
    }

    /// Guest physical address of the first portal instruction.
    pub(crate) const fn entry(&self) -> PhysAddr {
        self.base
    }

    /// Guest physical address of the immutable parameter page.
    pub(crate) fn parameters(&self) -> PhysAddr {
        self.base + chunk::FRAME_SIZE
    }
}

/// Arguments the assembly reads from the second portal page.
#[repr(C)]
struct Parameters {
    system_table: u64,
    guest_image_handle: u64,
    original_exit_boot_services: u64,
}

unsafe extern "C" {
    static pulzar_portal_start: u8;
    static pulzar_portal_data: u8;
    static pulzar_portal_end: u8;
}

core::arch::global_asm!(
    include_str!("portal.s"),
    SYSTEM_TABLE = const offset_of!(Parameters, system_table),
    GUEST_IMAGE_HANDLE = const offset_of!(Parameters, guest_image_handle),
    ORIGINAL_EXIT_BOOT_SERVICES = const offset_of!(Parameters, original_exit_boot_services),
    SYSTEM_BOOT_SERVICES = const offset_of!(SystemTable, boot_services),
    BOOT_EXIT_BOOT_SERVICES = const offset_of!(BootServices, exit_boot_services),
    BOOT_START_IMAGE = const offset_of!(BootServices, start_image),
    BOOT_CALCULATE_CRC32 = const offset_of!(BootServices, calculate_crc32),
    HEADER_SIZE = const offset_of!(Header, size),
    HEADER_CRC = const offset_of!(Header, crc),
    EXIT_SUCCEEDED = const EXIT_SUCCEEDED,
    START_RETURNED = const START_RETURNED,
);

/// Why the portal could not be prepared.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub(crate) enum PortalError {
    /// The direct map does not reach a fixed portal page.
    #[error("the direct map does not reach portal memory at {phys:#x}")]
    Unreachable {
        /// Physical address that could not be reached.
        phys: u64,
    },
    /// The assembled blob exceeds the two reserved pages.
    #[error("the {bytes:#x}-byte portal does not fit its {room:#x}-byte reservation")]
    TooLarge {
        /// Bytes in the assembled blob.
        bytes: usize,
        /// Bytes reserved in the chunk.
        room: usize,
    },
    /// The parameter symbol is not exactly one page after the entry symbol.
    #[error("the portal parameter page is not one page after its entry")]
    InvalidLayout,
    /// The firmware system table has no boot-services table.
    #[error("the UEFI system table has no boot-services table")]
    MissingBootServices,
}

/// The bytes copied into the portal reservation.
fn blob() -> &'static [u8] {
    let start = &raw const pulzar_portal_start;
    let end = &raw const pulzar_portal_end;
    let bytes = end as usize - start as usize;
    // SAFETY: both symbols delimit one contiguous assembly section, ordered by
    // the assembler; no Rust reference aliases its destination until copied.
    unsafe { core::slice::from_raw_parts(start, bytes) }
}

/// Offset of the parameter symbol from the entry symbol.
fn data_offset() -> usize {
    let start = &raw const pulzar_portal_start;
    let data = &raw const pulzar_portal_data;
    data as usize - start as usize
}
