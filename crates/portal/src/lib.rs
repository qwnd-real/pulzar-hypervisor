//! The two guest-callable pages that transfer firmware into the operating
//! system's boot manager.
//!
//! The guest this hypervisor starts is firmware, resumed from the state the
//! loader captured — but not at the instruction firmware stopped at. It is
//! started here instead, at a page of the hypervisor's own that is the only
//! hypervisor memory a guest is ever shown. What that page does is start the
//! boot manager firmware would have started anyway, with one thing changed:
//! the boot-services table's `ExitBootServices` pointer is replaced by a
//! wrapper of this crate's, so that the host learns the moment firmware's
//! services stop existing.
//!
//! The code is a position-independent assembly blob copied into fixed chunk
//! metadata, and the second page holds nothing but the three values it reads:
//! the system table, the image handle to start, and the original service the
//! wrapper calls. No host virtual address appears anywhere in it, because the
//! guest runs under firmware's page tables and not the hypervisor's.
//!
//! # It is meant to be taken away
//!
//! The portal is hypervisor memory the guest can execute, which is the one
//! thing the nested tables otherwise exist to prevent. It has to be visible for
//! as long as the wrapper can still be called — which ends when
//! `ExitBootServices` has returned and the guest has left the last portal
//! instruction behind. [`Portal::holds`] is how the exit handler recognizes
//! that moment, after which the pages are concealed and read as zeroes like the
//! rest of the hypervisor's memory.

#![no_std]

use core::mem::offset_of;

use handoff::Handoff;
use paging::{DirectMap, as_usize, chunk};
use thiserror::Error;
use uefi_raw::table::{Header, boot::BootServices, system::SystemTable};
use x86_64::PhysAddr;

/// The portal placed in the chunk for the initial guest processor.
#[derive(Debug)]
pub struct Portal {
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
    pub fn place(window: DirectMap, handoff: &Handoff) -> Result<Self, PortalError> {
        let base = PhysAddr::new(handoff.chunk_base + chunk::PORTAL_OFFSET);
        let at = window.ptr::<u8>(base).ok_or(PortalError::Unreachable {
            phys: base.as_u64(),
        })?;
        let blob = blob();
        let room = as_usize(chunk::PORTAL_SIZE);
        if data_offset() != as_usize(chunk::FRAME_SIZE) {
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
        }
        Self { base }.fill(window, handoff)
    }

    /// Guest physical address of the first portal instruction.
    #[must_use]
    pub const fn entry(&self) -> PhysAddr {
        self.base
    }

    /// Guest physical address of the immutable parameter page.
    #[must_use]
    pub fn parameters(&self) -> PhysAddr {
        self.base + chunk::FRAME_SIZE
    }

    /// Bytes the portal occupies, both pages together.
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        chunk::PORTAL_SIZE
    }

    /// Whether an address the guest is executing at is one of the portal's.
    ///
    /// The guest is entered at the portal's *physical* address and runs it
    /// under firmware's identity map, so while it is in the portal the address
    /// it executes at is the address the portal was placed at. An instruction
    /// pointer outside this range is therefore the guest running something that
    /// is not the portal — which is what says the portal can be taken away.
    ///
    /// The converse is a guess and is deliberately the harmless one: a guest
    /// address that merely happens to fall in this range keeps the portal a
    /// little longer, where the opposite mistake would take it away underneath
    /// a guest still executing it.
    #[must_use]
    pub fn holds(&self, rip: u64) -> bool {
        (self.base.as_u64()..self.base.as_u64() + chunk::PORTAL_SIZE).contains(&rip)
    }

    /// Writes the parameter page from firmware's live tables.
    fn fill(self, window: DirectMap, handoff: &Handoff) -> Result<Self, PortalError> {
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
        let parameters =
            window
                .ptr::<Parameters>(self.parameters())
                .ok_or(PortalError::Unreachable {
                    phys: self.parameters().as_u64(),
                })?;
        // SAFETY: the second portal page belongs only to this portal and has
        // `Parameters` alignment because every chunk page is frame-aligned.
        unsafe {
            parameters.as_ptr().write(Parameters {
                system_table: handoff.system_table as u64,
                guest_image_handle: handoff.guest_image_handle as u64,
                original_exit_boot_services: original,
            });
        }
        Ok(self)
    }
}

/// What the portal tells the host, in the register its `vmmcall` carries.
///
/// Two moments, and only two: the one where firmware's services have gone, and
/// the one where the boot manager came back instead of taking the machine.
/// Recognizable byte strings rather than small integers, so that a `vmmcall`
/// some other part of the guest makes cannot be read as one of these.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u64)]
pub enum Notification {
    /// The original `ExitBootServices` returned success. Firmware's services
    /// are gone, the memory map is final, and the guest is about to return into
    /// the boot manager.
    ExitSucceeded = u64::from_le_bytes(*b"EBS DONE"),
    /// `StartImage` returned rather than transferring the machine to the
    /// operating system, so the status it returned is all there is to report.
    StartReturned = u64::from_le_bytes(*b"STARTRET"),
}

impl Notification {
    /// The notification a marker names, or `None` for a value the portal never
    /// writes.
    #[must_use]
    pub const fn from_bits(bits: u64) -> Option<Self> {
        if bits == Self::ExitSucceeded as u64 {
            Some(Self::ExitSucceeded)
        } else if bits == Self::StartReturned as u64 {
            Some(Self::StartReturned)
        } else {
            None
        }
    }
}

/// Why the portal could not be prepared.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum PortalError {
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
    EXIT_SUCCEEDED = const Notification::ExitSucceeded as u64,
    START_RETURNED = const Notification::StartReturned as u64,
);

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
