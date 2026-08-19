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
//! metadata. The second page carries the values it reads, a copy of the
//! original boot-services table, and the state that keeps the first
//! `ExitBootServices` attempt distinct from retries. No host virtual address
//! appears anywhere in it, because the guest runs under firmware's page tables
//! and not the hypervisor's.
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

#[cfg(test)]
extern crate std;

use core::mem::offset_of;

use handoff::Handoff;
use paging::{DirectMap, as_usize, chunk};
use thiserror::Error;
use uefi_raw::table::{Header, boot::BootServices, system::SystemTable};
use x86_64::PhysAddr;

/// `EFI_BOOT_SERVICES_SIGNATURE`, as stored in the common table header.
const BOOT_SERVICES_SIGNATURE: u64 = u64::from_le_bytes(*b"BOOTSERV");

/// The portal placed in the chunk for the initial guest processor.
#[derive(Debug)]
pub struct Portal {
    base: PhysAddr,
    window: DirectMap,
    parameters: PhysAddr,
    boot_services: PhysAddr,
    boot_services_size: usize,
    loader_image_base: PhysAddr,
    loader_image_size: usize,
}

/// The loader image's disposition after its first EBS hook call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u64)]
enum LoaderState {
    Pending = 0,
    Handled = 1,
    Skipped = 2,
}

impl LoaderState {
    /// Decodes a value the host wrote into the portal parameter page.
    const fn from_bits(bits: u64) -> Option<Self> {
        match bits {
            value if value == Self::Pending as u64 => Some(Self::Pending),
            value if value == Self::Handled as u64 => Some(Self::Handled),
            value if value == Self::Skipped as u64 => Some(Self::Skipped),
            _ => None,
        }
    }
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
        let room = as_usize(chunk::PORTAL_SIZE);
        // Both pages, not just the first byte of the first: the whole
        // reservation is cleared and written below.
        let at = window
            .bytes_ptr::<u8>(base, room)
            .map_err(|_| PortalError::Unreachable {
                phys: base.as_u64(),
            })?;
        let blob = blob();
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
        Self {
            base,
            window,
            parameters: base + chunk::FRAME_SIZE,
            boot_services: PhysAddr::zero(),
            boot_services_size: 0,
            loader_image_base: PhysAddr::zero(),
            loader_image_size: 0,
        }
        .fill(handoff)
    }

    /// Guest physical address of the first portal instruction.
    #[must_use]
    pub const fn entry(&self) -> PhysAddr {
        self.base
    }

    /// Guest physical address of the parameter and snapshot page.
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

    /// Wipes the unloaded loader image through the host's direct map.
    ///
    /// This is called only after firmware's `UnloadImage` returned success.
    /// The portal and all host code run from memory outside this range, so the
    /// image can be cleared without invalidating the current instruction
    /// stream.
    ///
    /// # Errors
    ///
    /// [`PortalError::Unreachable`] if the direct map cannot reach every loader
    /// image byte.
    pub fn wipe_loader(&self) -> Result<(), PortalError> {
        let image = self
            .window
            .bytes_ptr::<u8>(self.loader_image_base, self.loader_image_size)
            .map_err(|_| PortalError::Unreachable {
                phys: self.loader_image_base.as_u64(),
            })?;
        // SAFETY: firmware has returned successfully from `UnloadImage`, so no
        // processor can still execute this image; the direct map reaches the
        // whole page-aligned range and no Rust reference aliases its bytes.
        unsafe { image.as_ptr().write_bytes(0, self.loader_image_size) };
        Ok(())
    }

    /// Marks the loader as handled after the first EBS hook.
    ///
    /// The portal's guest-visible copy of the handle is cleared together with
    /// the state, which makes a second unload impossible even if firmware calls
    /// the wrapper more than once.
    ///
    /// # Errors
    ///
    /// [`PortalError::InvalidLoaderState`] if the portal was not waiting for
    /// this first transition.
    pub fn mark_loader_handled(&self) -> Result<(), PortalError> {
        self.transition_loader(LoaderState::Handled)
    }

    /// Marks loader deletion as skipped after firmware rejected `UnloadImage`.
    ///
    /// The EBS call is still allowed to continue because the caller treats
    /// image deletion as best effort.
    ///
    /// # Errors
    ///
    /// [`PortalError::InvalidLoaderState`] if the portal was not waiting for
    /// this first transition.
    pub fn mark_loader_skipped(&self) -> Result<(), PortalError> {
        self.transition_loader(LoaderState::Skipped)
    }

    /// Restores the original `BootServices` table exactly as firmware published
    /// it.
    ///
    /// The copy includes the original function pointers, header fields, and
    /// CRC. It is written after successful EBS, before the guest resumes from
    /// the notification, and uses the direct map because boot services can no
    /// longer be called at that point.
    ///
    /// # Errors
    ///
    /// [`PortalError::Unreachable`] if either the saved copy or the live table
    /// falls outside the direct map.
    pub fn restore_boot_services(&self) -> Result<(), PortalError> {
        let saved = self
            .window
            .bytes_ptr::<u8>(
                self.parameters + as_u64(BOOT_SERVICES_COPY_OFFSET),
                self.boot_services_size,
            )
            .map_err(|_| PortalError::Unreachable {
                phys: (self.parameters + as_u64(BOOT_SERVICES_COPY_OFFSET)).as_u64(),
            })?;
        let live = self
            .window
            .bytes_ptr::<u8>(self.boot_services, self.boot_services_size)
            .map_err(|_| PortalError::Unreachable {
                phys: self.boot_services.as_u64(),
            })?;
        // SAFETY: the saved bytes occupy portal-owned metadata and the live table
        // is the firmware table that was patched by this portal. EBS succeeded,
        // so no boot-service implementation can concurrently mutate either one.
        unsafe {
            live.as_ptr()
                .copy_from_nonoverlapping(saved.as_ptr(), self.boot_services_size);
        };
        Ok(())
    }

    /// Writes the parameter page from firmware's live tables.
    fn fill(mut self, handoff: &Handoff) -> Result<Self, PortalError> {
        let system = self
            .window
            .ptr::<SystemTable>(PhysAddr::new_truncate(handoff.system_table as u64))
            .map_err(|_| PortalError::Unreachable {
                phys: handoff.system_table as u64,
            })?;
        // SAFETY: the handoff pointer is firmware's live system table and the
        // direct-map pointer above reaches that physical table.
        let boot_services = unsafe { (*system.as_ptr()).boot_services };
        let boot_services_phys = PhysAddr::new_truncate(boot_services as usize as u64);
        let boot_services = self
            .window
            .ptr::<BootServices>(boot_services_phys)
            .map_err(|_| PortalError::MissingBootServices)?;
        // SAFETY: firmware's boot-services table remains initialized and
        // readable because the guest has not called ExitBootServices yet.
        let header = unsafe { (*boot_services.as_ptr()).header };
        let table_size = boot_services_table_size(header)?;
        if handoff.loader_image_handle.is_null() {
            return Err(PortalError::MissingLoaderHandle);
        }
        if !handoff.loader_image_base.is_multiple_of(chunk::FRAME_SIZE)
            || handoff.loader_image_size == 0
            || !handoff.loader_image_size.is_multiple_of(chunk::FRAME_SIZE)
        {
            return Err(PortalError::InvalidLoaderImage {
                base: handoff.loader_image_base,
                size: handoff.loader_image_size,
            });
        }
        let source = self
            .window
            .bytes_ptr::<u8>(boot_services_phys, table_size)
            .map_err(|_| PortalError::Unreachable {
                phys: boot_services_phys.as_u64(),
            })?;
        let saved = self
            .window
            .bytes_ptr::<u8>(
                self.parameters + as_u64(BOOT_SERVICES_COPY_OFFSET),
                table_size,
            )
            .map_err(|_| PortalError::Unreachable {
                phys: (self.parameters + as_u64(BOOT_SERVICES_COPY_OFFSET)).as_u64(),
            })?;
        // SAFETY: the source is firmware's live table and the destination is the
        // portal's private parameter page; the two ranges cannot overlap because
        // the chunk was allocated before this table was copied.
        unsafe {
            saved
                .as_ptr()
                .copy_from_nonoverlapping(source.as_ptr(), table_size);
        };

        let parameters = self
            .window
            .ptr::<Parameters>(self.parameters)
            .map_err(|_| PortalError::Unreachable {
                phys: self.parameters.as_u64(),
            })?;
        // SAFETY: the second portal page belongs only to this portal and has
        // `Parameters` alignment because every chunk page is frame-aligned.
        unsafe {
            parameters.as_ptr().write(Parameters {
                system_table: handoff.system_table as u64,
                guest_image_handle: handoff.guest_image_handle as u64,
                loader_image_handle: handoff.loader_image_handle as u64,
                original_exit_boot_services: (*boot_services.as_ptr()).exit_boot_services as usize
                    as u64,
                loader_state: LoaderState::Pending as u64,
            });
        }
        self.boot_services = boot_services_phys;
        self.boot_services_size = table_size;
        self.loader_image_base = PhysAddr::new_truncate(handoff.loader_image_base);
        self.loader_image_size = as_usize(handoff.loader_image_size);
        Ok(self)
    }

    /// Transitions the loader state and clears its guest-visible handle.
    fn transition_loader(&self, state: LoaderState) -> Result<(), PortalError> {
        let parameters = self
            .window
            .ptr::<Parameters>(self.parameters)
            .map_err(|_| PortalError::Unreachable {
                phys: self.parameters.as_u64(),
            })?;
        // SAFETY: the parameter page is portal-owned, and the guest is stopped
        // in the intercepted VMMCALL while the host performs this transition.
        let current = unsafe { (*parameters.as_ptr()).loader_state };
        if LoaderState::from_bits(current) != Some(LoaderState::Pending) {
            return Err(PortalError::InvalidLoaderState { state: current });
        }
        // SAFETY: as above; this store publishes the state before the guest can
        // execute another EBS wrapper invocation.
        unsafe {
            (*parameters.as_ptr()).loader_image_handle = 0;
            (*parameters.as_ptr()).loader_state = state as u64;
        }
        Ok(())
    }
}

/// What the portal tells the host, in the register its `vmmcall` carries.
///
/// Four moments in the portal protocol: loader deletion, deletion skipped,
/// successful EBS, and a boot manager that returned instead of taking the
/// machine. Recognizable byte strings rather than small integers, so that a
/// `vmmcall` some other part of the guest makes cannot be read as one of these.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u64)]
pub enum Notification {
    /// `UnloadImage` returned success for the loader image.
    LoaderUnloaded = u64::from_le_bytes(*b"LDR FREE"),
    /// `UnloadImage` failed, so the host leaves the loader image in place.
    LoaderSkipped = u64::from_le_bytes(*b"LDR SKIP"),
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
        match bits {
            value if value == Self::LoaderUnloaded as u64 => Some(Self::LoaderUnloaded),
            value if value == Self::LoaderSkipped as u64 => Some(Self::LoaderSkipped),
            value if value == Self::ExitSucceeded as u64 => Some(Self::ExitSucceeded),
            value if value == Self::StartReturned as u64 => Some(Self::StartReturned),
            _ => None,
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
    /// The handoff did not contain the firmware handle needed by `UnloadImage`.
    #[error("the loader image handle is null")]
    MissingLoaderHandle,
    /// The live table did not identify itself as UEFI boot services or kept a
    /// nonzero value in the header field the specification reserves.
    #[error("invalid UEFI boot-services header (signature {signature:#x}, reserved {reserved:#x})")]
    InvalidBootServicesHeader {
        /// Signature firmware published in the table header.
        signature: u64,
        /// Value firmware published in the reserved header field.
        reserved: u32,
    },
    /// The live `BootServices` table is shorter than the fields this portal
    /// reads.
    #[error("the UEFI boot-services table is {size} bytes, shorter than {minimum}")]
    InvalidBootServicesSize {
        /// Bytes firmware reported.
        size: usize,
        /// Minimum bytes needed by this portal.
        minimum: usize,
    },
    /// The original `BootServices` table cannot fit in the portal parameter
    /// page.
    #[error("the {bytes}-byte UEFI boot-services table exceeds its {room}-byte snapshot room")]
    BootServicesTooLarge {
        /// Bytes in the live table.
        bytes: usize,
        /// Bytes available after the assembly parameters.
        room: usize,
    },
    /// The handoff did not describe a nonempty, page-aligned loader image.
    #[error("the loader image range {base:#x}+{size:#x} is not nonempty and page-aligned")]
    InvalidLoaderImage {
        /// Physical base firmware assigned to the image.
        base: u64,
        /// Page-rounded image size firmware reported.
        size: u64,
    },
    /// The host received a loader transition after it had already been handled.
    #[error("the loader state was already handled ({state:#x})")]
    InvalidLoaderState {
        /// State word found in the parameter page.
        state: u64,
    },
}

/// Arguments the assembly reads from the second portal page.
#[repr(C)]
struct Parameters {
    system_table: u64,
    guest_image_handle: u64,
    loader_image_handle: u64,
    original_exit_boot_services: u64,
    loader_state: u64,
}

/// Offset in the parameter page where the exact `BootServices` snapshot begins.
const BOOT_SERVICES_COPY_OFFSET: usize = size_of::<Parameters>();

/// Bytes available for the variable-sized `BootServices` snapshot.
const BOOT_SERVICES_COPY_CAPACITY: usize = as_usize(chunk::FRAME_SIZE) - BOOT_SERVICES_COPY_OFFSET;

/// Validates the live table header and returns the exact byte count to save.
fn boot_services_table_size(header: Header) -> Result<usize, PortalError> {
    if header.signature != BOOT_SERVICES_SIGNATURE || header.reserved != 0 {
        return Err(PortalError::InvalidBootServicesHeader {
            signature: header.signature,
            reserved: header.reserved,
        });
    }
    let size = header.size as usize;
    if size < size_of::<BootServices>() {
        return Err(PortalError::InvalidBootServicesSize {
            size,
            minimum: size_of::<BootServices>(),
        });
    }
    if size > BOOT_SERVICES_COPY_CAPACITY {
        return Err(PortalError::BootServicesTooLarge {
            bytes: size,
            room: BOOT_SERVICES_COPY_CAPACITY,
        });
    }
    Ok(size)
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
    LOADER_IMAGE_HANDLE = const offset_of!(Parameters, loader_image_handle),
    ORIGINAL_EXIT_BOOT_SERVICES = const offset_of!(Parameters, original_exit_boot_services),
    LOADER_STATE = const offset_of!(Parameters, loader_state),
    SYSTEM_BOOT_SERVICES = const offset_of!(SystemTable, boot_services),
    BOOT_EXIT_BOOT_SERVICES = const offset_of!(BootServices, exit_boot_services),
    BOOT_UNLOAD_IMAGE = const offset_of!(BootServices, unload_image),
    BOOT_START_IMAGE = const offset_of!(BootServices, start_image),
    BOOT_CALCULATE_CRC32 = const offset_of!(BootServices, calculate_crc32),
    HEADER_SIZE = const offset_of!(Header, size),
    HEADER_CRC = const offset_of!(Header, crc),
    EXIT_SUCCEEDED = const Notification::ExitSucceeded as u64,
    START_RETURNED = const Notification::StartReturned as u64,
    LOADER_UNLOADED = const Notification::LoaderUnloaded as u64,
    LOADER_SKIPPED = const Notification::LoaderSkipped as u64,
    LOADER_PENDING = const LoaderState::Pending as u64,
);

/// Converts a portal offset to the address arithmetic's `u64` representation.
const fn as_u64(value: usize) -> u64 {
    value as u64
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

#[cfg(test)]
mod tests {
    use core::mem::size_of;
    use std::boxed::Box;

    use x86_64::VirtAddr;

    use super::*;

    const TEST_PAGES: usize = 4;
    const TEST_BYTES: usize = as_usize(chunk::FRAME_SIZE) * TEST_PAGES;
    const LIVE_TABLE_OFFSET: usize = as_usize(chunk::FRAME_SIZE);
    const LOADER_OFFSET: usize = as_usize(chunk::FRAME_SIZE) * 2;

    #[repr(C, align(4096))]
    struct TestMemory {
        bytes: [u8; TEST_BYTES],
    }

    #[test]
    fn notifications_decode_only_defined_markers() {
        for notification in [
            Notification::LoaderUnloaded,
            Notification::LoaderSkipped,
            Notification::ExitSucceeded,
            Notification::StartReturned,
        ] {
            assert_eq!(
                Notification::from_bits(notification as u64),
                Some(notification)
            );
        }
        assert_eq!(Notification::from_bits(0), None);
    }

    #[test]
    fn boot_services_header_must_be_valid_and_fit() {
        let valid = Header {
            signature: BOOT_SERVICES_SIGNATURE,
            size: u32::try_from(size_of::<BootServices>()).unwrap(),
            ..Header::default()
        };
        assert_eq!(
            boot_services_table_size(valid),
            Ok(size_of::<BootServices>())
        );

        let invalid_signature = Header {
            signature: 0,
            ..valid
        };
        assert_eq!(
            boot_services_table_size(invalid_signature),
            Err(PortalError::InvalidBootServicesHeader {
                signature: 0,
                reserved: 0,
            })
        );

        let too_short = Header {
            size: u32::try_from(size_of::<BootServices>() - 1).unwrap(),
            ..valid
        };
        assert_eq!(
            boot_services_table_size(too_short),
            Err(PortalError::InvalidBootServicesSize {
                size: size_of::<BootServices>() - 1,
                minimum: size_of::<BootServices>(),
            })
        );

        let too_large = Header {
            size: u32::try_from(BOOT_SERVICES_COPY_CAPACITY + 1).unwrap(),
            ..valid
        };
        assert_eq!(
            boot_services_table_size(too_large),
            Err(PortalError::BootServicesTooLarge {
                bytes: BOOT_SERVICES_COPY_CAPACITY + 1,
                room: BOOT_SERVICES_COPY_CAPACITY,
            })
        );
    }

    #[test]
    fn loader_state_transitions_once_and_clears_the_handle() {
        let (portal, _memory) = fixture();
        portal.mark_loader_handled().unwrap();
        let first = parameters(&portal);
        assert_eq!(first.loader_image_handle, 0);
        assert_eq!(first.loader_state, LoaderState::Handled as u64);
        assert_eq!(
            portal.mark_loader_skipped(),
            Err(PortalError::InvalidLoaderState {
                state: LoaderState::Handled as u64,
            })
        );

        let (portal, _memory) = fixture();
        portal.mark_loader_skipped().unwrap();
        let second = parameters(&portal);
        assert_eq!(second.loader_image_handle, 0);
        assert_eq!(second.loader_state, LoaderState::Skipped as u64);
    }

    #[test]
    fn restoring_boot_services_copies_the_exact_saved_range() {
        let (mut portal, mut memory) = fixture();
        portal.boot_services_size = size_of::<BootServices>();
        memory.bytes.fill(0xcc);
        initialize_parameters(&portal);

        let saved =
            BOOT_SERVICES_COPY_OFFSET..BOOT_SERVICES_COPY_OFFSET + portal.boot_services_size;
        let live = LIVE_TABLE_OFFSET..LIVE_TABLE_OFFSET + portal.boot_services_size;
        let mut value = 0_u8;
        for byte in &mut memory.bytes[saved.clone()] {
            *byte = value;
            value = value.wrapping_add(1);
        }
        let expected = memory.bytes[saved].to_vec();

        portal.restore_boot_services().unwrap();

        assert_eq!(&memory.bytes[live.clone()], expected.as_slice());
        assert_eq!(memory.bytes[live.start - 1], 0xcc);
        assert_eq!(memory.bytes[live.end], 0xcc);
    }

    #[test]
    fn loader_wipe_is_all_or_nothing() {
        let (portal, mut memory) = fixture();
        let loader = LOADER_OFFSET..LOADER_OFFSET + as_usize(chunk::FRAME_SIZE);
        memory.bytes[loader.clone()].fill(0xa5);
        portal.wipe_loader().unwrap();
        assert!(memory.bytes[loader].iter().all(|byte| *byte == 0));

        let (mut portal, mut memory) = fixture();
        let loader = LOADER_OFFSET..TEST_BYTES;
        memory.bytes[loader.clone()].fill(0xa5);
        portal.loader_image_size = TEST_BYTES - LOADER_OFFSET + as_usize(chunk::FRAME_SIZE);
        assert!(matches!(
            portal.wipe_loader(),
            Err(PortalError::Unreachable { .. })
        ));
        assert!(memory.bytes[loader].iter().all(|byte| *byte == 0xa5));
    }

    fn fixture() -> (Portal, Box<TestMemory>) {
        let mut memory = Box::new(TestMemory {
            bytes: [0; TEST_BYTES],
        });
        let window = DirectMap::new(
            VirtAddr::from_ptr(memory.bytes.as_mut_ptr()),
            TEST_BYTES as u64,
        )
        .unwrap();
        let portal = Portal {
            base: PhysAddr::zero(),
            window,
            parameters: PhysAddr::zero(),
            boot_services: PhysAddr::new(LIVE_TABLE_OFFSET as u64),
            boot_services_size: size_of::<BootServices>(),
            loader_image_base: PhysAddr::new(LOADER_OFFSET as u64),
            loader_image_size: as_usize(chunk::FRAME_SIZE),
        };
        initialize_parameters(&portal);
        (portal, memory)
    }

    fn initialize_parameters(portal: &Portal) {
        let parameters = portal.window.ptr::<Parameters>(portal.parameters).unwrap();
        // SAFETY: the aligned test allocation backs the complete direct-map
        // window and the parameter structure occupies its first bytes alone.
        unsafe {
            parameters.as_ptr().write(Parameters {
                system_table: 0,
                guest_image_handle: 0,
                loader_image_handle: 1,
                original_exit_boot_services: 0,
                loader_state: LoaderState::Pending as u64,
            });
        }
    }

    fn parameters(portal: &Portal) -> Parameters {
        let parameters = portal.window.ptr::<Parameters>(portal.parameters).unwrap();
        // SAFETY: `fixture` initialized the structure and the portal transition
        // only updates fields within it while the allocation remains alive.
        unsafe { parameters.as_ptr().read() }
    }
}
