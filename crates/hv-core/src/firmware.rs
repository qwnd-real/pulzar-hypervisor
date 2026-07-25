//! The last thing the hypervisor asks of firmware.
//!
//! One call — unloading the loader's image — made through pointers that arrived
//! in the boot protocol and that all point into the half of the address space
//! the hypervisor is about to drop. Gathering them here keeps the window in
//! which they are valid to a single module that provably stops being used
//! afterwards.
//!
//! Neither pointer is trusted on arrival. A wrong one would be called as a
//! table of function pointers, so both are checked against the signature UEFI
//! puts in every table header.

use core::ptr::NonNull;

use uefi_raw::{
    Handle,
    table::{boot::BootServices, system::SystemTable},
};
use x86_64::instructions::interrupts;

use crate::error::CoreError;

/// `EFI_BOOT_SERVICES_SIGNATURE`, which the table carries in its header.
const BOOT_SERVICES_SIGNATURE: u64 = u64::from_le_bytes(*b"BOOTSERV");

/// Boot services, as far as the hypervisor still needs them.
#[derive(Debug)]
pub struct Firmware {
    boot: NonNull<BootServices>,
}

impl Firmware {
    /// Validates the system table the loader passed and finds boot services in
    /// it.
    ///
    /// # Errors
    ///
    /// [`CoreError::NotATable`] if either table is absent or its header does
    /// not carry the signature UEFI defines for it.
    ///
    /// # Safety
    ///
    /// `system_table` must be the pointer firmware gave the loader, and the
    /// firmware half of the address space must still be mapped.
    pub unsafe fn adopt(system_table: *mut SystemTable) -> Result<Self, CoreError> {
        /// What is being validated, for the error and nothing else.
        const SYSTEM: &str = "system table";
        /// As [`SYSTEM`].
        const BOOT: &str = "boot services table";

        let table = NonNull::new(system_table).ok_or(CoreError::NotATable { table: SYSTEM })?;
        // SAFETY: the caller guarantees this is firmware's own system table and
        // that the address space still maps it. The signature check below is
        // what turns that guarantee into something observable.
        let table = unsafe { table.as_ref() };
        if table.header.signature != SystemTable::SIGNATURE {
            return Err(CoreError::NotATable { table: SYSTEM });
        }
        let boot = NonNull::new(table.boot_services).ok_or(CoreError::NotATable { table: BOOT })?;
        // SAFETY: the system table identified itself, so the boot services
        // pointer in it is firmware's own and mapped alongside it.
        if unsafe { boot.as_ref() }.header.signature != BOOT_SERVICES_SIGNATURE {
            return Err(CoreError::NotATable { table: BOOT });
        }
        Ok(Self { boot })
    }

    /// Unloads a started image, returning every page firmware allocated for it.
    ///
    /// Firmware calls the image's own unload handler as part of this, so the
    /// image is still mapped and executable while the call runs. Wiping it can
    /// therefore only happen afterwards, never instead.
    ///
    /// Interrupts are masked again on the way out, unconditionally. A boot
    /// service returns at the task priority level firmware runs applications
    /// at, and lowering the level to it is defined to enable interrupts — so
    /// this call hands back a processor that will take them, whatever it was
    /// given. Nothing here is ready for that: firmware's interrupt controller
    /// is still programmed as firmware left it, and it is about to be pointed
    /// at a table that belongs to the hypervisor. Masking on return is what
    /// keeps a timer tick from arriving moments later with no visible
    /// connection to the call that permitted it.
    ///
    /// # Errors
    ///
    /// [`CoreError::Firmware`] carrying whatever firmware reported.
    ///
    /// # Safety
    ///
    /// `image` must be a handle firmware created for a started image that
    /// registered an unload handler, nothing may still be executing in that
    /// image, and no pointer into it may be used afterwards.
    pub unsafe fn unload_image(&self, image: Handle) -> Result<(), CoreError> {
        // SAFETY: the table identified itself when this `Firmware` was adopted,
        // so the entry is firmware's own `UnloadImage`; the caller vouches for
        // the handle.
        let status = unsafe { (self.boot.as_ref().unload_image)(image) };
        interrupts::disable();
        if status.is_success() {
            return Ok(());
        }
        Err(CoreError::Firmware {
            operation: "unload the loader's image",
            status,
        })
    }
}
