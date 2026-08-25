//! A device's configuration space, as uACPI reaches it.
//!
//! Firmware's bytecode reaches configuration space constantly: it is where a
//! device's interrupt routing is described, where a root complex's own
//! registers live, and where a great deal of platform behaviour is switched on.
//! uACPI opens one function at a time by its segment, bus, device and function
//! numbers, then reads and writes at offsets within it.
//!
//! None of the mechanism is here. The `pci` crate already reaches configuration
//! space both ways a processor can — through the memory aperture firmware
//! described, and through the legacy pair of ports for a bus no aperture covers
//! — and it already decides which of the two an address needs, serializes the
//! port path against other processors, and refuses an extended register on a
//! bus that cannot carry one. So a handle here is a function that survey
//! already found, and each access is one call.
//!
//! # A device that is not there
//!
//! uACPI asks to be told, and handles it by standing in a device whose every
//! register reads as all ones. That is what a great deal of bytecode probes for
//! to decide whether a device exists, so answering "not found" is not a failure
//! but the correct description of a machine — and it is the answer for anything
//! this module cannot reach, because a function the survey did not find is a
//! function nothing here can address.

use log::warn;
use pci::{Address, Bus, Function, Offset, Segment};
use uacpi_sys::{Status, raw};

/// Opens the function at `address` for reading and writing.
///
/// # Safety
///
/// Called by uACPI with storage for one handle.
pub(super) unsafe extern "C" fn uacpi_kernel_pci_device_open(
    address: raw::uacpi_pci_address,
    out: *mut raw::uacpi_handle,
) -> raw::uacpi_status {
    if out.is_null() {
        return Status::INVALID_ARGUMENT.code();
    }
    let Some(found) = locate(address) else {
        return Status::NOT_FOUND.code();
    };
    // SAFETY: the caller supplies storage for one handle, checked non-null above.
    // The function is part of the survey, which lives as long as the image, so the
    // handle stays valid however long uACPI keeps it.
    unsafe { out.write(core::ptr::from_ref(found).cast_mut().cast()) };
    Status::OK.code()
}

/// Closes what [`uacpi_kernel_pci_device_open`] opened, which takes nothing.
///
/// The handle names a function of the machine's survey rather than anything
/// this module allocated, so there is nothing to give back.
///
/// # Safety
///
/// Called by uACPI with a handle it opened.
pub(super) unsafe extern "C" fn uacpi_kernel_pci_device_close(_handle: raw::uacpi_handle) {}

/// Reads a byte of an open function's configuration space.
///
/// # Safety
///
/// Called by uACPI with a handle it opened and storage for the value.
pub(super) unsafe extern "C" fn uacpi_kernel_pci_read8(
    handle: raw::uacpi_handle,
    offset: raw::uacpi_size,
    out: *mut raw::uacpi_u8,
) -> raw::uacpi_status {
    // SAFETY: the caller vouches for the handle and the storage.
    unsafe { fetch(handle, offset, out, pci::read_u8) }
}

/// Reads a word of an open function's configuration space.
///
/// # Safety
///
/// As [`uacpi_kernel_pci_read8`].
pub(super) unsafe extern "C" fn uacpi_kernel_pci_read16(
    handle: raw::uacpi_handle,
    offset: raw::uacpi_size,
    out: *mut raw::uacpi_u16,
) -> raw::uacpi_status {
    // SAFETY: the caller vouches for the handle and the storage.
    unsafe { fetch(handle, offset, out, pci::read_u16) }
}

/// Reads a doubleword of an open function's configuration space.
///
/// # Safety
///
/// As [`uacpi_kernel_pci_read8`].
pub(super) unsafe extern "C" fn uacpi_kernel_pci_read32(
    handle: raw::uacpi_handle,
    offset: raw::uacpi_size,
    out: *mut raw::uacpi_u32,
) -> raw::uacpi_status {
    // SAFETY: the caller vouches for the handle and the storage.
    unsafe { fetch(handle, offset, out, pci::read_u32) }
}

/// Writes a byte of an open function's configuration space.
///
/// # Safety
///
/// Called by uACPI with a handle it opened. The value is what firmware's own
/// description of this platform says belongs in the register, which is the only
/// authority there is for writing one.
pub(super) unsafe extern "C" fn uacpi_kernel_pci_write8(
    handle: raw::uacpi_handle,
    offset: raw::uacpi_size,
    value: raw::uacpi_u8,
) -> raw::uacpi_status {
    let write = |function: &Function, at, value| {
        // SAFETY: the caller vouches that the value is one the register accepts.
        unsafe { pci::write_u8(function, at, value) }
    };
    // SAFETY: the caller vouches that the handle names a function of the survey.
    unsafe { store(handle, offset, value, write) }
}

/// Writes a word of an open function's configuration space.
///
/// # Safety
///
/// As [`uacpi_kernel_pci_write8`].
pub(super) unsafe extern "C" fn uacpi_kernel_pci_write16(
    handle: raw::uacpi_handle,
    offset: raw::uacpi_size,
    value: raw::uacpi_u16,
) -> raw::uacpi_status {
    let write = |function: &Function, at, value| {
        // SAFETY: the caller vouches that the value is one the register accepts.
        unsafe { pci::write_u16(function, at, value) }
    };
    // SAFETY: the caller vouches that the handle names a function of the survey.
    unsafe { store(handle, offset, value, write) }
}

/// Writes a doubleword of an open function's configuration space.
///
/// # Safety
///
/// As [`uacpi_kernel_pci_write8`].
pub(super) unsafe extern "C" fn uacpi_kernel_pci_write32(
    handle: raw::uacpi_handle,
    offset: raw::uacpi_size,
    value: raw::uacpi_u32,
) -> raw::uacpi_status {
    let write = |function: &Function, at, value| {
        // SAFETY: the caller vouches that the value is one the register accepts.
        unsafe { pci::write_u32(function, at, value) }
    };
    // SAFETY: the caller vouches that the handle names a function of the survey.
    unsafe { store(handle, offset, value, write) }
}

/// The function uACPI named, or nothing if the machine has no such function.
fn locate(named: raw::uacpi_pci_address) -> Option<&'static Function> {
    let Some(address) = Address::new(
        Segment::new(named.segment),
        Bus::new(named.bus),
        named.device,
        named.function,
    ) else {
        warn!(
            "core: uacpi named device {}, function {}, which no bus has",
            named.device, named.function
        );
        return None;
    };
    match pci::find(address) {
        Ok(found @ Some(_)) => found,
        // Firmware describes a great many devices that a given machine does not
        // populate, so this is ordinary and not worth a record of its own.
        Ok(None) => None,
        Err(error) => {
            warn!("core: uacpi asked for {address} and it could not be reached: {error}");
            None
        }
    }
}

/// Reads one value of any configuration width.
///
/// # Safety
///
/// The handle must be one [`uacpi_kernel_pci_device_open`] returned, and `out`
/// must be storage for one `T` or null.
unsafe fn fetch<T>(
    handle: raw::uacpi_handle,
    offset: raw::uacpi_size,
    out: *mut T,
    read: impl FnOnce(&Function, Offset) -> Result<T, pci::PciError>,
) -> raw::uacpi_status {
    if out.is_null() {
        return Status::INVALID_ARGUMENT.code();
    }
    // SAFETY: the caller vouches that the handle names a function of the survey.
    let Some((function, at)) = (unsafe { opened(handle, offset) }) else {
        return Status::INVALID_ARGUMENT.code();
    };
    match read(function, at) {
        Ok(value) => {
            // SAFETY: the caller supplies storage for one `T`, checked non-null
            // above.
            unsafe { out.write(value) };
            Status::OK.code()
        }
        Err(error) => {
            warn!("core: uacpi could not read offset {offset:#x} of a device: {error}");
            Status::NOT_FOUND.code()
        }
    }
}

/// Writes one value of any configuration width.
///
/// # Safety
///
/// The handle must be one [`uacpi_kernel_pci_device_open`] returned, and the
/// value must be one the register accepts.
unsafe fn store<T>(
    handle: raw::uacpi_handle,
    offset: raw::uacpi_size,
    value: T,
    write: impl FnOnce(&Function, Offset, T) -> Result<(), pci::PciError>,
) -> raw::uacpi_status {
    // SAFETY: the caller vouches that the handle names a function of the survey.
    let Some((function, at)) = (unsafe { opened(handle, offset) }) else {
        return Status::INVALID_ARGUMENT.code();
    };
    match write(function, at, value) {
        Ok(()) => Status::OK.code(),
        Err(error) => {
            warn!("core: uacpi could not write offset {offset:#x} of a device: {error}");
            Status::NOT_FOUND.code()
        }
    }
}

/// The function a handle names and the offset within it, if both are usable.
///
/// # Safety
///
/// The handle must be one [`uacpi_kernel_pci_device_open`] returned.
unsafe fn opened(
    handle: raw::uacpi_handle,
    offset: raw::uacpi_size,
) -> Option<(&'static Function, Offset)> {
    let Ok(offset) = u16::try_from(offset) else {
        warn!("core: uacpi named configuration offset {offset:#x}, which no device has");
        return None;
    };
    // SAFETY: the caller vouches that a non-null handle is the address of a
    // function of the machine's survey, which lives as long as the image.
    let function = unsafe { handle.cast::<Function>().as_ref() }?;
    Some((function, Offset::new(offset)))
}
