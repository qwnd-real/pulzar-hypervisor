//! The port address space, as uACPI reaches it.
//!
//! On this architecture the `SystemIO` address space is the `in` and `out`
//! family of instructions, and uACPI reaches it the way it reaches anything
//! else it cannot address directly: it claims a range once, then reads and
//! writes at offsets within it. Firmware's fixed registers live there — the
//! power management event and control blocks the FADT describes — and so does
//! most of what bytecode touches on an older platform.
//!
//! A claim exists to be checked against. Ports are sixteen bits wide on this
//! architecture while uACPI names them with sixty-four, and an offset is not
//! checked by anything else, so a range that does not fit the port space is
//! refused at the claim and an access past the end of one is refused where it
//! is made. Without that, a bad offset in firmware's own bytecode would be a
//! write to whichever unrelated device happened to answer at the port it landed
//! on.
//!
//! # Widths are not interchangeable
//!
//! A four-byte access is one instruction against one port, never four one-byte
//! accesses against consecutive ports: hardware decodes the width, and the two
//! are different transactions to a device. So each width has its own answer,
//! and each of them checks that the whole of what it is about to move fits
//! inside the claim.

use log::warn;
use uacpi_sys::{Status, raw};
use x86_64::instructions::port::{PortRead, PortWrite};

use crate::uacpi::handle::{borrow, displace, place};

/// One past the highest port this architecture has.
const PORTS: u64 = 0x1_0000;

/// How wide a port access of one value is.
///
/// The `in` and `out` family moves one, two or four bytes and nothing else, so
/// the widths are stated rather than derived from a type's size — which would
/// have to be narrowed to fit a port number, on a value that is never large
/// enough for the narrowing to be real.
trait Width {
    /// Bytes one access of this width moves.
    const BYTES: u16;
}

impl Width for u8 {
    const BYTES: u16 = 1;
}

impl Width for u16 {
    const BYTES: u16 = 2;
}

impl Width for u32 {
    const BYTES: u16 = 4;
}

/// A claimed run of ports.
struct Range {
    /// The first port of the run.
    base: u16,
    /// How many ports it covers, which may be the whole space and so needs a
    /// word wider than a port number.
    len: u32,
}

impl Range {
    /// The port `offset` bytes into the run, if a `width`-byte access there is
    /// still inside it.
    fn port(&self, offset: raw::uacpi_size, width: u16) -> Option<u16> {
        let offset = u32::try_from(offset).ok()?;
        let last = offset.checked_add(u32::from(width))?;
        if last > self.len {
            return None;
        }
        u16::try_from(u32::from(self.base) + offset).ok()
    }
}

/// Claims the ports at `[base, base + len)`.
///
/// # Safety
///
/// Called by uACPI with storage for one handle.
pub(super) unsafe extern "C" fn uacpi_kernel_io_map(
    base: raw::uacpi_io_addr,
    len: raw::uacpi_size,
    out: *mut raw::uacpi_handle,
) -> raw::uacpi_status {
    if out.is_null() {
        return Status::INVALID_ARGUMENT.code();
    }
    let bytes = u64::try_from(len).unwrap_or(u64::MAX);
    let claim = u16::try_from(base)
        .ok()
        .zip(u32::try_from(bytes).ok())
        .filter(|_| bytes != 0 && base.saturating_add(bytes) <= PORTS);
    let Some((base, len)) = claim else {
        warn!("core: uacpi asked for {bytes:#x} ports at {base:#x}, which is not a port range");
        return Status::INVALID_ARGUMENT.code();
    };
    let handle = place(Range { base, len });
    if handle.is_null() {
        return Status::OUT_OF_MEMORY.code();
    }
    // SAFETY: the caller supplies storage for one handle, checked non-null above.
    unsafe { out.write(handle) };
    Status::OK.code()
}

/// Gives back a claim.
///
/// # Safety
///
/// Called by uACPI with a handle [`uacpi_kernel_io_map`] returned, which it
/// will not use again.
pub(super) unsafe extern "C" fn uacpi_kernel_io_unmap(handle: raw::uacpi_handle) {
    // SAFETY: the caller vouches that the handle named a range from this module
    // and is not used again.
    unsafe { displace::<Range>(handle) };
}

/// Reads a byte from a claimed range.
///
/// # Safety
///
/// Called by uACPI with a handle it holds and storage for the value.
pub(super) unsafe extern "C" fn uacpi_kernel_io_read8(
    handle: raw::uacpi_handle,
    offset: raw::uacpi_size,
    out: *mut raw::uacpi_u8,
) -> raw::uacpi_status {
    // SAFETY: the caller vouches for the handle and the storage.
    unsafe { read(handle, offset, out) }
}

/// Reads a word from a claimed range.
///
/// # Safety
///
/// As [`uacpi_kernel_io_read8`].
pub(super) unsafe extern "C" fn uacpi_kernel_io_read16(
    handle: raw::uacpi_handle,
    offset: raw::uacpi_size,
    out: *mut raw::uacpi_u16,
) -> raw::uacpi_status {
    // SAFETY: the caller vouches for the handle and the storage.
    unsafe { read(handle, offset, out) }
}

/// Reads a doubleword from a claimed range.
///
/// # Safety
///
/// As [`uacpi_kernel_io_read8`].
pub(super) unsafe extern "C" fn uacpi_kernel_io_read32(
    handle: raw::uacpi_handle,
    offset: raw::uacpi_size,
    out: *mut raw::uacpi_u32,
) -> raw::uacpi_status {
    // SAFETY: the caller vouches for the handle and the storage.
    unsafe { read(handle, offset, out) }
}

/// Writes a byte to a claimed range.
///
/// # Safety
///
/// Called by uACPI with a handle it holds. The value is firmware's own bytecode
/// writing a platform register, which is what this hypervisor is reading the
/// platform through.
pub(super) unsafe extern "C" fn uacpi_kernel_io_write8(
    handle: raw::uacpi_handle,
    offset: raw::uacpi_size,
    value: raw::uacpi_u8,
) -> raw::uacpi_status {
    // SAFETY: the caller vouches for the handle and the value.
    unsafe { write(handle, offset, value) }
}

/// Writes a word to a claimed range.
///
/// # Safety
///
/// As [`uacpi_kernel_io_write8`].
pub(super) unsafe extern "C" fn uacpi_kernel_io_write16(
    handle: raw::uacpi_handle,
    offset: raw::uacpi_size,
    value: raw::uacpi_u16,
) -> raw::uacpi_status {
    // SAFETY: the caller vouches for the handle and the value.
    unsafe { write(handle, offset, value) }
}

/// Writes a doubleword to a claimed range.
///
/// # Safety
///
/// As [`uacpi_kernel_io_write8`].
pub(super) unsafe extern "C" fn uacpi_kernel_io_write32(
    handle: raw::uacpi_handle,
    offset: raw::uacpi_size,
    value: raw::uacpi_u32,
) -> raw::uacpi_status {
    // SAFETY: the caller vouches for the handle and the value.
    unsafe { write(handle, offset, value) }
}

/// Reads one value of any port width, checked against the claim.
///
/// # Safety
///
/// The handle must name a live [`Range`], and `out` must be storage for one `T`
/// or null.
unsafe fn read<T: PortRead + Width>(
    handle: raw::uacpi_handle,
    offset: raw::uacpi_size,
    out: *mut T,
) -> raw::uacpi_status {
    if out.is_null() {
        return Status::INVALID_ARGUMENT.code();
    }
    // SAFETY: the caller vouches that the handle names a live range.
    let claimed = unsafe { borrow::<Range>(handle) };
    let Some(port) = claimed.and_then(|range| range.port(offset, T::BYTES)) else {
        return refuse(handle, offset, T::BYTES);
    };
    // SAFETY: the port is inside a range uACPI claimed for exactly this, and the
    // width is the one the access was asked for. Reading a port has whatever
    // effect the device behind it defines, which is what firmware's own
    // description of the platform says should happen here.
    let value = unsafe { T::read_from_port(port) };
    // SAFETY: the caller supplies storage for one `T`, checked non-null above.
    unsafe { out.write(value) };
    Status::OK.code()
}

/// Writes one value of any port width, checked against the claim.
///
/// # Safety
///
/// The handle must name a live [`Range`], and the value must be one the
/// register behind the port accepts.
unsafe fn write<T: PortWrite + Width>(
    handle: raw::uacpi_handle,
    offset: raw::uacpi_size,
    value: T,
) -> raw::uacpi_status {
    // SAFETY: the caller vouches that the handle names a live range.
    let claimed = unsafe { borrow::<Range>(handle) };
    let Some(port) = claimed.and_then(|range| range.port(offset, T::BYTES)) else {
        return refuse(handle, offset, T::BYTES);
    };
    // SAFETY: as in `read`, and the caller vouches for the value.
    unsafe { T::write_to_port(port, value) };
    Status::OK.code()
}

/// Says why an access was not made, and answers uACPI with it.
fn refuse(handle: raw::uacpi_handle, offset: raw::uacpi_size, width: u16) -> raw::uacpi_status {
    warn!("core: uacpi asked for {width} bytes at offset {offset:#x} of the range {handle:p}");
    Status::INVALID_ARGUMENT.code()
}

/// Each width above is the size of the value it moves, which is what makes the
/// stated numbers a spelling of the type rather than an assumption about it.
const _: () = {
    assert!(<u8 as Width>::BYTES as usize == size_of::<u8>());
    assert!(<u16 as Width>::BYTES as usize == size_of::<u16>());
    assert!(<u32 as Width>::BYTES as usize == size_of::<u32>());
};
