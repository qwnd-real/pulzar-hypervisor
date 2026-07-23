//! First-stage UEFI loader for the pulzar hypervisor.
//!
//! Firmware loads this application as a PE/COFF image and calls its entry
//! point with boot services still active. Its eventual job is to locate the
//! hypervisor image, map it, and transfer control to it. For now it only
//! proves the toolchain end-to-end by printing a greeting.

#![no_main]
#![no_std]

use uefi::prelude::*;

/// UEFI entry point, invoked by firmware after image load.
///
/// Panicking here is acceptable: we are in early boot, guest
/// execution has not begun, and aborting the boot is the correct response to
/// a broken invariant.
#[entry]
fn main() -> Status {
    uefi::helpers::init().expect("UEFI logger/allocator must initialize");
    uefi::println!("pulzar hv-loader: Hello, World!");
    Status::SUCCESS
}
