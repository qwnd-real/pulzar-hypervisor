//! Hypervisor image for pulzar — currently a stub.
//!
//! The real hypervisor will be a freestanding image that `hv-loader` locates
//! on the EFI system partition (staged as `\pulzar.efi`), maps, and enters
//! directly; firmware never runs it. This stub stands in for that image so
//! the boot-media layout and the build/staging pipeline are exercised end to
//! end before the loader learns to load it. Until then it is inert: if
//! something does start it as a UEFI application (e.g. a user launches it
//! from the UEFI shell), it explains itself and declines to boot.

#![no_main]
#![no_std]

use uefi::prelude::*;

/// UEFI entry point, reached only if the image is started as an application
/// instead of being loaded by `hv-loader`.
///
/// Panicking here is acceptable: this runs under firmware boot services
/// before any guest exists, and aborting is the correct response to a broken
/// invariant.
#[entry]
fn main() -> Status {
    // Serial logging comes up before anything else so even this refusal
    // path is observable. Firmware calls this entry on the bootstrap
    // processor with application processors still parked, so no second
    // core can race this call — and `serial::init` tolerates concurrent
    // callers if that ever changes.
    serial::init().expect("serial logging must initialize");
    uefi::helpers::init().expect("UEFI allocator must initialize");
    uefi::println!("pulzar.efi is the hypervisor image; boot hv-loader (BOOTX64.EFI) instead");
    Status::UNSUPPORTED
}
