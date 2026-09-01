//! The machine's own choices, recorded where every part of the hypervisor can
//! read them.
//!
//! Some decisions belong to no subsystem: they are the owner's answers to
//! what this machine should be, and the parts of the hypervisor that act on
//! them should be reading one value rather than each carrying its own copy.
//! Today there is one — the seed the identify transforms are keyed by — and
//! it is compiled in, because a machine with no readable storage of its own
//! yet has nowhere else for it to live.
//!
//! # What is deliberately not here
//!
//! No accessor per value, no loading, no precedence rules. The day
//! configuration is read off a disk, this crate grows the reader and the
//! disk's layout — and the callers keep calling the same functions, which is
//! the reason the answers are behind functions rather than constants.

#![no_std]

use spoof::Seed;

/// The seed a machine's identify responses are spoofed with.
///
/// Same seed, same spoofed identities on every boot; a different seed,
/// different identities with no relation between them. It is compiled in
/// for now, so every machine running this build shares one — the disk-loaded
/// configuration that gives each machine its own is the crate's next shape,
/// and this function is the seam it arrives through.
#[must_use]
pub fn serial_seed() -> Seed {
    Seed::new([
        0x4d, 0x17, 0xc2, 0x83, 0x13, 0xf0, 0x9a, 0x5d, 0x0b, 0x76, 0xe1, 0x38, 0xa9, 0x54, 0x2c,
        0xdf,
    ])
}
