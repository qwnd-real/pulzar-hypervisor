//! What can stop the loader, and how a firmware status becomes one of them.
//!
//! Every failure ends the boot with its reason on the serial log, so each
//! variant carries enough to identify the step that produced it. That matters
//! most for firmware: a bare `EFI_NOT_FOUND` says nothing about which of a
//! dozen calls returned it, which is why [`Context`] pairs every status with
//! the operation that was attempted.

use core::fmt::Debug;

use paging::PagingError;
use thiserror::Error;
use uefi::Status;

use crate::image::ImageError;

/// Why the loader could not hand control to the hypervisor.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum LoaderError {
    /// A firmware call failed.
    #[error("failed to {operation}: {status}")]
    Firmware {
        /// What was being attempted, phrased to complete "failed to …".
        operation: &'static str,
        /// The status firmware returned.
        status: Status,
    },
    /// The hypervisor image is not one this loader can place.
    #[error("the hypervisor image is unusable: {0}")]
    Image(#[from] ImageError),
    /// The address space, or the memory backing it, could not be set up.
    #[error("the address space could not be built: {0}")]
    Paging(#[from] PagingError),
    /// A read returned fewer bytes than were asked for. Firmware reports this
    /// as success, so it has to be checked rather than trusted.
    #[error("reading {wanted:#x} bytes at {offset:#x} of the image returned {got:#x}")]
    ShortRead {
        /// Offset in the file the read started at.
        offset: u64,
        /// Bytes requested.
        wanted: usize,
        /// Bytes delivered.
        got: usize,
    },
    /// The volume produced something other than a regular file for the image's
    /// path — a directory, most plausibly.
    #[error("the hypervisor image is not a regular file")]
    NotARegularFile,
    /// Firmware's memory map does not fit the space the chunk sets aside for
    /// it. Truncating it would hand the hypervisor a map that silently omits
    /// memory, so the boot stops instead.
    #[error("firmware reported {entries} memory descriptors, more than the {capacity} that fit")]
    MemoryMapTooLarge {
        /// Descriptors firmware reported.
        entries: usize,
        /// Descriptors the chunk's metadata region holds.
        capacity: usize,
    },
}

/// Names the operation a firmware call was performing, so its status becomes a
/// [`LoaderError::Firmware`] that says what actually went wrong.
pub trait Context<T> {
    /// Attaches `operation`, phrased to complete the sentence "failed to …".
    fn context(self, operation: &'static str) -> Result<T, LoaderError>;
}

impl<T, D: Debug> Context<T> for uefi::Result<T, D> {
    fn context(self, operation: &'static str) -> Result<T, LoaderError> {
        self.map_err(|error| LoaderError::Firmware {
            operation,
            status: error.status(),
        })
    }
}
