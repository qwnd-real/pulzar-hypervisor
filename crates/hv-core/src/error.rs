//! Why the hypervisor could not finish bringing itself up.
//!
//! Every variant ends the same way — the boot stops and the processor halts —
//! so the point of naming them is the serial log, which is the only thing left
//! to diagnose from.

use acpi::AcpiError;
use descriptors::DescriptorError;
use handoff::HandoffError;
use paging::PagingError;
use thiserror::Error;
use uefi_raw::Status;

/// A failure during bring-up.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum CoreError {
    /// The boot protocol the loader passed cannot be used.
    #[error(transparent)]
    Handoff(#[from] HandoffError),
    /// The address space could not be adopted or edited.
    #[error(transparent)]
    Paging(#[from] PagingError),
    /// The firmware tables could not be read.
    #[error(transparent)]
    Acpi(#[from] AcpiError),
    /// The processor's own descriptor tables could not be set up.
    #[error(transparent)]
    Descriptors(#[from] DescriptorError),
    /// The allocator would not take the span reserved for the heap, which can
    /// only mean it is too small to hold the allocator's own bookkeeping.
    #[error("the allocator refused the {bytes:#x}-byte heap at {base:#x}")]
    HeapRefused {
        /// Where the heap was to start.
        base: u64,
        /// How large it was to be.
        bytes: u64,
    },
    /// A boot service refused the operation.
    #[error("could not {operation}: {status}")]
    Firmware {
        /// What was being attempted, as an infinitive.
        operation: &'static str,
        /// What firmware reported.
        status: Status,
    },
    /// A table the handoff pointed at is absent or does not identify itself.
    #[error("the {table} the loader described is not a valid UEFI table")]
    NotATable {
        /// Which table failed to check out.
        table: &'static str,
    },
    /// An address in the handoff is not one this processor can form.
    #[error("{value:#x} is not a usable address")]
    BadAddress {
        /// The offending value.
        value: u64,
    },
    /// Something the hypervisor depends on did not survive the transition to an
    /// address space of its own.
    #[error("{what} stopped working after the firmware half of the address space was dropped")]
    SelfCheckFailed {
        /// What was checked.
        what: &'static str,
    },
}
