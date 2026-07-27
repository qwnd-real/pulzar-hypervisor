//! Reaching a guest's memory.
//!
//! A guest uses three kinds of address and believes in only one of them. What
//! it calls a virtual address its own page tables turn into what it calls a
//! physical address, and that one means nothing to the machine until the nested
//! tables have said what it really is. This crate is both of those steps and
//! the bytes at the end of them.
//!
//! It exists as its own crate because the need is not instruction emulation's
//! alone. Anything acting on a guest's behalf — emulating a device, reading a
//! structure the guest published, writing one back — asks the same question,
//! and asking it in one place is what keeps three subsystems from each growing
//! a page-table walker with its own bugs.
//!
//! # Two views, one underneath the other
//!
//! [`Physical`] is a guest's physical memory: give it an address the guest
//! believes is physical and it translates through the nested tables and reaches
//! the result through the window onto physical memory.
//!
//! [`Linear`] is the same with the guest's own translation on top, which needs
//! to know how this guest is translating — [`Addressing`], which is `CR0`,
//! `CR3`, `CR4`, `EFER` and the segment bases, and which comes from wherever
//! the caller has them. A running guest's are in its control block; firmware's
//! are in the snapshot the loader captured before it changed anything. Neither
//! is this crate's business, which is why nothing here depends on there being a
//! virtual processor at all.
//!
//! # Why this is safe to call
//!
//! Reading arbitrary physical memory is not a safe operation and
//! [`DirectMap`](paging::DirectMap) rightly says so. What makes it safe *here*
//! is a property the nested tables already guarantee: every guest physical
//! address that is the hypervisor's own translates to one shared page of
//! zeroes, and that page is written once when it is created and never
//! referenced again. So an address obtained by translating through those tables
//! cannot point at anything the hypervisor holds a reference to, and the
//! unsafety is discharged rather than passed on.
//!
//! The same property is why a write consults
//! [`Translation::writable`](npt::Translation::writable) instead of assuming.
//! The shared page is not writable, and a write that ignored that would not
//! corrupt one guest page but every page that shadows onto it at once.
//!
//! # What this deliberately does not do
//!
//! It performs no permission check and updates no accessed or dirty bit. An
//! instruction being emulated has already completed its guest-level walk in
//! hardware — that is the only way a fault at the *nested* level was reached at
//! all — so the guest's own tables have already permitted the access and the
//! processor has already done whatever it does to those bits. A check here
//! could only disagree with the processor that just performed the walk.
//!
//! That argument is airtight for the address that faulted and weaker for one
//! the hardware may never have reached: an instruction with two memory operands
//! whose first faults may never have walked its second. Closing that gap means
//! checking the guest's own permissions and raising `#PF` when they are broken,
//! and nothing here raises anything.

#![no_std]

mod addressing;
mod linear;
mod physical;
mod walk;

use npt::NptError;
use paging::PagingError;
use thiserror::Error;

pub use crate::{
    addressing::{Addressing, Mode, Segment},
    linear::Linear,
    physical::{Physical, Written},
};

/// Counts and lengths are `usize` while addresses are `u64`, and the two are
/// converted wherever a slice meets an address. That is lossless exactly while
/// they are the same width, which this crate's only target guarantees.
const _: () = assert!(
    size_of::<usize>() == size_of::<u64>(),
    "this crate assumes 64-bit pointers"
);

/// A length as `u64`.
const fn as_u64(value: usize) -> u64 {
    value as u64
}

/// A length as `usize`.
///
/// Every such conversion goes through here, so the width assumption above is
/// stated once rather than at each call site. `try_from` in its place would be
/// error handling for a state the assertion rules out, on a path that must not
/// fail for reasons that cannot happen.
#[expect(
    clippy::cast_possible_truncation,
    reason = "usize is 64 bits wide on this crate's only target, asserted above"
)]
const fn as_usize(value: u64) -> usize {
    value as usize
}

/// How much of what is left of a copy can be moved before something has to be
/// translated again.
fn reachable(span: u64, left: usize) -> usize {
    as_usize(span.min(as_u64(left)))
}

/// Why a guest's memory could not be reached.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum MemoryError {
    /// Nothing describes this guest physical address yet.
    ///
    /// Not a failure so much as a question this crate cannot answer: a guest's
    /// memory is described as it is touched, and describing more of it needs
    /// the frame allocator, which belongs to whoever owns the chunk. The
    /// address is carried so that the caller can describe it and ask again.
    #[error("guest physical {gpa:#x} is not described by the nested tables")]
    Undescribed {
        /// The address in question.
        gpa: u64,
    },
    /// The guest's own page tables do not describe this address.
    #[error("the guest's tables do not translate {linear:#x}")]
    Untranslated {
        /// The address in question.
        linear: u64,
    },
    /// The address is not canonical, so no walk of it is meaningful.
    #[error("{linear:#x} is not a canonical address")]
    NonCanonical {
        /// The address in question.
        linear: u64,
    },
    /// The guest is translating through five levels of page table, which
    /// nothing here walks — as nothing here runs under, either.
    #[error("the guest is running with five-level paging, which pulzar does not walk")]
    FiveLevelGuest,
    /// A range runs off the end of the physical address space.
    #[error("a {bytes:#x}-byte range at {gpa:#x} leaves the physical address space")]
    Range {
        /// Where the range begins.
        gpa: u64,
        /// How long it is.
        bytes: u64,
    },
    /// The nested tables could not be walked.
    #[error(transparent)]
    Npt(#[from] NptError),
    /// The window onto physical memory does not reach an address.
    #[error(transparent)]
    Paging(#[from] PagingError),
}
