//! Raw bindings to uACPI, built from the submodule pinned in this workspace.
//!
//! uACPI is the ACPI implementation pulzar reads firmware's tables through and
//! evaluates its bytecode with. This crate is the boundary and nothing more: it
//! compiles the C, declares what the C exposes, and stops. It implements none
//! of the callbacks uACPI needs from its host — those are the host's, because
//! they are the only part of the interface that has to know what machine it is
//! running on. See [`kernel`] for what a host owes.
//!
//! # What is here
//!
//! [`raw`] is the generated surface: every type, constant and function uACPI's
//! public headers declare, spelled exactly as C spells them. Calling into it is
//! `unsafe` and stays that way — a `-sys` crate that wrapped things safely
//! would be guessing at invariants only its caller knows.
//!
//! The two things on top of it exist because both sides of that boundary need
//! them and neither should write them twice: [`Status`], which turns a returned
//! code into a `Result`, and [`Level`], which says how loud a log record from
//! inside uACPI was.
//!
//! # Targets
//!
//! The C is compiled for whatever cargo is building for. On the firmware side
//! that is `x86_64-unknown-uefi`; a plain host build also works, and exists so
//! that the crates layered on this one can run their tests natively.

#![no_std]

use core::{
    ffi::CStr,
    fmt::{self, Display, Formatter},
};

pub mod kernel;

/// The generated declarations for uACPI's public headers.
///
/// Everything in here is machine-written from the headers of the pinned
/// submodule at build time, so it describes exactly the library that gets
/// linked and cannot drift from it.
///
/// Nothing in this module is documented, named or shaped the way Rust would
/// have it, because all of it is C's names and C's shapes carried across
/// unchanged — which is the point. The lints that would say so are turned off
/// for this module alone, and this module holds nothing but the generated file:
/// the code is not editable, so a warning about it could only be silenced, and
/// silencing it here is what keeps every hand-written line in the workspace
/// under the full set.
#[expect(
    missing_docs,
    non_camel_case_types,
    non_upper_case_globals,
    unsafe_op_in_unsafe_fn,
    clippy::all,
    clippy::pedantic,
    reason = "machine-generated declarations that cannot be edited to comply"
)]
pub mod raw {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

/// The widths the generated declarations came out with, stated in Rust's terms.
///
/// `crates/uacpi-sys/widths.c` asserts the same numbers where the C compiler
/// decides them. Both are needed and neither is redundant: the C assertions
/// catch a data model that sizes a type differently from what is assumed here,
/// and these catch a generated declaration naming a Rust type that does not
/// match — which is the same mismatch seen from the only other side it can be
/// seen from. A declaration generated as `c_ulong` for the firmware target
/// passes the C assertions and fails these.
const _: () = {
    assert!(size_of::<raw::uacpi_status>() == 4);
    assert!(size_of::<raw::uacpi_log_level>() == 4);
    assert!(size_of::<raw::uacpi_phys_addr>() == 8);
    assert!(size_of::<raw::uacpi_io_addr>() == 8);
    assert!(size_of::<raw::uacpi_virt_addr>() == 8);
    assert!(size_of::<raw::uacpi_size>() == 8);
    assert!(size_of::<raw::uacpi_handle>() == 8);
    assert!(size_of::<raw::uacpi_cpu_flags>() == 8);
    assert!(size_of::<raw::uacpi_interrupt_state>() == 8);
    assert!(size_of::<raw::uacpi_thread_id>() == 8);
};

/// What a uACPI call reported.
///
/// A wrapper rather than the bare code, so that a status can never be mistaken
/// for a count or a handle, and so that the one place a status becomes a
/// `Result` is this type rather than every call site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Status(raw::uacpi_status);

impl Status {
    /// Everything went as asked.
    pub const OK: Self = Self(raw::UACPI_STATUS_OK);

    /// A host callback declining because pulzar does not implement it.
    pub const UNIMPLEMENTED: Self = Self(raw::UACPI_STATUS_UNIMPLEMENTED);

    /// A host callback declining because the machine has no such thing.
    pub const NOT_FOUND: Self = Self(raw::UACPI_STATUS_NOT_FOUND);

    /// A host callback declining because it was asked for something impossible.
    pub const INVALID_ARGUMENT: Self = Self(raw::UACPI_STATUS_INVALID_ARGUMENT);

    /// A host callback declining because there was no memory for it.
    pub const OUT_OF_MEMORY: Self = Self(raw::UACPI_STATUS_OUT_OF_MEMORY);

    /// A host callback declining because a range could not be mapped.
    pub const MAPPING_FAILED: Self = Self(raw::UACPI_STATUS_MAPPING_FAILED);

    /// A wait that ran out of time, which is not an error for every caller.
    pub const TIMEOUT: Self = Self(raw::UACPI_STATUS_TIMEOUT);

    /// A host callback declining because pulzar will not do this.
    pub const DENIED: Self = Self(raw::UACPI_STATUS_DENIED);

    /// A host callback declining for a reason that is the host's own bug.
    pub const INTERNAL_ERROR: Self = Self(raw::UACPI_STATUS_INTERNAL_ERROR);

    /// Wraps a code that came back from uACPI.
    #[must_use]
    pub const fn new(code: raw::uacpi_status) -> Self {
        Self(code)
    }

    /// The code, for handing back across the boundary.
    #[must_use]
    pub const fn code(self) -> raw::uacpi_status {
        self.0
    }

    /// Whether the call succeeded.
    #[must_use]
    pub const fn is_ok(self) -> bool {
        self.0 == raw::UACPI_STATUS_OK
    }

    /// Turns the code into a `Result`, so a failure cannot be walked past.
    ///
    /// # Errors
    ///
    /// The status itself, whenever it is not [`Status::OK`].
    pub const fn ok(self) -> Result<(), Self> {
        if self.is_ok() { Ok(()) } else { Err(self) }
    }

    /// What uACPI calls this code.
    ///
    /// uACPI answers for every value, including ones it does not define, so
    /// there is no case in which this has nothing to say.
    #[must_use]
    pub fn text(self) -> &'static str {
        // SAFETY: the function takes a status by value and returns a pointer to
        // one of its own string literals, with no state involved, so it is
        // callable at any time and the result is valid for the whole program.
        let text = unsafe { raw::uacpi_status_to_string(self.0) };
        if text.is_null() {
            return "unnamed uACPI status";
        }
        // SAFETY: a non-null return is one of uACPI's own literals, which is
        // nul-terminated and immutable for as long as the image runs.
        unsafe { CStr::from_ptr(text) }
            .to_str()
            .unwrap_or("unprintable uACPI status")
    }
}

impl Display for Status {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} ({:#x})", self.text(), self.0)
    }
}

impl core::error::Error for Status {}

/// How loud a log record from inside uACPI is.
///
/// uACPI's levels and `log`'s are the same five in the same order, but this
/// crate does not depend on `log` — the host does the mapping, and this is what
/// gives it something total to map from. A level uACPI has not defined is
/// reported as the loudest rather than dropped: a record that cannot be graded
/// is the last one that should go missing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    /// Only what threatens the ability to initialize or run at all.
    Error,
    /// Recoverable trouble, and aborts that do not matter much.
    Warn,
    /// State changes and progress, and nothing more.
    Info,
    /// Every operation region access, with some context around it.
    Trace,
    /// Every operation and micro-operation the interpreter processes.
    Debug,
}

impl Level {
    /// Grades a level that came across the boundary.
    #[must_use]
    pub const fn new(level: raw::uacpi_log_level) -> Self {
        match level {
            raw::UACPI_LOG_WARN => Self::Warn,
            raw::UACPI_LOG_INFO => Self::Info,
            raw::UACPI_LOG_TRACE => Self::Trace,
            raw::UACPI_LOG_DEBUG => Self::Debug,
            _ => Self::Error,
        }
    }

    /// The code uACPI spells this level as.
    #[must_use]
    pub const fn code(self) -> raw::uacpi_log_level {
        match self {
            Self::Error => raw::UACPI_LOG_ERROR,
            Self::Warn => raw::UACPI_LOG_WARN,
            Self::Info => raw::UACPI_LOG_INFO,
            Self::Trace => raw::UACPI_LOG_TRACE,
            Self::Debug => raw::UACPI_LOG_DEBUG,
        }
    }
}
