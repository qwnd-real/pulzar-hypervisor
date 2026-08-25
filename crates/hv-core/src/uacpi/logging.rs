//! Where uACPI's own records go.
//!
//! uACPI formats a line and hands it over already terminated by a newline; the
//! host decides what a line is worth and where it goes. Here that is the `log`
//! facade, which is the same place every other subsystem in this image reports
//! through, so uACPI's account of a machine interleaves with the hypervisor's
//! own rather than arriving on a channel of its own.
//!
//! The two sides grade records the same five ways, so nothing is lost in the
//! translation. What is worth doing is telling uACPI where the floor is: a
//! record it never formats costs nothing, while one it formats for this module
//! to drop costs a `snprintf` into a buffer per line — and on the interpreter's
//! tracing levels that is per operation.

use core::ffi::{CStr, c_char};

use log::{Level as Loudness, log, max_level};
use uacpi_sys::{Level, raw};

/// Tells uACPI not to format what this image would discard.
///
/// Called before uACPI is brought up, and again at no other point: `log`'s
/// maximum level is fixed for the life of this image, whether by the feature
/// that strips records out at compile time or by the backend that installed
/// itself.
pub fn adopt_level() {
    let level = match max_level().to_level() {
        Some(Loudness::Warn) => Level::Warn,
        Some(Loudness::Info) => Level::Info,
        Some(Loudness::Debug) => Level::Trace,
        Some(Loudness::Trace) => Level::Debug,
        // The quietest level uACPI offers answers for both the loudest one `log`
        // has and for nothing being recorded at all. In the second case the
        // records that still arrive are dropped by `log` for free, without a
        // buffer being formatted first.
        Some(Loudness::Error) | None => Level::Error,
    };
    // SAFETY: the setting is a plain field of uACPI's context, writable at any
    // point including before the subsystem is up, and this is called once from
    // the boot processor.
    unsafe { raw::uacpi_context_set_log_level(level.code()) };
}

/// Records one line uACPI formatted.
///
/// # Safety
///
/// Called by uACPI with a pointer to a nul-terminated string it owns for the
/// duration of the call.
pub(super) unsafe extern "C" fn uacpi_kernel_log(level: raw::uacpi_log_level, text: *const c_char) {
    if text.is_null() {
        return;
    }
    // SAFETY: uACPI passes a nul-terminated string that lives for this call, and
    // nothing derived from it escapes: the record is formatted before returning.
    let line = unsafe { CStr::from_ptr(text) };
    let Ok(line) = line.to_str() else {
        log!(Loudness::Warn, "uacpi: a record was not valid utf-8");
        return;
    };
    // uACPI terminates every line itself, and every backend behind `log` adds
    // its own terminator, so the one that arrived would leave a blank line.
    let line = line.trim_end_matches(['\r', '\n']);
    let loudness = match Level::new(level) {
        Level::Error => Loudness::Error,
        Level::Warn => Loudness::Warn,
        Level::Info => Loudness::Info,
        // uACPI's trace is per operation-region access and its debug is per
        // interpreter operation, which is one step louder again than what `log`
        // calls debug.
        Level::Trace => Loudness::Debug,
        Level::Debug => Loudness::Trace,
    };
    log!(loudness, "uacpi: {line}");
}
