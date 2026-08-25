//! How long things take, and what waiting means here.
//!
//! Every one of these goes to the `clock` crate, which is the machine's one
//! timebase: the timestamp counter, calibrated against whichever of firmware's
//! counters the tables described. That is also why none of it is available
//! during early table access — the clock is calibrated against a table, so the
//! tables come first.
//!
//! # Sleeping without a scheduler
//!
//! uACPI distinguishes stalling, which spins, from sleeping, which may yield.
//! Nothing here can yield: uACPI is brought up before any other processor is
//! started and there is no second thread of execution to yield to. So both
//! spin, and a sleep is a stall of a thousand times the length. The difference
//! the distinction exists for — not holding a processor for milliseconds at a
//! time — is real, and the cost is paid during bring-up on a machine with
//! nothing else to run.

use core::sync::atomic::{AtomicBool, Ordering};

use log::warn;
use uacpi_sys::raw;

/// Microseconds in a millisecond, for turning a sleep into a stall.
const MICROS_PER_MILLI: u64 = 1000;

/// The timeout uACPI spells "wait for as long as it takes".
const FOREVER: raw::uacpi_u16 = 0xFFFF;

/// Spins until `ready` answers true or `timeout` milliseconds have passed,
/// reporting which happened.
///
/// A timeout of zero is one attempt and no waiting, which is what uACPI asks a
/// non-blocking acquisition to be. [`FOREVER`] never gives up.
///
/// A machine with no clock yet cannot measure a finite timeout. Rather than
/// invent one, a finite wait there is the single attempt a zero timeout would
/// have been — reported once, because a caller in that position is either
/// uncontended, in which case nothing was lost, or waiting for something that
/// only another processor could provide, in which case no length of wait would
/// have helped.
pub fn wait_until(timeout: raw::uacpi_u16, mut ready: impl FnMut() -> bool) -> bool {
    if ready() {
        return true;
    }
    if timeout == 0 {
        return false;
    }
    if timeout == FOREVER {
        while !ready() {
            core::hint::spin_loop();
        }
        return true;
    }
    let Some(deadline) = clock::now() else {
        if !UNTIMED.swap(true, Ordering::Relaxed) {
            warn!("core: uacpi waited for {timeout} ms before the clock was up; it did not wait");
        }
        return false;
    };
    let deadline = deadline.nanos().saturating_add(nanos(timeout));
    loop {
        if ready() {
            return true;
        }
        // Asked after the attempt, so a deadline that has already passed still
        // gets the one attempt every wait is owed.
        if clock::now().is_none_or(|now| now.nanos() >= deadline) {
            return false;
        }
        core::hint::spin_loop();
    }
}

/// Reads a strictly monotonic count of nanoseconds since the clock was
/// installed.
///
/// Zero until it is, which is before uACPI is brought up far enough to ask.
///
/// # Safety
///
/// Called by uACPI.
pub(super) unsafe extern "C" fn uacpi_kernel_get_nanoseconds_since_boot() -> raw::uacpi_u64 {
    clock::now().map_or(0, clock::Instant::nanos)
}

/// Spins for `micros` microseconds.
///
/// # Safety
///
/// Called by uACPI.
pub(super) unsafe extern "C" fn uacpi_kernel_stall(micros: raw::uacpi_u8) {
    delay(u64::from(micros));
}

/// Waits for `millis` milliseconds, by spinning for as long.
///
/// # Safety
///
/// Called by uACPI.
pub(super) unsafe extern "C" fn uacpi_kernel_sleep(millis: raw::uacpi_u64) {
    delay(millis.saturating_mul(MICROS_PER_MILLI));
}

/// Spins for `micros` microseconds, or reports that it could not.
fn delay(micros: u64) {
    if clock::sleep_micros(micros).is_err() && !UNTIMED.swap(true, Ordering::Relaxed) {
        warn!("core: uacpi asked to wait {micros} us before the clock was up; it did not wait");
    }
}

/// Nanoseconds in a millisecond timeout.
fn nanos(millis: raw::uacpi_u16) -> u64 {
    u64::from(millis) * MICROS_PER_MILLI * MICROS_PER_MILLI
}

/// Whether a wait has already been refused for want of a clock, so that a
/// machine in that state says so once rather than per call.
static UNTIMED: AtomicBool = AtomicBool::new(false);
