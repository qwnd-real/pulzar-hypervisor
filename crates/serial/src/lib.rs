//! Serial logging over a 16550 UART for the firmware-side pulzar binaries.
//!
//! [`init`] probes the standard COM ports, configures the first one that
//! responds, and installs a [`log`] logger that writes
//! `[LEVEL module_path] message` lines to it. Lines carry no timestamp:
//! there is no reliable time source this early, and a fabricated one would
//! mislead.
//!
//! The crate is safe to use from any number of cores. The configured UART
//! sits behind a spinlock that is held for one whole log line at a time, so
//! concurrent records come out intact instead of interleaved byte-by-byte.
//! The lock is only ever acquired with interrupts disabled on the acquiring
//! core (and the interrupt flag restored afterwards), so a handler that logs
//! cannot deadlock against a lock its own core was holding when the
//! interrupt arrived. Non-maskable interrupts are the exception, and so is
//! anything else masking cannot hold off — a machine check, a fault: those
//! must not log, and [`emergency`] is what they write through instead, taking
//! no lock at all. Initialization is claimed atomically: a second
//! [`init`] — from the same core or a racing one — fails with a clear error
//! instead of reconfiguring the port underneath whoever is using it.
//! Nothing here allocates.

#![no_std]

mod uart;

use core::{
    fmt::{Arguments, Write},
    sync::atomic::{AtomicBool, AtomicU16, Ordering},
};

use log::{LevelFilter, Log, Metadata, Record};
use spin::Mutex;
use thiserror::Error;
use x86_64::instructions::interrupts;

use crate::uart::Uart;

/// Most verbose level that gets logged: everything in debug builds, `Info`
/// and up in release builds.
pub const MAX_LEVEL: LevelFilter = if cfg!(debug_assertions) {
    LevelFilter::Trace
} else {
    LevelFilter::Info
};

/// The UART all log output funnels through, behind the lock that keeps each
/// core's lines whole. `None` until [`init`] selects a port.
static UART: Mutex<Option<Uart>> = Mutex::new(None);

/// Claimed by the first [`init`] call so later calls fail instead of
/// touching a port that may already be in use.
static INIT_CLAIMED: AtomicBool = AtomicBool::new(false);

/// Base address of the port [`init`] selected, or zero before it did.
///
/// The same fact as [`UART`] holds, kept where it can be read without the
/// lock, which is the whole of what [`emergency`] needs: the port is already
/// configured by the time this is set, so addressing it again takes nothing
/// but the number.
static PORT: AtomicU16 = AtomicU16::new(0);

/// The logger [`init`] installs; it forwards every record to [`UART`].
static LOGGER: SerialLogger = SerialLogger;

/// Why [`init`] failed.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum InitError {
    /// [`init`] already ran (possibly on another core), or something else
    /// installed a global [`log`] logger first. The port keeps whatever
    /// configuration it has; nothing is torn down or reprogrammed.
    #[error("serial logging is already initialized")]
    AlreadyInitialized,
    /// No 16550-compatible UART answered at any standard COM port.
    #[error("no 16550-compatible UART found at any standard COM port")]
    NoUartFound,
}

/// Selects and configures a UART and installs the serial logger.
///
/// Call it once, before anything logs. The call is safe under concurrent
/// double-initialization, but it is one-shot: after any completed attempt —
/// including a failed probe, which retrying cannot cure — later calls
/// report [`InitError::AlreadyInitialized`].
///
/// # Errors
///
/// [`InitError::AlreadyInitialized`] if another `init` call claimed the
/// logger first, [`InitError::NoUartFound`] if no port passed detection.
pub fn init() -> Result<(), InitError> {
    // Relaxed suffices: this flag only elects a single initializer, while
    // the UART itself is published through the mutex and the logger through
    // `log`'s own synchronization.
    if INIT_CLAIMED
        .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
        .is_err()
    {
        return Err(InitError::AlreadyInitialized);
    }
    let uart = Uart::detect().ok_or(InitError::NoUartFound)?;
    PORT.store(uart.base(), Ordering::Relaxed);
    interrupts::without_interrupts(|| *UART.lock() = Some(uart));
    log::set_logger(&LOGGER).map_err(|_| InitError::AlreadyInitialized)?;
    log::set_max_level(MAX_LEVEL);
    Ok(())
}

/// Writes one line to the selected port, taking no lock and allocating
/// nothing.
///
/// For the paths that cannot use [`log`] because of what interrupted them: a
/// non-maskable interrupt, a machine check, a fault whose handler cannot
/// return. Every one of those can arrive while the interrupted code on this
/// very processor holds the logger's lock, and waiting for a lock that only
/// this processor could release is a processor that never comes back. So this
/// path waits for nothing.
///
/// The cost is that a line may interleave with one another processor is
/// writing. That is the right trade where it is used: the alternative to
/// mangled output is no output and a stopped machine.
///
/// Does nothing before [`init`] has selected a port, since there is nowhere to
/// write to.
pub fn emergency(args: Arguments<'_>) {
    let base = PORT.load(Ordering::Relaxed);
    if base == 0 {
        return;
    }
    // The writer cannot fail; an `Err` could only come from a broken `Display`
    // among the arguments, and there is nowhere left to report it from.
    let _ = writeln!(Uart::adopt(base), "{args}");
}

/// Forwards [`log`] records to the locked UART, one whole line per lock
/// acquisition.
struct SerialLogger;

impl Log for SerialLogger {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.level() <= MAX_LEVEL
    }

    fn log(&self, record: &Record<'_>) {
        if !self.enabled(record.metadata()) {
            return;
        }
        // Interrupts stay disabled on this core for the whole line, so an
        // interrupt handler that logs can never spin on a lock this core
        // already holds.
        interrupts::without_interrupts(|| {
            let mut uart = UART.lock();
            if let Some(uart) = uart.as_mut() {
                // Formatting straight into the locked writer keeps the whole
                // line — prefix, message, newline — contiguous on the wire
                // even when several cores log at once. The UART writer itself
                // cannot fail; an `Err` could only come from a broken
                // `Display` impl among the record's arguments, and there is
                // nowhere to report it from inside the logger.
                let _ = writeln!(
                    uart,
                    "[{} {}] {}",
                    record.level(),
                    record.target(),
                    record.args()
                );
            }
        });
    }

    fn flush(&self) {}
}
