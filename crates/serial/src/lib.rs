//! Logging for the firmware-side pulzar binaries, over whichever output the
//! machine has.
//!
//! [`init`] picks a backend, installs a [`log`] logger that writes
//! `[LEVEL module_path] message` lines to it, and never changes its mind
//! afterwards. Two backends exist and the choice between them is not a
//! preference:
//!
//! - QEMU's **debug console**, a single write-only I/O port with no line rate,
//!   no holding register to poll and no divisor to program. A byte costs one
//!   port write.
//! - A **16550 UART** at one of the standard COM ports, which is what a real
//!   machine has, and which clocks a byte out in about 260 µs.
//!
//! The debug console wins wherever it answers, by four orders of magnitude per
//! byte. That gap is the difference between a hypervisor that can describe its
//! own interrupt path and one whose guest starves while it tries: a UART line
//! long enough to be useful takes longer to send than the guest gets to run
//! between two exits.
//!
//! Lines carry no timestamp: there is no reliable time source this early, and a
//! fabricated one would mislead.
//!
//! The crate is safe to use from any number of cores. The chosen backend sits
//! behind a spinlock that is held for one whole log line at a time, so
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

mod debugcon;
mod uart;

use core::{
    fmt::{Arguments, Write},
    sync::atomic::{AtomicBool, AtomicU16, Ordering},
};

use log::{LevelFilter, Log, Metadata, Record};
use spin::Mutex;
use thiserror::Error;
use x86_64::instructions::interrupts;

use crate::{debugcon::Debugcon, uart::Uart};

/// Most verbose level that gets logged.
///
/// `Info` and up regardless of profile. What that costs depends entirely on
/// which backend answered: through the debug console a record is a run of port
/// writes and per-exit logging is affordable, while through a UART every byte
/// is clocked out at the line rate and on a virtualized machine each access is
/// itself a world switch — so the host spends longer reporting an exit than the
/// guest gets to run between two of them.
pub const MAX_LEVEL: LevelFilter = LevelFilter::Info;

/// The output all log records funnel through, behind the lock that keeps each
/// core's lines whole. `None` until [`init`] chooses one.
static OUTPUT: Mutex<Option<Output>> = Mutex::new(None);

/// Claimed by the first [`init`] call so later calls fail instead of
/// touching a port that may already be in use.
static INIT_CLAIMED: AtomicBool = AtomicBool::new(false);

/// Which backend [`init`] chose and, for the UART, where it lives.
///
/// The same fact as [`OUTPUT`] holds, kept where it can be read without the
/// lock, which is the whole of what [`emergency`] needs: whatever was chosen is
/// already configured by the time this is set, so addressing it again takes
/// nothing but this number.
static CHOSEN: AtomicU16 = AtomicU16::new(NONE_CHOSEN);

/// What [`CHOSEN`] holds before [`init`] has chosen anything.
///
/// Zero is not a COM port base and is not the debug console's marker, so it
/// cannot be mistaken for either.
const NONE_CHOSEN: u16 = 0;

/// What [`CHOSEN`] holds when the debug console was chosen.
///
/// Its port number, which is outside the range of COM port bases and so tells
/// the two backends apart on its own.
const DEBUGCON_CHOSEN: u16 = 0xE9;

/// The logger [`init`] installs; it forwards every record to [`OUTPUT`].
static LOGGER: SerialLogger = SerialLogger;

/// Where log records are written.
enum Output {
    /// QEMU's debug console, which is one port and no line rate.
    Debugcon(Debugcon),
    /// A 16550 UART at one of the standard COM ports.
    Uart(Uart),
}

impl Write for Output {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        match self {
            Self::Debugcon(debugcon) => debugcon.write_str(s),
            Self::Uart(uart) => uart.write_str(s),
        }
    }
}

/// Why [`init`] failed.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum InitError {
    /// [`init`] already ran (possibly on another core), or something else
    /// installed a global [`log`] logger first. The port keeps whatever
    /// configuration it has; nothing is torn down or reprogrammed.
    #[error("serial logging is already initialized")]
    AlreadyInitialized,
    /// Neither the debug console nor a 16550-compatible UART answered.
    #[error("no debug console and no 16550-compatible UART found")]
    NoUartFound,
}

/// Chooses an output and installs the logger.
///
/// Call it once, before anything logs. The call is safe under concurrent
/// double-initialization, but it is one-shot: after any completed attempt —
/// including a failed probe, which retrying cannot cure — later calls
/// report [`InitError::AlreadyInitialized`].
///
/// The debug console is tested for first and taken whenever it answers, because
/// nothing about a UART is better and everything about its speed is worse. Only
/// a machine without one falls through to probing the COM ports.
///
/// # Errors
///
/// [`InitError::AlreadyInitialized`] if another `init` call claimed the
/// logger first, [`InitError::NoUartFound`] if neither backend answered.
pub fn init() -> Result<(), InitError> {
    // Relaxed suffices: this flag only elects a single initializer, while
    // the output is published through the mutex and the logger through
    // `log`'s own synchronization.
    if INIT_CLAIMED
        .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
        .is_err()
    {
        return Err(InitError::AlreadyInitialized);
    }
    let (output, chosen) = if let Some(debugcon) = Debugcon::detect() {
        (Output::Debugcon(debugcon), DEBUGCON_CHOSEN)
    } else {
        let uart = Uart::detect().ok_or(InitError::NoUartFound)?;
        let base = uart.base();
        (Output::Uart(uart), base)
    };
    CHOSEN.store(chosen, Ordering::Relaxed);
    interrupts::without_interrupts(|| *OUTPUT.lock() = Some(output));
    log::set_logger(&LOGGER).map_err(|_| InitError::AlreadyInitialized)?;
    log::set_max_level(MAX_LEVEL);
    Ok(())
}

/// Writes one line to the chosen output, taking no lock and allocating
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
/// Does nothing before [`init`] has chosen an output, since there is nowhere to
/// write to.
pub fn emergency(args: Arguments<'_>) {
    // The writers cannot fail; an `Err` could only come from a broken `Display`
    // among the arguments, and there is nowhere left to report it from.
    match CHOSEN.load(Ordering::Relaxed) {
        NONE_CHOSEN => {}
        DEBUGCON_CHOSEN => {
            let _ = writeln!(Debugcon::adopt(), "{args}");
        }
        base => {
            let _ = writeln!(Uart::adopt(base), "{args}");
        }
    }
}

/// Forwards [`log`] records to the locked output, one whole line per lock
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
            let mut output = OUTPUT.lock();
            if let Some(output) = output.as_mut() {
                // Formatting straight into the locked writer keeps the whole
                // line — prefix, message, newline — contiguous on the wire
                // even when several cores log at once. The writer itself
                // cannot fail; an `Err` could only come from a broken
                // `Display` impl among the record's arguments, and there is
                // nowhere to report it from inside the logger.
                let _ = writeln!(
                    output,
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
