//! Logging for the firmware-side pulzar binaries, over whichever output the
//! machine has.
//!
//! [`init`] picks a backend, installs a [`log`] logger that writes
//! `[HH:MM:SS mmm] [LEVEL module_path] message` lines to it, and never changes
//! its mind afterwards. The timestamp is uptime from the installed clock, or
//! zero before that clock exists. Three backends exist and the choice between
//! them is not a preference:
//!
//! - QEMU's **debug console**, a single write-only I/O port with no line rate,
//!   no holding register to poll and no divisor to program. A byte costs one
//!   port write.
//! - A **16550 UART** at one of the standard COM ports, which is what a real
//!   machine with a header has, and which clocks a byte out in about 260 µs.
//! - The **UEFI frame buffer** firmware drew its console through, drawn to as
//!   pixels. It exists behind the `efifb` feature, because a screen is not
//!   always there to take and is a guest-facing device once it is: where the
//!   feature put a backend in the image, the screen has first priority —
//!   [`offer_screen`] takes the output away from whichever port answered first
//!   — and [`retire_screen`] gives it up before anything else draws.
//!
//! Ports are chosen by probing and the screen by being offered, so their
//! precedence is temporal: `init` answers with the best port it can find,
//! and a compiled-in screen displaces it the moment its description arrives.
//! What never happens is a port keeping output that a usable screen was
//! offered for.
//!
//! The costs are three orders of magnitude apart, and that gap is the
//! difference between a hypervisor that can describe its own interrupt path
//! and one whose guest starves while it tries: a UART line long enough to be
//! useful takes longer to send than the guest gets to run between two exits,
//! and a scrolled framebuffer line costs thousands of uncached stores. Which
//! is why the screen retires before the guest runs rather than fighting it
//! for the display, and why `quiet` exists for machines whose log nobody reads.
//!
//! # Machines with no output, and machines that must not spend time on it
//!
//! Both are ordinary and neither is a failure.
//!
//! A machine with no port at all is one nothing can be reported from at the
//! moment [`init`] runs — which is a reason to run it silently rather than a
//! reason not to run it: [`init`] answers [`InitError::NoUartFound`] and
//! callers carry on, after which [`log`] discards every record until a
//! backend appears. With the `efifb` feature compiled in, that is what
//! [`offer_screen`] is for: called with the handoff's screen description once
//! its bytes are reachable, it takes the empty output slot, installs the
//! logger if [`init`] never got that far, and every record after it lands on
//! the display.
//!
//! A machine with a port is the harder case, because *having* one is not the
//! same as anybody listening to it. Every desktop board with a serial header
//! has a working 16550 behind it whether or not a cable is plugged in, and the
//! transmitter clocks its bytes out at the line rate regardless — so on such a
//! machine the log costs exactly as much as it would if somebody were reading
//! it, and buys nothing. The `quiet` feature is for that machine: it takes
//! [`MAX_LEVEL`] to `Off` and, because that is `log`'s *static* maximum, every
//! record in every crate of the build is compiled out rather than filtered.
//! Nothing is formatted, no lock is taken, and the arguments are never
//! evaluated — which matters here beyond the bytes, because several of this
//! workspace's trace lines read a real controller register or query a
//! controller to build their arguments.
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
#[cfg(feature = "efifb")]
mod efifb;
mod uart;

#[cfg(feature = "efifb")]
use core::sync::atomic::AtomicU64;
use core::{
    fmt::{Arguments, Display, Formatter, Write},
    sync::atomic::{AtomicBool, AtomicU16, Ordering},
    time::Duration,
};

use log::{LevelFilter, Log, Metadata, Record};
use spin::Mutex;
#[cfg(feature = "efifb")]
use spin::Once;
use thiserror::Error;
use x86_64::instructions::interrupts;

#[cfg(feature = "efifb")]
use crate::efifb::{Canvas, Cursor, Efifb};
use crate::{debugcon::Debugcon, uart::Uart};

/// Most verbose level that gets logged.
///
/// `Info` and up, unless the `quiet` feature is on, in which case nothing is
/// logged at all and every record is compiled out rather than filtered at run
/// time.
///
/// What logging costs depends entirely on which backend answered: through the
/// debug console a record is a run of port writes and per-exit logging is
/// affordable, while through a UART every byte is clocked out at the line rate
/// and on a virtualized machine each access is itself a world switch — so the
/// host spends longer reporting an exit than the guest gets to run between two
/// of them. A real machine almost always has a UART and almost never has a
/// debug console, which is what `quiet` is for.
pub const MAX_LEVEL: LevelFilter = if cfg!(feature = "quiet") {
    LevelFilter::Off
} else {
    LevelFilter::Info
};

/// The `quiet` feature has to reach [`log`]'s *static* maximum level and not
/// merely this crate's, because that is the one the macros are expanded
/// against. A run-time filter would still format the record, still take the
/// lock, and — the part that matters most here — still evaluate the arguments,
/// which on several paths in this workspace means reading a real controller
/// register or querying a controller to build a line nothing will print.
///
/// So if the feature ever stops forwarding to `log`, this is what says so,
/// rather than a machine that is mysteriously slow again.
#[cfg(feature = "quiet")]
const _: () = assert!(
    log::STATIC_MAX_LEVEL as usize == LevelFilter::Off as usize,
    "the quiet feature must switch log's static maximum level off, not this crate's"
);

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

/// What [`CHOSEN`] holds when the frame buffer was chosen.
///
/// No port number is free to mean this, so it is a marker: no COM base and
/// not the debug console's, in a range nothing decodes.
const SCREEN_CHOSEN: u16 = u16::MAX;

/// Where the screen backend writes, once [`offer_screen`] has attached it.
///
/// The address is whatever the attaching side mapped — or identified, under
/// the loader — and never changes while it stands.
#[cfg(feature = "efifb")]
static SCREEN_ADDRESS: AtomicU64 = AtomicU64::new(0);

/// The attached screen's geometry, published last of the parts an emergency
/// reader needs.
///
/// A `Some` here answers for [`SCREEN_ADDRESS`] too: both are written before
/// this one fills, and the reading path comes through it first. On x86 the
/// stores cannot pass each other anyway; the ordering through the `Once` is
/// what makes that argument true rather than merely true on this machine.
#[cfg(feature = "efifb")]
static SCREEN_CANVAS: Once<Canvas> = Once::new();

/// Seconds in one minute.
const SECONDS_PER_MINUTE: u64 = 60;

/// Minutes in one hour.
const MINUTES_PER_HOUR: u64 = 60;

/// Seconds in one hour.
const SECONDS_PER_HOUR: u64 = MINUTES_PER_HOUR * SECONDS_PER_MINUTE;

/// The logger [`init`] installs; it forwards every record to [`OUTPUT`].
static LOGGER: SerialLogger = SerialLogger;

/// Where log records are written.
enum Output {
    /// QEMU's debug console, which is one port and no line rate.
    Debugcon(Debugcon),
    /// A 16550 UART at one of the standard COM ports.
    Uart(Uart),
    /// The frame buffer firmware drew its console through.
    #[cfg(feature = "efifb")]
    Screen(Efifb),
}

impl Write for Output {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        match self {
            Self::Debugcon(debugcon) => debugcon.write_str(s),
            Self::Uart(uart) => uart.write_str(s),
            #[cfg(feature = "efifb")]
            Self::Screen(screen) => screen.write_str(s),
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
    /// Neither the debug console nor a 16550-compatible UART answered. With
    /// the `efifb` feature compiled in this is not final: a screen attached
    /// afterwards through [`offer_screen`] takes the empty slot.
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
        base if base == SCREEN_CHOSEN => {
            // From the top-left corner rather than from wherever the locked
            // half got to: the cursor is not published per line, and an
            // emergency line over the oldest one is worth exactly as much as
            // one interleaved into a fresh position.
            #[cfg(feature = "efifb")]
            if let Some(canvas) = SCREEN_CANVAS.get() {
                let mut screen = Efifb {
                    address: SCREEN_ADDRESS.load(Ordering::Relaxed),
                    canvas: *canvas,
                    cursor: Cursor::default(),
                };
                let _ = writeln!(screen, "{args}");
            }
        }
        base => {
            let _ = writeln!(Uart::adopt(base), "{args}");
        }
    }
}

/// Offers the frame buffer the output, displacing whichever port answered
/// first.
///
/// Called with the handoff's screen description and an address its bytes are
/// reachable through — under the loader that is the physical base itself,
/// firmware's identity map still standing; under the hypervisor image it is
/// whatever the caller mapped the aperture at. The description decides
/// whether there is anything to take: an unusable or absent screen changes
/// nothing and answers `false`.
///
/// While the `efifb` feature is compiled in, the screen has first priority,
/// always. `init` runs before any screen description exists, so a port wins
/// the output at that moment by default — but many boards decode a COM
/// address whether or not anything is wired to the header, so "a port
/// answered" does not mean "somebody can read it". This call takes the output
/// back from whatever port holds it, permanently for this boot;
/// [`retire_screen`] does not give it back.
///
/// Answers `false` when nothing was taken: the description names no usable
/// screen, or this screen was already attached. Without the `efifb` feature
/// it compiles to that answer and nothing else.
#[expect(
    clippy::must_use_candidate,
    reason = "a declined offer is a documented ordinary outcome, not something callers must handle"
)]
pub fn offer_screen(screen: &handoff::Framebuffer, address: u64) -> bool {
    #[cfg(feature = "efifb")]
    {
        let Some(candidate) = Efifb::describe(screen, address) else {
            return false;
        };
        interrupts::without_interrupts(|| {
            let mut output = OUTPUT.lock();
            // A second offer changes nothing; anything else that answered
            // first is displaced. Priority here is temporal rather than
            // probed: `init` runs before the screen description exists, so
            // the ports take the output first and give it up the moment a
            // usable screen is offered — which is what an opt-in screen
            // backend means.
            if matches!(*output, Some(Output::Screen(_))) {
                return false;
            }
            // The logger first, because records start flowing the moment the
            // output is published; then the parts the emergency path reads,
            // canvas last, since a reader that finds it finds everything.
            let _ = log::set_logger(&LOGGER);
            log::set_max_level(MAX_LEVEL);
            SCREEN_ADDRESS.store(candidate.address, Ordering::Relaxed);
            let canvas = candidate.canvas;
            SCREEN_CANVAS.call_once(|| canvas);
            *output = Some(Output::Screen(candidate));
            CHOSEN.store(SCREEN_CHOSEN, Ordering::Relaxed);
            true
        })
    }
    #[cfg(not(feature = "efifb"))]
    {
        let _ = (screen, address);
        false
    }
}

/// Takes the screen back out of the output, for the moment it stops being
/// ours.
///
/// After this every record is discarded again, which is the point: the guest
/// that starts drawing owns the display, and a line of host text written
/// underneath it is neither visible to anyone nor free to produce. Ports are
/// not touched — they cost nothing while idle and never belong to the guest.
/// A screen that was never attached, or already retired, changes nothing;
/// without the `efifb` feature there is nothing to do and no way to ask.
pub fn retire_screen() {
    #[cfg(feature = "efifb")]
    if CHOSEN.swap(NONE_CHOSEN, Ordering::Relaxed) == SCREEN_CHOSEN {
        interrupts::without_interrupts(|| *OUTPUT.lock() = None);
    }
}

/// Forwards [`log`] records to the locked output, one whole line per lock
/// acquisition.
struct SerialLogger;

/// A log record's elapsed-time prefix, in nanoseconds since clock startup.
struct Uptime(Option<u64>);

impl Uptime {
    /// Reads the installed clock, or retains the pre-clock placeholder.
    fn now() -> Self {
        Self(clock::now().map(clock::Instant::nanos))
    }
}

impl Display for Uptime {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> core::fmt::Result {
        let Some(nanos) = self.0 else {
            return formatter.write_str("00:00:00 000");
        };
        let elapsed = Duration::from_nanos(nanos);
        let total_seconds = elapsed.as_secs();
        let hours = total_seconds / SECONDS_PER_HOUR;
        let minutes = total_seconds / SECONDS_PER_MINUTE % MINUTES_PER_HOUR;
        let seconds = total_seconds % SECONDS_PER_MINUTE;

        write!(
            formatter,
            "{hours:02}:{minutes:02}:{seconds:02} {:03}",
            elapsed.subsec_millis()
        )
    }
}

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
                    "[{}] [{} {}] {}",
                    Uptime::now(),
                    record.level(),
                    record.target(),
                    record.args()
                );
            }
        });
    }

    fn flush(&self) {}
}

#[cfg(test)]
mod tests {
    //! Uptime prefix formatting.

    extern crate std;

    use super::Uptime;

    #[test]
    fn uptime_uses_zero_before_clock_startup() {
        assert_eq!(std::format!("{}", Uptime(None)), "00:00:00 000");
    }

    #[test]
    fn uptime_formats_elapsed_milliseconds() {
        let nanos = 97_445_006_000_000;

        assert_eq!(std::format!("{}", Uptime(Some(nanos))), "27:04:05 006");
    }
}
