//! What this crate does when it cannot go on, and why it does it without help.
//!
//! Every path in here is reached because something the machine did cannot be
//! returned from, and every one of them can have interrupted code that holds a
//! lock this processor would have to run to release. So none of them goes
//! through [`log`]: the report is written straight to the port, taking nothing
//! and waiting for nothing, and then the processor stops.
//!
//! Output written this way can interleave with a line another processor is
//! writing. That is the trade being made — a mangled line is worth more than a
//! silent machine.

use core::fmt::Arguments;

use x86_64::VirtAddr;

use crate::{Interrupt, Vector, halt};

/// Reports an interrupt whose handler must not return, and stops.
pub(crate) fn interrupt(interrupt: &Interrupt) -> ! {
    report(format_args!("{interrupt}"))
}

/// Reports an interrupt that arrived with nothing to answer for it, and stops.
///
/// Not reachable as the code stands — a table of gates is loaded only after the
/// hypervisor has said what an unclaimed interrupt becomes — which is why
/// stopping is right: it means that ordering broke.
pub(crate) fn unanswered(interrupt: &Interrupt) -> ! {
    report(format_args!(
        "nothing has adopted the unclaimed interrupts, and {interrupt} arrived"
    ))
}

/// Reports an interrupt that found its own stack already in use with no level
/// left, and stops.
///
/// The frame the processor pushed is on top of a live one. Nothing here can put
/// that back, so it is said plainly and this processor goes no further.
pub(crate) fn nesting(vector: Vector, stack: VirtAddr) -> ! {
    report(format_args!(
        "{vector} arrived on a stack already in use down to {stack:#x}, with no level left"
    ))
}

/// Reports an event that arrived while the live global descriptor table was not
/// one this crate built, and stops.
///
/// The entry path finds this processor's own block through the task descriptor
/// at the end of that table, so a table that is not ours answers with an
/// address that is not one either. Judging it rather than reading it is what
/// keeps one unrecoverable event from becoming an endless fault; what is left
/// to say is the table it was judged on.
pub(crate) fn foreign_gdt(base: VirtAddr, limit: u16) -> ! {
    report(format_args!(
        "an event arrived on a global descriptor table that is not ours: base {base:#x}, limit \
         {limit:#x}"
    ))
}

/// Writes one line to the port and stops this processor.
fn report(what: Arguments<'_>) -> ! {
    serial::emergency(format_args!("descriptors: {what}"));
    halt()
}
