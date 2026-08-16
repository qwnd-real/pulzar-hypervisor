//! What every controller has done, and what it is doing now.
//!
//! # The machine this has to be debugged on has no serial port
//!
//! `serial::init` finds no port on it, so every log record is discarded before
//! it is formatted and logging is not a debugging strategy there. What is left
//! is memory, and the controllers are the right memory to put this in: they
//! live in a leaked `'static` page, they are never dropped, and they outlive
//! every guest. A debugger attached afterwards — or [`describe`] called from
//! anything that has a port — finds the whole history of each one.
//!
//! So every counter is a field of the controller it describes, and the two
//! kinds of record here are the two a post mortem needs: how often something
//! happened, and whether something happened at all.
//!
//! # Why the latches are here as well
//!
//! A guest drives most of the log sites in this crate: it writes a reserved
//! register offset, sends a command the architecture does not define, refuses
//! an interrupt, or changes face, as fast as it can take an exit. Each of those
//! is a line through `serial`'s machine-global lock with interrupts disabled,
//! so an unlatched one is a denial of service rather than a diagnostic — the
//! offending processor answers no doorbell and every other processor that logs
//! blocks behind the same lock.
//!
//! Every one of them therefore says its thing once per controller, out of the
//! one word below. Latching them here rather than each site keeping a flag is
//! what keeps the set countable: a reader can see what a controller has said
//! and what it has not, and a new log site has to take a bit rather than
//! inventing another mechanism.
//!
//! The latches are never cleared, not even by an `INIT`. A guest that resets
//! its own processor in a loop would otherwise re-arm the flood, which is the
//! same defect one round further out.

use core::{
    fmt::{self, Display, Formatter},
    sync::atomic::{AtomicU32, Ordering},
};

use log::info;

use crate::{hardware::sources::Refusal, machine::registry::lapics, registers::lvt::Entry};

/// Logs what each processor's controller is doing, and everything that has
/// happened to it since the machine came up.
///
/// The state that says something is stuck: a controller with interrupts
/// requested and never taken, or one still owing real hardware an
/// acknowledgement, is the shape both an interrupt storm and a lost wakeup show
/// up as — and the counts beside it say whether it has been that way all along.
pub fn describe(who: &str) {
    let Ok(page) = lapics() else {
        info!("{who}: the emulated controllers have not been installed");
        return;
    };
    for vlapic in page.all() {
        info!(
            "{who}: {} {} {} in {}{}, task priority {}, {} requested, {} in service, hardware {}",
            vlapic.index(),
            vlapic.apic_id(),
            if vlapic.startup().running() {
                "running"
            } else {
                "waiting to be started"
            },
            vlapic.mode(),
            if vlapic.base().bootstrap() {
                " as the bootstrap processor"
            } else {
                ""
            },
            vlapic.task_priority(),
            vlapic.requested_count(),
            vlapic.in_service_count(),
            vlapic.ledger().debts(),
        );
        info!(
            "{who}: {} has seen {}",
            vlapic.index(),
            vlapic.diagnostics().counts()
        );
        if let Some(vector) = vlapic.requested() {
            info!(
                "{who}: {} has {vector} requested at processor priority {}",
                vlapic.index(),
                vlapic.processor_priority()
            );
        }
    }
}

/// What has happened to one controller that nobody may have been there to read.
///
/// Counters and one-shot latches together, because they are one record: the
/// counts say how often, and a latch says that the line explaining it has
/// already been printed and will not be printed again.
#[derive(Debug)]
pub(crate) struct Diagnostics {
    /// Interrupts that arrived on real hardware and were handed to this guest.
    arrivals: AtomicU32,
    /// How many of those arrived level triggered, and so had an acknowledgement
    /// withheld or were deliberately acknowledged at once.
    level: AtomicU32,
    /// Interrupts this controller declined, which is its guest's own state
    /// saying no rather than anything going wrong.
    declined: AtomicU32,
    /// Interrupts this hypervisor was given and recorded in no controller at
    /// all, which is an interrupt lost.
    ///
    /// Counted against the controller nearest to where it was lost: the target
    /// where a message had one, and the sender where a redirectable interrupt
    /// had none that took it.
    dropped: AtomicU32,
    /// Periodic timer counts raised to the shortest period this hypervisor puts
    /// on real hardware.
    clamped: AtomicU32,
    /// Which of the one-shot diagnostics this controller has already said, one
    /// bit per [`Report`].
    said: AtomicU32,
    /// Which refusals of a local-vector-table entry's configuration it has
    /// already said, one bit per entry per kind of [`Refusal`].
    refusals: AtomicU32,
}

impl Diagnostics {
    /// Nothing has happened yet, which is what a controller is built with.
    pub(crate) const fn new() -> Self {
        Self {
            arrivals: AtomicU32::new(0),
            level: AtomicU32::new(0),
            declined: AtomicU32::new(0),
            dropped: AtomicU32::new(0),
            clamped: AtomicU32::new(0),
            said: AtomicU32::new(0),
            refusals: AtomicU32::new(0),
        }
    }

    /// Counts an interrupt that arrived on real hardware for this guest.
    ///
    /// Written only by the processor the controller belongs to — an arrival
    /// lands on the processor it was addressed to — so this is a counter on a
    /// line that processor already owns, unlike the machine-wide one it
    /// replaces. Relaxed and saturating: nothing orders anything against it,
    /// and a count that wrapped would be a diagnostic that lied where one that
    /// stops is a diagnostic that says "at least this many".
    pub(crate) fn arrived(&self, level: bool) {
        saturate(&self.arrivals);
        if level {
            saturate(&self.level);
        }
    }

    /// Counts an interrupt this controller declined.
    pub(crate) fn declined(&self) {
        saturate(&self.declined);
    }

    /// Counts an interrupt that reached no controller at all.
    pub(crate) fn dropped(&self) {
        saturate(&self.dropped);
    }

    /// Counts a periodic timer count raised to the shortest period.
    pub(crate) fn clamped(&self) {
        saturate(&self.clamped);
    }

    /// Whether this is the first time this controller has had `report` to make,
    /// and records that it has now.
    pub(crate) fn say(&self, report: Report) -> bool {
        let bit = report.bit();
        self.said.fetch_or(bit, Ordering::AcqRel) & bit == 0
    }

    /// Whether this is the first time this controller has refused an entry's
    /// configuration in this way, and records that it has now.
    ///
    /// Per kind as well as per entry, so that a vector refused in an entry
    /// cannot silence a delivery mode refused in the same one. Every refusal is
    /// re-derived on every reprogram, and a guest can leave a refused entry in
    /// place and then write an unrelated register as fast as it can take an
    /// exit.
    pub(crate) fn say_refusal(&self, entry: Entry, refusal: Refusal) -> bool {
        let bit = 1 << (entry.index() * Refusal::COUNT + refusal.kind());
        self.refusals.fetch_or(bit, Ordering::AcqRel) & bit == 0
    }

    /// Everything counted here, as one value.
    pub(crate) fn counts(&self) -> Counts {
        Counts {
            arrivals: self.arrivals.load(Ordering::Relaxed),
            level: self.level.load(Ordering::Relaxed),
            declined: self.declined.load(Ordering::Relaxed),
            dropped: self.dropped.load(Ordering::Relaxed),
            clamped: self.clamped.load(Ordering::Relaxed),
        }
    }
}

/// Adds one to a counter without ever wrapping.
///
/// A diagnostic that wrapped would report a small number for a machine that had
/// seen four billion of something, which is the one answer worse than no
/// answer. Saturating also keeps this off the list of things that can trap: the
/// plain addition it replaces panicked in a debug build, in an interrupt
/// handler, once the machine had seen `u32::MAX` arrivals.
fn saturate(counter: &AtomicU32) {
    let _ = counter.try_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
        count.checked_add(1)
    });
}

/// What one controller has counted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Counts {
    /// Interrupts handed to this guest from real hardware.
    arrivals: u32,
    /// How many of those were level triggered.
    level: u32,
    /// Interrupts this controller declined.
    declined: u32,
    /// Interrupts lost.
    dropped: u32,
    /// Periodic counts raised to the shortest period.
    clamped: u32,
}

impl Display for Counts {
    /// Only what is not zero, because a line reporting five zeroes says
    /// nothing. Arrivals are named even at zero, because a controller that
    /// has seen none at all is itself worth knowing.
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} arrivals", self.arrivals)?;
        for (count, what) in [
            (self.level, "level triggered"),
            (self.declined, "declined"),
            (self.dropped, "lost"),
            (self.clamped, "periods raised"),
        ] {
            if count != 0 {
                write!(formatter, ", {count} {what}")?;
            }
        }
        Ok(())
    }
}

/// One thing a controller says once and then stops saying.
///
/// Every one of these is a line a guest can otherwise produce as fast as it can
/// take an exit. The variant is the identity of the *site*, not of the value it
/// reports, because what a flood costs does not depend on which index or which
/// mode the guest used — so the first one carries the detail and the rest are
/// the same fact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Report {
    /// A read of a model-specific register this face refused.
    RefusedRead,
    /// A write of one.
    RefusedWrite,
    /// An offset the memory-mapped face reserves.
    IllegalRegister,
    /// A command whose delivery mode the architecture reserves.
    ReservedCommand,
    /// A command the architecture defines no processor as sending.
    IllegalCommand,
    /// A system-management interrupt through the interrupt command register,
    /// which this machine does not deliver.
    SystemManagement,
    /// A message aimed at a processor this hypervisor does not run.
    Unrun,
    /// A redirectable interrupt every processor it named refused.
    Unaccepted,
    /// An interrupt offered to a controller whose register file was being
    /// reset.
    Resetting,
    /// A doorbell that could not be sent, so a target will not look at its
    /// controller until it exits for another reason.
    Undelivered,
    /// An `INIT` another processor sent this one.
    InitSent,
    /// An `INIT` this processor applied to itself.
    Initialized,
    /// A start-up message another processor sent this one.
    StartupSent,
    /// A start-up message this processor took.
    Started,
    /// A reset that left a source armed or real hardware holding something.
    ResetUnsettled,
    /// A software-disable that left real hardware holding something.
    DisableUnsettled,
    /// A software-disable that left something armed behind the controller.
    DisableArmed,
    /// A change of face that left a source armed or real hardware holding
    /// something.
    FaceUnsettled,
    /// A change of face hardware could not be brought across.
    FaceUnprogrammed,
    /// The face the controller was last seen entering.
    FaceEntered,
    /// The real controller could not be taken into x2APIC behind its guest.
    Unpromoted,
    /// The real controller's logical destination does not agree with the
    /// guest's, so interrupts addressed logically do not reach it.
    LogicalDestination,
    /// The guest's periodic timer is being given a longer period than it asked
    /// for.
    TimerFloor,
    /// The guest's error entry does not name an interrupt this controller can
    /// deliver, so its error interrupt is recorded and not raised.
    ErrorEntry,
}

impl Report {
    /// Every one of them, so that the word they are latched in can be shown to
    /// be wide enough and each of them shown to have a bit of its own.
    const ALL: [Self; 24] = [
        Self::RefusedRead,
        Self::RefusedWrite,
        Self::IllegalRegister,
        Self::ReservedCommand,
        Self::IllegalCommand,
        Self::SystemManagement,
        Self::Unrun,
        Self::Unaccepted,
        Self::Resetting,
        Self::Undelivered,
        Self::InitSent,
        Self::Initialized,
        Self::StartupSent,
        Self::Started,
        Self::ResetUnsettled,
        Self::DisableUnsettled,
        Self::DisableArmed,
        Self::FaceUnsettled,
        Self::FaceUnprogrammed,
        Self::FaceEntered,
        Self::Unpromoted,
        Self::LogicalDestination,
        Self::TimerFloor,
        Self::ErrorEntry,
    ];

    /// Which bit of the latch this one takes.
    const fn bit(self) -> u32 {
        1 << (self as u32)
    }
}

/// Every report needs a bit of its own in the one word they share: one that
/// fell outside would silence another report rather than its own.
const _: () = assert!(
    Report::ALL.len() <= u32::BITS as usize,
    "every one-shot diagnostic needs a bit of the word they are latched in"
);

/// And so does every kind of refusal of every entry, in the other word.
const _: () = assert!(
    Entry::COUNT * Refusal::COUNT <= u32::BITS as usize,
    "every entry needs a bit per kind of refusal"
);

#[cfg(test)]
mod tests {
    //! The record a machine with no serial port is read by, so what it says has
    //! to be legible and each latch has to be its own.

    use alloc::format;

    use descriptors::Vector;

    use super::{Diagnostics, Ordering, Report};
    use crate::{hardware::sources::Refusal, registers::lvt::Entry};

    #[test]
    fn every_one_shot_diagnostic_has_a_bit_of_its_own() {
        // A shared bit is a report that silences another report rather than
        // itself, which is worse than no latch at all: the line that would have
        // explained a machine is missing and nothing says so.
        let diagnostics = Diagnostics::new();
        for (position, report) in Report::ALL.into_iter().enumerate() {
            assert_eq!(report.bit(), 1 << position, "{report:?}");
            assert!(diagnostics.say(report), "{report:?} was said by another");
        }
        for report in Report::ALL {
            assert!(!diagnostics.say(report), "{report:?} was said twice");
        }
    }

    #[test]
    fn a_refusal_is_latched_per_entry_and_per_kind() {
        let diagnostics = Diagnostics::new();
        let vector = Refusal::Vector(Vector::new(0x0F));
        let delivery = Refusal::Delivery(0b101);

        assert!(diagnostics.say_refusal(Entry::Lint0, vector));
        assert!(!diagnostics.say_refusal(Entry::Lint0, vector));
        assert!(
            diagnostics.say_refusal(Entry::Lint0, delivery),
            "a vector refused in an entry must not silence a mode refused in it"
        );
        assert!(
            diagnostics.say_refusal(Entry::Lint1, vector),
            "nor a refusal in another entry"
        );
    }

    #[test]
    fn counts_report_arrivals_always_and_everything_else_only_when_it_happened() {
        let diagnostics = Diagnostics::new();
        assert_eq!(format!("{}", diagnostics.counts()), "0 arrivals");

        diagnostics.arrived(false);
        assert_eq!(format!("{}", diagnostics.counts()), "1 arrivals");

        diagnostics.arrived(true);
        diagnostics.declined();
        diagnostics.dropped();
        diagnostics.clamped();
        assert_eq!(
            format!("{}", diagnostics.counts()),
            "2 arrivals, 1 level triggered, 1 declined, 1 lost, 1 periods raised"
        );
    }

    #[test]
    fn a_counter_stops_rather_than_wrapping() {
        // What the machine-wide arrival counter this replaces did instead was
        // panic in a debug build, inside an interrupt handler, after four billion
        // arrivals — and a wrapped count would report a machine that had seen
        // none.
        let diagnostics = Diagnostics::new();
        diagnostics.arrivals.store(u32::MAX - 1, Ordering::Relaxed);
        for _ in 0..3 {
            diagnostics.arrived(false);
        }
        assert_eq!(diagnostics.counts().arrivals, u32::MAX);
    }
}
