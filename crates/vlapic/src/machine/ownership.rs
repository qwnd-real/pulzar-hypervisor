//! Which processors this hypervisor runs, and when the answer stops changing.
//!
//! Two facts rather than one, and the second is the one a per-processor flag
//! cannot be. A processor's own claim is made by that processor, so "not taken
//! over yet" and "never going to be" read identically from the outside — and
//! the difference between them is whether a guest's startup message may be put
//! on real hardware. So the machine keeps a fence as well: once the host has
//! finished starting processors, whatever has not been taken over never will
//! be.

use core::sync::atomic::{AtomicBool, Ordering};

use log::warn;

use crate::{
    VlapicError,
    hardware::{sources, timer},
    machine::of,
    registers::{Phase, Vlapic},
};

/// Records that this hypervisor now runs this processor, so that a startup
/// message aimed at it is emulated rather than forwarded to real hardware.
///
/// Called by each processor as it comes up, as early as it can be — which is
/// before it publishes itself to the rest of the machine and before it spends
/// any time on anything else. Until it has been called, a startup message the
/// guest aims at this processor is sent to real hardware, and real hardware
/// would reset the host out from under it. Nothing in here needs this processor
/// to have joined the machine: the controller is found by the identifier its
/// own controller reports, which is why it can come first.
///
/// The order inside is the same argument one level down. Where the guest thinks
/// this processor stands is published before ownership, so that a startup
/// message arriving the instant ownership is taken finds a controller that
/// already knows it is waiting for one. The real sources follow, because they
/// are five register writes and the window they close is open from here.
/// Calibrating the timer comes last because it is the only slow step — it
/// measures a real frequency over milliseconds — and doing it while the
/// processor was still forwardable is what left a window wide enough for a
/// guest to reset a host processor in. It is safe there because the guest has
/// not started this processor, so its emulated timer is still at reset and
/// there is no appointment for calibration to destroy.
///
/// # Why the real sources are programmed here
///
/// Because they are the guest's, and until this runs they are firmware's. Every
/// processor brings its own controller up from the machine's ACPI tables, which
/// arm both interrupt pins *unmasked* as non-maskable interrupts wherever
/// firmware described one — while the controller this processor is joining with
/// is at its reset value, where every entry is masked. A platform
/// non-maskable interrupt in that window is taken by the host and handed to a
/// guest that has masked the pin it came in on, which for a processor the guest
/// has just started is a guest in real mode with no handler for it.
///
/// So the pins are the guest's from the moment it owns the processor, and
/// hardware says what the guest's own table says. The other half of that
/// decision is the host's: a platform non-maskable interrupt while the guest
/// has its pin masked is one nothing sees, because hardware is masked and the
/// host has no use of its own for one. What the host forwards is only ever what
/// the guest's own configuration asked for — through a pin it unmasked, or
/// through a passed-through I/O controller or device message it programmed for
/// the mode, neither of which this table has anything to do with.
///
/// `id` is the identifier this processor's own controller reports.
///
/// `joining` is where the guest's own view of this processor stands, which is
/// not the same question as whether the processor is running. Every processor
/// but the one the guest was entered on has never been started *by the guest*,
/// however long it has been executing the hypervisor's own code.
///
/// # Errors
///
/// [`VlapicError::NotInstalled`] before [`crate::install`],
/// [`VlapicError::NoLapic`] if the roster does not describe this processor, or
/// [`VlapicError::Apic`] if the real timer cannot be measured.
pub fn claim_processor(id: cpu::ApicId, joining: Joining) -> Result<(), VlapicError> {
    let vlapic = of(id)?;
    vlapic.startup().join(joining.phase());
    vlapic.take_ownership();
    if !sources::reprogram(vlapic) {
        warn!(
            "vlapic: {} joined the guest with a source of its own that hardware would not take, so \
             something may deliver into a guest that has it masked",
            vlapic.index()
        );
    }
    timer::calibrate(vlapic)?;
    Ok(())
}

/// Records that the host has finished taking processors over, so nothing the
/// guest sends may reach real hardware again.
///
/// The fence the per-processor claim cannot be. That claim is made by the
/// target itself, so "this processor has not been taken over yet" and "this
/// processor never will be" look identical from the outside — and the second
/// means a guest start-up message would put a real processor into real mode at
/// an address the guest chose, with no nested tables and no intercepts.
/// Anything the host was going to start has been started by the time this is
/// called, so from here on the second reading is the only one left and
/// forwarding stops being something this crate does at all.
///
/// Called once, whatever came of starting them: a processor that failed to
/// start is precisely the one that must not be forwarded to.
pub fn bring_up_finished() {
    BROUGHT_UP.store(true, Ordering::Release);
}

/// Where a processor's guest stands at the moment the hypervisor takes the
/// processor over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Joining {
    /// Already executing the guest, which is true of exactly one processor: the
    /// one the guest was entered on.
    Running,
    /// Never started by the guest, so it holds until the guest starts it —
    /// exactly as a processor still in reset would.
    WaitingForSipi,
}

impl Joining {
    /// The phase this is.
    const fn phase(self) -> Phase {
        match self {
            Self::Running => Phase::Running,
            Self::WaitingForSipi => Phase::WaitingForSipi(None),
        }
    }
}

/// Whether this hypervisor runs the processor a controller belongs to.
pub(crate) fn owns(vlapic: &Vlapic) -> bool {
    vlapic.owned()
}

/// Whether the host has finished taking the machine's processors over.
///
/// The machine-wide half of [`owns`]. After it, no processor is going to join
/// that has not joined already.
pub(crate) fn brought_up() -> bool {
    BROUGHT_UP.load(Ordering::Acquire)
}

/// Whether every processor that will ever be this hypervisor's already is.
///
/// One flag for the machine rather than one per processor, because the question
/// it answers is about the machine: not "has this processor joined" — which its
/// own controller records — but "is there still a processor that might". Set
/// once and never cleared: nothing starts a processor after the one thing that
/// starts them has finished.
static BROUGHT_UP: AtomicBool = AtomicBool::new(false);
