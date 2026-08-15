//! Being started, stopped and interrupted: the state a controller holds that is
//! not a register at all.
//!
//! Everything here is written by one processor and read by another, which is
//! what the orderings are for and why none of it can live with the exit loop
//! that consumes it. A startup message arrives from whichever processor sent it
//! and is applied by its target at an exit boundary; a non-maskable interrupt
//! is counted by the sender and drained by the target; and the two flags —
//! whether the target is watching its controller, and whether this hypervisor
//! owns the processor at all — are each set by one side and read by the other.
//!
//! What is *not* here is what any of it means. Applying a startup message is
//! [`crate::lifecycle`]'s, and whether one may be put on real hardware is
//! [`crate::delivery`]'s.

use core::sync::atomic::Ordering;

use crate::registers::Vlapic;

impl Vlapic {
    /// Whether this processor has stopped looking at this controller.
    ///
    /// True while it is inside the guest, and true while it is halted waiting
    /// to be started — the two states in which nothing it does will notice a
    /// bit being set here until something interrupts it. False while it is
    /// answering an exit, because it consults this controller before it goes
    /// back in.
    ///
    /// Read by a processor about to deliver something here, to decide whether
    /// the target has to be interrupted to notice.
    pub(crate) fn away(&self) -> bool {
        self.away.load(Ordering::SeqCst)
    }

    /// Records whether this processor has stopped looking at this controller.
    ///
    /// Sequentially consistent, and it has to be. This store and the load of a
    /// request bit that follows it must not be reordered against a deliverer's
    /// store of that request bit and its load of this flag — if both were
    /// allowed to be seen stale, the deliverer would decide no interruption was
    /// needed while this processor decided nothing was pending, and the
    /// interrupt would be lost until something unrelated happened to cause an
    /// exit.
    pub(crate) fn set_away(&self, away: bool) {
        self.away.store(away, Ordering::SeqCst);
    }

    /// Records that this processor's guest is owed a non-maskable interrupt.
    ///
    /// Set by whichever processor sent it, and drained by this one at its next
    /// exit — which is why it lives here and not with the rest of what that
    /// processor's exit loop owns.
    ///
    /// Sequentially consistent for the reason [`Vlapic::signalled`] gives: this
    /// store and the sender's read of [`Vlapic::away`] pair with the target's
    /// store of that flag and its read of this one, and a weaker ordering lets
    /// both sides miss.
    pub(crate) fn raise_nmi(&self) {
        let _ = self
            .nmi
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |count| {
                (count < 2).then(|| count + 1)
            });
    }

    /// Takes the outstanding non-maskable interrupt, if there is one.
    pub(crate) fn take_nmi(&self) -> bool {
        self.nmi
            .try_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                (count != 0).then(|| count - 1)
            })
            .is_ok()
    }

    /// Whether this hypervisor runs the processor this controller belongs to.
    ///
    /// Until it does, the processor is running firmware's own code on real
    /// hardware and a startup message aimed at it belongs on the real
    /// controller. Once it does, the same message must be emulated, because
    /// forwarding it would reset the host.
    ///
    /// Not on its own the question "may a startup message be forwarded". This
    /// is set by the processor it describes, so a processor that never gets
    /// there never sets it, and "not taken over yet" and "never going to be"
    /// read the same here. What tells them apart is machine-wide and lives in
    /// the crate root; [`crate::delivery`] is where the two are put together.
    pub(crate) fn owned(&self) -> bool {
        self.owned.load(Ordering::Acquire)
    }

    /// Records that this hypervisor now runs this processor.
    ///
    /// Called by that processor about itself, before it publishes itself to the
    /// rest of the machine: whoever started it waits for that publication, and
    /// a processor the machine believes is up while its controller still
    /// says it is firmware's is a processor a guest can reset for real.
    pub(crate) fn take_ownership(&self) {
        self.owned.store(true, Ordering::Release);
    }

    /// What this processor is doing about being started and stopped.
    pub(crate) fn startup(&self) -> Startup {
        Startup::from_bits(self.startup.load(Ordering::Acquire))
    }

    /// Whether this processor's guest is running rather than reset.
    pub(crate) fn running(&self) -> bool {
        self.startup() == Startup::Running
    }

    /// Whether a startup message has arrived that this processor has not
    /// applied yet.
    ///
    /// What a processor halted with nothing to run tests before it halts again.
    /// Sequentially consistent, and paired with [`Vlapic::set_away`]: the
    /// sender stores the message and then reads whether this processor is away,
    /// this processor stores that it is away and then reads for a message, and
    /// the total order over those four is what guarantees at least one of them
    /// sees the other. Without it a message could arrive between the test and
    /// the halt and be answered by nobody.
    pub(crate) fn signalled(&self) -> bool {
        self.startup.load(Ordering::SeqCst) != Startup::WaitingForSipi as u8
            || self.sipi_vector.load(Ordering::SeqCst) != NO_SIPI
    }

    /// Puts this processor into the state an INIT leaves it in, from any state.
    ///
    /// The vector a start-up message would have carried is cleared as part of
    /// the same transition, so that an INIT always wins a race against a
    /// start-up message already on its way: the target applies the reset and
    /// then finds nothing to start with.
    ///
    /// Both stores are sequentially consistent for the reason
    /// [`Vlapic::signalled`] gives.
    pub(crate) fn request_init(&self) {
        self.sipi_vector.store(NO_SIPI, Ordering::SeqCst);
        self.startup
            .store(Startup::InitRequested as u8, Ordering::SeqCst);
    }

    /// Offers a start-up page, which takes only if this processor is waiting
    /// for one.
    pub(crate) fn request_sipi(&self, page: StartupPage) -> bool {
        if self.startup() != Startup::WaitingForSipi {
            return false;
        }
        self.sipi_vector
            .store(u32::from(page.number()), Ordering::SeqCst);
        true
    }

    /// Takes the start-up page this processor was given, if it has one.
    pub(crate) fn take_sipi(&self) -> Option<StartupPage> {
        let vector = self.sipi_vector.swap(NO_SIPI, Ordering::AcqRel);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a start-up page number is eight bits, and the sentinel is the only wider value stored"
        )]
        (vector != NO_SIPI).then(|| StartupPage::new(vector as u8))
    }

    /// Moves this processor to a startup state it has decided to be in.
    pub(crate) fn set_startup(&self, startup: Startup) {
        self.startup.store(startup as u8, Ordering::Release);
    }
}

/// Where a start-up message tells a processor to begin executing.
///
/// A page number, and not a vector, although it travels in the vector field of
/// the command that carries it: a processor released by one begins in real mode
/// at the start of this page, and nothing is ever delivered on the number. The
/// two are kept apart so that neither can be handed to the other's callers —
/// which is a real hazard here, because the same eight bits of the same
/// register are a vector under every other delivery mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StartupPage(u8);

impl StartupPage {
    /// The page a start-up command's field names.
    pub(crate) const fn new(page: u8) -> Self {
        Self(page)
    }

    /// The number itself, for whoever has to put a processor there.
    #[must_use]
    pub const fn number(self) -> u8 {
        self.0
    }
}

/// What a processor is doing about being started and stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum Startup {
    /// Running the guest, or ready to.
    Running = 0,
    /// Another processor has sent an INIT that this one has not applied yet.
    InitRequested = 1,
    /// Reset and held, doing nothing until a start-up message arrives.
    WaitingForSipi = 2,
}

impl Startup {
    /// The state an encoding names, with anything undefined reading as running
    /// — the state every processor is in unless something put it elsewhere.
    const fn from_bits(bits: u8) -> Self {
        match bits {
            1 => Self::InitRequested,
            2 => Self::WaitingForSipi,
            _ => Self::Running,
        }
    }
}

/// The value stored where a start-up vector would be when there is none.
///
/// Outside the eight bits a vector occupies, so it cannot collide with one.
pub(super) const NO_SIPI: u32 = u32::MAX;
