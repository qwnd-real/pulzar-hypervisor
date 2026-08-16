//! Being started, stopped and interrupted: the state a controller holds that is
//! not a register at all.
//!
//! Everything here is written by one processor and read by another, which is
//! what the orderings are for and why none of it can live with the exit loop
//! that consumes it. An INIT and a start-up message arrive from whichever
//! processor sent them and are applied by their target at an exit boundary; a
//! non-maskable interrupt is counted by the sender and drained by the target;
//! and the two flags — whether the target is watching its controller, and
//! whether this hypervisor owns the processor at all — are each set by one side
//! and read by the other.
//!
//! # Being started is one word, and every change to it one step
//!
//! What a processor is doing about being started, and where a start-up message
//! told it to begin, are one answer to one question — so they are one word.
//! [`Startup`] is that word and [`Phase`] is what it holds, with the page
//! attached to the phases that can have one: a processor that is running
//! cannot be holding a page, and a page cannot be left behind by a phase
//! moving out from under it.
//!
//! Every change to it is a single step — three compare-exchanges and the one
//! store an INIT is — and that is what makes the reasoning finite. Two
//! operations running at once are those two steps in one of their two orders,
//! so it is enough to show that both orders leave a state some serial order of
//! the same messages produces. Which of the two wins is decided the way the
//! architecture decides it: an INIT replaces the word whatever was in it, a
//! start-up message only ever attaches its page to a processor that is not
//! running, and the target's own two steps — recording that it has performed
//! the reset, and taking the page it was given — take effect only against a
//! word no INIT has touched since they looked at it.
//!
//! The one other store is the one a processor makes about itself as it joins
//! the machine, before anything can be delivering to it at all.
//!
//! What is *not* here is what any of it means. Applying an INIT is
//! [`crate::lifecycle`]'s, and whether one may be put on real hardware is
//! [`crate::delivery`]'s.

use core::sync::atomic::{AtomicU32, Ordering};

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
    /// Sequentially consistent for the reason [`Startup::signalled`] gives:
    /// this store and the sender's read of [`Vlapic::away`] pair with the
    /// target's store of that flag and its read of this one, and a weaker
    /// ordering lets both sides miss.
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

    /// Discards the non-maskable interrupts this processor's guest was owed.
    ///
    /// Not part of [`Vlapic::reset_registers`], and the difference is what this
    /// counts: interrupts already *delivered* to the processor, waiting to be
    /// injected at its next entry, rather than anything the register file
    /// holds. So the two transitions that discard them are the two the
    /// architecture defines as a processor starting over — an INIT applied,
    /// and a start-up message taken — and a guest that merely switches its
    /// controller off keeps what it was owed, exactly as one that switches
    /// it back on does.
    ///
    /// An interrupt raised concurrently with this is discarded, which is one of
    /// the two orders it could have arrived in and the one nothing
    /// distinguishes from the other. It deliberately does not get the
    /// publish-and-recheck an ordinary arrival racing a reset gets: the
    /// sender is a physical non-maskable interrupt handler, which reaches
    /// [`Vlapic::raise_nmi`] while the processor is inside a reset it is in
    /// no position to wait for.
    pub(crate) fn discard_nmi(&self) {
        self.nmi.store(0, Ordering::SeqCst);
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
    pub(crate) const fn startup(&self) -> &Startup {
        &self.startup
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

/// What a processor is doing about being started and stopped, and where a
/// start-up message has told it to begin if one has arrived.
///
/// The page belongs to the phase rather than sitting beside it because the two
/// are only ever meaningful together. A processor that is running has nothing
/// to be started at, so that variant has no page to hold; a processor that has
/// been sent an INIT it has not applied yet may already have been sent the
/// start-up message that follows it, so that one does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Phase {
    /// Running the guest, or ready to.
    Running,
    /// Another processor has sent an INIT that this one has not applied yet.
    ///
    /// The reset it asks for is performed by the target itself, at its next
    /// exit boundary, which is what makes clearing a whole register file
    /// safe without a lock. A start-up message arriving in the meantime is
    /// kept rather than refused: the two are sent microseconds apart —
    /// firmware's own delay between them is ten — so a start-up that lands
    /// here is one the guest sent *after* the INIT, and refusing it strands
    /// the processor until the guest happens to send another.
    InitRequested(Option<StartupPage>),
    /// Reset and held, doing nothing until a start-up message arrives — and
    /// carrying the page one arrived with, until this processor takes it.
    WaitingForSipi(Option<StartupPage>),
}

impl Phase {
    /// The page a start-up message has left this processor, if one has.
    const fn page(self) -> Option<StartupPage> {
        match self {
            Self::Running => None,
            Self::InitRequested(page) | Self::WaitingForSipi(page) => page,
        }
    }

    /// The word this phase is kept as.
    const fn bits(self) -> u32 {
        let (phase, page) = match self {
            Self::Running => (RUNNING, None),
            Self::InitRequested(page) => (INIT_REQUESTED, page),
            Self::WaitingForSipi(page) => (WAITING_FOR_SIPI, page),
        };
        let (pending, number) = match page {
            Some(page) => (PAGE_PENDING, page.number()),
            None => (0, 0),
        };
        u32::from_le_bytes([phase | pending, number, 0, 0])
    }

    /// The phase a word holds.
    ///
    /// Total, and every encoding no [`Phase::bits`] produces reads as a running
    /// processor — which is what one is unless something put it elsewhere, and
    /// which carries no page, so a word that says both cannot answer with one.
    const fn from_bits(bits: u32) -> Self {
        let [phase, number, ..] = bits.to_le_bytes();
        let page = if phase & PAGE_PENDING == 0 {
            None
        } else {
            Some(StartupPage::new(number))
        };
        match phase & !PAGE_PENDING {
            INIT_REQUESTED => Self::InitRequested(page),
            WAITING_FOR_SIPI => Self::WaitingForSipi(page),
            _ => Self::Running,
        }
    }
}

/// The one word a processor's whole startup state is kept in.
///
/// Every operation on it is sequentially consistent. That is not caution: the
/// word takes part in the pairing with [`Vlapic::away`] that stops a message
/// being lost to a processor on its way into the guest, and that argument needs
/// a total order over the four accesses it is made of. Nothing is given up for
/// it either — the word changes once per INIT, once per start-up message and
/// once per exit boundary that has one to apply.
#[derive(Debug)]
pub(crate) struct Startup(AtomicU32);

impl Startup {
    /// A processor in this phase, which is how one is built.
    pub(crate) const fn new(phase: Phase) -> Self {
        Self(AtomicU32::new(phase.bits()))
    }

    /// What this processor is doing about being started and stopped.
    pub(crate) fn phase(&self) -> Phase {
        Phase::from_bits(self.0.load(Ordering::SeqCst))
    }

    /// Whether this processor's guest is running rather than reset.
    pub(crate) fn running(&self) -> bool {
        matches!(self.phase(), Phase::Running)
    }

    /// Whether anything has arrived that this processor has not applied yet.
    ///
    /// What a processor halted with nothing to run tests before it halts again,
    /// which is why it is the one question asked of the whole word: a reset to
    /// apply and a page to start at are both something to wake up for, and
    /// being held with neither is the only state in which halting again is
    /// right.
    ///
    /// Sequentially consistent, and paired with [`Vlapic::set_away`]: the
    /// sender stores the message and then reads whether this processor is
    /// away, this processor stores that it is away and then reads for a
    /// message, and the total order over those four is what guarantees at
    /// least one of them sees the other. Without it a message could arrive
    /// between the test and the halt and be answered by nobody.
    pub(crate) fn signalled(&self) -> bool {
        self.phase() != Phase::WaitingForSipi(None)
    }

    /// Puts this processor where an INIT leaves it, from wherever it was.
    ///
    /// One store of the whole word, which is the whole of what makes "an INIT
    /// always wins" true rather than usually true. Any page a start-up message
    /// had left is discarded with the phase it was attached to, so a target
    /// that was about to be started is instead reset and held — and a
    /// start-up message still on its way finds a processor that has an INIT
    /// to apply first.
    pub(crate) fn requested_init(&self) {
        self.0
            .store(Phase::InitRequested(None).bits(), Ordering::SeqCst);
    }

    /// Offers a start-up page, and says whether this processor took it.
    ///
    /// Taken by a processor that has been reset, whether or not it has applied
    /// the reset yet, and refused by one that is running — which is what makes
    /// the second of the pair a guest sends harmless: the first starts the
    /// processor and the second finds it already started.
    ///
    /// The page replaces any earlier one, because the register the message came
    /// out of holds one page and the last one written is the one the guest
    /// means.
    ///
    /// Refusing against the phase and attaching the page are one step, so an
    /// INIT landing between them cannot leave the page attached to a phase the
    /// INIT replaced: either the INIT is first, and the page is attached to the
    /// reset it has to wait for, or the page is first, and the INIT discards
    /// it.
    pub(crate) fn offered(&self, page: StartupPage) -> bool {
        self.0
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |bits| {
                match Phase::from_bits(bits) {
                    // Nothing to start: a processor that is running was either
                    // never reset or has already been released.
                    Phase::Running => None,
                    Phase::InitRequested(_) => Some(Phase::InitRequested(Some(page)).bits()),
                    Phase::WaitingForSipi(_) => Some(Phase::WaitingForSipi(Some(page)).bits()),
                }
            })
            .is_ok()
    }

    /// Records that the reset an INIT asked for has been performed, and leaves
    /// the processor where that reset leaves it.
    ///
    /// `bootstrap` is what decides where that is, and the asymmetry is the
    /// multiprocessor protocol's: an application processor waits for a start-up
    /// message, and the processor the machine came up on resumes execution
    /// instead — it is the one that *sends* the start-up messages, so a
    /// bootstrap processor held waiting for one would be waiting for itself.
    ///
    /// A page a start-up message left is carried across, and dropped for a
    /// bootstrap processor along with the wait it would have released.
    ///
    /// An INIT arriving while the reset was being performed is folded into it:
    /// the word says nothing more than that one is outstanding, and a reset
    /// that completes after it was sent is a reset it got. One arriving
    /// *after* this lands on the phase below and is applied at the next
    /// exit boundary.
    pub(crate) fn initialized(&self, bootstrap: bool) {
        let _ = self
            .0
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |bits| {
                match Phase::from_bits(bits) {
                    Phase::InitRequested(page) => Some(
                        if bootstrap {
                            Phase::Running
                        } else {
                            Phase::WaitingForSipi(page)
                        }
                        .bits(),
                    ),
                    // Nothing to record: the phase this leaves is only ever left
                    // by the processor the controller belongs to, which is the
                    // one here.
                    Phase::Running | Phase::WaitingForSipi(_) => None,
                }
            });
    }

    /// Takes the page this processor has been given, if it is waiting and has
    /// one, and starts running.
    ///
    /// Consuming the page and recording that the processor is running are one
    /// step, and that is what keeps an INIT from being swallowed. An INIT
    /// landing after the page was read and before the processor published
    /// itself as running used to be overwritten by that publication — and
    /// the phase was the only record of it, so the reset was lost and the
    /// start-up message that followed was refused by a processor that
    /// looked as though it had never been reset. Here the INIT makes the
    /// exchange fail, the page stays where the INIT left it, which is
    /// nowhere, and the next exit boundary applies the reset.
    pub(crate) fn started(&self) -> Option<StartupPage> {
        self.0
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |bits| {
                matches!(Phase::from_bits(bits), Phase::WaitingForSipi(Some(_)))
                    .then_some(Phase::Running.bits())
            })
            .ok()
            .and_then(|taken| Phase::from_bits(taken).page())
    }

    /// Puts this processor in `phase`, whatever it was doing.
    ///
    /// For a processor saying where its guest stands as it joins the machine,
    /// which is not a transition and cannot lose one: nothing is delivered to a
    /// controller belonging to a processor this hypervisor has not taken over,
    /// because a startup message aimed at one goes to real hardware instead.
    pub(crate) fn join(&self, phase: Phase) {
        self.0.store(phase.bits(), Ordering::SeqCst);
    }
}

/// [`Phase::Running`] in the word: no phase bits, and no page.
const RUNNING: u8 = 0;

/// [`Phase::InitRequested`] in the word.
const INIT_REQUESTED: u8 = 1;

/// [`Phase::WaitingForSipi`] in the word.
const WAITING_FOR_SIPI: u8 = 2;

/// The bit that says the byte beside the phase is a page a start-up message
/// left, rather than the zero a phase with no page is written with.
///
/// A bit of its own because page zero is a page a guest may name, so a sentinel
/// in the page's own byte would be a start-up message indistinguishable from
/// none at all.
const PAGE_PENDING: u8 = 1 << 7;

#[cfg(test)]
mod tests {
    //! The interleavings, written as the order the steps landed in rather than
    //! as timing. Every operation on the word is one store or one
    //! compare-exchange, so two of them running at once are exactly these steps
    //! in one of their two orders — and a test that plays an order out is a
    //! test of that race, on a host, with no threads and nothing to be
    //! flaky about.

    use super::{PAGE_PENDING, Phase, RUNNING, Startup, StartupPage};

    /// The page firmware's own start-up message named on the machine this was
    /// written on.
    const PAGE: StartupPage = StartupPage::new(0x87);

    /// A second page, for the cases where which of two messages took matters.
    const LATER: StartupPage = StartupPage::new(0x12);

    /// Every phase the word can hold, so that a test over all of them cannot
    /// silently stop covering one.
    const PHASES: [Phase; 5] = [
        Phase::Running,
        Phase::InitRequested(None),
        Phase::InitRequested(Some(PAGE)),
        Phase::WaitingForSipi(None),
        Phase::WaitingForSipi(Some(PAGE)),
    ];

    /// One step of the machine: what one processor does to the word in one
    /// operation.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Step {
        /// Another processor sent an INIT.
        Init,
        /// Another processor sent a start-up message naming a page.
        Startup(StartupPage),
        /// This processor performed the reset an INIT asked for.
        Reset {
            /// Whether it is the processor the machine came up on.
            bootstrap: bool,
        },
        /// This processor took whatever page it had been given.
        Take,
    }

    /// Every step, for the tests that play out all of their orderings.
    const STEPS: [Step; 5] = [
        Step::Init,
        Step::Startup(LATER),
        Step::Reset { bootstrap: false },
        Step::Reset { bootstrap: true },
        Step::Take,
    ];

    /// Performs one step, answering the page it started the processor at if it
    /// was the step that did.
    fn step(startup: &Startup, step: Step) -> Option<StartupPage> {
        match step {
            Step::Init => startup.requested_init(),
            Step::Startup(page) => {
                startup.offered(page);
            }
            Step::Reset { bootstrap } => startup.initialized(bootstrap),
            Step::Take => return startup.started(),
        }
        None
    }

    #[test]
    fn every_phase_survives_the_word_it_is_kept_in() {
        for phase in PHASES {
            assert_eq!(Phase::from_bits(phase.bits()), phase);
        }
        // Including both ends of the page's range, where a sentinel in its own
        // byte would have collided with a page a guest may name.
        for number in [0, 1, u8::MAX] {
            let page = StartupPage::new(number);
            for phase in [
                Phase::InitRequested(Some(page)),
                Phase::WaitingForSipi(Some(page)),
            ] {
                assert_eq!(Phase::from_bits(phase.bits()), phase, "{phase:?}");
            }
        }
    }

    #[test]
    fn a_running_processor_cannot_be_holding_a_page() {
        // What attaching the page to the phase buys, asserted from the other
        // side: the word can be written with a page and a running processor in
        // it, and there is no phase for it to read back as.
        let both = u32::from_le_bytes([RUNNING | PAGE_PENDING, PAGE.number(), 0, 0]);

        assert_eq!(Phase::from_bits(both), Phase::Running);
        assert_eq!(Phase::from_bits(both).page(), None);
        assert_eq!(Phase::Running.page(), None);
    }

    #[test]
    fn a_start_up_that_lands_before_the_init_is_applied_is_applied_after_it() {
        // The intermittent boot hang: firmware waits ten microseconds between
        // the two, and the reset is applied by the target at its next exit, so
        // the first start-up message routinely lands while the INIT is still
        // outstanding. Dropping it left the processor waiting for a second one.
        let startup = Startup::new(Phase::Running);
        startup.requested_init();

        assert!(startup.offered(PAGE), "a reset processor takes a page");
        assert_eq!(startup.phase(), Phase::InitRequested(Some(PAGE)));
        assert_eq!(
            startup.started(),
            None,
            "nothing starts before the reset it is waiting for"
        );

        startup.initialized(false);

        assert_eq!(startup.phase(), Phase::WaitingForSipi(Some(PAGE)));
        assert_eq!(startup.started(), Some(PAGE));
        assert_eq!(startup.phase(), Phase::Running);
    }

    #[test]
    fn an_init_replaces_a_start_up_that_had_already_arrived() {
        // The other order of the same pair, and the one the architecture decides
        // the other way: an INIT is not refused by anything, so the page goes
        // with the phase it was attached to.
        let startup = Startup::new(Phase::WaitingForSipi(None));

        assert!(startup.offered(PAGE));
        startup.requested_init();

        assert_eq!(startup.phase(), Phase::InitRequested(None));
        assert_eq!(startup.started(), None);
    }

    #[test]
    fn an_init_that_lands_while_a_start_up_is_being_taken_is_not_swallowed() {
        // The reset that used to be lost. The page is read, the INIT lands, and
        // the publication of a running processor overwrote the only record of
        // it — after which the guest's next start-up message was refused too,
        // because the processor looked as though it had never been reset.
        let startup = Startup::new(Phase::WaitingForSipi(Some(PAGE)));
        startup.requested_init();

        assert_eq!(startup.started(), None);
        assert_eq!(startup.phase(), Phase::InitRequested(None));
        assert!(!startup.running());
    }

    #[test]
    fn two_inits_leave_one_reset_to_apply() {
        let startup = Startup::new(Phase::Running);
        startup.requested_init();
        startup.requested_init();

        assert_eq!(startup.phase(), Phase::InitRequested(None));

        startup.initialized(false);

        assert_eq!(startup.phase(), Phase::WaitingForSipi(None));
    }

    #[test]
    fn an_init_that_arrives_while_the_reset_runs_is_folded_into_it() {
        // A reset that finishes after the second INIT was sent is a reset that
        // second INIT got: what it asks for is a cleared register file and a held
        // processor, and that is what it is left with.
        let startup = Startup::new(Phase::InitRequested(None));
        startup.requested_init();
        startup.initialized(false);

        assert_eq!(startup.phase(), Phase::WaitingForSipi(None));
    }

    #[test]
    fn a_bootstrap_processor_resumes_where_an_application_processor_waits() {
        // Only application processors enter the wait. The bootstrap processor is
        // the one that sends the start-up messages, so one held waiting for a
        // message would be waiting for itself — which is a guest down a
        // processor for good.
        let application = Startup::new(Phase::InitRequested(Some(PAGE)));
        application.initialized(false);

        assert_eq!(application.phase(), Phase::WaitingForSipi(Some(PAGE)));
        assert!(!application.running());

        let bootstrap = Startup::new(Phase::InitRequested(Some(PAGE)));
        bootstrap.initialized(true);

        assert_eq!(bootstrap.phase(), Phase::Running);
        assert!(bootstrap.running());
    }

    #[test]
    fn a_running_processor_takes_no_start_up_message() {
        // What makes the second of the pair a guest sends harmless.
        let startup = Startup::new(Phase::Running);

        assert!(!startup.offered(PAGE));
        assert_eq!(startup.phase(), Phase::Running);
        assert_eq!(startup.started(), None);
    }

    #[test]
    fn the_last_page_offered_is_the_one_the_processor_starts_at() {
        // One register carries the page, so the last message written to it is the
        // one the guest means.
        let startup = Startup::new(Phase::WaitingForSipi(None));

        assert!(startup.offered(PAGE));
        assert!(startup.offered(LATER));
        assert_eq!(startup.started(), Some(LATER));
    }

    #[test]
    fn a_held_processor_wakes_for_a_reset_or_a_page_and_for_nothing_else() {
        for phase in PHASES {
            let startup = Startup::new(phase);

            assert_eq!(
                startup.signalled(),
                phase != Phase::WaitingForSipi(None),
                "{phase:?}"
            );
            assert_eq!(startup.running(), phase == Phase::Running, "{phase:?}");
        }
    }

    #[test]
    fn no_start_up_is_taken_while_an_init_is_outstanding() {
        // Every pair of operations, in both orders, from every phase — which is
        // every interleaving of two concurrent operations there is, because each
        // of them is one step. What is asserted is the property all three races
        // broke: a processor with a reset it has not applied cannot be started,
        // and cannot be found running.
        for phase in PHASES {
            for first in STEPS {
                for second in STEPS {
                    let startup = Startup::new(phase);
                    let mut outstanding = matches!(phase, Phase::InitRequested(_));
                    for played in [first, second] {
                        let started = step(&startup, played);
                        assert!(
                            !(outstanding && started.is_some()),
                            "{phase:?} then {first:?} then {second:?} started at \
                             {started:?} with a reset outstanding"
                        );
                        outstanding = match played {
                            Step::Init => true,
                            Step::Reset { .. } => false,
                            Step::Startup(_) | Step::Take => outstanding,
                        };
                    }
                    assert!(
                        !(outstanding && startup.running()),
                        "{phase:?} then {first:?} then {second:?} is running with a \
                         reset outstanding"
                    );
                }
            }
        }
    }
}
