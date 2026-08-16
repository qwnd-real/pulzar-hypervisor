//! The controller itself: how one is built, how one is seeded from the hardware
//! it stands for, and the reset count every other module publishes against.
//!
//! The fields the rest of this tree reaches through named operations are
//! declared in [`crate::registers`]; what is here is the three things that
//! touch nearly all of them at once — construction, seeding and reset — plus
//! the four facts about a controller that are fixed when it is built and cannot
//! change afterwards.

use core::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};

use apic::LocalState;
use cpu::{ApicId, CpuIndex};
use x86_64::instructions::interrupts;

use crate::{
    hardware::model::Model,
    lifecycle::ledger::Ledger,
    registers::{
        Phase, Startup, Vlapic,
        base::ApicBase,
        bitmap::Bitmap,
        error::ErrorStatus,
        identity::{DESTINATION_FORMAT_MASK, FLAT_DESTINATION_FORMAT, LOGICAL_DESTINATION_MASK},
        interrupts::NOTHING_REPORTED,
        lvt::Entry,
        spurious::{SPURIOUS_RESET, SPURIOUS_WRITABLE},
        task_priority::TASK_PRIORITY_MASK,
        timer::TIMER_DIVIDE_MASK,
    },
};

impl Vlapic {
    /// A controller as a processor finds it coming out of reset.
    ///
    /// `startable` is firmware's answer to whether the processor may be brought
    /// up at all, which is not something the controller changes and not
    /// something any guest can change: a processor firmware described as
    /// unstartable is one no startup message may ever be put on real hardware
    /// for.
    pub(crate) fn new(
        index: CpuIndex,
        apic_id: ApicId,
        bootstrap: bool,
        startable: bool,
        model: Model,
    ) -> Self {
        let this = Self {
            index,
            apic_id,
            startable,
            model,
            base: AtomicU64::new(ApicBase::reset(bootstrap).bits()),
            request: Bitmap::new(),
            in_service: Bitmap::new(),
            trigger_mode: Bitmap::new(),
            task_priority: AtomicU32::new(0),
            logical_destination: AtomicU32::new(0),
            destination_format: AtomicU32::new(FLAT_DESTINATION_FORMAT),
            spurious: AtomicU32::new(SPURIOUS_RESET),
            lvt: [const { AtomicU32::new(Entry::RESET) }; Entry::COUNT],
            timer_divide: AtomicU32::new(0),
            timer_initial: AtomicU32::new(0),
            timer_frequency: AtomicU64::new(0),
            timer_clamp: AtomicU32::new(0),
            timer_clamp_reported: AtomicBool::new(false),
            timer_periodic_running: AtomicBool::new(false),
            command: AtomicU64::new(0),
            errors: ErrorStatus::new(),
            ledger: Ledger::new(),
            epoch: AtomicU64::new(0),
            startup: Startup::new(Phase::Running),
            away: AtomicBool::new(false),
            nmi: AtomicU8::new(0),
            owned: AtomicBool::new(false),
            reported: AtomicU64::new(NOTHING_REPORTED),
        };
        this.reset_registers();
        this
    }

    /// Puts this controller into the state firmware left the real one in.
    ///
    /// Called once, before the guest has run, on the controller belonging to
    /// the processor the capture was taken on. What it is for is that
    /// pulzar does not boot a fresh guest: it re-enters the firmware it
    /// found, and firmware expects its controller to hold what it left
    /// there. Every register below was read before anything overwrote it
    /// and would otherwise be gone — the real controller has since been
    /// masked, re-vectored and acknowledged out from under firmware by the
    /// host's own bring-up.
    ///
    /// # What is not seeded, and why
    ///
    /// The in-service bank. The host has already retired every bit of it on
    /// real hardware, and more decisively the guest does not resume inside
    /// firmware's interrupt handler — it resumes at a stub that calls back
    /// into firmware from the top. So nothing would ever acknowledge a
    /// seeded in-service bit, and the controller would refuse everything of
    /// that priority or lower for the rest of the machine's life.
    ///
    /// The request and trigger-mode banks *are* seeded, and cannot double up:
    /// the host retired what was in service and left what was merely
    /// requested where it was, so those vectors are still latched in the
    /// real controller too. When hardware delivers one, [`Vlapic::accept`]
    /// folds it into the bit already set exactly as hardware folds a
    /// repeated interrupt.
    ///
    /// The identifier and the version are not seeded either, and are not
    /// firmware's to give: the first is the real one this controller was built
    /// with, because every passed-through interrupt is routed by it, and the
    /// second describes the hardware behind this controller rather than what
    /// firmware saw.
    ///
    /// # Published as one step
    ///
    /// Bracketed by the reset count for the same reason
    /// [`Vlapic::reset_registers`] is, and not because anything is expected
    /// to be delivering here yet. The controllers are published to the
    /// machine before this runs, so what makes a lone store safe is an
    /// argument about interrupt masking two crates away — and the argument
    /// that would have to hold is the one the epoch exists to
    /// make unnecessary. An arrival that does race this publishes again into
    /// the seeded file, exactly as one racing a reset does.
    pub(crate) fn seed(&self, firmware: &LocalState, base: u64) {
        self.between_epochs(|| self.seeded(firmware, base));
    }

    /// The seeding itself, with nothing said about who else can see it.
    fn seeded(&self, firmware: &LocalState, base: u64) {
        self.base.store(
            ApicBase::seeded(base, self.base().bootstrap()).bits(),
            Ordering::Release,
        );
        self.task_priority.store(
            firmware.task_priority & TASK_PRIORITY_MASK,
            Ordering::Release,
        );
        // Stored raw rather than through the setters, which mask to what a guest
        // may write: these came out of hardware, so what they hold is by
        // definition what the register holds, and a guest reading one back has to
        // find firmware's value rather than a narrowed one.
        self.logical_destination.store(
            firmware.logical_destination & LOGICAL_DESTINATION_MASK,
            Ordering::Release,
        );
        self.destination_format.store(
            (firmware.destination_format & DESTINATION_FORMAT_MASK) | !DESTINATION_FORMAT_MASK,
            Ordering::Release,
        );
        self.spurious
            .store(firmware.spurious & SPURIOUS_WRITABLE, Ordering::Release);
        for (entry, value) in Entry::ALL.into_iter().zip(firmware.lvt()) {
            self.lvt[entry.index()].store(value & entry.writable(self.model), Ordering::Release);
        }
        self.timer_divide
            .store(firmware.timer_divide & TIMER_DIVIDE_MASK, Ordering::Release);
        self.timer_initial
            .store(firmware.timer_initial_count, Ordering::Release);
        self.command.store(firmware.command, Ordering::Release);
        self.errors.seed(firmware.error_status);
        self.request.seed(&firmware.interrupt_request);
        self.trigger_mode.seed(&firmware.trigger_mode);
    }

    /// Where this controller sits in the roster.
    pub(crate) const fn index(&self) -> CpuIndex {
        self.index
    }

    /// The identifier interrupts to this processor are addressed by, which is
    /// the real one.
    pub(crate) const fn apic_id(&self) -> ApicId {
        self.apic_id
    }

    /// Whether firmware described this processor as one that may be started.
    ///
    /// Fixed when the controller was built. A processor firmware marked
    /// unstartable is never brought up by the host and never will be, so the
    /// window in which a startup message for it belongs on real hardware does
    /// not exist.
    pub(crate) const fn startable(&self) -> bool {
        self.startable
    }

    /// The controller this guest was told it has.
    pub(crate) const fn model(&self) -> Model {
        self.model
    }

    /// The error status register and its write-then-read protocol.
    pub(crate) const fn errors(&self) -> &ErrorStatus {
        &self.errors
    }

    /// What real hardware is holding in service on this guest's behalf.
    pub(crate) const fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    /// Everything an INIT leaves behind, which is everything but the
    /// identifier, the version and the face in use.
    ///
    /// Bracketed by the reset count so that everything cleared here is cleared
    /// as one step as far as any deliverer is concerned. A processor delivering
    /// into this controller while it runs publishes against the count, sees it
    /// move, and publishes again — so no interrupt is left with its request bit
    /// set and its trigger mode cleared, which is the state that costs a real
    /// acknowledgement and kills a line.
    ///
    /// What is *not* cleared here is the count of non-maskable interrupts the
    /// processor has already been delivered. That is not register file, and the
    /// transitions that discard it are not the ones that reset a register file:
    /// see [`Vlapic::discard_nmi`].
    pub(crate) fn reset_registers(&self) {
        self.between_epochs(|| self.clear());
    }

    /// The clearing itself, with nothing said about who else can see it.
    fn clear(&self) {
        self.request.reset();
        self.in_service.reset();
        self.trigger_mode.reset();
        self.task_priority.store(0, Ordering::Release);
        self.logical_destination.store(0, Ordering::Release);
        self.destination_format
            .store(FLAT_DESTINATION_FORMAT, Ordering::Release);
        self.spurious.store(SPURIOUS_RESET, Ordering::Release);
        for entry in &self.lvt {
            entry.store(Entry::RESET, Ordering::Release);
        }
        self.timer_divide.store(0, Ordering::Release);
        self.timer_initial.store(0, Ordering::Release);
        self.clear_timer_clamp();
        self.set_timer_periodic_running(false);
        self.command.store(0, Ordering::Release);
        self.errors.reset();
    }

    /// Publishes a set of stores as one step, as far as anything delivering
    /// into this controller is concerned.
    ///
    /// The reset count is odd for exactly as long as `publishing` runs, and a
    /// deliverer that sees it move republishes into whatever was left behind.
    ///
    /// Host interrupts are held off for the whole of it, and that is not
    /// belt-and-braces either: both callers run on the processor the controller
    /// belongs to, with interrupts enabled, and a physical interrupt landing
    /// inside the odd window reaches [`Vlapic::accept`] *on the same processor*
    /// — where no amount of waiting can finish the stores it interrupted,
    /// because the frame that would finish them is the one it interrupted.
    /// It would spin out its patience and drop the interrupt. A
    /// non-maskable interrupt arriving there is not held off and does not
    /// need to be: it is counted, in one atomic, by a path that never
    /// consults the count.
    fn between_epochs(&self, publishing: impl FnOnce()) {
        interrupts::without_interrupts(|| {
            self.epoch.fetch_add(1, Ordering::SeqCst);
            publishing();
            self.epoch.fetch_add(1, Ordering::SeqCst);
        });
    }

    /// The reset count, once no reset is in progress.
    ///
    /// Odd means one is running, and one is only ever run by the processor the
    /// controller belongs to, with interrupts held off — so an odd count is
    /// always a *remote* processor part-way through a few dozen stores, and the
    /// wait is that long and no longer. `None` says one has been running for
    /// longer than that can take, which is a broken invariant rather than
    /// contention.
    pub(super) fn settled_epoch(&self) -> Option<u64> {
        (0..EPOCH_SPINS).find_map(|_| {
            let epoch = self.epoch.load(Ordering::SeqCst);
            if epoch.is_multiple_of(2) {
                return Some(epoch);
            }
            core::hint::spin_loop();
            None
        })
    }
}

/// How many times a deliverer re-reads a reset count that says a reset is
/// running before giving up on it.
///
/// Generous by design and not a tuned number: what it waits for is another
/// processor finishing a few dozen stores with interrupts held off, so a
/// deliverer that exhausts it has not met contention — it has met a processor
/// that stopped part-way through one, and no larger number would help.
const EPOCH_SPINS: u32 = 100_000;
