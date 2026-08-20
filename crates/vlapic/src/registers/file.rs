//! The controller itself: how one is built, how one is seeded from the hardware
//! it stands for, and the reset count every other module publishes against.
//!
//! The fields the rest of this tree reaches through named operations are
//! declared in [`crate::registers`]; what is here is the three things that
//! touch nearly all of them at once — construction, seeding and reset — plus
//! the four facts about a controller that are fixed when it is built and cannot
//! change afterwards.

use core::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};

use apic::{Extended, LocalState};
use cpu::{ApicId, CpuIndex};
use x86_64::instructions::interrupts;

use crate::{
    hardware::model::Model,
    lifecycle::ledger::Ledger,
    machine::diagnostics::Diagnostics,
    registers::{
        Phase, Startup, Vlapic,
        base::ApicBase,
        bitmap::Bitmap,
        error::ErrorStatus,
        icr::Command,
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
    ///
    /// `extended` is what the machine's real controllers offer above their
    /// architectural registers, and it decides one thing only: which of the two
    /// ledgers this controller settles its debts through. It is not part of
    /// [`Model`] because it is not part of the controller the guest is given —
    /// the guest is told the space is absent, and that is what keeps the
    /// registers this hypervisor uses out of its reach.
    pub(crate) fn new(
        index: CpuIndex,
        apic_id: ApicId,
        bootstrap: bool,
        startable: bool,
        model: Model,
        extended: Extended,
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
            external: Bitmap::new(),
            task_priority: AtomicU32::new(0),
            logical_destination: AtomicU32::new(0),
            destination_format: AtomicU32::new(FLAT_DESTINATION_FORMAT),
            spurious: AtomicU32::new(SPURIOUS_RESET),
            lvt: [const { AtomicU32::new(Entry::RESET) }; Entry::COUNT],
            timer_divide: AtomicU32::new(0),
            timer_initial: AtomicU32::new(0),
            timer_frequency: AtomicU64::new(0),
            command: AtomicU64::new(0),
            errors: ErrorStatus::new(),
            ledger: Ledger::new(extended),
            epoch: AtomicU64::new(0),
            startup: Startup::new(Phase::Running),
            away: AtomicBool::new(false),
            nmi: AtomicU8::new(0),
            owned: AtomicBool::new(false),
            avic_inhibited: AtomicBool::new(false),
            diagnostics: Diagnostics::new(),
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
    ///
    /// Every value is narrowed to what the register may hold, and that is not
    /// the same thing as narrowing it to what a *guest* may write: two of these
    /// registers have reserved fields that read as ones or as zeroes whoever
    /// wrote them, and the interrupt command register has a whole face's worth
    /// of them. What came out of hardware is by definition what the register
    /// holds, and what a guest reads back has to be something the architecture
    /// says that register can answer with — so firmware's reserved bits are
    /// dropped here rather than carried through to a guest's first read.
    fn seeded(&self, firmware: &LocalState, base: u64) {
        let base = ApicBase::seeded(base, self.base().bootstrap());
        self.base.store(base.bits(), Ordering::Release);
        self.task_priority.store(
            firmware.task_priority & TASK_PRIORITY_MASK,
            Ordering::Release,
        );
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
        // Narrowed in the face the controller is being seeded into, because that
        // is what decides how wide the destination half is. Left whole, this is
        // the one register that hands a guest firmware's reserved bits — the
        // delivery-status bit above all, which a guest reading the register and
        // writing back what it read would then be faulted for.
        self.command.store(
            firmware.command & Command::seedable(base.mode()),
            Ordering::Release,
        );
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

    /// What has happened to this controller, and what it has already said about
    /// it.
    ///
    /// Deliberately outside everything [`Vlapic::reset_registers`] clears. It
    /// is not register file: it is the record of what this *processor's*
    /// controller has done since the machine came up, which is what a
    /// machine with no serial port is debugged from, and an `INIT` is not
    /// an event that makes any of it untrue. Clearing the latches with the
    /// register file would also hand a guest that resets its own processor
    /// in a loop the log flood they exist to stop.
    pub(crate) const fn diagnostics(&self) -> &Diagnostics {
        &self.diagnostics
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
        self.external.reset();
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
        self.command.store(0, Ordering::Release);
        self.errors.reset();
        // A reset is one of the two boundaries at which a demotion is
        // reconsidered: whatever the hardware-driven path reported is state
        // of the guest that was, and the guest that comes out of the reset
        // is entitled to the acceleration.
        self.avic_inhibited.store(false, Ordering::Release);
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

    /// The reset count as it stands, for deciding whether something built
    /// from this register file is stale.
    ///
    /// Read at an entry boundary by the processor the controller belongs to,
    /// where a reset can only be one this processor performed at an exit
    /// boundary — so the count cannot be moving as it is read.
    pub(crate) fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    /// Whether this controller has been demoted back to software delivery.
    pub(crate) fn avic_inhibited(&self) -> bool {
        self.avic_inhibited.load(Ordering::Acquire)
    }

    /// Demotes this controller back to software delivery.
    ///
    /// What an exit handler does when the hardware-driven path reports
    /// something it cannot answer for: the control block's enable bit follows
    /// at the next entry, and the guest keeps running on the software path
    /// with no discontinuity it can see.
    pub(crate) fn inhibit_avic(&self) {
        self.avic_inhibited.store(true, Ordering::Release);
    }

    /// Allows the acceleration again, at a boundary that re-establishes the
    /// reasons it was taken away.
    ///
    /// Called when the guest changes which face its controller answers
    /// through: the demotion is a statement about the guest that was, and a
    /// guest that walks its faces is entitled to have the decision remade.
    pub(crate) fn permit_avic(&self) {
        self.avic_inhibited.store(false, Ordering::Release);
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
