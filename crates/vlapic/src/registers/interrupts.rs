//! What the controller has been given, what it has handed over, and what it is
//! still holding.
//!
//! The three bitmaps and the operations that move a vector between them. Which
//! processor may perform which is the division of labour [`crate::registers`]
//! states: any processor may accept an interrupt into a controller, and only
//! the processor the controller belongs to may take one out again.

use core::sync::atomic::Ordering;

use descriptors::Vector;
use log::{Level, log_enabled, trace};

use crate::{
    lifecycle::ledger::InService,
    priority::{self, Priority},
    registers::{Vlapic, error::Errors, icr::Trigger},
};

impl Vlapic {
    /// One slot of the interrupt-request register, as the guest reads it, or
    /// `None` for a slot the register does not have.
    pub(crate) fn request_slot(&self, slot: usize) -> Option<u32> {
        self.request.slot(slot)
    }

    /// One slot of the in-service register.
    pub(crate) fn in_service_slot(&self, slot: usize) -> Option<u32> {
        self.in_service.slot(slot)
    }

    /// One slot of the trigger-mode register.
    pub(crate) fn trigger_mode_slot(&self, slot: usize) -> Option<u32> {
        self.trigger_mode.slot(slot)
    }

    /// Accepts an interrupt into this controller, from anywhere.
    ///
    /// This is what delivery means, and it is the one operation any processor
    /// may perform on any controller. The trigger mode is recorded before the
    /// request, so that a processor which observes the request bit cannot then
    /// read a trigger mode from before it was set — the difference between the
    /// two decides whether an acknowledgement is owed to real hardware, and
    /// getting it wrong is a line that never fires again.
    ///
    /// Both are published against the reset count, and republished if a reset
    /// moved it. An interrupt that raced a reset is one whose moment is
    /// indistinguishable from just after it, and just after it is when the
    /// guest the reset produced is entitled to see it — so the retry
    /// converges on the new state rather than leaving a half-written event
    /// in the old one.
    ///
    /// Answers what became of it. A vector already requested and not yet
    /// accepted collapses into the one bit, exactly as hardware does, and is
    /// not a second interrupt.
    #[must_use]
    pub(crate) fn accept(&self, vector: Vector, trigger: Trigger) -> Accepted {
        // The controller never sets a request bit in the illegal range, and
        // records that it was asked to.
        if !priority::legal(vector) {
            self.errors.record(Errors::RECEIVE_ILLEGAL_VECTOR);
            return Accepted::Illegal;
        }
        // A controller that is switched off or software-disabled does not accept
        // interrupts. The special messages that reach a disabled controller
        // anyway — INIT, start-up, a non-maskable interrupt — do not come
        // through here.
        if !self.accepting() {
            return Accepted::Refused;
        }
        for _ in 0..PUBLISHES {
            let Some(epoch) = self.settled_epoch() else {
                // A reset that never finishes is a broken invariant rather than
                // contention, and spinning in a delivery path would take the
                // sender down with it.
                return Accepted::Resetting;
            };
            match trigger {
                Trigger::Level => self.trigger_mode.set(vector),
                Trigger::Edge => self.trigger_mode.clear(vector),
            };
            let coalesced = self.request.set(vector);
            if self.epoch.load(Ordering::SeqCst) == epoch {
                return if coalesced {
                    Accepted::Coalesced
                } else {
                    Accepted::Requested
                };
            }
        }
        Accepted::Resetting
    }

    /// What this controller has for its guest, left where it is.
    ///
    /// Deliberately does not consume anything. Whether the guest can actually
    /// be given an interrupt is not this controller's to know — the
    /// processor may have an event already being delivered, or a
    /// non-maskable interrupt that outranks this, or a closed interrupt
    /// window — and a controller that moved a vector out of the request
    /// register for an injection that then did not happen would have lost
    /// it. So this only nominates, and [`Vlapic::committed`] is what
    /// accounts for one that went in.
    ///
    /// Both answers come out of one [`Look`] at the register file, and that is
    /// not only a saving. They are two comparisons against the same requested
    /// vector, the same in-service bank and the same task priority, so reading
    /// the file twice would let a vector be admitted by one and refused by the
    /// other on state that moved in between — and the caller puts the two in
    /// the control block together.
    pub(crate) fn nominate(&self) -> Nomination {
        let look = self.look();
        let nomination = look.nomination();
        self.report(&look, nomination);
        nomination
    }

    /// Everything a nomination is decided from, read once.
    ///
    /// One scan of each bitmap and one load of each register. Every scan spans
    /// eight independent words and no single atomic operation covers them, so
    /// this is a snapshot rather than a linearisation point — see
    /// [`crate::registers::bitmap::Bitmap::highest`] for what that does and
    /// does not promise.
    fn look(&self) -> Look {
        Look {
            accepting: self.accepting(),
            requested: self.request.highest(),
            in_service: self.in_service.highest(),
            task: self.task_priority(),
        }
    }

    /// Says why this controller is nominating what it is, the first time it
    /// reaches any given answer.
    ///
    /// A controller that has stopped delivering says so once and then goes
    /// quiet: what makes an interrupt undeliverable is state that has to change
    /// before it becomes deliverable again, and the change is what gets
    /// reported. The three ways a nomination comes to nothing are
    /// indistinguishable to the caller, and they are three different faults — a
    /// controller its guest switched off, a controller with nothing to give,
    /// and a controller holding something back behind a priority that never
    /// falls.
    ///
    /// Nothing here runs unless the level it logs at is enabled, and the
    /// comparison that suppresses a repeat is inside that guard rather than
    /// outside it. It is a read-modify-write on a line every processor
    /// delivering into this controller loads, and this is called on every entry
    /// — so at a level that discards the line it would be a cache line stolen
    /// from every interrupt sender on the machine to produce nothing.
    fn report(&self, look: &Look, nomination: Nomination) {
        if !log_enabled!(Level::Trace) {
            return;
        }
        // Everything the report below names, packed into one word so that
        // "has this changed" is a single comparison rather than a lock. Each of
        // the three vectors gets nine bits, because "nothing" has to be as
        // distinguishable as any vector is and there are two hundred and
        // fifty-six of those.
        let processor = look.processor_priority();
        let bits = u64::from(look.accepting)
            | u64::from(number(look.requested)) << 1
            | u64::from(number(look.in_service)) << 10
            | u64::from(number(nomination.deliverable)) << 19
            | u64::from(look.task.get()) << 32
            | u64::from(processor.get()) << 40;
        if self.reported.swap(bits, Ordering::Relaxed) == bits {
            return;
        }
        let (task, in_service) = (look.task.get(), look.in_service);
        match (look.accepting, look.requested, nomination.deliverable) {
            (false, _, _) => trace!(
                "vlapic: {} is not accepting interrupts: mode {:?}, software {}",
                self.index(),
                self.mode(),
                if self.software_enabled() {
                    "enabled"
                } else {
                    "disabled"
                }
            ),
            (true, None, _) => trace!(
                "vlapic: {} has nothing requested; in service {in_service:?}, task priority \
                 {task:#x}",
                self.index()
            ),
            (true, Some(vector), None) => trace!(
                "vlapic: {} is holding {vector} back: processor priority {:#x} from task \
                 {task:#x} and in service {in_service:?}",
                self.index(),
                processor.get()
            ),
            (true, Some(_), Some(vector)) => trace!(
                "vlapic: {} nominates {vector}, processor priority {:#x}, in service \
                 {in_service:?}",
                self.index(),
                processor.get()
            ),
        }
    }

    /// Records that the guest really has been given `vector`, moving it from
    /// requested to in service.
    ///
    /// Called only after the injection has been established to have happened,
    /// and only by the processor this controller belongs to — which is the only
    /// one that ever clears a request bit.
    ///
    /// The request bit is cleared before the in-service bit is set, matching
    /// the controller's transition order. A concurrent reader can briefly see
    /// neither bit. Answers whether the vector really was still requested: a
    /// reset between the selection and the commitment leaves nothing to move,
    /// and nothing is then put in service.
    pub(crate) fn committed(&self, vector: Vector) -> bool {
        if !self.request.clear(vector) {
            return false;
        }
        self.in_service.set(vector);
        true
    }

    /// Whether anything is requested at all, whatever its priority.
    pub(crate) fn requested(&self) -> Option<Vector> {
        self.request.highest()
    }

    /// Acknowledges the interrupt the guest is servicing, and says which it
    /// was.
    ///
    /// Retires the highest in-service bit, which is the one the guest must have
    /// been handling: interrupts nest by priority, so the most recently taken
    /// is always the highest.
    ///
    /// Releasing the debt is part of the same operation rather than something a
    /// caller does afterwards, because the two must not come apart: a guest's
    /// acknowledgement is exactly the event that makes an acknowledgement to
    /// real hardware permissible, and nothing else ever will be.
    ///
    /// The trigger-mode bit is deliberately left where it is. Hardware writes
    /// that register when it accepts an interrupt and reads it when one is
    /// acknowledged; it does not clear it, and a guest reading the bank back
    /// after an acknowledgement has to find what hardware would have left. That
    /// is a readback and nothing more today — the level or edge decision this
    /// crate acts on comes from the *real* controller's own record, which is
    /// the one authority on how an interrupt actually arrived. Anything
    /// that made this register the oracle instead would make the order the
    /// bank is published in load-bearing, which is the reason
    /// [`crate::registers::bitmap`] gives for its orderings.
    ///
    /// `controller` is the real controller of the processor whose guest is
    /// acknowledging, which is this processor: a guest's acknowledgement comes
    /// out of the guest, and the guest runs nowhere else.
    #[must_use]
    pub(crate) fn end_of_interrupt(&self, controller: &impl InService) -> Option<Vector> {
        let vector = self.in_service.take_highest()?;
        self.ledger.release(vector, controller);
        Some(vector)
    }

    /// How many interrupts are requested and not yet taken.
    pub(crate) fn requested_count(&self) -> u32 {
        self.request.count()
    }

    /// How many the guest has taken and not yet acknowledged.
    pub(crate) fn in_service_count(&self) -> u32 {
        self.in_service.count()
    }
}

/// What a controller has for its guest, decided from one look at its register
/// file.
///
/// Two answers to two different questions, and the difference between them is
/// the one comparison this hypervisor must not be the one to make.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Nomination {
    /// The highest-priority interrupt the guest should take now, or `None` when
    /// nothing is requested, when what is requested does not outrank what the
    /// guest is already servicing, or when the guest's own task priority holds
    /// it back. The request stays pending either way, which is what makes a
    /// priority a filter rather than a discard.
    pub deliverable: Option<Vector>,
    /// The highest-priority interrupt nothing *but* the guest's own task
    /// priority may be holding back, which is what an interrupt window is armed
    /// for.
    ///
    /// A superset of [`Nomination::deliverable`], and deliberately so. A guest
    /// changes its task priority through its control register without exiting —
    /// the processor keeps the value in the control block — so a vector this
    /// crate ruled out on a priority it read at the last exit would stay ruled
    /// out however far the guest lowered that priority afterwards, and nothing
    /// would ever ask again. Arming the window for this hands the comparison to
    /// the hardware that owns the register, and the guest lowering its priority
    /// is what produces the exit.
    ///
    /// What is already in service is applied here rather than left to hardware
    /// too, and that is what keeps the arrangement from spinning: the control
    /// block carries only the task priority, so a vector armed while an
    /// interrupt of its own class or higher is still in service would have the
    /// processor report a window the guest cannot take anything through, and
    /// every exit would arm it again.
    pub blocked: Option<Vector>,
}

/// Everything one [`Nomination`] is decided from, read once.
struct Look {
    accepting: bool,
    requested: Option<Vector>,
    in_service: Option<Vector>,
    task: Priority,
}

impl Look {
    /// What this look nominates.
    fn nomination(&self) -> Nomination {
        let Some(vector) = self.requested.filter(|_| self.accepting) else {
            return Nomination::default();
        };
        Nomination {
            deliverable: priority::deliverable(vector, self.processor_priority()).then_some(vector),
            blocked: priority::deliverable(vector, self.servicing()).then_some(vector),
        }
    }

    /// The priority the guest is running at, both halves of it.
    fn processor_priority(&self) -> Priority {
        priority::processor_priority(self.task, self.in_service)
    }

    /// The half of that priority the interrupts this controller has already
    /// accepted impose, with the guest's task priority left out of it.
    fn servicing(&self) -> Priority {
        priority::processor_priority(Priority::NONE, self.in_service)
    }
}

/// What became of an interrupt offered to a controller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Accepted {
    /// Newly requested.
    Requested,
    /// Already requested and not yet accepted, so it folded into the one bit.
    Coalesced,
    /// Named a vector no controller may deliver, and was refused.
    Illegal,
    /// Offered to a controller that is not accepting interrupts.
    Refused,
    /// The register file was being reset underneath it and it was not
    /// published.
    ///
    /// Kept apart from [`Accepted::Refused`] because that one is the refusal
    /// the architecture defines — a controller its guest switched off,
    /// saying no to something it is entitled to say no to — and this one is
    /// this hypervisor failing to record an interrupt it was given. A
    /// caller that treated them alike would issue the real acknowledgement
    /// a level-triggered interrupt is withholding, on the grounds that the
    /// guest will never acknowledge it, while the line that raised it is
    /// still asserted.
    Resetting,
}

/// How many times an interrupt is published into a register file that is being
/// reset underneath it before it is given up on.
///
/// Two, and the second cannot be raced by the reset that raced the first: a
/// reset is a bounded run of stores by the processor the controller belongs to,
/// with interrupts held off, so by the time the second publish begins the reset
/// the first one lost to has finished. Only a *new* reset can move the count
/// again, and a caller retrying past that would be waiting on a guest that
/// resets its own processor in a loop — which is bounded work that never ends,
/// on a path entered from an interrupt handler.
const PUBLISHES: u32 = 2;

/// What [`Vlapic::reported`] holds before any selection has been reported.
///
/// Not a state any real packing produces, so the first selection always reports
/// however trivial it is.
pub(super) const NOTHING_REPORTED: u64 = u64::MAX;

/// A vector's number, widened to leave room for a value no vector has.
///
/// Used to pack an optional vector into the reported selection state, where
/// "nothing" has to be as distinguishable as any vector is — so it is the
/// ninth bit rather than a value inside the eight a vector occupies, which
/// would be one vector that could not be told from nothing at all.
const fn number(vector: Option<Vector>) -> u16 {
    match vector {
        Some(vector) => vector.number() as u16,
        None => NO_VECTOR,
    }
}

/// What [`number`] answers when there is no vector, which is the one value
/// outside a vector's range that nine bits hold.
const NO_VECTOR: u16 = 1 << 8;
