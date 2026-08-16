//! What the controller has been given, what it has handed over, and what it is
//! still holding.
//!
//! The three bitmaps and the operations that move a vector between them. Which
//! processor may perform which is the division of labour [`crate::registers`]
//! states: any processor may accept an interrupt into a controller, and only
//! the processor the controller belongs to may take one out again.

use core::sync::atomic::Ordering;

use descriptors::Vector;
use log::trace;

use crate::{
    lifecycle::ledger::InService,
    priority,
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

    /// The highest-priority interrupt the guest should take now, left where it
    /// is.
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
    /// Answers `None` when nothing is requested, or when what is requested does
    /// not outrank what the guest is already servicing — in which case the
    /// request stays pending, which is what makes a task priority a filter
    /// rather than a discard.
    pub(crate) fn select(&self) -> Option<Vector> {
        let selected = self.select_inner();
        self.report_selection(selected);
        selected
    }

    /// The highest-priority interrupt nothing but the guest's task priority may
    /// be holding back.
    ///
    /// A superset of [`Vlapic::select`], and the difference between them is the
    /// one comparison this hypervisor must not be the one to make. A guest
    /// changes its task priority through its control register without exiting —
    /// the processor keeps the value in the control block — so a vector this
    /// crate ruled out on a task priority it read at the last exit would stay
    /// ruled out however far the guest lowered that priority afterwards, and
    /// nothing would ever ask again.
    ///
    /// So the two halves of the processor priority are split. What is already
    /// in service is applied here, because it moves only when the guest
    /// acknowledges an interrupt and that always exits. The task priority is
    /// left to the hardware that owns it: this is what an interrupt window is
    /// armed for, and the processor raises one exactly when the guest's own
    /// priority admits the vector.
    ///
    /// Applying the in-service half here rather than leaving both to hardware
    /// is what keeps that arrangement from spinning. The control block
    /// carries only the task priority, so a vector armed while an interrupt
    /// of its own class or higher is still in service would have the
    /// processor report a window the guest cannot actually take anything
    /// through, and every exit would arm it again.
    pub(crate) fn pending(&self) -> Option<Vector> {
        if !self.accepting() {
            return None;
        }
        let vector = self.request.highest()?;
        priority::deliverable(vector, self.servicing()).then_some(vector)
    }

    /// [`Vlapic::select`] proper, with nothing said about what it decided.
    fn select_inner(&self) -> Option<Vector> {
        if !self.accepting() {
            return None;
        }
        let vector = self.request.highest()?;
        priority::deliverable(vector, self.processor_priority()).then_some(vector)
    }

    /// Says why this controller is nominating what it is, the first time it
    /// reaches any given answer.
    ///
    /// A controller that has stopped delivering says so once and then goes
    /// quiet, so this costs nothing on the path it sits on: what makes an
    /// interrupt undeliverable is state that has to change before it becomes
    /// deliverable again, and the change is what gets reported. The three ways
    /// a nomination comes to nothing are indistinguishable to the caller, and
    /// they are three different faults — a controller its guest switched off,
    /// a controller with nothing to give, and a controller holding something
    /// back behind a priority that never falls.
    fn report_selection(&self, selected: Option<Vector>) {
        let accepting = self.accepting();
        let requested = self.request.highest();
        let in_service = self.in_service.highest();
        let task = self.task_priority();
        let processor = self.processor_priority();
        // Everything the report below names, packed into one word so that
        // "has this changed" is a single comparison rather than a lock. Each of
        // the three vectors gets nine bits, because "nothing" has to be as
        // distinguishable as any vector is and there are two hundred and
        // fifty-six of those.
        let bits = u64::from(accepting)
            | u64::from(number(requested)) << 1
            | u64::from(number(in_service)) << 10
            | u64::from(number(selected)) << 19
            | u64::from(task.get()) << 32
            | u64::from(processor.get()) << 40;
        if self.reported.swap(bits, Ordering::Relaxed) == bits {
            return;
        }
        match (accepting, requested, selected) {
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
                "vlapic: {} has nothing requested; in service {in_service:?}, task priority {:#x}",
                self.index(),
                task.get()
            ),
            (true, Some(vector), None) => trace!(
                "vlapic: {} is holding {vector} back: processor priority {:#x} from task {:#x} \
                 and in service {in_service:?}",
                self.index(),
                processor.get(),
                task.get()
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
    /// `controller` is the real controller of the processor whose guest is
    /// acknowledging, which is this processor: a guest's acknowledgement comes
    /// out of the guest, and the guest runs nowhere else.
    pub(crate) fn end_of_interrupt(&self, controller: &impl InService) -> Option<Vector> {
        let vector = self.in_service.take_highest()?;
        self.trigger_mode.clear(vector);
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
