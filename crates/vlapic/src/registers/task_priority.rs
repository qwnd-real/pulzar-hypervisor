//! The three priority registers, of which the guest writes one.
//!
//! The task priority is the only one software sets; the processor and
//! arbitration priorities are computed from it and from the bitmaps on every
//! read, because the architecture defines them as functions of that state and a
//! stored copy is a copy that can be wrong. The arithmetic itself is
//! [`crate::priority`]'s; what is here is only which state each rule is applied
//! to.

use core::sync::atomic::Ordering;

use crate::{
    priority::{self, Priority},
    registers::Vlapic,
};

impl Vlapic {
    /// The task priority the guest has set.
    pub(crate) fn task_priority(&self) -> Priority {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "only the low byte of the task priority register holds anything; the rest is reserved"
        )]
        Priority::new(self.task_priority.load(Ordering::Acquire) as u8)
    }

    /// Sets the task priority.
    pub(crate) fn set_task_priority(&self, value: u32) {
        self.task_priority
            .store(value & TASK_PRIORITY_MASK, Ordering::Release);
    }

    /// Records the priority class the guest set through its control register.
    ///
    /// The control block carries only the four bits of the class, because that
    /// is all the control register carries — so what survives a world switch is
    /// the class and nothing else, and this stores the class over the whole
    /// byte.
    ///
    /// That is wider than a control-register write would be. A write to that
    /// register really does clear the task priority's subclass, but this runs
    /// on *every* exit, including exits that followed a write to the
    /// emulated register itself: a guest that stores `0x35` through its
    /// controller reads `0x30` back, because its read is an exit and the
    /// exit gets here first. The deviation is deliberate and matches what a
    /// hypervisor using this processor's task-priority virtualization can
    /// do. Nothing delivered depends on it —
    /// [`crate::priority::deliverable`] compares classes only —
    /// so what a guest loses is four bits of a register it wrote and no
    /// behaviour.
    ///
    /// `class` is those four bits, and turning them back into a priority is
    /// [`Priority::of_class`]'s so that the workspace has one statement of
    /// where a class sits in the byte — the same one the crates that write
    /// `V_INTR_PRIO` and `V_TPR` use.
    pub(crate) fn observe_task_priority(&self, class: u8) {
        self.task_priority
            .store(Priority::of_class(class).get().into(), Ordering::Release);
    }

    /// The priority this controller is actually servicing at.
    ///
    /// Both halves of it: what the guest has asked for through its task
    /// priority and what it is already handling. The delivery path does not
    /// go through this — it takes one look at the register file and
    /// computes both halves from it — so what is left here is the guest's
    /// own read of the register.
    pub(crate) fn processor_priority(&self) -> Priority {
        priority::processor_priority(self.task_priority(), self.in_service.highest())
    }

    /// The arbitration priority, which exists only in the older face.
    ///
    /// Vestigial on every processor this hypervisor can run on — nothing
    /// arbitrates over a bus and [`crate::delivery`] chooses a redirectable
    /// interrupt's target without asking — and computed anyway, because the
    /// register is readable and a guest that reads it is owed the number its
    /// own processor would have produced.
    pub(crate) fn arbitration_priority(&self) -> Priority {
        priority::arbitration_priority(
            self.task_priority(),
            self.in_service.highest(),
            self.request.highest(),
        )
    }
}

/// The task priority register's reserved bits are everything above the low
/// byte.
///
/// Reached by the model-specific-register face as well as by the store here,
/// because that face has to fault on exactly the bits this drops: a bit a guest
/// may not set has to be refused there and discarded here, and two spellings of
/// one mask would eventually disagree about which.
pub(crate) const TASK_PRIORITY_MASK: u32 = 0xFF;
