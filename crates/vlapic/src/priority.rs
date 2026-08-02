//! What the controller compares before it delivers anything.
//!
//! Every priority the local controller deals in is one byte: bits 7:4 are the
//! interrupt-priority class and bits 3:0 rank a vector within its class. A
//! vector *is* its own priority — the number the guest programmed into a local
//! vector table entry or an interrupt command is read directly as those two
//! nibbles — which is why the highest-numbered pending vector is also the
//! highest-priority one and a descending scan of a bitmap answers both
//! questions at once.
//!
//! Class 0, the vectors 0 through 15, is illegal: the architecture reserves
//! those numbers and the controller delivers none of them. That is not merely a
//! consequence of the arithmetic below and is written out as [`legal`].
//!
//! The task-priority register is 32 bits wide with 31:8 reserved, and the
//! arbitration- and processor-priority registers are the same shape. Only the
//! low byte carries meaning, so everything here is a [`Priority`] and the
//! reserved bits belong to the register-access layer, not to the arithmetic.
//!
//! Nothing here holds state. The processor- and arbitration-priority registers
//! are *computed* on every read from the task priority and the pending bitmaps
//! rather than stored, so this module is pure functions and the architecture's
//! rules are written down exactly once.

use core::fmt::{self, Display, Formatter};

use descriptors::Vector;

/// One priority byte: an interrupt-priority class in bits 7:4 and a rank within
/// that class in bits 3:0.
///
/// Ordering is the ordering of the whole byte, which is the architecture's
/// ordering of priorities. It is *not* the ordering that decides delivery —
/// see [`deliverable`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Priority(u8);

impl Priority {
    /// The lowest priority there is, which is what reset leaves the task
    /// priority at and what an empty bitmap stands in for.
    pub(crate) const NONE: Self = Self(0);

    /// The priority a byte denotes.
    pub(crate) const fn new(value: u8) -> Self {
        Self(value)
    }

    /// A vector's own priority, which is its number read as two nibbles.
    pub(crate) const fn of(vector: Vector) -> Self {
        Self(vector.number())
    }

    /// The interrupt-priority class, 0 through 15.
    pub(crate) const fn class(self) -> u8 {
        self.0 >> CLASS_SHIFT
    }

    /// The rank within the class, 0 through 15.
    pub(crate) const fn subclass(self) -> u8 {
        self.0 & SUBCLASS_MASK
    }

    /// The whole byte, as the guest reads it out of a register.
    pub(crate) const fn get(self) -> u8 {
        self.0
    }
}

impl Display for Priority {
    /// Both nibbles, because the class alone decides delivery and the subclass
    /// alone explains nothing.
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}:{}", self.class(), self.subclass())
    }
}

/// Whether a vector may be delivered at all.
///
/// Vectors 0 through 15 are class 0, which the architecture reserves; a guest
/// programming one of them into a local vector table entry or an interrupt
/// command has asked for something the controller must refuse rather than
/// deliver.
pub(crate) const fn legal(vector: Vector) -> bool {
    Priority::of(vector).class() != 0
}

/// The processor priority: the priority this processor is currently servicing
/// at, and the threshold everything is delivered against.
///
/// `in_service` is the highest vector whose in-service bit is set, or `None`
/// when none is. The architecture states the computation as three cases over
/// the two classes:
///
/// - task class greater: the whole task-priority byte,
/// - task class equal: the task class with the task subclass, which is again
///   the whole task-priority byte,
/// - task class lower: the serviced class with a zero subclass.
///
/// The first two cases produce the same byte, so they are one branch here.
pub(crate) fn processor_priority(task: Priority, in_service: Option<Vector>) -> Priority {
    let serviced = in_service.map_or(Priority::NONE, Priority::of);
    if task.class() >= serviced.class() {
        task
    } else {
        class_floor(serviced.class())
    }
}

/// The arbitration priority as Intel's P6 processors define it.
///
/// It described the priority a processor would bid with on the external APIC
/// bus during lowest-priority arbitration. No processor this hypervisor runs on
/// arbitrates over a bus, so the register is vestigial and computing it exists
/// to make a guest's read return the value its processor would have produced.
///
/// `in_service` and `requested` are the highest vectors with a bit set in the
/// in-service and interrupt-request registers, or `None` when the register is
/// empty. The fallback path combines the task and serviced classes with a
/// bitwise AND rather than a maximum: that is what the P6 definition specifies,
/// odd as it reads, and the two disagree whenever the classes share no bits.
pub(crate) fn intel_arbitration(
    task: Priority,
    in_service: Option<Vector>,
    requested: Option<Vector>,
) -> Priority {
    let serviced = in_service.map_or(Priority::NONE, Priority::of);
    let request = requested.map_or(Priority::NONE, Priority::of);
    if task.class() >= request.class() && task.class() > serviced.class() {
        task
    } else {
        class_floor((task.class() & serviced.class()).max(request.class()))
    }
}

/// The arbitration priority as AMD defines it.
///
/// The maximum of the three priorities rather than Intel's bitwise combination,
/// and the whole task-priority byte rather than a class floor whenever the task
/// priority is what wins — including when it merely ties, which is why the
/// comparison below is on classes and the result keeps the subclass.
///
/// The two definitions agree only by coincidence. A task class of 2 against a
/// serviced class of 4 gives 4 here and 0 under the P6 rule, because 2 and 4
/// share no bits.
pub(crate) fn amd_arbitration(
    task: Priority,
    in_service: Option<Vector>,
    requested: Option<Vector>,
) -> Priority {
    let serviced = in_service.map_or(Priority::NONE, Priority::of);
    let request = requested.map_or(Priority::NONE, Priority::of);
    let highest = serviced.class().max(request.class());
    if task.class() >= highest {
        task
    } else {
        class_floor(highest)
    }
}

/// Whether an interrupt at `vector` may be delivered against a processor
/// priority of `processor`.
///
/// Only the classes are compared, and the comparison is strict. The processor
/// priority's subclass takes no part in any delivery decision at all: the
/// architecture computes it solely so that a read of the register returns the
/// defined value, and treating a higher subclass within the same class as
/// blocking — or a lower one as admitting — is wrong in both directions.
///
/// The illegal class is rejected explicitly. It would also fall out of the
/// comparison, but only because a class is never negative, and that is an
/// accident of the encoding rather than the rule.
pub(crate) fn deliverable(vector: Vector, processor: Priority) -> bool {
    legal(vector) && Priority::of(vector).class() > processor.class()
}

/// The lowest priority in a class: that class with a zero subclass.
///
/// Every caller passes a class taken from [`Priority::class`], which is four
/// bits wide, so the shift discards nothing.
const fn class_floor(class: u8) -> Priority {
    Priority(class << CLASS_SHIFT)
}

/// How far the interrupt-priority class sits above the subclass.
const CLASS_SHIFT: u32 = 4;

/// The bits that rank a priority within its class.
const SUBCLASS_MASK: u8 = 0x0F;

#[cfg(test)]
mod tests {
    use descriptors::Vector;

    use super::{
        Priority, amd_arbitration, deliverable, intel_arbitration, legal, processor_priority,
    };

    #[test]
    fn processor_priority_is_the_task_priority_when_nothing_is_in_service() {
        let task = Priority::new(0x35);
        assert_eq!(processor_priority(task, None), task);
    }

    #[test]
    fn processor_priority_drops_the_subclass_when_only_a_vector_is_in_service() {
        let serviced = processor_priority(Priority::NONE, Some(Vector::new(0x43)));
        assert_eq!(serviced, Priority::new(0x40));
        assert_eq!(serviced.subclass(), 0);
    }

    #[test]
    fn processor_priority_keeps_the_task_subclass_within_one_class() {
        let task = Priority::new(0x45);
        assert_eq!(processor_priority(task, Some(Vector::new(0x4F))), task);
    }

    #[test]
    fn processor_priority_follows_the_in_service_vector_when_it_outranks_the_task() {
        let priority = processor_priority(Priority::new(0x25), Some(Vector::new(0x43)));
        assert_eq!(priority, Priority::new(0x40));
    }

    #[test]
    fn delivery_compares_classes_strictly() {
        let vector = Vector::new(0x30);
        assert!(!deliverable(vector, Priority::new(0x30)));
        assert!(!deliverable(vector, Priority::new(0x3F)));
        assert!(deliverable(vector, Priority::new(0x2F)));
    }

    #[test]
    fn the_reserved_class_is_illegal() {
        assert!(!legal(Vector::new(15)));
        assert!(legal(Vector::new(16)));
    }

    #[test]
    fn intel_arbitration_ands_the_classes_on_the_fallback_path() {
        // Task class 2 does not outrank the serviced class 4, so the fallback
        // applies: 2 AND 4 is 0, and the requested class 1 wins the maximum.
        let priority = intel_arbitration(
            Priority::new(0x27),
            Some(Vector::new(0x4A)),
            Some(Vector::new(0x1B)),
        );
        assert_eq!(priority, Priority::new(0x10));
    }

    #[test]
    fn amd_arbitration_takes_the_maximum_where_intel_takes_an_and() {
        // The same inputs: AMD answers with the serviced class 4, which is the
        // greatest of the three, where the P6 rule answered with 1.
        let task = Priority::new(0x27);
        let serviced = Some(Vector::new(0x4A));
        let requested = Some(Vector::new(0x1B));
        assert_eq!(
            amd_arbitration(task, serviced, requested),
            Priority::new(0x40)
        );
        assert_eq!(
            intel_arbitration(task, serviced, requested),
            Priority::new(0x10)
        );
    }

    #[test]
    fn amd_arbitration_keeps_the_task_subclass_on_a_tie() {
        // The task class equals the serviced class, so the task priority wins
        // and its subclass survives rather than being floored to the class.
        let priority = amd_arbitration(Priority::new(0x45), Some(Vector::new(0x4A)), None);
        assert_eq!(priority, Priority::new(0x45));
    }

    #[test]
    fn amd_arbitration_counts_what_is_merely_requested() {
        // Nothing in service and a task priority of zero: the request alone
        // decides, which is the contribution the P6 fallback can lose.
        let priority = amd_arbitration(Priority::NONE, None, Some(Vector::new(0x6C)));
        assert_eq!(priority, Priority::new(0x60));
    }
}
