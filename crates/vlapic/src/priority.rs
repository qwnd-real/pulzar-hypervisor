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
//!
//! # Written down exactly once, including outside this crate
//!
//! That last claim is load-bearing and is not confined to this crate. The
//! deliverability rule below is evaluated in halves — the in-service half here,
//! the task-priority half by the processor against the control block's
//! `V_INTR_PRIO` and `V_TPR` — so anything that arms a virtual interrupt or
//! mirrors a task priority into the control block is computing a priority class
//! and must compute it with the same arithmetic. [`Priority`] is therefore
//! public: it is the workspace's one statement of what a priority class is, and
//! the crates that put those fields in a control block reach it rather than
//! shifting a byte of their own.

use core::fmt::{self, Display, Formatter};

use descriptors::Vector;

/// One priority byte: an interrupt-priority class in bits 7:4 and a rank within
/// that class in bits 3:0.
///
/// Ordering is the ordering of the whole byte, which is the architecture's
/// ordering of priorities. It is *not* the ordering that decides delivery: that
/// compares interrupt-priority classes alone, and compares them strictly, so a
/// priority that is merely greater by its subclass blocks nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Priority(u8);

impl Priority {
    /// The lowest priority there is, which is what reset leaves the task
    /// priority at and what an empty bitmap stands in for.
    pub const NONE: Self = Self(0);

    /// The priority a byte denotes.
    #[must_use]
    pub const fn new(value: u8) -> Self {
        Self(value)
    }

    /// A vector's own priority, which is its number read as two nibbles.
    #[must_use]
    pub const fn of(vector: Vector) -> Self {
        Self(vector.number())
    }

    /// The lowest priority in an interrupt-priority class: that class with a
    /// zero subclass.
    ///
    /// What a class alone denotes, for the two places a class is all that is
    /// carried — the control block's virtual task priority, which is four bits,
    /// and the architecture's own definition of a processor priority whose
    /// class comes from a vector in service. Anything above the four bits a
    /// class occupies is not a class and is dropped, which is what makes
    /// this total.
    #[must_use]
    pub const fn of_class(class: u8) -> Self {
        Self((class & SUBCLASS_MASK) << CLASS_SHIFT)
    }

    /// The interrupt-priority class, 0 through 15.
    #[must_use]
    pub const fn class(self) -> u8 {
        self.0 >> CLASS_SHIFT
    }

    /// The rank within the class, 0 through 15.
    #[must_use]
    pub const fn subclass(self) -> u8 {
        self.0 & SUBCLASS_MASK
    }

    /// The whole byte, as the guest reads it out of a register.
    #[must_use]
    pub const fn get(self) -> u8 {
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

/// Whether the *architecture* permits a vector to be delivered at all.
///
/// Vectors 0 through 15 are class 0, which the architecture reserves; a guest
/// programming one of them into a local vector table entry or an interrupt
/// command has asked for something the controller must refuse rather than
/// deliver. Everything from 16 up is a vector a real controller accepts,
/// including 16 through 31 — those are reserved for the architecture's own use
/// *by software*, not refused *by the controller*, and the illegal-vector
/// errors cover class 0 alone.
///
/// This is the whole of the architectural rule and none of any other. A vector
/// that will be programmed onto the *host's* controller for a guest source has
/// a second, narrower rule to satisfy — the host's own vectors are already
/// claimed, and exception numbers are not deliverable from a source at all —
/// which is [`crate::hardware::sources`]'s, is not architectural, and is not
/// this. Passing here is necessary for such a source and not sufficient.
pub(crate) const fn legal(vector: Vector) -> bool {
    Priority::of(vector).class() != 0
}

/// The processor priority: the priority this processor is currently servicing
/// at, and the threshold everything is delivered against.
///
/// `in_service` is the highest vector whose in-service bit is set, or `None`
/// when none is. The class is the greater of the two classes, and what the
/// subclass becomes is the one place the two vendors' manuals do not say the
/// same thing:
///
/// - task class greater: the whole task-priority byte, subclass included,
/// - task class lower: the serviced class with a zero subclass,
/// - the classes equal: AMD fixes the subclass as the task priority's, which is
///   again the whole task-priority byte. Intel leaves it model-specific —
///   either the task subclass or zero — so following AMD is a permitted answer
///   there too, and no vendor dispatch is needed for it.
///
/// The first case and the tie therefore produce the same byte and are one
/// branch here. Nothing delivered depends on the choice in any case:
/// [`deliverable`] compares classes only, and the subclass exists so that a
/// guest's read of the register returns a defined value.
pub(crate) fn processor_priority(task: Priority, in_service: Option<Vector>) -> Priority {
    let serviced = in_service.map_or(Priority::NONE, Priority::of);
    if task.class() >= serviced.class() {
        task
    } else {
        Priority::of_class(serviced.class())
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
        Priority::of_class((task.class() & serviced.class()).max(request.class()))
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
        Priority::of_class(highest)
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
///
/// # One rule, evaluated in two places
///
/// The processor priority this is asked about is the greater of two classes,
/// and on the delivery path the two halves are not evaluated in the same place.
/// The in-service half is applied here, by [`Vlapic::nominate`]. The
/// task-priority half is left to the processor, which compares the control
/// block's `V_INTR_PRIO` against its `V_TPR` and raises an interrupt-window
/// exit exactly when the first is the greater — because a guest changes its
/// task priority through its control register without exiting, so a vector this
/// crate ruled out on a priority read at the last exit would stay ruled out
/// however far the guest lowered that priority afterwards.
///
/// `class(v) > isr_class && class(v) > tpr_class` is `class(v) >
/// max(isr_class, tpr_class)`, so the composition is this rule exactly —
/// *while* both halves compare classes and both compare strictly. If the
/// hardware half were ever armed one class high the failure would not be a
/// wrong value: the window exit would fire, this rule would refuse the vector,
/// nothing would be injected, and the guest would exit again immediately,
/// forever. So the class arithmetic has exactly one definition —
/// [`Priority::of`] and [`Priority::of_class`], which the crates that write
/// those two fields use —
/// and `deliverable_is_the_composition_hardware_completes` below is what pins
/// the two halves together.
///
/// [`Vlapic::nominate`]: crate::registers::Vlapic::nominate
pub(crate) fn deliverable(vector: Vector, processor: Priority) -> bool {
    legal(vector) && Priority::of(vector).class() > processor.class()
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

    /// Every vector there is, which is what the exhaustive checks below run
    /// over.
    fn all() -> impl Iterator<Item = Vector> {
        (0..=u8::MAX).map(Vector::new)
    }

    /// Every interrupt-priority class there is.
    fn classes() -> impl Iterator<Item = u8> {
        0..=0x0F
    }

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
        // A task priority carrying a subclass pins the arithmetic and not a
        // state a guest can observe: the exit path stores the control block's
        // four-bit class over this register on every exit, so a subclass written
        // through the page or a model-specific register survives only until the
        // next exit — and a guest's read of the register is itself one.
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
        // and its subclass survives rather than being floored to the class. As
        // above, the subclass is the arithmetic's and not a state a guest
        // reaches.
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

    #[test]
    fn a_class_is_the_whole_of_what_a_class_denotes() {
        for class in classes() {
            let priority = Priority::of_class(class);
            assert_eq!(priority.class(), class);
            assert_eq!(priority.subclass(), 0);
        }
        // Total for anything wider than a class, because the control block's
        // field is four bits and a fifth would shift a class out of the byte.
        assert_eq!(Priority::of_class(0xFF), Priority::of_class(0x0F));
    }

    #[test]
    fn deliverable_is_the_composition_hardware_completes() {
        // The invariant the split delivery path rests on: this crate applies the
        // in-service half and the processor applies the task-priority half
        // against `V_INTR_PRIO`/`V_TPR`, and the two together have to be the
        // whole rule. Checked over every vector against every class of both
        // halves, because the failure if they ever disagree is a vCPU that exits
        // forever rather than a wrong answer anybody would see.
        let serviced = classes().map(|class| Some(Vector::new(Priority::of_class(class).get())));
        for in_service in serviced.chain(core::iter::once(None)) {
            for class in classes() {
                let task = Priority::of_class(class);
                let whole = processor_priority(task, in_service);
                let ours = processor_priority(Priority::NONE, in_service);
                for vector in all() {
                    let hardware = Priority::of(vector).class() > task.class();
                    assert_eq!(
                        deliverable(vector, whole),
                        deliverable(vector, ours) && hardware,
                        "{vector} against task {task} and in service {in_service:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn deliverable_compares_the_class_and_only_the_class() {
        for vector in all() {
            for class in classes() {
                assert_eq!(
                    deliverable(vector, Priority::of_class(class)),
                    legal(vector) && Priority::of(vector).class() > class,
                    "{vector} against class {class}"
                );
                // The subclass takes no part in either direction: a full
                // subclass does not block and a zero one does not admit.
                let full = Priority::new(Priority::of_class(class).get() | 0x0F);
                assert_eq!(
                    deliverable(vector, full),
                    deliverable(vector, Priority::of_class(class)),
                    "{vector} against class {class} with a full subclass"
                );
            }
        }
    }

    #[test]
    fn the_reserved_class_is_never_deliverable_at_any_priority() {
        // The guard on `deliverable` is redundant in the arithmetic — a class is
        // never negative — so no input distinguishes its removal. What can be
        // asserted is the property it states, over every processor priority
        // there is.
        for number in 0..16 {
            let vector = Vector::new(number);
            assert!(!legal(vector), "{vector}");
            for byte in 0..=u8::MAX {
                assert!(!deliverable(vector, Priority::new(byte)), "{vector}");
            }
        }
    }

    #[test]
    fn a_masking_task_priority_admits_nothing() {
        // What an operating system writes to shut interrupts out through the
        // controller rather than through its flag: the top class blocks every
        // vector there is, including the top one, because the comparison is
        // strict.
        for task in [Priority::new(0xF0), Priority::new(0xFF)] {
            let processor = processor_priority(task, None);
            for vector in all() {
                assert!(!deliverable(vector, processor), "{vector} against {task}");
            }
        }
    }

    #[test]
    fn the_top_class_is_deliverable_below_itself_and_nowhere_else() {
        let top = Vector::new(u8::MAX);
        assert_eq!(Priority::of(top).class(), 0x0F);
        assert!(deliverable(top, Priority::of_class(0x0E)));
        assert!(!deliverable(top, Priority::of_class(0x0F)));
    }
}
