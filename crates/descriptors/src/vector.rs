//! Interrupt vectors, and everything the architecture fixes about each of
//! them.
//!
//! A vector number decides three things no software choice can change: whether
//! an architectural exception on it pushes an error code, whether a handler may
//! return at all, and — for the exceptions — what the condition is called.
//! Getting any of them wrong corrupts the handler's stack or silently resumes a
//! machine that cannot be resumed, so all three are stated once here and every
//! other part of the crate derives its behaviour from them.
//!
//! One distinction runs through the whole module and is easy to lose. A vector
//! number fixes what happens when *the processor raises that exception*. It
//! does not fix what happens when something else is delivered on the same
//! number: an external interrupt and a `INT n` executed in ring 0 push no error
//! code whatever number they carry. Nothing in the architecture lets an entry
//! point tell those apart, so the crate keeps them apart instead — see
//! [`crate::idt`] for the invariant that makes the distinction unnecessary, and
//! [`crate::claim`] for where it is enforced.
//!
//! The names are AMD's, because the host this runs on is an AMD processor. A
//! guest may believe it is on something else, and a vector number a guest is
//! shown means whatever that guest's vendor says it means; nothing here is a
//! statement about that.

use core::fmt::{self, Display, Formatter};

/// One of the interrupt vectors the processor can deliver.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Vector(u8);

/// Vectors below this are exceptions the architecture defines; from here up
/// they are the platform's to assign.
const EXCEPTIONS: u8 = 32;

/// The exceptions for which the processor pushes an error code, one bit per
/// vector. A handler for one of these is entered on a stack with an extra
/// eight bytes on it, and a handler compiled for the wrong shape returns to the
/// wrong address.
const PUSHES_ERROR_CODE: u32 = (1 << 8)
    | (1 << 10)
    | (1 << 11)
    | (1 << 12)
    | (1 << 13)
    | (1 << 14)
    | (1 << 17)
    | (1 << 21)
    | (1 << 29)
    | (1 << 30);

impl Vector {
    /// How many vectors an interrupt descriptor table has.
    pub const COUNT: usize = 256;

    /// The lowest vector the architecture leaves to the platform, and so the
    /// first one an interrupt controller may be told to deliver.
    pub const FIRST_EXTERNAL: Self = Self(EXCEPTIONS);

    /// `#DF`, the one vector the architecture gives no way back from.
    pub const DOUBLE_FAULT: Self = Self(8);

    /// `#MC`, the one vector this hypervisor stops on by policy.
    pub const MACHINE_CHECK: Self = Self(18);

    /// `#GP`, which is what the processor raises for an instruction that is
    /// legal but was asked to do something the machine does not allow — and so
    /// the exception a hypervisor both recovers from on its own behalf and
    /// hands to a guest on the guest's.
    pub const GENERAL_PROTECTION: Self = Self(13);

    /// The non-maskable interrupt, which no masking holds off and which
    /// therefore arrives in the middle of whatever this processor was doing —
    /// including inside a lock it will now never release, and including inside
    /// another handler.
    pub const NON_MASKABLE: Self = Self(2);

    /// The vector numbered `number`.
    #[must_use]
    pub const fn new(number: u8) -> Self {
        Self(number)
    }

    /// The number itself.
    #[must_use]
    pub const fn number(self) -> u8 {
        self.0
    }

    /// Whether this vector is one of the exceptions the architecture defines,
    /// rather than one the platform assigns.
    #[must_use]
    pub const fn is_exception(self) -> bool {
        self.0 < EXCEPTIONS
    }

    /// Whether the processor pushes an error code when it raises this vector's
    /// exception.
    ///
    /// This is a statement about the processor raising an exception and nothing
    /// else. An external interrupt or a software interrupt on one of these
    /// numbers arrives without an error code, and no handler can tell which
    /// happened; keeping those off these vectors is what makes the answer here
    /// usable.
    #[must_use]
    pub const fn pushes_error_code(self) -> bool {
        self.is_exception() && PUSHES_ERROR_CODE & (1 << self.0) != 0
    }

    /// What may follow this vector's handler.
    #[must_use]
    pub const fn resumption(self) -> Resumption {
        match self.0 {
            8 => Resumption::Impossible,
            18 => Resumption::FailStop,
            _ => Resumption::Resume,
        }
    }

    /// The stack the processor switches to before entering this vector's
    /// handler, or `None` to stay on the one it interrupted.
    ///
    /// A stack is switched exactly where the interrupted one cannot be trusted:
    /// because the fault may have been caused by that stack, because the
    /// condition can arrive in the middle of another handler, or because the
    /// interrupted stack may not exist at all.
    #[must_use]
    pub const fn stack(self) -> Option<InterruptStack> {
        match self.0 {
            1 => Some(InterruptStack::Debug),
            2 | 30 => Some(InterruptStack::NonMaskable),
            8 => Some(InterruptStack::DoubleFault),
            10..=12 => Some(InterruptStack::Segment),
            13 => Some(InterruptStack::Protection),
            14 => Some(InterruptStack::PageFault),
            18 => Some(InterruptStack::MachineCheck),
            _ => None,
        }
    }

    /// What the architecture calls this vector, for the exceptions it names.
    #[must_use]
    pub const fn name(self) -> Option<&'static str> {
        Some(match self.0 {
            0 => "#DE divide error",
            1 => "#DB debug",
            2 => "NMI non-maskable interrupt",
            3 => "#BP breakpoint",
            4 => "#OF overflow",
            5 => "#BR bound range exceeded",
            6 => "#UD invalid opcode",
            7 => "#NM device not available",
            8 => "#DF double fault",
            9 => "coprocessor segment overrun",
            10 => "#TS invalid TSS",
            11 => "#NP segment not present",
            12 => "#SS stack-segment fault",
            13 => "#GP general protection fault",
            14 => "#PF page fault",
            16 => "#MF x87 floating-point error",
            17 => "#AC alignment check",
            18 => "#MC machine check",
            19 => "#XF SIMD floating-point exception",
            21 => "#CP control protection exception",
            28 => "#HV hypervisor injection exception",
            29 => "#VC VMM communication exception",
            30 => "#SX security exception",
            _ => return None,
        })
    }
}

impl Display for Vector {
    /// The architecture's name where there is one, and the number otherwise.
    /// The reserved exception vectors fall into the second case, which is
    /// right: one of those arriving means something is wrong in a way no name
    /// would explain.
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self.name() {
            Some(name) => formatter.write_str(name),
            None => write!(formatter, "vector {}", self.0),
        }
    }
}

/// What may follow a vector's handler, and why.
///
/// Two different facts wear the same shape and are kept apart here, because
/// conflating them is how a hypervisor ends up claiming the architecture
/// forbids something the architecture merely makes conditional.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resumption {
    /// The handler may return, and the processor carries on with what it
    /// interrupted.
    Resume,
    /// The architecture defines no way back. `#DF` is raised because the
    /// processor could not deliver something else, and what a return would go
    /// back to is state the processor has already abandoned; there is no status
    /// to consult and no condition under which it becomes restartable.
    Impossible,
    /// The architecture would allow a return under conditions this hypervisor
    /// does not establish, so it stops instead.
    ///
    /// This is `#MC`. Whether a machine check is recoverable is a question
    /// about `MCG_STATUS` and the error banks: the saved instruction pointer
    /// may be reliable, the context may be uncorrupted, the error may be
    /// contained. Answering it needs a subsystem that reads and clears that
    /// state, and pulzar has none — so every machine check is fatal here as a
    /// stated policy, not because the processor said so.
    FailStop,
}

/// One of the stacks named by the task state segment's interrupt stack table.
///
/// The processor switches to one of these before entering a handler whose gate
/// names it, whatever the interrupted stack was. Seven slots exist, all seven
/// are used, and which conditions share one is a decision made per slot below
/// rather than a consequence of running out.
///
/// Sharing is not the only hazard. The processor reloads the *same* stack
/// pointer on every entry that selects a slot — it does not continue below a
/// frame that is already there — so a condition reaching its own slot twice
/// would overwrite the first frame with the second. That is what
/// [`crate::nesting`] exists for, and why each of these stacks is allocated
/// with room for more than one frame at a time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterruptStack {
    /// For `#DF`, raised because the processor could not deliver something
    /// else, which is very often because the stack it would have used is gone.
    DoubleFault,
    /// For the non-maskable interrupt and for `#SX`.
    ///
    /// Both arrive from outside and on no instruction of ours in particular:
    /// the first whatever masking says, the second because firmware can leave
    /// `VM_CR.R_INIT` set, which turns an external `INIT` into an exception.
    /// Neither can be held off until the interrupted stack is trustworthy, so
    /// neither may depend on it.
    ///
    /// They share a slot because the alternative is worse. `#SX` on its own
    /// stack would need an eighth slot, which does not exist; `#SX` with no
    /// stack of its own would be delivered on whatever `RSP` happened to be,
    /// which is the hazard being avoided. Nesting between the two is what the
    /// levels in [`crate::nesting`] cover: an `NMI` cannot interrupt itself, an
    /// `#SX` arriving inside an `NMI` handler lands one level down, and a depth
    /// no level is left for is a stated terminal case rather than a silent
    /// overwrite.
    NonMaskable,
    /// For `#MC`, which the hardware raises asynchronously and for the same
    /// reason must not depend on the interrupted stack.
    MachineCheck,
    /// For `#DB`, which single-stepping and breakpoints can raise anywhere at
    /// all, the middle of a stack switch included.
    Debug,
    /// For `#PF`, so that a stack running into its own guard page is still
    /// something that can be reported.
    PageFault,
    /// For `#GP`, the fault a non-canonical stack pointer produces.
    Protection,
    /// For `#TS`, `#NP` and `#SS`: the three faults the processor raises while
    /// loading a segment or using the stack segment.
    ///
    /// They share a slot because of the exception-combination rules rather than
    /// because they are unrecoverable — all three are faults, and all three
    /// report an instruction that could be restarted. What makes the sharing
    /// safe is that a second contributory exception raised while the processor
    /// is delivering one of these becomes `#DF`, which has a slot of its own.
    /// That is a statement about *contributory* pairs and nothing wider: an
    /// unrelated exception may perfectly well be raised while one of these
    /// handlers is running, which is why this slot has levels like every other.
    Segment,
}

impl InterruptStack {
    /// How many stacks the interrupt stack table holds.
    pub const COUNT: usize = 7;

    /// Every stack, so a caller can fill the table without repeating the list.
    pub const ALL: [Self; Self::COUNT] = [
        Self::DoubleFault,
        Self::NonMaskable,
        Self::MachineCheck,
        Self::Debug,
        Self::PageFault,
        Self::Protection,
        Self::Segment,
    ];

    /// Which slot of the interrupt stack table this stack occupies.
    ///
    /// Written out rather than derived from the declaration order, because a
    /// gate descriptor and the task state segment have to agree on it and a
    /// reordering that silently changed the mapping would put a handler on
    /// another handler's stack.
    #[must_use]
    pub const fn slot(self) -> u16 {
        match self {
            Self::DoubleFault => 0,
            Self::NonMaskable => 1,
            Self::MachineCheck => 2,
            Self::Debug => 3,
            Self::PageFault => 4,
            Self::Protection => 5,
            Self::Segment => 6,
        }
    }
}

const _: () = {
    let mut index = 0;
    let mut slot = 0;
    while index < InterruptStack::COUNT {
        assert!(
            InterruptStack::ALL[index].slot() == slot,
            "every stack occupies the slot its place in the list says, so that one \
             array can be indexed by either"
        );
        index += 1;
        slot += 1;
    }
};

impl Display for InterruptStack {
    /// What the slot is for, so an allocation failure names the stack that
    /// could not be backed rather than a number.
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::DoubleFault => "the double fault stack",
            Self::NonMaskable => "the non-maskable interrupt stack",
            Self::MachineCheck => "the machine check stack",
            Self::Debug => "the debug stack",
            Self::PageFault => "the page fault stack",
            Self::Protection => "the general protection stack",
            Self::Segment => "the segment fault stack",
        })
    }
}
