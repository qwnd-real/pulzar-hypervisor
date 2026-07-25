//! Interrupt vectors, and everything the architecture fixes about each of
//! them.
//!
//! A vector number decides three things no software choice can change: whether
//! the processor pushes an error code before entering the handler, whether the
//! handler may return at all, and — for the exceptions — what the condition is
//! called. Getting any of them wrong corrupts the handler's stack or silently
//! resumes a machine that cannot be resumed, so all three are stated once here
//! and every other part of the crate derives its behaviour from them.

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

/// The exceptions the architecture gives no defined way back from, one bit per
/// vector. A double fault is raised because the processor could not deliver
/// something else, and a machine check because the hardware itself reported a
/// failure; in both cases the state a return would restore is already gone.
const NEVER_RETURNS: u32 = (1 << 8) | (1 << 18);

impl Vector {
    /// How many vectors an interrupt descriptor table has.
    pub const COUNT: usize = 256;

    /// The lowest vector the architecture leaves to the platform, and so the
    /// first one an interrupt controller may be told to deliver.
    pub const FIRST_EXTERNAL: Self = Self(EXCEPTIONS);

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

    /// Whether the processor pushes an error code before entering this vector's
    /// handler.
    #[must_use]
    pub const fn pushes_error_code(self) -> bool {
        self.is_exception() && PUSHES_ERROR_CODE & (1 << self.0) != 0
    }

    /// Whether a handler for this vector may return to what it interrupted.
    #[must_use]
    pub const fn returns(self) -> bool {
        !(self.is_exception() && NEVER_RETURNS & (1 << self.0) != 0)
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
            2 => Some(InterruptStack::NonMaskable),
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
            19 => "#XM SIMD floating-point exception",
            20 => "#VE virtualization exception",
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

/// One of the stacks named by the task state segment's interrupt stack table.
///
/// The processor switches to one of these before entering a handler whose gate
/// names it, whatever the interrupted stack was. Seven slots exist and all
/// seven are used, so no two of these conditions can land on the same stack —
/// which matters because each of them is a condition the previous stack may be
/// the cause of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterruptStack {
    /// For `#DF`, raised because the processor could not deliver something
    /// else, which is very often because the stack it would have used is gone.
    DoubleFault,
    /// For the non-maskable interrupt, which arrives whatever the processor is
    /// in the middle of, including another handler.
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
    /// For `#TS`, `#NP` and `#SS`: the three faults raised while the processor
    /// is loading a segment or using the stack segment, which are one family
    /// and cannot be nested, since none of them is recoverable.
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
