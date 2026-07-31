//! The interrupt descriptor table: a gate for every one of the 256 vectors.
//!
//! # Why every vector, and how each entry point knows which it is
//!
//! A gate that is not present does not make its vector unreachable; it turns an
//! arrival into a `#NP` naming the vector that had no gate, which is a worse
//! version of the same event with the original context thrown away. And for a
//! hypervisor sharing vectors with a machine it does not own, an arrival on a
//! vector nobody expected is information, not an impossibility. So all 256 are
//! present.
//!
//! That needs 256 entry points, because the processor tells a handler nothing
//! about which vector it was entered for — the number exists only in the gate
//! that was used. The usual answer is a block of assembly stubs, generated at
//! build time or assembled into writable memory at run time, each pushing its
//! own number before jumping to shared code. Neither is needed here. An entry
//! point taking its vector as a *const generic* is a distinct function per
//! vector, with the number compiled into it, and the compiler writes the
//! interrupt entry and exit sequence itself — so there is no hand-written
//! assembly to get wrong, no code generated at run time, and no memory that has
//! to be writable and executable in turn.
//!
//! # The shapes an entry point comes in
//!
//! Two facts about a vector change what its entry point must look like, and both
//! are read off the vector rather than written out per gate.
//!
//! For ten of the exceptions the processor pushes an error code, which changes
//! the layout of what the handler is entered with; an entry point compiled for
//! the wrong shape returns to the wrong address. And two vectors must never
//! return at all, so their entry points are compiled as diverging functions —
//! which is what makes it impossible, rather than merely wrong, for `#DF` to
//! reach an `IRET`: there is no `IRET` in them to reach.
//!
//! # What an entry point assumes about who sent it
//!
//! The error-code shape is a fact about the processor *raising an exception*.
//! Nothing else that can be delivered on those numbers pushes an error code: not
//! an external interrupt, not a `INT n` executed in ring 0, and the processor
//! offers no way for the handler to tell. An entry point compiled for the coded
//! shape and entered without one reads its return address off by eight bytes.
//!
//! So the invariant is upheld on the other side, at every place that could send
//! one:
//!
//! - [`crate::claim`], the only way a controller is given a vector, refuses
//!   every vector below [`Vector::FIRST_EXTERNAL`].
//! - Nothing in this image executes a software interrupt. `INT3` and `INTO` are
//!   not emitted by the compiler for the code in it, and no `INT n` is written
//!   anywhere in the workspace.
//! - The legacy controllers deliver onto vectors 8 to 15 at reset, and the
//!   platform is free to hand this hypervisor a machine with them still like
//!   that. Interrupts therefore stay masked from the moment firmware's tables
//!   are replaced until every such source is silenced, which is what
//!   [`crate::Descriptors::unmask`] exists to make a caller state.
//!
//! # Two tables, briefly
//!
//! A gate that names an interrupt stack table slot is only meaningful once the
//! task register names the task state segment holding it. But the table has to
//! be live *before* the global descriptor table is replaced, or there is an
//! interval where the gates that are live name selectors that have changed
//! meaning underneath them. Both cannot be true of one table, so there are two:
//! one whose gates switch no stacks, live across the change, and the real one,
//! loaded the instant the task register is valid. See
//! [`crate::Tables::activate`].

use alloc::boxed::Box;

use x86_64::{
    VirtAddr,
    instructions::tables,
    structures::{DescriptorTablePointer, gdt::SegmentSelector, idt::InterruptStackFrame},
};

use crate::{DescriptorError, Resumption, Vector, dispatch};

/// A gate descriptor, as the processor reads it.
///
/// Written out here rather than taken from a dependency because the two shapes
/// of handler cannot be one Rust function type, and a table typed for one of
/// them, filled with the address of the other, is a contract broken however
/// identical the bytes come out. The bytes are the contract, so the bytes are
/// what this describes.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct Gate {
    /// Bits 0 to 15 of the entry point.
    offset_low: u16,
    /// The code selector the handler is entered with.
    selector: SegmentSelector,
    /// Which interrupt stack table slot to switch to, one-based, or zero to stay
    /// on the interrupted stack. Every other bit of the byte is reserved.
    stack: u8,
    /// Present, descriptor privilege level, and gate type.
    attributes: u8,
    /// Bits 16 to 31 of the entry point.
    offset_middle: u16,
    /// Bits 32 to 63 of the entry point.
    offset_high: u32,
    /// Reserved, and zero.
    reserved: u32,
}

/// Present, ring 0, 64-bit interrupt gate — which is the kind that masks
/// interrupts on entry, as opposed to a trap gate, which does not.
const PRESENT_INTERRUPT_GATE: u8 = 0x8E;

impl Gate {
    /// A gate that is not present, which is what every entry of a table is
    /// before its own gate is written. An arrival on one reports a `#NP` naming
    /// the vector rather than transferring anywhere.
    const MISSING: Self = Self {
        offset_low: 0,
        selector: SegmentSelector::NULL,
        stack: 0,
        attributes: 0,
        offset_middle: 0,
        offset_high: 0,
        reserved: 0,
    };

    /// A gate for `entry_point`, switching to `stack` if it is not zero.
    const fn new(entry_point: VirtAddr, selector: SegmentSelector, stack: u8) -> Self {
        let [b0, b1, b2, b3, b4, b5, b6, b7] = entry_point.as_u64().to_le_bytes();
        Self {
            offset_low: u16::from_le_bytes([b0, b1]),
            selector,
            stack,
            attributes: PRESENT_INTERRUPT_GATE,
            offset_middle: u16::from_le_bytes([b2, b3]),
            offset_high: u32::from_le_bytes([b4, b5, b6, b7]),
            reserved: 0,
        }
    }
}

/// One gate per vector, and nothing else, aligned so that no gate straddles a
/// cache line.
#[derive(Debug)]
#[repr(C, align(16))]
pub(crate) struct Idt([Gate; Vector::COUNT]);

/// Bytes one gate descriptor occupies in long mode.
const GATE_BYTES: usize = 16;

const _: () = assert!(
    size_of::<Gate>() == GATE_BYTES,
    "a gate descriptor is sixteen bytes"
);

const _: () = assert!(
    size_of::<Idt>() == GATE_BYTES * Vector::COUNT,
    "an interrupt descriptor table is one sixteen-byte gate per vector, and nothing more"
);

const _: () = assert!(
    align_of::<Idt>() >= GATE_BYTES,
    "a table of gates is aligned at least as strictly as one gate"
);

/// The limit the processor is given for a table, which it reads as one less than
/// the table's size: sixteen bytes for each of the 256 gates, less one.
const LIMIT: u16 = 16 * 256 - 1;

const _: () = assert!(
    LIMIT as usize == size_of::<Idt>() - 1,
    "the limit is the table's own size, less one"
);

/// Writes a gate for each of the 256 vectors.
///
/// Two levels of expansion, sixteen by sixteen, because a const generic
/// argument has to be a constant at the point of the call: there is no loop
/// that could produce one entry point per vector, and writing out 256 calls to
/// achieve it would be 256 chances to transpose a number.
macro_rules! gates {
    ($table:expr, $selector:expr, $stacks:expr) => {
        gates!(@high $table, $selector, $stacks, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    };
    (@high $table:expr, $selector:expr, $stacks:expr, $($high:literal),+) => {
        $( gates!(@low $table, $selector, $stacks, $high, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15); )+
    };
    (@low $table:expr, $selector:expr, $stacks:expr, $high:literal, $($low:literal),+) => {
        $( gate::<{ $high * 16 + $low }>($table, $selector, $stacks); )+
    };
}

impl Idt {
    /// Builds a table whose gates enter this image on `selector` and switch
    /// stacks as `stacks` says.
    ///
    /// # Errors
    ///
    /// [`DescriptorError::OutOfMemory`] if the table cannot be allocated.
    pub(crate) fn build(
        selector: SegmentSelector,
        stacks: Stacks,
    ) -> Result<Box<Self>, DescriptorError> {
        let mut table = Box::try_new(Self([Gate::MISSING; Vector::COUNT]))
            .map_err(|_| DescriptorError::OutOfMemory)?;
        gates!(&mut table, selector, stacks);
        Ok(table)
    }

    /// The pointer the processor is given for this table.
    fn pointer(&self) -> DescriptorTablePointer {
        DescriptorTablePointer {
            limit: LIMIT,
            base: VirtAddr::from_ptr(self),
        }
    }
}

/// Which stack the gates of a table switch to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stacks {
    /// The one the interrupted code was already on.
    ///
    /// What a table loaded before the task register can say and nothing more:
    /// the stacks it would otherwise name are in a task state segment the
    /// processor is not looking at yet.
    Interrupted,
    /// The one each vector asks for, out of this processor's own task state
    /// segment.
    Own,
}

/// Points this processor at `table`.
///
/// # Safety
///
/// `table` must outlive every moment the processor is running with it loaded,
/// which for the table a processor ends up on means forever. Its gates must name
/// a code selector the live global descriptor table describes, and if any of
/// them switches stacks, the task register must already name a task state
/// segment whose slots are filled.
pub(crate) unsafe fn load(table: &Idt) {
    // SAFETY: the pointer describes `table` itself, with the limit its own size
    // fixes; the caller vouches for its lifetime and for the state its gates
    // depend on.
    unsafe { tables::lidt(&table.pointer()) };
}

/// Writes the gate for one vector.
///
/// Both halves of the gate come from the vector itself: which entry point,
/// through the shape the processor enters it with and whether it may return, and
/// which stack, through the conditions that cannot trust the one they
/// interrupted.
fn gate<const NUMBER: u8>(table: &mut Idt, selector: SegmentSelector, stacks: Stacks) {
    let vector = Vector::new(NUMBER);
    let stack = match stacks {
        Stacks::Own => vector.stack().map_or(0, |stack| hardware_slot(stack.slot())),
        Stacks::Interrupted => 0,
    };
    table.0[usize::from(NUMBER)] = Gate::new(entry_point::<NUMBER>(), selector, stack);
}

/// The value a gate holds for a slot of the interrupt stack table.
///
/// The table's seven slots are numbered from one in a gate, because zero is what
/// a gate says to switch no stack at all. Everything else in this crate numbers
/// them from zero, since that is how they are indexed.
const fn hardware_slot(slot: u16) -> u8 {
    let [low, _] = (slot + 1).to_le_bytes();
    low
}

/// The entry point for `NUMBER`, of whichever of the four shapes its own facts
/// call for.
fn entry_point<const NUMBER: u8>() -> VirtAddr {
    let vector = Vector::new(NUMBER);
    match vector.resumption() {
        Resumption::Resume if vector.pushes_error_code() => {
            VirtAddr::from_ptr(coded::<NUMBER> as *const ())
        }
        Resumption::Resume => VirtAddr::from_ptr(plain::<NUMBER> as *const ()),
        // Not generic, and so not one function per vector: there are exactly two
        // of these and both are named here.
        Resumption::Impossible => VirtAddr::from_ptr(double_fault as *const ()),
        Resumption::FailStop => VirtAddr::from_ptr(machine_check as *const ()),
    }
}

/// The entry point for a vector the processor pushes no error code for.
extern "x86-interrupt" fn plain<const NUMBER: u8>(frame: InterruptStackFrame) {
    dispatch::deliver(Vector::new(NUMBER), &frame, None);
}

/// The entry point for a vector the processor pushes an error code for.
extern "x86-interrupt" fn coded<const NUMBER: u8>(frame: InterruptStackFrame, error_code: u64) {
    dispatch::deliver(Vector::new(NUMBER), &frame, Some(error_code));
}

/// The entry point for `#DF`, which the architecture gives no way back from.
///
/// Diverging, so that the compiler writes no `IRET` at all: a double fault is
/// raised because the processor could not deliver something else, and returning
/// would resume state it has already abandoned.
extern "x86-interrupt" fn double_fault(frame: InterruptStackFrame, error_code: u64) -> ! {
    dispatch::terminal(Vector::DOUBLE_FAULT, &frame, Some(error_code))
}

/// The entry point for `#MC`, which this hypervisor stops on as a policy rather
/// than because the architecture insists.
///
/// Diverging for the same reason as [`double_fault`]: the policy is in the type,
/// where nothing can register a handler that quietly overrides it. What it would
/// take to make a machine check recoverable is set out in
/// [`Resumption::FailStop`].
extern "x86-interrupt" fn machine_check(frame: InterruptStackFrame) -> ! {
    dispatch::terminal(Vector::MACHINE_CHECK, &frame, None)
}

const _: () = assert!(
    matches!(Vector::DOUBLE_FAULT.resumption(), Resumption::Impossible)
        && Vector::DOUBLE_FAULT.pushes_error_code(),
    "the double fault entry point is written for a coded vector nothing returns from"
);

const _: () = assert!(
    matches!(Vector::MACHINE_CHECK.resumption(), Resumption::FailStop)
        && !Vector::MACHINE_CHECK.pushes_error_code(),
    "the machine check entry point is written for a plain vector nothing returns from"
);
