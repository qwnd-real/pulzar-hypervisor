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
//! # The two shapes
//!
//! For ten of the exceptions the processor pushes an error code, which changes
//! the layout of what the handler is entered with. An entry point compiled for
//! the wrong shape returns to the wrong address, so the choice between the two
//! is not made by hand at each vector: it comes from
//! [`Vector::pushes_error_code`], the same predicate that states what the
//! processor does, applied to the very vector whose gate is being written.

use spin::Once;
use x86_64::{
    VirtAddr,
    instructions::tables,
    structures::{
        DescriptorTablePointer,
        idt::{Entry, HandlerFunc, InterruptStackFrame},
    },
};

use crate::{Vector, dispatch};

/// Bytes one gate descriptor occupies in long mode.
const GATE_BYTES: usize = 16;

/// The limit the processor is given for the table, which it reads as one less
/// than the table's size: sixteen bytes for each of the 256 gates, less one.
const LIMIT: u16 = 16 * 256 - 1;

const _: () = assert!(
    size_of::<Idt>() == GATE_BYTES * Vector::COUNT,
    "an interrupt descriptor table is one sixteen-byte gate per vector"
);

/// Builds the table and points the processor at it.
///
/// Must run after the global descriptor table is loaded and `CS` reloaded: a
/// gate records the code selector to enter its handler with, and that is taken
/// from whatever `CS` holds while the gate is written.
pub(crate) fn install() {
    let table = IDT.call_once(build);
    // SAFETY: `table` is a `'static` table of the size `LIMIT` describes, with a
    // gate for every vector, each naming an entry point in this image and a
    // stack slot the task state segment already filled. Nothing ever moves it or
    // gives it back, so the pointer the processor keeps stays good for as long
    // as the processor runs.
    unsafe { tables::lidt(&table.pointer()) };
}

/// The table itself. It outlives every reference the processor holds to it, so
/// it lives in a static rather than anywhere it could be dropped.
static IDT: Once<Idt> = Once::new();

/// One gate per vector.
///
/// Every entry is typed as taking no error code, because the entries are filled
/// in by address rather than by function: a single array cannot hold both
/// shapes, and the shape is guaranteed where the address is chosen instead.
#[derive(Debug)]
#[repr(C, align(16))]
struct Idt([Entry<HandlerFunc>; Vector::COUNT]);

impl Idt {
    /// The pointer the processor is given for this table.
    fn pointer(&'static self) -> DescriptorTablePointer {
        DescriptorTablePointer {
            limit: LIMIT,
            base: VirtAddr::from_ptr(self),
        }
    }
}

/// Writes a gate for each of the 256 vectors.
///
/// Two levels of expansion, sixteen by sixteen, because a const generic
/// argument has to be a constant at the point of the call: there is no loop
/// that could produce one entry point per vector, and writing out 256 calls to
/// achieve it would be 256 chances to transpose a number.
macro_rules! gates {
    ($table:expr) => {
        gates!(@high $table, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    };
    (@high $table:expr, $($high:literal),+) => {
        $( gates!(@low $table, $high, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15); )+
    };
    (@low $table:expr, $high:literal, $($low:literal),+) => {
        $( gate::<{ $high * 16 + $low }>($table); )+
    };
}

/// Fills a table with a gate for every vector.
fn build() -> Idt {
    let mut table = Idt([Entry::missing(); Vector::COUNT]);
    gates!(&mut table);
    table
}

/// Writes the gate for one vector.
///
/// Both halves of the gate come from the vector itself: which entry point,
/// through the shape the processor uses for it, and which stack, through the
/// conditions that cannot trust the one they interrupted.
fn gate<const NUMBER: u8>(table: &mut Idt) {
    let vector = Vector::new(NUMBER);
    let entry_point = if vector.pushes_error_code() {
        VirtAddr::from_ptr(coded::<NUMBER> as *const ())
    } else {
        VirtAddr::from_ptr(plain::<NUMBER> as *const ())
    };

    // SAFETY: `entry_point` is this crate's own entry point for this very
    // vector, generated for whichever of the two frame shapes the processor
    // builds for it — the choice above being made by the same predicate that
    // describes the processor's behaviour, applied to the same vector.
    let options = unsafe { table.0[usize::from(NUMBER)].set_handler_addr(entry_point) };
    if let Some(stack) = vector.stack() {
        // SAFETY: `slot` is one of the seven the interrupt stack table holds,
        // and every one of them was filled with a mapped, guarded stack before
        // this table could be built, let alone loaded. No two vectors that can
        // nest name the same slot.
        unsafe { options.set_stack_index(stack.slot()) };
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
