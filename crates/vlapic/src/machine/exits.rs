//! What each processor's own exit loop asks of its controller.
//!
//! Every one of these is about the processor calling it, and every one of them
//! is a question the exit path has to ask afresh each time: what the guest is
//! owed, what it was actually given, what priority it is running at, and
//! whether it is watching its controller at all. Nothing here decides anything
//! — the decisions are the controller's, in [`crate::registers`] — so what is
//! left is finding this processor's controller and reporting what it says.

use descriptors::Vector;
use log::trace;

use crate::{
    VlapicError,
    machine::current,
    priority::Priority,
    registers::{Nomination, Vlapic},
};

/// What this processor's controller has for its guest, left where it is.
///
/// Both answers at once, out of one look at the register file, because the
/// caller needs both and they are two comparisons against the same state: the
/// highest interrupt the guest should take now, and the highest one only the
/// guest's own task priority may be holding back. Asking twice would let a
/// vector be admitted by one question and refused by the other on state that
/// moved in between, and would scan both bitmaps twice on a path every exit
/// takes.
///
/// Nothing is consumed. Whether the guest can actually be given the first of
/// them is not the controller's to know — the processor may already have an
/// event part-way through delivery, or a non-maskable interrupt that goes
/// first, or an interrupt window that is shut — so the caller decides, and
/// reports back through [`committed`]. A controller that moved a vector out of
/// the request register for an injection that then did not happen would have
/// thrown the interrupt away, and for a level-triggered one would have stranded
/// the real acknowledgement owed for it as well.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub fn nominate() -> Result<Nomination, VlapicError> {
    current().map(Vlapic::nominate)
}

/// Records that the guest really has been given `vector`, moving it from
/// requested to in service.
///
/// The other half of [`nominate`], and the only thing that consumes a request.
/// Called once an injection is known to have happened.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub fn committed(vector: Vector) -> Result<(), VlapicError> {
    current().map(|vlapic| {
        if !vlapic.committed(vector) {
            // The request was withdrawn between the two halves, which a reset
            // arriving in that window does. Nothing is put in service: the
            // interrupt belonged to a guest that no longer exists.
            trace!(
                "vlapic: {} was given {vector}, which its controller no longer had requested",
                vlapic.index()
            );
        }
    })
}

/// Whether this processor's guest is owed a non-maskable interrupt, taking it
/// if so.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub fn take_nmi() -> Result<bool, VlapicError> {
    current().map(Vlapic::take_nmi)
}

/// Records that this processor's guest is owed a non-maskable interrupt.
///
/// Called from the host's own handler for one that arrived while the host was
/// running — in the window between a world switch restoring host state and the
/// next entry — which the host takes and the guest is still owed.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub fn raise_nmi() -> Result<(), VlapicError> {
    current().map(Vlapic::raise_nmi)
}

/// Records the priority class the guest set through `CR8`.
///
/// With interrupt masking virtualized, the processor keeps `CR8` in the control
/// block and writes it back on every exit. Observing it on every exit also
/// covers a write of the same class already present, which matters because a
/// `CR8` write clears the APIC task-priority subclass.
///
/// `class` is the four bits the control block carries, which is the whole of
/// what that control register holds.
///
/// A failure is deliberately not reported: this is called on every exit before
/// anything has been decided, and a processor with no controller has nothing
/// that could want the value.
pub fn observe_task_priority(class: u8) {
    if let Ok(vlapic) = current() {
        vlapic.observe_task_priority(class);
    }
}

/// The task priority this processor's guest has set.
///
/// The value the exit loop pushes into the control block's virtual task
/// priority, so that a guest which wrote the emulated register through the
/// page or a model-specific register finds its `CR8` answering the same
/// number — the two are one register on real hardware and must stay one. Only
/// the class of it goes in the control block, and [`Priority::class`] is where
/// that narrowing is written.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub fn task_priority() -> Result<Priority, VlapicError> {
    current().map(Vlapic::task_priority)
}

/// Records whether this processor has stopped looking at its controller.
///
/// The second half of the protocol that stops an interrupt being lost to a
/// processor that was entering the guest as it arrived. A caller must store
/// `true` and then consult [`nominate`] once more before it actually enters,
/// abandoning the entry if something appeared in between — and store `false` on
/// the way out, because a processor answering an exit will consult its
/// controller again on its own and needs nothing to remind it.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub fn set_away(away: bool) -> Result<(), VlapicError> {
    current().map(|vlapic| vlapic.set_away(away))
}
