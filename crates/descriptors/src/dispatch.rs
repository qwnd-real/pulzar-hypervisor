//! Where every interrupt arrives, and who gets to claim it.
//!
//! A pass-through hypervisor shares one set of interrupt vectors with a machine
//! it does not own. Some of what arrives is the hypervisor's own doing — its
//! APIC timer, an interprocessor interrupt it sent — and must be consumed
//! silently. The rest belongs to whoever was running and has to be given back.
//! Nothing about the vector number alone settles which is which: the same
//! vector can be either, depending on state only the subsystem that armed it
//! can see.
//!
//! So the decision is delegated. A subsystem [`register`]s a [`Handler`] on the
//! vector it uses; when that vector arrives the handler looks at its own state
//! and answers with a [`Disposition`]. [`Disposition::Consumed`] ends the
//! interrupt there and the entry path resumes what it interrupted.
//! [`Disposition::Passed`] means the interrupt was not the hypervisor's, and
//! neither is the question of what to do with it — it goes to the [`Unclaimed`]
//! the hypervisor supplied when it installed its tables. That callback is the
//! single seam this whole arrangement exists to provide: it is what will hand
//! an interrupt to a guest, and it is the only thing that has to change to make
//! that happen.
//!
//! Every vector has an entry point, including the ones the architecture
//! reserves and the ones no handler has claimed. An interrupt arriving on one
//! of those is not a case to be silently absorbed — it is the hypervisor
//! finding out something about the machine it did not know — so it reaches the
//! same callback and is reported with everything the processor said about it.

use core::{
    fmt::{self, Display, Formatter},
    mem,
    ptr::{NonNull, null_mut},
    sync::atomic::{AtomicPtr, Ordering},
};

use x86_64::{registers::control::Cr2, structures::idt::InterruptStackFrameValue};

use crate::{DescriptorError, Vector};

/// What a handler decided about the interrupt it was shown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Disposition {
    /// The hypervisor caused this interrupt and has dealt with it. Nothing
    /// else happens and the interrupted code resumes.
    Consumed,
    /// The interrupt was not the hypervisor's. What becomes of it is the
    /// hypervisor's decision, not the handler's.
    Passed,
}

/// A function pinned to one vector, which decides whether that interrupt was
/// the hypervisor's own.
///
/// It runs with interrupts masked, on whichever stack the vector's gate names,
/// and it must not assume anything about what it interrupted: the answer to
/// "was this mine?" has to come from the handler's own state.
pub type Handler = fn(&Interrupt) -> Disposition;

/// What the hypervisor does with an interrupt no handler claimed.
///
/// Returning from it resumes the interrupted code, which is what reinjecting
/// into a guest will do once there is a guest to reinject into. Until then
/// there is nothing an unclaimed interrupt can belong to, so the honest
/// implementation reports it and stops.
pub type Unclaimed = fn(&Interrupt);

/// One interrupt, as its handler sees it.
///
/// A copy of what the processor pushed rather than a borrow of it, so that
/// nothing a handler keeps can outlive the stack the interrupt was taken on.
#[derive(Clone, Copy, Debug)]
pub struct Interrupt {
    vector: Vector,
    frame: InterruptStackFrameValue,
    error_code: Option<u64>,
    fault_address: Option<u64>,
}

impl Interrupt {
    /// Which vector arrived.
    #[must_use]
    pub const fn vector(&self) -> Vector {
        self.vector
    }

    /// What the processor pushed: where it was, and what it was doing.
    #[must_use]
    pub const fn frame(&self) -> &InterruptStackFrameValue {
        &self.frame
    }

    /// The error code, for the exceptions that push one.
    #[must_use]
    pub const fn error_code(&self) -> Option<u64> {
        self.error_code
    }

    /// The address that could not be translated, for a page fault only.
    ///
    /// Read raw rather than as a virtual address, because the register holds
    /// whatever the faulting access named and that is free to be a value no
    /// address can be.
    #[must_use]
    pub const fn fault_address(&self) -> Option<u64> {
        self.fault_address
    }
}

impl Display for Interrupt {
    /// Everything the processor said, on one line, in the order it matters: the
    /// condition, where it happened, and then whatever extra the vector
    /// carries.
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} at rip {:#x}, cs {:#06x}, rflags {:#x}, rsp {:#x}",
            self.vector,
            self.frame.instruction_pointer,
            self.frame.code_segment.0,
            self.frame.cpu_flags.bits(),
            self.frame.stack_pointer,
        )?;
        if let Some(code) = self.error_code {
            write!(formatter, ", error code {code:#x}")?;
        }
        if let Some(address) = self.fault_address {
            write!(formatter, ", faulting address {address:#x}")?;
        }
        Ok(())
    }
}

/// One slot per vector, holding the handler that claimed it or nothing.
///
/// An array of pointers rather than anything behind a lock: this is read on
/// every interrupt, and a delivery path that could block on a lock another
/// processor holds would turn a contended registry into a stalled machine.
static HANDLERS: [AtomicPtr<()>; Vector::COUNT] =
    [const { AtomicPtr::new(null_mut()) }; Vector::COUNT];

/// What becomes of an interrupt no handler claimed. Null until the tables are
/// installed, which happens before the table naming these entry points is
/// loaded, so a delivery can never find it unset.
static UNCLAIMED: AtomicPtr<()> = AtomicPtr::new(null_mut());

/// Pins `handler` to `vector`.
///
/// A vector holds one handler for good. Two subsystems sharing a vector would
/// each have to decide an interrupt the other might own, and the second
/// registration silently replacing the first would be worse still, so the
/// second is refused.
///
/// # Errors
///
/// [`DescriptorError::VectorTaken`] if something already claimed the vector, or
/// [`DescriptorError::NotReturnable`] for a vector the architecture gives no
/// way back from — consuming one of those would mean resuming a machine whose
/// state the processor has already declared lost.
pub fn register(vector: Vector, handler: Handler) -> Result<(), DescriptorError> {
    if !vector.returns() {
        return Err(DescriptorError::NotReturnable { vector });
    }
    HANDLERS[usize::from(vector.number())]
        .compare_exchange(
            null_mut(),
            handler as *mut (),
            Ordering::Release,
            Ordering::Relaxed,
        )
        .map(drop)
        .map_err(|_| DescriptorError::VectorTaken { vector })
}

/// Records what becomes of an unclaimed interrupt.
///
/// Called before the interrupt descriptor table is loaded, so that no delivery
/// can happen while there is no answer to give.
pub(crate) fn adopt(unclaimed: Unclaimed) {
    UNCLAIMED.store(unclaimed as *mut (), Ordering::Release);
}

/// Where every entry point ends up.
///
/// `vector` and the presence of `error_code` both come from the entry point the
/// vector's own gate names, so neither can disagree with what the processor
/// actually delivered.
pub(crate) fn deliver(vector: Vector, frame: &InterruptStackFrameValue, error_code: Option<u64>) {
    let interrupt = Interrupt {
        vector,
        frame: *frame,
        error_code,
        // Read first and only here: the register keeps the address of the last
        // fault, so anything this path did that faulted would replace it.
        fault_address: (vector == PAGE_FAULT).then(Cr2::read_raw),
    };
    if claimed(vector).is_some_and(|handler| handler(&interrupt) == Disposition::Consumed) {
        return;
    }
    match unclaimed() {
        Some(unclaimed) => unclaimed(&interrupt),
        // Unreachable: the table naming this entry point is loaded only after
        // the callback is recorded. Stopping rather than returning is what keeps
        // a machine that proves otherwise from taking the same interrupt forever.
        None => crate::halt(),
    }
}

/// The vector whose faulting address the processor reports in `CR2`.
const PAGE_FAULT: Vector = Vector::new(14);

/// The handler that claimed `vector`, if any.
fn claimed(vector: Vector) -> Option<Handler> {
    // SAFETY: the only value `register` ever stores in a slot is a `Handler`
    // cast to a raw pointer, and null — which is filtered out first — is the
    // only other value a slot can hold. So the pointer here always came from a
    // real function, and `transmute` checks that the two types are the same
    // size, which makes this exactly the inverse of the cast that stored it.
    load(&HANDLERS[usize::from(vector.number())])
        .map(|slot| unsafe { mem::transmute::<NonNull<()>, Handler>(slot) })
}

/// The callback [`adopt`] recorded, if it has run.
fn unclaimed() -> Option<Unclaimed> {
    // SAFETY: as in `claimed`, for the one value `adopt` stores.
    load(&UNCLAIMED).map(|slot| unsafe { mem::transmute::<NonNull<()>, Unclaimed>(slot) })
}

/// The non-null contents of a slot.
fn load(slot: &AtomicPtr<()>) -> Option<NonNull<()>> {
    NonNull::new(slot.load(Ordering::Acquire))
}
