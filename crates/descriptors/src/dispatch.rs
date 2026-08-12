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
//! [`Disposition::Redirected`] is the third answer and belongs to the faults:
//! the interrupt was the hypervisor's own, and the code it interrupted carries
//! on somewhere other than at the instruction that raised it. It is how a
//! subsystem that deliberately attempts something the machine may refuse gets
//! the refusal back as a value.
//!
//! Every vector has an entry point, including the ones the architecture
//! reserves and the ones no handler has claimed. An interrupt arriving on one
//! of those is not a case to be silently absorbed — it is the hypervisor
//! finding out something about the machine it did not know — so it reaches the
//! same callback and is reported with everything the processor said about it.
//!
//! The two vectors nothing may return from never reach either. Their entry
//! points do not come back, so there is nothing for a callback to answer; they
//! are reported through [`crate::fatal`] and the processor stops.
//!
//! # Publication
//!
//! A slot holds a function pointer and is written once, with one compare and
//! exchange. There is no interval in which a registration is half-done: a
//! delivery either finds nothing claiming the vector or finds the handler that
//! claimed it, and the losing racer is told so rather than silently replacing
//! the winner. Reading one is a load and a comparison, which is what an entry
//! path can afford — a registry behind a lock would let one processor's
//! registration stall another processor's interrupt.

use core::{
    fmt::{self, Display, Formatter},
    ptr,
    sync::atomic::{AtomicPtr, Ordering},
};

use x86_64::{
    VirtAddr,
    registers::control::Cr2,
    structures::idt::{InterruptStackFrame, InterruptStackFrameValue},
};

use crate::{DescriptorError, Resumption, Vector, fatal, nesting::Guard};

/// What a handler decided about the interrupt it was shown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Disposition {
    /// The hypervisor caused this interrupt and has dealt with it, including
    /// acknowledging it to the controller that delivered it. Nothing else
    /// happens and the interrupted code resumes.
    Consumed,
    /// The interrupt was not the hypervisor's, and has not been acknowledged.
    /// What becomes of it is the hypervisor's decision, not the handler's.
    Passed,
    /// The hypervisor caused this interrupt, has dealt with it, and the
    /// interrupted code carries on at this address rather than at what raised
    /// it.
    ///
    /// For a fault, which is what this exists for. The processor reports the
    /// instruction that faulted rather than the one after it, so a handler
    /// answering [`Disposition::Consumed`] to a fault has arranged for the same
    /// fault to be raised forever. A handler that knows what the faulting
    /// instruction was doing — because it is the one that put it there — can
    /// say where that code carries on instead, which is how an operation
    /// the hypervisor is willing to have fail is allowed to fail.
    ///
    /// The address is written into the frame the processor returns through, so
    /// a handler answering this must name one the interrupted code can
    /// really carry on at: in this image, and reachable with the stack
    /// exactly as the faulting instruction left it. The stack is what makes
    /// that a real obligation rather than a formality — nothing here moves
    /// `RSP`, so the address has to belong to the routine the fault
    /// happened in.
    Redirected(VirtAddr),
}

/// A function pinned to one vector, which decides whether that interrupt was
/// the hypervisor's own.
///
/// It runs with interrupts masked, on whichever stack the vector's gate names,
/// and it must not assume anything about what it interrupted: the answer to
/// "was this mine?" has to come from the handler's own state.
///
/// # What a handler may not do
///
/// The interrupted code is this same processor, stopped mid-instruction. It may
/// hold any lock, be inside the allocator, be halfway through publishing
/// something. So a handler:
///
/// - must not block, and must not wait for anything a processor has to run to
///   provide — a lock this processor may already hold is the common case, and
///   waiting for it is a processor that never comes back;
/// - must not take a lock the interrupted code could be holding, which for a
///   maskable interrupt means any lock only ever taken with interrupts masked,
///   and for `NMI` and the exceptions means any lock at all. [`log`] is such a
///   lock: it is safe from a maskable handler, because the backend masks
///   interrupts for the whole of a line, and it is not safe from anything
///   masking cannot hold off, which must use [`serial::emergency`] instead;
/// - must not allocate, since the allocator is one of those locks;
/// - must not panic or unwind: there is no unwinding through an interrupt
///   entry, and a panic on this path is a fault inside a fault;
/// - must not unmask interrupts, which would let a second arrival nest inside
///   the first on the same stack;
/// - must return promptly, because everything else on this processor is stopped
///   until it does;
/// - must acknowledge the interrupt exactly once before answering
///   [`Disposition::Consumed`], and must not acknowledge it when answering
///   [`Disposition::Passed`]: a local controller holds an interrupt in service
///   until it is told otherwise, and an acknowledgement by the wrong owner ends
///   an interrupt somebody else is still handling.
pub type Handler = fn(&Interrupt) -> Disposition;

/// What the hypervisor does with an interrupt no handler claimed.
///
/// Returning from it resumes the interrupted code, which is what reinjecting
/// into a guest will do once there is a guest to reinject into. Until then
/// there is nothing an unclaimed interrupt can belong to, so the honest
/// implementation reports it and stops.
///
/// Everything [`Handler`] may not do, this may not do either, and one of those
/// restrictions binds harder here: this is the callback an unclaimed `NMI`
/// reaches, so it can be entered while the interrupted code on this processor
/// holds the logger's lock. It is also reached by every exception no handler
/// claimed, none of which masking held off.
///
/// It is never entered for a vector nothing may return from. Those do not come
/// back at all, so there would be nothing to do with an answer.
pub type Unclaimed = fn(&Interrupt);

/// One interrupt, as its handler sees it.
///
/// A copy of what the processor pushed rather than a borrow of it: nothing a
/// handler keeps refers to the frame on the stack it was entered on, so a
/// retained copy stays readable and stays what it was. What it is *not* is
/// current — it is what the processor said at the moment of entry, and the
/// stack it came from is free to be used again.
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
    ///
    /// It is the register's value as of entry, which is the faulting address as
    /// long as nothing on the way here faulted too. A fault inside this crate's
    /// own entry path would replace it, so the read happens before anything
    /// else does.
    #[must_use]
    pub const fn fault_address(&self) -> Option<u64> {
        self.fault_address
    }

    /// Everything the processor said about one arrival, taken as early as
    /// anything on this path can take it.
    fn new(vector: Vector, frame: &InterruptStackFrameValue, error_code: Option<u64>) -> Self {
        Self {
            vector,
            frame: *frame,
            error_code,
            // Read first and only here: the register keeps the address of the
            // last fault, so anything this path did that faulted would replace
            // it.
            fault_address: (vector == PAGE_FAULT).then(Cr2::read_raw),
        }
    }
}

impl Display for Interrupt {
    /// Everything the processor said, on one line, in the order it matters: the
    /// condition, where it happened, and then whatever extra the vector
    /// carries.
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} at rip {:#x}, cs {:#06x}, rflags {:#x}, rsp {:#x}, ss {:#06x}",
            self.vector,
            self.frame.instruction_pointer,
            self.frame.code_segment.0,
            self.frame.cpu_flags.bits(),
            self.frame.stack_pointer,
            self.frame.stack_segment.0,
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
static HANDLERS: [AtomicPtr<()>; Vector::COUNT] =
    [const { AtomicPtr::new(ptr::null_mut()) }; Vector::COUNT];

/// What becomes of an interrupt no handler claimed. Empty until the hypervisor
/// says, which is a precondition of any processor loading a table of gates, so
/// a delivery can never find it unset.
static UNCLAIMED: AtomicPtr<()> = AtomicPtr::new(ptr::null_mut());

/// Pins `handler` to `vector`.
///
/// A vector holds one handler for good. Two subsystems sharing a vector would
/// each have to decide an interrupt the other might own, and the second
/// registration silently replacing the first would be worse still, so the
/// second is refused.
///
/// The vector must be quiesced until this returns: a source that can deliver
/// while the slot is still empty gets the [`Unclaimed`] treatment, which for
/// something the hypervisor caused is the wrong answer. Every caller in this
/// workspace registers before switching its source on, which is the ordering
/// this asks for.
///
/// # Errors
///
/// [`DescriptorError::VectorTaken`] if something already claimed the vector,
/// [`DescriptorError::NotReturnable`] for the vector the architecture gives no
/// way back from, or [`DescriptorError::FailStop`] for the one this hypervisor
/// chooses not to return from. Consuming either would mean resuming a machine
/// whose state nothing here can vouch for.
pub fn register(vector: Vector, handler: Handler) -> Result<(), DescriptorError> {
    match vector.resumption() {
        Resumption::Resume => {}
        Resumption::Impossible => return Err(DescriptorError::NotReturnable { vector }),
        Resumption::FailStop => return Err(DescriptorError::FailStop { vector }),
    }
    HANDLERS[usize::from(vector.number())]
        .compare_exchange(
            ptr::null_mut(),
            erase(handler),
            Ordering::Release,
            Ordering::Relaxed,
        )
        .map(drop)
        .map_err(|_| DescriptorError::VectorTaken { vector })
}

/// Pins `handler` to the highest free vector in `first..=last`.
///
/// For subsystems that need a vector rather than a particular one — an
/// interprocessor interrupt is only ever addressed by the sender, so which
/// number it is matters to nobody but the priority the processor gives it.
/// Highest first, because on this architecture a vector's high nibble *is* its
/// priority.
///
/// The range must be one an interrupt controller may be told to deliver, which
/// means it may not reach below [`Vector::FIRST_EXTERNAL`]. An exception vector
/// handed to a controller would arrive without the error code its gate is
/// written for; see [`crate::idt`].
///
/// # Errors
///
/// [`DescriptorError::RangeReversed`] if `last` is below `first`,
/// [`DescriptorError::NotExternal`] if the range reaches into the exceptions,
/// [`DescriptorError::NoVectorFree`] if every vector in it is taken, or
/// whatever [`register`] reported for a vector that could not be claimed for
/// some other reason.
pub fn claim(first: Vector, last: Vector, handler: Handler) -> Result<Vector, DescriptorError> {
    if last < first {
        return Err(DescriptorError::RangeReversed { first, last });
    }
    if first.is_exception() {
        return Err(DescriptorError::NotExternal { vector: first });
    }
    for number in (first.number()..=last.number()).rev() {
        let vector = Vector::new(number);
        match register(vector, handler) {
            Ok(()) => return Ok(vector),
            Err(DescriptorError::VectorTaken { .. }) => {}
            Err(error) => return Err(error),
        }
    }
    Err(DescriptorError::NoVectorFree { first, last })
}

/// Records what becomes of an unclaimed interrupt.
///
/// One answer for the whole machine, given once and before any processor loads
/// a table of gates, so that no delivery can happen while there is nothing to
/// give it to. That ordering is enforced from the other side:
/// [`Tables::build`](crate::Tables::build) refuses until this has run.
///
/// # Errors
///
/// [`DescriptorError::AlreadyAdopted`] for a second call. Replacing this while
/// interrupts are being delivered through it would change what happens to a
/// guest's interrupts underneath the guest.
pub fn adopt(unclaimed: Unclaimed) -> Result<(), DescriptorError> {
    UNCLAIMED
        .compare_exchange(
            ptr::null_mut(),
            erase(unclaimed),
            Ordering::Release,
            Ordering::Relaxed,
        )
        .map(drop)
        .map_err(|_| DescriptorError::AlreadyAdopted)
}

/// Whether anything has said what becomes of an unclaimed interrupt.
pub(crate) fn adopted() -> bool {
    !UNCLAIMED.load(Ordering::Acquire).is_null()
}

/// Where the entry point of every vector that may return ends up.
///
/// `vector` and the presence of `error_code` both come from the entry point the
/// vector's own gate names, so neither can disagree with what the processor
/// actually delivered. The frame is borrowed mutably for the sake of one
/// handler answer out of three: a fault the hypervisor asked for is recovered
/// from by changing where the interrupted code resumes.
pub(crate) fn deliver(vector: Vector, frame: &mut InterruptStackFrame, error_code: Option<u64>) {
    // First, before anything else on this stack: it is what a second arrival on
    // the same stack lands below instead of on top of.
    let _level = Guard::enter(vector);
    let interrupt = Interrupt::new(vector, frame, error_code);
    match claimed(vector).map(|handler| handler(&interrupt)) {
        Some(Disposition::Consumed) => return,
        Some(Disposition::Redirected(rip)) => {
            // SAFETY: the address comes from the handler that claimed this
            // vector, which `Disposition::Redirected` obliges to name one the
            // interrupted code can carry on at with the stack as it stands.
            unsafe { resume_at(frame, rip) };
            return;
        }
        Some(Disposition::Passed) | None => {}
    }
    match unclaimed() {
        Some(unclaimed) => unclaimed(&interrupt),
        // Unreachable: a table of gates naming this entry point is loaded only
        // after the callback is recorded. Reporting and stopping rather than
        // returning is what keeps a machine that proves otherwise from taking the
        // same interrupt forever.
        None => fatal::unanswered(&interrupt),
    }
}

/// Where the entry point of a vector nothing may return from ends up.
///
/// No handler and no callback: registration refuses these vectors, and the
/// hypervisor's answer for an unclaimed interrupt is a function that returns,
/// which is the one thing that must not happen here. It is reported through the
/// path that depends on no lock and no allocation, since what raised it may be
/// the reason either is unavailable.
pub(crate) fn terminal(
    vector: Vector,
    frame: &InterruptStackFrameValue,
    error_code: Option<u64>,
) -> ! {
    fatal::interrupt(&Interrupt::new(vector, frame, error_code))
}

/// Puts the address the interrupted code is to carry on at into the frame the
/// processor returns through.
///
/// The instruction pointer and nothing else: the rest of the frame is what the
/// processor pushed, and the flags, the stack pointer and both selectors are
/// still the ones the interrupted code was running with.
///
/// # Safety
///
/// `rip` must be an address the interrupted code can carry on at, which is the
/// obligation [`Disposition::Redirected`] states in full. A handler naming one
/// that does not meet it returns the processor into whatever is at that
/// address.
unsafe fn resume_at(frame: &mut InterruptStackFrame, rip: VirtAddr) {
    // SAFETY: the caller vouches for `rip`, and the field written is the one the
    // architecture reads the return address out of. The write goes through the
    // volatile wrapper because the compiler is otherwise entitled to discard a
    // store to a frame nothing in this image reads again.
    unsafe { frame.as_mut() }.update(|frame| frame.instruction_pointer = rip);
}

/// The handler that claimed `vector`, if one did.
fn claimed(vector: Vector) -> Option<Handler> {
    restore(HANDLERS[usize::from(vector.number())].load(Ordering::Acquire))
}

/// The hypervisor's answer for an interrupt nothing claimed, if it has given
/// one.
fn unclaimed() -> Option<Unclaimed> {
    restore(UNCLAIMED.load(Ordering::Acquire))
}

/// A function pointer as the value a slot holds.
fn erase<T>(function: T) -> *mut ()
where
    T: Copy,
{
    // SAFETY: `T` is one of this module's function pointer types — the two
    // callers are the two `register`/`adopt` pairs — so it is one pointer wide
    // and a pointer is what comes out.
    unsafe { core::mem::transmute_copy(&function) }
}

/// What [`erase`] erased, or `None` for a slot nothing has written.
fn restore<T>(slot: *mut ()) -> Option<T>
where
    T: Copy,
{
    if slot.is_null() {
        return None;
    }
    // SAFETY: the slot is not null, so it holds what `erase` put there: a
    // function pointer of this very type, since each slot is written by exactly
    // one `register` or `adopt` and read back as the type that wrote it.
    Some(unsafe { core::mem::transmute_copy(&slot) })
}

/// The vector whose faulting address the processor reports in `CR2`.
const PAGE_FAULT: Vector = Vector::new(14);
