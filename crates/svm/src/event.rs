//! Event injection, and the record of what the guest was delivering when an
//! intercept cut it short.
//!
//! Two fields of the VMCB control area carry the same sixty-four bits in the
//! same order, so this module defines them once, as [`Event`]. `EVENTINJ`, at
//! offset 0A8h, is what a hypervisor writes to make the guest take an exception
//! or an interrupt. `EXITINTINFO`, at offset 088h, is what the processor writes
//! to say that the intercept it is reporting happened while the guest was
//! already delivering one through its interrupt descriptor table. They are two
//! ends of one problem: an event the guest is owed, written down in a form that
//! outlives the exit and can be handed back.
//!
//! Handing it back is not a nicety. By the time an intercept lands, an external
//! interrupt that was being delivered has already been acknowledged to the
//! controller and is pending nowhere any more — `EXITINTINFO` is the only
//! remaining evidence that the guest was owed it. A hypervisor that resumes
//! without re-injecting what it finds there loses the event outright.
//!
//! # An injected event is delivered, not proposed
//!
//! The guest takes it unconditionally, before its first instruction, and it is
//! not subject to intercept checks — a page fault written into `EVENTINJ`
//! reaches the guest's handler whether or not `#PF` is intercepted, which is
//! exactly what makes re-delivery possible at all. Exceptions raised *during*
//! that delivery are a different matter: those are ordinary guest exceptions
//! and are checked against the intercepts as usual, so a fault while pushing
//! the injected frame comes straight back out as a `#VMEXIT`.
//!
//! With virtual NMI masking disabled, an injected non-maskable interrupt does
//! not block delivery of further NMIs, where one the processor raised itself
//! blocks them until `IRET`. Enabling virtual NMI masking makes the processor
//! track that state for injected NMIs too; otherwise the hypervisor tracks the
//! window through an `IRET` intercept.
//!
//! # A malformed event does not deliver badly — it fails the VMRUN
//!
//! An event that is impossible in the guest's current mode, a `#BR` while the
//! guest is in 64-bit mode being the architecture's own example, does not
//! produce a wrong delivery. VMRUN exits immediately with `VMEXIT_INVALID` and
//! not one guest instruction runs. So does a reserved value in the type field,
//! and so does a type of exception whose vector does not name an exception —
//! which includes vector 2, because that is the non-maskable interrupt, and
//! NMI is a type of its own here rather than an exception.
//!
//! That is why an [`Event`] is built by naming what it is rather than by
//! setting bits. The difference between an event the guest takes and a guest
//! that never starts is a three-bit field and the vector beside it, and the
//! constructors are what keep the two apart.
//!
//! # Two events that cannot be injected at all
//!
//! A software interrupt needs the VMCB's next-RIP field to say where the guest
//! resumes once its handler returns, and that field is optional — support is
//! reported by `CPUID Fn8000_000A_EDX[NRIPS]`. Where it is absent `EVENTINJ`
//! cannot express the event correctly, and the hypervisor has to emulate the
//! injection instead of asking for it.
//!
//! The `#DB` fault a guest's ICEBP instruction raises cannot be injected
//! either. ICEBP performs no descriptor privilege check, where injecting a
//! software interrupt does, so the two are not the same event and no field here
//! names the difference; that injection is emulated as well.
//!
//! # Reading `EXITINTINFO`
//!
//! A set valid bit there means the intercept arrived in the middle of the
//! guest's own delivery of that event: it has not reached its handler, and the
//! hypervisor owes it to the guest once whatever caused the intercept is dealt
//! with. The classification is not the one the instruction names suggest —
//! INT1, also called ICEBP, along with INT3 and INTO, all record as exceptions,
//! and only `INT n`, opcode `CDh`, records as a software interrupt.
//!
//! When several exceptions pile up, the field records the aggregate of all of
//! them but the last. A `#GP` that resolves into a `#DF` on the way, and an
//! intercepted `#PF` taken while delivering that, leaves the `#DF` here and the
//! page fault in the exit code and exit information fields. Re-injecting what
//! is here resumes the guest; reflecting the intercepted fault into the guest
//! instead means combining the two by the architecture's own rules, and a `#DF`
//! plus a `#PF` is a triple fault.

use bitfield_struct::bitfield;
use descriptors::Vector;

#[bitfield(u64, new = false)]
#[derive(PartialEq, Eq)]
/// One event, in the encoding both `EVENTINJ` and `EXITINTINFO` use.
///
/// Written into the first, it is an event the guest will take before it runs
/// again. Read out of the second, it is an event the guest was already taking
/// when an intercept interrupted the delivery. Nothing in the bits
/// distinguishes the two directions, which is the point: an event read out of
/// one can be written straight into the other.
pub struct Event {
    /// The vector the event arrives on. Ignored when the kind is a non-maskable
    /// interrupt, and required to name an exception when the kind is one.
    #[bits(8, from = vector_from_bits, into = vector_into_bits)]
    pub vector: Vector,
    /// What the vector is qualifying, which decides how the processor delivers
    /// the event and whether it reads the vector at all.
    #[bits(3)]
    pub kind: EventKind,
    /// Whether an error code goes with the event: on injection, whether the
    /// processor pushes [`Event::error_code`] onto the guest's stack; on an
    /// exit, whether the delivery that was interrupted would have pushed one.
    pub error_code_valid: bool,
    /// Reserved.
    #[bits(19)]
    __: u32,
    /// Whether there is an event here at all. Clear in `EVENTINJ` nothing is
    /// injected, and clear in `EXITINTINFO` the intercept did not happen during
    /// a delivery and the guest is owed nothing.
    pub valid: bool,
    /// The error code, meaningful only while [`Event::error_code_valid`] is
    /// set. The processor ignores this field on injection otherwise, and leaves
    /// it undefined on exit.
    pub error_code: u32,
}

impl Event {
    /// No event.
    ///
    /// All zeroes, which is how the architecture spells the absence of one: the
    /// valid bit is what the processor looks at, and with it clear the rest of
    /// the field means nothing. This is the state `EVENTINJ` is left in
    /// whenever the guest is owed nothing, and it is named for what it
    /// means rather than for being an empty value, because writing an
    /// "event" is precisely what it does not do.
    #[must_use]
    pub const fn none() -> Self {
        Self::from_bits(0)
    }

    /// An exception the guest takes as though it had raised it itself.
    ///
    /// Whether an error code accompanies it is not a caller's choice: the
    /// architecture fixes which vectors push one, so the bit is taken from
    /// `vector` and the code itself is left zero.
    /// [`Event::exception_with_code`] supplies a code where there is room
    /// for one.
    ///
    /// `vector` must name an exception. Anything else fails the VMRUN with
    /// `VMEXIT_INVALID` before a single guest instruction runs, and vector 2 is
    /// one of those anything-elses — it is the non-maskable interrupt, which
    /// [`Event::nmi`] injects.
    ///
    /// Vectors 3 and 4 are delivered the way the traps raised by INT3 and INTO
    /// are, meaning the processor checks the privilege level of the interrupt
    /// descriptor before entering the handler. A guest whose own gate is
    /// unreachable from the privilege level it is running at therefore takes a
    /// `#GP` instead of the breakpoint it was sent.
    #[must_use]
    pub const fn exception(vector: Vector) -> Self {
        Self::none()
            .with_valid(true)
            .with_kind(EventKind::Exception)
            .with_vector(vector)
            .with_error_code_valid(vector.pushes_error_code())
    }

    /// The same, carrying `error_code`.
    ///
    /// The code is written whatever `vector` is, because the processor reads it
    /// only when the error-code-valid bit is set and that bit still comes from
    /// the vector. A code handed in for a vector that pushes none is therefore
    /// inert rather than eight bytes of surprise on the guest's stack.
    #[must_use]
    pub const fn exception_with_code(vector: Vector, error_code: u32) -> Self {
        Self::exception(vector).with_error_code(error_code)
    }

    /// An external interrupt on `vector`, indistinguishable to the guest from
    /// one an interrupt controller delivered.
    ///
    /// This is how an interrupt that was taken out of the guest's hands is
    /// given back to it — one recovered from `EXITINTINFO`, or one a virtual
    /// device raised on the guest's behalf.
    #[must_use]
    pub const fn interrupt(vector: Vector) -> Self {
        Self::none()
            .with_valid(true)
            .with_kind(EventKind::External)
            .with_vector(vector)
    }

    /// A non-maskable interrupt.
    ///
    /// The processor ignores the vector field for this kind. Vector 2 is
    /// written into it regardless, so that a dump of the raw field reads as
    /// the event it is rather than as vector zero.
    ///
    /// With virtual NMI masking enabled, the processor blocks further NMIs
    /// until the guest returns; otherwise the hypervisor must track that
    /// window.
    #[must_use]
    pub const fn nmi() -> Self {
        Self::none()
            .with_valid(true)
            .with_kind(EventKind::Nmi)
            .with_vector(NMI_VECTOR)
    }

    /// A software interrupt, the event `INT n` raises — and only that opcode,
    /// since INT1, INT3 and INTO are exceptions in this encoding.
    ///
    /// Injectable only on a processor that supports the VMCB's next-RIP field,
    /// which is where the guest's resumption address comes from. Where that
    /// support is missing the hypervisor must emulate the injection itself,
    /// because the field cannot describe the event completely.
    #[must_use]
    pub const fn software(vector: Vector) -> Self {
        Self::none()
            .with_valid(true)
            .with_kind(EventKind::Software)
            .with_vector(vector)
    }
}

/// What kind of event a vector is qualifying.
///
/// The four defined encodings are not interchangeable. The kind is what decides
/// whether the vector field is read at all, whether the privilege level of the
/// interrupt descriptor is checked before the handler is entered, and whether
/// the injection needs the next-RIP field to be supported.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum EventKind {
    /// An external or virtual interrupt: what arrives from an interrupt
    /// controller, and what a hypervisor injects for a virtual device.
    External = 0,
    /// Any of the encodings the architecture leaves undefined — 1, 5, 6 and 7.
    Reserved = 1,
    /// A non-maskable interrupt, whose vector field the processor ignores.
    Nmi = 2,
    /// An exception, fault or trap alike. The vector must name one.
    Exception = 3,
    /// A software interrupt, raised by `INT n` and by nothing else.
    Software = 4,
}

impl EventKind {
    /// The kind this encoding names, with everything undefined collapsing onto
    /// [`EventKind::Reserved`].
    ///
    /// Nothing here can fail, and nothing here needs to: this direction exists
    /// for `EXITINTINFO`, which the processor fills in, and refusing a value it
    /// wrote would leave a hypervisor unable to read the field at all.
    const fn from_bits(value: u8) -> Self {
        match value {
            0 => Self::External,
            2 => Self::Nmi,
            3 => Self::Exception,
            4 => Self::Software,
            _ => Self::Reserved,
        }
    }

    /// The encoding itself.
    ///
    /// A reserved kind encodes back to a reserved value rather than to one of
    /// the four defined ones, and deliberately so, because this is the
    /// direction that reaches hardware: VMRUN refuses a reserved type with
    /// `VMEXIT_INVALID`. A field that arrived reserved therefore stays reserved
    /// and fails loudly, instead of quietly becoming an external interrupt the
    /// guest never had coming. Which of the four reserved encodings it came
    /// from is not preserved, since nothing can act on the difference.
    const fn into_bits(self) -> u8 {
        self as u8
    }
}

#[bitfield(u64)]
#[derive(PartialEq, Eq)]
/// Whether the guest can take an interrupt right now, at control offset 068h.
///
/// Loaded on VMRUN and saved on `#VMEXIT`, so a guest interrupted at an
/// instruction boundary where interrupts are not recognized resumes at one.
/// Without that, every exit in such a window would quietly widen it.
pub struct InterruptState {
    /// Whether the guest is in an interrupt shadow: the single-instruction
    /// window, after an `STI` that sets `RFLAGS.IF` or a `MOV` to `SS`, during
    /// which the processor recognizes no interrupts and certain debug traps.
    /// The instruction the shadow covers has not run yet, and injecting into
    /// the guest here delivers an interrupt the guest arranged not to take.
    pub interrupt_shadow: bool,
    /// The guest's `RFLAGS.IF`, written back to the VMCB on `#VMEXIT`.
    ///
    /// Meaningful for the encrypted-virtualization extension pulzar does not
    /// implement, where the guest's register state is not readable out of the
    /// save area and this is the only place the flag can be seen.
    pub guest_interrupt_mask: bool,
    /// Reserved.
    #[bits(62)]
    __: u64,
}

/// The vector a non-maskable interrupt always arrives on, which is also the one
/// vector a type of exception may not name.
const NMI_VECTOR: Vector = Vector::new(2);

/// The vector an eight-bit field names.
const fn vector_from_bits(bits: u8) -> Vector {
    Vector::new(bits)
}

/// The eight bits naming a vector.
const fn vector_into_bits(vector: Vector) -> u8 {
    vector.number()
}

const _: () = assert!(
    size_of::<Event>() == 8,
    "an event is one quadword of the control area, in EVENTINJ and EXITINTINFO alike"
);
const _: () = assert!(
    size_of::<InterruptState>() == 8,
    "the interrupt state is one quadword of the control area"
);
