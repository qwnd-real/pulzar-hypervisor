//! What a guest is owed, and how it comes to take it.
//!
//! Deciding *that* a guest should take an interrupt is the interrupt
//! controller's job. Deciding *whether it can right now*, and putting it where
//! the processor will find it, is this crate's, and the two are kept apart
//! because they fail for unrelated reasons: the first is the Intel
//! architecture's priority rules, the second is what AMD's virtualization
//! extension will accept in a control block.
//!
//! # Two mechanisms, and why both
//!
//! The extension offers two ways to give a guest an interrupt, and a hypervisor
//! that used only one would be wrong in one direction or the other.
//!
//! `EVENTINJ` delivers unconditionally. The processor takes the event on entry
//! whatever the guest's flags say, so it is only correct once the guest has
//! been established to be willing — flags set, no interrupt shadow, priority
//! low enough. When that holds it is exactly right, and it is what actually
//! delivers every interrupt here.
//!
//! The virtual-interrupt fields deliver *conditionally*: the processor holds
//! the interrupt until the guest becomes willing. That is the wrong mechanism
//! for delivery — the guest would take a vector whose priority the host never
//! re-checked, against a controller whose state moved on — but it is exactly
//! the right mechanism for a *doorbell*. So the interrupt is never actually
//! delivered that way. `V_IRQ` is armed with the `VINTR` intercept purely to be
//! told the moment the window opens, at which point the interrupt is withdrawn
//! and re-decided from the controller's current state and injected properly.
//!
//! # The priority armed with the doorbell is what makes it ring
//!
//! The processor raises that exit when the guest's flag admits an interrupt
//! *and* the armed priority outranks the guest's own task priority. Both halves
//! matter, and the second is the only way the host hears about the register
//! that holds it: with interrupt masking virtualized, the guest's writes to its
//! task priority through the control register go straight into the control
//! block and take no exit at all. A hypervisor that decided for itself that the
//! priority was too high, and armed nothing, would be waiting for an exit that
//! only the guest lowering that priority could cause — and it has just arranged
//! for that not to cause one.
//!
//! So the vector the doorbell carries is the highest one the controller has
//! that is not already outranked by something in service, whether or not the
//! guest's task priority currently admits it, and the priority armed with it is
//! that vector's own. The comparison is then made where the register lives.
//!
//! # An interrupted delivery is owed back
//!
//! An intercept can happen part-way through the processor delivering an event —
//! between reading the descriptor and entering the handler. The event is then
//! neither taken nor still pending anywhere the hardware will find it, and a
//! hypervisor that did not put it back would silently lose an interrupt.
//! `EXITINTINFO` describes such an event, and [`Pending::harvest`] must be the
//! first thing any exit path does, before anything can overwrite the injection
//! field.
//!
//! # Non-maskable interrupts
//!
//! A guest's non-maskable interrupts are not intercepted: one that arrives
//! while the guest is running is the guest's to take through its own descriptor
//! table. What this crate handles is the other case — one that arrives while
//! the *host* is running, in the window between a world switch restoring host
//! state and re-entering the guest, which the host takes and the guest is still
//! owed.
//!
//! Giving it back needs the architecture's blocking rule honoured: from taking
//! one until the next `IRET`, a processor takes no further non-maskable
//! interrupt. Where the processor virtualizes that, [`Blocking::Virtual`] hands
//! it the flag and it is tracked in hardware. Where it does not,
//! [`Blocking::Iret`] intercepts `IRET` and tracks the window here, which costs
//! an exit per return from a handler and is why it is the fallback.

#![no_std]

use descriptors::Vector;
use log::{trace, warn};
use processor::SvmFeatures;
use svm::{CleanBits, Event, EventKind, intercept::Intercepts1};
use vcpu::Vcpu;

/// What a guest is owed and what has been done about it, for one processor.
///
/// Owned by whatever runs that processor's exit loop and touched by nothing
/// else — every field here is decided and acted on between one exit and the
/// next entry, on the processor the guest runs on. That is why none of it is
/// atomic, and it is the reason the interrupt controller's state is kept
/// somewhere else: that *is* reached from other processors.
#[derive(Clone, Copy, Debug)]
pub struct Pending {
    interrupted: Option<Interrupted>,
    nmi: u8,
    nmi_blocked: bool,
    awaiting_iret: Option<u64>,
    shutdown: bool,
    blocking: Blocking,
    /// The vector an interrupt window is currently armed for, or `None` when
    /// none is. The vector and not merely the fact, because the priority armed
    /// beside it is that vector's own — so a window armed for a different
    /// vector has to be rewritten rather than left alone.
    window: Option<Vector>,
}

impl Pending {
    /// Nothing owed, on a processor whose non-maskable interrupts are tracked
    /// whichever way it supports.
    #[must_use]
    pub fn new() -> Self {
        Self {
            interrupted: None,
            nmi: 0,
            nmi_blocked: false,
            awaiting_iret: None,
            shutdown: false,
            blocking: Blocking::of(processor::svm().map(|svm| svm.features)),
            window: None,
        }
    }

    /// Takes back an event whose delivery an intercept interrupted.
    ///
    /// The first thing an exit path does, before the exit is answered and
    /// before anything else can write the injection field. An event read here
    /// goes back in ahead of anything the controller has since decided,
    /// because it is not a new interrupt competing on priority — it is one the
    /// guest had already been given and was part-way through taking.
    pub fn harvest(&mut self, vcpu: &mut Vcpu, next_rip: Option<u64>) {
        let event = vcpu.control().exit_interrupt_info;
        if !event.valid() {
            return;
        }
        // A non-maskable interrupt interrupted mid-delivery is owed back as one
        // rather than as an event, so that it goes through the blocking rules
        // the rest of this type keeps.
        if event.kind() == EventKind::Nmi {
            self.raise_nmi();
            trace!("inject: an nmi delivery was interrupted; it stays owed");
            return;
        }
        if self.interrupted.is_some() {
            warn!("inject: replacing an interrupted event with a newer interrupted event");
        }
        self.interrupted = Some(Interrupted { event, next_rip });
        trace!("inject: took back an interrupted {event:?}");
    }

    /// Whether replaying this interrupted event needs its following address.
    #[must_use]
    pub fn needs_next_rip(vcpu: &Vcpu) -> bool {
        let event = vcpu.control().exit_interrupt_info;
        event.valid()
            && (event.kind() == EventKind::Software
                || (event.kind() == EventKind::Exception
                    && matches!(event.vector().number(), 3 | 4)))
    }

    /// Queues an exception raised while answering a guest exit.
    ///
    /// An interrupted event always stays ahead of a newly raised exception: it
    /// was already being delivered when the processor left the guest. A valid
    /// injection field is moved into that slot first, so no event is
    /// overwritten while the exit handler turns an access failure into an
    /// exception.
    pub fn raise_exception(&mut self, vcpu: &mut Vcpu, event: Event) {
        debug_assert!(event.valid() && event.kind() == EventKind::Exception);
        let previous = self
            .interrupted
            .take()
            .map(|interrupted| interrupted.event)
            .or_else(|| {
                let injected = vcpu.control().event_injection;
                injected.valid().then_some(injected)
            });
        if let Some(previous) = previous {
            match combine_exceptions(previous, event) {
                Combined::Event(combined) => Self::inject(vcpu, combined),
                Combined::Shutdown => {
                    self.shutdown = true;
                    vcpu.control_mut().event_injection = Event::none();
                    vcpu.soil(CleanBits::INTERRUPT);
                }
            }
            return;
        }
        Self::inject(vcpu, event);
    }

    /// Records that this processor took a non-maskable interrupt the guest is
    /// owed.
    ///
    /// Called from the host's own handler, which runs in interrupt context on
    /// this processor between a world switch and the next entry.
    pub const fn raise_nmi(&mut self) {
        if self.nmi < 2 {
            self.nmi += 1;
        }
    }

    /// Records that an intercepted return is about to execute.
    ///
    /// The intercept is removed for one attempt and the blocking state remains
    /// set until a later exit proves that the instruction pointer advanced.
    /// An event that wins before the return executes therefore cannot open the
    /// NMI window early.
    pub fn intercepted_iret(&mut self, vcpu: &mut Vcpu) {
        self.awaiting_iret = Some(vcpu.save().rip);
        vcpu.control_mut().intercept_1.remove(Intercepts1::IRET);
        vcpu.soil(CleanBits::INTERCEPTS);
    }

    /// Finishes a return whose execution a later exit has proved.
    pub fn complete_iret(&mut self, vcpu: &Vcpu) {
        if self.awaiting_iret.is_some_and(|rip| vcpu.save().rip != rip) {
            self.awaiting_iret = None;
            self.nmi_blocked = false;
        }
    }

    /// Forgets everything the guest was owed, which is what resetting the
    /// processor means.
    ///
    /// An interrupt owed to a guest that has since been reset is owed to
    /// nobody: the handler that would have taken it belongs to an operating
    /// system this processor is no longer running. So the control block is
    /// cleared as well as the bookkeeping — an injection field left valid
    /// across a reset would deliver, through the reset guest's own
    /// descriptor table, an event that predates it.
    ///
    /// The window and the non-maskable blocking are both withdrawn outright
    /// rather than through [`Pending::arm_window`], because one of them is not
    /// something that function has any business clearing: a processor reset
    /// part-way through a non-maskable interrupt would otherwise carry the
    /// blocking flag into its next life and take no further one, ever.
    pub fn reset(&mut self, vcpu: &mut Vcpu) {
        self.interrupted = None;
        self.nmi = 0;
        self.nmi_blocked = false;
        self.awaiting_iret = None;
        self.shutdown = false;
        self.window = None;
        let control = vcpu.control_mut();
        control.event_injection = Event::none();
        control.interrupt_control = control
            .interrupt_control
            .with_virtual_irq_pending(false)
            .with_virtual_vector(0)
            .with_virtual_priority(0)
            .with_virtual_nmi_masked(false);
        control
            .intercept_1
            .remove(Intercepts1::VINTR | Intercepts1::IRET);
        vcpu.soil(CleanBits::INTERRUPT | CleanBits::INTERCEPTS);
    }

    /// Whether anything at all is owed, which is what decides whether a guest
    /// that halted should be woken.
    #[must_use]
    pub const fn owed(&self) -> bool {
        self.interrupted.is_some() || self.nmi != 0 || self.shutdown
    }

    /// Whether exception delivery reached the architectural shutdown state.
    #[must_use]
    pub const fn shutdown(&self) -> bool {
        self.shutdown
    }

    /// Decides what the guest takes on its next entry, and puts it there.
    ///
    /// `deliverable` is the highest-priority vector the controller says the
    /// guest should take now, already checked against its task priority — or
    /// `None` if it should take nothing. The answer says what was actually
    /// injected, because a candidate is only consumed if it went in: the
    /// controller must not move a vector from requested to in-service for an
    /// injection that did not happen.
    ///
    /// `blocked` is the highest-priority vector the controller has that only
    /// the guest's task priority may be holding back, and is what an
    /// interrupt window is armed for. It is a wider question than
    /// `deliverable` on purpose: the guest changes that priority without
    /// exiting, so a window armed only for what the priority already admits
    /// would leave the guest lowering it and nothing noticing.
    pub fn commit(
        &mut self,
        vcpu: &mut Vcpu,
        deliverable: Option<Vector>,
        blocked: Option<Vector>,
    ) -> Injected {
        let injected = self.choose(vcpu, deliverable);
        // Armed whenever the controller is holding something the guest has not
        // been given, whichever of the two things is holding it back — its
        // interrupt flag, or its task priority. Withdrawn as soon as nothing is
        // waiting, because left armed it would exit on every window the guest
        // opens for the rest of its life.
        //
        // An injection of something else — a requeued event, a non-maskable
        // interrupt — leaves the window armed rather than withdrawing it. What
        // is waiting is still waiting, and the guest is about to run a handler
        // for something quite unrelated to it.
        let waiting = blocked.filter(|_| !matches!(injected, Injected::Interrupt(_)));
        self.arm_window(vcpu, waiting);
        injected
    }

    /// Which of the things owed goes in, in the order the architecture
    /// requires.
    fn choose(&mut self, vcpu: &mut Vcpu, candidate: Option<Vector>) -> Injected {
        // An injection field that still holds a valid event means the last
        // entry never happened — the processor refused the control block, or
        // the exit came before delivery. Overwriting it would lose that event.
        if vcpu.control().event_injection.valid() {
            trace!("inject: the injection field still holds a valid event; nothing new goes in");
            return Injected::Nothing;
        }
        // Ahead of everything, because this is not a new event: the guest was
        // already taking it.
        if let Some(interrupted) = self.interrupted.take() {
            interrupted.inject(vcpu);
            trace!("inject: requeued interrupted {:?}", interrupted.event);
            return Injected::Requeued;
        }
        if self.nmi != 0 && self.nmi_deliverable(vcpu) {
            self.nmi -= 1;
            self.block_nmi(vcpu);
            Self::inject(vcpu, Event::nmi());
            trace!("inject: injecting the owed non-maskable interrupt");
            return Injected::Nmi;
        }
        match candidate {
            Some(vector) if window_open(vcpu) => {
                Self::inject(vcpu, Event::interrupt(vector));
                trace!("inject: injecting {vector}");
                Injected::Interrupt(vector)
            }
            Some(vector) => {
                trace!("inject: declining {vector}; the guest's interrupt window is shut");
                Injected::Nothing
            }
            None => Injected::Nothing,
        }
    }

    /// Puts an event where the processor will take it on the next entry.
    fn inject(vcpu: &mut Vcpu, event: Event) {
        vcpu.control_mut().event_injection = event;
        vcpu.soil(CleanBits::INTERRUPT);
    }

    /// Whether a non-maskable interrupt may go in now.
    ///
    /// Unlike a maskable one this ignores the guest's flags entirely — that is
    /// what makes it non-maskable — and asks only whether the guest is already
    /// inside one, and whether it is in the single-instruction shadow after an
    /// instruction that defers recognition.
    fn nmi_deliverable(&self, vcpu: &Vcpu) -> bool {
        if vcpu.control().interrupt_state.interrupt_shadow() {
            return false;
        }
        match self.blocking {
            Blocking::Virtual => !vcpu.control().interrupt_control.virtual_nmi_masked(),
            Blocking::Iret => !self.nmi_blocked,
        }
    }

    /// Starts the window during which the guest takes no further non-maskable
    /// interrupt.
    fn block_nmi(&mut self, vcpu: &mut Vcpu) {
        match self.blocking {
            // The processor sets the blocking flag as it delivers the injected
            // NMI and clears it on the guest's own `IRET`.
            Blocking::Virtual => {}
            // Nothing tracks it, so the return has to be intercepted to learn
            // when the window closes.
            Blocking::Iret => {
                self.nmi_blocked = true;
                self.awaiting_iret = None;
                let control = vcpu.control_mut();
                control.intercept_1 |= Intercepts1::IRET;
                vcpu.soil(CleanBits::INTERCEPTS);
            }
        }
    }

    /// Arms or withdraws the conditional delivery that exists only to say when
    /// the guest becomes willing.
    ///
    /// The vector written here is never the one the guest takes; the interrupt
    /// is re-decided from the controller when the exit arrives. The
    /// *priority* beside it is not decoration, though — it is what the
    /// processor compares against the guest's own task priority to decide
    /// whether the guest is willing, so it has to be the priority of the
    /// vector really waiting. Anything lower arms a window the guest's
    /// priority suppresses; anything higher reports one the guest cannot
    /// take the waiting vector through, and that is an exit which arms
    /// itself again.
    fn arm_window(&mut self, vcpu: &mut Vcpu, waiting: Option<Vector>) {
        let iret =
            self.blocking == Blocking::Iret && self.nmi_blocked && self.awaiting_iret.is_none();
        if waiting == self.window && !iret {
            return;
        }
        // The window itself only changed if the answer differs from the one
        // already written; the `iret` case re-asserts the return intercept
        // without changing it, so only a real change is worth a line.
        let changed = waiting != self.window;
        self.window = waiting;
        if changed {
            match waiting {
                Some(vector) => trace!(
                    "inject: arming the interrupt window for {vector} at priority {:#x}",
                    priority_class(vector)
                ),
                None => trace!("inject: withdrawing the interrupt window"),
            }
        }
        let control = vcpu.control_mut();
        control.interrupt_control = match waiting {
            Some(vector) => control
                .interrupt_control
                .with_virtual_irq_pending(true)
                .with_virtual_vector(vector.number())
                .with_virtual_priority(priority_class(vector)),
            None => control
                .interrupt_control
                .with_virtual_irq_pending(false)
                .with_virtual_vector(0)
                .with_virtual_priority(0),
        };
        control
            .intercept_1
            .set(Intercepts1::VINTR, waiting.is_some());
        // The return from a handler stays intercepted only while a window that
        // nothing else tracks is open.
        if !iret {
            control.intercept_1.remove(Intercepts1::IRET);
        }
        vcpu.soil(CleanBits::INTERRUPT | CleanBits::INTERCEPTS);
    }
}

/// An event taken back from interrupted delivery, with any state its encoding
/// does not carry itself.
#[derive(Clone, Copy, Debug)]
struct Interrupted {
    event: Event,
    next_rip: Option<u64>,
}

impl Interrupted {
    /// Restores the event and the following address software delivery needs.
    fn inject(self, vcpu: &mut Vcpu) {
        if let Some(next_rip) = self.next_rip {
            vcpu.control_mut().next_rip = next_rip;
        }
        Pending::inject(vcpu, self.event);
    }
}

/// Combines two exceptions which arose while the first was being delivered.
///
/// A double fault replaces two contributory faults, or a page fault followed
/// by any non-benign fault. `#DF` followed by another exception is shutdown on
/// real hardware; VMRUN exposes that as a shutdown exit, which this layer does
/// not synthesize because it has no ownership of guest lifecycle policy.
fn combine_exceptions(first: Event, second: Event) -> Combined {
    if first.kind() != EventKind::Exception || second.kind() != EventKind::Exception {
        return Combined::Event(first);
    }
    if first.vector() == Vector::DOUBLE_FAULT {
        return Combined::Shutdown;
    }
    let first_class = exception_class(first.vector());
    let second_class = exception_class(second.vector());
    if (first_class == ExceptionClass::Contributory && second_class == ExceptionClass::Contributory)
        || (first_class == ExceptionClass::PageFault && second_class != ExceptionClass::Benign)
    {
        return Combined::Event(Event::exception_with_code(Vector::DOUBLE_FAULT, 0));
    }
    Combined::Event(second)
}

/// The architectural result of an exception colliding with exception delivery.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Combined {
    /// The event the guest must take next.
    Event(Event),
    /// A fault while delivering `#DF`, which shuts the virtual processor down.
    Shutdown,
}

/// How an exception participates in architectural double-fault combination.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExceptionClass {
    /// Never combines into a double fault.
    Benign,
    /// A segment, stack, or protection fault that combines with another one.
    Contributory,
    /// A page fault, which combines with any non-benign following fault.
    PageFault,
}

/// The architectural double-fault class of `vector`.
const fn exception_class(vector: Vector) -> ExceptionClass {
    match vector.number() {
        14 => ExceptionClass::PageFault,
        0 | 10 | 11 | 12 | 13 => ExceptionClass::Contributory,
        _ => ExceptionClass::Benign,
    }
}

impl Default for Pending {
    fn default() -> Self {
        Self::new()
    }
}

/// What went into the control block, and so what the caller must account for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Injected {
    /// The guest takes nothing on its next entry.
    Nothing,
    /// The guest takes this interrupt, which the controller must now move from
    /// requested to in service.
    Interrupt(Vector),
    /// The guest takes the non-maskable interrupt it was owed.
    Nmi,
    /// An event whose delivery was interrupted went back in. Nothing of the
    /// controller's was consumed.
    Requeued,
}

/// Whether the guest can take a maskable interrupt on its next entry.
///
/// Three things have to hold, and the first of them is only readable at all
/// because the host does not let the guest's flag mask the host's interrupts:
/// with virtualized interrupt masking the guest's `RFLAGS.IF` governs only what
/// the guest takes, so reading it says exactly what it looks like it says.
#[must_use]
pub fn window_open(vcpu: &Vcpu) -> bool {
    let state = vcpu.control().interrupt_state;
    // The single-instruction window after `STI` or a move to the stack segment,
    // during which the processor recognizes nothing. Injecting here delivers an
    // interrupt the guest arranged not to take.
    if state.interrupt_shadow() {
        return false;
    }
    vcpu.save().rflags & INTERRUPT_FLAG != 0
}

/// Prepares a control block to have interrupts taken away from its guest and
/// given back on the host's terms.
///
/// Two things, and neither is optional. Maskable interrupts are intercepted, so
/// that one arriving while the guest runs leaves the guest rather than being
/// delivered through the guest's own descriptor table to a handler that is not
/// the host's. And interrupt masking is virtualized, so that the guest's own
/// flag stops governing whether the *host* can be interrupted — without which a
/// guest that clears its flag would stop the machine answering its devices.
///
/// Physical non-maskable interrupts are intercepted at the SVM boundary. This
/// is required even with hardware vNMI: the intercept is what lets the host
/// distinguish an NMI taken during guest execution from one arriving while the
/// host is between entries. Hardware vNMI then tracks the guest's NMI-blocking
/// state without the software IRET fallback.
pub fn arm(vcpu: &mut Vcpu) {
    let features = processor::svm().map(|svm| svm.features);
    let control = vcpu.control_mut();
    control.intercept_1 |= Intercepts1::INTR | Intercepts1::NMI;
    control.interrupt_control = control
        .interrupt_control
        .with_intercept_interrupt_masking(true);
    if Blocking::supports_virtual(features) {
        control.interrupt_control = control
            .interrupt_control
            .with_virtual_gif(true)
            .with_virtual_gif_enable(true)
            .with_virtual_nmi_enable(true);
    }
    vcpu.soil(CleanBits::INTERCEPTS | CleanBits::INTERRUPT);
}

/// How the window during which a guest takes no further non-maskable interrupt
/// is kept track of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Blocking {
    /// The processor keeps the flag itself and clears it on the guest's own
    /// return, costing nothing.
    Virtual,
    /// The processor does not, so the return is intercepted and the window is
    /// tracked in software — an exit per return from any handler.
    Iret,
}

impl Blocking {
    /// Whether the processor can track virtual NMI blocking in hardware.
    const fn supports_virtual(features: Option<SvmFeatures>) -> bool {
        matches!(features, Some(features) if features.contains(SvmFeatures::VNMI) && features.contains(SvmFeatures::VGIF))
    }

    /// Which mechanism a processor with these features offers.
    fn of(features: Option<SvmFeatures>) -> Self {
        if Self::supports_virtual(features) {
            Self::Virtual
        } else {
            warn!(
                "inject: this processor does not virtualize non-maskable interrupt masking; intercepting the return from a handler instead"
            );
            Self::Iret
        }
    }
}

/// A vector's interrupt-priority class, which is the upper nibble.
const fn priority_class(vector: Vector) -> u8 {
    vector.number() >> PRIORITY_SHIFT
}

/// How far a vector is shifted to leave its priority class.
const PRIORITY_SHIFT: u8 = 4;

/// The bit of the guest's flags that says it is willing to take a maskable
/// interrupt.
const INTERRUPT_FLAG: u64 = 1 << 9;

#[cfg(test)]
mod tests {
    use descriptors::Vector;
    use processor::SvmFeatures;
    use svm::Event;

    use super::{Blocking, Combined, ExceptionClass, combine_exceptions, exception_class};

    #[test]
    fn virtual_nmi_requires_vgif_too() {
        assert!(!Blocking::supports_virtual(Some(SvmFeatures::VNMI)));
        assert!(!Blocking::supports_virtual(Some(SvmFeatures::VGIF)));
        assert!(Blocking::supports_virtual(Some(
            SvmFeatures::VNMI | SvmFeatures::VGIF
        )));
    }

    #[test]
    fn exception_classes_match_double_fault_rules() {
        assert_eq!(
            exception_class(Vector::new(13)),
            ExceptionClass::Contributory
        );
        assert_eq!(exception_class(Vector::new(14)), ExceptionClass::PageFault);
        assert_eq!(exception_class(Vector::new(6)), ExceptionClass::Benign);
    }

    #[test]
    fn contributory_faults_combine_into_double_fault() {
        let combined = combine_exceptions(
            Event::exception(Vector::new(13)),
            Event::exception(Vector::new(11)),
        );
        assert!(matches!(
            combined,
            Combined::Event(event) if event.vector() == Vector::DOUBLE_FAULT
        ));
    }

    #[test]
    fn fault_during_double_fault_shuts_down() {
        assert_eq!(
            combine_exceptions(
                Event::exception(Vector::DOUBLE_FAULT),
                Event::exception(Vector::new(14)),
            ),
            Combined::Shutdown
        );
    }
}
