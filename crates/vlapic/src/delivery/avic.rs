//! Completing in software what the hardware started delivering between the
//! guest's own processors.
//!
//! Fixed edge-triggered IPIs go through the hardware without an exit; what
//! reaches here is everything else — the modes the hardware does not
//! implement, the targets it could not reach, the failures it reports. Each
//! of those is answered by the failure's own rule, and the one invariant all
//! of them keep is that exactly one of the hardware and the software delivers
//! each interrupt: the hardware reports which of them already acted before it
//! exited, and nothing here acts twice.
//!
//! # The cause is the only thing that says which of them acted
//!
//! Which authority already delivered is a fact about the past, and the cause
//! the hardware reports is the whole of what states it. Whether the
//! acceleration is still permitted is a different question with a different
//! answer: the machine-wide inhibit, the guest's face and this processor's own
//! inhibit can all move while this processor is inside the guest, so a
//! predicate over them describes the moment the exit is *answered* and never
//! the moment it was raised. It decides how far the acceleration is withdrawn
//! afterwards, and nothing about what becomes of the command.
//!
//! [`answer_for`] is where the two are kept apart, and it is a table because
//! every arm of this exit is a rule the architecture states — one that should
//! be readable against that list rather than followed through a chain of
//! conditions. A predicate tested in front of the match answered six causes
//! with one answer, and re-delivered an interrupt the hardware had already
//! delivered every time it changed underneath a target-not-running exit.
//!
//! # A request and the signal that announces it name one authority
//!
//! Two authorities can be holding an interrupt for a guest, and each has a
//! signal of its own. A request in the software model is announced with the
//! host interrupt [`super::doorbell`] sends, which makes the target leave the
//! guest and consult that model on its way back in. A request in a backing page
//! would be announced with the hardware doorbell, a write of the target's
//! physical identifier that makes its processor re-evaluate the page without
//! leaving the guest at all.
//!
//! Only the first of the two is ever sent from here, and that is not a
//! preference: nothing on the host side writes another processor's page. What
//! the software path accepts, it accepts into the target's model, and a backing
//! page is written by the processor it belongs to — at the entry that hands
//! that model's interrupts to the hardware. A doorbell rung for one of those
//! requests would name a page the vector is not in: the target's hardware would
//! answer it by finding nothing, no exit would be raised, and nothing would be
//! left that could raise one.
//!
//! What the *hardware* deposits in a page it announces itself, and the one case
//! it cannot — a target that was not in the guest — is the only wake these
//! paths owe: a host interrupt each, counted in the sending controller, because
//! how many of them a guest costs is the measure of how often the acceleration
//! cannot finish what it started.
//!
//! # Which targets those are is asked of the tables, not of the model
//!
//! The exit carries one index and a command may name several processors, so the
//! set has to be worked out again — and the software model is a different
//! oracle from the two tables the hardware resolved through. [`Resolved`] is
//! what it is worked out of instead, and why is argued there.

use core::sync::atomic::{AtomicBool, Ordering};

use apic::LocalApic;
use descriptors::Vector;
use log::{error, trace, warn};
use svm::avic::{IncompleteIpiExit, IpiFailure};

use crate::{
    VlapicError,
    avic::activation,
    delivery::{doorbell, error},
    machine::{current, diagnostics::Report, ownership, registry},
    priority,
    registers::{
        Vlapic,
        base::Mode,
        error::Errors,
        icr::{Command, DestinationMode, Shorthand, Trigger},
    },
};

/// Answers an interrupt the hardware could not finish delivering between the
/// guest's own processors.
///
/// The exit is trap-like — the guest is past the write that asked for it — so
/// what is owed is to finish the delivery in whichever way the failure's rule
/// says, and to withdraw as much of the acceleration as the failure impugns.
/// [`answer_for`] is both of those as one table; performing it is the whole of
/// what is left here, in the order the two halves have to happen in: the report
/// first, so that a line and the demotion it explains cannot be separated by
/// another processor's; then the withdrawal, so that the completion below runs
/// on the path the withdrawal chose; then the command.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub(crate) fn incomplete_ipi(exit: IncompleteIpiExit) -> Result<(), VlapicError> {
    let sender = current()?;
    let command = Command::from_bits(exit.icr());
    let resolved = Resolved::of(command, sender.mode());
    let answer = answer_for(
        exit.cause(),
        activation::active_for(sender),
        resolved.several(),
    );
    say(sender, exit, command, answer);
    match answer.inhibit {
        Inhibit::Nothing => {}
        Inhibit::Processor => sender.inhibit_avic(),
        Inhibit::Machine(reason) => activation::inhibit_machine(reason),
    }
    match answer.completion {
        Completion::Software => activation::complete_command(exit.icr()),
        Completion::Wake => wake(sender, resolved),
        Completion::Discard => {
            // The refusal is the *sender's* error to report, as it is on the
            // software path — which catches the same vector before it names a
            // target, so that one guest mistake in a broadcast does not
            // contaminate the error status of every controller it named. This
            // arm is the only place it can be recorded now that the command
            // reaches no other: the sender's controller never sees the vector
            // again.
            error::noticed(sender, Errors::SEND_ILLEGAL_VECTOR);
            activation::clear_command_busy(sender)
        }
    }
}

/// What answering one reported failure owes: what becomes of the interrupt the
/// guest asked for, and how far the acceleration is withdrawn.
///
/// One value rather than two answers because they are one rule per cause, and
/// keeping them together is what makes the rule readable as a row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Answer {
    /// What becomes of the interrupt.
    completion: Completion,
    /// What stops trusting the acceleration.
    inhibit: Inhibit,
}

/// What becomes of an interrupt the hardware could not finish delivering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Completion {
    /// The software path delivers the command end to end. The hardware
    /// attempted nothing for these, so it owns the delivery whole and
    /// nothing is delivered twice.
    Software,
    /// The hardware set the request bits before it exited and could not
    /// announce them. Only the wake is owed; re-emulating the command would
    /// deliver every interrupt in it a second time.
    Wake,
    /// Nothing delivers it. The architecture refuses the command outright, as
    /// real hardware does, and what is left is the two records a refusal owes:
    /// the illegal vector in the sender's own error status, and the
    /// delivery-status bit the guest may be watching.
    Discard,
}

/// How far a reported failure withdraws the acceleration.
///
/// The scope is the scope of the thing that is wrong, which is what separates
/// the three: a failure the architecture reports in the ordinary course of
/// running a guest impugns nothing, a state out of step with the path this
/// processor is on impugns this processor, and a structure this hypervisor
/// maintains for the whole machine impugns the machine — inhibiting the sender
/// for one of those would leave every other processor addressing the same
/// destination raising the same exit, one per interrupt, until each had been
/// demoted in its turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Inhibit {
    /// Nothing.
    Nothing,
    /// This processor's controller goes back to the software path.
    Processor,
    /// The whole machine does, for the rest of its life, for this reason.
    Machine(&'static str),
}

/// What one reported failure owes, as a table over the three things that decide
/// it.
///
/// `accelerated` is whether the acceleration is still the authority for this
/// controller at the instant the exit is answered. It is a term of the inhibit
/// and of nothing else, and that is the point: the exit itself proves the
/// acceleration was armed when the hardware acted, so a predicate that has
/// since changed says something about this processor's future and nothing about
/// what was delivered. Letting it decide the completion re-delivered a
/// target-not-running command — the hardware's request bits already set, the
/// software's copy added on top, and the guest taking one IPI twice — whenever
/// a peer inhibited the machine while an exit was in flight.
///
/// `several` is whether the command named a set the exit's one index cannot,
/// which is [`Resolved::several`]. It matters for one cause. The delivery
/// sequence resolves the destinations in one step and delivers to them in the
/// next, so an exit means nothing was delivered — except for the invalid-page
/// cause, whose index is documented as a *physical-table* index for a
/// broadcast, which says the hardware can be part-way through a set when it
/// reports it. So for a set, that arm wakes what the hardware may have reached
/// instead of re-emulating a command it may have half-performed; for a single
/// destination there is no part-way and the software path delivers it.
const fn answer_for(cause: IpiFailure, accelerated: bool, several: bool) -> Answer {
    let inhibit = match cause {
        // The tables the hardware read are this hypervisor's own, and every
        // processor resolves through the same ones.
        IpiFailure::InvalidBackingPage => Inhibit::Machine(INVALID_BACKING_PAGE),
        // A failure the architecture reserves for the encrypted-virtualization
        // extension, which this machine does not run: something below is not
        // what it claims, and the machine stops trusting it rather than
        // continuing to guess.
        IpiFailure::UnacceleratedIpi => Inhibit::Machine(SECURE_DELIVERY),
        // Nothing is wrong with the acceleration itself for these. The one thing
        // that can be wrong is that the exit reached a processor the software is
        // delivering for, which is an enable bit or a table out of step with the
        // path the processor is on.
        IpiFailure::TargetNotRunning
        | IpiFailure::InvalidInterruptType
        | IpiFailure::InvalidTarget
        | IpiFailure::InvalidIpiVector => {
            if accelerated {
                Inhibit::Nothing
            } else {
                Inhibit::Processor
            }
        }
    };
    let completion = match cause {
        // The request bits are set and the targets are parked: the wake, and
        // nothing else.
        IpiFailure::TargetNotRunning => Completion::Wake,
        // The vector is one no controller may deliver, and no controller was
        // given it: real hardware discards such a command, and what the refusal
        // owes is a record rather than a delivery.
        IpiFailure::InvalidIpiVector => Completion::Discard,
        // A set the hardware may have been part-way through, per `several`.
        IpiFailure::InvalidBackingPage if several => Completion::Wake,
        // Everything else: the hardware attempted nothing, so the software path
        // owns the delivery whole.
        IpiFailure::InvalidInterruptType
        | IpiFailure::InvalidTarget
        | IpiFailure::InvalidBackingPage
        | IpiFailure::UnacceleratedIpi => Completion::Software,
    };
    Answer {
        completion,
        inhibit,
    }
}

/// Why the machine stops trusting the acceleration when the hardware reads an
/// entry naming a page of controller registers it cannot use.
const INVALID_BACKING_PAGE: &str =
    "a physical-table entry names a backing page the hardware could not use";

/// Why it stops trusting it when the hardware reports the failure the
/// architecture reserves for encrypted-virtualization delivery.
const SECURE_DELIVERY: &str = "the hardware reported a secure-delivery IPI failure";

/// Says what the hardware reported, at the level its cause deserves and no more
/// often than the guest may be allowed to make it say it.
///
/// How many of these exits a guest takes is the guest's own choice, so the two
/// lines above `trace!` are latched, each at the scope of the thing it reports:
/// once per controller for an exit that should not have reached a processor the
/// software is delivering for, and once per machine for the table fault, which
/// is one fault however many processors meet it. An unlatched line here is a
/// denial of service rather than a diagnostic — every byte leaves through a
/// polled serial register, and every other processor that logs blocks behind
/// the same lock while it goes out.
fn say(sender: &Vlapic, exit: IncompleteIpiExit, command: Command, answer: Answer) {
    if matches!(answer.inhibit, Inhibit::Processor) && sender.diagnostics().say(Report::OutOfStep) {
        warn!(
            "vlapic: {} took an exit only hardware delivery raises while the software is the \
             authority for its controller: {:?}, command {:#018x}; this vCPU stays on the software \
             path, and what the hardware had already done is answered by the cause",
            sender.index(),
            exit.cause(),
            exit.icr()
        );
    }
    match exit.cause() {
        IpiFailure::InvalidBackingPage => invalid_backing_page(sender, exit, command),
        // The machine-wide inhibit says this one once, and the exit has nothing
        // to add: the index means nothing for a failure about the command rather
        // than about a destination.
        IpiFailure::UnacceleratedIpi => {}
        IpiFailure::TargetNotRunning => trace!(
            "vlapic: waking the targets of an IPI the hardware delivered and could not announce: \
             command {:#018x}, index {:#x}",
            exit.icr(),
            exit.index()
        ),
        IpiFailure::InvalidInterruptType | IpiFailure::InvalidTarget => trace!(
            "vlapic: completing in software an IPI the hardware does not deliver: {:?}, command \
             {:#018x}",
            exit.cause(),
            exit.icr()
        ),
        IpiFailure::InvalidIpiVector => trace!(
            "vlapic: discarding an IPI of a vector the architecture does not deliver and \
             recording it against the sender: {:#018x}",
            exit.icr()
        ),
    }
}

/// Says once per machine that the hardware read a table entry naming a page of
/// controller registers it could not use, with what makes the entry findable.
///
/// The reported index and, where the index names an entry that could have
/// carried a page at all, the entry this hypervisor believes is at it — which
/// is what tells apart a table walked further than it was described to from one
/// described with a page the processor then refused.
///
/// Which of the two tables the index is an index into is not the field's own to
/// say. The hardware resolves a directed interprocessor interrupt through the
/// logical table where the command names a logical destination and through the
/// physical one otherwise, and it resolves a broadcast or a shorthand through
/// the physical table whatever the destination-mode field holds — so the mode
/// alone does not answer it. A logical entry names a processor rather than a
/// page, which is why there is nothing to show for one.
fn invalid_backing_page(sender: &Vlapic, exit: IncompleteIpiExit, command: Command) {
    if SAID_INVALID_BACKING_PAGE
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return;
    }
    if command.is_broadcast(sender.mode())
        || matches!(command.destination_mode(), DestinationMode::Physical)
    {
        error!(
            "vlapic: {} reported an invalid backing page for an IPI, command {:#018x}, physical \
             table index {:#x}, where this hypervisor has {:?}; the machine returns to software \
             delivery",
            sender.index(),
            exit.icr(),
            exit.index(),
            activation::physical_entry(exit.index())
        );
    } else {
        error!(
            "vlapic: {} reported an invalid backing page for an IPI, command {:#018x}, logical \
             table index {:#x}, whose entry names a processor rather than a page; the machine \
             returns to software delivery",
            sender.index(),
            exit.icr(),
            exit.index()
        );
    }
}

/// Whether the table fault has been reported.
static SAID_INVALID_BACKING_PAGE: AtomicBool = AtomicBool::new(false);

/// Wakes every target of the command the hardware could not announce a delivery
/// to.
///
/// The request bits are the hardware's already; a kick is a wakeup and not a
/// delivery, so a redundant one is only an exit, never a duplicate interrupt.
/// Which targets those are is [`Resolved`]'s to say, and [`owed`]'s to narrow.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub(crate) fn wake_targets(command_bits: u64) -> Result<(), VlapicError> {
    let from = current()?;
    wake(
        from,
        Resolved::of(Command::from_bits(command_bits), from.mode()),
    )
}

/// The same, for a caller that has already worked out what the hardware
/// resolved.
///
/// Taken as an argument rather than derived again because the answer is a term
/// of the arm table as well as of the walk, and one command must not be
/// resolved two ways.
fn wake(from: &Vlapic, resolved: Resolved) -> Result<(), VlapicError> {
    let page = registry::lapics()?;
    for target in page.all() {
        if resolved.names(target.apic_id().get())
            && owed(
                target.index() == from.index(),
                ownership::owns(target),
                activation::is_running(target.apic_id()).unwrap_or(false),
            )
        {
            kick(from, target);
        }
    }
    Ok(())
}

/// Which of the guest's processors the hardware resolved a command to.
///
/// Derived from the command alone, and that is the whole of what it is for. The
/// hardware resolves a physical destination by indexing the physical table with
/// it and a logical one by walking the logical table; the software model is the
/// other oracle — each controller's live face, its logical identifier and its
/// destination format — and the two disagree in every window where a guest is
/// moving one of those. A guest that enables the wider face one processor at a
/// time runs such a window once per processor: the target keeps the flat
/// logical entry it published in the older face, deliberately, so an older-face
/// sender's hardware resolves to it while a snapshot of its *new* mode says it
/// was never addressed. A wake set derived from the model then misses a target
/// the hardware really did deposit a request for, and that target stays parked
/// with a deliverable bit in its page and nothing left that would make it look.
/// A logical-destination write produces the same shape with no mode change in
/// it, because the model and the table do not move at the same instant.
///
/// So the only identity compared here is the one a guest cannot move — the
/// processor's own identifier, which is what indexes the physical table — and
/// every command whose set the exit's single index cannot name resolves to the
/// whole machine rather than to a set worked out from the model. A wake that
/// reaches a processor the command never named costs one exit; a wake that
/// misses one costs a vCPU that never runs again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Resolved {
    /// The sending processor and no other, which is owed no wake whatever names
    /// it: it is outside the guest — it is executing this — and consults its
    /// own controller on the way back in.
    Sender,
    /// The one processor whose identifier indexes the physical-table entry the
    /// hardware read.
    Identifier(u32),
    /// Every processor there is. A broadcast and the two inclusive shorthands
    /// name them all outright; a logical destination may name several, and
    /// which several is the question the model must not be asked.
    Every,
}

impl Resolved {
    /// What the hardware resolved `command` to, the destination field being
    /// read in the width the face `sender` wrote it through gives it.
    ///
    /// That face is the sender's for the same reason
    /// [`Command::is_broadcast`] takes it: the value was written into the
    /// sender's register, and one command reaches controllers that disagree
    /// about which face they are in.
    const fn of(command: Command, sender: Mode) -> Self {
        match command.shorthand() {
            Shorthand::Myself => Self::Sender,
            Shorthand::All | Shorthand::Others => Self::Every,
            Shorthand::None => match command.destination_mode() {
                DestinationMode::Logical => Self::Every,
                DestinationMode::Physical => {
                    if command.is_broadcast(sender) {
                        Self::Every
                    } else {
                        Self::Identifier(command.destination(sender))
                    }
                }
            },
        }
    }

    /// Whether the hardware could have resolved the command to the processor
    /// whose identifier is `apic_id`.
    const fn names(self, apic_id: u32) -> bool {
        match self {
            Self::Sender => false,
            Self::Identifier(identifier) => apic_id == identifier,
            Self::Every => true,
        }
    }

    /// Whether the command named a set the exit's one index cannot, which is
    /// what says the hardware may have been part-way through it — see
    /// [`answer_for`].
    const fn several(self) -> bool {
        matches!(self, Self::Every)
    }
}

/// Whether a target the hardware delivered to is owed the host interrupt that
/// makes it look at what was left in its backing page.
///
/// Three facts, and each excludes the wake for a reason of its own. Written as
/// a decision over values because a wake this refuses is a request bit standing
/// in a page with nothing left to make its processor read it, and because two
/// of the three are answers about a machine that can change underneath the
/// walk. Which targets are put to it at all is [`Resolved`]'s question and not
/// this one.
///
/// - The sender is not one of them, whatever the command named. It is outside
///   the guest — it is executing this — and consults its own controller on the
///   way back in, so an interrupt sent here would be one this processor answers
///   with an empty handler and nothing else. Both the all-inclusive shorthand
///   and a broadcast destination name it, which is how a great deal of firmware
///   and some kernels send.
/// - A processor this hypervisor does not run has no guest to be woken into.
///   The software path reports such a message as one no processor accepted, and
///   there is nothing here to add to that.
/// - A target that is in the guest now entered it after the hardware set the
///   request bit, and an entry re-evaluates the page it is entered with — so
///   the look this would ask for has already happened. Its own away flag is
///   deliberately not consulted, unlike [`doorbell::nudge`]'s: the hardware has
///   just reported the target as not running, and reading the flag would only
///   race that answer.
const fn owed(sender: bool, owned: bool, running: bool) -> bool {
    !sender && owned && !running
}

/// Wakes a target with the host doorbell interrupt, whether or not it said it
/// was away.
///
/// What this leaves out that [`doorbell::nudge`] has is the away flag, and
/// [`owed`] is where that and every other term of the decision are argued.
/// Nothing is re-examined here. The two are counted apart because which
/// authority is holding the interrupt decides where the target finds it.
fn kick(from: &Vlapic, target: &Vlapic) {
    doorbell::interrupt(from, target);
    // Counted against the sender, in the sender's own controller, because this
    // is the sender's path: the target is parked and is not executing anything
    // that could count it. A machine-wide word here was a read-modify-write
    // every processor made on the path the acceleration exists to make cheap,
    // and no processor could tell its own wakes from the machine's.
    from.diagnostics().kicked();
}

/// Gives one processor an interrupt that arrived for its guest through real
/// hardware, while its controller is driven by the acceleration.
///
/// The request is left in the backing page, where the hardware reads it. What
/// real hardware is owed is decided exactly as the software path decides it — a
/// level arrival below the host's own class keeps its acknowledgement until the
/// guest's, and everything else is acknowledged at once — and the same decision
/// is what the page is told: the trigger mode published with the request is
/// level for the arrival whose acknowledgement is being withheld and edge for
/// every other, because that bit is what makes the guest's own acknowledgement
/// raise the exit this hypervisor pays real hardware from. An arrival already
/// acknowledged owes nothing and is handed over as the edge it now is, which is
/// what [`crate::lifecycle::arrival`] gives the model for the same case.
///
/// What this controller will not take is refused exactly as the model refuses
/// it — the architecture's own rule about the vector first, then the guest's
/// own state — and real hardware is settled for it here rather than left
/// holding it: see [`arrival`] and [`refuse`].
///
/// # Errors
///
/// As [`crate::read_msr`].
pub(crate) fn arrive(
    vlapic: &Vlapic,
    local: LocalApic,
    vector: Vector,
    level: bool,
) -> Result<(), VlapicError> {
    let withholdable = crate::lifecycle::arrival::withholdable(vector, level);
    let trigger = match arrival(priority::legal(vector), vlapic.accepting(), withholdable) {
        Arrival::Withheld => {
            // The debt is recorded before the request is published, so that a
            // guest which acknowledges immediately finds the debt already there.
            vlapic.ledger().owe(vector);
            Trigger::Level
        }
        Arrival::Acknowledged => {
            local.end_of_interrupt();
            Trigger::Edge
        }
        Arrival::Illegal => {
            refuse(vlapic, local, vector, withholdable);
            // Recorded on this controller because it is the receiving one, and
            // an arrival has no sender in the guest to charge it against. The
            // self-interrupt arm of the intercepted path records the same error
            // the same way for the same reason.
            error::noticed(vlapic, Errors::RECEIVE_ILLEGAL_VECTOR);
            trace!(
                "vlapic: {} was handed {vector}, which no controller may deliver; the hardware is \
                 not told it arrived",
                vlapic.index()
            );
            return Ok(());
        }
        Arrival::Unaccepted => {
            refuse(vlapic, local, vector, withholdable);
            trace!(
                "vlapic: {} received {vector} but is not accepting it; the hardware will not be \
                 told it arrived",
                vlapic.index()
            );
            return Ok(());
        }
    };
    let requested = activation::request(vector, trigger)?;
    if !requested {
        trace!(
            "vlapic: {vector} coalesced into {}'s backing page",
            vlapic.index()
        );
    }
    // Nothing else is owed. The target is this processor — an arrival runs
    // on the processor the interrupt was addressed to — and this arrival is
    // itself the wake: it interrupted whichever state the processor was in,
    // and every path from here re-enters the guest through the loop that
    // evaluates the backing state on its way in.
    Ok(())
}

/// What an arrival from real hardware is owed, before anything is published
/// into a backing page.
///
/// The two questions [`Vlapic::accept`] asks of the same arrival, in the order
/// it asks them, and then the one this path asks that the model does not.
/// Written as a decision over values because what matters is the parity between
/// the two delivery domains: a vector the model refuses and the page accepts is
/// a request bit for a vector the architecture forbids, standing in a page this
/// hypervisor cannot easily audit, and it would arrive there the moment the
/// host's own vector allocator handed out a lower number than it does today.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arrival {
    /// Published as a level request, with real hardware's acknowledgement
    /// withheld until the guest gives its own.
    Withheld,
    /// Published as an edge, with real hardware acknowledged at once.
    Acknowledged,
    /// Not published: the architecture refuses the vector, and the receiving
    /// controller records the error.
    Illegal,
    /// Not published: the guest has switched its controller off or
    /// software-disabled it, which is a refusal it is entitled to make and
    /// which records nothing in any error status.
    Unaccepted,
}

/// What an arrival is owed, out of the three facts that decide it.
const fn arrival(legal: bool, accepting: bool, withholdable: bool) -> Arrival {
    // The vector first, as the model tests it first: a controller never sets a
    // request bit in the illegal range, whatever else is true of it.
    if !legal {
        return Arrival::Illegal;
    }
    if !accepting {
        return Arrival::Unaccepted;
    }
    if withholdable {
        Arrival::Withheld
    } else {
        Arrival::Acknowledged
    }
}

/// Records that this controller would not take an arrival, and settles what
/// real hardware is holding for it.
///
/// Real hardware accepted the interrupt before this hypervisor saw it and is
/// holding the vector in service. The guest has not been given it and will
/// therefore never acknowledge it, so the one event that could ever produce the
/// acknowledgement hardware is waiting for does not exist — and a vector left
/// in service there is a real controller refusing everything of that vector's
/// own interrupt-priority class or lower on this physical processor for the
/// rest of the machine's life.
///
/// Which of the two settlements is owed is the same question the arrival itself
/// turned on. An arrival whose acknowledgement could never have been withheld
/// has not been withheld, so it is simply issued, exactly as the software path
/// issues it before looking at what the controller answered. One whose
/// acknowledgement *would* have been withheld is a level line nobody has
/// quieted: issuing an acknowledgement for it would clear the sending I/O
/// controller's own record and the still-asserted line would arrive again at
/// once, into a controller that has just refused it. So the debt is written off
/// through the ledger instead, which is what lets a controller able to retire a
/// named vector stop the line rather than pay for it, and what makes a
/// controller that cannot keep the debt on record. That is
/// [`crate::lifecycle::arrival`]'s reasoning for the same case, reached through
/// the same two calls; the debt is taken on first because a debt is only ever
/// written off where one exists.
fn refuse(vlapic: &Vlapic, local: LocalApic, vector: Vector, withholdable: bool) {
    vlapic.diagnostics().declined();
    if withholdable {
        vlapic.ledger().owe(vector);
        vlapic.ledger().abandon(vector, &local);
    } else {
        local.end_of_interrupt();
    }
}

/// Warns once per machine about an arrival shape the acceleration cannot
/// represent faithfully, for the defensive corner that should never be
/// reached.
pub(crate) fn warn_external_once() {
    if WARNED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
    {
        warn!(
            "vlapic: an arrival the real controller never accepted reached the accelerated \
             path; it is delivered through the backing page, and its acknowledgement is the \
             guest's own legacy controller's"
        );
    }
}

/// Whether the external-arrival warning has been said.
static WARNED: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
mod tests {
    //! Four decisions, and between them they are the whole of what these paths
    //! settle without a machine: what each cause the hardware can report owes,
    //! which processors the hardware resolved a command to, which of those is
    //! owed a wake, and what an arrival from real hardware is owed before
    //! anything is published. What performing any of them does — a host
    //! interrupt, a software delivery, a page write — belongs to the modules
    //! that do it and is argued there.

    use svm::avic::IpiFailure;

    use super::{Answer, Arrival, Completion, Inhibit, Mode, Resolved, answer_for, arrival, owed};
    use crate::registers::icr::Command;

    /// Every cause the architecture defines, which is every row the arm table
    /// has: an encoding it does not define is answered by the decoder as the
    /// first of these, deliberately, so nothing else ever reaches the table.
    const CAUSES: [IpiFailure; 6] = [
        IpiFailure::InvalidInterruptType,
        IpiFailure::TargetNotRunning,
        IpiFailure::InvalidTarget,
        IpiFailure::InvalidBackingPage,
        IpiFailure::InvalidIpiVector,
        IpiFailure::UnacceleratedIpi,
    ];

    /// A fixed interrupt on an ordinary vector, as the low half of a command.
    const FIXED: u32 = 0x0000_0030;

    /// The same, addressed by a logical destination.
    const LOGICAL: u32 = FIXED | 1 << 11;

    /// The shorthand field's three named encodings, in the low half.
    const MYSELF: u32 = FIXED | 0b01 << 18;
    const ALL: u32 = FIXED | 0b10 << 18;
    const OTHERS: u32 = FIXED | 0b11 << 18;

    #[test]
    fn every_cause_is_answered_by_its_own_rule() {
        // The table read against the architecture's list of causes, one row per
        // cause and every row asserted whole: what becomes of the command, and
        // how far the acceleration is withdrawn. A row asserted field by field
        // would let a completion moved onto the wrong cause agree with whatever
        // the other field happened to say.
        //
        // `accelerated` is fixed true here, which is the state every one of these
        // exits is really raised in; what changes when it is false is the inhibit
        // alone, and the two tests below are that.
        for several in [false, true] {
            let expected = |cause| match cause {
                IpiFailure::TargetNotRunning => Answer {
                    completion: Completion::Wake,
                    inhibit: Inhibit::Nothing,
                },
                IpiFailure::InvalidInterruptType | IpiFailure::InvalidTarget => Answer {
                    completion: Completion::Software,
                    inhibit: Inhibit::Nothing,
                },
                IpiFailure::InvalidIpiVector => Answer {
                    completion: Completion::Discard,
                    inhibit: Inhibit::Nothing,
                },
                // The one cause the hardware may have been part-way through, and
                // the one whose fault is the machine's rather than this
                // processor's.
                IpiFailure::InvalidBackingPage => Answer {
                    completion: if several {
                        Completion::Wake
                    } else {
                        Completion::Software
                    },
                    inhibit: Inhibit::Machine(super::INVALID_BACKING_PAGE),
                },
                IpiFailure::UnacceleratedIpi => Answer {
                    completion: Completion::Software,
                    inhibit: Inhibit::Machine(super::SECURE_DELIVERY),
                },
            };
            for cause in CAUSES {
                assert_eq!(
                    answer_for(cause, true, several),
                    expected(cause),
                    "{cause:?}, several {several}"
                );
            }
        }
    }

    #[test]
    fn what_becomes_of_the_command_does_not_depend_on_the_acceleration() {
        // The defect the table replaces, stated as the property that forbids it.
        // The hardware's request set is a fact about the past and the predicate
        // is a statement about this processor's future, so no cause may answer
        // differently for it — least of all the not-running one, whose request
        // bits are already set: a peer inhibiting the machine while such an exit
        // was in flight used to turn it into a full software re-emulation, and the
        // guest took the same IPI twice.
        for cause in CAUSES {
            for several in [false, true] {
                assert_eq!(
                    answer_for(cause, true, several).completion,
                    answer_for(cause, false, several).completion,
                    "{cause:?}, several {several}"
                );
            }
        }
        assert_eq!(
            answer_for(IpiFailure::TargetNotRunning, false, false).completion,
            Completion::Wake
        );
    }

    #[test]
    fn an_exit_that_reaches_a_processor_the_software_delivers_for_demotes_it() {
        // What the predicate does decide, and the whole of it: the exit proves the
        // acceleration was armed when the hardware acted, so one answered while
        // the software is the authority is an enable bit or a table out of step
        // with the path the processor is on. The two machine-wide causes are not
        // narrowed by it — a demotion of the machine is already a demotion of this
        // processor.
        for cause in CAUSES {
            let inhibit = answer_for(cause, false, false).inhibit;
            let expected = match cause {
                IpiFailure::InvalidBackingPage => Inhibit::Machine(super::INVALID_BACKING_PAGE),
                IpiFailure::UnacceleratedIpi => Inhibit::Machine(super::SECURE_DELIVERY),
                _ => Inhibit::Processor,
            };
            assert_eq!(inhibit, expected, "{cause:?}");
        }
    }

    #[test]
    fn only_the_arm_that_may_have_been_part_way_through_asks_how_many_were_named() {
        // How many the command named is a term of one rule, because it is the one
        // cause whose index the architecture documents as a table index for a set —
        // which is what says the hardware can have delivered to part of that set
        // already. Every other cause answers the same for one destination and for
        // a machine's worth of them.
        for cause in CAUSES {
            if matches!(cause, IpiFailure::InvalidBackingPage) {
                continue;
            }
            for accelerated in [false, true] {
                assert_eq!(
                    answer_for(cause, accelerated, false),
                    answer_for(cause, accelerated, true),
                    "{cause:?}, accelerated {accelerated}"
                );
            }
        }
        assert_ne!(
            answer_for(IpiFailure::InvalidBackingPage, true, false).completion,
            answer_for(IpiFailure::InvalidBackingPage, true, true).completion
        );
    }

    #[test]
    fn a_directed_physical_command_resolves_to_the_one_identifier_it_names() {
        // The only case the hardware resolved to a single entry, and the only one
        // whose set can be named exactly. The identifier is compared as the
        // hardware compares it — the value that indexes the table — so a
        // processor whose own face cannot express its identifier is still the one
        // a wider sender reaches, and a narrower sender's eight-bit destination
        // is not silently matched against the low byte of a wider identifier.
        let narrow = Command::from_halves(FIXED, 0x0500_0000);
        assert_eq!(Resolved::of(narrow, Mode::XApic), Resolved::Identifier(5));
        let wide = Command::from_bits(0x0000_0105_0000_0030);
        assert_eq!(
            Resolved::of(wide, Mode::X2Apic),
            Resolved::Identifier(0x105)
        );
        assert!(Resolved::Identifier(5).names(5));
        assert!(!Resolved::Identifier(5).names(0x105));
        assert!(!Resolved::Identifier(5).several());
    }

    #[test]
    fn every_set_the_exits_one_index_cannot_name_resolves_to_the_machine() {
        // A broadcast in either face's spelling, both inclusive shorthands, and a
        // logical destination — which the model would have narrowed against each
        // controller's live face, identifier and format, and which is exactly the
        // narrowing that strands a target the hardware really did deliver to. Every
        // one of them names the whole machine here, and a redundant wake is one
        // exit against a vCPU that would otherwise never run again.
        for (command, sender) in [
            (Command::from_halves(FIXED, 0xFF00_0000), Mode::XApic),
            (Command::from_bits(0xFFFF_FFFF_0000_0030), Mode::X2Apic),
            (Command::from_halves(ALL, 0x0500_0000), Mode::XApic),
            (Command::from_halves(OTHERS, 0x0500_0000), Mode::XApic),
            (Command::from_halves(LOGICAL, 0x0200_0000), Mode::XApic),
            (Command::from_bits(0x0000_0002_0000_0830), Mode::X2Apic),
        ] {
            let resolved = Resolved::of(command, sender);
            assert_eq!(
                resolved,
                Resolved::Every,
                "{:#018x} through {sender}",
                command.bits()
            );
            assert!(resolved.several());
            for apic_id in [0, 5, 0x105] {
                assert!(resolved.names(apic_id));
            }
        }
    }

    #[test]
    fn a_command_naming_the_sender_alone_resolves_to_nobody_a_wake_is_owed_to() {
        // The sender is out of the guest — it is answering this exit — and
        // consults its own controller on the way back in, so the one processor
        // this command named is the one processor a wake would do nothing for.
        for sender in [Mode::XApic, Mode::X2Apic] {
            let resolved = Resolved::of(Command::from_halves(MYSELF, 0), sender);
            assert_eq!(resolved, Resolved::Sender);
            assert!(!resolved.several());
            for apic_id in [0, 5, 0x105] {
                assert!(!resolved.names(apic_id));
            }
        }
    }

    #[test]
    fn a_target_that_is_not_in_the_guest_is_owed_the_wake() {
        // The case the exit exists for: the hardware set the request bit in a
        // page whose processor is not looking at it, and nothing but this makes
        // it look.
        assert!(owed(false, true, false));
    }

    #[test]
    fn the_sender_is_never_woken_by_its_own_command() {
        // Every state the rest of the machine can be in, against the one fact
        // that decides it: a broadcast and the all-inclusive shorthand both name
        // the sender, and the sender is out of the guest and about to consult its
        // own controller. Its running bit reads clear at this point — the exit
        // withdrew it — so nothing but this excludes it.
        for owned in [false, true] {
            for running in [false, true] {
                assert!(
                    !owed(true, owned, running),
                    "owned {owned}, running {running}"
                );
            }
        }
    }

    #[test]
    fn nothing_is_owed_a_target_the_hardware_told_or_a_processor_nothing_runs() {
        // In the guest: its own entry re-evaluated the page after the request bit
        // was set. Not this hypervisor's: there is no guest on it to wake, and
        // the software path has already reported the message as accepted by
        // nobody.
        assert!(!owed(false, true, true));
        assert!(!owed(false, false, false));
        assert!(!owed(false, false, true));
    }

    #[test]
    fn a_broadcast_reaches_every_owned_target_but_the_sender() {
        // The two decisions composed, which is what the walk does: the resolution
        // puts every processor to the wake decision, and the wake decision keeps
        // the ones that are this hypervisor's and parked. A target the model would
        // have said was never addressed is among them, because the model was not
        // asked.
        let resolved = Resolved::of(Command::from_halves(ALL, 0), Mode::XApic);
        for (sender, owns, running, woken) in [
            (false, true, false, true),
            (false, true, true, false),
            (false, false, false, false),
            (true, true, false, false),
        ] {
            assert_eq!(
                resolved.names(7) && owed(sender, owns, running),
                woken,
                "sender {sender}, owned {owns}, running {running}"
            );
        }
    }

    #[test]
    fn an_arrival_on_a_vector_no_controller_may_deliver_is_refused_first() {
        // The parity the accelerated path was missing. The model refuses the
        // vector before it asks anything else about the controller, and so does
        // this: whatever else is true, a request bit for a vector the architecture
        // forbids must not be published into a page.
        for accepting in [false, true] {
            for withholdable in [false, true] {
                assert_eq!(
                    arrival(false, accepting, withholdable),
                    Arrival::Illegal,
                    "accepting {accepting}, withholdable {withholdable}"
                );
            }
        }
    }

    #[test]
    fn a_controller_its_guest_switched_off_takes_nothing_and_records_nothing() {
        // The refusal the architecture defines, told apart from the one above
        // because only one of them is an error the controller reports.
        for withholdable in [false, true] {
            assert_eq!(
                arrival(true, false, withholdable),
                Arrival::Unaccepted,
                "withholdable {withholdable}"
            );
        }
    }

    #[test]
    fn what_a_taken_arrival_is_published_as_is_what_hardware_is_owed() {
        // One decision behind both: the trigger mode the page is told is the same
        // answer as whether real hardware's acknowledgement is being withheld,
        // because that bit is what makes the guest's own acknowledgement raise the
        // exit the withheld one is paid from.
        assert_eq!(arrival(true, true, true), Arrival::Withheld);
        assert_eq!(arrival(true, true, false), Arrival::Acknowledged);
    }
}
