//! Interprocessor interrupts: how one processor asks another to do something.
//!
//! An interrupt is the only thing one processor can send another, and it
//! carries nothing but a vector. That is the whole of the mechanism and it is
//! less than it sounds, because a vector says nothing about who sent it — a
//! hypervisor that passes the platform through shares its vectors with devices
//! and with whatever was using the machine before. So this crate adds the three
//! things the hardware does not give.
//!
//! It records, per processor, what was sent to it, so that a handler can tell
//! an interrupt this hypervisor caused from one that merely arrived on the same
//! number. An arrival with nothing owed is answered [`Disposition::Passed`],
//! which hands it to whoever the hypervisor said should have unclaimed
//! interrupts — the seam that will one day give it to a guest.
//!
//! It carries a word of the sender's own choosing alongside, so that a handler
//! is told what to do and not merely that there is something to do. Sends that
//! coalesce have their words folded together by the interrupt's own [`Merge`],
//! which is what keeps a payload honest when the controller collapses several
//! requests into one delivery.
//!
//! And it records what has been *served*, so that a sender can wait. Sending is
//! asynchronous and some things are not: telling every other processor to
//! forget a translation is worthless unless you know they have.
//!
//! # Adding one
//!
//! [`register`] takes a handler and a merge and answers with an [`Ipi`]. The
//! vector it lands on is chosen here and matters to nobody: an interprocessor
//! interrupt is only ever named by the handle. What the payload means is
//! entirely the caller's, and the only thing this crate asks of it is that the
//! merge of two outstanding payloads describes everything both of them did.
//!
//! # Vectors
//!
//! Taken from the top of the range the platform may assign, downwards from
//! [`LAST`]. High because on this architecture a vector's upper nibble is its
//! priority, and an interprocessor interrupt is usually something another
//! processor is blocked waiting on. Below the two the interrupt controller
//! keeps for itself.
//!
//! # Translation shootdown
//!
//! [`install`] sets up one interprocessor interrupt of this crate's own and
//! hands it to the address space subsystem, which has no way to reach another
//! processor and must not gain one — it is underneath everything here. That is
//! the whole of the dependency: a function pointer, going downwards.
//!
//! Its handler invalidates what the payload describes and touches nothing else.
//! In particular it never asks for the address space lock, which is what keeps
//! a processor waiting for that lock from deadlocking against the processor
//! holding it and waiting for this acknowledgement.

#![no_std]

extern crate alloc;

mod mailbox;
mod shootdown;

use alloc::boxed::Box;
use core::num::NonZeroU64;

use apic::{ApicError, Command, Delivery, Target};
use cpu::{CpuError, CpuIndex};
use descriptors::{DescriptorError, Disposition, Interrupt, Vector};
use log::info;
use spin::Once;
use thiserror::Error;

use crate::mailbox::{Counts, Mailbox};

/// The highest vector an interprocessor interrupt may be given, one below the
/// pair the interrupt controller keeps for itself.
pub const LAST: Vector = Vector::new(0xFD);

/// The lowest. Everything from here up outranks every vector a device is given,
/// which is what an interprocessor interrupt wants: the processor sending one
/// is very often waiting for it.
pub const FIRST: Vector = Vector::new(0xF0);

/// How many interprocessor interrupts can exist at once.
///
/// Not a limit chosen here. It is how many vectors the range holds, which is
/// the real bound, and sizing the tables by it means registering one can only
/// fail for the reason that is actually true.
const CAPACITY: usize = (LAST.number() - FIRST.number()) as usize + 1;

/// What a handler is told about the arrival it is running for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Request {
    payload: Option<NonZeroU64>,
    vector: Vector,
}

impl Request {
    /// Everything the outstanding sends merged to, or `None` if there is
    /// nothing left to act on.
    ///
    /// `None` is not an error and not an empty request: it means an earlier run
    /// of this handler already took a payload that this delivery's send had
    /// been folded into, so the work is done and only the acknowledgement is
    /// outstanding. A handler that has nothing to do in that case should do
    /// nothing — it will still be counted as having answered.
    #[must_use]
    pub const fn payload(&self) -> Option<NonZeroU64> {
        self.payload
    }

    /// Which vector it arrived on.
    #[must_use]
    pub const fn vector(&self) -> Vector {
        self.vector
    }
}

/// What runs on the receiving processor when an interrupt it was sent arrives.
///
/// It runs with interrupts masked, in interrupt context, on whichever processor
/// was asked. It must not wait for anything a processor that could be waiting
/// for it might hold — the address space lock above all — and it does not
/// acknowledge the controller: that is done for it, after it returns.
pub type Handler = fn(Request);

/// How two outstanding payloads for one processor become the one payload the
/// single delivery answering both will carry.
///
/// The controller holds one request bit per vector, so a second send to a
/// processor that has not serviced the first does not produce a second
/// interrupt. Whatever the payload means, this is what has to fold two of them
/// into something that describes both — and it must describe *both*, since the
/// delivery that carries the result is the only one either send will get.
///
/// It runs on the sending processor, inside the update of the shared word, so
/// it must be cheap and it must not wait for anything.
pub type Merge = fn(NonZeroU64, NonZeroU64) -> NonZeroU64;

/// One kind of interprocessor interrupt, and the way to send it.
#[derive(Clone, Copy, Debug)]
pub struct Ipi {
    vector: Vector,
    slot: usize,
    merge: Merge,
}

impl Ipi {
    /// Which vector this ended up on.
    #[must_use]
    pub const fn vector(&self) -> Vector {
        self.vector
    }

    /// Sends this to one processor and returns without waiting.
    ///
    /// # Errors
    ///
    /// [`IpiError::NotOnline`] if the processor has not attached, which means
    /// there is nothing there to answer; or whatever writing the command
    /// reported.
    pub fn send(&self, target: CpuIndex, payload: NonZeroU64) -> Result<(), IpiError> {
        let block = cpu::by_index(target).ok_or(IpiError::NotOnline { index: target })?;
        let state = state()?;
        // Resolved before the debt is recorded, so that a controller which is
        // not up refuses the send without leaving a request behind for it.
        let local = apic::local()?;
        // The payload and the debt go in before the command goes out, so that a
        // processor which takes the interrupt immediately still finds what
        // explains it.
        state.mailbox(self.slot, target).owe(payload, self.merge);
        local.send(Command::new(
            Delivery::Fixed(self.vector),
            Target::One(block.apic_id()),
        ))?;
        Ok(())
    }

    /// Sends this to every attached processor but the one sending, and returns
    /// without waiting.
    ///
    /// Addressed one at a time rather than through the controller's
    /// everyone-else shorthand. The shorthand would also reach a processor that
    /// has been started but has not attached yet, which would be an arrival
    /// that processor has nothing owed for and would answer as somebody
    /// else's.
    ///
    /// # Errors
    ///
    /// Whatever writing a command reported. Returns how many were sent.
    pub fn broadcast(&self, payload: NonZeroU64) -> Result<usize, IpiError> {
        let mut sent = 0;
        for target in others() {
            self.send(target, payload)?;
            sent += 1;
        }
        Ok(sent)
    }

    /// Sends this to one processor and waits until it has run the handler.
    ///
    /// # Errors
    ///
    /// [`IpiError::Timeout`] if it did not answer in time, or whatever sending
    /// reported.
    pub fn send_and_wait(
        &self,
        target: CpuIndex,
        payload: NonZeroU64,
        micros: u64,
    ) -> Result<(), IpiError> {
        self.send(target, payload)?;
        self.wait(target, micros)
    }

    /// Sends this to every attached processor but the one sending, and waits
    /// until all of them have run the handler.
    ///
    /// Every one is sent before any is waited for. Waiting for each in turn
    /// would serialize what the machine is perfectly able to do at once, and
    /// the point of this call is usually that all of it has to finish.
    ///
    /// # Errors
    ///
    /// [`IpiError::Timeout`] if any did not answer in time, or whatever sending
    /// reported. Returns how many answered.
    pub fn broadcast_and_wait(&self, payload: NonZeroU64, micros: u64) -> Result<usize, IpiError> {
        let sent = self.broadcast(payload)?;
        for target in others() {
            self.wait(target, micros)?;
        }
        Ok(sent)
    }

    /// Waits until a processor has answered everything it was owed.
    ///
    /// Read once, before the wait, rather than tracked per send: everything
    /// outstanding at this moment includes everything the caller just sent, so
    /// waiting for the mailbox to catch up to it is at least as strong as
    /// waiting for one send in particular — and needs nothing remembered per
    /// target, which is what keeps a broadcast from having to allocate.
    fn wait(&self, target: CpuIndex, micros: u64) -> Result<(), IpiError> {
        let mailbox = state()?.mailbox(self.slot, target);
        let upto = mailbox.outstanding();
        for _ in 0..micros.div_ceil(POLL_MICROS) {
            // Spun rather than slept for most of the wait, because an answer is
            // normally a few hundred cycles away and sleeping would cost more
            // than it saves. The sleep is what bounds the whole thing.
            for _ in 0..SPINS_PER_POLL {
                if mailbox.caught_up(upto) {
                    return Ok(());
                }
                core::hint::spin_loop();
            }
            clock::sleep_micros(POLL_MICROS).map_err(|_| IpiError::Clock)?;
        }
        Err(IpiError::Timeout {
            index: target,
            vector: self.vector,
        })
    }
}

/// How long each wait between looks at a processor's progress lasts.
const POLL_MICROS: u64 = 100;

/// How many times a processor's progress is re-read before waiting at all.
const SPINS_PER_POLL: u32 = 10_000;

/// Registers `handler` on a vector of its own, folding coalesced sends with
/// `merge`.
///
/// # Errors
///
/// [`IpiError::NotInstalled`] before [`install`], or
/// [`IpiError::Descriptors`] if every vector in the range is taken.
pub fn register(handler: Handler, merge: Merge) -> Result<Ipi, IpiError> {
    let state = state()?;
    let vector = descriptors::claim(FIRST, LAST, arrived)?;
    // `claim` answers with a vector out of the range it was given, so this is
    // the slot that vector belongs to. Asked rather than assumed, because the
    // same arithmetic runs on the interrupt path, where being wrong is a fault
    // rather than an error.
    let slot = slot(vector).ok_or(DescriptorError::NoVectorFree {
        first: FIRST,
        last: LAST,
    })?;
    // Stored after the vector is claimed and before anything can be sent on it.
    // An arrival in between finds no handler and is passed, which is right:
    // nothing has been sent, so nothing on that vector can be ours yet.
    state.handlers[slot].call_once(|| handler);
    Ok(Ipi {
        vector,
        slot,
        merge,
    })
}

/// Sets up the tables every interprocessor interrupt shares, and gives the
/// address space subsystem a way to reach the other processors.
///
/// Called once, on the boot processor, after the processor roster is taken —
/// the tables have one entry per processor — and before any processor is
/// started, so that each of them can answer a shootdown from the moment it
/// attaches.
///
/// # Errors
///
/// [`IpiError::AlreadyInstalled`] for a second call, [`IpiError::Cpu`] if the
/// roster has not been taken, or whatever registering the shootdown reported.
pub fn install() -> Result<(), IpiError> {
    let processors = cpu::roster()?.count();
    // The cell runs the closure for the one caller that fills it and for no
    // other, so whether it ran is exactly whether this call built the tables.
    let mut built = false;
    STATE.call_once(|| {
        built = true;
        State {
            processors,
            mailboxes: (0..CAPACITY * processors).map(|_| Mailbox::new()).collect(),
            handlers: [const { Once::new() }; CAPACITY],
        }
    });
    if !built {
        return Err(IpiError::AlreadyInstalled);
    }
    shootdown::install()
}

/// Logs what has been sent and what has arrived.
///
/// The last of these is the one worth having: an interrupt arriving on one of
/// our vectors that we did not send is the machine telling us something about
/// itself, and it is invisible unless it is counted.
pub fn describe(who: &str) {
    if paging::shootdown::in_hardware() {
        info!("{who}: ipi translation shootdown is the processor's own, not an interrupt");
    }
    let Ok(state) = state() else {
        info!("{who}: ipi is not installed");
        return;
    };
    let counts = state.counts();
    info!(
        "{who}: ipi {} sent, {} served over {} arrivals, {} folded together",
        counts.owed,
        counts.served,
        counts.arrivals,
        counts.served.saturating_sub(counts.arrivals),
    );
    if counts.foreign > 0 {
        info!(
            "{who}: ipi {} arrivals on our vectors were not ours",
            counts.foreign
        );
    }
}

/// Where every interprocessor interrupt arrives.
///
/// One entry point for all of them, because the only thing that differs is the
/// vector — and the vector is in the arrival, so it is also the index of
/// everything else that differs.
///
/// Everything that could say this arrival is not ours is asked before anything
/// is consumed, so that a delivery which turns out to belong to someone else
/// leaves no trace: no payload taken, no debt settled, and nothing
/// acknowledged. An interrupt that will be given to a guest is acknowledged as
/// part of giving it to one.
fn arrived(interrupt: &Interrupt) -> Disposition {
    let vector = interrupt.vector();
    let Ok(state) = state() else {
        return Disposition::Passed;
    };
    let Some(slot) = slot(vector) else {
        return Disposition::Passed;
    };
    let Some(handler) = state.handlers[slot].get() else {
        return Disposition::Passed;
    };

    // SAFETY: this processor attached before it unmasked interrupts, and an
    // interprocessor interrupt cannot be delivered to it before then, so its
    // `GS` base points at its own block.
    let index = unsafe { cpu::current() }.index();
    let mailbox = state.mailbox(slot, index);
    let Some(claim) = mailbox.claim() else {
        return Disposition::Passed;
    };

    handler(Request {
        payload: NonZeroU64::new(claim.payload()),
        vector,
    });
    mailbox.done(claim);
    let _ = apic::end_of_interrupt();
    Disposition::Consumed
}

/// Which of the shared tables' slots a vector belongs to, or `None` for one
/// outside the range this crate claims from.
fn slot(vector: Vector) -> Option<usize> {
    vector
        .number()
        .checked_sub(FIRST.number())
        .map(usize::from)
        .filter(|slot| *slot < CAPACITY)
}

/// Every attached processor but this one.
fn others() -> impl Iterator<Item = CpuIndex> {
    // SAFETY: as in `arrived`. Anything sending an interprocessor interrupt has
    // attached, since attaching is what put it in the list it is about to send
    // to.
    let here = unsafe { cpu::current() }.index();
    cpu::online()
        .map(cpu::Block::index)
        .filter(move |index| *index != here)
}

/// The tables every interprocessor interrupt shares.
#[derive(Debug)]
struct State {
    processors: usize,
    mailboxes: Box<[Mailbox]>,
    handlers: [Once<Handler>; CAPACITY],
}

impl State {
    /// One processor's state for one interrupt.
    ///
    /// Laid out interrupt by interrupt rather than processor by processor,
    /// because a broadcast walks every processor of one interrupt and nothing
    /// ever walks every interrupt of one processor.
    fn mailbox(&self, slot: usize, index: CpuIndex) -> &Mailbox {
        &self.mailboxes[slot * self.processors + index.get()]
    }

    /// What every mailbox has seen, added up.
    fn counts(&self) -> Counts {
        self.mailboxes
            .iter()
            .map(Mailbox::counts)
            .fold(Counts::default(), Counts::and)
    }
}

/// Built once, by the boot processor, before any processor is started.
static STATE: Once<State> = Once::new();

/// The shared tables.
fn state() -> Result<&'static State, IpiError> {
    STATE.get().ok_or(IpiError::NotInstalled)
}

/// Why an interprocessor interrupt could not be set up or sent.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum IpiError {
    /// Nothing has set the shared tables up yet.
    #[error("interprocessor interrupts have not been installed")]
    NotInstalled,
    /// They have, and the tables are sized by a roster that must not change
    /// underneath them.
    #[error("interprocessor interrupts have already been installed")]
    AlreadyInstalled,
    /// The processor has not attached, so there is nothing there to answer.
    #[error("{index} is not online")]
    NotOnline {
        /// The processor that was addressed.
        index: CpuIndex,
    },
    /// A processor did not run the handler in the time it was given.
    #[error("{index} did not answer {vector} in time")]
    Timeout {
        /// The processor that did not answer.
        index: CpuIndex,
        /// What it was sent.
        vector: Vector,
    },
    /// There is no timebase, so a wait cannot be bounded.
    #[error("no timebase is installed")]
    Clock,
    /// The interrupt controller refused something.
    #[error(transparent)]
    Apic(#[from] ApicError),
    /// The descriptor tables refused something.
    #[error(transparent)]
    Descriptors(#[from] DescriptorError),
    /// The processor roster refused something.
    #[error(transparent)]
    Cpu(#[from] CpuError),
    /// The address space subsystem already has a way to reach the other
    /// processors, which can only mean something other than this crate gave it
    /// one.
    #[error(transparent)]
    Shootdown(#[from] paging::shootdown::AlreadyInstalled),
}
