//! Interprocessor interrupts: how one processor asks another to do something.
//!
//! An interrupt is the only thing one processor can send another, and it
//! carries nothing but a vector. That is the whole of the mechanism and it is
//! less than it sounds, because a vector says nothing about who sent it — a
//! hypervisor that passes the platform through shares its vectors with devices
//! and with whatever was using the machine before. So this crate adds the two
//! things the hardware does not give.
//!
//! It records, per processor, what was sent to it, so that a handler can tell
//! an interrupt this hypervisor caused from one that merely arrived on the same
//! number. An arrival with nothing owed is answered [`Disposition::Passed`],
//! which hands it to whoever the hypervisor said should have unclaimed
//! interrupts — the seam that will one day give it to a guest.
//!
//! And it records what has been *served*, so that a sender can wait. Sending is
//! asynchronous and some things are not: telling every other processor to
//! forget a translation is worthless unless you know they have.
//!
//! # Vectors
//!
//! Taken from the top of the range the platform may assign, downwards from
//! [`LAST`]. High because on this architecture a vector's upper nibble is its
//! priority, and an interprocessor interrupt is usually something another
//! processor is blocked waiting on. Below the two the interrupt controller
//! keeps for itself.
//!
//! Which number a given interrupt ends up on does not matter to anyone: it is
//! only ever named by [`Ipi`], never written down.
//!
//! # Translation shootdown
//!
//! [`install`] sets up one interprocessor interrupt of this crate's own and
//! hands it to the address space subsystem, which has no way to reach another
//! processor and must not gain one — it is underneath everything here. That is
//! the whole of the dependency: a function pointer, going downwards.
//!
//! Its handler reloads the page table root and touches nothing else. In
//! particular it never asks for the address space lock, which is what keeps a
//! processor waiting for that lock from deadlocking against the processor
//! holding it and waiting for this acknowledgement.

#![no_std]

extern crate alloc;

mod pending;
mod shootdown;

use alloc::{boxed::Box, vec::Vec};
use core::{
    mem,
    ptr::{NonNull, null_mut},
    sync::atomic::{AtomicPtr, AtomicU64, Ordering},
};

use apic::{ApicError, Command, Delivery, Target};
use cpu::{CpuError, CpuIndex};
use descriptors::{DescriptorError, Disposition, Interrupt, Vector};
use log::info;
use spin::Once;
use thiserror::Error;

use crate::pending::Slot;

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
    count: u64,
    vector: Vector,
}

impl Request {
    /// How many sends were folded into this one arrival.
    ///
    /// At least one. More than one means several processors, or one processor
    /// several times, asked before this processor got round to answering —
    /// which the controller cannot represent, since it holds a single bit
    /// per vector. A handler that does one unit of work per request needs
    /// this; one that brings something up to date can ignore it.
    #[must_use]
    pub const fn count(&self) -> u64 {
        self.count
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

/// One kind of interprocessor interrupt, and the way to send it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ipi {
    vector: Vector,
    slot: usize,
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
    pub fn send(&self, target: CpuIndex) -> Result<(), IpiError> {
        let block = cpu::by_index(target).ok_or(IpiError::NotOnline { index: target })?;
        // The count goes up before the command goes out, so that a processor
        // which takes the interrupt immediately still finds what explains it.
        state()?.slot(self.slot, target).owe();
        SENT.fetch_add(1, Ordering::Relaxed);
        apic::local()?.send(Command::new(
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
    pub fn broadcast(&self) -> Result<usize, IpiError> {
        let mut sent = 0;
        for target in others() {
            self.send(target)?;
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
    pub fn send_and_wait(&self, target: CpuIndex, micros: u64) -> Result<(), IpiError> {
        let before = state()?.slot(self.slot, target).progress();
        self.send(target)?;
        self.await_progress(target, before, micros)
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
    pub fn broadcast_and_wait(&self, micros: u64) -> Result<usize, IpiError> {
        let state = state()?;
        let targets = others()
            .map(|target| (target, state.slot(self.slot, target).progress()))
            .collect::<Vec<_>>();
        for (target, _) in &targets {
            self.send(*target)?;
        }
        for (target, before) in &targets {
            self.await_progress(*target, *before, micros)?;
        }
        Ok(targets.len())
    }

    /// Waits for a processor to have got further than it had.
    fn await_progress(&self, target: CpuIndex, before: u64, micros: u64) -> Result<(), IpiError> {
        let slot = state()?.slot(self.slot, target);
        for _ in 0..micros.div_ceil(POLL_MICROS) {
            if slot.progress() > before {
                return Ok(());
            }
            // Spun rather than slept for most of the wait, because an answer is
            // normally a few hundred cycles away and sleeping would cost more
            // than it saves. The sleep is what bounds the whole thing.
            for _ in 0..SPINS_PER_POLL {
                if slot.progress() > before {
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

/// Registers `handler` on a vector of its own.
///
/// # Errors
///
/// [`IpiError::NotInstalled`] before [`install`], or
/// [`IpiError::Descriptors`] if every vector in the range is taken.
pub fn register(handler: Handler) -> Result<Ipi, IpiError> {
    let state = state()?;
    let vector = descriptors::claim(FIRST, LAST, arrived)?;
    let slot = usize::from(vector.number() - FIRST.number());
    // Stored after the vector is claimed and before anything can be sent on it,
    // which is the only ordering that leaves no window: a claimed vector with no
    // handler behind it would answer an arrival as somebody else's.
    state.handlers[slot].store(handler as *mut (), Ordering::Release);
    Ok(Ipi { vector, slot })
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
pub fn install() -> Result<Ipi, IpiError> {
    let processors = cpu::roster()?.count();
    if STATE.is_completed() {
        return Err(IpiError::AlreadyInstalled);
    }
    STATE.call_once(|| State {
        processors,
        slots: (0..CAPACITY * processors).map(|_| Slot::new()).collect(),
        handlers: [const { AtomicPtr::new(null_mut()) }; CAPACITY],
    });
    shootdown::install()
}

/// Logs what has been sent and what has arrived.
///
/// The last of these is the one worth having: an interrupt arriving on one of
/// our vectors that we did not send is the machine telling us something about
/// itself, and it is invisible unless it is counted.
pub fn describe(who: &str) {
    let sent = SENT.load(Ordering::Relaxed);
    let served = SERVED.load(Ordering::Relaxed);
    let arrivals = ARRIVALS.load(Ordering::Relaxed);
    info!(
        "{who}: ipi {sent} sent, {served} served over {arrivals} arrivals, {} folded together",
        served.saturating_sub(arrivals),
    );
    let foreign = FOREIGN.load(Ordering::Relaxed);
    if foreign > 0 {
        info!("{who}: ipi {foreign} arrivals on our vectors were not ours");
    }
}

/// Interrupts sent.
static SENT: AtomicU64 = AtomicU64::new(0);

/// Requests run, which is at least as many as there were arrivals.
static SERVED: AtomicU64 = AtomicU64::new(0);

/// Arrivals that were ours.
static ARRIVALS: AtomicU64 = AtomicU64::new(0);

/// Arrivals on one of our vectors that we had not sent.
static FOREIGN: AtomicU64 = AtomicU64::new(0);

/// Where every interprocessor interrupt arrives.
///
/// One entry point for all of them, because the only thing that differs is the
/// vector — and the vector is in the arrival, so it is also the index of
/// everything else that differs.
fn arrived(interrupt: &Interrupt) -> Disposition {
    let vector = interrupt.vector();
    let Ok(state) = state() else {
        return Disposition::Passed;
    };
    let slot = usize::from(vector.number() - FIRST.number());

    // SAFETY: this processor attached before it unmasked interrupts, and an
    // interprocessor interrupt cannot be delivered to it before then, so its
    // `GS` base points at its own block.
    let index = unsafe { cpu::current() }.index();
    let count = state.slot(slot, index).take();
    if count == 0 {
        // Not one of ours. What becomes of it is the hypervisor's decision, and
        // it is deliberately not acknowledged here: an interrupt that will be
        // given to a guest is acknowledged as part of giving it to one.
        FOREIGN.fetch_add(1, Ordering::Relaxed);
        return Disposition::Passed;
    }

    ARRIVALS.fetch_add(1, Ordering::Relaxed);
    SERVED.fetch_add(count, Ordering::Relaxed);
    if let Some(handler) = state.handler(slot) {
        handler(Request { count, vector });
    }
    state.slot(slot, index).done(count);
    let _ = apic::end_of_interrupt();
    Disposition::Consumed
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
    slots: Box<[Slot]>,
    handlers: [AtomicPtr<()>; CAPACITY],
}

impl State {
    /// One processor's state for one interrupt.
    ///
    /// Laid out interrupt by interrupt rather than processor by processor,
    /// because a broadcast walks every processor of one interrupt and nothing
    /// ever walks every interrupt of one processor.
    fn slot(&self, slot: usize, index: CpuIndex) -> &Slot {
        &self.slots[slot * self.processors + index.get()]
    }

    /// What was registered on this slot, if anything.
    fn handler(&self, slot: usize) -> Option<Handler> {
        // SAFETY: the only value `register` stores in a slot is a `Handler` cast
        // to a raw pointer, and null — filtered out first — is the only other
        // value it can hold. `transmute` checks the two are the same size, which
        // makes this exactly the inverse of the cast that stored it.
        NonNull::new(self.handlers[slot].load(Ordering::Acquire))
            .map(|handler| unsafe { mem::transmute::<NonNull<()>, Handler>(handler) })
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
}
