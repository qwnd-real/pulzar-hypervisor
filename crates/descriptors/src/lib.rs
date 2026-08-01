//! The descriptor tables the hypervisor runs on, and the path every interrupt
//! takes through them.
//!
//! Firmware's tables live in the lower half of the address space and stop
//! existing the moment it is dropped, so having tables of our own is a
//! precondition for that step rather than an improvement on it. But the
//! interesting part is not that they exist — it is what the interrupt
//! descriptor table is for in a hypervisor that does not own the machine.
//!
//! # Sharing vectors with something else
//!
//! Pulzar passes the platform through. It will arm an APIC timer and send
//! interprocessor interrupts, and those arrive on the same vectors as
//! everything else that was already using the machine. Two things follow. Every
//! vector needs an entry point, because any of them may turn out to carry
//! something the hypervisor caused. And an entry point cannot decide on its own
//! what it is looking at, because whether a given arrival was the hypervisor's
//! own doing is visible only to the subsystem that armed it.
//!
//! So [`dispatch`] splits the two apart: a subsystem [`register`]s a
//! [`Handler`] on its vector and answers with a [`Disposition`], and anything
//! not claimed reaches the [`Unclaimed`] callback the hypervisor supplied. That
//! callback is where giving an interrupt back to a guest will happen.
//!
//! # Shape
//!
//! - [`vector`] states what the architecture fixes about each of the 256
//!   vectors: error code, what may follow the handler, name, and which stack it
//!   switches to. Everything else derives its behaviour from it rather than
//!   repeating it.
//! - [`gdt`] builds the segments and the task state segment, whose only real
//!   content is the seven stacks the most dangerous exceptions run on.
//! - [`nesting`] is what keeps one of those stacks usable when the condition it
//!   belongs to arrives while it is already in use.
//! - [`idt`] gives every vector a gate and an entry point that knows its own
//!   number.
//! - [`dispatch`] is where they all arrive and where the hypervisor's own
//!   interrupts are separated from everyone else's.
//! - [`fatal`] is the one report that depends on nothing, for the arrivals
//!   nothing can return from.
//!
//! # Every processor builds its own
//!
//! All three tables are per processor. For two of them there is no choice: a
//! task state segment holds stacks, and a stack cannot be shared — two
//! processors taking a double fault at once would take it on the same one — and
//! the global descriptor table holds the descriptor for that task state
//! segment.
//!
//! The third could in principle be shared, since a gate names an entry point
//! and a selector rather than anything per processor. It is not, for two
//! reasons. The selector in a gate is *this* processor's code selector, and
//! this crate places its code segment at whatever index firmware or the
//! trampoline was already using, which differs between them. And a table built
//! once by whichever processor got there first is a table every other processor
//! waits on: if that one faults mid-construction, the rest wait forever. Four
//! kilobytes per processor buys the absence of both problems.
//!
//! What becomes of an unclaimed interrupt is not a per-processor fact at all:
//! it is what this hypervisor does. So it is said once, with [`adopt`], and
//! saying it is a precondition of any processor building tables — an interrupt
//! must never arrive to find no answer.
//!
//! # Building and switching are two steps
//!
//! [`Tables::build`] allocates and writes everything; [`Tables::activate`]
//! changes the processor's registers and cannot fail. Splitting them is what
//! makes a failure harmless — nothing is loaded, and every stack the attempt
//! took is given back — and it is what lets a processor hold the address space
//! lock for the first step without holding it across the second, where a fault
//! would strand it for every other processor.

#![feature(abi_x86_interrupt, allocator_api)]
#![no_std]

extern crate alloc;

mod dispatch;
mod fatal;
mod gdt;
mod idt;
mod nesting;
mod vector;

use alloc::{boxed::Box, vec::Vec};

use log::info;
use paging::{AddressSpace, PagingError, Stack};
use spin::Mutex;
use thiserror::Error;
use x86_64::{
    VirtAddr,
    instructions::{hlt, interrupts, tables},
};

pub use crate::{
    dispatch::{Disposition, Handler, Interrupt, Unclaimed, adopt, claim, register},
    gdt::Selectors,
    vector::{InterruptStack, Resumption, Vector},
};
use crate::{
    gdt::Segments,
    idt::{Idt, Stacks},
};

/// Everything one processor needs to run on tables of its own, built and not
/// yet loaded.
///
/// Holding one of these means the allocation has already happened: the stacks
/// are backed, the tables are written, and [`Tables::activate`] is a sequence
/// of register loads that cannot fail. Dropping one instead gives up the tables
/// but not the stacks behind them, which is why nothing does.
#[derive(Debug)]
#[must_use = "tables that are never activated hold seven stacks nothing will use"]
pub struct Tables {
    /// Kept only so that what was allocated can be given back if a later step
    /// of the same build fails.
    stacks: [Stack; InterruptStack::COUNT],
    segments: Segments,
    /// The table this processor ends up on.
    table: Box<Idt>,
    /// The table it is on while the segments are being replaced, whose gates
    /// switch no stacks because the task register does not yet name a task
    /// state segment that has any.
    transition: Box<Idt>,
}

impl Tables {
    /// Allocates this processor's interrupt stacks and writes its three tables.
    ///
    /// Nothing is loaded and no register is touched. A failure leaves the
    /// processor exactly as it was, with every frame and every window run the
    /// attempt took already given back.
    ///
    /// # Errors
    ///
    /// [`DescriptorError::Unadopted`] if nothing has said yet what becomes of
    /// an unclaimed interrupt, since building tables that would then be
    /// loaded makes a delivery possible that has no answer;
    /// [`DescriptorError::AlreadyInstalled`] if this processor is already
    /// running on tables from here; [`DescriptorError::Stack`] if one of the
    /// seven interrupt stacks cannot be backed;
    /// [`DescriptorError::OutOfMemory`] if a table cannot be allocated; or
    /// whatever [`Segments::build`] reports about the tables being
    /// replaced.
    pub fn build(space: &mut AddressSpace) -> Result<Self, DescriptorError> {
        if !dispatch::adopted() {
            return Err(DescriptorError::Unadopted);
        }
        if installed_here() {
            return Err(DescriptorError::AlreadyInstalled);
        }
        let stacks = gdt::allocate_stacks(space)?;
        match Self::assemble(&stacks) {
            Ok((segments, table, transition)) => Ok(Self {
                stacks,
                segments,
                table,
                transition,
            }),
            Err(error) => {
                gdt::release_stacks(space, stacks.into_iter().rev());
                Err(error)
            }
        }
    }

    /// Everything after the stacks, which is everything that can only fail by
    /// running out of memory.
    ///
    /// The stacks stay with the caller rather than moving in here, because a
    /// failure below has to give them back and a consumed value cannot be
    /// returned alongside the error that consumed it.
    fn assemble(
        stacks: &[Stack; InterruptStack::COUNT],
    ) -> Result<(Segments, Box<Idt>, Box<Idt>), DescriptorError> {
        let segments = Segments::build(stacks)?;
        let code = segments.selectors().code;
        let table = Idt::build(code, Stacks::Own)?;
        let transition = Idt::build(code, Stacks::Interrupted)?;
        Ok((segments, table, transition))
    }

    /// Gives everything back without loading any of it.
    ///
    /// The counterpart of [`Tables::build`] for a caller that decides, between
    /// the two steps, that this processor is not going to come up after all.
    /// Nothing else can return the stacks: they belong to the address space,
    /// and these tables are the only thing that knows which seven they are.
    pub fn release(self, space: &mut AddressSpace) {
        gdt::release_stacks(space, self.stacks.into_iter().rev());
    }

    /// Switches this processor onto these tables.
    ///
    /// The order inside is forced by what each step makes true, and every
    /// intermediate state is one the processor could take an exception in:
    ///
    /// 1. Interrupts are masked and stay masked until the end. That holds off
    ///    the maskable arrivals; it does not hold off a fault, a machine check
    ///    or a non-maskable interrupt, which is why the steps below leave no
    ///    interval where one of those would be delivered through something
    ///    inconsistent.
    /// 2. The debug registers are disarmed, so that nothing firmware left armed
    ///    can raise `#DB` inside the path that handles `#DB`.
    /// 3. The transition table is loaded. Its gates name the code selector that
    ///    is live *now*, and switch no stacks, so it is valid before the
    ///    segments change and valid after — and from this instant every vector
    ///    reaches this image rather than firmware's handlers.
    /// 4. The segments are replaced and the task register is loaded. The code
    ///    segment lands at the index the live code selector already used and
    ///    the live stack descriptor is carried over unchanged, so the table
    ///    loaded in step 3 keeps meaning what it meant, and so does an `IRET`
    ///    that reloads either register.
    /// 5. The real table is loaded, and its gates may name interrupt stacks
    ///    because the task register now names the segment holding them.
    /// 6. The interrupt flag goes back to whatever it was on entry.
    ///
    /// # Errors
    ///
    /// [`DescriptorError::AlreadyInstalled`] if this processor is already
    /// running on tables from here — loading a second set would leave the first
    /// one's stacks and segments allocated with nothing able to reach them — or
    /// [`DescriptorError::OutOfMemory`] if the record of which processors have
    /// installed cannot be extended.
    pub fn activate(self) -> Result<Descriptors, DescriptorError> {
        let Self {
            segments,
            table,
            transition,
            stacks: _,
        } = self;
        let mut installed = INSTALLED.lock();
        if installed.contains(&live_table()) {
            return Err(DescriptorError::AlreadyInstalled);
        }
        // Room for this processor's entry taken before anything is loaded, so
        // that the record cannot fail to be written once it is true.
        installed
            .try_reserve(1)
            .map_err(|_| DescriptorError::OutOfMemory)?;

        let table: &'static Idt = Box::leak(table);
        let selectors = interrupts::without_interrupts(|| {
            gdt::disarm_debug();
            // SAFETY: `transition` lives until the end of this function, which is
            // past the load of `table` that replaces it. Its gates name the code
            // selector that is live at this instant and switch no stacks, so they
            // need nothing of the task register. `segments` was built on this
            // processor, which the record above proves is not already running on
            // tables of ours, so its task descriptor is not busy; it preserves
            // the live code and stack descriptors, so the gates just loaded stay
            // valid across it; and nothing in this image depends on the
            // segmentation or on an `FS`/`GS` base at this point in bring-up.
            // `table` is `'static` and its gates name the code selector the
            // segments just made ours and the stacks the task register now
            // reaches.
            unsafe {
                idt::load(&transition);
                let selectors = segments.activate();
                idt::load(table);
                selectors
            }
        });

        installed.push(VirtAddr::from_ptr(core::ptr::from_ref(table)));
        Ok(Descriptors {
            selectors,
            table: VirtAddr::from_ptr(core::ptr::from_ref(table)),
        })
    }
}

/// The descriptor tables one processor is running on.
///
/// Not copyable and not clonable: it says that a particular processor made a
/// one-way change to itself, and a second one of these would be a claim nothing
/// could make true.
#[derive(Debug)]
pub struct Descriptors {
    selectors: Selectors,
    /// Where this processor's own interrupt descriptor table is, which is what
    /// makes this token that processor's rather than any other's.
    table: VirtAddr,
}

impl Descriptors {
    /// The selectors into this processor's global descriptor table.
    ///
    /// Numbers for a log to print. See [`Selectors`] for why they are not
    /// anything to hand to another processor.
    #[must_use]
    pub const fn selectors(&self) -> Selectors {
        self.selectors
    }

    /// Unmasks interrupts on the calling processor.
    ///
    /// The one place in this workspace that turns delivery on, because there is
    /// a condition to state before it happens and this is where stating it
    /// belongs.
    ///
    /// # Errors
    ///
    /// [`DescriptorError::AnotherProcessor`] if the caller is not the processor
    /// these tables were activated on, which would mean unmasking on a
    /// processor whose own state nothing here has established.
    ///
    /// # Safety
    ///
    /// Nothing that can deliver an interrupt to this processor may be
    /// programmed with a vector below [`Vector::FIRST_EXTERNAL`]. In
    /// particular the legacy 8259 controllers, which deliver onto vectors 8
    /// to 15 as firmware leaves them at reset, must be masked or absent. An
    /// external interrupt on an exception's vector arrives without the
    /// error code that vector's gate is written for, and the handler would
    /// return to an address eight bytes from the right one.
    pub unsafe fn unmask(&self) -> Result<(), DescriptorError> {
        if live_table() != self.table {
            return Err(DescriptorError::AnotherProcessor);
        }
        interrupts::enable();
        Ok(())
    }

    /// Logs what was installed.
    pub fn describe(&self, who: &str) {
        info!(
            "{who}: gdt loaded, cs {:#06x}, ds {:#06x}, tss {:#06x}",
            self.selectors.code.0, self.selectors.data.0, self.selectors.task.0
        );
        info!(
            "{who}: idt loaded at {:#x}, {} vectors, {} on stacks of their own",
            self.table,
            Vector::COUNT,
            InterruptStack::COUNT
        );
    }
}

/// Stops this processor for good.
///
/// The loop is the whole of the guarantee. Masking holds off the ordinary
/// interrupts, and nothing else: a system management interrupt, a non-maskable
/// interrupt, a machine check or an `INIT` can still reach the processor, and
/// the first two of those return to the halt they interrupted — where the loop
/// masks again and halts again. `INIT` and reset abandon this context rather
/// than returning to it, and nothing here can or should prevent that.
///
/// Called from a handler that must not return, it never comes back to the
/// interrupted code at all: the loop is where that processor stays.
///
/// It lives here because this is the crate that owns what happens when the
/// processor cannot continue: an interrupt no one will ever claim is the case
/// this crate must answer on its own, and having one way to stop rather than
/// one per caller is what keeps that answer the same everywhere.
pub fn halt() -> ! {
    loop {
        interrupts::disable();
        hlt();
    }
}

/// Stops this processor until an interrupt reaches it, unless `ready` says
/// there is already something to do.
///
/// What a processor with nothing to run calls instead of spinning. Unlike
/// [`halt`] it comes back — the interrupt that woke the processor is what it
/// was waiting for, and its handler has already run by the time this returns.
///
/// # The mask is the whole of the point
///
/// A processor that tested a condition, found nothing, and then halted would
/// have a window between the two in which the interrupt announcing the
/// condition can arrive, be handled, and leave the processor to halt anyway —
/// waiting for a second announcement that nothing is going to send. So the test
/// happens with interrupts masked, and the wait begins in the same instruction
/// that unmasks them: an interrupt that arrives after the test cannot be taken
/// before the wait, so it wakes the processor rather than being lost to it.
///
/// Returns with interrupts unmasked, which is the only state a caller could
/// usefully have called it in.
pub fn wait_until(ready: impl FnOnce() -> bool) {
    interrupts::disable();
    if ready() {
        interrupts::enable();
        return;
    }
    interrupts::enable_and_hlt();
}

/// The interrupt descriptor table of every processor that has activated one.
///
/// A processor cannot be asked which it is this early — it has no block of its
/// own yet, and its interrupt controller has not been enabled — but it can be
/// asked what table it is running on, and that register is per processor. So
/// this is how "has this processor already installed" is answered: the live
/// table is one of these exactly when the answer is yes.
static INSTALLED: Mutex<Vec<VirtAddr>> = Mutex::new(Vec::new());

/// Whether the calling processor is already running on tables from here.
fn installed_here() -> bool {
    INSTALLED.lock().contains(&live_table())
}

/// Where the calling processor's interrupt descriptor table is.
fn live_table() -> VirtAddr {
    tables::sidt().base
}

/// Why the descriptor tables could not be set up or added to.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum DescriptorError {
    /// One of the seven interrupt stacks could not be allocated or mapped.
    #[error("{stack} could not be backed: {source}")]
    Stack {
        /// Which of the seven it was.
        stack: InterruptStack,
        /// What the address space said.
        #[source]
        source: PagingError,
    },
    /// A table or the block behind it could not be allocated.
    #[error("there is not enough memory left for a processor's descriptor tables")]
    OutOfMemory,
    /// The stack selector this processor is running on names a descriptor
    /// outside the table it is running on, which is a machine that should not
    /// have got this far.
    #[error("the stack selector {selector:#06x} is not in the table it indexes")]
    UnknownStackSegment {
        /// The selector in question.
        selector: u16,
    },
    /// Keeping the live code and stack selectors meaningful would need a table
    /// with more entries than one can have.
    #[error("a global descriptor table of {entries} entries is larger than one can be")]
    TooManySegments {
        /// How many entries it would have taken.
        entries: usize,
    },
    /// This processor is already running on tables from here, and a second set
    /// would abandon the first with nothing able to reach it.
    #[error("this processor already installed its descriptor tables")]
    AlreadyInstalled,
    /// Every interrupt stack was allocated and the table they go in still has a
    /// slot with nothing in it, which the list of stacks says cannot happen.
    #[error("the interrupt stack table was left with an empty slot")]
    StackTableIncomplete,
    /// The tables belong to a different processor from the one asking.
    #[error("these descriptor tables belong to another processor")]
    AnotherProcessor,
    /// Nothing has said what becomes of an interrupt no handler claims, so no
    /// table of gates may be built yet.
    #[error("nothing has adopted the unclaimed interrupts yet")]
    Unadopted,
    /// Something already said what becomes of an unclaimed interrupt, and it is
    /// one answer for the whole machine.
    #[error("the unclaimed interrupts have already been adopted")]
    AlreadyAdopted,
    /// Something already claimed the vector, and a vector holds one handler.
    #[error("{vector} already has a handler")]
    VectorTaken {
        /// The vector in question.
        vector: Vector,
    },
    /// The architecture gives no defined way back from this vector, so nothing
    /// may claim it: consuming it would mean resuming a machine whose state the
    /// processor has already declared lost.
    #[error("{vector} cannot be returned from, so nothing may claim it")]
    NotReturnable {
        /// The vector in question.
        vector: Vector,
    },
    /// This hypervisor stops on this vector rather than establishing what it
    /// would take to return from it, so nothing may claim it.
    #[error("{vector} is one this hypervisor stops on, so nothing may claim it")]
    FailStop {
        /// The vector in question.
        vector: Vector,
    },
    /// A range to allocate a vector from ended below where it started.
    #[error("no vector is both at or above {first} and at or below {last}")]
    RangeReversed {
        /// Low end as given.
        first: Vector,
        /// High end as given.
        last: Vector,
    },
    /// A range to allocate a vector from reached into the architecture's own
    /// vectors, which no interrupt controller may be told to deliver.
    #[error("{vector} is an exception's vector, which nothing external may use")]
    NotExternal {
        /// The first vector of the range, which is the one below the line.
        vector: Vector,
    },
    /// Every vector in the range asked for already has a handler.
    #[error("no vector between {first} and {last} is free")]
    NoVectorFree {
        /// Low end of the range searched.
        first: Vector,
        /// High end of the range searched.
        last: Vector,
    },
}
