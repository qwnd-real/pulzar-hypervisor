//! Bringing the processors that are not running yet up to running hypervisor
//! code.
//!
//! A processor that has not been started is held in reset and answers nothing.
//! Getting it out is a fixed sequence the architecture defines: an `INIT`
//! command puts it in a state where it is waiting to be told where to begin,
//! and a startup command tells it — as a page number, which it begins executing
//! at in real mode. Everything after that is the trampoline's, and everything
//! after *that* is the caller's.
//!
//! # One at a time
//!
//! Deliberately, and not for simplicity. A processor part-way through this has
//! no descriptor tables, no interrupt handlers, and interrupts masked; if it
//! were already something the rest of the machine counted as present, a
//! translation shootdown would wait for an acknowledgement it cannot give. So
//! each processor is taken all the way to attached before the next is started,
//! and "present" means attached rather than started.
//!
//! It also means one stack and one set of parameters are in flight at a time,
//! which is what lets the trampoline's parameter block be rewritten between
//! processors instead of one existing per processor. That is a real constraint
//! and not merely a convenience: the block may only be rewritten once the
//! processor before it has said it has finished reading it, so a processor that
//! begins and then stops somewhere in the middle stops the whole sequence.
//! There is nothing else to do with it — the alternative is to overwrite the
//! stack pointer and the entry point of a processor that may still be about to
//! read them, and two processors running on one stack is a worse machine than
//! one with fewer processors.
//!
//! A processor that never began at all is a different case and is only logged:
//! the architecture's own startup sequence defines the two commands and the
//! delays after them, and a processor that has not executed its first
//! instruction by the end of that is one the sequence has finished with.
//!
//! # Stacks are never handed on
//!
//! Every processor that is sent a startup command keeps the stack allocated for
//! it, whether or not it ever ran. Reclaiming one means proving that no
//! processor can still be executing on it, and a command the controller
//! accepted is not something that can be un-accepted. Sixty-four kilobytes per
//! processor that failed to start is the price of never having two of them on
//! one stack.
//!
//! # Never twice
//!
//! A startup command sent to a processor that is already running does not do
//! nothing: `INIT` resets it. So a processor is only ever a target if it is not
//! already one of the machine's, and firmware describing the same processor
//! twice — which its tables have two ways of doing — cannot turn into two
//! attempts. That is also what makes starting the rest of the machine later, or
//! twice, safe rather than catastrophic.
//!
//! # The mapping that has to exist while this runs
//!
//! The trampoline turns paging on while executing at a low physical address,
//! and the instruction after that is fetched through the tables it just loaded.
//! So the page has to be mapped at its own address for as long as any processor
//! is in the middle of this, and must not be afterwards — the firmware half of
//! the address space is gone by then, and putting one page back into it
//! permanently would be the only thing down there. It is mapped here and
//! unmapped here, and the unmapping is what tells every processor that just
//! used it to forget it.

use alloc::vec::Vec;
use core::sync::atomic::Ordering;

use cpu::ApicId;
use log::{info, warn};
use paging::{CacheType, Protection, Stack};
use x86_64::{PhysAddr, VirtAddr};

use crate::{
    ApicError, LocalApic, PAGE,
    icr::{Command, Delivery, Target},
    trampoline::Trampoline,
};

/// Pages behind each started processor's stack: sixty-four kilobytes, with an
/// unmapped guard page below and above.
///
/// The same order of magnitude as the stack the loader gives the boot
/// processor. A processor that has been started runs the same code every other
/// one does, so sizing its stack for less would only mean finding out later
/// which of them is the one that overflows.
const STACK_PAGES: u64 = 16;

/// How long a processor is left in reset before it is told where to begin.
///
/// The architecture's own figure for the `INIT` sequence.
const INIT_MICROS: u64 = 10_000;

/// How long a processor is given to answer a startup command before a second
/// one is sent.
///
/// Also the architecture's, and the reason two are sent at all: the first can
/// be missed by a processor that has not finished coming out of reset, and a
/// second is harmless to one that has already begun.
const STARTUP_MICROS: u64 = 200;

/// How long a processor that has begun executing is given to reach the point of
/// being attached.
///
/// Generous, because what happens in between is the whole of a processor's
/// bring-up including allocations that may contend for a lock, and because the
/// only cost of waiting too long is a slower boot on a machine that is already
/// broken.
const ATTACH_MICROS: u64 = 1_000_000;

/// One past the highest address a startup command can send a processor to.
const ONE_MIB: u64 = 1 << 20;

/// What starting the other processors came to.
///
/// Two counts over two populations, which is worth saying because they can
/// disagree in both directions: `startable` is how many distinct processors
/// firmware described as ones that may be started, and `online` is how many are
/// attached — which includes any that were attached before this ran and
/// excludes any firmware described as unstartable. Equal numbers mean
/// everything described came up; unequal ones mean the log has the detail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Started {
    /// How many processors are now attached, this one included.
    pub online: usize,
    /// How many distinct processors the machine's own tables described as
    /// startable, this one included.
    pub startable: usize,
}

/// Starts every processor firmware described as startable, and waits for each
/// to reach `main`.
///
/// `trampoline` is a reserved page below one megabyte, which the loader asked
/// firmware for. `main` is what each processor runs once it is executing 64-bit
/// code on a stack of its own: it never returns, and installing descriptor
/// tables is the first thing it should do, because until it does the processor
/// has no way to report anything that goes wrong.
///
/// A processor that never answered a startup command is logged and skipped: one
/// processor failing to start is not a reason to refuse to run a machine, and
/// the count returned is what actually happened rather than what was attempted.
/// A processor that answered and then stopped part-way is not skipped, because
/// starting anything else would mean overwriting parameters it may still read.
///
/// # Errors
///
/// [`ApicError::TrampolineUnreachable`] if the page is not a frame-aligned one
/// below one megabyte or is outside the direct map;
/// [`ApicError::NotInstalled`], [`ApicError::NotEnabled`] or
/// [`ApicError::NoApic`] if this processor's own controller cannot be reached;
/// [`ApicError::Cpu`] if the processor roster cannot be read;
/// [`ApicError::Paging`] if the page cannot be mapped or unmapped, which
/// includes some processor failing to acknowledge dropping the mapping;
/// [`ApicError::StartupUnresolved`] if a processor began starting and stopped
/// before it had read its parameters, which leaves the trampoline page mapped
/// because that processor may still be executing from it; or whatever placing
/// the trampoline reported.
pub fn start(trampoline: PhysAddr, main: fn() -> !) -> Result<Started, ApicError> {
    let base = trampoline.as_u64();
    if base >= ONE_MIB || !base.is_multiple_of(PAGE) {
        return Err(ApicError::TrampolineUnreachable { phys: base });
    }
    let local = crate::local()?;
    let here = local.id();
    let startable = startable()?;

    let root = paging::with(|space| space.root().start_address())?;
    // The whole page, not just its first byte: the trampoline is written over
    // all of it below.
    let at = paging::with(|space| {
        space
            .direct_map()
            .bytes_ptr::<u8>(trampoline, paging::as_usize(PAGE))
    })?
    .map_err(|_| ApicError::TrampolineUnreachable { phys: base })?;

    // Mapped at its own address, because the instruction after the one that
    // turns paging on is fetched from here.
    paging::with(|space| {
        // SAFETY: the page is firmware-reserved for exactly this and nothing
        // else maps it; the direct map's alias of it is read-write and
        // no-execute, so this adds an executable path to a page whose contents
        // are about to be written and nothing else.
        unsafe {
            space.map_region(
                VirtAddr::new(base),
                trampoline,
                PAGE,
                Protection::ReadExecute,
                CacheType::WriteBack,
            )
        }
    })??;

    // SAFETY: `at` is the direct map's address of the whole reserved page, which
    // is writable there and used by nothing else, and `trampoline` is its
    // physical address — checked above to be frame-aligned and below one
    // megabyte.
    let placed = unsafe { Trampoline::place(at.as_ptr(), trampoline, root, entry_address(main)) };
    let started = placed.and_then(|trampoline| start_each(local, &trampoline, &startable, here));

    // Unmapped whatever happened above: leaving one executable page of the low
    // half behind would outlast the reason it existed. This is also what makes
    // every processor that just ran from it forget the translation.
    //
    // Unless a processor is unaccounted for. One that began and never finished
    // reading its parameters may yet execute the instruction that turns paging
    // on, and that instruction is fetched from this page — so removing the
    // mapping is the one thing that could still make its situation worse.
    let unmapped = match &started {
        Err(ApicError::StartupUnresolved { apic_id }) => {
            warn!(
                "apic: leaving the trampoline mapped at {base:#x}; {apic_id} began starting and \
                 may still be executing from it"
            );
            Ok(())
        }
        _ => paging::with(|space| {
            // SAFETY: every processor is either past the point of using this page
            // — each is waited for before the next is started, and one that began
            // and did not finish stops the sequence above — or never reached it.
            unsafe { space.unmap_region(VirtAddr::new(base), PAGE) }
        })?,
    };

    // Both outcomes matter and only one can be returned. A failed cleanup is the
    // one that outlives the call — an executable page of the low half, or a
    // processor still translating one — so it is the one reported, and the other
    // is logged where it happened.
    match (started, unmapped) {
        (Ok(()), Ok(())) => Ok(Started {
            online: cpu::online_count(),
            startable: startable.len(),
        }),
        (Ok(()), Err(cleanup)) => Err(cleanup.into()),
        (Err(error), Ok(())) => Err(error),
        (Err(error), Err(cleanup)) => {
            warn!("apic: starting the other processors also failed: {error}");
            Err(cleanup.into())
        }
    }
}

/// Every processor firmware described as startable, named once each and in
/// order.
///
/// Deduplicated because a machine may describe one processor twice: the tables
/// have two ways of naming a processor, one of them older and narrower, and
/// firmware is free to use both. Two entries for one processor would otherwise
/// become two startup sequences, and the second would reset a processor that
/// the first had just brought up.
fn startable() -> Result<Vec<ApicId>, ApicError> {
    let mut startable: Vec<ApicId> = cpu::roster()?
        .entries()
        .iter()
        .filter(|entry| entry.startable())
        .map(cpu::Entry::apic_id)
        .collect();
    startable.sort_unstable();
    startable.dedup();
    Ok(startable)
}

/// Takes each processor from reset to attached, one before the next.
///
/// Whoever is running this is skipped, and so is anyone already attached: a
/// processor that is one of the machine's is one an `INIT` would reset rather
/// than start.
///
/// # Errors
///
/// [`ApicError::StartupUnresolved`] if a processor began and stopped before it
/// had taken its parameters out of the trampoline, which is the one failure
/// that stops the rest: the next processor's parameters go in the same place.
fn start_each(
    local: LocalApic,
    trampoline: &Trampoline,
    startable: &[ApicId],
    here: ApicId,
) -> Result<(), ApicError> {
    for target in startable.iter().copied() {
        if target == here || attached(target) {
            continue;
        }
        let stack = match allocate_stack() {
            Ok(stack) => stack,
            // Every processor after this one would ask for the same thing and
            // be refused the same way, so there is nothing to be gained by
            // asking again.
            Err(error) => {
                warn!("apic: no stack for {target} or anything after it: {error}");
                return Ok(());
            }
        };
        // Whatever comes of this, the stack stays this processor's. Handing it
        // to the next one would need proof that a command the controller
        // accepted can no longer produce a running processor, and there is no
        // such proof.
        match bring_up(local, trampoline, target, &stack) {
            Ok(()) => info!("apic: {target} started"),
            // It began and did not arrive, so where it is now is unknown and it
            // may still have the parameter block to read. Nothing else can be
            // started, because the next processor's parameters go in the same
            // place. A success is not this case: it ends in the target
            // publishing itself, which is a long way past its last read.
            Err(error) if trampoline.began() => {
                warn!(
                    "apic: {target} stopped after stage {} of starting: {error}",
                    trampoline.reached()
                );
                return Err(ApicError::StartupUnresolved { apic_id: target });
            }
            Err(error) => warn!("apic: {target} did not start: {error}"),
        }
    }
    Ok(())
}

/// A stack for one processor to run on, out of the machine's address space.
fn allocate_stack() -> Result<Stack, ApicError> {
    paging::with(|space| space.allocate_stack(STACK_PAGES))?.map_err(ApicError::from)
}

/// Takes one processor from reset to attached.
///
/// # Errors
///
/// Whatever sending it a command reported, [`ApicError::Clock`] if there is no
/// timebase to wait on, [`ApicError::NoStartupResponse`] if it never executed
/// the first instruction of the trampoline, or [`ApicError::AttachTimeout`] if
/// it began and never became one of the machine's.
fn bring_up(
    local: LocalApic,
    trampoline: &Trampoline,
    target: ApicId,
    stack: &Stack,
) -> Result<(), ApicError> {
    trampoline.prepare(stack.top().as_u64());

    // Everything the processor will read has to be in memory before the command
    // that lets it read anything. The command itself adds the barrier the
    // architecture requires of the interface it goes out through.
    core::sync::atomic::fence(Ordering::Release);

    local.send(Command::new(Delivery::Init, Target::One(target)))?;
    sleep(INIT_MICROS)?;

    for _ in 0..STARTUP_COMMANDS {
        local.send(Command::new(
            Delivery::Startup(trampoline.vector()),
            Target::One(target),
        ))?;
        sleep(STARTUP_MICROS)?;
        if trampoline.began() {
            break;
        }
    }
    if !trampoline.began() {
        return Err(ApicError::NoStartupResponse { apic_id: target });
    }
    wait_for(target)
}

/// Where a function the trampoline jumps to lives.
///
/// Through `usize` rather than straight to `u64`, because a function pointer is
/// an address and not a number, and the width it is an address in is the one
/// that says so.
fn entry_address(main: fn() -> !) -> u64 {
    main as usize as u64
}

/// How many startup commands a processor is sent before it is given up on.
///
/// Two, which is what the architecture's own sequence specifies: the first can
/// be missed by a processor still coming out of reset, and a second reaching
/// one that has already begun is ignored.
const STARTUP_COMMANDS: u32 = 2;

/// Waits for `target` to become one of the machine's.
///
/// Looked at once more after the last wait than before it, so that a processor
/// which publishes itself during that wait is seen to have arrived inside its
/// budget rather than reported as having missed it.
///
/// # Errors
///
/// [`ApicError::AttachTimeout`] if it never did, or [`ApicError::Clock`] if
/// there is no timebase to wait on.
fn wait_for(target: ApicId) -> Result<(), ApicError> {
    for _ in 0..ATTACH_MICROS.div_ceil(POLL_MICROS) {
        if attached(target) {
            return Ok(());
        }
        sleep(POLL_MICROS)?;
    }
    if attached(target) {
        return Ok(());
    }
    Err(ApicError::AttachTimeout { apic_id: target })
}

/// Whether a processor is already one of the machine's.
///
/// Attached rather than started, which is the distinction the whole of this
/// module turns on: a processor is only something other processors may send
/// interrupts to and wait on once it has published a block of its own.
fn attached(target: ApicId) -> bool {
    cpu::online().any(|block| block.apic_id() == target)
}

/// How long each wait for a processor to arrive lasts before looking again.
const POLL_MICROS: u64 = 1_000;

/// Waits, on the machine's timebase.
///
/// The delays in this sequence are the architecture's own figures and are not
/// approximations to be spun out: a startup command sent too soon after an
/// `INIT` is one the processor is not yet in a state to answer.
fn sleep(micros: u64) -> Result<(), ApicError> {
    clock::sleep_micros(micros).map_err(|_| ApicError::Clock)
}
