//! The global descriptor table, the task state segment, and the stacks the
//! processor switches to.
//!
//! Long mode barely uses segmentation, and what is here is only what the
//! architecture still insists on: a code segment to describe the mode the
//! processor runs in, a data segment for the registers that must hold
//! something, and a task state segment — which holds no task, and exists
//! entirely for the seven stack pointers in it.
//!
//! Those seven are the point. A gate descriptor may name one, and the processor
//! then switches to it before entering the handler, whatever the interrupted
//! stack was. That is the difference between reporting a stack overflow and
//! triple-faulting on the attempt to report it.
//!
//! Nothing here describes ring 3. Nothing this hypervisor runs leaves ring 0,
//! and a descriptor nobody can select is one fewer descriptor that can be
//! selected wrongly.
//!
//! # Two descriptors that are not ours
//!
//! A table is loaded by one instruction, and from that instruction on every
//! selector in every register means whatever the *new* table says at that
//! index. Two of those selectors are still live at that moment and the
//! processor can still act on them: the code selector, which a gate names and
//! `IRET` restores, and the stack selector, which `IRET` also restores from the
//! frame. Both of them at that point are firmware's, or the trampoline's.
//!
//! So both are kept meaningful across the change. The code segment this
//! hypervisor runs in is placed at *the index the live code selector already
//! uses*, so a gate or an `IRET` naming that index gets a valid 64-bit ring 0
//! code segment before the change and after it. The live stack descriptor is
//! copied to its own index unchanged, so an `IRET` that reloads it finds the
//! same segment it was written from. Everything this crate adds goes above
//! both.
//!
//! That is what makes the transition in [`crate::Tables::activate`] have no
//! window in it. The alternative — our own indices, chosen here — would mean an
//! interval in which the live interrupt descriptor table names selectors that
//! no longer describe what they did, and nothing masks the exceptions that
//! would be delivered through it.
//!
//! The consequence is that the selector *numbers* differ from processor to
//! processor, because firmware and the trampoline do not agree on where they
//! put their code segment. Nothing depends on them agreeing: each processor's
//! gates are written with its own numbers.
//!
//! # Building the table by hand
//!
//! The descriptor *values* come from [`Descriptor`], as everything about a
//! descriptor's layout should. The container does not: a table whose length is
//! decided by where the live code selector points cannot be a fixed-size type,
//! and the size it needs is known only on the processor being brought up.

use alloc::{boxed::Box, vec::Vec};
use core::mem::offset_of;

use log::warn;
use paging::{AddressSpace, Stack};
use x86_64::{
    PrivilegeLevel, VirtAddr,
    instructions::{
        segmentation::{CS, DS, ES, FS, GS, SS, Segment},
        tables::{self, load_tss},
    },
    registers::debug::{Dr7, Dr7Flags},
    structures::{
        DescriptorTablePointer,
        gdt::{Descriptor, SegmentSelector},
    },
};

use crate::{
    DescriptorError, InterruptStack,
    nesting::{self, Cpu},
};

/// Descriptors this crate adds: a code segment, a data segment, and the two
/// entries a task descriptor occupies.
const OWN_ENTRIES: usize = 4;

/// The 64-bit ring 0 code segment, as the value a table holds.
const CODE: u64 = match Descriptor::kernel_code_segment() {
    Descriptor::UserSegment(value) => value,
    Descriptor::SystemSegment(..) => panic!("a code segment is not a system descriptor"),
};

/// The writable ring 0 data segment, as the value a table holds.
const DATA: u64 = match Descriptor::kernel_data_segment() {
    Descriptor::UserSegment(value) => value,
    Descriptor::SystemSegment(..) => panic!("a data segment is not a system descriptor"),
};

/// Selectors into one processor's global descriptor table.
///
/// Numbers, and nothing more. They index the table of the processor that built
/// them — which is not the table another processor is running on, and which
/// puts its code segment wherever firmware happened to put its own. Nothing
/// here travels: they are reported so that a log says what a processor is
/// running on, and used by the code that built the table they belong to.
#[derive(Clone, Copy, Debug)]
pub struct Selectors {
    /// The 64-bit code segment this hypervisor executes in, at the index the
    /// code selector already had.
    pub code: SegmentSelector,
    /// The data segment every data segment register is pointed at.
    pub data: SegmentSelector,
    /// The task state segment holding the interrupt stacks. Its descriptor
    /// becomes busy when the task register is loaded, and a second load of a
    /// busy descriptor faults, so this is a one-shot value on one processor.
    pub task: SegmentSelector,
}

/// One processor's table, the block it describes, and the selectors into it —
/// built, and not yet loaded.
#[derive(Debug)]
pub(crate) struct Segments {
    /// The table itself. A `Vec` because its length depends on the table being
    /// replaced; never grown after this, so the addresses in it stay put.
    entries: Vec<u64>,
    /// What the task descriptor in `entries` points at.
    block: Box<Cpu>,
    /// The limit the processor is given, checked while it could still be
    /// refused.
    limit: u16,
    selectors: Selectors,
}

impl Segments {
    /// Builds a table for this processor around `stacks`, indexed by
    /// [`InterruptStack::slot`].
    ///
    /// Nothing is loaded and no register is touched: this is the part that can
    /// fail, and it happens while failing still means nothing has changed.
    ///
    /// # Errors
    ///
    /// [`DescriptorError::OutOfMemory`] if the table or the block cannot be
    /// allocated, [`DescriptorError::UnknownStackSegment`] if the live stack
    /// selector points outside the live table — which would mean the processor
    /// is already running on something inconsistent — or
    /// [`DescriptorError::TooManySegments`] if preserving the live indices
    /// would need a table larger than a table can be.
    pub(crate) fn build(stacks: &[Stack; InterruptStack::COUNT]) -> Result<Self, DescriptorError> {
        let block = Box::try_new(Cpu::new(stacks)).map_err(|_| DescriptorError::OutOfMemory)?;
        let code_index = CS::get_reg().index();
        let borrowed_stack = live_stack_descriptor()?;
        let first_own = code_index.max(borrowed_stack.map_or(0, |(index, _)| index)) + 1;
        let len = usize::from(first_own) + OWN_ENTRIES;
        let limit = u16::try_from(len * size_of::<u64>() - 1)
            .map_err(|_| DescriptorError::TooManySegments { entries: len })?;

        let mut entries = Vec::new();
        entries
            .try_reserve_exact(len)
            .map_err(|_| DescriptorError::OutOfMemory)?;
        entries.resize(len, 0);
        entries[usize::from(code_index)] = CODE;
        if let Some((index, descriptor)) = borrowed_stack {
            entries[usize::from(index)] = descriptor;
        }
        entries[usize::from(first_own)] = DATA;
        // SAFETY: the block is on the heap and is leaked before this descriptor
        // can be loaded, so what it points at outlives every processor that ever
        // reads it. Nothing else writes a task descriptor for this block.
        let task = unsafe { Descriptor::tss_segment_unchecked(block.task_state()) };
        place(&mut entries, first_own + 1, task);

        Ok(Self {
            entries,
            block,
            limit,
            selectors: Selectors {
                code: SegmentSelector::new(code_index, PrivilegeLevel::Ring0),
                data: SegmentSelector::new(first_own, PrivilegeLevel::Ring0),
                task: SegmentSelector::new(first_own + 1, PrivilegeLevel::Ring0),
            },
        })
    }

    /// The selectors into this table, needed to write gates before it is
    /// loaded.
    pub(crate) const fn selectors(&self) -> Selectors {
        self.selectors
    }

    /// Loads this table, points every segment register at it, and loads the
    /// task register.
    ///
    /// The table and the block become permanent here: the processor keeps a
    /// pointer to each for as long as it runs, so from this point on neither
    /// may move and neither may be dropped.
    ///
    /// All five data segment registers are reloaded even though 64-bit mode
    /// consults none of their descriptors, so that none is left naming an index
    /// this table leaves empty — and because a far transfer or a return to
    /// compatibility mode would consult them.
    ///
    /// # Safety
    ///
    /// - The calling processor must be the one this table was built on, and
    ///   must not already be running on a table this crate produced: the task
    ///   descriptor is marked busy by the task register load, and loading a
    ///   busy one faults.
    /// - Nothing may be relied upon across the call:
    ///   - the visible `FS` and `GS` selectors, which are replaced;
    ///   - the `FS` and `GS` bases, which loading those selectors sets to zero
    ///     — including `IA32_KERNEL_GS_BASE`'s counterpart in use, so whatever
    ///     needs a base must establish it afterwards;
    ///   - the descriptors behind any selector this table does not preserve,
    ///     which is every index except the live code and stack ones.
    /// - Interrupts must be masked, and the interrupt descriptor table live at
    ///   the moment of the call must have gates whose selectors this table
    ///   still describes and must name no stack in a task state segment — a
    ///   table loaded before the task register is has nowhere to switch to.
    pub(crate) unsafe fn activate(self) -> Selectors {
        let Self {
            entries,
            block,
            limit,
            selectors,
        } = self;
        let entries: &'static [u64] = Vec::leak(entries);
        let _: &'static Cpu = Box::leak(block);
        let pointer = DescriptorTablePointer {
            limit,
            base: VirtAddr::from_ptr(entries.as_ptr()),
        };

        // SAFETY: the table is `'static`, its limit was computed from its own
        // length, and it holds a 64-bit code segment at the index the live code
        // selector uses, a writable data segment at `data`, and an available task
        // descriptor at `task` naming a block that is now `'static` too. The
        // caller guarantees this processor is not already on a table of ours, so
        // that descriptor is not busy, and that nothing depends on the
        // segmentation being replaced.
        unsafe {
            tables::lgdt(&pointer);
            CS::set_reg(selectors.code);
            DS::set_reg(selectors.data);
            ES::set_reg(selectors.data);
            FS::set_reg(selectors.data);
            GS::set_reg(selectors.data);
            SS::set_reg(selectors.data);
            load_tss(selectors.task);
        }
        selectors
    }
}

/// Allocates this processor's interrupt stacks, indexed by
/// [`InterruptStack::slot`].
///
/// All or nothing: a failure part-way through gives back every stack that
/// succeeded before it, because a processor that cannot come up must not also
/// have consumed the frames and window address space of the ones that could.
///
/// # Errors
///
/// [`DescriptorError::Stack`] naming which of the seven could not be backed,
/// and why.
pub(crate) fn allocate_stacks(
    space: &mut AddressSpace,
) -> Result<[Stack; InterruptStack::COUNT], DescriptorError> {
    // A stack cannot be copied — that is what makes releasing one twice
    // unrepresentable — so the slots are filled one at a time and the array of
    // maybe-stacks is turned into an array of stacks once every one of them is
    // there.
    let mut taken: [Option<Stack>; InterruptStack::COUNT] = [const { None }; InterruptStack::COUNT];
    for wanted in InterruptStack::ALL {
        match space.allocate_stack(nesting::STACK_PAGES) {
            Ok(stack) => taken[usize::from(wanted.slot())] = Some(stack),
            Err(source) => {
                release_stacks(space, taken.iter_mut().rev().filter_map(Option::take));
                return Err(DescriptorError::Stack {
                    stack: wanted,
                    source,
                });
            }
        }
    }
    let [
        Some(fault),
        Some(nmi),
        Some(check),
        Some(debug),
        Some(page),
        Some(protection),
        Some(segment),
    ] = taken
    else {
        // Unreachable: the loop above filled the slot of every stack in
        // `InterruptStack::ALL`, and the assertion beside `InterruptStack::slot`
        // proves at compile time that those are the seven slots. Reported rather
        // than asserted because there is a caller to report it to.
        return Err(DescriptorError::StackTableIncomplete);
    };
    Ok([fault, nmi, check, debug, page, protection, segment])
}

/// Gives `stacks` back.
///
/// For the paths where the tables they were allocated for will not exist: a
/// build that failed after them, or one that failed later on. Callers pass them
/// in the reverse of the order they were taken.
///
/// Nothing is reported upwards: this is a rollback path, so a failure here is
/// already handling a failure and there is nowhere to propagate it to. Whatever
/// the address space could not prove detached stays allocated rather than being
/// handed out twice, which is what its own error says.
pub(crate) fn release_stacks(space: &mut AddressSpace, stacks: impl IntoIterator<Item = Stack>) {
    for stack in stacks {
        // SAFETY: none of these was ever put in a task state segment, let alone
        // switched to — a stack only becomes reachable when the table naming it
        // is loaded, and on these paths no table ever is.
        if let Err(error) = unsafe { space.release_stack(stack) } {
            warn!("descriptors: an interrupt stack could not be given back: {error}");
        }
    }
}

/// This processor's block, found through the table it is running on.
///
/// The task descriptor is the last thing [`Segments::build`] puts in a table,
/// so the last two entries of the live table are the descriptor naming this
/// processor's own block — and the table register is per processor, which is
/// what makes this answer the running processor's rather than anyone else's.
///
/// # Panics
///
/// Never. It is used on the interrupt entry path, so it reads a possible answer
/// out of the descriptor rather than checking one: a table that is not one of
/// ours would give an address that is not one either, and there is no state
/// left to report that from.
pub(crate) fn block() -> *mut Cpu {
    let live = tables::sgdt();
    let entries = (usize::from(live.limit) + 1) / size_of::<u64>();
    // SAFETY: the processor is running with this table loaded, so `limit + 1`
    // bytes at `base` are the table itself: readable, and holding at least the
    // two entries a task descriptor takes.
    let (low, high) = unsafe {
        let base = live.base.as_ptr::<u64>();
        (base.add(entries - 2).read(), base.add(entries - 1).read())
    };
    let base = (low >> 16) & 0x00FF_FFFF | (low >> 56) << 24 | (high & 0xFFFF_FFFF) << 32;
    (VirtAddr::new_truncate(base) - TSS_OFFSET).as_mut_ptr()
}

/// Where the task state segment sits inside the block the table describes.
const TSS_OFFSET: u64 = offset_of!(Cpu, tss) as u64;

/// Disarms every hardware breakpoint the processor might still be carrying.
///
/// Firmware is free to leave a breakpoint armed, and a breakpoint is the one
/// way a `#DB` can be raised inside the `#DB` path — the case
/// [`crate::nesting`] cannot rule out by reasoning about blocking. Clearing the
/// register removes it: with no address enabled and general-detect off, nothing
/// but single-stepping raises `#DB`, and nothing here ever sets that flag.
///
/// Reserved bits are preserved; only the enable and detect fields are cleared.
pub(crate) fn disarm_debug() {
    Dr7::write(Dr7Flags::empty().into());
}

/// Puts `descriptor` at `index`, and its second half after it if it has one.
fn place(entries: &mut [u64], index: u16, descriptor: Descriptor) {
    let index = usize::from(index);
    match descriptor {
        Descriptor::UserSegment(value) => entries[index] = value,
        Descriptor::SystemSegment(low, high) => {
            entries[index] = low;
            entries[index + 1] = high;
        }
    }
}

/// The live stack descriptor and the index it sits at, or `None` if the stack
/// selector is null.
///
/// `IRET` reloads `SS` from the frame it pops, so whatever the stack selector
/// is now has to keep describing the same segment in the table that replaces
/// the live one. A null selector needs nothing preserved: returning to ring 0
/// with one is allowed and describes no segment.
///
/// # Errors
///
/// [`DescriptorError::UnknownStackSegment`] if the selector points past the
/// live table's limit, which cannot happen on a processor that loaded it.
fn live_stack_descriptor() -> Result<Option<(u16, u64)>, DescriptorError> {
    let selector = SS::get_reg();
    let index = selector.index();
    if index == 0 {
        return Ok(None);
    }
    let live = tables::sgdt();
    let entries = (usize::from(live.limit) + 1) / size_of::<u64>();
    if usize::from(index) >= entries {
        return Err(DescriptorError::UnknownStackSegment {
            selector: selector.0,
        });
    }
    // SAFETY: the processor is running with this table loaded, and the index was
    // just checked to be inside it, so this reads one descriptor of the live
    // table.
    let descriptor = unsafe { live.base.as_ptr::<u64>().add(usize::from(index)).read() };
    Ok(Some((index, descriptor)))
}
