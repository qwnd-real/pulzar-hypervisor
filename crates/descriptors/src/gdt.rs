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

use paging::{AddressSpace, PagingError};
use spin::Once;
use x86_64::{
    VirtAddr,
    instructions::{
        segmentation::{CS, DS, ES, FS, GS, SS, Segment},
        tables::load_tss,
    },
    structures::{
        gdt::{Descriptor, GlobalDescriptorTable, SegmentSelector},
        tss::TaskStateSegment,
    },
};

use crate::InterruptStack;

/// Pages behind each interrupt stack: sixteen kilobytes, with an unmapped guard
/// page below and above.
///
/// These stacks exist to be switched to when the interrupted one cannot be
/// trusted, so they are sized for a handler that formats one line and stops —
/// generously, because the cost of getting it wrong is a fault with no stack
/// left to report it from.
const STACK_PAGES: u64 = 4;

/// The task state segment. The processor keeps a descriptor pointing at it for
/// as long as the task register is loaded, so it must never move.
static TSS: Once<TaskStateSegment> = Once::new();

/// The global descriptor table, together with the selectors into it.
static GDT: Once<Table> = Once::new();

/// Selectors into the global descriptor table.
///
/// A selector means nothing without the table it indexes, so they travel
/// together and are only ever produced by the code that built that table.
#[derive(Clone, Copy, Debug)]
pub struct Selectors {
    /// The 64-bit code segment this hypervisor executes in.
    pub code: SegmentSelector,
    /// The data segment every data segment register is pointed at.
    pub data: SegmentSelector,
    /// The task state segment holding the interrupt stacks.
    pub task: SegmentSelector,
}

/// Allocates the interrupt stacks, builds the tables, and switches the
/// processor onto them.
///
/// # Errors
///
/// [`PagingError`] if the chunk or the mapping window cannot back the seven
/// interrupt stacks.
pub(crate) fn install(space: &mut AddressSpace) -> Result<Selectors, PagingError> {
    let mut stacks = [VirtAddr::zero(); InterruptStack::COUNT];
    for stack in InterruptStack::ALL {
        stacks[usize::from(stack.slot())] = space.allocate_stack(STACK_PAGES)?.top();
    }
    let tss = TSS.call_once(|| task_state(&stacks));
    let table = GDT.call_once(|| Table::new(tss));

    // SAFETY: nothing in this image depends on the segmentation firmware set up
    // — it uses no segment base and makes no privilege transition — and the
    // table being replaced lives in the half of the address space that is about
    // to stop existing anyway.
    unsafe { table.activate() };
    Ok(table.selectors)
}

/// The task state segment, whose only load-bearing part is the table of stacks.
///
/// The privilege stack table stays zero: nothing here runs outside ring 0, so
/// there is no privilege transition that would need a stack, and a pointer
/// nothing can reach is better left absent than filled in with something
/// plausible.
fn task_state(stacks: &[VirtAddr; InterruptStack::COUNT]) -> TaskStateSegment {
    let mut tss = TaskStateSegment::new();
    tss.interrupt_stack_table = *stacks;
    tss
}

/// The table and the selectors into it, which are produced together and are
/// meaningless apart.
#[derive(Debug)]
struct Table {
    gdt: GlobalDescriptorTable,
    selectors: Selectors,
}

impl Table {
    /// Builds a table with one code segment, one data segment, and a descriptor
    /// for `tss`.
    fn new(tss: &'static TaskStateSegment) -> Self {
        let mut gdt = GlobalDescriptorTable::new();
        let selectors = Selectors {
            code: gdt.append(Descriptor::kernel_code_segment()),
            data: gdt.append(Descriptor::kernel_data_segment()),
            task: gdt.append(Descriptor::tss_segment(tss)),
        };
        Self { gdt, selectors }
    }

    /// Loads this table, points every segment register at it, and loads the
    /// task register.
    ///
    /// All five data segment registers are reloaded even though 64-bit mode
    /// ignores most of what they say, so that none is left holding a selector
    /// into a table that is no longer loaded.
    ///
    /// # Safety
    ///
    /// Nothing may depend on the segmentation currently in effect, including
    /// any `FS` or `GS` base.
    unsafe fn activate(&'static self) {
        self.gdt.load();
        // SAFETY: the table was just loaded and outlives the processor's
        // reference to it, being a `'static`. `code` selects a 64-bit code
        // segment in it, `data` a writable data segment, and `task` the task
        // state segment's descriptor; the caller guarantees that the selectors
        // being replaced are not relied upon.
        unsafe {
            CS::set_reg(self.selectors.code);
            DS::set_reg(self.selectors.data);
            ES::set_reg(self.selectors.data);
            FS::set_reg(self.selectors.data);
            GS::set_reg(self.selectors.data);
            SS::set_reg(self.selectors.data);
            load_tss(self.selectors.task);
        }
    }
}
