//! The block of state each processor reaches through its own `GS` base.
//!
//! Every processor needs to answer "which of us am I" without being told, and
//! needs to answer it on the interrupt path, where there is no handle to thread.
//! Reading the local APIC's identifier register would work — it is what the
//! roster is keyed on — but it costs a model-specific-register read at best and
//! an uncached bus cycle at worst, every time.
//!
//! The architecture already has a per-processor pointer for this. `GS` has a
//! 64-bit base that long mode otherwise leaves unused, and an access through it
//! is one instruction needing no register to hold an address. So each processor
//! points its `GS` base at a block of its own whose first field is the block's
//! own address, and [`current`] is a single load.
//!
//! # What lives here and what does not
//!
//! Only identity. Subsystems that keep per-processor data of their own keep it
//! in their own arrays, indexed by [`CpuIndex`] — which costs one load to
//! obtain, and which keeps this crate from having to know what those subsystems
//! are. A block full of slots for other people's state would be the same data
//! under a worse name.
//!
//! Whether a processor is up is not a field either. It is whether its block has
//! been published, which is the last thing attaching does: one fact, recorded
//! once, rather than a flag that could disagree with the table.
//!
//! # Why not `KERNEL_GS_BASE`
//!
//! Because nothing swaps yet. `swapgs` exists to exchange this base with a
//! shadow across a boundary where the other side owns `GS` — a ring 3 entry, or
//! a guest exit. Nothing in pulzar crosses such a boundary, so there is one base
//! and it is the live one. Entering a guest is what will introduce the shadow,
//! and it will introduce it because it needs it.
//!
//! # Why reading it is unsafe
//!
//! Because the base is zero until a processor attaches, and virtual address zero
//! is not mapped — the firmware half of the address space is gone by the time
//! any of this runs. So a read before attaching is not a null pointer to be
//! checked for, it is a page fault. [`current`] therefore states the
//! precondition instead of pretending to detect it, and [`attached`] is the safe
//! way to ask, at the cost of a register read.

use core::arch::asm;

use x86_64::{VirtAddr, registers::model_specific::GsBase};

use crate::{ApicId, CpuIndex};

/// One processor's own state.
///
/// `repr(C)` with `self_ptr` first, because [`current`] reads that field by
/// offset and the offset it reads is zero. Cache-line aligned so that two
/// processors touching their own blocks never contend for one line.
#[derive(Debug)]
#[repr(C, align(64))]
pub struct Block {
    /// This block's own address, so that a processor can turn its `GS` base into
    /// a reference with one load rather than by reading a model-specific
    /// register.
    self_ptr: *const Self,
    index: CpuIndex,
    apic_id: ApicId,
}

// SAFETY: a block is reached from other processors — that is what the table
// behind `by_index` is for — and every field is written once, before the block
// is published, and only read afterwards. What withholds the automatic
// implementations is `self_ptr` being a raw pointer; it only ever holds this
// block's own address, and the block is leaked, so it cannot dangle.
unsafe impl Send for Block {}
// SAFETY: as above.
unsafe impl Sync for Block {}

impl Block {
    /// Which processor this is.
    #[must_use]
    pub const fn index(&self) -> CpuIndex {
        self.index
    }

    /// The local APIC identifier interrupts to this processor are addressed to.
    #[must_use]
    pub const fn apic_id(&self) -> ApicId {
        self.apic_id
    }

    /// The block for a processor with this position and identifier.
    ///
    /// The self-pointer stays null: the block has no address until it has been
    /// leaked, so [`Block::activate`] is what fills it in.
    pub(crate) const fn new(index: CpuIndex, apic_id: ApicId) -> Self {
        Self {
            self_ptr: core::ptr::null(),
            index,
            apic_id,
        }
    }

    /// Fills in the self-pointer and points this processor's `GS` base at the
    /// block.
    ///
    /// Taking a `&'static mut` is what makes this safe to offer at all: it is
    /// the type-level form of "leaked", and a base that cannot outlive what it
    /// points at is the whole of what [`current`] needs. Which processor's block
    /// it is, and that no descriptor table is loaded afterwards to zero the base
    /// again, are correctness matters the one caller settles.
    pub(crate) fn activate(block: &'static mut Self) -> &'static Self {
        block.self_ptr = core::ptr::from_ref(block);
        let block: &'static Self = block;
        GsBase::write(VirtAddr::from_ptr(core::ptr::from_ref(block)));
        block
    }
}

/// The block of the processor this runs on.
///
/// One load. This is the reason the block exists at all: it is what an interrupt
/// handler calls to find out which processor it is on.
///
/// # Safety
///
/// This processor must have attached. Before that its `GS` base is zero, and
/// virtual address zero is not mapped, so the read is a page fault rather than a
/// null pointer that could be checked for. Every processor attaches immediately
/// after installing its descriptor tables and before unmasking interrupts, which
/// is what makes this hold on every path an interrupt can arrive through.
#[must_use]
pub unsafe fn current() -> &'static Block {
    let mut block: *const Block;
    // SAFETY: the caller guarantees the `GS` base points at their leaked block,
    // whose first field is its own address. `activate` is the only thing that
    // ever writes that field and it writes nothing else there.
    unsafe {
        asm!("mov {}, gs:[0]", out(reg) block, options(nostack, readonly, preserves_flags));
        &*block
    }
}

/// Whether this processor has attached, and so whether [`current`] may be
/// called.
///
/// Reads the base itself rather than following it, which is what makes this the
/// safe question and the slow one. For a check at bring-up or in a log line,
/// not for the interrupt path.
#[must_use]
pub fn attached() -> bool {
    GsBase::read() != VirtAddr::zero()
}
