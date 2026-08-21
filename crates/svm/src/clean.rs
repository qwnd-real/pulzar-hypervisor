//! Which parts of a guest's control block the processor may take from its own
//! cache instead of reading again.
//!
//! Entering a guest means the processor reads a kibibyte of control state, and
//! most of it has not changed since the last entry. So the processor is allowed
//! to keep some of that state in hardware between exits and reuse it, and this
//! field is how software says what is still good.
//!
//! The asymmetry in the rules is what makes them safe to work with. A bit that
//! is **set** says "you may reuse your cached copy" — and it is only ever a
//! hint, since any given processor may ignore it and re-read the block anyway.
//! A bit that is **clear** says "read this group again", and that is *always*
//! honoured. Clearing a bit can therefore only cost performance, while setting
//! one wrongly runs a guest on state that no longer exists. When in doubt,
//! clear.
//!
//! # The rule that is easy to get wrong
//!
//! Software must clear a bit every time it changes the state that bit covers.
//! That is the ordinary case and it is mechanical. The trap is the other rule:
//! the *whole* field must be zeroed when a guest runs for the first time, when
//! it runs on a different physical processor than it last ran on, or when its
//! control block has been moved to a different physical page. Missing any of
//! those is undefined behaviour rather than a slow guest.
//!
//! The reason is how the processor identifies a cached block: by the block's
//! **physical address**, not by its address-space identifier. Two guests that
//! reuse a page, or one guest that migrates between processors, are exactly the
//! cases where an address matches but the contents behind it do not.
//!
//! # What has no clean bit at all
//!
//! Some state is never cached, so no bit here covers it and it is always read
//! or written afresh: the translation-flush control, the interrupt shadow, all
//! of the status fields the processor reports an exit through, the event
//! injection field, and the stack pointer, instruction pointer, flags and
//! accumulator in the state save area. Nothing needs to be cleared for those.
//!
//! # On processors that do not cache
//!
//! Support is reported by the processor, and one without it neither caches
//! anything nor reads this field. Writing it is therefore harmless everywhere,
//! which is why nothing here is conditional on the feature.

use bitflags::bitflags;

bitflags! {
    /// Which groups of a guest's control block are unchanged since the
    /// processor last read them.
    ///
    /// Each flag covers a named group of fields, and the grouping is the whole
    /// value of the type: the fields a bit covers are not always adjacent in
    /// the block, and a hypervisor that edits one field and clears the wrong
    /// bit gets a guest running on stale state with no diagnostic. Each flag
    /// below names exactly what it covers.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct CleanBits: u32 {
        /// Every intercept vector, the timestamp offset, and the spin filter's
        /// count and threshold.
        const INTERCEPTS = 1 << 0;
        /// The addresses of both permission bitmaps: the one for ports and the
        /// one for model-specific registers.
        const PERMISSION_MAPS = 1 << 1;
        /// The address-space identifier translations are tagged with.
        const ASID = 1 << 2;
        /// The whole virtual interrupt control register, which is one quadword
        /// and so one group: the guest's task priority, whether an interrupt is
        /// pending for it, that interrupt's priority and vector, whether the task
        /// priority is ignored, the guest's own global interrupt flag and the bit
        /// that gives it one, its pending and masked non-maskable-interrupt state
        /// and the bit that virtualizes that masking, whether interrupt masking
        /// is virtualized at all — and the two bits that turn hardware-driven
        /// interrupt delivery on and choose which of its two faces it drives.
        ///
        /// Those last two are the ones worth naming, because the mistake this
        /// architecture warns about is reading them as the acceleration's own
        /// group. They are here and not in [`CleanBits::AVIC`], so software that
        /// turns the acceleration on or off and clears only that bit has told the
        /// processor it may reuse the word it just edited.
        const INTERRUPT = 1 << 3;
        /// Nested paging: the root of the second set of page tables and the
        /// guest's page attribute table.
        const NESTED_PAGING = 1 << 4;
        /// The guest's control registers zero, three and four, and its extended
        /// feature register.
        const CONTROL_REGISTERS = 1 << 5;
        /// The guest's debug registers six and seven.
        const DEBUG_REGISTERS = 1 << 6;
        /// The base and limit of the guest's global and interrupt descriptor
        /// tables.
        const DESCRIPTOR_TABLES = 1 << 7;
        /// The four data-ish segment registers — code, data, stack and extra —
        /// with their selectors, bases, limits and attributes, and the guest's
        /// privilege level.
        const SEGMENTS = 1 << 8;
        /// The guest's control register two, which holds the address of the
        /// last page fault it took.
        const FAULT_ADDRESS = 1 << 9;
        /// The guest's branch-record state: its debug control register, and the
        /// addresses the last branch and last interrupt came from and went to.
        const LAST_BRANCH = 1 << 10;
        /// Where the guest's hardware-driven interrupt controller is, and only
        /// that: the register base, the backing page, the physical and logical
        /// table pointers, and the largest index of the physical table, which
        /// the architecture keeps in the low bits of that table's own pointer.
        ///
        /// Nothing here says whether the acceleration is *on* — the two enable
        /// bits are [`CleanBits::INTERRUPT`]'s — so the two groups are not
        /// interchangeable, and a transition that moves a pointer and an enable
        /// bit owes both.
        const AVIC = 1 << 11;
        /// The guest's control-flow enforcement state: its supervisor control
        /// register, shadow stack pointer, and interrupt shadow stack table
        /// address.
        const CONTROL_FLOW = 1 << 12;
    }
}

impl CleanBits {
    /// Every group this architecture defines a bit for.
    ///
    /// Not the same as "everything in the block is unchanged" — the fields
    /// listed in this module's documentation as never cached are not covered by
    /// any bit and are always re-read regardless.
    pub const ALL_CACHED: Self = Self::all();

    /// Nothing may be reused: every group is read from the block.
    ///
    /// This is the correct value for a control block the processor cannot have
    /// a valid cached copy of — one that has never been run, one that has moved
    /// to a different physical page, or one being run on a processor other than
    /// the last one it ran on. In each of those the processor's cache may hold
    /// something under this block's physical address that belongs to a
    /// different guest entirely.
    #[must_use]
    pub const fn nothing_cached() -> Self {
        Self::empty()
    }

    /// Marks the groups in `changed` as needing to be read again.
    ///
    /// The operation a hypervisor performs after editing a control block, named
    /// so that it reads as what it is rather than as a bitwise complement.
    #[must_use]
    pub const fn soil(self, changed: Self) -> Self {
        Self::from_bits_truncate(self.bits() & !changed.bits())
    }

    /// Whether the processor may reuse its cached copy of every group in
    /// `groups`.
    #[must_use]
    pub const fn is_clean(self, groups: Self) -> bool {
        self.contains(groups)
    }
}

const _: () = assert!(
    size_of::<CleanBits>() == size_of::<u32>(),
    "the clean field is a doubleword of the control area",
);
