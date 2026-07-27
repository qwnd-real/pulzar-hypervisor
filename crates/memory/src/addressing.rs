//! How a guest turns an address into a physical one.
//!
//! Five registers and six segment bases decide everything about a guest's
//! translation: whether it translates at all, through how many levels, how wide
//! its addresses are, and what a segment adds to one. They are gathered here
//! rather than passed around individually because they are only meaningful
//! together — paging enabled with the extension bit clear is a different
//! machine from paging enabled with it set, and reading one without the other
//! is how a walker ends up confidently walking the wrong shape of table.
//!
//! Nothing here reads them from anywhere. A running guest keeps them in its
//! control block and firmware's are in the snapshot the loader captured, and
//! both arrive through [`Addressing::from_save`] because both are a state-save
//! area. That is the whole of this crate's relationship with the guest: it is
//! handed the state, never asked to go and find it.

use svm::SaveArea;

/// Bit of `CR0` that turns the guest's own page tables on.
const PG: u64 = 1 << 31;
/// Bit of `CR4` that widens a page table entry to eight bytes and adds a level.
const PAE: u64 = 1 << 5;
/// Bit of `CR4` that lets a page directory entry describe four megabytes.
const PSE: u64 = 1 << 4;
/// Bit of `CR4` that adds a fifth level of page table.
const LA57: u64 = 1 << 12;
/// Bit of `EFER` that says the processor is really in long mode, as opposed to
/// having merely been told to enter it.
const LMA: u64 = 1 << 10;

/// How a guest translates its addresses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Addressing {
    cr0: u64,
    cr3: u64,
    cr4: u64,
    efer: u64,
    bases: [u64; 6],
    long: bool,
    default_size: bool,
}

impl Addressing {
    /// How the guest a state-save area describes translates.
    #[must_use]
    pub fn from_save(save: &SaveArea) -> Self {
        Self {
            cr0: save.cr0,
            cr3: save.cr3,
            cr4: save.cr4,
            efer: save.efer,
            bases: [
                save.es.base,
                save.cs.base,
                save.ss.base,
                save.ds.base,
                save.fs.base,
                save.gs.base,
            ],
            long: save.cs.attributes.long(),
            default_size: save.cs.attributes.default_size(),
        }
    }

    /// Through what, if anything, this guest's addresses are translated.
    #[must_use]
    pub const fn mode(&self) -> Mode {
        if self.cr0 & PG == 0 {
            Mode::Unpaged
        } else if self.cr4 & PAE == 0 {
            Mode::Legacy
        } else if self.efer & LMA == 0 {
            Mode::Pae
        } else if self.cr4 & LA57 == 0 {
            Mode::Long
        } else {
            Mode::FiveLevel
        }
    }

    /// The root of the guest's own page tables, as a guest physical address.
    ///
    /// The low bits are the caller's to interpret, because what they hold
    /// depends on the mode: a process-context identifier under one control
    /// register bit, cache hints without it, and under the page-address
    /// extension a thirty-two-byte-aligned pointer rather than a page-aligned
    /// one.
    #[must_use]
    pub const fn root(&self) -> u64 {
        self.cr3
    }

    /// Whether the guest is executing 64-bit code, which is the code segment's
    /// long bit and not the mode alone — a long-mode guest running
    /// compatibility code has this clear.
    #[must_use]
    pub const fn long_mode(&self) -> bool {
        self.long
    }

    /// Whether the code segment's default operand and address size is the wider
    /// of the two it can be. Meaningless together with
    /// [`Addressing::long_mode`], which the architecture forbids.
    #[must_use]
    pub const fn default_size(&self) -> bool {
        self.default_size
    }

    /// Whether a page directory entry of this guest may describe four
    /// megabytes.
    ///
    /// Without the size extension the bit that would say so is reserved, so a
    /// guest that set it anyway gets what the hardware gives it: a walk that
    /// carries on to the table below.
    #[must_use]
    pub const fn large_pages(&self) -> bool {
        self.cr4 & PSE != 0
    }

    /// What this segment adds to an offset within it.
    ///
    /// In 64-bit mode four of the six are forced to zero and the other two are
    /// not, and a state-save area already holds whichever of those is true — so
    /// nothing here has a case for the mode.
    #[must_use]
    pub const fn base(&self, segment: Segment) -> u64 {
        self.bases[segment as usize]
    }
}

/// Through what, if anything, a guest's addresses are translated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Not translated at all: what the guest computes is what it reaches. Real
    /// mode and protected mode without paging both look like this, and a
    /// processor being brought up spends its first instructions here.
    Unpaged,
    /// Two levels of four-byte entries, describing four kilobytes or, with the
    /// size extension, four megabytes.
    Legacy,
    /// Four pointers, then two levels of eight-byte entries: four kilobytes or
    /// two megabytes, with thirty-two-bit addresses reaching physical memory
    /// above four gigabytes.
    Pae,
    /// Four levels of eight-byte entries: four kilobytes, two megabytes or one
    /// gigabyte.
    Long,
    /// Five levels. Nothing here walks them, and nothing here runs under them
    /// either — the hypervisor refuses to boot on a machine already using them.
    FiveLevel,
}

/// Which segment an address is an offset within.
///
/// The order is the architecture's own encoding of them, which is what lets the
/// base be an index rather than a match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Segment {
    /// The extra segment, which the writing string instructions always use.
    Es,
    /// The code segment.
    Cs,
    /// The stack segment.
    Ss,
    /// The data segment, which an ordinary memory operand uses unless a prefix
    /// says otherwise.
    Ds,
    /// The first of the two segments whose base is fully 64-bit.
    Fs,
    /// The second, and the one the swap-base instruction exchanges.
    Gs,
}

const _: () = assert!(
    Segment::Es as usize == 0 && Segment::Gs as usize == 5,
    "the segments must index the array of bases in the architecture's own order",
);
