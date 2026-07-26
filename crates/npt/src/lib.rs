//! The second set of page tables: guest physical addresses to system physical
//! addresses.
//!
//! Under nested paging a guest walks its own page tables freely and the
//! addresses it believes are physical are translated again by these. That is
//! what makes a guest fast — no shadow tables, no intercept on every change the
//! guest makes to its own mapping — and it is what lets a guest be given a view
//! of physical memory that is not the machine's.
//!
//! Here that view is almost the machine's own. Pulzar passes the hardware
//! through, so a guest physical address maps to the identical system physical
//! address, and these tables exist to say two things: *yes, identically* for
//! everything the machine has, and *no* for the one region that is the
//! hypervisor's.
//!
//! # Built as it is asked for
//!
//! Nothing is described until a guest touches it. A guest physical address with
//! no translation raises `#VMEXIT(NPF)`, the handler calls [`Npt::fault`], and
//! that describes the region containing the address. A machine with a terabyte
//! of memory therefore costs a handful of tables rather than one per gigabyte
//! it never touched, and a device aperture far above the last byte of RAM is
//! described the moment it is used and not before.
//!
//! The region a fault describes is the largest one whose whole extent can be
//! answered the same way:
//!
//! - the 1 GiB containing the address, where the processor has 1 GiB pages and
//!   the hypervisor's own memory is nowhere inside it;
//! - otherwise the 2 MiB containing it, on the same condition;
//! - otherwise the address is inside the hypervisor's own memory, and the 2 MiB
//!   containing it is shadowed (below).
//!
//! Those three are exhaustive because the reserved chunk is 2 MiB aligned and a
//! whole number of 2 MiB regions, which makes *overlapping* the chunk and
//! *lying inside* it the same thing at 2 MiB granularity. [`Npt::create`]
//! checks that rather than assuming it, because it is the one property the case
//! analysis rests on.
//!
//! # What the guest sees instead of the hypervisor
//!
//! Every guest physical page of the reserved chunk — pulzar's image, its heap,
//! its stacks, its descriptor tables, its control blocks and these tables
//! themselves — is mapped to one shared frame of zeroes, read-only. A guest
//! reading there sees zeroes rather than either the hypervisor or a fault,
//! which is what keeps a guest that merely walks physical memory out of
//! trouble.
//!
//! A guest *writing* there is a different matter, and is deliberately
//! unfinished: the write faults, [`Npt::fault`] reports
//! [`Resolution::Shadowed`], and with nothing to emulate the instruction with,
//! the guest re-executes it and faults again. That is a live-lock rather than a
//! corruption or a crash, and it is where instruction emulation will be hooked
//! in. It is stated here so that nobody diagnoses it as a bug in the tables.
//!
//! # Why the memory type is write-back everywhere
//!
//! Under nested paging the effective memory type of a guest access is the
//! guest's type combined with the nested one. A nested type of write-back
//! leaves the guest's choice in force for every type but write-combining, which
//! becomes write-combining-with-snooping — uncacheable either way, and coherent
//! as well. So write-back at this level is not a claim that the memory is
//! cacheable; it is the identity element that lets a guest mark its own device
//! apertures uncacheable and be obeyed.
//!
//! # No translation is ever invalidated
//!
//! Filling only ever turns a not-present entry present, and the architecture
//! requires no invalidation for that — the walker detects a constraint being
//! removed on its own. Flushing a guest's tagged translations is only necessary
//! when a hypervisor reduces permissions, clears present bits, or changes what
//! an address translates to, none of which happens here. That is why nothing in
//! this crate touches `TLB_CONTROL`.

#![no_std]

mod walk;

use log::info;
use paging::{DirectMap, Frames, chunk};
use processor::Features;
use svm::exit::NestedPageFault;
use thiserror::Error;
use x86_64::{PhysAddr, structures::paging::PageTableFlags};

use crate::walk::{Level, PARENT};

/// One guest's nested page tables.
///
/// Holds no allocator and no lock. Frames are handed in per call by whoever
/// owns the chunk's allocator, and exclusion is the caller's — which keeps this
/// a description of a translation rather than a second owner of the machine's
/// memory.
#[derive(Debug)]
pub struct Npt {
    root: PhysAddr,
    zero: PhysAddr,
    owned: Owned,
    window: DirectMap,
    large: bool,
}

impl Npt {
    /// Builds an empty set of tables for a guest whose physical memory is the
    /// machine's, less the hypervisor's own.
    ///
    /// Empty is the correct starting point rather than a stub: with no entry
    /// present, the guest's first access to any address faults, and
    /// [`Npt::fault`] is what turns that into a translation.
    ///
    /// # Errors
    ///
    /// [`NptError::OutOfFrames`] if the chunk cannot spare the root table and
    /// the shared page of zeroes, [`NptError::Unreachable`] if the window does
    /// not reach a frame it just handed out, or [`NptError::ChunkGeometry`] if
    /// the reserved chunk is not 2 MiB aligned and a whole number of 2 MiB
    /// regions — the property that makes the fill rule's cases exhaustive.
    pub fn create(frames: &mut Frames, window: DirectMap) -> Result<Self, NptError> {
        let base = frames.chunk_base().as_u64();
        let large = Level::Directory.span();
        if !base.is_multiple_of(large) || !chunk::CHUNK_SIZE.is_multiple_of(large) {
            return Err(NptError::ChunkGeometry {
                base,
                size: chunk::CHUNK_SIZE,
            });
        }
        let root = frame(frames, window)?;
        let zero = frame(frames, window)?;
        Ok(Self {
            root,
            zero,
            owned: Owned {
                base,
                end: base + chunk::CHUNK_SIZE,
            },
            window,
            large: processor::features().contains(Features::GIB_PAGES),
        })
    }

    /// The value a guest's control block names these tables by, in its nested
    /// table root field.
    #[must_use]
    pub const fn root(&self) -> PhysAddr {
        self.root
    }

    /// Describes the region containing a guest physical address that had no
    /// translation.
    ///
    /// `cause` is the first exit-information field of the nested page fault,
    /// decoded. Only its direction is read: what a fault means here does not
    /// otherwise depend on why the guest was touching the address.
    ///
    /// # Errors
    ///
    /// [`NptError::OutOfFrames`] if the chunk cannot spare a table,
    /// [`NptError::Unreachable`] if the window does not reach one, or
    /// [`NptError::LargePage`] if a large page already covers the address,
    /// which means something described this region at a granularity the
    /// fill rule never produces.
    pub fn fault(
        &mut self,
        frames: &mut Frames,
        gpa: PhysAddr,
        cause: NestedPageFault,
    ) -> Result<Resolution, NptError> {
        let ours = self.owned.contains(gpa);
        if !walk::translated(self.window, self.root, gpa)? {
            if ours {
                self.shadow(frames, gpa)?;
            } else {
                self.identity(frames, gpa)?;
            }
        }
        Ok(if ours && cause.write() {
            Resolution::Shadowed
        } else {
            Resolution::Mapped
        })
    }

    /// Logs the shape of the translation, which is the whole of what a guest's
    /// view of memory is.
    pub fn describe(&self, who: &str) {
        info!(
            "{who}: npt rooted at {:#x}, identity mapped in {} pages",
            self.root,
            if self.large { "1 GiB" } else { "2 MiB" },
        );
        info!(
            "{who}: npt shadows physical {:#x}..{:#x} onto {:#x}, read only",
            self.owned.base, self.owned.end, self.zero,
        );
    }

    /// Maps the largest region containing `gpa` that holds none of the
    /// hypervisor's own memory, to itself.
    ///
    /// The 2 MiB fallback needs no check of its own. This is only reached for
    /// an address outside the chunk, and a 2 MiB region that overlapped the
    /// chunk would lie entirely inside it — so the region containing an
    /// address outside the chunk cannot overlap it.
    fn identity(&mut self, frames: &mut Frames, gpa: PhysAddr) -> Result<(), NptError> {
        let huge = Level::Pointer;
        if self.large {
            let base = gpa.align_down(huge.span());
            if !self.owned.overlaps(base, huge.span()) {
                return self.itself(frames, base, huge);
            }
        }
        self.itself(
            frames,
            gpa.align_down(Level::Directory.span()),
            Level::Directory,
        )
    }

    /// Points every 4 KiB of the 2 MiB region containing `gpa` at the shared
    /// page of zeroes.
    ///
    /// The whole 2 MiB at once, because by the geometry [`Npt::create`]
    /// checked, all of it is the hypervisor's: describing it page by page
    /// as each one is touched would cost a fault per page and arrive at the
    /// same table.
    fn shadow(&mut self, frames: &mut Frames, gpa: PhysAddr) -> Result<(), NptError> {
        let base = gpa.align_down(Level::Directory.span());
        let mut table = walk::descend(self.window, self.root, base, Level::Page, frames)?;
        // SAFETY: `descend` returns a table of these nested tables, reached
        // through the window, and `&mut self` is the only handle to them.
        let table = unsafe { table.as_mut() };
        for entry in table.iter_mut() {
            entry.set_addr(self.zero, SHADOW);
        }
        Ok(())
    }

    /// Describes the `level`-sized region beginning at `base` as itself.
    ///
    /// `base` is both the guest physical address being described and the system
    /// physical address it is described as, which is the whole of what makes
    /// this an identity map — and the reason the two are one argument
    /// rather than two that could be passed the wrong way round.
    fn itself(
        &mut self,
        frames: &mut Frames,
        base: PhysAddr,
        level: Level,
    ) -> Result<(), NptError> {
        walk::map(self.window, self.root, base, level, base, LEAF, frames)
    }
}

/// What became of a guest physical address that faulted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resolution {
    /// A translation exists now, and the access will succeed when the guest
    /// retries it.
    Mapped,
    /// The guest tried to write memory the hypervisor owns. Reads there see
    /// zeroes; a write cannot be satisfied, so resuming the guest re-executes
    /// it and faults again. Stepping over it needs instruction emulation,
    /// which does not exist yet.
    Shadowed,
}

/// Why a guest physical address could not be described.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum NptError {
    /// The chunk has no frame left for a table.
    #[error("the reserved chunk has no frame left for a nested page table")]
    OutOfFrames,
    /// The window onto physical memory does not reach a table.
    #[error("nested page table at physical {phys:#x} is outside the window onto physical memory")]
    Unreachable {
        /// The table that could not be reached.
        phys: u64,
    },
    /// A large page already covers the address, and nothing here splits one.
    #[error("guest physical {gpa:#x} is already covered by a large page")]
    LargePage {
        /// The address in question.
        gpa: u64,
    },
    /// The reserved chunk is not aligned or sized as the shadow requires.
    #[error(
        "the reserved chunk at {base:#x}, {size:#x} bytes, is not a whole number of 2 MiB regions"
    )]
    ChunkGeometry {
        /// Where the chunk begins.
        base: u64,
        /// How long it is.
        size: u64,
    },
}

/// The span of physical memory that is the hypervisor's own.
///
/// The reserved chunk and nothing else: the loader copies pulzar's image into
/// chunk frames, and its heap, stacks, tables and control blocks all come from
/// the same place, so this one range is the whole of what a guest must not see.
#[derive(Clone, Copy, Debug)]
struct Owned {
    base: u64,
    end: u64,
}

impl Owned {
    /// Whether this address is the hypervisor's.
    fn contains(self, phys: PhysAddr) -> bool {
        (self.base..self.end).contains(&phys.as_u64())
    }

    /// Whether any of `span` bytes from `base` is the hypervisor's.
    fn overlaps(self, base: PhysAddr, span: u64) -> bool {
        let base = base.as_u64();
        base < self.end && self.base < base.saturating_add(span)
    }
}

/// Flags on a leaf that describes real memory the guest may use freely.
///
/// Write-back, which is the absence of both cache bits rather than a bit of its
/// own: it is what leaves the guest's own memory type in force.
const LEAF: PageTableFlags = PARENT;

/// Flags on a leaf that stands in for the hypervisor's own memory: readable and
/// executable so that a guest walking memory is not surprised, and not writable
/// so that the shared page of zeroes stays zero.
const SHADOW: PageTableFlags = PageTableFlags::PRESENT.union(PageTableFlags::USER_ACCESSIBLE);

/// A zeroed frame of the chunk, and proof that the window reaches it.
///
/// Reaching it is checked here rather than at first use because a frame the
/// window cannot describe is a broken window, and finding that out while
/// building the tables beats finding it out inside a fault handler.
fn frame(frames: &mut Frames, window: DirectMap) -> Result<PhysAddr, NptError> {
    let frame = frames
        .allocate(0)
        .ok_or(NptError::OutOfFrames)?
        .start_address();
    if window.virt(frame).is_none() {
        return Err(NptError::Unreachable {
            phys: frame.as_u64(),
        });
    }
    Ok(frame)
}

const _: () = assert!(
    chunk::CHUNK_SIZE.is_multiple_of(Level::Directory.span()),
    "the shadow describes the chunk in whole 2 MiB regions",
);
const _: () = assert!(
    chunk::CHUNK_ALIGN.is_multiple_of(Level::Directory.span()),
    "a 2 MiB region overlapping the chunk must lie entirely inside it",
);
const _: () = assert!(
    !SHADOW.contains(PageTableFlags::WRITABLE),
    "the page of zeroes standing in for the hypervisor must not be writable",
);
const _: () = assert!(
    LEAF.contains(PageTableFlags::USER_ACCESSIBLE)
        && !LEAF.contains(PageTableFlags::NO_EXECUTE)
        && !LEAF.contains(PageTableFlags::WRITE_THROUGH)
        && !LEAF.contains(PageTableFlags::NO_CACHE),
    "a guest reaches nested memory only through a user page, executes only \
     without no-execute, and keeps its own memory type only under write-back",
);
