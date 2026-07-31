//! Telling the other processors that a translation they may hold is gone.
//!
//! Invalidating a page table entry evicts the stale translation from the
//! processor that wrote it and from no other. Every other processor keeps
//! whatever its own translation lookaside buffer cached until something tells
//! it otherwise, and there are two ways to tell it.
//!
//! # In hardware, where the processor can
//!
//! `INVLPGB` invalidates on every processor in the machine at once, and
//! `TLBSYNC` waits for them to have done it. No interrupt is raised, no handler
//! runs, and nothing has to be installed for it to work — so where it exists it
//! is used, and the rest of this module is never reached.
//!
//! # By interrupt, everywhere else
//!
//! Sending an interprocessor interrupt is not this crate's job and must not
//! become it: the subsystem that owns interprocessor interrupts is built on the
//! one that starts the other processors, which is built on this one. So the
//! direction is inverted. This module holds a slot; whoever owns interprocessor
//! interrupts [`install`]s a function into it, and the address space calls
//! whatever is there after it has invalidated something.
//!
//! What crosses that slot is a [`Flush`] — what stopped being described, rather
//! than the bare "something did" that would leave the far side no choice but to
//! drop every translation it has. It is one word wide because it has to travel
//! as one, and it merges with [`Flush::hull`] because several of them can
//! coalesce into a single delivery.
//!
//! # Why an empty slot is sometimes an answer and sometimes a refusal
//!
//! Before any other processor has been started there is no other translation
//! lookaside buffer in the machine, so "tell everyone else" is already true
//! when nobody has been told. An empty slot therefore reports success — but
//! only while this really is the only processor running. Once the machine has
//! more than one processor online, an empty slot means an invalidation has no
//! way to reach the others, and reporting success would be reporting that a
//! stale translation had been dropped when nothing had asked anyone to drop it.
//!
//! So the answer is conditioned on the online count rather than assumed, which
//! is what lets the whole address space subsystem work unchanged on a machine
//! with one processor — including one where starting the others was
//! deliberately left out — without the same code silently lying on a machine
//! where they were started and the hook was never installed.
//!
//! # The canonical hole, and why a range may not simply be encoded
//!
//! A 64-bit virtual address is canonical only if bits 48 and above copy bit 47,
//! which leaves a hole in the middle of the address space that no address lies
//! in. A page number with the sign extension dropped is therefore *not* a
//! linear coordinate: the last page of the lower half and the first page of the
//! upper half have adjacent numbers and are half an address space apart.
//!
//! Everything here that turns numbers back into addresses answers for that. A
//! range whose numbering crosses the hole, or that would step past the top of
//! the address space, is not encoded as a range at all — it degrades to
//! [`Flush::EVERYTHING`], which is always correct and never forms an address
//! that does not exist.
//!
//! # What the installed function may not do
//!
//! It runs while the address space lock is held, and it runs to completion
//! before the invalidating operation returns. It must not take that lock, on
//! either the sending or the receiving side: a handler that waited for a lock
//! the processor it is answering already holds would stop the machine.

use core::num::NonZeroU64;

use bitfield_struct::bitfield;
use spin::Once;
use x86_64::{
    VirtAddr,
    instructions::tlb::{self, Invlpgb},
    structures::paging::{Page, PageSize, Size2MiB, Size4KiB},
};

use crate::cpu;

/// What stopped being described, and so what every processor has to drop.
///
/// One 64-bit word, because it travels to the other processors as one: an
/// interprocessor interrupt carries a vector and nothing else, so whatever a
/// handler is to know has to fit in the space the sender left for it.
///
/// A range rather than a single address because invalidations arrive in runs —
/// one mapping is many pages — and [`Flush::EVERYTHING`] rather than a very
/// long range because past a certain length reloading the page table root costs
/// less than the invalidations it replaces.
#[bitfield(u64, new = false)]
#[derive(PartialEq, Eq)]
pub struct Flush {
    /// How much each of [`Flush::pages`] covers, and so how the page number is
    /// to be read.
    ///
    /// Never zero for any value this module constructs, which is what lets the
    /// word double as its own presence flag: a transport that packs one of
    /// these alongside "nothing is owed" can spell the second as all zeroes.
    #[bits(2)]
    extent: Extent,
    /// The first page, numbered in pages of [`Flush::extent`]'s size.
    ///
    /// Thirty-six bits is the whole of a four-level address space: forty-eight
    /// significant bits, of which the low twelve are the offset within a 4 KiB
    /// page. The bits above that are the sign extension every canonical address
    /// carries and are restored rather than stored — which is why a number here
    /// is not a coordinate, and why every use of one is checked against the
    /// canonical hole.
    #[bits(36)]
    first: u64,
    /// How many pages, counted in [`Flush::extent`]'s size. Never zero, and
    /// never above [`MAX_PAGES`].
    pages: u16,
    /// Reserved.
    #[bits(10)]
    __: u16,
}

/// The longest run a single request can encode.
///
/// A property of the encoding — the count field is what it is — and
/// deliberately not the same question as whether stepping a run of a given
/// length is worth it on a given processor. That second question is
/// [`worth_stepping`], which the software path asks and the hardware path does
/// not.
const MAX_PAGES: u64 = 64;

/// The longest run this crate steps one page at a time on the local processor.
///
/// Past this, one write of the page table root costs less than the
/// invalidations it replaces — it drops translations that were still good and
/// they fault back in, which is cheaper than issuing hundreds of instructions
/// each of which is itself a serializing operation. The exact crossover is a
/// property of the processor; sixty-four is inside the range where neither
/// answer is much worse than the other on any implementation, and it is stated
/// here as the policy it is rather than being confused with what the encoding
/// can hold.
const STEP_LIMIT: u64 = MAX_PAGES;

/// Whether a run of `pages` is short enough to be worth invalidating one page
/// at a time rather than dropping everything.
const fn worth_stepping(pages: u64) -> bool {
    pages <= STEP_LIMIT
}

/// The bits of a virtual address that say anything: forty-eight, on the
/// four-level paging this crate builds. Everything above them is the sign
/// extension.
const SIGNIFICANT: u64 = (1 << 48) - 1;

/// One past the highest page number of the lower canonical half, at 4 KiB
/// granularity: the first number on the far side of the canonical hole.
const HOLE_AT_4KIB: u64 = 1 << (47 - 12);

/// The same boundary at 2 MiB granularity.
const HOLE_AT_2MIB: u64 = 1 << (47 - 21);

impl Flush {
    /// Every translation this processor has, including the ones marked global.
    ///
    /// Global is the part that is easy to miss. Writing the page table root
    /// evicts non-global translations only, and firmware is free to have marked
    /// its own mappings global — so a request that means "all of it" has to say
    /// so, or dropping the firmware half of the address space would leave the
    /// other processors still able to reach it.
    pub const EVERYTHING: Self = Self::from_bits(0)
        .with_extent(Extent::Everything)
        .with_pages(1);

    /// A run of `pages` 4 KiB pages starting at `first`.
    ///
    /// This is [`Flush::EVERYTHING`] for a run that is empty, longer than the
    /// encoding or the stepping policy allows, or that would cross the
    /// canonical hole or run past the top of the address space: the first
    /// because a request describing nothing cannot be distinguished from no
    /// request at all once it is packed into a word, the second because
    /// that is where invalidating one at a time stops paying, and the third
    /// because the numbering a range is encoded in is not continuous across
    /// the hole.
    #[must_use]
    pub fn small(first: Page<Size4KiB>, pages: u64) -> Self {
        Self::bounded(
            Extent::Small,
            page_number(first.start_address(), Size4KiB::SIZE),
            pages,
        )
    }

    /// A run of `pages` 2 MiB pages starting at `first`.
    ///
    /// Counted in 2 MiB pages rather than in the 4 KiB pages they contain,
    /// because invalidating any address inside a large page evicts that page's
    /// whole translation: stepping such a range in 4 KiB units would issue five
    /// hundred and twelve instructions where one does.
    ///
    /// As [`Flush::small`] for a run this cannot describe.
    #[must_use]
    pub fn large(first: Page<Size2MiB>, pages: u64) -> Self {
        Self::bounded(
            Extent::Large,
            page_number(first.start_address(), Size2MiB::SIZE),
            pages,
        )
    }

    /// One request covering everything both of these do.
    ///
    /// What folds several requests into the one delivery that answers them all.
    /// The result spans the gap between two disjoint runs, which invalidates
    /// more than either asked for and never less — the direction an error here
    /// has to fall, since a translation left behind is one to memory that has
    /// been given to something else.
    ///
    /// Runs of different page sizes collapse to [`Flush::EVERYTHING`] rather
    /// than being converted into common units. Mixing the two means a large
    /// mapping and a small one were unmapped close enough together that neither
    /// had been acknowledged, which is rare enough that the arithmetic to do
    /// better would cost more than it saves. So does a hull that would span the
    /// canonical hole, which is what two runs in opposite halves produce.
    #[must_use]
    pub fn hull(self, other: Self) -> Self {
        let extent = self.extent();
        if extent != other.extent() || extent == Extent::Everything {
            return Self::EVERYTHING;
        }
        let first = self.first().min(other.first());
        let end =
            (self.first() + u64::from(self.pages())).max(other.first() + u64::from(other.pages()));
        Self::bounded(extent, first, end - first)
    }

    /// Drops what this describes from the processor running it.
    ///
    /// The other half of a shootdown: whoever carried the request across is
    /// what calls this, on the processor that was asked.
    pub fn apply(self) {
        let Some((base, size, pages)) = self.range() else {
            return cpu::flush_translations();
        };
        for page in 0..pages {
            // The run was proved to stay in one canonical half and inside the
            // address space when it was encoded, so every address here exists.
            tlb::flush(VirtAddr::new_truncate(base.as_u64() + page * size));
        }
    }

    /// The request as the word it travels as.
    ///
    /// Never zero, and typed so that it cannot be: the extent occupies the low
    /// bits and no request leaves it unset. Transports that carry one of these
    /// alongside "nothing is pending" spell the second as an empty word, and a
    /// request that could collide with that spelling would be a request that
    /// could be dropped.
    #[must_use]
    pub const fn bits(self) -> NonZeroU64 {
        match NonZeroU64::new(self.into_bits()) {
            Some(word) => word,
            // Unreachable, and harmless if it were not: one is a run of no
            // pages, which [`Flush::from_word`] reads back as everything.
            None => NonZeroU64::MIN,
        }
    }

    /// The request a word carries.
    ///
    /// Anything the encoding does not define — a zero word above all, a run of
    /// no pages, a count past what the encoding admits, and a run whose
    /// numbering crosses the canonical hole or leaves the address space — reads
    /// back as [`Flush::EVERYTHING`]. A word this module did not write is one
    /// nothing can be concluded from, and dropping every translation is the
    /// only conclusion that cannot leave a stale one behind.
    #[must_use]
    pub fn from_word(word: u64) -> Self {
        let flush = Self::from_bits(word);
        match flush.extent() {
            Extent::Small | Extent::Large if flush.range().is_some() => flush,
            _ => Self::EVERYTHING,
        }
    }

    /// The run this describes, as `(first address, bytes per page, pages)`, or
    /// `None` where it describes everything or describes nothing usable.
    ///
    /// The single place an encoded range is turned back into addresses, so the
    /// canonical-hole and top-of-space questions are asked once instead of at
    /// each of the three places that step a range.
    fn range(self) -> Option<(VirtAddr, u64, u64)> {
        let (size, hole) = match self.extent() {
            Extent::Everything => return None,
            Extent::Small => (Size4KiB::SIZE, HOLE_AT_4KIB),
            Extent::Large => (Size2MiB::SIZE, HOLE_AT_2MIB),
        };
        let pages = u64::from(self.pages());
        let first = self.first();
        let end = first.checked_add(pages)?;
        // A run has to stay on one side of the hole — the numbering is not
        // continuous across it — and inside the numbering altogether.
        let same_half = first >= hole || end <= hole;
        if pages == 0 || !worth_stepping(pages) || !same_half || end > 2 * hole {
            return None;
        }
        Some((VirtAddr::new_truncate(first * size), size, pages))
    }

    /// A run, or [`Flush::EVERYTHING`] where one would not describe it.
    fn bounded(extent: Extent, first: u64, pages: u64) -> Self {
        let Ok(pages) = u16::try_from(pages) else {
            return Self::EVERYTHING;
        };
        let flush = Self::from_bits(0)
            .with_extent(extent)
            .with_first(first)
            .with_pages(pages);
        // Encoded first and validated after, so that exactly one function
        // decides what a usable range is and every constructor is held to it.
        if pages == 0 || u64::from(pages) > MAX_PAGES || flush.range().is_none() {
            return Self::EVERYTHING;
        }
        flush
    }
}

/// Which page of `size` an address falls in, counted from the bottom of the
/// address space and with the sign extension dropped.
///
/// Dropped rather than stored: the bits above the significant ones are not
/// information, they are a copy of the highest one that every canonical address
/// is required to carry. Keeping them would put a high-half page number four
/// orders of magnitude above the field that holds it, and
/// [`VirtAddr::new_truncate`] writes them back on the way out for free.
fn page_number(address: VirtAddr, size: u64) -> u64 {
    (address.as_u64() & SIGNIFICANT) / size
}

/// How much one page of a request covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum Extent {
    /// Pages of 4 KiB, which is everything the mapping window hands out.
    Small = 1,
    /// Pages of 2 MiB, which is how the direct map describes the chunk.
    Large = 2,
    /// Not a run at all: every translation there is.
    Everything = 3,
}

impl Extent {
    /// The extent this encoding names.
    ///
    /// Zero is not one of them and neither is anything else undefined, and both
    /// read back as [`Extent::Everything`]: a word that does not decode is one
    /// nothing here wrote, and dropping every translation is the only response
    /// to that which cannot leave a stale one behind.
    const fn from_bits(bits: u8) -> Self {
        match bits {
            1 => Self::Small,
            2 => Self::Large,
            _ => Self::Everything,
        }
    }

    /// The encoding itself.
    const fn into_bits(self) -> u8 {
        self as u8
    }
}

/// Makes every other processor drop what `flush` describes, and reports whether
/// every one of them did.
///
/// A `false` answer means some processor did not respond in the time it was
/// given. It is not a reason to retry — the invalidation has already happened —
/// but it does mean the machine is in a state the caller has to be told about,
/// so it becomes [`PagingError::ShootdownIncomplete`](crate::PagingError).
pub type Shootdown = fn(Flush) -> bool;

/// How many processors are running.
///
/// Answered by whoever counts the machine's processors, and asked on the one
/// path that has to distinguish "nobody to tell" from "no way to tell anyone".
pub type OnlineCount = fn() -> usize;

/// Records how this address space reaches the other processors.
///
/// One-shot: the second caller is refused rather than allowed to replace a hook
/// that invalidations may already be going through.
///
/// # Errors
///
/// [`AlreadyInstalled`] if something already installed one.
pub fn install(hook: Shootdown) -> Result<(), AlreadyInstalled> {
    // The cell runs the closure for the one caller that fills it and for no
    // other, so whether it ran is exactly whether this call installed the hook.
    let mut installed = false;
    HOOK.call_once(|| {
        installed = true;
        hook
    });
    installed.then_some(()).ok_or(AlreadyInstalled)
}

/// Records how the number of running processors is asked for.
///
/// Separate from [`install`] and installable before it, because the two answer
/// different questions and become available at different points in bring-up:
/// the machine's processors are surveyed before anything can reach them. Until
/// this is set the address space is entitled to believe it is alone, which is
/// true of the boot processor before it has looked.
///
/// # Errors
///
/// [`AlreadyInstalled`] if something already installed one.
pub fn watch(online: OnlineCount) -> Result<(), AlreadyInstalled> {
    let mut installed = false;
    ONLINE.call_once(|| {
        installed = true;
        online
    });
    installed.then_some(()).ok_or(AlreadyInstalled)
}

/// A second attempt to say how the other processors are reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("a translation shootdown hook is already installed")]
pub struct AlreadyInstalled;

/// Whether this machine invalidates on every processor in hardware, and so
/// needs nothing installed to shoot a translation down.
#[must_use]
pub fn in_hardware() -> bool {
    broadcaster().is_some()
}

/// Drops what `flush` describes from every processor but this one, reporting
/// whether they all did.
///
/// Three ways, in the order they are worth having: the instruction that does it
/// without involving software at all, then whatever was installed to reach the
/// other processors, then the observation that a machine which has started
/// nobody has nobody to tell.
pub(crate) fn broadcast(flush: Flush) -> bool {
    if let Some(broadcaster) = broadcaster() {
        flush.broadcast(*broadcaster);
        return true;
    }
    match HOOK.get() {
        Some(hook) => hook(flush),
        // No way to reach anyone. That is a complete answer only while there is
        // nobody to reach; otherwise this invalidation has not happened
        // everywhere, and saying it has is the one answer that leaves a stale
        // translation behind believing it does not.
        None => ONLINE.get().is_none_or(|online| online() <= 1),
    }
}

impl Flush {
    /// Issues this as a hardware broadcast and waits for every processor to
    /// have acknowledged it.
    ///
    /// Global translations are included in every case, ranged and not. The
    /// local invalidation this follows drops a global entry as readily as
    /// any other, and a broadcast that did not would leave the other
    /// processors holding exactly the entries the local one dropped.
    fn broadcast(self, invlpgb: Invlpgb) {
        let mut builder = invlpgb.build();
        builder.include_global();
        match self.range() {
            None => builder.flush(),
            Some((base, size, pages)) => {
                // Split by extent only to name the page size in the type: the
                // addresses and the count are the ones `range` already proved
                // stay inside one canonical half.
                if size == Size4KiB::SIZE {
                    let first = Page::<Size4KiB>::containing_address(base);
                    builder.pages(Page::range(first, first + pages)).flush();
                } else {
                    let first = Page::<Size2MiB>::containing_address(base);
                    builder.pages(Page::range(first, first + pages)).flush();
                }
            }
        }
        invlpgb.tlbsync();
    }
}

/// The instruction that broadcasts invalidations, if this processor has it.
///
/// Asked once. The answer is a property of the machine, and the question costs
/// a `CPUID` that has no business being on the path of every unmapping.
fn broadcaster() -> Option<&'static Invlpgb> {
    BROADCASTER.call_once(Invlpgb::new).as_ref()
}

/// How the other processors are reached, or empty while nothing can reach them.
static HOOK: Once<Shootdown> = Once::new();

/// How many processors are running, or empty while nothing has counted them.
static ONLINE: Once<OnlineCount> = Once::new();

/// Whether the processor invalidates on every processor by itself, decided the
/// first time anything asks.
static BROADCASTER: Once<Option<Invlpgb>> = Once::new();
