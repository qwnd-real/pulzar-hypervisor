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
//! # Why an empty slot is an answer and not a gap
//!
//! Before any other processor has been started there is no other translation
//! lookaside buffer in the machine, so "tell everyone else" is already true
//! when nobody has been told. An empty slot therefore reports success rather
//! than failing or refusing, which is what lets the whole address space
//! subsystem work unchanged on a machine with one processor — including one
//! where starting the others was deliberately left out.
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
    registers::control::{Cr4, Cr4Flags},
    structures::paging::{Page, PageSize, Size2MiB, Size4KiB},
};

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
    /// carries and are restored rather than stored.
    #[bits(36)]
    first: u64,
    /// How many pages, counted in [`Flush::extent`]'s size. Never zero, and
    /// never above [`MAX_PAGES`].
    pages: u16,
    /// Reserved.
    #[bits(10)]
    __: u16,
}

/// The longest run of pages worth invalidating one at a time.
///
/// Past this, one write of the page table root costs less than the
/// invalidations it replaces — it drops translations that were still good and
/// they fault back in, which is cheaper than issuing hundreds of instructions
/// each of which is itself a serializing operation. The exact crossover is a
/// property of the processor and not worth measuring for: what matters is that
/// a bounded number of invalidations is never wildly worse than the
/// alternative, and sixty-four is comfortably inside that on every
/// implementation.
const MAX_PAGES: u64 = 64;

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
    /// Longer than [`MAX_PAGES`], or empty, and this is [`Flush::EVERYTHING`]:
    /// the first because that is where invalidating one at a time stops paying,
    /// the second because a request describing nothing cannot be distinguished
    /// from no request at all once it is packed into a word.
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
    /// As [`Flush::small`] for a run that is empty or too long.
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
    /// better would cost more than it saves.
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
        let size = match self.extent() {
            Extent::Everything => return everything(),
            Extent::Small => Size4KiB::SIZE,
            Extent::Large => Size2MiB::SIZE,
        };
        let base = VirtAddr::new_truncate(self.first() * size);
        for page in 0..u64::from(self.pages()) {
            tlb::flush(base + page * size);
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
    /// Anything the encoding does not define — a zero word above all, and a run
    /// of no pages with it — reads back as [`Flush::EVERYTHING`]. A word this
    /// module did not write is one nothing can be concluded from, and dropping
    /// every translation is the only conclusion that cannot leave a stale one
    /// behind.
    #[must_use]
    pub const fn from_word(word: u64) -> Self {
        let flush = Self::from_bits(word);
        match flush.extent() {
            Extent::Small | Extent::Large if flush.pages() > 0 => flush,
            _ => Self::EVERYTHING,
        }
    }

    /// A run, or [`Flush::EVERYTHING`] where one would not describe it.
    fn bounded(extent: Extent, first: u64, pages: u64) -> Self {
        let Ok(pages) = u16::try_from(pages) else {
            return Self::EVERYTHING;
        };
        if pages == 0 || u64::from(pages) > MAX_PAGES {
            return Self::EVERYTHING;
        }
        Self::from_bits(0)
            .with_extent(extent)
            .with_first(first)
            .with_pages(pages)
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

/// The bits of a virtual address that say anything: forty-eight, on the
/// four-level paging this crate builds. Everything above them is the sign
/// extension.
const SIGNIFICANT: u64 = (1 << 48) - 1;

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
/// other processors, then the observation that a machine which has told nobody
/// has nobody to tell.
pub(crate) fn broadcast(flush: Flush) -> bool {
    if let Some(broadcaster) = broadcaster() {
        flush.broadcast(*broadcaster);
        return true;
    }
    HOOK.get().is_none_or(|hook| hook(flush))
}

impl Flush {
    /// Issues this as a hardware broadcast and waits for every processor to
    /// have acknowledged it.
    fn broadcast(self, invlpgb: Invlpgb) {
        let mut builder = invlpgb.build();
        match self.extent() {
            Extent::Everything => {
                builder.include_global();
                builder.flush();
            }
            Extent::Small => {
                let first = Page::<Size4KiB>::containing_address(VirtAddr::new_truncate(
                    self.first() * Size4KiB::SIZE,
                ));
                builder
                    .pages(Page::range(first, first + u64::from(self.pages())))
                    .flush();
            }
            Extent::Large => {
                let first = Page::<Size2MiB>::containing_address(VirtAddr::new_truncate(
                    self.first() * Size2MiB::SIZE,
                ));
                builder
                    .pages(Page::range(first, first + u64::from(self.pages())))
                    .flush();
            }
        }
        invlpgb.tlbsync();
    }
}

/// Drops every translation this processor has, global ones included.
fn everything() {
    tlb::flush_all();
    let cr4 = Cr4::read();
    if cr4.contains(Cr4Flags::PAGE_GLOBAL) {
        // SAFETY: clearing `CR4.PGE` invalidates all global translations and is
        // architecturally permitted at any time; the original value is restored
        // immediately, so nothing observes the intermediate state.
        unsafe {
            Cr4::write(cr4.difference(Cr4Flags::PAGE_GLOBAL));
            Cr4::write(cr4);
        }
    }
}

/// The instruction that broadcasts invalidations, if this processor has it.
///
/// Asked once. The answer is a property of the machine, and the question costs
/// a `CPUID` that has no business being on the path of every unmapping.
fn broadcaster() -> Option<&'static Invlpgb> {
    BROADCASTER.call_once(Invlpgb::new).as_ref()
}

/// How the other processors are reached, or empty while there are none.
static HOOK: Once<Shootdown> = Once::new();

/// Whether the processor invalidates on every processor by itself, decided the
/// first time anything asks.
static BROADCASTER: Once<Option<Invlpgb>> = Once::new();
