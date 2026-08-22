//! What a guest physical address means, and how far the same answer reaches.
//!
//! This is the whole of the decision and none of the machinery. Nothing here
//! knows about page tables, frames or the window onto physical memory: it is a
//! handful of records and one function over them, and bringing the hardware
//! tables to whatever it says belongs elsewhere.
//!
//! # One answer, over a run
//!
//! [`Map::resolve`] answers for a *run* of addresses rather than for a page,
//! because an answer that covered only a page is what would force every page to
//! be described on its own. The run it reports is the largest one over which
//! the answer does not change, so the granularity a translation is written down
//! at falls out of the answer instead of being probed for.
//!
//! # The order of the questions is not interchangeable
//!
//! An address is asked about in one order, and every step is there because the
//! step after it would otherwise answer first and answer wrongly:
//!
//! 1. **Above what the processor can address.** Not an address this machine
//!    has, so nothing may describe it as one.
//! 2. **Inside a region something other than the hardware answers for.** An
//!    address whose accesses must fault must never be described — not even as
//!    the memory really behind it.
//! 3. **A sunk page**, which the guest may write and nothing reads back.
//! 4. **A page of the hypervisor's own memory the guest is shown on purpose.**
//! 5. **The rest of the hypervisor's own memory**, which reads as one shared
//!    page of zeroes and can never take a write.
//! 6. **Ordinary memory**, which is the machine's own at the same address.
//!
//! Steps 2 to 4 are exclusive of each other, refused when a range is recorded
//! rather than untangled at every lookup. Step 5 is not: a region may lie
//! inside the hypervisor's own memory, and the order is what makes such an
//! address answer as the region.
//!
//! # Invariants
//!
//! - Every set is sorted and non-overlapping, checked when a range is added and
//!   relied on by every lookup.
//! - A run is never empty, and never reaches past a boundary of a category that
//!   outranks the one it answers with.
//! - [`Range::new`] is the only place that decides whether a range is valid.
//! - A mutation that fails part-way leaves the map saying less than was asked,
//!   never something untrue. The hardware tables are brought to whatever it
//!   says, so an address any of it reached is answered afresh the next time the
//!   guest touches it.

mod regions;

use bitflags::bitflags;
use log::info;
use paging::chunk;
use thiserror::Error;
use x86_64::PhysAddr;

use crate::map::regions::Regions;

/// Everything a guest's physical addresses mean that is not "ordinary memory".
///
/// Ordinary memory is the answer everywhere none of these records reaches,
/// which is almost everywhere, so nothing describes it and there is nothing to
/// keep in step with the machine's own memory map.
#[derive(Debug)]
pub(crate) struct Map {
    /// The memory that is the hypervisor's own.
    chunk: Range,
    /// One past the highest address this processor can reach.
    limit: u64,
    /// The regions something other than the hardware answers for.
    regions: Regions<Interposition, REGIONS>,
    /// The pages the guest may write that nothing reads back, and the frames
    /// their writes are kept by.
    sinks: Regions<PhysAddr, SINKS>,
    /// The pages of the hypervisor's own memory the guest is shown, and what it
    /// may do with each.
    exposures: Regions<Access, EXPOSED_PAGES>,
}

impl Map {
    /// A map of a machine whose memory is the guest's, less the hypervisor's
    /// own.
    ///
    /// `bits` is how many bits of a physical address the processor implements,
    /// which is what makes an address `Unaddressable` rather than merely
    /// unused. More than the entry format has room for is taken as that
    /// maximum: an address the tables cannot hold is not one this map can
    /// answer for either.
    pub(crate) const fn new(chunk: Range, bits: u8) -> Self {
        let bits = if bits < ADDRESS_BITS {
            bits
        } else {
            ADDRESS_BITS
        };
        Self {
            chunk,
            limit: 1 << bits,
            regions: Regions::new(),
            sinks: Regions::new(),
            exposures: Regions::new(),
        }
    }

    /// What this address means, and how far the same answer reaches.
    ///
    /// The run is clipped by the nearest edge of every category that outranks
    /// the one answered with, which is why an address in the middle of nowhere
    /// answers for gigabytes and an address a page below a device answers for
    /// exactly that page.
    pub(crate) fn resolve(&self, gpa: PhysAddr) -> Verdict {
        let address = gpa.as_u64();
        if address >= self.limit {
            return verdict(self.limit, ADDRESS_LIMIT, Kind::Unaddressable);
        }
        if let Some(region) = self.regions.find(address) {
            return verdict(
                region.range.first(),
                region.range.end(),
                Kind::Interposed {
                    tag: region.what.tag,
                    trap: region.what.trap,
                },
            );
        }
        if let Some(sunk) = self.sinks.find(address) {
            return verdict(
                sunk.range.first(),
                sunk.range.end(),
                Kind::Sink { spa: sunk.what },
            );
        }
        if let Some(shown) = self.exposures.find(address) {
            return verdict(
                shown.range.first(),
                shown.range.end(),
                Kind::Exposed {
                    spa: shown.range.base(),
                    access: shown.what,
                },
            );
        }
        // What is left is one answer over the run that no recorded range and
        // neither edge of the chunk falls inside.
        let (first, end) = [
            self.regions.run(address),
            self.sinks.run(address),
            self.exposures.run(address),
            self.chunk.run(address),
        ]
        .into_iter()
        .fold((0, self.limit), |(first, end), (lower, upper)| {
            (first.max(lower), end.min(upper))
        });
        if self.chunk.contains(address) {
            verdict(first, end, Kind::Shadow)
        } else {
            verdict(
                first,
                end,
                Kind::Ram {
                    spa: PhysAddr::new_truncate(first),
                    access: Access::all(),
                },
            )
        }
    }

    /// The memory that is the hypervisor's own.
    pub(crate) const fn chunk(&self) -> Range {
        self.chunk
    }

    /// Whether the map already says exactly this: a region taken over by
    /// exactly this range, letting exactly these accesses through.
    ///
    /// What tells a mutation with nothing to do from one with something to
    /// change. Exactly, and by the same geometry a region is given back by,
    /// because a request covering part of a region is not that region — and is
    /// refused when it is recorded rather than answered for here.
    pub(crate) fn interposed(&self, range: Range, trap: Trap) -> bool {
        self.regions
            .find(range.first())
            .is_some_and(|region| region.range == range && region.what.trap == trap)
    }

    /// Records a region something other than the hardware answers for, and
    /// answers by what name.
    ///
    /// # Errors
    ///
    /// [`MapError::Unaddressable`] if the region reaches past what the
    /// processor can address, [`MapError::Overlaps`] if any of it is
    /// already described another way, or [`MapError::Full`] if there is no
    /// room for another.
    pub(crate) fn interpose(&mut self, range: Range, trap: Trap) -> Result<RegionTag, MapError> {
        self.addressable(range)?;
        exclusive(
            range,
            self.sinks
                .clashing(range)
                .or_else(|| self.exposures.clashing(range)),
        )?;
        let tag = self.name()?;
        self.regions.insert(range, Interposition { trap, tag })?;
        Ok(tag)
    }

    /// Stops answering for a region, by exactly the range it was taken over by.
    ///
    /// # Errors
    ///
    /// [`MapError::NoRegion`] if no region has exactly this geometry.
    pub(crate) fn release(&mut self, range: Range) -> Result<(), MapError> {
        self.regions.remove(range)?;
        Ok(())
    }

    /// Records one page as somewhere the guest's writes may go and nothing
    /// reads, and the frame that keeps them.
    ///
    /// One page and not a range, because the frame is one frame: a run of pages
    /// answering with a single frame would be a run whose pages all translate
    /// to the same place, which is not something anything here wants and is
    /// a live-lock if a fault ever believed it.
    ///
    /// # Errors
    ///
    /// [`MapError::Geometry`] unless the address is page aligned,
    /// [`MapError::Unaddressable`] if it is past what the processor can
    /// address, [`MapError::Overlaps`] if it is already described another
    /// way — a page already sunk included — or [`MapError::Full`] if there
    /// is no room for another.
    pub(crate) fn sink(&mut self, page: PhysAddr, frame: PhysAddr) -> Result<(), MapError> {
        let page = Range::new(page, chunk::FRAME_SIZE)?;
        self.addressable(page)?;
        exclusive(
            page,
            self.regions
                .clashing(page)
                .or_else(|| self.exposures.clashing(page)),
        )?;
        self.sinks.insert(page, frame)
    }

    /// Records a range of the hypervisor's own memory as one the guest is
    /// shown.
    ///
    /// Kept a page at a time rather than as the range it was named by, because
    /// showing and taking back do not come in matching shapes: entry code is
    /// shown a page at a time with the access that page needs and retired in
    /// one call over all of it. A page is the granularity every one of
    /// these is described at in any case, so recording them that way loses
    /// nothing and makes both directions exact.
    ///
    /// # Errors
    ///
    /// [`MapError::Unaddressable`] if the range reaches past what the processor
    /// can address, [`MapError::Overlaps`] if any of it is already described
    /// another way, or [`MapError::Full`] if there is no room for every page.
    pub(crate) fn expose(&mut self, range: Range, access: Access) -> Result<(), MapError> {
        self.addressable(range)?;
        exclusive(
            range,
            self.regions
                .clashing(range)
                .or_else(|| self.sinks.clashing(range)),
        )?;
        range
            .pages()
            .try_for_each(|page| self.exposures.insert(page, access))
    }

    /// Stops showing every page of a range.
    ///
    /// # Errors
    ///
    /// [`MapError::NoRegion`] if a page of the range was not being shown.
    pub(crate) fn conceal(&mut self, range: Range) -> Result<(), MapError> {
        range.pages().try_for_each(|page| {
            self.exposures.remove(page)?;
            Ok(())
        })
    }

    /// Logs everything the map holds, which is all of what a guest's view of
    /// memory is that the machine's own memory map does not already say.
    pub(crate) fn describe(&self, who: &str) {
        info!(
            "{who}: npt answers for guest physical addresses below {:#x}",
            self.limit,
        );
        info!(
            "{who}: npt shadows physical {:#x}..{:#x}, read only",
            self.chunk.first(),
            self.chunk.end(),
        );
        for region in self.regions.iter() {
            info!(
                "{who}: npt traps physical {:#x}..{:#x} as region {}, {}",
                region.range.first(),
                region.range.end(),
                region.what.tag.number(),
                match region.what.trap {
                    Trap::Writes => "writes only",
                    Trap::Everything => "every access",
                },
            );
        }
        for sunk in self.sinks.iter() {
            info!(
                "{who}: npt sinks guest physical {:#x} onto frame {:#x}, writable and never read \
                 back",
                sunk.range.first(),
                sunk.what,
            );
        }
        for shown in self.exposures.iter() {
            info!(
                "{who}: npt shows guest physical {:#x} as itself, readable and {}executable",
                shown.range.first(),
                if shown.what.contains(Access::EXECUTE) {
                    ""
                } else {
                    "not "
                },
            );
        }
    }

    /// A name no region here already holds.
    ///
    /// Drawn from the regions themselves rather than from a counter, because a
    /// counter that wrapped would hand out a name a live region still holds,
    /// and whatever keyed a device by that name would then reach the wrong
    /// device. There are a thousand names for every region the set can
    /// hold, so one is free whenever there is room for a region at all.
    fn name(&self) -> Result<RegionTag, MapError> {
        (0..=u16::MAX)
            .map(RegionTag)
            .find(|candidate| {
                !self
                    .regions
                    .iter()
                    .any(|region| region.what.tag == *candidate)
            })
            .ok_or(MapError::Full { limit: REGIONS })
    }

    /// Refuses a range that reaches past what the processor can address.
    fn addressable(&self, range: Range) -> Result<(), MapError> {
        if range.end() > self.limit {
            return Err(MapError::Unaddressable {
                gpa: range.first().max(self.limit),
            });
        }
        Ok(())
    }
}

/// What a guest physical address means, and how far the same answer reaches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Verdict {
    /// The first address the answer covers, aligned to at least a page.
    pub base: PhysAddr,
    /// Bytes from [`Verdict::base`] the same answer covers, a whole number of
    /// pages and never none.
    pub span: u64,
    /// What those addresses are.
    pub kind: Kind,
}

impl Verdict {
    /// Whether the whole of `bytes` from `base` is inside the run this answers
    /// for.
    ///
    /// What turns an answer into a granularity: the coarsest page whose whole
    /// extent this covers is the coarsest one the answer may be written down
    /// in.
    #[must_use]
    pub fn covers(&self, base: PhysAddr, bytes: u64) -> bool {
        let Some(offset) = base.as_u64().checked_sub(self.base.as_u64()) else {
            return false;
        };
        self.span.saturating_sub(offset) >= bytes
    }
}

/// What a run of guest physical addresses is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Real memory the guest may use, beginning at `spa`.
    Ram {
        /// Where the run really is.
        spa: PhysAddr,
        /// What the guest may do there.
        access: Access,
    },
    /// The hypervisor's own memory. Reads see one shared page of zeroes; a
    /// write cannot be satisfied and never will be.
    Shadow,
    /// The hypervisor's own memory, shown to the guest on purpose: the code it
    /// is entered at and that code's parameters. Never writable.
    Exposed {
        /// Where the run really is.
        spa: PhysAddr,
        /// What the guest may do there.
        access: Access,
    },
    /// Present and writable, backed by a frame nothing reads back.
    ///
    /// What an interrupt controller the processor drives itself needs of its
    /// register page, and the whole of what it needs: the hardware checks that
    /// the page translates to memory the guest may write and then never touches
    /// what it translates to.
    Sink {
        /// The frame the guest's writes are kept by.
        spa: PhysAddr,
    },
    /// The hypervisor answers, not the hardware behind it.
    Interposed {
        /// The name whatever answers for the region is known by.
        tag: RegionTag,
        /// Which of the guest's accesses reach it.
        trap: Trap,
    },
    /// Above the processor's physical address width, so not an address this
    /// machine has at all.
    Unaddressable,
}

bitflags! {
    /// What a guest may do with a page it can reach.
    ///
    /// Reading is deliberately not representable. Not present is the
    /// architecture's only encoding that denies a read, so every page a guest can
    /// reach at all is readable, and a flag for it would be a flag that can be
    /// wrong.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Access: u8 {
        /// The guest may store there.
        const WRITE = 1 << 0;
        /// The guest may execute from there.
        const EXECUTE = 1 << 1;
    }
}

/// What a guest may still do for itself in a region something else answers for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Trap {
    /// Reads reach the hardware directly and cost nothing; writes fault.
    ///
    /// The choice for a device whose registers read back what they are, and
    /// where only what the guest writes has to be interfered with.
    Writes,
    /// Every access faults.
    ///
    /// The choice for a device the guest must be shown something other than the
    /// truth about. It costs an exit per read as well as per write, which is
    /// the price of the read never reaching the hardware.
    Everything,
}

/// The name a region is known by, handed out when it is recorded.
///
/// What lets something else key a device by a region without holding a second
/// copy of where the region is. Names are drawn from the live regions, so one
/// is only ever held by one region at a time, and a name is free to be handed
/// out again once the region holding it has gone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegionTag(pub(crate) u16);

impl RegionTag {
    /// The name, as a number something else can index by.
    #[must_use]
    pub const fn number(self) -> u16 {
        self.0
    }
}

/// A run of guest physical addresses: a whole number of pages, on a page
/// boundary, inside the address space the entry format can hold.
///
/// Valid by construction. [`Range::new`] is the only way to make one and the
/// only place that decides what a valid range is, so nothing holding one has to
/// check it again and no two callers can disagree about what they may ask for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Range {
    base: PhysAddr,
    bytes: u64,
}

impl Range {
    /// The run of `bytes` beginning at `base`.
    ///
    /// # Errors
    ///
    /// [`MapError::Geometry`] unless the run is a non-empty whole number of
    /// pages on a page boundary that stays inside the address space — a
    /// page being the granularity every permission these tables express
    /// comes in.
    pub fn new(base: PhysAddr, bytes: u64) -> Result<Self, MapError> {
        let first = base.as_u64();
        let shaped = bytes != 0
            && first.is_multiple_of(chunk::FRAME_SIZE)
            && bytes.is_multiple_of(chunk::FRAME_SIZE)
            && first
                .checked_add(bytes)
                .is_some_and(|end| end <= ADDRESS_LIMIT);
        if shaped {
            Ok(Self { base, bytes })
        } else {
            Err(MapError::Geometry { base: first, bytes })
        }
    }

    /// The page the run begins at.
    #[must_use]
    pub const fn base(self) -> PhysAddr {
        self.base
    }

    /// How long the run is.
    #[must_use]
    pub const fn bytes(self) -> u64 {
        self.bytes
    }

    /// The run's first address.
    pub(crate) const fn first(self) -> u64 {
        self.base.as_u64()
    }

    /// One past the run's last address.
    pub(crate) const fn end(self) -> u64 {
        self.first() + self.bytes
    }

    /// Whether an address is inside the run.
    pub(crate) const fn contains(self, gpa: u64) -> bool {
        self.first() <= gpa && gpa < self.end()
    }

    /// Whether any address is in both runs.
    pub(crate) const fn overlaps(self, other: Self) -> bool {
        self.first() < other.end() && other.first() < self.end()
    }

    /// Whether every address of `other` is in this run.
    pub(crate) const fn covers(self, other: Self) -> bool {
        self.first() <= other.first() && other.end() <= self.end()
    }

    /// The largest run containing `gpa` that this one neither begins nor ends
    /// inside.
    ///
    /// A set of one, answering what [`Regions::run`](regions::Regions::run)
    /// answers for many, so the chunk clips a run the same way every recorded
    /// range does.
    pub(crate) const fn run(self, gpa: u64) -> (u64, u64) {
        if self.contains(gpa) {
            (self.first(), self.end())
        } else if gpa < self.first() {
            (0, self.first())
        } else {
            (self.end(), u64::MAX)
        }
    }

    /// Every page of the run, from the lowest upwards.
    pub(crate) fn pages(self) -> impl Iterator<Item = Self> {
        (0..self.bytes / chunk::FRAME_SIZE).map(move |index| Self {
            base: self.base + index * chunk::FRAME_SIZE,
            bytes: chunk::FRAME_SIZE,
        })
    }

    /// The first address of every `span`-aligned region of the address space
    /// that any part of this run falls in.
    ///
    /// `span` is what one entry of some level describes, so this is exactly the
    /// set of regions at that level whose granularity a mutation over this run
    /// could have changed the need for.
    pub(crate) fn aligned(self, span: u64) -> impl Iterator<Item = PhysAddr> {
        (self.first() / span..=(self.end() - 1) / span)
            .map(move |region| PhysAddr::new_truncate(region * span))
    }
}

/// Why the map would not describe a range the way it was asked to.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum MapError {
    /// A range was not a whole number of pages on a page boundary inside the
    /// address space.
    #[error("a range at {base:#x} of {bytes:#x} bytes is not a whole number of pages")]
    Geometry {
        /// Where it was said to begin.
        base: u64,
        /// How long it was said to be.
        bytes: u64,
    },
    /// A range covers part of one the map already describes another way.
    #[error("a range at {base:#x} overlaps the one at {other:#x}")]
    Overlaps {
        /// Where the range that was refused begins.
        base: u64,
        /// Where the range it ran into begins.
        other: u64,
    },
    /// Nothing is recorded at exactly this range.
    ///
    /// Given back by exactly the range it was recorded by, deliberately: giving
    /// back half of one would leave the other half described by a record that
    /// no longer says where it is.
    #[error("nothing the map holds covers exactly {base:#x} for {bytes:#x} bytes")]
    NoRegion {
        /// Where the range was said to begin.
        base: u64,
        /// How long it was said to be.
        bytes: u64,
    },
    /// One of the map's sets has no room for another range.
    #[error("no room for another range; {limit} is the most the map holds")]
    Full {
        /// How many it holds.
        limit: usize,
    },
    /// An address above the processor's physical address width, which is not an
    /// address this machine has.
    #[error("guest physical {gpa:#x} is above what this processor can address")]
    Unaddressable {
        /// The address in question.
        gpa: u64,
    },
}

impl MapError {
    /// The refusal a range earns when something already describes part of it.
    ///
    /// Built here for every set and every category, so one sentence says what
    /// overlapping means however the overlap was found.
    pub(crate) const fn overlaps(range: Range, other: Range) -> Self {
        Self::Overlaps {
            base: range.first(),
            other: other.first(),
        }
    }
}

/// What the map says about a region something other than the hardware answers
/// for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Interposition {
    /// Which of the guest's accesses to it fault.
    trap: Trap,
    /// The name whatever answers for it is known by.
    tag: RegionTag,
}

/// A verdict over the run from `first` up to but not including `end`.
///
/// Every run the map answers with begins at or below the address it was asked
/// about, which is a physical address already, so the truncation has nothing to
/// take — and reaching for it rather than for a checked conversion is what
/// keeps a fault, which is where every one of these comes from, off a path that
/// can panic.
fn verdict(first: u64, end: u64, kind: Kind) -> Verdict {
    Verdict {
        base: PhysAddr::new_truncate(first),
        span: end - first,
        kind,
    }
}

/// Refuses a range that something already describes another way.
///
/// The categories are exclusive: an address whose accesses must fault cannot
/// also be one the guest may write freely, and neither can be a page of the
/// hypervisor's own memory the guest is shown. Asked once per mutation, of the
/// sets the mutation is not adding to — the set it *is* adding to refuses an
/// overlap of its own.
fn exclusive(range: Range, other: Option<Range>) -> Result<(), MapError> {
    match other {
        Some(other) => Err(MapError::overlaps(range, other)),
        None => Ok(()),
    }
}

/// How many regions something other than the hardware may answer for at once.
///
/// Four times what a pass-through hypervisor has needed here, six comparisons
/// to search, and no allocation on any path. A machine that wanted hundreds
/// would want a different structure rather than a longer array.
const REGIONS: usize = 64;

/// How many pages may be sunk at once.
///
/// One per register page of an interrupt controller the processor drives
/// itself, which is one machine-wide; the rest is room for a second controller
/// to want the same treatment without the limit being what stops it.
const SINKS: usize = 4;

/// How many pages of the hypervisor's own memory may be shown to the guest at
/// once.
///
/// The code the guest is entered at and that code's parameters are two.
const EXPOSED_PAGES: usize = 4;

/// How many bits of a physical address a nested entry has room for.
///
/// The architecture's own ceiling on the width a processor may implement, and
/// the width the address types here are bounded by, so it is also the widest
/// map any processor can ask for.
const ADDRESS_BITS: u8 = 52;

/// One past the highest address any processor can implement.
const ADDRESS_LIMIT: u64 = 1 << ADDRESS_BITS;

const _: () = assert!(
    REGIONS < 1 << 16,
    "every region must have a name the tag type can hold",
);
const _: () = assert!(
    ADDRESS_LIMIT.is_multiple_of(chunk::FRAME_SIZE),
    "the address space must be a whole number of pages, which is what lets a run \
     that leaves it be refused by comparing its end against the space's",
);

#[cfg(test)]
mod tests {
    //! The map as a pure function: every answer it gives, how far each one
    //! reaches, and every range it refuses to record.
    //!
    //! None of it needs a frame, a table or a window onto physical memory,
    //! which is the whole point of deciding what an address means apart
    //! from writing that decision down.

    use x86_64::PhysAddr;

    use super::{
        ADDRESS_LIMIT, Access, Kind, Map, MapError, REGIONS, Range, RegionTag, Trap, chunk,
    };

    /// The page every range here is measured in.
    const PAGE: u64 = chunk::FRAME_SIZE;

    /// What one entry of the level above a page describes.
    const LARGE: u64 = 2 << 20;

    /// What one entry of the level above that describes.
    const HUGE: u64 = 1 << 30;

    /// Where the hypervisor's own memory begins on the machine these describe.
    const CHUNK: u64 = 4 << 30;

    /// How long it is: whole 2 MiB regions, as a real chunk is.
    const CHUNK_BYTES: u64 = 64 << 20;

    /// A device aperture below the chunk, where a real machine's is.
    const DEVICE: u64 = 0xFEE0_0000;

    /// Ordinary memory with nothing else within a gigabyte of it.
    const NOWHERE: u64 = 8 << 30;

    /// Where the regions of a full set are laid out, a page apart.
    const FIELD: u64 = 256 << 20;

    /// How many bits of a physical address the processor here implements.
    const BITS: u8 = 40;

    /// One past the highest address it can reach.
    const LIMIT: u64 = 1 << BITS;

    #[test]
    fn an_address_with_nothing_near_it_is_answered_for_by_gigabytes() {
        let map = map();
        let verdict = map.resolve(PhysAddr::new(NOWHERE));

        assert_eq!(
            verdict.kind,
            Kind::Ram {
                spa: PhysAddr::new(CHUNK + CHUNK_BYTES),
                access: Access::all(),
            },
            "ordinary memory is the machine's own, and the guest may do anything with it"
        );
        assert_eq!(
            verdict.base.as_u64(),
            CHUNK + CHUNK_BYTES,
            "the run reaches back to the far side of the hypervisor's own memory"
        );
        assert_eq!(
            verdict.span,
            LIMIT - CHUNK - CHUNK_BYTES,
            "and forward to the last address the processor has"
        );
        assert!(
            verdict.covers(PhysAddr::new(NOWHERE), HUGE),
            "so a whole gigabyte of it is one entry"
        );
    }

    #[test]
    fn a_run_of_memory_stops_at_the_edge_of_a_region() {
        let mut map = map();
        // A page into the aperture rather than at its base, so that the region's
        // edge is not one a large page would have stopped at anyway.
        let region = DEVICE + PAGE;
        map.interpose(range(region, PAGE), Trap::Everything)
            .expect("a region below the hypervisor's own memory");
        let below = PhysAddr::new(region - PAGE);
        let verdict = map.resolve(below);

        assert_eq!(
            (verdict.base.as_u64(), verdict.span),
            (0, region),
            "the run stops exactly at the region's base"
        );
        assert!(
            !verdict.covers(below.align_down(LARGE), LARGE),
            "so the 2 MiB holding the page below a region cannot be described at once"
        );
        assert!(
            verdict.covers(below, PAGE),
            "while the page itself always can, which is what makes one page the floor"
        );
    }

    #[test]
    fn an_address_inside_a_region_answers_with_the_name_it_was_given() {
        let mut map = map();
        let tag = map
            .interpose(range(DEVICE, 2 * PAGE), Trap::Writes)
            .expect("a region of two pages");
        let verdict = map.resolve(PhysAddr::new(DEVICE + PAGE));

        assert_eq!(
            verdict.kind,
            Kind::Interposed {
                tag,
                trap: Trap::Writes,
            },
            "a region answers by the name it was handed and by what it lets through"
        );
        assert_eq!(
            (verdict.base.as_u64(), verdict.span),
            (DEVICE, 2 * PAGE),
            "and the whole of it is one answer"
        );
    }

    #[test]
    fn the_hypervisors_own_memory_answers_as_a_shadow() {
        let map = map();
        let inside = map.resolve(PhysAddr::new(CHUNK + CHUNK_BYTES / 2));

        assert_eq!(inside.kind, Kind::Shadow);
        assert_eq!(
            (inside.base.as_u64(), inside.span),
            (CHUNK, CHUNK_BYTES),
            "all of it is the same answer"
        );
        assert!(
            inside.covers(PhysAddr::new(CHUNK), LARGE),
            "and it is whole 2 MiB regions, which is what lets the shadow be written a table at a \
             time"
        );

        let above = map.resolve(PhysAddr::new(CHUNK + CHUNK_BYTES));
        assert_eq!(
            above.base.as_u64(),
            CHUNK + CHUNK_BYTES,
            "the memory above it begins where it ends"
        );
        assert!(
            !above.covers(PhysAddr::new(CHUNK + CHUNK_BYTES).align_down(HUGE), HUGE),
            "so the gigabyte the hypervisor's memory sits in is not described at once"
        );
        assert!(
            above.covers(PhysAddr::new(CHUNK + CHUNK_BYTES), LARGE),
            "while the 2 MiB above it is"
        );
    }

    #[test]
    fn an_address_above_the_processors_width_is_not_one_the_machine_has() {
        let mut map = map();
        let verdict = map.resolve(PhysAddr::new(LIMIT));

        assert_eq!(verdict.kind, Kind::Unaddressable);
        assert_eq!(
            (verdict.base.as_u64(), verdict.span),
            (LIMIT, ADDRESS_LIMIT - LIMIT),
            "everything above what the processor implements is the same non-answer"
        );
        assert_eq!(
            map.interpose(range(LIMIT, PAGE), Trap::Everything),
            Err(MapError::Unaddressable { gpa: LIMIT }),
            "and nothing may be recorded there either"
        );
    }

    #[test]
    fn a_sunk_page_answers_for_itself_and_its_neighbours_do_not() {
        let mut map = map();
        let page = NOWHERE + PAGE;
        let frame = PhysAddr::new(CHUNK + 2 * PAGE);
        map.sink(PhysAddr::new(page), frame)
            .expect("a page of ordinary memory can be sunk");

        assert_eq!(
            map.resolve(PhysAddr::new(page)).kind,
            Kind::Sink { spa: frame },
            "a sunk page answers with the frame its writes are kept by"
        );

        let below = map.resolve(PhysAddr::new(NOWHERE));
        assert_eq!(
            below.kind,
            Kind::Ram {
                spa: PhysAddr::new(CHUNK + CHUNK_BYTES),
                access: Access::all(),
            },
            "the page below it is ordinary memory"
        );
        assert_eq!(
            below.base.as_u64() + below.span,
            page,
            "whose run stops at the sunk page"
        );
        assert!(
            !below.covers(PhysAddr::new(NOWHERE), LARGE),
            "so the 2 MiB holding one is described a page at a time"
        );
        assert_eq!(
            map.resolve(PhysAddr::new(page + PAGE)).base.as_u64(),
            page + PAGE,
            "and the run above it starts where it ends"
        );
    }

    #[test]
    fn an_exposed_page_answers_with_the_access_it_was_given() {
        let mut map = map();
        let code = CHUNK + 8 * PAGE;
        map.expose(range(code, PAGE), Access::EXECUTE)
            .expect("the code a guest is entered at can be shown");
        map.expose(range(code + PAGE, PAGE), Access::empty())
            .expect("and that code's parameters");

        for (page, access) in [(code, Access::EXECUTE), (code + PAGE, Access::empty())] {
            assert_eq!(
                map.resolve(PhysAddr::new(page)).kind,
                Kind::Exposed {
                    spa: PhysAddr::new(page),
                    access,
                },
                "a page the guest is shown answers as itself, with what it was given"
            );
            assert!(
                !access.contains(Access::WRITE),
                "and never as memory the guest may write"
            );
        }

        let before = map.resolve(PhysAddr::new(code - PAGE));
        assert_eq!(
            before.kind,
            Kind::Shadow,
            "the page below is the rest of the hypervisor's own memory"
        );
        assert_eq!(
            before.base.as_u64() + before.span,
            code,
            "whose run stops where the shown pages begin"
        );
    }

    #[test]
    fn the_order_of_the_questions_puts_a_region_above_the_chunk() {
        let mut map = map();
        let inside = CHUNK + 4 * PAGE;
        let tag = map
            .interpose(range(inside, PAGE), Trap::Everything)
            .expect("a region inside the hypervisor's own memory");

        assert_eq!(
            map.resolve(PhysAddr::new(inside)).kind,
            Kind::Interposed {
                tag,
                trap: Trap::Everything,
            },
            "an address whose accesses must fault is never described, the chunk included"
        );

        let shadow = map.resolve(PhysAddr::new(CHUNK));
        assert_eq!(shadow.kind, Kind::Shadow);
        assert_eq!(
            shadow.span,
            4 * PAGE,
            "and the shadow around it stops at its edge"
        );
    }

    #[test]
    fn a_page_cannot_be_both_sunk_and_trapped() {
        let frame = PhysAddr::new(CHUNK + 2 * PAGE);
        let clash = MapError::Overlaps {
            base: DEVICE,
            other: DEVICE,
        };

        let mut trapped = map();
        trapped
            .interpose(range(DEVICE, PAGE), Trap::Everything)
            .expect("a region");
        assert_eq!(
            trapped.sink(PhysAddr::new(DEVICE), frame),
            Err(clash),
            "sinking a trapped page would describe what must stay undescribed"
        );

        let mut sunk = map();
        sunk.sink(PhysAddr::new(DEVICE), frame)
            .expect("a sunk page");
        assert_eq!(
            sunk.interpose(range(DEVICE, PAGE), Trap::Everything),
            Err(clash),
            "and trapping a sunk page would take back the frame its writes go to"
        );
    }

    #[test]
    fn a_range_must_be_a_whole_number_of_pages_on_a_page_boundary() {
        let malformed = [
            (DEVICE, 0),
            (DEVICE + 1, PAGE),
            (DEVICE, PAGE - 1),
            (ADDRESS_LIMIT - PAGE, 2 * PAGE),
            (PAGE, u64::MAX - PAGE + 1),
        ];

        for (base, bytes) in malformed {
            assert_eq!(
                Range::new(PhysAddr::new(base), bytes),
                Err(MapError::Geometry { base, bytes }),
                "a range at {base:#x} of {bytes:#x} bytes is not one an entry can express"
            );
        }
    }

    #[test]
    fn the_region_set_holds_every_region_it_says_it_does_and_finds_any_of_them() {
        let mut map = map();
        let count = u64::try_from(REGIONS).expect("the region count is an address apart");
        let mut names = [RegionTag(0); REGIONS];

        // Recorded out of order and read back in order, because a set that sorted
        // nothing would pass a test that filled it from the bottom upwards. The
        // stride is coprime with the count, so every region is recorded once.
        for (slot, index) in (0..count).map(|step| (step * 37) % count).enumerate() {
            names[slot] = map
                .interpose(range(FIELD + index * 2 * PAGE, PAGE), Trap::Everything)
                .expect("a region of a set that is not full yet");
        }
        for (slot, name) in names.iter().enumerate() {
            assert!(
                !names[..slot].contains(name),
                "no two regions may answer by the same name"
            );
        }
        for index in [0, count / 2, count - 1] {
            let base = FIELD + index * 2 * PAGE;
            let verdict = map.resolve(PhysAddr::new(base));
            assert!(
                matches!(verdict.kind, Kind::Interposed { .. }),
                "a region at either end of the set is found as readily as one in the middle"
            );
            assert_eq!((verdict.base.as_u64(), verdict.span), (base, PAGE));
        }
        assert_eq!(
            map.resolve(PhysAddr::new(FIELD + PAGE)).span,
            PAGE,
            "and the page between two of them is one page of ordinary memory"
        );
        assert_eq!(
            map.interpose(range(FIELD + count * 2 * PAGE, PAGE), Trap::Everything),
            Err(MapError::Full { limit: REGIONS }),
            "a set with no room refuses rather than growing"
        );
    }

    #[test]
    fn a_region_is_given_back_only_by_the_range_it_was_taken_by() {
        let mut map = map();
        map.interpose(range(DEVICE, 2 * PAGE), Trap::Everything)
            .expect("a region of two pages");

        for base in [DEVICE, DEVICE + PAGE] {
            assert_eq!(
                map.release(range(base, PAGE)),
                Err(MapError::NoRegion { base, bytes: PAGE }),
                "half a region would leave the other half described by a record that no longer \
                 says where it is"
            );
        }
        map.release(range(DEVICE, 2 * PAGE))
            .expect("the range it was taken by gives it back");
        assert!(
            matches!(map.resolve(PhysAddr::new(DEVICE)).kind, Kind::Ram { .. }),
            "and the memory behind it is ordinary again"
        );
    }

    /// A map of a machine with the hypervisor's own memory in the middle of it.
    fn map() -> Map {
        Map::new(range(CHUNK, CHUNK_BYTES), BITS)
    }

    /// A range the map will accept, which is what every test here names.
    fn range(base: u64, bytes: u64) -> Range {
        Range::new(PhysAddr::new(base), bytes)
            .expect("the tests name whole pages on page boundaries")
    }
}
