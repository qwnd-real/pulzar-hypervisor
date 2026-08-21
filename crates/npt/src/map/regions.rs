//! A sorted, non-overlapping set of guest physical ranges.
//!
//! One structure serves all three sets the map keeps — the regions something
//! other than the hardware answers for, the pages the guest may write that
//! nothing reads back, and the pages of the hypervisor's own memory it is shown
//! — because the questions asked of all three are the same two: *which range
//! covers this address*, and *how far from this address is the nearest edge of
//! any of them*.
//!
//! Sorted by base and never overlapping is what makes the second question
//! answerable by one search rather than a scan: the range a search lands on
//! bounds the run from above and the range before it bounds it from below.
//! Overlap is refused when a range is added, once, so no lookup has to check
//! for it again.
//!
//! Nothing here allocates. The array is inline, its length is a constant, and a
//! set with no room refuses rather than growing.

use crate::map::{MapError, Range};

/// One range of a guest's physical addresses and what the map says about them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Region<W> {
    /// Which addresses it covers.
    pub(crate) range: Range,
    /// What they mean.
    pub(crate) what: W,
}

/// At most `N` ranges of a guest's physical addresses, in order and disjoint.
///
/// The array is the only record of how many ranges there are: they occupy a
/// prefix of it and every slot above them is empty. That is what lets one
/// binary search run over the whole array — an empty slot answers every
/// question the way a range above the address would — so there is no length to
/// keep in step with the contents.
#[derive(Debug)]
pub(crate) struct Regions<W, const N: usize> {
    regions: [Option<Region<W>>; N],
}

impl<W: Copy, const N: usize> Regions<W, N> {
    /// A set with nothing in it.
    pub(crate) const fn new() -> Self {
        Self { regions: [None; N] }
    }

    /// The range covering `gpa`, or `None` if none does.
    pub(crate) fn find(&self, gpa: u64) -> Option<Region<W>> {
        self.at(self.above(gpa))
            .filter(|region| region.range.contains(gpa))
    }

    /// The largest run containing `gpa` that no range here begins or ends
    /// inside.
    ///
    /// For an address inside a range that is the range itself, and for one
    /// between two ranges it is the gap between them. Both are the same answer
    /// to the same question — how far from `gpa` this set stops saying what it
    /// says there — which is why one function answers for both.
    pub(crate) fn run(&self, gpa: u64) -> (u64, u64) {
        let at = self.above(gpa);
        let below = at
            .checked_sub(1)
            .and_then(|before| self.at(before))
            .map_or(0, |region| region.range.end());
        match self.at(at) {
            Some(region) if region.range.contains(gpa) => {
                (region.range.first(), region.range.end())
            }
            Some(region) => (below, region.range.first()),
            None => (below, u64::MAX),
        }
    }

    /// The range here that covers part of `range`, if one does.
    pub(crate) fn clashing(&self, range: Range) -> Option<Range> {
        self.at(self.above(range.first()))
            .map(|region| region.range)
            .filter(|other| other.overlaps(range))
    }

    /// Every range here, from the lowest upwards.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &Region<W>> {
        self.regions.iter().flatten()
    }

    /// Adds a range and what it means.
    ///
    /// # Errors
    ///
    /// [`MapError::Overlaps`] if a range here already covers part of it, or
    /// [`MapError::Full`] if the set has no room for another.
    pub(crate) fn insert(&mut self, range: Range, what: W) -> Result<(), MapError> {
        if let Some(other) = self.clashing(range) {
            return Err(MapError::overlaps(range, other));
        }
        if self.regions.last().is_some_and(Option::is_some) {
            return Err(MapError::Full { limit: N });
        }
        let at = self.above(range.first());
        // The highest slot is empty, so this moves every range from `at` upwards
        // one place and brings that empty slot back down to `at`.
        self.regions[at..].rotate_right(1);
        self.regions[at] = Some(Region { range, what });
        Ok(())
    }

    /// Removes the range that was added with exactly this geometry, and answers
    /// what it meant.
    ///
    /// Exactly, deliberately: removing part of a range would leave the rest of
    /// it described by a record that no longer says where it is.
    ///
    /// # Errors
    ///
    /// [`MapError::NoRegion`] if no range here has exactly this geometry.
    pub(crate) fn remove(&mut self, range: Range) -> Result<W, MapError> {
        let at = self.above(range.first());
        let Some(region) = self.at(at).filter(|region| region.range == range) else {
            return Err(MapError::NoRegion {
                base: range.first(),
                bytes: range.bytes(),
            });
        };
        self.regions[at] = None;
        // Carries the empty slot to the top, which closes the gap the removal
        // left and keeps every remaining range in order.
        self.regions[at..].rotate_left(1);
        Ok(region.what)
    }

    /// Where the first range whose last byte is above `gpa` sits.
    ///
    /// `N` when every range here ends at or below `gpa`, the empty set
    /// included. An empty slot answers the search the same way a range above
    /// `gpa` would, which is what makes searching the whole array sound.
    fn above(&self, gpa: u64) -> usize {
        self.regions
            .partition_point(|region| region.is_some_and(|region| region.range.end() <= gpa))
    }

    /// The range one slot holds, if it holds one.
    fn at(&self, index: usize) -> Option<Region<W>> {
        self.regions.get(index).copied().flatten()
    }
}
