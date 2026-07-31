//! Randomized placement of the high-half regions.
//!
//! The three regions the hypervisor owns — its image, the direct map, and the
//! mapping window — get independent random bases, each drawn from its own slice
//! of the high half. Separate slices rather than one pool because it makes
//! overlap impossible by construction instead of by a retry loop, and because
//! each region wants a different alignment. An attacker who learns one base
//! learns nothing about the others.
//!
//! Stacks are not randomized separately: they are carved out of the mapping
//! window, so their addresses move with the window's base.
//!
//! # What a placement is and is not
//!
//! It is diversification. It is worth exactly the entropy behind it, and the
//! number of distinct bases a region can land on is bounded by its slice and
//! its alignment: sixteen bits for the direct map, fifteen for the mapping
//! window, twenty-three for the image — and less than that for a region large
//! enough to take up much of its slice, since a larger region has fewer places
//! to start. Those are counts of *choices*, not a promise about the bits of any
//! particular address.
//!
//! It is not a defence that can be silently degraded. A placement drawn from a
//! predictable source is a placement an attacker can compute, and one drawn
//! from a source that broke half way through is worse than one that was never
//! claimed to be random at all — so the source is chosen explicitly by the
//! caller, [`Entropy::secure`] fails rather than substitutes, and a hardware
//! source that stops answering ends the draw instead of falling back to a
//! mixer.
//!
//! [`Entropy::best_effort`] exists for a machine or an emulator with no
//! hardware source, where a layout that at least differs between boots is
//! better than a fixed one. It says what it is, in its name and in a warning,
//! and nothing built on it may be described as randomized against an attacker.

use log::warn;
use processor::Features;
use x86_64::{VirtAddr, instructions::random::RdRand};

use crate::{PagingError, chunk, round_up, virt_at};

/// Bytes one PML4 entry covers.
const PML4_SPAN: u64 = 1 << 39;

/// First PML4 index of the high half.
const HIGH_HALF: u64 = 256;

/// PML4 indices the direct map is placed in: 64 TiB of address space, ending
/// exclusively at the first index the mapping window's slice begins at. Enough
/// for any physical address space this will see, and 1 GiB granularity leaves
/// sixteen bits' worth of starting points for a region much smaller than the
/// slice.
const DIRECT_MAP_INDICES: u64 = 128;

/// PML4 indices the mapping window is placed in: 32 TiB, fifteen bits' worth of
/// starting points at 1 GiB granularity.
const MAPPING_WINDOW_INDICES: u64 = 64;

/// PML4 indices the image is placed in: 16 TiB, twenty-three bits' worth at
/// 2 MiB granularity.
const IMAGE_INDICES: u64 = 32;

/// Alignment of the direct map and the mapping window. Both are described with
/// large pages, and 1 GiB alignment lets the direct map use them at the top
/// level.
const REGION_ALIGN: u64 = 1 << 30;

/// Alignment of the image. Sections are mapped with 4 KiB pages, but a 2 MiB
/// aligned base keeps the image's own large-page-sized neighbourhood to itself.
const IMAGE_ALIGN: u64 = 2 << 20;

/// Where each high-half region was placed.
#[derive(Clone, Copy, Debug)]
pub struct Placement {
    /// Virtual base the hypervisor image is relocated to.
    pub image_base: VirtAddr,
    /// Virtual base of the direct map.
    pub direct_map_base: VirtAddr,
    /// Virtual base of the mapping window.
    pub mapping_window_base: VirtAddr,
    /// Where the randomness behind these came from. Carried with the placement
    /// so that nothing downstream has to assume, and so a degraded boot says so
    /// wherever the layout is reported.
    pub source: Source,
}

/// Draws a placement for every high-half region.
///
/// `image_size` and `direct_map_size` are byte lengths, rounded up internally
/// to their region's alignment.
///
/// # Errors
///
/// [`PagingError::EmptyRegion`] for a region of no bytes, which has no base to
/// place; [`PagingError::Arithmetic`] if rounding a size up to its alignment
/// leaves the range of a `u64`; [`PagingError::RegionTooLarge`] if a region
/// does not fit the slice of the high half reserved for it — in practice only
/// reachable with a machine whose RAM exceeds the direct map's 64 TiB slice; or
/// [`PagingError::EntropyFailed`] if a hardware source stopped answering
/// part-way through, which leaves the placement undrawn rather than partly
/// random.
pub fn place(
    entropy: &mut Entropy,
    image_size: u64,
    direct_map_size: u64,
) -> Result<Placement, PagingError> {
    Ok(Placement {
        direct_map_base: draw(
            entropy,
            HIGH_HALF,
            DIRECT_MAP_INDICES,
            direct_map_size,
            REGION_ALIGN,
        )?,
        mapping_window_base: draw(
            entropy,
            HIGH_HALF + DIRECT_MAP_INDICES,
            MAPPING_WINDOW_INDICES,
            chunk::MAPPING_WINDOW_SIZE,
            REGION_ALIGN,
        )?,
        image_base: draw(
            entropy,
            HIGH_HALF + DIRECT_MAP_INDICES + MAPPING_WINDOW_INDICES,
            IMAGE_INDICES,
            image_size,
            IMAGE_ALIGN,
        )?,
        source: entropy.source(),
    })
}

/// Picks an `align`-aligned base for a `size`-byte region inside the `indices`
/// PML4 entries starting at `first_index`.
fn draw(
    entropy: &mut Entropy,
    first_index: u64,
    indices: u64,
    size: u64,
    align: u64,
) -> Result<VirtAddr, PagingError> {
    if size == 0 {
        return Err(PagingError::EmptyRegion);
    }
    let span = indices * PML4_SPAN;
    let size = round_up(size, align).ok_or(PagingError::Arithmetic {
        what: "rounding a region's size up to its alignment",
    })?;
    if size > span {
        return Err(PagingError::RegionTooLarge { size, span });
    }
    // `new_truncate` sign-extends bit 47, which turns an index of 256 or more
    // into its canonical high-half address.
    let start = VirtAddr::new_truncate(first_index * PML4_SPAN);
    let choices = (span - size) / align + 1;
    virt_at(
        start,
        entropy.below(choices)? * align,
        "the base of a high-half region",
    )
}

/// Where a placement's randomness came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// The processor's own entropy source. The only one a placement may be
    /// described as randomized against an attacker on.
    Hardware,
    /// A mixer seeded from the timestamp counter, because the machine offers
    /// nothing better and the caller asked for a layout anyway. It differs
    /// between boots on real hardware and can be narrowed down by anyone who
    /// can observe or control boot timing, which a virtual machine's owner can.
    Degraded,
}

/// Source of the randomness placement is drawn from.
///
/// Which one it is, is the caller's decision and not this type's: there is no
/// constructor that silently substitutes one for the other, and no path from
/// [`Source::Hardware`] to [`Source::Degraded`] at run time. A hardware source
/// that stops answering makes the draw fail, because a placement half of whose
/// bases came from `RDRAND` and half from a timestamp is one nothing can say
/// anything true about.
pub struct Entropy {
    source: Source,
    /// The instruction, if this is a hardware source.
    hardware: Option<RdRand>,
    /// State of the mixer, if this is a degraded one.
    state: u64,
}

impl Entropy {
    /// Opens the processor's hardware entropy source.
    ///
    /// # Errors
    ///
    /// [`PagingError::NoSecureEntropy`] if the processor has no `RDRAND`. A
    /// caller that would rather have a degraded layout than none has to say so,
    /// with [`Entropy::best_effort`].
    pub fn secure() -> Result<Self, PagingError> {
        // The processor crate answers whether the instruction exists; `RdRand`
        // is only the wrapper that issues it.
        processor::features()
            .contains(Features::RDRAND)
            .then(RdRand::new)
            .flatten()
            .map(|hardware| Self {
                source: Source::Hardware,
                hardware: Some(hardware),
                state: 0,
            })
            .ok_or(PagingError::NoSecureEntropy)
    }

    /// Opens a source seeded from the timestamp counter, for a machine that
    /// offers nothing better.
    ///
    /// Deliberately not a fallback anything reaches by itself. Whoever calls
    /// this has decided that a layout which merely differs between boots is
    /// worth having, and the warning it logs is part of that decision being on
    /// the record.
    #[must_use]
    pub fn best_effort() -> Self {
        warn!(
            "paging: drawing the high-half layout from the timestamp counter; \
             it will differ between boots and is not secure against an attacker \
             who can observe or control boot timing"
        );
        Self {
            source: Source::Degraded,
            hardware: None,
            state: processor::timestamp(),
        }
    }

    /// Which source this is.
    #[must_use]
    pub const fn source(&self) -> Source {
        self.source
    }

    /// Draws a value uniformly from `0..choices`.
    ///
    /// Uniform by rejection rather than by remainder: `next % choices` favours
    /// the low `2^64 % choices` values, and while that bias is unmeasurable for
    /// the choice counts here, the cost of not having it is one comparison and
    /// an occasional redraw. Getting it right where it does not matter is what
    /// keeps it right where it would.
    ///
    /// # Errors
    ///
    /// [`PagingError::EntropyFailed`] if the source would not answer.
    fn below(&mut self, choices: u64) -> Result<u64, PagingError> {
        if choices <= 1 {
            return Ok(0);
        }
        // Values at or above this many would be shared unevenly among the
        // choices, so they are drawn again instead.
        let limit = u64::MAX - u64::MAX % choices;
        loop {
            let value = self.next()?;
            if value < limit {
                return Ok(value % choices);
            }
        }
    }

    /// Draws the next raw value.
    ///
    /// `RDRAND` is allowed a bounded number of retries, as the architecture
    /// requires: it reports failure under heavy contention, and a fixed retry
    /// count avoids an unbounded loop if the source has actually broken. A
    /// source that has used them all is a source that has failed, and that is
    /// what is reported — the alternative, quietly finishing the draw with a
    /// mixer, would produce a placement described as hardware-random that
    /// partly is not.
    fn next(&mut self) -> Result<u64, PagingError> {
        const RETRIES: usize = 10;
        let Some(hardware) = self.hardware else {
            self.state ^= processor::timestamp();
            return Ok(mix(&mut self.state));
        };
        (0..RETRIES)
            .find_map(|_| hardware.get_u64())
            .ok_or(PagingError::EntropyFailed)
    }
}

/// One round of `SplitMix64`, whose avalanche turns a counter-like input into a
/// value with well-distributed bits — which is all the degraded source needs,
/// since whatever entropy it has comes from the timestamp and not from the
/// mixer.
fn mix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}
