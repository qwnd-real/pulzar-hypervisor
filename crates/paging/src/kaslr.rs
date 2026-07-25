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

use log::warn;
use x86_64::{VirtAddr, instructions::random::RdRand};

use crate::{PagingError, chunk};

/// Bytes one PML4 entry covers.
const PML4_SPAN: u64 = 1 << 39;

/// First PML4 index of the high half.
const HIGH_HALF: u64 = 256;

/// PML4 indices the direct map is placed in: 64 TiB, enough for any physical
/// address space this will see, and 1 GiB granularity gives sixteen bits of
/// placement entropy.
const DIRECT_MAP_INDICES: u64 = 128;

/// PML4 indices the mapping window is placed in: 32 TiB, fifteen bits at 1 GiB
/// granularity.
const MAPPING_WINDOW_INDICES: u64 = 64;

/// PML4 indices the image is placed in: 16 TiB, twenty-three bits at 2 MiB
/// granularity.
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
}

/// Draws a placement for every high-half region.
///
/// `image_size` and `direct_map_size` are byte lengths, rounded up internally
/// to their region's alignment.
///
/// # Errors
///
/// [`PagingError::RegionTooLarge`] if a region does not fit the slice of the
/// high half reserved for it — in practice only reachable with a machine whose
/// RAM exceeds the direct map's 64 TiB slice.
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
    let span = indices * PML4_SPAN;
    let size = size.next_multiple_of(align);
    if size > span {
        return Err(PagingError::RegionTooLarge { size, span });
    }
    // `new_truncate` sign-extends bit 47, which turns an index of 256 or more
    // into its canonical high-half address.
    let start = VirtAddr::new_truncate(first_index * PML4_SPAN);
    let choices = (span - size) / align + 1;
    Ok(start + entropy.next_u64() % choices * align)
}

/// Source of the randomness placement is drawn from.
///
/// `RDRAND` when the processor has it. The fallback exists so that a machine or
/// an emulator without it still gets a layout that differs between boots, but
/// it is not a substitute: it is derived from the timestamp counter, which an
/// attacker who can observe boot timing can narrow down. Its use is logged as a
/// warning for exactly that reason.
pub struct Entropy {
    source: Option<RdRand>,
    state: u64,
}

impl Entropy {
    /// Opens the best available source.
    #[must_use]
    pub fn new() -> Self {
        let source = RdRand::new();
        if source.is_none() {
            warn!("paging: no RDRAND; layout entropy degraded to the timestamp counter");
        }
        Self {
            source,
            state: timestamp(),
        }
    }

    /// Draws the next value.
    ///
    /// `RDRAND` is allowed a bounded number of retries, as the architecture
    /// requires: it reports failure under heavy contention, and a fixed retry
    /// count avoids an unbounded loop if the source has actually broken. A
    /// broken source falls through to the same mixer the missing-`RDRAND` case
    /// uses.
    pub fn next_u64(&mut self) -> u64 {
        const RETRIES: usize = 10;
        self.source
            .and_then(|source| (0..RETRIES).find_map(|_| source.get_u64()))
            .unwrap_or_else(|| {
                self.state ^= timestamp();
                mix(&mut self.state)
            })
    }
}

impl Default for Entropy {
    fn default() -> Self {
        Self::new()
    }
}

/// One round of `SplitMix64`, whose avalanche turns a counter-like input into a
/// value with well-distributed bits — which is all the fallback needs, since
/// the entropy comes from the timestamp and not from the mixer.
fn mix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// The timestamp counter, the only monotonically varying value available this
/// early in boot.
fn timestamp() -> u64 {
    // SAFETY: `rdtsc` is unconditionally available on x86-64 and reads a counter
    // without side effects. `CR4.TSD` could make it fault at CPL 3, and this
    // runs at CPL 0.
    unsafe { core::arch::x86_64::_rdtsc() }
}
