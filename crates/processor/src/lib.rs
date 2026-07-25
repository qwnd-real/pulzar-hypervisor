//! What the processor can do, asked once and answered from a cache.
//!
//! `CPUID` is the only way to ask, and every subsystem that adapts to a
//! processor feature would otherwise ask it directly — with its own leaf
//! numbers, its own bit positions, and its own idea of which leaves a given
//! machine implements. This crate is the one place that asks. Everything else
//! reads [`features`], gets a value it can keep, and never sees a leaf number.
//!
//! Only the features pulzar acts on are modelled. A feature nothing adapts to
//! is a feature nothing needs to know about, and adding it here on the chance
//! that something might would put a bit position in the codebase that no code
//! reads.
//!
//! # Why one answer serves every processor
//!
//! The answer is read on the first call and kept, which is sound because none
//! of these can differ between the processors of one machine or change while it
//! runs: they are fixed at reset and uniform across a package. That is not true
//! of `CPUID` in general — a hybrid processor's cores report different cache
//! and performance topology — so a genuinely per-core feature does not belong
//! in this cache and would have to be read on the core that cares.
//!
//! The cache is also what makes [`features`] cheap enough to call at a use site
//! instead of plumbing a snapshot through: `CPUID` serializes the processor and
//! costs a few hundred cycles, which is worth paying once.

#![no_std]

use bitflags::bitflags;
use log::info;
use raw_cpuid::{ApmInfo, CpuId, ExtendedProcessorFeatureIdentifiers, FeatureInfo};
use spin::Once;

bitflags! {
    /// The processor features pulzar adapts to or refuses to run without.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Features: u32 {
        /// 1 GiB pages are available, so the direct map costs one
        /// page-directory-pointer entry per gigabyte instead of a whole page
        /// directory.
        const GIB_PAGES = 1 << 0;
        /// The no-execute bit in a page table entry is honoured rather than
        /// reserved. Without it nothing can be mapped non-executable, which
        /// pulzar refuses to run without.
        const NO_EXECUTE = 1 << 1;
        /// The processor has a hardware entropy source.
        const RDRAND = 1 << 2;
        /// The timestamp counter runs at a constant rate whatever the power
        /// state and whichever core reads it, which is what makes it a timebase
        /// rather than merely a cycle counter.
        const INVARIANT_TSC = 1 << 3;
    }
}

impl Features {
    /// Logs what the processor reported, present and absent alike: which of
    /// these a machine lacks is what explains the paths pulzar takes on it.
    pub fn describe(&self, who: &str) {
        info!("{who}: processor has {self:?}");
        let missing = Self::all().difference(*self);
        if !missing.is_empty() {
            info!("{who}: processor lacks {missing:?}");
        }
    }

    /// Asks `CPUID` for each feature.
    ///
    /// An absent leaf answers `false` for everything it would have reported,
    /// which is the correct reading: a processor that does not implement the
    /// leaf describing a feature does not have the feature.
    fn read() -> Self {
        let cpuid = CpuId::new();
        let extended = cpuid.get_extended_processor_and_feature_identifiers();
        let mut features = Self::empty();
        features.set(
            Self::GIB_PAGES,
            extended
                .as_ref()
                .is_some_and(ExtendedProcessorFeatureIdentifiers::has_1gib_pages),
        );
        features.set(
            Self::NO_EXECUTE,
            extended
                .as_ref()
                .is_some_and(ExtendedProcessorFeatureIdentifiers::has_execute_disable),
        );
        features.set(
            Self::RDRAND,
            cpuid
                .get_feature_info()
                .as_ref()
                .is_some_and(FeatureInfo::has_rdrand),
        );
        features.set(
            Self::INVARIANT_TSC,
            cpuid
                .get_advanced_power_mgmt_info()
                .as_ref()
                .is_some_and(ApmInfo::has_invariant_tsc),
        );
        features
    }
}

/// What this processor can do.
#[must_use]
pub fn features() -> Features {
    *FEATURES.call_once(Features::read)
}

/// The timestamp counter.
///
/// Raw and unordered: the processor may execute the read before or after
/// neighbouring instructions, and the rate it counts at is only fixed when
/// [`Features::INVARIANT_TSC`] is present. Both are the caller's to establish —
/// a timebase has to calibrate the rate, and a measurement of a short span has
/// to serialize around the read.
#[must_use]
pub fn timestamp() -> u64 {
    // SAFETY: `rdtsc` is implemented by every processor that can run 64-bit
    // code and reads a counter without side effects. `CR4.TSD` can make it
    // fault outside ring 0, and pulzar never leaves ring 0.
    unsafe { core::arch::x86_64::_rdtsc() }
}

/// The features of the processor this image runs on, read on first use.
static FEATURES: Once<Features> = Once::new();
