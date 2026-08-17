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
//! [`svm()`] is the deliberate exception. Its leaf is read whole, because those
//! bits do not merely gate behaviour — they describe which fields the
//! virtualization structures actually have, and a hypervisor that guesses
//! wrong there builds a control block the processor reads differently than it
//! was written.
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

pub mod svm;

use bitflags::bitflags;
use log::info;
use raw_cpuid::{
    ApmInfo, CpuId, ExtendedFeatures, ExtendedProcessorFeatureIdentifiers, FeatureInfo,
    ProcessorCapacityAndFeatureInfo, native_cpuid::cpuid_count,
};
use spin::Once;

pub use crate::svm::{Svm, SvmFeatures, svm};

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
        /// The processor has a local interrupt controller on board. Without it
        /// there is no way to address another processor, and no way to take a
        /// timer interrupt that is the hypervisor's own, so pulzar refuses to
        /// run without it.
        const APIC = 1 << 4;
        /// The local interrupt controller can be put into x2APIC mode, where
        /// its registers are model-specific registers rather than a page of
        /// memory-mapped ones and identifiers are 32 bits wide rather than
        /// eight. A machine with more than 255 processors can only be addressed
        /// this way.
        const X2APIC = 1 << 5;
        /// The local timer can be armed with a timestamp counter deadline
        /// instead of a countdown, which is the only one of its modes that is
        /// not quantized to the timer's own divided tick.
        const TSC_DEADLINE = 1 << 6;
        /// The page attribute table is implemented, so `IA32_PAT` exists and the
        /// `PWT` and `PCD` bits of a page table entry select one of its entries
        /// rather than naming a cache type directly. Without it there is no
        /// register to read or program, and the memory type a mapping asks for
        /// cannot be established.
        const PAGE_ATTRIBUTE_TABLE = 1 << 7;
        /// The timestamp-counter adjustment register exists, so software can
        /// observe and update the cumulative changes made to the counter.
        const TSC_ADJUST = 1 << 8;
        /// The local interrupt controller has the extended register space AMD
        /// adds above the architectural registers, which is where a vector can
        /// be retired by name rather than by priority and where a single vector
        /// can be stopped from being accepted at all.
        ///
        /// A capability of the space as a whole. Which parts of it a particular
        /// controller implements is in the space's own feature register, so a
        /// processor reporting this is one whose controller may be asked and not
        /// one that necessarily has any given part.
        const EXTENDED_APIC_SPACE = 1 << 9;
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
        let basic = cpuid.get_feature_info();
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
            basic.as_ref().is_some_and(FeatureInfo::has_rdrand),
        );
        features.set(
            Self::INVARIANT_TSC,
            cpuid
                .get_advanced_power_mgmt_info()
                .as_ref()
                .is_some_and(ApmInfo::has_invariant_tsc),
        );
        features.set(
            Self::APIC,
            basic.as_ref().is_some_and(FeatureInfo::has_apic),
        );
        features.set(
            Self::X2APIC,
            basic.as_ref().is_some_and(FeatureInfo::has_x2apic),
        );
        features.set(
            Self::TSC_DEADLINE,
            basic.as_ref().is_some_and(FeatureInfo::has_tsc_deadline),
        );
        features.set(
            Self::PAGE_ATTRIBUTE_TABLE,
            basic.as_ref().is_some_and(FeatureInfo::has_pat),
        );
        features.set(
            Self::TSC_ADJUST,
            cpuid
                .get_extended_feature_info()
                .as_ref()
                .is_some_and(ExtendedFeatures::has_tsc_adjust_msr),
        );
        features.set(
            Self::EXTENDED_APIC_SPACE,
            extended
                .as_ref()
                .is_some_and(ExtendedProcessorFeatureIdentifiers::has_ext_apic_space),
        );
        features
    }
}

/// What this processor can do.
#[must_use]
pub fn features() -> Features {
    *FEATURES.call_once(Features::read)
}

/// How many bits of a physical address this processor implements.
///
/// The width every "must be zero" rule about a physical address is stated
/// against. A control register or a table pointer with a bit set at or above
/// this is not merely pointing at memory that is not there — writing one
/// faults, and handing one to the virtualization extension makes entering a
/// guest fail with a code that says nothing about which field was wrong.
///
/// The leaf reporting it is absent on processors old enough that the answer
/// could only have been 36, which is what that case reports.
#[must_use]
pub fn physical_address_bits() -> u8 {
    *PHYSICAL_ADDRESS_BITS.call_once(|| {
        CpuId::new()
            .get_processor_capacity_feature_info()
            .as_ref()
            .map_or(
                LEGACY_PHYSICAL_ADDRESS_BITS,
                ProcessorCapacityAndFeatureInfo::physical_address_bits,
            )
    })
}

/// The four registers returned by one raw `CPUID` query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuidResult {
    /// Accumulator result.
    pub eax: u32,
    /// Base-register result.
    pub ebx: u32,
    /// Count-register result.
    pub ecx: u32,
    /// Data-register result.
    pub edx: u32,
}

/// Executes the raw CPUID leaf and subleaf requested by a guest.
///
/// Feature discovery inside the hypervisor uses the cached typed queries
/// above. This raw form exists for virtualization, where the guest chooses the
/// leaf and policy edits the returned words before exposing them.
#[must_use]
pub fn cpuid(leaf: u32, subleaf: u32) -> CpuidResult {
    let result = cpuid_count(leaf, subleaf);
    CpuidResult {
        eax: result.eax,
        ebx: result.ebx,
        ecx: result.ecx,
        edx: result.edx,
    }
}

/// What a processor whose `CPUID` does not describe its address widths
/// implements, which is the width the extension defined before the leaf
/// reporting it existed.
const LEGACY_PHYSICAL_ADDRESS_BITS: u8 = 36;

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

/// How wide this processor's physical addresses are, read on first use.
static PHYSICAL_ADDRESS_BITS: Once<u8> = Once::new();
