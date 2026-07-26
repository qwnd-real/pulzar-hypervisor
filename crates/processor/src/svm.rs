//! What this processor's virtualization extension can do.
//!
//! Every machine that implements the extension implements a different subset of
//! it, and the differences are not cosmetic: whether the guest's own page
//! tables can be translated by a second set, whether the address of the
//! instruction after an intercepted one is supplied or must be decoded,
//! whether the processor caches parts of a guest's control block between
//! entries. A hypervisor that assumes any of these ends up either refusing
//! machines it could have run or running them wrongly.
//!
//! So the whole leaf is read, not just the parts something already acts on.
//! That is a deliberate departure from the rule the rest of this crate
//! follows: these bits describe the *shape of structures* the `svm` crate
//! defines, and a field that exists only on some processors is worth naming
//! next to the feature that gates it even before anything reads either.
//!
//! # Why this is separate from the other features
//!
//! [`crate::features`] answers what the processor can do at all. This answers
//! what it can do about virtualization, and the two are asked differently: the
//! whole leaf here is reserved on a processor without the extension, so it
//! cannot be read at all unless a different leaf says it exists. That is why
//! this is an [`Option`] and the other is not — absence is a real answer with
//! its own meaning, and it is the answer on every processor that is not AMD.

use bitflags::bitflags;
use log::info;
use raw_cpuid::{CpuId, ExtendedProcessorFeatureIdentifiers, cpuid};
use spin::Once;

/// The `CPUID` leaf describing the virtualization extension.
///
/// Reserved unless a processor reports the extension in the extended feature
/// identifiers, which is why nothing reads it without checking that first: on a
/// processor that does not implement it, whatever the leaf returns is not a
/// feature set but whatever the highest implemented leaf happens to answer.
const SVM_LEAF: u32 = 0x8000_000A;

bitflags! {
    /// What this processor's virtualization extension supports.
    ///
    /// The whole of the leaf's feature word, so that a machine can be described
    /// exactly rather than approximately.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct SvmFeatures: u32 {
        /// A guest's physical addresses are translated by a second set of page
        /// tables. Without this a hypervisor must maintain shadow page tables
        /// and intercept every change a guest makes to its own, which is both
        /// far slower and far more code.
        const NESTED_PAGING = 1 << 0;
        /// The processor maintains a guest's branch-record registers across
        /// entry and exit, so a guest can profile its own branches without
        /// every access being intercepted.
        const LBR_VIRTUALIZATION = 1 << 1;
        /// Enabling the extension can be locked off, and unlocked again only
        /// with a key. What matters to a hypervisor is the corollary: on a
        /// processor without it, the disable bit reading back set is how
        /// firmware says virtualization is off for good.
        const SVM_LOCK = 1 << 2;
        /// The address of the instruction after an intercepted one is supplied
        /// on exit. Without it a hypervisor must decode the guest's instruction
        /// to know how far to step it, and software interrupts cannot be
        /// injected correctly at all.
        const NEXT_RIP = 1 << 3;
        /// The guest's timestamp counter can be scaled by a ratio, not just
        /// offset — which is what lets a guest keep a consistent clock rate
        /// across machines whose cores run at different frequencies.
        const TSC_RATE_MSR = 1 << 4;
        /// The processor caches parts of a guest's control block between
        /// entries and consults the clean field to decide what to re-read. On a
        /// processor without it the clean field is neither read nor cached, so
        /// writing it is harmless.
        const VMCB_CLEAN = 1 << 5;
        /// Translations can be flushed for one address space rather than all of
        /// them, which is the difference between switching guests costing one
        /// address space's translations and costing the whole machine's.
        const FLUSH_BY_ASID = 1 << 6;
        /// The processor reports what an intercepted instruction was doing —
        /// the register it named, the address it touched — instead of leaving a
        /// hypervisor to fetch and decode the instruction itself.
        const DECODE_ASSISTS = 1 << 7;
        /// The guest's performance counters are maintained by the processor
        /// across entry and exit.
        const PMC_VIRTUALIZATION = 1 << 8;
        /// Repeated spinning in a guest can be intercepted after a threshold
        /// rather than on the first spin instruction, so that a lock held
        /// briefly does not cost an exit.
        const PAUSE_FILTER = 1 << 10;
        /// The spin filter also takes a cycle threshold, so that spins far
        /// enough apart in time are not counted together.
        const PAUSE_FILTER_THRESHOLD = 1 << 12;
        /// The processor can deliver interrupts to a guest's own interrupt
        /// controller in hardware, without an exit for each one.
        const AVIC = 1 << 13;
        /// A guest can save and restore processor state with the two
        /// instructions that do it without being intercepted, which is what
        /// makes running a hypervisor inside this one affordable.
        const VMSAVE_VIRTUALIZATION = 1 << 15;
        /// A guest gets its own global interrupt flag, so the instructions that
        /// clear and set it need not be intercepted.
        const VGIF = 1 << 16;
        /// A guest executing from a page its own tables call user memory can be
        /// trapped, which lets a hypervisor tell the two kinds of execution
        /// apart.
        const GUEST_MODE_EXECUTE_TRAP = 1 << 17;
        /// The guest's interrupt controller can be driven in hardware with
        /// 32-bit identifiers, which is what a guest with more than 255
        /// processors needs.
        const X2AVIC = 1 << 18;
        /// Which pages a guest may use for a supervisor shadow stack can be
        /// restricted in the second set of page tables.
        const SUPERVISOR_SHADOW_STACK = 1 << 19;
        /// The guest's speculation controls are maintained by the processor
        /// rather than intercepted.
        const SPEC_CTRL = 1 << 20;
        /// The guest's own page tables can be treated as read-only, so the
        /// processor does not write access and dirty bits into them.
        const READ_ONLY_GUEST_PAGE_TABLES = 1 << 21;
        /// A machine check raised in a guest whose own configuration would have
        /// shut it down is intercepted instead, so one guest's hardware fault
        /// does not take the machine with it.
        const HOST_MCE_OVERRIDE = 1 << 23;
        /// The broadcast invalidation instructions can be enabled for a guest
        /// and intercepted, rather than always raising an invalid opcode.
        const INVLPGB_TLBSYNC = 1 << 24;
        /// Non-maskable interrupt masking is virtualized, so a hypervisor need
        /// not track whether a guest is inside its handler by intercepting the
        /// return.
        const VNMI = 1 << 25;
        /// The guest's instruction-sampling state is maintained by the
        /// processor.
        const IBS_VIRTUALIZATION = 1 << 26;
        /// Writes to the interrupt controller's extended local vector entries
        /// are trapped after the fact rather than faulting.
        const EXT_LVT_OFFSET_FAULT_CHANGE = 1 << 27;
        /// The virtualization instructions exit before the processor checks
        /// their operand against reserved memory, which saves a hypervisor from
        /// intercepting a fault and emulating the instruction to find out what
        /// was meant.
        const SVME_ADDR_CHECK = 1 << 28;
        /// A guest taking too many bus locks can be intercepted, which is what
        /// stops one guest from starving the machine's memory bus.
        const BUS_LOCK_THRESHOLD = 1 << 29;
        /// A guest halting with no interrupt pending can be intercepted
        /// separately from an ordinary halt, so an idle guest can be descheduled
        /// without intercepting every halt it makes.
        const IDLE_HLT = 1 << 30;
    }
}

/// What this processor's virtualization extension is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Svm {
    /// Which revision of the extension this processor implements.
    pub revision: u8,
    /// How many address spaces translations can be tagged with. Identifier zero
    /// belongs to the host, so a guest may be given anything from one to this
    /// less one — and a machine reporting fewer than two can run no guest at
    /// all.
    pub asids: u32,
    /// What it supports.
    pub features: SvmFeatures,
}

impl Svm {
    /// Logs what the processor reported, present and absent alike.
    ///
    /// Which of these a machine lacks is what explains the paths a hypervisor
    /// takes on it, and several of them decide whether it can be run at all —
    /// so the absent ones are worth as many lines as the present ones.
    pub fn describe(&self, who: &str) {
        info!(
            "{who}: svm revision {}, {} address space identifiers",
            self.revision, self.asids
        );
        info!("{who}: svm has {:?}", self.features);
        let missing = SvmFeatures::all().difference(self.features);
        if !missing.is_empty() {
            info!("{who}: svm lacks {missing:?}");
        }
    }

    /// Asks `CPUID` what the extension is, or answers `None` on a processor
    /// that does not have one.
    ///
    /// The leaf is only read once the extended feature identifiers say the
    /// extension exists, because it is reserved otherwise — reading it anyway
    /// would return whatever the highest implemented leaf answers and decode
    /// that as a feature set.
    fn read() -> Option<Self> {
        let cpuid = CpuId::new();
        let supported = cpuid
            .get_extended_processor_and_feature_identifiers()
            .as_ref()
            .is_some_and(ExtendedProcessorFeatureIdentifiers::has_svm);
        if !supported {
            return None;
        }
        let leaf = cpuid!(SVM_LEAF);
        Some(Self {
            revision: revision(leaf.eax),
            asids: leaf.ebx,
            features: SvmFeatures::from_bits_truncate(leaf.edx),
        })
    }
}

/// The revision number out of the leaf's first word.
///
/// Masking to eight bits is what the field is, not a guard against a value
/// that might not fit, which is why this narrows without ceremony.
const fn revision(eax: u32) -> u8 {
    (eax & 0xFF) as u8
}

/// What this processor's virtualization extension can do, or `None` if it has
/// none.
///
/// Read on the first call and kept, which is sound for the same reason the
/// other feature answers are: this is fixed at reset and uniform across a
/// package.
#[must_use]
pub fn svm() -> Option<Svm> {
    *SVM.call_once(Svm::read)
}

/// The extension this image runs on, read on first use.
static SVM: Once<Option<Svm>> = Once::new();
