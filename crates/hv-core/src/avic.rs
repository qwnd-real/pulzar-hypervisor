//! Whether the guest's interrupts stay the host's to deliver.
//!
//! The hardware can be given the guest's interrupt controller outright, but
//! whether it should is a question about this machine rather than about any
//! guest: does the extension exist at all, do the silicon's documented errata
//! leave any part of delivery trustworthy, and do the machine's own
//! identifiers fit the tables the modes are built from. All of that is fixed
//! at boot, so the answer is taken once, here, where the roster and the
//! processor's own feature words are both known — and it never changes after.
//!
//! What changes at runtime — a processor descheduled, a table inhibited — is
//! state of the delivery path and belongs with it, not with this decision.

use core::cmp::min;

use cpu::Roster;
use log::info;
use processor::{MemoryEncryption, Svm, SvmFeatures};
use spin::Once;
use svm::avic::{MAX_PHYSICAL_ID, X2_EXTENDED_MAX_PHYSICAL_ID, X2_MAX_PHYSICAL_ID};

/// How the guest's interrupts are delivered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AvicMode {
    /// The host delivers every interrupt itself, as it always has.
    SoftwareOnly,
    /// The hardware drives the guest's controller, addressing its processors
    /// with eight-bit identifiers.
    XAvic,
    /// The hardware drives the guest's controller, addressing its processors
    /// with 32-bit identifiers.
    X2AvicCapable,
}

/// The boot-time decision about hardware interrupt delivery.
///
/// Immutable after boot: everything that could change the answer is fixed at
/// reset, so nothing later in the machine's life gets to argue with it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AvicPolicy {
    mode: AvicMode,
    ipi_virtual: bool,
    max_index: u16,
}

impl AvicPolicy {
    /// The policy a machine gets when nothing argues for hardware delivery.
    const fn software_only() -> Self {
        Self {
            mode: AvicMode::SoftwareOnly,
            ipi_virtual: false,
            max_index: 0,
        }
    }

    /// Whether the hardware may be given the guest's controller at all.
    #[must_use]
    pub(crate) const fn enabled(self) -> bool {
        !matches!(self.mode, AvicMode::SoftwareOnly)
    }

    /// Whether the hardware may be trusted with interrupts between the
    /// guest's own processors, which erratum #1235 takes away on some
    /// silicon even where it leaves the rest of delivery alone.
    #[must_use]
    pub(crate) const fn ipi_virtual(self) -> bool {
        self.ipi_virtual
    }

    /// Whether delivery uses 32-bit identifiers.
    #[must_use]
    pub(crate) const fn x2avic(self) -> bool {
        matches!(self.mode, AvicMode::X2AvicCapable)
    }

    /// The highest table index delivery may name, which is the smaller of the
    /// mode's limit and the machine's highest identifier.
    #[must_use]
    pub(crate) const fn max_index(self) -> u16 {
        self.max_index
    }

    /// Logs the decision, which is what explains the paths interrupts take on
    /// this machine for the rest of its life.
    fn describe(self) {
        info!(
            "core: avic policy: enabled {}, x2avic {}, ipi virtualization {}, max index {:#x}",
            self.enabled(),
            self.x2avic(),
            self.ipi_virtual(),
            self.max_index()
        );
    }
}

/// Decides how this machine's guest takes its interrupts, stores the answer,
/// and says what it was.
///
/// Called once, on the boot processor, after the roster exists and before
/// anything delivers.
pub(crate) fn establish(roster: &Roster) {
    let max_apic_id = roster
        .entries()
        .iter()
        .filter(|entry| entry.startable())
        .map(|entry| entry.apic_id().get())
        .max();
    let policy = decide(
        processor::svm().as_ref(),
        processor::identity().family(),
        max_apic_id,
    );
    POLICY.call_once(|| policy);
    policy.describe();
}

/// The decision, once taken.
///
/// # Panics
///
/// Before [`establish`], which the boot sequence orders before anything that
/// asks.
pub(crate) fn policy() -> &'static AvicPolicy {
    POLICY
        .get()
        .expect("the delivery policy is decided at boot, before anything asks for it")
}

/// The decision itself, kept pure so it can be read against the rules.
///
/// In order: the extension must exist at all; at least one processor must be
/// startable, because delivery is for the guest's processors and there must
/// be some; secure delivery, where it exists, must allow the host the writes
/// maintaining it needs; and the machine's identifiers must fit the tables of
/// whichever mode is chosen — which also keeps every identifier inside the
/// twelve bits a table entry names a physical processor with.
fn decide(svm: Option<&Svm>, family: u8, max_apic_id: Option<u32>) -> AvicPolicy {
    let Some(svm) = svm else {
        return AvicPolicy::software_only();
    };
    let Some(max_apic_id) = max_apic_id else {
        return AvicPolicy::software_only();
    };
    if !svm.features.contains(SvmFeatures::AVIC) {
        return AvicPolicy::software_only();
    }
    if svm.encryption.contains(MemoryEncryption::SECURE_AVIC)
        && !svm
            .encryption
            .contains(MemoryEncryption::HV_IN_USE_WRITES_ALLOWED)
    {
        return AvicPolicy::software_only();
    }
    // Erratum #1235 leaves delivery between a guest's own processors
    // untrustworthy on these families, whatever else it leaves alone.
    let ipi_virtual = family != ZEN_FAMILY && family != DHYANA_FAMILY;
    if svm.features.contains(SvmFeatures::X2AVIC) {
        let limit = if svm.features.contains(SvmFeatures::X2AVIC_EXT) {
            u32::from(X2_EXTENDED_MAX_PHYSICAL_ID)
        } else {
            u32::from(X2_MAX_PHYSICAL_ID)
        };
        if max_apic_id <= limit {
            return AvicPolicy {
                mode: AvicMode::X2AvicCapable,
                ipi_virtual,
                max_index: index_within(max_apic_id, limit),
            };
        }
    }
    if max_apic_id <= u32::from(MAX_PHYSICAL_ID) {
        return AvicPolicy {
            mode: AvicMode::XAvic,
            ipi_virtual,
            max_index: index_within(max_apic_id, u32::from(MAX_PHYSICAL_ID)),
        };
    }
    AvicPolicy::software_only()
}

/// The table index for an identifier already established to fit the mode's
/// limit.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the narrowed value is at most the mode's limit, which fits in twelve bits"
)]
fn index_within(max_apic_id: u32, limit: u32) -> u16 {
    min(max_apic_id, limit) as u16
}

/// The family of AMD processors erratum #1235 afflicts.
const ZEN_FAMILY: u8 = 0x17;

/// The family of Hygon processors erratum #1235 afflicts.
const DHYANA_FAMILY: u8 = 0x18;

/// The decision, taken once at boot.
static POLICY: Once<AvicPolicy> = Once::new();
