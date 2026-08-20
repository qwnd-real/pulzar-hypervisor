//! Whether the guest's interrupts stay the host's to deliver.
//!
//! The hardware can be given the guest's interrupt controller outright, but
//! whether it should is a question about this machine rather than about any
//! guest: does the extension exist at all, do the silicon's documented errata
//! leave any part of delivery trustworthy, do the machine's own identifiers fit
//! the tables the modes are built from, and is the face firmware left its
//! controller in one of the modes on offer. All of that is fixed at boot, so
//! the answer is taken once, here, where the roster, the processor's own
//! feature words and firmware's capture are all known — and it never changes
//! after.
//!
//! What changes at runtime — a processor descheduled, a table inhibited — is
//! state of the delivery path and belongs with it, not with this decision.

use cpu::Roster;
use log::info;
use processor::{MemoryEncryption, Svm};
use snapshot::FirmwareContext;
use spin::Once;
use svm::avic::MAX_PHYSICAL_ID;
use vcpu::AvicLimits;

/// How the guest's interrupts are delivered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AvicMode {
    /// The host delivers every interrupt itself, as it always has.
    SoftwareOnly,
    /// The hardware drives the guest's controller, addressing its processors
    /// with eight-bit identifiers.
    XAvic,
    /// The hardware drives the guest's controller, addressing its processors
    /// with 32-bit identifiers, reaching as far as this index.
    ///
    /// The limit travels with the mode because it is the mode's rather than the
    /// machine's — one page of table entries, or the eight the extended table
    /// may span — and everything downstream that has to know how far a table
    /// may be indexed is asking about the mode.
    X2AvicCapable {
        /// The highest table index this mode can name here.
        limit: u16,
    },
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
        self.x2avic_limit().is_some()
    }

    /// Whether a guest on this machine may be given the controller face its
    /// identifiers are reached through in model-specific registers.
    ///
    /// Two machines may offer it and one may not. A machine with no
    /// acceleration emulates that face as it emulates every other, so it is
    /// offered; a machine whose acceleration can drive it is offered it because
    /// the hardware will follow the guest into it. What is withheld is the
    /// middle case — an acceleration that exists and cannot drive that face —
    /// where a guest in it would be delivered to in software at exactly the
    /// moments it believed itself accelerated, and where the register page it
    /// would fall back on is a sink for the life of the machine.
    ///
    /// The one statement of it: the same answer decides the feature bit
    /// `CPUID` reports, the transitions a guest's own write of its base
    /// register may make, and the face a controller may be seeded into out of
    /// firmware's register.
    #[must_use]
    pub(crate) const fn x2apic_offered(self) -> bool {
        !self.enabled() || self.x2avic()
    }

    /// The highest table index the 32-bit face may name here, or `None` where
    /// the machine may not be driven in that face at all.
    ///
    /// Both halves of one answer, which is why it is one answer: a machine
    /// without the mode has no limit for it, and a limit without the mode is a
    /// number nothing may be judged against.
    #[must_use]
    pub(crate) const fn x2avic_limit(self) -> Option<u16> {
        match self.mode {
            AvicMode::X2AvicCapable { limit } => Some(limit),
            AvicMode::SoftwareOnly | AvicMode::XAvic => None,
        }
    }

    /// The highest table index delivery may name, which is the smaller of the
    /// mode's limit and the machine's highest identifier.
    ///
    /// The machine's answer rather than any one face's: the eight-bit face
    /// reaches less far than this wherever the machine has processors it cannot
    /// address, and that clamp belongs with the face rather than here, because
    /// a guest moves between the faces while it runs.
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
/// anything delivers. `firmware` is the capture the guest resumes out of, and
/// the face it left its controller in is one of the terms.
pub(crate) fn establish(roster: &Roster, firmware: &FirmwareContext) {
    let max_apic_id = roster
        .entries()
        .iter()
        .filter(|entry| entry.startable())
        .map(|entry| entry.apic_id().get())
        .max();
    let decided = decide(
        processor::svm().as_ref(),
        processor::identity().family(),
        max_apic_id,
        matches!(firmware.interrupts.mode(), Some(apic::Mode::X2Apic)),
    );
    POLICY.call_once(|| decided);
    // Logged from the cell rather than from the local, because the cell is what
    // the machine will use: a second call keeps the first caller's answer, and a
    // log line describing the value that lost would be the only record of the
    // decision saying the wrong thing.
    policy().describe();
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
/// be some; the reverse-map checks the encryption extension brings must leave
/// the host the writes maintaining delivery needs; the machine's identifiers
/// must fit the tables of whichever mode is chosen — which also keeps every
/// identifier inside the twelve bits a table entry names a physical processor
/// with; and the face the guest starts in must be one of the modes chosen.
fn decide(
    svm: Option<&Svm>,
    family: u8,
    max_apic_id: Option<u32>,
    firmware_x2apic: bool,
) -> AvicPolicy {
    let Some(svm) = svm else {
        return AvicPolicy::software_only();
    };
    let Some(max_apic_id) = max_apic_id else {
        return AvicPolicy::software_only();
    };
    // The one derivation of what this processor's delivery can do, shared with
    // the entry rules a control block is judged against, so that the tables are
    // built for the mode the same feature words will be read as later.
    let limits = AvicLimits::of(svm.features);
    if !limits.avic {
        return AvicPolicy::software_only();
    }
    // On a host running secure nested paging, entering a guest marks its
    // backing page in-use for the duration — for every guest, encrypted or not
    // — and a host write to such a page is a reverse-map violation unless the
    // extension says otherwise. One of the writes this subsystem makes is
    // exactly that: a request bit set by another processor while the target is
    // inside its guest.
    //
    // The term the rule wants is whether the extension is *switched on*, which
    // lives in a system register this hypervisor never writes and a guest's own
    // write of it is forwarded to. This is the widest fact `CPUID` offers
    // instead — the silicon implements it — so the refusal is conservative: a
    // machine whose firmware left it off loses hardware delivery it could have
    // had, which costs speed where the other direction costs a fault the guest
    // did nothing to earn.
    if svm
        .encryption
        .contains(MemoryEncryption::SECURE_NESTED_PAGING)
        && !svm
            .encryption
            .contains(MemoryEncryption::HV_IN_USE_WRITES_ALLOWED)
    {
        return AvicPolicy::software_only();
    }
    // Erratum #1235 leaves delivery between a guest's own processors
    // untrustworthy on these families, whatever else it leaves alone.
    let ipi_virtual = family != ZEN_FAMILY && family != DHYANA_FAMILY;
    // Narrowed once, to the width a table is indexed in: an identifier that
    // does not fit that width is one no mode can name, and each mode's own
    // limit is what decides between them.
    let index = u16::try_from(max_apic_id).ok();
    if let Some(limit) = limits.x2avic
        && let Some(max_index) = index.filter(|id| *id <= limit)
    {
        return AvicPolicy {
            mode: AvicMode::X2AvicCapable { limit },
            ipi_virtual,
            max_index,
        };
    }
    // The guest is the firmware this hypervisor found, and it resumes with its
    // controller in the face firmware left it in. Only the older face is left to
    // offer here, and a machine that offers only that one withholds the wider
    // face's feature bit — from a guest that is already using it, and whose
    // controller would have to be moved out from under it to make the two agree.
    // Delivery stays the host's instead, where both faces are emulated and the
    // guest keeps the one it has.
    if firmware_x2apic {
        return AvicPolicy::software_only();
    }
    if let Some(max_index) = index.filter(|id| *id <= MAX_PHYSICAL_ID) {
        return AvicPolicy {
            mode: AvicMode::XAvic,
            ipi_virtual,
            max_index,
        };
    }
    AvicPolicy::software_only()
}

/// The family of AMD processors erratum #1235 afflicts.
const ZEN_FAMILY: u8 = 0x17;

/// The family of Hygon processors erratum #1235 afflicts.
const DHYANA_FAMILY: u8 = 0x18;

/// The decision, taken once at boot.
static POLICY: Once<AvicPolicy> = Once::new();
