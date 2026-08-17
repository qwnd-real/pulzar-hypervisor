//! The local controller the guest is told it has.
//!
//! A guest works out what its controller can do from three places, and a
//! hypervisor that lets them disagree has handed it a machine that does not
//! exist. `CPUID` says which optional features the processor implements; the
//! version register says which version of the controller it is and how many
//! local vector table entries it has; and the registers themselves say what may
//! be written into them. This module is the one place those three are decided
//! together, so that every later question — is this entry present, may this
//! mode be entered, what does this reserved field fault on — is answered from
//! one description rather than from a constant somewhere.
//!
//! # Why it is derived rather than chosen
//!
//! Pulzar passes `CPUID` through. A guest therefore reads the real vendor,
//! family and model of the processor it is running on, and every optional
//! feature that processor reports. Inventing a controller that contradicts that
//! would be inventing a processor: a controller offering a
//! corrected-machine-check entry on hardware whose own controller has none and
//! would refuse to be programmed for it, or reporting a version its own entry
//! count contradicts.
//!
//! So the model is built from the machine — the vendor out of `CPUID`, the
//! version and the entry count out of one read of the real controller's version
//! register reconciled with the machine-check capability the same guest reads,
//! the optional interfaces out of the reported features — and every processor's
//! controller carries a copy of it.
//!
//! Both halves of the version register come from the same read, and that is not
//! only tidiness. A controller reporting a machine-derived entry count beside
//! an invented version number describes a part that never shipped, and software
//! that keys off the version — the boundary the architecture puts between the
//! discrete controller and this one is a version number, not a feature bit —
//! draws its conclusions about the count from it.
//!
//! # Whose manual this follows, and why there is no vendor dispatch
//!
//! AMD's. Pulzar is an SVM hypervisor, and SVM exists on AMD and on Hygon,
//! whose parts are AMD-derived and follow AMD's manual; no other vendor
//! implements it, so no other vendor's machine can execute this code. The two
//! places the vendors' controllers genuinely differ are therefore settled
//! rather than dispatched, and both are recorded here because the code no
//! longer shows them:
//!
//! - The arbitration priority is computed differently. AMD's is the greatest of
//!   the task, in-service and request priorities, keeping the task subclass
//!   when the task priority is what wins; Intel's P6 definition combines the
//!   task and in-service classes with a bitwise AND instead. The two disagree
//!   whenever those classes share no bits, which is most of the time.
//!   [`crate::priority`] implements AMD's, unconditionally.
//! - The error entry has a message-type field in bits 10:8 on AMD, where the
//!   whole of 11:8 is reserved on Intel. So a guest may program the mode of its
//!   own error interrupt here, and a vector left in that entry is an illegal
//!   one only when the mode is fixed.
//!
//! [`Vendor`] is still read, and is reported rather than branched on: a machine
//! that names itself something else is one where the controller being presented
//! and the processor the guest reads are described by different manuals, and
//! that is a fact worth having in a log rather than a branch worth taking.
//!
//! What the vendors do *not* disagree about, as far as anything here is
//! concerned, is which processor a redirectable interrupt goes to. That was the
//! chipset's choice rather than the processor's, neither vendor specifies it,
//! and [`crate::delivery`] makes it without asking the model.

use core::fmt::{self, Display, Formatter};

use apic::LocalApic;
use log::{info, warn};
use processor::Features;

use crate::registers::lvt::Entry;

/// The controller a guest is given, as everything above this module sees it.
///
/// Copied into every processor's controller rather than reached through a
/// global, because it is small, it never changes after installation, and the
/// paths that ask it questions are the paths a guest's every register access
/// takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Model {
    vendor: Vendor,
    version: u32,
    entries: usize,
    x2apic: bool,
    deadline: bool,
}

impl Model {
    /// The model this machine's own controller and processor describe.
    ///
    /// `local` is the controller of the processor this runs on, which is the
    /// boot processor's: [`crate::install`] has already established that it can
    /// be reached, and taking it as a parameter is what leaves this with no
    /// answer to invent for a controller that cannot be asked. That the whole
    /// machine is described by the one it reads is an assumption, and the one
    /// every part this hypervisor can run on satisfies — the version and the
    /// entry count are fixed at reset and uniform across a package, exactly as
    /// the features [`processor::features`] caches from one processor are.
    ///
    /// The version and the entry count come from one read of the version
    /// register, so the two halves of what a guest reads back cannot describe
    /// different controllers. The entry count is the real controller's because
    /// the sources behind those entries are the real ones: an entry the
    /// hardware does not have is an entry nothing could deliver from, so
    /// offering it to a guest would be offering a source that can never
    /// fire. It is then reconciled with the machine's machine-check
    /// capability, which is the other half of the same fact; see
    /// [`corrected_machine_check_entries`].
    pub(crate) fn of_machine(local: LocalApic) -> Self {
        let features = processor::features();
        let version = local.version();
        let entries = apic::lvt_entries(version) as usize;
        Self {
            vendor: Vendor::of_machine(),
            version: apic::version_number(version),
            entries: corrected_machine_check_entries(entries.clamp(Entry::FEWEST, Entry::COUNT)),
            x2apic: features.contains(Features::X2APIC),
            deadline: features.contains(Features::TSC_DEADLINE),
        }
    }

    /// Whether this controller has an entry at all.
    ///
    /// A controller has exactly the first however-many of [`Entry::ALL`], which
    /// is the order the architecture counts them in and is why that order is
    /// not the order the registers sit at.
    pub(crate) const fn has(self, entry: Entry) -> bool {
        entry.index() < self.entries
    }

    /// What the version register reports: the controller's own version in the
    /// low byte, and one less than its number of entries in the third.
    ///
    /// One less so that a controller always has at least one entry and the
    /// field cannot wrap, which is why [`Model::of_machine`] clamps the count
    /// to at least the fewest any implementation has reported.
    ///
    /// The bit that says the extended register space is present is deliberately
    /// not reported: the low byte is taken out of the machine's word on its
    /// own, so the machine's own answer to that question cannot leak through
    /// here. The space is the host's — it is where a withheld acknowledgement
    /// is settled by name — and a guest that found it announced would be a
    /// guest that could reach those registers.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the entry count is clamped to the seven the architecture defines, so one less than it fits the byte the field occupies"
    )]
    pub(crate) const fn version(self) -> u32 {
        ((self.entries - 1) as u32) << MAX_LVT_SHIFT | self.version
    }

    /// Whether the guest may put its controller into x2APIC mode.
    ///
    /// Gated on the real processor because `CPUID` is passed through: a guest
    /// told the feature is absent and then allowed to enter the mode anyway
    /// would be one whose own feature test says nothing.
    pub(crate) const fn x2apic(self) -> bool {
        self.x2apic
    }

    /// Whether the guest's timer has the timestamp-counter deadline mode.
    pub(crate) const fn deadline(self) -> bool {
        self.deadline
    }
}

impl Display for Model {
    /// Everything the model decided, because which of these a machine reports
    /// is what explains the paths this crate takes on it — and the one line
    /// a misbehaving part is diagnosed from.
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} version {:#x}, {} lvt entries, {} x2apic, {} deadline timer",
            self.vendor,
            self.version,
            self.entries,
            if self.x2apic { "with" } else { "without" },
            if self.deadline { "with" } else { "without" },
        )
    }
}

/// Whose architecture the processor behind this controller names itself with.
///
/// Read out of the same `CPUID` leaf the guest reads, and reported rather than
/// acted on. Everything this crate presents follows AMD's manual, because SVM
/// is AMD's and nothing else can execute this code — so what this answers is
/// not which rules to apply but whether the machine is one those rules
/// describe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Vendor {
    /// `AuthenticAMD`.
    Amd,
    /// `HygonGenuine`: AMD-derived parts that implement SVM and follow AMD's
    /// manual.
    Hygon,
    /// `GenuineIntel`, which implements no SVM and so cannot be running this.
    Intel,
    /// Anything else, which is what a processor whose vendor string has been
    /// rewritten underneath this hypervisor reports.
    Unknown,
}

impl Vendor {
    /// The vendor this processor names itself with.
    fn of_machine() -> Self {
        let leaf = processor::cpuid(VENDOR_LEAF, 0);
        // The twelve characters are spread across three registers in an order
        // that is not the order they are read in, which is the whole reason this
        // is written out rather than compared as a number.
        match [leaf.ebx, leaf.edx, leaf.ecx] {
            AMD => Self::Amd,
            HYGON => Self::Hygon,
            INTEL => Self::Intel,
            _ => Self::Unknown,
        }
    }

    /// Whether this is a vendor whose controller AMD's manual describes, which
    /// is the manual everything here follows.
    const fn amd_family(self) -> bool {
        matches!(self, Self::Amd | Self::Hygon)
    }
}

impl Display for Vendor {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Amd => "amd",
            Self::Hygon => "hygon",
            Self::Intel => "intel",
            Self::Unknown => "an unrecognised vendor",
        })
    }
}

/// Says what was decided, and says so on the machine where it is not the whole
/// truth.
///
/// The one line a part that misbehaves is diagnosed from, in a codebase where
/// every other subsystem describes itself at boot: from the count of
/// controllers alone there is no telling which controller the guest was given.
pub(crate) fn describe(who: &str, model: Model) {
    info!("{who}: presenting {model}");
    if !model.vendor.amd_family() {
        warn!(
            "{who}: this processor names itself {} and the controller its guest is being given \
             follows AMD's manual, which is the only one this hypervisor implements — the two \
             disagree about the arbitration priority and about the error entry's message type",
            model.vendor
        );
    }
}

/// The entry count this controller and the machine's machine-check reporting///
/// both stand behind.
///
/// The corrected-machine-check entry is the last of the seven the architecture
/// counts, so it is the only one two independent capabilities describe: the
/// controller's version register says whether the register is there, and
/// `IA32_MCG_CAP` says whether corrected errors are reported by an interrupt at
/// all. A guest reads both — the version register through its emulated
/// controller, the capability straight off the machine, since that register is
/// not intercepted — and a hypervisor that let the two disagree would hand it a
/// machine that does not exist.
///
/// So the entry is offered only when both say so, and the count is the whole of
/// what has to change for that: a controller has exactly the first
/// however-many of [`Entry::ALL`] and this one is last in the list.
///
/// The other direction cannot be reconciled from here and is reported instead.
/// A machine whose capability names the interrupt while its controller has no
/// entry for it is one where the register genuinely is not there, and a guest
/// that consults the capability and writes the entry finds no such register —
/// through the model-specific registers, a general protection fault. Nothing
/// here can invent the entry, because the source behind it is the real one.
fn corrected_machine_check_entries(entries: usize) -> usize {
    let last = Entry::CorrectedMachineCheck.index();
    match (last < entries, corrected_machine_check()) {
        (true, false) => {
            warn!(
                "vlapic: this controller has a {:?} entry and the machine does not report the \
                 interrupt it would deliver, so a guest is told it has {last} entries rather than \
                 {entries}",
                Entry::CorrectedMachineCheck
            );
            last
        }
        (false, true) => {
            warn!(
                "vlapic: the machine reports corrected machine-check interrupts and this \
                 controller has no {:?} entry, so a guest that programs one finds no such register",
                Entry::CorrectedMachineCheck
            );
            entries
        }
        _ => entries,
    }
}

/// Whether the machine reports corrected machine-check errors by an interrupt,
/// which is what the entry named after it exists to deliver.
///
/// Asked of the same register the guest asks, and asked through [`probe`]
/// because a machine without machine-check reporting has no such register and
/// faulting is the only way a processor answers whether an index is one. A
/// refused read is therefore the same answer as a clear bit: nothing to
/// interrupt about.
///
/// Requires the general protection vector to have been claimed, which
/// [`probe::install`] does during bring-up and before any controller is built.
fn corrected_machine_check() -> bool {
    probe::read(MACHINE_CHECK_CAPABILITY)
        .is_ok_and(|capability| capability & CORRECTED_INTERRUPT != 0)
}

/// `IA32_MCG_CAP`, which describes what the machine's machine-check reporting
/// can do.
const MACHINE_CHECK_CAPABILITY: u32 = 0x179;

/// The bit of it that says corrected errors raise an interrupt through the
/// local controller rather than only accumulating to be polled.
const CORRECTED_INTERRUPT: u64 = 1 << 10;

/// The `CPUID` leaf whose three registers spell the vendor.
const VENDOR_LEAF: u32 = 0;

/// `AuthenticAMD`, as the three registers hold it.
const AMD: [u32; 3] = [0x6874_7541, 0x6974_6E65, 0x444D_4163];

/// `HygonGenuine`.
const HYGON: [u32; 3] = [0x6F67_7948, 0x6E65_476E, 0x656E_6975];

/// `GenuineIntel`.
const INTEL: [u32; 3] = [0x756E_6547, 0x4965_6E69, 0x6C65_746E];

/// Bits the local-vector-table entry count is shifted by in the version
/// register, which is where the guest reads it beside the version itself.
const MAX_LVT_SHIFT: u32 = 16;

#[cfg(test)]
pub(crate) mod tests {
    //! Models are written out rather than derived from the machine the test
    //! runs on, which is not the guest's machine and is not necessarily a
    //! machine this hypervisor could boot on at all.
    //!
    //! [`Vendor::of_machine`] is the exception, and is checked against the
    //! strings themselves: it is pure data, it is the one thing here that
    //! silently answers about the wrong machine if a digit is wrong, and it
    //! cannot be exercised by running on a processor because a test runs on
    //! whatever processor it runs on.

    use alloc::{format, string::String};

    use super::{Model, Vendor};
    use crate::registers::lvt::Entry;

    /// A controller with every entry, on a processor implementing both optional
    /// interfaces.
    pub(crate) const AMD: Model = Model {
        vendor: Vendor::Amd,
        version: 0x14,
        entries: Entry::COUNT,
        x2apic: true,
        deadline: true,
    };

    /// The same, on a machine that names itself Intel — which no machine this
    /// hypervisor can boot on does, and which therefore changes nothing about
    /// what the model decides.
    pub(crate) const INTEL: Model = Model {
        vendor: Vendor::Intel,
        version: 0x14,
        entries: Entry::COUNT,
        x2apic: true,
        deadline: true,
    };

    /// A controller with only the entries every one of them has, on a processor
    /// with neither optional interface.
    pub(crate) const SPARSE: Model = Model {
        vendor: Vendor::Amd,
        version: 0x10,
        entries: 4,
        x2apic: false,
        deadline: false,
    };

    /// The twelve characters of a vendor string, in the register order the
    /// architecture spreads them across.
    fn spelled(registers: [u32; 3]) -> String {
        registers
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .map(char::from)
            .collect()
    }

    #[test]
    fn the_vendor_registers_spell_the_strings_in_the_order_cpuid_returns_them() {
        // The three registers are `ebx`, `edx`, `ecx` — not the order they are
        // read in — and each holds four characters little-endian. Every
        // vendor-dependent statement in this crate was decided from these twelve
        // bytes, so a wrong digit is a machine described by the wrong manual with
        // nothing to notice it.
        assert_eq!(spelled(super::AMD), "AuthenticAMD");
        assert_eq!(spelled(super::HYGON), "HygonGenuine");
        assert_eq!(spelled(super::INTEL), "GenuineIntel");
    }

    #[test]
    fn the_three_vendor_strings_are_distinct_and_amd_is_the_default() {
        // Which is the whole of `Vendor::of_machine`: three comparisons and a
        // fall-through. The fall-through has to be an unrecognised vendor rather
        // than a named one, because running at all means the processor implements
        // SVM — so an unrecognised string is a machine whose vendor was rewritten
        // underneath this hypervisor, not a machine following another manual.
        assert_ne!(super::AMD, super::HYGON);
        assert_ne!(super::AMD, super::INTEL);
        assert_ne!(super::HYGON, super::INTEL);
        assert!(Vendor::Amd.amd_family() && Vendor::Hygon.amd_family());
        assert!(!Vendor::Intel.amd_family() && !Vendor::Unknown.amd_family());
    }

    #[test]
    fn the_version_register_carries_both_halves_of_what_the_machine_reported() {
        // The count in the third byte and the version in the low one, out of one
        // read of the real register — a machine-derived count beside an invented
        // version describes a part that never shipped.
        assert_eq!(AMD.version(), 0x0006_0014);
        assert_eq!(SPARSE.version(), 0x0003_0010);
    }

    #[test]
    fn a_controller_has_exactly_the_first_of_the_entries_it_counts() {
        // Four entries is the timer, the two pins and the error entry, which is
        // the order the architecture counts them in and not the order they sit
        // at in the page.
        for entry in [Entry::Timer, Entry::Lint0, Entry::Lint1, Entry::Error] {
            assert!(SPARSE.has(entry), "{entry:?}");
        }
        for entry in [
            Entry::Performance,
            Entry::Thermal,
            Entry::CorrectedMachineCheck,
        ] {
            assert!(!SPARSE.has(entry), "{entry:?}");
            assert!(AMD.has(entry), "{entry:?}");
        }
    }

    #[test]
    fn the_optional_interfaces_follow_the_processor() {
        assert!(AMD.x2apic() && AMD.deadline());
        assert!(!SPARSE.x2apic() && !SPARSE.deadline());
    }

    #[test]
    fn the_model_says_what_it_decided() {
        // The one line a misbehaving part is diagnosed from on a machine that has
        // a serial port at all, and the fact the count of controllers alone does
        // not carry.
        assert_eq!(
            format!("{AMD}"),
            "amd version 0x14, 7 lvt entries, with x2apic, with deadline timer"
        );
        assert_eq!(
            format!("{SPARSE}"),
            "amd version 0x10, 4 lvt entries, without x2apic, without deadline timer"
        );
    }

    #[test]
    fn nothing_the_model_decides_depends_on_the_vendor() {
        // The decision this crate makes about vendors, as a test: the controller
        // presented is AMD's whatever the processor calls itself, because nothing
        // that calls itself anything else implements the extension this
        // hypervisor is built on. Two models alike but for the vendor therefore
        // answer every question identically — and the moment one of them stops,
        // there is a second architecture description in the crate that no machine
        // can select and nothing exercises.
        assert_ne!(AMD.vendor, INTEL.vendor);
        assert_eq!(AMD.version(), INTEL.version());
        assert_eq!(AMD.x2apic(), INTEL.x2apic());
        assert_eq!(AMD.deadline(), INTEL.deadline());
        for entry in Entry::ALL {
            assert_eq!(AMD.has(entry), INTEL.has(entry), "{entry:?}");
        }
    }

    #[test]
    fn the_corrected_machine_check_entry_needs_the_machine_to_report_the_interrupt() {
        // Both halves of one fact: the count is what carries the entry, so the
        // reconciliation is a count and the entry is the last of them.
        assert_eq!(Entry::CorrectedMachineCheck.index(), Entry::COUNT - 1);
        let with = Model {
            entries: Entry::COUNT,
            ..AMD
        };
        let without = Model {
            entries: Entry::COUNT - 1,
            ..AMD
        };
        assert!(with.has(Entry::CorrectedMachineCheck));
        assert!(!without.has(Entry::CorrectedMachineCheck));
        // And nothing else moves with it: the six entries below are the ones the
        // version register alone answers for.
        for entry in Entry::ALL
            .into_iter()
            .filter(|entry| *entry != Entry::CorrectedMachineCheck)
        {
            assert!(without.has(entry), "{entry:?}");
        }
    }
}
