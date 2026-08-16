//! The local controller the guest is told it has.
//!
//! A guest works out what its controller can do from three places, and a
//! hypervisor that lets them disagree has handed it a machine that does not
//! exist. `CPUID` says which optional features the processor implements; the
//! version register says how many local vector table entries the controller
//! has; and the registers themselves say what may be written into them. This
//! module is the one place those three are decided together, so that every
//! later question — is this entry present, may this mode be entered, which
//! arbitration rule applies, what does this reserved field fault on — is
//! answered from one description rather than from a constant somewhere.
//!
//! # Why it is derived rather than chosen
//!
//! Pulzar passes `CPUID` through. A guest therefore reads the real vendor,
//! family and model of the processor it is running on, and every optional
//! feature that processor reports. Inventing a controller that contradicts that
//! would be inventing a processor: an AMD guest computing an Intel arbitration
//! priority, or a controller offering a corrected-machine-check entry on
//! hardware whose own controller has none and would refuse to be programmed for
//! it.
//!
//! So the model is built from the machine — the vendor out of `CPUID`, the
//! entry count out of the real controller's version register reconciled with
//! the machine-check capability the same guest reads, the optional interfaces
//! out of the reported features — and every processor's controller carries a
//! copy of it.
//!
//! # What the two vendors actually disagree about
//!
//! Less than the register layout suggests, and never cosmetically.
//!
//! The arbitration priority is computed differently, and the difference is
//! visible to a guest that reads the register. Intel's P6 definition combines
//! the task and in-service classes with a bitwise AND; AMD's is the maximum of
//! the task, in-service and request priorities, keeping the task subclass when
//! the task priority is what wins. The two disagree whenever the classes share
//! no bits, which is most of the time.
//!
//! The error entry has a message-type field on AMD and does not on Intel, where
//! its delivery is fixed by the architecture. A guest that programs one on a
//! machine reporting an AMD processor and reads back a fixed delivery has been
//! told its write did not happen.
//!
//! What they do *not* disagree about, as far as anything here is concerned, is
//! which processor a redirectable interrupt goes to. That was the chipset's
//! choice rather than the processor's, neither vendor specifies it, and
//! [`crate::delivery`] makes it without asking the model.

use apic::LocalApic;
use descriptors::Vector;
use log::warn;
use processor::Features;

use crate::{
    priority::{self, Priority},
    registers::lvt::{Delivery, Entry},
};

/// The controller a guest is given, as everything above this module sees it.
///
/// Copied into every processor's controller rather than reached through a
/// global, because it is small, it never changes after installation, and the
/// paths that ask it questions are the paths a guest's every register access
/// takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Model {
    vendor: Vendor,
    entries: usize,
    x2apic: bool,
    deadline: bool,
}

impl Model {
    /// The model this machine's own controller and processor describe.
    ///
    /// The entry count comes from the real controller because the sources
    /// behind those entries are the real ones: an entry the hardware does
    /// not have is an entry nothing could deliver from, so offering it to a
    /// guest would be offering a source that can never fire. A controller
    /// that cannot be asked — which happens only if this is called before
    /// the local controller is up — is taken to have the architectural
    /// minimum rather than assumed to have everything, since advertising an
    /// absent entry is the failure that silently misleads a guest and
    /// advertising too few merely offers it less.
    ///
    /// The count is then reconciled with the machine's machine-check
    /// capability, which is the other half of the same fact; see
    /// [`corrected_machine_check_entries`].
    pub(crate) fn of_machine() -> Self {
        let features = processor::features();
        let entries = apic::local()
            .map(LocalApic::entries)
            .map_or(Entry::FEWEST, |entries| {
                usize::try_from(entries).unwrap_or(Entry::FEWEST)
            })
            .clamp(Entry::FEWEST, Entry::COUNT);
        Self {
            vendor: Vendor::of_machine(),
            entries: corrected_machine_check_entries(entries),
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

    /// What the version register reports, which is one less than the number of
    /// entries.
    ///
    /// One less so that a controller always has at least one entry and the
    /// field cannot wrap, which is why [`Model::of_machine`] refuses to
    /// build a model with none.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the entry count is clamped to the seven the architecture defines, so one less than it fits the byte the field occupies"
    )]
    pub(crate) const fn max_lvt(self) -> u32 {
        (self.entries - 1) as u32
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

    /// Whether an entry accepts a delivery mode in this model.
    ///
    /// The two pins accept the modes the architecture defines for a wire an
    /// external controller signals over, which is all of them. The rest accept
    /// the modes a source the controller raises itself can meaningfully ask
    /// for, which excludes INIT and an external interrupt. The error entry
    /// is where the vendors part: it has no delivery field at all on Intel
    /// and a message type on AMD.
    ///
    /// The one exception to all of that is a system-management interrupt, which
    /// no entry accepts, and that is this hypervisor's own departure rather
    /// than either vendor's rule. These entries are programmed onto the
    /// machine's own controller, so a guest that chose that mode would take
    /// the *host* into system-management mode over host state, running
    /// firmware's handler against a context it was not written for — and
    /// the guest would see nothing of it either way. The interrupt command
    /// register refuses to send one for exactly the same reason, and the
    /// decision belongs in one place rather than in both.
    ///
    /// INIT and an external interrupt reach a pin's entry from here and are
    /// refused where the entry is turned into a physical one, because what is
    /// wrong with them is not the shape of the entry;
    /// [`crate::hardware::sources`] is where that is stated.
    pub(crate) const fn allows(self, entry: Entry, delivery: Delivery) -> bool {
        match delivery {
            Delivery::Fixed => true,
            Delivery::NonMaskable => self.has_delivery(entry),
            Delivery::SystemManagement => false,
            Delivery::Init | Delivery::External => matches!(entry, Entry::Lint0 | Entry::Lint1),
        }
    }

    /// Whether an entry has a delivery-mode field software may write.
    ///
    /// The timer never does: its delivery is fixed by the architecture on both
    /// vendors. The error entry does on AMD alone.
    pub(crate) const fn has_delivery(self, entry: Entry) -> bool {
        match entry {
            Entry::Timer => false,
            Entry::Error => matches!(self.vendor, Vendor::Amd),
            _ => true,
        }
    }

    /// The arbitration priority this model computes.
    ///
    /// Vestigial in the sense that no processor Pulzar runs on arbitrates over
    /// a bus, and not vestigial in the sense that matters: the register is
    /// readable, a guest that reads it gets a number, and the number has to be
    /// the one its processor would have produced.
    pub(crate) fn arbitration_priority(
        self,
        task: Priority,
        in_service: Option<Vector>,
        requested: Option<Vector>,
    ) -> Priority {
        match self.vendor {
            Vendor::Amd => priority::amd_arbitration(task, in_service, requested),
            Vendor::Intel => priority::intel_arbitration(task, in_service, requested),
        }
    }
}

/// Whose architecture the guest's controller follows.
///
/// Not a preference. It is read out of the same `CPUID` leaf the guest reads,
/// so that the controller and the processor it is part of agree about whose
/// manual describes them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Vendor {
    /// `AuthenticAMD`.
    Amd,
    /// `GenuineIntel`, and anything else: the Intel definitions are the ones
    /// every other implementation of this architecture followed.
    Intel,
}

impl Vendor {
    /// The vendor this processor names itself with.
    fn of_machine() -> Self {
        let leaf = processor::cpuid(VENDOR_LEAF, 0);
        // The twelve characters are spread across three registers in an order
        // that is not the order they are read in, which is the whole reason this
        // is written out rather than compared as a number.
        if [leaf.ebx, leaf.edx, leaf.ecx] == AMD {
            Self::Amd
        } else {
            Self::Intel
        }
    }
}

/// The entry count this controller and the machine's machine-check reporting
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

#[cfg(test)]
pub(crate) mod tests {
    //! Models are written out rather than derived from the machine the test
    //! runs on, which is not the guest's machine and is not necessarily either
    //! vendor.

    use super::{Model, Vendor};
    use crate::registers::lvt::{Delivery, Entry};

    /// A controller with every entry, on a processor implementing both optional
    /// interfaces, following AMD's rules.
    pub(crate) const AMD: Model = Model {
        vendor: Vendor::Amd,
        entries: Entry::COUNT,
        x2apic: true,
        deadline: true,
    };

    /// The same, following Intel's.
    pub(crate) const INTEL: Model = Model {
        vendor: Vendor::Intel,
        entries: Entry::COUNT,
        x2apic: true,
        deadline: true,
    };

    /// A controller with only the entries every one of them has, on a processor
    /// with neither optional interface.
    pub(crate) const SPARSE: Model = Model {
        vendor: Vendor::Amd,
        entries: 4,
        x2apic: false,
        deadline: false,
    };

    #[test]
    fn the_version_register_reports_one_less_than_the_count() {
        assert_eq!(AMD.max_lvt(), 6);
        assert_eq!(SPARSE.max_lvt(), 3);
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
    fn no_entry_delivers_a_system_management_interrupt() {
        // The one delivery mode refused everywhere, including in the two entries
        // whose architectural shape accepts it: taking it would put the host into
        // system-management mode over host state, and the interrupt command
        // register refuses to send one for the same reason.
        for model in [AMD, INTEL, SPARSE] {
            for entry in Entry::ALL {
                assert!(
                    !model.allows(entry, Delivery::SystemManagement),
                    "{entry:?}"
                );
            }
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
