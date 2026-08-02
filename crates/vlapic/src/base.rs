//! The register a guest changes its controller's interface with, and the state
//! machine every write to it has to satisfy.
//!
//! Three unrelated things share `IA32_APIC_BASE`: whether this processor is the
//! one the machine started on, where the memory-mapped register page sits, and
//! the two bits that between them choose the interface. The last of those is
//! what the rest of this crate reads. With `EN` clear the controller answers
//! nothing at all; with `EN` alone it answers through the page; with `EN` and
//! `EXTD` together it answers through model-specific registers instead, which
//! is the same register file reached by index, with wider identifiers and
//! different rules about what faults. A guest changing those two bits is a
//! guest changing which of the two faces every later access arrives through, so
//! the value here is state this crate keeps rather than a number it stores.
//!
//! # Which writes are refused
//!
//! Only four moves between those states exist. A disabled controller may be
//! switched on into the older interface; from there it may go on into x2APIC,
//! or back to disabled; and out of x2APIC the only way is disabled, which takes
//! clearing both bits in a single write. Everything else is a general
//! protection fault. The two worth naming are going straight from x2APIC back
//! to the older interface, which software attempting to reset a controller
//! reaches for, and going straight from disabled into x2APIC, which skips a
//! state the processor insists on passing through. `EXTD` set with `EN` clear
//! is not a state at all, and a write reaching for it faults wherever it is
//! written from.
//!
//! A write that names the state the register is already in is not a transition
//! and is always allowed: software that reads the register, changes a field it
//! is entitled to, and writes it back is doing nothing wrong.
//!
//! # The register page does not move, and that is part of the machine
//!
//! The address field is architecturally writable on real hardware and is
//! refused here, so this is a way in which the machine Pulzar presents is
//! narrower than the one its `CPUID` describes. It is stated rather than
//! hidden: the guest's controller lives at [`ApicBase::DEFAULT_PAGE`] for the
//! whole life of the guest, and a write that would move it takes a general
//! protection fault.
//!
//! The reason is that the page is trapped in the nested page tables once,
//! before any guest has run, and nothing in this codebase can re-trap a range
//! while processors are executing. A guest whose write was accepted would go on
//! faulting on the old address and reading plain memory at the new one — a
//! controller that silently stopped working — so refusing is the honest answer
//! and [`BaseFault::Relocated`] is kept distinct from the architectural faults
//! so that a caller can tell the two apart in a log.
//!
//! Firmware and operating systems do not relocate the page in practice; the
//! default address is what every one of them expects to find.

use core::fmt::{self, Display, Formatter};

use thiserror::Error;

use crate::model::Model;

/// Which interface the guest's controller answers through, which is the whole
/// of what the two enable bits mean.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    /// Switched off: the register page decodes to nothing and the
    /// model-specific registers are not there.
    Disabled,
    /// The page of memory-mapped registers, with eight-bit identifiers.
    XApic,
    /// Model-specific registers, with 32-bit identifiers.
    X2Apic,
}

impl Mode {
    /// What to call this mode in a log line.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::XApic => "xapic",
            Self::X2Apic => "x2apic",
        }
    }
}

impl Display for Mode {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// The guest's `IA32_APIC_BASE`, as this crate keeps it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ApicBase(u64);

impl ApicBase {
    /// The index a guest reaches this register by.
    pub(crate) const MSR: u32 = 0x1B;

    /// Where the register page sits at reset, and — because this hypervisor
    /// refuses to move it — for as long as the guest runs.
    pub(crate) const DEFAULT_PAGE: u64 = 0xFEE0_0000;

    /// The value a processor comes out of reset holding.
    ///
    /// Enabled, in the older interface, at the default page: a guest that never
    /// touches this register still has a working controller, and firmware that
    /// only ever reads it finds what it expects. `bootstrap` is set for the one
    /// processor the guest starts on and for no other, and no later write can
    /// change that.
    pub(crate) fn reset(bootstrap: bool) -> Self {
        let flag = if bootstrap { BOOTSTRAP } else { 0 };
        Self(Self::DEFAULT_PAGE | GLOBAL_ENABLE | flag)
    }

    /// The register holding exactly these bits.
    pub(crate) const fn from_bits(bits: u64) -> Self {
        Self(bits)
    }

    /// What a guest reading the register gets.
    pub(crate) const fn bits(self) -> u64 {
        self.0
    }

    /// Which interface the controller currently answers through.
    ///
    /// The global enable decides on its own: with it clear the controller is
    /// off whatever `EXTD` says. That combination is not a state a guest
    /// can write — [`written`](Self::written) refuses it — so it is
    /// reachable only by handing [`from_bits`](Self::from_bits) a value
    /// that never came from a guest.
    pub(crate) const fn mode(self) -> Mode {
        if self.0 & GLOBAL_ENABLE == 0 {
            Mode::Disabled
        } else if self.0 & X2APIC_ENABLE == 0 {
            Mode::XApic
        } else {
            Mode::X2Apic
        }
    }

    /// The whole of the architectural address field, however wide this
    /// processor implements it.
    ///
    /// Not the same question as [`ApicBase::page`], and the difference is what
    /// a relocation check has to be made against. The reserved-bit test
    /// above is derived from the processor's own physical-address width,
    /// which on some processors is wider than the bits 51:12 the page mask
    /// keeps — so a write setting an address bit above the mask would pass
    /// the reserved test, disappear in the mask, and compare equal to where
    /// the page already is. The guest would then have been told its page
    /// moved while this hypervisor went on trapping the old one.
    const fn address(self) -> u64 {
        self.0 & !(RESERVED_LOW | BOOTSTRAP | X2APIC_ENABLE | GLOBAL_ENABLE)
    }

    /// Whether this is the processor the guest was started on.
    pub(crate) const fn bootstrap(self) -> bool {
        self.0 & BOOTSTRAP != 0
    }

    /// What this register becomes when the guest writes `value`, or why it may
    /// not.
    ///
    /// The checks run in the order the guest would meet them on real hardware:
    /// reserved bits first, then the transition, and only then this
    /// hypervisor's own refusal to let the page move. A write that is both
    /// architecturally illegal and moves the page is reported as the fault the
    /// processor would have raised, because that is the one the guest has to be
    /// told about.
    ///
    /// # Errors
    ///
    /// [`BaseFault::Reserved`] if any bit the architecture reserves was written
    /// non-zero, [`BaseFault::IllegalTransition`] if the two enable bits name a
    /// state this one cannot go to, or [`BaseFault::Relocated`] if the address
    /// field changed.
    pub(crate) fn written(self, value: u64, model: Model) -> Result<Self, BaseFault> {
        if value & reserved() != 0 {
            return Err(BaseFault::Reserved);
        }

        // The bootstrap flag records which processor the machine came up on.
        // Software cannot make a processor into that one, so the written bit is
        // dropped and the one already here carried through.
        let next = Self((value & !BOOTSTRAP) | (self.0 & BOOTSTRAP));

        // Checked against the raw value rather than against `next.mode()`,
        // which reports a controller with `EXTD` set and `EN` clear as merely
        // disabled and would let this through as a legal move.
        if value & X2APIC_ENABLE != 0 && value & GLOBAL_ENABLE == 0 {
            return Err(BaseFault::IllegalTransition);
        }
        // A guest whose `CPUID` says the processor has no x2APIC must not be
        // able to enter it. `CPUID` is passed through, so this is the real
        // processor's answer, and a guest allowed to enter a mode its own
        // feature test denies would be one whose feature tests mean nothing.
        if value & X2APIC_ENABLE != 0 && !model.x2apic() {
            return Err(BaseFault::Reserved);
        }
        if !permitted(self.mode(), next.mode()) {
            return Err(BaseFault::IllegalTransition);
        }
        // Compared across the whole architectural field rather than the page
        // this hypervisor traps, so that a guest cannot move the register page
        // by writing address bits above the ones the mask keeps.
        if next.address() != self.address() {
            return Err(BaseFault::Relocated);
        }
        Ok(next)
    }
}

/// Why a write to `IA32_APIC_BASE` did not take.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub(crate) enum BaseFault {
    /// A bit the architecture reserves was written non-zero: one of the eight
    /// below the flags, the one between the bootstrap flag and the enables, or
    /// one above the physical address space this processor implements.
    #[error("a reserved bit of the apic base register was written non-zero")]
    Reserved,
    /// The two enable bits name a state that cannot be reached from the one the
    /// controller is in, or a combination that is not a state at all.
    #[error("the apic base register cannot go to that mode from this one")]
    IllegalTransition,
    /// The write moved the memory-mapped register page.
    ///
    /// The page is trapped in the nested page tables once, before any guest has
    /// run, and nothing here can re-trap a range while processors are
    /// executing. Taking the write would leave the guest's controller at an
    /// address that is not intercepted and the interception on an address the
    /// guest no longer uses, which is a controller that silently stops working;
    /// refusing it keeps the two in agreement and gives the caller something it
    /// can report.
    #[error("the apic register page cannot be moved while the guest is running")]
    Relocated,
}

/// Whether the architecture allows a controller in `from` to be written into
/// `to`.
///
/// Staying put is allowed from everywhere. Of the moves that go somewhere, the
/// asymmetry is that x2APIC is entered only from the older interface and left
/// only into disabled: there is no edge from x2APIC back to xAPIC and none from
/// disabled straight into x2APIC.
const fn permitted(from: Mode, to: Mode) -> bool {
    matches!(
        (from, to),
        (Mode::Disabled, Mode::Disabled | Mode::XApic)
            | (Mode::XApic, Mode::XApic | Mode::X2Apic | Mode::Disabled)
            | (Mode::X2Apic, Mode::X2Apic | Mode::Disabled)
    )
}

/// Every bit a write must leave clear.
fn reserved() -> u64 {
    RESERVED_LOW | beyond_physical()
}

/// The bits at and above the width of a physical address on this processor.
///
/// The guest is given the machine's own width, so what it may put in the
/// address field is what the processor would have accepted. A processor
/// reporting a width of 64 or more would make the shift overflow, and a guest's
/// register write is not the place to discover that, so an unshiftable width
/// reserves nothing rather than panicking.
fn beyond_physical() -> u64 {
    u64::MAX
        .checked_shl(u32::from(processor::physical_address_bits()))
        .unwrap_or(0)
}

/// The reserved bits that sit below the address field: the low eight, and the
/// one between the bootstrap flag and the two enables.
const RESERVED_LOW: u64 = 0xFF | (1 << 9);

/// `BSP`: set on the processor the machine started on, and read-only to
/// software.
const BOOTSTRAP: u64 = 1 << 8;

/// `EXTD`: the controller answers through model-specific registers. Meaningless
/// without [`GLOBAL_ENABLE`], and writing it without that one faults.
const X2APIC_ENABLE: u64 = 1 << 10;

/// `EN`: the controller is switched on.
const GLOBAL_ENABLE: u64 = 1 << 11;

#[cfg(test)]
mod tests {
    //! What the reserved-bit check makes of the bits above the physical address
    //! space is not asserted here: that width comes from the `CPUID` of
    //! whatever processor the test runs on, and is not the guest's machine.

    use super::{ApicBase, BOOTSTRAP, BaseFault, GLOBAL_ENABLE, Mode, X2APIC_ENABLE};
    use crate::model::{self, Model};

    /// The model the transition tests use: one whose processor has x2APIC, so
    /// that entering it is refused for the state machine's reasons and never
    /// for the feature's.
    const MODEL: Model = model::tests::AMD;

    /// Every state, for the tests that have to try all of them.
    const STATES: [Mode; 3] = [Mode::Disabled, Mode::XApic, Mode::X2Apic];

    /// A written value naming a mode, with the page left where it belongs.
    fn bits(mode: Mode) -> u64 {
        let flags = match mode {
            Mode::Disabled => 0,
            Mode::XApic => GLOBAL_ENABLE,
            Mode::X2Apic => GLOBAL_ENABLE | X2APIC_ENABLE,
        };
        ApicBase::DEFAULT_PAGE | flags
    }

    /// A controller sitting in a mode.
    fn state(mode: Mode) -> ApicBase {
        ApicBase::from_bits(bits(mode))
    }

    #[test]
    fn reset_leaves_an_enabled_xapic() {
        let base = ApicBase::reset(true);
        assert_eq!(base.mode(), Mode::XApic);
        assert_eq!(base.bits() & ApicBase::DEFAULT_PAGE, ApicBase::DEFAULT_PAGE);
        assert!(base.bootstrap());
        assert!(!ApicBase::reset(false).bootstrap());
    }

    #[test]
    fn legal_transitions_are_taken() {
        for (from, to) in [
            (Mode::Disabled, Mode::XApic),
            (Mode::XApic, Mode::X2Apic),
            (Mode::XApic, Mode::Disabled),
            (Mode::X2Apic, Mode::Disabled),
        ] {
            assert_eq!(
                state(from).written(bits(to), MODEL).map(ApicBase::mode),
                Ok(to)
            );
        }
    }

    #[test]
    fn staying_put_is_not_a_transition() {
        for mode in STATES {
            assert_eq!(state(mode).written(bits(mode), MODEL), Ok(state(mode)));
        }
    }

    #[test]
    fn x2apic_is_left_only_through_disabled() {
        assert_eq!(
            state(Mode::X2Apic).written(bits(Mode::XApic), MODEL),
            Err(BaseFault::IllegalTransition)
        );
        assert_eq!(
            state(Mode::Disabled).written(bits(Mode::X2Apic), MODEL),
            Err(BaseFault::IllegalTransition)
        );
    }

    #[test]
    fn x2apic_without_the_global_enable_is_not_a_state() {
        let invalid = ApicBase::DEFAULT_PAGE | X2APIC_ENABLE;
        for mode in STATES {
            assert_eq!(
                state(mode).written(invalid, MODEL),
                Err(BaseFault::IllegalTransition)
            );
        }
    }

    #[test]
    fn the_bootstrap_flag_survives_a_write_clearing_it() {
        let written = ApicBase::reset(true)
            .written(bits(Mode::X2Apic), MODEL)
            .expect("entering x2apic from the reset state is a legal transition");
        assert_ne!(written.bits() & BOOTSTRAP, 0);
        assert_eq!(written.mode(), Mode::X2Apic);
    }

    #[test]
    fn moving_the_page_is_refused() {
        let elsewhere = (ApicBase::DEFAULT_PAGE + 0x1_0000) | GLOBAL_ENABLE;
        assert_eq!(
            state(Mode::XApic).written(elsewhere, MODEL),
            Err(BaseFault::Relocated)
        );
    }
}
