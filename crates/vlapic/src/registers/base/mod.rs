//! The register a guest changes its controller's interface with, and the state
//! machine every write to it has to satisfy.
//!
//! Three unrelated things share `IA32_APIC_BASE`: whether this processor is the
//! one the machine started on, where the memory-mapped register page sits, and
//! the two bits that between them choose the interface. The last of those is
//! what the rest of this crate reads. With `EN` clear the controller is
//! switched off; with `EN` alone it answers through the page; with `EN` and
//! `EXTD` together it answers through model-specific registers instead, which
//! is the same register file reached by index, with wider identifiers and
//! different rules about what faults. A guest changing those two bits is a
//! guest changing which of the two faces every later access arrives through, so
//! the value here is state this crate keeps rather than a number it stores.
//!
//! # What a switched-off controller still takes
//!
//! Architecturally, nothing: the enable bit is what makes a controller accept
//! interrupt messages at all, and clearing it leaves the processor as though it
//! had no on-chip controller. Everything that goes through
//! [`Vlapic::accept`](crate::registers::Vlapic::accept) honours that.
//!
//! Three messages do not go through it, and this is a deliberate deviation
//! rather than an oversight: a non-maskable interrupt, an INIT and a start-up
//! message are delivered to the *processor* rather than into the register file,
//! and this crate delivers them whatever the mode. What they are refused for is
//! a *software* disable — the enable bit of the spurious-vector register —
//! which the architecture says leaves all three deliverable, so refusing them
//! there would be the error in the other direction.
//!
//! The reason for the deviation is that a processor whose guest switched its
//! controller off is still a processor that guest may reset and restart, and
//! the only way it can be restarted is one of those three messages. Software
//! that shuts a controller down on its way out — which is what an operating
//! system does before handing the machine to a new kernel — would otherwise
//! leave every processor it had shut down unstartable, with nothing in the
//! architecture saying it should be. What a guest can get out of the deviation
//! is a non-maskable interrupt in a state hardware would not have delivered one
//! in, which is a message it sent itself.
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
//! The address field is architecturally writable on real hardware and is not
//! writable here, so this is a way in which the machine Pulzar presents is
//! narrower than the one its `CPUID` describes. It is stated rather than
//! hidden: the guest's controller lives at [`ApicBase::DEFAULT_PAGE`] for the
//! whole life of the guest, and a write naming any other address leaves it
//! where it is — so a guest that reads the register back finds the address it
//! did not get.
//!
//! The reason is that the page is trapped in the nested page tables once,
//! before any guest has run, and nothing in this codebase can re-trap a range
//! while processors are executing. A guest whose write was honoured would go on
//! faulting on the old address and reading plain memory at the new one — a
//! controller that silently stopped working.
//!
//! Ignoring the field is the narrower of the two deviations available. Raising
//! a general protection fault instead would invent an architectural fault for a
//! write the architecture defines, and would kill the software most likely to
//! make one: enabling a controller by writing the enable bit together with the
//! address firmware's own tables reported is how that is normally written, and
//! an unexpected fault at that point in a boot is a triple fault and a machine
//! with no output. A guest that reads back what it wrote and checks is instead
//! told the truth, which is that its controller did not move.
//!
//! The same fact decides what happens on a machine whose *firmware* had moved
//! the page before pulzar ran, and there the answer cannot be to ignore a
//! write: there is no guest instruction to ignore. Everything outside the one
//! trapped page is an identity map of machine physical memory, so a guest on
//! such a machine would reach the *real* local APIC at firmware's address,
//! untrapped — able to mask the host's interrupts, acknowledge them, reset the
//! host's processors and rename them. So [`ApicBase::misplaced`] answers that
//! question before any guest exists and [`crate::install`] refuses the machine
//! outright.
//!
//! Firmware and operating systems do not relocate the page in practice; the
//! default address is what every one of them expects to find.

mod transition;

use core::fmt::{self, Display, Formatter};

use apic::Controller;

pub(crate) use crate::registers::base::transition::{BaseFault, Transition};

/// Which interface the guest's controller answers through, which is the whole
/// of what the two enable bits mean.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    /// Switched off: the register page decodes to nothing, the model-specific
    /// registers are not there, and nothing offered to the controller is
    /// accepted. The three messages delivered to the processor rather than to
    /// the register file are the documented exception; see this module's own
    /// summary.
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
    pub(crate) const fn reset(bootstrap: bool) -> Self {
        let flag = if bootstrap { BOOTSTRAP } else { 0 };
        Self(Self::DEFAULT_PAGE | GLOBAL_ENABLE | flag)
    }

    /// The register a controller firmware left holding `value` should start in.
    ///
    /// Two fields are firmware's and two are not. The enable bits are the whole
    /// point: firmware that had taken its controller into x2APIC has to resume
    /// into one that is still there, because everything it programmed elsewhere
    /// on the machine — every logical destination in an I/O controller or a
    /// device's message — is addressed the way that face addresses things.
    /// Everything the architecture reserves is dropped, so that a guest reading
    /// the register back sees what it promises.
    ///
    /// The wider enable is dropped for either of two reasons, and this is the
    /// one producer of this register that can be handed it set without a
    /// transition ever having been judged — so both belong here.
    ///
    /// It goes if the global one did not survive. `EXTD` without `EN` is not a
    /// state the architecture defines, and a controller seeded with it would
    /// answer through neither face while every recovery a guest could attempt —
    /// reading the register, setting the enable bit, writing it back — took a
    /// general protection fault for a transition out of a state that does not
    /// exist.
    ///
    /// It goes as well where `x2apic_permitted` is false, which is the machine
    /// having no way to serve a guest that face. Such a controller would read
    /// back `EN|EXTD` while its guest's own `CPUID` denied the feature, and
    /// software that believes `CPUID` never looks at this register — so it
    /// would drive the memory-mapped face against a controller that had
    /// told the rest of this crate it was in the other one, finding
    /// registers absent and read-only that the face it is using has.
    /// Starting in the older face instead is the face that `CPUID`
    /// describes and that every guest knows how to be in.
    ///
    /// That half is the total statement rather than the working one: a machine
    /// whose firmware really did leave this register in the wider face is one
    /// where hardware delivery is declined altogether, and declining it is what
    /// makes the face servable again. What this rules out is a controller in a
    /// face nothing on the machine will answer for — whoever decided that, and
    /// whatever they decided it from.
    ///
    /// The address is the default one rather than firmware's, and the two are
    /// the same address wherever this is reached: the page is trapped once,
    /// before any guest has run, so a machine whose firmware had put it
    /// anywhere else is refused by [`crate::install`] and no controller on it
    /// is ever seeded. Writing the constant rather than carrying the field
    /// through is what makes that a property of this type rather than of
    /// the caller.
    ///
    /// The bootstrap flag is not firmware's either, and for a different reason:
    /// it is the roster's, established when the controller was built, and no
    /// value read from anywhere can change which processor the machine came up
    /// on.
    pub(crate) const fn seeded(value: u64, bootstrap: bool, x2apic_permitted: bool) -> Self {
        let flag = if bootstrap { BOOTSTRAP } else { 0 };
        let enables = value & (GLOBAL_ENABLE | X2APIC_ENABLE);
        let wider = enables & GLOBAL_ENABLE != 0 && x2apic_permitted;
        let enables = if wider {
            enables
        } else {
            enables & GLOBAL_ENABLE
        };
        Self(enables | Self::DEFAULT_PAGE | flag)
    }

    /// Where firmware left the register page, if it is not where this
    /// hypervisor traps one.
    ///
    /// The question [`crate::install`] refuses a machine on, and it is asked of
    /// the capture rather than of hardware so that every way of failing to read
    /// the controller reaches the same decision. Four of the five say something
    /// about the address field: a controller that was read, one firmware had
    /// switched off, one whose page is outside the window the capture could
    /// reach — which is a page that was moved a long way — and one whose two
    /// mode bits name no state at all. The address field is meaningful in every
    /// one of those, and in each of them the real controller ends up decoding
    /// at that address once the host switches it on, whatever the guest is
    /// shown.
    ///
    /// The exception is a processor with no controller at all, where there is
    /// no register to have read and the captured value is a zero nobody
    /// wrote. Such a machine has already been refused — installing the real
    /// controllers needs one — so answering `None` here leaves that refusal
    /// where it belongs rather than reporting a page at address zero.
    pub(crate) const fn misplaced(controller: Controller, value: u64) -> Option<u64> {
        if matches!(controller, Controller::Absent) || !Self::relocated(value) {
            return None;
        }
        Some(Self::page_of(value))
    }

    /// Whether a value read out of real hardware puts the register page
    /// somewhere other than where this hypervisor traps one.
    pub(crate) const fn relocated(value: u64) -> bool {
        Self::page_of(value) != Self::DEFAULT_PAGE
    }

    /// Where a value read out of real hardware puts the register page.
    pub(crate) const fn page_of(value: u64) -> u64 {
        Self(value).address()
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
    /// off whatever `EXTD` says. That combination is not a state, and neither
    /// producer of this register can build one — [`written`](Self::written)
    /// refuses a guest's attempt and [`seeded`](Self::seeded) drops the wider
    /// bit out of firmware's value — so it is reachable only by handing
    /// [`from_bits`](Self::from_bits) a word that came from neither.
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
    /// Every bit that is not one of the flags, which is bits 63:12 —
    /// deliberately wider than the 4 KiB page the field's low bits would
    /// give. The reserved-bit test a write is judged against is derived
    /// from the processor's own physical-address width, which on some
    /// processors is wider than a page mask of bits 51:12 would keep, so an
    /// address bit above such a mask would pass the reserved test and then
    /// disappear in the mask — which for [`ApicBase::page_of`], the question
    /// a machine is refused on, would be firmware's page reported as the one
    /// this hypervisor traps.
    const fn address(self) -> u64 {
        self.0 & !(RESERVED_LOW | BOOTSTRAP | X2APIC_ENABLE | GLOBAL_ENABLE)
    }

    /// Whether this is the processor the guest was started on.
    pub(crate) const fn bootstrap(self) -> bool {
        self.0 & BOOTSTRAP != 0
    }
}

/// The reserved bits that sit below the address field: the low eight, and the
/// one between the bootstrap flag and the two enables.
pub(super) const RESERVED_LOW: u64 = 0xFF | (1 << 9);

/// `BSP`: set on the processor the machine started on, and read-only to
/// software.
pub(super) const BOOTSTRAP: u64 = 1 << 8;

/// `EXTD`: the controller answers through model-specific registers. Meaningless
/// without [`GLOBAL_ENABLE`], and writing it without that one faults.
pub(super) const X2APIC_ENABLE: u64 = 1 << 10;

/// `EN`: the controller is switched on.
pub(super) const GLOBAL_ENABLE: u64 = 1 << 11;

#[cfg(test)]
mod tests {
    //! What the reserved-bit check makes of the bits above the physical address
    //! space is not asserted here: that width comes from the `CPUID` of
    //! whatever processor the test runs on, and is not the guest's machine.

    use super::{ApicBase, BOOTSTRAP, BaseFault, Controller, GLOBAL_ENABLE, Mode, X2APIC_ENABLE};
    use crate::hardware::model::{self, Model};

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
    fn a_processor_with_no_x2apic_refuses_it_by_its_own_rule() {
        // Named as the missing feature rather than as a reserved bit: the write
        // had no reserved bit set, and the two rules that refuse this move — a
        // processor without the mode, and a machine whose delivery policy will
        // not drive it — are what an operator reading the line has to tell apart.
        assert_eq!(
            state(Mode::XApic).written(bits(Mode::X2Apic), model::tests::SPARSE),
            Err(BaseFault::X2ApicUnsupported)
        );
        // And the same processor takes every other move, so the refusal is about
        // the mode and not about the register.
        for to in [Mode::XApic, Mode::Disabled] {
            assert_eq!(
                state(Mode::XApic)
                    .written(bits(to), model::tests::SPARSE)
                    .map(ApicBase::mode),
                Ok(to),
                "{to}"
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
    fn moving_the_page_is_ignored_rather_than_refused() {
        // Architecturally writable, and not writable here: the page is trapped
        // once before any guest runs. What a guest gets is the write it made
        // minus the part this machine cannot honour, rather than a fault the
        // architecture does not define for it.
        let elsewhere = (ApicBase::DEFAULT_PAGE + 0x1_0000) | GLOBAL_ENABLE;
        let written = state(Mode::XApic)
            .written(elsewhere, MODEL)
            .expect("a relocating write is taken, with the address left where it is");

        assert_eq!(written, state(Mode::XApic));
        assert_eq!(ApicBase::page_of(written.bits()), ApicBase::DEFAULT_PAGE);
    }

    #[test]
    fn enabling_a_controller_at_a_relocated_address_still_enables_it() {
        // The write the refusal used to kill: the enable bit set together with an
        // address out of firmware's tables, which is how software that does not
        // read-modify-write this register enables a controller.
        let forced = 0xFEC0_0000 | GLOBAL_ENABLE;
        let written = state(Mode::Disabled)
            .written(forced, MODEL)
            .expect("switching a controller on is a legal transition");

        assert_eq!(written.mode(), Mode::XApic);
        assert_eq!(ApicBase::page_of(written.bits()), ApicBase::DEFAULT_PAGE);
    }

    #[test]
    fn a_controller_is_never_seeded_into_a_state_that_does_not_exist() {
        // `EXTD` without `EN` is not a state, and every recovery from one faults:
        // the guest reads the register, sets the enable bit it is missing, and
        // writes back a move from disabled straight into x2APIC.
        let malformed = ApicBase::DEFAULT_PAGE | X2APIC_ENABLE;
        let seeded = ApicBase::seeded(malformed, false, true);

        assert_eq!(seeded.mode(), Mode::Disabled);
        assert_eq!(seeded.bits() & X2APIC_ENABLE, 0);
        // And what firmware really having been in x2APIC seeds, which is the
        // whole reason the enable bits are taken from it at all.
        let both = ApicBase::DEFAULT_PAGE | GLOBAL_ENABLE | X2APIC_ENABLE;

        assert_eq!(ApicBase::seeded(both, false, true).mode(), Mode::X2Apic);
    }

    #[test]
    fn a_controller_is_never_seeded_into_a_face_the_machine_will_not_serve() {
        // Every combination of the two enable bits, on a machine whose policy
        // withholds the wider face. Firmware leaving `EXTD` set is the case
        // this exists for: the guest would otherwise start in a face its own
        // `CPUID` denies, and software that believes `CPUID` drives the
        // memory-mapped one against a controller that answers as the other.
        for (value, permitted, expected) in [
            (0, false, Mode::Disabled),
            (X2APIC_ENABLE, false, Mode::Disabled),
            (GLOBAL_ENABLE, false, Mode::XApic),
            (GLOBAL_ENABLE | X2APIC_ENABLE, false, Mode::XApic),
            (0, true, Mode::Disabled),
            (X2APIC_ENABLE, true, Mode::Disabled),
            (GLOBAL_ENABLE, true, Mode::XApic),
            (GLOBAL_ENABLE | X2APIC_ENABLE, true, Mode::X2Apic),
        ] {
            let seeded = ApicBase::seeded(ApicBase::DEFAULT_PAGE | value, false, permitted);
            assert_eq!(seeded.mode(), expected, "{value:#x}, permitted {permitted}");
            // Whichever face it lands in, the register reads back a state that
            // exists: the wider bit is never set without the global one.
            assert!(
                seeded.bits() & X2APIC_ENABLE == 0 || seeded.bits() & GLOBAL_ENABLE != 0,
                "{value:#x}, permitted {permitted}"
            );
        }
    }

    /// Every way the capture can describe the controller it read, so that a new
    /// one cannot be added without a decision about it here.
    const CONTROLLERS: [Controller; 5] = [
        Controller::Read,
        Controller::Absent,
        Controller::Disabled,
        Controller::Unreachable,
        Controller::Malformed,
    ];

    #[test]
    fn firmware_leaving_the_page_where_it_belongs_is_not_misplaced() {
        for controller in CONTROLLERS {
            assert_eq!(
                ApicBase::misplaced(controller, bits(Mode::XApic)),
                None,
                "{controller:?} at the default page"
            );
        }
    }

    #[test]
    fn firmware_moving_the_page_is_misplaced_however_the_controller_was_read() {
        // Every one of these ends with the real controller decoding at the
        // address in the register once the host switches it on, so every one of
        // them has to reach the same refusal — including the two the relocation
        // check used to sit behind.
        let elsewhere = 0xFEC0_0000 | GLOBAL_ENABLE;
        for controller in CONTROLLERS
            .into_iter()
            .filter(|controller| *controller != Controller::Absent)
        {
            assert_eq!(
                ApicBase::misplaced(controller, elsewhere),
                Some(0xFEC0_0000),
                "{controller:?} with the page moved"
            );
        }
    }

    #[test]
    fn a_processor_with_no_controller_has_no_page_to_have_moved() {
        // The captured value is a zero nobody wrote, and answering "the page is
        // at zero" would refuse a machine for the wrong reason — installing the
        // real controllers has already refused it for the right one.
        assert_eq!(ApicBase::misplaced(Controller::Absent, 0), None);
    }

    #[test]
    fn where_a_captured_value_puts_the_page_is_the_whole_field_above_the_flags() {
        // The predicate a machine is refused on, asked directly. The field is
        // bits 63:12 rather than a page mask of 51:12, because the reserved-bit
        // test a *write* is judged against comes from the processor's own
        // physical-address width — so an address bit above 51 that passed that
        // test would otherwise vanish here, and firmware's page would be reported
        // as the one this hypervisor traps.
        assert!(!ApicBase::relocated(ApicBase::DEFAULT_PAGE | GLOBAL_ENABLE));
        assert_eq!(
            ApicBase::page_of(ApicBase::DEFAULT_PAGE | GLOBAL_ENABLE | BOOTSTRAP),
            ApicBase::DEFAULT_PAGE
        );
        for elsewhere in [0xFEC0_0000, 0x1000, 1 << 60] {
            assert!(ApicBase::relocated(elsewhere), "{elsewhere:#x}");
            assert_eq!(ApicBase::page_of(elsewhere | GLOBAL_ENABLE), elsewhere);
        }
        // The bits below the field are not part of it, whichever of them is set.
        assert_eq!(ApicBase::page_of(0xFFF | X2APIC_ENABLE), 0);
    }

    #[test]
    fn a_controller_already_in_a_state_that_does_not_exist_can_only_be_switched_off() {
        // `EXTD` without `EN` is reachable only through `from_bits`, which is what
        // a capture would reach if `seeded` did not drop the bit. From there the
        // register reads as merely disabled, so the moves out of it are the
        // disabled ones — and the write that would look like a repair, setting the
        // enable bit it is missing, is a move from disabled straight into x2APIC
        // and faults.
        let invalid = ApicBase::from_bits(ApicBase::DEFAULT_PAGE | X2APIC_ENABLE);
        assert_eq!(invalid.mode(), Mode::Disabled);
        assert_eq!(
            invalid
                .written(bits(Mode::XApic), MODEL)
                .map(ApicBase::mode),
            Ok(Mode::XApic)
        );
        assert_eq!(
            invalid.written(bits(Mode::X2Apic), MODEL),
            Err(BaseFault::IllegalTransition)
        );
    }
}
