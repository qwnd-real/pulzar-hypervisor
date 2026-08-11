//! `IA32_APIC_BASE`, the one register that says which interface a processor's
//! local APIC presents.
//!
//! Three things live in it: whether this processor is the one the machine
//! started on, whether the controller is switched on at all, and — the reason
//! this module exists — which of the two interfaces it answers through. It also
//! holds where the memory-mapped register page is, which is the processor's own
//! answer to a question firmware's tables answer separately.
//!
//! # The transition is one-way, and this module only ever goes forward
//!
//! A controller goes from disabled to the older interface to x2APIC and never
//! back: the architecture makes going straight from x2APIC to the older
//! interface a general protection fault, and the way back through disabled is
//! not something a running machine can do to itself. Two things follow, and
//! both are in [`enter`].
//!
//! It also means the three states are ordered, so [`enter`] asks for *at least*
//! a state rather than exactly one. A processor already in x2APIC that is asked
//! for the older interface has nothing to do and is left alone — which is not a
//! convenience: `INIT` preserves both of these bits, so a processor started on
//! a machine whose firmware was in x2APIC arrives in x2APIC, and refusing it
//! would lose the processor over a transition the architecture forbids anyway.
//! What a controller actually presents is therefore read back from this
//! register rather than assumed from what was asked for.
//!
//! The other is that x2APIC cannot be entered from disabled in one write. The
//! architecture defines only the step to the older interface out of disabled
//! and faults on the attempt to skip it, so a controller firmware switched off
//! is taken across in the two writes the architecture defines.
//!
//! Each write is checked rather than assumed. A write that the processor
//! accepts and does not act on would otherwise be found out by the first
//! register access faulting, with nothing left to say why.
//!
//! The base address is read and never written. Firmware chose where the
//! register page lives, and moving it would only mean that the hardware and
//! every table describing it disagreed.

use processor::Features;
use x86_64::{PhysAddr, registers::model_specific::Msr};

use crate::{ApicError, Mode};

/// The register itself.
const IA32_APIC_BASE: u32 = 0x1B;

/// The controller answers through model-specific registers.
pub(crate) const X2APIC_ENABLE: u64 = 1 << 10;

/// The controller is switched on. Clearing this is what the architecture calls
/// disabling it, and on many processors it cannot be set again.
pub(crate) const GLOBAL_ENABLE: u64 = 1 << 11;

/// The bits holding the physical address of the memory-mapped register page.
///
/// Frame-aligned and no wider than a physical address, so the field is the
/// whole of the register except the flags below it and the reserved bits above.
const ADDRESS_MASK: u64 = 0x000F_FFFF_FFFF_F000;

/// What a controller currently is, out of the two bits that say so.
///
/// Ordered, and that is the point: every transition the architecture defines
/// goes up this list, so a controller is only ever asked to reach a state at
/// least as far along it as the one it is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum State {
    /// Switched off. It has no registers to answer with.
    Disabled,
    /// Switched on, presenting the page of memory-mapped registers.
    XApic,
    /// Switched on, presenting model-specific registers.
    X2Apic,
}

impl State {
    /// What a value already read out of the register says the controller is, or
    /// `None` for the one combination of the two bits the architecture leaves
    /// undefined — the model-specific interface selected with the controller
    /// switched off.
    pub(crate) const fn of(base: u64) -> Option<Self> {
        match (base & GLOBAL_ENABLE != 0, base & X2APIC_ENABLE != 0) {
            (false, false) => Some(Self::Disabled),
            (true, false) => Some(Self::XApic),
            (true, true) => Some(Self::X2Apic),
            (false, true) => None,
        }
    }

    /// The state a mode asks for.
    const fn wanted(mode: Mode) -> Self {
        match mode {
            Mode::XApic => Self::XApic,
            Mode::X2Apic => Self::X2Apic,
        }
    }

    /// The bits that put the controller in this state, over whatever else the
    /// register holds.
    const fn bits(self) -> u64 {
        match self {
            Self::Disabled => 0,
            Self::XApic => GLOBAL_ENABLE,
            Self::X2Apic => GLOBAL_ENABLE | X2APIC_ENABLE,
        }
    }

    /// Which interface a switched-on controller in this state presents.
    pub(crate) const fn mode(self) -> Option<Mode> {
        match self {
            Self::Disabled => None,
            Self::XApic => Some(Mode::XApic),
            Self::X2Apic => Some(Mode::X2Apic),
        }
    }
}

/// Brings this processor's controller up to at least `mode`, and answers with
/// what it then presents.
///
/// At least, rather than exactly: the states are ordered and every transition
/// the architecture defines goes forwards along that order, so a controller
/// already past what was asked for is left where it is. A caller that needs to
/// know which interface it ended up with reads the answer rather than assuming
/// the request.
///
/// # Errors
///
/// [`ApicError::NoApic`] on a processor with no local controller,
/// [`ApicError::UndefinedBase`] if the register holds the one combination of
/// its two mode bits the architecture does not define, [`ApicError::NoX2Apic`]
/// if x2APIC was asked for and this processor does not implement it, or
/// [`ApicError::ModeNotEntered`] if the processor took a write and did not
/// change — which is what a controller firmware disabled for good looks like
/// from here.
pub(crate) fn enter(mode: Mode) -> Result<Mode, ApicError> {
    let wanted = State::wanted(mode);
    if wanted == State::X2Apic && !processor::features().contains(Features::X2APIC) {
        return Err(ApicError::NoX2Apic);
    }
    let start = state()?;
    // Every state the controller has to pass through, in order, and none it is
    // already past. The architecture defines no step from disabled straight to
    // x2APIC and faults on the attempt, so a switched-off controller asked for
    // x2APIC takes both of the steps it does define. Each is verified before the
    // next is attempted, because a step that did not happen would make the next
    // one the illegal one.
    for next in [State::XApic, State::X2Apic] {
        if next <= start || next > wanted {
            continue;
        }
        let base = read().ok_or(ApicError::NoApic)?;
        // SAFETY: the value differs from what the register already holds only in
        // the two enable bits, and it names the next state along the order the
        // architecture defines transitions in — never the undefined combination,
        // which `state` refuses, and never a step that skips the older
        // interface. The base address bits are carried through untouched, so the
        // controller does not move.
        unsafe { Msr::new(IA32_APIC_BASE).write(base | next.bits()) };
        if state()? != next {
            return Err(ApicError::ModeNotEntered);
        }
    }
    start.max(wanted).mode().ok_or(ApicError::ModeNotEntered)
}

/// What this processor's controller currently is.
///
/// The register is the only per-processor record of a transition each processor
/// makes for itself, which is what makes it the thing to ask before handing out
/// anything that reaches a controller's registers.
///
/// # Errors
///
/// [`ApicError::NoApic`] on a processor with no local controller, or
/// [`ApicError::UndefinedBase`] for the combination of the two mode bits the
/// architecture does not define.
pub(crate) fn state() -> Result<State, ApicError> {
    let base = read().ok_or(ApicError::NoApic)?;
    State::of(base).ok_or(ApicError::UndefinedBase)
}

/// Whether this processor's controller answers through model-specific
/// registers.
///
/// The question every register access asks, and the reason it is asked at each
/// one rather than remembered: a processor enters x2APIC for itself, while the
/// machine runs, and a remembered answer taken before that would go on reaching
/// a page the architecture has since made unavailable.
pub(crate) fn in_x2apic() -> bool {
    read().is_some_and(|base| base & X2APIC_ENABLE != 0)
}

/// Where this processor says its memory-mapped register page is, or `None` on a
/// processor with no local controller.
pub(crate) fn page() -> Option<PhysAddr> {
    read().map(page_of)
}

/// Where a value already read out of the register says the page is.
///
/// Truncating rather than checking: the mask has already cleared every bit
/// above the physical address space, so there is nothing left to lose and no
/// failure to report.
pub(crate) const fn page_of(value: u64) -> PhysAddr {
    PhysAddr::new_truncate(value & ADDRESS_MASK)
}

/// The register's current value, or `None` on a processor with no local APIC —
/// where the register does not exist and reading it faults.
///
/// The check is here rather than in each caller so that no path can reach the
/// read without it. It is also the whole of what this crate can ask: the same
/// feature bit reads as zero on a processor whose controller firmware switched
/// off at the hardware level, so such a machine is one this crate reports no
/// controller for rather than one it enables. Firmware that hands a machine on
/// through its own interrupt controllers has not done that.
pub(crate) fn read() -> Option<u64> {
    processor::features().contains(Features::APIC).then(|| {
        // SAFETY: `IA32_APIC_BASE` is architectural on every processor whose
        // `CPUID` reports a local APIC, which is what was just checked, and
        // reading it has no side effect.
        unsafe { Msr::new(IA32_APIC_BASE).read() }
    })
}

#[cfg(test)]
mod tests {
    use super::{GLOBAL_ENABLE, State, X2APIC_ENABLE, page_of};
    use crate::Mode;

    #[test]
    fn the_two_mode_bits_name_three_states_and_one_undefined_combination() {
        assert_eq!(State::of(0), Some(State::Disabled));
        assert_eq!(State::of(GLOBAL_ENABLE), Some(State::XApic));
        assert_eq!(
            State::of(GLOBAL_ENABLE | X2APIC_ENABLE),
            Some(State::X2Apic)
        );
        assert_eq!(State::of(X2APIC_ENABLE), None);
    }

    #[test]
    fn the_states_are_ordered_the_way_the_architecture_allows_transitions() {
        assert!(State::Disabled < State::XApic);
        assert!(State::XApic < State::X2Apic);
        assert_eq!(State::wanted(Mode::XApic), State::XApic);
        assert_eq!(State::wanted(Mode::X2Apic), State::X2Apic);
    }

    #[test]
    fn each_state_names_the_interface_it_presents() {
        assert_eq!(State::Disabled.mode(), None);
        assert_eq!(State::XApic.mode(), Some(Mode::XApic));
        assert_eq!(State::X2Apic.mode(), Some(Mode::X2Apic));
    }

    #[test]
    fn the_address_field_keeps_only_the_frame_the_page_is_in() {
        assert_eq!(page_of(0xFEE0_0000 | GLOBAL_ENABLE).as_u64(), 0xFEE0_0000);
        assert_eq!(page_of(u64::MAX).as_u64(), 0x000F_FFFF_FFFF_F000);
        assert_eq!(page_of(GLOBAL_ENABLE | X2APIC_ENABLE).as_u64(), 0);
    }
}
