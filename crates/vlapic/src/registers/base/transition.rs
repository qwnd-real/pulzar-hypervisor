//! What a write to the base register does to the controller behind it.
//!
//! [`super::ApicBase`] decides whether a write is one the architecture allows;
//! this is what the controller does about one that is. The three answers are
//! the three the architecture distinguishes, and the difference between them is
//! what survives: a write naming the state already held changes nothing, moving
//! between the faces or switching a disabled controller on keeps the whole
//! register file, and switching an enabled one off is a lifecycle boundary that
//! leaves the file as reset leaves it — which is why the physical hardware
//! behind it has to be brought across before anything virtual moves.

use core::sync::atomic::Ordering;

use thiserror::Error;

use crate::{
    hardware::{model::Model, sources, timer},
    registers::{
        Vlapic,
        base::{ApicBase, BOOTSTRAP, GLOBAL_ENABLE, Mode, RESERVED_LOW, X2APIC_ENABLE},
    },
};

impl Vlapic {
    /// Which face the guest is reaching this controller through.
    pub(crate) fn mode(&self) -> Mode {
        self.base().mode()
    }

    /// The base register as it stands.
    pub(crate) fn base(&self) -> ApicBase {
        ApicBase::from_bits(self.base.load(Ordering::Acquire))
    }

    /// Takes a write to the base register, and says what it became.
    ///
    /// Which transition it is decides everything below, and [`resets`] is the
    /// one statement of which of them throws the register file away.
    ///
    /// Entering x2APIC from the older face preserves everything the
    /// architecture says it preserves — the priorities, what is requested and
    /// in service, the table, the errors — because a guest doing it is
    /// changing how it addresses its controller and not asking for a new
    /// one. So the machine behind those registers is preserved too, and
    /// deliberately: a timer that is counting goes on counting, an armed
    /// deadline stays armed, and every acknowledgement real hardware is
    /// owed stays owed, because the entries and the in-service bits that
    /// make sense of all three survive the write. Quieting them here would
    /// take away a timer the guest is entitled to keep and an appointment
    /// it cannot re-derive.
    ///
    /// Switching a controller on preserves it too, for a different reason:
    /// there is nothing there to clear. What a guest gets back is the file
    /// the disable left at reset, plus whatever reached it while it was off
    /// — which is a non-maskable interrupt another processor sent it and an
    /// error a remote sender recorded against it, both of which are
    /// delivered to a controller whatever its mode and neither of which the
    /// enable is entitled to discard.
    ///
    /// Switching one off is the other thing entirely, and leaves the register
    /// file as reset leaves it. There the physical hardware has to be brought
    /// across first: an entry left armed goes on delivering into a controller
    /// the guest believes is switched off, and every acknowledgement owed
    /// has to be settled before the tokens that would have discharged it
    /// are deleted.
    ///
    /// # Errors
    ///
    /// Whatever the transition refused: a reserved bit, or a state that cannot
    /// be reached from this one.
    pub(crate) fn write_base(&self, value: u64) -> Result<Transition, BaseFault> {
        let current = self.base();
        let next = current.written(value, self.model)?;
        let (from, to) = (current.mode(), next.mode());
        if from == to {
            // Not a transition. Software that reads the register, changes a
            // field it is entitled to and writes it back has asked for nothing
            // to happen, and nothing does.
            self.base.store(next.bits(), Ordering::Release);
            return Ok(Transition::Unchanged);
        }
        if resets(from, to) {
            return Ok(self.switched_off(next));
        }
        self.base.store(next.bits(), Ordering::Release);
        if to == Mode::X2Apic {
            // The two exceptions the architecture names. The logical destination
            // stops being stored at all — x2APIC derives it from the identifier
            // — and the destination half of the command register has no
            // equivalent to carry over.
            self.logical_destination.store(0, Ordering::Release);
            self.command
                .store(self.command().low().into(), Ordering::Release);
        }
        Ok(Transition::Preserved)
    }

    /// Switches the controller off, which is the one transition that throws the
    /// register file away.
    ///
    /// The physical hardware is brought across before anything virtual moves,
    /// and in this order: a source that is still armed can deliver into
    /// whatever comes next, and a debt that is still outstanding needs the
    /// register file that records it.
    ///
    /// The controller the debts are paid through is this processor's, and it is
    /// this processor's because the write being taken came out of the guest
    /// running here — the same thing that makes quieting the sources and
    /// stopping the timer legitimate.
    fn switched_off(&self, next: ApicBase) -> Transition {
        let quiet = sources::quiesce(self) & timer::disarm(self);
        let settled = self.ledger.settle(&apic::local().ok());

        self.base.store(next.bits(), Ordering::Release);
        self.reset_registers();
        Transition::Changed { quiet, settled }
    }
}

/// Whether moving between these two faces leaves the register file as reset
/// leaves it.
///
/// One statement of which transition is a lifecycle boundary, because getting
/// it wrong in either direction is a defect: a reset that does not happen
/// leaves a switched-off controller holding a guest's timer configuration and
/// its in-service bits, and a reset that happens where the architecture does
/// not ask for one destroys state the guest is entitled to keep across it.
///
/// Only switching an enabled controller off does. Neither of the other two
/// moves that go anywhere is a reset: entering x2APIC preserves the whole file
/// by definition, and switching a disabled controller *on* has nothing to clear
/// — the file was left at reset by the disable that got it there, and what has
/// accumulated since is state a disabled controller legitimately holds. A
/// non-maskable interrupt it was sent, and an error a remote sender recorded
/// against it, are both delivered to a controller whatever its mode, and a
/// guest that switches its own controller on has not asked to lose either.
const fn resets(from: Mode, to: Mode) -> bool {
    matches!(to, Mode::Disabled) && !matches!(from, Mode::Disabled)
}

/// What a write to the base register did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Transition {
    /// The write named the state the register was already in, so nothing
    /// happened and nothing has to be reconciled.
    Unchanged,
    /// The controller changed face and kept its register file, which is what
    /// entering x2APIC preserves and what switching a controller on has nothing
    /// to discard — so nothing behind it was quieted and there was nothing to
    /// settle.
    Preserved,
    /// The controller was switched off and its register file was reset, so the
    /// physical hardware behind it was brought across first.
    Changed {
        /// Whether every source really was quieted first.
        quiet: bool,
        /// Whether every acknowledgement real hardware was owed really was
        /// settled first.
        settled: bool,
    },
}

impl ApicBase {
    /// What this register becomes when the guest writes `value`, or why it may
    /// not.
    ///
    /// The checks run in the order the guest would meet them on real hardware:
    /// reserved bits first, then the transition.
    ///
    /// Two fields of what a guest writes are not the guest's to change and are
    /// dropped rather than refused. The bootstrap flag is read-only to software
    /// on real hardware too. The address field is not: it is architecturally
    /// writable, and ignoring a write to it is this hypervisor's own deviation,
    /// taken because the page is trapped in the nested page tables once before
    /// any guest runs and nothing here can re-trap a range while processors are
    /// executing. Ignoring it leaves a guest that reads the register back
    /// seeing the address it did not get, and running; raising a fault the
    /// architecture does not define instead would kill software that writes
    /// the field wholesale, which is what the usual way of enabling a
    /// controller from a firmware-supplied address does.
    ///
    /// # Errors
    ///
    /// [`BaseFault::Reserved`] if any bit the architecture reserves was written
    /// non-zero, or [`BaseFault::IllegalTransition`] if the two enable bits
    /// name a state this one cannot go to.
    pub(crate) fn written(self, value: u64, model: Model) -> Result<Self, BaseFault> {
        if value & reserved() != 0 {
            return Err(BaseFault::Reserved);
        }

        // The two enable bits are the whole of what a guest may change here. The
        // bootstrap flag is carried through from where it was, and the address is
        // written as the constant this hypervisor traps rather than as what the
        // guest asked for — which is what makes the page not moving a property of
        // this type rather than of every caller, exactly as it is in
        // `ApicBase::seeded`.
        let next = Self(
            (value & (GLOBAL_ENABLE | X2APIC_ENABLE)) | Self::DEFAULT_PAGE | (self.0 & BOOTSTRAP),
        );

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

#[cfg(test)]
mod tests {
    //! Which transition throws the register file away is the one decision in
    //! this file that can be checked without a controller, and it is the one
    //! that costs a guest state it is entitled to keep if it is wrong.

    use super::{Mode, resets};

    /// Every state, so that a mode cannot be added without a decision here.
    const STATES: [Mode; 3] = [Mode::Disabled, Mode::XApic, Mode::X2Apic];

    #[test]
    fn only_switching_an_enabled_controller_off_resets_the_register_file() {
        for from in STATES {
            for to in STATES {
                assert_eq!(
                    resets(from, to),
                    to == Mode::Disabled && from != Mode::Disabled,
                    "{from} to {to}"
                );
            }
        }
    }

    #[test]
    fn switching_a_controller_on_keeps_what_reached_it_while_it_was_off() {
        // The enable edge, which used to reach the same reset as the two
        // disabling ones — destroying a non-maskable interrupt another processor
        // had sent the controller and every error a remote sender had recorded
        // against it, neither of which a controller has to be enabled to be
        // given.
        assert!(!resets(Mode::Disabled, Mode::XApic));
    }
}
