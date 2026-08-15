//! What a write to the base register does to the controller behind it.
//!
//! [`super::ApicBase`] decides whether a write is one the architecture allows;
//! this is what the controller does about one that is. The three answers are the
//! three the architecture distinguishes, and the difference between them is what
//! survives: a write naming the state already held changes nothing, the move
//! between the two faces preserves the whole register file, and everything else
//! is a lifecycle boundary that leaves the file as reset leaves it — which is why
//! the physical hardware behind it has to be brought across before anything
//! virtual moves.

use core::sync::atomic::Ordering;

use thiserror::Error;

use crate::{
    hardware::{model::Model, sources, timer},
    registers::{
        Vlapic,
        base::{
            ApicBase, BOOTSTRAP, GLOBAL_ENABLE, Mode, RESERVED_LOW, X2APIC_ENABLE,
        },
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
    /// A change of face is a lifecycle boundary rather than a different way of
    /// naming the same registers, and which of the two boundaries it is decides
    /// everything below.
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
    /// Switching the controller off is the other thing entirely, and leaves the
    /// register file as reset leaves it. There the physical hardware has to be
    /// brought across first: an entry left armed goes on delivering into a
    /// controller the guest believes is switched off, and every acknowledgement
    /// owed has to be settled before the tokens that would have discharged it
    /// are deleted.
    ///
    /// # Errors
    ///
    /// Whatever the transition refused: a reserved bit, a state that cannot be
    /// reached from this one, or an attempt to move the register page.
    pub(crate) fn write_base(&self, value: u64) -> Result<Transition, BaseFault> {
        let current = self.base();
        let next = current.written(value, self.model)?;
        if next.mode() == current.mode() {
            // Not a transition. Software that reads the register, changes a
            // field it is entitled to and writes it back has asked for nothing
            // to happen, and nothing does.
            self.base.store(next.bits(), Ordering::Release);
            return Ok(Transition::Unchanged);
        }
        if matches!(
            (current.mode(), next.mode()),
            (Mode::XApic, Mode::X2Apic) | (Mode::X2Apic, Mode::XApic)
        ) {
            self.base.store(next.bits(), Ordering::Release);
            // The two exceptions the architecture names. The logical destination
            // stops being stored at all — x2APIC derives it from the identifier
            // — and the destination half of the command register has no
            // equivalent to carry over.
            self.logical_destination.store(0, Ordering::Release);
            self.command
                .store(self.command().low().into(), Ordering::Release);
            return Ok(Transition::Preserved);
        }
        // Before anything virtual moves, and in this order: a source that is
        // still armed can deliver into whatever comes next, and a debt that is
        // still outstanding needs the register file that records it.
        let quiet = sources::quiesce(self) & timer::disarm(self);
        let settled = self.ledger.settle();

        self.base.store(next.bits(), Ordering::Release);
        self.reset_registers();
        Ok(Transition::Changed { quiet, settled })
    }
}

/// What a write to the base register did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Transition {
    /// The write named the state the register was already in, so nothing
    /// happened and nothing has to be reconciled.
    Unchanged,
    /// The controller changed face and kept its register file, which is what
    /// the architecture preserves across that one move — so nothing behind
    /// it was quieted and there was nothing to settle.
    Preserved,
    /// The controller changed face and its register file was reset, so the
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
