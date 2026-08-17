//! The register space AMD adds above the architectural local APIC, and the two
//! things in it a hypervisor that withholds acknowledgements needs.
//!
//! An acknowledgement to an ordinary local APIC carries no vector: it retires
//! whichever vector in service has the highest priority. That one fact is what
//! the emulated controller's software ledger is built around — a withheld
//! acknowledgement may only be issued at the moment its own vector *is* the
//! highest one held, so a debt is tracked through three states and every
//! payment is a read of the in-service bank followed by a write that depends on
//! what it said.
//!
//! The extended space removes the fact. Its acknowledgement register takes a
//! vector number, so there is nothing to read first and nothing to wait for;
//! and its interrupt-enable registers give every vector a bit deciding whether
//! this controller accepts an interrupt on it at all, which is a way to quiet
//! one line without reaching the I/O controller that raised it — hardware this
//! hypervisor passes through and cannot program.
//!
//! # What is used, and what is deliberately not
//!
//! Two parts of the four, and they are taken together or not at all: retiring a
//! named vector is what stops a refused interrupt blocking a whole priority
//! class, and the enable bit is what stops the line it left asserted arriving
//! again at once. Either alone leaves a case with no good answer, so a
//! controller offering only one of them is treated as offering neither.
//!
//! The extended local vector table entries are not touched. They deliver
//! machine-check thresholding and instruction-based sampling, neither of which
//! this hypervisor models, and firmware chooses their offsets — so programming
//! one would take a source away from whoever firmware told to expect it.
//!
//! The extended identifier is not switched on either, and that one would be
//! actively wrong: it widens the field this processor's identifier is read out
//! of, and every passed-through interrupt on this machine is already routed by
//! the identifier a guest believes. A bit firmware set is preserved rather than
//! cleared, for the same reason — it is not this crate's to decide either way.
//!
//! # Presence is asked in three steps, in that order
//!
//! `CPUID`, then the version register, then the space's own feature register.
//! The order is not a preference: reading a register the extended space does
//! not have latches an illegal-register-address error and raises the
//! controller's error interrupt, so the feature register at the bottom of the
//! space may only be read once something else has said the space is there.
//!
//! Every part of it is off at reset, so the control register is written before
//! any of it answers, and read back afterwards — a controller that did not take
//! the write is one this reports as unusable rather than one it goes on issuing
//! vector-named acknowledgements to that quietly do nothing.

use core::fmt::{self, Display, Formatter};

use bitflags::bitflags;
use descriptors::Vector;
use processor::Features;

use crate::{
    LocalApic, deliverable,
    register::{self, Access, Register},
};

bitflags! {
    /// What the extended register space offers on this machine's controllers.
    ///
    /// Every flag is a separate question and a machine can answer them
    /// independently, which is why this is not one boolean: what a log line has
    /// to distinguish is a processor without the space at all, one whose space
    /// lacks a part, and one that would not switch a part on. The empty set is
    /// the first of those, and is what every controller with no extended space
    /// answers.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Extended: u32 {
        /// `CPUID` and the version register both say the space is there, so its
        /// own feature register may be read.
        const PRESENT = 1 << 0;
        /// The space reports the register that retires a named vector.
        const SPECIFIC_EOI = 1 << 1;
        /// It reports the registers that decide which vectors are accepted.
        const INTERRUPT_ENABLE = 1 << 2;
        /// Both parts were switched on and read back switched on.
        const SWITCHED_ON = 1 << 3;
        /// The two parts this crate uses, which are used together or not at
        /// all.
        const BOTH = Self::SPECIFIC_EOI.bits() | Self::INTERRUPT_ENABLE.bits();
    }
}

impl Extended {
    /// Whether this controller can retire one named vector and stop another
    /// being accepted.
    ///
    /// The one question anything outside this module asks, because the two
    /// operations are only useful together.
    #[must_use]
    pub const fn usable(self) -> bool {
        self.contains(Self::BOTH.union(Self::SWITCHED_ON))
    }
}

impl Display for Extended {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        if !self.contains(Self::PRESENT) {
            return formatter.write_str("no extended apic space");
        }
        write!(
            formatter,
            "extended apic space with{} a specific acknowledgement and with{} interrupt enables, \
             {}",
            if self.contains(Self::SPECIFIC_EOI) {
                ""
            } else {
                "out"
            },
            if self.contains(Self::INTERRUPT_ENABLE) {
                ""
            } else {
                "out"
            },
            if self.contains(Self::SWITCHED_ON) {
                "switched on"
            } else {
                "not switched on"
            },
        )
    }
}

/// Retiring one named vector, and stopping one being accepted, on the
/// controller of the processor holding this.
///
/// Obtained only from [`LocalApic::specific`], and only on a controller whose
/// [`Extended`] is [usable](Extended::usable) — so holding one of these is the
/// proof that both operations reach a register that is there and switched on,
/// and neither of them has a failure to report. It belongs to the processor
/// that obtained it for the reason a [`LocalApic`] does: every register it
/// names answers about whoever is doing the reaching.
#[derive(Clone, Copy, Debug)]
pub struct Specific {
    /// The controller these operations are performed on, which carries no
    /// interface of its own and works out at each use how this processor
    /// reaches its registers.
    local: LocalApic,
}

impl Specific {
    /// The extended operations on `local`'s controller.
    pub(crate) const fn new(local: LocalApic) -> Self {
        Self { local }
    }

    /// Retires `vector`, whatever else this controller is holding in service.
    ///
    /// The operation the ordinary acknowledgement cannot express. It names its
    /// vector, so nothing has to establish that the vector owed is the highest
    /// one held — and a vector this controller is *not* holding retires nothing
    /// rather than retiring something else, which is what makes it safe to
    /// issue against a debt that turns out to have been discharged already.
    ///
    /// For a vector that arrived level triggered the controller also broadcasts
    /// an end-of-interrupt to the I/O controllers, clearing the remote request
    /// bit of whichever one sent it. That is what lets a line be re-armed, and
    /// it is also why a line whose device is still asserting is delivered again
    /// straight away: see [`Specific::set_enabled`], which is what a caller
    /// with nobody left to service the interrupt uses first.
    pub fn end_of_interrupt(self, vector: Vector) {
        self.local.access().retire(vector);
    }

    /// Decides whether this controller accepts an interrupt on `vector` at all.
    ///
    /// A cleared bit does not discard an arrival: the request is still latched
    /// in the request register, and what it stops is the request being accepted
    /// into service. So the interrupt is held rather than lost, and setting the
    /// bit again delivers it — which is the honest behaviour for a line that is
    /// still asserted, and the reason nothing here has to remember an arrival
    /// of its own.
    ///
    /// What it costs is that one vector. A request held in the request register
    /// is not held in service, so no priority class is blocked by it and every
    /// other vector on this processor goes on arriving — which is the whole
    /// difference from withholding an acknowledgement instead.
    ///
    /// A vector below the first the platform may assign is left alone. Its bit
    /// is in the one register of the bank whose low half the architecture
    /// reserves, and it is an architectural exception: no controller may be
    /// told to deliver on it, so there is nothing to decide and nothing here
    /// ever writes a reserved field.
    pub fn set_enabled(self, vector: Vector, enabled: bool) {
        if !deliverable(vector) {
            return;
        }
        let access = self.local.access();
        let (register, bit) = register::word(Register::INTERRUPT_ENABLE, vector.number());
        let held = access.read(register);
        let next = if enabled { held | bit } else { held & !bit };
        // SAFETY: the register is one of the eight the bank is spread across on
        // a controller that reported the bank, `word` cannot name anything
        // outside it, and every bit of this register is one software may write —
        // the sixteen the architecture reserves belong to vectors the check
        // above has already refused, so what is written back is what was read
        // with one bit of a writable field changed.
        unsafe { access.write(register, next) };
    }
}

/// Finds out what the extended space offers on this processor's controller and
/// switches on the parts of it this crate uses.
///
/// Called by every processor for its own controller, from bring-up, before
/// anything could ask a question this answers. What it returns is what the
/// controller was observed to do rather than what it was asked to do: the
/// enable bits are read back, because a controller that took the write and did
/// nothing would leave every later vector-named acknowledgement silently
/// retiring nothing at all.
pub(crate) fn install(access: Access) -> Extended {
    if !processor::features().contains(Features::EXTENDED_APIC_SPACE)
        || access.read(Register::VERSION) & EXTENDED_SPACE == 0
    {
        // Not merely unusable: there is no register at the bottom of the space
        // to read, and naming one would latch an illegal-register-address error
        // and raise this controller's error interrupt.
        return Extended::empty();
    }
    let offered = access.read(Register::EXTENDED_FEATURE);
    let mut found = Extended::PRESENT;
    found.set(Extended::SPECIFIC_EOI, offered & SPECIFIC_EOI_OFFERED != 0);
    found.set(
        Extended::INTERRUPT_ENABLE,
        offered & INTERRUPT_ENABLE_OFFERED != 0,
    );
    // Switched on together or not at all: one without the other leaves a refused
    // interrupt with no answer, so nothing is switched on that nothing will use.
    let wanted = if found.contains(Extended::BOTH) {
        SPECIFIC_EOI_ON | INTERRUPT_ENABLE_ON
    } else {
        0
    };
    // Whatever else the control register holds is left exactly as it was. The
    // one other bit the architecture defines here widens this processor's
    // identifier, which is what every passed-through interrupt on this machine
    // is routed by, and neither setting nor clearing it is this crate's to do.
    let held = access.read(Register::EXTENDED_CONTROL);
    // SAFETY: the register is present on a controller whose version register
    // says the space is, the two bits written are the ones the architecture
    // defines as switching on the two parts just read as offered, and every
    // other bit of the value came from the register itself.
    unsafe { access.write(Register::EXTENDED_CONTROL, held | wanted) };
    found.set(
        Extended::SWITCHED_ON,
        wanted != 0 && access.read(Register::EXTENDED_CONTROL) & wanted == wanted,
    );
    found
}

/// The version register's top bit, which says the extended space exists.
const EXTENDED_SPACE: u32 = 1 << 31;

/// The feature register's bit for the registers deciding which vectors are
/// accepted.
const INTERRUPT_ENABLE_OFFERED: u32 = 1 << 0;

/// The feature register's bit for the register that retires a named vector.
const SPECIFIC_EOI_OFFERED: u32 = 1 << 1;

/// The control register's bit that lets the interrupt-enable registers be
/// written.
///
/// It gates the writes and not the filtering: the bits go on deciding what this
/// controller accepts whether or not this is set, which is why handing the
/// machine on means putting the registers back before clearing this and not the
/// other way round.
const INTERRUPT_ENABLE_ON: u32 = 1 << 0;

/// The control register's bit that makes a write to the specific
/// acknowledgement register do anything. Without it the write is ignored.
const SPECIFIC_EOI_ON: u32 = 1 << 1;

#[cfg(test)]
mod tests {
    //! What can be checked without a controller: which combinations of the
    //! feature bits are usable, and that the description tells apart the
    //! machines a reader would have to act on differently.

    use alloc::{format, string::String};

    use super::Extended;

    #[test]
    fn a_controller_with_both_parts_switched_on_is_the_only_usable_one() {
        assert!(Extended::all().usable());
        for missing in [
            Extended::all().difference(Extended::SPECIFIC_EOI),
            Extended::all().difference(Extended::INTERRUPT_ENABLE),
            Extended::all().difference(Extended::SWITCHED_ON),
            Extended::PRESENT,
            Extended::empty(),
        ] {
            assert!(!missing.usable(), "{missing:?}");
        }
    }

    #[test]
    fn the_two_parts_are_named_together_because_they_are_used_together() {
        assert_eq!(
            Extended::BOTH,
            Extended::SPECIFIC_EOI | Extended::INTERRUPT_ENABLE
        );
        assert!(!Extended::BOTH.contains(Extended::PRESENT));
    }

    #[test]
    fn an_absent_space_is_told_apart_from_one_that_is_merely_unusable() {
        // Three different machines, and only one of them is a machine to look
        // into: one has no such registers, one has them and would not switch
        // them on, and one is missing a part.
        assert_eq!(format!("{}", Extended::empty()), "no extended apic space");
        let refused: String = format!("{}", Extended::all().difference(Extended::SWITCHED_ON));
        assert!(
            refused.contains("with a specific acknowledgement")
                && refused.contains("with interrupt enables")
                && refused.contains("not switched on"),
            "{refused}"
        );
        let partial: String = format!("{}", Extended::all().difference(Extended::INTERRUPT_ENABLE));
        assert!(partial.contains("without interrupt enables"), "{partial}");
    }
}
