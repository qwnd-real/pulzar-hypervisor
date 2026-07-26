//! Choosing a counter of known rate, taking it over, and giving it back.
//!
//! A reference is a counter whose frequency is known without measuring it,
//! which on a PC means one of two: the event timer, which reports its own tick
//! period, and the power management timer, whose rate ACPI fixes. Everything
//! else either has to be calibrated first or cannot be trusted to keep
//! counting.
//!
//! Both are tried in turn rather than the better one being picked and the other
//! discarded. A machine can describe an event timer that turns out not to work
//! — an address nothing can map, a block describing itself impossibly — and
//! refusing to boot over that while a perfectly good power management timer
//! sits beside it would be answering a question nobody asked.
//!
//! Taking one over can change the machine — a mapping in the hypervisor's own
//! address space, an event timer that has to be started — and all of that is
//! recorded in one place so that giving it back is one call rather than a
//! checklist at every exit.

use acpi::{Acpi, Fadt, Hpet, PmTimer};
use log::warn;
use paging::{AddressSpace, Mapping};
use x86_64::PhysAddr;

use crate::{
    ClockError,
    counter::{Counter, Kind},
    hpet::{self, Restore},
    pm_timer,
};

/// A counter of known rate the clock can measure against.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Reference {
    /// An event timer's register block, at this physical address.
    Hpet(PhysAddr),
    /// The power management timer's register.
    PmTimer(PmTimer),
}

impl Reference {
    /// Which piece of hardware this names.
    const fn kind(self) -> Kind {
        match self {
            Self::Hpet(_) => Kind::Hpet,
            Self::PmTimer(_) => Kind::PmTimer,
        }
    }
}

/// A reference counter the clock has taken over, and what it owes the machine
/// back.
#[derive(Debug)]
pub(crate) struct Borrowed {
    counter: Counter,
    mapping: Option<Mapping>,
    started: Option<Restore>,
}

impl Borrowed {
    /// A counter that has been taken over, the mapping it is read through where
    /// it needed one, and what the machine is owed back where anything is.
    pub(crate) const fn new(
        counter: Counter,
        mapping: Option<Mapping>,
        started: Option<Restore>,
    ) -> Self {
        Self {
            counter,
            mapping,
            started,
        }
    }

    /// Takes over the best reference the machine describes that can be taken
    /// over at all.
    ///
    /// # Errors
    ///
    /// [`ClockError::NoReference`] if the machine describes neither counter, or
    /// else whatever the best candidate's own opening reported once every
    /// candidate has refused: a register nothing can reach, hardware describing
    /// itself impossibly, or a mapping that could not be made.
    pub(crate) fn open(space: &mut AddressSpace, acpi: &Acpi) -> Result<Self, ClockError> {
        let mut refused = None;
        for reference in candidates(acpi).into_iter().flatten() {
            let kind = reference.kind();
            match Self::take(space, reference) {
                Ok(borrowed) => return Ok(borrowed),
                Err(error) => {
                    warn!("clock: the {kind} cannot serve as a reference: {error}");
                    // The best candidate's complaint is the informative one; a
                    // fallback failing afterwards says less about the machine.
                    refused = refused.or(Some(error));
                }
            }
        }
        Err(refused.unwrap_or(ClockError::NoReference))
    }

    /// The counter, for as long as this borrow lasts.
    ///
    /// Lent rather than handed over. A counter read through a mapping must not
    /// outlive that mapping, and tying it to this borrow is what makes that the
    /// borrow checker's problem instead of a reviewer's: [`Borrowed::release`]
    /// consumes the borrow, so no readable counter can survive it.
    pub(crate) const fn counter(&self) -> &Counter {
        &self.counter
    }

    /// Gives the counter back: stops the event timer if this clock started it,
    /// and releases the mapping.
    ///
    /// # Errors
    ///
    /// [`ClockError::Paging`] if the mapping cannot be released, which leaves
    /// the window address space it occupied spent but the machine's own
    /// hardware already restored.
    pub(crate) fn release(self, space: &mut AddressSpace) -> Result<(), ClockError> {
        if let Some(restore) = self.started {
            hpet::stop(&restore);
        }
        match self.mapping {
            // SAFETY: the counter that read through this mapping is dropped with
            // `self`, and nothing else was ever handed the address.
            Some(mapping) => unsafe { space.unmap(mapping) }.map_err(ClockError::Paging),
            None => Ok(()),
        }
    }

    /// Keeps the counter for good.
    ///
    /// The mapping is deliberately never released and an event timer this clock
    /// started is deliberately never stopped: from here the counter *is* the
    /// hypervisor's timebase, so the address it is read through has to stay
    /// valid and the ticks have to keep coming for as long as pulzar runs. This
    /// is the one path a counter may outlive its borrow by, and it is sound
    /// exactly because it is the path that never unmaps.
    pub(crate) fn keep(self) -> Counter {
        self.counter
    }

    /// Takes over the one counter `reference` names.
    fn take(space: &mut AddressSpace, reference: Reference) -> Result<Self, ClockError> {
        match reference {
            Reference::Hpet(base) => hpet::open(space, base),
            Reference::PmTimer(timer) => pm_timer::open(space, timer),
        }
    }
}

/// The references the machine describes, best first.
///
/// The event timer leads: it counts fast enough to calibrate against in
/// milliseconds, and it is the only one of the two that can keep time
/// afterwards. The power management timer follows, because a machine can be
/// built without an event timer and calibrating against a slow counter is still
/// better than guessing at a frequency.
fn candidates(acpi: &Acpi) -> [Option<Reference>; 2] {
    [
        acpi.hpet().and_then(Hpet::memory_base).map(Reference::Hpet),
        acpi.fadt().and_then(Fadt::pm_timer).map(Reference::PmTimer),
    ]
}
