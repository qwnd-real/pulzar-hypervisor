//! Choosing a counter of known rate, taking it over, and giving it back.
//!
//! A reference is a counter whose frequency is known without measuring it,
//! which on a PC means one of two: the event timer, which reports its own tick
//! period, and the power management timer, whose rate ACPI fixes. Everything
//! else either has to be calibrated first or cannot be trusted to keep
//! counting.
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
    counter::Counter,
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

/// The best reference the machine describes.
///
/// The event timer first: it counts fast enough to calibrate against in
/// milliseconds, and it is the only one of the two that can keep time
/// afterwards. The power management timer second, because a machine can be
/// built without an event timer and calibrating against a slow counter is still
/// better than guessing at a frequency.
///
/// # Errors
///
/// [`ClockError::NoReference`] if the machine describes neither, in which case
/// there is nothing left to measure against and no timebase can be established.
pub(crate) fn choose(acpi: &Acpi) -> Result<Reference, ClockError> {
    if let Some(base) = acpi.hpet().and_then(Hpet::memory_base) {
        return Ok(Reference::Hpet(base));
    }
    let timer = acpi
        .fadt()
        .and_then(Fadt::pm_timer)
        .ok_or(ClockError::NoReference)?;
    warn!("clock: no usable hpet; falling back to the acpi power management timer");
    Ok(Reference::PmTimer(timer))
}

/// A reference counter the clock has taken over, and what it owes the machine
/// back.
#[derive(Debug)]
pub(crate) struct Borrowed {
    /// The counter, readable for as long as this value exists.
    pub(crate) counter: Counter,
    /// The mapping it is read through, where it needed one.
    pub(crate) mapping: Option<Mapping>,
    /// The event timer this clock started, and what it has to undo to hand
    /// the block back the way it was found.
    pub(crate) started: Option<Restore>,
}

impl Borrowed {
    /// Takes over whichever counter `reference` names.
    ///
    /// # Errors
    ///
    /// Whatever the counter's own opening reports: a register nothing can
    /// reach, hardware describing itself impossibly, or a mapping that could
    /// not be made.
    pub(crate) fn open(space: &mut AddressSpace, reference: Reference) -> Result<Self, ClockError> {
        match reference {
            Reference::Hpet(base) => hpet::open(space, base),
            Reference::PmTimer(timer) => pm_timer::open(space, timer),
        }
    }

    /// The counter, for as long as this borrow lasts.
    pub(crate) const fn counter(&self) -> Counter {
        self.counter
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
    /// valid and the ticks have to keep coming for as long as pulzar runs.
    pub(crate) fn keep(self) -> Counter {
        self.counter
    }
}
