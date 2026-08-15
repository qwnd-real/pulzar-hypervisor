//! The controllers themselves, and the three ways of reaching one.
//!
//! By roster position, which is what a [`CpuIndex`] is and what the array is
//! indexed by; by identifier, which is a search because an identifier is not a
//! position; and "this processor's", which is the first of those applied to the
//! processor asking. That last mapping is written once, here, because it is the
//! whole of what makes indexing the array sound.

use alloc::boxed::Box;

use cpu::ApicId;
use spin::Once;

use crate::{VlapicError, registers::Vlapic};

/// Every processor's controller, answering for the one page they share.
#[derive(Debug)]
pub(crate) struct Page {
    lapics: Box<[Vlapic]>,
}

impl Page {
    /// One controller per processor in the roster.
    pub(crate) fn new(lapics: Box<[Vlapic]>) -> Self {
        Self { lapics }
    }

    /// Every controller, for the paths that walk them: delivering to a set of
    /// processors, and reporting.
    pub(crate) fn all(&self) -> &[Vlapic] {
        &self.lapics
    }

    /// The controller belonging to the processor asking.
    ///
    /// An access to the register page is made by the guest running on one
    /// processor, and the controller it means is that processor's. Nothing else
    /// would be a sensible answer: the address carries no processor in it
    /// precisely because the hardware it stands for is per-processor.
    ///
    /// The one place a processor is turned into a row of the array, and so the
    /// one place the soundness of doing that has to hold: the index comes from
    /// the roster the array was built from, and answers `None` rather than
    /// indexing past the end if it somehow did not.
    pub(crate) fn current(&self) -> Option<&Vlapic> {
        // SAFETY: nothing reaches this before the processor has attached — a
        // guest cannot be entered until bring-up is past that point, and the
        // interrupt path is reached only from a processor that has — so this
        // processor's `GS` base points at its own block.
        let index = unsafe { cpu::current() }.index();
        self.lapics.get(index.get())
    }
}

/// This processor's controller.
///
/// # Errors
///
/// [`VlapicError::NotInstalled`] before [`crate::install`], or
/// [`VlapicError::NoLapic`] if the roster does not describe this processor.
pub(crate) fn current() -> Result<&'static Vlapic, VlapicError> {
    lapics()?.current().ok_or(VlapicError::NoLapic)
}

/// The controller belonging to the processor with this identifier.
///
/// A search rather than an index, because an identifier is not a position — and
/// unlike [`current`] it asks nothing of the processor running it, which is
/// what lets a processor claim its own controller before it has joined the
/// machine.
///
/// # Errors
///
/// [`VlapicError::NotInstalled`] before [`crate::install`], or
/// [`VlapicError::NoLapic`] if the roster does not describe that processor.
pub(crate) fn of(id: ApicId) -> Result<&'static Vlapic, VlapicError> {
    lapics()?
        .all()
        .iter()
        .find(|vlapic| vlapic.apic_id() == id)
        .ok_or(VlapicError::NoLapic)
}

/// The controllers, once they exist.
pub(crate) fn lapics() -> Result<&'static Page, VlapicError> {
    LAPICS.get().copied().ok_or(VlapicError::NotInstalled)
}

/// Whether the controllers have been built already.
///
/// Asked by [`crate::install`] before it acquires anything, so that a second
/// call is cheap and leaves nothing behind.
pub(crate) fn installed() -> bool {
    LAPICS.is_completed()
}

/// Builds the device the register page is answered by, and publishes it.
///
/// Leaked rather than owned by the cell, because the device registered for the
/// guest's page needs the same controllers this crate reaches for delivery, and
/// a device is handed over as a boxed trait object. One allocation for the life
/// of the machine is the honest cost of that.
///
/// Answers the controllers and whether this call is the one that built them. A
/// second caller is told rather than silently sharing the first one's, and
/// nothing is allocated for it.
pub(crate) fn publish(lapics: Box<[Vlapic]>) -> (&'static Page, bool) {
    let mut built = false;
    let page = LAPICS.call_once(|| {
        built = true;
        &*Box::leak(Box::new(Page::new(lapics)))
    });
    (page, built)
}

/// Built once, by the boot processor, before any other processor is started.
static LAPICS: Once<&'static Page> = Once::new();
