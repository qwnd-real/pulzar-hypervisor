//! `IA32_APIC_BASE`, the one register that says which interface a processor's
//! local APIC presents.
//!
//! Three things live in it: whether this processor is the one the machine
//! started on, whether the controller is switched on at all, and — the reason
//! this module exists — which of the two interfaces it answers through.
//!
//! # The transition is one-way
//!
//! A controller goes from disabled to the older interface to x2APIC, and never
//! back: the architecture makes going straight from x2APIC to the older
//! interface a general protection fault, and the way back through disabled is
//! not something a running machine can do to itself. So the mode is chosen once,
//! for the whole machine, before any processor is switched, and every processor
//! then makes the same one-way trip for itself.
//!
//! The base address in it is deliberately never written. Firmware chose where
//! the register page lives and told ACPI, and moving it would only mean the
//! table and the hardware disagreed.

use x86_64::registers::model_specific::Msr;

use crate::{ApicError, Mode};

/// The register itself.
const IA32_APIC_BASE: u32 = 0x1B;

/// This processor is the one the machine started on.
const BOOTSTRAP: u64 = 1 << 8;

/// The controller answers through model-specific registers.
const X2APIC_ENABLE: u64 = 1 << 10;

/// The controller is switched on. Clearing this is what the architecture calls
/// disabling it, and on many processors it cannot be set again.
const GLOBAL_ENABLE: u64 = 1 << 11;

/// Which interface this processor's controller is presenting right now.
#[must_use]
pub(crate) fn mode() -> Mode {
    let value = read();
    if value & X2APIC_ENABLE != 0 {
        Mode::X2Apic
    } else {
        Mode::XApic
    }
}

/// Whether this is the processor the machine started on.
///
/// Firmware sets this at reset and it is the only thing that distinguishes the
/// boot processor from the rest without asking ACPI.
#[must_use]
pub fn is_bootstrap() -> bool {
    read() & BOOTSTRAP != 0
}

/// Physical base of the memory-mapped register page, as this processor reports
/// it.
///
/// Read for checking against what firmware's table said rather than for using:
/// the two disagreeing means something moved the controller, which is worth
/// knowing about.
#[must_use]
pub fn page() -> u64 {
    read() & PAGE_MASK
}

/// The bits of the register that hold the page's physical address.
const PAGE_MASK: u64 = 0x000F_FFFF_FFFF_F000;

/// Switches this processor's controller into `mode`, if it is not there
/// already.
///
/// # Errors
///
/// [`ApicError::NoX2Apic`] if x2APIC was asked for and this processor does not
/// implement it, or [`ApicError::ModeRegression`] if the older interface was
/// asked for on a processor already in x2APIC, which the architecture gives no
/// way back from.
pub(crate) fn enter(mode: Mode) -> Result<(), ApicError> {
    let current = read();
    let wanted = match mode {
        Mode::XApic => {
            if current & X2APIC_ENABLE != 0 {
                return Err(ApicError::ModeRegression);
            }
            current | GLOBAL_ENABLE
        }
        Mode::X2Apic => {
            if !processor::features().contains(processor::Features::X2APIC) {
                return Err(ApicError::NoX2Apic);
            }
            // Both bits at once. The architecture defines x2APIC-enabled with
            // the controller disabled as an invalid state and faults on the
            // attempt to reach it, so the two are never written apart.
            current | GLOBAL_ENABLE | X2APIC_ENABLE
        }
    };
    if wanted != current {
        // SAFETY: the value differs from what the register already holds only in
        // the two enable bits, and the combination is one the architecture
        // defines — the invalid one is refused above. The base address bits are
        // carried through untouched, so the controller does not move.
        unsafe { Msr::new(IA32_APIC_BASE).write(wanted) };
    }
    Ok(())
}

/// The register's current value.
fn read() -> u64 {
    // SAFETY: `IA32_APIC_BASE` is architectural on every processor that has a
    // local APIC, which `Features::APIC` is checked for before anything here
    // runs, and reading it has no side effect.
    unsafe { Msr::new(IA32_APIC_BASE).read() }
}
