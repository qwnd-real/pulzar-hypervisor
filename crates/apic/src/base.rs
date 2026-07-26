//! `IA32_APIC_BASE`, the one register that says which interface a processor's
//! local APIC presents.
//!
//! Three things live in it: whether this processor is the one the machine
//! started on, whether the controller is switched on at all, and — the reason
//! this module exists — which of the two interfaces it answers through. It also
//! holds where the memory-mapped register page is, which is the processor's own
//! answer to a question firmware's tables answer separately.
//!
//! # The transition is one-way
//!
//! A controller goes from disabled to the older interface to x2APIC, and never
//! back: the architecture makes going straight from x2APIC to the older
//! interface a general protection fault, and the way back through disabled is
//! not something a running machine can do to itself. So the mode is chosen
//! once, for the whole machine, before any processor is switched, and every
//! processor then makes the same one-way trip for itself.
//!
//! The transition is also checked rather than assumed. A write that the
//! processor accepts and does not act on would otherwise be found out by the
//! first register access faulting, with nothing left to say why.
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

/// Switches this processor's controller into `mode`, if it is not there
/// already.
///
/// # Errors
///
/// [`ApicError::NoApic`] on a processor with no local controller,
/// [`ApicError::NoX2Apic`] if x2APIC was asked for and this processor does not
/// implement it, [`ApicError::ModeRegression`] if the older interface was asked
/// for on a processor already in x2APIC, which the architecture gives no way
/// back from, or [`ApicError::ModeNotEntered`] if the processor took the write
/// and did not change — which is what a controller firmware disabled for good
/// looks like from here.
pub(crate) fn enter(mode: Mode) -> Result<(), ApicError> {
    let current = read().ok_or(ApicError::NoApic)?;
    let wanted = match mode {
        Mode::XApic => {
            if current & X2APIC_ENABLE != 0 {
                return Err(ApicError::ModeRegression);
            }
            current | GLOBAL_ENABLE
        }
        Mode::X2Apic => {
            if !processor::features().contains(Features::X2APIC) {
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
    // Only the two bits this touched: everything else in the register is
    // firmware's and is carried through, and a processor is free to report a
    // reserved bit however it likes.
    presents(mode)
        .then_some(())
        .ok_or(ApicError::ModeNotEntered)
}

/// Whether this processor's controller is switched on and presenting `mode`.
///
/// The register is the only per-processor record of a transition each processor
/// makes for itself, which is what makes it the thing to ask before handing out
/// anything that reaches a controller's registers.
pub(crate) fn presents(mode: Mode) -> bool {
    read().is_some_and(|current| {
        current & GLOBAL_ENABLE != 0
            && (current & X2APIC_ENABLE != 0) == matches!(mode, Mode::X2Apic)
    })
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
/// read without it.
pub(crate) fn read() -> Option<u64> {
    processor::features().contains(Features::APIC).then(|| {
        // SAFETY: `IA32_APIC_BASE` is architectural on every processor whose
        // `CPUID` reports a local APIC, which is what was just checked, and
        // reading it has no side effect.
        unsafe { Msr::new(IA32_APIC_BASE).read() }
    })
}
