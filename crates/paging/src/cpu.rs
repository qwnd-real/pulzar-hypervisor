//! The architectural state the address space depends on.
//!
//! What the processor *can* do is the [`processor`] crate's answer. This module
//! is the rest of it: what has to be programmed on the processor before the
//! subsystem's assumptions about page table entries hold, and what has to be
//! refused outright.
//!
//! Everything here happens once, during bring-up, before any mapping exists.
//! Two of the checks are refusals rather than adaptations: without `NX` there
//! is no way to honour the no-execute requests the rest of the subsystem makes,
//! and under 5-level paging the register the code treats as a PML4 pointer is a
//! PML5 pointer, so proceeding would corrupt an address space rather than build
//! one. Refusing to boot beats either.

use processor::Features;
use x86_64::registers::{
    control::{Cr4, Cr4Flags},
    model_specific::{Efer, EferFlags, Msr},
};

use crate::PagingError;

/// Enables `EFER.NXE` so the no-execute bit in page tables is honoured rather
/// than reserved.
///
/// # Errors
///
/// [`PagingError::NoExecuteUnsupported`] if the processor does not implement
/// `NX`, which would leave every mapping this crate marks non-executable
/// executable instead.
pub fn enable_no_execute() -> Result<(), PagingError> {
    if !processor::features().contains(Features::NO_EXECUTE) {
        return Err(PagingError::NoExecuteUnsupported);
    }
    // SAFETY: setting `EFER.NXE` only changes how bit 63 of a page table entry
    // is interpreted, from reserved-must-be-zero to no-execute. No entry
    // currently in CR3 can have it set, precisely because it is reserved while
    // the bit is clear, so no existing mapping changes meaning.
    unsafe { Efer::update(|flags| flags.insert(EferFlags::NO_EXECUTE_ENABLE)) };
    Ok(())
}

/// Refuses to continue under 5-level paging.
///
/// # Errors
///
/// [`PagingError::FiveLevelPaging`] if `CR4.LA57` is set.
pub fn refuse_five_level_paging() -> Result<(), PagingError> {
    if Cr4::read().contains(Cr4Flags::L5_PAGING) {
        return Err(PagingError::FiveLevelPaging);
    }
    Ok(())
}

/// `IA32_PAT`, whose eight entries name the cache type each `PAT:PCD:PWT`
/// combination in a page table entry selects.
const IA32_PAT: u32 = 0x277;

/// The layout the architecture leaves in `IA32_PAT` after reset: write-back,
/// write-through, uncached-minus, uncached, repeated for the upper four
/// entries.
///
/// The subsystem programs this rather than trusting what it inherits, so that
/// `PCD` and `PWT` alone name a cache type. That in turn is why cache selection
/// needs no `PAT` bit, whose position differs between 4 KiB and large pages.
const DEFAULT_PAT: u64 = 0x0007_0406_0007_0406;

/// Programs the architectural default `IA32_PAT` layout if firmware left
/// something else there.
///
/// Returns whether it had to be rewritten, which is worth logging: a firmware
/// that reprograms the PAT is unusual enough to want a record of.
#[must_use]
pub fn ensure_default_pat() -> bool {
    let mut pat = Msr::new(IA32_PAT);
    // SAFETY: `IA32_PAT` is architectural on every processor with PAT support,
    // which every 64-bit processor has, and reading an MSR has no side effects.
    if unsafe { pat.read() } == DEFAULT_PAT {
        return false;
    }
    // SAFETY: the value is the architecture's own reset layout, so every field
    // encodes a valid memory type and no #GP is possible. It is written before
    // this crate maps anything, and the only mappings that exist are
    // firmware's, which are write-back — the type entry 0 keeps.
    unsafe { pat.write(DEFAULT_PAT) };
    true
}
