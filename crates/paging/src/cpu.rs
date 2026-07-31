//! The architectural state the address space depends on.
//!
//! What the processor *can* do is the [`processor`] crate's answer. This module
//! is the rest of it: what has to be programmed on the processor before the
//! subsystem's assumptions about page table entries hold, and what has to be
//! refused outright.
//!
//! Three of the checks are refusals rather than adaptations. Without `NX` there
//! is no way to honour the no-execute requests the rest of the subsystem makes.
//! Under 5-level paging the register the code treats as a PML4 pointer is a
//! PML5 pointer, so proceeding would corrupt an address space rather than build
//! one. With process-context identifiers enabled, invalidating a translation
//! reaches only the current context and every shootdown this crate performs
//! would leave stale entries behind in the others. Refusing to boot beats any
//! of the three.
//!
//! # The page attribute table is per-processor state, not a global setting
//!
//! `PWT` and `PCD` in a page table entry do not name a cache type. They select
//! one of eight entries of `IA32_PAT`, and that register is
//! per-logical-processor and survives `INIT`. Two processors walking the *same*
//! page tables with different `IA32_PAT` values therefore give the same mapping
//! two different memory types, which is an architecturally undefined aliasing
//! of exactly the kind that produces corruption rather than a fault.
//!
//! So there is one published policy, [`PAT_POLICY`], and every processor
//! establishes it before it uses any shared mapping — the boot processor while
//! the address space is being built, each application processor as its first
//! act after reaching 64-bit code. [`establish_pat`] is that step, and it is
//! the same call on both.
//!
//! # Why changing it is not a bare `WRMSR`
//!
//! Writing `IA32_PAT` reinterprets every translation already cached and every
//! line already in the caches: a mapping firmware made with `PCD` set was one
//! memory type before the write and is another after it, while cached lines
//! from it are still around. The architecture prescribes a transition for this,
//! and [`establish_pat`] performs it — no-fill caching, write back and
//! invalidate, flush the translation buffers, write, then unwind the same way —
//! with interrupts masked so nothing observes the middle of it.
//!
//! That is also why establishing the policy is an `unsafe` operation with a
//! contract rather than a function returning a `bool`: it is sound only at a
//! point in bring-up its caller has to guarantee, and no signature can check
//! that.

use core::arch::asm;

use processor::Features;
use x86_64::{
    instructions::{interrupts, tlb},
    registers::{
        control::{Cr0, Cr0Flags, Cr4, Cr4Flags},
        model_specific::{Efer, EferFlags, Msr},
    },
};

use crate::PagingError;

/// Enables `EFER.NXE` so the no-execute bit in page tables is honoured rather
/// than reserved.
///
/// Idempotent, and the already-enabled case does not write: `WRMSR` to `EFER`
/// is a serializing operation that also invalidates cached translations, and
/// every processor runs this on its way up.
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
    if Efer::read().contains(EferFlags::NO_EXECUTE_ENABLE) {
        return Ok(());
    }
    // SAFETY: setting `EFER.NXE` only changes how bit 63 of a page table entry
    // is interpreted, from reserved-must-be-zero to no-execute. The bit was
    // clear a moment ago, so no entry reachable from the active `CR3` can have
    // bit 63 set — an entry that did would already be faulting as a reserved-bit
    // violation — and therefore no existing mapping changes meaning.
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

/// Refuses to continue with process-context identifiers enabled.
///
/// An invariant of this crate's translation shootdowns rather than a limitation
/// of its page tables: writing `CR3` with `PCIDE` set flushes the entered
/// context and no other, and the ranged invalidations this crate issues carry
/// no context. A machine running with `PCIDE` would keep stale translations in
/// every context it was not currently in, which is precisely the failure a
/// shootdown exists to prevent.
///
/// Nothing in pulzar sets `PCIDE`, so this asserts a property rather than
/// changing one — but it is asserted on every processor's way up, because the
/// cost of being wrong is silent.
///
/// # Errors
///
/// [`PagingError::PcidEnabled`] if `CR4.PCIDE` is set.
pub fn refuse_process_context_identifiers() -> Result<(), PagingError> {
    if Cr4::read().contains(Cr4Flags::PCID) {
        return Err(PagingError::PcidEnabled);
    }
    Ok(())
}

/// `IA32_PAT`, whose eight entries name the cache type each `PAT:PCD:PWT`
/// combination in a page table entry selects.
const IA32_PAT: u32 = 0x277;

/// The one page attribute table layout every processor in the machine runs
/// with: write-back, write-through, uncached-minus, uncached, repeated for the
/// upper four entries.
///
/// It is the architecture's own layout after reset, chosen for that reason: it
/// is what firmware most likely already has, so establishing it usually writes
/// nothing at all, and it is the layout under which `PCD` and `PWT` alone name
/// a cache type. That in turn is why [`crate::CacheType`] needs no `PAT` bit,
/// whose position differs between 4 KiB and large pages.
pub const PAT_POLICY: u64 = 0x0007_0406_0007_0406;

/// Establishes [`PAT_POLICY`] on the calling processor.
///
/// Reports whether it had to be written, which is worth logging on the boot
/// processor: a firmware that reprograms the page attribute table is unusual
/// enough to want a record of, and an application processor that needs a write
/// when the boot processor did not means the two started from different state.
///
/// Nothing is written when the register already holds the policy, which is the
/// common case and the only one that costs nothing.
///
/// # Errors
///
/// [`PagingError::PatUnsupported`] if the processor does not implement the page
/// attribute table, in which case `PWT` and `PCD` mean something this crate
/// does not model and no mapping it makes would have the cache type it asked
/// for.
///
/// # Safety
///
/// The calling processor must be in bring-up: it must not yet be using any
/// mapping whose `PWT`/`PCD` bits select an entry whose meaning this changes,
/// its own code and stack must be reached through write-back mappings, and no
/// other processor may be changing memory types at the same time. The
/// transition below leaves the processor with caching disabled for the length
/// of two `WBINVD`s, so it must also not be holding anything time-critical
/// open.
///
/// On the boot processor these hold while the address space is being built,
/// before anything of pulzar's is mapped. On an application processor they hold
/// at its entry point, before it maps or touches anything of its own.
pub unsafe fn establish_pat() -> Result<bool, PagingError> {
    if !processor::features().contains(Features::PAGE_ATTRIBUTE_TABLE) {
        return Err(PagingError::PatUnsupported);
    }
    let mut pat = Msr::new(IA32_PAT);
    // SAFETY: the feature check above proves `IA32_PAT` exists, and reading an
    // MSR has no side effects.
    if unsafe { pat.read() } == PAT_POLICY {
        return Ok(false);
    }
    interrupts::without_interrupts(|| {
        // SAFETY: the caller's contract carries the whole of this — bring-up,
        // write-back code and stack, no concurrent memory-type change — and
        // interrupts are masked, so nothing runs between the steps.
        unsafe { transition(&mut pat) };
    });
    Ok(true)
}

/// Writes [`PAT_POLICY`] with the cache and translation-buffer transition the
/// architecture requires around a change of memory-type settings.
///
/// No-fill caching first so nothing new is cached from mappings whose type is
/// about to change; write back and invalidate so nothing already cached under
/// the old interpretation survives; flush translations so no cached entry
/// carries the old type with it. Then the write, then the same two steps again
/// before caching is restored, so the processor resumes with nothing left over
/// from either side of the change.
///
/// # Safety
///
/// As [`establish_pat`], and interrupts must already be masked: the processor
/// spends the middle of this with caching disabled and with no valid cached
/// translations, and a handler entered there would run under memory types that
/// are in the middle of changing.
unsafe fn transition(pat: &mut Msr) {
    let cr0 = Cr0::read();
    // SAFETY: `CD` set with `NW` clear is the architecture's no-fill mode, valid
    // at any time; the original value is restored below.
    unsafe {
        Cr0::write(
            cr0.union(Cr0Flags::CACHE_DISABLE)
                .difference(Cr0Flags::NOT_WRITE_THROUGH),
        );
    }
    write_back_and_invalidate();
    flush_translations();
    // SAFETY: every field of `PAT_POLICY` encodes a memory type the architecture
    // defines, so the write cannot fault, and the caches and translation buffers
    // have just been emptied of anything that depended on the old value.
    unsafe { pat.write(PAT_POLICY) };
    write_back_and_invalidate();
    flush_translations();
    // SAFETY: restoring the value read a moment ago, on the same processor.
    unsafe { Cr0::write(cr0) };
}

/// Writes every dirty cache line back and invalidates the caches.
fn write_back_and_invalidate() {
    // SAFETY: `WBINVD` is valid in ring 0 on every processor this runs on and
    // has no operands. It is slow and it is architecturally required here: the
    // lines it writes back were cached under a memory-type interpretation that
    // is about to stop applying.
    unsafe { asm!("wbinvd", options(nostack, preserves_flags)) };
}

/// Drops every cached translation this processor has, global ones included.
///
/// Reloading the page table root leaves global translations behind, and
/// firmware is free to have marked its own mappings global — so those are
/// reached the only way they can be, by taking global translations away for as
/// long as it takes the processor to notice.
///
/// The one implementation of "drop everything", shared by the page attribute
/// table transition, by the shootdown that answers [`crate::shootdown::Flush`]
/// with no range, and by dropping the firmware half of the address space.
pub(crate) fn flush_translations() {
    tlb::flush_all();
    let cr4 = Cr4::read();
    if cr4.contains(Cr4Flags::PAGE_GLOBAL) {
        // SAFETY: clearing `CR4.PGE` invalidates all global translations and is
        // architecturally permitted at any time; the original value is restored
        // immediately, so nothing observes the intermediate state.
        unsafe {
            Cr4::write(cr4.difference(Cr4Flags::PAGE_GLOBAL));
            Cr4::write(cr4);
        }
    }
}
