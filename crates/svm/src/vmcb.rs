//! The page a guest is described by, and the page the processor puts our own
//! state in while that guest runs.
//!
//! Entering a guest is a single instruction handed one physical address. From
//! that address the processor reads everything: which of the guest's actions
//! come back to us, what register state to give it, and — on the way out —
//! what it did. The whole conversation is this page.
//!
//! It has two halves that behave quite differently. The first kibibyte is
//! control: what we decide about the guest, almost all of it written by us and
//! read by the processor. The three kibibytes after it are the guest's own
//! architectural state, which the processor both loads from and writes back to,
//! so it is as much an output as an input.
//!
//! # Why these are types and not addresses
//!
//! Both pages must be page-aligned, and the processor is given their physical
//! addresses with the low twelve bits ignored — so an unaligned block is not
//! slightly wrong, it is a different block. Making the alignment part of the
//! type means the compiler places these correctly wherever one is put, and no
//! allocation site has to remember. What a caller still owns is getting the
//! *physical* address of one, since these are described to the hardware by
//! physical address while we reach them through a virtual mapping.

use crate::{ControlArea, PAGE_BYTES, SaveArea};

/// Everything the processor is told about one guest.
///
/// The architecture recommends zeroing a newly allocated one, because unused
/// bytes must be zero and are reserved for meanings a later processor may
/// give them. [`Vmcb::zeroed`] is that, and it is the only way to make one —
/// a caller then assigns to the public fields of either half.
#[derive(Clone, Copy, Debug)]
#[repr(C, align(4096))]
pub struct Vmcb {
    /// How the guest runs: what is intercepted, what the processor should
    /// cache, and where it reports what happened.
    pub control: ControlArea,
    /// The register state the guest runs with, loaded on entry and written
    /// back on exit.
    pub save: SaveArea,
}

impl Vmcb {
    /// A control block with every byte zero.
    ///
    /// Not a runnable guest — the intercepts a guest may not run without are
    /// the caller's to set, and an all-zero block names no address space. It is
    /// the correct *starting* point, which is a different claim: every
    /// reserved byte is zero as the architecture requires, and every clean bit
    /// is clear, which is what a control block the processor has never seen
    /// must say.
    #[must_use]
    pub const fn zeroed() -> Self {
        Self {
            control: ControlArea::zeroed(),
            save: SaveArea::zeroed(),
        }
    }
}

layout! {
    Vmcb, size = PAGE_BYTES,
    0x000 => control,
    0x400 => save,
}

/// Where the processor saves our own state while a guest runs.
///
/// Entering a guest is a swap, and this is the other side of it: the processor
/// writes the host's state here on the way in and restores it on the way out.
/// Its address is given once per processor through a model-specific register
/// rather than per guest, because it belongs to the processor rather than to
/// anything running on it.
///
/// The state itself is in the same format and at the same offset as a guest's,
/// which is why this holds a [`SaveArea`] a kibibyte in rather than at the
/// start. Nothing reads the kibibyte before it — this page has no control half
/// — but the offset is architectural, so it is spelled out here rather than
/// left to a caller to remember.
///
/// Its contents are the processor's business, not ours. Software allocates the
/// page, zeroes it and hands over the address; reading the fields back is not
/// how any of this is meant to be used.
#[derive(Clone, Copy, Debug)]
#[repr(C, align(4096))]
pub struct HostSavePage {
    /// Where a guest's control area would be. Nothing uses it here.
    reserved_0x000: [u8; 0x400],
    /// The host state itself, written and restored by the processor.
    save: SaveArea,
}

impl HostSavePage {
    /// A host save page with every byte zero, which is what software is
    /// expected to hand the processor before the first entry into a guest.
    #[must_use]
    pub const fn zeroed() -> Self {
        Self {
            reserved_0x000: [0; 0x400],
            save: SaveArea::zeroed(),
        }
    }
}

layout! {
    HostSavePage, size = PAGE_BYTES,
    0x000 => reserved_0x000,
    0x400 => save,
}
