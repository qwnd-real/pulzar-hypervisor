//! Firmware's segment registers and descriptor tables, in the form a guest is
//! entered with.
//!
//! A segment register is not its selector. What the processor uses is the
//! descriptor that selector named at the moment it was loaded — a base, a limit
//! and a set of attributes — held in registers software cannot read back. So
//! the only way to say what firmware's segments are is to read the selectors,
//! find the descriptors they came from in firmware's own tables, and decode
//! them. That is the one part of this capture that is more than a register
//! read, and it is the part with somewhere to go wrong.
//!
//! # Three ways to get it wrong
//!
//! The attributes are not a slice of the descriptor. A descriptor in memory
//! keeps them in two runs with the upper nibble of the limit between them, and
//! the save area drops that gap. Copying the run across unchanged puts every
//! attribute above the present bit four positions too high, which the processor
//! does not reject — so they are decoded a field at a time here.
//!
//! The limit is stored already scaled. A descriptor whose granularity bit is
//! set counts pages; the save area counts bytes, and records the bit rather
//! than leaving it to be applied a second time.
//!
//! And in long mode a descriptor does not hold the base of `FS` or `GS`. Theirs
//! is a full sixty-four bits, in a model-specific register, of which the
//! descriptor keeps the lower thirty-two — usually zero, so trusting the
//! descriptor gives an answer that is wrong only on the machines where it
//! matters.

use core::arch::asm;

use bitfield_struct::bitfield;
use svm::{SaveArea, Segment, SegmentAttributes};
use x86_64::{
    VirtAddr,
    instructions::tables::{sgdt, sidt},
    registers::{
        model_specific::{FsBase, GsBase},
        segmentation::{CS, DS, ES, FS, GS, SS, Segment as _, SegmentSelector},
    },
    structures::DescriptorTablePointer,
};

/// Fills in both descriptor tables, every segment register, and the privilege
/// level firmware is running at.
///
/// # Safety
///
/// The descriptor tables the processor's own registers name must be readable in
/// the active address space. That holds of any processor running under the
/// address space its tables were loaded in, which is what makes this sound to
/// call at all.
pub(crate) unsafe fn capture(cpu: &mut SaveArea) {
    let gdt = sgdt();
    cpu.gdtr = table(&gdt);
    cpu.idtr = table(&sidt());

    // The local table is resolved first because it is where a selector with its
    // table bit set finds its descriptor, and its own descriptor in the global
    // table is the only thing that says where it is.
    //
    // SAFETY: as this function's contract. Both of these selectors name the
    // global table, which the register the processor just handed over describes.
    let local = unsafe { system(&gdt, local_descriptor_table()) };
    // SAFETY: as above.
    let task = unsafe { system(&gdt, task_register()) };
    cpu.ldtr = local;
    cpu.tr = task;

    let tables = Tables { global: gdt, local };
    // SAFETY: as above, and each selector below came out of the register that
    // is using it, so the table it names is the one it was resolved against
    // when it was loaded.
    unsafe {
        cpu.es = tables.resolve(ES::get_reg());
        cpu.cs = tables.resolve(CS::get_reg());
        cpu.ss = tables.resolve(SS::get_reg());
        cpu.ds = tables.resolve(DS::get_reg());
        cpu.fs = tables.resolve(FS::get_reg());
        cpu.gs = tables.resolve(GS::get_reg());
    }

    // The descriptor holds thirty-two bits of these two bases and the register
    // holds all sixty-four, so the register is what the segment actually has.
    cpu.fs.base = FsBase::read().as_u64();
    cpu.gs.base = GsBase::read().as_u64();

    // Read from the code selector rather than from any descriptor's privilege
    // field, which is the level the descriptor was written with rather than the
    // one the processor is running at.
    cpu.cpl = privilege(CS::get_reg());
}

/// The two tables a selector can name.
struct Tables {
    global: DescriptorTablePointer,
    local: Segment,
}

impl Tables {
    /// The segment `selector` names, or a NULL segment where it names nothing
    /// that can be resolved — which covers the NULL selector itself, an index
    /// past the end of its table, and a local table register that names none.
    ///
    /// # Safety
    ///
    /// As [`capture`].
    unsafe fn resolve(&self, selector: SegmentSelector) -> Segment {
        if is_null(selector) {
            return Segment::null();
        }
        let (base, limit) = if selector.0 & LOCAL_TABLE == 0 {
            (self.global.base.as_u64(), u32::from(self.global.limit))
        } else {
            (self.local.base, self.local.limit)
        };
        // SAFETY: the caller guarantees the tables are readable, and a table the
        // processor is not using has a zero limit here, which admits no read.
        unsafe { quadword(base, limit, offset_of(selector)) }
            .map_or_else(Segment::null, |low| decode(selector, low, 0))
    }
}

/// The segment a system selector names.
///
/// In long mode the descriptors behind the local descriptor table register and
/// the task register are sixteen bytes rather than eight, because their base is
/// a full sixty-four bits: the upper half lies in a second quadword the older
/// descriptor never had. Both halves have to be inside the table's limit or
/// neither is read.
///
/// # Safety
///
/// As [`capture`].
unsafe fn system(gdt: &DescriptorTablePointer, selector: SegmentSelector) -> Segment {
    if is_null(selector) {
        return Segment::null();
    }
    let base = gdt.base.as_u64();
    let limit = u32::from(gdt.limit);
    let offset = offset_of(selector);
    // SAFETY: the caller guarantees the global table is readable, and each half
    // is bounds-checked against the table's own limit before it is read.
    let halves = unsafe {
        quadword(base, limit, offset).zip(quadword(base, limit, offset + DESCRIPTOR_BYTES))
    };
    halves.map_or_else(Segment::null, |(low, high)| {
        decode(selector, low, upper_base(high))
    })
}

/// A descriptor-table register in the form the save area holds it.
///
/// No selector and no attributes, because a table register has neither: the
/// save area describes it with the same structure as a segment and leaves most
/// of that structure unused.
fn table(pointer: &DescriptorTablePointer) -> Segment {
    Segment {
        selector: 0,
        attributes: SegmentAttributes::new(),
        limit: u32::from(pointer.limit),
        base: pointer.base.as_u64(),
    }
}

/// The quadword at `offset` in the table at `base`, or `None` if the table's
/// limit does not admit all eight of its bytes.
///
/// # Safety
///
/// `base` must be the base of a descriptor table readable in the active address
/// space, and `limit` that table's own limit.
unsafe fn quadword(base: u64, limit: u32, offset: u64) -> Option<u64> {
    // A limit is the last valid offset, so a quadword fits only if its final
    // byte is still inside it.
    if offset + DESCRIPTOR_BYTES - 1 > u64::from(limit) {
        return None;
    }
    let address = VirtAddr::try_new(base.wrapping_add(offset)).ok()?;
    // SAFETY: the caller guarantees the table is readable, and the check above
    // keeps this inside the limit the table itself declares. The read is
    // unaligned because a descriptor table promises no alignment beyond its
    // entries, and firmware's is not this code's to assume anything about.
    Some(unsafe { address.as_ptr::<u64>().read_unaligned() })
}

/// One descriptor in the form the save area holds it.
fn decode(selector: SegmentSelector, low: u64, upper_base: u32) -> Segment {
    let descriptor = Descriptor::from_bits(low);
    let limit = descriptor.limit_low() | (descriptor.limit_high() << LIMIT_HIGH_SHIFT);
    Segment {
        selector: selector.0,
        // A field at a time, not a run of bits: the save area stores the two
        // attribute runs adjacent and the descriptor stores them four positions
        // apart, so moving them across as one value silently misplaces
        // everything above the present bit.
        attributes: SegmentAttributes::new()
            .with_kind(descriptor.kind())
            .with_descriptor(descriptor.descriptor())
            .with_dpl(descriptor.dpl())
            .with_present(descriptor.present())
            .with_available(descriptor.available())
            .with_long(descriptor.long())
            .with_default_size(descriptor.default_size())
            .with_granularity(descriptor.granularity()),
        // Scaled here, because the save area's limit counts bytes whatever the
        // descriptor counted, and the granularity bit above records only what
        // the descriptor said rather than something still to apply.
        limit: if descriptor.granularity() {
            (limit << PAGE_SHIFT) | PAGE_OFFSET_MASK
        } else {
            limit
        },
        base: u64::from(descriptor.base_low())
            | u64::from(descriptor.base_high()) << BASE_HIGH_SHIFT
            | u64::from(upper_base) << BASE_UPPER_SHIFT,
    }
}

/// A descriptor as it lies in a table.
///
/// Declared the way the architecture scatters it — the base in two runs, the
/// limit in two more, the attributes split around the upper limit nibble — so
/// that reassembling any of them is reading fields rather than writing out a
/// shift and a mask for each piece.
#[bitfield(u64)]
struct Descriptor {
    #[bits(16)]
    limit_low: u32,
    #[bits(24)]
    base_low: u32,
    #[bits(4)]
    kind: u8,
    descriptor: bool,
    #[bits(2)]
    dpl: u8,
    present: bool,
    #[bits(4)]
    limit_high: u32,
    available: bool,
    long: bool,
    default_size: bool,
    granularity: bool,
    #[bits(8)]
    base_high: u32,
}

/// The local descriptor table register's selector.
///
/// Read by instruction because there is no wrapper for it: a 64-bit kernel has
/// no reason to ask, which is exactly why a snapshot of somebody else's machine
/// has to.
fn local_descriptor_table() -> SegmentSelector {
    let selector: u16;
    // SAFETY: `SLDT` stores the local descriptor table register into its
    // operand. It is unprivileged, touches no memory, disturbs no flags, and
    // affects nothing but the register it is asked to write into.
    unsafe { asm!("sldt {:x}", out(reg) selector, options(nomem, nostack, preserves_flags)) };
    SegmentSelector(selector)
}

/// The task register's selector.
fn task_register() -> SegmentSelector {
    let selector: u16;
    // SAFETY: as `local_descriptor_table`. `STR` differs only in which register
    // it stores.
    unsafe { asm!("str {:x}", out(reg) selector, options(nomem, nostack, preserves_flags)) };
    SegmentSelector(selector)
}

/// Where in its table the descriptor `selector` names begins.
const fn offset_of(selector: SegmentSelector) -> u64 {
    (selector.0 >> INDEX_SHIFT) as u64 * DESCRIPTOR_BYTES
}

/// Whether a selector is the NULL one.
///
/// Index zero *of the global table*, which is why this is not merely a test of
/// the index: index zero of a local table is an ordinary descriptor.
const fn is_null(selector: SegmentSelector) -> bool {
    selector.0 & !PRIVILEGE == 0
}

/// The privilege level a selector was loaded at.
const fn privilege(selector: SegmentSelector) -> u8 {
    // Masked to the field's own two bits, so nothing can be lost narrowing it.
    (selector.0 & PRIVILEGE) as u8
}

/// The upper half of a system descriptor's base: the low doubleword of the
/// second quadword.
const fn upper_base(high: u64) -> u32 {
    (high & UPPER_BASE_MASK) as u32
}

/// Bytes one descriptor occupies. System descriptors take two of these;
/// everything a segment register can hold takes one.
const DESCRIPTOR_BYTES: u64 = 8;

/// Bits a selector's index is shifted by, above its table and privilege fields.
const INDEX_SHIFT: u16 = 3;

/// The bit that says a selector names the local table rather than the global
/// one.
const LOCAL_TABLE: u16 = 1 << 2;

/// A selector's requested privilege level.
const PRIVILEGE: u16 = 0b11;

/// Bits the upper nibble of a limit is shifted by once rejoined with the rest.
const LIMIT_HIGH_SHIFT: u32 = 16;

/// Bits a page count is shifted by to become a byte count.
const PAGE_SHIFT: u32 = 12;

/// The bytes inside the last page a scaled limit covers, which it includes.
const PAGE_OFFSET_MASK: u32 = 0xFFF;

/// Bits the top byte of a descriptor's own base is shifted by.
const BASE_HIGH_SHIFT: u32 = 24;

/// Bits the upper half of a system descriptor's base is shifted by.
const BASE_UPPER_SHIFT: u32 = 32;

/// The half of a system descriptor's second quadword that is base.
const UPPER_BASE_MASK: u64 = 0xFFFF_FFFF;
