//! The blob a processor being started executes, and the block of parameters it
//! reads.
//!
//! The assembly next to this file contains no address of any kind. Everything
//! it needs is written here, into a structure at a fixed offset in the same
//! page, by the processor doing the starting — which is the only one that knows
//! where the page ended up, where the page tables are, and which stack this
//! particular processor is to run on.
//!
//! The two sides agree about the layout without either of them writing it down
//! twice: the field offsets are computed here with [`offset_of`] and handed to
//! the assembler as constants. A field moving changes both at once, and a field
//! the assembly names that this structure does not have will not assemble. How
//! large the blob is and where its stages begin come back the other way, as
//! data the assembler emits outside the blob — so neither side subtracts one of
//! the other's addresses.
//!
//! # The descriptor table it loads
//!
//! Four entries, built here rather than assembled, because one of them has to
//! contain an address. The 32-bit code segment is based at the page, which is
//! what lets the first far jump name its target as an offset the assembler can
//! work out instead of a linear address only this code knows. The other two are
//! flat: the data segment describes all four gigabytes, and the 64-bit code
//! segment's base is ignored by the mode it exists to enter. One data segment
//! is enough — 64-bit mode ignores what the stack segment describes, so the
//! selector loaded in 32-bit mode is still the right one afterwards.
//!
//! All three usable descriptors are marked accessed. Loading a selector is how
//! a processor sets that bit, and it sets it by *writing the table* — which is
//! in the trampoline page, mapped read-only where the processor executes from.
//! The ordering is what makes that safe: supervisor write protection goes on in
//! the 64-bit stage, after the last of these descriptors has been loaded.
//! Presetting the bit is the belt to that braces, and is why nothing here
//! depends on whether a given processor bothers to check the bit before writing
//! it.
//!
//! # How far a processor got
//!
//! One word of the block, written by the processor being started as it passes
//! each stage that still can write — which means the stages before paging is
//! turned on. Once it is, this page is read-only to the processor executing
//! from it, and the last thing the trampoline does before entering Rust is make
//! that stick by turning supervisor write protection on.
//!
//! It answers whether the processor ever began, which decides whether a second
//! startup command is worth sending and whether the block is still someone's;
//! and where it stopped, which is the only diagnosis available, since there is
//! no interrupt descriptor table until Rust builds one and a fault before then
//! is a reset with nothing written down.

use core::{
    mem::offset_of,
    sync::atomic::{AtomicU32, Ordering},
};

use x86_64::PhysAddr;

use crate::{ApicError, PAGE};

/// Where the parameter block sits in the page.
///
/// Far enough in that the code has room, aligned enough for the eight-byte
/// fields in it, and inside the page with the whole block to spare. That the
/// code really does fit below it is checked rather than assumed, in
/// [`Trampoline::place`]; that the block itself fits inside the page is checked
/// below.
const PARAMETERS: usize = 0x800;

/// How many descriptors the trampoline's own table holds.
const DESCRIPTORS: u16 = 4;

/// The limit `lgdt` is given for it, which the processor reads as one less than
/// the table's size.
const TABLE_LIMIT: u16 = DESCRIPTORS * 8 - 1;

/// The parameter block is reached through a pointer formed by adding
/// [`PARAMETERS`] to a frame-aligned address, so the offset has to carry the
/// alignment on its own.
const _: () = assert!(
    PARAMETERS.is_multiple_of(align_of::<Parameters>()),
    "the parameter block's offset must carry its own alignment"
);

/// And it has to end inside the page as well as begin inside it. Adding a field
/// is otherwise a way to make [`Trampoline::place`] write past the one page its
/// caller promised.
const _: () = assert!(
    PARAMETERS + size_of::<Parameters>() <= paging::as_usize(PAGE),
    "the parameter block must fit in the trampoline page"
);

/// Selector for the 32-bit code segment: the first descriptor after the null
/// one.
const CODE32_SELECTOR: u16 = 8;

/// Selector for the flat data segment.
const DATA32_SELECTOR: u16 = 16;

/// Selector for the 64-bit code segment.
const CODE64_SELECTOR: u16 = 24;

/// `CR0.PE`, which makes the segment registers select descriptors.
const CR0_PROTECTED: u32 = 1 << 0;

/// `CR0.PG`, which turns the page tables on.
const CR0_PAGING: u32 = 1 << 31;

/// `CR0.WP`, which makes a read-only page read-only for privileged code too.
///
/// Set on its own, in 64-bit mode, and that is the whole of why it is a
/// separate constant. It cannot be set with paging: the far jump that follows
/// that write loads a descriptor out of the table in the trampoline page, and
/// loading a descriptor is how a processor sets its accessed bit — a write, to
/// a page mapped read-only where the trampoline executes from. The descriptors
/// carry that bit already set so the write is unnecessary, but no architecture
/// promises a processor checks before writing, and a fault there is a triple
/// fault: there is no interrupt descriptor table yet.
const CR0_WRITE_PROTECT: u32 = 1 << 16;

/// `CR0.MP`, which says there is a coprocessor to fault for.
const CR0_MONITOR_COPROCESSOR: u32 = 1 << 1;

/// What this processor's `CR0` is to have set beyond what it is already running
/// with.
const CR0_SET: u32 = CR0_MONITOR_COPROCESSOR;

/// What it is to have cleared: `CR0.EM`, which makes every vector instruction
/// fault, and the two cache bits.
///
/// A processor comes out of reset with the emulate bit set, and the hypervisor
/// reaches the vector registers by hand to move a guest's. The cache bits are
/// the two `INIT` leaves alone, so a processor firmware had running with its
/// caches off would keep them off for the rest of its life here; they are
/// cleared together because the architecture makes no-fill-with-write-through
/// an invalid combination and faults on it.
const CR0_CLEAR: u32 = !((1 << 2) | (1 << 30) | (1 << 29));

/// `CR4.PAE`, which long mode requires, together with the two bits that make
/// the vector registers usable and their exceptions reportable.
const CR4_LONG_MODE: u32 = (1 << 5) | (1 << 9) | (1 << 10);

/// The register holding long mode enable and no-execute enable.
const IA32_EFER: u32 = 0xC000_0080;

/// `EFER.LME` and `EFER.NXE`. The second has to be set before the page tables
/// are loaded, not after: they have the no-execute bit set in them, and it is
/// reserved-must-be-zero until this is written.
const EFER_LONG_MODE: u32 = (1 << 8) | (1 << 11);

/// Bytes the 64-bit stage takes off the stack before jumping into Rust.
///
/// A return slot and the four argument slots this target's calling convention
/// makes every caller reserve, whether or not the callee has four arguments to
/// put in them. Forty bytes leaves the stack pointer eight past a sixteen-byte
/// boundary, which is where a call would have left it.
const ENTRY_FRAME: u32 = 8 + 4 * 8;

/// The stack the entry point is given has to start on a sixteen-byte boundary
/// for that to be true, which a page-aligned one does.
const _: () = assert!(
    ENTRY_FRAME.is_multiple_of(8) && !ENTRY_FRAME.is_multiple_of(16),
    "the entry frame must leave the stack pointer where a call would have"
);

core::arch::global_asm!(
    include_str!("trampoline.s"),
    STAGE = const PARAMETERS + offset_of!(Parameters, stage),
    STAGE_REAL = const Stage::REAL,
    STAGE_PROTECTED_MODE = const Stage::PROTECTED,
    TABLE_POINTER = const PARAMETERS + offset_of!(Parameters, pointer),
    PROTECTED_ENTRY = const PARAMETERS + offset_of!(Parameters, protected_entry),
    LONG_ENTRY = const PARAMETERS + offset_of!(Parameters, long_entry),
    PAGE_BASE = const PARAMETERS + offset_of!(Parameters, page_base),
    PAGE_TABLE_ROOT = const PARAMETERS + offset_of!(Parameters, page_table_root),
    STACK_TOP = const PARAMETERS + offset_of!(Parameters, stack_top),
    ENTRY = const PARAMETERS + offset_of!(Parameters, entry),
    ENTRY_FRAME = const ENTRY_FRAME,
    DATA32_SELECTOR = const DATA32_SELECTOR,
    CR0_PROTECTED = const CR0_PROTECTED,
    CR0_PAGING = const CR0_PAGING,
    CR0_WRITE_PROTECT = const CR0_WRITE_PROTECT,
    CR0_SET = const CR0_SET,
    CR0_CLEAR = const CR0_CLEAR,
    CR4_LONG_MODE = const CR4_LONG_MODE,
    IA32_EFER = const IA32_EFER,
    EFER_LONG_MODE = const EFER_LONG_MODE,
);

unsafe extern "C" {
    /// First byte of the blob, and where a started processor begins.
    #[link_name = "pulzar_trampoline_start"]
    static BLOB_START: u8;
    /// How large the blob is and where its stages begin, emitted by the
    /// assembler beside it rather than derived from label addresses here.
    #[link_name = "pulzar_trampoline_extent"]
    static EXTENT: Extent;
}

/// How far a processor being started has got.
///
/// Not a Rust enum, because the value is written by the processor being started
/// with an ordinary instruction and an enum whose invariant that breaks would
/// be worse than a number. What is asked of it is how far along it is, so it is
/// compared rather than matched, which is what these have to be in order for.
struct Stage;

impl Stage {
    /// Nothing has run yet. What [`Trampoline::prepare`] leaves behind.
    const WAITING: u32 = 0;
    /// The first instructions, in real mode, before anything that can fail.
    const REAL: u32 = 1;
    /// Protected mode, before the control registers and the page tables — and
    /// the last stage that can say anything, because the next thing the
    /// processor does is turn paging on and make this page read-only to itself.
    const PROTECTED: u32 = 2;
}

/// Every stage a processor passes through is past the one before it.
const _: () = assert!(
    Stage::WAITING < Stage::REAL && Stage::REAL < Stage::PROTECTED,
    "the stages have to be ordered for `began` to mean anything"
);

/// How large the blob is, and how far into it each stage begins.
///
/// `usize` rather than `u64` because these are a length and two offsets, which
/// is what the assembler emits pointer-sized words of.
#[repr(C)]
struct Extent {
    bytes: usize,
    protected: usize,
    long: usize,
}

/// The pointer `lgdt` is given.
///
/// Packed because the architecture defines it as a 16-bit limit immediately
/// followed by the base, with no padding between them.
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
struct TablePointer {
    limit: u16,
    base: u32,
}

/// A far jump target: where to go, and what to be when it gets there.
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
struct FarPointer {
    offset: u32,
    selector: u16,
}

/// Everything the blob reads, written by the processor doing the starting.
///
/// `repr(C)` because the assembly names these fields by offset, and the offsets
/// it uses are computed from this declaration.
#[repr(C)]
struct Parameters {
    /// Null, 32-bit code based at the page, flat data, 64-bit code.
    gdt: [u64; 4],
    /// What `lgdt` is given for `gdt`.
    pointer: TablePointer,
    /// Where the first mode transition goes. An offset into the blob, because
    /// the 32-bit code segment it enters is based at this page.
    protected_entry: FarPointer,
    /// Where the second goes. A linear address, because a 64-bit code segment
    /// has no base to make an offset relative to.
    long_entry: FarPointer,
    /// Linear address of this page, which is how the 32-bit and 64-bit stages
    /// reach this block once the segments stop being based at it.
    page_base: u64,
    /// Physical address of the page tables every processor shares. Below four
    /// gigabytes, because the instruction that loads it is a 32-bit one.
    page_table_root: u64,
    /// Top of the stack the processor being started is to run on. One
    /// processor's, rewritten before each start.
    stack_top: u64,
    /// The 64-bit entry point it jumps to, which never returns.
    entry: u64,
    /// How far the processor being started has got. Written by it, read by
    /// whoever started it.
    stage: AtomicU32,
}

/// The trampoline, placed in its page and ready to be pointed at.
#[derive(Debug)]
pub(crate) struct Trampoline {
    page: PhysAddr,
    parameters: *mut Parameters,
}

impl Trampoline {
    /// Writes the blob and everything that does not change from one processor
    /// to the next into the page, and zeroes the rest of it.
    ///
    /// # Errors
    ///
    /// [`ApicError::TrampolineTooLarge`] if the blob has grown into the
    /// parameter block, or [`ApicError::PageTableRootTooHigh`] if the page
    /// tables are above four gigabytes, where the 32-bit instruction that loads
    /// them cannot name them.
    ///
    /// # Safety
    ///
    /// `at` must be the writable address of a whole reserved page whose
    /// physical address is `page`, and `page` must be below one megabyte
    /// and frame-aligned. Nothing else may be using that page: this
    /// overwrites all of it.
    pub(crate) unsafe fn place(
        at: *mut u8,
        page: PhysAddr,
        page_table_root: PhysAddr,
        entry: u64,
    ) -> Result<Self, ApicError> {
        let extent = extent();
        if extent.bytes > PARAMETERS {
            return Err(ApicError::TrampolineTooLarge {
                bytes: extent.bytes,
                room: PARAMETERS,
            });
        }
        if page_table_root.as_u64() > u64::from(u32::MAX) {
            return Err(ApicError::PageTableRootTooHigh {
                phys: page_table_root.as_u64(),
            });
        }
        // The whole page first, so that what is left between the blob and the
        // block, and after the block, is a known quantity rather than whatever
        // firmware last did with low memory.
        // SAFETY: the caller guarantees a whole writable page at `at`, used by
        // nothing else.
        unsafe { at.write_bytes(0, paging::as_usize(PAGE)) };
        // SAFETY: the assembler emits `extent.bytes` as the distance between the
        // first byte of the blob and the last, so the source is that many
        // readable bytes of this image's own text; the destination is the page
        // the caller vouched for, and the blob is shorter than the parameter
        // block's offset, which is inside it. The two cannot overlap, since the
        // blob lives in this image and the page is reserved low memory.
        unsafe { at.copy_from_nonoverlapping(&raw const BLOB_START, extent.bytes) };

        let base = page.as_u64();
        let protected = wide(extent.protected);
        let long = wide(extent.long);
        #[expect(
            clippy::cast_ptr_alignment,
            reason = "the page is frame-aligned and PARAMETERS is a multiple of the alignment, asserted at compile time"
        )]
        // SAFETY: `PARAMETERS` is inside the page the caller vouched for, which
        // is frame-aligned, and the offset is a multiple of the structure's
        // alignment — asserted where it is declared, so the sum is aligned too.
        // The structure also ends inside the page, which is asserted beside it.
        let parameters = unsafe { at.add(PARAMETERS) }.cast::<Parameters>();
        let value = Parameters {
            gdt: table(base),
            pointer: TablePointer {
                limit: TABLE_LIMIT,
                base: narrow(base + wide(PARAMETERS + offset_of!(Parameters, gdt))),
            },
            protected_entry: FarPointer {
                offset: narrow(protected),
                selector: CODE32_SELECTOR,
            },
            long_entry: FarPointer {
                offset: narrow(base + long),
                selector: CODE64_SELECTOR,
            },
            page_base: base,
            page_table_root: page_table_root.as_u64(),
            stack_top: 0,
            entry,
            stage: AtomicU32::new(Stage::WAITING),
        };
        // SAFETY: the pointer is inside the caller's page, aligned, and nothing
        // else refers to what is being overwritten — the blob copy stopped short
        // of it.
        unsafe { parameters.write(value) };
        Ok(Self { page, parameters })
    }

    /// Points the next start at `stack_top` and puts the stage back to nothing.
    ///
    /// Only safe to call once the processor started before it is accounted for:
    /// it has arrived somewhere its caller can see — publishing itself as one
    /// of the machine's processors is a long way past its last read of this
    /// block. A processor that did not publish progress after a startup command
    /// is not safe to reuse either, because the command may still be pending
    /// and there is no way to cancel or query it.
    pub(crate) fn prepare(&self, stack_top: u64) {
        // SAFETY: the block was written by `place` into a page nothing else uses,
        // and the caller establishes that the processor started before this one
        // has finished reading it, so nothing else is reading these while they
        // are written.
        unsafe { (&raw mut (*self.parameters).stack_top).write(stack_top) };
        self.stage().store(Stage::WAITING, Ordering::Release);
    }

    /// Whether the processor being started has executed the first instruction
    /// of the trampoline.
    ///
    /// It says so before it does anything that could fail, so a start that gets
    /// this far and no further says the fault is after the very first
    /// instructions rather than in delivering the command.
    pub(crate) fn began(&self) -> bool {
        self.reached() >= Stage::REAL
    }

    /// How far the processor being started got, for a line of log output.
    pub(crate) fn reached(&self) -> u32 {
        self.stage().load(Ordering::Acquire)
    }

    /// The word the processor being started reports itself through.
    ///
    /// The one field of the block written by two processors, which is why it is
    /// the only one that is an atomic. The other side of it is an ordinary
    /// aligned 32-bit store in the trampoline's own assembly: this architecture
    /// makes such a store indivisible and orders it after everything that
    /// processor did first, which is what an acquiring read of it here relies
    /// on.
    fn stage(&self) -> &AtomicU32 {
        // SAFETY: the block is inside a page nothing else uses, and this field is
        // the one both sides agreed would be written concurrently.
        unsafe { &(*self.parameters).stage }
    }

    /// The eight-bit page number a startup command names this page by.
    pub(crate) fn vector(&self) -> u8 {
        narrow_page(self.page.as_u64() / PAGE)
    }
}

/// How large the blob is and where its stages begin.
fn extent() -> &'static Extent {
    // SAFETY: the assembler emits these three words beside the blob, in a
    // read-only section of this image, and nothing ever writes them.
    unsafe { &EXTENT }
}

/// An offset inside the blob as the 64-bit value the arithmetic below is done
/// in.
///
/// The blob is a few hundred bytes and the block is at a fixed offset inside
/// one page, so nothing is lost.
const fn wide(value: usize) -> u64 {
    value as u64
}

/// The four descriptors the trampoline loads.
///
/// Only one of them contains an address. Basing the 32-bit code segment at the
/// page is what lets the first far jump name its target as an offset from the
/// start of the blob, which the assembler can compute, instead of a linear
/// address that only run-time code knows.
fn table(page: u64) -> [u64; 4] {
    /// Bits the access byte occupies: present, privilege, and what kind of
    /// segment this is.
    const ACCESS: u32 = 40;
    /// Bits the granularity and size flags occupy.
    const FLAGS: u32 = 52;

    /// Present, ring 0, code or data, executable, readable, accessed.
    const CODE: u64 = (0x9B << ACCESS) | LIMIT;
    /// Present, ring 0, code or data, writable, accessed.
    const DATA: u64 = (0x93 << ACCESS) | LIMIT;
    /// A limit of all ones with page granularity: the whole address space.
    const LIMIT: u64 = 0xFFFF | (0xF << 48) | (0x8 << FLAGS);
    /// Default operand size 32, which a 64-bit segment must not have.
    const WIDTH32: u64 = 0x4 << FLAGS;
    /// Long mode.
    const LONG: u64 = 0x2 << FLAGS;

    [0, CODE | WIDTH32 | based(page), DATA | WIDTH32, CODE | LONG]
}

/// A descriptor's base address, which the architecture splits across three
/// fields for reasons that stopped applying in 1985.
const fn based(base: u64) -> u64 {
    ((base & 0x00FF_FFFF) << 16) | ((base & 0xFF00_0000) << 32)
}

/// A linear address as the 32-bit field a descriptor table pointer or a far
/// pointer holds it in.
///
/// Everything narrowed here is inside the first megabyte or is a page table
/// root already checked against four gigabytes, so this loses nothing.
#[expect(
    clippy::cast_possible_truncation,
    reason = "every address narrowed here is below 4 GiB, checked where it enters"
)]
const fn narrow(value: u64) -> u32 {
    value as u32
}

/// A page number as the eight-bit field a startup command holds it in.
///
/// The page is below one megabyte, so its number is below 256.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the trampoline page is below 1 MiB, so its page number is below 256"
)]
const fn narrow_page(value: u64) -> u8 {
    value as u8
}

#[cfg(test)]
mod tests {
    use core::mem::{align_of, offset_of, size_of};

    use super::{
        CODE32_SELECTOR, CODE64_SELECTOR, DATA32_SELECTOR, ENTRY_FRAME, FarPointer, PARAMETERS,
        Parameters, TABLE_LIMIT, TablePointer, based, table,
    };
    use crate::PAGE;

    #[test]
    fn the_block_the_assembly_reads_is_where_the_assembly_is_told_it_is() {
        // The assembly is handed these offsets, so a change here is a change
        // there — but only if the two are computed from the same declaration,
        // which these assertions are what pin down.
        assert_eq!(size_of::<Parameters>(), 0x60);
        assert_eq!(align_of::<Parameters>(), 8);
        assert_eq!(offset_of!(Parameters, gdt), 0x00);
        assert_eq!(offset_of!(Parameters, pointer), 0x20);
        assert_eq!(offset_of!(Parameters, protected_entry), 0x26);
        assert_eq!(offset_of!(Parameters, long_entry), 0x2C);
        assert_eq!(offset_of!(Parameters, page_base), 0x38);
        assert_eq!(offset_of!(Parameters, page_table_root), 0x40);
        assert_eq!(offset_of!(Parameters, stack_top), 0x48);
        assert_eq!(offset_of!(Parameters, entry), 0x50);
        assert_eq!(offset_of!(Parameters, stage), 0x58);
    }

    #[test]
    fn the_block_fits_in_the_page_with_the_blob_below_it() {
        assert!(PARAMETERS.is_multiple_of(align_of::<Parameters>()));
        assert!(PARAMETERS + size_of::<Parameters>() <= paging::as_usize(PAGE));
        // The word two processors share has to be naturally aligned for the
        // ordinary store on the other side of it to be indivisible.
        assert!((PARAMETERS + offset_of!(Parameters, stage)).is_multiple_of(4));
    }

    #[test]
    fn the_architectural_operands_are_exactly_as_wide_as_the_architecture_says() {
        assert_eq!(size_of::<TablePointer>(), 6);
        assert_eq!(align_of::<TablePointer>(), 1);
        assert_eq!(size_of::<FarPointer>(), 6);
        assert_eq!(align_of::<FarPointer>(), 1);
        assert_eq!(TABLE_LIMIT, 31);
    }

    #[test]
    fn the_entry_frame_leaves_the_stack_where_a_call_would_have() {
        // A return slot plus the four the calling convention reserves, and a
        // stack pointer eight past a sixteen-byte boundary.
        assert_eq!(ENTRY_FRAME, 40);
        assert_eq!(u64::from(ENTRY_FRAME) % 16, 8);
    }

    #[test]
    fn the_descriptors_are_the_ones_the_stages_they_serve_require() {
        let [null, code32, data32, code64] = table(0);
        assert_eq!(null, 0);
        // Accessed, so that loading a selector needs no write to the table.
        assert_eq!(code32, 0x00CF_9B00_0000_FFFF);
        assert_eq!(data32, 0x00CF_9300_0000_FFFF);
        assert_eq!(code64, 0x00AF_9B00_0000_FFFF);
        // The one combination the architecture forbids: a 64-bit segment with a
        // 32-bit default operand size.
        assert_eq!(code64 & (1 << 53), 1 << 53, "long");
        assert_eq!(code64 & (1 << 54), 0, "default operand size");
        assert_eq!(code32 & (1 << 54), 1 << 54, "default operand size");
    }

    #[test]
    fn the_selectors_name_the_descriptors_the_table_puts_them_at() {
        assert_eq!(CODE32_SELECTOR / 8, 1);
        assert_eq!(DATA32_SELECTOR / 8, 2);
        assert_eq!(CODE64_SELECTOR / 8, 3);
    }

    #[test]
    fn a_descriptor_base_goes_into_the_three_fields_the_architecture_splits_it_across() {
        assert_eq!(based(0), 0);
        assert_eq!(based(0xFF000), 0x0000_000F_F000_0000);
        assert_eq!(based(0xFFFF_FFFF), 0xFF00_00FF_FFFF_0000);
        // Nothing lands in the low limit, the access byte, the limit's high
        // nibble, or the flags: the base occupies bits 16 to 39 and 56 to 63.
        for page in [0u64, 0x1000, 0x7F000, 0xFF000] {
            assert_eq!(based(page) & 0x00FF_FF00_0000_FFFF, 0, "{page:#x}");
        }
    }
}
