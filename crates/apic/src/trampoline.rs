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
//! the assembly names that this structure does not have will not assemble.
//!
//! # The descriptor table it loads
//!
//! Five entries, built here rather than assembled, because one of them has to
//! contain an address. The 32-bit code segment is based at the page, which is
//! what lets the first far jump name its target as an offset the assembler can
//! work out instead of a linear address only this code knows. The rest are
//! flat: the data segments describe all four gigabytes, and the 64-bit code
//! segment's base is ignored by the mode it exists to enter.

use core::{
    mem::offset_of,
    sync::atomic::{AtomicU32, Ordering},
};

use x86_64::PhysAddr;

use crate::ApicError;

/// Where the parameter block sits in the page.
///
/// Far enough in that the code has room, aligned enough for the eight-byte
/// fields in it, and inside the page with the whole block to spare. That the
/// code really does fit below it is checked rather than assumed, in
/// [`Trampoline::new`].
const PARAMETERS: usize = 0x800;

/// How many descriptors the trampoline's own table holds.
const DESCRIPTORS: u16 = 5;

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

/// `CR0.MP`, which says there is a coprocessor to fault for.
const CR0_MONITOR_COPROCESSOR: u32 = 1 << 1;

/// `CR0.EM`, which makes every SSE instruction fault. A processor comes out of
/// reset with it set, and the code being jumped to is compiled to use those
/// instructions.
const CR0_EMULATE_COPROCESSOR: u32 = 1 << 2;

/// What is left of `CR0` after the emulate bit is taken out of it.
const CR0_COPROCESSOR_CLEAR: u32 = !CR0_EMULATE_COPROCESSOR;

/// `CR4.PAE`, which long mode requires, together with the two bits that make
/// the SSE registers usable and their exceptions reportable.
const CR4_LONG_MODE: u32 = (1 << 5) | (1 << 9) | (1 << 10);

/// The register holding long mode enable and no-execute enable.
const IA32_EFER: u32 = 0xC000_0080;

/// `EFER.LME` and `EFER.NXE`. The second has to be set before the page tables
/// are loaded, not after: they have the no-execute bit set in them, and it is
/// reserved-must-be-zero until this is written.
const EFER_LONG_MODE: u32 = (1 << 8) | (1 << 11);

core::arch::global_asm!(
    include_str!("trampoline.s"),
    STARTED = const PARAMETERS + offset_of!(Parameters, started),
    TABLE_POINTER = const PARAMETERS + offset_of!(Parameters, pointer),
    PROTECTED_ENTRY = const PARAMETERS + offset_of!(Parameters, protected_entry),
    LONG_ENTRY = const PARAMETERS + offset_of!(Parameters, long_entry),
    PAGE_BASE = const PARAMETERS + offset_of!(Parameters, page_base),
    PAGE_TABLE_ROOT = const PARAMETERS + offset_of!(Parameters, page_table_root),
    STACK_TOP = const PARAMETERS + offset_of!(Parameters, stack_top),
    ENTRY = const PARAMETERS + offset_of!(Parameters, entry),
    DATA32_SELECTOR = const DATA32_SELECTOR,
    CR0_PROTECTED = const CR0_PROTECTED,
    CR0_PAGING = const CR0_PAGING,
    CR0_COPROCESSOR_SET = const CR0_MONITOR_COPROCESSOR,
    CR0_COPROCESSOR_CLEAR = const CR0_COPROCESSOR_CLEAR,
    CR4_LONG_MODE = const CR4_LONG_MODE,
    IA32_EFER = const IA32_EFER,
    EFER_LONG_MODE = const EFER_LONG_MODE,
);

unsafe extern "C" {
    /// First byte of the blob, and where a started processor begins.
    #[link_name = "pulzar_trampoline_start"]
    static BLOB_START: u8;
    /// Where its 32-bit stage begins.
    #[link_name = "pulzar_trampoline_protected"]
    static STAGE_PROTECTED: u8;
    /// Where its 64-bit stage begins.
    #[link_name = "pulzar_trampoline_long"]
    static STAGE_LONG: u8;
    /// One past its last byte.
    #[link_name = "pulzar_trampoline_end"]
    static BLOB_END: u8;
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
    /// Null, 32-bit code based at the page, flat data, 64-bit code, flat data.
    gdt: [u64; 5],
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
    /// Which processor this start is for, so that the one that arrives can be
    /// checked against the one that was asked for.
    apic_id: u32,
    /// Set by the processor being started, before it does anything else.
    started: AtomicU32,
}

/// The trampoline, placed in its page and ready to be pointed at.
#[derive(Debug)]
pub(crate) struct Trampoline {
    page: PhysAddr,
    parameters: *mut Parameters,
}

impl Trampoline {
    /// Copies the blob into the page and fills in everything that does not
    /// change from one processor to the next.
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
        let blob = blob();
        if blob.len() > PARAMETERS {
            return Err(ApicError::TrampolineTooLarge {
                bytes: blob.len(),
                room: PARAMETERS,
            });
        }
        if page_table_root.as_u64() > u64::from(u32::MAX) {
            return Err(ApicError::PageTableRootTooHigh {
                phys: page_table_root.as_u64(),
            });
        }
        // SAFETY: the caller guarantees a whole writable page at `at`, and the
        // blob is shorter than the parameter block's offset, which is inside it.
        // The two cannot overlap, since the blob lives in this image.
        unsafe { at.copy_from_nonoverlapping(blob.as_ptr(), blob.len()) };

        let base = page.as_u64();
        #[expect(
            clippy::cast_ptr_alignment,
            reason = "the page is frame-aligned and PARAMETERS is a multiple of the alignment, asserted at compile time"
        )]
        // SAFETY: `PARAMETERS` is inside the page the caller vouched for, which
        // is frame-aligned, and the offset is a multiple of the structure's
        // alignment — asserted where it is declared, so the sum is aligned too.
        let parameters = unsafe { at.add(PARAMETERS) }.cast::<Parameters>();
        let value = Parameters {
            gdt: table(base),
            pointer: TablePointer {
                limit: TABLE_LIMIT,
                base: narrow(base + PARAMETERS as u64 + offset_of!(Parameters, gdt) as u64),
            },
            protected_entry: FarPointer {
                offset: narrow(stage(&raw const STAGE_PROTECTED)),
                selector: CODE32_SELECTOR,
            },
            long_entry: FarPointer {
                offset: narrow(base + stage(&raw const STAGE_LONG)),
                selector: CODE64_SELECTOR,
            },
            page_base: base,
            page_table_root: page_table_root.as_u64(),
            stack_top: 0,
            entry,
            apic_id: 0,
            started: AtomicU32::new(0),
        };
        // SAFETY: the pointer is inside the caller's page, aligned, and nothing
        // else refers to what is being overwritten — the blob copy stopped short
        // of it.
        unsafe { parameters.write(value) };
        Ok(Self { page, parameters })
    }

    /// Points the next start at `stack_top`, records which processor it is for,
    /// and clears the flag that processor will set.
    pub(crate) fn prepare(&self, apic_id: u32, stack_top: u64) {
        // SAFETY: the block was written by `place` into a page nothing else uses,
        // and only one processor is ever being started at a time, so nothing else
        // is reading these while they are written.
        unsafe {
            (&raw mut (*self.parameters).stack_top).write(stack_top);
            (&raw mut (*self.parameters).apic_id).write(apic_id);
            self.started().store(0, Ordering::Release);
        }
    }

    /// Whether the processor being started has begun executing.
    ///
    /// It sets this before it does anything that could fail, so a start that
    /// gets this far and no further says the fault is after the very first
    /// instructions rather than in delivering the command.
    pub(crate) fn started(&self) -> &AtomicU32 {
        // SAFETY: the block is inside a page nothing else uses, and this field
        // is the one both sides agreed would be written concurrently — which is
        // why it is the only one that is an atomic.
        unsafe { &(*self.parameters).started }
    }

    /// The eight-bit page number a startup command names this page by.
    pub(crate) fn vector(&self) -> u8 {
        narrow_page(self.page.as_u64() / PAGE_SIZE)
    }
}

/// Bytes in the page a startup command's vector selects.
const PAGE_SIZE: u64 = 4096;

/// The blob, as bytes.
fn blob() -> &'static [u8] {
    let start = &raw const BLOB_START;
    let end = &raw const BLOB_END;
    // SAFETY: both are addresses of symbols the assembly block defines around a
    // contiguous run of bytes in this image's own text, so the range is one
    // allocated object and the length is non-negative. It is read-only and never
    // executed in place.
    unsafe { core::slice::from_raw_parts(start, end.offset_from_unsigned(start)) }
}

/// How far into the blob a stage begins.
///
/// Taken from the assembly's own symbols rather than written down here, so that
/// each far jump lands where the assembler put the code rather than where this
/// file guessed it would.
fn stage(label: *const u8) -> u64 {
    let start = &raw const BLOB_START;
    // SAFETY: every label passed here is inside the same contiguous run of bytes
    // the assembly block defines, so the difference between them is meaningful
    // and non-negative.
    wide(unsafe { label.offset_from_unsigned(start) })
}

/// An offset inside the blob as the 32-bit field a far pointer holds it in.
///
/// The blob is a few hundred bytes, so nothing is lost.
const fn wide(value: usize) -> u64 {
    value as u64
}

/// The five descriptors the trampoline loads.
///
/// Only one of them contains an address. Basing the 32-bit code segment at the
/// page is what lets the first far jump name its target as an offset from the
/// start of the blob, which the assembler can compute, instead of a linear
/// address that only run-time code knows.
fn table(page: u64) -> [u64; 5] {
    /// Bits the access byte occupies: present, privilege, and what kind of
    /// segment this is.
    const ACCESS: u32 = 40;
    /// Bits the granularity and size flags occupy.
    const FLAGS: u32 = 52;

    /// Present, ring 0, code or data, executable, readable, accessed.
    const CODE: u64 = (0x9A << ACCESS) | LIMIT;
    /// Present, ring 0, code or data, writable, accessed.
    const DATA: u64 = (0x92 << ACCESS) | LIMIT;
    /// A limit of all ones with page granularity: the whole address space.
    const LIMIT: u64 = 0xFFFF | (0xF << 48) | (0x8 << FLAGS);
    /// Default operand size 32, which a 64-bit segment must not have.
    const WIDTH32: u64 = 0x4 << FLAGS;
    /// Long mode.
    const LONG: u64 = 0x2 << FLAGS;

    [
        0,
        CODE | WIDTH32 | based(page),
        DATA | WIDTH32,
        CODE | LONG,
        DATA | WIDTH32,
    ]
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
