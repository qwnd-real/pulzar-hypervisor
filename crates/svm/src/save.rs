//! The architectural state a guest runs with, and the state it stops in.
//!
//! Three quarters of a guest's control block, and the one half of it the
//! processor both reads and writes: it loads a guest out of here on the way in
//! and writes back what that guest became on the way out. A field at the wrong
//! offset is therefore not a setting the processor ignores — it is a guest
//! entered with somebody else's control register, or a hypervisor reading a
//! register the processor never wrote. That is why every offset below is
//! asserted rather than merely written down.
//!
//! The same layout describes the host. What the processor saves of our own
//! state while a guest runs is this structure again, at the same offset into a
//! different page, which is why nothing here is named for the guest.
//!
//! # The size is load bearing
//!
//! A control block is exactly one page and this area begins a kibibyte into
//! it, so it is exactly three kibibytes — not "at least". Every reserved run,
//! including the trailing one, is sized to make that come out exact, and
//! [`SAVE_AREA_BYTES`] is asserted against the total. A reserved run one byte
//! short does not shrink the structure harmlessly; it shifts every field after
//! it and moves the end of the page.
//!
//! # Not every field is live
//!
//! Much of what is defined here only carries state when a feature is switched
//! on elsewhere, and the fields are inert otherwise — neither loaded nor
//! stored, so whatever they hold means nothing. The guest page-attribute table
//! is used only under nested paging. The five control-transfer registers are
//! swapped only where hardware acceleration of branch-record virtualization
//! exists and the control area has enabled it; the debug-extension control, the
//! branch-record stack and its select register ride on that same switch, on
//! processors that have the stack at all. The instruction-sampling block is
//! swapped only under sampling virtualization. Each field says which switch it
//! belongs to, because a hypervisor that reads one of them without having
//! turned the feature on is reading nothing.
//!
//! # The segment attributes are the trap
//!
//! Segment registers are stored here in a form that resembles an in-memory
//! descriptor but is not one: the attributes are squeezed into twelve bits
//! drawn from two separated fields of the original. [`SegmentAttributes`]
//! describes that packing and the rules that come with it, and is worth reading
//! before writing a single segment.

/// Bytes the state-save area occupies.
///
/// Fixed by where it sits rather than by what is defined in it: it begins a
/// kibibyte into a control block that is exactly one page, so it is exactly the
/// remaining three kibibytes however much of that the architecture has given
/// meaning to.
pub const SAVE_AREA_BYTES: usize = 0xC00;

/// Bytes one segment register occupies in the save area.
pub const SEGMENT_BYTES: usize = 16;

/// How many branch records the save area has room for.
///
/// The processor's own stack may be shorter than this, in which case the
/// entries past its length are unused — the space is reserved for the widest
/// stack the architecture allows rather than for any one processor's.
pub const LBR_STACK_ENTRIES: usize = 16;

/// Everything the processor loads a guest from and writes it back to.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct SaveArea {
    /// The extra segment.
    pub es: Segment,
    /// The code segment, whose privilege field will agree with [`Self::cpl`]
    /// but is not where the privilege level is read from.
    pub cs: Segment,
    /// The stack segment.
    pub ss: Segment,
    /// The data segment.
    pub ds: Segment,
    /// The first of the two segments whose base is fully 64-bit, and the one a
    /// 64-bit operating system usually points at per-thread storage.
    pub fs: Segment,
    /// The second segment with a full 64-bit base, which is also the one the
    /// swap-base instruction exchanges.
    pub gs: Segment,
    /// The global descriptor table register. Only its base and the lower
    /// sixteen bits of its limit exist; its selector and attributes are
    /// reserved.
    pub gdtr: Segment,
    /// The local descriptor table register.
    pub ldtr: Segment,
    /// The interrupt descriptor table register, described the same way as the
    /// global one.
    pub idtr: Segment,
    /// The task register. Like the code segment it never holds a NULL segment,
    /// so its present bit is ignored.
    pub tr: Segment,
    reserved_0x0a0: [u8; 0x2B],
    /// The privilege level the guest is running at.
    ///
    /// This field, and not any segment's own privilege field, is what the
    /// processor reads the guest's current privilege level from. It is ignored
    /// in real mode, where the level is forced to zero, and in virtual-8086
    /// mode, where it is forced to three.
    pub cpl: u8,
    reserved_0x0cc: [u8; 4],
    /// The extended feature register, which decides what mode the guest runs
    /// in. Its virtualization-enable bit must be set for a guest to be entered
    /// at all.
    pub efer: u64,
    reserved_0x0d8: [u8; 8],
    /// The guest's first performance event-select register.
    pub perf_ctl0: u64,
    /// The counter that event-select register drives.
    pub perf_ctr0: u64,
    /// The guest's second performance event-select register.
    pub perf_ctl1: u64,
    /// The counter that event-select register drives.
    pub perf_ctr1: u64,
    /// The guest's third performance event-select register.
    pub perf_ctl2: u64,
    /// The counter that event-select register drives.
    pub perf_ctr2: u64,
    /// The guest's fourth performance event-select register.
    pub perf_ctl3: u64,
    /// The counter that event-select register drives.
    pub perf_ctr3: u64,
    /// The guest's fifth performance event-select register.
    pub perf_ctl4: u64,
    /// The counter that event-select register drives.
    pub perf_ctr4: u64,
    /// The guest's sixth performance event-select register.
    pub perf_ctl5: u64,
    /// The counter that event-select register drives.
    pub perf_ctr5: u64,
    reserved_0x140: [u8; 8],
    /// The fourth control register: the feature switches, from page-size
    /// extensions through to the ones that gate the newer instruction sets.
    pub cr4: u64,
    /// The third control register, holding the root of the guest's own page
    /// tables. Under nested paging those tables produce addresses the guest
    /// believes are physical, which a second set then translates again.
    pub cr3: u64,
    /// The first control register: protection, paging, and the cache
    /// behaviour. A guest whose cache-disable bit is clear while its
    /// not-write-through bit is set is one the processor refuses to enter.
    pub cr0: u64,
    /// The debug control register, which arms the four address breakpoints.
    pub dr7: u64,
    /// The debug status register, which reports which of them fired.
    pub dr6: u64,
    /// The guest's flags, including the interrupt flag whose reach depends on
    /// whether interrupt masking is being virtualized for it.
    pub rflags: u64,
    /// Where the guest resumes.
    pub rip: u64,
    reserved_0x180: [u8; 0x40],
    /// The guest's retired-instruction counter.
    pub instructions_retired: u64,
    /// The guest's global performance-counter status: which counters have
    /// overflowed.
    pub performance_status: u64,
    /// The guest's global performance-counter control: which of the counters
    /// above are enabled.
    ///
    /// A doubleword, despite being described as a quadword. The manual's own
    /// two statements about it cannot both hold — a quadword here would run
    /// through 0x1D7, and 0x1D4 through 0x1D7 are separately declared reserved.
    /// The layout for encrypted guests describes the same register at the same
    /// offset as a doubleword, which is the reading taken here.
    pub performance_control: u32,
    reserved_0x1d4: [u8; 4],
    /// The guest's stack pointer.
    pub rsp: u64,
    /// The guest's shadow-stack control register, governing where control-flow
    /// enforcement applies.
    pub s_cet: u64,
    /// The guest's shadow stack pointer.
    pub ssp: u64,
    /// The guest's interrupt shadow-stack table address, which is where an
    /// interrupt taken under control-flow enforcement finds its shadow stack.
    pub isst_addr: u64,
    /// The accumulator, which is the one general-purpose register the processor
    /// swaps itself — every other one is the hypervisor's to save and restore.
    pub rax: u64,
    /// The segment selectors the fast system-call instructions load, and the
    /// entry point the 32-bit form of them uses.
    pub star: u64,
    /// The entry point a fast system call from 64-bit code lands on.
    pub lstar: u64,
    /// The entry point a fast system call from compatibility mode lands on.
    pub cstar: u64,
    /// Which flags a fast system call clears on entry.
    pub sfmask: u64,
    /// The base the swap-base instruction exchanges with the `GS` base, which
    /// is how a kernel recovers its own per-processor pointer on entry.
    pub kernel_gs_base: u64,
    /// The code selector the older fast system-call mechanism enters through;
    /// its stack and data selectors follow from this one by fixed offsets.
    pub sysenter_cs: u64,
    /// The stack pointer that mechanism enters with.
    pub sysenter_esp: u64,
    /// The instruction pointer that mechanism enters at.
    pub sysenter_eip: u64,
    /// The second control register, holding the address of the last page fault
    /// the guest took.
    pub cr2: u64,
    reserved_0x248: [u8; 0x20],
    /// The guest's page-attribute table, deciding what memory type each of its
    /// page-table encodings means.
    ///
    /// Read only when nested paging is on. Without it the guest's own register
    /// is in force and this is not consulted.
    pub g_pat: u64,
    /// The guest's debug control register, which is what enables branch
    /// recording and single-stepping on branches.
    ///
    /// Swapped only where hardware acceleration of branch-record virtualization
    /// is supported and the control area has enabled it. Otherwise the guest
    /// shares whatever the host left in the real register.
    pub dbgctl: u64,
    /// Where the guest's last recorded branch came from, under that same
    /// switch.
    pub br_from: u64,
    /// Where that branch went, under that same switch.
    pub br_to: u64,
    /// Where the guest's last recorded interrupt or exception was taken from,
    /// under that same switch.
    pub last_excp_from: u64,
    /// Where that interrupt or exception was delivered to, under that same
    /// switch.
    pub last_excp_to: u64,
    /// The guest's debug-extension control, which is what enables recording
    /// into the branch-record stack rather than into the single pair above.
    ///
    /// Swapped only on processors that have the stack, and only when
    /// branch-record virtualization is enabled in the control area.
    pub dbgextnctl: u64,
    reserved_0x2a0: [u8; 0x40],
    /// The guest's speculation control: the switches for the mitigations
    /// against speculative side channels.
    pub spec_ctrl: u64,
    reserved_0x2e8: [u8; 0x388],
    /// The guest's branch-record stack, in the order the processor addresses
    /// its registers.
    ///
    /// Live only where branch-record stack virtualization is supported and
    /// enabled in the control area, and asymmetric even then: entering a guest
    /// loads the stack but saves no host copy, and leaving one saves the
    /// guest's without restoring a host copy, because there is none to restore.
    pub lbr_stack: [LbrEntry; LBR_STACK_ENTRIES],
    /// Which kinds of control transfer the stack above records, and which it
    /// suppresses. Live under the same switch and swapped the same asymmetric
    /// way.
    pub lbr_select: u64,
    /// The guest's fetch-sampling control: whether fetch sampling is on and how
    /// often it takes a sample.
    ///
    /// This field and the nine after it hold state only under
    /// instruction-sampling virtualization, enabled in the control area.
    pub ibs_fetch_ctl: u64,
    /// The linear address of the instruction the sampled fetch was for.
    pub ibs_fetch_linear_address: u64,
    /// The guest's op-sampling control: whether op sampling is on and how often
    /// it takes a sample.
    pub ibs_op_ctl: u64,
    /// The address of the sampled instruction itself.
    pub ibs_op_rip: u64,
    /// What the sampled instruction did: its latency and the stages it spent it
    /// in.
    pub ibs_op_data: u64,
    /// The branch behaviour of the sampled instruction, where it was one.
    pub ibs_op_data2: u64,
    /// The memory behaviour of the sampled instruction: which caches and which
    /// translation structures it hit or missed.
    pub ibs_op_data3: u64,
    /// The linear address the sampled instruction accessed data at.
    pub ibs_dc_linear_address: u64,
    /// The target of the sampled branch.
    pub ibs_branch_target_rip: u64,
    /// The extended fetch-sampling control, carrying the fetch behaviour that
    /// did not fit the original control register.
    pub ibs_fetch_extended_ctl: u64,
    reserved_0x7c8: [u8; 0x438],
}

impl SaveArea {
    /// A save area with every byte zero.
    ///
    /// Not a runnable guest: an all-zero area names no code segment, has paging
    /// and protection off, and leaves the virtualization-enable bit of the
    /// extended feature register clear, which on its own is enough for the
    /// processor to refuse the entry. It is the correct starting point, which
    /// is a different claim — every reserved byte is zero as the architecture
    /// requires, and every segment is a NULL segment in the form the
    /// architecture defines rather than merely an empty one.
    #[must_use]
    pub const fn zeroed() -> Self {
        Self {
            es: Segment::null(),
            cs: Segment::null(),
            ss: Segment::null(),
            ds: Segment::null(),
            fs: Segment::null(),
            gs: Segment::null(),
            gdtr: Segment::null(),
            ldtr: Segment::null(),
            idtr: Segment::null(),
            tr: Segment::null(),
            reserved_0x0a0: [0; 0x2B],
            cpl: 0,
            reserved_0x0cc: [0; 4],
            efer: 0,
            reserved_0x0d8: [0; 8],
            perf_ctl0: 0,
            perf_ctr0: 0,
            perf_ctl1: 0,
            perf_ctr1: 0,
            perf_ctl2: 0,
            perf_ctr2: 0,
            perf_ctl3: 0,
            perf_ctr3: 0,
            perf_ctl4: 0,
            perf_ctr4: 0,
            perf_ctl5: 0,
            perf_ctr5: 0,
            reserved_0x140: [0; 8],
            cr4: 0,
            cr3: 0,
            cr0: 0,
            dr7: 0,
            dr6: 0,
            rflags: 0,
            rip: 0,
            reserved_0x180: [0; 0x40],
            instructions_retired: 0,
            performance_status: 0,
            performance_control: 0,
            reserved_0x1d4: [0; 4],
            rsp: 0,
            s_cet: 0,
            ssp: 0,
            isst_addr: 0,
            rax: 0,
            star: 0,
            lstar: 0,
            cstar: 0,
            sfmask: 0,
            kernel_gs_base: 0,
            sysenter_cs: 0,
            sysenter_esp: 0,
            sysenter_eip: 0,
            cr2: 0,
            reserved_0x248: [0; 0x20],
            g_pat: 0,
            dbgctl: 0,
            br_from: 0,
            br_to: 0,
            last_excp_from: 0,
            last_excp_to: 0,
            dbgextnctl: 0,
            reserved_0x2a0: [0; 0x40],
            spec_ctrl: 0,
            reserved_0x2e8: [0; 0x388],
            lbr_stack: [LbrEntry::empty(); LBR_STACK_ENTRIES],
            lbr_select: 0,
            ibs_fetch_ctl: 0,
            ibs_fetch_linear_address: 0,
            ibs_op_ctl: 0,
            ibs_op_rip: 0,
            ibs_op_data: 0,
            ibs_op_data2: 0,
            ibs_op_data3: 0,
            ibs_dc_linear_address: 0,
            ibs_branch_target_rip: 0,
            ibs_fetch_extended_ctl: 0,
            reserved_0x7c8: [0; 0x438],
        }
    }

    /// The state a processor is in the instant a reset or an `INIT` has left
    /// it, which is executing at the reset vector.
    ///
    /// Real mode, no paging, protection off, each segment sixty-four kibibytes
    /// long from address zero and the descriptor-table registers likewise —
    /// with the one exception that is the reset vector itself: the code
    /// segment's base is the top of the first four gigabytes less
    /// sixty-four kibibytes, its selector is that base's top sixteen bits,
    /// and the instruction pointer is sixteen bytes below the end of it.
    /// Which is where firmware's first instruction sits, and where a
    /// processor an `INIT` reset resumes.
    ///
    /// Not on its own a control block a guest can be entered from, for the
    /// reason [`SaveArea::started_at`] gives.
    #[must_use]
    pub fn at_reset() -> Self {
        let mut save = Self::zeroed();
        save.cs = Segment {
            selector: RESET_SELECTOR,
            attributes: CODE_SEGMENT,
            limit: REAL_MODE_LIMIT,
            base: RESET_SEGMENT_BASE,
        };
        let data = Segment {
            selector: 0,
            attributes: DATA_SEGMENT,
            limit: REAL_MODE_LIMIT,
            base: 0,
        };
        save.ds = data;
        save.es = data;
        save.fs = data;
        save.gs = data;
        save.ss = data;
        // Neither descriptor-table register has a selector or attributes at
        // all; only the base and the low half of the limit exist.
        save.gdtr.limit = REAL_MODE_LIMIT;
        save.idtr.limit = REAL_MODE_LIMIT;
        save.ldtr = Segment {
            selector: 0,
            attributes: LOCAL_DESCRIPTOR_TABLE,
            limit: REAL_MODE_LIMIT,
            base: 0,
        };
        save.tr = Segment {
            selector: 0,
            attributes: TASK_STATE_SEGMENT,
            limit: REAL_MODE_LIMIT,
            base: 0,
        };
        save.cr0 = CR0_AT_RESET;
        save.dr6 = DR6_AT_RESET;
        save.dr7 = DR7_AT_RESET;
        save.rflags = RFLAGS_AT_RESET;
        save.rip = RESET_INSTRUCTION_POINTER;
        save
    }

    /// The state a processor is in the instant a start-up message naming `page`
    /// has released it.
    ///
    /// Two architectural states in one, because a processor only ever reaches
    /// the second through the first. `INIT` leaves every register at the value
    /// [`SaveArea::at_reset`] writes, and a start-up message then changes
    /// exactly three things: the code segment's selector and base, which the
    /// message's vector gives the page number of, and the instruction pointer,
    /// which becomes zero. So execution begins at the very start of the page
    /// the message named.
    ///
    /// The one deliberate departure is the code segment's limit and the data
    /// segments', which the architecture leaves at sixty-four kibibytes even
    /// though the base can be anywhere in the first megabyte. That is real
    /// mode, not an approximation of it: sixteen-bit code addresses nothing
    /// beyond its own segment.
    ///
    /// Not on its own a control block a guest can be entered from. The extended
    /// feature register is zero here because that is what reset leaves it, and
    /// a guest whose virtualization-enable bit is clear is one the
    /// processor refuses — supplying it is the caller's, along with
    /// anything else the hypervisor rather than the architecture decides.
    #[must_use]
    pub fn started_at(page: u8) -> Self {
        let mut save = Self::at_reset();
        save.cs = Segment {
            selector: u16::from(page) << STARTUP_SELECTOR_SHIFT,
            attributes: CODE_SEGMENT,
            limit: REAL_MODE_LIMIT,
            base: u64::from(page) << STARTUP_BASE_SHIFT,
        };
        save.rip = 0;
        save
    }
}

/// The code segment's selector at the reset vector, which is its base's top
/// sixteen bits.
const RESET_SELECTOR: u16 = 0xF000;

/// The code segment's base at the reset vector: the top of the first four
/// gigabytes, less the sixty-four kibibytes the segment is long.
const RESET_SEGMENT_BASE: u64 = 0xFFFF_0000;

/// Where in that segment a processor starts, which is sixteen bytes below its
/// end — the whole of the room the architecture leaves for the jump firmware
/// puts there.
const RESET_INSTRUCTION_POINTER: u64 = 0xFFF0;

/// Bits a start-up message's vector is shifted by to give the code segment's
/// selector.
const STARTUP_SELECTOR_SHIFT: u16 = 8;

/// Bits it is shifted by to give the code segment's base, which is the address
/// of the page it names.
const STARTUP_BASE_SHIFT: u32 = 12;

/// How long every segment is coming out of reset: sixty-four kibibytes, which
/// is the whole of what sixteen-bit code can address within one.
const REAL_MODE_LIMIT: u32 = 0xFFFF;

/// The code segment a processor comes out of reset with: readable code, present
/// and sixteen-bit.
const CODE_SEGMENT: SegmentAttributes = SegmentAttributes::new()
    .with_kind(CODE_READABLE_ACCESSED)
    .with_descriptor(true)
    .with_present(true);

/// Every other segment: writable data, present and sixteen-bit.
const DATA_SEGMENT: SegmentAttributes = SegmentAttributes::new()
    .with_kind(DATA_WRITABLE_ACCESSED)
    .with_descriptor(true)
    .with_present(true);

/// The local descriptor table register, which names a system segment rather
/// than a code or data one.
const LOCAL_DESCRIPTOR_TABLE: SegmentAttributes = SegmentAttributes::new()
    .with_kind(SYSTEM_LDT)
    .with_present(true);

/// The task register, whose reset value names a busy sixteen-bit task — busy
/// because nothing has switched away from it.
const TASK_STATE_SEGMENT: SegmentAttributes = SegmentAttributes::new()
    .with_kind(SYSTEM_BUSY_TSS_16)
    .with_present(true);

/// A code segment that may be read as well as executed, and has been accessed.
const CODE_READABLE_ACCESSED: u8 = 0xB;

/// A data segment that may be written, and has been accessed.
const DATA_WRITABLE_ACCESSED: u8 = 0x3;

/// The system-segment type naming a local descriptor table.
const SYSTEM_LDT: u8 = 0x2;

/// The system-segment type naming a busy sixteen-bit task state segment.
const SYSTEM_BUSY_TSS_16: u8 = 0x3;

/// The first control register at reset: caching off, write-through off, and the
/// bit that has meant "a coprocessor is present" since it stopped being
/// optional.
///
/// Caching disabled *and* write-through disabled together, which matters: the
/// other way round — caching on with write-through off — is a combination the
/// processor refuses to enter a guest in.
const CR0_AT_RESET: u64 = 0x6000_0010;

/// The debug status register at reset, whose every defined bit reads inverted.
const DR6_AT_RESET: u64 = 0xFFFF_0FF0;

/// The debug control register at reset, which arms no breakpoint.
const DR7_AT_RESET: u64 = 0x0000_0400;

/// The flags at reset: nothing but the bit that is always set.
const RFLAGS_AT_RESET: u64 = 0x2;

layout! {
    SaveArea, size = SAVE_AREA_BYTES,
    0x000 => es,
    0x010 => cs,
    0x020 => ss,
    0x030 => ds,
    0x040 => fs,
    0x050 => gs,
    0x060 => gdtr,
    0x070 => ldtr,
    0x080 => idtr,
    0x090 => tr,
    0x0A0 => reserved_0x0a0,
    0x0CB => cpl,
    0x0CC => reserved_0x0cc,
    0x0D0 => efer,
    0x0D8 => reserved_0x0d8,
    0x0E0 => perf_ctl0,
    0x0E8 => perf_ctr0,
    0x0F0 => perf_ctl1,
    0x0F8 => perf_ctr1,
    0x100 => perf_ctl2,
    0x108 => perf_ctr2,
    0x110 => perf_ctl3,
    0x118 => perf_ctr3,
    0x120 => perf_ctl4,
    0x128 => perf_ctr4,
    0x130 => perf_ctl5,
    0x138 => perf_ctr5,
    0x140 => reserved_0x140,
    0x148 => cr4,
    0x150 => cr3,
    0x158 => cr0,
    0x160 => dr7,
    0x168 => dr6,
    0x170 => rflags,
    0x178 => rip,
    0x180 => reserved_0x180,
    0x1C0 => instructions_retired,
    0x1C8 => performance_status,
    0x1D0 => performance_control,
    0x1D4 => reserved_0x1d4,
    0x1D8 => rsp,
    0x1E0 => s_cet,
    0x1E8 => ssp,
    0x1F0 => isst_addr,
    0x1F8 => rax,
    0x200 => star,
    0x208 => lstar,
    0x210 => cstar,
    0x218 => sfmask,
    0x220 => kernel_gs_base,
    0x228 => sysenter_cs,
    0x230 => sysenter_esp,
    0x238 => sysenter_eip,
    0x240 => cr2,
    0x248 => reserved_0x248,
    0x268 => g_pat,
    0x270 => dbgctl,
    0x278 => br_from,
    0x280 => br_to,
    0x288 => last_excp_from,
    0x290 => last_excp_to,
    0x298 => dbgextnctl,
    0x2A0 => reserved_0x2a0,
    0x2E0 => spec_ctrl,
    0x2E8 => reserved_0x2e8,
    0x670 => lbr_stack,
    0x770 => lbr_select,
    0x778 => ibs_fetch_ctl,
    0x780 => ibs_fetch_linear_address,
    0x788 => ibs_op_ctl,
    0x790 => ibs_op_rip,
    0x798 => ibs_op_data,
    0x7A0 => ibs_op_data2,
    0x7A8 => ibs_op_data3,
    0x7B0 => ibs_dc_linear_address,
    0x7B8 => ibs_branch_target_rip,
    0x7C0 => ibs_fetch_extended_ctl,
    0x7C8 => reserved_0x7c8,
}

/// One segment register as the processor stores it.
///
/// Base and limit are stored expanded rather than scattered across a descriptor
/// the way they are in memory, so nothing here has to be reassembled. The base
/// is a full sixty-four bits, though for the code segment and the four legacy
/// data segments only its lower thirty-two are implemented — the upper half of
/// a base is meaningful only for `FS` and `GS`.
///
/// The two descriptor-table registers are described by this same structure with
/// most of it unused: `GDTR` and `IDTR` have no selector and no attributes, and
/// only the lower sixteen bits of their limit exist, because a table limit
/// never needed more.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Segment {
    /// The selector the guest loaded, which the processor keeps but does not
    /// re-resolve — the attributes, limit and base below are what it actually
    /// uses.
    pub selector: u16,
    /// The descriptor's attributes, packed into twelve bits.
    pub attributes: SegmentAttributes,
    /// The segment's limit, already scaled by granularity where the descriptor
    /// asked for it.
    pub limit: u32,
    /// Where the segment begins.
    pub base: u64,
}

impl Segment {
    /// A NULL segment: no selector, no base, no limit, and every attribute bit
    /// clear.
    ///
    /// All-zero attributes is what the architecture asks for a NULL segment
    /// specifically, rather than being merely the convenient value — clearing
    /// the present bit is how a NULL segment is signalled wherever that
    /// distinction is meaningful.
    #[must_use]
    pub const fn null() -> Self {
        Self {
            selector: 0,
            attributes: SegmentAttributes::new(),
            limit: 0,
            base: 0,
        }
    }
}

layout! {
    Segment, size = SEGMENT_BYTES,
    0x0 => selector,
    0x2 => attributes,
    0x4 => limit,
    0x8 => base,
}

const _: () = assert!(
    align_of::<Segment>() == align_of::<u64>(),
    "Segment must be aligned as its base is, or every segment after it moves",
);

/// A descriptor's attributes, as the save area packs them.
///
/// This is the field most often written wrongly, and the reason is that it
/// looks like part of a descriptor without being one. An in-memory 64-bit
/// descriptor keeps its attributes in two separated runs — bits 47:40 and bits
/// 55:52, with the upper nibble of the limit lying between them. The save area
/// drops that gap and stores the two runs adjacent, low run first, giving
/// twelve meaningful bits with the top four reserved. Copying the descriptor's
/// bits 55:40 across unchanged produces a value where every attribute above the
/// present bit sits four positions too high, which is not something the
/// processor will reject.
///
/// # Writing them
///
/// For a NULL segment, write every attribute bit zero. For anything else, write
/// that concatenation and nothing else — this describes a descriptor that
/// already exists, it is not a place to invent one.
///
/// # Reading them back
///
/// The present bit is what says whether a segment is NULL, but only where a
/// NULL segment is possible at all: the code segment and the task register
/// never hold one, so their present bit carries no such meaning and is ignored.
///
/// The current privilege level does not come from here. It is the save area's
/// own [`SaveArea::cpl`] field, and the code segment's privilege field will
/// agree with it — taking the privilege level from any segment's own field is
/// the second classic mistake, and it happens to work often enough to stay
/// hidden. In real mode the processor forces the level to zero and in
/// virtual-8086 mode to three, whatever either field says.
///
/// # Only some bits are observed
///
/// These attributes are loaded from memory software wrote, so they can describe
/// combinations no real descriptor could ever have held. The processor
/// tolerates that because it only ever looks at a subset, and which subset
/// depends on the register being loaded:
///
/// - Code segment: default size, long mode, present, and readable.
/// - Stack segment: default size, present, expand-down, writable, and the
///   code/data distinction. Present is observed in legacy and compatibility
///   mode only — in 64-bit mode every stack segment is treated as present.
/// - The four data segments: default size, present, privilege level,
///   expand-down, writable, and the code/data distinction.
/// - Local descriptor table register: present, the descriptor bit, and the
///   type.
/// - Task register: present, the descriptor bit, and the type.
///
/// Anything outside those subsets is carried across a world switch without
/// being acted on, so an inconsistent value survives silently rather than being
/// reported.
#[bitfield_struct::bitfield(u16)]
#[derive(PartialEq, Eq)]
pub struct SegmentAttributes {
    /// What kind of segment this is, and the access bits that travel with the
    /// kind: for data, expand-down and writable; for code, conforming and
    /// readable. Reading it needs the descriptor bit below, since the same
    /// encodings mean different things for a system segment.
    #[bits(4)]
    pub kind: u8,
    /// Whether this is a code or data segment rather than a system one. Clear
    /// for the local descriptor table and the task state segment, set for
    /// everything the guest loads into an ordinary segment register.
    pub descriptor: bool,
    /// The privilege level the descriptor was written with. Observed for the
    /// four data segments; it is not where the guest's current privilege level
    /// comes from.
    #[bits(2)]
    pub dpl: u8,
    /// Whether the segment is present. Clear is how a NULL segment is
    /// signalled, except for the code segment and the task register, which
    /// never hold one.
    pub present: bool,
    /// The bit the architecture leaves entirely to software. Nothing in the
    /// processor reads it.
    pub available: bool,
    /// Whether this code segment runs in 64-bit mode. Meaningful only for the
    /// code segment, and mutually exclusive with the default-size bit below.
    pub long: bool,
    /// The default operand and address size: on a code segment 32-bit rather
    /// than 16-bit operations, on the stack segment the width of the implicit
    /// stack pointer.
    pub default_size: bool,
    /// Whether the descriptor's limit counted pages rather than bytes. The
    /// limit stored alongside is already scaled, so this records what the
    /// descriptor said rather than something still to be applied.
    pub granularity: bool,
    #[bits(4)]
    __: u8,
}

/// One entry of the branch-record stack: where a control transfer came from and
/// where it went.
///
/// The two halves are separate model-specific registers on the processor and
/// are stored here interleaved, one pair per recorded transfer, in the order
/// those registers are addressed. Keeping them paired is what the data means —
/// a from without its to describes nothing.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct LbrEntry {
    /// The address the control transfer was taken from.
    pub from: u64,
    /// The address it went to.
    pub to: u64,
}

impl LbrEntry {
    /// An entry recording nothing.
    #[must_use]
    pub const fn empty() -> Self {
        Self { from: 0, to: 0 }
    }
}

layout! {
    LbrEntry, size = 2 * size_of::<u64>(),
    0x0 => from,
    0x8 => to,
}
