//! The memory-type range registers, answered out of a copy of this processor's
//! own rather than allowed to reach them.
//!
//! These describe which physical addresses are cacheable and how, and under
//! this hypervisor they are the *host's*: one set per core, shared by both of
//! its threads, and the thing every mapping the hypervisor and firmware made is
//! interpreted through. A guest write left to reach them reprograms the memory
//! types under the host and under whichever vcpu is running on the sibling
//! thread, with none of the cache-flush transition the architecture prescribes
//! for that, and with resident write-back lines describing memory that has just
//! stopped being write-back. What follows is corruption of whatever the sibling
//! was touching, discovered by the sibling as its own data having turned to
//! garbage.
//!
//! # There is no effect to emulate
//!
//! Under nested paging the memory type of a guest access is decided by the
//! guest's page tables against the guest's page-attribute table, combined with
//! the nested type — and these registers are not consulted for a guest access
//! at all. So this is not a partial emulation that could be completed later:
//! there is nothing for the guest's ranges to do, and the whole of what the
//! guest is owed is a set of registers that remembers what it wrote.
//!
//! # Seeded from the machine, once, per processor
//!
//! An empty set would be an honest register file and a dishonest machine: a
//! guest reading it would find every address defaulting to uncacheable, and an
//! operating system that brings its processors into agreement about their
//! ranges — Linux does, inside a `stop_machine` rendezvous — would find each of
//! them claiming something different from what firmware really programmed. So
//! each processor reads its own registers once, before its guest has run, and
//! the guest's first read is the truth about the core it is running on.
//!
//! The fixed-range routing bits need one extra step during seeding. AMD hides
//! them when `SYSCFG.MtrrFixDramModEn` is clear, so the host briefly opens that
//! read window, captures the complete fixed-range values, and restores the
//! control bit. Some virtual machine monitors do not expose that write; in
//! that case the ranges are captured as the hardware exposes them and the
//! unavailable routing bits remain clear. Guest accesses afterwards begin and
//! end in the per-processor copy, and the real MTRRs stay the host's for the
//! rest of their life.
//!
//! A start-up message does not disturb the copy, because `INIT` does not
//! disturb the registers: an application processor released by one comes up
//! with the ranges firmware gave it, which is exactly what makes the guest's
//! rendezvous find every processor already agreeing.
//!
//! # What a write is refused for
//!
//! Every refusal here is one real hardware makes, because a guest programming
//! its ranges probes for them. The architecture states only one of them in as
//! many words — a write of the capability register raises `#GP` — and expresses
//! the rest as `MBZ` in the register layouts, which means a write setting such
//! a bit faults rather than having the bit dropped.
//!
//! - The capability register is read-only, even written the value it holds.
//! - A variable range at or above the count that register reports, and every
//!   fixed range on a machine reporting none, does not exist.
//! - The type of the default-type register and of a variable range's base is
//!   the whole low byte, and only the five encodings the architecture defines
//!   are accepted: uncacheable, write-combining, write-through, write-protect
//!   and write-back. The standard encoding leaves bits 7:3 of it `MBZ`, so a
//!   byte holding anything else is refused either way.
//! - The default-type register reserves bits 63:12 and 9:8; a base reserves
//!   11:8; a mask reserves 10:0.
//! - Both variable-range registers reserve every address bit at or above the
//!   width this processor implements, which is what makes the width a run-time
//!   question rather than a constant.
//!
//! What is deliberately *not* refused is the arrangement of a range. The
//! architecture requires a range's size to be a power of two and its base to be
//! aligned to that size, but states those as software's obligations rather than
//! as write-time checks — a mask whose set bits are not one contiguous run is
//! accepted by the machine and leaves the range's behaviour undefined. Checking
//! it here would fault a guest that hardware would have obliged.
//!
//! # The two routing bits of a fixed range are not always there
//!
//! A fixed range's type byte has two forms. In the standard one its type is
//! bits 2:0 and bits 7:3 are `MBZ`. In the extended one bits 4:3 become `RdMem`
//! and `WrMem`, which route the range to memory rather than to the bus, and
//! only bits 7:5 are reserved. Which form is in force is not this module's
//! decision and not the guest's either: the MTRR control fields of `SYSCFG` are
//! shadowed with the ranges, so the guest and the fixed-range view stay in one
//! state without changing the host's core-shared controls.
//!
//! The switch governs access rather than storage. With it clear the two bits
//! read as zero and a write of them is dropped — not faulted — and the value
//! underneath survives, which is what lets firmware set the switch, program the
//! routing, and clear the switch again while the routing goes on working. So
//! this module consults the guest's virtual switch on each access and masks the
//! two bits out of both directions when it is clear, leaving bits 7:5 refused
//! either way.
//!
//! # What this machine answers
//!
//! Measured on the processor this was written for, and the basis for every rule
//! above: the capability register reports eight variable ranges, fixed ranges
//! and write-combining. Only range zero is valid — write-back over a mask of
//! `0000_ffff_8000_0800` — and the other seven are cleared. The default type is
//! uncacheable with both enables set. The fixed ranges hold plain write-back
//! over the first half-mebibyte, write-protect over the last four
//! four-kibibyte registers, and uncacheable between; no routing bit is set
//! anywhere, and `SYSCFG` reports the switch for them clear. Refusal of an
//! address bit begins at bit 48, which is the width `CPUID` reports.
//!
//! Two rules are not confirmed by that machine because it cannot exhibit them:
//! that a write naming write-combining is refused where the capability register
//! denies having it, and that the fixed ranges do not exist where it reports
//! none. Both follow from the register's own description, and this machine
//! reports both features present.

use core::{
    hint::spin_loop,
    sync::atomic::{AtomicBool, Ordering},
};

use bitflags::bitflags;
use log::{info, warn};
use probe::{read as probe_read, write as probe_write};
use x86_64::registers::model_specific::Msr;

use crate::msr::Fault;

/// Every register this module answers for, for whatever programs the permission
/// map.
///
/// The whole of the variable-range block is named rather than the ranges this
/// machine turned out to have, and every fixed-range register rather than only
/// those a machine with fixed ranges has: reaching one the machine lacks is a
/// general protection fault the guest is entitled to, and it cannot be given
/// one by code that never sees the access.
///
/// The same set as [`claims`], and the two have to be the same set: an index
/// intercepted and not claimed stops the guest, because no handler owns it, and
/// one claimed and not intercepted is executed against the machine's own
/// register.
pub fn intercepted() -> impl Iterator<Item = u32> {
    [MTRR_CAP, MTRR_DEF_TYPE]
        .into_iter()
        .chain(VARIABLE_FIRST..=VARIABLE_LAST)
        .chain(FIXED)
        .chain([SYSCFG])
}

/// Whether an index is one this module answers for.
pub(crate) fn claims(msr: u32) -> bool {
    Register::of(msr).is_some()
}

/// One processor's guest's memory-type ranges.
///
/// Per processor because the registers it stands for are, and seeded from those
/// registers because an operating system compares what its processors report.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Mtrrs {
    capability: Capability,
    default_type: u64,
    variable: [Range; MAX_RANGES],
    fixed: [u64; FIXED.len()],
    syscfg: SystemConfiguration,
}

impl Mtrrs {
    /// Reads this processor's own ranges, which is what its guest starts with.
    ///
    /// A range the capability register does not report, and every fixed range
    /// on a machine that reports none, is left empty and never answered for
    /// — so nothing here reads a register the machine might not have.
    pub(crate) fn seed() -> Self {
        let capability = Capability(read(MTRR_CAP));
        let syscfg = SystemConfiguration::seed();
        let mut mtrrs = Self {
            capability,
            default_type: read(MTRR_DEF_TYPE),
            variable: [Range::EMPTY; MAX_RANGES],
            fixed: [0; FIXED.len()],
            syscfg,
        };
        // The block alternates base, mask, base, mask, so stepping by two walks
        // the ranges and each mask is the number after its base.
        for (slot, base) in (VARIABLE_FIRST..=VARIABLE_LAST)
            .step_by(REGISTERS_PER_RANGE)
            .enumerate()
            .take(capability.ranges())
        {
            mtrrs.variable[slot] = Range {
                base: read(base),
                mask: read(base + 1),
            };
        }
        if capability.fixed() {
            mtrrs.fixed = capture_fixed();
        }
        info!(
            "exits: the guest's memory-type ranges are this processor's own: capability {:#x}, \
             default type {:#x}",
            capability.into_bits(),
            mtrrs.default_type,
        );
        mtrrs
    }

    /// What the guest reads from one of them.
    ///
    /// `routing` is the guest's virtual switch for the two routing bits of a
    /// fixed range, which read as zero while it is clear.
    ///
    /// # Errors
    ///
    /// [`Fault`] for an index this machine has no register at, which the guest
    /// takes as the general protection fault the access would have raised.
    pub(crate) fn read(&self, msr: u32, routing: Routing) -> Result<u64, Fault> {
        Ok(match self.present(msr)? {
            Register::Capability => self.capability.into_bits(),
            Register::DefaultType => self.default_type,
            Register::Base(range) => self.variable[range].base,
            Register::Mask(range) => self.variable[range].mask,
            Register::Fixed(slot) => routing.visible(self.fixed[slot]),
            Register::SystemConfiguration => self.syscfg.read()?,
        })
    }

    /// What a write of one of them does, which is to be remembered and to go no
    /// further.
    ///
    /// # Errors
    ///
    /// [`Fault`] for a register this machine does not have, for the capability
    /// register, which is read-only, and for a value real hardware would
    /// refuse.
    pub(crate) fn write(&mut self, msr: u32, value: u64, routing: Routing) -> Result<(), Fault> {
        match self.present(msr)? {
            // Read-only: it reports what the machine has, and a guest cannot
            // give itself another range by claiming one.
            Register::Capability => return Err(Fault),
            Register::DefaultType => {
                self.accepts(value, DEFAULT_TYPE_RESERVED)?;
                self.default_type = value;
            }
            Register::Base(range) => {
                self.accepts(value, BASE_RESERVED | above_address_width())?;
                self.variable[range].base = value;
            }
            // The one register here with no memory type in it: the whole of a
            // mask is address bits and the switch saying whether to compare them.
            Register::Mask(range) => {
                if value & (MASK_RESERVED | above_address_width()) != 0 {
                    return Err(Fault);
                }
                self.variable[range].mask = value;
            }
            // The routing bits are dropped rather than stored while the machine's
            // switch for them is clear, which is what the machine does with them:
            // the switch governs access, so whatever was underneath survives.
            Register::Fixed(slot) => {
                self.accepts_fixed(value)?;
                self.fixed[slot] = routing.merged(self.fixed[slot], value);
            }
            Register::SystemConfiguration => self.syscfg.write(value)?,
        }
        Ok(())
    }

    /// The guest-visible switch controlling extended fixed-range fields.
    pub(crate) const fn routing(&self) -> Routing {
        self.syscfg.routing()
    }

    /// Which register an index names, provided this machine has it.
    fn present(&self, msr: u32) -> Result<Register, Fault> {
        let register = Register::of(msr).ok_or(Fault)?;
        let present = match register {
            Register::Capability | Register::DefaultType | Register::SystemConfiguration => true,
            Register::Base(range) | Register::Mask(range) => range < self.capability.ranges(),
            Register::Fixed(_) => self.capability.fixed(),
        };
        present.then_some(register).ok_or(Fault)
    }

    /// Whether a register holding a memory type in its low byte may be given
    /// `value`, `reserved` being the bits that register assigns no meaning to.
    fn accepts(&self, value: u64, reserved: u64) -> Result<(), Fault> {
        if value & reserved != 0 {
            return Err(Fault);
        }
        let [encoding, ..] = value.to_le_bytes();
        self.accepts_type(encoding)
    }

    /// Whether a fixed-range register may be given `value`, every byte of which
    /// is one range's type and, where the guest's switch allows them, its two
    /// routing bits.
    ///
    /// The routing bits are not judged here at all: while the switch is clear a
    /// write of them is dropped rather than refused, and while it is set they
    /// are simply writable. Only the top three bits of a byte are `MBZ` in
    /// either form.
    fn accepts_fixed(&self, value: u64) -> Result<(), Fault> {
        value.to_le_bytes().into_iter().try_for_each(|byte| {
            if FixedByte::from_bits_retain(byte)
                .intersection(FixedByte::RESERVED)
                .bits()
                != 0
            {
                return Err(Fault);
            }
            self.accepts_type(byte & FixedByte::TYPE.bits())
        })
    }

    /// Whether a memory-type encoding is one this machine's registers take.
    fn accepts_type(&self, encoding: u8) -> Result<(), Fault> {
        match MemoryType::of(encoding) {
            // Refused by a processor that does not implement it, and the guest is
            // answered this machine's own capability register — so what it may
            // name has to be what that register told it.
            Some(MemoryType::WriteCombining) if !self.capability.write_combining() => Err(Fault),
            Some(_) => Ok(()),
            None => Err(Fault),
        }
    }
}

/// The part of `SYSCFG` that belongs to MTRR virtualization.
#[derive(Clone, Copy, Debug)]
struct SystemConfiguration {
    mtrr: Syscfg,
}

impl SystemConfiguration {
    /// Captures the host's MTRR controls for this processor.
    fn seed() -> Self {
        Self::from_value(read(SYSCFG))
    }

    /// Creates a shadow from a raw `SYSCFG` value.
    fn from_value(value: u64) -> Self {
        Self {
            mtrr: Syscfg::from_bits_retain(value).intersection(Syscfg::MTRR),
        }
    }

    /// Reads the guest-visible value, preserving hardware ownership of the
    /// unrelated system-configuration fields.
    fn read(self) -> Result<u64, Fault> {
        let hardware = probe_read(SYSCFG).map_err(|_| Fault)?;
        Ok(self.visible(hardware))
    }

    /// Applies a guest write to the non-MTRR fields and remembers the MTRR
    /// fields without changing the host's core-shared controls.
    fn write(&mut self, value: u64) -> Result<(), Fault> {
        let hardware = probe_read(SYSCFG).map_err(|_| Fault)?;
        let (host_value, mtrr) = Self::prepare_write(hardware, value);
        probe_write(SYSCFG, host_value).map_err(|_| Fault)?;
        self.mtrr = mtrr;
        Ok(())
    }

    /// Merges the virtual MTRR fields with hardware-owned fields.
    fn visible(self, hardware: u64) -> u64 {
        (hardware & !Syscfg::MTRR.bits()) | self.mtrr.bits()
    }

    /// Builds the hardware write and the shadow value for a guest write.
    fn prepare_write(hardware: u64, value: u64) -> (u64, Syscfg) {
        (
            (value & !Syscfg::MTRR.bits()) | (hardware & Syscfg::MTRR.bits()),
            Syscfg::from_bits_retain(value).intersection(Syscfg::MTRR),
        )
    }

    /// The guest-visible switch controlling fixed-range routing fields.
    const fn routing(self) -> Routing {
        Routing::from_syscfg(self.mtrr)
    }
}

/// Serializes the short host-side window used to expose hidden fixed-range
/// fields during per-processor initialization.
static FIXED_CAPTURE_LOCK: AtomicBool = AtomicBool::new(false);

/// Captures fixed ranges, opening AMD's hidden routing fields when available.
///
/// A virtual machine monitor may expose the fixed-range registers without
/// exposing the `SYSCFG` write that reveals their routing bits. The capture
/// remains useful in that case: the raw register values are retained and the
/// unavailable routing bits stay clear in the guest shadow.
fn capture_fixed() -> [u64; FIXED.len()] {
    let _window = FixedCaptureWindow::open();
    FIXED.map(read)
}

/// The temporary host-side `MtrrFixDramModEn` window.
struct FixedCaptureWindow {
    was_open: bool,
    opened: bool,
}

impl FixedCaptureWindow {
    /// Opens the read window after serializing access to the core-shared MSR.
    fn open() -> Self {
        while FIXED_CAPTURE_LOCK
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            spin_loop();
        }

        let value = read(SYSCFG);
        let syscfg = Syscfg::from_bits_retain(value);
        let was_open = syscfg.contains(Syscfg::FIX_DRAM_MOD_EN);
        let opened =
            was_open || probe_write(SYSCFG, value | Syscfg::FIX_DRAM_MOD_EN.bits()).is_ok();
        if !opened {
            warn!("exits: fixed-range routing bits are unavailable; preserving the raw ranges");
        }
        Self { was_open, opened }
    }
}

impl Drop for FixedCaptureWindow {
    fn drop(&mut self) {
        if self.opened && !self.was_open {
            let value = read(SYSCFG) & !Syscfg::FIX_DRAM_MOD_EN.bits();
            let _ = probe_write(SYSCFG, value);
        }
        FIXED_CAPTURE_LOCK.store(false, Ordering::Release);
    }
}

/// Whether the two routing bits of a fixed range's type bytes are reachable.
///
/// The guest's `SYSCFG` shadow holds the switch, so a fixed-range access
/// follows the same virtual state as the control-register read that changed it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Routing {
    modifiable: bool,
}

impl Routing {
    /// What a virtual `SYSCFG` value says about fixed-range routing.
    const fn from_syscfg(syscfg: Syscfg) -> Self {
        Self {
            modifiable: syscfg.contains(Syscfg::FIX_DRAM_MOD_EN),
        }
    }

    /// What a read of a fixed range answers: the routing bits of every byte
    /// read as zero while the switch is clear.
    const fn visible(self, value: u64) -> u64 {
        if self.modifiable {
            value
        } else {
            value & !FIXED_ROUTING
        }
    }

    /// What a write of a fixed range leaves stored: `written`, except that
    /// while the switch is clear the routing bits keep what they held, that
    /// part of the write having been dropped.
    const fn merged(self, stored: u64, written: u64) -> u64 {
        if self.modifiable {
            written
        } else {
            (written & !FIXED_ROUTING) | (stored & FIXED_ROUTING)
        }
    }
}

/// What the capability register reports about this machine's ranges.
///
/// Answered to the guest verbatim, which is what makes it the authority the
/// rest of this module judges against: the count of variable ranges served and
/// whether write-combining may be named have to be the ones the guest was told.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Capability(u64);

impl Capability {
    /// How many variable ranges this machine has.
    ///
    /// Taken out of the low byte rather than cast, and clamped: the
    /// architecture numbers sixteen registers for variable ranges, so a
    /// processor claiming more would be claiming ranges no guest can name.
    fn ranges(self) -> usize {
        let [count, ..] = self.0.to_le_bytes();
        usize::from(count).min(MAX_RANGES)
    }

    /// Whether the fixed-range registers exist at all.
    const fn fixed(self) -> bool {
        self.0 & FIXED_SUPPORTED != 0
    }

    /// Whether write-combining may be named as a memory type.
    const fn write_combining(self) -> bool {
        self.0 & WRITE_COMBINING_SUPPORTED != 0
    }

    /// The value a guest read of the register answers with.
    const fn into_bits(self) -> u64 {
        self.0
    }
}

/// One variable range, exactly as the guest last left it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Range {
    /// What the base register holds: the range's base address and its type.
    base: u64,
    /// What the mask register holds: which bits of an address the base is
    /// compared against, and whether the range is consulted at all.
    mask: u64,
}

impl Range {
    /// A range that is not consulted, the valid bit of its mask being clear.
    const EMPTY: Self = Self { base: 0, mask: 0 };
}

/// Which of the registers an index names.
///
/// Named rather than matched on as an arithmetic expression at each use site,
/// because what a read gives and what a write is refused for are written
/// separately and have to agree about which register they are talking about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Register {
    /// The read-only register reporting what ranges the machine has.
    Capability,
    /// The register holding the type of every address no range covers, and the
    /// two switches deciding which ranges are consulted.
    DefaultType,
    /// One variable range's base address and type, by range.
    Base(usize),
    /// One variable range's mask and valid bit, by range.
    Mask(usize),
    /// One fixed-range register, by position in [`FIXED`].
    Fixed(usize),
    /// The system configuration register's MTRR control fields.
    SystemConfiguration,
}

impl Register {
    /// Which of them an index names, or `None` for an index that is not one of
    /// these registers on any processor.
    fn of(msr: u32) -> Option<Self> {
        match msr {
            MTRR_CAP => Some(Self::Capability),
            MTRR_DEF_TYPE => Some(Self::DefaultType),
            VARIABLE_FIRST..=VARIABLE_LAST => {
                // The block alternates base, mask, base, mask, so the low bit of
                // the offset says which of a pair and the rest says which range.
                let [offset, ..] = (msr - VARIABLE_FIRST).to_le_bytes();
                let range = usize::from(offset) / REGISTERS_PER_RANGE;
                Some(if offset.is_multiple_of(2) {
                    Self::Base(range)
                } else {
                    Self::Mask(range)
                })
            }
            // Eleven scattered numbers with gaps between them, so a search
            // rather than arithmetic — and an index in one of those gaps is a
            // register on no processor.
            _ => FIXED
                .iter()
                .position(|&fixed| fixed == msr)
                .map(Self::Fixed)
                .or_else(|| (msr == SYSCFG).then_some(Self::SystemConfiguration)),
        }
    }
}

/// One of the memory types a range register may name.
///
/// Five of the eight encodings three bits can express, and the three left out
/// are what makes this an enumeration rather than a byte: a write naming one of
/// them raises `#GP` on real hardware, and a guest programming its ranges
/// relies on being told so.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MemoryType {
    /// Uncacheable.
    Uncacheable,
    /// Write-combining, which not every processor has.
    WriteCombining,
    /// Write-through.
    WriteThrough,
    /// Write-protected: reads may be cached, writes always reach memory.
    WriteProtect,
    /// Write-back.
    WriteBack,
}

impl MemoryType {
    /// Which type an encoding names, or `None` for one the architecture
    /// reserves.
    const fn of(encoding: u8) -> Option<Self> {
        match encoding {
            0 => Some(Self::Uncacheable),
            1 => Some(Self::WriteCombining),
            4 => Some(Self::WriteThrough),
            5 => Some(Self::WriteProtect),
            6 => Some(Self::WriteBack),
            _ => None,
        }
    }
}

/// What one of this machine's own range registers holds.
fn read(msr: u32) -> u64 {
    // SAFETY: every index this is called with is one of the memory-type range
    // registers, which are architectural on any processor able to run this
    // hypervisor, and the optional ones are read only where the capability
    // register says the machine has them. Reading one has no side effect.
    unsafe { Msr::new(msr).read() }
}

/// Bits of a physical address this processor does not implement, which either
/// variable-range register refuses a write of.
fn above_address_width() -> u64 {
    let bits = u32::from(processor::physical_address_bits());
    // A processor implementing all sixty-four would leave nothing above, and the
    // shift itself would be undefined.
    if bits >= u64::BITS { 0 } else { !0 << bits }
}

/// `MTRRcap`, reporting what ranges this machine has and which optional parts
/// of the mechanism it implements. Read-only.
const MTRR_CAP: u32 = 0xFE;

/// `MTRRdefType`, holding the type of every address no range covers and the two
/// switches deciding which ranges are consulted.
const MTRR_DEF_TYPE: u32 = 0x2FF;

/// `MTRRphysBase0`, the first register of the variable-range block.
const VARIABLE_FIRST: u32 = 0x200;

/// `MTRRphysMask7`, the last of it.
const VARIABLE_LAST: u32 = 0x20F;

/// The fixed-range registers, in the order of the addresses they describe.
///
/// Eleven scattered numbers rather than a range: one register for the first
/// sixty-four kibibytes, two for the sixteen-kibibyte blocks above it, and
/// eight for the four-kibibyte pages of the last quarter of the first mebibyte.
/// The numbers in the gaps between them are not registers on any processor, so
/// they are neither claimed nor intercepted and a guest reaching one is refused
/// by the machine itself.
const FIXED: [u32; 11] = [
    0x250, // MTRRfix64K_00000
    0x258, // MTRRfix16K_80000
    0x259, // MTRRfix16K_A0000
    0x268, // MTRRfix4K_C0000
    0x269, // MTRRfix4K_C8000
    0x26A, // MTRRfix4K_D0000
    0x26B, // MTRRfix4K_D8000
    0x26C, // MTRRfix4K_E0000
    0x26D, // MTRRfix4K_E8000
    0x26E, // MTRRfix4K_F0000
    0x26F, // MTRRfix4K_F8000
];

/// How many registers describe one variable range, which is a base and a mask.
const REGISTERS_PER_RANGE: usize = 2;

/// How many variable ranges the architecture gives register numbers to.
const MAX_RANGES: usize = 8;

/// `FIX`: whether this machine has the fixed-range registers.
const FIXED_SUPPORTED: u64 = 1 << 8;

/// `WC`: whether write-combining may be named as a memory type.
const WRITE_COMBINING_SUPPORTED: u64 = 1 << 10;

/// The memory type in the low byte of the default-type register and of every
/// variable range's base.
const TYPE_FIELD: u64 = 0xFF;

/// `FE`: whether the fixed ranges are consulted.
const FIXED_ENABLE: u64 = 1 << 10;

/// `E`: whether any range is consulted at all.
const ENABLE: u64 = 1 << 11;

/// Bits the default-type register assigns no meaning to.
///
/// Must-be-zero rather than ignored: a `WRMSR` setting one raises `#GP`, so
/// this is a mask a write is judged against rather than masked with — as are
/// the two below it.
const DEFAULT_TYPE_RESERVED: u64 = !(TYPE_FIELD | FIXED_ENABLE | ENABLE);

/// Where the address field of either variable-range register begins: a range is
/// described to four-kibibyte granularity and no finer.
const ADDRESS_SHIFT: u32 = 12;

/// That field, as the bits it occupies.
const ADDRESS_FIELD: u64 = !0 << ADDRESS_SHIFT;

/// `Valid`: whether a variable range is consulted.
const VALID: u64 = 1 << 11;

/// Bits a variable range's base register assigns no meaning to below its
/// address field.
const BASE_RESERVED: u64 = !(TYPE_FIELD | ADDRESS_FIELD);

/// Bits its mask register assigns no meaning to below the same field.
const MASK_RESERVED: u64 = !(VALID | ADDRESS_FIELD);

bitflags! {
    /// The defined fields in one fixed-range type byte.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    struct FixedByte: u8 {
        /// The three-bit memory type.
        const TYPE = 0b111;
        /// Routes writes to system memory.
        const WR_MEM = 1 << 3;
        /// Routes reads to system memory.
        const RD_MEM = 1 << 4;
        /// Both extended routing fields.
        const ROUTING = Self::WR_MEM.bits() | Self::RD_MEM.bits();
        /// Bits reserved in both fixed-range encodings.
        const RESERVED = !(Self::TYPE.bits() | Self::ROUTING.bits());
    }
}

bitflags! {
    /// The `SYSCFG` fields that affect the MTRR mechanism.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    struct Syscfg: u64 {
        /// Enables the fixed-range `RdMem` and `WrMem` attributes.
        const FIX_DRAM_EN = 1 << 18;
        /// Allows software to read and write fixed-range `RdMem` and `WrMem`.
        const FIX_DRAM_MOD_EN = 1 << 19;
        /// Enables variable-range DRAM controls.
        const VAR_DRAM_EN = 1 << 20;
        /// Enables `TOP_MEM2`.
        const TOM2_EN = 1 << 21;
        /// Forces the default type above 4 GiB to write-back.
        const TOM2_FORCE_WB = 1 << 22;
        /// All MTRR-related fields in `SYSCFG`.
        const MTRR = Self::FIX_DRAM_EN.bits()
            | Self::FIX_DRAM_MOD_EN.bits()
            | Self::VAR_DRAM_EN.bits()
            | Self::TOM2_EN.bits()
            | Self::TOM2_FORCE_WB.bits();
    }
}

/// Those routing bits across one fixed-range register.
const FIXED_ROUTING: u64 = u64::from_le_bytes([FixedByte::ROUTING.bits(); 8]);

/// `SYSCFG`, which among much else holds the MTRR controls shadowed here.
const SYSCFG: u32 = 0xC001_0010;

const _: () = assert!(
    DEFAULT_TYPE_RESERVED == !0xCFF,
    "the default-type register reserves bits 63:12 and 9:8",
);
const _: () = assert!(
    BASE_RESERVED == 0xF00 && MASK_RESERVED == 0x7FF,
    "a variable range reserves bits 11:8 of its base and 10:0 of its mask",
);
const _: () = assert!(
    FixedByte::RESERVED.bits() == 0b1110_0000 && FIXED_ROUTING == 0x1818_1818_1818_1818,
    "one byte of a fixed range reserves the three bits above its two routing bits",
);
const _: () = assert!(
    VARIABLE_LAST - VARIABLE_FIRST + 1 == 16 && MAX_RANGES == 8,
    "the variable ranges are sixteen register numbers, in base and mask pairs",
);

#[cfg(test)]
mod tests {
    use super::{
        Capability, ENABLE, FIXED, FIXED_ENABLE, FIXED_SUPPORTED, MAX_RANGES, MTRR_CAP,
        MTRR_DEF_TYPE, Mtrrs, Range, Register, Routing, SYSCFG, Syscfg, SystemConfiguration, VALID,
        VARIABLE_FIRST, VARIABLE_LAST, WRITE_COMBINING_SUPPORTED, claims, intercepted,
    };
    use crate::msr::Fault;

    /// A shadow whose capability register reports `capability`, with every
    /// range as reset leaves it.
    fn shadow(capability: u64) -> Mtrrs {
        Mtrrs {
            capability: Capability(capability),
            default_type: 0,
            variable: [Range::EMPTY; MAX_RANGES],
            fixed: [0; FIXED.len()],
            syscfg: SystemConfiguration::from_value(0),
        }
    }

    /// The switch for the routing bits as this machine's firmware leaves it,
    /// and as Linux insists on it: clear.
    const FIXED_ROUTING_LOCKED: Routing = Routing { modifiable: false };

    /// The same switch as firmware holds it while programming the routing.
    const FIXED_ROUTING_OPEN: Routing = Routing { modifiable: true };

    /// What this machine's own capability register reports: eight variable
    /// ranges, the fixed ranges, and write-combining.
    const MACHINE: u64 = 8 | FIXED_SUPPORTED | WRITE_COMBINING_SUPPORTED;

    /// The write-back encoding, which is what firmware programs memory as.
    const WRITE_BACK: u64 = 6;

    /// The write-combining encoding, the one type a machine may not have.
    const WRITE_COMBINING: u64 = 1;

    /// The two answers have to be the same set. An index intercepted and not
    /// claimed stops the guest, because no handler owns it; one claimed and not
    /// intercepted is executed against the machine's own register, which is the
    /// whole thing being prevented.
    #[test]
    fn every_intercepted_register_is_claimed_and_nothing_else_is() {
        let mut counted = 0;
        for msr in intercepted() {
            assert!(
                claims(msr),
                "{msr:#x} is intercepted but claimed by nothing"
            );
            counted += 1;
        }
        let claimed = (0..=u16::MAX).filter(|&msr| claims(u32::from(msr))).count()
            + usize::from(claims(SYSCFG));
        assert_eq!(counted, claimed);
    }

    #[test]
    fn the_variable_block_alternates_base_and_mask() {
        assert_eq!(Register::of(VARIABLE_FIRST), Some(Register::Base(0)));
        assert_eq!(Register::of(VARIABLE_FIRST + 1), Some(Register::Mask(0)));
        assert_eq!(
            Register::of(VARIABLE_LAST - 1),
            Some(Register::Base(MAX_RANGES - 1))
        );
        assert_eq!(
            Register::of(VARIABLE_LAST),
            Some(Register::Mask(MAX_RANGES - 1))
        );
    }

    /// The eleven fixed registers are scattered numbers, and every number
    /// between them was measured on this machine to be no register at all — so
    /// they must not be claimed, or the guest would be answered for an access
    /// it is owed a fault for.
    #[test]
    fn a_number_between_the_fixed_registers_is_not_one() {
        for msr in (0x251..=0x257).chain(0x25A..=0x267) {
            assert_eq!(Register::of(msr), None, "{msr:#x} is not a register");
            assert!(!claims(msr));
        }
        for &msr in &FIXED {
            assert!(claims(msr), "{msr:#x} is one");
        }
    }

    #[test]
    fn a_variable_range_remembers_what_the_guest_put_in_it() {
        let mut mtrrs = shadow(MACHINE);
        let (base, mask) = (0x1000 | WRITE_BACK, 0xF000 | VALID);
        assert_eq!(
            mtrrs.write(VARIABLE_FIRST, base, FIXED_ROUTING_LOCKED),
            Ok(())
        );
        assert_eq!(
            mtrrs.write(VARIABLE_FIRST + 1, mask, FIXED_ROUTING_LOCKED),
            Ok(())
        );
        assert_eq!(mtrrs.read(VARIABLE_FIRST, FIXED_ROUTING_LOCKED), Ok(base));
        assert_eq!(
            mtrrs.read(VARIABLE_FIRST + 1, FIXED_ROUTING_LOCKED),
            Ok(mask)
        );
    }

    /// The whole of what the machine was measured accepting and refusing in a
    /// variable range, register for register and bit for bit.
    ///
    /// The address ladder is the part worth pinning: refusal begins at the
    /// width `CPUID` reports, so a mistake in reading that width would show
    /// up here as a range the guest may not describe on a machine where it
    /// may.
    #[test]
    fn a_variable_range_accepts_and_refuses_exactly_what_the_machine_does() {
        let mut mtrrs = shadow(MACHINE);
        let (base, mask) = (VARIABLE_FIRST, VARIABLE_FIRST + 1);

        for encoding in [0, 1, 4, 5, 6] {
            assert_eq!(mtrrs.write(base, encoding, FIXED_ROUTING_LOCKED), Ok(()));
        }
        for encoding in [2, 3, 7] {
            assert_eq!(
                mtrrs.write(base, encoding, FIXED_ROUTING_LOCKED),
                Err(Fault)
            );
        }
        // The type is the whole low byte and only the five encodings are in it,
        // so every byte above seven is refused however its low three bits read.
        for byte in [0x08, 0x0E, 0x40, 0x86, 0xFF] {
            assert_eq!(mtrrs.write(base, byte, FIXED_ROUTING_LOCKED), Err(Fault));
        }
        for bit in 8..12 {
            assert_eq!(
                mtrrs.write(base, (1 << bit) | WRITE_BACK, FIXED_ROUTING_LOCKED),
                Err(Fault),
                "base bit {bit}"
            );
        }
        for bit in 0..11 {
            assert_eq!(
                mtrrs.write(mask, 1 << bit, FIXED_ROUTING_LOCKED),
                Err(Fault),
                "mask bit {bit}"
            );
        }
        let width = u32::from(processor::physical_address_bits());
        for bit in 12..u64::BITS {
            let refused = bit >= width;
            assert_eq!(
                mtrrs
                    .write(base, (1 << bit) | WRITE_BACK, FIXED_ROUTING_LOCKED)
                    .is_err(),
                refused,
                "base address bit {bit}"
            );
            assert_eq!(
                mtrrs.write(mask, 1 << bit, FIXED_ROUTING_LOCKED).is_err(),
                refused,
                "mask address bit {bit}"
            );
        }
    }

    /// The count answered in the capability register is the count served, or
    /// the guest is told about ranges it cannot then use.
    #[test]
    fn a_range_the_capability_register_does_not_report_does_not_exist() {
        let mut two = shadow(2 | FIXED_SUPPORTED | WRITE_COMBINING_SUPPORTED);
        assert_eq!(two.read(VARIABLE_FIRST + 2, FIXED_ROUTING_LOCKED), Ok(0));
        assert_eq!(
            two.read(VARIABLE_FIRST + 4, FIXED_ROUTING_LOCKED),
            Err(Fault)
        );
        assert_eq!(
            two.write(VARIABLE_FIRST + 4, WRITE_BACK, FIXED_ROUTING_LOCKED),
            Err(Fault)
        );
    }

    /// Measured: refused even when written the value it already holds.
    #[test]
    fn the_capability_register_refuses_every_write() {
        let mut mtrrs = shadow(MACHINE);
        assert_eq!(mtrrs.read(MTRR_CAP, FIXED_ROUTING_LOCKED), Ok(MACHINE));
        assert_eq!(
            mtrrs.write(MTRR_CAP, MACHINE, FIXED_ROUTING_LOCKED),
            Err(Fault)
        );
    }

    #[test]
    fn the_default_type_register_takes_its_three_fields_and_nothing_else() {
        let mut mtrrs = shadow(MACHINE);
        // What this machine holds: uncacheable, with both enables set.
        let firmware = ENABLE | FIXED_ENABLE;
        assert_eq!(
            mtrrs.write(MTRR_DEF_TYPE, firmware, FIXED_ROUTING_LOCKED),
            Ok(())
        );
        assert_eq!(
            mtrrs.read(MTRR_DEF_TYPE, FIXED_ROUTING_LOCKED),
            Ok(firmware)
        );
        let enabled = ENABLE | FIXED_ENABLE | WRITE_BACK;
        assert_eq!(
            mtrrs.write(MTRR_DEF_TYPE, enabled, FIXED_ROUTING_LOCKED),
            Ok(())
        );
        assert_eq!(mtrrs.read(MTRR_DEF_TYPE, FIXED_ROUTING_LOCKED), Ok(enabled));
        for bit in [8, 9, 12, 31, 32, 63] {
            assert_eq!(
                mtrrs.write(MTRR_DEF_TYPE, enabled | (1 << bit), FIXED_ROUTING_LOCKED),
                Err(Fault),
                "default-type bit {bit}"
            );
        }
        assert_eq!(
            mtrrs.write(MTRR_DEF_TYPE, 3, FIXED_ROUTING_LOCKED),
            Err(Fault)
        );
    }

    /// The values this machine's firmware really leaves in the fixed ranges,
    /// which a guest reads and writes straight back. Faulting any of them would
    /// fault the guest for its own read.
    #[test]
    fn a_fixed_range_takes_what_firmware_leaves_in_it() {
        let mut mtrrs = shadow(MACHINE);
        for firmware in [0x0606_0606_0606_0606, 0x0505_0505_0505_0505, 0] {
            assert_eq!(
                mtrrs.write(FIXED[0], firmware, FIXED_ROUTING_LOCKED),
                Ok(())
            );
            assert_eq!(mtrrs.read(FIXED[0], FIXED_ROUTING_LOCKED), Ok(firmware));
        }
        // The type of a fixed range is three bits, so the three above the routing
        // bits are reserved whichever form the byte is in.
        for bit in 5..8 {
            let value = u64::from_le_bytes([1 << bit; 8]);
            assert_eq!(
                mtrrs.write(FIXED[0], value, FIXED_ROUTING_LOCKED),
                Err(Fault),
                "fixed byte bit {bit}"
            );
        }
        // And an undefined type is refused byte by byte, not only in the lowest.
        for slot in 0..8 {
            let value = u64::from(2u8) << (slot * u8::BITS);
            assert_eq!(
                mtrrs.write(FIXED[0], value, FIXED_ROUTING_LOCKED),
                Err(Fault),
                "fixed byte {slot}"
            );
        }
    }

    /// Measured: this machine's `SYSCFG` reports the switch clear, so the two
    /// routing bits read as zero and a write of them is dropped rather than
    /// faulted — and what lies underneath survives, which is what lets firmware
    /// program the routing and then close the switch behind itself.
    #[test]
    fn the_routing_bits_of_a_fixed_range_follow_the_machine_s_switch() {
        let routed = u64::from_le_bytes([0x1E; 8]);
        let bare = u64::from_le_bytes([0x06; 8]);

        let mut locked = shadow(MACHINE);
        assert_eq!(locked.write(FIXED[0], routed, FIXED_ROUTING_LOCKED), Ok(()));
        assert_eq!(locked.read(FIXED[0], FIXED_ROUTING_LOCKED), Ok(bare));

        let mut open = shadow(MACHINE);
        assert_eq!(open.write(FIXED[0], routed, FIXED_ROUTING_OPEN), Ok(()));
        assert_eq!(open.read(FIXED[0], FIXED_ROUTING_OPEN), Ok(routed));
        // Closed behind firmware, the routing is still stored and still hidden.
        assert_eq!(open.read(FIXED[0], FIXED_ROUTING_LOCKED), Ok(bare));
        assert_eq!(open.write(FIXED[0], bare, FIXED_ROUTING_LOCKED), Ok(()));
        assert_eq!(open.read(FIXED[0], FIXED_ROUTING_OPEN), Ok(routed));
    }

    #[test]
    fn syscfg_keeps_mtrr_controls_in_the_guest_shadow() {
        let guest = Syscfg::FIX_DRAM_EN | Syscfg::TOM2_FORCE_WB;
        let hardware = guest.bits() | 1 << 23;
        let shadow = SystemConfiguration::from_value(guest.bits());

        assert_eq!(shadow.mtrr, guest);
        assert_eq!(shadow.visible(hardware), hardware);
        assert!(!shadow.routing().modifiable);
    }

    #[test]
    fn syscfg_writes_preserve_host_mtrr_controls() {
        let hardware = Syscfg::FIX_DRAM_EN.bits() | 1 << 23;
        let guest = Syscfg::FIX_DRAM_MOD_EN | Syscfg::TOM2_EN;
        let (host, mtrr) = SystemConfiguration::prepare_write(hardware, guest.bits() | 1 << 24);

        assert_eq!(host & Syscfg::MTRR.bits(), hardware & Syscfg::MTRR.bits());
        assert_eq!(host & !Syscfg::MTRR.bits(), 1 << 24);
        assert_eq!(mtrr, guest);
    }

    #[test]
    fn a_machine_with_no_fixed_ranges_has_none_to_answer_for() {
        let mut none = shadow(8 | WRITE_COMBINING_SUPPORTED);
        assert_eq!(none.read(FIXED[0], FIXED_ROUTING_LOCKED), Err(Fault));
        assert_eq!(none.write(FIXED[0], 0, FIXED_ROUTING_LOCKED), Err(Fault));
    }

    #[test]
    fn write_combining_is_refused_where_the_capability_register_denies_it() {
        let mut with = shadow(MACHINE);
        assert_eq!(
            with.write(MTRR_DEF_TYPE, WRITE_COMBINING, FIXED_ROUTING_LOCKED),
            Ok(())
        );
        let mut without = shadow(8 | FIXED_SUPPORTED);
        assert_eq!(
            without.write(MTRR_DEF_TYPE, WRITE_COMBINING, FIXED_ROUTING_LOCKED),
            Err(Fault)
        );
    }
}
