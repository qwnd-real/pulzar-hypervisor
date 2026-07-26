//! Message interrupts, and where the tables behind them live in memory.
//!
//! A message interrupt is a memory write the device performs instead of
//! asserting a wire. The two mechanisms for it differ in where the address and
//! the payload of that write are kept: the older one keeps them in
//! configuration space, so a device can have at most 32 of them and they must
//! be consecutive; the newer one keeps them in a table in the device's own
//! memory, so it can have up to 2048 and each is independent.
//!
//! That difference is the whole reason this module exists in the shape it does.
//! An older message interrupt is fully described by configuration space, which
//! a hypervisor already sees every access to. A newer one is described by a
//! table somewhere behind a base address register, which a hypervisor sees
//! nothing of until it arranges to — and arranging to means knowing the
//! physical address, which is what [`MsiX::table`] answers.
//!
//! # The count is stored one short
//!
//! The table size field holds the number of entries minus one, so that a
//! function with a single entry can encode it in a field of zeroes and a field
//! of all ones means the full 2048. Reading it as the count directly is the
//! single most common way to get this wrong, and it fails quietly: every
//! computed extent is 16 bytes short, which usually still lands on the same
//! page and so still works, until the day the table ends exactly on a boundary.
//!
//! # A table is only as reachable as the register that locates it
//!
//! The register a table lives behind is named by a three-bit field, and two of
//! its eight values are reserved. The other six name base address registers
//! that may be unimplemented, may describe I/O space rather than memory, or may
//! be the upper half of a wide register — which is never the right half to
//! name, because a wide register is located by its lower one. None of those
//! yields an address, and none of them is a reason to refuse the machine: the
//! region is reported absent, with the reason in the log.

use core::fmt::{self, Display, Formatter};

use bitfield_struct::bitfield;
use log::warn;
use x86_64::PhysAddr;

use crate::{
    Offset, PciError,
    access::Config,
    bar::{Bar, SLOTS},
};

/// Bytes a page of memory occupies, which is the granularity anything
/// intercepting one of these regions can work at.
pub const PAGE_BYTES: u64 = 4096;

/// A range of physical memory one function owns.
///
/// Carries the byte extent the specification defines and, separately, the pages
/// that extent touches. Both matter and they are not the same: the extent is
/// what the device uses, and the pages are what a hypervisor can protect, since
/// nested paging has no finer granularity than a page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    bar: u8,
    phys: PhysAddr,
    bytes: u64,
}

impl Region {
    /// Which base address register locates this region.
    #[must_use]
    pub const fn bar(&self) -> u8 {
        self.bar
    }

    /// Where the region begins.
    #[must_use]
    pub const fn phys(&self) -> PhysAddr {
        self.phys
    }

    /// Bytes the region occupies.
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Last byte inside the region.
    #[must_use]
    pub fn last(&self) -> u64 {
        self.phys.as_u64() + self.bytes - 1
    }

    /// First page the region touches.
    #[must_use]
    pub fn first_page(&self) -> PhysAddr {
        self.phys.align_down(PAGE_BYTES)
    }

    /// Pages the region touches, which is what protecting it costs.
    ///
    /// More than the extent divided by the page size whenever the region is not
    /// page-aligned, which the specification permits: only eight-byte alignment
    /// is required of it.
    #[must_use]
    pub fn pages(&self) -> u64 {
        (self.last() - self.first_page().as_u64()) / PAGE_BYTES + 1
    }

    /// Whether `phys` falls inside the region.
    #[must_use]
    pub fn contains(&self, phys: PhysAddr) -> bool {
        phys >= self.phys && phys.as_u64() <= self.last()
    }

    /// Last page the region touches.
    #[must_use]
    pub fn last_page(&self) -> PhysAddr {
        self.phys.align_down(PAGE_BYTES) + (self.pages() - 1) * PAGE_BYTES
    }

    /// Whether this region shares a page with `other`.
    ///
    /// Worth knowing before either is intercepted: two regions on one page
    /// cannot be given different treatment, and a region sharing a page with
    /// registers that are not a table at all means intercepting it traps
    /// accesses that have nothing to do with interrupts.
    #[must_use]
    pub fn shares_page_with(&self, other: &Self) -> bool {
        self.first_page() <= other.last_page() && other.first_page() <= self.last_page()
    }
}

impl Display for Region {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "bar {} {:#x}..={:#x} ({} pages from {:#x})",
            self.bar,
            self.phys,
            self.last(),
            self.pages(),
            self.first_page()
        )
    }
}

/// The older message interrupt mechanism, kept entirely in configuration space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Msi {
    at: Offset,
    address: Offset,
    upper: Option<Offset>,
    data: Offset,
    mask: Option<Offset>,
    pending: Option<Offset>,
    capable: u16,
    enabled: u16,
    on: bool,
}

impl Msi {
    /// Where the capability begins.
    #[must_use]
    pub const fn at(&self) -> Offset {
        self.at
    }

    /// The register holding the address messages are written to.
    #[must_use]
    pub const fn address(&self) -> Offset {
        self.address
    }

    /// The register holding the upper half of that address, if the function can
    /// write above four gigabytes.
    #[must_use]
    pub const fn upper(&self) -> Option<Offset> {
        self.upper
    }

    /// The register holding the payload written there.
    ///
    /// Where this sits depends on whether the function has an upper address
    /// register, which is the one thing about this capability's layout that
    /// moves.
    #[must_use]
    pub const fn data(&self) -> Offset {
        self.data
    }

    /// The register masking individual vectors, if the function has one.
    #[must_use]
    pub const fn mask(&self) -> Option<Offset> {
        self.mask
    }

    /// The register reporting vectors that arrived while masked, if the
    /// function has one.
    #[must_use]
    pub const fn pending(&self) -> Option<Offset> {
        self.pending
    }

    /// How many vectors the function can be given.
    #[must_use]
    pub const fn capable(&self) -> u16 {
        self.capable
    }

    /// How many vectors it has been given.
    #[must_use]
    pub const fn enabled(&self) -> u16 {
        self.enabled
    }

    /// Whether the function is delivering these at all.
    #[must_use]
    pub const fn on(&self) -> bool {
        self.on
    }

    /// Whether the function can write its messages above four gigabytes.
    #[must_use]
    pub const fn is_wide(&self) -> bool {
        self.upper.is_some()
    }
}

impl Display for Msi {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "msi at {} {} of {} vectors, {}, {}",
            self.at,
            self.enabled,
            self.capable,
            if self.is_wide() { "64-bit" } else { "32-bit" },
            if self.on { "enabled" } else { "disabled" },
        )
    }
}

/// The newer message interrupt mechanism, whose vectors live in a table in the
/// function's own memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MsiX {
    at: Offset,
    entries: u16,
    table: Option<Region>,
    pending: Option<Region>,
    on: bool,
    masked: bool,
}

impl MsiX {
    /// Where the capability begins.
    #[must_use]
    pub const fn at(&self) -> Offset {
        self.at
    }

    /// How many vectors the table holds.
    ///
    /// Always at least one and at most 2048: the field it comes from cannot
    /// encode a function that has this capability and no vectors.
    #[must_use]
    pub const fn entries(&self) -> u16 {
        self.entries
    }

    /// Where the vector table is in physical memory.
    ///
    /// `None` when the register naming it does not describe reachable memory,
    /// which the log will have said more about.
    #[must_use]
    pub const fn table(&self) -> Option<Region> {
        self.table
    }

    /// Where the array of pending bits is in physical memory.
    #[must_use]
    pub const fn pending(&self) -> Option<Region> {
        self.pending
    }

    /// Whether the function is delivering these at all.
    #[must_use]
    pub const fn on(&self) -> bool {
        self.on
    }

    /// Whether every vector is masked at the function level, which overrides
    /// the per-vector masks in the table.
    #[must_use]
    pub const fn masked(&self) -> bool {
        self.masked
    }
}

impl Display for MsiX {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "msi-x at {} {} vectors", self.at, self.entries)?;
        match self.table {
            Some(table) => write!(formatter, ", table {table}")?,
            None => formatter.write_str(", table unreachable")?,
        }
        if self.on {
            formatter.write_str(", enabled")?;
        }
        if self.masked {
            formatter.write_str(", masked")?;
        }
        Ok(())
    }
}

/// Reads the older capability.
///
/// # Errors
///
/// Whatever the configuration mechanism reported.
pub(crate) fn msi(config: &Config, at: Offset) -> Result<Msi, PciError> {
    let control = MsiControl::from_bits(config.u16(at.plus(CONTROL))?);
    let (upper, data) = if control.wide() {
        (Some(at.plus(UPPER_ADDRESS)), at.plus(WIDE_DATA))
    } else {
        (None, at.plus(NARROW_DATA))
    };
    // Both follow the payload register wherever it landed, and the extended
    // payload that may sit beside it does not move them.
    let (mask, pending) = if control.per_vector_masking() {
        (
            Some(data.plus(MASK_AFTER_DATA)),
            Some(data.plus(PENDING_AFTER_DATA)),
        )
    } else {
        (None, None)
    };
    Ok(Msi {
        at,
        address: at.plus(ADDRESS),
        upper,
        data,
        mask,
        pending,
        capable: vectors(control.capable()),
        enabled: vectors(control.enabled()),
        on: control.on(),
    })
}

/// Reads the newer capability, resolving its two regions through `bars`.
///
/// # Errors
///
/// Whatever the configuration mechanism reported.
pub(crate) fn msi_x(config: &Config, at: Offset, bars: &[Bar; SLOTS]) -> Result<MsiX, PciError> {
    let control = MsiXControl::from_bits(config.u16(at.plus(CONTROL))?);
    // The field holds one less than the count, so that a function with a single
    // vector encodes it as zeroes and a field of all ones means the full 2048.
    let entries = control.last_entry() + 1;
    let table = locate(
        config,
        bars,
        config.u32(at.plus(TABLE))?,
        u64::from(entries) * ENTRY_BYTES,
        "vector table",
    );
    let pending = locate(
        config,
        bars,
        config.u32(at.plus(PENDING_ARRAY))?,
        u64::from(entries).div_ceil(PENDING_BITS_PER_WORD) * PENDING_WORD_BYTES,
        "pending bit array",
    );
    let found = MsiX {
        at,
        entries,
        table,
        pending,
        on: control.on(),
        masked: control.masked(),
    };
    if let (Some(table), Some(pending)) = (table, pending)
        && table.shares_page_with(&pending)
    {
        warn!(
            "pci: {} keeps its msi-x table and pending bits on the same page",
            config.address()
        );
    }
    Ok(found)
}

/// Resolves one table-locating register against the function's own base
/// address registers.
///
/// The three low bits name a register and the rest is a byte offset into it.
/// Both halves can be wrong in ways that are the device's fault rather than
/// this crate's, so every one of them ends the same way: nothing located, and a
/// line in the log saying which.
fn locate(
    config: &Config,
    bars: &[Bar; SLOTS],
    value: u32,
    bytes: u64,
    what: &str,
) -> Option<Region> {
    let locator = Locator::from_bits(value);
    let bar = locator.bar();
    let offset = u64::from(locator.offset()) << OFFSET_SHIFT;
    // Indexing is what refuses the two reserved register numbers: there are
    // six registers and the field holds eight values.
    let Some(base) = bars.get(usize::from(bar)).and_then(|bar| bar.memory_base()) else {
        warn!(
            "pci: {} locates its msi-x {what} in register {bar}, which describes no memory",
            config.address()
        );
        return None;
    };
    let Some(phys) = extent(base, offset, bytes) else {
        warn!(
            "pci: {} locates its msi-x {what} {offset:#x} past {base:#x}, which is unreachable",
            config.address()
        );
        return None;
    };
    Some(Region { bar, phys, bytes })
}

/// Where a region of `bytes` bytes at `offset` past `base` begins.
///
/// `None` unless the whole of it lies inside the physical address space, so
/// that a region which is reported is one that can be read from end to end.
fn extent(base: PhysAddr, offset: u64, bytes: u64) -> Option<PhysAddr> {
    let start = base.as_u64().checked_add(offset)?;
    let last = start.checked_add(bytes - 1)?;
    PhysAddr::try_new(last).ok()?;
    PhysAddr::try_new(start).ok()
}

/// How many vectors a three-bit encoding names.
///
/// The field holds the base-two logarithm of the count, and only its first six
/// values are defined. A function using one of the other two is describing more
/// vectors than the mechanism has ever had, so it is held to the maximum rather
/// than believed.
fn vectors(field: u8) -> u16 {
    1 << field.min(MAX_VECTOR_SHIFT)
}

#[bitfield(u16)]
#[derive(PartialEq, Eq)]
/// The older capability's control register.
struct MsiControl {
    /// Whether the function is delivering these at all.
    pub on: bool,
    /// How many vectors the function can be given, as the base-two logarithm
    /// of the count.
    #[bits(3)]
    pub capable: u8,
    /// How many it has been given, in the same encoding.
    #[bits(3)]
    pub enabled: u8,
    /// Whether the function can write its messages above four gigabytes, which
    /// is what decides where the payload register sits.
    pub wide: bool,
    /// Whether the function can mask its vectors individually.
    pub per_vector_masking: bool,
    /// Whether the payload register is joined by an extended one.
    pub extended_data: bool,
    /// Whether that extended payload is in use.
    pub extended_data_enable: bool,
    /// Reserved.
    #[bits(5)]
    __: u8,
}

#[bitfield(u16)]
#[derive(PartialEq, Eq)]
/// The newer capability's control register.
struct MsiXControl {
    /// One less than the number of entries in the vector table.
    #[bits(11)]
    pub last_entry: u16,
    /// Reserved.
    #[bits(3)]
    __: u8,
    /// Whether every vector is masked at the function level, whatever the
    /// table says.
    pub masked: bool,
    /// Whether the function is delivering these at all.
    pub on: bool,
}

#[bitfield(u32)]
#[derive(PartialEq, Eq)]
/// A register locating one of the newer capability's two tables.
struct Locator {
    /// Which base address register the table lives behind. Six and seven are
    /// reserved, and a function has only six registers, so those name nothing.
    #[bits(3)]
    pub bar: u8,
    /// How far into that register the table begins, in eight-byte units —
    /// which is why a table is always eight-byte aligned.
    #[bits(29)]
    pub offset: u32,
}

/// Offset of the control register from the start of either capability.
const CONTROL: u16 = 2;

/// Offset of the older capability's address register.
const ADDRESS: u16 = 4;

/// Offset of the older capability's upper address register, when it has one.
const UPPER_ADDRESS: u16 = 8;

/// Offset of the payload register on a function that cannot write above four
/// gigabytes.
const NARROW_DATA: u16 = 8;

/// Offset of the payload register on a function that can.
const WIDE_DATA: u16 = 12;

/// Bytes from the payload register to the per-vector mask register.
const MASK_AFTER_DATA: u16 = 4;

/// Bytes from the payload register to the per-vector pending register.
const PENDING_AFTER_DATA: u16 = 8;

/// Largest shift a vector count field is allowed to name, which is 32 vectors.
/// The two encodings above it describe more vectors than this mechanism has
/// ever had.
const MAX_VECTOR_SHIFT: u8 = 5;

/// Offset of the register locating the vector table.
const TABLE: u16 = 4;

/// Offset of the register locating the pending bit array.
const PENDING_ARRAY: u16 = 8;

/// Bits a locating register's offset field is shifted by.
const OFFSET_SHIFT: u8 = 3;

/// Bytes one vector table entry occupies: an address, an upper address, a
/// payload and a mask.
const ENTRY_BYTES: u64 = 16;

/// Vectors one word of the pending bit array accounts for.
const PENDING_BITS_PER_WORD: u64 = 64;

/// Bytes one word of the pending bit array occupies.
const PENDING_WORD_BYTES: u64 = 8;
