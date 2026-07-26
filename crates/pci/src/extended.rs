//! The capabilities that live above the first 256 bytes.
//!
//! These exist only on PCI Express, and only where extended configuration space
//! is actually reachable — which means through the memory aperture, and not
//! below a bridge down onto plain PCI. Each is decoded on demand from the
//! offset [`crate::capability`] recorded, rather than during the walk, so that
//! a function nobody asks about costs nothing but the walk itself.
//!
//! # What is read and what is deliberately not
//!
//! Error reporting is read for its status registers, because a machine that
//! arrives at the hypervisor with errors already latched is worth saying so
//! about, and because every one of those registers is cleared by writing a one
//! to it — so reading them is the only thing that can be done without
//! destroying what another owner may still want.
//!
//! Virtualization is read for its geometry, and stops short of claiming to know
//! where a virtual function's registers are. The registers that would say are
//! base address registers like any others: they report the first virtual
//! function's base, and the stride between one and the next is a function of
//! their *size*, which cannot be learned without writing to them. So this
//! module reports the base, the count, and the routing arithmetic, and says
//! plainly that the per-function addresses are not available until sizing runs.

use core::fmt::{self, Display, Formatter};

use bitfield_struct::bitfield;

use crate::{
    Offset, PciError,
    access::Config,
    bar::{self, Bar, SLOTS},
};

/// What a function has noticed going wrong, in the detail PCI Express added.
///
/// A snapshot of the status registers as they were found. Every bit in them is
/// cleared by writing a one, so nothing here writes: what is latched belongs to
/// whoever set up the error reporting, and a hypervisor that cleared it would
/// be destroying the only record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Aer {
    at: Offset,
    version: u8,
    uncorrectable: u32,
    uncorrectable_mask: u32,
    correctable: u32,
    correctable_mask: u32,
    rooted: bool,
}

impl Aer {
    /// Where the capability begins.
    #[must_use]
    pub const fn at(&self) -> Offset {
        self.at
    }

    /// Which version of the capability this is.
    #[must_use]
    pub const fn version(&self) -> u8 {
        self.version
    }

    /// Errors that could not be corrected and have not been cleared.
    #[must_use]
    pub const fn uncorrectable(&self) -> u32 {
        self.uncorrectable
    }

    /// Which of those the function has been told not to report.
    #[must_use]
    pub const fn uncorrectable_mask(&self) -> u32 {
        self.uncorrectable_mask
    }

    /// Errors that were corrected and have not been cleared.
    #[must_use]
    pub const fn correctable(&self) -> u32 {
        self.correctable
    }

    /// Which of those the function has been told not to report.
    #[must_use]
    pub const fn correctable_mask(&self) -> u32 {
        self.correctable_mask
    }

    /// Whether the function has the root registers, which only a root port or
    /// an event collector does.
    #[must_use]
    pub const fn rooted(&self) -> bool {
        self.rooted
    }

    /// Whether anything is latched that was not masked away.
    #[must_use]
    pub const fn reporting(&self) -> bool {
        self.uncorrectable & !self.uncorrectable_mask != 0
            || self.correctable & !self.correctable_mask != 0
    }

    /// Reads the capability at `at`.
    ///
    /// `rooted` says whether this function is one of the two kinds that have
    /// the root registers. They are not read — nothing here needs them — but
    /// whether they exist is worth carrying, because a caller that does want
    /// them must not read them from a function that has something else there.
    ///
    /// # Errors
    ///
    /// Whatever the configuration mechanism reported.
    pub(crate) fn decode(config: &Config, at: Offset, rooted: bool) -> Result<Self, PciError> {
        Ok(Self {
            at,
            version: version(config, at)?,
            uncorrectable: config.u32(at.plus(UNCORRECTABLE_STATUS))?,
            uncorrectable_mask: config.u32(at.plus(UNCORRECTABLE_MASK))?,
            correctable: config.u32(at.plus(CORRECTABLE_STATUS))?,
            correctable_mask: config.u32(at.plus(CORRECTABLE_MASK))?,
            rooted,
        })
    }
}

impl Display for Aer {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "aer v{} uncorrectable {:#010x}/{:#010x} correctable {:#010x}/{:#010x}",
            self.version,
            self.uncorrectable,
            self.uncorrectable_mask,
            self.correctable,
            self.correctable_mask
        )
    }
}

/// How a device's functions are numbered, when there are more than eight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ari {
    at: Offset,
    next: u8,
    group: u8,
}

impl Ari {
    /// Where the capability begins.
    #[must_use]
    pub const fn at(&self) -> Offset {
        self.at
    }

    /// The next function of this device, or zero at the end of the chain.
    ///
    /// This is how a device with more than eight functions lists them: as a
    /// chain rather than as a range, so that the numbers need not be
    /// contiguous and a walk need not probe all 256.
    #[must_use]
    pub const fn next(&self) -> u8 {
        self.next
    }

    /// Which function group this function has been put in.
    #[must_use]
    pub const fn group(&self) -> u8 {
        self.group
    }

    /// Reads the capability at `at`.
    ///
    /// # Errors
    ///
    /// Whatever the configuration mechanism reported.
    pub(crate) fn decode(config: &Config, at: Offset) -> Result<Self, PciError> {
        let capability = AriCapability::from_bits(config.u16(at.plus(ARI_CAPABILITY))?);
        let control = AriControl::from_bits(config.u16(at.plus(ARI_CONTROL))?);
        Ok(Self {
            at,
            next: capability.next(),
            group: control.group(),
        })
    }
}

impl Display for Ari {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "ari next function {}", self.next)
    }
}

#[bitfield(u16)]
#[derive(PartialEq, Eq)]
/// What a port will let the devices below it do to one another.
///
/// The same fields appear in the capability register, saying what the port can
/// enforce, and in the control register, saying what it is enforcing. A
/// hypervisor that means to keep two devices apart needs the second to match
/// the first.
pub struct Control {
    /// Transactions between two functions of the same device are visible to the
    /// port above.
    pub source_validation: bool,
    /// Transactions are blocked from being translated in a way that would
    /// bypass the port.
    pub translation_blocking: bool,
    /// A request between two ports of the same switch is sent upward rather
    /// than across.
    pub request_redirect: bool,
    /// A completion between two ports of the same switch is sent upward.
    pub completion_redirect: bool,
    /// Upward-sent requests may be forwarded back down.
    pub upstream_forwarding: bool,
    /// Transactions are checked against the egress control vector.
    pub egress_control: bool,
    /// Requests carrying an already-translated address are checked too.
    pub translated_blocking: bool,
    /// Reserved in the control register; the size of the egress control vector
    /// in the capability register.
    #[bits(9)]
    __: u16,
}

impl Control {
    /// Whether every control in `capable` is switched on here.
    #[must_use]
    const fn enforces(self, capable: Self) -> bool {
        capable.into_bits() & !self.into_bits() & ENFORCEABLE == 0
    }
}

/// What a port is willing to let pass between the devices below it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Acs {
    at: Offset,
    capable: Control,
    control: Control,
    egress_entries: u8,
}

impl Acs {
    /// Where the capability begins.
    #[must_use]
    pub const fn at(&self) -> Offset {
        self.at
    }

    /// What the port can enforce.
    #[must_use]
    pub const fn capable(&self) -> Control {
        self.capable
    }

    /// What it is enforcing.
    #[must_use]
    pub const fn control(&self) -> Control {
        self.control
    }

    /// How many entries the egress control vector has.
    #[must_use]
    pub const fn egress_entries(&self) -> u8 {
        self.egress_entries
    }

    /// Whether every control the port can enforce is switched on.
    ///
    /// Which is what has to be true before two devices below one port can be
    /// given to different owners: anything the port could enforce and is not
    /// enforcing is a path between them that nothing is watching.
    #[must_use]
    pub const fn isolating(&self) -> bool {
        self.control.enforces(self.capable)
    }

    /// Reads the capability at `at`.
    ///
    /// # Errors
    ///
    /// Whatever the configuration mechanism reported.
    pub(crate) fn decode(config: &Config, at: Offset) -> Result<Self, PciError> {
        let capability = config.u16(at.plus(ACS_CAPABILITY))?;
        Ok(Self {
            at,
            capable: Control::from_bits(capability),
            control: Control::from_bits(config.u16(at.plus(ACS_CONTROL))?),
            egress_entries: u8::try_from(capability >> ACS_EGRESS_SHIFT).unwrap_or_default(),
        })
    }
}

impl Display for Acs {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "acs capable {:?} enforcing {:?}",
            self.capable, self.control
        )
    }
}

/// One base address register whose size software may change.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Resizable {
    bar: u8,
    supported: u32,
    size: u8,
}

impl Resizable {
    /// Which base address register this describes.
    #[must_use]
    pub const fn bar(&self) -> u8 {
        self.bar
    }

    /// Which sizes the register can be set to, as one bit per power of two
    /// starting at one megabyte.
    #[must_use]
    pub const fn supported(&self) -> u32 {
        self.supported
    }

    /// Bytes the register currently decodes.
    ///
    /// The encoding is the base-two logarithm of the size in megabytes, so the
    /// smallest a resizable register can be is one megabyte. Zero for an
    /// encoding naming a size larger than the address space, which no register
    /// should report and which is not a number to hand back wrapped.
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        match ONE_MEGABYTE.checked_shl(self.size as u32) {
            Some(bytes) => bytes,
            None => 0,
        }
    }
}

impl Display for Resizable {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "bar {} at {:#x} bytes", self.bar, self.bytes())
    }
}

/// Which of a function's base address registers can be resized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResizableBars {
    at: Offset,
    entries: [Resizable; SLOTS],
    count: u8,
}

impl ResizableBars {
    /// Where the capability begins.
    #[must_use]
    pub const fn at(&self) -> Offset {
        self.at
    }

    /// The registers this capability describes.
    #[must_use]
    pub fn entries(&self) -> &[Resizable] {
        &self.entries[..usize::from(self.count).min(SLOTS)]
    }

    /// Reads the capability at `at`.
    ///
    /// How many registers are described lives in the first entry's control
    /// register and nowhere else, so the count has to be read before the walk
    /// that uses it — and a count above six describes more registers than a
    /// function has, which is held to six rather than believed.
    ///
    /// # Errors
    ///
    /// Whatever the configuration mechanism reported.
    pub(crate) fn decode(config: &Config, at: Offset) -> Result<Self, PciError> {
        let first = ResizableControl::from_bits(config.u32(at.plus(RESIZABLE_CONTROL))?);
        let count = first.count().clamp(1, SLOTS_AS_U8);
        let mut entries = [Resizable::default(); SLOTS];
        for (index, entry) in entries.iter_mut().enumerate().take(usize::from(count)) {
            let step = u16::try_from(index).unwrap_or_default() * RESIZABLE_STRIDE;
            let capability =
                ResizableCapability::from_bits(config.u32(at.plus(RESIZABLE_CAPABILITY + step))?);
            let control =
                ResizableControl::from_bits(config.u32(at.plus(RESIZABLE_CONTROL + step))?);
            *entry = Resizable {
                bar: control.bar(),
                supported: capability.supported(),
                size: control.size(),
            };
        }
        Ok(Self { at, entries, count })
    }
}

/// The functions a physical function can conjure, and where their registers
/// would be.
///
/// Everything here is read and nothing is derived beyond the routing
/// arithmetic. In particular the addresses of a virtual function's registers
/// are *not* computed: the registers below give the first one's base, and the
/// distance to the next depends on how large each is, which only a write can
/// establish. What this does give is enough to do that later without walking
/// configuration space again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sriov {
    at: Offset,
    initial: u16,
    total: u16,
    current: u16,
    first_offset: u16,
    stride: u16,
    device: u16,
    page_size: u64,
    bars: [Bar; SLOTS],
    on: bool,
    decoding: bool,
}

impl Sriov {
    /// Where the capability begins.
    #[must_use]
    pub const fn at(&self) -> Offset {
        self.at
    }

    /// How many virtual functions firmware left configured.
    #[must_use]
    pub const fn initial(&self) -> u16 {
        self.initial
    }

    /// How many the physical function could have.
    #[must_use]
    pub const fn total(&self) -> u16 {
        self.total
    }

    /// How many it is set to have.
    #[must_use]
    pub const fn current(&self) -> u16 {
        self.current
    }

    /// The routing identifier of the first virtual function, as a distance from
    /// the physical function's own.
    #[must_use]
    pub const fn first_offset(&self) -> u16 {
        self.first_offset
    }

    /// The distance from one virtual function's routing identifier to the next.
    #[must_use]
    pub const fn stride(&self) -> u16 {
        self.stride
    }

    /// The device identifier the virtual functions report, which is not the
    /// physical function's.
    #[must_use]
    pub const fn device(&self) -> u16 {
        self.device
    }

    /// The page size the virtual functions' registers are aligned to.
    #[must_use]
    pub const fn page_size(&self) -> u64 {
        self.page_size
    }

    /// The base address registers describing the first virtual function.
    #[must_use]
    pub const fn bars(&self) -> &[Bar; SLOTS] {
        &self.bars
    }

    /// Whether the virtual functions exist right now.
    #[must_use]
    pub const fn on(&self) -> bool {
        self.on
    }

    /// Whether they answer memory accesses.
    #[must_use]
    pub const fn decoding(&self) -> bool {
        self.decoding
    }

    /// The routing identifier of virtual function `index`.
    ///
    /// Not an address: whether the identifier splits into a device and a
    /// function or is one flat function number depends on whether the
    /// alternative routing interpretation is in force above this function, and
    /// getting that wrong would name a different device entirely.
    ///
    /// `None` past the last virtual function, or if the arithmetic leaves the
    /// sixteen bits a routing identifier has.
    #[must_use]
    pub fn routing_id(&self, physical: u16, index: u16) -> Option<u16> {
        if index >= self.current {
            return None;
        }
        physical
            .checked_add(self.first_offset)?
            .checked_add(self.stride.checked_mul(index)?)
    }

    /// Reads the capability at `at`.
    ///
    /// # Errors
    ///
    /// Whatever the configuration mechanism reported.
    pub(crate) fn decode(config: &Config, at: Offset) -> Result<Self, PciError> {
        let control = SriovControl::from_bits(config.u16(at.plus(SRIOV_CONTROL))?);
        Ok(Self {
            at,
            initial: config.u16(at.plus(SRIOV_INITIAL))?,
            total: config.u16(at.plus(SRIOV_TOTAL))?,
            current: config.u16(at.plus(SRIOV_CURRENT))?,
            first_offset: config.u16(at.plus(SRIOV_FIRST_OFFSET))?,
            stride: config.u16(at.plus(SRIOV_STRIDE))?,
            device: config.u16(at.plus(SRIOV_DEVICE))?,
            page_size: page_size(config.u32(at.plus(SRIOV_PAGE_SIZE))?),
            bars: bar::decode(config, at.plus(SRIOV_BARS), SLOTS)?,
            on: control.on(),
            decoding: control.decoding(),
        })
    }
}

impl Display for Sriov {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "sriov {} of {} virtual functions, device {:#06x}, offset {} stride {}",
            self.current, self.total, self.device, self.first_offset, self.stride
        )?;
        if self.on {
            formatter.write_str(", enabled")?;
        }
        Ok(())
    }
}

/// The device's serial number, which is the only name it has that no other
/// device shares.
///
/// # Errors
///
/// Whatever the configuration mechanism reported.
pub(crate) fn serial_number(config: &Config, at: Offset) -> Result<u64, PciError> {
    let low = config.u32(at.plus(SERIAL_LOW))?;
    let high = config.u32(at.plus(SERIAL_HIGH))?;
    Ok((u64::from(high) << u32::BITS) | u64::from(low))
}

/// The page size a virtualization page-size register names.
///
/// One bit is set, and its position is the power of two above four kilobytes. A
/// register with none set describes no alignment at all, which is reported as
/// the smallest page rather than as the enormous one that counting the trailing
/// zeroes of zero would give.
fn page_size(field: u32) -> u64 {
    if field == 0 {
        return SMALLEST_PAGE;
    }
    SMALLEST_PAGE
        .checked_shl(field.trailing_zeros())
        .unwrap_or(SMALLEST_PAGE)
}

/// The version an extended capability's own header reports.
fn version(config: &Config, at: Offset) -> Result<u8, PciError> {
    Ok(Header::from_bits(config.u32(at)?).version())
}

#[bitfield(u32)]
#[derive(PartialEq, Eq)]
/// The header every extended capability starts with.
struct Header {
    /// Which capability this is.
    pub id: u16,
    /// Which version of it, which decides how many of its registers exist.
    #[bits(4)]
    pub version: u8,
    /// Where the next capability is, or zero at the end of the chain.
    #[bits(12)]
    pub next: u16,
}

#[bitfield(u16)]
#[derive(PartialEq, Eq)]
/// The routing capability register.
struct AriCapability {
    /// Whether the device's functions can be put in groups.
    pub grouping: bool,
    /// Whether those groups can each have their own access controls.
    pub group_access_control: bool,
    /// Reserved.
    #[bits(6)]
    __: u8,
    /// The next function of this device, or zero at the end of the chain.
    pub next: u8,
}

#[bitfield(u16)]
#[derive(PartialEq, Eq)]
/// The routing control register.
struct AriControl {
    /// Whether grouping is in use.
    pub grouping: bool,
    /// Whether per-group access controls are in use.
    pub group_access_control: bool,
    /// Reserved.
    #[bits(2)]
    __: u8,
    /// Which group this function has been put in.
    #[bits(3)]
    pub group: u8,
    /// Reserved.
    #[bits(9)]
    __: u16,
}

#[bitfield(u32)]
#[derive(PartialEq, Eq)]
/// One resizable register's capability register.
struct ResizableCapability {
    /// Reserved.
    #[bits(4)]
    __: u8,
    /// Which sizes the register can be set to, one bit per power of two.
    #[bits(28)]
    pub supported: u32,
}

#[bitfield(u32)]
#[derive(PartialEq, Eq)]
/// One resizable register's control register.
struct ResizableControl {
    /// Which base address register this entry describes.
    #[bits(3)]
    pub bar: u8,
    /// Reserved.
    #[bits(2)]
    __: u8,
    /// How many registers the capability describes. Only the first entry's
    /// copy of this field means anything.
    #[bits(3)]
    pub count: u8,
    /// The size the register decodes, as the base-two logarithm of the size in
    /// megabytes.
    #[bits(6)]
    pub size: u8,
    /// Reserved.
    #[bits(2)]
    __: u8,
    /// Which sizes the register can be set to, continued from the capability
    /// register.
    pub supported_upper: u16,
}

#[bitfield(u16)]
#[derive(PartialEq, Eq)]
/// The virtualization control register.
struct SriovControl {
    /// Whether the virtual functions exist.
    pub on: bool,
    /// Whether they are being migrated.
    pub migration: bool,
    /// Whether migration raises an interrupt.
    pub migration_interrupt: bool,
    /// Whether they answer memory accesses.
    pub decoding: bool,
    /// Whether the physical function's own routing is in the alternative
    /// interpretation, which the virtual functions' identifiers depend on.
    pub ari_capable_hierarchy: bool,
    /// Reserved.
    #[bits(11)]
    __: u16,
}

/// The controls a port can be asked to enforce, which is every field of the
/// register that is not padding.
const ENFORCEABLE: u16 = 0x7F;

/// Offset of the register latching errors that could not be corrected.
const UNCORRECTABLE_STATUS: u16 = 4;

/// Offset of the register saying which of those are not reported.
const UNCORRECTABLE_MASK: u16 = 8;

/// Offset of the register latching errors that were corrected.
const CORRECTABLE_STATUS: u16 = 16;

/// Offset of the register saying which of those are not reported.
const CORRECTABLE_MASK: u16 = 20;

/// Offset of the routing capability register.
const ARI_CAPABILITY: u16 = 4;

/// Offset of the routing control register.
const ARI_CONTROL: u16 = 6;

/// Offset of the register saying what a port can enforce.
const ACS_CAPABILITY: u16 = 4;

/// Offset of the register saying what it is enforcing.
const ACS_CONTROL: u16 = 6;

/// Bits the egress control vector's size is shifted by.
const ACS_EGRESS_SHIFT: u16 = 8;

/// Offset of the first resizable register's capability register.
const RESIZABLE_CAPABILITY: u16 = 4;

/// Offset of the first resizable register's control register.
const RESIZABLE_CONTROL: u16 = 8;

/// Bytes from one resizable entry to the next.
const RESIZABLE_STRIDE: u16 = 8;

/// The smallest size a resizable register can decode.
const ONE_MEGABYTE: u64 = 1 << 20;

/// Base address registers one function has, as the width the count is read at.
const SLOTS_AS_U8: u8 = 6;

/// Offset of the virtualization control register.
const SRIOV_CONTROL: u16 = 8;

/// Offset of the count firmware left configured.
const SRIOV_INITIAL: u16 = 12;

/// Offset of the count the function could have.
const SRIOV_TOTAL: u16 = 14;

/// Offset of the count it is set to have.
const SRIOV_CURRENT: u16 = 16;

/// Offset of the distance to the first virtual function.
const SRIOV_FIRST_OFFSET: u16 = 20;

/// Offset of the distance between virtual functions.
const SRIOV_STRIDE: u16 = 22;

/// Offset of the identifier the virtual functions report.
const SRIOV_DEVICE: u16 = 26;

/// Offset of the page size their registers are aligned to.
const SRIOV_PAGE_SIZE: u16 = 32;

/// Offset of the first of their base address registers.
const SRIOV_BARS: u16 = 36;

/// The smallest page a virtual function's registers can be aligned to.
const SMALLEST_PAGE: u64 = 4096;

/// Offset of the low half of the serial number.
const SERIAL_LOW: u16 = 4;

/// Offset of its high half.
const SERIAL_HIGH: u16 = 8;

const _: () = assert!(
    SLOTS == 6 && SLOTS_AS_U8 == 6,
    "the resizable count must be held to the number of registers a function has"
);
