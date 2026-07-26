//! What kind of thing a function is in a PCI Express hierarchy, and what its
//! link is doing.
//!
//! Everything in this capability answers one of two questions. What role does
//! this function play — is it an endpoint, the port at the top of a hierarchy,
//! one of the two sides of a switch, or a bridge onto something that is not PCI
//! Express at all? And what has its link negotiated?
//!
//! The first question is the one the rest of the crate cares about. A bridge
//! from PCI Express down to plain PCI answers extended configuration reads with
//! an error rather than with data, so the functions below it have no extended
//! capabilities to walk and asking anyway is a transaction that some platforms
//! escalate into a machine check. Only this capability distinguishes such a
//! bridge from a switch port, which looks identical from the header alone.
//!
//! # Registers that do not exist read as zero
//!
//! Which of this capability's registers a function implements depends on its
//! role and on the capability's own version, and an unimplemented one reads as
//! zero rather than faulting. So a decode that reads them all produces
//! plausible values — a link running at no speed and no width, a port with no
//! slot — that are indistinguishable from real answers.
//!
//! Every read here is therefore gated on the thing that makes it meaningful:
//! link registers only for a function that has a link, the second-generation
//! registers only at version two and above, and the alternative routing bits
//! only on a port that could forward them. What is not read stays absent rather
//! than becoming a zero that looks like an answer.

use core::fmt::{self, Display, Formatter};

use bitfield_struct::bitfield;

use crate::{Offset, PciError, access::Config};

/// The PCI Express capability of one function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Express {
    at: Offset,
    version: u8,
    role: Role,
    slot: bool,
    link: Option<Link>,
    ari_capable: bool,
    ari_enabled: bool,
}

impl Express {
    /// Where the capability begins.
    #[must_use]
    pub const fn at(&self) -> Offset {
        self.at
    }

    /// Which version of the capability this is. Everything from the second
    /// generation of registers onwards needs at least two.
    #[must_use]
    pub const fn version(&self) -> u8 {
        self.version
    }

    /// What the function is in the hierarchy.
    #[must_use]
    pub const fn role(&self) -> Role {
        self.role
    }

    /// Whether a physical slot hangs off this port.
    #[must_use]
    pub const fn slot(&self) -> bool {
        self.slot
    }

    /// What the link has negotiated, for a function that has one.
    #[must_use]
    pub const fn link(&self) -> Option<Link> {
        self.link
    }

    /// Whether this port can forward the alternative routing interpretation,
    /// which is what allows more than eight functions below it.
    #[must_use]
    pub const fn ari_capable(&self) -> bool {
        self.ari_capable
    }

    /// Whether it is doing so. Functions above seven are only reachable below a
    /// port where this is set.
    #[must_use]
    pub const fn ari_enabled(&self) -> bool {
        self.ari_enabled
    }

    /// Reads the capability at `at`.
    ///
    /// # Errors
    ///
    /// Whatever the configuration mechanism reported.
    pub(crate) fn decode(config: &Config, at: Offset) -> Result<Self, PciError> {
        let capabilities = Capabilities::from_bits(config.u16(at.plus(CAPABILITIES))?);
        let version = capabilities.version();
        let role = capabilities.role();

        let link = if role.has_link() {
            Some(Link::decode(config, at)?)
        } else {
            None
        };
        // The registers that carry these arrived with the second version of the
        // capability. On a version one function they are not reserved; they are
        // simply not there.
        let (ari_capable, ari_enabled) = if version >= SECOND_GENERATION && role.forwards() {
            (
                DeviceCapabilities2::from_bits(config.u32(at.plus(DEVICE_CAPABILITIES_2))?)
                    .ari_forwarding(),
                DeviceControl2::from_bits(config.u16(at.plus(DEVICE_CONTROL_2))?).ari_forwarding(),
            )
        } else {
            (false, false)
        };

        Ok(Self {
            at,
            version,
            role,
            slot: capabilities.slot_implemented() && role.takes_a_slot(),
            link,
            ari_capable,
            ari_enabled,
        })
    }
}

impl Display for Express {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "pcie v{} {}", self.version, self.role)?;
        if let Some(link) = self.link {
            write!(formatter, ", {link}")?;
        }
        if self.slot {
            formatter.write_str(", slot")?;
        }
        if self.ari_enabled {
            formatter.write_str(", ari forwarding")?;
        }
        Ok(())
    }
}

/// What a function is in a PCI Express hierarchy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// A device at the edge of the hierarchy.
    Endpoint,
    /// The same, but one that also answers as though it were on plain PCI.
    LegacyEndpoint,
    /// A port at the top of a hierarchy, belonging to the root complex.
    RootPort,
    /// The side of a switch that faces the root complex.
    UpstreamPort,
    /// A side of a switch that faces away from it.
    DownstreamPort,
    /// A bridge from PCI Express down onto plain PCI. Nothing below one of
    /// these has extended configuration space.
    ExpressToPci,
    /// A bridge from plain PCI up onto PCI Express.
    PciToExpress,
    /// An endpoint built into the root complex, which has no link of its own.
    IntegratedEndpoint,
    /// A collector for errors reported by integrated endpoints, which likewise
    /// has no link.
    EventCollector,
    /// An encoding the specification does not define.
    Reserved(u8),
}

impl Role {
    /// The role this encoding names.
    const fn from_bits(value: u8) -> Self {
        match value {
            0x0 => Self::Endpoint,
            0x1 => Self::LegacyEndpoint,
            0x4 => Self::RootPort,
            0x5 => Self::UpstreamPort,
            0x6 => Self::DownstreamPort,
            0x7 => Self::ExpressToPci,
            0x8 => Self::PciToExpress,
            0x9 => Self::IntegratedEndpoint,
            0xA => Self::EventCollector,
            other => Self::Reserved(other),
        }
    }

    /// The encoding itself.
    const fn into_bits(self) -> u8 {
        match self {
            Self::Endpoint => 0x0,
            Self::LegacyEndpoint => 0x1,
            Self::RootPort => 0x4,
            Self::UpstreamPort => 0x5,
            Self::DownstreamPort => 0x6,
            Self::ExpressToPci => 0x7,
            Self::PciToExpress => 0x8,
            Self::IntegratedEndpoint => 0x9,
            Self::EventCollector => 0xA,
            Self::Reserved(value) => value,
        }
    }

    /// Whether the function has a link of its own, and so link registers worth
    /// reading.
    #[must_use]
    pub const fn has_link(self) -> bool {
        !matches!(
            self,
            Self::IntegratedEndpoint | Self::EventCollector | Self::Reserved(_)
        )
    }

    /// Whether the function is a port with a hierarchy below it.
    #[must_use]
    pub const fn forwards(self) -> bool {
        matches!(self, Self::RootPort | Self::DownstreamPort)
    }

    /// Whether a physical slot can hang off the function, which is the only
    /// case where the slot bit means anything.
    #[must_use]
    pub const fn takes_a_slot(self) -> bool {
        self.forwards()
    }

    /// Whether everything below the function is plain PCI, and so has no
    /// extended configuration space to read.
    #[must_use]
    pub const fn hides_extended_space(self) -> bool {
        matches!(self, Self::ExpressToPci)
    }
}

impl Display for Role {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Endpoint => formatter.write_str("endpoint"),
            Self::LegacyEndpoint => formatter.write_str("legacy endpoint"),
            Self::RootPort => formatter.write_str("root port"),
            Self::UpstreamPort => formatter.write_str("upstream port"),
            Self::DownstreamPort => formatter.write_str("downstream port"),
            Self::ExpressToPci => formatter.write_str("pcie-to-pci bridge"),
            Self::PciToExpress => formatter.write_str("pci-to-pcie bridge"),
            Self::IntegratedEndpoint => formatter.write_str("integrated endpoint"),
            Self::EventCollector => formatter.write_str("event collector"),
            Self::Reserved(value) => write!(formatter, "reserved role {value:#x}"),
        }
    }
}

/// What a link is capable of and what it settled on.
///
/// A snapshot rather than a fact: link speed and width are renegotiated when
/// power management retrains a link, so this is what was true while the machine
/// was being surveyed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Link {
    max_speed: Speed,
    speed: Speed,
    max_width: u8,
    width: u8,
}

impl Link {
    /// The fastest the link can run.
    #[must_use]
    pub const fn max_speed(&self) -> Speed {
        self.max_speed
    }

    /// What it was running at.
    #[must_use]
    pub const fn speed(&self) -> Speed {
        self.speed
    }

    /// The widest the link can be.
    #[must_use]
    pub const fn max_width(&self) -> u8 {
        self.max_width
    }

    /// How wide it was.
    #[must_use]
    pub const fn width(&self) -> u8 {
        self.width
    }

    /// Whether the link came up short of what both ends can do.
    #[must_use]
    pub fn degraded(&self) -> bool {
        self.speed < self.max_speed || self.width < self.max_width
    }

    /// Reads the link registers of the capability at `at`.
    fn decode(config: &Config, at: Offset) -> Result<Self, PciError> {
        let capabilities = LinkCapabilities::from_bits(config.u32(at.plus(LINK_CAPABILITIES))?);
        let status = LinkStatus::from_bits(config.u16(at.plus(LINK_STATUS))?);
        Ok(Self {
            max_speed: capabilities.speed(),
            speed: status.speed(),
            max_width: capabilities.width(),
            width: status.width(),
        })
    }
}

impl Display for Link {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} x{}", self.speed, self.width)?;
        if self.degraded() {
            write!(formatter, " of {} x{}", self.max_speed, self.max_width)?;
        }
        Ok(())
    }
}

/// How fast one lane of a link runs.
///
/// Ordered by the encoding, which is also the order of increasing speed, so
/// that comparing two of these answers whether a link came up slow.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Speed {
    /// A speed this crate has no name for, or a link that is down.
    Unknown,
    /// 2.5 GT/s.
    Gen1,
    /// 5 GT/s.
    Gen2,
    /// 8 GT/s.
    Gen3,
    /// 16 GT/s.
    Gen4,
    /// 32 GT/s.
    Gen5,
    /// 64 GT/s.
    Gen6,
}

impl Speed {
    /// The speed this encoding names.
    const fn from_bits(value: u8) -> Self {
        match value {
            1 => Self::Gen1,
            2 => Self::Gen2,
            3 => Self::Gen3,
            4 => Self::Gen4,
            5 => Self::Gen5,
            6 => Self::Gen6,
            _ => Self::Unknown,
        }
    }

    /// The encoding itself.
    const fn into_bits(self) -> u8 {
        match self {
            Self::Unknown => 0,
            Self::Gen1 => 1,
            Self::Gen2 => 2,
            Self::Gen3 => 3,
            Self::Gen4 => 4,
            Self::Gen5 => 5,
            Self::Gen6 => 6,
        }
    }
}

impl Display for Speed {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown => formatter.write_str("unknown speed"),
            Self::Gen1 => formatter.write_str("2.5 GT/s"),
            Self::Gen2 => formatter.write_str("5 GT/s"),
            Self::Gen3 => formatter.write_str("8 GT/s"),
            Self::Gen4 => formatter.write_str("16 GT/s"),
            Self::Gen5 => formatter.write_str("32 GT/s"),
            Self::Gen6 => formatter.write_str("64 GT/s"),
        }
    }
}

#[bitfield(u16)]
#[derive(PartialEq, Eq)]
/// The register naming what a function is and which version of this capability
/// describes it.
struct Capabilities {
    /// Which version of the capability this is.
    #[bits(4)]
    pub version: u8,
    /// What the function is in the hierarchy.
    #[bits(4)]
    pub role: Role,
    /// Whether a physical slot hangs off this port. Meaningless unless the role
    /// is one that can have a slot.
    pub slot_implemented: bool,
    /// Which interrupt message number the capability's own events arrive on.
    #[bits(5)]
    pub interrupt_message: u8,
    /// Reserved.
    #[bits(2)]
    __: u8,
}

#[bitfield(u32)]
#[derive(PartialEq, Eq)]
/// What the link is capable of.
struct LinkCapabilities {
    /// The fastest the link can run.
    #[bits(4)]
    pub speed: Speed,
    /// The widest it can be, in lanes.
    #[bits(6)]
    pub width: u8,
    /// The rest describes power management, latency and reporting, none of
    /// which pulzar reads.
    #[bits(22)]
    __: u32,
}

#[bitfield(u16)]
#[derive(PartialEq, Eq)]
/// What the link settled on.
struct LinkStatus {
    /// What it is running at.
    #[bits(4)]
    pub speed: Speed,
    /// How wide it came up, in lanes.
    #[bits(6)]
    pub width: u8,
    /// Reserved.
    __: bool,
    /// Whether the link is retraining, in which case the two fields above are
    /// about to change.
    pub training: bool,
    /// Whether the slot's reference clock is shared with the port above.
    pub slot_clock: bool,
    /// Whether the link is up at all.
    pub active: bool,
    /// Whether the bandwidth changed for a reason software asked for.
    pub bandwidth_management: bool,
    /// Whether it changed for a reason software did not ask for.
    pub autonomous_bandwidth: bool,
}

#[bitfield(u32)]
#[derive(PartialEq, Eq)]
/// The second-generation device capabilities, which is where a port says
/// whether it can forward the alternative routing interpretation.
struct DeviceCapabilities2 {
    /// Completion timeout ranges and disabling, and atomic operation routing.
    #[bits(5)]
    __: u8,
    /// Whether the port can forward the alternative routing interpretation.
    pub ari_forwarding: bool,
    /// Everything else the second generation added, none of which pulzar reads.
    #[bits(26)]
    __: u32,
}

#[bitfield(u16)]
#[derive(PartialEq, Eq)]
/// The second-generation device control, which is where a port says whether it
/// is forwarding.
struct DeviceControl2 {
    /// Completion timeout value and disabling.
    #[bits(5)]
    __: u8,
    /// Whether the port is forwarding the alternative routing interpretation.
    /// Functions above seven are only reachable below a port where this is set.
    pub ari_forwarding: bool,
    /// Everything else the second generation added.
    #[bits(10)]
    __: u16,
}

/// Offset of the register naming the version and the role.
const CAPABILITIES: u16 = 2;

/// Offset of the register saying what the link can do.
const LINK_CAPABILITIES: u16 = 12;

/// Offset of the register saying what the link is doing.
const LINK_STATUS: u16 = 18;

/// Offset of the second-generation device capabilities.
const DEVICE_CAPABILITIES_2: u16 = 36;

/// Offset of the second-generation device control.
const DEVICE_CONTROL_2: u16 = 40;

/// The version from which the second-generation registers exist.
const SECOND_GENERATION: u8 = 2;
