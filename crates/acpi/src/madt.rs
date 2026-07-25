//! The Multiple APIC Description Table: every processor in the machine, and
//! every interrupt controller that serves them.
//!
//! A fixed head followed by a stream of variable-length structures, each
//! declaring its own type and length. That shape is why the table is walked by
//! length rather than indexed: a structure of a type this parser does not model
//! is stepped over without being understood, which is what lets one parser read
//! the tables of machines whose ACPI revisions differ.
//!
//! Only the structures an x86 hypervisor acts on are modelled. The SAPIC
//! entries belong to Itanium, the GIC entries to ARM, and the remainder
//! describe controllers that do not exist on a machine pulzar can run on; they
//! are counted and skipped rather than being silently invisible.
//!
//! Two kinds of processor entry exist and both are kept in one list. The
//! original one carries 8-bit identifiers and the x2APIC one carries 32-bit
//! identifiers, but a processor is a processor: widening the narrow fields
//! loses nothing, and the one thing that genuinely differs — whether the
//! processor has to be addressed through x2APIC — is recorded as such.

use alloc::vec::Vec;

use log::{info, warn};
use x86_64::PhysAddr;

use crate::{AcpiError, raw::Fields};

/// Offset of the 32-bit address of the local interrupt controller.
const LOCAL_APIC_ADDRESS: usize = 36;

/// Offset of the multiple APIC flags.
const FLAGS: usize = 40;

/// Offset of the first interrupt controller structure.
const FIRST_ENTRY: usize = 44;

/// Multiple APIC flag: the machine also has the pair of legacy 8259
/// controllers, whose inputs must be masked before the APICs are relied on.
const PCAT_COMPAT: u32 = 1;

/// Bytes every interrupt controller structure begins with: its type, then its
/// length. Field offsets below are given relative to the end of it, in the
/// order ACPI defines them.
const HEADER: usize = 2;

/// Processor Local APIC.
const LOCAL_APIC: u8 = 0;

/// I/O APIC.
const IO_APIC: u8 = 1;

/// Interrupt Source Override.
const SOURCE_OVERRIDE: u8 = 2;

/// Non-Maskable Interrupt Source.
const NMI_SOURCE: u8 = 3;

/// Local APIC NMI.
const LOCAL_APIC_NMI: u8 = 4;

/// Local APIC Address Override.
const LOCAL_APIC_OVERRIDE: u8 = 5;

/// Processor Local x2APIC.
const LOCAL_X2APIC: u8 = 9;

/// Local x2APIC NMI.
const LOCAL_X2APIC_NMI: u8 = 0xA;

/// Local APIC flag: firmware reports the processor as usable now.
const ENABLED: u32 = 1 << 0;

/// Local APIC flag: the processor is not usable now but may be brought online.
const ONLINE_CAPABLE: u32 = 1 << 1;

/// Mask of the polarity field of an MPS INTI flags word.
const POLARITY: u16 = 0b11;

/// Mask of the trigger mode field of an MPS INTI flags word.
const TRIGGER: u16 = 0b1100;

/// Bits the trigger mode field is shifted by.
const TRIGGER_SHIFT: u32 = 2;

/// The encoding ACPI leaves reserved in both fields of an MPS INTI flags word.
const RESERVED_ENCODING: u16 = 2;

/// Polarity encoding for a source that asserts high.
const ACTIVE_HIGH: u16 = 1;

/// Polarity encoding for a source that asserts low.
const ACTIVE_LOW: u16 = 3;

/// Trigger mode encoding for a source that asserts on the transition.
const EDGE: u16 = 1;

/// Trigger mode encoding for a source that asserts for as long as it holds.
const LEVEL: u16 = 3;

/// The machine's processors and interrupt controllers, owned rather than
/// pointed at.
#[derive(Debug)]
pub struct Madt {
    local_apic: PhysAddr,
    pic_8259: bool,
    processors: Vec<Processor>,
    io_apics: Vec<IoApic>,
    overrides: Vec<SourceOverride>,
    nmi_sources: Vec<NmiSource>,
    local_nmis: Vec<LocalNmi>,
    ignored: usize,
}

impl Madt {
    /// Physical address of the local interrupt controller, the same on every
    /// processor.
    ///
    /// This is the 32-bit address from the head of the table unless the table
    /// also carried an override, in which case it is the 64-bit one that
    /// replaces it.
    #[must_use]
    pub const fn local_apic(&self) -> PhysAddr {
        self.local_apic
    }

    /// Whether the machine also has the legacy 8259 controllers.
    #[must_use]
    pub const fn pic_8259(&self) -> bool {
        self.pic_8259
    }

    /// Every processor firmware described, in the order it described them.
    ///
    /// The order matters: ACPI defines the first entry as the boot processor,
    /// which is the one already running.
    #[must_use]
    pub fn processors(&self) -> &[Processor] {
        &self.processors
    }

    /// Every I/O APIC.
    #[must_use]
    pub fn io_apics(&self) -> &[IoApic] {
        &self.io_apics
    }

    /// Every bus interrupt that does not land on the global system interrupt of
    /// the same number.
    #[must_use]
    pub fn overrides(&self) -> &[SourceOverride] {
        &self.overrides
    }

    /// Every non-maskable interrupt wired to an I/O APIC input.
    #[must_use]
    pub fn nmi_sources(&self) -> &[NmiSource] {
        &self.nmi_sources
    }

    /// Every local APIC input wired as a non-maskable interrupt.
    #[must_use]
    pub fn local_nmis(&self) -> &[LocalNmi] {
        &self.local_nmis
    }

    /// Logs everything that was parsed.
    ///
    /// One line per processor, because which processors exist and which of them
    /// may be started is the whole reason this table is read.
    pub fn describe(&self, who: &str) {
        info!(
            "{who}: madt local apic at {:#x}, {} 8259 pics, {} processors, {} io apics",
            self.local_apic,
            if self.pic_8259 { "with" } else { "without" },
            self.processors.len(),
            self.io_apics.len(),
        );
        for processor in &self.processors {
            info!(
                "{who}: madt processor uid {}, apic id {}, {:?}, {}",
                processor.uid,
                processor.apic_id,
                processor.state,
                if processor.x2apic { "x2apic" } else { "xapic" },
            );
        }
        for io_apic in &self.io_apics {
            info!(
                "{who}: madt io apic {} at {:#x}, interrupts from {}",
                io_apic.id, io_apic.address, io_apic.first_gsi,
            );
        }
        for entry in &self.overrides {
            info!(
                "{who}: madt bus {} source {} is interrupt {}, {:?} {:?}",
                entry.bus, entry.source, entry.gsi, entry.polarity, entry.trigger,
            );
        }
        for source in &self.nmi_sources {
            info!(
                "{who}: madt nmi on interrupt {}, {:?} {:?}",
                source.gsi, source.polarity, source.trigger,
            );
        }
        for nmi in &self.local_nmis {
            info!(
                "{who}: madt local nmi on lint{} of {:?}, {:?} {:?}",
                nmi.input, nmi.target, nmi.polarity, nmi.trigger,
            );
        }
        if self.ignored > 0 {
            info!(
                "{who}: madt held {} structures of types pulzar does not model",
                self.ignored
            );
        }
    }

    /// Parses the table.
    ///
    /// # Errors
    ///
    /// [`AcpiError::Truncated`] if the head or any structure runs past the end
    /// of the table, [`AcpiError::ZeroLengthEntry`] if a structure declares a
    /// length no walk could get past, or [`AcpiError::BadAddress`] if an
    /// override names an address this processor cannot form.
    pub(crate) fn parse(table: &Fields<'_>) -> Result<Self, AcpiError> {
        let mut madt = Self {
            local_apic: crate::address(u64::from(table.u32(LOCAL_APIC_ADDRESS)?))?,
            pic_8259: table.u32(FLAGS)? & PCAT_COMPAT != 0,
            processors: Vec::new(),
            io_apics: Vec::new(),
            overrides: Vec::new(),
            nmi_sources: Vec::new(),
            local_nmis: Vec::new(),
            ignored: 0,
        };
        let mut offset = FIRST_ENTRY;
        while offset < table.size() {
            let kind = table.u8(offset)?;
            let length = usize::from(table.u8(offset + 1)?);
            if length < HEADER {
                return Err(AcpiError::ZeroLengthEntry {
                    phys: table.phys().as_u64(),
                    offset,
                });
            }
            madt.add(kind, &table.nested(offset, length)?)?;
            offset += length;
        }
        Ok(madt)
    }

    /// Takes in one interrupt controller structure.
    ///
    /// Every read is bounds checked against the structure's own declared
    /// length, so a structure that is shorter than its type requires is
    /// reported rather than read out of.
    fn add(&mut self, kind: u8, entry: &Fields<'_>) -> Result<(), AcpiError> {
        match kind {
            // ACPI processor UID, APIC ID, flags.
            LOCAL_APIC => self.processors.push(Processor {
                uid: u32::from(entry.u8(HEADER)?),
                apic_id: u32::from(entry.u8(HEADER + 1)?),
                state: ProcessorState::new(entry.u32(HEADER + 2)?),
                x2apic: false,
            }),
            // I/O APIC ID, one reserved byte, address, first global system
            // interrupt.
            IO_APIC => self.io_apics.push(IoApic {
                id: entry.u8(HEADER)?,
                address: crate::address(u64::from(entry.u32(HEADER + 2)?))?,
                first_gsi: entry.u32(HEADER + 6)?,
            }),
            // Bus, source on that bus, global system interrupt it arrives on,
            // MPS INTI flags.
            SOURCE_OVERRIDE => {
                let (polarity, trigger) = interrupt_flags(entry.u16(HEADER + 6)?);
                self.overrides.push(SourceOverride {
                    bus: entry.u8(HEADER)?,
                    source: entry.u8(HEADER + 1)?,
                    gsi: entry.u32(HEADER + 2)?,
                    polarity,
                    trigger,
                });
            }
            // MPS INTI flags, global system interrupt.
            NMI_SOURCE => {
                let (polarity, trigger) = interrupt_flags(entry.u16(HEADER)?);
                self.nmi_sources.push(NmiSource {
                    gsi: entry.u32(HEADER + 2)?,
                    polarity,
                    trigger,
                });
            }
            // ACPI processor UID, MPS INTI flags, local interrupt input.
            LOCAL_APIC_NMI => {
                let (polarity, trigger) = interrupt_flags(entry.u16(HEADER + 1)?);
                self.local_nmis.push(LocalNmi {
                    target: NmiTarget::new(u32::from(entry.u8(HEADER)?), u32::from(u8::MAX)),
                    input: entry.u8(HEADER + 3)?,
                    polarity,
                    trigger,
                });
            }
            // Two reserved bytes, then the address that replaces the 32-bit one
            // at the head of the table.
            LOCAL_APIC_OVERRIDE => self.local_apic = crate::address(entry.u64(HEADER + 2)?)?,
            // Two reserved bytes, x2APIC ID, flags, ACPI processor UID.
            LOCAL_X2APIC => self.processors.push(Processor {
                uid: entry.u32(HEADER + 10)?,
                apic_id: entry.u32(HEADER + 2)?,
                state: ProcessorState::new(entry.u32(HEADER + 6)?),
                x2apic: true,
            }),
            // MPS INTI flags, ACPI processor UID, local interrupt input.
            LOCAL_X2APIC_NMI => {
                let (polarity, trigger) = interrupt_flags(entry.u16(HEADER)?);
                self.local_nmis.push(LocalNmi {
                    target: NmiTarget::new(entry.u32(HEADER + 2)?, u32::MAX),
                    input: entry.u8(HEADER + 6)?,
                    polarity,
                    trigger,
                });
            }
            _ => self.ignored += 1,
        }
        Ok(())
    }
}

/// A processor, and the local APIC that serves it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Processor {
    uid: u32,
    apic_id: u32,
    state: ProcessorState,
    x2apic: bool,
}

impl Processor {
    /// The identifier ACPI's own description of this processor uses, which is
    /// what ties it to the rest of the firmware's tables.
    #[must_use]
    pub const fn uid(&self) -> u32 {
        self.uid
    }

    /// The local APIC identifier, which is the value an interrupt is addressed
    /// to.
    #[must_use]
    pub const fn apic_id(&self) -> u32 {
        self.apic_id
    }

    /// What firmware says may be done with this processor.
    #[must_use]
    pub const fn state(&self) -> ProcessorState {
        self.state
    }

    /// Whether firmware described this processor with an x2APIC structure, and
    /// so whether it must be addressed through x2APIC rather than through the
    /// 8-bit local APIC interface.
    #[must_use]
    pub const fn x2apic(&self) -> bool {
        self.x2apic
    }
}

/// What firmware says may be done with a processor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessorState {
    /// Present and usable.
    Enabled,
    /// Not usable as it stands, but the machine permits bringing it online.
    OnlineCapable,
    /// Not usable, and not to be started.
    Disabled,
}

impl ProcessorState {
    /// The state a local APIC structure's flags describe.
    ///
    /// Enabled outranks online-capable, because ACPI defines the second flag as
    /// saying only what may be done with a processor that is *not* already
    /// enabled.
    const fn new(flags: u32) -> Self {
        if flags & ENABLED != 0 {
            Self::Enabled
        } else if flags & ONLINE_CAPABLE != 0 {
            Self::OnlineCapable
        } else {
            Self::Disabled
        }
    }
}

/// An I/O APIC, and the global system interrupts it owns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IoApic {
    id: u8,
    address: PhysAddr,
    first_gsi: u32,
}

impl IoApic {
    /// This controller's identifier.
    #[must_use]
    pub const fn id(&self) -> u8 {
        self.id
    }

    /// Physical address of its registers.
    #[must_use]
    pub const fn address(&self) -> PhysAddr {
        self.address
    }

    /// The global system interrupt its first input carries. How many inputs it
    /// has is in the controller itself, not in this table.
    #[must_use]
    pub const fn first_gsi(&self) -> u32 {
        self.first_gsi
    }
}

/// A bus interrupt that does not arrive on the global system interrupt of the
/// same number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceOverride {
    bus: u8,
    source: u8,
    gsi: u32,
    polarity: Polarity,
    trigger: Trigger,
}

impl SourceOverride {
    /// The bus the source is on. ACPI defines the only bus that can be
    /// overridden as the ISA bus, numbered zero.
    #[must_use]
    pub const fn bus(&self) -> u8 {
        self.bus
    }

    /// The interrupt number on that bus.
    #[must_use]
    pub const fn source(&self) -> u8 {
        self.source
    }

    /// The global system interrupt it actually arrives on.
    #[must_use]
    pub const fn gsi(&self) -> u32 {
        self.gsi
    }

    /// How the source asserts.
    #[must_use]
    pub const fn polarity(&self) -> Polarity {
        self.polarity
    }

    /// When the source asserts.
    #[must_use]
    pub const fn trigger(&self) -> Trigger {
        self.trigger
    }
}

/// A non-maskable interrupt wired to an I/O APIC input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NmiSource {
    gsi: u32,
    polarity: Polarity,
    trigger: Trigger,
}

impl NmiSource {
    /// The global system interrupt it arrives on.
    #[must_use]
    pub const fn gsi(&self) -> u32 {
        self.gsi
    }

    /// How the source asserts.
    #[must_use]
    pub const fn polarity(&self) -> Polarity {
        self.polarity
    }

    /// When the source asserts.
    #[must_use]
    pub const fn trigger(&self) -> Trigger {
        self.trigger
    }
}

/// A local APIC input wired as a non-maskable interrupt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LocalNmi {
    target: NmiTarget,
    input: u8,
    polarity: Polarity,
    trigger: Trigger,
}

impl LocalNmi {
    /// Which processors this applies to.
    #[must_use]
    pub const fn target(&self) -> NmiTarget {
        self.target
    }

    /// Which of the local APIC's own interrupt inputs is wired.
    #[must_use]
    pub const fn input(&self) -> u8 {
        self.input
    }

    /// How the source asserts.
    #[must_use]
    pub const fn polarity(&self) -> Polarity {
        self.polarity
    }

    /// When the source asserts.
    #[must_use]
    pub const fn trigger(&self) -> Trigger {
        self.trigger
    }
}

/// Which processors a local interrupt structure applies to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NmiTarget {
    /// Every processor in the machine.
    All,
    /// The one processor with this ACPI processor UID.
    Processor(u32),
}

impl NmiTarget {
    /// The target `uid` names, where ACPI spells "every processor" as the
    /// all-ones value of whatever width the field it was read from has.
    const fn new(uid: u32, all: u32) -> Self {
        if uid == all {
            Self::All
        } else {
            Self::Processor(uid)
        }
    }
}

/// How a source asserts its interrupt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Polarity {
    /// Whatever the bus defines, which for the ISA bus is active high.
    BusDefault,
    /// Asserted high.
    ActiveHigh,
    /// Asserted low.
    ActiveLow,
}

/// When a source asserts its interrupt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Trigger {
    /// Whatever the bus defines, which for the ISA bus is edge triggered.
    BusDefault,
    /// On the transition.
    Edge,
    /// For as long as the level holds.
    Level,
}

/// The polarity and trigger mode an MPS INTI flags word describes.
///
/// ACPI leaves one of the four encodings of each field reserved. Firmware that
/// uses it has a defect, and this is one of the few places where refusing to
/// continue would be the worse answer: the bus default is what such firmware
/// should have written, taking it costs nothing, and declining to boot a
/// machine over one malformed interrupt description would help nobody. It is
/// logged so the defect is on the record rather than absorbed.
fn interrupt_flags(flags: u16) -> (Polarity, Trigger) {
    let polarity = flags & POLARITY;
    let trigger = (flags & TRIGGER) >> TRIGGER_SHIFT;
    if polarity == RESERVED_ENCODING || trigger == RESERVED_ENCODING {
        warn!(
            "acpi: madt interrupt flags {flags:#06x} use a reserved encoding; taking the bus default"
        );
    }
    (
        match polarity {
            ACTIVE_HIGH => Polarity::ActiveHigh,
            ACTIVE_LOW => Polarity::ActiveLow,
            _ => Polarity::BusDefault,
        },
        match trigger {
            EDGE => Trigger::Edge,
            LEVEL => Trigger::Level,
            _ => Trigger::BusDefault,
        },
    )
}
