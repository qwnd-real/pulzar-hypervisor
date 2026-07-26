//! The first sixty-four bytes, which every function has and only some of which
//! mean the same thing in each.
//!
//! Sixteen bytes are common to every layout — who made the function, what it
//! is, what it is allowed to do and what it has noticed going wrong — and the
//! rest depends on the layout byte at offset fourteen. An ordinary function
//! spends them on six base address registers and a subsystem identity; a bridge
//! spends them on the bus numbers it sits between and the three address windows
//! it forwards.
//!
//! # Every register is declared, reserved bits included
//!
//! The registers here are written as their layouts rather than as sets of named
//! constants, so that a field's position and width are stated once and the
//! shifts and masks that read it are generated from that. The reserved bits are
//! part of the declaration too, as padding: a layout that does not add up to
//! its register's width fails to compile, which is what makes the gaps in the
//! command and status registers checkable rather than merely commented.
//!
//! # Reserved layouts are recognized, not guessed at
//!
//! The layout field is seven bits and the specification defines three of the
//! 128 values. A function reporting one of the other 125 is not an endpoint
//! that can be read anyway: its registers past offset sixteen mean whatever its
//! designer decided, and decoding them as base address registers would produce
//! addresses that belong to nothing. So [`Layout::Reserved`] is a value the
//! rest of the crate carries and declines to interpret, rather than a case
//! folded into the common one.
//!
//! # Why the windows are an `Option` and not a pair of numbers
//!
//! A bridge says it forwards nothing by setting its window's base above its
//! limit — there is no enable bit. Decoded naively that produces a range which
//! runs backwards, and every later comparison against it is then quietly wrong
//! in a way that looks like an ordinary range. [`Window::new`] is where that is
//! turned into the absence it means, once, so no caller has to remember.

use core::fmt::{self, Display, Formatter};

use bitfield_struct::bitfield;

use crate::{Offset, PciError, access::Config, address::Bus};

/// The vendor that made a function, and the first register read to learn
/// whether the function is there at all.
pub(crate) const VENDOR: Offset = Offset::new(0x00);

/// Where a function's base address registers begin, whichever layout it has
/// and however many of them that layout gives it.
pub(crate) const BARS: Offset = Offset::new(0x10);

/// What a function is allowed to do on its bus.
pub(crate) const COMMAND: Offset = Offset::new(0x04);

/// An ordinary function's expansion ROM base address register.
pub(crate) const ROM: Offset = Offset::new(0x30);

/// A bridge's expansion ROM base address register.
pub(crate) const BRIDGE_ROM: Offset = Offset::new(0x38);

/// Where the capability list starts, for the two layouts that keep it here.
pub(crate) const CAPABILITIES: Offset = Offset::new(0x34);

/// Where the capability list starts on a `CardBus` bridge, which puts it
/// somewhere else because the registers it displaced were already spoken for.
pub(crate) const CARDBUS_CAPABILITIES: Offset = Offset::new(0x14);

/// Identifiers a function that is not there reads as.
///
/// A bus with nothing at an address answers a configuration read with all ones,
/// and some host bridges answer with zero instead. Neither is a vendor anyone
/// was ever assigned, so both mean absent.
pub(crate) const ABSENT: [u16; 2] = [0x0000, 0xFFFF];

/// The device identifier, unique to the vendor that assigned it.
const DEVICE: Offset = Offset::new(0x02);

/// What the function has noticed going wrong.
const STATUS: Offset = Offset::new(0x06);

/// The vendor's revision of this device, and what kind of device it is.
const REVISION: Offset = Offset::new(0x08);

/// Which layout the rest of the header has, and whether the device has more
/// than one function.
const LAYOUT: Offset = Offset::new(0x0E);

/// Who built the board this function is on, as opposed to the silicon.
const SUBSYSTEM_VENDOR: Offset = Offset::new(0x2C);

/// What the board is, as opposed to the silicon.
const SUBSYSTEM_DEVICE: Offset = Offset::new(0x2E);

/// Which interrupt line firmware wired this function's pin to, and which pin it
/// asserts.
const INTERRUPT: Offset = Offset::new(0x3C);

/// The three bus numbers a bridge sits between.
const BRIDGE_BUSES: Offset = Offset::new(0x18);

/// The low halves of a bridge's forwarded I/O window, and what the bus below it
/// has noticed going wrong.
const IO_WINDOW: Offset = Offset::new(0x1C);

/// A bridge's forwarded memory window.
const MEMORY_WINDOW: Offset = Offset::new(0x20);

/// The low halves of a bridge's forwarded prefetchable window.
const PREFETCH_WINDOW: Offset = Offset::new(0x24);

/// The high half of a bridge's forwarded prefetchable window base.
const PREFETCH_BASE_UPPER: Offset = Offset::new(0x28);

/// The high half of a bridge's forwarded prefetchable window limit.
const PREFETCH_LIMIT_UPPER: Offset = Offset::new(0x2C);

/// The high halves of a bridge's forwarded I/O window.
const IO_WINDOW_UPPER: Offset = Offset::new(0x30);

/// How a bridge treats what passes through it.
const BRIDGE_CONTROL: Offset = Offset::new(0x3E);

/// Which of the two address spaces a range belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Space {
    /// The sixty-four kilobytes of I/O ports.
    Port,
    /// The physical address space.
    Memory,
}

#[bitfield(u16)]
#[derive(PartialEq, Eq)]
/// What a function is allowed to do on its bus.
///
/// Everything set here was set by firmware or by a driver, and the two decode
/// bits are the ones a hypervisor most needs: a function that answers no
/// addresses at all has no registers to intercept, whatever its capabilities
/// say about where they would be.
pub struct Command {
    /// The function answers accesses to its I/O base address registers.
    pub io_space: bool,
    /// The function answers accesses to its memory base address registers.
    /// Its registers decode nothing while this is clear, including any table a
    /// hypervisor means to intercept.
    pub memory_space: bool,
    /// The function may start transactions of its own, which is what lets it
    /// write to memory and deliver message interrupts.
    pub bus_master: bool,
    /// The function monitors special cycles.
    pub special_cycles: bool,
    /// The function may use the memory write and invalidate command.
    pub write_invalidate: bool,
    /// The function snoops palette accesses rather than answering them.
    pub vga_snoop: bool,
    /// The function acts on parity errors rather than only noting them.
    pub parity_error_response: bool,
    /// Hardwired to zero since the register was defined.
    __: bool,
    /// The function may report system errors.
    pub serr: bool,
    /// The function may use fast back-to-back transactions.
    pub fast_back_to_back: bool,
    /// The function's legacy interrupt pin is disabled, which is how a function
    /// delivering message interrupts is left.
    pub interrupt_disable: bool,
    /// Reserved.
    #[bits(5)]
    __: u8,
}

impl Command {
    /// Whether the function answers addresses in `space`.
    #[must_use]
    pub const fn decodes(self, space: Space) -> bool {
        match space {
            Space::Port => self.io_space(),
            Space::Memory => self.memory_space(),
        }
    }

    /// The same permissions with `space` given up.
    ///
    /// What a register has to be set to before it can be written to safely, and
    /// the state a function is in before firmware configures it.
    #[must_use]
    pub const fn without(self, space: Space) -> Self {
        match space {
            Space::Port => self.with_io_space(false),
            Space::Memory => self.with_memory_space(false),
        }
    }
}

#[bitfield(u16)]
#[derive(PartialEq, Eq)]
/// What a function has noticed going wrong, and what it can do.
///
/// Most of these are cleared by writing a one to them, which is why nothing in
/// this crate ever writes the register, and why the command register beside it
/// is only ever written a word at a time.
pub struct Status {
    /// Reserved.
    #[bits(3)]
    __: u8,
    /// The function is asserting its legacy interrupt pin.
    pub interrupt: bool,
    /// The function has a capability list. The byte at offset 0x34 means
    /// nothing while this is clear.
    pub capabilities: bool,
    /// The function can run at 66 MHz.
    pub capable_66mhz: bool,
    /// Reserved.
    __: bool,
    /// The function accepts fast back-to-back transactions.
    pub fast_back_to_back: bool,
    /// The function saw a parity error while acting as bus master.
    pub master_parity_error: bool,
    /// How fast the function asserts its device-select signal.
    #[bits(2)]
    pub device_select_timing: u8,
    /// The function signalled a target abort.
    pub signalled_target_abort: bool,
    /// A target the function addressed aborted the transaction.
    pub received_target_abort: bool,
    /// Nothing answered a transaction the function started.
    pub received_master_abort: bool,
    /// The function signalled a system error.
    pub signalled_system_error: bool,
    /// The function detected a parity error.
    pub detected_parity_error: bool,
}

#[bitfield(u16)]
#[derive(PartialEq, Eq)]
/// How a bridge treats what passes through it.
pub struct BridgeControl {
    /// Parity errors on the bus below are acted on.
    pub parity_error_response: bool,
    /// System errors from the bus below are forwarded up.
    pub serr: bool,
    /// Legacy I/O ranges are not forwarded down, so a device above can keep
    /// them.
    pub isa: bool,
    /// Palette and frame buffer ranges are forwarded down regardless of the
    /// windows.
    pub vga: bool,
    /// The palette range is decoded at sixteen bits rather than ten.
    pub vga_16bit: bool,
    /// A transaction nothing below answers becomes a target abort rather than
    /// all ones.
    pub master_abort: bool,
    /// The bus below is held in reset. Every function under this bridge stops
    /// answering while it is set.
    pub secondary_reset: bool,
    /// Fast back-to-back transactions are allowed on the bus below.
    pub fast_back_to_back: bool,
    /// The upstream discard timer uses the shorter timeout.
    pub primary_discard_timer: bool,
    /// The downstream discard timer uses the shorter timeout.
    pub secondary_discard_timer: bool,
    /// A discard timer has expired.
    pub discard_timer_status: bool,
    /// An expiring discard timer raises a system error.
    pub discard_timer_serr: bool,
    /// Reserved.
    #[bits(4)]
    __: u8,
}

#[bitfield(u32)]
#[derive(PartialEq, Eq)]
/// The revision and the three bytes that say what kind of device this is.
struct Identity {
    /// The vendor's revision of this device.
    pub revision: u8,
    /// The register-level programming interface within the subclass.
    pub interface: u8,
    /// What kind of device this is, within its base class.
    pub sub: u8,
    /// What kind of device this is.
    pub base: u8,
}

#[bitfield(u8)]
#[derive(PartialEq, Eq)]
/// Which shape the rest of the header has, and whether there is more than one
/// function here.
struct LayoutByte {
    /// The layout itself.
    #[bits(7)]
    pub layout: u8,
    /// Whether the device has more than one function.
    pub multifunction: bool,
}

#[bitfield(u16)]
#[derive(PartialEq, Eq)]
/// Which interrupt line firmware wired this function to, and which pin it
/// asserts.
struct InterruptWiring {
    /// The line firmware chose.
    pub line: u8,
    /// The pin the function asserts: zero for none, one to four for A through
    /// D.
    pub pin: u8,
}

#[bitfield(u32)]
#[derive(PartialEq, Eq)]
/// The three bus numbers a bridge sits between, and its latency timer.
struct BridgeBuses {
    /// The bus the bridge's upstream side is on.
    pub primary: u8,
    /// The bus immediately below the bridge.
    pub secondary: u8,
    /// The highest bus number anywhere below the bridge.
    pub subordinate: u8,
    /// How long the bridge holds the bus below it.
    pub latency: u8,
}

#[bitfield(u32)]
#[derive(PartialEq, Eq)]
/// The low halves of a bridge's forwarded I/O window, and the status of the bus
/// below it.
struct IoWindow {
    /// How wide the base and limit registers are.
    #[bits(4)]
    pub base_width: u8,
    /// Address bits 15 to 12 of the first block forwarded.
    #[bits(4)]
    pub base: u8,
    /// How wide the limit register is, which must match the base.
    #[bits(4)]
    pub limit_width: u8,
    /// Address bits 15 to 12 of the last block forwarded.
    #[bits(4)]
    pub limit: u8,
    /// What the bus below has noticed going wrong.
    #[bits(16)]
    pub secondary_status: Status,
}

#[bitfield(u32)]
#[derive(PartialEq, Eq)]
/// One of a bridge's two memory windows.
struct MemoryWindow {
    /// How wide the base and limit registers are. Always zero for the
    /// non-prefetchable window, which cannot reach above four gigabytes.
    #[bits(4)]
    pub base_width: u8,
    /// Address bits 31 to 20 of the first megabyte forwarded.
    #[bits(12)]
    pub base: u16,
    /// How wide the limit register is, which must match the base.
    #[bits(4)]
    pub limit_width: u8,
    /// Address bits 31 to 20 of the last megabyte forwarded.
    #[bits(12)]
    pub limit: u16,
}

#[bitfield(u32)]
#[derive(PartialEq, Eq)]
/// The high halves of a bridge's forwarded I/O window.
struct IoWindowUpper {
    /// Address bits 31 to 16 of the first block forwarded.
    pub base: u16,
    /// Address bits 31 to 16 of the last block forwarded.
    pub limit: u16,
}

/// What kind of device a function is.
///
/// Three bytes, narrowing: a base class, a subclass within it, and a
/// programming interface within that. Only a handful of the combinations matter
/// to a hypervisor — chiefly which functions are bridges — so the names below
/// stop at the base class rather than reproducing a registry that changes
/// whenever a new kind of device is invented.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Class {
    base: u8,
    sub: u8,
    interface: u8,
}

impl Class {
    /// Bridges of every kind.
    pub const BRIDGE: u8 = 0x06;

    /// The subclass of a bridge onto the machine's own bus hierarchy.
    pub const HOST: u8 = 0x00;

    /// The subclass of a bridge from one PCI bus to another.
    pub const PCI_TO_PCI: u8 = 0x04;

    /// The subclass of a bridge from one PCI bus to another that also
    /// subtracts, meaning it claims whatever nothing else did.
    pub const PCI_TO_PCI_SUBTRACTIVE: u8 = 0x09;

    /// The subclass of a bridge onto a `CardBus` socket.
    pub const CARDBUS: u8 = 0x07;

    /// What kind of device this is.
    #[must_use]
    pub const fn base(self) -> u8 {
        self.base
    }

    /// What kind of device this is within its base class.
    #[must_use]
    pub const fn sub(self) -> u8 {
        self.sub
    }

    /// The register-level programming interface within the subclass.
    #[must_use]
    pub const fn interface(self) -> u8 {
        self.interface
    }

    /// Whether this function bridges one PCI bus onto another, and so may have
    /// a hierarchy below it.
    #[must_use]
    pub const fn is_pci_bridge(self) -> bool {
        self.base == Self::BRIDGE
            && (self.sub == Self::PCI_TO_PCI || self.sub == Self::PCI_TO_PCI_SUBTRACTIVE)
    }

    /// Whether this function is the root of a bus hierarchy.
    #[must_use]
    pub const fn is_host_bridge(self) -> bool {
        self.base == Self::BRIDGE && self.sub == Self::HOST
    }

    /// What the base class is called.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self.base {
            0x00 => "unclassified",
            0x01 => "mass storage",
            0x02 => "network",
            0x03 => "display",
            0x04 => "multimedia",
            0x05 => "memory",
            Self::BRIDGE => match self.sub {
                Self::HOST => "host bridge",
                Self::PCI_TO_PCI | Self::PCI_TO_PCI_SUBTRACTIVE => "pci bridge",
                Self::CARDBUS => "cardbus bridge",
                _ => "bridge",
            },
            0x07 => "communication",
            0x08 => "system peripheral",
            0x09 => "input",
            0x0A => "docking station",
            0x0B => "processor",
            0x0C => "serial bus",
            0x0D => "wireless",
            0x0E => "intelligent controller",
            0x0F => "satellite communication",
            0x10 => "encryption",
            0x11 => "signal processing",
            0x12 => "accelerator",
            0x13 => "instrumentation",
            0x40 => "coprocessor",
            _ => "device",
        }
    }
}

impl Display for Class {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{:02x}{:02x}{:02x} {}",
            self.base,
            self.sub,
            self.interface,
            self.name()
        )
    }
}

/// Which shape the rest of a function's header has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// An ordinary function: six base address registers, a subsystem identity
    /// and an expansion ROM.
    Endpoint,
    /// A bridge onto another PCI bus: two base address registers, the bus
    /// numbers it sits between and the windows it forwards.
    Bridge,
    /// A bridge onto a `CardBus` socket. Recognized so that its registers are
    /// left alone: what it keeps where an endpoint keeps its base address
    /// registers is a socket base, and no machine pulzar runs on has one.
    CardBus,
    /// A shape the specification does not define, which nothing here can read
    /// past the common sixteen bytes.
    Reserved(u8),
}

impl Layout {
    /// The shape this encoding names.
    const fn decode(value: u8) -> Self {
        match value {
            0x00 => Self::Endpoint,
            0x01 => Self::Bridge,
            0x02 => Self::CardBus,
            other => Self::Reserved(other),
        }
    }

    /// How many base address registers this layout has.
    #[must_use]
    pub const fn bars(self) -> usize {
        match self {
            Self::Endpoint => 6,
            Self::Bridge => 2,
            Self::CardBus | Self::Reserved(_) => 0,
        }
    }

    /// Where this layout keeps its capability list pointer.
    #[must_use]
    pub const fn capabilities(self) -> Option<Offset> {
        match self {
            Self::Endpoint | Self::Bridge => Some(CAPABILITIES),
            Self::CardBus => Some(CARDBUS_CAPABILITIES),
            Self::Reserved(_) => None,
        }
    }

    /// Where this layout keeps its expansion ROM base address register.
    #[must_use]
    pub const fn rom(self) -> Option<Offset> {
        match self {
            Self::Endpoint => Some(ROM),
            Self::Bridge => Some(BRIDGE_ROM),
            Self::CardBus | Self::Reserved(_) => None,
        }
    }
}

impl Display for Layout {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Endpoint => formatter.write_str("endpoint"),
            Self::Bridge => formatter.write_str("bridge"),
            Self::CardBus => formatter.write_str("cardbus"),
            Self::Reserved(value) => write!(formatter, "reserved layout {value:#04x}"),
        }
    }
}

/// One range a bridge forwards downstream.
///
/// Inclusive at both ends, because that is how the hardware describes it: a
/// limit register names the last address inside the window and not the first
/// one past it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    start: u64,
    end: u64,
}

impl Window {
    /// The window from `start` to `end`, or `None` if there is none.
    ///
    /// A bridge that forwards nothing says so by leaving its base above its
    /// limit, which is the only way it can: there is no enable bit. Turning
    /// that into an absence here is what keeps a backwards range from reaching
    /// any comparison.
    const fn new(start: u64, end: u64) -> Option<Self> {
        if start > end {
            return None;
        }
        Some(Self { start, end })
    }

    /// First address inside the window.
    #[must_use]
    pub const fn start(self) -> u64 {
        self.start
    }

    /// Last address inside the window.
    #[must_use]
    pub const fn end(self) -> u64 {
        self.end
    }

    /// Bytes the window spans.
    #[must_use]
    pub const fn bytes(self) -> u64 {
        self.end - self.start + 1
    }

    /// Whether `address` falls inside the window.
    #[must_use]
    pub const fn contains(self, address: u64) -> bool {
        address >= self.start && address <= self.end
    }
}

impl Display for Window {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:#x}..={:#x}", self.start, self.end)
    }
}

/// What a bridge sits between, and what it lets through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bridge {
    primary: Bus,
    secondary: Bus,
    subordinate: Bus,
    control: BridgeControl,
    secondary_status: Status,
    io: Option<Window>,
    memory: Option<Window>,
    prefetchable: Option<Window>,
}

impl Bridge {
    /// The bus the bridge's upstream side is on.
    ///
    /// Firmware is supposed to keep this equal to the bus the bridge is
    /// actually found on, and some firmware does not. Nothing here relies on
    /// it: the bus a function was found on is the one its address carries.
    #[must_use]
    pub const fn primary(&self) -> Bus {
        self.primary
    }

    /// The bus immediately below the bridge.
    #[must_use]
    pub const fn secondary(&self) -> Bus {
        self.secondary
    }

    /// The highest bus number anywhere below the bridge.
    #[must_use]
    pub const fn subordinate(&self) -> Bus {
        self.subordinate
    }

    /// How the bridge treats what passes through it.
    #[must_use]
    pub const fn control(&self) -> BridgeControl {
        self.control
    }

    /// What the bus below the bridge has noticed going wrong.
    #[must_use]
    pub const fn secondary_status(&self) -> Status {
        self.secondary_status
    }

    /// The I/O range forwarded downstream, if any.
    #[must_use]
    pub const fn io(&self) -> Option<Window> {
        self.io
    }

    /// The non-prefetchable memory range forwarded downstream, if any.
    #[must_use]
    pub const fn memory(&self) -> Option<Window> {
        self.memory
    }

    /// The prefetchable memory range forwarded downstream, if any.
    #[must_use]
    pub const fn prefetchable(&self) -> Option<Window> {
        self.prefetchable
    }

    /// Whether the bridge claims to route any bus at all.
    ///
    /// Firmware that has not configured a bridge leaves its bus numbers at
    /// zero, and a bridge whose subordinate is below its secondary routes a
    /// range that runs backwards. Neither describes a hierarchy to descend.
    #[must_use]
    pub const fn routes(&self) -> bool {
        self.secondary.get() != 0 && self.secondary.get() <= self.subordinate.get()
    }

    /// Whether `bus` lies in the range this bridge claims.
    #[must_use]
    pub const fn spans(&self, bus: Bus) -> bool {
        self.routes() && bus.get() >= self.secondary.get() && bus.get() <= self.subordinate.get()
    }

    /// Reads a bridge's own registers.
    ///
    /// # Errors
    ///
    /// Whatever the configuration mechanism reported.
    pub(crate) fn decode(config: &Config) -> Result<Self, PciError> {
        let buses = BridgeBuses::from_bits(config.u32(BRIDGE_BUSES)?);
        let io = IoWindow::from_bits(config.u32(IO_WINDOW)?);
        Ok(Self {
            primary: Bus::new(buses.primary()),
            secondary: Bus::new(buses.secondary()),
            subordinate: Bus::new(buses.subordinate()),
            control: BridgeControl::from_bits(config.u16(BRIDGE_CONTROL)?),
            secondary_status: io.secondary_status(),
            io: io_window(config, io)?,
            memory: memory_window(config)?,
            prefetchable: prefetchable_window(config)?,
        })
    }
}

/// Everything the common part of a header says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Common {
    pub(crate) vendor: u16,
    pub(crate) device: u16,
    pub(crate) subsystem_vendor: u16,
    pub(crate) subsystem_device: u16,
    pub(crate) revision: u8,
    pub(crate) class: Class,
    pub(crate) layout: Layout,
    pub(crate) multifunction: bool,
    pub(crate) command: Command,
    pub(crate) status: Status,
    pub(crate) interrupt_line: u8,
    pub(crate) interrupt_pin: u8,
}

impl Common {
    /// Reads the part of a header every layout shares.
    ///
    /// The subsystem identity is only where an endpoint keeps it, so it is read
    /// for that layout alone and left at zero elsewhere. Reading it anyway
    /// would take a bridge's prefetchable limit and report it as a board
    /// identifier.
    ///
    /// # Errors
    ///
    /// Whatever the configuration mechanism reported.
    pub(crate) fn decode(config: &Config) -> Result<Self, PciError> {
        let layout_byte = LayoutByte::from_bits(config.u8(LAYOUT)?);
        let layout = Layout::decode(layout_byte.layout());
        let identity = Identity::from_bits(config.u32(REVISION)?);
        let interrupt = InterruptWiring::from_bits(config.u16(INTERRUPT)?);
        let (subsystem_vendor, subsystem_device) = match layout {
            Layout::Endpoint => (config.u16(SUBSYSTEM_VENDOR)?, config.u16(SUBSYSTEM_DEVICE)?),
            _ => (0, 0),
        };
        Ok(Self {
            vendor: config.u16(VENDOR)?,
            device: config.u16(DEVICE)?,
            subsystem_vendor,
            subsystem_device,
            revision: identity.revision(),
            class: Class {
                base: identity.base(),
                sub: identity.sub(),
                interface: identity.interface(),
            },
            layout,
            multifunction: layout_byte.multifunction(),
            command: Command::from_bits(config.u16(COMMAND)?),
            status: Status::from_bits(config.u16(STATUS)?),
            interrupt_line: interrupt.line(),
            interrupt_pin: interrupt.pin(),
        })
    }
}

/// The I/O range a bridge forwards.
///
/// The two nibbles carry address bits 15 to 12, and the two beside them say
/// whether there are upper halves at all. Granularity is four kilobytes, so the
/// limit names the first byte of the last block and the rest of that block is
/// inside the window too.
fn io_window(config: &Config, window: IoWindow) -> Result<Option<Window>, PciError> {
    let mut start = u64::from(window.base()) << IO_SHIFT;
    let mut end = (u64::from(window.limit()) << IO_SHIFT) | IO_GRANULARITY;
    if window.base_width() == WIDE {
        let upper = IoWindowUpper::from_bits(config.u32(IO_WINDOW_UPPER)?);
        start |= u64::from(upper.base()) << IO_UPPER_SHIFT;
        end |= u64::from(upper.limit()) << IO_UPPER_SHIFT;
    }
    Ok(Window::new(start, end))
}

/// The non-prefetchable memory range a bridge forwards.
///
/// A megabyte of granularity, and never above four gigabytes: this window has
/// no upper halves, which is what the prefetchable one is for.
fn memory_window(config: &Config) -> Result<Option<Window>, PciError> {
    let window = MemoryWindow::from_bits(config.u32(MEMORY_WINDOW)?);
    Ok(Window::new(
        u64::from(window.base()) << MEMORY_SHIFT,
        (u64::from(window.limit()) << MEMORY_SHIFT) | MEMORY_GRANULARITY,
    ))
}

/// The prefetchable memory range a bridge forwards.
///
/// As the memory window, except that the width nibbles say whether the range
/// has upper halves — which is how a bridge forwards memory above four
/// gigabytes, and is the usual case for anything with a large aperture.
fn prefetchable_window(config: &Config) -> Result<Option<Window>, PciError> {
    let window = MemoryWindow::from_bits(config.u32(PREFETCH_WINDOW)?);
    let mut start = u64::from(window.base()) << MEMORY_SHIFT;
    let mut end = (u64::from(window.limit()) << MEMORY_SHIFT) | MEMORY_GRANULARITY;
    if window.base_width() == WIDE {
        start |= u64::from(config.u32(PREFETCH_BASE_UPPER)?) << PREFETCH_UPPER_SHIFT;
        end |= u64::from(config.u32(PREFETCH_LIMIT_UPPER)?) << PREFETCH_UPPER_SHIFT;
    }
    Ok(Window::new(start, end))
}

/// The width encoding that puts a window's top bits in registers of their own.
const WIDE: u8 = 0x1;

/// Bits an I/O window's address nibble is shifted by.
const IO_SHIFT: u8 = 12;

/// Bits an I/O window's upper half is shifted by.
const IO_UPPER_SHIFT: u8 = 16;

/// Bytes an I/O window's granularity leaves inside the last block.
const IO_GRANULARITY: u64 = 0xFFF;

/// Bits a memory window's address field is shifted by.
const MEMORY_SHIFT: u8 = 20;

/// Bytes a memory window's granularity leaves inside the last block.
const MEMORY_GRANULARITY: u64 = 0xF_FFFF;

/// Bits a prefetchable window's upper half is shifted by.
const PREFETCH_UPPER_SHIFT: u8 = 32;
