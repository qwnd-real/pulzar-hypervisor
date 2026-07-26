//! The two linked lists of optional registers, and where each one led.
//!
//! Everything a function can do beyond the sixty-four bytes of its header is
//! described by a capability: a small block of registers somewhere in
//! configuration space, reached by following a chain of pointers from a fixed
//! starting point. There are two such chains and they are not the same
//! mechanism. The older one lives in the first 256 bytes, is entered from a
//! byte in the header, and identifies each block with one byte. The one PCI
//! Express added lives above 256, is entered at a fixed address, and identifies
//! each block with two bytes and a version.
//!
//! Both are walked here into a [`Capabilities`], which holds where each block
//! pulzar understands was found and nothing else. Named fields rather than a
//! collection, so that the result is a fixed-size value a function record can
//! own outright — the intercept path that eventually reads these cannot
//! allocate, and a list of pairs would have to.
//!
//! # A chain from a device is not trusted to end
//!
//! Every pointer in either chain comes from the device. A device that is
//! broken, or that is answering all ones because it has fallen off the bus
//! mid-walk, describes a chain that loops or one that leaves configuration
//! space. Following either would hang the boot on the very machine that most
//! needs the log.
//!
//! So each walk carries a bitmap of the positions it has already been to, one
//! bit per position the chain is allowed to occupy, and stops the first time it
//! is sent somewhere twice. That is exact rather than a bounded count of steps:
//! it names the position that closed the loop, and it does not cut short a long
//! chain that is perfectly valid. Forty-eight bits cover the older chain and
//! 1024 the newer one, so neither bitmap is large enough to be worth avoiding.

use log::warn;

use crate::{Offset, PciError, access::Config, header::Layout};

/// Where each capability pulzar understands was found on one function.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Capabilities {
    power_management: Option<Offset>,
    vendor: Option<Offset>,
    msi: Option<Offset>,
    msi_x: Option<Offset>,
    express: Option<Offset>,
    advanced_error: Option<Offset>,
    routing_interpretation: Option<Offset>,
    access_control: Option<Offset>,
    resizable_bar: Option<Offset>,
    serial_number: Option<Offset>,
    virtualization: Option<Offset>,
    standard: u8,
    extended: u8,
}

impl Capabilities {
    /// The power management capability, which every function is supposed to
    /// have.
    #[must_use]
    pub const fn power_management(&self) -> Option<Offset> {
        self.power_management
    }

    /// The vendor-specific capability, whose contents mean whatever its vendor
    /// decided.
    #[must_use]
    pub const fn vendor(&self) -> Option<Offset> {
        self.vendor
    }

    /// The message signalled interrupt capability.
    #[must_use]
    pub const fn msi(&self) -> Option<Offset> {
        self.msi
    }

    /// The extended message signalled interrupt capability.
    #[must_use]
    pub const fn msi_x(&self) -> Option<Offset> {
        self.msi_x
    }

    /// The PCI Express capability, whose absence means the function sits on a
    /// bus that is not PCI Express however it looks from above.
    #[must_use]
    pub const fn express(&self) -> Option<Offset> {
        self.express
    }

    /// The advanced error reporting capability.
    #[must_use]
    pub const fn advanced_error(&self) -> Option<Offset> {
        self.advanced_error
    }

    /// The alternative routing identifier interpretation capability, which is
    /// what lets a device have more than eight functions.
    #[must_use]
    pub const fn routing_interpretation(&self) -> Option<Offset> {
        self.routing_interpretation
    }

    /// The access control services capability, which says what a port will let
    /// the devices below it do to each other.
    #[must_use]
    pub const fn access_control(&self) -> Option<Offset> {
        self.access_control
    }

    /// The resizable base address register capability.
    #[must_use]
    pub const fn resizable_bar(&self) -> Option<Offset> {
        self.resizable_bar
    }

    /// The device serial number capability.
    #[must_use]
    pub const fn serial_number(&self) -> Option<Offset> {
        self.serial_number
    }

    /// The single-root virtualization capability.
    #[must_use]
    pub const fn virtualization(&self) -> Option<Offset> {
        self.virtualization
    }

    /// How many blocks the older chain held, understood or not.
    #[must_use]
    pub const fn standard(&self) -> u8 {
        self.standard
    }

    /// How many blocks the newer chain held, understood or not.
    #[must_use]
    pub const fn extended(&self) -> u8 {
        self.extended
    }

    /// Records one block of the older chain.
    ///
    /// The first of a repeated identifier wins. A function listing one twice is
    /// malformed, and its second block is as likely to be the wrong one.
    fn note_standard(&mut self, id: u8, at: Offset) {
        self.standard = self.standard.saturating_add(1);
        let slot = match id {
            POWER_MANAGEMENT => &mut self.power_management,
            MSI => &mut self.msi,
            VENDOR => &mut self.vendor,
            EXPRESS => &mut self.express,
            MSI_X => &mut self.msi_x,
            _ => return,
        };
        slot.get_or_insert(at);
    }

    /// Records one block of the newer chain.
    fn note_extended(&mut self, id: u16, at: Offset) {
        self.extended = self.extended.saturating_add(1);
        let slot = match id {
            ADVANCED_ERROR => &mut self.advanced_error,
            SERIAL_NUMBER => &mut self.serial_number,
            ACCESS_CONTROL => &mut self.access_control,
            ROUTING_INTERPRETATION => &mut self.routing_interpretation,
            VIRTUALIZATION => &mut self.virtualization,
            RESIZABLE_BAR => &mut self.resizable_bar,
            _ => return,
        };
        slot.get_or_insert(at);
    }
}

/// Walks both chains of one function.
///
/// The older chain is only entered when the status register says there is one:
/// the byte it would start from is undefined otherwise, and a function that
/// happens to have something else there would be walked into nonsense.
///
/// The newer chain has no such flag — every function with extended
/// configuration space has one, even when it is empty — so what gates it is
/// whether that space answers at all. Its caller knows two things this does
/// not: whether the mechanism in use reaches past the first 256 bytes, and
/// whether the port above this function is a bridge down onto plain PCI, below
/// which an extended read is answered with an error rather than with data.
///
/// # Errors
///
/// Whatever the configuration mechanism reported.
pub(crate) fn walk(
    config: &Config,
    layout: Layout,
    listed: bool,
    reaches_extended: bool,
) -> Result<Capabilities, PciError> {
    let mut found = Capabilities::default();
    if let Some(start) = layout.capabilities().filter(|_| listed) {
        standard(config, start, &mut found)?;
    }
    if reaches_extended {
        extended(config, &mut found)?;
    }
    Ok(found)
}

/// Which power state the function is in.
///
/// A function in anything but the fully-on state answers configuration space
/// and nothing else, so a table reached through one of its address registers is
/// not readable while it stays there. Worth recording, rather than discovering
/// later as a read of all ones.
///
/// # Errors
///
/// Whatever the configuration mechanism reported.
pub(crate) fn power_state(config: &Config, at: Offset) -> Result<u8, PciError> {
    let control = config.u16(at.plus(POWER_CONTROL))?;
    Ok(u8::try_from(control & POWER_STATE_MASK).unwrap_or_default())
}

/// Walks the chain that lives in the first 256 bytes.
fn standard(config: &Config, start: Offset, found: &mut Capabilities) -> Result<(), PciError> {
    let mut visited = Visited::<STANDARD_WORDS>::new(FIRST_STANDARD);
    let mut pointer = config.u8(start)?;
    loop {
        // Both terminators are recognized before the pointer is masked. Masking
        // first would turn the all-ones that a departed device answers with into
        // a position that looks perfectly legal.
        if pointer == 0 || pointer == u8::MAX {
            return Ok(());
        }
        let at = Offset::new(u16::from(pointer & STANDARD_POINTER_MASK));
        if at.get() < FIRST_STANDARD {
            warn!(
                "pci: {} lists a capability at {at}, which is inside its header",
                config.address()
            );
            return Ok(());
        }
        if !visited.claim(at) {
            warn!(
                "pci: {} has a capability chain that returns to {at}",
                config.address()
            );
            return Ok(());
        }
        let id = config.u8(at)?;
        if id == u8::MAX {
            warn!(
                "pci: {} answers all ones for the capability at {at}",
                config.address()
            );
            return Ok(());
        }
        found.note_standard(id, at);
        pointer = config.u8(at.plus(1))?;
    }
}

/// Walks the chain that lives above the first 256 bytes.
fn extended(config: &Config, found: &mut Capabilities) -> Result<(), PciError> {
    let mut visited = Visited::<EXTENDED_WORDS>::new(0);
    let mut at = Offset::EXTENDED;
    loop {
        if !visited.claim(at) {
            warn!(
                "pci: {} has an extended capability chain that returns to {at}",
                config.address()
            );
            return Ok(());
        }
        let header = config.u32(at)?;
        // A function with no extended capabilities answers zero; one that has
        // fallen off the bus answers all ones.
        if header == 0 || header == u32::MAX {
            return Ok(());
        }
        let id = u16::try_from(header & EXTENDED_ID_MASK).unwrap_or(u16::MAX);
        if id == 0 || id == u16::MAX {
            return Ok(());
        }
        found.note_extended(id, at);

        let next = (header >> EXTENDED_NEXT_SHIFT) & EXTENDED_NEXT_MASK;
        if next == 0 {
            return Ok(());
        }
        let next = u16::try_from(next).unwrap_or_default();
        if !(Offset::LEGACY_BYTES..=LAST_EXTENDED).contains(&next) || !next.is_multiple_of(4) {
            warn!(
                "pci: {} has an extended capability chain pointing at {next:#05x}",
                config.address()
            );
            return Ok(());
        }
        at = Offset::new(next);
    }
}

/// Positions a capability walk has already been to.
///
/// One bit per doubleword-aligned position the chain could occupy, so being
/// sent to one twice is caught exactly rather than after some arbitrary number
/// of steps.
struct Visited<const WORDS: usize> {
    words: [u64; WORDS],
    first: u16,
}

impl<const WORDS: usize> Visited<WORDS> {
    /// A bitmap covering the positions from `first` upwards.
    const fn new(first: u16) -> Self {
        Self {
            words: [0; WORDS],
            first,
        }
    }

    /// Marks `at` as visited, reporting whether it had not been.
    ///
    /// A position outside the bitmap counts as already visited, which stops the
    /// walk. Nothing legitimate can be there: each bitmap covers exactly the
    /// positions its chain is allowed to occupy.
    fn claim(&mut self, at: Offset) -> bool {
        let Some(slot) = at
            .get()
            .checked_sub(self.first)
            .map(|from| usize::from(from / 4))
        else {
            return false;
        };
        let Some(word) = self.words.get_mut(slot / BITMAP_BITS) else {
            return false;
        };
        let mask = 1_u64 << (slot % BITMAP_BITS);
        let fresh = *word & mask == 0;
        *word |= mask;
        fresh
    }
}

/// Bits one word of a visit bitmap holds, which is the width of the `u64` the
/// bitmap is made of.
const BITMAP_BITS: usize = 64;

/// Positions the older chain can occupy: every doubleword from the end of the
/// header to the end of the space it lives in.
const STANDARD_SLOTS: usize = 48;

/// Words needed to hold one bit per position of the older chain.
const STANDARD_WORDS: usize = STANDARD_SLOTS.div_ceil(BITMAP_BITS);

/// Positions the newer chain could occupy if counted from zero, which is every
/// doubleword of configuration space. Counting from zero rather than from where
/// the chain starts keeps the arithmetic obvious and costs four words.
const EXTENDED_SLOTS: usize = 1024;

/// Words needed to hold one bit per position of the newer chain.
const EXTENDED_WORDS: usize = EXTENDED_SLOTS.div_ceil(BITMAP_BITS);

/// Where the older chain's blocks may start, which is just past the header.
const FIRST_STANDARD: u16 = 0x40;

/// Bits of an older pointer that carry a position. The low two are reserved,
/// and have been seen set.
const STANDARD_POINTER_MASK: u8 = 0xFC;

/// Last position a newer block can start at and still hold a header.
const LAST_EXTENDED: u16 = Offset::EXTENDED_BYTES - 4;

/// Bits of a newer header that carry the identifier.
const EXTENDED_ID_MASK: u32 = 0xFFFF;

/// Bits the newer chain's next pointer is shifted by.
const EXTENDED_NEXT_SHIFT: u32 = 20;

/// Bits of the newer chain's next pointer that carry a position.
const EXTENDED_NEXT_MASK: u32 = 0xFFF;

/// Offset of the power management control and status register from the start
/// of its capability.
const POWER_CONTROL: u16 = 4;

/// Bits of that register naming the state the function is in.
const POWER_STATE_MASK: u16 = 0b11;

/// Power management, which is also where a function's power state is read.
const POWER_MANAGEMENT: u8 = 0x01;

/// Message signalled interrupts.
const MSI: u8 = 0x05;

/// A block whose contents only its vendor knows.
const VENDOR: u8 = 0x09;

/// PCI Express, which says what kind of port a function is.
const EXPRESS: u8 = 0x10;

/// Extended message signalled interrupts, whose table is the thing a hypervisor
/// most wants the address of.
const MSI_X: u8 = 0x11;

/// Advanced error reporting.
const ADVANCED_ERROR: u16 = 0x0001;

/// The device's serial number, which is the only globally unique name it has.
const SERIAL_NUMBER: u16 = 0x0003;

/// Access control services.
const ACCESS_CONTROL: u16 = 0x000D;

/// Alternative routing identifier interpretation.
const ROUTING_INTERPRETATION: u16 = 0x000E;

/// Single-root input/output virtualization.
const VIRTUALIZATION: u16 = 0x0010;

/// Resizable base address registers.
const RESIZABLE_BAR: u16 = 0x0015;

const _: () = assert!(
    STANDARD_WORDS * BITMAP_BITS >= STANDARD_SLOTS,
    "the older bitmap must hold one bit per position its chain can occupy"
);
const _: () = assert!(
    EXTENDED_WORDS * BITMAP_BITS >= EXTENDED_SLOTS,
    "the newer bitmap must hold one bit per position its chain can occupy"
);
