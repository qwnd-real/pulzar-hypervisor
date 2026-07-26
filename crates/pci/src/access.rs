//! The two ways a processor reaches configuration space, behind one interface.
//!
//! PCI Express replaced the pair of I/O ports that used to carry configuration
//! cycles with a plain memory aperture, and firmware describes where that
//! aperture is in a table this crate does not have to parse — `acpi` already
//! did. So there are two mechanisms with the same register file behind them,
//! and nothing above this module should have to know which one a given access
//! took. [`Reach`] is the whole of that decision.
//!
//! The differences that survive the abstraction are three, and each is handled
//! by name rather than by a general mechanism, because three is all there is.
//!
//! The aperture reaches all 4096 bytes of a function's configuration space and
//! the ports reach only the first 256, so extended registers exist on one path
//! and not on the other. That is checked once, in [`Config::check`], and the
//! decoders ask [`Config::reaches_extended`] rather than asking which mechanism
//! they are on.
//!
//! An aperture access is a single aligned load or store and needs no
//! serialization: the hardware page walker and the processor's own access
//! ordering are the whole of what is required. A port access is two — a
//! selector write and then a data cycle — against one pair of ports that the
//! entire machine shares. Two processors interleaving those would read one
//! function's register through another function's selector, so the pair is
//! taken under a lock *and* with interrupts masked. The lock answers other
//! processors; masking answers this one, because an interrupt handler that
//! performed a configuration access in the middle of a transaction would
//! deadlock against a lock its own processor holds.
//!
//! And the data port is four bytes wide while the selector only names a dword,
//! so a byte or word access has to pick its lane by moving along the port.
//! [`lane`] is that, written once.
//!
//! # Why the mechanism is chosen per bus and not per machine
//!
//! A machine's MCFG may describe some of its buses and not others — several
//! allocations for one segment group, with gaps, is legal and happens. A bus in
//! such a gap is still reachable through the ports if it is in segment zero, so
//! the choice belongs to the bus rather than to the machine. It costs nothing:
//! the enumeration already visits buses one at a time.

use core::ptr;

use acpi::ConfigSpace;
use spin::Mutex;
use x86_64::{
    PhysAddr, VirtAddr,
    instructions::{interrupts, port::Port},
};

use crate::{
    Offset, PciError, Segment,
    address::{Address, BUS_BYTES, Bus},
};

/// Where one run of buses on one segment group has its configuration space
/// mapped.
///
/// One MCFG allocation, checked. Firmware may describe a segment group in
/// several of these, so an aperture is looked up by the bus it covers rather
/// than by the group it belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Aperture {
    segment: Segment,
    first: Bus,
    last: Bus,
    base: PhysAddr,
}

impl Aperture {
    /// The segment group this serves.
    #[must_use]
    pub const fn segment(&self) -> Segment {
        self.segment
    }

    /// Lowest bus number covered.
    #[must_use]
    pub const fn first_bus(&self) -> Bus {
        self.first
    }

    /// Highest bus number covered.
    #[must_use]
    pub const fn last_bus(&self) -> Bus {
        self.last
    }

    /// Physical address the aperture starts at, which is the configuration
    /// space of function zero of device zero on [`Aperture::first_bus`].
    #[must_use]
    pub const fn base(&self) -> PhysAddr {
        self.base
    }

    /// Bytes the aperture occupies.
    ///
    /// This is the range a hypervisor has to protect to see a guest's own
    /// configuration accesses, which is why it is public rather than an
    /// implementation detail of the sweep.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.buses() * BUS_BYTES
    }

    /// Buses covered.
    #[must_use]
    pub fn buses(&self) -> u64 {
        u64::from(self.last.get() - self.first.get()) + 1
    }

    /// Whether this aperture carries `bus` of `segment`.
    #[must_use]
    pub fn covers(&self, segment: Segment, bus: Bus) -> bool {
        self.segment == segment && bus >= self.first && bus <= self.last
    }

    /// Where the configuration space of `bus` begins.
    ///
    /// The bus is counted from [`Aperture::first_bus`], not from zero: the
    /// aperture's base is that bus's configuration space and not bus zero's.
    #[must_use]
    pub fn bus_base(&self, bus: Bus) -> Option<PhysAddr> {
        (bus >= self.first && bus <= self.last)
            .then(|| self.base + u64::from(bus.get() - self.first.get()) * BUS_BYTES)
    }

    /// Checks one of firmware's allocations and keeps what it describes.
    ///
    /// The `acpi` crate already dropped the allocations that describe nothing,
    /// so what is left to establish is that this one can actually be mapped:
    /// that its base sits on a bus boundary, and that the range it claims lies
    /// inside the physical address space this processor can form.
    ///
    /// # Errors
    ///
    /// [`PciError::BadAperture`] naming what did not hold. Every caller treats
    /// that as a range to leave alone rather than as a machine to refuse.
    pub(crate) fn adopt(space: &ConfigSpace) -> Result<Self, PciError> {
        let base = space.base();
        let segment = Segment::new(space.segment());
        let refuse = |reason| PciError::BadAperture {
            segment,
            base: base.as_u64(),
            reason,
        };
        if space.last_bus() < space.first_bus() {
            return Err(refuse("its bus range runs backwards"));
        }
        if !base.as_u64().is_multiple_of(BUS_BYTES) {
            return Err(refuse("its base does not sit on a bus boundary"));
        }
        let last = base
            .as_u64()
            .checked_add(space.bytes() - 1)
            .ok_or_else(|| refuse("it wraps the physical address space"))?;
        PhysAddr::try_new(last)
            .map_err(|_| refuse("it reaches past the physical address space"))?;
        Ok(Self {
            segment,
            first: Bus::new(space.first_bus()),
            last: Bus::new(space.last_bus()),
            base,
        })
    }
}

/// How one function's configuration space is reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reach {
    /// Through the memory aperture, at the address its page is mapped to.
    Mapped(VirtAddr),
    /// Through the legacy selector and data ports.
    Ports,
}

/// How the functions of one bus are reached while it is being swept.
///
/// A bus is mapped as a whole for the sweep, because a function that is absent
/// must be readable to be found absent, and there is no way to know which of
/// the 256 possible functions exist before reading them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Sweep {
    /// The bus's whole megabyte is mapped at this address.
    Mapped(VirtAddr),
    /// No aperture covers the bus, and the ports reach its first 256 bytes.
    Ports,
}

impl Sweep {
    /// How `address` is reached within this bus.
    pub(crate) fn reach(self, address: Address) -> Reach {
        match self {
            Self::Mapped(base) => Reach::Mapped(base + address.within_bus()),
            Self::Ports => Reach::Ports,
        }
    }
}

/// Access to one function's configuration space.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Config {
    address: Address,
    reach: Reach,
}

impl Config {
    /// Access to `address` through `reach`.
    ///
    /// # Safety
    ///
    /// A [`Reach::Mapped`] address must be where this function's configuration
    /// space is mapped, and must stay mapped, readable and writable for as long
    /// as this value is used. That is what makes every later read a safe
    /// method rather than an unsafe one.
    pub(crate) const unsafe fn new(address: Address, reach: Reach) -> Self {
        Self { address, reach }
    }

    /// Which function this reaches.
    pub(crate) const fn address(&self) -> Address {
        self.address
    }

    /// Whether extended configuration space is reachable, which decides
    /// whether this function has an extended capability list at all.
    pub(crate) const fn reaches_extended(&self) -> bool {
        matches!(self.reach, Reach::Mapped(_))
    }

    /// The byte at `offset`.
    ///
    /// # Errors
    ///
    /// [`PciError::OutOfReach`] if the mechanism in use does not carry that
    /// far.
    pub(crate) fn u8(&self, offset: Offset) -> Result<u8, PciError> {
        self.check(offset, 1)?;
        Ok(match self.reach {
            // SAFETY: `new`'s caller guarantees this is the live mapping of the
            // function's configuration space, and the check above puts the byte
            // inside it. Volatile because these are registers: what they hold is
            // the device's business and not the compiler's to assume about.
            Reach::Mapped(base) => unsafe {
                ptr::read_volatile((base + u64::from(offset.get())).as_ptr::<u8>())
            },
            Reach::Ports => {
                let mut data = Port::<u8>::new(lane(offset, 1));
                // SAFETY: the selector names this function's dword and the port
                // is its byte lane, both under the lock and with interrupts
                // masked, so nothing can select a different register between the
                // two accesses. Reading a configuration register has no effect
                // on the device.
                transact(self.address, offset, || unsafe { data.read() })
            }
        })
    }

    /// The little-endian word at `offset`.
    ///
    /// # Errors
    ///
    /// [`PciError::Misaligned`] unless `offset` is even, or
    /// [`PciError::OutOfReach`] as [`Config::u8`].
    pub(crate) fn u16(&self, offset: Offset) -> Result<u16, PciError> {
        self.check(offset, 2)?;
        Ok(match self.reach {
            // SAFETY: as in `u8`, and the check above establishes the two-byte
            // alignment the load needs.
            Reach::Mapped(base) => unsafe {
                ptr::read_volatile((base + u64::from(offset.get())).as_ptr::<u16>())
            },
            Reach::Ports => {
                let mut data = Port::<u16>::new(lane(offset, 2));
                // SAFETY: as in `u8`, at the word lane of the selected dword.
                transact(self.address, offset, || unsafe { data.read() })
            }
        })
    }

    /// The little-endian doubleword at `offset`.
    ///
    /// # Errors
    ///
    /// [`PciError::Misaligned`] unless `offset` is a multiple of four, or
    /// [`PciError::OutOfReach`] as [`Config::u8`].
    pub(crate) fn u32(&self, offset: Offset) -> Result<u32, PciError> {
        self.check(offset, 4)?;
        Ok(match self.reach {
            // SAFETY: as in `u8`, and the check above establishes the four-byte
            // alignment the load needs.
            Reach::Mapped(base) => unsafe {
                ptr::read_volatile((base + u64::from(offset.get())).as_ptr::<u32>())
            },
            Reach::Ports => {
                let mut data = Port::<u32>::new(lane(offset, 4));
                // SAFETY: as in `u8`. This is the whole selected dword, so no
                // lane arithmetic applies.
                transact(self.address, offset, || unsafe { data.read() })
            }
        })
    }

    /// Writes the byte at `offset`.
    ///
    /// # Errors
    ///
    /// As [`Config::u8`].
    ///
    /// # Safety
    ///
    /// The value must be one this register accepts from software that owns the
    /// device. Configuration space is where a device's decoding, its bus
    /// mastering and its interrupt delivery are set, so a wrong value ranges
    /// from a device that stops answering to one that writes over memory
    /// another device owns.
    pub(crate) unsafe fn write_u8(&self, offset: Offset, value: u8) -> Result<(), PciError> {
        self.check(offset, 1)?;
        match self.reach {
            // SAFETY: as in `u8`, and the caller vouches for the value.
            Reach::Mapped(base) => unsafe {
                ptr::write_volatile((base + u64::from(offset.get())).as_mut_ptr::<u8>(), value);
            },
            Reach::Ports => {
                let mut data = Port::<u8>::new(lane(offset, 1));
                // SAFETY: as in `u8`, and the caller vouches for the value.
                transact(self.address, offset, || unsafe { data.write(value) });
            }
        }
        Ok(())
    }

    /// Writes the word at `offset`.
    ///
    /// This is the only width the command register may be written at. A
    /// doubleword write to offset four would carry the status register with it,
    /// whose error bits are cleared by writing a one — so it would silently
    /// discard everything the device had latched there.
    ///
    /// # Errors
    ///
    /// As [`Config::u16`].
    ///
    /// # Safety
    ///
    /// As [`Config::write_u8`].
    pub(crate) unsafe fn write_u16(&self, offset: Offset, value: u16) -> Result<(), PciError> {
        self.check(offset, 2)?;
        match self.reach {
            // SAFETY: as in `u16`, and the caller vouches for the value.
            Reach::Mapped(base) => unsafe {
                ptr::write_volatile((base + u64::from(offset.get())).as_mut_ptr::<u16>(), value);
            },
            Reach::Ports => {
                let mut data = Port::<u16>::new(lane(offset, 2));
                // SAFETY: as in `u16`, and the caller vouches for the value.
                transact(self.address, offset, || unsafe { data.write(value) });
            }
        }
        Ok(())
    }

    /// Writes the doubleword at `offset`.
    ///
    /// # Errors
    ///
    /// As [`Config::u32`].
    ///
    /// # Safety
    ///
    /// As [`Config::write_u8`].
    pub(crate) unsafe fn write_u32(&self, offset: Offset, value: u32) -> Result<(), PciError> {
        self.check(offset, 4)?;
        match self.reach {
            // SAFETY: as in `u32`, and the caller vouches for the value.
            Reach::Mapped(base) => unsafe {
                ptr::write_volatile((base + u64::from(offset.get())).as_mut_ptr::<u32>(), value);
            },
            Reach::Ports => {
                let mut data = Port::<u32>::new(lane(offset, 4));
                // SAFETY: as in `u32`, and the caller vouches for the value.
                transact(self.address, offset, || unsafe { data.write(value) });
            }
        }
        Ok(())
    }

    /// Establishes that a `width`-byte access at `offset` is one this mechanism
    /// can carry.
    ///
    /// # Errors
    ///
    /// [`PciError::Misaligned`] for an access that straddles its own width, and
    /// [`PciError::OutOfReach`] for one past what the mechanism addresses.
    fn check(&self, offset: Offset, width: u16) -> Result<(), PciError> {
        if !offset.get().is_multiple_of(width) {
            return Err(PciError::Misaligned {
                address: self.address,
                offset,
                width,
            });
        }
        let reach = match self.reach {
            Reach::Mapped(_) => Offset::EXTENDED_BYTES,
            Reach::Ports => Offset::LEGACY_BYTES,
        };
        if u32::from(offset.get()) + u32::from(width) > u32::from(reach) {
            return Err(PciError::OutOfReach {
                address: self.address,
                offset,
                reach,
            });
        }
        Ok(())
    }
}

/// Selects `offset` of `address` and runs `action` against the data port.
///
/// The selector write and the data cycle are one indivisible transaction on a
/// resource the whole machine shares, which is what the lock and the mask are
/// for. Both are released as soon as the data cycle has happened.
fn transact<T>(address: Address, offset: Offset, action: impl FnOnce() -> T) -> T {
    let selector = select(address, offset);
    interrupts::without_interrupts(|| {
        let _guard = LEGACY.lock();
        let mut port = Port::<u32>::new(ADDRESS_PORT);
        // SAFETY: the selector port accepts any 32-bit value; writing it only
        // decides which register the data port then refers to, and has no effect
        // on any device. Pulzar runs at ring zero, where port access is allowed.
        unsafe { port.write(selector) };
        action()
    })
}

/// The selector naming `offset` of `address`.
///
/// Only the dword is named; which bytes of it an access touches is the data
/// port's business. Segment groups have no place in this mechanism, so an
/// address outside group zero can never get here — [`Sweep::Ports`] is only
/// ever chosen for group zero.
fn select(address: Address, offset: Offset) -> u32 {
    ENABLE
        | (u32::from(address.bus().get()) << BUS_SHIFT)
        | (u32::from(address.device()) << DEVICE_SHIFT)
        | (u32::from(address.function()) << FUNCTION_SHIFT)
        | (u32::from(offset.get()) & DWORD_SELECT)
}

/// The port a `width`-byte access to `offset` reads or writes.
///
/// The data port is a doubleword wide and the selector only names a
/// doubleword, so a narrower access picks its lane by moving along the port:
/// two bytes along for a word, up to three for a byte, and never for a
/// doubleword. The offset is aligned for its width before this is reached, so
/// the lane never straddles the end of the port.
fn lane(offset: Offset, width: u16) -> u16 {
    DATA_PORT + (offset.get() & (DWORD_BYTES - width))
}

/// The port a configuration selector is written to.
const ADDRESS_PORT: u16 = 0xCF8;

/// The port the selected register's contents appear at.
const DATA_PORT: u16 = 0xCFC;

/// Bit that makes a selector a configuration cycle rather than an ordinary I/O
/// access to the same ports.
const ENABLE: u32 = 1 << 31;

/// Bits the bus number is shifted by in a selector.
const BUS_SHIFT: u32 = 16;

/// Bits the device number is shifted by in a selector.
const DEVICE_SHIFT: u32 = 11;

/// Bits the function number is shifted by in a selector.
const FUNCTION_SHIFT: u32 = 8;

/// Bytes in the doubleword a selector names.
const DWORD_BYTES: u16 = 4;

/// The bits of an offset a selector carries: a doubleword index inside the 256
/// bytes this mechanism reaches. Which bytes of that doubleword an access
/// touches is the data port's business rather than the selector's.
const DWORD_SELECT: u32 = 0xFC;

/// Serializes the legacy selector and data ports, which the whole machine
/// shares and which no single access can claim atomically.
static LEGACY: Mutex<()> = Mutex::new(());
