//! The machine's PCI Express hierarchy, read once and kept.
//!
//! Everything a hypervisor eventually has to do about a device begins with
//! knowing the device is there, what it is, and where its registers answer.
//! None of that is discoverable from the processor, and only part of it is in
//! the firmware tables: those say where configuration space is mapped, and
//! configuration space says the rest. So the machine is surveyed once, during
//! bring-up, and turned into values the hypervisor owns outright.
//!
//! The immediate reason is interception. To trap a device's message interrupt
//! table, nested paging needs the table's *physical* address, which is the sum
//! of a base address register and an offset held in a capability — two things
//! that live in configuration space and nowhere else. [`msi::MsiX::table`] is
//! that address, along with the pages it occupies, which is the granularity
//! anything trapping it can work at.
//!
//! # Nothing here writes to a device
//!
//! Pulzar never calls `ExitBootServices`. The firmware environment is left
//! running, which means firmware's own drivers may still be driving the
//! console, a keyboard and a network interface while this crate runs. A survey
//! that wrote to a device — even to read a register's size back, which is the
//! one thing that needs a write — could take one of those out from under its
//! owner.
//!
//! So the survey reads, and the mapping it reads through is read-only, which
//! makes that a property of the page tables rather than a claim about the code.
//! The writing side exists — [`write_u8`] and its siblings, and [`size`] for
//! the registers that only report their extent when written — and is not called
//! from here. It is for later, once the hypervisor has intercepted the guest's
//! own `ExitBootServices` and the devices are its to configure.
//!
//! # Shape
//!
//! - [`Address`] names a function; [`Aperture`] is one range of configuration
//!   space, checked, out of the firmware table.
//! - [`header`] is the sixty-four bytes every function has, [`bar`] the
//!   registers saying where it answers, [`capability`] the two chains of
//!   optional registers, and [`msi`], [`express`] and [`extended`] the
//!   capabilities themselves.
//! - [`Function`] is everything one function said; [`Topology`] is all of them,
//!   with what sits above what.
//!
//! # What this crate does not do
//!
//! It does not configure anything, route an interrupt, or assign an address. It
//! does not know a base address register's size, because learning one needs a
//! write. And it does not find a device that appears after the survey: a
//! function that was not there is a function no mapping was kept for.
//!
//! # Allocation
//!
//! The survey is `alloc` collections, so it needs the global allocator up. It
//! runs during bring-up, where a heap too small to hold the machine's own
//! description must stop the boot. Everything after it is by value: a
//! [`Function`] owns no collection, so the eventual fault handler that reads
//! one allocates nothing.

#![no_std]

extern crate alloc;

pub mod bar;
pub mod capability;
pub mod express;
pub mod extended;
pub mod header;
pub mod msi;

mod access;
mod address;
mod scan;
mod size;

use alloc::vec::Vec;

use acpi::Acpi;
use log::{info, warn};
use paging::{AddressSpace, Mapping};
use spin::Once;
use thiserror::Error;
use x86_64::PhysAddr;

pub use crate::{
    access::Aperture,
    address::{Address, Bus, Offset, Segment},
    scan::{Function, Index, Root, Topology},
    size::size,
};

/// Everything the hypervisor keeps about the machine's devices.
#[derive(Debug)]
pub struct Pci {
    apertures: Vec<Aperture>,
    topology: Topology,
    /// Kept for their lifetime alone. Every one of these is a mapping of one
    /// function's configuration space that must outlive the survey, and
    /// releasing one would invalidate the address a [`Function`] carries.
    _mappings: Vec<Mapping>,
    buses: u32,
    unreachable: u32,
    unmapped: u32,
    leaked: u32,
}

impl Pci {
    /// Surveys the machine.
    ///
    /// Must be called before the address space stops being a value and becomes
    /// the machine's: it maps and unmaps hundreds of ranges, and doing that
    /// through the lock every processor shares would hold it for the duration.
    /// It should also come before the other processors are started, because
    /// each release of a transient mapping costs an interprocessor interrupt
    /// per processor once they are running and nothing at all before.
    ///
    /// A machine-shaped problem — no firmware table, an aperture that does not
    /// check out, a bus the mapping window will not hold — is a warning and a
    /// counter rather than a failure. What is left is the part of the machine
    /// that could be read, which is more useful than refusing to boot.
    ///
    /// # Errors
    ///
    /// [`PciError::AlreadyInstalled`] for a second call, or whatever a
    /// configuration read reported.
    pub fn install(space: &mut AddressSpace, acpi: &Acpi) -> Result<&'static Self, PciError> {
        if PCI.is_completed() {
            return Err(PciError::AlreadyInstalled);
        }
        let apertures = collect(acpi);
        let survey = scan::sweep(space, &apertures)?;
        Ok(PCI.call_once(|| Self {
            apertures,
            topology: survey.topology.unwrap_or_default(),
            _mappings: survey.mappings,
            buses: survey.buses,
            unreachable: survey.unreachable,
            unmapped: survey.unmapped,
            leaked: survey.leaked,
        }))
    }

    /// Every function the machine has.
    #[must_use]
    pub const fn topology(&self) -> &Topology {
        &self.topology
    }

    /// The ranges configuration space is mapped in.
    ///
    /// These are what a hypervisor has to protect to see a guest's own
    /// configuration accesses, which is the other half of intercepting a
    /// device.
    #[must_use]
    pub fn apertures(&self) -> &[Aperture] {
        &self.apertures
    }

    /// Whether the machine was read through the legacy ports throughout, and so
    /// whether anything above the first 256 bytes of any function was
    /// reachable.
    #[must_use]
    pub fn legacy(&self) -> bool {
        self.apertures.is_empty()
    }

    /// Logs what the machine turned out to be.
    ///
    /// A summary and not an inventory. One line per function is 160 kilobytes
    /// of serial output on a large machine, which is minutes of boot spent
    /// saying what a caller can ask [`Pci::topology`] for in an instant. What
    /// is worth a line each is the apertures, the bridges, and the functions
    /// with a table someone will want to intercept.
    pub fn describe(&self, who: &str) {
        if self.legacy() {
            info!("{who}: pci has no memory aperture; read through the legacy ports");
        }
        for aperture in &self.apertures {
            info!(
                "{who}: pci segment {} buses {}..={} at {:#x}, {:#x} bytes",
                aperture.segment(),
                aperture.first_bus(),
                aperture.last_bus(),
                aperture.base(),
                aperture.bytes(),
            );
        }

        let functions = self.topology.functions();
        let bridges = functions.iter().filter(|f| f.bridge().is_some()).count();
        info!(
            "{who}: pci swept {} buses, found {} functions in {} hierarchies, {bridges} bridges",
            self.buses,
            functions.len(),
            self.topology.roots().len(),
        );
        for root in self.topology.roots() {
            info!(
                "{who}: pci hierarchy at segment {} bus {}",
                root.segment(),
                root.bus()
            );
        }
        for function in functions.iter().filter(|f| f.bridge().is_some()) {
            describe_bridge(who, function);
        }
        for function in functions {
            if let Some(msi_x) = function.msi_x() {
                info!("{who}: {} {msi_x}", function.address());
            }
        }
        for function in functions {
            if function.aer().is_some_and(|aer| aer.reporting()) {
                info!("{who}: {} arrived with errors latched", function.address());
            }
        }
        info!(
            "{who}: pci indexed {} interceptable ranges, kept {} configuration mappings",
            self.topology.regions(),
            functions.len() - self.unmapped as usize,
        );
        if self.unreachable != 0 || self.unmapped != 0 || self.leaked != 0 {
            warn!(
                "{who}: pci left {} buses unread, {} functions unmapped, {} probe mappings leaked",
                self.unreachable, self.unmapped, self.leaked
            );
        }
    }
}

/// Logs one bridge and what it forwards.
fn describe_bridge(who: &str, function: &Function) {
    let Some(bridge) = function.bridge() else {
        return;
    };
    info!(
        "{who}: {} {} buses {}..={}",
        function.address(),
        function.class(),
        bridge.secondary(),
        bridge.subordinate(),
    );
    for (what, window) in [
        ("io", bridge.io()),
        ("mem", bridge.memory()),
        ("prefetch", bridge.prefetchable()),
    ] {
        if let Some(window) = window {
            info!("{who}: {} forwards {what} {window}", function.address());
        }
    }
}

/// The machine's devices, once they have been surveyed.
///
/// # Errors
///
/// [`PciError::NotInstalled`] before the survey has run.
pub fn topology() -> Result<&'static Topology, PciError> {
    installed().map(Pci::topology)
}

/// The function at `address`.
///
/// # Errors
///
/// [`PciError::NotInstalled`] before the survey has run.
pub fn find(address: Address) -> Result<Option<&'static Function>, PciError> {
    installed().map(|pci| pci.topology.find(address))
}

/// The function whose registers `phys` falls in.
///
/// # Errors
///
/// [`PciError::NotInstalled`] before the survey has run.
pub fn owner(phys: PhysAddr) -> Result<Option<&'static Function>, PciError> {
    installed().map(|pci| pci.topology.owner(phys))
}

/// The ranges configuration space is mapped in.
///
/// # Errors
///
/// [`PciError::NotInstalled`] before the survey has run.
pub fn apertures() -> Result<&'static [Aperture], PciError> {
    installed().map(Pci::apertures)
}

/// Reads a byte of a function's configuration space.
///
/// # Errors
///
/// [`PciError::Unmapped`] if no mapping was kept for the function, or whatever
/// the read reported.
pub fn read_u8(function: &Function, at: Offset) -> Result<u8, PciError> {
    function.config()?.u8(at)
}

/// Reads a word of a function's configuration space.
///
/// # Errors
///
/// As [`read_u8`], and [`PciError::Misaligned`] unless `at` is even.
pub fn read_u16(function: &Function, at: Offset) -> Result<u16, PciError> {
    function.config()?.u16(at)
}

/// Reads a doubleword of a function's configuration space.
///
/// # Errors
///
/// As [`read_u8`], and [`PciError::Misaligned`] unless `at` is a multiple of
/// four.
pub fn read_u32(function: &Function, at: Offset) -> Result<u32, PciError> {
    function.config()?.u32(at)
}

/// Writes a byte of a function's configuration space.
///
/// # Errors
///
/// As [`read_u8`].
///
/// # Safety
///
/// The value must be one the register accepts from software that owns the
/// device — and this hypervisor does not own any device while the firmware
/// environment it booted from is still running. Configuration space is where a
/// device's decoding, its bus mastering and its interrupt delivery live, so a
/// wrong value ranges from a device that stops answering to one that writes
/// over memory something else owns.
pub unsafe fn write_u8(function: &Function, at: Offset, value: u8) -> Result<(), PciError> {
    // SAFETY: the caller vouches for the value.
    unsafe { function.config()?.write_u8(at, value) }
}

/// Writes a word of a function's configuration space.
///
/// This is the width the command register must be written at: a doubleword
/// write to offset four carries the status register with it, and the status
/// register's error bits are cleared by writing a one — so it would silently
/// discard everything the device had latched.
///
/// # Errors
///
/// As [`read_u16`].
///
/// # Safety
///
/// As [`write_u8`].
pub unsafe fn write_u16(function: &Function, at: Offset, value: u16) -> Result<(), PciError> {
    // SAFETY: the caller vouches for the value.
    unsafe { function.config()?.write_u16(at, value) }
}

/// Writes a doubleword of a function's configuration space.
///
/// # Errors
///
/// As [`read_u32`].
///
/// # Safety
///
/// As [`write_u8`].
pub unsafe fn write_u32(function: &Function, at: Offset, value: u32) -> Result<(), PciError> {
    // SAFETY: the caller vouches for the value.
    unsafe { function.config()?.write_u32(at, value) }
}

/// The apertures firmware described, checked and put in order.
///
/// Sorted by segment and then by first bus, which is what lets the sweep rely
/// on meeting the port above a bus before the bus itself. Overlapping ranges
/// are firmware describing one bus twice; the first one wins, because mapping
/// both would alias the same registers at two addresses.
fn collect(acpi: &Acpi) -> Vec<Aperture> {
    let Some(mcfg) = acpi.mcfg() else {
        info!("pci: the machine describes no configuration aperture; using the legacy ports");
        return Vec::new();
    };
    let mut found: Vec<Aperture> = Vec::with_capacity(mcfg.spaces().len());
    for space in mcfg.spaces() {
        match Aperture::adopt(space) {
            Ok(aperture) => found.push(aperture),
            Err(error) => warn!("pci: {error}"),
        }
    }
    found.sort_unstable_by_key(|aperture| (aperture.segment(), aperture.first_bus()));

    let mut kept: Vec<Aperture> = Vec::with_capacity(found.len());
    for aperture in found {
        let overlaps = kept.last().is_some_and(|last| {
            last.segment() == aperture.segment() && aperture.first_bus() <= last.last_bus()
        });
        if overlaps {
            warn!(
                "pci: ignoring the aperture for segment {} buses {}..={}, which overlaps one already taken",
                aperture.segment(),
                aperture.first_bus(),
                aperture.last_bus(),
            );
            continue;
        }
        kept.push(aperture);
    }
    kept
}

/// The survey, once it has run.
///
/// # Errors
///
/// [`PciError::NotInstalled`] before it has.
fn installed() -> Result<&'static Pci, PciError> {
    PCI.get().ok_or(PciError::NotInstalled)
}

/// Why a device could not be read, or the machine could not be surveyed.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum PciError {
    /// The machine has already been surveyed.
    #[error("pci has already been installed")]
    AlreadyInstalled,
    /// The machine has not been surveyed yet, so there is nothing to ask about.
    #[error("pci has not been installed")]
    NotInstalled,
    /// An access does not sit on a boundary of its own width, which
    /// configuration space requires of every access.
    #[error("{address} register {offset} is not {width}-byte aligned")]
    Misaligned {
        /// The function in question.
        address: Address,
        /// Where the access was to be.
        offset: Offset,
        /// How wide it was to be.
        width: u16,
    },
    /// An access lies beyond what the mechanism carrying it can address. For
    /// the legacy ports that means anything above the first 256 bytes, which is
    /// every extended capability a function has.
    #[error("{address} register {offset} is past the {reach:#x} bytes this mechanism reaches")]
    OutOfReach {
        /// The function in question.
        address: Address,
        /// Where the access was to be.
        offset: Offset,
        /// How far the mechanism goes.
        reach: u16,
    },
    /// No mapping of a function's configuration space was kept, so it cannot be
    /// reached without making one.
    #[error("{address} has no kept mapping of its configuration space")]
    Unmapped {
        /// The function in question.
        address: Address,
    },
    /// A firmware allocation describes a range that cannot be used.
    #[error("the aperture for segment {segment} at {base:#x} is unusable: {reason}")]
    BadAperture {
        /// The segment group it claimed to serve.
        segment: Segment,
        /// Where it claimed to be.
        base: u64,
        /// What did not hold.
        reason: &'static str,
    },
    /// A base address register was asked about that the function does not have.
    #[error("{address} has no base address register {slot}")]
    NoSuchRegister {
        /// The function in question.
        address: Address,
        /// The register that was asked for.
        slot: usize,
    },
    /// A register did not hold what it was written back to, which means the
    /// device did not accept the restoration of a value this crate took away.
    #[error("{address} register {slot} did not survive being sized")]
    NotRestored {
        /// The function in question.
        address: Address,
        /// The register that was being sized.
        slot: usize,
    },
}

/// The machine's devices, surveyed once by the processor that brings the
/// hypervisor up.
static PCI: Once<Pci> = Once::new();
