//! The firmware description tables pulzar depends on, read once and kept.
//!
//! ACPI is the only thing that says how many processors a machine has, how its
//! interrupts are routed, and where PCI Express configuration space is mapped.
//! None of it can be discovered any other way, and all of it lives in memory
//! that belongs to firmware — memory a pass-through hypervisor hands on to
//! whatever boots after it. So the tables are read once, during bring-up, and
//! turned into values the hypervisor owns outright. After
//! [`Acpi::collect`] returns, nothing in this crate points at firmware's memory
//! any more.
//!
//! Nothing here depends on UEFI. The only thing needed from outside is the
//! physical address of the root pointer, which the boot protocol carries, and a
//! direct map to read through — so the tables can be collected after the
//! firmware half of the address space is gone, which is exactly when it
//! happens.
//!
//! # What is kept
//!
//! A directory of every table the root directory lists, so a table can be found
//! later without walking firmware's structures again, and full parses of the
//! ones that are needed now: the [`Madt`], for the processors and interrupt
//! controllers, the [`Mcfg`], for PCI Express configuration space, and the
//! [`Hpet`], for the counter the hypervisor keeps time with. Parsing the rest
//! when the rest is needed costs nothing that has been given up here, because
//! the directory kept their addresses.
//!
//! Only the [`Madt`] is required. A machine may legitimately have no PCI
//! Express and no event timer, so those two are parsed if present and reported
//! as absent if not — what to do without them belongs to whoever needed them.
//!
//! # How much firmware is trusted
//!
//! Structurally, none of it. Every address is checked against the direct map,
//! every length against the bytes actually present, and every checksum against
//! zero, so a table that does not add up is refused rather than read.
//!
//! Semantically, as much as possible. A table whose header does not check out
//! is dropped from the directory and the rest are kept; a reserved encoding in
//! a field is logged and treated as the default it should have been; a length
//! that does not divide evenly into entries is honoured for the entries it does
//! cover. The difference is deliberate: a corrupt structure cannot be read
//! safely, but a machine whose firmware is merely sloppy is still a machine
//! pulzar should run on, and the sloppiness belongs in the log rather than in a
//! refusal to boot.
//!
//! # Allocation
//!
//! The parsed tables are `alloc` collections, so collecting them needs the
//! global allocator to be up. It happens during bring-up, where a heap too
//! small to hold a machine's own description must stop the boot: there is no
//! guest yet, and nothing to hand control back to.

#![no_std]

extern crate alloc;

mod gas;
mod hpet;
mod madt;
mod mcfg;
mod raw;
mod rsdp;
mod sdt;

use alloc::vec::Vec;

use log::{info, warn};
use paging::DirectMap;
use thiserror::Error;
use x86_64::PhysAddr;

pub use crate::{
    gas::{GenericAddress, Space},
    hpet::Hpet,
    madt::{
        IoApic, LocalNmi, Madt, NmiSource, NmiTarget, Polarity, Processor, ProcessorState,
        SourceOverride, Trigger,
    },
    mcfg::{ConfigSpace, Mcfg},
    rsdp::Directory,
    sdt::{Signature, Table},
};
use crate::{
    raw::{Fields, Physical},
    rsdp::RootPointer,
};

/// Table lengths and entry counts are `u32` while addresses are `u64` and
/// indices are `usize`, so the three are converted constantly. That is lossless
/// exactly while `usize` is as wide as `u64`, which this crate's only target
/// guarantees — but silently would not be if that ever changed.
const _: () = assert!(
    size_of::<usize>() == size_of::<u64>(),
    "acpi assumes 64-bit pointers"
);

/// Everything the hypervisor keeps from firmware's ACPI tables.
#[derive(Debug)]
pub struct Acpi {
    revision: u8,
    directory: Directory,
    tables: Vec<Table>,
    madt: Madt,
    mcfg: Option<Mcfg>,
    hpet: Option<Hpet>,
}

impl Acpi {
    /// Reads the tables that hang off the root pointer at `rsdp`.
    ///
    /// # Errors
    ///
    /// [`AcpiError::NoRootPointer`] if `rsdp` is zero, which is how the boot
    /// protocol says firmware published none; [`AcpiError::MissingTable`] if
    /// the machine has no [`Madt`], without which its processors cannot be
    /// found; and otherwise whichever check the root pointer or a table failed.
    ///
    /// # Safety
    ///
    /// `rsdp` must be the address firmware published for its root pointer, and
    /// `map` must be the direct map of the active address space.
    pub unsafe fn collect(rsdp: u64, map: DirectMap) -> Result<Self, AcpiError> {
        // SAFETY: the caller guarantees `map` is the live direct map.
        let memory = unsafe { Physical::new(map) };
        let pointer = RootPointer::read(&memory, rsdp)?;
        let (directory, tables) = collect_directory(&memory, &pointer)?;

        let madt = lookup(&tables, Signature::MADT).ok_or(AcpiError::MissingTable {
            signature: Signature::MADT,
        })?;
        let madt = Madt::parse(&sdt::contents(&memory, madt)?)?;

        Ok(Self {
            revision: pointer.revision(),
            mcfg: optional(&memory, &tables, Signature::MCFG, Mcfg::parse)?,
            hpet: optional(&memory, &tables, Signature::HPET, Hpet::parse)?,
            directory,
            tables,
            madt,
        })
    }

    /// The ACPI revision firmware's root pointer claimed.
    #[must_use]
    pub const fn revision(&self) -> u8 {
        self.revision
    }

    /// The directory the tables were found through.
    #[must_use]
    pub const fn directory(&self) -> Directory {
        self.directory
    }

    /// Every table the directory listed and this crate could check, whether or
    /// not it was parsed.
    #[must_use]
    pub fn tables(&self) -> &[Table] {
        &self.tables
    }

    /// The first table with this signature, if there is one.
    ///
    /// The first, because a machine may legitimately list several tables under
    /// one signature — supplementary description tables being the usual case —
    /// and the ones this crate parses are never among them.
    #[must_use]
    pub fn table(&self, signature: Signature) -> Option<&Table> {
        lookup(&self.tables, signature)
    }

    /// The processors and interrupt controllers.
    #[must_use]
    pub const fn madt(&self) -> &Madt {
        &self.madt
    }

    /// PCI Express configuration space, if the machine has any.
    #[must_use]
    pub const fn mcfg(&self) -> Option<&Mcfg> {
        self.mcfg.as_ref()
    }

    /// The first high precision event timer, if the machine has one.
    ///
    /// The first, because a machine with several describes each in a table of
    /// its own and nothing pulzar does needs more than one counter.
    #[must_use]
    pub const fn hpet(&self) -> Option<&Hpet> {
        self.hpet.as_ref()
    }

    /// Logs everything that was collected.
    pub fn describe(&self, who: &str) {
        info!(
            "{who}: acpi revision {}, {} listing {} tables",
            self.revision,
            self.directory,
            self.tables.len(),
        );
        for table in &self.tables {
            info!(
                "{who}: acpi table {} at {:#x}, {:#x} bytes, revision {}",
                table.signature(),
                table.phys(),
                table.length(),
                table.revision(),
            );
        }
        self.madt.describe(who);
        match &self.mcfg {
            Some(mcfg) => mcfg.describe(who),
            None => info!("{who}: acpi has no mcfg, so no memory-mapped pci configuration space"),
        }
        match &self.hpet {
            Some(hpet) => hpet.describe(who),
            None => info!("{who}: acpi has no hpet"),
        }
    }
}

/// Why the firmware tables could not be read.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum AcpiError {
    /// Firmware published no root pointer, so the machine describes itself
    /// through no ACPI at all.
    #[error("firmware published no ACPI root pointer")]
    NoRootPointer,
    /// A firmware structure holds an address with bits set above the physical
    /// address space.
    #[error("{value:#x} is not a usable physical address")]
    BadAddress {
        /// The offending value.
        value: u64,
    },
    /// The direct map does not cover a structure firmware described, which
    /// means the memory map did not describe that range as memory.
    #[error("the direct map does not reach the {len:#x} bytes at physical {phys:#x}")]
    Unreachable {
        /// Where the structure was said to be.
        phys: u64,
        /// How much of it was wanted.
        len: usize,
    },
    /// A structure is shorter than a field a parser has to read out of it.
    #[error("the structure at {phys:#x} is {len} bytes, too short for {wanted} at offset {offset}")]
    Truncated {
        /// Where the structure is.
        phys: u64,
        /// How long it turned out to be.
        len: usize,
        /// Where the field starts.
        offset: usize,
        /// How many bytes the field needs.
        wanted: usize,
    },
    /// Nothing at the address the boot protocol carried spells the signature
    /// ACPI defines for a root pointer.
    #[error("physical {phys:#x} does not hold an ACPI root pointer")]
    NotARootPointer {
        /// The address that was checked.
        phys: u64,
    },
    /// A structure's bytes do not sum to zero, so it is corrupt.
    #[error("the {len} bytes at {phys:#x} do not sum to zero")]
    BadChecksum {
        /// Where the structure is.
        phys: u64,
        /// How much of it was summed.
        len: usize,
    },
    /// A directory does not identify itself as the kind of directory the root
    /// pointer said it was.
    #[error("the table at {phys:#x} is {found}, not the {expected} it was named as")]
    WrongSignature {
        /// Where the table is.
        phys: u64,
        /// What it should have been.
        expected: Signature,
        /// What it turned out to be.
        found: Signature,
    },
    /// Neither directory the root pointer names could be read.
    #[error("the ACPI root pointer names no readable table directory")]
    NoDirectory,
    /// A table pulzar cannot do without is absent.
    #[error("the machine has no {signature} table, which pulzar requires")]
    MissingTable {
        /// The signature that was looked for.
        signature: Signature,
    },
    /// A table holds a variable-length structure that declares a length no walk
    /// could get past.
    #[error("the table at {phys:#x} holds a zero-length structure at offset {offset}")]
    ZeroLengthEntry {
        /// Where the table is.
        phys: u64,
        /// Where in it the structure is.
        offset: usize,
    },
}

/// Reads the best directory that reads, and locates every table it lists.
///
/// The preferred directory is tried first and the next one only if it fails, so
/// a firmware that fills in a broken extended directory beside a sound legacy
/// one still boots. The failure that ends the attempt is the one reported,
/// since it is the one that describes the machine's most capable directory.
fn collect_directory(
    memory: &Physical,
    pointer: &RootPointer,
) -> Result<(Directory, Vec<Table>), AcpiError> {
    let mut refused = AcpiError::NoDirectory;
    for directory in pointer.directories() {
        match listed_tables(memory, directory) {
            Ok(tables) => return Ok((directory, tables)),
            Err(error) => {
                warn!("acpi: the {directory} is unusable: {error}");
                refused = error;
            }
        }
    }
    Err(refused)
}

/// Locates every table `directory` lists.
///
/// A table whose own header does not check out is dropped with a warning rather
/// than failing the whole directory. One unreadable table is not a reason to
/// refuse a machine, and a table pulzar actually needs going missing this way
/// is reported by its own absence.
fn listed_tables(memory: &Physical, directory: Directory) -> Result<Vec<Table>, AcpiError> {
    let header = sdt::locate(memory, directory.phys())?;
    if header.signature() != directory.signature() {
        return Err(AcpiError::WrongSignature {
            phys: directory.phys().as_u64(),
            expected: directory.signature(),
            found: header.signature(),
        });
    }
    let body = sdt::contents(memory, &header)?;
    let listed = body.size() - sdt::HEADER_BYTES;
    let stride = directory.stride();
    if !listed.is_multiple_of(stride) {
        warn!(
            "acpi: the {directory} ends {} bytes into an entry; ignoring the remainder",
            listed % stride
        );
    }

    let mut tables = Vec::new();
    for index in 0..listed / stride {
        let phys = address(directory.entry(&body, sdt::HEADER_BYTES + index * stride)?)?;
        match sdt::locate(memory, phys) {
            Ok(table) => tables.push(table),
            Err(error) => warn!("acpi: ignoring the table at {phys:#x}: {error}"),
        }
    }
    Ok(tables)
}

/// The first table with this signature.
fn lookup(tables: &[Table], signature: Signature) -> Option<&Table> {
    tables.iter().find(|table| table.signature() == signature)
}

/// Parses a table the machine may or may not have.
///
/// Absence is `Ok(None)`: every table reached this way describes hardware a
/// machine is allowed not to have. A table that is present but does not parse
/// is still an error, because that is firmware describing something incorrectly
/// rather than describing nothing.
fn optional<T>(
    memory: &Physical,
    tables: &[Table],
    signature: Signature,
    parse: impl FnOnce(&Fields<'_>) -> Result<T, AcpiError>,
) -> Result<Option<T>, AcpiError> {
    lookup(tables, signature)
        .map(|table| sdt::contents(memory, table).and_then(|body| parse(&body)))
        .transpose()
}

/// A physical address out of a firmware table.
///
/// # Errors
///
/// [`AcpiError::BadAddress`] if the value has bits set above the physical
/// address space, which no address firmware wrote should have.
fn address(value: u64) -> Result<PhysAddr, AcpiError> {
    PhysAddr::try_new(value).map_err(|_| AcpiError::BadAddress { value })
}

/// A length or count as a `usize`.
///
/// Every such conversion in the crate goes through here, so the width
/// assumption above is stated once rather than at each call site. `try_from` in
/// its place would be error handling for a state the assertion rules out.
#[expect(
    clippy::cast_possible_truncation,
    reason = "usize is 64 bits wide on this crate's only target, asserted above"
)]
const fn as_usize(value: u64) -> usize {
    value as usize
}

/// A length or index as a `u64`.
const fn as_u64(value: usize) -> u64 {
    value as u64
}
