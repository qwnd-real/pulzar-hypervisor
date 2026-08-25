//! The firmware description tables pulzar depends on, read once and kept.
//!
//! ACPI is the only thing that says how many processors a machine has, how its
//! interrupts are routed, and where PCI Express configuration space is mapped.
//! None of it can be discovered any other way, and all of it lives in memory
//! that belongs to firmware — memory a pass-through hypervisor hands on to
//! whatever boots after it. So the tables are read once, during bring-up, and
//! turned into values the hypervisor owns outright. After [`Acpi::collect`]
//! returns, nothing in this crate points at firmware's memory any more.
//!
//! # What reads the tables, and what parses them
//!
//! Finding a table is uACPI's. It reads the root pointer, chooses between the
//! two directories a root pointer can name, checks every header and every
//! checksum, keeps what passed, and maps a table on request — see [`tables`].
//! None of that is worth a second implementation, and the parts of it that are
//! easy to get subtly wrong are exactly the parts a shared implementation has
//! already got right.
//!
//! Reading what is inside a table is this crate's, and stays this crate's.
//! Every parser below turns firmware's bytes into a type the rest of the
//! hypervisor can use — a roster of processors, a set of configuration space
//! apertures, a description of a counter — which is a different job from
//! finding the bytes and answers to different requirements.
//!
//! # What is kept
//!
//! A directory of every table uACPI is holding, so a table can be named later
//! without asking again, and full parses of the ones that are needed now: the
//! [`Madt`], for the processors and interrupt controllers, the [`Mcfg`], for
//! PCI Express configuration space, and the [`Hpet`] and the timer of the
//! [`Fadt`], for the counters the hypervisor keeps time with. Parsing the rest
//! when the rest is needed costs nothing that has been given up here.
//!
//! Only the [`Madt`] is required. A machine may legitimately have no PCI
//! Express and no event timer, so those two are parsed if present and reported
//! as absent if not — what to do without them belongs to whoever needed them.
//!
//! # How much firmware is trusted
//!
//! Structurally, none of it. uACPI answers for a table's own extent and
//! checksum, and every read inside one is checked against the bytes the
//! structure actually occupies, so a field that runs past the end of its table
//! is refused rather than read.
//!
//! Semantically, as much as possible. A table that cannot be described is
//! dropped from the directory and the rest are kept; a reserved encoding in a
//! field is logged and treated as the default it should have been; a length
//! that does not divide evenly into entries is honoured for the entries it does
//! cover. The difference is deliberate: a corrupt structure cannot be read
//! safely, but a machine whose firmware is merely sloppy is still a machine
//! pulzar should run on, and the sloppiness belongs in the log rather than in a
//! refusal to boot.
//!
//! # What must already be true
//!
//! uACPI's table subsystem has to be up, which is the host's to arrange: it is
//! the host that knows where firmware published the root pointer and how to
//! reach physical memory. [`Acpi::collect`] reports uACPI's own refusal if it
//! is called before then, so the ordering is checked rather than assumed.
//!
//! # Allocation
//!
//! The parsed tables are `alloc` collections, so collecting them needs the
//! global allocator to be up. It happens during bring-up, where a heap too
//! small to hold a machine's own description must stop the boot: there is no
//! guest yet, and nothing to hand control back to.

#![no_std]

extern crate alloc;

mod fadt;
mod gas;
mod hpet;
mod madt;
mod mcfg;
mod raw;
mod sdt;
mod tables;

use alloc::vec::Vec;

use log::{info, warn};
use thiserror::Error;
use uacpi_sys::Status;
use x86_64::PhysAddr;

pub use crate::{
    fadt::{Fadt, PmTimer},
    gas::{GenericAddress, Space},
    hpet::Hpet,
    madt::{
        IoApic, LocalNmi, Madt, NmiSource, NmiTarget, Polarity, Processor, ProcessorState,
        SourceOverride, Trigger,
    },
    mcfg::{ConfigSpace, Mcfg},
    sdt::{Signature, Table},
};
use crate::{raw::Fields, tables::Held};

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
    tables: Vec<Table>,
    madt: Madt,
    mcfg: Option<Mcfg>,
    hpet: Option<Hpet>,
    fadt: Option<Fadt>,
}

impl Acpi {
    /// Reads the tables uACPI is holding.
    ///
    /// # Errors
    ///
    /// [`AcpiError::MissingTable`] if the machine has no [`Madt`], without
    /// which its processors cannot be found; [`AcpiError::Uacpi`] if uACPI
    /// refused a lookup, which before its table subsystem is up is what
    /// every lookup does; and otherwise whichever check a table's contents
    /// failed.
    pub fn collect() -> Result<Self, AcpiError> {
        let tables = directory();
        let madt = required(Signature::MADT, Madt::parse)?;
        Ok(Self {
            mcfg: optional(Signature::MCFG, Mcfg::parse)?,
            hpet: optional(Signature::HPET, Hpet::parse)?,
            fadt: optional(Signature::FADT, Fadt::parse)?,
            tables,
            madt,
        })
    }

    /// Every table uACPI is holding that this crate could describe, whether or
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
        self.tables
            .iter()
            .find(|table| table.signature() == signature)
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

    /// What was kept of the fixed hardware description, if the machine has one.
    ///
    /// Every machine does in practice — the FADT is how ACPI describes the
    /// platform's own registers — but nothing pulzar needs from it is worth
    /// refusing a machine over, so its absence is reported rather than fatal.
    #[must_use]
    pub const fn fadt(&self) -> Option<&Fadt> {
        self.fadt.as_ref()
    }

    /// Logs everything that was collected.
    pub fn describe(&self, who: &str) {
        info!("{who}: acpi holds {} tables", self.tables.len());
        for table in &self.tables {
            info!(
                "{who}: acpi table {} at index {}, {:#x} bytes, revision {}",
                table.signature(),
                table.index(),
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
        match &self.fadt {
            Some(fadt) => fadt.describe(who),
            None => info!("{who}: acpi has no fadt, so no fixed hardware description"),
        }
    }
}

/// Why the firmware tables could not be read.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum AcpiError {
    /// uACPI refused a lookup. Before its table subsystem is up that is every
    /// lookup, and afterwards it means firmware's directory itself is unusable.
    #[error("uACPI refused the request: {status}")]
    Uacpi {
        /// What uACPI reported.
        status: Status,
    },
    /// A firmware structure holds an address with bits set above the physical
    /// address space.
    #[error("{value:#x} is not a usable physical address")]
    BadAddress {
        /// The offending value.
        value: u64,
    },
    /// A structure is shorter than a field a parser has to read out of it.
    #[error("the structure at {at:#x} is {len} bytes, too short for {wanted} at offset {offset}")]
    Truncated {
        /// Where the structure was read.
        at: u64,
        /// How long it turned out to be.
        len: usize,
        /// Where the field starts.
        offset: usize,
        /// How many bytes the field needs.
        wanted: usize,
    },
    /// A table pulzar cannot do without is absent.
    #[error("the machine has no {signature} table, which pulzar requires")]
    MissingTable {
        /// The signature that was looked for.
        signature: Signature,
    },
    /// A table holds a variable-length structure that declares a length no walk
    /// could get past.
    #[error("the table at {at:#x} holds a zero-length structure at offset {offset}")]
    ZeroLengthEntry {
        /// Where the table was read.
        at: u64,
        /// Where in it the structure is.
        offset: usize,
    },
}

/// Describes every table uACPI is holding.
///
/// A table that cannot be described is dropped with a warning rather than
/// failing the whole directory: one unreadable table is not a reason to refuse
/// a machine, and a table pulzar actually needs going missing this way is
/// reported by its own absence. uACPI has already refused anything whose header
/// or checksum did not hold, so what is dropped here is a table it accepted and
/// this crate could not read a header out of — which should be nothing.
fn directory() -> Vec<Table> {
    let count = tables::count();
    let mut tables = Vec::with_capacity(count);
    for index in 0..count {
        match Held::at(index).and_then(|held| held.map(|held| held.describe()).transpose()) {
            Ok(Some(table)) => tables.push(table),
            // uACPI holds fewer tables than it did a moment ago, which is
            // possible while nothing holds a reference to them.
            Ok(None) => {}
            Err(error) => warn!("acpi: ignoring the table at index {index}: {error}"),
        }
    }
    tables
}

/// Parses a table the machine must have.
///
/// # Errors
///
/// [`AcpiError::MissingTable`] if it is absent, or whatever the lookup or the
/// parse reported.
fn required<T>(
    signature: Signature,
    parse: impl FnOnce(&Fields<'_>) -> Result<T, AcpiError>,
) -> Result<T, AcpiError> {
    optional(signature, parse)?.ok_or(AcpiError::MissingTable { signature })
}

/// Parses a table the machine may or may not have.
///
/// Absence is `Ok(None)`: every table reached this way describes hardware a
/// machine is allowed not to have. A table that is present but does not parse
/// is still an error, because that is firmware describing something incorrectly
/// rather than describing nothing.
fn optional<T>(
    signature: Signature,
    parse: impl FnOnce(&Fields<'_>) -> Result<T, AcpiError>,
) -> Result<Option<T>, AcpiError> {
    let Some(held) = Held::find(signature)? else {
        return Ok(None);
    };
    // Parsed while the table is still held, and the result is this crate's own
    // from then on: dropping the reference is what lets uACPI take the mapping
    // down, so nothing that borrows the table may outlive this.
    parse(&held.fields()?).map(Some)
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
