//! One guest, across every processor it runs on.
//!
//! A virtual processor is one processor's view of a guest; this is the guest.
//! What lives here is what every one of its processors shares and none of them
//! owns: the description of its memory, and the tag its cached translations are
//! kept apart by. What does not live here is anything about running — that is
//! [`vcpu`], and the split is deliberate, because entering a guest is a
//! per-processor act and deciding what a guest *is* is not.
//!
//! Pulzar runs one guest. The type is still a partition rather than a global,
//! because the interesting properties are all about which guest a thing belongs
//! to, and a design that answers that with "the only one" has to be taken apart
//! before it can answer anything else.
//!
//! # There is no roster
//!
//! A partition does not keep a list of its virtual processors. Each processor
//! creates its own and owns it, and nothing yet enumerates them — a table
//! nothing reads is a table that can quietly disagree with reality.
//!
//! # Two locks, in one order
//!
//! Resolving a nested page fault needs the nested tables and the chunk's frame
//! allocator, which live behind different locks. [`Partition::resolve`] takes
//! the address space's lock first and the tables' second, and nothing anywhere
//! takes them the other way round. That is the whole of the ordering, and it is
//! stated here because it is the only place in the crate where two are held at
//! once.

#![no_std]

mod asid;

use log::info;
use memory::{Addressing, Linear, Physical};
use npt::{Exposure, Npt, NptError, Resolution};
use paging::{AddressSpace, PagingError};
use spin::{Mutex, Once};
use svm::{SaveArea, exit::NestedPageFault};
use thiserror::Error;
use vcpu::{Guest, Host, Vcpu, VcpuError};
use x86_64::PhysAddr;

pub use crate::asid::{Asid, Asids};

/// One guest: its memory, and the tag its translations carry.
#[derive(Debug)]
pub struct Partition {
    npt: Mutex<Npt>,
    nested_cr3: PhysAddr,
    asid: Asid,
}

impl Partition {
    /// Builds the one guest this hypervisor runs.
    ///
    /// Its memory is described lazily, so nothing is mapped here: the tables
    /// start empty and the guest's first access to any address is what
    /// describes the region containing it.
    ///
    /// # Errors
    ///
    /// [`PartitionError::NoSvm`] on a processor with no virtualization
    /// extension, [`PartitionError::OutOfAsids`] on one that can tag no guest,
    /// or [`PartitionError::Npt`] if the chunk cannot back the tables.
    pub fn establish(space: &mut AddressSpace) -> Result<Self, PartitionError> {
        let asid = asids()?.take()?;
        let window = space.direct_map();
        let npt = Npt::create(space.frames(), window)?;
        Ok(Self {
            nested_cr3: npt.root(),
            npt: Mutex::new(npt),
            asid,
        })
    }

    /// Builds the calling processor's virtual processor for this guest.
    ///
    /// The control block it produces names this guest's memory and carries this
    /// guest's tag, and holds no guest state at all — no instruction pointer,
    /// no stack pointer, no segments. It describes where a guest would run,
    /// not a guest.
    ///
    /// # Errors
    ///
    /// [`PartitionError::Vcpu`] if the chunk cannot spare a control block or
    /// the window cannot reach it.
    pub fn attach(
        &self,
        host: &'static Host,
        space: &mut AddressSpace,
    ) -> Result<Vcpu, PartitionError> {
        let window = space.direct_map();
        Ok(Vcpu::create(
            host,
            space.frames(),
            window,
            Guest {
                nested_cr3: self.nested_cr3,
                asid: self.asid.number(),
            },
        )?)
    }

    /// Describes the region containing a guest physical address the guest could
    /// not reach.
    ///
    /// What an exit handler calls on a nested page fault: `gpa` is the second
    /// exit-information field and `cause` the first, decoded.
    ///
    /// # Errors
    ///
    /// [`PartitionError::Paging`] if the address space is not yet the
    /// machine's, or [`PartitionError::Npt`] if the region cannot be
    /// described.
    pub fn resolve(
        &self,
        gpa: PhysAddr,
        cause: NestedPageFault,
    ) -> Result<Resolution, PartitionError> {
        // The address space first and the tables second, everywhere, so that two
        // processors faulting at once cannot each hold what the other wants.
        Ok(paging::with(|space| {
            self.npt.lock().fault(space.frames(), gpa, cause)
        })??)
    }

    /// Makes immutable hypervisor-owned entry code or data visible to the
    /// guest.
    ///
    /// This may be called only while the guest has not run, for the same cache
    /// coherency reason as [`Npt::expose`]. The partition owns both locks
    /// required to update the nested tables and allocate any tables needed.
    ///
    /// # Errors
    ///
    /// [`PartitionError::Paging`] if the machine address space is unavailable,
    /// or [`PartitionError::Npt`] if the requested range cannot be exposed.
    pub fn expose(
        &self,
        space: &mut AddressSpace,
        gpa: PhysAddr,
        bytes: u64,
        exposure: Exposure,
    ) -> Result<(), PartitionError> {
        Ok(self
            .npt
            .lock()
            .expose(space.frames(), gpa, bytes, exposure)?)
    }

    /// Takes back what [`Partition::expose`] made visible, leaving the range
    /// reading as zeroes like the rest of the hypervisor's memory.
    ///
    /// Unlike exposing, this may be called while the guest is running — it is
    /// how entry code is retired once the guest is past it. It takes permission
    /// away, so every processor that has run this guest must discard what it
    /// cached from these tables before entering it again; with one processor
    /// running the guest, that is one flush on its next entry.
    ///
    /// # Errors
    ///
    /// [`PartitionError::Paging`] if the address space is not yet the
    /// machine's, or [`PartitionError::Npt`] if the range cannot be concealed.
    pub fn conceal(&self, gpa: PhysAddr, bytes: u64) -> Result<(), PartitionError> {
        // The address space first and the tables second, as everywhere else
        // here.
        Ok(paging::with(|space| {
            self.npt.lock().conceal(space.frames(), gpa, bytes)
        })??)
    }

    /// Borrows this guest's memory translated by one virtual processor's
    /// current save area.
    ///
    /// The nested tables remain locked for the closure, so every translation
    /// and read observes one coherent table state. The higher-ranked closure
    /// prevents the borrowed memory view from escaping that lock.
    pub fn with_memory<T>(
        &self,
        save: &SaveArea,
        use_memory: impl for<'a> FnOnce(Linear<'a>) -> T,
    ) -> T {
        let npt = self.npt.lock();
        let physical = Physical::new(&npt, npt.window());
        use_memory(Linear::new(physical, Addressing::from_save(save)))
    }

    /// The value a control block names this guest's memory by.
    #[must_use]
    pub const fn nested_cr3(&self) -> PhysAddr {
        self.nested_cr3
    }

    /// The tag this guest's cached translations carry.
    #[must_use]
    pub const fn asid(&self) -> Asid {
        self.asid
    }

    /// Logs what the guest is, which is its memory and its tag and nothing
    /// else.
    pub fn describe(&self, who: &str) {
        info!("{who}: partition tagged asid {}", self.asid.number());
        self.npt.lock().describe(who);
    }
}

/// The machine's supply of address space identifiers.
///
/// One supply for the machine rather than one per guest, because identifiers
/// are what keeps two guests' cached translations apart: an allocator per guest
/// would hand every guest the same first identifier and defeat the whole point.
/// Sized on first use from what the processor reports, which is fixed at reset
/// and the same on every processor of a package.
///
/// # Errors
///
/// [`PartitionError::NoSvm`] on a processor with no virtualization extension,
/// which is one that tags nothing because it runs no guest.
pub fn asids() -> Result<&'static Asids, PartitionError> {
    let svm = processor::svm().ok_or(PartitionError::NoSvm)?;
    Ok(ASIDS.call_once(|| Asids::new(svm.asids)))
}

/// What the processor said it supports, read on first use.
static ASIDS: Once<Asids> = Once::new();

/// Why a guest could not be established, joined or repaired.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum PartitionError {
    /// The processor has no virtualization extension, so there is nothing to
    /// build a guest on.
    #[error("this processor has no virtualization extension")]
    NoSvm,
    /// Every address space identifier the processor supports has been handed
    /// out.
    #[error("all {count} address space identifiers are in use")]
    OutOfAsids {
        /// How many the processor supports, the host's own included.
        count: u32,
    },
    /// The guest's memory could not be described.
    #[error(transparent)]
    Npt(#[from] NptError),
    /// A processor could not be prepared to run the guest.
    #[error(transparent)]
    Vcpu(#[from] VcpuError),
    /// The address space the tables are allocated from could not be reached.
    #[error(transparent)]
    Paging(#[from] PagingError),
}
