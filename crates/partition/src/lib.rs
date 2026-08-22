//! One guest, across every processor it runs on.
//!
//! A virtual processor is one processor's view of a guest; this is the guest.
//! What lives here is what every one of its processors shares and none of them
//! owns: the description of its memory, the tag its cached translations are
//! kept apart by, and — when the processor delivers its interrupts without an
//! exit — the tables that tell the hardware where. What does not live here is
//! anything about running — that is [`vcpu`], and the split is deliberate,
//! because entering a guest is a per-processor act and deciding what a guest
//! *is* is not.
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
//! # The devices belong to the guest, not to a processor
//!
//! Which regions of a guest's memory the hypervisor answers for is a property
//! of the guest: a guest physical address means the same thing on every
//! processor running it. So the sealed set lives here, filled once by
//! [`Partition::interpose`] before any processor has entered the guest, and
//! read through a shared reference by all of them afterwards.
//!
//! The registrar never escapes that call. It is built, filled and sealed inside
//! it, which is what keeps [`emulate`]'s guarantee intact — there is no moment
//! at which something able to trap a region coexists with a guest that could
//! have cached a translation of one.

#![no_std]

extern crate alloc;

mod asid;

use alloc::vec::Vec;

use emulate::{Mmio, MmioError, Region, Registrar};
use log::info;
/// How a guest translates its addresses, which is what
/// [`Partition::with_memory`] needs and what a caller reads out of a virtual
/// processor before borrowing one.
pub use memory::Addressing;
use memory::{Linear, Physical};
use npt::{Exposure, Npt, NptError, Resolution};
use paging::AddressSpace;
use spin::{Mutex, Once};
use svm::exit::NestedPageFault;
use thiserror::Error;
use vcpu::{AvicProvision, Guest, Host, Vcpu, VcpuError};
use vlapic::VlapicError;
use x86_64::PhysAddr;

pub use crate::asid::{Asid, Asids};

/// One guest: its memory, the tag its translations carry, and the tables its
/// interrupts run on when the hardware delivers them.
#[derive(Debug)]
pub struct Partition {
    npt: Mutex<Npt>,
    nested_cr3: PhysAddr,
    asid: Asid,
    avic: Option<vcpu::AvicTables>,
    devices: Once<Mmio>,
}

impl Partition {
    /// Builds the one guest this hypervisor runs.
    ///
    /// Its memory is described lazily, so nothing is mapped here: the tables
    /// start empty and the guest's first access to any address is what
    /// describes the region containing it.
    ///
    /// `avic` is the set of structures hardware-driven interrupt delivery runs
    /// on, or `None` while every interrupt still exits and is emulated. It is
    /// shared by every processor of the guest; what is per-processor is
    /// resolved when each one attaches.
    ///
    /// # Errors
    ///
    /// [`PartitionError::NoSvm`] on a processor with no virtualization
    /// extension, [`PartitionError::OutOfAsids`] on one that can tag no guest,
    /// or [`PartitionError::Npt`] if the chunk cannot back the tables.
    pub fn establish(
        space: &mut AddressSpace,
        avic: Option<vcpu::AvicTables>,
    ) -> Result<Self, PartitionError> {
        let asid = asids()?.take()?;
        let window = space.direct_map();
        let npt = Npt::create(space.frames(), window)?;
        Ok(Self {
            nested_cr3: npt.root(),
            npt: Mutex::new(npt),
            asid,
            avic,
            devices: Once::new(),
        })
    }

    /// Takes over the regions the hypervisor answers for instead of the
    /// hardware behind them.
    ///
    /// Called once, on the boot processor, before any processor has entered the
    /// guest — which is what trapping a region requires, since reducing what
    /// the nested tables permit while a guest is running would mean
    /// discarding every processor's cached translations first.
    ///
    /// Room for every region is reserved before any of them is taken over, so
    /// that running out of memory is a failure that has changed nothing rather
    /// than one discovered after the tables have been edited.
    ///
    /// # Errors
    ///
    /// [`PartitionError::AlreadyInterposed`] for a second call, or
    /// [`PartitionError::Mmio`] if a region cannot be taken over. A region that
    /// fails is undone in full; the regions taken over before it stay taken
    /// over, because a half-trapped guest is not one to hand back.
    pub fn interpose(
        &self,
        space: &mut AddressSpace,
        regions: impl IntoIterator<Item = Region>,
    ) -> Result<(), PartitionError> {
        let mut outcome = Ok(());
        let mut sealed = false;
        let regions = regions.into_iter().collect::<Vec<_>>();
        self.devices.call_once(|| {
            sealed = true;
            let mut npt = self.npt.lock();
            let mut registrar = Registrar::new(space, &mut npt);
            outcome = registrar
                .reserve(regions.len())
                .map_err(PartitionError::from);
            if outcome.is_ok() {
                for region in regions {
                    if let Err(error) = registrar.register(region) {
                        outcome = Err(error.into());
                        break;
                    }
                }
            }
            registrar.seal()
        });
        if !sealed {
            return Err(PartitionError::AlreadyInterposed);
        }
        outcome
    }

    /// What answers for the regions this guest is not allowed to reach the
    /// hardware through, or `None` before [`Partition::interpose`].
    #[must_use]
    pub fn devices(&self) -> Option<&Mmio> {
        self.devices.get()
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
    /// the window cannot reach it, or [`PartitionError::Vlapic`] if the
    /// interrupt structures exist but this processor has no place in them.
    pub fn attach(
        &self,
        host: &'static Host,
        space: &mut AddressSpace,
    ) -> Result<Vcpu, PartitionError> {
        let window = space.direct_map();
        let avic = match self.avic {
            Some(tables) => Some(AvicProvision {
                backing_page: vlapic::backing_page()?,
                tables,
            }),
            None => None,
        };
        Ok(Vcpu::create(
            host,
            space.frames(),
            window,
            Guest {
                nested_cr3: self.nested_cr3,
                asid: self.asid.number(),
                avic,
            },
        )?)
    }

    /// Sends one page of the guest's memory to the zero sink rather than the
    /// hardware behind it.
    ///
    /// What the interrupt controllers' register page becomes when the
    /// processor serves the controller itself: a read that no longer exits
    /// must still land somewhere. Called once, before any processor has
    /// entered the guest, for the same cache coherency reason as
    /// [`Partition::expose`].
    ///
    /// The page it leaves behind is the one exception to the rest of the chunk
    /// being an immutable page of zeroes to the guest, and it is an exception
    /// in both directions: the frame is writable, so the guest may store to
    /// hypervisor-owned memory and read back what it stored. That is what the
    /// acceleration requires — the redirect needs a writable leaf at the
    /// address it redirects away from — and it is harmless because the
    /// frame is allocated for this page alone, never released, and read
    /// back by nothing.
    ///
    /// # Errors
    ///
    /// [`PartitionError::Npt`] if the page cannot be sunk — because it is
    /// already sunk, because it is one the hypervisor has taken over, or
    /// because the chunk cannot spare the frame behind it.
    pub fn sink(&self, space: &mut AddressSpace, gpa: PhysAddr) -> Result<(), PartitionError> {
        Ok(self.npt.lock().sink(space.frames(), gpa)?)
    }

    /// Describes the region containing a guest physical address the guest could
    /// not reach.
    ///
    /// What an exit handler calls on a nested page fault: `gpa` is the second
    /// exit-information field and `cause` the first, decoded.
    ///
    /// # Errors
    ///
    /// [`PartitionError::Npt`] if the region cannot be described.
    pub fn resolve(
        &self,
        gpa: PhysAddr,
        cause: NestedPageFault,
    ) -> Result<Resolution, PartitionError> {
        Ok(self.npt.lock().fault(gpa, cause)?)
    }

    /// Makes immutable hypervisor-owned entry code or data visible to the
    /// guest.
    ///
    /// This may be called only while the guest has not run, for the same cache
    /// coherency reason as [`Npt::expose`]. The allocator the tables it needs
    /// come from is the caller's, which is what the address space is while the
    /// guest is being built.
    ///
    /// # Errors
    ///
    /// [`PartitionError::Npt`] if the requested range cannot be exposed.
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
    /// [`PartitionError::Npt`] if the range cannot be concealed.
    pub fn conceal(&self, gpa: PhysAddr, bytes: u64) -> Result<(), PartitionError> {
        Ok(self.npt.lock().conceal(gpa, bytes)?)
    }

    /// Borrows this guest's memory translated the way one virtual processor
    /// currently translates.
    ///
    /// The nested tables remain locked for the closure, so every translation
    /// and read observes one coherent table state. The higher-ranked closure
    /// prevents the borrowed memory view from escaping that lock.
    ///
    /// Takes how the guest translates rather than the state-save area it was
    /// read out of, because [`Addressing`] is a small copied value and a save
    /// area is not — and because a caller that borrows the save area to make
    /// this call has borrowed the virtual processor, which is usually the very
    /// thing the closure needs.
    ///
    /// The closure runs with the tables locked, so nothing it calls may ask for
    /// them again: a device answering an intercepted access must not resolve a
    /// fault.
    pub fn with_memory<T>(
        &self,
        addressing: Addressing,
        use_memory: impl for<'a> FnOnce(Linear<'a>) -> T,
    ) -> T {
        let npt = self.npt.lock();
        let physical = Physical::new(&npt, npt.window());
        use_memory(Linear::new(physical, addressing))
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
        match self.devices.get() {
            Some(devices) => devices.describe(who),
            None => info!("{who}: this guest's trapped regions have not been sealed yet"),
        }
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
    /// The set of trapped regions has already been sealed, and sealing it is
    /// what makes it safe to enter the guest.
    #[error("this guest's trapped regions have already been sealed")]
    AlreadyInterposed,
    /// A region could not be taken over.
    #[error(transparent)]
    Mmio(#[from] MmioError),
    /// The guest's memory could not be described.
    #[error(transparent)]
    Npt(#[from] NptError),
    /// A processor could not be prepared to run the guest.
    #[error(transparent)]
    Vcpu(#[from] VcpuError),
    /// The guest's interrupt structures could not be reached.
    #[error(transparent)]
    Vlapic(#[from] VlapicError),
}
