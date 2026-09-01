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
//! processor running it. So the set lives here, and every processor reaches the
//! same one.
//!
//! What answers for a region and where the region is are two records, kept
//! apart on purpose. Where it is belongs to the nested tables, which are the
//! only thing a fault can consult; what answers for it belongs to the set here;
//! and the two are tied by the name the tables hand out. So a device is
//! registered at any time, a region is trapped or given at any time, and
//! neither decision has to wait for the other.

#![no_std]

extern crate alloc;

mod asid;

use cpu::CpuIndex;
use emulate::{EmulateError, Mmio, MmioError, Outcome as Performed, Region};
use log::{error, info};
/// How a guest translates its addresses, which is what
/// [`Partition::with_memory`] needs and what a caller reads out of a virtual
/// processor before borrowing one.
pub use memory::Addressing;
use memory::{Linear, Physical};
use npt::{Answered, Change, Exposure, Npt, NptError, Outcome, RegionTag};
use paging::AddressSpace;
use spin::{Once, RwLock};
use svm::exit::NestedPageFault;
use thiserror::Error;
use vcpu::{AvicProvision, Guest, Host, Vcpu, VcpuError};
use vlapic::{GuestMemory, Redescribed, RegisterPage, VlapicError};
use x86_64::PhysAddr;

pub use crate::asid::{Asid, Asids};

/// One guest: its memory, the tag its translations carry, and the tables its
/// interrupts run on when the hardware delivers them.
#[derive(Debug)]
pub struct Partition {
    npt: Npt,
    nested_cr3: PhysAddr,
    asid: Asid,
    avic: Option<vcpu::AvicTables>,
    devices: RwLock<Mmio>,
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
            npt,
            asid,
            avic,
            devices: RwLock::new(Mmio::new()),
        })
    }

    /// Takes over the regions the hypervisor answers for instead of the
    /// hardware behind them.
    ///
    /// Two things per region, and they are separate decisions tied by one name.
    /// The nested tables are told to trap the region where the region is
    /// trapped at all, and answer with the name they gave it; and the
    /// device that answers for that name is recorded here. A region whose
    /// `trap` is `None` says nothing to the tables — something else already
    /// describes it, a page given to the guest over a frame nothing reads
    /// being the case that exists — and the name it already holds is the
    /// one its device is registered under.
    ///
    /// May be called at any time, as often as it likes: what a trap makes
    /// stricter is answered with rather than discharged here, so a caller can
    /// coalesce a set of them into one barrier.
    ///
    /// # Errors
    ///
    /// [`PartitionError::Unnamed`] for a device offered for a region nothing
    /// describes as one this hypervisor answers for, [`PartitionError::Npt`] if
    /// a region cannot be trapped, or [`PartitionError::Mmio`] if a device
    /// cannot be registered. A region that fails leaves the ones before it
    /// registered, because a half-described guest is not one to hand back.
    pub fn interpose(
        &self,
        space: &mut AddressSpace,
        regions: impl IntoIterator<Item = Region>,
    ) -> Result<Change, PartitionError> {
        let mut owed = Change::None;
        for region in regions {
            let (gpa, bytes) = (region.gpa, region.bytes);
            let tag = match region.trap {
                Some(trap) => {
                    let (tag, change) = self.npt.protect(space.frames(), gpa, bytes, trap)?;
                    owed = owed.and(change);
                    tag
                }
                // Nothing for the tables to change: whatever describes the region
                // already does, and a region they describe already has a name.
                None => {
                    self.npt
                        .region(gpa)
                        .ok_or(PartitionError::Unnamed { gpa: gpa.as_u64() })?
                        .tag
                }
            };
            self.devices
                .write()
                .register(space, tag, gpa, bytes, region.device)?;
        }
        Ok(owed)
    }

    /// Discharges what a change to the guest's memory made stricter, so that no
    /// processor can begin an access with a translation it invalidated.
    ///
    /// Every operation that describes a guest's memory answers with what it
    /// made stricter rather than discharging it, so a caller making several
    /// of them pays for one barrier rather than one each. A
    /// [`Change::None`] or a [`Change::Loosened`] costs nothing at all and
    /// reaches nobody.
    ///
    /// # Errors
    ///
    /// [`PartitionError::Npt`] if a processor inside the guest could not be
    /// made to leave it, which means it may still be acting on what the
    /// change replaced.
    pub fn barrier(&self, change: Change) -> Result<(), PartitionError> {
        Ok(self.npt.barrier(change)?)
    }

    /// The region something other than the hardware answers for that a guest
    /// physical address is in, or `None` if the hardware answers for it.
    ///
    /// What an exit path asks when it has an address and needs the name the
    /// device answering it is kept under — which is every exit that reports an
    /// access the hardware performed part of rather than faulting on.
    #[must_use]
    pub fn region(&self, gpa: PhysAddr) -> Option<Answered> {
        self.npt.region(gpa)
    }

    /// Performs the instruction behind an access to a region this hypervisor
    /// answers for.
    ///
    /// `tag` is the region the access is in, which the caller has from the
    /// fault or from [`Partition::region`]. Nothing is decoded until
    /// something is known to answer for it: a region that is described as
    /// this hypervisor's with no device registered is a state the two
    /// decisions being independent allows, and decoding an instruction to
    /// dispatch into nothing would report the wrong thing about it.
    ///
    /// # Errors
    ///
    /// [`EmulateError::NoDevice`] if nothing answers for that region, or
    /// whatever performing the instruction reports.
    pub fn dispatch(
        &self,
        vcpu: &mut Vcpu,
        tag: RegionTag,
        gpa: PhysAddr,
        cause: NestedPageFault,
    ) -> Result<Performed, EmulateError> {
        let devices = self.devices.read();
        if !devices.answers(tag) {
            return Err(EmulateError::NoDevice {
                region: tag.number(),
            });
        }
        let addressing = Addressing::from_save(vcpu.save());
        let physical = Physical::new(&self.npt, self.npt.window());
        devices.dispatch(vcpu, Linear::new(physical, addressing), gpa, cause)
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
    /// hardware behind it, and answers with what that made stricter.
    ///
    /// What the interrupt controllers' register page becomes when the
    /// processor serves the controller itself: a read that no longer exits
    /// must still land somewhere.
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
    pub fn sink(&self, space: &mut AddressSpace, gpa: PhysAddr) -> Result<Change, PartitionError> {
        let (_, change) = self.npt.sink(space.frames(), gpa)?;
        Ok(change)
    }

    /// Stops giving that page to the guest, leaving it with no translation and
    /// its frame held back until the barrier this owes has passed.
    ///
    /// The counterpart of [`Partition::sink`], and the direction that pays: a
    /// processor which has run this guest may hold a translation of the page to
    /// the frame being handed back, so the frame is not reusable until the
    /// barrier has returned.
    ///
    /// # Errors
    ///
    /// [`PartitionError::Npt`] if the page was not being sunk, or if the tables
    /// describing it cannot be reached.
    pub fn unsink(&self, gpa: PhysAddr) -> Result<Change, PartitionError> {
        Ok(self.npt.unsink(gpa)?)
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
    ) -> Result<Outcome, PartitionError> {
        Ok(self.npt.fault(gpa, cause)?)
    }

    /// Makes immutable hypervisor-owned entry code or data visible to the
    /// guest, and answers with what that made stricter.
    ///
    /// The allocator the tables it needs come from is the caller's, which is
    /// what the address space is while the guest is being built.
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
    ) -> Result<Change, PartitionError> {
        Ok(self.npt.expose(space.frames(), gpa, bytes, exposure)?)
    }

    /// Takes back what [`Partition::expose`] made visible, leaving the range
    /// reading as zeroes like the rest of the hypervisor's memory.
    ///
    /// How entry code is retired once the guest is past it. It takes permission
    /// away, so a processor that has run this guest may hold a translation
    /// these tables no longer justify; the barrier that answers for it is the
    /// caller's to discharge, and the processor making the next entry discards
    /// what it cached on the way in.
    ///
    /// # Errors
    ///
    /// [`PartitionError::Npt`] if the range cannot be concealed.
    pub fn conceal(&self, gpa: PhysAddr, bytes: u64) -> Result<Change, PartitionError> {
        Ok(self.npt.conceal(gpa, bytes)?)
    }

    /// Publishes that this processor is entering the guest, and arms the
    /// discard of what it cached of the guest's memory where that memory
    /// has changed since its last entry.
    ///
    /// What the tables need on the way in, and the whole of what being able to
    /// change a guest's memory while it runs costs a world switch.
    ///
    /// Which processor it is is read here rather than handed in. The tables
    /// reach a processor by its position in the machine's roster, and the only
    /// processor that can honestly answer that is the one entering — so a
    /// parameter would be a fact the caller has to fetch for this and could
    /// fetch wrongly.
    pub fn before_entry(&self, vcpu: &mut Vcpu) {
        self.npt.before_entry(vcpu, here());
    }

    /// Publishes that this processor has left the guest, so that a change to
    /// the guest's memory stops having to make it leave.
    pub fn after_exit(&self) {
        self.npt.after_exit(here());
    }

    /// Borrows this guest's memory translated the way one virtual processor
    /// currently translates.
    ///
    /// Takes how the guest translates rather than the state-save area it was
    /// read out of, because [`Addressing`] is a small copied value and a save
    /// area is not — and because a caller that borrows the save area to make
    /// this call has borrowed the virtual processor, which is usually the very
    /// thing the closure needs.
    ///
    /// What it hands out is a view coherent per entry rather than across the
    /// whole closure, which is exactly what the hardware gives the guest: the
    /// tables answer every translation out of whatever they say at the moment
    /// it is asked, and nothing is held to keep two of them agreeing. So the
    /// closure may reach for the tables again — a device answering an
    /// intercepted access may resolve a fault — because there is nothing here
    /// to deadlock on.
    pub fn with_memory<T>(
        &self,
        addressing: Addressing,
        use_memory: impl for<'a> FnOnce(Linear<'a>) -> T,
    ) -> T {
        let physical = Physical::new(&self.npt, self.npt.window());
        use_memory(Linear::new(physical, addressing))
    }

    /// Borrows this guest's memory as physical, the way the hardware behind a
    /// device sees it.
    ///
    /// For a holder that works in guest-physical terms and has no guest
    /// translation to offer — a device answering an access, reading the
    /// queues its hardware shares with the guest. The view is the same one
    /// [`Partition::with_memory`] builds its linear view on, with the same
    /// per-entry coherence, and the same absence of anything to deadlock on.
    pub fn with_physical<T>(&self, use_memory: impl for<'a> FnOnce(Physical<'a>) -> T) -> T {
        let physical = Physical::new(&self.npt, self.npt.window());
        use_memory(physical)
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
        self.npt.describe(who);
        self.devices.read().describe(who);
    }
}

/// How the interrupt controllers' register page is described in this guest's
/// memory.
///
/// The one page whose description follows something that changes while the
/// guest runs, and the direction the call comes from is the point: what it
/// should be is the controllers' own to decide, and only the guest knows where
/// its memory is described. So the decision is made above and handed down here,
/// at the entry that may have to change it.
impl GuestMemory for Partition {
    /// Brings the page to `wanted` and discharges whatever that owed.
    ///
    /// One of the two directions pays and the other does not. Giving the page
    /// grants permission — an entry that described nothing now describes memory
    /// the guest may write — and the walker notices a lifted constraint by
    /// itself, so nothing is owed. Withholding it takes permission away, and
    /// what the barrier does is stop every processor beginning an access with
    /// the translation it replaced.
    ///
    /// A failure is reported here rather than answered with, because what a
    /// caller can do about it is decide what to run rather than to look at the
    /// reason: it is the description of this guest's memory that would not
    /// move, and this is where that memory is.
    ///
    /// A barrier that could not reach every processor is one of those failures,
    /// and it leaves the page trapped with some processor possibly still
    /// holding the translation it replaced. That processor is one still
    /// inside the guest, which is a processor still driving its controller
    /// out of the page — the state the translation it holds is correct for.
    /// Every processor that gives the acceleration up does so at an entry,
    /// and an entry after a barrier has advanced what it compares against
    /// discards this guest's translations before the guest runs again, so
    /// the one that matters cannot be missed.
    fn describe_register_page(&self, wanted: RegisterPage) -> Redescribed {
        let page = vlapic::apic_page();
        let moved = match wanted {
            RegisterPage::Given => self.npt.give(page),
            RegisterPage::Interposed => self.npt.withhold(page),
        };
        let discharged = moved.and_then(|change| {
            self.npt.barrier(change)?;
            Ok(change)
        });
        match discharged {
            Ok(Change::None) => Redescribed::Already,
            Ok(Change::Loosened | Change::Tightened { .. }) => Redescribed::Moved,
            Err(cause) => {
                error!(
                    "partition: the controllers' register page at {page:#x} could not be described \
                     as {wanted:?}: {cause}"
                );
                Redescribed::Refused
            }
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
    /// A device was offered for a region nothing describes as one this
    /// hypervisor answers for, so there is no name to register it under.
    #[error("nothing describes guest physical {gpa:#x} as a region this hypervisor answers for")]
    Unnamed {
        /// Where the region was said to begin.
        gpa: u64,
    },
    /// A device could not be registered for a region.
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

/// Which processor is running this, as the nested tables name one.
///
/// Read rather than handed in, because a processor's position in the machine's
/// roster is a fact only that processor can answer for. One load through the
/// `GS` base, which is what a hook on the world switch may afford.
fn here() -> CpuIndex {
    // SAFETY: every processor attaches immediately after installing its
    // descriptor tables and before it is given a control block, so a processor
    // entering or leaving a guest has attached — and nothing in this image
    // loads a segment selector into `GS` afterwards, which is the one thing
    // that would zero the base again.
    unsafe { cpu::current() }.index()
}
