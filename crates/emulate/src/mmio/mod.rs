//! The regions a guest is not allowed to reach the hardware through, and what
//! answers for them instead.
//!
//! # Registering is a different phase from dispatching, on purpose
//!
//! Trapping a region reduces what the nested tables permit, and reducing that
//! while a guest is running would mean discarding every processor's cached
//! translations before the guest could be let go again. Nothing here does that,
//! because nothing here has to: regions are registered before the guest has
//! ever run, when no translation can have been cached.
//!
//! That is not a comment asking to be obeyed. [`Registrar`] is the only thing
//! with a `register`, it borrows the nested tables for as long as it exists,
//! and the only way to get an [`Mmio`] — the thing an exit handler holds — is
//! to consume the registrar with [`Registrar::seal`]. So by the time a guest
//! can run, there is nothing in existence that could trap a region and nothing
//! holding the tables that would have to be reached to try.
//!
//! # A region need not be trapped at all
//!
//! Trapping is how a guest's *own* access arrives here, and it is not the only
//! way one can. Where the processor serves a device itself and reports back the
//! accesses it declines to serve, the report carries the address and the
//! direction, and performing such an access is performing it against the device
//! that answers the region — with no fault involved and nothing for the nested
//! tables to say. So a [`Region`] whose `trap` is `None` is registered for
//! dispatch and leaves the tables untouched, and whatever the guest's ordinary
//! accesses reach instead is the caller's to arrange.
//!
//! # What a device is asked, and what it is not
//!
//! A handler is asked what the guest should see, and is *given* what the guest
//! did. Neither question involves the hardware unless the handler says so:
//!
//! On a read, [`Read::hardware`] performs the device read — but only if it is
//! called. A handler inventing a value out of its own state never touches the
//! device and never pays for it, and one that wants the real value with a bit
//! changed asks for it and changes the bit. There is no flag to set beforehand
//! and no branch decided in advance; not asking is how you do not pay.
//!
//! On a write, the value is already in hand: it came out of a register or an
//! immediate, and no device access was needed to find it. So a handler decides
//! only what should happen to it — reach the hardware unchanged, reach it
//! changed, or not reach it at all — and the framework performs whichever, at
//! the guest's own width and offset. No device emulator writes that code.
//!
//! # A device states what it can answer
//!
//! An aperture is not a byte array. A register that decodes only aligned 32-bit
//! accesses does something undefined with a misaligned 16-bit one — often an
//! abort, sometimes a wrong value, occasionally nothing — and none of that is
//! behaviour to reproduce by attempting it and seeing. A hypervisor that
//! forwards whatever the guest happened to encode is a hypervisor whose
//! stability depends on guest code being well behaved.
//!
//! So a [`Device`] declares its [`Capability`], every access is checked against
//! it before the device is asked anything, and an access outside it is refused
//! with a reason rather than attempted. That is also what confines the unsafe
//! part of this crate: [`Admitted`] is the only way to reach the hardware, and
//! only the capability check makes one.
//!
//! # A device is asked by every processor at once
//!
//! Both questions are asked through a shared reference, and a [`Device`] must
//! be [`Send`] and [`Sync`]. That is not a restriction imposed for tidiness; it
//! is what a region's address means. One region is one range of *guest
//! physical* addresses, and every processor running the guest reaches it — so
//! the thing answering for it is reached from every processor, concurrently,
//! and the only honest signature for that is a shared one. `Send` is required
//! as well because the set of regions belongs to the guest rather than to a
//! processor, and a guest is composed on one processor and run on all of them.
//!
//! A device whose registers are per-processor is therefore written the way the
//! hardware it stands for is built: one device, holding a table indexed by
//! processor, picking this processor's row out of it. A device that keeps
//! machine-wide state keeps it in whatever the state's own shape calls for —
//! atomics for a counter, a lock for a structure — and pays for that only where
//! it is genuinely shared, rather than paying for a lock around the whole
//! dispatch path because the signature demanded one.

#[cfg(test)]
pub(crate) mod harness;
mod window;

use alloc::{boxed::Box, vec::Vec};

use log::info;
use npt::{Npt, NptError};
use paging::{AddressSpace, CacheType, Mapping, PagingError, Protection, chunk::FRAME_SIZE};
use thiserror::Error;
use x86_64::PhysAddr;

use crate::{
    EmulateError, Inadmissible, Spanning,
    machine::{Cpu, Guest},
    mmio::window::{Admitted, Window},
    operand::Place,
    value::{Data, Width},
};

/// What answers for a region instead of the hardware behind it.
///
/// Both methods take a shared reference and the trait requires [`Send`] and
/// [`Sync`], because every processor running the guest reaches the same region.
/// A device that has state to change on an access owns whatever makes that
/// sound — a per-processor table, atomics, a lock around the part that is
/// really shared.
pub trait Device: Send + Sync {
    /// Which accesses this device can be asked to answer.
    ///
    /// Checked at registration and again before every access. A device that
    /// declares more than the hardware behind it really tolerates has widened
    /// the unsafe contract this crate rests on, which is why this is a
    /// declaration with a stated meaning rather than a hint.
    fn capability(&self) -> Capability;

    /// Whether this device ever reaches the hardware behind its region.
    ///
    /// Checked at registration, and it decides whether a mapping is made at
    /// all: a device that answers every read out of its own state and lets no
    /// write through has no use for one, and a mapping nothing uses is a
    /// writable alias of a device's registers at a host address for no reason.
    /// A device that declares [`Hardware::Untouched`] finds
    /// [`Read::hardware`] answering `None`, and a write it asks to let through
    /// is refused rather than performed.
    fn hardware(&self) -> Hardware;

    /// What the guest should see.
    ///
    /// Call [`Read::hardware`] for what the device really holds, or do not, and
    /// answer out of whatever state this device keeps.
    ///
    /// The answer must be [`Read::width`] bytes wide. One that is not is a
    /// contract error, reported as such, and no guest state is changed from it.
    fn read(&self, access: Read<'_>) -> Data;

    /// What should become of what the guest wrote.
    fn write(&self, access: Write<'_>) -> Commit;
}

/// Whether the hardware behind a region is reached at all.
///
/// The question is not whether a device *has* registers of its own — every one
/// of these stands for something — but whether this hypervisor touches them.
/// An emulated controller whose whole point is that the guest must not reach
/// the real one behind it answers every access out of its own state, and a
/// mapping of the real registers would then be a standing writable alias of
/// them that nothing reads and nothing writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hardware {
    /// The device's registers are mapped for as long as the region exists, so a
    /// handler may read them and may let a write through to them.
    Reached,
    /// Nothing behind the region is ever touched, so nothing maps it.
    Untouched,
}

/// Which accesses a device can be asked to answer.
///
/// The default is the conservative one: aligned accesses of any scalar width,
/// no vector accesses. A device that decodes less than that has to say so, and
/// one that tolerates more may say so.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capability {
    widths: Widths,
    vectors: Vectors,
}

impl Capability {
    /// Aligned accesses at any scalar width, and no vector access.
    ///
    /// What almost every register file wants, and what a device gets if it says
    /// nothing more.
    #[must_use]
    pub const fn scalar() -> Self {
        Self {
            widths: Widths::ANY,
            vectors: Vectors::Refused,
        }
    }

    /// Only accesses of exactly this width.
    ///
    /// For the many devices whose registers are all one size and whose
    /// behaviour at any other size is undefined — a controller whose every
    /// register is an aligned dword, most obviously.
    #[must_use]
    pub const fn only(width: Width) -> Self {
        Self {
            widths: Widths::one(width),
            vectors: Vectors::Refused,
        }
    }

    /// The same, also accepting sixteen-byte accesses as two quadwords.
    ///
    /// This hypervisor cannot make a sixteen-byte access in one bus
    /// transaction: doing so needs a vector register, the guest's own
    /// vector state is what is in them, and borrowing one across an access
    /// that can fault would abandon guest state with no other copy. So a
    /// vector access becomes two quadword transactions in ascending order,
    /// and a device may only opt in to that if being seen as two is
    /// harmless for it.
    #[must_use]
    pub const fn split_vectors(mut self) -> Self {
        self.vectors = Vectors::Split;
        self
    }

    /// Whether this device answers accesses of that width at all.
    #[must_use]
    pub const fn answers(&self, width: Width) -> bool {
        match width {
            Width::Vector => matches!(self.vectors, Vectors::Split),
            width => self.widths.contains(width),
        }
    }

    /// Checks one access against what this device answers, and vouches for it.
    ///
    /// The only thing in this crate that makes an [`Admitted`], which is the
    /// only thing that reaches hardware. Three questions, in the order that
    /// a failure of each is cheapest to report: is it inside the region, is
    /// it a width this device decodes, is it aligned as this device
    /// requires.
    ///
    /// # Errors
    ///
    /// [`Inadmissible`] naming which of the three failed.
    fn admit(self, window: &Window, offset: u64, width: Width) -> Result<Admitted, Inadmissible> {
        if !window.inside(offset, width) {
            return Err(Inadmissible::PastEnd);
        }
        if !self.answers(width) {
            return Err(if width == Width::Vector {
                // Distinguished from an ordinary width refusal because the reason
                // is not that the device is narrow: it is that this hypervisor
                // cannot make the transaction the device would need to see.
                Inadmissible::Indivisible
            } else {
                Inadmissible::Width
            });
        }
        // A vector access is made as two quadwords, so what has to be aligned for
        // the pointers it goes through is a quadword rather than sixteen bytes.
        let granularity = match width {
            Width::Vector => Width::Quad,
            width => width,
        };
        // Not merely the device's preference. The loads and stores this admits go
        // through a pointer of the width being moved, and one of those pointing at
        // an address that is not a multiple of its width is not a slow access but
        // an invalid one.
        //
        // x86 does permit a guest to make an unaligned scalar access, so a guest
        // can reach here having done something a real machine would have answered.
        // Serving it would mean splitting it into aligned pieces, which is several
        // bus transactions where the guest made one — the same objection that makes
        // a sixteen-byte access refusable, and answerable the same way: an explicit
        // per-device opt-in stating that being split is harmless. No device this
        // hypervisor has needs one, so there is nothing here to opt into, and the
        // access is refused rather than performed at a shape its device never
        // agreed to.
        if !offset.is_multiple_of(granularity.span()) {
            return Err(Inadmissible::Alignment);
        }
        Ok(Admitted::new(offset, width))
    }
}

impl Default for Capability {
    fn default() -> Self {
        Self::scalar()
    }
}

/// Which scalar widths a device decodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Widths(u8);

impl Widths {
    /// Every scalar width.
    const ANY: Self = Self(0b1111);

    /// Exactly one width.
    const fn one(width: Width) -> Self {
        Self(1 << Self::bit(width))
    }

    /// Whether this set holds that width.
    const fn contains(self, width: Width) -> bool {
        self.0 & (1 << Self::bit(width)) != 0
    }

    /// Which bit of the set a width is.
    const fn bit(width: Width) -> u32 {
        match width {
            Width::Byte => 0,
            Width::Word => 1,
            Width::Long => 2,
            Width::Quad => 3,
            // Vector accesses are not part of this set: whether one can be made at
            // all is a different question from which scalar widths a device
            // decodes, and conflating them would let `only(Vector)` describe a
            // device with no scalar registers.
            Width::Vector => 4,
        }
    }
}

/// Whether a device tolerates a sixteen-byte access being made as two.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Vectors {
    /// It does not, so a sixteen-byte access to it is refused.
    Refused,
    /// It does: two quadword transactions in ascending order are equivalent,
    /// for this device, to the one the guest made.
    Split,
}

/// A region to be answered for, and by what.
pub struct Region {
    /// Where the region begins in the guest's physical memory, which is also
    /// where the hardware behind it is. Page aligned.
    pub gpa: PhysAddr,
    /// How long it is. A whole number of pages, because permissions come a page
    /// at a time.
    pub bytes: u64,
    /// Which of the guest's accesses have to come back to us, or `None` where
    /// none of them do.
    ///
    /// `None` leaves the nested tables exactly as they are: the guest's own
    /// accesses go wherever the tables already send them, and the device is
    /// reached only for the accesses something else declines to serve and
    /// reports.
    pub trap: Option<Trap>,
    /// What answers them.
    pub device: Box<dyn Device>,
}

/// Which of a guest's accesses to a region have to come back to us.
pub use npt::Trap;

/// What should become of a write the guest made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Commit {
    /// Let the guest's own value reach the device.
    Hardware,
    /// Let this reach the device instead.
    ///
    /// Must be the same width as the guest's own write. A replacement of
    /// another width would be a different transaction against the device —
    /// at a different alignment, possibly past the end of the region —
    /// rather than the access the device was asked about, so it is refused.
    Replace(Data),
    /// The device never sees it.
    Discard,
}

/// A read the guest made of a device.
///
/// Borrows the window for exactly the callback it is passed to, and is neither
/// `Copy` nor `Clone`: a handler that could keep one could reach the hardware
/// after its access had finished, or from another processor, with nothing left
/// checking that the mapping still exists.
#[derive(Debug)]
pub struct Read<'a> {
    window: &'a Window,
    admitted: Admitted,
    gpa: PhysAddr,
}

impl Read<'_> {
    /// How far into the region the guest read.
    #[must_use]
    pub const fn offset(&self) -> u64 {
        self.admitted.offset()
    }

    /// Where the guest read, as it thinks of the address.
    #[must_use]
    pub const fn gpa(&self) -> PhysAddr {
        self.gpa
    }

    /// How much the guest read, which is how much the answer must be.
    #[must_use]
    pub const fn width(&self) -> Width {
        self.admitted.width()
    }

    /// What the device really holds there, read now, or `None` for a device
    /// that declared [`Hardware::Untouched`] and so has no mapping to read
    /// through.
    ///
    /// The access is made when this is called and not before, so a handler that
    /// does not need it does not make it. Calling it twice makes two device
    /// reads, which for a register that changes as it is read is two
    /// different answers — that is the device's behaviour, faithfully, and
    /// not something to hide behind a cache.
    #[must_use]
    pub fn hardware(&self) -> Option<Data> {
        self.window.read(&self.admitted)
    }
}

/// A write the guest made to a device.
///
/// Borrowed for the callback, as [`Read`] is and for the same reasons.
#[derive(Debug)]
pub struct Write<'a> {
    window: &'a Window,
    admitted: Admitted,
    gpa: PhysAddr,
    value: Data,
}

impl Write<'_> {
    /// How far into the region the guest wrote.
    #[must_use]
    pub const fn offset(&self) -> u64 {
        self.admitted.offset()
    }

    /// Where the guest wrote, as it thinks of the address.
    #[must_use]
    pub const fn gpa(&self) -> PhysAddr {
        self.gpa
    }

    /// How much the guest wrote, which is how wide a replacement must be.
    #[must_use]
    pub const fn width(&self) -> Width {
        self.admitted.width()
    }

    /// What the guest tried to write.
    #[must_use]
    pub const fn value(&self) -> Data {
        self.value
    }

    /// What the device holds there now, read before anything is written, or
    /// `None` for a device that declared [`Hardware::Untouched`].
    ///
    /// For the registers where what the guest wrote is only part of the answer:
    /// a bit to set in a field that must otherwise be left alone, or a
    /// write-to-clear register whose other bits must survive.
    #[must_use]
    pub fn hardware(&self) -> Option<Data> {
        self.window.read(&self.admitted)
    }
}

/// Trapping regions, before the guest runs.
///
/// Holds the nested tables and the address space for as long as it exists,
/// which is what makes the phase separation a property of the types rather than
/// a rule to remember: nothing else can reduce what the tables permit while a
/// registrar is alive, and a registrar cannot outlive [`Registrar::seal`].
pub struct Registrar<'a> {
    space: &'a mut AddressSpace,
    npt: &'a mut Npt,
    regions: Vec<Interposed>,
}

impl<'a> Registrar<'a> {
    /// Nothing trapped yet, with the tables to trap regions in.
    pub fn new(space: &'a mut AddressSpace, npt: &'a mut Npt) -> Self {
        Self {
            space,
            npt,
            regions: Vec::new(),
        }
    }

    /// Reserves room for that many regions, so that registering one cannot fail
    /// for want of memory after it has already changed the tables.
    ///
    /// # Errors
    ///
    /// [`MmioError::Storage`] if the heap cannot spare the room, which is worth
    /// knowing before anything has been mapped rather than after.
    pub fn reserve(&mut self, regions: usize) -> Result<(), MmioError> {
        self.regions
            .try_reserve(regions)
            .map_err(|_| MmioError::Storage { regions })
    }

    /// Takes over a region of the guest's physical memory.
    ///
    /// Either the whole region is taken over or nothing is. Three things have
    /// to happen — the device's registers are mapped where the device reaches
    /// them at all, the nested tables are told to trap the region where the
    /// region is trapped at all, and the device is remembered — and each of
    /// them can fail, so each is undone if a later one does. The order is
    /// chosen so that the failure of one leaves the least to undo: room to
    /// remember the region is reserved first, because a reservation is the
    /// only step that can fail *after* the tables have been changed and
    /// cannot be undone by changing them back.
    ///
    /// A region whose `trap` is `None` skips the middle step entirely and the
    /// tables are not reached at all, which is what makes registering a device
    /// and trapping a region two decisions rather than one.
    ///
    /// # Errors
    ///
    /// [`MmioError::Geometry`] unless the region is a whole number of pages on
    /// a page boundary that the processor can address,
    /// [`MmioError::Overlaps`] if another region already covers part of it,
    /// [`MmioError::Capability`] if the device declares one this crate
    /// cannot serve, [`MmioError::Storage`] if there is no room to remember
    /// it, [`MmioError::Paging`] if the mapping window has no room, or
    /// [`MmioError::Npt`] if the nested tables cannot describe it a page at
    /// a time. [`MmioError::Rollback`] if undoing a failed registration
    /// itself failed, which is the one case that leaves the guest's tables in a
    /// state this crate cannot describe.
    pub fn register(&mut self, region: Region) -> Result<(), MmioError> {
        let (gpa, bytes) = (region.gpa, region.bytes);
        let end = geometry(gpa, bytes)?;
        if self.regions.iter().any(|other| other.overlaps(gpa, bytes)) {
            return Err(MmioError::Overlaps { gpa: gpa.as_u64() });
        }
        // A device that answers nothing would trap every access and refuse every
        // one of them, which is a region the guest can never use rather than a
        // device.
        let capability = region.device.capability();
        if !Width::ALL.iter().any(|width| capability.answers(*width)) {
            return Err(MmioError::Capability { gpa: gpa.as_u64() });
        }
        // Reserved before anything external changes, because this is the only step
        // that cannot be undone by putting something back the way it was.
        self.reserve(1)?;

        let aperture = match region.device.hardware() {
            // SAFETY: this is a device aperture rather than memory — the caller is
            // registering it precisely because hardware answers there — so there is
            // nothing for a writable alias to conflict with. The range is checked
            // above to be page aligned, a whole number of pages, and within the
            // processor's physical address width. Uncached is what a device register
            // needs: a write that sat in a cache line would never reach the bus.
            Hardware::Reached => Aperture::Mapped(unsafe {
                self.space
                    .map_physical(gpa, bytes, Protection::ReadWrite, CacheType::UncachedMinus)
            }?),
            // Nothing to map. The device answers out of its own state and lets
            // nothing through, so a mapping of the registers behind it would be a
            // writable alias of somebody's hardware that no access ever reaches.
            Hardware::Untouched => Aperture::Untouched { bytes },
        };
        // Nothing to change for a region the guest's own accesses never fault
        // on: the tables already send them somewhere, and this device answers
        // only what is reported to it.
        if let Some(trap) = region.trap
            && let Err(error) = self.npt.protect(self.space.frames(), gpa, bytes, trap)
        {
            // The aperture was made one statement ago, nothing has been handed its
            // address, and the region is not in the list — so nothing derived from
            // it exists anywhere.
            if let Some(unmapping) = aperture.release(self.space) {
                // The trap is gone but the window is not, and the address space
                // has retired the run rather than handing it back. Reported rather
                // than logged: a caller that carries on believing the region was
                // simply refused would be wrong about how much of the machine is
                // still described.
                return Err(MmioError::Rollback {
                    gpa: gpa.as_u64(),
                    cause: unmapping,
                });
            }
            return Err(error.into());
        }

        // Cannot reallocate: the room was reserved above.
        self.regions.push(Interposed {
            gpa,
            end,
            trap: region.trap,
            aperture,
            device: region.device,
        });
        Ok(())
    }

    /// Closes the set of trapped regions, which is what makes it usable.
    ///
    /// After this there is no way to trap another, and the tables that would
    /// have to be reached to try are no longer borrowed — which is the
    /// point: every region a guest could reach is now described, and no
    /// cached translation anywhere can disagree with the tables.
    #[must_use]
    pub fn seal(self) -> Mmio {
        Mmio {
            regions: self.regions,
        }
    }
}

/// Whether a region is one the nested tables and the processor can describe,
/// and where it ends.
///
/// The end is computed as an integer and checked before anything is done with
/// it. A region one page below the top of the address space has an exclusive
/// end that is not itself an address, and constructing one as a [`PhysAddr`]
/// panics — on a path that is either setting a guest up or logging what it was
/// set up with.
fn geometry(gpa: PhysAddr, bytes: u64) -> Result<u64, MmioError> {
    let malformed = || MmioError::Geometry {
        gpa: gpa.as_u64(),
        bytes,
    };
    if bytes == 0 || !gpa.as_u64().is_multiple_of(FRAME_SIZE) || !bytes.is_multiple_of(FRAME_SIZE) {
        return Err(malformed());
    }
    let end = gpa.as_u64().checked_add(bytes).ok_or_else(malformed)?;
    // The architecture's own container for a physical address is narrower than
    // sixty-four bits, and this processor's is narrower again. A region past
    // either is one whose translations could not be built and whose addresses
    // could not be printed.
    let addressable = u64::MAX >> (u64::BITS - u32::from(processor::physical_address_bits()));
    if end - 1 > addressable || PhysAddr::try_new(end - 1).is_err() {
        return Err(MmioError::Unaddressable {
            gpa: gpa.as_u64(),
            bytes,
            bits: processor::physical_address_bits(),
        });
    }
    Ok(end)
}

/// The trapped regions of a guest that is allowed to run.
///
/// Answered through a shared reference, so one of these serves every processor
/// running the guest rather than one per processor — which is what the regions
/// themselves are, since a guest physical address means the same thing on all
/// of them.
pub struct Mmio {
    regions: Vec<Interposed>,
}

impl Mmio {
    /// Logs which regions are answered for and by what, which is the whole of
    /// what a guest's view of its devices differs by.
    pub fn describe(&self, who: &str) {
        if self.regions.is_empty() {
            info!("{who}: no region of the guest's memory is interposed on");
            return;
        }
        for region in &self.regions {
            // The last byte rather than one past it: one past the end of a region
            // at the top of the address space is not an address at all.
            let (gpa, last) = (region.gpa.as_u64(), region.end - 1);
            match region.window().base() {
                Some(base) => info!(
                    "{who}: interposing on guest physical {gpa:#x}..={last:#x}, reached at {base:#x}"
                ),
                None => info!(
                    "{who}: interposing on guest physical {gpa:#x}..={last:#x}, whose device \
                     answers without reaching the hardware behind it"
                ),
            }
        }
    }

    /// Gives every region back: the mappings unmapped, the traps removed, and
    /// the devices returned to the caller.
    ///
    /// The counterpart registration never had. A guest that was built and then
    /// abandoned — because a later step of bring-up failed, or because it is
    /// being taken down — otherwise leaks a window run per region and a trap
    /// slot per trapped one, and leaves the nested tables trapping addresses
    /// nothing answers for.
    ///
    /// Every region is attempted even if one fails, because stopping at the
    /// first failure would leave the rest in exactly the state this exists
    /// to get out of. What could not be undone is reported.
    ///
    /// # Errors
    ///
    /// [`MmioError::Rollback`] naming the first region whose mapping could not
    /// be removed. The devices are returned regardless: they are the
    /// caller's, and dropping them because the address space complained
    /// would lose whatever state they hold.
    ///
    /// # Safety
    ///
    /// No processor may be running the guest these regions belong to, and
    /// nothing derived from any window may still be in use.
    pub unsafe fn teardown(self, space: &mut AddressSpace, npt: &mut Npt) -> Teardown {
        let mut devices = Vec::new();
        let mut failure = None;
        for region in self.regions {
            if region.trap.is_some()
                && let Err(error) = npt.release(region.gpa, region.end - region.gpa.as_u64())
            {
                failure = failure.or(Some(MmioError::Npt(error)));
            }
            if let Some(cause) = region.aperture.release(space) {
                failure = failure.or(Some(MmioError::Rollback {
                    gpa: region.gpa.as_u64(),
                    cause,
                }));
            }
            devices.push(region.device);
        }
        Teardown { devices, failure }
    }

    /// Which region an access falls in, with every byte of it accounted for.
    ///
    /// `gpa` is where the access begins and `width` is how long it is, and both
    /// matter: an access that begins in a region and ends outside it belongs to
    /// neither, and one that begins in ordinary memory and ends in a region is
    /// not ordinary memory.
    ///
    /// # Errors
    ///
    /// [`EmulateError::Span`] if the access is partly in a region and partly
    /// not, or spans two regions.
    pub(crate) fn classify(
        &self,
        gpa: PhysAddr,
        width: Width,
        linear: u64,
    ) -> Result<Place, EmulateError> {
        let span = |reason| EmulateError::Span {
            linear,
            bytes: width.bytes(),
            reason,
        };
        // The last byte rather than one past the end, and as an integer rather than
        // an address: an access that ends where the physical address space does has
        // no one-past-the-end address, and constructing one panics.
        let last = gpa
            .as_u64()
            .checked_add(width.span() - 1)
            .ok_or_else(|| span(Spanning::Wraps))?;

        // One pass, asking both questions of each region as it goes. Two calls to
        // a single-address lookup would scan the whole vector twice for every
        // memory operand of every intercepted instruction.
        let mut first = None;
        let mut ends_inside = false;
        for (index, region) in self.regions.iter().enumerate() {
            if region.holds(gpa.as_u64()) {
                first = Some((index, gpa.as_u64() - region.gpa.as_u64()));
            }
            if region.holds(last) {
                ends_inside = true;
            }
        }

        match first {
            // Wholly inside one region, which is the only shape a single device
            // transaction describes.
            Some((index, offset)) if self.regions[index].holds(last) => Ok(Place::Device {
                index,
                offset,
                gpa,
                linear,
            }),
            // It begins in a region and ends somewhere else. Whether that somewhere
            // is another region or ordinary memory changes the diagnostic and
            // nothing else: either way the access is partly a device transaction
            // and partly not, and which bytes go where is not something the
            // instruction says.
            Some(_) => Err(span(if ends_inside {
                Spanning::TwoRegions
            } else {
                Spanning::Straddles
            })),
            // It begins outside every region. If it ends inside one it still
            // straddles; if it does not, nothing interposes on any byte of it and
            // it is the guest's own memory.
            None if ends_inside => Err(span(Spanning::Straddles)),
            None => Ok(Place::Memory(linear)),
        }
    }

    /// Whether an access to a region is one the device behind it answers,
    /// without asking the device anything or touching the hardware.
    ///
    /// The preflight an instruction runs before it consumes a device read: this
    /// is the same check the access itself makes, so a destination that
    /// passes here will not be refused later for a reason that was knowable
    /// now.
    ///
    /// # Errors
    ///
    /// [`EmulateError::Inadmissible`] if the access is not one this device
    /// answers.
    pub(crate) fn admits(
        &self,
        index: usize,
        offset: u64,
        gpa: PhysAddr,
        width: Width,
    ) -> Result<(), EmulateError> {
        let region = self.at(index)?;
        region
            .device
            .capability()
            .admit(&region.window(), offset, width)
            .map(|_| ())
            .map_err(|reason| EmulateError::Inadmissible {
                gpa: gpa.as_u64(),
                bytes: width.bytes(),
                reason,
            })
    }

    /// What the guest should see for a read of a trapped region.
    ///
    /// # Errors
    ///
    /// [`EmulateError::Inadmissible`] if the access is not one this device
    /// answers, or if the device answers with a value of a width other than
    /// the one it was asked about.
    pub(crate) fn read(
        &self,
        index: usize,
        offset: u64,
        gpa: PhysAddr,
        width: Width,
    ) -> Result<Data, EmulateError> {
        let region = self.at(index)?;
        let window = region.window();
        let admitted = region
            .device
            .capability()
            .admit(&window, offset, width)
            .map_err(|reason| EmulateError::Inadmissible {
                gpa: gpa.as_u64(),
                bytes: width.bytes(),
                reason,
            })?;
        let value = region.device.read(Read {
            window: &window,
            admitted,
            gpa,
        });
        // The answer decides how much of a register is written, how many bytes go
        // into the guest's memory, and how much of a vector register is cleared. A
        // handler that answered with the wrong width would silently change the
        // instruction the guest executed, so the disagreement is reported here
        // rather than propagated.
        if value.width() == width {
            return Ok(value);
        }
        Err(EmulateError::Inadmissible {
            gpa: gpa.as_u64(),
            bytes: width.bytes(),
            reason: Inadmissible::Answer {
                wanted: width,
                got: value.width(),
            },
        })
    }

    /// Performs what a device decided should become of a write.
    ///
    /// # Errors
    ///
    /// As [`Mmio::read`], and if the device replaces the value with one of
    /// another width.
    pub(crate) fn write(
        &self,
        index: usize,
        offset: u64,
        gpa: PhysAddr,
        value: Data,
    ) -> Result<(), EmulateError> {
        let region = self.at(index)?;
        let window = region.window();
        let inadmissible = |reason| EmulateError::Inadmissible {
            gpa: gpa.as_u64(),
            bytes: value.width().bytes(),
            reason,
        };
        let admitted = region
            .device
            .capability()
            .admit(&window, offset, value.width())
            .map_err(inadmissible)?;
        let decision = region.device.write(Write {
            window: &window,
            admitted: Admitted::new(offset, value.width()),
            gpa,
            value,
        });
        let committed = match decision {
            Commit::Hardware => value,
            Commit::Replace(instead) => instead,
            Commit::Discard => return Ok(()),
        };
        // Bound to the width the access was admitted at, immediately before the
        // hardware is touched. Admission checked the guest's own width; nothing
        // has checked the handler's answer until here, and a wider one would write
        // past the end of a mapping through a misaligned pointer.
        admitted.binds(&committed).map_err(inadmissible)?;
        // A device that declared the hardware behind its region untouched has no
        // mapping to write through, so a handler asking for one is a contract
        // error rather than something to perform.
        if !window.write(&admitted, &committed) {
            return Err(inadmissible(Inadmissible::Untouched));
        }
        Ok(())
    }

    /// One region, by the index [`Mmio::classify`] gave.
    fn at(&self, index: usize) -> Result<&Interposed, EmulateError> {
        self.regions
            .get(index)
            .ok_or(EmulateError::NoSuchRegion { index })
    }

    /// Performs the instruction behind a nested page fault, whatever it turns
    /// out to be.
    ///
    /// Generic over the register file and the guest's memory so that the whole
    /// of this path is exercised on the host against fakes. It
    /// monomorphizes to one implementation in the hypervisor image, so
    /// nothing here is dispatched dynamically or costs a call it did not
    /// before.
    ///
    /// # Errors
    ///
    /// As [`Mmio::dispatch`](crate::Mmio::dispatch).
    pub(crate) fn execute(
        &self,
        cpu: &mut impl Cpu,
        guest: &impl Guest,
        gpa: PhysAddr,
        cause: svm::exit::NestedPageFault,
    ) -> Result<crate::Outcome, EmulateError> {
        if cause.instruction_fetch() {
            // The fetch itself faulted, so there is no instruction to perform and
            // no length to compute. Whatever is wrong is wrong about the mapping
            // the guest is executing from, which is not this crate's to put right.
            return Err(EmulateError::FetchFault { gpa: gpa.as_u64() });
        }
        let instruction = crate::decode::instruction(cpu, guest)?;
        let reported = crate::plan::Reported { gpa, cause };
        let outcome = if let Some(width) = crate::mov::string(instruction.code()) {
            crate::string::perform(self, cpu, guest, &instruction, width, reported)?
        } else {
            crate::mov::perform(self, cpu, guest, &instruction, reported)?
        };
        // Only a completed instruction moves the guest past it. A batch with
        // repetitions left has to execute again, and one that owes a fault has to
        // execute again after the fault is delivered — in both cases from the
        // instruction itself, with the progress so far already in the registers.
        if outcome == crate::Outcome::Stepped {
            let next = crate::plan::after(cpu.save(), instruction.len()).ok_or(
                EmulateError::Undecodable {
                    rip: cpu.save().rip,
                    bytes: instruction.len(),
                    reason: crate::Undecodable::Overlong,
                },
            )?;
            cpu.save_mut().rip = next;
        }
        Ok(outcome)
    }
}

impl core::fmt::Debug for Mmio {
    /// The regions, not the devices: what answers for one is a trait object
    /// with no more to say about itself than its own address, and printing
    /// that would be noise.
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_list()
            .entries(self.regions.iter().map(|region| region.gpa))
            .finish()
    }
}

/// What became of taking a set of regions apart.
///
/// Carries the devices back out whatever happened, because they belong to
/// whoever registered them and may hold state that outlives the guest.
#[must_use = "the devices are returned here, and dropping this drops them"]
pub struct Teardown {
    devices: Vec<Box<dyn Device>>,
    failure: Option<MmioError>,
}

impl Teardown {
    /// The devices that answered for the regions.
    pub fn devices(&mut self) -> Vec<Box<dyn Device>> {
        core::mem::take(&mut self.devices)
    }

    /// What could not be undone, if anything.
    ///
    /// # Errors
    ///
    /// The first failure encountered while taking the regions apart.
    pub fn result(&self) -> Result<(), MmioError> {
        self.failure.map_or(Ok(()), Err)
    }
}

/// One region that is answered for, and by what.
struct Interposed {
    gpa: PhysAddr,
    /// One past the last byte, as an integer rather than an address: a region
    /// may end where the physical address space does, and one past that is
    /// not an address that can be constructed.
    end: u64,
    /// What the nested tables were told to fault on, if anything.
    ///
    /// Kept because giving a region back is the exact reverse of taking it
    /// over: a region the tables were never told about has nothing to tell
    /// them now, and asking them to release one would be asking about a range
    /// they have no record of.
    trap: Option<Trap>,
    aperture: Aperture,
    device: Box<dyn Device>,
}

impl Interposed {
    /// Whether this region covers that address.
    ///
    /// Takes the address as an integer, because one caller asks about the last
    /// byte of an access and that byte may be the last of the physical address
    /// space — which is a number but not a [`PhysAddr`] one can construct.
    fn holds(&self, gpa: u64) -> bool {
        (self.gpa.as_u64()..self.end).contains(&gpa)
    }

    /// Whether this region covers any of `bytes` from `gpa`.
    fn overlaps(&self, gpa: PhysAddr, bytes: u64) -> bool {
        let theirs = gpa.as_u64();
        theirs < self.end && self.gpa.as_u64() < theirs.saturating_add(bytes)
    }

    /// Where this device's registers are reachable.
    fn window(&self) -> Window {
        self.aperture.window()
    }
}

/// Where a region's registers are reachable, and what keeps them reachable.
///
/// Two variants in the hypervisor and one more in the tests. The distinction is
/// deliberately at this level and no deeper: everything above it — admission,
/// commit binding, the width of a transaction — is the same code either way, so
/// what the tests exercise is what runs on a machine.
enum Aperture {
    /// A mapping of the device's real aperture, held for as long as the region
    /// is.
    Mapped(Mapping),
    /// No mapping at all, for a device that declared the hardware behind its
    /// region untouched. The length is still kept, because admission is checked
    /// against it whether or not anything can be reached.
    Untouched {
        /// How long the region is.
        bytes: u64,
    },
    /// Bytes standing in for a device's registers.
    ///
    /// Leaked rather than owned, because a window is reached through a raw
    /// pointer while the region is only borrowed — exactly as a real aperture
    /// is, which is memory Rust does not own at all. Leaking a few
    /// kilobytes for the length of a test process is the price of the two
    /// paths being the same path.
    #[cfg(test)]
    Planted {
        /// Where the bytes are.
        base: x86_64::VirtAddr,
        /// How many of them there are.
        bytes: u64,
    },
}

impl Aperture {
    /// Where the registers this describes are reachable.
    fn window(&self) -> Window {
        match self {
            // SAFETY: the mapping was made when this region was registered, is
            // writable and uncached, covers exactly these bytes of a device
            // aperture, and is held by this value — so it outlives every window
            // made from it, which cannot escape the callback it is lent to.
            Self::Mapped(mapping) => unsafe { Window::new(mapping.addr(), mapping.bytes()) },
            Self::Untouched { bytes } => Window::unmapped(*bytes),
            // SAFETY: the bytes were leaked at construction, so they are live for
            // the rest of the process and nothing else holds a reference to them.
            #[cfg(test)]
            Self::Planted { base, bytes } => unsafe { Window::new(*base, *bytes) },
        }
    }

    /// Gives the addresses back, answering why if it could not.
    ///
    /// By value, because an aperture that has been released must not be
    /// reachable afterwards.
    fn release(self, space: &mut AddressSpace) -> Option<PagingError> {
        match self {
            // SAFETY: two callers, and each establishes the same thing. A failed
            // registration has given the mapping's address to nothing and has not
            // put the region in the list, and `teardown`'s caller guarantees no
            // processor is running the guest and that nothing derived from the
            // window is in use. The aperture is consumed here either way, so no
            // further access is representable.
            Self::Mapped(mapping) => unsafe { space.unmap(mapping) }.err(),
            // Nothing was mapped, so there is nothing to give back.
            Self::Untouched { .. } => None,
            // Deliberately leaked, and so nothing to give back: the bytes stand in
            // for a device aperture, which is not memory this process allocated.
            #[cfg(test)]
            Self::Planted { .. } => None,
        }
    }
}

/// Why a region could not be taken over.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum MmioError {
    /// The region is not a whole number of pages on a page boundary, which is
    /// the granularity the nested tables can give it permissions of its own
    /// at.
    #[error("a region at {gpa:#x} of {bytes:#x} bytes is not a whole number of pages")]
    Geometry {
        /// Where the region begins.
        gpa: u64,
        /// How long it is.
        bytes: u64,
    },
    /// The region reaches past what this processor can address physically.
    #[error(
        "a region at {gpa:#x} of {bytes:#x} bytes leaves this processor's {bits}-bit physical address space"
    )]
    Unaddressable {
        /// Where the region begins.
        gpa: u64,
        /// How long it is.
        bytes: u64,
        /// How many physical address bits this processor has.
        bits: u8,
    },
    /// Another region already covers part of this one, so which device answers
    /// for the overlap would depend on the order they were registered in.
    #[error("a region already covers part of {gpa:#x}")]
    Overlaps {
        /// Where the region begins.
        gpa: u64,
    },
    /// The device answers no access this crate can make, so every access to its
    /// region would trap and then be refused.
    #[error("the device for {gpa:#x} answers no access this hypervisor can make")]
    Capability {
        /// Where the region begins.
        gpa: u64,
    },
    /// There is no room to remember another region.
    #[error("there is no room to remember {regions} more interposed regions")]
    Storage {
        /// How many were asked for.
        regions: usize,
    },
    /// A failed registration could not be undone, so the guest's tables
    /// describe something this crate no longer accounts for.
    #[error("the window for the region at {gpa:#x} could not be removed: {cause}")]
    Rollback {
        /// Where the region begins.
        gpa: u64,
        /// Why the mapping could not be removed.
        cause: PagingError,
    },
    /// The nested tables could not describe the region a page at a time.
    #[error(transparent)]
    Npt(#[from] NptError),
    /// The device's registers could not be mapped.
    #[error(transparent)]
    Paging(#[from] PagingError),
}

#[cfg(test)]
mod tests {
    use super::{Capability, Commit, Vectors, Widths};
    use crate::value::{Data, Width};

    #[test]
    fn a_scalar_device_answers_every_scalar_width_and_no_vector() {
        let capability = Capability::scalar();
        for width in Width::ALL.into_iter().filter(|width| width.scalar()) {
            assert!(capability.answers(width), "{width:?} is a scalar width");
        }
        assert!(
            !capability.answers(Width::Vector),
            "a sixteen-byte access needs an explicit opt-in"
        );
    }

    #[test]
    fn a_device_of_one_width_answers_only_that_width() {
        for only in Width::ALL.into_iter().filter(|width| width.scalar()) {
            let capability = Capability::only(only);
            for width in Width::ALL {
                assert_eq!(
                    capability.answers(width),
                    width == only,
                    "a {only:?}-only device must not answer {width:?}"
                );
            }
        }
    }

    #[test]
    fn vector_accesses_are_answered_only_when_splitting_is_declared_harmless() {
        let capability = Capability::scalar().split_vectors();
        assert!(capability.answers(Width::Vector));
        // The scalar widths are unaffected by opting in to the wide one.
        for width in Width::ALL.into_iter().filter(|width| width.scalar()) {
            assert!(capability.answers(width));
        }
    }

    #[test]
    fn the_default_capability_is_the_conservative_one() {
        assert_eq!(Capability::default(), Capability::scalar());
    }

    #[test]
    fn a_width_set_holds_exactly_what_was_put_in_it() {
        assert_eq!(Widths::ANY.0.count_ones(), 4, "four scalar widths");
        for width in Width::ALL.into_iter().filter(|width| width.scalar()) {
            assert!(Widths::ANY.contains(width));
            assert!(Widths::one(width).contains(width));
            for other in Width::ALL
                .into_iter()
                .filter(|other| other.scalar() && *other != width)
            {
                assert!(!Widths::one(width).contains(other));
            }
        }
    }

    #[test]
    fn a_vector_is_not_one_of_the_scalar_widths_a_set_holds() {
        // Otherwise `only(Vector)` would describe a device with no scalar
        // registers as though it had all of them.
        assert!(!Widths::ANY.contains(Width::Vector));
    }

    #[test]
    fn splitting_a_vector_access_is_refused_until_a_device_declares_it_harmless() {
        assert_eq!(Capability::scalar().vectors, Vectors::Refused);
        assert_eq!(Capability::scalar().split_vectors().vectors, Vectors::Split);
    }

    #[test]
    fn a_decision_is_the_three_things_a_device_can_want() {
        // A guard on the shape of the public enum: a fourth option would need a
        // decision about what it means for the hardware, and adding one silently
        // is exactly what this asserts against.
        let replace = Commit::Replace(Data::from_u64(1, Width::Long));
        assert_ne!(replace, Commit::Hardware);
        assert_ne!(replace, Commit::Discard);
        assert_eq!(
            Commit::Replace(Data::from_u64(1, Width::Long)),
            replace,
            "a replacement is compared by its value"
        );
    }
}
