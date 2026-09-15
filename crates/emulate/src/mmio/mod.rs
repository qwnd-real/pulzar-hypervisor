//! The regions a guest is not allowed to reach the hardware through, and what
//! answers for them instead.
//!
//! # Registering a device and trapping a region are two decisions
//!
//! What answers for a region is recorded here. Where the region is, and which
//! of the guest's own accesses to it fault, is recorded in the nested page
//! tables — and the two are tied together by the name those tables hand out for
//! the region. So a device is registered at any time, in any order with respect
//! to the tables being told anything, and neither decision constrains the
//! other.
//!
//! That is not tidiness. There are two ways an access reaches a device and only
//! one of them is a fault. Where this hypervisor serves a region itself, the
//! tables trap it and every access exits. Where the *processor* serves it — the
//! interrupt acceleration driving a controller out of a backing page of its own
//! — the guest's accesses never fault, and the ones the hardware declines to
//! perform come back as an exit that names the address and the direction.
//! Performing one of those is performing it against the same device a fault
//! would have reached. The region has a name in both descriptions, so the
//! device is found the same way whichever brought the access here, and nothing
//! below this module needs to know which did.
//!
//! # There is one record of where a region is, and it is not here
//!
//! [`Mmio::classify`] asks the tables which region an address is in and indexes
//! by the name they answer with. Keeping a second copy of the geometry here
//! would be a second thing to keep in step, and the two would not have to
//! disagree loudly to be a device answering for an address it was never given.
//!
//! What is kept here is a device's aperture, whose length admission is checked
//! against — that is the length of a *mapping* rather than of the region, and
//! it is the only length a load or a store through it may be bounded by.
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

pub(crate) mod decode;
#[cfg(test)]
pub(crate) mod harness;
mod window;

use alloc::{boxed::Box, vec::Vec};

use log::info;
use npt::{MapError, Range, RegionTag};
use paging::{AddressSpace, CacheType, Mapping, PagingError, Protection};
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
    /// `None` is for a region something else already describes: the guest's own
    /// accesses go wherever the tables already send them, and this device is
    /// reached only for the ones that something declines to serve and reports.
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

/// What answers for each region of a guest something other than the hardware
/// answers for.
///
/// One slot per name the nested tables can hand out, indexed by the name, so
/// finding what answers for a region is one bounds-checked load and there is
/// nothing to allocate on any path. The array is what the tables' own bound on
/// a name buys: a name is the lowest one no live region holds, so it is never
/// higher than the number of regions that can hold one at once.
///
/// Answered through a shared reference, so one of these serves every processor
/// running the guest rather than one per processor — which is what the regions
/// themselves are, since a guest physical address means the same thing on all
/// of them.
pub struct Mmio {
    devices: [Option<Interposed>; RegionTag::LIMIT],
}

impl Default for Mmio {
    fn default() -> Self {
        Self::new()
    }
}

impl Mmio {
    /// Nothing answering for anything yet.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            devices: [const { None }; RegionTag::LIMIT],
        }
    }

    /// Records what answers for the region called `tag`, and maps the hardware
    /// behind it where the device reaches it.
    ///
    /// Says nothing to the nested tables. Whether the guest's own accesses to
    /// the region fault is their decision and a separate call; what this
    /// decides is only what answers when one arrives here.
    ///
    /// `gpa` and `bytes` are where the hardware behind the region is, which is
    /// what a mapping of it needs and the only thing they are used for. Where
    /// the *region* is stays the tables' to say.
    ///
    /// # Errors
    ///
    /// [`MmioError::Map`] unless the range is a whole number of pages on a page
    /// boundary, [`MmioError::Capability`] if the device declares one this
    /// crate cannot serve, [`MmioError::Registered`] if something already
    /// answers for that name, [`MmioError::Paging`] if the mapping window
    /// has no room, or [`MmioError::Rollback`] if a mapping made here could
    /// not be removed again after the registration was refused.
    pub fn register(
        &mut self,
        space: &mut AddressSpace,
        tag: RegionTag,
        gpa: PhysAddr,
        bytes: u64,
        device: Box<dyn Device>,
    ) -> Result<(), MmioError> {
        let region = tag.number();
        // One checker decides what a valid range is, and it is the one the tables
        // record the region by — so a range this accepts is one they would have
        // accepted and there is no shape only half the workspace believes in.
        let aperture = Range::new(gpa, bytes)?;
        // A device that answers nothing would trap every access and refuse every
        // one of them, which is a region the guest can never use rather than a
        // device.
        let capability = device.capability();
        if !Width::ALL.iter().any(|width| capability.answers(*width)) {
            return Err(MmioError::Capability { region });
        }
        let Some(slot) = self.devices.get_mut(usize::from(region)) else {
            return Err(MmioError::NoDevice { region });
        };
        if slot.is_some() {
            return Err(MmioError::Registered { region });
        }
        let mapped = match device.hardware() {
            // SAFETY: this is a device aperture rather than memory — the caller is
            // registering it precisely because hardware answers there — so there is
            // nothing for a writable alias to conflict with. The range is a whole
            // number of pages on a page boundary inside the address space the
            // entry format can hold, checked above. Uncached is what a device
            // register needs: a write that sat in a cache line would never reach
            // the bus.
            Hardware::Reached => Aperture::Mapped(unsafe {
                space.map_physical(
                    aperture.base(),
                    aperture.bytes(),
                    Protection::ReadWrite,
                    CacheType::UncachedMinus,
                )
            }?),
            // Nothing to map. The device answers out of its own state and lets
            // nothing through, so a mapping of the registers behind it would be a
            // writable alias of somebody's hardware that no access ever reaches.
            Hardware::Untouched => Aperture::Untouched {
                bytes: aperture.bytes(),
            },
        };
        *slot = Some(Interposed {
            aperture: mapped,
            device,
        });
        Ok(())
    }

    /// Stops answering for the region called `tag`, and answers with what did.
    ///
    /// The counterpart of [`Mmio::register`], and the whole of what giving a
    /// region up costs here: nothing is said to the nested tables, which are
    /// told separately or not at all, and the name is free for whatever the
    /// tables hand it to next.
    ///
    /// The mapping of the hardware behind the region comes back with the device
    /// rather than being removed here, because removing it needs the address
    /// space that made it and taking a device out of this set does not.
    pub fn forget(&mut self, tag: RegionTag) -> Option<Retired> {
        self.devices
            .get_mut(usize::from(tag.number()))
            .and_then(Option::take)
            .map(|region| Retired { tag, region })
    }

    /// Stops answering for every region, and answers with what did.
    ///
    /// The counterpart registration never had. A guest that was built and then
    /// abandoned — because a later step of bring-up failed, or because it is
    /// being taken down — otherwise leaks a window run per region and loses
    /// whatever state its devices hold.
    pub fn teardown(&mut self) -> Vec<Retired> {
        (0..u16::MAX)
            .map(RegionTag::new)
            .take(RegionTag::LIMIT)
            .filter_map(|tag| self.forget(tag))
            .collect()
    }

    /// Whether anything answers for the region called `tag`.
    ///
    /// What an exit path asks before it decodes an instruction to perform
    /// against a device: a region the tables trap with nothing registered for
    /// it is a state the two decisions being independent allows, and
    /// reporting it is better than decoding an instruction to dispatch into
    /// nothing.
    #[must_use]
    pub fn answers(&self, tag: RegionTag) -> bool {
        self.at(tag).is_ok()
    }

    /// Logs which regions are answered for and by what, which is the whole of
    /// what a guest's view of its devices differs by.
    ///
    /// Where each region is belongs to the nested tables and is logged with
    /// them, under the same name.
    pub fn describe(&self, who: &str) {
        let mut answered = false;
        for (region, interposed) in self.devices.iter().enumerate() {
            let Some(interposed) = interposed else {
                continue;
            };
            answered = true;
            match interposed.window().base() {
                Some(base) => {
                    info!(
                        "{who}: region {region} is answered for by a device reached at {base:#x}"
                    );
                }
                None => info!(
                    "{who}: region {region} is answered for by a device that never reaches the \
                     hardware behind it"
                ),
            }
        }
        if !answered {
            info!("{who}: no region of the guest's memory is answered for here");
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
    /// [`EmulateError::NoDevice`] if nothing answers for that region, or
    /// [`EmulateError::Inadmissible`] if the access is not one this device
    /// answers.
    pub(crate) fn admits(
        &self,
        tag: RegionTag,
        offset: u64,
        gpa: PhysAddr,
        width: Width,
    ) -> Result<(), EmulateError> {
        let region = self.at(tag)?;
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

    /// What the guest should see for a read of a region something answers for.
    ///
    /// # Errors
    ///
    /// [`EmulateError::NoDevice`] if nothing answers for that region,
    /// [`EmulateError::Inadmissible`] if the access is not one this device
    /// answers, or if the device answers with a value of a width other than
    /// the one it was asked about.
    pub(crate) fn read(
        &self,
        tag: RegionTag,
        offset: u64,
        gpa: PhysAddr,
        width: Width,
    ) -> Result<Data, EmulateError> {
        let region = self.at(tag)?;
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
        tag: RegionTag,
        offset: u64,
        gpa: PhysAddr,
        value: Data,
    ) -> Result<(), EmulateError> {
        let region = self.at(tag)?;
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

    /// What answers for the region of that name.
    ///
    /// A name higher than a slot and a slot nothing has been registered in are
    /// one answer, because they are the same thing to a caller: nothing answers
    /// for the region it named. That is a state the tables and this set being
    /// independent allows — a region trapped with no device — and it is
    /// reported rather than treated as impossible.
    fn at(&self, tag: RegionTag) -> Result<&Interposed, EmulateError> {
        self.devices
            .get(usize::from(tag.number()))
            .and_then(Option::as_ref)
            .ok_or(EmulateError::NoDevice {
                region: tag.number(),
            })
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
        let instruction = decode::instruction(cpu, guest)?;
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
    /// The names of the regions answered for, not the devices: what answers for
    /// one is a trait object with no more to say about itself than its own
    /// address, and printing that would be noise.
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_list()
            .entries(
                self.devices
                    .iter()
                    .enumerate()
                    .filter_map(|(region, interposed)| interposed.as_ref().map(|_| region)),
            )
            .finish()
    }
}

/// Which region an access falls in, with every byte of it accounted for.
///
/// `gpa` is where the access begins and `width` is how long it is, and both
/// matter: an access that begins in a region and ends outside it belongs to
/// neither, and one that begins in ordinary memory and ends in a region is not
/// ordinary memory.
///
/// Both questions are asked of the nested tables rather than of the set of
/// devices, because the tables hold the one record of where a region is. An
/// access wholly inside one costs a single search of that record; every other
/// shape costs two, which is what establishing that no byte of an access is in
/// a region takes.
///
/// # Errors
///
/// [`EmulateError::Span`] if the access is partly in a region and partly not,
/// or spans two regions.
pub(crate) fn classify(
    guest: &impl Guest,
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
    match guest.region(gpa) {
        // Wholly inside one region, which is the only shape a single device
        // transaction describes.
        Some(region) if region.range.contains(last) => Ok(Place::Device {
            tag: region.tag,
            offset: gpa.as_u64() - region.range.base().as_u64(),
            gpa,
            linear,
        }),
        // It begins in a region and ends somewhere else. Whether that somewhere
        // is another region or ordinary memory changes the diagnostic and
        // nothing else: either way the access is partly a device transaction
        // and partly not, and which bytes go where is not something the
        // instruction says.
        Some(_) => Err(span(if ends_inside(guest, last) {
            Spanning::TwoRegions
        } else {
            Spanning::Straddles
        })),
        // It begins outside every region. If it ends inside one it still
        // straddles; if it does not, nothing interposes on any byte of it and it
        // is the guest's own memory.
        None if ends_inside(guest, last) => Err(span(Spanning::Straddles)),
        None => Ok(Place::Memory(linear)),
    }
}

/// Whether the last byte of an access is inside a region something other than
/// the hardware answers for.
///
/// Takes the address as an integer, because the last byte of an access may be
/// the last byte of the physical address space — which is a number but not a
/// [`PhysAddr`] that can be constructed. One that is not an address is in no
/// region, the tables answering for nothing above the address space.
fn ends_inside(guest: &impl Guest, last: u64) -> bool {
    PhysAddr::try_new(last).is_ok_and(|end| guest.region(end).is_some())
}

/// One region's registration, taken back out of the set.
///
/// Two steps, because they need different things: the set stops naming the
/// region without an address space in hand, and the mapping of the hardware
/// behind it is handed back to the address space that made it.
#[must_use = "the device is returned here, and the mapping behind it is still held"]
pub struct Retired {
    /// Which region it answered for.
    tag: RegionTag,
    /// What answered, and where its registers were reachable.
    region: Interposed,
}

impl Retired {
    /// Which region this answered for.
    #[must_use]
    pub const fn tag(&self) -> RegionTag {
        self.tag
    }

    /// Gives the mapping of the hardware behind the region back, and answers
    /// with the device that was registered for it.
    ///
    /// The device comes back whether or not the mapping could be removed: it
    /// belongs to whoever registered it and may hold state that outlives the
    /// guest, so dropping it because the address space complained would lose
    /// that.
    ///
    /// # Errors
    ///
    /// [`MmioError::Rollback`] if the mapping could not be removed, which
    /// leaves the address space having retired the run rather than handed
    /// it back.
    ///
    /// # Safety
    ///
    /// No processor may be running the guest this region belonged to, and
    /// nothing derived from its window may still be in use.
    pub unsafe fn release(
        self,
        space: &mut AddressSpace,
    ) -> (Box<dyn Device>, Result<(), MmioError>) {
        // SAFETY: forwarded to the caller, whose obligations are exactly what
        // releasing an aperture asks for. The aperture is consumed here, so no
        // further access through it is representable.
        let cause = unsafe { self.region.aperture.release(space) };
        let outcome = cause.map_or(Ok(()), |cause| {
            Err(MmioError::Rollback {
                region: self.tag.number(),
                cause,
            })
        });
        (self.region.device, outcome)
    }
}

/// One region that is answered for, and by what.
///
/// Where the region is is deliberately not here: the nested tables record that,
/// and a second copy of it would be a second thing to keep in step. What the
/// aperture knows is how long the *mapping* of the hardware behind the region
/// is, which is what a load or a store through it has to be bounded by.
struct Interposed {
    aperture: Aperture,
    device: Box<dyn Device>,
}

impl Interposed {
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
    ///
    /// # Safety
    ///
    /// Nothing derived from a window made from this aperture may still be in
    /// use, which means no processor may be running the guest whose region it
    /// belonged to.
    unsafe fn release(self, space: &mut AddressSpace) -> Option<PagingError> {
        match self {
            // SAFETY: the caller guarantees that nothing derived from a window
            // made from this mapping is still in use. The aperture is consumed
            // here, so no further access through it is representable.
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

/// Why a device could not be registered for a region, or its registration given
/// back.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum MmioError {
    /// The device answers no access this crate can make, so every access to its
    /// region would arrive here and then be refused.
    #[error("the device for region {region} answers no access this hypervisor can make")]
    Capability {
        /// Which region it was offered for.
        region: u16,
    },
    /// Something already answers for that region, and which device an access
    /// reached would otherwise depend on the order the two were registered in.
    #[error("a device already answers for region {region}")]
    Registered {
        /// Which region was named.
        region: u16,
    },
    /// Nothing answers for the region named, either because no device was
    /// registered for it or because the name is not one this set has a place
    /// for.
    #[error("no device answers for region {region}")]
    NoDevice {
        /// Which region was named.
        region: u16,
    },
    /// A mapping of the hardware behind a region could not be removed, so the
    /// address space has retired the run rather than handed it back.
    #[error("the window for region {region} could not be removed: {cause}")]
    Rollback {
        /// Which region it belonged to.
        region: u16,
        /// Why the mapping could not be removed.
        cause: PagingError,
    },
    /// The range the hardware behind the region is at is not one the nested
    /// tables would describe.
    #[error(transparent)]
    Map(#[from] MapError),
    /// The device's registers could not be mapped.
    #[error(transparent)]
    Paging(#[from] PagingError),
}

#[cfg(test)]
mod tests {
    use alloc::{boxed::Box, vec::Vec};

    use npt::RegionTag;

    use super::{Capability, Commit, Mmio, Vectors, Widths, classify, harness::Harness};
    use crate::{
        EmulateError, Spanning,
        dispatch::{Answer, Recorder},
        machine::tests::{Machine, Memory},
        value::{Data, Width},
    };

    /// Where the region every test here puts a device at is, in the guest's
    /// physical memory.
    const APERTURE: u64 = 0xFEE0_0000;

    /// How long it is: one page, which is the smallest a region can be.
    const APERTURE_BYTES: u64 = 4096;

    /// The name the tables gave that region.
    const REGION: RegionTag = RegionTag::new(0);

    #[test]
    fn a_device_registered_for_a_region_the_tables_do_not_hold_is_never_reached() {
        // What answers for the region, with nothing having told the tables where
        // the region is — which is what registering a device without trapping
        // anything leaves behind.
        let (mmio, _) = Harness::new()
            .region(REGION, APERTURE, APERTURE_BYTES, Box::new(Recorder::new()))
            .seal();
        let machine = Machine::long_mode();
        let untouched = Memory::new(&machine);

        assert_eq!(
            classify(
                &untouched,
                x86_64::PhysAddr::new(APERTURE),
                Width::Long,
                0x8000
            ),
            Ok(crate::operand::Place::Memory(0x8000)),
            "with no region recorded the address is the guest's own memory, which \
             is exactly what leaving the tables alone means"
        );
        assert!(
            mmio.answers(REGION),
            "while the device is registered all the same, waiting for whichever way \
             an access to the region arrives"
        );
    }

    #[test]
    fn a_region_the_tables_hold_with_no_device_is_reported_rather_than_dispatched_into() {
        // The other direction: the tables know where the region is and nothing was
        // ever registered for it.
        let (_, regions) = Harness::new()
            .region(REGION, APERTURE, APERTURE_BYTES, Box::new(Recorder::new()))
            .seal();
        let machine = Machine::long_mode();
        let mut guest = Memory::new(&machine);
        guest.describing(regions);
        let nothing = Mmio::new();

        let place = classify(
            &guest,
            x86_64::PhysAddr::new(APERTURE + 8),
            Width::Long,
            0x8000,
        )
        .expect("the address is in a region the tables hold");

        assert_eq!(
            place,
            crate::operand::Place::Device {
                tag: REGION,
                offset: 8,
                gpa: x86_64::PhysAddr::new(APERTURE + 8),
                linear: 0x8000,
            },
            "the region answers with the name the tables gave it, whether or not \
             anything has been registered under that name"
        );
        assert!(!nothing.answers(REGION));
        assert_eq!(
            nothing.admits(REGION, 8, x86_64::PhysAddr::new(APERTURE + 8), Width::Long),
            Err(EmulateError::NoDevice { region: 0 }),
            "and an access to it is reported rather than performed against nothing"
        );
    }

    #[test]
    fn an_access_that_leaves_a_region_belongs_to_neither_side_of_the_edge() {
        let (_, regions) = Harness::new()
            .region(REGION, APERTURE, APERTURE_BYTES, Box::new(Recorder::new()))
            .seal();
        let machine = Machine::long_mode();
        let mut guest = Memory::new(&machine);
        guest.describing(regions);

        for (gpa, reason) in [
            (APERTURE + APERTURE_BYTES - 2, Spanning::Straddles),
            (APERTURE - 2, Spanning::Straddles),
        ] {
            assert_eq!(
                classify(&guest, x86_64::PhysAddr::new(gpa), Width::Long, 0x8000),
                Err(EmulateError::Span {
                    linear: 0x8000,
                    bytes: Width::Long.bytes(),
                    reason,
                }),
                "an access at {gpa:#x} is partly a device transaction and partly not"
            );
        }
    }

    #[test]
    fn a_device_is_found_by_name_across_a_removal_and_a_re_registration() {
        let (mut mmio, _) = Harness::new()
            .region(
                REGION,
                APERTURE,
                APERTURE_BYTES,
                Box::new(Recorder::new().reads(Answer::Invented(0x1111_1111))),
            )
            .seal();
        assert_eq!(
            read(&mmio),
            Data::from_u64(0x1111_1111, Width::Long),
            "the device registered under the name is the one an access reaches"
        );

        let retired = mmio.forget(REGION).expect("something answered for it");
        assert_eq!(retired.tag(), REGION);
        assert!(
            !mmio.answers(REGION),
            "after which the name answers for nothing"
        );

        let (again, _) = Harness::new()
            .region(
                REGION,
                APERTURE,
                APERTURE_BYTES,
                Box::new(Recorder::new().reads(Answer::Invented(0x2222_2222))),
            )
            .seal();
        assert_eq!(
            read(&again),
            Data::from_u64(0x2222_2222, Width::Long),
            "and a name handed out again reaches whatever was registered for it \
             this time, never what answered for it before"
        );
    }

    #[test]
    fn taking_the_set_apart_hands_every_device_back_and_leaves_no_region_named() {
        let second = RegionTag::new(1);
        let (mut mmio, _) = Harness::new()
            .region(REGION, APERTURE, APERTURE_BYTES, Box::new(Recorder::new()))
            .region(
                second,
                APERTURE + APERTURE_BYTES,
                APERTURE_BYTES,
                Box::new(Recorder::new().untouched()),
            )
            .seal();

        let retired = mmio.teardown();

        assert_eq!(
            retired.iter().map(super::Retired::tag).collect::<Vec<_>>(),
            [REGION, second],
            "every region that was answered for comes back, in the order the names \
             run"
        );
        assert!(!mmio.answers(REGION) && !mmio.answers(second));
        assert!(
            mmio.teardown().is_empty(),
            "and there is nothing left to take apart"
        );
    }

    /// What the device answering the region says for an aligned four-byte read
    /// at its base.
    fn read(mmio: &Mmio) -> Data {
        mmio.read(REGION, 0, x86_64::PhysAddr::new(APERTURE), Width::Long)
            .expect("the device answers an aligned four-byte read")
    }

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
