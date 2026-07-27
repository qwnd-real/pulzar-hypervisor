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
//! with a `register`, and the only way to get an [`Mmio`] — the thing an exit
//! handler holds — is to consume the registrar with [`Registrar::seal`]. The
//! precondition is therefore a property of the types: by the time a guest can
//! run, there is no longer anything in existence that could trap a region.
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

mod window;

use alloc::{boxed::Box, vec::Vec};

use log::{info, warn};
use npt::{Npt, NptError};
use paging::{AddressSpace, CacheType, Mapping, PagingError, Protection, chunk::FRAME_SIZE};
use thiserror::Error;
use x86_64::PhysAddr;

use crate::{
    EmulateError,
    mmio::window::Window,
    value::{Data, Width},
};

/// What answers for a region instead of the hardware behind it.
pub trait Device {
    /// What the guest should see.
    ///
    /// Call [`Read::hardware`] for what the device really holds, or do not, and
    /// answer out of whatever state this device keeps.
    fn read(&mut self, access: Read) -> Data;

    /// What should become of what the guest wrote.
    fn write(&mut self, access: Write) -> Commit;
}

/// A region to be answered for, and by what.
pub struct Region {
    /// Where the region begins in the guest's physical memory, which is also
    /// where the hardware behind it is. Page aligned.
    pub gpa: PhysAddr,
    /// How long it is. A whole number of pages, because permissions come a page
    /// at a time.
    pub bytes: u64,
    /// Which of the guest's accesses have to come back to us.
    pub trap: Trap,
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
    Replace(Data),
    /// The device never sees it.
    Discard,
}

/// A read the guest made of a device.
#[derive(Clone, Copy, Debug)]
pub struct Read {
    window: Window,
    offset: u64,
    gpa: PhysAddr,
    width: Width,
}

impl Read {
    /// How far into the region the guest read.
    #[must_use]
    pub const fn offset(&self) -> u64 {
        self.offset
    }

    /// Where the guest read, as it thinks of the address.
    #[must_use]
    pub const fn gpa(&self) -> PhysAddr {
        self.gpa
    }

    /// How much the guest read.
    #[must_use]
    pub const fn width(&self) -> Width {
        self.width
    }

    /// What the device really holds there, read now.
    ///
    /// The access is made when this is called and not before, so a handler that
    /// does not need it does not make it. Calling it twice makes two device
    /// reads, which for a register that changes as it is read is two different
    /// answers — that is the device's behaviour, faithfully, and not something
    /// to hide behind a cache.
    #[must_use]
    pub fn hardware(&self) -> Data {
        self.window.read(self.offset, self.width)
    }
}

/// A write the guest made to a device.
#[derive(Clone, Copy, Debug)]
pub struct Write {
    window: Window,
    offset: u64,
    gpa: PhysAddr,
    value: Data,
}

impl Write {
    /// How far into the region the guest wrote.
    #[must_use]
    pub const fn offset(&self) -> u64 {
        self.offset
    }

    /// Where the guest wrote, as it thinks of the address.
    #[must_use]
    pub const fn gpa(&self) -> PhysAddr {
        self.gpa
    }

    /// How much the guest wrote.
    #[must_use]
    pub const fn width(&self) -> Width {
        self.value.width()
    }

    /// What the guest tried to write.
    #[must_use]
    pub const fn value(&self) -> Data {
        self.value
    }

    /// What the device holds there now, read before anything is written.
    ///
    /// For the registers where what the guest wrote is only part of the answer:
    /// a bit to set in a field that must otherwise be left alone, or a
    /// write-to-clear register whose other bits must survive.
    #[must_use]
    pub fn hardware(&self) -> Data {
        self.window.read(self.offset, self.value.width())
    }
}

/// Trapping regions, before the guest runs.
#[derive(Default)]
pub struct Registrar {
    regions: Vec<Interposed>,
}

impl Registrar {
    /// Nothing trapped yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Takes over a region of the guest's physical memory.
    ///
    /// Three things happen and the order matters. The device's registers are
    /// mapped first, once and for good, because a region that is trapped
    /// without somewhere to reach the hardware is a region whose every
    /// access fails. Then the nested tables are told to trap it. Only then
    /// is the device remembered, so that a registrar never holds a region
    /// it did not finish taking over.
    ///
    /// # Errors
    ///
    /// [`MmioError::Geometry`] unless the region is a whole number of pages on
    /// a page boundary, [`MmioError::Overlaps`] if another region already
    /// covers part of it, [`MmioError::Paging`] if the mapping window has
    /// no room, or [`MmioError::Npt`] if the nested tables cannot describe
    /// it a page at a time.
    pub fn register(
        &mut self,
        space: &mut AddressSpace,
        npt: &mut Npt,
        region: Region,
    ) -> Result<(), MmioError> {
        let (gpa, bytes) = (region.gpa, region.bytes);
        if bytes == 0
            || !gpa.as_u64().is_multiple_of(FRAME_SIZE)
            || !bytes.is_multiple_of(FRAME_SIZE)
        {
            return Err(MmioError::Geometry {
                gpa: gpa.as_u64(),
                bytes,
            });
        }
        if self.regions.iter().any(|other| other.overlaps(gpa, bytes)) {
            return Err(MmioError::Overlaps { gpa: gpa.as_u64() });
        }

        // SAFETY: this is a device aperture rather than memory — the caller is
        // registering it precisely because hardware answers there — so there is
        // nothing for a writable alias to conflict with. Uncached is what a
        // device register needs: a write that sat in a cache line would never
        // reach the bus.
        let mapping = unsafe {
            space.map_physical(gpa, bytes, Protection::ReadWrite, CacheType::UncachedMinus)
        }?;
        if let Err(error) = npt.protect(space.frames(), gpa, bytes, region.trap) {
            // SAFETY: the mapping was made two statements ago, nothing has been
            // handed its address, and the region is not in the list — so nothing
            // derived from it exists anywhere.
            if let Err(unmapping) = unsafe { space.unmap(mapping) } {
                warn!(
                    "emulate: window for the untrapped region at {gpa:#x} stays mapped: {unmapping}"
                );
            }
            return Err(error.into());
        }

        self.regions.push(Interposed {
            gpa,
            mapping,
            device: region.device,
        });
        Ok(())
    }

    /// Closes the set of trapped regions, which is what makes it usable.
    ///
    /// After this there is no way to trap another, which is the point: every
    /// region a guest could reach is now described, and no cached translation
    /// anywhere can disagree with the tables.
    #[must_use]
    pub fn seal(self) -> Mmio {
        Mmio {
            regions: self.regions,
        }
    }
}

/// The trapped regions of a guest that is allowed to run.
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
            info!(
                "{who}: interposing on guest physical {:#x}..{:#x}, reached at {:#x}",
                region.gpa,
                region.gpa + region.mapping.bytes(),
                region.mapping.addr(),
            );
        }
    }

    /// Which region covers a guest physical address, and how far into it the
    /// address falls.
    pub(crate) fn find(&self, gpa: PhysAddr) -> Option<(usize, u64)> {
        self.regions
            .iter()
            .position(|region| region.holds(gpa))
            .map(|index| (index, gpa - self.regions[index].gpa))
    }

    /// What the guest should see for a read of a trapped region.
    ///
    /// # Errors
    ///
    /// [`EmulateError::Inadmissible`] if the access runs past the region it
    /// began in, or is not aligned to its own width — neither of which is
    /// something a device would ever be asked by real hardware.
    pub(crate) fn read(
        &mut self,
        index: usize,
        offset: u64,
        gpa: PhysAddr,
        width: Width,
    ) -> Result<Data, EmulateError> {
        let region = self.at(index)?;
        let window = region.window();
        if !window.admits(offset, width) {
            return Err(EmulateError::Inadmissible {
                gpa: gpa.as_u64(),
                bytes: width.bytes(),
            });
        }
        Ok(region.device.read(Read {
            window,
            offset,
            gpa,
            width,
        }))
    }

    /// Performs what a device decided should become of a write.
    ///
    /// # Errors
    ///
    /// As [`Mmio::read`].
    pub(crate) fn write(
        &mut self,
        index: usize,
        offset: u64,
        gpa: PhysAddr,
        value: Data,
    ) -> Result<(), EmulateError> {
        let region = self.at(index)?;
        let window = region.window();
        if !window.admits(offset, value.width()) {
            return Err(EmulateError::Inadmissible {
                gpa: gpa.as_u64(),
                bytes: value.width().bytes(),
            });
        }
        let decision = region.device.write(Write {
            window,
            offset,
            gpa,
            value,
        });
        match decision {
            Commit::Hardware => window.write(offset, value),
            Commit::Replace(instead) => window.write(offset, instead),
            Commit::Discard => {}
        }
        Ok(())
    }

    /// One region, by the index [`Mmio::find`] gave.
    fn at(&mut self, index: usize) -> Result<&mut Interposed, EmulateError> {
        self.regions
            .get_mut(index)
            .ok_or(EmulateError::NoSuchRegion { index })
    }
}

/// One region that is answered for, and by what.
struct Interposed {
    gpa: PhysAddr,
    mapping: Mapping,
    device: Box<dyn Device>,
}

impl Interposed {
    /// Whether this region covers that address.
    fn holds(&self, gpa: PhysAddr) -> bool {
        (self.gpa.as_u64()..self.gpa.as_u64() + self.mapping.bytes()).contains(&gpa.as_u64())
    }

    /// Whether this region covers any of `bytes` from `gpa`.
    fn overlaps(&self, gpa: PhysAddr, bytes: u64) -> bool {
        let (mine, theirs) = (self.gpa.as_u64(), gpa.as_u64());
        theirs < mine + self.mapping.bytes() && mine < theirs.saturating_add(bytes)
    }

    /// Where this device's registers are reachable.
    fn window(&self) -> Window {
        Window::new(self.mapping.addr(), self.mapping.bytes())
    }
}

/// Why a region could not be taken over.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum MmioError {
    /// The region is not a whole number of pages on a page boundary, which is
    /// the granularity the nested tables can give it permissions of its own at.
    #[error("a region at {gpa:#x} of {bytes:#x} bytes is not a whole number of pages")]
    Geometry {
        /// Where the region begins.
        gpa: u64,
        /// How long it is.
        bytes: u64,
    },
    /// Another region already covers part of this one, so which device answers
    /// for the overlap would depend on the order they were registered in.
    #[error("a region already covers part of {gpa:#x}")]
    Overlaps {
        /// Where the region begins.
        gpa: u64,
    },
    /// The nested tables could not describe the region a page at a time.
    #[error(transparent)]
    Npt(#[from] NptError),
    /// The device's registers could not be mapped.
    #[error(transparent)]
    Paging(#[from] PagingError),
}
