//! A set of regions and a guest's memory that agree about where each region is,
//! built without a machine underneath either.
//!
//! Registering a device needs an address space to map its aperture through, and
//! recording where a region is needs nested page tables; both exist only on a
//! booted machine. This plants a device in an [`Mmio`] over bytes standing in
//! for an aperture, and hands back the regions to tell a test guest's memory
//! about, so that everything above registration is exercised: admission, the
//! capability contract, commit binding, span classification, and the whole of
//! dispatch.
//!
//! What is *not* stubbed is the interesting part. The window is a real
//! [`Window`](super::window::Window) over real bytes reached through a raw
//! pointer, the volatile accesses are the same volatile accesses, and the
//! admission and binding checks are the production ones. The only difference is
//! where the bytes came from and who was told where the region is.

use alloc::{boxed::Box, vec, vec::Vec};

use npt::{Answered, Range, RegionTag};
use x86_64::{PhysAddr, VirtAddr};

use super::{Aperture, Device, Hardware, Interposed, Mmio};

/// Builds a set of regions over planted bytes.
pub(crate) struct Harness {
    devices: [Option<Interposed>; RegionTag::LIMIT],
    regions: Vec<Answered>,
}

impl Harness {
    /// Nothing answered for yet.
    pub(crate) fn new() -> Self {
        Self {
            devices: [const { None }; RegionTag::LIMIT],
            regions: Vec::new(),
        }
    }

    /// Adds a region at this guest physical address, that many bytes long,
    /// answered for by that device, under that name.
    ///
    /// The bytes behind it start as zeroes and are reached exactly as a real
    /// aperture is — and, as at registration, a device that declares it never
    /// reaches the hardware behind its region gets none of them.
    pub(crate) fn region(
        mut self,
        tag: RegionTag,
        gpa: u64,
        bytes: u64,
        device: Box<dyn Device>,
    ) -> Self {
        let aperture = match device.hardware() {
            Hardware::Reached => plant(bytes),
            Hardware::Untouched => Aperture::Untouched { bytes },
        };
        self.devices[usize::from(tag.number())] = Some(Interposed { aperture, device });
        self.regions.push(Answered {
            tag,
            range: Range::new(PhysAddr::new(gpa), bytes).expect("a region of whole pages"),
        });
        self
    }

    /// The set of devices and the regions a guest's memory has to agree about,
    /// as the nested tables and this set would.
    pub(crate) fn seal(self) -> (Mmio, Vec<Answered>) {
        (
            Mmio {
                devices: self.devices,
            },
            self.regions,
        )
    }
}

/// Bytes standing in for a device's registers, live for the rest of the
/// process.
///
/// Leaked deliberately. A window reaches its aperture through a raw pointer
/// with no lifetime attached, because a real aperture is memory Rust never
/// owned; a buffer that could be dropped while a window still named it would be
/// a difference between the test and the machine in exactly the direction that
/// hides mistakes.
fn plant(bytes: u64) -> Aperture {
    let length = usize::try_from(bytes).expect("a test aperture fits a host pointer");
    let planted = Box::leak(vec![0_u8; length].into_boxed_slice());
    Aperture::Planted {
        base: VirtAddr::from_ptr(planted.as_ptr()),
        bytes,
    }
}
