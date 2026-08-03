//! A set of trapped regions built without a machine underneath it.
//!
//! Registration needs an address space and nested page tables, which exist only
//! on a booted machine — so [`Registrar`](super::Registrar) cannot be reached
//! on the host, and neither could anything downstream of it. This builds the
//! same [`Mmio`] the registrar builds, over bytes standing in for a device
//! aperture, so that everything above registration is exercised by the tests:
//! admission, the capability contract, commit binding, span classification, and
//! the whole of dispatch.
//!
//! What is *not* stubbed is the interesting part. The window is a real
//! [`Window`](super::window::Window) over real bytes reached through a raw
//! pointer, the volatile accesses are the same volatile accesses, and the
//! admission and binding checks are the production ones. The only difference is
//! where the bytes came from.

use alloc::{boxed::Box, vec, vec::Vec};

use x86_64::{PhysAddr, VirtAddr};

use super::{Aperture, Device, Interposed, Mmio};

/// Builds a set of regions over planted bytes.
#[derive(Default)]
pub(crate) struct Harness {
    regions: Vec<Interposed>,
}

impl Harness {
    /// Nothing trapped yet.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Adds a region at this guest physical address, that many bytes long,
    /// answered for by that device.
    ///
    /// The bytes behind it start as zeroes and are reached exactly as a real
    /// aperture is.
    pub(crate) fn region(mut self, gpa: u64, bytes: u64, device: Box<dyn Device>) -> Self {
        let planted = plant(bytes);
        self.regions.push(Interposed {
            gpa: PhysAddr::new(gpa),
            end: gpa + bytes,
            aperture: planted,
            device,
        });
        self
    }

    /// The sealed set, as an exit handler would hold it.
    pub(crate) fn seal(self) -> Mmio {
        Mmio {
            regions: self.regions,
        }
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
