//! One device's registers, mapped once and kept.
//!
//! This is the whole answer to a question that otherwise has no good one: how
//! does the hypervisor touch a device's registers on an intercepted access
//! without mapping something first?
//!
//! It cannot use the direct map. That window covers memory and stops at the
//! last byte of it, so a device aperture sitting far above physical memory is
//! not in it at all — and even where one is, the direct map is write-back
//! everywhere, which is the wrong memory type for a device register and would
//! leave writes sitting in a cache line instead of on a bus.
//!
//! And it must not map and unmap per access. A mapping costs page tables and an
//! unmapping costs an interprocessor interrupt to every processor that might
//! hold the translation, which on the path taken once per intercepted register
//! access is not a cost to pay at all.
//!
//! So each region is mapped once, when it is registered, uncached and writable,
//! and the mapping is held for as long as the region is. An intercepted access
//! is then an offset and a volatile load or store, with nothing allocated and
//! nothing invalidated.

use x86_64::VirtAddr;

use crate::{
    value::{Data, Width},
    xmm,
};

/// Where a device's registers are readable, for as long as the region exists.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Window {
    base: VirtAddr,
    bytes: u64,
}

impl Window {
    /// A device's registers at this address, that many bytes of them.
    pub(crate) const fn new(base: VirtAddr, bytes: u64) -> Self {
        Self { base, bytes }
    }

    /// Whether an access of this width at this offset is one this can make.
    ///
    /// Asked before a handler is ever called, so that everything below it is
    /// spared the question, and it asks two things.
    ///
    /// Whether the access is inside the region: one that begins inside it can
    /// still end past it — a four-byte read two bytes from the end does — and
    /// what a device would make of the half that is somebody else's registers
    /// is not something to establish by trying it.
    ///
    /// And whether it is aligned to its own width, which every device register
    /// is and which a scalar access here has to be: the loads and stores below
    /// go through a pointer of the width being moved, and one of those pointing
    /// at an address that is not a multiple of its width is not a slow access
    /// but an invalid one. The sixteen-byte case is exempt because it does not
    /// go through a pointer at all — the instruction it uses is the one the
    /// architecture defines for unaligned vectors.
    pub(crate) fn admits(&self, offset: u64, width: Width) -> bool {
        let Ok(bytes) = u64::try_from(width.bytes()) else {
            return false;
        };
        let inside = offset
            .checked_add(bytes)
            .is_some_and(|end| end <= self.bytes);
        inside && (width == Width::Vector || offset.is_multiple_of(bytes))
    }

    /// Reads the device.
    ///
    /// The access is made at the width the guest used, because that is the
    /// transaction the guest asked for: a device is entitled to answer a
    /// four-byte read differently from four one-byte reads, and several do.
    pub(crate) fn read(&self, offset: u64, width: Width) -> Data {
        match width {
            Width::Byte => Data::from_u64(u64::from(self.load::<u8>(offset)), width),
            Width::Word => Data::from_u64(u64::from(self.load::<u16>(offset)), width),
            Width::Long => Data::from_u64(u64::from(self.load::<u32>(offset)), width),
            Width::Quad => Data::from_u64(self.load::<u64>(offset), width),
            // Sixteen bytes go through the instruction the architecture defines
            // for unaligned vector moves rather than through a pointer, and have
            // to: a pair of quadword reads would be two bus transactions where
            // the guest made one, and a device is entitled to tell the
            // difference.
            //
            // SAFETY: `admits` established that the whole access lies inside a
            // mapping this region has held since it was registered, and that
            // instruction requires no alignment.
            Width::Vector => Data::vector_from(unsafe { xmm::read_device(self.at(offset)) }),
        }
    }

    /// Writes the device, at the width the guest used and for the same reason.
    pub(crate) fn write(&self, offset: u64, value: Data) {
        match value.width() {
            Width::Byte => self.store::<u8>(offset, &value),
            Width::Word => self.store::<u16>(offset, &value),
            Width::Long => self.store::<u32>(offset, &value),
            Width::Quad => self.store::<u64>(offset, &value),
            // SAFETY: as in `read`, with the transfer the other way round.
            Width::Vector => unsafe {
                xmm::write_device(self.at(offset).cast_mut(), value.vector());
            },
        }
    }

    /// One device register, read as the integer it is.
    ///
    /// The pointer is formed at `T` rather than cast to it from a byte pointer,
    /// which is both what makes the alignment argument below hold and what
    /// keeps it from being a claim about a cast the compiler cannot check.
    fn load<T: Copy>(&self, offset: u64) -> T {
        // SAFETY: `admits` was checked before the handler that led here was
        // called, so the whole access lies inside a mapping this region has held
        // since it was registered, and the address is a multiple of the access
        // width — the region's base is page aligned and the offset was checked —
        // so it is aligned for `T`, whose size is that width. The read is
        // volatile because a device register is not memory: its value is not a
        // function of what was last written there, and the read itself is what
        // the device reacts to.
        unsafe { self.register::<T>(offset).read_volatile() }
    }

    /// One device register, written as the integer it is.
    fn store<T: TryFrom<u64> + Default>(&self, offset: u64, value: &Data) {
        // SAFETY: as in `load`. The write is volatile because it is the point: a
        // device register's whole purpose is that storing to it does something,
        // so the store must happen, must happen once, and must happen here
        // rather than being folded into a later one.
        unsafe { self.register::<T>(offset).write_volatile(narrow(value)) };
    }

    /// Where one device register of type `T` is.
    fn register<T>(&self, offset: u64) -> *mut T {
        (self.base + offset).as_mut_ptr::<T>()
    }

    /// Where an offset within the region is, as bytes.
    fn at(&self, offset: u64) -> *const u8 {
        (self.base + offset).as_ptr()
    }
}

/// The low bytes of a value, as whatever integer it is being stored as.
///
/// Narrowing is the point rather than a hazard: the value was built at the
/// width it is being stored at, so every bit this drops is already zero — which
/// is also why the conversion cannot fail, and why a fallback of zero is a
/// value nothing reaches rather than a guess at what was meant.
fn narrow<T: TryFrom<u64> + Default>(value: &Data) -> T {
    T::try_from(value.as_u64() & value.width().mask()).unwrap_or_default()
}
