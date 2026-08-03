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
//!
//! # What an access to hardware is allowed to assume
//!
//! Rust's rules for a volatile access outside any allocation require the access
//! not to trap. Nothing about a mapping proves that: a reserved register, a
//! device that has been removed, a width the device does not decode — any of
//! them can produce a machine-check or an abort that this hypervisor has no
//! handler for and could not resume from if it had.
//!
//! That is not something to discover by trying it, and it is not something a
//! bounds check answers. It is a property of the device, so it is stated by
//! whoever registers the device, in [`Capability`](crate::mmio::Capability),
//! and [`Admitted`] is the proof that a particular access has been checked
//! against it. A [`Window`] cannot be reached except through one.

use x86_64::VirtAddr;

use crate::{
    Inadmissible,
    value::{Data, Width},
};

/// Where a device's registers are readable, for as long as the region exists.
///
/// Deliberately not `Copy` and never handed out by value: a handler that could
/// keep one could reach the hardware after the access it was called for had
/// finished, from another processor, with no admission proof and no guarantee
/// the mapping still exists. Every path to one borrows it for the length of a
/// single callback.
#[derive(Debug)]
pub(crate) struct Window {
    base: VirtAddr,
    bytes: u64,
}

impl Window {
    /// A device's registers at this address, that many bytes of them.
    ///
    /// # Safety
    ///
    /// `base` must be the address of a live mapping of exactly `bytes` bytes,
    /// mapped writable and uncached, which outlives this value — and the
    /// physical memory behind it must be a device aperture rather than
    /// anything the hypervisor holds a reference to. Nothing about the type
    /// enforces this, which is why constructing one is unsafe and every
    /// access through it is checked.
    pub(crate) const unsafe fn new(base: VirtAddr, bytes: u64) -> Self {
        Self { base, bytes }
    }

    /// Where the region begins, which is only ever printed.
    pub(crate) const fn base(&self) -> VirtAddr {
        self.base
    }

    /// Whether an access of this width at this offset lies inside the region.
    ///
    /// One that begins inside it can still end past it — a four-byte read two
    /// bytes from the end does — and what a device would make of the half that
    /// is somebody else's registers is not something to establish by trying
    /// it.
    pub(crate) fn inside(&self, offset: u64, width: Width) -> bool {
        offset
            .checked_add(width.span())
            .is_some_and(|end| end <= self.bytes)
    }

    /// Reads the device.
    ///
    /// The access is made at the width the guest used, because that is the
    /// transaction the guest asked for: a device is entitled to answer a
    /// four-byte read differently from four one-byte reads, and several do.
    pub(crate) fn read(&self, admitted: &Admitted) -> Data {
        let offset = admitted.offset;
        match admitted.width {
            Width::Byte => Data::from_u64(u64::from(self.load::<u8>(offset)), Width::Byte),
            Width::Word => Data::from_u64(u64::from(self.load::<u16>(offset)), Width::Word),
            Width::Long => Data::from_u64(u64::from(self.load::<u32>(offset)), Width::Long),
            Width::Quad => Data::from_u64(self.load::<u64>(offset), Width::Quad),
            // Two quadwords, in ascending order. This is two bus transactions
            // where the guest made one, which is why a device that cannot
            // tolerate that is refused at registration rather than served here:
            // making it one transaction would mean borrowing a vector register
            // across a faultable access, and there is nowhere to put the guest's
            // value while it is borrowed.
            Width::Vector => {
                let mut bytes = [0; Width::Vector.bytes()];
                let (low, high) = bytes.split_at_mut(Width::Quad.bytes());
                low.copy_from_slice(&self.load::<u64>(offset).to_le_bytes());
                high.copy_from_slice(&self.load::<u64>(offset + Width::Quad.span()).to_le_bytes());
                Data::vector_from(bytes)
            }
        }
    }

    /// Writes the device, at the width the guest used and for the same reason.
    ///
    /// The value's width is not consulted: the admitted width is what decides
    /// the transaction, and [`Admitted::binds`] has already established
    /// that the value agrees with it. A device handler that returned a
    /// replacement of another width cannot reach this.
    pub(crate) fn write(&self, admitted: &Admitted, value: &Data) {
        let offset = admitted.offset;
        match admitted.width {
            Width::Byte => self.store::<u8>(offset, narrow(value)),
            Width::Word => self.store::<u16>(offset, narrow(value)),
            Width::Long => self.store::<u32>(offset, narrow(value)),
            Width::Quad => self.store::<u64>(offset, value.as_u64()),
            // As in `read`, and in the same order: low quadword first, so a
            // device whose registers are a command pair sees them written the way
            // the guest wrote them.
            Width::Vector => {
                let bytes = value.vector();
                let (low, high) = bytes.split_at(Width::Quad.bytes());
                self.store::<u64>(offset, quad(low));
                self.store::<u64>(offset + Width::Quad.span(), quad(high));
            }
        }
    }

    /// One device register, read as the integer it is.
    ///
    /// The pointer is formed at `T` rather than cast to it from a byte pointer,
    /// which is both what makes the alignment argument below hold and what
    /// keeps it from being a claim about a cast the compiler cannot check.
    fn load<T: Copy>(&self, offset: u64) -> T {
        // SAFETY: an `Admitted` is the proof that this access lies inside a
        // mapping this region has held since it was registered, is aligned to its
        // own width — so to `T`, whose size is that width — and is one the device
        // behind the mapping tolerates at this width and direction. The read is
        // volatile because a device register is not memory: its value is not a
        // function of what was last written there, and the read itself is what the
        // device reacts to.
        unsafe { self.register::<T>(offset).read_volatile() }
    }

    /// One device register, written as the integer it is.
    fn store<T>(&self, offset: u64, value: T) {
        // SAFETY: as in `load`. The write is volatile because it is the point: a
        // device register's whole purpose is that storing to it does something, so
        // the store must happen, must happen once, and must happen here rather
        // than being folded into a later one.
        unsafe { self.register::<T>(offset).write_volatile(value) };
    }

    /// Where one device register of type `T` is.
    fn register<T>(&self, offset: u64) -> *mut T {
        (self.base + offset).as_mut_ptr::<T>()
    }
}

/// An access that has been checked against the region it is in and against what
/// the device behind it answers.
///
/// The only way to reach a [`Window`], and it is not constructible outside this
/// module — [`Capability::admit`](crate::mmio::Capability::admit) is the one
/// thing that makes one. That is what keeps the check and the access from
/// drifting apart: there is no path to the hardware that does not carry the
/// proof, and the proof names the exact offset and width the access was checked
/// for.
#[derive(Debug)]
pub(crate) struct Admitted {
    offset: u64,
    width: Width,
}

impl Admitted {
    /// Records that an access of this width at this offset has been checked.
    ///
    /// Private on purpose: only the capability check may make one.
    pub(super) const fn new(offset: u64, width: Width) -> Self {
        Self { offset, width }
    }

    /// How wide the access was admitted to be.
    pub(crate) const fn width(&self) -> Width {
        self.width
    }

    /// How far into the region it was admitted at.
    pub(crate) const fn offset(&self) -> u64 {
        self.offset
    }

    /// This same proof, for a value that must be exactly as wide as what was
    /// admitted.
    ///
    /// What a device's replacement value goes through. Admission checked the
    /// width the *guest* used; a handler answering with another width would be
    /// a different transaction — possibly past the end of the region,
    /// possibly misaligned, certainly not the access the device was asked
    /// about — so the two are bound together here rather than hoped to
    /// agree.
    ///
    /// # Errors
    ///
    /// [`Inadmissible::Answer`] if the value is not the admitted width.
    pub(crate) fn binds(&self, value: &Data) -> Result<(), Inadmissible> {
        if value.width() == self.width {
            return Ok(());
        }
        Err(Inadmissible::Answer {
            wanted: self.width,
            got: value.width(),
        })
    }
}

/// The low bytes of a value, as whatever integer it is being stored as.
///
/// Narrowing is the point rather than a hazard: the value was built at the
/// width it is being stored at, so every bit this drops is already zero.
fn narrow<T: TryFrom<u64>>(value: &Data) -> T
where
    <T as TryFrom<u64>>::Error: core::fmt::Debug,
{
    T::try_from(value.as_u64() & value.width().mask())
        .expect("a value built at this width has nothing above it")
}

/// Eight bytes of a vector as the quadword they are.
fn quad(bytes: &[u8]) -> u64 {
    let mut value = [0; Width::Quad.bytes()];
    value.copy_from_slice(bytes);
    u64::from_le_bytes(value)
}

#[cfg(test)]
mod tests {
    use super::{Admitted, Window};
    use crate::{
        Inadmissible,
        value::{Data, Width},
    };

    /// A window over an address that is never dereferenced, for the checks that
    /// are pure arithmetic.
    ///
    /// Nothing in these tests reads or writes through it: what is being tested
    /// is which accesses are allowed to reach hardware, and reaching
    /// hardware is exactly what must not happen on a host.
    fn window(bytes: u64) -> Window {
        // SAFETY: the address is never dereferenced. Every test below asks only
        // whether an access would be admitted, which is arithmetic on the base
        // and the length.
        unsafe { Window::new(x86_64::VirtAddr::new(0x1000), bytes) }
    }

    #[test]
    fn an_access_that_ends_past_the_region_is_outside_it() {
        let window = window(4096);
        for width in Width::ALL {
            // The last access of this width that fits, and the first that does
            // not.
            let last = 4096 - width.span();
            assert!(
                window.inside(last, width),
                "{width:?} at {last:#x} is the last one that fits"
            );
            assert!(
                !window.inside(last + 1, width),
                "{width:?} at {:#x} ends one byte past the region",
                last + 1
            );
            assert!(window.inside(0, width), "{width:?} at zero fits");
        }
    }

    #[test]
    fn an_offset_at_or_past_the_end_is_outside_however_narrow_the_access() {
        let window = window(4096);
        for offset in [4096, 4097, u64::MAX / 2] {
            assert!(!window.inside(offset, Width::Byte));
        }
    }

    #[test]
    fn an_offset_whose_span_overflows_is_outside_rather_than_wrapping() {
        // The arithmetic must not wrap into a small number and pass the bounds
        // check, which is what an unchecked add would do here.
        let window = window(4096);
        for width in Width::ALL {
            assert!(
                !window.inside(u64::MAX - 1, width),
                "{width:?} near the top of the address space must not wrap into range"
            );
            assert!(!window.inside(u64::MAX, width));
        }
    }

    #[test]
    fn a_longer_region_admits_what_a_shorter_one_does_not() {
        // The length is load-bearing rather than decoration: the same access is
        // inside one region and past the end of another.
        assert!(!window(4096).inside(4096, Width::Byte));
        assert!(window(8192).inside(4096, Width::Byte));
    }

    #[test]
    fn a_proof_carries_the_exact_access_it_was_made_for() {
        let admitted = Admitted::new(0x40, Width::Long);
        assert_eq!(admitted.offset(), 0x40);
        assert_eq!(admitted.width(), Width::Long);
    }

    #[test]
    fn a_replacement_of_the_admitted_width_is_bound_to_the_proof() {
        let admitted = Admitted::new(0x40, Width::Long);
        assert_eq!(admitted.binds(&Data::from_u64(0x1234, Width::Long)), Ok(()));
    }

    #[test]
    fn a_replacement_of_any_other_width_is_refused() {
        // The critical case: a handler answering a one-byte write with sixteen
        // bytes would otherwise write fifteen bytes the guest never wrote, past
        // the end of a mapping and through a misaligned pointer.
        let admitted = Admitted::new(0x40, Width::Long);
        for got in Width::ALL.into_iter().filter(|width| *width != Width::Long) {
            assert_eq!(
                admitted.binds(&Data::from_u64(1, got)),
                Err(Inadmissible::Answer {
                    wanted: Width::Long,
                    got
                }),
                "a {got:?} replacement must not commit to a Long access"
            );
        }
    }
}
