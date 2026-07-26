//! The ACPI power management timer, as a counter to measure against.
//!
//! There is nothing to program and nothing to start: the timer is part of the
//! chipset's power management block, it runs whenever the machine is awake, and
//! ACPI fixes its rate rather than leaving firmware to report one. Opening it
//! is therefore either mapping four bytes or, in the usual case of a register
//! in I/O space, nothing at all.
//!
//! It is a reference and never a timebase. The counter is 24 or 32 bits wide,
//! so it wraps every few seconds or few minutes, and telling one wrap from two
//! would need something to poll it more often than that forever.

use core::num::NonZeroU64;

use acpi::{PmTimer, Space};
use paging::{AddressSpace, CacheType, Protection};
use x86_64::PhysAddr;

use crate::{
    ClockError, Frequency,
    counter::{Counter, Kind, Register},
    reference::Borrowed,
};

/// Bytes the timer's register occupies.
const REGISTER_BYTES: u64 = 4;

/// Alignment the register needs to be read at its width.
const ALIGN: u64 = REGISTER_BYTES;

/// The rate ACPI fixes for the timer.
const FREQUENCY: Frequency = Frequency::new(NonZeroU64::new(PmTimer::FREQUENCY).unwrap());

/// Opens the timer firmware described.
///
/// # Errors
///
/// [`ClockError::UnreachableRegister`] if firmware put the register in an
/// address space nothing can reach, [`ClockError::BadAddress`] if the address
/// is not one this processor can form, [`ClockError::Misaligned`] if it is not
/// aligned for a four-byte read, or [`ClockError::Paging`] if a
/// memory-mapped register cannot be mapped.
pub(crate) fn open(space: &mut AddressSpace, timer: PmTimer) -> Result<Borrowed, ClockError> {
    let address = timer.register().address();
    let (register, mapping) = match timer.register().space() {
        Space::Io => {
            let port = u16::try_from(address).map_err(|_| ClockError::BadAddress { address })?;
            (Register::Port(port), None)
        }
        Space::Memory => {
            let base =
                PhysAddr::try_new(address).map_err(|_| ClockError::BadAddress { address })?;
            if !address.is_multiple_of(ALIGN) {
                return Err(ClockError::Misaligned {
                    address,
                    align: ALIGN,
                });
            }
            // SAFETY: this is a device register firmware described, outside
            // every range the memory map calls memory, so no allocator owns it
            // and nothing else in this address space maps it. Read-only because
            // a counter is never written, and uncached-minus because a cached
            // alias would answer from a cache line instead of the chipset.
            let mapping = unsafe {
                space.map_physical(
                    base,
                    REGISTER_BYTES,
                    Protection::ReadOnly,
                    CacheType::UncachedMinus,
                )
            }?;
            (Register::Memory32(mapping.addr()), Some(mapping))
        }
        space @ Space::Other(_) => return Err(ClockError::UnreachableRegister { space }),
    };

    // SAFETY: a mapped register travels beside the counter in the `Borrowed`
    // that owns both, so it outlives every read, and it is four-byte aligned by
    // the check above. Reading the timer — through memory or through its port —
    // returns the count and does nothing else to the machine.
    let counter = unsafe { Counter::new(Kind::PmTimer, register, FREQUENCY, timer.bits()) };
    Ok(Borrowed::new(counter, mapping, None))
}
