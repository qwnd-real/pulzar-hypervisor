//! The Intel controllers: the one page of their register file that carries
//! an identity, answered for.
//!
//! An Intel controller keeps the address its interface is known by in a
//! receive address register — a pair of registers, `RAL0` and `RAH0`,
//! holding four bytes and two — and everything the guest asks about its
//! permanent address comes from that pair: the Linux driver copies it out
//! at probe and calls it the permanent address, an `NDIS` driver does the
//! same through its own query, and `ethtool` reports what the driver kept.
//! The controller loads the pair from its own nonvolatile memory at
//! power-on and at every reset, which is the one way the hardware's real
//! address could come back.
//!
//! So this driver traps the page the pair sits on — the page of receive
//! filters, which holds nothing the packet path touches — and answers every
//! read of the pair with the replacement address, whatever the hardware
//! behind the page holds. A reset reloads the real address into the
//! hardware; the guest reads the replacement anyway, because the answer
//! comes from the trap's own state rather than from the registers. Writes
//! the guest makes to the page all reach the hardware: a driver programs
//! the filters it believes in, and what it believes in is the replacement.
//!
//! The rest of the register file is untouched. In particular the
//! controller's nonvolatile memory is reached through registers on the
//! first page, which this driver leaves alone: a guest that dumps the
//! memory sees the hardware's own contents, the real address's copy
//! included, and a guest of the older `e1000` family — whose driver reads
//! its address out of that memory rather than out of the registers — is
//! shown the truth. Trapping the first page would answer for those too,
//! and it is the page the interrupt registers live on, which the packet
//! path pays for; the receive filters' page is the one an identity needs
//! and the only one taken.

use alloc::boxed::Box;

use emulate::{Capability, Commit, Data, Device, Hardware, Read, Region, Trap, Write};
use log::info;
use paging::{AddressSpace, CacheType, Protection};
use pci::Function;
use spoof::Seed;
use x86_64::PhysAddr;

use crate::{EthernetError, mac, overlaid};

/// How many bytes one page of the register file is.
const PAGE: u64 = 0x1000;

/// Where the page of receive filters sits within the register file: the
/// multicast table below the address registers, the VLAN table above them.
const FILTERS: u64 = 0x5000;

/// Where the receive address pair sits within that page.
const ADDRESS: u64 = 0x0400;

/// Takes over one controller: reads the address the hardware carries,
/// replaces it, and returns the region the replacement is answered through.
///
/// The region's device owns everything the answering needs, so the state
/// lives exactly as long as the registration does.
///
/// # Errors
///
/// [`EthernetError::Unaddressed`] for a first base address register that
/// decodes nothing, [`EthernetError::TooSmall`] for one that decodes less
/// than the page of filters, [`EthernetError::Misaligned`] for one the
/// tables cannot trap a page at, or [`EthernetError::Paging`] if the page
/// cannot be read.
pub(crate) fn take(
    space: &mut AddressSpace,
    function: &Function,
    seed: &Seed,
) -> Result<Region, EthernetError> {
    let base = function.bars()[0]
        .memory_base()
        .ok_or(EthernetError::Unaddressed)?;
    // SAFETY: the guest's ExitBootServices has been intercepted, so the
    // devices are this hypervisor's to configure: the guest is stopped
    // mid-call and the other processors have not been started, which
    // leaves no driver to lose a transaction to.
    let Some(extent) = unsafe { pci::size(function, 0) }? else {
        return Err(EthernetError::Unaddressed);
    };
    if extent < FILTERS + PAGE {
        return Err(EthernetError::TooSmall { bytes: extent });
    }
    if !base.as_u64().is_multiple_of(PAGE) {
        return Err(EthernetError::Misaligned {
            base: base.as_u64(),
        });
    }
    let original = original(space, base)?;
    let mac = mac::spoofed(original, seed);
    info!(
        "ethernet: {} at {:#x} answers for the address {} in place of {}",
        function.address(),
        base.as_u64(),
        mac,
        original,
    );
    Ok(Region {
        gpa: base + FILTERS,
        bytes: PAGE,
        trap: Some(Trap::Everything),
        device: Box::new(Filters { mac }),
    })
}

/// Reads the address the hardware carries, out of the receive address pair
/// it is loaded into.
///
/// # Errors
///
/// [`EthernetError::Paging`] if the page cannot be mapped for the reading.
fn original(space: &mut AddressSpace, base: PhysAddr) -> Result<mac::Mac, EthernetError> {
    // SAFETY: one page of device registers, which the caller has just
    // vouched is this hypervisor's to read, mapped read-only around this
    // one pair of reads and unmapped before it returns.
    unsafe {
        space.with_physical(
            base + FILTERS,
            PAGE,
            Protection::ReadOnly,
            CacheType::UncachedMinus,
            |filters| {
                // SAFETY: the address pair sits at a known offset inside
                // the page this mapping covers, and both of its registers
                // are read whole at their own width.
                let low = (filters + ADDRESS).as_u64() as *const u32;
                let high = (filters + ADDRESS + 4).as_u64() as *const u32;
                let low = low.read_volatile();
                let high = high.read_volatile();
                mac::Mac::new([
                    low as u8,
                    (low >> 8) as u8,
                    (low >> 16) as u8,
                    (low >> 24) as u8,
                    high as u8,
                    (high >> 8) as u8,
                ])
            },
        )
    }
    .map_err(EthernetError::from)
}

/// The page of receive filters, as a device the emulator reaches.
struct Filters {
    /// The address the guest is shown.
    mac: mac::Mac,
}

impl Device for Filters {
    fn capability(&self) -> Capability {
        // The filters are registers a driver may reach at any width and a
        // piece at a time, so every aligned scalar width is admitted; the
        // address pair is composed at whatever width reached it.
        Capability::scalar()
    }

    fn hardware(&self) -> Hardware {
        Hardware::Reached
    }

    fn read(&self, access: Read<'_>) -> Data {
        if (ADDRESS..ADDRESS + 8).contains(&access.offset()) {
            // The address pair: the replacement's six bytes where the
            // access touches them, and the hardware's two bytes of padding
            // that follow — one of which is the pair's valid bit, which a
            // reset leaves set and a driver that wrote the pair left as it
            // willed.
            return overlaid(access, &self.mac.bytes(), ADDRESS);
        }
        // Every other register on the page is the hardware's to answer: the
        // multicast and VLAN tables, the rest of the address array — none
        // of them carrying more of an identity than a filter does.
        access
            .hardware()
            .unwrap_or_else(|| Data::from_u64(0, access.width()))
    }

    fn write(&self, _access: Write<'_>) -> Commit {
        // Everything the guest writes to this page is the hardware's to
        // take: the filter tables, and the address pair itself when a
        // driver programs the address it believes in — which is the
        // replacement, because that is what it read.
        Commit::Hardware
    }
}

#[cfg(test)]
mod tests {
    use super::{ADDRESS, FILTERS, PAGE};

    #[test]
    fn the_address_pair_is_inside_the_filters_page() {
        assert!(ADDRESS + 8 <= PAGE);
        assert!(FILTERS + PAGE > FILTERS + ADDRESS);
    }
}
