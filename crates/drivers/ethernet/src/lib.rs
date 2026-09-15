//! The network controllers a guest reads its interfaces' identities from,
//! answered for so that what it reads names nothing.
//!
//! A network interface is known on its network by a media access control
//! address, and a controller carries that address in its registers — loaded
//! from its own nonvolatile memory at power-on and at every reset, and
//! asked for by whatever wants to know who the interface is: the guest's
//! driver at probe, the stack above it, `ethtool`, an `NDIS` query. This
//! driver replaces what those asks are answered with, the same way the
//! storage driver replaces what an identify command is answered with: the
//! address's serial half is replaced with what [`spoof`] makes of it under
//! the machine's seed, the maker's half is left alone, and the replacement
//! is stable across boots and unrelated to the original.
//!
//! # Two families, one decision each
//!
//! The controllers this answers for are Intel's and Realtek's, and within
//! each the register layout is the one layout: Intel's address lives in a
//! receive-address pair on a page of receive filters; Realtek's lives in
//! the registers at the start of a 256-byte register file, and in the
//! serial EEPROM behind a control byte inside that same file. What is
//! trapped is the page the identity sits on and nothing more — on both
//! families, because a page is the finest the tables can trap and a
//! wider trap is a cost the packet path pays for nothing.
//!
//! # What survives a reset
//!
//! The replacement is answered out of the trap's own state, so nothing the
//! guest does can bring the real address back. A device reset reloads the
//! registers from the hardware's own memory; the reads still answer with
//! the replacement, because they never consult the registers. The serial
//! EEPROM is deeper still: no reset reaches it, and the words a guest
//! shifts out of it come from this driver either way.
//!
//! # What each access costs
//!
//! Intel's page of receive filters holds nothing the packet path touches:
//! the rings, the interrupt registers, the message-signaled interrupt
//! tables are all on other pages, left alone. A guest reaches this page
//! when it asks about its address, programs a filter, or updates a
//! multicast table — probe-time and configuration-time work, none of it
//! per packet. Realtek's register file has no such separation: 256 bytes
//! hold the identity and the interrupt and transmit registers alike, so
//! the one page that carries the identity also carries what a driver
//! touches once per interrupt. Each of those accesses becomes one exit
//! more than it was, and the handler for an access that carries no
//! identity is one comparison and the hardware's own answer, so the exit
//! is the whole of the cost.
//!
//! # Failing open
//!
//! Every way a controller can fail to be taken over — a base address
//! register that decodes nothing or too little, a page that cannot be
//! mapped — ends the same way as the storage driver's failures: a loud
//! warning, and the controller left alone. A guest that boots reading one
//! real address is a failure of the spoofing; a guest that does not boot
//! is a failure of the hypervisor, which is the worse one.
//!
//! # What this driver deliberately does not do
//!
//! Intel's nonvolatile memory is reached through registers on the first
//! page of the register file, which is not taken: a guest that dumps it
//! sees the hardware's own contents, real address's copy included, and a
//! guest of the older `e1000` family — whose driver reads its address out
//! of that memory rather than out of the registers — is shown the truth.
//! The first page is where the interrupt registers live, and the identity
//! the receive-address pair carries does not need it.
//!
//! A Realtek controller also answers at a port-I/O bar, and a guest driver
//! that chooses it over the memory bar is not reached by these tables at
//! all — the choice is rare and off by default in the drivers that offer
//! it, and intercepting port I/O is machinery this hypervisor does not
//! have. Controllers made to the Realtek layout but sold under other
//! vendors' identifiers are left alone with the rest of the uninvited.

#![no_std]

extern crate alloc;

mod eeprom;
mod intel;
mod mac;
mod realtek;

use emulate::{Data, Read, Width};
use log::{info, warn};
use npt::Change;
use paging::PagingError;
use partition::{Partition, PartitionError};
use pci::PciError;
use spin::Once;
use thiserror::Error;

/// Intel's PCI vendor identifier.
const INTEL: u16 = 0x8086;

/// Realtek's.
const REALTEK: u16 = 0x10ec;

/// The controllers this driver took over, so that a second takeover can be
/// refused rather than leave two drivers answering for one register page.
static ADOPTED: Once<()> = Once::new();

/// Interposes every ethernet controller the machine has that one of the two
/// families answers for.
///
/// Called once, on the boot processor, after the guest's `ExitBootServices`
/// has been intercepted and before the guest's other processors are started.
/// A controller that cannot be taken over is left alone with a warning,
/// because a guest that boots reading one real address is the better
/// failure.
///
/// # Errors
///
/// [`EthernetError::AlreadyAdopted`] for a second call,
/// [`EthernetError::Pci`] if the machine was never surveyed,
/// [`EthernetError::Paging`] if the address space is not the machine's yet,
/// or [`EthernetError::Partition`] if the barrier owed for the trapped
/// pages could not be paid.
pub fn adopt(partition: &'static Partition) -> Result<(), EthernetError> {
    if ADOPTED.is_completed() {
        return Err(EthernetError::AlreadyAdopted);
    }
    let topology = pci::topology()?;
    let seed = config::serial_seed();
    // Taking a controller over asks its configuration space, drives its
    // serial EEPROM, and registers its page — all of it under the address
    // space's lock, which is the one exception the lock's rules allow, and
    // only because nothing else is running: the guest is stopped mid-call
    // and the other processors have not started.
    let owed = paging::with(|space| {
        let mut owed = Change::None;
        let mut taken = 0;
        for function in topology
            .functions()
            .iter()
            .filter(|function| function.class().is_ethernet())
        {
            let region = match function.vendor() {
                INTEL => intel::take(space, function, &seed),
                REALTEK => realtek::take(space, function, &seed),
                // Another maker's controller, or a clone answering to the
                // Realtek layout under another vendor's identifier: not one
                // this driver has taken responsibility for.
                _ => continue,
            };
            let region = match region {
                Ok(region) => region,
                Err(error) => {
                    warn!("ethernet: {} was left alone: {error}", function.address());
                    continue;
                }
            };
            // One registration per controller, so that a controller which
            // cannot be registered costs itself and nothing else — and so
            // that the changes the successful ones owe stay in hand to be
            // paid even then.
            match partition.interpose(space, [region]) {
                Ok(change) => {
                    owed = owed.and(change);
                    taken += 1;
                }
                Err(error) => {
                    warn!(
                        "ethernet: {} was left alone: its page could not be trapped: {error}",
                        function.address()
                    );
                }
            }
        }
        if taken == 0 {
            info!("ethernet: no controller was taken over");
        }
        Ok::<_, EthernetError>(owed)
    })??;
    partition.barrier(owed)?;
    ADOPTED.call_once(|| ());
    Ok(())
}

/// Why the machine's ethernet controllers are not being answered for.
#[derive(Debug, Error)]
pub enum EthernetError {
    /// They were taken over already, and a second takeover would leave two
    /// drivers answering for one register page.
    #[error("the machine's ethernet controllers were already taken over")]
    AlreadyAdopted,
    /// The machine was never surveyed, so there is nothing to find
    /// controllers in.
    #[error(transparent)]
    Pci(#[from] PciError),
    /// The address space is not the machine's, so there is nowhere to map a
    /// register page.
    #[error(transparent)]
    Paging(#[from] PagingError),
    /// The page could not be trapped, or the barrier owed for trapping it
    /// not paid.
    #[error(transparent)]
    Partition(#[from] PartitionError),
    /// The first base address register decodes nothing, or is not
    /// implemented at all.
    #[error("bar0 decodes nothing")]
    Unaddressed,
    /// The register decodes less than the page the identity sits on.
    #[error("bar0 is {bytes:#x} bytes, too small for the page the identity sits on")]
    TooSmall {
        /// How many bytes the register decodes.
        bytes: u64,
    },
    /// The register decodes from an address the tables cannot trap a page
    /// at.
    #[error("bar0 starts at {base:#x}, which is not a page boundary")]
    Misaligned {
        /// Where the register decodes from.
        base: u64,
    },
}

/// A read the hardware answers, with some of its bytes replaced.
///
/// The replacement begins at `at`, region-relative, and runs for as many
/// bytes as it has; an access is composed at whatever width and alignment
/// reached it, so a straddling access keeps the hardware's answer for the
/// bytes outside the replacement and takes the replacement's for the bytes
/// inside.
fn overlaid(access: Read<'_>, replacement: &[u8], at: u64) -> Data {
    let width = access.width();
    let mut bytes = [0_u8; 16];
    if let Some(real) = access.hardware() {
        let real = real.bytes();
        bytes[..real.len()].copy_from_slice(real);
    }
    for position in 0..width.bytes() {
        let absolute = access.offset() + position as u64;
        if absolute >= at {
            let within = (absolute - at) as usize;
            if within < replacement.len() {
                bytes[position] = replacement[within];
            }
        }
    }
    match width {
        Width::Vector => Data::vector_from(bytes),
        width => {
            let mut value = 0_u64;
            for (position, byte) in bytes.iter().enumerate().take(width.bytes()) {
                value |= u64::from(*byte) << (position * 8);
            }
            Data::from_u64(value, width)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{INTEL, REALTEK};

    #[test]
    fn the_vendors_are_the_two_families() {
        assert_eq!(INTEL, 0x8086);
        assert_eq!(REALTEK, 0x10ec);
    }
}
