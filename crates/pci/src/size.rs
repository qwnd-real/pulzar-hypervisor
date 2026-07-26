//! Learning how much address space a register decodes, which takes writing to
//! it.
//!
//! A base address register reports where a function answers and never how far.
//! The extent is reported only by writing all ones into the register and
//! reading back what stayed set: the bits a device does not implement read as
//! zero, so the lowest bit still set is the size, and the register has to be
//! put back afterwards.
//!
//! That is a destructive read dressed as a query, and it is why nothing in the
//! survey calls it. For the moments between the two writes the register decodes
//! somewhere else entirely — everything from the base upward, in practice —
//! which on a machine whose firmware is still driving its own devices can
//! answer a transaction that belonged to something else.
//!
//! # What the sequence has to do, in this order
//!
//! Decoding is switched off first, so that the register never describes a range
//! the function actually answers in. It is switched back on last. Between them
//! the register is saved, filled, read, and restored, and then read once more:
//! a register that does not hold what it was given back is one whose function
//! has been left in a state this crate cannot describe, which is worth an error
//! rather than a value.
//!
//! Interrupts are masked around the whole of it. Not for the device's sake —
//! the bus does not care — but because the window during which a function
//! decodes nothing should not also contain whatever an interrupt handler
//! decides to do.

use log::warn;
use x86_64::instructions::interrupts;

use crate::{
    Function, Offset, PciError,
    access::Config,
    bar::{Bar, Width},
    header::{self, Command, Space},
};

/// How many bytes the base address register in `slot` decodes.
///
/// `None` for a register that is not implemented, that is the upper half of the
/// one below it, or whose encoding made no sense when the machine was surveyed
/// — none of which has an extent, and none of which is worth writing to in
/// order to find that out again.
///
/// # Errors
///
/// [`PciError::NoSuchRegister`] for a slot beyond the six a function has,
/// [`PciError::Unmapped`] if no mapping was kept for the function,
/// [`PciError::NotRestored`] if the register did not take its old value back,
/// or whatever an access reported.
///
/// # Safety
///
/// The function must be one this hypervisor owns. While this runs, the function
/// decodes nothing and its register briefly describes a range it does not
/// answer in; doing that to a device another driver is using can lose that
/// driver a transaction, and doing it to one that is bus-mastering can lose a
/// transfer already in flight. Pulzar leaves the firmware environment running,
/// so that condition is not met for any device until the guest's own
/// `ExitBootServices` has been intercepted.
pub unsafe fn size(function: &Function, slot: usize) -> Result<Option<u64>, PciError> {
    let bar = *function.bars().get(slot).ok_or(PciError::NoSuchRegister {
        address: function.address(),
        slot,
    })?;
    let (space, mask, wide) = match bar {
        Bar::Port { .. } => (Space::Port, PORT_MASK, false),
        Bar::Memory {
            width: Width::Bits64,
            ..
        } => (Space::Memory, MEMORY_MASK, true),
        Bar::Memory {
            width: Width::Bits32 | Width::Bits32Low,
            ..
        } => (Space::Memory, MEMORY_MASK, false),
        Bar::Memory {
            width: Width::Reserved(_),
            ..
        }
        | Bar::Unset
        | Bar::Upper
        | Bar::Malformed => return Ok(None),
    };
    let at = header::BARS.plus(stride(slot));
    let config = function.config()?;

    // SAFETY: every write below is either a value the register just held or the
    // all-ones the specification defines as the way to ask a register its size,
    // and the command register is only ever given back what it already had with
    // one decode bit taken out and put back. The caller vouches for the device
    // being one this hypervisor may disturb at all.
    let probed = interrupts::without_interrupts(|| unsafe { probe(&config, at, space, wide) })?;

    let restored = config.u32(at)?;
    if restored != probed.saved {
        warn!(
            "pci: {} register {slot} came back as {restored:#010x} rather than {:#010x}",
            function.address(),
            probed.saved
        );
        return Err(PciError::NotRestored {
            address: function.address(),
            slot,
        });
    }
    Ok(extent(probed, mask))
}

/// What one sizing pass read back out of a register.
#[derive(Clone, Copy, Debug)]
struct Probed {
    saved: u32,
    low: u32,
    high: Option<u32>,
}

/// Switches decoding off, asks the register its size, and puts everything back.
///
/// # Safety
///
/// As [`size`]. The caller must also have masked interrupts, so that nothing
/// runs between switching decoding off and switching it back on.
unsafe fn probe(config: &Config, at: Offset, space: Space, wide: bool) -> Result<Probed, PciError> {
    let upper = at.plus(REGISTER_BYTES);
    let command = Command::from_bits(config.u16(header::COMMAND)?);
    // SAFETY: the command register is given back exactly what it held with one
    // decode bit cleared, which is the state a function is in before firmware
    // configures it.
    unsafe { config.write_u16(header::COMMAND, command.without(space).into_bits())? };

    let saved = config.u32(at)?;
    let saved_high = if wide { Some(config.u32(upper)?) } else { None };

    // SAFETY: all ones is what the specification defines as the request for a
    // register's size. The function decodes nothing while it is in place.
    unsafe { config.write_u32(at, u32::MAX)? };
    if wide {
        // SAFETY: as above, for the other half of the same register.
        unsafe { config.write_u32(upper, u32::MAX)? };
    }
    let low = config.u32(at)?;
    let high = if wide { Some(config.u32(upper)?) } else { None };

    // SAFETY: both halves are given back exactly what they held, and the
    // command register the value it had on entry.
    unsafe {
        config.write_u32(at, saved)?;
        if let Some(saved_high) = saved_high {
            config.write_u32(upper, saved_high)?;
        }
        config.write_u16(header::COMMAND, command.into_bits())?;
    }
    Ok(Probed { saved, low, high })
}

/// How much address space the bits that stayed set describe.
///
/// The lowest bit a device left set is the size, so complementing what came
/// back and adding one turns it into a byte count. A register that came back
/// with no address bits set at all is one the function does not implement.
///
/// A narrow register's answer only occupies its own thirty-two bits, so the
/// bits above it are filled in before the complement — otherwise every narrow
/// register would appear to decode the whole address space.
fn extent(probed: Probed, mask: u64) -> Option<u64> {
    let Some(high) = probed.high else {
        let narrow = u32::try_from(mask & u64::from(u32::MAX)).unwrap_or(u32::MAX);
        let value = probed.low & narrow;
        return (value != 0).then(|| u64::from((!value).wrapping_add(1)));
    };
    let value = ((u64::from(high) << u32::BITS) | u64::from(probed.low)) & mask;
    (value != 0).then(|| (!value).wrapping_add(1))
}

/// Bytes from one base address register to the next.
fn stride(slot: usize) -> u16 {
    u16::try_from(slot)
        .unwrap_or(u16::MAX)
        .saturating_mul(REGISTER_BYTES)
}

/// Bytes one base address register occupies.
const REGISTER_BYTES: u16 = 4;

/// Bits of a memory register that carry an address, widened to the size a pair
/// of them describes.
const MEMORY_MASK: u64 = !0xF;

/// Bits of an I/O register that carry an address.
const PORT_MASK: u64 = !0x3;
