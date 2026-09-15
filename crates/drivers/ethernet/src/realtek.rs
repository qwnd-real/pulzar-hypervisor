//! The Realtek controllers: the one page their whole register file lives
//! on, answered for where it carries an identity.
//!
//! A Realtek register file is 256 bytes, too small to fill a page, and it
//! holds everything the controller is: the address registers at its start,
//! the interrupt and transmit registers that follow, and the byte at
//! `0x50` that is both the lock on the configuration registers and the
//! four wires of a serial EEPROM. The tables cannot trap less than the
//! whole page, so the whole page is what this driver takes — and that
//! single page is the one place the controller's identity is kept:
//!
//! - The address registers, `MAC0` through `MAC4`, are where the Linux driver
//!   of the gigabit family reads the interface's address at probe. Every read
//!   of them is answered with the replacement.
//! - The serial EEPROM behind `0x50` is where the older `8139` family's driver
//!   reads the address from, shifting a command out and a word in one bit per
//!   raised clock. The word that comes back is this driver's answer, with the
//!   three words that spell the address replaced — the rest of what the EEPROM
//!   holds, signature word included, is the hardware's own, harvested before
//!   the guest runs.
//!
//! Both answers come out of the trap's own state, so a reset changes
//! nothing the guest can see: the hardware reloads the address registers
//! from its EEPROM, and the reads still answer with the replacement, and
//! the EEPROM itself is a thing no reset reaches. Writes the guest makes
//! all reach the hardware — a driver programs the address it believes in,
//! and toggles the EEPROM's wires for its own reasons, and the hardware
//! takes both exactly as the guest made them.
//!
//! The cost of this page is the cost of the controller's layout: the
//! interrupt registers a driver touches once per interrupt are on it, so
//! each of those accesses becomes one exit more than it was. There is no
//! smaller page to take, and the handler for an access that carries no
//! identity is one comparison and the hardware's own answer, so the exit
//! is the whole of the cost.

use alloc::{boxed::Box, vec::Vec};

use emulate::{Capability, Commit, Data, Device, Hardware, Read, Region, Trap, Write};
use log::info;
use paging::{AddressSpace, CacheType, Protection};
use pci::Function;
use spoof::Seed;
use x86_64::{PhysAddr, VirtAddr};

use crate::{EthernetError, eeprom, mac, overlaid};

/// How many bytes one page is.
const PAGE: u64 = 0x1000;

/// How many bytes the register file is.
const REGISTERS: u64 = 0x100;

/// Where the address registers sit within it.
const ADDRESS: u64 = 0x00;

/// Where the EEPROM's wires and the configuration lock sit within it.
const CONTROL: u64 = 0x50;

/// The word of the serial EEPROM a driver reads first, to learn how wide
/// its addresses are.
const SIGNATURE: u16 = eeprom::SIGNATURE;

/// The words of the serial EEPROM that spell the interface's address.
const IDENTITY: usize = 7;

/// How wide the addresses of an EEPROM whose signature word reads back are.
const WIDE: u32 = 8;

/// How wide the addresses of the smaller EEPROM are.
const NARROW: u32 = 6;

/// Takes over one controller: reads the address the hardware carries,
/// harvests the EEPROM behind its control byte, replaces the identity in
/// both, and returns the region the replacements are answered through.
///
/// The region's device owns everything the answering needs, so the state
/// lives exactly as long as the registration does.
///
/// # Errors
///
/// [`EthernetError::Unaddressed`] for a first base address register that
/// decodes nothing, [`EthernetError::TooSmall`] for one that decodes less
/// than the register file, or [`EthernetError::Paging`] if the page cannot
/// be mapped for the reading.
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
    if extent < REGISTERS {
        return Err(EthernetError::TooSmall { bytes: extent });
    }
    // The register file need not start where its page does, so the page is
    // what is trapped and the file's place inside it is what the device
    // works from.
    let page = PhysAddr::new(base.as_u64() & !(PAGE - 1));
    let within = base.as_u64() & (PAGE - 1);
    // SAFETY: one page of device registers, which the caller has just
    // vouched is this hypervisor's to drive, mapped read-write around the
    // harvest and unmapped before it returns. The writes go to the one
    // control byte the EEPROM's wires live in, in the sequences a driver
    // itself uses, and the byte is restored to what it held before.
    let (original, replacement, serial) = unsafe {
        space.with_physical(
            page,
            PAGE,
            Protection::ReadWrite,
            CacheType::UncachedMinus,
            |mapped| identify(VirtAddr::new(mapped.as_u64() + within), seed),
        )
    }?;
    info!(
        "ethernet: {} at {:#x} answers for the address {} in place of {}",
        function.address(),
        base.as_u64(),
        replacement,
        original,
    );
    Ok(Region {
        gpa: page,
        bytes: PAGE,
        trap: Some(Trap::Everything),
        device: Box::new(File {
            mac: replacement,
            serial,
            file: within,
        }),
    })
}

/// Reads the address the hardware carries and harvests the EEPROM behind
/// its control byte, with the address replaced in both: the original, the
/// replacement, and the EEPROM to serve.
///
/// # Safety
///
/// `file` must be the register file's mapping, for the whole of this call.
unsafe fn identify(file: VirtAddr, seed: &Seed) -> (mac::Mac, mac::Mac, Option<eeprom::Serial>) {
    // SAFETY: the caller vouches for the mapping, and the address
    // registers are at its start.
    let original = unsafe { read_mac(file) };
    let replacement = mac::spoofed(original, seed);
    // The control byte is a known offset inside the register file the
    // caller mapped, and nothing else runs to touch it.
    let control = (file + CONTROL).as_u64() as *mut u8;
    // SAFETY: the same byte, for the whole of the harvest.
    let saved = unsafe { control.read_volatile() };
    // SAFETY: the same byte, driven the way a driver drives it.
    let serial = unsafe { harvest(control, &replacement) };
    // SAFETY: the same byte, put back to what the firmware left it as.
    unsafe { control.write_volatile(saved) };
    (original, replacement, serial)
}

/// Reads the address the hardware carries, out of the registers it is
/// loaded into.
///
/// # Safety
///
/// `file` must be the register file's mapping, and the address registers
/// are at its start.
unsafe fn read_mac(file: VirtAddr) -> mac::Mac {
    // SAFETY: the caller vouches for the mapping, and the address pair is
    // read whole at its own width.
    unsafe {
        let low = file.as_u64() as *const u32;
        let high = (file + 4).as_u64() as *const u32;
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
    }
}

/// Reads the whole of the serial EEPROM behind a control byte, and returns
/// it with the address replaced — or `None` if nothing answered, which is
/// the gigabit controllers: they keep their address in the registers and
/// carry no EEPROM a driver can shift words out of.
///
/// # Safety
///
/// `control` must name the one-byte register the wires live in, for the
/// whole of the harvest; nothing else may drive it concurrently.
unsafe fn harvest(control: *mut u8, replacement: &mac::Mac) -> Option<eeprom::Serial> {
    // A driver reads the signature word with the wider addressing first:
    // an EEPROM that answers with it is the wider one, and one that does
    // not has served a rotation, which is what the narrower EEPROM does
    // with the wider command and what the one that is not there does with
    // any command at all.
    // SAFETY: the caller vouches for the register.
    let address_bits = if unsafe { eeprom::read(control, 0, WIDE) } == SIGNATURE {
        WIDE
    } else {
        NARROW
    };
    let mut words = Vec::new();
    for word in 0..1_u16 << address_bits {
        // SAFETY: the same register, one word at a time.
        words.push(unsafe { eeprom::read(control, word, address_bits) });
    }
    if words.iter().all(|&word| word == 0) {
        return None;
    }
    let [first, second, third] = replacement.nvm_words();
    words[IDENTITY] = first;
    words[IDENTITY + 1] = second;
    words[IDENTITY + 2] = third;
    Some(eeprom::Serial::new(words, address_bits))
}

/// The register file's page, as a device the emulator reaches.
struct File {
    /// The address the guest is shown.
    mac: mac::Mac,
    /// The EEPROM the guest shifts words out of, if the hardware has one.
    serial: Option<eeprom::Serial>,
    /// Where the register file starts within this page.
    file: u64,
}

impl Device for File {
    fn capability(&self) -> Capability {
        // The register file is reached a byte at a time and a dword at a
        // time, address registers whole and EEPROM wires singly, so every
        // aligned scalar width is admitted.
        Capability::scalar()
    }

    fn hardware(&self) -> Hardware {
        Hardware::Reached
    }

    fn read(&self, access: Read<'_>) -> Data {
        let address = self.file + ADDRESS;
        if (address..address + 8).contains(&access.offset()) {
            // The address registers: the replacement's six bytes where the
            // access touches them, and the hardware's two bytes of padding
            // that follow.
            return overlaid(access, &self.mac.bytes(), address);
        }
        let control = self.file + CONTROL;
        if access.offset() <= control && control < access.offset() + access.width().span() {
            // The control byte, with its data-out bit carrying what the
            // served EEPROM drives. While no transaction is under way the
            // bit is the hardware's own, which is the honest answer for a
            // chip that is not selected.
            if let Some(serial) = &self.serial {
                return patched(access, control, serial.drives());
            }
        }
        // Everything else on the page — the interrupt, transmit and
        // configuration registers, and any neighbouring device the page
        // happens to share — is the hardware's to answer.
        access
            .hardware()
            .unwrap_or_else(|| Data::from_u64(0, access.width()))
    }

    fn write(&self, access: Write<'_>) -> Commit {
        if let Some(serial) = &self.serial {
            if let Some(byte) = names(&access, self.file + CONTROL) {
                serial.wrote(byte);
            }
        }
        // Everything the guest writes is the hardware's to take: the
        // address registers themselves when a driver programs the address
        // it believes in, and the control byte whose wires the served
        // EEPROM watched without keeping anything from the hardware.
        Commit::Hardware
    }
}

/// A read that touches the control byte, with its data-out bit carrying
/// what the served EEPROM drives — set, cleared, or left as the hardware
/// has it.
///
/// The bit is the low bit of the control byte, wherever that byte falls
/// inside the width the guest read.
fn patched(access: Read<'_>, at: u64, driven: Option<bool>) -> Data {
    let mut value = access.hardware().map_or(0, |real| real.as_u64());
    let bit = 1 << (8 * (at - access.offset()));
    match driven {
        Some(true) => value |= bit,
        Some(false) => value &= !bit,
        None => {}
    }
    Data::from_u64(value, access.width())
}

/// The byte a write names at `at`, if it names one.
fn names(access: &Write<'_>, at: u64) -> Option<u8> {
    let first = access.offset();
    if at < first || at >= first + access.width().span() {
        return None;
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the byte is eight bits of the whole value the guest wrote"
    )]
    Some((access.value().as_u64() >> (8 * (at - first))) as u8)
}

#[cfg(test)]
mod tests {
    use super::{ADDRESS, CONTROL, IDENTITY, NARROW, PAGE, REGISTERS, WIDE};

    #[test]
    fn the_identity_sits_inside_the_register_file() {
        assert!(ADDRESS + 8 <= REGISTERS);
        assert!(CONTROL < REGISTERS);
        assert!(REGISTERS <= PAGE);
    }

    #[test]
    fn the_identity_words_are_three() {
        assert_eq!(
            crate::mac::Mac::new([1, 2, 3, 4, 5, 6]).nvm_words().len(),
            3
        );
        assert!(IDENTITY + 3 <= 1 << NARROW);
        assert!(IDENTITY + 3 <= 1 << WIDE);
    }

    #[test]
    fn the_wider_addressing_covers_the_narrower_ones_words() {
        assert!(WIDE > NARROW);
    }
}
