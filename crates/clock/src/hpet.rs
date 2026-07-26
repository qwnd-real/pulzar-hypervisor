//! The event timer's register block: what it says about itself, how it is
//! started, and how it is left as it was found.
//!
//! Three of the block's registers matter here. The capabilities register
//! reports the tick period in femtoseconds and whether the main counter is 64
//! bits wide, which between them are everything needed to turn its ticks into
//! nanoseconds. The configuration register has one bit that decides whether the
//! counter is running at all. The main counter is the value.
//!
//! # Firmware may own it
//!
//! On a machine booted through UEFI the timer is usually stopped: firmware has
//! no further use for it, and whatever runs next is expected to enable it
//! itself. Starting it is therefore often necessary, and it is always a change
//! to hardware that is not pulzar's. So the enable bit is read before it is
//! written, and a timer that was found stopped is stopped again once the clock
//! no longer needs it — unless the clock ends up keeping time from the very
//! counter it started, which cannot be stopped without stopping time.
//!
//! That bit does two things, which is why starting the timer is not a one-line
//! affair. It runs the main counter, and it is also the block's overall
//! interrupt enable: the specification requires it before any comparator can
//! deliver anything. A comparator firmware left armed would therefore start
//! firing the moment the counter did. So every comparator is silenced first and
//! put back as it was on the way out, and the machine gets a running counter
//! and nothing else.

use log::{info, warn};
use paging::{AddressSpace, CacheType, Protection};
use x86_64::{PhysAddr, VirtAddr};

use crate::{
    ClockError, Frequency,
    counter::{Counter, Kind, Register},
    reference::Borrowed,
};

/// Bytes the register block occupies, which the specification fixes at a
/// kilobyte whatever the block contains.
const BLOCK_BYTES: u64 = 1024;

/// Alignment the block must have for its registers to be read as the 64-bit
/// values they are.
const ALIGN: u64 = 8;

/// Offset of the capabilities and identifier register.
const CAPABILITIES: u64 = 0x00;

/// Offset of the general configuration register.
const CONFIGURATION: u64 = 0x10;

/// Offset of the main counter.
const MAIN_COUNTER: u64 = 0xF0;

/// Offset of the first comparator's configuration register.
const COMPARATOR_CONFIG: u64 = 0x100;

/// Bytes from one comparator's registers to the next.
const COMPARATOR_STRIDE: u64 = 0x20;

/// Capabilities: the main counter is 64 bits wide rather than 32.
const COUNTER_64BIT: u64 = 1 << 13;

/// Bits the comparator count, less one, is shifted by in the capabilities
/// register.
const COMPARATORS_SHIFT: u32 = 8;

/// Mask of the comparator count once shifted down.
const COMPARATORS_MASK: u64 = 0x1F;

/// Comparators whose registers fit inside the block the specification fixes the
/// size of. The identifier can claim up to thirty-two; a kilobyte has room for
/// the registers of twenty-four.
const MAPPED_COMPARATORS: u64 = (BLOCK_BYTES - COMPARATOR_CONFIG) / COMPARATOR_STRIDE;

/// Bits the tick period, in femtoseconds, is shifted by in the capabilities
/// register.
const PERIOD_SHIFT: u32 = 32;

/// The longest tick period the specification permits: 100 ns, which is a 10 MHz
/// counter. A block reporting a slower one is describing itself impossibly.
const MAX_PERIOD_FEMTOS: u64 = 100_000_000;

/// Configuration: the main counter advances, and comparators may deliver.
const ENABLE: u64 = 1 << 0;

/// Comparator configuration: this comparator raises its interrupt.
const COMPARATOR_INTERRUPT: u64 = 1 << 2;

/// What an event timer this clock started is owed before it is handed back.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Restore {
    /// Base of the register block, which stays mapped until the release.
    registers: VirtAddr,
    /// Comparators whose interrupt enable this clock cleared, one bit each.
    silenced: u32,
}

/// Maps the block, learns what it is, and starts it if firmware left it
/// stopped.
///
/// # Errors
///
/// [`ClockError::Misaligned`] if firmware put the block somewhere its registers
/// cannot be read from, [`ClockError::HpetPeriod`] if it reports a tick period
/// no timer could have, or [`ClockError::Paging`] if the block cannot be
/// mapped.
pub(crate) fn open(space: &mut AddressSpace, base: PhysAddr) -> Result<Borrowed, ClockError> {
    if !base.as_u64().is_multiple_of(ALIGN) {
        return Err(ClockError::Misaligned {
            address: base.as_u64(),
            align: ALIGN,
        });
    }
    // SAFETY: these are device registers firmware described, outside every
    // range the memory map calls memory, so no allocator owns them and nothing
    // else in this address space maps them for a purpose of its own.
    // Uncached-minus is what device registers need, since a cached alias would
    // answer a counter read from a cache line.
    let mapping = unsafe {
        space.map_physical(
            base,
            BLOCK_BYTES,
            Protection::ReadWrite,
            CacheType::UncachedMinus,
        )
    }?;

    let registers = mapping.addr();
    match probe(registers) {
        Ok((counter, started)) => Ok(Borrowed {
            counter,
            mapping: Some(mapping),
            started,
        }),
        Err(error) => {
            // The block is unusable, so its mapping serves nothing. An error
            // from releasing it is dropped rather than returned: it would
            // replace the reason the block was refused, which is the more
            // useful of the two.
            let _ = unsafe {
                // SAFETY: nothing derived from the mapping outlives this call —
                // `probe` failed, so no counter was built from it.
                space.unmap(mapping)
            };
            Err(error)
        }
    }
}

/// Hands the block back as it was found: the main counter halted, and every
/// comparator this clock silenced raising interrupts again.
///
/// The order is the point. Halting first means the comparator enables go back
/// while the block's overall enable is already clear, so restoring them cannot
/// deliver the very interrupt silencing them existed to prevent.
pub(crate) fn stop(restore: &Restore) {
    let configuration = read(restore.registers, CONFIGURATION);
    write(restore.registers, CONFIGURATION, configuration & !ENABLE);
    unsilence(restore.registers, restore.silenced);
    info!("clock: stopped the hpet again, as firmware had left it");
}

/// Reads what the block says about itself, starting it if it is not running.
///
/// Returns the counter and, where the enable bit had to be set, what the
/// machine is owed to get the block back the way it was.
fn probe(registers: VirtAddr) -> Result<(Counter, Option<Restore>), ClockError> {
    let capabilities = read(registers, CAPABILITIES);
    let femtos = capabilities >> PERIOD_SHIFT;
    let frequency = Frequency::from_period_femtos(femtos)
        .filter(|_| femtos <= MAX_PERIOD_FEMTOS)
        .ok_or(ClockError::HpetPeriod { femtos })?;

    let main = registers + MAIN_COUNTER;
    let (register, bits) = if capabilities & COUNTER_64BIT == 0 {
        (Register::Memory32(main), u32::BITS)
    } else {
        (Register::Memory64(main), u64::BITS)
    };

    let configuration = read(registers, CONFIGURATION);
    let started = if configuration & ENABLE == 0 {
        let silenced = silence(registers, comparators(registers, capabilities));
        write(registers, CONFIGURATION, configuration | ENABLE);
        info!("clock: firmware had left the hpet stopped; started it");
        Some(Restore {
            registers,
            silenced,
        })
    } else {
        None
    };

    // SAFETY: the main counter is inside the block this module just mapped, at
    // an offset that keeps it aligned for its width, and the mapping travels
    // beside the counter in the `Borrowed` that owns both — so it outlives every
    // read. Reading a counter has no effect on the timer.
    let counter = unsafe { Counter::new(Kind::Hpet, register, frequency, bits) };
    Ok((counter, started))
}

/// Stops every comparator raising an interrupt, and reports which ones had to
/// be stopped.
///
/// Nothing here wants the event timer's interrupts — the clock reads its
/// counter and that is all — so this is not a configuration pulzar imposes but
/// one it needs for the few milliseconds it holds the block. It also settles
/// the legacy replacement route, which can only carry interrupts a comparator
/// is raising in the first place.
fn silence(registers: VirtAddr, comparators: u64) -> u32 {
    let mut silenced = 0;
    for index in 0..comparators {
        let offset = COMPARATOR_CONFIG + index * COMPARATOR_STRIDE;
        let configuration = read(registers, offset);
        if configuration & COMPARATOR_INTERRUPT != 0 {
            write(registers, offset, configuration & !COMPARATOR_INTERRUPT);
            silenced |= 1_u32 << index;
        }
    }
    silenced
}

/// Puts back exactly the comparator interrupt enables [`silence`] cleared.
fn unsilence(registers: VirtAddr, silenced: u32) {
    for index in 0..MAPPED_COMPARATORS {
        if silenced & (1_u32 << index) != 0 {
            let offset = COMPARATOR_CONFIG + index * COMPARATOR_STRIDE;
            write(
                registers,
                offset,
                read(registers, offset) | COMPARATOR_INTERRUPT,
            );
        }
    }
}

/// Comparators in the block that the mapping actually reaches.
fn comparators(registers: VirtAddr, capabilities: u64) -> u64 {
    let claimed = ((capabilities >> COMPARATORS_SHIFT) & COMPARATORS_MASK) + 1;
    if claimed > MAPPED_COMPARATORS {
        warn!(
            "clock: the hpet at {:#x} claims {claimed} comparators, more than its own \
             {BLOCK_BYTES}-byte block has room for; leaving the last {} alone",
            registers.as_u64(),
            claimed - MAPPED_COMPARATORS,
        );
    }
    claimed.min(MAPPED_COMPARATORS)
}

/// The 64-bit register at `offset` in the block.
fn read(registers: VirtAddr, offset: u64) -> u64 {
    // SAFETY: `registers` is the base of a kilobyte this module mapped
    // read-write, and `offset` names a register inside it: a general register
    // by its own constant, or a comparator's by a stride whose index
    // `comparators` clamps to what the block holds. Both are eight-byte aligned
    // — the base by the check in `open`, the offsets by their own values.
    // Volatile because the block is device memory.
    unsafe { ptr_at(registers, offset).read_volatile() }
}

/// Writes the 64-bit register at `offset` in the block.
fn write(registers: VirtAddr, offset: u64, value: u64) {
    // SAFETY: as `read`. Every write here is a read-modify-write of one bit of
    // the register it names — the block's enable, or a comparator's interrupt
    // enable — so no field the clock does not own changes.
    unsafe { ptr_at(registers, offset).write_volatile(value) };
}

/// Where the register at `offset` is.
fn ptr_at(registers: VirtAddr, offset: u64) -> *mut u64 {
    (registers + offset).as_mut_ptr::<u64>()
}
