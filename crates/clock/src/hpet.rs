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

use log::info;
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

/// Capabilities: the main counter is 64 bits wide rather than 32.
const COUNTER_64BIT: u64 = 1 << 13;

/// Bits the tick period, in femtoseconds, is shifted by in the capabilities
/// register.
const PERIOD_SHIFT: u32 = 32;

/// The longest tick period the specification permits: 100 ns, which is a 10 MHz
/// counter. A block reporting a slower one is describing itself impossibly.
const MAX_PERIOD_FEMTOS: u64 = 100_000_000;

/// Configuration: the main counter advances.
const ENABLE: u64 = 1 << 0;

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
    // SAFETY: these are device registers firmware described and nothing else in
    // this address space maps them: they lie outside every range the memory map
    // calls memory, so no allocator owns them and the direct map does not reach
    // them. Uncached-minus is what device registers need, since a cached alias
    // would answer a counter read from a cache line.
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
            started: started.then_some(registers),
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

/// Stops the main counter, which is what a timer found stopped is owed.
pub(crate) fn stop(registers: VirtAddr) {
    let configuration = read(registers, CONFIGURATION);
    write(registers, CONFIGURATION, configuration & !ENABLE);
    info!("clock: stopped the hpet again, as firmware had left it");
}

/// Reads what the block says about itself, starting it if it is not running.
///
/// Returns the counter and whether the enable bit had to be set, which is what
/// says whether the machine is owed a stop.
fn probe(registers: VirtAddr) -> Result<(Counter, bool), ClockError> {
    let capabilities = read(registers, CAPABILITIES);
    let femtos = capabilities >> PERIOD_SHIFT;
    let frequency = Frequency::from_period_femtos(femtos)
        .filter(|_| femtos <= MAX_PERIOD_FEMTOS)
        .ok_or(ClockError::HpetPeriod { femtos })?;

    let counter = registers + MAIN_COUNTER;
    let (register, bits) = if capabilities & COUNTER_64BIT == 0 {
        (Register::Memory32(counter), u32::BITS)
    } else {
        (Register::Memory64(counter), u64::BITS)
    };

    let configuration = read(registers, CONFIGURATION);
    let stopped = configuration & ENABLE == 0;
    if stopped {
        write(registers, CONFIGURATION, configuration | ENABLE);
        info!("clock: firmware had left the hpet stopped; started it");
    }

    // SAFETY: the main counter is inside the block this module just mapped, at
    // an offset that keeps it aligned for its width, and the mapping travels
    // beside the counter in the `Borrowed` that owns both — so it outlives every
    // read. Reading a counter has no effect on the timer.
    let counter = unsafe { Counter::new(Kind::Hpet, register, frequency, bits) };
    Ok((counter, stopped))
}

/// The 64-bit register at `offset` in the block.
fn read(registers: VirtAddr, offset: u64) -> u64 {
    // SAFETY: `registers` is the base of a kilobyte this module mapped
    // read-write, `offset` is one of three constants inside it, and both are
    // eight-byte aligned — the base by the check in `open`, the offsets by
    // their own values. Volatile because the block is device memory.
    unsafe { ptr_at(registers, offset).read_volatile() }
}

/// Writes the 64-bit register at `offset` in the block.
fn write(registers: VirtAddr, offset: u64, value: u64) {
    // SAFETY: as `read`, and the only register written is the configuration
    // one, whose enable bit is the only bit this crate changes.
    unsafe { ptr_at(registers, offset).write_volatile(value) };
}

/// Where the register at `offset` is.
fn ptr_at(registers: VirtAddr, offset: u64) -> *mut u64 {
    (registers + offset).as_mut_ptr::<u64>()
}
