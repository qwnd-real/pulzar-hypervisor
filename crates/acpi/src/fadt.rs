//! The Fixed ACPI Description Table, read for its timer and nothing else.
//!
//! The FADT is the largest table ACPI defines and describes most of a machine's
//! fixed hardware: the power management registers, the sleep states, the
//! embedded controller, the boot architecture flags. One field of it is parsed
//! here — where the power management timer is — because one field of it is what
//! pulzar uses. The table's own address stays in the directory, so whatever
//! needs the rest later can read the rest later.
//!
//! # Why the timer
//!
//! It is the fallback reference for calibrating the timestamp counter on a
//! machine with no usable HPET. Two properties make it worth having in that
//! role: its frequency is fixed by ACPI at 3.579545 MHz — a third of the
//! original PC's colour burst clock, which is why a hypervisor still divides by
//! it — and it cannot be turned off, so reading it needs nothing set up first.
//!
//! What it cannot do is keep time. The counter is 24 or 32 bits wide, so it
//! wraps every few seconds or every few minutes, and only something polling it
//! more often than that could tell one wrap from two.

use log::{info, warn};

use crate::{
    AcpiError,
    gas::{self, GenericAddress},
    raw::Fields,
};

/// Offset of the timer's block address in the pre-2.0 form: a 32-bit I/O port.
const TIMER_BLOCK: usize = 76;

/// Offset of the number of bytes the timer's block decodes.
const TIMER_LENGTH: usize = 91;

/// Offset of the fixed feature flags.
const FLAGS: usize = 112;

/// Offset of the generic address that supersedes [`TIMER_BLOCK`] where firmware
/// fills it in.
const EXTENDED_TIMER_BLOCK: usize = 208;

/// Flag: the counter is 32 bits wide rather than 24.
const TIMER_VALUE_EXTENDED: u32 = 1 << 8;

/// Bytes the timer's register block decodes on a machine that has one. Any
/// other value means the machine has none.
const TIMER_BYTES: u8 = 4;

/// Bits the timer's register block is wide, whatever part of it counts.
const REGISTER_BITS: u8 = 32;

/// What pulzar keeps from the fixed hardware description.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fadt {
    pm_timer: Option<PmTimer>,
}

impl Fadt {
    /// The power management timer, if the machine has one.
    #[must_use]
    pub const fn pm_timer(&self) -> Option<PmTimer> {
        self.pm_timer
    }

    /// Logs what was kept.
    pub fn describe(&self, who: &str) {
        match self.pm_timer {
            Some(timer) => info!(
                "{who}: fadt power management timer at {}, {}-bit counter at {} Hz",
                timer.register,
                timer.bits,
                PmTimer::FREQUENCY,
            ),
            None => info!("{who}: fadt describes no power management timer"),
        }
    }

    /// Parses the table.
    ///
    /// The generic address wins over the legacy port wherever both are present:
    /// firmware that fills in both is required to describe the same register
    /// twice, and only the generic form can say that the register is in memory
    /// rather than in I/O space.
    ///
    /// # Errors
    ///
    /// [`AcpiError::Truncated`] if the table is shorter than the fixed fields
    /// every revision of it has.
    pub(crate) fn parse(table: &Fields<'_>) -> Result<Self, AcpiError> {
        let decoded = table.u8(TIMER_LENGTH)?;
        if decoded != TIMER_BYTES {
            if decoded != 0 {
                warn!(
                    "acpi: the fadt at {:#x} says its timer block decodes {decoded} bytes rather \
                     than {TIMER_BYTES}; treating the timer as absent",
                    table.at()
                );
            }
            return Ok(Self { pm_timer: None });
        }

        let register = match extended_register(table)? {
            Some(register) => register,
            None => GenericAddress::io(table.u32(TIMER_BLOCK)?, REGISTER_BITS),
        };
        let bits = if table.u32(FLAGS)? & TIMER_VALUE_EXTENDED == 0 {
            PmTimer::NARROW_BITS
        } else {
            PmTimer::WIDE_BITS
        };
        Ok(Self {
            pm_timer: (register.address() != 0).then_some(PmTimer { register, bits }),
        })
    }
}

/// The power management timer's counter register.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PmTimer {
    register: GenericAddress,
    bits: u32,
}

impl PmTimer {
    /// Ticks per second, which ACPI fixes rather than leaving for firmware to
    /// report.
    pub const FREQUENCY: u64 = 3_579_545;

    /// Bits of the counter on a machine whose firmware does not set the
    /// extended timer flag.
    pub const NARROW_BITS: u32 = 24;

    /// Bits of the counter on a machine whose firmware does.
    pub const WIDE_BITS: u32 = 32;

    /// Where the register is.
    #[must_use]
    pub const fn register(&self) -> GenericAddress {
        self.register
    }

    /// Bits of the register that count, which is [`PmTimer::NARROW_BITS`] or
    /// [`PmTimer::WIDE_BITS`].
    ///
    /// The block itself always decodes four bytes; this is how much of the
    /// value in them advances, and therefore where it wraps.
    #[must_use]
    pub const fn bits(&self) -> u32 {
        self.bits
    }
}

/// The generic address form of the timer's block, if the table is long enough
/// to carry one and firmware filled it in.
///
/// A table that stops before the field is an ACPI 1.0 table, where the field
/// does not exist; a zero address is firmware declining to use it. Both mean
/// the legacy port is the only description there is.
fn extended_register(table: &Fields<'_>) -> Result<Option<GenericAddress>, AcpiError> {
    if table.size() < EXTENDED_TIMER_BLOCK + gas::ADDRESS_BYTES {
        return Ok(None);
    }
    let register = GenericAddress::parse(table, EXTENDED_TIMER_BLOCK)?;
    Ok((register.address() != 0).then_some(register))
}
