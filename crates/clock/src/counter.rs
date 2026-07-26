//! A running hardware counter, whatever the hardware underneath it is.
//!
//! Three kinds of counter can serve pulzar — the processor's timestamp counter,
//! a memory-mapped register in an event timer, and a port belonging to the
//! chipset — and everything above this module wants the same three things from
//! each: read it, know how fast it counts, and know where it wraps. So there is
//! one type here and the differences between the three are a register
//! description and a width.
//!
//! Reads are volatile and never elided. A counter's value is the one thing in a
//! machine that changes without anything having written it, so a compiler that
//! reused the last one it saw would be right about the memory model and wrong
//! about the world.

use core::{
    fmt::{self, Display, Formatter},
    ptr,
};

use x86_64::{VirtAddr, instructions::port::Port};

use crate::Frequency;

/// A counter that is running and can be read.
#[derive(Clone, Copy, Debug)]
pub struct Counter {
    kind: Kind,
    register: Register,
    frequency: Frequency,
    bits: u32,
}

impl Counter {
    /// Which piece of hardware this is.
    #[must_use]
    pub const fn kind(&self) -> Kind {
        self.kind
    }

    /// How fast it counts.
    #[must_use]
    pub const fn frequency(&self) -> Frequency {
        self.frequency
    }

    /// Bits of its value that advance, and so where it wraps.
    #[must_use]
    pub const fn bits(&self) -> u32 {
        self.bits
    }

    /// A counter reached through `register`, ticking at `frequency`, whose
    /// value is `bits` wide.
    ///
    /// # Safety
    ///
    /// A memory-mapped register must stay mapped, readable, and aligned for the
    /// width it is read at, for as long as this counter exists. A port must be
    /// one whose reads have no effect on the machine. Both are what make every
    /// later [`Counter::read`] sound without a further check.
    pub(crate) const unsafe fn new(
        kind: Kind,
        register: Register,
        frequency: Frequency,
        bits: u32,
    ) -> Self {
        Self {
            kind,
            register,
            frequency,
            bits,
        }
    }

    /// The counter's current value.
    pub(crate) fn read(&self) -> u64 {
        match self.register {
            Register::Timestamp => processor::timestamp(),
            // SAFETY: `new`'s caller guarantees this address stays mapped,
            // readable and eight-byte aligned for as long as this counter
            // exists. The read is volatile because the value changes with time
            // rather than because something wrote it.
            Register::Memory64(at) => unsafe { ptr::read_volatile(at.as_ptr::<u64>()) },
            // SAFETY: as above, at four bytes and four-byte alignment.
            Register::Memory32(at) => u64::from(unsafe { ptr::read_volatile(at.as_ptr::<u32>()) }),
            Register::Port(port) => {
                let mut port = Port::<u32>::new(port);
                // SAFETY: `new`'s caller guarantees that reading this port has
                // no effect on the machine, and pulzar runs at ring 0, where
                // port access is permitted.
                u64::from(unsafe { port.read() })
            }
        }
    }

    /// Ticks from `earlier` to `later`, wraps included.
    ///
    /// Correct across one wrap and no more, which is what
    /// [`Counter::can_span`] exists to establish before a measurement rather
    /// than after it.
    ///
    /// A counter as wide as the value holding it is the exception, and
    /// deliberately so: wrapping one takes a century of uptime, so a reading
    /// below the one it is compared against is far more likely a counter that
    /// went *backwards* — another processor's timestamp counter, which counts
    /// at the same rate as this one but need not have started from the same
    /// value. Answering zero is wrong by that offset. Answering with a wrap
    /// would be wrong by five hundred years.
    pub(crate) const fn difference(&self, earlier: u64, later: u64) -> u64 {
        if self.bits >= u64::BITS {
            return later.saturating_sub(earlier);
        }
        later.wrapping_sub(earlier) & self.mask()
    }

    /// Nanoseconds the counter has advanced since it read `earlier`.
    pub(crate) fn nanos_since(&self, earlier: u64) -> u64 {
        self.frequency.nanos(self.difference(earlier, self.read()))
    }

    /// Whether a run of `ticks` fits inside the counter's width, so that a
    /// difference across it is that run and not that run less a wrap.
    pub(crate) const fn can_span(&self, ticks: u64) -> bool {
        ticks < self.mask()
    }

    /// Mask of the bits that advance.
    const fn mask(&self) -> u64 {
        if self.bits >= u64::BITS {
            u64::MAX
        } else {
            (1 << self.bits) - 1
        }
    }
}

/// Which piece of hardware a counter is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The processor's own timestamp counter.
    Tsc,
    /// The main counter of a high precision event timer.
    Hpet,
    /// The ACPI power management timer.
    PmTimer,
}

impl Display for Kind {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Tsc => "timestamp counter",
            Self::Hpet => "hpet",
            Self::PmTimer => "power management timer",
        })
    }
}

/// How a counter's value is reached.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Register {
    /// The timestamp counter, which is an instruction rather than a register.
    Timestamp,
    /// A 64-bit memory-mapped register.
    Memory64(VirtAddr),
    /// A 32-bit memory-mapped register.
    Memory32(VirtAddr),
    /// A 32-bit I/O port.
    Port(u16),
}
