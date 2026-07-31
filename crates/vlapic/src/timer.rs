//! The guest's timer, which is the real timer.
//!
//! The local controller's timer is the one part of it that is not emulated at
//! all. The guest's divide, count and mode are programmed straight onto the
//! hardware and the hardware counts them, because there is nothing to be gained
//! by counting them again in software: a timer is a decrementing register and a
//! comparison, and this hypervisor has no reason to lie about either.
//!
//! # The vector is not the guest's
//!
//! What is *not* passed through is the vector. The real entry is programmed
//! with a vector this hypervisor claimed for itself, and the guest's own vector
//! is injected when one arrives.
//!
//! That indirection is the whole reason the arrival is unambiguous. Vectors on
//! this machine are shared between the guest's devices, the hypervisor's own
//! interprocessor interrupts, and the controller's spurious and error vectors —
//! so an interrupt arriving on a number the guest chose says nothing about who
//! it was for. An interrupt arriving on a number only this module ever
//! programmed says exactly one thing.
//!
//! # Deadline mode
//!
//! In deadline mode the count registers stop meaning anything: writes to the
//! initial count are ignored and the current count reads as zero. The deadline
//! itself is a model-specific register the guest writes, which is intercepted
//! and passed through, and the hardware disarms itself when it fires.

use apic::{Divisor, TimerMode as HardwareMode};
use descriptors::Vector;
use log::warn;

use crate::{
    lvt::{Entry, TimerMode},
    state::Vlapic,
};

/// What the guest's timer has left to count.
///
/// Read from the hardware, because the hardware is what is counting. In
/// deadline mode the architecture defines this as reading zero.
pub(crate) fn remaining(vlapic: &Vlapic) -> u32 {
    if mode_of(vlapic) == Some(TimerMode::Deadline) {
        return 0;
    }
    apic::local()
        .and_then(|local| local.timer().remaining())
        .unwrap_or(0)
}

/// Brings the real timer into agreement with what the guest has programmed.
///
/// Called whenever the guest touches the timer's entry, its divide or its
/// count. Which of those it touched does not matter: the whole configuration is
/// short, and reprogramming all of it is both simpler and immune to a guest
/// that writes the registers in an order nothing anticipated.
///
/// Disarming rather than failing is the answer to everything unprogrammable
/// here. A masked entry, a zero count, a mode the hardware does not offer — all
/// of them mean the guest is owed no interrupt, and a timer left running would
/// deliver one.
pub(crate) fn reprogram(vlapic: &Vlapic, vector: Vector) {
    let Ok(timer) = apic::local().map(apic::LocalApic::timer) else {
        return;
    };
    let entry = vlapic.lvt(Entry::Timer);
    let count = vlapic.timer_initial();
    let outcome = match mode_of(vlapic) {
        // A masked entry delivers nothing, and neither should the hardware.
        _ if entry.masked() => timer.disarm(),
        // Deadline mode is armed by the model-specific register rather than by
        // a count, so there is nothing to start here — but the entry still has
        // to say deadline before a write to that register means anything.
        Some(TimerMode::Deadline) => arm_deadline(vlapic, vector),
        // A zero count stops the timer in both counting modes rather than
        // firing immediately.
        _ if count == 0 => timer.disarm(),
        Some(TimerMode::OneShot) => {
            timer.arm(vector, HardwareMode::OneShot, divisor(vlapic), count)
        }
        Some(TimerMode::Periodic) => {
            timer.arm(vector, HardwareMode::Periodic, divisor(vlapic), count)
        }
        // The architecture reserves the fourth encoding, and a controller given
        // it does nothing.
        None => timer.disarm(),
    };
    if let Err(error) = outcome {
        warn!(
            "vlapic: {} could not program its timer: {error}",
            vlapic.index()
        );
    }
}

/// Arms the deadline the guest last wrote, if it has written one.
fn arm_deadline(vlapic: &Vlapic, vector: Vector) -> Result<(), apic::ApicError> {
    let timer = apic::local()?.timer();
    match vlapic.timer_deadline() {
        0 => timer.disarm(),
        deadline => timer.arm_deadline(vector, deadline),
    }
}

/// Which counting mode the guest's entry selects, or `None` for the encoding
/// the architecture reserves.
fn mode_of(vlapic: &Vlapic) -> Option<TimerMode> {
    TimerMode::from_bits(vlapic.lvt(Entry::Timer).timer_mode())
}

/// How far the guest asked for the clock to be divided.
///
/// The encoding is three bits that are not adjacent — bit two sits between them
/// and is reserved — and every one of the eight it can name is a divisor the
/// hardware offers, so this cannot fail.
fn divisor(vlapic: &Vlapic) -> Divisor {
    let value = vlapic.timer_divide();
    match (value & 0b11) | ((value & 0b1000) >> 1) {
        0b000 => Divisor::By2,
        0b001 => Divisor::By4,
        0b010 => Divisor::By8,
        0b011 => Divisor::By16,
        0b100 => Divisor::By32,
        0b101 => Divisor::By64,
        0b110 => Divisor::By128,
        _ => Divisor::By1,
    }
}
