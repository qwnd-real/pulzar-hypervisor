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
/// here: a zero count, a mode the hardware does not offer, a deadline nobody
/// has written. All of them mean there is nothing to count, and a timer left
/// running would deliver an interrupt the guest is not owed.
///
/// # A masked entry still counts
///
/// The mask bit suppresses the interrupt and nothing else, and getting that
/// wrong stops an operating system dead. Masking the entry, writing a count and
/// watching it fall against a clock it already trusts is exactly how software
/// measures what its timer's rate is — it is the first thing an operating
/// system does with the timer and it wants no interrupt while doing it. So a
/// masked entry is programmed onto the hardware masked rather than disarmed,
/// and the count the guest reads back is the real one falling at the real rate.
pub(crate) fn reprogram(vlapic: &Vlapic, vector: Vector) {
    let Ok(timer) = apic::local().map(apic::LocalApic::timer) else {
        return;
    };
    let entry = vlapic.lvt(Entry::Timer);
    let count = vlapic.timer_initial();
    let outcome = match counting(vlapic) {
        // Deadline mode is armed by the model-specific register rather than by
        // a count, so there is nothing to start here — but the entry still has
        // to say deadline before a write to that register means anything. A
        // masked one is left disarmed, because unlike the counting modes it
        // leaves the guest nothing to read: the current count reads zero in
        // deadline mode whatever the timer is doing.
        None if mode_of(vlapic) == Some(TimerMode::Deadline) && !entry.masked() => {
            arm_deadline(vlapic, vector)
        }
        // Deadline mode with nothing to deliver to, or the fourth encoding,
        // which the architecture reserves and a controller given it does
        // nothing with.
        None => timer.disarm(),
        // A zero count stops the timer in both counting modes rather than
        // firing immediately.
        Some(_) if count == 0 => timer.disarm(),
        Some(mode) if entry.masked() => timer.count_down(mode, divisor(vlapic), count),
        Some(mode) => timer.arm(vector, mode, divisor(vlapic), count),
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

/// Which mode the hardware should count in, or `None` where there is no count
/// at all — deadline mode, which is armed by a register rather than by a
/// number, and the encoding the architecture reserves.
fn counting(vlapic: &Vlapic) -> Option<HardwareMode> {
    match mode_of(vlapic)? {
        TimerMode::OneShot => Some(HardwareMode::OneShot),
        TimerMode::Periodic => Some(HardwareMode::Periodic),
        TimerMode::Deadline => None,
    }
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
