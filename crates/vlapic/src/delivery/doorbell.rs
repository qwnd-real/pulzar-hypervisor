//! How a processor that has stopped looking at its controller is made to look
//! again.
//!
//! Accepting an interrupt into another processor's controller is a bit set in
//! an atomic, and it costs nothing. What costs something is that the target may
//! have stopped looking — inside the guest, or halted waiting to be started —
//! and will not look again until something makes it. So a real interrupt is
//! sent to force one.
//!
//! That pairing is where a lost wakeup would live, and the order on both sides
//! is what stops one:
//!
//! - Here: set the request bit, *then* read whether the target is away.
//! - There: store that it is away, *then* re-read what has been left for it.
//!
//! # Why that is enough, in the model rather than on this processor
//!
//! All four of those accesses are sequentially consistent, and that is what the
//! argument needs. The memory model gives sequentially consistent operations —
//! and only those — a single total order `S` that agrees with happens-before
//! and with each object's modification order. Suppose this side's load of the
//! flag reads "not away": then it reads a value not later than the target's
//! store in the flag's modification order, so it precedes nothing that would
//! let it be placed after that store, and the store is after it in `S`. Program
//! order puts the request-bit store before the flag load on this side and the
//! flag store before the request-bit scan on the target's, and `S` respects
//! both. Composing them puts the request-bit store before the target's scan in
//! `S` — and two sequentially consistent accesses to the same word must agree
//! with that word's modification order, so the scan sees the bit. If instead
//! this side reads "away", it sends the doorbell. There is no interleaving in
//! which both miss.
//!
//! A release read-modify-write and an acquire load do not give that, which is
//! what the halves of this used to be. A release RMW's load half is relaxed and
//! neither operation orders a store against a later load, so nothing in the
//! model forbids both sides being early. On x86 nothing can be: `lock or`
//! drains the store buffer and a sequentially consistent store compiles to
//! `xchg`, so both sides are already full barriers and both spellings emit the
//! same instructions. The processor was doing the work the model had not been
//! asked for — and would have gone on doing it right up until somebody relaxed
//! an ordering the comment had licensed.
//!
//! And a doorbell that arrives while the target is between `CLGI` and `VMRUN`
//! is not lost either: the interrupt is held by the cleared global interrupt
//! flag, and the maskable-interrupt intercept turns it into an immediate exit
//! on entry.

use core::num::NonZeroU64;

use apic::ApicError;
use cpu::CpuIndex;
use ipi::IpiError;
use log::{trace, warn};
use spin::Once;

use crate::{VlapicError, registers::Vlapic};

/// Acquires the interrupt this hypervisor rings a processor with.
///
/// Separate from [`publish`] because the two happen at different points of
/// installation: acquiring a vector can fail and is done before anything is
/// published, and publishing cannot fail and is done once everything that can
/// has succeeded.
///
/// # Errors
///
/// Whatever acquiring an interprocessor interrupt reported.
pub(crate) fn acquire() -> Result<ipi::Ipi, ipi::IpiError> {
    ipi::register(rung, merge)
}

/// Publishes the doorbell, after which a processor inside the guest can be made
/// to leave it.
pub(crate) fn publish(doorbell: ipi::Ipi) {
    DOORBELL.call_once(|| doorbell);
}

/// Makes a target that has stopped looking at its controller look at it again.
///
/// Two states need this and they need it for the same reason: a processor
/// inside the guest, and one halted waiting to be started. Neither will notice
/// a bit that has just been set until something interrupts it.
///
/// The sender never needs one of these for itself: it is already outside the
/// guest — it is executing this — and it consults its own controller before it
/// goes back in.
///
/// A doorbell that could not be sent is retried for the one failure a retry can
/// cure, and given up on at once for the rest: a controller whose command
/// register has not drained yet will drain, while a processor that never
/// attached or an identifier the host's current face cannot name will still be
/// that on the third attempt. Retrying those cost three real attempts and left
/// the target's mailbox owing three answers for work never done.
///
/// Giving up is not losing the interrupt. The request bit is set and stays set
/// — the target acts on it at its next exit — and what has been lost is only
/// the prompt that would have made that exit happen sooner. Nothing is recorded
/// in the sender's error status for it, because nothing the architecture
/// reports happened: the message *was* accepted, by a controller that has it.
/// What failed is this hypervisor's own way of making a processor look, so it
/// is said in the log, where the host's failures belong.
pub(super) fn nudge(from: &Vlapic, target: &Vlapic) {
    if target.index() == from.index() || !target.away() {
        return;
    }
    for attempt in 0..DOORBELL_ATTEMPTS {
        let error = match doorbell(target.index()) {
            Ok(()) => return,
            Err(error) => error,
        };
        if attempt + 1 == DOORBELL_ATTEMPTS || !worth_retrying(error) {
            warn!(
                "vlapic: {} left {} un-interrupted; it will not act until it exits for another \
                 reason: {error}",
                from.index(),
                target.index()
            );
            return;
        }
        trace!(
            "vlapic: {} could not interrupt {}, trying again: {error}",
            from.index(),
            target.index()
        );
    }
}

/// Whether another attempt at a doorbell could answer differently.
///
/// One failure can: a command that has not left the sending controller yet is a
/// controller that is busy, and it will not be busy for long. Everything else
/// is a fact about the machine — a processor that never attached, a doorbell
/// that was never published, an identifier the face in use cannot name — and is
/// the same fact on every attempt.
const fn worth_retrying(error: VlapicError) -> bool {
    matches!(
        error,
        VlapicError::Ipi(IpiError::Apic(ApicError::CommandStuck))
    )
}

/// How many times a doorbell that keeps failing in a way a retry could cure is
/// tried before the target is left to notice on its own.
const DOORBELL_ATTEMPTS: u32 = 3;

fn doorbell(target: CpuIndex) -> Result<(), VlapicError> {
    let doorbell = DOORBELL.get().ok_or(VlapicError::NotInstalled)?;
    doorbell.send(target, RING)?;
    Ok(())
}

/// How a processor inside the guest is made to leave it.
static DOORBELL: Once<ipi::Ipi> = Once::new();

/// What a doorbell carries, which is nothing.
///
/// The arrival is the whole message: it forces the target out of the guest, and
/// what to do about that is decided by reading the controller, not by reading a
/// payload. A constant is needed only because a send must carry something.
const RING: NonZeroU64 = NonZeroU64::new(1).unwrap();

/// What runs on a processor a doorbell was sent to.
///
/// Deliberately empty. Ringing it has already done the only thing it was for —
/// the guest has left, and the exit loop consults the controller on its way
/// back in.
fn rung(_: ipi::Request) {}

/// How two outstanding doorbells become one.
///
/// They carry nothing, so there is nothing to combine, and a processor that
/// left the guest once has left it for both.
fn merge(first: NonZeroU64, _: NonZeroU64) -> NonZeroU64 {
    first
}
