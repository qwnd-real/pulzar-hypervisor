//! How a processor that has stopped looking at its controller is made to look
//! again.
//!
//! Accepting an interrupt into another processor's controller is a bit set in an
//! atomic, and it costs nothing. What costs something is that the target may
//! have stopped looking — inside the guest, or halted waiting to be started —
//! and will not look again until something makes it. So a real interrupt is sent
//! to force one.
//!
//! That pairing is where a lost wakeup would live, and the order on both sides is
//! what stops one:
//!
//! - Here: set the request bit, *then* read whether the target is away.
//! - There: store that it is away, *then* re-read what has been left for it.
//!
//! Both stores are sequentially consistent, so at least one side sees the other.
//! If this side misses the flag, the target's re-read finds the bit; if the
//! target's re-read misses the bit, this side sees the flag and sends the
//! interrupt. There is no interleaving in which both miss.
//!
//! And a doorbell that arrives while the target is between `CLGI` and `VMRUN` is
//! not lost either: the interrupt is held by the cleared global interrupt flag,
//! and the maskable-interrupt intercept turns it into an immediate exit on entry.

use core::num::NonZeroU64;

use cpu::CpuIndex;
use log::{trace, warn};
use spin::Once;

use crate::{VlapicError, registers::{Vlapic, error::Errors}};

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
/// A doorbell that could not be sent is retried, because the alternative is a
/// processor that stalls until something unrelated happens to wake it — for a
/// halted one, possibly never. The retry is bounded and the failure is recorded
/// against the sender's error status afterwards: the architecture's nearest
/// equivalent is a message no processor accepted, which is exactly what this
/// is.
pub(super) fn nudge(from: &Vlapic, target: &Vlapic) {
    if target.index() == from.index() || !target.away() {
        return;
    }
    for _ in 0..DOORBELL_ATTEMPTS {
        match doorbell(target.index()) {
            Ok(()) => return,
            Err(error) => trace!(
                "vlapic: {} could not interrupt {}, trying again: {error}",
                from.index(),
                target.index()
            ),
        }
    }
    // The request itself is left published. It is real — the target will act on
    // it at its next exit — and what has been lost is only the prompt that would
    // have made that exit happen sooner.
    from.errors().record(Errors::SEND_ACCEPT);
    warn!(
        "vlapic: {} left {} un-interrupted; it will not act until it exits for another reason",
        from.index(),
        target.index()
    );
}

/// How many times a doorbell is tried before the target is left to notice on
/// its own.
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
