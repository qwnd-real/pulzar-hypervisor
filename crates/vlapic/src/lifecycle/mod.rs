//! What becomes of a controller between one guest and the next, and what real
//! hardware is owed across it.
//!
//! Two things that look unrelated and are the same subject. A guest that is
//! reset stops existing: nothing it was in the middle of will be finished, and
//! nothing it was owed will be collected. So the two questions a lifecycle
//! boundary asks are what the new guest finds — [`settle`] — and what the machine
//! is still holding for the old one — [`ledger`].
//!
//! Every operation here is performed by a processor about itself, at an exit
//! boundary. That is what makes resetting a whole register file safe without a
//! lock, and it is what makes acknowledging real hardware legitimate: an
//! acknowledgement goes to whichever controller the processor issuing it is
//! running on.

pub(crate) mod arrival;
pub(crate) mod ledger;
pub(crate) mod settle;

use crate::{VlapicError, machine::current, registers::Vlapic};

/// What this processor should do before entering the guest again, having
/// applied whatever startup message arrived for it.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub fn settle() -> Result<settle::Resumption, VlapicError> {
    current().map(settle::applied)
}

/// Waits, without running the guest, until a startup message arrives for this
/// processor.
///
/// What a processor whose guest has been reset does instead of spinning: there
/// is nothing to run, and there will be nothing to run until another processor
/// starts this one, which may never happen. The processor halts, and the
/// interrupt that wakes it is either the doorbell that says a message arrived
/// or something unrelated — so a caller consults [`settle`] again rather than
/// assuming the wait ended for the reason it was entered.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub fn hold() -> Result<(), VlapicError> {
    let vlapic = current()?;
    // Said before the message is looked for, and the reason is the whole of why
    // this is not a bare halt: a sender that misses this flag is one whose
    // message the test below finds, and a test that misses the message is one
    // the sender's doorbell wakes.
    vlapic.set_away(true);
    descriptors::wait_until(|| vlapic.signalled());
    vlapic.set_away(false);
    Ok(())
}

/// Whether this processor's guest is running, rather than reset and waiting to
/// be started again.
///
/// Consulted on the exit path, where a startup message that arrived while the
/// guest was running is a reason to stop running it.
///
/// # Errors
///
/// As [`crate::read_msr`].
pub fn running() -> Result<bool, VlapicError> {
    current().map(Vlapic::running)
}
