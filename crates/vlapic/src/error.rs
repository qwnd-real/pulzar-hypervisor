//! The error status register, and the write-then-read protocol that is the only
//! way to read it correctly.
//!
//! The register is 32 bits wide with bits 31:8 reserved, and it does not answer
//! with what the controller has noticed. It answers with what the guest's last
//! *write* latched. The write is a command rather than data — the value written
//! is discarded, and in x2APIC only zero may be written at all — and it does
//! three things at once: it clears whatever was previously latched, it latches
//! everything detected since the previous write, and it re-arms the error
//! interrupt. A guest that reads without writing first therefore reads a stale
//! answer, and one that reads twice with no write between them reads the same
//! answer twice.
//!
//! That is why the state here is two words rather than one. [`ErrorStatus`]
//! keeps a pending set, which every error the controller notices accumulates
//! into, and a latched word, which is all a read ever sees. Reserved bits
//! cannot appear in either, because [`Errors`] has no name for them.
//!
//! Both words are atomic, for different reasons. An error is recorded by
//! whichever processor notices it, and a badly formed interrupt aimed at this
//! controller is noticed by the sender rather than by the owner, so the pending
//! word is written from any processor. The latched word is only ever written by
//! the processor that owns the controller, but it is read alongside the pending
//! word, and making it an atomic too leaves one kind of access to reason about
//! instead of two.

use core::sync::atomic::{AtomicU32, Ordering};

use bitflags::bitflags;

bitflags! {
    /// The errors the controller distinguishes, one bit each.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct Errors: u32 {
        /// A checksum mismatch on a message this controller sent.
        ///
        /// Only processors whose controllers talk over the three-wire APIC bus
        /// can detect this; on the system bus of anything this hypervisor runs
        /// on the bit is reserved and nothing here ever sets it.
        const SEND_CHECKSUM = 1 << 0;
        /// A checksum mismatch on a message this controller received.
        ///
        /// Three-wire APIC bus only, and so never set here.
        const RECEIVE_CHECKSUM = 1 << 1;
        /// No processor accepted a message this controller sent.
        ///
        /// Three-wire APIC bus only, and so never set here.
        const SEND_ACCEPT = 1 << 2;
        /// No processor accepted a message this controller received.
        ///
        /// Three-wire APIC bus only, and so never set here.
        const RECEIVE_ACCEPT = 1 << 3;
        /// Software asked for a lowest-priority interprocessor interrupt on a
        /// controller that cannot send one.
        ///
        /// The request is refused before the message is formed, so nothing
        /// further about it is examined: an interrupt that is both refused here
        /// and carries an illegal vector sets this bit alone, never
        /// [`Errors::SEND_ILLEGAL_VECTOR`] as well.
        const REDIRECTABLE_IPI = 1 << 4;
        /// Software put a vector in 0..=15 in the interrupt command register or
        /// the self-interrupt register.
        ///
        /// Those sixteen vectors are the processor's own exceptions, and the
        /// architecture reserves them from the controller entirely.
        const SEND_ILLEGAL_VECTOR = 1 << 5;
        /// An interrupt arrived carrying a vector in 0..=15, whether from
        /// another controller or from one of this one's own local vector table
        /// entries.
        ///
        /// Such an interrupt is discarded rather than delivered: the controller
        /// never sets a request bit in 0..=15, so a guest cannot see one of
        /// these arrive except through this bit.
        const RECEIVE_ILLEGAL_VECTOR = 1 << 6;
        /// Software named a reserved address in the memory-mapped register page.
        ///
        /// This exists only in the memory-mapped face. Naming a reserved index
        /// through x2APIC is a general protection fault instead, and this bit is
        /// never set for it.
        const ILLEGAL_REGISTER_ADDRESS = 1 << 7;
    }
}

/// What the controller has noticed, and what a guest's read of the error status
/// register would answer with.
#[derive(Debug)]
pub(crate) struct ErrorStatus {
    pending: AtomicU32,
    latched: AtomicU32,
}

impl ErrorStatus {
    /// Nothing latched and nothing pending, which is what reset leaves this.
    pub(crate) const fn new() -> Self {
        Self {
            pending: AtomicU32::new(0),
            latched: AtomicU32::new(0),
        }
    }

    /// Records an error the controller noticed.
    ///
    /// The answer is whether this is the first error since the register was
    /// last re-armed, which is what decides whether an error interrupt
    /// should be raised. A storm must raise one interrupt rather than one
    /// per error: the architecture re-arms the triggering mechanism only on
    /// a write to the register, so every error after the first raises
    /// nothing until the guest writes again.
    ///
    /// No separate armed flag is needed, because the pending set already is one
    /// — it is empty exactly when the register has been written and nothing has
    /// been recorded since. Accumulating with a single read-modify-write is
    /// what makes that sound under concurrent records: of any number of
    /// processors recording at once, exactly one can observe the set empty,
    /// so exactly one is told to raise the interrupt.
    pub(crate) fn record(&self, errors: Errors) -> bool {
        let previous = self.pending.fetch_or(errors.bits(), Ordering::AcqRel);
        // Recording nothing arms nothing, so it must not claim the interrupt
        // even though it leaves an empty set behind it.
        previous == 0 && !errors.is_empty()
    }

    /// What a guest's read answers with: whatever the last write latched, not
    /// what has been noticed since.
    pub(crate) fn read(&self) -> u32 {
        self.latched.load(Ordering::Acquire)
    }

    /// The write half of the protocol, for a write of any value.
    ///
    /// Latches everything detected since the previous write, discarding what
    /// that write had latched, empties the pending set and thereby re-arms the
    /// error interrupt. Returns the newly latched value, which is what the next
    /// read will answer with.
    pub(crate) fn written(&self) -> u32 {
        let latched = self.pending.swap(0, Ordering::AcqRel);
        self.latched.store(latched, Ordering::Release);
        latched
    }

    /// Back to the reset state: nothing latched, nothing pending, armed.
    pub(crate) fn reset(&self) {
        self.pending.store(0, Ordering::Release);
        self.latched.store(0, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::{ErrorStatus, Errors};

    #[test]
    fn read_before_any_write_answers_zero() {
        let status = ErrorStatus::new();

        assert_eq!(status.read(), 0);
    }

    #[test]
    fn a_record_stays_invisible_until_a_write() {
        let status = ErrorStatus::new();
        status.record(Errors::RECEIVE_ILLEGAL_VECTOR);

        assert_eq!(status.read(), 0);
    }

    #[test]
    fn a_write_latches_what_was_recorded() {
        let status = ErrorStatus::new();
        let errors = Errors::SEND_ILLEGAL_VECTOR | Errors::ILLEGAL_REGISTER_ADDRESS;
        status.record(errors);

        assert_eq!(status.written(), errors.bits());
        assert_eq!(status.read(), errors.bits());
    }

    #[test]
    fn a_second_write_clears_what_the_first_latched() {
        let status = ErrorStatus::new();
        status.record(Errors::REDIRECTABLE_IPI);
        status.written();

        assert_eq!(status.written(), 0);
        assert_eq!(status.read(), 0);
    }

    #[test]
    fn only_the_first_error_after_a_write_arms_the_interrupt() {
        let status = ErrorStatus::new();

        assert!(status.record(Errors::SEND_ILLEGAL_VECTOR));
        assert!(!status.record(Errors::RECEIVE_ILLEGAL_VECTOR));

        status.written();

        assert!(status.record(Errors::ILLEGAL_REGISTER_ADDRESS));
    }
}
