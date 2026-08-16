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
//!
//! # Which errors exist at all
//!
//! Five of the eight bits the register has ever had, because the other three
//! are reserved on the processor this hypervisor presents: the two checksum
//! bits and the redirectable-interrupt bit. They are not named here, so nothing
//! can record one, a firmware capture cannot carry one through, and a guest
//! that checks that its processor's reserved bits read zero is not
//! contradicted. What a guest is told is only ever a condition its own `CPUID`
//! says its processor can report.
//!
//! # Raising the interrupt is a separate thing from recording the error
//!
//! Recording is what this module does; raising is
//! [`crate::delivery::error`]'s. The two are apart because the answer to
//! "should an error interrupt be raised" belongs to the controller the error
//! was recorded *on*, which is not always the processor that noticed it — so
//! what a record answers with is an obligation naming that controller, and
//! there is exactly one function that can discharge it.

use core::sync::atomic::{AtomicU32, Ordering};

use bitflags::bitflags;

use crate::registers::Vlapic;

bitflags! {
    /// The errors this controller distinguishes, one bit each.
    ///
    /// Three of the register's eight bits are absent, and the absence is the
    /// point: they are reserved on the processor being presented, so a bit with
    /// no name here is a bit no path can record and no capture can seed.
    ///
    /// - Bits 1:0, a checksum mismatch on a message sent or received. Only a
    ///   controller talking over the three-wire APIC bus can detect one, and
    ///   nothing this hypervisor runs on has that bus.
    /// - Bit 4, software asking for a lowest-priority interprocessor interrupt
    ///   on a controller that cannot send one. That is an Intel definition; the
    ///   processor presented here reserves the bit, so such a request is refused
    ///   with nothing recorded. It is refused before the message is formed, so
    ///   an interrupt that is both refused for the mode and carries an illegal
    ///   vector reports neither.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct Errors: u32 {
        /// No processor accepted a message this controller sent.
        ///
        /// Two things reach it, and both really are a message nobody took: one
        /// aimed at a processor this hypervisor does not run, and a redirectable
        /// one offered to every processor it named and refused by all of them.
        ///
        /// A doorbell that could not be sent is deliberately *not* one of them.
        /// The message was accepted — the request bit is set in a controller that
        /// has it — and what failed is this hypervisor's own way of making the
        /// target look at it sooner, which is not a condition the architecture
        /// has a bit for.
        const SEND_ACCEPT = 1 << 2;
        /// No processor accepted a message this controller received.
        ///
        /// Nothing here sets it: a message this controller receives is one it
        /// either accepts or refuses on its own behalf, and there is no third
        /// party to have refused it. It is named because the presented processor
        /// defines it, so a controller seeded from firmware's own register has to
        /// be able to carry one firmware left there.
        const RECEIVE_ACCEPT = 1 << 3;
        /// Software put a vector in 0..=15 in the interrupt command register or
        /// the self-interrupt register.
        ///
        /// Those sixteen vectors are the processor's own exceptions, and the
        /// architecture reserves them from the controller entirely.
        const SEND_ILLEGAL_VECTOR = 1 << 5;
        /// An interrupt arrived carrying a vector in 0..=15, whether from
        /// another controller, from one of this one's own local vector table
        /// entries, or from a self-interrupt.
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

impl Vlapic {
    /// Records an error this controller noticed, and answers the obligation to
    /// raise its error interrupt when this error is the one that arms it.
    ///
    /// The only way to record one, because the answer is about *this*
    /// controller and not about the processor that noticed the error. The two
    /// are not always the same: an interrupt aimed at a controller is examined
    /// by the sender, so a badly formed one is recorded here from another
    /// processor's context — and raising the interrupt there means publishing
    /// into this register file and making this processor look, neither of which
    /// a bare answer to the caller could express.
    ///
    /// So the answer names the controller it is about, and
    /// [`crate::delivery::error`] is the one thing that can discharge it.
    pub(crate) fn record_error(&self, errors: Errors) -> Option<Raise<'_>> {
        self.errors().record(errors).then_some(Raise(self))
    }
}

/// The obligation to raise one controller's error interrupt.
///
/// A value rather than a boolean, and one that borrows the controller it is
/// about, so that it can neither be dropped without a diagnostic nor discharged
/// against the wrong controller. Both were real: the answer this replaces was
/// computed at every recorder and read at none of them, which is how a
/// controller that advertises a writable, present error entry came to have
/// nothing that ever delivered its vector.
#[must_use = "an error that armed the error interrupt has to raise it or say why not"]
pub(crate) struct Raise<'a>(&'a Vlapic);

impl<'a> Raise<'a> {
    /// The controller whose error interrupt is owed, taking the obligation with
    /// it.
    ///
    /// Consuming, because there is one obligation and it is discharged once: a
    /// borrow would leave it in the caller's hands to be discharged again.
    pub(crate) const fn owed(self) -> &'a Vlapic {
        self.0
    }
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
    ///
    /// Accumulating *before* testing is also what bounds an error raised while
    /// an error is being raised: a nested record on the same controller always
    /// finds the set non-empty and answers no.
    ///
    /// This is not the answer to "should the host say something about it": a
    /// guest empties the pending set whenever it likes, so it can re-arm this
    /// as fast as it can write a register. What a log line is latched on is
    /// [`crate::machine::diagnostics`]'s, which nothing but a fresh controller
    /// clears.
    #[must_use]
    fn record(&self, errors: Errors) -> bool {
        let previous = self.pending.fetch_or(errors.bits(), Ordering::AcqRel);
        // Recording nothing arms nothing, so it must not claim the interrupt
        // even though it leaves an empty set behind it.
        previous == 0 && !errors.is_empty()
    }

    /// What a guest's read answers with: whatever the last write latched, not
    /// what has been noticed since.
    #[must_use]
    pub(crate) fn read(&self) -> u32 {
        self.latched.load(Ordering::Acquire)
    }

    /// The write half of the protocol, for a write of any value.
    ///
    /// Latches everything detected since the previous write, discarding what
    /// that write had latched, empties the pending set and thereby re-arms the
    /// error interrupt.
    pub(crate) fn written(&self) {
        let latched = self.pending.swap(0, Ordering::AcqRel);
        self.latched.store(latched, Ordering::Release);
    }

    /// Back to the reset state: nothing latched, nothing pending, armed.
    pub(crate) fn reset(&self) {
        self.pending.store(0, Ordering::Release);
        self.latched.store(0, Ordering::Release);
    }

    /// Latches what a real controller was already holding, from a capture of
    /// it.
    ///
    /// The latched word and not the pending one, because that is what the
    /// capture read: the register answers with whatever the last write latched,
    /// and the capture deliberately does not write. So the value is exactly
    /// what a guest's next read should answer with, and the pending set is
    /// emptied — which leaves the error interrupt armed, as it is after any
    /// write. Emptying it is a store rather than an assumption about the
    /// caller: the invariant this whole state machine rests on is that an empty
    /// pending set means an armed interrupt, and a controller seeded into a
    /// non-empty one would swallow its guest's first real error.
    ///
    /// Bits the architecture reserves on this processor are dropped rather than
    /// stored: hardware should not report them, and a guest must not read one
    /// back from a register this crate answers for.
    pub(crate) fn seed(&self, latched: u32) {
        self.pending.store(0, Ordering::Release);
        self.latched
            .store(latched & Errors::all().bits(), Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    //! The write-then-read protocol, the arming rule, and the three bits the
    //! presented processor does not have. The obligation a record answers with
    //! needs a controller and is exercised where it is discharged.

    use super::{ErrorStatus, Errors};

    #[test]
    fn read_before_any_write_answers_zero() {
        let status = ErrorStatus::new();

        assert_eq!(status.read(), 0);
    }

    #[test]
    fn a_record_stays_invisible_until_a_write() {
        let status = ErrorStatus::new();
        let _armed = status.record(Errors::RECEIVE_ILLEGAL_VECTOR);

        assert_eq!(status.read(), 0);
    }

    #[test]
    fn a_write_latches_what_was_recorded() {
        let status = ErrorStatus::new();
        let errors = Errors::SEND_ILLEGAL_VECTOR | Errors::ILLEGAL_REGISTER_ADDRESS;
        let _armed = status.record(errors);

        status.written();
        assert_eq!(status.read(), errors.bits());
    }

    #[test]
    fn a_second_write_clears_what_the_first_latched() {
        let status = ErrorStatus::new();
        let _armed = status.record(Errors::SEND_ACCEPT);
        status.written();

        status.written();
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

    #[test]
    fn recording_nothing_arms_nothing_and_leaves_the_arm_where_it_was() {
        // The one non-obvious line in `record`, and the one a plainer
        // `previous == 0` would get wrong in both directions: an empty record
        // leaves an empty pending set, so it must neither claim the interrupt
        // nor consume the arm the next real error is entitled to.
        let status = ErrorStatus::new();

        assert!(!status.record(Errors::empty()));
        assert!(status.record(Errors::SEND_ILLEGAL_VECTOR));
    }

    #[test]
    fn a_reset_empties_both_words_and_re_arms() {
        let status = ErrorStatus::new();
        let _armed = status.record(Errors::ILLEGAL_REGISTER_ADDRESS);
        status.written();

        status.reset();
        assert_eq!(status.read(), 0);
        assert!(
            status.record(Errors::ILLEGAL_REGISTER_ADDRESS),
            "a reset controller's first error arms its interrupt"
        );
    }

    #[test]
    fn seeding_drops_what_the_register_cannot_report_and_leaves_it_armed() {
        // A capture of firmware's own register, whole. The bits above the low
        // byte are reserved in every implementation, and the three the presented
        // processor reserves have no name here at all — so a guest must read
        // back neither, and the controller must come up armed rather than
        // swallowing its guest's first error.
        let status = ErrorStatus::new();
        let _armed = status.record(Errors::SEND_ILLEGAL_VECTOR);

        status.seed(u32::MAX);
        assert_eq!(status.read(), 0xEC);
        assert!(status.record(Errors::SEND_ACCEPT));
    }

    #[test]
    fn the_bits_the_presented_processor_reserves_have_no_name() {
        // Deleting them is what keeps them out of both words: nothing can record
        // one, and `seed`'s mask is the set itself rather than a second spelling
        // of it. A guest is told only about conditions its own processor reports.
        assert_eq!(Errors::all().bits(), 0xEC);
        for reserved in [0, 1, 4] {
            assert_eq!(
                Errors::all().bits() & (1 << reserved),
                0,
                "bit {reserved} is reserved on this processor"
            );
        }
    }
}
