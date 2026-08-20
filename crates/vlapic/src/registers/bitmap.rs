//! The three registers that are really one bit per vector.
//!
//! Interrupt-request, in-service and trigger-mode are each 256 bits, laid out
//! as eight 32-bit registers. A guest reads them a register at a time; the
//! controller uses them a bit at a time; and a bit belonging to one processor
//! is set by any of the others, because sending an interrupt is exactly that.
//!
//! So they are atomics, and this module exists so that the arithmetic turning a
//! vector into a slot and a bit is written once *for the controller a guest
//! sees*. Getting it wrong is an interrupt delivered as a different one.
//!
//! # Once here is not once in the machine
//!
//! The real controller has the same three registers and its own arithmetic for
//! them, in [`apic`], and there is no way to have one copy: that one addresses
//! real registers through whichever face the host is using, and this one
//! addresses an array of atomics. The two meet in
//! [`crate::lifecycle::ledger`], which compares a vector this module produced
//! against one the real controller reported — so what has to agree is the
//! *vector*, which is a [`Vector`] on both sides and not a slot or a bit.
//!
//! What could disagree is the number of slots, and it cannot: [`SLOTS`] is
//! [`apic::VECTOR_WORDS`]. A divergence there would corrupt a guest's read of
//! the top of one of these banks rather than lose an interrupt, because the
//! only consumers of a slot are the three readbacks a guest performs.
//!
//! # Which orderings, and why
//!
//! [`Bitmap::set`] and [`Bitmap::highest`] are sequentially consistent, and the
//! reason is not that a stronger ordering is safer. They are two of the four
//! accesses that make up the handshake which stops a wakeup being lost, and
//! that argument needs a single total order over all four:
//!
//! - A processor delivering an interrupt sets the request bit, then reads
//!   whether the target has stopped watching its controller
//!   ([`crate::registers::Vlapic::away`]).
//! - The target stores that it has stopped watching, then scans the request
//!   register one last time before entering the guest.
//!
//! Under the Rust and C++ memory model, sequentially consistent operations —
//! and only those — take part in one total order `S` consistent with
//! happens-before and with each object's modification order. With all four in
//! it, the deliverer reading "not away" forces its read after the target's
//! store in `S`; program order puts its own set before that read and the
//! target's store before its scan; so the set precedes the scan in `S`, and two
//! sequentially consistent accesses to the same word must agree with that
//! word's modification order — the scan sees the bit. Either the deliverer
//! sends a doorbell or the target finds the interrupt itself, and there is no
//! interleaving in which both miss.
//!
//! A release read-modify-write and an acquire load do not give that. A release
//! RMW's load half is relaxed and neither operation orders a store against a
//! later load, so the both-miss interleaving is permitted by the model even
//! though it cannot be produced on x86 — where `lock or` drains the store
//! buffer and a sequentially consistent store compiles to `xchg`, so both sides
//! are already full barriers. That is what makes this free here: on x86-64 a
//! sequentially consistent load is a plain `mov` and the read-modify-write is
//! the same `lock or` it was, so the argument is bought with no instructions
//! at all. It is stated in terms of the model because the model is what a
//! future reader — or a future target — is entitled to rely on.
//!
//! The trigger-mode register is written before the request bit by
//! [`crate::registers::Vlapic::accept`], and nothing in this crate reads a
//! trigger-mode bit back: the level-or-edge decision this hypervisor acts on
//! comes from the *real* controller's own record, and the virtual register is
//! there for the guest's own readback of a whole word. So that pair is not an
//! ordering anything depends on today. It is kept in that order because it is
//! the order hardware writes them in, and because anything that ever did make
//! the virtual register the oracle would depend on it — at which point the
//! release on the trigger-mode store and the acquire on its load become
//! load-bearing rather than merely true.

use core::sync::atomic::{AtomicU32, Ordering};

use descriptors::Vector;

/// How many 32-bit registers it takes to give all 256 vectors a bit.
///
/// The real controller's own count, because these registers are seeded from a
/// capture of the real ones and read back a slot at a time by a guest reading
/// the same bank. Two counts that had to agree and did not would be a guest
/// register read answered out of the wrong word.
pub(crate) const SLOTS: usize = apic::VECTOR_WORDS;

/// How many vectors one register describes.
const PER_SLOT: u8 = 32;

/// One bit per vector, spread across eight registers.
#[derive(Debug)]
pub(crate) struct Bitmap {
    slots: [AtomicU32; SLOTS],
}

impl Bitmap {
    /// How many vectors one of these describes, which is all of them.
    ///
    /// Worth naming because it bounds every loop that consumes bits: a caller
    /// draining one cannot go round more times than there are bits to clear.
    pub(crate) const CAPACITY: usize = SLOTS * PER_SLOT as usize;

    /// All clear, which is what reset leaves these.
    pub(crate) const fn new() -> Self {
        Self {
            slots: [const { AtomicU32::new(0) }; SLOTS],
        }
    }

    /// Sets a vector's bit, and says whether it was already set.
    ///
    /// The answer matters for the request register: a vector already requested
    /// and not yet accepted is one the architecture folds into the single bit,
    /// so a second arrival is not a second interrupt and nothing should be
    /// counted twice for it.
    ///
    /// Sequentially consistent because this is the store half of the handshake
    /// the module doc states, not merely a bit being published.
    pub(crate) fn set(&self, vector: Vector) -> bool {
        let (slot, bit) = place(vector);
        self.slots[slot].fetch_or(bit, Ordering::SeqCst) & bit != 0
    }

    /// Clears a vector's bit, and says whether it had been set.
    pub(crate) fn clear(&self, vector: Vector) -> bool {
        let (slot, bit) = place(vector);
        self.slots[slot].fetch_and(!bit, Ordering::AcqRel) & bit != 0
    }

    /// Whether a vector's bit is set.
    pub(crate) fn get(&self, vector: Vector) -> bool {
        let (slot, bit) = place(vector);
        self.slots[slot].load(Ordering::Acquire) & bit != 0
    }

    /// The highest-priority vector with its bit set, if any.
    ///
    /// Highest-numbered is highest-priority: the upper nibble of a vector is
    /// its priority class and the lower nibble ranks within the class, so one
    /// descending scan answers both.
    ///
    /// Sequentially consistent because this is the load half of the handshake
    /// the module doc states: every scan of a request register goes through
    /// here, including the last one a processor makes before it enters the
    /// guest.
    ///
    /// # A snapshot, not an instant
    ///
    /// The scan spans eight independent words and no single atomic operation
    /// covers them, so what comes back was not necessarily the highest set
    /// vector at any one moment: a bit set in a word this has already passed is
    /// missed. That is not a lost interrupt — the bit stays set, and the
    /// processor that set it either finds this one still watching and rings its
    /// doorbell or is the reason it looks again — but it does mean this is not
    /// a linearisation point and nothing may treat it as one.
    pub(crate) fn highest(&self) -> Option<Vector> {
        (0..SLOTS).rev().find_map(|slot| {
            let word = self.slots[slot].load(Ordering::SeqCst);
            // The highest set bit of a non-zero word, and nothing at all for an
            // empty one — which is the whole of what makes this total. An
            // arithmetic "thirty-one less the leading zeros" answers `-1` for an
            // empty word, and a wrapping one at that.
            Some(vector_at(slot, word.checked_ilog2()?))
        })
    }

    /// Takes the highest-priority vector with its bit set, clearing it.
    ///
    /// A scan followed by a separate clear, retried until the clear is the one
    /// that took the bit. The two are not one read-modify-write and cannot be,
    /// for the reason [`Bitmap::highest`] gives. What the retry establishes is
    /// only that the vector answered was really taken by this call and not by a
    /// concurrent one; it does not establish that the vector answered was the
    /// highest at any single instant.
    ///
    /// That weaker guarantee is enough for the one caller this has, and it
    /// rests on a precondition: exactly one thing consumes from a given
    /// bitmap, and nothing resets it concurrently. The in-service register
    /// is consumed only by the processor the controller belongs to,
    /// acknowledging one interrupt at a time out of its own guest. A second
    /// consumer, or a reset racing a take, would need a different structure
    /// — which is why the one other place in this crate that walks a bitmap
    /// down to nothing does it with [`Bitmap::highest`] and a clear of its
    /// own, inside a window where this processor's interrupts are held off.
    pub(crate) fn take_highest(&self) -> Option<Vector> {
        loop {
            let vector = self.highest()?;
            if self.clear(vector) {
                return Some(vector);
            }
            // Another processor took it between the scan and the clear. There
            // may still be a lower one, so look again rather than answering
            // with nothing.
        }
    }

    /// Whether no vector's bit is set.
    ///
    /// Eight loads and not one operation, so this can answer `true` for a
    /// bitmap that was never empty — a bit set in a word already passed, in
    /// a word not yet reached — exactly as [`Bitmap::highest`] can miss
    /// one. Its callers ask it of a bitmap only they write, or of one where
    /// a wrong answer costs a misleading line of diagnostics.
    pub(crate) fn is_empty(&self) -> bool {
        self.slots
            .iter()
            .all(|slot| slot.load(Ordering::Acquire) == 0)
    }

    /// How many bits are set, which is how many interrupts this describes.
    ///
    /// A snapshot in the same sense [`Bitmap::is_empty`] is: the count can
    /// include a bit that was cleared before the scan ended and miss one that
    /// was set. Diagnostic only, and that is why.
    pub(crate) fn count(&self) -> u32 {
        self.slots
            .iter()
            .map(|slot| slot.load(Ordering::Acquire).count_ones())
            .sum()
    }

    /// One of the eight registers, as the guest reads it, or `None` for a slot
    /// this bitmap does not have.
    ///
    /// Answering with the slot's absence rather than with a zero leaves what a
    /// missing slot means to the caller — and its only callers are the three
    /// guest bank readbacks, where the offset a slot came from is one of eight
    /// by construction and a zero would be indistinguishable from an empty
    /// register.
    pub(crate) fn slot(&self, slot: usize) -> Option<u32> {
        self.slots
            .get(slot)
            .map(|word| word.load(Ordering::Acquire))
    }

    /// Puts one of the eight registers at what another authority was holding,
    /// whatever this one held.
    ///
    /// For a bank that crossed to an authority which then owned it outright:
    /// the in-service bank of a controller the hardware has been driving is
    /// retired in the backing page with no exit for this side to hear, so
    /// what comes back replaces rather than joins what is here.
    ///
    /// Not part of the handshake [`Bitmap::set`] and [`Bitmap::highest`] are
    /// the two halves of, and neither are [`Bitmap::merge`] and
    /// [`Bitmap::retire`]: all three are performed by the processor the
    /// controller belongs to at a boundary where its guest is stopped, and
    /// the next thing to read what they leave is that processor's own scan.
    ///
    /// A slot the register file does not have cannot arrive here — the words a
    /// caller is putting back came out of a bank of this same length — and
    /// there is nothing to store for one that did.
    pub(crate) fn put(&self, slot: usize, word: u32) {
        if let Some(register) = self.slots.get(slot) {
            register.store(word, Ordering::Release);
        }
    }

    /// Adds every bit of `word` to one of the eight registers.
    ///
    /// For a bank both authorities may hold something in, where neither one's
    /// bits may be dropped: a request accepted on the software path while the
    /// hardware was being taken off the controller is in this bank alone, and
    /// one the hardware took is in the other.
    pub(crate) fn merge(&self, slot: usize, word: u32) {
        if let Some(register) = self.slots.get(slot) {
            register.fetch_or(word, Ordering::AcqRel);
        }
    }

    /// Clears the bits of `word` in one of the eight registers.
    ///
    /// For a bank handed to another authority: exactly what that authority took
    /// is cleared and nothing else, so a bit another processor set after the
    /// hand-over read the word is one this register file goes on holding.
    pub(crate) fn retire(&self, slot: usize, word: u32) {
        if let Some(register) = self.slots.get(slot) {
            register.fetch_and(!word, Ordering::AcqRel);
        }
    }

    /// Clears every bit, which is what reset and INIT leave these.
    pub(crate) fn reset(&self) {
        for slot in &self.slots {
            slot.store(0, Ordering::Release);
        }
    }

    /// Puts every register at what a real controller was holding.
    ///
    /// Only ever used to seed a controller from a capture of the hardware it
    /// stands for. Storing the registers one at a time rather than as one step
    /// is safe there because the caller brackets them with the controller's
    /// reset count, exactly as it brackets a reset: an arrival that races
    /// the seeding sees the count move and publishes again into the seeded
    /// register.
    pub(crate) fn seed(&self, words: &[u32; SLOTS]) {
        for (slot, word) in self.slots.iter().zip(words) {
            slot.store(*word, Ordering::Release);
        }
    }
}

/// Which register a vector's bit is in, and which bit of it.
fn place(vector: Vector) -> (usize, u32) {
    let number = vector.number();
    (
        usize::from(number / PER_SLOT),
        1 << u32::from(number % PER_SLOT),
    )
}

/// The vector a register's bit belongs to.
#[expect(
    clippy::cast_possible_truncation,
    reason = "eight slots of thirty-two bits is two hundred and fifty-six, which is the whole of a vector's range — and the assertion below is what keeps that true"
)]
fn vector_at(slot: usize, bit: u32) -> Vector {
    Vector::new((slot as u8) * PER_SLOT + (bit as u8))
}

/// The eight registers have to cover exactly the vectors there are, because
/// [`vector_at`] narrows a slot and a bit into one byte to name one: a ninth
/// slot would wrap round and answer with a vector from the bottom of the range,
/// and seven would leave the top of it unreachable.
const _: () = assert!(
    Bitmap::CAPACITY == Vector::COUNT,
    "a bitmap must give every vector exactly one bit"
);

#[cfg(test)]
mod tests {
    //! The arithmetic is checked in both directions over every vector, because
    //! getting it wrong is an interrupt delivered as a different one and
    //! nothing downstream would notice.

    use descriptors::Vector;

    use super::{Bitmap, PER_SLOT, SLOTS, place, vector_at};

    /// Every vector there is, which is what this module answers for.
    fn all() -> impl Iterator<Item = Vector> {
        (0..=u8::MAX).map(Vector::new)
    }

    #[test]
    fn every_vector_has_its_own_bit_and_answers_to_it() {
        for vector in all() {
            let (slot, bit) = place(vector);
            assert!(slot < SLOTS, "{vector} is in a slot that exists");
            assert_eq!(
                bit.count_ones(),
                1,
                "{vector} names exactly one bit of its slot"
            );
            assert_eq!(
                vector_at(slot, bit.trailing_zeros()),
                vector,
                "{vector} has to come back out of the slot and bit it went into"
            );
        }
    }

    #[test]
    fn the_slots_cover_every_vector_exactly_once() {
        let mut seen = [false; Bitmap::CAPACITY];
        for vector in all() {
            let (slot, bit) = place(vector);
            let index = slot * PER_SLOT as usize + bit.trailing_zeros() as usize;
            assert!(!seen[index], "{vector} shares a bit with another vector");
            seen[index] = true;
        }
        assert!(seen.into_iter().all(|used| used));
    }

    #[test]
    fn an_empty_bitmap_holds_no_highest_vector() {
        // The case the arithmetic used to answer wrongly: an empty word has no
        // highest set bit at all, and computing one from its leading zeros
        // produces a vector from the bottom of the range.
        let bitmap = Bitmap::new();
        assert_eq!(bitmap.highest(), None);
        assert_eq!(bitmap.take_highest(), None);
        assert!(bitmap.is_empty());
        assert_eq!(bitmap.count(), 0);
    }

    #[test]
    fn the_lowest_and_highest_vectors_are_both_reachable() {
        for vector in [Vector::new(0), Vector::new(u8::MAX)] {
            let bitmap = Bitmap::new();
            assert!(!bitmap.set(vector), "the bit was not already set");
            assert!(bitmap.get(vector));
            assert_eq!(bitmap.highest(), Some(vector));
            assert_eq!(bitmap.count(), 1);
        }
    }

    #[test]
    fn a_repeated_set_says_the_bit_was_already_there() {
        let bitmap = Bitmap::new();
        let vector = Vector::new(0x42);

        assert!(!bitmap.set(vector));
        assert!(
            bitmap.set(vector),
            "a second arrival folds into the one bit"
        );
        assert_eq!(bitmap.count(), 1);
        assert!(bitmap.clear(vector));
        assert!(!bitmap.clear(vector), "clearing a clear bit took nothing");
    }

    #[test]
    fn the_highest_vector_is_the_highest_number_whichever_slot_it_is_in() {
        let bitmap = Bitmap::new();
        // One in the lowest slot, one in the highest, and one in between.
        for vector in [Vector::new(0x10), Vector::new(0x7F), Vector::new(0xF0)] {
            bitmap.set(vector);
        }
        assert_eq!(bitmap.highest(), Some(Vector::new(0xF0)));
        assert_eq!(bitmap.count(), 3);
    }

    #[test]
    fn taking_the_highest_walks_down_in_priority_order() {
        let bitmap = Bitmap::new();
        let vectors = [Vector::new(0x21), Vector::new(0x5F), Vector::new(0xE0)];
        for vector in vectors {
            bitmap.set(vector);
        }
        for vector in vectors.into_iter().rev() {
            assert_eq!(bitmap.take_highest(), Some(vector));
        }
        assert_eq!(bitmap.take_highest(), None);
        assert!(bitmap.is_empty());
    }

    #[test]
    fn a_guest_reads_the_slot_its_vectors_bits_are_in() {
        let bitmap = Bitmap::new();
        // The first vector of the second register, and the last of the first.
        bitmap.set(Vector::new(32));
        bitmap.set(Vector::new(31));

        assert_eq!(bitmap.slot(0), Some(1 << 31));
        assert_eq!(bitmap.slot(1), Some(1));
        assert_eq!(bitmap.slot(2), Some(0));
        assert_eq!(
            bitmap.slot(SLOTS),
            None,
            "a slot past the eight the register file has is not a register"
        );
    }

    #[test]
    fn a_bank_crosses_between_authorities_a_word_at_a_time() {
        // The three operations a bank crossing between the two authorities that
        // may hold it is made of. `put` replaces, because what it comes from
        // owned the bank outright; `merge` adds, because both sides may hold
        // something the other never had; `retire` clears exactly what crossed,
        // so a bit another processor delivered in the meantime stays.
        let bitmap = Bitmap::new();
        bitmap.set(Vector::new(1));
        bitmap.put(0, 0b1100);
        assert_eq!(bitmap.slot(0), Some(0b1100), "what was held here is gone");
        bitmap.merge(0, 0b0011);
        assert_eq!(bitmap.slot(0), Some(0b1111));
        bitmap.retire(0, 0b0101);
        assert_eq!(bitmap.slot(0), Some(0b1010));
        assert_eq!(bitmap.count(), 2);
        // A slot the register file does not have is not one any of them reaches,
        // and nothing it holds moves.
        bitmap.put(SLOTS, !0);
        bitmap.merge(SLOTS, !0);
        bitmap.retire(SLOTS, !0);
        assert_eq!(bitmap.slot(0), Some(0b1010));
        assert_eq!(bitmap.count(), 2);
    }

    #[test]
    fn a_seeded_bitmap_reads_back_what_hardware_was_holding() {
        let bitmap = Bitmap::new();
        let mut words = [0; SLOTS];
        words[0] = 0b101;
        words[SLOTS - 1] = 1 << 31;
        bitmap.seed(&words);

        assert!(bitmap.get(Vector::new(0)) && bitmap.get(Vector::new(2)));
        assert_eq!(bitmap.highest(), Some(Vector::new(u8::MAX)));
        assert_eq!(bitmap.count(), 3);

        bitmap.reset();
        assert!(bitmap.is_empty());
    }
}
