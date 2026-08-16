//! The three registers that are really one bit per vector.
//!
//! Interrupt-request, in-service and trigger-mode are each 256 bits, laid out
//! as eight 32-bit registers. A guest reads them a register at a time; the
//! controller uses them a bit at a time; and a bit belonging to one processor
//! is set by any of the others, because sending an interrupt is exactly that.
//!
//! So they are atomics, and the whole of this module exists to make sure the
//! vector arithmetic that turns a vector into a slot and a bit is written once.
//! Getting it wrong is an interrupt delivered as a different one.
//!
//! # Which orderings, and why
//!
//! Setting a bit releases and reading one acquires. That is not
//! belt-and-braces: a processor delivering a level-triggered interrupt sets the
//! trigger-mode bit *before* the request bit, and the processor that observes
//! the request bit must not then read a trigger-mode bit from before it was
//! set. Release on the store and acquire on the load is what makes the pair
//! ordered; relaxed would let the second read be hoisted above the first and
//! deliver a level-triggered interrupt as an edge-triggered one, which is a
//! missing end-of-interrupt and a line that never fires again.

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
    pub(crate) fn set(&self, vector: Vector) -> bool {
        let (slot, bit) = place(vector);
        self.slots[slot].fetch_or(bit, Ordering::Release) & bit != 0
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
    pub(crate) fn highest(&self) -> Option<Vector> {
        (0..SLOTS).rev().find_map(|slot| {
            let word = self.slots[slot].load(Ordering::Acquire);
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
    /// that took the bit. The two are not one read-modify-write and cannot be:
    /// finding the highest set bit spans eight independent words, and no single
    /// atomic operation covers them. What the retry establishes is only that
    /// the vector answered was really taken by this call and not by a
    /// concurrent one; it does not establish that the vector answered was
    /// the highest at any single instant, because a higher bit set during
    /// the scan may be missed.
    ///
    /// That weaker guarantee is enough for the two callers this has, and both
    /// rely on the same precondition: exactly one processor consumes from a
    /// given bitmap, and nothing resets it concurrently. The in-service
    /// register is consumed only by the processor the controller belongs
    /// to, and the ledger only by the processor whose hardware owes the
    /// debt. A second consumer, or a reset racing a take, would need a
    /// different structure.
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
    pub(crate) fn is_empty(&self) -> bool {
        self.slots
            .iter()
            .all(|slot| slot.load(Ordering::Acquire) == 0)
    }

    /// How many bits are set, which is how many interrupts this describes.
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
