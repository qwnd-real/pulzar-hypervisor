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
pub(crate) const SLOTS: usize = 8;

/// How many vectors one register describes.
const PER_SLOT: u8 = 32;

/// One bit per vector, spread across eight registers.
#[derive(Debug, Default)]
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
            (word != 0).then(|| vector_at(slot, highest_bit(word)))
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

    /// One of the eight registers, as the guest reads it.
    pub(crate) fn slot(&self, slot: usize) -> u32 {
        self.slots
            .get(slot)
            .map_or(0, |word| word.load(Ordering::Acquire))
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
    /// stands for, before any guest has run and before anything can be
    /// delivering into it — which is what makes storing the registers one at a
    /// time rather than as one step correct here and nowhere else.
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
    reason = "eight slots of thirty-two bits is two hundred and fifty-six, which is the whole of a vector's range"
)]
fn vector_at(slot: usize, bit: u32) -> Vector {
    Vector::new((slot as u8) * PER_SLOT + (bit as u8))
}

/// Which bit of a non-zero word is the highest set one.
fn highest_bit(word: u32) -> u32 {
    u32::BITS - 1 - word.leading_zeros()
}
