//! Which of the named processors a redirectable interrupt goes to.
//!
//! The architecture does not answer this. Lowest-priority delivery was
//! arbitrated by the chipset, differently on different chipsets, and neither
//! vendor specifies which processor a conforming implementation picks — so any
//! choice that picks one of the processors the command named and does not
//! starve the rest is a correct one.
//!
//! # Not by priority, because this hypervisor cannot read one
//!
//! The obvious rule — pick the processor running at the lowest priority — is
//! one this hypervisor is not in a position to apply. A processor's task
//! priority is kept in its own control block while it runs, written there by
//! its guest through a control register that takes no exit, and copied into the
//! emulated register only at that processor's *own* next exit. So the value
//! another processor can read here is as old as the target's last exit — which
//! for a processor busy in the guest is exactly the case the rule is supposed
//! to notice. Picking the "least busy" processor from those values can pick the
//! busiest one, and a redirectable device interrupt latched behind a raised
//! task priority sits there while idle processors are passed over.
//!
//! So the choice is made from the one thing that is both meaningful and free:
//! the vector. `vector % count` picks the starting point among the processors
//! that are accepting, which is the vector hashing real chipsets and interrupt
//! remapping hardware use — it spreads unrelated interrupts across processors,
//! it needs no state anywhere, and it is deterministic, so the same interrupt
//! keeps arriving on the same processor and its handler's working set stays
//! where it is.

use descriptors::Vector;

use crate::registers::Vlapic;

/// The processors a redirectable interrupt may go to, in the order it should be
/// offered to them.
///
/// Every one of them, not just the first choice, because a processor that
/// refuses is not the end of the delivery: the interrupt is meant for *one* of
/// the set, and the next one is entitled to it. What excludes a processor from
/// the set at all is not accepting interrupts — a controller that is switched
/// off or software-disabled would refuse this and selecting it alone would lose
/// the delivery for every eligible processor with it.
///
/// The filter is not the same thing as the acceptance, and cannot be: whether a
/// controller is accepting is that guest's own state, and it may change between
/// the two. That is the other reason the whole set is offered in order rather
/// than one processor picked out of it.
///
/// It may also change between the count that picks the starting point and the
/// walk itself, because both go through this filter. Nothing rests on the two
/// agreeing: whichever way it moved, every processor the walk yields is one
/// that was accepting when it was looked at, and the walk covers all but at
/// most one rotation of them. What it must not do is name a processor that is
/// not accepting, and it cannot.
pub(super) fn candidates<'a>(
    named: impl Iterator<Item = &'a Vlapic> + Clone,
    vector: Vector,
) -> impl Iterator<Item = &'a Vlapic> {
    offered(named.filter(|target| target.accepting()), vector)
}

/// The order a redirectable interrupt is offered to the processors that may
/// take it.
///
/// The vector picks where the walk starts and it then goes round: every
/// candidate exactly once, beginning with the one the vector hashes to. Written
/// over anything rather than over controllers because that is all the rule is —
/// a rotation of a sequence — and because the rule is then checkable on its
/// own, which the acceptance it feeds is not.
///
/// An empty set starts nowhere: the remainder is undefined for a count of zero,
/// and both halves of the walk are empty whatever it answers.
fn offered<T>(
    eligible: impl Iterator<Item = T> + Clone,
    vector: Vector,
) -> impl Iterator<Item = T> {
    let count = eligible.clone().count();
    let first = usize::from(vector.number()).checked_rem(count).unwrap_or(0);
    eligible.clone().skip(first).chain(eligible.take(first))
}

#[cfg(test)]
mod tests {
    //! Over numbers rather than controllers: the rule is a rotation, and a
    //! controller cannot be built without a machine to put it in.

    use descriptors::Vector;

    use super::offered;

    /// Four processors, named by something a test can compare.
    const ELIGIBLE: [u32; 4] = [10, 11, 12, 13];

    /// The walk, as a collected list.
    fn walk(eligible: &[u32], vector: u8) -> alloc::vec::Vec<u32> {
        offered(eligible.iter().copied(), Vector::new(vector)).collect()
    }

    #[test]
    fn the_vector_picks_where_the_walk_starts() {
        assert_eq!(walk(&ELIGIBLE, 0x30), [10, 11, 12, 13]);
        assert_eq!(walk(&ELIGIBLE, 0x31), [11, 12, 13, 10]);
        assert_eq!(walk(&ELIGIBLE, 0x32), [12, 13, 10, 11]);
        assert_eq!(walk(&ELIGIBLE, 0x33), [13, 10, 11, 12]);
        assert_eq!(walk(&ELIGIBLE, 0x34), [10, 11, 12, 13]);
    }

    #[test]
    fn every_candidate_is_offered_the_interrupt_exactly_once() {
        // What makes the fall-through complete: a walk that stops early would
        // drop an interrupt every processor after it would have taken.
        for vector in 0..=u8::MAX {
            for count in 1..=ELIGIBLE.len() {
                let mut walked = walk(&ELIGIBLE[..count], vector);
                assert_eq!(walked.len(), count, "vector {vector:#x} over {count}");
                walked.sort_unstable();
                assert_eq!(walked, &ELIGIBLE[..count], "vector {vector:#x}");
            }
        }
    }

    #[test]
    fn a_set_with_nothing_in_it_is_walked_and_nothing_happens() {
        // The case that decides whether the sender is told: no processor the
        // command named is accepting interrupts, so the walk finds nobody and
        // the caller records that nobody took it.
        assert!(walk(&[], 0x30).is_empty());
        assert!(!offered([0u32; 0].into_iter(), Vector::new(0x30)).any(|_| true));
    }

    #[test]
    fn the_walk_stops_at_the_first_candidate_that_takes_it() {
        // The acceptance played out over the answers it would give: the first
        // two refuse, the third takes it, and the fourth is never asked.
        let answers = [false, false, true, false];
        let mut asked = alloc::vec::Vec::new();
        let taken = offered(answers.iter().enumerate(), Vector::new(0x30)).any(|(who, accepts)| {
            asked.push(who);
            *accepts
        });

        assert!(taken);
        assert_eq!(asked, [0, 1, 2]);
    }

    #[test]
    fn a_set_that_all_refuses_is_walked_to_the_end() {
        let answers = [false; 4];
        let mut asked = 0;
        let taken = offered(answers.iter(), Vector::new(0x31)).any(|accepts| {
            asked += 1;
            *accepts
        });

        assert!(!taken);
        assert_eq!(asked, answers.len(), "every one of them was asked");
    }
}
