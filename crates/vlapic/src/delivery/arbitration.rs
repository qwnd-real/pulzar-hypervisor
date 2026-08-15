//! Which of the named processors a redirectable interrupt goes to.
//!
//! The rule is the guest processor's own and the two vendors disagree about
//! both halves of it, which is why the choice belongs to
//! [`crate::hardware::model`] and only its application is here.

use crate::{hardware::model::Arbitration, registers::Vlapic};

/// Which of the named processors a redirectable interrupt should go to.
///
/// The rule is the guest processor's own, and the two vendors disagree about
/// both halves of it. AMD compares arbitration priorities, which count what a
/// processor has merely been sent as well as what it is servicing, and gives a
/// tie to the highest identifier. Intel's chipsets compared processor
/// priorities, which count only what is in service, and left a tie to whichever
/// answered first.
///
/// Neither picks a processor that is not accepting: a controller that is
/// switched off or software-disabled would refuse the interrupt, and selecting
/// it would lose the delivery for every eligible processor as well.
pub(super) fn least_busy<'a>(
    from: &Vlapic,
    targets: impl Iterator<Item = &'a Vlapic>,
) -> Option<&'a Vlapic> {
    let eligible = targets.filter(|target| target.accepting());
    match from.model().arbitration() {
        Arbitration::AmdArbitrationPriority => eligible.min_by(|left, right| {
            left.arbitration_priority()
                .cmp(&right.arbitration_priority())
                // A tie goes to the highest identifier, so the ordering is
                // reversed on the key that breaks it: the minimum of the pair
                // has to be the one that wins.
                .then_with(|| right.apic_id().get().cmp(&left.apic_id().get()))
        }),
        Arbitration::ProcessorPriority => {
            eligible.min_by_key(|target| target.processor_priority().get())
        }
    }
}
