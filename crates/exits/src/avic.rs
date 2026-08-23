//! The two exits a guest raises while the hardware drives its interrupt
//! controller.
//!
//! One says the hardware started delivering an interrupt between the guest's
//! own processors and could not finish; the other says the guest touched a
//! controller register the hardware does not implement. Neither can arrive
//! while the controller is driven in software — which is how it is driven
//! until this host decides otherwise — so both are answered defensively as
//! well as properly: the incomplete delivery is finished in whichever way its
//! failure's rule says, and the unaccelerated access is either bookkept —
//! the hardware completed it before it exited — or performed, at whichever of
//! the controller's two faces the guest reached the register through.
//!
//! # What the hardware was doing is asked of the control block
//!
//! Both handlers have a decision that turns on whether the acceleration was
//! driving this controller when the exit was raised, and that is a property of
//! the control block rather than of the controller: the machine-wide inhibit is
//! a store any processor may make, so the controller's own answer describes the
//! moment the exit is *answered* and never the moment it was raised.
//! [`vlapic::avic_accelerated`] is the block's answer, and the unaccelerated
//! access uses it because getting it wrong there costs a guest instruction.

use emulate::Outcome;
use inject::Pending;
use log::{error, trace};
use partition::Partition;
use svm::{
    avic::{IncompleteIpiExit, UnacceleratedAccessExit},
    exit::NestedPageFault,
};
use vcpu::{Flow, Vcpu};
use x86_64::PhysAddr;

use crate::{census::Census, nested};

/// Answers an interrupt the hardware could not finish delivering between the
/// guest's processors.
///
/// The exit is trap-like — the processor has already stepped the guest past
/// the request — and the finishing is [`vlapic`]'s, keyed by the failure the
/// hardware reported. Always resumes: every rule either completes the interrupt
/// in software, wakes whoever the hardware already delivered to, or discards a
/// command the architecture refuses outright, and a failure of any of the three
/// is logged rather than visited on the guest, which keeps running on whichever
/// path still works.
pub(crate) fn incomplete_ipi(vcpu: &Vcpu, census: &mut Census) -> Flow {
    let control = vcpu.control();
    let exit = IncompleteIpiExit::from_exit_info(control.exit_info_1, control.exit_info_2);
    census.incomplete_ipi(exit.reported());
    trace!(
        "exits: an interrupt the hardware delivers stopped: {:?} (identifier {}), icr {:#018x}, \
         index {:#x}",
        exit.cause(),
        exit.reported(),
        exit.icr(),
        exit.index()
    );
    if let Err(error) = vlapic::avic_incomplete_ipi(exit) {
        error!("exits: an incomplete IPI could not be completed: {error}");
    }
    Flow::Resume
}

/// Answers an access to a controller register the hardware does not
/// accelerate.
///
/// Three classes, and [`Answer`] is what tells them apart. A trap is an access
/// the hardware completed before it exited: the guest is past it, the value is
/// in the backing page, and what is owed is the bookkeeping the register asks
/// for beyond the store — the logical table an LDR write moves, the
/// acknowledgement a level EOI owes real hardware, the timer a count write
/// starts. A fault is an access the hardware never performed: the guest is
/// still at the instruction, and what is owed is the access itself. Which door
/// the guest reached the register through decides where that access is
/// performed, because the exit reports the same offset either way and only one
/// of the two faces put anything on the memory bus.
pub(crate) fn unaccelerated_access(
    vcpu: &mut Vcpu,
    partition: &Partition,
    interrupts: &mut Pending,
    census: &mut Census,
) -> Flow {
    let control = vcpu.control();
    let exit = UnacceleratedAccessExit::from_exit_info(control.exit_info_1, control.exit_info_2);
    census.noaccel(exit.offset());
    trace!(
        "exits: the guest touched a controller register the hardware does not accelerate: \
         offset {:#x}, {}",
        exit.offset(),
        if exit.is_write() { "write" } else { "read" }
    );
    match Answer::of(
        vlapic::avic_accelerated(vcpu),
        vlapic::avic_wider_face(vcpu),
        vlapic::avic_trap_access(exit.offset(), exit.is_write()),
    ) {
        Answer::Bookkeeping => {
            if let Err(error) = vlapic::avic_unaccelerated_trap(exit) {
                error!(
                    "exits: the bookkeeping for an unaccelerated access at {:#x} failed: {error}",
                    exit.offset()
                );
            }
            Flow::Resume
        }
        Answer::Registers => {
            if let Err(error) = vlapic::avic_unaccelerated_refused(exit) {
                error!(
                    "exits: the controller could not be handed back its own state as it stepped \
                     off the accelerated path: {error}"
                );
            }
            Flow::Resume
        }
        Answer::Memory => emulate(vcpu, partition, interrupts, exit),
    }
}

/// What answering one unaccelerated access is owed.
///
/// Three terms, two of them properties of the *control block* at the moment the
/// exit was raised. That is the whole of what this replaces: the guard used to
/// ask the model whether the acceleration was permitted, which is a store any
/// processor may make while this one is inside the guest — so this processor
/// could be running with the acceleration enabled in its block while the
/// predicate already answered false, and a trap would fall through to the
/// emulator. The emulator then decodes at the instruction *after* the access,
/// which a trap has already stepped the guest past, fails its provenance check,
/// and steps the guest over that next instruction. The guest silently skips an
/// instruction and the register write's bookkeeping never runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Answer {
    /// The hardware completed the access before it exited, so what is owed is
    /// the bookkeeping the register asks for beyond the store.
    Bookkeeping,
    /// The hardware performed no part of it and the guest reached the register
    /// through the memory-mapped page, so what is owed is the access itself,
    /// performed against the device that answers that page.
    Memory,
    /// The hardware performed no part of it and the guest reached the register
    /// as a model-specific register, so nothing was moved to or from memory
    /// and there is no instruction the page's device could be asked to
    /// perform.
    Registers,
}

impl Answer {
    /// What one exit is owed, out of what the control block was carrying when
    /// it was raised and how the register table classifies the access.
    ///
    /// `wider_face` cannot be true where `accelerated` is false: both are read
    /// out of the same two enable bits by the same derivation, and nothing but
    /// this processor writes them while it is answering its own exit. The block
    /// is asked first anyway — an exit that arrives with it unaccelerated is
    /// one no acceleration completed, whatever the register table would
    /// call it, so the access is performed like any other access to that
    /// page.
    const fn of(accelerated: bool, wider_face: bool, trap: bool) -> Self {
        if !accelerated {
            return Self::Memory;
        }
        if trap {
            return Self::Bookkeeping;
        }
        if wider_face {
            return Self::Registers;
        }
        Self::Memory
    }
}

/// Performs an access the hardware did not, against the device that answers
/// the register page.
///
/// The instruction is emulated exactly the way a nested fault in the page is
/// answered — decoded, validated against what the exit reported, and
/// performed against the same device, which advances the guest past it or
/// gives it the fault it earned. Nothing here knows which register the
/// access named: the device does, and its answer is the same answer the
/// access would have had without the acceleration.
///
/// Only for an access the guest really made through that page. The provenance
/// check is what requires it: the decoded instruction has to be a move whose
/// operand accounts for the address the exit reported, and a guest reaching its
/// controller through model-specific registers executed neither — see
/// [`Answer::Registers`].
fn emulate(
    vcpu: &mut Vcpu,
    partition: &Partition,
    interrupts: &mut Pending,
    exit: UnacceleratedAccessExit,
) -> Flow {
    // The access as the hardware would have reported it had the page faulted
    // instead of exiting: final, present, and in the direction the exit
    // names — which is all the emulator's provenance check asks of it.
    let cause = NestedPageFault::new()
        .with_present(true)
        .with_write(exit.is_write())
        .with_final_address(true);
    let gpa = PhysAddr::new(vlapic::apic_page().as_u64() + u64::from(exit.offset()));
    let Some(region) = partition.region(gpa) else {
        // The register page is a region of the guest whether or not the nested
        // tables trap it, so an address inside it that is in no region means the
        // guest was entered before its own memory was described, and resuming
        // would exit here for ever.
        error!(
            "exits: no region of the guest holds the unaccelerated access at offset {:#x}",
            exit.offset()
        );
        return Flow::Leave;
    };
    match partition.dispatch(vcpu, region.tag, gpa, cause) {
        Ok(Outcome::Stepped | Outcome::Repeating) => Flow::Resume,
        Ok(Outcome::Faulted(fault)) => nested::raise(vcpu, fault, interrupts),
        Err(error) => nested::unserviceable(vcpu, partition, gpa, error),
    }
}

#[cfg(test)]
mod tests {
    //! What one unaccelerated access is owed, which is the whole of what this
    //! module decides without a guest: the two handlers' other work is a
    //! control block and a device, and a host test has neither.

    use super::Answer;

    #[test]
    fn what_an_unaccelerated_access_is_owed_is_a_table_over_three_facts() {
        // Every state, read as the rule it is. The two the block answers are read
        // out of the same two enable bits, so the wider face without the
        // acceleration is not a state a caller can present — and it is answered
        // as the unaccelerated one anyway, because an exit that arrives with the
        // block unaccelerated is one no acceleration completed.
        for (accelerated, wider_face, trap, owed) in [
            (true, false, true, Answer::Bookkeeping),
            (true, true, true, Answer::Bookkeeping),
            (true, false, false, Answer::Memory),
            (true, true, false, Answer::Registers),
            (false, false, true, Answer::Memory),
            (false, false, false, Answer::Memory),
            (false, true, true, Answer::Memory),
            (false, true, false, Answer::Memory),
        ] {
            assert_eq!(
                Answer::of(accelerated, wider_face, trap),
                owed,
                "accelerated {accelerated}, wider face {wider_face}, trap {trap}"
            );
        }
    }

    #[test]
    fn a_trap_is_only_a_trap_while_the_block_carries_the_acceleration() {
        // The defect this replaces, stated as the property that forbids it: a
        // trap has already stepped the guest past the access, so answering one
        // with the emulator decodes at the instruction *after* it and steps the
        // guest over that one instead. Nothing but the block may decide it.
        for wider_face in [false, true] {
            assert_eq!(
                Answer::of(true, wider_face, true),
                Answer::Bookkeeping,
                "wider face {wider_face}"
            );
            assert_ne!(
                Answer::of(false, wider_face, true),
                Answer::Bookkeeping,
                "wider face {wider_face}"
            );
        }
    }

    #[test]
    fn a_fault_is_performed_at_the_face_the_guest_reached_the_register_through() {
        // The exit reports a page offset in both faces and says nothing about
        // which, and under the wider one no memory access took place at all — so
        // the register page's device has no instruction to decode and asking it
        // steps the guest over its own.
        assert_eq!(Answer::of(true, false, false), Answer::Memory);
        assert_eq!(Answer::of(true, true, false), Answer::Registers);
    }
}
