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
//! the hardware completed it before it exited — or performed, the way the
//! register page's device performs any access to it.

use emulate::Outcome;
use inject::Pending;
use log::{error, trace};
use partition::{Addressing, Partition};
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
/// hardware reported. Always resumes: every rule either completes the
/// interrupt in software or wakes whoever the hardware already delivered to,
/// and a failure of either is logged rather than visited on the guest, which
/// keeps running on whichever path still works.
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
/// Two classes, and the register table's classification of the offset is
/// what tells them apart. A trap is an access the hardware completed before
/// it exited: the guest is past it, the value is in the backing page, and
/// what is owed is the bookkeeping the register asks for beyond the store —
/// the logical table an LDR write moves, the acknowledgement a level EOI
/// owes real hardware, the timer a count write starts. A fault is an access
/// the hardware never performed: the guest is still at the instruction, and
/// what is owed is the access itself, performed against the device that
/// answers the register page and retired — or the fault the instruction
/// earned.
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
    // A trap is only a trap while the hardware is driving: an exit naming
    // one that arrives at any other time is an access nothing completed, and
    // is performed below like any other.
    if exit.is_write()
        && vlapic::avic_active().unwrap_or(false)
        && vlapic::avic_trap_access(exit.offset(), true)
    {
        if let Err(error) = vlapic::avic_unaccelerated_trap(exit) {
            error!(
                "exits: the bookkeeping for an unaccelerated access at {:#x} failed: {error}",
                exit.offset()
            );
        }
        return Flow::Resume;
    }
    emulate(vcpu, partition, interrupts, exit)
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
fn emulate(
    vcpu: &mut Vcpu,
    partition: &Partition,
    interrupts: &mut Pending,
    exit: UnacceleratedAccessExit,
) -> Flow {
    let Some(devices) = partition.devices() else {
        // The register page is a region of the guest whether or not the nested
        // tables trap it, and it is registered before any processor enters —
        // so an unsealed set means the guest was entered before its own memory
        // was described, and resuming would exit here for ever.
        error!(
            "exits: nothing answers for the unaccelerated access at offset {:#x}",
            exit.offset()
        );
        return Flow::Leave;
    };
    // The access as the hardware would have reported it had the page faulted
    // instead of exiting: final, present, and in the direction the exit
    // names — which is all the emulator's provenance check asks of it.
    let cause = NestedPageFault::new()
        .with_present(true)
        .with_write(exit.is_write())
        .with_final_address(true);
    let gpa = PhysAddr::new(vlapic::apic_page().as_u64() + u64::from(exit.offset()));
    let addressing = Addressing::from_save(vcpu.save());
    match partition.with_memory(addressing, |guest| {
        devices.dispatch(vcpu, guest, gpa, cause)
    }) {
        Ok(Outcome::Stepped | Outcome::Repeating) => Flow::Resume,
        Ok(Outcome::Faulted(fault)) => nested::raise(vcpu, fault, interrupts),
        Err(error) => nested::unserviceable(vcpu, partition, gpa, error),
    }
}
