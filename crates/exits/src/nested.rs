//! Guest physical addresses the second set of page tables had no answer for.

use descriptors::Vector;
use emulate::{Fault, Outcome};
use log::error;
use npt::Resolution;
use partition::{Addressing, Partition};
use svm::{CleanBits, Event, exit::NestedPageFault};
use vcpu::{Flow, Vcpu};
use x86_64::PhysAddr;

/// Describes the region the guest touched, or steps it past an access that will
/// never succeed.
///
/// A fault here is ordinary: a guest's memory is described as it is touched, so
/// most of these are the guest reaching a region for the first time and the
/// answer is to describe it and resume.
pub(crate) fn exit(vcpu: &mut Vcpu, partition: &Partition) -> Flow {
    let cause = NestedPageFault::from_bits(vcpu.control().exit_info_1);
    let gpa = PhysAddr::new_truncate(vcpu.control().exit_info_2);
    match partition.resolve(gpa, cause) {
        Ok(Resolution::Mapped) => Flow::Resume,
        Ok(Resolution::Shadowed) => discard(vcpu, partition, gpa),
        Ok(Resolution::Trapped) => interposed(vcpu, partition, gpa, cause),
        Err(error) => {
            error!("exits: nested fault at {gpa:#x} could not be resolved: {error}");
            Flow::Leave
        }
    }
}

/// Lets whatever answers for a region answer this access.
///
/// The instruction is performed against the device rather than against the
/// memory the guest aimed it at, and the guest is stepped past it — except for
/// a repeated move with repetitions left, which is deliberately left to execute
/// again, and for one that established that the guest owes an exception, which
/// is delivered instead.
fn interposed(
    vcpu: &mut Vcpu,
    partition: &Partition,
    gpa: PhysAddr,
    cause: NestedPageFault,
) -> Flow {
    let Some(devices) = partition.devices() else {
        // The region is trapped, so something meant to answer for it, but the
        // set was never sealed. Resuming would fault at the same address
        // forever.
        error!("exits: nothing answers for the trapped access at {gpa:#x}");
        return Flow::Leave;
    };
    let addressing = Addressing::from_save(vcpu.save());
    match partition.with_memory(addressing, |guest| {
        devices.dispatch(vcpu, guest, gpa, cause)
    }) {
        // Done, or stopped part way through a repeated move with its progress in
        // the guest's own registers. Either way the guest resumes and needs
        // nothing from us.
        Ok(Outcome::Stepped | Outcome::Repeating) => Flow::Resume,
        Ok(Outcome::Faulted(fault)) => raise(vcpu, fault),
        Err(error) => {
            error!("exits: the access to {gpa:#x} could not be performed: {error}");
            Flow::Leave
        }
    }
}

/// Gives the guest an exception its own instruction earned.
///
/// Not a hypervisor failure. The emulator established that the instruction it
/// was performing on the guest's behalf is one a real processor would have
/// faulted on — a misaligned operand where the encoding requires alignment, or
/// an address the guest's own tables do not describe — and the guest's handler
/// is where that belongs. A demand-paged operating system reaches this
/// constantly and expects to: it maps the page and the instruction runs again.
///
/// The instruction pointer is deliberately unchanged, so the handler returns to
/// the instruction rather than past it, and whatever progress a repeated move
/// made is already in the index and count registers.
fn raise(vcpu: &mut Vcpu, fault: Fault) -> Flow {
    let vector = Vector::new(fault.vector());
    if let Some(address) = fault.address() {
        // What the handler reads to find out which address it has to describe. The
        // architecture publishes it in `CR2`, and nothing else carries it.
        vcpu.save_mut().cr2 = address;
    }
    vcpu.control_mut().event_injection = Event::exception_with_code(vector, fault.code());
    vcpu.soil(CleanBits::INTERRUPT);
    Flow::Resume
}

/// Steps the guest past a write to memory that will never accept one.
///
/// The hypervisor's own memory reads as a shared page of zeroes and has no page
/// behind it to take a write. So the instruction is decoded for its length
/// alone, its write is dropped, and the guest carries on after it — which is
/// the only alternative to faulting on the same instruction forever.
fn discard(vcpu: &mut Vcpu, partition: &Partition, gpa: PhysAddr) -> Flow {
    let addressing = Addressing::from_save(vcpu.save());
    match partition.with_memory(addressing, |guest| emulate::next_rip(vcpu, guest)) {
        Ok(next) => {
            vcpu.save_mut().rip = next;
            Flow::Resume
        }
        Err(error) => {
            error!("exits: could not skip shadowed write at {gpa:#x}: {error}");
            Flow::Leave
        }
    }
}
