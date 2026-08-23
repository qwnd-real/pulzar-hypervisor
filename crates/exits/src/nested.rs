//! Guest physical addresses the second set of page tables had no answer for.

use core::sync::atomic::{AtomicBool, Ordering};

use descriptors::Vector;
use emulate::{EmulateError, Fault, Outcome};
use inject::Pending;
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
pub(crate) fn exit(vcpu: &mut Vcpu, partition: &Partition, interrupts: &mut Pending) -> Flow {
    let cause = NestedPageFault::from_bits(vcpu.control().exit_info_1);
    let gpa = PhysAddr::new_truncate(vcpu.control().exit_info_2);
    match partition.resolve(gpa, cause) {
        Ok(Resolution::Mapped) => Flow::Resume,
        Ok(Resolution::Shadowed) => step_over(vcpu, partition, gpa),
        Ok(Resolution::Trapped) => interposed(vcpu, partition, gpa, cause, interrupts),
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
    interrupts: &mut Pending,
) -> Flow {
    let Some(region) = partition.region(gpa) else {
        // The tables reported the address as one they do not answer for and now
        // say it is in no region at all, which nothing else can produce.
        error!("exits: the trapped access at {gpa:#x} is in no region");
        return Flow::Leave;
    };
    match partition.dispatch(vcpu, region.tag, gpa, cause) {
        // Done, or stopped part way through a repeated move with its progress in
        // the guest's own registers. Either way the guest resumes and needs
        // nothing from us.
        Ok(Outcome::Stepped | Outcome::Repeating) => Flow::Resume,
        Ok(Outcome::Faulted(fault)) => raise(vcpu, fault, interrupts),
        Err(error) => unserviceable(vcpu, partition, gpa, error),
    }
}

/// Steps the guest past an access to a device that no device can be asked
/// about.
///
/// Scoped by *region* rather than by which failure it was, and that is the
/// decision: this is reached only for an address the nested tables trap, which
/// means a device aperture this hypervisor interposed on. A guest access to one
/// of those in a shape the emulator cannot perform — a width or alignment the
/// device does not decode, an access that begins in the region and ends outside
/// it, a locked read-modify-write — is the guest doing something the
/// architecture leaves undefined for the device it aimed at. Undefined is not
/// fatal: real hardware answers such an access with unpredictable data and goes
/// on executing, so dropping it and resuming is the closest thing to that, and
/// ending the guest over it would stop a physical processor for good over one
/// guest instruction.
///
/// Enumerating the failures instead would be the wrong seam. The same variant
/// covers a malformed guest access and a device answering the wrong width, so
/// it cannot tell a guest's mistake from a hypervisor's, and every refusal the
/// emulator grows later would have to be classified again — while a failure
/// anywhere *other* than a trapped region stays fatal on its own path:
/// resolving the address, and a region nothing answers for, are both above
/// this.
///
/// Said once. The shapes that reach here are ones a guest can execute in a
/// loop, and a line per access through a serial port with interrupts masked
/// would be a worse denial of service than the halt this replaces.
pub(crate) fn unserviceable(
    vcpu: &mut Vcpu,
    partition: &Partition,
    gpa: PhysAddr,
    error: EmulateError,
) -> Flow {
    if REPORTED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
    {
        error!(
            "exits: the access to {gpa:#x} could not be performed: {error}; dropping it and \
             stepping the guest past it, and saying nothing about later ones"
        );
    }
    step_over(vcpu, partition, gpa)
}

/// Whether an unserviceable access has already been reported on this machine.
///
/// Never cleared: what it is for is that the first one is on the record and a
/// guest cannot turn the rest into a stall.
static REPORTED: AtomicBool = AtomicBool::new(false);

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
pub(crate) fn raise(vcpu: &mut Vcpu, fault: Fault, interrupts: &mut Pending) -> Flow {
    let vector = Vector::new(fault.vector());
    if let Some(address) = fault.address() {
        // What the handler reads to find out which address it has to describe. The
        // architecture publishes it in `CR2`, and nothing else carries it.
        vcpu.save_mut().cr2 = address;
        vcpu.soil(CleanBits::FAULT_ADDRESS);
    }
    interrupts.raise_exception(vcpu, Event::exception_with_code(vector, fault.code()));
    Flow::Resume
}

/// Steps the guest past an access this hypervisor is not going to perform.
///
/// Two callers and one answer, because for both of them the only alternative is
/// a guest faulting on the same instruction forever. The hypervisor's own
/// memory reads as a shared page of zeroes and has no page behind it to take a
/// write; and an access to a trapped region that no device can be asked about
/// has nothing to perform either. So the instruction is decoded for its length
/// alone, whatever it moved is dropped, and the guest carries on after it.
fn step_over(vcpu: &mut Vcpu, partition: &Partition, gpa: PhysAddr) -> Flow {
    let addressing = Addressing::from_save(vcpu.save());
    match partition.with_memory(addressing, |guest| emulate::next_rip(vcpu, guest)) {
        Ok(next) => {
            vcpu.save_mut().rip = next;
            Flow::Resume
        }
        Err(error) => {
            error!("exits: could not step past the access at {gpa:#x}: {error}");
            Flow::Leave
        }
    }
}
