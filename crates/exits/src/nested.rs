//! Guest physical addresses the second set of page tables had no answer for.

use log::error;
use npt::Resolution;
use partition::{Addressing, Partition};
use svm::exit::NestedPageFault;
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
/// again.
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
        Ok(_) => Flow::Resume,
        Err(error) => {
            error!("exits: the access to {gpa:#x} could not be performed: {error}");
            Flow::Leave
        }
    }
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
