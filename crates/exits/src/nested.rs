//! Guest physical addresses the second set of page tables had no answer for.

use log::error;
use npt::Resolution;
use partition::Partition;
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
        // A device the hypervisor interposes on, with nothing yet interposing.
        // Resuming would fault at the same address forever.
        Ok(Resolution::Trapped) => {
            error!("exits: no device handles trapped access at {gpa:#x}");
            Flow::Leave
        }
        Err(error) => {
            error!("exits: nested fault at {gpa:#x} could not be resolved: {error}");
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
    match partition.with_memory(vcpu.save(), |guest| emulate::next_rip(vcpu, guest)) {
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
