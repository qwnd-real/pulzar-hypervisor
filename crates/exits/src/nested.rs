//! Guest physical addresses the second set of page tables had no answer for.

use core::sync::atomic::{AtomicU64, Ordering};

use descriptors::Vector;
use emulate::{EmulateError, Fault, Outcome as Performed};
use inject::Pending;
use log::error;
use npt::{Outcome, RegionTag};
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
    let reported = vcpu.control().exit_info_2;
    // Not folded into an address the machine has. A guest physical address with
    // bits above what the entry format holds is one nothing describes, and
    // truncating it would answer for a different page — which is a guest told
    // that an address it cannot have is ordinary memory.
    let Ok(gpa) = PhysAddr::try_new(reported) else {
        return unaddressable(reported);
    };
    match partition.resolve(gpa, cause) {
        Ok(Outcome::Filled) => Flow::Resume,
        Ok(Outcome::Refused) => step_over(vcpu, partition, gpa),
        Ok(Outcome::WalkRefused) => walk_refused(vcpu, gpa, interrupts),
        Ok(Outcome::Interposed { tag }) => interposed(vcpu, partition, tag, gpa, cause, interrupts),
        Ok(Outcome::Unaddressable) => unaddressable(reported),
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
///
/// `tag` is the name the tables gave the region, out of the fault itself: the
/// same answer that said the address is not the hardware's to serve said which
/// region it is in, so nothing here asks a second time and no window exists in
/// which the two answers could differ.
fn interposed(
    vcpu: &mut Vcpu,
    partition: &Partition,
    tag: RegionTag,
    gpa: PhysAddr,
    cause: NestedPageFault,
    interrupts: &mut Pending,
) -> Flow {
    match partition.dispatch(vcpu, tag, gpa, cause) {
        // Done, or stopped part way through a repeated move with its progress in
        // the guest's own registers. Either way the guest resumes and needs
        // nothing from us.
        Ok(Performed::Stepped | Performed::Repeating) => Flow::Resume,
        Ok(Performed::Faulted(fault)) => raise(vcpu, fault, interrupts),
        Err(error) => unserviceable(vcpu, partition, gpa, error),
    }
}

/// Gives the guest the page fault its own architecture owes an address it
/// cannot translate.
///
/// The one nested fault stepping the guest over does not answer. What faulted
/// is a walk of the guest's own page tables — the processor reading a table the
/// guest put in memory the hypervisor owns and writing the bit that records the
/// access — so the instruction is not what needs performing: the translation it
/// waits for is, and nothing here can produce one. Resuming re-executes the
/// instruction and faults on the same walk, and stepping past it leaves the
/// next instruction to walk the same table.
///
/// So the guest is told what a real processor tells it when a translation
/// cannot be completed, which is the honest answer to a guest that pointed its
/// own page tables at memory it does not own.
///
/// The address the handler is given is the *guest physical* address of the
/// table the walk stopped at, because the exit reports no linear address at all
/// and there is none to be recovered — computing one would mean walking the
/// very tables that cannot be walked. It names the page the guest has to fix,
/// which is what the handler is for; the error code says a present page could
/// not be written, and leaves the privilege bit clear because the exit
/// describes the walk's own access and not the guest instruction's.
fn walk_refused(vcpu: &mut Vcpu, gpa: PhysAddr, interrupts: &mut Pending) -> Flow {
    error!(
        "exits: the guest's own page tables are at {gpa:#x}, which is memory it may not write; \
         raising a page fault in it rather than faulting on the same walk for ever"
    );
    raise(vcpu, Fault::page(gpa.as_u64(), WALK_REFUSED), interrupts)
}

/// The page-fault error code a refused page-table walk is delivered with: a
/// write to a page that was present.
const WALK_REFUSED: u32 = 0b11;

/// Ends the guest over an address no machine could have given it.
///
/// Above what the processor can address, so nothing describes it and nothing
/// may: describing it would mean describing some other page, and a guest whose
/// access reached here has already been answered in a way no real machine
/// answers. There is no exception the architecture owes for it either — a real
/// machine simply has no such address — so the guest stops.
fn unaddressable(gpa: u64) -> Flow {
    error!(
        "exits: the guest reached {gpa:#x}, which is above the physical addresses this processor \
         has; no description of its memory can cover one"
    );
    Flow::Leave
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
/// Reported when the count doubles, and counted always. The shapes that reach
/// here are ones a guest can execute in a loop, so a line per access through a
/// serial port with interrupts masked would be a worse denial of service than
/// the halt this replaces — while a line for the first alone leaves every later
/// one invisible, which is what made a guest quietly dropping thousands of its
/// own accesses look like a guest that had dropped one. Doubling bounds the
/// output at one line per bit of the counter for the life of the machine.
pub(crate) fn unserviceable(
    vcpu: &mut Vcpu,
    partition: &Partition,
    gpa: PhysAddr,
    error: EmulateError,
) -> Flow {
    let seen = UNSERVICEABLE.fetch_add(1, Ordering::Relaxed) + 1;
    if seen.is_power_of_two() {
        error!(
            "exits: the access to {gpa:#x} could not be performed: {error}; dropping it and \
             stepping the guest past it — {seen} so far on this machine, and the next line comes \
             when that doubles"
        );
    }
    step_over(vcpu, partition, gpa)
}

/// How many accesses to a region this hypervisor answers for it has dropped
/// rather than performed, machine wide.
///
/// Never cleared: what it is for is that the first one is on the record, that a
/// guest cannot turn the rest into a stall, and that the rest are still
/// counted.
static UNSERVICEABLE: AtomicU64 = AtomicU64::new(0);

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
