//! The moves that repeat.
//!
//! A guest copying to or from a device does not do it one value at a time. It
//! points the index registers at the two ends, puts a count in the count
//! register, and executes one instruction that moves the lot. Emulating that
//! one repetition per exit would turn a four-kilobyte copy into a thousand
//! world switches.
//!
//! So repetitions are performed in a batch, and the batch stops at whichever
//! comes first: the count reaching zero, either end reaching a page boundary,
//! or the transaction budget running out.
//!
//! # Why stopping early is safe, and why the page boundary is where
//!
//! A repeated string instruction is restartable by design. The architecture
//! keeps the whole of its progress in the index and count registers, so a
//! processor interrupted part way through one resumes by executing the same
//! instruction again with those registers where they were left.
//!
//! A batch that stops early does exactly that: the registers are updated, the
//! instruction pointer is *not* advanced, and the guest re-executes the
//! instruction. It faults again on the next page and the next batch carries on.
//! The guest can take an interrupt between the two, which is what keeps a long
//! copy from holding a processor inside one exit handler.
//!
//! The page boundary is the natural place to stop because it is the first point
//! at which nothing can be assumed any more: the next page may translate
//! somewhere unrelated, may be answered for by a different device or by none,
//! and may not be described at all.
//!
//! # The budget counts transactions, not repetitions
//!
//! A ceiling of one page of repetitions is not a bound on anything that
//! matters. A repetition against a device is a callback into code this crate
//! knows nothing about, which may take microseconds; four thousand of them is a
//! processor held inside one exit handler for long enough to lose interrupts
//! and miss deadlines. So the budget is small and is spent on the accesses that
//! are actually expensive — the ones that reach a device — while repetitions
//! between two pages of ordinary memory, which is what a guest's own `memcpy`
//! looks like, cost far less and are allowed more of.
//!
//! # What a failure part-way through means
//!
//! Once one repetition is complete, the batch has changed guest state that
//! cannot be undone — and if that repetition read a device, state that could
//! not be undone even in principle. So a later failure must not be reported as
//! though nothing had happened: the registers are left describing exactly the
//! progress made, and the outcome says whether the guest owes a fault or
//! whether the hypervisor failed. A guest that owes a page fault gets one, with
//! the count and indices where the architecture would have left them, and
//! resuming after it carries the copy on.

use iced_x86::{Instruction, OpKind};

use crate::{
    EmulateError, Outcome,
    machine::{Cpu, Guest},
    mmio::Mmio,
    operand::{self, DESTINATION, Place, SOURCE},
    plan::{self, Plan, Reported},
    value::{Data, Width},
};

/// The most device transactions one exit performs.
///
/// Small on purpose. Each one is a callback into a device emulator whose cost
/// this crate cannot see, so the bound has to be on the number of them rather
/// than on repetitions in general. Sixty-four is enough that a driver writing a
/// descriptor ring finishes in one exit and few enough that no guest can hold a
/// processor for long by asking for more.
const TRANSACTIONS: u32 = 64;

/// The most repetitions one exit performs when none of them reaches a device.
///
/// A copy between two pages of the guest's own memory is a few instructions per
/// repetition and no callback, so it can afford far more of them — and stopping
/// early would cost an exit per batch for a copy the guest could almost have
/// done itself. Still bounded, because a bound that a guest can raise is not
/// one.
const REPETITIONS: u32 = 4096;

/// Bit of the guest's flags that says the index registers count downwards.
const DIRECTION: u64 = 1 << 10;

/// Performs as many repetitions as can be done in this exit.
///
/// # Errors
///
/// [`EmulateError::Operand`] for a prefix the architecture does not give these
/// instructions a meaning for, and whatever resolving, reading or writing
/// either end reports — but only where nothing has been committed yet. Once a
/// repetition is complete, a later failure comes back as an outcome describing
/// the progress made rather than as an error that looks retryable.
pub(crate) fn perform(
    mmio: &Mmio,
    cpu: &mut impl Cpu,
    guest: &impl Guest,
    instruction: &Instruction,
    width: Width,
    fault: Reported,
) -> Result<Outcome, EmulateError> {
    let rip = cpu.save().rip;
    if instruction.has_repne_prefix() {
        // The architecture gives this prefix a meaning on the comparing string
        // instructions and none on these. Performing it as though it were the
        // other prefix would repeat something the guest did not ask to repeat.
        return Err(EmulateError::Operand {
            rip,
            operand: DESTINATION,
        });
    }
    let repeated = instruction.has_rep_prefix();
    let backwards = cpu.save().rflags & DIRECTION != 0;
    let counting = address_width(instruction).ok_or(EmulateError::Operand {
        rip,
        operand: DESTINATION,
    })?;

    let mut budget = Budget::new();
    // What the architecture keeps of this instruction's progress, so that a
    // failure part-way through can say exactly how much of it happened.
    let mut done = 0_u32;
    // How far one repetition moves each index register, which is the element width
    // in whichever direction the flag says.
    let by = if backwards {
        -width.step()
    } else {
        width.step()
    };
    // Both ends, resolved once. They are stepped from here rather than re-resolved,
    // and rebuilt only when a batch's page-bounded invariant stops holding.
    let mut walk = None;

    while budget.left() {
        if repeated && left(cpu, counting) == 0 {
            return Ok(Outcome::Stepped);
        }

        // Resolving is the expensive part — a walk of the guest's page tables per
        // end — so it happens once per batch rather than once per element. The
        // first time round there is nothing to step, and afterwards there is
        // nothing to step only if a previous step left the page it was licensed
        // for, which is also where the batch is about to stop.
        let plan = if let Some(stepped) = walk {
            stepped
        } else {
            let planned = match Plan::moving(
                mmio,
                cpu,
                guest,
                instruction,
                (width, width),
                (SOURCE, DESTINATION),
            ) {
                Ok(plan) => plan,
                Err(error) => return stop(done, error),
            };
            // Only the first repetition is the one the hardware reported: the ones
            // after it are this crate's own accesses, made because the guest asked
            // for them and not because anything trapped. So the fault is matched
            // against the first and the rest carry on from it.
            if done == 0 {
                planned.authenticate(rip, fault.gpa, fault.cause)?;
            }
            planned
        };

        if plan.from.interposed()
            && !operand::infallible(plan.to.place)
            && let Err(error) = preflight(mmio, guest, &plan)
        {
            return stop(done, error);
        }
        budget.spend(&plan);

        let value = match operand::load(mmio, cpu, guest, plan.from.place, width) {
            Ok(value) => value,
            Err(error) => return stop(done, error),
        };
        // Past this point the source may have been consumed, so a failure is
        // reported as a committed one rather than as something to retry.
        if let Err(error) = operand::store(mmio, cpu, guest, plan.to.place, value) {
            return committed(done, &plan, error);
        }
        advance(cpu, instruction, width, backwards, counting)?;
        done += 1;

        if !repeated {
            return Ok(Outcome::Stepped);
        }
        decrement(cpu, counting)?;
        if left(cpu, counting) == 0 {
            return Ok(Outcome::Stepped);
        }
        // Both ends move together or the batch ends. A step that leaves the page
        // its translation was established for is exactly the point at which nothing
        // about the next element can be assumed any more — the next page may
        // translate somewhere unrelated, be answered for by a different device, or
        // not be described at all — so the guest re-executes the instruction and
        // the next exit resolves it afresh.
        walk = plan.stepped(by);
        if walk.is_none() {
            return Ok(Outcome::Repeating);
        }
    }
    Ok(Outcome::Repeating)
}

/// What a failure means, given that `done` repetitions are already complete.
///
/// Two questions, and they are independent. Is this the guest's fault to take
/// or the hypervisor's failure to report? And has anything already happened
/// that the caller must not undo by retrying?
///
/// A guest fault is a guest fault whether or not any repetition completed. Zero
/// progress does not make an unmapped guest page into a hypervisor failure — it
/// is the ordinary way a demand-paged operating system grows its address space,
/// and a real processor delivers `#PF` and lets the handler map the page.
/// Reporting it as an error would stop a guest that hardware would simply have
/// faulted.
///
/// The progress matters to what is *left* rather than to what kind of answer
/// this is: the architecture keeps it in the index and count registers, which
/// are already where they should be, so the same outcome serves both cases.
fn stop(done: u32, error: EmulateError) -> Result<Outcome, EmulateError> {
    if let Some(fault) = guest_fault(&error) {
        return Ok(Outcome::Faulted(fault));
    }
    // The hypervisor's own failure. With nothing committed it stands as it is and
    // whoever asked may retry or report it freely; with repetitions behind it the
    // registers already describe progress the error does not mention, so it is
    // marked as a failure that has left state behind.
    Err(if done == 0 {
        error
    } else {
        EmulateError::Partial {
            completed: done,
            cause: describe(&error),
        }
    })
}

/// One line naming what went wrong, for a failure that is being reported
/// alongside the progress it stopped after.
///
/// The original error's own data cannot travel with it — a partial failure has
/// to carry the count, and nesting the error would make the type recursive — so
/// what is kept is the identity of the failure rather than its details. The
/// details were already logged by whoever produced them.
const fn describe(error: &EmulateError) -> &'static str {
    match error {
        EmulateError::Inadmissible { .. } => "the device would not answer the access",
        EmulateError::Span { .. } => "an element did not lie in one region",
        EmulateError::Discarded { .. } => "the destination would not take the write",
        EmulateError::Committed { .. } => "a device read had already happened",
        EmulateError::Memory(_) => "the guest's memory could not be reached",
        EmulateError::WidthMismatch { .. } => "two ends disagreed about the width",
        EmulateError::NoSuchRegion { .. } => "the region no longer exists",
        _ => "the instruction could not be carried on",
    }
}

/// The exception a guest is owed for this failure, if a real processor would
/// have raised one rather than the hypervisor having simply failed.
///
/// A guest whose page tables do not describe an address it named takes a page
/// fault; that is not a hypervisor failure, it is the ordinary way a
/// demand-paged operating system grows its address space. Everything else here
/// — a device that cannot answer, a region that vanished, a span nothing can
/// describe — is this hypervisor being unable to do something, and inventing a
/// guest exception for one of those would tell the guest about a fault in its
/// own code that does not exist.
fn guest_fault(error: &EmulateError) -> Option<plan::Fault> {
    match error {
        EmulateError::Memory(memory::MemoryError::Untranslated { linear }) => {
            // Present bit clear, and the error code says no more than that: the
            // hypervisor is not in a position to know whether the access was a
            // user-mode one or what the guest's own permissions were, and a code
            // that claimed to would be a code the guest could not trust.
            Some(plan::Fault::page(*linear, 0))
        }
        _ => None,
    }
}

/// A failure once the source has already been read.
///
/// A device source makes this the one case that must never look retryable: the
/// device has answered, the destination refused the value, and reading the
/// device again would not necessarily produce the same one. That outranks the
/// progress count, because a caller that retried would consume a second read
/// whether or not earlier repetitions succeeded.
///
/// A source in ordinary memory is only as bad as the progress behind it:
/// reading guest memory twice gives the same answer, so this goes back through
/// the same path everything else does.
fn committed(done: u32, plan: &Plan, error: EmulateError) -> Result<Outcome, EmulateError> {
    let Place::Device { gpa, .. } = plan.from.place else {
        return stop(done, error);
    };
    Err(EmulateError::Committed {
        gpa: gpa.as_u64(),
        consumed: plan.from.width.bytes(),
        cause: describe(&error),
    })
}

/// Proves the destination will accept a write before the source is consumed.
///
/// The same reasoning as the scalar path's: a device read cannot be given back,
/// so a destination that would refuse has to be found out about first.
fn preflight(mmio: &Mmio, guest: &impl Guest, plan: &Plan) -> Result<(), EmulateError> {
    match plan.to.place {
        Place::Device {
            index, offset, gpa, ..
        } => mmio.admits(index, offset, gpa, plan.to.width),
        Place::Memory(linear) => {
            guest
                .writable(linear, plan.to.width)?
                .then_some(())
                .ok_or(EmulateError::Discarded {
                    linear,
                    bytes: plan.to.width.bytes(),
                })
        }
        Place::Gpr(_) | Place::Vector(_) | Place::Immediate(_) => Ok(()),
    }
}

/// What one exit is allowed to spend, and on what.
///
/// Two counters rather than one, because the two kinds of repetition cost
/// entirely different amounts and a single bound would either stop a memory
/// copy far too early or let a device copy run far too long.
#[derive(Clone, Copy, Debug)]
struct Budget {
    transactions: u32,
    repetitions: u32,
}

impl Budget {
    /// A full budget.
    const fn new() -> Self {
        Self {
            transactions: TRANSACTIONS,
            repetitions: REPETITIONS,
        }
    }

    /// Whether anything is left to spend.
    const fn left(self) -> bool {
        self.transactions > 0 && self.repetitions > 0
    }

    /// Charges one repetition, and a transaction for each end of it that
    /// reaches a device.
    const fn spend(&mut self, plan: &Plan) {
        self.repetitions = self.repetitions.saturating_sub(1);
        let ends = plan.from.interposed() as u32 + plan.to.interposed() as u32;
        self.transactions = self.transactions.saturating_sub(ends);
    }
}

/// Moves each index register the instruction walks on by one repetition.
fn advance(
    cpu: &mut impl Cpu,
    instruction: &Instruction,
    width: Width,
    backwards: bool,
    counting: Width,
) -> Result<(), EmulateError> {
    let by = if backwards {
        -width.step()
    } else {
        width.step()
    };
    for operand in 0..instruction.op_count() {
        if let Some(register) = operand::index(instruction.op_kind(operand)) {
            step(cpu, register, by, counting)?;
        }
    }
    Ok(())
}

/// Takes one off the count register.
fn decrement(cpu: &mut impl Cpu, counting: Width) -> Result<(), EmulateError> {
    step(cpu, crate::gpr::RCX, -1, counting)
}

/// What is left of the count register, at the width the address size gives it.
fn left(cpu: &impl Cpu, counting: Width) -> u64 {
    cpu.gpr(crate::gpr::RCX) & counting.mask()
}

/// Adds to a register, at the width the address size gives it.
///
/// The width is not decoration. An instruction using 32-bit addresses walks
/// `ECX`, `ESI` and `EDI`, and a write to one of those clears the upper half of
/// the register rather than carrying into it — which is the rule
/// [`crate::gpr::merge`] already knows, applied here rather than restated.
fn step(cpu: &mut impl Cpu, register: u8, by: i64, counting: Width) -> Result<(), EmulateError> {
    let whole = cpu.gpr(register);
    let value = (whole & counting.mask()).wrapping_add_signed(by);
    let merged = crate::gpr::merge(whole, Data::from_u64(value, counting))?;
    cpu.set_gpr(register, merged);
    Ok(())
}

/// How wide the addresses this instruction walks are, out of the first implicit
/// operand that says so.
///
/// All of an instruction's implicit string operands use the same address size,
/// so the first one to name a width names all of them.
fn address_width(instruction: &Instruction) -> Option<Width> {
    (0..instruction.op_count()).find_map(|operand| match instruction.op_kind(operand) {
        OpKind::MemorySegSI | OpKind::MemoryESDI => Some(Width::Word),
        OpKind::MemorySegESI | OpKind::MemoryESEDI => Some(Width::Long),
        OpKind::MemorySegRSI | OpKind::MemoryESRDI => Some(Width::Quad),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::{Budget, REPETITIONS, TRANSACTIONS, guest_fault};
    use crate::{EmulateError, plan::Fault};

    #[test]
    fn a_fresh_budget_has_both_allowances() {
        let budget = Budget::new();
        assert!(budget.left());
        assert_eq!(budget.transactions, TRANSACTIONS);
        assert_eq!(budget.repetitions, REPETITIONS);
    }

    #[test]
    fn the_transaction_budget_is_far_smaller_than_the_repetition_one() {
        // The whole point of having two: a device callback is expensive enough
        // that a page of them is not a latency bound worth having.
        const {
            assert!(
                TRANSACTIONS < REPETITIONS / 8,
                "a device batch must be bounded much more tightly than a memory copy"
            );
        }
    }

    #[test]
    fn an_untranslated_address_is_a_page_fault_the_guest_is_owed() {
        // The case that must not stop the hypervisor: a demand-paged guest hits
        // this constantly and a real processor simply faults into its handler.
        let error = EmulateError::Memory(memory::MemoryError::Untranslated { linear: 0x1234 });
        assert_eq!(guest_fault(&error), Some(Fault::page(0x1234, 0)));
    }

    #[test]
    fn a_hypervisor_failure_is_not_turned_into_a_guest_exception() {
        // Inventing a fault for one of these would tell the guest about a problem
        // in its own code that does not exist.
        for error in [
            EmulateError::NoSuchRegion { index: 3 },
            EmulateError::Inadmissible {
                gpa: 0x1000,
                bytes: 4,
                reason: crate::Inadmissible::Width,
            },
            EmulateError::Span {
                linear: 0x1000,
                bytes: 4,
                reason: crate::Spanning::TwoRegions,
            },
            EmulateError::Memory(memory::MemoryError::Undescribed { gpa: 0x2000 }),
        ] {
            assert_eq!(
                guest_fault(&error),
                None,
                "{error:?} is the hypervisor failing, not the guest faulting"
            );
        }
    }
}
