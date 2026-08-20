//! The model-specific registers the guest is answered for rather than allowed
//! to reach.
//!
//! Six kinds are answered here, and they fail differently. The timestamp
//! counter and its adjustment register are backed by the control block's
//! offset, keeping native `RDTSC` and `RDTSCP` reads on the zero-exit path. The
//! two registers that decide whether the virtualization extension may be used
//! are answered with a lie the guest is entitled to believe, because a guest
//! allowed at the machine's own copies could turn the extension off underneath
//! the hypervisor running it. The extended feature register is the guest's own,
//! and is answered by hiding one bit of it and forcing that same bit back on
//! whatever the guest writes. The page-attribute table is the guest's own too,
//! and is answered out of the save area, because that is where the processor
//! reads the guest's memory types from while nested paging is on. The
//! memory-type range registers are answered out of [`crate::mtrr`], for the
//! same reason the page-attribute table is not answered from the register of
//! that name: they are the host's, and shared with the other thread of this
//! core. The interrupt controller's registers are answered by the guest's own
//! emulated controller, and an access the architecture does not allow is a
//! general protection fault the guest is given rather than an error the host
//! reports. The controller's doorbell is refused outright: a read of it faults
//! on the machine anyway, and a write of it would poke whichever physical
//! processor the guest named — which no guest is owed.
//!
//! A seventh kind is not answered at all so much as forwarded. The permission
//! map covers three ranges of the index space and an access outside all three
//! is intercepted whatever the map holds, so those arrive here whether this
//! crate wants them or not; they reach the machine's own register, and a
//! register the machine has none of refuses the access and the guest takes the
//! fault for it.
//!
//! # The page-attribute table is two registers with one name
//!
//! With nested paging on, the memory type of a guest access is decided by the
//! guest's page tables against the guest's table of types — which the processor
//! takes from the save area — and the register called `IA32_PAT` holds the
//! *host's*. There is no sense in which a guest can be allowed at it. Left
//! unintercepted, one guest `WRMSR` reprograms the memory types every mapping
//! the hypervisor itself is using is interpreted through, with none of the
//! cache-flush transition the architecture prescribes for that, while the
//! guest's own types stay exactly as they were. Both halves of that are silent:
//! the guest reads back what it wrote, believes it has write-combining, and
//! gets whatever it had before.
//!
//! # The extension is presented as firmware-disabled, consistently
//!
//! `CPUID` denies the extension exists, and [`VmCr`] is the one place a guest
//! can look afterwards — firmware does look. It is answered as a machine whose
//! firmware turned virtualization off and locked that decision, which is a
//! state guests already know how to be told about: the lock and the disable
//! both read back set, and the key that could lift the lock is answered as one
//! no firmware ever armed.
//!
//! The extended feature register has to agree with that story. Its
//! virtualization-enable bit is genuinely set, because the processor refuses to
//! enter a guest whose save area has it clear, so a guest reading the register
//! would otherwise find the extension switched on in the very machine whose
//! `CPUID` says it has none. It is hidden on the way out and forced back on the
//! way in, which leaves the guest a consistent view and the processor the bit
//! it insists on.

use inject::Pending;
use log::{error, trace};
use svm::{
    CleanBits, Event,
    avic::AVIC_DOORBELL,
    msr::{
        EFER, EFER_RESERVED, IA32_PAT, IA32_TSC, IA32_TSC_ADJUST, SVM_KEY, TSC_RATIO, VM_CR, VmCr,
    },
    permissions::{MsrAccess, msrpm_position},
};
use vcpu::{Flow, Vcpu};
use x86_64::registers::{control::Cr0Flags, model_specific::EferFlags};

use crate::{
    advance,
    mtrr::{self, Mtrrs},
};

/// What one processor's guest has been told about this machine's virtualization
/// extension.
///
/// Only the emulated [`VmCr`] needs remembering. The extended feature register
/// is the guest's own and lives in its save area, and the key register holds
/// nothing a guest may read back — so this is the whole of the state, and it is
/// per processor because the register it stands for is.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Virtualization {
    vm_cr: VmCr,
}

impl Virtualization {
    /// A guest that has written none of these registers yet.
    pub(crate) const fn new() -> Self {
        Self { vm_cr: DISABLED }
    }

    /// Puts the registers back the way a processor coming out of reset has
    /// them.
    ///
    /// `INIT` clears the three switches that have nothing to do with the
    /// extension and leaves the lock alone, which is what
    /// [`Virtualization::new`] already holds. It does not clear the disable
    /// either, because that too is only clearable while the lock is open — so a
    /// guest whose processor has just been started finds the extension exactly
    /// as unavailable as it was before.
    pub(crate) const fn reset(&mut self) {
        self.vm_cr = DISABLED;
    }

    /// Answers one intercepted register access.
    ///
    /// The routing rather than the answering: the four registers this type is
    /// about are answered below, and everything else is handed to whichever
    /// handler owns it. `mtrrs` is this processor's memory-type ranges, which
    /// are its own state rather than this type's — the two kinds of
    /// register have nothing in common but arriving through the same exit.
    pub(crate) fn exit(
        &mut self,
        vcpu: &mut Vcpu,
        mtrrs: &mut Mtrrs,
        interrupts: &mut Pending,
    ) -> Flow {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a model-specific register index is the low half of RCX; the architecture ignores the rest"
        )]
        let msr = vcpu.registers().rcx as u32;
        if let Some(register) = Timestamp::of(msr) {
            return timestamp(vcpu, register, interrupts);
        }
        if vlapic::claims(msr) {
            return controller(vcpu, msr, interrupts);
        }
        if mtrr::claims(msr) {
            return ranges(vcpu, mtrrs, msr, interrupts);
        }
        if msr == AVIC_DOORBELL {
            // A write would poke whichever physical processor the guest named,
            // and a read faults on the machine anyway — so both directions
            // are answered with the fault the access earns.
            trace!("exits: refused the guest's access to the interrupt doorbell");
            return refuse(vcpu, interrupts);
        }
        let Some(register) = Hidden::of(msr) else {
            if msrpm_position(msr).is_some() {
                // Every bit this hypervisor sets in the permission map is set for
                // one of the registers answered here, so a register the map covers
                // arriving here is the map disagreeing with this handler rather
                // than anything the guest did.
                error!("exits: unexpected intercepted MSR {msr:#x}");
                return Flow::Leave;
            }
            return passthrough(vcpu, msr, interrupts);
        };
        let answered = match MsrAccess::from_exit_info(vcpu.control().exit_info_1) {
            MsrAccess::Read => {
                let value = self.read(vcpu, register);
                answer(vcpu, value);
                Ok(())
            }
            MsrAccess::Write => self.write(vcpu, register, written(vcpu)),
        };
        match answered {
            Ok(()) => {
                advance(vcpu, BYTES);
                Flow::Resume
            }
            Err(Fault) => refuse(vcpu, interrupts),
        }
    }

    /// What the guest reads from one of them.
    ///
    /// None of the four can fault a read. Two are the emulated answer and
    /// nothing else; the other two are the guest's own registers out of its
    /// save area, one of them with the single bit removed that would
    /// contradict what it has been told.
    fn read(self, vcpu: &Vcpu, register: Hidden) -> u64 {
        match register {
            Hidden::VmCr => self.vm_cr.into_bits(),
            // Write-only in the strong sense — a read gives zero rather than
            // what firmware stored — so this is both what the guest would see on
            // this machine and the answer that keeps it consistent with a lock
            // nothing can lift.
            Hidden::Key => NO_KEY,
            Hidden::Efer => vcpu.save().efer & !SVME,
            // The guest's own table of memory types, which is a field of the save
            // area and not the register of that name. Answering from the register
            // would hand the guest the host's types and contradict its own last
            // write.
            Hidden::Pat => vcpu.guest_pat(),
        }
    }

    /// What a write of one of them does.
    fn write(&mut self, vcpu: &mut Vcpu, register: Hidden, value: u64) -> Result<(), Fault> {
        match register {
            Hidden::VmCr => self.write_vm_cr(value),
            // Dropped, and this is the register where dropping matters most: the
            // guest's write would land on the machine's own key, storing one
            // while the lock is open or lifting the lock while it is shut. Either
            // would put the host's extension at the disposal of the guest running
            // under it.
            Hidden::Key => {
                trace!("exits: dropped the guest's write of the extension's key register");
                Ok(())
            }
            Hidden::Efer => write_efer(vcpu, value),
            Hidden::Pat => write_pat(vcpu, value),
        }
    }

    /// Stores what a write of [`VmCr`] leaves behind.
    ///
    /// The lock is reported set, and a set lock is what makes this emulation
    /// short: writes of the lock itself and of the disable beside it are
    /// discarded, with no fault and no other trace, exactly as they are on a
    /// machine whose firmware locked the register. Reading it back is the only
    /// way a guest can tell, and what it reads back is unchanged.
    ///
    /// The other three switches are the guest's — an external debug port, the
    /// redirection of `INIT` into an exception, and A20 masking — and are
    /// stored so that a guest which sets one reads it back. None of them is
    /// acted on: each describes machine behaviour this hypervisor does not
    /// present to a guest at all, and a guest that could really disable the
    /// debug port would be reaching past its own boundary to do it.
    fn write_vm_cr(&mut self, value: u64) -> Result<(), Fault> {
        if value & VmCr::RESERVED != 0 {
            return Err(Fault);
        }
        self.vm_cr = VmCr::from_bits((value & !VmCr::LOCKED) | VmCr::LOCKED);
        Ok(())
    }
}

/// One of the registers that defines the guest's timestamp-counter domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Timestamp {
    /// The timestamp counter itself.
    Counter,
    /// The cumulative adjustment made to the counter.
    Adjust,
    /// SVM's guest frequency ratio, which the hidden extension must not expose.
    Ratio,
}

impl Timestamp {
    /// Which timestamp register an address names.
    const fn of(msr: u32) -> Option<Self> {
        match msr {
            IA32_TSC => Some(Self::Counter),
            IA32_TSC_ADJUST => Some(Self::Adjust),
            TSC_RATIO => Some(Self::Ratio),
            _ => None,
        }
    }
}

/// Answers an intercepted access to the guest's timestamp-counter state.
fn timestamp(vcpu: &mut Vcpu, register: Timestamp, interrupts: &mut Pending) -> Flow {
    let outcome = match MsrAccess::from_exit_info(vcpu.control().exit_info_1) {
        MsrAccess::Read => read_timestamp(vcpu, register).map(|value| answer(vcpu, value)),
        MsrAccess::Write => write_timestamp(vcpu, register, written(vcpu)),
    };
    match outcome {
        Ok(()) => {
            advance(vcpu, BYTES);
            Flow::Resume
        }
        Err(TimestampError::Fault) => refuse(vcpu, interrupts),
        Err(TimestampError::Controller(error)) => {
            error!("exits: the guest's timestamp state could not be updated: {error}");
            Flow::Leave
        }
    }
}

/// What the guest reads from one of its timestamp registers.
fn read_timestamp(vcpu: &Vcpu, register: Timestamp) -> Result<u64, TimestampError> {
    match register {
        Timestamp::Counter => Ok(vcpu.guest_timestamp()),
        Timestamp::Adjust if processor::features().contains(processor::Features::TSC_ADJUST) => {
            Ok(vcpu.tsc_adjust())
        }
        Timestamp::Adjust | Timestamp::Ratio => Err(TimestampError::Fault),
    }
}

/// What a guest write to one of its timestamp registers does.
fn write_timestamp(vcpu: &mut Vcpu, register: Timestamp, value: u64) -> Result<(), TimestampError> {
    match register {
        Timestamp::Counter => {
            let old_offset = vcpu.tsc_offset();
            let new_offset = offset_for_timestamp(value, processor::timestamp());
            let adjustment = offset_adjustment(old_offset, new_offset);
            rebase_timestamp(vcpu, new_offset, vcpu.tsc_adjust().wrapping_add(adjustment))
        }
        Timestamp::Adjust if processor::features().contains(processor::Features::TSC_ADJUST) => {
            let adjustment = offset_adjustment(vcpu.tsc_adjust(), value);
            rebase_timestamp(vcpu, vcpu.tsc_offset().wrapping_add(adjustment), value)
        }
        Timestamp::Adjust | Timestamp::Ratio => Err(TimestampError::Fault),
    }
}

/// Publishes one timestamp-offset change after preserving any armed deadline.
fn rebase_timestamp(
    vcpu: &mut Vcpu,
    new_offset: u64,
    new_adjust: u64,
) -> Result<(), TimestampError> {
    let adjustment = offset_adjustment(vcpu.tsc_offset(), new_offset);
    vlapic::adjust_deadline(adjustment).map_err(TimestampError::Controller)?;
    vcpu.set_tsc_offset(new_offset);
    vcpu.set_tsc_adjust(new_adjust);
    Ok(())
}

/// The offset that makes `physical` read as `timestamp` to the guest.
const fn offset_for_timestamp(timestamp: u64, physical: u64) -> u64 {
    timestamp.wrapping_sub(physical)
}

/// The wrapping amount by which an offset changed.
const fn offset_adjustment(old: u64, new: u64) -> u64 {
    new.wrapping_sub(old)
}

/// Why a timestamp register access could not be answered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TimestampError {
    /// The architecture requires the guest to take a general protection fault.
    Fault,
    /// The physical deadline could not be kept coherent with the new offset.
    Controller(vlapic::VlapicError),
}

/// One of the registers answered here rather than the machine's own.
///
/// Named rather than matched on as three addresses at each use site, because
/// the two halves of the answer — what a read gives and what a write does — are
/// written separately and have to agree about which register they are talking
/// about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Hidden {
    /// The register saying whether the extension may be used at all.
    VmCr,
    /// The write-only register a lock on it could be lifted through.
    Key,
    /// The extended feature register, which holds the enable bit itself.
    Efer,
    /// The page-attribute table, whose guest copy the processor takes from the
    /// save area while nested paging is on.
    Pat,
}

impl Hidden {
    /// Which of them an address names, or `None` for an address answered
    /// elsewhere.
    const fn of(msr: u32) -> Option<Self> {
        match msr {
            VM_CR => Some(Self::VmCr),
            SVM_KEY => Some(Self::Key),
            EFER => Some(Self::Efer),
            IA32_PAT => Some(Self::Pat),
            _ => None,
        }
    }
}

/// Bytes in `RDMSR` and in `WRMSR`, for a processor that does not report the
/// address after an intercepted instruction. The same length for both, which is
/// why one constant answers for either direction.
const BYTES: u64 = 2;

/// What the guest reads from [`VmCr`] before it has written anything, and what
/// it reads for the rest of its life through the two bits that matter.
///
/// Firmware disabled the extension and locked that decision. The lock is what
/// makes the answer stable: with it set, the architecture discards writes of
/// both bits, so a guest cannot even attempt to argue.
const DISABLED: VmCr = VmCr::new().with_lock(true).with_svm_disabled(true);

/// What a read of the key register answers, which is what a machine whose
/// firmware armed no key holds.
const NO_KEY: u64 = 0;

/// The bit of the extended feature register that enables the extension, which
/// the guest never sees and never clears.
const SVME: u64 = EferFlags::SECURE_VIRTUAL_MACHINE_ENABLE.bits();

/// The bit of the extended feature register that requests long mode, which
/// cannot be changed while paging is on.
const LME: u64 = EferFlags::LONG_MODE_ENABLE.bits();

/// The bit of the extended feature register the processor reports long mode as
/// active through. It is read-only: the processor sets it when paging is
/// enabled with long mode requested, and a write of the register cannot move
/// it.
const LMA: u64 = EferFlags::LONG_MODE_ACTIVE.bits();

/// A guest access the architecture answers with a general protection fault.
///
/// Carries nothing: every one of them is the same exception with the same error
/// code, and the guest is told no rather than told why.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Fault;

/// Stores what a write of the extended feature register leaves the guest
/// running with.
///
/// Two things a guest may not do, both of which really are faults on the
/// machine underneath and both of which would otherwise be discovered much
/// later and much worse. A reserved bit is refused because the processor's own
/// entry check refuses a save area holding one — the guest would be stopped by
/// a control block the hypervisor could not enter, rather than by the exception
/// its own architecture promised it. And long mode cannot be requested or
/// withdrawn while paging is enabled, because that is the one transition the
/// processor cannot make while it is translating addresses.
///
/// What is stored then differs from what was written in two bits. The
/// virtualization-enable bit is forced on, without which the next entry is
/// refused outright; the guest cannot tell, because it is hidden again on the
/// way out. And the long-mode-active bit keeps the value the processor gave it,
/// because it is the processor's report rather than a request — a guest that
/// wrote it would otherwise be telling itself it had reached a mode it has not.
fn write_efer(vcpu: &mut Vcpu, value: u64) -> Result<(), Fault> {
    if value & EFER_RESERVED != 0 {
        return Err(Fault);
    }
    let current = vcpu.save().efer;
    let paging = Cr0Flags::from_bits_retain(vcpu.save().cr0).contains(Cr0Flags::PAGING);
    if paging && (value ^ current) & LME != 0 {
        return Err(Fault);
    }
    let efer = (value & !(SVME | LMA)) | (current & LMA) | SVME;
    if efer == current {
        return Ok(());
    }
    vcpu.save_mut().efer = efer;
    vcpu.soil(CleanBits::CONTROL_REGISTERS);
    trace!("exits: the guest's EFER is now {efer:#x}");
    Ok(())
}

/// Stores the table of memory types the guest's own mappings are interpreted
/// through.
///
/// Straight into the save area, which is where the processor reads it from —
/// and, just as importantly, nowhere near the register of the same name, which
/// is the host's and whose meaning every mapping this hypervisor uses depends
/// on.
///
/// A value with a byte that names no memory type is refused rather than stored,
/// for the reason a reserved bit of the extended feature register is: the
/// processor's own entry check reads this field whenever nested paging is on,
/// so storing one would stop the guest with a control block that cannot be
/// entered instead of giving it the fault its architecture promised.
///
/// Nothing is flushed. The transition the architecture prescribes for changing
/// memory types under live mappings is the *guest's* to perform — it is the one
/// whose mappings are being reinterpreted, and `MtrrLib`, Windows and Linux all
/// perform it around their own write — and this hypervisor's own mappings are
/// not affected at all, which is the whole point of the interception.
fn write_pat(vcpu: &mut Vcpu, value: u64) -> Result<(), Fault> {
    match vcpu.set_guest_pat(value) {
        Ok(()) => {
            trace!("exits: the guest's page-attribute table is now {value:#x}");
            Ok(())
        }
        Err(invalid) => {
            trace!("exits: refusing the guest's page-attribute table {value:#x}: {invalid}");
            Err(Fault)
        }
    }
}

/// Answers an access to a register the permission map does not reach.
///
/// The map covers three ranges of the index space and the fourth vector in it
/// covers nothing, so an access to a register outside all three is intercepted
/// whatever the map says: there is no bit to clear for one of those, and a
/// hypervisor that did not answer them could not run a guest that touches one.
///
/// So the machine's own register answers, which is the honest answer in both
/// directions. A guest naming a register this machine really has gets what the
/// machine holds; a guest naming one of the very many indices that are not
/// registers at all gets the general protection fault the access would have
/// raised had no hypervisor been there. Which of the two an index is cannot be
/// decided here — only the processor knows — so the access is attempted and its
/// refusal is caught rather than predicted.
fn passthrough(vcpu: &mut Vcpu, msr: u32, interrupts: &mut Pending) -> Flow {
    let outcome = match MsrAccess::from_exit_info(vcpu.control().exit_info_1) {
        MsrAccess::Read => probe::read(msr).map(|value| answer(vcpu, value)),
        MsrAccess::Write => probe::write(msr, written(vcpu)),
    };
    match outcome {
        Ok(()) => {
            advance(vcpu, BYTES);
            Flow::Resume
        }
        Err(fault) => {
            trace!("exits: {fault}, so the guest takes the exception for it");
            refuse(vcpu, interrupts)
        }
    }
}

/// Answers an access to the guest's own interrupt controller.
///
/// The guest is given a general protection fault for anything the architecture
/// refuses — a reserved index, a read of a write-only register, a reserved bit
/// written non-zero — because that is what the access would have raised on real
/// hardware, and a guest probing its controller relies on being told no.
fn controller(vcpu: &mut Vcpu, msr: u32, interrupts: &mut Pending) -> Flow {
    let outcome = match MsrAccess::from_exit_info(vcpu.control().exit_info_1) {
        MsrAccess::Read => {
            vlapic::read_msr(msr, vcpu.tsc_offset()).map(|value| answer(vcpu, value))
        }
        MsrAccess::Write => vlapic::write_msr(msr, written(vcpu), vcpu.tsc_offset()),
    };
    match outcome {
        Ok(()) => {
            advance(vcpu, BYTES);
            Flow::Resume
        }
        Err(vlapic::VlapicError::Fault) => refuse(vcpu, interrupts),
        Err(error) => {
            error!("exits: the guest's controller could not answer for {msr:#x}: {error}");
            Flow::Leave
        }
    }
}

/// Answers an access to the guest's own memory-type ranges.
///
/// The guest is given a general protection fault for anything real hardware
/// refuses — a range this machine does not have, a write of the read-only
/// capability register, a reserved bit set, a memory type the architecture does
/// not define — because a guest programming its ranges probes them, and one
/// that silently accepted a range it cannot have would go on to trust it.
///
/// The virtual `SYSCFG` switch for the routing bits is read here, once per
/// access, because a guest may change it between one access and the next.
fn ranges(vcpu: &mut Vcpu, mtrrs: &mut Mtrrs, msr: u32, interrupts: &mut Pending) -> Flow {
    let routing = mtrrs.routing();
    let answered = match MsrAccess::from_exit_info(vcpu.control().exit_info_1) {
        MsrAccess::Read => mtrrs.read(msr, routing).map(|value| answer(vcpu, value)),
        MsrAccess::Write => mtrrs.write(msr, written(vcpu), routing),
    };
    let Ok(()) = answered else {
        trace!("exits: refusing the guest's access to memory-type register {msr:#x}");
        return refuse(vcpu, interrupts);
    };
    advance(vcpu, BYTES);
    Flow::Resume
}

/// Gives the guest the exception its access earned.
///
/// The instruction pointer is deliberately not advanced: the guest takes the
/// exception at the instruction that caused it, which is where its handler
/// expects to find it.
pub(crate) fn refuse(vcpu: &mut Vcpu, interrupts: &mut Pending) -> Flow {
    let general_protection = descriptors::Vector::GENERAL_PROTECTION;
    interrupts.raise_exception(vcpu, Event::exception_with_code(general_protection, 0));
    Flow::Resume
}

/// Puts a register's value where `RDMSR` leaves it: the low half in the
/// accumulator, the high half in the data register.
fn answer(vcpu: &mut Vcpu, value: u64) {
    vcpu.save_mut().rax = value & u64::from(u32::MAX);
    vcpu.registers_mut().rdx = value >> u32::BITS;
}

/// The value a `WRMSR` is writing, assembled from the same two halves.
fn written(vcpu: &Vcpu) -> u64 {
    (vcpu.registers().rdx << u32::BITS) | (vcpu.save().rax & u64::from(u32::MAX))
}

const _: () = assert!(
    DISABLED.into_bits() & VmCr::LOCKED == VmCr::LOCKED,
    "the answer given for the register must have both bits the lock protects set",
);
const _: () = assert!(
    EFER_RESERVED & (SVME | LME | LMA) == 0,
    "none of the three bits handled by name may be one a write is refused for",
);

#[cfg(test)]
mod tests {
    use super::{offset_adjustment, offset_for_timestamp};

    #[test]
    fn timestamp_offsets_use_architectural_wrapping_arithmetic() {
        assert_eq!(offset_for_timestamp(0, 1), u64::MAX);
        assert_eq!(offset_for_timestamp(7, u64::MAX), 8);
    }

    #[test]
    fn offset_adjustments_round_trip_across_wraparound() {
        for (old, new) in [(0, 1), (1, 0), (u64::MAX - 3, 5)] {
            assert_eq!(old.wrapping_add(offset_adjustment(old, new)), new);
        }
    }
}
