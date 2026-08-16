//! The face a guest reaches its controller through in x2APIC mode.
//!
//! The same registers as the memory-mapped page, reached as model-specific
//! registers instead, with the index derived from the offset. What differs is
//! entirely in how software is told it got something wrong: through the page
//! nothing faults, and here almost everything does.
//!
//! The architecture's phrasing for the reserved bits is `RsvdZ` — writing a
//! non-zero value to one is a general protection fault, and reading one gives
//! zero. That applies to the upper half of every register except the interrupt
//! command, which is the one register that is genuinely 64 bits wide.
//!
//! # These are entry points, not a device
//!
//! There is no page to trap, so nothing calls into here through [`emulate`]. A
//! guest in x2APIC mode reaches its controller with `RDMSR` and `WRMSR`, which
//! are intercepted through the permission map, and the exit handler calls
//! [`read_msr`] and [`write_msr`] directly.

use apic::{IA32_TSC_DEADLINE, X2APIC_BASE_MSR};
use log::{trace, warn};

use crate::{
    VlapicError,
    face::{
        dispatch::{self, Written, acted},
        table::{Access, Register, X2APIC_LAST_MSR},
    },
    hardware::{model::Model, timer},
    machine::current,
    registers::{
        SPURIOUS_WRITABLE, TASK_PRIORITY_MASK, TIMER_DIVIDE_MASK, Vlapic,
        base::{ApicBase, BaseFault, Mode},
        icr::Command,
        lvt::{Entry, TimerMode},
    },
};

/// What the guest reads from one of the controller's model-specific registers.
///
/// `tsc_offset` is the offset applied to this processor's guest timestamp and
/// is used to translate timestamp-counter deadline readback into that domain.
///
/// # Errors
///
/// [`VlapicError::NotInstalled`] before [`crate::install`],
/// [`VlapicError::NoLapic`] on a processor with no controller, or
/// [`VlapicError::Fault`] if the guest should take a general protection fault
/// for the access.
pub fn read_msr(index: u32, tsc_offset: u64) -> Result<u64, VlapicError> {
    let vlapic = current()?;
    let value = read(vlapic, index, tsc_offset).map_err(|fault| {
        warn!("vlapic: refusing a read of {index:#x}: {fault:?}");
        VlapicError::Fault
    })?;
    trace!(
        "vlapic: {} read {index:#05x} to its {value:#018x} model-specific register",
        vlapic.index()
    );
    Ok(value)
}

/// What a write to one of the controller's model-specific registers does.
///
/// `tsc_offset` is the offset applied to this processor's guest timestamp and
/// is used to translate a timestamp-counter deadline onto physical hardware.
///
/// # Errors
///
/// As [`read_msr`].
pub fn write_msr(index: u32, value: u64, tsc_offset: u64) -> Result<(), VlapicError> {
    let vlapic = current()?;
    let written = write(vlapic, index, value, tsc_offset).map_err(|fault| {
        warn!("vlapic: refusing a write of {value:#x} to {index:#x}: {fault:?}");
        VlapicError::Fault
    })?;
    trace!(
        "vlapic: {} wrote {value:#018x} to its {index:#05x} model-specific register",
        vlapic.index()
    );
    acted(vlapic, written);
    Ok(())
}

/// Every model-specific register the guest's controller answers for.
///
/// Handed to whatever programs the permission map. The whole of the range the
/// architecture reserves for the controller is named, not merely the indices
/// that hold a register: reaching an unassigned one is a fault the guest is
/// entitled to, and it cannot be given one by code that never sees the access.
pub fn intercepted() -> impl Iterator<Item = u32> {
    (X2APIC_BASE_MSR..=X2APIC_LAST_MSR).chain([ApicBase::MSR, IA32_TSC_DEADLINE])
}

/// Whether an index is one this crate answers for.
///
/// The whole of the controller's reserved range counts, not merely the indices
/// that name a register: reaching an unassigned one is a fault the guest has to
/// be given, and it cannot be given one by code that never sees the access.
///
/// The same set as [`intercepted`], and the two answers have to be the same
/// set: an index intercepted and not claimed stops the guest, because no
/// handler owns it, and one claimed and not intercepted is executed against the
/// machine's own registers.
#[must_use]
pub const fn claims(index: u32) -> bool {
    matches!(index, X2APIC_BASE_MSR..=X2APIC_LAST_MSR)
        || index == ApicBase::MSR
        || index == IA32_TSC_DEADLINE
}

/// What the guest reads.
///
/// # Errors
///
/// [`Fault`] if the index names nothing, names a register that does not exist
/// in the mode the guest is in, or names one that may not be read — each of
/// which the guest takes as a general protection fault.
pub(crate) fn read(vlapic: &Vlapic, index: u32, tsc_offset: u64) -> Result<u64, Fault> {
    if index == ApicBase::MSR {
        return Ok(vlapic.base().bits());
    }
    if index == IA32_TSC_DEADLINE {
        // Absent unless the processor reports it, and `CPUID` is passed through
        // — so a guest told the feature does not exist finds the register does
        // not exist either.
        if !vlapic.model().deadline() {
            return Err(Fault::NoSuchRegister);
        }
        return Ok(guest_deadline(timer::deadline(vlapic), tsc_offset));
    }
    let (register, access) = addressable(vlapic.mode(), vlapic.model(), index)?;
    if !matches!(access, Access::ReadOnly | Access::ReadWrite) {
        return Err(Fault::WriteOnly);
    }
    // The one register that is really 64 bits wide, and the one whose stored
    // value is not narrowed to what a guest may write before it gets there: a
    // controller is seeded with the value firmware left in the real register,
    // whole. So the read is masked with the same set the write is judged
    // against. Every other read has a zero upper half, which is what `RsvdZ`
    // requires — and this is the rest of that rule: a guest handed a bit the
    // register does not have would be faulted for writing back what it read.
    if register == Register::COMMAND_LOW {
        return Ok(vlapic.command().bits() & Command::WRITABLE_X2APIC);
    }
    Ok(u64::from(dispatch::read(vlapic, register)))
}

/// What a write does.
///
/// # Errors
///
/// As [`read`], and additionally [`Fault::Reserved`] if a bit the register
/// reserves was written non-zero — which through this face is a fault rather
/// than something quietly dropped.
pub(crate) fn write(
    vlapic: &Vlapic,
    index: u32,
    value: u64,
    tsc_offset: u64,
) -> Result<Written, Fault> {
    if index == ApicBase::MSR {
        return vlapic
            .write_base(value)
            .map(Written::ModeChanged)
            .map_err(Fault::Base);
    }
    if index == IA32_TSC_DEADLINE {
        if !vlapic.model().deadline() {
            return Err(Fault::NoSuchRegister);
        }
        // Hardware ignores this write outside deadline mode rather than
        // faulting, so a refusal here is not an error — but it has to be a
        // refusal and nothing else. Reporting a timer action for an ignored
        // write is what would let a `WRMSR` the architecture discards go on to
        // re-arm a timer.
        if vlapic.timer_mode() != Some(TimerMode::Deadline) {
            return Ok(Written::Nothing);
        }
        return Ok(Written::TimerDeadline(physical_deadline(value, tsc_offset)));
    }
    let model = vlapic.model();
    let (register, access) = addressable(vlapic.mode(), model, index)?;
    if !matches!(access, Access::ReadWrite | Access::WriteOnly) {
        return Err(Fault::ReadOnly);
    }
    // The one register that is genuinely 64 bits wide, and the one whose
    // reserved fields have to be judged before anything is stored: a write that
    // must raise a fault must not first have sent an interrupt, reset a
    // processor, or changed what a guest reads back.
    if register == Register::COMMAND_LOW {
        if value & !Command::WRITABLE_X2APIC != 0 {
            return Err(Fault::Reserved);
        }
        return Ok(Written::Command(vlapic.set_command(value)));
    }
    // Everything else is a 32-bit register whose upper half is reserved.
    if value > u64::from(u32::MAX) {
        return Err(Fault::Reserved);
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the upper half was just established to be zero"
    )]
    let narrow = value as u32;
    // Reserved bits within the low half fault here rather than being dropped,
    // which is the whole difference between the two faces: through the page a
    // stray bit is quietly discarded, and through a model-specific register it
    // is a general protection fault. Judged before the write, for the same
    // reason the command register is.
    if narrow & reserved(model, register) != 0 {
        return Err(Fault::Reserved);
    }
    Ok(dispatch::write(vlapic, register, narrow))
}

/// Translates a physical deadline into the timestamp domain the guest reads.
///
/// Zero is not a timestamp: it is the architectural spelling of a disarmed
/// timer and must remain zero under every offset.
const fn guest_deadline(physical: u64, offset: u64) -> u64 {
    if physical == 0 {
        0
    } else {
        physical.wrapping_add(offset)
    }
}

/// Translates a guest deadline into the timestamp domain physical hardware
/// compares against.
const fn physical_deadline(guest: u64, offset: u64) -> u64 {
    if guest == 0 {
        0
    } else {
        let physical = guest.wrapping_sub(offset);
        if physical == 0 { 1 } else { physical }
    }
}

/// Which bits of a register's low half a guest in x2APIC may not put a one in.
///
/// `RsvdZ`: a non-zero write to one of these is a general protection fault
/// rather than something quietly dropped, which is the whole of what this face
/// does that the page does not. So naming the reserved set per register is this
/// face's own work, and every mask here is the complement of the one the
/// register file stores through — a bit has to fault exactly when it would
/// otherwise be discarded, or a guest either reads back a bit it was allowed to
/// write or is refused one it was entitled to.
///
/// The local vector table entries are the one place the two are not
/// complements, and [`Entry::reserved`] is where that is stated: two of their
/// bits are the controller's own reports, which are dropped from a write and
/// not faulted on.
///
/// Pure in the model rather than taking a controller, so that the whole table
/// can be asserted without one.
fn reserved(model: Model, register: Register) -> u32 {
    match register {
        // Acknowledging is not a value and neither is re-arming the error
        // register. The architecture spells both as a write of zero, so every
        // bit of them is reserved.
        Register::END_OF_INTERRUPT | Register::ERROR_STATUS => u32::MAX,
        // Only the priority byte; the rest of the register is reserved.
        Register::TASK_PRIORITY => !TASK_PRIORITY_MASK,
        // The vector and the bit that software-enables the controller.
        Register::SPURIOUS => !SPURIOUS_WRITABLE,
        // A vector, and nothing else: the delivery mode is fixed, the
        // destination is this processor, and there is no shorthand to name.
        Register::SELF_IPI => !VECTOR,
        // Three bits that are not adjacent — the middle one is reserved.
        Register::TIMER_DIVIDE => !TIMER_DIVIDE_MASK,
        // The one register every bit of which is the guest's.
        Register::TIMER_INITIAL_COUNT => 0,
        // Every register this face can write and has not named above is a local
        // vector table entry, so anything else reaching here is a mis-decode.
        // Reserving all of it is the only safe answer to a register whose
        // reserved bits are unknown: the alternative direction lets a guest put
        // arbitrary bits into a register in the one function whose whole job is
        // to enumerate the bits it may not.
        other => Entry::of(other).map_or(u32::MAX, |entry| entry.reserved(model)),
    }
}

/// The register an index names and what may be done with it, if the guest may
/// name it at all.
///
/// Takes the mode and the model rather than reading them from the controller,
/// so that authorising an access and performing it are one decision made from
/// one value. Only the processor a controller belongs to writes its mode, and
/// it is the processor executing this, so the two loads could not disagree
/// today — but a guest authorised as an x2APIC controller and then answered as
/// the older one is the failure that would follow if one ever could, and one
/// load cannot have it.
///
/// # Errors
///
/// [`Fault::NotX2Apic`] outside x2APIC, or [`Fault::NoSuchRegister`] if the
/// index names nothing or names a register this controller does not have.
fn addressable(mode: Mode, model: Model, index: u32) -> Result<(Register, Access), Fault> {
    // Reaching the controller's registers at all is a fault outside x2APIC:
    // the indices are reserved until the guest has enabled the mode that
    // assigns them.
    if mode != Mode::X2Apic {
        return Err(Fault::NotX2Apic);
    }
    let register = Register::from_msr(index).ok_or(Fault::NoSuchRegister)?;
    match Access::of(register, mode, model) {
        Access::Absent => Err(Fault::NoSuchRegister),
        access => Ok((register, access)),
    }
}

/// A vector is the low eight bits of whatever register carries one, and in the
/// self-interrupt register it is the only field there is.
const VECTOR: u32 = 0xFF;

/// Why an access through this face is a general protection fault.
///
/// Every variant is the same exception as far as the guest is concerned. They
/// are kept apart because which one happened is worth logging: a guest reaching
/// the controller's registers in the wrong mode is doing something quite
/// different from one writing a reserved bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fault {
    /// The index names no register the architecture assigns.
    NoSuchRegister,
    /// The controller is not in x2APIC mode, so its registers have no indices.
    NotX2Apic,
    /// The register may not be written.
    ReadOnly,
    /// The register may not be read.
    WriteOnly,
    /// A bit the register reserves was written non-zero.
    Reserved,
    /// The base register refused the write.
    Base(BaseFault),
}

#[cfg(test)]
mod tests {
    //! A controller cannot be built on a host — one is made from a roster
    //! entry, and a roster comes from firmware's tables — so what is
    //! asserted here is the whole of what this face decides without one:
    //! which index names which register, which accesses are a general
    //! protection fault, and which bits of each register a write may put a
    //! one in.
    //!
    //! The negative cases are the ones that matter. A fault this face raises
    //! where hardware does not is a guest killed for doing something correct,
    //! and both of the defects this suite was written for were of that
    //! shape.

    use apic::{IA32_TSC_DEADLINE, X2APIC_BASE_MSR};

    use super::{
        Access, ApicBase, Command, Fault, addressable, claims, guest_deadline, intercepted,
        physical_deadline, reserved,
    };
    use crate::{
        hardware::model,
        registers::{base::Mode, lvt::Entry},
    };

    /// Bit 12 of a local vector table entry: the controller's own report that a
    /// delivery from the source has not yet reached the processor.
    const DELIVERY_STATUS: u32 = 1 << 12;

    /// Bit 14 of one: the controller's own report of an accepted,
    /// unacknowledged level-triggered interrupt from the pin.
    const REMOTE_IRR: u32 = 1 << 14;

    #[test]
    fn every_index_the_permission_map_traps_is_one_this_crate_answers_for() {
        // Both directions of a disagreement are fatal, and in different ways: an
        // index trapped and unclaimed reaches an exit handler that owns nothing
        // and stops the guest for good, and one claimed and untrapped is executed
        // by the guest against the machine's own controller.
        for index in intercepted() {
            assert!(claims(index), "{index:#x} is trapped and unclaimed");
        }
    }

    #[test]
    fn the_claimed_set_is_the_whole_of_what_the_architecture_dedicates() {
        // Interception is default-allow, so an index outside this set is one a
        // guest executes with no exit at all. The range is four times as long as
        // the registers reach, and every index in it belongs to the controller.
        assert!(claims(0x800) && claims(0x8FF) && claims(0x900) && claims(0xBFF));
        assert!(!claims(0x7FF) && !claims(0xC00));
        assert!(claims(ApicBase::MSR) && claims(IA32_TSC_DEADLINE));
        for index in [ApicBase::MSR - 1, ApicBase::MSR + 1, IA32_TSC_DEADLINE - 1] {
            assert!(!claims(index), "{index:#x}");
        }
    }

    #[test]
    fn the_controllers_registers_have_no_indices_outside_x2apic() {
        // Every one of them, including the base register's own block: until the
        // guest has enabled the mode that assigns these indices, they are
        // reserved and reaching one is a fault.
        for mode in [Mode::XApic, Mode::Disabled] {
            for index in [X2APIC_BASE_MSR, 0x802, 0x830, 0x83F, 0xBFF] {
                assert_eq!(
                    addressable(mode, model::tests::AMD, index),
                    Err(Fault::NotX2Apic),
                    "{index:#x} in {mode}"
                );
            }
        }
    }

    #[test]
    fn an_index_that_names_no_register_is_a_fault() {
        for index in [
            // Reserved inside the block the registers sit in.
            0x800, 0x801, 0x804, 0x805, 0x806, 0x807, 0x829, 0x82E, 0x83A, 0x83D,
            // Past the registers but inside the range the architecture dedicates
            // to the controller, which is where three quarters of it lies.
            0x840, 0x8FF, 0x900, 0xBFF,
        ] {
            assert_eq!(
                addressable(Mode::X2Apic, model::tests::AMD, index),
                Err(Fault::NoSuchRegister),
                "{index:#x}"
            );
        }
    }

    #[test]
    fn the_registers_the_wide_face_removed_are_a_fault_to_name() {
        // Arbitration priority, remote read, the destination format register and
        // the high half of the interrupt command. Their indices are reserved
        // here, so naming one is the same fault as naming an unassigned index —
        // and in particular not an illegal-register error, which this face never
        // records.
        for index in [0x809, 0x80C, 0x80E, 0x831] {
            assert_eq!(
                addressable(Mode::X2Apic, model::tests::AMD, index),
                Err(Fault::NoSuchRegister),
                "{index:#x}"
            );
        }
    }

    #[test]
    fn an_entry_the_controller_does_not_have_is_a_fault_to_name() {
        // The three optional local vector table entries, on a controller
        // reporting the fewest the architecture describes.
        for index in [0x833, 0x834, 0x82F] {
            assert_eq!(
                addressable(Mode::X2Apic, model::tests::SPARSE, index),
                Err(Fault::NoSuchRegister),
                "{index:#x}"
            );
            assert!(addressable(Mode::X2Apic, model::tests::AMD, index).is_ok());
        }
    }

    #[test]
    fn which_registers_may_be_read_and_which_written() {
        // The direction of every register this face has. What is not here is
        // absent, and is covered above.
        for (index, access) in [
            (0x802, Access::ReadOnly),
            (0x803, Access::ReadOnly),
            (0x808, Access::ReadWrite),
            (0x80A, Access::ReadOnly),
            (0x80B, Access::WriteOnly),
            (0x80D, Access::ReadOnly),
            (0x80F, Access::ReadWrite),
            (0x810, Access::ReadOnly),
            (0x818, Access::ReadOnly),
            (0x820, Access::ReadOnly),
            (0x828, Access::ReadWrite),
            (0x830, Access::ReadWrite),
            (0x832, Access::ReadWrite),
            (0x838, Access::ReadWrite),
            (0x839, Access::ReadOnly),
            (0x83E, Access::ReadWrite),
            (0x83F, Access::WriteOnly),
        ] {
            let (_, answered) = addressable(Mode::X2Apic, model::tests::AMD, index)
                .expect("every index here names a register this controller has");
            assert_eq!(answered, access, "{index:#x}");
        }
    }

    #[test]
    fn the_reserved_bits_of_every_register_this_face_may_write() {
        // Written out as literals so that a mask moving fails a test rather than
        // moving with it, and stated as what may *not* be written because that is
        // what the architecture spells `RsvdZ` and what a fault is raised for.
        for (index, reserved_bits) in [
            // Acknowledging and re-arming the error register are both a write of
            // zero and nothing else.
            (0x80B, u32::MAX),
            (0x828, u32::MAX),
            // The priority byte, the spurious vector with its enable bit, and
            // the divide's three non-adjacent bits.
            (0x808, 0xFFFF_FF00),
            (0x80F, 0xFFFF_FE00),
            (0x83E, 0xFFFF_FFF4),
            // The count, every bit of which is the guest's.
            (0x838, 0),
            // A vector, and nothing else.
            (0x83F, 0xFFFF_FF00),
            // The entries, whose per-entry shape is the local vector table's own
            // and is asserted there.
            (0x832, 0xFFF8_EF00),
            (0x835, 0xFFFE_0800),
            (0x837, 0xFFFE_E800),
        ] {
            let (register, _) = addressable(Mode::X2Apic, model::tests::AMD, index)
                .expect("every index here names a register this controller has");
            assert_eq!(
                reserved(model::tests::AMD, register),
                reserved_bits,
                "{index:#x}"
            );
        }
    }

    #[test]
    fn a_guest_may_write_back_the_entry_it_read() {
        // Reading an entry, changing one field and writing the whole of it back
        // is how software touches these registers — every operating system's
        // controller shutdown does exactly that to mask them. What it read has
        // whatever the controller had put in its two status bits, so faulting on
        // those would be a general protection fault for writing back a value the
        // controller itself supplied.
        for entry in Entry::ALL {
            let register = entry.register();
            assert_eq!(
                reserved(model::tests::AMD, register) & DELIVERY_STATUS,
                0,
                "{entry:?}"
            );
            // Remote IRR exists in the two entries that describe a wire, so
            // unlike delivery status it is genuinely reserved in the rest.
            assert_eq!(
                reserved(model::tests::AMD, register) & REMOTE_IRR == 0,
                entry.is_pin(),
                "{entry:?}"
            );
        }
    }

    #[test]
    fn the_wide_interrupt_command_accepts_what_brings_a_processor_up() {
        // The reserved-bit test a write of the interrupt command is judged
        // against, over the quadwords an operating system and this machine's
        // firmware actually write: INIT asserted, INIT de-asserted, and a
        // start-up with and without the level bit firmware sets. A fault on any
        // of these is a guest whose second processor never starts.
        for low in [0xC500_u64, 0x8500, 0x0608, 0x4608, 0x000C_4500] {
            let bits = (5 << 32) | low;
            assert_eq!(bits & !Command::WRITABLE_X2APIC, 0, "{bits:#018x}");
        }
        // And what it still refuses: the delivery-status bit this vendor requires
        // to be written as zero, bit 13, bits 17:16 and bits 31:20.
        for reserved_bit in [12, 13, 16, 17, 20, 31] {
            assert_ne!(
                (1_u64 << reserved_bit) & !Command::WRITABLE_X2APIC,
                0,
                "bit {reserved_bit}"
            );
        }
    }

    #[test]
    fn deadline_translation_preserves_disarmed_timers() {
        assert_eq!(physical_deadline(0, 0x1234), 0);
        assert_eq!(guest_deadline(0, 0x1234), 0);
    }

    #[test]
    fn deadline_translation_round_trips_with_wrapping_offsets() {
        for (deadline, offset) in [
            (1, 0),
            (0x1234_5678_9ABC_DEF0, 0x1111_2222_3333_4444),
            (u64::MAX, u64::MAX - 7),
        ] {
            assert_eq!(
                guest_deadline(physical_deadline(deadline, offset), offset),
                deadline
            );
        }
    }

    #[test]
    fn an_armed_guest_deadline_never_becomes_the_physical_disarm_value() {
        assert_eq!(physical_deadline(0x1234, 0x1234), 1);
    }
}
