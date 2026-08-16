//! The local APIC's registers, and the two ways a processor reaches them.
//!
//! There are two interfaces to the same controller. The older one puts its
//! registers in a 4 KiB page of memory-mapped space; x2APIC puts them in
//! model-specific registers instead, which is faster, needs no mapping, and is
//! the only way to address a processor whose identifier does not fit eight
//! bits.
//!
//! Nothing above this module should have to know which is in use, and nothing
//! here should have to say where a register is twice. It does not have to: the
//! model-specific register index is *derived* from the memory-mapped offset,
//! `0x800 + offset / 16`, which is how the architecture defined it. So there is
//! one list of registers, written once, and [`Access`] decides how it is
//! reached.
//!
//! # Where the two genuinely differ
//!
//! Four places, and the type system keeps three of them straight rather than a
//! convention doing it.
//!
//! The interrupt command register is two 32-bit registers in the older
//! interface and one 64-bit model-specific register in x2APIC, so it is offered
//! as [`Access::command`] and [`Access::send`] rather than as an offset, and no
//! caller assembles it twice. The two also need opposite things of their
//! caller: the older one is a posted write whose delivery status has to be
//! polled before the next command and whose two halves must not be interleaved
//! with anybody else's, while an x2APIC write is a single model-specific
//! register that the architecture deliberately exempts from `WRMSR`'s ordering
//! against earlier stores.
//!
//! The upper half of that command register and the destination format register
//! exist only in the older interface, and the difference is not that x2APIC
//! leaves them unused: the model-specific registers they would occupy are
//! absent, and naming one is a general protection fault rather than a zero. So
//! those two are [`MappedRegister`]s rather than [`Register`]s, reachable only
//! through the [`Page`] that *is* the older interface — which means only code
//! that has already established which interface is in use can name them.
//!
//! The fourth is x2APIC's register for sending a processor an interrupt to
//! itself. Nothing here uses it: every command this crate sends is addressed to
//! another processor, and one addressed to this processor goes through the same
//! command register as the rest.
//!
//! # What a [`Register`] is allowed to be
//!
//! Every one of them is a register software both reads and writes. The two that
//! are not are kept out of the list rather than trusted to a comment: the
//! acknowledgement register, which x2APIC faults a read of, is reachable only
//! as [`Access::acknowledge`], and the eight-register banks are reachable only
//! through [`bank`] and [`word`], so the offset arithmetic that could leave the
//! page cannot be written anywhere else.

use core::{
    ptr,
    sync::atomic::{Ordering, fence},
};

use x86_64::{VirtAddr, instructions::interrupts, registers::model_specific::Msr};

use crate::ApicError;

/// Something with an offset in the memory-mapped register page.
///
/// Implemented by both kinds of register, so that the page is read and written
/// through one pair of methods while [`Access`] — which may not be the page at
/// all — takes only the registers both interfaces have.
pub(crate) trait Offset: Copy {
    /// The byte offset into the page.
    fn offset(self) -> u64;
}

/// One of the local APIC's registers, named by its offset in the memory-mapped
/// page.
///
/// The offset is the canonical name even when x2APIC is in use, because it is
/// the one the model-specific register index is computed from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Register(u32);

impl Register {
    /// This processor's identifier.
    pub(crate) const ID: Self = Self(0x20);
    /// The controller's version, and how many local vector table entries it
    /// has.
    pub(crate) const VERSION: Self = Self(0x30);
    /// Task priority: which interrupt priorities this processor will accept.
    pub(crate) const TASK_PRIORITY: Self = Self(0x80);
    /// The priority this processor is actually servicing at, which is its task
    /// priority or the highest interrupt in service, whichever is higher.
    pub(crate) const PROCESSOR_PRIORITY: Self = Self(0xA0);
    /// Which logical destinations this processor answers to.
    pub(crate) const LOGICAL_DESTINATION: Self = Self(0xD0);
    /// Spurious interrupt vector, and the bit that software-enables the
    /// controller.
    pub(crate) const SPURIOUS: Self = Self(0xF0);
    /// First of the eight registers saying which vectors this processor has
    /// accepted and not yet acknowledged.
    pub(crate) const IN_SERVICE: Self = Self(0x100);
    /// First of the eight registers saying which of the vectors in service
    /// arrived level triggered.
    pub(crate) const TRIGGER_MODE: Self = Self(0x180);
    /// First of the eight registers saying which vectors have been delivered to
    /// this processor and not yet accepted.
    pub(crate) const INTERRUPT_REQUEST: Self = Self(0x200);
    /// Errors the controller noticed, latched until written.
    pub(crate) const ERROR_STATUS: Self = Self(0x280);
    /// Local vector table entry for corrected machine-check errors.
    pub(crate) const LVT_CORRECTED_MACHINE_CHECK: Self = Self(0x2F0);
    /// The low half of the interrupt command register, and the whole of it
    /// under x2APIC.
    pub(crate) const COMMAND_LOW: Self = Self(0x300);
    /// Local vector table entry for the controller's own timer.
    pub(crate) const LVT_TIMER: Self = Self(0x320);
    /// Local vector table entry for the thermal sensor.
    pub(crate) const LVT_THERMAL: Self = Self(0x330);
    /// Local vector table entry for the performance counters.
    pub(crate) const LVT_PERFORMANCE: Self = Self(0x340);
    /// Local vector table entry for the first local interrupt pin.
    pub(crate) const LVT_LINT0: Self = Self(0x350);
    /// Local vector table entry for the second local interrupt pin.
    pub(crate) const LVT_LINT1: Self = Self(0x360);
    /// Local vector table entry for the controller's own errors.
    pub(crate) const LVT_ERROR: Self = Self(0x370);
    /// What the timer counts down from.
    pub(crate) const TIMER_INITIAL_COUNT: Self = Self(0x380);
    /// What the timer has left.
    pub(crate) const TIMER_CURRENT_COUNT: Self = Self(0x390);
    /// How far the bus clock is divided before the timer counts it.
    pub(crate) const TIMER_DIVIDE: Self = Self(0x3E0);

    /// Written to acknowledge the interrupt currently being serviced.
    ///
    /// Private, and the reason is that it is the one register software may not
    /// read: x2APIC faults the attempt. [`Access::acknowledge`] is the whole of
    /// what anything needs of it.
    const END_OF_INTERRUPT: Self = Self(0xB0);

    /// The register `slots` slots past this one.
    ///
    /// Private, because unchecked offset arithmetic is how a register leaves
    /// the page it was promised to be in. [`bank`] and [`word`] are the
    /// only two things that need it, and both are bounded by
    /// [`VECTOR_SLOTS`].
    const fn offset_by(self, slots: u32) -> Self {
        Self(self.0 + slots * REGISTER_STRIDE)
    }

    /// The model-specific register x2APIC puts this register in.
    const fn msr(self) -> u32 {
        X2APIC_BASE_MSR + self.0 / REGISTER_STRIDE
    }
}

impl Offset for Register {
    fn offset(self) -> u64 {
        u64::from(self.0)
    }
}

/// One of the two registers only the memory-mapped interface has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MappedRegister(u32);

impl MappedRegister {
    /// The high half of the interrupt command register: the destination.
    /// x2APIC keeps the destination in the upper word of its single wide
    /// register instead.
    pub(crate) const COMMAND_HIGH: Self = Self(0x310);
    /// How a logical destination is matched.
    pub(crate) const DESTINATION_FORMAT: Self = Self(0xE0);
}

impl Offset for MappedRegister {
    fn offset(self) -> u64 {
        u64::from(self.0)
    }
}

/// Every local vector table entry, in the order the version register counts
/// them.
///
/// That order is the architecture's and is not the order the registers sit in.
/// A controller says how many entries it has, and it has exactly the first that
/// many of these — so one reporting four has the timer, the two pins and its
/// own error entry, and no performance, thermal or machine-check entry at all.
/// Touching one it does not have is undefined through the page and a general
/// protection fault through the model-specific registers, which is why nothing
/// here reaches for an entry without asking first.
const LVT_ENTRIES: [Register; 7] = [
    Register::LVT_TIMER,
    Register::LVT_LINT0,
    Register::LVT_LINT1,
    Register::LVT_ERROR,
    Register::LVT_PERFORMANCE,
    Register::LVT_THERMAL,
    Register::LVT_CORRECTED_MACHINE_CHECK,
];

/// How many of the controller's registers it takes to describe every vector one
/// bit at a time: two hundred and fifty-six vectors, thirty-two to a register.
pub(crate) const VECTOR_SLOTS: u32 = 8;

/// How many local vector table entries the architecture defines, and so how far
/// a controller's count is believed.
pub(crate) const DEFINED_LVT_ENTRIES: usize = LVT_ENTRIES.len();

/// The eight consecutive registers a bank of one bit per vector is spread
/// across, lowest vectors first.
///
/// Reversible, because the one question asked of a bank in reverse is which of
/// the vectors in it has the highest priority.
pub(crate) fn bank(
    first: Register,
) -> impl DoubleEndedIterator<Item = Register> + ExactSizeIterator {
    (0..VECTOR_SLOTS).map(move |slot| first.offset_by(slot))
}

/// Which register of a bank holds a vector's bit, and which bit of it.
pub(crate) fn word(first: Register, vector: u8) -> (Register, u32) {
    let number = u32::from(vector);
    (
        first.offset_by(number / u32::BITS),
        1 << (number % u32::BITS),
    )
}

/// Every local vector table entry a controller counting `entries` of them has,
/// in the order it counts them.
pub(crate) fn lvt_present(entries: u32) -> impl Iterator<Item = Register> {
    LVT_ENTRIES
        .into_iter()
        .zip(0..)
        .take_while(move |(_, index)| *index < entries)
        .map(|(register, _)| register)
}

/// Whether a controller counting `entries` local vector table entries has this
/// one.
pub(crate) fn has_lvt(register: Register, entries: u32) -> bool {
    lvt_present(entries).any(|candidate| candidate == register)
}

/// How many local vector table entries the version register reports.
///
/// The field holds one less than the count, so every controller has at least
/// one and the arithmetic cannot wrap. A controller reporting more than the
/// architecture defines is not trusted beyond that: [`lvt_present`] stops at
/// the entries this crate knows the offsets of.
///
/// Public together with [`version_number`] because both halves of that one
/// register describe the controller to whoever is presenting it — a hypervisor
/// handing a guest a local controller has to report both, and reading the
/// register twice to get them is two answers where the architecture gives one.
#[must_use]
pub const fn lvt_entries(version: u32) -> u32 {
    ((version >> LVT_COUNT_SHIFT) & VERSION_FIELD) + 1
}

/// The controller's version, out of the register that also carries the entry
/// count.
///
/// Public for the reason [`lvt_entries`] is.
#[must_use]
pub const fn version_number(version: u32) -> u32 {
    version & VERSION_FIELD
}

/// Index of the model-specific register the register at offset zero maps to.
///
/// Public because the emulated controller derives the same indices from the
/// same offsets, and the derivation is the architecture's rather than either
/// crate's: two copies of it would be two answers to the question of which
/// register a guest's `RDMSR` names.
pub const X2APIC_BASE_MSR: u32 = 0x800;

/// Bytes between one memory-mapped register and the next. Each is 32 bits wide
/// and each gets a 16-byte slot, which is why dividing by it turns an offset
/// into a model-specific register index.
///
/// Public for the reason [`X2APIC_BASE_MSR`] is.
pub const REGISTER_STRIDE: u32 = 16;

/// The single model-specific register x2APIC gives the interrupt command, in
/// place of the two the older interface splits it across.
const X2APIC_COMMAND_MSR: u32 = 0x830;

/// The version register's fields are one byte each.
const VERSION_FIELD: u32 = 0xFF;

/// Bits the version register's local-vector-table count is shifted by.
const LVT_COUNT_SHIFT: u32 = 16;

/// The bit the older interface sets in the command register while a command is
/// still being sent. x2APIC has no such bit: its write does not return until
/// the command has been accepted.
const DELIVERY_PENDING: u32 = 1 << 12;

/// The page of memory-mapped registers, at the address it was mapped to.
///
/// Holding one of these says the older interface is the one in use, which is
/// what makes it the only way to reach a [`MappedRegister`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Page(VirtAddr);

impl Page {
    /// The register page, mapped at `base`.
    ///
    /// # Safety
    ///
    /// `base` must be a whole 4 KiB local APIC register page, mapped for as
    /// long as this value is used, and mapped *uncached*. The architecture
    /// requires these registers to be reached strongly uncacheable: through a
    /// write-back mapping a read may be answered out of a cache line, which
    /// says nothing about what the controller currently holds, and a write may
    /// sit in a buffer past the point the machine depends on it having landed.
    pub(crate) const unsafe fn new(base: VirtAddr) -> Self {
        Self(base)
    }

    /// Reads a register.
    pub(crate) fn read(self, register: impl Offset) -> u32 {
        // SAFETY: a whole 4 KiB page is mapped uncached at this address, which
        // is what `new` requires, and every register is an offset inside it that
        // the architecture defines as a readable 32-bit slot — the offsets are
        // this module's own and the only derived ones are bounded by
        // `VECTOR_SLOTS`. Volatile because these are registers and the read is
        // the point.
        unsafe { ptr::read_volatile(self.pointer(register)) }
    }

    /// Writes a register.
    ///
    /// # Safety
    ///
    /// The value must be one the register accepts. What a wrong one does ranges
    /// from an interrupt arriving somewhere nothing expects it to a processor
    /// held in reset.
    pub(crate) unsafe fn write(self, register: impl Offset, value: u32) {
        // SAFETY: as in `read`, and the caller vouches for the value.
        unsafe { ptr::write_volatile(self.pointer(register), value) };
    }

    /// Where a register sits in the page.
    fn pointer(self, register: impl Offset) -> *mut u32 {
        (self.0 + register.offset()).as_mut_ptr::<u32>()
    }
}

/// How the local APIC's registers are reached on this processor, right now.
///
/// Derived at each use rather than remembered. Which interface a controller
/// presents is per processor and changes while the machine runs — a guest
/// entering x2APIC takes its own processor with it — and a remembered answer
/// taken before that would go on reaching a page the architecture has since
/// made unavailable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Access {
    /// Through the memory-mapped page.
    Mapped(Page),
    /// Through model-specific registers.
    Msr,
}

impl Access {
    /// Reads a register.
    pub(crate) fn read(self, register: Register) -> u32 {
        match self {
            Self::Mapped(page) => page.read(register),
            // SAFETY: the register exists because this variant is only chosen
            // once x2APIC is enabled on this processor, every `Register` is one
            // both interfaces have and software may read, and reading any of
            // them has no side effect the architecture does not document as one
            // of its uses.
            Self::Msr => truncate(unsafe { Msr::new(register.msr()).read() }),
        }
    }

    /// Writes a register.
    ///
    /// # Safety
    ///
    /// The value must be one the register accepts. What a wrong one does ranges
    /// from a general protection fault, for a reserved bit in a model-specific
    /// register, to an interrupt arriving somewhere nothing expects it.
    pub(crate) unsafe fn write(self, register: Register, value: u32) {
        match self {
            // SAFETY: the caller vouches for the value.
            Self::Mapped(page) => unsafe { page.write(register, value) },
            // SAFETY: as in `read`, and the caller vouches for the value. The
            // upper half is zero, which every one of these registers requires.
            Self::Msr => unsafe { Msr::new(register.msr()).write(u64::from(value)) },
        }
    }

    /// Acknowledges the interrupt this processor is currently servicing.
    ///
    /// Its own operation rather than a register, because it is the one register
    /// that is write-only: the architecture defines no value it returns and
    /// x2APIC faults a read of it.
    ///
    /// Owed for everything a controller delivered and for nothing else. The
    /// register takes no vector — it retires whichever interrupt in service has
    /// the highest priority — so anything withholding an acknowledgement has to
    /// establish that the one it owes is still that one.
    pub(crate) fn acknowledge(self) {
        // SAFETY: the register takes zero and nothing else, and writing it is
        // what the architecture defines as acknowledging.
        unsafe { self.write(Register::END_OF_INTERRUPT, 0) };
    }

    /// The interrupt command as one value, whichever interface holds it and in
    /// however many pieces.
    ///
    /// Only ever a description of what the register holds. The older interface
    /// splits it across two registers with nothing to read them together, so an
    /// answer is only coherent where nothing can be sending a command — which
    /// is true of firmware capture, before any of this crate's own senders
    /// exist, and is not something this can establish for itself. x2APIC
    /// reads it in one piece, and the architecture describes that read as a
    /// debugging aid rather than as the last value written.
    pub(crate) fn command(self) -> u64 {
        match self {
            Self::Mapped(page) => {
                u64::from(page.read(MappedRegister::COMMAND_HIGH)) << u32::BITS
                    | u64::from(page.read(Register::COMMAND_LOW))
            }
            // SAFETY: the register is architectural on a controller in x2APIC
            // mode, which is the only way this variant is reached, and reading
            // it neither sends a command nor disturbs one.
            Self::Msr => unsafe { Msr::new(X2APIC_COMMAND_MSR).read() },
        }
    }

    /// Sends an interrupt command, and makes everything already written to
    /// memory visible to whoever receives it.
    ///
    /// The two interfaces need opposite things here.
    ///
    /// The older one is three operations that have to be one: the delivery
    /// status of the previous command is polled, the destination is written,
    /// and then the low half is written, which is what sends it.
    /// Interleaving somebody else's destination with this command's is an
    /// interrupt — or an `INIT` — delivered to the wrong processor, so the
    /// whole of it runs with this processor's maskable interrupts held off.
    /// A non-maskable interrupt can still interpose, and a handler that
    /// sent a command from there would break this; nothing in this image
    /// does.
    ///
    /// x2APIC needs no exclusion at all, because the command is one
    /// model-specific register and one write. What it needs instead is a fence:
    /// the architecture deliberately relaxes `WRMSR`'s ordering for the APIC's
    /// own registers, so the command may reach the controller before stores
    /// this processor has already made are visible to the processor
    /// receiving it. The startup sequence depends on exactly that ordering,
    /// and on this architecture the store-store ordering a release fence
    /// would rely on is not enough — only a real barrier is.
    ///
    /// # Errors
    ///
    /// [`ApicError::CommandStuck`] if a previous command through the older
    /// interface is still pending after the wait, which means the controller
    /// has not accepted something the processor gave it and sending another
    /// would overwrite it.
    ///
    /// # Safety
    ///
    /// `command` must be a well-formed interrupt command. It is an instruction
    /// to interrupt a processor, and a malformed one can hold a processor in
    /// reset or deliver to a vector nothing is prepared for.
    pub(crate) unsafe fn send(self, command: u64) -> Result<(), ApicError> {
        match self {
            Self::Mapped(page) => interrupts::without_interrupts(|| {
                self.settle()?;
                // SAFETY: the destination half accepts any value in its top
                // eight bits, and writing it sends nothing on its own.
                unsafe { page.write(MappedRegister::COMMAND_HIGH, truncate(command >> u32::BITS)) };
                // SAFETY: the caller vouches for the command, the previous one
                // has left, and no other sender on this processor can have
                // reached the register since — interrupts are held off across
                // all three operations.
                unsafe { page.write(Register::COMMAND_LOW, truncate(command)) };
                Ok(())
            }),
            Self::Msr => {
                fence(Ordering::SeqCst);
                // SAFETY: the caller vouches for the command. The write does
                // not return until the controller has accepted it, so there is
                // nothing to wait for on either side of it.
                unsafe { Msr::new(X2APIC_COMMAND_MSR).write(command) };
                Ok(())
            }
        }
    }

    /// Waits for the older interface to finish sending whatever it was sending.
    ///
    /// A bounded spin rather than a delay: the wait is normally a handful of
    /// bus cycles, and a controller that has not finished after this many
    /// reads is not going to.
    fn settle(self) -> Result<(), ApicError> {
        for _ in 0..COMMAND_POLLS {
            if self.read(Register::COMMAND_LOW) & DELIVERY_PENDING == 0 {
                return Ok(());
            }
            core::hint::spin_loop();
        }
        Err(ApicError::CommandStuck)
    }
}

/// How many times the command register is read before a still-pending command
/// is called stuck. Each read goes to the controller and back, so this is
/// comfortably longer than any delivery and still a bounded wait on a broken
/// machine.
const COMMAND_POLLS: u32 = 1_000_000;

/// The low half of a 64-bit value.
///
/// Every register here is 32 bits wide, so narrowing is what reading one out of
/// a wider container means rather than a loss to guard against.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the local APIC's registers are 32 bits wide; the upper half is not part of the value"
)]
const fn truncate(value: u64) -> u32 {
    value as u32
}

#[cfg(test)]
mod tests {
    use super::{
        DEFINED_LVT_ENTRIES, LVT_ENTRIES, MappedRegister, Register, VECTOR_SLOTS, bank, has_lvt,
        lvt_entries, lvt_present, version_number, word,
    };

    /// Every register both interfaces have, with the model-specific index the
    /// architecture puts it in.
    const DERIVED: [(Register, u32); 22] = [
        (Register::ID, 0x802),
        (Register::VERSION, 0x803),
        (Register::TASK_PRIORITY, 0x808),
        (Register::PROCESSOR_PRIORITY, 0x80A),
        (Register::END_OF_INTERRUPT, 0x80B),
        (Register::LOGICAL_DESTINATION, 0x80D),
        (Register::SPURIOUS, 0x80F),
        (Register::IN_SERVICE, 0x810),
        (Register::TRIGGER_MODE, 0x818),
        (Register::INTERRUPT_REQUEST, 0x820),
        (Register::ERROR_STATUS, 0x828),
        (Register::LVT_CORRECTED_MACHINE_CHECK, 0x82F),
        (Register::COMMAND_LOW, 0x830),
        (Register::LVT_TIMER, 0x832),
        (Register::LVT_THERMAL, 0x833),
        (Register::LVT_PERFORMANCE, 0x834),
        (Register::LVT_LINT0, 0x835),
        (Register::LVT_LINT1, 0x836),
        (Register::LVT_ERROR, 0x837),
        (Register::TIMER_INITIAL_COUNT, 0x838),
        (Register::TIMER_CURRENT_COUNT, 0x839),
        (Register::TIMER_DIVIDE, 0x83E),
    ];

    #[test]
    fn every_offset_derives_the_architecture_s_own_model_specific_index() {
        for (register, msr) in DERIVED {
            assert_eq!(register.msr(), msr, "{register:?}");
        }
    }

    #[test]
    fn every_register_sits_in_the_page_on_a_sixteen_byte_boundary() {
        for (register, _) in DERIVED {
            assert!(register.0 < 4096, "{register:?}");
            assert_eq!(register.0 % 16, 0, "{register:?}");
        }
        for register in [
            MappedRegister::COMMAND_HIGH,
            MappedRegister::DESTINATION_FORMAT,
        ] {
            assert!(register.0 < 4096);
            assert_eq!(register.0 % 16, 0);
        }
    }

    #[test]
    fn the_two_registers_only_one_interface_has_would_derive_reserved_indices() {
        // Named as the reason they are a separate type: 0x80E and 0x831 are
        // reserved, and reaching either faults.
        assert_eq!(Register(MappedRegister::DESTINATION_FORMAT.0).msr(), 0x80E);
        assert_eq!(Register(MappedRegister::COMMAND_HIGH.0).msr(), 0x831);
    }

    #[test]
    fn a_bank_is_eight_consecutive_registers_and_stays_in_the_page() {
        let mut counted = 0;
        for (register, slot) in bank(Register::IN_SERVICE).zip(0..) {
            assert_eq!(register, Register(0x100 + 0x10 * slot));
            assert!(register.0 < 4096);
            counted += 1;
        }
        assert_eq!(counted, VECTOR_SLOTS);
    }

    #[test]
    fn a_vector_s_bit_is_in_the_slot_thirty_two_of_them_share() {
        assert_eq!(word(Register::IN_SERVICE, 0), (Register(0x100), 1));
        assert_eq!(word(Register::IN_SERVICE, 31), (Register(0x100), 1 << 31));
        assert_eq!(word(Register::IN_SERVICE, 32), (Register(0x110), 1));
        assert_eq!(
            word(Register::TRIGGER_MODE, 255),
            (Register(0x1F0), 1 << 31)
        );
    }

    #[test]
    fn the_version_register_counts_one_less_than_the_entries_it_has() {
        assert_eq!(lvt_entries(0x0000_0010), 1);
        assert_eq!(lvt_entries(0x0003_0010), 4);
        assert_eq!(lvt_entries(0x0006_0015), 7);
        assert_eq!(version_number(0x0006_0015), 0x15);
        assert_eq!(version_number(u32::MAX), 0xFF);
    }

    #[test]
    fn a_controller_has_exactly_the_first_however_many_entries_it_counts() {
        assert_eq!(lvt_present(0).count(), 0);
        assert!(lvt_present(4).eq(LVT_ENTRIES[..4].iter().copied()));
        assert!(has_lvt(Register::LVT_ERROR, 4));
        assert!(!has_lvt(Register::LVT_PERFORMANCE, 4));
        assert!(has_lvt(Register::LVT_CORRECTED_MACHINE_CHECK, 7));
    }

    #[test]
    fn a_controller_claiming_more_entries_than_exist_is_read_no_further() {
        assert_eq!(lvt_entries(0x00FF_0010), 256);
        assert_eq!(
            lvt_present(256).count(),
            DEFINED_LVT_ENTRIES,
            "no offset exists past the entries the architecture defines"
        );
    }
}
