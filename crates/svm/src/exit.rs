//! Why a guest stopped, and what it was doing when it did.
//!
//! Every exit reports a code, and the code decides how to read the two
//! information fields beside it. That second half is the awkward part: the same
//! sixty-four bits mean a port number and an access width after one exit, a
//! page-fault error code after another, and a register number after a third. So
//! the codes are here together with the decoders for those fields, because a
//! decoder applied to the wrong exit is silently plausible nonsense.
//!
//! # Why this is not one large enum
//!
//! Most of the code space is dense ranges rather than distinct reasons: sixteen
//! codes for reading a control register, sixteen for writing one, the same
//! again for debug registers, thirty-two for the exception vectors, and sixteen
//! more for control-register write traps. Spelling those as two hundred
//! variants would put the arithmetic that recovers the register number at every
//! use site.
//!
//! So [`ExitCode`] is the number as the processor reports it, with a name for
//! each code the architecture defines, and [`ExitCode::reason`] turns one into
//! a [`Reason`] whose variants carry the register number or the vector. Callers
//! match on meaning and never write a bare number.
//!
//! # The code that means the hypervisor made a mistake
//!
//! [`ExitCode::INVALID`] is not something the guest did. It says the control
//! block held a combination the processor refuses, and that *no* guest
//! instruction executed. It is what a hypervisor sees while it is still getting
//! the control block right, and it comes with no diagnostic beyond itself,
//! which is worth knowing before spending an afternoon looking for a fault in
//! the guest.

use core::fmt::{self, Debug, Display, Formatter};

use descriptors::Vector;

/// Why a guest stopped.
///
/// The value the processor writes into the control block. Comparisons against
/// the named constants below are the intended way to test one; [`reason`] is
/// the intended way to take one apart.
///
/// [`reason`]: ExitCode::reason
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct ExitCode(u64);

/// First code reporting a read of a control register.
const READ_CR_BASE: u64 = 0x000;
/// First code reporting a write of a control register.
const WRITE_CR_BASE: u64 = 0x010;
/// First code reporting a read of a debug register.
const READ_DR_BASE: u64 = 0x020;
/// First code reporting a write of a debug register.
const WRITE_DR_BASE: u64 = 0x030;
/// First code reporting an intercepted exception.
const EXCEPTION_BASE: u64 = 0x040;
/// First code reporting a control-register write that has already happened.
const WRITE_CR_TRAP_BASE: u64 = 0x090;

/// How many registers each indexed range covers.
const REGISTERS: u64 = 16;
/// How many exception vectors the exception range covers.
const EXCEPTIONS: u64 = 32;

impl ExitCode {
    /// A maskable interrupt arrived while the guest was running.
    pub const INTR: Self = Self(0x060);
    /// A non-maskable interrupt arrived.
    pub const NMI: Self = Self(0x061);
    /// A system management interrupt arrived.
    pub const SMI: Self = Self(0x062);
    /// The signal that resets a processor was sent to the guest.
    pub const INIT: Self = Self(0x063);
    /// An interrupt the hypervisor made pending became deliverable.
    pub const VINTR: Self = Self(0x064);
    /// The guest wrote a bit of control register nought other than the two
    /// concerning floating-point state.
    pub const CR0_SEL_WRITE: Self = Self(0x065);
    /// The guest read the interrupt descriptor table register.
    pub const IDTR_READ: Self = Self(0x066);
    /// The guest read the global descriptor table register.
    pub const GDTR_READ: Self = Self(0x067);
    /// The guest read the local descriptor table register.
    pub const LDTR_READ: Self = Self(0x068);
    /// The guest read the task register.
    pub const TR_READ: Self = Self(0x069);
    /// The guest wrote the interrupt descriptor table register.
    pub const IDTR_WRITE: Self = Self(0x06A);
    /// The guest wrote the global descriptor table register.
    pub const GDTR_WRITE: Self = Self(0x06B);
    /// The guest wrote the local descriptor table register.
    pub const LDTR_WRITE: Self = Self(0x06C);
    /// The guest wrote the task register.
    pub const TR_WRITE: Self = Self(0x06D);
    /// The guest read the timestamp counter.
    pub const RDTSC: Self = Self(0x06E);
    /// The guest read a performance counter.
    pub const RDPMC: Self = Self(0x06F);
    /// The guest pushed its flags.
    pub const PUSHF: Self = Self(0x070);
    /// The guest popped its flags.
    pub const POPF: Self = Self(0x071);
    /// The guest asked what processor it is running on.
    pub const CPUID: Self = Self(0x072);
    /// The guest returned from system management mode.
    pub const RSM: Self = Self(0x073);
    /// The guest returned from an interrupt handler.
    pub const IRET: Self = Self(0x074);
    /// The guest executed the software interrupt instruction. The vector it
    /// named is in the first information field.
    pub const SWINT: Self = Self(0x075);
    /// The guest invalidated its caches without writing them back.
    pub const INVD: Self = Self(0x076);
    /// The guest executed the spin hint often enough to trip the filter.
    pub const PAUSE: Self = Self(0x077);
    /// The guest halted.
    pub const HLT: Self = Self(0x078);
    /// The guest invalidated one of its own translations. The address is in the
    /// first information field.
    pub const INVLPG: Self = Self(0x079);
    /// The guest invalidated a translation in another address space.
    pub const INVLPGA: Self = Self(0x07A);
    /// The guest accessed a port the permission map covers. Decode the first
    /// information field with [`IoAccess`].
    pub const IOIO: Self = Self(0x07B);
    /// The guest accessed a model-specific register the permission map covers.
    /// The first information field says which direction.
    pub const MSR: Self = Self(0x07C);
    /// The guest switched tasks.
    pub const TASK_SWITCH: Self = Self(0x07D);
    /// The guest froze waiting for an external floating-point error signal.
    pub const FERR_FREEZE: Self = Self(0x07E);
    /// The guest triggered the condition that shuts a real processor down.
    pub const SHUTDOWN: Self = Self(0x07F);
    /// The guest tried to run a guest of its own.
    pub const VMRUN: Self = Self(0x080);
    /// The guest called the hypervisor deliberately.
    pub const VMMCALL: Self = Self(0x081);
    /// The guest tried to load processor state from a control block.
    pub const VMLOAD: Self = Self(0x082);
    /// The guest tried to save processor state to a control block.
    pub const VMSAVE: Self = Self(0x083);
    /// The guest tried to set the global interrupt flag.
    pub const STGI: Self = Self(0x084);
    /// The guest tried to clear the global interrupt flag.
    pub const CLGI: Self = Self(0x085);
    /// The guest tried to begin a measured launch.
    pub const SKINIT: Self = Self(0x086);
    /// The guest read the timestamp counter with the processor identifier.
    pub const RDTSCP: Self = Self(0x087);
    /// The guest executed the in-circuit emulator breakpoint.
    pub const ICEBP: Self = Self(0x088);
    /// The guest wrote its caches back and invalidated them.
    pub const WBINVD: Self = Self(0x089);
    /// The guest armed the address monitor.
    pub const MONITOR: Self = Self(0x08A);
    /// The guest waited on the address monitor.
    pub const MWAIT: Self = Self(0x08B);
    /// The guest waited on the address monitor while it was armed.
    pub const MWAIT_ARMED: Self = Self(0x08C);
    /// The guest enabled an extended processor state component.
    pub const XSETBV: Self = Self(0x08D);
    /// The guest read a processor register through the user-mode mechanism.
    pub const RDPRU: Self = Self(0x08E);
    /// The guest wrote its extended feature register, and the write has already
    /// taken effect.
    pub const EFER_WRITE_TRAP: Self = Self(0x08F);
    /// The guest broadcast a translation invalidation.
    pub const INVLPGB: Self = Self(0x0A0);
    /// The guest broadcast an invalidation with operands the architecture does
    /// not allow.
    pub const INVLPGB_ILLEGAL: Self = Self(0x0A1);
    /// The guest invalidated translations by address-space identifier.
    pub const INVPCID: Self = Self(0x0A2);
    /// The guest committed its outstanding memory operations.
    pub const MCOMMIT: Self = Self(0x0A3);
    /// The guest waited for its broadcast invalidations to finish.
    pub const TLBSYNC: Self = Self(0x0A4);
    /// The guest took a bus lock with its threshold counter exhausted. Both
    /// information fields are zero.
    pub const BUS_LOCK: Self = Self(0x0A5);
    /// The guest halted with no interrupt pending for it.
    pub const IDLE_HLT: Self = Self(0x0A6);
    /// A guest physical address could not be translated by the second set of
    /// page tables. Decode the first information field with [`NestedPageFault`]
    /// and read the faulting address from the second.
    pub const NPF: Self = Self(0x400);
    /// The hardware could not finish delivering an interrupt between the
    /// guest's own processors.
    pub const AVIC_INCOMPLETE_IPI: Self = Self(0x401);
    /// The guest touched an interrupt controller register the hardware does not
    /// handle on its own.
    pub const AVIC_NOACCEL: Self = Self(0x402);
    /// An encrypted guest made an explicit call to the hypervisor, part of an
    /// extension pulzar does not implement.
    pub const VMGEXIT: Self = Self(0x403);
    /// The control block held state the processor refuses, and no guest
    /// instruction ran. A hypervisor bug rather than anything the guest did.
    pub const INVALID: Self = Self(u64::MAX);
    /// The encrypted state area was busy, part of an extension pulzar does not
    /// implement.
    pub const BUSY: Self = Self(u64::MAX - 1);
    /// The other thread of this core was not idle when the guest required it to
    /// be.
    pub const IDLE_REQUIRED: Self = Self(u64::MAX - 2);
    /// The performance counter state was not valid for entry.
    pub const INVALID_PMC: Self = Self(u64::MAX - 3);

    /// The code reporting a read of this control register.
    #[must_use]
    pub const fn read_control_register(register: u8) -> Self {
        Self(READ_CR_BASE + (register as u64 % REGISTERS))
    }

    /// The code reporting a write of this control register.
    #[must_use]
    pub const fn write_control_register(register: u8) -> Self {
        Self(WRITE_CR_BASE + (register as u64 % REGISTERS))
    }

    /// The code reporting a read of this debug register.
    #[must_use]
    pub const fn read_debug_register(register: u8) -> Self {
        Self(READ_DR_BASE + (register as u64 % REGISTERS))
    }

    /// The code reporting a write of this debug register.
    #[must_use]
    pub const fn write_debug_register(register: u8) -> Self {
        Self(WRITE_DR_BASE + (register as u64 % REGISTERS))
    }

    /// The code reporting this exception, for the vectors that are exceptions.
    #[must_use]
    pub const fn exception(vector: Vector) -> Option<Self> {
        if !vector.is_exception() {
            return None;
        }
        Some(Self(EXCEPTION_BASE + vector.number() as u64))
    }

    /// The code reporting a control-register write that has already happened.
    #[must_use]
    pub const fn write_control_register_trap(register: u8) -> Self {
        Self(WRITE_CR_TRAP_BASE + (register as u64 % REGISTERS))
    }

    /// What this code means, or `None` if the architecture does not define it.
    ///
    /// An undefined code is worth distinguishing rather than lumping in with
    /// the rest: it means the processor reported something this code does not
    /// know about, which is a reason to stop rather than to guess.
    #[must_use]
    pub const fn reason(self) -> Option<Reason> {
        if let Some(reason) = self.indexed_reason() {
            return Some(reason);
        }
        Some(match self {
            Self::INTR => Reason::Interrupt,
            Self::NMI => Reason::Nmi,
            Self::SMI => Reason::Smi,
            Self::INIT => Reason::Init,
            Self::VINTR => Reason::VirtualInterrupt,
            Self::CR0_SEL_WRITE => Reason::SelectiveControlRegisterWrite,
            Self::IDTR_READ => Reason::ReadIdtr,
            Self::GDTR_READ => Reason::ReadGdtr,
            Self::LDTR_READ => Reason::ReadLdtr,
            Self::TR_READ => Reason::ReadTr,
            Self::IDTR_WRITE => Reason::WriteIdtr,
            Self::GDTR_WRITE => Reason::WriteGdtr,
            Self::LDTR_WRITE => Reason::WriteLdtr,
            Self::TR_WRITE => Reason::WriteTr,
            Self::RDTSC => Reason::Rdtsc,
            Self::RDPMC => Reason::Rdpmc,
            Self::PUSHF => Reason::Pushf,
            Self::POPF => Reason::Popf,
            Self::CPUID => Reason::Cpuid,
            Self::RSM => Reason::Rsm,
            Self::IRET => Reason::Iret,
            Self::SWINT => Reason::SoftwareInterrupt,
            Self::INVD => Reason::Invd,
            Self::PAUSE => Reason::Pause,
            Self::HLT => Reason::Hlt,
            Self::INVLPG => Reason::Invlpg,
            Self::INVLPGA => Reason::Invlpga,
            Self::IOIO => Reason::PortAccess,
            Self::MSR => Reason::MsrAccess,
            Self::TASK_SWITCH => Reason::TaskSwitch,
            Self::FERR_FREEZE => Reason::FerrFreeze,
            Self::SHUTDOWN => Reason::Shutdown,
            Self::VMRUN => Reason::Vmrun,
            Self::VMMCALL => Reason::Vmmcall,
            Self::VMLOAD => Reason::Vmload,
            Self::VMSAVE => Reason::Vmsave,
            Self::STGI => Reason::Stgi,
            Self::CLGI => Reason::Clgi,
            Self::SKINIT => Reason::Skinit,
            Self::RDTSCP => Reason::Rdtscp,
            Self::ICEBP => Reason::Icebp,
            Self::WBINVD => Reason::Wbinvd,
            Self::MONITOR => Reason::Monitor,
            Self::MWAIT => Reason::Mwait,
            Self::MWAIT_ARMED => Reason::MwaitArmed,
            Self::XSETBV => Reason::Xsetbv,
            Self::RDPRU => Reason::Rdpru,
            Self::EFER_WRITE_TRAP => Reason::WriteEferTrap,
            Self::INVLPGB => Reason::Invlpgb,
            Self::INVLPGB_ILLEGAL => Reason::InvlpgbIllegal,
            Self::INVPCID => Reason::Invpcid,
            Self::MCOMMIT => Reason::Mcommit,
            Self::TLBSYNC => Reason::Tlbsync,
            Self::BUS_LOCK => Reason::BusLock,
            Self::IDLE_HLT => Reason::IdleHlt,
            Self::NPF => Reason::NestedPageFault,
            Self::AVIC_INCOMPLETE_IPI => Reason::AvicIncompleteIpi,
            Self::AVIC_NOACCEL => Reason::AvicUnacceleratedAccess,
            Self::VMGEXIT => Reason::VmgExit,
            Self::INVALID => Reason::Invalid,
            Self::BUSY => Reason::Busy,
            Self::IDLE_REQUIRED => Reason::IdleRequired,
            Self::INVALID_PMC => Reason::InvalidPmc,
            _ => return None,
        })
    }

    /// The meaning of a code that falls in one of the indexed ranges.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "each range is at most thirty-two wide, so an offset within one is a byte"
    )]
    const fn indexed_reason(self) -> Option<Reason> {
        let code = self.0;
        Some(match code {
            _ if code < WRITE_CR_BASE => Reason::ReadControlRegister((code - READ_CR_BASE) as u8),
            _ if code < READ_DR_BASE => Reason::WriteControlRegister((code - WRITE_CR_BASE) as u8),
            _ if code < WRITE_DR_BASE => Reason::ReadDebugRegister((code - READ_DR_BASE) as u8),
            _ if code < EXCEPTION_BASE => Reason::WriteDebugRegister((code - WRITE_DR_BASE) as u8),
            _ if code < EXCEPTION_BASE + EXCEPTIONS => {
                Reason::Exception(Vector::new((code - EXCEPTION_BASE) as u8))
            }
            _ if code >= WRITE_CR_TRAP_BASE && code < WRITE_CR_TRAP_BASE + REGISTERS => {
                Reason::WriteControlRegisterTrap((code - WRITE_CR_TRAP_BASE) as u8)
            }
            _ => return None,
        })
    }

    /// The code as the control block holds it.
    #[must_use]
    pub const fn bits(self) -> u64 {
        self.0
    }

    /// The code from what the control block holds.
    #[must_use]
    pub const fn from_bits(bits: u64) -> Self {
        Self(bits)
    }
}

/// Prints what the code means rather than what it is, falling back to the
/// number for codes this crate does not know.
///
/// An exit code in a log is only useful as a reason, and looking one up in a
/// table is exactly the work this crate exists to have already done.
impl Debug for ExitCode {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self.reason() {
            Some(reason) => write!(formatter, "{reason:?}"),
            None => write!(formatter, "ExitCode({:#x})", self.0),
        }
    }
}

impl Display for ExitCode {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self.reason() {
            Some(Reason::Exception(vector)) => match vector.name() {
                Some(name) => write!(formatter, "exception {name}"),
                None => write!(formatter, "exception vector {}", vector.number()),
            },
            Some(reason) => write!(formatter, "{reason:?}"),
            None => write!(formatter, "unknown exit code {:#x}", self.0),
        }
    }
}

/// What an exit code means, with the ranges decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    /// The guest read this control register.
    ReadControlRegister(u8),
    /// The guest wrote this control register.
    WriteControlRegister(u8),
    /// The guest read this debug register.
    ReadDebugRegister(u8),
    /// The guest wrote this debug register.
    WriteDebugRegister(u8),
    /// This exception was raised in the guest. Its error code, if it has one,
    /// is in the first information field.
    Exception(Vector),
    /// The guest wrote this control register and the write has already taken
    /// effect.
    WriteControlRegisterTrap(u8),
    /// A maskable interrupt arrived while the guest was running.
    Interrupt,
    /// A non-maskable interrupt arrived.
    Nmi,
    /// A system management interrupt arrived.
    Smi,
    /// The signal that resets a processor was sent to the guest.
    Init,
    /// An interrupt made pending for the guest became deliverable.
    VirtualInterrupt,
    /// The guest wrote a bit of control register nought beyond the two
    /// concerning floating-point state.
    SelectiveControlRegisterWrite,
    /// The guest read the interrupt descriptor table register.
    ReadIdtr,
    /// The guest read the global descriptor table register.
    ReadGdtr,
    /// The guest read the local descriptor table register.
    ReadLdtr,
    /// The guest read the task register.
    ReadTr,
    /// The guest wrote the interrupt descriptor table register.
    WriteIdtr,
    /// The guest wrote the global descriptor table register.
    WriteGdtr,
    /// The guest wrote the local descriptor table register.
    WriteLdtr,
    /// The guest wrote the task register.
    WriteTr,
    /// The guest read the timestamp counter.
    Rdtsc,
    /// The guest read a performance counter.
    Rdpmc,
    /// The guest pushed its flags.
    Pushf,
    /// The guest popped its flags.
    Popf,
    /// The guest asked what processor it is running on.
    Cpuid,
    /// The guest returned from system management mode.
    Rsm,
    /// The guest returned from an interrupt handler.
    Iret,
    /// The guest executed the software interrupt instruction.
    SoftwareInterrupt,
    /// The guest invalidated its caches without writing them back.
    Invd,
    /// The guest spun often enough to trip the filter.
    Pause,
    /// The guest halted.
    Hlt,
    /// The guest invalidated one of its own translations.
    Invlpg,
    /// The guest invalidated a translation in another address space.
    Invlpga,
    /// The guest accessed a port that is intercepted.
    PortAccess,
    /// The guest accessed a model-specific register that is intercepted.
    MsrAccess,
    /// The guest switched tasks.
    TaskSwitch,
    /// The guest froze waiting for an external floating-point error signal.
    FerrFreeze,
    /// The guest triggered the condition that shuts a real processor down.
    Shutdown,
    /// The guest tried to run a guest of its own.
    Vmrun,
    /// The guest called the hypervisor deliberately.
    Vmmcall,
    /// The guest tried to load processor state from a control block.
    Vmload,
    /// The guest tried to save processor state to a control block.
    Vmsave,
    /// The guest tried to set the global interrupt flag.
    Stgi,
    /// The guest tried to clear the global interrupt flag.
    Clgi,
    /// The guest tried to begin a measured launch.
    Skinit,
    /// The guest read the timestamp counter with the processor identifier.
    Rdtscp,
    /// The guest executed the in-circuit emulator breakpoint.
    Icebp,
    /// The guest wrote its caches back and invalidated them.
    Wbinvd,
    /// The guest armed the address monitor.
    Monitor,
    /// The guest waited on the address monitor.
    Mwait,
    /// The guest waited on an armed address monitor.
    MwaitArmed,
    /// The guest enabled an extended processor state component.
    Xsetbv,
    /// The guest read a processor register through the user-mode mechanism.
    Rdpru,
    /// The guest wrote its extended feature register, after the fact.
    WriteEferTrap,
    /// The guest broadcast a translation invalidation.
    Invlpgb,
    /// The guest broadcast an invalidation the architecture does not allow.
    InvlpgbIllegal,
    /// The guest invalidated translations by address-space identifier.
    Invpcid,
    /// The guest committed its outstanding memory operations.
    Mcommit,
    /// The guest waited for its broadcast invalidations to finish.
    Tlbsync,
    /// The guest took a bus lock with its threshold exhausted.
    BusLock,
    /// The guest halted with nothing pending for it.
    IdleHlt,
    /// A guest physical address could not be translated.
    NestedPageFault,
    /// The hardware could not finish delivering an interrupt between the
    /// guest's processors.
    AvicIncompleteIpi,
    /// The guest touched an interrupt controller register the hardware does not
    /// accelerate.
    AvicUnacceleratedAccess,
    /// An encrypted guest called the hypervisor explicitly.
    VmgExit,
    /// The control block held state the processor refuses; nothing in the guest
    /// ran.
    Invalid,
    /// The encrypted state area was busy.
    Busy,
    /// The other thread of this core was not idle when required to be.
    IdleRequired,
    /// The performance counter state was not valid for entry.
    InvalidPmc,
}

/// What the guest was doing when a port access was intercepted.
///
/// The width fields are one-hot rather than a number, which is why the widths
/// are read through [`IoAccess::operand_bytes`] and
/// [`IoAccess::address_bytes`] instead of directly: exactly one bit of each
/// group should be set, and a report where that does not hold is not something
/// to silently round off.
///
/// The address of the instruction *after* the access is in the second
/// information field, so a hypervisor can emulate the access and resume without
/// decoding anything.
#[bitfield_struct::bitfield(u64)]
#[derive(PartialEq, Eq)]
pub struct IoAccess {
    /// The guest was reading from the port rather than writing to it.
    pub is_input: bool,
    __: bool,
    /// The access was one of the string forms, which move to or from memory.
    pub string: bool,
    /// The access was repeated by a prefix.
    pub repeated: bool,
    /// The operand was one byte wide.
    pub operand_8: bool,
    /// The operand was two bytes wide.
    pub operand_16: bool,
    /// The operand was four bytes wide.
    pub operand_32: bool,
    /// The address was two bytes wide.
    pub address_16: bool,
    /// The address was four bytes wide.
    pub address_32: bool,
    /// The address was eight bytes wide.
    pub address_64: bool,
    /// Which segment the string form used. The reading forms always report the
    /// extra segment, encoded as zero.
    #[bits(3)]
    pub segment: u8,
    #[bits(3)]
    __: u8,
    /// Which port was accessed.
    pub port: u16,
    #[bits(32)]
    __: u32,
}

impl IoAccess {
    /// How wide the operand was, or `None` if the report does not name exactly
    /// one width.
    #[must_use]
    pub const fn operand_bytes(self) -> Option<u8> {
        match (self.operand_8(), self.operand_16(), self.operand_32()) {
            (true, false, false) => Some(1),
            (false, true, false) => Some(2),
            (false, false, true) => Some(4),
            _ => None,
        }
    }

    /// How wide the address was, or `None` if the report does not name exactly
    /// one width.
    #[must_use]
    pub const fn address_bytes(self) -> Option<u8> {
        match (self.address_16(), self.address_32(), self.address_64()) {
            (true, false, false) => Some(2),
            (false, true, false) => Some(4),
            (false, false, true) => Some(8),
            _ => None,
        }
    }
}

/// Why a guest physical address could not be translated.
///
/// The low bits are an ordinary page-fault error code describing the attempted
/// access. The two above them are what makes a nested fault different from a
/// guest's own: they say whether the address that failed was the one the guest
/// was actually after, or an address of one of the guest's own page tables that
/// the processor had to translate on the way to it.
///
/// The guest physical address that faulted is in the second information field.
#[bitfield_struct::bitfield(u64)]
#[derive(PartialEq, Eq)]
pub struct NestedPageFault {
    /// A translation was present; the fault was a permission violation rather
    /// than a missing page.
    pub present: bool,
    /// The access was a write.
    pub write: bool,
    /// The access was made with user privilege. Walks of the guest's own page
    /// tables count as user accesses unless something overrides that.
    pub user: bool,
    /// A reserved bit was set in one of the entries walked.
    pub reserved_bit: bool,
    /// The access was an instruction fetch. Walks for the guest's page tables
    /// are always reported as data writes even when the access that needed them
    /// was a fetch.
    pub instruction_fetch: bool,
    __: bool,
    /// The access was a shadow stack access.
    pub shadow_stack: bool,
    #[bits(25)]
    __: u32,
    /// The fault happened translating the guest physical address the guest was
    /// actually after.
    pub final_address: bool,
    /// The fault happened translating one of the guest's own page tables.
    pub page_table_walk: bool,
    #[bits(3)]
    __: u8,
    /// The page was marked as a supervisor shadow stack page and that
    /// restriction is enabled.
    pub supervisor_shadow_stack: bool,
    #[bits(26)]
    __: u32,
}

/// Which register a guest named in an intercepted control-register access.
///
/// Two instructions reach control register nought without naming a register at
/// all — the one that loads the machine status word and the one that clears the
/// task-switched flag — and for those the processor reports no register number
/// and leaves [`MovCr::is_mov`] clear. A hypervisor that reads the register
/// number without checking that flag emulates the wrong instruction against
/// register zero.
#[bitfield_struct::bitfield(u64)]
#[derive(PartialEq, Eq)]
pub struct MovCr {
    /// Which general-purpose register the instruction named.
    #[bits(4)]
    pub register: u8,
    #[bits(59)]
    __: u64,
    /// The instruction really was a register-to-control-register move, so the
    /// register number above means something.
    pub is_mov: bool,
}

/// Which register a guest named in an intercepted debug-register access.
#[bitfield_struct::bitfield(u64)]
#[derive(PartialEq, Eq)]
pub struct MovDr {
    /// Which general-purpose register the instruction named.
    #[bits(4)]
    pub register: u8,
    #[bits(60)]
    __: u64,
}

const _: () = assert!(
    size_of::<ExitCode>() == size_of::<u64>(),
    "an exit code is a quadword of the control area",
);
const _: () = assert!(
    matches!(ExitCode::NPF.reason(), Some(Reason::NestedPageFault)),
    "the nested page fault code must decode to its own reason",
);
const _: () = assert!(
    matches!(
        ExitCode::read_control_register(3).reason(),
        Some(Reason::ReadControlRegister(3)),
    ),
    "an indexed control register code must decode back to its register",
);
const _: () = assert!(
    matches!(
        ExitCode::write_control_register(4).reason(),
        Some(Reason::WriteControlRegister(4)),
    ),
    "a control register write code must decode back to its register",
);
const _: () = assert!(
    matches!(
        ExitCode::write_control_register_trap(8).reason(),
        Some(Reason::WriteControlRegisterTrap(8)),
    ),
    "a write trap code must decode back to its register",
);
const _: () = assert!(
    ExitCode::exception(Vector::new(14)).is_some()
        && ExitCode::exception(Vector::new(14)).unwrap().bits() == 0x04E,
    "the page fault exception lands where the architecture puts it",
);
const _: () = assert!(
    ExitCode::exception(Vector::new(32)).is_none(),
    "a vector that is not an exception has no exit code",
);
