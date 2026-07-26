//! What a guest may not do without the hypervisor hearing about it.
//!
//! Running a guest is really a negotiation about this: everything the guest
//! does runs at full speed on the real processor *except* what is listed here,
//! and each of these costs an exit and a round trip through the hypervisor
//! every time the guest does it. Intercepting too little loses control of the
//! machine; intercepting too much makes the guest crawl. So these vectors are
//! the single most consequential thing in a guest's control block, and they are
//! six separate words because they accumulated over two decades rather than
//! because six is meaningful.
//!
//! # Two shapes of vector
//!
//! Four of the six are ordinary flag sets, one flag per thing a guest might do.
//! The other two are *indexed*: the control-register and debug-register vectors
//! give each register a bit for reading and another for writing, so what a
//! caller wants to say is "reads of control register nought", not "bit
//! sixteen". Those two get constructors that take the register number and put
//! the bit in the right half, because computing `16 + n` by hand at a use site
//! is exactly the arithmetic this module exists to do once.
//!
//! # The one that is not optional
//!
//! A guest that is allowed to execute the instruction that enters a guest could
//! run a guest of its own with a control block the hypervisor never saw. That
//! intercept must always be set, and entering a guest without it fails. It is
//! documented on the flag, because nothing else in this module has a
//! consequence like it.
//!
//! # Traps that fire after the fact
//!
//! Most intercepts are faults: the guest's instruction has not taken effect,
//! and the hypervisor decides what happens. A few — the extended feature
//! register write and the control-register write traps — are *traps*: the write
//! has already happened and the hypervisor is being told, not asked. They live
//! in the upper half of the second vector and are named for what they are, so
//! that nobody handles one expecting to be able to refuse it.

use bitflags::bitflags;
use descriptors::Vector;

/// How many control and debug registers the architecture gives an intercept bit
/// each, in both directions.
const REGISTERS: u8 = 16;

/// Bits between a register's read bit and its write bit in the indexed
/// vectors.
const WRITE_SHIFT: u8 = 16;

/// Which reads and writes of a control register are intercepted.
///
/// Bits zero to fifteen are reads of control registers nought to fifteen, and
/// bits sixteen to thirty-one are the corresponding writes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct ControlRegisterIntercepts(u32);

/// Which reads and writes of a debug register are intercepted.
///
/// Laid out exactly as [`ControlRegisterIntercepts`]: reads low, writes high.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct DebugRegisterIntercepts(u32);

/// Builds the two indexed vectors, which differ only in which registers they
/// name.
///
/// Both are the same sixteen-reads-then-sixteen-writes layout over the same
/// word, and writing that out twice would be two places for the shift to be
/// wrong.
macro_rules! indexed_intercepts {
    ($type:ty, $what:literal) => {
        impl $type {
            #[doc = concat!("No ", $what, " register access is intercepted.")]
            pub const EMPTY: Self = Self(0);

            #[doc = concat!("Every read and write of every ", $what, " register is intercepted.")]
            pub const ALL: Self = Self(u32::MAX);

            /// The same set, plus reads of this register.
            ///
            /// Only registers nought to fifteen exist, and a number above that
            /// names no register — it is masked to the four bits the field
            /// has rather than wrapping into the write half, which is what
            /// makes an out-of-range number harmless instead of silently
            /// intercepting something else.
            #[must_use]
            pub const fn with_read(self, register: u8) -> Self {
                Self(self.0 | (1 << (register % REGISTERS)))
            }

            /// The same set, plus writes of this register.
            ///
            /// Out-of-range numbers are masked as in
            #[doc = concat!("[`", stringify!($type), "::with_read`].")]
            #[must_use]
            pub const fn with_write(self, register: u8) -> Self {
                Self(self.0 | (1 << ((register % REGISTERS) + WRITE_SHIFT)))
            }

            /// Whether reads of this register are intercepted.
            #[must_use]
            pub const fn intercepts_read(self, register: u8) -> bool {
                self.0 & (1 << (register % REGISTERS)) != 0
            }

            /// Whether writes of this register are intercepted.
            #[must_use]
            pub const fn intercepts_write(self, register: u8) -> bool {
                self.0 & (1 << ((register % REGISTERS) + WRITE_SHIFT)) != 0
            }

            /// Everything intercepted by either set.
            #[must_use]
            pub const fn union(self, other: Self) -> Self {
                Self(self.0 | other.0)
            }

            /// The vector as the control block holds it.
            #[must_use]
            pub const fn bits(self) -> u32 {
                self.0
            }

            /// The vector from what the control block holds.
            #[must_use]
            pub const fn from_bits(bits: u32) -> Self {
                Self(bits)
            }
        }
    };
}

indexed_intercepts!(ControlRegisterIntercepts, "control");
indexed_intercepts!(DebugRegisterIntercepts, "debug");

/// Which exceptions raised inside the guest come back to the hypervisor instead
/// of being delivered through the guest's own descriptor table.
///
/// One bit per vector, and only the thirty-two architectural exception vectors
/// have one — an interrupt vector the platform assigns is not an exception and
/// cannot be intercepted this way.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct ExceptionIntercepts(u32);

impl ExceptionIntercepts {
    /// No exception is intercepted; the guest handles all of its own.
    pub const EMPTY: Self = Self(0);

    /// The same set, plus this exception.
    ///
    /// A vector that is not an exception leaves the set unchanged, because
    /// there is no bit for it: the field covers vectors nought to thirty-one
    /// and nothing else.
    #[must_use]
    pub const fn with(self, vector: Vector) -> Self {
        if !vector.is_exception() {
            return self;
        }
        Self(self.0 | (1 << vector.number()))
    }

    /// Whether this exception is intercepted.
    #[must_use]
    pub const fn intercepts(self, vector: Vector) -> bool {
        if !vector.is_exception() {
            return false;
        }
        self.0 & (1 << vector.number()) != 0
    }

    /// Everything intercepted by either set.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// An empty set, for building one up.
    #[must_use]
    pub const fn empty() -> Self {
        Self::EMPTY
    }

    /// The vector as the control block holds it.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// The vector from what the control block holds.
    #[must_use]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }
}

bitflags! {
    /// Interrupts, descriptor-table access, and the instruction intercepts that
    /// have been there since the beginning.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Intercepts1: u32 {
        /// A maskable interrupt arrives while the guest is running. Set with
        /// virtualized interrupt masking, this is how the hypervisor keeps
        /// interrupts that belong to it from being swallowed by a guest.
        const INTR = 1 << 0;
        /// A non-maskable interrupt arrives. Required if non-maskable interrupt
        /// masking is to be virtualized for the guest.
        const NMI = 1 << 1;
        /// A system management interrupt arrives.
        const SMI = 1 << 2;
        /// The guest is sent the signal that resets a processor.
        const INIT = 1 << 3;
        /// An interrupt the hypervisor made pending for the guest becomes
        /// deliverable — the guest has lowered its priority or re-enabled
        /// interrupts, and is now willing to take it.
        const VINTR = 1 << 4;
        /// The guest writes a bit of control register nought other than the two
        /// that only concern floating-point state. Those two are excluded
        /// because a guest toggles them on nearly every task switch, and
        /// intercepting that would be ruinous.
        const SELECTIVE_CR0_WRITE = 1 << 5;
        /// The guest reads the interrupt descriptor table register.
        const READ_IDTR = 1 << 6;
        /// The guest reads the global descriptor table register.
        const READ_GDTR = 1 << 7;
        /// The guest reads the local descriptor table register.
        const READ_LDTR = 1 << 8;
        /// The guest reads the task register.
        const READ_TR = 1 << 9;
        /// The guest writes the interrupt descriptor table register.
        const WRITE_IDTR = 1 << 10;
        /// The guest writes the global descriptor table register.
        const WRITE_GDTR = 1 << 11;
        /// The guest writes the local descriptor table register.
        const WRITE_LDTR = 1 << 12;
        /// The guest writes the task register.
        const WRITE_TR = 1 << 13;
        /// The guest reads the timestamp counter. Usually left clear in favour
        /// of the offset and ratio, which give the guest a consistent clock
        /// without an exit per read.
        const RDTSC = 1 << 14;
        /// The guest reads a performance counter.
        const RDPMC = 1 << 15;
        /// The guest pushes its flags onto its own stack.
        const PUSHF = 1 << 16;
        /// The guest pops its flags from its own stack.
        const POPF = 1 << 17;
        /// The guest asks what processor it is running on — the intercept that
        /// makes it possible to tell a guest something other than the truth.
        const CPUID = 1 << 18;
        /// The guest returns from system management mode.
        const RSM = 1 << 19;
        /// The guest returns from an interrupt handler. Without virtualized
        /// non-maskable interrupt masking, this is how a hypervisor learns that
        /// a guest has finished handling one.
        const IRET = 1 << 20;
        /// The guest executes the software interrupt instruction.
        const INTN = 1 << 21;
        /// The guest invalidates its caches without writing them back.
        const INVD = 1 << 22;
        /// The guest executes the spin hint. Worth intercepting only with the
        /// filter below, since a bare intercept fires on every spin.
        const PAUSE = 1 << 23;
        /// The guest halts. A guest with nothing to do is a guest whose
        /// processor could be given to somebody else.
        const HLT = 1 << 24;
        /// The guest invalidates one of its own translations.
        const INVLPG = 1 << 25;
        /// The guest invalidates a translation in another address space.
        const INVLPGA = 1 << 26;
        /// Port access is filtered through the port permission bitmap. Without
        /// this the map is not consulted at all.
        const IOIO_PROT = 1 << 27;
        /// Model-specific register access is filtered through the register
        /// permission bitmap. Without this the map is not consulted, and with
        /// it any register the map does not cover is intercepted.
        const MSR_PROT = 1 << 28;
        /// The guest switches tasks.
        const TASK_SWITCH = 1 << 29;
        /// The guest freezes waiting for an external floating-point error
        /// signal, a mechanism no modern machine wires up.
        const FERR_FREEZE = 1 << 30;
        /// The guest triggers the condition that would shut a real processor
        /// down — a fault while handling a fault while handling a fault.
        /// Intercepting it turns a guest killing itself into an exit rather
        /// than a dead machine.
        const SHUTDOWN = 1 << 31;
    }
}

bitflags! {
    /// The virtualization instructions and the newer instruction intercepts.
    ///
    /// The low half of the second vector. Its upper half is not flags but
    /// indexed write traps, which is why the vector as a whole is
    /// [`Intercepts2`] and this is only what fits in a flag set.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Intercepts2Flags: u32 {
        /// The guest runs a guest of its own. Must always be set: a guest
        /// permitted this could enter a guest with a control block the
        /// hypervisor has never inspected, and entering a guest without this
        /// intercept fails outright.
        const VMRUN = 1 << 0;
        /// The guest makes an explicit call to the hypervisor. The one
        /// intercept that exists to be used rather than to be defended against.
        const VMMCALL = 1 << 1;
        /// The guest loads processor state from a control block.
        const VMLOAD = 1 << 2;
        /// The guest saves processor state to a control block.
        const VMSAVE = 1 << 3;
        /// The guest sets the global interrupt flag.
        const STGI = 1 << 4;
        /// The guest clears the global interrupt flag.
        const CLGI = 1 << 5;
        /// The guest begins a measured launch.
        const SKINIT = 1 << 6;
        /// The guest reads the timestamp counter together with the processor
        /// identifier.
        const RDTSCP = 1 << 7;
        /// The guest executes the in-circuit emulator breakpoint.
        const ICEBP = 1 << 8;
        /// The guest writes its caches back and invalidates them, with or
        /// without the invalidation.
        const WBINVD = 1 << 9;
        /// The guest arms the address monitor.
        const MONITOR = 1 << 10;
        /// The guest waits on the address monitor, whether or not the monitor
        /// was armed.
        const MWAIT = 1 << 11;
        /// The guest waits on the address monitor, but only when it is armed.
        /// Checked before the unconditional intercept, so a hypervisor that
        /// sets both can tell a wait that would have slept from one that would
        /// not.
        const MWAIT_ARMED = 1 << 12;
        /// The guest enables an extended processor state component.
        const XSETBV = 1 << 13;
        /// The guest reads a processor register through the user-mode
        /// mechanism.
        const RDPRU = 1 << 14;
        /// The guest has written its extended feature register. A trap, not a
        /// fault: the write has already happened.
        const EFER_WRITE_TRAP = 1 << 15;
    }
}

/// The second intercept vector: virtualization instructions in its low half,
/// control-register write traps in its high half.
///
/// The two halves are different in kind, not just in position. The flags are
/// faults — the guest's instruction has not taken effect and the hypervisor
/// decides what happens next. The write traps fire *after* the guest's write
/// has already changed the register, so the hypervisor is being informed rather
/// than consulted. Mixing them up means writing a handler that tries to refuse
/// something that has already happened.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct Intercepts2(u32);

impl Intercepts2 {
    /// Nothing in either half is intercepted.
    pub const EMPTY: Self = Self(0);

    /// The set with these flags, and no write traps.
    #[must_use]
    pub const fn from_flags(flags: Intercepts2Flags) -> Self {
        Self(flags.bits())
    }

    /// The flags in the low half.
    #[must_use]
    pub const fn flags(self) -> Intercepts2Flags {
        Intercepts2Flags::from_bits_truncate(self.0 & LOW_HALF)
    }

    /// The same set with these flags added.
    #[must_use]
    pub const fn with_flags(self, flags: Intercepts2Flags) -> Self {
        Self(self.0 | flags.bits())
    }

    /// The same set, also trapping writes of this control register after they
    /// take effect.
    ///
    /// Out-of-range register numbers are masked to the four bits the field has,
    /// as in the indexed vectors, so one cannot reach down into the flags.
    #[must_use]
    pub const fn with_write_trap(self, register: u8) -> Self {
        Self(self.0 | (1 << ((register % REGISTERS) + WRITE_SHIFT)))
    }

    /// Whether writes of this control register are trapped after the fact.
    #[must_use]
    pub const fn traps_write(self, register: u8) -> bool {
        self.0 & (1 << ((register % REGISTERS) + WRITE_SHIFT)) != 0
    }

    /// The vector as the control block holds it.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// The vector from what the control block holds.
    #[must_use]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }
}

/// Mask of the half of the second vector that is flags rather than write
/// traps.
const LOW_HALF: u32 = 0x0000_FFFF;

bitflags! {
    /// The most recently added intercepts.
    ///
    /// Every one of these depends on the processor supporting it, and several
    /// have no effect at all on a machine that does not — which is why nothing
    /// here should be set without asking the processor first.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Intercepts3: u32 {
        /// The guest broadcasts a translation invalidation to every processor.
        const INVLPGB = 1 << 0;
        /// The guest broadcasts an invalidation with operands the architecture
        /// does not allow — an intercept for the illegal cases only, so the
        /// legal ones stay fast.
        const INVLPGB_ILLEGAL = 1 << 1;
        /// The guest invalidates translations by address-space identifier.
        const INVPCID = 1 << 2;
        /// The guest commits its outstanding memory operations.
        const MCOMMIT = 1 << 3;
        /// The guest waits for its broadcast invalidations to complete.
        /// Available only on processors that report the broadcast invalidation
        /// control feature.
        const TLBSYNC = 1 << 4;
        /// The guest takes a bus lock while its threshold counter has run out.
        /// Fires before the guest's instruction executes, which is what makes
        /// it possible to stop one guest from monopolizing the memory bus.
        const BUS_LOCK = 1 << 5;
        /// The guest halts with no interrupt pending for it. Distinct from the
        /// plain halt intercept, and lower priority: with both set, a halt that
        /// would block reports as an ordinary halt. It exists so a hypervisor
        /// can notice a genuinely idle guest without paying an exit for every
        /// halt that is about to be woken anyway.
        const IDLE_HLT = 1 << 6;
    }
}

/// What to do with the guest's cached translations on the way in.
///
/// Translations are tagged with an address-space identifier so that a guest's
/// need not be discarded when the hypervisor runs, but a hypervisor that
/// changes a guest's page tables must still get rid of what the processor
/// cached from them. This says how much to get rid of.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum TlbControl {
    /// Keep everything. The normal case, and the only one that costs nothing.
    #[default]
    DoNothing = 0,
    /// Discard every translation for every address space. Enormously
    /// expensive — it throws away the host's translations and every other
    /// guest's along with this one's. Intended for processors too old to do
    /// anything finer.
    FlushAll = 1,
    /// Discard this guest's translations. Requires a processor that reports
    /// flush-by-identifier.
    FlushGuest = 3,
    /// Discard this guest's translations except those it marked global.
    /// Requires flush-by-identifier.
    FlushGuestNonGlobal = 7,
}

impl TlbControl {
    /// The command an encoding names.
    ///
    /// Encodings the architecture reserves are read as [`TlbControl::FlushAll`]
    /// rather than as doing nothing: an unknown command is a control block this
    /// code did not write, and over-flushing is slow where under-flushing runs
    /// a guest on translations that no longer describe its memory.
    #[must_use]
    pub const fn from_bits(bits: u8) -> Self {
        match bits {
            0 => Self::DoNothing,
            3 => Self::FlushGuest,
            7 => Self::FlushGuestNonGlobal,
            _ => Self::FlushAll,
        }
    }

    /// The encoding for this command.
    #[must_use]
    pub const fn into_bits(self) -> u8 {
        self as u8
    }
}

bitflags! {
    /// What to do with the return-address predictor on the way in.
    ///
    /// The predictor guesses where a return instruction will go, and entries a
    /// guest left in it are entries the next thing to run inherits. These
    /// control how much of that crosses the boundary.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct ErapControl: u8 {
        /// Give the guest the processor's full predictor rather than the
        /// thirty-two entries it would otherwise be limited to.
        const ALLOW_LARGER_RAP = 1 << 0;
        /// Empty the predictor on the way into the guest, so nothing the host
        /// left in it can steer the guest's returns.
        const CLEAR_RAP = 1 << 1;
    }
}

const _: () = assert!(
    size_of::<ControlRegisterIntercepts>() == size_of::<u32>()
        && size_of::<DebugRegisterIntercepts>() == size_of::<u32>()
        && size_of::<ExceptionIntercepts>() == size_of::<u32>()
        && size_of::<Intercepts1>() == size_of::<u32>()
        && size_of::<Intercepts2>() == size_of::<u32>()
        && size_of::<Intercepts3>() == size_of::<u32>(),
    "every intercept vector is one doubleword of the control area",
);
const _: () = assert!(
    size_of::<TlbControl>() == 1 && size_of::<ErapControl>() == 1,
    "the translation and predictor controls are one byte each",
);
const _: () = assert!(
    Intercepts2::EMPTY
        .with_flags(Intercepts2Flags::VMRUN)
        .with_write_trap(0)
        .bits()
        == 1 | (1 << WRITE_SHIFT),
    "a write trap must not land in the flag half",
);
