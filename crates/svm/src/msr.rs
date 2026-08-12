//! The switches that live outside any guest's control block: whether this
//! extension may be turned on at all, and where the processor keeps our own
//! state while a guest has the machine.
//!
//! Everything else in this crate is memory the processor reads by physical
//! address, one page per guest. These are not. They are model-specific
//! registers, reached by number, and they exist whether or not any guest does —
//! which is what unites an otherwise unrelated-looking set. [`VM_CR`] and
//! [`SVM_KEY`] settle whether virtualization can be enabled on this machine at
//! all, a decision firmware usually made long before we ran. [`VM_HSAVE_PA`]
//! names the page the host's own state is swapped out to, and it belongs to the
//! processor rather than to any one guest, which is exactly why it is a
//! register and not a field of a control block. [`TSC_RATIO`] and [`IGNNE`]
//! adjust things a guest observes that no control block has a field for.
//!
//! # An address and a meaning, and nothing else
//!
//! Each register here is an address constant paired with a type describing the
//! bits at it. Nothing in this module reads or writes anything: a caller brings
//! its own way of reaching model-specific registers, and all this module
//! promises is that the value handed over means what the caller thinks it
//! means. Keeping the two halves separate is what makes the addresses useful to
//! code that intercepts a guest's access to a register it will never itself
//! write.
//!
//! # The one to read when nothing works
//!
//! A machine can report this extension in its `CPUID` leaves and still refuse
//! to run a guest, because firmware turned it off. [`VmCr`] is where that
//! shows, and reading it back is how a hypervisor tells "this processor cannot"
//! from "this machine was configured not to" — very different things to report
//! to whoever is trying to boot it.
//!
//! # The one register here that is not this extension's
//!
//! The bit that actually switches the extension on is not in this block. It
//! lives in the extended feature register, which is architectural rather than
//! particular to this extension — but a hypervisor hiding the extension has to
//! intercept that register too, and saying which permission bit to set takes an
//! address. So [`EFER`] and [`EFER_RESERVED`] are here, beside the block whose
//! meaning they are needed for, while the bit itself stays
//! [`EferFlags`](x86_64::registers::model_specific::EferFlags)`::SECURE_VIRTUAL_MACHINE_ENABLE`
//! as the `x86_64` crate spells it: one name for one bit.
//! [`VmCr::svm_disabled`] is the reason writing that bit can quietly fail to
//! take.
//!
//! # What is deliberately absent
//!
//! The doorbell register a hypervisor pokes to signal a processor already
//! running a guest is numbered in this same block, but it means nothing apart
//! from the hardware-driven interrupt controller it serves, so it is defined
//! beside that controller in [`crate::avic`].
//!
//! Past the registers below the block continues into the
//! encrypted-virtualization extension and into system management mode, neither
//! of which this crate models.

use core::fmt::{self, Debug, Formatter};

use bitfield_struct::bitfield;
use x86_64::{PhysAddr, registers::model_specific::EferFlags};

/// `VM_CR`, the register deciding whether this extension may be used on this
/// machine, with three unrelated switches sharing the space.
pub const VM_CR: u32 = 0xC001_0114;

/// The contents of [`VM_CR`].
///
/// Two of these five bits are the answer to why a machine whose `CPUID`
/// advertises this extension still will not run a guest; the other three are
/// unrelated debug and compatibility switches that happen to live in the same
/// register.
///
/// Those two interlock. [`lock`](Self::lock) protects itself and
/// [`svm_disabled`](Self::svm_disabled) from being written, and
/// [`svm_disabled`](Self::svm_disabled) is the effective switch. Read together
/// with the lock feature's own `CPUID` bit they distinguish three states: the
/// extension is usable, or firmware disabled it and left a key that could
/// re-enable it, or firmware disabled it on a processor with no key mechanism
/// at all — the last of which no amount of software can undo, and which has to
/// be reported to whoever is able to change the firmware setting.
#[bitfield(u64)]
#[derive(PartialEq, Eq)]
pub struct VmCr {
    /// `DPD`: disables the external hardware debug port along with certain
    /// debug features internal to the processor.
    pub debug_port_disabled: bool,
    /// `R_INIT`: an `INIT` signal that nothing intercepts arrives as a `#SX`
    /// security exception instead, so that software gets a chance to see it
    /// rather than losing the processor to a reset it never agreed to.
    pub init_redirect: bool,
    /// `DIS_A20M`: disables A20 masking, the compatibility arrangement that
    /// folds addresses over at the first megabyte.
    pub a20_masking_disabled: bool,
    /// `LOCK`: while this is set, writes to it and to
    /// [`svm_disabled`](Self::svm_disabled) are silently ignored — no fault and
    /// no diagnostic, the write simply does not happen, and reading the
    /// register back is the only way to notice. Clearing it takes a write of
    /// the matching key to [`SVM_KEY`] and nothing else; neither `INIT` nor
    /// `SKINIT` disturbs it. Firmware sets this when it means its decision
    /// about the extension to be final.
    pub lock: bool,
    /// `SVMDIS`: while this is set the extension cannot be enabled at all —
    /// writes of the extended feature register treat its enable bit as
    /// must-be-zero, so the bit never takes and the failure is silent.
    ///
    /// `CPUID` goes on reporting the extension as present the whole time, which
    /// is precisely why this register has to be read rather than the feature
    /// bit trusted: the feature bit says the silicon has the extension, and
    /// this says whether we are permitted to use it. On a processor that lacks
    /// the lock feature there is no key and no way back, and this bit read back
    /// is the entire answer — firmware disabled the extension and the machine
    /// must be reconfigured by hand.
    ///
    /// Setting this while the extension is already enabled raises `#GP`,
    /// whatever [`lock`](Self::lock) currently says. `INIT` clears it only when
    /// `LOCK` is clear, and `SKINIT` never does.
    pub svm_disabled: bool,
    #[bits(59)]
    __: u64,
}

impl VmCr {
    /// Every bit the architecture assigns a meaning to, which is the low five.
    ///
    /// Built from the fields rather than written as a number so that it cannot
    /// disagree with them.
    pub const DEFINED: u64 = Self::new()
        .with_debug_port_disabled(true)
        .with_init_redirect(true)
        .with_a20_masking_disabled(true)
        .with_lock(true)
        .with_svm_disabled(true)
        .into_bits();

    /// Every bit the architecture reserves, which a write must leave clear.
    ///
    /// These are must-be-zero rather than ignored: a `WRMSR` setting one raises
    /// `#GP`, so this is the mask a write is judged against rather than masked
    /// with.
    pub const RESERVED: u64 = !Self::DEFINED;

    /// The two bits [`lock`](Self::lock) protects, itself included.
    ///
    /// While it is set, a write of either is discarded without a fault and
    /// without any other trace, so this is the mask of what a locked register
    /// keeps regardless of what software writes.
    pub const LOCKED: u64 = Self::new()
        .with_lock(true)
        .with_svm_disabled(true)
        .into_bits();
}

const _: () = assert!(
    VmCr::DEFINED == 0b1_1111 && VmCr::LOCKED == 0b1_1000,
    "the five defined bits are the low five, and the locked pair the top two of them",
);

/// `EFER`, the architectural extended feature register, where the bit that
/// actually enables this extension lives.
///
/// Numbered outside this block and defined by the architecture rather than by
/// this extension, which is why the bits are `x86_64`'s
/// [`EferFlags`] and not a type of this crate's. The address is here because a
/// hypervisor hiding the extension has to intercept the register, and naming a
/// register to intercept takes its address.
pub const EFER: u32 = 0xC000_0080;

/// The bits of [`EFER`] that are reserved on every processor.
///
/// Reserved here means must-be-zero: a `WRMSR` that sets one raises `#GP`
/// rather than dropping it quietly, so this is the mask a write is judged
/// against rather than masked with.
///
/// The four flags a recent processor has above the ones [`EferFlags`] names are
/// deliberately treated as writable even though they are reserved on a
/// processor that lacks them. Which processor that is takes a feature query per
/// flag, and faulting a write that the machine underneath would have accepted
/// is the worse mistake of the two.
pub const EFER_RESERVED: u64 = !(EferFlags::all().bits()
    | EFER_MCOMMIT
    | EFER_INTERRUPTIBLE_WBINVD
    | EFER_UPPER_ADDRESS_IGNORE
    | EFER_AUTOMATIC_IBRS);

/// `MCOMMIT`: enables the instruction that waits for stores to become
/// non-cancellable.
const EFER_MCOMMIT: u64 = 1 << 17;

/// `INTWB`: makes cache writeback interruptible, so a long one does not hold a
/// processor past every interrupt it should have taken.
const EFER_INTERRUPTIBLE_WBINVD: u64 = 1 << 18;

/// `UAIE`: ignores the upper address bits rather than requiring them to be a
/// sign extension of the address.
const EFER_UPPER_ADDRESS_IGNORE: u64 = 1 << 20;

/// `AIBRSE`: keeps indirect branch prediction restricted whenever the processor
/// is in supervisor mode, without software having to ask each time.
const EFER_AUTOMATIC_IBRS: u64 = 1 << 21;

const _: () = assert!(
    EFER_RESERVED & EferFlags::SECURE_VIRTUAL_MACHINE_ENABLE.bits() == 0
        && EFER_RESERVED & EferFlags::LONG_MODE_ENABLE.bits() == 0
        && EFER_RESERVED & EferFlags::LONG_MODE_ACTIVE.bits() == 0,
    "the bits a hypervisor forces, tests and preserves must not be reserved",
);
const _: () = assert!(
    EFER_RESERVED == 0xFFFF_FFFF_FFC9_02FE,
    "the reserved bits of the extended feature register are 63:22, 19, 16, 9 and 7:1",
);

/// `IGNNE`, which drives the processor-internal signal of the same name.
pub const IGNNE: u32 = 0xC001_0115;

/// The contents of [`IGNNE`], which is readable and writable.
///
/// The signal this sets says whether an unmasked x87 floating-point error is
/// ignored rather than reported through the external pin the original design
/// used. Setting it here means something only once emulation of that signal has
/// been enabled in the hardware configuration register, because until then the
/// processor is still listening to the real pin and this internal state goes
/// unconsulted.
#[bitfield(u64)]
#[derive(PartialEq, Eq)]
pub struct Ignne {
    /// The current state of the internal signal. Every bit above it must be
    /// zero.
    pub asserted: bool,
    #[bits(63)]
    __: u64,
}

/// `SMM_CTL`, the write-only register driving entry into and exit from system
/// management mode by hand.
pub const SMM_CTL: u32 = 0xC001_0116;

/// A single write to [`SMM_CTL`].
///
/// The register is write-only and holds no state: each write is an action, or a
/// small ordered set of actions, performed once. That is why this type is built
/// by naming what the write does rather than by setting bits — the architecture
/// gives meaning to only nine of the thirty-two combinations those five bits
/// can express and leaves the behaviour of every other one undefined. Each
/// constructor below is one of the nine, so an undefined write cannot be
/// constructed at all.
///
/// Where a write does more than one thing the processor performs them in a
/// fixed order regardless of how the bits were set: enter, then the interrupt
/// special cycle, then dismiss, then the resume special cycle, then exit. The
/// constructors are named in that order.
///
/// Two conditions on the caller are not expressible in the type and stay its
/// responsibility. Entry and exit must be matched and must never nest, and
/// entering while already in system management mode or leaving while not in it
/// is undefined. And a write raises `#GP` once platform firmware has locked the
/// system management configuration by setting the `SmmLock` bit of the hardware
/// configuration register — after which nothing here can be used for the rest
/// of the machine's life.
///
/// The register is absent altogether on processors reporting the `NoSmmCtlMSR`
/// feature bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SmmControl(u64);

impl SmmControl {
    /// Clear the processor-internal flag recording that a system management
    /// interrupt is pending.
    #[must_use]
    pub const fn dismiss() -> Self {
        Self(DISMISS)
    }

    /// Enter system management mode: map its memory areas, record whether
    /// non-maskable interrupts were blocked, and block further non-maskable and
    /// system management interrupts.
    #[must_use]
    pub const fn enter() -> Self {
        Self(ENTER)
    }

    /// Send a system management interrupt special cycle.
    #[must_use]
    pub const fn smi_cycle() -> Self {
        Self(SMI_CYCLE)
    }

    /// Leave system management mode: unmap its memory areas, restore whatever
    /// masking of non-maskable interrupts was in force before, and re-enable
    /// system management interrupts unconditionally.
    #[must_use]
    pub const fn exit() -> Self {
        Self(EXIT)
    }

    /// Send a resume special cycle.
    #[must_use]
    pub const fn rsm_cycle() -> Self {
        Self(RSM_CYCLE)
    }

    /// Enter system management mode and announce it with a special cycle.
    #[must_use]
    pub const fn enter_with_smi_cycle() -> Self {
        Self(ENTER | SMI_CYCLE)
    }

    /// Enter system management mode and clear the pending flag — the form for
    /// an entry made in response to the very interrupt that flag records.
    #[must_use]
    pub const fn enter_and_dismiss() -> Self {
        Self(ENTER | DISMISS)
    }

    /// Enter system management mode, announce it with a special cycle, and
    /// clear the pending flag.
    #[must_use]
    pub const fn enter_with_smi_cycle_and_dismiss() -> Self {
        Self(ENTER | SMI_CYCLE | DISMISS)
    }

    /// Leave system management mode, sending a resume special cycle on the way
    /// out.
    #[must_use]
    pub const fn exit_with_rsm_cycle() -> Self {
        Self(RSM_CYCLE | EXIT)
    }

    /// The value to write to the register.
    #[must_use]
    pub const fn into_bits(self) -> u64 {
        self.0
    }
}

/// Clear the pending system management interrupt flag.
const DISMISS: u64 = 1 << 0;

/// Enter system management mode.
const ENTER: u64 = 1 << 1;

/// Send a system management interrupt special cycle.
const SMI_CYCLE: u64 = 1 << 2;

/// Leave system management mode.
const EXIT: u64 = 1 << 3;

/// Send a resume special cycle.
const RSM_CYCLE: u64 = 1 << 4;

/// `VM_HSAVE_PA`, naming the page the processor swaps host state through.
pub const VM_HSAVE_PA: u32 = 0xC001_0117;

/// The contents of [`VM_HSAVE_PA`]: where the processor writes the host's own
/// state on the way into a guest and reloads it from on the way out.
///
/// One page per processor, set up once before the first guest is ever entered
/// and never read by software afterwards — its contents are the processor's
/// business. A processor whose register is still zero cannot run a guest at
/// all: the attempt raises `#GP`, which is a puzzling way to find out that a
/// step of bring-up was skipped, so [`HostSaveAddress::new`] refuses zero
/// rather than let it get that far.
///
/// The value is a page address and this type will hold nothing else, because
/// the failure mode of getting it wrong is a fault at write time rather than
/// anything a caller could recover from: writing an address with any of its low
/// twelve bits set raises `#GP`, and so does writing one at or above the
/// largest physical address the processor implements. That second limit is
/// discovered from the processor at run time and so cannot be checked here — a
/// caller remains responsible for having allocated a page that genuinely
/// exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostSaveAddress(PhysAddr);

impl HostSaveAddress {
    /// The register's value for a host state-save area beginning at `page`.
    ///
    /// [`None`] if `page` is not page-aligned, which the register would fault
    /// on, or if it is zero, which the register accepts but which makes every
    /// later attempt to enter a guest fail instead.
    #[must_use]
    pub const fn new(page: PhysAddr) -> Option<Self> {
        let address = page.as_u64();
        if address == 0 || !address.is_multiple_of(HOST_SAVE_ALIGN) {
            return None;
        }
        Some(Self(page))
    }

    /// Where the host state-save area begins.
    #[must_use]
    pub const fn page(self) -> PhysAddr {
        self.0
    }

    /// The value to write to the register.
    #[must_use]
    pub const fn into_bits(self) -> u64 {
        self.0.as_u64()
    }
}

/// The boundary a host state-save area must begin on, which is the size of the
/// area itself.
const HOST_SAVE_ALIGN: u64 = crate::PAGE_BYTES as u64;

/// `SVM_KEY`, the write-only register a lock on this extension can be lifted
/// through.
pub const SVM_KEY: u32 = 0xC001_0118;

/// A value written to [`SVM_KEY`].
///
/// The register is write-only in the strong sense: reads return zero always, so
/// that a key set by firmware cannot be recovered by anything running later.
/// What a write does depends entirely on the state of [`VmCr::lock`] at the
/// moment it happens, and the two behaviours have nothing in common.
///
/// While the lock is clear a write *stores* the key, which is how firmware arms
/// the mechanism. While the lock is set a write *compares*: if the value
/// written equals the stored key and that key is non-zero, the lock clears; on
/// a mismatch, or against a stored key of zero, the write is ignored entirely
/// and the lock is left exactly as it was. Nothing distinguishes the two
/// outcomes at the point of the write, so software finds out whether the unlock
/// worked by reading [`VmCr::lock`] back afterwards.
///
/// The case worth naming is a stored key of zero while the lock is set: no
/// value can ever match it, and the lock is then beyond the reach of software
/// for the rest of the machine's life. Only a processor reset clears it.
///
/// This type is deliberately opaque. It converts to the bits a caller must
/// write and offers nothing else — no comparison, and a [`Debug`] that omits
/// the value — because a key that reaches a log is a key that no longer
/// protects anything.
#[derive(Clone, Copy)]
pub struct SvmKey(u64);

impl SvmKey {
    /// A key with the given value.
    #[must_use]
    pub const fn new(key: u64) -> Self {
        Self(key)
    }

    /// The value to write to the register.
    #[must_use]
    pub const fn into_bits(self) -> u64 {
        self.0
    }
}

impl Debug for SvmKey {
    /// Formats without the key, so that anything holding one stays printable
    /// without the key travelling wherever the output goes.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("SvmKey(..)")
    }
}

/// `TSC_RATIO`, scaling what a guest sees of the timestamp counter.
///
/// The address is the one oddity of this block: it sits a whole megabyte of
/// register space below its neighbours, at `C000_0104h` rather than
/// `C001_0104h`. The manual disagrees with itself here — its register summary
/// table gives the latter while the section defining the register, heading and
/// figure both, gives the former — and the section is the one that is right.
/// Operating systems have long programmed it at `C000_0104h`. The contradiction
/// is in the document, not in this constant.
pub const TSC_RATIO: u32 = 0xC000_0104;

/// The contents of [`TSC_RATIO`]: a fixed-point multiplier, eight bits of
/// integer above thirty-two bits of fraction, applied to the guest's view of
/// the timestamp counter.
///
/// The ratio is the frequency to present to a guest divided by this core's own
/// P0 frequency, and that quotient is what makes it useful: a guest migrated
/// between cores that run at different rates can be given one apparent
/// frequency across all of them, so that time does not visibly jump when it
/// moves.
///
/// It scales what the *guest* reads — from `RDTSC` and `RDTSCP`, and from the
/// timestamp counter, `MPERF` and `MPerfReadOnly` registers — and nothing else.
/// Host-mode reads of those same registers are unscaled, as are reads from
/// system management mode unless that code is itself running inside a guest.
/// The underlying counters are untouched: they advance at the rate they always
/// did, and a write of any of them by host or guest stores the value unscaled.
/// Only the view is bent.
///
/// What a guest reads combines this with the offset field of its control block:
///
/// ```text
/// guest TSC = P0 frequency * ratio * t
///           + control block TSC_OFFSET
///           + (last value written to the TSC) * ratio
/// ```
///
/// where `t` is the time since the counter was last written, or since reset if
/// it never was. The last written value is scaled too, which is the part most
/// easily forgotten when working out what a guest will see.
///
/// The register exists only on processors reporting the `TscRateMsr` feature
/// bit. Elsewhere a guest always sees the core's own rate and the control
/// block's offset is the only adjustment available.
#[bitfield(u64)]
#[derive(PartialEq, Eq)]
pub struct TscRatio {
    /// The fractional part, in units of two to the minus thirty-second.
    #[bits(32)]
    pub fraction: u32,
    /// The integer part, and the reason the ratio cannot reach two hundred and
    /// fifty-six.
    #[bits(8)]
    pub integer: u8,
    #[bits(24)]
    __: u32,
}

impl TscRatio {
    /// A ratio of exactly one, which is what the register holds coming out of
    /// reset: the guest sees the core's own P0 rate, unscaled.
    pub const ONE: Self = Self::new().with_integer(1);

    /// The ratio presenting `guest_hz` to a guest running on a core whose P0
    /// frequency is `core_hz`.
    ///
    /// The two arguments need only share a unit — hertz, kilohertz, anything —
    /// since what is computed is their quotient. It is truncated at the
    /// thirty-second fractional bit, a relative error below one part in four
    /// billion.
    ///
    /// [`None`] if `core_hz` is zero, if the quotient overflows the register's
    /// forty bits, or if it truncates to zero. That last case is not a hardware
    /// rule but a refusal to build a value that would stop the guest's clock
    /// altogether, which is never what a caller dividing one frequency by
    /// another meant.
    #[must_use]
    pub const fn from_frequencies(guest_hz: u64, core_hz: u64) -> Option<Self> {
        if core_hz == 0 {
            return None;
        }
        // Widened before the shift because a frequency in hertz needs more than
        // thirty-two bits well before any plausible clock rate, and the shift
        // would carry the top of it away.
        let scaled = (guest_hz as u128) << FRACTION_BITS;
        let ratio = scaled / core_hz as u128;
        if ratio == 0 || ratio > MAX_RATIO {
            return None;
        }
        Some(Self::from_bits(narrow(ratio)))
    }
}

/// Bits of the ratio that are fraction, and so the shift turning a plain
/// quotient into the register's fixed-point form.
const FRACTION_BITS: u32 = 32;

/// Bits of the ratio that are integer.
const INTEGER_BITS: u32 = 8;

/// The largest value the register's two fields hold between them.
const MAX_RATIO: u128 = (1 << (INTEGER_BITS + FRACTION_BITS)) - 1;

/// A ratio already bounded to the register's forty bits, in the sixty-four the
/// register is written with.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the only caller rejects anything wider than forty bits first"
)]
const fn narrow(ratio: u128) -> u64 {
    ratio as u64
}

const _: () = assert!(
    TscRatio::ONE.into_bits() == 0x0000_0001_0000_0000,
    "a ratio of one is the integer field set and nothing else",
);
const _: () = assert!(
    matches!(TscRatio::from_frequencies(100, 100), Some(TscRatio::ONE)),
    "presenting a guest the rate the core already runs at is the identity",
);
