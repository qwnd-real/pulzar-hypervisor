//! The general-purpose registers a world switch does not carry.
//!
//! Entering and leaving a guest moves a great deal of processor state, and
//! almost none of it is here. `RAX` and `RSP` are in the control block's
//! state-save area, written by `VMRUN` and read back by `#VMEXIT`. `RIP`,
//! `RFLAGS`, the control registers, the extended feature register, two debug
//! registers, four segment registers and both descriptor-table registers travel
//! the same way. `FS`, `GS`, `TR`, `LDTR` and the fast-system-call registers
//! are carried by `VMLOAD` and `VMSAVE`.
//!
//! What is left over is exactly the fourteen below, and they are left over
//! because nothing in the architecture moves them: a guest's `RBX` is simply
//! still in `RBX` after the exit. So the world switch stores them here and
//! loads them back, and this structure is the complete list of what it has to
//! touch.
//!
//! # Nothing is stored twice
//!
//! `RAX` and `RSP` are deliberately absent even though a caller wanting "the
//! guest's registers" wants all sixteen. Copying them out of the save area into
//! a second home would cost two loads and two stores on every single exit to
//! save one match arm at a use site. [`Vcpu::gpr`](crate::Vcpu::gpr) is that
//! match arm, written once.

/// The fourteen general-purpose registers the hypervisor is responsible for.
///
/// The order is the order the architecture encodes register numbers in, minus
/// the two the hardware carries itself. That is not for arithmetic — the
/// mapping from an encoded number to a field is a match, not an index — but it
/// means a dump of this structure reads in the order a disassembler names them.
///
/// Where these sit in memory is not this type's business but
/// [`Block`](crate::switch::Block)'s, which is the thing the world switch is
/// handed and which states what the placement has to achieve.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Registers {
    /// The count register, and the first argument of the fast system call
    /// convention.
    pub rcx: u64,
    /// The data register.
    pub rdx: u64,
    /// The base register.
    pub rbx: u64,
    /// The frame pointer.
    pub rbp: u64,
    /// The source index.
    pub rsi: u64,
    /// The destination index.
    pub rdi: u64,
    /// The first of the eight registers long mode added.
    pub r8: u64,
    /// The second.
    pub r9: u64,
    /// The third.
    pub r10: u64,
    /// The fourth.
    pub r11: u64,
    /// The fifth.
    pub r12: u64,
    /// The sixth.
    pub r13: u64,
    /// The seventh.
    pub r14: u64,
    /// The eighth.
    pub r15: u64,
}

impl Registers {
    /// Every register zero, which is what a virtual processor that has never
    /// run starts with.
    #[must_use]
    pub const fn zeroed() -> Self {
        Self {
            rcx: 0,
            rdx: 0,
            rbx: 0,
            rbp: 0,
            rsi: 0,
            rdi: 0,
            r8: 0,
            r9: 0,
            r10: 0,
            r11: 0,
            r12: 0,
            r13: 0,
            r14: 0,
            r15: 0,
        }
    }

    /// The register an encoded four-bit number names, or `None` for the two the
    /// hardware carries in the control block.
    ///
    /// The numbering is the architecture's own, which is what a
    /// control-register or debug-register intercept reports the guest's
    /// operand as. `None` is not a failure — it says the register is `RAX`
    /// or `RSP`, and where to find it.
    #[must_use]
    pub const fn by_number(&self, number: u8) -> Option<u64> {
        Some(match number {
            RCX => self.rcx,
            RDX => self.rdx,
            RBX => self.rbx,
            RBP => self.rbp,
            RSI => self.rsi,
            RDI => self.rdi,
            R8 => self.r8,
            R9 => self.r9,
            R10 => self.r10,
            R11 => self.r11,
            R12 => self.r12,
            R13 => self.r13,
            R14 => self.r14,
            R15 => self.r15,
            _ => return None,
        })
    }

    /// Writes the register an encoded four-bit number names, answering whether
    /// it was one of the fourteen kept here.
    ///
    /// `false` says the number named `RAX` or `RSP` and nothing was written,
    /// which is the caller's cue to write the control block instead.
    pub const fn set_by_number(&mut self, number: u8, value: u64) -> bool {
        let slot = match number {
            RCX => &mut self.rcx,
            RDX => &mut self.rdx,
            RBX => &mut self.rbx,
            RBP => &mut self.rbp,
            RSI => &mut self.rsi,
            RDI => &mut self.rdi,
            R8 => &mut self.r8,
            R9 => &mut self.r9,
            R10 => &mut self.r10,
            R11 => &mut self.r11,
            R12 => &mut self.r12,
            R13 => &mut self.r13,
            R14 => &mut self.r14,
            R15 => &mut self.r15,
            _ => return false,
        };
        *slot = value;
        true
    }
}

/// The number the architecture encodes `RAX` as, which is the accumulator in
/// the control block's state-save area rather than a field here.
pub const RAX: u8 = 0;
/// The number the architecture encodes `RCX` as.
const RCX: u8 = 1;
/// The number the architecture encodes `RDX` as.
const RDX: u8 = 2;
/// The number the architecture encodes `RBX` as.
const RBX: u8 = 3;
/// The number the architecture encodes `RSP` as, which is the stack pointer in
/// the control block's state-save area rather than a field here.
pub const RSP: u8 = 4;
/// The number the architecture encodes `RBP` as.
const RBP: u8 = 5;
/// The number the architecture encodes `RSI` as.
const RSI: u8 = 6;
/// The number the architecture encodes `RDI` as.
const RDI: u8 = 7;
/// The number the architecture encodes `R8` as.
const R8: u8 = 8;
/// The number the architecture encodes `R9` as.
const R9: u8 = 9;
/// The number the architecture encodes `R10` as.
const R10: u8 = 10;
/// The number the architecture encodes `R11` as.
const R11: u8 = 11;
/// The number the architecture encodes `R12` as.
const R12: u8 = 12;
/// The number the architecture encodes `R13` as.
const R13: u8 = 13;
/// The number the architecture encodes `R14` as.
const R14: u8 = 14;
/// The number the architecture encodes `R15` as.
const R15: u8 = 15;

const _: () = assert!(
    core::mem::offset_of!(Registers, r15) + size_of::<u64>() == 14 * size_of::<u64>(),
    "the fourteen registers must lie end to end, since the switch addresses them by offset",
);
