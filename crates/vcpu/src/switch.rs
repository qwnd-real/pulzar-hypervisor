//! The world switch: the handful of instructions between the hypervisor running
//! and the guest running.
//!
//! This is the one path every exit passes through twice, so what is *not* here
//! matters more than what is. The architecture already moves most of a
//! processor's state across the boundary, and every register this code touched
//! as well would be moved twice:
//!
//! - `VMRUN` saves the host's `CS`, `SS`, `DS`, `ES`, `RSP`, `RAX`, `RFLAGS`,
//!   the return address, `CR0`, `CR3`, `CR4`, `EFER`, `GDTR` and `IDTR`, and
//!   `#VMEXIT` puts them back. None of them is saved here.
//! - `VMRUN` loads, and `#VMEXIT` writes back, the guest's counterparts of all
//!   of those, plus `CR2`, `DR6`, `DR7`, the interrupt shadow and the virtual
//!   interrupt state. None of them is copied here.
//! - `VMLOAD` and `VMSAVE` carry `FS`, `GS`, `TR`, `LDTR`, `KernelGsBase`,
//!   `STAR`, `LSTAR`, `CSTAR`, `SFMASK` and the three `SYSENTER` registers.
//!   None of them is touched by hand here.
//!
//! What is left is fourteen general-purpose registers, and they are all this
//! code moves.
//!
//! # Why the host's own `VMSAVE` is missing
//!
//! `#VMEXIT` restores the host's segment *selectors* but not `FS`, `GS`, `TR`
//! or `LDTR`, which come back only from a `VMLOAD`. The hypervisor needs its
//! `GS` base — one load through it is how a processor answers which processor
//! it is — and its task register, so that must happen on every exit.
//!
//! The matching `VMSAVE` does not. None of the state those instructions carry
//! changes on a pulzar processor once it has installed its descriptor tables
//! and attached: there is no `swapgs`, nothing reloads the task register, and
//! nothing writes a fast-system-call register. So the snapshot is taken once,
//! by [`Host::install`](crate::Host::install), and every exit reloads it. That
//! removes a long-latency instruction from every single exit, and the invariant
//! it rests on is stated where the snapshot is taken.
//!
//! # Why no floating-point state is saved
//!
//! `VMRUN` and `#VMEXIT` do not swap x87, SSE or AVX state, so a guest's vector
//! registers are still live in the processor while the hypervisor runs, and
//! anything that used one would corrupt the guest. Nothing can: this image is
//! built for a target whose feature string is `-mmx,-sse,+soft-float`, so no
//! instruction the compiler emits can name a vector register. That is a
//! property of the target rather than a convention to be kept, which is what
//! makes it safe to save nothing.
//!
//! # Why this is assembly written into the loop rather than a function
//!
//! The guest destroys every general-purpose register, so anything the
//! hypervisor still needs afterwards has to be in memory across the entry. The
//! only question is who puts it there. A function with a calling convention has
//! to preserve the eight registers the convention calls its caller's, whether
//! or not any of them holds something — sixteen stack operations per exit to
//! protect, in the loop this serves, three live values. Written into the loop
//! with everything the guest touches declared clobbered, the compiler spills
//! exactly what is live and nothing else.
//!
//! Two further costs travel with the call. The larger is the return: the guest
//! runs between the call and it, and overruns the processor's return-address
//! predictor while it does, so the return mispredicts on every single exit —
//! the same effect the return-address control in the control area exists to be
//! able to ask for deliberately. The smaller is that a called switch cannot say
//! which register it wants the control block's address in, so the address
//! arrives wherever the convention put it and has to be moved into the one
//! register the three virtualization instructions accept.
//!
//! Two of the fourteen cannot be handled by declaring them, because the
//! compiler keeps both for itself: the frame pointer, which may be the only way
//! back to the frame, and the base register, which it reserves for a stack that
//! has had to be realigned. Those two are saved and restored by hand, which is
//! four instructions rather than sixteen.
//!
//! # What the caller owes, and why it is not repeated here
//!
//! The global interrupt flag is cleared by [`Vcpu::run`](crate::Vcpu::run)
//! before it makes the final decision to enter, so that no non-maskable
//! interrupt can arrive between the decision and the guest. This does not clear
//! it again: the flag is already clear by the time the block below starts, and
//! clearing a cleared flag is a microcoded instruction on the one path where
//! there is nothing to spend one on. The same goes for the ordinary interrupt
//! flag in the other direction — with masking virtualized it is what the guest
//! runs with, and the loop keeps it set rather than setting it per entry.

use core::{
    arch::asm,
    mem::{align_of, offset_of, size_of},
};

use x86_64::PhysAddr;

use crate::Registers;

/// Bytes a cache line occupies on the processors this runs on.
const CACHE_LINE: usize = 64;

/// Everything the world switch has to reach once the guest has run.
///
/// The guest leaves no register holding anything, so a switch coming out of one
/// has exactly two ways to find something: the stack, and whatever it can reach
/// from the one address it pushed there. This is that address. It holds the
/// fourteen registers the switch moves, and the one further value the switch
/// needs on the way out — which is why that value lives here rather than being
/// pushed beside the pointer or fetched again by the caller.
///
/// # What its placement is for
///
/// Aligned to a cache line and no larger than two, so the switch's fourteen
/// stores reach as few lines as they can and nothing else a virtual processor
/// holds shares one of them. The snapshot address then costs no line of its
/// own: it sits in what would otherwise be the padding after the fourteenth
/// register, in the second of the two lines the stores have already brought in.
#[derive(Clone, Copy, Debug)]
#[repr(C, align(64))]
pub(crate) struct Block {
    /// The registers the architecture leaves to the hypervisor.
    pub(crate) registers: Registers,
    /// Physical address of the control block this processor's own `FS`, `GS`,
    /// `TR` and `LDTR` come back from, as [`Host::install`] left them.
    ///
    /// Held as a plain quadword because the switch loads it as one.
    ///
    /// [`Host::install`]: crate::Host::install
    snapshot: u64,
}

impl Block {
    /// A block of zeroed registers that will restore `snapshot` on the way out
    /// of every guest entered through it.
    pub(crate) const fn new(snapshot: PhysAddr) -> Self {
        Self {
            registers: Registers::zeroed(),
            snapshot: snapshot.as_u64(),
        }
    }
}

/// Runs the guest a control block describes until it exits.
///
/// Returns when the guest has stopped and the host's state is back, with the
/// global interrupt flag set — so a physical interrupt that arrived while the
/// guest was running has already been delivered to the host's own handler by
/// the time this returns.
///
/// # Safety
///
/// `guest` must be the physical address of a page-aligned, write-back control
/// block whose state passes the processor's entry checks, and `block` must
/// carry the physical address of a control block a `VMSAVE` has been performed
/// into on *this* processor. The register half of `block` is overwritten
/// entirely.
///
/// The extension must be enabled in the extended feature register and the host
/// state-save address must have been programmed, both on this processor. The
/// caller must also not have entered this control block on another processor
/// since it was last entered here, nor moved it, without having first cleared
/// its clean field: the processor identifies its cached copy by the block's
/// physical address alone.
///
/// The global interrupt flag must already be clear and the ordinary one set:
/// this clears neither and sets neither, for the reasons given in this module.
///
/// `RAX` is the pivot of the whole sequence. It is the operand `VMLOAD`,
/// `VMRUN` and `VMSAVE` all take, and it is the one general-purpose register
/// the hardware carries itself — so the guest's value goes to the control block
/// on the way out and the host's comes back, which means the `VMSAVE` after the
/// exit already has its operand without anything being reloaded.
#[expect(
    clippy::inline_always,
    reason = "being in the caller's frame is the whole point: out of line this regains the call, \
              the return that mispredicts after every guest, and the eight preserved registers"
)]
#[inline(always)]
pub(crate) unsafe fn enter(block: &mut Block, guest: PhysAddr) {
    // SAFETY: the caller guarantees every precondition the switch relies on.
    // Every register the guest destroys is declared below except the two the
    // compiler will not accept there — the frame pointer and the base
    // register — and those two are saved and restored by the block itself.
    unsafe {
        asm!(
            // The two registers that cannot be declared, and then the one value
            // that has to outlive the guest: the block to store its registers
            // into, from which the control block the host's own state comes
            // back from is reached.
            "push rbp",
            "push rbx",
            "push rcx",

            // The guest's registers. RCX is the block's address and so is
            // loaded last, once nothing else needs it.
            "mov rdx, [rcx + {RDX}]",
            "mov rbx, [rcx + {RBX}]",
            "mov rbp, [rcx + {RBP}]",
            "mov rsi, [rcx + {RSI}]",
            "mov rdi, [rcx + {RDI}]",
            "mov r8,  [rcx + {R8}]",
            "mov r9,  [rcx + {R9}]",
            "mov r10, [rcx + {R10}]",
            "mov r11, [rcx + {R11}]",
            "mov r12, [rcx + {R12}]",
            "mov r13, [rcx + {R13}]",
            "mov r14, [rcx + {R14}]",
            "mov r15, [rcx + {R15}]",
            "mov rcx, [rcx + {RCX}]",

            // Nothing may arrive between here and the guest running: an
            // interrupt taken with half the guest's state loaded would be taken
            // in a world that does not exist. The caller cleared the global
            // interrupt flag; VMRUN sets it again as it enters the guest, and
            // #VMEXIT clears it again on the way back.
            "vmload rax",
            "vmrun rax",
            // #VMEXIT resumes here, with the global interrupt flag clear, RSP
            // as VMRUN found it and RAX back to the guest control block's
            // address — which is exactly the operand this needs. The guest's
            // FS, GS, TR and LDTR are still in the processor at this point and
            // nowhere else.
            "vmsave rax",

            // The guest's registers, before the host's own state comes back
            // rather than after it. Both of the block's lines may have been
            // evicted while the guest ran, and a VMLOAD standing in front of
            // these stores would hold the reads that fetch those lines for
            // ownership behind its microcode instead of letting them run under
            // it. Its RAX is already in the control block, which is what makes
            // RAX free to address the block with.
            "pop rax",
            "mov [rax + {RCX}], rcx",
            "mov [rax + {RDX}], rdx",
            "mov [rax + {RBX}], rbx",
            "mov [rax + {RBP}], rbp",
            "mov [rax + {RSI}], rsi",
            "mov [rax + {RDI}], rdi",
            "mov [rax + {R8}],  r8",
            "mov [rax + {R9}],  r9",
            "mov [rax + {R10}], r10",
            "mov [rax + {R11}], r11",
            "mov [rax + {R12}], r12",
            "mov [rax + {R13}], r13",
            "mov [rax + {R14}], r14",
            "mov [rax + {R15}], r15",

            // The host's own FS, GS, TR, LDTR and fast-system-call registers,
            // from the snapshot taken when this processor came up, which the
            // block carries so that nothing had to keep it in a register the
            // guest would have destroyed.
            "mov rax, [rax + {SNAPSHOT}]",
            "vmload rax",

            // Everything the guest left behind is now recorded, so an interrupt
            // arriving here can be taken safely — and one that caused this exit
            // is taken here, on the host's own descriptor table, with the host's
            // task register and GS base already restored one instruction ago.
            "stgi",

            "pop rbx",
            "pop rbp",

            RCX = const offset_of!(Block, registers.rcx),
            RDX = const offset_of!(Block, registers.rdx),
            RBX = const offset_of!(Block, registers.rbx),
            RBP = const offset_of!(Block, registers.rbp),
            RSI = const offset_of!(Block, registers.rsi),
            RDI = const offset_of!(Block, registers.rdi),
            R8 = const offset_of!(Block, registers.r8),
            R9 = const offset_of!(Block, registers.r9),
            R10 = const offset_of!(Block, registers.r10),
            R11 = const offset_of!(Block, registers.r11),
            R12 = const offset_of!(Block, registers.r12),
            R13 = const offset_of!(Block, registers.r13),
            R14 = const offset_of!(Block, registers.r14),
            R15 = const offset_of!(Block, registers.r15),
            SNAPSHOT = const offset_of!(Block, snapshot),

            inout("rax") guest.as_u64() => _,
            inout("rcx") core::ptr::from_mut(block) => _,
            out("rdx") _,
            out("rsi") _,
            out("rdi") _,
            out("r8") _,
            out("r9") _,
            out("r10") _,
            out("r11") _,
            out("r12") _,
            out("r13") _,
            out("r14") _,
            out("r15") _,
        );
    }
}

/// Prevents every physical interrupt class from being delivered to the host.
///
/// # Safety
///
/// SVM must be enabled on this processor and the caller must restore GIF with
/// [`enable_global_interrupts`] if it does not proceed through `VMRUN`, which
/// sets GIF as it enters the guest.
pub(crate) unsafe fn disable_global_interrupts() {
    // SAFETY: the caller guarantees SVM is enabled and owns the obligation to
    // restore GIF if VMRUN will not do so.
    unsafe { core::arch::asm!("clgi", options(nomem, nostack, preserves_flags)) };
}

/// Restores delivery of physical interrupts after an aborted guest entry.
///
/// # Safety
///
/// SVM must be enabled on this processor, and the caller must have cleared GIF
/// on the same processor without an intervening `VMRUN` restoring it.
pub(crate) unsafe fn enable_global_interrupts() {
    // SAFETY: the caller guarantees SVM is enabled and GIF was cleared on this
    // processor for an entry which is no longer going to execute.
    unsafe { core::arch::asm!("stgi", options(nomem, nostack, preserves_flags)) };
}

const _: () = assert!(
    align_of::<Block>() == CACHE_LINE,
    "the block the switch rewrites on every exit must not share a line",
);
const _: () = assert!(
    size_of::<Block>() <= 2 * CACHE_LINE,
    "the switch's fourteen stores must not reach a third cache line",
);
const _: () = assert!(
    offset_of!(Block, snapshot) >= size_of::<Registers>(),
    "the snapshot must sit past the registers, in what would be padding after them",
);
