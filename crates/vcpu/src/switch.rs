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
//! # Why a naked function
//!
//! An inline `asm!` block would have to declare every general-purpose register
//! clobbered, and the compiler would spill and reload around it — the same
//! work, less predictably, and with no say over which register holds the
//! control block's address at the moment `VMRUN` executes. A naked function is
//! the whole sequence and nothing else.

use core::{arch::naked_asm, mem::offset_of};

use x86_64::PhysAddr;

use crate::Registers;

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
/// block whose state passes the processor's entry checks, `host` the physical
/// address of a control block a `VMSAVE` has been performed into on *this*
/// processor, and `registers` must point at a block this call may overwrite
/// entirely.
///
/// The extension must be enabled in the extended feature register and the host
/// state-save address must have been programmed, both on this processor. The
/// caller must also not have entered this control block on another processor
/// since it was last entered here, nor moved it, without having first cleared
/// its clean field: the processor identifies its cached copy by the block's
/// physical address alone.
pub(crate) unsafe fn enter(registers: &mut Registers, guest: PhysAddr, host: PhysAddr) {
    // SAFETY: the caller guarantees every precondition the switch relies on,
    // and the borrow is what makes the register block's exclusivity the
    // compiler's business rather than the contract's.
    unsafe {
        switch(
            core::ptr::from_mut(registers),
            guest.as_u64(),
            host.as_u64(),
        );
    }
}

/// The switch.
///
/// This target's C ABI is the Windows one, so the three arguments arrive in
/// `RCX`, `RDX` and `R8`, and `RBX`, `RBP`, `RDI`, `RSI` and `R12` through
/// `R15` belong to the caller. Eight of those are pushed; the six volatile
/// registers are not, because the guest is welcome to whatever they held.
///
/// `RAX` is the pivot of the whole sequence. It is the operand `VMLOAD`,
/// `VMRUN` and `VMSAVE` all take, and it is the one general-purpose register
/// the hardware carries itself — so the guest's value goes to the control block
/// on the way out and the host's comes back, which means the `VMSAVE` after the
/// exit already has its operand without anything being reloaded.
///
/// # Safety
///
/// As [`enter`], which is the only caller and exists to state the same
/// preconditions in terms of the types they are really about.
#[unsafe(naked)]
unsafe extern "C" fn switch(registers: *mut Registers, guest: u64, host: u64) {
    naked_asm!(
        // The caller's registers, and then the two values that have to outlive
        // the guest: the block to store its registers into, and the control
        // block the host's own state comes back from.
        "push rbx",
        "push rbp",
        "push rdi",
        "push rsi",
        "push r12",
        "push r13",
        "push r14",
        "push r15",
        "push rcx",
        "push r8",

        // The operand of all three virtualization instructions below.
        "mov rax, rdx",

        // The guest's registers. RCX is the block's address and so is loaded
        // last, once nothing else needs it.
        "mov rbx, [rcx + {RBX}]",
        "mov rdx, [rcx + {RDX}]",
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

        // Nothing may arrive between here and the guest running: an interrupt
        // taken with half the guest's state loaded would be taken in a world
        // that does not exist. VMRUN sets the flag again as it enters the
        // guest, and #VMEXIT clears it again on the way back.
        "clgi",
        "vmload rax",
        "vmrun rax",
        // #VMEXIT resumes here, with the global interrupt flag clear, RSP as
        // VMRUN found it and RAX back to the guest control block's address —
        // which is exactly the operand this needs. The guest's FS, GS, TR and
        // LDTR are still in the processor at this point and nowhere else.
        "vmsave rax",

        // The host's own FS, GS, TR, LDTR and fast-system-call registers, from
        // the snapshot taken when this processor came up.
        "pop rax",
        "vmload rax",

        // The guest's registers. Its RAX is already in the control block, which
        // is what makes RAX free to address the block with.
        "pop rax",
        "mov [rax + {RBX}], rbx",
        "mov [rax + {RCX}], rcx",
        "mov [rax + {RDX}], rdx",
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

        // Everything the guest left behind is now recorded, so an interrupt
        // arriving here can be taken safely — and one that caused this exit is
        // taken here, on the host's own descriptor table, with the host's task
        // register and GS base already restored one instruction ago.
        "stgi",

        "pop r15",
        "pop r14",
        "pop r13",
        "pop r12",
        "pop rsi",
        "pop rdi",
        "pop rbp",
        "pop rbx",
        "ret",

        RCX = const offset_of!(Registers, rcx),
        RDX = const offset_of!(Registers, rdx),
        RBX = const offset_of!(Registers, rbx),
        RBP = const offset_of!(Registers, rbp),
        RSI = const offset_of!(Registers, rsi),
        RDI = const offset_of!(Registers, rdi),
        R8 = const offset_of!(Registers, r8),
        R9 = const offset_of!(Registers, r9),
        R10 = const offset_of!(Registers, r10),
        R11 = const offset_of!(Registers, r11),
        R12 = const offset_of!(Registers, r12),
        R13 = const offset_of!(Registers, r13),
        R14 = const offset_of!(Registers, r14),
        R15 = const offset_of!(Registers, r15),
    )
}
