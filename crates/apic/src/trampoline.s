// The first instructions a processor executes when it is started.
//
// A processor answering a startup command begins in real mode with CS set to the
// page it was pointed at and IP zero, which is why this begins in 16-bit code
// and why it has to be copied below one megabyte to run at all. What it does is
// get from there into 64-bit mode and jump into Rust, in three stages, touching
// nothing it was not given.
//
// Nothing here is an absolute address, and nothing here modifies itself. Every
// address it needs — the table to load, where each stage begins, the page
// tables, the stack, the entry point — is in a parameter block that the
// processor which sent the startup command filled in beforehand, at an offset
// this file is told rather than one it knows. The offsets arrive as assembler
// constants computed from the Rust structure's own field offsets, so the two
// cannot disagree about the layout.
//
// Both mode transitions therefore go through a far pointer in memory rather than
// an immediate. That is not only because an immediate would have to be an
// address this code cannot know: it is also the one form the assembler will take
// a computed value in.

.section .text.pulzar_trampoline, "ax"
.balign 16

.globl pulzar_trampoline_start
pulzar_trampoline_start:

.code16
    cli
    cld

    // The parameter block is in this same page, and in real mode a segment is
    // the only way to reach it. CS is the page; DS becomes the page too, so
    // every reference below is a plain offset into it.
    mov ax, cs
    mov ds, ax

    // Report having got this far, before anything that could fail. Paging is
    // off, so this needs no mapping and no write permission anywhere; the
    // processor that sent the startup command is watching this word to decide
    // whether to send a second one.
    mov dword ptr ds:[{STARTED}], 1

    // The 24-bit base a 16-bit `lgdt` loads is enough for a table inside this
    // page, and this page is below one megabyte by construction.
    lgdt ds:[{TABLE_POINTER}]

    mov eax, cr0
    or eax, {CR0_PROTECTED}
    mov cr0, eax

    // Into the 32-bit code segment, whose base is this page — so the offset in
    // the pointer is an offset into the blob, and the code below can go on
    // reaching the parameter block through CS.
    jmp fword ptr ds:[{PROTECTED_ENTRY}]

.code32
.globl pulzar_trampoline_protected
pulzar_trampoline_protected:
    mov ax, {DATA32_SELECTOR}
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax

    // The data segments are flat, so from here the parameter block is reached
    // through CS, which is the one segment still based at the page. The page's
    // own address is taken out of it now, because the segment bases stop meaning
    // anything one jump from here.
    mov ebx, cs:[{PAGE_BASE}]

    // Physical address extension, which long mode requires, and the two bits
    // that make the SSE registers usable. Those matter more than they look:
    // everything this jumps into is compiled for the ordinary 64-bit calling
    // convention, which passes and returns in SSE registers and copies memory
    // with them, and a processor out of reset has them turned off. Without this
    // the first Rust instruction is an invalid opcode.
    mov eax, cr4
    or eax, {CR4_LONG_MODE}
    mov cr4, eax

    // The other half of that arrangement: stop SSE instructions trapping, and
    // say there is a coprocessor to trap for.
    mov eax, cr0
    and eax, {CR0_COPROCESSOR_CLEAR}
    or eax, {CR0_COPROCESSOR_SET}
    mov cr0, eax

    // Long mode enable, and no-execute enable. The second is not optional and
    // has to happen here rather than afterwards: the page tables about to be
    // loaded have the no-execute bit set in them, and until this is written that
    // bit is reserved-must-be-zero. Loading them first would fault on the first
    // translation the processor made.
    mov ecx, {IA32_EFER}
    rdmsr
    or eax, {EFER_LONG_MODE}
    wrmsr

    // The page tables every processor shares. Thirty-two bits is all this
    // instruction writes, which is why the memory holding them is reserved below
    // four gigabytes.
    mov eax, cs:[{PAGE_TABLE_ROOT}]
    mov cr3, eax

    // Paging on. The instruction after this one is fetched through the tables
    // just loaded, at the address this code is executing from — which is why
    // this page is mapped at its own address for as long as processors are being
    // started.
    mov eax, cr0
    or eax, {CR0_PAGING}
    mov cr0, eax

    // Into 64-bit mode. A 64-bit code segment ignores its base, so the offset in
    // this pointer is a linear address rather than an offset into the blob.
    jmp fword ptr cs:[{LONG_ENTRY}]

.code64
.globl pulzar_trampoline_long
pulzar_trampoline_long:
    // A write to a 32-bit register outside 64-bit mode says nothing about the
    // upper half, so the page base is re-established as a 64-bit value before it
    // is used as one.
    mov ebx, ebx

    // Nothing in the low half is touched again after this. The stack and the
    // entry point are both in the high half, which the shared page tables map,
    // so from here on this processor is running where every other processor
    // already is.
    mov rsp, [rbx + {STACK_TOP}]
    mov rax, [rbx + {ENTRY}]

    // The entry point never returns, so nothing is pushed for it to return to.
    // The calling convention wants the stack 16-byte aligned at a call, which
    // means eight off it at the instruction a call would have landed on.
    sub rsp, 8
    jmp rax

.globl pulzar_trampoline_end
pulzar_trampoline_end:
