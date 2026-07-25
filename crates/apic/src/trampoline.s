// The first instructions a processor executes when it is started.
//
// A processor answering a startup command begins in real mode with CS set to
// the page it was pointed at and IP zero, which is why this begins in 16-bit
// code and why it has to be copied somewhere below one megabyte to run. What it
// does is get from there into 64-bit mode and jump into Rust, in three stages,
// touching nothing it was not given.
//
// Nothing here is an absolute address. Every one of them — the table to load,
// the two far jump targets, the page tables, the stack, the entry point — is in
// a parameter block the processor that sent the startup command filled in
// beforehand, at an offset this file is told rather than one it knows. So the
// blob can be copied anywhere without being patched, there is no code that
// modifies itself, and no address arithmetic happens in real mode.
//
// The offsets arrive as assembler constants computed from the Rust structure's
// own field offsets, so the two cannot disagree about the layout.

.section .text.pulzar_trampoline, "ax"
.balign 16

.globl pulzar_trampoline_start
pulzar_trampoline_start:

.code16
    cli
    cld

    // The parameter block is in this same page, and in real mode the only way to
    // reach it is through a segment. CS is the page; DS becomes the page too, so
    // every reference below is a plain offset into it.
    mov ax, cs
    mov ds, ax

    // Report having got this far, before anything that could fail. Paging is
    // off, so this needs no mapping and no write permission anywhere; the
    // processor that sent the startup command is watching this word to decide
    // whether a second command is needed.
    mov word ptr ds:[{STARTED}], 1

    lgdt ds:[{TABLE_POINTER}]

    // Protected mode. The far jump that follows is what makes CS mean a
    // descriptor rather than a paragraph, and it goes through a pointer in the
    // parameter block rather than a label, because a label here would be an
    // offset from the start of the image and not from the start of the page.
    mov eax, cr0
    or eax, {CR0_PROTECTED}
    mov cr0, eax
    ljmp ds:[{PROTECTED_ENTRY}]

.code32
pulzar_trampoline_protected:
    mov ax, {DATA32_SELECTOR}
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax

    // Physical address extension, which long mode requires, and the two bits
    // that make the SSE registers usable. Those matter more than they look:
    // everything this jumps into is compiled for the ordinary 64-bit calling
    // convention, which passes and returns in SSE registers and copies memory
    // with them, and a processor out of reset has them disabled. Without this
    // the first Rust instruction is an invalid opcode.
    mov eax, cr4
    or eax, {CR4_LONG_MODE}
    mov cr4, eax

    // Clear the bit that makes SSE instructions trap and set the one that says
    // there is a coprocessor, which is the other half of the same arrangement.
    mov eax, cr0
    and eax, ~{CR0_EMULATE_COPROCESSOR}
    or eax, {CR0_MONITOR_COPROCESSOR}
    mov cr0, eax

    // Long mode enable, and no-execute enable. The second is not optional and
    // has to happen here rather than later: the page tables about to be loaded
    // have the no-execute bit set in them, and until this is written that bit is
    // reserved-must-be-zero. Loading them first would fault on the first
    // translation.
    mov ecx, {IA32_EFER}
    rdmsr
    or eax, {EFER_LONG_MODE}
    wrmsr

    // The page tables every processor shares. Thirty-two bits is all this
    // instruction writes, which is why the memory they live in is reserved below
    // four gigabytes.
    mov eax, ds:[{PAGE_TABLE_ROOT}]
    mov cr3, eax

    // Paging on. The instruction after this one is fetched through the tables
    // just loaded, at the address this code is executing from, which is why that
    // page is mapped at its own address for as long as processors are being
    // started.
    mov eax, cr0
    or eax, {CR0_PAGING}
    mov cr0, eax
    ljmp ds:[{LONG_ENTRY}]

.code64
pulzar_trampoline_long:
    // Nothing in the low half is touched again after this. The stack and the
    // entry point are both in the high half, which the shared page tables map,
    // so from the jump onwards this processor is running where every other
    // processor already is.
    mov rsp, [{STACK_TOP}]
    mov rax, [{ENTRY}]

    // The entry point never returns, so nothing is pushed for it to return to.
    // Sixteen-byte alignment is what the calling convention requires at a call
    // instruction; a jump to a function entry wants the same alignment less the
    // return address that would have been pushed.
    sub rsp, 8
    jmp rax

.globl pulzar_trampoline_end
pulzar_trampoline_end:
