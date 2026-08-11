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
//
// # Saying how far it got
//
// Each stage records itself in the parameter block before doing anything that
// could fail, and only while paging is off — the page is mapped read-only where
// it executes from, so once paging and supervisor write protection are on there
// is nowhere here it may write. There is no interrupt descriptor table until Rust
// builds one either, so a fault anywhere in here resets the processor with
// nothing to say why; the stage number is the whole of what the processor that
// started it can find out afterwards.

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
    mov dword ptr ds:[{STAGE}], {STAGE_REAL}

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
    // anything one jump from here — and because it is the only way to *write* to
    // the block: a code segment is execute-read and can never be written, so the
    // stage below goes through the flat data segment at the linear address EBX
    // now holds.
    mov ebx, cs:[{PAGE_BASE}]
    mov dword ptr [ebx + {STAGE}], {STAGE_PROTECTED_MODE}

    // Physical address extension, which long mode requires, and the two bits
    // that make the vector registers usable and their exceptions reportable.
    // Nothing this jumps into is compiled to use those registers — the target
    // this image is built for has them turned off — but the hypervisor reaches
    // them by hand to move a guest's, and a processor comes out of reset with
    // the instructions that do so faulting.
    mov eax, cr4
    or eax, {CR4_LONG_MODE}
    mov cr4, eax

    // The whole of this processor's control-register policy, in one write rather
    // than accumulated onto whatever it came out of `INIT` with. `INIT` leaves
    // the two cache bits alone, so a processor firmware had running with its
    // caches disabled would otherwise carry that into the hypervisor and be a
    // hundred times slower than every other one; and the coprocessor bits decide
    // whether the vector instructions above fault.
    mov eax, cr0
    and eax, {CR0_CLEAR}
    or eax, {CR0_SET}
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
    // just loaded, at the address this code is executing from — which is why this
    // page is mapped at its own address for as long as processors are being
    // started.
    //
    // Supervisor write protection is deliberately *not* set here, one instruction
    // before the far jump below. That jump loads a descriptor out of the table in
    // this page, and loading a descriptor is how the processor sets its accessed
    // bit — a write, to a page mapped read-only where this code executes from.
    // The descriptors are built with that bit already set so the write is
    // unnecessary, but the architecture does not promise a processor skips it.
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

    // Supervisor write protection, now that the last descriptor has been loaded
    // out of the table in the low page and nothing here writes to that page
    // again. Without this, privileged code ignores the read-only bit in a page
    // table entry, so every mapping this hypervisor made read-only — its own
    // code, its own tables — would be writable on this processor and on no other.
    mov rax, cr0
    or rax, {CR0_WRITE_PROTECT}
    mov cr0, rax

    // The last two reads of the low half. The stack and the entry point are both
    // in the high half, which the shared page tables map, so from here on this
    // processor is running where every other processor already is.
    //
    // Nothing is written back. This page is mapped read-only where it executes
    // from, and supervisor write protection is on by now, so a store here would
    // fault with no handler to take it — which is also why the stage word is only
    // ever written with paging off. That the block has been read is something the
    // processor that started this one works out instead: it waits for this
    // processor to publish itself, which is a long way past here.
    mov rsp, [rbx + {STACK_TOP}]
    mov rax, [rbx + {ENTRY}]

    // The frame the entry point is entitled to. This target's calling convention
    // makes the caller reserve four argument slots whether or not the callee
    // takes four arguments, and they sit above the return address a call would
    // have pushed — so a prologue that homes a register writes into the thirty-two
    // bytes at `[rsp+8]`, which have to be stack and not the guard page above it.
    // The return slot itself is filled with zero: the entry point never returns,
    // and a return that happens anyway should fault on the first instruction
    // rather than continue into whatever the stack held.
    sub rsp, {ENTRY_FRAME}
    mov qword ptr [rsp], 0
    jmp rax

.globl pulzar_trampoline_end
pulzar_trampoline_end:

// How large the blob is and where each of its stages begins, as data rather than
// as a difference between two addresses Rust would have to subtract for itself.
// Outside the copied range, because it describes it.
.section .rodata.pulzar_trampoline, "a"
.balign 8
.globl pulzar_trampoline_extent
pulzar_trampoline_extent:
    .quad pulzar_trampoline_end - pulzar_trampoline_start
    .quad pulzar_trampoline_protected - pulzar_trampoline_start
    .quad pulzar_trampoline_long - pulzar_trampoline_start
