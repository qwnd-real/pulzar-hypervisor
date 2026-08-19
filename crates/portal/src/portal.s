// The portal blob: copied verbatim into two guest-physical pages by
// `Portal::place`. Everything below runs under firmware's own page tables,
// as firmware itself, so it must not reference any host virtual address —
// only the guest-physical parameter page reached via `rip`-relative
// addressing, and the pointers that page carries.
.section .text.pulzar_portal, "ax"
.balign 4096

// Entry point. The loader resumes the guest here instead of where firmware
// was captured, with rip pointing at the first byte of this page.
.globl pulzar_portal_start
pulzar_portal_start:
    // r11 -> the parameter page, one page after this one. Every later use of
    // {SYSTEM_TABLE}, {GUEST_IMAGE_HANDLE}, and {ORIGINAL_EXIT_BOOT_SERVICES}
    // is relative to this pointer.
    lea r11, [rip + pulzar_portal_data]

    // rbx -> firmware's boot-services table, read from the live system table
    // rather than baked into the blob, since the portal is position-
    // independent and carries no addresses of its own.
    mov rbx, [r11 + {SYSTEM_TABLE}]
    mov rbx, [rbx + {SYSTEM_BOOT_SERVICES}]

    // Patch the table: replace ExitBootServices with the wrapper below, so
    // the host is told the instant firmware's services stop existing. The
    // guest's own copy of the pointer is what gets overwritten here — the
    // boot manager will call this wrapper without knowing it changed.
    lea rax, [rip + pulzar_portal_exit_boot_services]
    mov [rbx + {BOOT_EXIT_BOOT_SERVICES}], rax

    // UEFI tables carry a CRC32 of their own bytes; having just modified one
    // field, that checksum is now wrong and must be recomputed before
    // anything else is allowed to read the table, or firmware and the boot
    // manager may reject it as corrupt. The field being hashed is zeroed
    // first, per the algorithm boot services use to derive it.
    mov dword ptr [rbx + {HEADER_CRC}], 0

    // CalculateCrc32(this = rbx, DataSize = header.size, &Crc32) via the
    // System V x86-64 ABI: rcx/rdx/r8/r9 are the first three arguments, and
    // 32 bytes of shadow space are reserved below rsp for the callee even
    // though UEFI is Microsoft-ABI, matching the convention `uefi_raw`
    // assumes throughout this crate.
    sub rsp, 48
    mov rcx, rbx
    mov edx, dword ptr [rbx + {HEADER_SIZE}]
    lea r8, [rsp + 32]
    mov rax, [rbx + {BOOT_CALCULATE_CRC32}]
    call rax
    mov r9d, dword ptr [rsp + 32]
    add rsp, 48

    // A failing CalculateCrc32 means the table cannot be trusted enough to
    // hand back to firmware at all; there is no safe way to continue, so the
    // guest is parked rather than risk starting the boot manager against a
    // table firmware will refuse, or worse, silently misread.
    test rax, rax
    jnz pulzar_portal_halt
    mov dword ptr [rbx + {HEADER_CRC}], r9d

    // StartImage(ImageHandle, &ExitDataSize, &ExitData). The guest image
    // handle comes from the parameter page rather than a register firmware
    // left it in, since the portal does not rely on any state from before
    // the loader captured it. ExitDataSize/ExitData are zeroed and discarded:
    // the host only cares whether control returns here at all, not why.
    lea r11, [rip + pulzar_portal_data]
    sub rsp, 48
    mov rcx, [r11 + {GUEST_IMAGE_HANDLE}]
    mov qword ptr [rsp + 32], 0
    mov qword ptr [rsp + 40], 0
    lea rdx, [rsp + 32]
    lea r8, [rsp + 40]
    mov rax, [rbx + {BOOT_START_IMAGE}]
    call rax
    add rsp, 48

    // StartImage is not supposed to return: a successful boot manager takes
    // the machine and never comes back. If execution reaches this point,
    // that transfer failed, and the host needs to know rather than watch the
    // guest run off into whatever StartImage left behind. rax still holds
    // StartImage's own EFI_STATUS, untouched since the call, so the host
    // gets it for free by reading rax alongside this notification.
    mov rdx, {START_RETURNED}
    vmmcall

// Reached only when the guest cannot be handed back safely. There is
// nothing left to attempt, so the processor is stopped for good; `hlt` is
// re-issued in a loop in case of a spurious wake.
.globl pulzar_portal_halt
pulzar_portal_halt:
    cli
1:
    hlt
    jmp 1b

// Installed in place of firmware's real ExitBootServices. Firmware and the
// boot manager see an ordinary function at this address and call it with
// the ordinary ExitBootServices arguments already in rcx/rdx; neither is
// touched here; both simply flow through to the original.
.globl pulzar_portal_exit_boot_services
pulzar_portal_exit_boot_services:
    // r11 is caller-saved by convention but is still in active use by the
    // caller's own code around this call site, so it is preserved rather
    // than assumed free, then reloaded with the parameter-page pointer this
    // wrapper needs.
    push r11
    lea r11, [rip + pulzar_portal_data]

    // Call through to firmware's real ExitBootServices, exactly as if the
    // wrapper were not here, aside from the shadow space this call site
    // itself owes its callee under the same convention used above.
    sub rsp, 32
    call qword ptr [r11 + {ORIGINAL_EXIT_BOOT_SERVICES}]
    add rsp, 32

    // A nonzero EFI_STATUS means boot services did not actually exit — the
    // memory map the guest passed was stale and firmware is asking for it to
    // be rebuilt and retried. Nothing observable to the host has happened
    // yet, so this call is silently passed back to let the guest retry.
    test rax, rax
    jnz 2f

    // Success: this is the one and only moment firmware's services are
    // guaranteed gone for good, since ExitBootServices cannot meaningfully
    // fail after returning success. The host learns this here rather than by
    // guessing at it from the guest's later behavior.
    mov rdx, {EXIT_SUCCEEDED}
    vmmcall
2:
    pop r11
    ret

// The parameter page: the only guest-physical addresses this blob knows,
// filled in by `Portal::fill` after the blob itself is copied. Nothing here
// is read before `pulzar_portal_start` runs, so the initial zeros below are
// never observed — they exist only so the layout matches `Parameters`
// exactly, with no gap the blob could read uninitialized.
.balign 4096
.globl pulzar_portal_data
pulzar_portal_data:
    .quad 0   // system_table
    .quad 0   // guest_image_handle
    .quad 0   // original_exit_boot_services

// Marks the end of the reservation `Portal::place` copies and bounds-checks
// against; nothing is emitted here, only the symbol `blob()` measures against
// `pulzar_portal_start` to know how many bytes to copy.
.globl pulzar_portal_end
pulzar_portal_end: