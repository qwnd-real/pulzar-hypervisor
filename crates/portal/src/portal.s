.section .text.pulzar_portal, "ax"
.balign 4096

.globl pulzar_portal_start
pulzar_portal_start:
    lea r11, [rip + pulzar_portal_data]
    mov rbx, [r11 + {SYSTEM_TABLE}]
    mov rbx, [rbx + {SYSTEM_BOOT_SERVICES}]

    lea rax, [rip + pulzar_portal_exit_boot_services]
    mov [rbx + {BOOT_EXIT_BOOT_SERVICES}], rax

    mov dword ptr [rbx + {HEADER_CRC}], 0
    sub rsp, 48
    mov rcx, rbx
    mov edx, dword ptr [rbx + {HEADER_SIZE}]
    lea r8, [rsp + 32]
    mov rax, [rbx + {BOOT_CALCULATE_CRC32}]
    call rax
    mov r9d, dword ptr [rsp + 32]
    add rsp, 48
    test rax, rax
    jnz pulzar_portal_halt
    mov dword ptr [rbx + {HEADER_CRC}], r9d

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
    mov rdx, {START_RETURNED}
    vmmcall

.globl pulzar_portal_halt
pulzar_portal_halt:
    cli
1:
    hlt
    jmp 1b

.globl pulzar_portal_exit_boot_services
pulzar_portal_exit_boot_services:
    push r11
    lea r11, [rip + pulzar_portal_data]
    sub rsp, 32
    call qword ptr [r11 + {ORIGINAL_EXIT_BOOT_SERVICES}]
    add rsp, 32
    test rax, rax
    jnz 2f
    mov rdx, {EXIT_SUCCEEDED}
    vmmcall
2:
    pop r11
    ret

.balign 4096
.globl pulzar_portal_data
pulzar_portal_data:
    .quad 0
    .quad 0
    .quad 0

.globl pulzar_portal_end
pulzar_portal_end:
