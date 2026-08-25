/*
 * uACPI's architecture helpers, replaced so that the fundamental widths on both
 * sides of the boundary are fixed rather than inferred.
 *
 * uACPI's own version of this header types uacpi_cpu_flags and
 * uacpi_interrupt_state as `unsigned long`, whose width is not a property of the
 * machine but of the data model: four bytes under a Windows one, eight under a
 * Unix one. The Rust target this is compiled for, x86_64-unknown-uefi, is the
 * awkward case where the two disagree — the C compiler has to be pointed at a
 * PE/COFF target, which makes `unsigned long` four bytes, while Rust's own
 * `c_ulong` for that target is eight. A spinlock's saved flags would then be
 * returned in one width and read back in another.
 *
 * So both are spelled as an explicit 64-bit type here, which no data model can
 * disagree about. The widths this assumes are asserted in widths.c, on this side
 * of the boundary, and mirrored on the Rust side.
 *
 * Everything else is uACPI's default, reproduced because replacing this header
 * replaces all of it.
 */
#pragma once

#include <uacpi/platform/atomic.h>
#include <uacpi/platform/types.h>

/*
 * The architecture requires the caches be written back before the platform is
 * put into a state that may lose their contents. This code always runs at the
 * most privileged level, so the instruction is always available.
 */
#define UACPI_ARCH_FLUSH_CPU_CACHE() __asm__ volatile("wbinvd" ::: "memory")

typedef uacpi_u64 uacpi_cpu_flags;
typedef uacpi_u64 uacpi_interrupt_state;

typedef void *uacpi_thread_id;

#define UACPI_ATOMIC_LOAD_THREAD_ID(ptr) \
    ((uacpi_thread_id)uacpi_atomic_load_ptr(ptr))

#define UACPI_ATOMIC_STORE_THREAD_ID(ptr, value) \
    uacpi_atomic_store_ptr(ptr, value)

/*
 * The value the host promises never to name a thread with, so that uACPI can
 * use it to mean that a mutex is unowned.
 */
#define UACPI_THREAD_ID_NONE ((uacpi_thread_id)-1)
