/*
 * The widths the Rust side of this boundary assumes, asserted where C decides
 * them.
 *
 * Every one of these is a value that crosses the boundary in a register or a
 * struct field, and every one of them is a width that some combination of
 * target triple and data model could change without any diagnostic: the
 * declarations would still generate, the library would still link, and a value
 * would be truncated or read out of the wrong half of a register at run time.
 *
 * The file compiles to nothing. Its whole output is the build failing when one
 * of these stops being true, which is the only way the mismatch is ever cheap to
 * find. crates/uacpi-sys/src/lib.rs states the same widths in Rust's terms.
 */

#include <uacpi/types.h>

#define UACPI_ASSERT_WIDTH(type, bytes) \
    _Static_assert(sizeof(type) == (bytes), #type " is not " #bytes " bytes wide")

UACPI_ASSERT_WIDTH(uacpi_u8, 1);
UACPI_ASSERT_WIDTH(uacpi_u16, 2);
UACPI_ASSERT_WIDTH(uacpi_u32, 4);
UACPI_ASSERT_WIDTH(uacpi_u64, 8);

UACPI_ASSERT_WIDTH(uacpi_char, 1);
UACPI_ASSERT_WIDTH(uacpi_bool, 1);

/* Enumerations, which C is free to size as it likes and clang sizes as an int. */
UACPI_ASSERT_WIDTH(uacpi_status, 4);
UACPI_ASSERT_WIDTH(uacpi_log_level, 4);

/* Addresses, of the three kinds uACPI tells apart. */
UACPI_ASSERT_WIDTH(uacpi_phys_addr, 8);
UACPI_ASSERT_WIDTH(uacpi_io_addr, 8);
UACPI_ASSERT_WIDTH(uacpi_virt_addr, 8);
UACPI_ASSERT_WIDTH(uacpi_size, 8);
UACPI_ASSERT_WIDTH(uacpi_handle, 8);

/*
 * The two the override header exists for, and the one it names a thread with.
 * A four-byte flags type here means the override was not picked up, and that a
 * spinlock's saved interrupt state is being returned in half the width the
 * caller reads back.
 */
UACPI_ASSERT_WIDTH(uacpi_cpu_flags, 8);
UACPI_ASSERT_WIDTH(uacpi_interrupt_state, 8);
UACPI_ASSERT_WIDTH(uacpi_thread_id, 8);
