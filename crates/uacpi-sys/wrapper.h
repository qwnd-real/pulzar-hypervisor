/*
 * Every public uACPI header, in one translation unit for the binding generator
 * to read. The generator descends into whatever these include, so the internal
 * headers arrive too and are filtered out on the Rust side rather than here:
 * a header left out of this list is a binding silently missing, while a type
 * pulled in by another header is at worst an unused declaration.
 */
#include <uacpi/acpi.h>
#include <uacpi/context.h>
#include <uacpi/event.h>
#include <uacpi/helpers.h>
#include <uacpi/io.h>
#include <uacpi/kernel_api.h>
#include <uacpi/log.h>
#include <uacpi/namespace.h>
#include <uacpi/notify.h>
#include <uacpi/opregion.h>
#include <uacpi/osi.h>
#include <uacpi/registers.h>
#include <uacpi/resources.h>
#include <uacpi/sleep.h>
#include <uacpi/status.h>
#include <uacpi/tables.h>
#include <uacpi/types.h>
#include <uacpi/uacpi.h>
#include <uacpi/utilities.h>
