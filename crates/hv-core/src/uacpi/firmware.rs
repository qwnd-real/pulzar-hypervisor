//! What firmware's own bytecode asks of the host directly.
//!
//! Two of ACPI's operators are requests rather than computations: `Breakpoint`,
//! which asks for a debugger's attention, and `Fatal`, which is firmware
//! declaring that the platform has reached a condition it cannot continue from.
//! Both arrive here, and uACPI carries on with the method either way — it
//! discards whatever this reports, because there is no answer that would change
//! what the bytecode does next.
//!
//! So the whole of the host's part is to say what happened, and the whole of
//! the judgement is how loudly. A breakpoint is firmware asking for a debugger
//! this machine does not have, which is worth a record and nothing more. A
//! `Fatal` is firmware describing its own platform as broken, in three values
//! the specification leaves entirely to the vendor — and it is the loudest
//! thing this module ever reports, because on a machine that raises one during
//! bring-up it is the single most useful line in the log.
//!
//! Neither ends the boot. The operating system that boots after this hypervisor
//! is the thing ACPI expects to act on a `Fatal`, by shutting the platform down
//! in an orderly way; a hypervisor that refused to boot instead would take that
//! decision away from it and would do so on the strength of bytecode it had
//! gone out of its way to execute.

use core::sync::atomic::{AtomicUsize, Ordering};

use log::{error, info};
use uacpi_sys::{Status, raw};

/// Records a request firmware's bytecode made.
///
/// # Safety
///
/// Called by uACPI with a pointer to one request it owns for the duration of
/// the call.
pub(super) unsafe extern "C" fn uacpi_kernel_handle_firmware_request(
    request: *mut raw::uacpi_firmware_request,
) -> raw::uacpi_status {
    if request.is_null() {
        return Status::INVALID_ARGUMENT.code();
    }
    // SAFETY: uACPI passes a pointer to one initialized request that lives for
    // this call. It is read rather than written, and nothing derived from it
    // escapes.
    let request = unsafe { &*request };
    match i32::from(request.type_) {
        raw::UACPI_FIRMWARE_REQUEST_TYPE_BREAKPOINT => {
            BREAKPOINTS.fetch_add(1, Ordering::Relaxed);
            // SAFETY: the discriminant says the request is a breakpoint, so the
            // breakpoint arm of the union is the initialized one.
            let ctx = unsafe { request.__bindgen_anon_1.breakpoint.ctx };
            info!("core: firmware's bytecode asked for a breakpoint at method context {ctx:p}");
        }
        raw::UACPI_FIRMWARE_REQUEST_TYPE_FATAL => {
            FATALS.fetch_add(1, Ordering::Relaxed);
            // SAFETY: the discriminant says the request is fatal, so the fatal arm
            // of the union is the initialized one.
            let fatal = unsafe { request.__bindgen_anon_1.fatal };
            error!(
                "core: firmware declares a fatal platform condition: type {:#x}, code {:#x}, \
                 argument {:#x}",
                fatal.type_, fatal.code, fatal.arg
            );
        }
        other => {
            // A request uACPI has grown and this has not. Reported rather than
            // ignored, because the only thing worse than not knowing what firmware
            // asked for is not knowing that it asked.
            error!("core: firmware's bytecode made request {other}, which this host cannot read");
        }
    }
    Status::OK.code()
}

/// Logs what firmware's bytecode has asked for.
pub fn describe(who: &str) {
    let breakpoints = BREAKPOINTS.load(Ordering::Relaxed);
    let fatals = FATALS.load(Ordering::Relaxed);
    if breakpoints == 0 && fatals == 0 {
        return;
    }
    info!(
        "{who}: firmware's bytecode raised {breakpoints} breakpoints and {fatals} fatal conditions"
    );
}

/// How many breakpoints firmware's bytecode has asked for.
static BREAKPOINTS: AtomicUsize = AtomicUsize::new(0);

/// How many fatal conditions firmware has declared.
static FATALS: AtomicUsize = AtomicUsize::new(0);
