//! The interrupt uACPI would service, and why this hypervisor does not give it
//! one.
//!
//! ACPI's event subsystem runs on the system control interrupt: firmware routes
//! a level-triggered, shareable line to it, the host installs a handler, and
//! general purpose events arrive there and are dispatched into bytecode. uACPI
//! asks its host to install that handler while the namespace is being brought
//! up, and pulzar answers that it will not.
//!
//! # Why
//!
//! Because the interrupt is not this hypervisor's to take. Pulzar passes the
//! machine through: what runs after bring-up is the firmware boot manager, and
//! then an operating system, and that operating system enters ACPI mode,
//! enables the general purpose events it cares about, and services the control
//! interrupt itself. The line is shared and level-triggered, so a second owner
//! is not a second observer — it is a handler that can acknowledge an event the
//! real owner has not seen yet, and turn a device's notification into a lost
//! one. There is no arrangement in which both service it and both are correct.
//!
//! So the platform's hardware state is left exactly as firmware set it, which
//! is also why [`super::initialize`] declines to enter ACPI mode. What pulzar
//! wants out of uACPI is the description of a machine — its tables, its
//! namespace, the values its bytecode computes — and none of that needs an
//! interrupt.
//!
//! # Why nothing asks
//!
//! Refusing here is the backstop rather than the mechanism. uACPI is compiled
//! without its event subsystem, so the code that would install a handler is not
//! in the image at all and bringing the namespace up never reaches for one.
//!
//! It has to be that way round. A refusal on its own is not a decision that
//! composes: loading the namespace initializes the event subsystem as one of
//! its steps, and a host that declines the interrupt there fails the whole step
//! — so the namespace a machine describes would depend on a refusal several
//! layers below the call that wanted it. Compiling the subsystem out states the
//! same judgement where it cannot fail anything.
//!
//! What is left here is a field uACPI's host interface still has, and a record
//! if it is ever reached. Both halves of the decision — a host that declines,
//! and a uACPI built without the subsystem that would ask — would have to
//! change together, and a line naming the interrupt is what makes the first
//! visible if the second ever does.
//!
//! # What it would take to answer instead
//!
//! Somewhere to route the line. A handler needs a vector, and a system
//! interrupt reaches a vector through an interrupt controller's redirection
//! entry — and the controller a system interrupt arrives through is one nothing
//! in this workspace drives. That is a subsystem, not a callback, and it would
//! exist to serve a decision this hypervisor has taken the other way.

use core::sync::atomic::{AtomicUsize, Ordering};

use log::{info, warn};
use uacpi_sys::{Status, raw};

/// Logs whether uACPI ever asked for the interrupt it is not given.
pub fn describe(who: &str) {
    let asked = ASKED.load(Ordering::Relaxed);
    if asked == 0 {
        info!("{who}: uacpi asked for no system interrupt, as bring-up intends");
    } else {
        warn!(
            "{who}: uacpi asked for a system interrupt {asked} times and was refused each time; \
             it was built without the event subsystem that would ask"
        );
    }
}

/// Declines to route a system interrupt to uACPI.
///
/// See this module: the interrupt belongs to whatever boots after this
/// hypervisor, and nothing pulzar asks of uACPI needs one.
///
/// # Safety
///
/// Called by uACPI with storage for one handle, which is left untouched because
/// no handler is installed to name.
pub(super) unsafe extern "C" fn uacpi_kernel_install_interrupt_handler(
    irq: raw::uacpi_u32,
    _handler: raw::uacpi_interrupt_handler,
    _ctx: raw::uacpi_handle,
    _out: *mut raw::uacpi_handle,
) -> raw::uacpi_status {
    ASKED.fetch_add(1, Ordering::Relaxed);
    warn!(
        "core: uacpi asked to service interrupt {irq}; the guest owns the acpi event subsystem, \
         so it is not serviced here"
    );
    Status::DENIED.code()
}

/// Reports that there was no handler to remove, because none was installed.
///
/// # Safety
///
/// Called by uACPI with a handle its own installation returned — which, since
/// installation is always refused, it can never hold.
pub(super) unsafe extern "C" fn uacpi_kernel_uninstall_interrupt_handler(
    _handler: raw::uacpi_interrupt_handler,
    _handle: raw::uacpi_handle,
) -> raw::uacpi_status {
    Status::NOT_FOUND.code()
}

/// How many times uACPI has asked for a system interrupt.
static ASKED: AtomicUsize = AtomicUsize::new(0);
