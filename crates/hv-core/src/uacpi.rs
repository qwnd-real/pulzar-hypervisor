//! What uACPI needs from the machine underneath it.
//!
//! uACPI is the ACPI implementation this hypervisor reads firmware's tables
//! through and evaluates its bytecode with, and it is written to be hosted: it
//! knows how ACPI works and nothing about the machine it is running on. Every
//! question it cannot answer for itself — reach this physical address, allocate
//! this many bytes, how long is a microsecond, take this lock — it asks through
//! a global C symbol the host defines. This module is every one of those
//! answers, and it lives here because `hv-core` is the only thing in the
//! workspace that has them all: the address space, the heap, the log, the
//! clock, the interrupt controllers and the devices.
//!
//! Nothing outside this module calls into it. What the rest of the hypervisor
//! wants is the `acpi` crate's parsed tables, and what this module does is make
//! those readable.
//!
//! # The two stages, and why there are two
//!
//! uACPI can bring itself up in two steps, and pulzar needs both because the
//! things it wants out of ACPI are the same things a full uACPI would need to
//! have already.
//!
//! [`early_tables`] sets up table access alone. At that point uACPI uses
//! exactly three of the answers below — the root pointer, mapping physical
//! memory, and the log — and no others, which is what makes it callable while
//! the heap is the only other thing standing. That is where the `acpi` crate
//! reads the machine's description from, and it has to be there: the clock is
//! calibrated against a timer the tables describe, the processors are counted
//! out of them, and the interrupt controllers and configuration space apertures
//! are found in them. None of that can wait for a subsystem that needs a clock.
//!
//! [`initialize`] is the rest: the namespace is built, every definition block
//! is executed, and the objects in it are initialized. It comes last in
//! bring-up, after the clock, the processors, the controllers and the devices,
//! because interpreting bytecode is what needs all of those.
//!
//! # What pulzar deliberately does not let uACPI do
//!
//! Own ACPI's hardware. This hypervisor reads the platform; it does not take
//! the platform over. What boots after it is firmware, and then an operating
//! system, and that operating system enters ACPI mode, enables the general
//! purpose events it wants, takes the global lock and services the system
//! control interrupt — in a way a pass-through hypervisor must not compete
//! with.
//!
//! So uACPI is compiled without the subsystems that would do any of that: the
//! event subsystem, the global lock and the fixed-event machinery are not in
//! the image. What is left is the interpreter and the namespace, which is
//! exactly what pulzar wants — the description of a machine, and the values its
//! bytecode computes. [`interrupts`] carries the whole of that reasoning, and
//! why the decision is a compile-time one rather than a callback that says no.
//!
//! ACPI mode is not entered either. [`initialize`] passes the flag that leaves
//! it alone, so nothing here writes the command register that switches the
//! platform from legacy to ACPI behaviour. The guest decides that for itself,
//! at the point it would have decided it on a machine with no hypervisor, and
//! finds the platform in the state firmware left it.
//!
//! # Reentrancy
//!
//! Every answer below can be called from inside uACPI, which can itself have
//! been entered from this module — a work item runs bytecode, which allocates,
//! which logs. So no answer here may take a lock that the path into uACPI is
//! already holding. In practice that means each of them takes at most its own
//! lock, holds it for the length of one operation, and calls nothing that could
//! come back around.
//!
//! The one lock that is not this module's own is the address space's, which
//! [`memory`] takes to map a range the direct map does not reach. That is
//! allowed because nothing under the address-space lock evaluates bytecode, and
//! it is why work items are drained on an ordinary call path rather than from
//! an interrupt handler.

mod firmware;
mod handle;
mod interrupts;
mod io;
mod logging;
mod memory;
mod pci;
mod sync;
mod time;
mod work;

use core::cell::UnsafeCell;

use log::{info, warn};
use paging::DirectMap;
use uacpi_sys::{Status, kernel, raw};

/// Publishes this image as uACPI's host, and gives it what it needs to read a
/// table.
///
/// Called once, on the boot processor, before [`early_tables`]. Both values
/// come from the loader: the root pointer is the address it read out of
/// firmware's configuration tables, and the direct map is the one it built.
///
/// A second call is reported rather than refused. Nothing calls this twice, and
/// the answer to a second caller is the same either way — the host that is
/// already installed is this one.
pub fn attach(map: DirectMap, rsdp: u64) {
    memory::attach(map, rsdp);
    if let Err(error) = kernel::install(&HOST) {
        warn!("core: uacpi was offered a second host: {error}");
    }
}

/// Everything uACPI asks of this machine, in the one place that answers it.
///
/// The field types are uACPI's own callback shapes, so a function named here
/// whose arguments are in the wrong order or whose integer is the wrong width
/// is a type error at the field rather than a symbol that links and misbehaves.
static HOST: kernel::Host = kernel::Host {
    get_rsdp: memory::uacpi_kernel_get_rsdp,
    map: memory::uacpi_kernel_map,
    unmap: memory::uacpi_kernel_unmap,
    log: logging::uacpi_kernel_log,
    pci_device_open: pci::uacpi_kernel_pci_device_open,
    pci_device_close: pci::uacpi_kernel_pci_device_close,
    pci_read8: pci::uacpi_kernel_pci_read8,
    pci_read16: pci::uacpi_kernel_pci_read16,
    pci_read32: pci::uacpi_kernel_pci_read32,
    pci_write8: pci::uacpi_kernel_pci_write8,
    pci_write16: pci::uacpi_kernel_pci_write16,
    pci_write32: pci::uacpi_kernel_pci_write32,
    io_map: io::uacpi_kernel_io_map,
    io_unmap: io::uacpi_kernel_io_unmap,
    io_read8: io::uacpi_kernel_io_read8,
    io_read16: io::uacpi_kernel_io_read16,
    io_read32: io::uacpi_kernel_io_read32,
    io_write8: io::uacpi_kernel_io_write8,
    io_write16: io::uacpi_kernel_io_write16,
    io_write32: io::uacpi_kernel_io_write32,
    alloc: memory::uacpi_kernel_alloc,
    free: memory::uacpi_kernel_free,
    get_nanoseconds_since_boot: time::uacpi_kernel_get_nanoseconds_since_boot,
    stall: time::uacpi_kernel_stall,
    sleep: time::uacpi_kernel_sleep,
    create_mutex: sync::uacpi_kernel_create_mutex,
    free_mutex: sync::uacpi_kernel_free_mutex,
    create_event: sync::uacpi_kernel_create_event,
    free_event: sync::uacpi_kernel_free_event,
    get_thread_id: sync::uacpi_kernel_get_thread_id,
    disable_interrupts: sync::uacpi_kernel_disable_interrupts,
    restore_interrupts: sync::uacpi_kernel_restore_interrupts,
    acquire_mutex: sync::uacpi_kernel_acquire_mutex,
    release_mutex: sync::uacpi_kernel_release_mutex,
    wait_for_event: sync::uacpi_kernel_wait_for_event,
    signal_event: sync::uacpi_kernel_signal_event,
    reset_event: sync::uacpi_kernel_reset_event,
    handle_firmware_request: firmware::uacpi_kernel_handle_firmware_request,
    install_interrupt_handler: interrupts::uacpi_kernel_install_interrupt_handler,
    uninstall_interrupt_handler: interrupts::uacpi_kernel_uninstall_interrupt_handler,
    create_spinlock: sync::uacpi_kernel_create_spinlock,
    free_spinlock: sync::uacpi_kernel_free_spinlock,
    lock_spinlock: sync::uacpi_kernel_lock_spinlock,
    unlock_spinlock: sync::uacpi_kernel_unlock_spinlock,
    schedule_work: work::uacpi_kernel_schedule_work,
    wait_for_work_completion: work::uacpi_kernel_wait_for_work_completion,
};

/// Bytes of scratch uACPI is given to describe the tables with before there is
/// a heap it could ask for the same memory from.
///
/// uACPI spends about fifty-six bytes per table here and hands the space back
/// once [`initialize`] has replaced it with an allocation, so this is a bound
/// on how many tables a machine may list during bring-up rather than a lasting
/// cost. A page is room for something over seventy of them, against the sixteen
/// uACPI keeps space for on its own — and a machine listing more tables than
/// this is one whose description is refused rather than half read.
const EARLY_TABLE_BYTES: usize = 4096;

/// Flags [`initialize`] brings uACPI up with.
///
/// Only one, and it is the whole of the difference between reading a platform
/// and taking it over: uACPI would otherwise enter ACPI mode, which means
/// writing firmware's own command register to change how the platform behaves.
/// The guest that boots after this hypervisor is the thing entitled to make
/// that change, and it makes it by writing the same register — through an
/// intercept this hypervisor does not need to second-guess.
const FLAGS: u64 = raw::UACPI_FLAG_NO_ACPI_MODE as u64;

/// Makes uACPI's table subsystem usable, with nothing behind it but the direct
/// map and the log.
///
/// [`attach`] must already have run, since this is the point uACPI asks where
/// the root pointer is and starts reading through the direct map.
///
/// # Errors
///
/// Whatever uACPI reported: that firmware published no root pointer, that the
/// structure at the published address is not one, or that no directory it names
/// could be read.
pub fn early_tables() -> Result<(), Status> {
    // Before uACPI has anything to say, so that nothing it would say is formatted
    // into a buffer for this image to discard.
    logging::adopt_level();
    let buffer = EARLY_TABLES.0.get();
    // SAFETY: the buffer is this image's own static storage and nothing else
    // ever names it. uACPI takes it as scratch for the table descriptors and
    // gives it up in `uacpi_initialize`, and this function is called once, on
    // the boot processor, before any other processor exists to race with it.
    let status = Status::new(unsafe {
        raw::uacpi_setup_early_table_access(buffer.cast(), EARLY_TABLE_BYTES)
    });
    status.ok()?;
    info!("core: uacpi table access is up on {EARLY_TABLE_BYTES:#x} bytes of scratch");
    Ok(())
}

/// Builds the namespace and runs what firmware's definition blocks put in it.
///
/// Everything uACPI can ask for has to be standing first: the heap, the log,
/// the clock, this processor's interrupt controller, and the configuration
/// space apertures. Bytecode reads and writes the platform's registers, so a
/// machine that gets here has already been described completely by
/// [`early_tables`].
///
/// # Errors
///
/// Whatever uACPI reported for whichever of the three stages failed. A machine
/// whose definition blocks do not load is one this hypervisor knows the shape
/// of but not the behaviour of, and the caller decides what that is worth.
pub fn initialize() -> Result<(), Status> {
    // SAFETY: every callback this reaches is defined in the submodules below and
    // every subsystem they stand on is up, which is what this function's
    // position in bring-up establishes. Called once, on the boot processor.
    Status::new(unsafe { raw::uacpi_initialize(FLAGS) }).ok()?;
    info!("core: uacpi subsystem initialized, acpi mode left as firmware set it");
    // SAFETY: as above, and the subsystem is initialized because the call
    // before this one succeeded.
    Status::new(unsafe { raw::uacpi_namespace_load() }).ok()?;
    info!("core: uacpi loaded the machine's definition blocks");
    // SAFETY: as above, and the namespace exists because the call before this
    // one succeeded.
    Status::new(unsafe { raw::uacpi_namespace_initialize() }).ok()?;
    info!("core: uacpi initialized the namespace");
    Ok(())
}

/// Logs what the host side of uACPI is holding.
pub fn describe(who: &str) {
    memory::describe(who);
    sync::describe(who);
    work::describe(who);
    interrupts::describe(who);
    firmware::describe(who);
}

/// The scratch [`early_tables`] hands over.
///
/// A cell rather than a plain array because uACPI writes through the pointer
/// for as long as it holds it, which is a shared reference to memory that is
/// being mutated — the one thing a `&'static [u8]` may never be. Nothing on the
/// Rust side ever reads it.
struct EarlyTables(UnsafeCell<[u64; EARLY_TABLE_BYTES / size_of::<u64>()]>);

// SAFETY: the contents are only ever reached through the raw pointer
// `early_tables` hands to uACPI, once, from the boot processor, and Rust never
// forms a reference to them. Declaring the wrapper shareable is what lets it be
// a static at all; it makes no claim about the bytes being safe to read.
unsafe impl Sync for EarlyTables {}

/// Aligned as a `u64` array, which is the pointer-size alignment uACPI asks
/// for.
static EARLY_TABLES: EarlyTables =
    EarlyTables(UnsafeCell::new([0; EARLY_TABLE_BYTES / size_of::<u64>()]));
