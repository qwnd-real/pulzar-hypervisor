//! The callbacks uACPI needs from whatever is hosting it.
//!
//! uACPI reaches its host through global C symbols with no context parameter:
//! the host defines them, the linker resolves them, and nothing checks that the
//! definitions have the shapes uACPI expects. A `#[unsafe(no_mangle)]` function
//! whose arguments are in the wrong order, or whose integer is a word too
//! narrow, links exactly as cleanly as a correct one and goes wrong at run
//! time.
//!
//! So this module defines all of them, once, and forwards each to a [`Host`] —
//! a struct with one field per callback, whose field types are the aliases
//! below. A host writes ordinary functions and names them in that struct, which
//! turns a mismatched shape into a type error at the field rather than into a
//! symbol that links and misbehaves. Each alias is also checked against the
//! generated declaration of uACPI's own `extern`, so the aliases cannot drift
//! from the headers either.
//!
//! # Why the callbacks are not simply defined by the host
//!
//! Because a host is not the only thing that links uACPI. Any firmware image
//! that pulls in a crate which reads ACPI tables links the whole library, and a
//! static library's undefined symbols have to be resolved whether or not
//! anything calls them — so an image with no reason to host uACPI would still
//! have to define forty-six functions it never runs, or fail to link. Owning
//! the symbols here means the boundary is satisfied by the crate that owns it,
//! and an image that hosts nothing gets a uACPI that refuses everything.
//!
//! Refusing is what happens before [`install`], and for an image that never
//! calls it. Every refusal is the strongest thing that can honestly be said
//! with no machine behind the call: no root pointer, no mapping, no memory, no
//! object. None of them is reachable from a caller that has not brought uACPI
//! up, because bringing uACPI up is what a host does after installing itself.
//!
//! # What a host must supply
//!
//! All of it. uACPI is compiled in its full configuration — interpreter,
//! namespace and event subsystem — so every field of [`Host`] is a callback the
//! library can make. The exception is the first four, which are the only ones
//! early table access uses: a host that has a direct map, a log and the root
//! pointer can read tables before it has an allocator, a clock or an interrupt
//! controller.

use core::{
    ffi::c_void,
    fmt::{self, Display, Formatter},
    ptr,
    sync::atomic::{AtomicPtr, Ordering},
};

use crate::{Status, raw};

/// Reports where firmware published the root pointer.
///
/// Early table access uses this, so it must answer before anything else in the
/// host is up.
pub type GetRsdp = unsafe extern "C" fn(*mut raw::uacpi_phys_addr) -> raw::uacpi_status;

/// Maps `len` bytes of physical memory and returns an address they can be read
/// through, or [`MAP_FAILED`].
///
/// The address may be misaligned, in which case the host rounds down to a page
/// boundary, maps enough pages to cover `len` bytes from the original address,
/// and returns the same offset within the first page.
pub type Map = unsafe extern "C" fn(raw::uacpi_phys_addr, raw::uacpi_size) -> *mut c_void;

/// Releases a mapping [`Map`] returned, given the same address and length.
pub type Unmap = unsafe extern "C" fn(*mut c_void, raw::uacpi_size);

/// Records one pre-formatted line, already terminated by a newline.
pub type Log = unsafe extern "C" fn(raw::uacpi_log_level, *const raw::uacpi_char);

/// What [`Map`] returns when it could not map the range.
///
/// uACPI spells this as a cast of `-1`, which is not a null pointer: a host
/// that returns null on failure reports success at an unusable address.
pub const MAP_FAILED: *mut c_void = usize::MAX as *mut c_void;

/// Opens a device's configuration space for reading and writing.
///
/// A device that is not there may be reported as [`Status::NOT_FOUND`], which
/// uACPI handles by standing in a device that reads as all ones — the shape a
/// great deal of firmware bytecode probes for absence with.
///
/// [`Status::NOT_FOUND`]: crate::Status::NOT_FOUND
pub type PciDeviceOpen =
    unsafe extern "C" fn(raw::uacpi_pci_address, *mut raw::uacpi_handle) -> raw::uacpi_status;

/// Releases what [`PciDeviceOpen`] returned.
pub type PciDeviceClose = unsafe extern "C" fn(raw::uacpi_handle);

/// Reads one byte of an open device's configuration space.
pub type PciRead8 = unsafe extern "C" fn(
    raw::uacpi_handle,
    raw::uacpi_size,
    *mut raw::uacpi_u8,
) -> raw::uacpi_status;

/// Reads two bytes of an open device's configuration space.
pub type PciRead16 = unsafe extern "C" fn(
    raw::uacpi_handle,
    raw::uacpi_size,
    *mut raw::uacpi_u16,
) -> raw::uacpi_status;

/// Reads four bytes of an open device's configuration space.
pub type PciRead32 = unsafe extern "C" fn(
    raw::uacpi_handle,
    raw::uacpi_size,
    *mut raw::uacpi_u32,
) -> raw::uacpi_status;

/// Writes one byte of an open device's configuration space.
pub type PciWrite8 =
    unsafe extern "C" fn(raw::uacpi_handle, raw::uacpi_size, raw::uacpi_u8) -> raw::uacpi_status;

/// Writes two bytes of an open device's configuration space.
pub type PciWrite16 =
    unsafe extern "C" fn(raw::uacpi_handle, raw::uacpi_size, raw::uacpi_u16) -> raw::uacpi_status;

/// Writes four bytes of an open device's configuration space.
pub type PciWrite32 =
    unsafe extern "C" fn(raw::uacpi_handle, raw::uacpi_size, raw::uacpi_u32) -> raw::uacpi_status;

/// Claims a range of the port address space for reading and writing.
pub type IoMap = unsafe extern "C" fn(
    raw::uacpi_io_addr,
    raw::uacpi_size,
    *mut raw::uacpi_handle,
) -> raw::uacpi_status;

/// Releases what [`IoMap`] returned.
pub type IoUnmap = unsafe extern "C" fn(raw::uacpi_handle);

/// Reads one byte at an offset into a claimed port range.
///
/// The width is the width the device expects and may not be split: four
/// one-byte accesses are not one four-byte access to hardware.
pub type IoRead8 = unsafe extern "C" fn(
    raw::uacpi_handle,
    raw::uacpi_size,
    *mut raw::uacpi_u8,
) -> raw::uacpi_status;

/// Reads two bytes at an offset into a claimed port range.
pub type IoRead16 = unsafe extern "C" fn(
    raw::uacpi_handle,
    raw::uacpi_size,
    *mut raw::uacpi_u16,
) -> raw::uacpi_status;

/// Reads four bytes at an offset into a claimed port range.
pub type IoRead32 = unsafe extern "C" fn(
    raw::uacpi_handle,
    raw::uacpi_size,
    *mut raw::uacpi_u32,
) -> raw::uacpi_status;

/// Writes one byte at an offset into a claimed port range.
pub type IoWrite8 =
    unsafe extern "C" fn(raw::uacpi_handle, raw::uacpi_size, raw::uacpi_u8) -> raw::uacpi_status;

/// Writes two bytes at an offset into a claimed port range.
pub type IoWrite16 =
    unsafe extern "C" fn(raw::uacpi_handle, raw::uacpi_size, raw::uacpi_u16) -> raw::uacpi_status;

/// Writes four bytes at an offset into a claimed port range.
pub type IoWrite32 =
    unsafe extern "C" fn(raw::uacpi_handle, raw::uacpi_size, raw::uacpi_u32) -> raw::uacpi_status;

/// Allocates a block of the given size, with unspecified contents.
pub type Alloc = unsafe extern "C" fn(raw::uacpi_size) -> *mut c_void;

/// Releases a block, given its address and the size it was allocated with.
///
/// The size is why the host needs no bookkeeping of its own: uACPI is built
/// with `UACPI_SIZED_FREES`, which is what makes this the layout Rust's
/// deallocation asks for. The address may be null, in which case there is
/// nothing to do.
pub type Free = unsafe extern "C" fn(*mut c_void, raw::uacpi_size);

/// Reads a strictly monotonic count of nanoseconds since the host started.
pub type GetNanosecondsSinceBoot = unsafe extern "C" fn() -> raw::uacpi_u64;

/// Spins for a number of microseconds, without yielding.
pub type Stall = unsafe extern "C" fn(raw::uacpi_u8);

/// Waits for a number of milliseconds, and may yield.
pub type Sleep = unsafe extern "C" fn(raw::uacpi_u64);

/// Creates a non-recursive mutex.
pub type CreateMutex = unsafe extern "C" fn() -> raw::uacpi_handle;

/// Destroys a mutex.
pub type FreeMutex = unsafe extern "C" fn(raw::uacpi_handle);

/// Creates a counting event, like a semaphore.
pub type CreateEvent = unsafe extern "C" fn() -> raw::uacpi_handle;

/// Destroys an event.
pub type FreeEvent = unsafe extern "C" fn(raw::uacpi_handle);

/// Names the thread that is running, with a value that is never
/// [`THREAD_ID_NONE`].
pub type GetThreadId = unsafe extern "C" fn() -> raw::uacpi_thread_id;

/// The identifier uACPI reserves to mean "no thread", which a host may never
/// return from [`GetThreadId`].
pub const THREAD_ID_NONE: raw::uacpi_thread_id = usize::MAX as raw::uacpi_thread_id;

/// Masks every interrupt on this processor and reports the state to restore.
pub type DisableInterrupts = unsafe extern "C" fn() -> raw::uacpi_interrupt_state;

/// Restores what [`DisableInterrupts`] reported.
pub type RestoreInterrupts = unsafe extern "C" fn(raw::uacpi_interrupt_state);

/// Acquires a mutex, waiting up to a number of milliseconds.
///
/// Zero means one non-blocking attempt and `0xFFFF` means wait for as long as
/// it takes. Failing to acquire within the timeout is
/// [`Status::TIMEOUT`](crate::Status::TIMEOUT), which is not an error; anything
/// else reported is treated as the host having broken.
pub type AcquireMutex =
    unsafe extern "C" fn(raw::uacpi_handle, raw::uacpi_u16) -> raw::uacpi_status;

/// Releases a mutex this thread holds.
pub type ReleaseMutex = unsafe extern "C" fn(raw::uacpi_handle);

/// Waits for an event's counter to be positive and takes one from it.
///
/// The timeout reads as it does for [`AcquireMutex`]. True means one was taken.
pub type WaitForEvent = unsafe extern "C" fn(raw::uacpi_handle, raw::uacpi_u16) -> raw::uacpi_bool;

/// Adds one to an event's counter, and may be called from an interrupt.
pub type SignalEvent = unsafe extern "C" fn(raw::uacpi_handle);

/// Puts an event's counter back to zero.
pub type ResetEvent = unsafe extern "C" fn(raw::uacpi_handle);

/// Answers a request the firmware's own bytecode made of the host, which is
/// either a breakpoint or a fatal error.
pub type HandleFirmwareRequest =
    unsafe extern "C" fn(*mut raw::uacpi_firmware_request) -> raw::uacpi_status;

/// Routes a system interrupt to a handler and reports a handle naming the
/// installation.
pub type InstallInterruptHandler = unsafe extern "C" fn(
    raw::uacpi_u32,
    raw::uacpi_interrupt_handler,
    raw::uacpi_handle,
    *mut raw::uacpi_handle,
) -> raw::uacpi_status;

/// Undoes what [`InstallInterruptHandler`] did, named by the handle it
/// reported.
pub type UninstallInterruptHandler =
    unsafe extern "C" fn(raw::uacpi_interrupt_handler, raw::uacpi_handle) -> raw::uacpi_status;

/// Creates a lock that may be taken from an interrupt handler.
pub type CreateSpinlock = unsafe extern "C" fn() -> raw::uacpi_handle;

/// Destroys a spinlock.
pub type FreeSpinlock = unsafe extern "C" fn(raw::uacpi_handle);

/// Takes a spinlock with interrupts masked, reporting the state to restore.
///
/// This cannot fail: there is no status to report and no caller prepared for
/// one.
pub type LockSpinlock = unsafe extern "C" fn(raw::uacpi_handle) -> raw::uacpi_cpu_flags;

/// Releases a spinlock and restores what [`LockSpinlock`] reported.
pub type UnlockSpinlock = unsafe extern "C" fn(raw::uacpi_handle, raw::uacpi_cpu_flags);

/// Queues work to run later, possibly from an interrupt handler.
pub type ScheduleWork = unsafe extern "C" fn(
    raw::uacpi_work_type,
    raw::uacpi_work_handler,
    raw::uacpi_handle,
) -> raw::uacpi_status;

/// Waits until every installed interrupt handler has returned and every queued
/// work item has run, in that order.
pub type WaitForWorkCompletion = unsafe extern "C" fn() -> raw::uacpi_status;

/// Everything one host answers, named so that a wrong shape is a type error.
///
/// Built as a `static` by the host and handed to [`install`]. There is one
/// field per callback and the field types are the aliases above, so a function
/// whose arguments are in the wrong order or whose integer is the wrong width
/// fails to compile at the field it was named in.
#[derive(Clone, Copy, Debug)]
pub struct Host {
    /// Where firmware published the root pointer.
    pub get_rsdp: GetRsdp,
    /// Reaching physical memory.
    pub map: Map,
    /// Giving a mapping back.
    pub unmap: Unmap,
    /// Recording a line.
    pub log: Log,
    /// Opening a device's configuration space.
    pub pci_device_open: PciDeviceOpen,
    /// Closing it again.
    pub pci_device_close: PciDeviceClose,
    /// Reading a byte of it.
    pub pci_read8: PciRead8,
    /// Reading a word of it.
    pub pci_read16: PciRead16,
    /// Reading a doubleword of it.
    pub pci_read32: PciRead32,
    /// Writing a byte of it.
    pub pci_write8: PciWrite8,
    /// Writing a word of it.
    pub pci_write16: PciWrite16,
    /// Writing a doubleword of it.
    pub pci_write32: PciWrite32,
    /// Claiming a run of ports.
    pub io_map: IoMap,
    /// Giving it back.
    pub io_unmap: IoUnmap,
    /// Reading a byte of it.
    pub io_read8: IoRead8,
    /// Reading a word of it.
    pub io_read16: IoRead16,
    /// Reading a doubleword of it.
    pub io_read32: IoRead32,
    /// Writing a byte of it.
    pub io_write8: IoWrite8,
    /// Writing a word of it.
    pub io_write16: IoWrite16,
    /// Writing a doubleword of it.
    pub io_write32: IoWrite32,
    /// Allocating.
    pub alloc: Alloc,
    /// Releasing.
    pub free: Free,
    /// Reading the monotonic clock.
    pub get_nanoseconds_since_boot: GetNanosecondsSinceBoot,
    /// Spinning for microseconds.
    pub stall: Stall,
    /// Waiting for milliseconds.
    pub sleep: Sleep,
    /// Creating a mutex.
    pub create_mutex: CreateMutex,
    /// Destroying one.
    pub free_mutex: FreeMutex,
    /// Creating an event.
    pub create_event: CreateEvent,
    /// Destroying one.
    pub free_event: FreeEvent,
    /// Naming the running thread.
    pub get_thread_id: GetThreadId,
    /// Masking interrupts.
    pub disable_interrupts: DisableInterrupts,
    /// Putting them back.
    pub restore_interrupts: RestoreInterrupts,
    /// Taking a mutex.
    pub acquire_mutex: AcquireMutex,
    /// Giving it back.
    pub release_mutex: ReleaseMutex,
    /// Waiting on an event.
    pub wait_for_event: WaitForEvent,
    /// Signalling one.
    pub signal_event: SignalEvent,
    /// Clearing one.
    pub reset_event: ResetEvent,
    /// Answering firmware's own bytecode.
    pub handle_firmware_request: HandleFirmwareRequest,
    /// Routing a system interrupt.
    pub install_interrupt_handler: InstallInterruptHandler,
    /// Unrouting it.
    pub uninstall_interrupt_handler: UninstallInterruptHandler,
    /// Creating a spinlock.
    pub create_spinlock: CreateSpinlock,
    /// Destroying one.
    pub free_spinlock: FreeSpinlock,
    /// Taking one.
    pub lock_spinlock: LockSpinlock,
    /// Releasing one.
    pub unlock_spinlock: UnlockSpinlock,
    /// Deferring work.
    pub schedule_work: ScheduleWork,
    /// Waiting for all of it to finish.
    pub wait_for_work_completion: WaitForWorkCompletion,
}

/// Makes `host` the answer to every callback uACPI makes from now on.
///
/// Called once, before uACPI is brought up, by whatever is going to host it.
/// Until it is — and forever, in an image that never calls it — every callback
/// refuses.
///
/// # Errors
///
/// [`AlreadyInstalled`] for a second call. Two hosts would mean uACPI reaching
/// a machine through one of them and giving the results back to the other, and
/// which one it reached would depend on when a call happened.
pub fn install(host: &'static Host) -> Result<(), AlreadyInstalled> {
    let host = ptr::from_ref(host).cast_mut();
    HOST.compare_exchange(ptr::null_mut(), host, Ordering::Release, Ordering::Relaxed)
        .map(drop)
        .map_err(|_| AlreadyInstalled)
}

/// Whether a host has been installed.
#[must_use]
pub fn installed() -> bool {
    host().is_some()
}

/// A second host was offered where one was already answering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AlreadyInstalled;

impl Display for AlreadyInstalled {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str("uACPI already has a host")
    }
}

impl core::error::Error for AlreadyInstalled {}

/// The installed host, if there is one.
fn host() -> Option<&'static Host> {
    // SAFETY: a non-null value here was published by `install` from a
    // `&'static Host`, and the acquire pairs with that release, so the pointee is
    // live and immutable for the rest of the program.
    unsafe { HOST.load(Ordering::Acquire).as_ref() }
}

/// The host, or null while there is none.
static HOST: AtomicPtr<Host> = AtomicPtr::new(ptr::null_mut());

/// Defines each callback uACPI calls, forwarding it to the installed host.
///
/// One arm per callback, each naming the symbol uACPI links against, the
/// [`Host`] field that answers it, and what to answer with when there is no
/// host. The forwarding itself is written once here rather than forty-six
/// times, because every one of them is the same three lines and the only
/// interesting part is the refusal.
macro_rules! forward {
    ($(
        $(#[doc = $doc:expr])*
        $symbol:ident => $field:ident($($argument:ident: $type:ty),* $(,)?)
            $(-> $result:ty)? , $refusal:expr;
    )*) => {
        $(
            $(#[doc = $doc])*
            ///
            /// # Safety
            ///
            /// Called by uACPI, which vouches for every argument it passes.
            #[unsafe(no_mangle)]
            unsafe extern "C" fn $symbol($($argument: $type),*) $(-> $result)? {
                match host() {
                    // SAFETY: the arguments are uACPI's own, forwarded unchanged,
                    // and the host vouched for its implementations having these
                    // shapes and contracts when it installed itself.
                    Some(host) => unsafe { (host.$field)($($argument),*) },
                    None => $refusal,
                }
            }
        )*
    };
}

forward! {
    /// Reports where firmware published the root pointer.
    uacpi_kernel_get_rsdp => get_rsdp(out: *mut raw::uacpi_phys_addr) -> raw::uacpi_status,
        Status::NOT_FOUND.code();
    /// Maps physical memory.
    uacpi_kernel_map => map(addr: raw::uacpi_phys_addr, len: raw::uacpi_size) -> *mut c_void,
        MAP_FAILED;
    /// Gives a mapping back.
    uacpi_kernel_unmap => unmap(addr: *mut c_void, len: raw::uacpi_size), ();
    /// Records one line uACPI formatted.
    uacpi_kernel_log => log(level: raw::uacpi_log_level, text: *const raw::uacpi_char), ();
    /// Opens a device's configuration space.
    uacpi_kernel_pci_device_open => pci_device_open(
        address: raw::uacpi_pci_address,
        out: *mut raw::uacpi_handle,
    ) -> raw::uacpi_status, Status::NOT_FOUND.code();
    /// Closes it again.
    uacpi_kernel_pci_device_close => pci_device_close(handle: raw::uacpi_handle), ();
    /// Reads a byte of it.
    uacpi_kernel_pci_read8 => pci_read8(
        device: raw::uacpi_handle,
        offset: raw::uacpi_size,
        value: *mut raw::uacpi_u8,
    ) -> raw::uacpi_status, Status::NOT_FOUND.code();
    /// Reads a word of it.
    uacpi_kernel_pci_read16 => pci_read16(
        device: raw::uacpi_handle,
        offset: raw::uacpi_size,
        value: *mut raw::uacpi_u16,
    ) -> raw::uacpi_status, Status::NOT_FOUND.code();
    /// Reads a doubleword of it.
    uacpi_kernel_pci_read32 => pci_read32(
        device: raw::uacpi_handle,
        offset: raw::uacpi_size,
        value: *mut raw::uacpi_u32,
    ) -> raw::uacpi_status, Status::NOT_FOUND.code();
    /// Writes a byte of it.
    uacpi_kernel_pci_write8 => pci_write8(
        device: raw::uacpi_handle,
        offset: raw::uacpi_size,
        value: raw::uacpi_u8,
    ) -> raw::uacpi_status, Status::NOT_FOUND.code();
    /// Writes a word of it.
    uacpi_kernel_pci_write16 => pci_write16(
        device: raw::uacpi_handle,
        offset: raw::uacpi_size,
        value: raw::uacpi_u16,
    ) -> raw::uacpi_status, Status::NOT_FOUND.code();
    /// Writes a doubleword of it.
    uacpi_kernel_pci_write32 => pci_write32(
        device: raw::uacpi_handle,
        offset: raw::uacpi_size,
        value: raw::uacpi_u32,
    ) -> raw::uacpi_status, Status::NOT_FOUND.code();
    /// Claims a run of ports.
    uacpi_kernel_io_map => io_map(
        base: raw::uacpi_io_addr,
        len: raw::uacpi_size,
        out: *mut raw::uacpi_handle,
    ) -> raw::uacpi_status, Status::UNIMPLEMENTED.code();
    /// Gives the claim back.
    uacpi_kernel_io_unmap => io_unmap(handle: raw::uacpi_handle), ();
    /// Reads a byte of it.
    uacpi_kernel_io_read8 => io_read8(
        handle: raw::uacpi_handle,
        offset: raw::uacpi_size,
        out: *mut raw::uacpi_u8,
    ) -> raw::uacpi_status, Status::UNIMPLEMENTED.code();
    /// Reads a word of it.
    uacpi_kernel_io_read16 => io_read16(
        handle: raw::uacpi_handle,
        offset: raw::uacpi_size,
        out: *mut raw::uacpi_u16,
    ) -> raw::uacpi_status, Status::UNIMPLEMENTED.code();
    /// Reads a doubleword of it.
    uacpi_kernel_io_read32 => io_read32(
        handle: raw::uacpi_handle,
        offset: raw::uacpi_size,
        out: *mut raw::uacpi_u32,
    ) -> raw::uacpi_status, Status::UNIMPLEMENTED.code();
    /// Writes a byte of it.
    uacpi_kernel_io_write8 => io_write8(
        handle: raw::uacpi_handle,
        offset: raw::uacpi_size,
        value: raw::uacpi_u8,
    ) -> raw::uacpi_status, Status::UNIMPLEMENTED.code();
    /// Writes a word of it.
    uacpi_kernel_io_write16 => io_write16(
        handle: raw::uacpi_handle,
        offset: raw::uacpi_size,
        value: raw::uacpi_u16,
    ) -> raw::uacpi_status, Status::UNIMPLEMENTED.code();
    /// Writes a doubleword of it.
    uacpi_kernel_io_write32 => io_write32(
        handle: raw::uacpi_handle,
        offset: raw::uacpi_size,
        value: raw::uacpi_u32,
    ) -> raw::uacpi_status, Status::UNIMPLEMENTED.code();
    /// Allocates a block.
    uacpi_kernel_alloc => alloc(size: raw::uacpi_size) -> *mut c_void, ptr::null_mut();
    /// Releases one.
    uacpi_kernel_free => free(mem: *mut c_void, size: raw::uacpi_size), ();
    /// Reads the monotonic clock.
    uacpi_kernel_get_nanoseconds_since_boot => get_nanoseconds_since_boot() -> raw::uacpi_u64, 0;
    /// Spins for microseconds.
    uacpi_kernel_stall => stall(micros: raw::uacpi_u8), ();
    /// Waits for milliseconds.
    uacpi_kernel_sleep => sleep(millis: raw::uacpi_u64), ();
    /// Creates a mutex.
    uacpi_kernel_create_mutex => create_mutex() -> raw::uacpi_handle, ptr::null_mut();
    /// Destroys one.
    uacpi_kernel_free_mutex => free_mutex(handle: raw::uacpi_handle), ();
    /// Creates an event.
    uacpi_kernel_create_event => create_event() -> raw::uacpi_handle, ptr::null_mut();
    /// Destroys one.
    uacpi_kernel_free_event => free_event(handle: raw::uacpi_handle), ();
    /// Names the running thread.
    uacpi_kernel_get_thread_id => get_thread_id() -> raw::uacpi_thread_id, UNHOSTED_THREAD;
    /// Masks interrupts.
    uacpi_kernel_disable_interrupts => disable_interrupts() -> raw::uacpi_interrupt_state, 0;
    /// Puts them back.
    uacpi_kernel_restore_interrupts => restore_interrupts(state: raw::uacpi_interrupt_state), ();
    /// Takes a mutex.
    uacpi_kernel_acquire_mutex => acquire_mutex(
        handle: raw::uacpi_handle,
        timeout: raw::uacpi_u16,
    ) -> raw::uacpi_status, Status::UNIMPLEMENTED.code();
    /// Gives one back.
    uacpi_kernel_release_mutex => release_mutex(handle: raw::uacpi_handle), ();
    /// Waits on an event.
    uacpi_kernel_wait_for_event => wait_for_event(
        handle: raw::uacpi_handle,
        timeout: raw::uacpi_u16,
    ) -> raw::uacpi_bool, false;
    /// Signals one.
    uacpi_kernel_signal_event => signal_event(handle: raw::uacpi_handle), ();
    /// Clears one.
    uacpi_kernel_reset_event => reset_event(handle: raw::uacpi_handle), ();
    /// Answers a request firmware's bytecode made.
    uacpi_kernel_handle_firmware_request => handle_firmware_request(
        request: *mut raw::uacpi_firmware_request,
    ) -> raw::uacpi_status, Status::OK.code();
    /// Routes a system interrupt.
    uacpi_kernel_install_interrupt_handler => install_interrupt_handler(
        irq: raw::uacpi_u32,
        handler: raw::uacpi_interrupt_handler,
        ctx: raw::uacpi_handle,
        out: *mut raw::uacpi_handle,
    ) -> raw::uacpi_status, Status::UNIMPLEMENTED.code();
    /// Unroutes one.
    uacpi_kernel_uninstall_interrupt_handler => uninstall_interrupt_handler(
        handler: raw::uacpi_interrupt_handler,
        handle: raw::uacpi_handle,
    ) -> raw::uacpi_status, Status::NOT_FOUND.code();
    /// Creates a spinlock.
    uacpi_kernel_create_spinlock => create_spinlock() -> raw::uacpi_handle, ptr::null_mut();
    /// Destroys one.
    uacpi_kernel_free_spinlock => free_spinlock(handle: raw::uacpi_handle), ();
    /// Takes one.
    uacpi_kernel_lock_spinlock => lock_spinlock(
        handle: raw::uacpi_handle,
    ) -> raw::uacpi_cpu_flags, 0;
    /// Releases one.
    uacpi_kernel_unlock_spinlock => unlock_spinlock(
        handle: raw::uacpi_handle,
        flags: raw::uacpi_cpu_flags,
    ), ();
    /// Defers work.
    uacpi_kernel_schedule_work => schedule_work(
        kind: raw::uacpi_work_type,
        handler: raw::uacpi_work_handler,
        ctx: raw::uacpi_handle,
    ) -> raw::uacpi_status, Status::UNIMPLEMENTED.code();
    /// Waits for deferred work to finish.
    uacpi_kernel_wait_for_work_completion => wait_for_work_completion() -> raw::uacpi_status,
        Status::OK.code();
}

/// What names the thread while there is no host to name it.
///
/// Not [`THREAD_ID_NONE`], because uACPI reserves that, and not null either: a
/// null thread would compare equal to an unowned mutex's owner field on some of
/// uACPI's paths. Nothing is mapped at the first page of an address space, so
/// this can be no real object.
const UNHOSTED_THREAD: raw::uacpi_thread_id = 1 as raw::uacpi_thread_id;

/// Each alias above, checked against the declaration generated from uACPI's own
/// header. Nothing reads these: they exist so that an alias that stops matching
/// the pinned uACPI fails the build here, in the crate that owns both the alias
/// and the definition, rather than at whichever host filled in the field.
mod matches_headers {
    use super::super::raw;

    const _: super::GetRsdp = raw::uacpi_kernel_get_rsdp;
    const _: super::Map = raw::uacpi_kernel_map;
    const _: super::Unmap = raw::uacpi_kernel_unmap;
    const _: super::Log = raw::uacpi_kernel_log;
    const _: super::PciDeviceOpen = raw::uacpi_kernel_pci_device_open;
    const _: super::PciDeviceClose = raw::uacpi_kernel_pci_device_close;
    const _: super::PciRead8 = raw::uacpi_kernel_pci_read8;
    const _: super::PciRead16 = raw::uacpi_kernel_pci_read16;
    const _: super::PciRead32 = raw::uacpi_kernel_pci_read32;
    const _: super::PciWrite8 = raw::uacpi_kernel_pci_write8;
    const _: super::PciWrite16 = raw::uacpi_kernel_pci_write16;
    const _: super::PciWrite32 = raw::uacpi_kernel_pci_write32;
    const _: super::IoMap = raw::uacpi_kernel_io_map;
    const _: super::IoUnmap = raw::uacpi_kernel_io_unmap;
    const _: super::IoRead8 = raw::uacpi_kernel_io_read8;
    const _: super::IoRead16 = raw::uacpi_kernel_io_read16;
    const _: super::IoRead32 = raw::uacpi_kernel_io_read32;
    const _: super::IoWrite8 = raw::uacpi_kernel_io_write8;
    const _: super::IoWrite16 = raw::uacpi_kernel_io_write16;
    const _: super::IoWrite32 = raw::uacpi_kernel_io_write32;
    const _: super::Alloc = raw::uacpi_kernel_alloc;
    const _: super::Free = raw::uacpi_kernel_free;
    const _: super::GetNanosecondsSinceBoot = raw::uacpi_kernel_get_nanoseconds_since_boot;
    const _: super::Stall = raw::uacpi_kernel_stall;
    const _: super::Sleep = raw::uacpi_kernel_sleep;
    const _: super::CreateMutex = raw::uacpi_kernel_create_mutex;
    const _: super::FreeMutex = raw::uacpi_kernel_free_mutex;
    const _: super::CreateEvent = raw::uacpi_kernel_create_event;
    const _: super::FreeEvent = raw::uacpi_kernel_free_event;
    const _: super::GetThreadId = raw::uacpi_kernel_get_thread_id;
    const _: super::DisableInterrupts = raw::uacpi_kernel_disable_interrupts;
    const _: super::RestoreInterrupts = raw::uacpi_kernel_restore_interrupts;
    const _: super::AcquireMutex = raw::uacpi_kernel_acquire_mutex;
    const _: super::ReleaseMutex = raw::uacpi_kernel_release_mutex;
    const _: super::WaitForEvent = raw::uacpi_kernel_wait_for_event;
    const _: super::SignalEvent = raw::uacpi_kernel_signal_event;
    const _: super::ResetEvent = raw::uacpi_kernel_reset_event;
    const _: super::HandleFirmwareRequest = raw::uacpi_kernel_handle_firmware_request;
    const _: super::InstallInterruptHandler = raw::uacpi_kernel_install_interrupt_handler;
    const _: super::UninstallInterruptHandler = raw::uacpi_kernel_uninstall_interrupt_handler;
    const _: super::CreateSpinlock = raw::uacpi_kernel_create_spinlock;
    const _: super::FreeSpinlock = raw::uacpi_kernel_free_spinlock;
    const _: super::LockSpinlock = raw::uacpi_kernel_lock_spinlock;
    const _: super::UnlockSpinlock = raw::uacpi_kernel_unlock_spinlock;
    const _: super::ScheduleWork = raw::uacpi_kernel_schedule_work;
    const _: super::WaitForWorkCompletion = raw::uacpi_kernel_wait_for_work_completion;
}
