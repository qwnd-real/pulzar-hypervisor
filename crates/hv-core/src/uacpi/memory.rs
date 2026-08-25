//! Reaching physical memory, and the memory uACPI asks for.
//!
//! Four of the answers uACPI needs are about memory, and three of them are the
//! only ones early table access uses: where firmware published the root
//! pointer, how to read a physical range, and how to stop reading it. The
//! fourth, the allocator, is not used until the namespace is built.
//!
//! # Two ways to reach a physical address, chosen by which one can work
//!
//! Almost everything uACPI asks for is memory: a table, a definition block, a
//! structure hanging off one. All of it is described as memory by firmware's own
//! memory map, and the direct map already describes all of that — so the answer
//! is arithmetic on an address, with no page table edited and no range claimed.
//! That matters, because it is the answer given for every table on every machine.
//!
//! What is left is operation regions in device memory, which is where a great
//! deal of firmware's bytecode does its work: the platform's own registers, an
//! event timer's block, a chipset's configuration. Those get a mapping of their
//! own out of the mapping window, uncached, because a device's registers are not
//! memory and must not be read from a cache line. Each one is remembered until it
//! is given back, since releasing it needs the value the address space handed out
//! rather than just the address.
//!
//! # Being inside the direct map is not being mapped by it
//!
//! The distinction the choice turns on is not the direct map's extent. The map
//! spans the whole physical address space — the loader sized it from the top of
//! physical memory, not from the sum of the memory it found — while what it
//! actually describes is the runs firmware's memory map called memory. Device
//! registers sit in the holes between those runs, and every hole is numerically
//! inside the map.
//!
//! So the question asked is whether the direct map really translates the range,
//! which only its page tables can answer. Before the address space becomes the
//! machine's there is nothing to ask — and nothing to ask about either, because
//! all early table access maps is tables, and tables are in memory. Afterwards
//! every range is checked, both ends of it, and anything the map does not
//! translate is treated as the device memory it is.

use alloc::{
    alloc::{Layout, alloc, dealloc},
    vec::Vec,
};
use core::{
    ffi::c_void,
    sync::atomic::{AtomicU64, Ordering},
};

use log::{info, warn};
use paging::{CacheType, DirectMap, Mapping, Protection};
use spin::{Mutex, Once};
use uacpi_sys::{Status, kernel, raw};
use x86_64::{PhysAddr, VirtAddr};

/// The alignment every allocation uACPI is given comes back on.
///
/// uACPI asks for bytes and says nothing about alignment, so the host has to
/// pick one that suits everything it might put there. Sixteen is what C's own
/// allocator guarantees on this architecture and covers every scalar and vector
/// type a compiler will place in such a block.
///
/// It is also part of the contract with [`free`]: Rust's deallocation needs the
/// layout the allocation was made with, and this constant is the half of that
/// layout uACPI does not carry back.
const ALIGNMENT: usize = 16;

/// Publishes what uACPI needs before it can read a single table.
///
/// Called once, on the boot processor, before [`super::early_tables`]. Both
/// values come from the loader: the root pointer is the address it read out of
/// firmware's configuration tables, and the direct map is the one it built.
///
/// A root pointer of zero is how the boot protocol says firmware published
/// none, and is published as such rather than refused here — [`get_rsdp`] is
/// where uACPI asks, and reporting it there is what puts the failure in uACPI's
/// own terms.
pub fn attach(map: DirectMap, rsdp: u64) {
    ROOT_POINTER.store(rsdp, Ordering::Relaxed);
    MAP.call_once(|| map);
}

/// Logs what the memory answers are holding.
pub fn describe(who: &str) {
    let devices = DEVICE_MAPPINGS.lock().len();
    info!(
        "{who}: uacpi holds {devices} device mapping{} outside the direct map",
        if devices == 1 { "" } else { "s" }
    );
}

/// Reports where firmware published the root pointer.
///
/// # Safety
///
/// Called by uACPI with a pointer to storage for one physical address.
pub(super) unsafe extern "C" fn uacpi_kernel_get_rsdp(
    out: *mut raw::uacpi_phys_addr,
) -> raw::uacpi_status {
    let rsdp = ROOT_POINTER.load(Ordering::Relaxed);
    if out.is_null() {
        return Status::INVALID_ARGUMENT.code();
    }
    if rsdp == 0 {
        // Either firmware published none, or nothing has been attached yet. The
        // two are the same answer to uACPI and differ only in the log.
        warn!("core: uacpi asked for the acpi root pointer and there is none");
        return Status::NOT_FOUND.code();
    }
    // SAFETY: the caller supplies storage for one physical address, checked
    // non-null above.
    unsafe { out.write(rsdp) };
    Status::OK.code()
}

/// Makes `len` bytes at a physical address readable, sub-page offset preserved.
///
/// # Safety
///
/// Called by uACPI. Nothing is dereferenced here; the returned address is
/// uACPI's to read for as long as it holds the mapping.
pub(super) unsafe extern "C" fn uacpi_kernel_map(
    addr: raw::uacpi_phys_addr,
    len: raw::uacpi_size,
) -> *mut c_void {
    let Some(map) = MAP.get() else {
        warn!("core: uacpi asked to map {addr:#x} before the direct map was attached");
        return kernel::MAP_FAILED;
    };
    if len == 0 {
        warn!("core: uacpi asked to map nothing at {addr:#x}");
        return kernel::MAP_FAILED;
    }
    let Ok(phys) = PhysAddr::try_new(addr) else {
        warn!("core: uacpi asked to map {addr:#x}, which is not a physical address");
        return kernel::MAP_FAILED;
    };
    let bytes = u64::try_from(len).unwrap_or(u64::MAX);
    if let Some(virt) = translated(map, phys, bytes) {
        return virt.as_mut_ptr();
    }
    device(phys, bytes)
}

/// Releases what [`uacpi_kernel_map`] returned.
///
/// # Safety
///
/// Called by uACPI with an address it was given and the length it asked for.
/// Nothing derived from the address may still be in use, which is uACPI's own
/// contract with itself.
pub(super) unsafe extern "C" fn uacpi_kernel_unmap(addr: *mut c_void, len: raw::uacpi_size) {
    let Some(map) = MAP.get() else {
        warn!("core: uacpi gave back a mapping before the direct map was attached");
        return;
    };
    let virt = addr as u64;
    let base = map.base().as_u64();
    if virt >= base && virt - base < map.size() {
        // The direct map describes it, and the direct map is never taken down.
        return;
    }
    release(virt, len as u64);
}

/// Allocates a block of the requested size.
///
/// # Safety
///
/// Called by uACPI. The block is uACPI's until it hands it to [`free`].
pub(super) unsafe extern "C" fn uacpi_kernel_alloc(size: raw::uacpi_size) -> *mut c_void {
    let Some(layout) = layout(size) else {
        return core::ptr::null_mut();
    };
    // SAFETY: the layout has a non-zero size, which is the whole of what the
    // global allocator asks of its caller.
    unsafe { alloc(layout) }.cast()
}

/// Releases a block, given the size it was allocated with.
///
/// # Safety
///
/// Called by uACPI with a block [`uacpi_kernel_alloc`] returned and the size it
/// was asked for, or with a null pointer. uACPI is built with
/// `UACPI_SIZED_FREES` precisely so that the size arrives here, because it is
/// half of the layout the global allocator needs back.
pub(super) unsafe extern "C" fn uacpi_kernel_free(mem: *mut c_void, size: raw::uacpi_size) {
    if mem.is_null() {
        return;
    }
    let Some(layout) = layout(size) else {
        // A block with an address but no size cannot have come from `alloc`, and
        // there is no layout that would release it. Leaking is the only answer
        // that is not undefined.
        warn!("core: uacpi gave back the block at {mem:p} with a size of {size}");
        return;
    };
    // SAFETY: the caller supplies a block this module allocated and the size it
    // was allocated with, and `layout` derives the same layout from the same
    // size, so this is the layout the allocation was made with.
    unsafe { dealloc(mem.cast(), layout) };
}

/// Where the direct map really reads `bytes` at `phys`, if it really does.
///
/// Numerically inside the map is not enough — see this module — so the map's own
/// page tables are asked, at both ends of the range, and a range they do not
/// both answer for belongs to [`device`] instead.
///
/// Before the address space became the machine's there is nothing to ask, and
/// nothing to ask about: all early table access maps is tables, and a table is in
/// memory the map describes. That is also the only window in which this is on the
/// path of every table on every machine, which is the one place the cost of a
/// walk would be worth avoiding.
fn translated(map: &DirectMap, phys: PhysAddr, bytes: u64) -> Option<VirtAddr> {
    let virt = map.reach(phys, bytes).ok()?;
    if !paging::adopted() {
        return Some(virt);
    }
    let last = virt + (bytes - 1);
    // A walk of this image's own tables and nothing else, which is what the
    // address-space lock is allowed to be held for.
    paging::with(|space| space.translate(virt).is_ok() && space.translate(last).is_ok())
        .ok()?
        .then_some(virt)
}

/// Maps a physical range the direct map does not describe, which is device
/// memory.
fn device(phys: PhysAddr, bytes: u64) -> *mut c_void {
    // SAFETY: the direct map does not translate this range, so firmware's memory
    // map did not describe it as memory and nothing in this image maps it as such.
    // Writable and uncached because bytecode uses these ranges to read and write
    // a device's registers, which a cached mapping would not reach.
    let mapped = paging::with(|space| unsafe {
        space.map_physical(phys, bytes, Protection::ReadWrite, CacheType::Uncached)
    });
    let mapping = match mapped {
        Ok(Ok(mapping)) => mapping,
        Ok(Err(error)) => {
            warn!("core: uacpi could not be given {bytes:#x} bytes at {phys:#x}: {error}");
            return kernel::MAP_FAILED;
        }
        Err(error) => {
            warn!("core: uacpi asked for {phys:#x} and the address space was unreachable: {error}");
            return kernel::MAP_FAILED;
        }
    };
    let addr = mapping.addr().as_mut_ptr();
    DEVICE_MAPPINGS.lock().push(mapping);
    addr
}

/// Gives back a device mapping, named by the address and length uACPI has.
fn release(virt: u64, bytes: u64) {
    let mut mappings = DEVICE_MAPPINGS.lock();
    let Some(index) = mappings
        .iter()
        .position(|mapping| mapping.addr().as_u64() == virt && mapping.bytes() == bytes)
    else {
        warn!("core: uacpi gave back {bytes:#x} bytes at {virt:#x}, which was never mapped");
        return;
    };
    let mapping = mappings.swap_remove(index);
    // Dropped before the address space is taken, so the two locks are never held
    // at once and this one is not held across a shootdown.
    drop(mappings);
    // SAFETY: uACPI states that nothing derived from a mapping it gives back is
    // still in use, and the mapping was removed from the list above, so no
    // second release of it can be represented.
    match paging::with(|space| unsafe { space.unmap(mapping) }) {
        Ok(Ok(())) => {}
        Ok(Err(error)) => warn!("core: uacpi's mapping at {virt:#x} could not be removed: {error}"),
        Err(error) => {
            warn!("core: uacpi's mapping at {virt:#x} outlived the address space: {error}");
        }
    }
}

/// The layout a block of `size` bytes is allocated and released with.
///
/// One function for both directions, so the two can never disagree. A size of
/// zero has no layout: the global allocator forbids it, and uACPI asking for
/// nothing is answered with nothing rather than with a block.
fn layout(size: raw::uacpi_size) -> Option<Layout> {
    (size != 0)
        .then(|| Layout::from_size_align(size, ALIGNMENT).ok())
        .flatten()
}

/// Where firmware published the root pointer, or zero for no root pointer.
static ROOT_POINTER: AtomicU64 = AtomicU64::new(0);

/// The direct map every physical read goes through.
static MAP: Once<DirectMap> = Once::new();

/// Every mapping made for a range the direct map does not reach.
///
/// Kept because releasing one needs the value the address space returned, not
/// just its address — the window slots behind it are given back by that value
/// and cannot be named any other way. Short by construction: bytecode holds an
/// operation region mapped only while it is using it.
static DEVICE_MAPPINGS: Mutex<Vec<Mapping>> = Mutex::new(Vec::new());
