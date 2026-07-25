//! First-stage UEFI loader for the pulzar hypervisor.
//!
//! Firmware starts this image; it ends by jumping into the hypervisor image
//! with an address space of pulzar's own making. In between it does five
//! things, in this order and for these reasons:
//!
//! 1. Registers an unload handler on itself, so the hypervisor can evict it
//!    later. Nothing else works if this does not.
//! 2. Reserves the one chunk of physical memory pulzar will own. It has to
//!    happen before the memory map is captured, so the chunk appears in the map
//!    as reserved rather than as memory something might hand out again.
//! 3. Captures firmware's memory map into that chunk, because the address space
//!    is sized from it and because firmware's own copy lives in memory the
//!    hypervisor will lose.
//! 4. Loads, relocates and maps the hypervisor image at a randomized high-half
//!    address, alongside a guarded stack, a direct map of physical memory and a
//!    window for explicit mappings.
//! 5. Publishes a [`Handoff`] describing all of it and jumps, with the new
//!    address space active.
//!
//! The address space it activates is deliberately half firmware's: the lower
//! half is copied from firmware's own tables, so boot services keep working
//! across the jump and the hypervisor can still unload this image. Discarding
//! that half is the hypervisor's job, once it no longer needs anything of
//! firmware's.
//!
//! Every step is logged over serial before it is relied upon, so a boot that
//! fails leaves behind the point it failed at.

#![no_main]
#![no_std]

mod error;
mod firmware;
mod image;

use core::{arch::asm, convert::Infallible, ptr::NonNull};

use clock::Wall;
use handoff::Handoff;
use log::{error, info};
use paging::{
    AddressSpace, CacheType, DirectMap, PagingError, Protection, Stack, buddy, chunk,
    kaslr::{self, Entropy, Placement},
};
use uefi::{Status, boot::MemoryDescriptor, entry};
use x86_64::{
    PhysAddr, VirtAddr,
    structures::paging::{PageSize, PhysFrame, Size2MiB},
};

use crate::{
    error::LoaderError,
    firmware::{ImageFile, Loader, Memory},
    image::Image,
};

/// Bytes of the image file read before anything else, to parse its headers
/// from.
///
/// It has to cover the DOS header, the PE headers and the whole section table.
/// One page covers all three with room to spare for any image `rust-lld`
/// produces, and [`Image::parse`] reports a buffer that turns out to be short
/// rather than reading past it. Spelled out rather than derived from
/// [`chunk::FRAME_SIZE`] because narrowing that `u64` is not a `const`
/// operation.
const PROBE_BYTES: usize = 4096;

/// Pages of the hypervisor's initial stack: 64 KiB, with an unmapped guard page
/// below and above. Enough for bring-up, which puts nothing large on the stack
/// and recurses nowhere.
const CORE_STACK_PAGES: u64 = 16;

/// Bytes of stack the entry point's calling convention expects to find already
/// reserved.
///
/// `efiapi` is the Microsoft x64 convention: the caller reserves 32 bytes of
/// shadow space above the return address for the callee to spill its register
/// arguments into, and a `call` leaves the return address itself below that.
/// This jumps rather than calls, so both are accounted for here — which also
/// produces the `RSP ≡ 8 (mod 16)` the convention requires on entry. Starting
/// at the stack top instead would put the shadow space in the upper guard page.
const ENTRY_SHADOW_SPACE: u64 = 0x28;

/// The handoff is written into the space the chunk sets aside for it, so it has
/// to fit there. Both sides are compile-time constants, so a mismatch is a
/// build failure rather than something to discover at the jump.
const _: () = assert!(
    size_of::<Handoff>() as u64 <= chunk::HANDOFF_SIZE,
    "the handoff does not fit the chunk region reserved for it"
);

/// Firmware's entry point.
///
/// Serial comes up first so that everything after it, including a failure, is
/// observable. There is nowhere to report a serial failure to, which is why it
/// is the one step whose error is a bare status.
#[entry]
fn main() -> Status {
    if serial::init().is_err() {
        return Status::DEVICE_ERROR;
    }
    info!("loader: pulzar hv-loader starting");
    match boot() {
        // `boot` only ever returns by failing; its success type is uninhabited.
        Ok(never) => match never {},
        Err(error) => {
            error!("loader: boot failed: {error}");
            Status::LOAD_ERROR
        }
    }
}

/// Everything from firmware's environment to the hypervisor's.
///
/// # Errors
///
/// The first failure of any step, which ends the boot. Nothing is rolled back:
/// the reserved chunk stays reserved and firmware is left as it was found,
/// which is what firmware expects of an application that returns an error.
fn boot() -> Result<Infallible, LoaderError> {
    let loader = firmware::claim_self()?;
    info!(
        "loader: own image at {:#x}, {:#x} bytes, unload handler registered",
        loader.base, loader.size
    );

    let chunk_base = firmware::allocate_chunk()?;
    info!(
        "loader: reserved chunk at {chunk_base:#x}, {:#x} bytes",
        chunk::CHUNK_SIZE
    );

    let memory = survey_memory(chunk_base)?;
    info!(
        "loader: copied {} memory descriptors, physical memory ends at {:#x}",
        memory.entries, memory.top_of_ram
    );

    let mut file = ImageFile::open()?;
    let image = probe(&mut file)?;
    info!(
        "loader: hypervisor image spans {:#x} bytes in {} sections",
        image.size(),
        image.sections().len()
    );

    let placement = kaslr::place(
        &mut Entropy::new(),
        image.size(),
        paging::direct_map_size(memory.top_of_ram),
    )?;
    // SAFETY: `chunk_base` is the base of a `CHUNK_SIZE` region firmware just
    // reserved for this image alone, which firmware's still-active address space
    // maps identically, and `top_of_ram` is one past the highest physical address
    // firmware's memory map describes.
    let mut space = unsafe { AddressSpace::build(chunk_base, memory.top_of_ram, placement) }?;
    let stack = load(&mut space, &mut file, &image, placement)?;

    // Firmware objects are closed here rather than dropped: the jump below never
    // returns, so no destructor at the end of this function would ever run.
    drop(file);
    space.describe("loader");

    let handoff = publish(
        &space,
        chunk_base,
        loader,
        memory,
        image.size(),
        stack,
        placement,
    )?;
    let entry = VirtAddr::new(image.entry(placement.image_base.as_u64()));
    info!(
        "loader: entering hypervisor at {entry:#x}, stack top {:#x}, handoff {handoff:#x}",
        stack.top()
    );
    // SAFETY: `space` maps the hypervisor's image at `entry` and its stack below
    // `stack.top()`, and its lower half is firmware's own, so the instructions
    // between the `CR3` load and the jump stay mapped where they are executing
    // from. `handoff` is a direct-map address of a `Handoff` this space maps, and
    // the entry point of a UEFI image never returns to its caller.
    unsafe { enter(space.root(), stack.top(), entry, handoff) }
}

/// Copies firmware's memory map into the chunk's metadata region.
///
/// # Errors
///
/// [`LoaderError::Paging`] if firmware's identity map does not reach the chunk,
/// or whatever [`firmware::capture_memory_map`] reports.
fn survey_memory(chunk_base: PhysAddr) -> Result<Memory, LoaderError> {
    let capacity = bytes(chunk::MEMORY_MAP_SIZE) / size_of::<MemoryDescriptor>();
    let destination = identity_ptr::<MemoryDescriptor>(chunk_base + chunk::MEMORY_MAP_OFFSET)?;
    // SAFETY: the destination is the chunk's memory-map region, `MEMORY_MAP_SIZE`
    // bytes long and owned by nothing else — no allocator ever hands out the
    // metadata frames — so it holds exactly `capacity` descriptors.
    unsafe { firmware::capture_memory_map(destination, capacity) }
}

/// Reads the front of the hypervisor image and parses its headers.
///
/// # Errors
///
/// [`LoaderError::Firmware`] if the read fails, or [`LoaderError::Image`] if
/// the image is not one this loader can place.
fn probe(file: &mut ImageFile) -> Result<Image, LoaderError> {
    let mut headers = [0; PROBE_BYTES];
    let read = file.read_at(0, &mut headers)?;
    Ok(Image::parse(&headers[..read])?)
}

/// Places the hypervisor image and its stack in the new address space.
///
/// # Errors
///
/// [`LoaderError::Paging`] if the chunk cannot back the image, its page tables
/// or its stack, [`LoaderError::Firmware`] or [`LoaderError::ShortRead`] if the
/// image cannot be read, or [`LoaderError::Image`] if it cannot be relocated.
fn load(
    space: &mut AddressSpace,
    file: &mut ImageFile,
    image: &Image,
    placement: Placement,
) -> Result<Stack, LoaderError> {
    let order = image_order(image.size());
    let span = chunk::FRAME_SIZE << order;
    let phys = space
        .frames()
        .allocate(order)
        .ok_or(PagingError::OutOfFrames { order })?
        .start_address();
    info!(
        "loader: image frames at {phys:#x}, {span:#x} bytes for {:#x} bytes of image",
        image.size()
    );

    write_image(file, image, phys, placement.image_base)?;
    map_image(space, image, phys, placement.image_base)?;
    // The direct map is read-write over all of physical memory, which would
    // otherwise leave a writable alias of the hypervisor's own code.
    space.protect_direct_map(phys, span, Protection::ReadOnly)?;

    let stack = space.allocate_stack(CORE_STACK_PAGES)?;
    info!(
        "loader: hypervisor stack {:#x}..{:#x}, guard page on either side",
        stack.bottom(),
        stack.top()
    );
    Ok(stack)
}

/// Order of the frame run the image is loaded into.
///
/// Whatever the image needs, but never less than a large page. A run of that
/// order is 2 MiB aligned and 2 MiB sized, which is what lets the direct map's
/// protection over it be tightened by changing flags instead of splitting the
/// large page that describes it.
fn image_order(size: u64) -> usize {
    let pages = bytes(size.div_ceil(chunk::FRAME_SIZE));
    let large = bytes(Size2MiB::SIZE / chunk::FRAME_SIZE);
    buddy::order_for(pages).max(buddy::order_for(large))
}

/// Copies the image into its frames and rebases it to `base`.
///
/// The frames are written through firmware's identity map, the only way to
/// reach them until the new address space is active. Uninitialized data needs
/// no zeroing: the frame allocator hands out zeroed runs.
///
/// # Errors
///
/// [`LoaderError::Paging`] if the identity map does not reach the frames,
/// [`LoaderError::Firmware`] or [`LoaderError::ShortRead`] if the file cannot
/// be read, or [`LoaderError::Image`] if a relocation cannot be applied.
fn write_image(
    file: &mut ImageFile,
    image: &Image,
    phys: PhysAddr,
    base: VirtAddr,
) -> Result<(), LoaderError> {
    let pointer = identity_ptr::<u8>(phys)?;
    // SAFETY: the run was just allocated from the chunk, so this is its only
    // reference; it is at least `image.size()` bytes long, and firmware's active
    // address space maps every byte of it read-write at its physical address.
    let destination =
        unsafe { NonNull::slice_from_raw_parts(pointer, bytes(image.size())).as_mut() };

    // Every section lies inside the image's own span, which `Image::parse`
    // established, so each slice below is in range.
    file.read_exact(0, &mut destination[..bytes(image.header_bytes())])?;
    for section in image.sections() {
        let start = bytes(section.offset);
        let end = start + bytes(section.file_size);
        file.read_exact(section.file_offset, &mut destination[start..end])?;
    }
    image.relocate(destination, base.as_u64())?;
    Ok(())
}

/// Maps the image at `base`, one mapping per section.
///
/// The headers are mapped too, read-only, so the image can find its own
/// structures. Protections come from the image itself, which is what makes
/// `.text` the only executable mapping and leaves nothing writable and
/// executable at once.
///
/// # Errors
///
/// [`LoaderError::Paging`] if the chunk cannot back the page tables the
/// mappings need.
fn map_image(
    space: &mut AddressSpace,
    image: &Image,
    phys: PhysAddr,
    base: VirtAddr,
) -> Result<(), LoaderError> {
    let headers = image.header_bytes().next_multiple_of(chunk::FRAME_SIZE);
    // SAFETY: the frames belong to the chunk this space owns and nothing else
    // maps them; the direct map's alias of them is made read-only by `load`.
    unsafe {
        space.map_region(
            base,
            phys,
            headers,
            Protection::ReadOnly,
            CacheType::WriteBack,
        )
    }?;
    info!("loader: mapped headers at {base:#x}, {headers:#x} bytes, read-only");

    for section in image.sections() {
        let virt = base + section.offset;
        // SAFETY: as above, and sections are ascending, non-overlapping and
        // inside the run, so no two of these mappings can collide.
        unsafe {
            space.map_region(
                virt,
                phys + section.offset,
                section.size,
                section.protection,
                CacheType::WriteBack,
            )
        }?;
        info!(
            "loader: mapped {} at {virt:#x}, {:#x} bytes, {:?}",
            section.name(),
            section.size,
            section.protection
        );
    }
    Ok(())
}

/// Writes the handoff into the chunk and returns the address the hypervisor
/// receives it at.
///
/// That address is a direct-map one, so it stays valid after the hypervisor
/// drops the firmware half of the address space.
///
/// # Errors
///
/// [`LoaderError::Firmware`] if the system table cannot be located, or
/// [`LoaderError::Paging`] if a window does not reach the chunk.
fn publish(
    space: &AddressSpace,
    chunk_base: PhysAddr,
    loader: Loader,
    memory: Memory,
    core_image_size: u64,
    stack: Stack,
    placement: Placement,
) -> Result<VirtAddr, LoaderError> {
    let system_table = uefi::table::system_table_raw().ok_or(LoaderError::Firmware {
        operation: "locate the UEFI system table",
        status: Status::NOT_FOUND,
    })?;
    let handoff = Handoff {
        magic: Handoff::MAGIC,
        version: Handoff::VERSION,
        size: narrow(size_of::<Handoff>()),
        system_table: system_table.as_ptr(),
        loader_image_handle: loader.handle.as_ptr(),
        loader_image_base: loader.base,
        loader_image_size: loader.size,
        chunk_base: chunk_base.as_u64(),
        chunk_size: chunk::CHUNK_SIZE,
        page_table_root: space.root().start_address().as_u64(),
        direct_map_base: space.direct_map().base().as_u64(),
        direct_map_size: space.direct_map().size(),
        mapping_window_base: space.mapping_window().as_u64(),
        mapping_window_size: chunk::MAPPING_WINDOW_SIZE,
        core_image_base: placement.image_base.as_u64(),
        core_image_size,
        stack_base: stack.bottom().as_u64(),
        stack_size: stack.pages() * chunk::FRAME_SIZE,
        memory_map: direct(space, chunk_base + chunk::MEMORY_MAP_OFFSET)?.as_u64(),
        memory_map_entries: narrow(memory.entries),
        memory_map_entry_size: narrow(size_of::<MemoryDescriptor>()),
        top_of_ram: memory.top_of_ram,
        acpi_rsdp: firmware::acpi_rsdp(),
        // Read here rather than at the start of the boot, so that the gap
        // between this reading and the hypervisor's clock coming up is as small
        // as the loader can make it: nothing measures that gap, and whatever it
        // is, the wall clock is behind by it for good.
        boot_wall_nanos: firmware::wall_clock().map_or(0, Wall::nanos),
    };

    let phys = chunk_base + chunk::HANDOFF_OFFSET;
    let pointer = identity_ptr::<Handoff>(phys)?;
    // SAFETY: the handoff region is the front of the chunk, set aside for exactly
    // this and never handed out by an allocator, and firmware's active address
    // space maps it read-write at its physical address.
    unsafe { pointer.write(handoff) };
    direct(space, phys)
}

/// Loads the new address space and enters the hypervisor.
///
/// One assembly block from the `CR3` load to the jump, because `RSP` moves to a
/// stack the compiler knows nothing about: anything it emitted in between could
/// spill to a stack that is no longer the current one. The return-address slot
/// the convention leaves below `RSP` stays zero — frames are handed out zeroed
/// — so an entry point that returns anyway faults at address zero instead of
/// executing whatever the stack happened to hold.
///
/// # Safety
///
/// `root` must be a PML4 that maps this code, `entry` and `stack`; `stack` must
/// be the top of a stack in it; `entry` must be an `efiapi` function that never
/// returns; and `handoff` must be an address in it that stays valid forever,
/// since nothing ever unmaps it.
unsafe fn enter(root: PhysFrame, stack: VirtAddr, entry: VirtAddr, handoff: VirtAddr) -> ! {
    // SAFETY: the caller guarantees the new address space maps this code, the
    // stack and the entry point. Interrupts are masked first because firmware's
    // interrupt descriptor table stops being the one in charge here, and the
    // hypervisor installs its own before it unmasks. `RCX` is `efiapi`'s first
    // integer argument, so the entry point receives the handoff pointer.
    unsafe {
        asm!(
            "cli",
            "mov cr3, {root}",
            "mov rsp, {stack}",
            "jmp {entry}",
            root = in(reg) root.start_address().as_u64(),
            stack = in(reg) stack.as_u64() - ENTRY_SHADOW_SPACE,
            entry = in(reg) entry.as_u64(),
            in("rcx") handoff.as_u64(),
            options(noreturn),
        )
    }
}

/// A pointer to `phys` through firmware's identity map.
///
/// Going through the paging crate's window rather than casting an integer keeps
/// the alignment check that comes with it: an unaligned pointer into physical
/// memory is a mistake in the caller, not something to paper over.
///
/// # Errors
///
/// [`LoaderError::Paging`] if firmware's identity map does not reach `phys`, or
/// if `phys` is not aligned for `T`.
fn identity_ptr<T>(phys: PhysAddr) -> Result<NonNull<T>, LoaderError> {
    DirectMap::identity()
        .ptr::<T>(phys)
        .ok_or_else(|| unreachable(phys))
}

/// Where the direct map the loader built makes `phys` readable.
///
/// # Errors
///
/// [`LoaderError::Paging`] if the direct map does not reach `phys`, which would
/// mean it was sized from a memory map that does not describe it.
fn direct(space: &AddressSpace, phys: PhysAddr) -> Result<VirtAddr, LoaderError> {
    space
        .direct_map()
        .virt(phys)
        .ok_or_else(|| unreachable(phys))
}

/// The error for a physical address the current window does not cover.
fn unreachable(phys: PhysAddr) -> LoaderError {
    LoaderError::Paging(PagingError::Unreachable {
        phys: phys.as_u64(),
    })
}

/// A byte count as a `usize`.
///
/// # Panics
///
/// Never on this target, where `usize` is as wide as the `u64` the paging
/// subsystem counts bytes in. Ending the boot is the right response if that
/// ever stops holding, and it would happen before the hypervisor runs.
pub(crate) fn bytes(value: u64) -> usize {
    usize::try_from(value).expect("a byte count must fit a usize on this target")
}

/// A count or address as a `u64`.
pub(crate) fn wide(value: usize) -> u64 {
    value as u64
}

/// A count as a `u32`, for the handoff's narrow fields.
///
/// # Panics
///
/// Never for the counts it is used on: a structure size, and a descriptor count
/// the chunk's memory-map region bounds to a few thousand.
fn narrow(value: usize) -> u32 {
    u32::try_from(value).expect("a handoff count must fit a u32")
}
