//! First-stage UEFI loader for the pulzar hypervisor.
//!
//! Firmware starts this image; it ends by jumping into the hypervisor image
//! with an address space of pulzar's own making. In between it does six
//! things, in this order and for these reasons:
//!
//! 1. Captures the state firmware was running with, before anything else at
//!    all. Every register in it is one the steps below overwrite, and none can
//!    be read back afterwards.
//! 2. Reserves the one chunk of physical memory pulzar will own. It has to
//!    happen before the memory map is captured, so the chunk appears in the map
//!    as reserved rather than as memory something might hand out again.
//! 3. Loads the guest image without starting it, retaining the resulting image
//!    handle for the guest portal.
//! 4. Captures firmware's memory map into the chunk, because the address space
//!    is sized from it and because firmware's own copy lives in memory the
//!    hypervisor will lose.
//! 5. Loads, relocates and maps the hypervisor image at a randomized high-half
//!    address, alongside a guarded stack, a direct map of physical memory and a
//!    window for explicit mappings.
//! 6. Publishes a [`Handoff`] describing all of it, with the captured firmware
//!    state beside it in the chunk, and jumps with the new address space
//!    active.
//!
//! The address space it activates is deliberately half firmware's: the lower
//! half is copied from firmware's own tables so the hypervisor can take a
//! coherent snapshot before discarding those host mappings. The guest keeps
//! firmware's original page tables and reaches them through nested paging.
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
use log::{error, info, warn};
use paging::{
    AddressSpace, CacheType, DirectMap, PagingError, Protection, Ram, Stack, buddy, chunk,
    kaslr::{self, Entropy, Placement},
};
use snapshot::FirmwareContext;
use uefi::{Status, boot::MemoryDescriptor, entry};
use x86_64::{
    PhysAddr, VirtAddr,
    structures::paging::{PageSize, PhysFrame, Size2MiB},
};

use crate::{
    error::LoaderError,
    firmware::{ImageFile, Loader, Memory, Reserved, Survey},
    image::{Image, ImageError},
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

/// Pages of the hypervisor's initial stack: 128 KiB, with an unmapped guard
/// page below and above.
///
/// Sized for what composing the guest costs rather than for what the code looks
/// like it should cost. The one guest is a several-kilobyte value — its
/// description of the guest's memory holds the region set inline, and the set
/// of devices answering for those regions holds a slot per region name — and it
/// is built, returned and moved into the cell that keeps it, which in a build
/// with no inlining is that value again in every frame of the chain. Measured
/// at sixty-nine kilobytes on the deepest of them, against the sixty-four this
/// used to be: the overflow landed on the guard page below, which is the one
/// way it could have been anything but silent.
const CORE_STACK_PAGES: u64 = 32;

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

/// The captured firmware state goes in the region beside it, on the same terms.
const _: () = assert!(
    size_of::<FirmwareContext>() as u64 <= chunk::FIRMWARE_CONTEXT_SIZE,
    "the firmware context does not fit the chunk region reserved for it"
);

/// The memory-map region holds the loader's decoded copy of firmware's
/// descriptors, one [`MemoryDescriptor`] per entry, and both images read them
/// by this same layout. Forty bytes is the UEFI version-1 descriptor size (a
/// `u32` type and four `u64` fields); a UEFI binding upgrade that changes the
/// struct must bump [`chunk::LAYOUT`] before either image may adopt the map,
/// and this assert turns forgetting that into a build failure instead of a
/// misread map.
const _: () = assert!(
    size_of::<MemoryDescriptor>() == 40,
    "the UEFI memory descriptor layout changed; bump chunk::LAYOUT and re-check both images"
);

/// Firmware's entry point.
///
/// The capture comes before serial, and serial before everything else. Serial
/// is first because everything after it, including a failure, has to be
/// observable — it is the one step whose error is a bare status, since there is
/// nowhere to report a serial failure to. The capture is ahead of even that
/// because bringing a serial port up reprograms one, and a snapshot of firmware
/// taken after pulzar has changed something is a snapshot of pulzar.
#[entry]
fn main() -> Status {
    // SAFETY: firmware's address space is the active one — nothing has run that
    // could have changed it — and in it a physical address is its own virtual
    // address, which is exactly what the identity window describes. The
    // descriptor tables read here are the ones the processor is running on.
    let firmware = unsafe { snapshot::capture(DirectMap::identity()) };
    // Not fatal, and for the reason the hypervisor image gives at its own call:
    // a machine with no output is one nothing can be reported from rather than
    // one that must not boot. Every record below is discarded while no logger is
    // installed, and the loader's work does not depend on any of them.
    let _ = serial::init();
    // Before the first record, deliberately: with the screen backend compiled
    // in this is what puts every line that follows on the display, and a boot
    // that freezes anywhere later is a boot whose last painted line says where
    // it stopped. The identity map still stands, so the bytes are reachable at
    // their physical address and need no mapping of ours.
    let screen = firmware::framebuffer();
    serial::offer_screen(&screen, screen.base);
    info!("loader: pulzar hv-loader starting");
    if screen.usable() {
        info!(
            "loader: console frame buffer at {:#x}, {}x{}, logging attached",
            screen.base, screen.width, screen.height
        );
    } else {
        warn!("loader: no usable console screen; logging stays on the ports");
    }
    firmware.describe("loader");
    match boot(&firmware, screen) {
        // `boot` only ever returns by failing; its success type is uninhabited.
        Ok(never) => match never {},
        Err(error) => {
            error!("loader: boot failed: {error}");
            error.status()
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
fn boot(
    firmware: &FirmwareContext,
    screen: handoff::Framebuffer,
) -> Result<Infallible, LoaderError> {
    // Read before anything slow, so the gap between this reading and the
    // hypervisor's own clock coming up is as small as the loader can make it.
    let boot_wall = firmware::wall_clock();
    let loader = firmware::loaded_self()?;
    info!(
        "loader: own image at {:#x}, {:#x} bytes",
        loader.base, loader.size
    );

    let reserved = firmware::reserve()?;
    let chunk_base = reserved.chunk;
    info!(
        "loader: reserved chunk at {chunk_base:#x}, {:#x} bytes, temporary boot-services-data \
         trampoline at {:#x}",
        chunk::CHUNK_SIZE,
        reserved.trampoline
    );

    let guest = firmware::load_guest()?;
    info!(
        "loader: guest image loaded as {:#x}",
        wide(guest.handle.as_ptr() as usize)
    );

    let survey = survey_memory(chunk_base)?;
    let memory = survey.memory;
    info!(
        "loader: copied {} memory descriptors in {} runs, physical memory ends at {:#x}",
        memory.entries,
        survey.ram.len(),
        memory.top_of_ram
    );

    let mut file = ImageFile::open()?;
    let image = probe(&mut file)?;
    info!(
        "loader: hypervisor image spans {:#x} bytes in {} sections",
        image.size(),
        image.sections().len()
    );

    let placement = kaslr::place(
        &mut entropy(),
        image.size(),
        paging::direct_map_size(memory.top_of_ram)?,
    )?;
    info!(
        "loader: image at {:#x}, direct map at {:#x}, window at {:#x}, entropy {:?}",
        placement.image_base,
        placement.direct_map_base,
        placement.mapping_window_base,
        placement.source
    );
    // SAFETY: `chunk_base` is the base of a `CHUNK_SIZE` region firmware just
    // reserved for this image alone, which firmware's still-active address space
    // maps identically and which the survey describes as memory; the runs come
    // from firmware's own map and describe RAM rather than device apertures; and
    // nothing of pulzar's has been mapped or established on this processor yet.
    let mut space = unsafe { AddressSpace::build(chunk_base, &survey.ram, placement) }?;
    let hypervisor = load(&mut space, &mut file, &image, placement)?;

    // Firmware objects are closed here rather than dropped: the jump below never
    // returns, so no destructor at the end of this function would ever run.
    drop(file);
    space.describe("loader");

    // The stack's top is read out before the inputs are assembled: the stack
    // itself moves into them, and it is not the sort of value that can be left
    // behind in two places at once.
    let stack_top = hypervisor.stack.top();
    let inputs = HandoffInputs {
        reserved,
        loader,
        guest,
        memory,
        hypervisor,
        placement,
        boot_wall,
        screen,
    };
    let handoff = publish(&space, inputs, firmware)?;
    let entry = VirtAddr::new(image.entry(placement.image_base.as_u64()));
    info!(
        "loader: entering hypervisor at {entry:#x}, stack top {stack_top:#x}, handoff {handoff:#x}"
    );
    // SAFETY: `space` maps the hypervisor's image at `entry` and its stack below
    // `stack.top()`, and its lower half is firmware's own, so the instructions
    // between the `CR3` load and the jump stay mapped where they are executing
    // from. `handoff` is a direct-map address of a `Handoff` this space maps, and
    // the entry point of a UEFI image never returns to its caller.
    unsafe { enter(space.root(), stack_top, entry, handoff) }
}

/// Copies firmware's memory map into the chunk's metadata region.
///
/// # Errors
///
/// [`LoaderError::Paging`] if firmware's identity map does not reach the chunk,
/// [`LoaderError::NotInMemoryMap`] if the map does not describe the chunk, or
/// whatever [`firmware::capture_memory_map`] reports.
fn survey_memory(chunk_base: PhysAddr) -> Result<Survey, LoaderError> {
    let capacity = bytes(chunk::MEMORY_MAP_SIZE) / size_of::<MemoryDescriptor>();
    let destination = identity_ptr::<MemoryDescriptor>(chunk_base + chunk::MEMORY_MAP_OFFSET)?;
    // SAFETY: the destination is the chunk's memory-map region, `MEMORY_MAP_SIZE`
    // bytes long and owned by nothing else — no allocator ever hands out the
    // metadata frames — so it holds exactly `capacity` descriptors.
    let survey = unsafe { firmware::capture_memory_map(destination, capacity) }?;
    verify_resident(&survey.ram, chunk_base, chunk::CHUNK_SIZE)?;
    Ok(survey)
}

/// Requires a reserved region to be covered by firmware's RAM description.
///
/// The direct map is built over the RAM runs [`firmware::capture_memory_map`]
/// distilled from firmware's map, and the hypervisor reaches the chunk only
/// through that map once firmware's identity map is gone. Firmware that omits
/// a reserved descriptor from the map would otherwise leave the chunk silently
/// outside the direct map, to fail much later and less readably.
fn verify_resident(ram: &[Ram], start: PhysAddr, len: u64) -> Result<(), LoaderError> {
    let start = start.as_u64();
    let end = start + len;
    if ram
        .iter()
        .any(|run| run.start() <= start && end <= run.end())
    {
        return Ok(());
    }
    Err(LoaderError::NotInMemoryMap { start, len })
}

/// Reads the front of the hypervisor image and parses its headers.
///
/// # Errors
///
/// [`LoaderError::Firmware`] if the read fails, [`LoaderError::ProbeTooSmall`]
/// if the image's headers do not fit the probe buffer, or
/// [`LoaderError::Image`] if the image is not one this loader can place.
fn probe(file: &mut ImageFile) -> Result<Image, LoaderError> {
    let mut headers = [0; PROBE_BYTES];
    let read = file.read_at(0, &mut headers)?;
    Image::parse(&headers[..read]).map_err(|error| {
        if matches!(error, ImageError::Truncated { .. }) && read == headers.len() {
            LoaderError::ProbeTooSmall {
                probe_bytes: headers.len(),
            }
        } else {
            error.into()
        }
    })
}

/// What placing the hypervisor image produced.
///
/// The two facts the handoff needs and nothing else has: how far the image
/// spans, and where its stack ended up. Together rather than separately because
/// they are one step's output, and because the handoff is assembled from a
/// bounded list of such outputs.
///
/// Not `Copy`, because a [`Stack`] is not: the value that names a stack is the
/// only thing entitled to give it back, and a second copy of it would be a
/// second entitlement.
#[derive(Debug)]
struct Hypervisor {
    stack: Stack,
    image_size: u64,
}

/// Values collected during loading that become the immutable handoff.
#[derive(Debug)]
struct HandoffInputs {
    reserved: Reserved,
    loader: Loader,
    guest: firmware::GuestImage,
    memory: Memory,
    hypervisor: Hypervisor,
    placement: Placement,
    boot_wall: Option<Wall>,
    screen: handoff::Framebuffer,
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
) -> Result<Hypervisor, LoaderError> {
    let order = image_order(image.size());
    let span = chunk::FRAME_SIZE << order;
    let phys = space.frames().allocate(order)?.start_address();
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
    Ok(Hypervisor {
        stack,
        image_size: image.size(),
    })
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

/// Writes the handoff and the captured firmware state into the chunk, and
/// returns the address the hypervisor receives the handoff at.
///
/// Both are written here because both are protocol rather than allocation: the
/// loader is the only thing that can produce either, and the hypervisor is the
/// only thing that reads them. The addresses are direct-map ones, so they stay
/// valid after the hypervisor drops the firmware half of the address space.
///
/// # Errors
///
/// [`LoaderError::Firmware`] if the system table cannot be located, or
/// [`LoaderError::Paging`] if a window does not reach the chunk.
fn publish(
    space: &AddressSpace,
    inputs: HandoffInputs,
    firmware: &FirmwareContext,
) -> Result<VirtAddr, LoaderError> {
    let system_table = uefi::table::system_table_raw().ok_or(LoaderError::Firmware {
        operation: "locate the UEFI system table",
        status: Status::NOT_FOUND,
    })?;
    let HandoffInputs {
        reserved,
        loader,
        guest,
        memory,
        hypervisor,
        placement,
        boot_wall,
        screen,
    } = inputs;
    let chunk_base = reserved.chunk;
    let context = chunk_base + chunk::FIRMWARE_CONTEXT_OFFSET;
    let handoff = Handoff {
        magic: Handoff::MAGIC,
        version: Handoff::VERSION,
        size: narrow(size_of::<Handoff>()),
        system_table: system_table.as_ptr(),
        loader_image_handle: loader.handle.as_ptr(),
        loader_image_base: loader.base,
        loader_image_size: loader.size,
        guest_image_handle: guest.handle.as_ptr(),
        chunk_base: chunk_base.as_u64(),
        chunk_size: chunk::CHUNK_SIZE,
        chunk_layout: chunk::LAYOUT,
        page_table_root: space.root().start_address().as_u64(),
        direct_map_base: space.direct_map().base().as_u64(),
        direct_map_size: space.direct_map().size(),
        mapping_window_base: space.mapping_window().as_u64(),
        mapping_window_size: chunk::MAPPING_WINDOW_SIZE,
        core_image_base: placement.image_base.as_u64(),
        core_image_size: hypervisor.image_size,
        stack_base: hypervisor.stack.bottom().as_u64(),
        stack_size: hypervisor.stack.pages() * chunk::FRAME_SIZE,
        memory_map: direct(space, chunk_base + chunk::MEMORY_MAP_OFFSET)?.as_u64(),
        memory_map_entries: narrow(memory.entries),
        memory_map_entry_size: narrow(size_of::<MemoryDescriptor>()),
        top_of_ram: memory.top_of_ram,
        acpi_rsdp: firmware::acpi_rsdp(),
        // Read at the start of the boot, before the slow image and page table
        // work, so the unmeasured gap between this reading and the
        // hypervisor's clock coming up is as small as the loader can make it.
        boot_wall_nanos: boot_wall.map_or(0, Wall::nanos),
        ap_trampoline_base: reserved.trampoline.as_u64(),
        firmware_context: direct(space, context)?.as_u64(),
        framebuffer: screen,
    };

    let pointer = identity_ptr::<FirmwareContext>(context)?;
    // SAFETY: the firmware-context region is set aside for exactly this and is
    // never handed out by an allocator — the metadata frames belong to no one —
    // and firmware's active address space maps it read-write at its physical
    // address.
    unsafe { pointer.write(*firmware) };

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
    Ok(DirectMap::identity().ptr::<T>(phys)?)
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
        .ok_or(LoaderError::Paging(PagingError::Unreachable {
            phys: phys.as_u64(),
            len: 1,
        }))
}

/// The source the high-half layout is drawn from.
///
/// Hardware entropy where the processor has it, and otherwise a layout that
/// merely differs between boots — chosen here, deliberately, rather than
/// substituted underneath the caller. A machine or an emulator without `RDRAND`
/// still gets a moving layout, and the log says in as many words that it is not
/// one an attacker cannot predict.
fn entropy() -> Entropy {
    Entropy::secure().unwrap_or_else(|error| {
        warn!("loader: {error}; the high-half layout will not be secure against an attacker");
        Entropy::best_effort()
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
