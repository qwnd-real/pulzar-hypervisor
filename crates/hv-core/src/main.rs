//! The pulzar hypervisor image.
//!
//! Firmware never starts this image. `hv-loader` maps it at a randomized
//! high-half address and jumps to its entry point with a [`Handoff`] in the
//! first argument register, so bring-up begins in an address space that is
//! already half pulzar's own. Making it entirely pulzar's own takes four steps,
//! in this order and for these reasons:
//!
//! 1. Adopt the address space the loader built, along with the frame allocator
//!    and the mapping window it left in the reserved chunk.
//! 2. Evict the loader: unload its image through boot services, then wipe the
//!    pages firmware just freed, so no part of the first stage is left in
//!    memory. This must happen while boot services are still reachable, which
//!    is why it comes before step 4 and not after.
//! 3. Install a GDT, TSS and IDT of our own. Firmware's live in the lower half
//!    and are about to become unreachable, so having our own is a prerequisite
//!    for step 4 rather than a later improvement.
//! 4. Drop the lower half. Boot services, the system table, the loader's former
//!    image and every other firmware address stop existing in one step. From
//!    here the only memory that can be reached is the chunk, this image, and
//!    whatever the direct map and the mapping window describe.
//!
//! Two things are established alongside those steps. A heap comes up as soon as
//! the address space is adopted, because it is a run of the chunk's frames and
//! nothing more, and everything that allocates needs it. The firmware
//! description tables are read last of all, deliberately after step 4: they lie
//! in memory that belongs to firmware and are reached through the direct map,
//! so reading them once every firmware address is gone is what proves that path
//! carries the hypervisor's own weight.
//!
//! `ExitBootServices` is deliberately never called, here or later: pulzar is a
//! pass-through hypervisor and leaves the firmware environment intact for
//! whatever boots after it.
//!
//! The same entry point is also reachable by starting `pulzar.efi` as an
//! ordinary UEFI application, in which case the first argument is a firmware
//! image handle rather than a handoff. That case is detected and refused, never
//! guessed at.

#![no_main]
#![no_std]

mod error;
mod firmware;
mod heap;

use core::{convert::Infallible, ffi::c_void, hint::black_box, panic::PanicInfo};

use acpi::Acpi;
use descriptors::{Descriptors, Interrupt, halt};
use handoff::{Handoff, HandoffError};
use log::{error, info, warn};
use paging::{AddressSpace, CacheType, Existing, PagingError, Protection, chunk};
use uefi_raw::Status;
use x86_64::{PhysAddr, VirtAddr, structures::paging::PhysFrame};

use crate::{error::CoreError, firmware::Firmware, heap::Heap};

/// Bytes of stack the self check writes and reads back after the transition.
const PROBE_BYTES: usize = 256;

/// Byte the stack probe writes. Any value other than zero would do; this one is
/// recognizable in a memory dump and cannot be confused with the zero a page
/// that was never written to reads as.
const PROBE_PATTERN: u8 = 0xA5;

/// Entry point, reached either by the loader's jump or by firmware starting
/// this image as an application.
///
/// `rust-lld` makes `efi_main` this target's PE entry point, which is the
/// address the loader reads out of the image's own headers. The first integer
/// argument is a handoff pointer in one case and an image handle in the other,
/// so telling the two apart happens before anything is trusted.
///
/// A status is only ever returned on the refusal path. Once a handoff is
/// accepted there is nowhere to return to: the loader jumped rather than
/// called, and its stack is gone.
#[unsafe(no_mangle)]
extern "efiapi" fn efi_main(argument: *const c_void) -> Status {
    if serial::init().is_err() {
        // Nothing that happened after this could be reported, so there is no
        // point in continuing.
        return Status::DEVICE_ERROR;
    }
    // SAFETY: `argument` is the first integer argument register, holding either
    // the handoff pointer the loader put there or the image handle firmware
    // passes to an application. Both are readable, and distinguishing them is
    // exactly what this call is for.
    let handoff = match unsafe { Handoff::from_ptr(argument) } {
        Ok(handoff) => handoff,
        Err(error @ HandoffError::NotAHandoff { .. }) => {
            info!("core: {error}");
            error!(
                "core: pulzar.efi is the hypervisor image; boot hv-loader (BOOTX64.EFI) instead"
            );
            return Status::UNSUPPORTED;
        }
        // The magic matched, so the loader jumped here and the two images
        // disagree about the protocol. There is no caller to return a status to.
        Err(error) => {
            error!("core: {error}");
            halt()
        }
    };
    match bring_up(handoff) {
        // `bring_up` only ever returns by failing; its success type is
        // uninhabited.
        Ok(never) => match never {},
        Err(error) => {
            error!("core: bring-up failed: {error}");
            halt()
        }
    }
}

/// Everything from the address space the loader built to one that needs no
/// firmware.
///
/// # Errors
///
/// The first failure of any step. Nothing is rolled back and nothing is
/// retried: a half-adopted address space cannot be handed back to firmware, and
/// the caller halts.
fn bring_up(handoff: &'static Handoff) -> Result<Infallible, CoreError> {
    announce(handoff);

    // SAFETY: the loader built this address space, activated it and jumped here,
    // so it is the live one; its direct map covers the chunk because the loader
    // sized it from the whole physical address space; and no other
    // `AddressSpace` exists in this image.
    let mut space = unsafe { AddressSpace::adopt(&adopted(handoff)?) }?;
    space.describe("core");

    let heap = Heap::establish(&mut space)?;
    heap.describe("core");

    evict_loader(&mut space, handoff)?;
    Descriptors::install(&mut space, unclaimed)?.describe("core");

    // SAFETY: this space is the active one, physical memory is reached through
    // its direct map rather than firmware's identity map, and nothing firmware
    // provided is used after this point: the loader is gone, boot services were
    // only ever reached from `evict_loader`, and the memory map the hypervisor
    // keeps is the copy in the chunk.
    unsafe { space.drop_lower_half() }?;
    info!("core: dropped the firmware half of the address space");

    let acpi = survey_machine(handoff, &space)?;
    heap.describe("core");
    self_check(&space, handoff)?;
    info!(
        "core: bring-up complete, {} processors described, halting",
        acpi.madt().processors().len()
    );
    halt()
}

/// Reads the machine's own description out of firmware's tables.
///
/// Everything the hypervisor will need to know about the hardware that is not
/// discoverable from the processor itself comes from here: how many processors
/// there are, how interrupts reach them, and where PCI Express configuration
/// space is mapped. It is parsed into memory of the hypervisor's own, because
/// the tables themselves stay firmware's and are handed on to whatever boots
/// next.
///
/// # Errors
///
/// [`CoreError::Acpi`] if the tables cannot be read, or if the machine has no
/// MADT — without which its other processors could never be started.
fn survey_machine(handoff: &Handoff, space: &AddressSpace) -> Result<Acpi, CoreError> {
    // SAFETY: `acpi_rsdp` is the address the loader read out of firmware's own
    // configuration table, and `space` is the active address space, whose direct
    // map covers every physical address the memory map described as memory —
    // which includes the ranges firmware keeps its tables in.
    let acpi = unsafe { Acpi::collect(handoff.acpi_rsdp, space.direct_map()) }?;
    acpi.describe("core");
    Ok(acpi)
}

/// Logs the layout the loader described, so the serial log holds both sides of
/// the protocol rather than only the side that wrote it.
fn announce(handoff: &Handoff) {
    info!(
        "core: handoff v{} accepted, {} bytes",
        handoff.version, handoff.size
    );
    info!(
        "core: image at {:#x}, {:#x} bytes; stack at {:#x}, {:#x} bytes",
        handoff.core_image_base, handoff.core_image_size, handoff.stack_base, handoff.stack_size
    );
    info!(
        "core: chunk at {:#x}, {:#x} bytes; pml4 at {:#x}",
        handoff.chunk_base, handoff.chunk_size, handoff.page_table_root
    );
    info!(
        "core: direct map at {:#x}, {:#x} bytes; window at {:#x}, {:#x} bytes",
        handoff.direct_map_base,
        handoff.direct_map_size,
        handoff.mapping_window_base,
        handoff.mapping_window_size
    );
    info!(
        "core: {} memory descriptors at {:#x}, physical memory ends at {:#x}",
        handoff.memory_map_entries, handoff.memory_map, handoff.top_of_ram
    );
    info!("core: acpi root pointer at {:#x}", handoff.acpi_rsdp);
}

/// The address-space description the paging subsystem adopts, out of the boot
/// protocol.
///
/// Every address is validated rather than assumed. The two images are separate
/// files that can be staged independently, and a handoff whose magic and
/// version both check out can still name an address this processor has no way
/// to form.
///
/// # Errors
///
/// [`CoreError::BadAddress`] for a value that is not a representable physical
/// or canonical virtual address, or [`CoreError::Paging`] if the page table
/// root is not frame-aligned.
fn adopted(handoff: &Handoff) -> Result<Existing, CoreError> {
    let root = phys(handoff.page_table_root)?;
    Ok(Existing {
        chunk_base: phys(handoff.chunk_base)?,
        root: PhysFrame::from_start_address(root).map_err(|_| PagingError::Misaligned {
            value: root.as_u64(),
            align: chunk::FRAME_SIZE,
        })?,
        direct_map_base: virt(handoff.direct_map_base)?,
        direct_map_size: handoff.direct_map_size,
        mapping_window_base: virt(handoff.mapping_window_base)?,
    })
}

/// Removes the first stage from the machine entirely.
///
/// Two steps that cannot be reordered. `UnloadImage` calls the loader's own
/// unload handler, which is code inside the loader's image, so the image has to
/// be mapped and executable while that call runs — the wipe can only follow it.
/// Firmware owns those pages again afterwards, and zeroing them is what keeps
/// the loader's code, its relocated pointers into the chunk, and whatever else
/// it left behind from being readable by whatever firmware hands them to next.
///
/// The wipe goes through a temporary mapping of our own rather than firmware's
/// identity alias, whose writability is firmware's choice and not ours, and
/// which stops existing a few steps later in any case.
///
/// # Errors
///
/// [`CoreError::NotATable`] if the handoff's system table does not check out,
/// [`CoreError::Firmware`] if firmware refuses the unload, or
/// [`CoreError::Paging`] if the loader's range cannot be mapped for the wipe.
fn evict_loader(space: &mut AddressSpace, handoff: &Handoff) -> Result<(), CoreError> {
    // SAFETY: the loader passed firmware's own system table pointer, and the
    // lower half of the address space — where it lives — is still mapped.
    let firmware = unsafe { Firmware::adopt(handoff.system_table) }?;
    // SAFETY: the handle is the one firmware gave the loader for its own image,
    // the loader registered an unload handler before publishing the handoff, and
    // execution left the loader for good at the jump into this image. Nothing
    // here points into it: the handoff lives in the reserved chunk.
    unsafe { firmware.unload_image(handoff.loader_image_handle) }?;
    info!("core: unloaded the loader's image");

    let base = phys(handoff.loader_image_base)?;
    let size = handoff.loader_image_size;
    let wipe = |virt: VirtAddr| {
        // SAFETY: `with_physical` maps `size` writable bytes at `virt` for the
        // duration of this call, and the closure is the only holder of that
        // address.
        unsafe { virt.as_mut_ptr::<u8>().write_bytes(0, bytes(size)) };
    };
    // SAFETY: firmware released this range in the call above and nothing has
    // allocated since, so nothing else maps it. It is page-aligned and a whole
    // number of pages long, because the loader rounded it up within an
    // allocation firmware made in pages.
    unsafe {
        space.with_physical(
            base,
            size,
            Protection::ReadWrite,
            CacheType::WriteBack,
            wipe,
        )
    }?;
    info!("core: wiped {size:#x} bytes of loader image at {base:#x}");
    Ok(())
}

/// Proves the surviving address space still does the three things everything
/// after this depends on.
///
/// A read through the direct map, a write to the stack, and a walk of our own
/// page tables. Dropping one page table entry too many would have broken at
/// least one of them, and finding that out here — with serial working and a
/// handler installed for every exception — beats finding it out as a fault
/// somewhere with no context left.
///
/// # Errors
///
/// [`CoreError::SelfCheckFailed`] naming whichever check did not hold, or
/// [`CoreError::BadAddress`] if the image base in the handoff is not canonical.
fn self_check(space: &AddressSpace, handoff: &Handoff) -> Result<(), CoreError> {
    // `black_box` forces a real load: the magic was already read before the
    // transition, and a cached value would make this check prove nothing.
    if black_box(handoff).magic != Handoff::MAGIC {
        return Err(CoreError::SelfCheckFailed {
            what: "reading the chunk through the direct map",
        });
    }

    let mut probe = [0_u8; PROBE_BYTES];
    probe.fill(PROBE_PATTERN);
    if black_box(&probe).iter().any(|byte| *byte != PROBE_PATTERN) {
        return Err(CoreError::SelfCheckFailed {
            what: "writing to the stack",
        });
    }

    let image = virt(handoff.core_image_base)?;
    let mapped = space.translate(image).ok_or(CoreError::SelfCheckFailed {
        what: "translating this image's own base",
    })?;
    info!("core: self check passed, {image:#x} still translates to {mapped:#x}");

    Ok(())
}

/// What becomes of an interrupt no handler claimed.
///
/// This is where an interrupt will be handed to a guest, once there is a guest
/// to hand it to. Until then the two kinds go opposite ways, and what separates
/// them is whether returning resolves anything.
///
/// An external interrupt is something the machine was already doing before
/// pulzar existed — firmware's timer is the one that arrives first — and
/// returning from it drops it, which is the whole of what can be done with an
/// interrupt that has no owner yet. It is still reported: an unexpected vector
/// is the hypervisor learning something about the machine it is on.
///
/// An exception cannot be dropped. The processor raised it about the very
/// instruction it would go back to, so returning without changing anything
/// raises it again, and again. At this stage it also means an invariant of the
/// address space or of the descriptor tables is already broken, which is not
/// something to continue past: it stops here, with everything the processor
/// said about it on the record.
fn unclaimed(interrupt: &Interrupt) {
    if interrupt.vector().is_exception() {
        error!("core: {interrupt}");
        halt()
    }
    warn!("core: ignoring unclaimed {interrupt}");
}

/// A physical address out of the boot protocol.
///
/// # Errors
///
/// [`CoreError::BadAddress`] if the value has bits set above the physical
/// address space.
fn phys(value: u64) -> Result<PhysAddr, CoreError> {
    PhysAddr::try_new(value).map_err(|_| CoreError::BadAddress { value })
}

/// A virtual address out of the boot protocol.
///
/// # Errors
///
/// [`CoreError::BadAddress`] if the value is not canonical.
fn virt(value: u64) -> Result<VirtAddr, CoreError> {
    VirtAddr::try_new(value).map_err(|_| CoreError::BadAddress { value })
}

/// A byte count as a `usize`.
///
/// # Panics
///
/// Never on this target, where `usize` is as wide as the `u64` the boot
/// protocol counts bytes in. This runs during bring-up, before any guest
/// exists, so stopping is the correct response if that ever stops holding.
fn bytes(value: u64) -> usize {
    usize::try_from(value).expect("a byte count must fit a usize on this target")
}

/// Logs a panic and stops.
///
/// Panics are confined to bring-up by design — everything on the eventual guest
/// path returns errors instead — so reaching here means an invariant broke
/// before any guest existed, and there is nothing to hand control back to.
#[panic_handler]
fn panic(info: &PanicInfo<'_>) -> ! {
    error!("core: panic: {info}");
    halt()
}
