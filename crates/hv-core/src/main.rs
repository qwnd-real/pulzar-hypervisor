//! The pulzar hypervisor image.
//!
//! Firmware never starts this image. `hv-loader` maps it at a randomized
//! high-half address and jumps to its entry point with a [`Handoff`] in the
//! first argument register. The core adopts that address space, establishes
//! host descriptor tables and virtualization state, then drops firmware's
//! identity-mapped half.
//!
//! The boot processor enters a guest initialized from the firmware snapshot.
//! Its first instruction is a two-page portal that starts the already-loaded
//! operating-system boot manager and wraps `ExitBootServices`. Only after the
//! original firmware service returns success does the wrapper notify the host,
//! which starts the application processors in their temporary wait loop. The
//! portal pages are the only hypervisor-owned pages ever visible to the guest,
//! and only until the guest has left them behind — nested paging presents the
//! rest of the reserved chunk as an immutable zero page throughout, and the
//! portal as one too once it has been taken back.
//!
//! The same entry point is also reachable by starting `pulzar.efi` as an
//! ordinary UEFI application, in which case the first argument is a firmware
//! image handle rather than a handoff. That case is detected and refused, never
//! guessed at.

#![no_main]
#![no_std]

mod error;
mod heap;

use core::{convert::Infallible, ffi::c_void, hint::black_box, panic::PanicInfo};

use acpi::Acpi;
use apic::Apic;
use clock::{Clock, Wall};
use descriptors::{Descriptors, Interrupt, Tables, Vector, halt};
use exits::{Boot, Exits};
use handoff::{Handoff, HandoffError};
use log::{error, info, warn};
use npt::Exposure;
use paging::{AddressSpace, Existing, PagingError, chunk};
use partition::Partition;
use pci::Pci;
use portal::Portal;
use snapshot::FirmwareContext;
use spin::Once;
use svm::intercept::{Intercepts1, Intercepts2Flags};
use uefi_raw::Status;
use vcpu::Vcpu;
use x86_64::{PhysAddr, VirtAddr, structures::paging::PhysFrame};

use crate::{error::CoreError, heap::Heap};

/// Bytes of stack the self check writes and reads back after the transition.
const PROBE_BYTES: usize = 256;

/// Microseconds the clock is asked to wait for once it is up, as a check that
/// the delay it produces is the delay that was asked for.
const CLOCK_PROBE_MICROS: u64 = 1000;

/// Byte the stack probe writes. Any value other than zero would do; this one is
/// recognizable in a memory dump and cannot be confused with the zero a page
/// that was never written to reads as.
const PROBE_PATTERN: u8 = 0xA5;

/// `EFER.SVME`, required in the guest save area even though guest SVM use is
/// intercepted and hidden from CPUID.
const EFER_SVME: u64 = 1 << 12;

/// Stack alignment required at a Microsoft x64 call site.
const CALL_STACK_ALIGN: u64 = 16;

/// The one guest this hypervisor runs.
///
/// Established on the boot processor before any other is started, and reached
/// by all of them afterwards: what it holds — the description of the guest's
/// memory and the tag its cached translations carry — is shared by every
/// processor that runs the guest, and none of them owns it.
static PARTITION: Once<Partition> = Once::new();

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

    // What becomes of an unclaimed interrupt is one answer for the machine;
    // descriptor tables are one set per processor. Saying the first is what
    // makes the second allowed.
    descriptors::adopt(unclaimed)?;
    Tables::build(&mut space)?.activate()?.describe("core");
    emulate::install()?;

    // SAFETY: this space is the active one, physical memory is reached through
    // its direct map rather than firmware's identity map, and nothing firmware
    // provided is used by the host after this point. Guest firmware remains in
    // physical memory and is reached only through nested translation; the
    // memory map the hypervisor keeps is the copy in the chunk.
    unsafe { space.drop_lower_half() }?;
    info!("core: dropped the firmware half of the address space");

    // Logged on this side of the transition rather than the other, because that
    // is what proves the capture survived it: nothing firmware left is
    // addressable any more, and these numbers come out of the chunk.
    inherited(handoff)?.describe("core");

    let acpi = survey_machine(handoff, &space)?;
    start_clock(&mut space, &acpi, handoff)?;

    // The roster first, because everything below it is sized by how many
    // processors firmware described; then the interrupt controllers, which is
    // where this processor learns what it is called; then a block of its own,
    // which cannot come before its descriptor tables because loading a segment
    // selector into `GS` zeroes the base a block is reached through.
    cpu::survey(acpi.madt().processors())?;
    let apic = Apic::install(&mut space, acpi.madt())?;
    apic.describe("core");
    cpu::attach(apic::local()?.id()?)?;
    ipi::install()?;

    // After the block, because enabling virtualization snapshots host state that
    // includes the `GS` base a block is reached through, and before any other
    // processor is started, because each of them joins a guest that has to
    // already exist.
    let partition = PARTITION.try_call_once(|| Partition::establish(&mut space))?;
    partition.describe("core");
    let portal = Portal::place(space.direct_map(), handoff)?;
    partition.expose(
        &mut space,
        portal.entry(),
        chunk::FRAME_SIZE,
        Exposure::ReadExecute,
    )?;
    partition.expose(
        &mut space,
        portal.parameters(),
        chunk::FRAME_SIZE,
        Exposure::ReadOnly,
    )?;
    let mut vcpu = virtualize(&mut space, inherited(handoff)?, portal.entry())?;

    // Last of the subsystems that take the address space by value, and
    // deliberately so. It maps and releases a range per bus, which costs nothing
    // while this is the only processor running and an interprocessor interrupt
    // per processor per release once the others are up — so it belongs before
    // `apic::start` — and it is by far the largest consumer of the mapping
    // window, so everything the machine needs to run has already taken its share.
    Pci::install(&mut space, &acpi)?.describe("core");

    // The last use of the address space as a value. From here it belongs to the
    // machine rather than to this function, and every processor reaches the same
    // one through the same lock.
    paging::adopt(space)?;

    heap.describe("core");
    cpu::describe("core");
    ipi::describe("core");
    paging::with(|space| self_check(space, handoff))??;
    info!("core: host bring-up complete, entering the firmware guest");
    run_guest(&mut vcpu, partition, portal, handoff)
}

/// Enters the guest on the boot processor and stays in its exits until one of
/// them ends the guest.
///
/// # Errors
///
/// [`CoreError::Exit`] with whichever exit the guest stopped at, or with the
/// rule the processor refused its control block for.
fn run_guest(
    vcpu: &mut Vcpu,
    partition: &'static Partition,
    portal: Portal,
    handoff: &Handoff,
) -> Result<Infallible, CoreError> {
    let mut exits = Exits::new(
        partition,
        portal,
        Boot {
            trampoline: PhysAddr::new_truncate(handoff.ap_trampoline_base),
            attach: ap_main,
        },
    );
    // SAFETY: this VCPU was created on this processor and has stayed on it, its
    // control block has not moved, and its save area holds the captured firmware
    // state with the portal as its first instruction — a guest this hypervisor
    // built and is entitled to run.
    Ok(unsafe { exits.run(vcpu) }?)
}

/// What every processor other than the boot processor runs, for good.
///
/// Reached from the trampoline with nothing but a stack and this address, so
/// the order is forced. Descriptor tables first, because until they are loaded
/// there is no way for this processor to report anything going wrong — a fault
/// before that point is a triple fault and a reset machine. Then its own
/// interrupt controller, which is what its identifier comes from, and only then
/// a block of its own and the interrupts that reach it.
fn ap_main() -> ! {
    let descriptors = match attach() {
        Ok((id, descriptors)) => {
            info!("core: {id} online");
            descriptors
        }
        Err(error) => {
            error!("core: an application processor could not come up: {error}");
            halt()
        }
    };
    // SAFETY: nothing can deliver below the first external vector. The boot
    // processor masked the legacy controllers before it started this one, and it
    // did so before anything unmasked; every other source on this machine is one
    // this hypervisor programmed, and a vector for one of those is only ever
    // handed out by `descriptors::claim`, which refuses the architecture's own.
    if let Err(error) = unsafe { descriptors.unmask() } {
        error!("core: an application processor could not take interrupts: {error}");
        halt()
    }
    loop {
        core::hint::spin_loop();
    }
}

/// Everything an application processor does before it is one of the machine's.
///
/// # Errors
///
/// The first failure of any step. There is nothing to roll back: a processor
/// that cannot finish this has nothing to go back to, and the caller stops it.
fn attach() -> Result<(cpu::ApicId, Descriptors), CoreError> {
    // Built while the address space is locked and switched to after it is
    // unlocked: a processor that fell over between the two would otherwise leave
    // that lock held for every processor after it.
    let descriptors = paging::with(Tables::build)??.activate()?;
    let id = apic::LocalApic::enable()?.id()?;
    cpu::attach(id)?;
    let host = paging::with(|space| {
        let window = space.direct_map();
        // SAFETY: this processor installed its descriptors and attached just
        // above, and it changes none of the host state `Host::install` captures.
        unsafe { vcpu::Host::install(space.frames(), window) }
    })??;
    host.describe("core");
    Ok((id, descriptors))
}

/// Turns virtualization on for the calling processor and gives it a place in
/// the guest.
///
/// Last of the per-processor steps, and it has to be: enabling the extension
/// takes a snapshot of the host state a world switch does not restore by itself
/// — the task register, the `GS` base — and both of those are established by
/// the two steps before it. A snapshot taken any earlier would be restored on
/// every exit, faithfully, and be wrong.
///
/// The control block this produces holds no guest state: no instruction
/// pointer, no stack pointer, no segments, nothing to run. So it is built,
/// reported on and handed straight back, which exercises every step of the path
/// — the chunk finding a page, the window reaching it, the block being
/// programmed, the entry rules being checked — without leaving a page allocated
/// for a guest that does not exist yet. Its report says in as many words that
/// it would not be entered, and which rule says so.
///
/// # Errors
///
/// [`CoreError::NoPartition`] if this processor came up before the guest
/// existed, [`CoreError::Vcpu`] if the extension cannot be enabled — a
/// processor without it, or firmware having turned it off — or
/// [`CoreError::Partition`] if the chunk cannot back a control block.
fn virtualize(
    space: &mut AddressSpace,
    firmware: &FirmwareContext,
    entry: PhysAddr,
) -> Result<Vcpu, CoreError> {
    let partition = PARTITION.get().ok_or(CoreError::NoPartition)?;
    let window = space.direct_map();
    // SAFETY: this processor has installed its descriptor tables and attached,
    // so the task register and the `GS` base hold what it will keep using;
    // nothing in this image changes either afterwards, nor any fast-system-call
    // register. Every processor reaches this once, on its own way up.
    let host = unsafe { vcpu::Host::install(space.frames(), window) }?;
    host.describe("core");

    let mut vcpu = partition.attach(host, space)?;
    *vcpu.save_mut() = firmware.cpu;
    vcpu.save_mut().rip = entry.as_u64();
    vcpu.save_mut().rsp &= !(CALL_STACK_ALIGN - 1);
    vcpu.save_mut().efer |= EFER_SVME;
    vcpu.control_mut().intercept_1 |= Intercepts1::CPUID;
    vcpu.control_mut().intercept_2 = vcpu
        .control()
        .intercept_2
        .with_flags(Intercepts2Flags::VMMCALL);
    vcpu.describe("core");
    Ok(vcpu)
}

/// Establishes the timebase from whichever counter the machine turned out to
/// have.
///
/// It comes after the firmware tables because it is built out of them — where
/// the event timer is, and what its fallback would be — and after the lower
/// half is gone because the counter is reached through a mapping of pulzar's
/// own, like everything else from here on.
///
/// The wall-clock reading in the boot protocol is what the clock counts forward
/// from. Zero is how the loader says firmware would not give it one, which is a
/// hypervisor with a monotonic clock and no dates, not a failure.
///
/// # Errors
///
/// [`CoreError::Clock`] if the machine describes no counter of known rate, if
/// nothing here can keep time on it, or if the counter's registers cannot be
/// reached.
fn start_clock(
    space: &mut AddressSpace,
    acpi: &Acpi,
    handoff: &Handoff,
) -> Result<Clock, CoreError> {
    let boot = (handoff.boot_wall_nanos != 0).then(|| Wall::from_nanos(handoff.boot_wall_nanos));
    let clock = Clock::install(space, acpi, boot)?;
    clock.describe("core");

    // A measured delay, because a clock that is out by an order of magnitude
    // still logs a plausible frequency, and the first thing that will depend on
    // this is a bring-up delay that has to be real.
    let before = clock.now();
    clock.sleep_micros(CLOCK_PROBE_MICROS);
    info!(
        "core: clock slept {CLOCK_PROBE_MICROS} us, measured {} ns",
        (clock.now() - before).as_nanos()
    );
    Ok(clock)
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
    match (handoff.boot_wall_nanos != 0).then(|| Wall::from_nanos(handoff.boot_wall_nanos)) {
        Some(wall) => info!("core: firmware's clock read {wall} during the loader"),
        None => info!("core: the loader got no wall-clock time from firmware"),
    }
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
fn self_check(space: &mut AddressSpace, handoff: &Handoff) -> Result<(), CoreError> {
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
///
/// Dropping one also means acknowledging it. The local interrupt controller
/// holds an interrupt in service until it is told otherwise, and goes on
/// refusing everything of that priority or lower until it is — so a processor
/// that ignored one without saying so would quietly stop accepting a whole
/// class of interrupts for the rest of its life. Reinjecting into a guest is
/// what will take this over, and acknowledging is part of that too. The
/// non-maskable interrupt is the one arrival that is not acknowledged: nothing
/// holds it in service, and an acknowledgement it did not need would end
/// whatever interrupt actually is.
///
/// # What this may not do
///
/// It is entered from the interrupt path, so it takes no lock the interrupted
/// code could be holding. For everything masking holds off, [`log`] is such a
/// lock and is safe to take, because the backend masks interrupts for a whole
/// line. For the two kinds that arrive anyway — an exception, and the
/// non-maskable interrupt — it is not, and the report goes straight to the port
/// through [`serial::emergency`] instead.
fn unclaimed(interrupt: &Interrupt) {
    if interrupt.vector().is_exception() {
        serial::emergency(format_args!("core: {interrupt}"));
        halt()
    }
    if interrupt.vector() == Vector::NON_MASKABLE {
        serial::emergency(format_args!("core: ignoring unclaimed {interrupt}"));
        return;
    }
    warn!("core: ignoring unclaimed {interrupt}");
    if let Err(error) = apic::end_of_interrupt() {
        error!(
            "core: could not acknowledge {}: {error}",
            interrupt.vector()
        );
    }
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

/// The state firmware was running with, out of the chunk.
///
/// The loader captured it before it had modified anything and left it beside
/// the boot protocol, because by now there is nowhere else it could come from:
/// the registers it describes hold pulzar's values, and firmware's own copies
/// of what it does not hold are in memory that is no longer addressable.
///
/// # Errors
///
/// [`CoreError::BadAddress`] if the handoff names an address this processor
/// cannot form.
fn inherited(handoff: &Handoff) -> Result<&'static FirmwareContext, CoreError> {
    let address = virt(handoff.firmware_context)?;
    // SAFETY: the loader wrote a `FirmwareContext` here, into the chunk region
    // set aside for exactly that, which no allocator hands out and nothing ever
    // frees — so the reference cannot dangle and `'static` is honest. The
    // address is a direct-map one, which is what keeps it mapped now that the
    // firmware half of the address space is gone.
    Ok(unsafe { &*address.as_ptr::<FirmwareContext>() })
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
