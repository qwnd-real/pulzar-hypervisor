//! Everything the loader asks of firmware.
//!
//! Boot services are only available while the loader runs, so every use of them
//! is gathered here: the loader's own image, the reserved chunk, the hypervisor
//! image's file, and firmware's memory map. Nothing in this module knows how a
//! page table or a PE image is built, and nothing outside it calls a boot
//! service.
//!
//! Firmware objects are closed explicitly rather than dropped, because the
//! loader ends by jumping into the hypervisor and never returns: no destructor
//! at the end of `main` will ever run.

extern crate alloc;

use alloc::vec::Vec;

use clock::{Civil, Wall};
use log::{info, warn};
use paging::Ram;
use uefi::{
    Handle, Status,
    boot::{self, AllocateType, LoadImageSource, MemoryDescriptor, MemoryType, ScopedProtocol},
    cstr16,
    mem::memory_map::{MemoryMap, MemoryMapMut},
    proto::{
        BootPolicy,
        device_path::{DevicePath, build},
        loaded_image::LoadedImage,
        media::{
            file::{File, FileAttribute, FileMode, RegularFile},
            fs::SimpleFileSystem,
        },
    },
    runtime, system,
    table::cfg::ConfigTableEntry,
};
use x86_64::PhysAddr;

use crate::{
    bytes,
    error::{Context, LoaderError},
    wide,
};

/// Path of the hypervisor image on the boot volume.
const IMAGE_PATH: &uefi::CStr16 = cstr16!("\\pulzar.efi");

/// UEFI path of the image the initial guest starts.
const GUEST_IMAGE_PATH: &uefi::CStr16 = cstr16!("\\EFI\\limine\\limine_x64.efi");

/// The loader's own image, as firmware describes it.
///
/// Its handle remains the parent of the preloaded guest image through
/// `StartImage`, while its range remains part of the firmware snapshot handed
/// to the guest.
#[derive(Clone, Copy, Debug)]
pub struct Loader {
    /// Handle firmware created when it loaded this image.
    pub handle: Handle,
    /// Base address of the loaded image. Firmware's identity map makes this
    /// both its physical and its virtual address.
    pub base: u64,
    /// Bytes the image occupies, rounded up to a whole number of pages —
    /// firmware allocated it in pages, so the rounding stays inside the
    /// allocation and the tail gets wiped with the rest.
    pub size: u64,
}

/// An EFI image that firmware has loaded but not started.
#[derive(Clone, Copy, Debug)]
pub struct GuestImage {
    /// Handle firmware assigned to the loaded image.
    pub handle: Handle,
}

/// Finds Windows Boot Manager and asks firmware to load it for the guest.
///
/// The path is searched on every Simple File System volume because firmware
/// gives their handles no meaningful order. More than one match is refused: a
/// boot decision made from handle order would not be reproducible.
///
/// # Errors
///
/// [`LoaderError::GuestImageMissing`] if no volume contains the configured
/// path, [`LoaderError::GuestImageAmbiguous`] if several do, or a firmware
/// error from protocol opening or image loading.
pub fn load_guest() -> Result<GuestImage, LoaderError> {
    let mut selected = None;
    for handle in
        boot::find_handles::<SimpleFileSystem>().context("enumerate filesystem volumes")?
    {
        if !contains_guest(handle)? {
            continue;
        }
        if selected.replace(handle).is_some() {
            return Err(LoaderError::GuestImageAmbiguous);
        }
    }
    let volume = selected.ok_or(LoaderError::GuestImageMissing)?;
    Ok(GuestImage {
        handle: load_from(volume)?,
    })
}

/// Whether `volume` contains the configured guest image.
///
/// A missing file is the ordinary answer for almost every volume. Any other
/// firmware error means the volume could not be inspected reliably and stops
/// the boot rather than making a selection from an incomplete search.
fn contains_guest(volume: Handle) -> Result<bool, LoaderError> {
    let mut filesystem = boot::open_protocol_exclusive::<SimpleFileSystem>(volume)
        .context("open a filesystem volume")?;
    let mut root = filesystem
        .open_volume()
        .context("open a filesystem root directory")?;
    match root.open(GUEST_IMAGE_PATH, FileMode::Read, FileAttribute::empty()) {
        Ok(_) => Ok(true),
        Err(error) if error.status() == Status::NOT_FOUND => Ok(false),
        Err(error) => Err(LoaderError::Firmware {
            operation: "inspect a filesystem volume for Windows Boot Manager",
            status: error.status(),
        }),
    }
}

/// Loads the guest image by its full device path on `volume`.
///
/// `LoadImage` needs a device path rather than a filesystem handle. The volume
/// already publishes its hardware path, so the file node is appended to that
/// path instead of reconstructing a disk or partition path from guessed
/// firmware details.
fn load_from(volume: Handle) -> Result<Handle, LoaderError> {
    let device = boot::open_protocol_exclusive::<DevicePath>(volume)
        .context("open the guest volume device path")?;
    let mut storage = Vec::new();
    let mut builder = build::DevicePathBuilder::with_vec(&mut storage);
    for node in device.node_iter() {
        builder = builder.push(&node).map_err(|_| LoaderError::GuestPath)?;
    }
    let path = builder
        .push(&build::media::FilePath {
            path_name: GUEST_IMAGE_PATH,
        })
        .map_err(|_| LoaderError::GuestPath)?
        .finalize()
        .map_err(|_| LoaderError::GuestPath)?;
    boot::load_image(
        boot::image_handle(),
        LoadImageSource::FromDevicePath {
            device_path: path,
            boot_policy: BootPolicy::ExactMatch,
        },
    )
    .context("load Windows Boot Manager")
}

/// Reports where firmware put the loader image.
///
/// # Errors
///
/// [`LoaderError::Firmware`] if the loaded-image protocol cannot be opened,
/// which would mean firmware did not give us the handle it started us with.
pub fn loaded_self() -> Result<Loader, LoaderError> {
    let handle = boot::image_handle();
    let image = boot::open_protocol_exclusive::<LoadedImage>(handle)
        .context("open the loader's own loaded-image protocol")?;
    let (base, size) = image.info();
    Ok(Loader {
        handle,
        base: wide(base.addr()),
        size: size.next_multiple_of(paging::chunk::FRAME_SIZE),
    })
}

/// The physical memory the loader takes out of firmware's hands for good.
///
/// Two regions, reserved together because they are the same kind of thing: both
/// outlive the loader, both are [`MemoryType::RESERVED`] so that nothing after
/// pulzar reuses them, and neither can be asked for once firmware is gone.
#[derive(Clone, Copy, Debug)]
pub struct Reserved {
    /// Base of the hypervisor's 64 MiB chunk, [`paging::chunk::CHUNK_ALIGN`]
    /// aligned and below 4 GiB.
    pub chunk: PhysAddr,
    /// Base of the page the other processors start executing on, below 1 MiB.
    pub trampoline: PhysAddr,
}

/// Reserves both regions.
///
/// # Errors
///
/// [`LoaderError::Firmware`] if firmware cannot satisfy either request.
pub fn reserve() -> Result<Reserved, LoaderError> {
    Ok(Reserved {
        chunk: allocate_chunk()?,
        trampoline: allocate_trampoline()?,
    })
}

/// Reserves the one region of physical memory the hypervisor will own.
///
/// [`MemoryType::RESERVED`] is what keeps the region out of every later
/// consumer's hands, firmware's included, and unlike loader-owned memory it
/// survives the loader being unloaded — which it must, since it holds the page
/// tables the hypervisor is running on by then.
///
/// Firmware only promises page alignment, so the request is one alignment
/// larger than the chunk and the base is rounded up inside it. The slack stays
/// reserved and unused: freeing it would hand a hole back in the middle of a
/// region whose whole purpose is to be untouchable.
///
/// The ceiling is not arbitrary. The hypervisor's PML4 is one of the chunk's
/// frames, and the last thing a processor being started does before it enters
/// long mode is load that address into `CR3` — while it is still in 32-bit
/// protected mode, where the register is 32 bits wide. A chunk above 4 GiB
/// would give it a page table it cannot name.
///
/// # Errors
///
/// [`LoaderError::Firmware`] if firmware has no contiguous region that large
/// below 4 GiB.
fn allocate_chunk() -> Result<PhysAddr, LoaderError> {
    let span = paging::chunk::CHUNK_SIZE + paging::chunk::CHUNK_ALIGN;
    let pages = bytes(span / paging::chunk::FRAME_SIZE);
    let base = boot::allocate_pages(
        AllocateType::MaxAddress(FOUR_GIB),
        MemoryType::RESERVED,
        pages,
    )
    .context("reserve the hypervisor's memory chunk below 4 GiB")?;
    Ok(PhysAddr::new(
        wide(base.addr().get()).next_multiple_of(paging::chunk::CHUNK_ALIGN),
    ))
}

/// One past the highest physical address a 32-bit `CR3` can name.
const FOUR_GIB: u64 = 1 << 32;

/// One past the highest physical address a startup interprocessor interrupt can
/// send a processor to: the vector is eight bits and the processor reads it as
/// `vector << 12`.
const ONE_MIB: u64 = 1 << 20;

/// Reserves the page the other processors will start executing on.
///
/// It has to be in the first megabyte because that is the only place a startup
/// interprocessor interrupt can point a processor at, and the first megabyte is
/// firmware's — it keeps its own idle processors parked somewhere in it. So the
/// page is asked for rather than picked out of the memory map, and firmware
/// answers with one nothing else is using.
///
/// [`MemoryType::RESERVED`] for the same reason the chunk uses it: the page
/// outlives the loader, and the other processors may be started at any point
/// after boot, not only during it.
///
/// # Errors
///
/// [`LoaderError::Firmware`] if firmware has no free page below 1 MiB, which
/// leaves no way to start another processor.
fn allocate_trampoline() -> Result<PhysAddr, LoaderError> {
    let base = boot::allocate_pages(AllocateType::MaxAddress(ONE_MIB), MemoryType::RESERVED, 1)
        .context("reserve the application processors' trampoline page below 1 MiB")?;
    Ok(PhysAddr::new(wide(base.addr().get())))
}

/// The hypervisor image's file, open for reading.
///
/// The file system protocol is held alongside the file because closing it would
/// take the volume the file lives on with it.
#[derive(Debug)]
pub struct ImageFile {
    // Declared before the protocol it came from, so that dropping this closes
    // the file first and the volume second.
    file: RegularFile,
    // Never read: held only so the volume stays open for as long as a file on it
    // does, and closed by dropping it.
    _volume: ScopedProtocol<SimpleFileSystem>,
}

impl ImageFile {
    /// Opens the hypervisor image on the volume the loader was started from.
    ///
    /// # Errors
    ///
    /// [`LoaderError::Firmware`] if the volume cannot be opened or the image is
    /// not on it, or [`LoaderError::NotARegularFile`] if the path names
    /// something other than a file.
    pub fn open() -> Result<Self, LoaderError> {
        let mut volume = boot::get_image_file_system(boot::image_handle())
            .context("open the file system the loader was started from")?;
        let mut root = volume
            .open_volume()
            .context("open the root directory of the boot volume")?;
        let handle = root
            .open(IMAGE_PATH, FileMode::Read, FileAttribute::empty())
            .context("open the hypervisor image")?;
        let file = handle
            .into_regular_file()
            .ok_or(LoaderError::NotARegularFile)?;
        Ok(Self {
            file,
            _volume: volume,
        })
    }

    /// Reads up to `buffer.len()` bytes from `offset`, returning how many
    /// arrived.
    ///
    /// # Errors
    ///
    /// [`LoaderError::Firmware`] if the seek or the read fails.
    pub fn read_at(&mut self, offset: u64, buffer: &mut [u8]) -> Result<usize, LoaderError> {
        self.file
            .set_position(offset)
            .context("seek in the hypervisor image")?;
        self.file
            .read(buffer)
            .context("read from the hypervisor image")
    }

    /// Fills `buffer` from `offset`, refusing a short read.
    ///
    /// A read that stops early is reported by firmware as success, so leaving
    /// it unchecked would map a partly loaded section and fault somewhere
    /// far from the cause.
    ///
    /// # Errors
    ///
    /// As [`ImageFile::read_at`], plus [`LoaderError::ShortRead`] if the file
    /// ends inside the requested range.
    pub fn read_exact(&mut self, offset: u64, buffer: &mut [u8]) -> Result<(), LoaderError> {
        let wanted = buffer.len();
        let got = self.read_at(offset, buffer)?;
        if got == wanted {
            return Ok(());
        }
        Err(LoaderError::ShortRead {
            offset,
            wanted,
            got,
        })
    }
}

/// Physical address of the ACPI root pointer firmware published, or zero if it
/// published none.
///
/// UEFI advertises one configuration table entry per ACPI generation, and
/// firmware that supports ACPI 2.0 or later publishes both. The newer entry
/// wins where both exist: it leads to a root pointer that carries a revision
/// and a 64-bit table directory, while the 1.0 entry can only ever describe
/// tables below 4 GiB. What either entry holds is a physical address, because
/// firmware runs the boot services phase on an identity map.
///
/// A machine with no ACPI at all is reported as zero rather than refused here.
/// Which tables the hypervisor cannot do without is the hypervisor's judgement
/// to make, not the loader's.
pub fn acpi_rsdp() -> u64 {
    system::with_config_table(|entries| {
        [ConfigTableEntry::ACPI2_GUID, ConfigTableEntry::ACPI_GUID]
            .into_iter()
            .find_map(|wanted| {
                entries
                    .iter()
                    .find(|entry| entry.guid == wanted)
                    .map(|entry| wide(entry.address.addr()))
            })
            .unwrap_or_default()
    })
}

/// The wall-clock time firmware's real-time clock reports, in UTC.
///
/// `GetTime` is a runtime service rather than a boot service, so it is one of
/// the few firmware calls that would still answer after boot services are gone.
/// It is made here all the same: the hypervisor drops the half of the address
/// space firmware's code lives in, so the reading has to be taken while that
/// half is still there and travel in the boot protocol instead.
///
/// The reading is normalized to UTC using the offset firmware states. The
/// daylight-saving flags beside it are not applied — they say whether the
/// reading has already been adjusted, not that it needs to be.
///
/// `None` where the machine has no real-time clock, firmware refuses the call,
/// or the answer is one the calendar does not admit, which is what a clock that
/// has lost its battery reports. That leaves the hypervisor with a monotonic
/// clock and no dates in its log, which is worth saying plainly and not worth
/// failing a boot over.
pub fn wall_clock() -> Option<Wall> {
    let time = runtime::get_time()
        .inspect_err(|error| warn!("loader: firmware would not report the time: {error}"))
        .ok()?;
    let wall = Wall::from_civil(Civil {
        year: time.year(),
        month: time.month(),
        day: time.day(),
        hour: time.hour(),
        minute: time.minute(),
        second: time.second(),
        nanosecond: time.nanosecond(),
        utc_offset_minutes: time.time_zone(),
    });
    match wall {
        Some(wall) => info!("loader: firmware's clock reads {wall}"),
        None => warn!("loader: firmware reported {time}, which is not a time"),
    }
    wall
}

/// What firmware's memory map told the loader.
#[derive(Clone, Copy, Debug)]
pub struct Memory {
    /// One past the highest physical address firmware describes as memory.
    /// Device apertures are excluded.
    pub top_of_ram: u64,
    /// Descriptors copied out of the map.
    pub entries: usize,
}

/// What firmware's memory map told the loader, and the memory it described.
///
/// The ranges travel beside the summary rather than inside it because the
/// summary is `Copy` and ends up in the handoff, while the ranges are only
/// needed for as long as it takes to build the direct map over them.
pub struct Survey {
    /// The two numbers the handoff carries.
    pub memory: Memory,
    /// Every run of memory firmware described, ascending and coalesced. What
    /// the direct map is built over, and nothing else is: the gaps between them
    /// are physical address space that answers to nothing, or that answers to a
    /// device, and a cached always-present alias of either is wrong.
    pub ram: Vec<Ram>,
}

/// Copies firmware's memory map into the chunk, finds the top of RAM, and
/// coalesces the runs of memory it describes.
///
/// The copy exists because firmware's own map lives in memory the hypervisor
/// stops being able to reach: it is allocated from the pool, in the half of the
/// address space that gets dropped. The stride is normalized to
/// [`MemoryDescriptor`]'s own size rather than firmware's, which is free to be
/// larger, so the hypervisor reads a plain array.
///
/// It is a snapshot, and boot services stay live afterwards — the pool
/// allocation this call itself makes and releases is not reflected in it. The
/// hypervisor uses it to know what memory exists, not what is currently free.
///
/// The map is sorted first, so the runs come out ascending and adjacent
/// descriptors of different types coalesce into one range. Both are what the
/// address space needs of them: it scans the runs to ask whether a whole page
/// is memory, and a scan wants an order.
///
/// Reserved, unusable and not-yet-accepted ranges all count as memory — they
/// are RAM whoever owns them, and counting reserved memory is what puts the
/// chunk itself under the direct map.
///
/// Device memory does not count, and that is the point of computing this at
/// all. The direct map exists to make reads and writes of RAM cheap; a device
/// aperture can sit terabytes above the last stick of RAM, and sizing the
/// direct map to reach it would cost page tables proportional to that gap for
/// ranges that must not be accessed through a cached, always-present mapping
/// anyway. Device registers are reached by mapping them explicitly, with the
/// caching and protection the device requires. So the three non-memory types
/// are left out: memory-mapped I/O and I/O port space, which belong to devices,
/// and Itanium processor code, which cannot occur here.
///
/// # Errors
///
/// [`LoaderError::Firmware`] if firmware will not produce a map,
/// [`LoaderError::MemoryMapTooLarge`] if it is larger than `capacity`
/// descriptors, or [`LoaderError::Paging`] if a descriptor describes a range
/// the address space cannot represent.
///
/// # Safety
///
/// `destination` must be writable for `capacity` descriptors.
pub unsafe fn capture_memory_map(
    destination: core::ptr::NonNull<MemoryDescriptor>,
    capacity: usize,
) -> Result<Survey, LoaderError> {
    /// Memory types whose descriptors describe something other than RAM.
    const NOT_MEMORY: [MemoryType; 3] = [
        MemoryType::MMIO,
        MemoryType::MMIO_PORT_SPACE,
        MemoryType::PAL_CODE,
    ];

    let mut map =
        boot::memory_map(MemoryType::LOADER_DATA).context("retrieve the UEFI memory map")?;
    // Ascending, so the runs below come out in the order the address space
    // wants to scan them and adjacent descriptors are adjacent here too.
    map.sort();
    let entries = map.len();
    if entries > capacity {
        return Err(LoaderError::MemoryMapTooLarge { entries, capacity });
    }
    let mut top_of_ram = 0;
    let mut ram: Vec<Ram> = Vec::new();
    for (index, descriptor) in map.entries().enumerate() {
        if !NOT_MEMORY.contains(&descriptor.ty) {
            let start = descriptor.phys_start;
            let end = start + descriptor.page_count * paging::chunk::FRAME_SIZE;
            top_of_ram = top_of_ram.max(end);
            match ram.last().copied() {
                // Firmware describes one stretch of memory as many descriptors,
                // one per owner. They are one range as far as the direct map is
                // concerned, and joining them here is what keeps it able to use
                // large pages across them.
                Some(last) if last.end() == start => {
                    let joined = Ram::new(last.start(), end)?;
                    let last = ram.len() - 1;
                    ram[last] = joined;
                }
                _ => ram.push(Ram::new(start, end)?),
            }
        }
        // SAFETY: `index` is below `entries`, which the check above holds to
        // `capacity`, so this stays inside the region the caller vouched for.
        // `write` does not read what was there, which matters because reserved
        // memory arrives holding whatever its last owner left.
        unsafe { destination.add(index).write(*descriptor) };
    }
    Ok(Survey {
        memory: Memory {
            top_of_ram,
            entries,
        },
        ram,
    })
}
