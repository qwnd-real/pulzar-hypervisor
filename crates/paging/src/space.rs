//! The address space: its construction, its phases, and everything mapped in
//! it.
//!
//! One [`AddressSpace`] owns a PML4, the chunk's frames, and the mapping
//! window's addresses. It exists in two forms, which differ only in how
//! physical memory is reachable *while it is being manipulated*:
//!
//! - In `hv-loader` it is built while firmware's identity map is active, so
//!   page table frames are reached at their physical addresses.
//! - In the hypervisor image it is adopted after the loader activated it, so
//!   page table frames are reached through the direct map it now contains.
//!
//! That is the `window` field, and it is the whole reason the same code can
//! build an address space it is not running in and then keep editing it once it
//! is.
//!
//! Phase transitions:
//!
//! - [`AddressSpace::build`] produces a PML4 whose lower half is firmware's,
//!   entry for entry, and whose high half is ours. Activating it is phase 2:
//!   boot services still work, the loader is still mapped where firmware put
//!   it, and the hypervisor image, the direct map and the mapping window have
//!   appeared in the high half.
//! - [`AddressSpace::drop_lower_half`] is phase 3. Every firmware address
//!   becomes unmapped in one step, which is the point: there is no window in
//!   which some firmware pointers work and others do not.
//!
//! Unmapping never frees intermediate page tables. That is deliberate, not an
//! omission: the mapping window is a fixed 1 GiB, so its tables are bounded at
//! one page directory plus 512 page tables — a little over 2 MiB of the 64 MiB
//! chunk — and keeping them costs that once instead of allocating and zeroing
//! tables again on every mapping.

use log::{info, warn};
use processor::Features;
use x86_64::{
    PhysAddr, VirtAddr,
    registers::control::{Cr3, Cr3Flags, Cr4, Cr4Flags},
    structures::paging::{
        Mapper, Page, PageSize, PageTable, PageTableFlags, PhysFrame, Size1GiB, Size2MiB, Size4KiB,
        Translate,
        mapper::{FlagUpdateError, MapToError, MappedPageTable, MapperFlush, UnmapError},
    },
};

use crate::{
    DirectMap, Frames, PagingError, Slots, as_u64, as_usize, buddy,
    chunk::{self, FRAME_SIZE},
    cpu,
    kaslr::Placement,
};

/// Entries in a page table at any level.
const ENTRIES: usize = 512;

/// First PML4 entry of the high half, and so the number of entries the lower
/// half occupies.
const HIGH_HALF: usize = 256;

/// A four-level address space under construction or in use.
#[derive(Debug)]
pub struct AddressSpace {
    root: PhysFrame,
    window: DirectMap,
    direct_map: DirectMap,
    frames: Frames,
    slots: Slots,
    features: Features,
}

impl AddressSpace {
    /// Builds the phase-2 address space: firmware's lower half, ours above it.
    ///
    /// The lower half is *shared* with firmware, not copied — only the 256 PML4
    /// entries are copied, and they keep pointing at firmware's own page
    /// tables. Cloning the tables would double the memory and desynchronize
    /// the moment firmware changed a mapping, and nothing needs a private
    /// copy of a half that is about to be discarded.
    ///
    /// The direct map is built here because everything afterwards depends on
    /// it, including the ability to edit these very page tables once the
    /// firmware half is gone.
    ///
    /// # Errors
    ///
    /// [`PagingError::FiveLevelPaging`] or
    /// [`PagingError::NoExecuteUnsupported`] for a machine this subsystem
    /// will not run on; [`PagingError::Misaligned`] if the chunk is not
    /// [`chunk::CHUNK_ALIGN`] aligned; [`PagingError::HighHalfInUse`] if
    /// firmware has a high-half mapping of its own, which would silently
    /// collide with ours; [`PagingError::OutOfFrames`] if the chunk cannot
    /// back the tables.
    ///
    /// # Safety
    ///
    /// `chunk_base` must be the base of a [`chunk::CHUNK_SIZE`]-byte reserved
    /// region nothing else uses, identity-mapped by firmware, and `top_of_ram`
    /// must be one past the highest physical address firmware describes as
    /// memory, device apertures excluded. The current address space must be
    /// firmware's, since the lower half is read from `CR3`.
    pub unsafe fn build(
        chunk_base: PhysAddr,
        top_of_ram: u64,
        placement: Placement,
    ) -> Result<Self, PagingError> {
        cpu::refuse_five_level_paging()?;
        cpu::enable_no_execute()?;
        if cpu::ensure_default_pat() {
            warn!("paging: firmware had reprogrammed IA32_PAT; restored the architectural layout");
        }
        if !chunk_base.as_u64().is_multiple_of(chunk::CHUNK_ALIGN) {
            return Err(PagingError::Misaligned {
                value: chunk_base.as_u64(),
                align: chunk::CHUNK_ALIGN,
            });
        }

        let window = DirectMap::identity();
        // SAFETY: the caller guarantees the chunk is reserved, unused, and
        // identity-mapped, which is what both allocators need of it.
        let mut frames = unsafe { Frames::create(chunk_base, window) }?;
        // SAFETY: as above.
        let slots = unsafe { Slots::create(chunk_base, window, placement.mapping_window_base) }?;
        let root = frames
            .allocate(0)
            .ok_or(PagingError::OutOfFrames { order: 0 })?;

        let mut space = Self {
            root,
            window,
            direct_map: DirectMap::new(
                placement.direct_map_base,
                crate::direct_map_size(top_of_ram),
            ),
            frames,
            slots,
            features: processor::features(),
        };
        // SAFETY: `root` is a freshly allocated, zeroed frame no one else refers
        // to, and the active address space is still firmware's.
        unsafe { space.inherit_lower_half() }?;
        space.build_direct_map()?;
        Ok(space)
    }

    /// Adopts an address space another image built and activated.
    ///
    /// From here on the direct map *is* the window: firmware's identity map may
    /// already be gone, and the chunk is only reachable through the mapping the
    /// loader made for it.
    ///
    /// # Errors
    ///
    /// As [`AddressSpace::build`] for the feature checks, plus
    /// [`PagingError::Unreachable`] or a [`PagingError::Buddy`] if the
    /// described chunk holds no allocator state — meaning the handoff
    /// describes memory that is not the loader's chunk.
    ///
    /// # Safety
    ///
    /// `existing` must describe the address space that is currently active, its
    /// direct map must already cover the chunk, and no other `AddressSpace` may
    /// be live for it.
    pub unsafe fn adopt(existing: &Existing) -> Result<Self, PagingError> {
        cpu::refuse_five_level_paging()?;
        cpu::enable_no_execute()?;
        let direct_map = DirectMap::new(existing.direct_map_base, existing.direct_map_size);
        // SAFETY: the caller guarantees the direct map is active and covers the
        // chunk, and that this is the only `AddressSpace` for it.
        let frames = unsafe { Frames::adopt(existing.chunk_base, direct_map) }?;
        // SAFETY: as above.
        let slots = unsafe {
            Slots::adopt(
                existing.chunk_base,
                direct_map,
                existing.mapping_window_base,
            )
        }?;
        Ok(Self {
            root: existing.root,
            window: direct_map,
            direct_map,
            frames,
            slots,
            features: processor::features(),
        })
    }

    /// Maps `len` bytes of physical memory at a caller-chosen virtual address.
    ///
    /// For regions whose address is dictated from outside — the hypervisor
    /// image, which has to land where it was relocated to. Everything else
    /// should take an address from the window with
    /// [`AddressSpace::map_physical`].
    ///
    /// # Errors
    ///
    /// [`PagingError::Misaligned`] unless both addresses are frame-aligned;
    /// [`PagingError::EmptyRegion`] for a zero length;
    /// [`PagingError::AlreadyMapped`] if any page is already in use;
    /// [`PagingError::OutOfFrames`] if the chunk cannot back the tables.
    ///
    /// # Safety
    ///
    /// The physical range must be memory the caller owns or is entitled to
    /// alias, and mapping it at `virt` must not create a second writable path
    /// to anything that expects to be exclusively owned.
    pub unsafe fn map_region(
        &mut self,
        virt: VirtAddr,
        phys: PhysAddr,
        len: u64,
        protection: Protection,
        cache: CacheType,
    ) -> Result<(), PagingError> {
        if len == 0 {
            return Err(PagingError::EmptyRegion);
        }
        for (value, align) in [(virt.as_u64(), FRAME_SIZE), (phys.as_u64(), FRAME_SIZE)] {
            if !value.is_multiple_of(align) {
                return Err(PagingError::Misaligned { value, align });
            }
        }
        let flags = protection.flags() | cache.flags();
        (0..len.div_ceil(FRAME_SIZE)).try_for_each(|index| {
            let offset = index * FRAME_SIZE;
            // SAFETY: the caller vouches for the physical range; `virt` is theirs
            // to choose and any collision is reported rather than overwritten.
            unsafe {
                self.map_one(
                    Page::<Size4KiB>::containing_address(virt + offset),
                    PhysFrame::containing_address(phys + offset),
                    flags,
                )
            }
        })
    }

    /// Maps `len` bytes of physical memory at an address taken from the mapping
    /// window.
    ///
    /// This is the mapping to use for memory that will be touched repeatedly:
    /// it gets its own protection and cache type, and the returned address is
    /// stable until it is unmapped. Casual one-off access to physical memory
    /// belongs on the direct map instead.
    ///
    /// The returned address preserves any sub-page offset in `phys`.
    ///
    /// # Errors
    ///
    /// [`PagingError::EmptyRegion`] for a zero length,
    /// [`PagingError::OutOfWindow`] if the window has no run that large, or
    /// [`PagingError::OutOfFrames`] if the chunk cannot back the tables. A
    /// failure part-way through leaves nothing mapped and nothing reserved.
    ///
    /// # Safety
    ///
    /// The physical range must be safe to alias with the requested protection:
    /// mapping something writable that another mapping also writes is the
    /// caller's problem to rule out.
    pub unsafe fn map_physical(
        &mut self,
        phys: PhysAddr,
        len: u64,
        protection: Protection,
        cache: CacheType,
    ) -> Result<Mapping, PagingError> {
        if len == 0 {
            return Err(PagingError::EmptyRegion);
        }
        let offset = phys.as_u64() % FRAME_SIZE;
        let base = phys - offset;
        let pages = as_usize((len + offset).div_ceil(FRAME_SIZE));
        let order = buddy::order_for(pages);
        let first = self
            .slots
            .allocate(order)
            .ok_or(PagingError::OutOfWindow { order })?;
        let flags = protection.flags() | cache.flags();
        for index in 0..as_u64(pages) {
            let page = first + index;
            let frame = PhysFrame::containing_address(base + index * FRAME_SIZE);
            // SAFETY: the caller vouches for the physical range, and `page` comes
            // from a run this call just reserved, so nothing else maps it.
            if let Err(error) = unsafe { self.map_one(page, frame, flags) } {
                self.unwind(first, index);
                self.release_slots(first, order);
                return Err(error);
            }
        }
        Ok(Mapping {
            virt: first.start_address() + offset,
            bytes: len,
            first,
            pages,
            order,
        })
    }

    /// Releases a mapping from [`AddressSpace::map_physical`].
    ///
    /// The physical memory is untouched — `map_physical` only borrowed it.
    ///
    /// # Errors
    ///
    /// [`PagingError::NotMapped`] if the mapping is already gone. On failure
    /// the window slots are deliberately *not* returned: handing out
    /// addresses that still have live translations would alias, so leaking
    /// the run is the safe outcome.
    ///
    /// # Safety
    ///
    /// Nothing derived from [`Mapping::addr`] may still be in use.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "taking the mapping by value is what makes unmapping it twice unrepresentable"
    )]
    pub unsafe fn unmap(&mut self, mapping: Mapping) -> Result<(), PagingError> {
        (0..as_u64(mapping.pages))
            .try_for_each(|index| self.unmap_one::<Size4KiB>(mapping.first + index).map(drop))?;
        self.slots.release(mapping.first, mapping.order)
    }

    /// Maps physical memory, hands its address to `action`, and unmaps it.
    ///
    /// The scoped form of [`AddressSpace::map_physical`], for the common case
    /// of a mapping that exists to do one thing — reading a firmware table,
    /// wiping a range — where a leaked mapping would be a silent bug.
    ///
    /// # Errors
    ///
    /// As [`AddressSpace::map_physical`] and [`AddressSpace::unmap`]. `action`
    /// has already run when an unmapping error is returned.
    ///
    /// # Safety
    ///
    /// As [`AddressSpace::map_physical`]. `action` must not let anything
    /// derived from the address escape, including through its return value.
    pub unsafe fn with_physical<T>(
        &mut self,
        phys: PhysAddr,
        len: u64,
        protection: Protection,
        cache: CacheType,
        action: impl FnOnce(VirtAddr) -> T,
    ) -> Result<T, PagingError> {
        // SAFETY: the caller vouches for the physical range.
        let mapping = unsafe { self.map_physical(phys, len, protection, cache) }?;
        let value = action(mapping.addr());
        // SAFETY: `action` has returned and the caller guarantees nothing derived
        // from the address outlived it.
        unsafe { self.unmap(mapping) }?;
        Ok(value)
    }

    /// Allocates a stack of `pages` pages with an unmapped guard page below and
    /// above it.
    ///
    /// Guards on both sides: below catches the overflow that stacks actually
    /// suffer, above catches a runaway write past the top, and because both are
    /// simply slots that were reserved and never mapped they cost nothing but
    /// address space.
    ///
    /// # Errors
    ///
    /// [`PagingError::EmptyRegion`] for zero pages,
    /// [`PagingError::OutOfWindow`] if the window has no run that large, or
    /// [`PagingError::OutOfFrames`] if the chunk cannot back the stack. A
    /// failure part-way through leaves nothing mapped, nothing reserved,
    /// and no frames allocated.
    pub fn allocate_stack(&mut self, pages: u64) -> Result<Stack, PagingError> {
        if pages == 0 {
            return Err(PagingError::EmptyRegion);
        }
        let order = buddy::order_for(as_usize(pages + 2));
        let first = self
            .slots
            .allocate(order)
            .ok_or(PagingError::OutOfWindow { order })?;
        let flags = Protection::ReadWrite.flags() | CacheType::WriteBack.flags();
        for index in 0..pages {
            let page = first + 1 + index;
            let Some(frame) = self.frames.allocate(0) else {
                self.unwind_owned(first + 1, index);
                self.release_slots(first, order);
                return Err(PagingError::OutOfFrames { order: 0 });
            };
            // SAFETY: `frame` was just allocated, so this space is its only
            // owner, and `page` is inside a run this call reserved and has not
            // mapped yet.
            if let Err(error) = unsafe { self.map_one(page, frame, flags) } {
                self.release_frame(frame);
                self.unwind_owned(first + 1, index);
                self.release_slots(first, order);
                return Err(error);
            }
        }
        let bottom = (first + 1).start_address();
        Ok(Stack {
            bottom,
            top: bottom + pages * FRAME_SIZE,
            pages,
            first,
            order,
        })
    }

    /// Reduces the direct map's protection over `len` bytes at `phys`.
    ///
    /// The direct map is read-write everywhere by default, which would leave a
    /// writable alias of the hypervisor's own image. Marking the image's frames
    /// read-only there removes that alias without giving up the direct map's
    /// usefulness for everything else.
    ///
    /// Both the address and the length must be 2 MiB aligned, because the
    /// direct map describes the chunk in 2 MiB pages exactly so that this
    /// can change flags rather than split a mapping.
    ///
    /// # Errors
    ///
    /// [`PagingError::Misaligned`] for a range that is not 2 MiB aligned and
    /// sized, [`PagingError::EmptyRegion`] for a zero length, or
    /// [`PagingError::NotMapped`] if the direct map does not cover the range.
    pub fn protect_direct_map(
        &mut self,
        phys: PhysAddr,
        len: u64,
        protection: Protection,
    ) -> Result<(), PagingError> {
        if len == 0 {
            return Err(PagingError::EmptyRegion);
        }
        for value in [phys.as_u64(), len] {
            if !value.is_multiple_of(Size2MiB::SIZE) {
                return Err(PagingError::Misaligned {
                    value,
                    align: Size2MiB::SIZE,
                });
            }
        }
        let flags = protection.flags() | CacheType::WriteBack.flags() | PageTableFlags::HUGE_PAGE;
        (0..len / Size2MiB::SIZE).try_for_each(|index| {
            let virt = self.direct_map.base() + phys.as_u64() + index * Size2MiB::SIZE;
            let page = Page::<Size2MiB>::containing_address(virt);
            // SAFETY: only one mapper is alive, and this changes protection on an
            // existing direct-map entry without changing what it points at.
            let mut mapper = unsafe { self.mapper() }?;
            // SAFETY: the direct map is this space's own mapping of physical
            // memory; tightening its protection cannot invalidate a reference
            // that was allowed to exist, because the direct map is documented as
            // a read/modify window and not a place to hold long-lived writable
            // references into.
            unsafe { mapper.update_flags(page, flags) }
                .map(MapperFlush::flush)
                .map_err(|error| flag_update_error(virt, &error))
        })
    }

    /// Makes this space the active one.
    ///
    /// # Safety
    ///
    /// Every address the caller will touch next — code, stack, and any data it
    /// still needs — must be mapped in this space. In the loader that is
    /// guaranteed by the lower half being firmware's; in the hypervisor it is
    /// guaranteed by the image and stack having been mapped in the high half.
    pub unsafe fn activate(&self) {
        // SAFETY: `root` is a PML4 this space built and owns, and the caller
        // guarantees the code that runs next is mapped in it.
        unsafe { Cr3::write(self.root, Cr3Flags::empty()) };
    }

    /// Phase 3: removes the firmware half of the address space.
    ///
    /// Clearing all 256 lower-half entries at once means firmware becomes
    /// unreachable atomically, rather than through a window in which some of
    /// its pointers work. The `CR4.PGE` toggle afterwards is not optional:
    /// writing `CR3` flushes non-global translations only, and firmware's
    /// identity map is free to have marked its entries global.
    ///
    /// # Errors
    ///
    /// [`PagingError::Unreachable`] if the window does not reach the PML4,
    /// which would mean the direct map was never built.
    ///
    /// # Safety
    ///
    /// This space must be active, its window must be the direct map rather than
    /// firmware's identity map, and no firmware pointer — boot services, the
    /// system table, the memory map, the loader's image — may be used
    /// afterwards.
    pub unsafe fn drop_lower_half(&mut self) -> Result<(), PagingError> {
        let mut root = self
            .window
            .ptr::<PageTable>(self.root.start_address())
            .ok_or(PagingError::Unreachable {
                phys: self.root.start_address().as_u64(),
            })?;
        // SAFETY: the window reaches the PML4, which this space owns; no other
        // reference to it is live because `mapper` only exists inside single
        // statements.
        for entry in unsafe { root.as_mut() }.iter_mut().take(HIGH_HALF) {
            entry.set_unused();
        }
        // SAFETY: the same space stays active; the write is what evicts the
        // non-global translations of the entries just cleared.
        unsafe { Cr3::write(self.root, Cr3Flags::empty()) };
        let cr4 = Cr4::read();
        if cr4.contains(Cr4Flags::PAGE_GLOBAL) {
            // SAFETY: clearing `CR4.PGE` invalidates all global translations and
            // is architecturally permitted at any time; the original value is
            // restored immediately, so nothing observes the intermediate state.
            unsafe {
                Cr4::write(cr4.difference(Cr4Flags::PAGE_GLOBAL));
                Cr4::write(cr4);
            }
        }
        Ok(())
    }

    /// Resolves `virt` in this space, or `None` if it is not mapped.
    #[must_use]
    pub fn translate(&self, virt: VirtAddr) -> Option<PhysAddr> {
        // SAFETY: only one mapper is alive, and translation does not write.
        unsafe { self.mapper() }
            .ok()
            .and_then(|mapper| mapper.translate_addr(virt))
    }

    /// Logs the layout, so a serial log records exactly what was built.
    pub fn describe(&self, who: &str) {
        info!(
            "{who}: pml4 {:#x}, chunk {:#x}, {} of {} frames free",
            self.root.start_address(),
            self.frames.chunk_base(),
            self.frames.free(),
            self.frames.capacity(),
        );
        info!(
            "{who}: direct map {:#x}+{:#x} ({} pages), window {:#x}+{:#x}, {} slots free",
            self.direct_map.base(),
            self.direct_map.size(),
            if self.features.contains(Features::GIB_PAGES) {
                "1 GiB"
            } else {
                "2 MiB"
            },
            self.slots.base(),
            chunk::MAPPING_WINDOW_SIZE,
            self.slots.free(),
        );
    }

    /// The chunk's frame allocator, for callers that need frames of their own.
    pub const fn frames(&mut self) -> &mut Frames {
        &mut self.frames
    }

    /// Physical address of this space's PML4.
    #[must_use]
    pub const fn root(&self) -> PhysFrame {
        self.root
    }

    /// The direct map this space provides.
    #[must_use]
    pub const fn direct_map(&self) -> DirectMap {
        self.direct_map
    }

    /// How this space currently reaches physical memory: firmware's identity
    /// map while the loader is building the space, the direct map
    /// afterwards.
    #[must_use]
    pub const fn window(&self) -> DirectMap {
        self.window
    }

    /// Virtual base of the mapping window.
    #[must_use]
    pub const fn mapping_window(&self) -> VirtAddr {
        self.slots.base()
    }

    /// Processor features this space was built against.
    #[must_use]
    pub const fn features(&self) -> Features {
        self.features
    }

    /// Copies firmware's 256 lower-half PML4 entries into ours and refuses a
    /// firmware that is already using the high half.
    ///
    /// # Safety
    ///
    /// `CR3` must hold firmware's PML4 and the window must reach it.
    unsafe fn inherit_lower_half(&mut self) -> Result<(), PagingError> {
        let firmware = Cr3::read().0.start_address();
        let source = self
            .window
            .ptr::<PageTable>(firmware)
            .ok_or(PagingError::Unreachable {
                phys: firmware.as_u64(),
            })?;
        let mut target = self
            .window
            .ptr::<PageTable>(self.root.start_address())
            .ok_or(PagingError::Unreachable {
                phys: self.root.start_address().as_u64(),
            })?;
        // SAFETY: `source` is the live PML4 named by `CR3` and `target` is a
        // frame this space just allocated, so the two are distinct and neither
        // has another live reference.
        let (source, target) = unsafe { (source.as_ref(), target.as_mut()) };
        if let Some(index) = (HIGH_HALF..ENTRIES).find(|index| !source[*index].is_unused()) {
            return Err(PagingError::HighHalfInUse { index });
        }
        for index in 0..HIGH_HALF {
            target[index] = source[index].clone();
        }
        Ok(())
    }

    /// Maps physical `0..size` at the direct map's base.
    ///
    /// Large pages throughout: 1 GiB where the processor supports them, 2 MiB
    /// otherwise. The gigabyte containing our own chunk is always described in
    /// 2 MiB pages, so that the image's frames can later be made read-only
    /// there without splitting anything.
    fn build_direct_map(&mut self) -> Result<(), PagingError> {
        let flags = Protection::ReadWrite.flags() | CacheType::WriteBack.flags();
        let owned = self.frames.chunk_base().as_u64();
        let owned = owned..owned + chunk::CHUNK_SIZE;
        let base = self.direct_map.base();
        for gib in (0..self.direct_map.size()).step_by(as_usize(Size1GiB::SIZE)) {
            let holds_chunk = gib < owned.end && owned.start < gib + Size1GiB::SIZE;
            if self.features.contains(Features::GIB_PAGES) && !holds_chunk {
                // SAFETY: the direct map is this space's own alias of physical
                // memory, established before anything else maps any of it, and
                // no-execute keeps it from being a path to executing data.
                unsafe {
                    self.map_one(
                        Page::<Size1GiB>::containing_address(base + gib),
                        PhysFrame::containing_address(PhysAddr::new(gib)),
                        flags,
                    )
                }?;
                continue;
            }
            for two in (gib..gib + Size1GiB::SIZE).step_by(as_usize(Size2MiB::SIZE)) {
                // SAFETY: as above.
                unsafe {
                    self.map_one(
                        Page::<Size2MiB>::containing_address(base + two),
                        PhysFrame::containing_address(PhysAddr::new(two)),
                        flags,
                    )
                }?;
            }
        }
        Ok(())
    }

    /// Maps one page of any size.
    ///
    /// # Safety
    ///
    /// `frame` must be memory the caller may alias at `page` with `flags`.
    unsafe fn map_one<S: PageSize>(
        &mut self,
        page: Page<S>,
        frame: PhysFrame<S>,
        flags: PageTableFlags,
    ) -> Result<(), PagingError>
    where
        for<'table> MappedPageTable<'table, DirectMap>: Mapper<S>,
    {
        // SAFETY: the mapper lives only for this statement, so it is the only one.
        let mut mapper = unsafe { self.mapper() }?;
        // SAFETY: the caller vouches for the aliasing; the frame allocator hands
        // out zeroed frames from the chunk for any intermediate table needed.
        unsafe { mapper.map_to(page, frame, flags, &mut self.frames) }
            .map(MapperFlush::flush)
            .map_err(|error| map_to_error(page.start_address(), &error))
    }

    /// Unmaps one page of any size, returning the frame it referred to.
    ///
    /// Intermediate tables are left in place; see the module documentation.
    fn unmap_one<S: PageSize>(&mut self, page: Page<S>) -> Result<PhysFrame<S>, PagingError>
    where
        for<'table> MappedPageTable<'table, DirectMap>: Mapper<S>,
    {
        // SAFETY: the mapper lives only for this statement, so it is the only one.
        let mut mapper = unsafe { self.mapper() }?;
        mapper
            .unmap(page)
            .map(|(frame, flush)| {
                flush.flush();
                frame
            })
            .map_err(|error| unmap_error(page.start_address(), &error))
    }

    /// Unmaps `count` pages from `first`, leaving their frames alone.
    ///
    /// The rollback path of a partly built mapping. A failure here cannot be
    /// propagated — it is already handling a failure — so it is logged and the
    /// address space stays consistent with what the allocators believe.
    fn unwind(&mut self, first: Page<Size4KiB>, count: u64) {
        for index in 0..count {
            let page = first + index;
            if let Err(error) = self.unmap_one::<Size4KiB>(page) {
                warn!(
                    "paging: rollback could not unmap {:#x}: {error}",
                    page.start_address()
                );
            }
        }
    }

    /// As [`AddressSpace::unwind`], for pages whose frames this space
    /// allocated.
    fn unwind_owned(&mut self, first: Page<Size4KiB>, count: u64) {
        for index in 0..count {
            let page = first + index;
            match self.unmap_one::<Size4KiB>(page) {
                Ok(frame) => self.release_frame(frame),
                Err(error) => warn!(
                    "paging: rollback could not unmap {:#x}: {error}",
                    page.start_address()
                ),
            }
        }
    }

    fn release_frame(&mut self, frame: PhysFrame) {
        if let Err(error) = self.frames.release(frame, 0) {
            warn!(
                "paging: rollback could not release frame {:#x}: {error}",
                frame.start_address()
            );
        }
    }

    fn release_slots(&mut self, first: Page<Size4KiB>, order: usize) {
        if let Err(error) = self.slots.release(first, order) {
            warn!(
                "paging: rollback could not release window run at {:#x}: {error}",
                first.start_address()
            );
        }
    }

    /// Borrows this space's PML4 as something `x86_64`'s mapper can drive.
    ///
    /// # Safety
    ///
    /// At most one mapper may be alive at a time: each call produces a `&mut`
    /// to the same PML4, and two of them would alias. Every caller confines
    /// the result to a single statement.
    unsafe fn mapper(&self) -> Result<MappedPageTable<'static, DirectMap>, PagingError> {
        let table = self
            .window
            .ptr::<PageTable>(self.root.start_address())
            .ok_or(PagingError::Unreachable {
                phys: self.root.start_address().as_u64(),
            })?;
        // SAFETY: the PML4 is a live, correctly aligned page table this space
        // owns, reachable through the window, and the caller guarantees this is
        // the only mapper. The `'static` borrow is sound because the chunk the
        // table lives in is never freed. `window` maps every frame the walker
        // will follow, since all of them come from that chunk.
        Ok(unsafe { MappedPageTable::new(&mut *table.as_ptr(), self.window) })
    }
}

/// Where an already-built address space keeps its parts.
///
/// The hypervisor image fills this in from the boot protocol; keeping it a
/// `paging` type rather than reading the protocol directly is what keeps this
/// crate independent of UEFI and of the handoff's layout.
#[derive(Clone, Copy, Debug)]
pub struct Existing {
    /// Physical base of the reserved chunk.
    pub chunk_base: PhysAddr,
    /// The active PML4.
    pub root: PhysFrame,
    /// Virtual base of the direct map.
    pub direct_map_base: VirtAddr,
    /// Bytes of physical memory the direct map covers.
    pub direct_map_size: u64,
    /// Virtual base of the mapping window.
    pub mapping_window_base: VirtAddr,
}

/// What a mapping permits.
///
/// An enum rather than a set of flags, so that write-and-execute is not
/// expressible: there is no variant for it, and therefore no way to ask for it
/// by accident. Read permission is implicit — a mapping nothing may read is a
/// mapping nobody wants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Protection {
    /// Readable only.
    ReadOnly,
    /// Readable and writable, never executable.
    ReadWrite,
    /// Readable and executable, never writable.
    ReadExecute,
}

impl Protection {
    /// The page table bits for this protection.
    ///
    /// `NO_EXECUTE` requires `EFER.NXE`, which [`AddressSpace::build`] and
    /// [`AddressSpace::adopt`] both establish before any mapping exists.
    #[must_use]
    pub const fn flags(self) -> PageTableFlags {
        let present = PageTableFlags::PRESENT;
        match self {
            Self::ReadOnly => present.union(PageTableFlags::NO_EXECUTE),
            Self::ReadWrite => present
                .union(PageTableFlags::WRITABLE)
                .union(PageTableFlags::NO_EXECUTE),
            Self::ReadExecute => present,
        }
    }
}

/// How a mapping is cached.
///
/// These four are exactly the types selectable by `PWT` and `PCD` against the
/// architectural `IA32_PAT` layout, which [`crate::cpu::ensure_default_pat`]
/// guarantees is in place. Write-combining and write-protected are absent
/// deliberately: they need the `PAT` bit, whose position differs between 4 KiB
/// pages and large pages, and nothing pulzar maps wants them. Note that an MTRR
/// can still force a stricter type over a range — MMIO in particular — which is
/// firmware's decision and not overridden here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheType {
    /// Cached, writes buffered. The type for ordinary RAM.
    WriteBack,
    /// Cached, writes also sent through. For memory another agent reads without
    /// snooping.
    WriteThrough,
    /// Uncached, but an MTRR may still allow caching. The usual choice for
    /// MMIO.
    UncachedMinus,
    /// Uncached, and no MTRR can override it.
    Uncached,
}

impl CacheType {
    /// The page table bits for this cache type.
    #[must_use]
    pub const fn flags(self) -> PageTableFlags {
        match self {
            Self::WriteBack => PageTableFlags::empty(),
            Self::WriteThrough => PageTableFlags::WRITE_THROUGH,
            Self::UncachedMinus => PageTableFlags::NO_CACHE,
            Self::Uncached => PageTableFlags::WRITE_THROUGH.union(PageTableFlags::NO_CACHE),
        }
    }
}

/// A live mapping of physical memory in the mapping window.
///
/// Not a `Drop` guard: unmapping needs the [`AddressSpace`] back, and a guard
/// that held it would make the space unusable for as long as any mapping
/// existed. Releasing it is [`AddressSpace::unmap`], or better,
/// [`AddressSpace::with_physical`], which cannot be forgotten.
#[derive(Debug)]
#[must_use = "a mapping that is never unmapped leaks window address space"]
pub struct Mapping {
    virt: VirtAddr,
    bytes: u64,
    first: Page<Size4KiB>,
    pages: usize,
    order: usize,
}

impl Mapping {
    /// Where the requested physical address is now readable, sub-page offset
    /// included.
    #[must_use]
    pub const fn addr(&self) -> VirtAddr {
        self.virt
    }

    /// Bytes that were requested.
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }
}

/// A stack with an unmapped guard page on either side.
#[derive(Clone, Copy, Debug)]
pub struct Stack {
    bottom: VirtAddr,
    top: VirtAddr,
    pages: u64,
    first: Page<Size4KiB>,
    order: usize,
}

impl Stack {
    /// Lowest mapped address. A push that goes below this faults.
    #[must_use]
    pub const fn bottom(&self) -> VirtAddr {
        self.bottom
    }

    /// One past the highest mapped address, which is where `RSP` starts.
    ///
    /// Page-aligned, and so also 16-byte aligned as the ABI requires.
    #[must_use]
    pub const fn top(&self) -> VirtAddr {
        self.top
    }

    /// Mapped pages, guards excluded.
    #[must_use]
    pub const fn pages(&self) -> u64 {
        self.pages
    }

    /// The window run backing the stack, guards included. Needed only to give
    /// the stack back.
    #[must_use]
    pub const fn run(&self) -> (Page<Size4KiB>, usize) {
        (self.first, self.order)
    }
}

/// Translates a mapping failure, discarding the page size the generic error
/// carries so that [`PagingError`] can stay a plain enum.
fn map_to_error<S: PageSize>(virt: VirtAddr, error: &MapToError<S>) -> PagingError {
    match error {
        MapToError::FrameAllocationFailed => PagingError::OutOfFrames { order: 0 },
        MapToError::ParentEntryHugePage => PagingError::ParentHugePage {
            virt: virt.as_u64(),
        },
        MapToError::PageAlreadyMapped(_) => PagingError::AlreadyMapped {
            virt: virt.as_u64(),
        },
    }
}

/// Translates an unmapping failure.
fn unmap_error(virt: VirtAddr, error: &UnmapError) -> PagingError {
    match error {
        UnmapError::ParentEntryHugePage => PagingError::ParentHugePage {
            virt: virt.as_u64(),
        },
        UnmapError::PageNotMapped => PagingError::NotMapped {
            virt: virt.as_u64(),
        },
        UnmapError::InvalidFrameAddress(phys) => PagingError::InvalidFrame {
            phys: phys.as_u64(),
        },
    }
}

/// Translates a protection-change failure.
fn flag_update_error(virt: VirtAddr, error: &FlagUpdateError) -> PagingError {
    match error {
        FlagUpdateError::PageNotMapped => PagingError::NotMapped {
            virt: virt.as_u64(),
        },
        FlagUpdateError::ParentEntryHugePage => PagingError::ParentHugePage {
            virt: virt.as_u64(),
        },
    }
}
