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
//! # Intermediate page tables are permanent
//!
//! Nothing here ever frees a page directory or a page table, only leaf entries.
//! That is a policy, and it is bounded: the mapping window is a fixed 1 GiB, so
//! its tables can never exceed one page directory plus 512 page tables — a
//! little over 2 MiB of the 64 MiB chunk — and the direct map's tables are
//! created once. Keeping them costs that ceiling once instead of allocating and
//! zeroing tables again on every mapping.
//!
//! It also decides what a failed mapping owes. `x86_64`'s mapper installs each
//! parent entry as it allocates the frame behind it, so a `map_to` that fails
//! at a later level has already created earlier ones and cannot be asked to
//! undo them. Under this policy that is not a leak to be reported: those tables
//! are exactly the tables a retry at the same address would have needed, and
//! they stay within the same ceiling. So no operation here promises that a
//! failure leaves no frames allocated — only that it leaves nothing *mapped*
//! that the caller did not ask for, and nothing owned by both the caller and
//! this space at once.
//!
//! # What a failed operation promises
//!
//! Two things, and they are the whole contract:
//!
//! 1. **Every entry this changed is invalidated everywhere before the call
//!    returns**, whether it returns success or failure. A partly finished range
//!    still had a prefix changed, and the other processors are told about that
//!    prefix on the way out.
//! 2. **Nothing is returned to an allocator unless it is proved detached.** A
//!    window run whose pages could not all be unmapped, and a frame whose page
//!    is still described, stay allocated for good — reported as
//!    [`PagingError::CleanupFailed`] rather than handed to the next caller.
//!    Retiring memory is a cost; handing out an address that still has a live
//!    translation is a corruption.

use log::{info, warn};
use processor::Features;
use x86_64::{
    PhysAddr, VirtAddr,
    registers::control::{Cr3, Cr3Flags},
    structures::paging::{
        Mapper, Page, PageSize, PageTable, PageTableFlags, PhysFrame, Size1GiB, Size2MiB, Size4KiB,
        mapper::{FlagUpdateError, MapToError, MappedPageTable, MapperFlush, UnmapError},
        page_table::PageTableEntry,
    },
};

use crate::{
    DirectMap, Frames, PagingError, Slots, as_u64, as_usize, buddy,
    chunk::{self, FRAME_SIZE},
    cpu,
    direct::PageTables,
    end_of,
    kaslr::Placement,
    phys_at,
    shootdown::{self, Flush},
    virt_at,
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
    /// firmware half is gone. It is built over `ram` and nothing else: see
    /// [`AddressSpace::build_direct_map`] for why the gaps matter.
    ///
    /// # Errors
    ///
    /// [`PagingError::FiveLevelPaging`], [`PagingError::NoExecuteUnsupported`],
    /// [`PagingError::PcidEnabled`] or [`PagingError::PatUnsupported`] for a
    /// machine this subsystem will not run on; [`PagingError::Misaligned`] if
    /// the chunk is not [`chunk::CHUNK_ALIGN`] aligned;
    /// [`PagingError::EmptyRegion`] if `ram` describes no memory;
    /// [`PagingError::Arithmetic`] if `ram` is not ascending and disjoint or
    /// reaches past the physical address space;
    /// [`PagingError::HighHalfInUse`] if firmware has a high-half mapping of
    /// its own, which would silently collide with ours;
    /// [`PagingError::OutOfFrames`] if the chunk cannot back the tables.
    ///
    /// # Safety
    ///
    /// `chunk_base` must be the base of a [`chunk::CHUNK_SIZE`]-byte reserved
    /// region nothing else uses, identity-mapped by firmware and described by
    /// one of the ranges in `ram`. `ram` must describe memory that behaves as
    /// RAM — never a device aperture, which this crate maps write-back. The
    /// current address space must be firmware's, since the lower half is read
    /// from `CR3`. The calling processor must be in bring-up, as
    /// [`cpu::establish_pat`] requires.
    pub unsafe fn build(
        chunk_base: PhysAddr,
        ram: &[Ram],
        placement: Placement,
    ) -> Result<Self, PagingError> {
        cpu::refuse_five_level_paging()?;
        cpu::refuse_process_context_identifiers()?;
        cpu::enable_no_execute()?;
        // SAFETY: the caller guarantees this processor is in bring-up: firmware's
        // address space is still the active one, nothing of pulzar's is mapped,
        // and no other processor is running.
        if unsafe { cpu::establish_pat() }? {
            warn!("paging: firmware had reprogrammed IA32_PAT; established pulzar's policy");
        }

        let ram = RamMap::new(ram)?;
        let window = DirectMap::identity();
        // SAFETY: the caller guarantees the chunk is reserved, unused, and
        // identity-mapped, which is what both allocators need of it.
        let mut frames = unsafe { Frames::create(chunk_base, window) }?;
        // SAFETY: as above.
        let slots = unsafe { Slots::create(chunk_base, window, placement.mapping_window_base) }?;
        let root = frames.allocate(0)?;

        let mut space = Self {
            root,
            window,
            direct_map: DirectMap::new(
                placement.direct_map_base,
                crate::direct_map_size(ram.top())?,
            )?,
            frames,
            slots,
            features: processor::features(),
        };
        // SAFETY: `root` is a freshly allocated, zeroed frame no one else refers
        // to, and the active address space is still firmware's.
        unsafe { space.inherit_lower_half() }?;
        space.build_direct_map(&ram)?;
        Ok(space)
    }

    /// Adopts an address space another image built and activated.
    ///
    /// From here on the direct map *is* the window: firmware's identity map may
    /// already be gone, and the chunk is only reachable through the mapping the
    /// loader made for it.
    ///
    /// Everything the other image recorded is checked rather than believed. The
    /// two are separate files that can be staged independently, so a handoff
    /// whose own version matches can still have been written by a loader that
    /// laid the chunk out differently, sized the window differently, or
    /// activated a different set of page tables than the one it described. Each
    /// of those is a value this image can compare against something it knows:
    /// its own compiled constants, the register the processor is running on,
    /// and — for the direct map, the one that cannot be checked against a
    /// constant — a walk of the live tables proving that the base really
    /// does map the chunk where it claims.
    ///
    /// # Errors
    ///
    /// As [`AddressSpace::build`] for the processor checks, plus
    /// [`PagingError::LayoutMismatch`] for a value the loader recorded that
    /// this image was not built for, [`PagingError::Unreachable`] or a
    /// [`PagingError::Buddy`] if the described chunk holds no allocator state,
    /// and [`PagingError::NotMapped`] or [`PagingError::TableUnreachable`] if
    /// the direct map does not actually describe the chunk where the handoff
    /// says it does.
    ///
    /// # Safety
    ///
    /// `existing` must describe the address space that is currently active, and
    /// no other `AddressSpace` may be live for it. The calling processor must
    /// be in bring-up, as [`cpu::establish_pat`] requires.
    pub unsafe fn adopt(existing: &Existing) -> Result<Self, PagingError> {
        cpu::refuse_five_level_paging()?;
        cpu::refuse_process_context_identifiers()?;
        cpu::enable_no_execute()?;
        // SAFETY: the caller guarantees this processor is in bring-up. The
        // policy is the same one the loader established, so on a machine where
        // both ran this writes nothing.
        if unsafe { cpu::establish_pat() }? {
            warn!("paging: IA32_PAT did not hold pulzar's policy on adoption; established it");
        }
        existing.check()?;

        let direct_map = DirectMap::new(existing.direct_map_base, existing.direct_map_size)?;
        // SAFETY: the caller guarantees the direct map is active, and `check`
        // proved the chunk lies inside its numeric reach; the walk below proves
        // the mapping itself is there. This is the only `AddressSpace` for it by
        // the caller's contract.
        let frames = unsafe { Frames::adopt(existing.chunk_base, direct_map) }?;
        // SAFETY: as above.
        let slots = unsafe {
            Slots::adopt(
                existing.chunk_base,
                direct_map,
                existing.mapping_window_base,
            )
        }?;
        let space = Self {
            root: existing.root,
            window: direct_map,
            direct_map,
            frames,
            slots,
            features: processor::features(),
        };
        space.prove_direct_map(existing.chunk_base)?;
        Ok(space)
    }

    /// Maps `len` bytes of physical memory at a caller-chosen virtual address.
    ///
    /// For regions whose address is dictated from outside — the hypervisor
    /// image, which has to land where it was relocated to. Everything else
    /// should take an address from the window with
    /// [`AddressSpace::map_physical`].
    ///
    /// All or nothing. The whole range is checked to be free before a single
    /// entry is written, and a mapping that fails anyway — the chunk running
    /// out of frames for a page table part way through — has the pages it
    /// did install removed again before it returns.
    ///
    /// # Errors
    ///
    /// [`PagingError::Misaligned`] unless both addresses are frame-aligned;
    /// [`PagingError::EmptyRegion`] for a zero length;
    /// [`PagingError::Arithmetic`] if the range does not fit the address space;
    /// [`PagingError::AlreadyMapped`] or [`PagingError::ParentHugePage`] if any
    /// page of the range is already spoken for; [`PagingError::OutOfFrames`] if
    /// the chunk cannot back the tables; or [`PagingError::CleanupFailed`] if
    /// undoing a partial mapping did not complete, which leaves the range
    /// partly mapped and says so.
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
        let pages = check_span(virt, len)?;
        if !phys.as_u64().is_multiple_of(FRAME_SIZE) {
            return Err(PagingError::Misaligned {
                value: phys.as_u64(),
                align: FRAME_SIZE,
            });
        }
        phys_at(
            phys,
            (pages - 1) * FRAME_SIZE,
            "the last frame of a mapped region",
        )?;
        // Nothing may be described here already. Asked before anything is
        // written, so a collision half way along a range leaves the range
        // untouched instead of half installed.
        for index in 0..pages {
            let page = page_at(virt, index)?;
            match self.leaf(page.start_address())? {
                Leaf::Absent => {}
                Leaf::Mapped { size, .. } if size == FRAME_SIZE => {
                    return Err(PagingError::AlreadyMapped {
                        virt: page.start_address().as_u64(),
                    });
                }
                Leaf::Mapped { .. } => {
                    return Err(PagingError::ParentHugePage {
                        virt: page.start_address().as_u64(),
                    });
                }
            }
        }

        let flags = protection.flags() | cache.flags();
        for index in 0..pages {
            let page = page_at(virt, index)?;
            let frame = PhysFrame::containing_address(phys_at(
                phys,
                index * FRAME_SIZE,
                "a frame of a mapped region",
            )?);
            // SAFETY: the caller vouches for the physical range, and the loop
            // above proved nothing describes `page` yet.
            let Err(error) = (unsafe { self.map_one(page, frame, flags, Freshness::Dictated) })
            else {
                continue;
            };
            // The prefix was never mapped before this call, so no processor can
            // have cached anything about it and removing it needs no broadcast.
            self.unwind(Page::containing_address(virt), index)?;
            return Err(error);
        }
        Ok(())
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
    /// [`PagingError::Arithmetic`] if the range does not fit the address space,
    /// [`PagingError::OutOfWindow`] if the window has no run that large, or
    /// [`PagingError::OutOfFrames`] if the chunk cannot back the tables. A
    /// failure part-way through leaves nothing mapped and nothing reserved,
    /// except where undoing it did not complete — then it is
    /// [`PagingError::CleanupFailed`] and the run is retired.
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
        let span = end_of(len, offset, "the span of a physical mapping")?;
        let pages = as_usize(span.div_ceil(FRAME_SIZE));
        let order = buddy::order_for(pages);
        let first = self.slots.allocate(order)?;
        let flags = protection.flags() | cache.flags();
        for index in 0..as_u64(pages) {
            let page = first + index;
            let frame = PhysFrame::containing_address(phys_at(
                base,
                index * FRAME_SIZE,
                "a frame of a physical mapping",
            )?);
            // SAFETY: the caller vouches for the physical range, and `page` comes
            // from a run this call just reserved, so nothing else maps it.
            let Err(error) = (unsafe { self.map_one(page, frame, flags, Freshness::Fresh) }) else {
                continue;
            };
            // Only give the addresses back once every page of the prefix is
            // proved gone; otherwise the run is retired with them.
            self.unwind(first, index)?;
            self.release_slots(first, order);
            return Err(error);
        }
        Ok(Mapping {
            virt: virt_at(
                first.start_address(),
                offset,
                "the address of a physical mapping",
            )?,
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
    /// The window slots are returned last, and only when every page of the run
    /// has been removed *and* every processor has acknowledged dropping the
    /// translations. Anything less and the run is retired: handing out an
    /// address some processor still translates would alias whatever is mapped
    /// there next, which is the one outcome worse than leaking a gigabyte's
    /// worth of address space one run at a time.
    ///
    /// # Errors
    ///
    /// [`PagingError::NotMapped`] or [`PagingError::ParentHugePage`] if the
    /// mapping is not there to remove, or
    /// [`PagingError::ShootdownIncomplete`] if some processor did not
    /// acknowledge dropping the translations. On any failure the run is retired
    /// rather than returned.
    ///
    /// # Safety
    ///
    /// Nothing derived from [`Mapping::addr`] may still be in use.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "taking the mapping by value is what makes unmapping it twice unrepresentable"
    )]
    pub unsafe fn unmap(&mut self, mapping: Mapping) -> Result<(), PagingError> {
        let pages = as_u64(mapping.pages);
        self.retract(mapping.first, pages)?;
        self.slots.release(mapping.first, mapping.order)?;
        Ok(())
    }

    /// Releases a mapping made with [`AddressSpace::map_region`].
    ///
    /// The physical memory is untouched, as with [`AddressSpace::unmap`]: this
    /// only removes the description of it. Intermediate tables stay, as
    /// everywhere in this crate.
    ///
    /// This is the counterpart `map_region` needs and `unmap` cannot be: a
    /// region mapped at a dictated address holds no window slots and so has no
    /// [`Mapping`] to give back.
    ///
    /// The whole range is checked before any of it is removed, so a range that
    /// is not entirely mapped leaves the address space untouched. If a removal
    /// fails anyway, every page that was removed is invalidated on every
    /// processor before the error is returned.
    ///
    /// # Errors
    ///
    /// [`PagingError::Misaligned`] unless `virt` is frame-aligned,
    /// [`PagingError::EmptyRegion`] for a zero length,
    /// [`PagingError::Arithmetic`] if the range does not fit the address space,
    /// [`PagingError::NotMapped`] or [`PagingError::ParentHugePage`] if any
    /// page of the range is not a 4 KiB mapping, or
    /// [`PagingError::ShootdownIncomplete`] if some processor did not
    /// acknowledge dropping the translations.
    ///
    /// # Safety
    ///
    /// Nothing derived from the range may still be in use, on this processor or
    /// any other.
    pub unsafe fn unmap_region(&mut self, virt: VirtAddr, len: u64) -> Result<(), PagingError> {
        let pages = check_span(virt, len)?;
        self.retract(Page::containing_address(virt), pages)
    }

    /// Maps physical memory, hands its address to `action`, and unmaps it.
    ///
    /// The scoped form of [`AddressSpace::map_physical`], for the common case
    /// of a mapping that exists to do one thing — reading a firmware table,
    /// wiping a range — where a leaked mapping would be a silent bug.
    ///
    /// The mapping is removed on every path out of `action` that this target
    /// has. `x86_64-unknown-uefi` aborts on panic rather than unwinding, so
    /// there is no third path in which the mapping could be left behind; a
    /// target that unwound would need a guard here, and would need a policy for
    /// what to do when the guard's own unmapping fails.
    ///
    /// # Errors
    ///
    /// As [`AddressSpace::map_physical`] and [`AddressSpace::unmap`]. `action`
    /// has already run when an unmapping error is returned.
    ///
    /// # Safety
    ///
    /// As [`AddressSpace::map_physical`]. `action` must not let anything
    /// derived from the address escape, including through its return value, and
    /// must not panic in a build that unwinds.
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
    /// [`PagingError::Arithmetic`] for a page count with no room for its
    /// guards, [`PagingError::OutOfWindow`] if the window has no run that
    /// large, or [`PagingError::OutOfFrames`] if the chunk cannot back the
    /// stack. A failure part-way through leaves nothing mapped and nothing
    /// reserved, and gives back every frame it had allocated for the stack
    /// itself — intermediate page tables stay, as everywhere here. Where
    /// undoing it did not complete it is [`PagingError::CleanupFailed`]
    /// instead, and whatever could not be proved detached is retired.
    pub fn allocate_stack(&mut self, pages: u64) -> Result<Stack, PagingError> {
        if pages == 0 {
            return Err(PagingError::EmptyRegion);
        }
        let with_guards = end_of(pages, 2, "the page count of a stack and its guards")?;
        let order = buddy::order_for(as_usize(with_guards));
        let first = self.slots.allocate(order)?;
        let flags = Protection::ReadWrite.flags() | CacheType::WriteBack.flags();
        for index in 0..pages {
            let page = first + 1 + index;
            let mapped = match self.frames.allocate(0) {
                Ok(frame) => {
                    // SAFETY: `frame` was just allocated, so this space is its
                    // only owner, and `page` is inside a run this call reserved
                    // and has not mapped yet.
                    let result = unsafe { self.map_one(page, frame, flags, Freshness::Fresh) };
                    if result.is_err() {
                        self.release_frame(frame);
                    }
                    result
                }
                Err(error) => Err(error),
            };
            let Err(error) = mapped else {
                continue;
            };
            // Only give the addresses back once every page of the prefix is
            // proved gone and its frame returned; otherwise both are retired.
            self.unwind_owned(first + 1, index)?;
            self.release_slots(first, order);
            return Err(error);
        }
        let bottom = (first + 1).start_address();
        Ok(Stack {
            bottom,
            top: virt_at(bottom, pages * FRAME_SIZE, "the top of a stack")?,
            pages,
            first,
            order,
        })
    }

    /// Gives a stack from [`AddressSpace::allocate_stack`] back: its pages are
    /// unmapped, the frames behind them return to the chunk, and the window run
    /// holding it and its two guards is released.
    ///
    /// The counterpart every staged allocation needs. A caller that allocates
    /// several stacks and fails part-way through has no other way to undo the
    /// ones that succeeded, and stacks are large enough that leaking them
    /// exhausts the chunk over a few retries.
    ///
    /// The stack is consumed, which is what makes releasing one twice
    /// unrepresentable, and nothing is given back to an allocator until every
    /// page is proved gone and every processor has said so.
    ///
    /// # Errors
    ///
    /// [`PagingError::CleanupFailed`] if a page could not be unmapped, or
    /// [`PagingError::ShootdownIncomplete`] if some processor did not
    /// acknowledge. Either way the addresses and any frame still described are
    /// retired rather than reused.
    ///
    /// # Safety
    ///
    /// Nothing may still be running on the stack, and no processor may still
    /// name it — in an interrupt stack table, a task state segment, or a saved
    /// stack pointer. The addresses go straight back to the window allocator,
    /// so a later mapping may be handed the very same range.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "taking the stack by value is what makes releasing it twice unrepresentable"
    )]
    pub unsafe fn release_stack(&mut self, stack: Stack) -> Result<(), PagingError> {
        let Stack {
            first,
            pages,
            order,
            ..
        } = stack;
        self.unwind_owned(first + 1, pages)?;
        broadcast(Flush::small(first + 1, pages))?;
        self.slots.release(first, order)?;
        Ok(())
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
    /// The whole range is checked to be described by 2 MiB pages before any of
    /// it is changed. If an update fails anyway, the pages already tightened
    /// are invalidated on every processor before the error is returned — a
    /// processor still holding the writable translation of a page this made
    /// read-only is exactly the alias the call exists to remove.
    ///
    /// # Errors
    ///
    /// [`PagingError::Misaligned`] for a range that is not 2 MiB aligned and
    /// sized, [`PagingError::EmptyRegion`] for a zero length,
    /// [`PagingError::Arithmetic`] if the range leaves the address space,
    /// [`PagingError::NotMapped`] or [`PagingError::ParentHugePage`] if the
    /// direct map does not describe the range in 2 MiB pages, or
    /// [`PagingError::ShootdownIncomplete`] if some processor did not
    /// acknowledge dropping the translations this tightened.
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
        self.direct_map.reach(phys, len)?;
        let count = len / Size2MiB::SIZE;
        let first = Page::<Size2MiB>::containing_address(self.direct_map_address(phys)?);
        // Every page of the range has to already be a 2 MiB direct-map page.
        // Asked before anything changes, so a range reaching one page past the
        // direct map's coverage leaves the pages before it alone.
        for index in 0..count {
            let page = first + index;
            match self.leaf(page.start_address())? {
                Leaf::Mapped { size, .. } if size == Size2MiB::SIZE => {}
                Leaf::Mapped { .. } => {
                    return Err(PagingError::ParentHugePage {
                        virt: page.start_address().as_u64(),
                    });
                }
                Leaf::Absent => {
                    return Err(PagingError::NotMapped {
                        virt: page.start_address().as_u64(),
                    });
                }
            }
        }

        let flags = protection.flags() | CacheType::WriteBack.flags() | PageTableFlags::HUGE_PAGE;
        let mut changed = 0;
        let outcome = (0..count).try_for_each(|index| {
            let page = first + index;
            // SAFETY: only one mapper is alive — it lives for this statement —
            // and this changes protection on an existing direct-map entry
            // without changing what it points at. Tightening the direct map
            // cannot invalidate a reference that was allowed to exist, because
            // the direct map is documented as a read and modify window rather
            // than a place to hold long-lived writable references into.
            let result = unsafe { self.mapper()?.update_flags(page, flags) }
                .map(MapperFlush::flush)
                .map_err(|error| flag_update_error(page.start_address(), &error));
            changed += u64::from(result.is_ok());
            result
        });
        // Whatever was tightened is announced, whether the range finished or
        // not: a prefix that is read-only here and writable elsewhere is the
        // alias this call exists to remove.
        let announced = if changed == 0 {
            Ok(())
        } else {
            broadcast(Flush::large(first, changed))
        };
        outcome.and(announced)
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
    /// its pointers work. The flush afterwards reaches global translations too,
    /// which is not optional: writing `CR3` flushes non-global translations
    /// only, and firmware's identity map is free to have marked its entries
    /// global.
    ///
    /// # Errors
    ///
    /// [`PagingError::Unreachable`] or [`PagingError::BadPointer`] if the
    /// window does not reach the PML4, which would mean the direct map was
    /// never built, or [`PagingError::ShootdownIncomplete`] if some processor
    /// did not acknowledge dropping what it had cached of the half just
    /// removed.
    ///
    /// # Safety
    ///
    /// This space must be active, its window must be the direct map rather than
    /// firmware's identity map, and no firmware pointer — boot services, the
    /// system table, the memory map, the loader's image — may be used
    /// afterwards.
    pub unsafe fn drop_lower_half(&mut self) -> Result<(), PagingError> {
        let mut root = self.window.ptr::<PageTable>(self.root.start_address())?;
        // SAFETY: the window reaches the whole PML4, which this space owns; no
        // other reference to it is live because every mapper this crate builds
        // exists inside a single statement, and `&mut self` is exclusive.
        for entry in unsafe { root.as_mut() }.iter_mut().take(HIGH_HALF) {
            entry.set_unused();
        }
        cpu::flush_translations();
        // Everything, because this is the one invalidation that also clears
        // entries firmware may have marked global — which writing the page
        // table root does not reach, on this processor or any other.
        broadcast(Flush::EVERYTHING)
    }

    /// Resolves `virt` in this space.
    ///
    /// A read-only walk over shared references, taken through `&self`. It
    /// deliberately does not go through `x86_64`'s mapper: that type is built
    /// from a `&mut PageTable`, so translating with it would mean creating a
    /// mutable reference to the hierarchy for an operation that writes nothing
    /// — and two callers doing so at once would be two mutable references
    /// to the same table, which is undefined behaviour whether or not
    /// either of them writes.
    ///
    /// # Errors
    ///
    /// [`PagingError::NotMapped`] if nothing describes the address, which is a
    /// fact about the address, or [`PagingError::TableUnreachable`] if a table
    /// on the way down could not be read through the window, which is a broken
    /// invariant of this subsystem. The two are separate because only the
    /// second says the answer is unknown rather than negative.
    pub fn translate(&self, virt: VirtAddr) -> Result<PhysAddr, PagingError> {
        match self.leaf(virt)? {
            Leaf::Mapped { phys, size } => phys_at(
                phys,
                virt.as_u64() & (size - 1),
                "the translation of an address",
            ),
            Leaf::Absent => Err(PagingError::NotMapped {
                virt: virt.as_u64(),
            }),
        }
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
        let source = self.window.ptr::<PageTable>(firmware)?;
        let mut target = self.window.ptr::<PageTable>(self.root.start_address())?;
        // SAFETY: `source` is the live PML4 named by `CR3` and `target` is a
        // frame this space just allocated, so the two are distinct and neither
        // has another live reference; the window reaches both in full.
        let (source, target) = unsafe { (source.as_ref(), target.as_mut()) };
        if let Some(index) = (HIGH_HALF..ENTRIES).find(|index| !source[*index].is_unused()) {
            return Err(PagingError::HighHalfInUse { index });
        }
        for index in 0..HIGH_HALF {
            target[index] = source[index].clone();
        }
        Ok(())
    }

    /// Maps the memory `ram` describes at the direct map's base.
    ///
    /// Large pages where a large page's worth of RAM is there to describe:
    /// 1 GiB where the processor supports them, 2 MiB otherwise, 4 KiB for the
    /// remainder around a boundary. The gigabyte containing our own chunk is
    /// always described in 2 MiB pages, so that the image's frames can later be
    /// made read-only there without splitting anything.
    ///
    /// Nothing outside `ram` is mapped, and that is the point rather than an
    /// optimization. A dense map of `0..top_of_ram` describes every hole
    /// between memory ranges, every device aperture that happens to sit
    /// numerically below the last stick of RAM, and the padding between the
    /// last byte of RAM and the gigabyte the map is rounded up to — all of
    /// them as cached, always-present, writable memory. For a hole that is
    /// an access to physical space that answers to nothing; for an aperture
    /// it is a write-back alias of a range some other mapping describes as
    /// uncached, which is an architecturally undefined conflict rather than
    /// a mistake with a defined outcome.
    ///
    /// The runs are walked rather than the address space probed, so the cost
    /// tracks how much memory there is instead of how high it ends: a machine
    /// whose RAM ends a terabyte up costs the same as one whose RAM ends at
    /// four gigabytes with the same amount of it. The alternative — asking of
    /// every span in the whole direct map whether it is memory — is work
    /// proportional to the address space, most of it spent re-asking about
    /// holes.
    fn build_direct_map(&mut self, ram: &RamMap) -> Result<(), PagingError> {
        let flags = Protection::ReadWrite.flags() | CacheType::WriteBack.flags();
        let owned = self.frames.chunk_base().as_u64();
        let owned = owned..owned + chunk::CHUNK_SIZE;
        for range in ram.ranges {
            let end = range.end();
            let mut at = range.start();
            // The run's two edges rarely sit on large-page boundaries and are
            // described in 4 KiB pages: the frames before the first aligned
            // 2 MiB span...
            let first_two = end.min(at.div_ceil(Size2MiB::SIZE) * Size2MiB::SIZE);
            while at < first_two {
                self.map_direct::<Size4KiB>(at, flags)?;
                at += FRAME_SIZE;
            }
            // ...the whole 2 MiB spans of the middle, promoted to 1 GiB pages
            // wherever a whole GiB is memory and does not hold the chunk...
            while at + Size2MiB::SIZE <= end {
                let full_gib = at.is_multiple_of(Size1GiB::SIZE) && at + Size1GiB::SIZE <= end;
                if self.features.contains(Features::GIB_PAGES)
                    && full_gib
                    && !(at < owned.end && owned.start < at + Size1GiB::SIZE)
                {
                    self.map_direct::<Size1GiB>(at, flags)?;
                    at += Size1GiB::SIZE;
                    continue;
                }
                self.map_direct::<Size2MiB>(at, flags)?;
                at += Size2MiB::SIZE;
            }
            // ...and the frames past the last aligned 2 MiB span.
            while at < end {
                self.map_direct::<Size4KiB>(at, flags)?;
                at += FRAME_SIZE;
            }
        }
        Ok(())
    }

    /// Maps one page of the direct map, covering physical `at`.
    fn map_direct<S: PageSize>(&mut self, at: u64, flags: PageTableFlags) -> Result<(), PagingError>
    where
        for<'table> MappedPageTable<'table, PageTables>: Mapper<S>,
    {
        let virt = virt_at(self.direct_map.base(), at, "a direct map address")?;
        // SAFETY: the direct map is this space's own alias of physical memory,
        // established before anything else maps any of it, and no-execute keeps
        // it from being a path to executing data. The space is not active yet,
        // so nothing can have cached a translation of this address.
        unsafe {
            self.map_one(
                Page::<S>::containing_address(virt),
                PhysFrame::containing_address(PhysAddr::new(at)),
                flags,
                Freshness::Fresh,
            )
        }
    }

    /// Proves the direct map really describes the chunk where it says it does.
    ///
    /// The one value in the handoff that cannot be checked against a compiled
    /// constant is where the loader put the direct map, because it is random by
    /// design. It can be checked against the machine, though: walking the live
    /// page tables from the address the handoff names to the physical address
    /// it should hold is a proof that the base, the tables and the chunk
    /// all agree. Nothing that follows would work if they did not, and
    /// everything that follows assumes it.
    fn prove_direct_map(&self, chunk_base: PhysAddr) -> Result<(), PagingError> {
        let virt = self.direct_map_address(chunk_base)?;
        let found = self.translate(virt)?;
        if found == chunk_base {
            return Ok(());
        }
        Err(PagingError::LayoutMismatch {
            field: "the direct map's mapping of the chunk",
            expected: chunk_base.as_u64(),
            found: found.as_u64(),
        })
    }

    /// Where the direct map makes `phys` readable.
    fn direct_map_address(&self, phys: PhysAddr) -> Result<VirtAddr, PagingError> {
        virt_at(
            self.direct_map.base(),
            phys.as_u64(),
            "a direct map address",
        )
    }

    /// Removes `pages` 4 KiB pages from `first` and tells every processor.
    ///
    /// The shared body of both unmapping operations, and the place their whole
    /// contract lives: the range is checked before it is touched, every page
    /// that was removed is announced whether the range finished or not, and the
    /// caller only gets `Ok` when both halves succeeded — which is what makes
    /// releasing the addresses afterwards safe.
    fn retract(&mut self, first: Page<Size4KiB>, pages: u64) -> Result<(), PagingError> {
        for index in 0..pages {
            let page = first + index;
            match self.leaf(page.start_address())? {
                Leaf::Mapped { size, .. } if size == FRAME_SIZE => {}
                Leaf::Mapped { .. } => {
                    return Err(PagingError::ParentHugePage {
                        virt: page.start_address().as_u64(),
                    });
                }
                Leaf::Absent => {
                    return Err(PagingError::NotMapped {
                        virt: page.start_address().as_u64(),
                    });
                }
            }
        }
        let mut removed = 0;
        let outcome = (0..pages).try_for_each(|index| {
            let result = self.unmap_one::<Size4KiB>(first + index).map(drop);
            removed += u64::from(result.is_ok());
            result
        });
        let announced = if removed == 0 {
            Ok(())
        } else {
            broadcast(Flush::small(first, removed))
        };
        outcome.and(announced)
    }

    /// Maps one page of any size.
    ///
    /// # Safety
    ///
    /// `frame` must be memory the caller may alias at `page` with `flags`, and
    /// `freshness` must be the truth about `page`: [`Freshness::Fresh`] claims
    /// no processor can hold a translation of it.
    unsafe fn map_one<S: PageSize>(
        &mut self,
        page: Page<S>,
        frame: PhysFrame<S>,
        flags: PageTableFlags,
        freshness: Freshness,
    ) -> Result<(), PagingError>
    where
        for<'table> MappedPageTable<'table, PageTables>: Mapper<S>,
    {
        // SAFETY: the mapper lives only for this statement, so it is the only
        // one, and `&mut self` makes this the only access to the hierarchy.
        let mut mapper = unsafe { self.mapper() }?;
        // SAFETY: the caller vouches for the aliasing; the frame allocator hands
        // out zeroed frames from the chunk for any intermediate table needed.
        let flush = unsafe { mapper.map_to(page, frame, flags, &mut self.frames) }
            .map_err(|error| map_to_error(page.start_address(), &error))?;
        match freshness {
            // The address came from the window allocator or belongs to a space
            // that is not active yet, so nothing has translated it and there is
            // nothing cached to evict. `INVLPG` is a serializing operation, and
            // this is the path that runs once per page of every mapping.
            Freshness::Fresh => flush.ignore(),
            // The caller chose the address, so something may have described it
            // before.
            Freshness::Dictated => flush.flush(),
        }
        Ok(())
    }

    /// Unmaps one page of any size, returning the frame it referred to.
    ///
    /// Intermediate tables are left in place; see the module documentation.
    fn unmap_one<S: PageSize>(&mut self, page: Page<S>) -> Result<PhysFrame<S>, PagingError>
    where
        for<'table> MappedPageTable<'table, PageTables>: Mapper<S>,
    {
        // SAFETY: the mapper lives only for this statement, so it is the only
        // one, and `&mut self` makes this the only access to the hierarchy.
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
    /// The rollback path of a partly built mapping. The pages are ones this
    /// call had just mapped for the first time, so no other processor can hold
    /// a translation of them and no broadcast is owed.
    ///
    /// # Errors
    ///
    /// [`PagingError::CleanupFailed`] naming the first page that could not be
    /// removed. The remaining pages are still attempted — leaving more mapped
    /// than necessary helps nobody — but the caller must not return the
    /// addresses to any allocator afterwards.
    fn unwind(&mut self, first: Page<Size4KiB>, count: u64) -> Result<(), PagingError> {
        let mut failure = None;
        for index in 0..count {
            let page = first + index;
            if let Err(error) = self.unmap_one::<Size4KiB>(page) {
                warn!(
                    "paging: rollback could not unmap {:#x}: {error}",
                    page.start_address()
                );
                failure.get_or_insert(PagingError::CleanupFailed {
                    virt: page.start_address().as_u64(),
                });
            }
        }
        failure.map_or(Ok(()), Err)
    }

    /// As [`AddressSpace::unwind`], for pages whose frames this space
    /// allocated.
    ///
    /// A frame is returned to the chunk only when the page describing it is
    /// proved gone. A frame whose page could not be removed stays allocated:
    /// it is still described by a live entry, and handing it out again would
    /// give two owners one piece of memory.
    ///
    /// # Errors
    ///
    /// As [`AddressSpace::unwind`].
    fn unwind_owned(&mut self, first: Page<Size4KiB>, count: u64) -> Result<(), PagingError> {
        let mut failure = None;
        for index in 0..count {
            let page = first + index;
            match self.unmap_one::<Size4KiB>(page) {
                Ok(frame) => self.release_frame(frame),
                Err(error) => {
                    warn!(
                        "paging: rollback could not unmap {:#x}: {error}",
                        page.start_address()
                    );
                    failure.get_or_insert(PagingError::CleanupFailed {
                        virt: page.start_address().as_u64(),
                    });
                }
            }
        }
        failure.map_or(Ok(()), Err)
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

    /// What the hierarchy says about `virt`, without writing anything.
    ///
    /// The one walk in this crate, used by translation, by every preflight, and
    /// by the proof that an adopted direct map is where it claims to be. Shared
    /// references only, each living no longer than the entry it reads.
    fn leaf(&self, virt: VirtAddr) -> Result<Leaf, PagingError> {
        let mut phys = self.root.start_address();
        // The three levels a walk can stop early at, either because nothing is
        // described or because a large page describes it here.
        for level in [Level::Pml4, Level::Pdpt, Level::Pd] {
            let Some(entry) = self.entry(phys, virt, level)? else {
                return Ok(Leaf::Absent);
            };
            if entry.flags().contains(PageTableFlags::HUGE_PAGE) {
                return match level.page_size() {
                    // Bit 7 of a PML4 entry is reserved: an entry that has it
                    // set is not one four-level paging defines, so nothing can
                    // be concluded from where it points.
                    None => Err(PagingError::InvalidFrame {
                        phys: entry.addr().as_u64(),
                    }),
                    Some(size) => Ok(Leaf::Mapped {
                        phys: entry.addr(),
                        size,
                    }),
                };
            }
            phys = entry.addr();
        }
        // At the last level bit 7 is the page attribute table selector rather
        // than a size, so a present entry here is always a 4 KiB leaf.
        Ok(self
            .entry(phys, virt, Level::Pt)?
            .map_or(Leaf::Absent, |entry| Leaf::Mapped {
                phys: entry.addr(),
                size: FRAME_SIZE,
            }))
    }

    /// The present entry `virt` selects in the table at `phys`, or `None` if it
    /// describes nothing.
    fn entry(
        &self,
        phys: PhysAddr,
        virt: VirtAddr,
        level: Level,
    ) -> Result<Option<PageTableEntry>, PagingError> {
        let table =
            self.window
                .ptr::<PageTable>(phys)
                .map_err(|_| PagingError::TableUnreachable {
                    phys: phys.as_u64(),
                })?;
        // SAFETY: the window covers the whole table, which is frame-sized and
        // frame-aligned, and this space owns every table reachable from its
        // root. The shared reference lives only for this read; nothing in this
        // crate writes a page table except through `&mut self`, and `&self` here
        // is what rules that out for as long as it exists.
        let entry = unsafe { table.as_ref() }[level.index_of(virt)].clone();
        Ok(entry
            .flags()
            .contains(PageTableFlags::PRESENT)
            .then_some(entry))
    }

    /// Borrows this space's PML4 as something `x86_64`'s mapper can drive.
    ///
    /// Takes the root and the window by value rather than borrowing `self`,
    /// which is what lets the caller pass `&mut self.frames` to the mapper in
    /// the same statement. The exclusion that makes it sound comes from the
    /// caller's own `&mut self`, not from a lifetime on the returned value —
    /// which is why this is `unsafe` and why every caller confines the mapper
    /// to one statement.
    ///
    /// # Safety
    ///
    /// The caller must hold exclusive access to this address space for as long
    /// as the returned mapper lives, and at most one mapper may exist at a
    /// time: each is a `&mut` to the same PML4, and two of them would
    /// alias.
    unsafe fn mapper<'table>(&self) -> Result<MappedPageTable<'table, PageTables>, PagingError> {
        let table = self.window.ptr::<PageTable>(self.root.start_address())?;
        // SAFETY: the PML4 is a live, correctly aligned page table this space
        // owns, reachable in full through the window, and the caller guarantees
        // exclusive access for the mapper's whole life. The window covers the
        // entire chunk — `Frames::create` and `Frames::adopt` both prove it —
        // and every frame the walker can follow comes from that chunk, which is
        // what `PageTables::new` promises.
        Ok(unsafe { MappedPageTable::new(&mut *table.as_ptr(), PageTables::new(self.window)) })
    }
}

/// Whether an address being mapped can have a stale translation anywhere.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Freshness {
    /// Nothing has ever described this address in this space, so nothing can
    /// have cached it: the address came from the window allocator, or the space
    /// is not active yet.
    Fresh,
    /// The address was chosen by the caller and may have described something
    /// before.
    Dictated,
}

/// Validates a frame-aligned, non-empty virtual region and answers how many
/// pages it spans.
///
/// The last page's address is formed here rather than in the loop that maps or
/// unmaps it, which is what rules out a range whose end is not an address
/// changing a prefix before finding that out.
fn check_span(virt: VirtAddr, len: u64) -> Result<u64, PagingError> {
    if len == 0 {
        return Err(PagingError::EmptyRegion);
    }
    if !virt.as_u64().is_multiple_of(FRAME_SIZE) {
        return Err(PagingError::Misaligned {
            value: virt.as_u64(),
            align: FRAME_SIZE,
        });
    }
    let pages = len.div_ceil(FRAME_SIZE);
    virt_at(
        virt,
        (pages - 1) * FRAME_SIZE,
        "the last page of a mapped region",
    )?;
    Ok(pages)
}

/// The `index`th page of a region starting at `virt`.
fn page_at(virt: VirtAddr, index: u64) -> Result<Page<Size4KiB>, PagingError> {
    virt_at(virt, index * FRAME_SIZE, "a page of a mapped region").map(Page::containing_address)
}

/// What a walk of the hierarchy found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Leaf {
    /// A present leaf entry, with the physical base it names and the number of
    /// bytes its page covers.
    Mapped {
        /// Physical base of the page.
        phys: PhysAddr,
        /// Bytes the page covers.
        size: u64,
    },
    /// Nothing describes the address.
    Absent,
}

/// One level of the four-level hierarchy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Level {
    /// The root table.
    Pml4,
    /// Page directory pointer table, whose entries may be 1 GiB pages.
    Pdpt,
    /// Page directory, whose entries may be 2 MiB pages.
    Pd,
    /// Page table, whose entries are always 4 KiB pages.
    Pt,
}

impl Level {
    /// The index `virt` selects at this level.
    fn index_of(self, virt: VirtAddr) -> usize {
        let index = match self {
            Self::Pml4 => virt.p4_index(),
            Self::Pdpt => virt.p3_index(),
            Self::Pd => virt.p2_index(),
            Self::Pt => virt.p1_index(),
        };
        usize::from(u16::from(index))
    }

    /// How much a large page at this level covers, or `None` where the
    /// architecture defines no large page.
    const fn page_size(self) -> Option<u64> {
        match self {
            Self::Pml4 => None,
            Self::Pdpt => Some(Size1GiB::SIZE),
            Self::Pd => Some(Size2MiB::SIZE),
            Self::Pt => Some(FRAME_SIZE),
        }
    }
}

/// A run of physical memory that behaves as RAM.
///
/// What the direct map is built over. Half-open, `start..end`, and never empty:
/// an empty range describes nothing and would only be a way to say nothing at
/// greater length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ram {
    start: u64,
    end: u64,
}

impl Ram {
    /// A run covering `start..end`.
    ///
    /// # Errors
    ///
    /// [`PagingError::EmptyRegion`] if the range is empty or inverted, or
    /// [`PagingError::Misaligned`] if either endpoint is not frame-aligned —
    /// the direct map describes whole frames, so a range that does not is one
    /// whose edge could only be honoured by rounding, in one direction or the
    /// other, and both directions are wrong.
    pub fn new(start: u64, end: u64) -> Result<Self, PagingError> {
        if end <= start {
            return Err(PagingError::EmptyRegion);
        }
        for value in [start, end] {
            if !value.is_multiple_of(FRAME_SIZE) {
                return Err(PagingError::Misaligned {
                    value,
                    align: FRAME_SIZE,
                });
            }
        }
        Ok(Self { start, end })
    }

    /// Where the run begins.
    #[must_use]
    pub const fn start(&self) -> u64 {
        self.start
    }

    /// One past where it ends.
    #[must_use]
    pub const fn end(&self) -> u64 {
        self.end
    }
}

/// Every run of RAM in the machine, ascending and disjoint.
///
/// Ascending because the answer to "how high does memory end" is the last
/// range, and disjoint because two ranges describing the same frame would make
/// the direct map's coverage depend on which was consulted. The direct map is
/// built by walking the ranges, so both properties keep that walk from ever
/// describing a byte twice.
struct RamMap<'a> {
    ranges: &'a [Ram],
}

impl<'a> RamMap<'a> {
    /// Checks that `ranges` is a usable description of physical memory.
    ///
    /// # Errors
    ///
    /// [`PagingError::EmptyRegion`] if there are no ranges at all, or
    /// [`PagingError::Arithmetic`] if they are not ascending and disjoint.
    fn new(ranges: &'a [Ram]) -> Result<Self, PagingError> {
        let Some(first) = ranges.first() else {
            return Err(PagingError::EmptyRegion);
        };
        let mut previous = first.end;
        for range in &ranges[1..] {
            if range.start < previous {
                return Err(PagingError::Arithmetic {
                    what: "an ascending, disjoint description of physical memory",
                });
            }
            previous = range.end;
        }
        Ok(Self { ranges })
    }

    /// One past the highest address any of them reaches.
    fn top(&self) -> u64 {
        self.ranges.last().map_or(0, Ram::end)
    }
}

/// Where an already-built address space keeps its parts.
///
/// The hypervisor image fills this in from the boot protocol; keeping it a
/// `paging` type rather than reading the protocol directly is what keeps this
/// crate independent of UEFI and of the handoff's layout.
///
/// Every field is something the adopting image checks. The sizes and the layout
/// identifier are compared against the constants this image was compiled with,
/// the root against the register the processor is running on, and the direct
/// map against a walk of the live tables.
#[derive(Clone, Copy, Debug)]
pub struct Existing {
    /// Which arrangement of the chunk's metadata the other image wrote. Must be
    /// [`chunk::LAYOUT`].
    pub layout: u64,
    /// Physical base of the reserved chunk.
    pub chunk_base: PhysAddr,
    /// Byte length of the reserved chunk. Must be [`chunk::CHUNK_SIZE`].
    pub chunk_size: u64,
    /// The active PML4.
    pub root: PhysFrame,
    /// Virtual base of the direct map.
    pub direct_map_base: VirtAddr,
    /// Bytes of physical memory the direct map covers.
    pub direct_map_size: u64,
    /// Virtual base of the mapping window.
    pub mapping_window_base: VirtAddr,
    /// Byte length of the mapping window. Must be
    /// [`chunk::MAPPING_WINDOW_SIZE`].
    pub mapping_window_size: u64,
}

impl Existing {
    /// Checks everything about this description that can be checked before any
    /// of it is used.
    ///
    /// # Errors
    ///
    /// [`PagingError::LayoutMismatch`] for a value that disagrees with this
    /// image's own constants or with the machine, or
    /// [`PagingError::Arithmetic`] for a region that does not fit the address
    /// space.
    fn check(&self) -> Result<(), PagingError> {
        for (field, expected, found) in [
            ("the chunk layout", chunk::LAYOUT, self.layout),
            ("the chunk size", chunk::CHUNK_SIZE, self.chunk_size),
            (
                "the mapping window size",
                chunk::MAPPING_WINDOW_SIZE,
                self.mapping_window_size,
            ),
            (
                "the active page table root",
                Cr3::read().0.start_address().as_u64(),
                self.root.start_address().as_u64(),
            ),
        ] {
            if expected != found {
                return Err(PagingError::LayoutMismatch {
                    field,
                    expected,
                    found,
                });
            }
        }

        // The root has to be one of the chunk's own frames. A root outside it is
        // one this image's allocator does not account for and would hand out.
        let root = self.root.start_address().as_u64();
        let chunk = self.chunk_base.as_u64();
        let chunk_end = end_of(chunk, self.chunk_size, "the end of the chunk")?;
        if root < chunk || root >= chunk_end {
            return Err(PagingError::LayoutMismatch {
                field: "the page table root's place in the chunk",
                expected: chunk,
                found: root,
            });
        }
        // The direct map must reach the whole chunk numerically before anything
        // asks it to reach any part of it.
        if chunk_end > self.direct_map_size {
            return Err(PagingError::LayoutMismatch {
                field: "the direct map's coverage of the chunk",
                expected: chunk_end,
                found: self.direct_map_size,
            });
        }

        // Two high-half regions that overlapped would be two owners of one
        // address, and the window allocator would hand out addresses the direct
        // map already describes.
        let map = self.direct_map_base.as_u64();
        let map_end = end_of(map, self.direct_map_size, "the end of the direct map")?;
        let window = self.mapping_window_base.as_u64();
        let window_end = end_of(window, self.mapping_window_size, "the end of the window")?;
        if map < window_end && window < map_end {
            return Err(PagingError::LayoutMismatch {
                field: "the direct map and the mapping window overlap",
                expected: map_end,
                found: window,
            });
        }
        Ok(())
    }
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
/// These four are exactly the types selectable by `PWT` and `PCD` against
/// [`cpu::PAT_POLICY`], which every processor establishes before it uses a
/// mapping. Write-combining and write-protected are absent deliberately: they
/// need the `PAT` bit, whose position differs between 4 KiB pages and large
/// pages, and nothing pulzar maps wants them. Note that an MTRR can still force
/// a stricter type over a range — MMIO in particular — which is firmware's
/// decision and not overridden here.
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
///
/// Neither `Copy` nor `Clone`, which is what makes releasing one twice
/// unrepresentable: [`AddressSpace::release_stack`] consumes it, and there is
/// no second value left to release. The run backing it is not exposed for the
/// same reason — nothing outside this module can name the addresses to give
/// them back by hand.
#[derive(Debug)]
#[must_use = "a stack that is never released leaks its frames and its addresses"]
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
}

/// Tells every other processor that translations it may hold no longer
/// describe anything.
///
/// Called after the local invalidation, and only where an entry stopped
/// describing what it used to. Making a mapping needs none of this: the address
/// came from the window allocator, so no processor has touched it and none can
/// have cached anything about it.
fn broadcast(flush: Flush) -> Result<(), PagingError> {
    if shootdown::broadcast(flush) {
        return Ok(());
    }
    Err(PagingError::ShootdownIncomplete)
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
