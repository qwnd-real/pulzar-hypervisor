//! One virtual processor: the control block that describes it, the registers
//! the hardware leaves to us, and the loop that runs it.
//!
//! # The loop, and what it deliberately does not do
//!
//! [`Vcpu::run`] enters the guest, and when the guest stops it hands this
//! virtual processor to a closure and enters again if that closure says to. It
//! decodes nothing and decides nothing: what an exit means is policy, and
//! policy belongs to whatever composed a guest out of these parts.
//!
//! It makes exactly two judgements of its own, and both are about itself rather
//! than about the guest. It checks the block once before the first entry, so
//! that a mistake in it is reported as the rule it breaks. And it recognizes
//! the exit code that means the processor refused the block, because that exit
//! executed no guest instruction — a loop that handed it on would spin forever
//! on a block that will never run.
//!
//! # The clean field is maintained, not published by hand
//!
//! The processor may keep parts of a control block in hardware between exits,
//! and the clean field is what says which parts are still worth reusing.
//! Getting it wrong in one direction costs performance and in the other runs a
//! guest on state that no longer exists, so it is not left to callers to
//! remember: a caller that edits the block calls [`Vcpu::soil`] with what it
//! edited, and [`Vcpu::run`] turns the accumulated edits into the field before
//! each entry. The steady state after the first entry is everything cached,
//! which is what makes an exit that changes nothing cost nothing extra.
//!
//! The first entry publishes a clean field of zero, because a block the
//! processor has never seen has nothing cached behind it.
//!
//! The translation-flush control is handled the same way and for the same
//! reason. A caller that took permission away in the guest's memory calls
//! [`Vcpu::flush`], and the next entry turns that into the narrowest command
//! this processor has for discarding one guest's translations — after which it
//! goes back to costing nothing, because the field is not one the processor
//! caches.
//!
//! # Two invariants a caller keeps
//!
//! The processor identifies its cached copy of a block by the block's *physical
//! address* and by nothing else — not by the address space identifier, and not
//! by anything naming the guest. So a block that moves, or that is entered on a
//! different processor than last time, has a cache behind it that may belong to
//! something else entirely. Both are stated on [`Vcpu::run`]. A hypervisor with
//! one virtual processor per physical processor, which is what a pass-through
//! hypervisor is, satisfies both by construction.

use core::ptr::NonNull;

use log::info;
use paging::{DirectMap, Frames};
use processor::{Features, SvmFeatures};
use svm::{
    CleanBits, ControlArea, ExitCode, Reason, SaveArea, Vmcb,
    control::NestedPagingControl,
    intercept::{Intercepts1, Intercepts2, Intercepts2Flags, TlbControl},
    msr::{EFER, IA32_PAT, IA32_TSC, IA32_TSC_ADJUST, SVM_KEY, TSC_RATIO, VM_CR},
    permissions::{MSRPM_BYTES, msrpm_position},
};
use x86_64::{
    PhysAddr,
    instructions::interrupts,
    registers::{control::EferFlags, model_specific::Msr},
    structures::paging::PhysFrame,
};

use crate::{
    Host, Invalid, Registers, VcpuError, invalid,
    registers::{RAX, RSP},
    switch,
};

/// The permission bits for the registers this layer always intercepts.
///
/// Three protect the machine's virtualization extension, two define the
/// guest's offset timestamp domain, the ratio register is hidden because the
/// extension itself is hidden, and the page-attribute table is virtualized
/// because under nested paging the guest's copy of it is a field of this block
/// rather than the register. Keeping them here makes their permission bits part
/// of every VCPU rather than policy a caller could forget to install.
const INTERCEPTED_MSRS: [u32; 7] = [
    VM_CR,
    SVM_KEY,
    EFER,
    IA32_TSC,
    IA32_TSC_ADJUST,
    TSC_RATIO,
    IA32_PAT,
];

/// What a guest's control block has to be told about the guest before it can
/// run at all.
///
/// Two values, and both belong to the guest as a whole rather than to any one
/// of its processors: where its memory is described, and what its translations
/// are tagged with. Everything else a control block needs is either fixed by
/// the architecture or is guest state that nothing here writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Guest {
    /// Root of the nested page tables the guest's physical addresses are
    /// translated by.
    pub nested_cr3: PhysAddr,
    /// Which address space the guest's translations are tagged with. Never
    /// zero: zero is the host's own, and a guest given it is refused.
    pub asid: u32,
}

/// Whether to enter the guest again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flow {
    /// Enter the guest again.
    Resume,
    /// Stop, and hand control back to whoever started the loop.
    Leave,
}

/// One virtual processor.
///
/// Not [`Send`] and not [`Sync`], and deliberately: a control block belongs to
/// the processor that has been entering it, because that processor is the one
/// whose cache is keyed on its address.
#[derive(Debug)]
pub struct Vcpu {
    registers: Registers,
    vmcb: NonNull<Vmcb>,
    vmcb_phys: PhysAddr,
    msrpm_phys: PhysAddr,
    host: &'static Host,
    tsc_adjust: u64,
    dirty: CleanBits,
    stale: bool,
}

impl Vcpu {
    /// Allocates a control block and programs the little of it that is not
    /// guest state.
    ///
    /// What is written is the mandatory intercept, nested paging and the two
    /// values in [`Guest`]. What is not written is every register the guest
    /// will run with: no instruction pointer, no stack pointer, no
    /// segments, no control registers. Those are a separate decision and a
    /// separate change, and a block in this state is infrastructure rather
    /// than a runnable guest — [`Vcpu::validate`] will say so.
    ///
    /// The block starts as a page of zeroes, which is what the architecture
    /// asks for: every reserved byte zero, and no clean bit set, which is
    /// what a block the processor has never seen must say.
    ///
    /// # Errors
    ///
    /// [`VcpuError::OutOfFrames`] if the chunk cannot spare a page, or
    /// [`VcpuError::Unreachable`] if the window onto physical memory does not
    /// reach it.
    pub fn create(
        host: &'static Host,
        frames: &mut Frames,
        window: DirectMap,
        guest: Guest,
    ) -> Result<Self, VcpuError> {
        let vmcb_phys = frames
            .allocate(0)
            .map_err(|_| VcpuError::OutOfFrames)?
            .start_address();
        let vmcb = window
            .ptr::<Vmcb>(vmcb_phys)
            .map_err(|_| VcpuError::Unreachable {
                phys: vmcb_phys.as_u64(),
            })?;
        let msrpm_phys = frames
            .allocate(1)
            .map_err(|_| VcpuError::OutOfFrames)?
            .start_address();
        let mut msrpm =
            window
                .ptr::<[u8; MSRPM_BYTES]>(msrpm_phys)
                .map_err(|_| VcpuError::Unreachable {
                    phys: msrpm_phys.as_u64(),
                })?;
        // SAFETY: the two-page run was just allocated to this VCPU, is zeroed,
        // and `msrpm` reaches all of it through the direct map.
        let msrpm = unsafe { msrpm.as_mut() };
        intercept(msrpm, INTERCEPTED_MSRS);
        let tsc_adjust = if processor::features().contains(Features::TSC_ADJUST) {
            // SAFETY: the feature bit establishes that the register exists, and
            // reading it has no side effects.
            unsafe { Msr::new(IA32_TSC_ADJUST).read() }
        } else {
            0
        };
        let mut vcpu = Self {
            registers: Registers::zeroed(),
            vmcb,
            vmcb_phys,
            msrpm_phys,
            host,
            tsc_adjust,
            // Everything counts as edited until the first entry, so that entry
            // publishes a clean field of zero.
            dirty: CleanBits::ALL_CACHED,
            // A guest that has never run has cached no translation of its own.
            stale: false,
        };

        let control = vcpu.control_mut();
        // A guest permitted to enter a guest of its own could run one with a
        // control block this hypervisor never inspected, so the architecture
        // refuses to start a guest without this and it is not a policy choice.
        control.intercept_1 = Intercepts1::MSR_PROT;
        control.intercept_2 = Intercepts2::from_flags(Intercepts2Flags::VMRUN);
        control.msr_permissions = msrpm_phys.as_u64();
        control.nested_paging = NestedPagingControl::new().with_enabled(true);
        control.nested_cr3 = guest.nested_cr3.as_u64();
        control.asid = guest.asid;
        Ok(vcpu)
    }

    /// Gives the control block's page back to the chunk.
    ///
    /// By value, because a control block the processor may still have cached
    /// under its physical address must not be reachable afterwards — and taking
    /// the virtual processor apart is what guarantees nothing enters it again.
    ///
    /// # Errors
    ///
    /// [`VcpuError::Paging`] if the allocator does not recognize the page,
    /// which would mean it never came from the chunk.
    ///
    /// # Safety
    ///
    /// This virtual processor must not be running: no `VMRUN` naming this block
    /// may be in flight on any processor, and the page is handed to whoever
    /// allocates next.
    pub unsafe fn release(self, frames: &mut Frames) -> Result<(), VcpuError> {
        frames.release(PhysFrame::containing_address(self.msrpm_phys), 1)?;
        Ok(frames.release(PhysFrame::containing_address(self.vmcb_phys), 0)?)
    }

    /// Runs the guest until `exits` says to stop.
    ///
    /// `exits` is called once per exit, with the host's own state back in the
    /// processor and both interrupt flags — the global one and the ordinary
    /// one — set, so it may do anything the hypervisor can normally do,
    /// including taking an interrupt, which is how an interrupt that arrived
    /// while the guest was running reaches the host's handler. The ordinary
    /// flag is also left set when this returns, because it was set to enter
    /// the loop and nothing here clears it again.
    ///
    /// # Errors
    ///
    /// [`VcpuError::Invalid`] if the control block breaks one of the
    /// architecture's rules, either as this found it or as the processor
    /// reported it. Nothing is rolled back: a block the processor refuses has
    /// run no guest instruction, so there is nothing to roll back.
    ///
    /// # Safety
    ///
    /// The control block must not have been entered on another processor since
    /// it was last entered on this one, and must not have been moved to a
    /// different physical page — the processor identifies its cached copy of a
    /// block by that address alone, so either would run the guest on state
    /// belonging to something else. Neither can happen to a virtual processor
    /// that stays on the physical processor it was created on.
    ///
    /// The guest state in the block must describe a guest this hypervisor is
    /// entitled to run: entering one is handing it the processor.
    pub unsafe fn run(
        &mut self,
        mut callback: impl FnMut(&mut Self, RunPhase) -> Flow,
    ) -> Result<(), VcpuError> {
        if let Some(invalid) = self.validate() {
            return Err(VcpuError::Invalid(invalid));
        }

        loop {
            interrupts::disable();
            // SAFETY: `Host::install` enabled SVM on this processor. VMRUN
            // restores GIF on the successful path; the early return below
            // restores it explicitly.
            unsafe { switch::disable_global_interrupts() };
            if callback(self, RunPhase::Enter) == Flow::Leave {
                // SAFETY: GIF was cleared immediately above and no VMRUN has
                // occurred to restore it.
                unsafe { switch::enable_global_interrupts() };
                interrupts::enable();
                return Ok(());
            }
            let clean = CleanBits::ALL_CACHED.soil(self.dirty);
            let flush = if self.stale {
                flush_command()
            } else {
                TlbControl::DoNothing
            };
            let control = self.control_mut();
            control.clean = clean;
            control.tlb_control = flush;
            self.dirty = CleanBits::nothing_cached();
            self.stale = false;

            // With interrupt masking virtualized, SVM takes the host's IF at
            // entry as the mask for physical interrupts while the guest runs.
            // GIF remains clear until VMRUN, so enabling IF here cannot deliver
            // anything into the half-restored state below. The helper includes
            // a NOP, which consumes STI's one-instruction interrupt shadow
            // before VMRUN samples the flag.
            interrupts::enable();

            // SAFETY: the block was checked above and after every exit that
            // edited it, `Host::install` enabled the extension and programmed
            // the host state-save address on this processor, and the snapshot
            // comes from that same call on this same processor. The caller
            // guarantees the block has not moved and has not been entered
            // elsewhere. GIF is clear across the final preparation and switch,
            // so physical interrupts cannot observe partially restored state.
            unsafe {
                switch::enter(&mut self.registers, self.vmcb_phys, self.host.snapshot());
            }

            if self.control().exit_code == ExitCode::INVALID {
                // No guest instruction ran, so resuming would produce this exit
                // again forever. The check is re-run rather than reported bare,
                // because the whole value of this exit is the rule behind it.
                return Err(VcpuError::Invalid(
                    self.validate().unwrap_or(Invalid::Unexplained),
                ));
            }
            if callback(self, RunPhase::Exit) == Flow::Leave {
                return Ok(());
            }
        }
    }

    /// The first rule of the architecture's this control block breaks, or
    /// `None` if it breaks none.
    ///
    /// Worth calling after editing a block and before running it again:
    /// entering with a block the processor refuses reports one code and no
    /// detail, where this reports the rule.
    #[must_use]
    pub fn validate(&self) -> Option<Invalid> {
        invalid::check(
            self.control(),
            self.save(),
            processor::physical_address_bits(),
        )
    }

    /// Marks the groups of the control block that have just been edited, so the
    /// next entry makes the processor read them again.
    ///
    /// Clearing more than was edited only costs a re-read; clearing less runs
    /// the guest on state that is no longer there. When unsure, clear.
    pub const fn soil(&mut self, groups: CleanBits) {
        self.dirty = self.dirty.union(groups);
    }

    /// Discards this guest's cached translations on the way into the next
    /// entry.
    ///
    /// What a caller calls after taking permission away in the guest's nested
    /// page tables. The processor may hold a translation those tables no longer
    /// justify, and nothing but a flush gets rid of it — filling an entry that
    /// was empty needs none of this, because the walker notices a constraint
    /// being lifted on its own.
    ///
    /// Consumed by the entry it applies to, so one edit costs one flush rather
    /// than a flush on every entry for the rest of the guest's life.
    pub const fn flush(&mut self) {
        self.stale = true;
    }

    /// The timestamp counter value the guest observes now.
    ///
    /// Hardware applies the same wrapping addition to native `RDTSC` and
    /// `RDTSCP` in guest mode. This is the matching answer for an intercepted
    /// read of the architectural timestamp-counter register.
    #[must_use]
    pub fn guest_timestamp(&self) -> u64 {
        processor::timestamp().wrapping_add(self.tsc_offset())
    }

    /// The offset hardware adds to this guest's timestamp counter.
    #[must_use]
    pub fn tsc_offset(&self) -> u64 {
        self.control().tsc_offset
    }

    /// The page-attribute table this guest's own mappings are interpreted
    /// through.
    ///
    /// Under nested paging this field is the guest's `IA32_PAT` and the
    /// register of that name is the host's, so this is what an intercepted
    /// read of it has to answer with. Reaching for the machine's register
    /// instead would tell the guest about memory types that are not the
    /// ones its mappings get.
    #[must_use]
    pub fn guest_pat(&self) -> u64 {
        self.save().g_pat
    }

    /// Sets it, or reports the rule that makes the value one the processor
    /// would refuse the guest for.
    ///
    /// Refused rather than stored, for the reason a reserved bit of the
    /// extended feature register is: the entry check examines this field
    /// whenever nested paging is on, so a save area holding an encoding the
    /// architecture does not define stops the guest with a control block
    /// the hypervisor cannot enter — where what the guest is owed is the
    /// general protection fault its own architecture promised it for the
    /// `WRMSR`.
    ///
    /// # Errors
    ///
    /// [`Invalid::PatEncoding`] naming the field and the encoding, for a value
    /// with a byte that is not a memory type.
    pub fn set_guest_pat(&mut self, value: u64) -> Result<(), Invalid> {
        if let Some(invalid) = invalid::pat(value) {
            return Err(invalid);
        }
        if value == self.save().g_pat {
            return Ok(());
        }
        self.save_mut().g_pat = value;
        self.soil(CleanBits::NESTED_PAGING);
        Ok(())
    }

    /// Changes the offset hardware adds to this guest's timestamp counter.
    ///
    /// The field belongs to the intercept clean group, so changing it also
    /// makes the next entry reload that group rather than using a cached copy.
    pub fn set_tsc_offset(&mut self, offset: u64) {
        if self.control().tsc_offset == offset {
            return;
        }
        self.control_mut().tsc_offset = offset;
        self.soil(CleanBits::INTERCEPTS);
    }

    /// The guest's cumulative timestamp-counter adjustment.
    #[must_use]
    pub const fn tsc_adjust(&self) -> u64 {
        self.tsc_adjust
    }

    /// Stores the guest's cumulative timestamp-counter adjustment.
    pub const fn set_tsc_adjust(&mut self, adjustment: u64) {
        self.tsc_adjust = adjustment;
    }

    /// Why the guest stopped, or `None` for a code the architecture does not
    /// define.
    #[must_use]
    pub fn reason(&self) -> Option<Reason> {
        self.control().exit_code.reason()
    }

    /// The general-purpose registers the hardware leaves to the hypervisor.
    #[must_use]
    pub const fn registers(&self) -> &Registers {
        &self.registers
    }

    /// The same, to write.
    pub const fn registers_mut(&mut self) -> &mut Registers {
        &mut self.registers
    }

    /// The register an encoded four-bit number names.
    ///
    /// The two the hardware carries itself come from the state-save area and
    /// the rest from the register block, which is what lets a caller act on
    /// the operand an intercept reported without knowing where it lives.
    #[must_use]
    pub fn gpr(&self, number: u8) -> u64 {
        let number = number & NUMBER;
        match self.registers.by_number(number) {
            Some(value) => value,
            None if number == RAX => self.save().rax,
            // The block holds all sixteen but these two, so this is the other.
            None => self.save().rsp,
        }
    }

    /// Writes the register an encoded four-bit number names.
    ///
    /// Soils the clean field where the register is one the processor caches.
    /// Neither the accumulator nor the stack pointer is cached, so writing
    /// either needs nothing cleared.
    pub fn set_gpr(&mut self, number: u8, value: u64) {
        let number = number & NUMBER;
        if self.registers.set_by_number(number, value) {
            return;
        }
        if number == RAX {
            self.save_mut().rax = value;
        } else {
            self.save_mut().rsp = value;
        }
    }

    /// How the guest runs, and how it stopped.
    #[must_use]
    pub fn control(&self) -> &ControlArea {
        // SAFETY: the block is a page of the reserved chunk this virtual
        // processor owns, reached through the window, and `&self` is what keeps
        // it from being written while this reference lives. The processor writes
        // it only during `VMRUN`, which needs `&mut self`.
        unsafe { &self.vmcb.as_ref().control }
    }

    /// The same, to write. Editing it means calling [`Vcpu::soil`] with what
    /// was edited.
    pub fn control_mut(&mut self) -> &mut ControlArea {
        // SAFETY: as in `control`, with `&mut self` making this the only
        // reference.
        unsafe { &mut self.vmcb.as_mut().control }
    }

    /// Makes the guest's reads and writes of these model-specific registers
    /// exit instead.
    ///
    /// Which registers a guest may not have is policy, and policy does not
    /// belong here — this takes whichever ones the caller names and says
    /// nothing about what they are for. The registers this layer intercepts on
    /// its own account are the virtualization registers a guest could use to
    /// reach the host's extension and the timestamp registers whose semantics
    /// depend on this control block's offset.
    ///
    /// An index outside the three ranges the architecture gives the permission
    /// map is skipped rather than refused: the map cannot express one, so the
    /// guest reaches it directly, and that is a fact about the architecture
    /// rather than a mistake by the caller.
    ///
    /// # Errors
    ///
    /// [`VcpuError::Unreachable`] if the permission map cannot be reached
    /// through `window`.
    pub fn intercept_msrs(
        &mut self,
        window: DirectMap,
        msrs: impl IntoIterator<Item = u32>,
    ) -> Result<(), VcpuError> {
        let mut map = window
            .ptr::<[u8; MSRPM_BYTES]>(self.msrpm_phys)
            .map_err(|_| VcpuError::Unreachable {
                phys: self.msrpm_phys.as_u64(),
            })?;
        // SAFETY: this map belongs to this VCPU, was allocated for it and is
        // reached through the direct map. The guest is not running: a VCPU is
        // not `Send`, so the only processor that could have entered it is this
        // one, and this one is here.
        let map = unsafe { map.as_mut() };
        intercept(map, msrs);
        self.soil(CleanBits::PERMISSION_MAPS);
        Ok(())
    }

    /// Puts this virtual processor into the state a real one is in when a
    /// start-up message naming `page` has released it.
    ///
    /// What a hypervisor does when the guest starts one of its own processors.
    pub fn start_at(&mut self, page: u8) {
        self.reborn(SaveArea::started_at(page));
    }

    /// Puts this virtual processor into the state a real one is in when an
    /// `INIT` has reset it, which is executing at the machine's reset vector.
    ///
    /// What a hypervisor does when the guest resets the processor it came up
    /// on: only application processors are held waiting to be started,
    /// because the bootstrap processor is the one that starts them.
    pub fn restart(&mut self) {
        self.reborn(SaveArea::at_reset());
    }

    /// Replaces everything this virtual processor was with the state a
    /// processor coming out of reset has.
    ///
    /// Nothing is edited: the whole state-save area is replaced, because a
    /// processor coming out of reset keeps nothing of what it was doing before,
    /// and an edit would leave whatever this virtual processor was last running
    /// showing through wherever the reset state happens to agree with zero.
    ///
    /// Two values the architecture would leave at reset are supplied anyway,
    /// and both are forced rather than chosen. The virtualization-enable bit,
    /// without which the processor refuses to enter the guest at all. And the
    /// guest's page-attribute table, which under nested paging is what its page
    /// tables are interpreted through — left at its own reset value of zero,
    /// every one of the guest's mappings would be uncacheable, which is correct
    /// and slow enough to look like a machine that has stopped.
    ///
    /// The cached copy of the block is abandoned in full, and so are this
    /// guest's cached translations on this processor: everything the processor
    /// remembers about this virtual processor describes one that no longer
    /// exists.
    fn reborn(&mut self, mut save: SaveArea) {
        save.efer |= EferFlags::SECURE_VIRTUAL_MACHINE_ENABLE.bits();
        save.g_pat = paging::cpu::PAT_POLICY;
        *self.save_mut() = save;
        // Reset clears every general-purpose register but one: the data register
        // holds the processor's own family, model and stepping, which is how
        // sixteen-bit code that predates `CPUID` identified what it was running
        // on. It is this processor's signature because that is the processor the
        // guest is running on.
        self.registers = Registers::zeroed();
        self.registers.rdx = u64::from(processor::cpuid(FEATURE_LEAF, 0).eax);
        self.soil(CleanBits::ALL_CACHED);
        self.flush();
    }

    /// The register state the guest runs with, and stopped in.
    #[must_use]
    pub fn save(&self) -> &SaveArea {
        // SAFETY: as in `control`.
        unsafe { &self.vmcb.as_ref().save }
    }

    /// The same, to write.
    pub fn save_mut(&mut self) -> &mut SaveArea {
        // SAFETY: as in `control_mut`.
        unsafe { &mut self.vmcb.as_mut().save }
    }

    /// Where the control block is, which is the address the processor is given
    /// and the identity its cache is keyed on.
    #[must_use]
    pub const fn vmcb(&self) -> PhysAddr {
        self.vmcb_phys
    }

    /// Logs what this virtual processor is, and whether it would run.
    pub fn describe(&self, who: &str) {
        let control = self.control();
        info!(
            "{who}: vcpu control block at {:#x}, asid {}, npt at {:#x}",
            self.vmcb_phys, control.asid, control.nested_cr3,
        );
        match self.validate() {
            Some(invalid) => info!("{who}: vcpu would not enter: {invalid}"),
            None => info!("{who}: vcpu would enter"),
        }
    }
}

/// Which side of a guest transition [`Vcpu::run`] is invoking its callback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunPhase {
    /// The control block is prepared and the guest is about to be entered.
    Enter,
    /// The guest has exited and the callback may inspect and answer the exit.
    Exit,
}

/// Sets both permission bits of every named register in a permission map.
///
/// An index the map does not cover is skipped: there is no bit for one, and an
/// access to it is intercepted unconditionally anyway.
fn intercept(map: &mut [u8; MSRPM_BYTES], msrs: impl IntoIterator<Item = u32>) {
    for msr in msrs {
        let Some(permission) = msrpm_position(msr) else {
            continue;
        };
        map[permission.read.byte] |= permission.read.mask();
        map[permission.write.byte] |= permission.write.mask();
    }
}

/// The narrowest way this processor can discard one guest's translations.
///
/// Flushing by identifier throws away this guest's and nothing else's. A
/// processor without it has only the instrument that discards every translation
/// on the machine, the host's included — enormously more expensive, and still
/// correct, which is what makes it the fallback rather than a refusal.
fn flush_command() -> TlbControl {
    match processor::svm() {
        Some(svm) if svm.features.contains(SvmFeatures::FLUSH_BY_ASID) => TlbControl::FlushGuest,
        _ => TlbControl::FlushAll,
    }
}

/// Bits of an encoded register number that name a register.
///
/// Every field the architecture reports an intercepted instruction's operand in
/// is four bits wide, so a wider value is not a register that this does not
/// know about — it is bits that were never part of the number.
const NUMBER: u8 = 0xF;

/// The `CPUID` leaf whose accumulator result is the processor's signature,
/// which is what reset leaves in the data register.
const FEATURE_LEAF: u32 = 1;

const _: () = assert!(
    RSP == 4 && RAX == 0,
    "the two registers the hardware carries are the ones the save area holds",
);
const _: () = assert!(
    RAX & NUMBER == RAX && RSP & NUMBER == RSP,
    "both must survive the mask, or the arms testing for them are unreachable",
);
const _: () = {
    let mut index = 0;
    while index < INTERCEPTED_MSRS.len() {
        assert!(
            msrpm_position(INTERCEPTED_MSRS[index]).is_some(),
            "every register this layer intercepts must have a bit in the permission map",
        );
        index += 1;
    }
};
