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
use svm::{
    CleanBits, ControlArea, ExitCode, Reason, SaveArea, Vmcb,
    control::NestedPagingControl,
    intercept::{Intercepts2, Intercepts2Flags},
};
use x86_64::{PhysAddr, structures::paging::PhysFrame};

use crate::{
    Host, Invalid, Registers, VcpuError, invalid,
    registers::{RAX, RSP},
    switch,
};

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
    host: &'static Host,
    dirty: CleanBits,
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
            .ok_or(VcpuError::OutOfFrames)?
            .start_address();
        let vmcb = window
            .ptr::<Vmcb>(vmcb_phys)
            .ok_or(VcpuError::Unreachable {
                phys: vmcb_phys.as_u64(),
            })?;
        let mut vcpu = Self {
            registers: Registers::zeroed(),
            vmcb,
            vmcb_phys,
            host,
            // Everything counts as edited until the first entry, so that entry
            // publishes a clean field of zero.
            dirty: CleanBits::ALL_CACHED,
        };

        let control = vcpu.control_mut();
        // A guest permitted to enter a guest of its own could run one with a
        // control block this hypervisor never inspected, so the architecture
        // refuses to start a guest without this and it is not a policy choice.
        control.intercept_2 = Intercepts2::from_flags(Intercepts2Flags::VMRUN);
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
        Ok(frames.release(PhysFrame::containing_address(self.vmcb_phys), 0)?)
    }

    /// Runs the guest until `exits` says to stop.
    ///
    /// `exits` is called once per exit, with the global interrupt flag set and
    /// the host's own state back in the processor, so it may do anything the
    /// hypervisor can normally do — including taking an interrupt, which is how
    /// an interrupt that arrived while the guest was running reaches the host's
    /// handler.
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
        mut exits: impl FnMut(&mut Self) -> Flow,
    ) -> Result<(), VcpuError> {
        if let Some(invalid) = self.validate() {
            return Err(VcpuError::Invalid(invalid));
        }
        loop {
            let clean = CleanBits::ALL_CACHED.soil(self.dirty);
            self.control_mut().clean = clean;
            self.dirty = CleanBits::nothing_cached();

            // SAFETY: the block was checked above and after every exit that
            // edited it, `Host::install` enabled the extension and programmed
            // the host state-save address on this processor, and the snapshot
            // comes from that same call on this same processor. The caller
            // guarantees the block has not moved and has not been entered
            // elsewhere.
            unsafe { switch::enter(&mut self.registers, self.vmcb_phys, self.host.snapshot()) };

            if self.control().exit_code == ExitCode::INVALID {
                // No guest instruction ran, so resuming would produce this exit
                // again forever. The check is re-run rather than reported bare,
                // because the whole value of this exit is the rule behind it.
                return Err(VcpuError::Invalid(
                    self.validate().unwrap_or(Invalid::Unexplained),
                ));
            }
            if exits(self) == Flow::Leave {
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

/// Bits of an encoded register number that name a register.
///
/// Every field the architecture reports an intercepted instruction's operand in
/// is four bits wide, so a wider value is not a register that this does not
/// know about — it is bits that were never part of the number.
const NUMBER: u8 = 0xF;

const _: () = assert!(
    RSP == 4 && RAX == 0,
    "the two registers the hardware carries are the ones the save area holds",
);
const _: () = assert!(
    RAX & NUMBER == RAX && RSP & NUMBER == RSP,
    "both must survive the mask, or the arms testing for them are unreachable",
);
