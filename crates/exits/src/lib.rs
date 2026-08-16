//! What a guest's exits mean.
//!
//! [`vcpu`] runs a guest and hands back every exit without interpreting one,
//! deliberately: what an exit means is policy. This is that policy — the loop
//! that answers each exit and decides whether the guest carries on.
//!
//! # What is answered, and how
//!
//! - `CPUID`, answered with the machine's own answer less the virtualization
//!   extension, so the guest does not discover the hypervisor underneath it.
//! - The model-specific registers that decide whether that extension may be
//!   used, answered consistently with that — and the extended feature register
//!   whose enable bit would otherwise contradict it.
//! - Every other model-specific register the permission map cannot cover, which
//!   the processor intercepts whatever that map says. Those reach the machine's
//!   own register, and one the machine does not have is a fault the guest
//!   takes.
//! - Nested page faults, which are how a guest's memory comes to be described
//!   at all, and how a write to hypervisor memory is stepped over.
//! - The two notifications the [`portal`] makes, which are the only two things
//!   the guest ever deliberately tells the host.
//!
//! Anything else stops the guest. A hypervisor that resumed an exit it did not
//! understand would be resuming a guest whose state it did not fix, and the
//! same exit would arrive again forever.
//!
//! # Firmware's guest passes through three stages
//!
//! The first guest is firmware, entered at the portal, and the host owes it
//! something different at each stage of its life. It starts inside the portal
//! with boot services live and the wrapper installed; it is handed the machine
//! when the wrapper reports `ExitBootServices` succeeded, which is where the
//! other processors are started; and the portal is gone from it at the first
//! exit taken outside those pages, which is where they are taken back and the
//! guest stops being able to see any hypervisor memory at all.
//!
//! # Every other processor waits to be started, and can be stopped again
//!
//! Only one processor is entered at the portal. The rest reach the guest
//! through [`Exits::joining`], and reach it held: the guest has never started
//! them, so their emulated controllers hold them exactly as reset holds a
//! processor that has had no start-up message. They halt rather than spin, and
//! the guest starting one is what releases it — at which point the processor
//! begins in real mode at the page the message named, in the state a processor
//! coming out of reset is in.
//!
//! That is not a one-way door. An operating system may reset a processor it has
//! already started, so the loop below is written for a guest that can stop
//! being runnable at any exit: a startup message takes the processor back out
//! of the guest, back to being held, and back in again when it is started
//! afresh.

#![no_std]

mod census;
mod cpuid;
mod firmware;
mod hidden_svm;
mod msr;
mod nested;

use core::convert::Infallible;

use inject::{Injected, Pending};
use log::{error, info, trace};
use partition::{Addressing, Partition};
use portal::Portal;
use svm::{CleanBits, Reason};
use thiserror::Error;
use vcpu::{Flow, RunPhase, Vcpu, VcpuError};
use vlapic::{Resumption, VlapicError};
use x86_64::instructions::interrupts;

pub use crate::firmware::Boot;
use crate::{census::Census, firmware::Firmware, msr::Virtualization};

/// The guest's exits, and everything the host needs to answer one.
///
/// One of these per processor, holding the things that are one processor's:
/// what its guest is owed, how far it has got out of firmware — which for every
/// processor but one is "there was no firmware" — and what it has been told
/// about this machine's virtualization extension. The guest's memory is
/// borrowed rather than held, because that part really is shared by all of
/// them.
#[derive(Debug)]
pub struct Exits<'a> {
    partition: &'a Partition,
    firmware: Option<Firmware>,
    interrupts: Pending,
    virtualization: Virtualization,
    left: Left,
    census: Census,
}

impl<'a> Exits<'a> {
    /// What will answer for a guest that is entered at `portal`.
    #[must_use]
    pub fn new(partition: &'a Partition, portal: Portal, boot: Boot) -> Self {
        Self {
            partition,
            firmware: Some(Firmware::new(portal, boot)),
            interrupts: Pending::new(),
            virtualization: Virtualization::new(),
            left: Left::Stopped,
            census: Census::new(),
        }
    }

    /// What will answer on a processor joining a guest that already exists.
    ///
    /// No portal, and nothing to do about firmware: the portal is the way
    /// *into* the guest and only one processor goes that way. This one
    /// reaches the same guest by being started by it, which cannot happen
    /// until firmware is long finished.
    #[must_use]
    pub fn joining(partition: &'a Partition) -> Self {
        Self {
            partition,
            firmware: None,
            interrupts: Pending::new(),
            virtualization: Virtualization::new(),
            left: Left::Stopped,
            census: Census::new(),
        }
    }

    /// Runs the guest until an exit nothing here answers for.
    ///
    /// Returns only by failing: a processor whose guest is runnable stays in
    /// this, and one whose guest is not waits here for it to become runnable
    /// again rather than returning — there is nothing else for a processor of a
    /// running guest to be doing.
    ///
    /// # Errors
    ///
    /// [`ExitError::Vcpu`] if the processor refuses the control block, naming
    /// the rule it breaks, [`ExitError::Stopped`] if an exit was one this crate
    /// could not answer for, or [`ExitError::Vlapic`] if this processor's
    /// emulated controller cannot be reached — without which nothing can decide
    /// whether the guest should be running at all. Nothing is rolled back: the
    /// guest is not entered again either way.
    ///
    /// # Safety
    ///
    /// As [`Vcpu::run`]: the control block must not have moved or been entered
    /// on another processor since this processor last entered it, and the guest
    /// state in it must describe a guest this hypervisor is entitled to run.
    pub unsafe fn run(&mut self, vcpu: &mut Vcpu) -> Result<Infallible, ExitError> {
        // Interrupts are taken away from the guest before it is ever entered,
        // rather than at the first exit: one arriving during the first
        // instruction the guest runs must already be the host's.
        inject::arm(vcpu);
        interrupts::enable();

        loop {
            // Waiting is not idleness: a processor whose guest has been reset
            // has nothing to run until another one starts it, which may never
            // happen, and spinning on that would burn a core for the life of the
            // machine.
            while !self.startable(vcpu)? {
                vlapic::hold()?;
            }
            // Deciding what the guest takes can also discover that there is no
            // longer a guest to give it to, because a startup message can arrive
            // at any point up to the entry itself.
            // SAFETY: forwarded to the caller, whose obligations are
            // `Vcpu::run`'s in full.
            unsafe {
                vcpu.run(|vcpu, phase| match phase {
                    RunPhase::Enter => self.enter(vcpu),
                    RunPhase::Exit => self.exit(vcpu),
                })
            }?;
            if self.left == Left::Stopped {
                return Err(ExitError::Stopped);
            }
        }
    }

    /// Whether this processor's guest may be entered, having applied whatever
    /// startup message arrived for it.
    ///
    /// The one place a virtual processor is built out of nothing. A guest that
    /// has just been told where to start gets the state a real processor has
    /// coming out of reset, and everything the host was holding for the guest
    /// that ran here before is dropped with it — an interrupt owed to a
    /// processor that has since been reset is owed to nobody.
    fn startable(&mut self, vcpu: &mut Vcpu) -> Result<bool, ExitError> {
        match vlapic::settle()? {
            Resumption::Carry => Ok(true),
            Resumption::Wait => Ok(false),
            Resumption::StartAt(page) => {
                info!(
                    "exits: the guest started this processor at page {:#x}",
                    page.number()
                );
                vcpu.start_at(page.number());
                self.interrupts.reset(vcpu);
                self.virtualization.reset();
                Ok(true)
            }
            // The bootstrap processor's answer to the same message, and the only
            // one that does not wait: it is the processor that sends the start-up
            // messages, so nothing would ever release it.
            Resumption::Restart => {
                info!("exits: the guest reset this processor, which restarts at the reset vector");
                vcpu.restart();
                self.interrupts.reset(vcpu);
                self.virtualization.reset();
                Ok(true)
            }
        }
    }

    /// Answers one exit.
    ///
    /// The order is not a matter of taste. An event whose delivery this exit
    /// interrupted has to be taken back before anything can overwrite the
    /// injection field; the guest's task priority has to be read out of the
    /// control block before anything consults it, because with virtualized
    /// interrupt masking the guest changes it without exiting; and both have to
    /// happen before the exit is answered, because answering one can send this
    /// processor an interrupt.
    fn exit(&mut self, vcpu: &mut Vcpu) -> Flow {
        let reason = vcpu.reason();
        self.census.record(vcpu);
        vlapic::observe_task_priority(vcpu.control().interrupt_control.virtual_tpr());
        // This processor is out of the guest and consults its controller below
        // before going back in, so nothing needs to interrupt it to make it
        // look.
        let _ = vlapic::set_away(false);
        self.interrupts.complete_iret(vcpu);
        let next_rip = if Pending::needs_next_rip(vcpu) {
            let addressing = Addressing::from_save(vcpu.save());
            match self
                .partition
                .with_memory(addressing, |guest| emulate::next_rip(vcpu, guest))
            {
                Ok(next_rip) => Some(next_rip),
                Err(error) => {
                    error!("exits: could not recover an interrupted software event: {error}");
                    self.left = Left::Stopped;
                    return Flow::Leave;
                }
            }
        } else {
            None
        };
        self.interrupts.harvest(vcpu, next_rip);
        // Before the exit is answered rather than after, because answering one
        // can resume the guest and the portal must be gone by the time it runs
        // again.
        if let Some(firmware) = &mut self.firmware {
            firmware.retire(vcpu, self.partition);
        }
        let flow = match reason {
            Some(Reason::Cpuid) => cpuid::exit(vcpu),
            Some(Reason::MsrAccess) => self.virtualization.exit(vcpu, &mut self.interrupts),
            Some(Reason::NestedPageFault) => {
                nested::exit(vcpu, self.partition, &mut self.interrupts)
            }
            Some(Reason::Vmmcall) => self.notified(vcpu),
            Some(
                Reason::Vmrun
                | Reason::Vmload
                | Reason::Vmsave
                | Reason::Stgi
                | Reason::Clgi
                | Reason::Skinit
                | Reason::Invlpga,
            ) => hidden_svm::refuse(vcpu, &mut self.interrupts),
            // Two exits that are answered by the fact of having happened.
            //
            // The first says the guest became willing to take an interrupt: the
            // window was armed to produce exactly this exit, and what to inject
            // is decided below for every exit alike. The second says a physical
            // interrupt arrived while the guest was running — it was taken by
            // the host at the world switch and has already been given to
            // whichever controller it was for, and the exit itself carries
            // nothing further.
            Some(Reason::VirtualInterrupt | Reason::Interrupt | Reason::Nmi) => Flow::Resume,
            // The guest left an interrupt handler, which ends the window during
            // which it takes no further non-maskable interrupt.
            Some(Reason::Iret) => {
                self.interrupts.intercepted_iret(vcpu);
                Flow::Resume
            }
            _ => {
                error!("exits: unhandled {:?}", vcpu.control().exit_code);
                Flow::Leave
            }
        };
        if flow == Flow::Leave {
            // Nothing here could answer the exit, whichever handler decided so.
            self.left = Left::Stopped;
            return flow;
        }
        Flow::Resume
    }

    /// Acts on one of the portal's two notifications, if this processor is the
    /// one that has a portal.
    fn notified(&mut self, vcpu: &mut Vcpu) -> Flow {
        let Some(firmware) = &mut self.firmware else {
            // The instruction is intercepted for the portal's sake alone, and
            // only the processor the guest was entered on ever goes through one.
            error!("exits: a processor with no portal issued VMMCALL");
            return Flow::Leave;
        };
        firmware.notified(vcpu)
    }

    /// Decides what the guest takes on its way back in, or that there is no
    /// longer a guest to enter.
    ///
    /// Everything asked of the controller here is a *last* look — nothing else
    /// will consult it before the guest is running again. So the store saying
    /// this processor has stopped watching comes before all of them, and
    /// that pairing is what stops any being lost to a processor that was
    /// entering the guest as the answer changed: a sender that misses the
    /// flag is one whose message one of these looks finds, and a
    /// look that misses the message is one the sender's doorbell interrupts.
    fn enter(&mut self, vcpu: &mut Vcpu) -> Flow {
        if self.interrupts.shutdown() {
            self.left = Left::Stopped;
            return Flow::Leave;
        }
        let _ = vlapic::set_away(true);
        // Before anything below reads the controller, because the interrupt
        // window one of them arms is judged by the processor against exactly
        // this field.
        Self::mirror_task_priority(vcpu);
        // A non-maskable interrupt another processor sent this one is held by
        // the controller, because the processor that sent it could not reach
        // what this exit loop owns.
        while vlapic::take_nmi().unwrap_or(false) {
            self.interrupts.raise_nmi();
            trace!("exits: this processor was owed a non-maskable interrupt");
        }
        // A startup message is not something an entry can carry: it says this
        // processor's guest no longer exists, so the loop takes the processor
        // back rather than entering a guest that has been reset out from under
        // it.
        if !vlapic::running().unwrap_or(true) {
            self.left = Left::Held;
            return Flow::Leave;
        }
        // Selected and committed as two steps, because only the second knows
        // whether the guest was actually given anything. The controller
        // nominates the highest-priority vector it has and keeps it requested;
        // the injection may then be declined — an event already part-way
        // through delivery goes first, a non-maskable interrupt outranks it, the
        // guest's interrupt window may be shut — and a controller that had
        // already consumed the request would have thrown the interrupt away.
        //
        // The second answer is the wider one, and is what an interrupt window
        // is armed for. A guest changes its task priority without exiting, so
        // the vector that priority is holding back has to be armed as well as
        // the one it admits — otherwise the guest lowering it is a change
        // nothing on this machine hears about.
        let candidate = vlapic::select().unwrap_or(None);
        let blocked = vlapic::pending().unwrap_or(None);
        let injected = self.interrupts.commit(vcpu, candidate, blocked);
        // Only when there was something to decide about. Every exit reaches
        // here, and a guest with nothing owed would otherwise describe that
        // several thousand times a second — but an entry that had a candidate,
        // or armed a window, or put something in is one of the few that says
        // where an interrupt went.
        if candidate.is_some() || blocked.is_some() || injected != Injected::Nothing {
            trace!(
                "exits: entering with candidate {candidate:?}, blocked {blocked:?}, injected \
                 {injected:?}, virtual tpr {:#x}",
                vcpu.control().interrupt_control.virtual_tpr()
            );
        } else {
            trace!(
                "exits: entering with candidate {candidate:?}, blocked {blocked:?}, injected \
                {injected:?}"
            );
        }
        if let Injected::Interrupt(vector) = injected {
            let _ = vlapic::committed(vector);
        }
        Flow::Resume
    }

    /// Brings the control block's virtual task priority into agreement with the
    /// emulated register.
    ///
    /// The guest's task priority is one register reached through two doors: the
    /// control block's copy, which the processor answers a guest's `CR8` access
    /// from and compares an armed interrupt window against, and the emulated
    /// register the guest writes through the page or a model-specific register.
    /// [`Exits::exit`] reads the first into the second on the way out; this
    /// writes the second back into the first on the way in, so that a guest
    /// which lowered its priority through the register file finds its own `CR8`
    /// answering the same number and its pending interrupts judged against it.
    ///
    /// Done on every entry rather than only after an exit, which is what covers
    /// the two entries that follow no exit at all: the first one, where the
    /// copy still reads zero and the emulated register holds what firmware
    /// left, and the one after a startup message, where the emulated
    /// register has just been reset and the copy has not.
    ///
    /// Only the class is held in the control block, which is the upper nibble
    /// of the emulated byte.
    fn mirror_task_priority(vcpu: &mut Vcpu) {
        let Ok(priority) = vlapic::task_priority() else {
            return;
        };
        let class = priority >> TPR_CLASS_SHIFT;
        if vcpu.control().interrupt_control.virtual_tpr() == class {
            return;
        }
        let control = vcpu.control_mut();
        control.interrupt_control = control.interrupt_control.with_virtual_tpr(class);
        vcpu.soil(CleanBits::INTERRUPT);
    }
}

/// Why the exit loop stopped entering the guest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Left {
    /// An exit nothing here could answer for, which ends the guest.
    Stopped,
    /// The guest reset this processor. There is nothing wrong and nothing to
    /// report: the processor waits to be started again.
    Held,
}

/// Why the guest stopped.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum ExitError {
    /// The processor would not enter the control block, or would not enter it
    /// again.
    #[error(transparent)]
    Vcpu(#[from] VcpuError),
    /// An exit was one nothing here could answer for, so the guest was left
    /// rather than resumed on state the host had not fixed.
    #[error("guest execution stopped after an unhandled VM exit")]
    Stopped,
    /// This processor's emulated controller could not be reached, so nothing
    /// could say whether its guest should be running at all.
    #[error(transparent)]
    Vlapic(#[from] VlapicError),
}

/// How far a task-priority byte is shifted to leave its interrupt-priority
/// class, which is the half of it the control block's virtual task priority
/// holds.
const TPR_CLASS_SHIFT: u8 = 4;

/// Moves the guest past an intercepted instruction the host has completed.
///
/// The processor reports the address of the following instruction where it can,
/// which costs nothing and is right for every encoding. `fallback` is the
/// instruction's length on a processor that does not — every instruction
/// intercepted here has exactly one encoding, so its length is a constant
/// rather than something that has to be decoded.
fn advance(vcpu: &mut Vcpu, fallback: u64) {
    let next = vcpu.control().next_rip;
    vcpu.save_mut().rip = if next == 0 {
        vcpu.save().rip.wrapping_add(fallback)
    } else {
        next
    };
}
