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
//! - The memory-type range registers, answered out of a copy of this
//!   processor's own. They decide nothing about a guest's memory under nested
//!   paging and everything about the host's, so a guest write must reach the
//!   copy and never the register.
//! - Every other model-specific register the permission map cannot cover, which
//!   the processor intercepts whatever that map says. Those reach the machine's
//!   own register, and one the machine does not have is a fault the guest
//!   takes.
//! - The two exits a hardware-driven interrupt controller raises: an
//!   inter-processor delivery the hardware could not finish, completed by the
//!   rule its failure names, and a register access it does not implement,
//!   either bookkept — the hardware finished it before it exited — or performed
//!   against the register page's device.
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

mod avic;
mod census;
mod cpuid;
mod firmware;
mod hidden_svm;
mod msr;
pub mod mtrr;
mod nested;

use core::convert::Infallible;

use inject::{Injected, Pending};
use log::{error, info, trace};
use partition::{Addressing, Partition};
use portal::Portal;
use svm::{CleanBits, Reason};
use thiserror::Error;
use vcpu::{Flow, RunPhase, Vcpu, VcpuError};
use vlapic::{Nomination, Resumption, VlapicError};
use x86_64::instructions::interrupts;

pub use crate::firmware::Boot;
use crate::{census::Census, firmware::Firmware, msr::Virtualization, mtrr::Mtrrs};

/// The guest's exits, and everything the host needs to answer one.
///
/// One of these per processor, holding the things that are one processor's:
/// what its guest is owed, how far it has got out of firmware — which for every
/// processor but one is "there was no firmware" — what it has been told about
/// this machine's virtualization extension, and the memory-type ranges it was
/// given in place of this core's own. The guest's memory is borrowed rather
/// than held, because that part really is shared by all of them.
#[derive(Debug)]
pub struct Exits<'a> {
    partition: &'a Partition,
    firmware: Option<Firmware>,
    interrupts: Pending,
    virtualization: Virtualization,
    mtrrs: Mtrrs,
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
            mtrrs: Mtrrs::seed(),
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
    ///
    /// The memory-type ranges are read here as they are on the processor that
    /// does go that way, and from this processor's own registers: what makes
    /// the guest's rendezvous over them find every processor agreeing is
    /// each one answering what firmware really left on the core it is
    /// running on.
    #[must_use]
    pub fn joining(partition: &'a Partition) -> Self {
        Self {
            partition,
            firmware: None,
            interrupts: Pending::new(),
            virtualization: Virtualization::new(),
            mtrrs: Mtrrs::seed(),
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
    /// injection field; the guest's task priority has to be taken out of
    /// whichever authority owned it before anything consults it, because the
    /// guest moves that register without exiting either way — through the
    /// control block where interrupt masking is virtualized, and through
    /// the backing page while the hardware drives its controller; and both
    /// have to happen before the exit is answered, because answering one
    /// can send this processor an interrupt.
    fn exit(&mut self, vcpu: &mut Vcpu) -> Flow {
        let reason = vcpu.reason();
        self.census.record(vcpu);
        let interrupts = vcpu.control().interrupt_control;
        vlapic::observe_task_priority(interrupts.avic_enable(), interrupts.virtual_tpr());
        // This processor is out of the guest and consults its controller below
        // before going back in, so nothing needs to interrupt it to make it
        // look.
        let _ = vlapic::set_away(false);
        // Withdrawn beside the away flag and for the same reason: a sender
        // that reads the running bit from here on takes the kick path, and
        // the rescan every park and every entry performs is what finds what
        // the kick is for.
        let _ = vlapic::avic_unpublish_running();
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
            Some(Reason::MsrAccess) => {
                self.virtualization
                    .exit(vcpu, &mut self.mtrrs, &mut self.interrupts)
            }
            Some(Reason::NestedPageFault) => {
                nested::exit(vcpu, self.partition, &mut self.interrupts)
            }
            Some(Reason::AvicIncompleteIpi) => avic::incomplete_ipi(vcpu, &mut self.census),
            Some(Reason::AvicUnacceleratedAccess) => avic::unaccelerated_access(
                vcpu,
                self.partition,
                &mut self.interrupts,
                &mut self.census,
            ),
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
            // The guest asked to be woken by an interrupt. What it is owed is in
            // software, so this is where the processor is parked rather than
            // inside a guest nothing would re-enter.
            Some(Reason::Hlt) => self.halted(vcpu),
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
        // The control block's acceleration follows the guest's own state at
        // every entry: the enable bit, the backing page and the tables move
        // here, and an entry that changes nothing pays one comparison for
        // it. A failure demotes to the software path below rather than
        // refusing the entry.
        if let Err(error) = vlapic::avic_reconcile(vcpu) {
            error!("exits: the interrupt acceleration could not be reconciled: {error}");
        }
        // Read out of the block rather than out of the controller, because the
        // block is what the processor is about to be entered with. The two agree
        // whenever the reconciliation above succeeded; where it could not, the
        // controller still asks for the hardware and the block was deliberately
        // left unarmed, and everything below has to serve the entry that is
        // actually going to happen.
        let accelerated = vcpu.control().interrupt_control.avic_enable();
        // Which of these two runs is which authority owns this processor's
        // controller for the run about to happen, and they are exclusive by
        // construction: one register, one writer, in each direction.
        if accelerated {
            // Everything the model is still holding that the hardware can
            // deliver goes into the backing page, so that the hardware both
            // delivers each vector and retires it. Nothing is injected below but
            // the one arrival no controller holds in service, and a failure here
            // leaves the interrupt in the model for the nomination to carry.
            if let Err(error) = vlapic::avic_hand_over() {
                error!(
                    "exits: the controller could not hand its interrupts to the hardware: {error}"
                );
            }
        } else {
            // Before anything below reads the controller, because the interrupt
            // window one of them arms is judged by the processor against exactly
            // this field.
            Self::mirror_task_priority(vcpu);
        }
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
        // Both answers come out of one look at the controller, and the second is
        // the wider one: what an interrupt window is armed for. A guest changes
        // its task priority without exiting, so the vector that priority is
        // holding back has to be armed as well as the one it admits — otherwise
        // the guest lowering it is a change nothing on this machine hears about.
        //
        // Under the acceleration the window is never armed at all: the
        // pending-interrupt fields are ignored on entry by a processor driving
        // the controller itself, and leaving one armed would exit on every
        // window the guest opens. What is nominated there is only what the
        // hand-over above could not give the hardware — an arrival the legacy pin
        // brought in, which no controller holds in service — and it is injected
        // outright the moment the guest is willing, which is all the software
        // path owes it.
        let nomination = if accelerated {
            Nomination {
                deliverable: vlapic::nominate().unwrap_or_default().deliverable,
                blocked: None,
            }
        } else {
            vlapic::nominate().unwrap_or_default()
        };
        let (candidate, blocked) = (nomination.deliverable, nomination.blocked);
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
        // Published last, when everything the entry prepared is in place and
        // the guest is about to run: from here until the exit, another
        // processor delivering an IPI may ring this one rather than exit.
        if accelerated {
            let _ = vlapic::avic_publish_running();
        }
        Flow::Resume
    }

    /// Parks this processor while its guest has nothing to take.
    ///
    /// The guest halted, which is a request to be woken by an interrupt, and
    /// what it is owed is in software: an interrupt window is a field read at
    /// an entry, so leaving the guest halted with one armed makes the
    /// wake-up depend on something re-entering it. Here the wait is the
    /// *host's* instead — the processor halts with its own interrupts
    /// enabled, where a physical arrival, a doorbell from another processor
    /// and a startup message all reach it — and the guest is re-entered
    /// with whatever that produced.
    ///
    /// The halt is stepped over first, because the wait below is what it asked
    /// for: coming back to it would halt twice for one request, and every wake
    /// would find the same instruction and do it again.
    ///
    /// Said to be away for the whole of the wait, and that is not decoration:
    /// a processor delivering into this controller only rings the doorbell for
    /// one that has stopped watching, and without the flag it would leave a
    /// request bit in a controller whose processor is asleep and set nothing to
    /// wake it.
    fn halted(&mut self, vcpu: &mut Vcpu) -> Flow {
        advance(vcpu, HLT_LENGTH);
        // Unpublished before the first look below, and the look is the point:
        // a request that lands between the withdrawal and the rescan is one
        // the rescan finds, and one that landed before it was answered by
        // the exit itself. The rescan inside [`Exits::wakeable`] reads the
        // backing page wherever the control block still has the acceleration
        // armed, which is where the request bits live.
        let _ = vlapic::avic_unpublish_running();
        let vcpu = &*vcpu;
        if self.wakeable(vcpu) {
            return Flow::Resume;
        }
        let _ = vlapic::set_away(true);
        trace!("exits: the guest halted with nothing to take; waiting for something to give it");
        while !self.wakeable(vcpu) {
            descriptors::wait_until(|| self.wakeable(vcpu));
        }
        let _ = vlapic::set_away(false);
        Flow::Resume
    }

    /// Whether there is anything for this processor's guest to be re-entered
    /// for.
    ///
    /// The exact question a halted processor asks, and it is the architecture's
    /// rather than a convenience: an interrupt the guest's own priority is
    /// holding back does not wake a halted processor on real hardware either,
    /// and a guest cannot lower that priority while it is halted. So a vector
    /// that is merely pending is not a reason, and the two things that reach a
    /// guest whatever it has masked are — a non-maskable interrupt, and a
    /// startup message from another processor.
    ///
    /// The backing page is consulted whenever the control block's own enable
    /// bit is set, and that is not the same condition as the acceleration
    /// being permitted. The permission can be withdrawn for the whole
    /// machine by any processor, while this one is parked and reaching no
    /// entry — and the page goes on collecting requests from every peer
    /// still driving its own block until each of them notices at its next
    /// entry. Judged by the permission, this processor would park again
    /// with the only copy of a vector in a page it had stopped consulting
    /// and the kick that announced it already spent. Judged by the enable
    /// bit, the entry this answer produces settles the page — it either
    /// leaves the hardware to deliver what is there or takes the page's
    /// state into the model, and an entry that clears the enable bit
    /// instead is one after which this is not asked again.
    fn wakeable(&self, vcpu: &Vcpu) -> bool {
        self.interrupts.owed()
            || !vlapic::running().unwrap_or(true)
            || (inject::interrupts_unmasked(vcpu)
                && (vlapic::nominate().unwrap_or_default().deliverable.is_some()
                    || (vcpu.control().interrupt_control.avic_enable()
                        && vlapic::avic_deliverable().unwrap_or(false))))
    }

    /// Brings the control block's virtual task priority into agreement with the
    /// emulated register.
    ///
    /// The guest's task priority is one register reached through two doors: the
    /// control block's copy, which the processor answers a guest's `CR8` access
    /// from and compares an armed interrupt window against, and the emulated
    /// register the guest writes through the page or a model-specific register.
    /// [`Exits::exit`] reads the first into the second on the way out, where
    /// the block is the authority for it; this writes the second back into
    /// the first on the way in, so that a guest which lowered its priority
    /// through the register file finds its own `CR8` answering the same
    /// number and its pending interrupts judged against it.
    ///
    /// Done on every entry the software delivers for, rather than only after an
    /// exit, which is what covers the two entries that follow no exit at all:
    /// the first one, where the copy still reads zero and the emulated
    /// register holds what firmware left, and the one after a startup
    /// message, where the emulated register has just been reset and the
    /// copy has not.
    ///
    /// Not done at all where the hardware drives the controller. The guest
    /// writes its task priority into the backing page there and the
    /// processor judges nothing against this field — the interrupt window
    /// it is compared with is one the acceleration ignores — so the model
    /// is given the page's byte at the exit instead, and the block's copy
    /// is left as the last software-driven entry wrote it.
    ///
    /// Only the class is held in the control block, which is the upper nibble
    /// of the emulated byte — and [`vlapic::Priority::class`] is where that
    /// narrowing is written, because the processor compares this field against
    /// the class the interrupt window is armed with and the two have to be the
    /// same nibble of the same rule.
    fn mirror_task_priority(vcpu: &mut Vcpu) {
        let Ok(priority) = vlapic::task_priority() else {
            return;
        };
        let class = priority.class();
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

/// How long the halt instruction is, for a processor that does not report the
/// address of the one after it.
const HLT_LENGTH: u64 = 1;

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
