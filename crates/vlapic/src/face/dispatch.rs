//! What reading or writing each register means.
//!
//! Both faces of the controller end up here, and that is the point. The
//! architecture defines a register's behaviour once and offers it through two
//! interfaces; a hypervisor that implemented the behaviour twice would be one
//! whose guest could tell which interface it was using by the answers it got.
//!
//! So the faces above this module do only what genuinely differs between them —
//! decoding an address, deciding whether a malformed access faults or merely
//! records an error, splitting a 64-bit register into halves — and every
//! question about what a register *is* is answered here.
//!
//! # Some reads are not reads
//!
//! Two registers change when read or write in ways that make the plain words
//! misleading, and both are called out where they are implemented: writing the
//! interrupt command register is how an interrupt is sent, and writing the
//! error status register is what makes a subsequent read report anything at
//! all.
//!
//! # And some writes ask for more than a store
//!
//! A guest's write to a controller is very often not merely a value: it can
//! send an interrupt to another processor, retire one on this one, or change
//! what real hardware is doing. What the register file itself cannot finish is
//! answered as a [`Written`] and performed by [`acted`], which is where the two
//! faces stop being two code paths.

use apic::LocalApic;
use descriptors::Vector;
use log::{trace, warn};

use crate::{
    delivery,
    face::table::{Bank, Register},
    hardware::{
        mirror::{entered, mirror_logical_destination},
        sources, timer,
    },
    lifecycle::settle,
    machine::registry::lapics,
    priority,
    registers::{
        Accepted, Vlapic,
        base::Transition,
        error::Errors,
        icr::{Command, Trigger},
        lvt::{Entry, TimerMode},
    },
};

/// What the guest sees when it reads a register.
///
/// Every register that is not simply stored is computed here rather than kept
/// up to date as things change, because the architecture defines most of them
/// as functions of other state and a stored copy is a copy that can be wrong.
pub(crate) fn read(vlapic: &Vlapic, register: Register) -> u32 {
    match register {
        Register::ID => vlapic.id_register(),
        Register::VERSION => vlapic.version(),
        Register::TASK_PRIORITY => u32::from(vlapic.task_priority().get()),
        Register::ARBITRATION_PRIORITY => u32::from(vlapic.arbitration_priority().get()),
        Register::PROCESSOR_PRIORITY => u32::from(vlapic.processor_priority().get()),
        Register::LOGICAL_DESTINATION => vlapic.logical_destination(),
        Register::DESTINATION_FORMAT => vlapic.destination_format(),
        Register::SPURIOUS => vlapic.spurious(),
        Register::ERROR_STATUS => vlapic.errors().read(),
        Register::COMMAND_LOW => vlapic.command().low(),
        Register::COMMAND_HIGH => vlapic.command().high(),
        Register::TIMER_INITIAL_COUNT => vlapic.timer_initial(),
        // The one register whose value is not this hypervisor's at all: the
        // guest's timer is the real timer, so what it has left is what the real
        // one has left.
        Register::TIMER_CURRENT_COUNT => timer::remaining(vlapic),
        Register::TIMER_DIVIDE => vlapic.timer_divide(),
        // Three registers that answer nothing. The first two are write-only and
        // the faces above refuse a read of either before reaching here, so
        // answering zero rather than asserting keeps a mis-decode from becoming
        // a fault on the interrupt path. The third was never implemented on any
        // processor this runs on, and reading it has no side effect worth
        // inventing.
        Register::END_OF_INTERRUPT | Register::SELF_IPI | Register::REMOTE_READ => 0,
        other => read_indexed(vlapic, other),
    }
}

/// What a write to a register does.
///
/// The answer says whether anything downstream has to happen that could fail —
/// sending an interrupt, reprogramming real hardware — so that the faces above
/// can report it without knowing what any register means.
pub(crate) fn write(vlapic: &Vlapic, register: Register, value: u32) -> Written {
    match register {
        Register::TASK_PRIORITY => {
            vlapic.set_task_priority(value);
            Written::Nothing
        }
        Register::LOGICAL_DESTINATION => {
            vlapic.set_logical_destination(value);
            Written::LogicalDestination
        }
        Register::DESTINATION_FORMAT => {
            vlapic.set_destination_format(value);
            Written::LogicalDestination
        }
        Register::SPURIOUS => {
            // Software-disabling a controller is a transition rather than a
            // flag: it masks every stored entry, and those entries are what
            // real hardware is programmed from, so the sources and the timer
            // both have to be brought back into agreement with them.
            //
            // Every other write to this register changes nothing hardware is
            // programmed from. The spurious vector itself never reaches the real
            // register — that one holds the bit which software-enables the
            // machine's own controller and is the host's — and re-enabling
            // leaves every entry masked, so there is nothing to reprogram until
            // the guest unmasks one.
            if vlapic.set_spurious(value) {
                Written::Disabled
            } else {
                Written::Nothing
            }
        }
        Register::END_OF_INTERRUPT => Written::EndOfInterrupt,
        Register::ERROR_STATUS => {
            vlapic.errors().written();
            Written::Nothing
        }
        // Writing the low half is what sends the command. Writing the high half
        // only says where the next one will go.
        Register::COMMAND_LOW => Written::Command(vlapic.set_command_low(value)),
        Register::COMMAND_HIGH => {
            vlapic.set_command_high(value);
            Written::Nothing
        }
        // Sending oneself an interrupt, which is the same thing as a command
        // with the self shorthand and a fixed delivery. Only the wide face has
        // this register at all — there is no offset it sits at — and it is
        // answered here rather than there because what a register means is one
        // statement whichever face reaches it.
        Register::SELF_IPI => Written::SelfIpi(Vector::new(vector_of(value))),
        // The write the architecture defines as starting a counting timer, and
        // the only one that does. Ignored entirely in deadline mode, where the
        // count registers stop meaning anything.
        Register::TIMER_INITIAL_COUNT => {
            if vlapic.set_timer_initial(value) {
                Written::TimerStarted
            } else {
                Written::Nothing
            }
        }
        // How fast the timer counts, which is configuration and not a start. A
        // guest that rewrites the divide — including writing back the value
        // already there, which operating systems do — has not asked for the
        // count to be reloaded, and reloading it would let a guest postpone its
        // own expiry indefinitely by touching an unrelated register.
        Register::TIMER_DIVIDE => {
            vlapic.set_timer_divide(value);
            // Inert in deadline mode, where the timer counts nothing at all and
            // the divide takes no part in when it fires. Reporting a timer
            // action for a write the architecture makes inert is what would send
            // the whole configuration path at the real timer's entry and divide
            // while a deadline is live.
            if vlapic.timer_mode() == Some(TimerMode::Deadline) {
                Written::Nothing
            } else {
                Written::Timer
            }
        }
        other => write_indexed(vlapic, other, value),
    }
}

/// The registers that are one of several of a kind: a slot of a bank, or a
/// local-vector-table entry.
fn read_indexed(vlapic: &Vlapic, register: Register) -> u32 {
    if let Some((bank, slot)) = register.bank() {
        return match bank {
            Bank::InService => vlapic.in_service_slot(slot),
            Bank::TriggerMode => vlapic.trigger_mode_slot(slot),
            Bank::InterruptRequest => vlapic.request_slot(slot),
        }
        // A slot the register file does not have cannot arrive here:
        // [`Register::bank`] answers only for the eight registers a bank spans,
        // and a bank has exactly as many slots as those registers because both
        // counts are the same constant. Answering zero is what a register that
        // held nothing would answer, which is the closest thing to nothing this
        // face can say.
        .unwrap_or(0);
    }
    Entry::of(register)
        .filter(|entry| vlapic.model().has(*entry))
        .map_or(0, |entry| vlapic.lvt_readback(entry).into_bits())
}

/// As [`read_indexed`]. Only the local-vector-table entries are writable; the
/// three banks are read-only and the faces above refuse a write to them.
fn write_indexed(vlapic: &Vlapic, register: Register, value: u32) -> Written {
    let Some(entry) = Entry::of(register).filter(|entry| vlapic.model().has(*entry)) else {
        return Written::Nothing;
    };
    let written = vlapic.write_lvt(entry, value);
    // An illegal vector is an error the architecture reports whether or not the
    // entry is masked — but only for an entry that would actually deliver one.
    // Every other delivery mode is an event the processor takes by its own
    // entry point and reads no vector for, so the field holds a number nothing
    // will ever look at, and reporting an error about it would be reporting one
    // the guest cannot act on and hardware would not have raised.
    //
    // Judged on what the write left in the entry rather than on a fresh read of
    // it, because the two are the same thing and only one of them is certain to
    // be: the value is already in hand.
    if vlapic.delivers_a_vector(entry, written) && !priority::legal(written.vector()) {
        vlapic.errors().record(Errors::RECEIVE_ILLEGAL_VECTOR);
    }
    match entry {
        // The timer's entry carries the mode it counts in, so writing it can
        // change what the real timer is doing — though not, on its own, start
        // it.
        Entry::Timer => Written::Timer,
        // The one entry with no source behind it. A controller's report of its
        // own errors is the host's, because the errors are the real
        // controller's; the guest's are delivered from its own error status
        // register instead, so nothing it writes here reaches hardware and there
        // is nothing to bring into agreement.
        Entry::Error => Written::Nothing,
        _ => Written::LocalVectorTable,
    }
}

/// The vector an eight-bit field of a wider value names.
#[expect(
    clippy::cast_possible_truncation,
    reason = "a vector is the low eight bits of the register it is written in"
)]
const fn vector_of(value: u32) -> u8 {
    value as u8
}

/// What a write asked for that the register file alone cannot finish.
///
/// A guest's write to a controller is very often not merely a store. It can
/// send an interrupt to another processor, retire one on this one, or change
/// what real hardware is doing — and all three can fail in ways worth
/// reporting, on paths the register file has no business reaching from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Written {
    /// The value was stored and that is all.
    Nothing,
    /// Send this.
    Command(Command),
    /// Send this processor an interrupt to itself, which is the same thing as a
    /// command with the self shorthand and a fixed delivery.
    SelfIpi(Vector),
    /// Retire the interrupt the guest is servicing, which may mean
    /// acknowledging real hardware.
    EndOfInterrupt,
    /// The local vector table has changed and real hardware has to be brought
    /// into agreement with it.
    LocalVectorTable,
    /// The guest software-disabled its controller, which masked every entry it
    /// has and must now stop every source and the timer.
    Disabled,
    /// The timer's configuration has changed — what it delivers, in which mode,
    /// at what rate. Not a request to start it.
    Timer,
    /// The guest wrote the timer's initial count, which is the write that
    /// starts a counting timer.
    TimerStarted,
    /// The guest armed its timer at a deadline.
    TimerDeadline(u64),
    /// Which logical destinations this processor answers to has changed, and
    /// the real controller has to be told, because hardware matches passed
    /// through interrupts against the real register rather than against this
    /// one.
    LogicalDestination,
    /// The guest changed which face it reaches its controller through. The
    /// virtual half of the transition is done; what it says is how much of the
    /// physical half succeeded, and that real hardware now has to be programmed
    /// from whatever the controller was left holding.
    ModeChanged(Transition),
}

/// Records that the guest named a register the older face reserves.
///
/// Only that face: reaching a reserved index through the model-specific
/// registers is a general protection fault instead, and the architecture is
/// explicit that this bit is not set for one.
///
/// `offset` is the raw offset the guest named rather than a register, because
/// the whole reason this is reached is that no register sits there.
pub(crate) fn illegal_register(vlapic: &Vlapic, offset: u64) {
    if vlapic.errors().record(Errors::ILLEGAL_REGISTER_ADDRESS) {
        // Only the first, because a guest probing its register page produces
        // one of these per probe and the error status register latches them all
        // into the same bit anyway.
        warn!(
            "vlapic: {} named a reserved register at offset {offset:#x}",
            vlapic.index()
        );
    }
}

/// Performs what a guest's write asked for beyond the value being stored.
///
/// Shared by both faces deliberately: a guest that sent an interrupt through
/// the memory-mapped command register and one that sent it through a
/// model-specific register have asked for exactly the same thing, and this is
/// where that stops being two code paths.
pub(crate) fn acted(vlapic: &Vlapic, written: Written) {
    match written {
        Written::Nothing => {}
        // The architecture defines acknowledging nothing as doing nothing, and
        // discharging whatever real hardware was owed for it is part of the
        // acknowledgement rather than something done after it.
        Written::EndOfInterrupt => {
            // The controller the withheld acknowledgements are paid through is
            // this processor's, and it is resolved once: the guest that is
            // acknowledging runs here, so the debts it discharges are this
            // processor's hardware's.
            let local = apic::local().ok();
            let retired = vlapic.end_of_interrupt(&local);
            trace!(
                "vlapic: {} acknowledged {retired:?}, leaving {} in service and {} requested at \
                 task priority {}, real in service {:?}, hardware {}",
                vlapic.index(),
                vlapic.in_service_count(),
                vlapic.requested_count(),
                vlapic.task_priority(),
                local.and_then(LocalApic::in_service_top),
                vlapic.ledger().debts()
            );
        }
        Written::Timer => {
            timer::reprogram(vlapic);
        }
        // The one write that starts a counting timer. What it delivers and in
        // which mode was settled when those registers were written, so this
        // does not reconfigure anything — reconfiguring here is what would move
        // the phase of a periodic tick on every unrelated write.
        Written::TimerStarted => timer::reload(vlapic),
        Written::TimerDeadline(deadline) => timer::arm_deadline(vlapic, deadline),
        Written::LocalVectorTable => {
            sources::reprogram(vlapic);
        }
        // Software-disabling masked every stored entry, and the timer is
        // programmed from its own entry rather than with the rest, so both have
        // to follow. Neither is disarmed: masking suppresses delivery and does
        // not stop a count the guest may still be reading.
        //
        // Hardware masks the entries in the same instant it clears the bit.
        // Here they are separate accesses with host interrupts deliverable
        // between them, so a physical source can still fire after the guest's
        // controller has stopped accepting. Nothing is delivered to the guest
        // either way — its controller refuses it, which is what hardware's
        // masked entry would have achieved — but an interrupt already on its way
        // when the bit cleared is refused where hardware would have latched it
        // in the request register and held it.
        //
        // And it is a lifecycle boundary, which is the third of them: a
        // controller that has stopped delivering is one whose guest cannot
        // acknowledge what real hardware is holding for it.
        Written::Disabled => {
            sources::reprogram(vlapic);
            timer::reprogram(vlapic);
            settle::disabled(vlapic);
        }
        Written::LogicalDestination => mirror_logical_destination(vlapic),
        Written::ModeChanged(transition) => entered(vlapic, transition),
        Written::Command(command) => match lapics() {
            Ok(page) => delivery::send(vlapic, page.all(), command),
            Err(error) => warn!("vlapic: a command could not be delivered: {error}"),
        },
        // A guest sending itself an interrupt is the sender, so a vector no
        // controller may deliver is its error to be told about rather than the
        // receiver's — even though the two are the same controller here.
        Written::SelfIpi(vector) => self_ipi(vlapic, vector),
    }
}

/// Gives the guest the interrupt it sent itself.
///
/// Nothing is owed for one of these whatever becomes of it: it came from the
/// guest rather than from a real source, so there is no real in-service bit
/// behind it and nothing to acknowledge. What the acceptance answers is
/// therefore only worth saying — and it is worth saying, because a controller
/// that refused its own guest's interrupt to itself has dropped something the
/// guest has no other way to notice.
fn self_ipi(vlapic: &Vlapic, vector: Vector) {
    if !priority::legal(vector) {
        vlapic.errors().record(Errors::SEND_ILLEGAL_VECTOR);
        return;
    }
    match vlapic.accept(vector, Trigger::Edge) {
        Accepted::Requested | Accepted::Coalesced => {}
        declined => trace!(
            "vlapic: {} sent itself {vector}, which its own controller did not take: {declined:?}",
            vlapic.index()
        ),
    }
}
