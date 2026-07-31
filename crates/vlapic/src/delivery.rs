//! Working out which processors a command names, and giving it to them.
//!
//! This is the one place a guest's interrupt command register becomes an
//! interrupt somewhere else, and almost all of it is the destination
//! arithmetic: four ways of naming processors, three of which can name several
//! at once.
//!
//! # Delivering is setting a bit, and then telling somebody
//!
//! Accepting an interrupt into another processor's controller is a bit set in
//! an atomic, and it costs nothing. What costs something is that the target may
//! be inside the guest, where it will not look at its controller until
//! something makes it leave — so a real interrupt is sent to force an exit.
//!
//! That pairing is where a lost wakeup would live, and the order on both sides
//! is what stops one:
//!
//! - Here: set the request bit, *then* read whether the target is in the guest.
//! - There: store that it is in the guest, *then* re-read the request bits.
//!
//! Both stores are sequentially consistent, so at least one side sees the
//! other. If this side misses the flag, the target's re-read finds the bit; if
//! the target's re-read misses the bit, this side sees the flag and sends the
//! interrupt. There is no interleaving in which both miss.
//!
//! And a doorbell that arrives while the target is between `CLGI` and `VMRUN`
//! is not lost either: the interrupt is held by the cleared global interrupt
//! flag, and the maskable-interrupt intercept turns it into an immediate exit
//! on entry.
//!
//! # Startup messages, before and after the processors are ours
//!
//! A guest may reset and start processors before this hypervisor has taken them
//! over. In that window the target really is executing firmware's own code on
//! real hardware, so INIT and start-up are forwarded to the real controller and
//! the processor starts exactly as it would have bare metal.
//!
//! Once a processor is running the hypervisor's own code, forwarding would
//! reset the *host*. From then on both are emulated: the target's controller is
//! reset in software and its virtual processor is parked until a start-up
//! vector arrives.

use apic::{Command as HardwareCommand, Delivery as HardwareDelivery, Target};
use descriptors::Vector;
use log::{trace, warn};

use crate::{
    icr::{Command, Delivery, DestinationMode, Shorthand, Trigger},
    state::{Startup, Vlapic},
};

/// Delivers a command the guest wrote to its interrupt command register.
///
/// Nothing here fails in a way the guest can see. A command naming a mode the
/// architecture reserves, or a processor that does not exist, is one real
/// hardware would also do nothing useful with — so it is recorded and dropped,
/// which is what a controller does with a message nobody accepts.
pub(crate) fn send(from: &Vlapic, lapics: &[Vlapic], command: Command) {
    let mode = from.mode();
    let Some(delivery) = command.delivery(mode) else {
        warn!(
            "vlapic: {} sent a command with a reserved delivery mode: {:#x}",
            from.index(),
            command.bits()
        );
        return;
    };
    // A synchronisation message that reloads arbitration identifiers and does
    // nothing else. No processor this hypervisor runs on arbitrates over a bus,
    // so there is nothing for it to reload.
    if command.is_init_deassert() {
        trace!(
            "vlapic: {} sent an init de-assert, which does nothing",
            from.index()
        );
        return;
    }

    let targets = resolve(from, lapics, command);
    match delivery {
        // The chipset picks, and it picks by priority. Choosing the least busy
        // of the named processors is what the architecture describes, and doing
        // it here rather than declining is what keeps a guest that uses it from
        // silently losing interrupts.
        Delivery::LowestPriority => {
            if let Some(target) = least_busy(&targets) {
                accept(from, target, command.vector(), command.trigger());
            }
        }
        Delivery::Fixed => {
            for target in targets.iter() {
                accept(from, target, command.vector(), command.trigger());
            }
        }
        Delivery::NonMaskable => {
            for target in targets.iter() {
                target.raise_nmi();
                nudge(from, target);
            }
        }
        Delivery::Init => {
            for target in targets.iter() {
                initialize(from, target);
            }
        }
        Delivery::Startup => {
            for target in targets.iter() {
                start(from, target, command.vector().number());
            }
        }
        // Taking a processor into system-management mode is not something this
        // hypervisor models, and a guest that asks for it is asking for
        // firmware behaviour it cannot see the results of anyway.
        Delivery::SystemManagement => warn!(
            "vlapic: {} sent a system-management interrupt, which is not delivered",
            from.index()
        ),
    }
}

/// Gives one processor an interrupt, and makes sure it notices.
fn accept(from: &Vlapic, target: &Vlapic, vector: Vector, trigger: Trigger) {
    target.accept(vector, trigger);
    nudge(from, target);
}

/// Forces a target out of the guest, if it is in one, so that it looks at its
/// controller.
///
/// The sender never needs one of these for itself: it is already outside the
/// guest — it is executing this — and it consults its own controller before it
/// goes back in.
fn nudge(from: &Vlapic, target: &Vlapic) {
    if target.index() == from.index() || !target.in_guest() {
        return;
    }
    if let Err(error) = crate::doorbell(target.index()) {
        warn!(
            "vlapic: {} could not interrupt {}: {error}",
            from.index(),
            target.index()
        );
    }
}

/// Resets a processor and leaves it waiting to be started.
fn initialize(from: &Vlapic, target: &Vlapic) {
    // A processor cannot reset itself this way. Real hardware permits the
    // command and the result is a processor that resets out from under the code
    // that sent it, which no sensible guest asks for and which here would take
    // down the virtual processor mid-exit.
    if target.index() == from.index() {
        warn!(
            "vlapic: {} sent itself an init, which is refused",
            from.index()
        );
        return;
    }
    if !crate::owns(target) {
        forward(from, target, HardwareDelivery::Init);
        return;
    }
    target.request_init();
    nudge(from, target);
}

/// Releases a processor from waiting, at the address the vector gives the page
/// number of.
fn start(from: &Vlapic, target: &Vlapic, vector: u8) {
    if target.index() == from.index() {
        return;
    }
    if !crate::owns(target) {
        forward(from, target, HardwareDelivery::Startup(vector));
        return;
    }
    // Refused unless the target is waiting for one, which is what makes the
    // second of the pair a guest sends harmless: the first starts the
    // processor, and the second finds it already running.
    if target.request_sipi(vector) {
        nudge(from, target);
    }
}

/// Sends a startup message to real hardware, for a processor this hypervisor
/// has not taken over yet.
fn forward(from: &Vlapic, target: &Vlapic, delivery: HardwareDelivery) {
    let outcome = apic::local().and_then(|local| {
        local.send(HardwareCommand::new(
            delivery,
            Target::One(target.apic_id()),
        ))
    });
    match outcome {
        Ok(()) => trace!(
            "vlapic: {} forwarded {delivery:?} to {} on real hardware",
            from.index(),
            target.apic_id()
        ),
        Err(error) => warn!(
            "vlapic: {} could not forward {delivery:?} to {}: {error}",
            from.index(),
            target.apic_id()
        ),
    }
}

/// Which of the named processors is running at the lowest priority.
fn least_busy<'a>(targets: &Targets<'a>) -> Option<&'a Vlapic> {
    targets
        .iter()
        .min_by_key(|target| target.processor_priority().get())
}

/// Which processors a command names.
///
/// A shorthand answers without looking at the destination field at all, which
/// is the architecture's rule and not a shortcut: the destination *mode* is
/// ignored too whenever one is used.
fn resolve<'a>(from: &Vlapic, lapics: &'a [Vlapic], command: Command) -> Targets<'a> {
    let mut targets = Targets::new();
    match command.shorthand() {
        Shorthand::Myself => {
            if let Some(here) = lapics.get(from.index().get()) {
                targets.push(here);
            }
        }
        Shorthand::All => targets.extend(lapics.iter()),
        Shorthand::Others => {
            targets.extend(
                lapics
                    .iter()
                    .filter(|target| target.index() != from.index()),
            );
        }
        Shorthand::None => {
            let destination = command.destination(from.mode());
            targets.extend(
                lapics
                    .iter()
                    .filter(|target| addresses(target, destination, command.destination_mode())),
            );
        }
    }
    targets
}

/// Whether a destination names this processor.
fn addresses(target: &Vlapic, destination: u32, mode: DestinationMode) -> bool {
    match mode {
        DestinationMode::Physical => physical(target, destination),
        DestinationMode::Logical => logical(target, destination),
    }
}

/// Whether a physical destination names this processor.
///
/// The all-ones value is a broadcast in both faces. In the older one the field
/// is only eight bits wide, so the value that broadcasts is the eight-bit one.
fn physical(target: &Vlapic, destination: u32) -> bool {
    if destination == broadcast(target) {
        return true;
    }
    target.apic_id().get() == destination
}

/// Whether a logical destination names this processor.
///
/// Three models, and which one applies is not a property of the command but of
/// the controller being matched against: the face it is in, and in the older
/// face the destination format register it was programmed with.
fn logical(target: &Vlapic, destination: u32) -> bool {
    if destination == broadcast(target) {
        return true;
    }
    let logical_id = target.logical_destination();
    match target.mode() {
        // Cluster in the high half, a bit per processor in the low half. The
        // only model x2APIC has.
        crate::base::Mode::X2Apic => {
            destination >> CLUSTER_SHIFT == logical_id >> CLUSTER_SHIFT
                && destination & logical_id & CLUSTER_MEMBERS != 0
        }
        _ if flat(target) => {
            // Eight processors, a bit each, in the top byte of the register.
            (destination & (logical_id >> XAPIC_LOGICAL_SHIFT)) & u32::from(u8::MAX) != 0
        }
        // Four-bit cluster address, then a four-bit mask within it.
        _ => {
            let cluster = (logical_id >> XAPIC_LOGICAL_SHIFT) >> XAPIC_CLUSTER_SHIFT;
            let members = (logical_id >> XAPIC_LOGICAL_SHIFT) & XAPIC_CLUSTER_MASK;
            destination >> XAPIC_CLUSTER_SHIFT == cluster
                && destination & XAPIC_CLUSTER_MASK & members != 0
        }
    }
}

/// Whether this controller matches logical destinations the flat way.
fn flat(target: &Vlapic) -> bool {
    target.destination_format() >> DESTINATION_FORMAT_SHIFT == FLAT_MODEL
}

/// The destination value that names every processor, in the width the face in
/// use gives the field.
fn broadcast(target: &Vlapic) -> u32 {
    match target.mode() {
        crate::base::Mode::X2Apic => Command::BROADCAST,
        _ => u32::from(u8::MAX),
    }
}

/// The processors one command names.
///
/// A fixed array rather than a heap allocation, because this is built on the
/// path a guest sends an interrupt on and that path must not allocate. Sized by
/// the most processors this hypervisor will address, which a command naming
/// more than is simply truncated at — and that is worth knowing about, so it is
/// reported.
struct Targets<'a> {
    targets: [Option<&'a Vlapic>; MAX_TARGETS],
    count: usize,
}

impl<'a> Targets<'a> {
    /// No processors named yet.
    const fn new() -> Self {
        Self {
            targets: [None; MAX_TARGETS],
            count: 0,
        }
    }

    /// Names one more.
    fn push(&mut self, target: &'a Vlapic) {
        if self.count >= MAX_TARGETS {
            warn!(
                "vlapic: a command named more than {MAX_TARGETS} processors; the rest are dropped"
            );
            return;
        }
        self.targets[self.count] = Some(target);
        self.count += 1;
    }

    /// Names all of them.
    fn extend(&mut self, named: impl Iterator<Item = &'a Vlapic>) {
        for target in named {
            self.push(target);
        }
    }

    /// The processors named, in the order they were named.
    fn iter(&self) -> impl Iterator<Item = &'a Vlapic> {
        self.targets[..self.count].iter().copied().flatten()
    }
}

/// How many processors one command may name.
const MAX_TARGETS: usize = 256;

/// Bits an x2APIC logical identifier's cluster is shifted by.
const CLUSTER_SHIFT: u32 = 16;

/// The part of an x2APIC logical identifier that names processors within a
/// cluster.
const CLUSTER_MEMBERS: u32 = 0xFFFF;

/// Bits the older face's logical identifier is shifted by: the top byte.
const XAPIC_LOGICAL_SHIFT: u32 = 24;

/// Bits a cluster address is shifted by within that byte.
const XAPIC_CLUSTER_SHIFT: u32 = 4;

/// The part of that byte naming processors within a cluster.
const XAPIC_CLUSTER_MASK: u32 = 0xF;

/// Bits the destination format register's model selector is shifted by.
const DESTINATION_FORMAT_SHIFT: u32 = 28;

/// The encoding that selects the flat model.
const FLAT_MODEL: u32 = 0xF;

/// Where a processor waiting to be started begins executing: the vector names
/// the page.
pub(crate) const fn startup_address(vector: u8) -> u64 {
    (vector as u64) << STARTUP_VECTOR_SHIFT
}

/// Bits a start-up vector is shifted by to give the address it names.
const STARTUP_VECTOR_SHIFT: u32 = 12;

/// Whether this processor is waiting to be started.
pub(crate) fn waiting(vlapic: &Vlapic) -> bool {
    vlapic.startup() == Startup::WaitingForSipi
}

/// Applies whatever startup message arrived for this processor, and says what
/// it should do now.
///
/// Called by a processor about itself, at an exit boundary, which is what makes
/// the reset safe: nothing else is looking at this controller's registers, and
/// this processor is not part-way through injecting anything.
pub(crate) fn settle(vlapic: &Vlapic) -> Resumption {
    if vlapic.startup() == Startup::InitRequested {
        // Exactly what hardware leaves behind: the identifier and the face
        // survive, everything else is as it was at reset.
        vlapic.reset_registers();
        vlapic.set_startup(Startup::WaitingForSipi);
    }
    if !waiting(vlapic) {
        return Resumption::Carry;
    }
    match vlapic.take_sipi() {
        Some(vector) => {
            vlapic.set_startup(Startup::Running);
            Resumption::StartAt(startup_address(vector))
        }
        None => Resumption::Wait,
    }
}

/// What a processor should do after its startup state has been settled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resumption {
    /// Carry on running the guest.
    Carry,
    /// Reset and held. Do not enter the guest.
    Wait,
    /// Begin executing the guest in real mode at this address.
    StartAt(u64),
}
