//! The hypercall a guest makes into this hypervisor, and how it is answered.
//!
//! One instruction, `VMMCALL`, already intercepted for the portal's sake — and
//! the portal is not its only user. A guest that wants to know what this
//! hypervisor is doing with its interrupt controllers has no other way to ask:
//! the controllers are emulated, so nothing inside the guest can read the state
//! the host is keeping, and `CPUID` deliberately says there is no hypervisor to
//! ask in the first place.
//!
//! # Which use of the instruction this is
//!
//! Decided by the command register alone, and by a selector in it that no small
//! number could be arrived at by accident. A word without it is not this
//! interface's, and this module says nothing about it — the portal's own
//! notifications travel through the same instruction, and so may anything else
//! in the guest.
//!
//! # Any privilege level
//!
//! The architecture puts no privilege restriction on the instruction, so the
//! call arrives here from ring three exactly as it does from ring zero, and
//! nothing below asks which. That is the point: a debugging tool inside the
//! guest needs no driver, and a hypervisor that refused ring three would be
//! inventing a restriction the architecture does not have.
//!
//! What that costs is that any process in the guest can read the guest's own
//! controllers. Nothing here reads or writes anything the guest does not
//! already own — every register in a dump is a register of the guest's own
//! controller, or of the hardware serving it, and the buffer is memory the
//! guest named and is checked to be the guest's to write.
//!
//! # Nothing here may stop the guest
//!
//! Every failure is a [`Status`] in the guest's own answer register: a buffer
//! that cannot be reached, a part of the machine the host could not read, a
//! version of the interface this build does not implement. The guest is stepped
//! over the instruction either way and carries on.

use core::mem::offset_of;

use apic::{LocalApic, Source};
use descriptors::Vector;
use hypercall::{
    ABSENT, ApicDump, Avic, AvicFlags, BANK_SLOTS, Command, DebtKind, Decoded, Emulated,
    EmulatedFlags, Header, LOGICAL_ENTRIES, LVT_ENTRIES, Mode, PAGE_BYTES, PHYSICAL_ENTRIES,
    Present, Real, RealFlags, Request, SOURCES, SourceState, Startup, Status, TimerMode, Wire,
};
use log::trace;
use memory::{Linear, Written};
use partition::{Addressing, Partition};
use vcpu::{Flow, Vcpu};
use vlapic::{Acceleration, Face, Ledger, Snapshot, StartupPhase, State, Timing};

use crate::{advance, firmware::VMMCALL_BYTES};

/// Answers an intercepted `VMMCALL` that is one of this hypervisor's own
/// hypercalls, or says that it is not one.
///
/// `None` is the whole of what this module does about an instruction that
/// carries no selector of this interface, and it is not a refusal: it says the
/// exit belongs to somebody else, and the caller goes on to whatever else in
/// the guest issues the instruction.
pub(crate) fn exit(vcpu: &mut Vcpu, partition: &Partition) -> Option<Flow> {
    let (command, first, second) = (vcpu.save().rax, vcpu.registers().rdi, vcpu.registers().rsi);
    let status = match hypercall::decode(command, first, second) {
        Decoded::Foreign => return None,
        Decoded::Refused(status) => status,
        Decoded::Request(request) => serve(vcpu, partition, request),
    };
    // Traced rather than reported, at every level of severity, because how often
    // this happens is the guest's choice: the instruction has no privilege
    // restriction, so any process in the guest can reach here as fast as it can
    // take an exit, and a line each would be a denial of service through a
    // machine-wide serial lock.
    trace!("exits: a guest hypercall {command:#018x} was answered with {status:?}");
    vcpu.save_mut().rax = status.word();
    advance(vcpu, VMMCALL_BYTES);
    Some(Flow::Resume)
}

/// Serves one validated request.
fn serve(vcpu: &Vcpu, partition: &Partition, request: Request) -> Status {
    match request.command {
        Command::APIC_DUMP => apic_dump(vcpu, partition, request),
        // Every command this interface defines is answered above. One that
        // reached here would be a command the shared decoder accepted and this
        // did not, which is a disagreement inside this hypervisor rather than
        // anything a guest did.
        _ => Status::UnknownCommand,
    }
}

/// Writes everything the calling processor's interrupt controllers hold into
/// the buffer the guest named.
///
/// The state is gathered before the guest's memory is reached at all, so that
/// what a dump holds is as near one instant as several dozen independent reads
/// can be, and so that nothing is being read while the nested tables are
/// locked.
fn apic_dump(vcpu: &Vcpu, partition: &Partition, request: Request) -> Status {
    let snapshot = vlapic::snapshot().ok();
    let real = real();
    partition.with_memory(Addressing::from_save(vcpu.save()), |guest| {
        let buffer = Buffer {
            guest,
            at: request.buffer,
        };
        match fill(&buffer, vcpu, snapshot.as_ref(), real.as_ref()) {
            Ok(()) => Status::Ok,
            Err(status) => status,
        }
    })
}

/// Fills the buffer in the order that makes a partly written one unreadable.
///
/// The whole of it is checked first, every section that could be read is
/// written next, and the header — which says which of them landed — is written
/// last. So a reader that finds the magic knows the sections the header claims
/// are really there, and one that finds a refusal knows nothing was written at
/// all.
///
/// # Errors
///
/// The [`Status`] the guest is owed: [`Status::Unwritable`] or
/// [`Status::Unreachable`] for a buffer that is not the guest's to write, and
/// [`Status::Faulted`] where it stopped being so while it was being written.
fn fill(
    buffer: &Buffer<'_>,
    vcpu: &Vcpu,
    snapshot: Option<&Snapshot>,
    real: Option<&Real>,
) -> Result<(), Status> {
    buffer.reserved()?;
    let mut present = Present::empty();
    if let Some(snapshot) = snapshot {
        buffer.section(offset_of!(ApicDump, emulated), emulated(snapshot).wire())?;
        present |= Present::EMULATED;
    }
    if let Some(real) = real {
        buffer.section(offset_of!(ApicDump, real), real.wire())?;
        present |= Present::REAL;
    }
    if backing(buffer)? {
        present |= Present::BACKING;
    }
    let physical = physical(buffer)?;
    if physical.is_some() {
        present |= Present::PHYSICAL;
    }
    let logical = logical(buffer)?;
    if logical.is_some() {
        present |= Present::LOGICAL;
    }
    // After the two tables, because the section carries how many entries of each
    // of them really landed.
    let acceleration = snapshot.and_then(|snapshot| snapshot.acceleration);
    let avic = avic(vcpu, acceleration, physical, logical);
    buffer.section(offset_of!(ApicDump, avic), avic.wire())?;
    present |= Present::AVIC;
    let header = header(vcpu, snapshot, present);
    buffer.section(offset_of!(ApicDump, header), header.wire())
}

/// Where a dump is being written, and how a section gets there.
struct Buffer<'a> {
    /// The calling guest's memory, at the addresses that guest itself uses.
    guest: Linear<'a>,
    /// Where in it the dump goes.
    at: u64,
}

impl Buffer<'_> {
    /// Whether the whole of the dump is the guest's own memory to write.
    ///
    /// Asked once, before a byte is written, and it is what stands between a
    /// guest and the rest of the machine: the address is a number out of a
    /// guest register, so nothing but this says it names memory the guest
    /// owns. The walk answers both halves of the question — the guest's own
    /// tables have to translate every page of the range, and the nested
    /// tables have to let the guest write what they translate to, which the
    /// pages standing in for hypervisor memory do not.
    ///
    /// # Errors
    ///
    /// [`Status::Unwritable`] where the range is not the guest's to write, or
    /// [`Status::Unreachable`] where its own tables do not describe it.
    fn reserved(&self) -> Result<(), Status> {
        match self.guest.writable(self.at, size_of::<ApicDump>()) {
            Ok(true) => Ok(()),
            Ok(false) => Err(Status::Unwritable),
            Err(_) => Err(Status::Unreachable),
        }
    }

    /// Writes one section at its own offset in the dump.
    ///
    /// # Errors
    ///
    /// [`Status::Faulted`] if the range stopped being the guest's to write
    /// since [`Buffer::reserved`] said it was, which only another processor
    /// of the same guest can have arranged. The dump is abandoned rather
    /// than finished around the hole, and what is never written is the
    /// header.
    fn section(&self, offset: usize, bytes: &[u8]) -> Result<(), Status> {
        match self.guest.write(self.at.wrapping_add(offset as u64), bytes) {
            Ok(Written::Committed) => Ok(()),
            Ok(Written::Discarded) | Err(_) => Err(Status::Faulted),
        }
    }
}

/// What a buffer says it is, and which sections of it were filled in.
fn header(vcpu: &Vcpu, snapshot: Option<&Snapshot>, present: Present) -> Header {
    Header::new(
        present,
        snapshot.map_or(0, |snapshot| count(snapshot.index)),
        snapshot.map_or(0, |snapshot| snapshot.apic_id),
        u32::from(vcpu.save().cpl),
        snapshot.map_or(0, |snapshot| count(snapshot.processors)),
    )
}

/// The emulated controller's section, out of the snapshot the controller's own
/// crate took.
fn emulated(snapshot: &Snapshot) -> Emulated {
    let mut emulated = Emulated {
        base: snapshot.base,
        command: snapshot.command,
        timer_deadline: snapshot.timer_deadline,
        timer_frequency: snapshot.timer_frequency,
        epoch: snapshot.epoch,
        flags: flags(snapshot.state),
        mode: face(snapshot.face).word(),
        id: snapshot.id,
        xapic_id: snapshot.xapic_id,
        version: snapshot.version,
        task_priority: u32::from(snapshot.task_priority.get()),
        processor_priority: u32::from(snapshot.processor_priority.get()),
        arbitration_priority: u32::from(snapshot.arbitration_priority.get()),
        logical_destination: snapshot.logical_destination,
        destination_format: snapshot.destination_format,
        spurious: snapshot.spurious,
        error_status: snapshot.error_status,
        timer_divide: snapshot.timer_divide,
        timer_initial: snapshot.timer_initial,
        timer_remaining: snapshot.timer_remaining,
        timer_mode: timing(snapshot.timing).word(),
        requested_count: snapshot.requested_count,
        in_service_count: snapshot.in_service_count,
        requested: vector(snapshot.requested),
        deliverable: vector(snapshot.nomination.deliverable),
        blocked: vector(snapshot.nomination.blocked),
        startup: startup(snapshot.startup).word(),
        startup_page: snapshot
            .startup_page
            .map_or(ABSENT, |page| u32::from(page.number())),
        lvt: snapshot.lvt,
        request: snapshot.request,
        in_service: snapshot.in_service,
        trigger_mode: snapshot.trigger_mode,
        external: snapshot.external,
        arrivals: snapshot.counted.arrivals,
        level: snapshot.counted.level,
        declined: snapshot.counted.declined,
        dropped: snapshot.counted.dropped,
        clamped: snapshot.counted.clamped,
        kicks: snapshot.counted.kicks,
        nudges: snapshot.counted.nudges,
        // The debts are the one part of the section that is not the same shape on
        // every machine, and they are filled in below.
        ..Emulated::default()
    };
    // Only one of the two arms of the ledger has anything to say, and which of
    // them a controller has was decided by what the machine's own controller can
    // do — so the fields of the other stay as the default leaves them, which the
    // interface documents as belonging to the arm that is not this one.
    match snapshot.ledger {
        Ledger::Deferred {
            owed,
            released,
            abandoned,
            strandings,
            phantoms,
        } => {
            emulated.debt_kind = DebtKind::Deferred.word();
            emulated.debts_owed = owed;
            emulated.debts_released = released;
            emulated.debts_abandoned = abandoned;
            emulated.debts_strandings = strandings;
            emulated.debts_phantoms = phantoms;
        }
        Ledger::Immediate {
            owed,
            blocked,
            blockings,
        } => {
            emulated.debt_kind = DebtKind::Immediate.word();
            emulated.debts_owed = owed;
            emulated.debts_blocked = blocked;
            emulated.debts_blockings = blockings;
        }
    }
    emulated
}

/// The machine's own controller, as far as this hypervisor's driver can read
/// one.
///
/// `None` where it cannot be reached at all, which is a processor whose
/// controller has not been switched on — nothing a running guest can be on, and
/// reported as an absent section rather than as a controller holding nothing.
fn real() -> Option<Real> {
    let local = apic::local().ok()?;
    let timer = local.timer();
    let extended = local.extended();
    let deadline = timer.deadline().ok();
    let mut flags = RealFlags::empty();
    flags.set(RealFlags::DEADLINE, deadline.is_some());
    flags.set(RealFlags::EXTENDED, extended.usable());
    Some(Real {
        flags,
        timer_deadline: deadline.unwrap_or(0),
        mode: match local.mode() {
            apic::Mode::XApic => Mode::XApic,
            apic::Mode::X2Apic => Mode::X2Apic,
        }
        .word(),
        id: local.id().get(),
        version: local.version(),
        entries: local.entries(),
        logical_destination: local.logical_destination(),
        extended: extended.bits(),
        timer_mode: match timer.mode() {
            Some(apic::TimerMode::OneShot) => TimerMode::OneShot,
            Some(apic::TimerMode::Periodic) => TimerMode::Periodic,
            Some(apic::TimerMode::Deadline) => TimerMode::Deadline,
            None => TimerMode::Reserved,
        }
        .word(),
        timer_initial: timer.initial(),
        timer_remaining: timer.remaining(),
        in_service_top: vector(local.in_service_top()),
        in_service: scan(|vector| local.in_service(vector)),
        trigger_mode: scan(|vector| local.arrived_level(vector)),
        sources: core::array::from_fn(|index| source(local, Source::ALL[index])),
    })
}

/// One bank of the real controller, a vector at a time.
///
/// The driver answers one bit per call and offers no way to read a whole slot,
/// so a bank costs two hundred and fifty-six register reads. That is what a
/// diagnostic pays for state nothing on the delivery path needs, and it is paid
/// only when a guest asks for a dump.
fn scan(held: impl Fn(Vector) -> bool) -> [u32; BANK_SLOTS] {
    let mut bank = [0; BANK_SLOTS];
    for number in 0..=u8::MAX {
        if held(Vector::new(number)) {
            let slot = usize::from(number) / u32::BITS as usize;
            bank[slot] |= 1 << (u32::from(number) % u32::BITS);
        }
    }
    bank
}

/// What the driver can say about one of the real controller's own sources.
fn source(local: LocalApic, source: Source) -> SourceState {
    let Ok(entry) = local.source(source) else {
        return SourceState::empty();
    };
    let mut state = SourceState::READ;
    state.set(SourceState::MASKED, entry.is_masked());
    state.set(SourceState::PENDING, entry.pending());
    state.set(SourceState::REMOTE_IRR, entry.remote_irr());
    state
}

/// The state of hardware-driven delivery, from the two authorities that have a
/// say in it.
///
/// The control block is what this processor was entered with, and so what
/// actually happened on the run that has just ended; the policy is what the
/// machine built and would like to be driving. Both are carried because they
/// can disagree, and a disagreement is the thing a reader is looking for.
fn avic(
    vcpu: &Vcpu,
    acceleration: Option<Acceleration>,
    physical: Option<usize>,
    logical: Option<usize>,
) -> Avic {
    let control = vcpu.control();
    let block = control.interrupt_control;
    let deliverable = vlapic::avic_deliverable().ok();
    let mut flags = AvicFlags::empty();
    flags.set(AvicFlags::PROVISIONED, acceleration.is_some());
    flags.set(
        AvicFlags::OWN_PAGE,
        acceleration.is_some_and(|acceleration| acceleration.own_page.is_some()),
    );
    flags.set(AvicFlags::ACCELERATED, vlapic::avic_accelerated(vcpu));
    flags.set(AvicFlags::WIDER_FACE, vlapic::avic_wider_face(vcpu));
    flags.set(AvicFlags::BLOCK_ENABLE, block.avic_enable());
    flags.set(AvicFlags::BLOCK_X2_ENABLE, block.x2avic_enable());
    flags.set(AvicFlags::DELIVERABLE_KNOWN, deliverable.is_some());
    flags.set(AvicFlags::DELIVERABLE, deliverable.unwrap_or(false));
    flags.set(AvicFlags::X2APIC_OFFERED, vlapic::x2apic_offered());
    flags.set(
        AvicFlags::IPI_VIRTUAL,
        acceleration.is_some_and(|acceleration| acceleration.ipi_virtual),
    );
    flags.set(
        AvicFlags::MACHINE_INHIBITED,
        acceleration.is_some_and(|acceleration| acceleration.machine_inhibited),
    );
    // What the table describes against what a dump can hold: a reader that was
    // handed the first five hundred entries of a longer table has to be told that
    // is what it has.
    let described = acceleration.map_or(0, |acceleration| usize::from(acceleration.max_index) + 1);
    flags.set(
        AvicFlags::PHYSICAL_TRUNCATED,
        physical.is_some_and(|entries| entries < described),
    );
    Avic {
        flags,
        apic_bar: control.avic_apic_bar,
        backing_page: control.avic_backing_page,
        own_backing_page: address(acceleration.and_then(|acceleration| acceleration.own_page)),
        logical_table: control.avic_logical_table,
        physical_table: control.avic_physical_table.address().as_u64(),
        policy_physical_table: address(
            acceleration.map(|acceleration| acceleration.physical_table),
        ),
        policy_logical_table: address(acceleration.map(|acceleration| acceleration.logical_table)),
        max_index: u32::from(control.avic_physical_table.max_index()),
        policy_max_index: acceleration.map_or(0, |acceleration| u32::from(acceleration.max_index)),
        physical_entries: count(physical.unwrap_or(0)),
        logical_entries: count(logical.unwrap_or(0)),
    }
}

/// Copies this processor's backing page into the dump, and says whether it
/// landed.
///
/// # Errors
///
/// As [`Buffer::section`]. A page the host could not read is not an error at
/// all: the section is left absent, which is what a machine that provisioned no
/// hardware delivery has.
fn backing(buffer: &Buffer<'_>) -> Result<bool, Status> {
    let mut page = [0; PAGE_BYTES];
    let Ok(bytes) = vlapic::read_backing_page(&mut page) else {
        return Ok(false);
    };
    buffer.section(offset_of!(ApicDump, backing), &page[..bytes])?;
    Ok(true)
}

/// Copies the physical table's entries into the dump, and says how many landed.
///
/// # Errors
///
/// As [`backing`].
fn physical(buffer: &Buffer<'_>) -> Result<Option<usize>, Status> {
    let mut entries = [0; PHYSICAL_ENTRIES * size_of::<u64>()];
    let Ok(bytes) = vlapic::read_physical_table(&mut entries) else {
        return Ok(None);
    };
    buffer.section(offset_of!(ApicDump, physical), &entries[..bytes])?;
    Ok(Some(bytes / size_of::<u64>()))
}

/// Copies the logical table's entries into the dump, and says how many landed.
///
/// # Errors
///
/// As [`backing`].
fn logical(buffer: &Buffer<'_>) -> Result<Option<usize>, Status> {
    let mut entries = [0; LOGICAL_ENTRIES * size_of::<u32>()];
    let Ok(bytes) = vlapic::read_logical_table(&mut entries) else {
        return Ok(None);
    };
    buffer.section(offset_of!(ApicDump, logical), &entries[..bytes])?;
    Ok(Some(bytes / size_of::<u32>()))
}

/// A vector as a dump carries one, which is [`ABSENT`] where there is none.
fn vector(vector: Option<Vector>) -> u32 {
    vector.map_or(ABSENT, |vector| u32::from(vector.number()))
}

/// An address as a dump carries one, which is zero where there is none — no
/// structure of this hypervisor's is ever at physical zero.
fn address(address: Option<x86_64::PhysAddr>) -> u64 {
    address.map_or(0, x86_64::PhysAddr::as_u64)
}

/// A count as a dump carries one.
///
/// Every count here is bounded by an array this hypervisor sized, so the
/// saturation is unreachable rather than a truncation a reader could be handed.
fn count(count: usize) -> u32 {
    u32::try_from(count).unwrap_or(u32::MAX)
}

/// The face the controller is in, as the interface names it.
fn face(face: Face) -> Mode {
    match face {
        Face::Disabled => Mode::Disabled,
        Face::XApic => Mode::XApic,
        Face::X2Apic => Mode::X2Apic,
    }
}

/// How the timer counts, as the interface names it — including the encoding the
/// architecture reserves, which a guest can write and a dump has to be able to
/// report.
fn timing(timing: Option<Timing>) -> TimerMode {
    match timing {
        Some(Timing::OneShot) => TimerMode::OneShot,
        Some(Timing::Periodic) => TimerMode::Periodic,
        Some(Timing::Deadline) => TimerMode::Deadline,
        None => TimerMode::Reserved,
    }
}

/// Where the processor is in the sequence that starts it, as the interface
/// names it.
fn startup(startup: StartupPhase) -> Startup {
    match startup {
        StartupPhase::Running => Startup::Running,
        StartupPhase::InitRequested => Startup::InitRequested,
        StartupPhase::WaitingForSipi => Startup::WaitingForSipi,
    }
}

/// The controller's condition, as the interface's own flag word.
///
/// A table rather than a cast, because the two sets are different vocabularies
/// that happen to agree today: the interface is versioned and the controller's
/// crate is not, so a bit that moves on either side has to be a line here
/// rather than a silent change of meaning.
fn flags(state: State) -> EmulatedFlags {
    let mut flags = EmulatedFlags::empty();
    for (held, flag) in [
        (State::BOOTSTRAP, EmulatedFlags::BOOTSTRAP),
        (State::STARTABLE, EmulatedFlags::STARTABLE),
        (State::SOFTWARE_ENABLED, EmulatedFlags::SOFTWARE_ENABLED),
        (State::ACCEPTING, EmulatedFlags::ACCEPTING),
        (State::RUNNING, EmulatedFlags::RUNNING),
        (State::AWAY, EmulatedFlags::AWAY),
        (State::OWNED, EmulatedFlags::OWNED),
        (State::AVIC_INHIBITED, EmulatedFlags::AVIC_INHIBITED),
        (State::DEADLINE_OFFERED, EmulatedFlags::DEADLINE_OFFERED),
        (State::X2APIC_OFFERED, EmulatedFlags::X2APIC_OFFERED),
    ] {
        flags.set(flag, state.contains(held));
    }
    flags
}

/// The banks and the entry table the interface carries are the ones this
/// hypervisor's own controllers have. They are separate statements — the
/// interface has no dependency on the rest of pulzar and must not grow one — so
/// a disagreement is caught here rather than by a reader finding a bank half
/// reported.
const _: () = assert!(
    BANK_SLOTS == apic::VECTOR_WORDS
        && LVT_ENTRIES == apic::LVT_ENTRIES
        && SOURCES == Source::ALL.len(),
    "the interface and this hypervisor's controllers disagree about a bank, the entry table or the \
     sources",
);
