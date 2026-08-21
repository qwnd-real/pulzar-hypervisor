//! Reads the interrupt controllers of the processor it runs on out of the
//! pulzar hypervisor underneath it, and prints them.
//!
//! There is no driver and no privileged step. The hypervisor answers the
//! hypercall at any privilege level, so this is an ordinary process: it hands
//! the host a buffer of its own, the host fills it in, and everything below is
//! formatting.
//!
//! # What it needs to be run under
//!
//! A guest of this hypervisor, on the processor whose controllers are wanted —
//! the dump is always of the processor the call is made on, because that is the
//! one whose controller the host can read without disturbing another. Run
//! anywhere else, the instruction the call is made with is not intercepted by
//! anything: the processor raises an invalid-opcode exception, and the
//! operating system ends this process with whatever it delivers for one. That
//! cannot be guarded against from inside — pulzar deliberately answers `CPUID`
//! as though no hypervisor were there, so there is no feature bit to ask first
//! — and it is why this is a tool run on purpose rather than something a
//! program does in passing.
//!
//! # Nothing here is platform-specific
//!
//! One instruction and some printing. The same source runs on any 64-bit x86
//! operating system, and nothing below asks which one it is.

use std::process::ExitCode;

use hypercall::{
    ABSENT, Answer, ApicDump, Avic, AvicFlags, BANK_SLOTS, DebtKind, Emulated, EmulatedFlags,
    Header, LOGICAL_ENTRIES, LVT_ENTRIES, Mismatch, Mode, PAGE_BYTES, Present, Real, RealFlags,
    SOURCES, SourceState, Startup, Status, TimerMode,
};

/// Asks for the dump, and prints it or says why there is none.
fn main() -> ExitCode {
    // Boxed rather than held on the stack: the structure is several pages, and a
    // buffer the host is about to write is exactly the thing not to put where a
    // deep stack might not have room for it.
    let mut dump = Box::new(ApicDump::zeroed());
    match hypercall::apic_dump(&mut dump) {
        Answer::Status(Status::Ok) => {}
        Answer::Status(status) => {
            eprintln!("apic-dump: the hypervisor refused the call: {status}");
            return ExitCode::from(REFUSED);
        }
        Answer::Unknown(word) => {
            eprintln!(
                "apic-dump: something intercepted the call and answered {word:#018x}, which is not \
                 a status this interface defines — this is probably not a pulzar guest"
            );
            return ExitCode::from(FOREIGN);
        }
    }
    if let Err(mismatch) = dump.validate() {
        eprintln!("apic-dump: the answer is not a dump this build can read: {mismatch}");
        return ExitCode::from(match mismatch {
            Mismatch::Magic { .. } => UNWRITTEN,
            Mismatch::Version { .. } | Mismatch::Size { .. } => STALE,
        });
    }
    report(&dump);
    ExitCode::SUCCESS
}

/// The hypervisor would not serve the call, and said why.
const REFUSED: u8 = 1;

/// Something answered that is not this interface.
const FOREIGN: u8 = 2;

/// The call was served and the buffer holds no dump, which nothing should be
/// able to produce.
const UNWRITTEN: u8 = 3;

/// The hypervisor and this tool were built against different versions of the
/// interface.
const STALE: u8 = 4;

/// Prints the whole of a dump, section by section, saying so where the host
/// could not read one.
fn report(dump: &ApicDump) {
    heading(&dump.header);
    section(
        dump.header.present,
        Present::EMULATED,
        "emulated controller",
        || {
            emulated(&dump.emulated);
        },
    );
    section(
        dump.header.present,
        Present::REAL,
        "the machine's own controller",
        || {
            real(&dump.real);
        },
    );
    section(
        dump.header.present,
        Present::AVIC,
        "hardware-driven delivery",
        || {
            avic(&dump.avic);
        },
    );
    section(
        dump.header.present,
        Present::BACKING,
        "backing page",
        || {
            backing(&dump.backing);
        },
    );
    section(
        dump.header.present,
        Present::PHYSICAL,
        "physical table",
        || {
            physical(dump, &dump.physical);
        },
    );
    section(
        dump.header.present,
        Present::LOGICAL,
        "logical table",
        || {
            logical(dump, &dump.logical);
        },
    );
}

/// What the dump is of, and which of its sections hold anything.
fn heading(header: &Header) {
    println!("pulzar apic dump");
    row(
        "processor",
        &format!("{} of {}", header.cpu_index, header.processors),
    );
    row("apic id", &format!("{:#010x}", header.apic_id));
    row("called at cpl", &format!("{}", header.cpl));
    row("interface version", &format!("{}", header.version));
    row("dump size", &format!("{} bytes", header.bytes));
}

/// Prints one section under its own heading, or says the host could not read
/// it.
///
/// A missing section is a line rather than nothing at all: the whole point of
/// the present flags is that "the host could not read this" and "this is all
/// zeroes" are different answers, and a reader must not have to guess which it
/// has.
fn section(present: Present, flag: Present, name: &str, print: impl FnOnce()) {
    println!();
    if !present.contains(flag) {
        println!("{name}: not read");
        return;
    }
    println!("{name}");
    print();
}

/// The emulated controller: the registers the guest programs and the state the
/// host keeps beside them.
fn emulated(section: &Emulated) {
    row("face", &mode(section.mode));
    row("base", &format!("{:#018x}", section.base));
    row("state", &emulated_state(section.flags));
    row("id", &format!("{:#010x}", section.id));
    row(
        "id in the older face",
        &format!("{:#010x}", section.xapic_id),
    );
    row("version", &format!("{:#010x}", section.version));
    row("task priority", &format!("{:#04x}", section.task_priority));
    row(
        "processor priority",
        &format!("{:#04x}", section.processor_priority),
    );
    row(
        "arbitration priority",
        &format!("{:#04x}", section.arbitration_priority),
    );
    row(
        "logical destination",
        &format!("{:#010x}", section.logical_destination),
    );
    row(
        "destination format",
        &format!("{:#010x}", section.destination_format),
    );
    row("spurious", &format!("{:#010x}", section.spurious));
    row("error status", &format!("{:#010x}", section.error_status));
    row("interrupt command", &format!("{:#018x}", section.command));
    row("requested", &vector(section.requested));
    row("deliverable now", &vector(section.deliverable));
    row("held by task priority", &vector(section.blocked));
    row("requested count", &format!("{}", section.requested_count));
    row("in service count", &format!("{}", section.in_service_count));
    row("startup", &startup(section.startup));
    row("startup page", &page(section.startup_page));
    row("reset count", &format!("{}", section.epoch));
    timer(
        section.timer_mode,
        section.timer_divide,
        section.timer_initial,
        section.timer_remaining,
        section.timer_deadline,
        Some(section.timer_frequency),
    );
    for (index, entry) in section.lvt.iter().enumerate() {
        row(&format!("lvt {}", EMULATED_SOURCES[index]), &lvt(*entry));
    }
    counts(section);
    debts(section);
    bank("request", &section.request);
    bank("in service", &section.in_service);
    bank("trigger mode", &section.trigger_mode);
    bank("external", &section.external);
}

/// What the controller has counted since the machine came up.
fn counts(section: &Emulated) {
    row("arrivals", &format!("{}", section.arrivals));
    row("of those level triggered", &format!("{}", section.level));
    row("declined", &format!("{}", section.declined));
    row("lost", &format!("{}", section.dropped));
    row("periods raised", &format!("{}", section.clamped));
    row("kicks sent", &format!("{}", section.kicks));
    row("nudges sent", &format!("{}", section.nudges));
}

/// What real hardware is holding in service for this guest.
///
/// Only the counts belonging to the ledger the controller actually has: the
/// others are not zero, they are absent, and printing them would invent a
/// machine that had none of something it cannot have.
fn debts(section: &Emulated) {
    match DebtKind::from_word(section.debt_kind) {
        Some(kind @ DebtKind::Deferred) => {
            row("ledger", kind.name());
            row("owed", &format!("{}", section.debts_owed));
            row("released", &format!("{}", section.debts_released));
            row("abandoned", &format!("{}", section.debts_abandoned));
            row("abandoned ever", &format!("{}", section.debts_strandings));
            row("phantoms", &format!("{}", section.debts_phantoms));
        }
        Some(kind @ DebtKind::Immediate) => {
            row("ledger", kind.name());
            row("owed", &format!("{}", section.debts_owed));
            row("blocked", &format!("{}", section.debts_blocked));
            row("blocked ever", &format!("{}", section.debts_blockings));
        }
        None => row("ledger", &unknown(section.debt_kind)),
    }
}

/// The machine's own controller, as far as the host can read one.
fn real(section: &Real) {
    row("face", &mode(section.mode));
    row("id", &format!("{:#010x}", section.id));
    row("version", &format!("{:#010x}", section.version));
    row("entries", &format!("{}", section.entries));
    row(
        "logical destination",
        &format!("{:#010x}", section.logical_destination),
    );
    row("extended space", &format!("{:#010x}", section.extended));
    row(
        "extended usable",
        yes(section.flags.contains(RealFlags::EXTENDED)),
    );
    row("in service, highest", &vector(section.in_service_top));
    timer(
        section.timer_mode,
        // The divide is not readable through the host's own driver, and neither
        // is the calibrated rate: both are the emulated controller's above.
        ABSENT,
        section.timer_initial,
        section.timer_remaining,
        section.timer_deadline,
        None,
    );
    row(
        "deadline readable",
        yes(section.flags.contains(RealFlags::DEADLINE)),
    );
    for (index, state) in section.sources.iter().enumerate() {
        row(&format!("source {}", REAL_SOURCES[index]), &source(*state));
    }
    bank("in service", &section.in_service);
    bank("trigger mode", &section.trigger_mode);
}

/// Hardware-driven delivery: what the control block carries, and what the
/// machine built for it.
///
/// The two are printed beside each other because a disagreement between them is
/// the thing worth finding: the block is what the processor was entered with,
/// and the policy is what the acceleration would like to be driving.
fn avic(section: &Avic) {
    row("state", &avic_state(section.flags));
    row("register page", &format!("{:#018x}", section.apic_bar));
    row(
        "block backing page",
        &format!("{:#018x}", section.backing_page),
    );
    row(
        "this processor's page",
        &format!("{:#018x}", section.own_backing_page),
    );
    row(
        "block physical table",
        &format!("{:#018x}", section.physical_table),
    );
    row(
        "policy physical table",
        &format!("{:#018x}", section.policy_physical_table),
    );
    row(
        "block logical table",
        &format!("{:#018x}", section.logical_table),
    );
    row(
        "policy logical table",
        &format!("{:#018x}", section.policy_logical_table),
    );
    row("block largest index", &format!("{}", section.max_index));
    row(
        "policy largest index",
        &format!("{}", section.policy_max_index),
    );
    row(
        "entries carried",
        &format!(
            "{} physical, {} logical",
            section.physical_entries, section.logical_entries
        ),
    );
}

/// The backing page, decoded as the register file it is.
///
/// Every register the architecture puts in the page is named, at its own
/// offset, and the three banks are printed as bit tables — which is the whole
/// reason a reader wants the page: while the hardware drives the controller,
/// these are the registers the guest is really being served out of, and the
/// model beside them can say something different.
fn backing(page: &[u8; PAGE_BYTES]) {
    for (offset, name) in PAGE_REGISTERS {
        row(name, &format!("{:#010x}", word(page, offset)));
    }
    for (offset, name) in PAGE_BANKS {
        let mut slots = [0; BANK_SLOTS];
        for (slot, held) in slots.iter_mut().enumerate() {
            *held = word(page, offset + slot * BANK_STRIDE);
        }
        bank(name, &slots);
    }
}

/// The table of the guest's processors, one line per entry that names one.
///
/// Entries that name no processor are counted rather than printed: a machine
/// with eight processors and a table sized for its identifiers has hundreds of
/// them, and a reader looking for a lost interrupt is looking for the ones that
/// are there.
fn physical(dump: &ApicDump, entries: &[u64]) {
    let carried = usize::try_from(dump.avic.physical_entries).unwrap_or(entries.len());
    let carried = carried.min(entries.len());
    if dump.avic.flags.contains(AvicFlags::PHYSICAL_TRUNCATED) {
        row(
            "truncated",
            &format!(
                "the table describes {} entries and this dump holds {carried}",
                dump.avic.policy_max_index + 1
            ),
        );
    }
    let mut named = 0;
    for (index, entry) in entries[..carried].iter().enumerate() {
        if entry & ENTRY_VALID == 0 {
            continue;
        }
        named += 1;
        row(
            &format!("entry {index}"),
            &format!(
                "{entry:#018x}  page {:#018x}, host apic id {:#x}{}",
                entry & ENTRY_PAGE,
                entry & ENTRY_HOST_ID,
                if entry & ENTRY_RUNNING == 0 {
                    ""
                } else {
                    ", running"
                }
            ),
        );
    }
    row("entries", &format!("{named} of {carried} name a processor"));
}

/// The table resolving a logical destination to one of the guest's processors.
fn logical(dump: &ApicDump, entries: &[u32]) {
    let carried = usize::try_from(dump.avic.logical_entries).unwrap_or(entries.len());
    let carried = carried.min(entries.len());
    let mut named = 0;
    for (slot, entry) in entries[..carried].iter().enumerate() {
        if entry & LOGICAL_VALID == 0 {
            continue;
        }
        named += 1;
        row(
            &format!("slot {slot}"),
            &format!(
                "{entry:#010x}  cluster {}, member {}, guest apic id {:#x}",
                slot / MEMBERS_PER_CLUSTER,
                slot % MEMBERS_PER_CLUSTER,
                entry & LOGICAL_GUEST_ID
            ),
        );
    }
    row("entries", &format!("{named} of {carried} name a processor"));
}

/// One bank of vectors, as a table of bits.
///
/// Thirty-two vectors to a row, lowest first, with the numbers a row covers
/// beside it. A dot is a vector the bank does not hold and a hash is one it
/// does, so a bank with something in it is legible at a glance and one with
/// nothing in it is obviously empty.
fn bank(name: &str, slots: &[u32; BANK_SLOTS]) {
    for (slot, held) in slots.iter().enumerate() {
        let first = slot * u32::BITS as usize;
        let label = if slot == 0 { name } else { "" };
        println!("  {label:<COLUMN$}{first:>3}: {}", bits(*held));
    }
}

/// One slot of a bank as a row of the table: a dot per vector it does not hold
/// and a hash per vector it does, lowest first.
fn bits(slot: u32) -> String {
    (0..u32::BITS)
        .map(|bit| if slot & (1 << bit) == 0 { '.' } else { '#' })
        .collect()
}

/// The timer's configuration, as much of it as the section carries.
///
/// The divide and the frequency are the emulated controller's alone — the
/// host's driver reads neither back — so both are given as [`ABSENT`] and
/// `None` where they are not part of the answer, rather than printed as zero.
fn timer(
    mode: u32,
    divide: u32,
    initial: u32,
    remaining: u32,
    deadline: u64,
    frequency: Option<u64>,
) {
    row("timer mode", &timing(mode));
    if divide != ABSENT {
        row("timer divide", &format!("{divide:#010x}"));
    }
    row("timer initial count", &format!("{initial:#010x}"));
    row("timer count now", &format!("{remaining:#010x}"));
    row("timer deadline", &format!("{deadline:#018x}"));
    if let Some(frequency) = frequency {
        row("timer rate", &format!("{frequency} ticks per second"));
    }
}

/// One named value, in the column the whole report lines up on.
fn row(name: &str, value: &str) {
    println!("  {name:<COLUMN$}{value}");
}

/// The face a controller is in, or the word the host wrote if this build does
/// not know it.
fn mode(word: u32) -> String {
    Mode::from_word(word).map_or_else(|| unknown(word), |mode| mode.name().to_owned())
}

/// How a timer counts, likewise.
fn timing(word: u32) -> String {
    TimerMode::from_word(word).map_or_else(|| unknown(word), |mode| mode.name().to_owned())
}

/// Where a processor is in the sequence that starts it, likewise.
fn startup(word: u32) -> String {
    Startup::from_word(word).map_or_else(|| unknown(word), |phase| phase.name().to_owned())
}

/// A value this build has no name for, printed as the word it is.
fn unknown(word: u32) -> String {
    format!("unknown ({word:#x})")
}

/// A vector, or that there is none.
fn vector(word: u32) -> String {
    if word == ABSENT {
        "none".to_owned()
    } else {
        format!("{word:#04x}")
    }
}

/// A start-up page, as its number and the address a processor would begin at.
fn page(word: u32) -> String {
    if word == ABSENT {
        "none".to_owned()
    } else {
        format!(
            "{word:#04x} (begins at {:#x})",
            u64::from(word) << PAGE_SHIFT
        )
    }
}

/// A yes-or-no answer, in a column of them.
fn yes(held: bool) -> &'static str {
    if held { "yes" } else { "no" }
}

/// One local vector table entry: the word, and the fields worth reading out of
/// it.
fn lvt(entry: u32) -> String {
    let fields = named([
        (entry & LVT_MASKED != 0, "masked"),
        (entry & LVT_SEND_PENDING != 0, "send pending"),
        (entry & LVT_REMOTE_IRR != 0, "remote irr"),
        (entry & LVT_LEVEL != 0, "level triggered"),
        (entry & LVT_ACTIVE_LOW != 0, "active low"),
    ]);
    format!(
        "{entry:#010x}  vector {:#04x}, delivery {:#05b}, {fields}",
        entry & LVT_VECTOR,
        (entry & LVT_DELIVERY) >> LVT_DELIVERY_SHIFT
    )
}

/// What the host could say about one of the real controller's sources.
fn source(state: SourceState) -> String {
    if !state.contains(SourceState::READ) {
        return "not read".to_owned();
    }
    named([
        (state.contains(SourceState::MASKED), "masked"),
        (state.contains(SourceState::PENDING), "send pending"),
        (state.contains(SourceState::REMOTE_IRR), "remote irr"),
    ])
}

/// The emulated controller's condition.
fn emulated_state(flags: EmulatedFlags) -> String {
    named([
        (flags.contains(EmulatedFlags::BOOTSTRAP), "bootstrap"),
        (flags.contains(EmulatedFlags::STARTABLE), "startable"),
        (
            flags.contains(EmulatedFlags::SOFTWARE_ENABLED),
            "software enabled",
        ),
        (flags.contains(EmulatedFlags::ACCEPTING), "accepting"),
        (flags.contains(EmulatedFlags::RUNNING), "running"),
        (flags.contains(EmulatedFlags::AWAY), "away"),
        (flags.contains(EmulatedFlags::OWNED), "owned"),
        (
            flags.contains(EmulatedFlags::AVIC_INHIBITED),
            "acceleration inhibited",
        ),
        (
            flags.contains(EmulatedFlags::DEADLINE_OFFERED),
            "deadline timer offered",
        ),
        (
            flags.contains(EmulatedFlags::X2APIC_OFFERED),
            "wider face offered",
        ),
    ])
}

/// The state of hardware-driven delivery on this processor.
fn avic_state(flags: AvicFlags) -> String {
    named([
        (flags.contains(AvicFlags::PROVISIONED), "provisioned"),
        (flags.contains(AvicFlags::OWN_PAGE), "has a page"),
        (flags.contains(AvicFlags::ACCELERATED), "block accelerated"),
        (flags.contains(AvicFlags::WIDER_FACE), "in the wider face"),
        (flags.contains(AvicFlags::BLOCK_ENABLE), "block enable"),
        (
            flags.contains(AvicFlags::BLOCK_X2_ENABLE),
            "block x2 enable",
        ),
        (
            flags.contains(AvicFlags::PROVISIONED) && !flags.contains(AvicFlags::DELIVERABLE_KNOWN),
            "page unreadable",
        ),
        (flags.contains(AvicFlags::DELIVERABLE), "page has something"),
        (
            flags.contains(AvicFlags::X2APIC_OFFERED),
            "wider face offered",
        ),
        (
            flags.contains(AvicFlags::IPI_VIRTUAL),
            "running bits trusted",
        ),
        (
            flags.contains(AvicFlags::MACHINE_INHIBITED),
            "machine inhibited",
        ),
        (
            flags.contains(AvicFlags::PHYSICAL_TRUNCATED),
            "physical table truncated",
        ),
    ])
}

/// The names of the conditions that hold, or a dash where none of them does.
fn named<'a>(held: impl IntoIterator<Item = (bool, &'a str)>) -> String {
    let names: Vec<&str> = held
        .into_iter()
        .filter_map(|(held, name)| held.then_some(name))
        .collect();
    if names.is_empty() {
        "-".to_owned()
    } else {
        names.join(", ")
    }
}

/// One register out of the backing page, at the offset the architecture puts
/// it.
fn word(page: &[u8; PAGE_BYTES], offset: usize) -> u32 {
    let bytes = page[offset..offset + size_of::<u32>()]
        .try_into()
        .expect("a register of the page is four bytes inside it");
    u32::from_le_bytes(bytes)
}

/// How wide the name column is, which every line of the report lines up on.
const COLUMN: usize = 32;

/// The emulated controller's local vector table entries, in the order the dump
/// carries them.
const EMULATED_SOURCES: [&str; LVT_ENTRIES] = [
    "timer",
    "lint0",
    "lint1",
    "error",
    "performance",
    "thermal",
    "corrected machine check",
];

/// The real controller's own sources, in the order the dump carries them.
const REAL_SOURCES: [&str; SOURCES] = [
    "timer",
    "lint0",
    "lint1",
    "thermal",
    "performance",
    "corrected machine check",
];

/// Every register the architecture puts in the controller's page, at its own
/// offset.
///
/// The write-only ones are left out: the acknowledgement register and the
/// self-interrupt register answer nothing, and a value read out of them would
/// be whatever the page happened to hold rather than a register.
const PAGE_REGISTERS: [(usize, &str); 19] = [
    (0x020, "id"),
    (0x030, "version"),
    (0x080, "task priority"),
    (0x090, "arbitration priority"),
    (0x0A0, "processor priority"),
    (0x0D0, "logical destination"),
    (0x0E0, "destination format"),
    (0x0F0, "spurious"),
    (0x280, "error status"),
    (0x2F0, "lvt corrected machine check"),
    (0x300, "interrupt command low"),
    (0x310, "interrupt command high"),
    (0x320, "lvt timer"),
    (0x330, "lvt thermal"),
    (0x340, "lvt performance"),
    (0x350, "lvt lint0"),
    (0x360, "lvt lint1"),
    (0x370, "lvt error"),
    (0x3E0, "timer divide"),
];

/// The three banks of vectors in the page, at the offset each of them begins
/// at.
const PAGE_BANKS: [(usize, &str); 3] = [
    (0x100, "in service"),
    (0x180, "trigger mode"),
    (0x200, "request"),
];

/// How far apart the slots of a bank are in the page: every register in it is a
/// doubleword on a sixteen-byte boundary.
const BANK_STRIDE: usize = 0x10;

/// The fields of a local vector table entry, as the architecture lays them out.
///
/// Named here rather than taken from the interface because the interface
/// deliberately carries the register as the word it is: what a bit of it means
/// is the reader's to say.
const LVT_VECTOR: u32 = 0xFF;

/// Which of the delivery modes the entry asks for.
const LVT_DELIVERY: u32 = 0b111 << LVT_DELIVERY_SHIFT;

/// Where that field sits.
const LVT_DELIVERY_SHIFT: u32 = 8;

/// A delivery from this source has been accepted and not yet handed to the
/// processor.
const LVT_SEND_PENDING: u32 = 1 << 12;

/// The pin is asserted low rather than high.
const LVT_ACTIVE_LOW: u32 = 1 << 13;

/// A level-triggered interrupt from this pin has been accepted and not
/// acknowledged.
const LVT_REMOTE_IRR: u32 = 1 << 14;

/// The pin is level triggered rather than edge triggered.
const LVT_LEVEL: u32 = 1 << 15;

/// The source delivers nothing.
const LVT_MASKED: u32 = 1 << 16;

/// The bit of a physical-table entry that says it describes a processor at all.
const ENTRY_VALID: u64 = 1 << 63;

/// The bit that says that processor is in the guest right now.
const ENTRY_RUNNING: u64 = 1 << 62;

/// Where its controller registers are backed.
const ENTRY_PAGE: u64 = 0x0000_FFFF_FFFF_F000;

/// Which physical processor is running it.
const ENTRY_HOST_ID: u64 = 0xFFF;

/// The bit of a logical-table entry that says it names a processor.
const LOGICAL_VALID: u32 = 1 << 31;

/// Which of the guest's processors it names.
const LOGICAL_GUEST_ID: u32 = 0xFF;

/// How many members a cluster of logical destinations has, which is what the
/// slots of the logical table are grouped in.
const MEMBERS_PER_CLUSTER: usize = 4;

/// Bits a page number is shifted by to become an address.
const PAGE_SHIFT: u32 = 12;

/// The logical table's slots divide into whole clusters, which is what makes a
/// slot number readable as a cluster and a member of it.
const _: () = assert!(
    LOGICAL_ENTRIES.is_multiple_of(MEMBERS_PER_CLUSTER),
    "the logical table does not divide into whole clusters",
);

/// Every register and every bank this report reads out of the backing page lies
/// inside it, which is what makes reading one a slice of the page rather than
/// something that can fail.
const _: () = {
    let mut index = 0;
    while index < PAGE_REGISTERS.len() {
        assert!(
            PAGE_REGISTERS[index].0 + size_of::<u32>() <= PAGE_BYTES,
            "a register this report names is outside the page",
        );
        index += 1;
    }
    let mut index = 0;
    while index < PAGE_BANKS.len() {
        assert!(
            PAGE_BANKS[index].0 + (BANK_SLOTS - 1) * BANK_STRIDE + size_of::<u32>() <= PAGE_BYTES,
            "a bank this report names runs off the end of the page",
        );
        index += 1;
    }
};

#[cfg(test)]
mod tests {
    //! There is no machine to run the report against here: the instruction it
    //! begins with is not intercepted on a host, and a dump cannot be produced
    //! without a hypervisor. So what is tested is the part that can go wrong
    //! without one — how a bank becomes a table, and that a whole report
    //! renders for a dump whose every field is filled in.

    use hypercall::{ApicDump, Header, Present};

    use super::{bits, report};

    #[test]
    fn a_slot_becomes_one_row_of_the_table() {
        assert_eq!(bits(0), ".".repeat(u32::BITS as usize));
        assert_eq!(bits(u32::MAX), "#".repeat(u32::BITS as usize));
        // Lowest vector first, which is what makes a row read like the numbers
        // beside it.
        assert!(bits(0b1101).starts_with("#.##"));
        assert!(bits(1 << 31).ends_with(".#"));
    }

    #[test]
    fn a_dump_with_every_section_present_renders() {
        // The whole report, over a buffer whose every byte is set: the page
        // offsets, the entry decoding and the column arithmetic all run, and any
        // of them reaching outside what it was given fails here rather than in a
        // guest.
        let mut dump = ApicDump::zeroed();
        dump.emulated.request = [u32::MAX; 8];
        dump.backing = [0xFF; hypercall::PAGE_BYTES];
        dump.physical = [u64::MAX; hypercall::PHYSICAL_ENTRIES];
        dump.logical = [u32::MAX; hypercall::LOGICAL_ENTRIES];
        dump.avic.physical_entries = u32::MAX;
        dump.avic.logical_entries = u32::MAX;
        dump.header = Header::new(Present::all(), 1, 2, 3, 4);

        report(&dump);
    }

    #[test]
    fn a_dump_with_nothing_in_it_renders_as_nothing_read() {
        // The other end of the same path: a host that could read no part of the
        // machine, which must print six "not read" lines rather than six sections
        // of zeroes.
        let mut dump = ApicDump::zeroed();
        dump.header = Header::new(Present::empty(), 0, 0, 0, 1);

        report(&dump);
    }
}
