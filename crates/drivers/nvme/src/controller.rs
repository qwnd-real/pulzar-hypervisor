//! One controller: the shadow of its admin queues, and the two faces it is
//! answered through.
//!
//! Everything a guest does with an `NVMe` controller passes through BAR0. The
//! first page holds the registers that configure the admin queues, and the
//! pages after it hold the doorbells that ring every queue — the admin pair
//! first, at the stride the capabilities reported, and every I/O queue's
//! pair interleaved after them. This driver traps writes to all of it and
//! lets every read through: a read of a register the hardware answers
//! honestly costs nothing, and the writes it watches are rare on every path
//! but one.
//!
//! That one path is the doorbell array. A page is the finest the nested
//! tables can trap, so the admin doorbells cannot be watched without the I/O
//! doorbells that share their page being watched too. An I/O doorbell write
//! is therefore one exit more than it would otherwise be, and everything
//! this file does is bent on making that exit cost nothing but the exit
//! itself: no lock, no allocation, one comparison and one volatile write.
//!
//! The array is not the only thing in the pages past the register file: a
//! controller may put its message-signaled interrupt table among them, as
//! the emulated one this workspace boots does. Writes there are forwarded
//! exactly as the guest made them — at the guest's own width, which is what
//! the forwarding below is written to preserve — and cost the one exit the
//! trapping already spends.

use alloc::boxed::Box;
use core::hint::spin_loop;

use emulate::{Capability, Commit, Data, Device, Hardware, Read, Region, Trap, Width, Write};
use log::{info, warn};
use paging::{AddressSpace, CacheType, Mapping, Protection};
use partition::Partition;
use pci::Function;
use spin::Mutex;
use x86_64::{PhysAddr, VirtAddr};

use crate::{
    NvmeError,
    command::{self, Identified},
    identify,
    regs::{self, Aqa, Cap, Cc, PAGE},
};

/// How many identify commands may be awaiting answers at once, and how many
/// stragglers from answers given up on may be remembered.
///
/// A guest submits identifies a handful at a time, at its own boot; more
/// than this arriving at once is a guest behaving in a way nothing does, and
/// the response it gets is the hardware's own.
const PENDING: usize = 16;

/// How long the driver waits for a completion after ringing the admin
/// submission doorbell, in spins.
///
/// On the order of a second at a few dozen cycles a spin. Identify commands
/// answer in microseconds on real hardware and on emulated; a controller
/// that has not answered by now has failed, and what happens next is the
/// same failing open as any other way of losing the queue.
const ANSWER_SPINS: u32 = 20_000_000;

/// One controller, taken over when the guest's firmware services ended and
/// answered for from then on.
pub(crate) struct Controller {
    /// The guest whose queues this driver reads and writes.
    partition: &'static Partition,
    /// Where BAR0 is in the guest's physical memory.
    bar0: PhysAddr,
    /// How far BAR0 decodes.
    extent: u64,
    /// This driver's own mapping of the doorbell array, kept for as long as
    /// the controller is answered for. The doorbells are rung through it
    /// rather than through the emulator's commit, because a submission has
    /// to reach the hardware *before* the answer is waited for, and a commit
    /// is not performed until the handler has returned.
    doorbells: Mapping,
    /// How far apart two queues' doorbells are.
    stride: u64,
    /// Everything a write can change.
    admin: Mutex<Admin>,
}

impl Controller {
    /// Takes a controller over: asks its first base address register how far
    /// it decodes, reads the capabilities for the doorbell stride, maps the
    /// doorbell array, and builds the state the writes build on.
    ///
    /// # Errors
    ///
    /// [`NvmeError::Unaddressed`] if the register decodes nothing or nothing
    /// at all, [`NvmeError::TooSmall`] if it holds no doorbell page,
    /// [`NvmeError::Misaligned`] if it is not page aligned, or whatever
    /// asking it or mapping behind it reported.
    pub(crate) fn take(
        partition: &'static Partition,
        space: &mut AddressSpace,
        function: &Function,
    ) -> Result<&'static Self, NvmeError> {
        let base = function.bars()[0]
            .memory_base()
            .ok_or(NvmeError::Unaddressed)?;
        // SAFETY: the guest's ExitBootServices has been intercepted, so the
        // devices are this hypervisor's to configure: the guest is stopped
        // mid-call and the other processors have not been started, which
        // leaves no driver to lose a transaction to.
        let Some(extent) = unsafe { pci::size(function, 0) }? else {
            return Err(NvmeError::Unaddressed);
        };
        if extent < regs::DOORBELLS + PAGE {
            return Err(NvmeError::TooSmall { bytes: extent });
        }
        if !base.as_u64().is_multiple_of(PAGE) {
            return Err(NvmeError::Misaligned {
                base: base.as_u64(),
            });
        }
        let capabilities = capabilities(space, base)?;
        let stride = regs::stride(Cap::from(capabilities).dstrd());
        // SAFETY: the doorbell array is device registers, outside every
        // range the memory map calls memory, and nothing else maps it for a
        // purpose of its own: the emulator maps nothing for an untouched
        // region. Uncached-minus, because a doorbell left in a cache line
        // would never reach the controller.
        let doorbells = unsafe {
            space.map_physical(
                base + regs::DOORBELLS,
                extent - regs::DOORBELLS,
                Protection::ReadWrite,
                CacheType::UncachedMinus,
            )
        }?;
        Ok(Box::leak(Box::new(Self {
            partition,
            bar0: base,
            extent,
            doorbells,
            stride,
            admin: Mutex::new(Admin::new()),
        })))
    }

    /// The two regions this driver answers for the controller with: the
    /// register file, whose writes configure the admin queues, and the
    /// doorbell array, where those queues are rung.
    ///
    /// A free-standing associate rather than a method because the faces it
    /// builds hold the controller for as long as the guest runs, which is
    /// the `'static` the controller was leaked into.
    pub(crate) fn regions(controller: &'static Self) -> [Region; 2] {
        [
            Region {
                gpa: controller.bar0,
                bytes: PAGE,
                trap: Some(Trap::Writes),
                device: Box::new(Registers(controller)),
            },
            Region {
                gpa: controller.bar0 + regs::DOORBELLS,
                bytes: controller.extent - regs::DOORBELLS,
                trap: Some(Trap::Writes),
                device: Box::new(Doorbells(controller)),
            },
        ]
    }

    /// Logs what taking the controller over found.
    pub(crate) fn describe(&self) {
        info!(
            "nvme: controller at {:#x}, {:#x} bytes, doorbells {} bytes apart",
            self.bar0.as_u64(),
            self.extent,
            self.stride
        );
    }

    /// Merges a write to the register file into the shadow of it. The write
    /// itself goes on to the hardware; what this remembers is what the
    /// admin queues were configured as.
    fn configured(&self, offset: u64, width: Width, value: u64) {
        let mut admin = self.admin.lock();
        match regs::named(offset) {
            regs::Named::Configuration => {
                admin.enable(Cc::from(low_double(value)).enabled());
            }
            regs::Named::SubsystemReset => {
                // A subsystem reset takes the controller down without the
                // enable ever falling, so the shadow learns of it here or
                // never. What it resets to is a disabled controller.
                admin.reset();
                admin.enabled = false;
            }
            regs::Named::Attributes => {
                // Only a whole dword names both queues' depths; a narrower
                // write names part of one, and the specification has
                // nothing to say about what that means, so the shadow keeps
                // the last whole word it saw.
                if width == Width::Long {
                    let attributes = Aqa::from(low_double(value));
                    info!(
                        "nvme: the admin queues are {} submissions by {} completions deep",
                        attributes.submissions(),
                        attributes.completions()
                    );
                    admin.attributes(attributes);
                }
            }
            regs::Named::SubmissionBase { within } => {
                admin.submission_base(within, width, value);
            }
            regs::Named::CompletionBase { within } => {
                admin.completion_base(within, width, value);
            }
            regs::Named::Other => {}
        }
    }

    /// What becomes of a doorbell write: every one but the admin submission
    /// queue's tail is rung and forgotten.
    fn doorbell(&self, offset: u64, width: Width, value: u64) -> Commit {
        // The one thing the storage path pays for the spoofing: a comparison
        // and a volatile write, with no lock held and nothing allocated. The
        // admin completion queue's head doorbell is here too, at the stride,
        // and needs nothing more — the driver reads the queue by phase, and
        // where the guest has consumed to is the hardware's business.
        if offset != 0 {
            self.ring(offset, width, value);
            return Commit::Discard;
        }
        self.submitted(width, value);
        Commit::Discard
    }

    /// Rings a doorbell: one write to the hardware, made exactly as the
    /// guest made it.
    ///
    /// The width is the guest's own, not a fixed one, and that is not
    /// polish: the pages past the register file hold registers of more than
    /// one width — a message-signaled interrupt table among them, on the
    /// controllers that put one there — and a doorbell page is also a page
    /// an interrupt table lives in. Forwarding a byte store as a dword
    /// would write past what the guest wrote, into a register it never
    /// named.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the value came from a Data of this width, so the bits above it are already clear"
    )]
    fn ring(&self, offset: u64, width: Width, value: u64) {
        let at = (self.doorbells.addr() + offset).as_u64();
        // SAFETY: the emulator admitted this access inside the region, which
        // bounds an access of its own width inside the region and aligns it
        // to that width, and the region is exactly what the mapping covers —
        // so each store below is aligned and stays inside the mapping.
        unsafe {
            match width {
                Width::Byte => (at as *mut u8).write_volatile(value as u8),
                Width::Word => (at as *mut u16).write_volatile(value as u16),
                Width::Long => (at as *mut u32).write_volatile(value as u32),
                Width::Quad | Width::Vector => (at as *mut u64).write_volatile(value),
            }
        }
    }

    /// The guest has filled the admin submission queue: read the entries it
    /// filled, ring the doorbell, and wait for the answers.
    fn submitted(&self, width: Width, value: u64) {
        {
            let mut admin = self.admin.lock();
            match admin.submission.map(|queue| queue.along) {
                Some(from) => self.gathered(&mut admin, from, low_word(value)),
                None => warn!(
                    "nvme: the submission doorbell rang with no queue tracked; the commands \
                     pass unwatched"
                ),
            }
        }
        self.ring(0, width, value);
        self.answered();
    }

    /// Reads the entries the guest newly filled and records the identify
    /// commands among them.
    fn gathered(&self, admin: &mut Admin, from: u16, tail: u16) {
        let Some(submission) = admin.submission else {
            return;
        };
        let (base, entries) = (submission.base, submission.entries);
        for step in 0..announced(from, tail, entries) {
            let at = (u32::from(from) + step) % u32::from(entries);
            let where_ = base + u64::from(at) * command::SUBMISSION;
            let mut entry = [0; 64];
            if let Err(error) = self
                .partition
                .with_physical(|physical| physical.read(where_, &mut entry))
            {
                warn!(
                    "nvme: an admin submission could not be read at {where_:?}: {error}; the \
                     commands from it pass unwatched"
                );
                break;
            }
            if let command::Admin::Identify(identify) = command::admin(&entry) {
                info!(
                    "nvme: identify {} asking for {:?} is held for its answer",
                    identify.cid, identify.what
                );
                admin.hold(identify);
            }
        }
        if let Some(submission) = &mut admin.submission {
            submission.along = tail % entries;
        }
    }

    /// Waits for the answers to every identify this controller owes, reading
    /// the admin completion queue as the hardware posts to it and spoofing
    /// each answer as it arrives.
    ///
    /// The wait is inside the doorbell write on purpose. The guest is
    /// stopped on this processor for as long as the handler runs, and the
    /// alternative is not a cleverer moment but none at all: a guest taking
    /// its answers by interrupt reads its identify buffer before it writes
    /// any doorbell again, so the spoof has to be in place before the write
    /// that submits the command returns. Identify commands are asked for a
    /// handful of times a boot, which is what makes waiting for one
    /// affordable.
    ///
    /// The queue is read even when nothing is awaited. The shadow's place in
    /// it has to follow the guest's traffic — most completions are of
    /// commands this driver never held — or the first identify after a
    /// quiet stretch would be read against a queue that has moved on, and
    /// its phase would never match.
    fn answered(&self) {
        let mut patience = ANSWER_SPINS;
        loop {
            let mut admin = self.admin.lock();
            self.harvested(&mut admin);
            if !admin.awaiting() {
                return;
            }
            drop(admin);
            if patience == 0 {
                self.admin.lock().abandoned();
                return;
            }
            spin_loop();
            patience -= 1;
        }
    }

    /// Reads every completion the hardware has posted, spoofing the identify
    /// responses among them and advancing the driver's place in the queue.
    fn harvested(&self, admin: &mut Admin) {
        let Some(queue) = admin.completion else {
            return;
        };
        let (base, entries, mut along) = (queue.base, queue.entries, queue.along);
        let mut phase = admin.phase;
        let mut scanned = 0_u16;
        self.partition.with_physical(|physical| {
            while scanned < entries {
                let where_ = base + u64::from(along) * command::COMPLETION;
                let mut entry = [0; 16];
                if let Err(error) = physical.read(where_, &mut entry) {
                    warn!(
                        "nvme: an admin completion could not be read at {where_:?}: {error}; \
                         the answers awaiting it pass unwatched"
                    );
                    admin.abandoned();
                    return;
                }
                let completion = command::completion(&entry);
                // An entry stamped with the phase the driver is not reading
                // is one from an earlier round of the queue, or nothing at
                // all: the hardware has posted nothing new.
                if completion.phase != phase {
                    return;
                }
                let held = if admin.straggler(completion.cid, along) {
                    // The answer to an identify this driver gave up on,
                    // whose identifier a newer command has since reused:
                    // nobody's answer, skipped so the newer command's own
                    // completion is the one that spoofs its buffer.
                    None
                } else {
                    admin.take(completion.cid)
                };
                if let Some(pending) = held {
                    if completion.succeeded {
                        match pending.what {
                            Identified::Controller => pending.response.controller(&physical),
                            Identified::Namespace => pending.response.namespace(&physical),
                        }
                    } else {
                        warn!(
                            "nvme: identify {} failed, so the hardware's answer is the only \
                             one",
                            pending.cid
                        );
                    }
                }
                let (next, wrapped) = advanced(along, entries);
                along = next;
                if wrapped {
                    phase = !phase;
                }
                scanned += 1;
            }
        });
        if let Some(queue) = &mut admin.completion {
            queue.along = along;
        }
        admin.phase = phase;
    }
}

/// How many submissions a ring from `from` to `tail` announces, in a queue
/// of `entries` positions.
///
/// Both positions are modulo the queue's depth, which the specification
/// requires to be a power of two and which the hardware counts in as well:
/// a tail of zero past a tail of the last entry is one step forward, not a
/// jump. A tail the queue cannot hold is taken modulo it, as the hardware
/// takes it, so a garbage value cannot strand the shadow.
///
/// The one shape this cannot see is a full queue submitted with one ring,
/// whose tail lands where it started — a submission that leaves no room for
/// the completion, which no driver makes. A queue one entry deep is the
/// degenerate case of the same ambiguity and is resolved the other way —
/// every ring announces the one entry — because a guest rings only to
/// submit, and a one-deep queue can hold nothing else.
fn announced(from: u16, tail: u16, entries: u16) -> u32 {
    if entries == 1 {
        return 1;
    }
    let (from, tail) = (u32::from(from), u32::from(tail) % u32::from(entries));
    (u32::from(entries) + tail - from) % u32::from(entries)
}

/// The position after `along` in a queue of `entries` positions, and whether
/// wrapping to it flipped the phase the hardware stamps fresh entries with.
///
/// A queue's phase flips once per lap, when the hardware's writing position
/// passes the last entry and starts again at the first, so the flip travels
/// with the driver's place in the queue rather than being counted apart from
/// it.
fn advanced(along: u16, entries: u16) -> (u16, bool) {
    let next = (along + 1) % entries;
    (next, next == 0)
}

/// What the driver tracks of the admin queues.
struct Admin {
    /// Whether the controller was enabled at the last write to its
    /// configuration. The fall of the enable is what tears the queues down.
    enabled: bool,
    /// The admin queues' attributes, once seen as a whole dword.
    attributes: Option<Aqa>,
    /// The submission queue's base address register, assembled from however
    /// many pieces the guest wrote it in.
    asq: Half,
    /// The completion queue's base address register, likewise.
    acq: Half,
    /// The submission queue, once both its base and its depth are known.
    submission: Option<Queue>,
    /// The completion queue, likewise.
    completion: Option<Queue>,
    /// The completion queue's phase: what the hardware stamps a fresh entry
    /// with. Starts set, because a fresh queue is zeroed memory and a fresh
    /// entry has to differ from it.
    phase: bool,
    /// Identify commands awaiting their answers.
    pending: [Option<Pending>; PENDING],
    /// Answers that may still arrive for identifies this driver gave up on,
    /// each as the command's identifier and the queue position it would
    /// appear at.
    stragglers: [Option<(u16, u16)>; PENDING],
}

impl Admin {
    /// The state of a controller that has been taken over but not yet
    /// configured, which is how its guest finds it.
    const fn new() -> Self {
        Self {
            enabled: false,
            attributes: None,
            asq: Half::new(),
            acq: Half::new(),
            submission: None,
            completion: None,
            phase: true,
            pending: [None; PENDING],
            stragglers: [None; PENDING],
        }
    }

    /// Records the controller's enable, forgetting the queues when it falls.
    ///
    /// A controller taken down loses its queues, and the next enable
    /// configures new ones; the shadow has to agree, or it will read
    /// submissions out of memory nothing occupies.
    fn enable(&mut self, enabled: bool) {
        if self.enabled && !enabled {
            self.reset();
        }
        self.enabled = enabled;
    }

    /// Records the queues' attributes and builds whatever queues they
    /// complete.
    fn attributes(&mut self, attributes: Aqa) {
        self.attributes = Some(attributes);
        self.reconciled();
    }

    /// Merges a write into the submission queue's base register, and builds
    /// the queues if the write changed what the register says.
    fn submission_base(&mut self, within: u64, width: Width, value: u64) {
        let before = self.asq.whole();
        self.asq.write(within, width, value);
        if before != self.asq.whole() {
            self.reconciled();
        }
    }

    /// Merges a write into the completion queue's base register, likewise.
    fn completion_base(&mut self, within: u64, width: Width, value: u64) {
        let before = self.acq.whole();
        self.acq.write(within, width, value);
        if before != self.acq.whole() {
            self.reconciled();
        }
    }

    /// Builds each queue whose base and depth are both known.
    ///
    /// Called whenever either of the two halves of that knowledge arrives,
    /// so whichever order the guest writes them in is the one that works.
    /// The base addresses are masked to their page, as the hardware reads
    /// them: the low bits are reserved, and a queue is page aligned by
    /// requirement.
    ///
    /// A completion queue built different from the one it replaces starts
    /// over, phase and all: the hardware stamps a fresh queue's entries with
    /// the fresh phase, and a rebuilt queue is fresh however the guest came
    /// to rebuild it.
    fn reconciled(&mut self) {
        let Some(attributes) = self.attributes else {
            return;
        };
        self.submission = self
            .asq
            .whole()
            .and_then(|base| Queue::at(base, attributes.submissions()));
        let completion = self
            .acq
            .whole()
            .and_then(|base| Queue::at(base, attributes.completions()));
        if completion != self.completion {
            self.phase = true;
        }
        self.completion = completion;
    }

    /// Whether any identify is awaiting an answer.
    fn awaiting(&self) -> bool {
        self.pending.iter().any(Option::is_some)
    }

    /// Records an identify to answer for, or warns and lets it through when
    /// there is no room to.
    fn hold(&mut self, identify: command::Identify) {
        let Some(slot) = self.pending.iter_mut().find(|slot| slot.is_none()) else {
            warn!(
                "nvme: {PENDING} identifies awaiting answers is more than this driver tracks; \
                 the newest passes unwatched"
            );
            return;
        };
        *slot = Some(Pending {
            what: identify.what,
            cid: identify.cid,
            response: identify::Response::new(identify.prp1, identify.prp2),
        });
    }

    /// Takes the identify a completion answers, if it is one this driver is
    /// holding.
    fn take(&mut self, cid: u16) -> Option<Pending> {
        self.pending
            .iter_mut()
            .find(|slot| slot.as_ref().is_some_and(|pending| pending.cid == cid))
            .and_then(Option::take)
    }

    /// Whether a completion at `position` for `cid` is one this driver gave
    /// up on, and spends the record of it.
    ///
    /// The identifier may have been reused by a newer command before the
    /// answer arrived, and the straggler must not be mistaken for the newer
    /// command's answer: that would spoof a buffer the hardware has not
    /// written yet and leave the real answer to land on top of it. The
    /// record is a position rather than a range because the guest's own
    /// driver serialises its admin submissions — nothing of its can complete
    /// ahead of a command it is still waiting on, so a straggler posts at
    /// the position the queue had reached when the driver gave up.
    fn straggler(&mut self, cid: u16, position: u16) -> bool {
        let found = self.stragglers.iter().position(
            |straggler| matches!(straggler, Some((held, at)) if *held == cid && *at == position),
        );
        match found {
            Some(slot) => {
                self.stragglers[slot] = None;
                true
            }
            None => false,
        }
    }

    /// Gives up on every identify awaiting an answer, warning for each: the
    /// responses reach the guest as the hardware wrote them.
    ///
    /// Each is remembered, in case its answer is merely late rather than
    /// never.
    fn abandoned(&mut self) {
        let position = self.completion.map_or(0, |queue| queue.along);
        let mut given_up = [0_u16; PENDING];
        let mut count = 0;
        for slot in &mut self.pending {
            if let Some(pending) = slot.take() {
                warn!(
                    "nvme: identify {} was still unanswered when the driver gave up; its \
                     answer passes as the hardware wrote it",
                    pending.cid
                );
                given_up[count] = pending.cid;
                count += 1;
            }
        }
        for &cid in &given_up[..count] {
            self.remember(cid, position);
        }
    }

    /// Records a straggler, dropping the oldest remembered one if every slot
    /// is taken — a straggler old enough to be dropped is one whose command
    /// the guest has long stopped waiting on.
    fn remember(&mut self, cid: u16, position: u16) {
        let slot = self
            .stragglers
            .iter()
            .position(Option::is_none)
            .unwrap_or(0);
        self.stragglers[slot] = Some((cid, position));
    }

    /// Forgets the admin queues, as a controller taken down does, and gives
    /// up on whatever answers it owed.
    fn reset(&mut self) {
        self.attributes = None;
        self.asq = Half::new();
        self.acq = Half::new();
        self.submission = None;
        self.completion = None;
        self.phase = true;
        self.stragglers = [None; PENDING];
        self.abandoned();
    }
}

/// One queue, as far as the driver tracks it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Queue {
    /// Where the queue is in the guest's physical memory.
    base: PhysAddr,
    /// How many entries it holds.
    entries: u16,
    /// How far along it the driver has accounted for: the submission
    /// queue's tail as the guest last rang it, the completion queue's head
    /// as far as the driver has read.
    along: u16,
}

impl Queue {
    /// A queue at `base` holding `entries` entries, which the driver has
    /// accounted for none of.
    ///
    /// `None` where the base is not an address the machine can hold, which
    /// is a command the hardware will fail anyway.
    fn at(base: u64, entries: u16) -> Option<Self> {
        let base = PhysAddr::try_new(base & !(PAGE - 1)).ok()?;
        Some(Self {
            base,
            entries,
            along: 0,
        })
    }
}

/// An identify command awaiting its answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Pending {
    /// What the response identifies.
    what: Identified,
    /// The command's identifier, which the completion carries back.
    cid: u16,
    /// Where the response's data buffer is.
    response: identify::Response,
}

/// A 64-bit register the guest may write in pieces, held until it says a
/// whole address.
#[derive(Clone, Copy)]
struct Half {
    /// What has been written so far.
    value: u64,
    /// Which bits of it have been written.
    written: u64,
}

impl Half {
    /// A register nothing has been written to.
    const fn new() -> Self {
        Self {
            value: 0,
            written: 0,
        }
    }

    /// Merges a write at `within` bytes into the register.
    ///
    /// The emulator admits only naturally aligned scalar writes, and an
    /// aligned write never straddles an eight-byte register, so the shift
    /// below cannot reach past the register's own bits.
    fn write(&mut self, within: u64, width: Width, value: u64) {
        let bits = width.mask() << (within * 8);
        self.value = (self.value & !bits) | ((value << (within * 8)) & bits);
        self.written |= bits;
    }

    /// The address the register says, once it says a whole one.
    ///
    /// That is every byte written — or the low half written and the high
    /// half never touched, which is how a guest programs a base below four
    /// gigabytes in one dword and is still a whole address, the high half
    /// reading the zero it was reset to.
    fn whole(&self) -> Option<u64> {
        if self.written == !0 {
            return Some(self.value);
        }
        let low = 0xffff_ffff_u64;
        (self.written & low == low && self.written & !low == 0).then_some(self.value & low)
    }
}

/// The register file's first page, as a device the emulator reaches.
struct Registers(&'static Controller);

impl Device for Registers {
    fn capability(&self) -> Capability {
        // The file holds 32-bit registers and 64-bit pairs, either of which
        // a guest may write a piece at a time, so every aligned scalar width
        // is admitted and the shadow decides what each write means.
        Capability::scalar()
    }

    fn hardware(&self) -> Hardware {
        Hardware::Reached
    }

    fn read(&self, access: Read<'_>) -> Data {
        // The region traps writes only, so this is reached only by a read of
        // a register through a path this driver does not stand in, and the
        // hardware's own answer is the faithful one.
        access
            .hardware()
            .unwrap_or_else(|| Data::from_u64(0, access.width()))
    }

    fn write(&self, access: Write<'_>) -> Commit {
        self.0
            .configured(access.offset(), access.width(), access.value().as_u64());
        Commit::Hardware
    }
}

/// The doorbell array, as a device the emulator reaches.
struct Doorbells(&'static Controller);

impl Device for Doorbells {
    fn capability(&self) -> Capability {
        Capability::scalar()
    }

    fn hardware(&self) -> Hardware {
        // The emulator has no way to ring a doorbell from inside a handler,
        // and a submission has to be rung before its answer is waited for,
        // so this driver maps the array itself and does its own writing.
        Hardware::Untouched
    }

    fn read(&self, access: Read<'_>) -> Data {
        // Doorbells are write-only, and the region traps writes only, so
        // this answers a question nothing asks. Zero is what a write-only
        // register answers.
        Data::from_u64(0, access.width())
    }

    fn write(&self, access: Write<'_>) -> Commit {
        self.0
            .doorbell(access.offset(), access.width(), access.value().as_u64());
        Commit::Discard
    }
}

/// The controller's capabilities, read through a mapping that exists only
/// for the reading.
///
/// # Errors
///
/// Whatever mapping the page or reading the register reported.
fn capabilities(space: &mut AddressSpace, base: PhysAddr) -> Result<u64, NvmeError> {
    // SAFETY: one page of device registers, which the caller has just vouched
    // is this hypervisor's to read, mapped read-only around this one access
    // and unmapped before it returns.
    unsafe {
        space.with_physical(
            base,
            PAGE,
            Protection::ReadOnly,
            CacheType::UncachedMinus,
            |registers| {
                // SAFETY: the capabilities are at the page's start, inside the
                // page this mapping covers.
                register_at(registers, regs::CAP).read_volatile()
            },
        )
    }
    .map_err(NvmeError::from)
}

/// A 64-bit register in a mapped block, as a pointer to read through.
fn register_at(base: VirtAddr, offset: u64) -> *const u64 {
    // The pointer is formed from an address inside a mapping the caller
    // vouches for, at an offset the caller says names a register of this
    // width; nothing is dereferenced here.
    (base + offset).as_u64() as *const u64
}

/// The low sixteen bits of a value, which is all a doorbell carries.
#[expect(
    clippy::cast_possible_truncation,
    reason = "a doorbell carries a sixteen-bit position, so its value is the write's low half"
)]
const fn low_word(value: u64) -> u16 {
    value as u16
}

/// The low thirty-two bits of a value, which is all a 32-bit register is.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the register is 32 bits, so its value is the write's low half"
)]
const fn low_double(value: u64) -> u32 {
    value as u32
}

#[cfg(test)]
mod tests {
    use emulate::Width;
    use x86_64::PhysAddr;

    use super::{Admin, Half, Queue, advanced, announced};
    use crate::{command::Identified, regs::Aqa};

    /// The admin state of a configured controller: two queues, both empty.
    fn configured() -> Admin {
        let mut admin = Admin::new();
        admin.attributes(Aqa::from(0x0007_000f));
        admin.submission_base(0, Width::Quad, 0x2_0000);
        admin.completion_base(0, Width::Quad, 0x2_1000);
        admin
    }

    #[test]
    fn a_ring_past_the_last_entry_is_one_step_forward() {
        // A thirty-two entry queue wrapping: the tail the hardware counts
        // and the tail this once mistook for a jump.
        assert_eq!(announced(31, 0, 32), 1);
        assert_eq!(announced(0, 31, 32), 31);
        assert_eq!(announced(5, 3, 32), 30);
    }

    #[test]
    fn a_tail_the_queue_cannot_hold_is_taken_modulo_it() {
        // A garbage doorbell value strands nothing: it is the position the
        // hardware would take it for.
        assert_eq!(announced(0, 40, 32), 8);
        assert_eq!(announced(30, 0xffff, 32), 1);
    }

    #[test]
    fn a_full_queue_in_one_ring_announces_nothing() {
        // The tail lands where it started, which no driver makes and which
        // this cannot distinguish from an empty ring.
        assert_eq!(announced(7, 7, 32), 0);
        assert_eq!(announced(7, 39, 32), 0);
    }

    #[test]
    fn a_one_entry_queue_announces_its_one_entry() {
        // Every ring is a submission, and a one-deep queue can hold nothing
        // but the submission it was rung for.
        assert_eq!(announced(0, 0, 1), 1);
        assert_eq!(announced(0, 5, 1), 1);
    }

    #[test]
    fn the_phase_flips_once_per_lap() {
        assert_eq!(advanced(0, 32), (1, false));
        assert_eq!(advanced(30, 32), (31, false));
        assert_eq!(advanced(31, 32), (0, true));
        assert_eq!(advanced(0, 1), (0, true));
    }

    #[test]
    fn a_base_written_in_pieces_is_whole_when_finished() {
        // The high dword first: half a register that names no address yet.
        let mut half = Half::new();
        half.write(4, Width::Long, 0x0000_0000);
        assert_eq!(half.whole(), None);
        half.write(0, Width::Long, 0x0000_1000);
        assert_eq!(half.whole(), Some(0x0000_1000));
    }

    #[test]
    fn a_low_dword_alone_is_a_whole_base_below_four_gigabytes() {
        let mut half = Half::new();
        half.write(0, Width::Long, 0x0000_1000);
        assert_eq!(half.whole(), Some(0x0000_1000));
        // A byte of the high half touched spoils it: the guest said
        // something about the high half and the rest of what it said is not
        // known.
        half.write(7, Width::Byte, 0x01);
        assert_eq!(half.whole(), None);
    }

    #[test]
    fn a_base_overwritten_in_pieces_takes_the_new_value() {
        let mut half = Half::new();
        half.write(0, Width::Quad, 0xffff_ffff_ffff_ffff);
        assert_eq!(half.whole(), Some(u64::MAX));
        // The new dword lands in its half of the register and leaves the
        // other half alone.
        half.write(0, Width::Long, 0x0000_2000);
        assert_eq!(half.whole(), Some(0xffff_ffff_0000_2000));
    }

    #[test]
    fn a_narrow_base_write_touches_only_its_own_bytes() {
        let mut half = Half::new();
        half.write(0, Width::Quad, u64::MAX);
        half.write(2, Width::Word, 0x0000);
        assert_eq!(half.whole(), Some(0xffff_ffff_0000_ffff));
        // Single bytes at the register's ends.
        half.write(0, Width::Byte, 0x12);
        half.write(7, Width::Byte, 0x34);
        assert_eq!(half.whole(), Some(0x34ff_ffff_0000_ff12));
    }

    #[test]
    fn both_queues_are_built_from_their_pieces() {
        let admin = configured();
        let submission = admin.submission.unwrap();
        assert_eq!(submission.base, PhysAddr::new(0x2_0000));
        assert_eq!(submission.entries, 0x10);
        assert_eq!(submission.along, 0);
        let completion = admin.completion.unwrap();
        assert_eq!(completion.base, PhysAddr::new(0x2_1000));
        assert_eq!(completion.entries, 8);
    }

    #[test]
    fn a_queue_is_built_whichever_half_arrives_last() {
        let mut admin = Admin::new();
        // The bases first, with nothing yet to say how deep the queues are.
        admin.submission_base(0, Width::Quad, 0x2_0000);
        admin.completion_base(0, Width::Quad, 0x2_1000);
        assert!(admin.submission.is_none());
        // The attributes last, which is the order no driver uses and the
        // order this nevertheless builds from.
        admin.attributes(Aqa::from(0x0000_0000));
        assert_eq!(admin.submission.map(|queue| queue.entries), Some(1));
    }

    #[test]
    fn a_base_off_the_physical_address_space_builds_no_queue() {
        let mut admin = Admin::new();
        admin.attributes(Aqa::from(0));
        admin.submission_base(0, Width::Quad, u64::MAX);
        assert!(admin.submission.is_none());
    }

    #[test]
    fn a_base_rewritten_in_place_rebuilds_the_queue() {
        // The specification says a guest takes the controller down to move a
        // queue; one that moves it anyway leaves a shadow that follows, at
        // the cost of the commands in flight going unwatched.
        let mut admin = configured();
        assert_eq!(
            admin.submission.map(|queue| queue.base),
            Some(PhysAddr::new(0x2_0000))
        );
        admin.submission_base(0, Width::Quad, 0x3_0000);
        assert_eq!(
            admin.submission.map(|queue| queue.base),
            Some(PhysAddr::new(0x3_0000))
        );
        assert_eq!(admin.submission.map(|queue| queue.along), Some(0));
    }

    #[test]
    fn a_base_rewritten_with_the_same_value_moves_nothing() {
        let mut admin = configured();
        let before = admin.submission;
        admin.submission_base(0, Width::Quad, 0x2_0000);
        assert_eq!(admin.submission, before);
    }

    #[test]
    fn a_rebuilt_completion_queue_starts_the_phase_over() {
        let mut admin = configured();
        admin.phase = false;
        admin.completion_base(0, Width::Quad, 0x2_2000);
        assert!(admin.phase);
    }

    #[test]
    fn the_fall_of_the_enable_forgets_the_configuration() {
        let mut admin = configured();
        admin.enable(true);
        admin.hold(identify(9));
        admin.enable(false);
        assert_eq!(admin.submission, None);
        assert_eq!(admin.completion, None);
        assert_eq!(admin.attributes, None);
        assert_eq!(admin.asq.whole(), None);
        assert!(admin.phase);
        assert!(!admin.awaiting());
        // A second fall, with nothing configured, changes nothing.
        admin.enable(false);
        assert_eq!(admin.submission, None);
    }

    #[test]
    fn the_rise_of_the_enable_keeps_what_was_configured() {
        let mut admin = configured();
        admin.enable(true);
        assert_eq!(admin.submission.map(|queue| queue.entries), Some(0x10));
    }

    #[test]
    fn held_identifies_are_taken_in_any_order() {
        let mut admin = Admin::new();
        for cid in 1..=3 {
            admin.hold(identify(cid));
        }
        assert!(admin.awaiting());
        // The second one's completion arrives first.
        assert_eq!(admin.take(2).map(|pending| pending.cid), Some(2));
        assert!(admin.awaiting());
        assert_eq!(admin.take(2), None);
        assert_eq!(admin.take(1).map(|pending| pending.cid), Some(1));
        assert_eq!(admin.take(3).map(|pending| pending.cid), Some(3));
        assert!(!admin.awaiting());
    }

    #[test]
    fn a_completion_of_a_command_never_held_answers_nothing() {
        let mut admin = Admin::new();
        // Nothing is held, so a completion of anything is not this driver's
        // business and must not be mistaken for one.
        assert_eq!(admin.take(0x1234), None);
        assert!(!admin.awaiting());
    }

    #[test]
    fn the_pending_fills_and_then_overflows_loudly() {
        let mut admin = Admin::new();
        for cid in 0..16 {
            admin.hold(identify(cid));
            assert!(admin.awaiting());
        }
        // The seventeenth does not fit; it passes unwatched and the sixteen
        // held stay held.
        admin.hold(identify(16));
        assert_eq!(admin.take(16), None);
        assert_eq!(admin.take(15).map(|pending| pending.cid), Some(15));
    }

    #[test]
    fn abandoning_empties_the_pending_and_remembers_the_stragglers() {
        let mut admin = configured();
        admin.hold(identify(7));
        admin.hold(identify(8));
        admin.abandoned();
        assert!(!admin.awaiting());
        // The answers may still arrive, at the head the queue had reached.
        assert!(admin.straggler(7, 0));
        assert!(!admin.straggler(7, 0));
        assert!(admin.straggler(8, 0));
        // A newer command reusing the identifier is not the straggler.
        admin.hold(identify(7));
        assert!(!admin.straggler(7, 1));
        assert_eq!(admin.take(7).map(|pending| pending.cid), Some(7));
    }

    #[test]
    fn the_straggler_slots_are_reused_when_they_run_out() {
        let mut admin = Admin::new();
        for cid in 0..16 {
            admin.hold(identify(cid));
        }
        admin.abandoned();
        // Sixteen stragglers fill the slots; a seventeenth drops the oldest.
        admin.hold(identify(100));
        admin.abandoned();
        assert!(!admin.straggler(0, 0));
        assert!(admin.straggler(100, 0));
    }

    #[test]
    fn a_queue_masks_its_base_to_the_page() {
        let queue = Queue::at(0x2_0abc, 4).unwrap();
        assert_eq!(queue.base, PhysAddr::new(0x2_0000));
    }

    /// One identify of the controller, held by `cid`.
    fn identify(cid: u16) -> crate::command::Identify {
        crate::command::Identify {
            cid,
            what: Identified::Controller,
            prp1: PhysAddr::new(0x1000),
            prp2: PhysAddr::new(0),
        }
    }
}
