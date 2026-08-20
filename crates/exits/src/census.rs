//! A running count of what a guest's exits were, for when it stops making
//! progress.
//!
//! Logging an exit as it happens costs more than the exit did: every byte
//! leaves through a polled UART register, and on a virtualized machine each of
//! those port accesses is itself a world switch. A guest reporting every exit
//! is a guest that never runs. So nothing is written per exit — the reasons are
//! counted, and one summary goes out every [`Census::INTERVAL`] exits.
//!
//! What the summary is for is telling apart the three ways a guest stops
//! getting anywhere, which look identical from outside: one reason repeating
//! forever at the same address is a guest spinning on something the host is not
//! answering; a varied spread with the address moving is a guest that is simply
//! slow; and silence — no summary at all — is a guest that has stopped taking
//! exits, which is the one case the counters cannot describe and their absence
//! can.

use log::info;
use svm::{ExitCode, avic::IpiFailure};
use vcpu::Vcpu;

/// One processor's exit counts since the last summary.
///
/// Keyed on the raw exit code rather than the decoded [`Reason`], because a
/// decoded reason carries the register or vector it concerns and so is not a
/// fixed set. The codes the architecture packs into its low range get a slot
/// each; the sparse ones above it share [`Census::SPARSE`] slots, which is
/// enough for every code a working guest produces and degrades by undercounting
/// rather than by growing.
#[derive(Debug)]
pub(crate) struct Census {
    /// Counts for the codes below [`Census::DENSE`], indexed by the code.
    dense: [u32; Self::DENSE],
    /// Counts for the codes at or above it, paired with the code they are for.
    sparse: [(ExitCode, u32); Self::SPARSE],
    /// How many of [`Census::sparse`] are in use.
    sparse_used: usize,
    /// Incomplete inter-processor deliveries, indexed by the failure the
    /// hardware reported: the two exits the acceleration raises deserve a
    /// finer account than their code alone, because it is the cause that
    /// says whether the machine is healthy.
    incomplete_ipi: [u32; Self::IPI_FAILURES],
    /// Unaccelerated register accesses, indexed by the slot of the register
    /// page they named: which register a guest cannot touch accelerated is
    /// the question this half answers.
    noaccel: [u32; Self::NOACCEL_SLOTS],
    /// Hardware doorbells rung since the last summary, as the vlapic crate
    /// counts them cumulatively.
    last_doorbells: u64,
    /// Host-interrupt kicks sent since the last summary, likewise.
    last_kicks: u64,
    /// Exits counted since the last summary.
    counted: u64,
    /// Exits counted since this processor entered its guest.
    lifetime: u64,
}

impl Census {
    /// Exit codes below this get a counter of their own.
    const DENSE: usize = 0x100;

    /// How many distinct codes at or above [`Census::DENSE`] are counted. The
    /// architecture defines four, and a guest reaching more than this many
    /// would be reporting codes no processor documents.
    const SPARSE: usize = 8;

    /// Exits between summaries. Large enough that the summary itself is not
    /// what the guest is spending its time on, small enough that a guest which
    /// is stuck says so within seconds.
    const INTERVAL: u64 = 2_000;

    /// How many reasons one summary names, most frequent first. A guest that is
    /// stuck repeats one or two; the rest of the distribution is noise.
    const NAMED: usize = 6;

    /// How many failures an incomplete delivery can report.
    const IPI_FAILURES: usize = 6;

    /// How many register slots the unaccelerated counts distinguish: the
    /// page's offset shifted past its four zero bits.
    const NOACCEL_SLOTS: usize = 256;

    /// A census that has counted nothing.
    pub(crate) const fn new() -> Self {
        Self {
            dense: [0; Self::DENSE],
            sparse: [(ExitCode::INVALID, 0); Self::SPARSE],
            sparse_used: 0,
            incomplete_ipi: [0; Self::IPI_FAILURES],
            noaccel: [0; Self::NOACCEL_SLOTS],
            last_doorbells: 0,
            last_kicks: 0,
            counted: 0,
            lifetime: 0,
        }
    }

    /// Counts one exit, and writes a summary if enough have accumulated.
    ///
    /// Called with the guest's state as the exit left it, which is what makes
    /// the address in the summary meaningful: it is where the guest was when it
    /// last stopped, so the same address in successive summaries is a guest
    /// that is not moving.
    pub(crate) fn record(&mut self, vcpu: &Vcpu) {
        let code = vcpu.control().exit_code;
        let raw = code.bits();
        // Saturating because a counter that wrapped would report a stuck guest
        // as an idle one, which is the opposite of what this is for.
        if let Ok(index) = usize::try_from(raw)
            && index < Self::DENSE
        {
            self.dense[index] = self.dense[index].saturating_add(1);
        } else {
            self.count_sparse(code);
        }
        self.counted = self.counted.saturating_add(1);
        self.lifetime = self.lifetime.saturating_add(1);
        if self.counted >= Self::INTERVAL {
            self.summarize(vcpu);
        }
    }

    /// Counts one exit whose code is outside the dense range, ignoring it if
    /// this census is already tracking [`Census::SPARSE`] distinct such codes.
    fn count_sparse(&mut self, code: ExitCode) {
        for entry in &mut self.sparse[..self.sparse_used] {
            if entry.0 == code {
                entry.1 = entry.1.saturating_add(1);
                return;
            }
        }
        if self.sparse_used < Self::SPARSE {
            self.sparse[self.sparse_used] = (code, 1);
            self.sparse_used += 1;
        }
    }

    /// Counts an incomplete inter-processor delivery by the failure the
    /// hardware reported.
    pub(crate) fn incomplete_ipi(&mut self, cause: IpiFailure) {
        let index = cause.into_bits() as usize;
        if let Some(count) = self.incomplete_ipi.get_mut(index) {
            *count = count.saturating_add(1);
        }
    }

    /// Counts an unaccelerated register access by the slot it named.
    pub(crate) fn noaccel(&mut self, offset: u16) {
        let index = usize::from(offset >> 4);
        if let Some(count) = self.noaccel.get_mut(index) {
            *count = count.saturating_add(1);
        }
    }

    /// Writes one summary and starts counting afresh.
    ///
    /// The lifetime total is not reset: it is what says whether a processor
    /// which has gone quiet ever ran at all.
    fn summarize(&mut self, vcpu: &Vcpu) {
        let save = vcpu.save();
        info!(
            "exits: {} exits ({} in all), guest at {:#x}, cs {:#x}, cr3 {:#x}, rflags {:#x}",
            self.counted, self.lifetime, save.rip, save.cs.selector, save.cr3, save.rflags
        );
        for _ in 0..Self::NAMED {
            let Some((code, count)) = self.hottest() else {
                break;
            };
            match code.reason() {
                Some(reason) => info!("exits:   {count} of {reason:?}"),
                None => info!("exits:   {count} of undocumented code {:#x}", code.bits()),
            }
            self.clear(code);
        }
        // The acceleration's own account, which the exit codes cannot give:
        // why deliveries between the guest's processors stopped, which
        // registers it still touches by hand, and how the running targets
        // were woken — the measure of whether the acceleration is doing its
        // job.
        for (id, count) in self.incomplete_ipi.iter().enumerate() {
            if *count > 0 {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "the index came from an array the failure count fits in"
                )]
                let cause = IpiFailure::from_bits(id as u32);
                info!("exits:   {count} incomplete IPI, cause {cause:?}");
            }
        }
        for (slot, count) in self.noaccel.iter().enumerate() {
            if *count > 0 {
                info!(
                    "exits:   {count} unaccelerated accesses at offset {:#x}",
                    slot << 4
                );
            }
        }
        let doorbells = vlapic::avic_doorbells().wrapping_sub(self.last_doorbells);
        let kicks = vlapic::avic_kicks().wrapping_sub(self.last_kicks);
        if doorbells > 0 || kicks > 0 {
            info!("exits:   {doorbells} hardware doorbells rung, {kicks} host-interrupt kicks");
        }
        self.last_doorbells = vlapic::avic_doorbells();
        self.last_kicks = vlapic::avic_kicks();
        self.dense.fill(0);
        self.sparse_used = 0;
        self.incomplete_ipi.fill(0);
        self.noaccel.fill(0);
        self.counted = 0;
    }

    /// The most frequent code still counted, or `None` once none is left.
    fn hottest(&self) -> Option<(ExitCode, u32)> {
        let dense = self
            .dense
            .iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .map(|(index, count)| (ExitCode::from_bits(index as u64), *count));
        let sparse = self.sparse[..self.sparse_used]
            .iter()
            .filter(|(_, count)| *count > 0)
            .copied();
        dense.chain(sparse).max_by_key(|(_, count)| *count)
    }

    /// Forgets `code`, so the next [`Census::hottest`] names the one below it.
    fn clear(&mut self, code: ExitCode) {
        if let Ok(index) = usize::try_from(code.bits())
            && index < Self::DENSE
        {
            self.dense[index] = 0;
            return;
        }
        for entry in &mut self.sparse[..self.sparse_used] {
            if entry.0 == code {
                entry.1 = 0;
                return;
            }
        }
    }
}
