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
use vlapic::Wakes;

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
    /// Incomplete inter-processor deliveries, indexed by the bucket the
    /// identifier the hardware reported falls in: the two exits the
    /// acceleration raises deserve a finer account than their code alone,
    /// because it is the cause that says whether the machine is healthy.
    incomplete_ipi: [u32; Self::IPI_FAILURES],
    /// Unaccelerated register accesses, indexed by the slot of the register
    /// page they named: which register a guest cannot touch accelerated is
    /// the question this half answers.
    noaccel: [u32; Self::NOACCEL_SLOTS],
    /// What this processor's controller had counted as sent by the last
    /// summary, so the next one names only the wakes sent since: the
    /// controller's counters are cumulative and outlive every guest.
    wakes: Wakes,
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

    /// How many buckets an incomplete delivery is counted in: one per failure
    /// the architecture defines, and [`Census::UNDEFINED_IPI`] beyond them.
    const IPI_FAILURES: usize = 7;

    /// The bucket an identifier the architecture does not define is counted in.
    ///
    /// Its index *is* such an identifier — the first one — which is what lets
    /// the summary name every bucket through the same decoder that refuses this
    /// one, rather than through a second table that could disagree with it.
    const UNDEFINED_IPI: usize = 6;

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
            wakes: Wakes {
                kicks: 0,
                nudges: 0,
            },
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
    ///
    /// Keyed on the raw identifier rather than on the decoded cause, because
    /// the decoder answers an undefined identifier with the cause whose
    /// treatment is safe for a request nothing models — the right answer
    /// for the handler and the wrong one here. Such an identifier is
    /// silicon describing something this hypervisor was not built for, and
    /// counted as the commonest legitimate cause it is the one thing in the
    /// summary a reader could not see.
    pub(crate) fn incomplete_ipi(&mut self, reported: u32) {
        let index = match IpiFailure::from_bits(reported) {
            Some(cause) => cause.into_bits() as usize,
            None => Self::UNDEFINED_IPI,
        };
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
        // The acceleration's own account, which the exit codes cannot give.
        // Every number below is work the acceleration did not do for itself: a
        // delivery between the guest's processors that stopped, a register the
        // guest still touches by hand, and a target that had to be interrupted
        // because no hardware announcement could reach it. What the acceleration
        // *did* raises no exit and is counted nowhere at all, so these are read
        // against the exit total above rather than against a total of their own.
        for (bucket, count) in self.incomplete_ipi.iter().enumerate() {
            if *count == 0 {
                continue;
            }
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the index came from an array the failure count fits in"
            )]
            let cause = IpiFailure::from_bits(bucket as u32);
            match cause {
                Some(cause) => info!("exits:   {count} incomplete IPI, cause {cause:?}"),
                // The bucket the decoder refuses, which is where an identifier
                // it refused was counted. Named rather than printed as an
                // absent cause, because this is the one line an operator reads
                // on a machine with no console.
                None => info!(
                    "exits:   {count} incomplete IPI of a cause the architecture does not define"
                ),
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
        // Both transports carry the same host interrupt, so what each number
        // says is where the target will find what it is being woken for: a page
        // the hardware wrote and could not announce, or a model the software
        // path accepted into.
        let sent = self.since(vlapic::wakes());
        for (count, what) in [
            (
                sent.kicks,
                "a request the hardware left in its backing page",
            ),
            (
                sent.nudges,
                "a request the software path accepted into its model",
            ),
        ] {
            if count > 0 {
                info!("exits:   {count} host interrupts to make a target look at {what}");
            }
        }
        self.dense.fill(0);
        self.sparse_used = 0;
        self.incomplete_ipi.fill(0);
        self.noaccel.fill(0);
        self.counted = 0;
    }

    /// What this processor has sent since the last summary, recording what it
    /// has now sent so that the next summary names only what came after.
    ///
    /// Per processor, out of that processor's own controller, exactly as every
    /// other number in a summary is — which is what makes each wake appear in
    /// one summary, on the processor that sent it. A machine-wide counter
    /// differenced against a per-processor last-seen value reported every wake
    /// on the machine in as many summaries as the machine has processors, each
    /// of them presenting it as its own.
    ///
    /// Saturating because the counters saturate: a controller that has sent
    /// `u32::MAX` of something stops counting, where a difference that wrapped
    /// would report a processor that had sent almost none.
    fn since(&mut self, now: Wakes) -> Wakes {
        let sent = Wakes {
            kicks: now.kicks.saturating_sub(self.wakes.kicks),
            nudges: now.nudges.saturating_sub(self.wakes.nudges),
        };
        self.wakes = now;
        sent
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

#[cfg(test)]
mod tests {
    //! Which bucket an incomplete delivery is counted in, and what a summary
    //! makes of the wakes its processor has sent. Those are the two decisions
    //! here that need no guest: everything else this type does is counting, and
    //! what it counts is an exit code the architecture assigns.

    use svm::avic::IpiFailure;
    use vlapic::Wakes;

    use super::Census;

    /// How often a bucket has been counted.
    fn counted(census: &Census, bucket: usize) -> u32 {
        census.incomplete_ipi[bucket]
    }

    #[test]
    fn each_defined_cause_is_counted_in_its_own_bucket() {
        let mut census = Census::new();
        for cause in [
            IpiFailure::InvalidInterruptType,
            IpiFailure::TargetNotRunning,
            IpiFailure::InvalidTarget,
            IpiFailure::InvalidBackingPage,
            IpiFailure::InvalidIpiVector,
            IpiFailure::UnacceleratedIpi,
        ] {
            census.incomplete_ipi(cause.into_bits());
            assert_eq!(counted(&census, cause.into_bits() as usize), 1, "{cause:?}");
        }
    }

    #[test]
    fn an_undefined_cause_is_told_from_the_commonest_one() {
        // The decoder answers an undefined identifier with the invalid-type
        // cause, which is the safe treatment for a request nothing models and
        // would be an invisible defect here: a machine reporting a failure this
        // hypervisor does not model would be indistinguishable in the summary
        // from the cause almost every healthy guest produces most of.
        let mut census = Census::new();
        for reported in [6, 7, 0x1FF, u32::MAX] {
            census.incomplete_ipi(reported);
        }
        assert_eq!(counted(&census, Census::UNDEFINED_IPI), 4);
        assert_eq!(
            counted(
                &census,
                IpiFailure::InvalidInterruptType.into_bits() as usize
            ),
            0,
            "the cause the decoder substitutes must not have been counted"
        );
    }

    #[test]
    fn the_undefined_bucket_is_an_identifier_the_architecture_leaves_undefined() {
        // What lets the summary name every bucket through the decoder rather
        // than through a second table: the bucket the undefined identifiers are
        // counted in is itself one the decoder refuses, so the line it produces
        // says "undefined" without anything having to remember which index that
        // was.
        let bucket = u32::try_from(Census::UNDEFINED_IPI).expect("a bucket index is small");
        assert_eq!(IpiFailure::from_bits(bucket), None);
        assert_eq!(Census::UNDEFINED_IPI, Census::IPI_FAILURES - 1);
        // And every bucket below it is a cause the decoder does name, so no
        // defined cause shares the undefined one's line.
        for defined in 0..bucket {
            assert!(IpiFailure::from_bits(defined).is_some(), "{defined}");
        }
    }

    #[test]
    fn every_wake_is_named_once_by_the_processor_that_sent_it() {
        // Four processors, one kick and one nudge each, and the summaries between
        // them account for those eight wakes exactly once. What each summary
        // differences is its own processor's controller, so a wake belongs to one
        // summary — where a machine-wide counter differenced against a
        // per-processor last-seen value put every wake on the machine into every
        // processor's next summary, each of them presenting it as its own.
        let mut censuses = [const { Census::new() }; 4];
        let named: u32 = censuses
            .iter_mut()
            .map(|census| {
                let sent = census.since(Wakes {
                    kicks: 1,
                    nudges: 1,
                });
                sent.kicks + sent.nudges
            })
            .sum();
        assert_eq!(named, 8);
    }

    #[test]
    fn a_summary_names_the_wakes_sent_since_the_last_one_and_no_others() {
        // The counters are the controller's and cumulative — they outlive every
        // guest and are what a machine with no serial port is read from
        // afterwards — so what a summary owes is the difference, per transport.
        let mut census = Census::new();
        let three = Wakes {
            kicks: 3,
            nudges: 0,
        };
        assert_eq!(census.since(three), three);
        assert_eq!(census.since(three), Wakes::default());
        // A transport that moved while the other stood still is the only one
        // named, which is the whole point of counting them apart.
        assert_eq!(
            census.since(Wakes {
                kicks: 3,
                nudges: 2
            }),
            Wakes {
                kicks: 0,
                nudges: 2
            }
        );
    }
}
