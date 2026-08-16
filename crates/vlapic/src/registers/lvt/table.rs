//! The table as the controller keeps it: what the guest wrote, what it reads
//! back, and what a write is allowed to change.
//!
//! Three of the bits in an entry are the controller's rather than software's,
//! so what the guest wrote and what the guest reads are deliberately two
//! operations. Which bits exist in which entry is [`Entry::writable`]'s, and
//! what any of it does to real hardware is [`crate::hardware::sources`]'s.

use core::sync::atomic::Ordering;

use crate::{
    hardware::sources,
    registers::{
        Vlapic,
        lvt::{Delivery, Entry, Lvt, MASKED, TimerMode},
    },
};

impl Vlapic {
    /// One local-vector-table entry as the guest last wrote it.
    ///
    /// What the guest programmed, which is what every source is programmed onto
    /// real hardware from. A guest *reading* the register gets
    /// [`Vlapic::lvt_readback`] instead, because three of the bits in it are
    /// hardware's to report rather than software's to set.
    pub(crate) fn lvt(&self, entry: Entry) -> Lvt {
        Lvt::from_bits(self.lvt[entry.index()].load(Ordering::Acquire))
    }

    /// One local-vector-table entry as the guest reads it.
    ///
    /// Two bits of an entry are the controller's and not software's, and both
    /// are answered from the real entry rather than from anything stored here —
    /// because the source behind the entry is the real one, and the real
    /// controller is what maintains them. The delivery-status bit says a
    /// delivery from this source has been accepted and not yet handed to the
    /// processor, and the remote-IRR bit says a level-triggered interrupt from
    /// this pin has been accepted and not yet acknowledged.
    ///
    /// The mask bit is answered from the stored entry, with one exception, and
    /// the exception is the only case the architecture has: hardware masks the
    /// performance-counter entry itself when the counter overflows, so a guest
    /// that armed that source and read it back unmasked would be told a source
    /// is live that hardware has already stopped.
    ///
    /// Nowhere else, because everywhere else the real entry may be masked for a
    /// reason of this hypervisor's rather than of the architecture's — a
    /// delivery mode or a vector it refuses to put on hardware, or a
    /// programming failure — and answering with that would be a mask bit the
    /// guest never wrote. Software changes one field of one of these registers
    /// by reading the whole of it, changing the field and writing it back, so
    /// an invented mask bit does not merely mislead: the guest's next write
    /// stores it, and the source is off for good with the guest's own
    /// registers saying it asked for that. Where a configuration is
    /// refused, the error status register is what says so.
    pub(crate) fn lvt_readback(&self, entry: Entry) -> Lvt {
        let stored = self.lvt(entry);
        let Some(source) = sources::source_of(entry) else {
            return stored;
        };
        let Ok(real) = apic::local().and_then(|local| local.source(source)) else {
            return stored;
        };
        let overflowed = matches!(entry, Entry::Performance) && real.is_masked();
        stored
            .with_send_pending(real.pending())
            .with_remote_irr(entry.is_pin() && real.remote_irr())
            .with_masked(stored.masked() || overflowed)
    }

    /// Takes a write to a local-vector-table entry, and answers with what the
    /// entry became.
    ///
    /// Three rules the architecture states about writes here, all enforced:
    /// bits the entry reserves are dropped rather than stored; while the
    /// controller is software-disabled the mask bit cannot be cleared; and a
    /// timer entry that crosses into or out of deadline mode leaves the initial
    /// count behind, because the count registers stop meaning anything there
    /// and hardware clears them as the mode changes. A guest coming back to
    /// a counting mode must not find an old count waiting to start a timer
    /// it never asked for.
    pub(crate) fn write_lvt(&self, entry: Entry, value: u32) -> Lvt {
        let mut kept = value & entry.writable(self.model);
        if !self.software_enabled() {
            kept |= MASKED;
        }
        let was = self.lvt[entry.index()].swap(kept, Ordering::AcqRel);
        if entry == Entry::Timer {
            let old_mode = timer_mode(was);
            let new_mode = timer_mode(kept);
            if old_mode != new_mode {
                self.set_timer_periodic_running(false);
            }
            if waits_for_a_deadline(was) != waits_for_a_deadline(kept) {
                self.timer_initial.store(0, Ordering::Release);
                self.clear_timer_clamp();
            }
        }
        Lvt::from_bits(kept)
    }

    /// Whether an entry, as it stands, would actually deliver a vector.
    ///
    /// Which is the only condition under which its vector field means anything.
    /// Every other delivery mode is an event the processor takes by its own
    /// architectural entry point and reads no vector for, so a number left in
    /// the field is not a vector at all and reporting it as an illegal one
    /// would be reporting an error about a field nothing reads.
    ///
    /// Asked of a value rather than of the stored register, so that a caller
    /// deciding this about a write it has just performed decides it about what
    /// it wrote.
    pub(crate) fn delivers_a_vector(&self, entry: Entry, lvt: Lvt) -> bool {
        !self.model.has_delivery(entry)
            || Delivery::from_bits(lvt.delivery()) == Some(Delivery::Fixed)
    }
}

/// Whether a timer entry selects the mode that counts nothing, and so the mode
/// the count registers mean nothing in.
fn waits_for_a_deadline(entry: u32) -> bool {
    timer_mode(entry) == Some(TimerMode::Deadline)
}

/// The timer mode encoded in a raw local-vector-table entry.
fn timer_mode(entry: u32) -> Option<TimerMode> {
    TimerMode::from_bits(Lvt::from_bits(entry).timer_mode())
}
