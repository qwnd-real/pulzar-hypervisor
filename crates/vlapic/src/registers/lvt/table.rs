//! The table as the controller keeps it: what the guest wrote, what it reads
//! back, and what a write is allowed to change.
//!
//! Three of the bits in an entry are the controller's rather than software's, so
//! what the guest wrote and what the guest reads are deliberately two
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
    /// Three bits of an entry are the controller's and not software's, and all
    /// three are answered from the real entry rather than from anything stored
    /// here — because the source behind the entry is the real one, and the real
    /// controller is what maintains them.
    ///
    /// The delivery-status bit says a delivery from this source is still in
    /// flight. The remote-IRR bit says a level-triggered interrupt from this
    /// pin has been accepted and not yet acknowledged. And the mask bit is
    /// not purely software's either: hardware sets it itself on the
    /// performance-counter entry when the counter overflows, so a guest that
    /// armed that source and reads it back unmasked would be told a source is
    /// live that hardware has already stopped.
    pub(crate) fn lvt_readback(&self, entry: Entry) -> Lvt {
        let stored = self.lvt(entry);
        let Some(source) = sources::source_of(entry) else {
            return stored;
        };
        let Ok(real) = apic::local().and_then(|local| local.source(source)) else {
            return stored;
        };
        stored
            .with_send_pending(real.pending())
            .with_remote_irr(entry.is_pin() && real.remote_irr())
            .with_masked(stored.masked() || real.is_masked())
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
    pub(crate) fn delivers_a_vector(&self, entry: Entry) -> bool {
        !self.model.has_delivery(entry)
            || Delivery::from_bits(self.lvt(entry).delivery()) == Some(Delivery::Fixed)
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
