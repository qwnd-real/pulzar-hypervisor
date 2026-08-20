//! The spurious-interrupt vector register, which is also the switch that
//! decides whether the controller delivers anything at all.
//!
//! Software-disabling a controller through it is a transition rather than a
//! flag, and that is the whole reason this register is not simply stored: the
//! architecture has it mask every local vector table entry, and the stored
//! entries are what real hardware is programmed from.

use core::sync::atomic::Ordering;

use crate::registers::{Vlapic, base::Mode, lvt::MASKED};

impl Vlapic {
    /// The spurious-interrupt vector register, which also holds the bit that
    /// software-enables the controller.
    pub(crate) fn spurious(&self) -> u32 {
        self.spurious.load(Ordering::Acquire)
    }

    /// Takes a write to the spurious-interrupt vector register, and says
    /// whether it switched the controller off.
    ///
    /// Software-disabling a controller is a transition and not a flag. The
    /// architecture has it mask every local vector table entry, and it means
    /// the stored entries themselves: a guest that disables its controller
    /// and reads an entry back must see the mask bit set, and a guest that
    /// re-enables it must have to unmask what it wants rather than finding
    /// its old sources live again. Doing it to the stored values is also
    /// what keeps real hardware honest, since that is what every source is
    /// programmed from.
    ///
    /// What is deliberately kept is everything already requested or in service.
    /// A disabled controller stops accepting; it does not retract what it has
    /// already taken.
    pub(crate) fn set_spurious(&self, value: u32) -> bool {
        let was = self.software_enabled();
        self.spurious
            .store(value & SPURIOUS_WRITABLE, Ordering::Release);
        let disabled = was && !self.software_enabled();
        if disabled {
            for entry in &self.lvt {
                entry.fetch_or(MASKED, Ordering::AcqRel);
            }
        }
        disabled
    }

    /// Whether a write of `value` would software-disable this controller.
    ///
    /// Asked before the write, because what it decides is an ordering rather
    /// than a value: the sources and the timer have to stop delivering on real
    /// hardware before the register file records that they have stopped. Only
    /// the processor this controller belongs to writes this register, and that
    /// is the processor asking, so the answer cannot have changed by the
    /// time [`Vlapic::set_spurious`] answers the same question of the same
    /// value.
    pub(crate) fn disabling(&self, value: u32) -> bool {
        self.software_enabled() && value & SOFTWARE_ENABLE == 0
    }

    /// Whether the guest has software-enabled its controller.
    ///
    /// A software-disabled controller holds every local-vector-table entry
    /// masked and refuses to unmask one, and stops accepting anything new —
    /// while keeping whatever is already requested or in service.
    pub(crate) fn software_enabled(&self) -> bool {
        self.spurious() & SOFTWARE_ENABLE != 0
    }

    /// Whether this controller is in a state that accepts interrupts at all.
    ///
    /// Both switches have to be on. A controller whose guest has cleared the
    /// global enable has no controller as far as its guest is concerned, and
    /// one that is merely software-disabled has stopped accepting — in both
    /// cases an interrupt offered to it is one it must refuse rather than
    /// hold.
    pub(crate) fn accepting(&self) -> bool {
        self.mode() != Mode::Disabled && self.software_enabled()
    }
}

/// The bit that makes a controller deliver anything at all.
const SOFTWARE_ENABLE: u32 = 1 << 8;

/// The spurious-interrupt vector register's reset value: every vector bit set
/// and the controller software-disabled.
pub(crate) const SPURIOUS_RESET: u32 = 0xFF;

/// What software may set in the spurious-interrupt vector register.
///
/// The vector and the software-enable bit. Focus-processor checking and
/// end-of-interrupt broadcast suppression are both refused: the first has no
/// meaning on any processor this runs on, and the second is reported
/// unsupported in the version register because the broadcast is performed by
/// hardware this hypervisor passes through.
///
/// Reached by the model-specific-register face as well, which has to fault on
/// exactly the bits this drops.
pub(crate) const SPURIOUS_WRITABLE: u32 = 0x1FF;
