//! The pair of legacy interrupt controllers, and why they have to be silenced.
//!
//! A PC that still has the two 8259s wires them ahead of everything else, and
//! at reset their sixteen inputs are mapped onto vectors 8 to 15 and 0x70 to
//! 0x77. The first eight of those are exceptions: vector 8 is the double fault,
//! which [`descriptors`] correctly refuses to let anything claim because the
//! architecture gives no way back from it. So a legacy interrupt arriving after
//! this hypervisor unmasks interrupts would be delivered as a fault that cannot
//! be returned from, over state that was perfectly fine.
//!
//! Remapping them somewhere harmless is the other answer and is the wrong one
//! here. Pulzar passes the platform through; it does not own the legacy
//! controllers and does not want their interrupts. Masking every input leaves
//! them configured as firmware left them and simply stops them asserting, which
//! is the smallest thing that makes unmasking safe.
//!
//! Whether the machine has them at all is firmware's to say, in the multiple
//! APIC description table's compatibility flag, so this is not done to a
//! machine that has none: writing to ports nothing answers on is how a machine
//! that never had 8259s gets a configuration it did not have.

use x86_64::instructions::port::Port;

/// Data port of the controller the first eight interrupts arrive through.
const PRIMARY_DATA: u16 = 0x21;

/// Data port of the controller cascaded into the first.
const SECONDARY_DATA: u16 = 0xA1;

/// Every input masked. The data port carries the mask once the controller is
/// initialized, which firmware left it as.
const ALL_MASKED: u8 = 0xFF;

/// Masks every input of both controllers.
pub(crate) fn mask() {
    // SAFETY: both are the architectural data ports of the legacy controllers on
    // a machine whose firmware said it has them, and writing all ones to an
    // initialized 8259's data port sets its interrupt mask and does nothing
    // else. The secondary is masked first so that no input of it can assert
    // through the cascade in the window between the two writes.
    unsafe {
        Port::<u8>::new(SECONDARY_DATA).write(ALL_MASKED);
        Port::<u8>::new(PRIMARY_DATA).write(ALL_MASKED);
    }
}
