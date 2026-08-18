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
//!
//! # What is deliberately not written
//!
//! A machine old enough to have these may also have an interrupt mode
//! configuration register, which is how the pre-ACPI world moved the interrupt
//! line running straight from the 8259 to the boot processor over to the APICs.
//! Pulzar does not write it. It is a change to how the platform is wired rather
//! than to what this hypervisor does with it, and it buys nothing here: every
//! input is masked, so nothing asserts down either path, and the local
//! controller's first pin — the other end of that wiring — is masked too unless
//! firmware described it as a non-maskable interrupt.

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

/// Puts both controllers' interrupt masks back to `masks`, the primary's first.
///
/// The other half of [`mask`], for a hypervisor whose guest is the firmware
/// that programmed these: what [`mask`] takes away is the path firmware's own
/// periodic timer arrives on, and firmware configured that path before this
/// hypervisor existed and will never configure it again.
///
/// The order is the opposite of [`mask`]'s and for the same reason. The primary
/// goes first so that the cascade input is open before anything of the
/// secondary's can assert through it, which is the order that cannot leave an
/// interrupt asserting into a controller whose path to the processor is still
/// shut.
pub(crate) fn restore(masks: [u8; 2]) {
    let [primary, secondary] = masks;
    // SAFETY: as `mask`, and each value came from reading the same port it is
    // written back to — so this can only return a controller to a mask it was
    // observed to hold.
    unsafe {
        Port::<u8>::new(PRIMARY_DATA).write(primary);
        Port::<u8>::new(SECONDARY_DATA).write(secondary);
    }
}

/// Both controllers' interrupt masks, the primary's first.
///
/// Reading the data port is how a mask is read back, and it needs no command
/// written first — unlike the in-service and request registers, which are
/// reached by writing a selector and so cannot be asked about without changing
/// what the controller answers next. So nothing here disturbs anything.
///
/// What comes back means nothing on a machine with no legacy controllers.
/// Whether it has them is firmware's to say, in a table read long after this,
/// and a port nothing decodes returns the floating bus rather than an error.
pub(crate) fn masks() -> [u8; 2] {
    let mut primary = Port::<u8>::new(PRIMARY_DATA);
    let mut secondary = Port::<u8>::new(SECONDARY_DATA);
    // SAFETY: both are the architectural data ports of the legacy controllers,
    // and reading one returns the interrupt mask register with no side effect
    // on the controller or on anything it is wired to.
    unsafe { [primary.read(), secondary.read()] }
}
