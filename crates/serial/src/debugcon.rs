//! QEMU's debug console, which is one I/O port and no line rate at all.
//!
//! The device decodes a single byte-wide port: a write appends that byte to
//! whatever the host attached, and there is no holding register to poll, no
//! divisor to program and no FIFO to drain. That is the whole of the driver,
//! and it is why this backend is preferred wherever it exists — a UART at
//! 38400 baud takes about 260 µs per byte, so a hundred-byte line costs 26 ms
//! of a processor that is supposed to be running a guest, while the same line
//! through this port costs a hundred port writes.
//!
//! It is not real hardware and makes no pretence of being any. Nothing outside
//! a virtual machine decodes the port, which is exactly what
//! [`Debugcon::detect`] establishes before anything is written through it.

use core::fmt;

use x86_64::instructions::port::Port;

/// The port QEMU's debug console decodes.
///
/// Fixed rather than probed: the device is configurable on the host side, but a
/// guest has no way to ask where it was put, and every hypervisor and firmware
/// that uses it uses this number.
const PORT: u16 = 0xE9;

/// What a read of the port answers when the device is there.
///
/// The port is write-only as far as output goes, and the read exists only to
/// say whether anything decodes it: the device answers with its own port
/// number, and a bus with nothing on it floats high instead.
const PRESENT: u8 = 0xE9;

/// The debug console, addressed through its one port.
pub(crate) struct Debugcon(Port<u8>);

impl Debugcon {
    /// Answers with the debug console if this machine has one.
    ///
    /// The test costs one port read and cannot disturb anything: the port
    /// carries no state, and reading it on a machine without the device
    /// returns the floating bus rather than faulting.
    pub(crate) fn detect() -> Option<Self> {
        let mut port = Port::<u8>::new(PORT);
        // SAFETY: a byte-wide read of a port that either belongs to the debug
        // console — which has no read side effects — or is decoded by nothing,
        // in which case the access is answered by the bus and touches nothing.
        let answer = unsafe { port.read() };
        (answer == PRESENT).then_some(Self(port))
    }

    /// Addresses the console again without testing for it.
    ///
    /// The port is stateless, so a second view of it differs from the first in
    /// nothing but who is allowed to write through it. This is what the
    /// lock-free reporting path uses.
    pub(crate) const fn adopt() -> Self {
        Self(Port::new(PORT))
    }

    /// Writes one byte, which is the whole of the protocol.
    pub(crate) fn write_byte(&mut self, byte: u8) {
        // SAFETY: the port was established to be the debug console's, whose
        // only defined write behaviour is appending the byte to the host's
        // output. It touches no memory and cannot block.
        unsafe { self.0.write(byte) };
    }
}

impl fmt::Write for Debugcon {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for byte in s.bytes() {
            self.write_byte(byte);
        }
        Ok(())
    }
}
