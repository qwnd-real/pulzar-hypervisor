//! Minimal 16550 UART driver backing the serial logger.
//!
//! The driver knows just enough to probe the standard COM ports, configure
//! one for polled transmit-only operation, and push bytes out. The only
//! instance lives inside this crate's global lock, so once [`Uart::detect`]
//! returns, every register access is serialized by that lock.

use core::fmt;

use x86_64::instructions::port::{Port, PortWriteOnly};

/// Standard PC COM-port base addresses, in probe order; the first port that
/// responds is selected.
const COM_BASES: [u16; 4] = [0x3F8, 0x2F8, 0x3E8, 0x2E8];

/// Line speed programmed into the divisor latch; change it here.
const BAUD: u32 = 38_400;

/// The divisor latch reference: the 16550's 1.8432 MHz input clock divided
/// by its fixed 16× oversampling, i.e. the line speed a divisor of 1 yields.
const UART_CLOCK: u32 = 115_200;

/// Divisor latch value for [`BAUD`].
#[expect(
    clippy::cast_possible_truncation,
    reason = "the assert right above the cast proves the divisor fits in u16"
)]
const DIVISOR: u16 = {
    let divisor = UART_CLOCK / BAUD;
    assert!(
        divisor > 0 && divisor <= 0xFFFF,
        "BAUD is out of the divisor latch's range"
    );
    divisor as u16
};

/// Byte written and read back by both detection tests.
const PROBE_PATTERN: u8 = 0xA5;

/// Iteration cap on the transmit poll, so a port that stops draining costs a
/// dropped byte rather than a processor. One byte at [`BAUD`] takes ~260 µs to
/// clock out and a port read takes ~1 µs, so a device that is working is never
/// anywhere near this. It matters most on the paths that report a fault: those
/// run with interrupts masked and must end, whatever the hardware does.
const TX_POLL_LIMIT: u32 = 100_000;

/// Iteration cap on the loopback receive poll, so probing an absent or
/// broken device cannot hang the boot. One byte at [`BAUD`] takes ~260 µs
/// to loop back and a port read takes ~1 µs, so this leaves two orders of
/// magnitude of headroom.
const LOOPBACK_POLL_LIMIT: u32 = 100_000;

/// Divisor latch access bit in the line control register.
const LCR_DLAB: u8 = 0x80;
/// Line control value for 8 data bits, no parity, one stop bit.
const LCR_8N1: u8 = 0x03;

/// FIFO control: enable both FIFOs.
const FCR_ENABLE: u8 = 0x01;
/// FIFO control: reset the receive FIFO.
const FCR_CLEAR_RX: u8 = 0x02;
/// FIFO control: reset the transmit FIFO.
const FCR_CLEAR_TX: u8 = 0x04;
/// FIFO control: interrupt trigger at 14 of 16 bytes. Interrupts are kept
/// disabled, so this only picks a conventional, valid FIFO mode.
const FCR_TRIGGER_14: u8 = 0xC0;

/// Modem control: assert Data Terminal Ready.
const MCR_DTR: u8 = 0x01;
/// Modem control: assert Request To Send.
const MCR_RTS: u8 = 0x02;
/// Modem control: auxiliary output 2, which gates the (unused) IRQ line and
/// is conventionally left set during normal operation.
const MCR_OUT2: u8 = 0x08;
/// Modem control: route the transmitter back into the receiver.
const MCR_LOOPBACK: u8 = 0x10;

/// Line status: a received byte is waiting in the receive buffer.
const LSR_DATA_READY: u8 = 0x01;
/// Line status: the transmit holding register can accept a byte.
const LSR_THR_EMPTY: u8 = 0x20;

/// One 16550-compatible UART, addressed through its I/O port registers.
pub struct Uart {
    /// Base address the registers below were derived from, kept so a port that
    /// is already configured can be addressed again without probing it.
    base: u16,
    /// Transmit/receive buffer; divisor low byte while DLAB is set.
    data: Port<u8>,
    /// Interrupt enable; divisor high byte while DLAB is set.
    interrupt_enable: PortWriteOnly<u8>,
    /// FIFO control (the write-only face of the shared FCR/IIR register).
    fifo_control: PortWriteOnly<u8>,
    /// Line control: word format and the DLAB divisor-latch switch.
    line_control: PortWriteOnly<u8>,
    /// Modem control: DTR/RTS/OUT2 and the loopback test mode.
    modem_control: PortWriteOnly<u8>,
    /// Line status: transmit-empty and data-ready flags.
    line_status: Port<u8>,
    /// Scratch register with no device function; used for detection.
    scratch: Port<u8>,
}

impl Uart {
    /// Probes the standard COM ports in order and returns the first one
    /// that responds, configured and ready to transmit.
    pub fn detect() -> Option<Self> {
        COM_BASES.into_iter().find_map(Self::probe)
    }

    /// Addresses a port that is already configured, without probing or
    /// reprogramming it.
    ///
    /// This is how a second, lock-free view of the selected UART is made: the
    /// registers are stateless addresses, so two views of one port differ in
    /// nothing but who is allowed to write through them.
    pub const fn adopt(base: u16) -> Self {
        Self::new(base)
    }

    /// The base address this UART's registers were derived from.
    pub const fn base(&self) -> u16 {
        self.base
    }

    /// Transmits one byte, waiting for the holding register to drain and
    /// dropping the byte if it never does.
    pub fn write_byte(&mut self, byte: u8) {
        // SAFETY: Polling the line status register and writing the transmit
        // buffer is the UART's polled-transmit protocol; both accesses go to
        // a device `detect` verified present and touch device state only,
        // never memory.
        unsafe {
            for _ in 0..TX_POLL_LIMIT {
                if self.line_status.read() & LSR_THR_EMPTY != 0 {
                    self.data.write(byte);
                    return;
                }
                core::hint::spin_loop();
            }
        }
    }

    fn probe(base: u16) -> Option<Self> {
        let mut uart = Self::new(base);
        // Configuration precedes detection because the loopback test clocks
        // a real byte through the transmitter, which needs a programmed
        // divisor; register writes to an absent port are inert.
        uart.configure();
        (uart.scratch_test() || uart.loopback_test()).then_some(uart)
    }

    const fn new(base: u16) -> Self {
        Self {
            base,
            data: Port::new(base),
            interrupt_enable: PortWriteOnly::new(base + 1),
            fifo_control: PortWriteOnly::new(base + 2),
            line_control: PortWriteOnly::new(base + 3),
            modem_control: PortWriteOnly::new(base + 4),
            line_status: Port::new(base + 5),
            scratch: Port::new(base + 7),
        }
    }

    /// Programs interrupts off, 8N1 at [`BAUD`], FIFOs on, and DTR/RTS
    /// asserted.
    fn configure(&mut self) {
        let [divisor_low, divisor_high] = DIVISOR.to_le_bytes();
        // SAFETY: Every access hits one of this UART's own registers at its
        // architecturally defined offset and alters device state only; if no
        // device decodes the base address the writes are ignored and the
        // subsequent probe fails.
        unsafe {
            self.interrupt_enable.write(0);
            self.line_control.write(LCR_DLAB);
            self.data.write(divisor_low);
            self.interrupt_enable.write(divisor_high);
            self.line_control.write(LCR_8N1);
            self.fifo_control
                .write(FCR_ENABLE | FCR_CLEAR_RX | FCR_CLEAR_TX | FCR_TRIGGER_14);
            self.modem_control.write(MCR_DTR | MCR_RTS | MCR_OUT2);
        }
    }

    /// Checks whether writing the scratch register stores the value; a bus
    /// with no device there floats and fails the read-back.
    fn scratch_test(&mut self) -> bool {
        // SAFETY: The scratch register is a storage byte with no device side
        // effects; reading a vacant port merely returns the floating bus.
        unsafe {
            self.scratch.write(PROBE_PATTERN);
            self.scratch.read() == PROBE_PATTERN
        }
    }

    /// Sends a byte through the UART's internal loopback and checks that it
    /// comes back intact; normal modem-control state is restored afterwards.
    fn loopback_test(&mut self) -> bool {
        // SAFETY: Loopback mode keeps the test byte inside the UART, so
        // nothing reaches the wire; all accesses touch device state only,
        // and a vacant port simply never reports data ready.
        unsafe {
            self.modem_control.write(MCR_LOOPBACK);
            self.data.write(PROBE_PATTERN);
            let mut received = None;
            for _ in 0..LOOPBACK_POLL_LIMIT {
                if self.line_status.read() & LSR_DATA_READY != 0 {
                    received = Some(self.data.read());
                    break;
                }
                core::hint::spin_loop();
            }
            self.modem_control.write(MCR_DTR | MCR_RTS | MCR_OUT2);
            received == Some(PROBE_PATTERN)
        }
    }
}

impl fmt::Write for Uart {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for byte in s.bytes() {
            self.write_byte(byte);
        }
        Ok(())
    }
}
