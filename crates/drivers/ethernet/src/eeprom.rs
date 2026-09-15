//! The serial EEPROM a Realtek controller keeps its identity in, served in
//! place of the real one.
//!
//! A 93C46 is 64 or 256 sixteen-bit words behind four wires the driver
//! toggles one write at a time: a chip select, a clock, a data bit going
//! in, and a data bit coming out. The wires live in the same byte as the
//! lock on the controller's configuration registers, so a driver is in
//! this protocol exactly while it holds the select high and leaves the
//! lock alone — and it leaves it the moment it is done, because the only
//! way to end a transaction is to drop the select.
//!
//! This module is that EEPROM as far as the guest ever sees it: the words
//! the controller's own EEPROM holds, with the three that spell the
//! interface's address replaced. It is a state machine over the wires
//! rather than a register a guest can name, which is what makes the
//! replacement survive anything the guest does — a controller reset never
//! reaches the EEPROM (a reset reloads the controller *from* it, and
//! nothing resets the EEPROM itself), and the only state here is the
//! transaction in flight, which begins and ends at the select's edges
//! exactly as the real chip's does.
//!
//! The same wires are driven once more, by [`read`], before the guest
//! runs: the words this serves are the real EEPROM's own with the address
//! replaced, so everything a driver reads — the signature word, the
//! configuration words, the address — is the truth about the hardware
//! except the one thing this driver exists to replace.

use alloc::{boxed::Box, vec::Vec};

use spin::Mutex;

/// The data-out bit, from the EEPROM to the driver.
pub(crate) const OUTPUT: u8 = 0x01;

/// The data-in bit, from the driver to the EEPROM.
const INPUT: u8 = 0x02;

/// The clock the driver raises to move either data bit.
const CLOCK: u8 = 0x04;

/// The chip select, high for the whole of a transaction.
const SELECT: u8 = 0x08;

/// What a driver sets around an EEPROM transaction to take the wires away
/// from their other role, locking nothing.
const ENABLE: u8 = 0x80;

/// The command a read is: a start bit, a two-bit read opcode, and then the
/// address, most significant bit first.
const READ: u32 = 0b0110;

/// How many command bits precede the address: a leading zero, the start
/// bit, and the two-bit opcode. A driver shifts one more bit than its
/// opcode and address alone account for, and that bit is always zero.
const PREFIX: u32 = 4;

/// How many command bits an EEPROM whose addresses are `address_bits` wide
/// latches before it answers.
const fn command_bits(address_bits: u32) -> u32 {
    PREFIX + 1 + address_bits
}

/// The signature word of a 256-word EEPROM, which is how a driver learns
/// how wide its EEPROM's addresses are.
pub(crate) const SIGNATURE: u16 = 0x8129;

/// The serial EEPROM behind a Realtek controller's control byte.
pub(crate) struct Serial {
    /// The words this one serves: the real EEPROM's, with the address
    /// replaced.
    words: Box<[u16]>,
    /// How many bits of a command name a word.
    address_bits: u32,
    /// One transaction's progress. Locked rather than atomic because
    /// nothing on the packet path touches it: a driver walks this protocol
    /// a handful of times per boot, on one processor, holding its own lock
    /// around the whole walk.
    progress: Mutex<Progress>,
}

/// Where one transaction has got to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Progress {
    /// Whether the chip is selected.
    selected: bool,
    /// The clock's level at the last write.
    clock: bool,
    /// How many command bits have been latched.
    latched: u32,
    /// The command itself, first bit latched in its top.
    command: u32,
    /// Whether the command was a read, which is the only command that
    /// answers with data.
    reading: bool,
    /// Which word is being served.
    word: usize,
    /// How many bits of it have been driven.
    served: u32,
    /// The bit the last clock drove.
    output: bool,
}

impl Progress {
    /// No transaction under way.
    const IDLE: Self = Self {
        selected: false,
        clock: false,
        latched: 0,
        command: 0,
        reading: false,
        word: 0,
        served: 0,
        output: false,
    };

    /// A transaction beginning: selected, nothing latched, nothing driven.
    const BEGIN: Self = Self {
        selected: true,
        ..Self::IDLE
    };
}

impl Serial {
    /// An EEPROM serving `words`, whose addresses are `address_bits` wide.
    pub(crate) fn new(words: Vec<u16>, address_bits: u32) -> Self {
        debug_assert!(words.len().is_power_of_two());
        Self {
            words: words.into_boxed_slice(),
            address_bits,
            progress: Mutex::new(Progress::IDLE),
        }
    }

    /// What a write of the control byte does to the EEPROM.
    ///
    /// The whole byte reaches the hardware regardless — the caller forwards
    /// it — so this decides only what the EEPROM itself makes of the wires,
    /// which is the data-out bit the next read carries.
    pub(crate) fn wrote(&self, register: u8) {
        let select = register & SELECT != 0;
        let clock = register & CLOCK != 0;
        let input = register & INPUT != 0;
        let mut progress = self.progress.lock();
        if !select {
            // Dropping the select is the one way a transaction ends, and it
            // ends it completely: the real chip forgets the half-latched
            // command, and the next transaction starts from nothing.
            *progress = Progress::IDLE;
            return;
        }
        if !progress.selected {
            // A transaction begins with the select rising, and the write
            // that raises it counts as no clock even if it sets the clock
            // bit too: no driver does that, and the real chip's select
            // dominates its clock.
            *progress = Progress::BEGIN;
            progress.clock = clock;
            return;
        }
        let rising = clock && !progress.clock;
        progress.clock = clock;
        if !rising {
            return;
        }
        let bits = command_bits(self.address_bits);
        if progress.latched < bits {
            progress.command = progress.command << 1 | u32::from(input);
            progress.latched += 1;
            if progress.latched == bits {
                self.decode(&mut progress);
            }
        } else if progress.reading {
            // The bit a driver reads after this clock is the next bit of
            // the word, most significant first. Sixteen of them exhaust a
            // word; the real chip then serves the next address's, so an
            // over-long read gets the same rotation out of this one.
            let word = self.words[progress.word];
            progress.output = word >> (15 - progress.served.min(15)) & 1 == 1;
            progress.served += 1;
            if progress.served == 16 {
                progress.word = progress.word + 1 & (self.words.len() - 1);
                progress.served = 0;
            }
        }
    }

    /// What the data-out bit is, while a transaction is under way.
    ///
    /// `None` when the chip is not selected, which is when its data-out
    /// pin means nothing and the hardware's own byte is the honest one to
    /// show. While a transaction runs — whatever the command — the answer
    /// is this EEPROM's, never the hardware's, or the words the driver
    /// shifted out would be the real ones.
    pub(crate) fn drives(&self) -> Option<bool> {
        let progress = self.progress.lock();
        progress.selected.then_some(progress.output)
    }

    /// Latches the end of a command and begins answering it.
    fn decode(&self, progress: &mut Progress) {
        let address = progress.command & (1 << self.address_bits) - 1;
        progress.reading = progress.command >> self.address_bits & 0xF == READ;
        progress.word = (address as usize) & (self.words.len() - 1);
        progress.served = 0;
        progress.output = false;
    }
}

/// Reads one word out of the serial EEPROM behind a control byte, by
/// driving the wires the way a driver does.
///
/// A whole transaction: select, shift the command out one bit per raised
/// clock, shift the word in the same way, deselect. The read between every
/// two writes is not decoration — it is what makes each write reach the
/// EEPROM before the next one does, which on real hardware is the timing
/// the chip needs and on anything emulated is the order the model sees.
///
/// # Safety
///
/// `control` must name the one-byte register the wires live in, for the
/// whole of this call; nothing else may drive it concurrently.
pub(crate) unsafe fn read(control: *mut u8, word: u16, address_bits: u32) -> u16 {
    let command = u32::from(word) | READ << address_bits;
    let mut value = 0_u16;
    // SAFETY: the caller vouches for the register, and every access below
    // goes to it at its own width.
    unsafe {
        let settle = |value: u8| {
            control.write_volatile(ENABLE | value);
            control.read_volatile();
        };
        let clock = |value: u8| {
            control.write_volatile(ENABLE | SELECT | value | CLOCK);
            control.read_volatile();
        };
        settle(0);
        settle(SELECT);
        for bit in (0..command_bits(address_bits)).rev() {
            let data = if command >> bit & 1 == 1 { INPUT } else { 0 };
            settle(SELECT | data);
            clock(data);
        }
        settle(SELECT);
        for _ in 0..16 {
            clock(0);
            value = value << 1 | u16::from(control.read_volatile() & OUTPUT != 0);
            settle(SELECT);
        }
        control.write_volatile(0);
        control.read_volatile();
    }
    value
}
#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;

    /// An EEPROM of 64 words whose addresses are six bits wide — the shape
    /// the smaller real ones have.
    fn small(words: [u16; 64]) -> Serial {
        Serial::new(Vec::from(words), 6)
    }

    /// Drives one read transaction at `word` the way a Realtek driver does,
    /// and returns what it shifted in.
    fn drive(serial: &Serial, word: u16, address_bits: u32) -> u16 {
        let command = u32::from(word) | READ << address_bits;
        let bits = 5 + address_bits;
        let write = |value: u8| serial.wrote(0x80 | value);
        write(0);
        write(SELECT);
        for bit in (0..bits).rev() {
            let data = if command >> bit & 1 == 1 { 0x02 } else { 0x00 };
            write(SELECT | data);
            write(SELECT | data | 0x04);
        }
        write(SELECT);
        let mut value = 0_u16;
        for _ in 0..16 {
            write(SELECT | 0x04);
            value = value << 1 | u16::from(serial.drives() == Some(true));
            write(SELECT);
        }
        write(0);
        value
    }

    #[test]
    fn idle_drives_nothing() {
        let serial = small([0xABCD; 64]);
        assert_eq!(serial.drives(), None);
    }

    #[test]
    fn a_read_answers_with_the_word() {
        let mut words = [0x0000; 64];
        words[0x2A] = 0xBEEF;
        words[0x2B] = 0x1234;
        let serial = small(words);
        assert_eq!(drive(&serial, 0x2A, 6), 0xBEEF);
        // Deselected afterwards, and ready for another transaction.
        assert_eq!(serial.drives(), None);
        assert_eq!(drive(&serial, 0x2B, 6), 0x1234);
    }

    #[test]
    fn an_over_long_read_rotates_into_the_next_word() {
        let mut words = [0x0000; 64];
        words[0] = 0x8001;
        words[1] = 0x8000;
        let serial = small(words);
        let command = READ << 6;
        let write = |value: u8| serial.wrote(0x80 | value);
        write(SELECT);
        for bit in (0..11).rev() {
            let data = if command >> bit & 1 == 1 { 0x02 } else { 0x00 };
            write(SELECT | data);
            write(SELECT | data | 0x04);
        }
        write(SELECT);
        let mut value = 0_u16;
        for _ in 0..16 {
            write(SELECT | 0x04);
            value = value << 1 | u16::from(serial.drives() == Some(true));
            write(SELECT);
        }
        assert_eq!(value, 0x8001);
        // The seventeenth bit is the next word's first, and the eighteenth
        // its second: what the real chip serves when a driver over-reads.
        write(SELECT | 0x04);
        assert_eq!(serial.drives(), Some(true));
        write(SELECT);
        write(SELECT | 0x04);
        assert_eq!(serial.drives(), Some(false));
        write(0);
    }

    #[test]
    fn a_wider_address_than_the_chip_has_reads_rotation_not_truth() {
        // A 256-word EEPROM's signature read as though the chip were the
        // 64-word one: a driver probing for the signature gets a rotation
        // of some word rather than either the signature or the word it
        // named, exactly as it would from the real hardware, and falls back
        // to the narrower addressing.
        let mut words = [0x0000; 64];
        words[0] = 0x0001;
        let serial = small(words);
        let probed = drive(&serial, 0, 8);
        assert_ne!(probed, SIGNATURE);
        assert_ne!(probed, 0x0001);
    }

    #[test]
    fn deselecting_forgets_a_half_latched_command() {
        let mut words = [0x0000; 64];
        words[3] = 0x7777;
        let serial = small(words);
        // Half a command: select and one clock.
        serial.wrote(0x80 | SELECT);
        serial.wrote(0x80 | SELECT | 0x04);
        // Deselect, then read word 3 in a fresh transaction.
        serial.wrote(0x00);
        assert_eq!(drive(&serial, 3, 6), 0x7777);
    }

    #[test]
    fn a_command_that_is_not_a_read_drives_zeroes() {
        let serial = small([0xFFFF; 64]);
        // The write command is 5, not 6.
        let command = 5_u32 << 6;
        let write = |value: u8| serial.wrote(0x80 | value);
        write(SELECT);
        for bit in (0..11).rev() {
            let data = if command >> bit & 1 == 1 { 0x02 } else { 0x00 };
            write(SELECT | data);
            write(SELECT | data | 0x04);
        }
        write(SELECT);
        write(SELECT | 0x04);
        // Selected and clocked, but a write: the data-out bit stays low.
        assert_eq!(serial.drives(), Some(false));
        write(0);
    }

    #[test]
    fn the_read_command_carries_the_opcode_above_the_address() {
        // Six in the bits above a six-bit address, the shape every command
        // in these tests is built with.
        assert_eq!(READ, 0b0110);
        assert_eq!(0b0110_000000, READ << 6);
    }
}
