//! Replacing the strings and byte fields that identify a device with ones
//! that identify it just as uniquely and nothing else about it.
//!
//! A serial number is a fingerprint: it names one physical device, and it
//! travels in the identify responses a guest reads off its storage. What
//! replaces it has to keep the properties a driver may be matching on — same
//! length, same shape — and the property a machine's owner needs, stability
//! across boots, while breaking the one it must not keep: the link to the
//! hardware's real identity.
//!
//! # The classes
//!
//! A character never leaves its class. A digit is replaced by a digit, and a
//! zero is never replaced at all — a zero is half of every serial number and
//! would flatten the transform's variety if it moved. A hexadecimal letter is
//! replaced by a hexadecimal letter in the same case, so a field whose format
//! is hex-cased stays hex-cased; a letter outside the hexadecimal alphabet is
//! replaced by one of those, so a serial that is plain text stays plain text;
//! and anything that is not a letter or a digit — the dashes, the spaces a
//! field is padded with — is left exactly where it was. The one exception is
//! the binary transform, where a field of all zeroes is a field saying "no
//! identifier here" and is returned untouched: giving an absent identifier a
//! value would be a lie a guest could trip over.
//!
//! # Determinism
//!
//! The replacement is chosen by a stream of decisions keyed by the seed and
//! varied by the whole input, so the same field under the same seed becomes
//! the same replacement every time it is asked for — on every boot, on every
//! processor — while two fields that differ anywhere are replaced by
//! unrelated values. The same character at different positions is generally
//! replaced by different characters, because the stream has moved on.
//!
//! # Not cryptography
//!
//! The mixing underneath is one round of a well-studied finalizer, chosen
//! because it is short, side-effect free, and thoroughly scrambles its input.
//! It is not a cryptographic primitive and this crate does not pretend it is
//! one: someone holding the seed and the algorithm recovers nothing, because
//! the transform needs no secret to invert by eye — its purpose is
//! unlinkability from the original, which a good scramble delivers, not
//! secrecy of the original, which nothing here provides.

#![no_std]

/// The key every replacement is chosen by.
///
/// Two machines given the same seed present the same relationship between
/// their devices' real and spoofed identities; two machines given different
/// seeds present no relationship at all. What the seed should be is the
/// machine owner's decision, recorded where machine-wide decisions live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Seed([u8; 16]);

impl Seed {
    /// A seed from its sixteen bytes.
    #[must_use]
    pub const fn new(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    /// The word the decision stream starts from: the seed's bytes folded
    /// through the mixer, so every seed byte touches the start.
    fn start(self) -> u64 {
        let mut start = 0_u64;
        let mut byte = 0;
        while byte < self.0.len() {
            start = mix(start ^ u64::from(self.0[byte]));
            byte += 1;
        }
        start
    }
}

/// Replaces an ASCII field with the same field in another identity.
///
/// Every character is replaced within its class as the crate documentation
/// describes: digits by digits, hexadecimal letters by hexadecimal letters of
/// the same case, other letters by other letters of the same case, zero and
/// every non-alphanumeric character by themselves.
#[must_use]
pub fn text<const N: usize>(original: &[u8; N], seed: &Seed) -> [u8; N] {
    let mut stream = Stream::new(seed, original);
    let mut replaced = *original;
    for byte in &mut replaced {
        *byte = substitute(*byte, stream.next());
    }
    replaced
}

/// Replaces a binary field with the same field in another identity.
///
/// Each byte is mixed with its own decision from the stream, so the
/// replacement is as long as the original and unrelated to it byte for byte.
/// A field of all zeroes — the encoding of "no identifier here" — is
/// returned untouched.
#[must_use]
pub fn bytes<const N: usize>(original: &[u8; N], seed: &Seed) -> [u8; N] {
    if original.iter().all(|&byte| byte == 0) {
        return *original;
    }
    let mut stream = Stream::new(seed, original);
    let mut replaced = *original;
    for byte in &mut replaced {
        *byte ^= low_byte(stream.next());
    }
    replaced
}

/// The decisions one replacement is driven by.
struct Stream(u64);

impl Stream {
    /// A stream keyed by the seed and varied by the whole input, so that two
    /// fields differing anywhere are replaced by unrelated values.
    fn new(seed: &Seed, input: &[u8]) -> Self {
        let mut state = seed.start();
        for &byte in input {
            state = mix(state ^ u64::from(byte));
        }
        Self(state)
    }

    /// The next decision.
    fn next(&mut self) -> u64 {
        self.0 = mix(self.0);
        self.0
    }
}

/// Replaces `original` with another member of its character class, chosen by
/// `choice`.
const fn substitute(original: u8, choice: u64) -> u8 {
    match original {
        b'0' => b'0',
        b'1'..=b'9' => b'1' + within(choice, 9),
        b'A'..=b'F' => b'A' + within(choice, 6),
        b'a'..=b'f' => b'a' + within(choice, 6),
        b'G'..=b'Z' => b'G' + within(choice, 20),
        b'g'..=b'z' => b'g' + within(choice, 20),
        other => other,
    }
}

/// `choice` reduced below `size`, as a byte.
#[expect(
    clippy::cast_possible_truncation,
    reason = "a remainder below `size` is below 20, so it always fits a byte"
)]
const fn within(choice: u64, size: u64) -> u8 {
    (choice % size) as u8
}

/// The low eight bits of `value`, as a byte. The mask keeps the cast exact:
/// nothing outside the byte's eight bits survives to be truncated.
const fn low_byte(value: u64) -> u8 {
    (value & 0xff) as u8
}

/// The splitmix64 finalizer: one round of shifts and multiplies that spreads
/// a word over the whole word. Chosen because it is short, constant, and
/// free of tables, not because it is cryptographic.
const fn mix(value: u64) -> u64 {
    let mut mixed = value;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    mixed ^ (mixed >> 31)
}

#[cfg(test)]
mod tests {
    use super::{Seed, bytes, text};

    /// A seed nothing in the tests shares with another.
    const SEED: Seed = Seed::new([
        0x0e, 0x55, 0x99, 0x2c, 0x7a, 0x41, 0xd8, 0x63, 0xb7, 0x10, 0xf4, 0x3d, 0x86, 0x29, 0xc5,
        0x58,
    ]);

    /// Another seed, differing from [`SEED`] everywhere.
    const OTHER: Seed = Seed::new([0x91; 16]);

    #[test]
    fn zero_and_specials_are_fixed() {
        let original = b"0-0 0_0.0\0";
        assert_eq!(&text(original, &SEED), original);
    }

    #[test]
    fn digits_stay_digits() {
        let replaced = text(b"123456789", &SEED);
        for (original, replaced) in b"123456789".iter().zip(replaced.iter()) {
            assert!(
                replaced.is_ascii_digit(),
                "{replaced} replaced the digit {original}"
            );
        }
    }

    #[test]
    fn hex_stays_hex_in_its_case() {
        let replaced = text(b"ABCDEFabcdef", &SEED);
        for (original, replaced) in b"ABCDEFabcdef".iter().zip(replaced.iter()) {
            assert_eq!(
                original.is_ascii_uppercase(),
                replaced.is_ascii_uppercase(),
                "{replaced} replaced {original}"
            );
            assert!(
                replaced.is_ascii_hexdigit(),
                "{replaced} replaced the hex digit {original}"
            );
        }
    }

    #[test]
    fn letters_outside_hex_stay_outside_it() {
        let replaced = text(b"GHIJKLMNOPQRSTUVWXYZ", &SEED);
        for (original, replaced) in b"GHIJKLMNOPQRSTUVWXYZ".iter().zip(replaced.iter()) {
            assert!(
                (b'G'..=b'Z').contains(replaced),
                "{replaced} replaced the letter {original}"
            );
        }
        let replaced = text(b"ghijklmnopqrstuvwxyz", &SEED);
        for (original, replaced) in b"ghijklmnopqrstuvwxyz".iter().zip(replaced.iter()) {
            assert!(
                (b'g'..=b'z').contains(replaced),
                "{replaced} replaced the letter {original}"
            );
        }
    }

    #[test]
    fn a_replacement_actually_replaces() {
        // A class preserved is not a field preserved: with every class this
        // shape holds, some character of it has to move.
        let original = b"S4XPNV0K8123456";
        let replaced = text(original, &SEED);
        assert_ne!(
            &replaced, original,
            "no character of a fully-replaceable serial moved"
        );
    }

    #[test]
    fn length_is_preserved() {
        let replaced = text(b"an-identifier", &SEED);
        assert_eq!(replaced.len(), b"an-identifier".len());
    }

    #[test]
    fn the_same_field_and_seed_agree_with_themselves() {
        assert_eq!(
            text(b"S3X9NX0K123456", &SEED),
            text(b"S3X9NX0K123456", &SEED)
        );
        assert_eq!(
            bytes(&[1, 2, 3, 4, 5, 6, 7, 8], &SEED),
            bytes(&[1, 2, 3, 4, 5, 6, 7, 8], &SEED)
        );
    }

    #[test]
    fn another_seed_makes_another_identity() {
        assert_ne!(
            text(b"S3X9NX0K123456", &SEED),
            text(b"S3X9NX0K123456", &OTHER)
        );
        assert_ne!(
            bytes(&[1, 2, 3, 4, 5, 6, 7, 8], &SEED),
            bytes(&[1, 2, 3, 4, 5, 6, 7, 8], &OTHER)
        );
    }

    #[test]
    fn a_real_serial_actually_changes() {
        let replaced = text(b"S3X9NX0K123456", &SEED);
        assert_ne!(&replaced, b"S3X9NX0K123456");
    }

    #[test]
    fn a_field_of_zeroes_is_untouched() {
        assert_eq!(bytes(&[0; 16], &SEED), [0; 16]);
    }

    #[test]
    fn a_real_identifier_actually_changes() {
        let original = [0x4c, 0x55, 0xa3, 0x71, 0x00, 0x22, 0x19, 0xf0];
        assert_ne!(bytes(&original, &SEED), original);
    }

    #[test]
    fn differing_fields_differ_in_their_replacements() {
        // Two identifiers one byte apart, replaced under the same seed: the
        // whole input varies the stream, so the replacements are unrelated
        // rather than off by the one byte.
        let near = [0x4c, 0x55, 0xa3, 0x71, 0x00, 0x22, 0x19, 0xf0];
        let far = [0x4c, 0x55, 0xa3, 0x71, 0x00, 0x22, 0x19, 0xf1];
        assert_ne!(bytes(&near, &SEED), bytes(&far, &SEED));
    }
}
