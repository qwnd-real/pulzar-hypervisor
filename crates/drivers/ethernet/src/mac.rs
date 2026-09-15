//! The address a network interface is known by, and what this driver does
//! to the one the hardware carries.
//!
//! A media access control address is six bytes, and they are not six equal
//! bytes. The first three name the organization that made the interface —
//! a public registry, the same for every card that maker sold — and the
//! last three are the card's own serial within it. A driver may reasonably
//! match on either half: installers pick a vendor, link layers just need
//! the whole thing to be one address, and no part of the system can
//! tolerate the address becoming a multicast one, which the low bit of the
//! first byte declares.
//!
//! So the replacement keeps the shape exactly the way [`spoof`] keeps the
//! shape of a serial number: the organization's three bytes are left alone
//! — they identify a maker of hardware, not one machine — and the serial
//! three are replaced with what the transform makes of them under the
//! machine's seed. The multicast bit travels with the organization's
//! bytes, so a unicast address stays unicast, and a card with no serial at
//! all keeps what it had: an all-zero serial is the encoding of "no
//! identifier here", and inventing one would be a change of shape rather
//! than a change of identity.
//!
//! The result is stable: same card, same seed, same replacement, on every
//! boot and every processor — and two cards of the same maker, whose real
//! addresses differ only in their serials, present replacements that
//! relate to each other no more than the originals did.

use core::fmt::{self, Display, Formatter};

use spoof::Seed;

/// How many bytes a media access control address is.
pub(crate) const BYTES: usize = 6;

/// One interface's media access control address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Mac([u8; BYTES]);

impl Mac {
    /// The address these bytes spell.
    pub(crate) const fn new(bytes: [u8; BYTES]) -> Self {
        Self(bytes)
    }

    /// The bytes of this address.
    pub(crate) const fn bytes(self) -> [u8; BYTES] {
        self.0
    }

    /// This address as the words of a serial EEPROM holds it: three
    /// sixteen-bit words, each a little-endian pair of bytes.
    pub(crate) const fn nvm_words(self) -> [u16; 3] {
        let bytes = self.0;
        [
            bytes[0] as u16 | (bytes[1] as u16) << 8,
            bytes[2] as u16 | (bytes[3] as u16) << 8,
            bytes[4] as u16 | (bytes[5] as u16) << 8,
        ]
    }
}

impl Display for Mac {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        // Colon-separated hexadecimal pairs: the spelling every tool that
        // names an interface's address agrees on, so a replacement is
        // recognizable at a glance beside the original.
        for (index, byte) in self.0.iter().enumerate() {
            if index > 0 {
                formatter.write_str(":")?;
            }
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Replaces the serial half of an address with what the transform makes of
/// it, keeping the organization's half where it was.
#[must_use]
pub(crate) fn spoofed(original: Mac, seed: &Seed) -> Mac {
    let mut replacement = original.bytes();
    let serial = spoof::bytes(&[replacement[3], replacement[4], replacement[5]], seed);
    replacement[3] = serial[0];
    replacement[4] = serial[1];
    replacement[5] = serial[2];
    Mac::new(replacement)
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;

    use spoof::Seed;

    use super::{BYTES, Mac, spoofed};

    /// Two addresses a machine's owner might have: different serials, same
    /// maker.
    const FIRST: Mac = Mac::new([0x52, 0x54, 0x00, 0x12, 0x34, 0x56]);
    const SECOND: Mac = Mac::new([0x52, 0x54, 0x00, 0x98, 0x76, 0x54]);

    /// The seed the rest of this workspace's tests use.
    fn seed() -> Seed {
        Seed::new([0x5e; 16])
    }

    #[test]
    fn keeps_the_organization() {
        let replaced = spoofed(FIRST, &seed()).bytes();
        assert_eq!(replaced[..3], FIRST.bytes()[..3]);
    }

    #[test]
    fn replaces_the_serial() {
        let replaced = spoofed(FIRST, &seed()).bytes();
        assert_ne!(replaced[3..], FIRST.bytes()[3..]);
    }

    #[test]
    fn same_address_same_seed_same_replacement() {
        assert_eq!(spoofed(FIRST, &seed()), spoofed(FIRST, &seed()));
    }

    #[test]
    fn different_addresses_unrelated_replacements() {
        assert_ne!(spoofed(FIRST, &seed()), spoofed(SECOND, &seed()));
    }

    #[test]
    fn all_zero_serial_is_left_alone() {
        let absent = Mac::new([0x52, 0x54, 0x00, 0x00, 0x00, 0x00]);
        assert_eq!(spoofed(absent, &seed()), absent);
    }

    #[test]
    fn unicast_stays_unicast() {
        assert_eq!(spoofed(FIRST, &seed()).bytes()[0] & 1, FIRST.bytes()[0] & 1);
    }

    #[test]
    fn nvm_words_pair_the_bytes() {
        let words = FIRST.nvm_words();
        assert_eq!(words[0].to_le_bytes(), [0x52, 0x54]);
        assert_eq!(words[1].to_le_bytes(), [0x00, 0x12]);
        assert_eq!(words[2].to_le_bytes(), [0x34, 0x56]);
    }

    #[test]
    fn display_reads_as_colon_separated_hexadecimal() {
        assert_eq!(FIRST.to_string().as_str(), "52:54:00:12:34:56");
        assert_eq!(
            Mac::new([0xff; BYTES]).to_string().as_str(),
            "ff:ff:ff:ff:ff:ff",
        );
    }
}
