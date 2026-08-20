//! One backing page, in the state the hardware must find it in.
//!
//! The processor serves a number of the controller's registers out of the
//! backing page without an exit, and some of them — the identifier and the
//! version — no guest write ever changes. So before the first entry the page
//! must already hold what those registers answer with, and what that is is
//! exactly the register file's reset state: the identifier the controller was
//! built with, the version of the hardware behind it, and the constants the
//! model's own reset stores.
//!
//! The image is built from those same constants rather than captured from a
//! live controller, for two reasons. A live controller is unreachable in a
//! host test, and a captured page would carry whatever its guest had written
//! since reset — the image has to describe reset itself.

use apic::REGISTER_STRIDE;
use cpu::ApicId;

use crate::{
    face::table::{PAGE, Register},
    registers::{FLAT_DESTINATION_FORMAT, SPURIOUS_RESET, lvt::Entry, xapic_word},
};

/// The register page as the hardware must find it: the reset state of the
/// register file, one thirty-two-bit register per sixteen-byte slot.
///
/// Every slot the reset state does not name is zero, which is what the
/// allocator hands out and what those registers read back — the bitmap banks,
/// the priorities, the timer and the interrupt command among them.
pub(super) struct ResetImage([u8; paging::as_usize(PAGE)]);

impl ResetImage {
    /// The image for the processor `id`, reporting `version`.
    pub(super) fn new(id: ApicId, version: u32) -> Self {
        let mut image = Self([0; paging::as_usize(PAGE)]);
        image.put(Register::ID, xapic_word(id));
        image.put(Register::VERSION, version);
        image.put(Register::DESTINATION_FORMAT, FLAT_DESTINATION_FORMAT);
        image.put(Register::SPURIOUS, SPURIOUS_RESET);
        for entry in Entry::ALL {
            image.put(entry.register(), Entry::RESET);
        }
        image
    }

    /// The page's bytes, in the order the hardware reads them.
    pub(super) fn bytes(&self) -> &[u8; paging::as_usize(PAGE)] {
        &self.0
    }

    /// Stores `value` in the slot the architecture assigns to `register`.
    fn put(&mut self, register: Register, value: u32) {
        let offset = register.offset() as usize;
        self.0[offset..offset + size_of::<u32>()].copy_from_slice(&value.to_le_bytes());
    }
}

/// Every register the reset state names sits on a sixteen-byte boundary inside
/// the page, which is what lets [`ResetImage::put`] address a slot by its
/// offset alone.
const _: () = assert!(
    PAGE.is_multiple_of(REGISTER_STRIDE as u64),
    "the register page holds a whole number of register slots",
);

#[cfg(test)]
mod tests {
    //! The image is what the hardware answers a guest's read with, and the
    //! model's reset is what the emulator answers the same read with, so the
    //! two are asserted to agree slot by slot.

    use cpu::ApicId;

    use super::ResetImage;
    use crate::{
        face::table::Register,
        registers::{FLAT_DESTINATION_FORMAT, SPURIOUS_RESET, lvt::Entry, xapic_word},
    };

    /// The identifier in the slot's own format, and a version word with a
    /// value in both of its fields, so a slot written to the wrong place is a
    /// failure rather than a zero that agrees with zero.
    const ID: ApicId = ApicId::new(4);
    const VERSION: u32 = 0x0050_0010;

    /// The word at `register`'s slot.
    fn word(image: &ResetImage, register: Register) -> u32 {
        let offset = register.offset() as usize;
        u32::from_le_bytes(image.0[offset..offset + 4].try_into().unwrap())
    }

    #[test]
    fn the_image_holds_the_reset_state() {
        let image = ResetImage::new(ID, VERSION);
        assert_eq!(word(&image, Register::ID), xapic_word(ID));
        assert_eq!(word(&image, Register::VERSION), VERSION);
        assert_eq!(
            word(&image, Register::DESTINATION_FORMAT),
            FLAT_DESTINATION_FORMAT
        );
        assert_eq!(word(&image, Register::SPURIOUS), SPURIOUS_RESET);
        for entry in Entry::ALL {
            assert_eq!(word(&image, entry.register()), Entry::RESET, "{entry:?}");
        }
    }

    #[test]
    fn every_other_slot_is_zero() {
        let image = ResetImage::new(ID, VERSION);
        let named = |offset: usize| {
            [Register::ID, Register::VERSION]
                .into_iter()
                .chain([Register::DESTINATION_FORMAT, Register::SPURIOUS])
                .chain(Entry::ALL.map(Entry::register))
                .any(|register| (register.offset() as usize) == offset)
        };
        let (slots, rest) = image.0.as_chunks::<16>();
        assert!(rest.is_empty());
        for (slot, chunk) in slots.iter().enumerate() {
            let offset = slot * 16;
            if named(offset) {
                continue;
            }
            assert_eq!(u32::from_le_bytes(chunk[..4].try_into().unwrap()), 0);
        }
    }
}
