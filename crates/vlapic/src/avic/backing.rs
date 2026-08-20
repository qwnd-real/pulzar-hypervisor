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
//!
//! One caller needs more than reset: a controller re-activated after its
//! guest has lived a while keeps the registers the architecture preserves,
//! and those are the model's rather than reset's. [`ResetImage::overlay`]
//! replaces slot by slot, so the page rebuilt from a live model is the same
//! statement as the reset one with the model's values in the model's slots.

use apic::REGISTER_STRIDE;
use cpu::ApicId;

use crate::{
    face::table::{PAGE, Register},
    registers::{FLAT_DESTINATION_FORMAT, SPURIOUS_RESET, lvt::Entry, xapic_word},
};

/// The register page as the hardware must find it: the reset state of the
/// register file, one thirty-two-bit register per sixteen-byte slot.
///
/// Every slot [`Register`] names and the reset state does not is zero, which is
/// what the allocator hands out and what those registers read back — the bitmap
/// banks, the priorities, the timer and the interrupt command among them. The
/// extended interrupt-LVT block is the exception and is written here, because
/// the architecture's reset value for it is not zero and it is not one of the
/// registers this crate models; see [`EXTENDED_LVT_FIRST`].
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
        for slot in 0..EXTENDED_LVT_SLOTS {
            image.store(EXTENDED_LVT_FIRST + slot * REGISTER_STRIDE, Entry::RESET);
        }
        image
    }

    /// The page's bytes, in the order the hardware reads them.
    pub(super) fn bytes(&self) -> &[u8; paging::as_usize(PAGE)] {
        &self.0
    }

    /// Replaces the reset value of one slot with a value taken from the live
    /// model.
    ///
    /// What an activation of a controller its guest has already lived with
    /// must show the hardware: the registers the architecture preserves are
    /// the model's, and everything else stays at the reset this image was
    /// built with.
    pub(super) fn overlay(&mut self, register: Register, value: u32) {
        self.put(register, value);
    }

    /// Stores `value` in the slot the architecture assigns to `register`.
    fn put(&mut self, register: Register, value: u32) {
        self.store(register.offset(), value);
    }

    /// Stores `value` in the slot at `offset`, which must be one inside the
    /// page.
    ///
    /// Takes an offset rather than a [`Register`] because one block of the page
    /// this writes is deliberately not a register here: the extended
    /// interrupt-LVT slots have to hold their reset value without becoming
    /// something a guest may reach through the software path.
    fn store(&mut self, offset: u32, value: u32) {
        let offset = offset as usize;
        self.0[offset..offset + size_of::<u32>()].copy_from_slice(&value.to_le_bytes());
    }
}

/// Where the extended interrupt-LVT block starts in the page.
///
/// AMD's extended controller space puts up to [`EXTENDED_LVT_SLOTS`] of these
/// at consecutive slots from here. They are not registers as far as the rest of
/// this crate is concerned — the whole extended space is answered as a reserved
/// address, deliberately, so that nothing in a guest can reach the two
/// registers this hypervisor settles withheld acknowledgements through — but
/// the hardware serves them out of this page when it reads them, which is a
/// path that answer does not sit on. So the page has to hold what they read
/// back on a real controller, and that is the masked entry reset leaves them at
/// rather than the zero the frame arrives as: a guest that probes one would
/// otherwise find an unmasked entry on vector zero.
const EXTENDED_LVT_FIRST: u32 = 0x500;

/// How many extended interrupt-LVT slots the architecture defines.
const EXTENDED_LVT_SLOTS: u32 = 8;

/// Every register the reset state names sits on a sixteen-byte boundary inside
/// the page, which is what lets [`ResetImage::store`] address a slot by its
/// offset alone.
const _: () = assert!(
    PAGE.is_multiple_of(REGISTER_STRIDE as u64),
    "the register page holds a whole number of register slots",
);

/// And the extended block, whose slots are named by offset here rather than by
/// [`Register`], lies wholly inside that page.
const _: () = assert!(
    (EXTENDED_LVT_FIRST + EXTENDED_LVT_SLOTS * REGISTER_STRIDE) as u64 <= PAGE,
    "the extended interrupt-LVT block lies inside the register page",
);

#[cfg(test)]
mod tests {
    //! The image is what the hardware answers a guest's read with, and the
    //! model's reset is what the emulator answers the same read with, so the
    //! two are asserted to agree slot by slot.

    use apic::REGISTER_STRIDE;
    use cpu::ApicId;

    use super::{EXTENDED_LVT_FIRST, EXTENDED_LVT_SLOTS, ResetImage};
    use crate::{
        face::table::Register,
        registers::{FLAT_DESTINATION_FORMAT, SPURIOUS_RESET, lvt::Entry, xapic_word},
    };

    /// The identifier in the slot's own format, and a version word with a
    /// value in both of its fields, so a slot written to the wrong place is a
    /// failure rather than a zero that agrees with zero.
    const ID: ApicId = ApicId::new(4);
    const VERSION: u32 = 0x0050_0010;

    /// The word at `offset`.
    fn word_at(image: &ResetImage, offset: u32) -> u32 {
        let offset = offset as usize;
        u32::from_le_bytes(image.0[offset..offset + 4].try_into().unwrap())
    }

    /// The word at `register`'s slot.
    fn word(image: &ResetImage, register: Register) -> u32 {
        word_at(image, register.offset())
    }

    /// Whether the slot at `offset` is one [`ResetImage::new`] writes.
    fn named(offset: usize) -> bool {
        let first = EXTENDED_LVT_FIRST as usize;
        let extended = first..first + EXTENDED_LVT_SLOTS as usize * REGISTER_STRIDE as usize;
        extended.contains(&offset)
            || [Register::ID, Register::VERSION]
                .into_iter()
                .chain([Register::DESTINATION_FORMAT, Register::SPURIOUS])
                .chain(Entry::ALL.map(Entry::register))
                .any(|register| register.offset() as usize == offset)
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
    fn the_extended_interrupt_entries_are_masked() {
        // The one block of the page whose reset value is not zero and which no
        // `Register` names. The hardware serves these out of the page, so a
        // guest that probes one has to find the masked entry a real controller
        // reports rather than an unmasked one on vector zero.
        let image = ResetImage::new(ID, VERSION);
        for slot in 0..EXTENDED_LVT_SLOTS {
            let offset = EXTENDED_LVT_FIRST + slot * REGISTER_STRIDE;
            assert_eq!(word_at(&image, offset), Entry::RESET, "{offset:#x}");
        }
        // Bounded: the slot below the block and the one above it are not part
        // of it.
        assert_eq!(word_at(&image, EXTENDED_LVT_FIRST - REGISTER_STRIDE), 0);
        assert_eq!(
            word_at(
                &image,
                EXTENDED_LVT_FIRST + EXTENDED_LVT_SLOTS * REGISTER_STRIDE
            ),
            0
        );
    }

    #[test]
    fn every_other_slot_is_zero() {
        let image = ResetImage::new(ID, VERSION);
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
