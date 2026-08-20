//! One backing page, in the state the hardware must find it in.
//!
//! Two things, and the second is what keeps the first from being the whole
//! answer.
//!
//! [`ResetImage`] is the page a controller coming out of reset needs. The
//! processor serves a number of the controller's registers out of the backing
//! page without an exit, and one of them — the version — no guest write ever
//! changes, so before the first entry the page must already hold what those
//! registers answer with. What that is is exactly the register file's reset
//! state, and the image is built from the same constants the model's own reset
//! stores rather than captured from a live controller: a live controller is
//! unreachable in a host test, and a captured page would carry whatever its
//! guest had written since reset.
//!
//! [`Projection`] is what a controller whose guest has already lived a while
//! holds instead, written over that image slot by slot. An activation is not an
//! architectural event — the guest cannot cause one, is never told of one, and
//! reads the same registers on both sides of it — so what has to survive it is
//! every register the page answers a read with, and not merely the ones a mode
//! change would preserve. A register left at its reset value there is a task
//! priority dropped to zero, an interrupt the guest is servicing that its
//! controller no longer knows about, or a timer told to count down from
//! nothing.
//!
//! # One set of registers, both directions
//!
//! The projection is the way back as well: a deactivation has to leave the
//! model holding what the hardware did while it drove. Both directions walk one
//! set, which is [`Projection::walk`] — a register carried into the page and
//! forgotten on the way out, or the reverse, is a value the guest watches
//! change at a boundary it cannot see, and two lists is how one of them comes
//! to be missing from the other.
//!
//! Which of those registers the *model* then takes back is a narrower question,
//! and [`Projection::into_model`] answers it: a register whose write the
//! hardware exits for is one the model was told about at that exit, so the
//! page's copy is a copy of a value the model already has. What is carried back
//! is what moves in the page with no exit at all.

use core::array::from_fn;

use apic::REGISTER_STRIDE;
use cpu::ApicId;
use descriptors::Vector;

use crate::{
    VlapicError,
    face::table::{PAGE, Register},
    registers::{
        FLAT_DESTINATION_FORMAT, SPURIOUS_RESET, Vlapic, base::Mode, bitmap::SLOTS, icr::Command,
        lvt::Entry, xapic_word,
    },
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

/// Every register a controller's backing page and its software model both hold.
///
/// The state that has to cross a lifecycle boundary, as a value rather than as
/// a sequence of stores: [`Projection::of`] takes one out of the model,
/// [`Projection::overlay`] writes it over a reset image, [`Projection::read`]
/// takes one back out of a page, and [`Projection::into_model`] hands the model
/// what the hardware moved while it drove.
///
/// Which registers those are is [`Projection::walk`]'s to say, once, and both
/// page directions go through it. The version is not among them and belongs to
/// the reset image instead: it describes the hardware behind the controller, so
/// neither the guest nor the face it is reached through can move it.
///
/// [`Default`] is the blank a page read fills in and is not a state any
/// controller is in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Projection {
    /// The identifier, in the shape the face gives it — which is the one
    /// register a face change moves without the guest writing anything.
    identifier: u32,
    task_priority: u32,
    logical_destination: u32,
    destination_format: u32,
    spurious: u32,
    /// What the guest's last write to the error status register latched, which
    /// is the whole of what a read of it answers with.
    errors: u32,
    /// The interrupt command, in the two halves the page holds it in.
    command_low: u32,
    command_high: u32,
    timer_initial: u32,
    timer_divide: u32,
    /// The local vector table, in [`Entry::ALL`]'s order.
    lvt: [u32; Entry::COUNT],
    /// The three banks, a word per slot.
    in_service: [u32; SLOTS],
    trigger_mode: [u32; SLOTS],
    request: [u32; SLOTS],
}

impl Projection {
    /// What the model holds, for a page about to be driven in `mode`.
    ///
    /// The mode is the caller's to fix rather than something read again here,
    /// for the reason [`crate::registers::identity`] gives: two of these
    /// registers are shaped by the face the controller is in, and a page built
    /// from two loads of it would be a page in neither face.
    pub(super) fn of(vlapic: &Vlapic, mode: Mode) -> Self {
        let command = vlapic.command();
        Self {
            identifier: vlapic.id_register(mode),
            task_priority: u32::from(vlapic.task_priority().get()),
            logical_destination: vlapic.logical_destination(mode),
            destination_format: vlapic.destination_format(),
            spurious: vlapic.spurious(),
            errors: vlapic.errors().read(),
            command_low: command.low(),
            command_high: command.high(),
            timer_initial: vlapic.timer_initial(),
            timer_divide: vlapic.timer_divide(),
            lvt: Entry::ALL.map(|entry| {
                if vlapic.model().has(entry) {
                    vlapic.lvt_readback(entry).into_bits()
                } else {
                    // An entry this controller does not have keeps the masked
                    // value reset leaves it at, rather than the zero the model
                    // answers a read of it with: the hardware serves the slot
                    // out of the page whatever the faces say about the
                    // register, so a guest that probes one has to find what a
                    // controller without the entry reports and not an unmasked
                    // entry on vector zero.
                    Entry::RESET
                }
            }),
            in_service: bank(|slot| vlapic.in_service_slot(slot)),
            trigger_mode: bank(|slot| vlapic.trigger_mode_slot(slot)),
            request: bank(|slot| vlapic.request_slot(slot)),
        }
    }

    /// Writes the projection over a reset image, slot by slot.
    ///
    /// Consuming, because the page is what a projection is for: what the
    /// hardware must find when it starts driving a controller its guest has
    /// lived with is this image, and nothing reads the value back afterwards.
    pub(super) fn overlay(mut self, image: &mut ResetImage) {
        self.walk(|offset, word| image.store(offset, *word));
    }

    /// Takes a projection back out of a page, slot by slot.
    ///
    /// `word` answers with the page's value at an offset. A slot that cannot be
    /// reached fails the whole read, and the first failure is the one reported:
    /// every slot of one page is reached the same way, so the walk finishes
    /// rather than being abandoned and what it costs is a few instructions on a
    /// path that is already reporting an error.
    pub(super) fn read(
        mut word: impl FnMut(u32) -> Result<u32, VlapicError>,
    ) -> Result<Self, VlapicError> {
        let mut projection = Self::default();
        let mut failure = None;
        projection.walk(|offset, slot| match word(offset) {
            Ok(value) => *slot = value,
            Err(error) => failure = failure.or(Some(error)),
        });
        failure.map_or(Ok(projection), Err)
    }

    /// Hands the model back what the hardware moved while it drove the
    /// controller.
    ///
    /// Narrower than what the projection carries the other way, and the
    /// difference is the exits. A register whose write the hardware traps, or
    /// whose access the permission map keeps, is one the model was told about
    /// at that exit — so the page's copy is a copy of a value the model
    /// already has, and taking it back would replace a value with itself.
    /// What is carried here is what moves in the page with no exit at all:
    /// the task priority the hardware performs whole, the interrupt command
    /// whose destination half it stores without exiting, and the three
    /// banks it and every other processor's delivery move between them.
    ///
    /// The identifier is in neither set. It is read-only in both faces and the
    /// model's is the real one, so what the page carries is a shape rather than
    /// a value. The logical destination is deliberately left alone for the
    /// mirror-image reason: the wider face derives it from the identifier
    /// instead of storing it, so a page written in that face holds a word the
    /// model must not be given — the older face's own register would come back
    /// holding a cluster mask it never wrote.
    pub(super) fn into_model(self, vlapic: &Vlapic) {
        vlapic.set_task_priority(self.task_priority);
        // The whole register out of the two halves the page keeps it in, and
        // unnarrowed: what the guest wrote is what its next read has to answer
        // with, whichever authority is serving the register by then.
        let _command =
            vlapic.set_command(Command::from_halves(self.command_low, self.command_high).bits());
        let banks = self
            .in_service
            .into_iter()
            .zip(self.trigger_mode)
            .zip(self.request);
        for (slot, ((in_service, trigger_mode), request)) in banks.enumerate() {
            for bit in 0..u32::BITS {
                let mask = 1 << bit;
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "eight slots of thirty-two bits is exactly the vector space"
                )]
                let vector = Vector::new((slot as u32 * u32::BITS + bit) as u8);
                if trigger_mode & mask != 0 {
                    vlapic.force_trigger_mode(vector);
                }
                if in_service & mask != 0 {
                    vlapic.force_in_service(vector);
                }
                // A request the model already holds in service is a level
                // arrival the software path is already tracking: requesting it
                // again would deliver it twice.
                if request & mask != 0 && !vlapic.holds_in_service(vector) {
                    vlapic.force_request(vector);
                }
            }
        }
    }

    /// Hands every slot the page holds on the model's behalf to `visit`, with
    /// the offset the architecture puts it at and the word this projection
    /// keeps it in.
    ///
    /// The one statement of which registers cross a lifecycle boundary, and
    /// both directions are a walk of it: the page is filled by writing each
    /// word out and read by storing each word back, so a register cannot be
    /// carried one way and forgotten the other. Every loss this projection
    /// exists to stop had exactly that shape.
    fn walk(&mut self, mut visit: impl FnMut(u32, &mut u32)) {
        visit(Register::ID.offset(), &mut self.identifier);
        visit(Register::TASK_PRIORITY.offset(), &mut self.task_priority);
        visit(
            Register::LOGICAL_DESTINATION.offset(),
            &mut self.logical_destination,
        );
        visit(
            Register::DESTINATION_FORMAT.offset(),
            &mut self.destination_format,
        );
        visit(Register::SPURIOUS.offset(), &mut self.spurious);
        visit(Register::ERROR_STATUS.offset(), &mut self.errors);
        visit(Register::COMMAND_LOW.offset(), &mut self.command_low);
        visit(Register::COMMAND_HIGH.offset(), &mut self.command_high);
        visit(
            Register::TIMER_INITIAL_COUNT.offset(),
            &mut self.timer_initial,
        );
        visit(Register::TIMER_DIVIDE.offset(), &mut self.timer_divide);
        for (entry, word) in Entry::ALL.into_iter().zip(&mut self.lvt) {
            visit(entry.register().offset(), word);
        }
        for (first, bank) in [
            (Register::IN_SERVICE, &mut self.in_service),
            (Register::TRIGGER_MODE, &mut self.trigger_mode),
            (Register::INTERRUPT_REQUEST, &mut self.request),
        ] {
            let mut offset = first.offset();
            for word in bank {
                visit(offset, word);
                offset += REGISTER_STRIDE;
            }
        }
    }
}

/// One bank's eight words, taken out of the model a slot at a time.
///
/// A slot the register file does not have cannot arrive here — the count of
/// slots is the bitmap's own — and a zero is what a register holding nothing
/// reads as, which is the closest this can come to saying nothing at all.
fn bank(mut slot_of: impl FnMut(usize) -> Option<u32>) -> [u32; SLOTS] {
    from_fn(|slot| slot_of(slot).unwrap_or(0))
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
    //! Two things, matching the two the module holds.
    //!
    //! The image is what the hardware answers a guest's read with, and the
    //! model's reset is what the emulator answers the same read with, so the
    //! two are asserted to agree slot by slot.
    //!
    //! The projection is asserted to be an inverse of itself over the page: the
    //! registers it carries are pinned as a list, and a projection with a
    //! distinct value in every one of them is written into a page-shaped buffer
    //! and read back. A controller cannot be built on a host, so what the model
    //! puts in each of those slots is not reachable from here — the fields are
    //! filled through the walk instead, which is what makes every test below
    //! cover a register added to the projection without naming it again.

    use alloc::vec::Vec;

    use apic::REGISTER_STRIDE;
    use cpu::ApicId;

    use super::{EXTENDED_LVT_FIRST, EXTENDED_LVT_SLOTS, Projection, ResetImage, SLOTS};
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

    /// A projection with a different value in every slot it carries, so that a
    /// slot written to the wrong place, or read back out of the wrong one, is a
    /// failure rather than a value that agrees with whatever was already there.
    fn distinct() -> Projection {
        let mut projection = Projection::default();
        let mut next = 0x1000_0001_u32;
        projection.walk(|_, word| {
            *word = next;
            next = next.wrapping_add(0x0101_0101);
        });
        projection
    }

    /// Every offset the projection carries, in the order it carries them.
    fn offsets() -> Vec<u32> {
        let mut offsets = Vec::new();
        Projection::default().walk(|offset, _| offsets.push(offset));
        offsets
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

    #[test]
    fn the_projection_carries_every_register_the_page_holds_for_the_model() {
        // Written out rather than derived, so that a register dropped from the
        // walk fails here rather than quietly stopping at whatever a reset left
        // in its slot — which is what every defect this projection replaces was.
        // The order is the walk's, because the two are one list.
        let expected: Vec<u32> = [
            Register::ID,
            Register::TASK_PRIORITY,
            Register::LOGICAL_DESTINATION,
            Register::DESTINATION_FORMAT,
            Register::SPURIOUS,
            Register::ERROR_STATUS,
            Register::COMMAND_LOW,
            Register::COMMAND_HIGH,
            Register::TIMER_INITIAL_COUNT,
            Register::TIMER_DIVIDE,
        ]
        .into_iter()
        .chain(Entry::ALL.map(Entry::register))
        .map(Register::offset)
        .chain(
            [
                Register::IN_SERVICE,
                Register::TRIGGER_MODE,
                Register::INTERRUPT_REQUEST,
            ]
            .into_iter()
            .flat_map(|first| {
                (first.offset()..)
                    .step_by(REGISTER_STRIDE as usize)
                    .take(SLOTS)
            }),
        )
        .collect();
        assert_eq!(offsets(), expected);
        // Each bank is eight consecutive slots and stops below the register that
        // follows it: the last slot the projection carries is the top of the
        // request bank, one below the error status.
        assert_eq!(expected.len(), 10 + Entry::COUNT + 3 * SLOTS);
        assert_eq!(
            expected.last().copied(),
            Some(Register::ERROR_STATUS.offset() - REGISTER_STRIDE)
        );
        // And no slot is carried twice, which is the one mistake a walk of
        // thirty-one fields can make that both directions would agree about: the
        // register that lost its slot would read back as the other's.
        let mut once = expected.clone();
        once.sort_unstable();
        once.dedup();
        assert_eq!(once.len(), expected.len());
    }

    #[test]
    fn a_projection_read_back_out_of_a_page_is_the_one_that_was_written() {
        // The whole of what makes the two directions inverses of each other, and
        // the reason the set is one list: a distinct value in every slot,
        // written over a reset image and taken back out of it. A field carried
        // one way and not the other, or written to another field's slot, comes
        // back as something else.
        let projection = distinct();
        let mut image = ResetImage::new(ID, VERSION);
        projection.overlay(&mut image);
        let read = Projection::read(|offset| Ok(word_at(&image, offset)))
            .expect("a page-shaped buffer answers at every offset the projection names");
        assert_eq!(read, projection);
    }

    #[test]
    fn the_projection_leaves_the_slots_it_does_not_carry_alone() {
        // The version is the reset image's, because nothing in a controller's
        // life moves it. The two computed priorities and the timer's remaining
        // count are not the model's to project at all — the first of them is the
        // register the acceleration answers a read of with a fault, and the
        // other two are the hardware's while it drives.
        let mut image = ResetImage::new(ID, VERSION);
        distinct().overlay(&mut image);
        assert_eq!(word(&image, Register::VERSION), VERSION);
        for register in [
            Register::ARBITRATION_PRIORITY,
            Register::PROCESSOR_PRIORITY,
            Register::TIMER_CURRENT_COUNT,
        ] {
            assert_eq!(word(&image, register), 0, "{register:?}");
        }
        // Nor is the extended block the image seeds one the projection reaches.
        for slot in 0..EXTENDED_LVT_SLOTS {
            let offset = EXTENDED_LVT_FIRST + slot * REGISTER_STRIDE;
            assert_eq!(word_at(&image, offset), Entry::RESET, "{offset:#x}");
        }
    }

    #[test]
    fn the_identifier_the_page_holds_is_the_one_the_face_shapes() {
        // Why the identifier is projected rather than left to the image: the
        // image is built in the older face's shape and the wider face uses the
        // whole register, so the two are different numbers for every identifier
        // but zero. The hardware derives the logical destination it matches an
        // interprocessor interrupt against from this slot, which is why a page
        // in the shape of the face before it is a processor the guest can no
        // longer address.
        assert_eq!(xapic_word(ID), 0x0400_0000);
        assert_eq!(ID.get(), 4);
        let mut image = ResetImage::new(ID, VERSION);
        assert_eq!(word(&image, Register::ID), xapic_word(ID));
        let wider = Projection {
            identifier: ID.get(),
            ..Projection::default()
        };
        wider.overlay(&mut image);
        assert_eq!(word(&image, Register::ID), ID.get());
    }
}
