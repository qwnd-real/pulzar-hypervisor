//! One backing page, in the state the hardware must find it in.
//!
//! Three things. The second is what keeps the first from being the whole
//! answer, and the third is what keeps the guest's interrupts in one authority
//! while the hardware has the page.
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
//!
//! [`Handover`] is the third, and it is what a guest's interrupts do at every
//! entry the hardware is driving. A vector the model is holding is one the
//! hardware knows nothing about: it is not in the bank the hardware delivers
//! out of, it does not raise the priority the hardware arbitrates with, and the
//! guest's acknowledgement of it would be performed against the page. So it
//! crosses into the page instead, and the model stops holding it — one
//! authority for the three banks for as long as the acceleration is on.
//!
//! # Neither direction may treat the page as bytes
//!
//! A backing page has writers other than the processor performing the
//! transition, and they do not stop for it. Another processor's hardware sets a
//! request bit in this page whenever its guest sends an interrupt here, and
//! this processor's own interrupt handler does the same for an arrival taken in
//! the host window — so a whole-page copy over the frame loses whatever landed
//! while it ran, and a single pass of loads over it misses whatever lands
//! behind the read. Both directions therefore go through one atomic per
//! register slot: [`ResetImage::publish`] stores every slot and reconciles the
//! request bank rather than overwriting it, and [`Projection::take`] takes each
//! bank word with a swap so a set that races it is either included in the value
//! taken or lands after it, never in neither.

use core::{
    array::from_fn,
    sync::atomic::{AtomicU32, Ordering},
};

use apic::REGISTER_STRIDE;
use cpu::ApicId;
use descriptors::Vector;

use crate::{
    VlapicError,
    face::table::{Bank, PAGE, Register},
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
    ///
    /// What provisioning writes into a freshly allocated frame, before anything
    /// on the machine can reach it. A page a guest has already been driven
    /// through is written by [`ResetImage::publish`] instead, because by then
    /// the frame has other writers.
    pub(super) fn bytes(&self) -> &[u8; paging::as_usize(PAGE)] {
        &self.0
    }

    /// Writes the image into a live backing page, one register slot at a time.
    ///
    /// `slot` answers with the page's word at an offset, and the first offset
    /// it cannot reach fails the whole publication — every slot of one page
    /// is reached the same way, so a failure names the frame rather than
    /// the slot.
    ///
    /// Every slot of the page is written, including the ones no register sits
    /// at: which of them a life just ended could have left something in is
    /// not a question this has to answer, and the reset image holds what
    /// each of them reads back.
    ///
    /// The interrupt-request bank is the exception, because it is the one bank
    /// another processor's hardware and this processor's own interrupt handler
    /// write without the model hearing about it. What becomes of a bit they
    /// left there is `life`'s answer: a page whose guest is still the one
    /// the model describes keeps it, so the bank becomes the union of the
    /// two and nothing is lost; a page whose guest has stopped existing
    /// keeps nothing, and every bit the model does not have is handed to
    /// `displaced` rather than dropped silently.
    pub(super) fn publish<'page>(
        &self,
        life: Life,
        mut slot: impl FnMut(u32) -> Result<&'page AtomicU32, VlapicError>,
        mut displaced: impl FnMut(Vector),
    ) -> Result<(), VlapicError> {
        for (offset, word) in self.slots() {
            let page = slot(offset)?;
            match (requests(offset), life) {
                // Every register but the request bank has one writer while the
                // guest is stopped, and it is this processor performing the
                // transition.
                (None, _) => page.store(word, Ordering::Release),
                (Some(_), Life::Same) => {
                    page.fetch_or(word, Ordering::AcqRel);
                }
                (Some(bank_slot), Life::Ended) => {
                    let mut lost = page.swap(word, Ordering::AcqRel) & !word;
                    while lost != 0 {
                        let bit = lost.trailing_zeros();
                        lost &= !(1 << bit);
                        displaced(vector_at(bank_slot, bit));
                    }
                }
            }
        }
        Ok(())
    }

    /// Every register slot of the image, as the offset the architecture puts it
    /// at and the word the image holds there.
    fn slots(&self) -> impl Iterator<Item = (u32, u32)> {
        let (words, _) = self.0.as_chunks::<{ size_of::<u32>() }>();
        (0..)
            .step_by(REGISTER_STRIDE as usize)
            .zip(words.iter().step_by(SLOT_WORDS))
            .map(|(offset, word)| (offset, u32::from_le_bytes(*word)))
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

/// Whose state a backing page holds, at the moment it is about to be rebuilt or
/// carried back.
///
/// The reset count answers it, and the answer is what every lifecycle boundary
/// turns on: a controller whose model has been reset since its page was built
/// is a page describing a guest that has stopped existing, and the architecture
/// requires that reset to clear exactly the state such a page still holds — the
/// task priority, the interrupts the guest was servicing, and the requests it
/// never took.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Life {
    /// The same guest's. Whatever the page gained since the model was last
    /// taken out of it is that guest's own.
    Same,
    /// A guest that has stopped existing. Nothing the page holds belongs to the
    /// life that follows it.
    Ended,
}

impl Life {
    /// Whether what the page holds may be carried into the model.
    ///
    /// Only where the model is the one the page was built from. Carrying a
    /// deactivation's page into a model a reset has just cleared would put back
    /// exactly what the reset removed: a task priority the new guest never set,
    /// an in-service bit nothing will ever acknowledge — which imposes a
    /// processor-priority floor for as long as that guest lives — and a request
    /// belonging to an operating system that has stopped existing.
    pub(super) const fn carried(self) -> bool {
        matches!(self, Self::Same)
    }
}

/// Which slot of the interrupt-request bank a page offset names, or nothing for
/// an offset in any other register.
///
/// Asked of the register table rather than computed here, because where the
/// three banks are is that table's one statement of it: a second one would be
/// how a rebuild comes to treat a bank word as an ordinary register.
fn requests(offset: u32) -> Option<usize> {
    match bank_at(offset) {
        Some((Bank::InterruptRequest, slot)) => Some(slot),
        Some((Bank::InService | Bank::TriggerMode, _)) | None => None,
    }
}

/// Which of the three banks a page offset is a slot of, and which slot, if it
/// is in one at all.
fn bank_at(offset: u32) -> Option<(Bank, usize)> {
    Register::at(u64::from(offset))?.bank()
}

/// The vector a bank slot's bit stands for.
///
/// Eight slots of thirty-two bits is exactly the vector space, so a slot inside
/// a bank and a bit inside a word name nothing outside it.
#[expect(
    clippy::cast_possible_truncation,
    reason = "eight slots of thirty-two bits is exactly the vector space"
)]
const fn vector_at(slot: usize, bit: u32) -> Vector {
    Vector::new((slot as u32 * u32::BITS + bit) as u8)
}

/// How many thirty-two-bit words one register slot of the page is long.
const SLOT_WORDS: usize = REGISTER_STRIDE as usize / size_of::<u32>();

/// Every register a controller's backing page and its software model both hold.
///
/// The state that has to cross a lifecycle boundary, as a value rather than as
/// a sequence of stores: [`Projection::of`] takes one out of the model,
/// [`Projection::overlay`] writes it over a reset image, [`Projection::take`]
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

    /// Takes a projection out of a page, emptying the three banks as it goes.
    ///
    /// `slot` answers with the page's word at an offset. A slot that cannot be
    /// reached fails the whole read, and the first failure is the one reported:
    /// every slot of one page is reached the same way, so the walk finishes
    /// rather than being abandoned and what it costs is a few instructions on a
    /// path that is already reporting an error. Nothing is taken out of a page
    /// the walk could not reach either, because the failure is the frame's and
    /// the first slot meets it.
    ///
    /// The three banks are *taken* rather than read, with one swap per word,
    /// and that is what makes this safe to run against a page another
    /// processor is still writing: a peer whose own control block has the
    /// acceleration armed goes on setting request bits here until its next
    /// entry, and a set that races the swap is either included in the value
    /// taken or lands in a word this has already emptied. It is never in
    /// neither. Emptying them is also what leaves the page holding nothing
    /// of this life for the activation that next rebuilds it to have to
    /// reconcile.
    pub(super) fn take<'page>(
        mut slot: impl FnMut(u32) -> Result<&'page AtomicU32, VlapicError>,
    ) -> Result<Self, VlapicError> {
        let mut projection = Self::default();
        let mut failure = None;
        projection.walk(|offset, word| match slot(offset) {
            Ok(page) => {
                *word = if bank_at(offset).is_some() {
                    page.swap(0, Ordering::AcqRel)
                } else {
                    page.load(Ordering::Acquire)
                };
            }
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
    ///
    /// # Two of the three banks are added to what the model holds; the third is taken
    ///
    /// The requests and the trigger modes are added, and that is deliberate
    /// rather than an oversight. An interrupt that arrived on the software path
    /// between the exit which ended the acceleration and the entry that runs
    /// this is in the model alone — nothing put it in the page, because by
    /// then nothing was driving the page — so a carry-back that replaced
    /// those banks would delete it.
    ///
    /// What makes that union safe rather than merely convenient is that the
    /// other side of it holds nothing stale by the time this runs.
    /// [`Projection::take`] empties the page's banks as it takes them, so
    /// no bit is carried twice; and a page whose guest has stopped existing
    /// is not carried at all, so the union can never be of two lives.
    ///
    /// The in-service bank is taken in place of the model's, because while the
    /// hardware drove it was the only authority for it: the guest acknowledges
    /// an interrupt against the page with no exit at all, and nothing puts a
    /// bit in the model's bank while the requests are the hardware's —
    /// [`Handover`] is what makes the second half of that true. Adding would
    /// keep every bit the guest acknowledged in that window, and each of those
    /// floors the controller's processor priority for as long as its guest
    /// lives. [`Vlapic::carry_back`] is where the asymmetry is argued in the
    /// model's own terms.
    ///
    /// A vector that comes back both requested and in service crosses as both.
    /// That is an ordinary state of a controller — a further arrival latched
    /// while the guest is still handling the last one — and both bits are what
    /// the hardware was holding, so nothing here may choose between them.
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
            vlapic.carry_back(slot, in_service, trigger_mode, request);
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

/// One bank slot's worth of what the model hands to the hardware at an entry it
/// is driving.
///
/// The three banks are the page's alone while the acceleration is on, so
/// anything the model is still holding has to cross into it before the guest
/// runs. What crosses is a value rather than a sequence of stores for the same
/// reason [`Projection`] is: which bits go and which stay is the decision, and
/// a decision of words can be read against the states it covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Handover {
    /// The requests the hardware may take over.
    requests: u32,
    /// The trigger modes of exactly those requests.
    trigger_modes: u32,
    /// What the guest is already servicing, which the hardware arbitrates
    /// against and retires without an exit.
    in_service: u32,
}

impl Handover {
    /// What the model is holding at bank slot `slot` that the hardware may take
    /// over.
    ///
    /// A slot the register file does not have cannot arrive here — the count of
    /// slots is the bitmap's own — and a zero is what a register holding
    /// nothing reads as, exactly as it is for [`Projection::of`].
    pub(super) fn of(vlapic: &Vlapic, slot: usize) -> Self {
        Self::from_words(
            vlapic.request_slot(slot).unwrap_or(0),
            vlapic.trigger_mode_slot(slot).unwrap_or(0),
            vlapic.in_service_slot(slot).unwrap_or(0),
            vlapic.external_slot(slot).unwrap_or(0),
        )
    }

    /// What one bank slot hands over, out of the four words the register file
    /// holds for it.
    ///
    /// Every request but one that came in through the pin that bypasses the
    /// controller. Such an arrival is acknowledged to a legacy controller the
    /// guest reaches directly, which is why this controller deliberately does
    /// not hold it in service — and the hardware holds everything it
    /// delivers, so handing one over would leave a bit in the page that
    /// nothing ever clears and a priority class blocked for as long as the
    /// guest lives. Those stay in the model, which injects them.
    ///
    /// The trigger modes are narrowed to the requests that cross. That bank is
    /// what the hardware reads to decide whether an acknowledgement raises an
    /// exit, and a bit for a vector this hand-over is not giving it is not this
    /// hand-over's to publish.
    ///
    /// The in-service bank crosses whole, because the hardware owns it outright
    /// while it drives: it computes the priority it arbitrates with out of that
    /// bank, and retires a bit there when the guest acknowledges.
    ///
    /// Pure in the words rather than taking a controller, so that it can be
    /// read against the states it covers — a controller cannot be built in
    /// a test.
    const fn from_words(requests: u32, trigger_modes: u32, in_service: u32, external: u32) -> Self {
        let requests = requests & !external;
        Self {
            requests,
            trigger_modes: trigger_modes & requests,
            in_service,
        }
    }

    /// Whether nothing crosses, which is every entry of a guest whose
    /// interrupts are the hardware's already.
    pub(super) const fn is_empty(self) -> bool {
        (self.requests | self.in_service) == 0
    }

    /// Publishes what crosses into a live backing page.
    ///
    /// `within` is how far into each bank this slot sits, which is the same
    /// step [`Projection::walk`] takes over one — the three banks are eight
    /// consecutive slots apiece and the slot is at the same distance into each.
    /// `word` answers with the page's word at an offset.
    ///
    /// The request bit is published last, and that ordering is the one
    /// [`Vlapic::accept`] states for the model's own banks: it is the bit that
    /// makes the hardware act, so everything the hardware classifies the
    /// interrupt by — the trigger mode that decides whether its acknowledgement
    /// raises an exit, the in-service state its priority is computed from — has
    /// to be there before it. Each store is Release for the same reason.
    ///
    /// Every one is an OR rather than a store: the page is what the hardware
    /// has been delivering out of, and a peer's hardware may be setting a
    /// request bit in it at this moment.
    ///
    /// # Errors
    ///
    /// As [`crate::read_msr`], from the first slot of the page that cannot be
    /// reached — which names the frame rather than the slot.
    pub(super) fn publish<'page>(
        self,
        within: u32,
        mut word: impl FnMut(u32) -> Result<&'page AtomicU32, VlapicError>,
    ) -> Result<(), VlapicError> {
        word(Register::TRIGGER_MODE.offset() + within)?
            .fetch_or(self.trigger_modes, Ordering::Release);
        word(Register::IN_SERVICE.offset() + within)?.fetch_or(self.in_service, Ordering::Release);
        word(Register::INTERRUPT_REQUEST.offset() + within)?
            .fetch_or(self.requests, Ordering::Release);
        Ok(())
    }

    /// Stops the model holding what the page has just been given.
    ///
    /// After the publish and never before it, for the reason
    /// [`Vlapic::handed_over`] states.
    pub(super) fn retire(self, vlapic: &Vlapic, slot: usize) {
        vlapic.handed_over(slot, self.requests, self.in_service);
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
    //! Four things, and the first three are the three the module holds.
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
    //!
    //! The hand-over is asserted as the decision it is — which of a bank slot's
    //! bits cross to the hardware and which the model keeps holding — and as
    //! the order the ones that cross are published in, which the closure
    //! that answers with a page slot is what observes.
    //!
    //! And both page directions are asserted to lose nothing to a writer they
    //! cannot exclude. A page here is an array of atomics, which is exactly
    //! what the two directions reach a real one as, so a test can be the
    //! concurrent writer: the closure that answers with a slot is where a
    //! peer's hardware ORs a request bit in, mid-walk, and what must not
    //! happen is a bit that ends up in neither the page nor the answer.

    use alloc::vec::Vec;
    use core::{array::from_fn, sync::atomic::AtomicU32};

    use apic::REGISTER_STRIDE;
    use cpu::ApicId;

    use super::{
        EXTENDED_LVT_FIRST, EXTENDED_LVT_SLOTS, Handover, Life, Ordering, PAGE, Projection,
        ResetImage, SLOTS, VlapicError,
    };
    use crate::{
        face::table::{AvicAccess, Register},
        registers::{FLAT_DESTINATION_FORMAT, SPURIOUS_RESET, lvt::Entry, xapic_word},
    };

    /// The identifier in the slot's own format, and a version word with a
    /// value in both of its fields, so a slot written to the wrong place is a
    /// failure rather than a zero that agrees with zero.
    const ID: ApicId = ApicId::new(4);
    const VERSION: u32 = 0x0050_0010;

    /// How many register slots one page holds.
    const SLOT_COUNT: usize = paging::as_usize(PAGE) / REGISTER_STRIDE as usize;

    /// A backing page, as the words both directions move through it.
    ///
    /// One atomic per register slot rather than per word of the page: neither
    /// direction reaches the twelve bytes the architecture leaves undefined
    /// above each register.
    type Page = [AtomicU32; SLOT_COUNT];

    /// A page with `word` in every slot.
    fn filled(word: u32) -> Page {
        from_fn(|_| AtomicU32::new(word))
    }

    /// The slot a page offset names.
    fn slot(page: &Page, offset: u32) -> Result<&AtomicU32, VlapicError> {
        page.get(offset as usize / REGISTER_STRIDE as usize)
            .ok_or(VlapicError::NoLapic)
    }

    /// The word the page holds at `offset`.
    fn holds(page: &Page, offset: u32) -> u32 {
        slot(page, offset)
            .expect("an offset inside the page names a slot")
            .load(Ordering::Relaxed)
    }

    /// The offset of one slot of a vector bank.
    fn bank_word(bank: Register, slot: u32) -> u32 {
        bank.offset() + slot * REGISTER_STRIDE
    }

    /// The offset of one slot of the interrupt-request bank.
    fn request_slot(slot: u32) -> u32 {
        bank_word(Register::INTERRUPT_REQUEST, slot)
    }

    /// The word at `offset`.
    fn word_at(image: &ResetImage, offset: u32) -> u32 {
        let offset = offset as usize;
        u32::from_le_bytes(image.0[offset..offset + 4].try_into().unwrap())
    }

    /// The word at `register`'s slot.
    fn word(image: &ResetImage, register: Register) -> u32 {
        word_at(image, register.offset())
    }

    /// The image a controller whose guest has lived a while is published from.
    fn image() -> ResetImage {
        let mut image = ResetImage::new(ID, VERSION);
        distinct().overlay(&mut image);
        image
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
    fn the_projection_carries_every_register_a_trapped_write_hands_back() {
        // What keeps the trap's readback and the rebuild from becoming two
        // statements of what a slot holds. A register whose write the hardware
        // completes into the page and then exits for is one the page is given the
        // model's answer for at that exit, so a rebuild has to put the same
        // answer there — a register the projection does not carry would be one
        // the guest watches change at a boundary it cannot see.
        //
        // Two of the trap set are left out, and both are registers the model
        // holds no value for: the acknowledgement, which is write-only, and the
        // remote read, which every read of answers zero — which is what a rebuilt
        // page holds in that slot, asserted below rather than assumed.
        let carried = offsets();
        for offset in (0..PAGE).step_by(REGISTER_STRIDE as usize) {
            let Some(register) = Register::at(offset) else {
                continue;
            };
            if register.avic_access(true) != AvicAccess::Trap
                || matches!(register, Register::END_OF_INTERRUPT | Register::REMOTE_READ)
            {
                continue;
            }
            assert!(
                carried.contains(&register.offset()),
                "{register:?} traps and the projection does not carry it"
            );
        }
        assert_eq!(word(&image(), Register::REMOTE_READ), 0);
    }

    #[test]
    fn a_projection_read_back_out_of_a_page_is_the_one_that_was_written() {
        // The whole of what makes the two directions inverses of each other, and
        // the reason the set is one list: a distinct value in every slot,
        // published into a page and taken back out of it. A field carried one way
        // and not the other, or written to another field's slot, comes back as
        // something else.
        let projection = distinct();
        let page = filled(0);
        image()
            .publish(Life::Ended, |offset| slot(&page, offset), |_| ())
            .expect("a page-shaped buffer answers at every offset");
        let taken = Projection::take(|offset| slot(&page, offset))
            .expect("a page-shaped buffer answers at every offset");
        assert_eq!(taken, projection);
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

    #[test]
    fn publishing_an_image_reaches_every_slot_of_the_page() {
        // What the whole-page copy this replaces gave for nothing: a page whose
        // previous life left something in a slot neither the image nor the
        // projection names — the processor priority the hardware recomputes above
        // all — is a page a guest reads that value back out of. So every slot is
        // written, and the sentinel is what a slot this misses would still hold.
        let page = filled(0xDEAD_BEEF);
        image()
            .publish(Life::Ended, |offset| slot(&page, offset), |_| ())
            .expect("a page-shaped buffer answers at every offset");
        for offset in (0..).step_by(REGISTER_STRIDE as usize).take(SLOT_COUNT) {
            assert_eq!(
                holds(&page, offset),
                word_at(&image(), offset),
                "{offset:#x}"
            );
        }
    }

    #[test]
    fn a_request_the_page_gained_survives_a_rebuild_of_the_same_life() {
        // The interleaving the byte copy lost: a peer's hardware, or this
        // processor's own interrupt handler in the host window, sets a request bit
        // after the image has been composed and before the page is written. The
        // guest it belongs to is the one the page is being rebuilt for, so the
        // bank becomes the union of the two and the hardware delivers it.
        let page = filled(0);
        let image = image();
        let arrival = 1 << 5;
        slot(&page, request_slot(2))
            .expect("a bank slot is inside the page")
            .fetch_or(arrival, Ordering::Relaxed);
        let mut displaced = Vec::new();
        image
            .publish(
                Life::Same,
                |offset| slot(&page, offset),
                |vector| displaced.push(vector),
            )
            .expect("a page-shaped buffer answers at every offset");
        assert_eq!(
            holds(&page, request_slot(2)),
            word_at(&image, request_slot(2)) | arrival
        );
        assert!(
            displaced.is_empty(),
            "nothing was displaced, so nothing is owed an answer"
        );
    }

    #[test]
    fn a_request_bit_lands_in_a_slot_the_walk_has_not_reached_yet() {
        // The same interleaving, one step tighter: the arrival lands *during* the
        // publication, in a slot it has yet to write. A store would erase it; the
        // union cannot, whichever side of the walk it falls on.
        let page = filled(0);
        let image = image();
        let arrival = 1 << 9;
        let mut arrived = false;
        image
            .publish(
                Life::Same,
                |offset| {
                    if !arrived && offset == request_slot(0) {
                        arrived = true;
                        slot(&page, request_slot(7))?.fetch_or(arrival, Ordering::Relaxed);
                    }
                    slot(&page, offset)
                },
                |_| (),
            )
            .expect("a page-shaped buffer answers at every offset");
        assert!(arrived, "the walk reached the request bank");
        assert_eq!(
            holds(&page, request_slot(7)) & arrival,
            arrival,
            "a request that raced the walk is still in the page"
        );
    }

    #[test]
    fn a_rebuild_for_a_life_that_ended_reports_every_request_it_deletes() {
        // The invariant a deleted request bit would otherwise break: real hardware
        // may be holding that vector in service for the guest, and the debt for it
        // is discharged by the guest acknowledging the interrupt this bit is the
        // only record of. So no bit may vanish without being handed back — and the
        // ones the model itself carries are not deletions at all.
        let page = filled(0);
        let image = image();
        for bank in 0..u32::try_from(SLOTS).expect("eight slots") {
            slot(&page, request_slot(bank))
                .expect("a bank slot is inside the page")
                .store(!0, Ordering::Relaxed);
        }
        let mut displaced = Vec::new();
        image
            .publish(
                Life::Ended,
                |offset| slot(&page, offset),
                |vector| displaced.push(vector),
            )
            .expect("a page-shaped buffer answers at every offset");
        for bank in 0..u32::try_from(SLOTS).expect("eight slots") {
            let carried = word_at(&image, request_slot(bank));
            assert_eq!(
                holds(&page, request_slot(bank)),
                carried,
                "the page holds what the model holds and nothing else"
            );
            for bit in 0..u32::BITS {
                let vector = super::vector_at(usize::try_from(bank).expect("eight slots"), bit);
                assert_eq!(
                    displaced.contains(&vector),
                    carried & (1 << bit) == 0,
                    "{vector} was set in the page"
                );
            }
        }
    }

    #[test]
    fn a_harvest_takes_the_banks_and_loses_nothing_that_lands_behind_it() {
        // A single pass of loads misses a request bit set behind the read, which
        // is then in neither the model nor a page anything will consult again. The
        // swap makes both outcomes safe: the bit is in the value taken, or it is
        // in the word the swap has already emptied — and the assertion is that it
        // is in one of them.
        let page = filled(0);
        let arrival = 1 << 11;
        let mut arrived = false;
        let taken = Projection::take(|offset| {
            if !arrived && offset == request_slot(0) {
                arrived = true;
                slot(&page, request_slot(4))?.fetch_or(arrival, Ordering::Relaxed);
            }
            slot(&page, offset)
        })
        .expect("a page-shaped buffer answers at every offset");
        assert!(arrived, "the walk reached the request bank");
        assert_eq!(
            taken.request[4], arrival,
            "the bit was taken rather than read past"
        );
        // And the banks are left empty, which is what makes the activation that
        // next rebuilds this page able to add the model's bits to what it finds
        // rather than having to tell two lives apart.
        for bank in 0..u32::try_from(SLOTS).expect("eight slots") {
            for first in [
                Register::IN_SERVICE,
                Register::TRIGGER_MODE,
                Register::INTERRUPT_REQUEST,
            ] {
                let offset = first.offset() + bank * REGISTER_STRIDE;
                assert_eq!(holds(&page, offset), 0, "{offset:#x}");
            }
        }
    }

    #[test]
    fn a_harvest_leaves_the_registers_that_are_not_banks_where_they_are() {
        // Only the banks are taken. A register the page holds a copy of is a
        // register the model may be asked for again — the task priority above all,
        // which the steady state carries back on every entry — and emptying its
        // slot would answer a guest's read out of a zero.
        let page = filled(0);
        image()
            .publish(Life::Ended, |offset| slot(&page, offset), |_| ())
            .expect("a page-shaped buffer answers at every offset");
        let before = holds(&page, Register::TASK_PRIORITY.offset());
        let taken = Projection::take(|offset| slot(&page, offset))
            .expect("a page-shaped buffer answers at every offset");
        assert_eq!(taken.task_priority, before);
        assert_eq!(holds(&page, Register::TASK_PRIORITY.offset()), before);
        assert_eq!(holds(&page, Register::SPURIOUS.offset()), taken.spurious);
    }

    #[test]
    fn only_the_life_that_continues_is_carried_into_the_model() {
        // The decision a deactivation makes, as a value: the page belongs to the
        // model it was built from, and a reset under an active controller replaces
        // that model with one the architecture has just required to be empty.
        assert!(Life::Same.carried());
        assert!(!Life::Ended.carried());
    }

    #[test]
    fn a_hand_over_leaves_a_pin_arrival_where_it_is() {
        // The one request that must not cross. The guest acknowledges an arrival
        // that came in through the pin to a legacy controller it reaches
        // directly, and the hardware holds in service everything it delivers — so
        // handing one over would leave a bit in the page that nothing ever clears
        // and a priority class blocked for as long as the guest lives.
        //
        // Asserted whole rather than field by field, because a word written into
        // another field is exactly the mistake three words of one slot can make.
        assert_eq!(
            Handover::from_words(0b1111, 0b1010, 1 << 8, 0b1000),
            Handover {
                requests: 0b0111,
                // The trigger modes narrow to the requests that cross: the level
                // record of a vector this is not giving the hardware is not this
                // hand-over's to publish into the bank that decides whether an
                // acknowledgement raises an exit.
                trigger_modes: 0b0010,
                // What the guest is already servicing crosses whole, because the
                // hardware arbitrates against that bank and retires a bit in it
                // without an exit.
                in_service: 1 << 8,
            }
        );
        assert!(!Handover::from_words(0b1111, 0b1010, 1 << 8, 0b1000).is_empty());
    }

    #[test]
    fn a_slot_holding_nothing_the_hardware_can_take_hands_over_nothing() {
        // The steady state, and the same answer by a second route: a slot with
        // nothing in it, and one holding only arrivals the model has to keep.
        assert!(Handover::from_words(0, 0, 0, 0).is_empty());
        assert!(Handover::from_words(0b0110, 0b0110, 0, 0b0110).is_empty());
        // An in-service bit alone is not nothing: the priority the hardware
        // arbitrates with is computed out of that bank.
        assert!(!Handover::from_words(0, 0, 1, 0).is_empty());
    }

    #[test]
    fn a_hand_over_publishes_the_request_last_and_into_the_slot_it_came_from() {
        // The order is the one `Vlapic::accept` states for the model's own banks:
        // the request bit is what makes the hardware act, so the trigger mode
        // that decides whether its acknowledgement raises an exit and the
        // in-service state its priority is computed from are both there first.
        // The closure that answers with a page word is what can see it.
        let page = filled(0);
        let mut reached = Vec::new();
        Handover::from_words(1 << 5, 1 << 5, 1 << 7, 0)
            .publish(3 * REGISTER_STRIDE, |offset| {
                reached.push(offset);
                slot(&page, offset)
            })
            .expect("a page-shaped buffer answers at every offset");
        assert_eq!(
            reached,
            [
                bank_word(Register::TRIGGER_MODE, 3),
                bank_word(Register::IN_SERVICE, 3),
                bank_word(Register::INTERRUPT_REQUEST, 3),
            ]
        );
        assert_eq!(holds(&page, bank_word(Register::TRIGGER_MODE, 3)), 1 << 5);
        assert_eq!(holds(&page, bank_word(Register::IN_SERVICE, 3)), 1 << 7);
        assert_eq!(
            holds(&page, bank_word(Register::INTERRUPT_REQUEST, 3)),
            1 << 5
        );
    }

    #[test]
    fn a_hand_over_adds_to_what_the_page_already_holds() {
        // The page is what the hardware has been delivering out of, and a peer's
        // hardware may be setting a request bit in it while this runs — so each of
        // the three words is ORed in rather than stored over.
        let banks = [
            Register::TRIGGER_MODE,
            Register::IN_SERVICE,
            Register::INTERRUPT_REQUEST,
        ];
        let page = filled(0);
        let held = 1 << 1;
        for bank in banks {
            slot(&page, bank_word(bank, 0))
                .expect("a bank slot is inside the page")
                .store(held, Ordering::Relaxed);
        }
        Handover::from_words(1 << 2, 1 << 2, 1 << 2, 0)
            .publish(0, |offset| slot(&page, offset))
            .expect("a page-shaped buffer answers at every offset");
        for bank in banks {
            assert_eq!(
                holds(&page, bank_word(bank, 0)),
                held | (1 << 2),
                "{bank:?}"
            );
        }
    }
}
