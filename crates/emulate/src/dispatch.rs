//! Whether an intercepted instruction does exactly what the guest's own
//! instruction would have done, and nothing else.
//!
//! These are the tests for the properties the rest of the crate is shaped
//! around: that an operand is resolved once from pre-operation state, that the
//! plan is checked against the fault the hardware reported, that a device is
//! never asked about an access it did not answer for, and that nothing
//! irreversible happens before everything fallible has been ruled out.
//!
//! Every test asserts the whole of what should have changed *and* that nothing
//! else did. A device transaction log is kept per device, so a test can say not
//! only what the guest ended up with but how many times hardware was touched,
//! at what width, and in what order — which is the part a device emulator
//! depends on and the part that is invisible in the guest's registers.

use alloc::{boxed::Box, sync::Arc, vec::Vec};

use spin::Mutex;
use svm::exit::NestedPageFault;
use x86_64::PhysAddr;

use crate::{
    Capability, Commit, Data, Device, EmulateError, Hardware, Outcome, Read, Width, Write,
    machine::tests::{Machine, Memory, PAGE},
    mmio::{Mmio, harness::Harness},
};

/// Where every test puts its device aperture, in guest physical space.
const APERTURE: u64 = 0xFEE0_0000;
/// How long that aperture is: one page, which is the smallest a region can be.
const APERTURE_BYTES: u64 = PAGE;
/// Where every test puts the instruction the guest stopped on.
const RIP: u64 = 0x1000;

/// One thing a device was asked to do, recorded in the order it was asked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Event {
    /// The device was asked what the guest should see.
    Read {
        /// How far into the region.
        offset: u64,
        /// How much.
        width: Width,
    },
    /// The device was told what the guest wrote.
    Write {
        /// How far into the region.
        offset: u64,
        /// What the guest wrote.
        value: Data,
    },
    /// The device reached its own hardware, which is a bus transaction.
    Hardware {
        /// How far into the region.
        offset: u64,
        /// How much.
        width: Width,
    },
}

/// A device that records what it was asked and answers as it was told to.
///
/// Deliberately not clever. What these tests are about is what the *emulator*
/// does, so the device's only jobs are to say what it can answer, to log
/// faithfully, and to be predictable.
pub(crate) struct Recorder {
    capability: Capability,
    hardware: Hardware,
    events: Arc<Mutex<Vec<Event>>>,
    answer: Answer,
    decision: Decision,
}

/// What a recorder answers a read with.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Answer {
    /// Whatever the hardware holds, which makes a transaction.
    Hardware,
    /// A value out of the device's own state, which makes none.
    Invented(u64),
    /// A value of a width the access did not ask for, which is a contract
    /// violation the framework has to catch.
    Mismatched(Width),
}

/// What a recorder does with a write.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Decision {
    /// Let the guest's own value through.
    Hardware,
    /// Let a value of the same width through instead.
    Replace(u64),
    /// Let a value of another width through, which the framework has to refuse
    /// before it reaches hardware.
    ReplaceWidth(Width, u64),
    /// Keep it.
    Discard,
}

impl Recorder {
    /// A device that answers every scalar width and reaches hardware.
    pub(crate) fn new() -> Self {
        Self {
            capability: Capability::scalar(),
            hardware: Hardware::Reached,
            events: Arc::new(Mutex::new(Vec::new())),
            answer: Answer::Hardware,
            decision: Decision::Hardware,
        }
    }

    /// The same, answering only what this capability allows.
    pub(crate) const fn answering(mut self, capability: Capability) -> Self {
        self.capability = capability;
        self
    }

    /// The same, for a device that never reaches the hardware behind its
    /// region — which is registered without a mapping of it.
    pub(crate) const fn untouched(mut self) -> Self {
        self.hardware = Hardware::Untouched;
        self
    }

    /// The same, answering reads this way.
    pub(crate) const fn reads(mut self, answer: Answer) -> Self {
        self.answer = answer;
        self
    }

    /// The same, deciding writes this way.
    pub(crate) const fn writes(mut self, decision: Decision) -> Self {
        self.decision = decision;
        self
    }

    /// A handle on the log, kept by the test after the device is boxed.
    pub(crate) fn log(&self) -> Arc<Mutex<Vec<Event>>> {
        Arc::clone(&self.events)
    }
}

impl Device for Recorder {
    fn capability(&self) -> Capability {
        self.capability
    }

    fn hardware(&self) -> Hardware {
        self.hardware
    }

    fn read(&self, access: Read<'_>) -> Data {
        self.events.lock().push(Event::Read {
            offset: access.offset(),
            width: access.width(),
        });
        match self.answer {
            Answer::Hardware => match access.hardware() {
                Some(value) => {
                    self.events.lock().push(Event::Hardware {
                        offset: access.offset(),
                        width: access.width(),
                    });
                    value
                }
                // A device with no aperture has nothing to read, and the log
                // shows no transaction happened. Zero, because an answer of the
                // asked-for width is still owed.
                None => Data::from_u64(0, access.width()),
            },
            Answer::Invented(value) => Data::from_u64(value, access.width()),
            Answer::Mismatched(width) => Data::from_u64(0xAA, width),
        }
    }

    fn write(&self, access: Write<'_>) -> Commit {
        self.events.lock().push(Event::Write {
            offset: access.offset(),
            value: access.value(),
        });
        match self.decision {
            Decision::Hardware => {
                self.events.lock().push(Event::Hardware {
                    offset: access.offset(),
                    width: access.width(),
                });
                Commit::Hardware
            }
            Decision::Replace(value) => {
                self.events.lock().push(Event::Hardware {
                    offset: access.offset(),
                    width: access.width(),
                });
                Commit::Replace(Data::from_u64(value, access.width()))
            }
            Decision::ReplaceWidth(width, value) => Commit::Replace(Data::from_u64(value, width)),
            Decision::Discard => Commit::Discard,
        }
    }
}

/// Everything one dispatch needs: a register file, a guest memory, and a set of
/// regions.
pub(crate) struct Fixture {
    pub(crate) machine: Machine,
    pub(crate) memory: Memory,
    pub(crate) mmio: Mmio,
    pub(crate) log: Arc<Mutex<Vec<Event>>>,
}

impl Fixture {
    /// A 64-bit guest with one device region, whose instruction is `bytes`.
    ///
    /// The instruction is put in the control block rather than in the guest's
    /// memory, which is where a nested page fault really leaves it, and the
    /// aperture is mapped into the guest's linear space at `linear`.
    pub(crate) fn new(bytes: &[u8], recorder: Recorder) -> Self {
        let log = recorder.log();
        let machine = Machine::long_mode().at(RIP).with_fetched(bytes);
        let mut memory = Memory::new(&machine);
        // The guest reaches the device at this linear address, and its page
        // translates to the aperture.
        memory.map_at(DEVICE_LINEAR, APERTURE);
        let mmio = Harness::new()
            .region(APERTURE, APERTURE_BYTES, Box::new(recorder))
            .seal();
        Self {
            machine,
            memory,
            mmio,
            log,
        }
    }

    /// The instruction, decoded as the guest's mode decodes it.
    /// Performs the instruction, as an exit handler would.
    pub(crate) fn dispatch(
        &mut self,
        gpa: u64,
        cause: NestedPageFault,
    ) -> Result<Outcome, EmulateError> {
        self.mmio
            .execute(&mut self.machine, &self.memory, PhysAddr::new(gpa), cause)
    }

    /// Performs it against the aperture, with a fault of the given direction —
    /// which is what the hardware reports for an ordinary trapped access.
    pub(crate) fn run(&mut self, write: bool, offset: u64) -> Result<Outcome, EmulateError> {
        self.dispatch(APERTURE + offset, reported(write))
    }

    /// What the device was asked, in order.
    pub(crate) fn events(&self) -> Vec<Event> {
        self.log.lock().clone()
    }

    /// How many times hardware was touched.
    pub(crate) fn transactions(&self) -> usize {
        self.events()
            .iter()
            .filter(|event| matches!(event, Event::Hardware { .. }))
            .count()
    }
}

/// Where the guest reaches its device, as a linear address.
pub(crate) const DEVICE_LINEAR: u64 = 0x8000;

/// A nested page fault report for an ordinary trapped data access.
///
/// The two flags that matter are set the way hardware sets them for the access
/// the guest actually made: the fault is on the final address rather than on a
/// walk of the guest's own tables.
pub(crate) fn reported(write: bool) -> NestedPageFault {
    NestedPageFault::new()
        .with_present(true)
        .with_write(write)
        .with_final_address(true)
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::{
        APERTURE, APERTURE_BYTES, Answer, DEVICE_LINEAR, Decision, Event, Fixture, RIP, Recorder,
        reported,
    };
    use crate::{
        Capability, Data, EmulateError, Inadmissible, Outcome, Provenance, Spanning, Width,
        machine::Cpu,
    };

    /// `mov eax, [rip+...]` reaching the device page, in the fixture's layout.
    ///
    /// Written as an absolute address through a scaled index with no base,
    /// which is the shortest way to name one fixed linear address in 64-bit
    /// code: `mov eax, [0x8000]`.
    const LOAD_EAX: [u8; 7] = [0x8B, 0x04, 0x25, 0x00, 0x80, 0x00, 0x00];
    /// `mov [0x8000], eax`.
    const STORE_EAX: [u8; 7] = [0x89, 0x04, 0x25, 0x00, 0x80, 0x00, 0x00];

    #[test]
    fn a_device_load_puts_what_the_device_answered_in_the_register() {
        let mut fixture = Fixture::new(
            &LOAD_EAX,
            Recorder::new().reads(Answer::Invented(0xDEAD_BEEF)),
        );
        let outcome = fixture.run(false, 0).expect("a trapped load is performed");
        assert_eq!(outcome, Outcome::Stepped);
        assert_eq!(fixture.machine.gpr(0), 0xDEAD_BEEF);
        // The instruction pointer moved past exactly this instruction.
        assert_eq!(fixture.machine.save().rip, RIP + LOAD_EAX.len() as u64);
        // The device was asked once, at the width the guest used, and never
        // reached hardware because it answered out of its own state.
        assert_eq!(
            fixture.events(),
            [Event::Read {
                offset: 0,
                width: Width::Long
            }]
        );
        assert_eq!(fixture.transactions(), 0);
    }

    #[test]
    fn a_device_load_that_reaches_hardware_makes_exactly_one_transaction() {
        let mut fixture = Fixture::new(&LOAD_EAX, Recorder::new().reads(Answer::Hardware));
        assert_eq!(fixture.run(false, 0).expect("performed"), Outcome::Stepped);
        assert_eq!(
            fixture.transactions(),
            1,
            "one guest access is one transaction"
        );
        assert_eq!(
            fixture.events()[1],
            Event::Hardware {
                offset: 0,
                width: Width::Long
            },
            "the transaction is at the width the guest used, not a wider one"
        );
    }

    #[test]
    fn a_device_store_hands_the_device_what_the_guest_wrote() {
        let mut fixture = Fixture::new(&STORE_EAX, Recorder::new());
        fixture.machine.set_gpr(0, 0x1234_5678);
        let outcome = fixture.run(true, 0).expect("a trapped store is performed");
        assert_eq!(outcome, Outcome::Stepped);
        assert_eq!(
            fixture.events()[0],
            Event::Write {
                offset: 0,
                value: crate::Data::from_u64(0x1234_5678, Width::Long)
            }
        );
        assert_eq!(fixture.transactions(), 1);
    }

    #[test]
    fn a_discarded_write_never_reaches_hardware() {
        let mut fixture = Fixture::new(&STORE_EAX, Recorder::new().writes(Decision::Discard));
        fixture.machine.set_gpr(0, 0x1234_5678);
        assert_eq!(fixture.run(true, 0).expect("performed"), Outcome::Stepped);
        assert_eq!(
            fixture.transactions(),
            0,
            "a discarded write is not a transaction"
        );
        // The instruction still retires: the guest's write happened as far as the
        // guest is concerned, which is what a device deciding to ignore it means.
        assert_eq!(fixture.machine.save().rip, RIP + STORE_EAX.len() as u64);
    }

    #[test]
    fn a_replacement_of_the_same_width_reaches_hardware_instead() {
        let mut fixture = Fixture::new(
            &STORE_EAX,
            Recorder::new().writes(Decision::Replace(0xFFFF_0000)),
        );
        fixture.machine.set_gpr(0, 1);
        assert_eq!(fixture.run(true, 0).expect("performed"), Outcome::Stepped);
        assert_eq!(fixture.transactions(), 1);
    }

    #[test]
    fn a_replacement_of_another_width_is_refused_before_it_reaches_hardware() {
        // The critical case. A one-byte write answered with sixteen bytes would
        // otherwise write past the end of the mapping through a misaligned
        // pointer, from safe device code.
        for width in [Width::Byte, Width::Word, Width::Quad, Width::Vector] {
            let mut fixture = Fixture::new(
                &STORE_EAX,
                Recorder::new().writes(Decision::ReplaceWidth(width, 0xFF)),
            );
            fixture.machine.set_gpr(0, 1);
            let error = fixture
                .run(true, 0)
                .expect_err("a replacement must be bound to the admitted width");
            assert!(
                matches!(
                    error,
                    EmulateError::Inadmissible {
                        reason: Inadmissible::Answer {
                            wanted: Width::Long,
                            got
                        },
                        ..
                    } if got == width
                ),
                "a {width:?} replacement of a Long write was not refused: {error:?}"
            );
            assert_eq!(fixture.transactions(), 0, "nothing may reach hardware");
            assert_eq!(
                fixture.machine.save().rip,
                RIP,
                "a refused commit must not retire the instruction"
            );
        }
    }

    #[test]
    fn a_device_answering_the_wrong_width_is_refused_before_the_register_changes() {
        for width in [Width::Byte, Width::Word, Width::Quad, Width::Vector] {
            let mut fixture =
                Fixture::new(&LOAD_EAX, Recorder::new().reads(Answer::Mismatched(width)));
            fixture.machine.set_gpr(0, 0x1111_2222_3333_4444);
            let error = fixture
                .run(false, 0)
                .expect_err("the answer is the wrong width");
            assert!(
                matches!(
                    error,
                    EmulateError::Inadmissible {
                        reason: Inadmissible::Answer {
                            wanted: Width::Long,
                            ..
                        },
                        ..
                    }
                ),
                "{error:?}"
            );
            assert_eq!(
                fixture.machine.gpr(0),
                0x1111_2222_3333_4444,
                "the destination must not change on a contract error"
            );
            assert_eq!(fixture.machine.save().rip, RIP);
        }
    }

    #[test]
    fn a_walk_of_the_guests_own_tables_is_not_an_access_to_perform() {
        let mut fixture = Fixture::new(&LOAD_EAX, Recorder::new());
        let cause = reported(false)
            .with_final_address(false)
            .with_page_table_walk(true);
        let error = fixture
            .dispatch(APERTURE, cause)
            .expect_err("a walk fault has no data access to emulate");
        assert!(
            matches!(
                error,
                EmulateError::Provenance {
                    reason: Provenance::PageTableWalk,
                    ..
                }
            ),
            "{error:?}"
        );
        assert_eq!(fixture.events(), [], "no device may be asked");
    }

    #[test]
    fn a_fault_not_reported_against_the_access_is_refused() {
        let mut fixture = Fixture::new(&LOAD_EAX, Recorder::new());
        let error = fixture
            .dispatch(APERTURE, reported(false).with_final_address(false))
            .expect_err("without a final-address fault the access is not established");
        assert!(
            matches!(
                error,
                EmulateError::Provenance {
                    reason: Provenance::NotFinal,
                    ..
                }
            ),
            "{error:?}"
        );
        assert_eq!(fixture.events(), []);
    }

    #[test]
    fn a_direction_the_instruction_does_not_perform_is_refused() {
        // The hardware says a write trapped; the instruction only reads the
        // device. Performing it anyway would make an access the processor never
        // made.
        let mut fixture = Fixture::new(&LOAD_EAX, Recorder::new());
        let error = fixture
            .run(true, 0)
            .expect_err("a reported write does not match a load");
        assert!(
            matches!(
                error,
                EmulateError::Provenance {
                    reason: Provenance::Direction { .. },
                    ..
                }
            ),
            "{error:?}"
        );
        assert_eq!(fixture.events(), [], "no device may be asked");
        assert_eq!(fixture.machine.save().rip, RIP);
    }

    #[test]
    fn a_fault_at_an_address_the_instruction_does_not_touch_is_refused() {
        let mut fixture = Fixture::new(&LOAD_EAX, Recorder::new());
        // Inside the region, but not a byte of the four this instruction reads.
        let error = fixture
            .run(false, 0x40)
            .expect_err("the reported address is not part of the access");
        assert!(
            matches!(
                error,
                EmulateError::Provenance {
                    reason: Provenance::Elsewhere,
                    ..
                }
            ),
            "{error:?}"
        );
        assert_eq!(fixture.events(), []);
    }

    #[test]
    fn a_fault_against_a_later_byte_of_the_access_is_accepted() {
        // A four-byte read may be reported against its third byte. Requiring the
        // first would refuse a legitimate exit and leave the guest faulting for
        // ever.
        for offset in 0..4 {
            let mut fixture = Fixture::new(&LOAD_EAX, Recorder::new().reads(Answer::Invented(7)));
            let outcome = fixture
                .run(false, offset)
                .unwrap_or_else(|error| panic!("byte {offset} of the access: {error:?}"));
            assert_eq!(outcome, Outcome::Stepped);
            assert_eq!(fixture.machine.gpr(0), 7);
        }
    }

    #[test]
    fn an_instruction_fetch_fault_is_not_an_access_to_emulate() {
        let mut fixture = Fixture::new(&LOAD_EAX, Recorder::new());
        let cause = reported(false).with_instruction_fetch(true);
        let error = fixture
            .dispatch(APERTURE, cause)
            .expect_err("the fetch itself faulted");
        assert!(
            matches!(error, EmulateError::FetchFault { .. }),
            "{error:?}"
        );
        assert_eq!(fixture.events(), []);
    }

    #[test]
    fn an_instruction_touching_no_trapped_region_is_reported_as_such() {
        // `mov eax, [0x9000]` — an address the fixture describes as ordinary
        // memory rather than as the device.
        const ELSEWHERE: [u8; 7] = [0x8B, 0x04, 0x25, 0x00, 0x90, 0x00, 0x00];
        let mut fixture = Fixture::new(&ELSEWHERE, Recorder::new());
        fixture.memory.map(0x9000);
        let error = fixture
            .run(false, 0)
            .expect_err("nothing interposes on the address it names");
        assert!(
            matches!(error, EmulateError::NotTrapped { .. }),
            "{error:?}"
        );
        assert_eq!(fixture.events(), []);
    }

    #[test]
    fn an_access_straddling_a_region_boundary_is_refused_before_any_transaction() {
        // Four bytes starting one byte before the end of the aperture: partly the
        // device's registers and partly whatever is past them. Routing the whole
        // access to the device would hand it a byte that is not its own; routing
        // it to memory would miss the device entirely.
        const AT_END: [u8; 7] = [0x8B, 0x04, 0x25, 0xFF, 0x8F, 0x00, 0x00];
        let mut fixture = Fixture::new(&AT_END, Recorder::new());
        // Physically contiguous with the aperture, so that the span itself is one
        // run of addresses and the only thing wrong with it is that half of it is
        // past the end of the region.
        fixture
            .memory
            .map_at(DEVICE_LINEAR + APERTURE_BYTES, APERTURE + APERTURE_BYTES);
        let error = fixture
            .run(false, APERTURE_BYTES - 1)
            .expect_err("the access leaves the region");
        assert!(
            matches!(
                error,
                EmulateError::Span {
                    reason: Spanning::Straddles,
                    ..
                }
            ),
            "{error:?}"
        );
        assert_eq!(fixture.events(), [], "no device may be asked");
    }

    #[test]
    fn a_device_that_answers_only_dwords_refuses_a_byte_access() {
        // `mov al, [0x8000]`
        const LOAD_AL: [u8; 7] = [0x8A, 0x04, 0x25, 0x00, 0x80, 0x00, 0x00];
        let mut fixture = Fixture::new(
            &LOAD_AL,
            Recorder::new().answering(Capability::only(Width::Long)),
        );
        let error = fixture
            .run(false, 0)
            .expect_err("the device decodes only dwords");
        assert!(
            matches!(
                error,
                EmulateError::Inadmissible {
                    reason: Inadmissible::Width,
                    ..
                }
            ),
            "{error:?}"
        );
        assert_eq!(fixture.events(), [], "the device must not be asked at all");
    }

    #[test]
    fn a_misaligned_access_to_an_alignment_requiring_device_is_refused() {
        // `mov eax, [0x8002]` — a dword two bytes into the aperture.
        const MISALIGNED: [u8; 7] = [0x8B, 0x04, 0x25, 0x02, 0x80, 0x00, 0x00];
        let mut fixture = Fixture::new(&MISALIGNED, Recorder::new());
        let error = fixture
            .run(false, 2)
            .expect_err("the pointer would not be aligned for its width");
        assert!(
            matches!(
                error,
                EmulateError::Inadmissible {
                    reason: Inadmissible::Alignment,
                    ..
                }
            ),
            "{error:?}"
        );
        assert_eq!(fixture.events(), []);
    }

    #[test]
    fn a_sixteen_byte_access_is_refused_unless_the_device_allows_being_split() {
        // `movdqu xmm0, [0x8000]`
        const LOAD_XMM: [u8; 9] = [0xF3, 0x0F, 0x6F, 0x04, 0x25, 0x00, 0x80, 0x00, 0x00];
        let mut fixture = Fixture::new(&LOAD_XMM, Recorder::new());
        let error = fixture
            .run(false, 0)
            .expect_err("one bus transaction of sixteen bytes cannot be made safely");
        assert!(
            matches!(
                error,
                EmulateError::Inadmissible {
                    reason: Inadmissible::Indivisible,
                    ..
                }
            ),
            "{error:?}"
        );
        assert_eq!(fixture.events(), []);
    }

    #[test]
    fn a_device_that_answers_every_scalar_width_is_asked_about_a_byte_access() {
        // The other side of the two tests above, and the property a device
        // relies on to serve an access the architecture leaves undefined rather
        // than have it refused before the device is consulted: what a malformed
        // access does is the device's decision only if the device is asked.
        //
        // `mov al, [0x8000]`
        const LOAD_AL: [u8; 7] = [0x8A, 0x04, 0x25, 0x00, 0x80, 0x00, 0x00];
        let mut fixture = Fixture::new(&LOAD_AL, Recorder::new().reads(Answer::Invented(0x5A)));
        let outcome = fixture.run(false, 0).expect("a byte access is admitted");
        assert_eq!(outcome, Outcome::Stepped);
        assert_eq!(
            fixture.events(),
            [Event::Read {
                offset: 0,
                width: Width::Byte
            }],
            "the device is asked at the width the guest used"
        );
        assert_eq!(fixture.machine.gpr(0) & 0xFF, 0x5A);
        assert_eq!(fixture.machine.save().rip, RIP + LOAD_AL.len() as u64);
    }

    #[test]
    fn a_device_that_declares_being_split_harmless_is_asked_about_sixteen_bytes() {
        // `movdqu xmm0, [0x8000]`
        const LOAD_XMM: [u8; 9] = [0xF3, 0x0F, 0x6F, 0x04, 0x25, 0x00, 0x80, 0x00, 0x00];
        let mut fixture = Fixture::new(
            &LOAD_XMM,
            Recorder::new()
                .answering(Capability::scalar().split_vectors())
                .reads(Answer::Invented(0x1234)),
        );
        let outcome = fixture
            .run(false, 0)
            .expect("splitting was declared harmless");
        assert_eq!(outcome, Outcome::Stepped);
        assert_eq!(
            fixture.events(),
            [Event::Read {
                offset: 0,
                width: Width::Vector
            }],
            "one guest access is one question, however many transactions it takes"
        );
    }

    #[test]
    fn a_device_with_no_hardware_behind_it_reaches_none_and_still_answers() {
        // A device that declared it never touches its hardware is registered
        // without a mapping, so asking for the hardware value answers nothing
        // and no transaction is made. The device still owes the guest an answer.
        let mut fixture = Fixture::new(&LOAD_EAX, Recorder::new().untouched());
        fixture.machine.set_gpr(0, 0x1111_2222_3333_4444);
        let outcome = fixture.run(false, 0).expect("the read is still answered");
        assert_eq!(outcome, Outcome::Stepped);
        assert_eq!(
            fixture.transactions(),
            0,
            "there is nothing to transact with"
        );
        assert_eq!(fixture.machine.gpr(0), 0, "the answer the device gave");
    }

    #[test]
    fn a_write_cannot_be_let_through_to_hardware_a_device_declared_untouched() {
        // The contract the missing mapping rests on. A device that declared it
        // never reaches its hardware and then asks for a write to reach it is
        // refused rather than served through some other address.
        let mut fixture = Fixture::new(&STORE_EAX, Recorder::new().untouched());
        fixture.machine.set_gpr(0, 0x1234_5678);
        let error = fixture
            .run(true, 0)
            .expect_err("there is no mapping to write through");
        assert!(
            matches!(
                error,
                EmulateError::Inadmissible {
                    reason: Inadmissible::Untouched,
                    ..
                }
            ),
            "{error:?}"
        );
        // The device asked — its log says so — and the framework is what refused
        // it, which is the property the missing mapping needs.
        assert_eq!(
            fixture.events().last(),
            Some(&Event::Hardware {
                offset: 0,
                width: Width::Long
            })
        );
        assert_eq!(
            fixture.machine.save().rip,
            RIP,
            "a refused commit must not retire the instruction"
        );
    }

    /// `rep movsb` — the copy a driver writes when it means "move this buffer".
    const REP_MOVSB: [u8; 2] = [0xF3, 0xA4];
    /// `movsb` without the prefix: exactly one element.
    const MOVSB: [u8; 1] = [0xA4];
    /// `rep stosd` — fill with the accumulator, four bytes at a time.
    const REP_STOSD: [u8; 2] = [0xF3, 0xAB];
    /// `lodsd` — one element into the accumulator.
    const LODSD: [u8; 1] = [0xAD];

    /// Which encoded number the source and destination index registers are.
    const RSI: u8 = 6;
    const RDI: u8 = 7;
    /// And the count register.
    const RCX: u8 = 1;

    #[test]
    fn one_unrepeated_string_move_copies_exactly_one_element() {
        let mut fixture = Fixture::new(&MOVSB, Recorder::new().reads(Answer::Invented(0x5A)));
        // Source is the device, destination is ordinary memory.
        fixture.memory.map(0x2000);
        fixture.machine.set_gpr(RSI, DEVICE_LINEAR);
        fixture.machine.set_gpr(RDI, 0x2000);
        fixture.machine.set_gpr(RCX, 0xFFFF);

        let outcome = fixture.run(false, 0).expect("one element is copied");
        assert_eq!(outcome, Outcome::Stepped);
        assert_eq!(fixture.memory.peek(0x2000, 1), [0x5A]);
        assert_eq!(
            fixture.memory.peek(0x2001, 1),
            [0],
            "only one element moves"
        );
        // Both indices step by one element, and the count is untouched without a
        // repeat prefix.
        assert_eq!(fixture.machine.gpr(RSI), DEVICE_LINEAR + 1);
        assert_eq!(fixture.machine.gpr(RDI), 0x2001);
        assert_eq!(
            fixture.machine.gpr(RCX),
            0xFFFF,
            "an unrepeated move leaves the count"
        );
        assert_eq!(fixture.machine.save().rip, RIP + MOVSB.len() as u64);
        assert_eq!(
            fixture.events().len(),
            1,
            "one element is one device access"
        );
    }

    #[test]
    fn a_repeated_move_copies_every_element_and_lands_on_zero() {
        let mut fixture = Fixture::new(&REP_MOVSB, Recorder::new().reads(Answer::Invented(0xC3)));
        fixture.memory.map(0x2000);
        fixture.machine.set_gpr(RSI, DEVICE_LINEAR);
        fixture.machine.set_gpr(RDI, 0x2000);
        fixture.machine.set_gpr(RCX, 8);

        let outcome = fixture.run(false, 0).expect("the batch completes");
        assert_eq!(
            outcome,
            Outcome::Stepped,
            "a finished batch retires the instruction"
        );
        assert_eq!(fixture.memory.peek(0x2000, 8), [0xC3; 8]);
        assert_eq!(
            fixture.memory.peek(0x2008, 1),
            [0],
            "nothing past the count"
        );
        assert_eq!(fixture.machine.gpr(RCX), 0);
        assert_eq!(fixture.machine.gpr(RSI), DEVICE_LINEAR + 8);
        assert_eq!(fixture.machine.gpr(RDI), 0x2008);
        assert_eq!(fixture.machine.save().rip, RIP + REP_MOVSB.len() as u64);
        assert_eq!(fixture.events().len(), 8, "one device access per element");
    }

    #[test]
    fn a_repeated_move_with_a_zero_count_does_nothing_at_all() {
        let mut fixture = Fixture::new(&REP_MOVSB, Recorder::new());
        // Deliberately unmapped indices: a zero-count REP must not translate them,
        // because a real processor never touches them either.
        fixture.machine.set_gpr(RSI, 0xDEAD_0000);
        fixture.machine.set_gpr(RDI, 0xBEEF_0000);
        fixture.machine.set_gpr(RCX, 0);

        let outcome = fixture
            .dispatch(APERTURE, reported(false))
            .expect("a zero-count repeat is a complete instruction");
        assert_eq!(outcome, Outcome::Stepped);
        assert_eq!(fixture.events(), [], "nothing is accessed");
        assert_eq!(fixture.machine.gpr(RSI), 0xDEAD_0000, "no index moves");
        assert_eq!(fixture.machine.gpr(RDI), 0xBEEF_0000);
        assert_eq!(fixture.machine.save().rip, RIP + REP_MOVSB.len() as u64);
    }

    #[test]
    fn the_direction_flag_makes_the_indices_count_downwards() {
        let mut fixture = Fixture::new(&MOVSB, Recorder::new().reads(Answer::Invented(1)));
        fixture.machine = fixture.machine.clone().backwards();
        fixture.memory.map(0x2000);
        fixture.machine.set_gpr(RSI, DEVICE_LINEAR + 0x10);
        fixture.machine.set_gpr(RDI, 0x2010);
        fixture.machine.set_gpr(RCX, 4);

        assert_eq!(
            fixture
                .run(false, 0x10)
                .expect("one element is copied backwards"),
            Outcome::Stepped
        );
        assert_eq!(fixture.machine.gpr(RSI), DEVICE_LINEAR + 0xF);
        assert_eq!(fixture.machine.gpr(RDI), 0x200F);
    }

    #[test]
    fn a_repeated_store_writes_the_accumulator_at_its_own_width() {
        let mut fixture = Fixture::new(&REP_STOSD, Recorder::new());
        fixture.machine.set_gpr(RDI, DEVICE_LINEAR);
        fixture.machine.set_gpr(RCX, 4);
        fixture.machine.set_gpr(0, 0x1122_3344);

        let outcome = fixture.run(true, 0).expect("the batch completes");
        assert_eq!(outcome, Outcome::Stepped);
        // Four dword writes, at ascending offsets, each carrying the accumulator.
        let writes: Vec<_> = fixture
            .events()
            .into_iter()
            .filter_map(|event| match event {
                Event::Write { offset, value } => Some((offset, value)),
                _ => None,
            })
            .collect();
        assert_eq!(writes.len(), 4);
        for (index, (offset, value)) in writes.into_iter().enumerate() {
            assert_eq!(
                offset,
                index as u64 * 4,
                "element {index} is at its own offset"
            );
            assert_eq!(value, Data::from_u64(0x1122_3344, Width::Long));
        }
        assert_eq!(fixture.machine.gpr(RDI), DEVICE_LINEAR + 16);
        assert_eq!(fixture.machine.gpr(RCX), 0);
        assert_eq!(
            fixture.machine.gpr(0),
            0x1122_3344,
            "the accumulator is a source only"
        );
    }

    #[test]
    fn a_load_string_replaces_the_accumulator_and_zero_extends_it() {
        let mut fixture =
            Fixture::new(&LODSD, Recorder::new().reads(Answer::Invented(0xAABB_CCDD)));
        fixture.machine.set_gpr(RSI, DEVICE_LINEAR);
        fixture.machine.set_gpr(0, 0xFFFF_FFFF_FFFF_FFFF);

        assert_eq!(
            fixture.run(false, 0).expect("one element is loaded"),
            Outcome::Stepped
        );
        assert_eq!(
            fixture.machine.gpr(0),
            0xAABB_CCDD,
            "a dword write clears the upper half of the accumulator"
        );
        assert_eq!(fixture.machine.gpr(RSI), DEVICE_LINEAR + 4);
    }

    #[test]
    fn a_batch_that_reaches_a_page_boundary_stops_and_says_so() {
        // The count runs past the end of the device page. The batch must stop at
        // the boundary with the registers describing its progress, so that the
        // guest re-executes the instruction and the next exit carries on — and
        // must not walk off the region.
        let mut fixture = Fixture::new(&REP_MOVSB, Recorder::new().reads(Answer::Invented(9)));
        fixture.memory.map(0x2000);
        let last = DEVICE_LINEAR + APERTURE_BYTES - 2;
        fixture.machine.set_gpr(RSI, last);
        fixture.machine.set_gpr(RDI, 0x2000);
        fixture.machine.set_gpr(RCX, 32);

        let outcome = fixture
            .run(false, APERTURE_BYTES - 2)
            .expect("the batch stops at the boundary");
        assert_eq!(
            outcome,
            Outcome::Repeating,
            "a batch with repetitions left must not retire the instruction"
        );
        assert_eq!(
            fixture.machine.save().rip,
            RIP,
            "the guest must execute the instruction again"
        );
        // Exactly the elements before the boundary were copied, and the registers
        // say so.
        let copied = 32 - fixture.machine.gpr(RCX);
        assert!(copied > 0, "a repeating batch must make progress");
        assert_eq!(fixture.machine.gpr(RSI), last + copied);
        assert_eq!(fixture.machine.gpr(RDI), 0x2000 + copied);
        assert_eq!(
            u64::try_from(fixture.events().len()).expect("a batch is small"),
            copied
        );
    }

    #[test]
    fn a_batch_whose_destination_is_not_writable_stops_before_reading_the_device() {
        // The preflight that makes a device read the last fallible step. Reading
        // first and discovering the destination afterwards would consume a device
        // read that cannot be given back.
        let mut fixture = Fixture::new(&MOVSB, Recorder::new());
        fixture.memory.map_read_only(0x2000);
        fixture.machine.set_gpr(RSI, DEVICE_LINEAR);
        fixture.machine.set_gpr(RDI, 0x2000);
        fixture.machine.set_gpr(RCX, 1);

        let error = fixture
            .run(false, 0)
            .expect_err("the destination will not take the write");
        assert!(matches!(error, EmulateError::Discarded { .. }), "{error:?}");
        assert_eq!(
            fixture.events(),
            [],
            "the device must not be read when the destination would refuse"
        );
        assert_eq!(
            fixture.machine.gpr(RSI),
            DEVICE_LINEAR,
            "no progress is made"
        );
        assert_eq!(fixture.machine.save().rip, RIP);
    }

    #[test]
    fn a_batch_stops_at_a_page_boundary_rather_than_walking_into_the_next_page() {
        // The destination reaches the end of its page part way through the count.
        // The batch stops there with its progress in the registers, the guest
        // re-executes the instruction, and whether the next page is mapped is then
        // the hardware's own walk to answer — which is how a real processor
        // arrives at #PF for it.
        let mut fixture = Fixture::new(&REP_MOVSB, Recorder::new().reads(Answer::Invented(0x77)));
        fixture.memory.map(0x2000);
        fixture.machine.set_gpr(RSI, DEVICE_LINEAR);
        fixture.machine.set_gpr(RDI, 0x2FFC);
        fixture.machine.set_gpr(RCX, 8);

        let outcome = fixture
            .run(false, 0)
            .expect("the batch stops at the boundary");
        assert_eq!(outcome, Outcome::Repeating);
        // Exactly the four elements up to the boundary, and the registers describe
        // precisely that.
        assert_eq!(fixture.machine.gpr(RCX), 4);
        assert_eq!(fixture.machine.gpr(RSI), DEVICE_LINEAR + 4);
        assert_eq!(fixture.machine.gpr(RDI), 0x3000);
        assert_eq!(fixture.memory.peek(0x2FFC, 4), [0x77; 4]);
        assert_eq!(
            fixture.machine.save().rip,
            RIP,
            "the instruction must be executed again"
        );
    }

    #[test]
    fn an_unmapped_guest_page_is_a_fault_the_guest_takes_rather_than_a_hypervisor_failure() {
        // A demand-paged guest hits this constantly. A real processor delivers #PF
        // and the handler maps the page; reporting it as an error would stop a
        // guest that hardware would simply have faulted. The count is one so that
        // no repetition can have completed first — zero progress must not turn a
        // guest fault into a hypervisor failure.
        let mut fixture = Fixture::new(&REP_MOVSB, Recorder::new().reads(Answer::Invented(1)));
        fixture.machine.set_gpr(RSI, DEVICE_LINEAR);
        fixture.machine.set_gpr(RDI, 0x5000);
        fixture.machine.set_gpr(RCX, 1);

        let outcome = fixture
            .run(false, 0)
            .expect("an unmapped guest page is the guest's fault to take");
        let Outcome::Faulted(fault) = outcome else {
            panic!("expected a guest fault, got {outcome:?}");
        };
        assert_eq!(fault.vector(), 14, "a missing page is #PF");
        assert_eq!(
            fault.address(),
            Some(0x5000),
            "CR2 names the address that faulted"
        );
        assert_eq!(
            fault.code(),
            0,
            "present bit clear, and nothing claimed beyond that"
        );
        // Nothing completed, so nothing moved, and the instruction is re-executed
        // once the guest has mapped the page.
        assert_eq!(fixture.machine.gpr(RCX), 1);
        assert_eq!(fixture.machine.gpr(RSI), DEVICE_LINEAR);
        assert_eq!(fixture.machine.gpr(RDI), 0x5000);
        assert_eq!(fixture.machine.save().rip, RIP);
        assert_eq!(
            fixture.events(),
            [],
            "the device must not be read for a repetition that cannot complete"
        );
    }

    #[test]
    fn a_batch_translates_each_end_once_rather_than_once_per_element() {
        // The cost that matters. A translation is a walk of the guest's own page
        // tables, and doing one per element per end turns a page-sized copy into
        // thousands of walks to answer a question whose answer a batch has already
        // bounded: the batch stops at the page boundary, so every element of it is
        // described by the same entry.
        let mut fixture = Fixture::new(&REP_MOVSB, Recorder::new().reads(Answer::Invented(0x42)));
        fixture.memory.map(0x2000);
        fixture.machine.set_gpr(RSI, DEVICE_LINEAR);
        fixture.machine.set_gpr(RDI, 0x2000);
        fixture.machine.set_gpr(RCX, 64);
        fixture.memory.forget_walks();

        let outcome = fixture.run(false, 0).expect("the batch completes");
        assert_eq!(outcome, Outcome::Stepped);
        assert_eq!(fixture.memory.peek(0x2000, 64), [0x42; 64]);
        // Two ends, resolved once each. The bound is deliberately loose — what is
        // being asserted is that the count does not grow with the element count.
        assert!(
            fixture.memory.walks() <= 4,
            "a 64-element batch took {} walks; it must not pay one per element",
            fixture.memory.walks()
        );
    }

    #[test]
    fn a_longer_batch_costs_no_more_translations_than_a_short_one() {
        // The property stated directly: translations are a function of how many
        // pages a batch touches, not of how many elements it moves.
        //
        // Both counts are within the transaction budget, so both batches run to
        // completion and the only thing differing between them is how many elements
        // they moved.
        let walks = |count: u64| {
            let mut fixture = Fixture::new(&REP_MOVSB, Recorder::new().reads(Answer::Invented(1)));
            fixture.memory.map(0x2000);
            fixture.machine.set_gpr(RSI, DEVICE_LINEAR);
            fixture.machine.set_gpr(RDI, 0x2000);
            fixture.machine.set_gpr(RCX, count);
            fixture.memory.forget_walks();
            assert_eq!(
                fixture.run(false, 0).expect("the batch completes"),
                Outcome::Stepped
            );
            fixture.memory.walks()
        };
        assert_eq!(
            walks(4),
            walks(48),
            "a batch twelve times longer must not cost twelve times the walks"
        );
    }

    #[test]
    fn stepping_within_a_batch_still_moves_the_device_offset_element_by_element() {
        // What a stepped batch must not get wrong: reusing a translation is only
        // sound if the offset into the region moves with it. An offset that stuck
        // would write every element to the device's first register.
        let mut fixture = Fixture::new(&REP_STOSD, Recorder::new());
        fixture.machine.set_gpr(RDI, DEVICE_LINEAR);
        fixture.machine.set_gpr(RCX, 8);
        fixture.machine.set_gpr(0, 0xAABB_CCDD);

        assert_eq!(fixture.run(true, 0).expect("performed"), Outcome::Stepped);
        let offsets: Vec<_> = fixture
            .events()
            .into_iter()
            .filter_map(|event| match event {
                Event::Write { offset, .. } => Some(offset),
                _ => None,
            })
            .collect();
        assert_eq!(offsets, (0..8).map(|index| index * 4).collect::<Vec<_>>());
    }

    #[test]
    fn a_backwards_batch_walks_down_element_by_element() {
        let mut fixture = Fixture::new(&REP_STOSD, Recorder::new());
        fixture.machine = fixture.machine.clone().backwards();
        fixture.machine.set_gpr(RDI, DEVICE_LINEAR + 0x1C);
        fixture.machine.set_gpr(RCX, 8);
        fixture.machine.set_gpr(0, 1);

        assert_eq!(
            fixture.run(true, 0x1C).expect("performed"),
            Outcome::Stepped
        );
        let offsets: Vec<_> = fixture
            .events()
            .into_iter()
            .filter_map(|event| match event {
                Event::Write { offset, .. } => Some(offset),
                _ => None,
            })
            .collect();
        assert_eq!(
            offsets,
            (0..8).map(|index| 0x1C - index * 4).collect::<Vec<_>>(),
            "a backwards batch must descend rather than repeat one offset"
        );
        assert_eq!(fixture.machine.gpr(RDI), DEVICE_LINEAR - 4);
    }

    #[test]
    fn an_overlapping_copy_moves_element_by_element_rather_than_as_a_snapshot() {
        // MOVS is defined element by element, so an overlapping forward copy
        // propagates the first element through the range — which is what a guest's
        // own hardware would do, and is not what a `memmove` would.
        let mut fixture = Fixture::new(&REP_MOVSB, Recorder::new());
        fixture.memory.map(0x2000);
        fixture.memory.fill(0x2000, &[0xAB, 0, 0, 0, 0]);
        fixture.machine.set_gpr(RSI, 0x2000);
        fixture.machine.set_gpr(RDI, 0x2001);
        fixture.machine.set_gpr(RCX, 4);

        // No device is involved, so this is not a trapped access; it is dispatched
        // to prove the copy semantics rather than the interposition.
        let error = fixture
            .run(false, 0)
            .expect_err("nothing here is interposed");
        assert!(
            matches!(error, EmulateError::NotTrapped { .. }),
            "{error:?}"
        );
        assert_eq!(
            fixture.memory.peek(0x2000, 5),
            [0xAB, 0, 0, 0, 0],
            "an instruction that touches no trapped region must change nothing"
        );
    }

    #[test]
    fn a_thirty_two_bit_address_size_wraps_the_indices_within_thirty_two_bits() {
        // `rep movsb` with a 0x67 address-size override walks ESI/EDI/ECX, and a
        // write to one of those clears the upper half of the register rather than
        // carrying into it.
        let mut fixture = Fixture::new(
            &[0xF3, 0x67, 0xA4],
            Recorder::new().reads(Answer::Invented(5)),
        );
        fixture.memory.map(0x2000);
        fixture
            .machine
            .set_gpr(RSI, 0xFFFF_FFFF_0000_0000 | DEVICE_LINEAR);
        fixture.machine.set_gpr(RDI, 0xFFFF_FFFF_0000_0000 | 0x2000);
        fixture.machine.set_gpr(RCX, 0xFFFF_FFFF_0000_0002);

        assert_eq!(
            fixture.run(false, 0).expect("two elements are copied"),
            Outcome::Stepped
        );
        assert_eq!(
            fixture.machine.gpr(RSI),
            DEVICE_LINEAR + 2,
            "a 32-bit index write clears the upper half rather than preserving it"
        );
        assert_eq!(fixture.machine.gpr(RDI), 0x2002);
        assert_eq!(
            fixture.machine.gpr(RCX),
            0,
            "the count is counted at 32 bits"
        );
    }

    #[test]
    fn a_load_string_at_a_narrow_width_preserves_the_upper_accumulator() {
        // `lodsb` writes AL and leaves bits 63:8 alone — the partial-register rule,
        // reached through the string path rather than through a decoded operand.
        let mut fixture = Fixture::new(&[0xAC], Recorder::new().reads(Answer::Invented(0x5A)));
        fixture.machine.set_gpr(RSI, DEVICE_LINEAR);
        fixture.machine.set_gpr(0, 0x1122_3344_5566_7788);

        assert_eq!(fixture.run(false, 0).expect("performed"), Outcome::Stepped);
        assert_eq!(fixture.machine.gpr(0), 0x1122_3344_5566_775A);
    }

    #[test]
    fn the_repne_prefix_has_no_meaning_on_these_and_is_refused() {
        // `repne movsb`. The architecture gives this prefix a meaning on the
        // comparing string instructions and none on these, so performing it as
        // though it were the other prefix would repeat what the guest did not ask
        // to repeat.
        let mut fixture = Fixture::new(&[0xF2, 0xA4], Recorder::new());
        fixture.memory.map(0x2000);
        fixture.machine.set_gpr(RSI, DEVICE_LINEAR);
        fixture.machine.set_gpr(RDI, 0x2000);
        fixture.machine.set_gpr(RCX, 4);

        let error = fixture
            .run(false, 0)
            .expect_err("the prefix has no meaning here");
        assert!(matches!(error, EmulateError::Operand { .. }), "{error:?}");
        assert_eq!(fixture.events(), []);
        assert_eq!(fixture.machine.gpr(RCX), 4, "nothing is repeated");
    }

    #[test]
    fn a_load_whose_address_comes_from_the_register_it_writes_is_resolved_once() {
        // `mov rax, [rax]` against the device. The move writes the register its
        // own address was computed from, so anything that re-derives the address
        // afterwards derives it from what was just read — and would conclude the
        // instruction touched no trapped region at all.
        const LOAD_RAX_MEM: [u8; 3] = [0x48, 0x8B, 0x00];
        let mut fixture = Fixture::new(
            &LOAD_RAX_MEM,
            Recorder::new().reads(Answer::Invented(0x4000)),
        );
        fixture.machine.set_gpr(0, DEVICE_LINEAR);
        let outcome = fixture
            .run(false, 0)
            .expect("resolved from pre-operation state");
        assert_eq!(outcome, Outcome::Stepped);
        assert_eq!(
            fixture.machine.gpr(0),
            0x4000,
            "the register holds what the device answered"
        );
        assert_eq!(fixture.machine.save().rip, RIP + LOAD_RAX_MEM.len() as u64);
    }
}
