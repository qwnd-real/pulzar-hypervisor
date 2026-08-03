//! What an instruction is going to do, worked out before any of it is done.
//!
//! A plan is the whole of an intercepted access: which bytes, in which
//! direction, against which region, at which offset into it. It is built from
//! the register state the guest stopped with and is not consulted again
//! afterwards — which is the point, because performing a move changes the
//! registers its own addresses were computed from.
//!
//! # Why the whole span is planned and not the first byte
//!
//! An access is a range of linear addresses, and nothing about the architecture
//! makes a range land anywhere in particular. Four bytes two bytes from the end
//! of a page are two pages; two pages are two translations; two translations
//! may be two guest physical addresses that are nowhere near each other, may be
//! one trapped region and one page of ordinary memory, or may be two regions
//! two different devices answer for.
//!
//! Classifying an access by its first byte answers all of those the same way
//! and is wrong in each of them differently: it routes somebody else's bytes to
//! a device, or writes half a value to hardware and half to memory, or reads a
//! device that the guest's access never reached. So a plan requires the span to
//! be one contiguous piece of one thing, and refuses what it cannot describe
//! rather than approximating it. A refusal costs the guest an unimplemented
//! access it can be told about; an approximation costs it a wrong value it
//! cannot.
//!
//! # Why the plan is checked against the fault
//!
//! The processor reported an address, a direction, and whether the fault was on
//! the access itself or on a walk of the guest's own page tables. All three are
//! evidence about which access trapped, and an emulator that ignores them
//! performs whichever access it decoded instead — which need not be the same
//! one. A guest with two processors can arrange for it not to be: one faults on
//! a device address, the other rewrites the page table entry, and an emulator
//! that re-derives the address performs the access against whatever the entry
//! now says.

use iced_x86::Instruction;
use memory::{Addressing, Mode};
use svm::{SaveArea, exit::NestedPageFault};
use x86_64::PhysAddr;

use crate::{
    EmulateError, Provenance, Spanning, as_u64,
    machine::{Cpu, Guest},
    mmio::Mmio,
    operand::{self, Place},
    value::Width,
};

/// Bytes in the smallest page, which is where anything about a linear address
/// stops being knowable without translating again.
pub(crate) const PAGE: u64 = 4096;

/// Which way an access goes, as far as anything outside the instruction is
/// concerned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Direction {
    /// The instruction takes bytes from here.
    Read,
    /// The instruction puts bytes here.
    Write,
}

impl Direction {
    /// What to call this in a diagnostic.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
        }
    }
}

/// One end of a move, with every byte of it accounted for.
///
/// Built once and then only read. Nothing here is recomputed while the
/// instruction is performed, because by then the registers it was computed from
/// may have been written by the instruction itself.
#[derive(Clone, Copy, Debug)]
pub(crate) struct End {
    /// Where the bytes are.
    pub(crate) place: Place,
    /// How many of them there are.
    pub(crate) width: Width,
    /// Which way they go.
    pub(crate) direction: Direction,
}

impl End {
    /// Whether this end is a region something answers for.
    pub(crate) const fn interposed(&self) -> bool {
        self.place.interposed()
    }

    /// Whether any byte of this end is at that guest physical address.
    ///
    /// The whole span rather than its first byte: a fault may perfectly well be
    /// reported against the third byte of a four-byte access, and requiring the
    /// first to match would refuse a legitimate exit.
    fn covers(&self, gpa: PhysAddr) -> bool {
        let Place::Device { gpa: base, .. } = self.place else {
            return false;
        };
        let (base, wanted) = (base.as_u64(), gpa.as_u64());
        wanted >= base && wanted - base < self.width.span()
    }
}

/// Everything an intercepted instruction is going to do.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Plan {
    /// Where the value comes from.
    pub(crate) from: End,
    /// Where it goes.
    pub(crate) to: End,
}

impl Plan {
    /// Both ends of a two-operand move, resolved from the state the guest
    /// stopped with.
    ///
    /// The source is resolved first only so that a failure names the operand a
    /// reader would look at first; neither resolution touches anything.
    ///
    /// # Errors
    ///
    /// Whatever resolving either operand reports, and [`EmulateError::Span`] if
    /// either end is a range that no single access covers.
    pub(crate) fn moving(
        mmio: &Mmio,
        cpu: &impl Cpu,
        guest: &impl Guest,
        instruction: &Instruction,
        widths: (Width, Width),
        operands: (u32, u32),
    ) -> Result<Self, EmulateError> {
        let (source, destination) = operands;
        let (from, to) = (
            End {
                place: operand::place(mmio, cpu, guest, instruction, source, widths.0)?,
                width: widths.0,
                direction: Direction::Read,
            },
            End {
                place: operand::place(mmio, cpu, guest, instruction, destination, widths.1)?,
                width: widths.1,
                direction: Direction::Write,
            },
        );
        Ok(Self { from, to })
    }

    /// Confirms that this plan accounts for the fault the hardware reported,
    /// and says which end of it trapped.
    ///
    /// Four things are checked, and none of them is redundant. The fault must
    /// be on the access rather than on a walk of the guest's own tables,
    /// because a walk has no data access to perform. It must be reported
    /// against the final address, because otherwise which access it belongs
    /// to is not established. Its direction must be one this instruction
    /// performs. And its address must be a byte of an end of this
    /// instruction that is actually interposed on.
    ///
    /// # Errors
    ///
    /// [`EmulateError::Provenance`] with the specific disagreement, or
    /// [`EmulateError::NotTrapped`] if the instruction touches no trapped
    /// region at all.
    pub(crate) fn authenticate(
        &self,
        rip: u64,
        gpa: PhysAddr,
        cause: NestedPageFault,
    ) -> Result<Trapped, EmulateError> {
        let provenance = |reason| EmulateError::Provenance {
            rip,
            gpa: gpa.as_u64(),
            reason,
        };
        if cause.page_table_walk() {
            return Err(provenance(Provenance::PageTableWalk));
        }
        if !cause.final_address() {
            return Err(provenance(Provenance::NotFinal));
        }

        // A trapped read and a trapped write are answered by different halves of
        // a device's interface, so which one the hardware saw decides which end
        // of the move this exit belongs to. An instruction whose only interposed
        // end goes the other way is not the instruction that faulted.
        let wanted = if cause.write() {
            Direction::Write
        } else {
            Direction::Read
        };
        let (end, trapped) = match wanted {
            Direction::Read => (&self.from, Trapped::Source),
            Direction::Write => (&self.to, Trapped::Destination),
        };
        if end.interposed() {
            return if end.covers(gpa) {
                Ok(trapped)
            } else {
                Err(provenance(Provenance::Elsewhere))
            };
        }
        // The reported direction names an end that is ordinary memory. If the
        // *other* end is a device then the instruction does touch one, but not
        // through the access that faulted — so the two disagree about which
        // access this exit is. If neither end is, nothing here answers for the
        // address at all.
        let other = match wanted {
            Direction::Read => &self.to,
            Direction::Write => &self.from,
        };
        Err(if other.interposed() {
            provenance(Provenance::Direction {
                reported: wanted.name(),
                decoded: other.direction.name(),
            })
        } else {
            EmulateError::NotTrapped { gpa: gpa.as_u64() }
        })
    }

    /// The same plan with both ends advanced by one element, or `None` if
    /// either end would leave the page its translation was established for.
    ///
    /// Both or neither, deliberately. A plan with one end stepped and the other
    /// left behind would be two halves of different repetitions, so a step
    /// either carries the whole plan or ends the batch — and the caller
    /// answers `None` by returning to the guest with its progress in the
    /// index registers.
    pub(crate) fn stepped(&self, by: i64) -> Option<Self> {
        let from = Stride::new(self.from)?.step(by)?;
        let to = Stride::new(self.to)?.step(by)?;
        Some(Self {
            from: from.end(),
            to: to.end(),
        })
    }
}

/// An exception the guest owes, raised by an instruction this crate was
/// performing on its behalf.
///
/// Kept apart from [`EmulateError`] because the two mean opposite things. An
/// error is the hypervisor failing to do something; a fault is the hypervisor
/// succeeding at establishing that the guest's own instruction is not allowed
/// to complete. A demand-paged guest hits the second constantly, and reporting
/// it as the first stops a guest that a real processor would simply have
/// faulted into its own handler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fault {
    /// Which exception, as the architecture vectors it.
    vector: u8,
    /// The error code it is delivered with.
    code: u32,
    /// The address to publish in `CR2`, for the one exception that has one.
    address: Option<u64>,
}

impl Fault {
    /// A general-protection fault with a zero error code, which is what an
    /// alignment requirement the guest broke raises.
    #[must_use]
    pub const fn protection() -> Self {
        Self {
            vector: GENERAL_PROTECTION,
            code: 0,
            address: None,
        }
    }

    /// A page fault at this address, with this error code.
    ///
    /// The address is what the handler reads out of `CR2`, and is the linear
    /// address the access was aimed at rather than anything translated.
    #[must_use]
    pub const fn page(linear: u64, code: u32) -> Self {
        Self {
            vector: PAGE_FAULT,
            code,
            address: Some(linear),
        }
    }

    /// Which exception this is.
    #[must_use]
    pub const fn vector(&self) -> u8 {
        self.vector
    }

    /// The error code to deliver it with.
    #[must_use]
    pub const fn code(&self) -> u32 {
        self.code
    }

    /// The address to publish in `CR2`, or `None` for an exception that does
    /// not have one.
    #[must_use]
    pub const fn address(&self) -> Option<u64> {
        self.address
    }
}

/// Vector of the general-protection exception.
const GENERAL_PROTECTION: u8 = 13;
/// Vector of the page-fault exception.
const PAGE_FAULT: u8 = 14;

/// What the hardware said about the fault that caused this exit.
///
/// The two halves of the report travel together because neither means anything
/// without the other: an address with no cause does not say which access it
/// belongs to, and a cause with no address does not say where.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Reported {
    /// The guest physical address the processor reported.
    pub(crate) gpa: PhysAddr,
    /// What it said about the access.
    pub(crate) cause: NestedPageFault,
}

/// Which end of a move the hardware trapped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Trapped {
    /// The instruction's source is the device.
    Source,
    /// Its destination is.
    Destination,
}

/// One end of a repeated move, and the licence to step it without translating
/// again.
///
/// A repeated string instruction walks one element at a time, and re-resolving
/// the end from the index register each time costs a walk of the guest's page
/// tables per element per end — two thousand of them for a `rep movsb` over a
/// page, to answer a question whose answer cannot have changed.
///
/// It cannot have changed because of what bounds a batch. A batch stops as soon
/// as either end would leave the page it is on, so every element of one batch
/// lies in the same page as the first, and one page is described by one entry:
/// the guest physical address of element *n* is the address of element zero
/// plus *n* times the width, and which region answers for it is the region that
/// answers for the page.
///
/// So the translation is done once and stepped, and this type is what makes
/// that safe rather than merely fast — it exists only inside a batch, it is
/// rebuilt whenever a batch ends, and [`Stride::step`] refuses to leave the
/// page it was built for instead of silently walking past the evidence.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Stride {
    end: End,
    /// Which page the translation belongs to, as a page number. A step that
    /// would leave it is refused rather than trusted.
    page: u64,
}

impl Stride {
    /// The licence to step this end, if it is one that can be stepped.
    ///
    /// An end that is not in memory — the accumulator a store-string reads, say
    /// — has no address to walk and no translation to reuse, so it is
    /// carried unchanged and [`Stride::step`] leaves it alone.
    ///
    /// The element must lie wholly within one page for the invariant to hold.
    /// It does for every aligned element and for most unaligned ones, and
    /// where it does not, `None` sends the caller back to the per-element
    /// path rather than letting a stride describe an access whose second
    /// half is somewhere it never looked.
    pub(crate) fn new(end: End) -> Option<Self> {
        let Some(linear) = end.place.linear() else {
            return Some(Self { end, page: 0 });
        };
        let page = linear / PAGE;
        // The last byte of the element, which for a straddling one is on the next
        // page — and a stride whose own element spans two pages could not step
        // within one.
        (linear.checked_add(end.width.span() - 1)? / PAGE == page).then_some(Self { end, page })
    }

    /// The end as it stands now, with the address of the current element.
    pub(crate) const fn end(&self) -> End {
        self.end
    }

    /// The same end advanced by one element, or `None` if that leaves the page
    /// this translation was established for.
    ///
    /// `None` is not a failure: it is the batch's own stopping condition, and
    /// the caller answers it by returning to the guest with the progress so
    /// far in the index registers.
    pub(crate) fn step(mut self, by: i64) -> Option<Self> {
        // An end that is not in memory does not move at all. The accumulator a
        // store-string reads and a load-string writes is the same register for
        // every repetition, so there is nothing to advance and nothing that could
        // leave a page — and treating "no address" as "cannot step" would end every
        // STOS and LODS batch after one element.
        let Some(linear) = self.end.place.linear() else {
            return Some(self);
        };
        let linear = linear.checked_add_signed(by)?;
        if linear / PAGE != self.page {
            return None;
        }
        // Within the page the whole shape of the access is unchanged: the same
        // entry describes it, so the guest physical address moves by exactly what
        // the linear address moved by, and the region answering for it is the one
        // that answered for the previous element.
        self.end.place = match self.end.place {
            Place::Memory(_) => Place::Memory(linear),
            Place::Device {
                index, offset, gpa, ..
            } => Place::Device {
                index,
                offset: offset.checked_add_signed(by)?,
                gpa: PhysAddr::new(gpa.as_u64().checked_add_signed(by)?),
                linear,
            },
            // Ruled out above: these are exactly the places with no address.
            place @ (Place::Gpr(_) | Place::Vector(_) | Place::Immediate(_)) => place,
        };
        Some(self)
    }
}

/// The guest physical addresses a linear range maps to, if it maps to one
/// contiguous run of them.
///
/// This is what makes an access describable at all. A range that translates
/// contiguously can be one transaction against one thing; a range that does not
/// is two accesses against two things, and which two is not something the
/// instruction says.
///
/// # Errors
///
/// [`EmulateError::Span`] if the bytes do not translate to consecutive
/// addresses or the arithmetic leaves the address space, or
/// [`EmulateError::Memory`] if some byte of the range cannot be translated at
/// all.
pub(crate) fn contiguous(
    guest: &impl Guest,
    linear: u64,
    width: Width,
) -> Result<PhysAddr, EmulateError> {
    let span = |reason| EmulateError::Span {
        linear,
        bytes: width.bytes(),
        reason,
    };
    let end = linear
        .checked_add(width.span() - 1)
        .ok_or_else(|| span(Spanning::Wraps))?;
    let base = guest.translate(linear)?;
    // Only the bytes that could land elsewhere are checked, which for an access
    // inside one page is none of them: the first and last byte of a range within a
    // page translate through the same entry by construction, so the single-page
    // case costs exactly the one translation above and no comparison at all.
    //
    // Where the range does cross a boundary, the walk starts at the *second* page.
    // Checking the first would re-translate the page `base` already came from and
    // compare it against itself — a guaranteed-true answer bought with a full page
    // walk. Every page after it is checked rather than only the last, because a
    // three-page range can be contiguous at both ends and not in the middle.
    //
    // Counted in pages rather than in addresses, because a range that ends at the
    // top of the address space has no "next page" address to compute: one past its
    // last page overflows, where one past its last page *number* is simply outside
    // the loop.
    for page in (linear / PAGE + 1)..=(end / PAGE) {
        let at = page * PAGE;
        if guest.translate(at)? != base + (at - linear) {
            return Err(span(Spanning::Discontiguous));
        }
    }
    Ok(base)
}

/// Where the guest resumes after an instruction of this length, at the width
/// the mode makes the instruction pointer.
///
/// A 16-bit guest's pointer is `IP` and a 32-bit guest's is `EIP`, and neither
/// carries into the half above it: an instruction ending at `0xFFFF` in real
/// mode resumes at zero, not at `0x10000`. Adding at sixty-four bits and
/// storing the result is correct only in long mode, and silently plausible
/// everywhere else — the guest carries on executing at an address one wrap away
/// from where it should be.
///
/// `None` if the length is not one an instruction can have, which is a decode
/// that went wrong rather than an address to compute.
pub(crate) fn after(save: &SaveArea, length: usize) -> Option<u64> {
    /// Bytes in the longest instruction the architecture allows.
    const LONGEST: usize = 15;
    if length == 0 || length > LONGEST {
        return None;
    }
    let addressing = Addressing::from_save(save);
    let next = save.rip.wrapping_add(as_u64(length));
    Some(match pointer(&addressing) {
        // The whole register, and the only case where a carry out of the low
        // half is a real carry.
        Width::Quad => next,
        // The low half of the register, with the upper half of the *old* value
        // kept: the architecture wraps the pointer within its width rather than
        // widening it.
        width => (save.rip & !width.mask()) | (next & width.mask()),
    })
}

/// How wide the instruction pointer is in this mode.
///
/// Not the operand size and not the address size: the pointer's own width,
/// which is what an instruction's length is added to.
fn pointer(addressing: &Addressing) -> Width {
    if addressing.long_mode() {
        return Width::Quad;
    }
    match addressing.mode() {
        // Unpaged is real mode or protected mode without paging, and which one
        // it is is the code segment's default size rather than the paging bit.
        Mode::Unpaged | Mode::Legacy | Mode::Pae | Mode::Long | Mode::FiveLevel => {
            if addressing.default_size() {
                Width::Long
            } else {
                Width::Word
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use iced_x86::Register;
    use svm::{SaveArea, SegmentAttributes};
    use x86_64::PhysAddr;

    use super::{Direction, End, PAGE, Stride, after, contiguous};
    use crate::{
        EmulateError, Spanning,
        machine::tests::{Machine, Memory},
        operand::Place,
        value::Width,
    };

    /// A save area in one of the three modes an instruction pointer has a width
    /// in.
    fn saved(long: bool, default_size: bool, rip: u64) -> SaveArea {
        let mut save = SaveArea::zeroed();
        save.cs.attributes = SegmentAttributes::new()
            .with_long(long)
            .with_default_size(default_size);
        save.rip = rip;
        save
    }

    #[test]
    fn a_long_mode_pointer_advances_through_the_whole_register() {
        assert_eq!(after(&saved(true, false, 0x1000), 5), Some(0x1005));
        // Long mode is the one width where a carry out of the low half is a
        // carry rather than a wrap.
        assert_eq!(
            after(&saved(true, false, 0xFFFF_FFFF), 2),
            Some(0x1_0000_0001)
        );
        assert_eq!(after(&saved(true, false, 0xFFFF), 2), Some(0x1_0001));
    }

    #[test]
    fn a_sixteen_bit_pointer_wraps_at_sixteen_bits() {
        // IP 0xFFFE plus a two-byte instruction resumes at zero, not 0x10000.
        assert_eq!(after(&saved(false, false, 0xFFFE), 2), Some(0));
        assert_eq!(after(&saved(false, false, 0xFFFF), 2), Some(1));
        // And the upper bits of the register are preserved rather than cleared,
        // which is what wrapping within a width means.
        assert_eq!(
            after(&saved(false, false, 0xDEAD_0000_FFFE), 2),
            Some(0xDEAD_0000_0000)
        );
    }

    #[test]
    fn a_thirty_two_bit_pointer_wraps_at_thirty_two_bits() {
        assert_eq!(after(&saved(false, true, 0xFFFF_FFFE), 2), Some(0));
        assert_eq!(after(&saved(false, true, 0xFFFF_FFFF), 3), Some(2));
        assert_eq!(
            after(&saved(false, true, 0xDEAD_FFFF_FFFE), 2),
            Some(0xDEAD_0000_0000)
        );
    }

    #[test]
    fn no_length_an_instruction_cannot_have_produces_an_address() {
        for length in [0, 16, 17, 100] {
            assert_eq!(
                after(&saved(true, false, 0x1000), length),
                None,
                "{length} is not a length an instruction has"
            );
        }
        for length in 1..=15 {
            assert!(after(&saved(true, false, 0x1000), length).is_some());
        }
    }

    #[test]
    fn an_access_inside_one_page_is_contiguous_without_asking_twice() {
        let machine = Machine::long_mode();
        let mut memory = Memory::new(&machine);
        memory.map(0x1000);
        let base = contiguous(&memory, 0x1000, Width::Quad).expect("one page is contiguous");
        assert_eq!(base.as_u64(), memory.gpa_of(0x1000));
    }

    #[test]
    fn an_access_crossing_into_a_page_that_does_not_follow_is_refused() {
        // The fake scatters pages deliberately, so two neighbouring linear pages
        // are not neighbouring physical ones — which is the case a real guest
        // can arrange and which byte-zero classification gets wrong.
        let machine = Machine::long_mode();
        let mut memory = Memory::new(&machine);
        memory.map(0x1000).map(0x2000);
        let error = contiguous(&memory, 0x1FFE, Width::Long)
            .expect_err("a span across scattered pages is not one access");
        assert!(matches!(
            error,
            EmulateError::Span {
                reason: Spanning::Discontiguous,
                ..
            }
        ));
    }

    #[test]
    fn an_access_crossing_into_a_page_that_does_follow_is_allowed() {
        let machine = Machine::long_mode();
        let mut memory = Memory::new(&machine);
        // Placed so that the second page really is the physical page after the
        // first, which is the only shape a single transaction can describe.
        memory.map_at(0x1000, 0x8000).map_at(0x2000, 0x8000 + PAGE);
        let base = contiguous(&memory, 0x1FFC, Width::Quad).expect("physically contiguous");
        assert_eq!(base.as_u64(), 0x8FFC);
    }

    #[test]
    fn an_access_whose_later_bytes_are_not_described_reports_the_translation_failure() {
        let machine = Machine::long_mode();
        let mut memory = Memory::new(&machine);
        memory.map(0x1000);
        let error =
            contiguous(&memory, 0xFFE, Width::Long).expect_err("the second page is not described");
        assert!(
            matches!(error, EmulateError::Memory(_)),
            "an undescribed page is a translation failure, not a span failure: {error:?}"
        );
    }

    #[test]
    fn an_access_that_leaves_the_address_space_is_refused_before_translating() {
        let machine = Machine::long_mode();
        let memory = Memory::new(&machine);
        let error = contiguous(&memory, u64::MAX - 2, Width::Quad)
            .expect_err("the arithmetic leaves the address space");
        assert!(matches!(
            error,
            EmulateError::Span {
                reason: Spanning::Wraps,
                ..
            }
        ));
    }

    #[test]
    fn a_direction_names_itself_for_a_diagnostic() {
        assert_eq!(Direction::Read.name(), "read");
        assert_eq!(Direction::Write.name(), "write");
    }

    /// One end in the guest's memory at a linear address, as a batch would have
    /// resolved it.
    fn memory(linear: u64, width: Width) -> End {
        End {
            place: Place::Memory(linear),
            width,
            direction: Direction::Read,
        }
    }

    /// One end in a device region, translating to `gpa` at `offset` into it.
    fn device(linear: u64, gpa: u64, offset: u64, width: Width) -> End {
        End {
            place: Place::Device {
                index: 0,
                offset,
                gpa: PhysAddr::new(gpa),
                linear,
            },
            width,
            direction: Direction::Write,
        }
    }

    #[test]
    fn stepping_moves_a_memory_end_by_one_element() {
        let stride = Stride::new(memory(0x1000, Width::Long)).expect("inside one page");
        let stepped = stride.step(4).expect("still inside the page");
        assert_eq!(stepped.end().place, Place::Memory(0x1004));
    }

    #[test]
    fn stepping_a_device_end_moves_its_address_and_its_offset_together() {
        // The whole point of stepping rather than re-resolving: within one page the
        // guest physical address moves by exactly what the linear address moved by,
        // and the region answering for it cannot have changed. An offset that
        // drifted from the address would send the device a transaction for the
        // wrong register.
        let stride =
            Stride::new(device(0x8000, 0xFEE0_0000, 0, Width::Long)).expect("inside one page");
        let stepped = stride.step(4).expect("still inside the page");
        assert_eq!(
            stepped.end().place,
            Place::Device {
                index: 0,
                offset: 4,
                gpa: PhysAddr::new(0xFEE0_0004),
                linear: 0x8004,
            }
        );
    }

    #[test]
    fn stepping_backwards_walks_down_within_the_page() {
        let stride =
            Stride::new(device(0x8010, 0xFEE0_0010, 0x10, Width::Long)).expect("inside one page");
        let stepped = stride.step(-4).expect("still inside the page");
        assert_eq!(
            stepped.end().place,
            Place::Device {
                index: 0,
                offset: 0xC,
                gpa: PhysAddr::new(0xFEE0_000C),
                linear: 0x800C,
            }
        );
    }

    #[test]
    fn a_step_that_would_leave_the_page_ends_the_batch_instead() {
        // The invariant the whole optimization rests on. One page is one entry, so
        // a translation may be reused within it and nowhere else — the next page
        // may translate somewhere unrelated, be answered for by a different device,
        // or not be described at all.
        let last = PAGE - 4;
        let stride = Stride::new(memory(last, Width::Long)).expect("inside one page");
        assert_eq!(
            stride.step(4).map(|stepped| stepped.end().place),
            None,
            "a step onto the next page must end the batch rather than be assumed"
        );
        // And the same going the other way, off the bottom of the page.
        let stride = Stride::new(memory(PAGE, Width::Long)).expect("inside one page");
        assert!(stride.step(-4).is_none());
    }

    #[test]
    fn an_end_that_is_not_in_memory_steps_without_moving() {
        // The accumulator a store-string reads is the same register for every
        // repetition. Treating "no address" as "cannot step" would end every STOS
        // and LODS batch after a single element.
        let end = End {
            place: Place::Gpr(Register::EAX),
            width: Width::Long,
            direction: Direction::Read,
        };
        let stride = Stride::new(end).expect("a register end is steppable");
        let stepped = stride.step(4).expect("a register end never leaves a page");
        assert_eq!(stepped.end().place, Place::Gpr(Register::EAX));
    }

    #[test]
    fn an_element_that_straddles_a_page_is_not_given_a_stride() {
        // Such an element was translated as a span across two entries, and no
        // single-page licence describes it — so it goes back through the full
        // per-element path rather than being stepped from evidence that covers
        // only half of it.
        assert!(Stride::new(memory(PAGE - 2, Width::Long)).is_none());
        assert!(Stride::new(memory(PAGE - 1, Width::Word)).is_none());
        // One that ends exactly at the boundary is still within its page.
        assert!(Stride::new(memory(PAGE - 4, Width::Long)).is_some());
    }

    #[test]
    fn a_step_near_the_top_of_the_address_space_does_not_wrap_into_range() {
        let stride = Stride::new(memory(u64::MAX - 3, Width::Long));
        assert!(
            stride.is_none_or(|stride| stride.step(4).is_none()),
            "an access at the very top must not wrap round to zero and look valid"
        );
    }
}
