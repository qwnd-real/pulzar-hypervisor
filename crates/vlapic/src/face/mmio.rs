//! The face a guest reaches its controller through when it is memory mapped.
//!
//! One page of guest physical memory, at the same address on every processor
//! and each of them seeing its own controller through it — which is why the
//! device registered for it is one device holding a table of controllers, and
//! picks this processor's row out of it on each access rather than there being
//! one device per processor.
//!
//! # Two ways an access gets here, and one of them is not a fault
//!
//! Where this hypervisor serves the page itself, the page is trapped in the
//! nested page tables and every access to it faults into the emulator, which
//! performs it against the device below.
//!
//! Where the processor serves the page instead — the acceleration driving the
//! controller out of a backing page of its own — the page is not trapped, the
//! guest's accesses never fault, and most of them never reach this hypervisor
//! at all. The ones the acceleration will not perform are reported as an exit
//! that names the offset and the direction, and the exit path performs them
//! against this same device. So what arrives here is a subset rather than
//! something different, and nothing below needs to know which of the two
//! brought it.
//!
//! # This face exists only while the controller is in the older mode
//!
//! Neither route is taken away when the mode that has a register page is. The
//! trap is installed once, before any guest runs, and cannot be removed while
//! processors are executing; the acceleration's own redirection likewise
//! outlives the mode it belongs to. So the aperture outlives the mode either
//! way, and everything reaching it is gated on the controller actually being in
//! that mode, before an offset is decoded or an error recorded. A guest that
//! has switched its controller off, or moved it to the model-specific
//! registers, must not find a second way to reach the same registers.
//!
//! What it finds instead is what an address nothing decodes answers with, which
//! is all-ones: outside that mode the page is not claimed by anything, so the
//! guest's load reaches a bus where nothing answers. Zero would be a device
//! standing where none is, and it inverts every test software makes to find out
//! whether a controller is there.
//!
//! # What this face does that the other does not
//!
//! Nothing here can fault. Reaching a reserved address through the page is not
//! an exception; it records the illegal-register-address error and the access
//! otherwise does nothing. Writing a read-only register does nothing. Reading a
//! write-only one answers zero.
//!
//! That is the whole of the difference in behaviour, and it is why this module
//! is thin: what a register *means* is [`crate::face::dispatch`]'s, and only
//! what a malformed access does is decided here.
//!
//! # Widths
//!
//! Every register is 32 bits on a 128-bit boundary and the architecture
//! requires software to reach it with an aligned 32-bit access. Anything else
//! is undefined on real hardware, so there is no behaviour to reproduce — but
//! "undefined" is not "fatal", and a guest that makes such an access is
//! entitled to go on executing. So every width is accepted and only the
//! four-byte one names a register: a narrower, wider or sixteen-byte access
//! reads zero and is discarded on the way in, with nothing recorded. It is
//! deliberately *not* answered as a reserved address, because the guest named a
//! perfectly good one and the error the architecture defines for a reserved
//! address would be an error about something that did not happen.
//!
//! The width is judged after the mode, because a width is a question about a
//! register and outside that mode there are none: every width reads all-ones
//! there, as every width would from an address nothing claims.
//!
//! Two shapes never reach this module at all, because the emulator cannot make
//! the transaction they would need: an access straddling the end of the page,
//! and a locked read-modify-write of a register. Those are dropped and stepped
//! over by the exit path, which is the same disposition by another route.

use alloc::boxed::Box;

use emulate::{Capability, Commit, Data, Device, Hardware, Read, Region, Trap, Width, Write};
use log::trace;
use x86_64::PhysAddr;

use crate::{
    VlapicError,
    face::{
        dispatch,
        table::{Access, PAGE, Register},
    },
    machine::registry::{Page, lapics},
    registers::{
        Vlapic,
        base::{ApicBase, Mode},
    },
};

/// The region of the guest's memory this crate answers for.
///
/// Handed to whatever traps regions before the guest runs, and handed over
/// whether or not the guest's own accesses are what reach it.
///
/// Where the software model serves the page, every access is trapped, reads
/// included: the values a guest reads out of its controller are this crate's
/// answers and never the hardware's. Where the processor serves the page
/// itself, out of the backing page provisioning built, the nested tables are
/// left alone and no access of the guest's arrives here — but the device is
/// still owed, because the accesses the acceleration declines to perform come
/// back as an exit naming the address and the direction, and performing one of
/// those is performing it against this device.
///
/// Which of the two it is follows from whether the acceleration was
/// provisioned, so this is asked after [`crate::provision`] rather than before.
///
/// # Errors
///
/// [`VlapicError::NotInstalled`] before [`crate::install`].
pub fn region() -> Result<Region, VlapicError> {
    let page = lapics()?;
    Ok(Region {
        gpa: PhysAddr::new(ApicBase::DEFAULT_PAGE),
        bytes: PAGE,
        trap: (!crate::avic::activation::provisioned()).then_some(Trap::Everything),
        device: Box::new(Aperture(page)),
    })
}

/// What answers for the guest's register page.
///
/// A thin wrapper rather than the controllers themselves, because a device is
/// handed over as an owned trait object and the same controllers are reached
/// from elsewhere for delivery.
#[derive(Debug)]
struct Aperture(&'static Page);

impl Device for Aperture {
    fn capability(&self) -> Capability {
        self.0.capability()
    }

    fn hardware(&self) -> Hardware {
        self.0.hardware()
    }

    fn read(&self, access: Read<'_>) -> Data {
        self.0.read(access)
    }

    fn write(&self, access: Write<'_>) -> Commit {
        self.0.write(access)
    }
}

impl Device for Page {
    /// Every access the emulator can make, and the one that names a register is
    /// the aligned 32-bit one.
    ///
    /// Every register of the controller is 32 bits on a 128-bit boundary, and
    /// the architecture requires software to reach one with an aligned 32-bit
    /// access. Anything else is undefined on real hardware — but a declaration
    /// here is not "ignore the others", it is "refuse the transaction", and a
    /// refused transaction is one the exit path cannot perform for a guest that
    /// is entitled to go on running. So the widths are admitted and
    /// [`Page::decode`] is what declines to name a register for them.
    ///
    /// Sixteen bytes are admitted as two eight-byte transactions, which for
    /// this device cannot matter: nothing behind the page is ever reached,
    /// so there is no bus transaction for a split to be visible in.
    fn capability(&self) -> Capability {
        Capability::scalar().split_vectors()
    }

    /// Nothing behind this page is ever touched.
    ///
    /// The whole point of the device is that the guest must not reach the real
    /// controller: every read is answered out of the emulated register file and
    /// every write is discarded. A mapping of the real controller would be a
    /// standing writable alias of its acknowledge, command and priority
    /// registers at a host address nothing uses.
    fn hardware(&self) -> Hardware {
        Hardware::Untouched
    }

    fn read(&self, access: Read<'_>) -> Data {
        let width = access.width();
        let (vlapic, register) = match self.decode(access.offset(), width) {
            Decoded::Register(vlapic, register) => (vlapic, register),
            // The page is there and the guest named nothing in it.
            Decoded::NoRegister => return Data::from_u64(0, width),
            // There is no page. Answered as an address nothing claims answers,
            // because that is what this one is when the controller is not in the
            // mode that has it.
            Decoded::NoPage => return unclaimed(width),
        };
        // A write-only register answers zero rather than recording an error:
        // the architecture defines no error for reading one through this face,
        // and inventing one would be a guest told about something that did not
        // happen.
        let value = match Access::of(register, vlapic.mode(), vlapic.model()) {
            Access::WriteOnly | Access::Absent => 0,
            Access::ReadOnly | Access::ReadWrite => register_value(vlapic, register),
        };
        trace!(
            "vlapic: cpu {} read its {:#05x} register through the page and got {:#010x}",
            vlapic.index(),
            register.offset(),
            value
        );
        Data::from_u64(u64::from(value), width)
    }

    fn write(&self, access: Write<'_>) -> Commit {
        let Decoded::Register(vlapic, register) = self.decode(access.offset(), access.width())
        else {
            return Commit::Discard;
        };
        if !matches!(
            Access::of(register, vlapic.mode(), vlapic.model()),
            Access::ReadWrite | Access::WriteOnly
        ) {
            // Writing a read-only register through this face does nothing and
            // is not an error the architecture reports.
            return Commit::Discard;
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the access was established to be exactly four bytes wide by `decode`"
        )]
        let value = access.value().as_u64() as u32;
        trace!(
            "vlapic: cpu {} wrote {:#010x} to its {:#05x} register through the page",
            vlapic.index(),
            value,
            register.offset()
        );
        // While the hardware drives the controller, the one register a guest
        // writes without exiting lives in the backing page: a write that
        // still reaches this face goes where the hardware reads it, and the
        // model follows at whichever transition carries the state back.
        if register == Register::TASK_PRIORITY
            && crate::avic::activation::active_for(vlapic)
            && crate::avic::activation::backing_write_task_priority(value).is_ok()
        {
            return Commit::Discard;
        }
        dispatch::acted(vlapic, dispatch::write(vlapic, register, value));
        // Nothing the guest writes here reaches the hardware behind the page.
        // The page a guest sees is this hypervisor's answer, and the real
        // controller is programmed deliberately and separately from it.
        Commit::Discard
    }
}

impl Page {
    /// Whose controller an access is for and which register it names, or why it
    /// names none.
    ///
    /// The face itself is gated before any register is decoded, and that comes
    /// first for a reason. How this aperture is reached is settled once, before
    /// any guest runs, and stays settled for the life of the machine — but the
    /// registers behind it exist only while the controller is in the older
    /// mode. A guest that has switched its controller off, or moved it to
    /// the model-specific registers, has no memory-mapped face at all, and
    /// one that still answered would be two programming interfaces to one
    /// controller at once: a globally disabled guest could go on sending
    /// interprocessor interrupts, acknowledging real hardware and
    /// reprogramming physical sources through a page the architecture says
    /// is not there.
    ///
    /// Nothing is recorded for an access outside that mode either. The
    /// illegal-register-address error belongs to a controller that has a
    /// register page and was given a bad offset in it; a controller with no
    /// page has not been given a bad offset, and inventing the error would
    /// tell the guest about a fault in a face it is not using.
    ///
    /// A controller that cannot be found at all is answered the same way, and
    /// for the same reason: the roster this array was built from is where the
    /// index came from, so a miss is a broken invariant rather than anything a
    /// guest did — and either way there is no controller behind this page for
    /// the processor asking, which is exactly what having no page means.
    ///
    /// The two ways of being malformed *within* the face are kept apart,
    /// because the architecture describes only one of them. An offset no
    /// register sits at — including one for a register this controller does
    /// not have in this mode — *is* an illegal register address, and is
    /// recorded as one. A width the architecture leaves undefined is not:
    /// the address was legal, so a guest told its address was reserved
    /// would be told about something that did not happen.
    fn decode(&self, offset: u64, width: Width) -> Decoded<'_> {
        let Some(vlapic) = self.current() else {
            return Decoded::NoPage;
        };
        let mode = vlapic.mode();
        if mode != Mode::XApic {
            return Decoded::NoPage;
        }
        match named(offset, width) {
            Named::Register(register)
                if Access::of(register, mode, vlapic.model()) != Access::Absent =>
            {
                Decoded::Register(vlapic, register)
            }
            Named::Undefined => Decoded::NoRegister,
            // An offset no register sits at, and one whose register this
            // controller does not have in the mode it is in, are the same
            // reserved address as far as a guest is concerned.
            Named::Register(_) | Named::Reserved => {
                dispatch::illegal_register(vlapic, offset);
                Decoded::NoRegister
            }
        }
    }
}

/// What an access to this page reaches.
///
/// Three answers rather than two, because a guest that named nothing inside a
/// page that is there and one whose controller has no page at all are owed
/// different values: the first is a register file that holds nothing at that
/// address, and the second is an address on a bus that nothing claims.
#[derive(Debug)]
enum Decoded<'a> {
    /// The controller the access is for, and the register it names.
    Register(&'a Vlapic, Register),
    /// The page is there and no register in it was named — a reserved address,
    /// which has been recorded, or a width the architecture leaves undefined,
    /// which has not.
    NoRegister,
    /// There is no memory-mapped face at all, because the controller is not in
    /// the mode that has one.
    NoPage,
}

/// What an address nothing claims answers a read with.
///
/// All-ones, at whatever width the guest read it. That is what a load from an
/// unclaimed physical address gives on the machines this runs on, and what the
/// reference implementation of this controller answers for its own page once
/// the mode that decodes it is gone.
fn unclaimed(width: Width) -> Data {
    match width {
        // Sixteen bytes is wider than one value and every byte of the answer has
        // to be filled, because a device asked at a width must answer at it.
        Width::Vector => Data::vector_from([u8::MAX; VECTOR_BYTES]),
        width => Data::from_u64(u64::MAX, width),
    }
}

/// How many bytes the widest access the emulator makes carries.
const VECTOR_BYTES: usize = Width::Vector.bytes();

/// What a register answers with, taken from wherever the register lives.
///
/// While the hardware drives the controller, the registers it serves itself
/// are the backing page's — the task priority the guest set without exiting,
/// the priorities the hardware computes, and the three banks it moves
/// vectors between — and a read of one of them must not be answered out of a
/// model the hardware has not been consulting. Everything else is the
/// model's in both worlds: the hardware completes those writes into the page
/// and exits, and the trap's bookkeeping carries the value across.
///
/// A backing page that cannot be reached falls back to the model's answer
/// rather than to a value invented here: a guest reading its controller is
/// owed an answer, and the model's is the nearest true one.
fn register_value(vlapic: &Vlapic, register: Register) -> u32 {
    let hardware_owned = matches!(
        register,
        Register::TASK_PRIORITY | Register::ARBITRATION_PRIORITY | Register::PROCESSOR_PRIORITY
    ) || register.bank().is_some();
    if hardware_owned
        && crate::avic::activation::active_for(vlapic)
        && let Ok(value) = crate::avic::activation::backing_read(register)
    {
        return value;
    }
    dispatch::read(vlapic, register)
}

/// What the shape of an access to this page names.
///
/// Shape alone: the offset and the width, with nothing said about which
/// controller is being reached or what it has. Kept apart from [`Page::decode`]
/// because it is the whole of the malformed-access policy and the only part of
/// it that can be checked without a machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Named {
    /// This register.
    Register(Register),
    /// Nothing, because the width is one the architecture leaves undefined
    /// here. Not an error it reports: the address the guest named was a
    /// perfectly good one.
    Undefined,
    /// Nothing, because no register sits at that offset — which is the one
    /// thing the architecture does call an illegal register address.
    Reserved,
}

/// Which register an access of this shape names, if any.
///
/// The width is judged first and is judged separately, because the two ways of
/// naming nothing are different answers to the guest. A four-byte access is the
/// only one that reaches a register at all — every register is 32 bits, and
/// this is also what stops a sixteen-byte write being applied as a register
/// write of the low four bytes of a vector store — and an access of any other
/// width is undefined rather than misaddressed.
const fn named(offset: u64, width: Width) -> Named {
    if !matches!(width, Width::Long) {
        return Named::Undefined;
    }
    match Register::at(offset) {
        Some(register) => Named::Register(register),
        None => Named::Reserved,
    }
}

#[cfg(test)]
mod tests {
    //! What a controller does with a well-formed access is
    //! [`crate::face::dispatch`]'s and is tested there. What is here is the
    //! whole of what this face decides: which accesses reach it at all, and
    //! what a malformed one names.
    //!
    //! A controller cannot be built on a host — one is made from a roster
    //! entry, and a roster comes from firmware's tables — so the decisions
    //! are checked where they are made rather than through `read` and
    //! `write`.

    use alloc::boxed::Box;

    use emulate::{Device, Hardware, Width};

    use super::{Named, Page, named, unclaimed};
    use crate::face::table::Register;

    /// The device, with no controllers behind it: every declaration it makes is
    /// about the page rather than about any one processor's controller.
    fn page() -> Page {
        Page::new(Box::new([]))
    }

    /// Every width a guest can reach this page with. Spelled out here because
    /// the emulator's own list of them is not public, and a width missing from
    /// this one is a width nothing below asserts anything about.
    const WIDTHS: [Width; 5] = [
        Width::Byte,
        Width::Word,
        Width::Long,
        Width::Quad,
        Width::Vector,
    ];

    #[test]
    fn every_access_the_emulator_can_make_is_admitted() {
        // Not a preference. A width this device refuses is a transaction the
        // exit path cannot perform, and the guest instruction that made it is one
        // the architecture merely leaves undefined — so refusing here would end a
        // guest, and with it a physical processor, over `mov al, [0xFEE00030]`.
        let capability = page().capability();
        for width in WIDTHS {
            assert!(
                capability.answers(width),
                "{width:?} has to reach the device to be answered by it"
            );
        }
    }

    #[test]
    fn nothing_behind_the_page_is_ever_reached() {
        // The declaration that stops the real controller's registers being
        // mapped writable into host address space for the life of the machine.
        assert_eq!(page().hardware(), Hardware::Untouched);
    }

    #[test]
    fn only_a_four_byte_access_names_a_register() {
        assert_eq!(named(0x20, Width::Long), Named::Register(Register::ID));
        for width in WIDTHS.into_iter().filter(|width| *width != Width::Long) {
            assert_eq!(
                named(0x20, width),
                Named::Undefined,
                "a {width:?} access at a perfectly good offset is undefined, not misaddressed"
            );
        }
    }

    #[test]
    fn an_offset_no_register_sits_at_is_a_reserved_address() {
        for offset in [
            // Four-byte aligned, but not on the 128-bit boundary a register sits
            // on: the last dword of the version register's slot.
            0x34,
            // The last dword of the page, which is aligned and still names
            // nothing.
            0xFFC, // Past the end of the page altogether.
            0x1000,
        ] {
            assert_eq!(
                named(offset, Width::Long),
                Named::Reserved,
                "{offset:#x} names no register"
            );
        }
    }

    #[test]
    fn a_reserved_offset_at_an_undefined_width_is_still_only_undefined() {
        // The distinction the error status register depends on: the
        // illegal-register-address bit is about an address, and a guest whose
        // width was wrong has not been told its address was reserved. The width
        // is decided first, so this stays `Undefined` however bad the offset is.
        assert_eq!(named(0x1000, Width::Byte), Named::Undefined);
        assert_eq!(named(0x34, Width::Vector), Named::Undefined);
    }

    #[test]
    fn an_address_nothing_claims_reads_as_all_ones() {
        // What a controller not in the older mode answers with, at every width
        // the emulator can ask about. Zero would be a device standing where none
        // is, and it is the answer software tests for when it wants to know
        // whether this page is decoded at all.
        for width in WIDTHS {
            let answer = unclaimed(width);
            assert_eq!(answer.width(), width);
            assert!(
                answer.bytes().iter().all(|byte| *byte == u8::MAX),
                "{width:?} answered {:?}",
                answer.bytes()
            );
        }
    }
}
