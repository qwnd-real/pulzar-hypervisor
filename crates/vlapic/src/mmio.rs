//! The face a guest reaches its controller through when it is memory mapped.
//!
//! One page of guest physical memory, trapped in the nested page tables, whose
//! every access comes back here. The page is at the same address on every
//! processor and each of them sees its own controller through it — which is why
//! the device registered for it is one device holding a table of controllers,
//! and picks this processor's row out of it on each access rather than there
//! being one device per processor.
//!
//! # This face exists only while the controller is in the older mode
//!
//! The page is trapped once, before any guest runs, and cannot be untrapped
//! while processors are executing — so the aperture outlives the mode it
//! belongs to. Everything reaching it is therefore gated on the controller
//! actually being in that mode, before an offset is decoded or an error
//! recorded. A guest that has switched its controller off, or moved it to the
//! model-specific registers, must not find a second way to reach the same
//! registers.
//!
//! # What this face does that the other does not
//!
//! Nothing here can fault. Reaching a reserved address through the page is not
//! an exception; it records the illegal-register-address error and the access
//! otherwise does nothing. Writing a read-only register does nothing. Reading a
//! write-only one answers zero.
//!
//! That is the whole of the difference in behaviour, and it is why this module
//! is thin: what a register *means* is [`crate::access`]'s, and only what a
//! malformed access does is decided here.
//!
//! # Widths
//!
//! Every register is 32 bits on a 128-bit boundary and the architecture
//! requires software to reach it with an aligned 32-bit access. Anything else
//! is undefined on real hardware, which means there is no behaviour to
//! reproduce — so a narrower or wider access is answered as an access to a
//! reserved address, which is the closest defined thing and is at least
//! consistent.

use alloc::boxed::Box;

use emulate::{Capability, Commit, Data, Device, Read, Width, Write};
use log::info;

use crate::{
    access,
    base::Mode,
    register::{Access, Register},
    state::Vlapic,
};

/// Every processor's controller, answering for the one page they share.
#[derive(Debug)]
pub(crate) struct Page {
    lapics: Box<[Vlapic]>,
}

impl Page {
    /// One controller per processor in the roster.
    pub(crate) fn new(lapics: Box<[Vlapic]>) -> Self {
        Self { lapics }
    }

    /// Every controller, for the paths that walk them: delivering to a set of
    /// processors, and reporting.
    pub(crate) fn all(&self) -> &[Vlapic] {
        &self.lapics
    }

    /// The controller belonging to the processor asking.
    ///
    /// An access to this page is made by the guest running on one processor,
    /// and the controller it means is that processor's. Nothing else would be a
    /// sensible answer: the address carries no processor in it precisely
    /// because the hardware it stands for is per-processor.
    fn current(&self) -> Option<&Vlapic> {
        // SAFETY: the guest cannot have reached this page before the processor
        // running it attached — attaching happens during bring-up, long before
        // any guest is entered — so this processor's `GS` base points at its
        // own block.
        let index = unsafe { cpu::current() }.index();
        self.lapics.get(index.get())
    }
}

impl Device for Page {
    /// Aligned 32-bit accesses and nothing else.
    ///
    /// Every register of the controller is 32 bits on a 128-bit boundary, and
    /// the architecture requires software to reach one with an aligned
    /// 32-bit access. Anything else is undefined on real hardware, so there
    /// is no behaviour to reproduce — and declaring it here means such an
    /// access is refused before this device is asked about it, rather than
    /// being decoded into an illegal-register error that claims the guest
    /// named a bad offset when what it really did was use the wrong width.
    fn capability(&self) -> Capability {
        Capability::only(Width::Long)
    }

    fn read(&self, access: Read<'_>) -> Data {
        let width = access.width();
        let Some((vlapic, register)) = self.decode(access.offset(), width) else {
            return Data::from_u64(0, width);
        };
        // A write-only register answers zero rather than recording an error:
        // the architecture defines no error for reading one through this face,
        // and inventing one would be a guest told about something that did not
        // happen.
        let value = match Access::of(register, vlapic.mode(), vlapic.model()) {
            Access::WriteOnly | Access::Absent => 0,
            Access::ReadOnly | Access::ReadWrite => access::read(vlapic, register),
        };
        info!(
            "vlapic: cpu {} read its {:#05x} register through the page and got {:#010x}",
            vlapic.index(),
            register.offset(),
            value
        );
        Data::from_u64(u64::from(value), width)
    }

    fn write(&self, access: Write<'_>) -> Commit {
        let Some((vlapic, register)) = self.decode(access.offset(), access.width()) else {
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
        info!(
            "vlapic: cpu {} wrote {:#010x} to its {:#05x} register through the page",
            vlapic.index(),
            value,
            register.offset()
        );
        crate::acted(vlapic, access::write(vlapic, register, value));
        // Nothing the guest writes here reaches the hardware behind the page.
        // The page a guest sees is this hypervisor's answer, and the real
        // controller is programmed deliberately and separately from it.
        Commit::Discard
    }
}

impl Page {
    /// Whose controller an access is for and which register it names, or `None`
    /// if this face answers for it at all.
    ///
    /// The face itself is gated before any register is decoded, and that comes
    /// first for a reason. This aperture is trapped once, before any guest
    /// runs, and stays trapped for the life of the machine — but the
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
    /// A malformed access within the face — one that is not four bytes, or not
    /// on a 128-bit boundary, or at an offset no register sits at — is one the
    /// architecture leaves undefined or explicitly calls an illegal register
    /// address. Both are answered the same way, because the error the guest is
    /// entitled to be told about is the same one.
    fn decode(&self, offset: u64, width: Width) -> Option<(&Vlapic, Register)> {
        let vlapic = self.current()?;
        if vlapic.mode() != Mode::XApic {
            return None;
        }
        let register = (width == Width::Long)
            .then(|| Register::at(offset))
            .flatten()
            .filter(|register| {
                Access::of(*register, vlapic.mode(), vlapic.model()) != Access::Absent
            });
        if register.is_none() {
            access::illegal_register(vlapic, Register::at(offset));
        }
        register.map(|register| (vlapic, register))
    }
}
