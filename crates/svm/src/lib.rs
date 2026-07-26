//! The structures AMD's secure virtual machine extension is programmed
//! through, stated once and checked by the compiler.
//!
//! A hypervisor talks to this extension almost entirely through memory. One
//! page describes a guest completely — what to intercept, what state to run it
//! with, and what it did to come back — and the processor reads that page by
//! offset. Nothing checks the offsets for us: a field written four bytes from
//! where the architecture expects it does not fail, it runs the guest with
//! somebody else's control register and fails much later, somewhere else. That
//! failure mode is the reason this crate exists and the reason it is shaped the
//! way it is.
//!
//! So every structure here is laid out to match the architecture byte for byte,
//! and every one of those offsets is asserted at compile time by [`layout`].
//! Getting an offset wrong is a build error rather than a fault in a guest.
//!
//! # Definitions only
//!
//! This crate holds no `unsafe`, touches no hardware, and executes no
//! instruction. It says what the structures *are*: their layout, their fields,
//! the encodings those fields accept, and the model-specific registers that
//! sit outside them. Allocating a control block, filling one in, running a
//! guest and decoding why it came back are all somebody else's work, built on
//! top of these definitions.
//!
//! That boundary is what lets the definitions be shared. The loader, the
//! hypervisor and anything that later inspects a guest all agree on these
//! types without any of them depending on how another one drives the hardware.
//!
//! # Reserved means reserved
//!
//! The architecture requires that unused bytes of a control block be zero, and
//! reserves the right to give them meaning later. Every reserved run here is
//! therefore a *private* field: a caller builds a block with a `zeroed`
//! constructor and assigns to the public fields, and the reserved space is not
//! nameable, so it cannot be written by accident. Reserved *bits* inside a
//! register get the same treatment through unnamed bitfield members, which read
//! back as zero and have no setter.
//!
//! # Shape
//!
//! - [`Vmcb`] is the page a guest is described by: a [`ControlArea`] saying how
//!   it runs and a [`SaveArea`] holding the state it runs with.
//!   [`HostSavePage`] is where the processor puts our own state meanwhile.
//! - [`intercept`] is what a guest may not do without asking, [`permissions`]
//!   the two bitmaps refining that for ports and model-specific registers.
//! - [`exit`] is why a guest stopped, and [`event`] is what is handed to it on
//!   the way back in.
//! - [`clean`] is which parts of a control block the processor may take from
//!   its own cache instead of re-reading.
//! - [`avic`] is the interrupt controller the hardware can drive on a guest's
//!   behalf, and [`msr`] the switches that live outside any control block.
//!
//! # What is not here
//!
//! The encrypted-virtualization extension — the encrypted state area, the
//! guest-host communication block, the reverse map table and their
//! model-specific registers — is not modelled. Pulzar does not implement it.
//! Where its fields fall inside a structure this crate does define, they are
//! present as plain values so that the layout stays exact, and are documented
//! as belonging to an extension nothing here drives.

#![no_std]

/// Asserts a structure's size and the offset of every field in it.
///
/// A layout that has to match hardware is only correct if it is checked, and
/// the check is worth more than the declaration: a mistyped array length in one
/// reserved run silently shifts every field after it, which no amount of
/// reading the declaration reliably catches. Naming each offset separately from
/// the total size is what makes that impossible — both can only hold at once if
/// every gap between them is the right width.
///
/// The offsets are written in the order and notation the architecture uses, so
/// the invocation can be read against the manual line by line.
macro_rules! layout {
    ($type:ty, size = $size:expr, $($offset:expr => $field:ident),* $(,)?) => {
        const _: () = {
            assert!(
                size_of::<$type>() == $size,
                concat!(stringify!($type), " is not the size the architecture defines"),
            );
            $(
                assert!(
                    core::mem::offset_of!($type, $field) == $offset,
                    concat!(
                        stringify!($type), ".", stringify!($field),
                        " is not at the offset the architecture defines",
                    ),
                );
            )*
        };
    };
}

pub mod avic;
pub mod clean;
pub mod control;
pub mod event;
pub mod exit;
pub mod intercept;
pub mod msr;
pub mod permissions;
pub mod save;

mod vmcb;

pub use crate::{
    clean::CleanBits,
    control::ControlArea,
    event::{Event, EventKind, InterruptState},
    exit::{ExitCode, Reason},
    save::{SaveArea, Segment, SegmentAttributes},
    vmcb::{HostSavePage, Vmcb},
};

/// Bytes in the page every one of these structures is aligned to and sized by.
///
/// The control block, the host save area and each of the interrupt
/// controller's tables are all exactly this, and the processor is given their
/// physical addresses with the low twelve bits ignored — so anything less than
/// page alignment is not a smaller mistake, it is a different address.
pub const PAGE_BYTES: usize = 4096;
