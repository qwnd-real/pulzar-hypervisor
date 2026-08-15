//! One controller per processor, and everything about the machine they belong to
//! rather than about any one of them.
//!
//! # Why the controllers are one array for the machine
//!
//! Because that is what the hardware they stand for is. The memory-mapped face is
//! one page of guest physical memory at the same address on every processor, each
//! seeing its own controller through it, and an interprocessor interrupt is one
//! processor reaching into another's controller — so neither the page nor the
//! delivery path can be given a per-processor structure to reach.
//!
//! # The array is indexed by roster position, and that is what makes it sound
//!
//! A [`cpu::CpuIndex`] is a position in the roster firmware described, and the
//! array is built by mapping over that same roster in order. Nothing else mints
//! one, so an index cannot name a row that does not exist — and the mapping from
//! index to controller is written exactly once, in [`registry`], so it cannot be
//! written differently anywhere.

pub(crate) mod diagnostics;
pub(crate) mod exits;
pub(crate) mod install;
pub(crate) mod ownership;
pub(crate) mod registry;

pub(crate) use crate::machine::registry::{current, of};
