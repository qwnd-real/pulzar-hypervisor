//! How a guest reaches its controller.
//!
//! Two faces onto one register file. Through the memory-mapped page a register is
//! a byte offset; through x2APIC it is a model-specific register index, and the
//! index is *derived* from the offset. So there is one list of registers
//! ([`table`]) and one statement of what reading or writing each of them means
//! ([`dispatch`]), and the two faces above them do only what genuinely differs:
//! decoding an address, deciding whether a malformed access faults or merely
//! records an error, and splitting a 64-bit register into halves.
//!
//! A hypervisor that implemented the behaviour twice would be one whose guest
//! could tell which face it was using by the answers it got.

mod dispatch;
pub(crate) mod table;

pub(crate) mod mmio;
pub(crate) mod msr;
