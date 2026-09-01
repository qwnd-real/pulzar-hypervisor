//! The register file of an `NVMe` controller: where in BAR0 each register is,
//! and which of them this driver has a reason to shadow.
//!
//! Only the registers that configure the admin queues matter here. The rest
//! of the file — capabilities the guest reads, interrupts it masks, the
//! version it checks — passes through to the hardware unwatched, and so does
//! every read of anything: this driver traps writes, and reads of a register
//! the hardware answers honestly cost nothing.

use bitfield_struct::bitfield;

/// The controller's capabilities (`NVMe` 2.0 §3.1.1), of which this driver
/// needs one field: the doorbell stride, which says where each queue's
/// doorbells are.
#[bitfield(u64)]
pub(crate) struct Cap {
    /// Reserved.
    #[bits(32)]
    _below: u64,
    /// The doorbell stride, as a power of two scaled from four bytes: a
    /// queue's doorbells sit this far apart.
    #[bits(4)]
    pub dstrd: u8,
    /// Reserved.
    #[bits(28)]
    _above: u64,
}

/// The controller's configuration (`NVMe` 2.0 §3.1.2), of which this driver
/// needs one bit: the enable, whose fall takes the admin queues with it.
#[bitfield(u32)]
pub(crate) struct Cc {
    /// Whether the controller is enabled. Writing it back to zero is how a
    /// guest tears the queues down before building them again.
    pub enabled: bool,
    /// Reserved.
    #[bits(31)]
    _above: u32,
}

/// The admin queues' attributes (`NVMe` 2.0 §3.1.4): two depths, twelve bits
/// each, each stored as its value less one.
#[bitfield(u32)]
#[derive(PartialEq, Eq)]
pub(crate) struct Aqa {
    /// How many entries the submission queue holds, less one.
    #[bits(12)]
    pub submissions_less_one: u16,
    /// Reserved.
    #[bits(4)]
    _between: u8,
    /// How many entries the completion queue holds, less one.
    #[bits(12)]
    pub completions_less_one: u16,
    /// Reserved.
    #[bits(4)]
    _above: u8,
}

impl Aqa {
    /// How many entries the submission queue holds.
    #[must_use]
    pub const fn submissions(self) -> u16 {
        self.submissions_less_one() + 1
    }

    /// How many entries the completion queue holds.
    #[must_use]
    pub const fn completions(self) -> u16 {
        self.completions_less_one() + 1
    }
}

/// Where the controller's capabilities are: the first register, and the one
/// read when a controller is taken over.
pub(crate) const CAP: u64 = 0x00;

/// Where the controller's configuration is.
const CC: u64 = 0x14;

/// Where the NVM subsystem reset is: a register a guest writes to take the
/// whole subsystem down without ever touching the enable.
const NSSR: u64 = 0x20;

/// Where the admin queues' attributes are.
const AQA: u64 = 0x24;

/// Where the admin submission queue's base address is. Sixty-four bits, so
/// the guest may write it as one quadword or in pieces.
const ASQ: u64 = 0x28;

/// Where the admin completion queue's base address is, likewise.
const ACQ: u64 = 0x30;

/// Where the doorbell array begins: one page into the file, past every
/// register the specification defines there.
pub(crate) const DOORBELLS: u64 = 0x1000;

/// How large a page is, which is both the register file's extent and the
/// doorbell array's alignment.
pub(crate) const PAGE: u64 = 0x1000;

/// How far apart two queues' doorbells are, given the stride the
/// controller's capabilities reported.
#[must_use]
pub(crate) const fn stride(dstrd: u8) -> u64 {
    4 << dstrd
}

/// Which of the registers this driver shadows, if any, a write lands in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Named {
    /// The controller's configuration: its enable bit, whose fall takes the
    /// admin queues with it.
    Configuration,
    /// The NVM subsystem reset, which takes the controller down without the
    /// enable ever falling.
    SubsystemReset,
    /// The admin queues' attributes: both queues' depths.
    Attributes,
    /// The admin submission queue's base, and where within the register the
    /// write landed.
    SubmissionBase { within: u64 },
    /// The admin completion queue's base, and where within the register the
    /// write landed.
    CompletionBase { within: u64 },
    /// Nothing this driver shadows.
    Other,
}

/// Names the register a write at `offset` from the start of the file lands
/// in.
///
/// The emulator admits only naturally aligned scalar writes, and every
/// register here is at least four-byte aligned and either four or eight
/// bytes wide, so an admitted write never straddles two registers: naming
/// the one it starts in names the whole of it.
#[must_use]
pub(crate) fn named(offset: u64) -> Named {
    if offset == CC {
        Named::Configuration
    } else if offset == NSSR {
        Named::SubsystemReset
    } else if offset == AQA {
        Named::Attributes
    } else if (ASQ..ASQ + 8).contains(&offset) {
        Named::SubmissionBase {
            within: offset - ASQ,
        }
    } else if (ACQ..ACQ + 8).contains(&offset) {
        Named::CompletionBase {
            within: offset - ACQ,
        }
    } else {
        Named::Other
    }
}

#[cfg(test)]
mod tests {
    use super::{Aqa, Cap, Named, named};

    #[test]
    fn the_doorbell_stride_is_four_bytes_scaled() {
        assert_eq!(super::stride(0), 4);
        assert_eq!(super::stride(2), 16);
        assert_eq!(super::stride(4), 64);
    }

    #[test]
    fn the_stride_is_the_capabilities_upper_half() {
        let capabilities = Cap::from(0x0000_0005_0000_0000);
        assert_eq!(capabilities.dstrd(), 5);
    }

    #[test]
    fn the_queues_depths_are_stored_less_one() {
        let attributes = Aqa::from(0x00ff_0006);
        assert_eq!(attributes.submissions(), 7);
        assert_eq!(attributes.completions(), 0x100);
    }

    #[test]
    fn every_shadowed_register_is_named() {
        assert_eq!(named(0x14), Named::Configuration);
        assert_eq!(named(0x20), Named::SubsystemReset);
        assert_eq!(named(0x24), Named::Attributes);
        assert_eq!(named(0x28), Named::SubmissionBase { within: 0 });
        assert_eq!(named(0x2c), Named::SubmissionBase { within: 4 });
        assert_eq!(named(0x2f), Named::SubmissionBase { within: 7 });
        assert_eq!(named(0x30), Named::CompletionBase { within: 0 });
        assert_eq!(named(0x37), Named::CompletionBase { within: 7 });
        assert_eq!(named(0x00), Named::Other);
        assert_eq!(named(0x1c), Named::Other);
        assert_eq!(named(0x38), Named::Other);
        assert_eq!(named(0x1000), Named::Other);
    }
}
