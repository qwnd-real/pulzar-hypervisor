//! The two bitmaps that decide, port by port and register by register, what a
//! guest may touch without the hypervisor hearing about it.
//!
//! Both intercept vectors have a bit for "intercept port access" and a bit for
//! "intercept model-specific register access", but those are all-or-nothing,
//! and all-or-nothing is unusable: a guest that exits on every port access and
//! every register read cannot run. These bitmaps are the refinement. With the
//! corresponding intercept set, the processor consults the map and exits only
//! for the ports and registers that are actually someone else's business.
//!
//! This module states their shapes and works out where a given port or register
//! lands in one. It allocates nothing and reads nothing: a caller allocates the
//! pages, and these functions say which bit to set.
//!
//! # Both are addressed physically
//!
//! The processor is given each map's physical address and reads it directly,
//! ignoring the low twelve bits — so both must be page-aligned, and both must
//! live in ordinary write-back memory. A map whose last byte falls at or above
//! the highest physical address the processor implements is not a warning, it
//! makes entering the guest fail outright.
//!
//! # A set bit means intercepted
//!
//! In both maps, one means "this comes back to the hypervisor" and zero means
//! "the guest may do this freely". A freshly zeroed map therefore intercepts
//! nothing, which is the permissive default rather than the safe one — worth
//! knowing before allocating one and setting the intercept bit.

/// Bytes the port permission bitmap occupies.
///
/// Three pages, which is one more than the obvious two. A port access may be
/// four bytes wide starting at any port, including the very last one, and the
/// processor checks a bit for every port the access covers — so an access at
/// port `0xFFFF` reads three bits past the end of the sixty-four kibibit map.
/// Those three bits have to exist, and rounding up to a whole page is the only
/// way to give them to it.
pub const IOPM_BYTES: usize = 12 * 1024;

/// Bytes the model-specific register permission bitmap occupies.
///
/// Two pages holding four independent two-kibibyte vectors, one per covered
/// range of register numbers.
pub const MSRPM_BYTES: usize = 8 * 1024;

/// Alignment both maps require, which is also the granularity the processor
/// ignores in the addresses it is given.
pub const MAP_ALIGN: usize = 4096;

/// Where a permission lives in a bitmap.
///
/// A byte index and a bit within that byte, which is what a caller needs to set
/// or clear one and is more useful than a flat bit number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitPosition {
    /// Which byte of the map.
    pub byte: usize,
    /// Which bit of that byte, counting from the least significant.
    pub bit: u8,
}

impl BitPosition {
    /// The position of flat bit `index`.
    const fn at(index: usize) -> Self {
        Self {
            byte: index / u8::BITS as usize,
            #[expect(
                clippy::cast_possible_truncation,
                reason = "a remainder modulo eight is one byte wide by construction"
            )]
            bit: (index % u8::BITS as usize) as u8,
        }
    }

    /// A mask selecting this bit within its byte.
    #[must_use]
    pub const fn mask(self) -> u8 {
        1 << self.bit
    }
}

/// Where a port's permission bit lives.
///
/// Bit `n` of the map is port `n`, so this is division by eight — but it is
/// worth having a name, because the interesting rule is what a caller must do
/// with it. An access wider than one byte covers several consecutive ports, and
/// the processor intercepts it if *any* of their bits is set. To intercept a
/// wide register a caller must therefore set the bit for every byte of it, and
/// to leave one alone must clear all of them.
#[must_use]
pub const fn iopm_position(port: u16) -> BitPosition {
    BitPosition::at(port as usize)
}

/// Which of the four vectors of the register bitmap covers a register, and
/// where the vector begins.
///
/// The ranges are not contiguous and not in one block, which is the whole
/// reason this lookup exists rather than a subtraction at each use site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MsrRange {
    /// Lowest register number the vector covers.
    first: u32,
    /// Byte offset of the vector within the map.
    offset: usize,
}

/// The ranges covered, in the order their vectors appear in the map.
///
/// Each vector is two kibibytes covering eight thousand registers at two bits
/// each. The fourth vector is reserved and covers nothing, which is why there
/// are three entries here and not four.
const MSR_RANGES: [MsrRange; 3] = [
    MsrRange {
        first: 0x0000_0000,
        offset: 0x0000,
    },
    MsrRange {
        first: 0xC000_0000,
        offset: 0x0800,
    },
    MsrRange {
        first: 0xC001_0000,
        offset: 0x1000,
    },
];

/// How many registers one vector covers.
const MSRS_PER_RANGE: u32 = 8 * 1024;

/// Bytes one vector occupies.
const RANGE_BYTES: usize = 2 * 1024;

/// Where a register's two permission bits live.
///
/// Every register gets a pair: the lower bit controls reading and the upper
/// controls writing. They are returned named rather than as a pair of numbers
/// because a caller that transposes them builds a map that intercepts writes
/// while letting reads through, which is both wrong and quiet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MsrPermission {
    /// Where the bit intercepting reads of this register lives.
    pub read: BitPosition,
    /// Where the bit intercepting writes of this register lives.
    pub write: BitPosition,
}

/// Where a register's permission bits live, if the map covers it at all.
///
/// `None` is a real answer rather than a failure, and an important one: when
/// register interception is enabled, any access to a register outside these
/// three ranges is intercepted unconditionally. There is no bit to clear to
/// allow it, so a hypervisor that wants such a register to be fast cannot have
/// it, and one that assumed the map covered everything would silently never see
/// those accesses coming.
#[must_use]
pub const fn msrpm_position(msr: u32) -> Option<MsrPermission> {
    let mut index = 0;
    while index < MSR_RANGES.len() {
        let range = MSR_RANGES[index];
        if msr >= range.first && msr - range.first < MSRS_PER_RANGE {
            let within = (msr - range.first) as usize;
            let read = range.offset * u8::BITS as usize + within * BITS_PER_MSR;
            return Some(MsrPermission {
                read: BitPosition::at(read),
                write: BitPosition::at(read + 1),
            });
        }
        index += 1;
    }
    None
}

/// Bits each register gets in the map: one for reads, one for writes.
const BITS_PER_MSR: usize = 2;

/// Which direction an intercepted register access was.
///
/// Reported in the first exit-information field, as a whole word holding zero
/// or one — small enough that a hypervisor is tempted to test it inline, and
/// exactly the kind of bare zero-or-one that reads wrong six months later.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MsrAccess {
    /// The guest was reading the register.
    Read,
    /// The guest was writing it.
    Write,
}

impl MsrAccess {
    /// The direction the exit information reports.
    ///
    /// Anything other than zero is a write, which is how the architecture
    /// describes the field.
    #[must_use]
    pub const fn from_exit_info(info: u64) -> Self {
        if info == 0 { Self::Read } else { Self::Write }
    }
}

const _: () = assert!(
    IOPM_BYTES.is_multiple_of(MAP_ALIGN) && MSRPM_BYTES.is_multiple_of(MAP_ALIGN),
    "both maps must be a whole number of pages so their addresses can be page aligned",
);
const _: () = assert!(
    IOPM_BYTES * u8::BITS as usize >= u16::MAX as usize + 1 + 3,
    "the port map must hold a bit for every port and three more for a wide access at the top",
);
const _: () = assert!(
    MSR_RANGES[1].offset - MSR_RANGES[0].offset == RANGE_BYTES
        && MSR_RANGES[2].offset - MSR_RANGES[1].offset == RANGE_BYTES,
    "the register map's vectors are two kibibytes apart",
);
const _: () = assert!(
    MSRS_PER_RANGE as usize * BITS_PER_MSR == RANGE_BYTES * u8::BITS as usize,
    "each vector must hold two bits for every register it covers",
);

const _: () = {
    let Some(first) = msrpm_position(0x0000_0000) else {
        panic!("the first register of the low range must be covered")
    };
    assert!(
        first.read.byte == 0 && first.read.bit == 0 && first.write.bit == 1,
        "register zero takes the first two bits of the map",
    );
};
const _: () = {
    let Some(efer) = msrpm_position(0xC000_0080) else {
        panic!("the extended feature register must be covered")
    };
    assert!(
        efer.read.byte == 0x800 + 0x80 / 4,
        "the extended feature register lands in the second vector",
    );
};
const _: () = {
    let Some(hsave) = msrpm_position(0xC001_0117) else {
        panic!("the host save address register must be covered")
    };
    assert!(
        hsave.read.byte == 0x1000 + 0x117 / 4,
        "the host save address register lands in the third vector",
    );
};
const _: () = assert!(
    msrpm_position(0x0000_2000).is_none(),
    "a register just past the low range is covered by no vector",
);
const _: () = assert!(
    msrpm_position(0xC002_0000).is_none(),
    "a register above every range is covered by no vector",
);
const _: () = {
    let top = iopm_position(u16::MAX);
    assert!(
        top.byte == 0x1FFF && top.bit == 7,
        "the last port is the last bit of the first two pages",
    );
};
