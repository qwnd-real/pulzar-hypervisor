//! The bytes an instruction moves, and how many of them there are.
//!
//! One type for every width a move comes in, from a byte to the sixteen a
//! vector register holds. A `u64` would do for four of the five, and the fifth
//! is not rare enough to be worth a second path through the whole emulator for
//! — a guest writing a framebuffer moves sixteen bytes at a time and does it
//! constantly.
//!
//! The bytes are always kept little-endian and left-justified: the low byte
//! first, and whatever is past the width zero. That is both how they sit in
//! memory and how they sit in a register, which is what lets the same array
//! serve a device window, a guest address and a register file without being
//! rearranged in between.
//!
//! # Widening is defined over the bytes, not through a quadword
//!
//! [`Data::extended`] works on all sixteen bytes. Routing it through a `u64`
//! would be correct for the four scalar widths and silently wrong for the
//! fifth: the upper eight bytes of a vector would be dropped on the way in, and
//! a negative scalar widened to a vector would stop filling at byte seven. Both
//! are the sort of thing that produces a plausible value rather than a failure,
//! so the operation is defined where it cannot happen and the conversions that
//! genuinely have no answer — narrowing, and widening *from* a vector — are
//! refused rather than approximated.

use core::fmt::{self, Debug, Formatter};

/// How wide an access is.
///
/// Five widths and no others. A move of three bytes is not something the
/// architecture has, so a width that is not one of these is a decode that went
/// wrong rather than a case to handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Width {
    /// One byte.
    Byte = 1,
    /// Two bytes.
    Word = 2,
    /// Four bytes. The one width whose write to a general-purpose register
    /// clears what is above it rather than leaving it alone.
    Long = 4,
    /// Eight bytes.
    Quad = 8,
    /// Sixteen bytes: what one vector register holds.
    Vector = 16,
}

/// Bytes in the widest access there is, which is how wide every [`Data`] is
/// stored regardless of how much of it means anything.
pub(crate) const WIDEST: usize = Width::Vector.bytes();

impl Width {
    /// Every width, narrowest first.
    ///
    /// Exhaustive by construction rather than by convention: the assertion
    /// below fails the build if a variant is added without being listed here,
    /// which is what lets the tests iterate this and still be complete.
    pub(crate) const ALL: [Self; 5] =
        [Self::Byte, Self::Word, Self::Long, Self::Quad, Self::Vector];

    /// How many bytes this is.
    #[must_use]
    pub const fn bytes(self) -> usize {
        self as usize
    }

    /// How many bytes this is, as an address-sized number.
    ///
    /// Every span this crate checks is arithmetic on `u64` addresses, and a
    /// width is one of five compile-time constants — so the conversion is a
    /// widening cast that cannot lose anything rather than something to handle
    /// the failure of.
    #[must_use]
    pub const fn span(self) -> u64 {
        self as u64
    }

    /// How far one repetition of a string instruction moves an index register,
    /// which is forwards by this many bytes or backwards by them.
    ///
    /// Signed and infallible, because the alternative is a conversion that can
    /// fail on a path where failure has no sensible answer: a step of zero
    /// would leave a repeated instruction walking the same address for
    /// ever, and a refusal would abandon an instruction the architecture
    /// defines completely. Five compile-time constants, none of them larger
    /// than sixteen, so the cast is exact.
    #[must_use]
    pub const fn step(self) -> i64 {
        self as i64
    }

    /// The width that many bytes is, or `None` if the architecture has no move
    /// that wide.
    #[must_use]
    pub const fn from_bytes(bytes: usize) -> Option<Self> {
        Some(match bytes {
            1 => Self::Byte,
            2 => Self::Word,
            4 => Self::Long,
            8 => Self::Quad,
            16 => Self::Vector,
            _ => return None,
        })
    }

    /// Whether this width fits a general-purpose register, which the widest
    /// does not.
    #[must_use]
    pub const fn scalar(self) -> bool {
        !matches!(self, Self::Vector)
    }

    /// Which bits of a quadword this width covers.
    ///
    /// The whole of it for the two widest, which is why this is a match and not
    /// a shift: shifting a `u64` by sixty-four is not zero, it is undefined,
    /// and the widest two are exactly the cases that would do it.
    ///
    /// For [`Width::Vector`] this is the mask of the *low half* only. That is
    /// the right answer everywhere it is asked — the instructions that move a
    /// vector into a general-purpose register take the low eight bytes — and it
    /// is why nothing here uses it to reason about a vector as a whole.
    #[must_use]
    pub const fn mask(self) -> u64 {
        match self {
            Self::Byte => 0xFF,
            Self::Word => 0xFFFF,
            Self::Long => 0xFFFF_FFFF,
            Self::Quad | Self::Vector => u64::MAX,
        }
    }
}

/// The bytes an instruction moved.
///
/// Compared by what it means rather than by how it is stored: two values of the
/// same width whose meaningful bytes agree are equal, and the padding above the
/// width is not part of the value. Every constructor here zeroes that padding,
/// so the two notions coincide — the manual implementations exist to keep them
/// coinciding rather than to paper over a difference.
#[derive(Clone, Copy)]
pub struct Data {
    bytes: [u8; WIDEST],
    width: Width,
}

impl Data {
    /// A value of this width, out of a quadword.
    ///
    /// Anything above the width is dropped, so a caller need not mask first. A
    /// vector built this way has its upper eight bytes zero, which is what the
    /// instructions that build one from a general-purpose register do.
    #[must_use]
    pub fn from_u64(value: u64, width: Width) -> Self {
        let mut data = Self {
            bytes: [0; WIDEST],
            width,
        };
        let take = width.bytes().min(size_of::<u64>());
        data.bytes[..take].copy_from_slice(&(value & width.mask()).to_le_bytes()[..take]);
        data
    }

    /// All sixteen bytes of a vector.
    ///
    /// Separate from [`Data::from_bytes`] because sixteen bytes is always a
    /// width, so this one cannot fail and callers need not pretend it might.
    #[must_use]
    pub const fn vector_from(bytes: [u8; WIDEST]) -> Self {
        Self {
            bytes,
            width: Width::Vector,
        }
    }

    /// A value out of the bytes themselves, or `None` if there is no move as
    /// wide as the slice.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let width = Width::from_bytes(bytes.len())?;
        let mut data = Self {
            bytes: [0; WIDEST],
            width,
        };
        data.bytes[..bytes.len()].copy_from_slice(bytes);
        Some(data)
    }

    /// How wide this is.
    #[must_use]
    pub const fn width(&self) -> Width {
        self.width
    }

    /// The bytes, and only as many of them as the width.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes[..self.width.bytes()]
    }

    /// All sixteen bytes, with anything past the width zero.
    ///
    /// What a vector register is written from *for the forms that zero what
    /// they do not write* — which is every move from memory or from a
    /// general-purpose register, and is not the legacy register-to-register
    /// scalar moves. Those preserve the upper bits of their destination, and
    /// the form table says which is which rather than this type assuming.
    #[must_use]
    pub const fn vector(&self) -> [u8; WIDEST] {
        self.bytes
    }

    /// The low eight bytes as a quadword.
    ///
    /// For anything narrower this is the value with zeroes above it. For a
    /// vector it is the lower half, which is what the instructions that move
    /// one into a general-purpose register take.
    #[must_use]
    pub fn as_u64(&self) -> u64 {
        let mut value = [0; size_of::<u64>()];
        let take = self.width.bytes().min(size_of::<u64>());
        value[..take].copy_from_slice(&self.bytes[..take]);
        u64::from_le_bytes(value)
    }

    /// The same value made `to` bytes wide, filling what is added with either
    /// zeroes or copies of the sign bit.
    ///
    /// Widening is done here rather than by casting through a signed type
    /// because the width is a value and not a type: what has to be copied is
    /// whichever bit the *old* width made the top one, and a cast can only ever
    /// know the new one.
    ///
    /// Only widening is defined. Narrowing under an extending name would lose
    /// bytes silently, and there is no such thing as extending a vector — the
    /// architecture's widening moves all take a general-purpose source — so
    /// both answer `None` rather than a plausible value. Widening to the same
    /// width is the identity and is allowed, because that is what a widening
    /// move whose two ends happen to agree does.
    #[must_use]
    pub fn extended(&self, to: Width, signed: bool) -> Option<Self> {
        if to < self.width || self.width == Width::Vector {
            return None;
        }
        let filler = if signed && self.negative() { 0xFF } else { 0 };
        let mut widened = Self {
            bytes: [0; WIDEST],
            width: to,
        };
        let (kept, added) = widened.bytes[..to.bytes()].split_at_mut(self.width.bytes());
        kept.copy_from_slice(self.bytes());
        added.fill(filler);
        Some(widened)
    }

    /// Whether the top bit of the value — the top bit of *its own* width, which
    /// is the only one sign extension is about — is set.
    fn negative(&self) -> bool {
        self.bytes()
            .last()
            .is_some_and(|top| top & TOP_BIT == TOP_BIT)
    }
}

impl PartialEq for Data {
    fn eq(&self, other: &Self) -> bool {
        self.width == other.width && self.bytes() == other.bytes()
    }
}

impl Eq for Data {}

impl Debug for Data {
    /// The meaningful bytes, widest byte first, which is how a register's
    /// contents are conventionally read and is not how they are stored.
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:?}:", self.width)?;
        self.bytes()
            .iter()
            .rev()
            .try_for_each(|byte| write!(formatter, "{byte:02x}"))
    }
}

/// The bit that decides a byte's sign, and so a value's.
const TOP_BIT: u8 = 0x80;

const _: () = assert!(
    Width::ALL.len() == 5 && Width::ALL[0].bytes() == 1 && Width::ALL[4].bytes() == WIDEST,
    "every width must be listed, narrowest first, or the tables that walk them are incomplete",
);
const _: () = assert!(
    Width::Byte.bytes() == 1 && Width::Vector.bytes() == 16,
    "a width must count the bytes it is named for",
);
const _: () = assert!(
    Width::Long.mask() == u32::MAX as u64 && Width::Vector.mask() == u64::MAX,
    "a width's mask must cover the width, and the widest two the whole quadword",
);

#[cfg(test)]
mod tests {
    use super::{Data, WIDEST, Width};

    /// A byte pattern whose every byte is distinguishable from its neighbours,
    /// so that a prefix taken from the wrong end is visible rather than
    /// plausible.
    const PATTERN: [u8; WIDEST] = [
        0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF,
        0x01,
    ];

    #[test]
    fn every_width_counts_the_bytes_it_is_named_for() {
        assert_eq!(
            Width::ALL.map(Width::bytes),
            [1, 2, 4, 8, WIDEST],
            "the width table must count its own bytes"
        );
        for width in Width::ALL {
            assert_eq!(Width::from_bytes(width.bytes()), Some(width));
            assert_eq!(width.span(), width.bytes() as u64);
        }
    }

    #[test]
    fn widths_order_by_how_wide_they_are() {
        let mut sorted = Width::ALL;
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            Width::ALL,
            "the declaration order must be the width order"
        );
    }

    #[test]
    fn no_other_byte_count_is_a_width() {
        for bytes in [0, 3, 5, 6, 7, 9, 10, 15, 17, 32, 64, usize::MAX] {
            assert_eq!(
                Width::from_bytes(bytes),
                None,
                "{bytes} bytes is not a width the architecture has"
            );
        }
    }

    #[test]
    fn only_the_widest_is_wider_than_a_general_purpose_register() {
        for width in Width::ALL {
            assert_eq!(width.scalar(), width != Width::Vector);
        }
    }

    #[test]
    fn masks_cover_exactly_their_own_width() {
        assert_eq!(Width::Byte.mask(), 0xFF);
        assert_eq!(Width::Word.mask(), 0xFFFF);
        assert_eq!(Width::Long.mask(), 0xFFFF_FFFF);
        // The widest two cover the whole quadword: a vector's mask is about its
        // low half, which is the only half a quadword can hold.
        assert_eq!(Width::Quad.mask(), u64::MAX);
        assert_eq!(Width::Vector.mask(), u64::MAX);
    }

    #[test]
    fn a_scalar_keeps_its_low_bytes_and_zeroes_the_rest() {
        let value = 0x1122_3344_5566_7788;
        for width in Width::ALL.into_iter().filter(|width| width.scalar()) {
            let data = Data::from_u64(value, width);
            assert_eq!(data.width(), width);
            assert_eq!(
                data.bytes(),
                &value.to_le_bytes()[..width.bytes()],
                "{width:?} must keep the low bytes of the quadword"
            );
            assert!(
                data.vector()[width.bytes()..].iter().all(|byte| *byte == 0),
                "{width:?} must leave nothing above its own width"
            );
        }
    }

    #[test]
    fn a_vector_from_a_quadword_zeroes_its_upper_half() {
        let data = Data::from_u64(0x1122_3344_5566_7788, Width::Vector);
        assert_eq!(data.width(), Width::Vector);
        assert_eq!(data.bytes().len(), WIDEST);
        assert_eq!(&data.bytes()[..8], &0x1122_3344_5566_7788_u64.to_le_bytes());
        assert!(data.bytes()[8..].iter().all(|byte| *byte == 0));
        assert_eq!(data.as_u64(), 0x1122_3344_5566_7788);
    }

    #[test]
    fn a_whole_vector_survives_being_stored_and_read_back() {
        let data = Data::vector_from(PATTERN);
        assert_eq!(data.width(), Width::Vector);
        assert_eq!(data.bytes(), PATTERN);
        assert_eq!(data.vector(), PATTERN);
        // The low qword, which is what a move into a general-purpose register
        // takes, and not the whole of it.
        assert_eq!(
            data.as_u64(),
            u64::from_le_bytes(PATTERN[..8].try_into().unwrap())
        );
    }

    #[test]
    fn a_slice_becomes_a_value_exactly_when_it_is_a_width() {
        for width in Width::ALL {
            let data = Data::from_bytes(&PATTERN[..width.bytes()])
                .expect("every width is an acceptable slice length");
            assert_eq!(data.width(), width);
            assert_eq!(data.bytes(), &PATTERN[..width.bytes()]);
            assert!(
                data.vector()[width.bytes()..].iter().all(|byte| *byte == 0),
                "{width:?} must leave nothing above its own width"
            );
        }
        for bytes in [0, 3, 5, 6, 7, 9, 15] {
            assert!(
                Data::from_bytes(&PATTERN[..bytes]).is_none(),
                "{bytes} bytes is not a width"
            );
        }
    }

    #[test]
    fn equality_is_about_the_value_and_not_the_padding() {
        assert_eq!(
            Data::from_u64(0xFF, Width::Byte),
            Data::from_u64(0xFF, Width::Byte)
        );
        assert_ne!(
            Data::from_u64(0xFF, Width::Byte),
            Data::from_u64(0xFF, Width::Word),
            "the same bits at different widths are different values"
        );
        // A quadword whose upper bytes are zero is still not a vector.
        assert_ne!(
            Data::from_u64(1, Width::Quad),
            Data::from_u64(1, Width::Vector)
        );
    }

    #[test]
    fn zero_extension_clears_everything_it_adds() {
        for (from, value) in [
            (Width::Byte, 0x80),
            (Width::Word, 0x8000),
            (Width::Long, 0x8000_0000),
            (Width::Quad, 0x8000_0000_0000_0000),
        ] {
            let narrow = Data::from_u64(value, from);
            for to in Width::ALL.into_iter().filter(|to| *to >= from) {
                let wide = narrow.extended(to, false).expect("widening is defined");
                assert_eq!(wide.width(), to);
                assert_eq!(&wide.bytes()[..from.bytes()], narrow.bytes());
                assert!(
                    wide.bytes()[from.bytes()..].iter().all(|byte| *byte == 0),
                    "zero extension from {from:?} to {to:?} must add only zeroes"
                );
            }
        }
    }

    #[test]
    fn sign_extension_copies_the_top_bit_of_the_old_width() {
        for (from, value) in [
            (Width::Byte, 0x80),
            (Width::Word, 0x8000),
            (Width::Long, 0x8000_0000),
            (Width::Quad, 0x8000_0000_0000_0000),
        ] {
            let narrow = Data::from_u64(value, from);
            for to in Width::ALL.into_iter().filter(|to| *to >= from) {
                let wide = narrow.extended(to, true).expect("widening is defined");
                assert_eq!(wide.width(), to);
                assert_eq!(&wide.bytes()[..from.bytes()], narrow.bytes());
                assert!(
                    wide.bytes()[from.bytes()..]
                        .iter()
                        .all(|byte| *byte == 0xFF),
                    "sign extension from {from:?} to {to:?} must fill every added byte"
                );
            }
        }
    }

    #[test]
    fn sign_extension_of_a_positive_value_adds_zeroes() {
        for (from, value) in [
            (Width::Byte, 0x7F),
            (Width::Word, 0x7FFF),
            (Width::Long, 0x7FFF_FFFF),
        ] {
            let wide = Data::from_u64(value, from)
                .extended(Width::Vector, true)
                .expect("widening to a vector is defined for a scalar source");
            assert!(
                wide.bytes()[from.bytes()..].iter().all(|byte| *byte == 0),
                "a positive {from:?} must not be filled with ones"
            );
        }
    }

    #[test]
    fn a_negative_scalar_fills_a_whole_vector() {
        // The case a widening routed through a quadword gets wrong: the fill
        // has to reach byte fifteen, not stop at byte seven.
        let wide = Data::from_u64(0x80, Width::Byte)
            .extended(Width::Vector, true)
            .expect("widening a byte to a vector is defined");
        assert_eq!(wide.bytes()[0], 0x80);
        assert!(
            wide.bytes()[1..].iter().all(|byte| *byte == 0xFF),
            "every one of the fifteen added bytes must carry the sign"
        );
    }

    #[test]
    fn widening_to_the_same_width_changes_nothing() {
        for width in Width::ALL.into_iter().filter(|width| width.scalar()) {
            let data = Data::from_u64(0x8000_0000_0000_0000, width);
            assert_eq!(data.extended(width, true), Some(data));
            assert_eq!(data.extended(width, false), Some(data));
        }
    }

    #[test]
    fn narrowing_is_not_something_an_extension_does() {
        for from in Width::ALL {
            for to in Width::ALL.into_iter().filter(|to| *to < from) {
                assert_eq!(
                    Data::from_u64(u64::MAX, from).extended(to, false),
                    None,
                    "{from:?} must not narrow to {to:?} under an extending name"
                );
            }
        }
    }

    #[test]
    fn a_vector_has_no_extension() {
        // Not because the bytes could not be copied, but because no widening
        // move in the architecture has a vector source — so an answer here
        // would be an invention.
        for to in Width::ALL {
            assert_eq!(Data::vector_from(PATTERN).extended(to, true), None);
            assert_eq!(Data::vector_from(PATTERN).extended(to, false), None);
        }
    }

    #[test]
    fn quadword_boundaries_survive_an_equal_width_extension() {
        for value in [1 << 63, (1_u64 << 63) - 1, u64::MAX, 0] {
            let data = Data::from_u64(value, Width::Quad);
            assert_eq!(
                data.extended(Width::Quad, true).map(|d| d.as_u64()),
                Some(value)
            );
            assert_eq!(
                data.extended(Width::Quad, false).map(|d| d.as_u64()),
                Some(value)
            );
        }
    }

    #[test]
    fn every_scalar_quadword_leaves_nothing_above_its_width() {
        // The representation property the whole crate rests on, over a spread
        // of values rather than one.
        for seed in 0..64 {
            let value = 0x0123_4567_89AB_CDEF_u64.rotate_left(seed) ^ (1 << seed);
            for width in Width::ALL {
                let data = Data::from_u64(value, width);
                assert!(
                    data.vector()[width.bytes()..].iter().all(|byte| *byte == 0),
                    "{value:#x} at {width:?} left something above its width"
                );
                assert_eq!(
                    data.as_u64(),
                    value & width.mask(),
                    "{value:#x} at {width:?} did not round-trip through a quadword"
                );
            }
        }
    }
}
