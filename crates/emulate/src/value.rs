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

/// How wide an access is.
///
/// Five widths and no others. A move of three bytes is not something the
/// architecture has, so a width that is not one of these is a decode that went
/// wrong rather than a case to handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
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

impl Width {
    /// How many bytes this is.
    #[must_use]
    pub const fn bytes(self) -> usize {
        self as usize
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

    /// Which bits of a quadword this width covers.
    ///
    /// The whole of it for the two widest, which is why this is a match and not
    /// a shift: shifting a `u64` by sixty-four is not zero, it is undefined,
    /// and the widest two are exactly the cases that would do it.
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Data {
    bytes: [u8; Width::Vector.bytes()],
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
            bytes: [0; Width::Vector.bytes()],
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
    pub const fn vector_from(bytes: [u8; Width::Vector.bytes()]) -> Self {
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
            bytes: [0; Width::Vector.bytes()],
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
    /// What a vector register is written from. The zeroes are not padding to be
    /// ignored: a move of four or eight bytes into a vector register clears the
    /// rest of it, and this is that.
    #[must_use]
    pub const fn vector(&self) -> [u8; Width::Vector.bytes()] {
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
    #[must_use]
    pub fn extended(&self, to: Width, signed: bool) -> Self {
        let value = self.as_u64();
        let mask = self.width.mask();
        let negative = signed && value & (mask ^ (mask >> 1)) != 0;
        Self::from_u64(if negative { value | !mask } else { value }, to)
    }
}

const _: () = assert!(
    Width::Byte.bytes() == 1 && Width::Vector.bytes() == 16,
    "a width must count the bytes it is named for",
);
const _: () = assert!(
    Width::Long.mask() == u32::MAX as u64 && Width::Vector.mask() == u64::MAX,
    "a width's mask must cover the width, and the widest two the whole quadword",
);
