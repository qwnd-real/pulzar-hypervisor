//! The UEFI frame buffer, which is a screen and no port at all.
//!
//! Firmware's console drew its last frame into a linear frame buffer whose
//! description travels in the handoff; this backend draws log lines into the
//! same pixels, so a machine with neither a debug console nor a UART header
//! still has something to say through. [`Efifb::describe`] is built from that
//! description plus an address bytes are written through — the physical base
//! under the loader, an explicit mapping under the hypervisor image, where the
//! aperture is device memory the direct map deliberately does not cover.
//!
//! The mapping side decides what the bytes cost; this module only assumes the
//! result behaves like memory. The hypervisor maps it `UncachedMinus`: the
//! PAT says uncached while firmware's MTRRs keep their veto, so a range the
//! firmware marked write-combining for scanout stays write-combining. Either
//! way plain stores reach the screen without a flush on x86, which is why
//! nothing here is volatile: the writes are ordinary pixel updates, the
//! scanout re-reads them continuously, and no ordering between two of our own
//! stores can change what a finished glyph looks like.
//!
//! The backend owns the display only until the guest starts drawing. It is
//! attached after both ports decline, and [`serial::retire_screen`] takes it
//! back before the guest is entered — past that point every line is discarded
//! rather than fought over with whatever the guest puts on the display.
//!
//! Two writers exist. The locked one runs one whole line at a time under the
//! crate's output lock, like the ports do. The other is [`serial::emergency`],
//! which takes no lock, reconstructs its view from parts the owner publishes,
//! and continues from the top-left corner rather than from wherever the locked
//! half got to; its line may interleave with another processor's mid-line,
//! which is the same trade the ports make: mangled output beats none.

use core::fmt;

use font8x8::legacy::BASIC_LEGACY;
use handoff::Framebuffer;

/// Pixels each glyph cell is wide and tall on the display.
///
/// One source pixel of the 8×8 glyph becomes a block of this many by this
/// many, which is what keeps an ordinary terminal readable on a panel a
/// modern firmware picks by default.
const SCALE: usize = 2;

/// Glyph width in source pixels, which is the font's own.
const GLYPH_WIDTH: usize = 8;
/// Glyph height in source pixels, which is also the font's own.
const GLYPH_HEIGHT: usize = 8;

/// Cell size on the display, in pixels.
const CELL_WIDTH: usize = GLYPH_WIDTH * SCALE;
/// Cell size on the display, in pixels.
const CELL_HEIGHT: usize = GLYPH_HEIGHT * SCALE;

/// Foreground colour of a drawn glyph: white.
const FOREGROUND: [u8; 3] = [0xFF; 3];
/// Background colour behind and around a drawn glyph: black.
const BACKGROUND: [u8; 3] = [0x00; 3];

/// The lowest character code with a glyph worth drawing.
const FIRST_PRINTABLE: u8 = 0x20;
/// The highest character code in the carried font table.
const LAST_PRINTABLE: u8 = 0x7F;
/// What a character outside the printable range is drawn as.
const UNKNOWN_GLYPH: u8 = b'?';

/// Which byte of a pixel holds which channel.
///
/// Both carried orders are four bytes wide with the fourth byte ignored by
/// the hardware; they differ only in how the three visible ones are spelled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Channels {
    /// Red, green, blue, then the ignored byte.
    RedGreenBlue,
    /// Blue, green, red, then the ignored byte.
    BlueGreenRed,
}

impl Channels {
    /// Reads the order out of a handoff description.
    ///
    /// [`Framebuffer::usable`] refuses every format but the two below, so a
    /// `None` here means the description was never checked, not that a checked
    /// one slipped past it.
    pub(crate) const fn of(format: u32) -> Option<Self> {
        match format {
            Framebuffer::RGBX => Some(Self::RedGreenBlue),
            Framebuffer::BGRX => Some(Self::BlueGreenRed),
            _ => None,
        }
    }

    /// Lays a colour out in the display's byte order.
    const fn encode(self, colour: [u8; 3]) -> [u8; 4] {
        match self {
            Self::RedGreenBlue => [colour[0], colour[1], colour[2], 0],
            Self::BlueGreenRed => [colour[2], colour[1], colour[0], 0],
        }
    }
}

/// Where the next glyph goes, in cells.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Cursor {
    /// Cells from the left edge of the display.
    pub(crate) column: usize,
    /// Cells from the top edge of the display.
    pub(crate) row: usize,
}

/// The geometry of a screen, and everything drawing needs besides the bytes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Canvas {
    /// Visible pixels per scan line.
    pub(crate) width: u32,
    /// Visible scan lines.
    pub(crate) height: u32,
    /// Bytes from one scan line's first pixel to the next.
    pub(crate) pitch: u32,
    /// How a pixel's channels are laid out in those bytes.
    channels: Channels,
}

impl Canvas {
    /// Reads a canvas out of a handoff description, refusing unusable ones.
    pub(crate) fn of(screen: &Framebuffer) -> Option<Self> {
        if !screen.usable() {
            return None;
        }
        let channels = Channels::of(screen.format)?;
        Some(Self {
            width: screen.width,
            height: screen.height,
            pitch: screen.pitch,
            channels,
        })
    }

    /// Bytes the drawing surface spans, which is what a mapping must cover.
    pub(crate) fn span(&self) -> u64 {
        u64::from(self.pitch) * u64::from(self.height)
    }

    /// Bytes from one scan line's first pixel to the next.
    fn pitch_bytes(&self) -> usize {
        usize::try_from(self.pitch).expect("a pitch fits a usize")
    }

    /// Cells across, which is the width less any trailing partial cell.
    fn columns(&self) -> usize {
        let cell = u32::try_from(CELL_WIDTH).expect("a cell width fits u32");
        usize::try_from(self.width / cell).expect("cells across fit a usize")
    }

    /// Cells down.
    fn rows(&self) -> usize {
        let cell = u32::try_from(CELL_HEIGHT).expect("a cell height fits u32");
        usize::try_from(self.height / cell).expect("cells down fit a usize")
    }

    /// Writes one string, starting wherever the cursor says and scrolling
    /// when the bottom row fills.
    ///
    /// Characters the carried font has no answer for draw as
    /// [`UNKNOWN_GLYPH`]; a newline moves to the first cell of the next row,
    /// filling a row that was already the last one by scrolling everything up.
    pub(crate) fn write_str(&self, pixels: &mut [u8], cursor: &mut Cursor, text: &str) {
        for character in text.chars() {
            let glyph = match u8::try_from(character) {
                Ok(byte @ FIRST_PRINTABLE..=LAST_PRINTABLE) => byte,
                Ok(b'\n') => {
                    self.newline(pixels, cursor);
                    continue;
                }
                _ => UNKNOWN_GLYPH,
            };
            self.draw(pixels, cursor, glyph);
            cursor.column += 1;
            if cursor.column == self.columns() {
                self.newline(pixels, cursor);
            }
        }
    }

    /// Moves to the first cell of the next row, scrolling a full display.
    fn newline(&self, pixels: &mut [u8], cursor: &mut Cursor) {
        cursor.column = 0;
        cursor.row += 1;
        if cursor.row == self.rows() {
            cursor.row -= 1;
            self.scroll(pixels);
        }
    }

    /// Moves every scan line up one cell height and clears the freed row.
    fn scroll(&self, pixels: &mut [u8]) {
        let lift = self.pitch_bytes() * CELL_HEIGHT;
        let tail = pixels.len() - lift;
        pixels.copy_within(lift.., 0);
        pixels[tail..].fill(0);
    }

    /// Draws one glyph into the cell the cursor names, background included,
    /// so a cell never shows a blend of this line and the last one.
    fn draw(&self, pixels: &mut [u8], cursor: &Cursor, byte: u8) {
        let ink = self.channels.encode(FOREGROUND);
        let blank = self.channels.encode(BACKGROUND);
        let origin_x = cursor.column * CELL_WIDTH;
        let origin_y = cursor.row * CELL_HEIGHT;
        for (row_index, row) in BASIC_LEGACY[usize::from(byte)].into_iter().enumerate() {
            for bit in 0..GLYPH_WIDTH {
                // The font spells rows top-down with bit zero leftmost.
                let colour = if row & (1 << bit) != 0 { ink } else { blank };
                for dy in 0..SCALE {
                    for dx in 0..SCALE {
                        self.write_pixel(
                            pixels,
                            origin_x + bit * SCALE + dx,
                            origin_y + row_index * SCALE + dy,
                            colour,
                        );
                    }
                }
            }
        }
    }

    /// Stores one pixel, ignoring coordinates off the visible area rather
    /// than wrapping them into a neighbouring scan line.
    fn write_pixel(&self, pixels: &mut [u8], x: usize, y: usize, colour: [u8; 4]) {
        let width = usize::try_from(self.width).expect("a width fits a usize");
        let height = usize::try_from(self.height).expect("a height fits a usize");
        if x >= width || y >= height {
            return;
        }
        // Widening: the constant is `u32`, and `usize` carries no lossless
        // conversion from it on this target.
        let offset = y * self.pitch_bytes() + x * Framebuffer::BYTES_PER_PIXEL as usize;
        pixels[offset..offset + colour.len()].copy_from_slice(&colour);
    }
}

/// The frame buffer as a writer: the geometry, the address its bytes live at,
/// and the cursor between lines.
pub(crate) struct Efifb {
    /// Address bytes are written through, which the mapper chose.
    pub(crate) address: u64,
    /// The screen's geometry.
    pub(crate) canvas: Canvas,
    /// Where the next glyph goes.
    pub(crate) cursor: Cursor,
}

impl Efifb {
    /// Builds the backend over a handoff description, writing through
    /// `address`.
    ///
    /// The address must be the first writable byte of exactly
    /// [`Canvas::span`] bytes — the caller mapped the described framebuffer
    /// there, and nothing unmaps or retargets that translation for as long as
    /// the machine runs, which is what makes every later slice sound.
    pub(crate) fn describe(screen: &Framebuffer, address: u64) -> Option<Self> {
        Some(Self {
            address,
            canvas: Canvas::of(screen)?,
            cursor: Cursor::default(),
        })
    }
}

impl fmt::Write for Efifb {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let Efifb {
            address,
            canvas,
            cursor,
        } = self;
        let span = usize::try_from(canvas.span()).expect("the span fits a usize");
        // SAFETY: `address` was vouched for by whoever built this writer
        // through `describe`: it names the first byte of a live mapping
        // covering `canvas.span()` bytes, established once and never moved,
        // so the slice cannot leave it. Writers are serialized by the crate's
        // output lock except the emergency path, whose interleaving is
        // accepted where it is used; and no reference into these bytes
        // outlives this call.
        let pixels = unsafe { core::slice::from_raw_parts_mut(*address as *mut u8, span) };
        canvas.write_str(pixels, cursor, s);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    //! Drawing decisions, over buffers that stand in for the mapping.

    use super::{
        BACKGROUND, CELL_HEIGHT, CELL_WIDTH, Canvas, Channels, Cursor, FOREGROUND, GLYPH_HEIGHT,
        SCALE,
    };
    extern crate std;
    use std::{vec, vec::Vec};

    /// A screen of two cells across and two down, with no padding.
    fn small_canvas() -> Canvas {
        let cell_width = u32::try_from(CELL_WIDTH).expect("cell width fits");
        let cell_height = u32::try_from(CELL_HEIGHT).expect("cell height fits");
        Canvas {
            width: 2 * cell_width,
            height: 2 * cell_height,
            pitch: 2 * cell_width * handoff::Framebuffer::BYTES_PER_PIXEL,
            channels: Channels::RedGreenBlue,
        }
    }

    /// A buffer standing in for the mapping of [`small_canvas`].
    fn small_pixels() -> Vec<u8> {
        vec![0xEE; small_canvas().span().try_into().expect("span fits")]
    }

    /// Reads the channel words of the pixel at `(x, y)` as red, green, blue.
    fn pixel(pixels: &[u8], x: usize, y: usize) -> [u8; 3] {
        let pitch = usize::try_from(small_canvas().pitch).expect("pitch fits");
        let offset = y * pitch + x * 4;
        [pixels[offset + 2], pixels[offset + 1], pixels[offset]]
    }

    /// Coordinates of every inked pixel, found by scanning for foreground.
    fn inked(pixels: &[u8]) -> Vec<(usize, usize)> {
        let canvas = small_canvas();
        let width = usize::try_from(canvas.width).expect("width fits");
        let height = usize::try_from(canvas.height).expect("height fits");
        let mut found = Vec::new();
        for y in 0..height {
            for x in 0..width {
                if pixel(pixels, x, y) == FOREGROUND {
                    found.push((x, y));
                }
            }
        }
        found
    }

    #[test]
    fn a_description_is_read_only_when_it_is_usable() {
        let usable = handoff::Framebuffer {
            base: 0xF000_0000,
            pitch: 7680,
            width: 1920,
            height: 1080,
            format: handoff::Framebuffer::BGRX,
        };
        assert!(Canvas::of(&usable).is_some());

        let blt_only = handoff::Framebuffer {
            format: 3,
            ..usable
        };
        assert!(Canvas::of(&blt_only).is_none());

        let short_pitch = handoff::Framebuffer {
            pitch: 1919 * 4,
            ..usable
        };
        assert!(Canvas::of(&short_pitch).is_none());
    }

    #[test]
    fn the_span_covers_every_scan_line_including_the_padding() {
        let padded = Canvas {
            pitch: 40 * 4,
            ..small_canvas()
        };
        assert_eq!(padded.span(), 40 * 4 * 32);
    }

    #[test]
    fn a_glyph_draws_its_pixels_and_the_background_fills_the_rest_of_the_cell() {
        let canvas = small_canvas();
        let mut pixels = small_pixels();
        canvas.draw(&mut pixels, &Cursor::default(), b'|');
        // Inside the cell every pixel is one of the two colours — never the
        // stand-in pattern, which is what a background-less draw would leave
        // behind around the glyph.
        for y in 0..CELL_HEIGHT {
            for x in 0..CELL_WIDTH {
                let seen = pixel(&pixels, x, y);
                assert!(
                    seen == FOREGROUND || seen == BACKGROUND,
                    "pixel ({x}, {y}) is neither ink nor background"
                );
            }
        }
        // And outside it nothing was touched.
        assert_eq!(pixel(&pixels, CELL_WIDTH + 1, CELL_HEIGHT + 1), [0xEE; 3]);
    }

    #[test]
    fn one_source_pixel_becomes_a_block_of_the_scale() {
        let canvas = small_canvas();
        let mut pixels = small_pixels();
        canvas.draw(&mut pixels, &Cursor::default(), b'|');
        let lit = inked(&pixels);
        assert!(!lit.is_empty(), "the glyph drew nothing");
        // Every lit source pixel appears as SCALE×SCALE screen pixels: each
        // lit position has its whole block beside it.
        for (x, y) in &lit {
            let source_x = x / SCALE;
            let source_y = y / SCALE;
            for dy in 0..SCALE {
                for dx in 0..SCALE {
                    assert_eq!(
                        pixel(&pixels, source_x * SCALE + dx, source_y * SCALE + dy),
                        FOREGROUND,
                        "block at ({source_x}, {source_y}) incomplete"
                    );
                }
            }
        }
    }

    #[test]
    fn the_channel_order_decides_which_byte_leads_the_pixel() {
        let rgbx = small_canvas();
        let bgrx = Canvas {
            channels: Channels::BlueGreenRed,
            ..rgbx
        };
        let mut left = small_pixels();
        let mut right = small_pixels();
        rgbx.draw(&mut left, &Cursor::default(), b'|');
        bgrx.draw(&mut right, &Cursor::default(), b'|');
        // White reads identically in both orders, so the first byte of an inked
        // pixel names the layout: red under one spelling, blue under the other.
        let first_red = inked(&left)[0];
        assert_eq!(pixel(&left, first_red.0, first_red.1)[0], 0xFF);
        let first_blue = inked(&right)[0];
        assert_eq!(pixel(&right, first_blue.0, first_blue.1)[2], 0xFF);
        assert_eq!((first_red.0, first_red.1), (first_blue.0, first_blue.1));
    }

    #[test]
    fn a_newline_moves_to_the_first_cell_of_the_next_row() {
        let canvas = small_canvas();
        let mut pixels = small_pixels();
        let mut cursor = Cursor::default();
        canvas.write_str(&mut pixels, &mut cursor, "\n");
        assert_eq!(cursor, Cursor { column: 0, row: 1 });
    }

    #[test]
    fn filling_the_last_cell_of_a_row_starts_the_next_one() {
        let canvas = small_canvas();
        let mut pixels = small_pixels();
        let mut cursor = Cursor::default();
        canvas.write_str(&mut pixels, &mut cursor, "..");
        assert_eq!(cursor.column, 0);
        assert_eq!(cursor.row, 1);
    }

    #[test]
    fn a_newline_on_the_last_row_scrolls_one_cell_height_and_clears_behind_it() {
        let canvas = small_canvas();
        let lift = usize::try_from(canvas.pitch).expect("pitch fits") * CELL_HEIGHT;
        let mut pixels = small_pixels();
        let last_row = canvas.rows() - 1;
        let mut cursor = Cursor {
            column: 0,
            row: last_row,
        };
        canvas.draw(&mut pixels, &cursor, b'#');
        let drawn = inked(&pixels);
        assert!(!drawn.is_empty());

        canvas.newline(&mut pixels, &mut cursor);

        assert_eq!(cursor.row, last_row);
        // Every inked pixel moved up exactly one cell height and no ink is
        // left where it was.
        for (x, y) in drawn {
            assert_eq!(pixel(&pixels, x, y - CELL_HEIGHT), FOREGROUND);
            assert_eq!(pixel(&pixels, x, y), BACKGROUND);
        }
        // And the freed row holds zeros rather than the stand-in pattern.
        assert!(pixels[pixels.len() - lift..].iter().all(|byte| *byte == 0));
    }

    #[test]
    fn characters_without_glyphs_draw_as_the_unknown_mark() {
        let canvas = small_canvas();
        let mut pixels = small_pixels();
        let mut cursor = Cursor::default();
        canvas.write_str(&mut pixels, &mut cursor, "\u{2603}");
        // The unknown mark drew into the first cell, which is what disturbing
        // its stand-in pattern proves; the point is that nothing panicked and
        // nothing was skipped.
        assert!(pixels[..CELL_WIDTH * 4].iter().any(|byte| *byte != 0xEE));
        assert_eq!(cursor.column, 1);
    }

    #[test]
    fn glyphs_are_eight_rows_tall_so_a_cell_is_scale_times_that() {
        // Pinned because the scroll distance and the cell grid both derive
        // from it; the constant lives in the font's shape, not ours.
        assert_eq!(GLYPH_HEIGHT, 8);
        assert_eq!(CELL_HEIGHT, SCALE * GLYPH_HEIGHT);
    }
}
