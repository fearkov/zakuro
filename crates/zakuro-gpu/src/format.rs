//! pixel formats and the tiled layout the PICA200 stores images in.

/// color formats a framebuffer or a display transfer can use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorFormat {
    Rgba8,
    Rgb8,
    Rgb565,
    Rgb5A1,
    Rgba4,
}

impl ColorFormat {
    pub fn from_raw(value: u32) -> ColorFormat {
        match value & 7 {
            0 => ColorFormat::Rgba8,
            1 => ColorFormat::Rgb8,
            2 => ColorFormat::Rgb565,
            3 => ColorFormat::Rgb5A1,
            _ => ColorFormat::Rgba4,
        }
    }

    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            ColorFormat::Rgba8 => 4,
            ColorFormat::Rgb8 => 3,
            ColorFormat::Rgb565 | ColorFormat::Rgb5A1 | ColorFormat::Rgba4 => 2,
        }
    }

    /// decodes one pixel to straight 8-bit RGBA.
    pub fn decode(self, bytes: &[u8]) -> [u8; 4] {
        match self {
            // the GPU stores color components in ABGR order in memory.
            ColorFormat::Rgba8 => [bytes[3], bytes[2], bytes[1], bytes[0]],
            ColorFormat::Rgb8 => [bytes[2], bytes[1], bytes[0], 0xFF],
            ColorFormat::Rgb565 => {
                let v = u16::from_le_bytes([bytes[0], bytes[1]]);
                [
                    expand5(((v >> 11) & 0x1F) as u8),
                    expand6(((v >> 5) & 0x3F) as u8),
                    expand5((v & 0x1F) as u8),
                    0xFF,
                ]
            }
            ColorFormat::Rgb5A1 => {
                let v = u16::from_le_bytes([bytes[0], bytes[1]]);
                [
                    expand5(((v >> 11) & 0x1F) as u8),
                    expand5(((v >> 6) & 0x1F) as u8),
                    expand5(((v >> 1) & 0x1F) as u8),
                    if v & 1 != 0 { 0xFF } else { 0 },
                ]
            }
            ColorFormat::Rgba4 => {
                let v = u16::from_le_bytes([bytes[0], bytes[1]]);
                [
                    expand4(((v >> 12) & 0xF) as u8),
                    expand4(((v >> 8) & 0xF) as u8),
                    expand4(((v >> 4) & 0xF) as u8),
                    expand4((v & 0xF) as u8),
                ]
            }
        }
    }

    /// encodes straight 8-bit RGBA into this format.
    pub fn encode(self, rgba: [u8; 4], out: &mut [u8]) {
        match self {
            ColorFormat::Rgba8 => {
                out[0] = rgba[3];
                out[1] = rgba[2];
                out[2] = rgba[1];
                out[3] = rgba[0];
            }
            ColorFormat::Rgb8 => {
                out[0] = rgba[2];
                out[1] = rgba[1];
                out[2] = rgba[0];
            }
            ColorFormat::Rgb565 => {
                let v = ((rgba[0] as u16 >> 3) << 11)
                    | ((rgba[1] as u16 >> 2) << 5)
                    | (rgba[2] as u16 >> 3);
                out[..2].copy_from_slice(&v.to_le_bytes());
            }
            ColorFormat::Rgb5A1 => {
                let v = ((rgba[0] as u16 >> 3) << 11)
                    | ((rgba[1] as u16 >> 3) << 6)
                    | ((rgba[2] as u16 >> 3) << 1)
                    | (rgba[3] >= 0x80) as u16;
                out[..2].copy_from_slice(&v.to_le_bytes());
            }
            ColorFormat::Rgba4 => {
                let v = ((rgba[0] as u16 >> 4) << 12)
                    | ((rgba[1] as u16 >> 4) << 8)
                    | ((rgba[2] as u16 >> 4) << 4)
                    | (rgba[3] as u16 >> 4);
                out[..2].copy_from_slice(&v.to_le_bytes());
            }
        }
    }
}

#[inline]
const fn expand4(v: u8) -> u8 {
    (v << 4) | v
}

#[inline]
const fn expand5(v: u8) -> u8 {
    (v << 3) | (v >> 2)
}

#[inline]
const fn expand6(v: u8) -> u8 {
    (v << 2) | (v >> 4)
}

/// offset in pixels of (x, y) within an 8x8 Morton-ordered tile.
#[inline]
pub const fn morton_interleave(x: u32, y: u32) -> u32 {
    let x = x & 7;
    let y = y & 7;
    (x & 1) | ((y & 1) << 1) | ((x & 2) << 1) | ((y & 2) << 2) | ((x & 4) << 2) | ((y & 4) << 3)
}

/// byte offset of pixel (x, y) in a tiled image width pixels across.
#[inline]
pub const fn morton_offset(x: u32, y: u32, width: u32, bytes_per_pixel: u32) -> u32 {
    // whole tiles first, then the position inside the tile.
    let tile_x = x / 8;
    let tile_y = y / 8;
    let tiles_per_row = width / 8;
    let tile_index = tile_y * tiles_per_row + tile_x;
    (tile_index * 64 + morton_interleave(x, y)) * bytes_per_pixel
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn morton_covers_a_tile_exactly_once() {
        let mut seen = [false; 64];
        for y in 0..8 {
            for x in 0..8 {
                let index = morton_interleave(x, y) as usize;
                assert!(!seen[index], "duplicate index {index} at ({x},{y})");
                seen[index] = true;
            }
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn morton_matches_known_values() {
        // the first row of a tile interleaves to 0, 1, 4, 5, 16, 17, 20, 21.
        let row: Vec<u32> = (0..8).map(|x| morton_interleave(x, 0)).collect();
        assert_eq!(row, vec![0, 1, 4, 5, 16, 17, 20, 21]);
        // the first column interleaves to 0, 2, 8, 10, 32, 34, 40, 42.
        let column: Vec<u32> = (0..8).map(|y| morton_interleave(0, y)).collect();
        assert_eq!(column, vec![0, 2, 8, 10, 32, 34, 40, 42]);
    }

    #[test]
    fn color_formats_round_trip() {
        for format in [
            ColorFormat::Rgba8,
            ColorFormat::Rgb8,
            ColorFormat::Rgb565,
            ColorFormat::Rgb5A1,
            ColorFormat::Rgba4,
        ] {
            let mut buf = [0u8; 4];
            // use values that survive the precision of every format.
            let original = [0xFF, 0x00, 0xFF, 0xFF];
            format.encode(original, &mut buf);
            let decoded = format.decode(&buf);
            assert_eq!(decoded, original, "{format:?}");
        }
    }
}
