//! a stand-in for the console's shared font.

/// where the font block a title is handed starts, past the status word and
/// the rest of the block header.
pub const FONT_OFFSET: u32 = 0x80;

/// section offsets inside a BCFNT point at the section's *body*, past its
/// magic and size, and are absolute addresses in the block as mapped.
const SECTION_BODY: u32 = 8;

/// the glyph sheet's shape, one cell per character, in a single sheet.
const CELL_WIDTH: u8 = 24;
const CELL_HEIGHT: u8 = 24;
/// how much each source bitmap pixel is enlarged.
const GLYPH_SCALE: u16 = 3;
/// horizontal advance per character, the source glyphs leave a column or
/// two of their eight blank, so the cell width would space text out.
const CHARACTER_WIDTH: u8 = 21;
const SHEET_WIDTH: u16 = 256;
const SHEET_HEIGHT: u16 = 256;
/// 4-bit alpha, the format a real shared font's sheets use.
const SHEET_FORMAT_A4: u16 = 11;

/// characters the map covers, printable ASCII.
const CODE_BEGIN: u16 = 0x20;
const CODE_END: u16 = 0x7E;

/// builds the font block, laid out for being mapped at base.
pub fn build(base: u32) -> Vec<u8> {
    let glyphs = (CODE_END - CODE_BEGIN + 1) as u32;
    let sheet_size = SHEET_WIDTH as u32 * SHEET_HEIGHT as u32 / 2; // 4 bits per pixel

    // sizes first, so every section can be given the address of the next.
    const CFNT_SIZE: u32 = 0x14;
    const FINF_SIZE: u32 = 0x20;
    const TGLP_HEADER_SIZE: u32 = 0x20;
    const CMAP_HEADER_SIZE: u32 = 0x14;
    const CWDH_HEADER_SIZE: u32 = 0x10;

    let finf_at = FONT_OFFSET + CFNT_SIZE;
    let tglp_at = finf_at + FINF_SIZE;
    let sheet_at = tglp_at + TGLP_HEADER_SIZE;
    let cwdh_at = sheet_at + sheet_size;
    let cwdh_size = CWDH_HEADER_SIZE + align_up(glyphs * 3, 4);
    let cmap_at = cwdh_at + cwdh_size;
    // direct mapping needs one word of payload, the index the range starts at.
    let cmap_size = CMAP_HEADER_SIZE + 4;
    let total = cmap_at + cmap_size;

    let mut out = vec![0u8; total as usize];

    // the status word a title polls before it touches anything else.
    put32(&mut out, 0x00, 2);

    // CFNT, the file header.
    write_magic(&mut out, FONT_OFFSET, b"CFNT");
    put16(&mut out, FONT_OFFSET + 4, 0xFEFF); // little endian
    put16(&mut out, FONT_OFFSET + 6, CFNT_SIZE as u16);
    put32(&mut out, FONT_OFFSET + 8, 0x0300_0000); // version
    put32(&mut out, FONT_OFFSET + 12, total - FONT_OFFSET);
    put32(&mut out, FONT_OFFSET + 16, 4); // FINF, TGLP, CWDH, CMAP

    // FINF, the font's metrics, and where the other sections are.
    write_magic(&mut out, finf_at, b"FINF");
    put32(&mut out, finf_at + 4, FINF_SIZE);
    out[finf_at as usize + 8] = 1; // glyph-sheet font
    out[finf_at as usize + 9] = CELL_HEIGHT + 1; // line feed
    put16(&mut out, finf_at + 10, 0); // index used for unmapped characters
    out[finf_at as usize + 12] = 0; // default left spacing
    out[finf_at as usize + 13] = CELL_WIDTH; // default glyph width
    out[finf_at as usize + 14] = CHARACTER_WIDTH; // default character width
    out[finf_at as usize + 15] = 1; // UTF-16
    put32(&mut out, finf_at + 16, base + tglp_at + SECTION_BODY);
    put32(&mut out, finf_at + 20, base + cwdh_at + SECTION_BODY);
    put32(&mut out, finf_at + 24, base + cmap_at + SECTION_BODY);
    out[finf_at as usize + 28] = CELL_HEIGHT;
    out[finf_at as usize + 29] = CELL_WIDTH;
    out[finf_at as usize + 30] = CELL_HEIGHT - GLYPH_SCALE as u8; // ascent

    // TGLP, the sheet the glyph images live in.
    write_magic(&mut out, tglp_at, b"TGLP");
    put32(&mut out, tglp_at + 4, TGLP_HEADER_SIZE + sheet_size);
    out[tglp_at as usize + 8] = CELL_WIDTH;
    out[tglp_at as usize + 9] = CELL_HEIGHT;
    out[tglp_at as usize + 10] = CELL_HEIGHT - GLYPH_SCALE as u8; // baseline
    out[tglp_at as usize + 11] = CELL_WIDTH; // widest character
    put32(&mut out, tglp_at + 12, sheet_size);
    put16(&mut out, tglp_at + 16, 1); // one sheet
    put16(&mut out, tglp_at + 18, SHEET_FORMAT_A4);
    put16(&mut out, tglp_at + 20, SHEET_WIDTH / CELL_WIDTH as u16);
    put16(&mut out, tglp_at + 22, SHEET_HEIGHT / CELL_HEIGHT as u16);
    put16(&mut out, tglp_at + 24, SHEET_WIDTH);
    put16(&mut out, tglp_at + 26, SHEET_HEIGHT);
    put32(&mut out, tglp_at + 28, base + sheet_at);
    draw_glyph_sheet(&mut out, sheet_at);

    // CWDH, how wide each glyph is. The font is monospaced, so all equal.
    write_magic(&mut out, cwdh_at, b"CWDH");
    put32(&mut out, cwdh_at + 4, cwdh_size);
    put16(&mut out, cwdh_at + 8, 0);
    put16(&mut out, cwdh_at + 10, (glyphs - 1) as u16);
    put32(&mut out, cwdh_at + 12, 0); // no further width sections
    for glyph in 0..glyphs {
        let at = (cwdh_at + CWDH_HEADER_SIZE + glyph * 3) as usize;
        out[at] = 0; // left spacing
        out[at + 1] = CELL_WIDTH; // glyph width
        out[at + 2] = CHARACTER_WIDTH; // character width
    }

    // CMAP, which glyph each character uses. One direct range is enough.
    write_magic(&mut out, cmap_at, b"CMAP");
    put32(&mut out, cmap_at + 4, cmap_size);
    put16(&mut out, cmap_at + 8, CODE_BEGIN);
    put16(&mut out, cmap_at + 10, CODE_END);
    put16(&mut out, cmap_at + 12, 0); // direct mapping
    put16(&mut out, cmap_at + 14, 0);
    put32(&mut out, cmap_at + 16, 0); // no further maps
    put16(&mut out, cmap_at + 20, 0); // first glyph index of the range

    out
}

/// rewrites a font block's internal pointers for the address the guest has just
/// mapped it at.
pub fn relocate(memory: &mut crate::memory::Memory, block: u32, paddr: u32) {
    use zakuro_cpu::Bus;

    let header = block + FONT_OFFSET;
    let mut magic = [0u8; 4];
    memory.read_bytes(header, &mut magic);
    if &magic != b"CFNT" {
        return;
    }

    let header_size = memory.read16(header + 6) as u32;
    let sections = memory.read32(header + 16);

    let mut finf = None;
    let mut tglp = None;
    let mut cwdh = None;
    let mut cmap = None;

    let mut at = header + header_size;
    for _ in 0..sections.min(16) {
        let mut magic = [0u8; 4];
        memory.read_bytes(at, &mut magic);
        let size = memory.read32(at + 4);
        match &magic {
            b"FINF" => finf = finf.or(Some(at)),
            b"TGLP" => tglp = tglp.or(Some(at)),
            b"CWDH" => cwdh = cwdh.or(Some(at)),
            b"CMAP" => cmap = cmap.or(Some(at)),
            _ => break,
        }
        if size == 0 {
            break;
        }
        at += size;
    }

    let Some(finf) = finf else { return };
    // the block is mapped read-only, so the edits go to the physical pages
    // behind it rather than through the guest's view of them.
    let mut put = |at: u32, value: u32| {
        memory.write_physical(paddr + (at - block), &value.to_le_bytes());
    };
    for (field, section) in [(16, tglp), (20, cwdh), (24, cmap)] {
        if let Some(section) = section {
            put(finf + field, section + SECTION_BODY);
        }
    }
    if let Some(tglp) = tglp {
        // the sheet follows its section header directly.
        put(tglp + 28, tglp + 0x20);
    }

    log::debug!("relocated the shared font's sections for its mapping at 0x{block:08X}");
}

/// paints each character's bitmap into its cell on the glyph sheet.
fn draw_glyph_sheet(out: &mut [u8], sheet_at: u32) {
    use crate::services::glyphs::GLYPHS;

    let columns = SHEET_WIDTH / CELL_WIDTH as u16;
    for (index, glyph) in GLYPHS.iter().enumerate() {
        let cell_x = (index as u16 % columns) * CELL_WIDTH as u16;
        let cell_y = (index as u16 / columns) * CELL_HEIGHT as u16;

        for (row, bits) in glyph.iter().enumerate() {
            for column in 0..8u16 {
                if bits >> (7 - column) & 1 == 0 {
                    continue;
                }
                for dy in 0..GLYPH_SCALE {
                    for dx in 0..GLYPH_SCALE {
                        let x = cell_x + column * GLYPH_SCALE + dx;
                        let y = cell_y + row as u16 * GLYPH_SCALE + dy;
                        let texel = morton_index(x as u32, y as u32, SHEET_WIDTH as u32);
                        // two 4-bit texels per byte, low nibble first.
                        let at = (sheet_at + texel / 2) as usize;
                        if at < out.len() {
                            out[at] |= if texel.is_multiple_of(2) { 0x0F } else { 0xF0 };
                        }
                    }
                }
            }
        }
    }
}

/// index of a texel in a PICA-tiled image, 8x8 tiles in raster order, and a
/// Z-order curve inside each tile.
fn morton_index(x: u32, y: u32, width: u32) -> u32 {
    let (tile_x, tile_y) = (x / 8, y / 8);
    let (px, py) = (x % 8, y % 8);
    let mut inside = 0;
    for bit in 0..3 {
        inside |= ((px >> bit) & 1) << (2 * bit);
        inside |= ((py >> bit) & 1) << (2 * bit + 1);
    }
    (tile_y * (width / 8) + tile_x) * 64 + inside
}

fn write_magic(out: &mut [u8], at: u32, magic: &[u8; 4]) {
    out[at as usize..at as usize + 4].copy_from_slice(magic);
}

fn put32(out: &mut [u8], at: u32, value: u32) {
    out[at as usize..at as usize + 4].copy_from_slice(&value.to_le_bytes());
}

fn put16(out: &mut [u8], at: u32, value: u16) {
    out[at as usize..at as usize + 2].copy_from_slice(&value.to_le_bytes());
}

fn align_up(value: u32, align: u32) -> u32 {
    value.div_ceil(align) * align
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read32(data: &[u8], at: u32) -> u32 {
        u32::from_le_bytes(data[at as usize..at as usize + 4].try_into().unwrap())
    }

    /// walks the block the way a title's font loader does, check the status,
    /// find the header, then follow each section offset and confirm it lands
    /// on that section's magic.
    #[test]
    fn the_block_parses_as_a_font() {
        let base = 0x1800_0000;
        let font = build(base);

        assert_eq!(read32(&font, 0), 2, "the status word should say loaded");
        assert_eq!(&font[0x80..0x84], b"CFNT");
        assert_eq!(read32(&font, FONT_OFFSET + 16), 4, "four sections");
        assert_eq!(
            read32(&font, FONT_OFFSET + 12) as usize,
            font.len() - FONT_OFFSET as usize,
            "the recorded size should match the block"
        );

        // FINF follows the header, and its three offsets are absolute.
        let finf = FONT_OFFSET + 0x14;
        assert_eq!(&font[finf as usize..finf as usize + 4], b"FINF");
        for (offset_at, magic) in [(16, b"TGLP"), (20, b"CWDH"), (24, b"CMAP")] {
            let pointer = read32(&font, finf + offset_at);
            // the offset points past the magic and size, so step back.
            let at = (pointer - base - SECTION_BODY) as usize;
            assert_eq!(
                &font[at..at + 4],
                magic,
                "offset at +{offset_at} should reach {}",
                std::str::from_utf8(magic).unwrap()
            );
        }
    }

    /// the sheet has to be entirely inside the block, or a title reading a
    /// glyph walks off the end of the mapping.
    #[test]
    fn the_glyph_sheet_is_within_the_block() {
        let base = 0x1800_0000;
        let font = build(base);
        let finf = FONT_OFFSET + 0x14;
        let tglp = (read32(&font, finf + 16) - base - SECTION_BODY) + SECTION_BODY;

        let sheet_size = read32(&font, tglp + 4);
        let sheet_at = read32(&font, tglp + 20) - base;
        assert!(
            (sheet_at + sheet_size) as usize <= font.len(),
            "the sheet runs past the end of the block"
        );
    }
}
