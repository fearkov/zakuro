//! a stand-in for the console's shared font. its characters come from a
//! font the host has for Japanese, the alphabets and Japanese a title
//! writes in, or where there is none, printable ASCII from a small bitmap
//! font of our own.

/// where the font block a title is handed starts, past the status word and
/// the rest of the block header.
pub const FONT_OFFSET: u32 = 0x80;

/// section offsets inside a BCFNT point at the section's *body*, past its
/// magic and size, and are absolute addresses in the block as mapped.
const SECTION_BODY: u32 = 8;

/// every glyph's cell, in both fonts.
const CELL_WIDTH: u8 = 24;
const CELL_HEIGHT: u8 = 24;
/// cells sit a pixel apart on the sheet, which is where a font's renderer
/// looks for each glyph.
const CELL_STRIDE_X: u16 = CELL_WIDTH as u16 + 1;
const CELL_STRIDE_Y: u16 = CELL_HEIGHT as u16 + 1;
const SHEET_WIDTH: u16 = 256;
const SHEET_HEIGHT: u16 = 256;
const COLUMNS: u16 = SHEET_WIDTH / CELL_STRIDE_X;
const ROWS: u16 = SHEET_HEIGHT / CELL_STRIDE_Y;
/// 4-bit alpha, the format a real shared font's sheets use.
const SHEET_FORMAT_A4: u16 = 11;
/// the GPU takes a texture's address in 8-byte steps, a console's font has
/// its sheets further aligned, to 128 bytes.
const SHEET_ALIGN: u32 = 0x80;
/// the console's own font block, which a title may map no more of.
const BLOCK_LIMIT: u32 = 0x33_2000;
/// characters in a row that get a map of their own rather than going in
/// the list of the rest.
const DIRECT_RUN: usize = 16;

/// the bitmap font enlarges each pixel this much, and moves on less than
/// the cell, its glyphs leave a column or two of their eight blank.
const GLYPH_SCALE: u16 = 3;
const CHARACTER_WIDTH: u8 = 21;

/// the host's font at this many pixels an em, with its baseline on this
/// row of the cell, which leaves room below for descenders.
const EM: f32 = 22.0;
const HOST_BASELINE: u8 = 20;

/// one character, its image in its cell and how wide it is.
#[derive(Clone)]
struct Glyph {
    code: u16,
    /// the cell's pixels, rows of CELL_WIDTH, each 0 to 15.
    alpha: Box<[u8]>,
    /// where the image starts from the pen, how much of it to draw, and
    /// how far the pen moves on.
    left: i8,
    width: u8,
    advance: u8,
}

/// a font's glyphs in the order of their characters, the row its baseline
/// is on, and how far its widest characters move the pen.
struct Face {
    glyphs: Vec<Glyph>,
    baseline: u8,
    advance: u8,
}

/// builds the font block, laid out for being mapped at base.
pub fn build(base: u32) -> Vec<u8> {
    match host_face() {
        Some(face) => write(base, face),
        None => write(base, &bitmap_face()),
    }
}

/// the font block for face, laid out for being mapped at base.
fn write(base: u32, face: &Face) -> Vec<u8> {
    let glyphs = &face.glyphs;
    let count = glyphs.len() as u32;
    let per_sheet = (COLUMNS * ROWS) as u32;
    let sheets = count.div_ceil(per_sheet).max(1);
    let sheet_size = SHEET_WIDTH as u32 * SHEET_HEIGHT as u32 / 2; // 4 bits per pixel
    let maps = maps(glyphs);

    // sizes first, so every section can be given the address of the next.
    const CFNT_SIZE: u32 = 0x14;
    const FINF_SIZE: u32 = 0x20;
    const TGLP_HEADER_SIZE: u32 = 0x20;
    const CMAP_HEADER_SIZE: u32 = 0x14;
    const CWDH_HEADER_SIZE: u32 = 0x10;

    let finf_at = FONT_OFFSET + CFNT_SIZE;
    let tglp_at = finf_at + FINF_SIZE;
    let sheet_at = align_up(tglp_at + TGLP_HEADER_SIZE, SHEET_ALIGN);
    let cwdh_at = sheet_at + sheets * sheet_size;
    let cwdh_size = CWDH_HEADER_SIZE + align_up(count * 3, 4);
    let mut cmap_at = Vec::with_capacity(maps.len());
    let mut end = cwdh_at + cwdh_size;
    for map in &maps {
        cmap_at.push(end);
        end += CMAP_HEADER_SIZE + align_up(map.payload(), 4);
    }
    let total = end;
    if total > BLOCK_LIMIT {
        log::warn!("the shared font's stand-in takes 0x{total:X} bytes, more than a title may map");
    }

    let mut out = vec![0u8; total as usize];

    // the status word a title polls before it touches anything else, the
    // region's font, the standard one, and how big the font is, which some
    // titles read to tell the font from a resource of their own
    put32(&mut out, 0x00, 2);
    put32(&mut out, 0x04, 1);
    put32(&mut out, 0x08, total - FONT_OFFSET);

    // the file header, CFNU rather than CFNT as the console keeps the font,
    // its pointers already absolute, which the SDK takes as it is instead
    // of turning offsets into pointers a second time
    write_magic(&mut out, FONT_OFFSET, b"CFNU");
    put16(&mut out, FONT_OFFSET + 4, 0xFEFF); // little endian
    put16(&mut out, FONT_OFFSET + 6, CFNT_SIZE as u16);
    put32(&mut out, FONT_OFFSET + 8, 0x0300_0000); // version
    put32(&mut out, FONT_OFFSET + 12, total - FONT_OFFSET);
    put32(&mut out, FONT_OFFSET + 16, 3 + maps.len() as u32); // FINF, TGLP, CWDH, the CMAPs

    // FINF, the font's metrics, and where the other sections are. a
    // character the font lacks shows as its question mark
    let unknown = glyphs.iter().position(|glyph| glyph.code == u16::from(b'?')).unwrap_or(0);
    write_magic(&mut out, finf_at, b"FINF");
    put32(&mut out, finf_at + 4, FINF_SIZE);
    out[finf_at as usize + 8] = 1; // glyph-sheet font
    out[finf_at as usize + 9] = CELL_HEIGHT + 1; // line feed
    put16(&mut out, finf_at + 10, unknown as u16);
    out[finf_at as usize + 12] = 0; // default left spacing
    out[finf_at as usize + 13] = CELL_WIDTH; // default glyph width
    out[finf_at as usize + 14] = face.advance; // default character width
    out[finf_at as usize + 15] = 1; // UTF-16
    put32(&mut out, finf_at + 16, base + tglp_at + SECTION_BODY);
    put32(&mut out, finf_at + 20, base + cwdh_at + SECTION_BODY);
    put32(&mut out, finf_at + 24, base + cmap_at[0] + SECTION_BODY);
    out[finf_at as usize + 28] = CELL_HEIGHT;
    out[finf_at as usize + 29] = CELL_WIDTH;
    out[finf_at as usize + 30] = face.baseline; // ascent

    // TGLP, the sheets the glyph images live in.
    write_magic(&mut out, tglp_at, b"TGLP");
    put32(&mut out, tglp_at + 4, cwdh_at - tglp_at);
    out[tglp_at as usize + 8] = CELL_WIDTH;
    out[tglp_at as usize + 9] = CELL_HEIGHT;
    out[tglp_at as usize + 10] = face.baseline;
    out[tglp_at as usize + 11] = glyphs.iter().map(|glyph| glyph.advance).max().unwrap_or(CELL_WIDTH).max(CELL_WIDTH);
    put32(&mut out, tglp_at + 12, sheet_size);
    put16(&mut out, tglp_at + 16, sheets as u16);
    put16(&mut out, tglp_at + 18, SHEET_FORMAT_A4);
    put16(&mut out, tglp_at + 20, COLUMNS);
    put16(&mut out, tglp_at + 22, ROWS);
    put16(&mut out, tglp_at + 24, SHEET_WIDTH);
    put16(&mut out, tglp_at + 26, SHEET_HEIGHT);
    put32(&mut out, tglp_at + 28, base + sheet_at);
    for (index, glyph) in glyphs.iter().enumerate() {
        let (sheet, cell) = (index as u32 / per_sheet, index as u32 % per_sheet);
        draw_glyph(&mut out[(sheet_at + sheet * sheet_size) as usize..][..sheet_size as usize], cell as u16, glyph);
    }

    // CWDH, how wide each glyph is.
    write_magic(&mut out, cwdh_at, b"CWDH");
    put32(&mut out, cwdh_at + 4, cwdh_size);
    put16(&mut out, cwdh_at + 8, 0);
    put16(&mut out, cwdh_at + 10, count.saturating_sub(1) as u16);
    put32(&mut out, cwdh_at + 12, 0); // no further width sections
    for (index, glyph) in glyphs.iter().enumerate() {
        let at = (cwdh_at + CWDH_HEADER_SIZE) as usize + index * 3;
        out[at] = glyph.left as u8;
        out[at + 1] = glyph.width;
        out[at + 2] = glyph.advance;
    }

    // CMAP, which glyph each character uses, each map pointing on to the
    // next, runs of characters in a row first and the list of the rest last.
    for (i, (map, &at)) in maps.iter().zip(&cmap_at).enumerate() {
        let next = cmap_at.get(i + 1).map_or(0, |&next| base + next + SECTION_BODY);
        write_magic(&mut out, at, b"CMAP");
        put32(&mut out, at + 4, CMAP_HEADER_SIZE + align_up(map.payload(), 4));
        put32(&mut out, at + 16, next);
        match map {
            Map::Direct { begin, end, first } => {
                put16(&mut out, at + 8, *begin);
                put16(&mut out, at + 10, *end);
                put16(&mut out, at + 12, 0); // direct
                put16(&mut out, at + 20, *first);
            }
            Map::Scan(pairs) => {
                put16(&mut out, at + 8, pairs.first().map_or(0, |pair| pair.0));
                put16(&mut out, at + 10, pairs.last().map_or(0, |pair| pair.0));
                put16(&mut out, at + 12, 2); // scan
                put16(&mut out, at + 20, pairs.len() as u16);
                for (n, &(code, index)) in pairs.iter().enumerate() {
                    put16(&mut out, at + 22 + n as u32 * 4, code);
                    put16(&mut out, at + 24 + n as u32 * 4, index);
                }
            }
        }
    }

    out
}

/// how a CMAP gives the glyph of a character.
enum Map {
    /// characters from begin to end, the glyphs from first on in order.
    Direct { begin: u16, end: u16, first: u16 },
    /// pairs of a character and its glyph, in the order of the characters,
    /// which the SDK searches.
    Scan(Vec<(u16, u16)>),
}

impl Map {
    fn payload(&self) -> u32 {
        match self {
            Map::Direct { .. } => 2,
            Map::Scan(pairs) => 2 + pairs.len() as u32 * 4,
        }
    }
}

/// the maps that give each glyph's character, glyphs in order of their
/// characters: runs of characters in a row each a direct map, the rest in
/// one scanned list after them.
fn maps(glyphs: &[Glyph]) -> Vec<Map> {
    let mut maps = Vec::new();
    let mut rest = Vec::new();
    let mut start = 0;
    while start < glyphs.len() {
        let mut end = start + 1;
        while end < glyphs.len() && glyphs[end].code == glyphs[end - 1].code + 1 {
            end += 1;
        }
        if end - start >= DIRECT_RUN {
            maps.push(Map::Direct { begin: glyphs[start].code, end: glyphs[end - 1].code, first: start as u16 });
        } else {
            rest.extend((start..end).map(|index| (glyphs[index].code, index as u16)));
        }
        start = end;
    }
    if !rest.is_empty() || maps.is_empty() {
        maps.push(Map::Scan(rest));
    }
    maps
}

/// puts a glyph's image in a cell of a sheet, laid out as the GPU reads
/// a texture.
fn draw_glyph(sheet: &mut [u8], cell: u16, glyph: &Glyph) {
    // past the line that starts each cell
    let cell_x = (cell % COLUMNS) * CELL_STRIDE_X + 1;
    let cell_y = (cell / COLUMNS) * CELL_STRIDE_Y + 1;
    for (row, line) in glyph.alpha.chunks(CELL_WIDTH as usize).enumerate() {
        for (column, &alpha) in line.iter().enumerate().filter(|&(_, &alpha)| alpha != 0) {
            let texel = morton_index((cell_x + column as u16) as u32, (cell_y + row as u16) as u32, SHEET_WIDTH as u32);
            // two 4-bit texels per byte, low nibble first.
            sheet[texel as usize / 2] |= (alpha & 0xF) << if texel.is_multiple_of(2) { 0 } else { 4 };
        }
    }
}

/// printable ASCII from the 8x8 bitmap font, each pixel enlarged.
fn bitmap_face() -> Face {
    use crate::services::glyphs::GLYPHS;

    let glyphs = GLYPHS
        .iter()
        .enumerate()
        .map(|(index, rows)| {
            let mut alpha = vec![0u8; CELL_WIDTH as usize * CELL_HEIGHT as usize].into_boxed_slice();
            for (row, bits) in rows.iter().enumerate() {
                for column in (0..8).filter(|column| bits >> (7 - column) & 1 != 0) {
                    for dy in 0..GLYPH_SCALE as usize {
                        for dx in 0..GLYPH_SCALE as usize {
                            let (x, y) = (column * GLYPH_SCALE as usize + dx, row * GLYPH_SCALE as usize + dy);
                            alpha[y * CELL_WIDTH as usize + x] = 15;
                        }
                    }
                }
            }
            Glyph { code: 0x20 + index as u16, alpha, left: 0, width: CELL_WIDTH, advance: CHARACTER_WIDTH }
        })
        .collect();
    Face { glyphs, baseline: CELL_HEIGHT - GLYPH_SCALE as u8, advance: CHARACTER_WIDTH }
}

/// the characters the console's font has, near enough: JIS X 0208's, with
/// the NEC row of circled numbers, the Latin alphabets of European titles,
/// and full and half width forms.
fn characters() -> Vec<char> {
    let mut characters: Vec<char> = include_str!("jis0208.txt").chars().filter(|&c| c != '\n').collect();
    let ranges = [
        0x20..=0x7E,
        0xA0..=0x17F,
        0x2010..=0x2027,
        0x2030..=0x203B,
        0x20AC..=0x20AC,
        0x2122..=0x2122,
        0x2190..=0x2199,
        0x2460..=0x2473,
        0x3000..=0x30FF,
        0xFF01..=0xFF9F,
        0xFFE0..=0xFFE6,
    ];
    characters.extend(ranges.into_iter().flatten().filter_map(char::from_u32));
    characters.sort_unstable();
    characters.dedup();
    characters
}

/// the host's font for Japanese made into glyphs, the first time a title
/// asks for the shared font, none where the host has no such font.
fn host_face() -> Option<&'static Face> {
    static FACE: std::sync::OnceLock<Option<Face>> = std::sync::OnceLock::new();
    FACE.get_or_init(|| {
        let (path, data) = host_font()?;
        let face = rasterize(&data)?;
        log::info!("the shared font's stand-in draws {} characters from {}", face.glyphs.len(), path.display());
        Some(face)
    })
    .as_ref()
}

/// glyphs for the characters a font has, at EM pixels an em.
fn rasterize(data: &[u8]) -> Option<Face> {
    use ab_glyph::{point, Font, FontRef, PxScale};
    use rayon::prelude::*;

    let font = FontRef::try_from_slice_and_index(data, 0).ok()?;
    let per_unit = EM / font.units_per_em()?;
    let scale = PxScale::from(font.height_unscaled() * per_unit);
    let (width, height) = (CELL_WIDTH as usize, CELL_HEIGHT as usize);
    let glyphs: Vec<Glyph> = characters()
        .par_iter()
        .filter_map(|&character| {
            let id = font.glyph_id(character);
            // the glyph a font has for what it lacks
            if id.0 == 0 {
                return None;
            }
            let advance = (font.h_advance_unscaled(id) * per_unit).round().clamp(0.0, 255.0) as u8;
            let mut alpha = vec![0u8; width * height].into_boxed_slice();
            let (mut left, mut drawn) = (0, 0);
            if let Some(outline) = font.outline_glyph(id.with_scale_and_position(scale, point(0.0, HOST_BASELINE as f32))) {
                let bounds = outline.px_bounds();
                left = bounds.min.x.clamp(-128.0, 127.0) as i8;
                drawn = (bounds.max.x - bounds.min.x).clamp(0.0, CELL_WIDTH as f32) as u8;
                outline.draw(|x, y, coverage| {
                    let (x, y) = (x as usize, bounds.min.y as i32 + y as i32);
                    if x < width && (0..height as i32).contains(&y) {
                        let pixel = &mut alpha[y as usize * width + x];
                        *pixel = (*pixel).max((coverage * 15.0).round().clamp(0.0, 15.0) as u8);
                    }
                });
            }
            Some(Glyph { code: character as u16, alpha, left, width: drawn, advance })
        })
        .collect();
    // a font that has none of Japanese is no better than the bitmap one
    if !glyphs.iter().any(|glyph| glyph.code == 0x3042) {
        return None;
    }
    let advance = (font.h_advance_unscaled(font.glyph_id('\u{3042}')) * per_unit).round().clamp(0.0, 255.0) as u8;
    Some(Face { glyphs, baseline: HOST_BASELINE, advance })
}

/// the first font for Japanese the host has, and where it is.
fn host_font() -> Option<(std::path::PathBuf, Vec<u8>)> {
    font_candidates().into_iter().find_map(|path| Some((path.clone(), std::fs::read(&path).ok()?)))
}

/// where Windows keeps its fonts for Japanese.
#[cfg(windows)]
fn font_candidates() -> Vec<std::path::PathBuf> {
    let folder = std::path::PathBuf::from(std::env::var_os("WINDIR").unwrap_or_else(|| "C:\\Windows".into())).join("Fonts");
    ["YuGothM.ttc", "meiryo.ttc", "msgothic.ttc"].iter().map(|name| folder.join(name)).collect()
}

/// where macOS keeps its fonts for Japanese.
#[cfg(target_os = "macos")]
fn font_candidates() -> Vec<std::path::PathBuf> {
    ["/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc", "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"]
        .iter()
        .map(std::path::PathBuf::from)
        .collect()
}

/// where Linux distributions put Noto's fonts for Japanese, the medium
/// weight nearest the console's, or else the font fontconfig picks.
#[cfg(all(unix, not(target_os = "macos")))]
fn font_candidates() -> Vec<std::path::PathBuf> {
    let folders = ["/usr/share/fonts/noto-cjk", "/usr/share/fonts/opentype/noto", "/usr/share/fonts/google-noto-cjk", "/usr/share/fonts/noto"];
    let mut paths: Vec<std::path::PathBuf> = ["NotoSansCJK-Medium.ttc", "NotoSansCJK-Regular.ttc", "NotoSansCJK-Bold.ttc"]
        .iter()
        .flat_map(|name| folders.iter().map(move |folder| std::path::Path::new(folder).join(name)))
        .collect();
    if !paths.iter().any(|path| path.exists()) {
        let matched = std::process::Command::new("fc-match").args(["--format=%{file}", ":lang=ja"]).output();
        if let Some(output) = matched.ok().filter(|output| output.status.success()) {
            paths.push(String::from_utf8_lossy(&output.stdout).into_owned().into());
        }
    }
    paths
}

/// moves a font block's internal pointers to the address the guest has just
/// mapped it at, from wherever it was laid out for before, a dump's console
/// or an earlier mapping.
pub fn relocate(memory: &mut crate::memory::Memory, block: u32, paddr: u32) {
    use zakuro_cpu::Bus;

    let header = block + FONT_OFFSET;
    let mut magic = [0u8; 4];
    memory.read_bytes(header, &mut magic);
    if &magic != b"CFNU" && &magic != b"CFNT" {
        return;
    }

    let header_size = memory.read16(header + 6) as u32;
    let sections = memory.read32(header + 16);
    let (mut finf, mut tglp) = (None, None);
    let mut at = header + header_size;
    for _ in 0..sections.min(16) {
        let mut magic = [0u8; 4];
        memory.read_bytes(at, &mut magic);
        let size = memory.read32(at + 4);
        match &magic {
            b"FINF" => finf = finf.or(Some(at)),
            b"TGLP" => tglp = tglp.or(Some(at)),
            _ => {}
        }
        if size == 0 || (finf.is_some() && tglp.is_some()) {
            break;
        }
        at += size;
    }
    let (Some(finf), Some(tglp)) = (finf, tglp) else { return };

    // FINF's pointer to TGLP tells where the block was laid out for
    let laid_out = memory.read32(finf + 16).wrapping_sub(tglp - block + SECTION_BODY);
    let delta = block.wrapping_sub(laid_out);
    if delta == 0 {
        return;
    }
    // the block is mapped read-only, so the edits go to the physical pages
    // behind it rather than through the guest's view of them.
    let moved = |memory: &mut crate::memory::Memory, at: u32| -> u32 {
        let pointer = memory.read32(at);
        let moved = pointer.wrapping_add(delta);
        memory.write_physical(paddr + (at - block), &moved.to_le_bytes());
        moved
    };
    // FINF's three, the sheet's, then the widths and maps chained on, each
    // pointing at the body of the next past its magic and size
    for field in [16, 20, 24] {
        moved(memory, finf + field);
    }
    moved(memory, tglp + 28);
    for (first, link) in [(finf + 20, 4), (finf + 24, 8)] {
        let mut body = memory.read32(first);
        for _ in 0..256 {
            if memory.read32(body + link) == 0 {
                break;
            }
            body = moved(memory, body + link);
        }
    }

    log::debug!("relocated the shared font for its mapping at 0x{block:08X}");
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

    fn read16(data: &[u8], at: u32) -> u16 {
        u16::from_le_bytes(data[at as usize..at as usize + 2].try_into().unwrap())
    }

    fn read32(data: &[u8], at: u32) -> u32 {
        u32::from_le_bytes(data[at as usize..at as usize + 4].try_into().unwrap())
    }

    const BASE: u32 = 0x1800_0000;
    const FINF: u32 = FONT_OFFSET + 0x14;

    /// the glyph a character has, the way a title's font library finds it:
    /// down the maps from FINF's first, a direct one counting from its
    /// first glyph, a scanned one searched.
    fn glyph_of(font: &[u8], code: u16) -> Option<u16> {
        let mut body = read32(font, FINF + 24);
        while body != 0 {
            let at = body - BASE - SECTION_BODY;
            let (begin, end, method) = (read16(font, at + 8), read16(font, at + 10), read16(font, at + 12));
            if (begin..=end).contains(&code) {
                match method {
                    0 => return Some(read16(font, at + 20) + (code - begin)),
                    2 => {
                        let count = read16(font, at + 20) as u32;
                        let pairs: Vec<(u16, u16)> = (0..count).map(|n| (read16(font, at + 22 + n * 4), read16(font, at + 24 + n * 4))).collect();
                        if let Ok(found) = pairs.binary_search_by_key(&code, |pair| pair.0) {
                            return Some(pairs[found].1);
                        }
                    }
                    _ => panic!("map method {method}"),
                }
            }
            body = read32(font, at + 16);
        }
        None
    }

    /// the 4-bit alpha of a pixel of a glyph's cell.
    fn pixel(font: &[u8], glyph: u16, x: u16, y: u16) -> u8 {
        let tglp = read32(font, FINF + 16) - BASE - SECTION_BODY;
        let sheet_size = read32(font, tglp + 12);
        let sheet_at = read32(font, tglp + 28) - BASE;
        let per_sheet = (COLUMNS * ROWS) as u32;
        let (sheet, cell) = (glyph as u32 / per_sheet, glyph % per_sheet as u16);
        let texel = morton_index(((cell % COLUMNS) * CELL_STRIDE_X + 1 + x) as u32, ((cell / COLUMNS) * CELL_STRIDE_Y + 1 + y) as u32, SHEET_WIDTH as u32);
        let byte = font[(sheet_at + sheet * sheet_size + texel / 2) as usize];
        if texel.is_multiple_of(2) { byte & 0xF } else { byte >> 4 }
    }

    /// a face of the given characters, each with one pixel lit where its
    /// character says, and its code for a width.
    fn face_of(codes: &[u16]) -> Face {
        let glyphs = codes
            .iter()
            .map(|&code| {
                let mut alpha = vec![0u8; CELL_WIDTH as usize * CELL_HEIGHT as usize].into_boxed_slice();
                alpha[(code as usize % CELL_HEIGHT as usize) * CELL_WIDTH as usize + code as usize % CELL_WIDTH as usize] = 9;
                Glyph { code, alpha, left: -1, width: 20, advance: (code % 23) as u8 }
            })
            .collect();
        Face { glyphs, baseline: 20, advance: 22 }
    }

    /// walks the block the way a title's font loader does, check the status,
    /// find the header, then follow each section offset and confirm it lands
    /// on that section's magic.
    #[test]
    fn the_block_parses_as_a_font() {
        let font = write(BASE, &bitmap_face());

        assert_eq!(read32(&font, 0), 2, "the status word should say loaded");
        assert_eq!(&font[0x80..0x84], b"CFNU");
        assert_eq!(read32(&font, FONT_OFFSET + 16), 4, "four sections");
        assert_eq!(
            read32(&font, FONT_OFFSET + 12) as usize,
            font.len() - FONT_OFFSET as usize,
            "the recorded size should match the block"
        );

        // FINF follows the header, and its three offsets are absolute.
        assert_eq!(&font[FINF as usize..FINF as usize + 4], b"FINF");
        for (offset_at, magic) in [(16, b"TGLP"), (20, b"CWDH"), (24, b"CMAP")] {
            let pointer = read32(&font, FINF + offset_at);
            // the offset points past the magic and size, so step back.
            let at = (pointer - BASE - SECTION_BODY) as usize;
            assert_eq!(
                &font[at..at + 4],
                magic,
                "offset at +{offset_at} should reach {}",
                std::str::from_utf8(magic).unwrap()
            );
        }
        assert_eq!(glyph_of(&font, u16::from(b'A')), Some(0x41 - 0x20));
        assert_eq!(glyph_of(&font, 0x3042), None);
    }

    /// the sheets are where the GPU can be pointed at, an address it takes
    /// in 8-byte steps. one 4 bytes off had it read every glyph shifted.
    #[test]
    fn the_glyph_sheets_are_aligned() {
        let font = write(BASE, &face_of(&(0x20..0x400).collect::<Vec<u16>>()));
        let tglp = read32(&font, FINF + 16) - BASE - SECTION_BODY;
        assert_eq!(read32(&font, tglp + 28) % SHEET_ALIGN, 0);
        assert_eq!(read32(&font, tglp + 12) % SHEET_ALIGN, 0, "and each sheet after the first");
    }

    /// the sheets have to be entirely inside the block, or a title reading a
    /// glyph walks off the end of the mapping.
    #[test]
    fn the_glyph_sheets_are_within_the_block() {
        for face in [bitmap_face(), face_of(&(0x20..0x1000).collect::<Vec<u16>>())] {
            let font = write(BASE, &face);
            let tglp = read32(&font, FINF + 16) - BASE - SECTION_BODY;
            let sheets = read16(&font, tglp + 16) as u32;
            let end = read32(&font, tglp + 28) - BASE + sheets * read32(&font, tglp + 12);
            assert!(end <= read32(&font, FINF + 20) - BASE - SECTION_BODY, "the sheets run into the widths");
            assert!(sheets as usize * (COLUMNS * ROWS) as usize >= face.glyphs.len());
        }
    }

    /// every character finds its own glyph, in runs of characters in a row
    /// and scattered ones alike, the glyph's pixel and widths where it put
    /// them, and a character the font lacks finds none.
    #[test]
    fn every_character_finds_its_glyph() {
        let mut codes: Vec<u16> = (0x20..=0x7E).collect();
        codes.extend([0xA9, 0xE9, 0x2026]);
        codes.extend(0x3041..=0x3096);
        codes.extend([0x4E00, 0x4E03, 0x4E07, 0x4E09, 0x4E0A, 0x6F22, 0x9F8D, 0xFF1F]);
        let face = face_of(&codes);
        let font = write(BASE, &face);
        assert!(read32(&font, FONT_OFFSET + 16) > 4, "runs and a list of the rest");
        let cwdh = read32(&font, FINF + 20) - BASE - SECTION_BODY + 0x10;
        for (index, &code) in codes.iter().enumerate() {
            assert_eq!(glyph_of(&font, code), Some(index as u16), "character {code:04X}");
            let (x, y) = (code % CELL_WIDTH as u16, code % CELL_HEIGHT as u16);
            assert_eq!(pixel(&font, index as u16, x, y), 9, "character {code:04X}'s pixel");
            let widths = &font[(cwdh + index as u32 * 3) as usize..][..3];
            assert_eq!(widths, [0xFF, 20, (code % 23) as u8], "character {code:04X}'s widths");
        }
        for missing in [0x1F, 0x7F, 0x3040, 0x4E01, 0xFFFF] {
            assert_eq!(glyph_of(&font, missing), None, "character {missing:04X}");
        }
    }

    /// with a font for Japanese on the host, kana, kanji and the alphabet
    /// all have glyphs with something drawn, and the block stays as small
    /// as the console's.
    #[test]
    fn the_host_font_draws_japanese() {
        let Some(face) = host_face() else {
            eprintln!("no font for Japanese on this host");
            return;
        };
        let font = write(BASE, face);
        assert!(font.len() as u32 <= BLOCK_LIMIT, "0x{:X} bytes", font.len());
        for character in ['あ', 'ア', '漢', '字', 'A', 'z', 'é', '！', '・'] {
            let glyph = glyph_of(&font, character as u16).unwrap_or_else(|| panic!("no glyph for {character}"));
            let lit = (0..CELL_HEIGHT as u16).flat_map(|y| (0..CELL_WIDTH as u16).map(move |x| (x, y))).filter(|&(x, y)| pixel(&font, glyph, x, y) != 0).count();
            assert!(lit > 4, "{character} has {lit} pixels drawn");
        }
        assert!(glyph_of(&font, ' ' as u16).is_some());
    }
}
