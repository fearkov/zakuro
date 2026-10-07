//! texture packs, pictures that take the place of a game's textures, laid
//! out the way Citra and Azahar read them. a pack is a folder of PNGs, in
//! any folders under it, each named tex1_<width>x<height>_<hash>_<format>
//! for the texture it replaces, the hash being CityHash64 of that texture,
//! and a pack.json saying how the hash was taken and which way up the PNGs
//! are. only textures read from memory are replaced, never one the GPU drew,
//! and only on the host's GPU.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use crate::format::morton_interleave;
use crate::texture::TextureFormat;

/// how a pack's hashes were taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hashing {
    /// of the texture's bytes as memory holds them, what pack.json's
    /// use_new_hash asks for.
    Bytes,
    /// of the texture as Citra uploaded it, in rows from the bottom, the
    /// formats past RGBA4 decoded to RGBA8, what packs without a pack.json
    /// or with use_new_hash false were made with.
    Rows,
}

/// a texture pack, its pictures by the hash of the texture each replaces.
pub struct Pack {
    hashing: Hashing,
    materials: HashMap<u64, Arc<Material>>,
}

/// one picture of a pack, read from disk once a texture it replaces is
/// drawn with, and let go of once the GPU has it.
pub struct Material {
    hash: u64,
    path: PathBuf,
    /// whether the rows are stored from the bottom up, so they get turned
    /// over on reading.
    flipped: bool,
    state: Mutex<State>,
}

enum State {
    OnDisk,
    /// being read, and whether it is still wanted once it has been.
    Reading { wanted: bool },
    Read(Arc<Picture>),
    /// unreadable, or more than the GPU takes, which was said once.
    Broken,
}

/// a picture read, RGBA8 rows from the top, at its own size and then at
/// each half size down to a texel, one level after another.
pub struct Picture {
    pub width: u32,
    pub height: u32,
    pub texels: Vec<u8>,
    pub levels: Vec<Level>,
}

/// one size of a picture, where its bytes start in the texels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Level {
    pub offset: usize,
    pub width: u32,
    pub height: u32,
}

/// the largest picture read, as big as any pack's.
const MAX_SIZE: u32 = 8192;

/// the largest size of a picture kept, a bigger one is halved down to it.
/// the GPU draws at 4 times the console's resolution at most, where even
/// the largest textures, 1024 texels across, take 4096 pixels.
const KEPT_SIZE: u32 = 4096;

impl Pack {
    /// the pack in dir, none when it holds no picture.
    pub fn open(dir: &Path) -> Option<Pack> {
        if !dir.is_dir() {
            return None;
        }
        let mut files = Vec::new();
        let mut scanned = HashSet::from_iter(dir.canonicalize().ok());
        scan(dir, 0, &mut scanned, &mut files);
        // the pack.json at the top, or for a pack put in with the folders
        // around it, such as its title id's, the one nearest the top
        let top = dir.join("pack.json");
        let nested = files.iter().filter(|path| path.file_name().is_some_and(|name| name == "pack.json")).min_by_key(|path| path.components().count());
        let config = match nested {
            Some(nested) if !top.is_file() => {
                log::info!("the texture pack's pack.json is {}", nested.display());
                Config::read(nested)
            }
            _ => Config::read(&top),
        };
        let mut materials: HashMap<u64, Arc<Material>> = HashMap::new();
        let (mut duplicates, mut other_formats) = (0, 0);
        for path in files {
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else { continue };
            let Some(hashes) = config.hashes(name, &mut other_formats) else { continue };
            for hash in hashes {
                // the first file in the order Windows lists them wins, as
                // in Citra, and packs do hold the same picture twice
                match materials.entry(hash) {
                    std::collections::hash_map::Entry::Occupied(_) => duplicates += 1,
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(Arc::new(Material {
                            hash,
                            path: path.clone(),
                            flipped: !config.upright,
                            state: Mutex::new(State::OnDisk),
                        }));
                    }
                }
            }
        }
        if other_formats > 0 {
            log::warn!("{other_formats} pictures of the texture pack in {} are DDS or KTX, which Zakuro does not read, only PNG", dir.display());
        }
        if materials.is_empty() {
            return None;
        }
        let hashing = match config.hashing {
            Hashing::Bytes => "by their bytes",
            Hashing::Rows => "as Citra uploaded them",
        };
        log::info!(
            "texture pack in {}, {} pictures, textures hashed {hashing}{}",
            dir.display(),
            materials.len(),
            if duplicates > 0 { format!(", {duplicates} files left out for another of the same hash") } else { String::new() }
        );
        Some(Pack { hashing: config.hashing, materials })
    }

    /// the picture replacing a texture of these bytes, format and size,
    /// with rows where the texture is laid out for the hash when the pack
    /// hashes them as Citra did. the hash pack.json does not ask for is
    /// tried too, as packs carry Citra's pack.json made for the other one,
    /// or pictures of both.
    pub fn find(&self, data: &[u8], format: TextureFormat, width: u32, height: u32, rows: &mut Vec<u8>) -> Option<Arc<Material>> {
        let other = match self.hashing {
            Hashing::Bytes => Hashing::Rows,
            Hashing::Rows => Hashing::Bytes,
        };
        [self.hashing, other].into_iter().find_map(|hashing| self.materials.get(&hash(hashing, data, format, width, height, rows)?).cloned())
    }
}

impl Material {
    /// the hash of the textures it replaces, which tells its pictures apart.
    pub fn hash(&self) -> u64 {
        self.hash
    }

    /// the picture once it has been read, and otherwise none, starting to
    /// read it unless that is under way or failed before.
    pub fn picture(self: &Arc<Self>) -> Option<Arc<Picture>> {
        let mut state = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        match &mut *state {
            State::Read(picture) => Some(picture.clone()),
            State::OnDisk => {
                *state = State::Reading { wanted: true };
                let material = Arc::downgrade(self);
                readers().spawn(move || {
                    // a game closed meanwhile takes its pack with it
                    if let Some(material) = material.upgrade() {
                        material.read();
                    }
                });
                None
            }
            State::Reading { wanted } => {
                *wanted = true;
                None
            }
            State::Broken => None,
        }
    }

    fn read(&self) {
        let read = read_png(&self.path, self.flipped);
        let mut state = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        *state = match (&*state, read) {
            (State::Broken, _) => return,
            (_, Err(error)) => {
                log::warn!("could not read the texture pack's {}, {error}", self.path.display());
                State::Broken
            }
            (State::Reading { wanted: true }, Ok(picture)) => State::Read(Arc::new(picture)),
            // let go of while it was read
            _ => State::OnDisk,
        };
    }

    /// the GPU has the picture, or no longer draws what it replaces, so the
    /// memory it took goes, also once a read under way ends, and it is read
    /// again when it is needed again.
    pub fn release(&self) {
        let mut state = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        match &mut *state {
            State::Read(_) => *state = State::OnDisk,
            State::Reading { wanted } => *wanted = false,
            State::OnDisk | State::Broken => {}
        }
    }

    /// the GPU can't take the picture, the texture stays as it is.
    pub fn refuse(&self) {
        *self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = State::Broken;
    }
}

/// the threads pictures are read on, half the cores and at most four, so
/// that the game goes on meanwhile.
fn readers() -> &'static rayon::ThreadPool {
    static READERS: OnceLock<rayon::ThreadPool> = OnceLock::new();
    READERS.get_or_init(|| {
        let cores = std::thread::available_parallelism().map_or(2, |cores| cores.get());
        rayon::ThreadPoolBuilder::new()
            .num_threads((cores / 2).clamp(1, 4))
            .thread_name(|index| format!("texture pack {index}"))
            .build()
            .expect("threads to read texture packs on")
    })
}

/// what pack.json says, or what holds without one.
#[derive(Debug, PartialEq)]
struct Config {
    hashing: Hashing,
    /// whether the PNGs are stored with their top row first, as dumped,
    /// pack.json's flip_png_files.
    upright: bool,
    /// file names pack.json gives hashes to, besides the hash in a name.
    mapped: HashMap<String, Vec<u64>>,
}

impl Config {
    fn read(path: &Path) -> Config {
        // without a pack.json the hashes are Citra's from before pack.json
        let legacy = Config { hashing: Hashing::Rows, upright: true, mapped: HashMap::new() };
        let text = match std::fs::read(path) {
            Ok(bytes) if !bytes.is_empty() => String::from_utf8_lossy(&bytes).into_owned(),
            _ => return legacy,
        };
        match Config::parse(&text) {
            Ok(config) => config,
            Err(error) => {
                log::warn!("could not read {}, {error}, taking the pack as one without it", path.display());
                legacy
            }
        }
    }

    /// the options missing from a pack.json are those Citra starts with,
    /// where it would stop.
    fn parse(text: &str) -> Result<Config, String> {
        let json: serde_json::Value = serde_json::from_str(&without_comments(text.trim_start_matches('\u{feff}'))).map_err(|error| error.to_string())?;
        let option = |name: &str, default: bool| json["options"][name].as_bool().unwrap_or(default);
        let hashing = if option("use_new_hash", true) { Hashing::Bytes } else { Hashing::Rows };
        let mut mapped: HashMap<String, Vec<u64>> = HashMap::new();
        if let Some(textures) = json["textures"].as_object() {
            for (key, value) in textures {
                let Some(hash) = leading_hex(key.trim_start()) else {
                    log::warn!("the texture pack's pack.json maps {key}, which is not a hash");
                    continue;
                };
                let files = match value {
                    serde_json::Value::String(file) => vec![file.as_str()],
                    serde_json::Value::Array(files) => files.iter().filter_map(serde_json::Value::as_str).collect(),
                    _ => Vec::new(),
                };
                for file in files {
                    // a path in it stands for the file of that name anywhere
                    let name = file.rsplit(['/', '\\']).next().unwrap_or(file);
                    mapped.entry(name.to_owned()).or_default().push(hash);
                }
            }
        }
        Ok(Config { hashing, upright: option("flip_png_files", true), mapped })
    }

    /// the hashes a file of the pack replaces textures of, none when it is
    /// not a color picture Zakuro reads, counting DDS and KTX ones.
    fn hashes(&self, name: &str, other_formats: &mut usize) -> Option<Vec<u64>> {
        // a name, then the kind of map it is, then the format, as Citra
        // splits them, the format in lower case
        let parts: Vec<&str> = name.split('.').collect();
        if parts.len() < 2 || parts.len() > 3 {
            return None;
        }
        // in any case, as Citra before Azahar took .PNG, which packs have
        match parts[parts.len() - 1].to_ascii_lowercase().as_str() {
            "png" => {}
            "dds" | "ktx" => {
                *other_formats += 1;
                return None;
            }
            _ => return None,
        }
        // normal maps, which only Citra's OpenGL renderer draws with
        if parts.len() == 3 && parts[1].eq_ignore_ascii_case("norm") {
            return None;
        }
        let mut hashes = self.mapped.get(name).cloned().unwrap_or_default();
        if let Some(hash) = name_hash(parts[0]) {
            if !hashes.contains(&hash) {
                hashes.push(hash);
            }
        }
        (!hashes.is_empty()).then_some(hashes)
    }
}

/// the hash in a picture's name, tex1_<width>x<height>_<hash>_<format>
/// with anything after, such as Azahar's _mip0.
fn name_hash(name: &str) -> Option<u64> {
    // what follows a decimal number text starts with
    fn decimal(text: &str) -> Option<&str> {
        let end = text.find(|c: char| !c.is_ascii_digit()).unwrap_or(text.len());
        (end > 0).then(|| &text[end..])
    }
    let rest = name.strip_prefix("tex1_")?;
    let rest = decimal(rest)?.strip_prefix('x')?;
    let rest = decimal(rest)?.strip_prefix('_')?;
    let rest = rest.strip_prefix("0x").or_else(|| rest.strip_prefix("0X")).unwrap_or(rest);
    let end = rest.find(|c: char| !c.is_ascii_hexdigit()).unwrap_or(rest.len());
    let hash = u64::from_str_radix(&rest[..end], 16).ok()?;
    decimal(rest[end..].strip_prefix('_')?)?;
    Some(hash)
}

/// the hexadecimal number text starts with.
fn leading_hex(text: &str) -> Option<u64> {
    let text = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")).unwrap_or(text);
    let end = text.find(|c: char| !c.is_ascii_hexdigit()).unwrap_or(text.len());
    u64::from_str_radix(&text[..end], 16).ok()
}

/// JSON with its // and /* */ comments taken out, which Citra allows.
fn without_comments(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    let mut string = false;
    while let Some(c) = chars.next() {
        if string {
            out.push(c);
            if c == '\\' {
                out.extend(chars.next());
            } else if c == '"' {
                string = false;
            }
            continue;
        }
        match (c, chars.peek()) {
            ('"', _) => {
                string = true;
                out.push(c);
            }
            ('/', Some('/')) => {
                while chars.next_if(|&c| c != '\n').is_some() {}
            }
            ('/', Some('*')) => {
                chars.next();
                let mut last = ' ';
                for c in chars.by_ref() {
                    if last == '*' && c == '/' {
                        break;
                    }
                    last = c;
                }
                out.push(' ');
            }
            _ => out.push(c),
        }
    }
    out
}

/// every file under dir, each folder's entries in the order Windows lists
/// them, which decides between pictures of the same hash as in Citra. links
/// are followed, to a folder not scanned already.
fn scan(dir: &Path, depth: u32, scanned: &mut HashSet<PathBuf>, files: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<(String, PathBuf)> =
        entries.filter_map(Result::ok).map(|entry| (entry.file_name().to_string_lossy().to_uppercase(), entry.path())).collect();
    entries.sort();
    for (_, path) in entries {
        if path.is_dir() {
            let new = path.canonicalize().is_ok_and(|real| scanned.insert(real));
            if new && depth < 64 {
                scan(&path, depth + 1, scanned, files);
            }
        } else {
            files.push(path);
        }
    }
}

/// a texture's hash as the pack's were taken, none for a texture no pack
/// can have.
pub fn hash(hashing: Hashing, data: &[u8], format: TextureFormat, width: u32, height: u32, rows: &mut Vec<u8>) -> Option<u64> {
    if matches!(format, TextureFormat::Unknown(_)) {
        return None;
    }
    match hashing {
        Hashing::Bytes => Some(cityhasher::hash(data)),
        Hashing::Rows => {
            citra_rows(data, format, width, height, rows)?;
            Some(cityhasher::hash(rows.as_slice()))
        }
    }
}

/// bytes a texel takes laid out as Citra uploaded it.
fn row_bytes(format: TextureFormat) -> usize {
    match format {
        TextureFormat::Rgb8 => 3,
        TextureFormat::Rgba5551 | TextureFormat::Rgb565 | TextureFormat::Rgba4 => 2,
        _ => 4,
    }
}

/// e4 of Citra's color conversions, a 4 bit value made 8 bits.
fn expand4(value: u8) -> u8 {
    (value & 15) * 17
}

/// lays a texture out as Citra uploaded it, in rows from the bottom, the
/// first five formats in their own bytes and the rest decoded to RGBA8 as
/// Citra decoded them. that differs from texture.rs for HILO8 and for ETC1
/// colors past the ends of their range, so it is a decoder of its own.
fn citra_rows(data: &[u8], format: TextureFormat, width: u32, height: u32, out: &mut Vec<u8>) -> Option<()> {
    let (width, height) = (width as usize, height as usize);
    let size = width * height * format.bits_per_pixel() as usize / 8;
    if width % 8 != 0 || height % 8 != 0 || data.len() < size {
        return None;
    }
    let data = &data[..size];
    out.clear();
    out.resize(width * height * row_bytes(format), 0);
    let (rows, sizes) = (out.as_mut_slice(), (width, height));
    match format {
        TextureFormat::Rgba8 => untile::<4, 4>(data, rows, sizes, |texel| texel),
        TextureFormat::Rgb8 => untile::<3, 3>(data, rows, sizes, |texel| texel),
        TextureFormat::Rgba5551 | TextureFormat::Rgb565 | TextureFormat::Rgba4 => untile::<2, 2>(data, rows, sizes, |texel| texel),
        TextureFormat::La8 => untile::<2, 4>(data, rows, sizes, |[a, i]| [i, i, i, a]),
        TextureFormat::Hilo8 => untile::<2, 4>(data, rows, sizes, |[low, high]| [high, low, 0, 255]),
        TextureFormat::L8 => untile::<1, 4>(data, rows, sizes, |[i]| [i, i, i, 255]),
        TextureFormat::A8 => untile::<1, 4>(data, rows, sizes, |[a]| [0, 0, 0, a]),
        TextureFormat::La4 => untile::<1, 4>(data, rows, sizes, |[texel]| {
            let (i, a) = (expand4(texel >> 4), expand4(texel));
            [i, i, i, a]
        }),
        TextureFormat::L4 => untile_nibbles(data, rows, sizes, |n| [expand4(n), expand4(n), expand4(n), 255]),
        TextureFormat::A4 => untile_nibbles(data, rows, sizes, |n| [0, 0, 0, expand4(n)]),
        TextureFormat::Etc1 => untile_etc1(data, rows, sizes, false),
        TextureFormat::Etc1A4 => untile_etc1(data, rows, sizes, true),
        TextureFormat::Unknown(_) => return None,
    }
    Some(())
}

/// where each texel of an 8x8 tile lies, by its place in memory.
const PLACES: [(usize, usize); 64] = {
    let mut places = [(0, 0); 64];
    let mut y = 0;
    while y < 8 {
        let mut x = 0;
        while x < 8 {
            places[morton_interleave(x, y) as usize] = (x as usize, y as usize);
            x += 1;
        }
        y += 1;
    }
    places
};

/// where a tile of a texture starts in rows from the bottom, its first row
/// of memory the last of its rows there, and the texels across a row.
fn tile_start(index: usize, (width, height): (usize, usize)) -> usize {
    let tiles_across = width / 8;
    let (left, top) = (index % tiles_across * 8, index / tiles_across * 8);
    (height - 1 - top) * width + left
}

/// lays out texels that take IN bytes in memory as OUT bytes each.
fn untile<const IN: usize, const OUT: usize>(data: &[u8], rows: &mut [u8], sizes: (usize, usize), convert: impl Fn([u8; IN]) -> [u8; OUT]) {
    let rows = rows.as_chunks_mut::<OUT>().0;
    for (index, tile) in data.as_chunks::<IN>().0.as_chunks::<64>().0.iter().enumerate() {
        let start = tile_start(index, sizes);
        for (&texel, &(x, y)) in tile.iter().zip(&PLACES) {
            rows[start - y * sizes.0 + x] = convert(texel);
        }
    }
}

/// the same for 4 bit texels, two to a byte, the first in the low bits.
fn untile_nibbles(data: &[u8], rows: &mut [u8], sizes: (usize, usize), convert: impl Fn(u8) -> [u8; 4]) {
    let rows = rows.as_chunks_mut::<4>().0;
    for (index, tile) in data.as_chunks::<32>().0.iter().enumerate() {
        let start = tile_start(index, sizes);
        for (&byte, places) in tile.iter().zip(PLACES.as_chunks::<2>().0) {
            let [(x0, y0), (x1, y1)] = *places;
            rows[start - y0 * sizes.0 + x0] = convert(byte & 15);
            rows[start - y1 * sizes.0 + x1] = convert(byte >> 4);
        }
    }
}

/// the same for ETC1 blocks, four 4x4 ones to a tile, across then down,
/// each after 8 bytes of 4 bit alphas in ETC1A4.
fn untile_etc1(data: &[u8], rows: &mut [u8], sizes: (usize, usize), alpha: bool) {
    let rows = rows.as_chunks_mut::<4>().0;
    let block_bytes = if alpha { 16 } else { 8 };
    let word = |bytes: &[u8]| u64::from_le_bytes(bytes.try_into().expect("8 bytes"));
    for (index, block) in data.chunks_exact(block_bytes).enumerate() {
        let (tile, inside) = (index / 4, index % 4);
        let start = tile_start(tile, sizes) - inside / 2 * 4 * sizes.0 + inside % 2 * 4;
        let (alphas, color) = if alpha { (word(&block[..8]), word(&block[8..])) } else { (u64::MAX, word(block)) };
        let colors = etc1_colors(color);
        let flip = color >> 32 & 1 != 0;
        for x in 0..4 {
            for y in 0..4 {
                // a texel's two bits are 16 apart, numbered down the columns
                let texel = 4 * x + y;
                let second = usize::from(if flip { y >= 2 } else { x >= 2 });
                let pick = ((color >> (16 + texel) & 1) << 1 | color >> texel & 1) as usize;
                let [r, g, b] = colors[second][pick];
                rows[start - y * sizes.0 + x] = [r, g, b, expand4((alphas >> (4 * texel)) as u8)];
            }
        }
    }
}

/// how far ETC1's tables move a texel from its half's color.
const ETC1_MODIFIERS: [[i32; 2]; 8] = [[2, 8], [5, 17], [9, 29], [13, 42], [18, 60], [24, 80], [33, 106], [47, 183]];

/// the colors the texels of an ETC1 block read as a little-endian word can
/// take, for each of its halves, by a texel's two bits, the high one moving
/// the color down. a differential color past 5 bits wraps, as Citra has it.
fn etc1_colors(block: u64) -> [[[u8; 3]; 4]; 2] {
    let differential = block >> 33 & 1 != 0;
    [false, true].map(|second| {
        let base = if differential {
            [59, 51, 43].map(|shift| {
                let value = (block >> shift & 31) as i32;
                let delta = ((block >> (shift - 3) & 7) as i32) << 29 >> 29;
                let value = (if second { value + delta } else { value }) as u8 as u32;
                (value << 3 | value >> 2) as u8
            })
        } else {
            // each half's 4 bit colors, the second half's 4 bits below the first's
            let below = if second { 4 } else { 0 };
            [60, 52, 44].map(|shift| expand4((block >> (shift - below)) as u8))
        };
        let table_shift = if second { 34 } else { 37 };
        let [small, large] = ETC1_MODIFIERS[(block >> table_shift & 7) as usize];
        [small, large, -small, -large].map(|modifier| base.map(|c| (c as i32 + modifier).clamp(0, 255) as u8))
    })
}

/// a PNG as RGBA8 rows from the top, turned over when stored from the
/// bottom, halved down to KEPT_SIZE, with its smaller sizes after it.
fn read_png(path: &Path, flipped: bool) -> Result<Picture, String> {
    let file = std::fs::File::open(path).map_err(|error| error.to_string())?;
    let mut decoder = png::Decoder::new_with_limits(std::io::BufReader::new(file), png::Limits { bytes: 1 << 30 });
    decoder.set_transformations(png::Transformations::normalize_to_color8());
    let mut reader = decoder.read_info().map_err(|error| error.to_string())?;
    // the size first, before memory for the picture is taken
    let (width, height) = (reader.info().width, reader.info().height);
    if width == 0 || height == 0 || width > MAX_SIZE || height > MAX_SIZE {
        return Err(format!("it is {width}x{height}"));
    }
    let size = reader.output_buffer_size().ok_or("it is too big")?;
    let mut buffer = vec![0; size];
    let info = reader.next_frame(&mut buffer).map_err(|error| error.to_string())?;
    let channels = match info.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Indexed => return Err("its palette was not expanded".to_owned()),
    };
    let (w, h) = (width as usize, height as usize);
    let mut texels = if channels == 4 && !flipped && info.line_size == w * 4 {
        buffer.truncate(w * h * 4);
        buffer
    } else {
        let mut texels = Vec::with_capacity(w * h * 4);
        for row in 0..h {
            let source = if flipped { h - 1 - row } else { row };
            let line = &buffer[source * info.line_size..][..w * channels];
            match channels {
                4 => texels.extend_from_slice(line),
                3 => line.as_chunks::<3>().0.iter().for_each(|&[r, g, b]| texels.extend_from_slice(&[r, g, b, 255])),
                2 => line.as_chunks::<2>().0.iter().for_each(|&[i, a]| texels.extend_from_slice(&[i, i, i, a])),
                _ => line.iter().for_each(|&p| texels.extend_from_slice(&[p, p, p, 255])),
            }
        }
        texels
    };
    let mut top = Level { offset: 0, width, height };
    while top.width > KEPT_SIZE || top.height > KEPT_SIZE {
        let smaller = halve(&mut texels, top);
        texels = texels.split_off(smaller.offset);
        top = Level { offset: 0, ..smaller };
    }
    texels.reserve(texels.len() / 3 + 64);
    let mut levels = vec![top];
    while let Some(&above) = levels.last().filter(|level| level.width > 1 || level.height > 1) {
        levels.push(halve(&mut texels, above));
    }
    Ok(Picture { width: top.width, height: top.height, texels, levels })
}

/// adds the level half the size of above to texels, each of its texels the
/// average of 2x2 of above's, the colors weighed by their alpha, so that
/// the color of clear texels, black in most pictures, does not darken the
/// edges of what shows.
fn halve(texels: &mut Vec<u8>, above: Level) -> Level {
    let level = Level { offset: texels.len(), width: (above.width / 2).max(1), height: (above.height / 2).max(1) };
    let (above_width, above_height) = (above.width as usize, above.height as usize);
    for y in 0..level.height as usize {
        // a level one texel high or wide takes the same texel twice
        let rows = [(2 * y).min(above_height - 1), (2 * y + 1).min(above_height - 1)];
        for x in 0..level.width as usize {
            let columns = [(2 * x).min(above_width - 1), (2 * x + 1).min(above_width - 1)];
            let (mut weighed, mut plain, mut alpha) = ([0u32; 3], [0u32; 3], 0u32);
            for row in rows {
                for column in columns {
                    let at = above.offset + (row * above_width + column) * 4;
                    let a = texels[at + 3] as u32;
                    for channel in 0..3 {
                        weighed[channel] += texels[at + channel] as u32 * a;
                        plain[channel] += texels[at + channel] as u32;
                    }
                    alpha += a;
                }
            }
            let [r, g, b] = match alpha {
                0 => plain.map(|total| ((total + 2) / 4) as u8),
                _ => weighed.map(|total| ((total + alpha / 2) / alpha) as u8),
            };
            texels.extend_from_slice(&[r, g, b, ((alpha + 2) / 4) as u8]);
        }
    }
    level
}

/// ZAKURO_TEXTURE_HASHES=<file> writes each texture's hashes there as it is
/// first seen, its size and format, then by its bytes and as Citra uploaded
/// it, to check them against a pack's names. ZAKURO_DUMP_TEXTURES=<folder>
/// saves each one there as Citra names a picture without a pack.json, at
/// ZAKURO_DUMP_SCALE times its size, as a start for a pack.
pub(crate) fn record(data: &[u8], texels: &[[u8; 4]], format: TextureFormat, width: u32, height: u32) {
    /// the file, the folder, the scale, and the textures already seen.
    type Record = Mutex<(Option<std::fs::File>, Option<PathBuf>, u32, std::collections::HashSet<u64>)>;
    static RECORD: OnceLock<Record> = OnceLock::new();
    let record = RECORD.get_or_init(|| {
        let file = std::env::var_os("ZAKURO_TEXTURE_HASHES").and_then(|path| std::fs::OpenOptions::new().create(true).append(true).open(path).ok());
        let dump = std::env::var_os("ZAKURO_DUMP_TEXTURES").map(PathBuf::from);
        let scale = std::env::var("ZAKURO_DUMP_SCALE").ok().and_then(|scale| scale.parse().ok()).unwrap_or(1u32).clamp(1, 8);
        Mutex::new((file, dump, scale, std::collections::HashSet::new()))
    });
    let mut rows = Vec::new();
    let (Some(bytes), Some(laid_out)) = (
        hash(Hashing::Bytes, data, format, width, height, &mut rows),
        hash(Hashing::Rows, data, format, width, height, &mut rows),
    ) else {
        return;
    };
    let mut record = record.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let (file, dump, scale, seen) = &mut *record;
    if !seen.insert(bytes) {
        return;
    }
    let number = match format {
        TextureFormat::Unknown(number) => number,
        _ => (0..14).find(|&number| TextureFormat::from_raw(number) == format).unwrap_or(0),
    };
    if let Some(file) = file {
        let _ = std::io::Write::write_all(file, format!("{width}x{height} {number} {bytes:016X} {laid_out:016X}\n").as_bytes());
    }
    if let Some(dump) = dump.as_ref().filter(|_| width.is_power_of_two() && height.is_power_of_two()) {
        let path = dump.join(format!("tex1_{width}x{height}_{laid_out:016X}_{number}_mip0.png"));
        let (wide, high) = (width * *scale, height * *scale);
        let mut scaled = Vec::with_capacity((wide * high * 4) as usize);
        for y in 0..high {
            for x in 0..wide {
                scaled.extend_from_slice(&texels[((y / *scale) * width + x / *scale) as usize]);
            }
        }
        let written = std::fs::create_dir_all(dump).and_then(|()| std::fs::File::create(&path)).map_err(|error| error.to_string()).and_then(|file| {
            let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), wide, high);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            encoder.write_header().and_then(|mut writer| writer.write_image_data(&scaled)).map_err(|error| error.to_string())
        });
        if let Err(error) = written {
            log::warn!("could not save {}, {error}", path.display());
        }
    }
}

/// whether ZAKURO_TEXTURE_HASHES or ZAKURO_DUMP_TEXTURES asks for textures.
pub(crate) fn recording() -> bool {
    static RECORDING: OnceLock<bool> = OnceLock::new();
    *RECORDING.get_or_init(|| std::env::var_os("ZAKURO_TEXTURE_HASHES").is_some() || std::env::var_os("ZAKURO_DUMP_TEXTURES").is_some())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CityHash64 of (i * 31 + 7) & 0xFF for i below each length, from
    /// Azahar's own cityhash.cpp.
    const CITY: [(usize, u64); 13] = [
        (0, 0x9AE16A3B2F90404F),
        (1, 0x57821EFDEE1B7472),
        (4, 0xDCF15D9B38B260FC),
        (8, 0xD14693440D28F69A),
        (16, 0xD9B28EC31BE83978),
        (17, 0xFFB7CD799A150D69),
        (33, 0x37BD955848865317),
        (64, 0x98F31C3485EFCAFD),
        (65, 0xDFAC987FBCCBC482),
        (128, 0xD44ACA4DD8FF4E3B),
        (255, 0xA11961986B66AB6A),
        (1000, 0x13763B9280A42CA3),
        (4096, 0xA90603EB23A61B29),
    ];

    #[test]
    fn cityhash_is_the_one_citra_has() {
        for (length, expected) in CITY {
            let data: Vec<u8> = (0..length).map(|i| (i * 31 + 7) as u8).collect();
            assert_eq!(cityhasher::hash::<u64>(&data), expected, "length {length}");
        }
    }

    /// width, height, format and the hashes by bytes and as Citra uploaded
    /// the texture, from Azahar's own texture_codec.h over the bytes of
    /// next_byte, taken in this order.
    const TEXTURES: [(u32, u32, u32, u64, u64); 42] = [
        (8, 8, 0, 0x8BB17B66492BC46A, 0xF324964DBD8408A5),
        (8, 8, 1, 0xDBED6FE1B7A74F77, 0x18AF200D2597A12B),
        (8, 8, 2, 0x98C90418F415DA53, 0x23520AA21D30E35A),
        (8, 8, 3, 0xA1BD1D61F630379E, 0x8A468A8119B4DFB2),
        (8, 8, 4, 0x1AA2294C24AABCD9, 0x4607B7E19DBE3736),
        (8, 8, 5, 0xA64B3E382A24B6DF, 0x48BA0CC01C307202),
        (8, 8, 6, 0xB5EEB15FC80301EF, 0x94E1728B088F966C),
        (8, 8, 7, 0x5B6BEF453EDCAC54, 0x5E30125B092B33BE),
        (8, 8, 8, 0x89377ECFE49531C6, 0x38A882BAAD3B9DC1),
        (8, 8, 9, 0xDD5C3642F318AD01, 0x6E3F35968C3F19E9),
        (8, 8, 10, 0x3383F7012DF294E0, 0xD7E2A5039D44C5C6),
        (8, 8, 11, 0x4E831EFF3CCAFDD1, 0xC467D3569A74628F),
        (8, 8, 12, 0xE1E3ED57109FEA36, 0x7AD1CA168583DBFF),
        (8, 8, 13, 0x86769F4B9D60AD30, 0x8106D67CF8D2E829),
        (16, 16, 0, 0x21A5627C2EB1619B, 0x02B794371ECB6D0A),
        (16, 16, 1, 0x3D77FECBD30DB4BE, 0xCB9FB9CC1043EC1D),
        (16, 16, 2, 0x68968A6488E17F90, 0xD30E264A46C63AB7),
        (16, 16, 3, 0xA9295919A8A9C634, 0x1E3513434C05D028),
        (16, 16, 4, 0xCAD7CE4D553C1916, 0x142CEA1B09D61039),
        (16, 16, 5, 0xF8C69C3726BACCA6, 0xF79AE966A19480A6),
        (16, 16, 6, 0xA755B0F722FACB70, 0xC103363A239377FF),
        (16, 16, 7, 0x645B48B285C3774D, 0x3D2D15E381E46631),
        (16, 16, 8, 0xB8B3026757913A70, 0x5E7890B677912C1D),
        (16, 16, 9, 0x2A5D7B0A89F42671, 0xCFB0026A0873F9E9),
        (16, 16, 10, 0x1B01BE1D2F9C2F59, 0x586EF38CA8D614FA),
        (16, 16, 11, 0x4E35339313C321D4, 0xE9AA5DEB736B70ED),
        (16, 16, 12, 0x118639A46AA8CC6E, 0xE3F0186C46FEE91A),
        (16, 16, 13, 0xFBFA22790B2548CB, 0x2BF9E1C0F5BECFC1),
        (32, 8, 0, 0x324991634D90FF20, 0xCD938121C2E1F97A),
        (32, 8, 1, 0xAC15CE902950D25A, 0xE0CBF0F813F78D1E),
        (32, 8, 2, 0x28415277F1039B67, 0x36DE06BA1194DEF8),
        (32, 8, 3, 0xBC77B4AECF34E420, 0x556CDE1A6949D5FE),
        (32, 8, 4, 0x1E58418D8D8D0DB2, 0xE85327E83376678F),
        (32, 8, 5, 0x7A56F5F9291BF4B5, 0xB5731ECA56BD5F85),
        (32, 8, 6, 0x2A3F4DC3AA49F3B3, 0x1A88B43B74984F58),
        (32, 8, 7, 0x9043055FDDC7B036, 0x1B9E40EBE85DE74B),
        (32, 8, 8, 0x4F338D78BD3188E5, 0xF180DFF3A13E69C8),
        (32, 8, 9, 0x34C44BDD48D4B2A3, 0x5B3A1B025942B1DA),
        (32, 8, 10, 0x75822D9D71AD1997, 0xFAB1D1DC77748E21),
        (32, 8, 11, 0xF9C1C8F4923D0082, 0x24EF5BF6B8B60B22),
        (32, 8, 12, 0x409CB27235DF421A, 0xB16D6418B78D96BA),
        (32, 8, 13, 0x8435AD8FCD1517DB, 0xF6F414D0595328D2),
    ];

    /// xorshift64*, the bytes the textures above hold.
    fn next_byte(state: &mut u64) -> u8 {
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 56) as u8
    }

    #[test]
    fn every_format_hashes_as_in_citra() {
        let mut state = 0x243F_6A88_85A3_08D3;
        let mut rows = Vec::new();
        for (width, height, number, bytes, laid_out) in TEXTURES {
            let format = TextureFormat::from_raw(number);
            let size = (width * height * format.bits_per_pixel() / 8) as usize;
            let data: Vec<u8> = (0..size).map(|_| next_byte(&mut state)).collect();
            assert_eq!(hash(Hashing::Bytes, &data, format, width, height, &mut rows), Some(bytes), "{width}x{height} {format:?} by bytes");
            assert_eq!(hash(Hashing::Rows, &data, format, width, height, &mut rows), Some(laid_out), "{width}x{height} {format:?} laid out");
        }
    }

    #[test]
    fn a_name_gives_its_hash_whatever_follows() {
        assert_eq!(name_hash("tex1_256x128_00AB12CD34EF5678_13"), Some(0x00AB_12CD_34EF_5678));
        assert_eq!(name_hash("tex1_256x128_00ab12cd34ef5678_13_mip0"), Some(0x00AB_12CD_34EF_5678));
        assert_eq!(name_hash("tex1_8x8_1F_0anything"), Some(0x1F));
        assert_eq!(name_hash("tex1_8x8_1F"), None);
        assert_eq!(name_hash("tex1_8x8__0"), None);
        assert_eq!(name_hash("tex2_8x8_1F_0"), None);
        assert_eq!(name_hash("tex1_8_1F_0"), None);
        assert_eq!(name_hash("tex1_8x8_10000000000000000_0"), None);
    }

    #[test]
    fn pack_json_takes_comments_and_fills_in_what_it_lacks() {
        let text = "\u{feff}{ // made by hand\n \"options\": { \"use_new_hash\": false, /* \"flip_png_files\": true */ \"skip_mipmap\": true },\n \"textures\": { \"0x00000000000000AB\": \"ui/a.png\", \"cd\": [\"b.png\", \"c\\\\d.png\"], \"no\": \"e.png\", \"note\": \"// not a comment\" } }";
        let config = Config::parse(text).unwrap();
        assert_eq!(config.hashing, Hashing::Rows);
        assert!(config.upright);
        assert_eq!(config.mapped["a.png"], vec![0xAB]);
        assert_eq!(config.mapped["b.png"], vec![0xCD]);
        assert_eq!(config.mapped["d.png"], vec![0xCD]);
        assert!(!config.mapped.contains_key("e.png"));
        assert!(Config::parse("{\"options\": {\"flip_png_files\": false}}").is_ok_and(|config| config.hashing == Hashing::Bytes && !config.upright));
        assert!(Config::parse("{ nope").is_err());
    }

    fn write_png(path: &Path, width: u32, height: u32, texels: &[u8]) {
        let file = std::fs::File::create(path).unwrap();
        let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.write_header().unwrap().write_image_data(texels).unwrap();
    }

    #[test]
    fn a_pack_takes_the_first_of_a_hash_and_leaves_out_what_it_cannot_read() {
        let dir = std::env::temp_dir().join(format!("zakuro-pack-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("UI/PAL")).unwrap();
        let texel = [9, 8, 7, 255];
        for name in [
            "UI/tex1_8x8_00000000000000AA_13_mip0.png",
            "UI/tex1_8x8_00000000000000AA_13.png",
            "UI/PAL/tex1_8x8_00000000000000AA_13.png",
            "UI/tex1_8x8_00000000000000BB_0.norm.png",
            "UI/tex1_8x8_00000000000000CC_0.PNG",
            "UI/tex1_8x8_00000000000000DD_0.dds",
            "mapped.png",
        ] {
            write_png(&dir.join(name), 1, 1, &texel);
        }
        std::fs::write(dir.join("pack.json"), "{\"options\": {\"use_new_hash\": true}, \"textures\": {\"EE\": \"x/mapped.png\"}}").unwrap();
        let pack = Pack::open(&dir).unwrap();
        assert_eq!(pack.hashing, Hashing::Bytes);
        let mut hashes: Vec<u64> = pack.materials.keys().copied().collect();
        hashes.sort();
        assert_eq!(hashes, vec![0xAA, 0xCC, 0xEE]);
        assert!(pack.materials[&0xAA].path.ends_with("UI/PAL/tex1_8x8_00000000000000AA_13.png"));
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(Pack::open(&dir).is_none());
    }

    /// a pack put in with its title id's folder around it, as packs come,
    /// still has its pack.json read, and a link back up is not gone round.
    #[test]
    fn a_pack_in_its_own_folders_finds_its_pack_json() {
        let dir = std::env::temp_dir().join(format!("zakuro-pack-nested-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let inner = dir.join("user/load/textures/0004000000033500");
        std::fs::create_dir_all(inner.join("UI")).unwrap();
        write_png(&inner.join("UI/tex1_8x8_00000000000000AA_13_mip0.png"), 1, 1, &[1, 2, 3, 255]);
        std::fs::write(inner.join("pack.json"), "{\"options\": {\"use_new_hash\": true, \"flip_png_files\": false}}").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(&dir, inner.join("UI/up")).unwrap();
        let pack = Pack::open(&dir).unwrap();
        assert_eq!(pack.hashing, Hashing::Bytes);
        assert!(pack.materials[&0xAA].flipped);
        assert_eq!(pack.materials.len(), 1);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// a pack whose pack.json asks for one hash finds pictures named for
    /// the other.
    #[test]
    fn both_hashes_find_pictures() {
        let data: Vec<u8> = (0..256).map(|i| (i * 7 % 251) as u8).collect();
        let mut rows = Vec::new();
        let format = TextureFormat::Rgba8;
        let bytes = hash(Hashing::Bytes, &data, format, 8, 8, &mut rows).unwrap();
        let laid_out = hash(Hashing::Rows, &data, format, 8, 8, &mut rows).unwrap();
        let material = |hash| Arc::new(Material { hash, path: PathBuf::new(), flipped: false, state: Mutex::new(State::OnDisk) });
        for hashing in [Hashing::Bytes, Hashing::Rows] {
            for named in [bytes, laid_out] {
                let pack = Pack { hashing, materials: HashMap::from([(named, material(named))]) };
                assert_eq!(pack.find(&data, format, 8, 8, &mut rows).map(|found| found.hash), Some(named));
            }
        }
    }

    /// a picture let go of while it is read is dropped once read, and one
    /// wanted again meanwhile is kept.
    #[test]
    fn a_picture_let_go_of_while_read_is_not_kept() {
        let path = std::env::temp_dir().join(format!("zakuro-material-{}.png", std::process::id()));
        write_png(&path, 1, 1, &[1, 2, 3, 255]);
        let material = Material { hash: 1, path: path.clone(), flipped: false, state: Mutex::new(State::Reading { wanted: true }) };
        material.release();
        material.read();
        assert!(matches!(*material.state.lock().unwrap(), State::OnDisk));
        *material.state.lock().unwrap() = State::Reading { wanted: true };
        material.read();
        assert!(matches!(*material.state.lock().unwrap(), State::Read(_)));
        material.refuse();
        material.read();
        assert!(matches!(*material.state.lock().unwrap(), State::Broken));
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn a_picture_reads_upright_with_its_smaller_sizes() {
        let path = std::env::temp_dir().join(format!("zakuro-picture-{}.png", std::process::id()));
        // 3x2, the top row red, the bottom row blue
        let mut texels = Vec::new();
        for color in [[255, 0, 0, 255], [0, 0, 255, 255]] {
            for _ in 0..3 {
                texels.extend_from_slice(&color);
            }
        }
        write_png(&path, 3, 2, &texels);
        let upright = read_png(&path, false).unwrap();
        assert_eq!((upright.width, upright.height), (3, 2));
        assert_eq!(&upright.texels[..4], &[255, 0, 0, 255]);
        assert_eq!(upright.levels, vec![Level { offset: 0, width: 3, height: 2 }, Level { offset: 24, width: 1, height: 1 }]);
        assert_eq!(&upright.texels[24..28], &[128, 0, 128, 255]);
        assert_eq!(upright.texels.len(), 28);
        let flipped = read_png(&path, true).unwrap();
        assert_eq!(&flipped.texels[..4], &[0, 0, 255, 255]);
        std::fs::remove_file(&path).unwrap();
    }

    /// the clear black around what shows does not darken its edges in the
    /// smaller sizes.
    #[test]
    fn clear_texels_do_not_darken_the_smaller_sizes() {
        let mut texels = [[255, 255, 255, 255], [0, 0, 0, 0], [255, 255, 255, 255], [0, 0, 0, 0]].concat();
        let level = halve(&mut texels, Level { offset: 0, width: 2, height: 2 });
        assert_eq!(level, Level { offset: 16, width: 1, height: 1 });
        assert_eq!(&texels[16..], &[255, 255, 255, 128]);
    }

    /// a picture past KEPT_SIZE is halved down to it before its smaller
    /// sizes are made.
    #[test]
    fn a_huge_picture_is_halved_to_the_size_kept() {
        let path = std::env::temp_dir().join(format!("zakuro-huge-{}.png", std::process::id()));
        let width = KEPT_SIZE * 2;
        write_png(&path, width, 2, &vec![200; (width * 2 * 4) as usize]);
        let picture = read_png(&path, false).unwrap();
        assert_eq!((picture.width, picture.height), (KEPT_SIZE, 1));
        assert_eq!(picture.levels[0], Level { offset: 0, width: KEPT_SIZE, height: 1 });
        assert_eq!(picture.levels.last().map(|level| (level.width, level.height)), Some((1, 1)));
        assert_eq!(&picture.texels[..4], &[200; 4]);
        std::fs::remove_file(&path).unwrap();
    }
}
