//! the games in a folder, with the name and icon each carries.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};

/// file endings of games Zakuro opens.
const GAME_FILES: [&str; 4] = ["3ds", "cci", "cxi", "cia"];
/// the icon's side, in pixels.
pub const ICON_SIZE: usize = 48;

pub struct Game {
    pub path: PathBuf,
    pub name: String,
    pub publisher: String,
    pub program_id: u64,
    /// RGBA.
    pub icon: Option<Vec<u8>>,
    /// 3dsrecomp installed code for it.
    pub recompiled: bool,
    /// its mods folder has something in it.
    pub modded: bool,
    /// why it can't be played, when it can't, an encrypted dump say.
    pub problem: Option<String>,
    /// its update in the folder, the newest there.
    pub update: Option<Addon>,
    /// its DLC in the folder.
    pub dlc: Vec<Addon>,
}

/// an update or DLC in the folder, which goes with the game of its id.
#[derive(Debug, Clone)]
pub struct Addon {
    pub path: PathBuf,
    pub title_id: u64,
    pub version: u16,
}

/// the games in the chosen folder, found in the background.
#[derive(Default)]
pub struct Library {
    pub games: Vec<Game>,
    scanning: Option<Receiver<Vec<Game>>>,
    /// set when the games changed and their icons need uploading.
    pub changed: bool,
    /// where the games' mods folders are.
    data_dir: Option<PathBuf>,
}

impl Library {
    /// a library whose games keep their mods under data_dir.
    pub fn new(data_dir: Option<PathBuf>) -> Library {
        Library { data_dir, ..Library::default() }
    }

    /// starts looking through folder again.
    pub fn scan(&mut self, folder: &Path) {
        let (send, receive) = channel();
        let folder = folder.to_owned();
        let data_dir = self.data_dir.clone();
        std::thread::spawn(move || {
            let _ = send.send(scan(&folder, data_dir.as_deref()));
        });
        self.scanning = Some(receive);
    }

    pub fn is_scanning(&self) -> bool {
        self.scanning.is_some()
    }

    /// takes in what a scan found, once it is done.
    pub fn poll(&mut self) {
        let Some(receive) = &self.scanning else { return };
        match receive.try_recv() {
            Ok(games) => {
                self.games = games;
                self.scanning = None;
                self.changed = true;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
            Err(std::sync::mpsc::TryRecvError::Disconnected) => self.scanning = None,
        }
    }

    /// looks again at which games have recompiled code.
    pub fn refresh_recompiled(&mut self) {
        for game in &mut self.games {
            game.recompiled = zakuro_core::recompiled::installed(game.program_id).is_some();
        }
    }
}

fn scan(folder: &Path, data_dir: Option<&Path>) -> Vec<Game> {
    let Ok(entries) = std::fs::read_dir(folder) else { return Vec::new() };
    let paths = entries.filter_map(|entry| entry.ok().map(|entry| entry.path())).filter(|path| {
        path.extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| GAME_FILES.contains(&extension.to_ascii_lowercase().as_str()))
    });
    let (mut games, mut addons) = (Vec::new(), Vec::new());
    for path in paths {
        match addon(&path) {
            Some(Ok(addon)) => addons.push(addon),
            Some(Err(problem)) => games.push(unplayable(&path, problem)),
            None => games.push(read_or_list_unreadable(&path, data_dir)),
        }
    }
    // each update and DLC goes with the game of its id, the newest update
    for addon in addons {
        let update = zakuro_fs::TitleKind::of(addon.title_id) == zakuro_fs::TitleKind::Update;
        let game = games.iter_mut().find(|game| game.problem.is_none() && game.program_id & 0xFFFF_FFFF == addon.title_id & 0xFFFF_FFFF);
        match (game, update) {
            (Some(game), true) => {
                if game.update.as_ref().is_none_or(|known| known.version < addon.version) {
                    game.update = Some(addon);
                }
            }
            (Some(game), false) => game.dlc.push(addon),
            (None, true) => games.push(unplayable(&addon.path, "An update for a game that isn't in this folder".to_owned())),
            (None, false) => games.push(unplayable(&addon.path, "DLC for a game that isn't in this folder".to_owned())),
        }
    }
    // the ones that can't be played go last
    games.sort_by_key(|game| (game.problem.is_some(), game.name.to_lowercase()));
    games
}

/// the update or DLC in the file at path, none when it holds neither, and
/// why it can't be used when it can't.
fn addon(path: &Path) -> Option<Result<Addon, String>> {
    if !path.extension().is_some_and(|extension| extension.eq_ignore_ascii_case("cia")) {
        return None;
    }
    let image = zakuro_fs::RomImage::open(path).ok()?;
    let cia = zakuro_fs::Cia::read(image.data()).ok()?;
    if cia.kind() == zakuro_fs::TitleKind::Game {
        return None;
    }
    if cia.contents.iter().any(|content| content.offset.is_some() && content.encrypted()) {
        return Some(Err("Encrypted, Zakuro needs a decrypted dump".to_owned()));
    }
    Some(Ok(Addon { path: path.to_owned(), title_id: cia.title_id, version: cia.version }))
}

/// a game read, or one that made reading it fail on a bug listed as
/// unreadable, rather than taking the whole library with it.
fn read_or_list_unreadable(path: &Path, data_dir: Option<&Path>) -> Game {
    std::panic::catch_unwind(|| read_game(path, data_dir)).unwrap_or_else(|_| {
        log::error!("reading {} failed", path.display());
        unplayable(path, "Zakuro couldn't read it, please report it".to_owned())
    })
}

/// a file in the library that can't be played, and why.
fn unplayable(path: &Path, problem: String) -> Game {
    Game {
        path: path.to_owned(),
        name: path.file_stem().map(|stem| stem.to_string_lossy().into_owned()).unwrap_or_default(),
        publisher: String::new(),
        program_id: 0,
        icon: None,
        recompiled: false,
        modded: false,
        problem: Some(problem),
        update: None,
        dlc: Vec::new(),
    }
}

fn read_game(path: &Path, data_dir: Option<&Path>) -> Game {
    let file_name = path.file_stem().map(|stem| stem.to_string_lossy().into_owned()).unwrap_or_default();
    let empty = std::fs::metadata(path).is_ok_and(|metadata| metadata.len() == 0);
    let loaded = if empty { Err(None) } else { zakuro_fs::Title::load(path).map_err(Some) };
    let title = match loaded {
        Ok(title) => title,
        Err(error) => {
            log::debug!("can't play {}", path.display());
            let problem = match error {
                Some(error) => problem(&error),
                None => "Empty, its download may not have finished".to_owned(),
            };
            return unplayable(path, problem);
        }
    };
    let program_id = title.program_id();
    let smdh = title.exefs_file("icon").and_then(Smdh::parse);
    Game {
        path: path.to_owned(),
        name: smdh.as_ref().map(|s| s.name.clone()).filter(|name| !name.is_empty()).unwrap_or(file_name),
        publisher: smdh.as_ref().map(|s| s.publisher.clone()).unwrap_or_default(),
        program_id,
        icon: smdh.map(|s| s.icon),
        recompiled: zakuro_core::recompiled::installed(program_id).is_some(),
        modded: data_dir.is_some_and(|data_dir| zakuro_core::mods::present(data_dir, program_id)),
        problem: None,
        update: None,
        dlc: Vec::new(),
    }
}

/// what the library says about a file it can't play.
fn problem(error: &zakuro_fs::FsError) -> String {
    use zakuro_fs::FsError;
    match error {
        FsError::Encrypted(_) | FsError::EncryptedCia => "Encrypted, Zakuro needs a decrypted dump".to_owned(),
        FsError::UnreadableRomFs(_) => "Partly encrypted or damaged, it needs dumping again".to_owned(),
        FsError::NotAGame(what) => {
            let mut what = what.to_string();
            if let Some(first) = what.get_mut(..1) {
                first.make_ascii_uppercase();
            }
            format!("{what} Zakuro couldn't read, it goes in the folder next to its game")
        }
        _ => format!("Can't be read, {error}"),
    }
}

/// the part of a title's icon file the library shows.
struct Smdh {
    name: String,
    publisher: String,
    icon: Vec<u8>,
}

impl Smdh {
    /// where the titles start, 16 of them, one per language.
    const TITLES: usize = 0x08;
    const TITLE_SIZE: usize = 0x200;
    /// English, which comes second.
    const ENGLISH: usize = 1;
    /// the 48 pixel icon, RGB565 in 8 by 8 tiles.
    const LARGE_ICON: usize = 0x24C0;

    fn parse(data: &[u8]) -> Option<Smdh> {
        if data.get(..4)? != b"SMDH" || data.len() < Self::LARGE_ICON + ICON_SIZE * ICON_SIZE * 2 {
            return None;
        }
        // English, or the first language that has a name
        let title = |language: usize| {
            let at = Self::TITLES + language * Self::TITLE_SIZE;
            let text = |offset: usize, size: usize| utf16(&data[at + offset..at + offset + size]);
            (text(0, 0x80), text(0x180, 0x80))
        };
        let (name, publisher) = std::iter::once(Self::ENGLISH)
            .chain(0..16)
            .map(title)
            .find(|(name, _)| !name.is_empty())
            .unwrap_or_default();
        Some(Smdh { name, publisher, icon: decode_icon(&data[Self::LARGE_ICON..]) })
    }
}

/// UTF-16 up to the first zero, with line breaks as spaces.
fn utf16(bytes: &[u8]) -> String {
    let units: Vec<u16> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .take_while(|&unit| unit != 0)
        .collect();
    String::from_utf16_lossy(&units).replace('\n', " ").trim().to_owned()
}

/// the icon as RGBA, from RGB565 in 8 by 8 tiles whose pixels go in Z order.
fn decode_icon(data: &[u8]) -> Vec<u8> {
    let mut rgba = vec![0u8; ICON_SIZE * ICON_SIZE * 4];
    let tiles = ICON_SIZE / 8;
    for y in 0..ICON_SIZE {
        for x in 0..ICON_SIZE {
            let tile = (y / 8) * tiles + x / 8;
            let (tx, ty) = (x % 8, y % 8);
            let within = (tx & 1) | ((ty & 1) << 1) | ((tx & 2) << 1) | ((ty & 2) << 2) | ((tx & 4) << 2) | ((ty & 4) << 3);
            let at = (tile * 64 + within) * 2;
            let value = u16::from_le_bytes([data[at], data[at + 1]]);
            let scale = |bits: u16, max: u16| (bits as u32 * 255 / max as u32) as u8;
            let out = (y * ICON_SIZE + x) * 4;
            rgba[out..out + 4].copy_from_slice(&[
                scale(value >> 11, 31),
                scale((value >> 5) & 63, 63),
                scale(value & 31, 31),
                255,
            ]);
        }
    }
    rgba
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn files_that_cant_be_played_say_why() {
        assert_eq!(problem(&zakuro_fs::FsError::EncryptedCia), "Encrypted, Zakuro needs a decrypted dump");
        assert_eq!(
            problem(&zakuro_fs::FsError::NotAGame("an update")),
            "An update Zakuro couldn't read, it goes in the folder next to its game"
        );
    }

    #[test]
    fn titles_are_read_in_english_first() {
        let mut data = vec![0u8; 0x36C0];
        data[..4].copy_from_slice(b"SMDH");
        let put = |data: &mut Vec<u8>, at: usize, text: &str| {
            for (i, unit) in text.encode_utf16().enumerate() {
                data[at + i * 2..at + i * 2 + 2].copy_from_slice(&unit.to_le_bytes());
            }
        };
        // Japanese first, then English
        put(&mut data, 0x08, "ポケモン");
        put(&mut data, 0x208, "Pokémon\nAlpha Sapphire");
        put(&mut data, 0x208 + 0x180, "Nintendo");
        let smdh = Smdh::parse(&data).unwrap();
        assert_eq!(smdh.name, "Pokémon Alpha Sapphire");
        assert_eq!(smdh.publisher, "Nintendo");
    }

    #[test]
    fn icon_pixels_come_out_of_their_tiles() {
        let mut data = vec![0u8; ICON_SIZE * ICON_SIZE * 2];
        // the second pixel of the first tile is (1, 0), the third (0, 1)
        data[2..4].copy_from_slice(&0xF800u16.to_le_bytes());
        data[4..6].copy_from_slice(&0x001Fu16.to_le_bytes());
        // the first pixel of the second tile is (8, 0)
        data[128..130].copy_from_slice(&0x07E0u16.to_le_bytes());
        let rgba = decode_icon(&data);
        let pixel = |x: usize, y: usize| &rgba[(y * ICON_SIZE + x) * 4..(y * ICON_SIZE + x) * 4 + 4];
        assert_eq!(pixel(1, 0), [255, 0, 0, 255]);
        assert_eq!(pixel(0, 1), [0, 0, 255, 255]);
        assert_eq!(pixel(8, 0), [0, 255, 0, 255]);
    }

    /// an update and DLC in the folder go with their game, the newest
    /// update, and one whose game isn't there is listed as such.
    #[test]
    fn updates_and_dlc_go_with_their_game() {
        use zakuro_fs::testing::{cia, ncch};
        let folder = std::env::temp_dir().join(format!("zakuro-library-addons-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&folder);
        std::fs::create_dir_all(&folder).unwrap();
        let game = 0x0004_0000_00AB_CD00u64;
        let update = |version: u16| cia(game | 0x000E_0000_0000, version, &[(0, Some(&ncch(game | 0x000E_0000_0000, Some(&[2; 8]), &[])))]);
        std::fs::write(folder.join("game.cxi"), ncch(game, Some(&[1; 8]), &[])).unwrap();
        std::fs::write(folder.join("old.cia"), update(0x0400)).unwrap();
        std::fs::write(folder.join("new.cia"), update(0x0410)).unwrap();
        let dlc = game | 0x008C_0000_0000;
        std::fs::write(folder.join("dlc.cia"), cia(dlc, 1, &[(0, Some(&ncch(dlc, None, &[])))])).unwrap();
        let orphan = 0x0004_000E_0000_0001u64;
        std::fs::write(folder.join("orphan.cia"), cia(orphan, 1, &[(0, Some(&ncch(orphan, Some(&[3; 8]), &[])))])).unwrap();

        let games = scan(&folder, None);
        let found = games.iter().find(|entry| entry.program_id == game).unwrap();
        assert_eq!(found.update.as_ref().map(|update| update.version), Some(0x0410));
        assert_eq!(found.dlc.len(), 1);
        let unplayable: Vec<&str> = games.iter().filter_map(|entry| entry.problem.as_deref()).collect();
        assert_eq!(unplayable, ["An update for a game that isn't in this folder"]);
        assert_eq!(games.len(), 2);
        std::fs::remove_dir_all(folder).unwrap();
    }
}
