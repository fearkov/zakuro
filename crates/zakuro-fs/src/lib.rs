//! ROM container parsing, NCSD cartridge images, CIA archives, NCCH
//! partitions, ExeFS, RomFS and the BLZ compression the CTR SDK applies to
//! .code.

pub mod cia;
pub mod exefs;
pub mod layered;
pub mod lz77;
pub mod ncch;
pub mod ncsd;
pub mod patch;
mod reader;
pub mod romfs;
pub mod romfs_build;
#[cfg(any(test, feature = "testing"))]
pub mod testing;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::Mmap;

pub use cia::{version_name, Cia, Content, TitleKind};
pub use exefs::ExeFs;
pub use ncch::{CodeSetInfo, ExHeader, MemoryType, NcchHeader, SystemMode};
pub use ncsd::Ncsd;
pub use romfs::RomFs;

#[derive(Debug, thiserror::Error)]
pub enum FsError {
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    #[error("bad magic at 0x{offset:X}: expected {expected:?}, got {got:?}")]
    BadMagic {
        expected: [u8; 4],
        got: [u8; 4],
        offset: usize,
    },

    #[error("read past end of data: offset 0x{offset:X} in a {len}-byte region")]
    Truncated { offset: usize, len: usize },

    #[error("unrecognized ROM format")]
    UnknownFormat,

    #[error("cartridge has no executable partition")]
    NoExecutablePartition,

    #[error("this NCCH is encrypted (crypto method 0x{0:02X}); Zakuro needs a decrypted dump")]
    Encrypted(u8),

    #[error("this .cia is encrypted; Zakuro needs a decrypted one")]
    EncryptedCia,

    #[error("this .cia is {0}, not a game")]
    NotAGame(&'static str),

    #[error("malformed CIA: {0}")]
    BadCia(&'static str),

    #[error("the RomFS can't be read, the dump is still partly encrypted or damaged: {0}")]
    UnreadableRomFs(String),

    #[error("ExeFS has no .code section")]
    NoCode,

    #[error("malformed BLZ stream: {0}")]
    BadLz77(&'static str),

    #[error("malformed RomFS: {0}")]
    BadRomFs(&'static str),

    #[error("path not found in RomFS: {0}")]
    PathNotFound(String),

    #[error("malformed patch: {0}")]
    BadPatch(&'static str),

    #[error("the patch is for another version of the file")]
    PatchMismatch,
}

/// a memory-mapped ROM file. Games are up to 4 GiB, so nothing is read eagerly.
pub struct RomImage {
    path: PathBuf,
    map: Mmap,
}

impl std::fmt::Debug for RomImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RomImage").field("path", &self.path).field("len", &self.map.len()).finish()
    }
}

impl RomImage {
    pub fn open(path: impl AsRef<Path>) -> Result<RomImage, FsError> {
        let path = path.as_ref().to_path_buf();
        let file = std::fs::File::open(&path)?;
        // SAFETY: the ROM is a regular file we only ever read.
        let map = unsafe { Mmap::map(&file)? };
        Ok(RomImage { path, map })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn data(&self) -> &[u8] {
        &self.map
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

/// a loaded title, the executable NCCH of a cartridge, with its headers parsed
/// and its filesystems located.
pub struct Title {
    image: Arc<RomImage>,
    /// offset of the CXI inside the image.
    ncch_offset: u64,
    pub ncch: NcchHeader,
    pub exheader: ExHeader,
    pub exefs: ExeFs,
    exefs_offset: u64,
    pub romfs: Option<RomFs>,
    /// the RomFS with mods over it, when there are any.
    pub layered: Option<std::sync::Arc<layered::Layered>>,
    /// the update installed for it, whose code and exheader it runs.
    update: Option<Box<Update>>,
}

/// a game's update, the code, exheader and data that take the place of the
/// game's own, from the update's CIA.
pub struct Update {
    pub title: Title,
    pub title_id: u64,
    pub version: u16,
}

impl Update {
    pub fn load(path: impl AsRef<Path>) -> Result<Update, FsError> {
        let image = Arc::new(RomImage::open(path)?);
        if !Cia::detect(image.data()) {
            return Err(FsError::BadCia("an update comes as a CIA"));
        }
        let cia = Cia::read(image.data())?;
        if cia.kind() != TitleKind::Update {
            return Err(FsError::BadCia("it is not an update"));
        }
        let content = cia.content(0).ok_or(FsError::NoExecutablePartition)?;
        let offset = content.offset.ok_or(FsError::NoExecutablePartition)?;
        if content.encrypted() {
            return Err(FsError::EncryptedCia);
        }
        let title = Title::from_image(image, offset)?;
        Ok(Update { title, title_id: cia.title_id, version: cia.version })
    }
}

/// downloadable content, from its CIA, contents of data a game opens by
/// their index, each an NCCH with a RomFS.
pub struct Dlc {
    image: Arc<RomImage>,
    pub title_id: u64,
    pub version: u16,
    /// every content its TMD lists, those the file holds with a place.
    pub contents: Vec<Content>,
}

impl Dlc {
    pub fn load(path: impl AsRef<Path>) -> Result<Dlc, FsError> {
        let image = Arc::new(RomImage::open(path)?);
        if !Cia::detect(image.data()) {
            return Err(FsError::BadCia("DLC comes as a CIA"));
        }
        let cia = Cia::read(image.data())?;
        if cia.kind() != TitleKind::Dlc {
            return Err(FsError::BadCia("it is not DLC"));
        }
        if cia.contents.iter().any(|content| content.offset.is_some() && content.encrypted()) {
            return Err(FsError::EncryptedCia);
        }
        Ok(Dlc { image, title_id: cia.title_id, version: cia.version, contents: cia.contents })
    }

    pub fn image(&self) -> Arc<RomImage> {
        self.image.clone()
    }

    /// where the RomFS of the content of an index lies in the file, the
    /// window a game reads it through, none when the file leaves the content
    /// out or it has no RomFS it can read.
    pub fn romfs(&self, index: u16) -> Option<(u64, u64)> {
        let offset = self.contents.iter().find(|content| content.index == index)?.offset?;
        let data = self.image.data();
        let ncch = NcchHeader::parse(data.get(offset as usize..)?).ok()?;
        if !ncch.has_romfs() || (!ncch.is_decrypted() && ncch.crypto_method() != 0) {
            return None;
        }
        let romfs = RomFs::parse(data, offset + ncch.romfs_offset).ok()?;
        Some((romfs.base, (offset + ncch.romfs_offset + ncch.romfs_size).saturating_sub(romfs.base)))
    }
}

impl Title {
    pub fn load(path: impl AsRef<Path>) -> Result<Title, FsError> {
        let image = Arc::new(RomImage::open(path)?);
        let data = image.data();

        let ncch_offset = if Ncsd::detect(data) {
            let ncsd = Ncsd::parse(data)?;
            log::info!(
                "NCSD cartridge, media id {:016X}, image size {} MiB",
                ncsd.media_id,
                ncsd.image_size / (1024 * 1024)
            );
            ncsd.executable_partition()?.offset
        } else if Cia::detect(data) {
            let cia = Cia::parse(data)?;
            log::info!("CIA archive, title {:016X}", cia.title_id);
            cia.executable().ok_or(FsError::NoExecutablePartition)?.0
        } else if NcchHeader::detect(data) {
            0
        } else {
            return Err(FsError::UnknownFormat);
        };
        Title::from_image(image, ncch_offset)
    }

    /// the title whose NCCH is at ncch_offset in image.
    fn from_image(image: Arc<RomImage>, ncch_offset: u64) -> Result<Title, FsError> {
        let data = image.data();
        let ncch_data = data.get(ncch_offset as usize..).ok_or(FsError::NoExecutablePartition)?;
        let ncch = NcchHeader::parse(ncch_data)?;

        if !ncch.is_decrypted() && ncch.crypto_method() != 0 {
            return Err(FsError::Encrypted(ncch.crypto_method()));
        }

        let exheader_raw = &ncch_data[NcchHeader::SIZE..NcchHeader::SIZE + ExHeader::HASHED_SIZE];
        let exheader = ExHeader::parse(exheader_raw)?;

        let exefs_offset = ncch_offset + ncch.exefs_offset;
        let exefs = ExeFs::parse(&data[exefs_offset as usize..])?;

        // a game reads its data from the RomFS, without one that reads it
        // stops on its first file
        let romfs = if ncch.has_romfs() {
            let fs = RomFs::parse(data, ncch_offset + ncch.romfs_offset)
                .map_err(|error| FsError::UnreadableRomFs(error.to_string()))?;
            Some(fs)
        } else {
            None
        };

        Ok(Title {
            image,
            ncch_offset,
            ncch,
            exheader,
            exefs,
            exefs_offset,
            romfs,
            layered: None,
            update: None,
        })
    }

    /// runs the update's code and exheader from now on, and has its RomFS
    /// for the game to open besides its own.
    pub fn attach_update(&mut self, update: Update) {
        self.exheader = update.title.exheader.clone();
        self.update = Some(Box::new(update));
    }

    pub fn update(&self) -> Option<&Update> {
        self.update.as_deref()
    }

    pub fn update_mut(&mut self) -> Option<&mut Update> {
        self.update.as_deref_mut()
    }

    /// the image the title is read from, to read it elsewhere.
    pub fn shared_image(&self) -> Arc<RomImage> {
        self.image.clone()
    }

    /// the exheader's bytes that describe the program, the update's when
    /// there is one.
    pub fn exheader_bytes(&self) -> &[u8] {
        if let Some(update) = &self.update {
            return update.title.exheader_bytes();
        }
        let start = self.ncch_offset as usize + NcchHeader::SIZE;
        &self.image.data()[start..start + ExHeader::HASHED_SIZE]
    }

    /// lays the mod in dir over the RomFS. what it changed, when it changed
    /// anything.
    pub fn lay_mods(&mut self, dir: &Path) -> Result<Option<layered::Changes>, FsError> {
        let Some(romfs) = &self.romfs else {
            return Ok(None);
        };
        let layered = layered::Layered::new(romfs, self.image.data(), dir)?;
        let changes = layered.as_ref().map(|layered| layered.changes);
        self.layered = layered.map(std::sync::Arc::new);
        Ok(changes)
    }

    pub fn image(&self) -> &RomImage {
        &self.image
    }

    pub fn ncch_offset(&self) -> u64 {
        self.ncch_offset
    }

    /// raw bytes of an ExeFS entry, from the update when it has it.
    pub fn exefs_file(&self, name: &str) -> Option<&[u8]> {
        if let Some(file) = self.update.as_ref().and_then(|update| update.title.exefs_file(name)) {
            return Some(file);
        }
        let entry = self.exefs.find(name)?;
        let start = (self.exefs_offset + exefs::HEADER_SIZE + entry.offset) as usize;
        let end = start.checked_add(entry.size as usize)?;
        self.image.data().get(start..end)
    }

    /// the executable image, decompressed if the exheader says it is BLZ'd,
    /// the update's when there is one.
    pub fn code(&self) -> Result<Vec<u8>, FsError> {
        if let Some(update) = &self.update {
            return update.title.code();
        }
        let raw = self.exefs_file(".code").ok_or(FsError::NoCode)?;
        if self.exheader.compress_code {
            lz77::decompress(raw)
        } else {
            Ok(raw.to_vec())
        }
    }

    /// reads len bytes at offset from a RomFS file.
    pub fn read_romfs(&self, file: &romfs::FileEntry, offset: u64, len: usize) -> Option<&[u8]> {
        let fs = self.romfs.as_ref()?;
        if offset >= file.data_size {
            return Some(&[]);
        }
        let avail = (file.data_size - offset) as usize;
        let len = len.min(avail);
        let start = (fs.file_data_offset(file) + offset) as usize;
        self.image.data().get(start..start.checked_add(len)?)
    }

    pub fn program_id(&self) -> u64 {
        self.ncch.program_id
    }

    /// human-readable one-liner for the log banner.
    pub fn describe(&self) -> String {
        format!(
            "{} [{}] title={:016X} v{}",
            self.exheader.title, self.ncch.product_code, self.ncch.program_id, self.ncch.version
        )
    }
}

#[cfg(test)]
mod addon_tests {
    use super::*;
    use crate::testing::{cia, ncch, write};

    const GAME: u64 = 0x0004_0000_00AB_CD00;
    const UPDATE: u64 = 0x0004_000E_00AB_CD00;
    const DLC: u64 = 0x0004_008C_00AB_CD00;

    /// a game runs its update's code and exheader, and has the update's
    /// RomFS besides its own.
    #[test]
    fn a_game_runs_its_updates_code() {
        let game = write("update", "game.cxi", &ncch(GAME, Some(&[1; 8]), &[("a.bin", b"game")]));
        let update = write("update", "update.cia", &cia(UPDATE, 0x0410, &[(0, Some(&ncch(UPDATE, Some(&[2; 12]), &[("a.bin", b"update")])))]));
        let mut title = Title::load(&game).unwrap();
        let loaded = Update::load(&update).unwrap();
        assert_eq!((loaded.title_id, loaded.version), (UPDATE, 0x0410));
        title.attach_update(loaded);
        assert_eq!(title.code().unwrap(), vec![2; 12]);
        assert_eq!(title.exheader.text.size, 12);
        assert_eq!(title.program_id(), GAME);
        let update_romfs = title.update().unwrap().title.romfs.as_ref().unwrap();
        assert!(update_romfs.lookup("a.bin").is_ok());
        assert!(title.romfs.as_ref().unwrap().lookup("a.bin").is_ok());
        assert!(Update::load(&game).is_err(), "a game is not an update");
        std::fs::remove_file(game).unwrap();
        std::fs::remove_file(update).unwrap();
    }

    /// DLC lists every content its TMD has, and the RomFS of each it holds
    /// reads by its index.
    #[test]
    fn dlc_contents_open_by_their_index() {
        let first = ncch(DLC, None, &[("quests/1.bin", b"one")]);
        let third = ncch(DLC, None, &[("quests/3.bin", b"three")]);
        let path = write("dlc", "dlc.cia", &cia(DLC, 3, &[(0, Some(&first)), (1, None), (2, Some(&third))]));
        let dlc = Dlc::load(&path).unwrap();
        assert_eq!((dlc.title_id, dlc.version, dlc.contents.len()), (DLC, 3, 3));
        assert!(dlc.romfs(1).is_none(), "the file leaves content 1 out");
        let image = dlc.image();
        for (index, path, data) in [(0, "quests/1.bin", &b"one"[..]), (2, "quests/3.bin", b"three")] {
            let (offset, size) = dlc.romfs(index).unwrap();
            let window = &image.data()[offset as usize..(offset + size) as usize];
            let romfs = RomFs::parse_level3(window, 0).unwrap();
            let file = romfs.lookup(path).unwrap();
            let at = romfs.file_data_offset(&file) as usize;
            assert_eq!(&window[at..at + file.data_size as usize], data);
        }
        std::fs::remove_file(path).unwrap();
    }
}
