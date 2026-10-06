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

use std::path::{Path, PathBuf};

use memmap2::Mmap;

pub use cia::Cia;
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
    image: RomImage,
    /// offset of the CXI inside the image.
    ncch_offset: u64,
    pub ncch: NcchHeader,
    pub exheader: ExHeader,
    pub exefs: ExeFs,
    exefs_offset: u64,
    pub romfs: Option<RomFs>,
    /// the RomFS with mods over it, when there are any.
    pub layered: Option<std::sync::Arc<layered::Layered>>,
}

impl Title {
    pub fn load(path: impl AsRef<Path>) -> Result<Title, FsError> {
        let image = RomImage::open(path)?;
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
            cia.executable.0
        } else if NcchHeader::detect(data) {
            0
        } else {
            return Err(FsError::UnknownFormat);
        };

        let ncch_data = &data[ncch_offset as usize..];
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
        })
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

    /// raw bytes of an ExeFS entry.
    pub fn exefs_file(&self, name: &str) -> Option<&[u8]> {
        let entry = self.exefs.find(name)?;
        let start = (self.exefs_offset + exefs::HEADER_SIZE + entry.offset) as usize;
        let end = start.checked_add(entry.size as usize)?;
        self.image.data().get(start..end)
    }

    /// the executable image, decompressed if the exheader says it is BLZ'd.
    pub fn code(&self) -> Result<Vec<u8>, FsError> {
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
