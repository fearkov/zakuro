//! ROM container parsing, NCSD cartridge images, NCCH partitions, ExeFS, RomFS
//! and the BLZ compression the CTR SDK applies to .code.

pub mod exefs;
pub mod lz77;
pub mod ncch;
pub mod ncsd;
mod reader;
pub mod romfs;
pub mod romfs_build;

use std::path::{Path, PathBuf};

use memmap2::Mmap;

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

    #[error("ExeFS has no .code section")]
    NoCode,

    #[error("malformed BLZ stream: {0}")]
    BadLz77(&'static str),

    #[error("malformed RomFS: {0}")]
    BadRomFs(&'static str),

    #[error("path not found in RomFS: {0}")]
    PathNotFound(String),
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

        let romfs = if ncch.has_romfs() {
            match RomFs::parse(data, ncch_offset + ncch.romfs_offset) {
                Ok(fs) => Some(fs),
                Err(e) => {
                    log::warn!("failed to parse RomFS: {e}");
                    None
                }
            }
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
        })
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
