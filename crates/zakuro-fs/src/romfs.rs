//! RomFS, the read-only filesystem holding a game's assets.

use crate::reader::Reader;
use crate::FsError;

/// offsets into the RomFS image of the four metadata tables.
#[derive(Debug, Clone, Copy)]
pub struct Level3Header {
    pub dir_hash_offset: u32,
    pub dir_hash_size: u32,
    pub dir_meta_offset: u32,
    pub dir_meta_size: u32,
    pub file_hash_offset: u32,
    pub file_hash_size: u32,
    pub file_meta_offset: u32,
    pub file_meta_size: u32,
    pub file_data_offset: u32,
}

/// marker used by the metadata tables for "no such entry".
pub const INVALID: u32 = 0xFFFF_FFFF;

#[derive(Debug, Clone)]
pub struct DirEntry {
    pub parent: u32,
    pub next_sibling: u32,
    pub first_child: u32,
    pub first_file: u32,
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct FileEntry {
    pub parent: u32,
    pub next_sibling: u32,
    /// offset from the start of the file-data region.
    pub data_offset: u64,
    pub data_size: u64,
    pub name: String,
}

/// a parsed RomFS.
#[derive(Debug, Clone)]
pub struct RomFs {
    /// offset of the level-3 image from the start of the backing file.
    pub base: u64,
    pub header: Level3Header,
    dir_meta: Vec<u8>,
    file_meta: Vec<u8>,
}

impl RomFs {
    /// walks the IVFC header to find where the level-3 image (the actual
    /// directory tree) starts, relative to the IVFC header.
    pub fn level3_offset(data: &[u8]) -> Result<u64, FsError> {
        let r = Reader::new(data);
        r.magic(0x00, b"IVFC")?;
        if r.u32(0x04)? != 0x0001_0000 {
            return Err(FsError::BadRomFs("unexpected IVFC version"));
        }
        let master_hash_size = r.u32(0x08)? as u64;
        // levels start at 0x0C with a 0x18 stride, +0x10 within a level is its
        // block size as a power of two.
        let lvl3_block_log2 = r.u32(0x0C + 2 * 0x18 + 0x10)?;
        if lvl3_block_log2 > 31 {
            return Err(FsError::BadRomFs("absurd level 3 block size"));
        }
        Ok(align_up(0x60 + master_hash_size, 1u64 << lvl3_block_log2))
    }

    /// data is the whole backing image, romfs_offset points at the IVFC
    /// header inside it.
    pub fn parse(data: &[u8], romfs_offset: u64) -> Result<RomFs, FsError> {
        let ivfc = Reader::new(data).at(romfs_offset as usize)?;
        let level3_rel = Self::level3_offset(ivfc.bytes(0, ivfc.len().min(0x1000))?)?;
        Self::parse_level3(data, romfs_offset + level3_rel)
    }

    /// parses an image that is already the level-3 directory structure, with no
    /// IVFC hash tree in front of it.
    pub fn parse_level3(data: &[u8], base: u64) -> Result<RomFs, FsError> {
        let r = Reader::new(data).at(base as usize)?;
        let header_len = r.u32(0x00)?;
        if header_len != 0x28 {
            return Err(FsError::BadRomFs("level 3 header is not 0x28 bytes"));
        }
        let header = Level3Header {
            dir_hash_offset: r.u32(0x04)?,
            dir_hash_size: r.u32(0x08)?,
            dir_meta_offset: r.u32(0x0C)?,
            dir_meta_size: r.u32(0x10)?,
            file_hash_offset: r.u32(0x14)?,
            file_hash_size: r.u32(0x18)?,
            file_meta_offset: r.u32(0x1C)?,
            file_meta_size: r.u32(0x20)?,
            file_data_offset: r.u32(0x24)?,
        };

        let dir_meta = r
            .bytes(header.dir_meta_offset as usize, header.dir_meta_size as usize)?
            .to_vec();
        let file_meta = r
            .bytes(
                header.file_meta_offset as usize,
                header.file_meta_size as usize,
            )?
            .to_vec();

        Ok(RomFs {
            base,
            header,
            dir_meta,
            file_meta,
        })
    }

    pub fn root(&self) -> Result<DirEntry, FsError> {
        self.dir_at(0)
    }

    pub fn dir_at(&self, offset: u32) -> Result<DirEntry, FsError> {
        let r = Reader::new(&self.dir_meta).at(offset as usize)?;
        let name_len = r.u32(0x14)? as usize;
        Ok(DirEntry {
            parent: r.u32(0x00)?,
            next_sibling: r.u32(0x04)?,
            first_child: r.u32(0x08)?,
            first_file: r.u32(0x0C)?,
            name: utf16_name(r.bytes(0x18, name_len)?),
        })
    }

    pub fn file_at(&self, offset: u32) -> Result<FileEntry, FsError> {
        let r = Reader::new(&self.file_meta).at(offset as usize)?;
        let name_len = r.u32(0x1C)? as usize;
        Ok(FileEntry {
            parent: r.u32(0x00)?,
            next_sibling: r.u32(0x04)?,
            data_offset: r.u64(0x08)?,
            data_size: r.u64(0x10)?,
            name: utf16_name(r.bytes(0x20, name_len)?),
        })
    }

    /// children of dir, as (offset, entry) pairs.
    pub fn subdirs(&self, dir: &DirEntry) -> Vec<(u32, DirEntry)> {
        let mut out = Vec::new();
        let mut cursor = dir.first_child;
        while cursor != INVALID {
            match self.dir_at(cursor) {
                Ok(e) => {
                    let next = e.next_sibling;
                    out.push((cursor, e));
                    cursor = next;
                }
                Err(_) => break,
            }
        }
        out
    }

    pub fn files(&self, dir: &DirEntry) -> Vec<(u32, FileEntry)> {
        let mut out = Vec::new();
        let mut cursor = dir.first_file;
        while cursor != INVALID {
            match self.file_at(cursor) {
                Ok(e) => {
                    let next = e.next_sibling;
                    out.push((cursor, e));
                    cursor = next;
                }
                Err(_) => break,
            }
        }
        out
    }

    /// resolves a /-separated path to a file entry.
    pub fn lookup(&self, path: &str) -> Result<FileEntry, FsError> {
        let mut dir = self.root()?;
        // an image's contents normally hang off a nameless directory under
        // the root rather than off the root itself, and a path never spells
        // that level out, so step through it.
        if let Some((_, nameless)) = self
            .subdirs(&dir)
            .into_iter()
            .find(|(_, d)| d.name.is_empty())
        {
            dir = nameless;
        }
        let mut parts: Vec<&str> = path.split('/').filter(|p| !p.is_empty()).collect();
        let Some(file_name) = parts.pop() else {
            return Err(FsError::PathNotFound(path.to_owned()));
        };
        for part in parts {
            let next = self
                .subdirs(&dir)
                .into_iter()
                .find(|(_, d)| d.name.eq_ignore_ascii_case(part))
                .map(|(_, d)| d);
            dir = next.ok_or_else(|| FsError::PathNotFound(path.to_owned()))?;
        }
        self.files(&dir)
            .into_iter()
            .find(|(_, f)| f.name.eq_ignore_ascii_case(file_name))
            .map(|(_, f)| f)
            .ok_or_else(|| FsError::PathNotFound(path.to_owned()))
    }

    /// absolute offset of a file's bytes in the backing image.
    pub fn file_data_offset(&self, file: &FileEntry) -> u64 {
        self.base + self.header.file_data_offset as u64 + file.data_offset
    }

    /// number of entries, for logging and sanity checks.
    pub fn count_entries(&self) -> (usize, usize) {
        let mut dirs = 0usize;
        let mut files = 0usize;
        let mut stack = vec![0u32];
        while let Some(off) = stack.pop() {
            let Ok(dir) = self.dir_at(off) else { continue };
            dirs += 1;
            files += self.files(&dir).len();
            stack.extend(self.subdirs(&dir).into_iter().map(|(o, _)| o));
        }
        (dirs, files)
    }
}

fn utf16_name(raw: &[u8]) -> String {
    let units: Vec<u16> = raw
        .as_chunks::<2>().0.iter()
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    String::from_utf16_lossy(&units)
}

#[inline]
fn align_up(value: u64, align: u64) -> u64 {
    (value + align - 1) & !(align - 1)
}
