//! ExeFS, a flat, ten-entry archive holding .code, banner, icon and
//! occasionally logo.

use crate::reader::Reader;
use crate::FsError;

pub const HEADER_SIZE: u64 = 0x200;
pub const MAX_FILES: usize = 10;

#[derive(Debug, Clone)]
pub struct ExeFsEntry {
    pub name: String,
    /// offset from the start of the ExeFS *data*, i.e. after the 0x200 header.
    pub offset: u64,
    pub size: u64,
    pub hash: [u8; 32],
}

#[derive(Debug, Clone)]
pub struct ExeFs {
    pub entries: Vec<ExeFsEntry>,
}

impl ExeFs {
    pub fn parse(data: &[u8]) -> Result<ExeFs, FsError> {
        let r = Reader::new(data);
        let mut entries = Vec::new();
        for i in 0..MAX_FILES {
            let name = r.ascii(i * 0x10, 8)?;
            let offset = r.u32(i * 0x10 + 0x08)? as u64;
            let size = r.u32(i * 0x10 + 0x0C)? as u64;
            if name.is_empty() && size == 0 {
                continue;
            }
            // hashes are stored in reverse order at the end of the header.
            let mut hash = [0u8; 32];
            hash.copy_from_slice(r.bytes(0x200 - (i + 1) * 0x20, 0x20)?);
            entries.push(ExeFsEntry {
                name,
                offset,
                size,
                hash,
            });
        }
        Ok(ExeFs { entries })
    }

    pub fn find(&self, name: &str) -> Option<&ExeFsEntry> {
        self.entries.iter().find(|e| e.name == name)
    }
}
