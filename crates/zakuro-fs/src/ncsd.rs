//! NCSD (.3ds / .cci) cartridge images.

use crate::reader::Reader;
use crate::FsError;

/// every offset and size in an NCSD/NCCH header is in units of this many bytes.
pub const MEDIA_UNIT: u64 = 0x200;

#[derive(Debug, Clone, Copy)]
pub struct Partition {
    /// byte offset of the partition from the start of the file.
    pub offset: u64,
    pub size: u64,
    pub fs_type: u8,
    pub crypt_type: u8,
}

impl Partition {
    pub fn is_present(&self) -> bool {
        self.size != 0
    }
}

#[derive(Debug, Clone)]
pub struct Ncsd {
    /// total image size in bytes, from the header (may exceed a trimmed file).
    pub image_size: u64,
    pub media_id: u64,
    pub partitions: [Partition; 8],
}

impl Ncsd {
    pub const MAGIC_OFFSET: usize = 0x100;

    /// returns true if data starts with a plausible NCSD header.
    pub fn detect(data: &[u8]) -> bool {
        data.len() > Self::MAGIC_OFFSET + 4 && &data[Self::MAGIC_OFFSET..Self::MAGIC_OFFSET + 4] == b"NCSD"
    }

    pub fn parse(data: &[u8]) -> Result<Ncsd, FsError> {
        let r = Reader::new(data);
        // the first 0x100 bytes are the RSA-2048 signature over the header.
        let h = r.at(Self::MAGIC_OFFSET)?;
        h.magic(0x00, b"NCSD")?;

        let image_size = h.u32(0x04)? as u64 * MEDIA_UNIT;
        let media_id = h.u64(0x08)?;

        let mut partitions = [Partition {
            offset: 0,
            size: 0,
            fs_type: 0,
            crypt_type: 0,
        }; 8];
        for (i, part) in partitions.iter_mut().enumerate() {
            part.offset = h.u32(0x20 + i * 8)? as u64 * MEDIA_UNIT;
            part.size = h.u32(0x24 + i * 8)? as u64 * MEDIA_UNIT;
            // 0x10 bytes of FS types followed by 0x10 bytes of crypt types.
            part.fs_type = h.u8(0x88 + i)?;
            part.crypt_type = h.u8(0x90 + i)?;
        }

        Ok(Ncsd {
            image_size,
            media_id,
            partitions,
        })
    }

    /// the CXI that holds the game's code and RomFS.
    pub fn executable_partition(&self) -> Result<Partition, FsError> {
        let p = self.partitions[0];
        if !p.is_present() {
            return Err(FsError::NoExecutablePartition);
        }
        Ok(p)
    }
}
