//! bounds-checked little-endian reads over a byte slice.

use crate::FsError;

#[derive(Clone, Copy)]
pub struct Reader<'a> {
    data: &'a [u8],
    base: usize,
}

impl<'a> Reader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Reader { data, base: 0 }
    }

    /// a view of data[offset..], used so structure fields can be written with
    /// the offsets straight out of the 3dbrew tables.
    pub fn at(&self, offset: usize) -> Result<Reader<'a>, FsError> {
        if offset > self.data.len() {
            return Err(FsError::Truncated {
                offset: self.base + offset,
                len: self.data.len(),
            });
        }
        Ok(Reader {
            data: &self.data[offset..],
            base: self.base + offset,
        })
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn bytes(&self, offset: usize, len: usize) -> Result<&'a [u8], FsError> {
        let end = offset.checked_add(len).ok_or(FsError::Truncated {
            offset: self.base + offset,
            len: self.data.len(),
        })?;
        self.data.get(offset..end).ok_or(FsError::Truncated {
            offset: self.base + offset,
            len: self.data.len(),
        })
    }

    pub fn u8(&self, offset: usize) -> Result<u8, FsError> {
        Ok(self.bytes(offset, 1)?[0])
    }

    pub fn u16(&self, offset: usize) -> Result<u16, FsError> {
        let b = self.bytes(offset, 2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    pub fn u32(&self, offset: usize) -> Result<u32, FsError> {
        let b = self.bytes(offset, 4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn u64(&self, offset: usize) -> Result<u64, FsError> {
        let b = self.bytes(offset, 8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    pub fn magic(&self, offset: usize, expected: &[u8; 4]) -> Result<(), FsError> {
        let got = self.bytes(offset, 4)?;
        if got != expected {
            return Err(FsError::BadMagic {
                expected: *expected,
                got: [got[0], got[1], got[2], got[3]],
                offset: self.base + offset,
            });
        }
        Ok(())
    }

    /// reads a fixed-size field and trims trailing NULs, for the ASCII fields
    /// in NCCH headers (product code, title, ...).
    pub fn ascii(&self, offset: usize, len: usize) -> Result<String, FsError> {
        let raw = self.bytes(offset, len)?;
        let end = raw.iter().position(|&b| b == 0).unwrap_or(len);
        Ok(String::from_utf8_lossy(&raw[..end]).into_owned())
    }
}
