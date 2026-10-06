//! IPS and BPS, the two patch formats 3DS mods ship changes to a file in.

use crate::FsError;

/// data with an IPS patch applied, PATCH with 24-bit offsets or IPS32 with
/// 32-bit ones. a record past the end grows the data.
pub fn ips(patch: &[u8], data: &[u8]) -> Result<Vec<u8>, FsError> {
    let wide = patch.starts_with(b"IPS32");
    if !wide && !patch.starts_with(b"PATCH") {
        return Err(FsError::BadPatch("not an IPS patch"));
    }
    let (offset_len, end): (usize, &[u8]) = if wide { (4, b"EEOF") } else { (3, b"EOF") };
    let number = |bytes: &[u8]| bytes.iter().fold(0usize, |value, &byte| value << 8 | byte as usize);
    let bytes = |at: usize, len: usize| patch.get(at..at + len).ok_or_else(cut_short);

    let mut out = data.to_vec();
    let mut at = 5;
    loop {
        let record = bytes(at, offset_len)?;
        at += offset_len;
        if record == end {
            // a plain IPS patch can end with the size to cut the data to
            if let Some(size) = patch.get(at..at + 3).filter(|_| !wide) {
                out.truncate(number(size));
            }
            return Ok(out);
        }
        let offset = number(record);
        let size = number(bytes(at, 2)?);
        at += 2;
        let (len, fill) = if size == 0 {
            // a run, its length then the byte it repeats
            let run = bytes(at, 3)?;
            at += 3;
            (number(&run[..2]), Some(run[2]))
        } else {
            (size, None)
        };
        if out.len() < offset + len {
            out.resize(offset + len, 0);
        }
        match fill {
            Some(value) => out[offset..offset + len].fill(value),
            None => {
                out[offset..offset + len].copy_from_slice(bytes(at, len)?);
                at += len;
            }
        }
    }
}

/// the target a BPS patch makes from source. the patch carries checksums
/// of both, so one made for another version of the file is turned down
/// rather than making garbage.
pub fn bps(patch: &[u8], source: &[u8]) -> Result<Vec<u8>, FsError> {
    if !patch.starts_with(b"BPS1") || patch.len() < 4 + 12 {
        return Err(FsError::BadPatch("not a BPS patch"));
    }
    let footer = patch.len() - 12;
    let word = |at: usize| u32::from_le_bytes(patch[at..at + 4].try_into().unwrap());
    if crc32fast::hash(&patch[..footer + 8]) != word(footer + 8) {
        return Err(FsError::BadPatch("the BPS patch is damaged"));
    }
    if crc32fast::hash(source) != word(footer) {
        return Err(FsError::PatchMismatch);
    }

    let mut reader = Reader { patch: &patch[..footer], at: 4 };
    let source_size = reader.number()?;
    let target_size = reader.number()?;
    let metadata = reader.number()?;
    reader.at += metadata;
    if source_size != source.len() {
        return Err(FsError::PatchMismatch);
    }

    let mut target = Vec::with_capacity(target_size);
    let (mut source_at, mut target_at) = (0usize, 0usize);
    let signed = |value: usize| if value & 1 != 0 { -((value >> 1) as isize) } else { (value >> 1) as isize };
    while reader.at < footer {
        let action = reader.number()?;
        let length = (action >> 2) + 1;
        match action & 3 {
            // the source's bytes where the target is now
            0 => {
                let start = target.len();
                target.extend_from_slice(source.get(start..start + length).ok_or_else(cut_short)?);
            }
            // bytes from the patch itself
            1 => {
                target.extend_from_slice(reader.patch.get(reader.at..reader.at + length).ok_or_else(cut_short)?);
                reader.at += length;
            }
            // bytes from elsewhere in the source
            2 => {
                source_at = source_at.checked_add_signed(signed(reader.number()?)).ok_or_else(cut_short)?;
                target.extend_from_slice(source.get(source_at..source_at + length).ok_or_else(cut_short)?);
                source_at += length;
            }
            // bytes from earlier in the target, one at a time, as a copy can
            // overlap what it is making
            _ => {
                target_at = target_at.checked_add_signed(signed(reader.number()?)).ok_or_else(cut_short)?;
                for _ in 0..length {
                    let byte = *target.get(target_at).ok_or_else(cut_short)?;
                    target.push(byte);
                    target_at += 1;
                }
            }
        }
    }
    if target.len() != target_size || crc32fast::hash(&target) != word(footer + 4) {
        return Err(FsError::BadPatch("the BPS patch made something other than its target"));
    }
    Ok(target)
}

fn cut_short() -> FsError {
    FsError::BadPatch("the patch is cut short")
}

/// reads BPS's variable-length numbers.
struct Reader<'a> {
    patch: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn number(&mut self) -> Result<usize, FsError> {
        let (mut value, mut shift) = (0usize, 1usize);
        loop {
            let byte = *self.patch.get(self.at).ok_or_else(cut_short)?;
            self.at += 1;
            let low = (byte as usize & 0x7F).checked_mul(shift).ok_or_else(cut_short)?;
            value = value.checked_add(low).ok_or_else(cut_short)?;
            if byte & 0x80 != 0 {
                return Ok(value);
            }
            shift = shift.checked_mul(0x80).ok_or_else(cut_short)?;
            value = value.checked_add(shift).ok_or_else(cut_short)?;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ips_records_runs_and_growth() {
        let mut patch = b"PATCH".to_vec();
        // two bytes at 1
        patch.extend_from_slice(&[0, 0, 1, 0, 2, b'X', b'Y']);
        // a run of four Zs at 6, past the end
        patch.extend_from_slice(&[0, 0, 6, 0, 0, 0, 4, b'Z']);
        patch.extend_from_slice(b"EOF");
        assert_eq!(ips(&patch, b"abcde").unwrap(), b"aXYde\0ZZZZ");
    }

    #[test]
    fn ips_can_cut_the_data_short() {
        let mut patch = b"PATCH".to_vec();
        patch.extend_from_slice(&[0, 0, 0, 0, 1, b'H']);
        patch.extend_from_slice(b"EOF");
        patch.extend_from_slice(&[0, 0, 3]);
        assert_eq!(ips(&patch, b"hello").unwrap(), b"Hel");
    }

    #[test]
    fn ips32_has_four_byte_offsets() {
        let mut patch = b"IPS32".to_vec();
        patch.extend_from_slice(&[0, 0, 0, 2, 0, 1, b'!']);
        patch.extend_from_slice(b"EEOF");
        assert_eq!(ips(&patch, b"abc").unwrap(), b"ab!");
    }

    /// a BPS number, as the format writes them.
    fn encode(mut value: usize, out: &mut Vec<u8>) {
        loop {
            let low = (value & 0x7F) as u8;
            value >>= 7;
            if value == 0 {
                out.push(low | 0x80);
                return;
            }
            out.push(low);
            value -= 1;
        }
    }

    /// a BPS patch from source to target made of actions, each its kind,
    /// its length and what follows it.
    fn bps_patch(source: &[u8], target: &[u8], actions: &[(usize, usize, &[u8])]) -> Vec<u8> {
        let mut patch = b"BPS1".to_vec();
        encode(source.len(), &mut patch);
        encode(target.len(), &mut patch);
        encode(0, &mut patch);
        for &(kind, length, rest) in actions {
            encode((length - 1) << 2 | kind, &mut patch);
            patch.extend_from_slice(rest);
        }
        patch.extend_from_slice(&crc32fast::hash(source).to_le_bytes());
        patch.extend_from_slice(&crc32fast::hash(target).to_le_bytes());
        let whole = crc32fast::hash(&patch);
        patch.extend_from_slice(&whole.to_le_bytes());
        patch
    }

    #[test]
    fn bps_reads_from_the_source_the_patch_and_the_target() {
        let source = b"hello world";
        let target = b"hello there world!ababab";
        let mut back = Vec::new();
        // the source's "world" is at 6, the target's "ab" at 18
        encode(6 << 1, &mut back);
        let mut target_back = Vec::new();
        encode(18 << 1, &mut target_back);
        let patch = bps_patch(
            source,
            target,
            &[(0, 6, b""), (1, 6, b"there "), (2, 5, &back), (1, 3, b"!ab"), (3, 4, &target_back)],
        );
        assert_eq!(bps(&patch, source).unwrap(), target);
    }

    #[test]
    fn bps_turns_down_another_file() {
        let patch = bps_patch(b"one", b"two", &[(1, 3, b"two")]);
        assert!(matches!(bps(&patch, b"six"), Err(FsError::PatchMismatch)));
        assert_eq!(bps(&patch, b"one").unwrap(), b"two");
    }
}
