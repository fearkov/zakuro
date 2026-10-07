//! small NCCH and CIA files for tests, of the shape the console's have,
//! for the crates that read them.

use crate::romfs_build::{self, BuildFile};

fn put(out: &mut [u8], at: usize, bytes: &[u8]) {
    out[at..at + bytes.len()].copy_from_slice(bytes);
}

/// a decrypted NCCH of program_id, its code all text in one page when it
/// has code, and a RomFS of files, the way DLC has only a RomFS.
pub fn ncch(program_id: u64, code: Option<&[u8]>, files: &[(&str, &[u8])]) -> Vec<u8> {
    let mut out = vec![0; 0x800];
    put(&mut out, 0x100, b"NCCH");
    put(&mut out, 0x118, &program_id.to_le_bytes());
    out[0x18F] = 0x04;
    if let Some(code) = code {
        // the exheader after the header, the ExeFS at 0x600 with .code its
        // only file
        put(&mut out, 0x1A0, &3u32.to_le_bytes());
        put(&mut out, 0x210, &[0x0010_0000, 1, code.len() as u32].map(u32::to_le_bytes).concat());
        put(&mut out, 0x600, b".code");
        put(&mut out, 0x60C, &(code.len() as u32).to_le_bytes());
        out.extend_from_slice(code);
    }
    // the RomFS, an IVFC header without hashes and level 3 right after
    let files: Vec<_> = files.iter().map(|(path, data)| BuildFile { path: path.to_string(), data: data.to_vec() }).collect();
    let at = out.len().next_multiple_of(0x200);
    out.resize(at + 0x60, 0);
    put(&mut out, at, b"IVFC");
    put(&mut out, at + 0x04, &0x0001_0000u32.to_le_bytes());
    out[at + 0x4C] = 4;
    out.extend(romfs_build::build(&files));
    let size = (out.len() - at).div_ceil(0x200);
    out.resize(at + size * 0x200, 0);
    put(&mut out, 0x1B0, &((at / 0x200) as u32).to_le_bytes());
    put(&mut out, 0x1B4, &(size as u32).to_le_bytes());
    out
}

/// a CIA of title_id at version, without certificates or a ticket, with a
/// content record for each of contents, by index, and the bytes of those
/// given.
pub fn cia(title_id: u64, version: u16, contents: &[(u16, Option<&[u8]>)]) -> Vec<u8> {
    // the TMD's signature, then its header, with the records 0x9C4 into it
    let header = 0x2040 + 0x140;
    let tmd_size = 0x140 + 0x9C4 + contents.len() * 0x30;
    let mut out = vec![0; 0x2040 + tmd_size];
    put(&mut out, 0, &0x2020u32.to_le_bytes());
    put(&mut out, 0x10, &(tmd_size as u32).to_le_bytes());
    put(&mut out, 0x2040, &0x0001_0004u32.to_be_bytes());
    put(&mut out, header + 0x4C, &title_id.to_be_bytes());
    put(&mut out, header + 0x9C, &version.to_be_bytes());
    put(&mut out, header + 0x9E, &(contents.len() as u16).to_be_bytes());
    for (i, &(index, data)) in contents.iter().enumerate() {
        let record = header + 0x9C4 + i * 0x30;
        put(&mut out, record, &(0x100 + index as u32).to_be_bytes());
        put(&mut out, record + 0x04, &index.to_be_bytes());
        put(&mut out, record + 0x08, &(data.map_or(0x200, <[u8]>::len) as u64).to_be_bytes());
        if data.is_some() {
            out[0x20 + index as usize / 8] |= 0x80 >> (index % 8);
        }
    }
    for data in contents.iter().filter_map(|&(_, data)| data) {
        out.resize(out.len().next_multiple_of(0x40), 0);
        out.extend_from_slice(data);
    }
    out
}

/// writes data to a file in the temporary folder named for test.
pub fn write(test: &str, name: &str, data: &[u8]) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("zakuro-{}-{test}-{name}", std::process::id()));
    std::fs::write(&path, data).unwrap();
    path
}
