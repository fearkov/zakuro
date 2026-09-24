//! stand-ins for the console's shared data archives.

use zakuro_fs::romfs_build::{self, BuildFile};

/// regions the country archive is partitioned by, as their directory names.
const REGIONS: [&str; 6] = ["CN", "EU", "JP", "KR", "TW", "US"];

/// country ids we describe, with the region each belongs to and its name.
const COUNTRIES: [(&str, u8, &str); 6] = [
    ("CN", 160, "China"),
    ("EU", 110, "United Kingdom"),
    ("JP", 1, "Japan"),
    ("KR", 136, "Korea"),
    ("TW", 128, "Taiwan"),
    ("US", 49, "United States"),
];

/// number of localised name slots in every entry, 12 languages, then four
/// repeats of slot 1 that the format requires.
const NAME_SLOTS: usize = 16;
/// bytes per localised name.
const NAME_SIZE: usize = 0x80;

/// builds the region manifest archive (the country and region tables).
pub fn region_manifest() -> Vec<u8> {
    let mut files = Vec::new();

    for region in REGIONS {
        let countries: Vec<(u8, &str)> = COUNTRIES
            .iter()
            .filter(|(r, _, _)| *r == region)
            .map(|(_, id, name)| (*id, *name))
            .collect();

        files.push(BuildFile {
            path: format!("{region}/country_LZ.bin"),
            data: lz11_store(&country_list(&countries)),
        });

        for (id, name) in countries {
            files.push(BuildFile {
                path: format!("{region}/{id}_LZ.bin"),
                data: lz11_store(&division_list(id, name)),
            });
        }
    }

    romfs_build::build(&files)
}

/// builds the profanity filter archive.
pub fn bad_word_list() -> Vec<u8> {
    // version and word count, both zero, no entries follow.
    let data = vec![0u8; 8];
    romfs_build::build(&[BuildFile {
        path: "badwordlist_ver.txt".into(),
        data: data.clone(),
    }])
}

/// the per-region table of countries, two sections (the second patches the
/// first), then a copy of each section-0 entry's sort order, then a bitmap
/// of which countries are eShop-locked.
fn country_list(countries: &[(u8, &str)]) -> Vec<u8> {
    let mut out = Vec::new();

    for _section in 0..2 {
        out.extend_from_slice(&(countries.len() as u32).to_le_bytes());
        for (index, (id, name)) in countries.iter().enumerate() {
            out.extend_from_slice(&[0, 0, 0, *id]);
            out.extend_from_slice(&1u32.to_le_bytes()); // one division
            out.extend_from_slice(&0u32.to_le_bytes());
            write_names(&mut out, name);
            write_sort(&mut out, index as u8);
            out.extend_from_slice(&[0u8; 0x20]);
        }
    }

    for (index, _) in countries.iter().enumerate() {
        write_sort(&mut out, index as u8);
    }
    write_lock_bitmap(&mut out, countries.len());
    out
}

/// the per-country table of divisions (states, prefectures and the like).
fn division_list(country: u8, name: &str) -> Vec<u8> {
    let mut out = Vec::new();
    let divisions = 1usize;

    for section in 0..2 {
        // the first section stores one less than it holds, the reader adds
        // it back. The second stores its real length.
        let stored = if section == 0 {
            divisions as u32 - 1
        } else {
            divisions as u32
        };
        out.extend_from_slice(&stored.to_le_bytes());
        for division in 0..divisions {
            out.extend_from_slice(&[0, 0, division as u8, country]);
            write_names(&mut out, name);
            write_sort(&mut out, division as u8);
            out.extend_from_slice(&0u16.to_le_bytes()); // latitude
            out.extend_from_slice(&0u16.to_le_bytes()); // longitude
        }
    }

    for division in 0..divisions {
        write_sort(&mut out, division as u8);
    }
    write_lock_bitmap(&mut out, divisions);
    out
}

/// writes the 16 name slots.
fn write_names(out: &mut Vec<u8>, name: &str) {
    let mut encoded = Vec::with_capacity(NAME_SIZE);
    for unit in name.encode_utf16() {
        encoded.extend_from_slice(&unit.to_le_bytes());
    }
    encoded.resize(NAME_SIZE, 0);

    for _ in 0..NAME_SLOTS {
        out.extend_from_slice(&encoded);
    }
}

/// writes the 16 sort-order slots, where the last four repeat slot 0.
fn write_sort(out: &mut Vec<u8>, index: u8) {
    out.extend_from_slice(&[index; NAME_SLOTS]);
}

/// writes the trailing bitmap of eShop-locked entries, one bit each, all
/// clear, nothing here is locked.
fn write_lock_bitmap(out: &mut Vec<u8>, entries: usize) {
    for _ in 0..(entries / 32 + 1) {
        out.extend_from_slice(&0u32.to_le_bytes());
    }
}

/// wraps data in an LZ11 container without actually compressing it, every byte
/// is emitted as a literal.
fn lz11_store(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + data.len() / 8 + 8);
    out.push(0x11);
    let size = data.len() as u32;
    out.extend_from_slice(&size.to_le_bytes()[..3]);

    for chunk in data.chunks(8) {
        out.push(0); // eight literals, no back-references
        out.extend_from_slice(chunk);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use zakuro_fs::romfs::RomFs;

    /// decompresses an LZ11 stream, so a test can check what a title would
    /// actually read rather than the container around it.
    fn lz11_decompress(data: &[u8]) -> Vec<u8> {
        assert_eq!(data[0], 0x11, "not an LZ11 stream");
        let size = u32::from_le_bytes([data[1], data[2], data[3], 0]) as usize;
        let mut out = Vec::with_capacity(size);
        let mut cursor = 4;
        while out.len() < size {
            let flags = data[cursor];
            cursor += 1;
            for bit in 0..8 {
                if out.len() >= size {
                    break;
                }
                assert_eq!(flags >> (7 - bit) & 1, 0, "unexpected back-reference");
                out.push(data[cursor]);
                cursor += 1;
            }
        }
        out
    }

    #[test]
    fn the_region_manifest_is_a_readable_archive() {
        let image = region_manifest();
        let romfs = RomFs::parse_level3(&image, 0).expect("the manifest should be a RomFS");

        let entry = romfs
            .lookup("US/country_LZ.bin")
            .expect("the US country table should exist");
        let start = romfs.file_data_offset(&entry) as usize;
        let raw = &image[start..start + entry.data_size as usize];
        let decompressed = lz11_decompress(raw);

        // one country in the US region, and it is the United States.
        assert_eq!(u32::from_le_bytes(decompressed[0..4].try_into().unwrap()), 1);
        assert_eq!(decompressed[4..8], [0, 0, 0, 49]);

        romfs
            .lookup("US/49_LZ.bin")
            .expect("the US division table should exist");
    }

    /// every entry is a fixed size, and a reader walks them by that size, so
    /// a single byte of drift makes the whole table unreadable.
    #[test]
    fn entries_are_the_documented_size() {
        let countries = [(49u8, "United States")];
        let list = country_list(&countries);
        // two sections of (count + one 0x83C entry), then one sort copy of
        // 16 bytes, then one word of lock bits.
        assert_eq!(list.len(), 2 * (4 + 0x83C) + 16 + 4);

        let divisions = division_list(49, "United States");
        assert_eq!(divisions.len(), 2 * (4 + 0x818) + 16 + 4);
    }

    #[test]
    fn the_bad_word_list_is_a_readable_archive() {
        let image = bad_word_list();
        RomFs::parse_level3(&image, 0).expect("the word list should be a RomFS");
    }
}
