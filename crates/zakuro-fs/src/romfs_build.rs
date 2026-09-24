//! builds a RomFS level-3 image.

/// a file to place in the built image.
pub struct BuildFile {
    /// path from the root, e.g. US/country_LZ.bin. Only forward slashes.
    pub path: String,
    pub data: Vec<u8>,
}

const INVALID: u32 = 0xFFFF_FFFF;

/// one directory while the tree is being assembled.
struct Dir {
    name: String,
    parent: usize,
    children: Vec<usize>,
    files: Vec<usize>,
    /// offset in the built metadata table, filled in during layout.
    offset: u32,
}

/// one file while the tree is being assembled.
struct File {
    name: String,
    parent: usize,
    data_offset: u64,
    data_size: u64,
    offset: u32,
}

/// builds a level-3 RomFS image containing files.
pub fn build(files: &[BuildFile]) -> Vec<u8> {
    let mut dirs = vec![
        Dir {
            name: String::new(),
            parent: 0,
            children: vec![1],
            files: Vec::new(),
            offset: 0,
        },
        Dir {
            name: String::new(),
            parent: 0,
            children: Vec::new(),
            files: Vec::new(),
            offset: 0,
        },
    ];
    // everything below is placed under the nameless directory, not the root.
    const CONTENT_ROOT: usize = 1;
    let mut entries: Vec<File> = Vec::new();
    let mut data = Vec::new();

    for file in files {
        let mut parent = CONTENT_ROOT;
        let mut components: Vec<&str> = file.path.split('/').filter(|c| !c.is_empty()).collect();
        let Some(name) = components.pop() else {
            continue;
        };

        for component in components {
            let existing = dirs[parent]
                .children
                .iter()
                .copied()
                .find(|&child| dirs[child].name == component);
            parent = match existing {
                Some(child) => child,
                None => {
                    dirs.push(Dir {
                        name: component.to_owned(),
                        parent,
                        children: Vec::new(),
                        files: Vec::new(),
                        offset: 0,
                    });
                    let child = dirs.len() - 1;
                    dirs[parent].children.push(child);
                    child
                }
            };
        }

        // file data is 16-byte aligned, which is what a real image does and
        // what keeps a reader's 64-bit offsets from straddling a boundary.
        while data.len() % 16 != 0 {
            data.push(0);
        }
        let data_offset = data.len() as u64;
        data.extend_from_slice(&file.data);

        entries.push(File {
            name: name.to_owned(),
            parent,
            data_offset,
            data_size: file.data.len() as u64,
            offset: 0,
        });
        let index = entries.len() - 1;
        dirs[parent].files.push(index);
    }

    // lay the metadata tables out so every entry knows its own offset before
    // anything has to write a sibling or child pointer to it.
    let mut dir_meta_size = 0u32;
    for dir in dirs.iter_mut() {
        dir.offset = dir_meta_size;
        dir_meta_size += dir_entry_size(&dir.name);
    }
    let mut file_meta_size = 0u32;
    for file in entries.iter_mut() {
        file.offset = file_meta_size;
        file_meta_size += file_entry_size(&file.name);
    }

    let dir_buckets = bucket_count(dirs.len());
    let file_buckets = bucket_count(entries.len());
    let mut dir_hash = vec![INVALID; dir_buckets];
    let mut file_hash = vec![INVALID; file_buckets];

    let mut dir_meta = Vec::with_capacity(dir_meta_size as usize);
    for (index, dir) in dirs.iter().enumerate() {
        let parent = dirs[dir.parent].offset;
        let next_sibling = sibling_of(&dirs, dir.parent, index, |d| &d.children)
            .map_or(INVALID, |s| dirs[s].offset);
        let first_child = dir.children.first().map_or(INVALID, |&c| dirs[c].offset);
        let first_file = dir.files.first().map_or(INVALID, |&f| entries[f].offset);

        // the root is not in the hash table, nothing ever looks it up by name.
        let next_hash = if index == 0 {
            INVALID
        } else {
            let bucket = hash(parent, &dir.name) as usize % dir_buckets;
            let previous = dir_hash[bucket];
            dir_hash[bucket] = dir.offset;
            previous
        };

        dir_meta.extend_from_slice(&parent.to_le_bytes());
        dir_meta.extend_from_slice(&next_sibling.to_le_bytes());
        dir_meta.extend_from_slice(&first_child.to_le_bytes());
        dir_meta.extend_from_slice(&first_file.to_le_bytes());
        dir_meta.extend_from_slice(&next_hash.to_le_bytes());
        write_name(&mut dir_meta, &dir.name);
    }

    let mut file_meta = Vec::with_capacity(file_meta_size as usize);
    for (index, file) in entries.iter().enumerate() {
        let parent = dirs[file.parent].offset;
        let next_sibling = sibling_of(&dirs, file.parent, index, |d| &d.files)
            .map_or(INVALID, |s| entries[s].offset);

        let bucket = hash(parent, &file.name) as usize % file_buckets;
        let next_hash = file_hash[bucket];
        file_hash[bucket] = file.offset;

        file_meta.extend_from_slice(&parent.to_le_bytes());
        file_meta.extend_from_slice(&next_sibling.to_le_bytes());
        file_meta.extend_from_slice(&file.data_offset.to_le_bytes());
        file_meta.extend_from_slice(&file.data_size.to_le_bytes());
        file_meta.extend_from_slice(&next_hash.to_le_bytes());
        write_name(&mut file_meta, &file.name);
    }

    // header, its own size, then (offset, size) for each of the four tables,
    // then where the file data starts. Every table is 4-byte aligned.
    const HEADER_SIZE: u32 = 0x28;
    let dir_hash_offset = HEADER_SIZE;
    let dir_hash_size = (dir_hash.len() * 4) as u32;
    let dir_meta_offset = dir_hash_offset + dir_hash_size;
    let file_hash_offset = dir_meta_offset + dir_meta_size;
    let file_hash_size = (file_hash.len() * 4) as u32;
    let file_meta_offset = file_hash_offset + file_hash_size;
    let file_data_offset = align_up(file_meta_offset + file_meta_size, 16);

    let mut out = Vec::with_capacity(file_data_offset as usize + data.len());
    for value in [
        HEADER_SIZE,
        dir_hash_offset,
        dir_hash_size,
        dir_meta_offset,
        dir_meta_size,
        file_hash_offset,
        file_hash_size,
        file_meta_offset,
        file_meta_size,
        file_data_offset,
    ] {
        out.extend_from_slice(&value.to_le_bytes());
    }
    for value in &dir_hash {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out.extend_from_slice(&dir_meta);
    for value in &file_hash {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out.extend_from_slice(&file_meta);
    out.resize(file_data_offset as usize, 0);
    out.extend_from_slice(&data);
    out
}

/// the entry after index in its parent's list, if any.
fn sibling_of(
    dirs: &[Dir],
    parent: usize,
    index: usize,
    list: impl Fn(&Dir) -> &Vec<usize>,
) -> Option<usize> {
    let siblings = list(&dirs[parent]);
    let position = siblings.iter().position(|&s| s == index)?;
    siblings.get(position + 1).copied()
}

fn dir_entry_size(name: &str) -> u32 {
    0x18 + align_up(name_bytes(name).len() as u32, 4)
}

fn file_entry_size(name: &str) -> u32 {
    0x20 + align_up(name_bytes(name).len() as u32, 4)
}

fn name_bytes(name: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(name.len() * 2);
    for unit in name.encode_utf16() {
        out.extend_from_slice(&unit.to_le_bytes());
    }
    out
}

fn write_name(out: &mut Vec<u8>, name: &str) {
    let bytes = name_bytes(name);
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&bytes);
    while !out.len().is_multiple_of(4) {
        out.push(0);
    }
}

/// the hash a name lands on, as the console's own lookup computes it.
fn hash(parent: u32, name: &str) -> u32 {
    let mut hash = parent ^ 123_456_789;
    for unit in name.encode_utf16() {
        hash = hash.rotate_right(5);
        hash ^= unit as u32;
    }
    hash
}

/// bucket count for a hash table holding entries, using the same shape the
/// console's own images do, odd, and a little larger than the entry count.
fn bucket_count(entries: usize) -> usize {
    // diagnostic, a large prime table makes the bucket a title reads reveal
    // the hash of the name it is looking for, which is otherwise invisible.
    if let Ok(forced) = std::env::var("ZAKURO_ROMFS_BUCKETS") {
        if let Ok(forced) = forced.parse::<usize>() {
            return forced;
        }
    }
    let mut count = entries.max(1) as u32;
    if count < 3 {
        count = 3;
    } else if count < 19 {
        count |= 1;
    } else {
        while count.is_multiple_of(2)
            || count.is_multiple_of(3)
            || count.is_multiple_of(5)
            || count.is_multiple_of(7)
            || count.is_multiple_of(11)
            || count.is_multiple_of(13)
            || count.is_multiple_of(17)
        {
            count += 1;
        }
    }
    count as usize
}

fn align_up(value: u32, align: u32) -> u32 {
    value.div_ceil(align) * align
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::romfs::RomFs;

    /// a built image has to read back through the parser that reads real
    /// cartridges, or it is not a RomFS, just bytes shaped like one.
    #[test]
    fn a_built_image_reads_back_through_the_parser() {
        let image = build(&[
            BuildFile {
                path: "US/country_LZ.bin".into(),
                data: vec![1, 2, 3, 4, 5],
            },
            BuildFile {
                path: "US/other.bin".into(),
                data: vec![9; 40],
            },
            BuildFile {
                path: "root.bin".into(),
                data: vec![7, 7],
            },
        ]);

        let romfs = RomFs::parse_level3(&image, 0).expect("the built image should parse");
        let found = romfs.lookup("US/country_LZ.bin").expect("file should exist");
        assert_eq!(found.data_size, 5);
        let start = romfs.file_data_offset(&found) as usize;
        assert_eq!(&image[start..start + 5], &[1, 2, 3, 4, 5]);

        let root = romfs.lookup("root.bin").expect("root file should exist");
        assert_eq!(root.data_size, 2);

        let other = romfs.lookup("US/other.bin").expect("sibling should exist");
        assert_eq!(other.data_size, 40);
        let start = romfs.file_data_offset(&other) as usize;
        assert_eq!(&image[start..start + 40], &[9; 40]);
    }

    /// looks a name up the way the console does, hash it, index the bucket
    /// table, then walk the collision chain.
    fn hash_lookup(image: &[u8], parent_offset: u32, name: &str, files: bool) -> Option<u32> {
        let word = |at: usize| u32::from_le_bytes(image[at..at + 4].try_into().unwrap());
        let (table_offset, table_size, meta_offset) = if files {
            (word(0x14) as usize, word(0x18) as usize, word(0x1C) as usize)
        } else {
            (word(0x04) as usize, word(0x08) as usize, word(0x0C) as usize)
        };

        let buckets = table_size / 4;
        let bucket = super::hash(parent_offset, name) as usize % buckets;
        let mut entry = word(table_offset + bucket * 4);

        while entry != INVALID {
            let base = meta_offset + entry as usize;
            let (next_hash_at, name_len_at) = if files { (0x18, 0x1C) } else { (0x10, 0x14) };
            let entry_parent = word(base);
            let name_len = word(base + name_len_at) as usize;
            let name_at = base + name_len_at + 4;
            let units: Vec<u16> = image[name_at..name_at + name_len]
                .as_chunks::<2>().0.iter()
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            if entry_parent == parent_offset && String::from_utf16_lossy(&units) == name {
                return Some(entry);
            }
            entry = word(base + next_hash_at);
        }
        None
    }

    #[test]
    fn names_are_reachable_through_the_hash_tables() {
        let image = build(&[
            BuildFile {
                path: "US/country_LZ.bin".into(),
                data: vec![1, 2, 3],
            },
            BuildFile {
                path: "US/49_LZ.bin".into(),
                data: vec![4, 5, 6],
            },
            BuildFile {
                path: "JP/country_LZ.bin".into(),
                data: vec![7],
            },
        ]);

        // a title starts by resolving the nameless directory in the root,
        // exactly as this does, and everything else hangs off that.
        let content_root =
            hash_lookup(&image, 0, "", false).expect("the nameless root should hash-resolve");

        let us = hash_lookup(&image, content_root, "US", false).expect("US should hash-resolve");
        let jp = hash_lookup(&image, content_root, "JP", false).expect("JP should hash-resolve");
        assert_ne!(us, jp);

        hash_lookup(&image, us, "country_LZ.bin", true)
            .expect("the US country table should hash-resolve");
        hash_lookup(&image, us, "49_LZ.bin", true)
            .expect("the US division table should hash-resolve");
        hash_lookup(&image, jp, "country_LZ.bin", true)
            .expect("the JP country table should hash-resolve");

        // a name that is not there must not resolve to a neighbour that
        // happens to share a bucket.
        assert!(hash_lookup(&image, us, "missing.bin", true).is_none());
    }

    #[test]
    fn directories_nest() {
        let image = build(&[BuildFile {
            path: "a/b/c/deep.bin".into(),
            data: vec![42],
        }]);
        let romfs = RomFs::parse_level3(&image, 0).expect("parse");
        let found = romfs.lookup("a/b/c/deep.bin").expect("nested file");
        assert_eq!(found.data_size, 1);
    }
}
