//! a game's RomFS with a mod laid over it, from the folders Luma3DS and
//! Citra take mods in. romfs/ holds files that replace the game's or are
//! added to them, romfs_ext/ holds .ips and .bps patches for the game's
//! files and .stub files, each taking away the file it is named after.
//! the game's own file data stays where it is in the ROM, the new tree's
//! metadata goes in front of it and the mod's files after it.

use std::path::{Path, PathBuf};

use crate::romfs::RomFs;
use crate::romfs_build::{self, Dir, File};
use crate::{patch, FsError};

/// what a mod changed in the RomFS.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Changes {
    pub replaced: usize,
    pub added: usize,
    pub patched: usize,
    pub removed: usize,
}

impl Changes {
    pub fn any(&self) -> bool {
        self.replaced + self.added + self.patched + self.removed > 0
    }
}

/// where some of the layered image's bytes come from.
enum Source {
    /// the ROM image, from this offset.
    Image(u64),
    /// a file of the mod's.
    Host(PathBuf),
    /// a patched file, kept whole.
    Memory(Vec<u8>),
}

/// a stretch of the layered image after the metadata.
struct Piece {
    start: u64,
    size: u64,
    source: Source,
}

/// a game's RomFS with a mod over it, read like the level-3 image it
/// stands in for.
pub struct Layered {
    /// the header and the metadata tables, up to where the file data starts.
    meta: Vec<u8>,
    /// the file data, in order.
    pieces: Vec<Piece>,
    size: u64,
    pub changes: Changes,
}

impl std::fmt::Debug for Layered {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Layered").field("size", &self.size).field("changes", &self.changes).finish()
    }
}

impl Layered {
    /// romfs, a RomFS in image, with the mod in dir over it. None when the
    /// mod changes nothing in it.
    pub fn new(romfs: &RomFs, image: &[u8], dir: &Path) -> Result<Option<Layered>, FsError> {
        let mut tree = Tree::read(romfs)?;
        let content = tree.content();
        let mut changes = Changes::default();
        let files = dir.join("romfs");
        if files.is_dir() {
            tree.lay(&files, content, &mut changes)?;
        }
        let patches = dir.join("romfs_ext");
        if patches.is_dir() {
            tree.patch(&patches, content, (romfs, image), &mut changes)?;
        }
        if !changes.any() {
            return Ok(None);
        }
        Ok(Some(tree.layout(romfs, changes)))
    }

    /// how many bytes the layered image has.
    pub fn len(&self) -> u64 {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// reads the layered image from offset into out, image being the ROM's
    /// bytes. how many bytes there were to read.
    pub fn read(&self, image: &[u8], offset: u64, out: &mut [u8]) -> usize {
        if offset >= self.size {
            return 0;
        }
        let count = (out.len() as u64).min(self.size - offset) as usize;
        let out = &mut out[..count];
        // what no piece covers is padding
        out.fill(0);
        let end = offset + count as u64;

        let meta_end = self.meta.len() as u64;
        if offset < meta_end {
            let to = meta_end.min(end);
            out[..(to - offset) as usize].copy_from_slice(&self.meta[offset as usize..to as usize]);
        }

        let first = self.pieces.partition_point(|piece| piece.start + piece.size <= offset);
        for piece in self.pieces[first..].iter().take_while(|piece| piece.start < end) {
            let from = piece.start.max(offset);
            let to = (piece.start + piece.size).min(end);
            if from >= to {
                continue;
            }
            let dest = &mut out[(from - offset) as usize..(to - offset) as usize];
            let within = from - piece.start;
            match &piece.source {
                Source::Image(at) => {
                    let start = (at + within) as usize;
                    if let Some(bytes) = image.get(start..start + dest.len()) {
                        dest.copy_from_slice(bytes);
                    }
                }
                Source::Memory(data) => dest.copy_from_slice(&data[within as usize..within as usize + dest.len()]),
                Source::Host(path) => read_host(path, within, dest),
            }
        }
        count
    }

    /// the files the mod put in, replacing, patching or adding them, whose
    /// paths wanted takes, with their bytes. a path goes from where the
    /// mod's paths start, / separated.
    pub fn modded_files(&self, wanted: impl Fn(&str) -> bool) -> Vec<(String, Vec<u8>)> {
        let mut files = Vec::new();
        let Ok(romfs) = RomFs::parse_level3(&self.meta, 0) else { return files };
        let Ok(root) = romfs.root() else { return files };
        let mut pending = vec![(String::new(), root)];
        while let Some((path, dir)) = pending.pop() {
            for (_, file) in romfs.files(&dir) {
                let path = join(&path, &file.name);
                if !wanted(&path) {
                    continue;
                }
                if let Some(bytes) = self.modded(romfs.file_data_offset(&file), file.data_size) {
                    files.push((path, bytes));
                }
            }
            for (_, child) in romfs.subdirs(&dir) {
                pending.push((join(&path, &child.name), child));
            }
        }
        files
    }

    /// the size bytes at start in the image, when they are a file the mod
    /// put there.
    fn modded(&self, start: u64, size: u64) -> Option<Vec<u8>> {
        let piece = self.pieces.iter().find(|piece| piece.start == start && piece.size == size)?;
        let mut bytes = vec![0; size as usize];
        match &piece.source {
            Source::Image(_) => return None,
            Source::Memory(data) => bytes.copy_from_slice(data),
            Source::Host(path) => read_host(path, 0, &mut bytes),
        }
        Some(bytes)
    }
}

/// a file of the tree being put together.
struct Leaf {
    name: String,
    size: u64,
    /// where its data is from the start of the game's file data, while it
    /// is the game's.
    game: Option<u64>,
    /// what the mod puts in its place.
    modded: Option<Source>,
}

/// the game's tree, with the mod's changes made to it.
struct Tree {
    dirs: Vec<Dir>,
    leaves: Vec<Leaf>,
    /// where the game's file data ends.
    game_data: u64,
}

impl Tree {
    fn read(romfs: &RomFs) -> Result<Tree, FsError> {
        let root = romfs.root()?;
        let mut tree = Tree { dirs: vec![dir(root.name.clone(), 0)], leaves: Vec::new(), game_data: 0 };
        let mut pending = vec![(root, 0)];
        while let Some((entry, index)) = pending.pop() {
            for (_, file) in romfs.files(&entry) {
                tree.game_data = tree.game_data.max(file.data_offset + file.data_size);
                tree.leaves.push(Leaf { name: file.name, size: file.data_size, game: Some(file.data_offset), modded: None });
                tree.dirs[index].files.push(tree.leaves.len() - 1);
            }
            for (_, child) in romfs.subdirs(&entry) {
                tree.dirs.push(dir(child.name.clone(), index));
                let child_index = tree.dirs.len() - 1;
                tree.dirs[index].children.push(child_index);
                pending.push((child, child_index));
            }
        }
        Ok(tree)
    }

    /// the directory a mod's paths start from, the root, or the nameless
    /// one under it some images keep everything in.
    fn content(&self) -> usize {
        match self.dirs[0].children.as_slice() {
            &[only] if self.dirs[0].files.is_empty() && self.dirs[only].name.is_empty() => only,
            _ => 0,
        }
    }

    /// the directory named name in parent. names are matched exactly first,
    /// then ignoring case, as mods made on Windows can have them either way.
    fn find_dir(&self, parent: usize, name: &str) -> Option<usize> {
        let children = &self.dirs[parent].children;
        let by = |same: &dyn Fn(&str) -> bool| children.iter().copied().find(|&child| same(&self.dirs[child].name));
        by(&|other| other == name).or_else(|| by(&|other| other.eq_ignore_ascii_case(name)))
    }

    /// the file named name in parent, as a leaf.
    fn find_file(&self, parent: usize, name: &str) -> Option<usize> {
        let files = &self.dirs[parent].files;
        let by = |same: &dyn Fn(&str) -> bool| files.iter().copied().find(|&leaf| same(&self.leaves[leaf].name));
        by(&|other| other == name).or_else(|| by(&|other| other.eq_ignore_ascii_case(name)))
    }

    /// the mod's files in host, over the directory at.
    fn lay(&mut self, host: &Path, at: usize, changes: &mut Changes) -> Result<(), FsError> {
        for (path, name) in entries(host)? {
            if path.is_dir() {
                let child = match self.find_dir(at, &name) {
                    Some(child) => child,
                    None => {
                        let child = self.dirs.len();
                        self.dirs.push(dir(name, at));
                        self.dirs[at].children.push(child);
                        child
                    }
                };
                self.lay(&path, child, changes)?;
                continue;
            }
            let size = std::fs::metadata(&path)?.len();
            match self.find_file(at, &name) {
                Some(leaf) => {
                    self.leaves[leaf].size = size;
                    self.leaves[leaf].modded = Some(Source::Host(path));
                    changes.replaced += 1;
                }
                None => {
                    let leaf = self.leaves.len();
                    self.leaves.push(Leaf { name, size, game: None, modded: Some(Source::Host(path)) });
                    self.dirs[at].files.push(leaf);
                    changes.added += 1;
                }
            }
        }
        Ok(())
    }

    /// the patches and stubs in host, for the files of the directory at.
    fn patch(&mut self, host: &Path, at: usize, game: (&RomFs, &[u8]), changes: &mut Changes) -> Result<(), FsError> {
        for (path, name) in entries(host)? {
            if path.is_dir() {
                match self.find_dir(at, &name) {
                    Some(child) => self.patch(&path, child, game, changes)?,
                    None => log::warn!("mods: {} is for a directory the game doesn't have", path.display()),
                }
                continue;
            }
            let Some((target, kind)) = name.rsplit_once('.') else {
                log::warn!("mods: {} is not a patch or a stub", path.display());
                continue;
            };
            let kind = kind.to_ascii_lowercase();
            if !["stub", "ips", "bps"].contains(&kind.as_str()) {
                log::warn!("mods: {} is not a patch or a stub", path.display());
                continue;
            }
            let Some(leaf) = self.find_file(at, target) else {
                log::warn!("mods: {} is for a file the game doesn't have", path.display());
                continue;
            };
            if kind == "stub" {
                self.dirs[at].files.retain(|&other| other != leaf);
                changes.removed += 1;
                continue;
            }
            let patch = std::fs::read(&path)?;
            let data = self.data(leaf, game)?;
            let patched = if kind == "ips" { patch::ips(&patch, &data) } else { patch::bps(&patch, &data) };
            match patched {
                Ok(patched) => {
                    self.leaves[leaf].size = patched.len() as u64;
                    self.leaves[leaf].modded = Some(Source::Memory(patched));
                    changes.patched += 1;
                }
                Err(error) => log::warn!("mods: {}: {error}", path.display()),
            }
        }
        Ok(())
    }

    /// a leaf's bytes as they are now, a patch goes over what romfs/ put
    /// there when it put something.
    fn data(&self, leaf: usize, (romfs, image): (&RomFs, &[u8])) -> Result<Vec<u8>, FsError> {
        let leaf = &self.leaves[leaf];
        match (&leaf.modded, leaf.game) {
            (Some(Source::Host(path)), _) => Ok(std::fs::read(path)?),
            (Some(Source::Memory(data)), _) => Ok(data.clone()),
            (_, Some(at)) => {
                let start = (romfs.base + romfs.header.file_data_offset as u64 + at) as usize;
                let bytes = image.get(start..start + leaf.size as usize);
                Ok(bytes.ok_or(FsError::BadRomFs("a file runs past the end of the image"))?.to_vec())
            }
            _ => Ok(Vec::new()),
        }
    }

    /// the image, the game's file data keeping its offsets and the mod's
    /// coming after it.
    fn layout(mut self, romfs: &RomFs, changes: Changes) -> Layered {
        let mut end = self.game_data.next_multiple_of(16);
        let mut files = Vec::new();
        let mut modded = Vec::new();
        for index in 0..self.dirs.len() {
            for leaf in std::mem::take(&mut self.dirs[index].files) {
                let leaf = &mut self.leaves[leaf];
                let data_offset = match leaf.modded.take() {
                    Some(source) => {
                        let at = end;
                        end = (end + leaf.size).next_multiple_of(16);
                        modded.push(Piece { start: at, size: leaf.size, source });
                        at
                    }
                    None => leaf.game.unwrap_or(0),
                };
                self.dirs[index].files.push(files.len());
                files.push(File { name: std::mem::take(&mut leaf.name), parent: index, data_offset, data_size: leaf.size, offset: 0 });
            }
        }

        let meta = romfs_build::metadata(&mut self.dirs, &mut files);
        let base = meta.len() as u64;
        let game = Piece { start: base, size: self.game_data, source: Source::Image(romfs.base + romfs.header.file_data_offset as u64) };
        let mut pieces = vec![game];
        pieces.extend(modded.into_iter().map(|piece| Piece { start: base + piece.start, ..piece }));
        Layered { meta, pieces, size: base + end, changes }
    }
}

fn dir(name: String, parent: usize) -> Dir {
    Dir { name, parent, children: Vec::new(), files: Vec::new(), offset: 0 }
}

/// the path of name in the directory at path. paths skip the nameless
/// directories, the root and the one some images keep everything in.
fn join(path: &str, name: &str) -> String {
    if path.is_empty() || name.is_empty() {
        format!("{path}{name}")
    } else {
        format!("{path}/{name}")
    }
}

/// a host directory's entries and their names, in order of name, so a
/// mod lays out the same way every time.
fn entries(host: &Path) -> Result<Vec<(PathBuf, String)>, FsError> {
    let mut entries = Vec::new();
    for entry in std::fs::read_dir(host)? {
        let path = entry?.path();
        match path.file_name().and_then(|name| name.to_str()) {
            Some(name) => entries.push((path.clone(), name.to_owned())),
            None => log::warn!("mods: {} has a name a RomFS can't hold", path.display()),
        }
    }
    entries.sort_by(|a, b| a.1.cmp(&b.1));
    Ok(entries)
}

/// reads a mod's file from at into out, as much of it as there is.
fn read_host(path: &Path, at: u64, out: &mut [u8]) {
    use std::io::{Read, Seek, SeekFrom};
    let Ok(mut file) = std::fs::File::open(path) else {
        log::warn!("mods: {} can't be read", path.display());
        return;
    };
    if file.seek(SeekFrom::Start(at)).is_err() {
        return;
    }
    let mut filled = 0;
    while filled < out.len() {
        match file.read(&mut out[filled..]) {
            Ok(0) | Err(_) => break,
            Ok(read) => filled += read,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::romfs_build::BuildFile;

    /// a folder for one test's mod, empty.
    fn folder(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("zakuro-mods-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn put(dir: &Path, path: &str, data: &[u8]) {
        let path = dir.join(path);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, data).unwrap();
    }

    /// a game's RomFS, its files at the root as games have them.
    fn game(files: &[(&str, &[u8])]) -> Vec<u8> {
        let files: Vec<BuildFile> = files.iter().map(|(path, data)| BuildFile { path: (*path).into(), data: data.to_vec() }).collect();
        romfs_build::assemble(&files, false)
    }

    /// the whole layered image, read the way a game reads it.
    fn whole(layered: &Layered, image: &[u8]) -> Vec<u8> {
        let mut out = vec![0xAA; layered.len() as usize];
        assert_eq!(layered.read(image, 0, &mut out), out.len());
        out
    }

    fn file(image: &[u8], path: &str) -> Option<Vec<u8>> {
        let romfs = RomFs::parse_level3(image, 0).unwrap();
        let entry = romfs.lookup(path).ok()?;
        let start = romfs.file_data_offset(&entry) as usize;
        Some(image[start..start + entry.data_size as usize].to_vec())
    }

    #[test]
    fn a_mod_replaces_adds_patches_and_removes_files() {
        let image = game(&[
            ("Data/model.bin", b"old model"),
            ("Data/text.bin", b"hello"),
            ("Data/unused.bin", b"gone soon"),
            ("Sound/bgm.bin", b"music"),
            ("top.bin", b"top"),
        ]);
        let romfs = RomFs::parse_level3(&image, 0).unwrap();
        let mods = folder("all");
        put(&mods, "romfs/Data/model.bin", b"a new, bigger model");
        put(&mods, "romfs/Data/extra.bin", b"extra");
        put(&mods, "romfs/New/inside.bin", b"inside");
        let mut ips = b"PATCH".to_vec();
        ips.extend_from_slice(&[0, 0, 0, 0, 1, b'J']);
        ips.extend_from_slice(b"EOF");
        put(&mods, "romfs_ext/Data/text.bin.ips", &ips);
        put(&mods, "romfs_ext/Data/unused.bin.stub", b"");

        let layered = Layered::new(&romfs, &image, &mods).unwrap().expect("the mod changes the RomFS");
        assert_eq!(layered.changes, Changes { replaced: 1, added: 2, patched: 1, removed: 1 });
        let layered_image = whole(&layered, &image);
        assert_eq!(file(&layered_image, "Data/model.bin").unwrap(), b"a new, bigger model");
        assert_eq!(file(&layered_image, "Data/extra.bin").unwrap(), b"extra");
        assert_eq!(file(&layered_image, "New/inside.bin").unwrap(), b"inside");
        assert_eq!(file(&layered_image, "Data/text.bin").unwrap(), b"Jello");
        assert_eq!(file(&layered_image, "Data/unused.bin"), None);
        assert_eq!(file(&layered_image, "Sound/bgm.bin").unwrap(), b"music");
        assert_eq!(file(&layered_image, "top.bin").unwrap(), b"top");
        std::fs::remove_dir_all(mods).unwrap();
    }

    /// reads that start and end anywhere get the same bytes as reading the
    /// whole image, games read in chunks.
    #[test]
    fn reads_in_pieces_match_the_whole() {
        let image = game(&[("a.bin", &[1; 100]), ("b.bin", &[2; 37])]);
        let romfs = RomFs::parse_level3(&image, 0).unwrap();
        let mods = folder("pieces");
        put(&mods, "romfs/b.bin", &[3; 50]);
        put(&mods, "romfs/c.bin", &[4; 20]);
        let layered = Layered::new(&romfs, &image, &mods).unwrap().unwrap();
        let all = whole(&layered, &image);
        for start in 0..all.len() {
            for len in [1, 7, 16, 64] {
                let mut out = vec![0xAA; len];
                let read = layered.read(&image, start as u64, &mut out);
                assert_eq!(&out[..read], &all[start..(start + len).min(all.len())]);
            }
        }
        std::fs::remove_dir_all(mods).unwrap();
    }

    #[test]
    fn names_match_without_case_and_keep_the_games() {
        let image = game(&[("Data/Model.bin", b"old")]);
        let romfs = RomFs::parse_level3(&image, 0).unwrap();
        let mods = folder("case");
        put(&mods, "romfs/data/model.bin", b"new");
        let layered = Layered::new(&romfs, &image, &mods).unwrap().unwrap();
        assert_eq!(layered.changes, Changes { replaced: 1, ..Changes::default() });
        let layered_image = whole(&layered, &image);
        let romfs = RomFs::parse_level3(&layered_image, 0).unwrap();
        let root = romfs.root().unwrap();
        assert_eq!(romfs.subdirs(&root)[0].1.name, "Data");
        assert_eq!(file(&layered_image, "Data/Model.bin").unwrap(), b"new");
        std::fs::remove_dir_all(mods).unwrap();
    }

    /// images that keep everything in a nameless directory under the root
    /// take a mod's paths from there.
    #[test]
    fn a_nameless_content_directory_takes_the_mod() {
        let image = romfs_build::build(&[BuildFile { path: "US/country.bin".into(), data: vec![1] }]);
        let romfs = RomFs::parse_level3(&image, 0).unwrap();
        let mods = folder("nameless");
        put(&mods, "romfs/US/country.bin", &[2, 2]);
        let layered = Layered::new(&romfs, &image, &mods).unwrap().unwrap();
        assert_eq!(layered.changes, Changes { replaced: 1, ..Changes::default() });
        assert_eq!(file(&whole(&layered, &image), "US/country.bin").unwrap(), [2, 2]);
        assert_eq!(layered.modded_files(|_| true), [("US/country.bin".to_owned(), vec![2, 2])]);
        std::fs::remove_dir_all(mods).unwrap();
    }

    /// the files a mod replaced, patched or added read back by their paths,
    /// in the game's spelling, and the game's own and those the mod took
    /// away do not.
    #[test]
    fn the_files_a_mod_put_in_read_back_by_path() {
        let image = game(&[
            ("cro/Battle.cro", b"old battle"),
            ("cro/Field.cro", b"field"),
            ("cro/Gone.cro", b"gone"),
            ("static.crs", b"crs"),
            ("Data/model.bin", b"model"),
        ]);
        let romfs = RomFs::parse_level3(&image, 0).unwrap();
        let mods = folder("modded");
        put(&mods, "romfs/cro/battle.cro", b"new battle");
        put(&mods, "romfs/cro/Extra.cro", b"extra");
        put(&mods, "romfs/Data/model.bin", b"new model");
        let mut ips = b"PATCH".to_vec();
        ips.extend_from_slice(&[0, 0, 0, 0, 1, b'C']);
        ips.extend_from_slice(b"EOF");
        put(&mods, "romfs_ext/static.crs.ips", &ips);
        put(&mods, "romfs_ext/cro/Gone.cro.stub", b"");

        let layered = Layered::new(&romfs, &image, &mods).unwrap().unwrap();
        let mut modules = layered.modded_files(|path| path.ends_with(".cro") || path == "static.crs");
        modules.sort();
        let expected = [("cro/Battle.cro", &b"new battle"[..]), ("cro/Extra.cro", b"extra"), ("static.crs", b"Crs")];
        assert_eq!(modules, expected.map(|(path, bytes)| (path.to_owned(), bytes.to_vec())));
        assert_eq!(layered.modded_files(|_| true).len(), 4);
        std::fs::remove_dir_all(mods).unwrap();
    }

    #[test]
    fn an_empty_mod_changes_nothing() {
        let image = game(&[("a.bin", b"a")]);
        let romfs = RomFs::parse_level3(&image, 0).unwrap();
        let mods = folder("empty");
        std::fs::create_dir_all(mods.join("romfs")).unwrap();
        assert!(Layered::new(&romfs, &image, &mods).unwrap().is_none());
        std::fs::remove_dir_all(mods).unwrap();
    }
}
