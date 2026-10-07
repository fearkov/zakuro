//! mods, kept in the data directory's mods folder under the title's id, laid
//! out the way Luma3DS and Citra take them.

use std::path::{Path, PathBuf};

use zakuro_common::memory_map::PAGE_SIZE;
use zakuro_fs::{lz77, patch, FsError, Title};

/// the folder a title's mods go in.
pub fn dir(data_dir: &Path, program_id: u64) -> PathBuf {
    let mods = data_dir.join("mods");
    let upper = mods.join(format!("{program_id:016X}"));
    let lower = mods.join(format!("{program_id:016x}"));
    if !upper.exists() && lower.exists() { lower } else { upper }
}

/// the files that change a title's code, each in the mod's folder or in its
/// exefs folder.
const CODE: [&str; 3] = ["code.bin", "code.ips", "code.bps"];

/// whether the title's mods folder holds a mod.
pub fn present(data_dir: &Path, program_id: u64) -> bool {
    let dir = dir(data_dir, program_id);
    let files = ["romfs", "romfs_ext"].iter().any(|part| std::fs::read_dir(dir.join(part)).is_ok_and(|mut entries| entries.next().is_some()));
    files || CODE.iter().any(|name| find(&dir, name).is_some())
}

fn find(dir: &Path, name: &str) -> Option<PathBuf> {
    [dir.join("exefs").join(name), dir.join(name)].into_iter().find(|path| path.is_file())
}

/// a patch format, applying a patch to data.
type Patch = fn(&[u8], &[u8]) -> Result<Vec<u8>, FsError>;

/// the title's code with its mods' changes to it, a code.bin in its place,
/// then a code.ips or code.bps patch over it, and whether there were any.
pub fn code(title: &Title, data_dir: Option<&Path>) -> Result<(Vec<u8>, bool), FsError> {
    let original = title.code()?;
    let Some(data_dir) = data_dir else {
        return Ok((original, false));
    };
    let mut code = original.clone();
    let dir = dir(data_dir, title.program_id());
    if let Some(path) = find(&dir, "code.bin") {
        match std::fs::read(&path) {
            Ok(bytes) => {
                code = decompressed(bytes, title);
                log::info!("mods: the game's code is {}", path.display());
            }
            Err(error) => log::warn!("mods: {}: {error}", path.display()),
        }
    }
    let patches: [(&str, Patch); 2] = [("code.ips", patch::ips), ("code.bps", patch::bps)];
    for (name, apply) in patches {
        let Some(path) = find(&dir, name) else { continue };
        match std::fs::read(&path).map_err(FsError::from).and_then(|patch| apply(&patch, &code)) {
            Ok(patched) => {
                code = patched;
                log::info!("mods: {} patched the game's code", path.display());
            }
            Err(error) => log::warn!("mods: {}: {error}", path.display()),
        }
    }
    let modded = code != original;
    Ok((code, modded))
}

/// a code.bin as the game runs it. most are kept decompressed, one that is
/// too short for the segments the exheader describes is still compressed.
fn decompressed(bytes: Vec<u8>, title: &Title) -> Vec<u8> {
    let exheader = &title.exheader;
    let needed = (exheader.text.num_pages + exheader.rodata.num_pages) * PAGE_SIZE + exheader.data.size;
    if bytes.len() >= needed as usize {
        return bytes;
    }
    lz77::decompress(&bytes).unwrap_or(bytes)
}

/// lays the title's mods over it, saying in the log what they changed.
pub fn lay(title: &mut Title, data_dir: Option<&Path>) {
    let Some(data_dir) = data_dir else {
        return;
    };
    let dir = dir(data_dir, title.program_id());
    if !dir.is_dir() {
        return;
    }
    match title.lay_mods(&dir) {
        Ok(Some(changes)) => log::info!(
            "mods from {}: {} files replaced, {} added, {} patched, {} removed",
            dir.display(),
            changes.replaced,
            changes.added,
            changes.patched,
            changes.removed
        ),
        Ok(None) => log::info!("mods: {} changes nothing in the game's files", dir.display()),
        Err(error) => log::warn!("mods: {} can't be laid over the game: {error}", dir.display()),
    }
}
