//! mods, in the data dir's mods folder under the title id, laid out the way Luma3DS and Citra take them

use std::path::{Path, PathBuf};

use zakuro_common::memory_map::PAGE_SIZE;
use zakuro_fs::ncch::ExHeader;
use zakuro_fs::{lz77, patch, FsError, Title};

/// the folder a title's mods go in
pub fn dir(data_dir: &Path, program_id: u64) -> PathBuf {
    let mods = data_dir.join("mods");
    let upper = mods.join(format!("{program_id:016X}"));
    let lower = mods.join(format!("{program_id:016x}"));
    if !upper.exists() && lower.exists() { lower } else { upper }
}

/// a title's texture pack folder inside its mods folder, laid out like Citra's load/textures/<title id>
pub fn textures(data_dir: &Path, program_id: u64) -> PathBuf {
    dir(data_dir, program_id).join("textures")
}

/// files changing a title's code, each in the mod's folder or its exefs folder
const CODE: [&str; 3] = ["code.bin", "code.ips", "code.bps"];

/// the exheader a mod gives instead of the title's, in its folder, needed by mods that grow the code
const EXHEADER: &str = "exheader.bin";

/// does the title's mods folder hold a mod
pub fn present(data_dir: &Path, program_id: u64) -> bool {
    let dir = dir(data_dir, program_id);
    let files = ["romfs", "romfs_ext"].iter().any(|part| std::fs::read_dir(dir.join(part)).is_ok_and(|mut entries| entries.next().is_some()));
    files || CODE.iter().any(|name| find(&dir, name).is_some()) || dir.join(EXHEADER).is_file()
}

/// the title mods' exheader.bin (the 0x400 bytes describing the program) and what it says
/// when it looks like the title's (code starting where the title's does, an encrypted one wouldn't)
/// the ExeFS code is the game's own, so compression stays as the game's exheader says
pub fn exheader(title: &Title, data_dir: &Path) -> Option<(Vec<u8>, ExHeader)> {
    let path = dir(data_dir, title.program_id()).join(EXHEADER);
    let mut bytes = std::fs::read(&path).ok()?;
    let Some(described) = bytes.get(..ExHeader::HASHED_SIZE) else {
        log::warn!("mods: {} is {} bytes, an exheader has {}", path.display(), bytes.len(), ExHeader::HASHED_SIZE);
        return None;
    };
    match ExHeader::parse(described) {
        Ok(mut exheader) if exheader.text.address == title.exheader.text.address => {
            exheader.compress_code = title.exheader.compress_code;
            bytes.truncate(ExHeader::HASHED_SIZE);
            bytes[0x0D] = bytes[0x0D] & !1 | title.exheader.compress_code as u8;
            Some((bytes, exheader))
        }
        Ok(_) => {
            log::warn!("mods: {} is not this game's exheader, or is still encrypted, it is left out", path.display());
            None
        }
        Err(error) => {
            log::warn!("mods: {}: {error}", path.display());
            None
        }
    }
}

fn find(dir: &Path, name: &str) -> Option<PathBuf> {
    [dir.join("exefs").join(name), dir.join(name)].into_iter().find(|path| path.is_file())
}

/// a patch format, applies a patch to data
type Patch = fn(&[u8], &[u8]) -> Result<Vec<u8>, FsError>;

/// the title's code w/ its mods' changes, a code.bin instead of it, then a code.ips or code.bps patch on top, plus whether there were any
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

/// a code.bin as the game runs it, most are kept decompressed, one too short for the exheader's segments is still compressed
fn decompressed(bytes: Vec<u8>, title: &Title) -> Vec<u8> {
    let exheader = &title.exheader;
    let needed = (exheader.text.num_pages + exheader.rodata.num_pages) * PAGE_SIZE + exheader.data.size;
    if bytes.len() >= needed as usize {
        return bytes;
    }
    lz77::decompress(&bytes).unwrap_or(bytes)
}

/// lays the title's mods over it, logging what they changed
pub fn lay(title: &mut Title, data_dir: Option<&Path>) {
    let Some(data_dir) = data_dir else {
        return;
    };
    let dir = dir(data_dir, title.program_id());
    if !dir.is_dir() {
        return;
    }
    if let Some((_, exheader)) = exheader(title, data_dir) {
        log::info!("mods: the game's exheader is {}", dir.join(EXHEADER).display());
        title.exheader = exheader;
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
    // and over the update's files, a game reads those besides its own
    if let Some(update) = title.update_mut() {
        match update.title.lay_mods(&dir) {
            Ok(Some(changes)) => log::info!(
                "mods over the update's files: {} replaced, {} added, {} patched, {} removed",
                changes.replaced,
                changes.added,
                changes.patched,
                changes.removed
            ),
            Ok(None) => {}
            Err(error) => log::warn!("mods: {} can't be laid over the update: {error}", dir.display()),
        }
    }
}
