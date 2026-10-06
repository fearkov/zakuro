//! mods, kept in the data directory's mods folder under the title's id, laid
//! out the way Luma3DS and Citra take them.

use std::path::{Path, PathBuf};

use zakuro_fs::Title;

/// the folder a title's mods go in.
pub fn dir(data_dir: &Path, program_id: u64) -> PathBuf {
    let mods = data_dir.join("mods");
    let upper = mods.join(format!("{program_id:016X}"));
    let lower = mods.join(format!("{program_id:016x}"));
    if !upper.exists() && lower.exists() { lower } else { upper }
}

/// whether the title's mods folder holds a mod.
pub fn present(data_dir: &Path, program_id: u64) -> bool {
    let dir = dir(data_dir, program_id);
    ["romfs", "romfs_ext"].iter().any(|part| std::fs::read_dir(dir.join(part)).is_ok_and(|mut entries| entries.next().is_some()))
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
