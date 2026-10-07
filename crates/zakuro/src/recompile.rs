//! recompiling a game with 3dsrecomp, on a thread of its own while the rest
//! carries on, with the code and the modules its mods change.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use recomp3ds::build::{self, Event};
use recomp3ds::compile::Compiler;

/// the compiler a recompile uses: one there is, or Zig, downloaded first
/// into this folder of tools.
#[derive(Debug, Clone, PartialEq)]
pub enum Toolchain {
    Ready(Compiler),
    Download(PathBuf),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stage {
    /// fetching the compiler, how much of it has arrived, 0 to 1.
    Downloading(f32),
    /// finding the code and writing it as C.
    Generating,
    Compiling { done: usize, total: usize },
    /// linking the library and putting it where Zakuro finds it.
    Installing,
    Done,
    Failed(String),
}

impl Stage {
    pub fn finished(&self) -> bool {
        matches!(self, Stage::Done | Stage::Failed(_))
    }

    /// how far along it is, 0 to 1, going by the part that takes longest.
    pub fn fraction(&self) -> f32 {
        match self {
            Stage::Downloading(arrived) => *arrived,
            Stage::Generating => 0.05,
            Stage::Compiling { done, total } => 0.05 + 0.9 * *done as f32 / (*total).max(1) as f32,
            Stage::Installing => 0.95,
            Stage::Done | Stage::Failed(_) => 1.0,
        }
    }

    /// where an event of the build leaves it, none for one that changes
    /// nothing.
    fn after(event: &Event) -> Option<Stage> {
        match event {
            Event::Generated { .. } => Some(Stage::Compiling { done: 0, total: 1 }),
            Event::Compiled { done, total } => Some(Stage::Compiling { done: *done, total: *total }),
            Event::Built { .. } | Event::Installed(_) => Some(Stage::Installing),
            Event::Note(_) => None,
        }
    }
}

struct State {
    stage: Stage,
    finished_at: Option<Instant>,
}

pub struct Job {
    pub program_id: u64,
    pub name: String,
    /// the user was told how it ended.
    pub announced: bool,
    started: Instant,
    state: Arc<Mutex<State>>,
    cancel: Arc<AtomicBool>,
}

impl Job {
    /// starts recompiling the game at rom with toolchain's compiler, with
    /// what its mods in data_dir change. the compilers run below normal
    /// priority, so that a game can be played meanwhile.
    pub fn start(rom: &Path, program_id: u64, name: &str, data_dir: Option<&Path>, toolchain: Toolchain) -> Job {
        let first = if matches!(toolchain, Toolchain::Download(_)) { Stage::Downloading(0.0) } else { Stage::Generating };
        let state = Arc::new(Mutex::new(State { stage: first, finished_at: None }));
        let cancel = Arc::new(AtomicBool::new(false));
        let (progress, stop, rom) = (state.clone(), cancel.clone(), rom.to_owned());
        let (game, data_dir) = (name.to_owned(), data_dir.map(Path::to_owned));
        std::thread::spawn(move || {
            let events = |event: Event| {
                if let (Some(stage), Ok(mut state)) = (Stage::after(&event), progress.lock()) {
                    state.stage = stage;
                }
            };
            let compiler = match toolchain {
                Toolchain::Ready(compiler) => Ok(compiler),
                Toolchain::Download(tools) => {
                    let arrived = |fraction: f32| {
                        if let Ok(mut state) = progress.lock() {
                            state.stage = Stage::Downloading(fraction);
                        }
                    };
                    let downloaded = crate::zig::download(&tools, &arrived, &stop);
                    if let Ok(mut state) = progress.lock() {
                        state.stage = Stage::Generating;
                    }
                    downloaded
                }
            };
            let result = compiler.and_then(|compiler| recompile(&rom, program_id, &game, data_dir.as_deref(), compiler, &stop, &events));
            if let Ok(mut state) = progress.lock() {
                state.stage = match result {
                    Ok(_) => Stage::Done,
                    Err(_) if stop.load(Ordering::Relaxed) => Stage::Failed("cancelled".to_owned()),
                    Err(error) => Stage::Failed(error),
                };
                state.finished_at = Some(Instant::now());
            }
        });
        Job { program_id, name: name.to_owned(), announced: false, started: Instant::now(), state, cancel }
    }

    pub fn stage(&self) -> Stage {
        self.state.lock().map(|state| state.stage.clone()).unwrap_or(Stage::Failed("lost track".to_owned()))
    }

    /// how long it has run, or ran.
    pub fn elapsed(&self) -> Duration {
        let finished = self.state.lock().ok().and_then(|state| state.finished_at);
        finished.unwrap_or_else(Instant::now) - self.started
    }

    /// stops it after the files being compiled, the C being written first
    /// if it is at that.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// recompiles the game at rom, its code and its modules the way its mods
/// in data_dir leave them when they change any.
fn recompile(
    rom: &Path,
    program_id: u64,
    name: &str,
    data_dir: Option<&Path>,
    compiler: Compiler,
    cancel: &AtomicBool,
    events: &(dyn Fn(Event) + Sync),
) -> Result<PathBuf, String> {
    let changes = data_dir.and_then(|data_dir| Changes::read(rom, program_id, data_dir));
    let files = |path: &str| changes.as_ref()?.module(path);
    log::info!("recompiling {name} with {}", compiler.describe());
    let mut options = build::Options { cancel: Some(cancel), compiler: Some(compiler), background: true, ..build::Options::default() };
    if let Some(changes) = &changes {
        log::info!(
            "recompiling {name} with its mods, which change {}. the library replaces one made without them, and should the mods go, the functions they changed run in the interpreter",
            changes.describe()
        );
        options.mods = recomp3ds::Mods { code: changes.code.as_deref(), exheader: changes.exheader.as_deref(), romfs: Some(&files) };
    }
    build::build(rom, &options, events)
}

/// what a game's mods change in what gets recompiled.
struct Changes {
    /// the code, decompressed, when they change it.
    code: Option<Vec<u8>>,
    /// the game's modules and static.crs they replace or patch, by their
    /// paths in the RomFS, with their bytes.
    modules: Vec<(String, Vec<u8>)>,
    /// the exheader they give, which says where the code's segments lie.
    exheader: Option<Vec<u8>>,
}

impl Changes {
    /// what the mods in data_dir change for the game at rom, none when they
    /// leave its code and its modules alone.
    fn read(rom: &Path, program_id: u64, data_dir: &Path) -> Option<Changes> {
        if !zakuro_core::mods::present(data_dir, program_id) {
            return None;
        }
        let mut title = zakuro_fs::Title::load(rom).inspect_err(|error| log::warn!("mods: {}: {error}", rom.display())).ok()?;
        let exheader = zakuro_core::mods::exheader(&title, data_dir).map(|(bytes, _)| bytes);
        zakuro_core::mods::lay(&mut title, Some(data_dir));
        let code = match zakuro_core::mods::code(&title, Some(data_dir)) {
            Ok((code, modded)) => modded.then_some(code),
            Err(error) => {
                log::warn!("mods: the game's code can't be read: {error}");
                None
            }
        };
        // 3dsrecomp reads the modules the game has, one the mod adds is left
        // to the interpreter
        let modules = match (&title.layered, &title.romfs) {
            (Some(layered), Some(game)) => layered.modded_files(|path| is_module(path) && game.lookup(path).is_ok()),
            _ => Vec::new(),
        };
        (code.is_some() || !modules.is_empty()).then_some(Changes { code, modules, exheader })
    }

    /// the bytes the mods put in place of the module at path.
    fn module(&self, path: &str) -> Option<Vec<u8>> {
        self.modules.iter().find(|(at, _)| at.eq_ignore_ascii_case(path)).map(|(_, bytes)| bytes.clone())
    }

    /// what they change, the code and the modules by path.
    fn describe(&self) -> String {
        let code = self.code.as_ref().map(|_| "the code");
        code.into_iter().chain(self.modules.iter().map(|(path, _)| path.as_str())).collect::<Vec<_>>().join(", ")
    }
}

/// whether the RomFS file at path is code 3dsrecomp reads, a CRO module or
/// the static module that describes the executable.
fn is_module(path: &str) -> bool {
    let path = path.to_ascii_lowercase();
    path.ends_with(".cro") || path == "static.crs"
}

#[cfg(test)]
mod tests {
    use super::*;
    use zakuro_fs::romfs_build::{self, BuildFile};

    #[test]
    fn the_build_moves_the_stage_along() {
        let generated = Event::Generated { files: 279, bytes: 1, overrides: 0 };
        assert_eq!(Stage::after(&generated), Some(Stage::Compiling { done: 0, total: 1 }));
        assert_eq!(Stage::after(&Event::Compiled { done: 140, total: 279 }), Some(Stage::Compiling { done: 140, total: 279 }));
        assert_eq!(Stage::after(&Event::Installed("/x".into())), Some(Stage::Installing));
        assert_eq!(Stage::after(&Event::Note("hm".to_owned())), None);
        assert!(Stage::Compiling { done: 140, total: 279 }.fraction() > 0.5);
    }

    /// the id of the game the tests make, which no real game has.
    const PROGRAM_ID: u64 = 0x0004_0000_0FF3_DE00;

    fn put(out: &mut [u8], at: usize, bytes: &[u8]) {
        out[at..at + bytes.len()].copy_from_slice(bytes);
    }

    /// writes a decrypted .cxi to a file named after test, its code all
    /// text and its RomFS holding files.
    fn rom(test: &str, code: &[u8], files: &[(&str, &[u8])]) -> PathBuf {
        // the NCCH header, never encrypted, the exheader after it, and the
        // ExeFS at 0x600 with .code its only file
        let mut out = vec![0; 0x800];
        put(&mut out, 0x100, b"NCCH");
        put(&mut out, 0x118, &PROGRAM_ID.to_le_bytes());
        out[0x18F] = 0x04;
        put(&mut out, 0x1A0, &3u32.to_le_bytes());
        put(&mut out, 0x210, &[0x0010_0000, 1, code.len() as u32].map(u32::to_le_bytes).concat());
        put(&mut out, 0x600, b".code");
        put(&mut out, 0x60C, &(code.len() as u32).to_le_bytes());
        out.extend_from_slice(code);
        // the RomFS, an IVFC header without hashes and level 3 right after
        let files: Vec<_> = files.iter().map(|(path, data)| BuildFile { path: path.to_string(), data: data.to_vec() }).collect();
        let at = out.len().next_multiple_of(0x200);
        out.resize(at + 0x60, 0);
        put(&mut out, at, b"IVFC");
        put(&mut out, at + 0x04, &0x0001_0000u32.to_le_bytes());
        out[at + 0x4C] = 4;
        out.extend(romfs_build::build(&files));
        let size = (out.len() - at).div_ceil(0x200) as u32;
        put(&mut out, 0x1B0, &((at / 0x200) as u32).to_le_bytes());
        put(&mut out, 0x1B4, &size.to_le_bytes());
        let path = std::env::temp_dir().join(format!("zakuro-recompile-{}-{test}.cxi", std::process::id()));
        std::fs::write(&path, out).unwrap();
        path
    }

    fn write(path: &Path, data: &[u8]) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, data).unwrap();
    }

    /// what a mod does to the code and to the game's modules is what gets
    /// recompiled, and a mod of other files changes nothing recompiled.
    #[test]
    fn the_code_and_the_modules_a_mod_changes_are_found() {
        let files: [(&str, &[u8]); 4] = [("static.crs", b"crs"), ("cro/Battle.cro", b"battle"), ("cro/Field.cro", b"field"), ("a.bin", b"a")];
        let rom = rom("found", &[0; 8], &files);
        let data_dir = std::env::temp_dir().join(format!("zakuro-recompile-{}-found", std::process::id()));
        let mods = zakuro_core::mods::dir(&data_dir, PROGRAM_ID);
        assert!(Changes::read(&rom, PROGRAM_ID, &data_dir).is_none());
        write(&mods.join("romfs/a.bin"), b"b");
        write(&mods.join("romfs/cro/Extra.cro"), b"extra");
        assert!(Changes::read(&rom, PROGRAM_ID, &data_dir).is_none(), "neither the code nor the game's modules changed");

        let ips = |data: &[u8]| [&b"PATCH"[..], &[0, 0, 0, 0, data.len() as u8], data, b"EOF"].concat();
        write(&mods.join("exefs/code.ips"), &ips(&[0xAA]));
        write(&mods.join("romfs/cro/battle.cro"), b"modded battle");
        write(&mods.join("romfs_ext/static.crs.ips"), &ips(b"C"));
        let changes = Changes::read(&rom, PROGRAM_ID, &data_dir).unwrap();
        assert_eq!(changes.code.as_deref(), Some(&[0xAA, 0, 0, 0, 0, 0, 0, 0][..]));
        assert_eq!(changes.module("cro/Battle.cro").as_deref(), Some(&b"modded battle"[..]));
        assert_eq!(changes.module("static.crs").as_deref(), Some(&b"Crs"[..]));
        assert_eq!(changes.module("cro/Field.cro"), None);
        assert_eq!(changes.module("cro/Extra.cro"), None);
        let mut described: Vec<String> = changes.describe().split(", ").map(str::to_owned).collect();
        described.sort();
        assert_eq!(described, ["cro/Battle.cro", "static.crs", "the code"]);

        std::fs::remove_dir_all(data_dir).unwrap();
        std::fs::remove_file(rom).unwrap();
    }

    /// a mod's exheader.bin takes the place of the game's when it is the
    /// game's own with its code made longer, and goes to 3dsrecomp, and one
    /// that looks like another game's, or still encrypted, is left out.
    #[test]
    fn a_mods_exheader_is_taken_when_it_is_the_games() {
        let rom = rom("exheader", &[0; 8], &[]);
        let data_dir = std::env::temp_dir().join(format!("zakuro-recompile-{}-exheader", std::process::id()));
        let mods = zakuro_core::mods::dir(&data_dir, PROGRAM_ID);
        let mut exheader = vec![0; 0x800];
        put(&mut exheader, 0x10, &[0x0010_0000, 1, 12].map(u32::to_le_bytes).concat());
        write(&mods.join("exheader.bin"), &exheader);
        write(&mods.join("code.bin"), &[0xAA; 12]);
        let laid = || {
            let mut title = zakuro_fs::Title::load(&rom).unwrap();
            zakuro_core::mods::lay(&mut title, Some(&data_dir));
            title.exheader.text.size
        };
        assert_eq!(laid(), 12);
        let changes = Changes::read(&rom, PROGRAM_ID, &data_dir).unwrap();
        assert_eq!(changes.exheader.as_deref().map(<[u8]>::len), Some(0x400));
        assert_eq!(changes.code.as_deref(), Some(&[0xAA; 12][..]));

        put(&mut exheader, 0x10, &0x0020_0000u32.to_le_bytes());
        write(&mods.join("exheader.bin"), &exheader);
        assert_eq!(laid(), 8, "another game's exheader is left out");
        assert!(Changes::read(&rom, PROGRAM_ID, &data_dir).unwrap().exheader.is_none());

        std::fs::remove_dir_all(data_dir).unwrap();
        std::fs::remove_file(rom).unwrap();
    }

    #[test]
    fn modules_are_cro_files_and_static_crs() {
        assert!(is_module("cro/Battle.cro") && is_module("Battle.CRO") && is_module("static.crs"));
        assert!(!is_module("cro/static.crs") && !is_module("cro/Battle.crr") && !is_module("a.bin"));
    }
}
