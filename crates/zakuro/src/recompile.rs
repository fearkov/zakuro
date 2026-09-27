//! recompiling a game with 3dsrecomp, which runs on its own while the rest
//! carries on.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq)]
pub enum Stage {
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
            Stage::Generating => 0.05,
            Stage::Compiling { done, total } => 0.05 + 0.9 * *done as f32 / (*total).max(1) as f32,
            Stage::Installing => 0.95,
            Stage::Done | Stage::Failed(_) => 1.0,
        }
    }
}

struct State {
    stage: Stage,
    finished_at: Option<Instant>,
    /// the last thing 3dsrecomp complained about.
    error: String,
}

pub struct Job {
    pub program_id: u64,
    pub name: String,
    /// the user was told how it ended.
    pub announced: bool,
    started: Instant,
    state: Arc<Mutex<State>>,
    child: Arc<Mutex<Option<Child>>>,
}

impl Job {
    /// runs recompiler on the game at rom.
    pub fn start(recompiler: &Path, rom: &Path, program_id: u64, name: &str) -> Result<Job, String> {
        let mut child = Command::new(recompiler)
            .arg("build")
            .arg(rom)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| format!("could not run {}, {error}", recompiler.display()))?;
        let stdout = child.stdout.take().expect("piped above");
        let stderr = child.stderr.take().expect("piped above");
        let state = Arc::new(Mutex::new(State { stage: Stage::Generating, finished_at: None, error: String::new() }));
        let child = Arc::new(Mutex::new(Some(child)));

        let errors = state.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if let Ok(mut state) = errors.lock() {
                    state.error = line;
                }
            }
        });

        let (progress, waiting) = (state.clone(), child.clone());
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if let (Some(stage), Ok(mut state)) = (parse(&line), progress.lock()) {
                    state.stage = stage;
                }
            }
            // the output ended, so it finished or was stopped
            let status = waiting.lock().ok().and_then(|mut child| child.take()).map(|mut child| child.wait());
            if let Ok(mut state) = progress.lock() {
                state.stage = match status {
                    Some(Ok(status)) if status.success() => Stage::Done,
                    Some(Ok(status)) if state.error.is_empty() => Stage::Failed(format!("3dsrecomp stopped with {status}")),
                    Some(Ok(_)) => Stage::Failed(state.error.clone()),
                    Some(Err(error)) => Stage::Failed(error.to_string()),
                    None => Stage::Failed("cancelled".to_owned()),
                };
                state.finished_at = Some(Instant::now());
            }
        });

        Ok(Job { program_id, name: name.to_owned(), announced: false, started: Instant::now(), state, child })
    }

    pub fn stage(&self) -> Stage {
        self.state.lock().map(|state| state.stage.clone()).unwrap_or(Stage::Failed("lost track".to_owned()))
    }

    /// how long it has run, or ran.
    pub fn elapsed(&self) -> Duration {
        let finished = self.state.lock().ok().and_then(|state| state.finished_at);
        finished.unwrap_or_else(Instant::now) - self.started
    }

    pub fn cancel(&self) {
        if let Ok(mut child) = self.child.lock() {
            if let Some(mut running) = child.take() {
                let _ = running.kill();
                let _ = running.wait();
            }
        }
    }
}

/// what a line of 3dsrecomp's output says about where it is.
fn parse(line: &str) -> Option<Stage> {
    if let Some(rest) = line.strip_prefix("compiled ") {
        let (done, total) = rest.split_once(" of ")?;
        return Some(Stage::Compiling { done: done.trim().parse().ok()?, total: total.trim().parse().ok()? });
    }
    if line.starts_with("wrote ") {
        return Some(Stage::Compiling { done: 0, total: 1 });
    }
    if line.starts_with("built ") {
        return Some(Stage::Installing);
    }
    None
}

/// 3dsrecomp, where the settings say or else on the path.
pub fn find(configured: Option<&Path>) -> Option<PathBuf> {
    if let Some(path) = configured {
        return path.is_file().then(|| path.to_owned());
    }
    let paths = std::env::var_os("PATH")?;
    std::env::split_paths(&paths).map(|dir| dir.join("3dsrecomp")).find(|path| path.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_is_read_from_the_output() {
        assert_eq!(parse("wrote 279 files, 432 MiB of C, 0 overrides"), Some(Stage::Compiling { done: 0, total: 1 }));
        assert_eq!(parse("compiled 140 of 279"), Some(Stage::Compiling { done: 140, total: 279 }));
        assert_eq!(parse("built /x/000400000011C500.so in 591.9s"), Some(Stage::Installing));
        assert_eq!(parse("something else"), None);
        assert!(Stage::Compiling { done: 140, total: 279 }.fraction() > 0.5);
    }
}
