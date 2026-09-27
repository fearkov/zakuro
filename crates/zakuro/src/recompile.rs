//! recompiling a game with 3dsrecomp, on a thread of its own while the rest
//! carries on.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use recomp3ds::build::{self, Event};

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
    /// starts recompiling the game at rom.
    pub fn start(rom: &Path, program_id: u64, name: &str) -> Job {
        let state = Arc::new(Mutex::new(State { stage: Stage::Generating, finished_at: None }));
        let cancel = Arc::new(AtomicBool::new(false));
        let (progress, stop, rom) = (state.clone(), cancel.clone(), rom.to_owned());
        std::thread::spawn(move || {
            let events = |event: Event| {
                if let (Some(stage), Ok(mut state)) = (Stage::after(&event), progress.lock()) {
                    state.stage = stage;
                }
            };
            let options = build::Options { cancel: Some(&stop), ..build::Options::default() };
            let result = build::build(&rom, &options, &events);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_build_moves_the_stage_along() {
        let generated = Event::Generated { files: 279, bytes: 1, overrides: 0 };
        assert_eq!(Stage::after(&generated), Some(Stage::Compiling { done: 0, total: 1 }));
        assert_eq!(Stage::after(&Event::Compiled { done: 140, total: 279 }), Some(Stage::Compiling { done: 140, total: 279 }));
        assert_eq!(Stage::after(&Event::Installed("/x".into())), Some(Stage::Installing));
        assert_eq!(Stage::after(&Event::Note("hm".to_owned())), None);
        assert!(Stage::Compiling { done: 140, total: 279 }.fraction() > 0.5);
    }
}
