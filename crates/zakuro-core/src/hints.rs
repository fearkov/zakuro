//! where a title ran in the interpreter while it had a recompiled library,
//! code 3dsrecomp did not find. the addresses go in a file next to the
//! library, which the next build takes as functions.

use std::collections::BTreeSet;
use std::ops::Range;
use std::path::{Path, PathBuf};

/// how often the addresses go to the file, in frames, a minute.
pub const SAVE_EVERY: u64 = 3600;

pub struct Hints {
    path: PathBuf,
    /// the executable's code, the only place addresses are the same every
    /// run.
    text: Range<u32>,
    /// odd for Thumb.
    seen: BTreeSet<u32>,
    saved: usize,
    /// the instruction before was interpreted too.
    interpreting: bool,
}

impl Hints {
    /// the hints for a library, with those an earlier run wrote down.
    pub fn new(library: &Path, text: Range<u32>) -> Hints {
        let path = library.with_extension("hints");
        let seen: BTreeSet<u32> = std::fs::read_to_string(&path)
            .unwrap_or_default()
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .filter_map(|line| u32::from_str_radix(line.trim_start_matches("0x"), 16).ok())
            .collect();
        let saved = seen.len();
        Hints { path, text, seen, saved, interpreting: false }
    }

    /// the library ran some code.
    pub fn library_ran(&mut self) {
        self.interpreting = false;
    }

    /// the interpreter ran an instruction at pc that the library has no
    /// code for. where that follows the library's code, missing code starts,
    /// what it goes on to call the next build finds from there.
    pub fn interpreted(&mut self, pc: u32, thumb: bool) {
        let at = pc & !1;
        if !self.interpreting && self.text.contains(&at) {
            self.seen.insert(at | thumb as u32);
        }
        self.interpreting = true;
    }

    /// writes the file when there is anything new.
    pub fn save(&mut self) {
        if self.seen.len() == self.saved {
            return;
        }
        let mut text = String::from("# where Zakuro interpreted this title's code, odd for Thumb, which 3dsrecomp build recompiles\n");
        for address in &self.seen {
            text.push_str(&format!("{address:08X}\n"));
        }
        let partial = self.path.with_extension("hints.new");
        match std::fs::write(&partial, text).and_then(|()| std::fs::rename(&partial, &self.path)) {
            Ok(()) => self.saved = self.seen.len(),
            Err(error) => log::warn!("could not write {}, {error}", self.path.display()),
        }
    }
}
