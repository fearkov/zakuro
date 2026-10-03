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

    /// the library had code but not budget for the whole block, so the
    /// interpreter goes on through it. what it runs until the library takes
    /// over again is known code, not missing code.
    pub fn library_declined(&mut self) {
        self.interpreting = true;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn what_runs_after_the_library_declines_is_no_hint() {
        let library = std::env::temp_dir().join(format!("zakuro-hints-{}", std::process::id())).join("lib.so");
        let mut hints = Hints::new(&library, 0x0010_0000..0x0020_0000);
        // the rest of a block the budget did not cover
        hints.library_declined();
        hints.interpreted(0x0010_0004, false);
        hints.interpreted(0x0010_0008, false);
        assert!(hints.seen.is_empty());
        // code the library does not have, right after it ran
        hints.library_ran();
        hints.interpreted(0x0010_0100, true);
        assert_eq!(hints.seen.iter().copied().collect::<Vec<_>>(), vec![0x0010_0101]);
    }
}
