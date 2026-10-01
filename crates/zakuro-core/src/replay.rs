//! a run's inputs written down frame by frame, with the clock it started
//! at, so the run can be played back exactly. a line per change of input,
//! the frame it starts at, the buttons, the circle pad and the touch.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use crate::services::hid::{InputState, PadState};

pub struct Recorder {
    out: BufWriter<File>,
    last: Option<InputState>,
    frame: u64,
}

impl Recorder {
    /// starts a recording of a run whose clock starts at clock.
    pub fn create(path: &Path, clock: u64) -> std::io::Result<Recorder> {
        let mut out = BufWriter::new(File::create(path)?);
        writeln!(out, "clock {clock}")?;
        out.flush()?;
        Ok(Recorder { out, last: None, frame: 0 })
    }

    /// takes the input of the frame about to run.
    pub fn record(&mut self, input: InputState) {
        if self.last != Some(input) {
            let (x, y) = input.touch.map_or((-1, -1), |(x, y)| (x as i32, y as i32));
            let line = format!("{} {:x} {:?} {:?} {x} {y}", self.frame, input.buttons.bits(), input.circle_x, input.circle_y);
            // written through at once, the run may end in a crash
            if writeln!(self.out, "{line}").and_then(|()| self.out.flush()).is_err() {
                log::warn!("could not write to the input recording");
            }
            self.last = Some(input);
        }
        self.frame += 1;
    }
}

pub struct Replay {
    clock: u64,
    changes: Vec<(u64, InputState)>,
    next: usize,
    current: InputState,
}

impl Replay {
    pub fn open(path: &Path) -> std::io::Result<Replay> {
        let bad = |what: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("bad recording, {what}"));
        let mut lines = BufReader::new(File::open(path)?).lines();
        let first = lines.next().ok_or_else(|| bad("empty"))??;
        let clock = first.strip_prefix("clock ").and_then(|c| c.trim().parse().ok()).ok_or_else(|| bad("no clock"))?;
        let mut changes = Vec::new();
        for line in lines {
            let line = line?;
            let fields: Vec<&str> = line.split_whitespace().collect();
            let [frame, buttons, x, y, tx, ty] = fields[..] else { return Err(bad(&line)) };
            let parse = || -> Option<(u64, InputState)> {
                let (tx, ty): (i32, i32) = (tx.parse().ok()?, ty.parse().ok()?);
                Some((
                    frame.parse().ok()?,
                    InputState {
                        buttons: PadState::from_bits_retain(u32::from_str_radix(buttons, 16).ok()?),
                        circle_x: x.parse().ok()?,
                        circle_y: y.parse().ok()?,
                        touch: (tx >= 0).then_some((tx as u16, ty as u16)),
                    },
                ))
            };
            changes.push(parse().ok_or_else(|| bad(&line))?);
        }
        Ok(Replay { clock, changes, next: 0, current: InputState::default() })
    }

    /// the clock the recorded run started at.
    pub fn clock(&self) -> u64 {
        self.clock
    }

    /// the input of a frame, frames asked for in order.
    pub fn input(&mut self, frame: u64) -> InputState {
        while let Some(&(start, input)) = self.changes.get(self.next) {
            if start > frame {
                break;
            }
            self.current = input;
            self.next += 1;
        }
        self.current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// what a recording writes, a replay of it gives back frame for frame.
    #[test]
    fn a_recording_plays_back() {
        let path = std::env::temp_dir().join(format!("zakuro-replay-{}.txt", std::process::id()));
        let pressed = InputState { buttons: PadState::A, circle_x: 0.5, circle_y: -1.0, touch: Some((10, 20)) };
        let mut recorder = Recorder::create(&path, 1234).unwrap();
        for frame in 0..10 {
            recorder.record(if (3..6).contains(&frame) { pressed } else { InputState::default() });
        }
        drop(recorder);
        let mut replay = Replay::open(&path).unwrap();
        assert_eq!(replay.clock(), 1234);
        for frame in 0..10 {
            let expected = if (3..6).contains(&frame) { pressed } else { InputState::default() };
            assert_eq!(replay.input(frame), expected);
        }
        let _ = std::fs::remove_file(path);
    }
}
