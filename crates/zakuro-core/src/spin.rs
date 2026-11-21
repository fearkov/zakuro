//! loops in recompiled code that poll the clock
//! a thread alone on the core w/ nothing to do till some time spins in a loop that reads the clock and takes and
//! drops a mutex, each turn the same except the ticks it reads, till one's late enough
//! on a console that costs nothing, here every turn runs
//!
//! once a turn is seen to go the same way whatever the turns b4 it left, the run skips whole turns at once,
//! up to the last b4 the loop would go another way or the next thing happens
//! it leaves the machine like running every one of them would (the clock, the history, the preempt and the instructions counted)
//! the code then runs the turn after them for real, which goes the same way and so writes over everything the skipped ones would have left
//!
//! which way a turn goes from a tick is found by running its code on the side (see
//! recompiled::Library::run_on_the_side) on a copy of the core
//! the clock answered as it'd be then, the mutex taken and dropped in a count of its own, and what it writes put back after
//! we get its svcs, when each comes, the cycles it takes and the bytes it changes
//! a turn goes another way past a tick (like a time reached does) and the same b4 it,
//! so the turns to skip are found by halving (binary search ftw)
//!
//! a try starts at a tick the host answers when the svc history shows a loop,
//! and the ticks a loop's other svcs bring to the host b4 then have the next one come too
//! a loop going another way soon gets tried again once it did, for the next one it comes back to,
//! and one whose turn can't be learned waits, longer each time

use crate::kernel::svc::SYSTEM_TICK_COST;
use crate::kernel::thread::ThreadId;
use crate::recompiled::{self, Core, Sandbox};
use crate::{System, HISTORY_LENGTH, PREEMPT_INTERVAL};

/// the most svcs a turn of a polling loop makes
const MOST_SVCS: usize = 16;
/// skipping fewer turns than this isn't worth finding them
const FEWEST_TURNS: u64 = 4;
/// how close the turn where a loop goes another way gets found, the turns between just run
const ROUGHLY: u64 = 16;
/// after a loop's turn couldn't be learned, no try starts for this long at first,
/// twice as long each time that loop fails again, up to the most
const RETRY: u64 = 8192;
const MOST_RETRY: u64 = 2 * 1024 * 1024;

/// loops skipped, and the one learned
pub struct SpinLoops {
    /// b4 this cycle no try starts, an svc pays one comparison
    next_try: u64,
    /// the loop whose turn couldn't be learned last (its thread, the svc at and the svcs in a turn) and how long its next try waited
    failed: Option<(ThreadId, u32, usize)>,
    retry: u64,
    learned: Option<Box<Learned>>,
    /// where turns run on the side write, made at the first try
    sandbox: Option<Box<Sandbox>>,
    /// the loop learned is where the code is, at another of its ticks, a try wants the next one
    wants_tick: bool,
    /// cycles and turns skipped, for reports
    pub skipped: u64,
    pub turns: u64,
}

impl SpinLoops {
    /// loops skipped or, when off, never looked for, ZAKURO_SPIN=0 turns them off too
    pub fn new(on: bool) -> SpinLoops {
        let on = on && std::env::var("ZAKURO_SPIN").map_or(true, |value| value != "0");
        SpinLoops { next_try: if on { 0 } else { u64::MAX }, failed: None, retry: 0, learned: None, sandbox: None, wants_tick: false, skipped: 0, turns: 0 }
    }

    /// forgets the loop learned and when to try next, for a loaded state
    pub(crate) fn forget(&mut self) {
        self.learned = None;
        self.failed = None;
        self.retry = 0;
        if self.next_try != u64::MAX {
            self.next_try = 0;
        }
    }

    /// is an svc at tick worth a try, the history showing a loop
    #[inline(always)]
    pub(crate) fn due(&self, tick: u64) -> bool {
        tick >= self.next_try
    }

    /// does the try just made want the next tick, the code being in the loop learned at another of its ticks
    pub(crate) fn wants_tick(&mut self) -> bool {
        std::mem::take(&mut self.wants_tick)
    }
}

/// what a turn run on the side did
/// its svcs, the addr after each (bit 0 set for Thumb),
/// the cycles from the turn's start it came at and its number (the first is the one reading the clock it starts
/// at), the cycles to the next turn's, and the bytes it wrote
#[derive(PartialEq, Eq)]
struct Turn {
    svcs: Vec<(u32, u64, u32)>,
    cost: u64,
    written: Vec<(u32, recompiled::Bytes)>,
}

struct Learned {
    thread: ThreadId,
    thumb: bool,
    /// the core where a turn starts, the clock reading svc made and not answered yet, the registers the same at every turn's start, by mask
    regs: [u32; 16],
    constant: u16,
    turn: Turn,
    /// the instructions a turn runs, its cycles minus what reading the clock costs
    instructions: u64,
    /// the mutex its waits take
    mutex: Option<u32>,
    /// the latest tick a turn was found to go the same way from, and the
    /// first it was found to go another way from
    same_until: u64,
    other_from: Option<u64>,
}

/// turns skipped, the cycles and instructions they took and their svcs,
/// for the svc to be answered after them
pub(crate) struct Skipped {
    pub cycles: u64,
    pub instructions: u64,
    pub svcs: u64,
}

/// what a try came to
enum Tried {
    Skipped(Skipped),
    /// a loop it'd skip if not for what comes first, tried again past that
    Past(u64),
    /// the loop learned, at another of its ticks, tried again at the next
    Next,
    /// a turn of the loop that couldn't be learned, tried again later
    Failed,
    /// nothing to skip now, tried again at the next tick the history shows
    /// a loop at
    Nothing,
}

impl System {
    /// b4 the clock is read in place at at (the core the code has at the svc, tick what it reads),
    /// skips whole turns of a loop polling it when the code is in one
    /// the history shows its turns of period svcs (see spin_period), up to the event or the deadline at most,
    /// w/ the history and the preempt left like answering each of their svcs would
    /// the svc then has to be answered after them, this makes sure tick_in_place does
    /// none, having done nothing, when there's nothing to skip
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn spin_skip(
        &mut self,
        core: &Core,
        tick: u64,
        at: u32,
        period: usize,
        event: Option<u64>,
        deadline: Option<u64>,
        library: &recompiled::Library,
    ) -> Option<Skipped> {
        // another thing watching what the CPU writes
        if self.memory.logging_writes() {
            return None;
        }
        let tried = self.try_spin_skip(core, tick, at, period, event, deadline, library);
        let thread = self.kernel.current_thread;
        let this = thread.map(|thread| (thread, at, period));
        match tried {
            Tried::Skipped(skipped) => {
                if self.spin.failed == this {
                    (self.spin.failed, self.spin.retry) = (None, 0);
                }
                return Some(skipped);
            }
            Tried::Past(past) => self.spin.next_try = past,
            Tried::Next => self.spin.wants_tick = true,
            Tried::Failed => {
                // the same loop failing again waits longer
                self.spin.retry = if self.spin.failed == this { (self.spin.retry * 2).min(MOST_RETRY) } else { RETRY };
                self.spin.failed = this;
                self.spin.next_try = tick + self.spin.retry;
            }
            Tried::Nothing => {}
        }
        None
    }

    #[allow(clippy::too_many_arguments)]
    fn try_spin_skip(
        &mut self,
        core: &Core,
        tick: u64,
        at: u32,
        period: usize,
        event: Option<u64>,
        deadline: Option<u64>,
        library: &recompiled::Library,
    ) -> Tried {
        let Some(thread) = self.kernel.current_thread else { return Tried::Nothing };
        // nothing but the thread and the clock till the horizon, and the pass any preempt b4 it has due is a quiet one
        let after = tick + SYSTEM_TICK_COST;
        if !self.plain_in_place(after, at, event, deadline, library) || !self.kernel.quiet_pass(after) {
            return Tried::Nothing;
        }
        let Some(quiet) = self.kernel.quiet_until() else { return Tried::Nothing };
        let horizon = [event, deadline].into_iter().flatten().fold(quiet.min(self.next_frame_boundary).min(self.next_audio_frame), u64::min);
        // too close to skip the turns of the loop learned, which is likely the one
        if self.spin.learned.as_ref().is_some_and(|learned| horizon.saturating_sub(after) / learned.turn.cost < FEWEST_TURNS + 2) {
            return Tried::Past(horizon);
        }
        // the loop learned, at another of its turn's ticks
        if self.spin.learned.as_ref().is_some_and(|learned| self.learned_turn_at(learned, thread, period).is_some_and(|phase| phase != 0)) {
            return Tried::Next;
        }
        let known = self.spin.learned.take().filter(|learned| self.still_learned(learned, thread, core, period));
        let mut learned = match known {
            Some(learned) => learned,
            None => match self.learn_turn(library, thread, core, tick, period) {
                Some(learned) => learned,
                None => return Tried::Failed,
            },
        };
        let turns = self.turns_to_skip(library, &mut learned, core, tick, horizon);
        let cost = learned.turn.cost;
        let after = tick + turns * cost + SYSTEM_TICK_COST;
        let skipped = (turns >= FEWEST_TURNS && self.plain_in_place(after, at, event, deadline, library))
            .then(|| self.skip_turns(&learned, tick, turns));
        // too few turns b4 the loop goes another way, so the next try comes once it did instead of at the horizon
        // a loop the code comes back to meanwhile would spin all the way to that
        let past = learned.other_from.map_or(horizon, |other| other.max(tick + cost).min(horizon));
        self.spin.learned = Some(learned);
        match skipped {
            Some(skipped) => Tried::Skipped(skipped),
            None => Tried::Past(past),
        }
    }

    /// how many svcs a turn of a loop around the svc at at makes, the last in the history repeating that often w/ the one at at first
    /// the history holds each svc answered in place, none when they don't repeat
    #[inline]
    pub(crate) fn spin_period(&self, at: u32) -> Option<usize> {
        // back 0 is the newest
        let entry = |back: usize| self.history[(self.history_index + 2 * HISTORY_LENGTH - 1 - back) % HISTORY_LENGTH];
        (1..=MOST_SVCS).find(|&period| entry(period - 1) == at && (period..HISTORY_LENGTH).all(|back| entry(back) == entry(back - period)))
    }

    /// does the history repeat (the svcs answered in place being a loop's turns), a try at skipping them wants the next tick for that
    #[inline]
    pub(crate) fn history_repeats(&self) -> bool {
        let entry = |back: usize| self.history[(self.history_index + 2 * HISTORY_LENGTH - 1 - back) % HISTORY_LENGTH];
        (1..=MOST_SVCS).any(|period| (period..HISTORY_LENGTH).all(|back| entry(back) == entry(back - period)))
    }

    /// where in a turn of the loop learned the svc at hand is,
    /// by the history's last turn (the svcs the thread made),
    /// how many svcs after the one the learned turns start at
    /// none when the history isn't that loop's
    fn learned_turn_at(&self, learned: &Learned, thread: ThreadId, period: usize) -> Option<usize> {
        let entry = |back: usize| self.history[(self.history_index + 2 * HISTORY_LENGTH - 1 - back) % HISTORY_LENGTH];
        let svcs = &learned.turn.svcs;
        (learned.thread == thread && svcs.len() == period)
            .then(|| (0..period).find(|&phase| (0..period).all(|index| entry(period - 1 - index) == svcs[(phase + index) % period].0)))
            .flatten()
    }

    /// is the loop learned the one the core's in,
    /// the one thread at the svc its turns start at and the core's registers the same as at each turn's start
    fn still_learned(&self, learned: &Learned, thread: ThreadId, core: &Core, period: usize) -> bool {
        learned.thumb == core.thumb()
            && self.learned_turn_at(learned, thread, period) == Some(0)
            && (0..16).all(|index| learned.constant & (1 << index) == 0 || core.r[index] == learned.regs[index])
    }

    /// what a turn of the loop from the core at tick does, and if it goes the same way whatever the turns b4 it left
    /// a turn, the turn after it, and that one again from the core instead, which has to go the same way and leave the same core and memory
    /// none when the three don't
    fn learn_turn(&mut self, library: &recompiled::Library, thread: ThreadId, core: &Core, tick: u64, period: usize) -> Option<Box<Learned>> {
        self.memory.log_all_writes();
        let learned = self.learn_on_the_side(library, thread, core, tick, period);
        self.put_back();
        self.memory.stop_logging_writes();
        learned
    }

    fn learn_on_the_side(&mut self, library: &recompiled::Library, thread: ThreadId, core: &Core, tick: u64, period: usize) -> Option<Box<Learned>> {
        let mut mutex = None;
        let (first, after_first) = self.turn_on_the_side(library, core, tick, period, &mut mutex)?;
        let next = tick + first.cost;
        let (second, after_second) = self.turn_on_the_side(library, &after_first, next, period, &mut mutex)?;
        let both = self.changed_on_the_side();
        self.put_back();
        let (alone, after_alone) = self.turn_on_the_side(library, core, next, period, &mut mutex)?;
        let alone_changed = self.changed_on_the_side();
        let same_way = (&first.svcs, first.cost) == (&second.svcs, second.cost) && first == alone;
        if !same_way || after_alone != after_second || alone_changed != both {
            return None;
        }
        let constant = (0..16).filter(|&index| core.r[index] == after_first.r[index] && core.r[index] == after_second.r[index]);
        let ticks = first.svcs.iter().filter(|&&(_, _, number)| number == 0x28).count() as u64;
        Some(Box::new(Learned {
            thread,
            thumb: core.thumb(),
            regs: core.r,
            constant: constant.fold(0u16, |mask, index| mask | 1 << index),
            instructions: first.cost - ticks * SYSTEM_TICK_COST,
            turn: first,
            mutex,
            same_until: next,
            other_from: None,
        }))
    }

    /// does a turn of the loop learned from the core at tick go the way it learned
    fn goes_the_same(&mut self, library: &recompiled::Library, learned: &Learned, core: &Core, tick: u64) -> bool {
        let mut mutex = learned.mutex;
        self.memory.log_all_writes();
        let turn = self.turn_on_the_side(library, core, tick, learned.turn.svcs.len(), &mut mutex);
        self.put_back();
        self.memory.stop_logging_writes();
        turn.is_some_and(|(turn, _)| turn == learned.turn) && mutex == learned.mutex
    }

    /// runs a turn of the loop on the side from the core at tick, returns what it did,
    /// the bytes it wrote (counting those of the turns since the last put_back) and the core it left
    fn turn_on_the_side(
        &mut self,
        library: &recompiled::Library,
        core: &Core,
        tick: u64,
        period: usize,
        mutex: &mut Option<u32>,
    ) -> Option<(Turn, Core)> {
        let mut sandbox = self.spin.sandbox.take().unwrap_or_else(|| Box::new(Sandbox::new()));
        let system: *mut System = self;
        // safe bc this runs from an svc of the library's code, nothing else holds the system meanwhile
        let ran = unsafe { library.run_on_the_side(system, &mut sandbox, core, tick, period, mutex) };
        let written = sandbox.written(&self.memory);
        self.spin.sandbox = Some(sandbox);
        let side = ran?;
        Some((Turn { svcs: side.svcs, cost: side.cost, written }, side.end))
    }

    /// the bytes turns run on the side changed since the last put_back, w/ what they hold
    fn changed_on_the_side(&mut self) -> Vec<(u32, u8)> {
        self.spin.sandbox.as_ref().map_or_else(Vec::new, |sandbox| sandbox.changed(&mut self.memory))
    }

    /// puts back what turns run on the side wrote
    fn put_back(&mut self) {
        if let Some(sandbox) = &mut self.spin.sandbox {
            sandbox.put_back(&mut self.memory);
        }
    }

    /// how many turns of the loop learned to skip from the core at tick
    /// all going the same way, the turn after them too (it runs for real and so writes over everything they'd have
    /// left), and the start of the next b4 horizon so that one runs whole b4 anything looks
    /// found by running that last turn and halving when it goes another way
    fn turns_to_skip(&mut self, library: &recompiled::Library, learned: &mut Learned, core: &Core, tick: u64, horizon: u64) -> u64 {
        let cost = learned.turn.cost;
        let start = |turn: u64| tick + turn * cost;
        let most = (horizon.saturating_sub(tick + SYSTEM_TICK_COST + 1) / cost).saturating_sub(1);
        if most < FEWEST_TURNS {
            return most;
        }
        // where it went another way is past, the turns from here have to be found again
        if learned.other_from.is_some_and(|other| tick >= other) {
            (learned.same_until, learned.other_from) = (0, None);
        }
        // the ones starting by same_until go the same way
        let same = if learned.same_until >= tick { (learned.same_until - tick) / cost + 1 } else { 0 };
        if most < same {
            return most;
        }
        // the first going another way is known, the turns b4 it only as far as same_until
        if learned.other_from.is_some() {
            return same.saturating_sub(1);
        }
        if self.goes_the_same(library, learned, core, start(most)) {
            learned.same_until = start(most);
            return most;
        }
        // the first going another way, roughly, the last known to go the same way b4 it is the one that runs
        let (mut same_turns, mut other) = (same, most);
        while other > same_turns + ROUGHLY {
            let middle = same_turns + (other - same_turns) / 2;
            if self.goes_the_same(library, learned, core, start(middle)) {
                same_turns = middle + 1;
            } else {
                other = middle;
            }
        }
        if same_turns > 0 {
            learned.same_until = learned.same_until.max(start(same_turns - 1));
        }
        learned.other_from = Some(start(other));
        same_turns.saturating_sub(1)
    }

    /// skips turns of the loop learned from tick,
    /// noting each of their svcs in the history and having the preempts they hit pass like their svcs would,
    /// each one at the first svc at or after it
    fn skip_turns(&mut self, learned: &Learned, tick: u64, turns: u64) -> Skipped {
        let turn = &learned.turn;
        let period = turn.svcs.len() as u64;
        let notes = turns * period;
        let noted = notes.min(HISTORY_LENGTH as u64);
        self.history_index = (self.history_index + ((notes - noted) % HISTORY_LENGTH as u64) as usize) % HISTORY_LENGTH;
        for note in notes - noted..notes {
            self.note_plain(turn.svcs[(note % period) as usize].0);
        }
        // where the clock is after each svc
        let after = |at: u64, index: usize| {
            let (_, offset, number) = turn.svcs[index];
            tick + at * turn.cost + offset + if number == 0x28 { SYSTEM_TICK_COST } else { 0 }
        };
        loop {
            let next = self.next_preempt;
            let first = (next.saturating_sub(tick) / turn.cost).saturating_sub(1);
            let found = (first..first + 3).flat_map(|at| (0..period as usize).map(move |index| (at, index))).find(|&(at, index)| after(at, index) >= next);
            match found {
                Some((at, index)) if at < turns => self.next_preempt = after(at, index) + PREEMPT_INTERVAL,
                _ => break,
            }
        }
        let cycles = turns * turn.cost;
        self.spin.skipped += cycles;
        self.spin.turns += turns;
        Skipped { cycles, instructions: turns * learned.instructions, svcs: notes }
    }
}
