//! loops that change nothing
//! a thread alone on the core polling memory nothing else changes b4 the next event just spins till then
//! costs a real core nothing and the interpreter every turn
//! once a turn leaves the core and every byte it wrote as they were (no kernel calls, no events) the next turns would too
//! so the run skips straight to the next event

use zakuro_cpu::Snapshot;

use crate::kernel::thread::ThreadId;
use crate::System;

/// a thread running this long w/o calling the kernel gets watched
const QUIET: u64 = 2048;
/// a turn longer than this isn't a polling loop
const LONGEST_TURN: u32 = 4096;
/// after a watch finds no such loop the next waits this long, doubling each miss up to the max
const RETRY: u64 = 65_536;
const MOST_RETRY: u64 = 2 * 1024 * 1024;
/// while other threads could run, how long till we look again
const SHARED: u64 = 4096;
/// below sp, what calls pushed and popped again, nobody reads it
const STACK_SCRATCH: u32 = 0x800;

pub struct IdleLoops {
    watch: Option<Watch>,
    /// no watch starts b4 this cycle, a step pays one compare
    next_check: u64,
    /// how long the last wait after a fruitless watch was
    retry: u64,
    /// cycles skipped, for reports
    pub skipped: u64,
}

impl IdleLoops {
    /// loops skipped, or never looked for when off
    pub fn new(on: bool) -> IdleLoops {
        IdleLoops { watch: None, next_check: if on { 0 } else { u64::MAX }, retry: 0, skipped: 0 }
    }

    /// does a step have anything to look at, a watch running or one due
    #[inline(always)]
    pub(crate) fn due(&self, now: u64) -> bool {
        now >= self.next_check || self.watch.is_some()
    }
}

/// a loop watched from a point in it, w/ the core as it was there
struct Watch {
    thread: ThreadId,
    pc: u32,
    core: Snapshot,
    steps: u32,
    /// came back to the point once and took the core again there
    /// a loop's first turn starts from what was b4 it, the next ones from what a turn leaves
    again: bool,
}

impl System {
    /// where the run skips to when a lone thread spins in a loop that changes nothing and has nothing to do till then
    pub(crate) fn idle_until(&mut self, deadline: Option<u64>) -> Option<u64> {
        let thread = self.kernel.current_thread?;
        let now = self.cpu.cycles;
        let Some(watch) = &mut self.idle.watch else {
            if self.kernel.others_runnable() {
                self.idle.next_check = now + SHARED;
            } else {
                self.idle.watch = Some(Watch { thread, pc: self.cpu.regs[15], core: self.cpu.snapshot(), steps: 0, again: false });
                self.memory.log_writes();
            }
            return None;
        };
        watch.steps += 1;
        if watch.thread != thread || watch.steps > LONGEST_TURN {
            self.give_up_watching();
            return None;
        }
        if self.cpu.regs[15] != watch.pc {
            return None;
        }
        let core = self.cpu.snapshot();
        if core != watch.core {
            if !watch.again {
                watch.again = true;
                watch.core = core;
                self.memory.log_writes();
            }
            return None;
        }
        let sp = self.cpu.regs[13];
        if !self.memory.writes_undone(sp.saturating_sub(STACK_SCRATCH)..sp) {
            return None;
        }
        // a turn that changed nothing, the next ones wouldn't either
        self.idle.watch = None;
        self.memory.stop_logging_writes();
        self.idle.retry = 0;
        self.idle.next_check = now;
        if self.kernel.others_runnable() {
            return None;
        }
        let mut until = self.next_frame_boundary.min(self.next_audio_frame);
        if let Some(deadline) = deadline {
            until = until.min(deadline);
        }
        if let Some(event) = self.kernel.next_event() {
            until = until.min(event);
        }
        if until <= now {
            self.kernel.reschedule_pending = true;
            return None;
        }
        self.idle.skipped += until - now;
        Some(until)
    }

    /// a watch that found nothing, the next one waits longer
    fn give_up_watching(&mut self) {
        self.idle.watch = None;
        self.memory.stop_logging_writes();
        self.idle.retry = (self.idle.retry * 2).clamp(RETRY, MOST_RETRY);
        self.idle.next_check = self.cpu.cycles + self.idle.retry;
    }

    /// smth besides the loop happened (kernel call, event, recompiled code, another thread getting the core),
    /// a watched turn can't span that
    /// busy when the running thread did it
    pub(crate) fn idle_interrupted(&mut self, busy: bool) {
        if self.idle.watch.take().is_some() {
            self.memory.stop_logging_writes();
        }
        if busy {
            self.idle.next_check = self.idle.next_check.max(self.cpu.cycles + QUIET);
        }
    }
}
