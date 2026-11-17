//! the emulated console, cpu, memory, HLE kernel and services, GPU, and the loop driving them all

pub mod camera;
pub mod cheats;
pub mod enhancements;
pub mod cro;
pub mod download_play;
pub mod hints;
pub mod idle;
pub mod spin;
pub mod kernel;
pub mod loader;
pub mod memory;
pub mod microphone;
pub mod mods;
pub mod replay;
pub mod room;
pub mod recompiled;
pub mod services;
pub mod state;

use std::collections::{BTreeMap, BTreeSet};

use zakuro_common::memory_map::*;
use zakuro_common::ConsoleModel;
use zakuro_cpu::{Cpu, Exit};
use zakuro_fs::Title;
use zakuro_gpu::{Gpu, GpuMemory, Renderer, SoftwareRenderer};

use kernel::thread::{ThreadId, WaitResult, WaitSyscall, THREAD_EXIT_MAGIC};
use kernel::Kernel;
use memory::{Memory, MemoryState, Permission};
use services::hid::InputState;
use services::ServiceState;

/// console settings the frontend can change
#[derive(Debug, Clone)]
pub struct Config {
    pub model: ConsoleModel,
    pub new3ds: bool,
    /// 0 is japan, 1 usa, 2 europe
    pub region: u8,
    pub language: u8,
    pub slider_3d: f32,
    /// a lib 3dsrecomp built for the title, or a dir holding one named after its title id
    pub recompiled: Option<std::path::PathBuf>,
    /// recompiled code linked into the program, beats a lib
    pub linked: Option<recompiled::Linked>,
    /// w/ no lib given, run whatever `3dsrecomp build` installed for the title, if anything
    pub find_recompiled: bool,
    /// where saves and dumped system files live, the cwd when none
    pub data_dir: Option<std::path::PathBuf>,
    /// draw on the host GPU thru Vulkan when there's one that can
    pub hardware_renderer: bool,
    /// swap the title's textures for the texture pack in its mods folder, needs the host GPU
    pub texture_packs: bool,
    /// the title's update CIA, whose code and data it runs
    pub update: Option<std::path::PathBuf>,
    /// the title's DLC CIAs
    pub dlc: Vec<std::path::PathBuf>,
    /// enhancements turned on, by name, per title id
    pub enhancements: std::collections::BTreeMap<u64, Vec<String>>,
    /// resolution multiplier for the host GPU
    pub resolution: u32,
    /// the Vulkan presenter's device, to draw on and show the screens straight from when it can
    pub device: Option<std::sync::Arc<zakuro_gpu::SharedDevice>>,
    /// console clock start in millis since 1900, host time when none, a fixed one makes runs repeat exactly
    pub clock: Option<u64>,
    /// draw a picture between every two for titles drawing every other refresh (see System::pace)
    /// when the host GPU draws the 3D and shows the screens straight from its device
    pub interpolate: bool,
    /// the owner's name, up to 10 chars, titles show it and local wireless tells others
    pub username: String,
    /// the number telling this console apart over local wireless, config block 0x00090000
    pub console_id: u64,
    /// the game's Download Play child decrypted, a host sends it to guests, the card's copy is encrypted
    pub dlp_child: Option<std::path::PathBuf>,
    /// what a Download Play child runs w/, host addr and network passphrase, APT:GetWirelessRebootInfo's answer
    pub wireless_reboot: Option<[u8; 16]>,
    /// the interpreter skips to the next event past loops that change nothing, see idle
    pub skip_idle_loops: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            model: ConsoleModel::Old3ds,
            new3ds: false,
            region: services::cfg::REGION_USA,
            language: services::cfg::LANGUAGE_ENGLISH,
            slider_3d: 0.0,
            recompiled: None,
            linked: None,
            find_recompiled: false,
            data_dir: None,
            hardware_renderer: false,
            texture_packs: true,
            update: None,
            dlc: Vec::new(),
            enhancements: std::collections::BTreeMap::new(),
            resolution: 1,
            device: None,
            clock: None,
            interpolate: false,
            username: "Zakuro".into(),
            console_id: 0x5A4B_0000_0000_0001,
            dlp_child: None,
            wireless_reboot: None,
            skip_idle_loops: true,
        }
    }
}

/// how a frame of emulation ended
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameOutcome {
    /// a full frame's cycles ran
    Completed,
    /// the title called svcExitProcess
    Exited,
    /// the title called svcBreak or hit an instr we don't have
    Faulted,
}

/// a screen's picture as RGBA plus width and height
pub type Screen = (std::sync::Arc<Vec<u8>>, u32, u32);

/// frame interpolation's refresh count, see System::pace
#[derive(Default)]
struct Pacing {
    refreshes: u64,
    /// refresh the top screen last got a new picture at, and that picture
    new_at: u64,
    last: Option<zakuro_gpu::ScreenRef>,
    /// how the draws of the frame b4 matched the one b4 that
    matches: Option<zakuro_gpu::Matches>,
    /// the screens show the in between pictures this refresh, the title's own next refresh
    between: bool,
    /// refreshes that showed in between pictures so far
    shown: u64,
}

pub struct System {
    pub cpu: Cpu,
    pub memory: Memory,
    pub kernel: Kernel,
    pub services: ServiceState,
    pub gpu: Gpu,
    pub renderer: Box<dyn Renderer>,
    pub title: Option<Title>,
    /// the title's DLC
    pub dlc: Vec<zakuro_fs::Dlc>,
    /// the title's cheats, from its file in the data folder
    pub cheats: Vec<cheats::Cheat>,
    pub config: Config,

    pub exited: bool,
    pub broke: bool,
    /// everything the title wrote w/ svcOutputDebugString
    pub debug_output: String,
    pub unimplemented_svcs: BTreeSet<u32>,
    pub services_seen: BTreeSet<String>,
    pub lcd_force_black: bool,
    /// the scaled picture each screen had at the last present, it goes up at the next once the GPU had a frame to draw it
    showing: [Option<zakuro_gpu::ScreenRef>; 3],
    /// frame interpolation's refresh count
    pacing: Pacing,
    /// the dynamic module loader's state
    pub cro: cro::CroManager,
    /// what the cameras see, from the frontend
    pub camera: Option<Box<dyn camera::Camera>>,
    /// what the mic hears, from the frontend
    pub microphone: Option<Box<dyn microphone::Microphone>>,
    /// the room local wireless goes thru, none w/ wireless off
    pub room: Option<room::Link>,
    /// errors the title reported thru err:f, newest last
    pub fatal_errors: Vec<String>,
    /// undefined instrs we've seen, keeps the log readable
    undefined_seen: BTreeMap<u32, u32>,
    /// branches into unmapped memory, keyed by target
    pub prefetch_aborts: BTreeMap<u32, u32>,
    /// the last few instr addrs so a fault can say how it got there
    history: [u32; HISTORY_LENGTH],
    history_index: usize,
    /// frames done since boot
    pub frames: u64,
    /// the console clock at boot, millis since 1900
    pub(crate) boot_clock: u64,
    /// cycle the next end of frame work is due at
    next_frame_boundary: u64,
    /// cycle the DSP finishes its next audio frame at
    next_audio_frame: u64,
    /// cycle the scheduler gets forced to run next
    next_preempt: u64,

    /// sampling profiler, how often each thread was caught at each PC
    pub profile: Option<BTreeMap<(ThreadId, u32), u64>>,
    /// steps since the profiler last looked
    profile_skip: u32,

    /// loops that change nothing, the interpreter skips them
    pub idle: idle::IdleLoops,
    /// loops in recompiled code polling the clock, see spin
    pub spin: spin::SpinLoops,

    /// AOT recompiled code, runs instead of the interpreter wherever it has smth
    pub recompiled: Option<recompiled::Library>,
    /// where the title ran in the interpreter despite the lib
    pub hints: Option<hints::Hints>,
    /// instrs the interpreter ran for lack of recompiled code vs ones recompiled code ran, to see how much the lib covers
    pub interpreted_instructions: u64,
    pub recompiled_instructions: u64,
}

/// cycles per frame of the screens, they refresh at 268111856 / 4481136 so ~59.83 Hz not 60
/// a title keeping time by frames drifts ahead of its sound otherwise, rhythm game notes slide off the music over a song
pub const CYCLES_PER_FRAME: u64 = 4_481_136;

/// cycles per audio frame, 160 samples of exactly 8192 cycles each (the DSP's 32728 Hz is the cpu clock / 8192)
pub const CYCLES_PER_AUDIO_FRAME: u64 = 160 * 8192;
/// samples the DSP plays per second, one every 8192 cycles
pub const AUDIO_SAMPLE_RATE: f64 = kernel::thread::CPU_CLOCK_HZ as f64 / 8192.0;

/// how often the scheduler gets forced to run even if no thread yields
pub const PREEMPT_INTERVAL: u64 = 8192;

/// instr addrs kept for fault reports
const HISTORY_LENGTH: usize = 64;
// the svcs recompiled code answers itself go in the history from its trail
const _: () = assert!(HISTORY_LENGTH <= recomp_abi::TRAIL);

/// what one [System::step] call did
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepOutcome {
    Ran,
    Exited,
    Faulted,
}

impl System {
    pub fn new(config: Config) -> System {
        let boot_clock = config.clock.unwrap_or_else(memory::config::host_clock);
        let idle = idle::IdleLoops::new(config.skip_idle_loops);
        let spin = spin::SpinLoops::new(config.skip_idle_loops);
        let app_bytes = 64 * 1024 * 1024;
        let mut services = ServiceState::default();
        if let Some(dir) = &config.data_dir {
            services.fs.user_dir = dir.join("user");
        }
        System {
            cpu: Cpu::new(),
            memory: Memory::new(config.new3ds, app_bytes),
            kernel: Kernel::new(0, memory::MemoryRegion::Application, linear_heap_base(0)),
            services,
            gpu: Gpu::new(),
            renderer: Box::new(SoftwareRenderer::default()),
            title: None,
            dlc: Vec::new(),
            cheats: Vec::new(),
            config,
            exited: false,
            broke: false,
            debug_output: String::new(),
            unimplemented_svcs: BTreeSet::new(),
            services_seen: BTreeSet::new(),
            lcd_force_black: false,
            showing: [None, None, None],
            pacing: Pacing::default(),
            cro: cro::CroManager::default(),
            camera: None,
            microphone: None,
            room: None,
            fatal_errors: Vec::new(),
            undefined_seen: BTreeMap::new(),
            prefetch_aborts: BTreeMap::new(),
            history: [0; HISTORY_LENGTH],
            history_index: 0,
            frames: 0,
            boot_clock,
            next_frame_boundary: CYCLES_PER_FRAME,
            next_audio_frame: CYCLES_PER_AUDIO_FRAME,
            next_preempt: PREEMPT_INTERVAL,
            profile: None,
            profile_skip: 0,
            idle,
            spin,
            recompiled: None,
            hints: None,
            interpreted_instructions: 0,
            recompiled_instructions: 0,
        }
    }

    // process setup

    /// maps the page w/ a thread's TLS block if it isn't yet, and clears the block (an ended thread may have used it)
    pub fn map_tls_page(&mut self, thread: ThreadId) {
        let tls = self.kernel.thread(thread).tls;
        let page = tls & !PAGE_MASK;
        if self.memory.mapping_at(page).is_some() {
            self.memory.write_bytes(tls, &[0; TLS_ENTRY_SIZE as usize]);
            return;
        }
        let Some(block) = self
            .memory
            .phys
            .allocate(memory::MemoryRegion::Base, PAGE_SIZE)
        else {
            log::error!("out of memory allocating a TLS page");
            return;
        };
        self.memory.map(
            page,
            block.addr,
            PAGE_SIZE,
            Permission::RW,
            MemoryState::Locked,
        );
        // TLS must start zeroed, the IPC command buffer lives in it
        self.memory.zero_physical(block.addr, PAGE_SIZE);
    }

    // the run loop

    /// runs one frame of emulation
    pub fn run_frame(&mut self) -> FrameOutcome {
        // HID samples its inputs on its own, pressed or not
        let input = self.services.hid.input;
        self.idle_interrupted(false);
        services::hid::update(self, input);
        cheats::run(self);
        // run up to the boundary step() would end the frame at and end it here once,
        // not again when the next frame's first step sees it passed
        let deadline = self.next_frame_boundary;
        while self.cpu.cycles < deadline {
            match self.step(Some(deadline)) {
                StepOutcome::Ran => {}
                StepOutcome::Exited => return FrameOutcome::Exited,
                StepOutcome::Faulted => return FrameOutcome::Faulted,
            }
        }
        self.next_frame_boundary = self.cpu.cycles + CYCLES_PER_FRAME;
        self.end_frame();
        FrameOutcome::Completed
    }

    /// moves the machine one instr forward, doing any due scheduling first
    pub fn step(&mut self, deadline: Option<u64>) -> StepOutcome {
        if self.exited {
            return StepOutcome::Exited;
        }
        if self.broke {
            return StepOutcome::Faulted;
        }

        // fire the end of frame work on time even when a tool is stepping, else a title waiting on vblank never wakes
        if self.cpu.cycles >= self.next_frame_boundary {
            self.next_frame_boundary = self.cpu.cycles + CYCLES_PER_FRAME;
            self.end_frame();
        }
        if self.cpu.cycles >= self.next_audio_frame {
            self.next_audio_frame = self.cpu.cycles + CYCLES_PER_AUDIO_FRAME;
            self.audio_frame();
        }

        if self.kernel.reschedule_pending || self.kernel.current_thread.is_none() {
            self.reschedule();
        }

        if self.kernel.current_thread.is_none() {
            // everything's blocked
            let mut limit = deadline.unwrap_or(self.cpu.cycles + CYCLES_PER_FRAME);
            // an audio thread waits on the DSP's interrupt, which only fires at an audio frame boundary, skipping one would lose it
            if self.services.dsp.running {
                limit = limit.min(self.next_audio_frame);
            }
            match self.kernel.next_event() {
                Some(tick) if tick > self.cpu.cycles => self.cpu.cycles = tick.min(limit),
                _ => self.cpu.cycles = limit,
            }
            self.kernel.reschedule_pending = true;
            return StepOutcome::Ran;
        }

        // a thread whose entry returned branches here
        if self.cpu.regs[15] == THREAD_EXIT_MAGIC {
            if let Some(id) = self.kernel.current_thread {
                self.kernel.end_thread(id);
            }
            return StepOutcome::Ran;
        }

        // a branch thru an uninitialized func ptr lands on the null page, whose zero words decode as harmless no ops
        let pc = self.cpu.regs[15];
        if !self.memory.is_executable(pc) {
            let count = self.prefetch_aborts.entry(pc).or_insert(0);
            *count += 1;
            if *count == 1 {
                let thread = self
                    .kernel
                    .current()
                    .map_or("?".to_owned(), |t| t.name.clone());
                log::error!(
                    "prefetch abort: {thread} branched to unmapped 0x{pc:08X} from 0x{:08X}",
                    self.cpu.regs[14]
                );
                log::error!("{:?}", self.cpu);
                log::error!("the instructions leading here were:");
                let recent = self.recent_instructions();
                for &entry in recent.iter().rev().take(8).rev() {
                    let address = entry & !1;
                    let mut bytes = [0u8; 8];
                    self.memory.read_bytes(address & !3, &mut bytes);
                    log::error!(
                        "  0x{address:08X} {} [{:02X?}]",
                        if entry & 1 != 0 { "T" } else { "A" },
                        bytes
                    );
                }
                self.fatal_errors.push(format!(
                    "branch to unmapped 0x{pc:08X} (lr 0x{:08X})",
                    self.cpu.regs[14]
                ));
            }
            // stop the thread not the machine, the rest of the title may still get somewhere, the report says what happened
            if let Some(id) = self.kernel.current_thread {
                self.kernel.end_thread(id);
            }
            return StepOutcome::Ran;
        }

        self.note(pc);

        let exit = match self.run_recompiled(deadline) {
            Some(exit) => {
                self.ran_recompiled();
                exit
            }
            None => {
                // a thread polling in a loop that changes nothing waits for the next event, like on a console spinning freely
                if let Some(until) = self.idle.due(self.cpu.cycles).then(|| self.idle_until(deadline)).flatten() {
                    self.cpu.cycles = until;
                    self.kernel.reschedule_pending = true;
                    return StepOutcome::Ran;
                }
                self.interpreted_instructions += 1;
                // past the step's start when the code ran svcs in place
                let (pc, thumb) = (self.cpu.regs[15], self.cpu.cpsr.thumb);
                if let (Some(hints), Some(library)) = (&mut self.hints, &self.recompiled) {
                    if !library.has_code(pc | thumb as u32) {
                        hints.interpreted(pc, thumb);
                    }
                }
                self.sync_gpu_for_cpu();
                let exit = self.cpu.step(&mut self.memory);
                self.memory.clear_gpu_sync();
                exit
            }
        };
        if exit.is_some() {
            self.idle_interrupted(true);
        }
        match exit {
            None => {}
            Some(Exit::Supervisor(number)) => kernel::svc::dispatch(self, number),
            Some(Exit::Undefined { pc, opcode }) => {
                let count = self.undefined_seen.entry(pc).or_insert(0);
                *count += 1;
                if *count == 1 {
                    log::error!("undefined instruction 0x{opcode:08X} at 0x{pc:08X}");
                    log::error!("{:?}", self.cpu);
                }
                // skip it and keep going, one bad decode shouldn't end the session while the cpu is still being filled in
                let step = if self.cpu.cpsr.thumb { 2 } else { 4 };
                self.cpu.regs[15] = pc.wrapping_add(step);
            }
            Some(Exit::Breakpoint { pc, imm }) => {
                log::warn!("bkpt #{imm} at 0x{pc:08X}");
                let step = if self.cpu.cpsr.thumb { 2 } else { 4 };
                self.cpu.regs[15] = pc.wrapping_add(step);
            }
            Some(Exit::Halted) => {
                self.cpu.resume();
                self.kernel.reschedule_pending = true;
            }
            Some(Exit::Timeout) => {}
        }
        let _ = pc;

        self.preempt();

        if self.exited {
            StepOutcome::Exited
        } else if self.broke {
            StepOutcome::Faulted
        } else {
            StepOutcome::Ran
        }
    }

    /// runs recompiled code from the pc to the next thing the scheduler must look at, when the lib has code there
    /// its return stands in for what one interpreted step would give
    fn run_recompiled(&mut self, deadline: Option<u64>) -> Option<Option<Exit>> {
        let library = self.recompiled.as_ref()?;
        if !library.has_code(self.cpu.regs[15] | self.cpu.cpsr.thumb as u32) {
            return None;
        }
        let Some((budget, event, limit)) = self.recompiled_budget(deadline) else {
            return Some(None);
        };
        // the lib stays out of the system while its code runs, so an svc the code runs in place gets the system to itself
        let library = self.recompiled.take()?;
        self.sync_gpu_for_cpu();
        let (ran, stop, event) = library.run(self, budget, deadline, event, limit);
        self.memory.clear_gpu_sync();
        self.recompiled = Some(library);
        self.recompiled_instructions += ran;
        if event.is_some_and(|tick| self.cpu.cycles >= tick) {
            self.kernel.reschedule_pending = true;
        }
        if (ran == 0 && matches!(stop, recompiled::Stop::Left)) || matches!(stop, recompiled::Stop::Declined) {
            // not enough budget for a whole block, at the start or after an in place svc, whose step this is now
            if let Some(hints) = &mut self.hints {
                hints.library_declined();
            }
            return None;
        }
        Some(match stop {
            recompiled::Stop::Svc(number) => Some(Exit::Supervisor(number)),
            recompiled::Stop::Exit(exit) => Some(exit),
            recompiled::Stop::Left | recompiled::Stop::Declined => None,
        })
    }

    /// how many cycles recompiled code can run from here b4 the scheduler must look
    /// the tick of the next thing a thread waits for, and the cycle the budget runs out at (only the kernel and the boundaries move it)
    /// none when that's due already, so the scheduler looks first
    fn recompiled_budget(&mut self, deadline: Option<u64>) -> Option<(u64, Option<u64>, u64)> {
        let mut limit = self.next_frame_boundary.min(self.next_audio_frame);
        // a due thread or timer wants the scheduler at its tick, a run that reached it has it look first
        let event = self.kernel.next_event();
        if let Some(tick) = event {
            if tick <= self.cpu.cycles {
                self.kernel.reschedule_pending = true;
                return None;
            }
            limit = limit.min(tick);
        }
        // a stop for the scheduler unwinds every guest call the code is in, and the host reenters them one by one
        // only worth it when another thread can run
        if self.kernel.others_runnable() {
            limit = limit.min(self.next_preempt);
        }
        if let Some(deadline) = deadline {
            limit = limit.min(deadline);
        }
        Some((limit.saturating_sub(self.cpu.cycles).max(1), event, limit))
    }

    /// runs an svc recompiled code made from inside the code and lets the code go on after it for as long as the result says
    /// when the next step would just run that code on anyway
    /// does what the rest of this step and the start of the next would, same order,
    /// and gives none wherever they'd do more (end a frame, switch threads)
    /// the code stops for those and leaves the rest to the steps
    /// the code then charges what follows to the budget as its funcs cut it, a stop reenters thru the func an entry names
    pub(crate) fn svc_in_place(
        &mut self,
        number: u32,
        event: Option<u64>,
        deadline: Option<u64>,
        library: &recompiled::Library,
    ) -> Option<(u64, Option<u64>, u64)> {
        let thread = self.kernel.current_thread;
        let resume = (self.cpu.regs[15], self.cpu.cpsr.thumb);
        // what the step does once the code stopped at the svc
        self.memory.clear_gpu_sync();
        if event.is_some_and(|tick| self.cpu.cycles >= tick) {
            self.kernel.reschedule_pending = true;
        }
        self.ran_recompiled();
        self.idle_interrupted(true);
        kernel::svc::dispatch(self, number);
        self.preempt();
        // plus what the next one does b4 running code
        let cycles = self.cpu.cycles;
        if self.exited || self.broke || deadline.is_some_and(|deadline| cycles >= deadline) {
            return None;
        }
        if cycles >= self.next_frame_boundary || cycles >= self.next_audio_frame {
            return None;
        }
        if self.kernel.reschedule_pending || self.kernel.current_thread.is_none() {
            self.reschedule();
        }
        // same thread, still after the svc, code the lib has
        let (pc, thumb) = (self.cpu.regs[15], self.cpu.cpsr.thumb);
        if self.kernel.current_thread != thread || (pc, thumb) != resume || !self.memory.is_executable(pc) || !library.has_code(pc | thumb as u32)
        {
            return None;
        }
        self.note(pc);
        let budget = self.recompiled_budget(deadline)?;
        self.sync_gpu_for_cpu();
        Some(budget)
    }

    /// GetSystemTick from recompiled code, where svc_in_place would only read the clock (see plain_in_place)
    /// limit is what recompiled_budget gave last, still valid since only the kernel moves it and only the code ran since
    /// does what svc_in_place would in the same order, regs left to the caller, and returns the budget the code goes on w/
    /// none (having done nothing) for svc_in_place to run it
    pub(crate) fn tick_in_place(
        &mut self,
        cycles: u64,
        at: u32,
        event: Option<u64>,
        deadline: Option<u64>,
        limit: u64,
        library: &recompiled::Library,
    ) -> Option<u64> {
        let after = cycles + kernel::svc::SYSTEM_TICK_COST;
        if !self.plain_in_place(after, at, event, deadline, library) {
            return None;
        }
        self.cpu.cycles = cycles;
        self.ran_recompiled();
        self.idle_interrupted(true);
        self.cpu.cycles = after;
        self.preempt_in_place(after);
        self.note_plain(at);
        Some(limit.saturating_sub(after).max(1))
    }

    /// what tick_in_place did for count ticks recompiled code read itself at at, back to back,
    /// the last at tick last, see recomp_abi::Context
    pub(crate) fn ticks_in_place(&mut self, at: u32, count: u32, last: u64) {
        self.cpu.cycles = last;
        self.ran_recompiled();
        // the last one moves the next idle check furthest
        self.idle_interrupted(true);
        self.cpu.cycles = last + kernel::svc::SYSTEM_TICK_COST;
        // the history holds at most the last HISTORY_LENGTH of them
        let count = count as usize;
        let noted = count.min(HISTORY_LENGTH);
        self.history_index = (self.history_index + count - noted) % HISTORY_LENGTH;
        for _ in 0..noted {
            self.note_plain(at);
        }
    }

    /// what tick_in_place, wait_in_place and release_in_place did for the svcs recompiled code answered itself
    /// done of them, the last at tick last, clock left at after, addrs after the last ones in trail, see recomp_abi::Context
    /// what they did to the kernel is the caller's problem
    pub(crate) fn answered_in_place(&mut self, trail: &[u32; recomp_abi::TRAIL], done: u32, last: u64, after: u64) {
        self.cpu.cycles = last;
        self.ran_recompiled();
        // the last one moves the next idle check furthest
        self.idle_interrupted(true);
        self.cpu.cycles = after;
        // the history holds at most the last HISTORY_LENGTH of them, the trail has those
        let done = done as usize;
        let noted = done.min(HISTORY_LENGTH);
        self.history_index = (self.history_index + done - noted) % HISTORY_LENGTH;
        for at in done - noted..done {
            self.note_plain(trail[at % recomp_abi::TRAIL]);
        }
    }

    /// what a GetSystemTick recompiled code reads itself must end b4 (see recomp_abi::Context)
    /// the first thing plain_in_place makes tick_in_place leave for the step, or the preempt (passed itself when quiet)
    /// none when plain_in_place fails whatever the tick, only the step and svcs change that
    pub(crate) fn tick_due(&self, event: Option<u64>, deadline: Option<u64>) -> Option<u64> {
        let plain = !self.exited
            && !self.broke
            && !self.kernel.reschedule_pending
            && self.kernel.current_thread.is_some()
            && self.profile.is_none()
            && !log::log_enabled!(log::Level::Trace);
        let first = self.next_preempt.min(self.next_frame_boundary).min(self.next_audio_frame);
        plain.then(|| [event, deadline].into_iter().flatten().fold(first, u64::min))
    }

    /// WaitSynchronizationN from recompiled code (regs = its r0 to r4) the tick_in_place way
    /// when the wait's satisfied right away and changes no thread, which keeps limit
    /// returns the budget and the index the wait gives
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn wait_in_place(
        &mut self,
        cycles: u64,
        at: u32,
        regs: [u32; 5],
        event: Option<u64>,
        deadline: Option<u64>,
        limit: u64,
        library: &recompiled::Library,
    ) -> Option<(u64, u32)> {
        let (pointer, count, wait_all) = (regs[1], regs[2] as usize, regs[3] != 0);
        if count == 0 || count > kernel::WAIT_AT_ONCE || !self.plain_in_place(cycles, at, event, deadline, library) {
            return None;
        }
        let mut handles = [0; kernel::WAIT_AT_ONCE];
        for (i, handle) in handles[..count].iter_mut().enumerate() {
            *handle = self.memory.plain_read32(pointer.wrapping_add(4 * i as u32))?;
        }
        let (objects, index) = self.kernel.satisfied_now(&handles[..count], wait_all, cycles)?;
        let thread = self.kernel.current_thread?;
        self.cpu.cycles = cycles;
        self.ran_recompiled();
        self.idle_interrupted(true);
        if wait_all {
            for &object in &objects[..count] {
                self.kernel.acquire_public(object, thread);
            }
        } else {
            self.kernel.acquire_public(objects[index as usize], thread);
        }
        self.preempt_in_place(cycles);
        self.note_plain(at);
        Some((limit.saturating_sub(cycles).max(1), index))
    }

    /// ReleaseMutex on handle (its r0) from recompiled code the tick_in_place way
    /// when the mutex stays held, or freeing it has a pass due that changes nothing (Kernel::quiet_release), which keeps limit
    /// returns the budget the code goes on w/
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn release_in_place(
        &mut self,
        cycles: u64,
        at: u32,
        handle: u32,
        event: Option<u64>,
        deadline: Option<u64>,
        limit: u64,
        library: &recompiled::Library,
    ) -> Option<u64> {
        if !self.plain_in_place(cycles, at, event, deadline, library) {
            return None;
        }
        let (object, frees) = self.kernel.quiet_release(handle, cycles)?;
        self.cpu.cycles = cycles;
        self.ran_recompiled();
        self.idle_interrupted(true);
        if let Some(kernel::object::KObject::Mutex(mutex)) = self.kernel.objects.get_mut(object) {
            mutex.lock_count = mutex.lock_count.saturating_sub(1);
            if frees {
                mutex.owner = None;
                // what the pass says, current thread kept
                self.kernel.all_blocked = false;
            }
        }
        self.preempt_in_place(cycles);
        self.note_plain(at);
        Some(limit.saturating_sub(cycles).max(1))
    }

    /// would svc_in_place (for an svc changing no thread and leaving the clock at after) do nothing around it
    /// but note at and pass the preempt quietly (see preempt_in_place)
    /// event, deadline and the frame and audio boundaries all ahead of after, the preempt too or its pass a no op
    /// nothing pending for the scheduler, and code at at (pc w/ bit 0 for Thumb)
    fn plain_in_place(&self, after: u64, at: u32, event: Option<u64>, deadline: Option<u64>, library: &recompiled::Library) -> bool {
        let ahead = |tick: u64| after < tick;
        !self.exited
            && !self.broke
            && !self.kernel.reschedule_pending
            && self.kernel.current_thread.is_some()
            && self.profile.is_none()
            && !log::log_enabled!(log::Level::Trace)
            && event.is_none_or(ahead)
            && deadline.is_none_or(ahead)
            && (ahead(self.next_preempt) || self.kernel.quiet_pass(after))
            && ahead(self.next_frame_boundary)
            && ahead(self.next_audio_frame)
            && self.memory.is_executable(at & !1)
            && library.has_code(at)
    }

    /// what preempt and its due pass do after an svc leaving the clock at after, when plain_in_place found that pass quiet
    /// next preempt from after, all_blocked as the pass leaves it
    /// a lone thread spinning on the clock and a mutex hits one every 8192 cycles, no longer a whole pass each
    /// none while the preempt is ahead
    fn preempt_in_place(&mut self, after: u64) {
        if after >= self.next_preempt {
            self.next_preempt = after + PREEMPT_INTERVAL;
            self.kernel.all_blocked = false;
        }
    }

    /// what note does w/o a profile, for at (pc w/ bit 0 for Thumb)
    fn note_plain(&mut self, at: u32) {
        self.history[self.history_index] = at;
        self.history_index = (self.history_index + 1) % HISTORY_LENGTH;
    }

    /// the scheduling pass a step makes first when due
    fn reschedule(&mut self) {
        self.kernel.reschedule_pending = false;
        let tick = self.cpu.cycles;
        if self.kernel.schedule(&mut self.cpu, tick) {
            self.idle_interrupted(true);
        }
        // a thread released here that keeps the core gets its result too b4 it runs on, not over whatever it does next
        self.apply_wait_result();
    }

    /// notes where a step runs the guest from, for fault reports and the profiler
    fn note(&mut self, pc: u32) {
        self.history[self.history_index] = pc | self.cpu.cpsr.thumb as u32;
        self.history_index = (self.history_index + 1) % HISTORY_LENGTH;
        if self.profile.is_some() {
            self.sample();
        }
    }

    /// what a step does after recompiled code ran
    fn ran_recompiled(&mut self) {
        // its writes skip the log a watch keeps
        self.idle_interrupted(false);
        if let Some(hints) = &mut self.hints {
            hints.library_ran();
        }
    }

    /// gives the scheduler a regular chance, so a thread that never syscalls can't hog the core
    fn preempt(&mut self) {
        if self.cpu.cycles >= self.next_preempt {
            self.next_preempt = self.cpu.cycles + PREEMPT_INTERVAL;
            self.kernel.reschedule_pending = true;
        }
    }

    /// makes a cpu read of where the GPU drew pull that down first, till clear_gpu_sync
    fn sync_gpu_for_cpu(&mut self) {
        let gpu = &mut self.gpu as *mut Gpu as *mut ();
        let linear_base = self.kernel.linear_base;
        // safe bc the GPU stays put and only the cpu's reads and writes follow the ptr till the sync gets cleared
        unsafe { self.memory.set_gpu_sync(memory::GpuSync { gpu, linear_base, sync: sync_for_cpu, sync_write: sync_for_cpu_write }) };
    }

    /// the most recently run instr addrs, oldest first
    pub fn recent_instructions(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(HISTORY_LENGTH);
        for offset in 0..HISTORY_LENGTH {
            let entry = self.history[(self.history_index + offset) % HISTORY_LENGTH];
            if entry != 0 {
                out.push(entry);
            }
        }
        out
    }

    /// starts collecting PC samples
    pub fn enable_profiler(&mut self) {
        self.profile = Some(BTreeMap::new());
    }

    /// the hottest sampled spots, most frequent first
    pub fn hot_spots(&self, count: usize) -> Vec<(String, u32, u64)> {
        let Some(profile) = &self.profile else {
            return Vec::new();
        };
        let mut entries: Vec<(String, u32, u64)> =
            profile.iter().map(|(&(thread, pc), &hits)| (self.thread_name(thread), pc, hits)).collect();
        entries.sort_by_key(|&(_, _, hits)| std::cmp::Reverse(hits));
        entries.truncate(count);
        entries
    }

    /// how many samples caught each thread running, most first
    pub fn profiled_threads(&self) -> Vec<(String, u64)> {
        let Some(profile) = &self.profile else {
            return Vec::new();
        };
        let mut threads: BTreeMap<ThreadId, u64> = BTreeMap::new();
        for (&(thread, _), &hits) in profile {
            *threads.entry(thread).or_insert(0) += hits;
        }
        let mut threads: Vec<(String, u64)> = threads.into_iter().map(|(thread, hits)| (self.thread_name(thread), hits)).collect();
        threads.sort_by_key(|&(_, hits)| std::cmp::Reverse(hits));
        threads
    }

    fn thread_name(&self, id: ThreadId) -> String {
        self.kernel.threads.get(id as usize).map_or_else(|| format!("thread {id}"), |thread| thread.name.clone())
    }

    fn sample(&mut self) {
        // one step in so many, a prime so it doesn't beat in sync w/ a loop, keeps the profiler from slowing the run
        self.profile_skip = self.profile_skip.wrapping_add(1);
        if !self.profile_skip.is_multiple_of(61) {
            return;
        }
        let Some(id) = self.kernel.current_thread else {
            return;
        };
        let pc = self.cpu.regs[15];
        if let Some(profile) = &mut self.profile {
            *profile.entry((id, pc)).or_insert(0) += 1;
        }
    }

    /// writes back the result regs of a syscall that had blocked
    fn apply_wait_result(&mut self) {
        let Some(id) = self.kernel.current_thread else {
            return;
        };
        let thread = self.kernel.thread_mut(id);
        let (Some(result), Some(syscall)) = (thread.wait_result.take(), thread.wait_syscall.take())
        else {
            return;
        };

        match (syscall, result) {
            (WaitSyscall::WaitSynchronization1, WaitResult::Signaled(_)) => {
                self.cpu.regs[0] = 0;
            }
            (WaitSyscall::WaitSynchronizationN, WaitResult::Signaled(index)) => {
                self.cpu.regs[0] = 0;
                self.cpu.regs[1] = index as u32;
            }
            (WaitSyscall::ArbitrateAddress, WaitResult::Signaled(_)) => {
                self.cpu.regs[0] = 0;
            }
            (_, WaitResult::TimedOut) => {
                self.cpu.regs[0] = zakuro_common::result::errors::TIMEOUT.0;
            }
            (WaitSyscall::SleepThread, _) => {
                self.cpu.regs[0] = 0;
            }
        }
    }

    /// everything between frames, vblank, input, clock
    fn end_frame(&mut self) {
        self.idle_interrupted(false);
        self.frames += 1;
        if self.frames.is_multiple_of(hints::SAVE_EVERY) {
            if let Some(hints) = &mut self.hints {
                hints.save();
            }
        }

        // refresh the kernel's shared page so the guest's clock moves
        let tick = self.cpu.cycles;
        memory::config::update_datetime(self.memory.phys.shared_page_mut(), self.boot_clock, tick);

        // regather the timers minus the ones closed since, they fire when the scheduler reaches their tick
        self.kernel.timers.clear();
        for (id, object) in self.kernel.objects.iter() {
            if let kernel::object::KObject::Timer(_) = object {
                self.kernel.timers.push(id);
            }
        }

        self.renderer.end_frame();

        // grab any buffer swap the game queued straight in GSP shared memory b4 the LCDs latch whatever's configured
        services::gsp::refresh(self);
        if self.gpu.interpolates() {
            self.pace();
        }

        // both LCDs finish scanning out, in that order
        services::gsp::signal_interrupt(self, services::gsp::InterruptId::Pdc0);
        services::gsp::signal_interrupt(self, services::gsp::InterruptId::Pdc1);

    }

    /// frame interpolation at a refresh
    /// while a title sets a new picture every other refresh, the GPU draws the in between one with each
    /// from the previous frame's draws and this one's (see zakuro_gpu's interpolation)
    /// it goes up at the refresh the title set its picture, the title's own at the next (a refresh late) so each gets one refresh
    /// a title at full rate or slower shows its own right away and nothing gets drawn in between
    fn pace(&mut self) {
        let pacing = &mut self.pacing;
        pacing.refreshes += 1;
        pacing.between = false;
        let top = self.screen_ref(zakuro_common::Screen::Top, false);
        let pacing = &mut self.pacing;
        if top.is_none() || top == pacing.last {
            if pacing.refreshes - pacing.new_at > 2 {
                // the title slowed down or stopped, its next frame's in between pictures would show too late
                self.gpu.set_twins(false);
            }
            return;
        }
        let gap = pacing.refreshes - pacing.new_at;
        pacing.new_at = pacing.refreshes;
        pacing.last = top;
        let matches = self.gpu.frame_done();
        let pacing = &mut self.pacing;
        pacing.matches = matches;
        pacing.between = gap == 2 && matches.is_some_and(|matches| matches.smooth());
        pacing.shown += pacing.between as u64;
        // the next frame gets in betweens too while the title sticks to every other refresh
        self.gpu.set_twins(gap == 2);
    }

    /// draws an in between picture for every two of the title's from now on (when it draws every other refresh, see pace), or stops
    /// says whether it does, needs the host GPU showing the screens
    pub fn set_interpolation(&mut self, on: bool) -> bool {
        self.pacing = Pacing::default();
        self.gpu.set_interpolation(on)
    }

    /// how the last frame's draws matched the one b4, for frame interpolation
    pub fn interpolation_matches(&self) -> Option<zakuro_gpu::Matches> {
        self.pacing.matches
    }

    /// do the screens show in between pictures this refresh
    pub fn shows_between(&self) -> bool {
        self.pacing.between
    }

    /// the DSP finishing an audio frame, plays its voices one frame on and interrupts the title, whose sound lib updates once per interrupt
    fn audio_frame(&mut self) {
        self.idle_interrupted(false);
        // the whole opening froze till this ran every 4.9 ms, goddamn audio timing
        if self.services.dsp.running {
            services::dsp::advance(self);
            services::dsp::signal_semaphore(self);
        }
        // devices reporting at a title set rate, this is fine enough for them
        services::ir_rst::tick(self);
        services::ir::tick(self);
        services::cam::tick(self);
        services::mic::tick(self);
        services::uds::tick(self);
        services::dlp::tick(self);
    }

    /// stereo samples played since the last call, at AUDIO_SAMPLE_RATE
    pub fn take_audio(&mut self) -> Vec<[i16; 2]> {
        std::mem::take(&mut self.services.dsp.output)
    }

    /// what the software keyboard asks for while a title waits on it
    pub fn keyboard_request(&self) -> Option<&services::keyboard::Request> {
        services::keyboard::request(self)
    }

    /// closes the software keyboard w/ the typed text and the pressed button, 0 is the leftmost and the last one confirms
    pub fn answer_keyboard(&mut self, text: &str, button: usize) {
        services::keyboard::answer(self, text, button);
    }

    /// what the buttons, circle pad and touch screen are doing, HID reports it from the next frame on
    pub fn set_input(&mut self, input: InputState) {
        self.services.hid.input = input;
    }

    // GPU plumbing

    pub fn gpu_read_register(&mut self, offset: u32) -> u32 {
        self.gpu.read_external(offset)
    }

    pub fn gpu_write_register(&mut self, offset: u32, value: u32) {
        self.gpu.write_external(offset, value);
    }

    pub fn set_framebuffer(
        &mut self,
        screen: u32,
        active: u32,
        left: u32,
        right: u32,
        stride: u32,
        format: u32,
    ) {
        self.gpu
            .set_framebuffer(screen, active, left, right, stride, format);
    }

    pub fn submit_command_list(&mut self, paddr: u32, size: u32) {
        let linear_base = self.kernel.linear_base;
        let mut guest = GuestMemory {
            writes: 0,
            linear_base,
            memory: &mut self.memory,
        };
        self.gpu
            .process_command_list(&mut guest, self.renderer.as_mut(), paddr, size);
    }

    /// makes guest memory hold what the host GPU drew over a range, b4 smth other than a draw touches it
    pub fn sync_gpu(&mut self, addr: u32, len: u32) {
        let linear_base = self.kernel.linear_base;
        let mut guest = GuestMemory {
            writes: 0,
            linear_base,
            memory: &mut self.memory,
        };
        self.gpu.sync_memory(&mut guest, addr, len);
    }

    /// writes what a service hands the title (a file it read, say) over guest memory, after what the host GPU drew there
    /// on the console the drawing hit memory long b4, and the title can put the buffer where one it finished drawing to was
    pub fn write_from_service(&mut self, addr: u32, data: &[u8]) {
        if !data.is_empty() {
            self.sync_gpu(addr, data.len() as u32);
        }
        self.memory.write_bytes(addr, data);
    }

    pub fn memory_fill(&mut self, start: u32, end: u32, value: u32, width: u32) {
        let linear_base = self.kernel.linear_base;
        let mut guest = GuestMemory {
            writes: 0,
            linear_base,
            memory: &mut self.memory,
        };
        self.gpu.memory_fill(&mut guest, start, end, value, width);
    }

    pub fn display_transfer(
        &mut self,
        input: u32,
        output: u32,
        input_dimensions: u32,
        output_dimensions: u32,
        flags: u32,
    ) {
        let linear_base = self.kernel.linear_base;
        let mut guest = GuestMemory {
            writes: 0,
            linear_base,
            memory: &mut self.memory,
        };
        self.gpu.display_transfer(
            &mut guest,
            input,
            output,
            input_dimensions,
            output_dimensions,
            flags,
        );
    }

    pub fn texture_copy(
        &mut self,
        input: u32,
        output: u32,
        size: u32,
        input_gap: u32,
        output_gap: u32,
    ) {
        let linear_base = self.kernel.linear_base;
        let mut guest = GuestMemory {
            writes: 0,
            linear_base,
            memory: &mut self.memory,
        };
        self.gpu
            .texture_copy(&mut guest, input, output, size, input_gap, output_gap);
    }

    /// the nine words gsp::ImportDisplayCaptureInfo returns, virtual addrs as the title gave them
    pub fn display_capture_info(&mut self) -> [u32; 9] {
        let top = self.gpu.framebuffers[0];
        let bottom = self.gpu.framebuffers[1];
        let address = |paddr: u32| if paddr == 0 { 0 } else { services::gsp::physical_to_virtual(self, paddr) };
        [
            address(top.address_left()),
            address(top.address_right()),
            top.format,
            top.stride,
            address(bottom.address_left()),
            address(bottom.address_right()),
            bottom.format,
            bottom.stride,
            0,
        ]
    }

    /// a screen as RGBA at the host GPU's resolution, plus width and height
    /// a picture the GPU didn't draw (or the cpu changed since) is the console's own scaled up, so the size stays the same every frame
    pub fn read_screen_scaled(&mut self, screen: zakuro_common::Screen) -> Screen {
        self.read_scaled(screen, false)
    }

    /// the top screen's right eye picture, like read_screen_scaled gives the left, when the title draws in 3D
    pub fn read_right_eye_scaled(&mut self) -> Option<Screen> {
        self.draws_in_3d().then(|| self.read_scaled(zakuro_common::Screen::Top, true))
    }

    fn read_scaled(&mut self, screen: zakuro_common::Screen, right: bool) -> Screen {
        let (width, height) = (screen.width(), screen.height());
        let scale = self.gpu.scale();
        if let Some(scaled) = self.scaled_screen(screen, right) {
            return scaled;
        }
        let native = self.read_screen_at(screen, right);
        if scale == 1 {
            return (std::sync::Arc::new(native), width, height);
        }
        (std::sync::Arc::new(grow(&native, width, scale)), width * scale, height * scale)
    }

    /// does the title draw the top screen in 3D, a picture per eye
    /// its format asks for the screen's two buffers (the 2nd for the right eye),
    /// the console's GSP only turns 3D on w/ bit 5 set and bit 6 clear
    pub fn draws_in_3d(&self) -> bool {
        let config = self.gpu.framebuffers[0];
        config.format & 0x60 == 0x20 && config.address_right() != 0
    }

    /// moves the 3D slider, titles read it every frame to decide if and how deep to draw 3D, 0 down and 1 all the way up
    pub fn set_slider_3d(&mut self, value: f32) {
        self.config.slider_3d = value;
        memory::config::set_slider_3d(self.memory.phys.shared_page_mut(), value);
    }

    /// where a screen's picture lives on the host GPU, for a presenter sharing its device to draw straight from
    /// none when the GPU doesn't show the screens itself or didn't draw what the screen shows (read_screen_scaled has that)
    /// the presenter's work runs after the GPU draws the picture, so the newest goes up
    pub fn gpu_screen(&mut self, screen: zakuro_common::Screen) -> Option<zakuro_gpu::GpuScreen> {
        self.gpu_picture(screen, false)
    }

    /// the top screen's right eye picture on the host GPU, like gpu_screen gives the left, when the title draws in 3D
    pub fn gpu_right_eye(&mut self) -> Option<zakuro_gpu::GpuScreen> {
        if !self.draws_in_3d() {
            return None;
        }
        self.gpu_picture(zakuro_common::Screen::Top, true)
    }

    fn gpu_picture(&mut self, screen: zakuro_common::Screen, right: bool) -> Option<zakuro_gpu::GpuScreen> {
        if !self.gpu.shows_directly() {
            return None;
        }
        let now = self.screen_ref(screen, right)?;
        // frame interpolation's in between picture, where the screen has one
        if self.pacing.between {
            if let Some(between) = self.gpu.between_image(now) {
                return Some(between);
            }
        }
        self.gpu.screen_image(now)
    }

    /// a screen's picture as the GPU showing the screens has it, or the in between of it and the previous (see pace), for saving
    /// waits for the GPU
    pub fn upright_screen(&mut self, screen: zakuro_common::Screen, between: bool) -> Option<Screen> {
        let now = self.screen_ref(screen, false)?;
        let (image, scale) = self.gpu.upright_picture(now, between)?;
        Some((image, screen.width() * scale, screen.height() * scale))
    }

    /// the scaled picture the host GPU drew for a screen, when it's still what the screen shows
    fn scaled_screen(&mut self, screen: zakuro_common::Screen, right: bool) -> Option<Screen> {
        let (width, height) = (screen.width(), screen.height());
        let index = match screen {
            _ if right => 2,
            zakuro_common::Screen::Top => 0,
            zakuro_common::Screen::Bottom => 1,
        };
        let Some(now) = self.screen_ref(screen, right) else {
            self.showing[index] = None;
            return None;
        };
        // what the screen had a present ago goes up now, always that one, so each picture stays up as long as the title kept it
        let show = self.showing[index].replace(now).unwrap_or(now);
        let (image, scale) = self.gpu.scaled_picture(show).or_else(|| self.gpu.scaled_picture(now))?;
        Some((image, width * scale, height * scale))
    }

    /// the picture the host GPU drew for a screen when it's still what's shown, the top one's right eye when right
    fn screen_ref(&mut self, screen: zakuro_common::Screen, right: bool) -> Option<zakuro_gpu::ScreenRef> {
        let (width, height) = (screen.width(), screen.height());
        let index = match screen {
            zakuro_common::Screen::Top => 0,
            zakuro_common::Screen::Bottom => 1,
        };
        let config = self.gpu.framebuffers[index];
        let address = if right { config.address_right() } else { config.address_left() };
        let format = config.color_format();
        let bpp = format.bytes_per_pixel() as u32;
        // plain buffers go a row per screen column like display transfers leave them, rows can be longer than the screen
        let stride = if config.stride == 0 { height * bpp } else { config.stride };
        if self.lcd_force_black || address == 0 || !stride.is_multiple_of(bpp) || stride < height * bpp {
            return None;
        }
        // no sync first, that'd wait for the GPU to finish the picture
        // compared in place, copying both screens every frame costs more than comparing
        let base = services::gsp::physical_to_virtual(self, address);
        let len = width * stride;
        let mut guest = GuestMemory { linear_base: self.kernel.linear_base, memory: &mut self.memory, writes: 0 };
        if let Some(bytes) = guest.slice(base, len as usize) {
            return self.gpu.scaled_screen(base, (height, width), stride / bpp, format, bytes);
        }
        let mut bytes = vec![0u8; len as usize];
        self.memory.read_bytes(base, &mut bytes);
        self.gpu.scaled_screen(base, (height, width), stride / bpp, format, &bytes)
    }

    /// reads one screen into a plain RGBA8 buffer for presenting
    pub fn read_screen(&mut self, screen: zakuro_common::Screen) -> Vec<u8> {
        self.read_screen_at(screen, false)
    }

    /// same, the top screen's right eye picture when right
    fn read_screen_at(&mut self, screen: zakuro_common::Screen, right: bool) -> Vec<u8> {
        let index = match screen {
            zakuro_common::Screen::Top => 0,
            zakuro_common::Screen::Bottom => 1,
        };
        let config = self.gpu.framebuffers[index];
        let width = screen.width();
        let height = screen.height();
        let mut out = vec![0u8; (width * height * 4) as usize];

        let address = if right { config.address_right() } else { config.address_left() };
        if self.lcd_force_black || address == 0 {
            return out;
        }

        let format = config.color_format();
        let bpp = format.bytes_per_pixel();
        let stride = if config.stride == 0 {
            height * bpp as u32
        } else {
            config.stride
        };

        let base = services::gsp::physical_to_virtual(self, address);
        let source_len = (stride * width) as usize;
        self.sync_gpu(base, source_len as u32);
        let mut source = vec![0u8; source_len];
        self.memory.read_bytes(base, &mut source);

        for x in 0..width {
            for y in 0..height {
                // column major in memory and the panel scans bottom to top, bc of course it does
                let offset = (x * stride + (height - 1 - y) * bpp as u32) as usize;
                if offset + bpp > source.len() {
                    continue;
                }
                let pixel = format.decode(&source[offset..offset + bpp]);
                let dst = ((y * width + x) * 4) as usize;
                out[dst..dst + 4].copy_from_slice(&pixel);
            }
        }
        out
    }

    /// a one line summary for the window title and the log
    pub fn status_line(&self) -> String {
        let mut line = format!(
            "frame {} | {} threads | {} modules | {} draws | {} transfers | {} fills | gpu {:.1?}",
            self.frames,
            self.kernel.live_thread_count(),
            self.cro.len(),
            self.gpu.draw_calls,
            self.gpu.transfers,
            self.gpu.fills,
            self.gpu.busy,
        );
        if self.gpu.interpolates() {
            line += &format!(" | in between {}", self.pacing.shown);
        }
        if let Some(library) = &self.recompiled {
            // the fallbacks ran inside recompiled code, which counted them
            let interpreted = self.interpreted_instructions + library.fallbacks();
            let total = (self.interpreted_instructions + self.recompiled_instructions).max(1);
            line += &format!(" | interpreted {:.2}%", interpreted as f64 * 100.0 / total as f64);
        }
        line
    }
}

/// lets the GPU reach guest memory by physical addr
struct GuestMemory<'a> {
    memory: &'a mut Memory,
    linear_base: u32,
    /// writes thru it so far, see GpuMemory::writes
    writes: u64,
}

impl GuestMemory<'_> {
    /// physical addr of an addr translate made, in the linear heap or VRAM
    fn physical(&self, addr: u32) -> Option<u32> {
        // linear heap and VRAM are physical memory in order, so whatever translate made of a physical addr leads back to it
        // VRAM first, a new 3DS heap size reaches it from the old linear heap's base
        if (VRAM_VADDR..VRAM_VADDR + VRAM_SIZE).contains(&addr) {
            Some(VRAM_PADDR + (addr - VRAM_VADDR))
        } else if addr >= self.linear_base && addr - self.linear_base < FCRAM_SIZE_NEW3DS {
            Some(FCRAM_PADDR + (addr - self.linear_base))
        } else {
            None
        }
    }

    /// the bytes at addr in host memory, when they sit together
    fn host(&mut self, addr: u32, len: usize) -> Option<&mut [u8]> {
        let physical = self.physical(addr)?;
        self.memory.phys.host_slice_mut(physical, u32::try_from(len).ok()?)
    }
}

impl GpuMemory for GuestMemory<'_> {
    fn read(&mut self, addr: u32, out: &mut [u8]) {
        self.memory.read_bytes(addr, out);
    }

    fn write(&mut self, addr: u32, data: &[u8]) {
        self.writes += 1;
        self.memory.write_bytes(addr, data);
    }

    fn read_u32(&mut self, addr: u32) -> u32 {
        // not the way the cpu reads, that could make the GPU write back from inside itself
        self.memory.peek32(addr)
    }

    fn guard(&mut self, addr: u32, len: u32) {
        self.memory.guard_cpu_reads(addr, len);
    }

    fn guard_writes(&mut self, addr: u32, len: u32) {
        self.memory.guard_cpu_writes(addr, len);
    }

    fn slice(&mut self, addr: u32, len: usize) -> Option<&[u8]> {
        self.host(addr, len).map(|slice| &*slice)
    }

    fn slice_mut(&mut self, addr: u32, len: usize) -> Option<&mut [u8]> {
        self.writes += 1;
        if let Some(physical) = self.physical(addr) {
            self.memory.wrote_physical(physical, u32::try_from(len).ok()?);
        }
        self.host(addr, len)
    }

    fn writes(&self) -> Option<u64> {
        Some(self.writes)
    }

    fn watch(&mut self, addr: u32, len: u32) -> Option<u64> {
        let physical = self.physical(addr)?;
        // in one piece, like host hands it out
        self.memory.phys.host_slice_mut(physical, len)?;
        Some(self.memory.watch_physical(physical, len))
    }

    fn unchanged(&mut self, addr: u32, len: u32, watch: u64) -> bool {
        self.physical(addr).is_some_and(|physical| self.memory.unchanged_physical(physical, len, watch))
    }

    fn translate(&self, paddr: u32) -> u32 {
        if paddr >= FCRAM_PADDR {
            self.linear_base + (paddr - FCRAM_PADDR)
        } else if (VRAM_PADDR..VRAM_PADDR + VRAM_SIZE).contains(&paddr) {
            VRAM_VADDR + (paddr - VRAM_PADDR)
        } else {
            paddr
        }
    }
}

/// writes back what the GPU drew over a range, for a cpu read waiting on it
///
/// # Safety
///
/// gpu must be the system's GPU, nothing else holds it while the cpu runs
unsafe fn sync_for_cpu(gpu: *mut (), linear_base: u32, memory: &mut Memory, addr: u32, len: u32) {
    log::debug!(
        target: "zakuro_core::memory",
        "the CPU reads 0x{addr:08X}..0x{:08X}, which the GPU drew, written back first",
        addr.wrapping_add(len)
    );
    let gpu = unsafe { &mut *(gpu as *mut Gpu) };
    gpu.sync_depth(&mut GuestMemory { memory, linear_base, writes: 0 }, addr, len);
}

/// writes back what the GPU drew over a range, for a cpu write landing on it
///
/// # Safety
///
/// same as sync_for_cpu
unsafe fn sync_for_cpu_write(gpu: *mut (), linear_base: u32, memory: &mut Memory, addr: u32, len: u32) {
    log::debug!(
        target: "zakuro_core::memory",
        "the CPU writes 0x{addr:08X}..0x{:08X}, which the GPU drew, written back first",
        addr.wrapping_add(len)
    );
    let gpu = unsafe { &mut *(gpu as *mut Gpu) };
    gpu.sync_memory(&mut GuestMemory { memory, linear_base, writes: 0 }, addr, len);
}

impl Drop for System {
    /// what the interpreter ran outlives the run
    fn drop(&mut self) {
        if let Some(hints) = &mut self.hints {
            hints.save();
        }
    }
}

/// an RGBA picture of some width scaled up by a whole factor, each pixel repeated
fn grow(pixels: &[u8], width: u32, scale: u32) -> Vec<u8> {
    let (width, scale) = (width as usize, scale as usize);
    let mut out = Vec::with_capacity(pixels.len() * scale * scale);
    for row in pixels.chunks_exact(width * 4) {
        let mut grown = Vec::with_capacity(row.len() * scale);
        for pixel in row.as_chunks::<4>().0 {
            for _ in 0..scale {
                grown.extend_from_slice(pixel);
            }
        }
        for _ in 0..scale {
            out.extend_from_slice(&grown);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel::object::KObject;
    use kernel::thread::ThreadStatus;
    use zakuro_cpu::Bus;

    fn mutex_state(system: &System, handle: u32) -> (Option<ThreadId>, u32) {
        let object = system.kernel.handles.resolve(handle).expect("a mutex handle");
        match system.kernel.objects.get(object) {
            Some(KObject::Mutex(mutex)) => (mutex.owner, mutex.lock_count),
            _ => panic!("not a mutex"),
        }
    }

    /// a holder that took a mutex twice (svcCreateMutex locked plus svcWaitSynchronization1) and a waiter blocked
    /// on it, holder back on the core
    fn holder_and_waiter(system: &mut System) -> (ThreadId, ThreadId, u32) {
        let holder = system.kernel.create_thread("holder", 0x0010_0000, 0x1000_0000, 0, 0x30, 0);
        let waiter = system.kernel.create_thread("waiter", 0x0010_0000, 0x0FF0_0000, 0, 0x30, 0);
        system.map_tls_page(holder);
        system.map_tls_page(waiter);
        system.kernel.schedule(&mut system.cpu, 0);
        assert_eq!(system.kernel.current_thread, Some(holder));
        system.cpu.regs[1] = 1;
        kernel::svc::dispatch(system, 0x13);
        let handle = system.cpu.regs[1];
        let wait = |system: &mut System| {
            system.cpu.regs[0] = handle;
            system.cpu.regs[2] = u32::MAX;
            system.cpu.regs[3] = u32::MAX;
            kernel::svc::dispatch(system, 0x24);
        };
        wait(system);
        assert_eq!(mutex_state(system, handle), (Some(holder), 2));
        system.kernel.current_thread = Some(waiter);
        system.kernel.thread_mut(waiter).status = ThreadStatus::Running;
        system.kernel.thread_mut(holder).status = ThreadStatus::Ready;
        wait(system);
        assert_eq!(system.kernel.thread(waiter).status, ThreadStatus::WaitSync);
        system.kernel.current_thread = Some(holder);
        system.kernel.thread_mut(holder).status = ThreadStatus::Running;
        system.kernel.reschedule_pending = false;
        (holder, waiter, handle)
    }

    /// the holder ended w/ the mutex all given back, the waiter gets it once at the next schedule
    fn waiter_gets_it(system: &mut System, (holder, waiter, handle): (ThreadId, ThreadId, u32)) {
        assert_eq!(system.kernel.thread(holder).status, ThreadStatus::Dead);
        assert_eq!(mutex_state(system, handle), (None, 0));
        let tick = system.cpu.cycles;
        system.kernel.schedule(&mut system.cpu, tick);
        assert_eq!(system.kernel.current_thread, Some(waiter), "{}", system.kernel.describe_wait(waiter));
        assert!(matches!(system.kernel.thread(waiter).wait_result, Some(WaitResult::Signaled(0))));
        assert_eq!(mutex_state(system, handle), (Some(waiter), 1));
    }

    #[test]
    fn svc_exit_thread_lets_go_of_the_mutexes() {
        let mut system = System::new(Config::default());
        let threads = holder_and_waiter(&mut system);
        kernel::svc::dispatch(&mut system, 0x09);
        waiter_gets_it(&mut system, threads);
    }

    #[test]
    fn returning_from_the_entry_point_lets_go_of_the_mutexes() {
        let mut system = System::new(Config::default());
        let threads = holder_and_waiter(&mut system);
        system.cpu.regs[15] = THREAD_EXIT_MAGIC;
        assert_eq!(system.step(None), StepOutcome::Ran);
        waiter_gets_it(&mut system, threads);
    }

    #[test]
    fn a_branch_to_unmapped_memory_lets_go_of_the_mutexes() {
        let mut system = System::new(Config::default());
        let threads = holder_and_waiter(&mut system);
        system.cpu.regs[15] = 0x0000_1000;
        assert!(!system.memory.is_executable(0x0000_1000));
        assert_eq!(system.step(None), StepOutcome::Ran);
        waiter_gets_it(&mut system, threads);
    }

    /// one thread at create_thread's entry, its code a branch to itself, current and running
    fn one_spinning_thread(system: &mut System) -> ThreadId {
        one_thread_running(system, &[0xEAFF_FFFE])
    }

    /// one thread running code at 0x00100000, current, stack and r0 in the same page (stack at 0x00100F00, r0 at 0x00100800)
    fn one_thread_running(system: &mut System, code: &[u32]) -> ThreadId {
        use memory::{MemoryState, Permission};
        let block = system.memory.phys.allocate(memory::MemoryRegion::Application, 0x1000).unwrap();
        system.memory.map(0x0010_0000, block.addr, 0x1000, Permission::RW | Permission::EXECUTE, MemoryState::Code);
        for (index, word) in code.iter().enumerate() {
            system.memory.write32(0x0010_0000 + index as u32 * 4, *word);
        }
        let id = system.kernel.create_thread("main", 0x0010_0000, 0x0010_0F00, 0, 0x30, 0);
        system.map_tls_page(id);
        system.kernel.schedule(&mut system.cpu, 0);
        assert_eq!(system.kernel.current_thread, Some(id));
        system.cpu.regs[0] = 0x0010_0800;
        id
    }

    /// a loop of svcs (tick, take a mutex, tick again holding it, drop it, sleep) the way titles poll a clock, at 0x00100000
    /// r9 counts the rounds
    const POLLING: [u32; 15] = [
        0xEF00_0028, // svc 0x28, the tick in r0 and r1
        0xE289_9001, // add r9, r9, #1
        0xE3E0_0000, // mvn r0, #0
        0xE3E0_4000, // mvn r4, #0, no timeout
        0xE1A0_1008, // mov r1, r8, where the mutex's handle is
        0xE3A0_2001, // mov r2, #1
        0xE3A0_3000, // mov r3, #0
        0xEF00_0025, // svc 0x25, waiting for the mutex
        0xEF00_0028, // svc 0x28
        0xE1A0_0005, // mov r0, r5
        0xEF00_0014, // svc 0x14, releasing the mutex in r5
        0xE1A0_0006, // mov r0, r6
        0xE3A0_1000, // mov r1, #0
        0xEF00_000A, // svc 0x0A, sleeping for r6 nanoseconds
        0xEAFF_FFF0, // b 0x00100000
    ];

    /// a loop that only reads the tick and sums what it reads so every read shows at the end, at 0x00100040, r9 counts rounds
    const TICKING: [u32; 4] = [
        0xEF00_0028, // svc 0x28
        0xE289_9001, // add r9, r9, #1
        0xE087_7000, // add r7, r7, r0
        0xEAFF_FFFB, // b 0x00100040
    ];

    /// a loop taking the mutex twice, reading the tick (summed like TICKING) and dropping the mutex twice, never sleeping
    /// at 0x00100060, r9 counts rounds
    const LOCKING: [u32; 17] = [
        0xE289_9001, // add r9, r9, #1
        0xE3E0_0000, // mvn r0, #0
        0xE3E0_4000, // mvn r4, #0
        0xE1A0_1008, // mov r1, r8
        0xE3A0_2001, // mov r2, #1
        0xE3A0_3000, // mov r3, #0
        0xEF00_0025, // svc 0x25
        0xE3E0_0000, // mvn r0, #0
        0xE1A0_1008, // mov r1, r8
        0xEF00_0025, // svc 0x25, taking it again
        0xEF00_0028, // svc 0x28
        0xE087_7000, // add r7, r7, r0
        0xE1A0_0005, // mov r0, r5
        0xEF00_0014, // svc 0x14, still held
        0xE1A0_0005, // mov r0, r5
        0xEF00_0014, // svc 0x14, free
        0xEAFF_FFEE, // b 0x00100060
    ];

    /// a loop taking the mutex twice, dropping it once, sleeping r6 nanos still holding it and dropping it again
    /// at 0x00100100, r9 counts rounds
    const HOLDING: [u32; 18] = [
        0xE289_9001, // add r9, r9, #1
        0xE3E0_0000, // mvn r0, #0
        0xE3E0_4000, // mvn r4, #0
        0xE1A0_1008, // mov r1, r8
        0xE3A0_2001, // mov r2, #1
        0xE3A0_3000, // mov r3, #0
        0xEF00_0025, // svc 0x25
        0xE3E0_0000, // mvn r0, #0
        0xE1A0_1008, // mov r1, r8
        0xEF00_0025, // svc 0x25, taking it again
        0xE1A0_0005, // mov r0, r5
        0xEF00_0014, // svc 0x14, still held
        0xE1A0_0006, // mov r0, r6
        0xE3A0_1000, // mov r1, #0
        0xEF00_000A, // svc 0x0A, sleeping with it
        0xE1A0_0005, // mov r0, r5
        0xEF00_0014, // svc 0x14, free
        0xEAFF_FFED, // b 0x00100100
    ];

    /// a loop setting a one shot timer r6 nanos ahead and reading the tick twice (summed), at 0x00100180, r9 counts rounds
    const TIMING: [u32; 12] = [
        0xE1A0_000A, // mov r0, r10, the timer's handle
        0xE3A0_1000, // mov r1, #0, no interval
        0xE1A0_2006, // mov r2, r6, the delay
        0xE3A0_3000, // mov r3, #0
        0xE3A0_4000, // mov r4, #0
        0xEF00_001B, // svc 0x1B, setting the timer
        0xEF00_0028, // svc 0x28
        0xE087_7000, // add r7, r7, r0
        0xEF00_0028, // svc 0x28
        0xE087_7000, // add r7, r7, r0
        0xE289_9001, // add r9, r9, #1
        0xEAFF_FFF3, // b 0x00100180
    ];

    /// a loop waiting on that timer, at 0x00100200, r9 counts rounds
    const WAITING: [u32; 6] = [
        0xE1A0_000A, // mov r0, r10
        0xE3E0_2000, // mvn r2, #0, no timeout
        0xE3E0_3000, // mvn r3, #0
        0xEF00_0024, // svc 0x24
        0xE289_9001, // add r9, r9, #1
        0xEAFF_FFF9, // b 0x00100200
    ];

    /// a loop taking the mutex twice and dropping it twice, each time at the same svc, reading the tick in between
    /// like a title's lock and unlock funcs taking a mutex again inside, at 0x00100280, r12 counts holds, r9 rounds
    const NESTING: [u32; 17] = [
        0xE289_9001, // add r9, r9, #1
        0xE3E0_0000, // mvn r0, #0
        0xE3E0_4000, // mvn r4, #0
        0xE1A0_1008, // mov r1, r8
        0xE3A0_2001, // mov r2, #1
        0xE3A0_3000, // mov r3, #0
        0xEF00_0025, // svc 0x25
        0xE28C_C001, // add r12, r12, #1
        0xE35C_0002, // cmp r12, #2
        0xBAFF_FFF6, // blt 0x00100284, taking it again
        0xEF00_0028, // svc 0x28
        0xE087_7000, // add r7, r7, r0
        0xE1A0_0005, // mov r0, r5
        0xEF00_0014, // svc 0x14
        0xE25C_C001, // subs r12, r12, #1
        0x1AFF_FFFB, // bne 0x001002B0, letting it go again
        0xEAFF_FFEE, // b 0x00100280
    ];

    /// a loop waiting for a time like a title's scheduler waits for its next task, at 0x00100300
    /// reads the clock till it reaches the target r11 points at, taking and dropping the mutex
    /// and leaving the tick it read in the fourth word each turn (the next turn overwrites it)
    /// then sets the next target r6 after the tick it got there at and counts in r9
    /// its turns all look the same till the target, which the run skips to
    const SCHEDULING: [u32; 23] = [
        0xEF00_0028, // svc 0x28
        0xE1A0_7000, // mov r7, r0
        0xE59B_2000, // ldr r2, [r11], the target
        0xE59B_3004, // ldr r3, [r11, #4]
        0xE050_2002, // subs r2, r0, r2
        0xE0D1_3003, // sbcs r3, r1, r3
        0x2A00_0009, // bhs 0x00100344, there
        0xE3E0_0000, // mvn r0, #0
        0xE3E0_4000, // mvn r4, #0
        0xE1A0_1008, // mov r1, r8
        0xE3A0_2001, // mov r2, #1
        0xE3A0_3000, // mov r3, #0
        0xEF00_0025, // svc 0x25
        0xE1A0_0005, // mov r0, r5
        0xEF00_0014, // svc 0x14
        0xE58B_700C, // str r7, [r11, #12]
        0xEAFF_FFEE, // b 0x00100300
        0xE090_2006, // adds r2, r0, r6
        0xE2A1_3000, // adc r3, r1, #0
        0xE58B_2000, // str r2, [r11]
        0xE58B_3004, // str r3, [r11, #4]
        0xE289_9001, // add r9, r9, #1
        0xEAFF_FFE8, // b 0x00100300
    ];

    /// a thread sleeping r6 nanos, then making the first thread's SCHEDULING hit its target sooner, 0x5000 cycles after the tick it woke at
    /// at 0x00100380, like a title making its scheduler run a task another thread readied
    const POSTING: [u32; 10] = [
        0xE1A0_0006, // mov r0, r6
        0xE3A0_1000, // mov r1, #0
        0xEF00_000A, // svc 0x0A
        0xEF00_0028, // svc 0x28
        0xE59F_300C, // ldr r3, [pc, #12], TARGETS_AT
        0xE280_2A05, // add r2, r0, #0x5000
        0xE583_2000, // str r2, [r3]
        0xE583_1004, // str r1, [r3, #4]
        0xEAFF_FFF6, // b 0x00100380
        TARGETS_AT,
    ];

    /// SCHEDULING w/ its waiting in its own block, at 0x00100400
    /// each waiting turn leaves the block that'd move the target halfway, the code charges it whole on entry
    /// so the code's turns take longer than the instrs they run
    const LEAVING: [u32; 21] = [
        0xEF00_0028, // svc 0x28
        0xE59B_2000, // ldr r2, [r11]
        0xE59B_3004, // ldr r3, [r11, #4]
        0xE050_2002, // subs r2, r0, r2
        0xE0D1_3003, // sbcs r3, r1, r3
        0x3A00_0005, // bcc 0x00100430, waiting on
        0xE090_2006, // adds r2, r0, r6
        0xE2A1_3000, // adc r3, r1, #0
        0xE58B_2000, // str r2, [r11]
        0xE58B_3004, // str r3, [r11, #4]
        0xE289_9001, // add r9, r9, #1
        0xEAFF_FFF3, // b 0x00100400
        0xE3E0_0000, // mvn r0, #0
        0xE3E0_4000, // mvn r4, #0
        0xE1A0_1008, // mov r1, r8
        0xE3A0_2001, // mov r2, #1
        0xE3A0_3000, // mov r3, #0
        0xEF00_0025, // svc 0x25
        0xE1A0_0005, // mov r0, r5
        0xEF00_0014, // svc 0x14
        0xEAFF_FFEA, // b 0x00100400
    ];

    /// SCHEDULING reading the clock twice a turn thru its own func at 0x00100480, like titles read it thru one
    /// both reads make the same svc w/ the same sp and lr, the spot each comes back to on the stack
    const TWICE: [u32; 24] = [
        0xEB00_001E, // bl 0x00100500, the clock
        0xE1A0_7000, // mov r7, r0
        0xEB00_001C, // bl 0x00100500, again
        0xE59B_2000, // ldr r2, [r11]
        0xE59B_3004, // ldr r3, [r11, #4]
        0xE050_2002, // subs r2, r0, r2
        0xE0D1_3003, // sbcs r3, r1, r3
        0x2A00_0009, // bhs 0x001004C8, there
        0xE3E0_0000, // mvn r0, #0
        0xE3E0_4000, // mvn r4, #0
        0xE1A0_1008, // mov r1, r8
        0xE3A0_2001, // mov r2, #1
        0xE3A0_3000, // mov r3, #0
        0xEF00_0025, // svc 0x25
        0xE1A0_0005, // mov r0, r5
        0xEF00_0014, // svc 0x14
        0xE58B_700C, // str r7, [r11, #12]
        0xEAFF_FFED, // b 0x00100480
        0xE090_2006, // adds r2, r0, r6
        0xE2A1_3000, // adc r3, r1, #0
        0xE58B_2000, // str r2, [r11]
        0xE58B_3004, // str r3, [r11, #4]
        0xE289_9001, // add r9, r9, #1
        0xEAFF_FFE7, // b 0x00100480
    ];

    /// the func TWICE reads the clock thru, at 0x00100500, it calls the one making the svc
    const TICKER: [u32; 6] = [
        0xE52D_E004, // str lr, [sp, #-4]!
        0xEB00_0001, // bl 0x00100510
        0xE49D_F004, // ldr pc, [sp], #4
        0xE1A0_0000, // nop
        0xEF00_0028, // svc 0x28
        0xE12F_FF1E, // bx lr
    ];

    /// SCHEDULING w/o the mutex at 0x00100540, next target set from the tick the previous turn left in the fourth word
    /// the turn that goes another way reads what earlier turns wrote, skipping them has to leave that intact
    const SETTLING: [u32; 15] = [
        0xEF00_0028, // svc 0x28
        0xE59B_2000, // ldr r2, [r11], the target
        0xE59B_3004, // ldr r3, [r11, #4]
        0xE050_2002, // subs r2, r0, r2
        0xE0D1_3003, // sbcs r3, r1, r3
        0x2A00_0001, // bhs 0x00100560, there
        0xE58B_000C, // str r0, [r11, #12]
        0xEAFF_FFF7, // b 0x00100540
        0xE59B_200C, // ldr r2, [r11, #12], the tick the turn before read
        0xE092_2006, // adds r2, r2, r6
        0xE2A1_3000, // adc r3, r1, #0
        0xE58B_2000, // str r2, [r11]
        0xE58B_3004, // str r3, [r11, #4]
        0xE289_9001, // add r9, r9, #1
        0xEAFF_FFF0, // b 0x00100540
    ];

    /// a loop reading the clock and counting turns in the third word r11 points at, at 0x001003C0
    /// nothing of it stays in regs, every turn changes memory so none gets skipped
    const TALLYING: [u32; 6] = [
        0xEF00_0028, // svc 0x28
        0xE59B_2008, // ldr r2, [r11, #8]
        0xE282_2001, // add r2, r2, #1
        0xE58B_2008, // str r2, [r11, #8]
        0xE3A0_2000, // mov r2, #0
        0xEAFF_FFF9, // b 0x001003C0
    ];

    /// where the threads find the mutex handle
    const MUTEX_HANDLE_AT: u32 = 0x0010_0800;
    /// where SCHEDULING's targets live, a word pair per thread
    const TARGETS_AT: u32 = 0x0010_0880;

    /// the loops' blocks the way 3dsrecomp cuts code, each w/ its length, the svc it ends on and where it goes next
    fn block(at: u32) -> Option<(i32, Option<u32>, u32)> {
        Some(match at {
            0x0010_0000 => (1, Some(0x28), 0x0010_0004),
            0x0010_0004 => (7, Some(0x25), 0x0010_0020),
            0x0010_0020 => (1, Some(0x28), 0x0010_0024),
            0x0010_0024 => (2, Some(0x14), 0x0010_002C),
            0x0010_002C => (3, Some(0x0A), 0x0010_0038),
            0x0010_0038 => (1, None, 0x0010_0000),
            0x0010_0040 => (1, Some(0x28), 0x0010_0044),
            0x0010_0044 => (3, None, 0x0010_0040),
            0x0010_0060 => (7, Some(0x25), 0x0010_007C),
            0x0010_007C => (3, Some(0x25), 0x0010_0088),
            0x0010_0088 => (1, Some(0x28), 0x0010_008C),
            0x0010_008C => (3, Some(0x14), 0x0010_0098),
            0x0010_0098 => (2, Some(0x14), 0x0010_00A0),
            0x0010_00A0 => (1, None, 0x0010_0060),
            0x0010_0100 => (7, Some(0x25), 0x0010_011C),
            0x0010_011C => (3, Some(0x25), 0x0010_0128),
            0x0010_0128 => (2, Some(0x14), 0x0010_0130),
            0x0010_0130 => (3, Some(0x0A), 0x0010_013C),
            0x0010_013C => (2, Some(0x14), 0x0010_0144),
            0x0010_0144 => (1, None, 0x0010_0100),
            0x0010_0180 => (6, Some(0x1B), 0x0010_0198),
            0x0010_0198 => (1, Some(0x28), 0x0010_019C),
            0x0010_019C => (2, Some(0x28), 0x0010_01A4),
            0x0010_01A4 => (3, None, 0x0010_0180),
            0x0010_0200 => (4, Some(0x24), 0x0010_0210),
            0x0010_0210 => (2, None, 0x0010_0200),
            0x0010_0280 => (1, None, 0x0010_0284),
            0x0010_0284 => (6, Some(0x25), 0x0010_029C),
            // where the two branches go is the code's call
            0x0010_029C => (3, None, 0x0010_02A8),
            0x0010_02A8 => (1, Some(0x28), 0x0010_02AC),
            0x0010_02AC => (1, None, 0x0010_02B0),
            0x0010_02B0 => (2, Some(0x14), 0x0010_02B8),
            0x0010_02B8 => (2, None, 0x0010_02C0),
            0x0010_02C0 => (1, None, 0x0010_0280),
            0x0010_0300 => (1, Some(0x28), 0x0010_0304),
            // where the branch goes is the code's call
            0x0010_0304 => (6, None, 0x0010_031C),
            0x0010_031C => (6, Some(0x25), 0x0010_0334),
            0x0010_0334 => (2, Some(0x14), 0x0010_033C),
            0x0010_033C => (2, None, 0x0010_0300),
            0x0010_0344 => (6, None, 0x0010_0300),
            0x0010_0380 => (3, Some(0x0A), 0x0010_038C),
            0x0010_038C => (1, Some(0x28), 0x0010_0390),
            0x0010_0390 => (5, None, 0x0010_0380),
            0x0010_03C0 => (1, Some(0x28), 0x0010_03C4),
            0x0010_03C4 => (5, None, 0x0010_03C0),
            0x0010_0400 => (1, Some(0x28), 0x0010_0404),
            // charged whole, where it goes is the code's call
            0x0010_0404 => (11, None, 0x0010_0400),
            0x0010_0430 => (6, Some(0x25), 0x0010_0448),
            0x0010_0448 => (2, Some(0x14), 0x0010_0450),
            0x0010_0450 => (1, None, 0x0010_0400),
            // calls and returns, they go where lr and the stack say
            0x0010_0480 => (1, None, 0x0010_0500),
            0x0010_0484 => (2, None, 0x0010_0500),
            0x0010_048C => (5, None, 0x0010_04A0),
            0x0010_04A0 => (6, Some(0x25), 0x0010_04B8),
            0x0010_04B8 => (2, Some(0x14), 0x0010_04C0),
            0x0010_04C0 => (2, None, 0x0010_0480),
            0x0010_04C8 => (6, None, 0x0010_0480),
            0x0010_0500 => (2, None, 0x0010_0510),
            0x0010_0508 => (1, None, 0),
            0x0010_0510 => (1, Some(0x28), 0x0010_0514),
            0x0010_0514 => (1, None, 0),
            0x0010_0540 => (1, Some(0x28), 0x0010_0544),
            // where the branch goes is the code's call
            0x0010_0544 => (5, None, 0x0010_0558),
            0x0010_0558 => (2, None, 0x0010_0540),
            0x0010_0560 => (7, None, 0x0010_0540),
            _ => return None,
        })
    }

    /// ticks the code below read itself
    static TICKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    /// waits and releases the code below answered itself
    static HELD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    /// the word at address where the page tables reach it directly, like recomp_held_at reads it
    unsafe fn word_at(ctx: *mut recomp_abi::Context, address: u32) -> Option<u32> {
        unsafe {
            let page = *(*ctx).read_pages.add(address as usize >> 12);
            let offset = (address & 0xFFF) as usize;
            (!page.is_null() && offset <= 0xFFC).then(|| u32::from_le_bytes(std::array::from_fn(|i| *page.add(offset + i))))
        }
    }

    /// writes the word at address like code does, thru the context's page for it or the host callback
    unsafe fn set_word(ctx: *mut recomp_abi::Context, address: u32, value: u32) {
        unsafe {
            let page = *(*ctx).write_pages.add(address as usize >> 12);
            let offset = (address & 0xFFF) as usize;
            if !page.is_null() && offset <= 0xFFC {
                for (i, byte) in value.to_le_bytes().into_iter().enumerate() {
                    *page.add(offset + i) = byte;
                }
            } else {
                ((*(*ctx).host).write32)(ctx, address, value);
            }
        }
    }

    /// flags of a minus b like cmp and subs set them, plus whether a < b signed (what lt takes)
    unsafe fn subtract_flags(ctx: *mut recomp_abi::Context, a: u32, b: u32) -> bool {
        unsafe {
            let (result, borrow) = a.overflowing_sub(b);
            let overflow = (a as i32).overflowing_sub(b as i32).1;
            (*ctx).n = (result >> 31) as u8;
            (*ctx).z = (result == 0) as u8;
            (*ctx).c = !borrow as u8;
            (*ctx).v = overflow as u8;
            (*ctx).n != (*ctx).v
        }
    }

    /// what ANSWERED leaves of an svc the code answered itself at at
    unsafe fn answered(ctx: *mut recomp_abi::Context, at: u32) {
        unsafe {
            (*ctx).tick_last = (*ctx).tick_end - (*ctx).budget as i64 as u64;
            (*ctx).trail[(*ctx).done as usize % recomp_abi::TRAIL] = at;
            (*ctx).done += 1;
        }
    }

    /// the loops above as 3dsrecomp writes code, each block charged to the budget on entry
    /// their svcs run in place (reading the clock themselves where the host allows) or leave for the step, like v4 code
    /// they never return, only stop
    unsafe extern "C" fn polling<const IN_PLACE: bool>(ctx: *mut recomp_abi::Context) -> u32 {
        unsafe {
            let mut at = (*ctx).r[15];
            loop {
                let Some((size, svc, mut next)) = block(at) else {
                    (*ctx).exit = recomp_abi::EXIT_UNWIND;
                    return recomp_abi::STOPPED;
                };
                if (*ctx).budget < size {
                    (*ctx).r[15] = at;
                    (*ctx).exit = recomp_abi::EXIT_BUDGET;
                    return recomp_abi::STOPPED;
                }
                (*ctx).budget -= size;
                match at {
                    0x0010_0004 | 0x0010_0060 | 0x0010_0100 => {
                        (*ctx).r[9] = (*ctx).r[9].wrapping_add(1);
                        (*ctx).r[0] = u32::MAX;
                        (*ctx).r[4] = u32::MAX;
                        (*ctx).r[1] = (*ctx).r[8];
                        (*ctx).r[2] = 1;
                        (*ctx).r[3] = 0;
                    }
                    0x0010_0024 | 0x0010_0128 | 0x0010_013C => (*ctx).r[0] = (*ctx).r[5],
                    0x0010_002C | 0x0010_0130 => {
                        (*ctx).r[0] = (*ctx).r[6];
                        (*ctx).r[1] = 0;
                    }
                    0x0010_0044 => {
                        (*ctx).r[9] = (*ctx).r[9].wrapping_add(1);
                        (*ctx).r[7] = (*ctx).r[7].wrapping_add((*ctx).r[0]);
                    }
                    0x0010_007C | 0x0010_011C => {
                        (*ctx).r[0] = u32::MAX;
                        (*ctx).r[1] = (*ctx).r[8];
                    }
                    0x0010_008C => {
                        (*ctx).r[7] = (*ctx).r[7].wrapping_add((*ctx).r[0]);
                        (*ctx).r[0] = (*ctx).r[5];
                    }
                    0x0010_0098 => (*ctx).r[0] = (*ctx).r[5],
                    0x0010_0180 => {
                        let r = &mut (*ctx).r;
                        (r[0], r[1], r[2], r[3], r[4]) = (r[10], 0, r[6], 0, 0);
                    }
                    0x0010_019C => (*ctx).r[7] = (*ctx).r[7].wrapping_add((*ctx).r[0]),
                    0x0010_01A4 => {
                        (*ctx).r[7] = (*ctx).r[7].wrapping_add((*ctx).r[0]);
                        (*ctx).r[9] = (*ctx).r[9].wrapping_add(1);
                    }
                    0x0010_0200 => {
                        let r = &mut (*ctx).r;
                        (r[0], r[2], r[3]) = (r[10], u32::MAX, u32::MAX);
                    }
                    0x0010_0210 => (*ctx).r[9] = (*ctx).r[9].wrapping_add(1),
                    0x0010_0280 => (*ctx).r[9] = (*ctx).r[9].wrapping_add(1),
                    0x0010_0284 => {
                        let r = &mut (*ctx).r;
                        (r[0], r[4], r[1], r[2], r[3]) = (u32::MAX, u32::MAX, r[8], 1, 0);
                    }
                    // add r12, r12, #1, cmp r12, #2, blt
                    0x0010_029C => {
                        (*ctx).r[12] = (*ctx).r[12].wrapping_add(1);
                        let less = subtract_flags(ctx, (*ctx).r[12], 2);
                        if less {
                            next = 0x0010_0284;
                        }
                    }
                    0x0010_02AC => (*ctx).r[7] = (*ctx).r[7].wrapping_add((*ctx).r[0]),
                    0x0010_02B0 => (*ctx).r[0] = (*ctx).r[5],
                    // subs r12, r12, #1, bne
                    0x0010_02B8 => {
                        let held = (*ctx).r[12];
                        subtract_flags(ctx, held, 1);
                        (*ctx).r[12] = held.wrapping_sub(1);
                        if (*ctx).z == 0 {
                            next = 0x0010_02B0;
                        }
                    }
                    // mov r7, r0, ldr r2, [r11], ldr r3, [r11, #4],
                    // subs r2, r0, r2, sbcs r3, r1, r3, bhs
                    0x0010_0304 => {
                        let r = &mut (*ctx).r;
                        r[7] = r[0];
                        let (low, high) = (word_at(ctx, r[11]).unwrap(), word_at(ctx, r[11] + 4).unwrap());
                        let (below, borrow) = r[0].overflowing_sub(low);
                        let (part, first) = r[1].overflowing_sub(high);
                        let (above, second) = part.overflowing_sub(borrow as u32);
                        let overflow = ((r[1] ^ high) & (r[1] ^ above)) >> 31 != 0;
                        (r[2], r[3]) = (below, above);
                        (*ctx).n = (above >> 31) as u8;
                        (*ctx).z = (above == 0) as u8;
                        (*ctx).c = !(first || second) as u8;
                        (*ctx).v = overflow as u8;
                        if (*ctx).c != 0 {
                            next = 0x0010_0344;
                        }
                    }
                    0x0010_031C => {
                        let r = &mut (*ctx).r;
                        (r[0], r[4], r[1], r[2], r[3]) = (u32::MAX, u32::MAX, r[8], 1, 0);
                    }
                    0x0010_0334 => (*ctx).r[0] = (*ctx).r[5],
                    // the store left to the interpreter, like code leaves an instr it has no C for
                    0x0010_033C => {
                        ((*(*ctx).host).interpret)(ctx, 0x0010_033C, SCHEDULING[15]);
                        if (*ctx).exit != 0 {
                            return recomp_abi::STOPPED;
                        }
                    }
                    // the next target, adds then adc, its first half stored by the interpreter, and counted
                    0x0010_0344 => {
                        let r = &mut (*ctx).r;
                        let (low, step) = (r[0], r[6]);
                        let (sum, carry) = low.overflowing_add(step);
                        let above = r[1].wrapping_add(carry as u32);
                        (r[2], r[3]) = (sum, above);
                        let at = r[11];
                        (*ctx).n = (sum >> 31) as u8;
                        (*ctx).z = (sum == 0) as u8;
                        (*ctx).c = carry as u8;
                        (*ctx).v = (((low ^ sum) & (step ^ sum)) >> 31 != 0) as u8;
                        ((*(*ctx).host).interpret)(ctx, 0x0010_034C, SCHEDULING[19]);
                        if (*ctx).exit != 0 {
                            return recomp_abi::STOPPED;
                        }
                        set_word(ctx, at + 4, above);
                        (*ctx).r[9] = (*ctx).r[9].wrapping_add(1);
                    }
                    0x0010_0390 => {
                        let r = &mut (*ctx).r;
                        (r[3], r[2]) = (TARGETS_AT, r[0].wrapping_add(0x5000));
                        let (low, high) = (r[2], r[1]);
                        set_word(ctx, TARGETS_AT, low);
                        set_word(ctx, TARGETS_AT + 4, high);
                    }
                    0x0010_0404 => {
                        let r = &mut (*ctx).r;
                        let (low, high) = (word_at(ctx, r[11]).unwrap(), word_at(ctx, r[11] + 4).unwrap());
                        let (below, borrow) = r[0].overflowing_sub(low);
                        let (part, first) = r[1].overflowing_sub(high);
                        let (above, second) = part.overflowing_sub(borrow as u32);
                        (r[2], r[3]) = (below, above);
                        (*ctx).n = (above >> 31) as u8;
                        (*ctx).z = (above == 0) as u8;
                        (*ctx).c = !(first || second) as u8;
                        (*ctx).v = (((r[1] ^ high) & (r[1] ^ above)) >> 31 != 0) as u8;
                        if (*ctx).c == 0 {
                            next = 0x0010_0430;
                        } else {
                            let r = &mut (*ctx).r;
                            let (low, step) = (r[0], r[6]);
                            let (sum, carry) = low.overflowing_add(step);
                            let above = r[1].wrapping_add(carry as u32);
                            (r[2], r[3]) = (sum, above);
                            r[9] = r[9].wrapping_add(1);
                            let at = r[11];
                            (*ctx).n = (sum >> 31) as u8;
                            (*ctx).z = (sum == 0) as u8;
                            (*ctx).c = carry as u8;
                            (*ctx).v = (((low ^ sum) & (step ^ sum)) >> 31 != 0) as u8;
                            set_word(ctx, at, sum);
                            set_word(ctx, at + 4, above);
                        }
                    }
                    0x0010_0430 => {
                        let r = &mut (*ctx).r;
                        (r[0], r[4], r[1], r[2], r[3]) = (u32::MAX, u32::MAX, r[8], 1, 0);
                    }
                    0x0010_0448 => (*ctx).r[0] = (*ctx).r[5],
                    0x0010_0480 => (*ctx).r[14] = 0x0010_0484,
                    0x0010_0484 => {
                        (*ctx).r[7] = (*ctx).r[0];
                        (*ctx).r[14] = 0x0010_048C;
                    }
                    0x0010_048C => {
                        let r = &mut (*ctx).r;
                        let (low, high) = (word_at(ctx, r[11]).unwrap(), word_at(ctx, r[11] + 4).unwrap());
                        let (below, borrow) = r[0].overflowing_sub(low);
                        let (part, first) = r[1].overflowing_sub(high);
                        let (above, second) = part.overflowing_sub(borrow as u32);
                        (r[2], r[3]) = (below, above);
                        (*ctx).n = (above >> 31) as u8;
                        (*ctx).z = (above == 0) as u8;
                        (*ctx).c = !(first || second) as u8;
                        (*ctx).v = (((r[1] ^ high) & (r[1] ^ above)) >> 31 != 0) as u8;
                        if (*ctx).c != 0 {
                            next = 0x0010_04C8;
                        }
                    }
                    0x0010_04A0 => {
                        let r = &mut (*ctx).r;
                        (r[0], r[4], r[1], r[2], r[3]) = (u32::MAX, u32::MAX, r[8], 1, 0);
                    }
                    0x0010_04B8 => (*ctx).r[0] = (*ctx).r[5],
                    0x0010_04C0 => set_word(ctx, (*ctx).r[11] + 12, (*ctx).r[7]),
                    0x0010_04C8 => {
                        let r = &mut (*ctx).r;
                        let (low, step) = (r[0], r[6]);
                        let (sum, carry) = low.overflowing_add(step);
                        let above = r[1].wrapping_add(carry as u32);
                        (r[2], r[3]) = (sum, above);
                        r[9] = r[9].wrapping_add(1);
                        let at = r[11];
                        (*ctx).n = (sum >> 31) as u8;
                        (*ctx).z = (sum == 0) as u8;
                        (*ctx).c = carry as u8;
                        (*ctx).v = (((low ^ sum) & (step ^ sum)) >> 31 != 0) as u8;
                        set_word(ctx, at, sum);
                        set_word(ctx, at + 4, above);
                    }
                    // str lr, [sp, #-4]!, bl
                    0x0010_0500 => {
                        let sp = (*ctx).r[13] - 4;
                        set_word(ctx, sp, (*ctx).r[14]);
                        (*ctx).r[13] = sp;
                        (*ctx).r[14] = 0x0010_0508;
                    }
                    // ldr pc, [sp], #4
                    0x0010_0508 => {
                        let sp = (*ctx).r[13];
                        next = word_at(ctx, sp).unwrap();
                        (*ctx).r[13] = sp + 4;
                    }
                    0x0010_0514 => next = (*ctx).r[14],
                    // ldr r2, [r11], ldr r3, [r11, #4], subs r2, r0, r2,
                    // sbcs r3, r1, r3, bhs
                    0x0010_0544 => {
                        let r = &mut (*ctx).r;
                        let (low, high) = (word_at(ctx, r[11]).unwrap(), word_at(ctx, r[11] + 4).unwrap());
                        let (below, borrow) = r[0].overflowing_sub(low);
                        let (part, first) = r[1].overflowing_sub(high);
                        let (above, second) = part.overflowing_sub(borrow as u32);
                        (r[2], r[3]) = (below, above);
                        (*ctx).n = (above >> 31) as u8;
                        (*ctx).z = (above == 0) as u8;
                        (*ctx).c = !(first || second) as u8;
                        (*ctx).v = (((r[1] ^ high) & (r[1] ^ above)) >> 31 != 0) as u8;
                        if (*ctx).c != 0 {
                            next = 0x0010_0560;
                        }
                    }
                    0x0010_0558 => set_word(ctx, (*ctx).r[11] + 12, (*ctx).r[0]),
                    // next target from the tick the previous turn left
                    0x0010_0560 => {
                        let r = &mut (*ctx).r;
                        let (low, step) = (word_at(ctx, r[11] + 12).unwrap(), r[6]);
                        let (sum, carry) = low.overflowing_add(step);
                        let above = r[1].wrapping_add(carry as u32);
                        (r[2], r[3]) = (sum, above);
                        r[9] = r[9].wrapping_add(1);
                        let at = r[11];
                        (*ctx).n = (sum >> 31) as u8;
                        (*ctx).z = (sum == 0) as u8;
                        (*ctx).c = carry as u8;
                        (*ctx).v = (((low ^ sum) & (step ^ sum)) >> 31 != 0) as u8;
                        set_word(ctx, at, sum);
                        set_word(ctx, at + 4, above);
                    }
                    0x0010_03C4 => {
                        let at = (*ctx).r[11] + 8;
                        set_word(ctx, at, word_at(ctx, at).unwrap().wrapping_add(1));
                        (*ctx).r[2] = 0;
                    }
                    _ => {}
                }
                if let Some(number) = svc {
                    (*ctx).r[15] = next;
                    if !IN_PLACE {
                        (*ctx).svc = number;
                        (*ctx).exit = recomp_abi::EXIT_SVC;
                        return recomp_abi::STOPPED;
                    }
                    // the tick read and the lent mutex taken and dropped where the host allows, like SVC_TICK, SVC_WAIT and SVC_RELEASE
                    let held = (*ctx).budget > (*ctx).held_floor;
                    if number == 0x28 && (*ctx).tick_at == next && (*ctx).budget > (*ctx).tick_floor {
                        let tick = (*ctx).tick_end - (*ctx).budget as i64 as u64;
                        answered(ctx, next);
                        (*ctx).budget -= (*ctx).tick_cost;
                        (*ctx).ticks += 1;
                        (*ctx).r[0] = tick as u32;
                        (*ctx).r[1] = (tick >> 32) as u32;
                        TICKED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    } else if number == 0x25
                        && (*ctx).wait_at == next
                        && held
                        && (*ctx).r[2] == 1
                        && word_at(ctx, (*ctx).r[1]) == Some((*ctx).held)
                    {
                        answered(ctx, next);
                        (*ctx).holding += 1;
                        (*ctx).r[0] = 0;
                        (*ctx).r[1] = 0;
                        HELD.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    } else if number == 0x14 && (*ctx).release_at == next && held && (*ctx).r[0] == (*ctx).held && (*ctx).holding != 0 {
                        answered(ctx, next);
                        (*ctx).holding -= 1;
                        (*ctx).r[0] = 0;
                        HELD.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    } else {
                        ((*(*ctx).host).svc)(ctx, number);
                        if (*ctx).exit != 0 {
                            return recomp_abi::STOPPED;
                        }
                    }
                }
                at = next;
            }
        }
    }

    /// threads on the loops above as recompiled code sharing a mutex, each w/ its start, nanos slept per round and priority
    /// plus a timer nobody waits on firing every so many nanos, if any
    fn polling_system(in_place: bool, threads: &[(u32, u32, u32)], timer: Option<u32>) -> System {
        use memory::{MemoryState, Permission};
        // the host's clock would set the two apart
        let mut system = System::new(Config { clock: Some(0), ..Config::default() });
        let block = system.memory.phys.allocate(memory::MemoryRegion::Application, 0x1000).unwrap();
        system.memory.map(0x0010_0000, block.addr, 0x1000, Permission::RW | Permission::EXECUTE, MemoryState::Code);
        for (address, code) in [(0x0010_0000, &POLLING[..]), (0x0010_0040, &TICKING[..]), (0x0010_0060, &LOCKING[..]), (0x0010_0100, &HOLDING[..]), (0x0010_0180, &TIMING[..]), (0x0010_0200, &WAITING[..]), (0x0010_0280, &NESTING[..]), (0x0010_0300, &SCHEDULING[..]), (0x0010_0380, &POSTING[..]), (0x0010_03C0, &TALLYING[..]), (0x0010_0400, &LEAVING[..]), (0x0010_0480, &TWICE[..]), (0x0010_0500, &TICKER[..]), (0x0010_0540, &SETTLING[..])] {
            for (index, word) in code.iter().enumerate() {
                system.memory.write32(address + index as u32 * 4, *word);
            }
        }
        system.cpu.regs[1] = 0;
        kernel::svc::dispatch(&mut system, 0x13);
        let mutex = system.cpu.regs[1];
        system.memory.write32(MUTEX_HANDLE_AT, mutex);
        if let Some(interval) = timer {
            system.cpu.regs[1] = 1; // sticky
            kernel::svc::dispatch(&mut system, 0x1A);
            let regs = &mut system.cpu.regs;
            (regs[0], regs[2], regs[3], regs[1], regs[4]) = (regs[1], interval, 0, interval, 0);
            kernel::svc::dispatch(&mut system, 0x1B);
        }
        // the one TIMING sets and WAITING waits on
        system.cpu.regs[1] = 0; // one shot
        kernel::svc::dispatch(&mut system, 0x1A);
        let set = system.cpu.regs[1];
        for (index, &(entry, sleep, priority)) in threads.iter().enumerate() {
            let stack = 0x0010_0F00 - index as u32 * 0x100;
            let id = system.kernel.create_thread(format!("thread {index}"), entry, stack, 0, priority, 0);
            system.map_tls_page(id);
            let regs = &mut system.kernel.thread_mut(id).context.regs;
            regs[5] = mutex;
            regs[6] = sleep;
            regs[8] = MUTEX_HANDLE_AT;
            regs[10] = set;
            regs[11] = TARGETS_AT + index as u32 * 16;
        }
        system.kernel.schedule(&mut system.cpu, 0);
        let code: recomp_abi::Code = if in_place { polling::<true> } else { polling::<false> };
        let addresses = [
            0x0010_0000, 0x0010_0004, 0x0010_0020, 0x0010_0024, 0x0010_002C, 0x0010_0038, 0x0010_0040, 0x0010_0044, 0x0010_0060,
            0x0010_007C, 0x0010_0088, 0x0010_008C, 0x0010_0098, 0x0010_00A0, 0x0010_0100, 0x0010_011C, 0x0010_0128, 0x0010_0130,
            0x0010_013C, 0x0010_0144, 0x0010_0180, 0x0010_0198, 0x0010_019C, 0x0010_01A4, 0x0010_0200, 0x0010_0210,
            0x0010_0280, 0x0010_0284, 0x0010_029C, 0x0010_02A8, 0x0010_02AC, 0x0010_02B0, 0x0010_02B8, 0x0010_02C0,
            0x0010_0300, 0x0010_0304, 0x0010_031C, 0x0010_0334, 0x0010_033C, 0x0010_0344, 0x0010_0380, 0x0010_038C,
            0x0010_0390, 0x0010_03C0, 0x0010_03C4, 0x0010_0400, 0x0010_0404, 0x0010_0430, 0x0010_0448, 0x0010_0450,
            0x0010_0480, 0x0010_0484, 0x0010_048C, 0x0010_04A0, 0x0010_04B8, 0x0010_04C0, 0x0010_04C8, 0x0010_0500, 0x0010_0508,
            0x0010_0510, 0x0010_0514, 0x0010_0540, 0x0010_0544, 0x0010_0558, 0x0010_0560,
        ];
        let entries: &'static [recomp_abi::Entry] = Box::leak(Box::from(addresses.map(|address| recomp_abi::Entry { address, code })));
        let abi = if in_place { recomp_abi::ABI } else { recomp_abi::OLDEST_ABI };
        // safe bc it's a table that lives on, of code working on a context like 3dsrecomp's
        let linked = unsafe { recompiled::Linked::new(0, abi, entries.as_ptr().cast(), entries.len() as u32, std::ptr::null(), 0) };
        system.recompiled = Some(recompiled::Library::linked(&linked).unwrap());
        system
    }

    /// runs the threads six frames as code leaving for the step at each svc and as code running them in place
    /// both must leave the machine the same, returns in place svcs, cycles of skipped loop turns and the most rounds a thread did
    fn the_same_both_ways(threads: &[(u32, u32, u32)], timer: Option<u32>) -> (u64, u64, u32) {
        let mut systems = [polling_system(false, threads, timer), polling_system(true, threads, timer)];
        let [steps, in_place] = &mut systems;
        for frame in 0..6 {
            assert_eq!(steps.run_frame(), FrameOutcome::Completed);
            assert_eq!(in_place.run_frame(), FrameOutcome::Completed);
            assert_eq!(in_place.cpu.cycles, steps.cpu.cycles, "{threads:x?} {timer:?} frame {frame}");
            assert_eq!(in_place.history, steps.history, "{threads:x?} {timer:?} frame {frame}");
            assert!(in_place.save_state().unwrap() == steps.save_state().unwrap(), "{threads:x?} {timer:?} elsewhere at frame {frame}");
        }
        // the current thread's regs are the cpu's
        let rounds = steps.kernel.threads.iter().map(|thread| thread.context.regs[9]).chain([steps.cpu.regs[9]]);
        assert_eq!(steps.recompiled.as_ref().unwrap().in_place(), 0);
        assert_eq!(steps.spin.skipped, 0);
        (in_place.recompiled.as_ref().unwrap().in_place(), in_place.spin.skipped, rounds.max().unwrap())
    }

    /// code running its svcs in place leaves the machine exactly like code leaving for the step at each
    /// thru ticks, scheduling passes, yields, sleeps, preemption and frames, just stopping way less
    /// threads, loops, sleeps and priorities vary so the svcs hit every kind of stop
    /// one func runs every label here, cutting the code the same whether entered or continuing
    #[test]
    fn svcs_run_in_place_leave_the_machine_as_steps_would() {
        if !recompiled::svcs_in_place() {
            eprintln!("ZAKURO_SVC_IN_PLACE=0 has svcs leave the code, nothing to compare");
            return;
        }
        let mut seed = 0x2545_F491u32;
        let mut pick = |choices: &[u32]| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            choices[(seed >> 16) as usize % choices.len()]
        };
        let mut ran_in_place = 0;
        for case in 0..60 {
            let mut threads: Vec<(u32, u32, u32)> = (0..1 + case % 3)
                .map(|_| {
                    let entry = pick(&[0x0010_0000, 0x0010_0000, 0x0010_0040, 0x0010_0060, 0x0010_0100, 0x0010_0280, 0x0010_0280]);
                    (entry, pick(&[0, 0, 900, 13_000, 50_000, 400_000]), pick(&[0x2F, 0x30, 0x30, 0x31]))
                })
                .collect();
            // a thread waiting on a timer that the one setting it reads the clock just ahead of, then it comes due in the code
            if case % 5 == 4 {
                threads.push((0x0010_0200, 0, pick(&[0x2F, 0x30, 0x31])));
                threads.push((0x0010_0180, pick(&[2_000, 20_000, 60_000]), pick(&[0x30, 0x31])));
            }
            let timer = (case % 4 == 3).then(|| pick(&[70_000, 300_000, 2_000_000]));
            let (in_place, _, rounds) = the_same_both_ways(&threads, timer);
            assert!(rounds > 100, "{threads:x?} went round");
            ran_in_place += in_place;
        }
        assert!(ran_in_place > 100_000, "{ran_in_place} svcs ran in place");
        // a thread waiting for a time alone or w/ others asleep or waiting (the run skips turns)
        // and w/ others ready or timers due (it doesn't)
        let mut skipped = 0;
        for (threads, timer) in [
            (&[(0x0010_0300, 13_000, 0x30)][..], None),
            (&[(0x0010_0300, 50_000, 0x30)], None),
            (&[(0x0010_0300, 400_000, 0x30)], None),
            (&[(0x0010_0300, 2_000_000, 0x30)], None),
            (&[(0x0010_0300, 900, 0x30)], None),
            (&[(0x0010_0300, 400_000, 0x31), (0x0010_0000, 400_000, 0x30)], None),
            (&[(0x0010_0300, 2_000_000, 0x31), (0x0010_0000, 50_000, 0x2F)], None),
            (&[(0x0010_0300, 50_000, 0x30), (0x0010_0200, 0, 0x2F), (0x0010_0180, 20_000, 0x31)], None),
            (&[(0x0010_0300, 400_000, 0x30), (0x0010_0300, 400_000, 0x30)], None),
            (&[(0x0010_0300, 300_000, 0x30), (0x0010_0300, 2_000_000, 0x31)], None),
            (&[(0x0010_0300, 400_000, 0x30)], Some(300_000)),
            (&[(0x0010_0300, 2_000_000, 0x30)], Some(70_000)),
            // a thread readying the scheduler's task while it waits
            (&[(0x0010_0300, 2_000_000, 0x31), (0x0010_0380, 1_000_000, 0x30)], None),
            (&[(0x0010_0300, 400_000, 0x31), (0x0010_0380, 3_000_000, 0x30)], None),
            // turns the code charges more than the instrs they run
            (&[(0x0010_0400, 50_000, 0x30)], None),
            (&[(0x0010_0400, 2_000_000, 0x30)], None),
            (&[(0x0010_0400, 400_000, 0x31), (0x0010_0000, 400_000, 0x30)], None),
            (&[(0x0010_0400, 2_000_000, 0x31), (0x0010_0380, 1_000_000, 0x30)], None),
            // the clock read twice a turn thru a func
            (&[(0x0010_0480, 50_000, 0x30)], None),
            (&[(0x0010_0480, 2_000_000, 0x30)], None),
            (&[(0x0010_0480, 400_000, 0x31), (0x0010_0000, 400_000, 0x30)], None),
            (&[(0x0010_0480, 2_000_000, 0x31), (0x0010_0380, 1_000_000, 0x30)], None),
            // the turn going another way reads what earlier turns left
            (&[(0x0010_0540, 50_000, 0x30)], None),
            (&[(0x0010_0540, 400_000, 0x30)], None),
            (&[(0x0010_0540, 2_000_000, 0x30)], None),
            (&[(0x0010_0540, 400_000, 0x31), (0x0010_0380, 3_000_000, 0x30)], None),
            // turns that change memory every time
            (&[(0x0010_03C0, 0, 0x30)], None),
            (&[(0x0010_03C0, 0, 0x31), (0x0010_0000, 400_000, 0x30)], None),
        ] {
            let (_, skips, _) = the_same_both_ways(threads, timer);
            skipped += skips;
        }
        assert!(skipped > 1_000_000, "{skipped} cycles of turns skipped");
        let ticked = TICKED.load(std::sync::atomic::Ordering::Relaxed);
        assert!(ticked > 50_000, "the code read {ticked} ticks itself");
        let held = HELD.load(std::sync::atomic::Ordering::Relaxed);
        assert!(held > 50_000, "the code took and let go of the mutex {held} times itself");
    }

    /// runs steps till the run reaches tick or steps run out, returns how many it took
    fn steps_to(system: &mut System, tick: u64, steps: usize) -> usize {
        (0..steps).take_while(|_| {
            system.step(None);
            system.cpu.cycles < tick
        }).count() + 1
    }

    /// a thread alone on the core spinning on nothing waits for the next event (next audio frame here)
    /// instead of the interpreter running every turn till then
    #[test]
    fn a_thread_spinning_alone_skips_to_the_next_event() {
        let mut system = System::new(Config::default());
        one_spinning_thread(&mut system);
        let audio = system.next_audio_frame;
        assert!(steps_to(&mut system, audio, 40_000) < 20_000, "skipped");
        assert_eq!(system.cpu.cycles, audio, "to the audio frame, not past it");
        assert!(system.idle.skipped > 0);
    }

    /// a loop counting in memory changes smth every turn and runs every one
    #[test]
    fn a_loop_that_counts_is_not_skipped() {
        let mut system = System::new(Config::default());
        // ldr r1, [r0], add r1, r1, #1, str r1, [r0], b back
        one_thread_running(&mut system, &[0xE590_1000, 0xE281_1001, 0xE580_1000, 0xEAFF_FFFB]);
        for _ in 0..40_000 {
            system.step(None);
        }
        assert_eq!(system.idle.skipped, 0);
        assert_eq!(system.memory.read32(0x0010_0800), 10_000, "every turn counted");
    }

    /// a loop that puts back what it wrote (like taking and dropping a lock) plus push and pop changes nothing and gets skipped
    #[test]
    fn a_loop_that_puts_back_what_it_wrote_is_skipped() {
        let mut system = System::new(Config::default());
        // ldr r1, [r0], add r1, r1, #1, str r1, [r0], sub r1, r1, #1,
        // str r1, [r0], push {r4, lr}, pop {r4, lr}, b back
        let code = [0xE590_1000, 0xE281_1001, 0xE580_1000, 0xE241_1001, 0xE580_1000, 0xE92D_4010, 0xE8BD_4010, 0xEAFF_FFF7];
        one_thread_running(&mut system, &code);
        system.memory.write32(0x0010_0800, 7);
        let audio = system.next_audio_frame;
        assert!(steps_to(&mut system, audio, 40_000) < 20_000, "skipped");
        assert_eq!(system.memory.read32(0x0010_0800), 7);
    }

    /// a skip stops where a sleeping thread wakes, it gets the core then
    #[test]
    fn a_skip_stops_where_a_sleeping_thread_wakes() {
        let mut system = System::new(Config::default());
        one_spinning_thread(&mut system);
        let sleeper = system.kernel.create_thread("sleeper", 0x0010_0000, 0x0010_0E00, 0, 0x20, 0);
        system.map_tls_page(sleeper);
        let thread = system.kernel.thread_mut(sleeper);
        thread.status = ThreadStatus::Sleeping;
        thread.wakeup_at = Some(20_000);
        let steps = (0..40_000).take_while(|_| {
            system.step(None);
            system.kernel.current_thread != Some(sleeper)
        });
        assert!(steps.count() < 20_000, "skipped");
        assert!((20_000..20_008).contains(&system.cpu.cycles), "woke at {}", system.cpu.cycles);
    }

    /// waits on a handle forever, like svcWaitSynchronization1
    fn wait_on(system: &mut System, handle: u32) {
        system.cpu.regs[0] = handle;
        system.cpu.regs[2] = u32::MAX;
        system.cpu.regs[3] = u32::MAX;
        kernel::svc::dispatch(system, 0x24);
    }

    /// steps till the thread runs again, returns the tick it does
    fn runs_again(system: &mut System, id: ThreadId) -> u64 {
        for _ in 0..1000 {
            if system.kernel.current_thread == Some(id) && system.kernel.thread(id).status == ThreadStatus::Running {
                return system.cpu.cycles;
            }
            system.step(None);
        }
        panic!("the thread never ran again");
    }

    /// a wait the very next schedule ends, on a thread keeping the core bc nothing else can run
    /// returns its result b4 the thread goes on, not the handle it got (overwritten later)
    #[test]
    fn a_wait_the_next_schedule_ends_returns_its_result() {
        let mut system = System::new(Config::default());
        let id = one_spinning_thread(&mut system);
        system.cpu.regs[1] = 0;
        kernel::svc::dispatch(&mut system, 0x17);
        let event = system.cpu.regs[1];
        wait_on(&mut system, event);
        assert_eq!(system.kernel.thread(id).status, ThreadStatus::WaitSync);
        let object = system.kernel.resolve(event).unwrap();
        system.kernel.signal_event(object);
        system.step(None);
        assert_eq!(system.kernel.current_thread, Some(id));
        assert_eq!(system.cpu.regs[0], 0, "the wait's result, not the handle");
    }

    /// the top screen's 2nd buffer is the right eye's only while its format asks for 3D like the console's GSP takes it
    /// bit 5 set, bit 6 clear, and the title gave it one
    #[test]
    fn the_right_eye_shows_only_while_the_title_asks_for_3d() {
        let mut system = System::new(Config::default());
        let config = &mut system.gpu.framebuffers[0];
        config.address_a_left = 0x1830_0000;
        config.address_a_right = 0x1834_6500;
        for (format, right) in [(0x20, true), (0x60, false), (0x40, false), (0x00, false)] {
            system.gpu.framebuffers[0].format = 0x0300 | format | 1;
            assert_eq!(system.draws_in_3d(), right, "format 0x{format:X}");
        }
        system.gpu.framebuffers[0].address_a_right = 0;
        system.gpu.framebuffers[0].format = 0x0321;
        assert!(!system.draws_in_3d(), "no buffer for the right eye");
        // the slider games read moves w/ it
        system.set_slider_3d(0.5);
        let page = system.memory.phys.shared_page_mut();
        assert_eq!((f32::from_le_bytes(page[0x80..0x84].try_into().unwrap()), page[0x84]), (0.5, 0), "the LED not forced off");
    }

    /// a zero sleep w/ nobody else ready is a no op, the caller keeps the core and no pass is due
    /// titles poll a worker in a tight svcSleepThread(0) loop, paying a full pass each time was most of what one such title spent
    #[test]
    fn a_zero_sleep_with_nobody_else_ready_does_not_schedule() {
        let mut system = System::new(Config::default());
        let id = one_spinning_thread(&mut system);
        system.kernel.reschedule_pending = false;
        system.cpu.regs[0] = 0;
        system.cpu.regs[1] = 0;
        kernel::svc::dispatch(&mut system, 0x0A);
        assert_eq!(system.kernel.thread(id).status, ThreadStatus::Running, "it keeps the core");
        assert!(!system.kernel.reschedule_pending, "no pass is due");
        assert_eq!(system.cpu.regs[0], 0);
    }

    /// a zero sleep w/ another thread ready still steps aside, so a title polling a worker keeps handing it the core
    #[test]
    fn a_zero_sleep_with_another_ready_steps_aside() {
        let mut system = System::new(Config::default());
        let first = one_spinning_thread(&mut system);
        let second = system.kernel.create_thread("second", 0x0010_0000, 0x0FF0_0000, 0, 0x30, 0);
        system.map_tls_page(second);
        system.kernel.thread_mut(second).status = ThreadStatus::Ready;
        system.cpu.regs[0] = 0;
        system.cpu.regs[1] = 0;
        kernel::svc::dispatch(&mut system, 0x0A);
        assert_eq!(system.kernel.thread(first).status, ThreadStatus::Sleeping, "it steps aside");
        assert!(system.kernel.reschedule_pending, "the ready thread gets a pass");
    }

    /// a zero sleep while a thread is due to wake still gets the pass then, so it wakes on time not at the next preempt
    #[test]
    fn a_zero_sleep_wakes_a_thread_that_is_due() {
        let mut system = System::new(Config::default());
        let first = one_spinning_thread(&mut system);
        let second = system.kernel.create_thread("second", 0x0010_0000, 0x0FF0_0000, 0, 0x30, 0);
        system.map_tls_page(second);
        let thread = system.kernel.thread_mut(second);
        thread.status = ThreadStatus::Sleeping;
        thread.wakeup_at = Some(system.cpu.cycles);
        system.kernel.reschedule_pending = false;
        system.cpu.regs[0] = 0;
        system.cpu.regs[1] = 0;
        kernel::svc::dispatch(&mut system, 0x0A);
        assert_eq!(system.kernel.thread(first).status, ThreadStatus::Sleeping, "it steps aside for the pass");
        assert!(system.kernel.reschedule_pending, "the sleeper due gets a pass");
    }

    /// a service request makes the thread wait for the reply while a lower thread runs, like on the console
    /// Star Fox 64 3D marks a save job running once the more urgent thread doing it exists,
    /// which must be waiting on its first request by then
    #[test]
    fn a_request_lets_the_threads_below_run_until_the_reply() {
        let mut system = System::new(Config::default());
        let id = one_spinning_thread(&mut system);
        let below = system.kernel.create_thread("below", 0x0010_0000, 0x0FF0_0000, 0, 0x31, 0);
        system.map_tls_page(below);
        let port = system.kernel.objects.insert(KObject::ClientPort("srv:".into()));
        let handle = system.kernel.handles.create(&mut system.kernel.objects, port, "srv:");
        // srv:RegisterClient
        let buffer = system.kernel.thread(id).tls + 0x80;
        system.memory.write32(buffer, 0x0001_0002);
        system.memory.write32(buffer + 4, 0x20);
        system.cpu.regs[0] = handle;
        let asked = system.cpu.cycles;
        kernel::svc::dispatch(&mut system, 0x32);
        assert_eq!((system.cpu.regs[0], system.memory.read32(buffer + 4)), (0, 0), "answered");
        assert_eq!(system.kernel.thread(id).status, ThreadStatus::Sleeping, "waiting for the reply");

        assert!(runs_again(&mut system, below) < asked + 100, "the thread below runs at once");
        let mut steps = 0;
        while system.kernel.current_thread != Some(id) || system.kernel.thread(id).status != ThreadStatus::Running {
            system.step(None);
            steps += 1;
            assert!(steps < 40_000, "the thread never got its reply");
        }
        let due = asked + kernel::svc::REQUEST_TICKS;
        assert!((due..due + PREEMPT_INTERVAL).contains(&system.cpu.cycles), "replied at {}, due at {due}", system.cpu.cycles);
    }

    /// a timer fires at its tick not the next frame, and its waiter wakes then
    /// a pulse timer is gone once it woke a waiter and comes back a period after it was due
    #[test]
    fn a_timer_fires_at_its_tick() {
        let mut system = System::new(Config::default());
        let id = one_spinning_thread(&mut system);
        // a pulse timer due in a ms and every ms after
        system.cpu.regs[1] = 2;
        kernel::svc::dispatch(&mut system, 0x1A);
        let timer = system.cpu.regs[1];
        system.cpu.regs[0] = timer;
        system.cpu.regs[2] = 1_000_000;
        system.cpu.regs[3] = 0;
        system.cpu.regs[1] = 1_000_000;
        system.cpu.regs[4] = 0;
        kernel::svc::dispatch(&mut system, 0x1B);
        let period = kernel::thread::nanos_to_ticks(1_000_000);

        wait_on(&mut system, timer);
        let woke = runs_again(&mut system, id);
        assert!((period..period + PREEMPT_INTERVAL).contains(&woke), "woke at {woke}, due at {period}");
        assert_eq!(system.cpu.regs[0], 0);

        wait_on(&mut system, timer);
        assert_eq!(system.kernel.thread(id).status, ThreadStatus::WaitSync, "the pulse is gone");
        let woke = runs_again(&mut system, id);
        assert!((2 * period..2 * period + PREEMPT_INTERVAL).contains(&woke), "woke at {woke}, due at {}", 2 * period);
    }

    /// a pulse timer due in a ms and every ms after
    fn pulse_timer(system: &mut System) -> u32 {
        system.cpu.regs[1] = 2;
        kernel::svc::dispatch(system, 0x1A);
        let timer = system.cpu.regs[1];
        system.cpu.regs[0] = timer;
        system.cpu.regs[2] = 1_000_000;
        system.cpu.regs[3] = 0;
        system.cpu.regs[1] = 1_000_000;
        system.cpu.regs[4] = 0;
        kernel::svc::dispatch(system, 0x1B);
        timer
    }

    /// a pulse timer releases every waiter when it fires, not just the first, and a fire nobody waits for is lost
    #[test]
    fn a_pulse_timer_wakes_every_waiter_then_clears() {
        let mut system = System::new(Config::default());
        let first = one_spinning_thread(&mut system);
        let second = system.kernel.create_thread("second", 0x0010_0000, 0x0FF0_0000, 0, 0x30, 0);
        system.map_tls_page(second);
        let timer = pulse_timer(&mut system);
        let period = kernel::thread::nanos_to_ticks(1_000_000);
        for id in [first, second] {
            system.kernel.current_thread = Some(id);
            system.kernel.thread_mut(id).status = ThreadStatus::Running;
            wait_on(&mut system, timer);
            assert_eq!(system.kernel.thread(id).status, ThreadStatus::WaitSync);
        }
        system.kernel.current_thread = None;
        while system.cpu.cycles < period {
            system.step(None);
        }
        system.step(None);
        for id in [first, second] {
            assert_ne!(system.kernel.thread(id).status, ThreadStatus::WaitSync, "thread {id} woke");
        }

        // the next fire comes w/ nobody waiting and is gone after
        system.cpu.cycles = 2 * period + 10;
        system.kernel.current_thread = Some(first);
        system.kernel.thread_mut(first).status = ThreadStatus::Running;
        wait_on(&mut system, timer);
        assert_eq!(system.kernel.thread(first).status, ThreadStatus::WaitSync, "the fire nothing waited for is lost");
    }

    /// cancelling a timer that came due b4 the scheduler looked keeps its fire, the console's already happened
    #[test]
    fn cancelling_a_timer_due_keeps_its_fire() {
        let mut system = System::new(Config::default());
        let id = one_spinning_thread(&mut system);
        system.cpu.regs[1] = 0;
        kernel::svc::dispatch(&mut system, 0x1A);
        let timer = system.cpu.regs[1];
        system.cpu.regs[0] = timer;
        system.cpu.regs[2] = 1_000_000;
        system.cpu.regs[3] = 0;
        system.cpu.regs[1] = 0;
        system.cpu.regs[4] = 0;
        kernel::svc::dispatch(&mut system, 0x1B);
        let other = system.kernel.create_thread("other", 0x0010_0000, 0x0FF0_0000, 0, 0x30, 0);
        system.map_tls_page(other);
        wait_on(&mut system, timer);
        // the other thread cancels it after it came due
        system.cpu.cycles = kernel::thread::nanos_to_ticks(1_000_000) + 100;
        system.kernel.current_thread = Some(other);
        system.kernel.thread_mut(other).status = ThreadStatus::Running;
        system.cpu.regs[0] = timer;
        kernel::svc::dispatch(&mut system, 0x1C);
        let tick = system.cpu.cycles;
        system.kernel.schedule(&mut system.cpu, tick);
        assert_ne!(system.kernel.thread(id).status, ThreadStatus::WaitSync, "the waiter woke");
    }

    /// closing a handle while a thread waits on its object neither wakes the thread as if signalled
    /// nor hands the object's slot to the next one made, the object goes once the wait ends
    #[test]
    fn closing_what_a_thread_waits_on_keeps_it_waiting() {
        let mut system = System::new(Config::default());
        let id = one_spinning_thread(&mut system);
        system.cpu.regs[1] = 0;
        kernel::svc::dispatch(&mut system, 0x17);
        let event = system.cpu.regs[1];
        let object = system.kernel.resolve(event).unwrap();
        // for a ms
        system.cpu.regs[0] = event;
        system.cpu.regs[2] = 1_000_000;
        system.cpu.regs[3] = 0;
        kernel::svc::dispatch(&mut system, 0x24);
        let other = system.kernel.create_thread("other", 0x0010_0000, 0x0FF0_0000, 0, 0x30, 0);
        system.map_tls_page(other);
        system.kernel.current_thread = Some(other);
        system.kernel.thread_mut(other).status = ThreadStatus::Running;
        system.cpu.regs[0] = event;
        kernel::svc::dispatch(&mut system, 0x23);
        let tick = system.cpu.cycles;
        system.kernel.schedule(&mut system.cpu, tick);
        assert_eq!(system.kernel.thread(id).status, ThreadStatus::WaitSync, "not woken as if signalled");
        system.cpu.regs[1] = 0;
        kernel::svc::dispatch(&mut system, 0x17);
        assert_ne!(system.kernel.resolve(system.cpu.regs[1]), Some(object));

        let late = kernel::thread::nanos_to_ticks(1_000_000) + 1;
        system.kernel.schedule(&mut system.cpu, late);
        assert!(matches!(system.kernel.thread(id).wait_result, Some(kernel::thread::WaitResult::TimedOut)));
        assert!(system.kernel.objects.get(object).is_none(), "gone after the wait");
    }

    /// a pulse event wakes its waiters when signalled and is gone after, the next wait waits
    #[test]
    fn a_pulse_event_does_not_stay_signalled() {
        let mut system = System::new(Config::default());
        let id = one_spinning_thread(&mut system);
        system.cpu.regs[1] = 2;
        kernel::svc::dispatch(&mut system, 0x17);
        let event = system.cpu.regs[1];
        wait_on(&mut system, event);
        let object = system.kernel.resolve(event).unwrap();
        system.kernel.signal_event(object);
        runs_again(&mut system, id);
        wait_on(&mut system, event);
        assert_eq!(system.kernel.thread(id).status, ThreadStatus::WaitSync);
    }

    /// an arbiter wait's timeout takes r5 as its high word, minus 1 waits forever
    #[test]
    fn an_arbiter_timeout_is_64_bits() {
        let mut system = System::new(Config::default());
        let id = one_spinning_thread(&mut system);
        kernel::svc::dispatch(&mut system, 0x21);
        let arbiter = system.cpu.regs[1];
        let address = system.kernel.thread(id).tls + 0x100;
        system.memory.write32(address, 0);
        for (low, high, wakes) in [(u32::MAX, u32::MAX, false), (0x2A05_F200, 1, true)] {
            system.kernel.thread_mut(id).status = ThreadStatus::Running;
            system.cpu.regs[0] = arbiter;
            system.cpu.regs[1] = address;
            // wait if less than, w/ a timeout
            system.cpu.regs[2] = 3;
            system.cpu.regs[3] = 1;
            system.cpu.regs[4] = low;
            system.cpu.regs[5] = high;
            kernel::svc::dispatch(&mut system, 0x22);
            let wakeup = system.kernel.thread(id).wakeup_at;
            assert_eq!(wakeup.is_some(), wakes);
            if wakes {
                // five secs
                let five = kernel::thread::nanos_to_ticks(5_000_000_000);
                assert!(wakeup.unwrap() >= five, "{wakeup:?}");
            }
        }
    }

    /// a thread whose timed arbiter wait ran out and then ended left its entry on the arbiter
    /// a signal for one waiter skips it for the thread really waiting there, and the dead one stays dead
    #[test]
    fn an_arbiter_signal_wakes_only_threads_still_waiting() {
        let mut system = System::new(Config::default());
        let ended = system.kernel.create_thread("ended", 0x0010_0000, 0x1000_0000, 0, 0x30, 0);
        let signaller = system.kernel.create_thread("signaller", 0x0010_0000, 0x0FF0_0000, 0, 0x30, 0);
        let parked = system.kernel.create_thread("parked", 0x0010_0000, 0x0FE0_0000, 0, 0x30, 0);
        for id in [ended, signaller, parked] {
            system.map_tls_page(id);
        }
        system.kernel.schedule(&mut system.cpu, 0);
        assert_eq!(system.kernel.current_thread, Some(ended));
        kernel::svc::dispatch(&mut system, 0x21);
        let arbiter = system.cpu.regs[1];
        let address = system.kernel.thread(ended).tls + 0x100;
        system.memory.write32(address, 0);
        let arbitrate = |system: &mut System, who: ThreadId, kind: u32| {
            system.kernel.current_thread = Some(who);
            system.kernel.thread_mut(who).status = ThreadStatus::Running;
            system.cpu.regs[0] = arbiter;
            system.cpu.regs[1] = address;
            system.cpu.regs[2] = kind;
            system.cpu.regs[3] = 1;
            system.cpu.regs[4] = 1000;
            kernel::svc::dispatch(system, 0x22);
        };
        // waits if less than 1 for a microsecond and times out
        arbitrate(&mut system, ended, 3);
        system.kernel.current_thread = None;
        system.kernel.schedule(&mut system.cpu, 1_000_000);
        assert!(matches!(system.kernel.thread(ended).wait_result, Some(WaitResult::TimedOut)));
        // then ends
        system.kernel.current_thread = Some(ended);
        system.kernel.thread_mut(ended).status = ThreadStatus::Running;
        kernel::svc::dispatch(&mut system, 0x09);
        // another waits forever, a third signals one waiter
        arbitrate(&mut system, parked, 1);
        assert_eq!(system.kernel.thread(parked).status, ThreadStatus::WaitArbiter);
        arbitrate(&mut system, signaller, 0);
        assert_eq!(system.kernel.thread(ended).status, ThreadStatus::Dead);
        assert_eq!(system.kernel.thread(parked).status, ThreadStatus::Ready);
    }
}
