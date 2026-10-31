//! the emulated console, CPU, memory, HLE kernel, HLE services and GPU, and
//! the loop that drives them.

pub mod camera;
pub mod cheats;
pub mod enhancements;
pub mod cro;
pub mod download_play;
pub mod hints;
pub mod idle;
pub mod kernel;
pub mod loader;
pub mod memory;
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

/// console settings the frontend can change.
#[derive(Debug, Clone)]
pub struct Config {
    pub model: ConsoleModel,
    pub new3ds: bool,
    /// 0 = Japan, 1 = USA, 2 = Europe.
    pub region: u8,
    pub language: u8,
    pub slider_3d: f32,
    /// a library 3dsrecomp built for the title, or a directory holding one
    /// named after its title id.
    pub recompiled: Option<std::path::PathBuf>,
    /// recompiled code linked into the program, which comes before a
    /// library.
    pub linked: Option<recompiled::Linked>,
    /// with no library given, run the one 3dsrecomp build installed for the
    /// title, if it did.
    pub find_recompiled: bool,
    /// where saves and dumped system files live, the working directory
    /// when there is none.
    pub data_dir: Option<std::path::PathBuf>,
    /// draw on the host's GPU through Vulkan, when there is one that can.
    pub hardware_renderer: bool,
    /// draw the pictures of the texture pack in the title's mods folder in
    /// place of its textures, which takes the host's GPU.
    pub texture_packs: bool,
    /// the title's update, an update's CIA, whose code and data it runs.
    pub update: Option<std::path::PathBuf>,
    /// the title's downloadable content, DLC CIAs.
    pub dlc: Vec<std::path::PathBuf>,
    /// the enhancements turned on, by name, for each title by its id.
    pub enhancements: std::collections::BTreeMap<u64, Vec<String>>,
    /// how many times the console's resolution the host's GPU draws at.
    pub resolution: u32,
    /// the Vulkan presenter's device, to draw on and show the screens
    /// straight from, when it can.
    pub device: Option<std::sync::Arc<zakuro_gpu::SharedDevice>>,
    /// the time the console's clock starts at, in milliseconds since 1900,
    /// the host's when none, a fixed one makes runs repeat exactly.
    pub clock: Option<u64>,
    /// draw a picture in between each two of a title that draws one every
    /// other refresh, see System::pace, when the host's GPU draws the 3D
    /// and shows the screens straight from its device.
    pub interpolate: bool,
    /// the console owner's name, up to ten characters, which titles show
    /// and local wireless tells others.
    pub username: String,
    /// the number that tells this console apart from others over local
    /// wireless, the configuration's block 0x00090000.
    pub console_id: u64,
    /// the game's Download Play child decrypted, which a host sends its
    /// guests. the one on a card is encrypted.
    pub dlp_child: Option<std::path::PathBuf>,
    /// what a Download Play child runs with, the host's address and the
    /// passphrase of its network, APT:GetWirelessRebootInfo's answer.
    pub wireless_reboot: Option<[u8; 16]>,
    /// the interpreter skips to the next event past loops that change
    /// nothing, see idle.
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

/// how a frame of emulation ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameOutcome {
    /// a full frame's worth of cycles ran.
    Completed,
    /// the title called svcExitProcess.
    Exited,
    /// the title called svcBreak, or hit an instruction we do not have.
    Faulted,
}

/// a screen's picture as RGBA, and its width and height.
pub type Screen = (std::sync::Arc<Vec<u8>>, u32, u32);

/// frame interpolation's count of the refreshes, see System::pace.
#[derive(Default)]
struct Pacing {
    refreshes: u64,
    /// the refresh the top screen got its last new picture at, and that
    /// picture.
    new_at: u64,
    last: Option<zakuro_gpu::ScreenRef>,
    /// how the draws of the frame before it matched the frame before that.
    matches: Option<zakuro_gpu::Matches>,
    /// the screens show the pictures in between at this refresh, and the
    /// title's own at the next.
    between: bool,
    /// the refreshes that showed pictures in between so far.
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
    /// the title's downloadable content.
    pub dlc: Vec<zakuro_fs::Dlc>,
    /// the title's cheats, from its file in the data folder.
    pub cheats: Vec<cheats::Cheat>,
    pub config: Config,

    pub exited: bool,
    pub broke: bool,
    /// everything the title wrote with svcOutputDebugString.
    pub debug_output: String,
    pub unimplemented_svcs: BTreeSet<u32>,
    pub services_seen: BTreeSet<String>,
    pub lcd_force_black: bool,
    /// the scaled picture each screen had at the last presentation, which
    /// goes up at the next, once the GPU had a frame's time to draw it.
    showing: [Option<zakuro_gpu::ScreenRef>; 3],
    /// frame interpolation's count of the refreshes.
    pacing: Pacing,
    /// the dynamic module loader's state.
    pub cro: cro::CroManager,
    /// what the cameras see, from the frontend.
    pub camera: Option<Box<dyn camera::Camera>>,
    /// the room local wireless goes through, none with wireless off.
    pub room: Option<room::Link>,
    /// errors the title reported through err:f, newest last.
    pub fatal_errors: Vec<String>,
    /// undefined instructions we have seen, so the log stays readable.
    undefined_seen: BTreeMap<u32, u32>,
    /// branches into unmapped memory, keyed by target.
    pub prefetch_aborts: BTreeMap<u32, u32>,
    /// the last few instruction addresses, so a fault can say how it got there.
    history: [u32; HISTORY_LENGTH],
    history_index: usize,
    /// frames completed since boot.
    pub frames: u64,
    /// the console's clock at boot, in milliseconds since 1900.
    pub(crate) boot_clock: u64,
    /// cycle count at which the next end-of-frame work is due.
    next_frame_boundary: u64,
    /// cycle count at which the DSP next finishes an audio frame.
    next_audio_frame: u64,
    /// cycle count at which the scheduler is next forced to run.
    next_preempt: u64,

    /// sampling profiler, counts how often each thread was found at each PC.
    pub profile: Option<BTreeMap<(ThreadId, u32), u64>>,
    /// steps since the profiler last looked.
    profile_skip: u32,

    /// loops that change nothing, which the interpreter skips past.
    pub idle: idle::IdleLoops,

    /// code recompiled ahead of time, which runs instead of the interpreter
    /// wherever it has something.
    pub recompiled: Option<recompiled::Library>,
    /// where the title ran in the interpreter despite the library.
    pub hints: Option<hints::Hints>,
    /// instructions the interpreter ran for want of recompiled code, and
    /// the ones recompiled code ran, to see how much the library covers.
    pub interpreted_instructions: u64,
    pub recompiled_instructions: u64,
}

/// cycles in one frame of the screens, which refresh at 268111856 / 4481136,
/// about 59.83 Hz and not 60. a title that keeps time by frames runs that
/// much ahead of its sound otherwise, a rhythm game's notes drift off the
/// music over a song.
pub const CYCLES_PER_FRAME: u64 = 4_481_136;

/// cycles in one audio frame, 160 samples, each exactly 8192 cycles long
/// (the DSP's 32728 Hz is the CPU clock divided by 8192).
pub const CYCLES_PER_AUDIO_FRAME: u64 = 160 * 8192;
/// samples the DSP plays each second, one every 8192 cycles.
pub const AUDIO_SAMPLE_RATE: f64 = kernel::thread::CPU_CLOCK_HZ as f64 / 8192.0;

/// how often the scheduler is forced to run even if no thread yields.
pub const PREEMPT_INTERVAL: u64 = 8192;

/// how many instruction addresses are kept for fault reports.
const HISTORY_LENGTH: usize = 64;

/// what one call to [System::step] did.
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
            recompiled: None,
            hints: None,
            interpreted_instructions: 0,
            recompiled_instructions: 0,
        }
    }

    // -- process setup ------------------------------------------------------

    /// maps the page holding a thread's TLS block, if it is not mapped yet,
    /// and clears the block, which a thread that ended may have used.
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
        // TLS must start zeroed, the IPC command buffer lives in it.
        self.memory.zero_physical(block.addr, PAGE_SIZE);
    }

    // -- the run loop -------------------------------------------------------

    /// runs one frame's worth of emulation.
    pub fn run_frame(&mut self) -> FrameOutcome {
        // HID samples its inputs on its own, whether or not anyone pressed
        // anything
        let input = self.services.hid.input;
        self.idle_interrupted(false);
        services::hid::update(self, input);
        cheats::run(self);
        // run up to the boundary step() would end the frame at, and end it
        // here, once, instead of again when the next frame's first step
        // finds the boundary passed
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

    /// advances the machine by one instruction, doing whatever scheduling is
    /// due first.
    pub fn step(&mut self, deadline: Option<u64>) -> StepOutcome {
        if self.exited {
            return StepOutcome::Exited;
        }
        if self.broke {
            return StepOutcome::Faulted;
        }

        // fire the end-of-frame work on schedule even when a tool is stepping,
        // or a title waiting on vertical blank would never be woken.
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
            // everything is blocked.
            let mut limit = deadline.unwrap_or(self.cpu.cycles + CYCLES_PER_FRAME);
            // an audio thread waits on the DSP's interrupt, which only fires
            // at an audio frame boundary, skipping past one would lose it.
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

        // a thread whose entry point returned branches to this address.
        if self.cpu.regs[15] == THREAD_EXIT_MAGIC {
            if let Some(id) = self.kernel.current_thread {
                self.kernel.end_thread(id);
            }
            return StepOutcome::Ran;
        }

        // a branch through an uninitialized function pointer lands on the null
        // page, whose zero words decode as harmless no-ops.
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
            // stop the thread rather than the machine, the rest of the title
            // may still make progress, and the report says what happened.
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
                // a thread polling in a loop that changes nothing waits for
                // the next event, as it would on a console spinning freely
                if let Some(until) = self.idle.due(self.cpu.cycles).then(|| self.idle_until(deadline)).flatten() {
                    self.cpu.cycles = until;
                    self.kernel.reschedule_pending = true;
                    return StepOutcome::Ran;
                }
                self.interpreted_instructions += 1;
                // past the start of the step when the code ran svcs in place
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
                // skip it and keep going, one bad decode should not end the
                // session while the CPU is still being filled in.
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

    /// runs recompiled code from the pc up to the next thing the scheduler
    /// has to look at, when the library has code there. what it returns
    /// stands in for what one interpreted step would.
    fn run_recompiled(&mut self, deadline: Option<u64>) -> Option<Option<Exit>> {
        let library = self.recompiled.as_ref()?;
        if !library.has_code(self.cpu.regs[15] | self.cpu.cpsr.thumb as u32) {
            return None;
        }
        let Some((budget, event)) = self.recompiled_budget(deadline) else {
            return Some(None);
        };
        // the library stays out of the system while its code runs, so that
        // an svc the code runs in place gets the system to itself
        let library = self.recompiled.take()?;
        self.sync_gpu_for_cpu();
        let (ran, stop, event) = library.run(self, budget, deadline, event);
        self.memory.clear_gpu_sync();
        self.recompiled = Some(library);
        self.recompiled_instructions += ran;
        if event.is_some_and(|tick| self.cpu.cycles >= tick) {
            self.kernel.reschedule_pending = true;
        }
        if (ran == 0 && matches!(stop, recompiled::Stop::Left)) || matches!(stop, recompiled::Stop::Declined) {
            // not enough budget left for a whole block, at the start or
            // after an svc the code ran in place, whose step this is now
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

    /// how many cycles recompiled code can run from here before the
    /// scheduler has to look, and the tick of the next thing a thread waits
    /// for. none when that is due already, which has the scheduler look
    /// first.
    fn recompiled_budget(&mut self, deadline: Option<u64>) -> Option<(u64, Option<u64>)> {
        let mut limit = self.next_frame_boundary.min(self.next_audio_frame);
        // a thread or a timer due wants the scheduler at its tick, a run
        // that reached it has it look first
        let event = self.kernel.next_event();
        if let Some(tick) = event {
            if tick <= self.cpu.cycles {
                self.kernel.reschedule_pending = true;
                return None;
            }
            limit = limit.min(tick);
        }
        // a stop for the scheduler unwinds every guest call the code is in,
        // which the host then enters again one at a time. it only changes
        // anything when another thread can run
        if self.kernel.others_runnable() {
            limit = limit.min(self.next_preempt);
        }
        if let Some(deadline) = deadline {
            limit = limit.min(deadline);
        }
        Some((limit.saturating_sub(self.cpu.cycles).max(1), event))
    }

    /// runs an svc that recompiled code made, from inside the code, and
    /// lets the code go on after it, for as long as the result says, when
    /// the next step would only run that code on. it does what the rest of
    /// this step and the start of the next would do, in the same order, and
    /// gives none wherever they do more, end a frame or switch threads say,
    /// which the code stops for, leaving the rest to the steps. the code
    /// then charges what follows to the budget as the functions it is in cut
    /// it, where a stop has it entered again through the function an entry
    /// names, whose blocks can be cut otherwise.
    pub(crate) fn svc_in_place(
        &mut self,
        number: u32,
        event: Option<u64>,
        deadline: Option<u64>,
        library: &recompiled::Library,
    ) -> Option<(u64, Option<u64>)> {
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
        // and what the next one does before it runs code
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
        // the same thread, still after the svc, which is code the library has
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

    /// the scheduling pass a step makes first when it is due.
    fn reschedule(&mut self) {
        self.kernel.reschedule_pending = false;
        let tick = self.cpu.cycles;
        if self.kernel.schedule(&mut self.cpu, tick) {
            self.idle_interrupted(true);
        }
        // a thread released here that keeps the core gets its result
        // too, before it runs on, not over whatever it does next
        self.apply_wait_result();
    }

    /// notes where a step runs the guest from, for fault reports and the
    /// profiler.
    fn note(&mut self, pc: u32) {
        self.history[self.history_index] = pc | self.cpu.cpsr.thumb as u32;
        self.history_index = (self.history_index + 1) % HISTORY_LENGTH;
        if self.profile.is_some() {
            self.sample();
        }
    }

    /// what a step does after recompiled code ran.
    fn ran_recompiled(&mut self) {
        // its writes do not go through the log a watch keeps
        self.idle_interrupted(false);
        if let Some(hints) = &mut self.hints {
            hints.library_ran();
        }
    }

    /// gives the scheduler a chance on a regular cadence, so that a thread
    /// which never makes a syscall cannot monopolise the core.
    fn preempt(&mut self) {
        if self.cpu.cycles >= self.next_preempt {
            self.next_preempt = self.cpu.cycles + PREEMPT_INTERVAL;
            self.kernel.reschedule_pending = true;
        }
    }

    /// has a read of the CPU from where the GPU drew bring that down first,
    /// until clear_gpu_sync.
    fn sync_gpu_for_cpu(&mut self) {
        let gpu = &mut self.gpu as *mut Gpu as *mut ();
        let linear_base = self.kernel.linear_base;
        // SAFETY: the GPU stays where it is, and nothing but the CPU's reads
        // and writes follow the pointer until the sync is cleared
        unsafe { self.memory.set_gpu_sync(memory::GpuSync { gpu, linear_base, sync: sync_for_cpu, sync_write: sync_for_cpu_write }) };
    }

    /// the instruction addresses executed most recently, oldest first.
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

    /// starts collecting PC samples.
    pub fn enable_profiler(&mut self) {
        self.profile = Some(BTreeMap::new());
    }

    /// the hottest sampled locations, most frequent first.
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

    /// how many samples found each thread running, most first.
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
        // one step in so many, a prime so it does not beat in step with a
        // loop, keeps a profile from slowing the run down
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

    /// writes back the result registers of a syscall that had blocked.
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

    /// everything that happens between frames, vertical blank, input, clock.
    fn end_frame(&mut self) {
        self.idle_interrupted(false);
        self.frames += 1;
        if self.frames.is_multiple_of(hints::SAVE_EVERY) {
            if let Some(hints) = &mut self.hints {
                hints.save();
            }
        }

        // refresh the kernel's shared page so the guest's clock advances.
        let tick = self.cpu.cycles;
        memory::config::update_datetime(self.memory.phys.shared_page_mut(), self.boot_clock, tick);

        // gather the timers again, without the ones closed since. they fire
        // when the scheduler gets to their tick
        self.kernel.timers.clear();
        for (id, object) in self.kernel.objects.iter() {
            if let kernel::object::KObject::Timer(_) = object {
                self.kernel.timers.push(id);
            }
        }

        self.renderer.end_frame();

        // pick up any buffer swap the game queued directly in GSP shared
        // memory before the LCDs latch whatever is currently configured.
        services::gsp::refresh(self);
        if self.gpu.interpolates() {
            self.pace();
        }

        // both LCDs finish scanning out, in that order.
        services::gsp::signal_interrupt(self, services::gsp::InterruptId::Pdc0);
        services::gsp::signal_interrupt(self, services::gsp::InterruptId::Pdc1);

    }

    /// frame interpolation at a refresh. while a title sets a new picture
    /// every other refresh, the GPU draws the picture in between along with
    /// each, from the frame before's draws and this one's, see zakuro_gpu's
    /// interpolation. it goes up at the refresh the title set the picture
    /// at, and the title's own at the next, a refresh late, so the two take
    /// a refresh each. a title at the full rate, or slower, shows its own
    /// at once, and draws nothing in between.
    fn pace(&mut self) {
        let pacing = &mut self.pacing;
        pacing.refreshes += 1;
        pacing.between = false;
        let top = self.screen_ref(zakuro_common::Screen::Top, false);
        let pacing = &mut self.pacing;
        if top.is_none() || top == pacing.last {
            if pacing.refreshes - pacing.new_at > 2 {
                // the title slowed down or stopped, the pictures in between
                // of its next frame would be shown too late
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
        // the next frame is drawn in between too while the title keeps to
        // every other refresh
        self.gpu.set_twins(gap == 2);
    }

    /// draws a picture in between each two of the title's own from now on,
    /// when it draws one every other refresh, see pace, or stops. it says
    /// whether it does, it takes the host's GPU showing the screens.
    pub fn set_interpolation(&mut self, on: bool) -> bool {
        self.pacing = Pacing::default();
        self.gpu.set_interpolation(on)
    }

    /// how the draws of the last frame matched the frame before's, for frame
    /// interpolation.
    pub fn interpolation_matches(&self) -> Option<zakuro_gpu::Matches> {
        self.pacing.matches
    }

    /// whether the screens show the pictures in between at this refresh.
    pub fn shows_between(&self) -> bool {
        self.pacing.between
    }

    /// the DSP finishing an audio frame, it plays its voices one frame on and
    /// interrupts the title, whose sound library does one update per interrupt.
    fn audio_frame(&mut self) {
        self.idle_interrupted(false);
        // the whole opening froze until this ran every 4.9 ms. goddamn audio timing
        if self.services.dsp.running {
            services::dsp::advance(self);
            services::dsp::signal_semaphore(self);
        }
        // the devices that report at a rate a title sets, which this is
        // fine enough for
        services::ir_rst::tick(self);
        services::ir::tick(self);
        services::cam::tick(self);
        services::uds::tick(self);
        services::dlp::tick(self);
    }

    /// the stereo samples the console played since the last call, at
    /// AUDIO_SAMPLE_RATE.
    pub fn take_audio(&mut self) -> Vec<[i16; 2]> {
        std::mem::take(&mut self.services.dsp.output)
    }

    /// what the software keyboard asks for, while a title waits on it.
    pub fn keyboard_request(&self) -> Option<&services::keyboard::Request> {
        services::keyboard::request(self)
    }

    /// closes the software keyboard with the text typed and the button
    /// pressed, zero being the leftmost and the last one confirming.
    pub fn answer_keyboard(&mut self, text: &str, button: usize) {
        services::keyboard::answer(self, text, button);
    }

    /// what the buttons, circle pad and touch screen are doing, which HID
    /// reports from the next frame on.
    pub fn set_input(&mut self, input: InputState) {
        self.services.hid.input = input;
    }

    // -- GPU plumbing -------------------------------------------------------

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

    /// makes guest memory hold what the host GPU drew over a range, before
    /// something other than a draw reads or writes it.
    pub fn sync_gpu(&mut self, addr: u32, len: u32) {
        let linear_base = self.kernel.linear_base;
        let mut guest = GuestMemory {
            writes: 0,
            linear_base,
            memory: &mut self.memory,
        };
        self.gpu.sync_memory(&mut guest, addr, len);
    }

    /// writes what a service hands the title, a file it read among them,
    /// over guest memory, after what the host GPU drew there. on the console
    /// the drawing was in memory long before, and the title can have put
    /// the buffer where one it was done drawing to was.
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

    /// the nine words gsp::ImportDisplayCaptureInfo returns, the addresses
    /// virtual, as the title gave them.
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

    /// a screen as RGBA at the resolution the host's GPU draws at, and its
    /// width and height. a picture the GPU did not draw, or the CPU changed
    /// since, is the console's own grown to it, so the size stays the same
    /// from frame to frame.
    pub fn read_screen_scaled(&mut self, screen: zakuro_common::Screen) -> Screen {
        self.read_scaled(screen, false)
    }

    /// the top screen's picture for the right eye, as read_screen_scaled
    /// gives the left one's, when the title draws in 3D.
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

    /// whether the title draws the top screen in 3D, a picture for each
    /// eye. its format asks for the screen's two buffers, the second for
    /// the right eye, which the console's GSP turns the 3D on for only
    /// with bit 5 set and bit 6 clear.
    pub fn draws_in_3d(&self) -> bool {
        let config = self.gpu.framebuffers[0];
        config.format & 0x60 == 0x20 && config.address_right() != 0
    }

    /// moves the 3D slider, which titles read every frame to decide whether
    /// and how deep to draw in 3D, 0 down and 1 all the way up.
    pub fn set_slider_3d(&mut self, value: f32) {
        self.config.slider_3d = value;
        memory::config::set_slider_3d(self.memory.phys.shared_page_mut(), value);
    }

    /// where a screen's picture is on the host's GPU, for a presenter sharing
    /// its device to draw straight from, none when the GPU does not show the
    /// screens itself or did not draw what the screen shows, read_screen_scaled
    /// has that. the presenter's work runs after the GPU draws the picture, so
    /// the newest goes up.
    pub fn gpu_screen(&mut self, screen: zakuro_common::Screen) -> Option<zakuro_gpu::GpuScreen> {
        self.gpu_picture(screen, false)
    }

    /// the top screen's picture for the right eye on the host's GPU, as
    /// gpu_screen gives the left one's, when the title draws in 3D.
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
        // frame interpolation's picture in between, where the screen has one
        if self.pacing.between {
            if let Some(between) = self.gpu.between_image(now) {
                return Some(between);
            }
        }
        self.gpu.screen_image(now)
    }

    /// a screen's picture as the GPU showing the screens has it, or the
    /// picture in between it and the one before, see pace, for saving it.
    /// it waits for the GPU.
    pub fn upright_screen(&mut self, screen: zakuro_common::Screen, between: bool) -> Option<Screen> {
        let now = self.screen_ref(screen, false)?;
        let (image, scale) = self.gpu.upright_picture(now, between)?;
        Some((image, screen.width() * scale, screen.height() * scale))
    }

    /// the picture the host's GPU drew scaled for a screen, when it is still
    /// what the screen shows.
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
        // what the screen had a presentation ago goes up now, always that
        // one, so each picture stays up as long as the title kept it
        let show = self.showing[index].replace(now).unwrap_or(now);
        let (image, scale) = self.gpu.scaled_picture(show).or_else(|| self.gpu.scaled_picture(now))?;
        Some((image, width * scale, height * scale))
    }

    /// the picture the host's GPU drew for a screen, when it is still what
    /// the screen shows, the top one's for the right eye when right.
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
        // plain buffers a row per screen column, as display transfers leave
        // them, rows can be longer than the screen
        let stride = if config.stride == 0 { height * bpp } else { config.stride };
        if self.lcd_force_black || address == 0 || !stride.is_multiple_of(bpp) || stride < height * bpp {
            return None;
        }
        // no sync first, that would wait for the GPU to finish the picture.
        // compared where it lies, a copy of both screens every frame costs
        // more than the comparing
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

    /// reads one screen into a straight RGBA8 buffer for presentation.
    pub fn read_screen(&mut self, screen: zakuro_common::Screen) -> Vec<u8> {
        self.read_screen_at(screen, false)
    }

    /// the same, the top screen's picture for the right eye when right.
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
                // column-major in memory, and the panel scans bottom to top.
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

    /// a one-line summary for the window title and the log.
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

/// adapter letting the GPU reach guest memory by physical address.
struct GuestMemory<'a> {
    memory: &'a mut Memory,
    linear_base: u32,
    /// the writes through it so far, see GpuMemory::writes.
    writes: u64,
}

impl GuestMemory<'_> {
    /// the physical address of an address translate made, in the linear
    /// heap or VRAM.
    fn physical(&self, addr: u32) -> Option<u32> {
        // the linear heap and VRAM are physical memory in order, so what
        // translate made of a physical address leads back to it. VRAM first,
        // from the old linear heap's base a New 3DS's heap size reaches it
        if (VRAM_VADDR..VRAM_VADDR + VRAM_SIZE).contains(&addr) {
            Some(VRAM_PADDR + (addr - VRAM_VADDR))
        } else if addr >= self.linear_base && addr - self.linear_base < FCRAM_SIZE_NEW3DS {
            Some(FCRAM_PADDR + (addr - self.linear_base))
        } else {
            None
        }
    }

    /// the bytes at addr in the host's memory, where they sit together.
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
        // not as the CPU reads, which could have the GPU write back from
        // inside itself
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
        // in one piece, as host hands it out
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

/// writes back what the GPU drew over a range, for a read of the CPU
/// waiting on it.
///
/// # Safety
///
/// gpu has to be the system's GPU, which nothing else holds while the CPU
/// runs.
unsafe fn sync_for_cpu(gpu: *mut (), linear_base: u32, memory: &mut Memory, addr: u32, len: u32) {
    log::debug!(
        target: "zakuro_core::memory",
        "the CPU reads 0x{addr:08X}..0x{:08X}, which the GPU drew, written back first",
        addr.wrapping_add(len)
    );
    let gpu = unsafe { &mut *(gpu as *mut Gpu) };
    gpu.sync_depth(&mut GuestMemory { memory, linear_base, writes: 0 }, addr, len);
}

/// writes back what the GPU drew over a range, for a write of the CPU
/// landing over it.
///
/// # Safety
///
/// as sync_for_cpu.
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
    /// what the interpreter ran lasts past the run.
    fn drop(&mut self) {
        if let Some(hints) = &mut self.hints {
            hints.save();
        }
    }
}

/// an RGBA picture of a given width grown by a whole factor, each pixel
/// repeated.
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

    /// a holder that took a mutex twice, through svcCreateMutex locked and
    /// svcWaitSynchronization1, and a waiter blocked on it, with the holder
    /// back on the core.
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

    /// the holder ended with the mutex all given back, and the waiter gets it
    /// once at the next schedule.
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

    /// one thread at the entry create_thread gives, its code a branch to
    /// itself, current and running.
    fn one_spinning_thread(system: &mut System) -> ThreadId {
        one_thread_running(system, &[0xEAFF_FFFE])
    }

    /// one thread running code at 0x00100000, current, with its stack and
    /// r0 in the same page, the stack at 0x00100F00 and r0 at 0x00100800.
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

    /// a loop of three svcs, the tick, a mutex's release and a sleep, the
    /// way titles poll a clock, at 0x00100000.
    const POLLING: [u32; 8] = [
        0xEF00_0028, // svc 0x28, the tick in r0 and r1
        0xE284_4001, // add r4, r4, #1
        0xE1A0_0005, // mov r0, r5
        0xEF00_0014, // svc 0x14, releasing the mutex in r5
        0xE1A0_0006, // mov r0, r6
        0xE3A0_1000, // mov r1, #0
        0xEF00_000A, // svc 0x0A, sleeping for r6 nanoseconds
        0xEAFF_FFF7, // b 0x00100000
    ];

    /// a loop that only reads the tick, at 0x00100040.
    const TICKING: [u32; 3] = [
        0xEF00_0028, // svc 0x28
        0xE284_4001, // add r4, r4, #1
        0xEAFF_FFFC, // b 0x00100040
    ];

    /// the blocks of POLLING and TICKING the way 3dsrecomp cuts code, each
    /// with its length, the svc it ends with and where it goes on.
    fn block(at: u32) -> Option<(i32, Option<u32>, u32)> {
        Some(match at {
            0x0010_0000 => (1, Some(0x28), 0x0010_0004),
            0x0010_0004 => (3, Some(0x14), 0x0010_0010),
            0x0010_0010 => (3, Some(0x0A), 0x0010_001C),
            0x0010_001C => (1, None, 0x0010_0000),
            0x0010_0040 => (1, Some(0x28), 0x0010_0044),
            0x0010_0044 => (2, None, 0x0010_0040),
            _ => return None,
        })
    }

    /// POLLING and TICKING as 3dsrecomp writes code, each block charged to
    /// the budget as it starts. their svcs run in place, or leave for the
    /// step, as in code made for version 4 of the interface.
    unsafe extern "C" fn polling<const IN_PLACE: bool>(ctx: *mut recomp_abi::Context) {
        unsafe {
            let mut at = (*ctx).r[15];
            loop {
                let Some((size, svc, next)) = block(at) else {
                    (*ctx).exit = recomp_abi::EXIT_UNWIND;
                    return;
                };
                if (*ctx).budget < size {
                    (*ctx).r[15] = at;
                    (*ctx).exit = recomp_abi::EXIT_BUDGET;
                    return;
                }
                (*ctx).budget -= size;
                match at {
                    0x0010_0004 => {
                        (*ctx).r[4] = (*ctx).r[4].wrapping_add(1);
                        (*ctx).r[0] = (*ctx).r[5];
                    }
                    0x0010_0010 => {
                        (*ctx).r[0] = (*ctx).r[6];
                        (*ctx).r[1] = 0;
                    }
                    0x0010_0044 => (*ctx).r[4] = (*ctx).r[4].wrapping_add(1),
                    _ => {}
                }
                if let Some(number) = svc {
                    (*ctx).r[15] = next;
                    if !IN_PLACE {
                        (*ctx).svc = number;
                        (*ctx).exit = recomp_abi::EXIT_SVC;
                        return;
                    }
                    ((*(*ctx).host).svc)(ctx, number);
                    if (*ctx).exit != 0 {
                        return;
                    }
                }
                at = next;
            }
        }
    }

    /// threads on POLLING or TICKING as recompiled code, sharing a mutex,
    /// each with where it starts, the nanoseconds it sleeps each time round
    /// and its priority.
    fn polling_system(in_place: bool, threads: &[(u32, u32, u32)]) -> System {
        use memory::{MemoryState, Permission};
        // the clock the host has would set the two apart
        let mut system = System::new(Config { clock: Some(0), ..Config::default() });
        let block = system.memory.phys.allocate(memory::MemoryRegion::Application, 0x1000).unwrap();
        system.memory.map(0x0010_0000, block.addr, 0x1000, Permission::RW | Permission::EXECUTE, MemoryState::Code);
        for (address, code) in [(0x0010_0000, &POLLING[..]), (0x0010_0040, &TICKING[..])] {
            for (index, word) in code.iter().enumerate() {
                system.memory.write32(address + index as u32 * 4, *word);
            }
        }
        system.cpu.regs[1] = 0;
        kernel::svc::dispatch(&mut system, 0x13);
        let mutex = system.cpu.regs[1];
        for (index, &(entry, sleep, priority)) in threads.iter().enumerate() {
            let stack = 0x0010_0F00 - index as u32 * 0x100;
            let id = system.kernel.create_thread(format!("thread {index}"), entry, stack, 0, priority, 0);
            system.map_tls_page(id);
            let regs = &mut system.kernel.thread_mut(id).context.regs;
            regs[5] = mutex;
            regs[6] = sleep;
        }
        system.kernel.schedule(&mut system.cpu, 0);
        let code: recomp_abi::Code = if in_place { polling::<true> } else { polling::<false> };
        let addresses = [0x0010_0000, 0x0010_0004, 0x0010_0010, 0x0010_001C, 0x0010_0040, 0x0010_0044];
        let entries: &'static [recomp_abi::Entry] = Box::leak(Box::from(addresses.map(|address| recomp_abi::Entry { address, code })));
        let abi = if in_place { recomp_abi::ABI } else { recomp_abi::OLDEST_ABI };
        // SAFETY: a table that lives on, of code working on a context as
        // 3dsrecomp's does
        let linked = unsafe { recompiled::Linked::new(0, abi, entries.as_ptr().cast(), entries.len() as u32, std::ptr::null(), 0) };
        system.recompiled = Some(recompiled::Library::linked(&linked).unwrap());
        system
    }

    /// code that runs its svcs in place leaves the machine just as code
    /// that leaves for the step at each does, through ticks, scheduling
    /// passes, yields, sleeps, preemption and frames, only stopping far
    /// less. the threads, their loops, sleeps and priorities vary, so that
    /// the svcs fall on every kind of stop. one function runs every label
    /// here, cutting the code the same way whether it is entered or goes on.
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
        for case in 0..40 {
            let threads: Vec<(u32, u32, u32)> = (0..1 + case % 3)
                .map(|_| {
                    let entry = pick(&[0x0010_0000, 0x0010_0000, 0x0010_0040]);
                    (entry, pick(&[0, 0, 900, 13_000, 50_000, 400_000]), pick(&[0x2F, 0x30, 0x30, 0x31]))
                })
                .collect();
            let mut systems = [polling_system(false, &threads), polling_system(true, &threads)];
            for system in &mut systems {
                for _ in 0..6 {
                    assert_eq!(system.run_frame(), FrameOutcome::Completed);
                }
            }
            let [steps, in_place] = &mut systems;
            assert_eq!(in_place.cpu.cycles, steps.cpu.cycles, "{threads:x?}");
            assert_eq!(in_place.history, steps.history, "{threads:x?}");
            assert!(in_place.save_state().unwrap() == steps.save_state().unwrap(), "{threads:x?} ends elsewhere");
            // the current thread's registers are the CPU's
            let rounds = steps.kernel.threads.iter().map(|thread| thread.context.regs[4]).chain([steps.cpu.regs[4]]);
            assert!(rounds.max().unwrap() > 100, "{threads:x?} went round");
            assert_eq!(steps.recompiled.as_ref().unwrap().in_place(), 0);
            ran_in_place += in_place.recompiled.as_ref().unwrap().in_place();
        }
        assert!(ran_in_place > 100_000, "{ran_in_place} svcs ran in place");
    }

    /// runs steps until the run gets to tick or steps runs out, how many it
    /// took.
    fn steps_to(system: &mut System, tick: u64, steps: usize) -> usize {
        (0..steps).take_while(|_| {
            system.step(None);
            system.cpu.cycles < tick
        }).count() + 1
    }

    /// a thread alone on the core spinning on nothing waits for the next
    /// event, the next audio frame here, instead of the interpreter running
    /// every turn of its loop until then.
    #[test]
    fn a_thread_spinning_alone_skips_to_the_next_event() {
        let mut system = System::new(Config::default());
        one_spinning_thread(&mut system);
        let audio = system.next_audio_frame;
        assert!(steps_to(&mut system, audio, 40_000) < 20_000, "skipped");
        assert_eq!(system.cpu.cycles, audio, "to the audio frame, not past it");
        assert!(system.idle.skipped > 0);
    }

    /// a loop that counts in memory changes something every turn, and runs
    /// every one of them.
    #[test]
    fn a_loop_that_counts_is_not_skipped() {
        let mut system = System::new(Config::default());
        // ldr r1, [r0]; add r1, r1, #1; str r1, [r0]; b back
        one_thread_running(&mut system, &[0xE590_1000, 0xE281_1001, 0xE580_1000, 0xEAFF_FFFB]);
        for _ in 0..40_000 {
            system.step(None);
        }
        assert_eq!(system.idle.skipped, 0);
        assert_eq!(system.memory.read32(0x0010_0800), 10_000, "every turn counted");
    }

    /// a loop that puts back what it wrote, the way taking and letting go of
    /// a lock does, and pushes and pops, changes nothing and is skipped.
    #[test]
    fn a_loop_that_puts_back_what_it_wrote_is_skipped() {
        let mut system = System::new(Config::default());
        // ldr r1, [r0]; add r1, r1, #1; str r1, [r0]; sub r1, r1, #1;
        // str r1, [r0]; push {r4, lr}; pop {r4, lr}; b back
        let code = [0xE590_1000, 0xE281_1001, 0xE580_1000, 0xE241_1001, 0xE580_1000, 0xE92D_4010, 0xE8BD_4010, 0xEAFF_FFF7];
        one_thread_running(&mut system, &code);
        system.memory.write32(0x0010_0800, 7);
        let audio = system.next_audio_frame;
        assert!(steps_to(&mut system, audio, 40_000) < 20_000, "skipped");
        assert_eq!(system.memory.read32(0x0010_0800), 7);
    }

    /// a skip stops at the wake of a thread sleeping, which gets the core
    /// then.
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

    /// waits on a handle for good, as svcWaitSynchronization1.
    fn wait_on(system: &mut System, handle: u32) {
        system.cpu.regs[0] = handle;
        system.cpu.regs[2] = u32::MAX;
        system.cpu.regs[3] = u32::MAX;
        kernel::svc::dispatch(system, 0x24);
    }

    /// steps until the thread runs again, the tick it does.
    fn runs_again(system: &mut System, id: ThreadId) -> u64 {
        for _ in 0..1000 {
            if system.kernel.current_thread == Some(id) && system.kernel.thread(id).status == ThreadStatus::Running {
                return system.cpu.cycles;
            }
            system.step(None);
        }
        panic!("the thread never ran again");
    }

    /// a wait the very next schedule ends, on a thread that keeps the core
    /// as nothing else can run, returns its result before the thread goes
    /// on, rather than the handle it was given, written over later.
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

    /// the top screen's second buffer is the right eye's only while its
    /// format asks for the 3D the way the console's GSP takes it, bit 5 set
    /// and bit 6 clear, and the title gave it one.
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
        // the slider games read moves with it
        system.set_slider_3d(0.5);
        let page = system.memory.phys.shared_page_mut();
        assert_eq!((f32::from_le_bytes(page[0x80..0x84].try_into().unwrap()), page[0x84]), (0.5, 0), "the LED not forced off");
    }

    /// a zero sleep with nobody else ready to run is a no-op: the calling
    /// thread keeps the core, and no scheduling pass is due. titles poll a
    /// worker in a tight svcSleepThread(0) loop, and paying a full pass per
    /// iteration there was most of what one such title spent.
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

    /// a zero sleep with another thread ready still steps aside, so a title
    /// polling a worker keeps handing it the core.
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

    /// a zero sleep while a thread is due to wake still has the pass then,
    /// so that thread wakes on time, not at the next preemption.
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

    /// a request to a service has the thread wait for the reply, and a
    /// thread below it runs meanwhile, as on the console. Star Fox 64 3D
    /// marks a save job running once the more urgent thread doing it is
    /// made, which has to be waiting on its first request by then.
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

    /// a timer fires at its tick, not at the next frame, and the thread on
    /// it wakes then. a pulse timer is gone once it woke a waiter, and comes
    /// again a period after it was due.
    #[test]
    fn a_timer_fires_at_its_tick() {
        let mut system = System::new(Config::default());
        let id = one_spinning_thread(&mut system);
        // a pulse timer, due in a millisecond and every millisecond after
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

    /// a pulse timer due in a millisecond and every millisecond after.
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

    /// a pulse timer releases every thread waiting on it when it fires, not
    /// only the first, and a fire nothing waits for is lost.
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

        // the next fire comes with nothing waiting, and is gone after
        system.cpu.cycles = 2 * period + 10;
        system.kernel.current_thread = Some(first);
        system.kernel.thread_mut(first).status = ThreadStatus::Running;
        wait_on(&mut system, timer);
        assert_eq!(system.kernel.thread(first).status, ThreadStatus::WaitSync, "the fire nothing waited for is lost");
    }

    /// cancelling a timer that came due before the scheduler looked keeps
    /// its fire, as the console's had already happened.
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

    /// a handle closed while a thread waits on its object neither wakes the
    /// thread as if it was signalled nor gives the object's place to the
    /// next one made, and the object goes once the wait is over.
    #[test]
    fn closing_what_a_thread_waits_on_keeps_it_waiting() {
        let mut system = System::new(Config::default());
        let id = one_spinning_thread(&mut system);
        system.cpu.regs[1] = 0;
        kernel::svc::dispatch(&mut system, 0x17);
        let event = system.cpu.regs[1];
        let object = system.kernel.resolve(event).unwrap();
        // for a millisecond
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

    /// a pulse event wakes what waits on it when signalled and is gone
    /// after, the next wait on it waits.
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

    /// an arbiter wait's timeout takes r5 as its high word, -1 waits for
    /// good.
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
            // wait if less than, with a timeout
            system.cpu.regs[2] = 3;
            system.cpu.regs[3] = 1;
            system.cpu.regs[4] = low;
            system.cpu.regs[5] = high;
            kernel::svc::dispatch(&mut system, 0x22);
            let wakeup = system.kernel.thread(id).wakeup_at;
            assert_eq!(wakeup.is_some(), wakes);
            if wakes {
                // five seconds
                let five = kernel::thread::nanos_to_ticks(5_000_000_000);
                assert!(wakeup.unwrap() >= five, "{wakeup:?}");
            }
        }
    }

    /// a thread whose timed arbiter wait ran out, and that then ended, left
    /// its entry on the arbiter. a signal for one waiter on the address
    /// passes over it to the thread really waiting there, and the dead one
    /// stays dead.
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
        // waits if less than 1, for a microsecond, and times out
        arbitrate(&mut system, ended, 3);
        system.kernel.current_thread = None;
        system.kernel.schedule(&mut system.cpu, 1_000_000);
        assert!(matches!(system.kernel.thread(ended).wait_result, Some(WaitResult::TimedOut)));
        // then ends
        system.kernel.current_thread = Some(ended);
        system.kernel.thread_mut(ended).status = ThreadStatus::Running;
        kernel::svc::dispatch(&mut system, 0x09);
        // another waits for good, and a third signals one waiter
        arbitrate(&mut system, parked, 1);
        assert_eq!(system.kernel.thread(parked).status, ThreadStatus::WaitArbiter);
        arbitrate(&mut system, signaller, 0);
        assert_eq!(system.kernel.thread(ended).status, ThreadStatus::Dead);
        assert_eq!(system.kernel.thread(parked).status, ThreadStatus::Ready);
    }
}
