//! the emulated console, CPU, memory, HLE kernel, HLE services and GPU, and
//! the loop that drives them.

pub mod cro;
pub mod hints;
pub mod kernel;
pub mod loader;
pub mod memory;
pub mod replay;
pub mod recompiled;
pub mod services;

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
    /// how many times the console's resolution the host's GPU draws at.
    pub resolution: u32,
    /// the Vulkan presenter's device, to draw on and show the screens
    /// straight from, when it can.
    pub device: Option<std::sync::Arc<zakuro_gpu::SharedDevice>>,
    /// the time the console's clock starts at, in milliseconds since 1900,
    /// the host's when none, a fixed one makes runs repeat exactly.
    pub clock: Option<u64>,
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
            resolution: 1,
            device: None,
            clock: None,
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

pub struct System {
    pub cpu: Cpu,
    pub memory: Memory,
    pub kernel: Kernel,
    pub services: ServiceState,
    pub gpu: Gpu,
    pub renderer: Box<dyn Renderer>,
    pub title: Option<Title>,
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
    showing: [Option<zakuro_gpu::ScreenRef>; 2],
    /// the dynamic module loader's state.
    pub cro: cro::CroManager,
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
    pub profile: Option<BTreeMap<(String, u32), u64>>,

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
        let app_bytes = 64 * 1024 * 1024;
        let mut services = ServiceState::default();
        if let Some(dir) = &config.data_dir {
            services.fs.user_dir = dir.join("user");
        }
        System {
            cpu: Cpu::new(),
            memory: Memory::new(config.new3ds, app_bytes),
            kernel: Kernel::new(0, memory::MemoryRegion::Application, linear_heap_base(config.new3ds)),
            services,
            gpu: Gpu::new(),
            renderer: Box::new(SoftwareRenderer::default()),
            title: None,
            config,
            exited: false,
            broke: false,
            debug_output: String::new(),
            unimplemented_svcs: BTreeSet::new(),
            services_seen: BTreeSet::new(),
            lcd_force_black: false,
            showing: [None, None],
            cro: cro::CroManager::default(),
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
            recompiled: None,
            hints: None,
            interpreted_instructions: 0,
            recompiled_instructions: 0,
        }
    }

    // -- process setup ------------------------------------------------------

    /// maps the page holding a thread's TLS block, if it is not mapped yet.
    pub fn map_tls_page(&mut self, thread: ThreadId) {
        let tls = self.kernel.thread(thread).tls;
        let page = tls & !PAGE_MASK;
        if self.memory.mapping_at(page).is_some() {
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
        services::hid::update(self, input);
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
            self.kernel.reschedule_pending = false;
            let tick = self.cpu.cycles;
            if self.kernel.schedule(&mut self.cpu, tick) {
                self.apply_wait_result();
            }
        }

        if self.kernel.current_thread.is_none() {
            // everything is blocked.
            let mut limit = deadline.unwrap_or(self.cpu.cycles + CYCLES_PER_FRAME);
            // an audio thread waits on the DSP's interrupt, which only fires
            // at an audio frame boundary, skipping past one would lose it.
            if self.services.dsp.running {
                limit = limit.min(self.next_audio_frame);
            }
            match self.next_wakeup() {
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

        self.history[self.history_index] = pc | self.cpu.cpsr.thumb as u32;
        self.history_index = (self.history_index + 1) % HISTORY_LENGTH;

        if self.profile.is_some() {
            self.sample();
        }

        let exit = match self.run_recompiled(deadline) {
            Some(exit) => {
                if let Some(hints) = &mut self.hints {
                    hints.library_ran();
                }
                exit
            }
            None => {
                self.interpreted_instructions += 1;
                let thumb = self.cpu.cpsr.thumb;
                if let (Some(hints), Some(library)) = (&mut self.hints, &self.recompiled) {
                    if !library.has_code(pc | thumb as u32) {
                        hints.interpreted(pc, thumb);
                    }
                }
                let gpu = &mut self.gpu as *mut Gpu as *mut ();
                let linear_base = self.kernel.linear_base;
                // SAFETY: the GPU stays where it is, and nothing but the CPU's
                // reads follow the pointer, until the step is over
                unsafe { self.memory.set_gpu_sync(memory::GpuSync { gpu, linear_base, sync: sync_for_cpu }) };
                let exit = self.cpu.step(&mut self.memory);
                self.memory.clear_gpu_sync();
                exit
            }
        };
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

        // give the scheduler a chance on a regular cadence so that a thread
        // which never makes a syscall cannot monopolise the core.
        if self.cpu.cycles >= self.next_preempt {
            self.next_preempt = self.cpu.cycles + PREEMPT_INTERVAL;
            self.kernel.reschedule_pending = true;
        }

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
        let mut limit = self.next_frame_boundary.min(self.next_audio_frame);
        // a stop for the scheduler unwinds every guest call the code is in,
        // which the host then enters again one at a time. it only changes
        // anything when another thread can run, otherwise the run goes on to
        // the next tick a thread or a timer wakes at
        if self.kernel.others_runnable() {
            limit = limit.min(self.next_preempt);
        } else if let Some(tick) = self.next_wakeup().filter(|tick| *tick > self.cpu.cycles) {
            limit = limit.min(tick);
        }
        if let Some(deadline) = deadline {
            limit = limit.min(deadline);
        }
        let budget = limit.saturating_sub(self.cpu.cycles).max(1);
        // a read of the CPU from where the GPU drew has that come down
        // first, with nothing else holding the GPU until the run is over
        let gpu = &mut self.gpu as *mut Gpu as *mut ();
        let linear_base = self.kernel.linear_base;
        // SAFETY: as in step, until the run is over
        unsafe { self.memory.set_gpu_sync(memory::GpuSync { gpu, linear_base, sync: sync_for_cpu }) };
        let (ran, stop) = library.run(&mut self.cpu, &mut self.memory, budget);
        self.memory.clear_gpu_sync();
        self.recompiled_instructions += ran;
        if ran == 0 && matches!(stop, recompiled::Stop::Left) {
            // not enough budget left for a whole block
            if let Some(hints) = &mut self.hints {
                hints.library_declined();
            }
            return None;
        }
        Some(match stop {
            recompiled::Stop::Svc(number) => Some(Exit::Supervisor(number)),
            recompiled::Stop::Exit(exit) => Some(exit),
            recompiled::Stop::Left => None,
        })
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
        let mut entries: Vec<(String, u32, u64)> = profile
            .iter()
            .map(|((thread, pc), hits)| (thread.clone(), *pc, *hits))
            .collect();
        entries.sort_by_key(|&(_, _, hits)| std::cmp::Reverse(hits));
        entries.truncate(count);
        entries
    }

    fn sample(&mut self) {
        let Some(id) = self.kernel.current_thread else {
            return;
        };
        let name = self.kernel.thread(id).name.clone();
        let pc = self.cpu.regs[15];
        if let Some(profile) = &mut self.profile {
            *profile.entry((name, pc)).or_insert(0) += 1;
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

    /// the earliest tick any blocked thread or armed timer wants to wake at.
    fn next_wakeup(&self) -> Option<u64> {
        let threads = self
            .kernel
            .threads
            .iter()
            .filter(|t| t.status.is_blocked())
            .filter_map(|t| t.wakeup_at);
        let timers = self.kernel.objects.iter().filter_map(|(_, object)| {
            match object {
                kernel::object::KObject::Timer(timer) => timer.fire_at,
                _ => None,
            }
        });
        threads.chain(timers).min()
    }

    /// everything that happens between frames, vertical blank, input, clock.
    fn end_frame(&mut self) {
        self.frames += 1;
        if self.frames.is_multiple_of(hints::SAVE_EVERY) {
            if let Some(hints) = &mut self.hints {
                hints.save();
            }
        }

        // refresh the kernel's shared page so the guest's clock advances.
        let tick = self.cpu.cycles;
        memory::config::update_datetime(self.memory.phys.shared_page_mut(), self.boot_clock, tick);

        // fire the expired timers.
        let mut signalled = Vec::new();
        for (id, object) in self.kernel.objects.iter() {
            if let kernel::object::KObject::Timer(timer) = object {
                if timer.fire_at.is_some_and(|at| tick >= at) {
                    signalled.push(id);
                }
            }
        }
        for id in signalled {
            if let Some(kernel::object::KObject::Timer(timer)) = self.kernel.objects.get_mut(id) {
                timer.signaled = true;
                timer.fire_at = if timer.interval > 0 {
                    Some(tick + timer.interval)
                } else {
                    None
                };
            }
            self.kernel.reschedule_pending = true;
        }

        self.renderer.end_frame();

        // pick up any buffer swap the game queued directly in GSP shared
        // memory before the LCDs latch whatever is currently configured.
        services::gsp::apply_framebuffer_updates(self);

        // both LCDs finish scanning out, in that order.
        services::gsp::signal_interrupt(self, services::gsp::InterruptId::Pdc0);
        services::gsp::signal_interrupt(self, services::gsp::InterruptId::Pdc1);

    }

    /// the DSP finishing an audio frame, it plays its voices one frame on and
    /// interrupts the title, whose sound library does one update per interrupt.
    fn audio_frame(&mut self) {
        // the whole opening froze until this ran every 4.9 ms. goddamn audio timing
        if self.services.dsp.running {
            services::dsp::advance(self);
            services::dsp::signal_semaphore(self);
        }
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
            linear_base,
            memory: &mut self.memory,
        };
        self.gpu.sync_memory(&mut guest, addr, len);
    }

    pub fn memory_fill(&mut self, start: u32, end: u32, value: u32, width: u32) {
        let linear_base = self.kernel.linear_base;
        let mut guest = GuestMemory {
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
            linear_base,
            memory: &mut self.memory,
        };
        self.gpu
            .texture_copy(&mut guest, input, output, size, input_gap, output_gap);
    }

    /// the nine words gsp::ImportDisplayCaptureInfo returns.
    pub fn display_capture_info(&mut self) -> [u32; 9] {
        let top = self.gpu.framebuffers[0];
        let bottom = self.gpu.framebuffers[1];
        [
            top.address_left(),
            top.address_right(),
            top.format,
            top.stride,
            bottom.address_left(),
            bottom.address_right(),
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
        let (width, height) = (screen.width(), screen.height());
        let scale = self.gpu.scale();
        if let Some(scaled) = self.scaled_screen(screen) {
            return scaled;
        }
        let native = self.read_screen(screen);
        if scale == 1 {
            return (std::sync::Arc::new(native), width, height);
        }
        (std::sync::Arc::new(grow(&native, width, scale)), width * scale, height * scale)
    }

    /// where a screen's picture is on the host's GPU, for a presenter sharing
    /// its device to draw straight from, none when the GPU does not show the
    /// screens itself or did not draw what the screen shows, read_screen_scaled
    /// has that. the presenter's work runs after the GPU draws the picture, so
    /// the newest goes up.
    pub fn gpu_screen(&mut self, screen: zakuro_common::Screen) -> Option<zakuro_gpu::GpuScreen> {
        if !self.gpu.shows_directly() {
            return None;
        }
        let now = self.screen_ref(screen)?;
        self.gpu.screen_image(now)
    }

    /// the picture the host's GPU drew scaled for a screen, when it is still
    /// what the screen shows.
    fn scaled_screen(&mut self, screen: zakuro_common::Screen) -> Option<Screen> {
        let (width, height) = (screen.width(), screen.height());
        let index = match screen {
            zakuro_common::Screen::Top => 0,
            zakuro_common::Screen::Bottom => 1,
        };
        let Some(now) = self.screen_ref(screen) else {
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
    /// the screen shows.
    fn screen_ref(&mut self, screen: zakuro_common::Screen) -> Option<zakuro_gpu::ScreenRef> {
        let (width, height) = (screen.width(), screen.height());
        let index = match screen {
            zakuro_common::Screen::Top => 0,
            zakuro_common::Screen::Bottom => 1,
        };
        let config = self.gpu.framebuffers[index];
        let address = config.address_left();
        let format = config.color_format();
        let bpp = format.bytes_per_pixel() as u32;
        // plain buffers a row per screen column, as display transfers leave
        // them, rows can be longer than the screen
        let stride = if config.stride == 0 { height * bpp } else { config.stride };
        if self.lcd_force_black || address == 0 || !stride.is_multiple_of(bpp) || stride < height * bpp {
            return None;
        }
        // no sync first, that would wait for the GPU to finish the picture
        let base = services::gsp::physical_to_virtual(self, address);
        let len = width * stride;
        let mut guest = vec![0u8; len as usize];
        self.memory.read_bytes(base, &mut guest);
        self.gpu.scaled_screen(base, (height, width), stride / bpp, format, &guest)
    }

    /// reads one screen into a straight RGBA8 buffer for presentation.
    pub fn read_screen(&mut self, screen: zakuro_common::Screen) -> Vec<u8> {
        let index = match screen {
            zakuro_common::Screen::Top => 0,
            zakuro_common::Screen::Bottom => 1,
        };
        let config = self.gpu.framebuffers[index];
        let width = screen.width();
        let height = screen.height();
        let mut out = vec![0u8; (width * height * 4) as usize];

        let address = config.address_left();
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
}

impl GpuMemory for GuestMemory<'_> {
    fn read(&mut self, addr: u32, out: &mut [u8]) {
        self.memory.read_bytes(addr, out);
    }

    fn write(&mut self, addr: u32, data: &[u8]) {
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

    fn slice(&mut self, addr: u32, len: usize) -> Option<&[u8]> {
        // the linear heap and VRAM are physical memory in order, so what
        // translate made of a physical address leads back to it
        let physical = if addr >= self.linear_base && addr - self.linear_base < FCRAM_SIZE_NEW3DS {
            FCRAM_PADDR + (addr - self.linear_base)
        } else if (VRAM_VADDR..VRAM_VADDR + VRAM_SIZE).contains(&addr) {
            VRAM_PADDR + (addr - VRAM_VADDR)
        } else {
            return None;
        };
        self.memory.phys.host_slice_mut(physical, u32::try_from(len).ok()?).map(|slice| &*slice)
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
    gpu.sync_depth(&mut GuestMemory { memory, linear_base }, addr, len);
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
