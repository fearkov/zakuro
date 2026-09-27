//! Zakuro's frontend, a window showing the library of games or a game with
//! a menu over it, and the loop that drives the emulated console one frame
//! at a time.

mod audio;
mod cli;
mod gui;
mod input;
mod library;
mod menus;
mod present;
mod recompile;
mod settings;

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Fullscreen, Window, WindowId};

use zakuro_common::Screen;
use zakuro_core::{loader, Config, FrameOutcome, System};
use zakuro_gpu::{layout, Overlay, PresentError, RendererKind, ScreenImage};

use gui::Gui;
use input::Keyboard;
use library::Library;
use menus::{Action, Menus};
use present::Backend;
use recompile::{Job, Stage};
use settings::{Renderer, Settings};

pub use zakuro_core::recompiled::Linked;

/// runs the emulator as the command line says, on recompiled code linked
/// into the program when there is some.
pub fn run(linked: Option<Linked>) {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let options = match cli::parse() {
        Ok(options) => options,
        Err(message) => {
            eprintln!("zakuro: {message}");
            std::process::exit(2);
        }
    };
    let settings = Settings::load();

    let data_dir = options.data.clone().map(PathBuf::from).or_else(default_data_dir);
    if let Some(dir) = &data_dir {
        bring_saves(dir);
    }

    let mut app = App {
        keyboard: Keyboard::new(settings.keys.clone()),
        scale: options.scale.unwrap_or(settings.scale).max(1),
        options,
        settings,
        linked,
        data_dir,
        game: None,
        library: Library::default(),
        menus: Menus::default(),
        jobs: Vec::new(),
        gui: None,
        backend: None,
        window: None,
        audio: None,
        mouse_down: false,
        cursor: (0.0, 0.0),
        last_title_update: Instant::now(),
        next_frame: Instant::now(),
        skipped: 0,
        paused: false,
        stop: false,
    };

    if app.options.test_pattern {
        // the test pattern has no CPU program to run, paused keeps just
        // presenting what is there
        let system = paint_test_pattern(app.config());
        app.game = Some(Running::new(system, PathBuf::new(), "test pattern".to_owned()));
        app.paused = true;
    } else if let Some(rom) = app.options.rom.clone() {
        if let Err(error) = app.play(Path::new(&rom)) {
            eprintln!("zakuro: {error}");
            std::process::exit(1);
        }
    }
    if app.options.profile {
        if let Some(game) = &mut app.game {
            game.system.enable_profiler();
        }
    }

    if let Some(frames) = app.options.headless {
        if let Some(game) = &mut app.game {
            run_headless(&mut game.system, frames);
        }
        return;
    }

    if let Some(folder) = app.settings.games.clone() {
        app.library.scan(&folder);
    }
    app.audio = if app.options.mute {
        None
    } else {
        audio::Audio::open(zakuro_core::AUDIO_SAMPLE_RATE).inspect_err(|error| log::warn!("no sound, {error}")).ok()
    };
    app.apply_volume();

    let event_loop = EventLoop::new().expect("create an event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    if let Err(error) = event_loop.run_app(&mut app) {
        eprintln!("zakuro: {error}");
    }
}

/// the system's place for a program's data.
fn default_data_dir() -> Option<PathBuf> {
    let var = |name: &str| std::env::var_os(name).filter(|value| !value.is_empty()).map(PathBuf::from);
    let dir = if cfg!(windows) {
        var("APPDATA")
    } else if cfg!(target_os = "macos") {
        var("HOME").map(|home| home.join("Library/Application Support"))
    } else {
        var("XDG_DATA_HOME").or_else(|| var("HOME").map(|home| home.join(".local/share")))
    };
    dir.map(|dir| dir.join("zakuro"))
}

/// saves Zakuro kept in the working directory before it had a place for
/// them, copied there the first time. the old ones stay as they were.
fn bring_saves(data: &Path) {
    let (old, new) = (Path::new("user"), data.join("user"));
    if new.exists() || !old.is_dir() {
        return;
    }
    // a copy cut short is left under another name, and tried again
    let partial = data.join("user.copying");
    let copied = std::fs::remove_dir_all(&partial)
        .or_else(|error| if error.kind() == std::io::ErrorKind::NotFound { Ok(()) } else { Err(error) })
        .and_then(|()| copy_dir(old, &partial))
        .and_then(|()| std::fs::rename(&partial, &new));
    match copied {
        Ok(()) => log::info!("copied the saves in ./user to {}, the old ones stay as they were", new.display()),
        Err(error) => log::warn!("could not copy the saves in ./user to {}, {error}", new.display()),
    }
}

fn copy_dir(from: &Path, to: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let target = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}

/// builds a System with no title loaded and both screens filled with a
/// solid color, to check the presentation path (framebuffer config -> guest
/// memory -> window) in isolation from any title's boot logic.
fn paint_test_pattern(config: Config) -> System {
    use zakuro_core::memory::{MemoryRegion, MemoryState, Permission};

    let mut system = System::new(config);

    // ABGR in memory for format 0 (Rgba8), red = 00 00 00 FF?
    let paint = |system: &mut System, screen: Screen, rgba: [u8; 4]| {
        let width = screen.width();
        let height = screen.height();
        let stride = height * 4;
        let size = stride * width;

        let block = system
            .memory
            .phys
            .allocate(MemoryRegion::Base, size)
            .expect("allocate test pattern framebuffer");

        let vaddr = zakuro_core::services::gsp::physical_to_virtual(system, block.addr);
        system.memory.map(
            vaddr,
            block.addr,
            size,
            Permission::READ | Permission::WRITE,
            MemoryState::Shared,
        );

        let mut pixel = [0u8; 4];
        pixel[3] = rgba[0]; // r
        pixel[2] = rgba[1]; // g
        pixel[1] = rgba[2]; // b
        pixel[0] = rgba[3]; // a
        let mut data = Vec::with_capacity(size as usize);
        for _ in 0..(size / 4) {
            data.extend_from_slice(&pixel);
        }
        system.memory.write_physical(block.addr, &data);

        let index = match screen {
            Screen::Top => 0,
            Screen::Bottom => 1,
        };
        system.set_framebuffer(index, 0, block.addr, block.addr, stride, 0);
    };

    paint(&mut system, Screen::Top, [220, 40, 40, 255]); // red
    paint(&mut system, Screen::Bottom, [40, 90, 220, 255]); // blue
    system
}

fn run_headless(system: &mut System, frames: u64) {
    let start = Instant::now();
    let mut outcome = FrameOutcome::Completed;
    let mut ran = 0;
    for _ in 0..frames {
        outcome = system.run_frame();
        ran += 1;
        if outcome != FrameOutcome::Completed {
            break;
        }
    }
    let elapsed = start.elapsed();
    println!("{outcome:?} after {ran} frames in {elapsed:.2?}");
    println!(
        "{:.1} MIPS, {}",
        system.cpu.cycles as f64 / elapsed.as_secs_f64() / 1e6,
        system.status_line()
    );
    if !system.fatal_errors.is_empty() {
        println!("fatal errors: {}", system.fatal_errors.join("; "));
    }
    for (thread, pc, hits) in system.hot_spots(10) {
        println!("  hot {thread:<8} 0x{pc:08X} {hits}");
    }
}

/// a game being played.
struct Running {
    system: System,
    path: PathBuf,
    name: String,
    /// frames run since counting_since, for the frame rate.
    frames: u32,
    counting_since: Instant,
    fps: f32,
}

impl Running {
    fn new(system: System, path: PathBuf, name: String) -> Running {
        Running { system, path, name, frames: 0, counting_since: Instant::now(), fps: 0.0 }
    }

    fn count_frame(&mut self) {
        self.frames += 1;
        let elapsed = self.counting_since.elapsed();
        if elapsed >= Duration::from_millis(500) {
            self.fps = self.frames as f32 / elapsed.as_secs_f32();
            self.frames = 0;
            self.counting_since = Instant::now();
        }
    }
}

struct App {
    options: cli::Options,
    settings: Settings,
    linked: Option<Linked>,
    data_dir: Option<PathBuf>,
    game: Option<Running>,
    library: Library,
    menus: Menus,
    jobs: Vec<Job>,
    gui: Option<Gui>,
    // backend must be declared (and therefore dropped) before window, Rust
    // drops struct fields in declaration order, and the GL surface's Drop
    // calls eglDestroySurface, which on Wayland does a protocol round-trip
    // against the window's wl_surface.
    backend: Option<Backend>,
    window: Option<Window>,
    keyboard: Keyboard,
    audio: Option<audio::Audio>,
    /// the window's size, times the console's.
    scale: u32,
    /// the left button is down, and where the pointer is, in window pixels.
    mouse_down: bool,
    cursor: (f32, f32),
    last_title_update: Instant,
    /// when the next frame is due. frames run to a schedule rather than one
    /// after another, so that a slow one is made up by those after it and
    /// the sound, made as the console runs, keeps up.
    next_frame: Instant,
    /// frames in a row not shown while catching up.
    skipped: u32,
    /// stopped with F1, without the menu.
    paused: bool,
    stop: bool,
}

/// one 3DS frame at 60 Hz.
const FRAME_TIME: Duration = Duration::from_nanos(16_666_667);
/// how far behind the schedule may fall before it starts over instead of
/// running fast to catch up, after a pause say.
const CATCH_UP_LIMIT: Duration = Duration::from_millis(200);
/// frames that may go unshown in a row while catching up.
const MAX_SKIPPED: u32 = 4;

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let size = winit::dpi::LogicalSize::new(400 * self.scale, 480 * self.scale);
        let attributes = Window::default_attributes().with_title("Zakuro").with_inner_size(size);
        let renderer = self.options.renderer.unwrap_or(match self.settings.renderer {
            Renderer::Vulkan => RendererKind::Vulkan,
            Renderer::OpenGl => RendererKind::OpenGl,
        });

        let backend = Backend::create(event_loop, attributes.clone(), renderer).or_else(|error| {
            if renderer != RendererKind::Vulkan {
                return Err(error);
            }
            // a machine without a working Vulkan driver still gets a window
            log::warn!("could not start the Vulkan backend, {error}, presenting with OpenGL instead");
            Backend::create(event_loop, attributes, RendererKind::OpenGl)
        });
        match backend {
            Ok((window, backend)) => {
                log::info!("presenting with the {} backend", backend.name());
                self.gui = Some(Gui::new(&window));
                self.window = Some(window);
                self.backend = Some(backend);
            }
            Err(error) => {
                eprintln!("zakuro: could not start the {renderer:?} backend: {error}");
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let consumed = match (&mut self.gui, &self.window) {
            (Some(gui), Some(window)) => gui.event(window, &event),
            _ => false,
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(backend) = &mut self.backend {
                    backend.resize(size.width, size.height);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => self.key(event, consumed),
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    self.mouse_down = state == ElementState::Pressed && !self.pointer_taken(consumed);
                    self.touch();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = (position.x as f32, position.y as f32);
                self.touch();
            }
            WindowEvent::RedrawRequested => self.step(event_loop),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

impl App {
    /// the console's settings for a game, from the command line and the
    /// settings.
    fn config(&self) -> Config {
        let interpret = self.options.interpreter || !self.settings.recompiled;
        Config {
            new3ds: self.options.new3ds,
            data_dir: self.data_dir.clone(),
            recompiled: self.options.recompiled.clone().filter(|_| !interpret).map(Into::into),
            linked: self.linked.filter(|_| !interpret),
            find_recompiled: !interpret,
            hardware_renderer: self.options.hardware_rasterizer.unwrap_or(self.settings.hardware_rasterizer),
            ..Config::default()
        }
    }

    /// starts the game at path, in place of any other.
    fn play(&mut self, path: &Path) -> Result<(), String> {
        // the old game and what it holds on the GPU go first
        self.game = None;
        let system = loader::load(path, self.config()).map_err(|error| format!("could not open {}, {error}", path.display()))?;
        let name = self
            .library
            .games
            .iter()
            .find(|game| game.path == path)
            .map(|game| game.name.clone())
            .or_else(|| path.file_stem().map(|stem| stem.to_string_lossy().into_owned()))
            .unwrap_or_default();
        self.game = Some(Running::new(system, path.to_owned(), name));
        self.menus.menu_open = false;
        self.keyboard.release();
        self.paused = false;
        self.next_frame = Instant::now();
        Ok(())
    }

    fn back_to_library(&mut self) {
        self.game = None;
        self.menus.menu_open = false;
        self.keyboard.release();
    }

    fn key(&mut self, event: KeyEvent, consumed: bool) {
        let pressed = event.state == ElementState::Pressed;
        let PhysicalKey::Code(code) = event.physical_key else { return };
        // a binding waiting for a key takes it, Escape leaves it as it was
        if let (true, Some(index)) = (pressed, self.menus.rebinding) {
            self.menus.rebinding = None;
            if code != KeyCode::Escape {
                if let Some((_, key)) = self.settings.keys.all_mut().into_iter().nth(index) {
                    *key = code;
                }
                self.apply_settings();
            }
            return;
        }
        if pressed && !event.repeat {
            match code {
                KeyCode::Escape if self.game.is_some() => {
                    self.menus.menu_open = !self.menus.menu_open;
                    self.keyboard.release();
                    return;
                }
                KeyCode::F1 => {
                    self.paused = !self.paused;
                    log::info!("{}", if self.paused { "paused" } else { "resumed" });
                }
                KeyCode::F11 => self.toggle_fullscreen(),
                _ => {}
            }
        }
        let to_game = self.game.is_some()
            && !self.menus.menu_open
            && !self.menus.settings_open
            && !consumed
            && !self.gui.as_ref().is_some_and(Gui::wants_keyboard);
        // letting go always gets through, so that no button stays held
        if to_game || !pressed {
            self.keyboard.key(event.physical_key, pressed);
        }
    }

    /// whether the pointer is busy with something other than the game.
    fn pointer_taken(&self, consumed: bool) -> bool {
        consumed || self.game.is_none() || self.menus.menu_open || self.gui.as_ref().is_some_and(Gui::wants_pointer)
    }

    /// touches the bottom screen while the button is down over it.
    fn touch(&mut self) {
        let Some(window) = &self.window else { return };
        let size = window.inner_size();
        let (_, bottom) = layout(size.width, size.height);
        let x = self.cursor.0 - bottom.x;
        let y = self.cursor.1 - bottom.y;
        let inside = x >= 0.0 && y >= 0.0 && x < bottom.width && y < bottom.height;
        self.keyboard.touch((self.mouse_down && inside).then(|| {
            ((x / bottom.width * 320.0) as u16, (y / bottom.height * 240.0) as u16)
        }));
    }

    fn toggle_fullscreen(&mut self) {
        if let Some(window) = &self.window {
            let full = window.fullscreen().is_some();
            window.set_fullscreen((!full).then_some(Fullscreen::Borderless(None)));
        }
    }

    fn apply_volume(&self) {
        if let Some(audio) = &self.audio {
            audio.set_volume(if self.settings.mute { 0.0 } else { self.settings.volume });
        }
    }

    /// saves the settings and puts to use what can change right away.
    fn apply_settings(&mut self) {
        self.settings.save();
        self.keyboard.set_keys(self.settings.keys.clone());
        self.apply_volume();
        if self.options.scale.is_none() && self.settings.scale.max(1) != self.scale {
            self.scale = self.settings.scale.max(1);
            if let Some(window) = &self.window {
                let _ = window.request_inner_size(winit::dpi::LogicalSize::new(400 * self.scale, 480 * self.scale));
            }
        }
    }

    fn recompile(&mut self, index: usize) {
        let Some(game) = self.library.games.get(index) else { return };
        self.jobs.retain(|job| job.program_id != game.program_id || !job.stage().finished());
        if self.jobs.iter().any(|job| job.program_id == game.program_id) {
            return;
        }
        self.jobs.push(Job::start(&game.path, game.program_id, &game.name));
    }

    /// tells about recompiles as they finish.
    fn poll_jobs(&mut self) {
        let mut finished = Vec::new();
        for job in &mut self.jobs {
            if !job.announced && job.stage().finished() {
                job.announced = true;
                finished.push((job.program_id, job.name.clone(), job.stage()));
            }
        }
        for (program_id, name, stage) in finished {
            match stage {
                Stage::Done => {
                    self.library.refresh_recompiled();
                    let playing = self.game.as_ref().is_some_and(|game| game.system.title.as_ref().is_some_and(|title| title.program_id() == program_id));
                    if playing {
                        self.menus.message = Some(format!("{name} is recompiled. Reset it from the menu, Esc, to run it on the new code."));
                    }
                }
                Stage::Failed(error) if error != "cancelled" => {
                    self.menus.message = Some(format!("Recompiling {name} failed, {error}"));
                }
                _ => {}
            }
        }
    }

    fn act(&mut self, action: Action, event_loop: &ActiveEventLoop) {
        match action {
            Action::Play(path) => {
                if let Err(error) = self.play(&path) {
                    self.menus.message = Some(error);
                }
            }
            Action::Recompile(index) => self.recompile(index),
            Action::CancelRecompile(program_id) => {
                for job in self.jobs.iter().filter(|job| job.program_id == program_id) {
                    job.cancel();
                }
            }
            Action::ChooseFolder => {
                if let Some(folder) = rfd::FileDialog::new().set_title("Where your games are").pick_folder() {
                    self.settings.games = Some(folder.clone());
                    self.settings.save();
                    self.library.scan(&folder);
                }
            }
            Action::ChooseBackground => {
                let picked = rfd::FileDialog::new()
                    .set_title("A picture for the library")
                    .add_filter("Pictures", &["png", "jpg", "jpeg", "webp", "bmp"])
                    .pick_file();
                if let Some(file) = picked {
                    self.settings.background = Some(file);
                    self.settings.save();
                }
            }
            Action::Rescan => {
                if let Some(folder) = self.settings.games.clone() {
                    self.library.scan(&folder);
                }
            }
            Action::Resume => {
                self.menus.menu_open = false;
                self.keyboard.release();
            }
            Action::Reset => {
                if let Some(path) = self.game.as_ref().map(|game| game.path.clone()) {
                    if let Err(error) = self.play(&path) {
                        self.menus.message = Some(error);
                    }
                }
            }
            Action::Library => self.back_to_library(),
            Action::Fullscreen => self.toggle_fullscreen(),
            Action::Quit => event_loop.exit(),
            Action::Settings => self.apply_settings(),
        }
    }

    fn step(&mut self, event_loop: &ActiveEventLoop) {
        if self.stop {
            event_loop.exit();
            return;
        }
        let now = Instant::now();
        if now > self.next_frame + CATCH_UP_LIMIT {
            self.next_frame = now;
        }
        self.library.poll();
        self.poll_jobs();

        let playing = self.game.is_some() && !self.paused && !self.menus.menu_open && !self.menus.settings_open;
        if playing {
            self.emulate();
        }

        self.next_frame += FRAME_TIME;
        // behind the schedule, showing the frame would wait on the display,
        // so it goes unshown, a few at most
        let behind = Instant::now() > self.next_frame;
        if playing && behind && self.skipped < MAX_SKIPPED {
            self.skipped += 1;
        } else {
            self.skipped = 0;
            self.present(event_loop);
        }

        if self.last_title_update.elapsed() >= Duration::from_millis(500) {
            self.last_title_update = Instant::now();
            if let Some(window) = &self.window {
                let title = match &self.game {
                    Some(game) => format!("Zakuro - {} - {}", game.name, game.system.status_line()),
                    None => "Zakuro".to_owned(),
                };
                window.set_title(&title);
            }
            let underruns = self.audio.as_ref().map_or(0, |audio| audio.take_underruns());
            if underruns > 0 && playing {
                log::warn!("the sound ran dry {underruns} times, the emulation is falling behind");
            }
        }

        let now = Instant::now();
        if now < self.next_frame {
            std::thread::sleep(self.next_frame - now);
        }
    }

    /// runs one frame of the game.
    fn emulate(&mut self) {
        let Some(game) = &mut self.game else { return };
        game.system.set_input(self.keyboard.state());
        let outcome = game.system.run_frame();
        game.count_frame();
        let sound = game.system.take_audio();
        if let Some(audio) = &self.audio {
            audio.push(&sound);
        }
        match outcome {
            FrameOutcome::Completed => {}
            FrameOutcome::Exited => {
                log::info!("the title exited");
                self.back_to_library();
            }
            FrameOutcome::Faulted => {
                log::error!("the title stopped on a fault");
                let errors = game.system.fatal_errors.join("; ");
                if !errors.is_empty() {
                    log::error!("{errors}");
                }
                self.menus.message = Some(format!("The game stopped on a fault. {errors}"));
                self.back_to_library();
            }
        }
    }

    /// the menus, and the actions they asked for, done.
    fn interface(&mut self, event_loop: &ActiveEventLoop) -> Overlay {
        let (Some(gui), Some(window)) = (&mut self.gui, &self.window) else { return Overlay::default() };
        let show_fps = self.settings.show_fps;
        let game = self.game.as_ref().map(|game| (game.name.clone(), game.fps, game.system.recompiled.is_some()));
        let (menus, library, settings, jobs) = (&mut self.menus, &self.library, &mut self.settings, &self.jobs);
        let mut actions = Vec::new();
        let overlay = gui.frame(window, |ui| {
            match &game {
                Some((name, fps, recompiled)) => {
                    actions.extend(menus.game(ui, name, show_fps.then_some(*fps), *recompiled, jobs))
                }
                None => actions.extend(menus.library(ui, library, settings, jobs)),
            }
            actions.extend(menus.settings(ui.ctx(), settings));
            menus.message(ui.ctx());
        });
        self.library.changed = false;
        for action in actions {
            self.act(action, event_loop);
        }
        overlay
    }

    fn present(&mut self, event_loop: &ActiveEventLoop) {
        let overlay = self.interface(event_loop);
        let (top, bottom) = match &mut self.game {
            Some(game) => (game.system.read_screen(Screen::Top), game.system.read_screen(Screen::Bottom)),
            None => (Vec::new(), Vec::new()),
        };
        let Some(backend) = &mut self.backend else { return };
        let result = backend.present(
            ScreenImage { width: Screen::Top.width(), height: Screen::Top.height(), pixels: &top },
            ScreenImage { width: Screen::Bottom.width(), height: Screen::Bottom.height(), pixels: &bottom },
            &overlay,
        );
        match result {
            Ok(()) | Err(PresentError::OutOfDate) => {}
            Err(error) => {
                log::error!("presentation failed: {error}");
                event_loop.exit();
            }
        }
    }
}

/// keeps the renderer kind referenced even when a backend feature is off.
const _: RendererKind = RendererKind::Software;
