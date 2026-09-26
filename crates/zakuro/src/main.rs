//! Zakuro's frontend, a window, a presentation backend, and the loop that
//! drives the emulated console one frame at a time.

mod cli;
mod input;
mod present;

use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use zakuro_common::Screen;
use zakuro_core::{loader, Config, FrameOutcome, System};
use zakuro_gpu::{layout, PresentError, RendererKind, ScreenImage};

use input::Keyboard;
use present::Backend;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let options = match cli::parse() {
        Ok(options) => options,
        Err(message) => {
            eprintln!("zakuro: {message}");
            std::process::exit(2);
        }
    };

    let config = Config {
        new3ds: options.new3ds,
        recompiled: options.recompiled.clone().map(Into::into),
        ..Config::default()
    };

    let mut system = if options.test_pattern {
        paint_test_pattern(config)
    } else {
        match loader::load(&options.rom, config) {
            Ok(system) => system,
            Err(error) => {
                eprintln!("zakuro: could not load {}: {error}", options.rom);
                std::process::exit(1);
            }
        }
    };
    if options.profile {
        system.enable_profiler();
    }

    if let Some(frames) = options.headless {
        run_headless(&mut system, frames);
        return;
    }

    let event_loop = EventLoop::new().expect("create an event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    // the test pattern has no CPU program to run, paused already skips
    // run_frame every tick, which is exactly "just keep presenting what's
    // there".
    let paused = options.test_pattern;
    let mut app = App {
        system,
        options,
        window: None,
        backend: None,
        keyboard: Keyboard::default(),
        last_title_update: Instant::now(),
        frame_start: Instant::now(),
        paused,
        stop: false,
    };
    if let Err(error) = event_loop.run_app(&mut app) {
        eprintln!("zakuro: {error}");
    }
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

struct App {
    system: System,
    options: cli::Options,
    // backend must be declared (and therefore dropped) before window, Rust
    // drops struct fields in declaration order, and the GL surface's Drop
    // calls eglDestroySurface, which on Wayland does a protocol round-trip
    // against the window's wl_surface.
    backend: Option<Backend>,
    window: Option<Window>,
    keyboard: Keyboard,
    last_title_update: Instant,
    frame_start: Instant,
    paused: bool,
    stop: bool,
}

/// one 3DS frame at 60 Hz.
const FRAME_TIME: Duration = Duration::from_nanos(16_666_667);

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let scale = self.options.scale.max(1);
        let size = winit::dpi::LogicalSize::new(400 * scale, 480 * scale);
        let attributes = Window::default_attributes()
            .with_title("Zakuro")
            .with_inner_size(size);

        let backend = Backend::create(event_loop, attributes.clone(), self.options.renderer).or_else(|error| {
            if self.options.renderer != RendererKind::Vulkan {
                return Err(error);
            }
            // a machine without a working Vulkan driver still gets a window
            log::warn!("could not start the Vulkan backend, {error}, presenting with OpenGL instead");
            Backend::create(event_loop, attributes, RendererKind::OpenGl)
        });
        match backend {
            Ok((window, backend)) => {
                log::info!("presenting with the {} backend", backend.name());
                self.window = Some(window);
                self.backend = Some(backend);
            }
            Err(error) => {
                eprintln!("zakuro: could not start the {:?} backend: {error}", self.options.renderer);
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(backend) = &mut self.backend {
                    backend.resize(size.width, size.height);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                use winit::keyboard::{KeyCode, PhysicalKey};
                let pressed = event.state == ElementState::Pressed;
                if pressed && event.physical_key == PhysicalKey::Code(KeyCode::Escape) {
                    event_loop.exit();
                    return;
                }
                if pressed && event.physical_key == PhysicalKey::Code(KeyCode::F1) {
                    self.paused = !self.paused;
                    log::info!("{}", if self.paused { "paused" } else { "resumed" });
                }
                self.keyboard.key(event.physical_key, pressed);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left && state == ElementState::Released {
                    self.keyboard.touch(None);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                // only clicks inside the bottom screen count as touches.
                if let Some(window) = &self.window {
                    let size = window.inner_size();
                    let (_, bottom) = layout(size.width, size.height);
                    let x = position.x as f32 - bottom.x;
                    let y = position.y as f32 - bottom.y;
                    if x >= 0.0 && y >= 0.0 && x < bottom.width && y < bottom.height {
                        let sx = (x / bottom.width * 320.0) as u16;
                        let sy = (y / bottom.height * 240.0) as u16;
                        self.keyboard.touch(Some((sx, sy)));
                    }
                }
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
    fn step(&mut self, event_loop: &ActiveEventLoop) {
        if self.stop {
            return;
        }
        self.frame_start = Instant::now();

        if !self.paused {
            self.system.set_input(self.keyboard.state());
            match self.system.run_frame() {
                FrameOutcome::Completed => {}
                FrameOutcome::Exited => {
                    log::info!("the title exited");
                    self.stop = true;
                }
                FrameOutcome::Faulted => {
                    log::error!("the title stopped on a fault");
                    if !self.system.fatal_errors.is_empty() {
                        log::error!("{}", self.system.fatal_errors.join("; "));
                    }
                    self.stop = true;
                }
            }
        }

        let top = self.system.read_screen(Screen::Top);
        let bottom = self.system.read_screen(Screen::Bottom);

        if let Some(backend) = &mut self.backend {
            let result = backend.present(
                ScreenImage {
                    width: Screen::Top.width(),
                    height: Screen::Top.height(),
                    pixels: &top,
                },
                ScreenImage {
                    width: Screen::Bottom.width(),
                    height: Screen::Bottom.height(),
                    pixels: &bottom,
                },
            );
            match result {
                Ok(()) | Err(PresentError::OutOfDate) => {}
                Err(error) => {
                    log::error!("presentation failed: {error}");
                    event_loop.exit();
                }
            }
        }

        if self.last_title_update.elapsed() >= Duration::from_millis(500) {
            self.last_title_update = Instant::now();
            if let Some(window) = &self.window {
                window.set_title(&format!("Zakuro - {}", self.system.status_line()));
            }
        }

        // pace to 60 Hz.
        let elapsed = self.frame_start.elapsed();
        if elapsed < FRAME_TIME {
            std::thread::sleep(FRAME_TIME - elapsed);
        }
    }
}

/// keeps the renderer kind referenced even when a backend feature is off.
const _: RendererKind = RendererKind::Software;
