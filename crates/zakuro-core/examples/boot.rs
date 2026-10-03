//! cargo run -p zakuro-core --example boot -- <rom> [frames]

use std::collections::BTreeSet;

use zakuro_core::services::hid::{InputState, PadState};
use zakuro_core::{loader, Config, FrameOutcome};

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: boot <rom> [frames]");
        std::process::exit(2);
    };
    let frames: u64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(60);

    let config = Config {
        language: std::env::var("ZAKURO_LANGUAGE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(zakuro_core::services::cfg::LANGUAGE_ENGLISH),
        // ZAKURO_RECOMPILED=path runs the code 3dsrecomp built for the title.
        recompiled: std::env::var("ZAKURO_RECOMPILED").ok().map(Into::into),
        // ZAKURO_RASTERIZER=hardware draws on the host GPU.
        hardware_renderer: std::env::var("ZAKURO_RASTERIZER").is_ok_and(|v| v == "hardware"),
        // ZAKURO_RESOLUTION=3 draws at three times the console's resolution.
        resolution: std::env::var("ZAKURO_RESOLUTION").ok().and_then(|v| v.parse().ok()).unwrap_or(1),
        // ZAKURO_CLOCK=milliseconds since 1900 starts the clock there, so that
        // runs repeat exactly.
        clock: std::env::var("ZAKURO_CLOCK").ok().and_then(|v| v.parse().ok()),
        // ZAKURO_PRESENT=direct draws on a device of the kind a Vulkan
        // presenter shares, which shows the screens straight from the GPU.
        device: std::env::var("ZAKURO_PRESENT").is_ok_and(|v| v == "direct").then(|| {
            let device = zakuro_gpu::SharedDevice::new().unwrap_or_else(|error| {
                eprintln!("no device to show the screens from: {error}");
                std::process::exit(1);
            });
            std::sync::Arc::new(device)
        }),
        ..Config::default()
    };
    // ZAKURO_REPLAY=file plays back what ZAKURO_RECORD wrote down in the
    // frontend, from the same clock
    let mut replay = std::env::var_os("ZAKURO_REPLAY").map(|file| {
        zakuro_core::replay::Replay::open(std::path::Path::new(&file)).unwrap_or_else(|error| {
            eprintln!("could not read {}: {error}", file.to_string_lossy());
            std::process::exit(1);
        })
    });
    let config = Config { clock: replay.as_ref().map(|replay| replay.clock()).or(config.clock), ..config };
    let mut system = match loader::load(&path, config) {
        Ok(system) => system,
        Err(e) => {
            eprintln!("failed to load {path}: {e}");
            std::process::exit(1);
        }
    };

    if std::env::var("ZAKURO_PROFILE").is_ok() {
        system.enable_profiler();
    }

    println!("--- booting for {frames} frames ---");
    let start = std::time::Instant::now();
    // how long each frame took, which shows the stutter an average hides
    let mut frame_times = Vec::with_capacity(frames as usize);
    let mut outcome = FrameOutcome::Completed;
    let mut executed = 0u64;

    // headless runs otherwise never press a button, so a title parked on an
    // intro or "press start" screen, which is most titles, most of the time,
    // would sit there for the entire run no matter how many frames are given.
    let mash_buttons = std::env::var("ZAKURO_MASH_BUTTONS").is_ok();

    // a scripted alternative for reaching a specific screen reproducibly,
    // ZAKURO_INPUT=300:A,420:DOWN+A holds each listed button from that
    // frame for a few frames, 500:@160x180 touches the bottom screen.
    let script = std::env::var("ZAKURO_INPUT")
        .map(|spec| parse_input_script(&spec))
        .unwrap_or_default();
    // frames to save the screens at, besides the last one,
    // ZAKURO_DUMP_AT=900,1200.
    let dump_at: BTreeSet<u64> = std::env::var("ZAKURO_DUMP_AT")
        .map(|spec| spec.split(',').filter_map(|f| f.trim().parse().ok()).collect())
        .unwrap_or_default();

    // verbose logs from one frame onwards only, so a trace of a late screen
    // is not buried under everything before it, ZAKURO_LOG_FROM=1000 with
    // RUST_LOG=info,zakuro_gpu=trace.
    let log_from: Option<u64> = std::env::var("ZAKURO_LOG_FROM").ok().and_then(|f| f.parse().ok());
    if log_from.is_some() {
        log::set_max_level(log::LevelFilter::Info);
    }

    // everything the console played, written to a WAV at the end,
    // ZAKURO_WAV=/tmp/zakuro.wav.
    let wav = std::env::var("ZAKURO_WAV").ok();
    let mut audio: Vec<[i16; 2]> = Vec::new();

    // reads both screens after every frame the way a window showing them
    // does, for timing a run like one, ZAKURO_PRESENT=1, or ZAKURO_PRESENT=
    // direct for one whose presenter shares the renderer's device.
    let present = std::env::var("ZAKURO_PRESENT").is_ok();

    // what gets typed when a title opens the software keyboard,
    // ZAKURO_KEYBOARD=Link.
    let typed = std::env::var("ZAKURO_KEYBOARD").unwrap_or_else(|_| "Zakuro".to_owned());

    let profile_from: Option<u64> = std::env::var("ZAKURO_PROFILE_FROM").ok().and_then(|f| f.parse().ok());
    for frame in 0..frames {
        if profile_from == Some(frame) {
            system.enable_profiler();
        }
        if log_from == Some(frame) {
            log::set_max_level(log::LevelFilter::Trace);
            println!("--- verbose logging from frame {frame} ---");
        }
        if let Some(replay) = &mut replay {
            system.set_input(replay.input(frame));
        } else if !script.is_empty() {
            let held = || script.iter().filter(|(start, ..)| (*start..*start + 6).contains(&frame));
            let buttons = held().fold(PadState::empty(), |held, (_, buttons, _)| held | *buttons);
            let touch = held().find_map(|(.., touch)| *touch);
            // the circle pad's directions push it all the way
            let axis = |plus: PadState, minus: PadState| buttons.contains(plus) as i8 as f32 - buttons.contains(minus) as i8 as f32;
            system.set_input(InputState {
                buttons,
                touch,
                circle_x: axis(PadState::CIRCLE_RIGHT, PadState::CIRCLE_LEFT),
                circle_y: axis(PadState::CIRCLE_UP, PadState::CIRCLE_DOWN),
            });
        } else if mash_buttons {
            let pressed = frame % 40 < 4;
            // some first-boot prompts (language/EULA screens) wait for a
            // touchscreen tap rather than a button, so tap the bottom screen's
            // center in the same window the buttons are held.
            let tap = (frame / 40) as u16;
            let x = 40 + (tap % 5) * 60;
            let y = 30 + ((tap / 5) % 6) * 35;
            // cycle through every button, one at a time, so a screen that
            // wants a specific one is not missed.
            const BUTTONS: [PadState; 8] = [
                PadState::A,
                PadState::B,
                PadState::START,
                PadState::SELECT,
                PadState::X,
                PadState::Y,
                PadState::UP,
                PadState::DOWN,
            ];
            let button = BUTTONS[(frame / 40) as usize % BUTTONS.len()];
            system.set_input(InputState {
                buttons: if pressed { button } else { PadState::empty() },
                touch: if pressed { Some((x, y)) } else { None },
                ..InputState::default()
            });
        }
        let frame_start = std::time::Instant::now();
        let faults = system.memory.fault_summary().len();
        outcome = system.run_frame();
        executed = frame + 1;
        // when a new unmapped page is touched, which frame it was
        if system.memory.fault_summary().len() != faults {
            println!("frame {executed}: touched an unmapped page");
        }
        if let Some(request) = system.keyboard_request() {
            let confirm = request.buttons.len() - 1;
            system.answer_keyboard(&typed, confirm);
        }
        if present {
            for (screen, _) in SCREENS {
                if system.gpu_screen(screen).is_none() {
                    std::hint::black_box(system.read_screen_scaled(screen));
                }
            }
        }
        frame_times.push(frame_start.elapsed());
        if wav.is_some() {
            audio.extend(system.take_audio());
        }
        if dump_at.contains(&executed) {
            for (screen, name) in SCREENS {
                save_screen(&mut system, screen, &temp(&format!("zakuro-{name}-{executed}.ppm")));
            }
            // ZAKURO_DUMP_MEM_AT=1 saves the ZAKURO_DUMP_MEM ranges with the
            // screens too, to watch memory change from frame to frame
            if std::env::var("ZAKURO_DUMP_MEM_AT").is_ok() {
                for (addr, len) in std::env::var("ZAKURO_DUMP_MEM").iter().flat_map(|spec| {
                    spec.split(';')
                        .filter_map(|range| {
                            let (addr, len) = range.split_once(',')?;
                            Some((u32::from_str_radix(addr.trim_start_matches("0x"), 16).ok()?, len.parse::<usize>().ok()?))
                        })
                        .collect::<Vec<_>>()
                }) {
                    let mut bytes = vec![0u8; len];
                    system.memory.read_bytes(addr, &mut bytes);
                    let _ = std::fs::write(temp(&format!("zakuro-dump-0x{addr:08X}-{executed}.bin")), &bytes);
                }
            }
        }
        if outcome != FrameOutcome::Completed {
            break;
        }
    }

    let elapsed = start.elapsed();
    if let Some(path) = &wav {
        match write_wav(path, &audio) {
            Ok(()) => println!("wrote {} seconds of sound to {path}", audio.len() / zakuro_core::AUDIO_SAMPLE_RATE as usize),
            Err(error) => eprintln!("could not write {path}, {error}"),
        }
    }
    println!("\n--- result ---");
    println!("outcome:       {outcome:?} after {executed} frames in {elapsed:.2?}");
    // ZAKURO_SLOW=25 lists the frames that took longer than 25 ms, to see
    // when a stutter happens
    if let Some(limit) = std::env::var("ZAKURO_SLOW").ok().and_then(|ms| ms.parse::<f64>().ok()) {
        let slow: Vec<String> = frame_times
            .iter()
            .enumerate()
            .filter(|(_, time)| time.as_secs_f64() * 1000.0 > limit)
            .map(|(frame, time)| format!("{}:{:.0}", frame + 1, time.as_secs_f64() * 1000.0))
            .collect();
        println!("slow frames:   {}", slow.join(" "));
    }
    if !frame_times.is_empty() {
        frame_times.sort_unstable();
        let at = |part: f64| frame_times[((frame_times.len() - 1) as f64 * part) as usize].as_secs_f64() * 1000.0;
        let slow = frame_times.iter().filter(|time| time.as_secs_f64() > 1.0 / 60.0).count();
        println!(
            "frame times:   median {:.1} ms, 95% {:.1} ms, 99% {:.1} ms, worst {:.1} ms, {slow} over a 60th of a second",
            at(0.5),
            at(0.95),
            at(0.99),
            at(1.0)
        );
    }
    println!("instructions:  {}", system.cpu.cycles);
    println!(
        "speed:         {:.2} MIPS",
        system.cpu.cycles as f64 / elapsed.as_secs_f64() / 1_000_000.0
    );
    println!("pc:            0x{:08X}", system.cpu.current_pc());
    println!("threads:       {}", system.kernel.threads.len());
    for thread in &system.kernel.threads {
        println!(
            "  {:<10} {:?} prio {} pc 0x{:08X}",
            thread.name, thread.status, thread.priority, thread.context.regs[15]
        );
        println!(
            "             r4 0x{:08X} r5 0x{:08X} r6 0x{:08X}",
            thread.context.regs[4], thread.context.regs[5], thread.context.regs[6]
        );
        if thread.status.is_blocked() {
            println!("             {}", system.kernel.describe_wait(thread.id));
            println!(
                "             lr 0x{:08X} sp 0x{:08X} r0 0x{:08X} r1 0x{:08X}",
                thread.context.regs[14],
                thread.context.regs[13],
                thread.context.regs[0],
                thread.context.regs[1]
            );
        }
    }

    if let Ok(spec) = std::env::var("ZAKURO_DUMP_MEM") {
        for range in spec.split(';') {
            let mut parts = range.split(',');
            let (Some(addr), Some(len)) = (parts.next(), parts.next()) else {
                continue;
            };
            let addr = u32::from_str_radix(addr.trim_start_matches("0x"), 16).unwrap();
            let len: usize = len.parse().unwrap();
            let mut bytes = vec![0u8; len];
            system.memory.read_bytes(addr, &mut bytes);
            let path = temp(&format!("zakuro-dump-0x{addr:08X}.bin"));
            std::fs::write(&path, &bytes).unwrap();
            println!("dumped 0x{addr:08X}..+0x{len:X} -> {path}");
        }
    }

    println!("handles:       {}", system.kernel.handles.len());
    if std::env::var("ZAKURO_HANDLES").is_ok() {
        let mut handles: Vec<_> = system.kernel.handles.iter().collect();
        handles.sort_by_key(|&(handle, _)| handle);
        for (handle, object) in handles {
            let kind = system.kernel.objects.get(object).map_or("missing", |o| o.type_name());
            println!("  0x{handle:08X} {kind} {}", system.kernel.handles.label(handle));
        }
    }
    println!(
        "services:      {}",
        system
            .services_seen
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
    );

    if !system.unimplemented_svcs.is_empty() {
        let list: Vec<String> = system
            .unimplemented_svcs
            .iter()
            .map(|n| format!("0x{n:02X}"))
            .collect();
        println!("missing svcs:  {}", list.join(" "));
    }

    if !system.services.unimplemented.is_empty() {
        println!("missing service commands:");
        let mut by_service: std::collections::BTreeMap<String, BTreeSet<String>> =
            Default::default();
        for ((service, command), count) in &system.services.unimplemented {
            by_service
                .entry(service.clone())
                .or_default()
                .insert(format!("0x{command:04X}x{count}"));
        }
        for (service, commands) in by_service {
            println!(
                "  {:<12} {}",
                service,
                commands.into_iter().collect::<Vec<_>>().join(" ")
            );
        }
    }

    let hot = system.hot_spots(std::env::var("ZAKURO_HOT").ok().and_then(|v| v.parse().ok()).unwrap_or(15));
    if !hot.is_empty() {
        println!("hot spots:");
        for (thread, pc, hits) in hot {
            println!("  {thread:<8} 0x{pc:08X}  {hits}");
        }
    }

    if !system.cro.modules.is_empty() {
        println!("modules:");
        for module in &system.cro.modules {
            println!(
                "  {:<20} at 0x{:08X}, {} exports",
                module.name,
                module.base,
                module.exports.len()
            );
        }
    }
    if !system.cro.unresolved.is_empty() {
        println!(
            "unresolved imports: {}",
            system.cro.unresolved.join(" ")
        );
    }

    // save what the screens hold, so there is something to look at as well as
    // read.
    for (screen, name) in SCREENS {
        let path = temp(&format!("zakuro-{name}.ppm"));
        let distinct = save_screen(&mut system, screen, &path);
        let sample: Vec<String> = distinct
            .iter()
            .take(4)
            .map(|c| format!("#{:02X}{:02X}{:02X}", c[0], c[1], c[2]))
            .collect();
        println!(
            "{name} screen:   {} distinct colors {} -> {path}",
            distinct.len(),
            sample.join(" ")
        );
    }

    for (index, name) in [(0, "top"), (1, "bottom")] {
        let config = system.gpu.framebuffers[index];
        println!(
            "{name} framebuffer: A 0x{:08X} B 0x{:08X} active {} (showing 0x{:08X}) \
             stride {} format 0x{:X}",
            config.address_a_left,
            config.address_b_left,
            config.active,
            config.address_left(),
            config.stride,
            config.format
        );
    }

    println!("gpu:           {}", system.status_line());
    if !system.fatal_errors.is_empty() {
        println!("fatal errors:  {}", system.fatal_errors.join("; "));
    }

    let faults = system.memory.fault_summary();
    if !faults.is_empty() {
        println!("unmapped pages touched: {}", faults.len());
        for (addr, count) in faults.iter().take(10) {
            println!("  0x{addr:08X} x{count}");
        }
    }

    if !system.debug_output.is_empty() {
        println!("--- guest debug output ---\n{}", system.debug_output);
    }
}

const SCREENS: [(zakuro_common::Screen, &str); 2] = [
    (zakuro_common::Screen::Top, "top"),
    (zakuro_common::Screen::Bottom, "bottom"),
];

/// writes a screen out as a PPM and returns the distinct colors on it.
fn save_screen(
    system: &mut zakuro_core::System,
    screen: zakuro_common::Screen,
    path: &str,
) -> std::collections::HashSet<[u8; 3]> {
    let (pixels, width, height) = system.read_screen_scaled(screen);
    let mut ppm = format!("P6\n{width} {height}\n255\n").into_bytes();
    for chunk in pixels.as_chunks::<4>().0 {
        ppm.extend_from_slice(&chunk[..3]);
    }
    let _ = std::fs::write(path, ppm);
    pixels
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| [c[0], c[1], c[2]])
        .collect()
}

/// a file in the system's temporary directory, /tmp on Linux.
fn temp(name: &str) -> String {
    std::env::temp_dir().join(name).display().to_string()
}

/// parses frame:BUTTON[+BUTTON...] entries separated by commas, where @XxY
/// touches the bottom screen at X, Y and CUP, CDOWN, CLEFT and CRIGHT push
/// the circle pad.
fn parse_input_script(spec: &str) -> Vec<(u64, PadState, Option<(u16, u16)>)> {
    spec.split(',')
        .filter_map(|entry| {
            let (frame, buttons) = entry.trim().split_once(':')?;
            let mut touch = None;
            let buttons = buttons.split('+').try_fold(PadState::empty(), |held, name| {
                if let Some((x, y)) = name.trim().strip_prefix('@').and_then(|at| at.split_once('x')) {
                    touch = Some((x.parse().ok()?, y.parse().ok()?));
                    return Some(held);
                }
                let button = match name.trim().to_ascii_uppercase().as_str() {
                    "A" => PadState::A,
                    "B" => PadState::B,
                    "X" => PadState::X,
                    "Y" => PadState::Y,
                    "L" => PadState::L,
                    "R" => PadState::R,
                    "START" => PadState::START,
                    "SELECT" => PadState::SELECT,
                    "UP" => PadState::UP,
                    "DOWN" => PadState::DOWN,
                    "LEFT" => PadState::LEFT,
                    "RIGHT" => PadState::RIGHT,
                    "CUP" => PadState::CIRCLE_UP,
                    "CDOWN" => PadState::CIRCLE_DOWN,
                    "CLEFT" => PadState::CIRCLE_LEFT,
                    "CRIGHT" => PadState::CIRCLE_RIGHT,
                    other => {
                        eprintln!("unknown button '{other}' in ZAKURO_INPUT");
                        return None;
                    }
                };
                Some(held | button)
            })?;
            Some((frame.trim().parse().ok()?, buttons, touch))
        })
        .collect()
}

/// a 16-bit stereo WAV at the DSP's rate.
fn write_wav(path: &str, samples: &[[i16; 2]]) -> std::io::Result<()> {
    let rate = zakuro_core::AUDIO_SAMPLE_RATE.round() as u32;
    let data = (samples.len() * 4) as u32;
    let mut bytes = Vec::with_capacity(44 + data as usize);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&rate.to_le_bytes());
    bytes.extend_from_slice(&(rate * 4).to_le_bytes());
    bytes.extend_from_slice(&4u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data.to_le_bytes());
    for [left, right] in samples {
        bytes.extend_from_slice(&left.to_le_bytes());
        bytes.extend_from_slice(&right.to_le_bytes());
    }
    std::fs::write(path, bytes)
}
