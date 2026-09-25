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
        ..Config::default()
    };
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
    let mut outcome = FrameOutcome::Completed;
    let mut executed = 0u64;

    // headless runs otherwise never press a button, so a title parked on an
    // intro or "press start" screen, which is most titles, most of the time,
    // would sit there for the entire run no matter how many frames are given.
    let mash_buttons = std::env::var("ZAKURO_MASH_BUTTONS").is_ok();

    // a scripted alternative for reaching a specific screen reproducibly,
    // ZAKURO_INPUT=300:A,420:DOWN+A holds each listed button from that
    // frame for a few frames.
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

    for frame in 0..frames {
        if log_from == Some(frame) {
            log::set_max_level(log::LevelFilter::Trace);
            println!("--- verbose logging from frame {frame} ---");
        }
        if !script.is_empty() {
            let buttons = script
                .iter()
                .filter(|(start, _)| (*start..*start + 6).contains(&frame))
                .fold(PadState::empty(), |held, (_, buttons)| held | *buttons);
            system.set_input(InputState {
                buttons,
                ..InputState::default()
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
        outcome = system.run_frame();
        executed = frame + 1;
        if dump_at.contains(&executed) {
            for (screen, name) in SCREENS {
                save_screen(&mut system, screen, &format!("/tmp/zakuro-{name}-{executed}.ppm"));
            }
        }
        if outcome != FrameOutcome::Completed {
            break;
        }
    }

    let elapsed = start.elapsed();
    println!("\n--- result ---");
    println!("outcome:       {outcome:?} after {executed} frames in {elapsed:.2?}");
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
            let path = format!("/tmp/zakuro-dump-0x{addr:08X}.bin");
            std::fs::write(&path, &bytes).unwrap();
            println!("dumped 0x{addr:08X}..+0x{len:X} -> {path}");
        }
    }

    println!("handles:       {}", system.kernel.handles.len());
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

    let hot = system.hot_spots(15);
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
        let path = format!("/tmp/zakuro-{name}.ppm");
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
    let pixels = system.read_screen(screen);
    let (width, height) = (screen.width() as usize, screen.height() as usize);
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

/// parses frame:BUTTON[+BUTTON...] entries separated by commas.
fn parse_input_script(spec: &str) -> Vec<(u64, PadState)> {
    spec.split(',')
        .filter_map(|entry| {
            let (frame, buttons) = entry.trim().split_once(':')?;
            let buttons = buttons.split('+').try_fold(PadState::empty(), |held, name| {
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
                    other => {
                        eprintln!("unknown button '{other}' in ZAKURO_INPUT");
                        return None;
                    }
                };
                Some(held | button)
            })?;
            Some((frame.trim().parse().ok()?, buttons))
        })
        .collect()
}
