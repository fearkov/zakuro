//! cargo run -p zakuro-core --example boot -- <rom> [frames]

use std::collections::BTreeSet;

use zakuro_core::services::hid::{InputState, PadState};
use zakuro_core::{loader, Config, FrameOutcome};

fn main() {
    // Cranelift's JIT logs each shader it compiles whole at info
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info,cranelift_jit=warn")).init();

    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: boot <rom> [frames]");
        std::process::exit(2);
    };
    let frames: u64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(60);

    let config = Config {
        // ZAKURO_NEW3DS=1 runs the title on a New 3DS.
        new3ds: std::env::var("ZAKURO_NEW3DS").is_ok_and(|v| v == "1"),
        // ZAKURO_NICK=name names the console, which tells it apart over
        // local wireless too
        username: std::env::var("ZAKURO_NICK").unwrap_or_else(|_| "Zakuro".into()),
        console_id: std::env::var("ZAKURO_NICK").map_or(0x5A4B_0000_0000_0001, |nick| {
            nick.bytes().fold(0xCBF2_9CE4_8422_2325u64, |hash, byte| (hash ^ byte as u64).wrapping_mul(0x0100_0000_01B3))
        }),
        language: std::env::var("ZAKURO_LANGUAGE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(zakuro_core::services::cfg::LANGUAGE_ENGLISH),
        // ZAKURO_RECOMPILED=path runs the code 3dsrecomp built for the title.
        recompiled: std::env::var("ZAKURO_RECOMPILED").ok().map(Into::into),
        // ZAKURO_FIND_RECOMPILED=1 runs the library 3dsrecomp installed for
        // it, as the frontend does.
        find_recompiled: std::env::var_os("ZAKURO_FIND_RECOMPILED").is_some(),
        // ZAKURO_RASTERIZER=hardware draws on the host GPU.
        hardware_renderer: std::env::var("ZAKURO_RASTERIZER").is_ok_and(|v| v == "hardware"),
        // ZAKURO_RESOLUTION=3 draws at three times the console's resolution.
        resolution: std::env::var("ZAKURO_RESOLUTION").ok().and_then(|v| v.parse().ok()).unwrap_or(1),
        // ZAKURO_UPDATE=file.cia runs the title with its update, and
        // ZAKURO_DLC=a.cia:b.cia with its DLC.
        update: std::env::var_os("ZAKURO_UPDATE").map(Into::into),
        dlc: std::env::var_os("ZAKURO_DLC").map(|paths| std::env::split_paths(&paths).collect()).unwrap_or_default(),
        // ZAKURO_DATA=dir keeps saves in dir/user and takes mods from
        // dir/mods, the working directory's user folder and no mods without it.
        data_dir: std::env::var_os("ZAKURO_DATA").map(Into::into),
        // ZAKURO_CLOCK=milliseconds since 1900 starts the clock there, so that
        // runs repeat exactly.
        clock: std::env::var("ZAKURO_CLOCK").ok().and_then(|v| v.parse().ok()),
        // ZAKURO_PRESENT=direct draws on a device of the kind a Vulkan
        // presenter shares, which shows the screens straight from the GPU.
        // ZAKURO_INTERPOLATE=1 draws the pictures in between, which takes
        // ZAKURO_PRESENT=direct, and saves them with the screens
        interpolate: std::env::var("ZAKURO_INTERPOLATE").is_ok_and(|v| v == "1"),
        skip_idle_loops: std::env::var("ZAKURO_IDLE").map_or(true, |v| v != "0"),
        // ZAKURO_3D=0.5 is where the 3D slider is, the right eye's picture
        // saved with the screens when the game draws one
        slider_3d: std::env::var("ZAKURO_3D").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0),
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

    // ZAKURO_ENHANCE="60 FPS" turns on the game's enhancements of those
    // names, separated by commas, and ZAKURO_ENHANCE_OFF_AT=frame turns them
    // all off there, as a player does in the cheats window
    let enhance = std::env::var("ZAKURO_ENHANCE").unwrap_or_default();
    for cheat in system.cheats.iter_mut().filter(|cheat| cheat.builtin && enhance.split(',').any(|name| name.trim() == cheat.name)) {
        cheat.enabled = true;
        println!("enhancement {} on", cheat.name);
    }
    let enhance_off_at: Option<u64> = std::env::var("ZAKURO_ENHANCE_OFF_AT").ok().and_then(|f| f.parse().ok());

    // ZAKURO_ROOM_HOST=port opens a room and joins it, ZAKURO_ROOM=host:port
    // joins one, under ZAKURO_NICK, for local wireless with another run
    let _server = std::env::var("ZAKURO_ROOM_HOST").ok().and_then(|port| port.parse().ok()).map(|port| {
        zakuro_core::room::Server::start(port, "boot", "", 8).unwrap_or_else(|error| {
            eprintln!("could not open a room on port {port}: {error}");
            std::process::exit(1);
        })
    });
    let address = std::env::var("ZAKURO_ROOM").ok().or_else(|| _server.as_ref().map(|server| format!("127.0.0.1:{}", server.port())));
    let _connection = address.map(|address| {
        let nickname = std::env::var("ZAKURO_NICK").unwrap_or_else(|_| format!("boot{}", std::process::id()));
        let connection = zakuro_core::room::Connection::join(&address, &nickname, "").unwrap_or_else(|error| {
            eprintln!("could not join the room at {address}: {error}");
            std::process::exit(1);
        });
        system.room = Some(connection.link());
        println!("in the room at {address} as {nickname}");
        connection
    });

    // ZAKURO_CAMERA=picture.ppm is what the cameras see, a binary PPM,
    // whole, the way a QR code is shown to them.
    if let Some(path) = std::env::var_os("ZAKURO_CAMERA") {
        match std::fs::read(&path).ok().and_then(|bytes| read_ppm(&bytes)) {
            Some(picture) => system.camera = Some(Box::new(zakuro_core::camera::Still(std::sync::Arc::new(picture)))),
            None => eprintln!("could not read {} as a binary PPM", path.to_string_lossy()),
        }
    }

    // ZAKURO_MIC=noise is what the microphone hears, blowing into it say,
    // or ZAKURO_MIC=sound.wav, a WAV of 16-bit samples over and over, and
    // ZAKURO_MIC_FRAMES=900-1200 hears it only between those frames
    let mic_frame = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    if let Some(source) = std::env::var_os("ZAKURO_MIC") {
        let heard: Option<Box<dyn zakuro_core::microphone::Microphone>> = if source == "noise" {
            Some(Box::new(zakuro_core::microphone::Noise::new(0.8)))
        } else {
            std::fs::read(&source)
                .ok()
                .and_then(|bytes| zakuro_core::microphone::Recording::from_wav(&bytes))
                .map(|recording| Box::new(recording) as Box<dyn zakuro_core::microphone::Microphone>)
        };
        match heard {
            Some(heard) => {
                let frames = std::env::var("ZAKURO_MIC_FRAMES").ok().and_then(|spec| {
                    let (from, to) = spec.split_once('-')?;
                    Some(from.parse().ok()?..to.parse().ok()?)
                });
                system.microphone = Some(Box::new(Between { heard, frames, frame: mic_frame.clone() }));
            }
            None => eprintln!("could not read {} as a WAV of 16-bit samples", source.to_string_lossy()),
        }
    }

    println!("--- booting for {frames} frames ---");
    let start = std::time::Instant::now();
    // how long each frame took, which shows the stutter an average hides
    let mut frame_times = Vec::with_capacity(frames as usize);
    let mut outcome = FrameOutcome::Completed;
    let mut executed = 0u64;
    // refreshes that showed a picture in between, for frame interpolation
    let mut betweens = 0u64;

    // headless runs otherwise never press a button, so a title parked on an
    // intro or "press start" screen, which is most titles, most of the time,
    // would sit there for the entire run no matter how many frames are given.
    let mash_buttons = std::env::var("ZAKURO_MASH_BUTTONS").is_ok();
    // mashing up to a frame and following ZAKURO_INPUT from there, to get
    // through the screens before one quickly and then act on it,
    // ZAKURO_MASH_UNTIL=4400 with ZAKURO_MASH_BUTTONS.
    let mash_until: Option<u64> = std::env::var("ZAKURO_MASH_UNTIL").ok().and_then(|f| f.parse().ok());

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
    // ZAKURO_SWAP_AT=frame:library switches the running game to that library
    // at that frame, as the frontend does when a recompile finishes, and
    // with frame:library:built it first moves built over library the way
    // 3dsrecomp installs one.
    let swap: Option<(u64, String, Option<String>)> = std::env::var("ZAKURO_SWAP_AT").ok().and_then(|spec| {
        let mut parts = spec.splitn(3, ':');
        Some((parts.next()?.parse().ok()?, parts.next()?.to_owned(), parts.next().map(str::to_owned)))
    });
    // ZAKURO_SAVE_AT=frame:file saves the state at that frame, and
    // ZAKURO_LOAD=file goes on from a saved state, at its frame
    let save_at: Option<(u64, String)> = std::env::var("ZAKURO_SAVE_AT").ok().and_then(|spec| {
        let (at, file) = spec.split_once(':')?;
        Some((at.parse().ok()?, file.to_owned()))
    });
    let mut first = 0;
    if let Some(file) = std::env::var_os("ZAKURO_LOAD") {
        let started = std::time::Instant::now();
        match std::fs::read(&file).map_err(|error| error.to_string()).and_then(|bytes| system.load_state(&bytes)) {
            Ok(()) => {
                first = system.frames;
                println!("loaded the state at frame {first} in {:?}", started.elapsed());
            }
            Err(error) => {
                eprintln!("could not load {}: {error}", file.to_string_lossy());
                std::process::exit(1);
            }
        }
    }
    // ZAKURO_LOAD_AT=frame:file loads a saved state at that frame, once,
    // and goes on from the state's frame
    let mut load_at: Option<(u64, String)> = std::env::var("ZAKURO_LOAD_AT").ok().and_then(|spec| {
        let (at, file) = spec.split_once(':')?;
        Some((at.parse().ok()?, file.to_owned()))
    });
    let mut frame = first;
    while frame < frames {
        if enhance_off_at == Some(frame) {
            for cheat in system.cheats.iter_mut().filter(|cheat| cheat.builtin && cheat.enabled) {
                cheat.enabled = false;
                println!("frame {frame}: enhancement {} off", cheat.name);
            }
        }
        if let Some((_, file)) = load_at.take_if(|(at, _)| *at == frame) {
            match std::fs::read(&file).map_err(|error| error.to_string()).and_then(|bytes| system.load_state(&bytes)) {
                Ok(()) => {
                    println!("frame {frame}: loaded the state, back at frame {}", system.frames);
                    frame = system.frames;
                }
                Err(error) => println!("frame {frame}: could not load the state, {error}"),
            }
        }
        if let Some((_, file)) = save_at.as_ref().filter(|(at, _)| *at == frame) {
            let started = std::time::Instant::now();
            match system.save_state().map_err(|error| error.to_string()).and_then(|bytes| {
                std::fs::write(file, &bytes).map_err(|error| error.to_string()).map(|()| bytes.len())
            }) {
                Ok(length) => println!("frame {frame}: saved the state, {} KiB, in {:?}", length / 1024, started.elapsed()),
                Err(error) => println!("frame {frame}: could not save the state, {error}"),
            }
        }
        if let Some((at, library, built)) = swap.as_ref().filter(|(at, ..)| *at == frame) {
            if let Some(built) = built {
                std::fs::rename(built, library).expect("moving the new library into place");
            }
            match loader::swap_recompiled(&mut system, std::path::Path::new(library)) {
                Ok(()) => println!("frame {at}: switched to {library}"),
                Err(error) => println!("frame {at}: could not switch, {error}"),
            }
            // which files of the library are mapped now, a deleted one is
            // the old library still loaded
            let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
            let name = std::path::Path::new(library).file_name().unwrap().to_string_lossy().into_owned();
            let mapped: std::collections::BTreeSet<String> = maps
                .lines()
                .filter(|line| line.contains(&name))
                .map(|line| line.split_whitespace().skip(5).collect::<Vec<_>>().join(" "))
                .collect();
            println!("frame {at}: mapped {mapped:?}");
        }
        if profile_from == Some(frame) {
            system.enable_profiler();
        }
        if log_from == Some(frame) {
            log::set_max_level(log::LevelFilter::Trace);
            println!("--- verbose logging from frame {frame} ---");
        }
        let mashing = mash_buttons && (script.is_empty() || mash_until.is_some_and(|until| frame < until));
        if let Some(replay) = &mut replay {
            system.set_input(replay.input(frame));
        } else if !script.is_empty() && !mashing {
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
                c_stick_x: axis(PadState::C_STICK_RIGHT, PadState::C_STICK_LEFT),
                c_stick_y: axis(PadState::C_STICK_UP, PadState::C_STICK_DOWN),
                ..InputState::default()
            });
        } else if mashing {
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
        mic_frame.store(frame, std::sync::atomic::Ordering::Relaxed);
        let frame_start = std::time::Instant::now();
        let faults = system.memory.fault_summary().len();
        outcome = system.run_frame();
        executed = frame + 1;
        betweens += system.shows_between() as u64;
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
            if let Some(right) = system.read_right_eye_scaled() {
                save_picture(&right, &temp(&format!("zakuro-top-right-{executed}.ppm")));
            }
            for (screen, name) in SCREENS {
                save_screen(&mut system, screen, &temp(&format!("zakuro-{name}-{executed}.ppm")));
                // and as the GPU has them, with the pictures in between
                if system.gpu.interpolates() {
                    if let Some(own) = system.upright_screen(screen, false) {
                        save_picture(&own, &temp(&format!("zakuro-{name}-gpu-{executed}.ppm")));
                    }
                    if let Some(between) = system.upright_screen(screen, true) {
                        save_picture(&between, &temp(&format!("zakuro-{name}-between-{executed}.ppm")));
                    }
                }
            }
            if let Some(matches) = system.interpolation_matches() {
                println!("frame {executed}: {matches:?}, in between {}", system.shows_between());
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
        frame += 1;
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
    if system.gpu.interpolates() {
        println!("in between:    {betweens} of {executed} refreshes");
    }
    println!("instructions:  {}", system.cpu.cycles);
    if system.idle.skipped > 0 {
        println!("idle skipped:  {} cycles of loops that changed nothing", system.idle.skipped);
    }
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

    let threads = system.profiled_threads();
    if !threads.is_empty() {
        let total: u64 = threads.iter().map(|(_, hits)| hits).sum();
        println!("profiled threads:");
        for (thread, hits) in threads {
            println!("  {thread:<8} {:5.1}%", hits as f64 * 100.0 / total as f64);
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
            "{name} framebuffer: A 0x{:08X} B 0x{:08X} active {} (showing 0x{:08X}, right eye 0x{:08X}) \
             stride {} format 0x{:X}",
            config.address_a_left,
            config.address_b_left,
            config.active,
            config.address_left(),
            config.address_right(),
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

/// saves a picture as a PPM.
fn save_picture((pixels, width, height): &zakuro_core::Screen, path: &str) {
    let mut ppm = format!("P6\n{width} {height}\n255\n").into_bytes();
    for chunk in pixels.as_chunks::<4>().0 {
        ppm.extend_from_slice(&chunk[..3]);
    }
    let _ = std::fs::write(path, ppm);
}

/// a binary PPM's picture, P6 with eight bit channels.
fn read_ppm(bytes: &[u8]) -> Option<zakuro_core::camera::Picture> {
    // the magic, the width, the height and the largest value, apart by
    // white space, then one more and the pixels
    let mut fields = Vec::new();
    let mut at = 0;
    while fields.len() < 4 {
        while bytes.get(at)?.is_ascii_whitespace() {
            at += 1;
        }
        let start = at;
        while !bytes.get(at)?.is_ascii_whitespace() {
            at += 1;
        }
        fields.push(std::str::from_utf8(&bytes[start..at]).ok()?);
    }
    let (width, height): (u32, u32) = (fields[1].parse().ok()?, fields[2].parse().ok()?);
    if fields[0] != "P6" || fields[3] != "255" {
        return None;
    }
    let rgb = bytes.get(at + 1..at + 1 + (width * height * 3) as usize)?.to_vec();
    Some(zakuro_core::camera::Picture { width, height, rgb, fit: zakuro_core::camera::Fit::Whole })
}

/// a file in the system's temporary directory, /tmp on Linux.
fn temp(name: &str) -> String {
    std::env::temp_dir().join(name).display().to_string()
}

/// parses frame:BUTTON[+BUTTON...] entries separated by commas, where @XxY
/// touches the bottom screen at X, Y, CUP, CDOWN, CLEFT and CRIGHT push
/// the circle pad and CSUP, CSDOWN, CSLEFT and CSRIGHT the C-stick.
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
                    "CSUP" => PadState::C_STICK_UP,
                    "CSDOWN" => PadState::C_STICK_DOWN,
                    "CSLEFT" => PadState::C_STICK_LEFT,
                    "CSRIGHT" => PadState::C_STICK_RIGHT,
                    "ZL" => PadState::ZL,
                    "ZR" => PadState::ZR,
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

/// a microphone that hears its sound only between some frames and silence
/// otherwise, and says when the title starts and stops sampling.
struct Between {
    heard: Box<dyn zakuro_core::microphone::Microphone>,
    frames: Option<std::ops::Range<u64>>,
    frame: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl zakuro_core::microphone::Microphone for Between {
    fn sampling(&mut self, on: bool) {
        let frame = self.frame.load(std::sync::atomic::Ordering::Relaxed);
        println!("frame {frame}: the title {} sampling", if on { "starts" } else { "stops" });
        self.heard.sampling(on);
    }

    fn read(&mut self, rate: f64, out: &mut [f32]) {
        let frame = self.frame.load(std::sync::atomic::Ordering::Relaxed);
        if self.frames.as_ref().is_none_or(|frames| frames.contains(&frame)) {
            self.heard.read(rate, out);
        } else {
            out.fill(0.0);
        }
    }
}
