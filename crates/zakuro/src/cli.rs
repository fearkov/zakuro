//! command line parsing

use std::path::PathBuf;

use zakuro_gpu::RendererKind;

#[derive(Debug, Clone)]
pub struct Options {
    /// the game to start, none to open the library
    pub rom: Option<PathBuf>,
    /// the settings' choice when not given
    pub renderer: Option<RendererKind>,
    /// window scale relative to the console's 400x480 combined screens, the settings' when not given
    pub scale: Option<u32>,
    /// run w/o a window for this many frames, then report
    pub headless: Option<u64>,
    pub profile: bool,
    pub new3ds: bool,
    /// skip loading a ROM entirely and just paint both screens solid colors
    pub test_pattern: bool,
    /// a library of recompiled code for the title, or a directory of them
    pub recompiled: Option<String>,
    /// draw on the host GPU instead of in software, the settings' choice when not given
    pub hardware_rasterizer: Option<bool>,
    /// where saves live, instead of the usual place
    pub data: Option<String>,
    /// interpret everything, whatever recompiled code there is
    pub interpreter: bool,
    /// play no sound (shh)
    pub mute: bool,
    /// start in full screen, like F11
    pub fullscreen: bool,
    /// where the window opens, so full screen lands on that screen
    pub at: Option<(i32, i32)>,
    /// pads only play while the window's in front, so a front end can show its own window on top
    pub focus_pad: bool,
    /// fast forward at most this many times the console's speed, no cap when none
    pub fast_forward: Option<u32>,
    /// after a headless run, save the state to this file
    pub state_save: Option<String>,
    /// start the game from this state, booting as usual when it won't load
    pub state_load: Option<String>,
}

const USAGE: &str = "\
zakuro - a high-level-emulation Nintendo 3DS emulator

usage: zakuro [rom.3ds|.cxi] [options]

Without a game it opens the library, the games in a folder you pick. The
options below override the settings for this run.

options:
  --renderer <vulkan|gl|software>  presentation backend (default: vulkan,
                                   or gl where Vulkan does not start)
  --rasterizer <hardware|software> where the 3D is drawn (default: hardware,
                                   the host GPU through Vulkan, or software
                                   where that does not start)
  --scale <n>                      window scale factor (default: 2)
  (Esc opens the menu over a game, F1 pauses, F11 goes fullscreen)
  --headless <frames>              run without a window and print a report
  --new3ds                         emulate a New 3DS
  --profile                        collect a sampling profile and print it
  --recompiled <path>              run code 3dsrecomp built for the title, a
                                   library or a directory holding <title id>.so
                                   (.dylib on macOS, .dll on Windows, default:
                                   the one 3dsrecomp build installed for it, in
                                   the system's place for data,
                                   ~/.local/share/3dsrecomp,
                                   ~/Library/Application Support/3dsrecomp or
                                   %APPDATA%\\3dsrecomp)
  --interpreter                    interpret everything, even with recompiled
                                   code around
  --mute                           play no sound
  --fullscreen                     start in full screen
  --at <x,y>                       open the window there, so full screen is on that screen
  --focus-pad                      controllers only play while the window is in front
  --fast-forward <n>               fast forward at most n times the console's speed
  --state-save <file>              with --headless, save the state to file after the frames
  --state-load <file>              start the game from a state saved to file
  --data <dir>                     where saves live (default: the system's
                                   place for data, ~/.local/share/zakuro,
                                   ~/Library/Application Support/zakuro or
                                   %APPDATA%\\zakuro)
  -h, --help                       show this message

The system font cannot be generated: put a dump at sysdata/shared_font.bin, in
the working directory or where saves live, for titles that render text with it.
";

pub fn parse() -> Result<Options, String> {
    let mut rom = None;
    let mut options = Options {
        rom: None,
        renderer: None,
        scale: None,
        headless: None,
        profile: false,
        new3ds: false,
        test_pattern: false,
        recompiled: None,
        hardware_rasterizer: None,
        data: None,
        interpreter: false,
        mute: false,
        fullscreen: false,
        at: None,
        focus_pad: false,
        fast_forward: None,
        state_save: None,
        state_load: None,
    };

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            "--renderer" => {
                let value = args.next().ok_or("--renderer needs a value")?;
                options.renderer = Some(
                    RendererKind::parse(&value).ok_or_else(|| format!("unknown renderer '{value}'"))?,
                );
            }
            "--scale" => {
                let value = args.next().ok_or("--scale needs a value")?;
                options.scale = Some(value.parse().map_err(|_| format!("'{value}' is not a scale"))?);
            }
            "--headless" => {
                let value = args.next().ok_or("--headless needs a frame count")?;
                options.headless = Some(
                    value
                        .parse()
                        .map_err(|_| format!("'{value}' is not a frame count"))?,
                );
            }
            "--new3ds" => options.new3ds = true,
            "--rasterizer" => {
                let value = args.next().ok_or("--rasterizer needs a value")?;
                options.hardware_rasterizer = Some(match value.as_str() {
                    "hardware" | "vulkan" | "gpu" => true,
                    "software" | "cpu" => false,
                    other => return Err(format!("unknown rasterizer '{other}'")),
                });
            }
            "--recompiled" => {
                options.recompiled = Some(args.next().ok_or("--recompiled needs a path")?);
            }
            "--data" => {
                options.data = Some(args.next().ok_or("--data needs a directory")?);
            }
            "--interpreter" => options.interpreter = true,
            "--mute" => options.mute = true,
            "--fullscreen" => options.fullscreen = true,
            "--focus-pad" => options.focus_pad = true,
            "--state-save" => options.state_save = Some(args.next().ok_or("--state-save needs a file")?),
            "--state-load" => options.state_load = Some(args.next().ok_or("--state-load needs a file")?),
            "--fast-forward" => {
                let times = args.next().and_then(|v| v.parse::<u32>().ok()).filter(|&n| n >= 2);
                options.fast_forward = Some(times.ok_or("--fast-forward takes a speed of 2 or more")?);
            }
            "--at" => {
                let at = args.next().and_then(|v| {
                    let (x, y) = v.split_once(',')?;
                    Some((x.trim().parse().ok()?, y.trim().parse().ok()?))
                });
                options.at = Some(at.ok_or("--at takes x,y, as numbers")?);
            }
            "--profile" => options.profile = true,
            "--test-pattern" => options.test_pattern = true,
            // macOS hands an app it hasn't run b4 a process number
            other if cfg!(target_os = "macos") && other.starts_with("-psn_") => {}
            other if other.starts_with('-') => {
                return Err(format!("unknown option '{other}'"));
            }
            path => rom = Some(PathBuf::from(path)),
        }
    }

    options.rom = rom;
    if options.headless.is_some() && options.rom.is_none() && !options.test_pattern {
        print!("{USAGE}");
        return Err("--headless needs a game".to_owned());
    }
    Ok(options)
}
