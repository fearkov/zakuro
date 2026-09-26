//! command line parsing.

use zakuro_gpu::RendererKind;

#[derive(Debug, Clone)]
pub struct Options {
    pub rom: String,
    pub renderer: RendererKind,
    /// window scale relative to the console's 400x480 combined screens.
    pub scale: u32,
    /// run without a window for this many frames, then report.
    pub headless: Option<u64>,
    pub profile: bool,
    pub new3ds: bool,
    /// skip loading a ROM entirely and just paint both screens solid colors.
    pub test_pattern: bool,
    /// a library of recompiled code for the title, or a directory of them.
    pub recompiled: Option<String>,
}

const USAGE: &str = "\
zakuro - a high-level-emulation Nintendo 3DS emulator

usage: zakuro <rom.3ds|.cxi> [options]

options:
  --renderer <vulkan|gl|software>  presentation backend (default: vulkan,
                                   or gl where Vulkan does not start)
  --scale <n>                      window scale factor (default: 2)
  --headless <frames>              run without a window and print a report
  --new3ds                         emulate a New 3DS
  --profile                        collect a sampling profile and print it
  --recompiled <path>              run code 3dsrecomp built for the title, a
                                   library or a directory holding <title id>.so
  -h, --help                       show this message

The system font cannot be generated: put a dump at sysdata/shared_font.bin for
titles that render text with it.
";

pub fn parse() -> Result<Options, String> {
    let mut rom = None;
    let mut options = Options {
        rom: String::new(),
        renderer: RendererKind::Vulkan,
        scale: 2,
        headless: None,
        profile: false,
        new3ds: false,
        test_pattern: false,
        recompiled: None,
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
                options.renderer = RendererKind::parse(&value)
                    .ok_or_else(|| format!("unknown renderer '{value}'"))?;
            }
            "--scale" => {
                let value = args.next().ok_or("--scale needs a value")?;
                options.scale = value
                    .parse()
                    .map_err(|_| format!("'{value}' is not a scale"))?;
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
            "--recompiled" => {
                options.recompiled = Some(args.next().ok_or("--recompiled needs a path")?);
            }
            "--profile" => options.profile = true,
            "--test-pattern" => options.test_pattern = true,
            other if other.starts_with('-') => {
                return Err(format!("unknown option '{other}'"));
            }
            path => rom = Some(path.to_owned()),
        }
    }

    if !options.test_pattern {
        options.rom = rom.ok_or_else(|| {
            print!("{USAGE}");
            "no ROM given".to_owned()
        })?;
    }
    Ok(options)
}
