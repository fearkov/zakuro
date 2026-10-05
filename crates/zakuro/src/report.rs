//! the log the app keeps in a file next to the saves, and what a report on
//! a game wants, put together to copy.

use std::collections::VecDeque;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::{Mutex, PoisonError};
use std::time::Instant;

use zakuro_core::System;

/// the lines a report takes from the end of the log.
const REPORT_LINES: usize = 40;
/// the lines kept for one, and for the file until it is open.
const KEPT: usize = 400;

struct Sink {
    file: Option<File>,
    recent: VecDeque<String>,
    /// the GPU the screen is drawn on, kept when the presenter says, it is
    /// long gone from the recent lines by the time a report is wanted.
    gpu: Option<String>,
}

impl Sink {
    fn keep(&mut self, line: String) {
        if self.recent.len() == KEPT {
            self.recent.pop_front();
        }
        self.recent.push_back(line);
    }
}

static SINK: Mutex<Sink> = Mutex::new(Sink { file: None, recent: VecDeque::new(), gpu: None });

/// the log as the terminal shows it, written to the file and kept for a
/// report as well.
struct Logger {
    terminal: env_logger::Logger,
    started: Instant,
}

impl log::Log for Logger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        self.terminal.enabled(metadata)
    }

    fn log(&self, record: &log::Record) {
        if !self.terminal.matches(record) {
            return;
        }
        self.terminal.log(record);
        let elapsed = self.started.elapsed().as_secs_f64();
        let line = format!("[{elapsed:9.3} {:<5} {}] {}", record.level(), record.target(), record.args());
        let gpu = record.target().starts_with("zakuro_gpu::backend").then(|| device(&record.args().to_string())).flatten();
        let mut sink = SINK.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(file) = &mut sink.file {
            let _ = writeln!(file, "{line}");
        }
        if gpu.is_some() {
            sink.gpu = gpu;
        }
        sink.keep(line);
    }

    fn flush(&self) {
        self.terminal.flush();
        if let Some(file) = &mut SINK.lock().unwrap_or_else(PoisonError::into_inner).file {
            let _ = file.flush();
        }
    }
}

/// logs what RUST_LOG asks for, info and up without it, and puts a panic in
/// the file as well, the terminal shows it as it always has.
pub fn init() {
    let terminal = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).build();
    let level = terminal.filter();
    let started = Instant::now();
    if log::set_boxed_logger(Box::new(Logger { terminal, started })).is_ok() {
        log::set_max_level(level);
    }
    let shown = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let thread = std::thread::current();
        let line = format!(
            "[{:9.3} PANIC] thread '{}' {}",
            started.elapsed().as_secs_f64(),
            thread.name().unwrap_or("<unnamed>"),
            info.to_string().replace('\n', " ")
        );
        // where it came from, for the file only
        let backtrace = std::backtrace::Backtrace::force_capture().to_string();
        let mut sink = SINK.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(file) = &mut sink.file {
            let _ = writeln!(file, "{line}\n{backtrace}");
        }
        sink.keep(line);
        drop(sink);
        shown(info);
    }));
}

/// the GPU a presenter's line names, Vulkan on the device, or OpenGL in its
/// version on the renderer, with which of the two.
fn device(message: &str) -> Option<String> {
    if let Some(rest) = message.strip_prefix("Vulkan on ") {
        return Some(format!("{} (Vulkan)", rest.split(',').next().unwrap_or(rest)));
    }
    let (_, renderer) = message.strip_prefix("OpenGL ")?.split_once(" on ")?;
    Some(format!("{renderer} (OpenGL)"))
}

/// writes the log to zakuro.log in the data folder from now on, starting
/// with what was logged before. the last run's stays as zakuro.previous.log.
pub fn to_file(dir: &Path) {
    let path = dir.join("zakuro.log");
    let _ = std::fs::create_dir_all(dir);
    let _ = std::fs::rename(&path, dir.join("zakuro.previous.log"));
    let Ok(mut file) = File::create(&path) else { return };
    let mut sink = SINK.lock().unwrap_or_else(PoisonError::into_inner);
    for line in &sink.recent {
        let _ = writeln!(file, "{line}");
    }
    sink.file = Some(file);
}

/// a report on a game, the fields the report channel asks for, those only
/// the player knows left to fill in, then the end of the log.
pub fn text(name: &str, system: &System) -> String {
    let (recent, gpu) = {
        let sink = SINK.lock().unwrap_or_else(PoisonError::into_inner);
        (sink.recent.iter().cloned().collect::<Vec<_>>(), sink.gpu.clone().unwrap_or_else(|| "unknown".into()))
    };
    let title = system.title.as_ref();
    let serial = title.map_or("unknown".into(), |title| format!("{:016X}", title.program_id()));
    let region = title.map_or("unknown", |title| region(&title.ncch.product_code));
    let mut out = String::new();
    let mut field = |key: &str, value: &str| out.push_str(&format!("{key}: {value}\n"));
    field("Name", name);
    field("Serial", &serial);
    field("Region", region);
    field("OS", &os());
    field("CPU", &cpu());
    field("GPU", &gpu);
    field("Zakuro version", concat!("v", env!("CARGO_PKG_VERSION")));
    field("Recompiled", if system.recompiled.is_some() { "yes" } else { "no" });
    field("Status", "");
    field("What happens", "");
    out.push_str("\n```\n");
    for line in &recent[recent.len().saturating_sub(REPORT_LINES)..] {
        out.push_str(line);
        out.push('\n');
    }
    out.push_str("```\n");
    out
}

/// the region a product code's last letter stands for, CTR-P-ARAP for
/// Europe.
fn region(product_code: &str) -> &'static str {
    match product_code.trim_end_matches('\0').chars().last() {
        Some('E') => "USA",
        Some('P') => "EUR",
        Some('J') => "JPN",
        Some('K') => "KOR",
        Some('C') => "CHN",
        Some('T') => "TWN",
        Some('A') => "World",
        _ => "unknown",
    }
}

/// the system's name, the distribution's on Linux.
fn os() -> String {
    let pretty = std::fs::read_to_string("/etc/os-release").ok().and_then(|release| {
        release.lines().find_map(|line| line.strip_prefix("PRETTY_NAME=").map(|name| name.trim_matches('"').to_owned()))
    });
    match (std::env::consts::OS, pretty) {
        ("linux", Some(name)) => format!("Linux ({name})"),
        ("windows", _) => "Windows".into(),
        ("macos", _) => "macOS".into(),
        (os, _) => os.into(),
    }
}

/// the processor's name, as it gives it.
fn cpu() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        let mut bytes = Vec::with_capacity(48);
        for leaf in 0x8000_0002u32..=0x8000_0004 {
            let regs = std::arch::x86_64::__cpuid(leaf);
            for word in [regs.eax, regs.ebx, regs.ecx, regs.edx] {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
        }
        let name = String::from_utf8_lossy(&bytes).trim_matches(char::from(0)).trim().to_owned();
        if !name.is_empty() {
            return name;
        }
    }
    "unknown".into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presenters_name_the_gpu() {
        assert_eq!(device("Vulkan on NVIDIA GeForce RTX 3060").as_deref(), Some("NVIDIA GeForce RTX 3060 (Vulkan)"));
        assert_eq!(device("Vulkan on AMD Radeon 780M, the renderer's device too").as_deref(), Some("AMD Radeon 780M (Vulkan)"));
        assert_eq!(
            device("OpenGL 4.6 (Core Profile) Mesa 25.2.3 on Mesa Intel(R) Xe Graphics (TGL GT2)").as_deref(),
            Some("Mesa Intel(R) Xe Graphics (TGL GT2) (OpenGL)")
        );
        assert_eq!(device("presenting at 60 Hz"), None);
    }

    #[test]
    fn product_codes_name_their_region() {
        assert_eq!(region("CTR-P-ARAP"), "EUR");
        assert_eq!(region("CTR-P-AJRE"), "USA");
        assert_eq!(region("CTR-P-ABCJ\0\0\0\0\0\0"), "JPN");
        assert_eq!(region(""), "unknown");
    }
}
