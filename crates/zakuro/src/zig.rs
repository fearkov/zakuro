//! Zig's C compiler, zig cc, downloaded into Zakuro's own folder when the
//! computer has no C compiler and the player says yes, so that recompiling
//! a game needs nothing installed by hand.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use recomp3ds::compile::Compiler;

/// the release Zakuro downloads.
const VERSION: &str = "0.17.0";

/// the list of mirrors the Zig project asks tools to download from, before
/// ziglang.org itself.
const MIRRORS: &str = "https://ziglang.org/download/community-mirrors.txt";

/// a release's archive for one system, as ziglang.org/download/index.json
/// gives it.
struct Archive {
    name: &'static str,
    size: u64,
    sha256: &'static str,
}

/// the archive for the system Zakuro runs on, none where Zig has none.
fn archive() -> Option<Archive> {
    let (name, size, sha256) = match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => ("zig-x86_64-linux-0.17.0.tar.xz", 57_332_648, "1cbe9df9f27e6b78d14ccbca43b6703a404ef79ef1c463de901d7f088d4e2026"),
        ("linux", "aarch64") => ("zig-aarch64-linux-0.17.0.tar.xz", 52_877_280, "9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8"),
        ("windows", "x86_64") => ("zig-x86_64-windows-0.17.0.zip", 100_268_137, "b5663f69581dcf391293fbf16c06cb80d81d806545ce618b4d0bab7f0eb8c428"),
        ("windows", "aarch64") => ("zig-aarch64-windows-0.17.0.zip", 95_992_853, "0a59d91fa1cb40cf068e9b0954434ce973500c7a2ea749f1e01af62cdab52d26"),
        ("macos", "x86_64") => ("zig-x86_64-macos-0.17.0.tar.xz", 59_317_572, "4f9a1c5269aa17ebda5e6d3c2b89d6cbf36f7d2b22a0306e9ab98f25f95529c6"),
        ("macos", "aarch64") => ("zig-aarch64-macos-0.17.0.tar.xz", 53_985_220, "b607e9b9234790a008116ae5bdb71c6243b84b9fb42a53a9e70fde41c06c536a"),
        _ => return None,
    };
    Some(Archive { name, size, sha256 })
}

/// how many bytes the download is, none when there is nothing to download
/// for this system.
pub fn download_size() -> Option<u64> {
    archive().map(|archive| archive.size)
}

/// where Zakuro keeps the tools it downloads: the local application data on
/// Windows, which does not roam with the user, and the data folder
/// elsewhere.
pub fn tools_dir(data_dir: Option<&Path>) -> Option<PathBuf> {
    if cfg!(windows) {
        let local = std::env::var_os("LOCALAPPDATA").filter(|dir| !dir.is_empty())?;
        Some(PathBuf::from(local).join("zakuro").join("tools"))
    } else {
        data_dir.map(|dir| dir.join("tools"))
    }
}

fn folder(tools: &Path) -> PathBuf {
    tools.join(format!("zig-{VERSION}"))
}

/// the compiler, when it has been downloaded into tools.
pub fn installed(tools: &Path) -> Option<Compiler> {
    let program = folder(tools).join(if cfg!(windows) { "zig.exe" } else { "zig" });
    program.is_file().then(|| Compiler::new(program, &["cc"]))
}

/// the program downloads go through, curl, which Linux, macOS and Windows
/// 10 and later mostly have, or else wget, which some Linux systems have
/// instead.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Fetcher {
    Curl,
    Wget,
}

impl Fetcher {
    fn find() -> Option<Fetcher> {
        let runs = |program: &str| {
            Command::new(program).arg("--version").stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok_and(|status| status.success())
        };
        [Fetcher::Curl, Fetcher::Wget].into_iter().find(|fetcher| runs(fetcher.program()))
    }

    fn program(self) -> &'static str {
        match self {
            Fetcher::Curl => "curl",
            Fetcher::Wget => "wget",
        }
    }

    /// a command downloading url to path, or to its output without one,
    /// giving up after seconds.
    fn command(self, url: &str, path: Option<&Path>, seconds: u32) -> Command {
        let mut command = Command::new(self.program());
        match self {
            Fetcher::Curl => {
                command.args(["-fL", "--silent", "--show-error", "--connect-timeout", "20", "--max-time", &seconds.to_string()]);
                if let Some(path) = path {
                    command.arg("-o").arg(path);
                }
            }
            // wget has no limit on the whole download, a stall ends it
            Fetcher::Wget => {
                command.args(["--quiet", "--timeout=20", &format!("--read-timeout={}", seconds.min(60)), "-O"]);
                match path {
                    Some(path) => command.arg(path),
                    None => command.arg("-"),
                };
            }
        }
        command.arg(url);
        command
    }
}

/// whether Zakuro can download Zig here, having curl or wget to.
pub fn can_download() -> bool {
    Fetcher::find().is_some()
}

/// downloads Zig into tools and unpacks it, telling progress how much of the
/// archive has arrived, 0 to 1. the mirrors the Zig project lists go first,
/// in a random order, and ziglang.org last, as it asks of tools, and what
/// arrives has to match the SHA-256 above whichever served it.
pub fn download(tools: &Path, progress: &dyn Fn(f32), cancel: &AtomicBool) -> Result<Compiler, String> {
    let archive = archive().ok_or("Zakuro has no compiler to download for this system")?;
    let fetcher = Fetcher::find().ok_or("Zakuro downloads with curl or wget, and this computer has neither")?;
    std::fs::create_dir_all(tools).map_err(|error| format!("could not create {}, {error}", tools.display()))?;
    let partial = tools.join(format!("{}.part", archive.name));
    let mut sources = mirrors(fetcher);
    sources.push(format!("https://ziglang.org/download/{VERSION}"));
    let mut last = String::new();
    for source in sources {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let url = format!("{}/{}", source.trim_end_matches('/'), archive.name);
        log::info!("downloading Zig from {url}");
        match fetch(fetcher, &url, &partial, archive.size, progress, cancel).and_then(|()| verify(&partial, &archive)) {
            Ok(()) => {
                let unpacked = unpack(&partial, tools);
                let _ = std::fs::remove_file(&partial);
                unpacked?;
                return installed(tools).ok_or_else(|| "Zig was unpacked but its zig program is missing".to_owned());
            }
            Err(error) => {
                log::warn!("Zig from {url}: {error}");
                last = error;
            }
        }
    }
    let _ = std::fs::remove_file(&partial);
    if cancel.load(Ordering::Relaxed) {
        return Err("cancelled".to_owned());
    }
    Err(format!("could not download Zig, from ziglang.org or any of its mirrors, the last said {last}"))
}

/// the mirrors the Zig project lists, in a random order, none when the list
/// can't be had.
fn mirrors(fetcher: Fetcher) -> Vec<String> {
    let listed = fetcher.command(MIRRORS, None, 30).stderr(Stdio::null()).output();
    let mut mirrors: Vec<String> = match listed {
        Ok(output) if output.status.success() => String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(str::trim)
            .filter(|line| line.starts_with("https://"))
            .map(str::to_owned)
            .collect(),
        _ => Vec::new(),
    };
    // a shuffle seeded by the clock, so that the downloads spread over them
    let mut seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |time| time.as_nanos() as u64) | 1;
    for i in (1..mirrors.len()).rev() {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        mirrors.swap(i, (seed % (i as u64 + 1)) as usize);
    }
    mirrors
}

/// downloads url to path, following how much of size has arrived.
fn fetch(fetcher: Fetcher, url: &str, path: &Path, size: u64, progress: &dyn Fn(f32), cancel: &AtomicBool) -> Result<(), String> {
    let _ = std::fs::remove_file(path);
    let mut child = fetcher
        .command(url, Some(path), 1800)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("{} could not start, {error}", fetcher.program()))?;
    loop {
        if cancel.load(Ordering::Relaxed) {
            let _ = child.kill();
            let _ = child.wait();
            return Err("cancelled".to_owned());
        }
        let arrived = std::fs::metadata(path).map_or(0, |metadata| metadata.len());
        progress((arrived as f32 / size as f32).min(1.0));
        match child.try_wait() {
            Ok(Some(status)) if status.success() => return Ok(()),
            Ok(Some(_)) => {
                let mut error = String::new();
                if let Some(mut stderr) = child.stderr.take() {
                    let _ = std::io::Read::read_to_string(&mut stderr, &mut error);
                }
                return Err(error.trim().to_owned());
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(250)),
            Err(error) => return Err(error.to_string()),
        }
    }
}

/// whether the archive at path is the one Zakuro knows, by its size and its
/// SHA-256.
fn verify(path: &Path, archive: &Archive) -> Result<(), String> {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path).map_err(|error| error.to_string())?;
    let size = file.metadata().map_err(|error| error.to_string())?.len();
    if size != archive.size {
        return Err(format!("{size} bytes arrived, not {}", archive.size));
    }
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher).map_err(|error| error.to_string())?;
    let hash: String = hasher.finalize().iter().map(|byte| format!("{byte:02x}")).collect();
    if hash != archive.sha256 {
        return Err(format!("its SHA-256 is {hash}, not the one Zakuro knows"));
    }
    Ok(())
}

/// unpacks the archive into its folder in tools with tar, which reads the
/// zip on Windows too. the archive holds a single folder, renamed to
/// zig-<version> once it is all there.
fn unpack(archive: &Path, tools: &Path) -> Result<(), String> {
    let staging = tools.join(format!("zig-{VERSION}.unpacking"));
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging).map_err(|error| error.to_string())?;
    let xz = archive.to_string_lossy().ends_with(".tar.xz.part");
    let status = Command::new("tar")
        .arg(if xz { "-xJf" } else { "-xf" })
        .arg(archive)
        .arg("-C")
        .arg(&staging)
        .status()
        .map_err(|error| format!("tar could not start, {error}"))?;
    if !status.success() {
        let _ = std::fs::remove_dir_all(&staging);
        return Err("tar could not unpack it".to_owned());
    }
    let inner = std::fs::read_dir(&staging)
        .map_err(|error| error.to_string())?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .find(|path| path.is_dir())
        .ok_or("the archive held no folder")?;
    let target = folder(tools);
    let _ = std::fs::remove_dir_all(&target);
    std::fs::rename(&inner, &target).map_err(|error| error.to_string())?;
    let _ = std::fs::remove_dir_all(&staging);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_archive_is_named_for_the_release() {
        let archive = archive().expect("a Zig for the system the tests run on");
        assert!(archive.name.contains(VERSION));
        assert_eq!(archive.sha256.len(), 64);
    }

    #[test]
    fn a_downloaded_zig_is_called_as_zig_cc() {
        let tools = std::env::temp_dir().join(format!("zakuro-zig-{}", std::process::id()));
        assert!(installed(&tools).is_none());
        let program = folder(&tools).join(if cfg!(windows) { "zig.exe" } else { "zig" });
        std::fs::create_dir_all(program.parent().unwrap()).unwrap();
        std::fs::write(&program, b"").unwrap();
        assert_eq!(installed(&tools), Some(Compiler::new(&program, &["cc"])));
        std::fs::remove_dir_all(tools).unwrap();
    }

    #[test]
    fn wget_writes_where_curl_would() {
        let args = |command: Command| command.get_args().map(|arg| arg.to_string_lossy().into_owned()).collect::<Vec<_>>();
        let curl = args(Fetcher::Curl.command("https://x/z", Some(Path::new("/t/z.part")), 1800));
        assert!(curl.windows(2).any(|pair| pair == ["-o", "/t/z.part"]) && curl.last().map(String::as_str) == Some("https://x/z"));
        let wget = args(Fetcher::Wget.command("https://x/z", Some(Path::new("/t/z.part")), 1800));
        assert!(wget.windows(2).any(|pair| pair == ["-O", "/t/z.part"]) && wget.last().map(String::as_str) == Some("https://x/z"));
        assert!(args(Fetcher::Wget.command("https://x/list", None, 30)).windows(2).any(|pair| pair == ["-O", "-"]));
    }

    #[test]
    fn an_archive_of_the_wrong_size_or_hash_is_turned_down() {
        let path = std::env::temp_dir().join(format!("zakuro-zig-archive-{}", std::process::id()));
        std::fs::write(&path, b"abc").unwrap();
        let wrong_size = Archive { name: "x", size: 4, sha256: "" };
        assert!(verify(&path, &wrong_size).unwrap_err().contains("3 bytes arrived"));
        let wrong_hash = Archive { name: "x", size: 3, sha256: "00" };
        assert!(verify(&path, &wrong_hash).unwrap_err().contains("SHA-256"));
        let right = Archive { name: "x", size: 3, sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" };
        assert!(verify(&path, &right).is_ok());
        std::fs::remove_file(path).unwrap();
    }
}
