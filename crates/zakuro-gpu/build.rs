//! builds the second panel's lease code (drm_lease/, C++) on Linux w/ the Vulkan presenter,
//! when wayland-scanner and the wayland-client and libdrm headers are there, w/o them the second display isn't offered

use std::path::{Path, PathBuf};
use std::process::Command;

const SOURCES: [&str; 4] = ["drm_lease_client.cpp", "drm_lease_dl.cpp", "drm_lease_touch.cpp", "drm_lease_window.cpp"];

fn main() {
    println!("cargo:rustc-check-cfg=cfg(drm_lease)");
    let linux = std::env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "linux");
    if !linux || std::env::var_os("CARGO_FEATURE_VULKAN").is_none() {
        return;
    }
    let folder = Path::new("drm_lease");
    println!("cargo:rerun-if-changed=drm_lease");
    let generated = PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
    let include_dirs = match headers_and_protocol(folder, &generated) {
        Ok(dirs) => dirs,
        Err(missing) => {
            println!("cargo:warning=the second display is left out, {missing}");
            return;
        }
    };
    let mut build = cc::Build::new();
    build.cpp(true).std("c++17").warnings(false).includes(&include_dirs);
    for source in SOURCES {
        build.file(folder.join(source));
    }
    build.compile("drm_lease");
    // the protocol's interfaces, C that wayland-scanner writes
    cc::Build::new()
        .file(generated.join("drm-lease-v1-protocol.c"))
        .includes(&include_dirs)
        .warnings(false)
        .compile("drm_lease_protocol");
    // dlopen, part of libc from glibc 2.34 on, a library of its own b4
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-cfg=drm_lease");
}

/// the include folders wayland-client's and libdrm's pkg-config files name, plus the folder the protocol's
/// client header and interfaces get generated into
fn headers_and_protocol(folder: &Path, generated: &Path) -> Result<Vec<PathBuf>, String> {
    let mut dirs = Vec::new();
    for package in ["wayland-client", "libdrm"] {
        let output = Command::new("pkg-config")
            .args(["--cflags-only-I", package])
            .output()
            .map_err(|error| format!("pkg-config could not run: {error}"))?;
        if !output.status.success() {
            return Err(format!("pkg-config finds no {package}"));
        }
        let flags = String::from_utf8_lossy(&output.stdout).into_owned();
        dirs.extend(flags.split_whitespace().filter_map(|flag| flag.strip_prefix("-I")).map(PathBuf::from));
    }
    for (kind, file) in [("client-header", "drm-lease-v1-client-protocol.h"), ("private-code", "drm-lease-v1-protocol.c")] {
        let status = Command::new("wayland-scanner")
            .arg(kind)
            .arg(folder.join("drm-lease-v1.xml"))
            .arg(generated.join(file))
            .status()
            .map_err(|error| format!("wayland-scanner could not run: {error}"))?;
        if !status.success() {
            return Err(format!("wayland-scanner {kind} failed on drm-lease-v1.xml"));
        }
    }
    dirs.push(generated.to_path_buf());
    Ok(dirs)
}
