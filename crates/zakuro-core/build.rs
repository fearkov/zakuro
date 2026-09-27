//! the examples run the game's code on the main thread, whose stack is
//! 8 MiB on Linux and only 1 MiB on Windows unless the program asks for
//! more.

fn main() {
    let windows = std::env::var("CARGO_CFG_TARGET_FAMILY").is_ok_and(|family| family == "windows");
    if windows {
        let stack = 8 << 20;
        match std::env::var("CARGO_CFG_TARGET_ENV").as_deref() {
            Ok("msvc") => println!("cargo:rustc-link-arg-examples=/STACK:{stack}"),
            _ => println!("cargo:rustc-link-arg-examples=-Wl,--stack,{stack}"),
        }
    }
}
