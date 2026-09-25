# Zakuro

A WIP HLE Nintendo 3DS emulator written in Rust.

<img width="797" height="983" alt="image" src="https://github.com/user-attachments/assets/c18deb0d-3949-4493-83df-0142928d8c2d" />


I started developing this project in October 2025, before
[feargba](https://github.com/fearkov/feargba). 

I first wrote it in C++, but I was learning Rust at the time and noticed there wasn't a working 3DS emulator written in Rust, so I switched.

This is a personal experimental project. You can use it to play games, but that was never the main goal. Some games boot, but they run well below full speed. There's no audio output, fragment lighting or JIT yet.

To run it you need a decrypted ROM:

    cargo run --release -p zakuro -- path/to/rom.3ds

No copyrighted data is included. I do not condone piracy, and I will not help you with that. So, don't ask me about that.

Controls:

| 3DS | Keyboard |
|---|---|
| A / B / X / Y | X / Z / S / A |
| L / R | Q / W |
| Start / Select | Enter / Backspace |
| D-pad | Arrow keys |
| Circle pad | I / J / K / L |
| Touch screen | Mouse |
| Pause / Quit | F1 / Esc |

Contributions are welcome. Using AI is fine sometimes, but the code must always be reviewed by a human. Code that is entirely vibecoded will be discarded.

MIT license.
