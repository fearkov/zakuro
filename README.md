# Zakuro

A WIP HLE Nintendo 3DS emulator written in Rust.

<img width="791" height="976" alt="image" src="https://github.com/user-attachments/assets/8900d84d-27c1-44ee-91b3-be8d5fc79b54" />

I started developing this project in October 2025, before
[feargba](https://github.com/fearkov/feargba). I first wrote it in C++, but I
was learning Rust at the time and noticed there wasn't a working 3DS emulator
written in Rust, so I switched.

This is a personal, experimental project. You can use it to play games, but
that was never the main goal.

Some games boot, but they run well below full speed. There's no audio output,
fragment lighting or JIT yet.

To run it you need a decrypted ROM:

    cargo run --release -p zakuro -- path/to/rom.3ds

No copyrighted data is included. I do not condone piracy, and I will not help
you with that.

Controls: arrow keys for the d-pad, IJKL for the circle pad, X Z S A for
A B X Y, Q and W for L and R, Enter for Start, Backspace for Select, and the
mouse for the touch screen. F1 pauses and Esc quits.

Contributions are welcome. Using AI is fine, but the code must always be
reviewed by a human.

MIT license.
