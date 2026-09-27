# Zakuro

A WIP HLE Nintendo 3DS emulator written in Rust.

<img width="797" height="983" alt="image" src="https://github.com/user-attachments/assets/c18deb0d-3949-4493-83df-0142928d8c2d" />


I started developing this project in October 2025, before
[feargba](https://github.com/fearkov/feargba). 

I first wrote it in C++, but I was learning Rust at the time and noticed there wasn't a working 3DS emulator written in Rust, so I switched.

This is a personal experimental project. You can use it to play games, but that was never the main goal. Some games boot. On the interpreter alone they run below full speed, but Pokémon Alpha Sapphire runs at full speed with its code recompiled by [3dsrecomp](https://github.com/fearkov/3dsrecomp). There's no JIT yet.

To install it you need Rust 1.95 or newer. On Linux, building also needs pkg-config and the ALSA development files (libasound2-dev on Debian and Ubuntu, alsa-lib on Arch):

    cargo install --git https://github.com/fearkov/zakuro --locked zakuro

Cargo puts it in ~/.cargo/bin, which has to be on your PATH. Then `zakuro` in a terminal opens it. To play you need a decrypted ROM:

    zakuro path/to/rom.3ds

Without a ROM it opens a library with the games in a folder you pick, where you can also recompile them. Esc brings up a menu over the game, and the settings (controls, sound, graphics, a background for the library) are in there too.

It can also run code recompiled ahead of time with [3dsrecomp](https://github.com/fearkov/3dsrecomp), which installs the same way and also needs a C compiler. Build it once per game and Zakuro picks it up on its own. Anything the library doesn't cover still goes through the interpreter:

    cargo install --git https://github.com/fearkov/3dsrecomp --locked recomp3ds
    3dsrecomp build path/to/rom.3ds
    zakuro path/to/rom.3ds

--recompiled points it at another library, and --interpreter runs everything in the interpreter.

No copyrighted data is included. I do not condone piracy, and I will not help you with that. So, don't ask me about that.

Controls:

| 3DS | Keyboard |
|---|---|
| A / B / X / Y | X / Z / S / A |
| L / R | Q / W |
| Start / Select | Enter / Backspace |
| D-pad | Arrow keys |
| Circle pad | I / J / K / L |
| Touch screen | Mouse (click) |
| Menu / Pause / Fullscreen | Esc / F1 / F11 |

Contributions are welcome. Using AI is fine sometimes, but the code must always be reviewed by a human. Code that is entirely vibecoded will be discarded.

MIT license.
