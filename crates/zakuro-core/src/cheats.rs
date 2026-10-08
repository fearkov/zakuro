//! cheats, Gateway and Action Replay codes, in the file Citra and Azahar keep
//! them in, cheats/<title id>.txt in the data folder, so that one made for
//! them works here as it is. the cheats that are on run once a frame.

use std::path::{Path, PathBuf};

use zakuro_common::memory_map::PAGE_SIZE;
use zakuro_cpu::Bus;

use crate::System;

/// one cheat, its name, whether it is on, and its code.
#[derive(Debug, Clone, PartialEq)]
pub struct Cheat {
    pub name: String,
    pub enabled: bool,
    /// the lines after a * in the file, besides the one that turns it on.
    pub notes: Vec<String>,
    /// its lines as written, "XXXXXXXX YYYYYYYY" each.
    lines: Vec<String>,
    code: Vec<Option<Line>>,
}

/// a line of code taken apart, its type being its first digit, or its first
/// two when that is D.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Line {
    kind: u8,
    first: u32,
    address: u32,
    value: u32,
}

/// the line Citra writes in a cheat that is on.
const ENABLED: &str = "*citra_enabled";

/// lines one cheat runs in a frame at most, so that a loop that never ends
/// does not stop the game.
const MOST_STEPS: usize = 100_000;

impl Line {
    fn parse(text: &str) -> Option<Line> {
        if text.len() != 17 || !text.is_ascii() {
            return None;
        }
        let first = u32::from_str_radix(&text[..8], 16).ok()?;
        let value = u32::from_str_radix(&text[9..], 16).ok()?;
        let kind = match (first >> 28) as u8 {
            0xD => (first >> 24) as u8,
            kind => kind,
        };
        Some(Line { kind, first, address: first & 0x0FFF_FFFF, value })
    }
}

/// whether a line is a code, "XXXXXXXX YYYYYYYY".
pub fn is_code(line: &str) -> bool {
    Line::parse(line).is_some()
}

impl Cheat {
    pub fn new(name: &str, lines: Vec<String>, notes: Vec<String>) -> Cheat {
        let code = lines.iter().map(|line| Line::parse(line)).collect();
        Cheat { name: name.to_owned(), enabled: false, notes, lines, code }
    }

    pub fn lines(&self) -> &[String] {
        &self.lines
    }
}

/// the file of a title's cheats.
pub fn path(data_dir: &Path, program_id: u64) -> PathBuf {
    data_dir.join("cheats").join(format!("{program_id:016X}.txt"))
}

/// the cheats in a file of Citra's, none when there is no file.
pub fn load(path: &Path) -> Vec<Cheat> {
    std::fs::read(path).map(|bytes| parse(&String::from_utf8_lossy(&bytes))).unwrap_or_default()
}

/// cheats as Citra writes them, a name in brackets, the notes and whether it
/// is on after a *, then the code.
pub fn parse(text: &str) -> Vec<Cheat> {
    let mut cheats = Vec::new();
    let (mut name, mut notes, mut lines, mut enabled) = (String::new(), Vec::new(), Vec::new(), false);
    let mut finish = |name: &str, notes: &mut Vec<String>, lines: &mut Vec<String>, enabled: &mut bool| {
        if !lines.is_empty() {
            let mut cheat = Cheat::new(name, std::mem::take(lines), std::mem::take(notes));
            cheat.enabled = *enabled;
            cheats.push(cheat);
        }
        notes.clear();
        *enabled = false;
    };
    for line in text.lines() {
        let line = line.replace('\0', "");
        let line = line.trim();
        if line.len() >= 2 && line.starts_with('[') {
            finish(&name, &mut notes, &mut lines, &mut enabled);
            name = line[1..line.len() - 1].to_owned();
        } else if let Some(note) = line.strip_prefix('*') {
            if line == ENABLED {
                enabled = true;
            } else {
                notes.push(note.to_owned());
            }
        } else if !line.is_empty() {
            lines.push(line.to_owned());
        }
    }
    finish(&name, &mut notes, &mut lines, &mut enabled);
    cheats
}

/// cheats as Citra writes them, for its file.
pub fn to_text(cheats: &[Cheat]) -> String {
    let mut text = String::new();
    for cheat in cheats {
        text += &format!("[{}]\n", cheat.name);
        if cheat.enabled {
            text += ENABLED;
            text += "\n";
        }
        for note in &cheat.notes {
            text += &format!("*{note}\n");
        }
        for line in &cheat.lines {
            text += line;
            text += "\n";
        }
        text += "\n";
    }
    text
}

/// writes the cheats to their file, and the folder it is in.
pub fn save(path: &Path, cheats: &[Cheat]) -> std::io::Result<()> {
    if let Some(folder) = path.parent() {
        std::fs::create_dir_all(folder)?;
    }
    std::fs::write(path, to_text(cheats))
}

/// what a cheat's code keeps between its lines.
#[derive(Default)]
struct State {
    reg: u32,
    offset: u32,
    /// the conditions failed and not yet closed, the lines meanwhile skipped.
    if_flag: u32,
    loop_count: u32,
    loop_back_line: usize,
    loop_flag: bool,
}

/// where code lies in memory, the executable's text and the modules'.
struct CodeRanges {
    text: (u32, u32),
    modules: Vec<(u32, u32)>,
    /// what the cheats changed of them, the text and the modules by index.
    text_changed: bool,
    modules_changed: Vec<usize>,
}

impl CodeRanges {
    fn of(system: &System) -> CodeRanges {
        let text = system.title.as_ref().map_or((0, 0), |title| {
            let text = &title.exheader.text;
            (text.address, text.address + text.num_pages * PAGE_SIZE)
        });
        let modules = system.cro.modules.iter().map(|module| (module.base, module.base + module.size)).collect();
        CodeRanges { text, modules, text_changed: false, modules_changed: Vec::new() }
    }

    fn wrote(&mut self, addr: u32) {
        if (self.text.0..self.text.1).contains(&addr) {
            self.text_changed = true;
        }
        for (index, &(start, end)) in self.modules.iter().enumerate() {
            if (start..end).contains(&addr) && !self.modules_changed.contains(&index) {
                self.modules_changed.push(index);
            }
        }
    }
}

/// runs the cheats that are on, once a frame. a cheat that changes code has
/// what was recompiled of it checked again, so that the interpreter runs the
/// functions it changed.
pub fn run(system: &mut System) {
    if !system.cheats.iter().any(|cheat| cheat.enabled) {
        return;
    }
    let mut ranges = CodeRanges::of(system);
    let pad = system.services.hid.previous.bits();
    for index in 0..system.cheats.len() {
        if system.cheats[index].enabled {
            let code = std::mem::take(&mut system.cheats[index].code);
            execute(system, &code, pad, &mut ranges);
            system.cheats[index].code = code;
        }
    }
    if let Some(library) = system.recompiled.as_mut() {
        if ranges.text_changed && !library.checks_itself() {
            log::warn!("cheats changed the game's code, which its recompiled code, made by an older 3dsrecomp, can't tell, recompiling the game again makes them work");
        } else if ranges.text_changed {
            let stale = library.check(&mut system.memory);
            log::info!("cheats changed the game's code, {stale} functions run in the interpreter");
        }
        for index in ranges.modules_changed {
            let module = &system.cro.modules[index];
            library.place(&module.name, module.base, &mut system.memory);
        }
    }
}

fn mapped(system: &System, addr: u32, size: u32) -> bool {
    system.memory.mapping_at(addr).is_some() && system.memory.mapping_at(addr.wrapping_add(size - 1)).is_some()
}

fn read(system: &mut System, addr: u32, size: u32) -> u32 {
    if !mapped(system, addr, size) {
        return 0;
    }
    match size {
        1 => system.memory.read8(addr) as u32,
        2 => system.memory.read16(addr) as u32,
        _ => system.memory.read32(addr),
    }
}

/// writes value at addr, when memory there is mapped and holds another.
/// code the game can't write gets written all the same, as a Gateway does.
fn write(system: &mut System, ranges: &mut CodeRanges, addr: u32, size: u32, value: u32) {
    if !mapped(system, addr, size) || read(system, addr, size) == value & (u32::MAX >> (32 - size * 8)) {
        return;
    }
    let writable = system.memory.mapping_at(addr).is_some_and(|mapping| mapping.permission.contains(crate::memory::Permission::WRITE));
    match size {
        _ if !writable => {
            // the words the bytes are in, each written whole
            for (i, byte) in value.to_le_bytes().into_iter().take(size as usize).enumerate() {
                let at = addr.wrapping_add(i as u32);
                let (word_at, shift) = (at & !3, (at & 3) * 8);
                let word = system.memory.read32(word_at) & !(0xFF << shift) | (byte as u32) << shift;
                system.memory.write32_privileged(word_at, word);
            }
        }
        1 => system.memory.write8(addr, value as u8),
        2 => system.memory.write16(addr, value as u16),
        _ => system.memory.write32(addr, value),
    }
    ranges.wrote(addr);
}

/// runs a cheat's code, the way Citra and Azahar run it.
fn execute(system: &mut System, code: &[Option<Line>], pad: u32, ranges: &mut CodeRanges) {
    let null = Line { kind: 0xFF, first: 0, address: 0, value: 0 };
    let line_at = |n: usize| code.get(n).copied().flatten().unwrap_or(null);
    let mut s = State::default();
    let (mut n, mut steps) = (0usize, 0usize);
    while n < code.len() && steps < MOST_STEPS {
        steps += 1;
        let line = line_at(n);
        let addr = line.address.wrapping_add(s.offset);
        if s.if_flag > 0 {
            // skipped, though conditions inside still nest and patches
            // still skip their data
            match line.kind {
                0x3..=0xA | 0xDD => s.if_flag += 1,
                0xE => n = patch(system, ranges, &line, &s, code, n),
                0xD0 => s.if_flag -= 1,
                0xD2 if s.loop_flag => {
                    n = s.loop_back_line;
                    continue;
                }
                0xD2 => s = State::default(),
                _ => {}
            }
            n += 1;
            continue;
        }
        let compare32 = |system: &mut System, s: &mut State, holds: fn(u32, u32) -> bool| {
            if !holds(line.value, read(system, addr, 4)) {
                s.if_flag += 1;
            }
        };
        let compare16 = |system: &mut System, s: &mut State, holds: fn(u16, u16) -> bool| {
            let masked = (!line.value >> 16) as u16 & read(system, addr, 2) as u16;
            if !holds(line.value as u16, masked) {
                s.if_flag += 1;
            }
        };
        match line.kind {
            0x0 => write(system, ranges, addr, 4, line.value),
            0x1 => write(system, ranges, addr, 2, line.value),
            0x2 => write(system, ranges, addr, 1, line.value),
            0x3 => compare32(system, &mut s, |value, memory| value > memory),
            0x4 => compare32(system, &mut s, |value, memory| value < memory),
            0x5 => compare32(system, &mut s, |value, memory| value == memory),
            0x6 => compare32(system, &mut s, |value, memory| value != memory),
            0x7 => compare16(system, &mut s, |value, memory| value > memory),
            0x8 => compare16(system, &mut s, |value, memory| value < memory),
            0x9 => compare16(system, &mut s, |value, memory| value == memory),
            0xA => compare16(system, &mut s, |value, memory| value != memory),
            0xB => s.offset = read(system, addr, 4),
            0xC => {
                s.loop_flag = s.loop_count < line.value;
                s.loop_count += 1;
                s.loop_back_line = n;
            }
            0xD0 => s.if_flag = s.if_flag.saturating_sub(1),
            0xD1 | 0xD2 if s.loop_flag => {
                n = s.loop_back_line;
                continue;
            }
            0xD1 => s.loop_count = 0,
            0xD2 => s = State::default(),
            0xD3 => s.offset = line.value,
            0xD4 => s.reg = s.reg.wrapping_add(line.value),
            0xD5 => s.reg = line.value,
            0xD6..=0xD8 => {
                let size = [4, 2, 1][(line.kind - 0xD6) as usize];
                write(system, ranges, line.value.wrapping_add(s.offset), size, s.reg);
                s.offset = s.offset.wrapping_add(size);
            }
            0xD9..=0xDB => {
                let size = [4, 2, 1][(line.kind - 0xD9) as usize];
                s.reg = read(system, line.value.wrapping_add(s.offset), size);
            }
            0xDC => s.offset = s.offset.wrapping_add(line.value),
            // the buttons held
            0xDD => {
                if pad & line.value != line.value {
                    s.if_flag += 1;
                }
            }
            0xE => n = patch(system, ranges, &line, &s, code, n),
            _ => {}
        }
        n += 1;
    }
}

/// an E code, value bytes to write at its address, taken from the lines
/// after it, both of their words. the last line of its data, where the code
/// goes on after.
fn patch(system: &mut System, ranges: &mut CodeRanges, line: &Line, s: &State, code: &[Option<Line>], n: usize) -> usize {
    let word = |n: usize, first: bool| code.get(n).copied().flatten().map_or(0, |line| if first { line.first } else { line.value });
    if s.if_flag > 0 {
        return n + line.value.div_ceil(8) as usize;
    }
    let (mut left, mut addr) = (line.value, line.address.wrapping_add(s.offset));
    let (mut at, mut first, mut shift) = (n, true, 0);
    if left > 0 {
        at += 1;
    }
    while left >= 4 {
        let value = word(at, first);
        if !first && left > 4 {
            at += 1;
        }
        first = !first;
        write(system, ranges, addr, 4, value);
        addr = addr.wrapping_add(4);
        left -= 4;
    }
    while left > 0 {
        write(system, ranges, addr, 1, word(at, first) >> shift);
        addr = addr.wrapping_add(1);
        left -= 1;
        shift += 8;
    }
    at
}

#[cfg(test)]
mod tests {
    use super::*;

    const FILE: &str = "[Infinite health]\n*citra_enabled\n*by someone\n00123456 00000063\n\n[Walk through walls]\nD3000000 00100000\n20000004 000000FF\nD2000000 00000000\n";

    #[test]
    fn a_file_of_citras_reads_and_writes_back_the_same() {
        let cheats = parse(FILE);
        assert_eq!(cheats.len(), 2);
        assert_eq!((cheats[0].name.as_str(), cheats[0].enabled), ("Infinite health", true));
        assert_eq!(cheats[0].notes, ["by someone"]);
        assert_eq!(cheats[1].lines().len(), 3);
        assert!(!cheats[1].enabled);
        assert_eq!(parse(&to_text(&cheats)), cheats);
    }

    /// each kind of line does to memory what it does in Citra.
    #[test]
    fn codes_write_compare_loop_and_patch() {
        use zakuro_fs::testing::{ncch, write as write_file};
        let rom = write_file("cheats", "game.cxi", &ncch(0x0004_0000_0FF3_DE02, Some(&[0; 0x80]), &[]));
        let mut system = crate::loader::load(&rom, crate::Config::default()).unwrap();
        let text = "\
[write]\n00100010 12345678\n\
[true condition]\n50100010 12345678\n20100020 000000AA\nD0000000 00000000\n\
[false condition]\n50100010 00000000\n20100021 000000BB\nD0000000 00000000\n\
[loop]\nD3000000 00100030\nD5000000 00000007\nC0000000 00000003\nD8000000 00000000\nD1000000 00000000\nD2000000 00000000\n\
[patch]\nE0100040 00000006\n11223344 55667788\n\
[button]\nDD000000 00000001\n20100050 000000CC\nD0000000 00000000\n";
        system.cheats = parse(text);
        for cheat in &mut system.cheats {
            cheat.enabled = true;
        }
        run(&mut system);
        let mut memory = [0; 0x60];
        system.memory.read_bytes(0x0010_0000, &mut memory);
        assert_eq!(&memory[0x10..0x14], &0x1234_5678u32.to_le_bytes());
        assert_eq!((memory[0x20], memory[0x21]), (0xAA, 0));
        assert_eq!(&memory[0x30..0x35], &[7, 7, 7, 7, 0]);
        assert_eq!(&memory[0x40..0x47], &[0x44, 0x33, 0x22, 0x11, 0x88, 0x77, 0]);
        assert_eq!(memory[0x50], 0, "A is not held");
        system.services.hid.previous = crate::services::hid::PadState::from_bits_truncate(1);
        run(&mut system);
        system.memory.read_bytes(0x0010_0050, &mut memory[..1]);
        assert_eq!(memory[0], 0xCC, "A is held");
        std::fs::remove_file(rom).unwrap();
    }

    #[test]
    fn lines_take_their_type_from_the_first_digits() {
        assert_eq!(Line::parse("D3000000 00100000").map(|line| line.kind), Some(0xD3));
        let write = Line::parse("1234ABCD 0000FFFF").unwrap();
        assert_eq!((write.kind, write.address, write.value), (0x1, 0x0234_ABCD, 0xFFFF));
        assert!(Line::parse("1234ABCD 0000FFF").is_none());
        assert!(Line::parse("ZZZZZZZZ 00000000").is_none());
    }
}
