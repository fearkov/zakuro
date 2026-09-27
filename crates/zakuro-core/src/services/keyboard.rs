//! the software keyboard, the library applet a title starts to have a name
//! or some other text typed in. the title waits while it is open, the
//! frontend shows it and answers with the text and the button pressed.

use crate::kernel::object::{KObject, ObjectId};
use crate::services::apt::{self, Parameter};
use crate::System;

pub const APPLET_ID: u32 = 0x401;

// where the fields the keyboard reads and writes sit in its configuration
const BUTTON_COUNT: usize = 0x04;
const VALID_INPUT: usize = 0x08;
const MAX_TEXT_LENGTH: usize = 0x20;
/// three labels of up to 16 UTF-16 units, each with its terminating zero.
const BUTTON_TEXT: usize = 0x26;
const BUTTON_TEXT_UNITS: usize = 17;
const HINT_TEXT: usize = 0x90;
const HINT_TEXT_UNITS: usize = 65;
// eight flags, which buttons submit text and the language come between
const INITIAL_TEXT_OFFSET: usize = 0x120;
const RESULT: usize = 0x138;
const TEXT_OFFSET: usize = 0x144;
const TEXT_LENGTH: usize = 0x148;
/// the configuration is at least this long to hold the fields above.
const CONFIG_SIZE: usize = 0x14C;

/// what a title asks the keyboard for.
#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    pub hint: String,
    /// the text the keyboard starts with.
    pub text: String,
    /// the longest text it takes, in UTF-16 units.
    pub max_length: usize,
    /// the buttons' labels, left to right, the last one confirms.
    pub buttons: Vec<String>,
    /// what the text has to be for the confirming button to take it.
    pub valid: Valid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Valid {
    Anything,
    NotEmpty,
    NotEmptyNotBlank,
    NotBlank,
    /// exactly the longest length.
    FixedLength,
}

impl Valid {
    fn from_raw(value: u32) -> Valid {
        match value {
            1 => Valid::NotEmpty,
            2 => Valid::NotEmptyNotBlank,
            3 => Valid::NotBlank,
            4 => Valid::FixedLength,
            _ => Valid::Anything,
        }
    }
}

impl Request {
    /// whether the confirming button takes text.
    pub fn accepts(&self, text: &str) -> bool {
        let length = text.encode_utf16().count();
        let blank = !text.is_empty() && text.chars().all(char::is_whitespace);
        length <= self.max_length
            && match self.valid {
                Valid::Anything => true,
                Valid::NotEmpty => length > 0,
                Valid::NotEmptyNotBlank => length > 0 && !blank,
                Valid::NotBlank => !blank,
                Valid::FixedLength => length == self.max_length,
            }
    }
}

/// a keyboard a title started and waits on.
pub struct Pending {
    pub request: Request,
    config: Vec<u8>,
    /// the block the text goes back in.
    memory: Option<ObjectId>,
}

/// opens the keyboard with the configuration a title sent and the block
/// it shares for the text.
pub fn start(system: &mut System, mut config: Vec<u8>, memory: Option<ObjectId>) {
    config.resize(config.len().max(CONFIG_SIZE), 0);
    let word = |at: usize| u32::from_le_bytes(config[at..at + 4].try_into().unwrap());
    let buttons = word(BUTTON_COUNT).min(2) as usize + 1;
    let defaults: &[&str] = match buttons {
        1 => &["OK"],
        2 => &["Cancel", "OK"],
        _ => &["Cancel", "I Forgot", "OK"],
    };
    let label = |slot: usize| utf16(&config[BUTTON_TEXT + slot * BUTTON_TEXT_UNITS * 2..], BUTTON_TEXT_UNITS);
    // two buttons can be labeled in the left and right slots, the middle
    // one left empty
    let slots: &[usize] = match buttons {
        2 if label(1).is_empty() => &[0, 2],
        _ => &[0, 1, 2],
    };
    let labels = (0..buttons)
        .map(|i| {
            let text = label(slots[i]);
            if text.is_empty() { defaults[i].to_owned() } else { text }
        })
        .collect();
    let block = block(system, memory);
    let max_length = match u16::from_le_bytes([config[MAX_TEXT_LENGTH], config[MAX_TEXT_LENGTH + 1]]) as usize {
        // no limit, as much as the block holds
        0 => block.map_or(64, |(_, size)| (size as usize / 2).saturating_sub(1)),
        length => length,
    };
    let initial = word(INITIAL_TEXT_OFFSET);
    let text = match block {
        Some((paddr, size)) if initial < size => {
            let units = ((size - initial) as usize / 2).min(max_length + 1);
            let mut bytes = vec![0; units * 2];
            if let Some(slice) = system.memory.phys.host_slice_mut(paddr + initial, bytes.len() as u32) {
                bytes.copy_from_slice(slice);
            }
            utf16(&bytes, units)
        }
        _ => String::new(),
    };
    let request = Request {
        hint: utf16(&config[HINT_TEXT..], HINT_TEXT_UNITS),
        text,
        max_length,
        buttons: labels,
        valid: Valid::from_raw(word(VALID_INPUT)),
    };
    log::info!("keyboard: a title asks for text, {request:?}");
    system.services.apt.keyboard = Some(Pending { request, config, memory });
}

/// what the keyboard a title waits on asks for.
pub fn request(system: &System) -> Option<&Request> {
    system.services.apt.keyboard.as_ref().map(|pending| &pending.request)
}

/// closes the keyboard with the text typed and the button pressed, zero
/// being the leftmost. the text goes at the start of the shared block.
pub fn answer(system: &mut System, text: &str, button: usize) {
    let Some(pending) = system.services.apt.keyboard.take() else { return };
    let mut config = pending.config;
    let buttons = pending.request.buttons.len();
    let button = button.min(buttons - 1);
    let mut units: Vec<u16> = text.encode_utf16().take(pending.request.max_length).collect();
    if let Some((paddr, size)) = block(system, pending.memory) {
        units.truncate((size as usize / 2).saturating_sub(1));
        let bytes: Vec<u8> = units.iter().chain([&0]).flat_map(|unit| unit.to_le_bytes()).collect();
        system.memory.write_physical(paddr, &bytes);
    }
    // one button clicks as 0, two as 1 and 2, three as 3 to 5
    let result = match buttons {
        1 => 0,
        2 => 1 + button as u32,
        _ => 3 + button as u32,
    };
    config[RESULT..RESULT + 4].copy_from_slice(&result.to_le_bytes());
    config[TEXT_OFFSET..TEXT_OFFSET + 4].copy_from_slice(&0u32.to_le_bytes());
    config[TEXT_LENGTH..TEXT_LENGTH + 2].copy_from_slice(&(units.len() as u16).to_le_bytes());
    log::info!("keyboard: closed with {text:?} on button {button}");
    apt::send_parameter(
        system,
        Parameter {
            sender: APPLET_ID,
            destination: apt::APPLICATION,
            signal: apt::SIGNAL_WAKEUP_BY_EXIT,
            buffer: config,
            object: None,
        },
    );
}

/// where the shared block is and how big.
fn block(system: &System, memory: Option<ObjectId>) -> Option<(u32, u32)> {
    match system.kernel.objects.get(memory?)? {
        KObject::SharedMemory(block) => Some((block.paddr, block.size)),
        _ => None,
    }
}

/// UTF-16 text of at most units units, up to its terminating zero.
fn utf16(bytes: &[u8], units: usize) -> String {
    let units: Vec<u16> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .take(units)
        .map(|pair| u16::from_le_bytes(*pair))
        .take_while(|&unit| unit != 0)
        .collect();
    String::from_utf16_lossy(&units)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_confirming_button_takes_only_valid_text() {
        let request = Request {
            hint: String::new(),
            text: String::new(),
            max_length: 8,
            buttons: vec!["Cancel".to_owned(), "OK".to_owned()],
            valid: Valid::NotEmptyNotBlank,
        };
        assert!(request.accepts("Link"));
        assert!(!request.accepts(""));
        assert!(!request.accepts("   "));
        assert!(!request.accepts("Linkkkkkk"));
    }

    /// the text and the button go back where titles look for them, in a
    /// configuration laid out the way the SDK lays it out.
    #[test]
    fn the_answer_goes_where_titles_read_it() {
        use crate::kernel::object::SharedMemory;
        let mut system = System::new(crate::Config::default());
        let block = system.memory.phys.allocate(crate::memory::MemoryRegion::Base, 0x1000).unwrap();
        let memory = system.kernel.objects.insert(KObject::SharedMemory(SharedMemory {
            name: "keyboard".into(),
            address: 0,
            size: 0x1000,
            paddr: block.addr,
            mapped_at: None,
        }));
        system.memory.write_physical(block.addr, &[0x4C, 0, 0x69, 0, 0, 0]);
        let mut config = vec![0u8; 0x400];
        config[BUTTON_COUNT] = 1;
        config[VALID_INPUT] = 2;
        config[MAX_TEXT_LENGTH] = 8;
        // no result yet, and the shared block's size and version after the
        // initial text's offset
        config[RESULT..RESULT + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        config[0x130..0x134].copy_from_slice(&0x1000u32.to_le_bytes());

        start(&mut system, config, Some(memory));
        let asked = request(&system).unwrap();
        assert_eq!((asked.text.as_str(), asked.max_length), ("Li", 8));
        assert_eq!(asked.buttons, ["Cancel", "OK"]);

        answer(&mut system, "Tatl", 1);
        let parameter = system.services.apt.parameter.clone().unwrap();
        let config = parameter.buffer;
        assert_eq!(parameter.signal, apt::SIGNAL_WAKEUP_BY_EXIT);
        assert_eq!(config[RESULT..RESULT + 4], 2u32.to_le_bytes());
        assert_eq!(config[TEXT_OFFSET..TEXT_OFFSET + 4], [0; 4]);
        assert_eq!(config[TEXT_LENGTH..TEXT_LENGTH + 2], 4u16.to_le_bytes());
        assert_eq!(config[0x130..0x134], 0x1000u32.to_le_bytes());
        let mut text = [0u8; 10];
        text.copy_from_slice(system.memory.phys.host_slice_mut(block.addr, 10).unwrap());
        assert_eq!(utf16(&text, 5), "Tatl");
        assert!(request(&system).is_none());
    }

    #[test]
    fn labels_read_up_to_their_end() {
        let bytes: Vec<u8> = "OK".encode_utf16().chain([0, 0x41]).flat_map(u16::to_le_bytes).collect();
        assert_eq!(utf16(&bytes, 17), "OK");
    }
}
