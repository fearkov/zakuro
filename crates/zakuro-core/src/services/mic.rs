//! mic:u, the microphone. what it hears comes from the frontend, see
//! crate::microphone, and goes into the block the title shares with the
//! service, at the rate and in the encoding the title asks for, round and
//! round or once through. the block's last word says where in it the next
//! sample goes, and the buffer full event is signaled each time the samples
//! reach the end.

use crate::kernel::ipc::{CommandBuffer, Descriptor, Header};
use crate::kernel::object::{KObject, ObjectId};
use crate::kernel::sync::ResetType;
use crate::kernel::thread::CPU_CLOCK_HZ;
use crate::System;
use zakuro_common::result::errors;

/// the encodings, unsigned and signed, of 8 and 16 bits.
const PCM8: u8 = 0;
const PCM16: u8 = 1;
const PCM8_SIGNED: u8 = 2;
const PCM16_SIGNED: u8 = 3;

/// cycles between samples at a rate, the fastest one every 8192 cycles as
/// the DSP's, 32728 a second, the others a half, a third and a quarter of
/// it, which are the 32730, 16360, 10910 and 8180 Hz they go by.
fn sample_ticks(rate: u8) -> u64 {
    8192 * (rate.min(3) as u64 + 1)
}

/// samples a second at a rate.
fn sample_rate(rate: u8) -> f64 {
    CPU_CLOCK_HZ as f64 / sample_ticks(rate) as f64
}

/// an encoding as it is called.
fn encoding_name(encoding: u8) -> &'static str {
    match encoding {
        PCM8 => "8 bits unsigned",
        PCM16 => "16 bits unsigned",
        PCM8_SIGNED => "8 bits signed",
        PCM16_SIGNED => "16 bits signed",
        _ => "an unknown encoding",
    }
}

/// bytes a sample takes in an encoding.
fn sample_bytes(encoding: u8) -> u32 {
    if encoding & 1 == 0 { 1 } else { 2 }
}

#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct MicState {
    /// the block the title shares and the size it gave.
    block: Option<ObjectId>,
    block_size: u32,
    sampling: Option<Sampling>,
    gain: u8,
    power: bool,
    clamp: bool,
    allow_shell_closed: bool,
    client_version: u32,
    /// the buffer full event, made the first time it is asked for.
    event: Option<ObjectId>,
}

impl MicState {
    /// whether a title samples.
    pub fn is_sampling(&self) -> bool {
        self.sampling.is_some()
    }
}

/// what the title asked for when it started sampling, and how far it got.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
struct Sampling {
    encoding: u8,
    rate: u8,
    /// where in the block the samples go and how many bytes they take.
    offset: u32,
    size: u32,
    /// round and round, or once through.
    looping: bool,
    /// where the next sample goes, from offset.
    position: u32,
    /// the tick the next sample is due at.
    next: u64,
}

pub fn handle(system: &mut System, buffer: &CommandBuffer, header: Header) -> bool {
    let command = header.command_id();
    let word = |system: &mut System, index: u32| buffer.get(&mut system.memory, index);
    match command {
        // MapSharedMem(size, block)
        0x0001 => {
            let size = word(system, 1);
            let handle = word(system, 3);
            stop(system);
            unmap(system);
            match system.kernel.resolve(handle) {
                Some(object) => {
                    // the service holds on to it, a title closing its handle
                    // must not take it away
                    system.kernel.objects.add_ref(object);
                    system.services.mic.block = Some(object);
                    system.services.mic.block_size = size;
                    buffer.reply(&mut system.memory, command, &[]);
                }
                None => buffer.reply_error(&mut system.memory, command, errors::INVALID_HANDLE.0),
            }
            true
        }
        // UnmapSharedMem
        0x0002 => {
            stop(system);
            unmap(system);
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // StartSampling(encoding, rate, offset, size, loop)
        0x0003 => {
            let encoding = word(system, 1) as u8 & 3;
            let rate = word(system, 2) as u8 & 3;
            let offset = word(system, 3);
            let size = word(system, 4);
            let looping = word(system, 5) & 0xFF != 0;
            let next = system.cpu.cycles + sample_ticks(rate);
            log::debug!(
                "mic: sampling {} at {:.0} Hz, {} bytes from 0x{offset:X}, {}",
                encoding_name(encoding),
                sample_rate(rate),
                size,
                if looping { "round and round" } else { "once through" }
            );
            system.services.mic.sampling = Some(Sampling { encoding, rate, offset, size, looping, position: 0, next });
            let event = event(system);
            system.kernel.clear_event(event);
            write_position(system);
            sampling(system);
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // AdjustSampling(rate)
        0x0004 => {
            let rate = word(system, 1) as u8 & 3;
            let now = system.cpu.cycles;
            if let Some(sampling) = &mut system.services.mic.sampling {
                sampling.rate = rate;
                sampling.next = now + sample_ticks(rate);
            }
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // StopSampling
        0x0005 => {
            stop(system);
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // IsSampling
        0x0006 => {
            let on = system.services.mic.sampling.is_some();
            buffer.reply(&mut system.memory, command, &[on as u32]);
            true
        }
        // GetBufferFullEvent
        0x0007 => {
            let object = event(system);
            let handle = system.kernel.handles.create(&mut system.kernel.objects, object, "mic event");
            buffer.set(&mut system.memory, 0, Header::new(command, 1, 2).0);
            buffer.set(&mut system.memory, 1, 0);
            buffer.set(&mut system.memory, 2, Descriptor::handles(1));
            buffer.set(&mut system.memory, 3, handle);
            true
        }
        // SetGain(gain), how much the microphone's sound is raised, which a
        // host's microphone has done already
        0x0008 => {
            system.services.mic.gain = word(system, 1) as u8;
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // GetGain
        0x0009 => {
            let gain = system.services.mic.gain;
            buffer.reply(&mut system.memory, command, &[gain as u32]);
            true
        }
        // SetPower(on)
        0x000A => {
            system.services.mic.power = word(system, 1) & 0xFF != 0;
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // GetPower
        0x000B => {
            let power = system.services.mic.power;
            buffer.reply(&mut system.memory, command, &[power as u32]);
            true
        }
        // SetIirFilterMic(size, filter), a filter on what the microphone
        // hears, which the sound from a host's has no need of. the filter's
        // buffer goes back to the title
        0x000C => {
            reply_with_buffers(system, buffer, header);
            true
        }
        // SetClamp(on)
        0x000D => {
            system.services.mic.clamp = word(system, 1) & 0xFF != 0;
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // GetClamp
        0x000E => {
            let clamp = system.services.mic.clamp;
            buffer.reply(&mut system.memory, command, &[clamp as u32]);
            true
        }
        // SetAllowShellClosed(allow)
        0x000F => {
            system.services.mic.allow_shell_closed = word(system, 1) & 0xFF != 0;
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // SetClientVersion(version)
        0x0010 => {
            system.services.mic.client_version = word(system, 1);
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        _ => false,
    }
}

/// a successful reply, with the request's buffer descriptors handed back as
/// they came.
fn reply_with_buffers(system: &mut System, buffer: &CommandBuffer, header: Header) {
    let first = 1 + header.normal_params();
    let translate: Vec<u32> = (0..header.translate_params()).map(|i| buffer.get(&mut system.memory, first + i)).collect();
    buffer.set(&mut system.memory, 0, Header::new(header.command_id(), 1, translate.len() as u32).0);
    buffer.set(&mut system.memory, 1, 0);
    for (i, &word) in translate.iter().enumerate() {
        buffer.set(&mut system.memory, 2 + i as u32, word);
    }
}

/// stops sampling, telling the frontend's microphone.
fn stop(system: &mut System) {
    if system.services.mic.sampling.take().is_some() {
        sampling(system);
    }
}

/// lets go of the block.
fn unmap(system: &mut System) {
    if let Some(object) = system.services.mic.block.take() {
        system.kernel.objects.release(object);
    }
    system.services.mic.block_size = 0;
}

/// tells the frontend's microphone whether a title samples.
pub(crate) fn sampling(system: &mut System) {
    let on = system.services.mic.sampling.is_some();
    if let Some(microphone) = &mut system.microphone {
        microphone.sampling(on);
    }
}

/// the buffer full event, made the first time it is asked for.
fn event(system: &mut System) -> ObjectId {
    if let Some(event) = system.services.mic.event {
        return event;
    }
    let object = system.kernel.objects.insert(KObject::Event(crate::kernel::sync::Event::new(ResetType::OneShot, "MIC:BufferFull")));
    // the service keeps it, a title closing its handle must not take it away
    system.kernel.objects.add_ref(object);
    system.services.mic.event = Some(object);
    object
}

/// where the block's bytes are and how many the title said it has, when
/// there is one.
fn block(system: &System) -> Option<(u32, u32)> {
    let object = system.services.mic.block?;
    match system.kernel.objects.get(object) {
        Some(KObject::SharedMemory(block)) => Some((block.paddr, block.size.min(system.services.mic.block_size))),
        _ => None,
    }
}

/// writes into the block, when there is one and the bytes fit.
fn write(system: &mut System, offset: u32, bytes: &[u8]) {
    let Some((paddr, size)) = block(system) else { return };
    if offset as u64 + bytes.len() as u64 <= size as u64 {
        system.memory.write_physical(paddr + offset, bytes);
    }
}

/// writes where the next sample goes into the block's last word, from the
/// block's start.
fn write_position(system: &mut System) {
    let Some(sampling) = system.services.mic.sampling else { return };
    let Some((_, size)) = block(system) else { return };
    if size >= 4 {
        write(system, size - 4, &(sampling.offset + sampling.position).to_le_bytes());
    }
}

/// what was heard in an encoding.
fn encode(heard: &[f32], encoding: u8) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(heard.len() * 2);
    for &sample in heard {
        let sample = sample.clamp(-1.0, 1.0);
        match encoding {
            PCM8 => bytes.push((sample * 127.0 + 128.0).round() as u8),
            PCM8_SIGNED => bytes.push((sample * 127.0).round() as i8 as u8),
            PCM16 => bytes.extend_from_slice(&((sample * 32767.0 + 32768.0).round() as u16).to_le_bytes()),
            _ => bytes.extend_from_slice(&((sample * 32767.0).round() as i16).to_le_bytes()),
        }
    }
    bytes
}

/// writes the samples due into the block, signaling the event at its end,
/// where sampling starts over or stops.
pub fn tick(system: &mut System) {
    let Some(mut sampling) = system.services.mic.sampling else { return };
    let now = system.cpu.cycles;
    if sampling.next > now {
        return;
    }
    let ticks = sample_ticks(sampling.rate);
    let width = sample_bytes(sampling.encoding);
    // whole samples only, an odd size leaves its last byte alone
    let room = sampling.size / width;
    if room == 0 {
        return;
    }
    let due = (now - sampling.next) / ticks + 1;
    sampling.next += due * ticks;
    // more than the buffer holds, after a long stall, and only the latest
    // fill it
    let count = due.min(room as u64) as usize;
    let mut heard = vec![0.0f32; count];
    if let Some(microphone) = &mut system.microphone {
        microphone.read(sample_rate(sampling.rate), &mut heard);
    }
    let bytes = encode(&heard, sampling.encoding);

    let (mut written, mut full, mut done) = (0usize, false, false);
    while written < bytes.len() {
        let space = (room * width - sampling.position) as usize;
        let length = space.min(bytes.len() - written);
        write(system, sampling.offset + sampling.position, &bytes[written..written + length]);
        written += length;
        sampling.position += length as u32;
        if sampling.position >= room * width {
            full = true;
            if sampling.looping {
                sampling.position = 0;
            } else {
                done = true;
                break;
            }
        }
    }
    system.services.mic.sampling = Some(sampling);
    write_position(system);
    if done {
        log::debug!("mic: the buffer is full, sampling stops");
        stop(system);
    }
    if full {
        let event = event(system);
        system.kernel.signal_event(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::object::SharedMemory;
    use crate::microphone::{Microphone, Recording};
    use crate::services::tests::system_with_thread;
    use crate::services::{handle_request, Target};
    use std::sync::{Arc, Mutex};

    /// a microphone that counts its samples up, 1/1000 more each, and tells
    /// whether it was asked to listen.
    struct Counting {
        next: f32,
        on: Arc<Mutex<bool>>,
    }

    impl Microphone for Counting {
        fn sampling(&mut self, on: bool) {
            *self.on.lock().unwrap() = on;
        }

        fn read(&mut self, _rate: f64, out: &mut [f32]) {
            for sample in out {
                self.next += 0.001;
                *sample = self.next;
            }
        }
    }

    /// a system with a block of size bytes the title shares with mic:u, and
    /// a way to send it requests.
    fn with_block(size: u32) -> (System, impl Fn(&mut System, &[u32]) -> u32, u32) {
        let (mut system, buffer) = system_with_thread();
        let block = system.memory.phys.allocate(crate::memory::MemoryRegion::Base, size.next_multiple_of(0x1000)).unwrap();
        let object = system.kernel.objects.insert(KObject::SharedMemory(SharedMemory {
            name: "mic".into(),
            address: 0,
            size,
            paddr: block.addr,
            mapped_at: None,
        }));
        let handle = system.kernel.handles.create(&mut system.kernel.objects, object, "mic");
        let request = move |system: &mut System, words: &[u32]| {
            for (i, word) in words.iter().enumerate() {
                buffer.set(&mut system.memory, i as u32, *word);
            }
            handle_request(system, Target::service("mic:u".into(), 0));
            buffer.get(&mut system.memory, 1)
        };
        // MapSharedMem(size, block)
        assert_eq!(request(&mut system, &[0x0001_0042, size, 0, handle]), 0);
        (system, request, block.addr)
    }

    fn read_word(system: &mut System, address: u32) -> u32 {
        let mut word = [0u8; 4];
        system.memory.read_physical(address, &mut word);
        u32::from_le_bytes(word)
    }

    /// what the frontend's microphone hears lands in the block at the rate
    /// the title asked for, 16 bits signed, round and round, the last word
    /// saying where the next sample goes, and the event at each turn.
    #[test]
    fn samples_go_round_the_buffer() {
        let size = 0x1000;
        let (mut system, request, block) = with_block(size);
        let on = Arc::new(Mutex::new(false));
        system.microphone = Some(Box::new(Counting { next: 0.0, on: on.clone() }));
        // GetBufferFullEvent
        assert_eq!(request(&mut system, &[0x0007_0000]), 0);
        let event = system.services.mic.event.expect("the event, once asked for");
        // StartSampling(PCM16 signed, 16360 Hz, at 0, all but the last word, loop)
        let data = size - 4;
        assert_eq!(request(&mut system, &[0x0003_0140, PCM16_SIGNED as u32, 1, 0, data, 1]), 0);
        assert!(*on.lock().unwrap(), "the frontend's microphone is on");
        // IsSampling
        assert_eq!((request(&mut system, &[0x0006_0000]), system.services.mic.sampling.is_some()), (0, true));

        // 100 samples' time
        system.cpu.cycles += sample_ticks(1) * 100;
        tick(&mut system);
        assert_eq!(read_word(&mut system, block + size - 4), 200);
        let mut first = [0u8; 4];
        system.memory.read_physical(block, &mut first);
        assert_eq!(i16::from_le_bytes([first[0], first[1]]), (0.001f32 * 32767.0).round() as i16);
        assert_eq!(i16::from_le_bytes([first[2], first[3]]), (0.002f32 * 32767.0).round() as i16);
        let signaled = |system: &System| matches!(system.kernel.objects.get(event), Some(KObject::Event(event)) if event.signaled);
        assert!(!signaled(&system));

        // past the end, the rest starts over at the beginning
        let room = data / 2;
        system.cpu.cycles += sample_ticks(1) * (room as u64 - 100 + 10);
        tick(&mut system);
        assert_eq!(read_word(&mut system, block + size - 4), 20);
        assert!(signaled(&system), "the event at the end of the buffer");
        system.memory.read_physical(block, &mut first);
        let newest = (room + 1) as f32 * 0.001;
        assert_eq!(i16::from_le_bytes([first[0], first[1]]), (newest.min(1.0) * 32767.0).round() as i16);

        // StopSampling
        assert_eq!(request(&mut system, &[0x0005_0000]), 0);
        assert!(!*on.lock().unwrap(), "and off when it stops");
    }

    /// once through, 8 bits unsigned at an offset, sampling stops at the
    /// end of the buffer, the bytes after it untouched.
    #[test]
    fn once_through_stops_at_the_end() {
        let (mut system, request, block) = with_block(0x1000);
        system.memory.write_physical(block + 0x110, &[0xEE]);
        system.microphone = Some(Box::new(Recording::new(vec![0.0, 1.0], sample_rate(0))));
        // StartSampling(PCM8, 32730 Hz, at 0x10, 0x100 bytes, once)
        assert_eq!(request(&mut system, &[0x0003_0140, PCM8 as u32, 0, 0x10, 0x100, 0]), 0);
        system.cpu.cycles += sample_ticks(0) * 0x200;
        tick(&mut system);
        let mut bytes = [0u8; 4];
        system.memory.read_physical(block + 0x10, &mut bytes);
        assert_eq!(bytes, [128, 255, 128, 255]);
        let mut after = [0u8; 1];
        system.memory.read_physical(block + 0x110, &mut after);
        assert_eq!(after, [0xEE]);
        assert!(system.services.mic.sampling.is_none());
        assert_eq!(read_word(&mut system, block + 0x1000 - 4), 0x110);
    }

    /// the encodings put silence and the loudest sound where the console
    /// does.
    #[test]
    fn encodings() {
        assert_eq!(encode(&[0.0, 1.0, -1.0], PCM8), [128, 255, 1]);
        assert_eq!(encode(&[0.0, -1.0], PCM8_SIGNED), [0, 0x81]);
        assert_eq!(encode(&[0.0], PCM16), 0x8000u16.to_le_bytes());
        assert_eq!(encode(&[-1.0], PCM16_SIGNED), (-32767i16).to_le_bytes());
    }

    /// no microphone from the frontend is silence, and the settings a
    /// title reads back are the ones it set.
    #[test]
    fn settings_come_back_and_silence_without_a_microphone() {
        let (mut system, request, block) = with_block(0x1000);
        // SetGain(0x40), GetGain, SetPower(on), GetPower
        assert_eq!(request(&mut system, &[0x0008_0040, 0x40]), 0);
        assert_eq!(request(&mut system, &[0x0009_0000]), 0);
        assert_eq!(system.services.mic.gain, 0x40);
        assert_eq!(request(&mut system, &[0x000A_0040, 1]), 0);
        assert!(system.services.mic.power);
        // StartSampling(PCM16 unsigned, 8180 Hz, at 0, 0x100, loop)
        assert_eq!(request(&mut system, &[0x0003_0140, PCM16 as u32, 3, 0, 0x100, 1]), 0);
        system.cpu.cycles += sample_ticks(3) * 4;
        tick(&mut system);
        let mut bytes = [0u8; 8];
        system.memory.read_physical(block, &mut bytes);
        assert_eq!(bytes, [0, 0x80, 0, 0x80, 0, 0x80, 0, 0x80]);
        assert!(system.services.unimplemented.is_empty());
    }
}
