//! the audio DSP's voices, simulated for their timing.

use zakuro_cpu::Bus;

use crate::memory::Memory;

pub const VOICES: usize = 24;
/// samples in one audio frame.
const FRAME_SAMPLES: usize = 160;

const CONFIG_SIZE: u32 = 192;
const STATUS_SIZE: u32 = 12;
/// region 1 of the shared memory starts this many bytes after region 0.
const REGION_STRIDE: u32 = 0x2_0000;
/// the longest embedded buffer taken at face value, some titles give the
/// embedded buffer a nonsensical length.
const MAX_EMBEDDED_LENGTH: u32 = 44_100;

/// bits of a configuration's dirty word, which fields the title changed.
mod dirty {
    pub const PARTIAL_EMBEDDED_BUFFER: u32 = 1 << 3;
    pub const PARTIAL_RESET: u32 = 1 << 4;
    pub const ENABLE: u32 = 1 << 16;
    pub const RATE_MULTIPLIER: u32 = 1 << 18;
    pub const BUFFER_QUEUE: u32 = 1 << 19;
    pub const PLAY_POSITION: u32 = 1 << 21;
    pub const SYNC_COUNT: u32 = 1 << 28;
    pub const RESET: u32 = 1 << 29;
    pub const EMBEDDED_BUFFER: u32 = 1 << 30;
}

/// offsets into one voice's configuration.
mod config {
    pub const DIRTY: u32 = 0x00;
    pub const RATE_MULTIPLIER: u32 = 0x34;
    pub const BUFFERS_DIRTY: u32 = 0x4A;
    pub const BUFFERS: u32 = 0x4C;
    pub const BUFFER_SIZE: u32 = 20;
    pub const ENABLE: u32 = 0xA0;
    pub const SYNC_COUNT: u32 = 0xA2;
    pub const PLAY_POSITION: u32 = 0xA4;
    pub const LENGTH: u32 = 0xB0;
    pub const FLAGS2: u32 = 0xBC;
    pub const BUFFER_ID: u32 = 0xBE;
}

/// addresses of the structures the voices use, in region 0.
#[derive(Debug, Clone, Copy)]
pub struct Layout {
    pub configurations: u32,
    pub statuses: u32,
    pub frame_counter: u32,
}

#[derive(Debug, Clone, Copy)]
struct Buffer {
    length: u32,
    looping: bool,
    id: u16,
    /// queued buffers report when they start, the embedded one does not.
    from_queue: bool,
    play_position: u32,
    has_played: bool,
}

#[derive(Debug, Clone)]
struct Voice {
    enabled: bool,
    sync_count: u16,
    rate: f64,
    queue: Vec<Buffer>,
    /// samples left in the buffer being played, and how long it is.
    remaining: u32,
    length: u32,
    /// input consumed but not yet a whole sample.
    fraction: f64,
    current_buffer_id: u16,
    last_buffer_id: u16,
    /// set when a new queued buffer starts, reported once.
    buffer_update: bool,
    /// where in its buffer the voice was at the start of the frame.
    position: u32,
}

impl Default for Voice {
    fn default() -> Self {
        Voice {
            enabled: false,
            sync_count: 0,
            rate: 1.0,
            queue: Vec::new(),
            remaining: 0,
            length: 0,
            fraction: 0.0,
            current_buffer_id: 0,
            last_buffer_id: 0,
            buffer_update: false,
            position: 0,
        }
    }
}

impl Voice {
    /// starts the next buffer, the lowest id waiting.
    fn dequeue(&mut self) -> bool {
        let Some(index) = (0..self.queue.len()).min_by_key(|&i| self.queue[i].id) else {
            return false;
        };
        let buffer = self.queue[index];
        if buffer.looping {
            self.queue[index].has_played = true;
        } else {
            self.queue.remove(index);
        }

        // only the first time through starts at the play position.
        let start = if buffer.has_played { 0 } else { buffer.play_position.min(buffer.length) };
        self.length = buffer.length;
        self.remaining = buffer.length - start;
        self.position = start;
        self.fraction = 0.0;
        self.current_buffer_id = buffer.id;
        self.last_buffer_id = 0;
        self.buffer_update = buffer.from_queue && !buffer.has_played;
        true
    }

    /// plays one audio frame's worth of input.
    fn play_frame(&mut self) {
        if self.remaining == 0 {
            if self.dequeue() {
                return;
            }
            // out of buffers, the voice switches itself off.
            self.enabled = false;
            self.buffer_update = true;
            self.last_buffer_id = self.current_buffer_id;
            self.position = 0;
            return;
        }

        self.position = self.length - self.remaining;
        let mut output = 0;
        while output < FRAME_SAMPLES {
            if self.remaining == 0 && !self.dequeue() {
                break;
            }
            // each output sample consumes rate input samples.
            let covers = ((self.remaining as f64 - self.fraction) / self.rate).ceil().max(1.0) as usize;
            let take = covers.min(FRAME_SAMPLES - output);
            let input = self.fraction + take as f64 * self.rate;
            let consumed = (input.floor() as u32).min(self.remaining);
            self.fraction = (input - consumed as f64).max(0.0);
            self.remaining -= consumed;
            output += take;
        }
    }
}

#[derive(Debug, Clone)]
pub struct Voices {
    voices: Vec<Voice>,
}

impl Default for Voices {
    fn default() -> Self {
        Voices {
            voices: vec![Voice::default(); VOICES],
        }
    }
}

impl Voices {
    /// one audio frame, take in what the title changed, play, report.
    pub fn tick(&mut self, memory: &mut Memory, layout: Layout) {
        let read = current_region(memory, layout);
        let write = 1 - read;
        if log::log_enabled!(log::Level::Trace) {
            log::trace!(
                "dsp: tick, frame counters {} / {}, reading region {read}",
                memory.read16(layout.frame_counter),
                memory.read16(layout.frame_counter + REGION_STRIDE),
            );
        }
        let configurations = layout.configurations + read * REGION_STRIDE;
        let statuses = layout.statuses + write * REGION_STRIDE;

        for (index, voice) in self.voices.iter_mut().enumerate() {
            let base = configurations + index as u32 * CONFIG_SIZE;
            let was_enabled = voice.enabled;
            let queued = voice.queue.len();
            parse_config(voice, memory, base);
            if voice.enabled != was_enabled || voice.queue.len() > queued {
                log::debug!(
                    "dsp: voice {index} {} with {} queued (ids {:?}), rate {}, playing {} with {} left",
                    if voice.enabled { "on" } else { "off" },
                    voice.queue.len(),
                    voice.queue.iter().map(|b| (b.id, b.length, b.looping)).collect::<Vec<_>>(),
                    voice.rate,
                    voice.current_buffer_id,
                    voice.remaining,
                );
            }
            if voice.enabled {
                voice.play_frame();
            }
            write_status(voice, memory, statuses + index as u32 * STATUS_SIZE);
        }
    }
}

/// which region the title wrote most recently, allowing for the frame
/// counters wrapping around.
fn current_region(memory: &mut Memory, layout: Layout) -> u32 {
    let first = memory.read16(layout.frame_counter);
    let second = memory.read16(layout.frame_counter + REGION_STRIDE);
    if first == 0xFFFF && second != 0xFFFE {
        return 1;
    }
    if second == 0xFFFF && first != 0xFFFE {
        return 0;
    }
    if first > second {
        0
    } else {
        1
    }
}

/// reads a 32-bit value the way the DSP stores it, high half first.
fn read_dsp32(memory: &mut Memory, address: u32) -> u32 {
    ((memory.read16(address) as u32) << 16) | memory.read16(address + 2) as u32
}

fn write_dsp32(memory: &mut Memory, address: u32, value: u32) {
    memory.write16(address, (value >> 16) as u16);
    memory.write16(address + 2, value as u16);
}

/// applies whatever the title changed in a voice's configuration, then
/// clears its dirty flags the way the firmware does.
fn parse_config(voice: &mut Voice, memory: &mut Memory, base: u32) {
    let flags = memory.read32(base + config::DIRTY);
    if flags == 0 {
        return;
    }

    if flags & dirty::RESET != 0 {
        *voice = Voice::default();
    }
    if flags & dirty::PARTIAL_RESET != 0 {
        voice.queue.clear();
    }
    if flags & dirty::ENABLE != 0 {
        voice.enabled = memory.read8(base + config::ENABLE) != 0;
    }
    if flags & dirty::SYNC_COUNT != 0 {
        voice.sync_count = memory.read16(base + config::SYNC_COUNT);
    }
    if flags & dirty::RATE_MULTIPLIER != 0 {
        let rate = f32::from_bits(memory.read32(base + config::RATE_MULTIPLIER)) as f64;
        voice.rate = if rate > 0.0 && rate.is_finite() { rate } else { 1.0 };
    }

    let play_position = if flags & dirty::PLAY_POSITION != 0 {
        read_dsp32(memory, base + config::PLAY_POSITION)
    } else {
        0
    };

    // the title lengthened the buffer that is already playing.
    if flags & dirty::PARTIAL_EMBEDDED_BUFFER != 0 {
        let length = read_dsp32(memory, base + config::LENGTH);
        let played = voice.length - voice.remaining;
        voice.length = length.max(played);
        voice.remaining = voice.length - played;
    }

    if flags & dirty::EMBEDDED_BUFFER != 0 {
        voice.queue.push(Buffer {
            length: read_dsp32(memory, base + config::LENGTH).min(MAX_EMBEDDED_LENGTH),
            looping: memory.read16(base + config::FLAGS2) & 0x2 != 0,
            id: memory.read16(base + config::BUFFER_ID),
            from_queue: false,
            play_position,
            has_played: false,
        });
    }

    if flags & dirty::BUFFER_QUEUE != 0 {
        let queued = memory.read16(base + config::BUFFERS_DIRTY);
        for slot in 0..4 {
            if queued & (1 << slot) == 0 {
                continue;
            }
            let buffer = base + config::BUFFERS + slot * config::BUFFER_SIZE;
            let length = read_dsp32(memory, buffer + 4);
            if length != 0 {
                voice.queue.push(Buffer {
                    length,
                    looping: memory.read8(buffer + 15) != 0,
                    id: memory.read16(buffer + 16),
                    from_queue: true,
                    play_position: 0,
                    has_played: false,
                });
            }
        }
        memory.write16(base + config::BUFFERS_DIRTY, 0);
    }

    memory.write32(base + config::DIRTY, 0);
}

fn write_status(voice: &mut Voice, memory: &mut Memory, base: u32) {
    memory.write8(base, voice.enabled as u8);
    memory.write8(base + 1, voice.buffer_update as u8);
    voice.buffer_update = false;
    memory.write16(base + 2, voice.sync_count);
    write_dsp32(memory, base + 4, voice.position);
    memory.write16(base + 8, voice.current_buffer_id);
    memory.write16(base + 10, voice.last_buffer_id);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn voice_with(buffers: &[(u16, u32)]) -> Voice {
        let mut voice = Voice {
            enabled: true,
            ..Voice::default()
        };
        for &(id, length) in buffers {
            voice.queue.push(Buffer {
                length,
                looping: false,
                id,
                from_queue: true,
                play_position: 0,
                has_played: false,
            });
        }
        voice
    }

    #[test]
    fn buffers_play_lowest_id_first_and_report_starting() {
        let mut voice = voice_with(&[(7, 320), (6, 320)]);
        voice.play_frame();
        assert_eq!(voice.current_buffer_id, 6);
        assert!(voice.buffer_update, "a queued buffer reports that it started");
    }

    /// at a rate of one, a 320-sample buffer lasts two audio frames.
    #[test]
    fn a_voice_moves_through_its_buffers_in_real_time() {
        let mut voice = voice_with(&[(1, 320), (2, 320)]);
        voice.play_frame(); // starts buffer 1
        voice.play_frame(); // plays 160
        assert_eq!(voice.current_buffer_id, 1);
        assert_eq!(voice.remaining, 160);
        voice.play_frame(); // finishes buffer 1
        voice.play_frame(); // starts buffer 2 on the way
        assert_eq!(voice.current_buffer_id, 2);
    }

    #[test]
    fn a_higher_rate_consumes_input_faster() {
        let mut voice = voice_with(&[(1, 1000)]);
        voice.rate = 2.0;
        voice.play_frame();
        voice.play_frame();
        assert_eq!(voice.remaining, 1000 - 320);
    }

    #[test]
    fn running_out_of_buffers_switches_the_voice_off() {
        let mut voice = voice_with(&[(4, 160)]);
        for _ in 0..3 {
            voice.play_frame();
        }
        assert!(!voice.enabled);
        assert_eq!(voice.last_buffer_id, 4);
    }

    #[test]
    fn a_looping_buffer_keeps_playing() {
        let mut voice = voice_with(&[]);
        voice.queue.push(Buffer {
            length: 160,
            looping: true,
            id: 9,
            from_queue: true,
            play_position: 0,
            has_played: false,
        });
        for _ in 0..10 {
            voice.play_frame();
        }
        assert!(voice.enabled);
        assert_eq!(voice.current_buffer_id, 9);
    }
}
