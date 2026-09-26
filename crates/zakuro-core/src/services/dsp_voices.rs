//! the audio DSP's voices, which play their buffers, report on them, and
//! give the mixer what they sound like.

use zakuro_cpu::Bus;

use crate::memory::Memory;

pub const VOICES: usize = 24;
/// samples in one audio frame.
pub const FRAME_SAMPLES: usize = 160;

const CONFIG_SIZE: u32 = 192;
const STATUS_SIZE: u32 = 12;
/// the 16 ADPCM coefficients each voice has.
const COEFFICIENTS_SIZE: u32 = 32;
/// region 1 of the shared memory starts this many bytes after region 0.
pub const REGION_STRIDE: u32 = 0x2_0000;
/// the longest embedded buffer taken at face value, some titles give the
/// embedded buffer a nonsensical length.
const MAX_EMBEDDED_LENGTH: u32 = 44_100;

/// one frame of stereo samples.
pub type Frame = [[i16; 2]; FRAME_SAMPLES];
/// one frame of an intermediate mix, two left and two right channels.
pub type QuadFrame = [[i32; 4]; FRAME_SAMPLES];

/// bits of a configuration's dirty word, which fields the title changed.
mod dirty {
    pub const FORMAT: u32 = 1 << 0;
    pub const MONO_OR_STEREO: u32 = 1 << 1;
    pub const ADPCM_COEFFICIENTS: u32 = 1 << 2;
    pub const PARTIAL_EMBEDDED_BUFFER: u32 = 1 << 3;
    pub const PARTIAL_RESET: u32 = 1 << 4;
    pub const ENABLE: u32 = 1 << 16;
    pub const INTERPOLATION: u32 = 1 << 17;
    pub const RATE_MULTIPLIER: u32 = 1 << 18;
    pub const BUFFER_QUEUE: u32 = 1 << 19;
    pub const PLAY_POSITION: u32 = 1 << 21;
    pub const FILTERS_ENABLED: u32 = 1 << 22;
    pub const SIMPLE_FILTER: u32 = 1 << 23;
    pub const BIQUAD_FILTER: u32 = 1 << 24;
    /// one bit per intermediate mix from here up.
    pub const GAIN: u32 = 1 << 25;
    pub const SYNC_COUNT: u32 = 1 << 28;
    pub const RESET: u32 = 1 << 29;
    pub const EMBEDDED_BUFFER: u32 = 1 << 30;
}

/// offsets into one voice's configuration.
mod config {
    pub const DIRTY: u32 = 0x00;
    pub const GAIN: u32 = 0x04;
    pub const RATE_MULTIPLIER: u32 = 0x34;
    pub const INTERPOLATION: u32 = 0x38;
    pub const FILTERS_ENABLED: u32 = 0x3A;
    pub const SIMPLE_FILTER: u32 = 0x3C;
    pub const BIQUAD_FILTER: u32 = 0x40;
    pub const BUFFERS_DIRTY: u32 = 0x4A;
    pub const BUFFERS: u32 = 0x4C;
    pub const BUFFER_SIZE: u32 = 20;
    pub const ENABLE: u32 = 0xA0;
    pub const SYNC_COUNT: u32 = 0xA2;
    pub const PLAY_POSITION: u32 = 0xA4;
    pub const ADDRESS: u32 = 0xAC;
    pub const LENGTH: u32 = 0xB0;
    pub const FLAGS1: u32 = 0xB4;
    pub const ADPCM_YN: u32 = 0xB8;
    pub const FLAGS2: u32 = 0xBC;
    pub const BUFFER_ID: u32 = 0xBE;
}

/// addresses of the structures the voices use, in region 0.
#[derive(Debug, Clone, Copy)]
pub struct Layout {
    pub configurations: u32,
    pub statuses: u32,
    pub coefficients: u32,
    pub frame_counter: u32,
}

/// where voices read their samples, physical memory.
pub trait SampleMemory {
    fn physical(&mut self, address: u32, length: u32) -> Option<&[u8]>;
}

impl SampleMemory for Memory {
    fn physical(&mut self, address: u32, length: u32) -> Option<&[u8]> {
        self.phys.host_slice_mut(address, length).map(|slice| &*slice)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    Pcm8,
    Pcm16,
    Adpcm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Interpolation {
    /// the firmware's polyphase filter, played as linear.
    Polyphase,
    Linear,
    None,
}

#[derive(Debug, Clone, Copy)]
struct Buffer {
    address: u32,
    length: u32,
    format: Format,
    stereo: bool,
    looping: bool,
    id: u16,
    /// the ADPCM history to start from, when the title set it.
    adpcm: Option<[i16; 2]>,
    /// queued buffers report when they start, the embedded one does not.
    from_queue: bool,
    play_position: u32,
    has_played: bool,
}

/// the first order filter, y = b0 x + a1 y1, with 15 fraction bits.
#[derive(Debug, Clone, Copy)]
struct SimpleFilter {
    b0: i32,
    a1: i32,
    y1: [i32; 2],
}

impl Default for SimpleFilter {
    fn default() -> Self {
        SimpleFilter { b0: 1 << 15, a1: 0, y1: [0; 2] }
    }
}

/// the second order filter, with 14 fraction bits and its feedback negated.
#[derive(Debug, Clone, Copy)]
struct BiquadFilter {
    b: [i32; 3],
    a: [i32; 2],
    x: [[i32; 2]; 2],
    y: [[i32; 2]; 2],
}

impl Default for BiquadFilter {
    fn default() -> Self {
        BiquadFilter { b: [1 << 14, 0, 0], a: [0; 2], x: [[0; 2]; 2], y: [[0; 2]; 2] }
    }
}

#[derive(Debug, Clone, Default)]
struct Filters {
    simple: Option<SimpleFilter>,
    biquad: Option<BiquadFilter>,
}

impl Filters {
    fn process(&mut self, samples: &mut [[i16; 2]]) {
        for sample in samples {
            for (channel, out) in sample.iter_mut().enumerate() {
                let mut value = *out as i32;
                if let Some(f) = &mut self.simple {
                    value = ((f.b0 * value + f.a1 * f.y1[channel]) >> 15).clamp(-32768, 32767);
                    f.y1[channel] = value;
                }
                if let Some(f) = &mut self.biquad {
                    let x0 = value;
                    value = ((f.b[0] * x0 + f.b[1] * f.x[0][channel] + f.b[2] * f.x[1][channel]
                        + f.a[0] * f.y[0][channel]
                        + f.a[1] * f.y[1][channel])
                        >> 14)
                        .clamp(-32768, 32767);
                    f.x[1][channel] = f.x[0][channel];
                    f.x[0][channel] = x0;
                    f.y[1][channel] = f.y[0][channel];
                    f.y[0][channel] = value;
                }
                *out = value as i16;
            }
        }
    }
}

#[derive(Debug, Clone)]
struct Voice {
    enabled: bool,
    sync_count: u16,
    rate: f64,
    format: Format,
    stereo: bool,
    interpolation: Interpolation,
    filters: Filters,
    /// how loud the voice is in each channel of each intermediate mix.
    gain: [[f32; 4]; 3],
    /// the gains a mix fades from over the next frame, after they changed.
    ramp: [Option<[f32; 4]>; 3],
    coefficients: [i16; 16],
    /// the last two ADPCM samples, which the next ones follow on from.
    history: [i16; 2],
    queue: Vec<Buffer>,
    /// the buffer being played.
    playing: Option<Buffer>,
    /// an ADPCM buffer, decoded when it started.
    decoded: Vec<i16>,
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
    /// what the voice played this frame.
    frame: Frame,
}

impl Default for Voice {
    fn default() -> Self {
        Voice {
            enabled: false,
            sync_count: 0,
            rate: 1.0,
            format: Format::Pcm16,
            stereo: false,
            interpolation: Interpolation::Polyphase,
            filters: Filters::default(),
            gain: [[0.0; 4]; 3],
            ramp: [None; 3],
            coefficients: [0; 16],
            history: [0; 2],
            queue: Vec::new(),
            playing: None,
            decoded: Vec::new(),
            remaining: 0,
            length: 0,
            fraction: 0.0,
            current_buffer_id: 0,
            last_buffer_id: 0,
            buffer_update: false,
            position: 0,
            frame: [[0; 2]; FRAME_SAMPLES],
        }
    }
}

impl Voice {
    /// starts the next buffer, the lowest id waiting.
    fn dequeue(&mut self, memory: &mut impl SampleMemory) -> bool {
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

        if let Some(history) = buffer.adpcm {
            self.history = history;
        }
        self.decoded.clear();
        if buffer.format == Format::Adpcm {
            let bytes = buffer.length.div_ceil(14) * 8;
            match memory.physical(buffer.address & !3, bytes) {
                Some(data) => self.decoded = decode_adpcm(data, buffer.length as usize, &self.coefficients, &mut self.history),
                None => log::debug!("dsp: an ADPCM buffer at 0x{:08X} is not in memory", buffer.address),
            }
        }
        self.playing = Some(buffer);
        true
    }

    /// input sample index of the buffer being played, both channels.
    fn input(&self, memory: &mut impl SampleMemory, index: u32) -> [i16; 2] {
        let Some(buffer) = &self.playing else { return [0; 2] };
        let index = index.min(buffer.length.saturating_sub(1));
        let channels = if buffer.stereo { 2 } else { 1 };
        let at = buffer.address & !3;
        let [left, right] = match buffer.format {
            Format::Adpcm => {
                let sample = self.decoded.get(index as usize).copied().unwrap_or(0);
                [sample, sample]
            }
            Format::Pcm8 => match memory.physical(at + index * channels, channels) {
                Some(data) => [data[0] as i8 as i16 * 256, data[data.len() - 1] as i8 as i16 * 256],
                None => [0; 2],
            },
            Format::Pcm16 => match memory.physical(at + index * channels * 2, channels * 2) {
                Some(data) => [
                    i16::from_le_bytes([data[0], data[1]]),
                    i16::from_le_bytes([data[data.len() - 2], data[data.len() - 1]]),
                ],
                None => [0; 2],
            },
        };
        [left, right]
    }

    /// the output sample at the current position, between two inputs.
    fn sample(&self, memory: &mut impl SampleMemory) -> [i16; 2] {
        let at = self.length - self.remaining;
        let x0 = self.input(memory, at);
        if self.interpolation == Interpolation::None || self.fraction == 0.0 {
            return x0;
        }
        let x1 = self.input(memory, at + 1);
        let between = |a: i16, b: i16| (a as f64 + (b as f64 - a as f64) * self.fraction) as i16;
        [between(x0[0], x1[0]), between(x0[1], x1[1])]
    }

    /// plays one audio frame's worth of input.
    fn play_frame(&mut self, memory: &mut impl SampleMemory) {
        self.frame = [[0; 2]; FRAME_SAMPLES];
        if self.remaining == 0 {
            if self.dequeue(memory) {
                return;
            }
            // out of buffers, the voice switches itself off and reports the
            // buffer it finished as the last one, with none current. a title
            // waits on that to know its sound is over.
            self.enabled = false;
            self.buffer_update = true;
            self.last_buffer_id = self.current_buffer_id;
            self.current_buffer_id = 0;
            self.position = 0;
            return;
        }

        self.position = self.length - self.remaining;
        let mut output = 0;
        while output < FRAME_SAMPLES {
            if self.remaining == 0 && !self.dequeue(memory) {
                break;
            }
            self.frame[output] = self.sample(memory);
            output += 1;
            // each output sample consumes rate input samples.
            let input = self.fraction + self.rate;
            let consumed = (input.floor() as u32).min(self.remaining);
            self.fraction = (input - consumed as f64).max(0.0);
            self.remaining -= consumed;
        }
        self.filters.process(&mut self.frame[..output]);
    }

    /// adds what the voice played to one intermediate mix, fading from the
    /// gains it had when they just changed.
    fn mix_into(&mut self, mix: &mut QuadFrame, index: usize) {
        let gains = self.gain[index];
        let from = self.ramp[index].take();
        if !self.enabled {
            return;
        }
        for (i, (out, sample)) in mix.iter_mut().zip(&self.frame).enumerate() {
            let progress = i as f32 / (FRAME_SAMPLES - 1) as f32;
            let gain = |c: usize| from.map_or(gains[c], |from| from[c] + (gains[c] - from[c]) * progress);
            out[0] += (gain(0) * sample[0] as f32) as i32;
            out[1] += (gain(1) * sample[1] as f32) as i32;
            out[2] += (gain(2) * sample[0] as f32) as i32;
            out[3] += (gain(3) * sample[1] as f32) as i32;
        }
    }
}

/// GC ADPCM, frames of 8 bytes holding a header and 14 samples of 4 bits,
/// each predicted from the two before it.
fn decode_adpcm(data: &[u8], count: usize, coefficients: &[i16; 16], history: &mut [i16; 2]) -> Vec<i16> {
    let mut samples = Vec::with_capacity(count);
    let [mut yn1, mut yn2] = history.map(|h| h as i32);
    for frame in data.chunks(8) {
        let header = frame[0];
        let scale = 1 << (header & 0xF);
        let predictor = ((header >> 4) & 7) as usize;
        let (c1, c2) = (coefficients[predictor * 2] as i32, coefficients[predictor * 2 + 1] as i32);
        for &byte in &frame[1..] {
            for nibble in [byte >> 4, byte & 0xF] {
                if samples.len() == count {
                    *history = [yn1 as i16, yn2 as i16];
                    return samples;
                }
                // the nibble is signed
                let xn = (((nibble as i32) << 28) >> 28) * scale;
                let value = (((xn << 11) + 0x400 + c1 * yn1 + c2 * yn2) >> 11).clamp(-32768, 32767);
                yn2 = yn1;
                yn1 = value;
                samples.push(value as i16);
            }
        }
    }
    *history = [yn1 as i16, yn2 as i16];
    samples
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
    /// one audio frame, take in what the title changed, play, report, and
    /// return the three intermediate mixes the voices made.
    pub fn tick(&mut self, memory: &mut Memory, layout: Layout) -> [QuadFrame; 3] {
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
        let coefficients = layout.coefficients + read * REGION_STRIDE;
        let statuses = layout.statuses + write * REGION_STRIDE;

        let mut mixes = [[[0; 4]; FRAME_SAMPLES]; 3];
        for (index, voice) in self.voices.iter_mut().enumerate() {
            let base = configurations + index as u32 * CONFIG_SIZE;
            let was_enabled = voice.enabled;
            let queued = voice.queue.len();
            parse_config(voice, memory, base, coefficients + index as u32 * COEFFICIENTS_SIZE);
            if voice.enabled != was_enabled || voice.queue.len() > queued {
                log::debug!(
                    "dsp: voice {index} {} with {} queued (ids {:?}), rate {}, playing {} with {} left, \
                     {:?} {} {:?}, gains {:?}",
                    if voice.enabled { "on" } else { "off" },
                    voice.queue.len(),
                    voice.queue.iter().map(|b| (b.id, b.length, b.looping)).collect::<Vec<_>>(),
                    voice.rate,
                    voice.current_buffer_id,
                    voice.remaining,
                    voice.format,
                    if voice.stereo { "stereo" } else { "mono" },
                    voice.interpolation,
                    voice.gain,
                );
            }
            if voice.enabled {
                voice.play_frame(memory);
            }
            for (mix, frame) in mixes.iter_mut().enumerate() {
                voice.mix_into(frame, mix);
            }
            write_status(voice, memory, statuses + index as u32 * STATUS_SIZE);
        }
        mixes
    }
}

/// which region the title wrote most recently, allowing for the frame
/// counters wrapping around.
pub fn current_region(memory: &mut Memory, layout: Layout) -> u32 {
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
fn parse_config(voice: &mut Voice, memory: &mut Memory, base: u32, coefficients: u32) {
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
    if flags & dirty::INTERPOLATION != 0 {
        voice.interpolation = match memory.read8(base + config::INTERPOLATION) {
            1 => Interpolation::Linear,
            2 => Interpolation::None,
            _ => Interpolation::Polyphase,
        };
    }
    if flags & dirty::ADPCM_COEFFICIENTS != 0 {
        for (i, coefficient) in voice.coefficients.iter_mut().enumerate() {
            *coefficient = memory.read16(coefficients + i as u32 * 2) as i16;
        }
    }
    for mix in 0..3 {
        if flags & (dirty::GAIN << mix) != 0 {
            voice.ramp[mix] = Some(voice.gain[mix]);
            for channel in 0..4 {
                let at = base + config::GAIN + (mix as u32 * 4 + channel as u32) * 4;
                let gain = f32::from_bits(memory.read32(at));
                voice.gain[mix][channel] = if gain.is_finite() { gain } else { 0.0 };
            }
        }
    }
    if flags & dirty::FILTERS_ENABLED != 0 {
        let enabled = memory.read16(base + config::FILTERS_ENABLED);
        // a filter switched off forgets its state and coefficients
        if enabled & 1 == 0 {
            voice.filters.simple = None;
        } else if voice.filters.simple.is_none() {
            voice.filters.simple = Some(SimpleFilter::default());
        }
        if enabled & 2 == 0 {
            voice.filters.biquad = None;
        } else if voice.filters.biquad.is_none() {
            voice.filters.biquad = Some(BiquadFilter::default());
        }
    }
    let half = |memory: &mut Memory, offset: u32| memory.read16(base + offset) as i16 as i32;
    if flags & dirty::SIMPLE_FILTER != 0 {
        let filter = voice.filters.simple.get_or_insert_with(SimpleFilter::default);
        filter.b0 = half(memory, config::SIMPLE_FILTER);
        filter.a1 = half(memory, config::SIMPLE_FILTER + 2);
    }
    if flags & dirty::BIQUAD_FILTER != 0 {
        let filter = voice.filters.biquad.get_or_insert_with(BiquadFilter::default);
        let field = |memory: &mut Memory, i: u32| memory.read16(base + config::BIQUAD_FILTER + i * 2) as i16 as i32;
        filter.a = [field(memory, 1), field(memory, 0)];
        filter.b = [field(memory, 4), field(memory, 3), field(memory, 2)];
    }
    // the embedded buffer brings its format along
    let flags1 = memory.read16(base + config::FLAGS1);
    if flags & (dirty::FORMAT | dirty::EMBEDDED_BUFFER) != 0 {
        voice.format = match (flags1 >> 2) & 3 {
            0 => Format::Pcm8,
            2 => Format::Adpcm,
            _ => Format::Pcm16,
        };
    }
    if flags & (dirty::MONO_OR_STEREO | dirty::EMBEDDED_BUFFER) != 0 {
        voice.stereo = flags1 & 3 == 2;
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
        if let Some(playing) = &mut voice.playing {
            playing.length = voice.length;
        }
    }

    if flags & dirty::EMBEDDED_BUFFER != 0 {
        let flags2 = memory.read16(base + config::FLAGS2);
        voice.queue.push(Buffer {
            address: read_dsp32(memory, base + config::ADDRESS),
            length: read_dsp32(memory, base + config::LENGTH).min(MAX_EMBEDDED_LENGTH),
            format: voice.format,
            stereo: voice.stereo,
            looping: flags2 & 0x2 != 0,
            id: memory.read16(base + config::BUFFER_ID),
            adpcm: (flags2 & 1 != 0).then(|| {
                [
                    memory.read16(base + config::ADPCM_YN) as i16,
                    memory.read16(base + config::ADPCM_YN + 2) as i16,
                ]
            }),
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
                    address: read_dsp32(memory, buffer),
                    length,
                    format: voice.format,
                    stereo: voice.stereo,
                    looping: memory.read8(buffer + 15) != 0,
                    id: memory.read16(buffer + 16),
                    adpcm: (memory.read8(buffer + 14) != 0)
                        .then(|| [memory.read16(buffer + 10) as i16, memory.read16(buffer + 12) as i16]),
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

    const BASE: u32 = 0x2000_0000;

    /// physical memory holding one buffer's bytes at BASE.
    struct Samples(Vec<u8>);

    impl SampleMemory for Samples {
        fn physical(&mut self, address: u32, length: u32) -> Option<&[u8]> {
            let start = address.checked_sub(BASE)? as usize;
            self.0.get(start..start + length as usize)
        }
    }

    fn silence() -> Samples {
        Samples(vec![0; 0x10000])
    }

    fn buffer(id: u16, length: u32) -> Buffer {
        Buffer {
            address: BASE,
            length,
            format: Format::Pcm16,
            stereo: false,
            looping: false,
            id,
            adpcm: None,
            from_queue: true,
            play_position: 0,
            has_played: false,
        }
    }

    fn voice_with(buffers: &[(u16, u32)]) -> Voice {
        let mut voice = Voice {
            enabled: true,
            ..Voice::default()
        };
        for &(id, length) in buffers {
            voice.queue.push(buffer(id, length));
        }
        voice
    }

    #[test]
    fn buffers_play_lowest_id_first_and_report_starting() {
        let mut voice = voice_with(&[(7, 320), (6, 320)]);
        voice.play_frame(&mut silence());
        assert_eq!(voice.current_buffer_id, 6);
        assert!(voice.buffer_update, "a queued buffer reports that it started");
    }

    /// at a rate of one, a 320-sample buffer lasts two audio frames.
    #[test]
    fn a_voice_moves_through_its_buffers_in_real_time() {
        let mut memory = silence();
        let mut voice = voice_with(&[(1, 320), (2, 320)]);
        voice.play_frame(&mut memory); // starts buffer 1
        voice.play_frame(&mut memory); // plays 160
        assert_eq!(voice.current_buffer_id, 1);
        assert_eq!(voice.remaining, 160);
        voice.play_frame(&mut memory); // finishes buffer 1
        voice.play_frame(&mut memory); // starts buffer 2 on the way
        assert_eq!(voice.current_buffer_id, 2);
    }

    #[test]
    fn a_higher_rate_consumes_input_faster() {
        let mut memory = silence();
        let mut voice = voice_with(&[(1, 1000)]);
        voice.rate = 2.0;
        voice.play_frame(&mut memory);
        voice.play_frame(&mut memory);
        assert_eq!(voice.remaining, 1000 - 320);
    }

    #[test]
    fn running_out_of_buffers_switches_the_voice_off() {
        let mut memory = silence();
        let mut voice = voice_with(&[(4, 160)]);
        for _ in 0..3 {
            voice.play_frame(&mut memory);
        }
        assert!(!voice.enabled);
        assert_eq!(voice.last_buffer_id, 4);
        assert_eq!(voice.current_buffer_id, 0, "no buffer is current once it stopped");
    }

    #[test]
    fn a_looping_buffer_keeps_playing() {
        let mut memory = silence();
        let mut voice = voice_with(&[]);
        voice.queue.push(Buffer { length: 160, looping: true, id: 9, ..buffer(9, 160) });
        for _ in 0..10 {
            voice.play_frame(&mut memory);
        }
        assert!(voice.enabled);
        assert_eq!(voice.current_buffer_id, 9);
    }

    /// a ramp of PCM16 samples, played at half speed, comes out with the
    /// samples between them filled in.
    #[test]
    fn pcm16_is_read_and_interpolated() {
        let bytes: Vec<u8> = (0..400i16).flat_map(|i| (i * 100).to_le_bytes()).collect();
        let mut memory = Samples(bytes);
        let mut voice = voice_with(&[(1, 400)]);
        voice.rate = 0.5;
        voice.play_frame(&mut memory); // starts the buffer
        voice.play_frame(&mut memory);
        assert_eq!(&voice.frame[..5], &[[0, 0], [50, 50], [100, 100], [150, 150], [200, 200]]);
    }

    #[test]
    fn stereo_pcm8_keeps_its_channels_apart() {
        let mut memory = Samples([1u8, 0xFF].repeat(200));
        let mut voice = voice_with(&[]);
        voice.queue.push(Buffer { format: Format::Pcm8, stereo: true, ..buffer(1, 200) });
        voice.play_frame(&mut memory);
        voice.play_frame(&mut memory);
        assert_eq!(voice.frame[0], [256, -256]);
    }

    /// the decoder follows the reference implementation's filter, and one
    /// buffer's history carries on into the next.
    #[test]
    fn adpcm_decodes_with_its_coefficients() {
        let mut coefficients = [0i16; 16];
        // predictor 1, y = x + y1
        coefficients[2] = 1 << 11;
        let frame = [0x10, 0x12, 0x30, 0, 0, 0, 0, 0];
        let mut history = [100, 0];
        let samples = decode_adpcm(&frame, 4, &coefficients, &mut history);
        assert_eq!(samples, [101, 103, 106, 106]);
        assert_eq!(history, [106, 106]);
    }

    #[test]
    fn gains_fade_in_over_a_frame_once_changed() {
        let mut voice = Voice { enabled: true, ..Voice::default() };
        voice.frame = [[1000, 1000]; FRAME_SAMPLES];
        voice.ramp[0] = Some([0.0; 4]);
        voice.gain[0] = [1.0; 4];
        let mut mix = [[0; 4]; FRAME_SAMPLES];
        voice.mix_into(&mut mix, 0);
        assert_eq!(mix[0], [0; 4]);
        assert_eq!(mix[FRAME_SAMPLES - 1], [1000; 4]);
        let mut again = [[0; 4]; FRAME_SAMPLES];
        voice.mix_into(&mut again, 0);
        assert_eq!(again[0], [1000; 4], "the fade happens once");
    }
}
