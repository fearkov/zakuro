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

/// one frame of stereo samples.
pub type Frame = [[i16; 2]; FRAME_SAMPLES];
/// one frame of an intermediate mix, two left and two right channels, one
/// channel after the other as the title sees it.
pub type QuadFrame = [[i32; FRAME_SAMPLES]; 4];

/// the loudest gain mixed the quick way. a sample times three of them, as a
/// fade between two can reach on the way, is still well within i32.
const QUICK_GAIN: f32 = 16384.0;

/// how far through a frame each of its samples is, for gains fading over it.
const PROGRESS: [f32; FRAME_SAMPLES] = {
    let mut progress = [0.0; FRAME_SAMPLES];
    let mut i = 0;
    while i < FRAME_SAMPLES {
        progress[i] = i as f32 / (FRAME_SAMPLES - 1) as f32;
        i += 1;
    }
    progress
};

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

/// what an ADPCM buffer was decoded from, its bytes and the coefficients
/// and history the decoding started with, and the history it left.
#[derive(Debug, Clone)]
struct AdpcmSource {
    data: Vec<u8>,
    count: usize,
    coefficients: [i16; 16],
    start: [i16; 2],
    end: [i16; 2],
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
    /// what decoded came from. a looping buffer comes around the same
    /// again and again, and is not decoded again.
    decoded_from: Option<AdpcmSource>,
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
    /// the buffer ran out on the frame's last sample, so the next one
    /// waiting follows on from the next frame's first, as on the console,
    /// rather than after a frame of silence.
    ran_out_at_frame_end: bool,
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
            decoded_from: None,
            remaining: 0,
            length: 0,
            fraction: 0.0,
            current_buffer_id: 0,
            last_buffer_id: 0,
            buffer_update: false,
            position: 0,
            frame: [[0; 2]; FRAME_SAMPLES],
            ran_out_at_frame_end: false,
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
        if buffer.format == Format::Adpcm {
            let bytes = buffer.length.div_ceil(14) * 8;
            match memory.physical(buffer.address & !3, bytes) {
                Some(data) => self.decode(data, buffer.length as usize),
                None => {
                    self.forget_decoded();
                    log::debug!("dsp: an ADPCM buffer at 0x{:08X} is not in memory", buffer.address);
                }
            }
        } else {
            self.forget_decoded();
        }
        self.playing = Some(buffer);
        true
    }

    /// decodes an ADPCM buffer on from the history, unless decoded holds it
    /// already, which leaves the history where decoding it again would.
    fn decode(&mut self, data: &[u8], count: usize) {
        let start = self.history;
        if let Some(source) = &self.decoded_from {
            if source.count == count
                && source.start == start
                && source.coefficients == self.coefficients
                && source.data == data
            {
                self.history = source.end;
                return;
            }
        }
        decode_adpcm(data, count, &self.coefficients, &mut self.history, &mut self.decoded);
        let mut copy = self.decoded_from.take().map(|source| source.data).unwrap_or_default();
        copy.clear();
        copy.extend_from_slice(data);
        self.decoded_from =
            Some(AdpcmSource { data: copy, count, coefficients: self.coefficients, start, end: self.history });
    }

    fn forget_decoded(&mut self) {
        self.decoded.clear();
        self.decoded_from = None;
    }

    /// plays one audio frame's worth of input.
    fn play_frame(&mut self, memory: &mut impl SampleMemory) {
        self.frame = [[0; 2]; FRAME_SAMPLES];
        let follows_on = std::mem::take(&mut self.ran_out_at_frame_end) && !self.queue.is_empty();
        if self.remaining == 0 && !follows_on {
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
            output = self.play_buffer(memory, output);
        }
        self.ran_out_at_frame_end = self.remaining == 0 && output == FRAME_SAMPLES;
        self.filters.process(&mut self.frame[..output]);
    }

    /// plays the buffer at hand into the frame from output on, at least one
    /// sample and on until the frame is full or the buffer ran out, and
    /// returns where the frame got to. the samples come straight from the
    /// buffer's memory when all of it is there.
    fn play_buffer(&mut self, memory: &mut impl SampleMemory, output: usize) -> usize {
        let Some(buffer) = self.playing else {
            return self.play_from::<false>(output, 0, |_| [0; 2]);
        };
        let last = buffer.length.saturating_sub(1);
        let size = match buffer.format {
            Format::Adpcm => {
                let decoded = std::mem::take(&mut self.decoded);
                // the samples up to last, all there unless the title made
                // the buffer longer, read with no check on each
                let output = match decoded.get(..=last as usize) {
                    Some(samples) => self.play_from::<false>(output, last, |index| [samples[index as usize]; 2]),
                    None => self.play_from::<false>(output, last, |index| {
                        [decoded.get(index as usize).copied().unwrap_or(0); 2]
                    }),
                };
                self.decoded = decoded;
                return output;
            }
            Format::Pcm8 => 1,
            Format::Pcm16 => 2,
        };
        let channels = if buffer.stereo { 2 } else { 1 };
        let bytes = buffer.length.checked_mul(size * channels).filter(|_| buffer.length != 0);
        let data = bytes.and_then(|bytes| memory.physical(buffer.address & !3, bytes));
        match (data, buffer.format, buffer.stereo) {
            (Some(data), Format::Pcm8, false) => self.play_from::<false>(output, last, |index| {
                [data[index as usize] as i8 as i16 * 256; 2]
            }),
            (Some(data), Format::Pcm8, true) => self.play_from::<true>(output, last, |index| {
                let at = index as usize * 2;
                [data[at] as i8 as i16 * 256, data[at + 1] as i8 as i16 * 256]
            }),
            (Some(data), Format::Pcm16, false) => self.play_from::<false>(output, last, |index| {
                let at = index as usize * 2;
                [i16::from_le_bytes([data[at], data[at + 1]]); 2]
            }),
            (Some(data), Format::Pcm16, true) => self.play_from::<true>(output, last, |index| {
                let at = index as usize * 4;
                [i16::from_le_bytes([data[at], data[at + 1]]), i16::from_le_bytes([data[at + 2], data[at + 3]])]
            }),
            // not all of it in memory, each sample read as it comes
            _ => self.play_from::<true>(output, last, |index| read_pcm(memory, &buffer, index)),
        }
    }

    /// plays input into the frame from output on, read giving the samples
    /// of the buffer at hand by index, up to last, at least one sample and
    /// on until the frame is full or the buffer ran out, and returns where
    /// the frame got to. without STEREO the channels of each input sample
    /// are the same, and only the first is worked out.
    fn play_from<const STEREO: bool>(
        &mut self,
        mut output: usize,
        last: u32,
        mut read: impl FnMut(u32) -> [i16; 2],
    ) -> usize {
        let fits = self.rate < (1u32 << 31) as f64;
        loop {
            if fits && self.remaining > 0 && (0.0..1.0).contains(&self.fraction) {
                // a whole rate from no fraction leaves none, nothing to
                // interpolate
                let whole = self.fraction == 0.0 && self.rate == (self.rate as u32) as f64;
                return if self.interpolation == Interpolation::None || whole {
                    self.play_steady::<STEREO, false>(output, last, read)
                } else {
                    self.play_steady::<STEREO, true>(output, last, read)
                };
            }
            // a sample the long way, for a buffer that started spent, a
            // fraction of one or more or a rate past what u32 counts
            let at = self.length - self.remaining;
            let x0 = read(at.min(last));
            self.frame[output] = if self.interpolation == Interpolation::None || self.fraction == 0.0 {
                x0
            } else {
                lerp::<STEREO>(x0, read((at + 1).min(last)), self.fraction)
            };
            output += 1;
            // each output sample consumes rate input samples
            let input = self.fraction + self.rate;
            let consumed = (input as u32).min(self.remaining);
            self.fraction = (input - consumed as f64).max(0.0);
            self.remaining -= consumed;
            if output == FRAME_SAMPLES || self.remaining == 0 {
                return output;
            }
        }
    }

    /// plays on as play_from does, with the fraction below one, which makes
    /// what each output sample consumes the whole of the rate or one more,
    /// and leaves the fraction to a subtraction rather than a trip through
    /// integers. LERP interpolates, which at no fraction gives the input
    /// sample as it is.
    fn play_steady<const STEREO: bool, const LERP: bool>(
        &mut self,
        mut output: usize,
        last: u32,
        mut read: impl FnMut(u32) -> [i16; 2],
    ) -> usize {
        let (length, rate) = (self.length, self.rate);
        let (mut remaining, mut fraction) = (self.remaining, self.fraction);
        let whole = rate as u32;
        let (below, above) = (whole as f64, whole as f64 + 1.0);
        loop {
            let at = length - remaining;
            let x0 = read(at.min(last));
            self.frame[output] = if LERP { lerp::<STEREO>(x0, read((at + 1).min(last)), fraction) } else { x0 };
            output += 1;
            let input = fraction + rate;
            let over = input >= above;
            let consumed = whole + over as u32;
            if consumed >= remaining {
                // the buffer runs out with this sample
                let consumed = (input as u32).min(remaining);
                fraction = (input - consumed as f64).max(0.0);
                remaining -= consumed;
                break;
            }
            fraction = input - if over { above } else { below };
            remaining -= consumed;
            if output == FRAME_SAMPLES {
                break;
            }
        }
        self.remaining = remaining;
        self.fraction = fraction;
        output
    }

    /// adds what the voice played to the intermediate mixes, each fading
    /// from the gains it had when they just changed.
    fn mix_into(&mut self, mixes: &mut [QuadFrame; 3]) {
        let ramps = std::mem::take(&mut self.ramp);
        if !self.enabled {
            return;
        }
        let inputs = [self.frame.map(|[left, _]| left as f32), self.frame.map(|[_, right]| right as f32)];
        for ((mix, gains), from) in mixes.iter_mut().zip(self.gain).zip(ramps) {
            for (channel, out) in mix.iter_mut().enumerate() {
                let (input, gain) = (&inputs[channel & 1], gains[channel]);
                match from.map(|from| from[channel]) {
                    // a gain fading to where it was stays put, the same to
                    // the last bit but for the sign of a zero, which adds
                    // nothing either way
                    Some(from) if from != gain => fade(out, input, from, gain),
                    _ if gain == 0.0 => {}
                    _ => add(out, input, gain),
                }
            }
        }
    }
}

/// the sample a fraction of the way from one input to the next, both
/// channels or, without STEREO, the first for both.
fn lerp<const STEREO: bool>(x0: [i16; 2], x1: [i16; 2], fraction: f64) -> [i16; 2] {
    let between = |a: i16, b: i16| (a as f64 + (b as f64 - a as f64) * fraction) as i16;
    let left = between(x0[0], x1[0]);
    [left, if STEREO { between(x0[1], x1[1]) } else { left }]
}

/// one sample of a PCM buffer read from memory by itself, both channels.
fn read_pcm(memory: &mut impl SampleMemory, buffer: &Buffer, index: u32) -> [i16; 2] {
    let channels = if buffer.stereo { 2 } else { 1 };
    let at = buffer.address & !3;
    if buffer.format == Format::Pcm8 {
        return match memory.physical(at + index * channels, channels) {
            Some(data) => [data[0] as i8 as i16 * 256, data[data.len() - 1] as i8 as i16 * 256],
            None => [0; 2],
        };
    }
    match memory.physical(at + index * channels * 2, channels * 2) {
        Some(data) => [
            i16::from_le_bytes([data[0], data[1]]),
            i16::from_le_bytes([data[data.len() - 2], data[data.len() - 1]]),
        ],
        None => [0; 2],
    }
}

/// adds samples to a channel of a mix at a steady gain.
fn add(out: &mut [i32; FRAME_SAMPLES], input: &[f32; FRAME_SAMPLES], gain: f32) {
    if gain.abs() <= QUICK_GAIN {
        for (out, sample) in out.iter_mut().zip(input) {
            // SAFETY: a sample is 32768 at most either way, so the product
            // is a number within 2^29, which truncates the same as the
            // saturating cast does, only four at a time where that goes one
            *out += unsafe { (gain * sample).to_int_unchecked::<i32>() };
        }
    } else {
        for (out, sample) in out.iter_mut().zip(input) {
            *out += (gain * sample) as i32;
        }
    }
}

/// adds samples to a channel of a mix at a gain going from one value to
/// another over the frame.
fn fade(out: &mut [i32; FRAME_SAMPLES], input: &[f32; FRAME_SAMPLES], from: f32, to: f32) {
    let change = to - from;
    if from.abs() <= QUICK_GAIN && to.abs() <= QUICK_GAIN {
        for ((out, sample), progress) in out.iter_mut().zip(input).zip(PROGRESS) {
            // SAFETY: the change is within two quick gains and the gain on
            // the way within three, so as in add the product is a number
            // well within i32
            *out += unsafe { ((from + change * progress) * sample).to_int_unchecked::<i32>() };
        }
    } else {
        for ((out, sample), progress) in out.iter_mut().zip(input).zip(PROGRESS) {
            *out += ((from + change * progress) * sample) as i32;
        }
    }
}

/// GC ADPCM, frames of 8 bytes holding a header and 14 samples of 4 bits,
/// each predicted from the two before it, decoded into samples.
fn decode_adpcm(data: &[u8], count: usize, coefficients: &[i16; 16], history: &mut [i16; 2], samples: &mut Vec<i16>) {
    samples.clear();
    samples.reserve(count);
    let [mut yn1, mut yn2] = history.map(|h| h as i32);
    for frame in data.chunks(8) {
        let header = frame[0];
        let scale = 1 << (header & 0xF);
        let predictor = ((header >> 4) & 7) as usize;
        let (c1, c2) = (coefficients[predictor * 2] as i32, coefficients[predictor * 2 + 1] as i32);
        let mut decode = |nibble: u8| {
            // the nibble is signed
            let xn = (((nibble as i32) << 28) >> 28) * scale;
            let value = (((xn << 11) + 0x400 + c1 * yn1 + c2 * yn2) >> 11).clamp(-32768, 32767);
            yn2 = yn1;
            yn1 = value;
            value as i16
        };
        // a whole frame goes without counting each sample
        if frame.len() == 8 && count - samples.len() >= 14 {
            let mut decoded = [0; 14];
            for (pair, &byte) in decoded.as_chunks_mut().0.iter_mut().zip(&frame[1..]) {
                *pair = [decode(byte >> 4), decode(byte & 0xF)];
            }
            samples.extend_from_slice(&decoded);
            continue;
        }
        for &byte in &frame[1..] {
            for nibble in [byte >> 4, byte & 0xF] {
                if samples.len() == count {
                    *history = [yn1 as i16, yn2 as i16];
                    return;
                }
                samples.push(decode(nibble));
            }
        }
    }
    *history = [yn1 as i16, yn2 as i16];
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

        let mut mixes = [[[0; FRAME_SAMPLES]; 4]; 3];
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
            voice.mix_into(&mut mixes);
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
        voice.ran_out_at_frame_end = false;
    }
    if flags & dirty::ENABLE != 0 {
        voice.enabled = memory.read8(base + config::ENABLE) != 0;
        voice.ran_out_at_frame_end = false;
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
            length: read_dsp32(memory, base + config::LENGTH),
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

    /// a buffer that ends on a frame's last sample is followed by the next
    /// with no silent frame between, a stream of them would stutter.
    #[test]
    fn queued_buffers_play_on_without_a_gap() {
        let mut memory = Samples(1000i16.to_le_bytes().repeat(0x8000));
        let mut voice = voice_with(&[(1, 320), (2, 320)]);
        voice.play_frame(&mut memory); // starts buffer 1
        for frame in 0..4 {
            voice.play_frame(&mut memory);
            assert!(voice.frame.iter().all(|s| s[0] == 1000), "frame {frame} has a gap");
        }
        assert_eq!((voice.current_buffer_id, voice.remaining), (2, 0));
        voice.play_frame(&mut memory);
        assert!(!voice.enabled, "and still switches off as soon as it runs out");
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
        let bytes: Vec<u8> = (0..400i16).flat_map(|i| i.wrapping_mul(100).to_le_bytes()).collect();
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
        let mut samples = vec![7; 20];
        decode_adpcm(&frame, 4, &coefficients, &mut history, &mut samples);
        assert_eq!(samples, [101, 103, 106, 106]);
        assert_eq!(history, [106, 106]);
    }

    /// an ADPCM buffer comes from the last decoding only when that started
    /// from the same bytes, coefficients and history, and ran as long.
    #[test]
    fn adpcm_is_decoded_again_unless_nothing_changed() {
        let mut data: Vec<u8> = (0..32u8).map(|i| i.wrapping_mul(37)).collect();
        // the first frame predicts from the history it starts with
        data[0] = 0x12;
        let mut voice = Voice::default();
        voice.coefficients[2] = 1 << 11;
        voice.coefficients[3] = -(1 << 9);
        let steps: [(usize, [i16; 2], bool); 7] = [
            (20, [0, 0], false),
            (20, [0, 0], false),
            (15, [0, 0], false),
            (15, [5, -5], false),
            (15, [5, -5], true),
            (28, [5, -5], false),
            (28, [5, -5], false),
        ];
        for (step, (count, history, coefficients_changed)) in steps.into_iter().enumerate() {
            if coefficients_changed {
                voice.coefficients[4] = 3000;
            }
            data[9] ^= (step == 6) as u8;
            voice.history = history;
            voice.decode(&data, count);
            let (mut expected, mut end) = (Vec::new(), history);
            decode_adpcm(&data, count, &voice.coefficients, &mut end, &mut expected);
            assert_eq!((&voice.decoded, voice.history), (&expected, end), "step {step}");
        }
    }

    /// ADPCM decodes to the bit as it did a sample at a time, frames whole
    /// or cut short, however many samples are asked for.
    #[test]
    fn adpcm_decodes_as_the_sample_at_a_time_reference() {
        let mut random = Random(0xD1B5_4A32_D192_ED03);
        let mut samples = Vec::new();
        for case in 0..3000 {
            let data: Vec<u8> = (0..random.below(80)).map(|_| random.next() as u8).collect();
            let count = random.below(150) as usize;
            let coefficients: [i16; 16] = std::array::from_fn(|_| random.next() as i16 >> (3 + random.below(3)));
            let start = [random.next() as i16, random.next() as i16];
            let (mut history, mut expected_history) = (start, start);
            decode_adpcm(&data, count, &coefficients, &mut history, &mut samples);
            let expected = reference::decode_adpcm(&data, count, &coefficients, &mut expected_history);
            assert_eq!((&samples, history), (&expected, expected_history), "case {case}");
        }
    }

    /// an embedded buffer plays as long as it says, a sound of a second or
    /// two is common.
    #[test]
    fn an_embedded_buffer_keeps_its_length() {
        use crate::memory::{MemoryState, Permission};
        use zakuro_common::memory_map::{DSP_RAM_PADDR, DSP_RAM_SIZE, DSP_RAM_VADDR};
        let mut system = crate::System::new(crate::Config::default());
        let memory = &mut system.memory;
        memory.map(DSP_RAM_VADDR, DSP_RAM_PADDR, DSP_RAM_SIZE, Permission::RW, MemoryState::Static);
        let base = DSP_RAM_VADDR + 0x4_0000;
        write_dsp32(memory, base + config::ADDRESS, 0x2000_0000);
        write_dsp32(memory, base + config::LENGTH, 69_057);
        memory.write32(base + config::DIRTY, dirty::EMBEDDED_BUFFER);
        let mut voice = Voice::default();
        parse_config(&mut voice, memory, base, base + 0x1000);
        assert_eq!(voice.queue[0].length, 69_057);
    }

    #[test]
    fn gains_fade_in_over_a_frame_once_changed() {
        let mut voice = Voice { enabled: true, ..Voice::default() };
        voice.frame = [[1000, 1000]; FRAME_SAMPLES];
        voice.ramp[0] = Some([0.0; 4]);
        voice.gain[0] = [1.0; 4];
        let mut mixes = [[[0; FRAME_SAMPLES]; 4]; 3];
        voice.mix_into(&mut mixes);
        let sample = |mix: &QuadFrame, at: usize| mix.iter().map(|channel| channel[at]).collect::<Vec<_>>();
        assert_eq!(sample(&mixes[0], 0), [0; 4]);
        assert_eq!(sample(&mixes[0], FRAME_SAMPLES - 1), [1000; 4]);
        let mut again = [[[0; FRAME_SAMPLES]; 4]; 3];
        voice.mix_into(&mut again);
        assert_eq!(sample(&again[0], 0), [1000; 4], "the fade happens once");
    }

    /// the voices as they played and mixed a sample at a time, which the
    /// faster ways have to match to the bit.
    mod reference {
        use super::super::*;

        fn input(voice: &Voice, memory: &mut impl SampleMemory, index: u32) -> [i16; 2] {
            let Some(buffer) = &voice.playing else { return [0; 2] };
            let index = index.min(buffer.length.saturating_sub(1));
            let channels = if buffer.stereo { 2 } else { 1 };
            let at = buffer.address & !3;
            let [left, right] = match buffer.format {
                Format::Adpcm => {
                    let sample = voice.decoded.get(index as usize).copied().unwrap_or(0);
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

        fn sample(voice: &Voice, memory: &mut impl SampleMemory) -> [i16; 2] {
            let at = voice.length - voice.remaining;
            let x0 = input(voice, memory, at);
            if voice.interpolation == Interpolation::None || voice.fraction == 0.0 {
                return x0;
            }
            let x1 = input(voice, memory, at + 1);
            let between = |a: i16, b: i16| (a as f64 + (b as f64 - a as f64) * voice.fraction) as i16;
            [between(x0[0], x1[0]), between(x0[1], x1[1])]
        }

        pub fn decode_adpcm(data: &[u8], count: usize, coefficients: &[i16; 16], history: &mut [i16; 2]) -> Vec<i16> {
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

        fn dequeue(voice: &mut Voice, memory: &mut impl SampleMemory) -> bool {
            let Some(index) = (0..voice.queue.len()).min_by_key(|&i| voice.queue[i].id) else {
                return false;
            };
            let buffer = voice.queue[index];
            if buffer.looping {
                voice.queue[index].has_played = true;
            } else {
                voice.queue.remove(index);
            }
            let start = if buffer.has_played { 0 } else { buffer.play_position.min(buffer.length) };
            voice.length = buffer.length;
            voice.remaining = buffer.length - start;
            voice.position = start;
            voice.fraction = 0.0;
            voice.current_buffer_id = buffer.id;
            voice.last_buffer_id = 0;
            voice.buffer_update = buffer.from_queue && !buffer.has_played;
            if let Some(history) = buffer.adpcm {
                voice.history = history;
            }
            voice.decoded.clear();
            if buffer.format == Format::Adpcm {
                let bytes = buffer.length.div_ceil(14) * 8;
                if let Some(data) = memory.physical(buffer.address & !3, bytes) {
                    voice.decoded = decode_adpcm(data, buffer.length as usize, &voice.coefficients, &mut voice.history);
                }
            }
            voice.playing = Some(buffer);
            true
        }

        pub fn play_frame(voice: &mut Voice, memory: &mut impl SampleMemory) {
            voice.frame = [[0; 2]; FRAME_SAMPLES];
            let follows_on = std::mem::take(&mut voice.ran_out_at_frame_end) && !voice.queue.is_empty();
            if voice.remaining == 0 && !follows_on {
                if dequeue(voice, memory) {
                    return;
                }
                voice.enabled = false;
                voice.buffer_update = true;
                voice.last_buffer_id = voice.current_buffer_id;
                voice.current_buffer_id = 0;
                voice.position = 0;
                return;
            }

            voice.position = voice.length - voice.remaining;
            let mut output = 0;
            while output < FRAME_SAMPLES {
                if voice.remaining == 0 && !dequeue(voice, memory) {
                    break;
                }
                voice.frame[output] = sample(voice, memory);
                output += 1;
                let input = voice.fraction + voice.rate;
                let consumed = (input.floor() as u32).min(voice.remaining);
                voice.fraction = (input - consumed as f64).max(0.0);
                voice.remaining -= consumed;
            }
            voice.ran_out_at_frame_end = voice.remaining == 0 && output == FRAME_SAMPLES;
            voice.filters.process(&mut voice.frame[..output]);
        }

        /// a mix with the channels of each sample together, as it was.
        pub type Interleaved = [[i32; 4]; FRAME_SAMPLES];

        pub fn interleave(mix: &QuadFrame) -> Interleaved {
            std::array::from_fn(|sample| std::array::from_fn(|channel| mix[channel][sample]))
        }

        pub fn mix_into(voice: &mut Voice, mix: &mut Interleaved, index: usize) {
            let gains = voice.gain[index];
            let from = voice.ramp[index].take();
            if !voice.enabled {
                return;
            }
            for (i, (out, sample)) in mix.iter_mut().zip(&voice.frame).enumerate() {
                let progress = i as f32 / (FRAME_SAMPLES - 1) as f32;
                let gain = |c: usize| from.map_or(gains[c], |from| from[c] + (gains[c] - from[c]) * progress);
                out[0] += (gain(0) * sample[0] as f32) as i32;
                out[1] += (gain(1) * sample[1] as f32) as i32;
                out[2] += (gain(2) * sample[0] as f32) as i32;
                out[3] += (gain(3) * sample[1] as f32) as i32;
            }
        }
    }

    /// a small xorshift generator, the same numbers every run.
    struct Random(u64);

    impl Random {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }

        fn chance(&mut self, percent: u64) -> bool {
            self.below(100) < percent
        }

        fn pick<T: Copy>(&mut self, items: &[T]) -> T {
            items[self.below(items.len() as u64) as usize]
        }

        /// somewhere in [0, 1).
        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// bytes of memory the random buffers play from, at BASE.
    const MEMORY_SIZE: u32 = 0x4000;

    /// a buffer of any kind a title could queue, now and then not all of it
    /// in memory or none of it.
    fn random_buffer(random: &mut Random) -> Buffer {
        let length = match random.below(6) {
            0 => random.pick(&[0, 1, 2, 3]),
            1 | 2 => random.below(60) as u32,
            _ => random.below(3000) as u32,
        };
        let address = match random.below(10) {
            0 => BASE - 0x100,
            1 => BASE + MEMORY_SIZE - random.below(64) as u32,
            _ => BASE + random.below(MEMORY_SIZE as u64 / 2) as u32,
        };
        Buffer {
            address,
            length,
            format: random.pick(&[Format::Pcm8, Format::Pcm16, Format::Adpcm]),
            stereo: random.chance(50),
            looping: random.chance(25),
            id: random.below(8) as u16,
            adpcm: random.chance(50).then(|| [random.next() as i16, random.next() as i16]),
            from_queue: random.chance(50),
            play_position: if random.chance(70) { 0 } else { random.below(length as u64 + 20) as u32 },
            has_played: false,
        }
    }

    /// a voice set up the way a title could have it, part way through
    /// whatever it plays.
    fn random_voice(random: &mut Random) -> Voice {
        let rates = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 32728.0 / 32000.0, 22050.0 / 32728.0, 1e-3, 7.75, 300.0, 3e9, 1e12];
        let mut voice = Voice {
            enabled: true,
            rate: if random.chance(70) { random.pick(&rates) } else { random.unit() * 4.0 + 1e-9 },
            interpolation: random.pick(&[Interpolation::Polyphase, Interpolation::Linear, Interpolation::None]),
            fraction: match random.below(6) {
                0 => 0.0,
                1 => random.pick(&[1.0, 1.5, 3.25, 1.0 - f64::EPSILON / 2.0]),
                _ => random.unit(),
            },
            ran_out_at_frame_end: random.chance(20),
            ..Voice::default()
        };
        // as loud as a title's coefficients get, louder would overflow the
        // decoder's sum in a test build
        for coefficient in &mut voice.coefficients {
            *coefficient = random.next() as i16 >> (3 + random.below(3));
        }
        voice.history = [random.next() as i16, random.next() as i16];
        if random.chance(20) {
            voice.filters.simple = Some(SimpleFilter { b0: (random.next() as i16) as i32, a1: (random.next() as i16 >> 2) as i32, y1: [0; 2] });
        }
        if random.chance(20) {
            voice.filters.biquad = Some(BiquadFilter {
                b: [random.next() as i16 as i32 >> 2, random.next() as i16 as i32 >> 3, random.next() as i16 as i32 >> 3],
                a: [random.next() as i16 as i32 >> 3, random.next() as i16 as i32 >> 4],
                ..BiquadFilter::default()
            });
        }
        for _ in 0..random.below(5) {
            voice.queue.push(random_buffer(random));
        }
        // a buffer lengthened before it started plays silence
        if random.chance(5) {
            voice.length = random.below(400) as u32;
            voice.remaining = voice.length;
        }
        voice
    }

    /// asserts two voices are in the same state, to the bit.
    fn assert_same(voice: &Voice, reference: &Voice, context: &str) {
        assert_eq!(voice.frame, reference.frame, "{context}: samples");
        assert_eq!(voice.fraction.to_bits(), reference.fraction.to_bits(), "{context}: fraction");
        assert_eq!(
            (voice.enabled, voice.remaining, voice.length, voice.position, voice.ran_out_at_frame_end, voice.history),
            (
                reference.enabled,
                reference.remaining,
                reference.length,
                reference.position,
                reference.ran_out_at_frame_end,
                reference.history,
            ),
            "{context}"
        );
        assert_eq!(
            (voice.current_buffer_id, voice.last_buffer_id, voice.buffer_update),
            (reference.current_buffer_id, reference.last_buffer_id, reference.buffer_update),
            "{context}: report"
        );
        assert_eq!(format!("{:?}", voice.playing), format!("{:?}", reference.playing), "{context}: playing");
        assert_eq!(format!("{:?}", voice.queue), format!("{:?}", reference.queue), "{context}: queue");
        assert_eq!(format!("{:?}", voice.filters), format!("{:?}", reference.filters), "{context}: filters");
        assert_eq!(voice.decoded, reference.decoded, "{context}: decoded");
    }

    /// voices play to the bit what they did a sample at a time, whatever
    /// the format, rate, interpolation and buffers, and the title changing
    /// things between frames.
    #[test]
    fn playing_matches_the_sample_at_a_time_reference() {
        let mut random = Random(0x9E37_79B9_7F4A_7C15);
        for case in 0..3000 {
            let mut memory = Samples((0..MEMORY_SIZE).map(|_| random.next() as u8).collect());
            let mut voice = random_voice(&mut random);
            let mut reference = voice.clone();
            for frame in 0..10 {
                if random.chance(20) {
                    let buffer = random_buffer(&mut random);
                    voice.queue.push(buffer);
                    reference.queue.push(buffer);
                }
                // the title lengthening the buffer playing, as parse_config does
                if random.chance(10) {
                    let length = random.below(3000) as u32;
                    for voice in [&mut voice, &mut reference] {
                        let played = voice.length - voice.remaining;
                        voice.length = length.max(played);
                        voice.remaining = voice.length - played;
                        if let Some(playing) = &mut voice.playing {
                            playing.length = voice.length;
                        }
                    }
                }
                // a new rate part way through a buffer, whole ones too
                if random.chance(10) {
                    let rate = if random.chance(50) { random.pick(&[1.0, 2.0, 3.0]) } else { random.unit() * 3.0 + 0.01 };
                    voice.rate = rate;
                    reference.rate = rate;
                }
                // the title writing over the samples playing, or changing
                // the coefficients
                if let Some(playing) = voice.playing.filter(|_| random.chance(15)) {
                    let at = (playing.address & !3).wrapping_sub(BASE) as u64 + random.below(playing.length as u64 / 2 + 1);
                    if let Some(byte) = memory.0.get_mut(at as usize) {
                        *byte = random.next() as u8;
                    }
                }
                if random.chance(5) {
                    let at = random.below(16) as usize;
                    let coefficient = random.next() as i16 >> 4;
                    voice.coefficients[at] = coefficient;
                    reference.coefficients[at] = coefficient;
                }
                if !voice.enabled && random.chance(50) {
                    voice.enabled = true;
                    reference.enabled = true;
                }
                if voice.enabled {
                    voice.play_frame(&mut memory);
                    reference::play_frame(&mut reference, &mut memory);
                }
                assert_same(&voice, &reference, &format!("case {case}, frame {frame}"));
            }
        }
    }

    /// a buffer running off the end of the DSP's memory goes on into the
    /// memory after it, read a sample at a time.
    #[test]
    fn a_buffer_across_two_memories_plays_from_both() {
        use zakuro_common::memory_map::{DSP_RAM_PADDR, DSP_RAM_SIZE};
        let mut memory = Memory::new(false, 64 << 20);
        let end = DSP_RAM_PADDR + DSP_RAM_SIZE;
        for (i, address) in (end - 64..end + 64).enumerate() {
            memory.phys.host_slice_mut(address, 1).unwrap()[0] = i as u8 | 1;
        }
        let mut voice = voice_with(&[]);
        voice.queue.push(Buffer { address: end - 32, stereo: true, ..buffer(1, 32) });
        voice.rate = 0.75;
        let mut reference = voice.clone();
        for frame in 0..3 {
            voice.play_frame(&mut memory);
            reference::play_frame(&mut reference, &mut memory);
            assert_same(&voice, &reference, &format!("frame {frame}"));
            if frame == 1 {
                assert_ne!(voice.frame[30], [0; 2], "the samples past the end are there");
            }
        }
    }

    /// what the voices add to the mixes is to the bit what they added a
    /// sample at a time, steady, fading, fading back to where they were,
    /// silent and at gains far too loud.
    #[test]
    fn mixing_matches_the_sample_at_a_time_reference() {
        let mut random = Random(0x2545_F491_4F6C_DD1D);
        let gains = [
            0.0,
            -0.0,
            1.0,
            0.5,
            -0.75,
            1e-40,
            3.0,
            QUICK_GAIN,
            -QUICK_GAIN,
            QUICK_GAIN + 0.002,
            1e9,
            3e38,
            -3e38,
            f32::MAX,
            f32::MIN_POSITIVE,
        ];
        for case in 0..4000 {
            let loud = random.chance(30);
            let gain = |random: &mut Random| {
                if loud {
                    random.pick(&gains)
                } else {
                    (random.unit() * 4.0 - 2.0) as f32
                }
            };
            let mut voice = Voice { enabled: random.chance(90), ..Voice::default() };
            let quiet = random.chance(10);
            for sample in &mut voice.frame {
                *sample = if quiet { [0; 2] } else { [random.next() as i16, random.next() as i16] };
            }
            for mix in 0..3 {
                voice.gain[mix] = match random.below(4) {
                    0 => [0.0; 4],
                    1 => [gain(&mut random); 4],
                    _ => std::array::from_fn(|_| gain(&mut random)),
                };
                voice.ramp[mix] = match random.below(5) {
                    0 | 1 => None,
                    2 => Some(voice.gain[mix]),
                    // the same gains, a zero of the other sign
                    3 => Some(voice.gain[mix].map(|g| if g == 0.0 { -g } else { g })),
                    _ => Some(std::array::from_fn(|_| gain(&mut random))),
                };
            }
            let mut reference = voice.clone();
            // far too loud a voice is the only one, it would overflow the
            // sum in a test build
            let start = if loud { 0 } else { 1 << 28 };
            let mut mixes = [[[0; FRAME_SAMPLES]; 4]; 3];
            for value in mixes.iter_mut().flatten().flatten() {
                *value = random.below(start as u64 * 2 + 1) as i32 - start;
            }
            let mut expected = mixes.map(|mix| reference::interleave(&mix));
            voice.mix_into(&mut mixes);
            for (mix, frame) in expected.iter_mut().enumerate() {
                reference::mix_into(&mut reference, frame, mix);
            }
            assert_eq!(mixes.map(|mix| reference::interleave(&mix)), expected, "case {case}");
            assert_eq!(voice.ramp, [None; 3], "case {case}: the fades are done");
        }
    }

    #[test]
    fn fades_go_through_a_frame_as_the_division_does() {
        for (i, progress) in PROGRESS.iter().enumerate() {
            let at = std::hint::black_box(i as f32) / std::hint::black_box((FRAME_SAMPLES - 1) as f32);
            assert_eq!(progress.to_bits(), at.to_bits());
        }
    }
}
