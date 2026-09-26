//! the audio DSP's final mixer, which brings the voices' intermediate mixes
//! together into what the speakers play.

use zakuro_cpu::Bus;

use super::dsp_voices::{Frame, QuadFrame, FRAME_SAMPLES, REGION_STRIDE};
use crate::memory::Memory;

/// bits of the configuration's dirty word.
mod dirty {
    /// one bit per aux bus from here up.
    pub const AUX_BUS_ENABLE: u32 = 1 << 8;
    pub const MASTER_VOLUME: u32 = 1 << 16;
    /// one bit per aux bus from here up.
    pub const AUX_RETURN_VOLUME: u32 = 1 << 24;
    pub const OUTPUT_FORMAT: u32 = 1 << 26;
}

/// offsets into the configuration.
mod config {
    pub const DIRTY: u32 = 0x00;
    pub const MASTER_VOLUME: u32 = 0x04;
    pub const AUX_RETURN_VOLUME: u32 = 0x08;
    pub const OUTPUT_FORMAT: u32 = 0x16;
    pub const AUX_BUS_ENABLE: u32 = 0x28;
}

/// bytes of one intermediate mix a title can process, 4 channels of 32-bit
/// samples.
const AUX_MIX_SIZE: u32 = 4 * FRAME_SAMPLES as u32 * 4;

/// addresses of the structures the mixer uses, in region 0.
#[derive(Debug, Clone, Copy)]
pub struct Layout {
    pub configuration: u32,
    pub final_samples: u32,
    pub intermediate_samples: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Output {
    Mono,
    Stereo,
}

#[derive(Debug, Clone)]
pub struct Mixer {
    /// how loud each intermediate mix comes out, the first being the master
    /// volume and the others the aux returns.
    volume: [f32; 3],
    /// whether mixes 1 and 2 go through the title first, for its effects.
    aux: [bool; 2],
    output: Output,
    /// the mixes as they reach the output.
    mixes: [QuadFrame; 3],
}

impl Default for Mixer {
    fn default() -> Self {
        Mixer { volume: [0.0; 3], aux: [false; 2], output: Output::Stereo, mixes: [[[0; 4]; FRAME_SAMPLES]; 3] }
    }
}

impl Mixer {
    /// mixes one frame from what the voices made, with read the region the
    /// title wrote last, and returns what the speakers play.
    pub fn tick(&mut self, memory: &mut Memory, layout: Layout, read: u32, mixes: [QuadFrame; 3]) -> Frame {
        let write = 1 - read;
        self.parse_config(memory, layout.configuration + read * REGION_STRIDE);

        // mixes 1 and 2 go out to the title, and what it made of the last
        // ones comes back
        let (from, to) = (layout.intermediate_samples + read * REGION_STRIDE, layout.intermediate_samples + write * REGION_STRIDE);
        self.mixes[0] = mixes[0];
        for aux in 0..2 {
            let offset = aux as u32 * AUX_MIX_SIZE;
            if self.aux[aux] {
                for (sample, values) in self.mixes[aux + 1].iter_mut().enumerate() {
                    for (channel, value) in values.iter_mut().enumerate() {
                        *value = memory.read32(from + offset + sample_at(channel, sample)) as i32;
                    }
                }
                for (sample, values) in mixes[aux + 1].iter().enumerate() {
                    for (channel, &value) in values.iter().enumerate() {
                        memory.write32(to + offset + sample_at(channel, sample), value as u32);
                    }
                }
            } else {
                self.mixes[aux + 1] = mixes[aux + 1];
            }
        }

        let mut frame = [[0i16; 2]; FRAME_SAMPLES];
        for (volume, mix) in self.volume.iter().zip(&self.mixes) {
            downmix(&mut frame, mix, *volume, self.output);
        }
        let samples = layout.final_samples + write * REGION_STRIDE;
        for (i, pair) in frame.iter().enumerate() {
            memory.write16(samples + i as u32 * 4, pair[0] as u16);
            memory.write16(samples + i as u32 * 4 + 2, pair[1] as u16);
        }
        frame
    }

    fn parse_config(&mut self, memory: &mut Memory, base: u32) {
        let flags = memory.read32(base + config::DIRTY);
        if flags == 0 {
            return;
        }
        let before = (self.volume, self.aux, self.output);
        let volume = |memory: &mut Memory, offset: u32| {
            let volume = f32::from_bits(memory.read32(base + offset));
            if volume.is_finite() { volume } else { 0.0 }
        };
        if flags & dirty::MASTER_VOLUME != 0 {
            self.volume[0] = volume(memory, config::MASTER_VOLUME);
        }
        for aux in 0..2 {
            if flags & (dirty::AUX_RETURN_VOLUME << aux) != 0 {
                self.volume[aux + 1] = volume(memory, config::AUX_RETURN_VOLUME + aux as u32 * 4);
            }
            if flags & (dirty::AUX_BUS_ENABLE << aux) != 0 {
                self.aux[aux] = memory.read16(base + config::AUX_BUS_ENABLE + aux as u32 * 2) != 0;
            }
        }
        if flags & dirty::OUTPUT_FORMAT != 0 {
            // surround comes out as stereo
            self.output = if memory.read16(base + config::OUTPUT_FORMAT) == 0 { Output::Mono } else { Output::Stereo };
        }
        if before != (self.volume, self.aux, self.output) {
            log::debug!(
                "dsp: mixer volumes {:?}, aux buses {:?}, {:?} output",
                self.volume, self.aux, self.output
            );
        }
        memory.write32(base + config::DIRTY, 0);
    }
}

/// where a sample of a channel sits in a mix the title processes, channel
/// by channel.
fn sample_at(channel: usize, sample: usize) -> u32 {
    ((channel * FRAME_SAMPLES + sample) * 4) as u32
}

/// adds a four channel mix to the stereo output at a volume.
fn downmix(frame: &mut Frame, mix: &QuadFrame, volume: f32, output: Output) {
    let clamp = |value: f32| (value as i32).clamp(-32768, 32767);
    for (out, sample) in frame.iter_mut().zip(mix) {
        let [a, b, c, d] = sample.map(|s| s as f32 * volume);
        let (left, right) = match output {
            Output::Mono => {
                let mono = clamp((a + b + c + d) / 2.0);
                (mono, mono)
            }
            Output::Stereo => (clamp(a + c), clamp(b + d)),
        };
        out[0] = (out[0] as i32 + left).clamp(-32768, 32767) as i16;
        out[1] = (out[1] as i32 + right).clamp(-32768, 32767) as i16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixes_come_down_to_stereo_at_their_volume() {
        let mut frame = [[0i16; 2]; FRAME_SAMPLES];
        let mix = [[1000, 2000, 300, 400]; FRAME_SAMPLES];
        downmix(&mut frame, &mix, 0.5, Output::Stereo);
        assert_eq!(frame[0], [650, 1200]);
        downmix(&mut frame, &mix, 0.5, Output::Mono);
        assert_eq!(frame[0], [650 + 925, 1200 + 925]);
    }

    #[test]
    fn the_output_clips_instead_of_wrapping() {
        let mut frame = [[30000i16, -30000]; FRAME_SAMPLES];
        let mix = [[10000, -10000, 0, 0]; FRAME_SAMPLES];
        downmix(&mut frame, &mix, 1.0, Output::Stereo);
        assert_eq!(frame[0], [32767, -32768]);
    }
}
