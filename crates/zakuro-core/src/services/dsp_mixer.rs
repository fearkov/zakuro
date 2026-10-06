//! the audio DSP's final mixer, which brings the voices' intermediate mixes
//! together into what the speakers play.

use zakuro_common::memory_map::PAGE_BITS;
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
        Mixer { volume: [0.0; 3], aux: [false; 2], output: Output::Stereo, mixes: [[[0; FRAME_SAMPLES]; 4]; 3] }
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
                read_mix(memory, from + offset, &mut self.mixes[aux + 1]);
                write_mix(memory, to + offset, &mixes[aux + 1]);
            } else {
                self.mixes[aux + 1] = mixes[aux + 1];
            }
        }

        let mut frame = [[0i16; 2]; FRAME_SAMPLES];
        for (volume, mix) in self.volume.iter().zip(&self.mixes) {
            // a mix at no volume adds nothing
            if *volume != 0.0 {
                downmix(&mut frame, mix, *volume, self.output);
            }
        }
        write_frame(memory, layout.final_samples + write * REGION_STRIDE, &frame);
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

/// whether every page of a range is mapped. the DSP's memory always is, to
/// read and write, and copying a range of it at once does what going
/// through it a word at a time does, as there is no GPU drawing to bring
/// down while the DSP runs. anywhere else goes a word at a time, missing
/// each word as it did.
fn mapped(memory: &Memory, address: u32, size: u32) -> bool {
    pages(address, size).all(|page| memory.is_readable(page, 1))
}

/// the same for writing, a page can be there to read and not to write.
fn writable(memory: &Memory, address: u32, size: u32) -> bool {
    pages(address, size).all(|page| memory.is_writable(page, 1))
}

/// where each page a range touches starts.
fn pages(address: u32, size: u32) -> impl Iterator<Item = u32> {
    (address >> PAGE_BITS..=(address + size - 1) >> PAGE_BITS).map(|page| page << PAGE_BITS)
}

/// reads what the title made of a mix, laid out channel by channel as
/// ours is.
fn read_mix(memory: &mut Memory, address: u32, mix: &mut QuadFrame) {
    if !mapped(memory, address, AUX_MIX_SIZE) {
        for sample in 0..FRAME_SAMPLES {
            for (channel, values) in mix.iter_mut().enumerate() {
                values[sample] = memory.read32(address + sample_at(channel, sample)) as i32;
            }
        }
        return;
    }
    let mut bytes = [0; AUX_MIX_SIZE as usize];
    memory.read_bytes(address, &mut bytes);
    for (values, channel) in mix.iter_mut().zip(bytes.as_chunks::<{ FRAME_SAMPLES * 4 }>().0) {
        for (value, word) in values.iter_mut().zip(channel.as_chunks().0) {
            *value = i32::from_le_bytes(*word);
        }
    }
}

/// writes a mix out for the title to process.
fn write_mix(memory: &mut Memory, address: u32, mix: &QuadFrame) {
    if !writable(memory, address, AUX_MIX_SIZE) {
        for sample in 0..FRAME_SAMPLES {
            for (channel, values) in mix.iter().enumerate() {
                memory.write32(address + sample_at(channel, sample), values[sample] as u32);
            }
        }
        return;
    }
    let mut bytes = [0; AUX_MIX_SIZE as usize];
    for (channel, values) in bytes.as_chunks_mut::<{ FRAME_SAMPLES * 4 }>().0.iter_mut().zip(mix) {
        for (word, value) in channel.as_chunks_mut().0.iter_mut().zip(values) {
            *word = value.to_le_bytes();
        }
    }
    memory.write_bytes(address, &bytes);
}

/// writes what the speakers play where the title finds it.
fn write_frame(memory: &mut Memory, address: u32, frame: &Frame) {
    if !writable(memory, address, FRAME_SAMPLES as u32 * 4) {
        for (i, pair) in frame.iter().enumerate() {
            memory.write16(address + i as u32 * 4, pair[0] as u16);
            memory.write16(address + i as u32 * 4 + 2, pair[1] as u16);
        }
        return;
    }
    let mut bytes = [0; FRAME_SAMPLES * 4];
    for (out, [left, right]) in bytes.as_chunks_mut().0.iter_mut().zip(frame) {
        let ([a, b], [c, d]) = (left.to_le_bytes(), right.to_le_bytes());
        *out = [a, b, c, d];
    }
    memory.write_bytes(address, &bytes);
}

/// adds a four channel mix to the stereo output at a volume.
fn downmix(frame: &mut Frame, mix: &QuadFrame, volume: f32, output: Output) {
    let clamp = |value: f32| (value as i32).clamp(-32768, 32767);
    for (sample, out) in frame.iter_mut().enumerate() {
        let [a, b, c, d] = std::array::from_fn(|channel| mix[channel][sample] as f32 * volume);
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
        let mix = [1000, 2000, 300, 400].map(|value| [value; FRAME_SAMPLES]);
        downmix(&mut frame, &mix, 0.5, Output::Stereo);
        assert_eq!(frame[0], [650, 1200]);
        downmix(&mut frame, &mix, 0.5, Output::Mono);
        assert_eq!(frame[0], [650 + 925, 1200 + 925]);
    }

    #[test]
    fn the_output_clips_instead_of_wrapping() {
        let mut frame = [[30000i16, -30000]; FRAME_SAMPLES];
        let mix = [10000, -10000, 0, 0].map(|value| [value; FRAME_SAMPLES]);
        downmix(&mut frame, &mix, 1.0, Output::Stereo);
        assert_eq!(frame[0], [32767, -32768]);
    }

    /// the mixer as it was, a word of memory and a sample at a time, with
    /// the channels of each sample together.
    mod reference {
        use super::super::*;

        pub type Interleaved = [[i32; 4]; FRAME_SAMPLES];

        pub fn tick(
            mixer: &mut Mixer,
            kept: &mut [Interleaved; 3],
            memory: &mut Memory,
            layout: Layout,
            read: u32,
            mixes: [Interleaved; 3],
        ) -> Frame {
            let write = 1 - read;
            mixer.parse_config(memory, layout.configuration + read * REGION_STRIDE);
            let (from, to) =
                (layout.intermediate_samples + read * REGION_STRIDE, layout.intermediate_samples + write * REGION_STRIDE);
            kept[0] = mixes[0];
            for aux in 0..2 {
                let offset = aux as u32 * AUX_MIX_SIZE;
                if mixer.aux[aux] {
                    for (sample, values) in kept[aux + 1].iter_mut().enumerate() {
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
                    kept[aux + 1] = mixes[aux + 1];
                }
            }
            let mut frame = [[0i16; 2]; FRAME_SAMPLES];
            for (volume, mix) in mixer.volume.iter().zip(kept.iter()) {
                downmix(&mut frame, mix, *volume, mixer.output);
            }
            let samples = layout.final_samples + write * REGION_STRIDE;
            for (i, pair) in frame.iter().enumerate() {
                memory.write16(samples + i as u32 * 4, pair[0] as u16);
                memory.write16(samples + i as u32 * 4 + 2, pair[1] as u16);
            }
            frame
        }

        fn downmix(frame: &mut Frame, mix: &Interleaved, volume: f32, output: Output) {
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

        fn pick<T: Copy>(&mut self, items: &[T]) -> T {
            items[self.below(items.len() as u64) as usize]
        }
    }

    /// the mixer comes out to the bit as it did a word and a sample at a
    /// time: the speakers, what the title gets to process and what it finds
    /// played, with the DSP's memory there or, failing that, every access
    /// missing as it did.
    #[test]
    fn mixing_down_matches_the_word_at_a_time_reference() {
        use crate::memory::{MemoryState, Permission};
        use zakuro_common::memory_map::{DSP_RAM_PADDR, DSP_RAM_SIZE, DSP_RAM_VADDR};
        let at = |address: u32| DSP_RAM_VADDR + (address << 1) + 0x40000;
        let layout = Layout { configuration: at(0x9430), final_samples: at(0x8540), intermediate_samples: at(0x9492) };
        let volumes = [0.0f32, -0.0, 1.0, 0.5, 0.25, 2.0, 1e30, f32::INFINITY, f32::NAN, 3e-39, -1.0, -0.5, -3e38];
        let mut random = Random(0x5851_F42D_4C95_7F2D);
        // the DSP's memory as mapped, by offset, size and permission: all of
        // it, none, all but the page the first aux mix runs into, and all of
        // it read only
        let whole = [(0, DSP_RAM_SIZE, Permission::RW)];
        let gap = [(0, 0x5_3000, Permission::RW), (0x5_4000, DSP_RAM_SIZE - 0x5_4000, Permission::RW)];
        let read_only = [(0, DSP_RAM_SIZE, Permission::READ)];
        let mappings: [&[(u32, u32, Permission)]; 6] = [&whole, &whole, &whole, &[], &gap, &read_only];
        for (case, mapping) in mappings.into_iter().enumerate() {
            let mut memories = [Memory::new(false, 64 << 20), Memory::new(false, 64 << 20)];
            for memory in &mut memories {
                for &(offset, size, permission) in mapping {
                    memory.map(DSP_RAM_VADDR + offset, DSP_RAM_PADDR + offset, size, permission, MemoryState::Static);
                }
            }
            let (mut mixer, mut expected) = (Mixer::default(), Mixer::default());
            let mut kept = [[[0; 4]; FRAME_SAMPLES]; 3];
            for frame in 0..40 {
                let read = random.below(2) as u32;
                // what the title changed, and what it made of the last mixes
                let config = layout.configuration + read * REGION_STRIDE;
                let flags = random.next() as u32 & (0x0700_0300 | dirty::MASTER_VOLUME);
                let mut writes = vec![(config + config::DIRTY, flags.to_le_bytes().to_vec())];
                for offset in [config::MASTER_VOLUME, config::AUX_RETURN_VOLUME, config::AUX_RETURN_VOLUME + 4] {
                    writes.push((config + offset, random.pick(&volumes).to_le_bytes().to_vec()));
                }
                for aux in 0..2 {
                    writes.push((config + config::AUX_BUS_ENABLE + aux * 2, vec![random.below(2) as u8, 0]));
                }
                writes.push((config + config::OUTPUT_FORMAT, vec![random.below(3) as u8, 0]));
                let processed: Vec<u8> =
                    (0..2 * AUX_MIX_SIZE / 4).flat_map(|_| (random.next() as i32 >> 12).to_le_bytes()).collect();
                writes.push((layout.intermediate_samples + read * REGION_STRIDE, processed));
                for memory in &mut memories {
                    for (address, bytes) in &writes {
                        memory.write_bytes(*address, bytes);
                    }
                }
                let loud = random.below(4) == 0;
                let mixes: [QuadFrame; 3] = std::array::from_fn(|_| {
                    std::array::from_fn(|_| {
                        std::array::from_fn(|_| if loud { random.next() as i32 } else { random.next() as i32 >> 12 })
                    })
                });
                let interleaved = mixes.map(|mix| {
                    std::array::from_fn(|sample| std::array::from_fn(|channel| mix[channel][sample]))
                });
                let [memory, reference_memory] = &mut memories;
                let played = mixer.tick(memory, layout, read, mixes);
                let wanted = reference::tick(&mut expected, &mut kept, reference_memory, layout, read, interleaved);
                assert_eq!(played, wanted, "case {case}, frame {frame}");
                let mut regions = [vec![0; 0x3_0000], vec![0; 0x3_0000]];
                for (memory, region) in memories.iter_mut().zip(&mut regions) {
                    memory.read_bytes(at(0x8000), region);
                }
                assert!(regions[0] == regions[1], "case {case}, frame {frame}: memory");
                assert_eq!(memories[0].fault_summary(), memories[1].fault_summary(), "case {case}, frame {frame}");
            }
        }
    }
}
