//! what the console's microphone hears, which the frontend gives: the
//! host's microphone, a sound from a file, noise, or nothing.

use std::sync::Arc;

/// where the microphone's sound comes from.
pub trait Microphone: Send {
    /// a title started or stopped sampling, a host's microphone may be open
    /// only meanwhile.
    fn sampling(&mut self, on: bool);
    /// what was heard since the last call, out.len() samples at rate a
    /// second, from -1 to 1, silence where nothing was.
    fn read(&mut self, rate: f64, out: &mut [f32]);
}

/// a sound played over and over while a title samples, from its start each
/// time sampling starts.
pub struct Recording {
    samples: Arc<[f32]>,
    /// its samples a second.
    rate: f64,
    /// where in it the next sample is read, in its samples.
    position: f64,
}

impl Recording {
    pub fn new(samples: Vec<f32>, rate: f64) -> Recording {
        Recording { samples: samples.into(), rate, position: 0.0 }
    }

    /// the sound of a WAV of 16-bit samples, its channels together.
    pub fn from_wav(bytes: &[u8]) -> Option<Recording> {
        let u16_at = |at: usize| Some(u16::from_le_bytes(bytes.get(at..at + 2)?.try_into().ok()?));
        let u32_at = |at: usize| Some(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?));
        if bytes.get(..4)? != b"RIFF" || bytes.get(8..12)? != b"WAVE" {
            return None;
        }
        let (mut channels, mut rate, mut data) = (0usize, 0u32, None);
        let mut at = 12;
        while at + 8 <= bytes.len() {
            let length = u32_at(at + 4)? as usize;
            let body = at + 8;
            match bytes.get(at..at + 4)? {
                b"fmt " => {
                    // PCM, 16 bits
                    if u16_at(body)? != 1 || u16_at(body + 14)? != 16 {
                        return None;
                    }
                    channels = u16_at(body + 2)? as usize;
                    rate = u32_at(body + 4)?;
                }
                b"data" => data = Some(bytes.get(body..(body + length).min(bytes.len()))?),
                _ => {}
            }
            // chunks are padded to an even length
            at = body + length + (length & 1);
        }
        let data = data?;
        if channels == 0 || rate == 0 {
            return None;
        }
        let samples = data
            .chunks_exact(2 * channels)
            .map(|frame| {
                let sum: f32 = frame.as_chunks::<2>().0.iter().map(|&sample| i16::from_le_bytes(sample) as f32).sum();
                sum / channels as f32 / 32768.0
            })
            .collect();
        Some(Recording::new(samples, rate as f64))
    }
}

impl Microphone for Recording {
    fn sampling(&mut self, on: bool) {
        if on {
            self.position = 0.0;
        }
    }

    fn read(&mut self, rate: f64, out: &mut [f32]) {
        let length = self.samples.len();
        if length == 0 || rate <= 0.0 {
            out.fill(0.0);
            return;
        }
        let step = self.rate / rate;
        for sample in out {
            let first = self.position as usize % length;
            let fraction = self.position.fract() as f32;
            let (a, b) = (self.samples[first], self.samples[(first + 1) % length]);
            *sample = a + (b - a) * fraction;
            self.position = (self.position + step) % length as f64;
        }
    }
}

/// white noise, which is how blowing into a microphone sounds to a game.
pub struct Noise {
    /// how loud, from 0 to 1.
    pub level: f32,
    seed: u32,
}

impl Noise {
    pub fn new(level: f32) -> Noise {
        Noise { level, seed: 1 }
    }

    /// the next of its samples.
    pub fn sample(&mut self) -> f32 {
        // a linear congruential generator, the same noise every run
        self.seed = self.seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((self.seed >> 8) as f32 / (1 << 23) as f32 - 1.0) * self.level
    }
}

impl Microphone for Noise {
    fn sampling(&mut self, _on: bool) {}

    fn read(&mut self, _rate: f64, out: &mut [f32]) {
        for sample in out {
            *sample = self.sample();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// a WAV of two channels comes out as one, at its rate, and a recording
    /// read at twice its rate falls between its samples.
    #[test]
    fn a_wav_reads_back_at_any_rate() {
        let mut wav = Vec::new();
        let frames: [[i16; 2]; 2] = [[16384, 16384], [-16384, 0]];
        let data: Vec<u8> = frames.iter().flatten().flat_map(|s| s.to_le_bytes()).collect();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        for value in [1u16, 2] {
            wav.extend_from_slice(&value.to_le_bytes());
        }
        wav.extend_from_slice(&8000u32.to_le_bytes());
        wav.extend_from_slice(&32000u32.to_le_bytes());
        for value in [4u16, 16] {
            wav.extend_from_slice(&value.to_le_bytes());
        }
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
        wav.extend_from_slice(&data);
        let mut recording = Recording::from_wav(&wav).expect("a WAV");
        assert_eq!(&*recording.samples, &[0.5, -0.25]);
        let mut out = [0.0; 4];
        recording.read(16000.0, &mut out);
        assert_eq!(out, [0.5, 0.125, -0.25, 0.125]);
    }

    /// noise stays within its level and is not silence.
    #[test]
    fn noise_is_as_loud_as_asked() {
        let mut noise = Noise::new(0.5);
        let mut out = [0.0; 1000];
        noise.read(16000.0, &mut out);
        assert!(out.iter().all(|sample| sample.abs() <= 0.5));
        assert!(out.iter().map(|sample| sample.abs()).sum::<f32>() / 1000.0 > 0.2);
    }
}
