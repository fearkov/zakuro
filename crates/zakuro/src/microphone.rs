//! what the console's microphone hears on the computer: its own microphone,
//! open only while a game listens, and the sound of blowing into it while a
//! key is held, for games that ask for that.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample};
use zakuro_core::microphone::{Microphone, Noise};

/// how far behind the computer's microphone a game may fall, in seconds,
/// before the oldest of what it heard goes, down to KEEP.
const MOST: f64 = 0.25;
const KEEP: f64 = 0.05;
/// how loud blowing is, a breath on a microphone is about as loud as it
/// gets.
const BLOWING: f32 = 0.8;

/// what the computer's microphone heard and the game has not, one channel,
/// at its own rate.
#[derive(Default)]
struct Heard {
    samples: VecDeque<f32>,
    /// its samples a second, none until it is open.
    rate: f64,
}

pub struct HostMicrophone {
    /// whether to open the computer's microphone, or only blow.
    device: bool,
    heard: Arc<Mutex<Heard>>,
    /// whether the thread reading the microphone should go on, while there
    /// is one.
    running: Option<Arc<AtomicBool>>,
    /// where between the first two samples heard the next one read falls.
    position: f64,
    /// held while the blowing key is.
    blowing: Arc<AtomicBool>,
    noise: Noise,
}

impl HostMicrophone {
    /// the computer's microphone when device says so, blowing in any case
    /// while blowing is set.
    pub fn new(device: bool, blowing: Arc<AtomicBool>) -> HostMicrophone {
        HostMicrophone {
            device,
            heard: Arc::default(),
            running: None,
            position: 0.0,
            blowing,
            noise: Noise::new(BLOWING),
        }
    }
}

impl Microphone for HostMicrophone {
    fn sampling(&mut self, on: bool) {
        match (on && self.device, &self.running) {
            (true, None) => {
                let running = Arc::new(AtomicBool::new(true));
                let (heard, keep) = (self.heard.clone(), running.clone());
                let started = std::thread::Builder::new().name("microphone".into()).spawn(move || listen(&heard, &keep));
                match started {
                    Ok(_) => self.running = Some(running),
                    Err(error) => log::warn!("could not start listening to the microphone, {error}"),
                }
            }
            (false, Some(running)) => {
                running.store(false, Ordering::Relaxed);
                self.running = None;
                if let Ok(mut heard) = self.heard.lock() {
                    *heard = Heard::default();
                }
                self.position = 0.0;
            }
            _ => {}
        }
    }

    fn read(&mut self, rate: f64, out: &mut [f32]) {
        match self.heard.lock() {
            Ok(mut heard) => self.position = take(&mut heard, self.position, rate, out),
            Err(_) => out.fill(0.0),
        }
        if self.blowing.load(Ordering::Relaxed) {
            for sample in out {
                *sample = (*sample + self.noise.sample()).clamp(-1.0, 1.0);
            }
        }
    }
}

impl Drop for HostMicrophone {
    fn drop(&mut self) {
        self.sampling(false);
    }
}

/// fills out with what was heard at rate samples a second, from position
/// between the first two, between samples where the rates differ and
/// silence where nothing was heard yet. it drops what is used, and the
/// oldest when the game fell too far behind, and gives where it got to.
fn take(heard: &mut Heard, mut position: f64, rate: f64, out: &mut [f32]) -> f64 {
    if heard.rate <= 0.0 || rate <= 0.0 {
        out.fill(0.0);
        return 0.0;
    }
    if heard.samples.len() as f64 > heard.rate * MOST {
        let excess = heard.samples.len() - (heard.rate * KEEP) as usize;
        heard.samples.drain(..excess);
        position = 0.0;
    }
    let step = heard.rate / rate;
    for sample in out.iter_mut() {
        let first = position as usize;
        if first + 1 >= heard.samples.len() {
            // ran dry, the game is ahead of the microphone
            *sample = 0.0;
            continue;
        }
        let (a, b) = (heard.samples[first], heard.samples[first + 1]);
        *sample = a + (b - a) * position.fract() as f32;
        position += step;
    }
    let used = (position as usize).min(heard.samples.len());
    heard.samples.drain(..used);
    position - used as f64
}

/// opens the computer's default microphone and keeps what it hears while
/// running stays set.
fn listen(heard: &Arc<Mutex<Heard>>, running: &AtomicBool) {
    let host = cpal::default_host();
    let Some(device) = host.default_input_device() else {
        log::warn!("there is no microphone to listen with");
        return;
    };
    let supported = match device.default_input_config() {
        Ok(supported) => supported,
        Err(error) => {
            log::warn!("the microphone can't be opened, {error}");
            return;
        }
    };
    let format = supported.sample_format();
    let config: cpal::StreamConfig = supported.into();
    if let Ok(mut heard) = heard.lock() {
        *heard = Heard { samples: VecDeque::new(), rate: config.sample_rate.0 as f64 };
    }
    let stream = match format {
        SampleFormat::F32 => stream::<f32>(&device, &config, heard.clone()),
        SampleFormat::I16 => stream::<i16>(&device, &config, heard.clone()),
        SampleFormat::U16 => stream::<u16>(&device, &config, heard.clone()),
        SampleFormat::I32 => stream::<i32>(&device, &config, heard.clone()),
        other => Err(format!("it gives {other} samples, which Zakuro does not take")),
    };
    let stream = match stream.and_then(|stream| stream.play().map(|()| stream).map_err(|error| error.to_string())) {
        Ok(stream) => stream,
        Err(error) => {
            log::warn!("the microphone can't be opened, {error}");
            return;
        }
    };
    let name = device.name().unwrap_or_else(|_| "the microphone".into());
    log::info!("the microphone is on, {name} at {} Hz", config.sample_rate.0);
    while running.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(20));
    }
    drop(stream);
    log::info!("the microphone is off");
}

fn stream<T: SizedSample>(device: &cpal::Device, config: &cpal::StreamConfig, heard: Arc<Mutex<Heard>>) -> Result<cpal::Stream, String>
where
    f32: FromSample<T>,
{
    let channels = (config.channels as usize).max(1);
    // a second at most waits, the game takes it far sooner
    let most = config.sample_rate.0 as usize;
    device
        .build_input_stream(
            config,
            move |input: &[T], _| {
                let Ok(mut heard) = heard.lock() else { return };
                for frame in input.chunks(channels) {
                    let sum: f32 = frame.iter().map(|&sample| sample.to_sample::<f32>()).sum();
                    heard.samples.push_back(sum / frame.len() as f32);
                }
                let length = heard.samples.len();
                if length > most {
                    heard.samples.drain(..length - most);
                }
            },
            |error| log::warn!("the microphone, {error}"),
            None,
        )
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn heard(samples: &[f32], rate: f64) -> Heard {
        Heard { samples: samples.iter().copied().collect(), rate }
    }

    /// at half its rate every other sample comes through, and what is used
    /// is dropped, the rest waits for the next read.
    #[test]
    fn the_microphone_is_read_at_the_games_rate() {
        let mut heard = heard(&[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 32000.0);
        let mut out = [0.0; 2];
        let position = take(&mut heard, 0.0, 16000.0, &mut out);
        assert_eq!(out, [0.0, 0.2]);
        assert_eq!((heard.samples.len(), position), (2, 0.0));
        assert_eq!(heard.samples[0], 0.4);
    }

    /// at a rate between, the samples fall between the microphone's.
    #[test]
    fn rates_between_fall_between_samples() {
        let mut heard = heard(&[0.0, 1.0, 0.0, 1.0, 0.0], 48000.0);
        let mut out = [0.0; 3];
        let position = take(&mut heard, 0.0, 32000.0, &mut out);
        assert_eq!(out, [0.0, 0.5, 1.0]);
        assert_eq!(position, 0.5);
        assert_eq!(heard.samples.len(), 1);
    }

    /// a game ahead of the microphone hears silence, and one far behind
    /// hears only the latest.
    #[test]
    fn ahead_is_silence_and_far_behind_drops_the_oldest() {
        let mut ahead = heard(&[0.5], 100.0);
        let mut out = [1.0; 3];
        take(&mut ahead, 0.0, 100.0, &mut out);
        assert_eq!(out, [0.0; 3]);

        let samples: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let mut behind = heard(&samples, 100.0);
        let mut out = [0.0; 2];
        take(&mut behind, 0.0, 100.0, &mut out);
        // a quarter of a second at most waits, down to 5 samples kept
        assert_eq!(out, [95.0, 96.0]);
    }

    /// with nothing open, blowing is what the game hears.
    #[test]
    fn blowing_is_heard_without_a_microphone() {
        let blowing = Arc::new(AtomicBool::new(false));
        let mut microphone = HostMicrophone::new(false, blowing.clone());
        microphone.sampling(true);
        let mut out = [1.0; 64];
        microphone.read(16364.0, &mut out);
        assert_eq!(out, [0.0; 64]);
        blowing.store(true, Ordering::Relaxed);
        microphone.read(16364.0, &mut out);
        assert!(out.iter().map(|sample| sample.abs()).sum::<f32>() / 64.0 > 0.2);
    }
}
