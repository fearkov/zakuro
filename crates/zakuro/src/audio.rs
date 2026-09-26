//! the console's sound, played on the host's default output device.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample};

/// how much sound to keep waiting, in the console's samples, about 60 ms.
/// more rides out uneven frames, less answers sooner.
const TARGET: usize = 2048;
/// how much faster or slower than the rates say playback may go to keep
/// the queue near its target, too little to hear.
const DRIFT: f64 = 0.005;

/// the console's samples waiting to be played.
struct Queue {
    samples: VecDeque<[f32; 2]>,
    /// how far the output is between the first two samples.
    fraction: f64,
    /// console samples per output sample.
    step: f64,
    /// ran dry, and waits to fill up before it plays again.
    starved: bool,
}

impl Queue {
    /// the next output sample, between two of the console's.
    fn next(&mut self) -> [f32; 2] {
        if self.starved {
            if self.samples.len() < TARGET {
                return [0.0; 2];
            }
            self.starved = false;
        }
        if self.samples.len() < 2 {
            self.starved = true;
            return [0.0; 2];
        }
        let (a, b) = (self.samples[0], self.samples[1]);
        let t = self.fraction as f32;
        let out = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
        // a longer queue plays a little faster, a shorter one slower
        let error = (self.samples.len() as f64 - TARGET as f64) / TARGET as f64;
        self.fraction += self.step * (1.0 + error.clamp(-1.0, 1.0) * DRIFT);
        while self.fraction >= 1.0 && self.samples.len() > 1 {
            self.samples.pop_front();
            self.fraction -= 1.0;
        }
        out
    }
}

pub struct Audio {
    queue: Arc<Mutex<Queue>>,
    _stream: cpal::Stream,
}

impl Audio {
    /// opens the default output for sound made at rate samples a second.
    pub fn open(rate: f64) -> Result<Audio, String> {
        let host = cpal::default_host();
        let device = host.default_output_device().ok_or("there is no sound output device")?;
        let supported = device.default_output_config().map_err(|e| e.to_string())?;
        let format = supported.sample_format();
        let config: cpal::StreamConfig = supported.into();
        let queue = Arc::new(Mutex::new(Queue {
            samples: VecDeque::new(),
            fraction: 0.0,
            step: rate / config.sample_rate.0 as f64,
            starved: true,
        }));
        let stream = match format {
            SampleFormat::F32 => stream::<f32>(&device, &config, queue.clone()),
            SampleFormat::I16 => stream::<i16>(&device, &config, queue.clone()),
            SampleFormat::U16 => stream::<u16>(&device, &config, queue.clone()),
            SampleFormat::I32 => stream::<i32>(&device, &config, queue.clone()),
            other => return Err(format!("the output takes {other} samples, which Zakuro does not make")),
        }?;
        stream.play().map_err(|e| e.to_string())?;
        Ok(Audio { queue, _stream: stream })
    }

    /// queues what the console played.
    pub fn push(&self, samples: &[[i16; 2]]) {
        let Ok(mut queue) = self.queue.lock() else { return };
        queue.samples.extend(samples.iter().map(|s| s.map(|v| v as f32 / 32768.0)));
        // far ahead, after a stall on the output side, it drops the oldest
        // rather than lag behind the picture
        if queue.samples.len() > TARGET * 4 {
            let excess = queue.samples.len() - TARGET;
            queue.samples.drain(..excess);
        }
    }
}

fn stream<T: SizedSample + FromSample<f32>>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    queue: Arc<Mutex<Queue>>,
) -> Result<cpal::Stream, String> {
    let channels = config.channels as usize;
    device
        .build_output_stream(
            config,
            move |out: &mut [T], _| {
                let Ok(mut queue) = queue.lock() else { return };
                for frame in out.chunks_mut(channels) {
                    let [left, right] = queue.next();
                    match frame {
                        [mono] => *mono = T::from_sample((left + right) / 2.0),
                        [l, r, rest @ ..] => {
                            *l = T::from_sample(left);
                            *r = T::from_sample(right);
                            rest.fill(T::EQUILIBRIUM);
                        }
                        [] => {}
                    }
                }
            },
            |error| log::warn!("sound output, {error}"),
            None,
        )
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn queue(samples: usize, step: f64) -> Queue {
        Queue {
            samples: (0..samples).map(|i| [i as f32, -(i as f32)]).collect(),
            fraction: 0.0,
            step,
            starved: false,
        }
    }

    #[test]
    fn output_falls_between_the_consoles_samples() {
        let mut queue = queue(TARGET, 0.5);
        assert_eq!(queue.next(), [0.0, 0.0]);
        assert_eq!(queue.next(), [0.5, -0.5]);
        assert_eq!(queue.next(), [1.0, -1.0]);
    }

    #[test]
    fn a_queue_that_ran_dry_waits_to_fill_up() {
        let mut queue = queue(1, 1.0);
        assert_eq!(queue.next(), [0.0; 2]);
        assert!(queue.starved);
        queue.samples.extend((0..TARGET).map(|_| [1.0, 1.0]));
        assert_eq!(queue.next(), [0.0, -0.0], "it starts again from where it stopped");
    }

    #[test]
    fn a_long_queue_plays_a_little_faster() {
        let mut long = queue(TARGET * 2, 1.0);
        let mut short = queue(TARGET, 1.0);
        for _ in 0..1000 {
            long.next();
            short.next();
        }
        assert!(TARGET * 2 - long.samples.len() > 1000, "the long queue went through more than it played");
        assert!(TARGET - short.samples.len() < 1000, "the short queue went through less than it played");
    }
}
