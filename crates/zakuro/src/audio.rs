//! the console's sound, played on the host's default output device.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample};

/// how much sound to keep waiting, in the console's samples, about 90 ms.
/// more rides out uneven frames, less answers sooner.
const TARGET: usize = 3072;
/// how much a queue that ran dry waits for before it plays again.
const RESUME: usize = TARGET / 4;
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
    /// the last sample played, which fades out when the queue runs dry
    /// instead of stopping short.
    last: [f32; 2],
    /// times the queue ran dry.
    underruns: u64,
    /// how much a queue that ran dry waits for before it plays again.
    refill: usize,
}

impl Queue {
    /// the next output sample, between two of the console's.
    fn next(&mut self) -> [f32; 2] {
        if !self.starved && self.samples.len() < 2 {
            self.starved = true;
            self.underruns += 1;
        }
        if self.starved {
            if self.samples.len() < self.refill {
                self.last = self.last.map(|v| v * 0.995);
                return self.last;
            }
            self.starved = false;
            self.refill = RESUME;
        }
        let (a, b) = (self.samples[0], self.samples[1]);
        let t = self.fraction as f32;
        let out = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
        self.last = out;
        // a longer queue plays a little faster, a shorter one slower
        let error = (self.samples.len() as f64 - TARGET as f64) / TARGET as f64;
        self.fraction += self.step * (1.0 + error.clamp(-1.0, 1.0) * DRIFT);
        while self.fraction >= 1.0 && self.samples.len() > 1 {
            self.samples.pop_front();
            self.fraction -= 1.0;
        }
        out
    }

    /// stops it while the game is stopped, without counting that as running
    /// dry, and has it fill all the way up before it plays again, so the
    /// first frames after do not run it dry either.
    fn hold(&mut self) {
        self.starved = true;
        self.refill = TARGET;
        self.underruns = 0;
    }

    /// drops what is queued, holding as above, fading out from what played
    /// last.
    fn clear(&mut self) {
        self.samples.clear();
        self.fraction = 0.0;
        self.hold();
    }
}

/// the pieces of sound stretching moves around, about 16 ms, and how far
/// apart they go out, half of that.
const GRAIN: usize = 512;
const HOP: usize = GRAIN / 2;
/// how far a piece may shift, about 4 ms, to line up with the one before.
const SEARCH: usize = 128;
/// sound gets at most twice as long.
const MOST_STRETCH: f64 = 2.0;

/// time stretching, sound made longer without its pitch changing, for when
/// the emulation falls behind and the queue would run dry. overlapping
/// pieces of it go out further apart than they came in, each shifted to
/// line up with the one before, which is WSOLA.
struct Stretch {
    /// what came in and has not gone out yet.
    input: Vec<[f32; 2]>,
    /// where in input the next piece is due.
    position: f64,
    /// where the last piece came from, none before the first.
    last: Option<usize>,
    /// the last piece's second half, which the next one's first adds to.
    tail: Vec<[f32; 2]>,
    /// how much longer sound comes out, eased toward what the queue asks.
    factor: f64,
    window: Vec<f32>,
}

impl Stretch {
    fn new() -> Stretch {
        // a periodic Hann window, whose halves add up to one
        let window = (0..GRAIN).map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / GRAIN as f32).cos()).collect();
        Stretch { input: Vec::new(), position: 0.0, last: None, tail: vec![[0.0; 2]; HOP], factor: 1.0, window }
    }

    /// forgets what came in, to start over on other sound.
    fn clear(&mut self) {
        self.input.clear();
        self.position = 0.0;
        self.last = None;
        self.tail.fill([0.0; 2]);
        self.factor = 1.0;
    }

    /// stretches samples by about factor, passing out what is ready.
    fn process(&mut self, samples: &[[f32; 2]], factor: f64, out: &mut Vec<[f32; 2]>) {
        self.input.extend_from_slice(samples);
        self.factor += (factor - self.factor) * 0.2;
        let mono = |s: [f32; 2]| s[0] + s[1];
        loop {
            let due = self.position as usize;
            let natural = self.last.map_or(due, |last| last + HOP);
            if due + SEARCH + GRAIN > self.input.len() || natural + GRAIN > self.input.len() {
                break;
            }
            // the piece around where it is due that most looks like the
            // last one's continuation, which is where it is due when
            // nothing is stretched
            let start = if natural == due {
                due
            } else {
                let template = &self.input[natural..natural + HOP];
                let score = |candidate: usize| {
                    let (mut dot, mut energy) = (0.0f32, 1e-9f32);
                    for (a, &b) in self.input[candidate..candidate + HOP].iter().zip(template) {
                        dot += mono(*a) * mono(b);
                        energy += mono(*a) * mono(*a);
                    }
                    dot / energy.sqrt()
                };
                (due.saturating_sub(SEARCH)..=due + SEARCH).max_by(|&a, &b| score(a).total_cmp(&score(b))).unwrap_or(due)
            };
            let piece = &self.input[start..start + GRAIN];
            let (first, second) = piece.split_at(HOP);
            let (rising, falling) = self.window.split_at(HOP);
            for ((tail, sample), w) in self.tail.iter_mut().zip(first).zip(rising) {
                out.push([tail[0] + sample[0] * w, tail[1] + sample[1] * w]);
            }
            for ((tail, sample), w) in self.tail.iter_mut().zip(second).zip(falling) {
                *tail = [sample[0] * w, sample[1] * w];
            }
            self.last = Some(start);
            self.position += HOP as f64 / self.factor;
        }
        // what no piece can reach any more goes
        let used = (self.position as usize).min(self.last.unwrap_or(0)).saturating_sub(SEARCH);
        if used > 0 {
            self.input.drain(..used);
            self.position -= used as f64;
            self.last = self.last.map(|last| last - used);
        }
    }
}

pub struct Audio {
    queue: Arc<Mutex<Queue>>,
    /// 0 to 1, applied as samples come in.
    volume: std::cell::Cell<f32>,
    stretch: std::cell::RefCell<Stretch>,
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
            last: [0.0; 2],
            underruns: 0,
            refill: RESUME,
        }));
        let stream = match format {
            SampleFormat::F32 => stream::<f32>(&device, &config, queue.clone()),
            SampleFormat::I16 => stream::<i16>(&device, &config, queue.clone()),
            SampleFormat::U16 => stream::<u16>(&device, &config, queue.clone()),
            SampleFormat::I32 => stream::<i32>(&device, &config, queue.clone()),
            other => return Err(format!("the output takes {other} samples, which Zakuro does not make")),
        }?;
        stream.play().map_err(|e| e.to_string())?;
        Ok(Audio { queue, volume: std::cell::Cell::new(1.0), stretch: std::cell::RefCell::new(Stretch::new()), _stream: stream })
    }

    /// how many times the sound ran dry since the last call.
    pub fn take_underruns(&self) -> u64 {
        self.queue.lock().map(|mut queue| std::mem::take(&mut queue.underruns)).unwrap_or(0)
    }

    /// stops the sound while the game is stopped, see Queue::hold.
    pub fn hold(&self) {
        if let Ok(mut queue) = self.queue.lock() {
            queue.hold();
        }
    }

    /// drops the sound queued and half stretched, for a game starting or
    /// stopping, so none of the last one's plays at the start of the next.
    pub fn clear(&self) {
        if let Ok(mut queue) = self.queue.lock() {
            queue.clear();
        }
        self.stretch.borrow_mut().clear();
    }

    pub fn set_volume(&self, volume: f32) {
        self.volume.set(volume.clamp(0.0, 1.0));
    }

    /// queues what the console played, stretched when the queue runs low
    /// so that it plays on rather than stops.
    pub fn push(&self, samples: &[[i16; 2]]) {
        let scale = self.volume.get() / 32768.0;
        let samples: Vec<[f32; 2]> = samples.iter().map(|s| s.map(|v| v as f32 * scale)).collect();
        let Ok(mut queue) = self.queue.lock() else { return };
        enqueue(&mut queue, &mut self.stretch.borrow_mut(), &samples);
    }
}

/// adds what the console played to the queue, stretched by how far the
/// queue is from its target.
fn enqueue(queue: &mut Queue, stretch: &mut Stretch, samples: &[[f32; 2]]) {
    let short = TARGET.saturating_sub(queue.samples.len()) as f64 / TARGET as f64;
    let mut stretched = Vec::with_capacity(samples.len() * 2);
    stretch.process(samples, (1.0 + short).min(MOST_STRETCH), &mut stretched);
    queue.samples.extend(stretched);
    // far ahead, after a stall on the output side, it drops the oldest
    // rather than lag behind the picture
    if queue.samples.len() > TARGET * 4 {
        let excess = queue.samples.len() - TARGET;
        queue.samples.drain(..excess);
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
            last: [0.0; 2],
            underruns: 0,
            refill: RESUME,
        }
    }

    /// a tone at 440 Hz, the console's rate.
    fn tone(count: usize) -> Vec<[f32; 2]> {
        (0..count)
            .map(|i| {
                let v = (std::f32::consts::TAU * 440.0 * i as f32 / 32728.0).sin() * 0.5;
                [v, v]
            })
            .collect()
    }

    /// how many times a sound crosses zero going up, which tells its pitch.
    fn rises(samples: &[[f32; 2]]) -> usize {
        samples.windows(2).filter(|w| w[0][0] < 0.0 && w[1][0] >= 0.0).count()
    }

    #[test]
    fn unstretched_sound_comes_out_as_it_went_in() {
        let input = tone(20_000);
        let mut stretch = Stretch::new();
        let mut out = Vec::new();
        for chunk in input.chunks(546) {
            stretch.process(chunk, 1.0, &mut out);
        }
        // past the first piece fading in, every sample is the one that came in
        assert!(out.len() > 18_000);
        for (a, b) in out[HOP..].iter().zip(&input[HOP..]) {
            assert!((a[0] - b[0]).abs() < 1e-4);
        }
    }

    #[test]
    fn stretched_sound_is_longer_at_the_same_pitch() {
        let input = tone(32_728);
        let mut stretch = Stretch::new();
        let mut out = Vec::new();
        for chunk in input.chunks(546) {
            stretch.process(chunk, 1.5, &mut out);
        }
        let ratio = out.len() as f64 / input.len() as f64;
        assert!(ratio > 1.35 && ratio < 1.55, "stretched by {ratio}");
        // the same number of cycles a second
        let pitch = rises(&out) as f64 / out.len() as f64 * 32728.0;
        assert!((pitch - 440.0).abs() < 10.0, "pitch {pitch}");
    }

    #[test]
    fn output_falls_between_the_consoles_samples() {
        let mut queue = queue(TARGET, 0.5);
        assert_eq!(queue.next(), [0.0, 0.0]);
        assert_eq!(queue.next(), [0.5, -0.5]);
        assert_eq!(queue.next(), [1.0, -1.0]);
    }

    #[test]
    fn a_queue_that_ran_dry_fades_and_waits_to_fill_up() {
        let mut queue = queue(1, 1.0);
        queue.last = [0.5, 0.5];
        let faded = queue.next();
        assert!(queue.starved);
        assert_eq!(queue.underruns, 1);
        assert!(faded[0] < 0.5 && faded[0] > 0.4, "it fades out rather than stopping short");
        queue.samples.extend((0..RESUME).map(|_| [1.0, 1.0]));
        assert_eq!(queue.next(), [0.0, -0.0], "it starts again from where it stopped");
    }

    /// stopped while the game is, it does not count running dry, and plays
    /// again only once it is full.
    #[test]
    fn a_held_queue_waits_to_fill_up_and_counts_nothing() {
        let mut queue = queue(0, 1.0);
        queue.last = [0.5, 0.5];
        queue.hold();
        assert!(queue.next()[0] < 0.5);
        assert_eq!(queue.underruns, 0);
        queue.samples.extend((0..RESUME).map(|_| [1.0, 1.0]));
        assert!(queue.next()[0] < 0.5, "what is enough after running dry is not after a hold");
        queue.samples.extend((0..TARGET).map(|_| [1.0, 1.0]));
        assert_eq!(queue.next(), [1.0, 1.0]);
        assert_eq!(queue.underruns, 0);
        // running dry later is counted, and waits for the usual amount
        queue.samples.clear();
        queue.next();
        assert_eq!(queue.underruns, 1);
        queue.samples.extend((0..RESUME).map(|_| [1.0, 1.0]));
        assert_eq!(queue.next(), [1.0, 1.0]);
    }

    /// a game starting or stopping drops what the last one queued, and the
    /// new sound plays once there is enough of it.
    #[test]
    fn a_cleared_queue_plays_none_of_the_old_sound() {
        let mut queue = queue(TARGET, 1.0);
        queue.clear();
        queue.next();
        queue.samples.extend((0..TARGET).map(|_| [-1.0, -1.0]));
        assert_eq!(queue.next(), [-1.0, -1.0]);
        assert_eq!(queue.underruns, 0);
    }

    /// how far behind the picture the sound is, in ms, after each second of
    /// a minute where the console makes exactly as much sound as the output
    /// plays, but a frame now and then comes late and the next ones catch up,
    /// as when the emulation hitches.
    fn latency_with_hitches() -> Vec<f64> {
        const CONSOLE: f64 = 32728.498;
        const OUTPUT: f64 = 48000.0;
        let mut queue = queue(0, CONSOLE / OUTPUT);
        let mut stretch = Stretch::new();
        let mut seconds = Vec::new();
        let tone = tone(40_000);
        let mut at = 0;
        let mut late: Vec<[f32; 2]> = Vec::new();
        for frame in 0..3600usize {
            // a frame's worth of sound, held back every 45th frame for three
            // frames' time
            let count = (CONSOLE * (frame + 1) as f64 / 60.0) as usize - (CONSOLE * frame as f64 / 60.0) as usize;
            let mut chunk: Vec<[f32; 2]> = (0..count).map(|i| tone[(at + i) % tone.len()]).collect();
            at += count;
            if frame % 45 == 0 {
                late.append(&mut chunk);
            } else if frame % 45 == 3 {
                late.append(&mut chunk);
                enqueue(&mut queue, &mut stretch, &late);
                late.clear();
            } else if late.is_empty() {
                enqueue(&mut queue, &mut stretch, &chunk);
            } else {
                late.append(&mut chunk);
            }
            for _ in 0..(OUTPUT / 60.0) as usize {
                queue.next();
            }
            if frame % 60 == 59 {
                seconds.push(queue.samples.len() as f64 / CONSOLE * 1000.0);
            }
        }
        seconds
    }

    #[test]
    fn hitches_do_not_put_the_sound_further_and_further_behind() {
        let seconds = latency_with_hitches();
        let first = seconds[5];
        let last = *seconds.last().unwrap();
        let most = seconds.iter().cloned().fold(0.0, f64::max);
        eprintln!("latency by second {:?}", seconds.iter().map(|ms| ms.round() as i64).collect::<Vec<_>>());
        assert!(last < first + 30.0, "behind by {first:.0} ms after 5 s and {last:.0} ms after a minute");
        assert!(most < 160.0, "up to {most:.0} ms behind");
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
