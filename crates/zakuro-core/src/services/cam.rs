//! cam:u, the cameras
//! what they see comes from the frontend (webcam or a picture), scaled to the title's size, flipped and colored as it set
//! in its format, written into its buffer at the camera's frame rate, both outer cameras and the inner one see the same thing

use crate::kernel::ipc::{CommandBuffer, Descriptor, Header};
use crate::kernel::object::{KObject, ObjectId};
use crate::kernel::sync::ResetType;
use crate::kernel::thread::CPU_CLOCK_HZ;
use crate::System;
use zakuro_common::result::errors;

/// the cameras as their select bits, outer right, inner, outer left
const CAMERAS: usize = 3;
/// the ports pictures come in thru, a camera on each
const PORTS: usize = 2;
/// the formats a camera puts out
const YUV422: u8 = 0;
const RGB565: u8 = 1;
/// the flips
const FLIP_HORIZONTAL: u8 = 1;
const FLIP_VERTICAL: u8 = 2;
const FLIP_BOTH: u8 = 3;
/// the effects
const MONO: u8 = 1;
const SEPIA: u8 = 2;
const NEGATIVE: u8 = 3;
const NEGATIVE_FILM: u8 = 4;
const SEPIA_01: u8 = 5;
/// how many vsync times a title can ask for
const VSYNC_TIMINGS: usize = 5;

/// what a camera puts out in each of its two contexts
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
struct Context {
    width: u16,
    height: u16,
    format: u8,
    flip: u8,
    effect: u8,
}

impl Default for Context {
    fn default() -> Self {
        Context { width: 640, height: 480, format: YUV422, flip: 0, effect: 0 }
    }
}

#[derive(Debug, Default, Clone, Copy, serde::Serialize, serde::Deserialize)]
struct Camera {
    contexts: [Context; 2],
    context: usize,
    frame_rate: u8,
}

/// a port, its camera, what it captures and where the next picture goes
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
struct Port {
    camera: Option<usize>,
    capturing: bool,
    /// where the title wants the next picture, how many bytes, and when it lands, while capturing
    receive: Option<(u32, u32)>,
    lands_at: Option<u64>,
    trimming: bool,
    /// corners of the kept part, left, top, right, bottom
    trim: [i16; 4],
    transfer_bytes: u32,
    next_vsync: Option<u64>,
    /// ticks of the latest vsyncs, newest first
    vsyncs: Vec<u64>,
    /// the received, vsync and buffer error events
    events: Option<[ObjectId; 3]>,
}

const RECEIVED: usize = 0;
const VSYNC: usize = 1;
const BUFFER_ERROR: usize = 2;

#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct CamState {
    cameras: [Camera; CAMERAS],
    ports: [Port; PORTS],
}

/// the sizes a title picks from
fn size(index: u32) -> (u16, u16) {
    match index {
        0 => (640, 480),
        1 => (320, 240),
        2 => (160, 120),
        3 => (352, 288),
        4 => (176, 144),
        5 => (256, 192),
        6 => (512, 384),
        7 => (400, 240),
        _ => (320, 240),
    }
}

/// ticks between frames at a frame rate, the highest a varying one hits
fn frame_ticks(rate: u8) -> u64 {
    let per_second: f64 = match rate {
        0..=2 | 10 => 15.0,
        3 => 10.0,
        4 => 8.5,
        5 => 5.0,
        6 | 7 | 11 => 20.0,
        _ => 30.0,
    };
    (CPU_CLOCK_HZ as f64 / per_second) as u64
}

/// cameras a selection names
fn cameras(select: u32) -> impl Iterator<Item = usize> {
    (0..CAMERAS).filter(move |i| select & 1 << i != 0)
}

/// ports a selection names
fn ports(select: u32) -> impl Iterator<Item = usize> {
    (0..PORTS).filter(move |i| select & 1 << i != 0)
}

/// contexts a selection names
fn contexts(select: u32) -> impl Iterator<Item = usize> {
    (0..2).filter(move |i| select & 1 << i != 0)
}

pub fn handle(system: &mut System, buffer: &CommandBuffer, header: Header) -> bool {
    let command = header.command_id();
    let word = |system: &mut System, index: u32| buffer.get(&mut system.memory, index);
    match command {
        // StartCapture(ports)
        0x0001 => {
            let now = system.cpu.cycles;
            for port in ports(word(system, 1)) {
                let state = &mut system.services.cam;
                let Some(camera) = state.ports[port].camera else {
                    log::warn!("cam: port {port} captures with no camera on it");
                    continue;
                };
                let ticks = frame_ticks(state.cameras[camera].frame_rate);
                let port = &mut state.ports[port];
                if !port.capturing {
                    port.capturing = true;
                    if port.receive.is_some() {
                        port.lands_at = Some(now + ticks);
                    }
                }
            }
            capturing(system);
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // StopCapture(ports)
        0x0002 => {
            for port in ports(word(system, 1)) {
                let port = &mut system.services.cam.ports[port];
                port.capturing = false;
                port.lands_at = None;
            }
            capturing(system);
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // IsBusy(ports), IsFinishedReceiving(ports)
        0x0003 | 0x0008 => {
            let select = word(system, 1);
            let state = &system.services.cam;
            let all = |test: &dyn Fn(&Port) -> bool| ports(select).all(|port| test(&state.ports[port]));
            let answer = if command == 0x0003 { all(&|port| port.capturing) } else { all(&|port| port.receive.is_none()) };
            buffer.reply(&mut system.memory, command, &[answer as u32]);
            true
        }
        // ClearBuffer(ports)
        0x0004 => {
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // GetVsyncInterruptEvent, GetBufferErrorInterruptEvent(port)
        0x0005 | 0x0006 => {
            let port = ports(word(system, 1)).next().unwrap_or(0);
            let which = if command == 0x0005 { VSYNC } else { BUFFER_ERROR };
            let object = events(system, port)[which];
            reply_with_event(system, buffer, command, object);
            true
        }
        // SetReceiving(destination, port, size, transfer unit, process)
        0x0007 => {
            let destination = word(system, 1);
            let port = ports(word(system, 2)).next().unwrap_or(0);
            let size = word(system, 3);
            let object = events(system, port)[RECEIVED];
            system.kernel.clear_event(object);
            let now = system.cpu.cycles;
            let state = &mut system.services.cam;
            let ticks = state.ports[port].camera.map_or(frame_ticks(0), |camera| frame_ticks(state.cameras[camera].frame_rate));
            let port = &mut state.ports[port];
            port.receive = Some((destination, size));
            port.lands_at = port.capturing.then_some(now + ticks);
            reply_with_event(system, buffer, command, object);
            true
        }
        // SetTransferLines(port, lines, width, height)
        0x0009 => {
            let lines = word(system, 2) & 0xFFFF;
            let width = word(system, 3) & 0xFFFF;
            for port in ports(word(system, 1)) {
                system.services.cam.ports[port].transfer_bytes = lines * width * 2;
            }
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // GetMaxLines(width, height), GetMaxBytes(width, height), what one transfer takes, like the console works it out
        0x000A | 0x000D => {
            let width = word(system, 1) & 0xFFFF;
            let height = word(system, 2) & 0xFFFF;
            let answer = if command == 0x000A { max_lines(width, height) } else { max_bytes(width, height) };
            match answer {
                Some(value) => buffer.reply(&mut system.memory, command, &[value]),
                None => buffer.reply_error(&mut system.memory, command, errors::OUT_OF_RANGE.0),
            }
            true
        }
        // SetTransferBytes(port, bytes, width, height)
        0x000B => {
            let bytes = word(system, 2) & 0xFFFF;
            for port in ports(word(system, 1)) {
                system.services.cam.ports[port].transfer_bytes = bytes;
            }
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // GetTransferBytes(port)
        0x000C => {
            let port = ports(word(system, 1)).next().unwrap_or(0);
            let bytes = system.services.cam.ports[port].transfer_bytes;
            buffer.reply(&mut system.memory, command, &[bytes]);
            true
        }
        // SetTrimming(ports, on)
        0x000E => {
            let on = word(system, 2) & 0xFF != 0;
            for port in ports(word(system, 1)) {
                system.services.cam.ports[port].trimming = on;
            }
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // IsTrimming(port)
        0x000F => {
            let port = ports(word(system, 1)).next().unwrap_or(0);
            let on = system.services.cam.ports[port].trimming;
            buffer.reply(&mut system.memory, command, &[on as u32]);
            true
        }
        // SetTrimmingParams(ports, left, top, right, bottom)
        0x0010 => {
            let trim = [2, 3, 4, 5].map(|i| word(system, i) as i16);
            for port in ports(word(system, 1)) {
                system.services.cam.ports[port].trim = trim;
            }
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // GetTrimmingParams(port)
        0x0011 => {
            let port = ports(word(system, 1)).next().unwrap_or(0);
            let trim = system.services.cam.ports[port].trim.map(|corner| corner as u16 as u32);
            buffer.reply(&mut system.memory, command, &trim);
            true
        }
        // SetTrimmingParamsCenter(ports, trimmed width and height, camera width and height)
        0x0012 => {
            let [width, height, camera_width, camera_height] = [2, 3, 4, 5].map(|i| word(system, i) as i16);
            let (left, top) = ((camera_width - width) / 2, (camera_height - height) / 2);
            for port in ports(word(system, 1)) {
                system.services.cam.ports[port].trim = [left, top, left + width, top + height];
            }
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // Activate(cameras), none turns them all off
        // the inner or outer right camera goes on port 1, outer left on port 2 (and on port 1 too when alone)
        // a camera on starts its vsyncs, which a title waits for b4 capturing
        0x0013 => {
            let select = word(system, 1) & 7;
            let first = [0, 1, 2].into_iter().find(|&camera| select & 1 << camera != 0);
            let second = (select & 4 != 0).then_some(2);
            let now = system.cpu.cycles;
            let state = &mut system.services.cam;
            let rates = state.cameras.map(|camera| camera.frame_rate);
            for (port, camera) in state.ports.iter_mut().zip([first, second]) {
                if camera.is_none() || port.camera != camera {
                    port.capturing = false;
                    port.lands_at = None;
                    port.next_vsync = camera.map(|camera| now + frame_ticks(rates[camera]));
                }
                port.camera = camera;
            }
            capturing(system);
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // SwitchContext(cameras, context)
        0x0014 => {
            let context = contexts(word(system, 2)).next().unwrap_or(0);
            for camera in cameras(word(system, 1)) {
                system.services.cam.cameras[camera].context = context;
            }
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // FlipImage(cameras, flip, contexts)
        0x001D => {
            let (select, flip, context_select) = (word(system, 1), word(system, 2) as u8, word(system, 3));
            set_contexts(system, select, context_select, |context| context.flip = flip);
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // SetDetailSize(cameras, width, height, crop corners, contexts)
        0x001E => {
            let (width, height) = (word(system, 2) as u16, word(system, 3) as u16);
            let (select, context_select) = (word(system, 1), word(system, 8));
            set_contexts(system, select, context_select, |context| {
                (context.width, context.height) = (width, height);
            });
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // SetSize(cameras, size, contexts)
        0x001F => {
            let (width, height) = size(word(system, 2) & 0xFF);
            let (select, context_select) = (word(system, 1), word(system, 3));
            set_contexts(system, select, context_select, |context| {
                (context.width, context.height) = (width, height);
            });
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // SetFrameRate(cameras, rate)
        0x0020 => {
            let rate = word(system, 2) as u8;
            for camera in cameras(word(system, 1)) {
                system.services.cam.cameras[camera].frame_rate = rate;
            }
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // SetEffect(cameras, effect, contexts)
        0x0022 => {
            let (select, effect, context_select) = (word(system, 1), word(system, 2) as u8, word(system, 3));
            set_contexts(system, select, context_select, |context| context.effect = effect);
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // SetOutputFormat(cameras, format, contexts)
        0x0025 => {
            let (select, format, context_select) = (word(system, 1), word(system, 2) as u8, word(system, 3));
            set_contexts(system, select, context_select, |context| context.format = format);
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // GetLatestVsyncTiming(port, count), into the title's static buffer
        0x002A => {
            let port = ports(word(system, 1)).next().unwrap_or(0);
            let count = (word(system, 2) as usize).min(VSYNC_TIMINGS);
            let mut bytes = Vec::with_capacity(count * 8);
            for i in 0..count {
                let tick = system.services.cam.ports[port].vsyncs.get(i).copied().unwrap_or(0);
                bytes.extend_from_slice(&tick.to_le_bytes());
            }
            let tls = system.kernel.current().map_or(0, |thread| thread.tls);
            let (destination, capacity) = buffer.static_buffer(&mut system.memory, tls, 0);
            let length = bytes.len().min(capacity as usize);
            system.memory.write_bytes(destination, &bytes[..length]);
            buffer.set(&mut system.memory, 0, Header::new(command, 1, 2).0);
            buffer.set(&mut system.memory, 1, 0);
            buffer.set(&mut system.memory, 2, Descriptor::static_buffer(length as u32, 0));
            buffer.set(&mut system.memory, 3, destination);
            true
        }
        // GetStereoCameraCalibrationData, a real console's, a title using it spins forever on invalid data
        0x002B => {
            buffer.reply(&mut system.memory, command, &stereo_calibration());
            true
        }
        // ReadRegisterI2cExclusive, ReadMcuVariableI2cExclusive
        0x002F | 0x0030 => {
            buffer.reply(&mut system.memory, command, &[0]);
            true
        }
        // GetImageQualityCalibrationData
        0x0032 => {
            buffer.reply(&mut system.memory, command, &[0; 6]);
            true
        }
        // SetPackageParameterWithoutContext(cameras, ..., frame rate, ...), bytes back to back
        0x0033 => {
            let first = word(system, 1);
            let second = word(system, 2);
            let rate = (second >> 16) as u8;
            for camera in cameras(first & 0xFF) {
                system.services.cam.cameras[camera].frame_rate = rate;
            }
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // SetPackageParameterWithContext(cameras, contexts, flip, effect, size)
        // and WithContextDetail(cameras, contexts, flip, effect, width, height, crop), bytes back to back
        0x0034 | 0x0035 => {
            let first = word(system, 1);
            let [select, context_select, flip, effect] = first.to_le_bytes();
            let (width, height) = if command == 0x0034 {
                size(word(system, 2) & 0xFF)
            } else {
                let second = word(system, 2);
                (second as u16, (second >> 16) as u16)
            };
            set_contexts(system, select as u32, context_select as u32, |context| {
                context.flip = flip;
                context.effect = effect;
                (context.width, context.height) = (width, height);
            });
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // GetSuitableY2rStandardCoefficient, ITU R BT.601 full range, how the pictures come
        0x0036 => {
            buffer.reply(&mut system.memory, command, &[0]);
            true
        }
        // DriverInitialize, everything as it starts
        0x0039 => {
            let state = &mut system.services.cam;
            state.cameras = Default::default();
            for port in &mut state.ports {
                *port = Port { events: port.events, ..Port::default() };
            }
            capturing(system);
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // DriverFinalize
        0x003A => {
            for port in &mut system.services.cam.ports {
                port.capturing = false;
                port.camera = None;
                port.receive = None;
                port.lands_at = None;
                port.next_vsync = None;
            }
            capturing(system);
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // GetActivatedCamera
        0x003B => {
            let select = system.services.cam.ports.iter().filter_map(|port| port.camera).fold(0, |select, camera| select | 1 << camera);
            buffer.reply(&mut system.memory, command, &[select]);
            true
        }
        // IsAutoExposure, IsAutoWhiteBalance, always
        0x001A | 0x001C => {
            buffer.reply(&mut system.memory, command, &[1]);
            true
        }
        // GetSleepCamera
        0x003C => {
            buffer.reply(&mut system.memory, command, &[0]);
            true
        }
        // settings a picture taken this way has no use for
        // exposure, white balance, sharpness, photo mode, contrast, lens correction, their measuring windows
        // noise, syncing the two outer cameras, calibration, the shutter sound and sleep
        0x0015..=0x0019 | 0x001B | 0x0021 | 0x0023 | 0x0024 | 0x0026..=0x0029 | 0x002C..=0x002E | 0x0031 | 0x0037
        | 0x0038 | 0x003D | 0x003E => {
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        _ => false,
    }
}

/// changes the named contexts of the named cameras
fn set_contexts(system: &mut System, cameras_select: u32, contexts_select: u32, change: impl Fn(&mut Context)) {
    for camera in cameras(cameras_select) {
        for context in contexts(contexts_select) {
            change(&mut system.services.cam.cameras[camera].contexts[context]);
        }
    }
}

/// tells the frontend's camera if any port captures
pub(crate) fn capturing(system: &mut System) {
    let on = system.services.cam.ports.iter().any(|port| port.capturing);
    if let Some(camera) = &mut system.camera {
        camera.capturing(on);
    }
}

/// max lines of width a transfer takes, none when nothing fits
fn max_lines(width: u32, height: u32) -> Option<u32> {
    const UNIT: u32 = 256;
    const MOST: u32 = 2560;
    if width == 0 || height == 0 || !(width * height * 2).is_multiple_of(UNIT) {
        return None;
    }
    let mut lines = (MOST / width).min(height);
    while lines > 0 && (!height.is_multiple_of(lines) || !(lines * width * 2).is_multiple_of(UNIT)) {
        lines -= 1;
    }
    (lines > 0).then_some(lines)
}

/// max bytes a transfer takes, none when nothing fits
fn max_bytes(width: u32, height: u32) -> Option<u32> {
    const UNIT: u32 = 256;
    const MOST: u32 = 2560;
    if width == 0 || height == 0 || !(width * height * 2).is_multiple_of(UNIT) {
        return None;
    }
    let mut bytes = MOST;
    while !(width * height * 2).is_multiple_of(bytes) {
        bytes -= UNIT;
    }
    Some(bytes)
}

/// a console's stereo calibration, angle and distance of the two outer cameras
fn stereo_calibration() -> [u32; 16] {
    let float = |value: f32| value.to_bits();
    [
        0, // the XY rotation is not valid, and three reserved bytes
        float(1.0),
        float(0.0),
        float(-35.0),
        float(0.0),
        float(0.0),
        float(0.0),
        float(64.0),
        float(64.0),
        float(250.0),
        float(35.0),
        640 | 480 << 16,
        0,
        0,
        0,
        0,
    ]
}

/// a port's events, made the first time they're asked for
fn events(system: &mut System, port: usize) -> [ObjectId; 3] {
    if let Some(events) = system.services.cam.ports[port].events {
        return events;
    }
    let names = ["CAM:Received", "CAM:Vsync", "CAM:BufferError"];
    let events = names.map(|name| {
        let object = system.kernel.objects.insert(KObject::Event(crate::kernel::sync::Event::new(ResetType::OneShot, name)));
        // the service keeps them, a title closing its handles must not take them away
        system.kernel.objects.add_ref(object);
        object
    });
    system.services.cam.ports[port].events = Some(events);
    events
}

fn reply_with_event(system: &mut System, buffer: &CommandBuffer, command: u16, object: ObjectId) {
    let handle = system.kernel.handles.create(&mut system.kernel.objects, object, "cam event");
    buffer.set(&mut system.memory, 0, Header::new(command, 1, 2).0);
    buffer.set(&mut system.memory, 1, 0);
    buffer.set(&mut system.memory, 2, Descriptor::handles(1));
    buffer.set(&mut system.memory, 3, handle);
}

/// signals due vsyncs and lands due pictures
pub fn tick(system: &mut System) {
    let now = system.cpu.cycles;
    for index in 0..PORTS {
        let port = &system.services.cam.ports[index];
        let Some(camera) = port.camera else { continue };
        let ticks = frame_ticks(system.services.cam.cameras[camera].frame_rate);
        if port.next_vsync.is_some_and(|at| at <= now) {
            let port = &mut system.services.cam.ports[index];
            port.next_vsync = Some(now + ticks);
            port.vsyncs.insert(0, now);
            port.vsyncs.truncate(VSYNC_TIMINGS);
            let object = events(system, index)[VSYNC];
            system.kernel.signal_event(object);
        }
        let port = &system.services.cam.ports[index];
        if let (Some(at), Some((destination, size))) = (port.lands_at, port.receive) {
            if at <= now {
                let picture = take(system, index, camera);
                let length = picture.len().min(size as usize);
                system.memory.write_bytes(destination, &picture[..length]);
                let port = &mut system.services.cam.ports[index];
                port.receive = None;
                port.lands_at = None;
                let object = events(system, index)[RECEIVED];
                system.kernel.signal_event(object);
            }
        }
    }
}

/// a picture as a camera on a port puts it out, at its size, flipped, colored, in its format, trimmed as the port says
fn take(system: &mut System, port: usize, camera: usize) -> Vec<u8> {
    let state = &system.services.cam;
    let unit = state.cameras[camera];
    let context = unit.contexts[unit.context];
    let (trimming, trim) = (state.ports[port].trimming, state.ports[port].trim);
    let (width, height) = (context.width as u32, context.height as u32);
    let picture = system.camera.as_mut().and_then(|camera| camera.picture());
    let mut pixels = match picture {
        Some(picture) => picture.scaled(width, height),
        None => vec![[0; 3]; (width * height) as usize],
    };
    flip(&mut pixels, width as usize, context.flip);
    let mut bytes = encode(&pixels, width as usize, context.format, context.effect);
    if trimming {
        let [left, top, right, bottom] = trim.map(|corner| corner.max(0) as usize);
        let (right, bottom) = (right.min(width as usize), bottom.min(height as usize));
        if left < right && top < bottom {
            let row = width as usize * 2;
            bytes = (top..bottom).flat_map(|y| bytes[y * row + left * 2..y * row + right * 2].to_vec()).collect();
        }
    }
    bytes
}

fn flip(pixels: &mut [[u8; 3]], width: usize, flip: u8) {
    if width == 0 {
        return;
    }
    if flip == FLIP_HORIZONTAL || flip == FLIP_BOTH {
        for row in pixels.chunks_mut(width) {
            row.reverse();
        }
    }
    if flip == FLIP_VERTICAL || flip == FLIP_BOTH {
        let rows = pixels.len() / width;
        for y in 0..rows / 2 {
            let (top, bottom) = pixels.split_at_mut((rows - 1 - y) * width);
            top[y * width..(y + 1) * width].swap_with_slice(&mut bottom[..width]);
        }
    }
}

/// Y, U and V of a pixel, ITU R BT.601 full range
fn yuv([r, g, b]: [u8; 3]) -> [f32; 3] {
    let (r, g, b) = (r as f32, g as f32, b as f32);
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    [y, (b - y) * 0.564 + 128.0, (r - y) * 0.713 + 128.0]
}

/// pixels in a camera's format, 2 bytes each, w/ its effect
fn encode(pixels: &[[u8; 3]], width: usize, format: u8, effect: u8) -> Vec<u8> {
    let byte = |value: f32| value.round().clamp(0.0, 255.0) as u8;
    // the effects work on Y, U and V
    let colored = |[y, u, v]: [f32; 3]| match effect {
        MONO => [y, 128.0, 128.0],
        SEPIA | SEPIA_01 => [y, 114.0, 144.0],
        NEGATIVE => [255.0 - y, 255.0 - u, 255.0 - v],
        NEGATIVE_FILM => [255.0 - y, u, v],
        _ => [y, u, v],
    };
    let mut bytes = Vec::with_capacity(pixels.len() * 2);
    if format == RGB565 {
        for &pixel in pixels {
            // the effect back from Y, U and V, when there's one
            let [r, g, b] = if effect == 0 {
                pixel
            } else {
                let [y, u, v] = colored(yuv(pixel));
                let (u, v) = (u - 128.0, v - 128.0);
                [byte(y + 1.402 * v), byte(y - 0.344 * u - 0.714 * v), byte(y + 1.772 * u)]
            };
            let value = (r as u16 >> 3) << 11 | (g as u16 >> 2) << 5 | b as u16 >> 3;
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        return bytes;
    }
    // Y each, U and V shared by the pair, Y U Y V
    for row in pixels.chunks(width.max(1)) {
        for pair in row.chunks(2) {
            let first = colored(yuv(pair[0]));
            let second = pair.get(1).map_or(first, |&pixel| colored(yuv(pixel)));
            bytes.extend_from_slice(&[
                byte(first[0]),
                byte((first[1] + second[1]) / 2.0),
                byte(second[0]),
                byte((first[2] + second[2]) / 2.0),
            ]);
        }
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{Fit, Picture, Still};
    use crate::services::tests::system_with_thread;
    use crate::services::{handle_request, Target};
    use std::sync::Arc;

    /// what the console answers for transfer sizes, titles take them as is
    #[test]
    fn transfers_are_sized_as_the_console_sizes_them() {
        assert_eq!(max_lines(640, 480), Some(4));
        assert_eq!(max_bytes(640, 480), Some(2560));
        // a 400 wide line is never a whole number of transfer units in the lines that fit, only bytes are
        assert_eq!(max_lines(400, 240), None);
        assert_eq!(max_bytes(400, 240), Some(2560));
        assert_eq!(max_lines(3, 3), None);
    }

    /// white is max Y w/ U and V centered, red in RGB565 is all red
    #[test]
    fn colors_come_out_in_the_formats() {
        let yuv = encode(&[[255; 3], [255; 3]], 2, YUV422, 0);
        assert_eq!(yuv, [255, 128, 255, 128]);
        let rgb = encode(&[[255, 0, 0]], 1, RGB565, 0);
        assert_eq!(u16::from_le_bytes([rgb[0], rgb[1]]), 0xF800);
    }

    /// a title that activates the outer camera, starts capturing and asks for a picture
    /// gets the frontend's at its size and format after a frame, plus the event
    #[test]
    fn a_picture_lands_in_the_title_buffer() {
        let (mut system, buffer) = system_with_thread();
        // a red picture
        let picture = Picture { width: 2, height: 2, rgb: [255, 0, 0].repeat(4), fit: Fit::Fill };
        system.camera = Some(Box::new(Still(Arc::new(picture))));
        let request = |system: &mut System, words: &[u32]| {
            for (i, word) in words.iter().enumerate() {
                buffer.set(&mut system.memory, i as u32, *word);
            }
            handle_request(system, Target::service("cam:u".into(), 0));
            buffer.get(&mut system.memory, 1)
        };
        // Activate(outer right), SetSize(QQVGA), SetOutputFormat(RGB565)
        assert_eq!(request(&mut system, &[0x0013_0040, 1]), 0);
        assert_eq!(request(&mut system, &[0x001F_00C0, 1, 2, 1]), 0);
        assert_eq!(request(&mut system, &[0x0025_00C0, 1, RGB565 as u32, 1]), 0);
        let destination = 0x0800_0000;
        let page = system.memory.phys.allocate(crate::memory::MemoryRegion::Base, 0x10000).unwrap();
        system.memory.map(destination, page.addr, 0x10000, crate::memory::Permission::RW, crate::memory::MemoryState::Private);
        // StartCapture(port 1), SetReceiving(destination, port 1, the size)
        assert_eq!(request(&mut system, &[0x0001_0040, 1]), 0);
        assert_eq!(request(&mut system, &[0x0007_0102, destination, 1, 160 * 120 * 2, 0x2800, 0, 0xFFFF8001]), 0);
        // IsFinishedReceiving, not yet
        assert_eq!((request(&mut system, &[0x0008_0040, 1]), buffer.get(&mut system.memory, 2)), (0, 0));
        system.cpu.cycles += frame_ticks(0);
        tick(&mut system);
        assert_eq!((request(&mut system, &[0x0008_0040, 1]), buffer.get(&mut system.memory, 2)), (0, 1));
        let mut corner = [0u8; 2];
        system.memory.read_bytes(destination + (160 * 120 - 1) * 2, &mut corner);
        assert_eq!(u16::from_le_bytes(corner), 0xF800);
    }
}
