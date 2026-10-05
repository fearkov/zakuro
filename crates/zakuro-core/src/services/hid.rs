//! hid:USER, buttons, circle pad, touch screen and motion sensors.


use crate::kernel::ipc::{CommandBuffer, Descriptor, Header};
use crate::kernel::object::{KObject, SharedMemory};
use crate::kernel::sync::ResetType;
use crate::System;

pub const SHARED_MEMORY_SIZE: u32 = 0x2B0;

/// offsets of the four ring buffers inside the shared block.
const PAD_BASE: u32 = 0x00;
const TOUCH_BASE: u32 = 0xA8;
const ACCELEROMETER_BASE: u32 = 0x108;
const GYROSCOPE_BASE: u32 = 0x158;

/// where the motion sensors' samples start in their sections, past the
/// header and the raw sample, and how many each keeps.
const MOTION_ENTRIES: u32 = 0x20;
const ACCELEROMETER_SAMPLES: u32 = 8;
const GYROSCOPE_SAMPLES: u32 = 32;
/// what the accelerometer reads for one g.
const ONE_G: f32 = 512.0;
/// what the gyroscope reads for one degree a second, what
/// GetGyroscopeLowRawToDpsCoefficient tells titles.
const GYROSCOPE_PER_DPS: f32 = 14.375;
/// how often the sensors are read, once a frame.
const SAMPLE_RATE: f32 = 60.0;

bitflags::bitflags! {
    /// button bits exactly as the hardware reports them.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct PadState: u32 {
        const A = 1 << 0;
        const B = 1 << 1;
        const SELECT = 1 << 2;
        const START = 1 << 3;
        const RIGHT = 1 << 4;
        const LEFT = 1 << 5;
        const UP = 1 << 6;
        const DOWN = 1 << 7;
        const R = 1 << 8;
        const L = 1 << 9;
        const X = 1 << 10;
        const Y = 1 << 11;
        /// set by the driver when the circle pad is pushed far enough in a
        /// direction, games use these instead of reading the axes.
        const CIRCLE_RIGHT = 1 << 28;
        const CIRCLE_LEFT = 1 << 29;
        const CIRCLE_UP = 1 << 30;
        const CIRCLE_DOWN = 1 << 31;
    }
}

/// what the frontend feeds in each frame.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct InputState {
    pub buttons: PadState,
    /// circle pad, -1.0 to 1.0 on each axis.
    pub circle_x: f32,
    pub circle_y: f32,
    /// touch position in screen pixels, when touched.
    pub touch: Option<(u16, u16)>,
    /// how the console is tilted away from upright, in radians, toward the
    /// right of the screen and toward its bottom, the way dragging the mouse
    /// tilts it. the sensors report it, and how fast it changes.
    pub tilt: [f32; 2],
}

#[derive(Default)]
pub struct HidState {
    pub shared_memory_handle: Option<u32>,
    pub shared_memory_address: u32,
    /// the physical pages behind the block.
    pub shared_memory_paddr: u32,
    pub events: Vec<u32>,
    pub event_objects: Vec<crate::kernel::object::ObjectId>,
    pub previous: PadState,
    pub pad_index: u32,
    pub touch_index: u32,
    /// how many times a title turned each sensor on and not yet off, they
    /// report only while on.
    pub accelerometer_users: u32,
    pub gyroscope_users: u32,
    pub accelerometer_index: u32,
    pub gyroscope_index: u32,
    /// the tick each section's ring last started over at, and the one
    /// before, for the pad, touch, accelerometer and gyroscope.
    pub reset_ticks: [(u64, u64); 4],
    /// how the console was turned a frame ago, for how fast it turns.
    pub orientation: Option<Quaternion>,
    /// the input the frontend gave last.
    pub input: InputState,
}

pub fn handle(system: &mut System, buffer: &CommandBuffer, header: Header) -> bool {
    match header.command_id() {
        // GetIPCHandles -> shared memory plus five event handles.
        0x000A => {
            ensure_resources(system);
            let handle = system.services.hid.shared_memory_handle.unwrap_or(0);
            let events = system.services.hid.events.clone();

            buffer.set(&mut system.memory, 0, Header::new(0x000A, 1, 7).0);
            buffer.set(&mut system.memory, 1, 0);
            buffer.set(&mut system.memory, 2, Descriptor::handles(6));
            buffer.set(&mut system.memory, 3, handle);
            for (i, event) in events.iter().enumerate() {
                buffer.set(&mut system.memory, 4 + i as u32, *event);
            }
            true
        }
        // EnableAccelerometer / DisableAccelerometer / EnableGyroscopeLow /
        // DisableGyroscopeLow
        0x0011..=0x0014 => {
            let hid = &mut system.services.hid;
            match header.command_id() {
                0x0011 => hid.accelerometer_users += 1,
                0x0012 => hid.accelerometer_users = hid.accelerometer_users.saturating_sub(1),
                0x0013 => hid.gyroscope_users += 1,
                _ => hid.gyroscope_users = hid.gyroscope_users.saturating_sub(1),
            }
            buffer.reply(&mut system.memory, header.command_id(), &[]);
            true
        }
        // GetGyroscopeLowRawToDpsCoefficient
        0x0015 => {
            buffer.reply(&mut system.memory, 0x0015, &[(14.375f32).to_bits()]);
            true
        }
        // GetGyroscopeLowCalibrateParam
        0x0016 => {
            buffer.reply(&mut system.memory, 0x0016, &[0, 0, 0, 0, 0]);
            true
        }
        // GetSoundVolume
        0x0017 => {
            buffer.reply(&mut system.memory, 0x0017, &[0x3F]);
            true
        }
        _ => false,
    }
}

fn ensure_resources(system: &mut System) {
    if system.services.hid.shared_memory_handle.is_some() {
        return;
    }

    let block = system
        .memory
        .phys
        .allocate(crate::memory::MemoryRegion::Base, SHARED_MEMORY_SIZE)
        .expect("HID shared memory");
    let object = system
        .kernel
        .objects
        .insert(KObject::SharedMemory(SharedMemory {
            name: "HID".into(),
            address: 0,
            size: SHARED_MEMORY_SIZE,
            paddr: block.addr,
            mapped_at: None,
        }));
    let handle = system
        .kernel
        .handles
        .create(&mut system.kernel.objects, object, "HID shared memory");
    system.services.hid.shared_memory_handle = Some(handle);

    for name in [
        "HID:PadOrTouch1",
        "HID:PadOrTouch2",
        "HID:Accelerometer",
        "HID:Gyroscope",
        "HID:DebugPad",
    ] {
        let (object, handle) = system.kernel.create_event(ResetType::OneShot, name);
        // signalled every frame, so kept whatever the title does with its
        // handle, a closed one can't hand the event's place to the next object
        system.kernel.objects.add_ref(object);
        system.services.hid.events.push(handle);
        system.services.hid.event_objects.push(object);
    }
}

/// offset of a section's ring of samples, past its header.
const SECTION_ENTRIES: u32 = 0x28;

/// writes a word into the shared block's physical pages.
fn put32(system: &mut System, paddr: u32, value: u32) {
    system.memory.write_physical(paddr, &value.to_le_bytes());
}

/// writes a halfword into the shared block's physical pages.
fn put16(system: &mut System, paddr: u32, value: u16) {
    system.memory.write_physical(paddr, &value.to_le_bytes());
}

/// the sections, in the order their reset ticks are kept.
const PAD: usize = 0;
const TOUCH: usize = 1;
const ACCELEROMETER: usize = 2;
const GYROSCOPE: usize = 3;

/// writes the header every HID section starts with, the ticks its ring last
/// started over at, the latest then the one before, and the index of the
/// sample just written. readers count new samples by the index and take a
/// changed tick as the ring having come round, so it moves only then.
fn write_section_header(system: &mut System, section: u32, which: usize, index: u32, tick: u64) {
    let ticks = &mut system.services.hid.reset_ticks[which];
    if index == 0 {
        *ticks = (tick & 0x7FFF_FFFF_FFFF_FFFF, ticks.0);
    }
    let (latest, previous) = *ticks;
    put32(system, section, latest as u32);
    put32(system, section + 4, (latest >> 32) as u32);
    put32(system, section + 8, previous as u32);
    put32(system, section + 12, (previous >> 32) as u32);
    put32(system, section + 0x10, index);
}

/// appends one sample to each ring buffer and signals the pad event.
pub fn update(system: &mut System, input: InputState) {
    let base = system.services.hid.shared_memory_paddr;
    if base == 0 {
        return;
    }

    let mut buttons = input.buttons;
    // the driver synthesises the circle-pad direction bits from the axes.
    const THRESHOLD: f32 = 0.5;
    buttons.set(PadState::CIRCLE_RIGHT, input.circle_x > THRESHOLD);
    buttons.set(PadState::CIRCLE_LEFT, input.circle_x < -THRESHOLD);
    buttons.set(PadState::CIRCLE_UP, input.circle_y > THRESHOLD);
    buttons.set(PadState::CIRCLE_DOWN, input.circle_y < -THRESHOLD);

    let previous = system.services.hid.previous;
    let additions = buttons.bits() & !previous.bits();
    let removals = !buttons.bits() & previous.bits();
    system.services.hid.previous = buttons;

    let index = system.services.hid.pad_index;
    let next = (index + 1) % 8;
    system.services.hid.pad_index = next;

    let tick = system.cpu.cycles;
    write_section_header(system, base + PAD_BASE, PAD, next, tick);
    // the buttons held now, past the header
    put32(system, base + PAD_BASE + 0x1C, buttons.bits());

    // circle pad range is roughly +-150 on hardware.
    let circle_x = (input.circle_x.clamp(-1.0, 1.0) * 150.0) as i16;
    let circle_y = (input.circle_y.clamp(-1.0, 1.0) * 150.0) as i16;

    let entry = base + PAD_BASE + SECTION_ENTRIES + next * 0x10;
    put32(system, entry, buttons.bits());
    put32(system, entry + 4, additions);
    put32(system, entry + 8, removals);
    put16(system, entry + 12, circle_x as u16);
    put16(system, entry + 14, circle_y as u16);

    // touch screen.
    let touch_index = system.services.hid.touch_index;
    let touch_next = (touch_index + 1) % 8;
    system.services.hid.touch_index = touch_next;
    write_section_header(system, base + TOUCH_BASE, TOUCH, touch_next, tick);
    // the touch section's entries start earlier than the pad's, its header
    // has no circle-pad fields to make room for.
    let touch_entry = base + TOUCH_BASE + 0x20 + touch_next * 8;
    match input.touch {
        Some((x, y)) => {
            put16(system, touch_entry, x);
            put16(system, touch_entry + 2, y);
            put32(system, touch_entry + 4, 1);
        }
        None => {
            put16(system, touch_entry, 0);
            put16(system, touch_entry + 2, 0);
            put32(system, touch_entry + 4, 0);
        }
    }

    let objects = system.services.hid.event_objects.clone();
    for object in objects.iter().take(2) {
        system.kernel.signal_event(*object);
    }

    // the sensors report the tilt, and how fast it changes, kept track of
    // whether or not they are on. titles that turn one on wait for its event
    let (gravity, rate) = motion(&mut system.services.hid, input.tilt);
    let accelerometer = gravity.map(|g| (g * ONE_G).round().clamp(-32768.0, 32767.0) as i16);
    let gyroscope = rate.map(|r| (r * GYROSCOPE_PER_DPS).round().clamp(-32768.0, 32767.0) as i16);
    let hid = &system.services.hid;
    if hid.accelerometer_users > 0 {
        let index = (hid.accelerometer_index + 1) % ACCELEROMETER_SAMPLES;
        system.services.hid.accelerometer_index = index;
        motion_sample(system, base + ACCELEROMETER_BASE, ACCELEROMETER, index, tick, accelerometer);
        if let Some(&event) = objects.get(2) {
            system.kernel.signal_event(event);
        }
    }
    let hid = &system.services.hid;
    if hid.gyroscope_users > 0 {
        let index = (hid.gyroscope_index + 1) % GYROSCOPE_SAMPLES;
        system.services.hid.gyroscope_index = index;
        motion_sample(system, base + GYROSCOPE_BASE, GYROSCOPE, index, tick, gyroscope);
        if let Some(&event) = objects.get(3) {
            system.kernel.signal_event(event);
        }
    }
}

/// a rotation, w then x, y and z.
pub type Quaternion = [f32; 4];

fn multiply([aw, ax, ay, az]: Quaternion, [bw, bx, by, bz]: Quaternion) -> Quaternion {
    [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]
}

fn inverse([w, x, y, z]: Quaternion) -> Quaternion {
    [w, -x, -y, -z]
}

/// v turned by q.
fn rotate(q: Quaternion, [x, y, z]: [f32; 3]) -> [f32; 3] {
    let [_, x, y, z] = multiply(multiply(q, [0.0, x, y, z]), inverse(q));
    [x, y, z]
}

/// what the accelerometer and the gyroscope read with the console tilted so,
/// gravity in g and how fast it turns in degrees a second, both the
/// console's way round. the console turns about the axis across the
/// direction of the tilt, and lies upright with gravity down its y axis, as
/// in the motion emulation of Citra, which titles were tried with.
fn motion(hid: &mut HidState, [x, y]: [f32; 2]) -> ([f32; 3], [f32; 3]) {
    let angle = x.hypot(y);
    let q = if angle > 0.0 {
        let (sin, cos) = (angle / 2.0).sin_cos();
        [cos, -y / angle * sin, 0.0, x / angle * sin]
    } else {
        [1.0, 0.0, 0.0, 0.0]
    };
    let previous = hid.orientation.replace(q).unwrap_or(q);
    let inverse = inverse(q);
    let gravity = rotate(inverse, [0.0, -1.0, 0.0]);
    // the derivative of q, twice, against q, turns a frame's change into
    // radians a second about the axes of the world
    let change = [q[0] - previous[0], q[1] - previous[1], q[2] - previous[2], q[3] - previous[3]];
    let [_, rx, ry, rz] = multiply(change, inverse);
    let degrees = 2.0 * SAMPLE_RATE * 180.0 / std::f32::consts::PI;
    let rate = rotate(inverse, [rx * degrees, ry * degrees, rz * degrees]);
    (gravity, rate)
}

/// writes a motion sensor's sample as both its raw one and entry index.
fn motion_sample(system: &mut System, section: u32, which: usize, index: u32, tick: u64, sample: [i16; 3]) {
    write_section_header(system, section, which, index, tick);
    for at in [section + 0x18, section + MOTION_ENTRIES + index * 6] {
        for (i, value) in sample.iter().enumerate() {
            put16(system, at + i as u32 * 2, *value as u16);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        a.iter().zip(b).all(|(a, b)| (a - b).abs() < 1e-3)
    }

    /// upright and still, gravity is down the console's y axis and nothing
    /// turns.
    #[test]
    fn untilted_the_console_reads_upright_and_still() {
        let mut hid = HidState::default();
        let (gravity, rate) = motion(&mut hid, [0.0, 0.0]);
        assert!(close(gravity, [0.0, -1.0, 0.0]));
        assert!(close(rate, [0.0, 0.0, 0.0]));
    }

    /// tilted a quarter turn, gravity leaves the y axis for the one the tilt
    /// went toward, and a steady tilt does not turn.
    #[test]
    fn a_tilt_moves_gravity_and_holding_it_stops_the_turning() {
        let quarter = std::f32::consts::FRAC_PI_2;
        let mut hid = HidState::default();
        motion(&mut hid, [quarter, 0.0]);
        let (gravity, rate) = motion(&mut hid, [quarter, 0.0]);
        assert!(close(gravity, [-1.0, 0.0, 0.0]) || close(gravity, [1.0, 0.0, 0.0]), "{gravity:?}");
        assert!(close(rate, [0.0, 0.0, 0.0]));
        let mut hid = HidState::default();
        motion(&mut hid, [0.0, quarter]);
        let (gravity, _) = motion(&mut hid, [0.0, quarter]);
        assert!(close(gravity, [0.0, 0.0, -1.0]) || close(gravity, [0.0, 0.0, 1.0]), "{gravity:?}");
    }

    /// tilting by a little in a frame turns at that much sixty times a
    /// second, about the axis across the tilt.
    #[test]
    fn tilting_turns_as_fast_as_the_tilt_changes() {
        let mut hid = HidState::default();
        motion(&mut hid, [0.0, 0.0]);
        let step = 0.01f32;
        let (_, rate) = motion(&mut hid, [step, 0.0]);
        let expected = step.to_degrees() * SAMPLE_RATE;
        let speed = rate.iter().map(|r| r * r).sum::<f32>().sqrt();
        assert!((speed - expected).abs() < expected * 0.01, "{speed} against {expected}");
        assert!(rate[0].abs() < 1e-3 && rate[1].abs() < 1e-3, "{rate:?}");
    }
}
