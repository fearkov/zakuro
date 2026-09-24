//! hid:USER, buttons, circle pad and touch screen.


use crate::kernel::ipc::{CommandBuffer, Descriptor, Header};
use crate::kernel::object::{KObject, SharedMemory};
use crate::kernel::sync::ResetType;
use crate::System;

pub const SHARED_MEMORY_SIZE: u32 = 0x2B0;

/// offsets of the four ring buffers inside the shared block.
const PAD_BASE: u32 = 0x00;
const TOUCH_BASE: u32 = 0xA8;
const ACCELEROMETER_BASE: u32 = 0x108;
const GYROSCOPE_BASE: u32 = 0x168;

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
#[derive(Debug, Clone, Copy, Default)]
pub struct InputState {
    pub buttons: PadState,
    /// circle pad, -1.0 to 1.0 on each axis.
    pub circle_x: f32,
    pub circle_y: f32,
    /// touch position in screen pixels, when touched.
    pub touch: Option<(u16, u16)>,
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

/// writes the header every HID section starts with, two timestamps, then the
/// index of the sample just written.
fn write_section_header(system: &mut System, section: u32, index: u32, tick: u64) {
    let tick = tick & 0x7FFF_FFFF_FFFF_FFFF;
    put32(system, section, tick as u32);
    put32(system, section + 4, (tick >> 32) as u32);
    // the second timestamp is the previous update's, so the two differ.
    let previous = tick.saturating_sub(1);
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
    write_section_header(system, base + PAD_BASE, next, tick);

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
    write_section_header(system, base + TOUCH_BASE, touch_next, tick);
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

    // accelerometer and gyroscope keep a flat, resting sample so that titles
    // reading them see something plausible rather than noise.
    put32(system, base + ACCELEROMETER_BASE, 0);
    put32(system, base + GYROSCOPE_BASE, 0);

    let objects = system.services.hid.event_objects.clone();
    for object in objects.iter().take(2) {
        system.kernel.signal_event(*object);
    }
}
