//! ir:rst, the New 3DS's ZL, ZR and C stick, a title reads them from a block laid out like HID's pad section, updated as often as it asks

use crate::kernel::ipc::{CommandBuffer, Descriptor, Header};
use crate::kernel::object::{KObject, ObjectId, SharedMemory};
use crate::kernel::sync::ResetType;
use crate::kernel::thread::CPU_CLOCK_HZ;
use crate::services::hid::{self, PadState};
use crate::System;

/// the block, a page, the layout takes 0x98 bytes of it
const SHARED_MEMORY_SIZE: u32 = 0x1000;
/// where the ring of entries starts, past the ticks and the index
const ENTRIES: u32 = 0x18;
const ENTRY_COUNT: u32 = 8;
/// how far the C stick reports each way
const C_STICK_RANGE: f32 = 0x9C as f32;

#[derive(Default, serde::Serialize, serde::Deserialize)]
pub struct IrRstState {
    /// the block and its physical addr, plus the event each update signals, made when a title first asks (maybe b4 it initializes)
    memory: Option<(ObjectId, u32)>,
    event: Option<ObjectId>,
    /// ticks between updates, none while not initialized
    period: Option<u64>,
    next_update: u64,
    /// the entry the next update writes
    next: u32,
    previous: PadState,
    /// the tick entry 0 was last written at, and the one b4
    reset_ticks: (u64, u64),
}

pub fn handle(system: &mut System, buffer: &CommandBuffer, header: Header) -> bool {
    let command = header.command_id();
    match command {
        // GetHandles => the block and the update event
        0x0001 => {
            let (memory, event) = resources(system);
            let kernel = &mut system.kernel;
            let memory = kernel.handles.create(&mut kernel.objects, memory, "ir:rst shared memory");
            let event = kernel.handles.create(&mut kernel.objects, event, "ir:rst update event");
            buffer.set(&mut system.memory, 0, Header::new(command, 1, 3).0);
            buffer.set(&mut system.memory, 1, 0);
            buffer.set(&mut system.memory, 2, Descriptor::handles(2));
            buffer.set(&mut system.memory, 3, memory);
            buffer.set(&mut system.memory, 4, event);
            true
        }
        // Initialize(update period in ms, raw C stick)
        0x0002 => {
            let milliseconds = buffer.get(&mut system.memory, 1).max(1) as u64;
            resources(system);
            let period = milliseconds * CPU_CLOCK_HZ / 1000;
            let state = &mut system.services.ir_rst;
            state.period = Some(period);
            state.next_update = system.cpu.cycles + period;
            state.next = 0;
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // Shutdown
        0x0003 => {
            system.services.ir_rst.period = None;
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // what the console reads of the C stick's own chip, nothing titles act on
        0x0004..=0x0009 => {
            buffer.reply(&mut system.memory, command, &[0, 0]);
            true
        }
        _ => false,
    }
}

/// the block and the event, made the first time they're asked for
fn resources(system: &mut System) -> (ObjectId, ObjectId) {
    if let (Some((memory, _)), Some(event)) = (system.services.ir_rst.memory, system.services.ir_rst.event) {
        return (memory, event);
    }
    let block = system
        .memory
        .phys
        .allocate(crate::memory::MemoryRegion::Base, SHARED_MEMORY_SIZE)
        .expect("ir:rst shared memory");
    system.memory.zero_physical(block.addr, SHARED_MEMORY_SIZE);
    let kernel = &mut system.kernel;
    let memory = kernel.objects.insert(KObject::SharedMemory(SharedMemory {
        name: "IRRST".into(),
        address: 0,
        size: SHARED_MEMORY_SIZE,
        paddr: block.addr,
        mapped_at: None,
    }));
    let event = kernel.objects.insert(KObject::Event(crate::kernel::sync::Event::new(ResetType::OneShot, "IRRST:Update")));
    // the service keeps both, a title closing its handles must not take
    // them away
    kernel.objects.add_ref(memory);
    kernel.objects.add_ref(event);
    system.services.ir_rst.memory = Some((memory, block.addr));
    system.services.ir_rst.event = Some(event);
    (memory, event)
}

/// writes an entry when one is due, the buttons and stick as the frontend last gave them
pub fn tick(system: &mut System) {
    let state = &system.services.ir_rst;
    let (Some(period), Some((_, base)), Some(event)) = (state.period, state.memory, state.event) else { return };
    let now = system.cpu.cycles;
    if now < state.next_update {
        return;
    }
    // one update for however many periods went by, the latest input is all
    // there is to report
    let next_update = state.next_update + period * ((now - state.next_update) / period + 1);
    let input = system.services.hid.input;
    let current = (input.buttons & (PadState::ZL | PadState::ZR))
        | hid::directions(input.c_stick_x, input.c_stick_y, [
            PadState::C_STICK_RIGHT,
            PadState::C_STICK_LEFT,
            PadState::C_STICK_UP,
            PadState::C_STICK_DOWN,
        ]);
    let state = &mut system.services.ir_rst;
    let index = state.next;
    state.next = (index + 1) % ENTRY_COUNT;
    state.next_update = next_update;
    let previous = std::mem::replace(&mut state.previous, current);
    if index == 0 {
        state.reset_ticks = (now, state.reset_ticks.0);
    }
    let (latest, before) = state.reset_ticks;

    let axis = |value: f32| (value.clamp(-1.0, 1.0) * C_STICK_RANGE).round() as i16;
    let mut entry = [0u8; 0x10];
    entry[0..4].copy_from_slice(&current.bits().to_le_bytes());
    entry[4..8].copy_from_slice(&(current & !previous).bits().to_le_bytes());
    entry[8..12].copy_from_slice(&(previous & !current).bits().to_le_bytes());
    entry[12..14].copy_from_slice(&axis(input.c_stick_x).to_le_bytes());
    entry[14..16].copy_from_slice(&axis(input.c_stick_y).to_le_bytes());
    system.memory.write_physical(base + ENTRIES + index * 0x10, &entry);
    system.memory.write_physical(base, &latest.to_le_bytes());
    system.memory.write_physical(base + 8, &before.to_le_bytes());
    system.memory.write_physical(base + 0x10, &index.to_le_bytes());
    system.kernel.signal_event(event);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::hid::InputState;
    use crate::services::tests::system_with_thread;
    use crate::services::{handle_request, Target};

    /// once initialized, each period an entry goes in the block w/ ZL, ZR and the C stick's place and direction,
    /// plus what changed since the one b4
    #[test]
    fn the_c_stick_and_zl_and_zr_reach_the_block() {
        let (mut system, buffer) = system_with_thread();
        buffer.set(&mut system.memory, 0, Header::new(0x0001, 0, 0).0);
        handle_request(&mut system, Target::service("ir:rst".into(), 0));
        assert_eq!(buffer.get(&mut system.memory, 1), 0);
        let (_, base) = system.services.ir_rst.memory.unwrap();

        // Initialize(10 ms, not raw)
        buffer.set(&mut system.memory, 0, Header::new(0x0002, 2, 0).0);
        buffer.set(&mut system.memory, 1, 10);
        buffer.set(&mut system.memory, 2, 0);
        handle_request(&mut system, Target::service("ir:rst".into(), 0));

        system.set_input(InputState { buttons: PadState::ZR | PadState::A, c_stick_x: 1.0, c_stick_y: -0.25, ..InputState::default() });
        system.cpu.cycles += CPU_CLOCK_HZ / 100;
        tick(&mut system);
        let mut entry = [0u8; 0x10];
        system.memory.read_physical(base + ENTRIES, &mut entry);
        let word = |entry: &[u8], at: usize| u32::from_le_bytes(entry[at..at + 4].try_into().unwrap());
        assert_eq!(word(&entry, 0), (PadState::ZR | PadState::C_STICK_RIGHT).bits());
        assert_eq!(word(&entry, 4), word(&entry, 0));
        assert_eq!(word(&entry, 8), 0);
        assert_eq!(i16::from_le_bytes([entry[12], entry[13]]), 0x9C);
        assert_eq!(i16::from_le_bytes([entry[14], entry[15]]), -39);

        // not due yet, nothing more
        tick(&mut system);
        assert_eq!(system.services.ir_rst.next, 1);

        system.set_input(InputState::default());
        system.cpu.cycles += CPU_CLOCK_HZ / 100;
        tick(&mut system);
        let mut index = [0u8; 4];
        system.memory.read_physical(base + 0x10, &mut index);
        assert_eq!(u32::from_le_bytes(index), 1);
        system.memory.read_physical(base + ENTRIES + 0x10, &mut entry);
        assert_eq!(word(&entry, 0), 0);
        assert_eq!(word(&entry, 8), (PadState::ZR | PadState::C_STICK_RIGHT).bits());
    }
}
