//! ir:USER, the infrared link titles look for the Circle Pad Pro over
//! a title asking for one finds one, the C stick, ZL and ZR stand in for its stick and buttons like on a New 3DS
//! it answers the title's requests and reports as often as asked, in packets put in the receive buffer of the block the title shares
//! anything else a title connects to is never there

use crate::kernel::ipc::{CommandBuffer, Descriptor, Header};
use crate::kernel::object::{KObject, ObjectId};
use crate::kernel::sync::ResetType;
use crate::kernel::thread::CPU_CLOCK_HZ;
use crate::services::hid::PadState;
use crate::System;
use zakuro_common::result::{Level, Module, ResultCode, Summary};

// where the shared block keeps the link's state
const CONNECTION_STATUS: u32 = 0x08;
const TRYING_TO_CONNECT: u32 = 0x09;
const CONNECTION_ROLE: u32 = 0x0A;
const MACHINE_ID: u32 = 0x0B;
const CONNECTED: u32 = 0x0C;
const NETWORK_ID: u32 = 0x0D;
const INITIALIZED: u32 = 0x0E;
/// the header, then where the receive buffer's packets start and end
const HEADER_SIZE: usize = 0x20;
/// where the receive buffer's packets start and end, and the buffer, its packet table first, the packets' bytes after
const RECEIVE_INFO: u32 = 0x10;
const RECEIVE_BUFFER: u32 = 0x20;

/// the device a title connects to for a Circle Pad Pro
const CIRCLE_PAD_PRO: u32 = 1;
/// what a connected link reads in the block's status bytes
const STATUS_CONNECTED: u8 = 2;
const ROLE: u8 = 2;

/// sending with nothing connected
const NOT_CONNECTED: ResultCode = ResultCode::new(Level::Status, Summary::InvalidState, Module::Ir, 13);

/// what a Circle Pad Pro is asked, the first byte of a request
const READ_INPUT: u8 = 1;
const READ_CALIBRATION: u8 = 2;
/// what it answers with, the first byte of a response
const INPUT: u8 = 0x10;
const CALIBRATION: u8 = 0x11;

/// the stick's center and how far from it a full push goes
/// the calibration says take the center off, scale by 0.625 and divide by 8, giving the 0x9C each way the circle pad reports
const STICK_CENTER: f32 = 2048.0;
const STICK_REACH: f32 = 0x9C as f32 * 8.0 / CALIBRATION_SCALE;
const CALIBRATION_SCALE: f32 = 0.625;

#[derive(Default, serde::Serialize, serde::Deserialize)]
pub struct IrState {
    /// the block a title shares for the link's state and data
    pub memory: Option<ObjectId>,
    /// the receive, send and connection status events
    events: Option<[ObjectId; 3]>,
    /// the receive buffer, when the block holds its state too
    receive: Option<Receive>,
    connected: bool,
    /// ticks between its reports, none till a title asks for them, and when the next is due
    period: Option<u64>,
    next_report: u64,
}

/// the receive buffer's room and the packets in it
#[derive(Debug, Default, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
struct Receive {
    /// its size (packet table included) and how many the table holds
    size: u32,
    slots: u32,
    /// the first packet in the table, the one past the last, and how many
    begin: u32,
    end: u32,
    count: u32,
}

/// the connection status event, after the receive and send ones
const RECEIVED: usize = 0;
const SENT: usize = 1;
const CONNECTION: usize = 2;

pub fn handle(system: &mut System, buffer: &CommandBuffer, header: Header) -> bool {
    let command = header.command_id();
    match command {
        // InitializeIrNop / InitializeIrNopShared(sizes and counts, baud rate, block), the block's handle comes last
        // only the shared form keeps the receive buffer's state in the block, where a title reads what came in
        0x0001 | 0x0018 => {
            let normal = (header.0 >> 6) & 0x3F;
            let handle = buffer.get(&mut system.memory, normal + 2);
            let ir = &mut system.services.ir;
            ir.memory = system.kernel.resolve(handle);
            ir.connected = false;
            ir.period = None;
            ir.receive = (command == 0x0018).then(|| Receive {
                size: buffer.get(&mut system.memory, 2),
                slots: buffer.get(&mut system.memory, 3),
                ..Receive::default()
            });
            write(system, &[0; HEADER_SIZE], 0);
            write(system, &[1], INITIALIZED);
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // FinalizeIrNop
        0x0002 => {
            let ir = &mut system.services.ir;
            ir.memory = None;
            ir.receive = None;
            ir.connected = false;
            ir.period = None;
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // RequireConnection(device), a Circle Pad Pro is always there
        0x0006 if buffer.get(&mut system.memory, 1) & 0xFF == CIRCLE_PAD_PRO && system.services.ir.receive.is_some() => {
            log::debug!("ir:USER: a Circle Pad Pro is connected");
            system.services.ir.connected = true;
            system.services.ir.period = None;
            write(system, &[STATUS_CONNECTED], CONNECTION_STATUS);
            write(system, &[0], TRYING_TO_CONNECT);
            write(system, &[ROLE], CONNECTION_ROLE);
            write(system, &[1], CONNECTED);
            signal(system, CONNECTION);
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // WaitConnection / RequireConnection / AutoConnection / AnyConnection,
        // w/ nothing else to connect to the attempt is over as soon as it starts, and Disconnect
        0x0005..=0x0009 => {
            log::debug!("ir:USER: command 0x{command:04X}, nothing is connected");
            system.services.ir.connected = false;
            system.services.ir.period = None;
            write(system, &[0], CONNECTION_STATUS);
            write(system, &[0], TRYING_TO_CONNECT);
            write(system, &[0], CONNECTION_ROLE);
            write(system, &[0], CONNECTED);
            signal(system, CONNECTION);
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // GetReceiveEvent / GetSendEvent / GetConnectionStatusEvent
        0x000A..=0x000C => {
            let object = events(system)[command as usize - 0x000A];
            let handle = system.kernel.handles.create(&mut system.kernel.objects, object, "ir:USER event");
            buffer.set(&mut system.memory, 0, Header::new(command, 1, 2).0);
            buffer.set(&mut system.memory, 1, 0);
            buffer.set(&mut system.memory, 2, Descriptor::handles(1));
            buffer.set(&mut system.memory, 3, handle);
            true
        }
        // SendIrNop / SendIrNopLarge(size, buffer), a request for the
        // Circle Pad Pro
        0x000D | 0x000E => {
            if !system.services.ir.connected {
                buffer.reply_error(&mut system.memory, command, NOT_CONNECTED.0);
                return true;
            }
            let size = buffer.get(&mut system.memory, 1).min(0x1000);
            let pointer = buffer.get(&mut system.memory, 3);
            let mut request = vec![0; size as usize];
            system.memory.read_bytes(pointer, &mut request);
            answer(system, &request);
            signal(system, SENT);
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // ClearReceiveBuffer
        0x0003 => {
            if let Some(receive) = &mut system.services.ir.receive {
                *receive = Receive { size: receive.size, slots: receive.slots, ..Receive::default() };
            }
            write_receive_info(system);
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // ReleaseReceivedData(count), the title's done w/ the oldest packets
        0x0019 => {
            let count = buffer.get(&mut system.memory, 1);
            if let Some(receive) = &mut system.services.ir.receive {
                let count = count.min(receive.count);
                receive.count -= count;
                receive.begin = (receive.begin + count) % receive.slots.max(1);
            }
            write_receive_info(system);
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // SetOwnMachineId(id)
        0x001A => {
            let id = buffer.get(&mut system.memory, 1) as u8;
            write(system, &[id], MACHINE_ID);
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // ClearSendBuffer, nothing waits to be sent
        0x0004 => {
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // GetConnectionStatus / GetConnectionRole
        0x0013 | 0x0017 => {
            let connected = system.services.ir.connected;
            let value = if !connected { 0 } else if command == 0x0013 { STATUS_CONNECTED } else { ROLE };
            buffer.reply(&mut system.memory, command, &[value as u32]);
            true
        }
        // GetLatestReceiveErrorResult / GetLatestSendErrorResult / GetTryingToConnectStatus
        0x0011 | 0x0012 | 0x0014 => {
            buffer.reply(&mut system.memory, command, &[0]);
            true
        }
        // GetReceiveSizeFreeAndUsed / GetSendSizeFreeAndUsed
        0x0015 | 0x0016 => {
            buffer.reply(&mut system.memory, command, &[0, 0]);
            true
        }
        _ => false,
    }
}

/// what the Circle Pad Pro does w/ a request
fn answer(system: &mut System, request: &[u8]) {
    match request.first() {
        // ReadInput(period in ms, ?), reports from now on at that period, zero stops them
        Some(&READ_INPUT) if request.len() >= 2 => {
            let milliseconds = request[1] as u64;
            log::debug!("ir:USER: the Circle Pad Pro reports every {milliseconds} ms");
            let ir = &mut system.services.ir;
            ir.period = (milliseconds > 0).then_some(milliseconds * CPU_CLOCK_HZ / 1000);
            ir.next_report = system.cpu.cycles + ir.period.unwrap_or(0);
        }
        // ReadCalibrationData(response time, offset, size), offset and size are 16 bit
        Some(&READ_CALIBRATION) if request.len() >= 6 => {
            let offset = u16::from_le_bytes([request[2], request[3]]);
            let size = u16::from_le_bytes([request[4], request[5]]);
            log::debug!("ir:USER: the Circle Pad Pro's calibration, 0x{size:X} bytes at 0x{offset:X}");
            let data = calibration();
            let (start, length) = ((offset & !0xF) as usize, (size & !0xF) as usize);
            if start + length > data.len() {
                log::warn!("ir:USER: the Circle Pad Pro's calibration has nothing at 0x{offset:X}+0x{size:X}");
                return;
            }
            let mut response = vec![CALIBRATION];
            response.extend_from_slice(&offset.to_le_bytes());
            response.extend_from_slice(&size.to_le_bytes());
            response.extend_from_slice(&data[start..start + length]);
            receive(system, &response);
        }
        _ => log::warn!("ir:USER: the Circle Pad Pro does not know the request {request:02X?}"),
    }
}

/// the Circle Pad Pro's calibration, four identical sets, the stick's center and how its values scale, each w/ its checksum
fn calibration() -> [u8; 0x40] {
    let mut data = [0u8; 0x40];
    for set in data.chunks_mut(0x10) {
        // the center, two 12 bit fields
        let center = 0x800u32 | 0x800 << 12;
        set[1..4].copy_from_slice(&center.to_le_bytes()[..3]);
        set[4..8].copy_from_slice(&CALIBRATION_SCALE.to_le_bytes());
        set[8..12].copy_from_slice(&CALIBRATION_SCALE.to_le_bytes());
        set[12..15].fill(0xFF);
        set[15] = crc8(&set[..15]);
    }
    data
}

/// reports what the stick and buttons do when a report's due
pub fn tick(system: &mut System) {
    let ir = &system.services.ir;
    let (true, Some(period)) = (ir.connected, ir.period) else { return };
    let now = system.cpu.cycles;
    if now < ir.next_report {
        return;
    }
    // one report for however many periods went by
    let next_report = ir.next_report + period * ((now - ir.next_report) / period + 1);
    system.services.ir.next_report = next_report;
    let report = report(&system.services.hid.input);
    receive(system, &report);
}

/// a Circle Pad Pro report, the stick's place in two 12 bit fields, then the battery and its buttons (set while not held)
fn report(input: &crate::services::hid::InputState) -> [u8; 6] {
    let axis = |value: f32| (STICK_CENTER + value.clamp(-1.0, 1.0) * STICK_REACH).round().clamp(0.0, 4095.0) as u32;
    let stick = axis(input.c_stick_x) | axis(input.c_stick_y) << 12;
    let [low, middle, high, _] = stick.to_le_bytes();
    // a full battery, and the Pro's own R, which the New 3DS has none of
    let mut buttons = 0x1F | 1 << 7;
    if !input.buttons.contains(PadState::ZL) {
        buttons |= 1 << 5;
    }
    if !input.buttons.contains(PadState::ZR) {
        buttons |= 1 << 6;
    }
    [INPUT, low, middle, high, buttons, 0]
}

/// puts what the Circle Pad Pro sent in the receive buffer as a packet and tells the title
fn receive(system: &mut System, payload: &[u8]) {
    let Some(mut state) = system.services.ir.receive else { return };
    let mut network = [0u8];
    read(system, &mut network, NETWORK_ID);
    let mut packet = vec![0xA5, network[0]];
    // the size in one byte below 0x40, in two past it
    if payload.len() < 0x40 {
        packet.push(payload.len() as u8);
    } else {
        packet.push(0x40 | (payload.len() >> 8) as u8);
        packet.push(payload.len() as u8);
    }
    packet.extend_from_slice(payload);
    packet.push(crc8(&packet));

    if state.count >= state.slots || state.slots == 0 {
        log::warn!("ir:USER: the receive buffer is full, a packet from the Circle Pad Pro is lost");
        return;
    }
    let table = state.slots * 8;
    let room = state.size.saturating_sub(table);
    let length = packet.len() as u32;
    // after the last packet, in the ring the bytes go round
    let offset = if state.count == 0 {
        0
    } else {
        let last = slot(system, (state.end + state.slots - 1) % state.slots);
        let first = slot(system, state.begin);
        let offset = (last.0 + last.1) % room.max(1);
        let free = (first.0 + room - offset) % room.max(1);
        if length > free {
            log::warn!("ir:USER: the receive buffer is full, a packet from the Circle Pad Pro is lost");
            return;
        }
        offset
    };
    if length > room {
        return;
    }
    for (i, byte) in packet.iter().enumerate() {
        write(system, &[*byte], RECEIVE_BUFFER + table + (offset + i as u32) % room);
    }
    let mut entry = [0u8; 8];
    entry[..4].copy_from_slice(&offset.to_le_bytes());
    entry[4..].copy_from_slice(&length.to_le_bytes());
    write(system, &entry, RECEIVE_BUFFER + state.end * 8);
    state.end = (state.end + 1) % state.slots;
    state.count += 1;
    system.services.ir.receive = Some(state);
    write_receive_info(system);
    signal(system, RECEIVED);
}

/// where packet index of the table is in the buffer's bytes, and its length
fn slot(system: &mut System, index: u32) -> (u32, u32) {
    let mut entry = [0u8; 8];
    read(system, &mut entry, RECEIVE_BUFFER + index * 8);
    (u32::from_le_bytes(entry[..4].try_into().unwrap()), u32::from_le_bytes(entry[4..].try_into().unwrap()))
}

fn write_receive_info(system: &mut System) {
    let Some(receive) = system.services.ir.receive else { return };
    let mut info = [0u8; 0x10];
    info[..4].copy_from_slice(&receive.begin.to_le_bytes());
    info[4..8].copy_from_slice(&receive.end.to_le_bytes());
    info[8..12].copy_from_slice(&receive.count.to_le_bytes());
    write(system, &info, RECEIVE_INFO);
}

/// the CRC8 the packets and calibration end w/, x^8 + x^2 + x + 1
fn crc8(bytes: &[u8]) -> u8 {
    bytes.iter().fold(0u8, |crc, &byte| {
        (0..8).fold(crc ^ byte, |crc, _| if crc & 0x80 != 0 { (crc << 1) ^ 0x07 } else { crc << 1 })
    })
}

/// the events, made the first time a title asks for one
fn events(system: &mut System) -> [ObjectId; 3] {
    if let Some(events) = system.services.ir.events {
        return events;
    }
    let names = ["ir:USER:Receive", "ir:USER:Send", "ir:USER:ConnectionStatus"];
    let events = names.map(|name| {
        let object = system.kernel.objects.insert(KObject::Event(crate::kernel::sync::Event::new(ResetType::OneShot, name)));
        // the service keeps them, a title closing its handles must not take them away
        system.kernel.objects.add_ref(object);
        object
    });
    system.services.ir.events = Some(events);
    events
}

fn signal(system: &mut System, event: usize) {
    let object = events(system)[event];
    system.kernel.signal_event(object);
}

/// where the shared block's bytes are, when there's one
fn block(system: &System) -> Option<(u32, u32)> {
    let object = system.services.ir.memory?;
    match system.kernel.objects.get(object) {
        Some(KObject::SharedMemory(block)) => Some((block.paddr, block.size)),
        _ => None,
    }
}

/// writes into the shared block, when there's one
fn write(system: &mut System, bytes: &[u8], offset: u32) {
    let Some((paddr, size)) = block(system) else { return };
    if (offset as usize + bytes.len()) as u32 <= size {
        system.memory.write_physical(paddr + offset, bytes);
    }
}

/// reads from the shared block, zeroes when there's none
fn read(system: &mut System, bytes: &mut [u8], offset: u32) {
    bytes.fill(0);
    let Some((paddr, size)) = block(system) else { return };
    if (offset as usize + bytes.len()) as u32 <= size {
        system.memory.read_physical(paddr + offset, bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::object::SharedMemory;
    use crate::services::hid::InputState;
    use crate::services::tests::system_with_thread;
    use crate::services::{handle_request, Target};

    /// the checksum matches the one known Circle Pad Pro calibration data has
    #[test]
    fn the_checksum_is_the_one_the_calibration_has() {
        let set = [0x00, 0x00, 0x08, 0x80, 0x85, 0xEB, 0x11, 0x3F, 0x85, 0xEB, 0x11, 0x3F, 0xFF, 0xFF, 0xFF];
        assert_eq!(crc8(&set), 0xF5);
        let mut other = set;
        other[0] = 0xFF;
        assert_eq!(crc8(&other), 0x65);
    }

    /// a full push reads 0x9C each way thru the calibration, like the circle pad
    #[test]
    fn a_full_push_reads_as_far_as_the_circle_pad() {
        let input = InputState { c_stick_x: 1.0, c_stick_y: -1.0, buttons: PadState::ZL, ..InputState::default() };
        let report = report(&input);
        let stick = u32::from_le_bytes([report[1], report[2], report[3], 0]);
        let calibrated = |raw: u32| ((raw as f32 - 2048.0) * CALIBRATION_SCALE / 8.0).round();
        assert_eq!(calibrated(stick & 0xFFF), 156.0);
        assert_eq!(calibrated(stick >> 12), -156.0);
        // ZL held, ZR and R not
        assert_eq!(report[4] & 0xE0, 0xC0);
    }

    /// a title connecting to a Circle Pad Pro gets its calibration when it asks and its reports at the period it
    /// asks for, as packets in the receive buffer, and lets go of them
    #[test]
    fn a_circle_pad_pro_answers_and_reports() {
        let (mut system, buffer) = system_with_thread();
        let block = system.memory.phys.allocate(crate::memory::MemoryRegion::Base, 0x1000).unwrap();
        let object = system.kernel.objects.insert(KObject::SharedMemory(SharedMemory {
            name: "ir".into(),
            address: 0,
            size: 0x1000,
            paddr: block.addr,
            mapped_at: None,
        }));
        let handle = system.kernel.handles.create(&mut system.kernel.objects, object, "ir");
        let request = |system: &mut System, words: &[u32]| {
            for (i, word) in words.iter().enumerate() {
                buffer.set(&mut system.memory, i as u32, *word);
            }
            handle_request(system, Target::service("ir:USER".into(), 0));
            buffer.get(&mut system.memory, 1)
        };
        // InitializeIrNopShared(0x1000, receive 0x800 for 16, send 0x800 for 16, baud, block)
        assert_eq!(request(&mut system, &[0x0018_0182, 0x1000, 0x800, 16, 0x800, 16, 4, 0, handle]), 0);
        // RequireConnection(Circle Pad Pro)
        assert_eq!(request(&mut system, &[0x0006_0040, 1]), 0);
        let mut status = [0u8; 0x10];
        system.memory.read_physical(block.addr, &mut status);
        assert_eq!((status[8], status[0xC], status[0xE]), (2, 1, 1));

        // ReadCalibrationData(0x10 bytes at 0)
        let data = 0x0800_0000;
        let page = system.memory.phys.allocate(crate::memory::MemoryRegion::Base, 0x1000).unwrap();
        system.memory.map(data, page.addr, 0x1000, crate::memory::Permission::RW, crate::memory::MemoryState::Private);
        system.memory.write_bytes(data, &[READ_CALIBRATION, 0, 0, 0, 0x10, 0]);
        assert_eq!(request(&mut system, &[0x000D_0042, 6, (6 << 14) | 2, data]), 0);
        // ReadInput every 10 ms
        system.memory.write_bytes(data, &[READ_INPUT, 10, 0]);
        assert_eq!(request(&mut system, &[0x000D_0042, 3, (3 << 14) | 2, data]), 0);
        system.set_input(InputState { c_stick_x: -1.0, ..InputState::default() });
        system.cpu.cycles += CPU_CLOCK_HZ / 100;
        tick(&mut system);

        let mut info = [0u8; 12];
        system.memory.read_physical(block.addr + RECEIVE_INFO, &mut info);
        let word = |bytes: &[u8], at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        assert_eq!((word(&info, 0), word(&info, 4), word(&info, 8)), (0, 2, 2));
        let packet = |system: &mut System, index: u32| {
            let (offset, length) = slot(system, index);
            let mut bytes = vec![0u8; length as usize];
            system.memory.read_physical(block.addr + RECEIVE_BUFFER + 16 * 8 + offset, &mut bytes);
            bytes
        };
        let first = packet(&mut system, 0);
        // 0xA5, the network, the size, the payload and the checksum
        assert_eq!(&first[..4], &[0xA5, 0, 0x15, CALIBRATION]);
        assert_eq!(first.len(), 3 + 0x15 + 1);
        assert_eq!(crc8(&first[..first.len() - 1]), *first.last().unwrap());
        let second = packet(&mut system, 1);
        assert_eq!(&second[..4], &[0xA5, 0, 6, INPUT]);
        assert_eq!(u32::from_le_bytes([second[4], second[5], second[6], 0]) & 0xFFF, 51);

        // ReleaseReceivedData(1)
        assert_eq!(request(&mut system, &[0x0019_0040, 1]), 0);
        system.memory.read_physical(block.addr + RECEIVE_INFO, &mut info);
        assert_eq!((word(&info, 0), word(&info, 4), word(&info, 8)), (1, 2, 1));
    }
}
