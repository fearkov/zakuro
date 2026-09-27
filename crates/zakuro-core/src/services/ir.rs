//! ir:USER, the infrared link titles look for the Circle Pad Pro over.
//! nothing is ever attached, so every attempt to connect ends disconnected,
//! but the events a title waits on exist and the shared block says as much.

use crate::kernel::ipc::{CommandBuffer, Descriptor, Header};
use crate::kernel::object::{KObject, ObjectId};
use crate::kernel::sync::ResetType;
use crate::System;

// where the shared block keeps the link's state
const CONNECTION_STATUS: u32 = 0x08;
const TRYING_TO_CONNECT: u32 = 0x09;
const CONNECTED: u32 = 0x0C;
const INITIALIZED: u32 = 0x0E;
/// the header, then where the receive buffer's packets start and end.
const HEADER_SIZE: usize = 0x20;

#[derive(Default)]
pub struct IrState {
    /// the block a title shares for the link's state and data.
    pub memory: Option<ObjectId>,
    /// the receive, send and connection status events.
    events: Option<[ObjectId; 3]>,
}

/// the connection status event, after the receive and send ones.
const CONNECTION: usize = 2;

pub fn handle(system: &mut System, buffer: &CommandBuffer, header: Header) -> bool {
    let command = header.command_id();
    match command {
        // InitializeIrNop / InitializeIrNopShared(sizes and counts, baud
        // rate, block), the block's handle comes last
        0x0001 | 0x0018 => {
            let normal = (header.0 >> 6) & 0x3F;
            let handle = buffer.get(&mut system.memory, normal + 2);
            system.services.ir.memory = system.kernel.resolve(handle);
            write(system, &[0; HEADER_SIZE], 0);
            write(system, &[1], INITIALIZED);
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // FinalizeIrNop
        0x0002 => {
            system.services.ir.memory = None;
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // WaitConnection / RequireConnection / AutoConnection /
        // AnyConnection, there is nothing to connect to, so the attempt is
        // over as soon as it starts, and Disconnect
        0x0005..=0x0009 => {
            write(system, &[0], CONNECTION_STATUS);
            write(system, &[0], TRYING_TO_CONNECT);
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
        // ClearReceiveBuffer / ClearSendBuffer / SendIrNop / SendIrNopLarge /
        // ReleaseReceivedData / SetOwnMachineId, with nobody on the other
        // end there is nothing to clear, send or release
        0x0003 | 0x0004 | 0x000D | 0x000E | 0x0019 | 0x001A => {
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // GetLatestReceiveErrorResult / GetLatestSendErrorResult /
        // GetConnectionStatus / GetTryingToConnectStatus / GetConnectionRole
        0x0011..=0x0014 | 0x0017 => {
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

/// the events, made the first time a title asks for one.
fn events(system: &mut System) -> [ObjectId; 3] {
    if let Some(events) = system.services.ir.events {
        return events;
    }
    let names = ["ir:USER:Receive", "ir:USER:Send", "ir:USER:ConnectionStatus"];
    let events = names.map(|name| {
        let object = system.kernel.objects.insert(KObject::Event(crate::kernel::sync::Event::new(ResetType::OneShot, name)));
        // the service holds on to them, a title closing its handles must
        // not take them away
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

/// writes into the shared block, when there is one.
fn write(system: &mut System, bytes: &[u8], offset: u32) {
    let Some(object) = system.services.ir.memory else { return };
    if let Some(KObject::SharedMemory(block)) = system.kernel.objects.get(object) {
        if (offset as usize + bytes.len()) as u32 <= block.size {
            let paddr = block.paddr + offset;
            system.memory.write_physical(paddr, bytes);
        }
    }
}
