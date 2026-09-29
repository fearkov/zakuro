//! nfc:u, the reader amiibo are read with. the reader is there and scans,
//! but no tag ever comes into range.

use crate::kernel::ipc::{CommandBuffer, Descriptor, Header};
use crate::kernel::object::{KObject, ObjectId};
use crate::kernel::sync::ResetType;
use crate::System;

/// the reader's states as GetTagState reports them.
const NOT_INITIALIZED: u32 = 0;
const NOT_SCANNING: u32 = 1;
const SCANNING: u32 = 2;

/// what CommunicationGetStatus reports once the reader is up.
const COMMUNICATION_READY: u32 = 2;

#[derive(Default)]
pub struct NfcState {
    tag_state: u32,
    /// the tag in range and out of range events.
    events: Option<[ObjectId; 2]>,
}

pub fn handle(system: &mut System, buffer: &CommandBuffer, header: Header) -> bool {
    let command = header.command_id();
    let reply = |system: &mut System, values: &[u32]| buffer.reply(&mut system.memory, command, values);
    match command {
        // Initialize
        0x0001 => {
            system.services.nfc.tag_state = NOT_SCANNING;
            reply(system, &[]);
        }
        // Shutdown
        0x0002 => {
            system.services.nfc.tag_state = NOT_INITIALIZED;
            reply(system, &[]);
        }
        // StartCommunication, StopCommunication and ResetTagScanState
        0x0003 | 0x0004 | 0x0008 => reply(system, &[]),
        // StartTagScanning
        0x0005 => {
            system.services.nfc.tag_state = SCANNING;
            reply(system, &[]);
        }
        // StopTagScanning
        0x0006 => {
            system.services.nfc.tag_state = NOT_SCANNING;
            reply(system, &[]);
        }
        // GetTagInRangeEvent and GetTagOutOfRangeEvent
        0x000B | 0x000C => {
            let object = events(system)[command as usize - 0x000B];
            let handle = system.kernel.handles.create(&mut system.kernel.objects, object, "nfc:u event");
            buffer.set(&mut system.memory, 0, Header::new(command, 1, 2).0);
            buffer.set(&mut system.memory, 1, 0);
            buffer.set(&mut system.memory, 2, Descriptor::handles(1));
            buffer.set(&mut system.memory, 3, handle);
        }
        // GetTagState
        0x000D => {
            let state = system.services.nfc.tag_state;
            reply(system, &[state]);
        }
        // CommunicationGetStatus
        0x000F => reply(system, &[COMMUNICATION_READY]),
        // CommunicationGetResult, how the last attempt to talk to the
        // reader went
        0x0012 => reply(system, &[0]),
        _ => return false,
    }
    true
}

/// the events, made the first time a title asks for one.
fn events(system: &mut System) -> [ObjectId; 2] {
    if let Some(events) = system.services.nfc.events {
        return events;
    }
    let events = ["nfc:u:TagInRange", "nfc:u:TagOutOfRange"].map(|name| {
        let object = system.kernel.objects.insert(KObject::Event(crate::kernel::sync::Event::new(ResetType::OneShot, name)));
        // the service keeps them, a title closing its handles must not take
        // them away
        system.kernel.objects.add_ref(object);
        object
    });
    system.services.nfc.events = Some(events);
    events
}
