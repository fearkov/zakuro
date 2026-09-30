//! cecd:u, StreetPass. the daemon is there and does what it is told, but
//! never meets another console.

use crate::kernel::ipc::{CommandBuffer, Descriptor, Header};
use crate::kernel::object::{KObject, ObjectId};
use crate::kernel::sync::ResetType;
use crate::System;

/// the daemon's states as GetCecdState reports them.
const WORKING: u32 = 0;
const IDLE: u32 = 1;

pub struct CecdState {
    state: u32,
    /// the info changed and state changed events.
    events: Option<[ObjectId; 2]>,
}

impl Default for CecdState {
    fn default() -> Self {
        CecdState { state: IDLE, events: None }
    }
}

pub fn handle(system: &mut System, buffer: &CommandBuffer, header: Header) -> bool {
    let command = header.command_id();
    let reply = |system: &mut System, values: &[u32]| buffer.reply(&mut system.memory, command, values);
    match command {
        // Start and Stop, a title waits for the state changed event after
        // either before it goes on
        0x000B | 0x000C => {
            system.services.cecd.state = if command == 0x000B { WORKING } else { IDLE };
            let changed = events(system)[1];
            system.kernel.signal_event(changed);
            reply(system, &[]);
        }
        // GetCecdState
        0x000E => {
            let state = system.services.cecd.state;
            reply(system, &[state]);
        }
        // GetCecInfoEventHandle and GetChangeStateEventHandle
        0x000F | 0x0010 => {
            let object = events(system)[command as usize - 0x000F];
            let handle = system.kernel.handles.create(&mut system.kernel.objects, object, "cecd:u event");
            buffer.set(&mut system.memory, 0, Header::new(command, 1, 2).0);
            buffer.set(&mut system.memory, 1, 0);
            buffer.set(&mut system.memory, 2, Descriptor::handles(1));
            buffer.set(&mut system.memory, 3, handle);
        }
        _ => return false,
    }
    true
}

/// the events, made the first time a title asks for one.
fn events(system: &mut System) -> [ObjectId; 2] {
    if let Some(events) = system.services.cecd.events {
        return events;
    }
    let events = ["cecd:u:InfoEvent", "cecd:u:ChangeStateEvent"].map(|name| {
        let object = system.kernel.objects.insert(KObject::Event(crate::kernel::sync::Event::new(ResetType::OneShot, name)));
        // the service keeps them, a title closing its handles must not take
        // them away
        system.kernel.objects.add_ref(object);
        object
    });
    system.services.cecd.events = Some(events);
    events
}
