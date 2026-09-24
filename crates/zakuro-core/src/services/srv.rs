//! srv:, the service directory.

use crate::kernel::ipc::{CommandBuffer, Header};
use crate::kernel::object::{ClientSession, KObject};
use crate::kernel::sync::ResetType;
use crate::System;

#[derive(Default)]
pub struct SrvState {
    /// handle of the notification semaphore, created on demand.
    pub notification_semaphore: Option<u32>,
    /// pending notification ids, newest last.
    pub notifications: Vec<u32>,
}

pub fn handle(system: &mut System, buffer: &CommandBuffer, header: Header) -> bool {
    match header.command_id() {
        // RegisterClient
        0x0001 => {
            buffer.reply(&mut system.memory, 0x0001, &[]);
            true
        }
        // EnableNotification, hands back a semaphore the caller waits on.
        0x0002 => {
            let handle = match system.services.srv.notification_semaphore {
                Some(handle) => handle,
                None => {
                    let object = system
                        .kernel
                        .objects
                        .insert(KObject::Semaphore(crate::kernel::sync::Semaphore {
                            name: "srv:notification".into(),
                            count: 0,
                            max_count: 16,
                        }));
                    let handle =
                        system
                            .kernel
                            .handles
                            .create(&mut system.kernel.objects, object, "srv:notification");
                    system.services.srv.notification_semaphore = Some(handle);
                    handle
                }
            };
            buffer.reply_with_handle(&mut system.memory, 0x0002, handle);
            true
        }
        // GetServiceHandle(name[8], name_len, flags)
        0x0005 => {
            let mut bytes = [0u8; 8];
            bytes[0..4].copy_from_slice(&buffer.get(&mut system.memory, 1).to_le_bytes());
            bytes[4..8].copy_from_slice(&buffer.get(&mut system.memory, 2).to_le_bytes());
            let len = buffer.get(&mut system.memory, 3).min(8) as usize;
            let name = String::from_utf8_lossy(&bytes[..len]).into_owned();

            let object = system
                .kernel
                .objects
                .insert(KObject::ClientSession(ClientSession {
                    service: name.clone(),
                    subhandle: 0,
                }));
            let handle = system
                .kernel
                .handles
                .create(&mut system.kernel.objects, object, &name);
            log::debug!("srv:GetServiceHandle(\"{name}\") -> 0x{handle:X}");
            system.services_seen.insert(name);
            buffer.reply_with_handle(&mut system.memory, 0x0005, handle);
            true
        }
        // subscribe / Unsubscribe
        0x0009 | 0x000A => {
            buffer.reply(&mut system.memory, header.command_id(), &[]);
            true
        }
        // ReceiveNotification
        0x000B => {
            let notification = system.services.srv.notifications.pop().unwrap_or(0);
            buffer.reply(&mut system.memory, 0x000B, &[notification]);
            true
        }
        // IsServiceRegistered, everything a title asks for is "present", since
        // an unimplemented service still answers.
        0x000E => {
            buffer.reply(&mut system.memory, 0x000E, &[1]);
            true
        }
        _ => false,
    }
}

/// queues a notification and bumps the semaphore, which is how the system
/// tells a title about things like the home button being pressed.
pub fn notify(system: &mut System, id: u32) {
    system.services.srv.notifications.insert(0, id);
    if let Some(handle) = system.services.srv.notification_semaphore {
        if let Some(object) = system.kernel.handles.resolve(handle) {
            if let Some(KObject::Semaphore(semaphore)) = system.kernel.objects.get_mut(object) {
                semaphore.count = (semaphore.count + 1).min(semaphore.max_count);
                system.kernel.reschedule_pending = true;
            }
        }
    }
}

/// unused for now, kept so the reset-type import stays meaningful when
/// notification events are added.
pub const _NOTIFICATION_RESET: ResetType = ResetType::OneShot;
