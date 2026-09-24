//! APT:U, the applet manager.

use crate::kernel::ipc::{CommandBuffer, Descriptor, Header};
use crate::kernel::sync::ResetType;
use crate::System;

/// signals a parameter can carry. Wakeup is the one that starts a title.
pub const SIGNAL_NONE: u32 = 0;
pub const SIGNAL_WAKEUP: u32 = 1;

#[derive(Debug, Clone)]
pub struct Parameter {
    pub sender: u32,
    pub destination: u32,
    pub signal: u32,
    pub buffer: Vec<u8>,
}

#[derive(Default)]
pub struct AptState {
    pub lock: Option<u32>,
    pub notification_event: Option<u32>,
    pub parameter_event: Option<u32>,
    pub parameter: Option<Parameter>,
    pub cpu_time_limit: u32,
    pub initialized: bool,
    /// address and handle of the shared font block, once mapped.
    pub shared_font: Option<(u32, u32)>,
}

/// the value the status word at the start of the shared font block takes once
/// the system has finished loading it.
const FONT_STATUS_LOADED: u32 = 2;

/// where the shared font block is mapped for the guest.
const SHARED_FONT_VADDR: u32 = 0x1800_0000;

/// where a dumped shared font is looked for, relative to the working
/// directory and then to the user's data directory.
const SHARED_FONT_PATHS: &[&str] = &[
    "sysdata/shared_font.bin",
    "shared_font.bin",
];

/// maps the system's shared font, if the user has dumped it.
fn shared_font(system: &mut System) -> Option<(u32, u32)> {
    if let Some(cached) = system.services.apt.shared_font {
        return Some(cached);
    }

    let dump = SHARED_FONT_PATHS
        .iter()
        .find_map(|path| std::fs::read(path).ok());

    let data = match dump {
        Some(mut data) if data.len() > 0x84 && &data[0x80..0x84] == b"CFNT" => {
            // a dump taken from a console may have been captured before the
            // system finished loading it, make sure it reads as loaded.
            data[0..4].copy_from_slice(&FONT_STATUS_LOADED.to_le_bytes());
            data
        }
        Some(_) => {
            log::warn!("the shared font dump does not contain a CFNT structure, ignoring it");
            return None;
        }
        None => {
            // a title does not just use the font, it parses it, an empty block
            // fails that parse and the title never gets going.
            log::info!("no shared font file found; using the generated ASCII font");
            crate::services::shared_font::build(SHARED_FONT_VADDR)
        }
    };

    let size = zakuro_common::bits::align_up(data.len() as u32, 0x1000);
    let block = system
        .memory
        .phys
        .allocate(crate::memory::MemoryRegion::Base, size)?;

    let object = system
        .kernel
        .objects
        .insert(crate::kernel::object::KObject::SharedMemory(
            crate::kernel::object::SharedMemory {
                name: "SharedFont".into(),
                address: 0,
                size,
                paddr: block.addr,
                mapped_at: None,
            },
        ));
    let handle = system
        .kernel
        .handles
        .create(&mut system.kernel.objects, object, "SharedFont");

    // map it where the guest can see it, then copy the font in.
    let address = SHARED_FONT_VADDR;
    system.memory.map(
        address,
        block.addr,
        size,
        crate::memory::Permission::READ,
        crate::memory::MemoryState::Shared,
    );
    system.memory.write_physical(block.addr, &data);

    log::info!("mapped the shared font at 0x{address:08X} ({} KiB)", data.len() / 1024);
    system.services.apt.shared_font = Some((address, handle));
    Some((address, handle))
}

pub fn handle(system: &mut System, buffer: &CommandBuffer, header: Header) -> bool {
    let command = header.command_id();
    match command {
        // GetLockHandle(flags) -> result, applet attributes, power state, lock
        0x0001 => {
            let attributes = buffer.get(&mut system.memory, 1);
            let lock = match system.services.apt.lock {
                Some(handle) => handle,
                None => {
                    let object = system
                        .kernel
                        .objects
                        .insert(crate::kernel::object::KObject::Mutex(
                            crate::kernel::sync::Mutex::new("APT:lock"),
                        ));
                    let handle = system.kernel.handles.create(
                        &mut system.kernel.objects,
                        object,
                        "APT:lock",
                    );
                    system.services.apt.lock = Some(handle);
                    handle
                }
            };
            buffer.set(&mut system.memory, 0, Header::new(0x0001, 3, 2).0);
            buffer.set(&mut system.memory, 1, 0);
            buffer.set(&mut system.memory, 2, attributes);
            buffer.set(&mut system.memory, 3, 0); // power button not pressed
            buffer.set(&mut system.memory, 4, Descriptor::handles(1));
            buffer.set(&mut system.memory, 5, lock);
            true
        }

        // initialize(app id, attributes) -> result + notification and
        // parameter events.
        0x0002 => {
            let (_, notification) = system
                .kernel
                .create_event(ResetType::OneShot, "APT:notification");
            let (parameter_object, parameter) = system
                .kernel
                .create_event(ResetType::OneShot, "APT:parameter");
            system.services.apt.notification_event = Some(notification);
            system.services.apt.parameter_event = Some(parameter);
            system.services.apt.initialized = true;

            // the application is started by being handed a Wakeup parameter,
            // so queue it now and signal the event that says one is waiting.
            system.services.apt.parameter = Some(Parameter {
                sender: 0,
                destination: buffer.get(&mut system.memory, 1),
                signal: SIGNAL_WAKEUP,
                buffer: Vec::new(),
            });
            system.kernel.signal_event(parameter_object);

            buffer.set(&mut system.memory, 0, Header::new(0x0002, 1, 3).0);
            buffer.set(&mut system.memory, 1, 0);
            buffer.set(&mut system.memory, 2, Descriptor::handles(2));
            buffer.set(&mut system.memory, 3, notification);
            buffer.set(&mut system.memory, 4, parameter);
            true
        }

        // enable / Finalize / GetAppletManInfo / IsRegistered and friends.
        0x0003 | 0x0004 => {
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // GetAppletManInfo -> active applet position, requested id, menu id,
        // active id.
        0x0005 => {
            buffer.reply(&mut system.memory, 0x0005, &[0, 0, 0x101, 0x300]);
            true
        }
        // GetAppletInfo -> title id (u64), media type, registered, loaded,
        // attributes.
        0x0006 => {
            let title = system.kernel.program_id;
            buffer.reply(
                &mut system.memory,
                0x0006,
                &[title as u32, (title >> 32) as u32, 0, 1, 1, 0],
            );
            true
        }
        // IsRegistered -> true, so a title believes the applet it asked about
        // exists.
        0x0009 => {
            buffer.reply(&mut system.memory, 0x0009, &[1]);
            true
        }
        // InquireNotification -> no pending notification.
        0x000B => {
            buffer.reply(&mut system.memory, 0x000B, &[0]);
            true
        }
        // SendParameter
        0x000C => {
            buffer.reply(&mut system.memory, 0x000C, &[]);
            true
        }
        // ReceiveParameter / GlanceParameter
        0x000D | 0x000E => {
            let glance = command == 0x000E;
            let parameter = if glance {
                system.services.apt.parameter.clone()
            } else {
                system.services.apt.parameter.take()
            };

            match parameter {
                Some(parameter) => {
                    let size = parameter.buffer.len() as u32;
                    buffer.set(&mut system.memory, 0, Header::new(command, 4, 4).0);
                    buffer.set(&mut system.memory, 1, 0);
                    buffer.set(&mut system.memory, 2, parameter.sender);
                    buffer.set(&mut system.memory, 3, parameter.signal);
                    buffer.set(&mut system.memory, 4, size);
                    // no handle travels with a wakeup parameter.
                    buffer.set(&mut system.memory, 5, Descriptor::handles(1));
                    buffer.set(&mut system.memory, 6, 0);
                    buffer.set(&mut system.memory, 7, Descriptor::static_buffer(size, 0));
                    let (ptr, _) = {
                        let tls = system.kernel.current().map_or(0, |t| t.tls);
                        buffer.static_buffer(&mut system.memory, tls, 0)
                    };
                    buffer.set(&mut system.memory, 8, ptr);
                    if !parameter.buffer.is_empty() && ptr != 0 {
                        system.memory.write_bytes(ptr, &parameter.buffer);
                    }
                }
                None => {
                    // 0xC8A0CFFC, "no parameter is waiting".
                    buffer.reply_error(&mut system.memory, command, 0xC8A0_CFFC);
                }
            }
            true
        }
        // CancelParameter -> succeeded
        0x000F => {
            system.services.apt.parameter = None;
            buffer.reply(&mut system.memory, 0x000F, &[1]);
            true
        }
        // PrepareToStartApplication / StartApplication and the rest of the
        // launching machinery, nothing to do while only one title runs.
        0x0015 | 0x0016 | 0x0017 | 0x0018 | 0x0019 | 0x001B | 0x001E | 0x001F => {
            buffer.reply(&mut system.memory, command, &[]);
            true
        }
        // NotifyToWait
        0x0043 => {
            buffer.reply(&mut system.memory, 0x0043, &[]);
            true
        }
        // GetSharedFont -> the address the font was mapped at, plus its block.
        0x0044 => {
            match shared_font(system) {
                Some((address, handle)) => {
                    buffer.set(&mut system.memory, 0, Header::new(0x0044, 2, 2).0);
                    buffer.set(&mut system.memory, 1, 0);
                    buffer.set(&mut system.memory, 2, address);
                    buffer.set(&mut system.memory, 3, Descriptor::handles(1));
                    buffer.set(&mut system.memory, 4, handle);
                }
                None => {
                    // this is what hardware-less emulation has to say, there is
                    // no system font to hand over.
                    buffer.reply_error(&mut system.memory, 0x0044, 0xFFFF_FFFF);
                }
            }
            true
        }
        // ReceiveDeliverArg / SetWirelessRebootInfo and the rest of the
        // hand-off machinery.
        0x0035 | 0x0045 | 0x0046 | 0x0047 => {
            buffer.reply(&mut system.memory, command, &[0, 0]);
            true
        }
        // AppletUtility -> result plus one output word.
        0x004B => {
            buffer.reply(&mut system.memory, 0x004B, &[0]);
            true
        }
        // SetAppCpuTimeLimit / GetAppCpuTimeLimit
        0x004F => {
            system.services.apt.cpu_time_limit = buffer.get(&mut system.memory, 2);
            buffer.reply(&mut system.memory, 0x004F, &[]);
            true
        }
        0x0050 => {
            let limit = system.services.apt.cpu_time_limit;
            buffer.reply(&mut system.memory, 0x0050, &[limit]);
            true
        }
        // SetScreenCapPostPermission / GetScreenCapPostPermission
        0x0055 => {
            buffer.reply(&mut system.memory, 0x0055, &[]);
            true
        }
        0x0056 => {
            buffer.reply(&mut system.memory, 0x0056, &[1]);
            true
        }
        _ => false,
    }
}
