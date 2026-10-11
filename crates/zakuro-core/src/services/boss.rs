//! boss:U and boss:P, SpotPass. a console without a connection keeps the
//! tasks a title registers and its settings, it only never downloads
//! anything, so the service answers as that console does, with no data
//! and nothing new, rather than with a network error a title never sees
//! from it. New Super Mario Bros. 2 unregisters its task on starting with a
//! save and stops when that fails.

use crate::kernel::ipc::{CommandBuffer, Descriptor, Header};
use crate::kernel::object::{KObject, ObjectId};
use crate::kernel::sync::{Event, ResetType};
use crate::System;

/// what reading or asking about downloaded data answers, there is none:
/// status, invalid state, BOSS, 67.
const NS_DATA_NOT_FOUND: u32 = 0xC8A0_F843;

/// the properties ReceiveProperty answers from the tasks registered, how
/// many there are and their ids, 8 bytes each in a list of 0x400.
const TOTAL_TASKS: u32 = 0x35;
const TASK_ID_LIST: u32 = 0x36;
const TASK_ID_SIZE: usize = 8;

#[derive(Default)]
pub struct BossState {
    /// whether the player turned the title's SpotPass off.
    optout: u32,
    /// the event a finished task signals, which none ever does.
    task_finish: Option<ObjectId>,
    /// the tasks registered since the title started. the console keeps
    /// them between runs too, unregistering one from an earlier run
    /// succeeds all the same.
    tasks: Vec<[u8; TASK_ID_SIZE]>,
}

/// answers a request, false when the command is not one of SpotPass's.
pub fn handle(system: &mut System, buffer: &CommandBuffer, header: Header) -> bool {
    let command = header.command_id();
    match command {
        // InitializeSession, SetStorageInfo, UnregisterStorage,
        // RegisterNewArrivalEvent, GetTaskIdList, SendPropertyHandle,
        // DeleteNsData, SetNsDataAdditionalInfo, SetNsDataNewFlag,
        // RegisterStorageEntry, SetStorageOption
        0x0001 | 0x0002 | 0x0003 | 0x0008 | 0x000E | 0x0015 | 0x0026 | 0x0029 | 0x002B | 0x002F | 0x0031 => {
            buffer.reply(&mut system.memory, command, &[]);
        }
        // GetStorageInfo, GetNewArrivalFlag, GetNsDataAdditionalInfo,
        // GetNsDataNewFlag, GetErrorCode
        0x0004 | 0x0007 | 0x002A | 0x002C | 0x002E => {
            buffer.reply(&mut system.memory, command, &[0]);
        }
        // SetOptoutFlag / GetOptoutFlag
        0x0009 => {
            system.services.boss.optout = buffer.get(&mut system.memory, 1) & 0xFF;
            buffer.reply(&mut system.memory, command, &[]);
        }
        0x000A => {
            let optout = system.services.boss.optout;
            buffer.reply(&mut system.memory, command, &[optout]);
        }
        // RegisterTask / RegisterImmediateTask(size, unknown, unknown,
        // task id)
        0x000B | 0x0035 => {
            let id = task_id(system, buffer, 1, 5);
            if !system.services.boss.tasks.contains(&id) {
                system.services.boss.tasks.push(id);
            }
            reply_with_buffers(system, buffer, header, &[]);
        }
        // UnregisterTask(size, unknown, task id)
        0x000C => {
            let id = task_id(system, buffer, 1, 4);
            system.services.boss.tasks.retain(|&task| task != id);
            reply_with_buffers(system, buffer, header, &[]);
        }
        // the other commands that take a task id or other data in buffers
        // and hand them back, RegisterPrivateRootCa,
        // RegisterPrivateClientCert, ReconfigureTask, GetStepIdList,
        // SendProperty, UpdateTaskInterval, UpdateTaskCount, StartTask,
        // StartTaskImmediate, CancelTask, GetTaskInfo, StartBgImmediate,
        // SetTaskQuery, GetTaskQuery
        0x0005 | 0x0006 | 0x000D | 0x000F | 0x0014 | 0x0017 | 0x0018 | 0x001C | 0x001D | 0x001E | 0x0025 | 0x0033
        | 0x0036 | 0x0037 => reply_with_buffers(system, buffer, header, &[]),
        // GetNsDataIdList and its variants, no entries
        0x0010..=0x0013 => reply_with_buffers(system, buffer, header, &[0, 0]),
        // ReceiveProperty(id, size, buffer), the tasks registered, and zero
        // for the rest
        0x0016 => {
            let property = buffer.get(&mut system.memory, 1) & 0xFFFF;
            let size = buffer.get(&mut system.memory, 2);
            let pointer = buffer.get(&mut system.memory, 4);
            let mut value = vec![0u8; size.min(0x1_0000) as usize];
            let tasks = &system.services.boss.tasks;
            let known: Vec<u8> = match property {
                TOTAL_TASKS => (tasks.len() as u16).to_le_bytes().to_vec(),
                TASK_ID_LIST => tasks.iter().flatten().copied().collect(),
                _ => Vec::new(),
            };
            let shared = known.len().min(value.len());
            value[..shared].copy_from_slice(&known[..shared]);
            if pointer != 0 {
                system.memory.write_bytes(pointer, &value);
            }
            reply_with_buffers(system, buffer, header, &[size]);
        }
        // GetTaskInterval, GetTaskCount, GetTaskServiceStatus,
        // GetTaskStatus, GetTaskError, GetTaskProperty0
        0x0019 | 0x001A | 0x001B | 0x0023 | 0x0024 | 0x0034 => reply_with_buffers(system, buffer, header, &[0]),
        // GetTaskState, GetTaskResult, GetTaskCommErrorCode
        0x0020..=0x0022 => reply_with_buffers(system, buffer, header, &[0, 0, 0]),
        // GetTaskFinishHandle, a handle of the title's own to the event
        // the service keeps
        0x001F => {
            let object = match system.services.boss.task_finish {
                Some(object) => object,
                None => {
                    let object = system.kernel.objects.insert(KObject::Event(Event::new(ResetType::OneShot, "BOSS:task finish")));
                    // a title closing its handle must not take it away
                    system.kernel.objects.add_ref(object);
                    system.services.boss.task_finish = Some(object);
                    object
                }
            };
            let handle = system.kernel.handles.create(&mut system.kernel.objects, object, "BOSS:task finish");
            buffer.set(&mut system.memory, 0, Header::new(command, 1, 2).0);
            buffer.set(&mut system.memory, 1, 0);
            buffer.set(&mut system.memory, 2, Descriptor::handles(1));
            buffer.set(&mut system.memory, 3, handle);
        }
        // GetNsDataHeaderInfo, ReadNsData, GetNsDataLastUpdate, of data
        // that was never downloaded
        0x0027 | 0x0028 | 0x002D => buffer.reply_error(&mut system.memory, command, NS_DATA_NOT_FOUND),
        // GetStorageEntryInfo
        0x0030 => buffer.reply(&mut system.memory, command, &[0, 0]),
        // GetStorageOption
        0x0032 => buffer.reply(&mut system.memory, command, &[0, 0, 0, 0]),
        _ => return false,
    }
    true
}

/// the task id in a request's buffer, whose size is the word at size and
/// whose address the word at pointer, padded to 8 bytes.
fn task_id(system: &mut System, buffer: &CommandBuffer, size: u32, pointer: u32) -> [u8; TASK_ID_SIZE] {
    let size = (buffer.get(&mut system.memory, size) as usize).min(TASK_ID_SIZE);
    let pointer = buffer.get(&mut system.memory, pointer);
    let mut id = [0u8; TASK_ID_SIZE];
    if pointer != 0 {
        system.memory.read_bytes(pointer, &mut id[..size]);
    }
    id
}

/// a successful reply with values, then the request's buffer descriptors
/// handed back as they came.
fn reply_with_buffers(system: &mut System, buffer: &CommandBuffer, header: Header, values: &[u32]) {
    let first = 1 + header.normal_params();
    let translate: Vec<u32> = (0..header.translate_params()).map(|i| buffer.get(&mut system.memory, first + i)).collect();
    let normal = values.len() as u32 + 1;
    buffer.set(&mut system.memory, 0, Header::new(header.command_id(), normal, translate.len() as u32).0);
    buffer.set(&mut system.memory, 1, 0);
    for (i, &value) in values.iter().enumerate() {
        buffer.set(&mut system.memory, 2 + i as u32, value);
    }
    for (i, &word) in translate.iter().enumerate() {
        buffer.set(&mut system.memory, 1 + normal + i as u32, word);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::tests::system_with_thread;
    use crate::services::{handle_request, Target};

    /// a title starting with a save unregisters its task, which works
    /// offline and hands the buffer back.
    #[test]
    fn a_task_unregisters_offline() {
        let (mut system, buffer) = system_with_thread();
        // UnregisterTask(size, unknown, mapped buffer)
        buffer.set(&mut system.memory, 0, Header::new(0x000C, 2, 2).0);
        buffer.set(&mut system.memory, 1, 8);
        buffer.set(&mut system.memory, 2, 0);
        buffer.set(&mut system.memory, 3, (8 << 4) | 0xA);
        buffer.set(&mut system.memory, 4, 0x1234_5000);

        handle_request(&mut system, Target::service("boss:U".into(), 0));

        assert_eq!(buffer.header(&mut system.memory), Header::new(0x000C, 1, 2));
        assert_eq!(buffer.get(&mut system.memory, 1), 0);
        assert_eq!(buffer.get(&mut system.memory, 2), (8 << 4) | 0xA);
        assert_eq!(buffer.get(&mut system.memory, 3), 0x1234_5000);
        assert!(system.services.unimplemented.is_empty());
    }

    /// the tasks registered are listed, and unregistering one this run
    /// never saw succeeds.
    #[test]
    fn registered_tasks_are_listed() {
        let (mut system, buffer) = system_with_thread();
        let id = buffer.address() + 0x100;
        let list = buffer.address() + 0x110;
        system.memory.write_bytes(id, b"TASK01\0\0");
        let register = |system: &mut System, command: u16| {
            // RegisterTask(size, unknown, unknown, id) / UnregisterTask(size,
            // unknown, id)
            let normal = if command == 0x000B { 3 } else { 2 };
            buffer.set(&mut system.memory, 0, Header::new(command, normal, 2).0);
            buffer.set(&mut system.memory, 1, 8);
            buffer.set(&mut system.memory, 1 + normal, (8 << 4) | 0xA);
            buffer.set(&mut system.memory, 2 + normal, id);
            handle_request(system, Target::service("boss:U".into(), 0));
            assert_eq!(buffer.get(&mut system.memory, 1), 0);
        };
        let receive = |system: &mut System, property: u32, size: u32| -> Vec<u8> {
            buffer.set(&mut system.memory, 0, Header::new(0x0016, 2, 2).0);
            buffer.set(&mut system.memory, 1, property);
            buffer.set(&mut system.memory, 2, size);
            buffer.set(&mut system.memory, 3, (size << 4) | 0xC);
            buffer.set(&mut system.memory, 4, list);
            handle_request(system, Target::service("boss:U".into(), 0));
            let mut out = vec![0u8; size as usize];
            system.memory.read_bytes(list, &mut out);
            out
        };
        register(&mut system, 0x000B);
        assert_eq!(receive(&mut system, TOTAL_TASKS, 2), [1, 0]);
        assert_eq!(&receive(&mut system, TASK_ID_LIST, 16)[..8], b"TASK01\0\0");
        register(&mut system, 0x000C);
        assert_eq!(receive(&mut system, TOTAL_TASKS, 2), [0, 0]);
        register(&mut system, 0x000C);
    }

    /// each title handle to the finish event is its own, closing one leaves
    /// the event to the next.
    #[test]
    fn the_task_finish_event_outlives_a_closed_handle() {
        let (mut system, buffer) = system_with_thread();
        let handle = |system: &mut System| {
            buffer.set(&mut system.memory, 0, Header::new(0x001F, 0, 0).0);
            handle_request(system, Target::service("boss:U".into(), 0));
            buffer.get(&mut system.memory, 3)
        };
        let first = handle(&mut system);
        assert!(system.kernel.handles.close(&mut system.kernel.objects, first));
        let second = handle(&mut system);
        assert!(system.kernel.resolve(second).is_some());
    }

    /// the opt-out flag reads back as it was set.
    #[test]
    fn the_optout_flag_is_kept() {
        let (mut system, buffer) = system_with_thread();
        buffer.set(&mut system.memory, 0, Header::new(0x0009, 1, 0).0);
        buffer.set(&mut system.memory, 1, 0xFFFF_FF01);
        handle_request(&mut system, Target::service("boss:U".into(), 0));
        buffer.set(&mut system.memory, 0, Header::new(0x000A, 0, 0).0);
        handle_request(&mut system, Target::service("boss:U".into(), 0));
        assert_eq!(buffer.get(&mut system.memory, 1), 0);
        assert_eq!(buffer.get(&mut system.memory, 2), 1);
    }
}
