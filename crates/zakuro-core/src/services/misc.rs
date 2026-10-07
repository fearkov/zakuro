//! small services that need only a handful of commands to keep a title moving.

use crate::kernel::ipc::{CommandBuffer, Header};
use crate::System;

/// what am answers for a title that is not installed.
const TITLE_NOT_FOUND: u32 = 0xD8A0_83FA;

pub fn handle(
    system: &mut System,
    buffer: &CommandBuffer,
    header: Header,
    service: &str,
) -> bool {
    let command = header.command_id();
    match service {
        // -- Power and shell state ------------------------------------------
        "ptm:u" | "ptm:s" | "ptm:sysm" | "ptm:play" => match command {
            // GetShellState, 1 = open.
            0x0005 => {
                buffer.reply(&mut system.memory, command, &[1]);
                true
            }
            // GetBatteryLevel, 5 = full.
            0x0007 => {
                buffer.reply(&mut system.memory, command, &[5]);
                true
            }
            // GetBatteryChargeState, not charging.
            0x0008 => {
                buffer.reply(&mut system.memory, command, &[0]);
                true
            }
            // GetPedometerState / GetTotalStepCount
            0x0009 | 0x000C => {
                buffer.reply(&mut system.memory, command, &[0]);
                true
            }
            // GetStepHistory(hours, start time, buffer), no steps in any of
            // those hours
            0x000B => {
                let hours = buffer.get(&mut system.memory, 1).min(0x8000);
                let pointer = buffer.get(&mut system.memory, 5);
                system.memory.write_bytes(pointer, &vec![0; hours as usize * 2]);
                buffer.reply(&mut system.memory, command, &[]);
                true
            }
            _ => false,
        },

        // -- SSL, which works without a network, its connections do not -----
        "ssl:C" => match command {
            // Initialize(process id). Pokémon Ultra Sun and Moon set it up
            // when saving, and stop when it fails
            0x0001 => {
                buffer.reply(&mut system.memory, command, &[]);
                true
            }
            // GenerateRandomData(size, buffer)
            0x0011 => {
                let size = buffer.get(&mut system.memory, 1).min(0x10_0000);
                let descriptor = buffer.get(&mut system.memory, 2);
                let pointer = buffer.get(&mut system.memory, 3);
                let bytes = random_bytes(system.cpu.cycles, size as usize);
                system.memory.write_bytes(pointer, &bytes);
                buffer.set(&mut system.memory, 0, Header::new(command, 1, 2).0);
                buffer.set(&mut system.memory, 1, 0);
                buffer.set(&mut system.memory, 2, descriptor);
                buffer.set(&mut system.memory, 3, pointer);
                true
            }
            _ => false,
        },

        // -- Network daemons ------------------------------------------------
        "ndm:u" => match command {
            // EnterExclusiveState / LeaveExclusiveState / SuspendDaemons /
            // ResumeDaemons / OverrideDefaultDaemons, all no-ops offline.
            0x0001 | 0x0002 | 0x0006 | 0x0007 | 0x0008 | 0x0009 | 0x000A | 0x000E | 0x0014 => {
                buffer.reply(&mut system.memory, command, &[]);
                true
            }
            _ => false,
        },

        // -- Installed titles -----------------------------------------------
        // the DLC and the update given with the title are installed, a title
        // asking about others hears they are not there
        "am:app" => match command {
            // GetDLCContentInfoCount(media type, title id), every content
            // the DLC's TMD lists
            0x1001 => {
                let title_id = read_u64(system, buffer, 2);
                match system.dlc.iter().find(|dlc| dlc.title_id == title_id) {
                    Some(dlc) => {
                        let count = dlc.contents.len() as u32;
                        buffer.reply(&mut system.memory, command, &[count]);
                    }
                    None => buffer.reply_error(&mut system.memory, command, TITLE_NOT_FOUND),
                }
                true
            }
            // FindDLCContentInfos(media type, title id, count, the indices,
            // where their infos go)
            0x1002 => {
                let title_id = read_u64(system, buffer, 2);
                let count = buffer.get(&mut system.memory, 4);
                let (indices, out) = (buffer.get(&mut system.memory, 6), buffer.get(&mut system.memory, 8));
                let Some(dlc) = system.dlc.iter().position(|dlc| dlc.title_id == title_id) else {
                    buffer.reply_error(&mut system.memory, command, TITLE_NOT_FOUND);
                    return true;
                };
                for i in 0..count {
                    use zakuro_cpu::Bus;
                    let index = system.memory.read16(indices + i * 2);
                    if let Some(content) = system.dlc[dlc].contents.iter().find(|content| content.index == index).copied() {
                        write_content_info(system, out + i * CONTENT_INFO_SIZE, &content);
                    }
                }
                buffer.reply(&mut system.memory, command, &[]);
                true
            }
            // ListDLCContentInfos(count, media type, title id, first, where
            // the infos go), how many it wrote
            0x1003 => {
                let count = buffer.get(&mut system.memory, 1) as usize;
                let title_id = read_u64(system, buffer, 3);
                let first = buffer.get(&mut system.memory, 5) as usize;
                let out = buffer.get(&mut system.memory, 7);
                let Some(dlc) = system.dlc.iter().find(|dlc| dlc.title_id == title_id) else {
                    buffer.reply_error(&mut system.memory, command, TITLE_NOT_FOUND);
                    return true;
                };
                let contents: Vec<zakuro_fs::Content> = dlc.contents.iter().skip(first).take(count).copied().collect();
                for (i, content) in contents.iter().enumerate() {
                    write_content_info(system, out + i as u32 * CONTENT_INFO_SIZE, content);
                }
                buffer.reply(&mut system.memory, command, &[contents.len() as u32]);
                true
            }
            // GetDLCTitleInfos and GetPatchTitleInfos(media type, count, the
            // title ids, where their infos go)
            0x1005 | 0x100D => {
                let count = buffer.get(&mut system.memory, 2);
                let (ids, out) = (buffer.get(&mut system.memory, 4), buffer.get(&mut system.memory, 6));
                let mut missing = false;
                for i in 0..count {
                    use zakuro_cpu::Bus;
                    let title_id = system.memory.read32(ids + i * 8) as u64 | (system.memory.read32(ids + i * 8 + 4) as u64) << 32;
                    let dlc = system.dlc.iter().find(|dlc| dlc.title_id == title_id).map(|dlc| dlc.version);
                    let update = system.title.as_ref().and_then(|title| title.update()).filter(|update| update.title_id == title_id);
                    let version = if command == 0x1005 { dlc } else { update.map(|update| update.version) };
                    let Some(version) = version else {
                        missing = true;
                        continue;
                    };
                    // the id, a size of 0, the version, and the type Citra gives
                    let at = out + i * 0x18;
                    for (offset, word) in [(0, title_id as u32), (4, (title_id >> 32) as u32), (8, 0), (12, 0), (16, version as u32), (20, 0x40)] {
                        system.memory.write32(at + offset, word);
                    }
                }
                if missing {
                    buffer.reply_error(&mut system.memory, command, TITLE_NOT_FOUND);
                } else {
                    buffer.reply(&mut system.memory, command, &[]);
                }
                true
            }
            // ListDataTitleTicketInfos(count, title id, first, where they
            // go), a ticket for each DLC given
            0x1007 => {
                let count = buffer.get(&mut system.memory, 1);
                let title_id = read_u64(system, buffer, 2);
                let first = buffer.get(&mut system.memory, 4);
                let out = buffer.get(&mut system.memory, 6);
                let version = system.dlc.iter().find(|dlc| dlc.title_id == title_id).map(|dlc| dlc.version);
                match version.filter(|_| first == 0 && count > 0) {
                    Some(version) => {
                        use zakuro_cpu::Bus;
                        for (offset, word) in [(0, title_id as u32), (4, (title_id >> 32) as u32), (8, 0), (12, 0), (16, version as u32), (20, 0)] {
                            system.memory.write32(out + offset, word);
                        }
                        buffer.reply(&mut system.memory, command, &[1]);
                    }
                    None => buffer.reply(&mut system.memory, command, &[0]),
                }
                true
            }
            // GetNumDataTitleTickets and IsDataTitleInUse
            0x1006 | 0x1009 => {
                buffer.reply_error(&mut system.memory, command, TITLE_NOT_FOUND);
                true
            }
            _ => false,
        },

        // -- Wifi connection ------------------------------------------------
        "ac:u" | "ac:i" => match command {
            // GetWifiStatus, 0 = not connected, which is the truth.
            0x000D => {
                buffer.reply(&mut system.memory, command, &[0]);
                true
            }
            // GetLastErrorCode / GetStatus
            0x000A | 0x000E => {
                buffer.reply(&mut system.memory, command, &[0]);
                true
            }
            // CloseAsync and friends still have to signal their event.
            0x0005 | 0x0008 => {
                buffer.reply(&mut system.memory, command, &[]);
                true
            }
            // RegisterDisconnectEvent and SetClientVersion
            0x0030 | 0x0040 => {
                buffer.reply(&mut system.memory, command, &[]);
                true
            }
            // IsConnected, no
            0x003E => {
                buffer.reply(&mut system.memory, command, &[0]);
                true
            }
            _ => false,
        },

        // -- Accounts, background downloads and http ------------------------
        // setting a session up works offline, only its requests need the
        // network
        "act:u" | "act:a" | "boss:U" | "boss:P" | "http:C" => match command {
            0x0001 => {
                buffer.reply(&mut system.memory, command, &[]);
                true
            }
            _ => false,
        },

        _ => false,
    }
}

/// bytes no one can tell from random, the same for the same moment of a run
/// so that runs repeat, SplitMix64 from the console's clock.
fn random_bytes(seed: u64, len: usize) -> Vec<u8> {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut bytes: Vec<u8> = Vec::with_capacity(len + 8);
    while bytes.len() < len {
        bytes.extend_from_slice(&next().to_le_bytes());
    }
    bytes.truncate(len);
    bytes
}

/// the size of AM's description of a title's content.
const CONTENT_INFO_SIZE: u32 = 0x18;

/// a u64 the command passes as two words from index.
fn read_u64(system: &mut System, buffer: &CommandBuffer, index: u32) -> u64 {
    buffer.get(&mut system.memory, index) as u64 | (buffer.get(&mut system.memory, index + 1) as u64) << 32
}

/// writes AM's description of a content at at, its index, type, id and
/// size, and that it is owned, and there, when the file holds it.
fn write_content_info(system: &mut System, at: u32, content: &zakuro_fs::Content) {
    use zakuro_cpu::Bus;
    const OWNED: u32 = 0x02;
    const DOWNLOADED: u32 = 0x01;
    let ownership = OWNED | if content.offset.is_some() { DOWNLOADED } else { 0 };
    for (offset, word) in [
        (0, content.index as u32 | (content.flags as u32) << 16),
        (4, content.id),
        (8, content.size as u32),
        (12, (content.size >> 32) as u32),
        (16, ownership),
        (20, 0),
    ] {
        system.memory.write32(at + offset, word);
    }
}
