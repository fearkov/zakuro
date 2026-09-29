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
        // no add-on content or update is installed, a title asking about one
        // hears it is not there instead of reading an empty description
        "am:app" => match command {
            // GetDLCContentInfoCount, FindDLCContentInfos, ListDLCContentInfos,
            // GetDLCTitleInfos and GetPatchTitleInfos
            0x1001 | 0x1002 | 0x1003 | 0x1005 | 0x1006 | 0x1009 => {
                buffer.reply_error(&mut system.memory, command, TITLE_NOT_FOUND);
                true
            }
            // ListDataTitleTicketInfos, no tickets
            0x1007 => {
                buffer.reply(&mut system.memory, command, &[0]);
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
