//! small services that need only a handful of commands to keep a title moving.

use crate::kernel::ipc::{CommandBuffer, Header};
use crate::System;

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
            _ => false,
        },

        // -- Network daemons ------------------------------------------------
        "ndm:u" => match command {
            // EnterExclusiveState / LeaveExclusiveState / SuspendDaemons /
            // ResumeDaemons / OverrideDefaultDaemons, all no-ops offline.
            0x0001 | 0x0002 | 0x0006 | 0x0007 | 0x0008 | 0x0009 | 0x000A | 0x000E => {
                buffer.reply(&mut system.memory, command, &[]);
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
            _ => false,
        },

        _ => false,
    }
}
