//! err:f, the fatal error reporter.

use zakuro_cpu::Bus;

use crate::kernel::ipc::{CommandBuffer, Header};
use crate::System;

/// ERRF_ErrType.
fn error_type_name(value: u8) -> &'static str {
    match value {
        0 => "generic",
        1 => "corrupted",
        2 => "card removed",
        3 => "CPU exception",
        4 => "result failure",
        5 => "logged",
        _ => "unknown",
    }
}

pub fn handle(system: &mut System, buffer: &CommandBuffer, header: Header) -> bool {
    match header.command_id() {
        // throw(ERRF_FatalErrInfo)
        0x0001 => {
            let word = |system: &mut System, index: u32| buffer.get(&mut system.memory, index);

            let first = word(system, 1);
            let kind = (first & 0xFF) as u8;
            let revision_high = (first >> 8) & 0xFF;
            let revision_low = (first >> 16) & 0xFFFF;
            let result = word(system, 2);
            let pc = word(system, 3);
            let process_id = word(system, 4);

            log::error!("=== the title reported a fatal error ===");
            log::error!(
                "  type    {} ({kind})",
                error_type_name(kind)
            );
            log::error!("  result  0x{result:08X}");
            log::error!("  pc      0x{pc:08X}");
            log::error!("  process 0x{process_id:08X}");
            log::error!("  rev     {revision_high}.{revision_low}");

            match kind {
                // A CPU exception carries register state.
                3 => {
                    let exception_type = word(system, 5) & 0xFF;
                    log::error!(
                        "  exception type {exception_type}, fault 0x{:08X}",
                        word(system, 6)
                    );
                    for i in 0..8 {
                        log::error!("  r{i} = 0x{:08X}", word(system, 7 + i));
                    }
                }
                // everything else may carry a message.
                _ => {
                    let address = buffer.address() + 5 * 4;
                    let mut bytes = [0u8; 0x60];
                    for (i, byte) in bytes.iter_mut().enumerate() {
                        *byte = system.memory.read8(address + i as u32);
                    }
                    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
                    let message = String::from_utf8_lossy(&bytes[..end]);
                    if !message.trim().is_empty() {
                        log::error!("  message \"{}\"", message.trim());
                    }
                }
            }

            system.fatal_errors.push(format!(
                "{} error 0x{result:08X} at pc 0x{pc:08X}",
                error_type_name(kind)
            ));

            buffer.reply(&mut system.memory, 0x0001, &[]);
            true
        }
        // SetUserString
        0x0002 => {
            buffer.reply(&mut system.memory, 0x0002, &[]);
            true
        }
        _ => false,
    }
}
