//! ldr:ro, the dynamic module loader.

use crate::cro::CroError;
use crate::kernel::ipc::{CommandBuffer, Header};
use crate::System;

mod command {
    pub const INITIALIZE: u16 = 0x0001;
    pub const LOAD_CRR: u16 = 0x0002;
    pub const UNLOAD_CRR: u16 = 0x0003;
    pub const LOAD_CRO: u16 = 0x0004;
    pub const UNLOAD_CRO: u16 = 0x0005;
    pub const LINK_CRO: u16 = 0x0006;
    pub const UNLINK_CRO: u16 = 0x0007;
    pub const SHUTDOWN: u16 = 0x0008;
    pub const LOAD_CRO_NEW: u16 = 0x0009;
}

/// 0xD9012402, the module is not a valid CRO.
const ERROR_NOT_LOADED: u32 = 0xD901_2402;

pub fn handle(system: &mut System, buffer: &CommandBuffer, header: Header) -> bool {
    use command::*;
    let id = header.command_id();
    match id {
        // initialize(crs buffer, size, address, <process>)
        INITIALIZE => {
            let source = buffer.get(&mut system.memory, 1);
            let size = buffer.get(&mut system.memory, 2);
            let address = buffer.get(&mut system.memory, 3);

            let base = place(system, address, source, size);
            match system.cro.initialize(&mut system.memory, base, size) {
                Ok(()) => buffer.reply(&mut system.memory, id, &[]),
                Err(error) => {
                    log::error!("ldr:ro Initialize failed: {error:?}");
                    buffer.reply_error(&mut system.memory, id, ERROR_NOT_LOADED);
                }
            }
            true
        }

        // the certificate tables are a signature check we do not perform.
        LOAD_CRR | UNLOAD_CRR => {
            buffer.reply(&mut system.memory, id, &[]);
            true
        }

        // LoadCRO(buffer, address, size, data segment, 0, data size,
        //         bss segment, bss size, auto link, fix level, crr, <process>)
        LOAD_CRO | LOAD_CRO_NEW => {
            let source = buffer.get(&mut system.memory, 1);
            let address = buffer.get(&mut system.memory, 2);
            let size = buffer.get(&mut system.memory, 3);
            let data_segment = buffer.get(&mut system.memory, 4);
            let data_segment_size = buffer.get(&mut system.memory, 6);
            let bss_segment = buffer.get(&mut system.memory, 7);
            let bss_segment_size = buffer.get(&mut system.memory, 8);
            let auto_link = buffer.get(&mut system.memory, 9) != 0;

            log::debug!(
                "ldr:ro LoadCRO: buffer 0x{source:08X} -> 0x{address:08X}, 0x{size:X} bytes, \
                 data 0x{data_segment:08X}+0x{data_segment_size:X}, \
                 bss 0x{bss_segment:08X}+0x{bss_segment_size:X}, auto link {auto_link}"
            );
            let base = place(system, address, source, size);
            let result = system.cro.load(
                &mut system.memory,
                base,
                size,
                data_segment,
                data_segment_size,
                bss_segment,
                bss_segment_size,
                auto_link,
            );
            match result {
                Ok(fix_size) => buffer.reply(&mut system.memory, id, &[fix_size]),
                Err(CroError::NotACro) => {
                    log::error!("ldr:ro: the buffer at 0x{source:08X} is not a CRO");
                    buffer.reply_error(&mut system.memory, id, ERROR_NOT_LOADED);
                }
                Err(error) => {
                    log::error!("ldr:ro LoadCRO failed: {error:?}");
                    buffer.reply_error(&mut system.memory, id, ERROR_NOT_LOADED);
                }
            }
            true
        }

        UNLOAD_CRO => {
            let address = buffer.get(&mut system.memory, 1);
            system.cro.unload(&mut system.memory, address);
            buffer.reply(&mut system.memory, id, &[]);
            true
        }

        // a module is linked as it loads, so these have nothing left to do.
        LINK_CRO | UNLINK_CRO | SHUTDOWN => {
            buffer.reply(&mut system.memory, id, &[]);
            true
        }

        _ => false,
    }
}

/// makes the module visible at the address the title wants to run it from,
/// and returns that address.
fn place(system: &mut System, address: u32, source: u32, size: u32) -> u32 {
    if address == 0 || address == source {
        return source;
    }
    if system.memory.mirror(address, source, size) {
        address
    } else {
        log::warn!(
            "ldr:ro: could not mirror 0x{source:08X} at 0x{address:08X}, using the buffer"
        );
        source
    }
}
