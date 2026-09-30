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
            let fix_level = buffer.get(&mut system.memory, 10);

            log::debug!(
                "ldr:ro LoadCRO: buffer 0x{source:08X} -> 0x{address:08X}, 0x{size:X} bytes, \
                 data 0x{data_segment:08X}+0x{data_segment_size:X}, \
                 bss 0x{bss_segment:08X}+0x{bss_segment_size:X}, auto link {auto_link}"
            );
            let base = copy_in(system, address, source, size);
            let result = system.cro.load(
                &mut system.memory,
                base,
                size,
                data_segment,
                data_segment_size,
                bss_segment,
                bss_segment_size,
                auto_link,
                fix_level,
            );
            match result {
                Ok(fix_size) => {
                    copy_back(system, base, size);
                    shrink(system, base, if fix_size == 0 { size } else { fix_size });
                    if let (Some(library), Some(module)) = (&mut system.recompiled, system.cro.modules.last()) {
                        library.place(&module.name, module.base);
                    }
                    buffer.reply(&mut system.memory, id, &[fix_size])
                }
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
            let module = system.cro.modules.iter().find(|m| m.base == address);
            if let (Some(library), Some(module)) = (&mut system.recompiled, module) {
                library.place(&module.name, 0);
            }
            system.cro.unload(&mut system.memory, address);
            copy_out(system, address);
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

/// gives a module memory of its own at the address the title wants to run
/// it from, a copy of its buffer, and returns that address. ldr:ro takes the
/// buffer's pages away from the title while the module is loaded, what the
/// title writes to the buffer meanwhile, like a module's data it puts back
/// on unloading, must not reach a module running from memory the title
/// has since given to another.
fn copy_in(system: &mut System, address: u32, source: u32, size: u32) -> u32 {
    use crate::memory::{MemoryRegion, MemoryState, Permission};
    if address == 0 || address == source {
        return source;
    }
    let Some(block) = system.memory.phys.allocate(MemoryRegion::System, size) else {
        log::warn!("ldr:ro: no memory for a 0x{size:X}-byte module, running it from its buffer");
        return place(system, address, source, size);
    };
    let mut bytes = vec![0u8; size as usize];
    system.memory.read_bytes(source, &mut bytes);
    system.memory.write_physical(block.addr, &bytes);
    system.memory.map(address, block.addr, block.size, Permission::RW | Permission::EXECUTE, MemoryState::Code);
    system.cro.copies.insert(address, crate::cro::Copy { buffer: source, block, fixed: size });
    address
}

/// puts the module as ldr:ro left it into the title's buffer as well, the
/// title copies the module's data, relocated, out of its buffer once
/// LoadCRO returns, and on the console the two are the same memory then.
fn copy_back(system: &mut System, address: u32, size: u32) {
    let Some(copy) = system.cro.copies.get(&address).copied() else { return };
    let mut bytes = vec![0u8; size as usize];
    system.memory.read_bytes(address, &mut bytes);
    system.memory.write_bytes(copy.buffer, &bytes);
}

/// keeps what fixing left of a module and lets the rest of its memory go,
/// the title maps the next module over it.
fn shrink(system: &mut System, address: u32, fixed: u32) {
    use crate::memory::{MemoryState, Permission};
    let Some(copy) = system.cro.copies.get_mut(&address) else { return };
    copy.fixed = fixed;
    let kept = zakuro_common::bits::align_up(fixed, 0x1000);
    if kept >= copy.block.size {
        return;
    }
    let block = copy.block;
    let tail = crate::memory::physical::PhysicalBlock { addr: block.addr + kept, size: block.size - kept };
    copy.block.size = kept;
    system.memory.unmap(address, block.size);
    system.memory.map(address, block.addr, kept, Permission::RW | Permission::EXECUTE, MemoryState::Code);
    system.memory.phys.free(tail);
}

/// hands an unloaded module back to the title in its buffer, the part
/// fixing left, as ldr:ro left it, and lets its memory go.
fn copy_out(system: &mut System, address: u32) {
    let Some(copy) = system.cro.copies.remove(&address) else { return };
    let mut bytes = vec![0u8; copy.fixed.min(copy.block.size) as usize];
    system.memory.read_bytes(address, &mut bytes);
    system.memory.write_bytes(copy.buffer, &bytes);
    system.memory.unmap(address, copy.block.size);
    system.memory.phys.free(copy.block);
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
