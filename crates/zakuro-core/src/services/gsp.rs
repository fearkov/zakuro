//! gsp::Gpu, the GPU service.

use zakuro_common::memory_map::{FCRAM_PADDR, PAGE_SIZE};
use zakuro_cpu::Bus;

use crate::kernel::ipc::{CommandBuffer, Descriptor, Header};
use crate::kernel::object::{KObject, SharedMemory};
use crate::System;

/// size of the GSP shared memory block.
pub const SHARED_MEMORY_SIZE: u32 = 0x1000;
/// start of the GX command ring within it.
const COMMAND_BUFFER_BASE: u32 = 0x800;
const COMMAND_BUFFER_STRIDE: u32 = 0x200;
const INTERRUPT_QUEUE_STRIDE: u32 = 0x40;

/// start of the FrameBufferInfo table, one 0x40-byte record per screen per
/// client, [client][screen].
const FRAMEBUFFER_INFO_BASE: u32 = 0x200;
const FRAMEBUFFER_INFO_CLIENT_STRIDE: u32 = 0x80;
const FRAMEBUFFER_INFO_SCREEN_STRIDE: u32 = 0x40;
/// offset of framebuf_info[0] within a screen's record, framebuf_info[1]
/// follows immediately at + 0x1C.
const FRAMEBUFFER_ENTRY_BASE: u32 = 0x04;
const FRAMEBUFFER_ENTRY_STRIDE: u32 = 0x1C;

/// interrupts GSP relays to the client.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum InterruptId {
    Psc0 = 0,
    Psc1 = 1,
    /// top screen vertical blank.
    Pdc0 = 2,
    /// bottom screen vertical blank.
    Pdc1 = 3,
    /// display transfer / texture copy finished.
    Ppf = 4,
    /// command list finished.
    P3d = 5,
    Dma = 6,
}

/// one entry from the GX command ring.
#[derive(Debug, Clone, Copy)]
pub struct GxCommand {
    pub id: u8,
    pub data: [u32; 7],
}

#[derive(Default)]
pub struct GspState {
    pub shared_memory_handle: Option<u32>,
    pub shared_memory_address: u32,
    /// event the client waits on for relayed interrupts.
    pub interrupt_event: Option<u32>,
    pub interrupt_event_object: Option<crate::kernel::object::ObjectId>,
    /// which slot in the shared memory this client owns.
    pub thread_index: u32,
    /// true once a client has taken GPU rights.
    pub has_right: bool,
    /// commands pulled off the ring that the GPU has not run yet.
    pub pending: Vec<GxCommand>,
}

/// command ids, straight from 3dbrew's GSP::GPU table.
mod command {
    pub const WRITE_HW_REGS: u16 = 0x0001;
    pub const WRITE_HW_REGS_WITH_MASK: u16 = 0x0002;
    pub const WRITE_HW_REG_REPEAT: u16 = 0x0003;
    pub const READ_HW_REGS: u16 = 0x0004;
    pub const SET_BUFFER_SWAP: u16 = 0x0005;
    pub const SET_COMMAND_LIST: u16 = 0x0006;
    pub const REQUEST_DMA: u16 = 0x0007;
    pub const FLUSH_DATA_CACHE: u16 = 0x0008;
    pub const INVALIDATE_DATA_CACHE: u16 = 0x0009;
    pub const REGISTER_INTERRUPT_EVENTS: u16 = 0x000A;
    pub const SET_LCD_FORCE_BLACK: u16 = 0x000B;
    pub const TRIGGER_CMD_REQ_QUEUE: u16 = 0x000C;
    pub const SET_DISPLAY_TRANSFER: u16 = 0x000D;
    pub const SET_TEXTURE_COPY: u16 = 0x000E;
    pub const SET_MEMORY_FILL: u16 = 0x000F;
    pub const SET_AXI_CONFIG_QOS_MODE: u16 = 0x0010;
    pub const SET_PERF_LOG_MODE: u16 = 0x0011;
    pub const GET_PERF_LOG: u16 = 0x0012;
    pub const REGISTER_INTERRUPT_RELAY_QUEUE: u16 = 0x0013;
    pub const UNREGISTER_INTERRUPT_RELAY_QUEUE: u16 = 0x0014;
    pub const TRY_ACQUIRE_RIGHT: u16 = 0x0015;
    pub const ACQUIRE_RIGHT: u16 = 0x0016;
    pub const RELEASE_RIGHT: u16 = 0x0017;
    pub const IMPORT_DISPLAY_CAPTURE_INFO: u16 = 0x0018;
    pub const SAVE_VRAM_SYS_AREA: u16 = 0x0019;
    pub const RESTORE_VRAM_SYS_AREA: u16 = 0x001A;
    pub const RESET_GPU_CORE: u16 = 0x001B;
    pub const SET_LED_FORCE_OFF: u16 = 0x001C;
    pub const SET_INTERNAL_PRIORITIES: u16 = 0x001E;
    pub const STORE_DATA_CACHE: u16 = 0x001F;
}

pub fn handle(system: &mut System, buffer: &CommandBuffer, header: Header) -> bool {
    use command::*;
    let id = header.command_id();
    match id {
        // WriteHWRegs(offset, size, <static buffer>)
        WRITE_HW_REGS => {
            let offset = buffer.get(&mut system.memory, 1);
            let size = buffer.get(&mut system.memory, 2);
            let source = buffer.get(&mut system.memory, 4);
            write_hw_regs(system, offset, size, source, None);
            buffer.reply(&mut system.memory, id, &[]);
            true
        }
        // WriteHWRegsWithMask(offset, size, <buffer>, <mask buffer>)
        WRITE_HW_REGS_WITH_MASK => {
            let offset = buffer.get(&mut system.memory, 1);
            let size = buffer.get(&mut system.memory, 2);
            let source = buffer.get(&mut system.memory, 4);
            let mask = buffer.get(&mut system.memory, 6);
            write_hw_regs(system, offset, size, source, Some(mask));
            buffer.reply(&mut system.memory, id, &[]);
            true
        }
        // WriteHWRegRepeat(offset, size, <buffer>), writes the same register
        // repeatedly, which is how command data is fed to the GPU.
        WRITE_HW_REG_REPEAT => {
            let offset = buffer.get(&mut system.memory, 1);
            let size = buffer.get(&mut system.memory, 2);
            let source = buffer.get(&mut system.memory, 4);
            for i in 0..size / 4 {
                let value = system.memory.read32(source + i * 4);
                system.gpu_write_register(offset, value);
            }
            buffer.reply(&mut system.memory, id, &[]);
            true
        }
        // ReadHWRegs(offset, size)
        READ_HW_REGS => {
            let offset = buffer.get(&mut system.memory, 1);
            let size = buffer.get(&mut system.memory, 2);
            let dest = buffer.get(&mut system.memory, 4);
            for i in 0..size / 4 {
                let value = system.gpu_read_register(offset + i * 4);
                system.memory.write32(dest + i * 4, value);
            }
            buffer.reply(&mut system.memory, id, &[]);
            true
        }
        // SetBufferSwap(screen, FrameBufferInfo)
        SET_BUFFER_SWAP => {
            let screen = buffer.get(&mut system.memory, 1);
            let active = buffer.get(&mut system.memory, 2);
            let left = buffer.get(&mut system.memory, 3);
            let right = buffer.get(&mut system.memory, 4);
            let stride = buffer.get(&mut system.memory, 5);
            let format = buffer.get(&mut system.memory, 6);
            log::debug!(
                "gsp: SetBufferSwap screen {screen} active {active} left 0x{left:08X} right \
                 0x{right:08X} stride {stride} format 0x{format:X}"
            );
            system.set_framebuffer(screen, active, left, right, stride, format);
            buffer.reply(&mut system.memory, id, &[]);
            true
        }
        // SetCommandList(address, size, ...), the immediate form of the GX
        // command with the same name.
        SET_COMMAND_LIST => {
            let address = buffer.get(&mut system.memory, 1) & !7;
            let size = buffer.get(&mut system.memory, 2) & !3;
            system.submit_command_list(address, size);
            signal_interrupt(system, InterruptId::P3d);
            buffer.reply(&mut system.memory, id, &[]);
            true
        }
        // RequestDma(source, destination, size)
        REQUEST_DMA => {
            let source = buffer.get(&mut system.memory, 1);
            let dest = buffer.get(&mut system.memory, 2);
            let size = buffer.get(&mut system.memory, 3);
            let mut data = vec![0u8; size as usize];
            system.memory.read_bytes(source, &mut data);
            system.memory.write_bytes(dest, &data);
            signal_interrupt(system, InterruptId::Dma);
            buffer.reply(&mut system.memory, id, &[]);
            true
        }
        // the cache maintenance commands, memory is coherent for us.
        FLUSH_DATA_CACHE | INVALIDATE_DATA_CACHE | STORE_DATA_CACHE => {
            buffer.reply(&mut system.memory, id, &[]);
            true
        }
        // RegisterInterruptEvents
        REGISTER_INTERRUPT_EVENTS => {
            buffer.reply(&mut system.memory, id, &[]);
            true
        }
        SET_LCD_FORCE_BLACK => {
            system.lcd_force_black = buffer.get(&mut system.memory, 1) != 0;
            buffer.reply(&mut system.memory, id, &[]);
            true
        }
        // TriggerCmdReqQueue, run whatever the client queued in shared memory.
        TRIGGER_CMD_REQ_QUEUE => {
            process_command_queue(system);
            buffer.reply(&mut system.memory, id, &[]);
            true
        }
        // SetDisplayTransfer(input, output, input dim, output dim, flags)
        SET_DISPLAY_TRANSFER => {
            let input = buffer.get(&mut system.memory, 1);
            let output = buffer.get(&mut system.memory, 2);
            let input_dimensions = buffer.get(&mut system.memory, 3);
            let output_dimensions = buffer.get(&mut system.memory, 4);
            let flags = buffer.get(&mut system.memory, 5);
            system.display_transfer(input, output, input_dimensions, output_dimensions, flags);
            signal_interrupt(system, InterruptId::Ppf);
            buffer.reply(&mut system.memory, id, &[]);
            true
        }
        // SetTextureCopy(input, output, size, input gap, output gap, flags)
        SET_TEXTURE_COPY => {
            let input = buffer.get(&mut system.memory, 1);
            let output = buffer.get(&mut system.memory, 2);
            let size = buffer.get(&mut system.memory, 3);
            let input_gap = buffer.get(&mut system.memory, 4);
            let output_gap = buffer.get(&mut system.memory, 5);
            system.texture_copy(input, output, size, input_gap, output_gap);
            signal_interrupt(system, InterruptId::Ppf);
            buffer.reply(&mut system.memory, id, &[]);
            true
        }
        // SetMemoryFill(start0, value0, end0, start1, value1, end1, control0,
        // control1)
        SET_MEMORY_FILL => {
            for bank in 0..2u32 {
                let start = buffer.get(&mut system.memory, 1 + bank * 3);
                let value = buffer.get(&mut system.memory, 2 + bank * 3);
                let end = buffer.get(&mut system.memory, 3 + bank * 3);
                let control = buffer.get(&mut system.memory, 7 + bank);
                if start == 0 || end <= start {
                    continue;
                }
                let width = fill_width(control);
                system.memory_fill(start, end, value, width);
                signal_interrupt(
                    system,
                    if bank == 0 {
                        InterruptId::Psc0
                    } else {
                        InterruptId::Psc1
                    },
                );
            }
            buffer.reply(&mut system.memory, id, &[]);
            true
        }
        // configuration knobs with no effect on us.
        SET_AXI_CONFIG_QOS_MODE
        | SET_PERF_LOG_MODE
        | SET_LED_FORCE_OFF
        | SET_INTERNAL_PRIORITIES
        | SAVE_VRAM_SYS_AREA
        | RESTORE_VRAM_SYS_AREA
        | RESET_GPU_CORE => {
            buffer.reply(&mut system.memory, id, &[]);
            true
        }
        GET_PERF_LOG => {
            buffer.reply(&mut system.memory, id, &[0]);
            true
        }
        // RegisterInterruptRelayQueue(flags, event handle)
        //   -> result, thread index, shared memory handle
        REGISTER_INTERRUPT_RELAY_QUEUE => {
            let event = buffer.get(&mut system.memory, 3);
            system.services.gsp.interrupt_event = Some(event);
            system.services.gsp.interrupt_event_object = system.kernel.resolve(event);

            let handle = ensure_shared_memory(system);
            let index = system.services.gsp.thread_index;
            log::debug!("gsp: registered interrupt relay queue, slot {index}");
            // 0x2A07 alongside success tells the caller the GPU was already
            // initialized, so it skips its own first-time setup.
            buffer.set(&mut system.memory, 0, Header::new(id, 2, 2).0);
            buffer.set(&mut system.memory, 1, 0x2A07);
            buffer.set(&mut system.memory, 2, index);
            buffer.set(&mut system.memory, 3, Descriptor::handles(1));
            buffer.set(&mut system.memory, 4, handle);
            true
        }
        UNREGISTER_INTERRUPT_RELAY_QUEUE => {
            system.services.gsp.interrupt_event = None;
            system.services.gsp.interrupt_event_object = None;
            buffer.reply(&mut system.memory, id, &[]);
            true
        }
        TRY_ACQUIRE_RIGHT | ACQUIRE_RIGHT => {
            system.services.gsp.has_right = true;
            buffer.reply(&mut system.memory, id, &[]);
            true
        }
        RELEASE_RIGHT => {
            system.services.gsp.has_right = false;
            buffer.reply(&mut system.memory, id, &[]);
            true
        }
        IMPORT_DISPLAY_CAPTURE_INFO => {
            let info = system.display_capture_info();
            buffer.set(&mut system.memory, 0, Header::new(id, 9, 0).0);
            buffer.set(&mut system.memory, 1, 0);
            for (i, value) in info.iter().enumerate() {
                buffer.set(&mut system.memory, i as u32 + 2, *value);
            }
            true
        }
        _ => false,
    }
}

/// applies any buffer swap a game has queued in the FrameBufferInfo table for
/// either screen, the way real GSP does once per vertical blank.
pub fn apply_framebuffer_updates(system: &mut System) {
    let base = system.services.gsp.shared_memory_address;
    if base == 0 {
        return;
    }
    let client = system.services.gsp.thread_index * FRAMEBUFFER_INFO_CLIENT_STRIDE;

    for screen in 0..2u32 {
        let record = base + FRAMEBUFFER_INFO_BASE + client + screen * FRAMEBUFFER_INFO_SCREEN_STRIDE;
        let header = system.memory.read32(record);
        let index = header & 0xFF;
        let is_dirty = (header >> 8) & 0xFF;
        if is_dirty == 0 {
            continue;
        }

        let entry = record + FRAMEBUFFER_ENTRY_BASE + index * FRAMEBUFFER_ENTRY_STRIDE;
        let left = system.memory.read32(entry + 0x04);
        let right = system.memory.read32(entry + 0x08);
        let stride = system.memory.read32(entry + 0x0C);
        let format = system.memory.read32(entry + 0x10);
        system.set_framebuffer(screen, index, left, right, stride, format);

        // real GSP clears the flag once it has picked the update up, so a
        // title that polls it to know its previous frame was consumed keeps
        // moving instead of assuming GSP is stuck.
        system.memory.write32(record, index);
    }
}

/// decodes the fill width from a memory-fill control word.
fn fill_width(control: u32) -> u32 {
    match control & 0x300 {
        0x000 => 2, // 16 bits per pixel
        0x100 => 3, // 24 bits
        _ => 4,     // 32 bits
    }
}

/// creates the shared memory block on first use and maps it for the client.
fn ensure_shared_memory(system: &mut System) -> u32 {
    if let Some(handle) = system.services.gsp.shared_memory_handle {
        return handle;
    }

    let block = system
        .memory
        .phys
        .allocate(crate::memory::MemoryRegion::Base, SHARED_MEMORY_SIZE)
        .expect("GSP shared memory");

    let object = system
        .kernel
        .objects
        .insert(KObject::SharedMemory(SharedMemory {
            name: "GSP".into(),
            address: 0,
            size: SHARED_MEMORY_SIZE,
            paddr: block.addr,
            mapped_at: None,
        }));
    let handle = system
        .kernel
        .handles
        .create(&mut system.kernel.objects, object, "GSP shared memory");
    system.services.gsp.shared_memory_handle = Some(handle);
    handle
}

/// address of this client's interrupt queue inside the shared block.
fn interrupt_queue_address(system: &System) -> Option<u32> {
    let base = system.services.gsp.shared_memory_address;
    (base != 0).then(|| base + system.services.gsp.thread_index * INTERRUPT_QUEUE_STRIDE)
}

/// posts an interrupt to the client and signals its event.
pub fn signal_interrupt(system: &mut System, interrupt: InterruptId) {
    let Some(queue) = interrupt_queue_address(system) else {
        return;
    };

    let index = system.memory.read8(queue) as u32;
    let count = system.memory.read8(queue + 1) as u32;

    if count >= 0x34 {
        // the queue is full, hardware sets the error byte and drops it.
        system.memory.write8(queue + 2, 1);
        return;
    }

    let slot = (index + count) % 0x34;
    system.memory.write8(queue + 0xC + slot, interrupt as u8);
    system.memory.write8(queue + 1, (count + 1) as u8);

    if let Some(object) = system.services.gsp.interrupt_event_object {
        system.kernel.signal_event(object);
    }
}

fn write_hw_regs(system: &mut System, offset: u32, size: u32, source: u32, mask: Option<u32>) {
    // GSP only lets a client touch the GPU register window.
    if offset >= 0x420000 || size > 0x80 {
        log::warn!("gsp WriteHWRegs out of range: offset 0x{offset:X} size 0x{size:X}");
        return;
    }
    for i in 0..size / 4 {
        let value = system.memory.read32(source + i * 4);
        let register = offset + i * 4;
        match mask {
            Some(mask_ptr) => {
                let mask = system.memory.read32(mask_ptr + i * 4);
                let old = system.gpu_read_register(register);
                system.gpu_write_register(register, (old & !mask) | (value & mask));
            }
            None => system.gpu_write_register(register, value),
        }
    }
}

/// pulls every queued GX command off the ring and runs it.
fn process_command_queue(system: &mut System) {
    let base = system.services.gsp.shared_memory_address;
    if base == 0 {
        return;
    }
    let ring = base + COMMAND_BUFFER_BASE
        + system.services.gsp.thread_index * COMMAND_BUFFER_STRIDE;

    loop {
        let control = system.memory.read32(ring);
        let index = control & 0xFF;
        let count = (control >> 8) & 0xFF;
        if count == 0 {
            break;
        }

        let entry = ring + 0x20 + (index % 15) * 0x20;
        let header = system.memory.read32(entry);
        let mut data = [0u32; 7];
        for (i, word) in data.iter_mut().enumerate() {
            *word = system.memory.read32(entry + 4 + i as u32 * 4);
        }

        let command = GxCommand {
            id: (header & 0xFF) as u8,
            data,
        };

        // consume it before running, so a command that re-enters GSP sees a
        // consistent ring.
        let next_index = (index + 1) % 15;
        system
            .memory
            .write32(ring, next_index | ((count - 1) << 8));

        execute_command(system, command);
    }
}

fn execute_command(system: &mut System, command: GxCommand) {
    log::trace!(
        "gx command 0x{:02X}: {:08X?}",
        command.id,
        &command.data[..5]
    );
    match command.id {
        // RequestDma(source, destination, size)
        0x00 => {
            let (src, dst, size) = (command.data[0], command.data[1], command.data[2]);
            let mut buf = vec![0u8; size as usize];
            system.memory.read_bytes(src, &mut buf);
            system.memory.write_bytes(dst, &buf);
            signal_interrupt(system, InterruptId::Dma);
        }
        // ProcessCommandList(address, size, ...)
        0x01 => {
            let address = command.data[0] & !7;
            let size = command.data[1] & !3;
            system.submit_command_list(address, size);
            signal_interrupt(system, InterruptId::P3d);
        }
        // MemoryFill(start0, value0, end0, start1, value1, end1, control)
        0x02 => {
            for bank in 0..2 {
                let start = command.data[bank * 3];
                let value = command.data[bank * 3 + 1];
                let end = command.data[bank * 3 + 2];
                if start == 0 || end <= start {
                    continue;
                }
                let width = fill_width(command.data[6] >> (bank as u32 * 16));
                system.memory_fill(start, end, value, width);
                signal_interrupt(
                    system,
                    if bank == 0 {
                        InterruptId::Psc0
                    } else {
                        InterruptId::Psc1
                    },
                );
            }
        }
        // DisplayTransfer(input, output, input dim, output dim, flags)
        0x03 => {
            system.display_transfer(
                command.data[0],
                command.data[1],
                command.data[2],
                command.data[3],
                command.data[4],
            );
            signal_interrupt(system, InterruptId::Ppf);
        }
        // TextureCopy(input, output, size, input gap, output gap, flags)
        0x04 => {
            system.texture_copy(
                command.data[0],
                command.data[1],
                command.data[2],
                command.data[3],
                command.data[4],
            );
            signal_interrupt(system, InterruptId::Ppf);
        }
        // CacheFlush
        0x05 => {}
        other => log::warn!("unknown GX command 0x{other:02X}"),
    }
}

/// converts a physical address in a GX command to the linear-heap virtual
/// address it corresponds to, which is where our memory actually is.
pub fn physical_to_virtual(system: &System, paddr: u32) -> u32 {
    if paddr >= FCRAM_PADDR {
        system.kernel.linear_base + (paddr - FCRAM_PADDR)
    } else if (0x1800_0000..0x1860_0000).contains(&paddr) {
        zakuro_common::memory_map::VRAM_VADDR + (paddr - 0x1800_0000)
    } else {
        paddr
    }
}

/// rounds a size up to whole pages, used when mapping the shared block.
pub const fn page_align(size: u32) -> u32 {
    (size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{MemoryState, Permission};
    use crate::{Config, System};

    /// maps the GSP shared memory block and points shared_memory_address
    /// at it, the way svcMapMemoryBlock does for a real title, so a test
    /// can write into the FrameBufferInfo table the way a game would.
    fn system_with_mapped_gsp_shm() -> (System, u32) {
        let mut system = System::new(Config::default());
        ensure_shared_memory(&mut system);
        let handle = system.services.gsp.shared_memory_handle.unwrap();
        let object = system.kernel.resolve(handle).unwrap();
        let paddr = match system.kernel.objects.get(object) {
            Some(KObject::SharedMemory(block)) => block.paddr,
            _ => panic!("expected the GSP block to exist"),
        };
        let vaddr = 0x1000_0000;
        system.memory.map(
            vaddr,
            paddr,
            SHARED_MEMORY_SIZE,
            Permission::READ | Permission::WRITE,
            MemoryState::Shared,
        );
        system.services.gsp.shared_memory_address = vaddr;
        (system, vaddr)
    }

    /// this is the exact layout recovered by scanning a running Pokémon Alpha
    /// Sapphire's GSP shared memory for the physical addresses a real
    /// display transfer had just written into, a title toggles is_dirty
    /// and index in the FrameBufferInfo header instead of calling
    /// SetBufferSwap every frame, and a vblank that does not pick this up
    /// leaves the screen showing whatever address was configured at launch
    /// forever, no matter how correctly the rest of the GPU pipeline renders.
    #[test]
    fn picks_up_a_swap_queued_directly_in_shared_memory() {
        let (mut system, vaddr) = system_with_mapped_gsp_shm();
        let record = vaddr + FRAMEBUFFER_INFO_BASE; // screen 0, client 0
        system.memory.write32(record, 0x0000_0101); // index 1, is_dirty 1
        let entry = record + FRAMEBUFFER_ENTRY_BASE + FRAMEBUFFER_ENTRY_STRIDE; // buf[1]
        system.memory.write32(entry + 0x04, 0x1F30_0000); // address_left
        system.memory.write32(entry + 0x0C, 720); // stride
        system.memory.write32(entry + 0x10, 0x341); // format

        apply_framebuffer_updates(&mut system);

        assert_eq!(system.gpu.framebuffers[0].address_left(), 0x1F30_0000);
        assert_eq!(system.gpu.framebuffers[0].stride, 720);
        assert_eq!(
            system.memory.read32(record) & 0xFF00,
            0,
            "is_dirty should be cleared once the swap is applied"
        );
    }

    /// a record with is_dirty clear (the steady state right after a swap
    /// was applied) must not be re-applied every frame, that would fight a
    /// title that legitimately wants to hold the same buffer for two frames.
    #[test]
    fn leaves_a_clean_record_alone() {
        let (mut system, vaddr) = system_with_mapped_gsp_shm();
        system.set_framebuffer(0, 0, 0x1234_0000, 0, 720, 0x341);

        apply_framebuffer_updates(&mut system);

        assert_eq!(system.gpu.framebuffers[0].address_left(), 0x1234_0000);
        let _ = vaddr;
    }
}
