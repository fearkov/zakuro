//! turning a ROM into a running process.

use zakuro_common::memory_map::*;
use zakuro_common::ConsoleModel;
use zakuro_fs::{MemoryType, Title};

use crate::memory::{self, MemoryState, Permission};
use crate::{Config, System};

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error(transparent)]
    Fs(#[from] zakuro_fs::FsError),
    #[error("the code image is too small for the segments the exheader describes")]
    ShortCodeImage,
}

/// the main thread's stack ends where the shared-memory region begins, which
/// is what the retail kernel does.
const STACK_TOP: u32 = SHARED_MEMORY_VADDR;

pub fn load(path: impl AsRef<std::path::Path>, mut config: Config) -> Result<System, LoadError> {
    let title = Title::load(path)?;
    log::info!("loaded {}", title.describe());

    let exheader = &title.exheader;
    let app_bytes = exheader.system_mode.application_memory();
    let region = match exheader.memory_type {
        MemoryType::Application => memory::MemoryRegion::Application,
        MemoryType::System => memory::MemoryRegion::System,
        MemoryType::Base => memory::MemoryRegion::Base,
    };

    // a title marked New3DS-only has to run as one.
    if title.ncch.platform() == 2 {
        config.new3ds = true;
        config.model = ConsoleModel::New3ds;
    }

    let mut system = System::new(config);
    system.memory = memory::Memory::new(system.config.new3ds, app_bytes);
    system.kernel = crate::kernel::Kernel::new(
        title.program_id(),
        region,
        linear_heap_base(system.config.new3ds),
    );

    map_special_pages(&mut system, app_bytes);
    map_code(&mut system, &title)?;
    map_stack(&mut system, exheader.stack_size);

    system.kernel.heap_top = HEAP_VADDR;

    // the main thread starts at the beginning of .text.
    let entry = exheader.text.address;
    let priority = exheader.main_thread_priority as u32;
    let main = system.kernel.create_thread(
        "main",
        entry,
        STACK_TOP,
        0,
        priority,
        exheader.ideal_processor as i32,
    );
    system.map_tls_page(main);
    system.kernel.reschedule_pending = true;

    log::info!(
        "entry 0x{entry:08X}, stack top 0x{STACK_TOP:08X}, priority {priority}, {} MiB app memory",
        app_bytes / (1024 * 1024)
    );

    system.title = Some(title);
    Ok(system)
}

fn map_special_pages(system: &mut System, app_bytes: u32) {
    let model = system.config.model;
    let app_mem_type = system.title.as_ref().map_or(0, |t| {
        t.exheader.system_mode as u32
    });
    let sys = system
        .memory
        .phys
        .region_size(memory::MemoryRegion::System);
    let base = system.memory.phys.region_size(memory::MemoryRegion::Base);

    memory::config::init_config_mem(
        system.memory.phys.config_mem_mut(),
        model,
        app_mem_type,
        app_bytes,
        sys,
        base,
    );
    let slider = system.config.slider_3d;
    memory::config::init_shared_page(system.memory.phys.shared_page_mut(), model, slider);

    // both pages are read-only to the guest and live in AXI WRAM.
    system.memory.map(
        CONFIG_MEM_VADDR,
        AXI_WRAM_PADDR,
        CONFIG_MEM_SIZE,
        Permission::READ,
        MemoryState::Static,
    );
    system.memory.map(
        SHARED_PAGE_VADDR,
        AXI_WRAM_PADDR + (SHARED_PAGE_VADDR - CONFIG_MEM_VADDR),
        SHARED_PAGE_SIZE,
        Permission::READ,
        MemoryState::Static,
    );

    // VRAM is mapped into every process.
    system.memory.map(
        VRAM_VADDR,
        VRAM_PADDR,
        VRAM_SIZE,
        Permission::RW,
        MemoryState::Static,
    );
    // so is the DSP's memory.
    system.memory.map(
        DSP_RAM_VADDR,
        DSP_RAM_PADDR,
        DSP_RAM_SIZE,
        Permission::RW,
        MemoryState::Static,
    );
}

fn map_code(system: &mut System, title: &Title) -> Result<(), LoadError> {
    let code = title.code()?;
    let exheader = &title.exheader;

    let segments = [
        (".text", exheader.text, Permission::RX, 0u32),
        (
            ".rodata",
            exheader.rodata,
            Permission::READ,
            exheader.text.num_pages * PAGE_SIZE,
        ),
        (
            ".data",
            exheader.data,
            Permission::RW,
            (exheader.text.num_pages + exheader.rodata.num_pages) * PAGE_SIZE,
        ),
    ];

    for (name, info, permission, source_offset) in segments {
        if info.num_pages == 0 {
            continue;
        }
        let mapped_size = info.num_pages * PAGE_SIZE;
        let block = system
            .memory
            .phys
            .allocate(system.kernel.memory_region, mapped_size)
            .expect("FCRAM for a code segment");
        system.memory.map(
            info.address,
            block.addr,
            mapped_size,
            permission,
            MemoryState::Code,
        );

        // zero the whole mapped region first so the padding between the
        // segment's real size and its page-aligned size is defined.
        system.memory.zero_physical(block.addr, mapped_size);

        let start = source_offset as usize;
        let end = start + info.size as usize;
        let bytes = code.get(start..end).ok_or(LoadError::ShortCodeImage)?;
        // the write has to go to physical memory, .text and .rodata are about
        // to be visible to the guest without write permission.
        system.memory.write_physical(block.addr, bytes);

        log::debug!(
            "mapped {name} at 0x{:08X}, 0x{:X} bytes in 0x{mapped_size:X}",
            info.address,
            info.size
        );
    }

    // BSS follows .data and must start zeroed.
    if exheader.bss_size > 0 {
        let bss_start = exheader.data.address + exheader.data.num_pages * PAGE_SIZE;
        let bss_size = zakuro_common::bits::align_up(exheader.bss_size, PAGE_SIZE);
        let block = system
            .memory
            .phys
            .allocate(system.kernel.memory_region, bss_size)
            .expect("FCRAM for BSS");
        system.memory.map(
            bss_start,
            block.addr,
            bss_size,
            Permission::RW,
            MemoryState::Private,
        );
        system.memory.zero_physical(block.addr, bss_size);
        log::debug!("mapped .bss at 0x{bss_start:08X}, 0x{bss_size:X} bytes");
    }

    Ok(())
}

fn map_stack(system: &mut System, stack_size: u32) {
    let size = zakuro_common::bits::align_up(stack_size.max(0x4000), PAGE_SIZE);
    let base = STACK_TOP - size;
    let block = system
        .memory
        .phys
        .allocate(system.kernel.memory_region, size)
        .expect("FCRAM for the main stack");
    system.memory.map(
        base,
        block.addr,
        size,
        Permission::RW,
        MemoryState::Locked,
    );
    system.memory.zero_physical(block.addr, size);
    log::debug!("mapped the main stack at 0x{base:08X}, 0x{size:X} bytes");
}
