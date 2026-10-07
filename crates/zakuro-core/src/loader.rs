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
    let mut title = Title::load(path)?;
    log::info!("loaded {}", title.describe());
    crate::mods::lay(&mut title, config.data_dir.as_deref());

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
    let modded = map_code(&mut system, &title)?;
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

    if let Some(linked) = &system.config.linked {
        if linked.program_id() != title.program_id() {
            log::warn!(
                "the code linked in is for title {:016X}, not {:016X}, interpreting everything",
                linked.program_id(),
                title.program_id()
            );
        } else {
            match crate::recompiled::Library::linked(linked) {
                Ok(library) => {
                    log::info!("running the recompiled code linked in, {}", library.describe());
                    system.recompiled = Some(library);
                }
                Err(error) => log::warn!("could not use the code linked in, {error}, interpreting everything"),
            }
        }
    } else if let Some(path) = system.config.recompiled.clone().or_else(|| installed(&system, &title)) {
        let path = if path.is_dir() { path.join(recomp_abi::library_name(title.program_id())) } else { path };
        match crate::recompiled::Library::open(&path) {
            // code a mod changed would go on running as it was recompiled
            Ok(library) if modded && !library.checks_itself() => log::warn!(
                "a mod changed the game's code and {} can't tell where, which code recompiled again can, interpreting everything",
                path.display()
            ),
            Ok(mut library) => {
                log::info!("running recompiled code from {}, {}", path.display(), library.describe());
                if library.outdated() {
                    log::warn!("an older 3dsrecomp made {}, recompiling the game again brings the newest improvements", path.display());
                }
                let stale = library.check(&mut system.memory);
                if stale > 0 {
                    log::warn!(
                        "{stale} functions changed since the game was recompiled, by a mod or in another version of the game, they run in the interpreter"
                    );
                }
                system.recompiled = Some(library);
                let text = exheader_text(&title);
                system.hints = Some(crate::hints::Hints::new(&path, text));
            }
            Err(error) => log::warn!("could not load {}, {error}, interpreting everything", path.display()),
        }
    }

    let mut on_gpu = false;
    if system.config.hardware_renderer {
        match system.gpu.enable_hardware_renderer(system.config.resolution, system.config.device.clone()) {
            Ok(name) => {
                log::info!("drawing on {name} through Vulkan");
                on_gpu = true;
            }
            Err(error) => log::warn!("{error}, drawing in software"),
        }
    }
    if let Some(data_dir) = system.config.data_dir.as_deref().filter(|_| system.config.texture_packs) {
        let dir = crate::mods::textures(data_dir, title.program_id());
        if !on_gpu {
            if std::fs::read_dir(&dir).is_ok_and(|mut entries| entries.next().is_some()) {
                log::warn!("the texture pack in {} needs the 3D drawn on the GPU, it is left out", dir.display());
            }
        } else if let Some(pack) = zakuro_gpu::pack::Pack::open(&dir) {
            system.gpu.set_texture_pack(Some(std::sync::Arc::new(pack)));
        }
    }

    system.title = Some(title);
    Ok(system)
}

/// switches a running game to the library at path, which 3dsrecomp has just
/// built for it, between two steps, when no recompiled code is running. the
/// old library goes first, so that the new file is the one loaded even where
/// it took the old one's place, then the new one is checked against memory
/// and told where the modules already loaded are, as at boot. the game goes
/// on from where it was. an error says why it can't.
pub fn swap_recompiled(system: &mut System, path: &std::path::Path) -> Result<(), String> {
    if system.config.linked.is_some() || !system.config.find_recompiled {
        return Err("the game runs code linked in, or a library given to it, or nothing but the interpreter".to_owned());
    }
    let title = system.title.as_ref().ok_or("no game is running")?;
    let text = exheader_text(title);
    if let Some(hints) = &mut system.hints {
        hints.save();
    }
    system.hints = None;
    system.recompiled = None;
    let mut library = crate::recompiled::Library::open(path).map_err(|error| format!("could not load {}, {error}", path.display()))?;
    if !library.checks_itself() {
        let modded = crate::mods::code(title, system.config.data_dir.as_deref()).is_ok_and(|(_, modded)| modded);
        if modded {
            return Err(format!("a mod changed the game's code and {} can't tell where", path.display()));
        }
    }
    log::info!("running recompiled code from {} from here on, {}", path.display(), library.describe());
    let stale = library.check(&mut system.memory);
    if stale > 0 {
        log::warn!("{stale} functions changed since the game was recompiled, by a mod or in another version of the game, they run in the interpreter");
    }
    for module in &system.cro.modules {
        library.place(&module.name, module.base, &mut system.memory);
    }
    system.hints = Some(crate::hints::Hints::new(path, text));
    system.recompiled = Some(library);
    Ok(())
}

/// where the executable's code is.
fn exheader_text(title: &Title) -> std::ops::Range<u32> {
    let text = &title.exheader.text;
    text.address..text.address + text.num_pages * PAGE_SIZE
}

/// the library 3dsrecomp build installed for the title, when the system is
/// to look for one.
fn installed(system: &System, title: &Title) -> Option<std::path::PathBuf> {
    if !system.config.find_recompiled {
        return None;
    }
    let found = recomp_abi::installed(title.program_id());
    if found.is_none() {
        log::info!(
            "no recompiled code for {:016X}, interpreting, 3dsrecomp build makes it",
            title.program_id()
        );
    }
    found
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
    let clock = system.boot_clock;
    memory::config::init_shared_page(system.memory.phys.shared_page_mut(), model, slider, clock);

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

/// maps the code segments, saying whether a mod changed them.
fn map_code(system: &mut System, title: &Title) -> Result<bool, LoadError> {
    let (code, modded) = crate::mods::code(title, system.config.data_dir.as_deref())?;
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
        code.get(start..end).ok_or(LoadError::ShortCodeImage)?;
        // the image's bytes to the end of the segment's pages, as the console
        // loads them, a mod's code among them in what padded the segment.
        // the write has to go to physical memory, .text and .rodata are about
        // to be visible to the guest without write permission.
        let paged = &code[start..code.len().min(start + mapped_size as usize)];
        system.memory.write_physical(block.addr, paged);

        log::debug!(
            "mapped {name} at 0x{:08X}, 0x{:X} bytes in 0x{mapped_size:X}",
            info.address,
            info.size
        );
    }

    // BSS starts right where .data's bytes end, in the rest of its last page,
    // and must start zeroed. what does not fit there gets pages of its own.
    let bss_start = exheader.data.address + exheader.data.num_pages * PAGE_SIZE;
    let bss_end = exheader.data.address + exheader.data.size + exheader.bss_size;
    if bss_end > bss_start {
        let bss_size = zakuro_common::bits::align_up(bss_end - bss_start, PAGE_SIZE);
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
        // the image goes on past .data's pages into the BSS, flat, as Luma3DS
        // and Citra lay it, where mods made with Magikoopa put their code, in
        // BSS their exheader.bin makes bigger
        let past_data = ((exheader.text.num_pages + exheader.rodata.num_pages + exheader.data.num_pages) * PAGE_SIZE) as usize;
        if let Some(tail) = code.get(past_data..).filter(|tail| !tail.is_empty()) {
            let fits = tail.len().min(bss_size as usize);
            system.memory.write_physical(block.addr, &tail[..fits]);
            log::info!("0x{fits:X} bytes of code past .data, from a mod, are in the BSS at 0x{bss_start:08X}");
            if fits < tail.len() {
                log::warn!("0x{:X} bytes of a mod's code reach past the BSS and are left out", tail.len() - fits);
            }
        }
    }

    Ok(modded)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// the id of the game the tests make, which no real game has.
    const PROGRAM_ID: u64 = 0x0004_0000_0FF3_DE01;

    fn put(out: &mut [u8], at: usize, bytes: &[u8]) {
        out[at..at + bytes.len()].copy_from_slice(bytes);
    }

    /// writes a decrypted .cxi whose code is all text and that has no RomFS.
    fn rom(test: &str, code: &[u8]) -> std::path::PathBuf {
        // the NCCH header, the exheader after it, and the ExeFS at 0x600
        // with .code its only file
        let mut out = vec![0; 0x800];
        put(&mut out, 0x100, b"NCCH");
        put(&mut out, 0x118, &PROGRAM_ID.to_le_bytes());
        out[0x18F] = 0x04;
        put(&mut out, 0x1A0, &3u32.to_le_bytes());
        put(&mut out, 0x210, &[0x0010_0000, 1, code.len() as u32].map(u32::to_le_bytes).concat());
        put(&mut out, 0x600, b".code");
        put(&mut out, 0x60C, &(code.len() as u32).to_le_bytes());
        out.extend_from_slice(code);
        let path = std::env::temp_dir().join(format!("zakuro-loader-{}-{test}.cxi", std::process::id()));
        std::fs::write(&path, out).unwrap();
        path
    }

    /// a mod's code past .data, in a BSS its exheader.bin makes bigger, is
    /// in memory once the game loads, the way Magikoopa's mods put it.
    #[test]
    fn code_a_mod_puts_past_data_lands_in_the_bss() {
        let rom = rom("bss-code", &[0; 8]);
        let data_dir = std::env::temp_dir().join(format!("zakuro-loader-{}-bss-code", std::process::id()));
        let mods = crate::mods::dir(&data_dir, PROGRAM_ID);
        std::fs::create_dir_all(&mods).unwrap();
        // the text in one page, then two pages of BSS right after it
        let mut exheader = vec![0; 0x400];
        put(&mut exheader, 0x10, &[0x0010_0000u32, 1, 8].map(u32::to_le_bytes).concat());
        put(&mut exheader, 0x20, &[0x0010_1000u32, 0, 0].map(u32::to_le_bytes).concat());
        put(&mut exheader, 0x30, &[0x0010_1000u32, 0, 0].map(u32::to_le_bytes).concat());
        put(&mut exheader, 0x3C, &0x2000u32.to_le_bytes());
        std::fs::write(mods.join("exheader.bin"), &exheader).unwrap();
        // the text, a stub in its padding, and code past it
        let mut code = vec![0; 0x1000];
        put(&mut code, 0x800, &[0x22; 4]);
        code.extend_from_slice(&[0x11; 16]);
        std::fs::write(mods.join("code.bin"), &code).unwrap();

        let config = Config { data_dir: Some(data_dir.clone()), ..Config::default() };
        let mut system = load(&rom, config).unwrap();
        let mut read = [0; 16];
        system.memory.read_bytes(0x0010_1000, &mut read);
        assert_eq!(read, [0x11; 16]);
        let mut stub = [0; 4];
        system.memory.read_bytes(0x0010_0800, &mut stub);
        assert_eq!(stub, [0x22; 4]);

        std::fs::remove_dir_all(data_dir).unwrap();
        std::fs::remove_file(rom).unwrap();
    }
}
