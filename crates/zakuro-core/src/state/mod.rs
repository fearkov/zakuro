//! saved states, the whole console as it was at a moment, to go back to. a
//! file holds a header, then the state of everything but memory as
//! postcard writes it, then each memory, all of them in LZ4. a state is for
//! the game it was saved from and the version of Zakuro that saved it.

pub(crate) mod arrays;

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use std::collections::BTreeMap;

use crate::cheats::Overwrote;
use crate::cro::CroManager;
use crate::kernel::Kernel;
use crate::memory::MapState;
use crate::services::fs::SavedFile;
use crate::services::ServiceState;
use crate::{GuestMemory, Pacing, System, HISTORY_LENGTH};

const MAGIC: &[u8; 8] = b"ZAKUROST";
/// what a state holds changes from one version to the next.
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// everything but memory, as it is written.
#[derive(Serialize)]
struct SavedRef<'a> {
    cpu: &'a zakuro_cpu::Cpu,
    kernel: &'a Kernel,
    services: &'a ServiceState,
    files: Vec<(u32, SavedFile)>,
    cro: &'a CroManager,
    gpu: zakuro_gpu::state::GpuState,
    map: MapState,
    console: Console,
}

/// the same, as it is read back.
#[derive(Deserialize)]
struct Saved {
    cpu: zakuro_cpu::Cpu,
    kernel: Kernel,
    services: ServiceState,
    files: Vec<(u32, SavedFile)>,
    cro: CroManager,
    gpu: zakuro_gpu::state::GpuState,
    map: MapState,
    console: Console,
}

/// the system's own, how far it ran and what it is due to do next.
#[derive(Serialize, Deserialize)]
struct Console {
    exited: bool,
    broke: bool,
    debug_output: String,
    fatal_errors: Vec<String>,
    lcd_force_black: bool,
    frames: u64,
    boot_clock: u64,
    next_frame_boundary: u64,
    next_audio_frame: u64,
    next_preempt: u64,
    #[serde(with = "arrays")]
    history: [u32; HISTORY_LENGTH],
    history_index: u64,
    interpreted_instructions: u64,
    recompiled_instructions: u64,
    /// what each enhancement overwrote, by its name, so that turning it
    /// off after loading puts that back too.
    enhancements: Vec<(String, BTreeMap<u32, Overwrote>)>,
}

/// the ROM images and the RomFS layers the title's files read, in the
/// order a state names them by.
type Images = (Vec<Arc<zakuro_fs::RomImage>>, Vec<Arc<zakuro_fs::layered::Layered>>);

fn images(system: &System) -> Images {
    let (mut images, mut layers) = (Vec::new(), Vec::new());
    if let Some(title) = &system.title {
        images.push(title.shared_image());
        layers.extend(title.layered.clone());
        if let Some(update) = title.update() {
            images.push(update.title.shared_image());
            layers.extend(update.title.layered.clone());
        }
    }
    images.extend(system.dlc.iter().map(|dlc| dlc.image()));
    (images, layers)
}

impl System {
    /// the console as it is, between frames, as a saved state.
    pub fn save_state(&mut self) -> Result<Vec<u8>, String> {
        // what the host's GPU drew comes down first, so that memory holds
        // all of it
        let linear_base = self.kernel.linear_base;
        self.gpu.write_back_all(&mut GuestMemory { memory: &mut self.memory, linear_base, writes: 0 });
        let (images, layers) = images(self);
        let mut files = Vec::with_capacity(self.services.fs.files.len());
        for (&handle, file) in &self.services.fs.files {
            let saved = file.save((&images, &layers)).ok_or_else(|| format!("the open file {} reads from no image of the game's", file.path))?;
            files.push((handle, saved));
        }
        let console = Console {
            exited: self.exited,
            broke: self.broke,
            debug_output: self.debug_output.clone(),
            fatal_errors: self.fatal_errors.clone(),
            lcd_force_black: self.lcd_force_black,
            frames: self.frames,
            boot_clock: self.boot_clock,
            next_frame_boundary: self.next_frame_boundary,
            next_audio_frame: self.next_audio_frame,
            next_preempt: self.next_preempt,
            history: self.history,
            history_index: self.history_index as u64,
            interpreted_instructions: self.interpreted_instructions,
            recompiled_instructions: self.recompiled_instructions,
            enhancements: self
                .cheats
                .iter()
                .filter(|cheat| cheat.builtin && !cheat.overwrote.is_empty())
                .map(|cheat| (cheat.name.clone(), cheat.overwrote.clone()))
                .collect(),
        };
        let saved = SavedRef {
            cpu: &self.cpu,
            kernel: &self.kernel,
            services: &self.services,
            files,
            cro: &self.cro,
            gpu: self.gpu.save_state(),
            map: self.memory.save_map(),
            console,
        };
        let state = postcard::to_allocvec(&saved).map_err(|error| format!("the state could not be written, {error}"))?;

        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(VERSION.len() as u32).to_le_bytes());
        out.extend_from_slice(VERSION.as_bytes());
        out.extend_from_slice(&self.kernel.program_id.to_le_bytes());
        section(&mut out, &state);
        let phys = &self.memory.phys;
        for memory in [&phys.fcram, &phys.vram, &phys.dsp_ram, &phys.axi_wram] {
            section(&mut out, memory);
        }
        Ok(out)
    }

    /// puts the console back as a saved state had it. nothing changes when
    /// the state is not one this game and this Zakuro can take.
    pub fn load_state(&mut self, bytes: &[u8]) -> Result<(), String> {
        self.idle_interrupted(true);
        let mut reader = Reader(bytes);
        if reader.take(MAGIC.len())? != MAGIC {
            return Err("it is not a saved state".into());
        }
        let length = reader.u32()? as usize;
        let version = String::from_utf8_lossy(reader.take(length)?).into_owned();
        if version != VERSION {
            return Err(format!("it was saved by Zakuro {version}, and this is {VERSION}"));
        }
        if reader.u64()? != self.kernel.program_id {
            return Err("it was saved from another game".into());
        }
        let state = reader.section()?;
        let saved: Saved = postcard::from_bytes(&state).map_err(|error| format!("the state could not be read, {error}"))?;
        let memories = [reader.section()?, reader.section()?, reader.section()?, reader.section()?];
        let phys = &self.memory.phys;
        let sizes = [phys.fcram.len(), phys.vram.len(), phys.dsp_ram.len(), phys.axi_wram.len()];
        if memories.iter().zip(sizes).any(|(memory, size)| memory.len() != size) {
            return Err("the console's memory is not the size the state has".into());
        }
        let (images, layers) = images(self);
        let mut files = std::collections::HashMap::new();
        for (handle, file) in saved.files {
            let file = file.open((&images, &layers)).ok_or("an open file of the state reads from no image of the game's")?;
            files.insert(handle, file);
        }

        // all of it read, the console takes it
        let phys = &mut self.memory.phys;
        for (memory, store) in memories.iter().zip([&mut phys.fcram, &mut phys.vram, &mut phys.dsp_ram, &mut phys.axi_wram]) {
            store.copy_from_slice(memory);
        }
        self.memory.load_map(saved.map)?;
        self.cpu = saved.cpu;
        self.kernel = saved.kernel;
        let user_dir = std::mem::take(&mut self.services.fs.user_dir);
        self.services = saved.services;
        self.services.fs.user_dir = user_dir;
        self.services.fs.files = files;
        // the computer's camera on while the state has a camera capturing
        crate::services::cam::capturing(self);
        // and the microphone while it has a title sampling
        crate::services::mic::sampling(self);
        // the recompiled code of the modules goes where the state has them
        if let Some(library) = &mut self.recompiled {
            for module in &self.cro.modules {
                library.place(&module.name, 0, &mut self.memory);
            }
        }
        self.cro = saved.cro;
        if let Some(library) = &mut self.recompiled {
            for module in &self.cro.modules {
                library.place(&module.name, module.base, &mut self.memory);
            }
        }
        self.gpu.load_state(&saved.gpu)?;
        let console = saved.console;
        self.exited = console.exited;
        self.broke = console.broke;
        self.debug_output = console.debug_output;
        self.fatal_errors = console.fatal_errors;
        self.lcd_force_black = console.lcd_force_black;
        self.frames = console.frames;
        self.boot_clock = console.boot_clock;
        self.next_frame_boundary = console.next_frame_boundary;
        self.next_audio_frame = console.next_audio_frame;
        self.next_preempt = console.next_preempt;
        self.history = console.history;
        self.history_index = console.history_index as usize % HISTORY_LENGTH;
        self.interpreted_instructions = console.interpreted_instructions;
        self.recompiled_instructions = console.recompiled_instructions;
        // an enhancement off now, that was on when the state was saved, puts
        // back what it overwrote at the next frame
        let mut overwrote: std::collections::HashMap<String, BTreeMap<u32, Overwrote>> = console.enhancements.into_iter().collect();
        for cheat in self.cheats.iter_mut().filter(|cheat| cheat.builtin) {
            cheat.overwrote = overwrote.remove(&cheat.name).unwrap_or_default();
        }
        // what was about to be shown, and the pace of the pictures in
        // between, start again
        self.showing = [None, None, None];
        self.pacing = Pacing::default();
        self.prefetch_aborts.clear();
        self.undefined_seen.clear();
        Ok(())
    }
}

/// a part of the file, how long it is in LZ4 and its bytes in LZ4.
fn section(out: &mut Vec<u8>, bytes: &[u8]) {
    let compressed = lz4_flex::compress_prepend_size(bytes);
    out.extend_from_slice(&(compressed.len() as u64).to_le_bytes());
    out.extend_from_slice(&compressed);
}

/// reads a state's file from the start on.
struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, length: usize) -> Result<&'a [u8], String> {
        if self.0.len() < length {
            return Err("the state ends early".into());
        }
        let (taken, rest) = self.0.split_at(length);
        self.0 = rest;
        Ok(taken)
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn section(&mut self) -> Result<Vec<u8>, String> {
        let length = self.u64()?;
        let compressed = self.take(usize::try_from(length).map_err(|_| "a part of the state is too long")?)?;
        lz4_flex::decompress_size_prepended(compressed).map_err(|error| format!("a part of the state is damaged, {error}"))
    }
}

#[cfg(test)]
mod tests {
    use crate::kernel::sync::ResetType;
    use crate::memory::{MemoryRegion, MemoryState, Permission};
    use crate::services::tests::system_with_thread;

    /// what changes after a state is saved goes back to how it was when it
    /// is loaded, registers, memory, the memory map and the kernel's objects.
    #[test]
    fn a_state_puts_everything_back() {
        let (mut system, _) = system_with_thread();
        let page = system.memory.phys.allocate(MemoryRegion::Application, 0x2000).unwrap();
        system.memory.map(0x0800_0000, page.addr, 0x2000, Permission::RW, MemoryState::Private);
        system.memory.write_bytes(0x0800_0000, b"saved");
        system.cpu.regs[3] = 7;
        let (event, handle) = system.kernel.create_event(ResetType::OneShot, "saved");
        system.frames = 600;

        let state = system.save_state().unwrap();
        system.memory.write_bytes(0x0800_0000, b"after");
        system.memory.unmap(0x0800_1000, 0x1000);
        system.cpu.regs[3] = 9;
        system.kernel.handles.close(&mut system.kernel.objects, handle);
        system.frames = 900;

        system.load_state(&state).unwrap();
        let mut bytes = [0u8; 5];
        system.memory.read_bytes(0x0800_0000, &mut bytes);
        assert_eq!(&bytes, b"saved");
        assert!(system.memory.mapping_at(0x0800_1000).is_some());
        assert_eq!(system.cpu.regs[3], 7);
        assert_eq!(system.kernel.resolve(handle), Some(event));
        assert_eq!(system.frames, 600);
    }

    /// an enhancement on when a state was saved and off when it is loaded,
    /// in another session, puts back what it overwrote.
    #[test]
    fn a_state_keeps_what_an_enhancement_overwrote() {
        let (mut system, _) = system_with_thread();
        let page = system.memory.phys.allocate(MemoryRegion::Application, 0x1000).unwrap();
        system.memory.map(0x0800_0000, page.addr, 0x1000, Permission::RW, MemoryState::Private);
        system.memory.write_bytes(0x0800_0000, &[1]);
        let mut fps = crate::cheats::Cheat::new("60 FPS", vec!["28000000 00000000".into()], Vec::new());
        fps.builtin = true;
        fps.enabled = true;
        system.cheats = vec![fps];
        crate::cheats::run(&mut system);
        let state = system.save_state().unwrap();

        // Zakuro started again, with the enhancement off
        system.cheats[0].enabled = false;
        system.cheats[0].overwrote.clear();
        system.load_state(&state).unwrap();
        let mut byte = [9];
        system.memory.read_bytes(0x0800_0000, &mut byte);
        assert_eq!(byte, [0], "the state has what the enhancement wrote");
        crate::cheats::run(&mut system);
        system.memory.read_bytes(0x0800_0000, &mut byte);
        assert_eq!(byte, [1]);
    }

    /// a state of another game, or not a state at all, is turned down and
    /// nothing changes.
    #[test]
    fn a_state_of_another_game_is_turned_down() {
        let (mut system, _) = system_with_thread();
        let state = system.save_state().unwrap();
        system.cpu.regs[0] = 5;
        system.kernel.program_id = 0x0004_0000_0000_1234;
        assert!(system.load_state(&state).is_err());
        assert!(system.load_state(b"nothing").is_err());
        assert_eq!(system.cpu.regs[0], 5);
    }
}
