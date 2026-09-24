//! CRO/CRS dynamic modules, loading, rebasing and linking.

pub mod format;

use std::collections::HashMap;

use zakuro_common::VAddr;
use zakuro_cpu::Bus;

use crate::memory::Memory;
use format::*;

/// a module the loader has linked in.
#[derive(Debug, Clone)]
pub struct Module {
    pub name: String,
    pub base: VAddr,
    pub size: u32,
    /// exported symbol name to address.
    pub exports: HashMap<String, VAddr>,
    /// exported symbols addressed by index rather than by name.
    pub indexed_exports: Vec<VAddr>,
    /// addresses of the module's segments, by segment index.
    pub segments: Vec<VAddr>,
    pub auto_link: bool,
}

#[derive(Debug, Default)]
pub struct CroManager {
    /// the static module, which holds the symbols the main executable exports.
    pub crs: Option<VAddr>,
    /// loaded modules, oldest first. The CRS is not in here.
    pub modules: Vec<Module>,
    /// imports we could not resolve, for diagnostics.
    pub unresolved: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CroError {
    NotACro,
    Truncated,
    BadSegment,
}

type Result<T> = std::result::Result<T, CroError>;

// ---------------------------------------------------------------------------
// Raw field access
// ---------------------------------------------------------------------------

/// reads one header field of the module at base.
fn field(memory: &mut Memory, base: VAddr, offset: u32) -> u32 {
    memory.read32(base + offset)
}

fn set_field(memory: &mut Memory, base: VAddr, offset: u32, value: u32) {
    memory.write32(base + offset, value);
}

fn segment(memory: &mut Memory, base: VAddr, index: u32) -> Option<Segment> {
    if index >= field(memory, base, header::SEGMENT_NUM) {
        return None;
    }
    let table = field(memory, base, header::SEGMENT_TABLE_OFFSET);
    let entry = table + index * SEGMENT_ENTRY_SIZE;
    Some(Segment {
        offset: memory.read32(entry),
        size: memory.read32(entry + 4),
        kind: SegmentType::from_raw(memory.read32(entry + 8)),
    })
}

fn set_segment(memory: &mut Memory, base: VAddr, index: u32, value: &Segment) {
    let table = field(memory, base, header::SEGMENT_TABLE_OFFSET);
    let entry = table + index * SEGMENT_ENTRY_SIZE;
    memory.write32(entry, value.offset);
    memory.write32(entry + 4, value.size);
    memory.write32(entry + 8, value.kind as u32);
}

/// resolves a segment tag to an address inside the module.
fn tag_to_address(memory: &mut Memory, base: VAddr, tag: SegmentTag) -> Option<VAddr> {
    let entry = segment(memory, base, tag.segment())?;
    if tag.offset() >= entry.size {
        return None;
    }
    Some(entry.offset + tag.offset())
}

fn read_string(memory: &mut Memory, address: VAddr) -> String {
    if address == 0 {
        return String::new();
    }
    memory.read_cstring(address, 512)
}

// ---------------------------------------------------------------------------
// Relocation
// ---------------------------------------------------------------------------

/// applies one relocation.
fn apply_relocation(
    memory: &mut Memory,
    target: VAddr,
    kind: RelocationType,
    addend: u32,
    symbol: VAddr,
    target_final: VAddr,
) {
    let write = |memory: &mut Memory, value: u32| {
        if !memory.write32_privileged(target, value) {
            log::warn!("CRO relocation targets unmapped 0x{target:08X}");
        }
    };
    match kind {
        RelocationType::Nothing => {}
        RelocationType::AbsoluteAddress => {
            write(memory, symbol.wrapping_add(addend));
        }
        RelocationType::RelativeAddress => {
            write(memory, symbol.wrapping_add(addend).wrapping_sub(target_final));
        }
        // no title has been seen to use these, a module that did would end up
        // calling the wrong address, so say so rather than writing something
        // plausible.
        other => log::warn!("CRO relocation type {other:?} is not implemented"),
    }
}

/// walks a batch of external relocations, applying each with symbol, and
/// records whether the batch is now resolved.
fn apply_relocation_batch(
    memory: &mut Memory,
    base: VAddr,
    batch: VAddr,
    symbol: VAddr,
    resolved: bool,
) {
    let mut address = batch;
    loop {
        let tag = SegmentTag(memory.read32(address));
        let kind = RelocationType::from_raw(memory.read8(address + 4));
        let is_end = memory.read8(address + 5) != 0;
        let addend = memory.read32(address + 8);

        if let Some(target) = tag_to_address(memory, base, tag) {
            apply_relocation(memory, target, kind, addend, symbol, target);
        }

        if is_end {
            break;
        }
        address += RELOCATION_ENTRY_SIZE;
        // a malformed table must not loop forever.
        if address > batch + RELOCATION_ENTRY_SIZE * 0x1000 {
            log::warn!("CRO relocation batch at 0x{batch:08X} has no end marker");
            break;
        }
    }
    // the third byte of the first entry records whether the batch is resolved.
    memory.write8(batch + 6, resolved as u8);
}

fn batch_is_resolved(memory: &mut Memory, batch: VAddr) -> bool {
    memory.read8(batch + 6) != 0
}

// ---------------------------------------------------------------------------
// Rebasing
// ---------------------------------------------------------------------------

impl CroManager {
    /// prepares a module that the guest has just read into memory, turns every
    /// stored offset into an address and applies its internal relocations.
    #[allow(clippy::too_many_arguments)]
    fn rebase(
        &mut self,
        memory: &mut Memory,
        base: VAddr,
        size: u32,
        data_segment: VAddr,
        data_segment_size: u32,
        bss_segment: VAddr,
        bss_segment_size: u32,
        is_static: bool,
    ) -> Result<()> {
        let mut magic = [0u8; 4];
        memory.read_bytes(base + header::MAGIC, &mut magic);
        if &magic != b"CRO0" && &magic != b"CRS0" {
            return Err(CroError::NotACro);
        }

        // every (offset, size) pair from the code segment onwards holds a file
        // offset that becomes an address.
        let name_offset = field(memory, base, header::NAME_OFFSET);
        if name_offset != 0 {
            set_field(memory, base, header::NAME_OFFSET, name_offset + base);
        }
        let mut offset_field = header::FIRST_REBASED;
        while offset_field <= header::LAST_REBASED {
            let value = field(memory, base, offset_field);
            if value != 0 {
                set_field(memory, base, offset_field, value + base);
            }
            offset_field += 8;
        }

        // segments, code and read-only data stay in the module's buffer, while
        // data and BSS are given their own.
        let segment_num = field(memory, base, header::SEGMENT_NUM);
        if log::log_enabled!(log::Level::Debug) {
            let table = field(memory, base, header::SEGMENT_TABLE_OFFSET);
            log::debug!(
                "CRO 0x{base:08X}: file size 0x{:X}, {segment_num} segments at 0x{table:08X}, \
                 code 0x{:08X}+0x{:X}, data 0x{:08X}+0x{:X}",
                field(memory, base, header::FILE_SIZE),
                field(memory, base, header::CODE_OFFSET),
                field(memory, base, header::CODE_SIZE),
                field(memory, base, header::DATA_OFFSET),
                field(memory, base, header::DATA_SIZE),
            );
            for i in 0..segment_num.min(8) {
                if let Some(entry) = segment(memory, base, i) {
                    log::debug!(
                        "  segment {i}: {:?} at 0x{:08X}, 0x{:X} bytes",
                        entry.kind, entry.offset, entry.size
                    );
                }
            }
        }
        let mut previous_data_segment = 0u32;
        for index in 0..segment_num {
            let Some(mut entry) = segment(memory, base, index) else {
                continue;
            };
            // the static module's segment table does not describe itself, it
            // describes the main executable, so its offsets are already
            // absolute addresses.
            if is_static {
                continue;
            }

            match entry.kind {
                SegmentType::Data if entry.size != 0 => {
                    if entry.size > data_segment_size {
                        log::error!(
                            "CRO at 0x{base:08X}: data segment is 0x{:X} bytes but only \
                             0x{data_segment_size:X} were provided",
                            entry.size
                        );
                        return Err(CroError::BadSegment);
                    }
                    previous_data_segment = entry.offset + base;
                    entry.offset = data_segment;
                }
                SegmentType::Bss if entry.size != 0 => {
                    if entry.size > bss_segment_size {
                        log::error!(
                            "CRO at 0x{base:08X}: BSS is 0x{:X} bytes but only \
                             0x{bss_segment_size:X} were provided",
                            entry.size
                        );
                        return Err(CroError::BadSegment);
                    }
                    entry.offset = bss_segment;
                }
                _ if entry.offset != 0 => {
                    entry.offset += base;
                    if entry.offset > base + size {
                        log::error!(
                            "CRO at 0x{base:08X}: segment {index} ({:?}) starts at \
                             0x{:08X}, past the end of the 0x{size:X}-byte module",
                            entry.kind,
                            entry.offset
                        );
                        return Err(CroError::BadSegment);
                    }
                }
                _ => {}
            }
            set_segment(memory, base, index, &entry);
        }

        rebase_table(
            memory,
            base,
            header::EXPORT_NAMED_SYMBOL_TABLE_OFFSET,
            header::EXPORT_NAMED_SYMBOL_NUM,
            EXPORT_NAMED_SYMBOL_ENTRY_SIZE,
            &[0],
        );
        rebase_table(
            memory,
            base,
            header::IMPORT_MODULE_TABLE_OFFSET,
            header::IMPORT_MODULE_NUM,
            IMPORT_MODULE_ENTRY_SIZE,
            &[0, 4, 12],
        );
        rebase_table(
            memory,
            base,
            header::IMPORT_NAMED_SYMBOL_TABLE_OFFSET,
            header::IMPORT_NAMED_SYMBOL_NUM,
            IMPORT_NAMED_SYMBOL_ENTRY_SIZE,
            &[0, 4],
        );
        rebase_table(
            memory,
            base,
            header::IMPORT_INDEXED_SYMBOL_TABLE_OFFSET,
            header::IMPORT_INDEXED_SYMBOL_NUM,
            IMPORT_INDEXED_SYMBOL_ENTRY_SIZE,
            &[4],
        );
        rebase_table(
            memory,
            base,
            header::IMPORT_ANONYMOUS_SYMBOL_TABLE_OFFSET,
            header::IMPORT_ANONYMOUS_SYMBOL_NUM,
            IMPORT_ANONYMOUS_SYMBOL_ENTRY_SIZE,
            &[4],
        );

        self.apply_internal_relocations(memory, base, previous_data_segment);
        Ok(())
    }

    /// relocations that point within the module itself.
    fn apply_internal_relocations(
        &self,
        memory: &mut Memory,
        base: VAddr,
        previous_data_segment: VAddr,
    ) {
        let table = field(memory, base, header::INTERNAL_RELOCATION_TABLE_OFFSET);
        let count = field(memory, base, header::INTERNAL_RELOCATION_NUM);

        for index in 0..count {
            let entry = table + index * RELOCATION_ENTRY_SIZE;
            let tag = SegmentTag(memory.read32(entry));
            let kind = RelocationType::from_raw(memory.read8(entry + 4));
            let symbol_segment = memory.read8(entry + 5) as u32;
            let addend = memory.read32(entry + 8);

            let Some(target_segment) = segment(memory, base, tag.segment()) else {
                continue;
            };
            if tag.offset() >= target_segment.size {
                continue;
            }
            let target_final = target_segment.offset + tag.offset();
            // a relocation into the data segment has to be written to the copy
            // still sitting in the module's buffer, because the guest has not
            // moved it yet.
            let target = if target_segment.kind == SegmentType::Data {
                previous_data_segment + tag.offset()
            } else {
                target_final
            };

            let Some(symbol) = segment(memory, base, symbol_segment) else {
                continue;
            };
            apply_relocation(memory, target, kind, addend, symbol.offset, target_final);
        }
    }

    /// marks every external relocation batch unresolved, which is the state a
    /// freshly loaded module starts in.
    fn reset_external_relocations(&self, memory: &mut Memory, base: VAddr) {
        let table = field(memory, base, header::EXTERNAL_RELOCATION_TABLE_OFFSET);
        let count = field(memory, base, header::EXTERNAL_RELOCATION_NUM);
        for index in 0..count {
            memory.write8(table + index * RELOCATION_ENTRY_SIZE + 6, 0);
        }
    }

    /// builds the symbol index for a module that has just been rebased.
    fn index_exports(&self, memory: &mut Memory, base: VAddr) -> (HashMap<String, VAddr>, Vec<VAddr>) {
        let mut named = HashMap::new();
        let table = field(memory, base, header::EXPORT_NAMED_SYMBOL_TABLE_OFFSET);
        let count = field(memory, base, header::EXPORT_NAMED_SYMBOL_NUM);
        for index in 0..count {
            let entry = table + index * EXPORT_NAMED_SYMBOL_ENTRY_SIZE;
            let name_address = memory.read32(entry);
            let name = read_string(memory, name_address);
            let tag = SegmentTag(memory.read32(entry + 4));
            match tag_to_address(memory, base, tag) {
                Some(address) => {
                    log::trace!("CRO export '{name}' -> 0x{address:08X}");
                    named.insert(name, address);
                }
                None => log::debug!(
                    "CRO export '{name}' has an unresolvable segment tag 0x{:08X}",
                    tag.0
                ),
            }
        }

        let mut indexed = Vec::new();
        let table = field(memory, base, header::EXPORT_INDEXED_SYMBOL_TABLE_OFFSET);
        let count = field(memory, base, header::EXPORT_INDEXED_SYMBOL_NUM);
        for index in 0..count {
            let entry = table + index * EXPORT_INDEXED_SYMBOL_ENTRY_SIZE;
            let tag = SegmentTag(memory.read32(entry));
            indexed.push(tag_to_address(memory, base, tag).unwrap_or(0));
        }

        (named, indexed)
    }

    fn read_segments(&self, memory: &mut Memory, base: VAddr) -> Vec<VAddr> {
        let count = field(memory, base, header::SEGMENT_NUM);
        (0..count)
            .map(|i| segment(memory, base, i).map_or(0, |s| s.offset))
            .collect()
    }

    // -- linking ------------------------------------------------------------

    /// looks a symbol up in every module that has been loaded, newest first,
    /// then in the static module.
    fn lookup(&self, name: &str) -> Option<VAddr> {
        self.modules
            .iter()
            .rev()
            .find_map(|module| module.exports.get(name).copied())
    }

    fn module_by_name(&self, name: &str) -> Option<&Module> {
        self.modules.iter().find(|m| m.name == name)
    }

    /// resolves the named symbols a module imports.
    fn apply_import_named_symbols(&mut self, memory: &mut Memory, base: VAddr) {
        let table = field(memory, base, header::IMPORT_NAMED_SYMBOL_TABLE_OFFSET);
        let count = field(memory, base, header::IMPORT_NAMED_SYMBOL_NUM);

        for index in 0..count {
            let entry = table + index * IMPORT_NAMED_SYMBOL_ENTRY_SIZE;
            let batch = memory.read32(entry + 4);
            if batch == 0 || batch_is_resolved(memory, batch) {
                continue;
            }
            let name_address = memory.read32(entry);
            let name = read_string(memory, name_address);
            match self.lookup(&name) {
                Some(address) => apply_relocation_batch(memory, base, batch, address, true),
                None => {
                    if self.unresolved.len() < 64 {
                        self.unresolved.push(name);
                    }
                }
            }
        }
    }

    /// resolves the indexed and anonymous symbols a module imports from the
    /// specific modules it names.
    fn apply_module_imports(&mut self, memory: &mut Memory, base: VAddr) {
        let table = field(memory, base, header::IMPORT_MODULE_TABLE_OFFSET);
        let count = field(memory, base, header::IMPORT_MODULE_NUM);

        for index in 0..count {
            let entry = table + index * IMPORT_MODULE_ENTRY_SIZE;
            let name_address = memory.read32(entry);
            let name = read_string(memory, name_address);
            let indexed_table = memory.read32(entry + 4);
            let indexed_count = memory.read32(entry + 8);
            let anonymous_table = memory.read32(entry + 12);
            let anonymous_count = memory.read32(entry + 16);

            let Some(source) = self.module_by_name(&name).cloned() else {
                continue;
            };

            for i in 0..indexed_count {
                let symbol_entry = indexed_table + i * IMPORT_INDEXED_SYMBOL_ENTRY_SIZE;
                let symbol_index = memory.read32(symbol_entry) as usize;
                let batch = memory.read32(symbol_entry + 4);
                if batch == 0 || batch_is_resolved(memory, batch) {
                    continue;
                }
                let Some(&address) = source.indexed_exports.get(symbol_index) else {
                    continue;
                };
                apply_relocation_batch(memory, base, batch, address, true);
            }

            for i in 0..anonymous_count {
                let symbol_entry = anonymous_table + i * IMPORT_ANONYMOUS_SYMBOL_ENTRY_SIZE;
                let tag = SegmentTag(memory.read32(symbol_entry));
                let batch = memory.read32(symbol_entry + 4);
                if batch == 0 || batch_is_resolved(memory, batch) {
                    continue;
                }
                let Some(&segment_address) = source.segments.get(tag.segment() as usize) else {
                    continue;
                };
                apply_relocation_batch(
                    memory,
                    base,
                    batch,
                    segment_address + tag.offset(),
                    true,
                );
            }
        }
    }

    /// resolves imports in modules already loaded that this module satisfies.
    fn apply_exports_to_loaded(&mut self, memory: &mut Memory, new_index: usize) {
        let new_module = self.modules[new_index].clone();

        for (index, module) in self.modules.clone().iter().enumerate() {
            if index == new_index {
                continue;
            }
            let base = module.base;
            let table = field(memory, base, header::IMPORT_NAMED_SYMBOL_TABLE_OFFSET);
            let count = field(memory, base, header::IMPORT_NAMED_SYMBOL_NUM);
            for i in 0..count {
                let entry = table + i * IMPORT_NAMED_SYMBOL_ENTRY_SIZE;
                let batch = memory.read32(entry + 4);
                if batch == 0 || batch_is_resolved(memory, batch) {
                    continue;
                }
                let name_address = memory.read32(entry);
            let name = read_string(memory, name_address);
                if let Some(&address) = new_module.exports.get(&name) {
                    apply_relocation_batch(memory, base, batch, address, true);
                }
            }

            // and the indexed/anonymous symbols they import from us by name.
            let table = field(memory, base, header::IMPORT_MODULE_TABLE_OFFSET);
            let count = field(memory, base, header::IMPORT_MODULE_NUM);
            for i in 0..count {
                let entry = table + i * IMPORT_MODULE_ENTRY_SIZE;
                let wanted_address = memory.read32(entry);
                let wanted = read_string(memory, wanted_address);
                if wanted != new_module.name {
                    continue;
                }
                let indexed_table = memory.read32(entry + 4);
                let indexed_count = memory.read32(entry + 8);
                let anonymous_table = memory.read32(entry + 12);
                let anonymous_count = memory.read32(entry + 16);

                for j in 0..indexed_count {
                    let symbol_entry = indexed_table + j * IMPORT_INDEXED_SYMBOL_ENTRY_SIZE;
                    let symbol_index = memory.read32(symbol_entry) as usize;
                    let batch = memory.read32(symbol_entry + 4);
                    if batch == 0 || batch_is_resolved(memory, batch) {
                        continue;
                    }
                    if let Some(&address) = new_module.indexed_exports.get(symbol_index) {
                        apply_relocation_batch(memory, base, batch, address, true);
                    }
                }

                for j in 0..anonymous_count {
                    let symbol_entry = anonymous_table + j * IMPORT_ANONYMOUS_SYMBOL_ENTRY_SIZE;
                    let tag = SegmentTag(memory.read32(symbol_entry));
                    let batch = memory.read32(symbol_entry + 4);
                    if batch == 0 || batch_is_resolved(memory, batch) {
                        continue;
                    }
                    if let Some(&segment_address) =
                        new_module.segments.get(tag.segment() as usize)
                    {
                        apply_relocation_batch(
                            memory,
                            base,
                            batch,
                            segment_address + tag.offset(),
                            true,
                        );
                    }
                }
            }
        }
    }

    // -- public API ---------------------------------------------------------

    /// ldr:ro Initialize, registers the static module, whose exports every
    /// other module links against.
    pub fn initialize(&mut self, memory: &mut Memory, base: VAddr, size: u32) -> Result<()> {
        self.rebase(memory, base, size, 0, 0, 0, 0, true)?;
        self.reset_external_relocations(memory, base);

        let (exports, indexed_exports) = self.index_exports(memory, base);
        let segments = self.read_segments(memory, base);
        let name_offset = field(memory, base, header::MODULE_NAME_OFFSET);
        let name = read_string(memory, name_offset);

        log::info!(
            "ldr:ro: static module '{name}' at 0x{base:08X} exports {} named symbols",
            exports.len()
        );

        self.crs = Some(base);
        self.modules.push(Module {
            name,
            base,
            size,
            exports,
            indexed_exports,
            segments,
            auto_link: true,
        });
        Ok(())
    }

    /// ldr:ro LoadCRO, rebases, relocates and links a module.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        &mut self,
        memory: &mut Memory,
        base: VAddr,
        size: u32,
        data_segment: VAddr,
        data_segment_size: u32,
        bss_segment: VAddr,
        bss_segment_size: u32,
        auto_link: bool,
    ) -> Result<u32> {
        self.rebase(
            memory,
            base,
            size,
            data_segment,
            data_segment_size,
            bss_segment,
            bss_segment_size,
            false,
        )?;
        self.reset_external_relocations(memory, base);

        let (exports, indexed_exports) = self.index_exports(memory, base);
        let segments = self.read_segments(memory, base);
        let name_offset = field(memory, base, header::MODULE_NAME_OFFSET);
        let name = read_string(memory, name_offset);

        self.modules.push(Module {
            name: name.clone(),
            base,
            size,
            exports,
            indexed_exports,
            segments,
            auto_link,
        });
        let index = self.modules.len() - 1;

        // link it against what is already there, then let what is already
        // there link against it.
        self.apply_import_named_symbols(memory, base);
        self.apply_module_imports(memory, base);
        self.apply_exports_to_loaded(memory, index);

        log::debug!(
            "ldr:ro: loaded '{name}' at 0x{base:08X} ({} exports, {} modules linked)",
            self.modules[index].exports.len(),
            self.modules.len()
        );

        // the guest reads the fixed size back to learn how much of its buffer
        // it may reuse.
        Ok(field(memory, base, header::FIX_SIZE))
    }

    /// ldr:ro UnloadCRO.
    pub fn unload(&mut self, memory: &mut Memory, base: VAddr) {
        let Some(index) = self.modules.iter().position(|m| m.base == base) else {
            return;
        };
        let module = self.modules.remove(index);
        log::debug!("ldr:ro: unloaded '{}'", module.name);

        // anything that linked against it has to go back to unresolved, or it
        // would keep calling into memory the guest is about to reuse.
        for other in self.modules.clone() {
            let other_base = other.base;
            let table = field(memory, other_base, header::IMPORT_NAMED_SYMBOL_TABLE_OFFSET);
            let count = field(memory, other_base, header::IMPORT_NAMED_SYMBOL_NUM);
            for i in 0..count {
                let entry = table + i * IMPORT_NAMED_SYMBOL_ENTRY_SIZE;
                let batch = memory.read32(entry + 4);
                if batch == 0 || !batch_is_resolved(memory, batch) {
                    continue;
                }
                let name_address = memory.read32(entry);
            let name = read_string(memory, name_address);
                if module.exports.contains_key(&name) {
                    apply_relocation_batch(memory, other_base, batch, 0, false);
                }
            }
        }
    }

    /// number of modules currently linked, for the diagnostics overlay.
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }
}

/// adds the module's base to a set of offset fields in every entry of a table.
fn rebase_table(
    memory: &mut Memory,
    base: VAddr,
    table_field: u32,
    count_field: u32,
    entry_size: u32,
    offsets: &[u32],
) {
    let table = field(memory, base, table_field);
    let count = field(memory, base, count_field);
    if table == 0 {
        return;
    }
    for index in 0..count {
        let entry = table + index * entry_size;
        for &offset in offsets {
            let value = memory.read32(entry + offset);
            if value != 0 {
                memory.write32(entry + offset, value + base);
            }
        }
    }
}
