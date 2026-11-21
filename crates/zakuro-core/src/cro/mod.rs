//! CRO and CRS dynamic modules, loading, rebasing and linking

pub mod format;

use std::collections::HashMap;

use zakuro_common::VAddr;
use zakuro_cpu::Bus;

use crate::memory::Memory;
use format::*;

/// a module the loader linked in
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Module {
    pub name: String,
    pub base: VAddr,
    pub size: u32,
    /// exported symbol name to addr
    pub exports: HashMap<String, VAddr>,
    /// exported symbols by index instead of name
    pub indexed_exports: Vec<VAddr>,
    /// addrs of the module's segments, by index
    pub segments: Vec<VAddr>,
    pub auto_link: bool,
}

#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct CroManager {
    /// the static module, holds what the main exe exports
    pub crs: Option<VAddr>,
    /// loaded modules oldest first, the CRS isn't in here
    pub modules: Vec<Module>,
    /// imports we couldn't resolve, for diagnostics
    pub unresolved: Vec<String>,
    /// modules ldr:ro gave their own memory, by where they run
    /// the buffer the title loaded them from, the memory, and how much fixing left
    pub copies: HashMap<VAddr, Copy>,
}

/// a module running from its own memory instead of the title's buffer
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Copy {
    pub buffer: VAddr,
    pub block: crate::memory::physical::PhysicalBlock,
    pub fixed: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CroError {
    NotACro,
    Truncated,
    BadSegment,
}

type Result<T> = std::result::Result<T, CroError>;

// raw field access

/// reads one header field of the module at base
fn field(memory: &mut Memory, base: VAddr, offset: u32) -> u32 {
    memory.read32(base + offset)
}

fn set_field(memory: &mut Memory, base: VAddr, offset: u32, value: u32) {
    memory.write32(base + offset, value);
}

// the lists the title walks

/// adds a module to the end of its list, the title walks it from the static module to find exports
/// the static module's next field heads the self linking modules, its previous field the others
/// a list's head keeps its tail in its previous field, the tail has no next
fn register(memory: &mut Memory, crs: VAddr, base: VAddr, auto_link: bool) {
    let head_field = if auto_link { header::NEXT_CRO } else { header::PREVIOUS_CRO };
    let head = field(memory, crs, head_field);
    if head == 0 {
        set_field(memory, crs, head_field, base);
        set_field(memory, base, header::PREVIOUS_CRO, base);
    } else {
        let tail = field(memory, head, header::PREVIOUS_CRO);
        set_field(memory, tail, header::NEXT_CRO, base);
        set_field(memory, base, header::PREVIOUS_CRO, tail);
        set_field(memory, head, header::PREVIOUS_CRO, base);
    }
    set_field(memory, base, header::NEXT_CRO, 0);
}

/// takes a module out of its list
fn unregister(memory: &mut Memory, crs: VAddr, base: VAddr) {
    let next = field(memory, base, header::NEXT_CRO);
    let previous = field(memory, base, header::PREVIOUS_CRO);
    // one in a list has a previous (itself when alone), one loaded b4 lists were kept (older saved state) is in none
    if previous == 0 {
        return;
    }
    let heads = [header::NEXT_CRO, header::PREVIOUS_CRO];
    if let Some(head_field) = heads.into_iter().find(|&head_field| field(memory, crs, head_field) == base) {
        // the head, the one after it heads the list now and keeps the tail
        if next != 0 {
            set_field(memory, next, header::PREVIOUS_CRO, previous);
        }
        set_field(memory, crs, head_field, next);
    } else if next != 0 {
        set_field(memory, previous, header::NEXT_CRO, next);
        set_field(memory, next, header::PREVIOUS_CRO, previous);
    } else {
        // the tail, the one b4 it is the tail now
        set_field(memory, previous, header::NEXT_CRO, 0);
        for head_field in heads {
            let head = field(memory, crs, head_field);
            if head != 0 && field(memory, head, header::PREVIOUS_CRO) == base {
                set_field(memory, head, header::PREVIOUS_CRO, previous);
            }
        }
    }
    set_field(memory, base, header::NEXT_CRO, 0);
    set_field(memory, base, header::PREVIOUS_CRO, 0);
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

/// segment tag to an addr inside the module
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

// relocation

/// applies one reloc
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
        // no title's been seen using these, a module that did would call the wrong addr, so complain instead of writing smth plausible
        other => log::warn!("CRO relocation type {other:?} is not implemented"),
    }
}

/// walks a batch of external relocs, applying each w/ symbol, and records if the batch is resolved now
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
        // a malformed table mustn't loop forever
        if address > batch + RELOCATION_ENTRY_SIZE * 0x1000 {
            log::warn!("CRO relocation batch at 0x{batch:08X} has no end marker");
            break;
        }
    }
    // the first entry's 3rd byte says if the batch is resolved
    memory.write8(batch + 6, resolved as u8);
}

fn batch_is_resolved(memory: &mut Memory, batch: VAddr) -> bool {
    memory.read8(batch + 6) != 0
}

/// the module's own handler for imports nobody provides, they point there till smth does and again once it leaves
fn unresolved_symbol(memory: &mut Memory, base: VAddr) -> VAddr {
    let tag = SegmentTag(field(memory, base, header::ON_UNRESOLVED_SEGMENT_TAG));
    tag_to_address(memory, base, tag).unwrap_or(0)
}

/// entry size of each table the header lists from the module name on, 1 for the byte counted ones
const TABLE_ENTRY_SIZES: [u32; 15] = [
    1,
    SEGMENT_ENTRY_SIZE,
    EXPORT_NAMED_SYMBOL_ENTRY_SIZE,
    EXPORT_INDEXED_SYMBOL_ENTRY_SIZE,
    1,
    EXPORT_TREE_ENTRY_SIZE,
    IMPORT_MODULE_ENTRY_SIZE,
    RELOCATION_ENTRY_SIZE,
    IMPORT_NAMED_SYMBOL_ENTRY_SIZE,
    IMPORT_INDEXED_SYMBOL_ENTRY_SIZE,
    IMPORT_ANONYMOUS_SYMBOL_ENTRY_SIZE,
    1,
    8,
    RELOCATION_ENTRY_SIZE,
    RELOCATION_ENTRY_SIZE,
];

/// first table each fix level drops, everything after it goes too
/// level 1 drops what only loading needed, 2 the imports too, 3 the exports
const FIX_BARRIERS: [u32; 4] = [
    header::SIZE,
    header::STATIC_ANONYMOUS_SYMBOL_TABLE_OFFSET,
    header::IMPORT_MODULE_TABLE_OFFSET,
    header::EXPORT_NAMED_SYMBOL_TABLE_OFFSET,
];

// rebasing

impl CroManager {
    /// preps a module the guest just read into memory, turns every stored offset into an addr and applies its internal relocs
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

        // every (offset, size) pair from the code segment on holds a file offset that becomes an addr
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

        // code and rodata segments stay in the module's buffer, data and BSS get their own
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
            // the static module's segment table describes the main exe, not itself, so its offsets are already absolute
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

    /// relocs pointing inside the module itself
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
            // a reloc into the data segment goes to the copy still in the module's buffer, the guest hasn't moved it yet
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

    /// points every import at the module's unresolved handler and marks it unresolved, how a fresh module starts
    fn reset_external_relocations(&self, memory: &mut Memory, base: VAddr) {
        let unresolved = unresolved_symbol(memory, base);
        let table = field(memory, base, header::EXTERNAL_RELOCATION_TABLE_OFFSET);
        let count = field(memory, base, header::EXTERNAL_RELOCATION_NUM);
        for index in 0..count {
            let entry = table + index * RELOCATION_ENTRY_SIZE;
            if unresolved != 0 {
                let tag = SegmentTag(memory.read32(entry));
                let kind = RelocationType::from_raw(memory.read8(entry + 4));
                let addend = memory.read32(entry + 8);
                if let Some(target) = tag_to_address(memory, base, tag) {
                    apply_relocation(memory, target, kind, addend, unresolved, target);
                }
            }
            memory.write8(entry + 6, 0);
        }
    }

    /// drops the tables the fix level says aren't needed so the title can reclaim that memory, returns how much is left
    fn fix(&self, memory: &mut Memory, base: VAddr, size: u32, level: u32) -> u32 {
        let Some(&barrier) = FIX_BARRIERS.get(level as usize).filter(|_| level != 0) else {
            return 0;
        };
        // what's kept ends w/ the code or the last table b4 the barrier
        let mut end = field(memory, base, header::CODE_OFFSET) + field(memory, base, header::CODE_SIZE);
        let mut offset = header::MODULE_NAME_OFFSET;
        for entry_size in TABLE_ENTRY_SIZES {
            let table = field(memory, base, offset);
            let count = field(memory, base, offset + 4);
            if offset < barrier {
                if count != 0 {
                    end = end.max(table + count * entry_size);
                }
            } else {
                set_field(memory, base, offset + 4, 0);
            }
            offset += 8;
        }
        let fixed = zakuro_common::bits::align_up(end.saturating_sub(base), 0x1000).min(size);
        // dropped tables point where the module now ends, empty
        let mut offset = barrier;
        while offset <= header::STATIC_RELOCATION_TABLE_OFFSET {
            set_field(memory, base, offset, base + fixed);
            offset += 8;
        }
        set_field(memory, base, header::FIXED_SIZE, fixed);
        fixed
    }

    /// builds the symbol index for a freshly rebased module
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

    // linking

    /// looks a symbol up in the static module, then in auto linked modules in load order, like ldr:ro
    fn lookup(&self, name: &str) -> Option<VAddr> {
        self.modules
            .iter()
            .filter(|module| module.auto_link)
            .find_map(|module| module.exports.get(name).copied())
    }

    fn module_by_name(&self, name: &str) -> Option<&Module> {
        self.modules.iter().find(|m| m.name == name)
    }

    /// resolves the named symbols a module imports
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

    /// resolves the indexed and anonymous symbols a module imports from the modules it names
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

    /// resolves imports in already loaded modules that this one satisfies
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

            // plus the indexed and anonymous symbols they import from us by name
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

    // public api

    /// ldr:ro Initialize, registers the static module everyone links against
    pub fn initialize(&mut self, memory: &mut Memory, base: VAddr, size: u32) -> Result<()> {
        self.rebase(memory, base, size, 0, 0, 0, 0, true)?;
        self.reset_external_relocations(memory, base);

        let (exports, indexed_exports) = self.index_exports(memory, base);
        let segments = self.read_segments(memory, base);
        let name_offset = field(memory, base, header::MODULE_NAME_OFFSET);
        let name = read_string(memory, name_offset);
        // nothing hangs off it yet
        set_field(memory, base, header::NEXT_CRO, 0);
        set_field(memory, base, header::PREVIOUS_CRO, 0);

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

    /// ldr:ro LoadCRO, rebases, relocates and links a module
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
        fix_level: u32,
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
        if let Some(crs) = self.crs {
            register(memory, crs, base, auto_link);
        }

        // link it against what's there, then let what's there link against it
        self.apply_import_named_symbols(memory, base);
        self.apply_module_imports(memory, base);
        self.apply_exports_to_loaded(memory, index);

        log::debug!(
            "ldr:ro: loaded '{name}' at 0x{base:08X} ({} exports, {} modules linked)",
            self.modules[index].exports.len(),
            self.modules.len()
        );

        // the guest learns from the fixed size how much of its buffer it can take back
        Ok(self.fix(memory, base, size, fix_level))
    }

    /// ldr:ro UnloadCRO
    pub fn unload(&mut self, memory: &mut Memory, base: VAddr) {
        let Some(index) = self.modules.iter().position(|m| m.base == base) else {
            return;
        };
        let module = self.modules.remove(index);
        log::debug!("ldr:ro: unloaded '{}'", module.name);

        // anything linked against it goes back to its own unresolved handler,
        // else it'd keep calling into memory the guest is about to reuse
        // named imports go by name, indexed and anonymous ones by the module they name
        for other in self.modules.clone().iter().filter(|other| other.auto_link) {
            let other_base = other.base;
            let unresolved = unresolved_symbol(memory, other_base);
            let reset = |memory: &mut Memory, batch: VAddr| {
                if batch != 0 && batch_is_resolved(memory, batch) {
                    apply_relocation_batch(memory, other_base, batch, unresolved, false);
                }
            };

            let table = field(memory, other_base, header::IMPORT_NAMED_SYMBOL_TABLE_OFFSET);
            let count = field(memory, other_base, header::IMPORT_NAMED_SYMBOL_NUM);
            for i in 0..count {
                let entry = table + i * IMPORT_NAMED_SYMBOL_ENTRY_SIZE;
                let name_address = memory.read32(entry);
                let name = read_string(memory, name_address);
                if module.exports.contains_key(&name) {
                    let batch = memory.read32(entry + 4);
                    reset(memory, batch);
                }
            }

            let table = field(memory, other_base, header::IMPORT_MODULE_TABLE_OFFSET);
            let count = field(memory, other_base, header::IMPORT_MODULE_NUM);
            for i in 0..count {
                let entry = table + i * IMPORT_MODULE_ENTRY_SIZE;
                let name_address = memory.read32(entry);
                if read_string(memory, name_address) != module.name {
                    continue;
                }
                let tables = [
                    (memory.read32(entry + 4), memory.read32(entry + 8), IMPORT_INDEXED_SYMBOL_ENTRY_SIZE),
                    (memory.read32(entry + 12), memory.read32(entry + 16), IMPORT_ANONYMOUS_SYMBOL_ENTRY_SIZE),
                ];
                for (table, count, entry_size) in tables {
                    for j in 0..count {
                        let batch = memory.read32(table + j * entry_size + 4);
                        reset(memory, batch);
                    }
                }
            }
        }

        if let Some(crs) = self.crs {
            unregister(memory, crs, base);
        }
        unrebase(memory, base);
    }

    /// modules linked rn, for the diagnostics overlay
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }
}

/// adds the module's base to some offset fields in every entry of a table
fn rebase_table(
    memory: &mut Memory,
    base: VAddr,
    table_field: u32,
    count_field: u32,
    entry_size: u32,
    offsets: &[u32],
) {
    move_table(memory, base, table_field, count_field, entry_size, offsets, true);
}

/// turns a table's entry offsets into addrs, or back
fn move_table(
    memory: &mut Memory,
    base: VAddr,
    table_field: u32,
    count_field: u32,
    entry_size: u32,
    offsets: &[u32],
    forward: bool,
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
                let moved = if forward { value.wrapping_add(base) } else { value.wrapping_sub(base) };
                memory.write32(entry + offset, moved);
            }
        }
    }
}

/// tables whose entries point elsewhere in the module, and where in an entry the pointers are
const POINTER_TABLES: [(u32, u32, u32, &[u32]); 5] = [
    (header::EXPORT_NAMED_SYMBOL_TABLE_OFFSET, header::EXPORT_NAMED_SYMBOL_NUM, EXPORT_NAMED_SYMBOL_ENTRY_SIZE, &[0]),
    (header::IMPORT_MODULE_TABLE_OFFSET, header::IMPORT_MODULE_NUM, IMPORT_MODULE_ENTRY_SIZE, &[0, 4, 12]),
    (header::IMPORT_NAMED_SYMBOL_TABLE_OFFSET, header::IMPORT_NAMED_SYMBOL_NUM, IMPORT_NAMED_SYMBOL_ENTRY_SIZE, &[0, 4]),
    (header::IMPORT_INDEXED_SYMBOL_TABLE_OFFSET, header::IMPORT_INDEXED_SYMBOL_NUM, IMPORT_INDEXED_SYMBOL_ENTRY_SIZE, &[4]),
    (header::IMPORT_ANONYMOUS_SYMBOL_TABLE_OFFSET, header::IMPORT_ANONYMOUS_SYMBOL_NUM, IMPORT_ANONYMOUS_SYMBOL_ENTRY_SIZE, &[4]),
];

/// puts a module back the way its file has it (offsets not addrs), that's how ldr:ro hands the buffer back to the title
fn unrebase(memory: &mut Memory, base: VAddr) {
    for (table, count, entry_size, offsets) in POINTER_TABLES {
        move_table(memory, base, table, count, entry_size, offsets, false);
    }

    let segment_num = field(memory, base, header::SEGMENT_NUM);
    for index in 0..segment_num {
        let Some(mut entry) = segment(memory, base, index) else { continue };
        entry.offset = match entry.kind {
            SegmentType::Bss => 0,
            _ if entry.offset != 0 => entry.offset.wrapping_sub(base),
            _ => 0,
        };
        set_segment(memory, base, index, &entry);
    }

    let name_offset = field(memory, base, header::NAME_OFFSET);
    if name_offset != 0 {
        set_field(memory, base, header::NAME_OFFSET, name_offset.wrapping_sub(base));
    }
    let mut offset_field = header::FIRST_REBASED;
    while offset_field <= header::LAST_REBASED {
        let value = field(memory, base, offset_field);
        if value != 0 {
            set_field(memory, base, offset_field, value.wrapping_sub(base));
        }
        offset_field += 8;
    }
    set_field(memory, base, header::FIXED_SIZE, 0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{MemoryRegion, MemoryState, Permission};

    /// the static module's header plus four modules', in their own memory
    fn headers() -> (crate::System, VAddr, [VAddr; 4]) {
        let mut system = crate::System::new(crate::Config::default());
        let block = system.memory.phys.allocate(MemoryRegion::Application, 0x1000).unwrap();
        system.memory.map(0x0070_0000, block.addr, 0x1000, Permission::RW, MemoryState::Private);
        let at = |index: u32| 0x0070_0000 + index * 0x200;
        (system, at(0), [at(1), at(2), at(3), at(4)])
    }

    /// a header's next and previous fields
    fn links(memory: &mut Memory, base: VAddr) -> (VAddr, VAddr) {
        (field(memory, base, header::NEXT_CRO), field(memory, base, header::PREVIOUS_CRO))
    }

    /// modules join the end of their list and leave from anywhere, the title walking it from the static module finds them in order
    #[test]
    fn modules_go_in_and_out_of_the_lists_the_title_walks() {
        let (mut system, crs, [a, b, c, d]) = headers();
        let memory = &mut system.memory;
        register(memory, crs, a, true);
        register(memory, crs, b, true);
        register(memory, crs, c, true);
        register(memory, crs, d, false);
        assert_eq!(links(memory, crs), (a, d), "the heads of both lists");
        assert_eq!(links(memory, a), (b, c), "the head keeps the tail");
        assert_eq!(links(memory, b), (c, a));
        assert_eq!(links(memory, c), (0, b));
        assert_eq!(links(memory, d), (0, d), "alone in its list");

        unregister(memory, crs, b);
        assert_eq!((links(memory, a), links(memory, c), links(memory, b)), ((c, c), (0, a), (0, 0)), "from the middle");
        unregister(memory, crs, c);
        assert_eq!(links(memory, a), (0, a), "the tail");
        register(memory, crs, b, true);
        unregister(memory, crs, a);
        assert_eq!((links(memory, crs), links(memory, b)), ((b, d), (0, b)), "the head");
        unregister(memory, crs, b);
        unregister(memory, crs, d);
        assert_eq!(links(memory, crs), (0, 0), "the last ones");
        // one in no list (from b4 lists were kept) stays out
        register(memory, crs, a, true);
        unregister(memory, crs, c);
        assert_eq!((links(memory, crs), links(memory, a)), ((a, 0), (0, a)), "in none");
    }
}
