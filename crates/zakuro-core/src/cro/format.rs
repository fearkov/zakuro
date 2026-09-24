//! on-disk layout of a CRO/CRS module.

use zakuro_common::VAddr;

/// byte offsets of the header fields.
pub mod header {
    pub const MAGIC: u32 = 0x080;
    pub const NAME_OFFSET: u32 = 0x084;
    pub const NEXT_CRO: u32 = 0x088;
    pub const PREVIOUS_CRO: u32 = 0x08C;
    pub const FILE_SIZE: u32 = 0x090;
    pub const BSS_SIZE: u32 = 0x094;
    pub const FIX_SIZE: u32 = 0x0A4;

    /// the first field holding an offset that becomes an address on rebase.
    pub const FIRST_REBASED: u32 = 0x0B0;

    pub const CODE_OFFSET: u32 = 0x0B0;
    pub const CODE_SIZE: u32 = 0x0B4;
    pub const DATA_OFFSET: u32 = 0x0B8;
    pub const DATA_SIZE: u32 = 0x0BC;
    pub const MODULE_NAME_OFFSET: u32 = 0x0C0;
    pub const MODULE_NAME_SIZE: u32 = 0x0C4;
    pub const SEGMENT_TABLE_OFFSET: u32 = 0x0C8;
    pub const SEGMENT_NUM: u32 = 0x0CC;
    pub const EXPORT_NAMED_SYMBOL_TABLE_OFFSET: u32 = 0x0D0;
    pub const EXPORT_NAMED_SYMBOL_NUM: u32 = 0x0D4;
    pub const EXPORT_INDEXED_SYMBOL_TABLE_OFFSET: u32 = 0x0D8;
    pub const EXPORT_INDEXED_SYMBOL_NUM: u32 = 0x0DC;
    pub const EXPORT_STRINGS_OFFSET: u32 = 0x0E0;
    pub const EXPORT_STRINGS_SIZE: u32 = 0x0E4;
    pub const EXPORT_TREE_TABLE_OFFSET: u32 = 0x0E8;
    pub const EXPORT_TREE_NUM: u32 = 0x0EC;
    pub const IMPORT_MODULE_TABLE_OFFSET: u32 = 0x0F0;
    pub const IMPORT_MODULE_NUM: u32 = 0x0F4;
    pub const EXTERNAL_RELOCATION_TABLE_OFFSET: u32 = 0x0F8;
    pub const EXTERNAL_RELOCATION_NUM: u32 = 0x0FC;
    pub const IMPORT_NAMED_SYMBOL_TABLE_OFFSET: u32 = 0x100;
    pub const IMPORT_NAMED_SYMBOL_NUM: u32 = 0x104;
    pub const IMPORT_INDEXED_SYMBOL_TABLE_OFFSET: u32 = 0x108;
    pub const IMPORT_INDEXED_SYMBOL_NUM: u32 = 0x10C;
    pub const IMPORT_ANONYMOUS_SYMBOL_TABLE_OFFSET: u32 = 0x110;
    pub const IMPORT_ANONYMOUS_SYMBOL_NUM: u32 = 0x114;
    pub const IMPORT_STRINGS_OFFSET: u32 = 0x118;
    pub const IMPORT_STRINGS_SIZE: u32 = 0x11C;
    pub const STATIC_ANONYMOUS_SYMBOL_TABLE_OFFSET: u32 = 0x120;
    pub const STATIC_ANONYMOUS_SYMBOL_NUM: u32 = 0x124;
    pub const INTERNAL_RELOCATION_TABLE_OFFSET: u32 = 0x128;
    pub const INTERNAL_RELOCATION_NUM: u32 = 0x12C;
    pub const STATIC_RELOCATION_TABLE_OFFSET: u32 = 0x130;
    pub const STATIC_RELOCATION_NUM: u32 = 0x134;

    /// the last field holding an offset that becomes an address on rebase.
    pub const LAST_REBASED: u32 = STATIC_RELOCATION_TABLE_OFFSET;

    pub const SIZE: u32 = 0x138;
}

/// which part of a module a segment holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentType {
    Code,
    RoData,
    Data,
    Bss,
}

impl SegmentType {
    pub fn from_raw(value: u32) -> SegmentType {
        match value {
            0 => SegmentType::Code,
            1 => SegmentType::RoData,
            2 => SegmentType::Data,
            _ => SegmentType::Bss,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Segment {
    pub offset: u32,
    pub size: u32,
    pub kind: SegmentType,
}

pub const SEGMENT_ENTRY_SIZE: u32 = 0xC;
pub const EXPORT_NAMED_SYMBOL_ENTRY_SIZE: u32 = 8;
pub const EXPORT_INDEXED_SYMBOL_ENTRY_SIZE: u32 = 4;
pub const EXPORT_TREE_ENTRY_SIZE: u32 = 8;
pub const IMPORT_MODULE_ENTRY_SIZE: u32 = 0x14;
pub const IMPORT_NAMED_SYMBOL_ENTRY_SIZE: u32 = 8;
pub const IMPORT_INDEXED_SYMBOL_ENTRY_SIZE: u32 = 8;
pub const IMPORT_ANONYMOUS_SYMBOL_ENTRY_SIZE: u32 = 8;
pub const RELOCATION_ENTRY_SIZE: u32 = 0xC;

/// a position inside a module, a segment index in the low four bits and a byte
/// offset into that segment in the rest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentTag(pub u32);

impl SegmentTag {
    pub const fn segment(self) -> u32 {
        self.0 & 0xF
    }

    pub const fn offset(self) -> u32 {
        self.0 >> 4
    }
}

/// what a relocation does to its target word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelocationType {
    /// leaves the target alone, used to pad a batch.
    Nothing,
    /// *target = symbol + addend
    AbsoluteAddress,
    /// *target = symbol + addend - target
    RelativeAddress,
    ThumbBranch,
    ArmBranch,
    ModifyArmBranch,
    AlignedRelativeAddress,
    Unknown(u8),
}

impl RelocationType {
    /// the numbering is not contiguous and does not start at one, a module's
    /// jump tables are AbsoluteAddress, which is 2, so reading 2 as
    /// "relative" writes offsets where addresses belong and the module
    /// branches into the first page of memory.
    pub fn from_raw(value: u8) -> RelocationType {
        match value {
            0 => RelocationType::Nothing,
            2 | 10 => RelocationType::AbsoluteAddress,
            3 => RelocationType::RelativeAddress,
            4 => RelocationType::ThumbBranch,
            5 => RelocationType::ArmBranch,
            6 => RelocationType::ModifyArmBranch,
            42 => RelocationType::AlignedRelativeAddress,
            other => RelocationType::Unknown(other),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct InternalRelocation {
    pub target: SegmentTag,
    pub kind: RelocationType,
    pub symbol_segment: u32,
    pub addend: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct ExternalRelocation {
    pub target: SegmentTag,
    pub kind: RelocationType,
    pub is_batch_end: bool,
    pub addend: u32,
}

/// a module the loader has linked in.
#[derive(Debug, Clone)]
pub struct LoadedModule {
    pub name: String,
    pub base: VAddr,
    pub size: u32,
}

#[cfg(test)]
mod tests {
    use super::header::*;
    use super::*;

    /// Pokémon Alpha Sapphire's DllLangSelect.cro patches its jump table
    /// with type-2 relocations, whose entries are loaded straight into the PC
    /// by ldr pc, [pc, r1, lsl #2]. They must therefore be absolute.
    #[test]
    fn type_two_is_an_absolute_address() {
        assert_eq!(RelocationType::from_raw(2), RelocationType::AbsoluteAddress);
        assert_eq!(RelocationType::from_raw(10), RelocationType::AbsoluteAddress);
        assert_eq!(RelocationType::from_raw(3), RelocationType::RelativeAddress);
        assert_eq!(RelocationType::from_raw(0), RelocationType::Nothing);
    }

    /// every table in a module is laid out end to end, so a correct set of
    /// field offsets makes each table's end land exactly on the next table's
    /// start.
    #[test]
    fn header_offsets_match_a_real_module() {
        let values: &[(u32, u32)] = &[
            (CODE_OFFSET, 0x0000_0180),
            (CODE_SIZE, 0x0000_2E80),
            (DATA_OFFSET, 0x0000_3C2C),
            (DATA_SIZE, 0x0000_002C),
            (MODULE_NAME_OFFSET, 0x0000_3000),
            (MODULE_NAME_SIZE, 0x0000_000E),
            (SEGMENT_TABLE_OFFSET, 0x0000_3010),
            (SEGMENT_NUM, 0x0000_0006),
            (EXPORT_NAMED_SYMBOL_TABLE_OFFSET, 0x0000_3058),
            (EXPORT_NAMED_SYMBOL_NUM, 0x0000_0001),
            (EXPORT_TREE_TABLE_OFFSET, 0x0000_3060),
            (EXPORT_TREE_NUM, 0x0000_0002),
            (EXPORT_STRINGS_OFFSET, 0x0000_3070),
            (EXPORT_STRINGS_SIZE, 0x0000_0013),
            (IMPORT_MODULE_TABLE_OFFSET, 0x0000_3084),
            (IMPORT_MODULE_NUM, 0x0000_0001),
            (EXTERNAL_RELOCATION_TABLE_OFFSET, 0x0000_3098),
            (EXTERNAL_RELOCATION_NUM, 0x0000_005F),
            (IMPORT_ANONYMOUS_SYMBOL_TABLE_OFFSET, 0x0000_350C),
            (IMPORT_ANONYMOUS_SYMBOL_NUM, 0x0000_005D),
            (IMPORT_STRINGS_OFFSET, 0x0000_37F4),
            (IMPORT_STRINGS_SIZE, 0x0000_0009),
            (INTERNAL_RELOCATION_TABLE_OFFSET, 0x0000_3800),
            (INTERNAL_RELOCATION_NUM, 0x0000_0059),
        ];
        let get = |wanted: u32| {
            values
                .iter()
                .find(|(field, _)| *field == wanted)
                .map(|(_, value)| *value)
                .expect("field in the table above")
        };

        // "DllLangSelect\0" is fourteen bytes.
        assert_eq!(get(MODULE_NAME_SIZE), 14);

        assert_eq!(
            get(SEGMENT_TABLE_OFFSET) + get(SEGMENT_NUM) * SEGMENT_ENTRY_SIZE,
            get(EXPORT_NAMED_SYMBOL_TABLE_OFFSET),
            "the segment table should end where the export table starts"
        );
        assert_eq!(
            get(EXPORT_NAMED_SYMBOL_TABLE_OFFSET)
                + get(EXPORT_NAMED_SYMBOL_NUM) * EXPORT_NAMED_SYMBOL_ENTRY_SIZE,
            get(EXPORT_TREE_TABLE_OFFSET)
        );
        assert_eq!(
            get(EXPORT_TREE_TABLE_OFFSET) + get(EXPORT_TREE_NUM) * EXPORT_TREE_ENTRY_SIZE,
            get(EXPORT_STRINGS_OFFSET)
        );
        assert_eq!(
            get(IMPORT_MODULE_TABLE_OFFSET) + get(IMPORT_MODULE_NUM) * IMPORT_MODULE_ENTRY_SIZE,
            get(EXTERNAL_RELOCATION_TABLE_OFFSET)
        );
        assert_eq!(
            get(EXTERNAL_RELOCATION_TABLE_OFFSET)
                + get(EXTERNAL_RELOCATION_NUM) * RELOCATION_ENTRY_SIZE,
            get(IMPORT_ANONYMOUS_SYMBOL_TABLE_OFFSET)
        );
        assert_eq!(
            get(IMPORT_ANONYMOUS_SYMBOL_TABLE_OFFSET)
                + get(IMPORT_ANONYMOUS_SYMBOL_NUM) * IMPORT_ANONYMOUS_SYMBOL_ENTRY_SIZE,
            get(IMPORT_STRINGS_OFFSET)
        );
        // the internal relocations run right up to the data segment.
        assert_eq!(
            get(INTERNAL_RELOCATION_TABLE_OFFSET)
                + get(INTERNAL_RELOCATION_NUM) * RELOCATION_ENTRY_SIZE,
            get(DATA_OFFSET)
        );
    }
}
