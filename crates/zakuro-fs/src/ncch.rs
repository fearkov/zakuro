//! NCCH containers (.cxi executables and .cfa archives) and their extended
//! header.

use crate::ncsd::MEDIA_UNIT;
use crate::reader::Reader;
use crate::FsError;

#[derive(Debug, Clone)]
pub struct NcchHeader {
    pub content_size: u64,
    pub partition_id: u64,
    pub program_id: u64,
    pub maker_code: String,
    pub version: u16,
    pub product_code: String,

    pub exheader_size: u32,

    pub logo_offset: u64,
    pub logo_size: u64,
    pub plain_offset: u64,
    pub plain_size: u64,
    pub exefs_offset: u64,
    pub exefs_size: u64,
    pub exefs_hash_size: u64,
    pub romfs_offset: u64,
    pub romfs_size: u64,
    pub romfs_hash_size: u64,

    pub flags: [u8; 8],
}

impl NcchHeader {
    pub const SIZE: usize = 0x200;
    /// the header proper starts after the 0x100-byte RSA signature.
    pub const HEADER_OFFSET: usize = 0x100;

    pub fn detect(data: &[u8]) -> bool {
        data.len() > Self::HEADER_OFFSET + 4
            && &data[Self::HEADER_OFFSET..Self::HEADER_OFFSET + 4] == b"NCCH"
    }

    pub fn parse(data: &[u8]) -> Result<NcchHeader, FsError> {
        let h = Reader::new(data).at(Self::HEADER_OFFSET)?;
        h.magic(0x00, b"NCCH")?;

        let mut flags = [0u8; 8];
        flags.copy_from_slice(h.bytes(0x88, 8)?);

        Ok(NcchHeader {
            content_size: h.u32(0x04)? as u64 * MEDIA_UNIT,
            partition_id: h.u64(0x08)?,
            maker_code: h.ascii(0x10, 2)?,
            version: h.u16(0x12)?,
            program_id: h.u64(0x18)?,
            product_code: h.ascii(0x50, 0x10)?,
            exheader_size: h.u32(0x80)?,
            plain_offset: h.u32(0x90)? as u64 * MEDIA_UNIT,
            plain_size: h.u32(0x94)? as u64 * MEDIA_UNIT,
            logo_offset: h.u32(0x98)? as u64 * MEDIA_UNIT,
            logo_size: h.u32(0x9C)? as u64 * MEDIA_UNIT,
            exefs_offset: h.u32(0xA0)? as u64 * MEDIA_UNIT,
            exefs_size: h.u32(0xA4)? as u64 * MEDIA_UNIT,
            exefs_hash_size: h.u32(0xA8)? as u64 * MEDIA_UNIT,
            romfs_offset: h.u32(0xB0)? as u64 * MEDIA_UNIT,
            romfs_size: h.u32(0xB4)? as u64 * MEDIA_UNIT,
            romfs_hash_size: h.u32(0xB8)? as u64 * MEDIA_UNIT,
            flags,
        })
    }

    /// crypto method, 0x00 original, 0x01 7.x, 0x0A Secure3, 0x0B Secure4.
    pub fn crypto_method(&self) -> u8 {
        self.flags[3]
    }

    /// 1 = CTR (Old3DS), 2 = "snake" (New3DS only).
    pub fn platform(&self) -> u8 {
        self.flags[4]
    }

    pub fn is_executable(&self) -> bool {
        self.flags[5] & 0x02 != 0
    }

    /// true when the partition is stored in the clear, which is the case for
    /// decrypted dumps. Anything else needs the console's AES key scrambler.
    pub fn is_decrypted(&self) -> bool {
        self.flags[7] & 0x04 != 0
    }

    pub fn uses_fixed_key(&self) -> bool {
        self.flags[7] & 0x01 != 0
    }

    pub fn has_romfs(&self) -> bool {
        self.romfs_size != 0 && self.flags[7] & 0x02 == 0
    }
}

// ---------------------------------------------------------------------------
// Extended header
// ---------------------------------------------------------------------------

/// where one segment of .code is mapped, straight out of the exheader.
#[derive(Debug, Clone, Copy, Default)]
pub struct CodeSetInfo {
    pub address: u32,
    /// size of the mapped region, in pages.
    pub num_pages: u32,
    /// size of the actual data, in bytes (<= num_pages * 0x1000).
    pub size: u32,
}

impl CodeSetInfo {
    fn parse(r: &Reader<'_>, offset: usize) -> Result<Self, FsError> {
        Ok(CodeSetInfo {
            address: r.u32(offset)?,
            num_pages: r.u32(offset + 4)?,
            size: r.u32(offset + 8)?,
        })
    }
}

/// how much FCRAM the kernel reserves for the process, from the ARM11 kernel
/// capability descriptors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MemoryType {
    #[default]
    Application,
    System,
    Base,
}

impl MemoryType {
    fn from_raw(raw: u32) -> MemoryType {
        match raw {
            2 => MemoryType::System,
            3 => MemoryType::Base,
            _ => MemoryType::Application,
        }
    }
}

/// Old3DS system mode, which decides the FCRAM split between the application
/// and the system. ORAS ships as Prod (64 MiB for the game).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SystemMode {
    /// 64 MiB application / 32 MiB system / 32 MiB base
    #[default]
    Prod,
    /// 96 MiB application
    Dev1,
    /// 80 MiB application
    Dev2,
    /// 72 MiB application
    Dev3,
    /// 32 MiB application
    Dev4,
}

impl SystemMode {
    fn from_raw(raw: u8) -> SystemMode {
        match raw {
            2 => SystemMode::Dev1,
            3 => SystemMode::Dev2,
            4 => SystemMode::Dev3,
            5 => SystemMode::Dev4,
            _ => SystemMode::Prod,
        }
    }

    /// bytes of FCRAM the APPLICATION memory region gets.
    pub fn application_memory(self) -> u32 {
        match self {
            SystemMode::Prod => 64 * 1024 * 1024,
            SystemMode::Dev1 => 96 * 1024 * 1024,
            SystemMode::Dev2 => 80 * 1024 * 1024,
            SystemMode::Dev3 => 72 * 1024 * 1024,
            SystemMode::Dev4 => 32 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExHeader {
    pub title: String,
    /// when set, ExeFS .code is BLZ-compressed.
    pub compress_code: bool,
    pub sd_application: bool,
    pub remaster_version: u16,

    pub text: CodeSetInfo,
    pub rodata: CodeSetInfo,
    pub data: CodeSetInfo,
    pub bss_size: u32,
    pub stack_size: u32,

    pub dependencies: Vec<u64>,
    pub savedata_size: u64,
    pub jump_id: u64,

    pub program_id: u64,
    pub core_version: u32,
    pub main_thread_priority: u8,
    pub ideal_processor: u8,
    pub affinity_mask: u8,
    pub system_mode: SystemMode,
    /// New3DS system mode, 0 legacy, 1 = 124 MiB, 2 = 178 MiB.
    pub n3ds_system_mode: u8,
    /// New3DS CPU clock, 0 = 268 MHz, 1 = 804 MHz.
    pub n3ds_cpu_speed: u8,
    pub enable_l2_cache: bool,
    pub resource_limit_category: u8,
    pub service_access: Vec<String>,

    pub memory_type: MemoryType,
    pub handle_table_size: u32,
    /// SVC numbers the title is allowed to call.
    pub allowed_svcs: [u32; 4],
}

impl ExHeader {
    /// size of the region covered by the NCCH header's exheader hash.
    pub const HASHED_SIZE: usize = 0x400;
    /// size of the whole exheader region in the file.
    pub const FULL_SIZE: usize = 0x800;

    pub fn parse(data: &[u8]) -> Result<ExHeader, FsError> {
        let r = Reader::new(data);
        let sci = r.at(0x000)?;
        let aci = r.at(0x200)?;

        let flags = sci.u8(0x0D)?;

        let mut dependencies = Vec::new();
        for i in 0..48 {
            let dep = sci.u64(0x40 + i * 8)?;
            if dep != 0 {
                dependencies.push(dep);
            }
        }

        // ARM11 local system capabilities live at the start of the ACI.
        let flag1 = aci.u8(0x0C)?;
        let flag2 = aci.u8(0x0D)?;
        let flag0 = aci.u8(0x0E)?;

        let mut service_access = Vec::new();
        for i in 0..34 {
            let name = aci.ascii(0x50 + i * 8, 8)?;
            if !name.is_empty() {
                service_access.push(name);
            }
        }

        // ARM11 kernel capabilities, 28 tagged u32 descriptors at ACI+0x170.
        let mut memory_type = MemoryType::Application;
        let mut handle_table_size = 0x200;
        let mut allowed_svcs = [0u32; 4];
        for i in 0..28 {
            let desc = aci.u32(0x170 + i * 4)?;
            if desc == 0xFFFF_FFFF {
                continue;
            }
            let tag = desc >> 20;
            if tag & 0xF80 == 0xF00 {
                // allowed syscall mask, 24 SVC numbers per descriptor.
                let index = ((desc >> 24) & 7) as usize * 24;
                let bits = desc & 0x00FF_FFFF;
                for bit in 0..24 {
                    if bits & (1 << bit) != 0 {
                        let svc = index + bit;
                        allowed_svcs[svc / 32] |= 1 << (svc % 32);
                    }
                }
            } else if tag & 0xFF0 == 0xFE0 {
                handle_table_size = desc & 0x3FF;
            } else if tag & 0xFF8 == 0xFF0 {
                // misc parameters, bits 8..11 hold the memory type.
                memory_type = MemoryType::from_raw((desc >> 8) & 0xF);
            }
            // the remaining tags describe mapped IO ranges and interrupts,
            // which an HLE kernel does not need to honour.
        }

        Ok(ExHeader {
            title: sci.ascii(0x00, 8)?,
            compress_code: flags & 0x01 != 0,
            sd_application: flags & 0x02 != 0,
            remaster_version: sci.u16(0x0E)?,
            text: CodeSetInfo::parse(&sci, 0x10)?,
            stack_size: sci.u32(0x1C)?,
            rodata: CodeSetInfo::parse(&sci, 0x20)?,
            data: CodeSetInfo::parse(&sci, 0x30)?,
            bss_size: sci.u32(0x3C)?,
            dependencies,
            savedata_size: sci.u64(0x1C0)?,
            jump_id: sci.u64(0x1C8)?,
            program_id: aci.u64(0x00)?,
            core_version: aci.u32(0x08)?,
            main_thread_priority: aci.u8(0x0F)?,
            ideal_processor: flag0 & 0x3,
            affinity_mask: (flag0 >> 2) & 0x3,
            system_mode: SystemMode::from_raw((flag0 >> 4) & 0xF),
            n3ds_system_mode: (flag1 >> 4) & 0xF,
            n3ds_cpu_speed: (flag2 >> 1) & 0x1,
            enable_l2_cache: flag2 & 0x1 != 0,
            resource_limit_category: aci.u8(0x16F)?,
            service_access,
            memory_type,
            handle_table_size,
            allowed_svcs,
        })
    }

    /// total bytes the three code segments occupy once mapped, including BSS.
    pub fn code_span(&self) -> u32 {
        let end = self.data.address + self.data.num_pages * 0x1000 + self.bss_size;
        end.saturating_sub(self.text.address)
    }
}
