//! virtual memory, the guest page table, and the [Bus] the CPU talks to.

pub mod config;
pub mod physical;

use std::collections::BTreeMap;

use zakuro_common::memory_map::*;
use zakuro_common::VAddr;
use zakuro_cpu::Bus;

pub use physical::{MemoryRegion, PhysicalBlock, PhysicalMemory};

bitflags::bitflags! {
    /// page permissions as svcQueryMemory reports them.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Permission: u32 {
        const READ = 1;
        const WRITE = 2;
        const EXECUTE = 4;
    }
}

impl Permission {
    pub const RW: Permission = Permission::READ.union(Permission::WRITE);
    pub const RX: Permission = Permission::READ.union(Permission::EXECUTE);
}

/// MemoryState as the kernel reports it through svcQueryMemory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MemoryState {
    Free = 0,
    Reserved = 1,
    Io = 2,
    Static = 3,
    Code = 4,
    Private = 5,
    Shared = 6,
    Continuous = 7,
    Aliased = 8,
    Alias = 9,
    AliasCode = 10,
    Locked = 11,
}

/// one contiguous virtual mapping.
#[derive(Debug, Clone, Copy)]
pub struct Mapping {
    pub base: VAddr,
    pub size: u32,
    pub paddr: u32,
    pub permission: Permission,
    pub state: MemoryState,
}

/// what svcQueryMemory hands back.
#[derive(Debug, Clone, Copy)]
pub struct MemoryInfo {
    pub base: VAddr,
    pub size: u32,
    pub permission: u32,
    pub state: u32,
}

pub struct Memory {
    pub phys: PhysicalMemory,

    /// host pointer for each 4 KiB page, or null when unmapped.
    read_table: Vec<*mut u8>,
    write_table: Vec<*mut u8>,

    /// mappings keyed by base address, for svcQueryMemory and unmapping.
    mappings: BTreeMap<VAddr, Mapping>,

    /// accesses to unmapped addresses, deduplicated so one bad loop does not
    /// produce a gigabyte of log.
    faults: BTreeMap<VAddr, u32>,
}

// SAFETY: the pointers in the page tables all point into allocations owned by
// phys, which lives in the same struct and whose buffers are never
// reallocated. Nothing hands them out.
unsafe impl Send for Memory {}

impl Memory {
    pub fn new(new3ds: bool, app_bytes: u32) -> Memory {
        Memory {
            phys: PhysicalMemory::new(new3ds, app_bytes),
            read_table: vec![std::ptr::null_mut(); PAGE_TABLE_ENTRIES],
            write_table: vec![std::ptr::null_mut(); PAGE_TABLE_ENTRIES],
            mappings: BTreeMap::new(),
            faults: BTreeMap::new(),
        }
    }

    /// maps size bytes of physical memory at vaddr.
    pub fn map(
        &mut self,
        vaddr: VAddr,
        paddr: u32,
        size: u32,
        permission: Permission,
        state: MemoryState,
    ) {
        // services hand out blocks sized to their contents rather than to a
        // page, so round up rather than refusing.
        let vaddr = vaddr & !PAGE_MASK;
        let size = (size + PAGE_MASK) & !PAGE_MASK;

        for page in 0..size / PAGE_SIZE {
            let page_vaddr = vaddr + page * PAGE_SIZE;
            let page_paddr = paddr + page * PAGE_SIZE;
            let Some(host) = self.phys.host_slice_mut(page_paddr, PAGE_SIZE) else {
                log::error!("map: no physical memory at 0x{page_paddr:08X}");
                continue;
            };
            let ptr = host.as_mut_ptr();
            let index = (page_vaddr >> PAGE_BITS) as usize;
            if permission.contains(Permission::READ) {
                self.read_table[index] = ptr;
            }
            if permission.contains(Permission::WRITE) {
                self.write_table[index] = ptr;
            }
        }

        self.mappings.insert(
            vaddr,
            Mapping {
                base: vaddr,
                size,
                paddr,
                permission,
                state,
            },
        );
    }

    /// maps the physical pages behind source at destination as well, so
    /// both addresses name the same memory.
    pub fn mirror(&mut self, destination: VAddr, source: VAddr, size: u32) -> bool {
        let size = (size + PAGE_MASK) & !PAGE_MASK;
        let mut done = 0;
        while done < size {
            let from = source + done;
            let Some(mapping) = self.mapping_at(from).copied() else {
                log::warn!("mirror: source 0x{from:08X} is not mapped");
                return false;
            };
            let offset = from - mapping.base;
            let chunk = (mapping.size - offset).min(size - done);
            self.map(
                destination + done,
                mapping.paddr + offset,
                chunk,
                mapping.permission | Permission::RW,
                MemoryState::Alias,
            );
            done += chunk;
        }
        true
    }

    pub fn unmap(&mut self, vaddr: VAddr, size: u32) {
        for page in 0..size / PAGE_SIZE {
            let index = ((vaddr + page * PAGE_SIZE) >> PAGE_BITS) as usize;
            self.read_table[index] = std::ptr::null_mut();
            self.write_table[index] = std::ptr::null_mut();
        }
        self.mappings.retain(|_, m| {
            !(m.base >= vaddr && m.base + m.size <= vaddr.wrapping_add(size))
        });
    }

    pub fn mapping_at(&self, vaddr: VAddr) -> Option<&Mapping> {
        self.mappings
            .range(..=vaddr)
            .next_back()
            .map(|(_, m)| m)
            .filter(|m| vaddr < m.base.wrapping_add(m.size))
    }

    /// svcQueryMemory, describes the mapping containing vaddr, or the free
    /// gap it sits in.
    pub fn query(&self, vaddr: VAddr) -> MemoryInfo {
        if let Some(m) = self.mapping_at(vaddr) {
            return MemoryInfo {
                base: m.base,
                size: m.size,
                permission: m.permission.bits(),
                state: m.state as u32,
            };
        }

        // report the free gap between the surrounding mappings, which is what
        // allocators walking the address space expect.
        let start = self
            .mappings
            .range(..=vaddr)
            .next_back()
            .map(|(_, m)| m.base.wrapping_add(m.size))
            .unwrap_or(0);
        let end = self
            .mappings
            .range(vaddr..)
            .next()
            .map(|(&base, _)| base)
            .unwrap_or(0xFFFF_F000);

        MemoryInfo {
            base: start,
            size: end.wrapping_sub(start),
            permission: 0,
            state: MemoryState::Free as u32,
        }
    }

    pub fn mappings(&self) -> impl Iterator<Item = &Mapping> {
        self.mappings.values()
    }

    /// writes a word to any mapped page, whatever its permissions say.
    pub fn write32_privileged(&mut self, vaddr: VAddr, value: u32) -> bool {
        let offset = (vaddr & PAGE_MASK) as usize;
        if offset > PAGE_SIZE as usize - 4 {
            // straddles a page, fall back to the byte path, which resolves
            // each page separately.
            let bytes = value.to_le_bytes();
            for (i, byte) in bytes.iter().enumerate() {
                let address = vaddr.wrapping_add(i as u32);
                let ptr = self.read_table[(address >> PAGE_BITS) as usize];
                if ptr.is_null() {
                    return false;
                }
                // SAFETY: a non-null entry points at a live PAGE_SIZE slice.
                unsafe { *ptr.add((address & PAGE_MASK) as usize) = *byte };
            }
            return true;
        }

        let ptr = self.read_table[(vaddr >> PAGE_BITS) as usize];
        if ptr.is_null() {
            return false;
        }
        // SAFETY: as above, and the offset is within the page.
        unsafe {
            std::ptr::copy_nonoverlapping(value.to_le_bytes().as_ptr(), ptr.add(offset), 4);
        }
        true
    }

    /// true when instructions may be fetched from vaddr.
    pub fn is_executable(&self, vaddr: VAddr) -> bool {
        !self.read_table[(vaddr >> PAGE_BITS) as usize].is_null()
    }

    /// true when every page in the range is mapped and readable.
    pub fn is_readable(&self, vaddr: VAddr, size: u32) -> bool {
        (0..size.div_ceil(PAGE_SIZE)).all(|page| {
            let index = ((vaddr.wrapping_add(page * PAGE_SIZE)) >> PAGE_BITS) as usize;
            !self.read_table[index].is_null()
        })
    }

    // -- bulk access --------------------------------------------------------

    /// copies a block of guest memory out, page by page so it can straddle
    /// mappings. Bytes from unmapped pages read as zero.
    pub fn read_bytes(&mut self, vaddr: VAddr, out: &mut [u8]) {
        let mut done = 0usize;
        while done < out.len() {
            let addr = vaddr.wrapping_add(done as u32);
            let offset = (addr & PAGE_MASK) as usize;
            let chunk = (PAGE_SIZE as usize - offset).min(out.len() - done);
            match self.page_read(addr) {
                Some(page) => out[done..done + chunk]
                    .copy_from_slice(&page[offset..offset + chunk]),
                None => {
                    self.note_fault(addr);
                    out[done..done + chunk].fill(0);
                }
            }
            done += chunk;
        }
    }

    pub fn write_bytes(&mut self, vaddr: VAddr, data: &[u8]) {
        let mut done = 0usize;
        while done < data.len() {
            let addr = vaddr.wrapping_add(done as u32);
            let offset = (addr & PAGE_MASK) as usize;
            let chunk = (PAGE_SIZE as usize - offset).min(data.len() - done);
            match self.page_write(addr) {
                Some(page) => page[offset..offset + chunk]
                    .copy_from_slice(&data[done..done + chunk]),
                None => self.note_fault(addr),
            }
            done += chunk;
        }
    }

    /// writes straight to physical memory, bypassing the page table and its
    /// permissions.
    pub fn write_physical(&mut self, paddr: u32, data: &[u8]) -> bool {
        match self.phys.host_slice_mut(paddr, data.len() as u32) {
            Some(slice) => {
                slice.copy_from_slice(data);
                true
            }
            None => {
                log::error!(
                    "write of {} bytes to unbacked physical address 0x{paddr:08X}",
                    data.len()
                );
                false
            }
        }
    }

    /// zeroes a physical range, for BSS and freshly allocated pages.
    pub fn zero_physical(&mut self, paddr: u32, size: u32) -> bool {
        match self.phys.host_slice_mut(paddr, size) {
            Some(slice) => {
                slice.fill(0);
                true
            }
            None => {
                log::error!("zeroing unbacked physical address 0x{paddr:08X}");
                false
            }
        }
    }

    /// reads a NUL-terminated ASCII string, capped so a corrupt pointer cannot
    /// make us read forever.
    pub fn read_cstring(&mut self, vaddr: VAddr, max: usize) -> String {
        let mut bytes = Vec::new();
        for i in 0..max {
            let b = self.read8(vaddr.wrapping_add(i as u32));
            if b == 0 {
                break;
            }
            bytes.push(b);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    #[inline(always)]
    fn page_read(&self, addr: VAddr) -> Option<&[u8]> {
        let ptr = self.read_table[(addr >> PAGE_BITS) as usize];
        if ptr.is_null() {
            return None;
        }
        // SAFETY: a non-null entry was installed by map from a live slice of
        // phys that is exactly PAGE_SIZE long.
        Some(unsafe { std::slice::from_raw_parts(ptr, PAGE_SIZE as usize) })
    }

    #[inline(always)]
    fn page_write(&mut self, addr: VAddr) -> Option<&mut [u8]> {
        let ptr = self.write_table[(addr >> PAGE_BITS) as usize];
        if ptr.is_null() {
            return None;
        }
        // SAFETY: as above, and &mut self means no other borrow is live.
        Some(unsafe { std::slice::from_raw_parts_mut(ptr, PAGE_SIZE as usize) })
    }

    /// the read and write page tables, a host pointer per 4 KiB page or
    /// null, for code that reads memory without going through the bus.
    pub fn page_tables(&self) -> (*const *mut u8, *const *mut u8) {
        (self.read_table.as_ptr(), self.write_table.as_ptr())
    }

    fn note_fault(&mut self, addr: VAddr) {
        let page = addr & !PAGE_MASK;
        let count = self.faults.entry(page).or_insert(0);
        *count += 1;
        if *count == 1 {
            log::warn!("access to unmapped guest address 0x{addr:08X}");
        }
    }

    /// unmapped pages touched so far, for the diagnostics overlay.
    pub fn fault_summary(&self) -> Vec<(VAddr, u32)> {
        self.faults.iter().map(|(&a, &c)| (a, c)).collect()
    }
}

// ---------------------------------------------------------------------------
// The CPU-facing bus
// ---------------------------------------------------------------------------

impl Bus for Memory {
    #[inline(always)]
    fn read8(&mut self, addr: u32) -> u8 {
        match self.page_read(addr) {
            Some(page) => page[(addr & PAGE_MASK) as usize],
            None => {
                self.note_fault(addr);
                0
            }
        }
    }

    #[inline(always)]
    fn read16(&mut self, addr: u32) -> u16 {
        let offset = (addr & PAGE_MASK) as usize;
        if offset <= PAGE_SIZE as usize - 2 {
            match self.page_read(addr) {
                Some(page) => u16::from_le_bytes([page[offset], page[offset + 1]]),
                None => {
                    self.note_fault(addr);
                    0
                }
            }
        } else {
            // straddles a page boundary.
            let mut buf = [0u8; 2];
            self.read_bytes(addr, &mut buf);
            u16::from_le_bytes(buf)
        }
    }

    #[inline(always)]
    fn read32(&mut self, addr: u32) -> u32 {
        let offset = (addr & PAGE_MASK) as usize;
        if offset <= PAGE_SIZE as usize - 4 {
            match self.page_read(addr) {
                Some(page) => u32::from_le_bytes([
                    page[offset],
                    page[offset + 1],
                    page[offset + 2],
                    page[offset + 3],
                ]),
                None => {
                    self.note_fault(addr);
                    0
                }
            }
        } else {
            let mut buf = [0u8; 4];
            self.read_bytes(addr, &mut buf);
            u32::from_le_bytes(buf)
        }
    }

    #[inline(always)]
    fn write8(&mut self, addr: u32, value: u8) {
        match self.page_write(addr) {
            Some(page) => page[(addr & PAGE_MASK) as usize] = value,
            None => self.note_fault(addr),
        }
    }

    #[inline(always)]
    fn write16(&mut self, addr: u32, value: u16) {
        let offset = (addr & PAGE_MASK) as usize;
        if offset <= PAGE_SIZE as usize - 2 {
            match self.page_write(addr) {
                Some(page) => page[offset..offset + 2].copy_from_slice(&value.to_le_bytes()),
                None => self.note_fault(addr),
            }
        } else {
            self.write_bytes(addr, &value.to_le_bytes());
        }
    }

    #[inline(always)]
    fn write32(&mut self, addr: u32, value: u32) {
        let offset = (addr & PAGE_MASK) as usize;
        if offset <= PAGE_SIZE as usize - 4 {
            match self.page_write(addr) {
                Some(page) => page[offset..offset + 4].copy_from_slice(&value.to_le_bytes()),
                None => self.note_fault(addr),
            }
        } else {
            self.write_bytes(addr, &value.to_le_bytes());
        }
    }
}
