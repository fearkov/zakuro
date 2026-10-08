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
    /// the same for the CPU's reads, which recompiled code reads too, but
    /// null over pages the host GPU drew and guest memory does not have yet,
    /// so that the CPU reading one asks for it first, see guard_cpu_reads.
    cpu_read_table: Vec<*mut u8>,
    /// the bytes of those pages the GPU drew, start to end, apart and in
    /// order. pages are shared, a buffer's first and last with whatever
    /// the title keeps next to it, which the CPU reads without waiting.
    guards: BTreeMap<VAddr, VAddr>,
    /// the same for the CPU's writes, which recompiled code makes too, null
    /// over pages the host GPU drew and guest memory does not have yet, so
    /// that the drawing reaches memory before a write of the CPU there, see
    /// guard_cpu_writes.
    cpu_write_table: Vec<*mut u8>,
    /// the bytes of those, as guards.
    write_guards: BTreeMap<VAddr, VAddr>,
    /// how the GPU's drawing comes down for such a read or write, while the
    /// CPU runs.
    gpu_sync: Option<GpuSync>,

    /// mappings keyed by base address, for svcQueryMemory and unmapping.
    mappings: BTreeMap<VAddr, Mapping>,

    /// accesses to unmapped addresses, deduplicated so one bad loop does not
    /// produce a gigabyte of log.
    faults: BTreeMap<VAddr, u32>,
}

/// what writes back the host GPU's drawing over a range, for a read or a
/// write of the CPU waiting on it. the system sets it to its GPU while the
/// CPU runs, when nothing else holds the GPU.
#[derive(Debug, Clone, Copy)]
pub struct GpuSync {
    pub gpu: *mut (),
    pub linear_base: u32,
    /// # Safety
    ///
    /// gpu has to point at what the function takes it for, which nothing
    /// else holds while it runs.
    pub sync: unsafe fn(gpu: *mut (), linear_base: u32, memory: &mut Memory, addr: VAddr, len: u32),
    /// the same for a write of the CPU, which has what every buffer the
    /// GPU drew there holds written back, depth or color.
    ///
    /// # Safety
    ///
    /// as sync.
    pub sync_write: unsafe fn(gpu: *mut (), linear_base: u32, memory: &mut Memory, addr: VAddr, len: u32),
}

// SAFETY: the pointers in the page tables all point into allocations owned by
// phys, which lives in the same struct and whose buffers are never
// reallocated. Nothing hands them out. the GPU sync's is only followed while
// the system that set it runs the CPU.
unsafe impl Send for Memory {}

impl Memory {
    pub fn new(new3ds: bool, app_bytes: u32) -> Memory {
        Memory {
            phys: PhysicalMemory::new(new3ds, app_bytes),
            read_table: vec![std::ptr::null_mut(); PAGE_TABLE_ENTRIES],
            write_table: vec![std::ptr::null_mut(); PAGE_TABLE_ENTRIES],
            cpu_read_table: vec![std::ptr::null_mut(); PAGE_TABLE_ENTRIES],
            guards: BTreeMap::new(),
            cpu_write_table: vec![std::ptr::null_mut(); PAGE_TABLE_ENTRIES],
            write_guards: BTreeMap::new(),
            gpu_sync: None,
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
                if !guarded_page(&self.guards, index) {
                    self.cpu_read_table[index] = ptr;
                }
            }
            if permission.contains(Permission::WRITE) {
                self.write_table[index] = ptr;
                if !guarded_page(&self.write_guards, index) {
                    self.cpu_write_table[index] = ptr;
                }
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
            self.cpu_read_table[index] = std::ptr::null_mut();
            self.cpu_write_table[index] = std::ptr::null_mut();
        }
        // a mapping the range covers only part of keeps the rest, at the
        // pages behind it, so freeing them later frees the right ones
        let end = vaddr as u64 + size as u64;
        let touched: Vec<Mapping> = self
            .mappings
            .values()
            .filter(|m| (m.base as u64) < end && (vaddr as u64) < m.base as u64 + m.size as u64)
            .copied()
            .collect();
        for m in touched {
            self.mappings.remove(&m.base);
            if m.base < vaddr {
                self.mappings.insert(m.base, Mapping { size: vaddr - m.base, ..m });
            }
            let m_end = m.base as u64 + m.size as u64;
            if m_end > end {
                let base = end as u32;
                self.mappings.insert(
                    base,
                    Mapping { base, size: (m_end - end) as u32, paddr: m.paddr + (base - m.base), ..m },
                );
            }
        }
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

    /// true when every page in the range is mapped and writable.
    pub fn is_writable(&self, vaddr: VAddr, size: u32) -> bool {
        (0..size.div_ceil(PAGE_SIZE)).all(|page| {
            let index = ((vaddr.wrapping_add(page * PAGE_SIZE)) >> PAGE_BITS) as usize;
            !self.write_table[index].is_null()
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
    /// the CPU's tables, for recompiled code.
    pub fn page_tables(&self) -> (*const *mut u8, *const *mut u8) {
        (self.cpu_read_table.as_ptr(), self.cpu_write_table.as_ptr())
    }

    /// the host GPU drew over a range, which guest memory gets only when
    /// asked, so the CPU's reads there ask first. games read what the GPU
    /// drew, Ocarina of Time 3D the depth where the sun is, to tell whether
    /// to draw its lens flare. the rest of a page the range shares is read
    /// the slow way while it is guarded, without asking.
    pub fn guard_cpu_reads(&mut self, vaddr: VAddr, len: u32) {
        add_guard(&mut self.guards, &mut self.cpu_read_table, vaddr, len);
    }

    /// the host GPU drew over a range, which guest memory gets only when
    /// asked, so the CPU's writes there have it written back first. on the
    /// console the drawing reached memory at once, and a title can put
    /// something else where a buffer it is done with was, Tomodachi Life its
    /// objects, which the drawing written back later would have broken.
    pub fn guard_cpu_writes(&mut self, vaddr: VAddr, len: u32) {
        add_guard(&mut self.write_guards, &mut self.cpu_write_table, vaddr, len);
    }

    /// how the CPU's reads of what the GPU drew come down from now on.
    ///
    /// # Safety
    ///
    /// sync.gpu has to stay what sync.sync and sync.sync_write take it for,
    /// and nothing else may hold it, until clear_gpu_sync, as reads and
    /// writes of the CPU follow it.
    pub unsafe fn set_gpu_sync(&mut self, sync: GpuSync) {
        self.gpu_sync = Some(sync);
    }

    /// the CPU stopped running, its reads and writes where the GPU drew see
    /// memory as it is.
    pub fn clear_gpu_sync(&mut self) {
        self.gpu_sync = None;
    }

    /// the page a read of the CPU of len bytes falls in, after what the GPU
    /// drew there came down.
    #[inline(always)]
    fn cpu_page_read(&mut self, addr: VAddr, len: u32) -> Option<&[u8]> {
        let ptr = self.cpu_read_table[(addr >> PAGE_BITS) as usize];
        if ptr.is_null() {
            return self.cpu_page_read_guarded(addr, len);
        }
        // SAFETY: as page_read, the CPU's table holds the same pointers
        Some(unsafe { std::slice::from_raw_parts(ptr, PAGE_SIZE as usize) })
    }

    #[cold]
    fn cpu_page_read_guarded(&mut self, addr: VAddr, len: u32) -> Option<&[u8]> {
        self.unguard(addr, len);
        self.page_read(addr)
    }

    /// has the GPU write back what it drew where a read of the CPU falls,
    /// the whole of each range it guarded there, and lets the CPU read the
    /// pages nothing guarded is left on. without a sync, outside the CPU,
    /// they stay guarded.
    fn unguard(&mut self, addr: VAddr, len: u32) {
        let Some(sync) = self.gpu_sync else { return };
        for (from, to) in take_guards(&mut self.guards, addr, len) {
            // SAFETY: the system set it while it runs the CPU, which is the
            // only thing reading through this
            unsafe { (sync.sync)(sync.gpu, sync.linear_base, self, from, to - from) };
            for index in (from >> PAGE_BITS) as usize..=((to - 1) >> PAGE_BITS) as usize {
                if !guarded_page(&self.guards, index) {
                    self.cpu_read_table[index] = self.read_table[index];
                }
            }
        }
    }

    /// the page a write of the CPU of len bytes falls in, after what the
    /// GPU drew there came down.
    #[inline(always)]
    fn cpu_page_write(&mut self, addr: VAddr, len: u32) -> Option<&mut [u8]> {
        let ptr = self.cpu_write_table[(addr >> PAGE_BITS) as usize];
        if ptr.is_null() {
            return self.cpu_page_write_guarded(addr, len);
        }
        // SAFETY: as page_write, the CPU's table holds the same pointers
        Some(unsafe { std::slice::from_raw_parts_mut(ptr, PAGE_SIZE as usize) })
    }

    #[cold]
    fn cpu_page_write_guarded(&mut self, addr: VAddr, len: u32) -> Option<&mut [u8]> {
        self.unguard_writes(addr, len);
        self.page_write(addr)
    }

    /// has the GPU write back what it drew where a write of the CPU falls,
    /// as unguard does for reads, before the write lands over it.
    fn unguard_writes(&mut self, addr: VAddr, len: u32) {
        let Some(sync) = self.gpu_sync else { return };
        for (from, to) in take_guards(&mut self.write_guards, addr, len) {
            // SAFETY: as in unguard
            unsafe { (sync.sync_write)(sync.gpu, sync.linear_base, self, from, to - from) };
            for index in (from >> PAGE_BITS) as usize..=((to - 1) >> PAGE_BITS) as usize {
                if !guarded_page(&self.write_guards, index) {
                    self.cpu_write_table[index] = self.write_table[index];
                }
            }
        }
    }

    /// a word of guest memory as anything but the CPU reads it, what the GPU
    /// drew there or not.
    pub fn peek32(&mut self, addr: VAddr) -> u32 {
        let mut buf = [0u8; 4];
        self.read_bytes(addr, &mut buf);
        u32::from_le_bytes(buf)
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

/// adds a range to those guarded, one range of the ones it overlaps, and
/// takes the pages it covers out of the table the CPU goes through. buffers
/// packed end to end stay apart, an access to one does not have the other
/// written back.
fn add_guard(guards: &mut BTreeMap<VAddr, VAddr>, table: &mut [*mut u8], vaddr: VAddr, len: u32) {
    let (mut start, mut end) = (vaddr, vaddr.saturating_add(len));
    if start >= end {
        return;
    }
    let joined: Vec<(VAddr, VAddr)> =
        guards.range(..end).rev().take_while(|(_, &to)| to > start).map(|(&from, &to)| (from, to)).collect();
    for (from, to) in joined {
        guards.remove(&from);
        (start, end) = (start.min(from), end.max(to));
    }
    guards.insert(start, end);
    let pages = (vaddr >> PAGE_BITS) as usize..=((vaddr.saturating_add(len - 1)) >> PAGE_BITS) as usize;
    table[pages].fill(std::ptr::null_mut());
}

/// whether any of a page is still guarded.
fn guarded_page(guards: &BTreeMap<VAddr, VAddr>, index: usize) -> bool {
    let start = (index as u32) << PAGE_BITS;
    let end = start.wrapping_add(PAGE_SIZE);
    // the ranges are apart, only the last one starting before the page's
    // end can reach into it
    guards.range(..end).next_back().is_some_and(|(_, &to)| to > start)
}

/// the guarded ranges an access of len bytes at addr falls in, no longer
/// guarded.
fn take_guards(guards: &mut BTreeMap<VAddr, VAddr>, addr: VAddr, len: u32) -> Vec<(VAddr, VAddr)> {
    let end = addr.saturating_add(len.max(1));
    let hit: Vec<(VAddr, VAddr)> =
        guards.range(..end).rev().take_while(|(_, &to)| to > addr).map(|(&from, &to)| (from, to)).collect();
    for (from, _) in &hit {
        guards.remove(from);
    }
    hit
}

impl Bus for Memory {
    #[inline(always)]
    fn read8(&mut self, addr: u32) -> u8 {
        match self.cpu_page_read(addr, 1) {
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
            match self.cpu_page_read(addr, 2) {
                Some(page) => u16::from_le_bytes([page[offset], page[offset + 1]]),
                None => {
                    self.note_fault(addr);
                    0
                }
            }
        } else {
            // straddles a page boundary.
            self.unguard(addr, 2);
            let mut buf = [0u8; 2];
            self.read_bytes(addr, &mut buf);
            u16::from_le_bytes(buf)
        }
    }

    #[inline(always)]
    fn read32(&mut self, addr: u32) -> u32 {
        let offset = (addr & PAGE_MASK) as usize;
        if offset <= PAGE_SIZE as usize - 4 {
            match self.cpu_page_read(addr, 4) {
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
            self.unguard(addr, 4);
            let mut buf = [0u8; 4];
            self.read_bytes(addr, &mut buf);
            u32::from_le_bytes(buf)
        }
    }

    #[inline(always)]
    fn write8(&mut self, addr: u32, value: u8) {
        match self.cpu_page_write(addr, 1) {
            Some(page) => page[(addr & PAGE_MASK) as usize] = value,
            None => self.note_fault(addr),
        }
    }

    #[inline(always)]
    fn write16(&mut self, addr: u32, value: u16) {
        let offset = (addr & PAGE_MASK) as usize;
        if offset <= PAGE_SIZE as usize - 2 {
            match self.cpu_page_write(addr, 2) {
                Some(page) => page[offset..offset + 2].copy_from_slice(&value.to_le_bytes()),
                None => self.note_fault(addr),
            }
        } else {
            self.unguard_writes(addr, 2);
            self.write_bytes(addr, &value.to_le_bytes());
        }
    }

    #[inline(always)]
    fn write32(&mut self, addr: u32, value: u32) {
        let offset = (addr & PAGE_MASK) as usize;
        if offset <= PAGE_SIZE as usize - 4 {
            match self.cpu_page_write(addr, 4) {
                Some(page) => page[offset..offset + 4].copy_from_slice(&value.to_le_bytes()),
                None => self.note_fault(addr),
            }
        } else {
            self.unguard_writes(addr, 4);
            self.write_bytes(addr, &value.to_le_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// a GPU that writes back 0xAB over what it is asked for, counting.
    struct Drawn {
        asked: u32,
    }

    unsafe fn write_back(gpu: *mut (), _linear_base: u32, memory: &mut Memory, addr: VAddr, len: u32) {
        let drawn = unsafe { &mut *(gpu as *mut Drawn) };
        drawn.asked += 1;
        memory.write_bytes(addr, &vec![0xAB; len as usize]);
    }

    /// a read of the CPU where the GPU drew has it written back first, once
    /// for the whole of what was guarded, while the emulator's own reads,
    /// and the CPU's with no GPU to ask, see memory as it is.
    #[test]
    fn the_cpu_reads_what_the_gpu_drew() {
        let mut memory = Memory::new(false, 0x0400_0000);
        let block = memory.phys.allocate(MemoryRegion::Application, 0x3000).unwrap();
        memory.map(0x1F00_0000, block.addr, 0x3000, Permission::RW, MemoryState::Private);
        memory.guard_cpu_reads(0x1F00_1000, 0x2000);
        assert_eq!(memory.read32(0x1F00_1000), 0);

        let mut drawn = Drawn { asked: 0 };
        let gpu = &mut drawn as *mut Drawn as *mut ();
        // SAFETY: drawn outlives the reads below, which are all that use it
        unsafe { memory.set_gpu_sync(GpuSync { gpu, linear_base: 0, sync: write_back, sync_write: write_back }) };
        let mut bytes = [0u8; 4];
        memory.read_bytes(0x1F00_1000, &mut bytes);
        assert_eq!(bytes, [0; 4]);
        assert_eq!(memory.read32(0x1F00_0000), 0);
        assert_eq!(drawn.asked, 0);

        assert_eq!(memory.read32(0x1F00_2004), 0xABAB_ABAB);
        assert_eq!(memory.read8(0x1F00_1000), 0xAB);
        assert_eq!(drawn.asked, 1);
        // until the GPU draws there again
        memory.guard_cpu_reads(0x1F00_1000, 4);
        assert_eq!(memory.read16(0x1F00_1002), 0xABAB);
        assert_eq!(drawn.asked, 2);
        // a page the GPU drew only part of, the title keeping its own next
        // to it, which the CPU reads without waiting
        memory.write32(0x1F00_2000, 0x1234_5678);
        memory.guard_cpu_reads(0x1F00_2800, 0x100);
        assert_eq!(memory.read32(0x1F00_2000), 0x1234_5678);
        assert_eq!(drawn.asked, 2);
        assert_eq!(memory.read32(0x1F00_28FC), 0xABAB_ABAB);
        assert_eq!(drawn.asked, 3);
        // a read starting before a range and ending in it asks too
        memory.guard_cpu_reads(0x1F00_2808, 0x10);
        assert_eq!(memory.read32(0x1F00_2806), 0xABAB_ABAB);
        assert_eq!(drawn.asked, 4);
        // buffers end to end stay apart
        memory.guard_cpu_reads(0x1F00_1000, 0x800);
        memory.guard_cpu_reads(0x1F00_1800, 0x800);
        memory.read32(0x1F00_1C00);
        assert_eq!(drawn.asked, 5);
        assert_eq!(memory.guards.len(), 1);
        // and mapping a page again keeps what is guarded on it
        memory.map(0x1F00_1000, block.addr + 0x1000, 0x1000, Permission::RW, MemoryState::Private);
        memory.read32(0x1F00_1000);
        assert_eq!(drawn.asked, 6);
        memory.clear_gpu_sync();
    }

    /// a write of the CPU where the GPU drew has the drawing written back
    /// first, once for the whole of what was guarded, so the write lands
    /// over it, while reads, and writes elsewhere, go on as they are.
    #[test]
    fn the_cpu_writes_over_what_the_gpu_drew() {
        let mut memory = Memory::new(false, 0x0400_0000);
        let block = memory.phys.allocate(MemoryRegion::Application, 0x3000).unwrap();
        memory.map(0x1F00_0000, block.addr, 0x3000, Permission::RW, MemoryState::Private);
        memory.guard_cpu_writes(0x1F00_1000, 0x2000);

        let mut drawn = Drawn { asked: 0 };
        let gpu = &mut drawn as *mut Drawn as *mut ();
        // SAFETY: drawn outlives the writes below, which are all that use it
        unsafe { memory.set_gpu_sync(GpuSync { gpu, linear_base: 0, sync: write_back, sync_write: write_back }) };
        assert_eq!(memory.read32(0x1F00_1000), 0);
        memory.write32(0x1F00_0FFC, 1);
        assert_eq!(drawn.asked, 0);

        memory.write8(0x1F00_1004, 0x12);
        assert_eq!(drawn.asked, 1);
        assert_eq!(memory.read32(0x1F00_1004), 0xABAB_AB12);
        memory.write16(0x1F00_2FFE, 0x3456);
        assert_eq!(drawn.asked, 1);
        assert_eq!(memory.read32(0x1F00_2FFC), 0x3456_ABAB);
        // a write from the page before reaching into a range asks too
        memory.guard_cpu_writes(0x1F00_1000, 0x10);
        memory.write32(0x1F00_0FFE, 0x7788_9900);
        assert_eq!(drawn.asked, 2);
        assert_eq!(memory.read32(0x1F00_1000), 0xABAB_7788);
        // and a page mapped again keeps it
        memory.guard_cpu_writes(0x1F00_2000, 4);
        memory.map(0x1F00_2000, block.addr + 0x2000, 0x1000, Permission::RW, MemoryState::Private);
        memory.write8(0x1F00_2000, 0);
        assert_eq!(drawn.asked, 3);
        assert_eq!(memory.read32(0x1F00_2000), 0xABAB_AB00);
        memory.clear_gpu_sync();
    }
}
