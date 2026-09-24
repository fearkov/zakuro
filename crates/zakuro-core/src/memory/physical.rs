//! physical memory and the FCRAM page allocator.

use zakuro_common::memory_map::*;

/// which of the kernel's three FCRAM regions an allocation comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryRegion {
    Application,
    System,
    Base,
}

/// a contiguous run of physical pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalBlock {
    pub addr: u32,
    pub size: u32,
}

/// backing store for every physical memory the CPU can reach.
pub struct PhysicalMemory {
    pub fcram: Box<[u8]>,
    pub vram: Box<[u8]>,
    pub dsp_ram: Box<[u8]>,
    /// AXI WRAM.
    pub axi_wram: Box<[u8]>,

    /// one bit per FCRAM page, set when allocated.
    allocated: Vec<bool>,
    /// region boundaries as page indices into allocated.
    regions: [(u32, u32); 3],
}

impl PhysicalMemory {
    /// app_bytes comes from the exheader's system mode, the rest of FCRAM is
    /// split between the SYSTEM and BASE regions the way the retail kernel
    /// does it.
    pub fn new(new3ds: bool, app_bytes: u32) -> PhysicalMemory {
        let total = fcram_size(new3ds);
        let app = app_bytes.min(total);
        // retail Old3DS gives SYSTEM 32 MiB and BASE whatever is left, the
        // New3DS extra 128 MiB all goes to the application region when the
        // title asks for it.
        let system = (total - app).min(32 * 1024 * 1024);
        let base = total - app - system;

        let pages = (total / PAGE_SIZE) as usize;
        let app_pages = app / PAGE_SIZE;
        let system_pages = system / PAGE_SIZE;
        let base_pages = base / PAGE_SIZE;

        PhysicalMemory {
            fcram: vec![0; total as usize].into_boxed_slice(),
            vram: vec![0; VRAM_SIZE as usize].into_boxed_slice(),
            dsp_ram: vec![0; DSP_RAM_SIZE as usize].into_boxed_slice(),
            axi_wram: vec![0; AXI_WRAM_SIZE as usize].into_boxed_slice(),
            allocated: vec![false; pages],
            regions: [
                (0, app_pages),
                (app_pages, system_pages),
                (app_pages + system_pages, base_pages),
            ],
        }
    }

    /// the kernel's read-only configuration page, at the start of AXI WRAM.
    pub fn config_mem_mut(&mut self) -> &mut [u8] {
        let offset = (CONFIG_MEM_VADDR - AXI_WRAM_PADDR) as usize;
        &mut self.axi_wram[offset..offset + CONFIG_MEM_SIZE as usize]
    }

    /// the page the kernel keeps updating with time, battery and wifi state.
    pub fn shared_page_mut(&mut self) -> &mut [u8] {
        let offset = (SHARED_PAGE_VADDR - AXI_WRAM_PADDR) as usize;
        &mut self.axi_wram[offset..offset + SHARED_PAGE_SIZE as usize]
    }

    fn region_pages(&self, region: MemoryRegion) -> (u32, u32) {
        self.regions[match region {
            MemoryRegion::Application => 0,
            MemoryRegion::System => 1,
            MemoryRegion::Base => 2,
        }]
    }

    pub fn region_size(&self, region: MemoryRegion) -> u32 {
        self.region_pages(region).1 * PAGE_SIZE
    }

    pub fn region_used(&self, region: MemoryRegion) -> u32 {
        let (start, count) = self.region_pages(region);
        let used = self.allocated[start as usize..(start + count) as usize]
            .iter()
            .filter(|&&a| a)
            .count();
        used as u32 * PAGE_SIZE
    }

    pub fn region_free(&self, region: MemoryRegion) -> u32 {
        self.region_size(region) - self.region_used(region)
    }

    /// allocates size bytes of physically contiguous FCRAM.
    pub fn allocate(&mut self, region: MemoryRegion, size: u32) -> Option<PhysicalBlock> {
        let needed = (align_up(size, PAGE_SIZE) / PAGE_SIZE) as usize;
        if needed == 0 {
            return Some(PhysicalBlock {
                addr: FCRAM_PADDR,
                size: 0,
            });
        }

        let (start, count) = self.region_pages(region);
        let (start, count) = (start as usize, count as usize);

        let mut run = 0usize;
        for i in start..start + count {
            if self.allocated[i] {
                run = 0;
                continue;
            }
            run += 1;
            if run == needed {
                let first = i + 1 - needed;
                for page in &mut self.allocated[first..first + needed] {
                    *page = true;
                }
                return Some(PhysicalBlock {
                    addr: FCRAM_PADDR + (first as u32 * PAGE_SIZE),
                    size: needed as u32 * PAGE_SIZE,
                });
            }
        }
        None
    }

    pub fn free(&mut self, block: PhysicalBlock) {
        let first = ((block.addr - FCRAM_PADDR) / PAGE_SIZE) as usize;
        let count = (block.size / PAGE_SIZE) as usize;
        let end = (first + count).min(self.allocated.len());
        for page in &mut self.allocated[first..end] {
            *page = false;
        }
    }

    /// host slice backing a physical address range, if it is real memory.
    pub fn host_slice_mut(&mut self, paddr: u32, size: u32) -> Option<&mut [u8]> {
        let (base, store): (u32, &mut Box<[u8]>) = match paddr {
            p if (VRAM_PADDR..VRAM_PADDR + VRAM_SIZE).contains(&p) => (VRAM_PADDR, &mut self.vram),
            p if (DSP_RAM_PADDR..DSP_RAM_PADDR + DSP_RAM_SIZE).contains(&p) => {
                (DSP_RAM_PADDR, &mut self.dsp_ram)
            }
            p if (AXI_WRAM_PADDR..AXI_WRAM_PADDR + AXI_WRAM_SIZE).contains(&p) => {
                (AXI_WRAM_PADDR, &mut self.axi_wram)
            }
            p if p >= FCRAM_PADDR => (FCRAM_PADDR, &mut self.fcram),
            _ => return None,
        };
        let offset = (paddr - base) as usize;
        let end = offset.checked_add(size as usize)?;
        store.get_mut(offset..end)
    }
}

#[inline]
const fn align_up(value: u32, align: u32) -> u32 {
    (value + align - 1) & !(align - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_contiguously_and_frees() {
        let mut phys = PhysicalMemory::new(false, 64 * 1024 * 1024);
        let a = phys.allocate(MemoryRegion::Application, 0x4000).unwrap();
        let b = phys.allocate(MemoryRegion::Application, 0x1000).unwrap();
        assert_eq!(a.addr, FCRAM_PADDR);
        assert_eq!(a.size, 0x4000);
        assert_eq!(b.addr, FCRAM_PADDR + 0x4000);

        assert_eq!(phys.region_used(MemoryRegion::Application), 0x5000);
        phys.free(a);
        assert_eq!(phys.region_used(MemoryRegion::Application), 0x1000);

        // the freed hole is reused.
        let c = phys.allocate(MemoryRegion::Application, 0x2000).unwrap();
        assert_eq!(c.addr, FCRAM_PADDR);
    }

    #[test]
    fn refuses_to_overcommit() {
        let mut phys = PhysicalMemory::new(false, 0x4000);
        assert!(phys.allocate(MemoryRegion::Application, 0x4000).is_some());
        assert!(phys.allocate(MemoryRegion::Application, 0x1000).is_none());
    }

    #[test]
    fn regions_do_not_overlap() {
        let phys = PhysicalMemory::new(false, 64 * 1024 * 1024);
        assert_eq!(phys.region_size(MemoryRegion::Application), 64 << 20);
        assert_eq!(phys.region_size(MemoryRegion::System), 32 << 20);
        assert_eq!(phys.region_size(MemoryRegion::Base), 32 << 20);
    }
}
