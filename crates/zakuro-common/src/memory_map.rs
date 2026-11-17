//! the 3DS address space, physical and as a userland process sees it

use crate::VAddr;

// physical memory

/// ARM9 internal RAM
pub const ARM9_RAM_PADDR: u32 = 0x0800_0000;
pub const ARM9_RAM_SIZE: u32 = 0x0010_0000;

pub const IO_PADDR: u32 = 0x1000_0000;
pub const IO_SIZE: u32 = 0x0800_0000;

pub const VRAM_PADDR: u32 = 0x1800_0000;
pub const VRAM_SIZE: u32 = 0x0060_0000; // 6 MiB, two 3 MiB banks

pub const DSP_RAM_PADDR: u32 = 0x1FF0_0000;
pub const DSP_RAM_SIZE: u32 = 0x0008_0000;

pub const AXI_WRAM_PADDR: u32 = 0x1FF8_0000;
pub const AXI_WRAM_SIZE: u32 = 0x0008_0000;

pub const FCRAM_PADDR: u32 = 0x2000_0000;
pub const FCRAM_SIZE_OLD3DS: u32 = 0x0800_0000; // 128 MiB
pub const FCRAM_SIZE_NEW3DS: u32 = 0x1000_0000; // 256 MiB

// process virtual memory

/// where the ExeFS .code image gets mapped
pub const PROCESS_IMAGE_VADDR: VAddr = 0x0010_0000;
pub const PROCESS_IMAGE_MAX_SIZE: u32 = 0x03F0_0000;

/// base of the region svcControlMemory(HEAP) carves out of
pub const HEAP_VADDR: VAddr = 0x0800_0000;
pub const HEAP_SIZE: u32 = 0x0800_0000;

/// shared memory blocks services hand out (GSP, HID and co) land here
pub const SHARED_MEMORY_VADDR: VAddr = 0x1000_0000;
pub const SHARED_MEMORY_SIZE: u32 = 0x0400_0000;

/// the linear heap maps to FCRAM at a constant offset so userland can compute physical addrs for the GPU w/o a syscall
/// old and new 3DS put it in different places
pub const LINEAR_HEAP_VADDR_OLD3DS: VAddr = 0x1400_0000;
pub const LINEAR_HEAP_SIZE_OLD3DS: u32 = 0x0800_0000;
pub const LINEAR_HEAP_VADDR_NEW3DS: VAddr = 0x3000_0000;
pub const LINEAR_HEAP_SIZE_NEW3DS: u32 = 0x1000_0000;

/// VRAM as mapped into the process (for games drawing straight to VRAM)
pub const VRAM_VADDR: VAddr = 0x1F00_0000;

pub const DSP_RAM_VADDR: VAddr = 0x1FF0_0000;

/// readonly page the kernel fills w/ hw and firmware info
pub const CONFIG_MEM_VADDR: VAddr = 0x1FF8_0000;
pub const CONFIG_MEM_SIZE: u32 = 0x1000;

/// readonly page the kernel keeps updating (time, wifi level, battery and stuff)
pub const SHARED_PAGE_VADDR: VAddr = 0x1FF8_1000;
pub const SHARED_PAGE_SIZE: u32 = 0x1000;

/// per thread local storage, thread N gets TLS_AREA_VADDR + N * TLS_ENTRY_SIZE
pub const TLS_AREA_VADDR: VAddr = 0x1FF8_2000;
pub const TLS_ENTRY_SIZE: u32 = 0x200;
/// offset of the IPC command buffer in a thread's TLS page
pub const TLS_IPC_COMMAND_BUFFER: u32 = 0x80;
/// offset of the IPC static buffer descriptors in a thread's TLS page
pub const TLS_IPC_STATIC_BUFFERS: u32 = 0x180;

/// page size for the MMU and every memory syscall
pub const PAGE_SIZE: u32 = 0x1000;
pub const PAGE_BITS: u32 = 12;
pub const PAGE_MASK: u32 = PAGE_SIZE - 1;
/// entries in a flat 32 bit page table
pub const PAGE_TABLE_ENTRIES: usize = 1 << (32 - PAGE_BITS);

/// firmware 8.0's kernel version, titles made for it or later get the linear heap at 0x30000000 on either console
pub const NEW_LINEAR_HEAP_KERNEL: u16 = 0x22C;

/// where the linear heap starts for a title, goes by the kernel it was made for (exheader), not the console
pub const fn linear_heap_base(kernel_version: u16) -> VAddr {
    if kernel_version >= NEW_LINEAR_HEAP_KERNEL {
        LINEAR_HEAP_VADDR_NEW3DS
    } else {
        LINEAR_HEAP_VADDR_OLD3DS
    }
}

/// FCRAM size for the console model
pub const fn fcram_size(new3ds: bool) -> u32 {
    if new3ds {
        FCRAM_SIZE_NEW3DS
    } else {
        FCRAM_SIZE_OLD3DS
    }
}
