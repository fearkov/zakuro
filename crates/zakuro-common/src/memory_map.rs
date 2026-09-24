//! the 3DS address space, both physical and as seen by a userland process.

use crate::VAddr;

// ---------------------------------------------------------------------------
// Physical memory
// ---------------------------------------------------------------------------

/// ARM9 internal RAM.
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

// ---------------------------------------------------------------------------
// Process virtual memory
// ---------------------------------------------------------------------------

/// where the ExeFS .code image is mapped.
pub const PROCESS_IMAGE_VADDR: VAddr = 0x0010_0000;
pub const PROCESS_IMAGE_MAX_SIZE: u32 = 0x03F0_0000;

/// base of the region svcControlMemory(HEAP) carves out of.
pub const HEAP_VADDR: VAddr = 0x0800_0000;
pub const HEAP_SIZE: u32 = 0x0800_0000;

/// shared memory blocks handed out by services (GSP, HID, ...) land here.
pub const SHARED_MEMORY_VADDR: VAddr = 0x1000_0000;
pub const SHARED_MEMORY_SIZE: u32 = 0x0400_0000;

/// the linear heap is the region whose virtual addresses map to FCRAM with a
/// constant offset, so that userland can compute physical addresses for the
/// GPU without a syscall. Old3DS and New3DS put it in different places.
pub const LINEAR_HEAP_VADDR_OLD3DS: VAddr = 0x1400_0000;
pub const LINEAR_HEAP_SIZE_OLD3DS: u32 = 0x0800_0000;
pub const LINEAR_HEAP_VADDR_NEW3DS: VAddr = 0x3000_0000;
pub const LINEAR_HEAP_SIZE_NEW3DS: u32 = 0x1000_0000;

/// VRAM as mapped into the process (used by games that render straight to VRAM).
pub const VRAM_VADDR: VAddr = 0x1F00_0000;

pub const DSP_RAM_VADDR: VAddr = 0x1FF0_0000;

/// read-only page the kernel exposes with hardware/firmware info.
pub const CONFIG_MEM_VADDR: VAddr = 0x1FF8_0000;
pub const CONFIG_MEM_SIZE: u32 = 0x1000;

/// read-only page updated by the kernel (time, wifi level, battery, ...).
pub const SHARED_PAGE_VADDR: VAddr = 0x1FF8_1000;
pub const SHARED_PAGE_SIZE: u32 = 0x1000;

/// per-thread local storage. Thread N gets TLS_AREA_VADDR + N * TLS_ENTRY_SIZE.
pub const TLS_AREA_VADDR: VAddr = 0x1FF8_2000;
pub const TLS_ENTRY_SIZE: u32 = 0x200;
/// offset of the IPC command buffer inside a thread's TLS page.
pub const TLS_IPC_COMMAND_BUFFER: u32 = 0x80;
/// offset of the IPC static buffer descriptors inside a thread's TLS page.
pub const TLS_IPC_STATIC_BUFFERS: u32 = 0x180;

/// page size used by the MMU and by every memory syscall.
pub const PAGE_SIZE: u32 = 0x1000;
pub const PAGE_BITS: u32 = 12;
pub const PAGE_MASK: u32 = PAGE_SIZE - 1;
/// number of entries in a flat 32-bit page table.
pub const PAGE_TABLE_ENTRIES: usize = 1 << (32 - PAGE_BITS);

/// returns the linear-heap base for the given console model.
pub const fn linear_heap_base(new3ds: bool) -> VAddr {
    if new3ds {
        LINEAR_HEAP_VADDR_NEW3DS
    } else {
        LINEAR_HEAP_VADDR_OLD3DS
    }
}

/// returns the FCRAM size for the given console model.
pub const fn fcram_size(new3ds: bool) -> u32 {
    if new3ds {
        FCRAM_SIZE_NEW3DS
    } else {
        FCRAM_SIZE_OLD3DS
    }
}
