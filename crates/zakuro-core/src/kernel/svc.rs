//! supervisor call dispatch.

use zakuro_common::memory_map::*;
use zakuro_common::result::errors;
use zakuro_common::VAddr;
use zakuro_cpu::Bus;

use super::object::{KObject, ObjectId, CURRENT_PROCESS, CURRENT_THREAD};
use super::sync::{self, ArbitrationType, ResetType};
use super::thread::{nanos_to_ticks, ThreadStatus, WaitResult, WaitSyscall, THREAD_EXIT_MAGIC};
use crate::memory::{MemoryState, Permission};
use crate::System;

// MemOp, as libctru defines it.
const MEMOP_FREE: u32 = 1;
const MEMOP_RESERVE: u32 = 2;
const MEMOP_ALLOC: u32 = 3;
const MEMOP_MAP: u32 = 4;
const MEMOP_UNMAP: u32 = 5;
const MEMOP_PROTECT: u32 = 6;
const MEMOP_OP_MASK: u32 = 0xFF;
const MEMOP_LINEAR_FLAG: u32 = 0x1_0000;

/// reports a handle a syscall could not resolve.
fn invalid_handle(system: &mut System, handle: u32, what: &str) -> u32 {
    log::warn!(
        "{what}: handle 0x{handle:X} does not exist (called from 0x{:08X})",
        system.cpu.current_pc()
    );
    errors::INVALID_HANDLE.0
}

/// runs one supervisor call.
pub fn dispatch(system: &mut System, number: u32) {
    if log::log_enabled!(log::Level::Trace) {
        let thread = system
            .kernel
            .current()
            .map_or("?".to_owned(), |t| t.name.clone());
        log::trace!(
            "[{thread}] svc 0x{number:02X} r0={:08X} r1={:08X} r2={:08X} r3={:08X} @0x{:08X}",
            system.cpu.regs[0],
            system.cpu.regs[1],
            system.cpu.regs[2],
            system.cpu.regs[3],
            system.cpu.current_pc(),
        );
    }
    match number {
        0x01 => control_memory(system),
        0x02 => query_memory(system),
        0x03 => exit_process(system),
        0x08 => create_thread(system),
        0x09 => exit_thread(system),
        0x0A => sleep_thread(system),
        0x0B => get_thread_priority(system),
        0x0C => set_thread_priority(system),
        0x11 => system.cpu.regs[0] = 0, // GetCurrentProcessorNumber
        0x13 => create_mutex(system),
        0x14 => release_mutex(system),
        0x15 => create_semaphore(system),
        0x16 => release_semaphore(system),
        0x17 => create_event(system),
        0x18 => signal_event(system),
        0x19 => clear_event(system),
        0x1A => create_timer(system),
        0x1B => set_timer(system),
        0x1C => cancel_timer(system),
        0x1D => clear_timer(system),
        0x1E => create_memory_block(system),
        0x1F => map_memory_block(system),
        0x20 => unmap_memory_block(system),
        0x21 => create_address_arbiter(system),
        0x22 => arbitrate_address(system),
        0x23 => close_handle(system),
        0x24 => wait_synchronization1(system),
        0x25 => wait_synchronization_n(system),
        0x27 => duplicate_handle(system),
        0x28 => get_system_tick(system),
        0x2A => get_system_info(system),
        0x2B => get_process_info(system),
        0x2D => connect_to_port(system),
        0x32 => send_sync_request(system),
        0x35 => get_process_id(system),
        0x37 => get_thread_id(system),
        0x38 => get_resource_limit(system),
        0x39 => get_resource_limit_limit_values(system),
        0x3A => get_resource_limit_current_values(system),
        0x3C => svc_break(system),
        0x3D => output_debug_string(system),
        // cache maintenance.
        0x54..=0x56 => {
            system.cpu.regs[0] = 0;
        }
        _ => {
            log::warn!(
                "unimplemented svc 0x{number:02X} at 0x{:08X} (r0={:08X} r1={:08X} r2={:08X} r3={:08X})",
                system.cpu.current_pc(),
                system.cpu.regs[0],
                system.cpu.regs[1],
                system.cpu.regs[2],
                system.cpu.regs[3],
            );
            system.unimplemented_svcs.insert(number);
            system.cpu.regs[0] = errors::UNIMPLEMENTED.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

/// svcControlMemory, r0 = operation, r1 = addr0, r2 = addr1, r3 = size,
/// r4 = permission. Returns the mapped address in r1.
fn control_memory(system: &mut System) {
    let operation = system.cpu.regs[0];
    let addr0 = system.cpu.regs[1];
    let addr1 = system.cpu.regs[2];
    let size = system.cpu.regs[3];
    let permission = system.cpu.regs[4];

    let op = operation & MEMOP_OP_MASK;
    let linear = operation & MEMOP_LINEAR_FLAG != 0;

    if size & PAGE_MASK != 0 {
        log::warn!("svcControlMemory with unaligned size 0x{size:X}");
        system.cpu.regs[0] = errors::MISALIGNED_SIZE.0;
        return;
    }

    let perm = Permission::from_bits_truncate(permission & 0x7);

    match op {
        MEMOP_ALLOC if linear => {
            let region = system.kernel.memory_region;
            let Some(block) = system.memory.phys.allocate(region, size) else {
                log::error!("out of linear memory allocating 0x{size:X}");
                system.cpu.regs[0] = errors::OUT_OF_MEMORY.0;
                return;
            };
            // the whole point of the linear heap is that the guest can derive
            // a physical address from a virtual one, so the mapping offset
            // must match exactly.
            let vaddr = system.kernel.linear_base + (block.addr - FCRAM_PADDR);
            system.memory.map(
                vaddr,
                block.addr,
                block.size,
                perm | Permission::READ,
                MemoryState::Continuous,
            );
            system.kernel.linear_top = system.kernel.linear_top.max(vaddr + block.size);
            log::debug!(
                "linear alloc 0x{size:X} -> va 0x{vaddr:08X} pa 0x{:08X}",
                block.addr
            );
            system.cpu.regs[0] = 0;
            system.cpu.regs[1] = vaddr;
        }
        MEMOP_ALLOC => {
            let base = if addr0 == 0 {
                system.kernel.heap_top
            } else {
                addr0
            };
            let region = system.kernel.memory_region;
            let Some(block) = system.memory.phys.allocate(region, size) else {
                log::error!("out of heap memory allocating 0x{size:X}");
                system.cpu.regs[0] = errors::OUT_OF_MEMORY.0;
                return;
            };
            system.memory.map(
                base,
                block.addr,
                block.size,
                perm | Permission::READ,
                MemoryState::Private,
            );
            system.kernel.heap_top = system.kernel.heap_top.max(base + block.size);
            log::debug!("heap alloc 0x{size:X} at 0x{base:08X}");
            system.cpu.regs[0] = 0;
            system.cpu.regs[1] = base;
        }
        MEMOP_FREE => {
            if let Some(mapping) = system.memory.mapping_at(addr0).copied() {
                system.memory.phys.free(crate::memory::PhysicalBlock {
                    addr: mapping.paddr,
                    size: size.min(mapping.size),
                });
            }
            system.memory.unmap(addr0, size);
            system.cpu.regs[0] = 0;
            system.cpu.regs[1] = addr0;
        }
        MEMOP_MAP => {
            // mirror the pages backing addr1 at addr0.
            if let Some(mapping) = system.memory.mapping_at(addr1).copied() {
                let offset = addr1 - mapping.base;
                system.memory.map(
                    addr0,
                    mapping.paddr + offset,
                    size,
                    perm | Permission::RW,
                    MemoryState::Aliased,
                );
                system.cpu.regs[0] = 0;
            } else {
                log::warn!("svcControlMemory MAP of unmapped source 0x{addr1:08X}");
                system.cpu.regs[0] = errors::INVALID_ADDRESS.0;
            }
            system.cpu.regs[1] = addr0;
        }
        MEMOP_UNMAP => {
            system.memory.unmap(addr0, size);
            system.cpu.regs[0] = 0;
            system.cpu.regs[1] = addr0;
        }
        MEMOP_PROTECT | MEMOP_RESERVE => {
            // nothing enforces permissions yet, so these are accepted as-is.
            system.cpu.regs[0] = 0;
            system.cpu.regs[1] = addr0;
        }
        _ => {
            log::warn!("svcControlMemory with unknown operation 0x{operation:X}");
            system.cpu.regs[0] = errors::INVALID_ENUM_VALUE.0;
        }
    }
}

/// svcQueryMemory, r2 = address.
fn query_memory(system: &mut System) {
    let addr = system.cpu.regs[2];
    let info = system.memory.query(addr);
    system.cpu.regs[0] = 0;
    system.cpu.regs[1] = info.base;
    system.cpu.regs[2] = info.size;
    system.cpu.regs[3] = info.permission;
    system.cpu.regs[4] = info.state;
    system.cpu.regs[5] = 0;
}

/// svcCreateMemoryBlock, r0 = other permission, r1 = address, r2 = size,
/// r3 = own permission. Returns the handle in r1.
fn create_memory_block(system: &mut System) {
    let addr = system.cpu.regs[1];
    let size = system.cpu.regs[2];

    // an address of zero asks the kernel to take the pages from the BASE
    // region instead of the caller's address space.
    let paddr = if addr == 0 {
        match system
            .memory
            .phys
            .allocate(crate::memory::MemoryRegion::Base, size)
        {
            Some(block) => block.addr,
            None => {
                system.cpu.regs[0] = errors::OUT_OF_MEMORY.0;
                return;
            }
        }
    } else {
        match system.memory.mapping_at(addr) {
            Some(mapping) => mapping.paddr + (addr - mapping.base),
            None => {
                system.cpu.regs[0] = errors::INVALID_ADDRESS.0;
                return;
            }
        }
    };

    let object = system
        .kernel
        .objects
        .insert(KObject::SharedMemory(super::object::SharedMemory {
            name: format!("MemoryBlock@{addr:08X}"),
            address: addr,
            size,
            paddr,
            mapped_at: None,
        }));
    let handle = system
        .kernel
        .handles
        .create(&mut system.kernel.objects, object, "MemoryBlock");
    system.cpu.regs[0] = 0;
    system.cpu.regs[1] = handle;
}

/// svcMapMemoryBlock, r0 = handle, r1 = address, r2 = own permission,
/// r3 = other permission.
fn map_memory_block(system: &mut System) {
    let handle = system.cpu.regs[0];
    let addr = system.cpu.regs[1];
    let permission = system.cpu.regs[2];

    let Some(object) = system.kernel.resolve(handle) else {
        system.cpu.regs[0] = invalid_handle(system, handle, "svcMapMemoryBlock");
        return;
    };
    let Some(KObject::SharedMemory(block)) = system.kernel.objects.get(object) else {
        system.cpu.regs[0] = invalid_handle(system, handle, "svcMapMemoryBlock");
        return;
    };

    let (paddr, size) = (block.paddr, block.size);
    let addr = if addr == 0 {
        // let the kernel choose, stack the block after whatever is already in
        // the shared memory region.
        let base = SHARED_MEMORY_VADDR;
        let mut candidate = base;
        while system.memory.mapping_at(candidate).is_some() {
            candidate += PAGE_SIZE;
        }
        candidate
    } else {
        addr
    };

    let perm = Permission::from_bits_truncate(permission & 0x7) | Permission::READ;
    system
        .memory
        .map(addr, paddr, size, perm, MemoryState::Shared);

    let name = if let Some(KObject::SharedMemory(block)) = system.kernel.objects.get_mut(object) {
        block.mapped_at = Some(addr);
        block.name.clone()
    } else {
        String::new()
    };

    // the services that publish a shared block need to know where the guest
    // put it, because that is where they write their state each frame.
    match name.as_str() {
        "GSP" => system.services.gsp.shared_memory_address = addr,
        "HID" => {
            system.services.hid.shared_memory_address = addr;
            system.services.hid.shared_memory_paddr = paddr;
        }
        // a font's internal pointers are absolute, so they only mean
        // anything once the guest has chosen where the block lives.
        "SharedFont" => crate::services::shared_font::relocate(&mut system.memory, addr, paddr),
        _ => {}
    }

    log::debug!("mapped {name} block handle 0x{handle:X} at 0x{addr:08X} (0x{size:X} bytes)");
    system.cpu.regs[0] = 0;
}

fn unmap_memory_block(system: &mut System) {
    let handle = system.cpu.regs[0];
    let addr = system.cpu.regs[1];
    if let Some(object) = system.kernel.resolve(handle) {
        if let Some(KObject::SharedMemory(block)) = system.kernel.objects.get(object) {
            let size = block.size;
            system.memory.unmap(addr, size);
        }
    }
    system.cpu.regs[0] = 0;
}

// ---------------------------------------------------------------------------
// Threads
// ---------------------------------------------------------------------------

/// svcCreateThread, r0 = priority, r1 = entry, r2 = argument,
/// r3 = stack top, r4 = processor id. Returns the handle in r1.
fn create_thread(system: &mut System) {
    let priority = system.cpu.regs[0];
    let entry = system.cpu.regs[1];
    let arg = system.cpu.regs[2];
    let stack_top = system.cpu.regs[3];
    let processor_id = system.cpu.regs[4] as i32;

    let id = system.kernel.create_thread(
        format!("thread{}", system.kernel.threads.len()),
        entry,
        stack_top,
        arg,
        priority,
        processor_id,
    );
    // the new thread's TLS page has to exist before it runs.
    system.map_tls_page(id);

    let handle = system.kernel.thread_handle(id);
    system.cpu.regs[0] = 0;
    system.cpu.regs[1] = handle;
}

fn exit_thread(system: &mut System) {
    if let Some(thread) = system.kernel.current_mut() {
        log::debug!("thread {} exited", thread.name);
        thread.status = ThreadStatus::Dead;
        thread.clear_wait();
    }
    system.kernel.current_thread = None;
    system.kernel.reschedule_pending = true;
}

fn exit_process(system: &mut System) {
    log::info!("guest called svcExitProcess");
    for thread in &mut system.kernel.threads {
        thread.status = ThreadStatus::Dead;
    }
    system.kernel.current_thread = None;
    system.kernel.reschedule_pending = true;
    system.exited = true;
}

/// svcSleepThread, r0:r1 = nanoseconds as a signed 64-bit value.
fn sleep_thread(system: &mut System) {
    let nanos = (system.cpu.regs[0] as u64) | ((system.cpu.regs[1] as u64) << 32);
    let tick = system.cpu.cycles;
    // a zero sleep is a yield.
    let ticks = if nanos == 0 { 0 } else { nanos_to_ticks(nanos) };
    system.kernel.sleep_current(ticks, tick);
    system.cpu.regs[0] = 0;
}

fn get_thread_priority(system: &mut System) {
    let handle = system.cpu.regs[1];
    match resolve_thread(system, handle) {
        Some(id) => {
            system.cpu.regs[0] = 0;
            system.cpu.regs[1] = system.kernel.thread(id).priority;
        }
        None => system.cpu.regs[0] = invalid_handle(system, handle, "svcGetThreadPriority"),
    }
}

fn set_thread_priority(system: &mut System) {
    let handle = system.cpu.regs[0];
    let priority = system.cpu.regs[1];
    match resolve_thread(system, handle) {
        Some(id) => {
            system.kernel.thread_mut(id).priority = priority.min(super::LOWEST_PRIORITY);
            system.kernel.reschedule_pending = true;
            system.cpu.regs[0] = 0;
        }
        None => system.cpu.regs[0] = invalid_handle(system, handle, "svcSetThreadPriority"),
    }
}

fn get_thread_id(system: &mut System) {
    let handle = system.cpu.regs[1];
    match resolve_thread(system, handle) {
        Some(id) => {
            system.cpu.regs[0] = 0;
            system.cpu.regs[1] = system.kernel.thread(id).guest_id;
        }
        None => system.cpu.regs[0] = invalid_handle(system, handle, "svcGetThreadId"),
    }
}

fn resolve_thread(system: &mut System, handle: u32) -> Option<super::thread::ThreadId> {
    if handle == CURRENT_THREAD {
        return system.kernel.current_thread;
    }
    match system.kernel.objects.get(system.kernel.handles.resolve(handle)?) {
        Some(KObject::Thread(id)) => Some(*id),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Synchronisation
// ---------------------------------------------------------------------------

/// svcCreateMutex, r1 = initially locked. Returns the handle in r1.
fn create_mutex(system: &mut System) {
    let initially_locked = system.cpu.regs[1] != 0;
    let mut mutex = sync::Mutex::new("Mutex");
    if initially_locked {
        mutex.owner = system.kernel.current_thread;
        mutex.lock_count = 1;
    }
    let object = system.kernel.objects.insert(KObject::Mutex(mutex));
    let handle = system
        .kernel
        .handles
        .create(&mut system.kernel.objects, object, "Mutex");
    system.cpu.regs[0] = 0;
    system.cpu.regs[1] = handle;
}

fn release_mutex(system: &mut System) {
    let handle = system.cpu.regs[0];
    let Some(object) = system.kernel.resolve(handle) else {
        system.cpu.regs[0] = invalid_handle(system, handle, "svcReleaseMutex");
        return;
    };
    if let Some(KObject::Mutex(mutex)) = system.kernel.objects.get_mut(object) {
        mutex.lock_count = mutex.lock_count.saturating_sub(1);
        if mutex.lock_count == 0 {
            mutex.owner = None;
            system.kernel.reschedule_pending = true;
        }
        system.cpu.regs[0] = 0;
    } else {
        system.cpu.regs[0] = invalid_handle(system, handle, "svcReleaseMutex");
    }
}

/// svcCreateSemaphore, r1 = initial count, r2 = maximum count.
fn create_semaphore(system: &mut System) {
    let initial = system.cpu.regs[1] as i32;
    let max = system.cpu.regs[2] as i32;
    let object = system
        .kernel
        .objects
        .insert(KObject::Semaphore(sync::Semaphore {
            name: "Semaphore".into(),
            count: initial,
            max_count: max,
        }));
    let handle = system
        .kernel
        .handles
        .create(&mut system.kernel.objects, object, "Semaphore");
    system.cpu.regs[0] = 0;
    system.cpu.regs[1] = handle;
}

/// svcReleaseSemaphore, r1 = handle, r2 = release count.
fn release_semaphore(system: &mut System) {
    let handle = system.cpu.regs[1];
    let count = system.cpu.regs[2] as i32;
    let Some(object) = system.kernel.resolve(handle) else {
        system.cpu.regs[0] = invalid_handle(system, handle, "svcReleaseSemaphore");
        return;
    };
    if let Some(KObject::Semaphore(semaphore)) = system.kernel.objects.get_mut(object) {
        let previous = semaphore.count;
        semaphore.count = (semaphore.count + count).min(semaphore.max_count);
        system.kernel.reschedule_pending = true;
        system.cpu.regs[0] = 0;
        system.cpu.regs[1] = previous as u32;
    } else {
        system.cpu.regs[0] = invalid_handle(system, handle, "svcReleaseSemaphore");
    }
}

/// svcCreateEvent, r1 = reset type. Returns the handle in r1.
fn create_event(system: &mut System) {
    let reset_type = ResetType::from_raw(system.cpu.regs[1]);
    let (_, handle) = system.kernel.create_event(reset_type, "Event");
    system.cpu.regs[0] = 0;
    system.cpu.regs[1] = handle;
}

fn signal_event(system: &mut System) {
    let handle = system.cpu.regs[0];
    match system.kernel.resolve(handle) {
        Some(object) => {
            system.kernel.signal_event(object);
            system.cpu.regs[0] = 0;
        }
        None => system.cpu.regs[0] = invalid_handle(system, handle, "svcSignalEvent"),
    }
}

fn clear_event(system: &mut System) {
    let handle = system.cpu.regs[0];
    match system.kernel.resolve(handle) {
        Some(object) => {
            system.kernel.clear_event(object);
            system.cpu.regs[0] = 0;
        }
        None => system.cpu.regs[0] = invalid_handle(system, handle, "svcClearEvent"),
    }
}

/// svcCreateTimer, r1 = reset type.
fn create_timer(system: &mut System) {
    let reset_type = ResetType::from_raw(system.cpu.regs[1]);
    let object = system.kernel.objects.insert(KObject::Timer(sync::Timer {
        name: "Timer".into(),
        reset_type,
        signaled: false,
        fire_at: None,
        interval: 0,
    }));
    let handle = system
        .kernel
        .handles
        .create(&mut system.kernel.objects, object, "Timer");
    system.cpu.regs[0] = 0;
    system.cpu.regs[1] = handle;
}

/// svcSetTimer, r0 = handle, r2:r3 = initial delay, r1:r4 = interval, both
/// in nanoseconds.
fn set_timer(system: &mut System) {
    let handle = system.cpu.regs[0];
    let initial = (system.cpu.regs[2] as u64) | ((system.cpu.regs[3] as u64) << 32);
    let interval = (system.cpu.regs[1] as u64) | ((system.cpu.regs[4] as u64) << 32);
    let now = system.cpu.cycles;

    let Some(object) = system.kernel.resolve(handle) else {
        system.cpu.regs[0] = invalid_handle(system, handle, "svcSetTimer");
        return;
    };
    if let Some(KObject::Timer(timer)) = system.kernel.objects.get_mut(object) {
        timer.fire_at = Some(now + nanos_to_ticks(initial));
        timer.interval = nanos_to_ticks(interval);
        timer.signaled = false;
        system.cpu.regs[0] = 0;
    } else {
        system.cpu.regs[0] = invalid_handle(system, handle, "svcSetTimer");
    }
}

fn cancel_timer(system: &mut System) {
    let handle = system.cpu.regs[0];
    if let Some(object) = system.kernel.resolve(handle) {
        if let Some(KObject::Timer(timer)) = system.kernel.objects.get_mut(object) {
            timer.fire_at = None;
        }
    }
    system.cpu.regs[0] = 0;
}

fn clear_timer(system: &mut System) {
    let handle = system.cpu.regs[0];
    if let Some(object) = system.kernel.resolve(handle) {
        if let Some(KObject::Timer(timer)) = system.kernel.objects.get_mut(object) {
            timer.signaled = false;
        }
    }
    system.cpu.regs[0] = 0;
}

fn create_address_arbiter(system: &mut System) {
    let object = system
        .kernel
        .objects
        .insert(KObject::AddressArbiter(sync::AddressArbiter::default()));
    let handle = system
        .kernel
        .handles
        .create(&mut system.kernel.objects, object, "AddressArbiter");
    system.cpu.regs[0] = 0;
    system.cpu.regs[1] = handle;
}

/// svcArbitrateAddress, r0 = handle, r1 = address, r2 = type, r3 = value,
/// r4 = timeout in nanoseconds.
fn arbitrate_address(system: &mut System) {
    let handle = system.cpu.regs[0];
    let address = system.cpu.regs[1];
    let Some(kind) = ArbitrationType::from_raw(system.cpu.regs[2]) else {
        system.cpu.regs[0] = errors::INVALID_ENUM_VALUE.0;
        return;
    };
    let value = system.cpu.regs[3] as i32;
    let timeout_nanos = system.cpu.regs[4] as u64;

    let Some(object) = system.kernel.resolve(handle) else {
        system.cpu.regs[0] = invalid_handle(system, handle, "svcArbitrateAddress");
        return;
    };

    system.cpu.regs[0] = 0;

    log::trace!(
        "svcArbitrateAddress({kind:?}) on 0x{address:08X} value {value} (memory holds {})",
        system.memory.read32(address) as i32
    );

    if kind == ArbitrationType::Signal {
        // wake up to value threads parked on this address, a negative value
        // means all of them.
        let mut woken = 0;
        let unlimited = value < 0;
        let mut to_wake = Vec::new();
        if let Some(KObject::AddressArbiter(arbiter)) = system.kernel.objects.get_mut(object) {
            arbiter.waiters.retain(|&(thread, addr)| {
                if addr == address && (unlimited || woken < value) {
                    woken += 1;
                    to_wake.push(thread);
                    false
                } else {
                    true
                }
            });
        }
        for thread in to_wake {
            let thread = system.kernel.thread_mut(thread);
            thread.clear_wait();
            thread.wait_result = Some(WaitResult::Signaled(0));
            thread.status = ThreadStatus::Ready;
        }
        system.kernel.reschedule_pending = true;
        return;
    }

    // the waiting forms compare the word at address against value and only
    // sleep when it is smaller.
    let current = system.memory.read32(address) as i32;
    if current >= value {
        return;
    }
    if kind.decrements() {
        system.memory.write32(address, (current - 1) as u32);
    }

    let Some(thread_id) = system.kernel.current_thread else {
        return;
    };
    if let Some(KObject::AddressArbiter(arbiter)) = system.kernel.objects.get_mut(object) {
        arbiter.waiters.push((thread_id, address));
    }
    let now = system.cpu.cycles;
    let thread = system.kernel.thread_mut(thread_id);
    thread.clear_wait();
    thread.wait_address = Some(address);
    thread.wakeup_at = if kind.has_timeout() {
        Some(now + nanos_to_ticks(timeout_nanos))
    } else {
        None
    };
    thread.status = ThreadStatus::WaitArbiter;
    thread.wait_syscall = Some(WaitSyscall::ArbitrateAddress);
    system.kernel.reschedule_pending = true;
}

/// svcWaitSynchronization1, r0 = handle, r2:r3 = timeout in nanoseconds.
fn wait_synchronization1(system: &mut System) {
    let handle = system.cpu.regs[0];
    let timeout = (system.cpu.regs[2] as u64) | ((system.cpu.regs[3] as u64) << 32);
    let signed_timeout = timeout as i64;

    let Some(object) = system.kernel.resolve(handle) else {
        log::warn!("svcWaitSynchronization1 on invalid handle 0x{handle:X}");
        system.cpu.regs[0] = errors::INVALID_HANDLE.0;
        return;
    };

    let Some(thread_id) = system.kernel.current_thread else {
        system.cpu.regs[0] = 0;
        return;
    };

    if system.kernel.is_signaled(object, thread_id) {
        system.kernel.acquire_public(object, thread_id);
        system.cpu.regs[0] = 0;
        return;
    }

    if signed_timeout == 0 {
        system.cpu.regs[0] = errors::TIMEOUT.0;
        return;
    }

    let tick = system.cpu.cycles;
    let ticks = (signed_timeout > 0).then(|| nanos_to_ticks(timeout));
    system.kernel.begin_wait(vec![object], false, ticks, tick);
    // the result registers are filled in when the wait completes.
    system.kernel.thread_mut(thread_id).wait_syscall = Some(WaitSyscall::WaitSynchronization1);
}

/// svcWaitSynchronizationN, r0 = timeout low, r1 = handle array, r2 = count,
/// r3 = wait-for-all, r4 = timeout high.
fn wait_synchronization_n(system: &mut System) {
    let handles_ptr = system.cpu.regs[1];
    let count = system.cpu.regs[2];
    let wait_all = system.cpu.regs[3] != 0;
    let timeout = (system.cpu.regs[0] as u64) | ((system.cpu.regs[4] as u64) << 32);
    let signed_timeout = timeout as i64;

    let mut objects = Vec::with_capacity(count as usize);
    for i in 0..count {
        let handle = system.memory.read32(handles_ptr + i * 4);
        match system.kernel.resolve(handle) {
            Some(object) => objects.push(object),
            None => {
                log::warn!("svcWaitSynchronizationN: invalid handle 0x{handle:X} at index {i}");
                system.cpu.regs[0] = errors::INVALID_HANDLE.0;
                return;
            }
        }
    }

    let Some(thread_id) = system.kernel.current_thread else {
        system.cpu.regs[0] = 0;
        return;
    };

    let states: Vec<bool> = objects
        .iter()
        .map(|&object| system.kernel.is_signaled(object, thread_id))
        .collect();
    let satisfied = if wait_all {
        states.iter().all(|&s| s)
    } else {
        states.iter().any(|&s| s)
    };

    if satisfied {
        let index = if wait_all {
            for &object in &objects {
                system.kernel.acquire_public(object, thread_id);
            }
            0
        } else {
            let index = states.iter().position(|&s| s).unwrap();
            system.kernel.acquire_public(objects[index], thread_id);
            index
        };
        system.cpu.regs[0] = 0;
        system.cpu.regs[1] = index as u32;
        return;
    }

    if signed_timeout == 0 {
        system.cpu.regs[0] = errors::TIMEOUT.0;
        return;
    }

    let tick = system.cpu.cycles;
    let ticks = (signed_timeout > 0).then(|| nanos_to_ticks(timeout));
    system.kernel.begin_wait(objects, wait_all, ticks, tick);
    system.kernel.thread_mut(thread_id).wait_syscall = Some(WaitSyscall::WaitSynchronizationN);
}

// ---------------------------------------------------------------------------
// Handles, ports and services
// ---------------------------------------------------------------------------

fn close_handle(system: &mut System) {
    let handle = system.cpu.regs[0];
    system
        .kernel
        .handles
        .close(&mut system.kernel.objects, handle);
    system.cpu.regs[0] = 0;
}

fn duplicate_handle(system: &mut System) {
    let handle = system.cpu.regs[1];
    match system.kernel.resolve(handle) {
        Some(object) => {
            let label = system.kernel.handles.label(handle).to_owned();
            let new_handle = system.kernel.handles.duplicate_object(
                &mut system.kernel.objects,
                object,
                &label,
            );
            system.cpu.regs[0] = 0;
            system.cpu.regs[1] = new_handle;
        }
        None => {
            log::warn!("svcDuplicateHandle on invalid handle 0x{handle:X}");
            system.cpu.regs[0] = errors::INVALID_HANDLE.0;
        }
    }
}

/// svcConnectToPort, r1 = pointer to the port name.
fn connect_to_port(system: &mut System) {
    let name_ptr = system.cpu.regs[1];
    let name = system.memory.read_cstring(name_ptr, 12);
    log::debug!("svcConnectToPort(\"{name}\")");

    let object = system
        .kernel
        .objects
        .insert(KObject::ClientPort(name.clone()));
    let handle = system
        .kernel
        .handles
        .create(&mut system.kernel.objects, object, &name);
    system.cpu.regs[0] = 0;
    system.cpu.regs[1] = handle;
}

/// svcSendSyncRequest, r0 = session handle.
fn send_sync_request(system: &mut System) {
    let handle = system.cpu.regs[0];
    let Some(object) = system.kernel.resolve(handle) else {
        log::warn!("svcSendSyncRequest on invalid handle 0x{handle:X}");
        system.cpu.regs[0] = errors::INVALID_HANDLE.0;
        return;
    };

    let target = match system.kernel.objects.get(object) {
        Some(KObject::ClientPort(name)) => crate::services::Target::port(name.clone()),
        Some(KObject::ClientSession(session)) => {
            crate::services::Target::service(session.service.clone(), session.subhandle)
        }
        _ => {
            log::warn!("svcSendSyncRequest on a {} handle", {
                system
                    .kernel
                    .objects
                    .get(object)
                    .map_or("missing", |o| o.type_name())
            });
            system.cpu.regs[0] = errors::INVALID_HANDLE.0;
            return;
        }
    };

    system.cpu.regs[0] = 0;
    crate::services::handle_request(system, target);
}

// ---------------------------------------------------------------------------
// Information
// ---------------------------------------------------------------------------

fn get_system_tick(system: &mut System) {
    let tick = system.cpu.cycles;
    system.cpu.regs[0] = tick as u32;
    system.cpu.regs[1] = (tick >> 32) as u32;
}

/// svcGetSystemInfo, r1 = type, r2 = parameter.
fn get_system_info(system: &mut System) {
    let kind = system.cpu.regs[1];
    let param = system.cpu.regs[2];
    let value: u64 = match (kind, param) {
        // memory usage of each FCRAM region.
        (0, 0) => system
            .memory
            .phys
            .region_used(crate::memory::MemoryRegion::Application) as u64,
        (0, 1) => system
            .memory
            .phys
            .region_used(crate::memory::MemoryRegion::System) as u64,
        (0, 2) => system
            .memory
            .phys
            .region_used(crate::memory::MemoryRegion::Base) as u64,
        (2, 0) => system.kernel.live_thread_count() as u64,
        // kernel version.
        (26, _) => 0,
        _ => {
            log::debug!("svcGetSystemInfo({kind}, {param}) is not implemented");
            0
        }
    };
    system.cpu.regs[0] = 0;
    system.cpu.regs[1] = value as u32;
    system.cpu.regs[2] = (value >> 32) as u32;
}

/// svcGetProcessInfo, r1 = process handle, r2 = type.
fn get_process_info(system: &mut System) {
    let kind = system.cpu.regs[2];
    let value: u64 = match kind {
        // total memory the process has committed.
        2 | 20 => system
            .memory
            .phys
            .region_used(system.kernel.memory_region) as u64,
        _ => 0,
    };
    system.cpu.regs[0] = 0;
    system.cpu.regs[1] = value as u32;
    system.cpu.regs[2] = (value >> 32) as u32;
}

fn get_process_id(system: &mut System) {
    let handle = system.cpu.regs[1];
    let _ = handle;
    system.cpu.regs[0] = 0;
    system.cpu.regs[1] = system.kernel.process_id;
}

fn get_resource_limit(system: &mut System) {
    let _process = system.cpu.regs[1];
    let object = system.kernel.objects.insert(KObject::ResourceLimit);
    let handle = system
        .kernel
        .handles
        .create(&mut system.kernel.objects, object, "ResourceLimit");
    system.cpu.regs[0] = 0;
    system.cpu.regs[1] = handle;
}

/// resource limit names, as svcGetResourceLimitLimitValues indexes them.
const RESOURCE_PRIORITY: u32 = 0;
const RESOURCE_COMMIT: u32 = 1;
const RESOURCE_THREAD: u32 = 2;
const RESOURCE_EVENT: u32 = 3;
const RESOURCE_MUTEX: u32 = 4;
const RESOURCE_SEMAPHORE: u32 = 5;
const RESOURCE_TIMER: u32 = 6;
const RESOURCE_SHARED_MEMORY: u32 = 7;
const RESOURCE_ADDRESS_ARBITER: u32 = 8;
const RESOURCE_CPU_TIME: u32 = 9;

fn resource_limit_values(system: &mut System, current: bool) {
    let values_ptr = system.cpu.regs[0];
    let names_ptr = system.cpu.regs[2];
    let count = system.cpu.regs[3];

    for i in 0..count {
        let name = system.memory.read32(names_ptr + i * 4);
        let value: i64 = match (name, current) {
            (RESOURCE_COMMIT, false) => {
                system.memory.phys.region_size(system.kernel.memory_region) as i64
            }
            (RESOURCE_COMMIT, true) => {
                system.memory.phys.region_used(system.kernel.memory_region) as i64
            }
            (RESOURCE_PRIORITY, _) => 0x18,
            (RESOURCE_THREAD, false) => 128,
            (RESOURCE_THREAD, true) => system.kernel.live_thread_count() as i64,
            (RESOURCE_EVENT, false) => 128,
            (RESOURCE_MUTEX, false) => 64,
            (RESOURCE_SEMAPHORE, false) => 32,
            (RESOURCE_TIMER, false) => 16,
            (RESOURCE_SHARED_MEMORY, false) => 32,
            (RESOURCE_ADDRESS_ARBITER, false) => 8,
            (RESOURCE_CPU_TIME, _) => 0,
            _ => 0,
        };
        system
            .memory
            .write32(values_ptr + i * 8, value as u32);
        system
            .memory
            .write32(values_ptr + i * 8 + 4, (value >> 32) as u32);
    }
    system.cpu.regs[0] = 0;
}

fn get_resource_limit_limit_values(system: &mut System) {
    resource_limit_values(system, false);
}

fn get_resource_limit_current_values(system: &mut System) {
    resource_limit_values(system, true);
}

// ---------------------------------------------------------------------------
// Debug
// ---------------------------------------------------------------------------

fn svc_break(system: &mut System) {
    let reason = system.cpu.regs[0];
    let kind = match reason {
        0 => "panic",
        1 => "assert",
        2 => "user",
        _ => "unknown",
    };
    log::error!(
        "guest called svcBreak ({kind}) at 0x{:08X}",
        system.cpu.current_pc()
    );
    log::error!("{:?}", system.cpu);
    log::error!("the instructions leading here were:");
    let recent = system.recent_instructions();
    for &entry in recent.iter().rev().take(24).rev() {
        log::error!(
            "  0x{:08X} {}",
            entry & !1,
            if entry & 1 != 0 { "T" } else { "A" }
        );
    }
    system.broke = true;
}

/// svcOutputDebugString, r0 = pointer, r1 = length.
fn output_debug_string(system: &mut System) {
    let ptr = system.cpu.regs[0];
    let len = system.cpu.regs[1].min(0x1000) as usize;
    let mut buf = vec![0u8; len];
    system.memory.read_bytes(ptr, &mut buf);
    let text = String::from_utf8_lossy(&buf);
    log::info!("[guest] {}", text.trim_end());
    system.debug_output.push_str(text.trim_end());
    system.debug_output.push('\n');
    system.cpu.regs[0] = 0;
}

/// exposed so the syscall layer can consume a signal it already checked.
impl super::Kernel {
    pub fn acquire_public(&mut self, object: ObjectId, waiter: super::thread::ThreadId) {
        self.acquire(object, waiter)
    }
}

/// the address a thread returns to when its entry point falls off the end.
pub const THREAD_EXIT_ADDRESS: VAddr = THREAD_EXIT_MAGIC;

/// unused placeholder keeping CURRENT_PROCESS referenced until process
/// handles are wired up.
pub const _CURRENT_PROCESS: u32 = CURRENT_PROCESS;
