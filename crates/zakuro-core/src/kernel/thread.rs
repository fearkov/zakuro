//! threads and their saved CPU state.

use zakuro_common::VAddr;
use zakuro_cpu::{Cpu, Psr};

use super::object::ObjectId;

/// index into the kernel's thread table.
pub type ThreadId = u32;

/// the ARM11 runs at 268.111856 MHz.
pub const CPU_CLOCK_HZ: u64 = 268_111_856;

pub const fn nanos_to_ticks(nanos: u64) -> u64 {
    // done in two steps to keep the intermediate inside 64 bits for the
    // multi-second timeouts games use when waiting on network operations.
    (nanos / 1_000_000_000) * CPU_CLOCK_HZ + (nanos % 1_000_000_000) * CPU_CLOCK_HZ / 1_000_000_000
}

pub const fn ticks_to_nanos(ticks: u64) -> u64 {
    (ticks / CPU_CLOCK_HZ) * 1_000_000_000 + (ticks % CPU_CLOCK_HZ) * 1_000_000_000 / CPU_CLOCK_HZ
}

/// everything that has to be saved across a context switch.
#[derive(Debug, Clone)]
pub struct ThreadContext {
    pub regs: [u32; 16],
    pub cpsr: Psr,
    pub vfp_regs: [u32; 32],
    pub fpscr: u32,
    pub fpexc: u32,
    /// CP15 c13,c0,3, points at this thread's TLS page.
    pub tls_pointer: u32,
}

impl ThreadContext {
    pub fn new(entry: VAddr, stack_top: VAddr, arg: u32, tls: VAddr) -> ThreadContext {
        let mut regs = [0u32; 16];
        regs[0] = arg;
        regs[13] = stack_top;
        regs[15] = entry & !1;
        // the kernel sets LR to an address that ends the thread, we recognize
        // it in the SVC handler instead of needing real code there.
        regs[14] = THREAD_EXIT_MAGIC;

        // bit 0 of the entry point selects Thumb, as it does for BX.
        let cpsr = Psr {
            thumb: entry & 1 != 0,
            ..Psr::default()
        };

        ThreadContext {
            regs,
            cpsr,
            vfp_regs: [0; 32],
            fpscr: 0,
            fpexc: 1 << 30,
            tls_pointer: tls,
        }
    }

    pub fn save_from(&mut self, cpu: &Cpu) {
        self.regs = cpu.regs;
        self.cpsr = cpu.cpsr;
        self.vfp_regs = cpu.vfp.regs;
        self.fpscr = cpu.vfp.fpscr;
        self.fpexc = cpu.vfp.fpexc;
        self.tls_pointer = cpu.cp15.thread_id_ro;
    }

    pub fn restore_to(&self, cpu: &mut Cpu) {
        cpu.regs = self.regs;
        cpu.cpsr = self.cpsr;
        cpu.vfp.regs = self.vfp_regs;
        cpu.vfp.fpscr = self.fpscr;
        cpu.vfp.fpexc = self.fpexc;
        cpu.cp15.thread_id_ro = self.tls_pointer;
        // a reservation never survives a context switch.
        cpu.exclusive_addr = None;
    }
}

/// the return address the kernel gives a fresh thread.
pub const THREAD_EXIT_MAGIC: u32 = 0xFFFF_FFF0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadStatus {
    Running,
    Ready,
    /// sleeping until a tick, with no object involved.
    Sleeping,
    /// waiting on one or more kernel objects.
    WaitSync,
    /// parked on an address arbiter.
    WaitArbiter,
    /// created but not started.
    Dormant,
    Dead,
}

impl ThreadStatus {
    pub fn is_blocked(self) -> bool {
        matches!(
            self,
            ThreadStatus::Sleeping | ThreadStatus::WaitSync | ThreadStatus::WaitArbiter
        )
    }
}

#[derive(Debug, Clone)]
pub struct Thread {
    pub id: ThreadId,
    pub name: String,
    pub context: ThreadContext,
    /// 0 is the highest priority, 0x3F the lowest.
    pub priority: u32,
    pub status: ThreadStatus,

    pub entry: VAddr,
    pub stack_top: VAddr,
    pub tls: VAddr,
    pub processor_id: i32,

    /// objects this thread is waiting on, and whether all of them are needed.
    pub wait_objects: Vec<ObjectId>,
    pub wait_all: bool,
    /// tick at which a timeout expires. None means wait forever.
    pub wakeup_at: Option<u64>,
    /// address this thread is parked on, for arbiter waits.
    pub wait_address: Option<VAddr>,
    /// set when the wait was satisfied, so the syscall knows what to return.
    pub wait_result: Option<WaitResult>,
    /// which syscall is waiting for that result.
    pub wait_syscall: Option<WaitSyscall>,

    /// handed to svcGetThreadId.
    pub guest_id: u32,
}

/// which syscall put a thread to sleep, so the scheduler knows how to write
/// the result back when it wakes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitSyscall {
    WaitSynchronization1,
    WaitSynchronizationN,
    ArbitrateAddress,
    SleepThread,
}

/// why a blocked thread woke up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitResult {
    /// the wait succeeded, the value is the index of the object that did it.
    Signaled(usize),
    TimedOut,
}

impl Thread {
    // every one of these comes from svcCreateThread's arguments, so bundling
    // them into a struct would only move the same list somewhere else.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: ThreadId,
        name: impl Into<String>,
        entry: VAddr,
        stack_top: VAddr,
        arg: u32,
        priority: u32,
        tls: VAddr,
        processor_id: i32,
    ) -> Thread {
        Thread {
            id,
            name: name.into(),
            context: ThreadContext::new(entry, stack_top, arg, tls),
            priority,
            status: ThreadStatus::Ready,
            entry,
            stack_top,
            tls,
            processor_id,
            wait_objects: Vec::new(),
            wait_all: false,
            wakeup_at: None,
            wait_address: None,
            wait_result: None,
            wait_syscall: None,
            guest_id: id,
        }
    }

    pub fn is_runnable(&self) -> bool {
        matches!(self.status, ThreadStatus::Ready | ThreadStatus::Running)
    }

    /// clears everything a wait left behind.
    pub fn clear_wait(&mut self) {
        self.wait_objects.clear();
        self.wait_all = false;
        self.wakeup_at = None;
        self.wait_address = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_conversion_round_trips() {
        for nanos in [0u64, 1_000, 1_000_000, 1_000_000_000, 16_666_666] {
            let ticks = nanos_to_ticks(nanos);
            let back = ticks_to_nanos(ticks);
            // allow a tick of slop, which is 3.7 ns.
            assert!(
                back.abs_diff(nanos) < 10,
                "{nanos} ns -> {ticks} ticks -> {back} ns"
            );
        }
    }

    #[test]
    fn one_second_is_the_clock_rate() {
        assert_eq!(nanos_to_ticks(1_000_000_000), CPU_CLOCK_HZ);
    }
}
