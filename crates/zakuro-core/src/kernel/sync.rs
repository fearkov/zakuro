//! synchronisation primitives.

use zakuro_common::VAddr;

use super::thread::ThreadId;

/// how an event behaves once a waiter picks it up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResetType {
    /// stays signalled until svcClearEvent.
    OneShot,
    /// releases exactly one waiter and clears itself.
    Sticky,
    /// timer-only, fires repeatedly.
    Pulse,
}

impl ResetType {
    pub fn from_raw(value: u32) -> ResetType {
        match value {
            1 => ResetType::Sticky,
            2 => ResetType::Pulse,
            _ => ResetType::OneShot,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Event {
    pub name: String,
    pub reset_type: ResetType,
    pub signaled: bool,
}

impl Event {
    pub fn new(reset_type: ResetType, name: impl Into<String>) -> Event {
        Event {
            name: name.into(),
            reset_type,
            signaled: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Mutex {
    pub name: String,
    pub owner: Option<ThreadId>,
    /// mutexes on the 3DS are recursive, the count is how many times the owner
    /// has taken it.
    pub lock_count: u32,
}

impl Mutex {
    pub fn new(name: impl Into<String>) -> Mutex {
        Mutex {
            name: name.into(),
            owner: None,
            lock_count: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Semaphore {
    pub name: String,
    pub count: i32,
    pub max_count: i32,
}

#[derive(Debug, Clone)]
pub struct Timer {
    pub name: String,
    pub reset_type: ResetType,
    pub signaled: bool,
    /// absolute tick when the timer next fires, or None when stopped.
    pub fire_at: Option<u64>,
    /// period in ticks for a repeating timer.
    pub interval: u64,
}

/// an address arbiter is the 3DS's futex, threads sleep on a guest address and
/// are woken by whoever writes to it.
#[derive(Debug, Clone, Default)]
pub struct AddressArbiter {
    pub name: String,
    /// threads currently parked, in the order they arrived.
    pub waiters: Vec<(ThreadId, VAddr)>,
}

/// svcArbitrateAddress operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArbitrationType {
    Signal,
    WaitIfLessThan,
    DecrementAndWaitIfLessThan,
    WaitIfLessThanWithTimeout,
    DecrementAndWaitIfLessThanWithTimeout,
}

impl ArbitrationType {
    pub fn from_raw(value: u32) -> Option<ArbitrationType> {
        Some(match value {
            0 => ArbitrationType::Signal,
            1 => ArbitrationType::WaitIfLessThan,
            2 => ArbitrationType::DecrementAndWaitIfLessThan,
            3 => ArbitrationType::WaitIfLessThanWithTimeout,
            4 => ArbitrationType::DecrementAndWaitIfLessThanWithTimeout,
            _ => return None,
        })
    }

    pub fn is_wait(self) -> bool {
        !matches!(self, ArbitrationType::Signal)
    }

    pub fn has_timeout(self) -> bool {
        matches!(
            self,
            ArbitrationType::WaitIfLessThanWithTimeout
                | ArbitrationType::DecrementAndWaitIfLessThanWithTimeout
        )
    }

    pub fn decrements(self) -> bool {
        matches!(
            self,
            ArbitrationType::DecrementAndWaitIfLessThan
                | ArbitrationType::DecrementAndWaitIfLessThanWithTimeout
        )
    }
}
