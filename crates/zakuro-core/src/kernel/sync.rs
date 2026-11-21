//! sync primitives

use zakuro_common::VAddr;

use super::thread::ThreadId;

/// how an event or timer behaves once a waiter picks it up
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ResetType {
    /// releases exactly one waiter and clears itself
    OneShot,
    /// stays signalled till cleared
    Sticky,
    /// releases its waiters when signalled and clears, lost when nobody waits
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Mutex {
    pub name: String,
    pub owner: Option<ThreadId>,
    /// 3DS mutexes are recursive, the count is how many times the owner took it
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Semaphore {
    pub name: String,
    pub count: i32,
    pub max_count: i32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Timer {
    pub name: String,
    pub reset_type: ResetType,
    pub signaled: bool,
    /// absolute tick the timer fires next, None when stopped
    pub fire_at: Option<u64>,
    /// period in ticks for a repeating timer
    pub interval: u64,
}

/// an address arbiter is the 3DS's futex, threads sleep on a guest addr and whoever writes it wakes them
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct AddressArbiter {
    pub name: String,
    /// threads parked rn, in arrival order
    pub waiters: Vec<(ThreadId, VAddr)>,
}

/// svcArbitrateAddress ops
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
