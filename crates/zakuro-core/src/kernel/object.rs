//! kernel objects and the per-process handle table.

use std::collections::HashMap;

use zakuro_common::VAddr;

use super::sync::{AddressArbiter, Event, Mutex, Semaphore, Timer};
use super::thread::ThreadId;

/// a guest-visible handle.
pub type Handle = u32;

/// svcGetProcessId and friends accept these instead of a real handle.
pub const CURRENT_PROCESS: Handle = 0xFFFF_8001;
pub const CURRENT_THREAD: Handle = 0xFFFF_8000;

/// index into the kernel's object slab.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ObjectId(pub u32);

#[derive(Debug)]
pub enum KObject {
    Thread(ThreadId),
    Event(Event),
    Mutex(Mutex),
    Semaphore(Semaphore),
    Timer(Timer),
    AddressArbiter(AddressArbiter),
    SharedMemory(SharedMemory),
    /// the client end of an IPC session.
    ClientSession(ClientSession),
    /// a named port a game can connect to, e.g. srv:.
    ClientPort(String),
    Process,
    ResourceLimit,
}

impl KObject {
    pub fn type_name(&self) -> &'static str {
        match self {
            KObject::Thread(_) => "Thread",
            KObject::Event(_) => "Event",
            KObject::Mutex(_) => "Mutex",
            KObject::Semaphore(_) => "Semaphore",
            KObject::Timer(_) => "Timer",
            KObject::AddressArbiter(_) => "AddressArbiter",
            KObject::SharedMemory(_) => "SharedMemory",
            KObject::ClientSession(_) => "ClientSession",
            KObject::ClientPort(_) => "ClientPort",
            KObject::Process => "Process",
            KObject::ResourceLimit => "ResourceLimit",
        }
    }
}

/// a block of memory shared between processes, or between a process and a
/// service. GSP and HID both hand one of these to the game.
#[derive(Debug, Clone)]
pub struct SharedMemory {
    pub name: String,
    /// address in the owning process, or 0 when the kernel picks.
    pub address: VAddr,
    pub size: u32,
    /// physical address backing it, so it can be mapped elsewhere.
    pub paddr: u32,
    pub mapped_at: Option<VAddr>,
}

/// the game's end of a connection to a service.
#[derive(Debug, Clone)]
pub struct ClientSession {
    /// which HLE service handles requests on this session.
    pub service: String,
    /// per-session state, used by services like FS that hand out sub-sessions.
    pub subhandle: u32,
}

/// one slab entry.
#[derive(Debug)]
struct Slot {
    object: KObject,
    /// number of handles naming this object.
    refcount: u32,
}

#[derive(Debug, Default)]
pub struct ObjectStore {
    slots: Vec<Option<Slot>>,
    free: Vec<u32>,
}

impl ObjectStore {
    pub fn insert(&mut self, object: KObject) -> ObjectId {
        let slot = Slot {
            object,
            refcount: 0,
        };
        if let Some(index) = self.free.pop() {
            self.slots[index as usize] = Some(slot);
            ObjectId(index)
        } else {
            self.slots.push(Some(slot));
            ObjectId(self.slots.len() as u32 - 1)
        }
    }

    pub fn get(&self, id: ObjectId) -> Option<&KObject> {
        self.slots
            .get(id.0 as usize)
            .and_then(|s| s.as_ref())
            .map(|s| &s.object)
    }

    pub fn get_mut(&mut self, id: ObjectId) -> Option<&mut KObject> {
        self.slots
            .get_mut(id.0 as usize)
            .and_then(|s| s.as_mut())
            .map(|s| &mut s.object)
    }

    fn add_ref(&mut self, id: ObjectId) {
        if let Some(Some(slot)) = self.slots.get_mut(id.0 as usize) {
            slot.refcount += 1;
        }
    }

    fn release(&mut self, id: ObjectId) {
        let Some(Some(slot)) = self.slots.get_mut(id.0 as usize) else {
            return;
        };
        slot.refcount = slot.refcount.saturating_sub(1);
        if slot.refcount == 0 {
            self.slots[id.0 as usize] = None;
            self.free.push(id.0);
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (ObjectId, &KObject)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.as_ref().map(|s| (ObjectId(i as u32), &s.object)))
    }
}

#[derive(Debug)]
pub struct HandleTable {
    entries: HashMap<Handle, ObjectId>,
    next: Handle,
    /// names, kept only for logs, a handle leak is far easier to find when the
    /// log can say what leaked.
    labels: HashMap<Handle, String>,
}

impl Default for HandleTable {
    fn default() -> Self {
        HandleTable {
            entries: HashMap::new(),
            // real handles start well above zero so that a zeroed variable
            // used as a handle fails loudly.
            next: 0x0001_0000,
            labels: HashMap::new(),
        }
    }
}

impl HandleTable {
    pub fn create(&mut self, store: &mut ObjectStore, id: ObjectId, label: &str) -> Handle {
        let handle = self.next;
        self.next += 1;
        self.entries.insert(handle, id);
        self.labels.insert(handle, label.to_owned());
        store.add_ref(id);
        handle
    }

    pub fn resolve(&self, handle: Handle) -> Option<ObjectId> {
        self.entries.get(&handle).copied()
    }

    /// creates a second handle naming an object that is already known.
    pub fn duplicate_object(
        &mut self,
        store: &mut ObjectStore,
        id: ObjectId,
        label: &str,
    ) -> Handle {
        self.create(store, id, label)
    }

    pub fn close(&mut self, store: &mut ObjectStore, handle: Handle) -> bool {
        match self.entries.remove(&handle) {
            Some(id) => {
                self.labels.remove(&handle);
                store.release(id);
                true
            }
            None => false,
        }
    }

    pub fn label(&self, handle: Handle) -> &str {
        self.labels.get(&handle).map_or("?", |s| s.as_str())
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (Handle, ObjectId)> + '_ {
        self.entries.iter().map(|(&h, &id)| (h, id))
    }
}
