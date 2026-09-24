//! the high-level-emulated CTR kernel, objects, threads and scheduling.

pub mod ipc;
pub mod object;
pub mod svc;
pub mod sync;
pub mod thread;

use zakuro_common::memory_map::{TLS_AREA_VADDR, TLS_ENTRY_SIZE};
use zakuro_common::VAddr;
use zakuro_cpu::Cpu;

use crate::memory::MemoryRegion;
use object::{Handle, HandleTable, KObject, ObjectId, ObjectStore, CURRENT_PROCESS, CURRENT_THREAD};
use sync::ResetType;
use thread::{Thread, ThreadId, ThreadStatus, WaitResult};

/// lowest (numerically highest) priority a thread can have.
pub const LOWEST_PRIORITY: u32 = 0x3F;

pub struct Kernel {
    pub objects: ObjectStore,
    pub handles: HandleTable,
    pub threads: Vec<Thread>,
    pub current_thread: Option<ThreadId>,

    /// next free slot in the TLS area.
    next_tls_slot: u32,

    pub process_id: u32,
    pub program_id: u64,
    pub memory_region: MemoryRegion,

    /// current top of the svcControlMemory heap.
    pub heap_top: VAddr,
    /// current top of the linear heap.
    pub linear_top: VAddr,
    pub linear_base: VAddr,

    /// set when something happened that might make a different thread
    /// runnable, so the run loop knows to call [Kernel::schedule].
    pub reschedule_pending: bool,
    /// set when every thread is blocked, so the run loop can idle instead of
    /// spinning.
    pub all_blocked: bool,

    /// cached objects backing the pseudo-handles, so that duplicating
    /// CUR_THREAD_HANDLE twice yields the same object.
    thread_objects: std::collections::HashMap<ThreadId, ObjectId>,
    process_object: Option<ObjectId>,
}

impl Kernel {
    pub fn new(program_id: u64, memory_region: MemoryRegion, linear_base: VAddr) -> Kernel {
        Kernel {
            objects: ObjectStore::default(),
            handles: HandleTable::default(),
            threads: Vec::new(),
            current_thread: None,
            next_tls_slot: 0,
            process_id: 0x22,
            program_id,
            memory_region,
            heap_top: zakuro_common::memory_map::HEAP_VADDR,
            linear_top: linear_base,
            linear_base,
            reschedule_pending: false,
            all_blocked: false,
            thread_objects: std::collections::HashMap::new(),
            process_object: None,
        }
    }

    // -- threads ------------------------------------------------------------

    /// allocates the next TLS page slot for a new thread.
    pub fn allocate_tls(&mut self) -> VAddr {
        let addr = TLS_AREA_VADDR + self.next_tls_slot * TLS_ENTRY_SIZE;
        self.next_tls_slot += 1;
        addr
    }

    pub fn create_thread(
        &mut self,
        name: impl Into<String>,
        entry: VAddr,
        stack_top: VAddr,
        arg: u32,
        priority: u32,
        processor_id: i32,
    ) -> ThreadId {
        let id = self.threads.len() as ThreadId;
        let tls = self.allocate_tls();
        let thread = Thread::new(
            id,
            name,
            entry,
            stack_top,
            arg,
            priority.min(LOWEST_PRIORITY),
            tls,
            processor_id,
        );
        log::debug!(
            "created thread {id} '{}' entry=0x{entry:08X} sp=0x{stack_top:08X} prio={priority}",
            thread.name
        );
        self.threads.push(thread);
        self.reschedule_pending = true;
        id
    }

    pub fn thread(&self, id: ThreadId) -> &Thread {
        &self.threads[id as usize]
    }

    pub fn thread_mut(&mut self, id: ThreadId) -> &mut Thread {
        &mut self.threads[id as usize]
    }

    pub fn current(&self) -> Option<&Thread> {
        self.current_thread.map(|id| self.thread(id))
    }

    pub fn current_mut(&mut self) -> Option<&mut Thread> {
        let id = self.current_thread?;
        Some(self.thread_mut(id))
    }

    /// the object representing a thread, created on first use so that every
    /// handle to the same thread names the same object.
    pub fn thread_object(&mut self, id: ThreadId) -> ObjectId {
        if let Some(&object) = self.thread_objects.get(&id) {
            return object;
        }
        let object = self.objects.insert(KObject::Thread(id));
        self.thread_objects.insert(id, object);
        object
    }

    pub fn process_object(&mut self) -> ObjectId {
        if let Some(object) = self.process_object {
            return object;
        }
        let object = self.objects.insert(KObject::Process);
        self.process_object = Some(object);
        object
    }

    /// registers a thread object so the guest can hold a handle to it.
    pub fn thread_handle(&mut self, id: ThreadId) -> Handle {
        let object = self.thread_object(id);
        self.handles.create(&mut self.objects, object, "Thread")
    }

    /// resolves a handle, including the two pseudo-handles that name the
    /// calling thread and its process.
    pub fn resolve(&mut self, handle: Handle) -> Option<ObjectId> {
        match handle {
            CURRENT_THREAD => {
                let id = self.current_thread?;
                Some(self.thread_object(id))
            }
            CURRENT_PROCESS => Some(self.process_object()),
            _ => self.handles.resolve(handle),
        }
    }

    // -- scheduling ---------------------------------------------------------

    /// wakes anything whose wait has been satisfied, then switches to the
    /// highest-priority runnable thread.
    pub fn schedule(&mut self, cpu: &mut Cpu, tick: u64) -> bool {
        for id in 0..self.threads.len() as ThreadId {
            self.try_satisfy_wait(id, tick);
        }

        let next = self.pick_next();
        self.all_blocked = next.is_none();

        match (self.current_thread, next) {
            (Some(current), Some(next)) if current == next => {
                self.threads[current as usize].status = ThreadStatus::Running;
                false
            }
            (current, Some(next)) => {
                if let Some(current) = current {
                    let thread = &mut self.threads[current as usize];
                    thread.context.save_from(cpu);
                    if thread.status == ThreadStatus::Running {
                        thread.status = ThreadStatus::Ready;
                    }
                }
                let thread = &mut self.threads[next as usize];
                thread.status = ThreadStatus::Running;
                thread.context.restore_to(cpu);
                self.current_thread = Some(next);
                true
            }
            (Some(current), None) => {
                // everything is blocked.
                let thread = &mut self.threads[current as usize];
                thread.context.save_from(cpu);
                if thread.status == ThreadStatus::Running {
                    thread.status = ThreadStatus::Ready;
                }
                self.current_thread = None;
                true
            }
            (None, None) => false,
        }
    }

    /// the highest-priority runnable thread, round-robining within a priority
    /// by preferring the one after the current thread.
    fn pick_next(&self) -> Option<ThreadId> {
        let best = self
            .threads
            .iter()
            .filter(|t| t.is_runnable())
            .map(|t| t.priority)
            .min()?;

        let count = self.threads.len();
        let start = self.current_thread.map_or(0, |id| id as usize + 1);
        (0..count)
            .map(|offset| (start + offset) % count)
            .find(|&index| {
                let t = &self.threads[index];
                t.is_runnable() && t.priority == best
            })
            .map(|index| index as ThreadId)
    }

    /// checks one blocked thread's condition and unblocks it if satisfied.
    fn try_satisfy_wait(&mut self, id: ThreadId, tick: u64) -> bool {
        let thread = &self.threads[id as usize];
        if !thread.status.is_blocked() {
            return false;
        }

        let timed_out = thread.wakeup_at.is_some_and(|at| tick >= at);

        match thread.status {
            ThreadStatus::Sleeping => {
                if timed_out {
                    let thread = &mut self.threads[id as usize];
                    thread.clear_wait();
                    thread.status = ThreadStatus::Ready;
                    return true;
                }
            }
            ThreadStatus::WaitSync => {
                let objects = thread.wait_objects.clone();
                let wait_all = thread.wait_all;

                let states: Vec<bool> = objects
                    .iter()
                    .map(|&object| self.is_signaled(object, id))
                    .collect();

                let satisfied = if wait_all {
                    states.iter().all(|&s| s)
                } else {
                    states.iter().any(|&s| s)
                };

                if satisfied {
                    let index = if wait_all {
                        for &object in &objects {
                            self.acquire(object, id);
                        }
                        0
                    } else {
                        let index = states.iter().position(|&s| s).unwrap();
                        self.acquire(objects[index], id);
                        index
                    };
                    let thread = &mut self.threads[id as usize];
                    thread.clear_wait();
                    thread.wait_result = Some(WaitResult::Signaled(index));
                    thread.status = ThreadStatus::Ready;
                    return true;
                }

                if timed_out {
                    let thread = &mut self.threads[id as usize];
                    thread.clear_wait();
                    thread.wait_result = Some(WaitResult::TimedOut);
                    thread.status = ThreadStatus::Ready;
                    return true;
                }
            }
            ThreadStatus::WaitArbiter
                // arbiter waits are released explicitly by a signalling
                // thread, only the timeout is handled here.
                if timed_out => {
                    let thread = &mut self.threads[id as usize];
                    thread.clear_wait();
                    thread.wait_result = Some(WaitResult::TimedOut);
                    thread.status = ThreadStatus::Ready;
                    return true;
                }
            _ => {}
        }
        false
    }

    /// whether waiting on object would succeed right now for waiter.
    pub fn is_signaled(&self, object: ObjectId, waiter: ThreadId) -> bool {
        match self.objects.get(object) {
            Some(KObject::Event(event)) => event.signaled,
            Some(KObject::Timer(timer)) => timer.signaled,
            Some(KObject::Semaphore(semaphore)) => semaphore.count > 0,
            Some(KObject::Mutex(mutex)) => {
                mutex.owner.is_none() || mutex.owner == Some(waiter)
            }
            Some(KObject::Thread(id)) => {
                self.threads[*id as usize].status == ThreadStatus::Dead
            }
            // with HLE services a request is answered before the syscall
            // returns, so a session is never something to wait on.
            Some(KObject::ClientSession(_)) | Some(KObject::ClientPort(_)) => true,
            Some(KObject::SharedMemory(_)) | Some(KObject::Process)
            | Some(KObject::ResourceLimit) | Some(KObject::AddressArbiter(_)) => true,
            None => true,
        }
    }

    /// consumes the signal a successful wait picked up.
    fn acquire(&mut self, object: ObjectId, waiter: ThreadId) {
        match self.objects.get_mut(object) {
            Some(KObject::Event(event)) => {
                if event.reset_type == ResetType::OneShot {
                    event.signaled = false;
                }
            }
            Some(KObject::Timer(timer)) => {
                if timer.reset_type == ResetType::OneShot {
                    timer.signaled = false;
                }
            }
            Some(KObject::Semaphore(semaphore)) => semaphore.count -= 1,
            Some(KObject::Mutex(mutex)) => {
                mutex.owner = Some(waiter);
                mutex.lock_count += 1;
            }
            _ => {}
        }
    }

    /// blocks the current thread on a set of objects.
    pub fn begin_wait(
        &mut self,
        objects: Vec<ObjectId>,
        wait_all: bool,
        timeout_ticks: Option<u64>,
        tick: u64,
    ) {
        let Some(id) = self.current_thread else { return };
        let thread = &mut self.threads[id as usize];
        thread.wait_objects = objects;
        thread.wait_all = wait_all;
        thread.wakeup_at = timeout_ticks.map(|t| tick.saturating_add(t));
        thread.wait_result = None;
        thread.status = ThreadStatus::WaitSync;
        self.reschedule_pending = true;
    }

    pub fn sleep_current(&mut self, ticks: u64, tick: u64) {
        let Some(id) = self.current_thread else { return };
        let thread = &mut self.threads[id as usize];
        thread.clear_wait();
        thread.wakeup_at = Some(tick.saturating_add(ticks));
        thread.status = ThreadStatus::Sleeping;
        self.reschedule_pending = true;
    }

    /// signals an event, waking anything waiting on it at the next scheduling
    /// point.
    pub fn signal_event(&mut self, object: ObjectId) {
        if let Some(KObject::Event(event)) = self.objects.get_mut(object) {
            event.signaled = true;
            self.reschedule_pending = true;
        }
    }

    pub fn clear_event(&mut self, object: ObjectId) {
        if let Some(KObject::Event(event)) = self.objects.get_mut(object) {
            event.signaled = false;
        }
    }

    /// creates an event object and a handle for it in one step, which is what
    /// nearly every caller wants.
    pub fn create_event(&mut self, reset_type: ResetType, name: &str) -> (ObjectId, Handle) {
        let id = self
            .objects
            .insert(KObject::Event(sync::Event::new(reset_type, name)));
        let handle = self.handles.create(&mut self.objects, id, name);
        (id, handle)
    }

    /// describes what a blocked thread is waiting for, for diagnostics.
    pub fn describe_wait(&self, id: ThreadId) -> String {
        let thread = self.thread(id);
        match thread.status {
            ThreadStatus::Sleeping => match thread.wakeup_at {
                Some(at) => format!("sleeping until tick {at}"),
                None => "sleeping forever".into(),
            },
            ThreadStatus::WaitArbiter => format!(
                "arbiter on 0x{:08X}",
                thread.wait_address.unwrap_or(0)
            ),
            ThreadStatus::WaitSync => {
                let objects: Vec<String> = thread
                    .wait_objects
                    .iter()
                    .map(|&object| {
                        let label = self
                            .handles
                            .iter()
                            .find(|(_, id)| *id == object)
                            .map(|(handle, _)| self.handles.label(handle).to_owned())
                            .unwrap_or_else(|| "?".into());
                        let kind = self
                            .objects
                            .get(object)
                            .map_or("missing", |o| o.type_name());
                        let signalled = self.is_signaled(object, id);
                        format!("{label}:{kind}{}", if signalled { "(ready)" } else { "" })
                    })
                    .collect();
                format!(
                    "waiting for {} of [{}]",
                    if thread.wait_all { "all" } else { "any" },
                    objects.join(", ")
                )
            }
            other => format!("{other:?}"),
        }
    }

    /// number of threads that are not dead, for the diagnostics overlay.
    pub fn live_thread_count(&self) -> usize {
        self.threads
            .iter()
            .filter(|t| t.status != ThreadStatus::Dead)
            .count()
    }
}
