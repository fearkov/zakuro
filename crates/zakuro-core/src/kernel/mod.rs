//! the HLE CTR kernel, objects, threads and scheduling

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

/// lowest (numerically highest) priority a thread can have
pub const LOWEST_PRIORITY: u32 = 0x3F;

/// max handles satisfied_now looks at
pub(crate) const WAIT_AT_ONCE: usize = 8;

/// how long a thread may run w/o waiting b4 the ones below it get a share, four frames
const STARVATION_TICKS: u64 = 4 * crate::CYCLES_PER_FRAME;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Kernel {
    pub objects: ObjectStore,
    pub handles: HandleTable,
    pub threads: Vec<Thread>,
    pub current_thread: Option<ThreadId>,

    /// next free TLS slot
    next_tls_slot: u32,
    /// TLS blocks of threads that ended, reused for the next ones
    free_tls: std::collections::BTreeSet<VAddr>,
    /// what svcGetThreadId gives the next thread, ids never get reused
    pub next_guest_id: u32,

    pub process_id: u32,
    pub program_id: u64,
    pub memory_region: MemoryRegion,

    /// top of the svcControlMemory heap rn
    pub heap_top: VAddr,
    /// top of the linear heap rn
    pub linear_top: VAddr,
    pub linear_base: VAddr,
    /// exheader flag making addressless memory blocks come from the process's own region, at the bottom w/ its linear heap
    pub shared_device_memory: bool,

    /// set when smth happened that might make another thread runnable, so the run loop calls [Kernel::schedule]
    pub reschedule_pending: bool,
    /// set when every thread is blocked, so the run loop can idle instead of spinning
    pub all_blocked: bool,
    /// flips while a thread starves the ones below it, giving them every other slice
    relief: bool,

    /// cached objects behind the pseudo handles, so duplicating CUR_THREAD_HANDLE twice gives the same object
    thread_objects: zakuro_common::hash::IntMap<ThreadId, ObjectId>,
    process_object: Option<ObjectId>,

    /// the timers, scanned for the next one due instead of every object, rebuilt each frame
    pub timers: Vec<ObjectId>,
    /// pulse events and timers signalled since the last schedule, they clear once it released whatever waited on them
    pub pulsed: Vec<ObjectId>,
}

impl Kernel {
    pub fn new(program_id: u64, memory_region: MemoryRegion, linear_base: VAddr) -> Kernel {
        Kernel {
            objects: ObjectStore::default(),
            handles: HandleTable::default(),
            threads: Vec::new(),
            current_thread: None,
            next_tls_slot: 0,
            free_tls: std::collections::BTreeSet::new(),
            next_guest_id: 0,
            process_id: 0x22,
            program_id,
            memory_region,
            heap_top: zakuro_common::memory_map::HEAP_VADDR,
            linear_top: linear_base,
            linear_base,
            shared_device_memory: false,
            reschedule_pending: false,
            all_blocked: false,
            relief: false,
            thread_objects: zakuro_common::hash::IntMap::default(),
            process_object: None,
            timers: Vec::new(),
            pulsed: Vec::new(),
        }
    }

    // threads

    /// a TLS block for a new thread, lowest one an ended thread gave back first
    pub fn allocate_tls(&mut self) -> VAddr {
        if let Some(addr) = self.free_tls.pop_first() {
            return addr;
        }
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
        // an ended thread nothing names anymore gives its slot to the new one
        // Pokemon Sun starts one every other frame for local wireless, the table and every schedule would grow forever
        let id = match self.ended_thread() {
            Some(id) => {
                self.forget_thread(id);
                id
            }
            None => self.threads.len() as ThreadId,
        };
        let tls = self.allocate_tls();
        let mut thread = Thread::new(
            id,
            name,
            entry,
            stack_top,
            arg,
            priority.min(LOWEST_PRIORITY),
            tls,
            processor_id,
        );
        thread.guest_id = self.next_guest_id;
        self.next_guest_id += 1;
        log::debug!(
            "created thread {id} '{}' entry=0x{entry:08X} sp=0x{stack_top:08X} prio={priority}",
            thread.name
        );
        match self.threads.get_mut(id as usize) {
            Some(place) => *place = thread,
            None => self.threads.push(thread),
        }
        self.reschedule_pending = true;
        id
    }

    /// an ended thread no handle or wait names anymore
    fn ended_thread(&self) -> Option<ThreadId> {
        let unnamed = |id: ThreadId| {
            self.thread_objects.get(&id).is_none_or(|&object| self.objects.refcount(object) <= 1)
        };
        (0..self.threads.len() as ThreadId).find(|&id| self.threads[id as usize].status == ThreadStatus::Dead && unnamed(id))
    }

    /// drops what still names an ended thread b4 its slot goes to a new one
    /// the object kept for its handles, plus entries a timed out wait left on address arbiters
    fn forget_thread(&mut self, id: ThreadId) {
        if let Some(object) = self.thread_objects.remove(&id) {
            self.objects.release(object);
        }
        let arbiters: Vec<ObjectId> = self
            .objects
            .iter()
            .filter(|(_, object)| matches!(object, KObject::AddressArbiter(_)))
            .map(|(object, _)| object)
            .collect();
        for object in arbiters {
            if let Some(KObject::AddressArbiter(arbiter)) = self.objects.get_mut(object) {
                arbiter.waiters.retain(|&(thread, _)| thread != id);
            }
        }
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

    /// could a thread besides the running one take the core
    pub fn others_runnable(&self) -> bool {
        self.threads.iter().enumerate().any(|(id, thread)| Some(id as ThreadId) != self.current_thread && thread.is_runnable())
    }

    /// can a loop turn take and drop handle's mutex w/ only the current thread running
    /// a mutex that's free or the current thread's, w/ nobody waiting on it
    pub(crate) fn quiet_mutex(&self, handle: Handle) -> bool {
        if handle == CURRENT_THREAD || handle == CURRENT_PROCESS {
            return false;
        }
        let (Some(object), Some(current)) = (self.handles.resolve(handle), self.current_thread) else { return false };
        let Some(KObject::Mutex(mutex)) = self.objects.get(object) else { return false };
        mutex.owner.is_none_or(|owner| owner == current)
            && !self.threads.iter().any(|thread| thread.status == ThreadStatus::WaitSync && thread.wait_objects.contains(&object))
    }

    /// the tick from which a preempt's due pass stops being quiet no matter what (see quiet_pass)
    /// the first timer firing (waited on or not), or the current thread running long enough for the others to get a share
    pub(crate) fn quiet_until(&self) -> Option<u64> {
        let current = self.current_thread?;
        let timer = |id: &ObjectId| match self.objects.get(*id) {
            Some(KObject::Timer(timer)) => timer.fire_at,
            _ => None,
        };
        let starving = self.threads[current as usize].running_since + STARVATION_TICKS + 1;
        Some(self.timers.iter().filter_map(timer).fold(starving, u64::min))
    }

    /// would the pass a preempt has due at tick change nothing but all_blocked (which it leaves false)
    /// the current thread runs on alone, not long enough to share, no wait result for it, no timer due, nothing pulsed
    /// anything else that could wake a thread has a pass due,
    /// a thread sleeping till a tick wakes on its own (caller has that ahead of tick)
    /// and a pass leaves no wait it could satisfy
    pub(crate) fn quiet_pass(&self, tick: u64) -> bool {
        let Some(current) = self.current_thread else { return false };
        let running = &self.threads[current as usize];
        running.status == ThreadStatus::Running
            && !running.waited
            && tick.saturating_sub(running.running_since) <= STARVATION_TICKS
            && running.wait_result.is_none()
            && running.wait_syscall.is_none()
            && self.pulsed.is_empty()
            && !self.timer_due(tick)
            && !self.others_runnable()
    }

    pub fn current_mut(&mut self) -> Option<&mut Thread> {
        let id = self.current_thread?;
        Some(self.thread_mut(id))
    }

    /// the object for a thread, made on first use so every handle to a thread names the same object
    pub fn thread_object(&mut self, id: ThreadId) -> ObjectId {
        if let Some(&object) = self.thread_objects.get(&id) {
            return object;
        }
        let object = self.objects.insert(KObject::Thread(id));
        // kept for every handle after, closed or not
        self.objects.add_ref(object);
        self.thread_objects.insert(id, object);
        object
    }

    pub fn process_object(&mut self) -> ObjectId {
        if let Some(object) = self.process_object {
            return object;
        }
        let object = self.objects.insert(KObject::Process);
        self.objects.add_ref(object);
        self.process_object = Some(object);
        object
    }

    /// registers a thread object so the guest can hold a handle to it
    pub fn thread_handle(&mut self, id: ThreadId) -> Handle {
        let object = self.thread_object(id);
        self.handles.create(&mut self.objects, object, "Thread")
    }

    /// resolves a handle, incl the two pseudo handles for the calling thread and its process
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

    // scheduling

    /// wakes anything whose wait got satisfied, then switches to the highest priority runnable thread
    pub fn schedule(&mut self, cpu: &mut Cpu, tick: u64) -> bool {
        self.fire_timers(tick);
        for id in 0..self.threads.len() as ThreadId {
            self.try_satisfy_wait(id, tick);
        }
        for id in std::mem::take(&mut self.pulsed) {
            match self.objects.get_mut(id) {
                Some(KObject::Event(event)) => event.signaled = false,
                Some(KObject::Timer(timer)) => timer.signaled = false,
                _ => {}
            }
        }

        if let Some(current) = self.current_thread {
            let thread = &mut self.threads[current as usize];
            if !thread.is_runnable() {
                thread.waited = true;
            }
        }
        let mut next = self.pick_next();
        self.all_blocked = next.is_none();
        // the console runs strictly by priority, but its threads wait on hw and on services for as long as the work takes
        // ours wait the same short while for any request, so a thread spinning on another's work would hang
        // one running this long w/o waiting shares the core w/ the ready ones
        if let Some(chosen) = next {
            let thread = &mut self.threads[chosen as usize];
            if thread.waited {
                thread.waited = false;
                thread.running_since = tick;
            }
            if tick.saturating_sub(thread.running_since) > STARVATION_TICKS {
                let priority = thread.priority;
                self.relief = !self.relief;
                if self.relief {
                    next = self.pick_other(chosen, priority).or(next);
                }
            }
        }
        if let Some(chosen) = next {
            let thread = &mut self.threads[chosen as usize];
            if thread.waited {
                thread.waited = false;
                thread.running_since = tick;
            }
        }

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
                // everything's blocked
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

    /// the most urgent runnable thread besides one, at its priority or below
    fn pick_other(&self, chosen: ThreadId, priority: u32) -> Option<ThreadId> {
        self.threads
            .iter()
            .enumerate()
            .filter(|&(id, t)| id as ThreadId != chosen && t.is_runnable() && t.priority >= priority)
            .min_by_key(|(_, t)| t.priority)
            .map(|(id, _)| id as ThreadId)
    }

    /// highest priority runnable thread, round robin within a priority by preferring the one after the current
    fn pick_next(&self) -> Option<ThreadId> {
        let best = self
            .threads
            .iter()
            .filter(|t| t.is_runnable())
            .map(|t| t.priority)
            .min()?;

        // the running thread keeps the core unless smth strictly more urgent is ready
        // waking a same priority thread doesn't hand over, only waiting or yielding does
        if let Some(current) = self.current_thread {
            let thread = &self.threads[current as usize];
            if thread.status == ThreadStatus::Running && thread.priority <= best {
                return Some(current);
            }
        }

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

    /// fires timers whose time came (once, however many periods passed) and rearms periodic ones for their first period after tick
    /// counted from when they came due, like the console counts, a pulse releases its waiters then clears
    /// returns whether any fired
    fn fire_timers(&mut self, tick: u64) -> bool {
        let mut fired = false;
        for &id in &self.timers {
            if let Some(KObject::Timer(timer)) = self.objects.get_mut(id) {
                let Some(at) = timer.fire_at.filter(|&at| at <= tick) else { continue };
                timer.signaled = true;
                timer.fire_at = (timer.interval > 0)
                    .then(|| at.saturating_add(timer.interval.saturating_mul((tick - at) / timer.interval + 1)));
                if timer.reset_type == ResetType::Pulse {
                    self.pulsed.push(id);
                }
                fired = true;
            }
        }
        fired
    }

    /// fires what came due b4 a title sets, clears, cancels or waits on a timer, like the console's interrupt would have by then
    /// a pulse nobody waits on is lost
    pub(crate) fn catch_up_timers(&mut self, tick: u64) {
        if !self.fire_timers(tick) {
            return;
        }
        self.reschedule_pending = true;
        let threads = &self.threads;
        let objects = &mut self.objects;
        self.pulsed.retain(|&id| {
            let waited = threads.iter().any(|t| t.status == ThreadStatus::WaitSync && t.wait_objects.contains(&id));
            if !waited {
                if let Some(KObject::Timer(timer)) = objects.get_mut(id) {
                    timer.signaled = false;
                }
            }
            waited
        });
    }

    /// earliest tick a blocked thread or timer wants the scheduler at
    /// a timer only matters then to a thread waiting on it, the rest fire when the scheduler next looks or the title touches them
    pub fn next_event(&self) -> Option<u64> {
        // nothing blocked is the common case, then no thread wants a tick and no timer means anything
        // so a run asking at every stop pays almost nothing
        let mut best: Option<u64> = None;
        let mut any_blocked = false;
        for thread in &self.threads {
            if thread.status.is_blocked() {
                any_blocked = true;
                if let Some(at) = thread.wakeup_at {
                    best = Some(best.map_or(at, |b: u64| b.min(at)));
                }
            }
        }
        if !any_blocked {
            return None;
        }
        for &id in &self.timers {
            let waited = self.threads.iter().any(|t| t.status.is_blocked() && t.wait_objects.contains(&id));
            if !waited {
                continue;
            }
            if let Some(KObject::Timer(timer)) = self.objects.get(id) {
                if let Some(at) = timer.fire_at {
                    best = Some(best.map_or(at, |b: u64| b.min(at)));
                }
            }
        }
        best
    }

    /// checks one blocked thread's condition and unblocks it if satisfied
    fn try_satisfy_wait(&mut self, id: ThreadId, tick: u64) -> bool {
        let thread = &self.threads[id as usize];
        if !thread.status.is_blocked() {
            return false;
        }

        let timed_out = thread.wakeup_at.is_some_and(|at| tick >= at);

        match thread.status {
            ThreadStatus::Sleeping => {
                if timed_out {
                    self.end_wait(id);
                    self.threads[id as usize].status = ThreadStatus::Ready;
                    return true;
                }
            }
            ThreadStatus::WaitSync => {
                let wait_all = thread.wait_all;
                // every object checked b4 any gets acquired, like the kernel does, w/o copying the list per blocked thread every schedule
                let count = thread.wait_objects.len();
                let signaled = |k: usize| self.is_signaled(self.threads[id as usize].wait_objects[k], id);
                let satisfied = if wait_all { (0..count).all(signaled).then_some(0) } else { (0..count).position(signaled) };

                if let Some(index) = satisfied {
                    if wait_all {
                        for k in 0..count {
                            let object = self.threads[id as usize].wait_objects[k];
                            self.acquire(object, id);
                        }
                    } else {
                        let object = self.threads[id as usize].wait_objects[index];
                        self.acquire(object, id);
                    }
                    self.end_wait(id);
                    let thread = &mut self.threads[id as usize];
                    thread.wait_result = Some(WaitResult::Signaled(index));
                    thread.status = ThreadStatus::Ready;
                    return true;
                }

                if timed_out {
                    self.end_wait(id);
                    let thread = &mut self.threads[id as usize];
                    thread.wait_result = Some(WaitResult::TimedOut);
                    thread.status = ThreadStatus::Ready;
                    return true;
                }
            }
            ThreadStatus::WaitArbiter
                // arbiter waits get released by a signalling thread, only the timeout is handled here
                if timed_out => {
                    self.end_wait(id);
                    let thread = &mut self.threads[id as usize];
                    thread.wait_result = Some(WaitResult::TimedOut);
                    thread.status = ThreadStatus::Ready;
                    return true;
                }
            _ => {}
        }
        false
    }

    /// what WaitSynchronizationN on handles would find at tick w/ nothing else to do first
    /// no timer due, every handle naming an object, the wait satisfied right away for the current thread
    /// returns the objects (as many as handles) and the index the wait gives
    pub(crate) fn satisfied_now(&self, handles: &[Handle], wait_all: bool, tick: u64) -> Option<([ObjectId; WAIT_AT_ONCE], u32)> {
        let thread = self.current_thread?;
        if handles.len() > WAIT_AT_ONCE || self.timer_due(tick) {
            return None;
        }
        let mut objects = [ObjectId(0); WAIT_AT_ONCE];
        for (object, &handle) in objects.iter_mut().zip(handles) {
            if handle == CURRENT_THREAD || handle == CURRENT_PROCESS {
                return None;
            }
            *object = self.handles.resolve(handle)?;
        }
        let waited = &objects[..handles.len()];
        let signaled = |object: &ObjectId| self.is_signaled(*object, thread);
        let index = if wait_all { waited.iter().all(signaled).then_some(0) } else { waited.iter().position(signaled) }?;
        Some((objects, index as u32))
    }

    /// does a timer come due by tick, the next pass fires it
    fn timer_due(&self, tick: u64) -> bool {
        let due = |id: &ObjectId| matches!(self.objects.get(*id), Some(KObject::Timer(timer)) if timer.fire_at.is_some_and(|at| at <= tick));
        self.timers.iter().any(due)
    }

    /// the mutex ReleaseMutex on handle drops at tick and whether it frees it, when the pass a freed mutex has due would change nothing
    /// same idea as satisfied_now finding a wait over at once
    /// no timer due, nothing pulsed, nobody waiting on the mutex, nothing more urgent ready,
    /// current thread runs on not long enough to share, no wait result left
    /// anything else that could wake a thread has a pass due and a pass leaves no wait it could satisfy, so the mutex is all that's new
    /// a thread sleeping till a tick wakes on its own, the caller has that ahead of tick
    pub(crate) fn quiet_release(&self, handle: Handle, tick: u64) -> Option<(ObjectId, bool)> {
        if handle == CURRENT_THREAD || handle == CURRENT_PROCESS {
            return None;
        }
        let object = self.handles.resolve(handle)?;
        let Some(KObject::Mutex(mutex)) = self.objects.get(object) else { return None };
        // held more than once it just counts down
        if mutex.lock_count > 1 {
            return Some((object, false));
        }
        let current = self.current_thread?;
        let running = &self.threads[current as usize];
        if running.status != ThreadStatus::Running
            || running.waited
            || tick.saturating_sub(running.running_since) > STARVATION_TICKS
            || running.wait_result.is_some()
            || running.wait_syscall.is_some()
            || !self.pulsed.is_empty()
            || self.timer_due(tick)
        {
            return None;
        }
        let quiet = self.threads.iter().enumerate().all(|(id, thread)| match thread.status {
            ThreadStatus::WaitSync => !thread.wait_objects.contains(&object),
            ThreadStatus::Ready | ThreadStatus::Running => id as ThreadId == current || thread.priority >= running.priority,
            _ => true,
        });
        quiet.then_some((object, true))
    }

    /// can recompiled code take and drop handle's mutex itself for a while from tick (see recomp_abi::Context)
    /// only the current thread can touch it, the mutex free or the current thread's, plus what quiet_release checks
    /// (nobody waiting, nothing more urgent ready and so on), none of which changes while only the thread's code runs
    /// returns the mutex, how many times the thread holds it, and the tick that holds until (a timer due or the lower threads' share)
    pub(crate) fn lendable(&self, handle: Handle, tick: u64) -> Option<(ObjectId, u32, u64)> {
        if handle == CURRENT_THREAD || handle == CURRENT_PROCESS || self.all_blocked {
            return None;
        }
        let object = self.handles.resolve(handle)?;
        let Some(KObject::Mutex(mutex)) = self.objects.get(object) else { return None };
        let current = self.current_thread?;
        if mutex.owner.is_some_and(|owner| owner != current) {
            return None;
        }
        let running = &self.threads[current as usize];
        if running.status != ThreadStatus::Running
            || running.waited
            || running.wait_result.is_some()
            || running.wait_syscall.is_some()
            || !self.pulsed.is_empty()
        {
            return None;
        }
        let quiet = self.threads.iter().enumerate().all(|(id, thread)| match thread.status {
            ThreadStatus::WaitSync => !thread.wait_objects.contains(&object),
            ThreadStatus::Ready | ThreadStatus::Running => id as ThreadId == current || thread.priority >= running.priority,
            _ => true,
        });
        let timer = |id: &ObjectId| match self.objects.get(*id) {
            Some(KObject::Timer(timer)) => timer.fire_at,
            _ => None,
        };
        let until = self.timers.iter().filter_map(timer).fold(running.running_since + STARVATION_TICKS + 1, u64::min);
        (quiet && tick < until).then_some((object, mutex.lock_count, until))
    }

    /// the mutex lent to recompiled code as the code left it, held count times by the current thread, or free
    pub(crate) fn held_as(&mut self, object: ObjectId, count: u32) {
        let owner = self.current_thread.filter(|_| count > 0);
        if let Some(KObject::Mutex(mutex)) = self.objects.get_mut(object) {
            mutex.lock_count = count;
            mutex.owner = owner;
        }
    }

    /// would waiting on object succeed right now for waiter
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
            // HLE services answer a request b4 the syscall returns, so a session is never smth to wait on
            Some(KObject::ClientSession(_)) | Some(KObject::ClientPort(_)) => true,
            Some(KObject::SharedMemory(_)) | Some(KObject::Process)
            | Some(KObject::ResourceLimit) | Some(KObject::AddressArbiter(_)) => true,
            None => true,
        }
    }

    /// eats the signal a successful wait picked up
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

    /// blocks the current thread on a set of objects
    pub fn begin_wait(
        &mut self,
        objects: Vec<ObjectId>,
        wait_all: bool,
        timeout_ticks: Option<u64>,
        tick: u64,
    ) {
        let Some(id) = self.current_thread else { return };
        // what a thread waits on stays while it does, a handle closed meanwhile can't free it and hand its slot to the next object
        for &object in &objects {
            self.objects.add_ref(object);
        }
        let thread = &mut self.threads[id as usize];
        thread.wait_objects = objects;
        thread.wait_all = wait_all;
        thread.wakeup_at = timeout_ticks.map(|t| tick.saturating_add(t));
        thread.wait_result = None;
        thread.status = ThreadStatus::WaitSync;
        self.reschedule_pending = true;
    }

    /// ends a thread's wait, dropping the objects it held onto while waiting
    fn end_wait(&mut self, id: ThreadId) {
        for object in std::mem::take(&mut self.threads[id as usize].wait_objects) {
            self.objects.release(object);
        }
        self.threads[id as usize].clear_wait();
    }

    /// could a thread besides the current one run
    pub fn others_ready(&self) -> bool {
        self.threads
            .iter()
            .enumerate()
            .any(|(id, thread)| Some(id as ThreadId) != self.current_thread && thread.is_runnable())
    }

    pub fn sleep_current(&mut self, ticks: u64, tick: u64) {
        let Some(id) = self.current_thread else { return };
        let thread = &mut self.threads[id as usize];
        thread.clear_wait();
        thread.wakeup_at = Some(tick.saturating_add(ticks));
        thread.status = ThreadStatus::Sleeping;
        self.reschedule_pending = true;
    }

    /// ends a thread, it gives back every mutex it holds all the way down like the real kernel
    /// one held by a dead thread would block its next waiter forever
    pub fn end_thread(&mut self, id: ThreadId) {
        self.end_wait(id);
        let thread = &mut self.threads[id as usize];
        thread.status = ThreadStatus::Dead;
        thread.wait_result = None;
        thread.wait_syscall = None;
        self.free_tls.insert(thread.tls);
        let held: Vec<ObjectId> = self
            .objects
            .iter()
            .filter(|(_, object)| matches!(object, KObject::Mutex(mutex) if mutex.owner == Some(id)))
            .map(|(object, _)| object)
            .collect();
        for object in held {
            if let Some(KObject::Mutex(mutex)) = self.objects.get_mut(object) {
                log::debug!("thread {} ended holding {}, it goes to whoever waits on it", self.threads[id as usize].name, mutex.name);
                mutex.owner = None;
                mutex.lock_count = 0;
            }
        }
        if self.current_thread == Some(id) {
            self.current_thread = None;
        }
        self.reschedule_pending = true;
    }

    /// signals an event, waking its waiters at the next scheduling point
    pub fn signal_event(&mut self, object: ObjectId) {
        let waited = self.threads.iter().any(|t| t.status == ThreadStatus::WaitSync && t.wait_objects.contains(&object));
        if let Some(KObject::Event(event)) = self.objects.get_mut(object) {
            // a pulse wakes its waiters now and is gone after, lost when nobody waits
            if event.reset_type == ResetType::Pulse {
                if !waited {
                    return;
                }
                self.pulsed.push(object);
            }
            event.signaled = true;
            self.reschedule_pending = true;
        }
    }

    pub fn clear_event(&mut self, object: ObjectId) {
        if let Some(KObject::Event(event)) = self.objects.get_mut(object) {
            event.signaled = false;
        }
    }

    /// makes an event object and a handle in one go, what nearly every caller wants
    pub fn create_event(&mut self, reset_type: ResetType, name: &str) -> (ObjectId, Handle) {
        let id = self
            .objects
            .insert(KObject::Event(sync::Event::new(reset_type, name)));
        let handle = self.handles.create(&mut self.objects, id, name);
        (id, handle)
    }

    /// describes what a blocked thread waits for, for diagnostics
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
                        // who holds a mutex, a waiter stuck on one wants to know
                        let owner = match self.objects.get(object) {
                            Some(KObject::Mutex(mutex)) => match mutex.owner {
                                Some(owner) => {
                                    let holder = self.thread(owner);
                                    let dead = if holder.status == ThreadStatus::Dead { " (dead)" } else { "" };
                                    format!(" held by {}{dead}", holder.name)
                                }
                                None => String::new(),
                            },
                            _ => String::new(),
                        };
                        format!("{label}:{kind}{owner}{}", if signalled { "(ready)" } else { "" })
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

    /// threads not dead, for the diagnostics overlay
    pub fn live_thread_count(&self) -> usize {
        self.threads
            .iter()
            .filter(|t| t.status != ThreadStatus::Dead)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// a thread ending while holding a mutex (however many times) gives it back so a waiter wakes and gets it, like the console
    /// Project Mirai DX hung on its first screen w/o this
    #[test]
    fn a_thread_that_ends_lets_go_of_its_mutexes() {
        let mut kernel = Kernel::new(0, MemoryRegion::Application, 0x1400_0000);
        let holder = kernel.create_thread("holder", 0x0010_0000, 0x1000_0000, 0, 0x30, 0);
        let waiter = kernel.create_thread("waiter", 0x0010_0000, 0x0FF0_0000, 0, 0x30, 0);
        let mut mutex = sync::Mutex::new("Mutex");
        mutex.owner = Some(holder);
        mutex.lock_count = 2;
        let object = kernel.objects.insert(KObject::Mutex(mutex));
        // a handle the title holds plus the wait's own hold, like begin_wait takes it
        kernel.objects.add_ref(object);
        kernel.objects.add_ref(object);
        let thread = &mut kernel.threads[waiter as usize];
        thread.wait_objects = vec![object];
        thread.status = ThreadStatus::WaitSync;
        let mut cpu = Cpu::new();

        // held, the waiter stays blocked
        kernel.schedule(&mut cpu, 0);
        assert_eq!(kernel.thread(waiter).status, ThreadStatus::WaitSync);

        kernel.end_thread(holder);
        assert_eq!(kernel.thread(holder).status, ThreadStatus::Dead);
        let Some(KObject::Mutex(mutex)) = kernel.objects.get(object) else { panic!("the mutex is gone") };
        assert_eq!((mutex.owner, mutex.lock_count), (None, 0));

        kernel.schedule(&mut cpu, 1);
        let thread = kernel.thread(waiter);
        assert!(!thread.status.is_blocked());
        assert!(matches!(thread.wait_result, Some(WaitResult::Signaled(0))));
        let Some(KObject::Mutex(mutex)) = kernel.objects.get(object) else { panic!("the mutex is gone") };
        assert_eq!((mutex.owner, mutex.lock_count), (Some(waiter), 1));
    }

    /// an ended thread gives its TLS block to the next thread, and its slot too once its handle is closed, w/ a new id for the title
    /// Pokemon Sun starts and ends a thread every other frame
    #[test]
    fn a_thread_that_ended_gives_its_place_to_the_next() {
        let mut kernel = Kernel::new(0, MemoryRegion::Application, 0x1400_0000);
        kernel.create_thread("main", 0x0010_0000, 0x1000_0000, 0, 0x30, 0);
        let first = kernel.create_thread("first", 0x0010_0000, 0x0FF0_0000, 0, 0x30, 0);
        let handle = kernel.thread_handle(first);
        let tls = kernel.thread(first).tls;
        kernel.end_thread(first);

        // handle still open, only the TLS block goes to the next
        let second = kernel.create_thread("second", 0x0010_0000, 0x0FF0_0000, 0, 0x30, 0);
        assert_ne!(second, first);
        assert_eq!(kernel.thread(second).tls, tls);
        kernel.end_thread(second);

        assert!(kernel.handles.close(&mut kernel.objects, handle));
        let third = kernel.create_thread("third", 0x0010_0000, 0x0FF0_0000, 0, 0x30, 0);
        assert_eq!(third, first);
        assert_eq!(kernel.thread(third).status, ThreadStatus::Ready);
        assert_eq!(kernel.thread(third).guest_id, 3);
        assert_eq!(kernel.thread(third).tls, tls);
        // the new thread's own object, not the ended one's
        let object = kernel.thread_object(third);
        assert_eq!(kernel.objects.refcount(object), 1);
    }
}
