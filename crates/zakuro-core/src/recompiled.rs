//! runs code 3dsrecomp turned into a lib ahead of time
//! the lib's funcs work on a context holding the regs and reach memory thru the same page tables as the interpreter
//! calling back here for anything else, whatever the lib has no code for stays w/ the interpreter

use std::cell::Cell;
use std::ffi::c_void;
use std::path::Path;

use recomp_abi::{Code, Context, Host, EXIT_BUDGET, EXIT_SVC, EXIT_UNWIND, NOTHING_HELD, TRAIL};
use zakuro_cpu::{Bus, Cpu, Exit};

use crate::kernel::object::ObjectId;
use crate::memory::Memory;
use crate::System;

pub use recomp_abi::Linked;

/// the lib `3dsrecomp build` installed for a title, if any
pub fn installed(program_id: u64) -> Option<std::path::PathBuf> {
    recomp_abi::installed(program_id)
}

/// is the lib at path worth rebuilding, from an older 3dsrecomp than this build has or not running at all
pub fn outdated(path: &Path) -> bool {
    !recomp_abi::Library::open(path).is_ok_and(|code| code.generation() >= recomp_abi::GENERATION)
}

/// recompiled code and how much work it handed back
pub struct Library {
    code: recomp_abi::Library,
    /// instrs the code handed to the interpreter one at a time
    fallbacks: Cell<u64>,
    /// svcs the code ran in place and went on after
    in_place: Cell<u64>,
    /// where the code reads the clock itself (see recomp_abi::Context), the last spot tick_in_place answered GetSystemTick at
    tick_at: Cell<u32>,
    /// where the code takes and drops its lent mutex (see recomp_abi::Context), the last spots wait_in_place and release_in_place ran at
    wait_at: Cell<u32>,
    release_at: Cell<u32>,
    /// code found for addrs lately, each in a slot its addr picks
    /// a jump thru a reg looks its target up, otherwise that's a search thru every func
    found: Box<[Found]>,
}

/// a Library::found slot, an addr and the code found for it
type Found = Cell<(u32, Option<Code>)>;

/// Library::found slots, a power of two
const FOUND: usize = 65536;

/// the Library::found slot an addr goes in, top bits of addr times the golden ratio
/// ARM addrs step by 4 and Thumb ones by 2 w/ bit 0 set, low bits would leave half the slots to one of them
fn slot(address: u32) -> usize {
    (address.wrapping_mul(0x9E37_79B1) >> (32 - FOUND.trailing_zeros())) as usize
}

/// why a run of recompiled code ended
pub enum Stop {
    /// an svc, the cpu's already past it
    Svc(u32),
    /// the interpreter stopped on an instr the code handed it
    Exit(Exit),
    /// budget ran out, or execution left for code the lib doesn't have
    Left,
    /// after an in place svc the code had no budget for the next block, the interpreter takes it like it would in the step after the svc
    Declined,
}

/// what the callbacks reach thru the context
struct Machine<'a> {
    /// the system the code runs in, the lib's out of it meanwhile
    system: *mut System,
    library: &'a Library,
    /// what the interpreter stopped w/ inside a callback
    pending: Option<Exit>,
    /// cycle count and budget when the code started (or went on after an svc), and instrs it ran b4
    cycles: u64,
    budget: i32,
    ran: u64,
    /// the step's deadline and the next event's tick, the budget stops there
    deadline: Option<u64>,
    event: Option<u64>,
    /// the cycle count the budget runs out at, as last worked out
    limit: u64,
    /// an svc the code ran needed the step to do more, the code left for it
    left: bool,
    /// the code went on after an svc it ran
    went_on: bool,
    /// cycle count a self read GetSystemTick must come b4 (cost included), none when the code reads none
    due: Option<u64>,
    /// the mutex lent to the code (see recomp_abi::Context) and the cycle count a self answered wait or release must come b4
    held: Option<(ObjectId, u64)>,
    /// can the code be lent one, made for interface v8 or later
    lends: bool,
    /// a tick the code would answer itself has to come to the host,
    /// a try at skipping turns of a clock polling loop takes it, see crate::spin
    tick_wanted: bool,
}

impl Machine<'_> {
    /// what tick_in_place, wait_in_place and release_in_place would have done for the svcs the code answered
    /// itself since the host last looked
    /// see recomp_abi::Context, the ones in the trail (last at tick_last) or ticks from tick_at for pre v8 code
    ///
    /// # Safety
    ///
    /// same as machine, and nothing else may hold the system
    #[inline(always)]
    unsafe fn caught_up(&mut self, ctx: &mut Context) {
        // mostly the code answered nothing since
        if ctx.done != 0 || ctx.ticks != 0 {
            unsafe { self.catch_up(ctx) };
        }
    }

    /// caught_up, when the code answered smth
    ///
    /// # Safety
    ///
    /// same as machine, and nothing else may hold the system
    unsafe fn catch_up(&mut self, ctx: &mut Context) {
        let done = std::mem::take(&mut ctx.done);
        let ticks = std::mem::take(&mut ctx.ticks);
        let cost = ctx.tick_cost as u64;
        // a tick (the last one) costs, a wait or release costs nothing
        let last_tick = done == 0 || ctx.trail[(done - 1) as usize % TRAIL] == ctx.tick_at;
        let after = ctx.tick_last + if last_tick { cost } else { 0 };
        let budget = ctx.tick_end.saturating_sub(after).min(i32::MAX as u64) as i32;
        // what ran b4 the last svc, ticks aside
        self.ran += (self.budget.saturating_sub(budget).max(0) as u64).saturating_sub(cost * ticks as u64);
        self.cycles = after;
        self.budget = budget;
        self.went_on = true;
        self.library.in_place.set(self.library.in_place.get() + done.max(ticks) as u64);
        let system = unsafe { &mut *self.system };
        if let Some((object, _)) = self.held {
            system.kernel.held_as(object, ctx.holding);
        }
        match done {
            0 => system.ticks_in_place(ctx.tick_at, ticks, ctx.tick_last),
            _ => system.answered_in_place(&ctx.trail, done, ctx.tick_last, after),
        }
    }

    /// takes back the mutex lent to the code, the host can do anything to it now
    #[inline(always)]
    fn take_back(&mut self, ctx: &mut Context) {
        self.held = None;
        ctx.held_floor = NOTHING_HELD;
    }

    /// lends the code the mutex a wait or release ran in place at at took or dropped alone (see recomp_abi::Context)
    /// when nothing but the code's thread can touch it for a while, the mutex is a release's handle or where a wait's handle is
    ///
    /// # Safety
    ///
    /// same as machine, and nothing else may hold the system
    unsafe fn lend(&mut self, ctx: &mut Context, (mutex, wait): (u32, bool), at: u32) {
        let system = unsafe { &*self.system };
        let Some(due) = self.due else { return };
        let handle = match wait {
            true => match system.memory.plain_read32(mutex) {
                Some(handle) => handle,
                None => return,
            },
            false => mutex,
        };
        let Some((object, holding, until)) = system.kernel.lendable(handle, system.cpu.cycles) else { return };
        self.held = Some((object, due.min(until)));
        ctx.held = handle;
        ctx.holding = holding;
        let (site, kept) = if wait { (&mut ctx.wait_at, &self.library.wait_at) } else { (&mut ctx.release_at, &self.library.release_at) };
        *site = at;
        kept.set(at);
        self.held_from(ctx);
    }

    /// where the code may answer a wait or release itself from here, w/ the cycle count and budget the host just gave it
    /// one at t is clear of the mutex's due when t < due, t being tick_end minus the budget
    fn held_from(&self, ctx: &mut Context) {
        ctx.held_floor = self.held.map_or(NOTHING_HELD, |(_, due)| {
            let floor = ctx.tick_end as i64 - due as i64;
            floor.clamp(i32::MIN as i64, i32::MAX as i64) as i32
        });
    }

    /// where the code may answer GetSystemTick itself from here, w/ the cycle count and budget the host just gave it
    /// a tick at t is clear of due when t + cost < due, t being tick_end minus the budget
    fn ticks_from(&self, ctx: &mut Context) {
        ctx.tick_end = self.cycles + self.budget.max(0) as u64;
        ctx.tick_floor = self.due.map_or(i32::MAX, |due| {
            let floor = ctx.tick_end as i64 + ctx.tick_cost as i64 - due as i64;
            floor.clamp(i32::MIN as i64, i32::MAX as i64) as i32
        });
        self.held_from(ctx);
    }
}

/// # Safety
///
/// ctx must be a context run set up, its user field pointing at a live Machine
unsafe fn machine<'a>(ctx: *mut Context) -> &'a mut Machine<'a> {
    unsafe { &mut *((*ctx).user as *mut Machine) }
}

/// the memory the code runs on
///
/// # Safety
///
/// same as machine, and nothing else may hold the system
unsafe fn memory<'a>(ctx: *mut Context) -> &'a mut Memory {
    unsafe { &mut (*machine(ctx).system).memory }
}

unsafe extern "C" fn read8(ctx: *mut Context, address: u32) -> u8 {
    unsafe { memory(ctx).read8(address) }
}

unsafe extern "C" fn read16(ctx: *mut Context, address: u32) -> u16 {
    unsafe { memory(ctx).read16(address) }
}

unsafe extern "C" fn read32(ctx: *mut Context, address: u32) -> u32 {
    unsafe { memory(ctx).read32(address) }
}

unsafe extern "C" fn write8(ctx: *mut Context, address: u32, value: u8) {
    unsafe { memory(ctx).write8(address, value) }
}

unsafe extern "C" fn write16(ctx: *mut Context, address: u32, value: u16) {
    unsafe { memory(ctx).write16(address, value) }
}

unsafe extern "C" fn write32(ctx: *mut Context, address: u32, value: u32) {
    unsafe { memory(ctx).write32(address, value) }
}

/// runs one instr the code left to the interpreter
unsafe extern "C" fn interpret(ctx: *mut Context, address: u32, _opcode: u32) {
    unsafe {
        let machine = machine(ctx);
        machine.library.fallbacks.set(machine.library.fallbacks.get() + 1);
        let ctx = &mut *ctx;
        machine.caught_up(ctx);
        machine.take_back(ctx);
        let system = &mut *machine.system;
        load(ctx, &mut system.cpu);
        system.cpu.regs[15] = address;
        let exit = system.cpu.step(&mut system.memory);
        store(&system.cpu, ctx);
        let step = if ctx.thumb != 0 { 2 } else { 4 };
        if exit.is_some() || ctx.r[15] != address.wrapping_add(step) {
            machine.pending = exit;
            ctx.exit = EXIT_UNWIND;
        }
    }
}

unsafe extern "C" fn lookup(ctx: *mut Context, address: u32) -> Option<Code> {
    unsafe { machine(ctx).library.lookup(address) }
}

/// do svcs run in place, unless ZAKURO_SVC_IN_PLACE is 0, then the code stops at every one like b4
pub(crate) fn svcs_in_place() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ZAKURO_SVC_IN_PLACE").map_or(true, |value| value != "0"))
}

/// runs an svc the code made, the code goes on after it when the next step would just run it on (see System::svc_in_place)
/// one that can't run in place stops the code for the step to run it
unsafe extern "C" fn svc(ctx: *mut Context, number: u32) {
    unsafe {
        let machine = machine(ctx);
        let ctx = &mut *ctx;
        machine.caught_up(ctx);
        let lent = machine.held.is_some();
        machine.take_back(ctx);
        if !svcs_in_place() || !crate::kernel::svc::in_place(number) {
            ctx.svc = number;
            ctx.exit = EXIT_SVC;
            return;
        }
        let system = &mut *machine.system;
        // as far as the step goes the code stopped at the svc
        let ran = machine.budget.saturating_sub(ctx.budget).max(0) as u64;
        machine.ran += ran;
        // the clock, an instant wait and a dropped mutex, what titles call more than anything
        // w/o the rest of a step around them when that'd change nothing
        let cycles = machine.cycles + ran;
        let at = ctx.r[15] | ctx.thumb as u32;
        let (event, deadline, limit, library) = (machine.event, machine.deadline, machine.limit, machine.library);
        let preempt = system.next_preempt;
        let mut tick_at = None;
        // a mutex taken or dropped alone, the code may then do it itself, see Machine::lend
        let mut mutex = None;
        let done = match number {
            0x28 => {
                // whole turns of a clock polling loop get skipped first, see crate::spin
                let mut cycles = cycles;
                if let Some(period) = system.spin.due(cycles).then(|| system.spin_period(at)).flatten() {
                    let core = Core::of(ctx, &system.cpu);
                    if let Some(skipped) = system.spin_skip(&core, cycles, at, period, event, deadline, library) {
                        machine.ran += skipped.instructions;
                        library.in_place.set(library.in_place.get() + skipped.svcs);
                        cycles += skipped.cycles;
                    }
                }
                system.tick_in_place(cycles, at, event, deadline, limit, library).inspect(|_| {
                    ctx.r[0] = cycles as u32;
                    ctx.r[1] = (cycles >> 32) as u32;
                    // the code reads the clock itself here from now on
                    tick_at = Some(at);
                })
            }
            0x25 => {
                let regs = [ctx.r[0], ctx.r[1], ctx.r[2], ctx.r[3], ctx.r[4]];
                system.wait_in_place(cycles, at, regs, event, deadline, limit, library).map(|(budget, index)| {
                    ctx.r[0] = 0;
                    ctx.r[1] = index;
                    mutex = (regs[2] == 1).then_some((regs[1], true));
                    budget
                })
            }
            0x14 => {
                let handle = ctx.r[0];
                system.release_in_place(cycles, at, handle, event, deadline, limit, library).inspect(|_| {
                    ctx.r[0] = 0;
                    mutex = Some((handle, false));
                })
            }
            _ => None,
        };
        if let Some(budget) = done {
            machine.cycles = system.cpu.cycles;
            ctx.budget = budget.min(i32::MAX as u64) as i32;
            machine.went_on = true;
            library.in_place.set(library.in_place.get() + 1);
            machine.budget = ctx.budget;
            // nothing it looks at changed, unless the budget got cut short,
            // a preempt passed (moving the next one) or a skip try wanted this tick
            let passed = system.next_preempt != preempt;
            if passed {
                machine.due = system.tick_due(event, deadline);
            }
            let wanted = number == 0x28 && std::mem::take(&mut machine.tick_wanted);
            if passed || wanted || machine.cycles + machine.budget as u64 != ctx.tick_end {
                machine.ticks_from(ctx);
            }
            if let Some(at) = tick_at {
                ctx.tick_at = at;
                library.tick_at.set(at);
            }
            if let Some(mutex) = mutex.filter(|_| machine.lends) {
                machine.lend(ctx, mutex, at);
            } else if lent && tick_at.is_some() {
                // a tick changes nothing the mutex was lent on, nor does a preempt's pass it hit
                // so the code keeps it instead of coming back w/ its next wait to get it lent again
                let (handle, release_at) = (ctx.held, ctx.release_at);
                machine.lend(ctx, (handle, false), release_at);
            }
            // a try at skipping turns of a clock polling loop is due (the history shows one), it takes the next tick, see crate::spin
            // a turn's other svcs may be the ones hitting the preempts every time,
            // or a try was at another of the turn's ticks than its first
            let wants = match number {
                0x28 => system.spin.wants_tick(),
                _ => system.spin.due(machine.cycles) && system.history_repeats(),
            };
            if wants {
                ctx.tick_floor = i32::MAX;
                machine.tick_wanted = true;
            }
            return;
        }
        load(ctx, &mut system.cpu);
        system.cpu.cycles = machine.cycles + ran;
        let next = system.svc_in_place(number, machine.event, machine.deadline, machine.library);
        store(&system.cpu, ctx);
        machine.cycles = system.cpu.cycles;
        match next {
            Some((budget, event, limit)) => {
                ctx.budget = budget.min(i32::MAX as u64) as i32;
                machine.event = event;
                machine.limit = limit;
                machine.went_on = true;
                machine.library.in_place.set(machine.library.in_place.get() + 1);
                machine.budget = ctx.budget;
                machine.due = (*machine.system).tick_due(event, machine.deadline);
                machine.ticks_from(ctx);
            }
            None => {
                // nothing's left of the run but unwinding, its event already looked at
                machine.event = None;
                machine.left = true;
                ctx.exit = EXIT_UNWIND;
            }
        }
        machine.budget = ctx.budget;
    }
}

static HOST: Host = Host { read8, read16, read32, write8, write16, write32, interpret, lookup, svc };

fn load(ctx: &Context, cpu: &mut Cpu) {
    cpu.regs = ctx.r;
    cpu.cpsr.n = ctx.n != 0;
    cpu.cpsr.z = ctx.z != 0;
    cpu.cpsr.c = ctx.c != 0;
    cpu.cpsr.v = ctx.v != 0;
    cpu.cpsr.q = ctx.q != 0;
    cpu.cpsr.ge = ctx.ge;
    cpu.cpsr.thumb = ctx.thumb != 0;
    cpu.exclusive_addr = (ctx.exclusive != 0).then_some(ctx.exclusive_address);
}

fn store(cpu: &Cpu, ctx: &mut Context) {
    ctx.r = cpu.regs;
    ctx.n = cpu.cpsr.n as u8;
    ctx.z = cpu.cpsr.z as u8;
    ctx.c = cpu.cpsr.c as u8;
    ctx.v = cpu.cpsr.v as u8;
    ctx.q = cpu.cpsr.q as u8;
    ctx.ge = cpu.cpsr.ge;
    ctx.thumb = cpu.cpsr.thumb as u8;
    ctx.exclusive = cpu.exclusive_addr.is_some() as u8;
    ctx.exclusive_address = cpu.exclusive_addr.unwrap_or(0);
    ctx.tls = cpu.cp15.thread_id_ro;
}

impl Library {
    /// a lib 3dsrecomp built
    pub fn open(path: &Path) -> Result<Library, String> {
        recomp_abi::Library::open(path).map(Library::new)
    }

    /// the code linked into the program
    pub fn linked(linked: &Linked) -> Result<Library, String> {
        recomp_abi::Library::linked(linked).map(Library::new)
    }

    fn new(mut code: recomp_abi::Library) -> Library {
        // ZAKURO_NATIVES=0 runs the title's own code for the C lib funcs the lib does on the host
        if std::env::var("ZAKURO_NATIVES").is_ok_and(|value| value == "0") && code.set_natives(false) {
            log::info!("the library's natives are off, the title's own code runs for them");
        }
        Library {
            code,
            fallbacks: Cell::new(0),
            in_place: Cell::new(0),
            tick_at: Cell::new(recomp_abi::NO_TICKS),
            wait_at: Cell::new(recomp_abi::NO_TICKS),
            release_at: Cell::new(recomp_abi::NO_TICKS),
            found: Library::nothing_found(),
        }
    }

    /// every found slot empty, w/ an addr no code starts at in each
    fn nothing_found() -> Box<[Found]> {
        (0..FOUND).map(|_| Cell::new((u32::MAX, None))).collect()
    }

    /// did the code come from an older 3dsrecomp than this build has, so recompiling makes it better
    pub fn outdated(&self) -> bool {
        self.code.generation() < recomp_abi::GENERATION
    }

    /// can the code tell which funcs changed in memory, gen 2 and up can
    pub fn checks_itself(&self) -> bool {
        self.code.origins(0).is_some()
    }

    /// instrs the code handed to the interpreter
    pub fn fallbacks(&self) -> u64 {
        self.fallbacks.get()
    }

    /// svcs the code ran in place and went on after
    pub fn in_place(&self) -> u64 {
        self.in_place.get()
    }

    /// how many funcs and modules have code
    pub fn describe(&self) -> String {
        self.code.describe()
    }

    /// code that can run from address (bit 0 for Thumb) in the exe or a loaded module
    fn lookup(&self, address: u32) -> Option<Code> {
        let slot = &self.found[slot(address)];
        match slot.get() {
            (seen, code) if seen == address => code,
            _ => {
                let code = self.code.lookup(address);
                slot.set((address, code));
                code
            }
        }
    }

    pub fn has_code(&self, address: u32) -> bool {
        self.lookup(address).is_some()
    }

    /// marks the exe's funcs whose code in memory isn't what they were recompiled from (a mod or another game version does that)
    /// so the interpreter runs them, returns how many
    pub fn check(&mut self, memory: &mut Memory) -> usize {
        let stale = self.code.check(0, 0, |address, bytes| memory.read_bytes(address, bytes));
        self.found = Library::nothing_found();
        stale
    }

    /// tells module name's code where the title loaded it (0 on unload) and checks its code once it's in memory
    pub fn place(&mut self, name: &str, base: u32, memory: &mut Memory) {
        let Some(index) = self.code.module_index(name) else { return };
        self.code.place(index, base);
        // what was found b4 may have moved or gone
        self.found = Library::nothing_found();
        if base == 0 {
            return;
        }
        log::info!("recompiled code for {name} runs at 0x{base:08X}");
        let stale = self.code.check(index + 1, base, |address, bytes| memory.read_bytes(address, bytes));
        if stale > 0 {
            log::warn!("{stale} functions of {name} changed since it was recompiled, they run in the interpreter");
        }
    }

    /// runs recompiled code from the cpu's pc for about budget instrs, returns how many ran,
    /// why it stopped and the tick its budget last stopped at
    /// the code runs what svcs it can in place, deadline, event and limit are what the step had for those
    /// the lib must be out of the system, the code gets it to itself
    pub fn run(&self, system: &mut System, budget: u64, deadline: Option<u64>, event: Option<u64>, limit: u64) -> (u64, Stop, Option<u64>) {
        let system: *mut System = system;
        let budget = budget.min(i32::MAX as u64) as i32;
        // safe bc everything the code reaches of the system comes from this one ptr, nothing else uses it till the run's over
        let (cycles, read_pages, write_pages, vfp, fpscr) = unsafe {
            let (read_pages, write_pages) = (*system).memory.page_tables();
            // the code works on the VFP regs in place so the interpreter sees its changes w/o copying them around
            let vfp = &raw mut (*system).cpu.vfp.regs as *mut u32;
            let fpscr = &raw mut (*system).cpu.vfp.fpscr;
            ((*system).cpu.cycles, read_pages, write_pages, vfp, fpscr)
        };
        let mut ctx = Context {
            r: [0; 16],
            n: 0,
            z: 0,
            c: 0,
            v: 0,
            q: 0,
            thumb: 0,
            ge: 0,
            exclusive: 0,
            budget,
            exit: 0,
            svc: 0,
            depth: 0,
            exclusive_address: 0,
            tls: 0,
            read_pages,
            write_pages,
            vfp,
            fpscr,
            host: &HOST,
            user: std::ptr::null_mut(),
            stack_limit: 0,
            tick_end: 0,
            tick_last: 0,
            tick_floor: i32::MAX,
            tick_cost: crate::kernel::svc::SYSTEM_TICK_COST as i32,
            tick_at: recomp_abi::NO_TICKS,
            ticks: 0,
            wait_at: recomp_abi::NO_TICKS,
            release_at: recomp_abi::NO_TICKS,
            held: 0,
            holding: 0,
            held_floor: NOTHING_HELD,
            done: 0,
            trail: [0; TRAIL],
        };
        // safe, same as above
        store(unsafe { &(*system).cpu }, &mut ctx);
        // the code reads the clock, takes and drops its lent mutex itself where it last did,
        // while that's still code to run (only the step changes it)
        // safe, same as above
        let runs = |at: u32| at != recomp_abi::NO_TICKS && unsafe { (*system).memory.is_executable(at & !1) } && self.has_code(at);
        for (site, kept) in [(&mut ctx.tick_at, &self.tick_at), (&mut ctx.wait_at, &self.wait_at), (&mut ctx.release_at, &self.release_at)] {
            if runs(kept.get()) {
                *site = kept.get();
            }
        }
        // safe, same as above
        let due = unsafe { (*system).tick_due(event, deadline) };
        let lends = self.code.abi() >= 8;
        let mut machine = Machine {
            system,
            library: self,
            pending: None,
            cycles,
            budget,
            ran: 0,
            deadline,
            event,
            limit,
            left: false,
            went_on: false,
            due,
            held: None,
            lends,
            tick_wanted: false,
        };
        ctx.user = &mut machine as *mut Machine as *mut c_void;
        machine.ticks_from(&mut ctx);

        let stop = loop {
            if ctx.budget <= 0 {
                break Stop::Left;
            }
            let Some(code) = self.lookup(ctx.r[15] | ctx.thumb as u32) else {
                break Stop::Left;
            };
            ctx.exit = 0;
            ctx.depth = 0;
            ctx.stack_limit = recomp_abi::stack_limit();
            // safe bc the code came from the lib and ctx is set up how it expects, user pointing at machine
            // where it went is in the context, its return value is just for other code
            unsafe { code(&mut ctx) };
            // safe, same as above, the code returned
            unsafe { machine.caught_up(&mut ctx) };
            if let Some(exit) = machine.pending.take() {
                break Stop::Exit(exit);
            }
            // the step does the rest, the svc ran
            if machine.left {
                break Stop::Left;
            }
            match ctx.exit {
                EXIT_SVC => break Stop::Svc(ctx.svc),
                EXIT_BUDGET if machine.went_on && ctx.budget == machine.budget => break Stop::Declined,
                EXIT_BUDGET => break Stop::Left,
                _ => {}
            }
        };

        // safe bc the code's done w/ the system
        let cpu = unsafe { &mut (*system).cpu };
        load(&ctx, cpu);
        let ran = machine.budget.saturating_sub(ctx.budget.max(0)).max(0) as u64;
        // the interpreter counted the instrs it ran for the code, the budget has them already
        cpu.cycles = machine.cycles + ran;
        (machine.ran + ran, stop, machine.event)
    }
}

// turns run on the side, see crate::spin

/// recompiled code's core at an svc, all a side turn starts from
/// regs, flags, monitor and VFP regs, which the code works on where the system's cpu has them
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct Core {
    pub r: [u32; 16],
    /// n, z, c, v, q, thumb, ge and exclusive, as the context has them
    flags: [u8; 8],
    exclusive_address: u32,
    tls: u32,
    vfp: [u32; 32],
    fpscr: u32,
}

impl Core {
    /// the core the code has at an svc, the context's plus the cpu's VFP regs
    pub(crate) fn of(ctx: &Context, cpu: &Cpu) -> Core {
        Core {
            r: ctx.r,
            flags: [ctx.n, ctx.z, ctx.c, ctx.v, ctx.q, ctx.thumb, ctx.ge, ctx.exclusive],
            exclusive_address: ctx.exclusive_address,
            tls: ctx.tls,
            vfp: cpu.vfp.regs,
            fpscr: cpu.vfp.fpscr,
        }
    }

    pub(crate) fn thumb(&self) -> bool {
        self.flags[5] != 0
    }
}

/// where side turns write, a write table w/ no pages so their code writes every byte thru the host
/// plus the pages they really wrote w/ the bytes written, and what each byte held b4, to put back after
pub(crate) struct Sandbox {
    table: Vec<*mut u8>,
    /// pages written since put_back, the first used ones, the rest kept for the next w/ no byte written
    touched: Vec<Touched>,
    used: usize,
    /// the one written last
    last: usize,
    /// each byte written since put_back, once, w/ what it held b4 the first write, minus what the interpreter wrote for their code b4 that
    old: Vec<Old>,
}

/// bytes written in a page, a bit each
pub(crate) type Bytes = [u64; PAGE_SIZE / 64];

/// a page side turns wrote
struct Touched {
    page: usize,
    at: *mut u8,
    written: Bytes,
}

/// a byte side turns wrote, where the host has it, its addr and what it held
struct Old {
    at: *mut u8,
    address: u32,
    old: u8,
}

const PAGE_SIZE: usize = zakuro_common::memory_map::PAGE_SIZE as usize;
const PAGE_BITS: u32 = zakuro_common::memory_map::PAGE_BITS;

impl Sandbox {
    pub(crate) fn new() -> Sandbox {
        Sandbox {
            table: vec![std::ptr::null_mut(); zakuro_common::memory_map::PAGE_TABLE_ENTRIES],
            touched: Vec::new(),
            used: 0,
            last: 0,
            old: Vec::new(),
        }
    }

    /// the page written at (the cpu writes it plainly at real), noted from now on, or none when it can't
    #[inline]
    fn page(&mut self, page: usize, real: *const *mut u8) -> Option<&mut Touched> {
        if self.last < self.used && self.touched[self.last].page == page {
            return Some(&mut self.touched[self.last]);
        }
        self.other_page(page, real)
    }

    fn other_page(&mut self, page: usize, real: *const *mut u8) -> Option<&mut Touched> {
        if let Some(index) = self.touched[..self.used].iter().position(|touched| touched.page == page) {
            self.last = index;
            return Some(&mut self.touched[index]);
        }
        // safe bc the cpu's write table has an entry for every page
        let at = unsafe { *real.add(page) };
        if at.is_null() {
            return None;
        }
        if self.used == self.touched.len() {
            self.touched.push(Touched { page, at, written: [0; PAGE_SIZE / 64] });
        }
        let touched = &mut self.touched[self.used];
        (touched.page, touched.at) = (page, at);
        self.last = self.used;
        self.used += 1;
        Some(touched)
    }

    /// each byte side turns since put_back changed (of those written) w/ what it holds now, by addr
    pub(crate) fn changed(&self, memory: &mut Memory) -> Vec<(u32, u8)> {
        // safe bc a written byte is in one of the cpu's pages, there while the turns ran
        let mut changed: Vec<(u32, u8)> =
            self.old.iter().filter_map(|old| Some((old.address, unsafe { *old.at })).filter(|&(_, now)| now != old.old)).collect();
        if memory.logged_any() {
            for (addr, old) in memory.logged_old() {
                let mut now = [0u8];
                memory.read_bytes(addr, &mut now);
                if now[0] != old {
                    changed.push((addr, now[0]));
                }
            }
        }
        changed.sort_unstable();
        changed.dedup();
        changed
    }

    /// bytes side turns since put_back wrote (the interpreter's for their code too) by page
    pub(crate) fn written(&mut self, memory: &Memory) -> Vec<(u32, Bytes)> {
        self.touched[..self.used].sort_unstable_by_key(|touched| touched.page);
        self.last = 0;
        let mut written: Vec<(u32, Bytes)> = self.touched[..self.used].iter().map(|touched| (touched.page as u32, touched.written)).collect();
        if memory.logged_any() {
            for (addr, _) in memory.logged_old() {
                let page = addr >> PAGE_BITS;
                let index = match written.binary_search_by_key(&page, |&(page, _)| page) {
                    Ok(index) => index,
                    Err(index) => {
                        written.insert(index, (page, [0; PAGE_SIZE / 64]));
                        index
                    }
                };
                let offset = (addr as usize) % PAGE_SIZE;
                written[index].1[offset / 64] |= 1 << (offset % 64);
            }
        }
        written
    }

    /// puts back what the side turns since wrote (what the interpreter wrote for their code first) and forgets them
    pub(crate) fn put_back(&mut self, memory: &mut Memory) {
        memory.undo_writes();
        for old in self.old.drain(..) {
            // safe, same as in changed
            unsafe { *old.at = old.old };
        }
        for touched in &mut self.touched[..self.used] {
            touched.written = [0; PAGE_SIZE / 64];
        }
        (self.used, self.last) = (0, 0);
    }
}

/// a side turn, its svcs (addr after each w/ bit 0 for Thumb, cycles from the start each came at, its number)
/// the cycles it took and the core the next turn starts from
pub(crate) struct Side {
    pub svcs: Vec<(u32, u64, u32)>,
    pub cost: u64,
    pub end: Core,
}

/// a turn running on the side
struct Run<'a> {
    system: *mut System,
    library: &'a Library,
    sandbox: *mut Sandbox,
    /// where the turn starts, pc after its svc, Thumb, sp and lr
    start: (u32, u8, u32, u32),
    period: usize,
    tick: u64,
    /// budget the code started w/, b4 the first svc's cost
    from: i32,
    svcs: Vec<(u32, u64, u32)>,
    mutex: Option<u32>,
    holding: u32,
    /// cycles it took and the core the next turn starts from, once it got there
    end: Option<(u64, Core)>,
    failed: bool,
    vfp: [u32; 32],
    fpscr: u32,
}

/// the turn running on the side, the context's user points at it
unsafe fn run<'a>(ctx: *mut Context) -> &'a mut Run<'a> {
    unsafe { &mut *((*ctx).user as *mut Run) }
}

impl Run<'_> {
    /// writes the len bytes of value at address for real, noting each w/ what it held b4 the first write
    /// a page the cpu can't write plainly (guarded, watched, missing) isn't a polling loop's
    #[inline]
    fn write(&mut self, address: u32, value: u32, len: usize) {
        // safe bc the system and sandbox outlive the run and nothing else holds them meanwhile
        let (system, sandbox) = unsafe { (&mut *self.system, &mut *self.sandbox) };
        let offset = address as usize % PAGE_SIZE;
        // within a page and one word of its bytes, like nearly all are
        if offset % 64 + len > 64 {
            return self.write_bytes(address, value, len);
        }
        let (_, real) = system.memory.page_tables();
        let Some(touched) = sandbox.page((address >> PAGE_BITS) as usize, real) else {
            self.failed = true;
            return;
        };
        let at = touched.at;
        let (word, mask) = (offset / 64, ((1u64 << len) - 1) << (offset % 64));
        let mut fresh = mask & !touched.written[word];
        if fresh != 0 {
            touched.written[word] |= mask;
            let logged = system.memory.logged_any();
            while fresh != 0 {
                let byte = word * 64 + fresh.trailing_zeros() as usize;
                fresh &= fresh - 1;
                let address = (address & !(PAGE_SIZE as u32 - 1)) | byte as u32;
                // what the interpreter wrote there for the code already isn't what the byte held
                // safe bc a touched page is a whole one of the cpu's
                let old = unsafe { *at.add(byte) };
                let old = if logged { system.memory.logged_old_at(address).unwrap_or(old) } else { old };
                sandbox.old.push(Old { at: unsafe { at.add(byte) }, address, old });
            }
        }
        // safe, same as above, the bytes are inside the page
        unsafe { std::ptr::copy_nonoverlapping(value.to_le_bytes().as_ptr(), at.add(offset), len) };
    }

    /// write, byte by byte, for bytes across pages or words
    #[cold]
    fn write_bytes(&mut self, address: u32, value: u32, len: usize) {
        for (offset, &byte) in value.to_le_bytes()[..len].iter().enumerate() {
            self.write(address.wrapping_add(offset as u32), byte as u32, 1);
        }
    }

    fn core(&self, ctx: &Context) -> Core {
        Core {
            r: ctx.r,
            flags: [ctx.n, ctx.z, ctx.c, ctx.v, ctx.q, ctx.thumb, ctx.ge, ctx.exclusive],
            exclusive_address: ctx.exclusive_address,
            tls: ctx.tls,
            vfp: self.vfp,
            fpscr: self.fpscr,
        }
    }
}

unsafe extern "C" fn side_read8(ctx: *mut Context, address: u32) -> u8 {
    unsafe { (*run(ctx).system).memory.read8(address) }
}

unsafe extern "C" fn side_read16(ctx: *mut Context, address: u32) -> u16 {
    unsafe { (*run(ctx).system).memory.read16(address) }
}

unsafe extern "C" fn side_read32(ctx: *mut Context, address: u32) -> u32 {
    unsafe { (*run(ctx).system).memory.read32(address) }
}

unsafe extern "C" fn side_write8(ctx: *mut Context, address: u32, value: u8) {
    unsafe { run(ctx).write(address, value as u32, 1) }
}

unsafe extern "C" fn side_write16(ctx: *mut Context, address: u32, value: u16) {
    unsafe { run(ctx).write(address, value as u32, 2) }
}

unsafe extern "C" fn side_write32(ctx: *mut Context, address: u32, value: u32) {
    unsafe { run(ctx).write(address, value, 4) }
}

/// memory as the interpreter reaches it for a side turn
/// its own type so the interpreter the steps run stays as compiled, it gets inlined where it's called
struct SideBus<'a>(&'a mut Memory);

impl Bus for SideBus<'_> {
    fn read8(&mut self, addr: u32) -> u8 {
        self.0.read8(addr)
    }

    fn read16(&mut self, addr: u32) -> u16 {
        self.0.read16(addr)
    }

    fn read32(&mut self, addr: u32) -> u32 {
        self.0.read32(addr)
    }

    fn write8(&mut self, addr: u32, value: u8) {
        self.0.write8(addr, value)
    }

    fn write16(&mut self, addr: u32, value: u16) {
        self.0.write16(addr, value)
    }

    fn write32(&mut self, addr: u32, value: u32) {
        self.0.write32(addr, value)
    }
}

/// an instr the code leaves to the interpreter, its writes noted to undo w/ the rest
unsafe extern "C" fn side_interpret(ctx: *mut Context, address: u32, _opcode: u32) {
    unsafe {
        let run = run(ctx);
        let ctx = &mut *ctx;
        let system = &mut *run.system;
        let mut cpu = system.cpu.clone();
        load(ctx, &mut cpu);
        (cpu.vfp.regs, cpu.vfp.fpscr) = (run.vfp, run.fpscr);
        cpu.regs[15] = address;
        let exit = cpu.step(&mut SideBus(&mut system.memory));
        store(&cpu, ctx);
        (run.vfp, run.fpscr) = (cpu.vfp.regs, cpu.vfp.fpscr);
        let step = if ctx.thumb != 0 { 2 } else { 4 };
        if exit.is_some() || ctx.r[15] != address.wrapping_add(step) {
            run.failed = true;
            ctx.exit = EXIT_UNWIND;
        }
    }
}

unsafe extern "C" fn side_lookup(ctx: *mut Context, address: u32) -> Option<Code> {
    unsafe { run(ctx).library.lookup(address) }
}

/// an svc of a side turn, the clock read as the code would see it then
/// a wait on the mutex alone takes it and its release drops it, counted separately
/// the one the next turn starts at ends it, any other svc (or one where the turn would wait) fails it
unsafe extern "C" fn side_svc(ctx: *mut Context, number: u32) {
    unsafe {
        let run = run(ctx);
        let ctx = &mut *ctx;
        let system = &mut *run.system;
        let elapsed = (run.from - ctx.budget) as u64;
        if run.svcs.len() == run.period {
            let back = number == 0x28 && (ctx.r[15], ctx.thumb, ctx.r[13], ctx.r[14]) == run.start && run.holding == 0;
            match back {
                true => run.end = Some((elapsed, run.core(ctx))),
                false => run.failed = true,
            }
            ctx.exit = EXIT_UNWIND;
            return;
        }
        run.svcs.push((ctx.r[15] | ctx.thumb as u32, elapsed, number));
        match number {
            0x28 => {
                let now = run.tick + elapsed;
                ctx.r[0] = now as u32;
                ctx.r[1] = (now >> 32) as u32;
                ctx.budget -= crate::kernel::svc::SYSTEM_TICK_COST as i32;
            }
            // WaitSynchronizationN on the mutex alone, wait_in_place answers it w/ the handle where the cpu reads it plainly
            0x25 => match system.memory.plain_read32(ctx.r[1]) {
                Some(handle) if ctx.r[2] == 1 && *run.mutex.get_or_insert(handle) == handle && system.kernel.quiet_mutex(handle) => {
                    run.holding += 1;
                    ctx.r[0] = 0;
                    ctx.r[1] = 0;
                }
                _ => {
                    run.failed = true;
                    ctx.exit = EXIT_UNWIND;
                }
            },
            0x14 if run.mutex == Some(ctx.r[0]) && run.holding > 0 => {
                run.holding -= 1;
                ctx.r[0] = 0;
            }
            _ => {
                run.failed = true;
                ctx.exit = EXIT_UNWIND;
            }
        }
    }
}

static SIDE: Host = Host {
    read8: side_read8,
    read16: side_read16,
    read32: side_read32,
    write8: side_write8,
    write16: side_write16,
    write32: side_write32,
    interpret: side_interpret,
    lookup: side_lookup,
    svc: side_svc,
};

impl Library {
    /// runs one loop turn on the side from core at the svc reading the clock it starts at (answered w/ tick)
    /// up to the svc period svcs later, the next turn's first, which it leaves unanswered, see side_svc
    /// none when it does anything else or comes back elsewhere, its writes stay for sandbox.put_back
    ///
    /// # Safety
    ///
    /// the system outlives the call and nothing else uses it meanwhile, like inside an svc of the lib's own code
    pub(crate) unsafe fn run_on_the_side(
        &self,
        system: *mut System,
        sandbox: &mut Sandbox,
        core: &Core,
        tick: u64,
        period: usize,
        mutex: &mut Option<u32>,
    ) -> Option<Side> {
        const BUDGET: i32 = 1 << 20;
        let faulted = unsafe { (*system).memory.faulted() };
        let (read_pages, _) = unsafe { (*system).memory.page_tables() };
        let mut svcs = Vec::with_capacity(period);
        svcs.push((core.r[15] | core.thumb() as u32, 0, 0x28));
        let mut run = Run {
            system,
            library: self,
            sandbox,
            start: (core.r[15], core.flags[5], core.r[13], core.r[14]),
            period,
            tick,
            from: BUDGET,
            svcs,
            mutex: *mutex,
            holding: 0,
            end: None,
            failed: false,
            vfp: core.vfp,
            fpscr: core.fpscr,
        };
        let [n, z, c, v, q, thumb, ge, exclusive] = core.flags;
        let mut ctx = Context {
            r: core.r,
            n,
            z,
            c,
            v,
            q,
            thumb,
            ge,
            exclusive,
            budget: BUDGET - crate::kernel::svc::SYSTEM_TICK_COST as i32,
            exit: 0,
            svc: 0,
            depth: 0,
            exclusive_address: core.exclusive_address,
            tls: core.tls,
            read_pages,
            write_pages: run.sandbox_table(),
            vfp: run.vfp.as_mut_ptr(),
            fpscr: &raw mut run.fpscr,
            host: &SIDE,
            user: std::ptr::null_mut(),
            stack_limit: 0,
            tick_end: 0,
            tick_last: 0,
            tick_floor: i32::MAX,
            tick_cost: crate::kernel::svc::SYSTEM_TICK_COST as i32,
            tick_at: recomp_abi::NO_TICKS,
            ticks: 0,
            wait_at: recomp_abi::NO_TICKS,
            release_at: recomp_abi::NO_TICKS,
            held: 0,
            holding: 0,
            held_floor: NOTHING_HELD,
            done: 0,
            trail: [0; TRAIL],
        };
        ctx.r[0] = tick as u32;
        ctx.r[1] = (tick >> 32) as u32;
        ctx.user = &raw mut run as *mut c_void;
        while run.end.is_none() && !run.failed {
            let code = match self.lookup(ctx.r[15] | ctx.thumb as u32) {
                Some(code) if ctx.budget > 0 => code,
                _ => break,
            };
            ctx.exit = 0;
            ctx.depth = 0;
            ctx.stack_limit = recomp_abi::stack_limit();
            // safe bc the code came from the lib and ctx is set up how it expects, user pointing at run
            unsafe { code(&mut ctx) };
            if ctx.exit == EXIT_BUDGET {
                break;
            }
        }
        *mutex = run.mutex;
        let clean = !run.failed && unsafe { (*system).memory.faulted() } == faulted;
        let (svcs, end) = (std::mem::take(&mut run.svcs), run.end.take());
        end.filter(|_| clean).map(|(cost, end)| Side { svcs, cost, end })
    }
}

impl Run<'_> {
    /// the write table the turn's code goes thru
    fn sandbox_table(&self) -> *const *mut u8 {
        // safe bc the sandbox outlives the run
        unsafe { (*self.sandbox).table.as_ptr() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// a lib that won't open has nothing to run, recompiling fixes it
    #[test]
    fn a_library_that_cant_be_opened_is_outdated() {
        assert!(outdated(Path::new("/nowhere/at/all.so")));
    }

    #[test]
    fn arm_and_thumb_code_use_every_slot() {
        let arm: std::collections::BTreeSet<usize> = (0..FOUND as u32).map(|i| slot(0x0010_0000 + 4 * i)).collect();
        let thumb: std::collections::BTreeSet<usize> = (0..FOUND as u32).map(|i| slot(0x0010_0001 + 2 * i)).collect();
        assert!(arm.len() > FOUND * 3 / 4, "{} of {FOUND}", arm.len());
        assert!(thumb.len() > FOUND * 3 / 4, "{} of {FOUND}", thumb.len());
    }
}
