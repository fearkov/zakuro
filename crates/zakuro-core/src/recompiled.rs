//! running code that 3dsrecomp turned into a library ahead of time. the
//! library's functions work on a context holding the registers and reach
//! memory through the same page tables the interpreter uses, calling back
//! here for anything else. whatever the library has no code for stays with
//! the interpreter.

use std::cell::Cell;
use std::ffi::c_void;
use std::path::Path;

use recomp_abi::{Code, Context, Host, EXIT_BUDGET, EXIT_SVC, EXIT_UNWIND, NOTHING_HELD, TRAIL};
use zakuro_cpu::{Bus, Cpu, Exit};

use crate::kernel::object::ObjectId;
use crate::memory::Memory;
use crate::System;

pub use recomp_abi::Linked;

/// the library 3dsrecomp build installed for a title, if there is one.
pub fn installed(program_id: u64) -> Option<std::path::PathBuf> {
    recomp_abi::installed(program_id)
}

/// whether the library at path is worth building again, having come from
/// an older 3dsrecomp than this build has, or not running at all.
pub fn outdated(path: &Path) -> bool {
    !recomp_abi::Library::open(path).is_ok_and(|code| code.generation() >= recomp_abi::GENERATION)
}

/// recompiled code, and how much of its work it handed back.
pub struct Library {
    code: recomp_abi::Library,
    /// instructions the code handed to the interpreter one at a time.
    fallbacks: Cell<u64>,
    /// svcs the code ran in place and went on after.
    in_place: Cell<u64>,
    /// where the code reads the clock itself, see recomp_abi::Context, the
    /// last place tick_in_place answered GetSystemTick at.
    tick_at: Cell<u32>,
    /// where the code takes and lets go of the mutex it is lent, see
    /// recomp_abi::Context, the last places wait_in_place and
    /// release_in_place ran at.
    wait_at: Cell<u32>,
    release_at: Cell<u32>,
    /// the code found for addresses lately, each in a slot its address
    /// picks. a jump through a register looks its target up, which is a
    /// search through every function otherwise.
    found: Box<[Found]>,
}

/// a slot of Library::found, an address and the code found for it.
type Found = Cell<(u32, Option<Code>)>;

/// slots of Library::found, a power of two.
const FOUND: usize = 65536;

/// the slot of Library::found an address goes in, the top bits of the
/// address times the golden ratio. ARM addresses step by four and Thumb ones
/// by two with bit 0 set, and taking low bits would leave half the slots to
/// one of them.
fn slot(address: u32) -> usize {
    (address.wrapping_mul(0x9E37_79B1) >> (32 - FOUND.trailing_zeros())) as usize
}

/// why a run of recompiled code ended.
pub enum Stop {
    /// an svc, the cpu is already past it.
    Svc(u32),
    /// the interpreter stopped on an instruction the code handed it.
    Exit(Exit),
    /// the budget ran out, or execution left for code the library does not
    /// have.
    Left,
    /// after an svc it ran in place, the code had no budget for the next
    /// block, which the interpreter goes through instead, as it would have
    /// in the step after the svc.
    Declined,
}

/// what the callbacks reach through the context.
struct Machine<'a> {
    /// the system the code runs in, which the library is out of meanwhile.
    system: *mut System,
    library: &'a Library,
    /// what the interpreter stopped with inside a callback.
    pending: Option<Exit>,
    /// the cycle count and the budget when the code started, or went on
    /// after an svc, and how many instructions it ran before.
    cycles: u64,
    budget: i32,
    ran: u64,
    /// the deadline the step had, and the next event's tick, which the
    /// budget stops at.
    deadline: Option<u64>,
    event: Option<u64>,
    /// the cycle count the budget runs out at, as last worked out.
    limit: u64,
    /// an svc the code ran had the step do more, which the code left for.
    left: bool,
    /// the code went on after an svc it ran.
    went_on: bool,
    /// the cycle count a GetSystemTick the code reads itself has to come
    /// before, its cost included, or none when the code reads none.
    due: Option<u64>,
    /// the mutex lent to the code, see recomp_abi::Context, and the cycle
    /// count a wait or a release the code answers itself has to come
    /// before.
    held: Option<(ObjectId, u64)>,
    /// whether the code can be lent one, having been made for version 8 of
    /// the interface or a later one.
    lends: bool,
    /// a tick the code would answer itself has to come to the host, which
    /// a try at skipping turns of a loop polling the clock takes, see
    /// crate::spin.
    tick_wanted: bool,
}

impl Machine<'_> {
    /// what tick_in_place, wait_in_place and release_in_place would have
    /// done for the svcs the code answered itself since the host last
    /// looked, see recomp_abi::Context: the ones in the trail, the last at
    /// tick_last, or for code made before version 8, ticks from tick_at.
    ///
    /// # Safety
    ///
    /// as for machine, and nothing else may hold the system.
    #[inline(always)]
    unsafe fn caught_up(&mut self, ctx: &mut Context) {
        // mostly the code answered nothing since
        if ctx.done != 0 || ctx.ticks != 0 {
            unsafe { self.catch_up(ctx) };
        }
    }

    /// caught_up, when the code answered something.
    ///
    /// # Safety
    ///
    /// as for machine, and nothing else may hold the system.
    unsafe fn catch_up(&mut self, ctx: &mut Context) {
        let done = std::mem::take(&mut ctx.done);
        let ticks = std::mem::take(&mut ctx.ticks);
        let cost = ctx.tick_cost as u64;
        // a tick, the last one, costs, a wait or a release nothing
        let last_tick = done == 0 || ctx.trail[(done - 1) as usize % TRAIL] == ctx.tick_at;
        let after = ctx.tick_last + if last_tick { cost } else { 0 };
        let budget = ctx.tick_end.saturating_sub(after).min(i32::MAX as u64) as i32;
        // what ran before the last svc, the ticks aside
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

    /// takes back the mutex lent to the code, which the host can do
    /// anything to now.
    #[inline(always)]
    fn take_back(&mut self, ctx: &mut Context) {
        self.held = None;
        ctx.held_floor = NOTHING_HELD;
    }

    /// lends the code the mutex a wait or a release ran in place at at took
    /// or let go of alone, see recomp_abi::Context, when nothing but the
    /// code's thread can touch it for a while. the mutex is a release's
    /// handle, or where a wait's handle is.
    ///
    /// # Safety
    ///
    /// as for machine, and nothing else may hold the system.
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

    /// where the code may answer a wait or a release itself from here, with
    /// the cycle count and budget the host just gave it: one at t is clear of
    /// the mutex's due when t < due, where t is tick_end less the budget.
    fn held_from(&self, ctx: &mut Context) {
        ctx.held_floor = self.held.map_or(NOTHING_HELD, |(_, due)| {
            let floor = ctx.tick_end as i64 - due as i64;
            floor.clamp(i32::MIN as i64, i32::MAX as i64) as i32
        });
    }

    /// where the code may answer GetSystemTick itself from here, with the
    /// cycle count and budget the host just gave it: a tick at t is clear
    /// of due when t + cost < due, where t is tick_end less the budget.
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
/// ctx has to be a context run set up, whose user field points at a live
/// Machine.
unsafe fn machine<'a>(ctx: *mut Context) -> &'a mut Machine<'a> {
    unsafe { &mut *((*ctx).user as *mut Machine) }
}

/// the memory the code runs on.
///
/// # Safety
///
/// as for machine, and nothing else may hold the system.
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

/// runs one instruction the code left to the interpreter.
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

/// whether svcs run in place, unless ZAKURO_SVC_IN_PLACE is 0, which has
/// the code stop at every one as before.
pub(crate) fn svcs_in_place() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ZAKURO_SVC_IN_PLACE").map_or(true, |value| value != "0"))
}

/// runs an svc the code made, which the code goes on after when the next
/// step would only run it on, see System::svc_in_place. one that can't run
/// in place stops the code for the step to run it.
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
        // as far as the step goes, the code stopped at the svc
        let ran = machine.budget.saturating_sub(ctx.budget).max(0) as u64;
        machine.ran += ran;
        // the clock, a wait that is over at once and a mutex let go of,
        // which titles call more than anything, without the rest of a step
        // around them when that would change nothing
        let cycles = machine.cycles + ran;
        let at = ctx.r[15] | ctx.thumb as u32;
        let (event, deadline, limit, library) = (machine.event, machine.deadline, machine.limit, machine.library);
        let preempt = system.next_preempt;
        let mut tick_at = None;
        // a mutex taken or let go of alone, which the code may then go on
        // to do itself, see Machine::lend
        let mut mutex = None;
        let done = match number {
            0x28 => {
                // whole turns of a loop polling the clock skipped first, see
                // crate::spin
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
                    // the code reads the clock here itself from now on
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
            // nothing it looks at changed, unless the budget was cut short,
            // a preempt passed, which moves the next, or this is the tick a
            // try at skipping turns wanted
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
                // a tick changes nothing the mutex was lent on, nor does
                // the pass of a preempt it came to, so the code keeps it,
                // rather than coming back with its next wait to have it
                // lent again
                let (handle, release_at) = (ctx.held, ctx.release_at);
                machine.lend(ctx, (handle, false), release_at);
            }
            // a try at skipping turns of a loop polling the clock is due,
            // the history showing one, which takes the next tick, see
            // crate::spin. a turn's other svcs may be the ones coming to the
            // preempts every time, or a try was at another of the turn's
            // ticks than the one it starts at
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
                // nothing is left of the run but the unwinding, its event
                // looked at already
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
    /// a library 3dsrecomp built.
    pub fn open(path: &Path) -> Result<Library, String> {
        recomp_abi::Library::open(path).map(Library::new)
    }

    /// the code linked into the program.
    pub fn linked(linked: &Linked) -> Result<Library, String> {
        recomp_abi::Library::linked(linked).map(Library::new)
    }

    fn new(mut code: recomp_abi::Library) -> Library {
        // ZAKURO_NATIVES=0 has the title's own code run for the C library
        // functions the library does on the host
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

    /// every slot of found empty, an address no code starts at in each.
    fn nothing_found() -> Box<[Found]> {
        (0..FOUND).map(|_| Cell::new((u32::MAX, None))).collect()
    }

    /// whether the code came from an older 3dsrecomp than the one this
    /// build has, so that recompiling the game again makes it better.
    pub fn outdated(&self) -> bool {
        self.code.generation() < recomp_abi::GENERATION
    }

    /// whether the code can tell the functions whose code changed in
    /// memory, which it can from generation 2 on.
    pub fn checks_itself(&self) -> bool {
        self.code.origins(0).is_some()
    }

    /// how many instructions the code has handed to the interpreter.
    pub fn fallbacks(&self) -> u64 {
        self.fallbacks.get()
    }

    /// how many svcs the code ran in place, going on after them.
    pub fn in_place(&self) -> u64 {
        self.in_place.get()
    }

    /// how many functions and modules there is code for.
    pub fn describe(&self) -> String {
        self.code.describe()
    }

    /// the code that can run from address, bit 0 set for Thumb, in the
    /// executable or in a module that is loaded.
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

    /// marks the executable's functions whose code in memory is not what
    /// they were recompiled from, which a mod or another version of the
    /// game does, so that the interpreter runs them. how many there are.
    pub fn check(&mut self, memory: &mut Memory) -> usize {
        let stale = self.code.check(0, 0, |address, bytes| memory.read_bytes(address, bytes));
        self.found = Library::nothing_found();
        stale
    }

    /// tells the code of the module called name where the title loaded it,
    /// zero when it unloads it, and checks the module's code once it is in
    /// memory.
    pub fn place(&mut self, name: &str, base: u32, memory: &mut Memory) {
        let Some(index) = self.code.module_index(name) else { return };
        self.code.place(index, base);
        // what was found before may have moved or gone
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

    /// runs recompiled code from the cpu's pc for about budget instructions,
    /// returning how many ran, why it stopped and the tick of the event its
    /// budget last stopped at. the code runs the svcs it can in place, going
    /// on after them, and deadline, event and limit are what the step had,
    /// for those. the library has to be out of the system, which the code
    /// has to itself.
    pub fn run(&self, system: &mut System, budget: u64, deadline: Option<u64>, event: Option<u64>, limit: u64) -> (u64, Stop, Option<u64>) {
        let system: *mut System = system;
        let budget = budget.min(i32::MAX as u64) as i32;
        // SAFETY: everything the code reaches of the system comes from this
        // one pointer, which nothing else uses until the run is over
        let (cycles, read_pages, write_pages, vfp, fpscr) = unsafe {
            let (read_pages, write_pages) = (*system).memory.page_tables();
            // the code works on the VFP registers where they are, so the
            // interpreter sees its changes without copying them around
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
        // SAFETY: as above
        store(unsafe { &(*system).cpu }, &mut ctx);
        // the code reads the clock, takes the mutex it is lent and lets it
        // go itself where it last did, while that is still code to run, which
        // nothing but the step changes
        // SAFETY: as above
        let runs = |at: u32| at != recomp_abi::NO_TICKS && unsafe { (*system).memory.is_executable(at & !1) } && self.has_code(at);
        for (site, kept) in [(&mut ctx.tick_at, &self.tick_at), (&mut ctx.wait_at, &self.wait_at), (&mut ctx.release_at, &self.release_at)] {
            if runs(kept.get()) {
                *site = kept.get();
            }
        }
        // SAFETY: as above
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
            // SAFETY: the code came from the library and ctx is set up the
            // way it expects, with user pointing at machine. where it went
            // is in the context, what it gives back is only for other code
            unsafe { code(&mut ctx) };
            // SAFETY: as above, the code returned
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

        // SAFETY: the code is done with the system
        let cpu = unsafe { &mut (*system).cpu };
        load(&ctx, cpu);
        let ran = machine.budget.saturating_sub(ctx.budget.max(0)).max(0) as u64;
        // the interpreter counted the instructions it ran for the code, which
        // the budget already has
        cpu.cycles = machine.cycles + ran;
        (machine.ran + ran, stop, machine.event)
    }
}

// -- turns run on the side, see crate::spin ------------------------------

/// the core of recompiled code at an svc, all a turn run on the side starts
/// from: the registers, the flags, the monitor and the VFP registers, which
/// the code works on where the system's CPU has them.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct Core {
    pub r: [u32; 16],
    /// n, z, c, v, q, thumb, ge and exclusive, as the context has them.
    flags: [u8; 8],
    exclusive_address: u32,
    tls: u32,
    vfp: [u32; 32],
    fpscr: u32,
}

impl Core {
    /// the core the code has at an svc, the context's and the CPU's VFP
    /// registers.
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

/// where turns run on the side write: a write table with no page in it,
/// which has their code write every byte through the host, the pages it
/// wrote, saved first and written for real, to be put back after, and the
/// bytes written.
pub(crate) struct Sandbox {
    table: Vec<*mut u8>,
    saved: Vec<(usize, *mut u8, Box<[u8]>)>,
    written: Vec<u32>,
}

impl Sandbox {
    pub(crate) fn new() -> Sandbox {
        Sandbox { table: vec![std::ptr::null_mut(); zakuro_common::memory_map::PAGE_TABLE_ENTRIES], saved: Vec::new(), written: Vec::new() }
    }

    /// each byte the turns run since put_back changed, of those written,
    /// with what it holds now.
    pub(crate) fn changed(&self, memory: &mut Memory) -> Vec<(u32, u8)> {
        let page_size = zakuro_common::memory_map::PAGE_SIZE as usize;
        let mut changed: Vec<(u32, u8)> = Vec::new();
        for &addr in &self.written {
            let page = addr as usize / page_size;
            if let Some((_, at, old)) = self.saved.iter().find(|(saved, _, _)| *saved == page) {
                let offset = addr as usize % page_size;
                // SAFETY: a page saved is one of the CPU's, there while the turns ran
                let now = unsafe { *at.add(offset) };
                if now != old[offset] {
                    changed.push((addr, now));
                }
            }
        }
        for (addr, old) in memory.logged_old() {
            let mut now = [0u8];
            memory.read_bytes(addr, &mut now);
            if now[0] != old {
                changed.push((addr, now[0]));
            }
        }
        changed.sort_unstable();
        changed.dedup();
        changed
    }

    /// the bytes the turns run since put_back wrote, the interpreter's for
    /// their code too.
    pub(crate) fn written(&self, memory: &Memory) -> Vec<u32> {
        let mut written = self.written.clone();
        written.extend(memory.logged_old().into_iter().map(|(addr, _)| addr));
        written.sort_unstable();
        written.dedup();
        written
    }

    /// puts back what the turns run since wrote, what the interpreter wrote
    /// for their code and then the pages saved, and forgets them.
    pub(crate) fn put_back(&mut self, memory: &mut Memory) {
        memory.undo_writes();
        for (_, at, old) in self.saved.drain(..) {
            // SAFETY: as in changed
            unsafe { std::ptr::copy_nonoverlapping(old.as_ptr(), at, old.len()) };
        }
        self.written.clear();
    }
}

/// a turn run on the side: its svcs, the address after each with bit 0
/// set for Thumb, the cycles from the start each came at and its number,
/// the cycles it took and the core the next turn starts from.
pub(crate) struct Side {
    pub svcs: Vec<(u32, u64, u32)>,
    pub cost: u64,
    pub end: Core,
}

/// a turn running on the side.
struct Run<'a> {
    system: *mut System,
    library: &'a Library,
    sandbox: *mut Sandbox,
    /// where the turn starts, the pc after its svc, Thumb, sp and lr.
    start: (u32, u8, u32, u32),
    period: usize,
    tick: u64,
    /// the budget the code started with, before the first svc's cost.
    from: i32,
    svcs: Vec<(u32, u64, u32)>,
    mutex: Option<u32>,
    holding: u32,
    /// the cycles it took and the core the next turn starts from, once it
    /// got there.
    end: Option<(u64, Core)>,
    failed: bool,
    vfp: [u32; 32],
    fpscr: u32,
}

/// the turn running on the side, which the context's user points at.
unsafe fn run<'a>(ctx: *mut Context) -> &'a mut Run<'a> {
    unsafe { &mut *((*ctx).user as *mut Run) }
}

impl Run<'_> {
    /// writes bytes at address for real, noting them, after saving each
    /// page they fall in. a page the CPU can't write plainly, guarded,
    /// watched or not there, is no polling loop's.
    fn write(&mut self, address: u32, bytes: &[u8]) {
        let page_size = zakuro_common::memory_map::PAGE_SIZE as usize;
        // SAFETY: the system and the sandbox outlive the run, and nothing
        // else holds them meanwhile
        let (system, sandbox) = unsafe { (&mut *self.system, &mut *self.sandbox) };
        let (_, real) = system.memory.page_tables();
        for (offset, &byte) in bytes.iter().enumerate() {
            let at = address.wrapping_add(offset as u32);
            let page = at as usize / page_size;
            let ptr = match sandbox.saved.iter().find(|(saved, _, _)| *saved == page) {
                Some(&(_, ptr, _)) => ptr,
                None => {
                    // SAFETY: the CPU's write table has an entry for every page
                    let ptr = unsafe { *real.add(page) };
                    if ptr.is_null() {
                        self.failed = true;
                        return;
                    }
                    // SAFETY: a non-null entry is a whole page
                    let mut old: Box<[u8]> = unsafe { std::slice::from_raw_parts(ptr, page_size) }.into();
                    // what the interpreter wrote there for the code already
                    // is not what the page held
                    for (addr, byte) in system.memory.logged_old() {
                        if addr as usize / page_size == page {
                            old[addr as usize % page_size] = byte;
                        }
                    }
                    sandbox.saved.push((page, ptr, old));
                    ptr
                }
            };
            // SAFETY: as above
            unsafe { *ptr.add(at as usize % page_size) = byte };
            sandbox.written.push(at);
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
    unsafe { run(ctx).write(address, &[value]) }
}

unsafe extern "C" fn side_write16(ctx: *mut Context, address: u32, value: u16) {
    unsafe { run(ctx).write(address, &value.to_le_bytes()) }
}

unsafe extern "C" fn side_write32(ctx: *mut Context, address: u32, value: u32) {
    unsafe { run(ctx).write(address, &value.to_le_bytes()) }
}

/// memory as the interpreter reaches it for a turn on the side, a type of
/// its own so that the interpreter the steps run stays as the compiler
/// built it, the interpreter being inlined where it is called.
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

/// an instruction the code leaves to the interpreter, its writes noted to
/// be undone with the rest.
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

/// an svc of a turn on the side: the clock read as the code would find it
/// then, a wait on the mutex alone taking it and its release letting it go
/// in a count of their own. the one the next turn starts at ends it, and
/// any other svc, or one where the turn would wait, fails it.
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
            // WaitSynchronizationN on the mutex alone, which wait_in_place
            // answers with the handle where the CPU reads it plainly
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
    /// runs a turn of a loop on the side, from core at the svc reading the
    /// clock it starts at, answered with tick, up to the svc period svcs on,
    /// the next turn's first, which it leaves unanswered, see side_svc.
    /// none when it does anything else or comes back elsewhere. what it
    /// wrote stays, for sandbox.put_back.
    ///
    /// # Safety
    ///
    /// the system outlives the call and nothing else uses it meanwhile, as
    /// inside an svc of the code, which is the library's own.
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
            // SAFETY: the code came from the library and ctx is set up the
            // way it expects, with user pointing at run
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
    /// the write table the turn's code goes through.
    fn sandbox_table(&self) -> *const *mut u8 {
        // SAFETY: the sandbox outlives the run
        unsafe { (*self.sandbox).table.as_ptr() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// a library that doesn't open has nothing to run, recompiling is what
    /// mends it.
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
