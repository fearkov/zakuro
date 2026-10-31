//! running code that 3dsrecomp turned into a library ahead of time. the
//! library's functions work on a context holding the registers and reach
//! memory through the same page tables the interpreter uses, calling back
//! here for anything else. whatever the library has no code for stays with
//! the interpreter.

use std::cell::Cell;
use std::ffi::c_void;
use std::path::Path;

use recomp_abi::{Code, Context, Host, EXIT_BUDGET, EXIT_SVC, EXIT_UNWIND};
use zakuro_cpu::{Bus, Cpu, Exit};

use crate::memory::Memory;
use crate::System;

pub use recomp_abi::Linked;

/// the library 3dsrecomp build installed for a title, if there is one.
pub fn installed(program_id: u64) -> Option<std::path::PathBuf> {
    recomp_abi::installed(program_id)
}

/// recompiled code, and how much of its work it handed back.
pub struct Library {
    code: recomp_abi::Library,
    /// instructions the code handed to the interpreter one at a time.
    fallbacks: Cell<u64>,
    /// svcs the code ran in place and went on after.
    in_place: Cell<u64>,
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
    /// an svc the code ran had the step do more, which the code left for.
    left: bool,
    /// the code went on after an svc it ran.
    went_on: bool,
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
        if !svcs_in_place() || !crate::kernel::svc::in_place(number) {
            ctx.svc = number;
            ctx.exit = EXIT_SVC;
            return;
        }
        let system = &mut *machine.system;
        // as far as the step goes, the code stopped at the svc
        let ran = machine.budget.saturating_sub(ctx.budget).max(0) as u64;
        machine.ran += ran;
        load(ctx, &mut system.cpu);
        system.cpu.cycles = machine.cycles + ran;
        let next = system.svc_in_place(number, machine.event, machine.deadline, machine.library);
        store(&system.cpu, ctx);
        machine.cycles = system.cpu.cycles;
        match next {
            Some((budget, event)) => {
                ctx.budget = budget.min(i32::MAX as u64) as i32;
                machine.event = event;
                machine.went_on = true;
                machine.library.in_place.set(machine.library.in_place.get() + 1);
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

    fn new(code: recomp_abi::Library) -> Library {
        Library { code, fallbacks: Cell::new(0), in_place: Cell::new(0), found: Library::nothing_found() }
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
    /// on after them, and deadline and event are what the step had, for
    /// those. the library has to be out of the system, which the code has
    /// to itself.
    pub fn run(&self, system: &mut System, budget: u64, deadline: Option<u64>, event: Option<u64>) -> (u64, Stop, Option<u64>) {
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
        };
        // SAFETY: as above
        store(unsafe { &(*system).cpu }, &mut ctx);
        let mut machine =
            Machine { system, library: self, pending: None, cycles, budget, ran: 0, deadline, event, left: false, went_on: false };
        ctx.user = &mut machine as *mut Machine as *mut c_void;

        let stop = loop {
            if ctx.budget <= 0 {
                break Stop::Left;
            }
            let Some(code) = self.lookup(ctx.r[15] | ctx.thumb as u32) else {
                break Stop::Left;
            };
            ctx.exit = 0;
            ctx.depth = 0;
            // SAFETY: the code came from the library and ctx is set up the
            // way it expects, with user pointing at machine
            unsafe { code(&mut ctx) };
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arm_and_thumb_code_use_every_slot() {
        let arm: std::collections::BTreeSet<usize> = (0..FOUND as u32).map(|i| slot(0x0010_0000 + 4 * i)).collect();
        let thumb: std::collections::BTreeSet<usize> = (0..FOUND as u32).map(|i| slot(0x0010_0001 + 2 * i)).collect();
        assert!(arm.len() > FOUND * 3 / 4, "{} of {FOUND}", arm.len());
        assert!(thumb.len() > FOUND * 3 / 4, "{} of {FOUND}", thumb.len());
    }
}
