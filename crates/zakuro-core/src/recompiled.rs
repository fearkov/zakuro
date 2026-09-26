//! running code that 3dsrecomp turned into a library ahead of time. the
//! library's functions work on a context holding the registers and reach
//! memory through the same page tables the interpreter uses, calling back
//! here for anything else. whatever the library has no code for stays with
//! the interpreter.

use std::ffi::{c_char, c_void, CStr};
use std::path::Path;

use zakuro_cpu::{Bus, Cpu, Exit};

use crate::memory::Memory;

/// the interface version, which has to match the library's.
const ABI: u32 = 4;

const EXIT_SVC: u32 = 1;
const EXIT_BUDGET: u32 = 2;
const EXIT_UNWIND: u32 = 3;

type Code = unsafe extern "C" fn(*mut Context);

#[repr(C)]
struct Host {
    read8: unsafe extern "C" fn(*mut Context, u32) -> u8,
    read16: unsafe extern "C" fn(*mut Context, u32) -> u16,
    read32: unsafe extern "C" fn(*mut Context, u32) -> u32,
    write8: unsafe extern "C" fn(*mut Context, u32, u8),
    write16: unsafe extern "C" fn(*mut Context, u32, u16),
    write32: unsafe extern "C" fn(*mut Context, u32, u32),
    interpret: unsafe extern "C" fn(*mut Context, u32, u32),
    lookup: unsafe extern "C" fn(*mut Context, u32) -> Option<Code>,
}

#[repr(C)]
struct Context {
    r: [u32; 16],
    n: u8,
    z: u8,
    c: u8,
    v: u8,
    q: u8,
    thumb: u8,
    ge: u8,
    exclusive: u8,
    budget: i32,
    exit: u32,
    svc: u32,
    depth: u32,
    exclusive_address: u32,
    tls: u32,
    read_pages: *const *mut u8,
    write_pages: *const *mut u8,
    vfp: *mut u32,
    fpscr: *mut u32,
    host: *const Host,
    user: *mut c_void,
}

#[repr(C)]
struct Entry {
    address: u32,
    code: Code,
}

#[repr(C)]
struct Module {
    name: *const c_char,
    base: *mut u32,
    size: u32,
    count: u32,
    entries: *const Entry,
}

/// the tables of recompiled code linked into the program itself, rather
/// than loaded from a library, which is how a title's own executable built
/// with 3dsrecomp port runs.
#[derive(Debug, Clone, Copy)]
pub struct Linked {
    program_id: u64,
    abi: u32,
    entries: *const c_void,
    count: u32,
    modules: *const c_void,
    module_count: u32,
}

impl Linked {
    /// # Safety
    ///
    /// the pointers and counts have to be the recomp_ symbols of code that
    /// 3dsrecomp generated, linked into this program.
    pub unsafe fn new(
        program_id: u64,
        abi: u32,
        entries: *const c_void,
        count: u32,
        modules: *const c_void,
        module_count: u32,
    ) -> Linked {
        Linked { program_id, abi, entries, count, modules, module_count }
    }

    /// the title the code was recompiled from.
    pub fn program_id(&self) -> u64 {
        self.program_id
    }
}

// SAFETY: the tables live as long as the program and are read only, apart
// from the module bases, which only the thread running the system touches.
unsafe impl Send for Linked {}
unsafe impl Sync for Linked {}

/// a library of recompiled code, loaded.
pub struct Library {
    entries: *const Entry,
    count: usize,
    modules: *const Module,
    module_count: usize,
    /// the modules the title has loaded, as base, size and index.
    loaded: Vec<(u32, u32, usize)>,
    /// instructions the code handed to the interpreter one at a time.
    fallbacks: std::cell::Cell<u64>,
    /// the library the tables are in, none when they are linked in.
    _library: Option<libloading::Library>,
}

// SAFETY: the tables the pointers lead to are read only, apart from the
// module bases, which only the thread running the system touches.
unsafe impl Send for Library {}

/// why a run of recompiled code ended.
pub enum Stop {
    /// an svc, the cpu is already past it.
    Svc(u32),
    /// the interpreter stopped on an instruction the code handed it.
    Exit(Exit),
    /// the budget ran out, or execution left for code the library does not
    /// have.
    Left,
}

/// what the callbacks reach through the context.
struct Machine<'a> {
    cpu: &'a mut Cpu,
    memory: &'a mut Memory,
    library: &'a Library,
    /// what the interpreter stopped with inside a callback.
    pending: Option<Exit>,
}

/// # Safety
///
/// ctx has to be a context run set up, whose user field points at a live
/// Machine.
unsafe fn machine<'a>(ctx: *mut Context) -> &'a mut Machine<'a> {
    unsafe { &mut *((*ctx).user as *mut Machine) }
}

unsafe extern "C" fn read8(ctx: *mut Context, address: u32) -> u8 {
    unsafe { machine(ctx).memory.read8(address) }
}

unsafe extern "C" fn read16(ctx: *mut Context, address: u32) -> u16 {
    unsafe { machine(ctx).memory.read16(address) }
}

unsafe extern "C" fn read32(ctx: *mut Context, address: u32) -> u32 {
    unsafe { machine(ctx).memory.read32(address) }
}

unsafe extern "C" fn write8(ctx: *mut Context, address: u32, value: u8) {
    unsafe { machine(ctx).memory.write8(address, value) }
}

unsafe extern "C" fn write16(ctx: *mut Context, address: u32, value: u16) {
    unsafe { machine(ctx).memory.write16(address, value) }
}

unsafe extern "C" fn write32(ctx: *mut Context, address: u32, value: u32) {
    unsafe { machine(ctx).memory.write32(address, value) }
}

/// runs one instruction the code left to the interpreter.
unsafe extern "C" fn interpret(ctx: *mut Context, address: u32, _opcode: u32) {
    unsafe {
        let machine = machine(ctx);
        machine.library.fallbacks.set(machine.library.fallbacks.get() + 1);
        let ctx = &mut *ctx;
        load(ctx, machine.cpu);
        machine.cpu.regs[15] = address;
        let exit = machine.cpu.step(machine.memory);
        store(machine.cpu, ctx);
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

static HOST: Host = Host { read8, read16, read32, write8, write16, write32, interpret, lookup };

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

fn find(entries: &[Entry], address: u32) -> Option<Code> {
    entries.binary_search_by_key(&address, |entry| entry.address).ok().map(|i| entries[i].code)
}

impl Library {
    pub fn open(path: &Path) -> Result<Library, String> {
        // SAFETY: a library 3dsrecomp built, whose symbols have the types
        // its recomp.h gives them, which the version check makes sure of.
        unsafe {
            let library = libloading::Library::new(path).map_err(|e| e.to_string())?;
            let symbol = |name: &[u8]| -> Result<*const c_void, String> {
                library.get::<*const c_void>(name).map(|s| *s).map_err(|e| e.to_string())
            };
            let abi = *(symbol(b"recomp_abi")? as *const u32);
            let count = *(symbol(b"recomp_entry_count")? as *const u32);
            let entries = symbol(b"recomp_entries")?;
            let module_count = *(symbol(b"recomp_module_count")? as *const u32);
            let modules = symbol(b"recomp_modules")?;
            let linked = Linked { program_id: 0, abi, entries, count, modules, module_count };
            Library::from_tables(&linked, Some(library))
        }
    }

    /// the code linked into the program.
    pub fn linked(linked: &Linked) -> Result<Library, String> {
        Library::from_tables(linked, None)
    }

    fn from_tables(linked: &Linked, library: Option<libloading::Library>) -> Result<Library, String> {
        if linked.abi != ABI {
            return Err(format!("it was built for version {} of the interface, this is {ABI}", linked.abi));
        }
        Ok(Library {
            entries: linked.entries as *const Entry,
            count: linked.count as usize,
            modules: linked.modules as *const Module,
            module_count: linked.module_count as usize,
            loaded: Vec::new(),
            fallbacks: std::cell::Cell::new(0),
            _library: library,
        })
    }

    fn entries(&self) -> &[Entry] {
        // SAFETY: the table lives as long as the library does
        unsafe { std::slice::from_raw_parts(self.entries, self.count) }
    }

    fn modules(&self) -> &[Module] {
        // SAFETY: as above
        unsafe { std::slice::from_raw_parts(self.modules, self.module_count) }
    }

    /// how many instructions the code has handed to the interpreter.
    pub fn fallbacks(&self) -> u64 {
        self.fallbacks.get()
    }

    /// how many functions and modules the library has code for.
    pub fn describe(&self) -> String {
        format!("{} entry points, {} modules", self.count, self.module_count)
    }

    /// the code that can run from address, bit 0 set for Thumb, in the
    /// executable or in a module that is loaded.
    fn lookup(&self, address: u32) -> Option<Code> {
        find(self.entries(), address).or_else(|| {
            let &(base, _, index) =
                self.loaded.iter().find(|&&(base, size, _)| address.wrapping_sub(base) < size)?;
            let module = &self.modules()[index];
            // SAFETY: the table lives as long as the library does
            let entries = unsafe { std::slice::from_raw_parts(module.entries, module.count as usize) };
            find(entries, address - base)
        })
    }

    pub fn has_code(&self, address: u32) -> bool {
        self.lookup(address).is_some()
    }

    /// tells the code of the module called name where the title loaded it,
    /// zero when it unloads it.
    pub fn place(&mut self, name: &str, base: u32) {
        // SAFETY: the table holds string literals
        let index = self.modules().iter().position(|m| unsafe { CStr::from_ptr(m.name) }.to_bytes() == name.as_bytes());
        let Some(index) = index else { return };
        let module = &self.modules()[index];
        // SAFETY: base points at the module's variable in the library, which
        // the code only reads on entry
        unsafe { *module.base = base };
        let size = module.size;
        self.loaded.retain(|&(_, _, i)| i != index);
        if base != 0 {
            self.loaded.push((base, size, index));
            log::info!("recompiled code for {name} runs at 0x{base:08X}");
        }
    }

    /// runs recompiled code from the cpu's pc for about budget instructions,
    /// returning how many ran and why it stopped.
    pub fn run(&self, cpu: &mut Cpu, memory: &mut Memory, budget: u64) -> (u64, Stop) {
        let cycles = cpu.cycles;
        let (read_pages, write_pages) = memory.page_tables();
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
            budget: budget.min(i32::MAX as u64) as i32,
            exit: 0,
            svc: 0,
            depth: 0,
            exclusive_address: 0,
            tls: 0,
            read_pages,
            write_pages,
            vfp: std::ptr::null_mut(),
            fpscr: std::ptr::null_mut(),
            host: &HOST,
            user: std::ptr::null_mut(),
        };
        store(cpu, &mut ctx);
        let mut machine = Machine { cpu, memory, library: self, pending: None };
        // the code works on the VFP registers where they are, so the
        // interpreter sees its changes without copying them around
        ctx.vfp = machine.cpu.vfp.regs.as_mut_ptr();
        ctx.fpscr = &mut machine.cpu.vfp.fpscr;
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
            match ctx.exit {
                EXIT_SVC => break Stop::Svc(ctx.svc),
                EXIT_BUDGET => break Stop::Left,
                _ => {}
            }
        };

        load(&ctx, machine.cpu);
        let ran = budget.saturating_sub(ctx.budget.max(0) as u64);
        // the interpreter counted the instructions it ran for the code, which
        // the budget already has
        machine.cpu.cycles = cycles + ran;
        (ran, stop)
    }
}
