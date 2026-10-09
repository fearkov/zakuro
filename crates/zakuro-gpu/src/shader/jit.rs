//! the shaders the CPU runs compiled to the host's own code, with Cranelift.
//! a compiled program shades four vertices at a time, each register a row
//! of lanes per component the way batch keeps eight, and computes exactly
//! what the interpreter computes, operation for operation. the flow of the
//! program is followed ahead of time: each instruction is compiled once for
//! every way of being in the calls, ifs and loops it is reached in. where
//! the vertices of a batch would go different ways at a branch, it gives
//! the batch back, and batch shades it, following each way.

use std::collections::HashMap;
use std::mem::offset_of;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{types, AbiParam, Block, FuncRef, InstBuilder, MemFlagsData, Value};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use super::batch::Triangle;
use super::isa::OpCode;
use super::{Op, Operand, Program, ShaderUnit, Vec4, FLOAT_UNIFORMS, INPUT_REGISTERS, OUTPUT_REGISTERS, PROGRAM_SIZE, TEMP_REGISTERS};

/// how many vertices a compiled program shades together, a vector of the
/// host's each.
const LANES: usize = 4;

/// how many instructions the interpreter runs at most, past which it takes
/// a program for broken and stops it.
const BUDGET: i64 = 0x10000;

/// how many ways of reaching its instructions a program compiles at most,
/// each one at most once between two looks at the budget.
const MOST_STATES: usize = 16_384;

/// how deep calls, ifs and loops go, past which a program is left to the
/// interpreter.
const DEEPEST: usize = 16;

/// how many programs' code is kept, past which it starts over.
const KEPT: usize = 1024;

/// what a compiled program reads and writes, the registers as rows of lanes.
#[repr(C, align(16))]
pub(super) struct Context {
    inputs: [[[f32; LANES]; 4]; INPUT_REGISTERS],
    outputs: [[[f32; LANES]; 4]; OUTPUT_REGISTERS],
    /// what the helpers take and give back.
    scratch: [[f32; LANES]; 4],
    /// what a uniform past the last reads.
    zero: [f32; 4],
    uniforms: *const Vec4,
    bools: u32,
    ints: [[u8; 4]; 4],
    emitted: *mut Emitted,
}

impl Context {
    fn new(unit: &ShaderUnit) -> Context {
        Context {
            inputs: [[[0.0; LANES]; 4]; INPUT_REGISTERS],
            outputs: [[[0.0; LANES]; 4]; OUTPUT_REGISTERS],
            scratch: [[0.0; LANES]; 4],
            zero: [0.0; 4],
            uniforms: unit.float_uniforms.as_ptr(),
            bools: unit.bool_uniforms as u32,
            ints: unit.int_uniforms,
            emitted: std::ptr::null_mut(),
        }
    }

    /// the inputs the program reads into rows, lanes past the last vertex
    /// repeating it, so that they branch its way.
    fn fill(&mut self, read: u16, inputs: &[[Vec4; INPUT_REGISTERS]]) {
        let last = inputs.len() - 1;
        for register in (0..INPUT_REGISTERS).filter(|register| read & (1 << register) != 0) {
            for (component, row) in self.inputs[register].iter_mut().enumerate() {
                *row = std::array::from_fn(|lane| inputs[lane.min(last)][register][component]);
            }
        }
    }

    /// the outputs the program writes out of rows, for the vertices there are.
    fn drain(&self, written: u16, outputs: &mut [[Vec4; OUTPUT_REGISTERS]]) {
        for register in (0..OUTPUT_REGISTERS).filter(|register| written & (1 << register) != 0) {
            for (lane, output) in outputs.iter_mut().enumerate() {
                output[register] = std::array::from_fn(|component| self.outputs[register][component][lane]);
            }
        }
    }
}

/// what the lanes of a geometry shader emit, the way batch's emitters keep
/// it for eight.
pub(super) struct Emitted {
    outputs: u16,
    slot: [usize; LANES],
    completes: [bool; LANES],
    slots: [Triangle; LANES],
    triangles: [Vec<Triangle>; LANES],
}

impl Emitted {
    fn new(outputs: u16) -> Emitted {
        Emitted {
            outputs,
            slot: [0; LANES],
            completes: [false; LANES],
            slots: [[[[0.0; 4]; OUTPUT_REGISTERS]; 3]; LANES],
            triangles: std::array::from_fn(|_| Vec::new()),
        }
    }
}

// the operations too rare to be worth compiling, done the interpreter's way
// on the rows in scratch

extern "C" fn floor(context: *mut Context) {
    // SAFETY: the compiled program passes its context, which nothing else
    // touches while it runs
    let scratch = unsafe { &mut (*context).scratch };
    for row in scratch.iter_mut() {
        *row = row.map(f32::floor);
    }
}

extern "C" fn exp2(context: *mut Context) {
    // SAFETY: as in floor
    let scratch = unsafe { &mut (*context).scratch };
    scratch[0] = scratch[0].map(f32::exp2);
}

extern "C" fn log2(context: *mut Context) {
    // SAFETY: as in floor
    let scratch = unsafe { &mut (*context).scratch };
    scratch[0] = scratch[0].map(f32::log2);
}

/// litp's x, y and w.
extern "C" fn lit(context: *mut Context) {
    // SAFETY: as in floor
    let scratch = unsafe { &mut (*context).scratch };
    scratch[0] = scratch[0].map(|x| x.max(0.0));
    scratch[1] = scratch[1].map(|y| y.clamp(-127.9961, 127.9961));
    scratch[2] = scratch[2].map(|w| w.max(0.0));
}

extern "C" fn set_emit(context: *mut Context, slot: u32, completes: u32) {
    // SAFETY: as in floor, and emitted is the caller's for as long as the
    // program runs, or none
    let Some(emitted) = (unsafe { (*context).emitted.as_mut() }) else { return };
    emitted.slot = [slot as usize; LANES];
    emitted.completes = [completes != 0; LANES];
}

extern "C" fn emit(context: *mut Context) {
    // SAFETY: as in set_emit
    let (outputs, emitted) = unsafe { (&(*context).outputs, (*context).emitted.as_mut()) };
    let Some(emitted) = emitted else { return };
    let written = emitted.outputs;
    let lanes = emitted.slots.iter_mut().zip(&mut emitted.triangles).zip(emitted.slot.iter().zip(&emitted.completes));
    for (lane, ((slots, triangles), (&slot, &completes))) in lanes.enumerate() {
        for register in (0..OUTPUT_REGISTERS).filter(|register| written & (1 << register) != 0) {
            slots[slot][register] = std::array::from_fn(|component| outputs[register][component][lane]);
        }
        if completes {
            triangles.push(*slots);
        }
    }
}

/// a program compiled for one entry point.
pub(super) struct Compiled {
    module: Option<JITModule>,
    function: unsafe extern "C" fn(*mut Context) -> u32,
    /// the batches it ran and gave back, it gives up on a program whose
    /// vertices part ways often, which batch shades better.
    runs: AtomicU32,
    returned: AtomicU32,
    given_up: AtomicBool,
}

// SAFETY: the code is finished and never written again, and the module that
// holds it is only touched again to free it, once no one has it
unsafe impl Send for Compiled {}
unsafe impl Sync for Compiled {}

impl Drop for Compiled {
    fn drop(&mut self) {
        if let Some(module) = self.module.take() {
            // SAFETY: no one has the function any more, whoever ran it held
            // the program
            unsafe { module.free_memory() };
        }
    }
}

impl Compiled {
    /// runs it over up to LANES vertices filled in, false when it gave the
    /// batch back.
    fn run(&self, context: &mut Context) -> bool {
        // SAFETY: the function was compiled for a context laid out like this
        // one, which outlives the call
        unsafe { (self.function)(context) == 0 }
    }

    /// how many batches it gave back, for the tests.
    #[cfg(test)]
    pub(super) fn returned(&self) -> u32 {
        self.returned.load(Ordering::Relaxed)
    }

    /// counts batches run and given back, giving up once a quarter of them
    /// came back.
    fn counted(&self, runs: u32, returned: u32) {
        let runs = self.runs.fetch_add(runs, Ordering::Relaxed) + runs;
        let returned = self.returned.fetch_add(returned, Ordering::Relaxed) + returned;
        if runs >= 64 && returned * 4 > runs && !self.given_up.swap(true, Ordering::Relaxed) {
            log::debug!(target: "zakuro_gpu::shader::jit", "a program whose vertices part ways {returned} times in {runs} goes back to the interpreter");
        }
    }
}

/// whether compiling is on, ZAKURO_SHADER_JIT=0 turns it off.
fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ZAKURO_SHADER_JIT").map_or(true, |value| value != "0"))
}

/// a program's code for an entry point, on its way or there.
enum Slot {
    Compiling,
    /// none where it can't be compiled.
    Done(Option<Arc<Compiled>>),
}

/// each program's code by its fingerprint and entry point, kept apart from
/// the few programs a unit keeps decoded, which titles go through faster
/// than they would want to compile them again.
fn codes() -> std::sync::MutexGuard<'static, HashMap<(u64, u32), Slot>> {
    static KEPT: OnceLock<Mutex<HashMap<(u64, u32), Slot>>> = OnceLock::new();
    KEPT.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// the unit's program compiled for its entry point, none while it compiles,
/// where it can't be, or where it gave up. the first time it is asked for
/// it starts compiling on a thread of its own, a big program takes long
/// enough to hitch a frame, and the interpreter shades meanwhile.
pub(super) fn compiled(unit: &ShaderUnit, program: &Arc<Program>) -> Option<Arc<Compiled>> {
    if !enabled() {
        return None;
    }
    let key = (program.fingerprint, unit.entry_point);
    let mut kept = codes();
    match kept.get(&key) {
        Some(Slot::Done(found)) => return found.clone().filter(|compiled| !compiled.given_up.load(Ordering::Relaxed)),
        Some(Slot::Compiling) => return None,
        None => {}
    }
    if kept.len() >= KEPT {
        kept.clear();
    }
    // the tests look at what the compiled code gives from the first run
    if cfg!(test) {
        let found = compile_logged(program, key.1);
        kept.insert(key, Slot::Done(found.clone()));
        return found;
    }
    kept.insert(key, Slot::Compiling);
    drop(kept);
    let program = program.clone();
    let spawned = std::thread::Builder::new().name("shader compiler".to_owned()).spawn(move || {
        let found = compile_logged(&program, key.1);
        codes().insert(key, Slot::Done(found));
    });
    if let Err(error) = spawned {
        log::warn!("could not start compiling a shader, {error}");
        codes().insert(key, Slot::Done(None));
    }
    None
}

/// compiles the program for an entry point, saying why not when it can't.
fn compile_logged(program: &Program, entry: u32) -> Option<Arc<Compiled>> {
    let start = std::time::Instant::now();
    let found = match compile(program, entry) {
        Ok(compiled) => Some(Arc::new(compiled)),
        Err(error) => {
            log::debug!(target: "zakuro_gpu::shader::jit", "the program {:016X} at {entry} is not compiled, {error}", program.fingerprint);
            None
        }
    };
    log::trace!(target: "zakuro_gpu::shader::jit", "the program {:016X} at {entry} took {:?}", program.fingerprint, start.elapsed());
    found
}

/// shades the vertices, LANES at a time, giving the batches the program
/// gives back to fallback.
pub(super) fn run_vertices(
    compiled: &Compiled,
    unit: &ShaderUnit,
    program: &Program,
    inputs: &[[Vec4; INPUT_REGISTERS]],
    outputs: &mut [[Vec4; OUTPUT_REGISTERS]],
    mut fallback: impl FnMut(&[[Vec4; INPUT_REGISTERS]], &mut [[Vec4; OUTPUT_REGISTERS]]),
) {
    let mut context = Context::new(unit);
    let (mut runs, mut returned) = (0, 0);
    for (inputs, outputs) in inputs.chunks(LANES).zip(outputs.chunks_mut(LANES)) {
        context.fill(program.inputs, inputs);
        runs += 1;
        if compiled.run(&mut context) {
            context.drain(program.outputs, outputs);
        } else {
            returned += 1;
            fallback(inputs, outputs);
        }
    }
    compiled.counted(runs, returned);
}

/// runs the geometry shader over the primitives' inputs, LANES at a time,
/// giving each one's triangles, the batches the program gives back going to
/// fallback.
pub(super) fn run_geometry(
    compiled: &Compiled,
    unit: &ShaderUnit,
    program: &Program,
    inputs: &[[Vec4; INPUT_REGISTERS]],
    triangles: &mut Vec<Vec<Triangle>>,
    mut fallback: impl FnMut(&[[Vec4; INPUT_REGISTERS]], &mut Vec<Vec<Triangle>>),
) {
    let mut context = Context::new(unit);
    let (mut runs, mut returned) = (0, 0);
    for inputs in inputs.chunks(LANES) {
        context.fill(program.inputs, inputs);
        let mut emitted = Emitted::new(program.outputs);
        context.emitted = &mut emitted;
        runs += 1;
        let ran = compiled.run(&mut context);
        context.emitted = std::ptr::null_mut();
        if ran {
            triangles.extend(emitted.triangles.into_iter().take(inputs.len()));
        } else {
            returned += 1;
            fallback(inputs, triangles);
        }
    }
    compiled.counted(runs, returned);
}

/// a call, if or loop the program is in, as the interpreter's blocks keep
/// it. a loop's repeats left are counted where the code runs, by depth.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Frame {
    end: u32,
    return_address: u32,
    start: u32,
    /// the integer uniform a loop steps aL by, none for anything else.
    integer: Option<u32>,
}

/// where the program is, an instruction and the frames it is in.
type State = (u32, Vec<Frame>);

/// the helpers compiled code calls.
struct Helpers {
    floor: FuncRef,
    exp2: FuncRef,
    log2: FuncRef,
    lit: FuncRef,
    set_emit: FuncRef,
    emit: FuncRef,
}

/// compiles the program for an entry point, why not when the host has no
/// backend or the program goes too many ways.
pub(super) fn compile(program: &Program, entry: u32) -> Result<Compiled, String> {
    let mut flags = settings::builder();
    for (name, value) in [("opt_level", "speed"), ("use_colocated_libcalls", "false"), ("is_pic", "false")] {
        flags.set(name, value).map_err(|error| format!("{name}, {error}"))?;
    }
    let isa = cranelift_native::builder()?.finish(settings::Flags::new(flags)).map_err(|error| error.to_string())?;
    if isa.pointer_type() != types::I64 {
        return Err("the host's pointers are not 64-bit".to_owned());
    }
    let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    let helpers: [(&str, *const u8); 6] = [
        ("zakuro_floor", floor as *const u8),
        ("zakuro_exp2", exp2 as *const u8),
        ("zakuro_log2", log2 as *const u8),
        ("zakuro_lit", lit as *const u8),
        ("zakuro_set_emit", set_emit as *const u8),
        ("zakuro_emit", emit as *const u8),
    ];
    for (name, pointer) in helpers {
        builder.symbol(name, pointer);
    }
    let mut module = JITModule::new(builder);
    let pointer = types::I64;

    let mut on_context = module.make_signature();
    on_context.params.push(AbiParam::new(pointer));
    let mut on_slot = on_context.clone();
    on_slot.params.extend([AbiParam::new(types::I32), AbiParam::new(types::I32)]);
    let mut declare = |name: &str, signature| module.declare_function(name, Linkage::Import, signature).map_err(|error| error.to_string());
    let ids = [
        declare("zakuro_floor", &on_context)?,
        declare("zakuro_exp2", &on_context)?,
        declare("zakuro_log2", &on_context)?,
        declare("zakuro_lit", &on_context)?,
        declare("zakuro_set_emit", &on_slot)?,
        declare("zakuro_emit", &on_context)?,
    ];

    let mut context = module.make_context();
    context.func.signature.params.push(AbiParam::new(pointer));
    context.func.signature.returns.push(AbiParam::new(types::I32));
    let mut functions = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut context.func, &mut functions);
    let [floor, exp2, log2, lit, set_emit, emit] = ids.map(|id| module.declare_func_in_func(id, builder.func));
    let helpers = Helpers { floor, exp2, log2, lit, set_emit, emit };

    // given up halfway, the function is left unfinished
    Compiler::build(&mut builder, program, entry, helpers).ok_or("it goes too many ways or too deep")?;
    builder.seal_all_blocks();
    builder.finalize(module.target_config());

    let id = module.declare_function("shader", Linkage::Export, &context.func.signature).map_err(|error| error.to_string())?;
    // ZAKURO_SHADER_JIT_DUMP=1 prints the host's code each program became
    let dump = std::env::var_os("ZAKURO_SHADER_JIT_DUMP").is_some();
    context.set_disasm(dump);
    module.define_function(id, &mut context).map_err(|error| format!("{error:?}"))?;
    if let Some(code) = context.compiled_code().and_then(|code| code.vcode.clone()).filter(|_| dump) {
        eprintln!("{code}");
    }
    module.clear_context(&mut context);
    module.finalize_definitions().map_err(|error| error.to_string())?;
    let code = module.get_finalized_function(id);
    // SAFETY: the function takes a pointer to a context and gives back a
    // 32-bit word, as its signature says
    let function = unsafe { std::mem::transmute::<*const u8, unsafe extern "C" fn(*mut Context) -> u32>(code) };
    Ok(Compiled {
        module: Some(module),
        function,
        runs: AtomicU32::new(0),
        returned: AtomicU32::new(0),
        given_up: AtomicBool::new(false),
    })
}

/// the memory the code reads that nothing writes while it runs.
fn constant() -> MemFlagsData {
    MemFlagsData::trusted().with_readonly()
}

struct Compiler<'a, 'b, 'c> {
    builder: &'a mut FunctionBuilder<'b>,
    program: &'c Program,
    helpers: Helpers,
    context: Value,
    uniforms: Value,
    temps: [[Variable; 4]; TEMP_REGISTERS],
    outputs: [[Variable; 4]; OUTPUT_REGISTERS],
    /// a0 and a1, a lane each.
    address: [Variable; 2],
    loop_counter: Variable,
    /// the condition registers, a lane each, all ones for true.
    conditions: [Variable; 2],
    /// each depth's loop's repeats left.
    repeats: [Variable; DEEPEST],
    /// the instructions the interpreter would still run.
    budget: Variable,
    states: HashMap<State, Block>,
    pending: Vec<(State, Block)>,
    /// jumps to a state compiled before, which may close a loop, and so
    /// look at the budget first.
    checks: Vec<(Block, Block)>,
    /// gives the batch back.
    give_back: Block,
    /// the lanes the code being compiled runs for, all of them when none,
    /// the others keeping what they had, inside a branch the vertices
    /// parted at.
    mask: Option<Value>,
}

impl<'a, 'b, 'c> Compiler<'a, 'b, 'c> {
    fn build(builder: &'a mut FunctionBuilder<'b>, program: &'c Program, entry: u32, helpers: Helpers) -> Option<()> {
        // the first block filled in is the one the function starts at
        let start = builder.create_block();
        builder.append_block_params_for_function_params(start);
        let give_back = builder.create_block();
        builder.switch_to_block(start);
        let context = builder.block_params(start)[0];
        let uniforms = builder.ins().load(types::I64, constant(), context, offset_of!(Context, uniforms) as i32);
        let vector = |builder: &mut FunctionBuilder, value: f32| {
            let scalar = builder.ins().f32const(value);
            builder.ins().splat(types::F32X4, scalar)
        };
        let zero = vector(builder, 0.0);
        let one = vector(builder, 1.0);
        let declare = |builder: &mut FunctionBuilder, ty, value: Value| {
            let variable = builder.declare_var(ty);
            builder.def_var(variable, value);
            variable
        };
        let temps = std::array::from_fn(|_| std::array::from_fn(|component| declare(builder, types::F32X4, if component == 3 { one } else { zero })));
        let outputs = std::array::from_fn(|_| std::array::from_fn(|_| declare(builder, types::F32X4, zero)));
        let none = builder.ins().iconst(types::I32, 0);
        let lanes_none = builder.ins().splat(types::I32X4, none);
        let address = std::array::from_fn(|_| declare(builder, types::I32X4, lanes_none));
        let conditions = std::array::from_fn(|_| declare(builder, types::I32X4, lanes_none));
        let loop_counter = declare(builder, types::I32, none);
        let repeats = std::array::from_fn(|_| declare(builder, types::I32, none));
        let full = builder.ins().iconst(types::I32, BUDGET);
        let budget = declare(builder, types::I32, full);

        let mut compiler = Compiler {
            builder,
            program,
            helpers,
            context,
            uniforms,
            temps,
            outputs,
            address,
            loop_counter,
            conditions,
            repeats,
            budget,
            states: HashMap::new(),
            pending: Vec::new(),
            checks: Vec::new(),
            give_back,
            mask: None,
        };
        let first = compiler.edge(entry, Vec::new());
        compiler.builder.ins().jump(first, &[]);
        while let Some((state, block)) = compiler.pending.pop() {
            if compiler.states.len() > MOST_STATES {
                return None;
            }
            compiler.builder.switch_to_block(block);
            compiler.state(state)?;
            while let Some((check, target)) = compiler.checks.pop() {
                compiler.builder.switch_to_block(check);
                let budget = compiler.builder.use_var(compiler.budget);
                let low = compiler.builder.ins().icmp_imm_s(IntCC::SignedLessThan, budget, MOST_STATES as i64);
                compiler.builder.ins().brif(low, compiler.give_back, &[], target, &[]);
            }
        }
        compiler.builder.switch_to_block(give_back);
        let returned = compiler.builder.ins().iconst(types::I32, 1);
        compiler.builder.ins().return_(&[returned]);
        Some(())
    }

    /// the block a jump to a state goes to, the state's own when it is new,
    /// one that looks at the budget first when it was there before.
    fn edge(&mut self, pc: u32, frames: Vec<Frame>) -> Block {
        let state = (pc, frames);
        if let Some(&block) = self.states.get(&state) {
            let check = self.builder.create_block();
            self.checks.push((check, block));
            return check;
        }
        let block = self.builder.create_block();
        self.states.insert(state.clone(), block);
        self.pending.push((state, block));
        block
    }

    fn jump(&mut self, pc: u32, frames: Vec<Frame>) {
        let target = self.edge(pc, frames);
        self.builder.ins().jump(target, &[]);
    }

    /// compiles one state: the blocks ending there as the interpreter leaves
    /// them, then its instruction.
    fn state(&mut self, (mut pc, mut frames): State) -> Option<()> {
        let depth = frames.len();
        while let Some(top) = frames.last().copied() {
            if pc != top.end {
                break;
            }
            if let Some(integer) = top.integer {
                // an iteration of a loop ends, aL steps, and it goes again
                // while it has repeats left
                let at = offset_of!(Context, ints) as u32 + integer * 4 + 2;
                let increment = self.builder.ins().sload8(types::I32, constant(), self.context, at as i32);
                let counter = self.builder.use_var(self.loop_counter);
                let stepped = self.builder.ins().iadd(counter, increment);
                self.builder.def_var(self.loop_counter, stepped);
                let level = frames.len() - 1;
                let repeats = self.builder.use_var(self.repeats[level]);
                let again = self.builder.create_block();
                let mut left = frames.clone();
                left.pop();
                let done = self.edge(top.return_address, left);
                self.builder.ins().brif(repeats, again, &[], done, &[]);
                self.builder.switch_to_block(again);
                let fewer = self.builder.ins().iadd_imm_s(repeats, -1);
                self.builder.def_var(self.repeats[level], fewer);
                self.jump(top.start, frames);
                return Some(());
            }
            pc = top.return_address;
            frames.pop();
        }
        if frames.len() != depth {
            // the state the blocks that ended leave it in, compiled once
            self.jump(pc, frames);
            return Some(());
        }
        if pc as usize >= PROGRAM_SIZE {
            self.finish();
            return Some(());
        }
        let budget = self.builder.use_var(self.budget);
        let spent = self.builder.ins().iadd_imm_s(budget, -1);
        self.builder.def_var(self.budget, spent);

        let op = self.program.ops[pc as usize];
        let instruction = op.instruction;
        let next = pc + 1;
        match op.opcode {
            OpCode::End => self.finish(),
            OpCode::SetEmit => {
                let slot = self.builder.ins().iconst(types::I32, instruction.emit_vertex().min(2) as i64);
                let completes = self.builder.ins().iconst(types::I32, instruction.emit_primitive() as i64);
                self.builder.ins().call(self.helpers.set_emit, &[self.context, slot, completes]);
                self.jump(next, frames);
            }
            OpCode::Emit => {
                self.store_outputs();
                self.builder.ins().call(self.helpers.emit, &[self.context]);
                self.jump(next, frames);
            }
            OpCode::Loop => {
                if frames.len() >= DEEPEST {
                    return None;
                }
                let integer = instruction.integer_index() & 3;
                let at = offset_of!(Context, ints) as u32 + integer * 4;
                let repeats = self.builder.ins().uload8(types::I32, constant(), self.context, at as i32);
                let first = self.builder.ins().uload8(types::I32, constant(), self.context, (at + 1) as i32);
                self.builder.def_var(self.loop_counter, first);
                self.builder.def_var(self.repeats[frames.len()], repeats);
                let (last, _) = instruction.flow_target();
                frames.push(Frame { end: last + 1, return_address: last + 1, start: next, integer: Some(integer) });
                self.jump(next, frames);
            }
            OpCode::Call
            | OpCode::CallU
            | OpCode::CallC
            | OpCode::JmpC
            | OpCode::JmpU
            | OpCode::IfC
            | OpCode::IfU
            | OpCode::Break
            | OpCode::BreakC => self.flow(&op, pc, frames)?,
            OpCode::Nop => self.jump(next, frames),
            _ => {
                self.arithmetic(&op);
                self.jump(next, frames);
            }
        }
        Some(())
    }

    /// a flow instruction, which goes one way or the other, every lane the
    /// same way or else the batch goes back.
    fn flow(&mut self, op: &Op, pc: u32, frames: Vec<Frame>) -> Option<()> {
        let instruction = op.instruction;
        let (taken_pc, taken_frames) = branch(op, pc, true, &frames);
        let (other_pc, other_frames) = branch(op, pc, false, &frames);
        if taken_frames.len() > DEEPEST {
            return None;
        }
        let bool_set = |compiler: &mut Self| {
            let bools = compiler.builder.ins().load(types::I32, constant(), compiler.context, offset_of!(Context, bools) as i32);
            compiler.builder.ins().band_imm_u(bools, 1i64 << instruction.bool_index())
        };
        match op.opcode {
            OpCode::Call | OpCode::Break => self.jump(taken_pc, taken_frames),
            OpCode::CallU | OpCode::IfU | OpCode::JmpU => {
                let set = bool_set(self);
                // jmpu can test for either value, the low bit of its count
                // inverts it
                let inverted = op.opcode == OpCode::JmpU && instruction.flow_target().1 & 1 != 0;
                let taken = self.edge(taken_pc, taken_frames);
                let other = self.edge(other_pc, other_frames);
                if inverted {
                    self.builder.ins().brif(set, other, &[], taken, &[]);
                } else {
                    self.builder.ins().brif(set, taken, &[], other, &[]);
                }
            }
            _ => {
                let lanes = self.condition(instruction);
                let taken = self.edge(taken_pc, taken_frames);
                let other = self.edge(other_pc, other_frames);
                let all = self.builder.ins().vall_true(lanes);
                let none_check = self.builder.create_block();
                self.builder.ins().brif(all, taken, &[], none_check, &[]);
                self.builder.switch_to_block(none_check);
                let any = self.builder.ins().vany_true(lanes);
                // where the lanes part, an if or a call that only works out
                // values runs for the lanes going each way, the others
                // keeping theirs, as they would have going the other way
                let (destination, count) = instruction.flow_target();
                let parted = match op.opcode {
                    OpCode::IfC => {
                        let (then, otherwise, next) = if_ranges(pc, destination, count);
                        (self.program.plain(then, 0) && self.program.plain(otherwise, 0)).then_some(([then, otherwise], next))
                    }
                    OpCode::CallC => {
                        let routine = (destination, destination + count);
                        self.program.plain(routine, 0).then_some(([routine, (pc + 1, pc + 1)], pc + 1))
                    }
                    _ => None,
                };
                let Some(([first, second], next)) = parted else {
                    self.builder.ins().brif(any, self.give_back, &[], other, &[]);
                    return Some(());
                };
                let both = self.builder.create_block();
                self.builder.ins().brif(any, both, &[], other, &[]);
                self.builder.switch_to_block(both);
                let rest = self.builder.ins().bnot(lanes);
                self.predicated(first, lanes);
                self.predicated(second, rest);
                self.jump(next, frames);
            }
        }
        Some(())
    }

    /// compiles the instructions of a range Program::plain allows for the
    /// lanes in mask, ifs inside narrowing it further.
    fn predicated(&mut self, (start, end): (u32, u32), mask: Value) {
        let mut pc = start;
        while pc < end {
            let op = self.program.ops[pc as usize];
            let budget = self.builder.use_var(self.budget);
            let spent = self.builder.ins().iadd_imm_s(budget, -1);
            self.builder.def_var(self.budget, spent);
            match op.opcode {
                OpCode::IfC | OpCode::IfU => {
                    let lanes = if op.opcode == OpCode::IfC {
                        self.condition(op.instruction)
                    } else {
                        let bools = self.builder.ins().load(types::I32, constant(), self.context, offset_of!(Context, bools) as i32);
                        let set = self.builder.ins().band_imm_u(bools, 1i64 << op.instruction.bool_index());
                        let set = self.builder.ins().icmp_imm_u(IntCC::NotEqual, set, 0);
                        let set = self.builder.ins().bmask(types::I32, set);
                        self.builder.ins().splat(types::I32X4, set)
                    };
                    let (destination, count) = op.instruction.flow_target();
                    let (then, otherwise, next) = if_ranges(pc, destination, count);
                    let taken = self.builder.ins().band(mask, lanes);
                    let others = self.builder.ins().band_not(mask, lanes);
                    self.predicated(then, taken);
                    self.predicated(otherwise, others);
                    pc = next;
                }
                _ => {
                    let outer = self.mask.replace(mask);
                    self.arithmetic(&op);
                    self.mask = outer;
                    pc += 1;
                }
            }
        }
    }

    /// sets a variable to a value, in the lanes being compiled for, the
    /// others keeping what they had.
    fn assign(&mut self, variable: Variable, value: Value) {
        let value = match self.mask {
            Some(mask) => {
                let old = self.builder.use_var(variable);
                let floats = self.builder.func.dfg.value_type(value) == types::F32X4;
                let mask = if floats { self.floats(mask) } else { mask };
                self.builder.ins().bitselect(mask, value, old)
            }
            None => value,
        };
        self.builder.def_var(variable, value);
    }

    /// the lanes a conditional flow instruction goes its way in, all ones.
    fn condition(&mut self, instruction: super::isa::Instruction) -> Value {
        let reference = instruction.condition_reference();
        let [x, y] = [0, 1].map(|component| {
            let condition = self.builder.use_var(self.conditions[component]);
            if reference[component] { condition } else { self.builder.ins().bnot(condition) }
        });
        match instruction.condition_op() {
            0 => self.builder.ins().bor(x, y),
            1 => self.builder.ins().band(x, y),
            2 => x,
            _ => y,
        }
    }

    fn store_outputs(&mut self) {
        for register in (0..OUTPUT_REGISTERS).filter(|register| self.program.outputs & (1 << register) != 0) {
            for component in 0..4 {
                let value = self.builder.use_var(self.outputs[register][component]);
                let at = offset_of!(Context, outputs) + (register * 4 + component) * 16;
                self.builder.ins().store(MemFlagsData::trusted(), value, self.context, at as i32);
            }
        }
    }

    fn finish(&mut self) {
        self.store_outputs();
        let ran = self.builder.ins().iconst(types::I32, 0);
        self.builder.ins().return_(&[ran]);
    }

    fn splat(&mut self, value: f32) -> Value {
        let scalar = self.builder.ins().f32const(value);
        self.builder.ins().splat(types::F32X4, scalar)
    }

    /// a mask's lanes as floats, to pick between floats with.
    fn floats(&mut self, mask: Value) -> Value {
        self.builder.ins().bitcast(types::F32X4, MemFlagsData::new(), mask)
    }

    /// the four components of a uniform at a pointer, the same in every lane.
    fn uniform_at(&mut self, pointer: Value, flags: MemFlagsData) -> [Value; 4] {
        std::array::from_fn(|component| {
            let scalar = self.builder.ins().load(types::F32, flags, pointer, (component * 4) as i32);
            self.builder.ins().splat(types::F32X4, scalar)
        })
    }

    /// where uniform index is, or what reads as zero when it is past the
    /// last one.
    fn uniform_pointer(&mut self, index: Value) -> Value {
        let inside = self.builder.ins().icmp_imm_u(IntCC::UnsignedLessThan, index, FLOAT_UNIFORMS as i64);
        let wide = self.builder.ins().uextend(types::I64, index);
        let offset = self.builder.ins().ishl_imm_u(wide, 4);
        let at = self.builder.ins().iadd(self.uniforms, offset);
        let zero = self.builder.ins().iadd_imm_s(self.context, offset_of!(Context, zero) as i64);
        self.builder.ins().select(inside, at, zero)
    }

    /// a source's components, swizzled and negated as its descriptor says.
    /// read again rather than taken from before, for the rare way around a
    /// branch, which then holds nothing live past it.
    fn source(&mut self, operand: &Operand, again: bool) -> [Value; 4] {
        let flags = if again { MemFlagsData::trusted() } else { constant() };
        let register = operand.register as usize;
        let value: [Value; 4] = match register {
            0x00..=0x0F => std::array::from_fn(|component| {
                let at = offset_of!(Context, inputs) + (register * 4 + component) * 16;
                self.builder.ins().load(types::F32X4, flags, self.context, at as i32)
            }),
            0x10..=0x1F => std::array::from_fn(|component| self.builder.use_var(self.temps[register - 0x10][component])),
            _ => {
                let base = register as i64 - 0x20;
                match operand.index {
                    0 => {
                        let pointer = self.builder.ins().iadd_imm_s(self.uniforms, base * 16);
                        self.uniform_at(pointer, flags)
                    }
                    3 => {
                        let counter = self.builder.use_var(self.loop_counter);
                        let index = self.builder.ins().iadd_imm_s(counter, base);
                        let pointer = self.uniform_pointer(index);
                        self.uniform_at(pointer, flags)
                    }
                    index => {
                        // each lane's own uniform, an address register apart
                        let offsets = self.builder.use_var(self.address[index as usize - 1]);
                        let mut rows = [self.splat(0.0); 4];
                        for lane in 0..LANES as u8 {
                            let offset = self.builder.ins().extractlane(offsets, lane);
                            let index = self.builder.ins().iadd_imm_s(offset, base);
                            let pointer = self.uniform_pointer(index);
                            for (component, row) in rows.iter_mut().enumerate() {
                                let scalar = self.builder.ins().load(types::F32, flags, pointer, (component * 4) as i32);
                                *row = self.builder.ins().insertlane(*row, scalar, lane);
                            }
                        }
                        rows
                    }
                }
            }
        };
        std::array::from_fn(|component| {
            let row = value[operand.swizzle[component] as usize];
            if operand.negate { self.builder.ins().fneg(row) } else { row }
        })
    }

    fn write(&mut self, op: &Op, value: [Value; 4]) {
        let register = op.destination as usize;
        let row = match register {
            0x00..=0x0F => self.outputs[register],
            0x10..=0x1F => self.temps[register - 0x10],
            _ => return,
        };
        // the mask's most significant bit selects x
        for (component, variable) in row.into_iter().enumerate() {
            if op.mask & (0b1000 >> component) != 0 {
                self.assign(variable, value[component]);
            }
        }
    }

    /// whether any lane of any of the values is NaN.
    fn any_nan(&mut self, values: &[Value]) -> Value {
        let masks: Vec<Value> = values.iter().map(|&value| self.builder.ins().fcmp(FloatCC::Unordered, value, value)).collect();
        let mask = masks.into_iter().reduce(|a, b| self.builder.ins().bor(a, b)).expect("a value");
        self.builder.ins().vany_true(mask)
    }

    /// the values worked out the quick way, or the careful way where a lane
    /// of them is NaN, which the shader's multiply would have made zero.
    fn checked(&mut self, quick: Vec<Value>, careful: impl FnOnce(&mut Self) -> Vec<Value>) -> Vec<Value> {
        let nan = self.any_nan(&quick);
        let slow = self.builder.create_block();
        let join = self.builder.create_block();
        for _ in &quick {
            self.builder.append_block_param(join, types::F32X4);
        }
        let quick: Vec<_> = quick.into_iter().map(Into::into).collect();
        self.builder.ins().brif(nan, slow, &[], join, &quick);
        self.builder.switch_to_block(slow);
        // a store first, so that what the careful way reads again is read
        // again, not taken from the quick way's loads, which would then have
        // to stay live past the branch
        let nothing = self.builder.ins().iconst(types::I32, 0);
        self.builder.ins().store(MemFlagsData::trusted(), nothing, self.context, offset_of!(Context, scratch) as i32);
        let fixed: Vec<_> = careful(self).into_iter().map(Into::into).collect();
        self.builder.ins().jump(join, &fixed);
        self.builder.switch_to_block(join);
        self.builder.block_params(join).to_vec()
    }

    /// the shader's multiply, zero rather than NaN for zero times infinity.
    fn multiply(&mut self, a: Value, b: Value) -> Value {
        let product = self.builder.ins().fmul(a, b);
        let nan = self.builder.ins().fcmp(FloatCC::Unordered, product, product);
        let a_nan = self.builder.ins().fcmp(FloatCC::Unordered, a, a);
        let b_nan = self.builder.ins().fcmp(FloatCC::Unordered, b, b);
        let either = self.builder.ins().bor(a_nan, b_nan);
        let made = self.builder.ins().band_not(nan, either);
        let made = self.floats(made);
        let zero = self.splat(0.0);
        self.builder.ins().bitselect(made, zero, product)
    }

    /// a dot product of an instruction's two sources over the first count
    /// components, which adds up exactly the way the interpreter's does.
    fn dot(&mut self, op: &Op, a: &[Value; 4], b: &[Value; 4], count: usize) -> Value {
        let start = std::iter::empty::<f32>().sum::<f32>();
        let mut sum = self.splat(start);
        for component in 0..count {
            let product = self.builder.ins().fmul(a[component], b[component]);
            sum = self.builder.ins().fadd(sum, product);
        }
        let op = *op;
        self.checked(vec![sum], move |compiler| {
            let (a, b) = (compiler.source(&op.sources[0], true), compiler.source(&op.sources[1], true));
            let mut sum = compiler.splat(start);
            for component in 0..count {
                let product = compiler.multiply(a[component], b[component]);
                sum = compiler.builder.ins().fadd(sum, product);
            }
            vec![sum]
        })[0]
    }

    /// floats through a helper, the rows in scratch and back.
    fn helped(&mut self, helper: FuncRef, rows: &[Value]) -> Vec<Value> {
        let at = |row: usize| (offset_of!(Context, scratch) + row * 16) as i32;
        for (row, &value) in rows.iter().enumerate() {
            self.builder.ins().store(MemFlagsData::trusted(), value, self.context, at(row));
        }
        self.builder.ins().call(helper, &[self.context]);
        (0..rows.len()).map(|row| self.builder.ins().load(types::F32X4, MemFlagsData::trusted(), self.context, at(row))).collect()
    }

    /// true or false in every lane, as one and zero.
    fn ones(&mut self, mask: Value) -> Value {
        let one = self.splat(1.0);
        let one = self.builder.ins().bitcast(types::I32X4, MemFlagsData::new(), one);
        let picked = self.builder.ins().band(mask, one);
        self.floats(picked)
    }

    fn arithmetic(&mut self, op: &Op) {
        if !op.opcode.writes() && !matches!(op.opcode, OpCode::Mova | OpCode::Cmp) {
            // what the interpreter leaves alone
            return;
        }
        let sources = if matches!(op.opcode, OpCode::Mad | OpCode::MadI) { 3 } else { 2 };
        let values: Vec<[Value; 4]> = op.sources[..sources].iter().map(|operand| self.source(operand, false)).collect();
        let (a, b) = (values[0], values[1]);
        let pairs = |compiler: &mut Self, f: fn(&mut Self, Value, Value) -> Value| -> [Value; 4] {
            std::array::from_fn(|component| f(compiler, a[component], b[component]))
        };
        let result: [Value; 4] = match op.opcode {
            OpCode::Add => pairs(self, |c, x, y| c.builder.ins().fadd(x, y)),
            OpCode::Mul => {
                let quick: Vec<Value> = (0..4).map(|component| self.builder.ins().fmul(a[component], b[component])).collect();
                let op = *op;
                let fixed = self.checked(quick, move |compiler| {
                    let (a, b) = (compiler.source(&op.sources[0], true), compiler.source(&op.sources[1], true));
                    (0..4).map(|component| compiler.multiply(a[component], b[component])).collect()
                });
                std::array::from_fn(|component| fixed[component])
            }
            OpCode::Mad | OpCode::MadI => {
                let c = values[2];
                let quick: Vec<Value> = (0..4)
                    .map(|component| {
                        let product = self.builder.ins().fmul(a[component], b[component]);
                        self.builder.ins().fadd(product, c[component])
                    })
                    .collect();
                let op = *op;
                let fixed = self.checked(quick, move |compiler| {
                    let [a, b, c] = [0, 1, 2].map(|source| compiler.source(&op.sources[source], true));
                    (0..4)
                        .map(|component| {
                            let product = compiler.multiply(a[component], b[component]);
                            compiler.builder.ins().fadd(product, c[component])
                        })
                        .collect()
                });
                std::array::from_fn(|component| fixed[component])
            }
            // x when it is greater, else y, which is how NaN behaves on hardware
            OpCode::Max => pairs(self, |c, x, y| {
                let greater = c.builder.ins().fcmp(FloatCC::GreaterThan, x, y);
                let greater = c.floats(greater);
                c.builder.ins().bitselect(greater, x, y)
            }),
            OpCode::Min => pairs(self, |c, x, y| {
                let less = c.builder.ins().fcmp(FloatCC::LessThan, x, y);
                let less = c.floats(less);
                c.builder.ins().bitselect(less, x, y)
            }),
            OpCode::Dp3 => [self.dot(op, &a, &b, 3); 4],
            OpCode::Dp4 => [self.dot(op, &a, &b, 4); 4],
            OpCode::Dph | OpCode::DphI => {
                let dot = self.dot(op, &a, &b, 3);
                [self.builder.ins().fadd(dot, b[3]); 4]
            }
            OpCode::Mov => a,
            OpCode::Flr => {
                let floored = self.helped(self.helpers.floor, &a);
                std::array::from_fn(|component| floored[component])
            }
            OpCode::Rcp => {
                let one = self.splat(1.0);
                [self.builder.ins().fdiv(one, a[0]); 4]
            }
            OpCode::Rsq => {
                let one = self.splat(1.0);
                let root = self.builder.ins().sqrt(a[0]);
                [self.builder.ins().fdiv(one, root); 4]
            }
            OpCode::Ex2 => [self.helped(self.helpers.exp2, &[a[0]])[0]; 4],
            OpCode::Lg2 => [self.helped(self.helpers.log2, &[a[0]])[0]; 4],
            OpCode::Sge | OpCode::SgeI => pairs(self, |c, x, y| {
                let mask = c.builder.ins().fcmp(FloatCC::GreaterThanOrEqual, x, y);
                c.ones(mask)
            }),
            OpCode::Slt | OpCode::SltI => pairs(self, |c, x, y| {
                let mask = c.builder.ins().fcmp(FloatCC::LessThan, x, y);
                c.ones(mask)
            }),
            OpCode::Dst | OpCode::DstI => {
                let quick = self.builder.ins().fmul(a[1], b[1]);
                let op = *op;
                let product = self.checked(vec![quick], move |compiler| {
                    let (a, b) = (compiler.source(&op.sources[0], true), compiler.source(&op.sources[1], true));
                    vec![compiler.multiply(a[1], b[1])]
                })[0];
                [self.splat(1.0), product, a[2], b[3]]
            }
            OpCode::LitP => {
                let zero = self.splat(0.0);
                let x = self.builder.ins().fcmp(FloatCC::GreaterThanOrEqual, a[0], zero);
                let w = self.builder.ins().fcmp(FloatCC::GreaterThanOrEqual, a[3], zero);
                self.assign(self.conditions[0], x);
                self.assign(self.conditions[1], w);
                let lit = self.helped(self.helpers.lit, &[a[0], a[1], a[3]]);
                [lit[0], lit[1], zero, lit[2]]
            }
            OpCode::Mova => {
                if op.mask & 0b1000 != 0 {
                    let x = self.builder.ins().fcvt_to_sint_sat(types::I32X4, a[0]);
                    self.assign(self.address[0], x);
                }
                if op.mask & 0b0100 != 0 {
                    let y = self.builder.ins().fcvt_to_sint_sat(types::I32X4, a[1]);
                    self.assign(self.address[1], y);
                }
                return;
            }
            OpCode::Cmp => {
                let modes = op.instruction.compare_modes();
                for component in 0..2 {
                    let condition = match modes[component] {
                        0 => Some(FloatCC::Equal),
                        1 => Some(FloatCC::NotEqual),
                        2 => Some(FloatCC::LessThan),
                        3 => Some(FloatCC::LessThanOrEqual),
                        4 => Some(FloatCC::GreaterThan),
                        5 => Some(FloatCC::GreaterThanOrEqual),
                        _ => None,
                    };
                    let mask = match condition {
                        Some(condition) => self.builder.ins().fcmp(condition, a[component], b[component]),
                        None => {
                            let all = self.builder.ins().iconst(types::I32, -1);
                            self.builder.ins().splat(types::I32X4, all)
                        }
                    };
                    self.assign(self.conditions[component], mask);
                }
                return;
            }
            _ => return,
        };
        self.write(op, result);
    }
}

/// an if's body when it is taken, its else when it is not, and where both
/// come back to, as the interpreter's blocks have them.
fn if_ranges(pc: u32, destination: u32, count: u32) -> ((u32, u32), (u32, u32), u32) {
    let then_end = (pc + 1) + (destination - (pc + 1).min(destination));
    ((pc + 1, then_end), (destination, destination + count), destination + count)
}

/// how deep ifs inside a branch the lanes parted at are compiled for both.
const DEEPEST_PARTED: usize = 8;

impl Program {
    /// whether a range of instructions only works out values, with ifs
    /// inside that come back within it, which compiles for the lanes going
    /// either way of a branch.
    fn plain(&self, (start, end): (u32, u32), depth: usize) -> bool {
        if depth > DEEPEST_PARTED || end as usize > PROGRAM_SIZE {
            return false;
        }
        let mut pc = start;
        while pc < end {
            let op = &self.ops[pc as usize];
            match op.opcode {
                OpCode::IfC | OpCode::IfU => {
                    let (destination, count) = op.instruction.flow_target();
                    let (then, otherwise, next) = if_ranges(pc, destination, count);
                    if next <= pc || next > end || !self.plain(then, depth + 1) || !self.plain(otherwise, depth + 1) {
                        return false;
                    }
                    pc = next;
                }
                OpCode::Call
                | OpCode::CallC
                | OpCode::CallU
                | OpCode::JmpC
                | OpCode::JmpU
                | OpCode::Loop
                | OpCode::Break
                | OpCode::BreakC
                | OpCode::End
                | OpCode::Emit
                | OpCode::SetEmit => return false,
                _ => pc += 1,
            }
        }
        true
    }
}

/// where a flow instruction goes, taken or not, and the frames it is in
/// then, as the interpreter's blocks would be.
fn branch(op: &Op, pc: u32, taken: bool, frames: &[Frame]) -> State {
    let (destination, count) = op.instruction.flow_target();
    let mut frames = frames.to_vec();
    let mut enter = |start: u32, count: u32, return_address: u32| {
        frames.push(Frame { end: start + count, return_address, start, integer: None });
        start
    };
    let next = match op.opcode {
        OpCode::Call | OpCode::CallU | OpCode::CallC if taken => enter(destination, count, pc + 1),
        OpCode::JmpC | OpCode::JmpU if taken => destination,
        OpCode::IfC | OpCode::IfU if taken => enter(pc + 1, destination - (pc + 1).min(destination), destination + count),
        OpCode::IfC | OpCode::IfU => enter(destination, count, destination + count),
        // leave the innermost loop, and any if or call inside it
        OpCode::Break | OpCode::BreakC if taken => {
            let mut next = pc + 1;
            while let Some(frame) = frames.pop() {
                if frame.integer.is_some() {
                    next = frame.return_address;
                    break;
                }
            }
            next
        }
        _ => pc + 1,
    };
    (next, frames)
}
