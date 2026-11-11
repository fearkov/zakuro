//! the shaders the CPU runs compiled to the host's own code, with Cranelift.
//! a compiled program shades four vertices at a time, each register a row
//! of lanes per component the way batch keeps eight, and computes exactly
//! what the interpreter computes, operation for operation. the flow of the
//! program is followed ahead of time: each instruction is compiled once for
//! every way of being in the calls, ifs and loops it is reached in, every
//! such state found first, with the ways into each, and then compiled, a
//! run of states nothing else leads into making up one block. where the
//! vertices of a batch go different ways at a branch, ifs, calls and jumps
//! ahead over instructions that only work out values run for the lanes
//! going each way under masks, and elsewhere it gives the batch back, which
//! batch shades following each way.

use std::collections::HashMap;
use std::mem::offset_of;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{types, AbiParam, Block, FuncRef, InstBuilder, MemFlagsData, Value};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use super::batch::Triangle;
use super::isa::OpCode;
use super::{if_ranges, Op, Operand, Program, ShaderUnit, Vec4, FLOAT_UNIFORMS, INPUT_REGISTERS, OUTPUT_REGISTERS, PROGRAM_SIZE, TEMP_REGISTERS};

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

/// how hard a program is compiled: quickly the first time, with the register
/// allocator that takes a third of the time for code a fifth slower, so that
/// it runs compiled soon after it is first drawn with, and again in full
/// once it has shaded HOT batches, the quick code shading meanwhile.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Tier {
    Quick,
    Full,
}

/// how many batches a program compiled quickly shades before it is compiled
/// again in full.
const HOT: u64 = 4096;

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
        // a whole batch, as nearly all are, reads its lanes without
        // checking each one against the last
        if let Ok(full) = <&[[Vec4; INPUT_REGISTERS]; LANES]>::try_from(inputs) {
            for register in registers(read) {
                for (component, row) in self.inputs[register].iter_mut().enumerate() {
                    *row = std::array::from_fn(|lane| full[lane][register][component]);
                }
            }
            return;
        }
        let last = inputs.len() - 1;
        for register in registers(read) {
            for (component, row) in self.inputs[register].iter_mut().enumerate() {
                *row = std::array::from_fn(|lane| inputs[lane.min(last)][register][component]);
            }
        }
    }

    /// the outputs the program writes out of rows, for the vertices there are.
    fn drain(&self, written: u16, outputs: &mut [[Vec4; OUTPUT_REGISTERS]]) {
        if let Ok(full) = <&mut [[Vec4; OUTPUT_REGISTERS]; LANES]>::try_from(&mut *outputs) {
            for register in registers(written) {
                for (lane, output) in full.iter_mut().enumerate() {
                    output[register] = std::array::from_fn(|component| self.outputs[register][component][lane]);
                }
            }
            return;
        }
        for register in registers(written) {
            for (lane, output) in outputs.iter_mut().enumerate() {
                output[register] = std::array::from_fn(|component| self.outputs[register][component][lane]);
            }
        }
    }
}

/// the registers a mask has, lowest first.
fn registers(mut mask: u16) -> impl Iterator<Item = usize> {
    std::iter::from_fn(move || {
        let register = (mask != 0).then(|| mask.trailing_zeros() as usize)?;
        mask &= mask - 1;
        Some(register)
    })
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
    tier: Tier,
    /// whether it is being compiled again in full.
    upgrading: AtomicBool,
    /// the batches it ran and gave back, it gives up on a program whose
    /// vertices part ways often, which batch shades better. wide enough
    /// not to come round in a long session.
    runs: AtomicU64,
    returned: AtomicU64,
    /// the batches given back where the lanes parted ways, for the tests.
    parted: AtomicU64,
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
        let ran = unsafe { (self.function)(context) } == 0;
        if !ran {
            self.parted.fetch_add(1, Ordering::Relaxed);
        }
        ran
    }

    /// how many batches it gave back where their lanes parted ways, for the
    /// tests.
    #[cfg(test)]
    pub(super) fn parted(&self) -> u64 {
        self.parted.load(Ordering::Relaxed)
    }

    /// how many batches it gave back for any reason, for the tests.
    #[cfg(test)]
    pub(super) fn returned(&self) -> u64 {
        self.returned.load(Ordering::Relaxed)
    }

    /// counts batches run and given back, giving up once a quarter of them
    /// came back.
    fn counted(&self, runs: u64, returned: u64) {
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
        Some(Slot::Done(found)) => {
            let found = found.clone().filter(|compiled| !compiled.given_up.load(Ordering::Relaxed))?;
            if found.tier == Tier::Quick && found.runs.load(Ordering::Relaxed) >= HOT && !found.upgrading.swap(true, Ordering::Relaxed) {
                drop(kept);
                upgrade(program.clone(), key, found.clone());
            }
            return Some(found);
        }
        Some(Slot::Compiling) => return None,
        None => {}
    }
    if kept.len() >= KEPT {
        kept.clear();
    }
    // the tests look at what the compiled code gives from the first run
    if cfg!(test) {
        let found = compile_logged(program, key.1, Tier::Full);
        kept.insert(key, Slot::Done(found.clone()));
        return found;
    }
    kept.insert(key, Slot::Compiling);
    drop(kept);
    let program = program.clone();
    let spawned = std::thread::Builder::new().name("shader jit".to_owned()).spawn(move || {
        let found = compile_logged(&program, key.1, Tier::Quick);
        codes().insert(key, Slot::Done(found));
    });
    if let Err(error) = spawned {
        log::warn!("could not start compiling a shader, {error}");
        codes().insert(key, Slot::Done(None));
    }
    None
}

/// compiles a program in full on a thread of its own, its quick code
/// shading meanwhile, and puts it in the quick code's place, unless that
/// has gone since.
fn upgrade(program: Arc<Program>, key: (u64, u32), quick: Arc<Compiled>) {
    let spawned = std::thread::Builder::new().name("shader jit".to_owned()).spawn(move || {
        let Some(full) = compile_logged(&program, key.1, Tier::Full) else { return };
        let mut kept = codes();
        // a program that gave up while it compiled stays given up
        let given_up = quick.given_up.load(Ordering::Relaxed);
        if !given_up && matches!(kept.get(&key), Some(Slot::Done(Some(current))) if Arc::ptr_eq(current, &quick)) {
            kept.insert(key, Slot::Done(Some(full)));
        }
    });
    if let Err(error) = spawned {
        log::warn!("could not start compiling a shader in full, {error}");
    }
}

/// compiles the program for an entry point, saying why not when it can't.
fn compile_logged(program: &Program, entry: u32, tier: Tier) -> Option<Arc<Compiled>> {
    let start = std::time::Instant::now();
    let found = match compile(program, entry, tier) {
        Ok(compiled) => Some(Arc::new(compiled)),
        Err(error) => {
            log::debug!(target: "zakuro_gpu::shader::jit", "the program {:016X} at {entry} is not compiled, {error}", program.fingerprint);
            None
        }
    };
    log::trace!(
        target: "zakuro_gpu::shader::jit",
        "the program {:016X} at {entry} took {:?} {tier:?}, {} instructions",
        program.fingerprint,
        start.elapsed(),
        program.ops.iter().filter(|op| op.opcode != OpCode::Nop && op.instruction.0 != 0).count()
    );
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

/// what compiling a state does, and the states it goes on to by their
/// number in the graph.
#[derive(Clone, Copy)]
enum Step {
    /// the program ends.
    Finish,
    /// blocks ending there are left, without running anything.
    Leave(usize),
    /// an iteration of a loop ends, aL steps, and it goes again or on.
    Iterate { integer: u32, level: usize, again: usize, done: usize },
    /// an instruction that goes on to one state, a loop it starts at depth
    /// included, or a call or break, which always go.
    Run { pc: u32, depth: usize, next: usize },
    /// a branch on a boolean uniform, taken or not.
    Uniform { pc: u32, taken: usize, other: usize },
    /// a branch on the conditions, taken or not, and how the lanes going
    /// each way run under masks where they part, then on together.
    Conditional { pc: u32, taken: usize, other: usize, parted: Option<(Parted, usize)> },
}

/// how the lanes of a batch that part at a conditional branch each run their
/// own way under masks, see Program::maskable.
#[derive(Clone, Copy)]
enum Parted {
    /// an if's bodies, the lanes taking it running the first and the others
    /// the second, or a call's routine and nothing.
    Bodies([(u32, u32); 2]),
    /// a jump ahead, the lanes not taking it running on from the next
    /// instruction and the others from where it lands, to the end.
    Ahead { lands: u32, end: u32 },
}

impl Step {
    /// the states it goes on to.
    fn onward(&self) -> impl Iterator<Item = usize> {
        let (first, second, third) = match *self {
            Step::Finish => (None, None, None),
            Step::Leave(next) | Step::Run { next, .. } => (Some(next), None, None),
            Step::Iterate { again, done, .. } => (Some(again), Some(done), None),
            Step::Uniform { taken, other, .. } => (Some(taken), Some(other), None),
            Step::Conditional { taken, other, parted, .. } => (Some(taken), Some(other), parted.map(|(_, next)| next)),
        };
        first.into_iter().chain(second).chain(third)
    }
}

/// every state a program reaches from an entry point, found before any is
/// compiled, so that a state nothing else leads into is compiled in the
/// block of the one before it, and a block is sealed once everything that
/// goes into it is there.
struct Graph {
    steps: Vec<Step>,
    /// how many jumps and branches go into each state, the first one's
    /// from the function's start counted, and whether any is a branch.
    into: Vec<u32>,
    branched: Vec<bool>,
    /// the states each ahead of those it leads to, but by going back.
    order: Vec<usize>,
}

impl Graph {
    /// the states from the entry point on, none when there are too many or
    /// they go too deep.
    fn explore(program: &Program, entry: u32) -> Option<Graph> {
        let first: State = (entry, Vec::new());
        let mut index = HashMap::from([(first.clone(), 0)]);
        let mut states = vec![first];
        let mut steps = vec![None];
        let (mut into, mut branched) = (vec![1], vec![false]);
        let mut unplanned = vec![0];
        let mut masked = 0;
        while let Some(at) = unplanned.pop() {
            let state = states[at].clone();
            let mut reach = |state: State, branch: bool| -> Option<usize> {
                let found = match index.get(&state) {
                    Some(&found) => found,
                    None => {
                        if states.len() > MOST_STATES {
                            return None;
                        }
                        let found = states.len();
                        index.insert(state.clone(), found);
                        states.push(state);
                        steps.push(None);
                        into.push(0);
                        branched.push(false);
                        unplanned.push(found);
                        found
                    }
                };
                into[found] += 1;
                branched[found] |= branch;
                Some(found)
            };
            let step = plan(program, &state, &mut masked, &mut reach)?;
            steps[at] = Some(step);
        }
        let steps: Vec<Step> = steps.into_iter().map(|step| step.expect("every state planned")).collect();
        // depth first, each state after all it leads to, then the other way
        let mut seen = vec![false; steps.len()];
        let mut order = Vec::with_capacity(steps.len());
        let mut walk = vec![(0, steps[0].onward())];
        seen[0] = true;
        while let Some((at, onward)) = walk.last_mut() {
            match onward.find(|&next| !seen[next]) {
                Some(next) => {
                    seen[next] = true;
                    walk.push((next, steps[next].onward()));
                }
                None => {
                    order.push(*at);
                    walk.pop();
                }
            }
        }
        order.reverse();
        Some(Graph { steps, into, branched, order })
    }
}

/// what a state does, the states it goes on to found or added by reach,
/// with whether it branches to them. none where it goes too deep.
/// masked counts the instructions compiled under masks so far, see parted.
fn plan(program: &Program, (pc, frames): &State, masked: &mut usize, reach: &mut impl FnMut(State, bool) -> Option<usize>) -> Option<Step> {
    let (mut pc, mut frames) = (*pc, frames.clone());
    let depth = frames.len();
    // the blocks ending there, as the interpreter leaves them
    while let Some(top) = frames.last().copied() {
        if pc != top.end {
            break;
        }
        if let Some(integer) = top.integer {
            let level = frames.len() - 1;
            let again = reach((top.start, frames.clone()), false)?;
            frames.pop();
            let done = reach((top.return_address, frames), true)?;
            return Some(Step::Iterate { integer, level, again, done });
        }
        pc = top.return_address;
        frames.pop();
    }
    if frames.len() != depth {
        return Some(Step::Leave(reach((pc, frames), false)?));
    }
    if pc as usize >= PROGRAM_SIZE {
        return Some(Step::Finish);
    }
    let op = &program.ops[pc as usize];
    let next = pc + 1;
    match op.opcode {
        OpCode::End => Some(Step::Finish),
        OpCode::Loop => {
            if depth >= DEEPEST {
                return None;
            }
            let integer = op.instruction.integer_index() & 3;
            let (last, _) = op.instruction.flow_target();
            frames.push(Frame { end: last + 1, return_address: last + 1, start: next, integer: Some(integer) });
            Some(Step::Run { pc, depth, next: reach((next, frames), false)? })
        }
        OpCode::Call
        | OpCode::CallU
        | OpCode::CallC
        | OpCode::JmpC
        | OpCode::JmpU
        | OpCode::IfC
        | OpCode::IfU
        | OpCode::Break
        | OpCode::BreakC => {
            let taken = branch(op, pc, true, &frames);
            if taken.1.len() > DEEPEST {
                return None;
            }
            let other = branch(op, pc, false, &frames);
            match op.opcode {
                OpCode::Call | OpCode::Break => Some(Step::Run { pc, depth, next: reach(taken, false)? }),
                OpCode::CallU | OpCode::IfU | OpCode::JmpU => {
                    Some(Step::Uniform { pc, taken: reach(taken, true)?, other: reach(other, true)? })
                }
                _ => {
                    let (taken, other) = (reach(taken, true)?, reach(other, true)?);
                    let parted = match parted(program, op, pc, &frames, masked) {
                        Some((parted, next)) => Some((parted, reach((next, frames), false)?)),
                        None => None,
                    };
                    Some(Step::Conditional { pc, taken, other, parted })
                }
            }
        }
        _ => Some(Step::Run { pc, depth, next: reach((next, frames), false)? }),
    }
}

/// how the lanes of a batch run their own ways where they part at a
/// conditional branch, an if, a call or a jump ahead over instructions that
/// only work out values, and where all of them go on from, as they would
/// have going each way. jumps ahead overlapping one another each compile
/// what the others jumped over, so past MOST_STATES instructions compiled
/// under masks, counted in masked, the batch goes back instead.
fn parted(program: &Program, op: &Op, pc: u32, frames: &[Frame], masked: &mut usize) -> Option<(Parted, u32)> {
    let (destination, count) = op.instruction.flow_target();
    let parted = match op.opcode {
        OpCode::IfC => {
            let (then, otherwise, next) = if_ranges(pc, destination, count);
            (program.maskable(then, 0, &[]) && program.maskable(otherwise, 0, &[])).then_some((Parted::Bodies([then, otherwise]), next))
        }
        OpCode::CallC => {
            let routine = (destination, destination + count);
            program.maskable(routine, 0, &[]).then_some((Parted::Bodies([routine, (pc + 1, pc + 1)]), pc + 1))
        }
        // the lanes not jumping must not leave a block on the way to where
        // all of them meet
        OpCode::JmpC if destination > pc => {
            let end = program.ahead(pc, destination)?;
            frames.last().is_none_or(|top| !(pc + 1..end).contains(&top.end)).then_some((Parted::Ahead { lands: destination, end }, end))
        }
        _ => None,
    }?;
    let size = match parted.0 {
        Parted::Bodies([first, second]) => (first.1 - first.0 + second.1 - second.0) as usize,
        Parted::Ahead { end, .. } => (end - pc - 1) as usize,
    };
    *masked += size;
    (*masked <= MOST_STATES).then_some(parted)
}

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
pub(super) fn compile(program: &Program, entry: u32, tier: Tier) -> Result<Compiled, String> {
    let mut flags = settings::builder();
    // the verifier looks the code over for Cranelift's own sake, a quarter
    // of the time compiling took, the tests have it
    let verify = if cfg!(debug_assertions) { "true" } else { "false" };
    let allocator = match tier {
        Tier::Quick => "single_pass",
        Tier::Full => "backtracking",
    };
    let flagged = [
        ("opt_level", "speed"),
        ("use_colocated_libcalls", "false"),
        ("is_pic", "false"),
        ("enable_verifier", verify),
        ("regalloc_algorithm", allocator),
        ("enable_probestack", "true"),
        ("probestack_strategy", "inline"),
    ];
    for (name, value) in flagged {
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

    // none when its states go too many ways or too deep, found before
    // anything is built
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
        tier,
        upgrading: AtomicBool::new(false),
        runs: AtomicU64::new(0),
        returned: AtomicU64::new(0),
        parted: AtomicU64::new(0),
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
    /// the instructions the interpreter would still run, counted only in a
    /// program that can come back to an instruction it ran, see may_loop.
    budget: Variable,
    counted: bool,
    graph: Graph,
    /// each state's block, made when something first jumps or branches to
    /// it, none for a state compiled in the block of the one before.
    blocks: Vec<Option<Block>>,
    /// whether each state's code is there.
    done: Vec<bool>,
    /// the jumps and branches still to come into each state's block, which
    /// is sealed once there are none, so that the variables read in it are
    /// looked up right away rather than all left open to the end.
    awaited: Vec<u32>,
    /// the states the jump or branch being made goes to, see to.
    landing: Vec<usize>,
    /// jumps back to a state compiled before, which may close a loop, and so
    /// look at the budget first, the block that looks and the state.
    checks: Vec<(Block, usize)>,
    /// gives the batch back, where its lanes parted ways.
    give_back: Block,
    /// the lanes the code being compiled runs for, all of them when none,
    /// the others keeping what they had, inside a branch the vertices
    /// parted at.
    mask: Option<Value>,
}

impl<'a, 'b, 'c> Compiler<'a, 'b, 'c> {
    fn build(builder: &'a mut FunctionBuilder<'b>, program: &'c Program, entry: u32, helpers: Helpers) -> Option<()> {
        let graph = Graph::explore(program, entry)?;
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
            counted: program.may_loop(),
            blocks: vec![None; graph.steps.len()],
            done: vec![false; graph.steps.len()],
            awaited: graph.into.clone(),
            landing: Vec::new(),
            graph,
            checks: Vec::new(),
            give_back,
            mask: None,
        };
        compiler.builder.seal_block(start);
        // the first state, in the start's block unless something goes back
        // to it, then the rest, each with its block ahead of the states it
        // leads to but by going back
        compiler.onward(0);
        compiler.check_budget();
        for at in 0..compiler.graph.order.len() {
            let state = compiler.graph.order[at];
            if compiler.done[state] {
                continue;
            }
            let block = compiler.block(state);
            compiler.builder.switch_to_block(block);
            compiler.emit(state);
            compiler.check_budget();
        }
        compiler.builder.switch_to_block(give_back);
        let returned = compiler.builder.ins().iconst(types::I32, 1);
        compiler.builder.ins().return_(&[returned]);
        Some(())
    }

    /// an instruction run, counted against the budget where it is kept.
    fn spend(&mut self) {
        if self.counted {
            let budget = self.builder.use_var(self.budget);
            let spent = self.builder.ins().iadd_imm_s(budget, -1);
            self.builder.def_var(self.budget, spent);
        }
    }

    /// fills in the blocks that look at the budget before going back to a
    /// state, the batch going back once the interpreter would have stopped.
    fn check_budget(&mut self) {
        while let Some((check, state)) = self.checks.pop() {
            self.builder.seal_block(check);
            self.builder.switch_to_block(check);
            let budget = self.builder.use_var(self.budget);
            let low = self.builder.ins().icmp_imm_s(IntCC::SignedLessThan, budget, MOST_STATES as i64);
            let target = self.block(state);
            self.landing.push(state);
            self.builder.ins().brif(low, self.give_back, &[], target, &[]);
            self.landed();
        }
    }

    /// a state's block, made the first time it is asked for.
    fn block(&mut self, state: usize) -> Block {
        *self.blocks[state].get_or_insert_with(|| self.builder.create_block())
    }

    /// the block a jump or branch to a state goes to: the state's own, or,
    /// going back to one compiled before in a program that keeps the budget,
    /// one that looks at the budget first. the state is counted in once the
    /// jump or branch is there, see landed.
    fn to(&mut self, state: usize) -> Block {
        if self.counted && self.done[state] {
            let check = self.builder.create_block();
            self.checks.push((check, state));
            return check;
        }
        self.landing.push(state);
        self.block(state)
    }

    /// counts in the jump or branch just made to the states to gave, sealing
    /// each block nothing more goes into.
    fn landed(&mut self) {
        while let Some(state) = self.landing.pop() {
            self.awaited[state] -= 1;
            if self.awaited[state] == 0 {
                let block = self.block(state);
                self.builder.seal_block(block);
            }
        }
    }

    /// whether a state is compiled where the one jumping to it is, nothing
    /// else leading into it.
    fn inline(&self, state: usize) -> bool {
        self.graph.into[state] == 1 && !self.graph.branched[state]
    }

    /// goes on to a state with a jump, compiling it right there when nothing
    /// else leads into it.
    fn onward(&mut self, state: usize) {
        if self.inline(state) {
            self.emit(state);
            return;
        }
        let target = self.to(state);
        self.builder.ins().jump(target, &[]);
        self.landed();
    }

    /// compiles a state, and on into the states only it leads to.
    fn emit(&mut self, mut state: usize) {
        loop {
            self.done[state] = true;
            let next = match self.graph.steps[state] {
                Step::Finish => {
                    self.finish();
                    return;
                }
                Step::Leave(next) => next,
                Step::Iterate { integer, level, again, done } => {
                    // an iteration of a loop ends, aL steps, and it goes again
                    // while it has repeats left
                    let at = offset_of!(Context, ints) as u32 + integer * 4 + 2;
                    let increment = self.builder.ins().sload8(types::I32, constant(), self.context, at as i32);
                    let counter = self.builder.use_var(self.loop_counter);
                    let stepped = self.builder.ins().iadd(counter, increment);
                    self.builder.def_var(self.loop_counter, stepped);
                    let repeats = self.builder.use_var(self.repeats[level]);
                    let more = self.builder.create_block();
                    let done = self.to(done);
                    self.builder.ins().brif(repeats, more, &[], done, &[]);
                    self.landed();
                    self.builder.seal_block(more);
                    self.builder.switch_to_block(more);
                    let fewer = self.builder.ins().iadd_imm_s(repeats, -1);
                    self.builder.def_var(self.repeats[level], fewer);
                    again
                }
                Step::Run { pc, depth, next } => {
                    self.spend();
                    self.instruction(pc, depth);
                    next
                }
                Step::Uniform { pc, taken, other } => {
                    self.spend();
                    self.uniform(pc, taken, other);
                    return;
                }
                Step::Conditional { pc, taken, other, parted } => {
                    self.spend();
                    match self.conditional(pc, taken, other, parted) {
                        Some(next) => next,
                        None => return,
                    }
                }
            };
            if !self.inline(next) {
                let target = self.to(next);
                self.builder.ins().jump(target, &[]);
                self.landed();
                return;
            }
            state = next;
        }
    }

    /// an instruction that goes on to the next state.
    fn instruction(&mut self, pc: u32, depth: usize) {
        let op = self.program.ops[pc as usize];
        let instruction = op.instruction;
        match op.opcode {
            OpCode::SetEmit => {
                let slot = self.builder.ins().iconst(types::I32, instruction.emit_vertex().min(2) as i64);
                let completes = self.builder.ins().iconst(types::I32, instruction.emit_primitive() as i64);
                self.builder.ins().call(self.helpers.set_emit, &[self.context, slot, completes]);
            }
            OpCode::Emit => {
                self.store_outputs();
                self.builder.ins().call(self.helpers.emit, &[self.context]);
            }
            OpCode::Loop => {
                let integer = instruction.integer_index() & 3;
                let at = offset_of!(Context, ints) as u32 + integer * 4;
                let repeats = self.builder.ins().uload8(types::I32, constant(), self.context, at as i32);
                let first = self.builder.ins().uload8(types::I32, constant(), self.context, (at + 1) as i32);
                self.builder.def_var(self.loop_counter, first);
                self.builder.def_var(self.repeats[depth], repeats);
            }
            // calls and breaks are where they go
            OpCode::Nop | OpCode::Call | OpCode::Break => {}
            _ => self.arithmetic(&op),
        }
    }

    /// a branch on a boolean uniform, every lane the same way.
    fn uniform(&mut self, pc: u32, taken: usize, other: usize) {
        let op = self.program.ops[pc as usize];
        let instruction = op.instruction;
        let bools = self.builder.ins().load(types::I32, constant(), self.context, offset_of!(Context, bools) as i32);
        let set = self.builder.ins().band_imm_u(bools, 1i64 << instruction.bool_index());
        // jmpu can test for either value, the low bit of its count inverts it
        let inverted = op.opcode == OpCode::JmpU && instruction.flow_target().1 & 1 != 0;
        let (taken, other) = (self.to(taken), self.to(other));
        if inverted {
            self.builder.ins().brif(set, other, &[], taken, &[]);
        } else {
            self.builder.ins().brif(set, taken, &[], other, &[]);
        }
        self.landed();
    }

    /// a branch on the conditions, every lane the same way, or where they
    /// part each running its own way under masks, else the batch going
    /// back. the state they go on to together comes back.
    fn conditional(&mut self, pc: u32, taken: usize, other: usize, parted: Option<(Parted, usize)>) -> Option<usize> {
        let lanes = self.condition(self.program.ops[pc as usize].instruction);
        let all = self.builder.ins().vall_true(lanes);
        let none_check = self.builder.create_block();
        let taken = self.to(taken);
        self.builder.ins().brif(all, taken, &[], none_check, &[]);
        self.landed();
        self.builder.seal_block(none_check);
        self.builder.switch_to_block(none_check);
        let any = self.builder.ins().vany_true(lanes);
        let other = self.to(other);
        let Some((parted, next)) = parted else {
            self.builder.ins().brif(any, self.give_back, &[], other, &[]);
            self.landed();
            return None;
        };
        // where the lanes part, an if, a call or a jump ahead over what only
        // works out values runs for the lanes going each way, the others
        // keeping theirs, as they would have going the other way
        let both = self.builder.create_block();
        self.builder.ins().brif(any, both, &[], other, &[]);
        self.landed();
        self.builder.seal_block(both);
        self.builder.switch_to_block(both);
        let rest = self.builder.ins().bnot(lanes);
        match parted {
            Parted::Bodies([first, second]) => {
                self.predicated(first, lanes, Vec::new());
                self.predicated(second, rest, Vec::new());
            }
            Parted::Ahead { lands, end } => self.predicated((pc + 1, end), rest, vec![(lands, lanes)]),
        }
        Some(next)
    }

    /// compiles a range Program::maskable allows for the lanes in mask, ifs
    /// inside narrowing it, jumps ahead taking their lanes to where they land.
    /// lands are lanes that jumps from before it land later in it.
    fn predicated(&mut self, (start, end): (u32, u32), mask: Value, mut lands: Vec<(u32, Value)>) {
        let mut flowing = mask;
        let mut pc = start;
        while pc < end {
            // the lanes landing here run on with those already here
            for at in (0..lands.len()).rev() {
                if lands[at].0 == pc {
                    let (_, landed) = lands.swap_remove(at);
                    flowing = self.builder.ins().bor(flowing, landed);
                }
            }
            let op = self.program.ops[pc as usize];
            self.spend();
            let (destination, count) = op.instruction.flow_target();
            match op.opcode {
                OpCode::IfC | OpCode::IfU => {
                    let lanes = match op.opcode {
                        OpCode::IfC => self.condition(op.instruction),
                        _ => self.uniform_lanes(op.instruction.bool_index(), false),
                    };
                    let (then, otherwise, next) = if_ranges(pc, destination, count);
                    let taken = self.builder.ins().band(flowing, lanes);
                    let others = self.builder.ins().band_not(flowing, lanes);
                    self.predicated(then, taken, Vec::new());
                    self.predicated(otherwise, others, Vec::new());
                    pc = next;
                }
                OpCode::JmpC | OpCode::JmpU => {
                    // jmpu can test for either value, the low bit of its count
                    // inverts it
                    let lanes = match op.opcode {
                        OpCode::JmpC => self.condition(op.instruction),
                        _ => self.uniform_lanes(op.instruction.bool_index(), count & 1 != 0),
                    };
                    let jumping = self.builder.ins().band(flowing, lanes);
                    lands.push((destination, jumping));
                    flowing = self.builder.ins().band_not(flowing, lanes);
                    pc += 1;
                }
                _ => {
                    let outer = self.mask.replace(flowing);
                    self.arithmetic(&op);
                    self.mask = outer;
                    pc += 1;
                }
            }
        }
    }

    /// a boolean uniform in every lane, all ones where it is set, or where it
    /// is not when inverted.
    fn uniform_lanes(&mut self, index: u32, inverted: bool) -> Value {
        let bools = self.builder.ins().load(types::I32, constant(), self.context, offset_of!(Context, bools) as i32);
        let set = self.builder.ins().band_imm_u(bools, 1i64 << index);
        let test = if inverted { IntCC::Equal } else { IntCC::NotEqual };
        let set = self.builder.ins().icmp_imm_u(test, set, 0);
        let set = self.builder.ins().bmask(types::I32, set);
        self.builder.ins().splat(types::I32X4, set)
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
    fn source(&mut self, operand: &Operand) -> [Value; 4] {
        let flags = constant();
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

    /// the shader's multiply, zero rather than NaN for zero times infinity,
    /// a NaN product of two numbers zeroed in each lane the way the
    /// interpreter's is, without a branch.
    fn multiply(&mut self, a: Value, b: Value) -> Value {
        let product = self.builder.ins().fmul(a, b);
        let nan = self.builder.ins().fcmp(FloatCC::Unordered, product, product);
        let numbers = self.builder.ins().fcmp(FloatCC::Ordered, a, b);
        let zeroed = self.builder.ins().band(nan, numbers);
        let zeroed = self.floats(zeroed);
        // the bits of +0.0 are none, one and-not on x86
        self.builder.ins().band_not(product, zeroed)
    }

    /// a dot product of an instruction's two sources over the first count
    /// components, which adds up exactly the way the interpreter's does, its
    /// fourth source's w added after for dph.
    fn dot(&mut self, a: &[Value; 4], b: &[Value; 4], count: usize, w: bool) -> Value {
        let start = std::iter::empty::<f32>().sum::<f32>();
        let mut sum = self.splat(start);
        for component in 0..count {
            let product = self.multiply(a[component], b[component]);
            sum = self.builder.ins().fadd(sum, product);
        }
        if w {
            sum = self.builder.ins().fadd(sum, b[3]);
        }
        sum
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
        let values: Vec<[Value; 4]> = op.sources[..sources].iter().map(|operand| self.source(operand)).collect();
        let (a, b) = (values[0], values[1]);
        let pairs = |compiler: &mut Self, f: fn(&mut Self, Value, Value) -> Value| -> [Value; 4] {
            std::array::from_fn(|component| f(compiler, a[component], b[component]))
        };
        let result: [Value; 4] = match op.opcode {
            OpCode::Add => pairs(self, |c, x, y| c.builder.ins().fadd(x, y)),
            OpCode::Mul => pairs(self, |c, x, y| c.multiply(x, y)),
            OpCode::Mad | OpCode::MadI => {
                let c = values[2];
                std::array::from_fn(|component| {
                    let product = self.multiply(a[component], b[component]);
                    self.builder.ins().fadd(product, c[component])
                })
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
            OpCode::Dp3 => [self.dot(&a, &b, 3, false); 4],
            OpCode::Dp4 => [self.dot(&a, &b, 4, false); 4],
            OpCode::Dph | OpCode::DphI => [self.dot(&a, &b, 3, true); 4],
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
                let one = self.splat(1.0);
                let product = self.multiply(a[1], b[1]);
                [one, product, a[2], b[3]]
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

/// how deep ifs inside a branch the lanes parted at are compiled for both.
const DEEPEST_PARTED: usize = 8;

impl Program {
    /// whether the program can come back to an instruction it ran, by a
    /// loop, a jump back, an if whose blocks come back to before it, or a
    /// break, which out of any loop leaves every block and goes on after
    /// itself, before the call that may have brought it there. one that
    /// can't runs each of its ways of reaching an instruction at most once,
    /// far fewer instructions than the interpreter stops a program at.
    fn may_loop(&self) -> bool {
        self.ops.iter().zip(0u32..).any(|(op, pc)| {
            let (destination, count) = op.instruction.flow_target();
            match op.opcode {
                OpCode::Loop | OpCode::Break | OpCode::BreakC => true,
                OpCode::JmpC | OpCode::JmpU => destination <= pc,
                OpCode::IfC | OpCode::IfU => destination + count <= pc,
                _ => false,
            }
        })
    }

    /// whether a range of instructions only works out values, with ifs
    /// inside that come back within it and jumps ahead that land in it, no
    /// deeper in ifs than they are, which compiles for the lanes going
    /// either way of a branch under masks. lands are where jumps from before
    /// the range land in it, which have to be at its own depth too.
    fn maskable(&self, (start, end): (u32, u32), depth: usize, lands: &[u32]) -> bool {
        if depth > DEEPEST_PARTED || start > end || end as usize > PROGRAM_SIZE {
            return false;
        }
        // the instructions at this depth, where a jump may land
        let mut level = Vec::new();
        let mut landing = lands.to_vec();
        let mut pc = start;
        while pc < end {
            level.push(pc);
            let op = &self.ops[pc as usize];
            let (destination, count) = op.instruction.flow_target();
            match op.opcode {
                OpCode::IfC | OpCode::IfU => {
                    let (then, otherwise, next) = if_ranges(pc, destination, count);
                    if next <= pc || next > end || !self.maskable(then, depth + 1, &[]) || !self.maskable(otherwise, depth + 1, &[]) {
                        return false;
                    }
                    pc = next;
                }
                OpCode::JmpC | OpCode::JmpU if pc < destination && destination <= end => {
                    landing.push(destination);
                    pc += 1;
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
        landing.iter().all(|&at| at == end || level.binary_search(&at).is_ok())
    }

    /// where the lanes that part at a jump ahead from pc to destination all
    /// run on together again, the furthest the jumps between land, when what
    /// they run apart is maskable.
    fn ahead(&self, pc: u32, destination: u32) -> Option<u32> {
        let mut end = destination;
        let mut at = pc + 1;
        while at < end && (at as usize) < PROGRAM_SIZE {
            let op = &self.ops[at as usize];
            let (target, count) = op.instruction.flow_target();
            match op.opcode {
                OpCode::JmpC | OpCode::JmpU if target > at => end = end.max(target),
                OpCode::IfC | OpCode::IfU => {
                    let (_, _, next) = if_ranges(at, target, count);
                    if next > at {
                        end = end.max(next);
                        at = next;
                        continue;
                    }
                }
                _ => {}
            }
            at += 1;
        }
        self.maskable((pc + 1, end), 0, &[destination]).then_some(end)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// quick code asked for once it has shaded HOT batches comes back as it
    /// is while the program compiles in full, which then takes its place.
    #[test]
    fn hot_quick_code_is_compiled_again_in_full() {
        let mut unit = ShaderUnit::new();
        // mov o5, c77, which no other test runs
        unit.program[0] = (0x13 << 26) | (0x05 << 21) | ((0x20 + 77) << 12);
        unit.program[1] = 0x22 << 26;
        unit.prepare();
        let program = unit.decoded.clone().expect("prepared");
        let key = (program.fingerprint, unit.entry_point);
        let quick = Arc::new(compile(&program, unit.entry_point, Tier::Quick).expect("compiled"));
        quick.runs.store(HOT, Ordering::Relaxed);
        codes().insert(key, Slot::Done(Some(quick.clone())));
        let found = compiled(&unit, &program).expect("the quick code");
        assert!(Arc::ptr_eq(&found, &quick));
        let start = std::time::Instant::now();
        while !matches!(codes().get(&key), Some(Slot::Done(Some(current))) if current.tier == Tier::Full) {
            assert!(start.elapsed().as_secs() < 60, "never compiled in full");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }
}
