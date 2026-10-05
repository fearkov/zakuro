//! the vertex and geometry shaders run over several vertices or primitives
//! at once, one lane each, so
//! that every instruction works through all of them together with vector
//! instructions. a lane computes exactly what the interpreter computes for
//! its vertex, operation for operation, and the rare operations run lane by
//! lane through the interpreter's own arithmetic. where the vertices branch
//! apart, the ones going the other way carry on in a copy of the batch.

use wide::f32x8;

use super::isa::OpCode;
use super::{Block, Op, Operand, Program, ShaderUnit, Vec4};
use super::{FLOAT_UNIFORMS, INPUT_REGISTERS, OUTPUT_REGISTERS, PROGRAM_SIZE, TEMP_REGISTERS};

/// how many vertices a batch shades together.
pub const LANES: usize = 8;

/// one component of a register, for every lane.
type Lanes = f32x8;
/// a register for every lane, a row of lanes per component.
type Wide = [Lanes; 4];

const ZERO: Wide = [f32x8::ZERO; 4];

/// super::TEMP_START in every lane.
const TEMP_START: Wide = [f32x8::ZERO, f32x8::ZERO, f32x8::ZERO, f32x8::ONE];

/// the float uniforms in every lane, put there once for all of a draw's
/// batches instead of at every read.
pub(super) struct Uniforms([Wide; FLOAT_UNIFORMS]);

impl Uniforms {
    pub(super) fn new(unit: &ShaderUnit) -> Uniforms {
        Uniforms(unit.float_uniforms.map(|uniform| uniform.map(f32x8::splat)))
    }

    /// a uniform, none past the last one.
    #[inline(always)]
    fn at(&self, index: i32) -> &Wide {
        match usize::try_from(index).ok().and_then(|index| self.0.get(index)) {
            Some(value) => value,
            None => &ZERO,
        }
    }
}

/// a batch of vertices' registers.
#[derive(Clone)]
struct Batch {
    input: [Wide; INPUT_REGISTERS],
    output: [Wide; OUTPUT_REGISTERS],
    temp: [Wide; TEMP_REGISTERS],
    /// a0 and a1 for every lane, the loop counter is the same for all.
    address: [[i32; LANES]; 2],
    loop_counter: i32,
    condition: [[bool; LANES]; 2],
}

/// every lane, one bit each.
const ALL: u8 = u8::MAX;

/// the output registers of lanes that went their own way at a branch.
type Fork = (u8, [Wide; OUTPUT_REGISTERS]);

/// one emitted triangle, as each vertex's output registers.
pub(super) type Triangle = [[Vec4; OUTPUT_REGISTERS]; 3];

/// what the lanes of a geometry shader emit, each its own vertices and
/// triangles, the way the interpreter's emitter keeps them for one.
pub(super) struct Emitters {
    slot: [usize; LANES],
    completes: [bool; LANES],
    slots: [Triangle; LANES],
    pub(super) triangles: [Vec<Triangle>; LANES],
}

impl Emitters {
    pub(super) fn new() -> Emitters {
        Emitters {
            slot: [0; LANES],
            completes: [false; LANES],
            slots: [[[[0.0; 4]; OUTPUT_REGISTERS]; 3]; LANES],
            triangles: std::array::from_fn(|_| Vec::new()),
        }
    }
}

/// shades up to LANES vertices, writing their output registers.
pub(super) fn run(
    unit: &ShaderUnit,
    uniforms: &Uniforms,
    program: &Program,
    inputs: &[[Vec4; INPUT_REGISTERS]],
    outputs: &mut [[Vec4; OUTPUT_REGISTERS]],
    blocks: &mut Vec<Block>,
    forks: &mut Vec<Fork>,
) {
    debug_assert!(!inputs.is_empty() && inputs.len() <= LANES);
    let mut batch = start(program, inputs);
    blocks.clear();
    forks.clear();
    let finished = execute(unit, uniforms, program, &mut batch, blocks, forks, unit.entry_point, 0, ALL, None);
    forks.push((finished, batch.output));
    // each lane's outputs from the copy it finished in. the registers
    // nothing writes stay zero, as they come
    for (lanes, registers) in forks.iter() {
        for register in (0..OUTPUT_REGISTERS).filter(|register| program.outputs & (1 << register) != 0) {
            for component in 0..4 {
                let values = registers[register][component].to_array();
                for (lane, output) in outputs.iter_mut().enumerate() {
                    if lanes & (1 << lane) != 0 {
                        output[register][component] = values[lane];
                    }
                }
            }
        }
    }
}

/// runs the geometry shader over up to LANES primitives' inputs, each
/// lane's triangles going in its emitter. lanes past the last primitive
/// repeat it and what they emit is theirs to drop.
pub(super) fn run_geometry(
    unit: &ShaderUnit,
    uniforms: &Uniforms,
    program: &Program,
    inputs: &[[Vec4; INPUT_REGISTERS]],
    emitters: &mut Emitters,
    blocks: &mut Vec<Block>,
    forks: &mut Vec<Fork>,
) {
    debug_assert!(!inputs.is_empty() && inputs.len() <= LANES);
    let mut batch = start(program, inputs);
    blocks.clear();
    forks.clear();
    execute(unit, uniforms, program, &mut batch, blocks, forks, unit.entry_point, 0, ALL, Some(emitters));
}

/// a batch with the inputs in its lanes and everything else zero.
fn start(program: &Program, inputs: &[[Vec4; INPUT_REGISTERS]]) -> Batch {
    let mut batch = Batch {
        input: [ZERO; INPUT_REGISTERS],
        output: [ZERO; OUTPUT_REGISTERS],
        temp: [TEMP_START; TEMP_REGISTERS],
        address: [[0; LANES]; 2],
        loop_counter: 0,
        condition: [[false; LANES]; 2],
    };
    // lanes past the last vertex repeat it, so that they branch its way.
    // only the registers the program reads matter
    let last = inputs.len() - 1;
    for register in (0..INPUT_REGISTERS).filter(|register| program.inputs & (1 << register) != 0) {
        for (component, row) in batch.input[register].iter_mut().enumerate() {
            let lanes: [f32; LANES] = std::array::from_fn(|lane| inputs[lane.min(last)][register][component]);
            *row = f32x8::from(lanes);
        }
    }
    batch
}

/// each active lane's output registers into its emitter's slot, a
/// triangle out when the slot completes one.
fn emit(batch: &Batch, program: &Program, active: u8, emitters: &mut Emitters) {
    for register in (0..OUTPUT_REGISTERS).filter(|register| program.outputs & (1 << register) != 0) {
        for component in 0..4 {
            let values = batch.output[register][component].to_array();
            for lane in (0..LANES).filter(|lane| active & (1 << lane) != 0) {
                let slot = emitters.slot[lane];
                emitters.slots[lane][slot][register][component] = values[lane];
            }
        }
    }
    for lane in (0..LANES).filter(|lane| active & (1 << lane) != 0) {
        if emitters.completes[lane] {
            let triangle = emitters.slots[lane];
            emitters.triangles[lane].push(triangle);
        }
    }
}

/// runs the program over the batch from pc on, for the lanes in active,
/// which all go the same way. at a branch they part at, the ones going the
/// other way carry on in a copy of the batch, whose outputs go in forks.
/// the lanes still here at the end come back.
#[allow(clippy::too_many_arguments)]
fn execute(
    unit: &ShaderUnit,
    uniforms: &Uniforms,
    program: &Program,
    batch: &mut Batch,
    blocks: &mut Vec<Block>,
    forks: &mut Vec<Fork>,
    mut pc: u32,
    mut budget: u32,
    mut active: u8,
    mut emitters: Option<&mut Emitters>,
) -> u8 {
    // where uniforms go to be read like registers
    let mut scratch = [ZERO; 3];

    loop {
        while let Some(top) = blocks.last_mut() {
            if pc != top.end {
                break;
            }
            batch.loop_counter = batch.loop_counter.wrapping_add(top.increment);
            if top.repeat == 0 {
                pc = top.return_address;
                blocks.pop();
            } else {
                top.repeat -= 1;
                pc = top.start;
            }
        }
        // a program that never ends is a decoding bug, the interpreter caps
        // it and so does this
        if pc as usize >= PROGRAM_SIZE || budget >= 0x10000 {
            break;
        }
        budget += 1;

        let op = &program.ops[pc as usize];
        match op.opcode {
            OpCode::End => break,
            OpCode::Nop => pc += 1,
            // what a vertex shader runs has nothing to emit to
            OpCode::SetEmit => {
                if let Some(emitters) = emitters.as_deref_mut() {
                    let (slot, completes) = (op.instruction.emit_vertex().min(2), op.instruction.emit_primitive());
                    for lane in (0..LANES).filter(|lane| active & (1 << lane) != 0) {
                        emitters.slot[lane] = slot;
                        emitters.completes[lane] = completes;
                    }
                }
                pc += 1;
            }
            OpCode::Emit => {
                if let Some(emitters) = emitters.as_deref_mut() {
                    emit(batch, program, active, emitters);
                }
                pc += 1;
            }
            OpCode::Mad | OpCode::MadI => {
                let [sa, sb, sc] = &mut scratch;
                let a = operand(unit, uniforms, batch, &op.sources[0], sa);
                let b = operand(unit, uniforms, batch, &op.sources[1], sb);
                let c = operand(unit, uniforms, batch, &op.sources[2], sc);
                let result: Wide = std::array::from_fn(|i| multiply(a.row(i), b.row(i)) + c.row(i));
                write(batch, op, &result);
                pc += 1;
            }
            OpCode::Loop => {
                let instruction = op.instruction;
                let integers = unit.int_uniforms[instruction.integer_index() as usize & 3];
                let (last, _) = instruction.flow_target();
                batch.loop_counter = integers[1] as i32;
                blocks.push(Block {
                    end: last + 1,
                    return_address: last + 1,
                    repeat: integers[0] as u32,
                    increment: integers[2] as i8 as i32,
                    start: pc + 1,
                    is_loop: true,
                });
                pc += 1;
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
                let taken = condition(unit, batch, op) & active;
                let others = active & !taken;
                if taken != 0 && others != 0 {
                    // the lanes that do not take it carry on in a copy
                    let mut copy = batch.clone();
                    let mut copy_blocks = blocks.clone();
                    let next = branch(op, pc, false, &mut copy_blocks);
                    let finished =
                        execute(unit, uniforms, program, &mut copy, &mut copy_blocks, forks, next, budget, others, emitters.as_deref_mut());
                    forks.push((finished, copy.output));
                    active = taken;
                }
                pc = branch(op, pc, taken != 0, blocks);
            }
            _ => {
                arithmetic(unit, uniforms, batch, op, &mut scratch);
                pc += 1;
            }
        }
    }
    active
}

/// where a flow instruction goes, taken or not, entering or leaving the
/// blocks that takes.
fn branch(op: &Op, pc: u32, taken: bool, blocks: &mut Vec<Block>) -> u32 {
    let (destination, count) = op.instruction.flow_target();
    let mut enter = |start: u32, count: u32, return_address: u32| {
        blocks.push(Block { end: start + count, return_address, repeat: 0, increment: 0, start, is_loop: false });
        start
    };
    match op.opcode {
        OpCode::Call | OpCode::CallU | OpCode::CallC if taken => enter(destination, count, pc + 1),
        OpCode::JmpC | OpCode::JmpU if taken => destination,
        OpCode::IfC | OpCode::IfU if taken => enter(pc + 1, destination - (pc + 1).min(destination), destination + count),
        OpCode::IfC | OpCode::IfU => enter(destination, count, destination + count),
        // leave the innermost loop, and any if or call inside it
        OpCode::Break | OpCode::BreakC if taken => {
            let mut next = pc + 1;
            while let Some(block) = blocks.pop() {
                if block.is_loop {
                    next = block.return_address;
                    break;
                }
            }
            next
        }
        _ => pc + 1,
    }
}

/// the shader's multiply, zero rather than NaN for zero times infinity.
/// a NaN is rare, so the lanes are only looked at when there is one.
#[inline(always)]
fn multiply(a: Lanes, b: Lanes) -> Lanes {
    let product = a * b;
    let nan = product.is_nan();
    if !nan.any() {
        return product;
    }
    let fixed = nan & !a.is_nan() & !b.is_nan();
    fixed.select(f32x8::ZERO, product)
}

/// every lane of a row through f, the interpreter's arithmetic one value
/// at a time, for the operations too rare to be worth doing wide.
#[inline(always)]
fn each(row: Lanes, f: impl Fn(f32) -> f32) -> Lanes {
    f32x8::from(row.to_array().map(f))
}

/// a float uniform, none past the last one.
#[inline(always)]
fn uniform(unit: &ShaderUnit, index: i32) -> Vec4 {
    if (0..FLOAT_UNIFORMS as i32).contains(&index) {
        unit.float_uniforms[index as usize]
    } else {
        super::ZERO
    }
}

/// a source as an instruction reads it, its rows in swizzled order, read
/// where they are, and whether they are negated.
struct Source<'a> {
    rows: [&'a Lanes; 4],
    negate: bool,
}

impl Source<'_> {
    #[inline(always)]
    fn row(&self, component: usize) -> Lanes {
        let row = *self.rows[component];
        if self.negate { -row } else { row }
    }
}

/// a source's value in every lane, swizzled and negated as its descriptor
/// says. a uniform is the same in every lane unless an address register
/// offsets it, which puts it together in scratch.
#[inline(always)]
fn operand<'a>(unit: &ShaderUnit, uniforms: &'a Uniforms, batch: &'a Batch, operand: &Operand, scratch: &'a mut Wide) -> Source<'a> {
    let register = operand.register;
    let value: &'a Wide = match register {
        0x00..=0x0F => &batch.input[register as usize],
        0x10..=0x1F => &batch.temp[(register - 0x10) as usize],
        _ => {
            let base = register as i32 - 0x20;
            match operand.index {
                0 => uniforms.at(base),
                3 => uniforms.at(base.wrapping_add(batch.loop_counter)),
                index => {
                    let offsets = &batch.address[index as usize - 1];
                    let picked: [Vec4; LANES] = std::array::from_fn(|lane| uniform(unit, base.wrapping_add(offsets[lane])));
                    *scratch = std::array::from_fn(|component| f32x8::from(picked.map(|value| value[component])));
                    scratch
                }
            }
        }
    };
    Source { rows: operand.swizzle.map(|component| &value[component as usize]), negate: operand.negate }
}

#[inline(always)]
fn write(batch: &mut Batch, op: &Op, value: &Wide) {
    let register = op.destination;
    let slot = match register {
        0x00..=0x0F => &mut batch.output[register as usize],
        0x10..=0x1F => &mut batch.temp[(register - 0x10) as usize],
        _ => return,
    };
    // the mask's most significant bit selects x
    for (component, row) in slot.iter_mut().enumerate() {
        if op.mask & (0b1000 >> component) != 0 {
            *row = value[component];
        }
    }
}

/// what a sum of floats starts from, so that dot products add up exactly
/// the way the interpreter's do.
#[inline(always)]
fn sum_start() -> f32 {
    std::iter::empty::<f32>().sum()
}

/// a dot product over the first count components, in every lane.
#[inline(always)]
fn dot(a: &Source, b: &Source, count: usize) -> Lanes {
    let mut sum = f32x8::splat(sum_start());
    for component in 0..count {
        sum += a.row(component) * b.row(component);
    }
    // a lane without NaN had none in its products, where the shader's
    // multiply changes nothing, so the sum is looked at once
    if !sum.is_nan().any() {
        return sum;
    }
    let mut sum = f32x8::splat(sum_start());
    for component in 0..count {
        sum += multiply(a.row(component), b.row(component));
    }
    sum
}

/// true or false in every lane, as one and zero.
#[inline(always)]
fn select(mask: Lanes) -> Lanes {
    mask.select(f32x8::ONE, f32x8::ZERO)
}

#[inline(always)]
fn arithmetic(unit: &ShaderUnit, uniforms: &Uniforms, batch: &mut Batch, op: &Op, scratch: &mut [Wide; 3]) {
    let [sa, sb, _] = scratch;
    let a = operand(unit, uniforms, batch, &op.sources[0], sa);
    let b = operand(unit, uniforms, batch, &op.sources[1], sb);
    let pairs = |f: fn(Lanes, Lanes) -> Lanes| -> Wide { std::array::from_fn(|i| f(a.row(i), b.row(i))) };
    let result: Wide = match op.opcode {
        OpCode::Add => pairs(|x, y| x + y),
        OpCode::Mul => pairs(multiply),
        // x when it is greater, else y, which is how NaN behaves on hardware
        OpCode::Max => pairs(|x, y| x.simd_gt(y).select(x, y)),
        OpCode::Min => pairs(|x, y| x.simd_lt(y).select(x, y)),
        OpCode::Dp3 => [dot(&a, &b, 3); 4],
        OpCode::Dp4 => [dot(&a, &b, 4); 4],
        OpCode::Dph | OpCode::DphI => [dot(&a, &b, 3) + b.row(3); 4],
        OpCode::Mov => std::array::from_fn(|i| a.row(i)),
        OpCode::Flr => std::array::from_fn(|i| each(a.row(i), f32::floor)),
        OpCode::Rcp => [f32x8::ONE / a.row(0); 4],
        OpCode::Rsq => [f32x8::ONE / a.row(0).sqrt(); 4],
        OpCode::Ex2 => [each(a.row(0), f32::exp2); 4],
        OpCode::Lg2 => [each(a.row(0), f32::log2); 4],
        OpCode::Sge | OpCode::SgeI => pairs(|x, y| select(x.simd_ge(y))),
        OpCode::Slt | OpCode::SltI => pairs(|x, y| select(x.simd_lt(y))),
        OpCode::Dst | OpCode::DstI => [f32x8::ONE, multiply(a.row(1), b.row(1)), a.row(2), b.row(3)],
        OpCode::LitP => {
            let (x, y, w) = (a.row(0), a.row(1), a.row(3));
            let conditions = [x.to_array().map(|x| x >= 0.0), w.to_array().map(|w| w >= 0.0)];
            let result = [each(x, |x| x.max(0.0)), each(y, |y| y.clamp(-127.9961, 127.9961)), f32x8::ZERO, each(w, |w| w.max(0.0))];
            batch.condition = conditions;
            result
        }
        OpCode::Mova => {
            let (x, y) = (a.row(0).to_array(), a.row(1).to_array());
            if op.mask & 0b1000 != 0 {
                batch.address[0] = x.map(|x| x as i32);
            }
            if op.mask & 0b0100 != 0 {
                batch.address[1] = y.map(|y| y as i32);
            }
            return;
        }
        OpCode::Cmp => {
            let modes = op.instruction.compare_modes();
            let conditions: [[bool; LANES]; 2] = std::array::from_fn(|component| {
                let (x, y) = (a.row(component).to_array(), b.row(component).to_array());
                std::array::from_fn(|lane| super::compare(modes[component], x[lane], y[lane]))
            });
            batch.condition = conditions;
            return;
        }
        other => {
            log::debug!("unimplemented shader opcode {other:?}");
            return;
        }
    };
    write(batch, op, &result);
}

/// the lanes a flow instruction goes its way in, one bit each.
fn condition(unit: &ShaderUnit, batch: &Batch, op: &Op) -> u8 {
    let instruction = op.instruction;
    let all = |taken: bool| if taken { ALL } else { 0 };
    match op.opcode {
        OpCode::Call | OpCode::Break => ALL,
        OpCode::CallU | OpCode::IfU => all(unit.bool_uniforms & (1 << instruction.bool_index()) != 0),
        // jmpu can test for either value, the low bit of its count inverts it
        OpCode::JmpU => {
            let set = unit.bool_uniforms & (1 << instruction.bool_index()) != 0;
            all(set == (instruction.flow_target().1 & 1 == 0))
        }
        _ => {
            let reference = instruction.condition_reference();
            let mut lanes = 0;
            for lane in 0..LANES {
                let x = batch.condition[0][lane] == reference[0];
                let y = batch.condition[1][lane] == reference[1];
                let taken = match instruction.condition_op() {
                    0 => x || y,
                    1 => x && y,
                    2 => x,
                    _ => y,
                };
                lanes |= (taken as u8) << lane;
            }
            lanes
        }
    }
}
