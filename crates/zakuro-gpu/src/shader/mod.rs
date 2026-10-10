//! the PICA200's shader units, which run both vertex and geometry shaders.

pub mod isa;
mod batch;
#[cfg(feature = "vulkan")]
pub(crate) mod glsl;
#[cfg(feature = "jit")]
mod jit;

use std::collections::HashMap;
use std::sync::Arc;

use isa::{Instruction, OpCode, OperandDescriptor};

pub const PROGRAM_SIZE: usize = 4096;
pub const DESCRIPTOR_SIZE: usize = 128;

// fingerprint hashes both four words at a time
const _: () = assert!(PROGRAM_SIZE.is_multiple_of(4) && DESCRIPTOR_SIZE.is_multiple_of(4));
pub const FLOAT_UNIFORMS: usize = 96;
pub const INPUT_REGISTERS: usize = 16;
pub const OUTPUT_REGISTERS: usize = 16;
pub const TEMP_REGISTERS: usize = 16;

/// a four-component vector, which is the only data type the shader has.
pub type Vec4 = [f32; 4];

/// how many decoded programs a unit keeps before starting over.
const DECODED_KEPT: usize = 64;

pub const ZERO: Vec4 = [0.0; 4];

/// what a temporary register holds before the program writes it, which
/// titles count on. Project Mirai DX takes r15.w for a position's w before
/// ever writing it, and a w of 0 leaves out the line its notes run along.
pub const TEMP_START: Vec4 = [0.0, 0.0, 0.0, 1.0];

/// shader state that persists between vertices.
#[derive(Clone)]
pub struct ShaderUnit {
    pub program: Box<[u32; PROGRAM_SIZE]>,
    pub descriptors: Box<[u32; DESCRIPTOR_SIZE]>,
    pub float_uniforms: Box<[Vec4; FLOAT_UNIFORMS]>,
    pub int_uniforms: [[u8; 4]; 4],
    pub bool_uniforms: u16,
    pub entry_point: u32,

    /// upload cursors, driven by the command list.
    pub program_write_offset: usize,
    pub descriptor_write_offset: usize,
    float_uniform_index: usize,
    float_uniform_component: usize,
    float_uniform_wide: bool,
    float_uniform_staging: [u32; 4],
    /// the program decoded, none since it or its descriptors changed.
    decoded: Option<Arc<Program>>,
    /// programs decoded before, by fingerprint. titles upload the program
    /// again each time they switch shaders, often the same few.
    decoded_before: HashMap<u64, Arc<Program>>,
}

impl Default for ShaderUnit {
    fn default() -> Self {
        Self::new()
    }
}

impl ShaderUnit {
    pub fn new() -> ShaderUnit {
        ShaderUnit {
            program: Box::new([0; PROGRAM_SIZE]),
            descriptors: Box::new([0; DESCRIPTOR_SIZE]),
            float_uniforms: Box::new([ZERO; FLOAT_UNIFORMS]),
            int_uniforms: [[0; 4]; 4],
            bool_uniforms: 0,
            entry_point: 0,
            program_write_offset: 0,
            descriptor_write_offset: 0,
            float_uniform_index: 0,
            float_uniform_component: 0,
            float_uniform_wide: false,
            float_uniform_staging: [0; 4],
            decoded: None,
            decoded_before: HashMap::new(),
        }
    }

    pub fn upload_program(&mut self, word: u32) {
        if self.program_write_offset < PROGRAM_SIZE {
            if self.program[self.program_write_offset] != word {
                self.program[self.program_write_offset] = word;
                self.decoded = None;
            }
            self.program_write_offset += 1;
        }
    }

    pub fn upload_descriptor(&mut self, word: u32) {
        if self.descriptor_write_offset < DESCRIPTOR_SIZE {
            if self.descriptors[self.descriptor_write_offset] != word {
                self.descriptors[self.descriptor_write_offset] = word;
                self.decoded = None;
            }
            self.descriptor_write_offset += 1;
        }
    }

    /// decodes the program, if it changed, before running it on many
    /// vertices.
    pub fn prepare(&mut self) {
        if self.decoded.is_some() {
            return;
        }
        let fingerprint = self.fingerprint();
        let program = match self.decoded_before.get(&fingerprint) {
            Some(program) => program.clone(),
            None => {
                let program = Arc::new(Program::decode(self));
                if self.decoded_before.len() >= DECODED_KEPT {
                    self.decoded_before.clear();
                }
                self.decoded_before.insert(fingerprint, program.clone());
                program
            }
        };
        self.decoded = Some(program);
    }

    /// the input registers the program reads, one bit each, all of them
    /// until it is prepared.
    pub fn inputs_read(&self) -> u16 {
        self.decoded.as_ref().map_or(u16::MAX, |program| program.inputs)
    }

    /// a hash of the program and its descriptors, the same for the same
    /// words.
    pub fn fingerprint(&self) -> u64 {
        if let Some(program) = &self.decoded {
            return program.fingerprint;
        }
        // four lanes the CPU works on side by side, titles switch programs
        // often enough for one chain through every word to show
        let mix = |hash: u64, word: u64| (hash.rotate_left(5) ^ word).wrapping_mul(0x517C_C1B7_2722_0A95);
        let mut lanes = [0u64; 4];
        let words = self.program.as_chunks::<4>().0.iter().chain(self.descriptors.as_chunks::<4>().0);
        for chunk in words {
            for (lane, &word) in lanes.iter_mut().zip(chunk) {
                *lane = mix(*lane, word as u64);
            }
        }
        lanes.into_iter().fold(0, mix)
    }

    /// starts a float uniform upload at the register the raw value names.
    pub fn set_float_uniform_index(&mut self, raw: u32) {
        self.float_uniform_index = (raw & 0x7F) as usize;
        self.float_uniform_component = 0;
        self.float_uniform_wide = raw & 0x8000_0000 != 0;
    }

    /// float uniforms arrive component by component, three words carry four
    /// packed 24-bit floats, or four words carry plain singles.
    #[inline]
    pub fn upload_float_uniform(&mut self, word: u32) {
        let needed = if self.float_uniform_wide { 4 } else { 3 };
        if self.float_uniform_component < 4 {
            self.float_uniform_staging[self.float_uniform_component] = word;
        }
        self.float_uniform_component += 1;
        if self.float_uniform_component < needed {
            return;
        }
        self.float_uniform_component = 0;
        self.set_float_uniform(self.float_uniform_staging, self.float_uniform_wide);
    }

    /// float uniform words in a row, as upload_float_uniform takes them,
    /// a whole vector at a time once a vector starts. a draw's matrices
    /// come this way, most of the words in a command list.
    pub fn upload_float_uniforms(&mut self, mut words: &[u32]) {
        // the rest of a vector the words before left unfinished
        while self.float_uniform_component != 0 {
            let [word, rest @ ..] = words else {
                return;
            };
            self.upload_float_uniform(*word);
            words = rest;
        }
        let rest = if self.float_uniform_wide {
            self.upload_vectors::<4>(words)
        } else {
            self.upload_vectors::<3>(words)
        };
        for &word in rest {
            self.upload_float_uniform(word);
        }
    }

    /// uploads the whole vectors of n words at the start of words, and
    /// gives back the words after them.
    #[inline]
    fn upload_vectors<'a, const N: usize>(&mut self, words: &'a [u32]) -> &'a [u32] {
        let (vectors, rest) = words.as_chunks::<N>();
        // staged as a word at a time stages them, the last vector's words
        // stay there
        let mut staged = self.float_uniform_staging;
        for vector in vectors {
            staged[..N].copy_from_slice(vector);
            self.set_float_uniform(staged, N == 4);
        }
        self.float_uniform_staging = staged;
        rest
    }

    /// decodes a vector's staged words into the uniform the index names,
    /// and moves the index on.
    #[inline(always)]
    fn set_float_uniform(&mut self, staged: [u32; 4], wide: bool) {
        let value = if wide {
            [
                f32::from_bits(staged[3]),
                f32::from_bits(staged[2]),
                f32::from_bits(staged[1]),
                f32::from_bits(staged[0]),
            ]
        } else {
            // the four components are packed most significant first across
            // three words.
            [
                isa::decode_float24(staged[2] & 0x00FF_FFFF),
                isa::decode_float24(((staged[1] & 0x0000_FFFF) << 8) | (staged[2] >> 24)),
                isa::decode_float24(((staged[0] & 0x0000_00FF) << 16) | (staged[1] >> 16)),
                isa::decode_float24(staged[0] >> 8),
            ]
        };

        if value.iter().any(|c| c.is_nan()) {
            log::trace!(
                target: "zakuro_gpu::shader::nan",
                "uniform c{} set to NaN {value:?} from {:08X?} ({} bit)",
                self.float_uniform_index,
                staged,
                if wide { 32 } else { 24 },
            );
        }
        if self.float_uniform_index < FLOAT_UNIFORMS {
            self.float_uniforms[self.float_uniform_index] = value;
        }
        self.float_uniform_index += 1;
    }

    /// the upload cursors, the staged words and the programs decoded, to
    /// tell two units apart in tests.
    #[cfg(test)]
    pub(crate) fn cursors(&self) -> (usize, usize, bool, [u32; 4], Option<u64>, Vec<u64>) {
        let mut decoded_before: Vec<u64> = self.decoded_before.keys().copied().collect();
        decoded_before.sort_unstable();
        (
            self.float_uniform_index,
            self.float_uniform_component,
            self.float_uniform_wide,
            self.float_uniform_staging,
            self.decoded.as_ref().map(|program| program.fingerprint),
            decoded_before,
        )
    }
}

/// per-vertex shader state.
pub struct ShaderState {
    pub input: [Vec4; INPUT_REGISTERS],
    pub output: [Vec4; OUTPUT_REGISTERS],
    temp: [Vec4; TEMP_REGISTERS],
    address: [i32; 3],
    condition: [bool; 2],
    /// open calls, if bodies and loops, innermost last.
    blocks: Vec<Block>,
}

/// a range of instructions that, once finished, sends execution elsewhere.
#[derive(Debug, Clone, Copy)]
struct Block {
    /// the address just past the block's last instruction.
    end: u32,
    /// where execution goes once the block is done.
    return_address: u32,
    /// further iterations to run, zero for anything but a loop.
    repeat: u32,
    /// added to the loop counter each time the block completes.
    increment: i32,
    /// where each further iteration starts.
    start: u32,
    is_loop: bool,
}

impl Default for ShaderState {
    fn default() -> Self {
        Self::new()
    }
}

impl ShaderState {
    pub fn new() -> ShaderState {
        ShaderState {
            input: [ZERO; INPUT_REGISTERS],
            output: [ZERO; OUTPUT_REGISTERS],
            temp: [TEMP_START; TEMP_REGISTERS],
            address: [0; 3],
            condition: [false; 2],
            blocks: Vec::with_capacity(16),
        }
    }

    fn reset(&mut self) {
        self.output = [ZERO; OUTPUT_REGISTERS];
        self.temp = [TEMP_START; TEMP_REGISTERS];
        self.address = [0; 3];
        self.condition = [false; 2];
        self.blocks.clear();
    }

    /// enters a block starting at start and running count instructions.
    fn enter(&mut self, start: u32, count: u32, return_address: u32) -> u32 {
        self.blocks.push(Block {
            end: start + count,
            return_address,
            repeat: 0,
            increment: 0,
            start,
            is_loop: false,
        });
        start
    }
}

/// what a geometry shader's emit produces, up to three vertices at a time,
/// sent on as a triangle when setemit says the next one completes it.
#[derive(Default)]
pub struct Emitter {
    slots: [[Vec4; OUTPUT_REGISTERS]; 3],
    slot: usize,
    completes_primitive: bool,
    /// finished triangles, as each vertex's output registers.
    pub triangles: Vec<[[Vec4; OUTPUT_REGISTERS]; 3]>,
}

/// one source of an instruction, with its descriptor applied ahead of time.
#[derive(Debug, Clone, Copy, Default)]
struct Operand {
    register: u32,
    /// the component each one takes.
    swizzle: [u8; 4],
    negate: bool,
    /// the address register that offsets it, zero for none.
    index: u32,
}

/// an instruction with everything it reads from its encoding and its
/// descriptor worked out, which only the program changing changes.
#[derive(Debug, Clone, Copy)]
struct Op {
    instruction: Instruction,
    opcode: OpCode,
    destination: u32,
    /// the components written, the most significant bit being x.
    mask: u32,
    sources: [Operand; 3],
}

impl Op {
    fn decode(instruction: Instruction, descriptors: &[u32; DESCRIPTOR_SIZE]) -> Op {
        let opcode = instruction.opcode();
        let operand = |descriptor: OperandDescriptor, source: u32, register: u32, index: u32| {
            let (swizzle, negate) = descriptor.source(source);
            Operand { register, swizzle, negate, index }
        };
        match opcode {
            OpCode::Mad | OpCode::MadI => {
                let descriptor =
                    OperandDescriptor(descriptors[instruction.mad_descriptor_index() as usize % DESCRIPTOR_SIZE]);
                let (src1, src2, src3) = instruction.mad_sources();
                // the address register applies to whichever source is the
                // wide one
                let index = instruction.mad_address_register_index();
                let (index2, index3) = if opcode == OpCode::MadI { (0, index) } else { (index, 0) };
                Op {
                    instruction,
                    opcode,
                    destination: instruction.mad_destination(),
                    mask: descriptor.destination_mask(),
                    sources: [
                        operand(descriptor, 1, src1, 0),
                        operand(descriptor, 2, src2, index2),
                        operand(descriptor, 3, src3, index3),
                    ],
                }
            }
            _ => {
                let descriptor =
                    OperandDescriptor(descriptors[instruction.descriptor_index() as usize % DESCRIPTOR_SIZE]);
                let (src1, src2) = instruction.sources();
                // only the wide operand can be a uniform, and only a uniform
                // read can be indexed by an address register
                let index = instruction.address_register_index();
                let (index1, index2) = if opcode.is_inverted() { (0, index) } else { (index, 0) };
                Op {
                    instruction,
                    opcode,
                    destination: instruction.destination(),
                    mask: descriptor.destination_mask(),
                    sources: [
                        operand(descriptor, 1, src1, index1),
                        operand(descriptor, 2, src2, index2),
                        Operand::default(),
                    ],
                }
            }
        }
    }
}

/// a shader unit's program, decoded once to be run on every vertex.
pub struct Program {
    ops: Box<[Op]>,
    /// the input registers anything reads and the output registers
    /// anything writes, one bit each.
    inputs: u16,
    outputs: u16,
    /// a hash of the words it came from, which the GPU keeps programs by.
    fingerprint: u64,
}

impl Program {
    fn decode(unit: &ShaderUnit) -> Program {
        let ops: Box<[Op]> = unit.program.iter().map(|&word| Op::decode(Instruction(word), &unit.descriptors)).collect();
        let (mut inputs, mut outputs) = (0u16, 0u16);
        for op in ops.iter() {
            let sources = match op.opcode {
                OpCode::Mad | OpCode::MadI => 3,
                OpCode::Mova | OpCode::Cmp => 2,
                opcode if opcode.writes() => 2,
                _ => continue,
            };
            for source in &op.sources[..sources] {
                if source.register < 0x10 {
                    inputs |= 1 << source.register;
                }
            }
            if op.opcode.writes() && op.destination < 0x10 {
                outputs |= 1 << op.destination;
            }
        }
        Program { ops, inputs, outputs, fingerprint: unit.fingerprint() }
    }
}

/// runs the vertex shader over many vertices, a batch of them at a time,
/// giving what running it on each would give.
pub fn run_vertices(unit: &ShaderUnit, inputs: &[[Vec4; INPUT_REGISTERS]]) -> Vec<[Vec4; OUTPUT_REGISTERS]> {
    let decoded;
    let program = match &unit.decoded {
        Some(program) => program.as_ref(),
        None => {
            decoded = Program::decode(unit);
            &decoded
        }
    };
    // tracing NaNs is the interpreter's
    let trace_nan = log::log_enabled!(target: "zakuro_gpu::shader::nan", log::Level::Trace);
    let mut outputs = vec![[ZERO; OUTPUT_REGISTERS]; inputs.len()];
    let (mut blocks, mut forks) = (Vec::with_capacity(16), Vec::new());
    // the program compiled, the batches it gives back shaded the way below
    #[cfg(feature = "jit")]
    if let Some(compiled) = unit.decoded.as_ref().filter(|_| !trace_nan).and_then(|program| jit::compiled(unit, program)) {
        let mut uniforms = None;
        jit::run_vertices(&compiled, unit, program, inputs, &mut outputs, |inputs, outputs| {
            let uniforms = uniforms.get_or_insert_with(|| batch::Uniforms::new(unit));
            batch::run(unit, uniforms, program, inputs, outputs, &mut blocks, &mut forks);
        });
        return outputs;
    }
    let mut state = ShaderState::new();
    let uniforms = batch::Uniforms::new(unit);
    for (inputs, outputs) in inputs.chunks(batch::LANES).zip(outputs.chunks_mut(batch::LANES)) {
        if !trace_nan {
            batch::run(unit, &uniforms, program, inputs, outputs, &mut blocks, &mut forks);
            continue;
        }
        for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
            state.input = *input;
            execute(unit, program, &mut state, None);
            *output = state.output;
        }
    }
    outputs
}

/// runs the shader over one vertex.
pub fn run(unit: &ShaderUnit, state: &mut ShaderState) {
    match &unit.decoded {
        Some(program) => execute(unit, program, state, None),
        None => execute(unit, &Program::decode(unit), state, None),
    }
}

/// runs a geometry shader over many primitives' inputs, a batch of them
/// at a time, giving each one's triangles in order, what running it on
/// each would give.
pub fn run_geometry_many(unit: &ShaderUnit, inputs: &[[Vec4; INPUT_REGISTERS]]) -> Vec<Vec<batch::Triangle>> {
    let decoded;
    let program = match &unit.decoded {
        Some(program) => program.as_ref(),
        None => {
            decoded = Program::decode(unit);
            &decoded
        }
    };
    let trace_nan = log::log_enabled!(target: "zakuro_gpu::shader::nan", log::Level::Trace);
    let mut triangles = Vec::with_capacity(inputs.len());
    let (mut blocks, mut forks) = (Vec::with_capacity(16), Vec::new());
    #[cfg(feature = "jit")]
    if let Some(compiled) = unit.decoded.as_ref().filter(|_| !trace_nan).and_then(|program| jit::compiled(unit, program)) {
        let mut uniforms = None;
        jit::run_geometry(&compiled, unit, program, inputs, &mut triangles, |inputs, triangles| {
            let uniforms = uniforms.get_or_insert_with(|| batch::Uniforms::new(unit));
            let mut emitters = batch::Emitters::new();
            batch::run_geometry(unit, uniforms, program, inputs, &mut emitters, &mut blocks, &mut forks);
            triangles.extend(emitters.triangles.into_iter().take(inputs.len()));
        });
        return triangles;
    }
    let uniforms = batch::Uniforms::new(unit);
    for inputs in inputs.chunks(batch::LANES) {
        if trace_nan {
            for input in inputs {
                let mut state = ShaderState::new();
                state.input = *input;
                let mut emitter = Emitter::default();
                execute(unit, program, &mut state, Some(&mut emitter));
                triangles.push(emitter.triangles);
            }
            continue;
        }
        let mut emitters = batch::Emitters::new();
        batch::run_geometry(unit, &uniforms, program, inputs, &mut emitters, &mut blocks, &mut forks);
        triangles.extend(emitters.triangles.into_iter().take(inputs.len()));
    }
    triangles
}

/// runs a geometry shader over one primitive's worth of input, collecting
/// what it emits.
pub fn run_geometry(unit: &ShaderUnit, state: &mut ShaderState, emitter: &mut Emitter) {
    match &unit.decoded {
        Some(program) => execute(unit, program, state, Some(emitter)),
        None => execute(unit, &Program::decode(unit), state, Some(emitter)),
    }
}

fn execute(unit: &ShaderUnit, program: &Program, state: &mut ShaderState, mut emitter: Option<&mut Emitter>) {
    state.reset();
    // debugging aid, RUST_LOG=zakuro_gpu::shader::nan=trace reports the
    // first instruction of each run whose result is NaN, with its operands.
    let trace_nan = log::log_enabled!(target: "zakuro_gpu::shader::nan", log::Level::Trace);
    let mut nan_reported = false;
    let mut pc = unit.entry_point;
    // a program that never ends is a decoding bug, cap it rather than hang.
    let mut budget = 0u32;

    loop {
        // finishing a block, go round a loop again, or return to whatever
        // follows it. Several blocks can end at the same address.
        while let Some(top) = state.blocks.last_mut() {
            if pc != top.end {
                break;
            }
            state.address[2] = state.address[2].wrapping_add(top.increment);
            if top.repeat == 0 {
                pc = top.return_address;
                state.blocks.pop();
            } else {
                top.repeat -= 1;
                pc = top.start;
            }
        }

        if pc as usize >= PROGRAM_SIZE || budget >= 0x10000 {
            break;
        }
        budget += 1;

        let op = &program.ops[pc as usize];
        let instruction = op.instruction;
        let mut next = pc + 1;

        match op.opcode {
            OpCode::End => break,
            OpCode::Nop => {}
            OpCode::SetEmit => {
                if let Some(emitter) = emitter.as_deref_mut() {
                    emitter.slot = instruction.emit_vertex().min(2);
                    emitter.completes_primitive = instruction.emit_primitive();
                }
            }
            OpCode::Emit => {
                if let Some(emitter) = emitter.as_deref_mut() {
                    emitter.slots[emitter.slot] = state.output;
                    if emitter.completes_primitive {
                        emitter.triangles.push(emitter.slots);
                    }
                }
            }

            OpCode::Mad | OpCode::MadI => {
                let operands = (trace_nan && !nan_reported).then(|| operand_values(unit, state, instruction));
                multiply_add(unit, state, op);
                if let Some(operands) = operands {
                    nan_reported = report_nan(state, pc, instruction, &operands);
                }
            }

            OpCode::Call | OpCode::CallU | OpCode::CallC => {
                let (destination, count) = instruction.flow_target();
                if flow_condition(unit, state, instruction) {
                    next = state.enter(destination, count, pc + 1);
                }
            }
            OpCode::JmpC | OpCode::JmpU => {
                if flow_condition(unit, state, instruction) {
                    next = instruction.flow_target().0;
                }
            }
            OpCode::IfC | OpCode::IfU => {
                let (destination, count) = instruction.flow_target();
                next = if flow_condition(unit, state, instruction) {
                    // run the body, then skip the else block.
                    state.enter(pc + 1, destination - (pc + 1).min(destination), destination + count)
                } else {
                    // run the else block, then carry on after it.
                    state.enter(destination, count, destination + count)
                };
            }
            OpCode::Loop => {
                let integers = unit.int_uniforms[instruction.integer_index() as usize & 3];
                let (last, _) = instruction.flow_target();
                state.address[2] = integers[1] as i32;
                state.blocks.push(Block {
                    end: last + 1,
                    return_address: last + 1,
                    repeat: integers[0] as u32,
                    increment: integers[2] as i8 as i32,
                    start: pc + 1,
                    is_loop: true,
                });
            }
            OpCode::Break | OpCode::BreakC => {
                if instruction.opcode() == OpCode::Break
                    || flow_condition(unit, state, instruction)
                {
                    // leave the innermost loop, and any if or call inside it.
                    while let Some(block) = state.blocks.pop() {
                        if block.is_loop {
                            next = block.return_address;
                            break;
                        }
                    }
                }
            }

            _ => {
                let operands = (trace_nan && !nan_reported).then(|| operand_values(unit, state, instruction));
                arithmetic(unit, state, op);
                if let Some(operands) = operands {
                    nan_reported = report_nan(state, pc, instruction, &operands);
                }
            }
        }

        pc = next;
    }
}

/// the registers an instruction reads and their values, for tracing.
fn operand_values(unit: &ShaderUnit, state: &ShaderState, instruction: Instruction) -> Vec<(u32, Vec4)> {
    let registers: Vec<u32> = match instruction.opcode() {
        OpCode::Mad | OpCode::MadI => {
            let (a, b, c) = instruction.mad_sources();
            vec![a, b, c]
        }
        _ => {
            let (a, b) = instruction.sources();
            vec![a, b]
        }
    };
    registers
        .into_iter()
        .map(|register| (register, source_value(unit, state, register)))
        .collect()
}

/// logs the instruction if it just produced the run's first NaN.
fn report_nan(state: &ShaderState, pc: u32, instruction: Instruction, operands: &[(u32, Vec4)]) -> bool {
    let nan = |v: &Vec4| v.iter().any(|c| c.is_nan());
    if !state.temp.iter().chain(state.output.iter()).any(nan) {
        return false;
    }
    log::trace!(
        target: "zakuro_gpu::shader::nan",
        "first NaN at pc {pc}: 0x{:08X} {:?}, operands {operands:?}, address registers {:?}, \
         temporaries {:?}",
        instruction.0,
        instruction.opcode(),
        state.address,
        state.temp,
    );
    true
}

/// applies an address register to a uniform operand.
fn indexed(state: &ShaderState, register: u32, index: u32) -> u32 {
    if register < 0x20 || index == 0 {
        return register;
    }
    let offset = state.address[(index - 1) as usize % 3];
    let uniform = (register as i32 - 0x20).wrapping_add(offset);
    // out of range reads fall on no uniform at all.
    if (0..FLOAT_UNIFORMS as i32).contains(&uniform) {
        uniform as u32 + 0x20
    } else {
        u32::MAX
    }
}

fn source_value(unit: &ShaderUnit, state: &ShaderState, register: u32) -> Vec4 {
    match register {
        0x00..=0x0F => state.input[register as usize],
        0x10..=0x1F => state.temp[(register - 0x10) as usize],
        _ => unit
            .float_uniforms
            .get(register.wrapping_sub(0x20) as usize)
            .copied()
            .unwrap_or(ZERO),
    }
}

/// a source's value, swizzled and negated as its descriptor says.
#[inline]
fn operand(unit: &ShaderUnit, state: &ShaderState, operand: &Operand) -> Vec4 {
    let register = indexed(state, operand.register, operand.index);
    let value = source_value(unit, state, register);
    let [x, y, z, w] = operand.swizzle.map(|component| value[component as usize]);
    if operand.negate {
        [-x, -y, -z, -w]
    } else {
        [x, y, z, w]
    }
}

fn arithmetic(unit: &ShaderUnit, state: &mut ShaderState, op: &Op) {
    let src1 = operand(unit, state, &op.sources[0]);
    let src2 = operand(unit, state, &op.sources[1]);
    let instruction = op.instruction;

    let result: Vec4 = match op.opcode {
        OpCode::Add => component_wise(src1, src2, |a, b| a + b),
        OpCode::Mul => component_wise(src1, src2, multiply),
        // written so NaN behaves as it does on hardware, max(0, NaN) is NaN
        // but max(NaN, 0) is 0.
        OpCode::Max => component_wise(src1, src2, |a, b| if a > b { a } else { b }),
        OpCode::Min => component_wise(src1, src2, |a, b| if a < b { a } else { b }),
        OpCode::Dp3 => [dot(&src1[..3], &src2[..3]); 4],
        OpCode::Dp4 => [dot(&src1, &src2); 4],
        // dph takes the fourth component of the first operand as one, which
        // is how a position is transformed without padding it first.
        OpCode::Dph | OpCode::DphI => [dot(&src1[..3], &src2[..3]) + src2[3]; 4],
        OpCode::Mov => src1,
        OpCode::Flr => [
            src1[0].floor(),
            src1[1].floor(),
            src1[2].floor(),
            src1[3].floor(),
        ],
        OpCode::Rcp => [1.0 / src1[0]; 4],
        OpCode::Rsq => [1.0 / src1[0].sqrt(); 4],
        OpCode::Ex2 => [src1[0].exp2(); 4],
        OpCode::Lg2 => [src1[0].log2(); 4],
        OpCode::Sge | OpCode::SgeI => {
            component_wise(src1, src2, |a, b| (a >= b) as u32 as f32)
        }
        OpCode::Slt | OpCode::SltI => {
            component_wise(src1, src2, |a, b| (a < b) as u32 as f32)
        }
        // dst builds the classic attenuation vector (1, d, d², 1/d).
        OpCode::Dst | OpCode::DstI => {
            [1.0, multiply(src1[1], src2[1]), src1[2], src2[3]]
        }
        // prepares a lighting computation, and records which of the two
        // terms that matter were non-negative.
        OpCode::LitP => {
            state.condition = [src1[0] >= 0.0, src1[3] >= 0.0];
            [
                src1[0].max(0.0),
                src1[1].clamp(-127.9961, 127.9961),
                0.0,
                src1[3].max(0.0),
            ]
        }
        OpCode::Mova => {
            let mask = op.mask;
            if mask & 0b1000 != 0 {
                state.address[0] = src1[0] as i32;
            }
            if mask & 0b0100 != 0 {
                state.address[1] = src1[1] as i32;
            }
            return;
        }
        OpCode::Cmp => {
            let modes = instruction.compare_modes();
            state.condition[0] = compare(modes[0], src1[0], src2[0]);
            state.condition[1] = compare(modes[1], src1[1], src2[1]);
            return;
        }
        other => {
            log::debug!("unimplemented shader opcode {other:?}");
            return;
        }
    };

    write_masked(state, op.destination, op.mask, result);
}

fn multiply_add(unit: &ShaderUnit, state: &mut ShaderState, op: &Op) {
    let src1 = operand(unit, state, &op.sources[0]);
    let src2 = operand(unit, state, &op.sources[1]);
    let src3 = operand(unit, state, &op.sources[2]);

    let result = [
        multiply(src1[0], src2[0]) + src3[0],
        multiply(src1[1], src2[1]) + src3[1],
        multiply(src1[2], src2[2]) + src3[2],
        multiply(src1[3], src2[3]) + src3[3],
    ];
    write_masked(state, op.destination, op.mask, result);
}

/// the shader's multiply, which gives zero rather than NaN for zero times
/// infinity.
#[inline(always)]
fn multiply(a: f32, b: f32) -> f32 {
    let product = a * b;
    // without short circuits, so that it stays a select many lanes do at once
    if product.is_nan() & !a.is_nan() & !b.is_nan() {
        0.0
    } else {
        product
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| multiply(x, y)).sum()
}

#[inline]
fn component_wise(a: Vec4, b: Vec4, f: impl Fn(f32, f32) -> f32) -> Vec4 {
    [f(a[0], b[0]), f(a[1], b[1]), f(a[2], b[2]), f(a[3], b[3])]
}

fn write_masked(state: &mut ShaderState, register: u32, mask: u32, value: Vec4) {
    let slot = match register {
        0x00..=0x0F => &mut state.output[register as usize],
        0x10..=0x1F => &mut state.temp[(register - 0x10) as usize],
        _ => return,
    };
    // the mask's most significant bit selects x.
    for component in 0..4 {
        if mask & (0b1000 >> component) != 0 {
            slot[component] = value[component];
        }
    }
}

#[inline(always)]
fn compare(mode: u32, a: f32, b: f32) -> bool {
    match mode {
        0 => a == b,
        1 => a != b,
        2 => a < b,
        3 => a <= b,
        4 => a > b,
        5 => a >= b,
        _ => true,
    }
}

fn flow_condition(unit: &ShaderUnit, state: &ShaderState, instruction: Instruction) -> bool {
    match instruction.opcode() {
        OpCode::Call => true,
        OpCode::CallU | OpCode::IfU => unit.bool_uniforms & (1 << instruction.bool_index()) != 0,
        // jmpu can test for either value, the low bit of its (otherwise
        // unused) count field inverts the condition.
        OpCode::JmpU => {
            let set = unit.bool_uniforms & (1 << instruction.bool_index()) != 0;
            set == (instruction.flow_target().1 & 1 == 0)
        }
        _ => {
            let reference = instruction.condition_reference();
            let x = state.condition[0] == reference[0];
            let y = state.condition[1] == reference[1];
            match instruction.condition_op() {
                0 => x || y,
                1 => x && y,
                2 => x,
                _ => y,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_with(program: &[u32], descriptors: &[u32]) -> ShaderUnit {
        let mut unit = ShaderUnit::new();
        for (i, &word) in program.iter().enumerate() {
            unit.program[i] = word;
        }
        for (i, &word) in descriptors.iter().enumerate() {
            unit.descriptors[i] = word;
        }
        unit
    }

    /// descriptor 0, write every component, identity swizzle on both sources.
    const IDENTITY: u32 = 0xF | (0b00_01_10_11 << 5) | (0b00_01_10_11 << 14);

    /// a little random number generator, the same numbers every run.
    struct Random(u64);

    impl Random {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }

        fn below(&mut self, n: u32) -> u32 {
            self.next() % n
        }

        /// mostly ordinary numbers, now and then a zero, an infinity or a NaN.
        fn float(&mut self) -> f32 {
            match self.below(40) {
                0 => 0.0,
                1 => -0.0,
                2 => f32::INFINITY,
                3 => f32::NEG_INFINITY,
                4 => f32::NAN,
                _ => (self.next() as f32 / u32::MAX as f32 - 0.5) * 16.0,
            }
        }
    }

    /// what shading each vertex on its own gives.
    fn one_by_one(unit: &ShaderUnit, inputs: &[[Vec4; INPUT_REGISTERS]]) -> Vec<[Vec4; OUTPUT_REGISTERS]> {
        inputs
            .iter()
            .map(|&input| {
                let mut state = ShaderState::new();
                state.input = input;
                run(unit, &mut state);
                state.output
            })
            .collect()
    }

    /// the batches alone, without the program compiled.
    fn batched(unit: &ShaderUnit, inputs: &[[Vec4; INPUT_REGISTERS]]) -> Vec<[Vec4; OUTPUT_REGISTERS]> {
        let program = unit.decoded.as_ref().expect("a prepared unit");
        let mut outputs = vec![[ZERO; OUTPUT_REGISTERS]; inputs.len()];
        let (mut blocks, mut forks) = (Vec::new(), Vec::new());
        let uniforms = batch::Uniforms::new(unit);
        for (inputs, outputs) in inputs.chunks(batch::LANES).zip(outputs.chunks_mut(batch::LANES)) {
            batch::run(unit, &uniforms, program, inputs, outputs, &mut blocks, &mut forks);
        }
        outputs
    }

    /// the unit's program compiles, as it has to wherever there is a
    /// compiler.
    fn assert_compiled(unit: &ShaderUnit) {
        #[cfg(feature = "jit")]
        if let Err(error) = jit::compile(unit.decoded.as_ref().expect("a prepared unit"), unit.entry_point) {
            panic!("the program did not compile, {error}");
        }
    }

    /// the same results, a NaN matching any NaN and zeros their sign.
    fn same(a: &[[Vec4; OUTPUT_REGISTERS]], b: &[[Vec4; OUTPUT_REGISTERS]]) -> bool {
        let same = |x: f32, y: f32| (x.is_nan() && y.is_nan()) || x.to_bits() == y.to_bits();
        a.len() == b.len() && a.iter().flatten().flatten().zip(b.iter().flatten().flatten()).all(|(&x, &y)| same(x, y))
    }

    /// the batches give exactly what the interpreter gives, over programs
    /// made of every arithmetic instruction with random registers,
    /// swizzles, masks and address registers.
    #[test]
    fn batches_shade_like_the_interpreter() {
        let mut random = Random(1);
        let arithmetic = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x12, 0x13];
        let inverted = [0x18, 0x19, 0x1A, 0x1B];
        for _ in 0..300 {
            let mut unit = ShaderUnit::new();
            for uniform in unit.float_uniforms.iter_mut() {
                *uniform = std::array::from_fn(|_| random.float());
            }
            for descriptor in unit.descriptors.iter_mut() {
                *descriptor = (random.next() & 0x7FFF_FFF0) | (random.below(15) + 1);
            }
            let mut program = Vec::new();
            // address registers from an input, for the indexed reads after
            program.push((0x12 << 26) | (random.below(16) << 12) | random.below(32));
            for _ in 0..24 {
                let (destination, index, descriptor) = (random.below(32), random.below(4), random.below(32));
                let wide = random.below(0x80);
                let narrow = random.below(0x20);
                let word = match random.below(10) {
                    0..=5 => {
                        let op = arithmetic[random.below(arithmetic.len() as u32) as usize];
                        (op << 26) | (destination << 21) | (index << 19) | (wide << 12) | (narrow << 7) | descriptor
                    }
                    6 => {
                        let op = inverted[random.below(inverted.len() as u32) as usize];
                        (op << 26) | (destination << 21) | (index << 19) | (narrow << 14) | (wide << 7) | descriptor
                    }
                    7 => {
                        // cmp, whose first mode shares a bit with the opcode
                        let modes = (random.below(8) << 24) | (random.below(8) << 21);
                        (0x2E << 26) | modes | (index << 19) | (wide << 12) | (narrow << 7) | descriptor
                    }
                    8 => {
                        let (src1, src3) = (random.below(0x20), random.below(0x20));
                        (0b111 << 29) | (destination << 24) | (index << 22) | (src1 << 17) | (wide << 10) | (src3 << 5) | descriptor
                    }
                    _ => {
                        let (src1, src2) = (random.below(0x20), random.below(0x20));
                        (0b110 << 29) | (destination << 24) | (index << 22) | (src1 << 17) | (src2 << 12) | (wide << 5) | descriptor
                    }
                };
                program.push(word);
            }
            program.push(0x22 << 26);
            for (i, &word) in program.iter().enumerate() {
                unit.program[i] = word;
            }
            unit.prepare();
            let count = 1 + random.below(20) as usize;
            let inputs: Vec<[Vec4; INPUT_REGISTERS]> = (0..count)
                .map(|_| std::array::from_fn(|_| std::array::from_fn(|_| random.float())))
                .collect();
            let single = one_by_one(&unit, &inputs);
            assert!(same(&run_vertices(&unit, &inputs), &single), "program {program:08X?}");
            assert_compiled(&unit);
            assert!(same(&batched(&unit, &inputs), &single), "program {program:08X?} batched");
        }
    }

    /// programs whose vertices branch apart every way the shader can, an
    /// if and else, a conditional jump, a loop left early and a call, come
    /// out the same batched as one vertex at a time.
    #[test]
    fn batches_that_branch_apart_shade_like_the_interpreter() {
        let mut random = Random(3);
        let arithmetic = [0x00, 0x01, 0x02, 0x03, 0x08, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x13];
        for _ in 0..300 {
            let mut unit = ShaderUnit::new();
            for uniform in unit.float_uniforms.iter_mut() {
                *uniform = std::array::from_fn(|_| random.float());
            }
            for descriptor in unit.descriptors.iter_mut() {
                *descriptor = (random.next() & 0x7FFF_FFF0) | (random.below(15) + 1);
            }
            unit.int_uniforms[0] = [2 + random.below(3) as u8, 0, 1, 0];
            let arith = |random: &mut Random| {
                let op = arithmetic[random.below(arithmetic.len() as u32) as usize];
                let (destination, wide, narrow) = (random.below(32), random.below(0x80), random.below(0x20));
                (op << 26) | (destination << 21) | (wide << 12) | (narrow << 7) | random.below(32)
            };
            let cmp = |random: &mut Random| {
                let modes = (random.below(6) << 24) | (random.below(6) << 21);
                (0x2E << 26) | modes | (random.below(0x80) << 12) | (random.below(0x20) << 7)
            };
            // which condition bits to compare against, and how x and y combine
            let test = |random: &mut Random| (random.below(4) << 24) | (random.below(4) << 22);
            let mut program: Vec<u32> = Vec::new();
            for _ in 0..6 {
                match random.below(5) {
                    0 | 1 => (0..3).for_each(|_| program.push(arith(&mut random))),
                    2 => {
                        program.push(cmp(&mut random));
                        let at = program.len();
                        program.push(0);
                        (0..1 + random.below(3)).for_each(|_| program.push(arith(&mut random)));
                        let otherwise = program.len() as u32;
                        let count = random.below(3);
                        (0..count).for_each(|_| program.push(arith(&mut random)));
                        program[at] = (0x28 << 26) | test(&mut random) | (otherwise << 10) | count;
                    }
                    3 => {
                        program.push(cmp(&mut random));
                        let at = program.len();
                        program.push(0);
                        (0..1 + random.below(3)).for_each(|_| program.push(arith(&mut random)));
                        program[at] = (0x2C << 26) | test(&mut random) | ((program.len() as u32) << 10);
                    }
                    _ => {
                        let at = program.len();
                        program.push(0);
                        program.push(arith(&mut random));
                        program.push(cmp(&mut random));
                        program.push((0x23 << 26) | test(&mut random));
                        program.push(arith(&mut random));
                        program[at] = (0x29 << 26) | (((program.len() - 1) as u32) << 10);
                    }
                }
            }
            // a call to a routine past the end
            program.push(cmp(&mut random));
            let call = program.len();
            program.push(0);
            program.push(arith(&mut random));
            program.push(0x22 << 26);
            let routine = program.len() as u32;
            (0..2).for_each(|_| program.push(arith(&mut random)));
            program[call] = (0x25 << 26) | test(&mut random) | (routine << 10) | 2;
            for (i, &word) in program.iter().enumerate() {
                unit.program[i] = word;
            }
            unit.prepare();
            let count = 1 + random.below(24) as usize;
            let inputs: Vec<[Vec4; INPUT_REGISTERS]> = (0..count)
                .map(|_| std::array::from_fn(|_| std::array::from_fn(|_| random.float())))
                .collect();
            let single = one_by_one(&unit, &inputs);
            assert!(same(&run_vertices(&unit, &inputs), &single), "program {program:08X?}");
            assert_compiled(&unit);
            assert!(same(&batched(&unit, &inputs), &single), "program {program:08X?} batched");
        }
    }

    /// ifs nested in ifs, on conditions and on booleans, and a call, whose
    /// vertices part ways at every one, come out the same compiled as one
    /// vertex at a time, without a batch going back.
    #[test]
    fn parted_branches_shade_like_the_interpreter() {
        fn block(random: &mut Random, program: &mut Vec<u32>, depth: u32) {
            let arithmetic = [0x00, 0x01, 0x02, 0x03, 0x08, 0x0B, 0x0C, 0x0D, 0x0E, 0x12, 0x13];
            for _ in 0..1 + random.below(3) {
                match random.below(if depth < 3 { 5 } else { 3 }) {
                    0 | 1 => {
                        let op = arithmetic[random.below(arithmetic.len() as u32) as usize];
                        let (destination, wide, narrow) = (random.below(32), random.below(0x80), random.below(0x20));
                        program.push((op << 26) | (destination << 21) | (random.below(4) << 19) | (wide << 12) | (narrow << 7) | random.below(32));
                    }
                    2 => {
                        let modes = (random.below(6) << 24) | (random.below(6) << 21);
                        program.push((0x2E << 26) | modes | (random.below(0x80) << 12) | (random.below(0x20) << 7));
                    }
                    _ => {
                        let at = program.len();
                        program.push(0);
                        block(random, program, depth + 1);
                        let otherwise = program.len() as u32;
                        if random.below(3) != 0 {
                            block(random, program, depth + 1);
                        }
                        let count = program.len() as u32 - otherwise;
                        program[at] = match random.below(4) {
                            0 => (0x27 << 26) | (random.below(16) << 22),
                            _ => (0x28 << 26) | (random.below(4) << 24) | (random.below(4) << 22),
                        } | (otherwise << 10)
                            | count;
                    }
                }
            }
        }
        let mut random = Random(5);
        for _ in 0..300 {
            let mut unit = ShaderUnit::new();
            for uniform in unit.float_uniforms.iter_mut() {
                *uniform = std::array::from_fn(|_| random.float());
            }
            for descriptor in unit.descriptors.iter_mut() {
                *descriptor = (random.next() & 0x7FFF_FFF0) | (random.below(15) + 1);
            }
            unit.bool_uniforms = random.next() as u16;
            // the address registers and conditions from the inputs first
            let mut program = vec![(0x12 << 26) | (random.below(16) << 12) | random.below(32)];
            block(&mut random, &mut program, 0);
            // a call to a routine past the end
            let call = program.len();
            program.push(0);
            block(&mut random, &mut program, 2);
            program.push(0x22 << 26);
            let routine = program.len() as u32;
            block(&mut random, &mut program, 3);
            let count = program.len() as u32 - routine;
            program[call] = (0x25 << 26) | (random.below(4) << 24) | (random.below(4) << 22) | (routine << 10) | count;
            for (i, &word) in program.iter().enumerate() {
                unit.program[i] = word;
            }
            unit.prepare();
            let count = 1 + random.below(24) as usize;
            let inputs: Vec<[Vec4; INPUT_REGISTERS]> = (0..count)
                .map(|_| std::array::from_fn(|_| std::array::from_fn(|_| random.float())))
                .collect();
            assert!(same(&run_vertices(&unit, &inputs), &one_by_one(&unit, &inputs)), "program {program:08X?}");
            #[cfg(feature = "jit")]
            {
                let compiled = jit::compiled(&unit, unit.decoded.as_ref().expect("prepared")).expect("compiled");
                assert_eq!(compiled.parted(), 0, "program {program:08X?} gave batches back where they parted");
            }
        }
    }

    /// how fast a typical transform and lighting program shades, compared
    /// with one vertex at a time, cargo test -- --ignored --nocapture.
    #[test]
    #[ignore]
    fn batch_speed() {
        let mut random = Random(7);
        let mut unit = ShaderUnit::new();
        for uniform in unit.float_uniforms.iter_mut() {
            *uniform = std::array::from_fn(|_| (random.next() % 100) as f32 / 50.0 - 1.0);
        }
        unit.descriptors[0] = IDENTITY;
        let dp4 = |dest: u32, uniform: u32, src: u32| (0x02 << 26) | (dest << 21) | (uniform << 12) | (src << 7);
        let mut program = Vec::new();
        // four dot products into r0, four into o0, a multiply-add, a
        // max, a mul, a mov, the kind of thing a vertex program does,
        // eight times over, about as long as the programs titles run
        for _ in 0..8 {
            for i in 0..4 {
                program.push(dp4(0x10, 0x20 + i, 0));
            }
            for i in 0..4 {
                program.push(dp4(0x00, 0x24 + i, 0x10));
            }
            program.push((0b111 << 29) | (0x11 << 24) | (1 << 17) | (0x28 << 10) | (0x10 << 5));
            program.push((0x0C << 26) | (0x12 << 21) | (0x11 << 12) | (1 << 7));
            program.push((0x08 << 26) | (0x01 << 21) | (0x29 << 12) | (0x12 << 7));
            program.push((0x13 << 26) | (0x02 << 21) | (2 << 12));
        }
        program.push(0x22 << 26);
        for (i, &word) in program.iter().enumerate() {
            unit.program[i] = word;
        }
        unit.prepare();
        // a draw's worth, which the cache holds, as a draw's inputs are,
        // shaded many times over
        let inputs: Vec<[Vec4; INPUT_REGISTERS]> =
            (0..4096).map(|_| std::array::from_fn(|_| std::array::from_fn(|_| (random.next() % 100) as f32 / 50.0))).collect();
        const TIMES: usize = 50;
        let instructions = (program.len() - 1) as f64 * inputs.len() as f64 * TIMES as f64;
        let timed = |shade: &dyn Fn() -> Vec<[Vec4; OUTPUT_REGISTERS]>| {
            let start = std::time::Instant::now();
            for _ in 1..TIMES {
                std::hint::black_box(shade());
            }
            let outputs = shade();
            (outputs, start.elapsed().as_secs_f64())
        };
        // once to compile it, then timed
        run_vertices(&unit, &inputs[..8]);
        let (fast, compiled_time) = timed(&|| run_vertices(&unit, &inputs));
        let (wide, batched_time) = timed(&|| batched(&unit, &inputs));
        let (single, alone) = timed(&|| one_by_one(&unit, &inputs));
        assert!(same(&wide, &single) && same(&fast, &single));
        println!(
            "compiled {:.2} ns an instruction a vertex, batched {:.2}, one by one {:.2}",
            compiled_time * 1e9 / instructions,
            batched_time * 1e9 / instructions,
            alone * 1e9 / instructions
        );
    }

    /// the compiled program alone, many times over, for a profiler.
    #[test]
    #[ignore]
    fn compiled_speed() {
        let mut random = Random(7);
        let mut unit = ShaderUnit::new();
        for uniform in unit.float_uniforms.iter_mut() {
            *uniform = std::array::from_fn(|_| (random.next() % 100) as f32 / 50.0 - 1.0);
        }
        unit.descriptors[0] = IDENTITY;
        let dp4 = |dest: u32, uniform: u32, src: u32| (0x02 << 26) | (dest << 21) | (uniform << 12) | (src << 7);
        let mut program = Vec::new();
        for _ in 0..8 {
            for i in 0..4 {
                program.push(dp4(0x10, 0x20 + i, 0));
            }
            for i in 0..4 {
                program.push(dp4(0x00, 0x24 + i, 0x10));
            }
            program.push((0b111 << 29) | (0x11 << 24) | (1 << 17) | (0x28 << 10) | (0x10 << 5));
            program.push((0x0C << 26) | (0x12 << 21) | (0x11 << 12) | (1 << 7));
            program.push((0x08 << 26) | (0x01 << 21) | (0x29 << 12) | (0x12 << 7));
            program.push((0x13 << 26) | (0x02 << 21) | (2 << 12));
        }
        program.push(0x22 << 26);
        for (i, &word) in program.iter().enumerate() {
            unit.program[i] = word;
        }
        unit.prepare();
        // a draw's worth, which the cache holds, as a draw's inputs are
        let inputs: Vec<[Vec4; INPUT_REGISTERS]> =
            (0..4096).map(|_| std::array::from_fn(|_| std::array::from_fn(|_| (random.next() % 100) as f32 / 50.0))).collect();
        run_vertices(&unit, &inputs[..8]);
        let start = std::time::Instant::now();
        for _ in 0..400 {
            std::hint::black_box(run_vertices(&unit, &inputs));
        }
        let instructions = (program.len() - 1) as f64 * inputs.len() as f64 * 400.0;
        println!("compiled {:.3} ns an instruction a vertex", start.elapsed().as_secs_f64() * 1e9 / instructions);
    }

    /// a branch half the vertices take and half do not still sends each
    /// its own way.
    #[test]
    fn a_branch_that_splits_a_batch() {
        // cmp c0, v0 with x less than, so v0 above zero, then an if on x,
        // taking mov o0, c1 or the else, mov o0, c2
        let cmp = (0x2E << 26) | (2 << 24) | (0x20 << 12);
        let ifc = (0x28 << 26) | (1 << 25) | (2 << 22) | (3 << 10) | 1;
        let then = (0x13 << 26) | (0x21 << 12);
        let otherwise = (0x13 << 26) | (0x22 << 12);
        let mut unit = unit_with(&[cmp, ifc, then, otherwise, 0x22 << 26], &[IDENTITY]);
        unit.float_uniforms[1] = [1.0; 4];
        unit.float_uniforms[2] = [2.0; 4];
        unit.prepare();
        let inputs: Vec<[Vec4; INPUT_REGISTERS]> = (0..12)
            .map(|i| {
                let mut input = [ZERO; INPUT_REGISTERS];
                input[0] = [if i % 2 == 0 { 1.0 } else { -1.0 }; 4];
                input
            })
            .collect();
        let outputs = run_vertices(&unit, &inputs);
        for (i, output) in outputs.iter().enumerate() {
            assert_eq!(output[0], if i % 2 == 0 { [1.0; 4] } else { [2.0; 4] });
        }
        assert!(same(&outputs, &one_by_one(&unit, &inputs)));
    }

    /// a loop adds up the uniforms its counter walks over.
    #[test]
    fn a_loop_walks_uniforms_the_same_way() {
        // loop i0 over add r0, r0, c0[aL], then mov o0, r0
        let looped = (0x29 << 26) | (1 << 10);
        let add = (0x10 << 21) | (3 << 19) | (0x20 << 12) | (0x10 << 7);
        let mov = (0x13 << 26) | (0x10 << 12);
        let mut unit = unit_with(&[looped, add, mov, 0x22 << 26], &[IDENTITY]);
        unit.int_uniforms[0] = [3, 2, 1, 0];
        for (i, uniform) in unit.float_uniforms.iter_mut().enumerate() {
            *uniform = [i as f32; 4];
        }
        unit.prepare();
        let inputs = vec![[ZERO; INPUT_REGISTERS]; 5];
        let outputs = run_vertices(&unit, &inputs);
        // four iterations, over c2 to c5, onto r0's (0, 0, 0, 1)
        assert_eq!(outputs[4][0], [14.0, 14.0, 14.0, 15.0]);
        assert!(same(&outputs, &one_by_one(&unit, &inputs)));
    }

    #[test]
    fn moves_an_input_to_an_output() {
        // mov o0, v0, end
        let unit = unit_with(&[0x13 << 26, 0x22 << 26], &[IDENTITY]);
        let mut state = ShaderState::new();
        state.input[0] = [1.0, 2.0, 3.0, 4.0];
        run(&unit, &mut state);
        assert_eq!(state.output[0], [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn adds_two_inputs() {
        // add o0, v0, v1, end   (src1 = 0, src2 = 1, dest = 0)
        let add = 1 << 7;
        let unit = unit_with(&[add, 0x22 << 26], &[IDENTITY]);
        let mut state = ShaderState::new();
        state.input[0] = [1.0, 2.0, 3.0, 4.0];
        state.input[1] = [10.0, 20.0, 30.0, 40.0];
        run(&unit, &mut state);
        assert_eq!(state.output[0], [11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn honours_the_destination_mask() {
        // only the x and z components are written.
        let descriptor = 0b1010 | (0b00_01_10_11 << 5) | (0b00_01_10_11 << 14);
        let unit = unit_with(&[0x13 << 26, 0x22 << 26], &[descriptor]);
        let mut state = ShaderState::new();
        state.input[0] = [1.0, 2.0, 3.0, 4.0];
        run(&unit, &mut state);
        assert_eq!(state.output[0], [1.0, 0.0, 3.0, 0.0]);
    }

    /// a temporary read before anything is written to it holds a w of 1,
    /// shaded on its own or in a batch.
    #[test]
    fn an_unwritten_temporary_has_a_w_of_one() {
        // mov o0, r15, end
        let unit = unit_with(&[(0x13 << 26) | (0x1F << 12), 0x22 << 26], &[IDENTITY]);
        let inputs = [[ZERO; INPUT_REGISTERS]];
        assert_eq!(one_by_one(&unit, &inputs)[0][0], [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(run_vertices(&unit, &inputs)[0][0], [0.0, 0.0, 0.0, 1.0]);
    }

    /// a shader with no end must still terminate.
    #[test]
    fn a_runaway_program_stops() {
        // jmp 0, forever.
        let jump = 0x2C << 26;
        let unit = unit_with(&[jump], &[IDENTITY]);
        let mut state = ShaderState::new();
        run(&unit, &mut state);
    }

    /// a program that never ends stops where the interpreter stops it, the
    /// compiled code giving the batch back once it runs that long.
    #[test]
    fn a_runaway_program_stops_the_same_compiled() {
        // jmpc 0 on a condition that holds, forever
        let mut unit = unit_with(&[0x2C << 26], &[IDENTITY]);
        unit.prepare();
        let inputs = vec![[[1.0; 4]; INPUT_REGISTERS]; 4];
        assert_compiled(&unit);
        assert!(same(&run_vertices(&unit, &inputs), &one_by_one(&unit, &inputs)));
    }

    /// a program calling itself goes too deep to compile, and the batches
    /// shade it.
    #[test]
    fn a_program_too_deep_to_compile_still_shades() {
        // call 0 for one instruction, which calls it again
        let mut unit = unit_with(&[(0x24 << 26) | 1, 0x22 << 26], &[IDENTITY]);
        unit.prepare();
        #[cfg(feature = "jit")]
        assert!(jit::compile(unit.decoded.as_ref().expect("prepared"), 0).is_err());
        let inputs = vec![[[1.0; 4]; INPUT_REGISTERS]; 4];
        assert!(same(&run_vertices(&unit, &inputs), &one_by_one(&unit, &inputs)));
    }

    #[test]
    fn dot_product_broadcasts_to_every_component() {
        // dp4 o0, v0, v1, end
        let dp4 = (0x02 << 26) | (1 << 7);
        let unit = unit_with(&[dp4, 0x22 << 26], &[IDENTITY]);
        let mut state = ShaderState::new();
        state.input[0] = [1.0, 2.0, 3.0, 4.0];
        state.input[1] = [1.0, 1.0, 1.0, 1.0];
        run(&unit, &mut state);
        assert_eq!(state.output[0], [10.0; 4]);
    }

    /// descriptor with identity swizzles on all three sources.
    const IDENTITY3: u32 = IDENTITY | (0b00_01_10_11 << 23);

    #[test]
    fn opcode_eight_is_multiply() {
        // mul o0, v0, v1, end
        let mul = (0x08 << 26) | (1 << 7);
        let unit = unit_with(&[mul, 0x22 << 26], &[IDENTITY]);
        let mut state = ShaderState::new();
        state.input[0] = [1.0, 2.0, 3.0, 4.0];
        state.input[1] = [2.0, 3.0, 4.0, 5.0];
        run(&unit, &mut state);
        assert_eq!(state.output[0], [2.0, 6.0, 12.0, 20.0]);
    }

    #[test]
    fn multiply_add_reads_a_uniform_through_its_wide_second_source() {
        // mad o0, v0, c1, v1, end   (dest 0, src1 v0, src2 c1 = 0x21, src3 v1)
        let mad = (0x38 << 26) | (0x21 << 10) | (1 << 5);
        let mut unit = unit_with(&[mad, 0x22 << 26], &[IDENTITY3]);
        unit.float_uniforms[1] = [10.0, 10.0, 10.0, 10.0];
        let mut state = ShaderState::new();
        state.input[0] = [1.0, 2.0, 3.0, 4.0];
        state.input[1] = [0.5; 4];
        run(&unit, &mut state);
        assert_eq!(state.output[0], [10.5, 20.5, 30.5, 40.5]);
    }

    #[test]
    fn inverted_multiply_add_reads_a_uniform_through_its_third_source() {
        // madi o0, v0, v1, c2, end   (src2 v1 narrow, src3 c2 = 0x22 wide)
        let madi = (0x30 << 26) | (1 << 12) | (0x22 << 5);
        let mut unit = unit_with(&[madi, 0x22 << 26], &[IDENTITY3]);
        unit.float_uniforms[2] = [100.0; 4];
        let mut state = ShaderState::new();
        state.input[0] = [1.0, 2.0, 3.0, 4.0];
        state.input[1] = [2.0; 4];
        run(&unit, &mut state);
        assert_eq!(state.output[0], [102.0, 104.0, 106.0, 108.0]);
    }

    /// a loop runs its whole body count + 1 times, not just its last
    /// instruction.
    #[test]
    fn a_loop_repeats_its_whole_body() {
        // 0  loop i0, last = 2
        // 1  add r0, r0, v0
        // 2  add r0, r0, v0
        // 3  mov o0, r0
        // 4  end
        let program = [
            (0x29 << 26) | (2 << 10),
            (0x10 << 21) | (0x10 << 12),
            (0x10 << 21) | (0x10 << 12),
            (0x13 << 26) | (0x10 << 12),
            0x22 << 26,
        ];
        let mut unit = unit_with(&program, &[IDENTITY]);
        unit.int_uniforms[0] = [2, 0, 1, 0];
        let mut state = ShaderState::new();
        state.input[0] = [1.0; 4];
        run(&unit, &mut state);
        // three iterations of two additions each, onto r0's (0, 0, 0, 1).
        assert_eq!(state.output[0], [6.0, 6.0, 6.0, 7.0]);
    }

    #[test]
    fn jmpu_can_jump_on_a_false_boolean() {
        // 0  jmpu !b0, 2   (count bit 0 set inverts the test)
        // 1  mov o0, v0
        // 2  end
        let program = [
            (0x2D << 26) | (2 << 10) | 1,
            0x13 << 26,
            0x22 << 26,
        ];
        let unit = unit_with(&program, &[IDENTITY]);
        let mut state = ShaderState::new();
        state.input[0] = [1.0; 4];
        run(&unit, &mut state);
        assert_eq!(state.output[0], ZERO, "b0 is false, so the move is skipped");
    }

    /// a program decoded ahead of time is decoded again once the command
    /// list uploads another.
    #[test]
    fn uploading_a_program_decodes_it_again() {
        let mut unit = unit_with(&[0x13 << 26, 0x22 << 26], &[IDENTITY]);
        unit.prepare();
        let mut state = ShaderState::new();
        state.input[0] = [1.0, 2.0, 3.0, 4.0];
        state.input[1] = [10.0, 20.0, 30.0, 40.0];
        run(&unit, &mut state);
        assert_eq!(state.output[0], [1.0, 2.0, 3.0, 4.0]);
        // add o0, v0, v1 in place of the move
        unit.program_write_offset = 0;
        unit.upload_program(1 << 7);
        unit.prepare();
        run(&unit, &mut state);
        assert_eq!(state.output[0], [11.0, 22.0, 33.0, 44.0]);
    }

    /// float uniforms sent as a run land as they do a word at a time,
    /// packed and wide, starting part way through a vector or not, ending
    /// part way through one or not, past the last uniform too.
    #[test]
    fn uniforms_sent_as_a_run_land_as_one_word_at_a_time() {
        let mut random = Random(7);
        for case in 0..2000 {
            let mut runs = ShaderUnit::new();
            // a wide upload before leaves a fourth staged word behind
            runs.set_float_uniform_index(0x8000_0000);
            for _ in 0..random.below(5) {
                runs.upload_float_uniform(random.next());
            }
            let wide = if random.below(2) == 0 { 0x8000_0000 } else { 0 };
            runs.set_float_uniform_index(random.below(0x80) | wide);
            for _ in 0..random.below(4) {
                runs.upload_float_uniform(random.next());
            }
            let mut words = runs.clone();
            let run: Vec<u32> = (0..random.below(60))
                .map(|_| if random.below(4) == 0 { random.float().to_bits() } else { random.next() })
                .collect();
            runs.upload_float_uniforms(&run);
            for &word in &run {
                words.upload_float_uniform(word);
            }
            assert_eq!(runs.cursors(), words.cursors(), "case {case}");
            let bits = |unit: &ShaderUnit| unit.float_uniforms.as_flattened().iter().map(|c| c.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&runs), bits(&words), "case {case}");
        }
    }

    #[test]
    fn zero_times_infinity_is_zero() {
        assert_eq!(multiply(0.0, f32::INFINITY), 0.0);
        assert!(multiply(f32::NAN, 1.0).is_nan());
    }

    /// a geometry shader turning one point into a triangle, three emits,
    /// the last one completing the primitive.
    #[test]
    fn a_geometry_shader_emits_triangles() {
        let setemit = |slot: u32, primitive: bool| (0x2B << 26) | (slot << 24) | ((primitive as u32) << 23);
        let emit = 0x2A << 26;
        let mov = 0x13 << 26; // mov o0, v0
        let add = 1 << 7; // add o0, v0, v1
        let program = [
            setemit(0, false),
            mov,
            emit,
            setemit(1, false),
            add,
            emit,
            setemit(2, true),
            mov,
            emit,
            0x22 << 26,
        ];
        let unit = unit_with(&program, &[IDENTITY]);
        let mut state = ShaderState::new();
        state.input[0] = [1.0, 2.0, 0.0, 1.0];
        state.input[1] = [10.0, 0.0, 0.0, 0.0];
        let mut emitter = Emitter::default();
        run_geometry(&unit, &mut state, &mut emitter);

        assert_eq!(emitter.triangles.len(), 1);
        let [a, b, c] = emitter.triangles[0];
        assert_eq!(a[0], [1.0, 2.0, 0.0, 1.0]);
        assert_eq!(b[0], [11.0, 2.0, 0.0, 1.0]);
        assert_eq!(c[0], [1.0, 2.0, 0.0, 1.0]);
    }
}
