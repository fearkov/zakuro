//! the PICA200's shader units, which run both vertex and geometry shaders.

pub mod isa;

use isa::{Instruction, OpCode, OperandDescriptor};

pub const PROGRAM_SIZE: usize = 4096;
pub const DESCRIPTOR_SIZE: usize = 128;
pub const FLOAT_UNIFORMS: usize = 96;
pub const INPUT_REGISTERS: usize = 16;
pub const OUTPUT_REGISTERS: usize = 16;
pub const TEMP_REGISTERS: usize = 16;

/// a four-component vector, which is the only data type the shader has.
pub type Vec4 = [f32; 4];

pub const ZERO: Vec4 = [0.0; 4];

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
        }
    }

    pub fn upload_program(&mut self, word: u32) {
        if self.program_write_offset < PROGRAM_SIZE {
            self.program[self.program_write_offset] = word;
            self.program_write_offset += 1;
        }
    }

    pub fn upload_descriptor(&mut self, word: u32) {
        if self.descriptor_write_offset < DESCRIPTOR_SIZE {
            self.descriptors[self.descriptor_write_offset] = word;
            self.descriptor_write_offset += 1;
        }
    }

    /// starts a float uniform upload at the register the raw value names.
    pub fn set_float_uniform_index(&mut self, raw: u32) {
        self.float_uniform_index = (raw & 0x7F) as usize;
        self.float_uniform_component = 0;
        self.float_uniform_wide = raw & 0x8000_0000 != 0;
    }

    /// float uniforms arrive component by component, three words carry four
    /// packed 24-bit floats, or four words carry plain singles.
    pub fn upload_float_uniform(&mut self, word: u32) {
        let needed = if self.float_uniform_wide { 4 } else { 3 };
        if self.float_uniform_component < 4 {
            self.float_uniform_staging[self.float_uniform_component] = word;
        }
        self.float_uniform_component += 1;
        if self.float_uniform_component < needed {
            return;
        }

        let staged = self.float_uniform_staging;
        self.float_uniform_component = 0;

        let value = if self.float_uniform_wide {
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
                if self.float_uniform_wide { 32 } else { 24 },
            );
        }
        if self.float_uniform_index < FLOAT_UNIFORMS {
            self.float_uniforms[self.float_uniform_index] = value;
        }
        self.float_uniform_index += 1;
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
            temp: [ZERO; TEMP_REGISTERS],
            address: [0; 3],
            condition: [false; 2],
            blocks: Vec::with_capacity(16),
        }
    }

    fn reset(&mut self) {
        self.output = [ZERO; OUTPUT_REGISTERS];
        self.temp = [ZERO; TEMP_REGISTERS];
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

/// runs the shader over one vertex.
pub fn run(unit: &ShaderUnit, state: &mut ShaderState) {
    execute(unit, state, None);
}

/// runs a geometry shader over one primitive's worth of input, collecting
/// what it emits.
pub fn run_geometry(unit: &ShaderUnit, state: &mut ShaderState, emitter: &mut Emitter) {
    execute(unit, state, Some(emitter));
}

fn execute(unit: &ShaderUnit, state: &mut ShaderState, mut emitter: Option<&mut Emitter>) {
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

        let instruction = Instruction(unit.program[pc as usize]);
        let mut next = pc + 1;

        match instruction.opcode() {
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
                multiply_add(unit, state, instruction);
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
                arithmetic(unit, state, instruction);
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

fn arithmetic(unit: &ShaderUnit, state: &mut ShaderState, instruction: Instruction) {
    let descriptor = OperandDescriptor(
        unit.descriptors[instruction.descriptor_index() as usize % DESCRIPTOR_SIZE],
    );

    let (mut src1_register, mut src2_register) = instruction.sources();
    // only the wide operand can be a uniform, and only a uniform read can be
    // indexed by an address register.
    let index = instruction.address_register_index();
    if instruction.opcode().is_inverted() {
        src2_register = indexed(state, src2_register, index);
    } else {
        src1_register = indexed(state, src1_register, index);
    }

    let src1 = descriptor.apply_source1(source_value(unit, state, src1_register));
    let src2 = descriptor.apply_source2(source_value(unit, state, src2_register));

    let result: Vec4 = match instruction.opcode() {
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
            let mask = descriptor.destination_mask();
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

    write_masked(
        state,
        instruction.destination(),
        descriptor.destination_mask(),
        result,
    );
}

fn multiply_add(unit: &ShaderUnit, state: &mut ShaderState, instruction: Instruction) {
    let descriptor = OperandDescriptor(
        unit.descriptors[instruction.mad_descriptor_index() as usize % DESCRIPTOR_SIZE],
    );
    let (src1_register, mut src2_register, mut src3_register) = instruction.mad_sources();
    // the address register applies to whichever source is the wide one.
    let index = instruction.mad_address_register_index();
    if instruction.opcode() == OpCode::MadI {
        src3_register = indexed(state, src3_register, index);
    } else {
        src2_register = indexed(state, src2_register, index);
    }

    let src1 = descriptor.apply_source1(source_value(unit, state, src1_register));
    let src2 = descriptor.apply_source2(source_value(unit, state, src2_register));
    let src3 = descriptor.apply_source3(source_value(unit, state, src3_register));

    let result = [
        multiply(src1[0], src2[0]) + src3[0],
        multiply(src1[1], src2[1]) + src3[1],
        multiply(src1[2], src2[2]) + src3[2],
        multiply(src1[3], src2[3]) + src3[3],
    ];
    write_masked(
        state,
        instruction.mad_destination(),
        descriptor.destination_mask(),
        result,
    );
}

/// the shader's multiply, which gives zero rather than NaN for zero times
/// infinity.
#[inline]
fn multiply(a: f32, b: f32) -> f32 {
    let product = a * b;
    if product.is_nan() && !a.is_nan() && !b.is_nan() {
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

    /// a shader with no end must still terminate.
    #[test]
    fn a_runaway_program_stops() {
        // jmp 0, forever.
        let jump = 0x2C << 26;
        let unit = unit_with(&[jump], &[IDENTITY]);
        let mut state = ShaderState::new();
        run(&unit, &mut state);
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
        // three iterations of two additions each.
        assert_eq!(state.output[0], [6.0; 4]);
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
