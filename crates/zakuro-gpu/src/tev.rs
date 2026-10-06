//! the PICA200's texture environment, six configurable combiner stages.

/// where each stage's five registers start.
const STAGE_REGISTERS: [usize; 6] = [0x0C0, 0x0C8, 0x0D0, 0x0D8, 0x0F0, 0x0F8];

/// register holding which stages write to the buffer that
/// [Source::PreviousBuffer] reads.
const REG_UPDATE_BUFFER: usize = 0x0E0;
/// register holding the buffer's initial color.
const REG_BUFFER_COLOR: usize = 0x0FD;
/// the registers the environment is decoded from, each stage's five, then
/// the buffer's update bits and its color.
const REGISTERS_READ: usize = 6 * 5 + 2;

/// where one of a stage's three inputs comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    PrimaryColor,
    PrimaryFragmentColor,
    SecondaryFragmentColor,
    Texture(usize),
    PreviousBuffer,
    Constant,
    Previous,
}

impl Source {
    fn from_raw(value: u32) -> Source {
        match value & 0xF {
            0x0 => Source::PrimaryColor,
            0x1 => Source::PrimaryFragmentColor,
            0x2 => Source::SecondaryFragmentColor,
            0x3 => Source::Texture(0),
            0x4 => Source::Texture(1),
            0x5 => Source::Texture(2),
            0x6 => Source::Texture(3),
            0xD => Source::PreviousBuffer,
            0xE => Source::Constant,
            _ => Source::Previous,
        }
    }
}

/// how the stages fold their inputs together.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Operation {
    Replace,
    Modulate,
    Add,
    AddSigned,
    Lerp,
    Subtract,
    Dot3Rgb,
    Dot3Rgba,
    MultiplyAdd,
    AddMultiply,
}

impl Operation {
    fn from_raw(value: u32) -> Operation {
        match value & 0xF {
            0 => Operation::Replace,
            1 => Operation::Modulate,
            2 => Operation::Add,
            3 => Operation::AddSigned,
            4 => Operation::Lerp,
            5 => Operation::Subtract,
            6 => Operation::Dot3Rgb,
            7 => Operation::Dot3Rgba,
            8 => Operation::MultiplyAdd,
            _ => Operation::AddMultiply,
        }
    }

    /// how many of a stage's three inputs the operation reads.
    fn inputs(self) -> usize {
        match self {
            Operation::Replace => 1,
            Operation::Lerp | Operation::MultiplyAdd | Operation::AddMultiply => 3,
            _ => 2,
        }
    }
}

/// one configured stage.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Stage {
    color_sources: [Source; 3],
    alpha_sources: [Source; 3],
    color_operands: [u32; 3],
    alpha_operands: [u32; 3],
    color_op: Operation,
    alpha_op: Operation,
    constant: [f32; 4],
    color_scale: f32,
    alpha_scale: f32,
}

/// the whole texture environment, decoded from the register file once per
/// draw rather than per pixel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TexEnv {
    stages: [Stage; 6],
    /// stages that hand the previous stage's result on unchanged.
    passthrough: [bool; 6],
    /// whether each of the first four stages writes its color and its
    /// alpha into the buffer later stages read as PreviousBuffer.
    update_color: [bool; 4],
    update_alpha: [bool; 4],
    buffer_color: [f32; 4],
}

impl Stage {
    /// replace, taking the previous stage's color and alpha as they are,
    /// at a scale of one, a stage that changes nothing.
    fn is_passthrough(&self) -> bool {
        self.color_op == Operation::Replace
            && self.alpha_op == Operation::Replace
            && self.color_sources[0] == Source::Previous
            && self.alpha_sources[0] == Source::Previous
            && self.color_operands[0] == 0
            && self.alpha_operands[0] == 0
            && self.color_scale == 1.0
            && self.alpha_scale == 1.0
    }
}

impl TexEnv {
    pub fn read(registers: &[u32]) -> TexEnv {
        TexEnv::decode(&TexEnv::registers_read(registers))
    }

    /// the registers decode takes, in its order.
    fn registers_read(registers: &[u32]) -> [u32; REGISTERS_READ] {
        let mut read = [0; REGISTERS_READ];
        for (stage, base) in STAGE_REGISTERS.into_iter().enumerate() {
            read[stage * 5..stage * 5 + 5].copy_from_slice(&registers[base..base + 5]);
        }
        read[30] = registers[REG_UPDATE_BUFFER];
        read[31] = registers[REG_BUFFER_COLOR];
        read
    }

    fn decode(read: &[u32; REGISTERS_READ]) -> TexEnv {
        let (stage_registers, _) = read.as_chunks::<5>();
        let stages: [Stage; 6] = std::array::from_fn(|i| {
            let [source, operand, combiner, constant, scale] = stage_registers[i];

            Stage {
                color_sources: [
                    Source::from_raw(source),
                    Source::from_raw(source >> 4),
                    Source::from_raw(source >> 8),
                ],
                alpha_sources: [
                    Source::from_raw(source >> 16),
                    Source::from_raw(source >> 20),
                    Source::from_raw(source >> 24),
                ],
                color_operands: [operand & 0xF, (operand >> 4) & 0xF, (operand >> 8) & 0xF],
                alpha_operands: [
                    (operand >> 12) & 0x7,
                    (operand >> 16) & 0x7,
                    (operand >> 20) & 0x7,
                ],
                color_op: Operation::from_raw(combiner),
                alpha_op: Operation::from_raw(combiner >> 16),
                constant: unpack(constant),
                color_scale: scale_factor(scale),
                alpha_scale: scale_factor(scale >> 16),
            }
        });

        // bits 8-11 say which of stages 0-3 write their color into the
        // buffer, bits 12-15 their alpha.
        let update = read[30];
        TexEnv {
            stages,
            passthrough: stages.map(|stage| stage.is_passthrough()),
            update_color: std::array::from_fn(|i| update & (0x100 << i) != 0),
            update_alpha: std::array::from_fn(|i| update & (0x1000 << i) != 0),
            buffer_color: unpack(read[31]),
        }
    }

    /// whether any stage reads the colors fragment lighting produces, when
    /// none does there is no point in working them out.
    pub fn reads_lighting(&self) -> bool {
        let lit = |source: &Source| matches!(source, Source::PrimaryFragmentColor | Source::SecondaryFragmentColor);
        self.stages.iter().zip(self.passthrough).any(|(stage, passthrough)| {
            !passthrough
                && (stage.color_sources[..stage.color_op.inputs()].iter().any(lit)
                    || (stage.color_op != Operation::Dot3Rgba
                        && stage.alpha_sources[..stage.alpha_op.inputs()].iter().any(lit)))
        })
    }

    /// runs every stage for one fragment.
    /// fragment is the primary and secondary color fragment lighting
    /// produced, when it is on.
    pub fn apply(&self, primary: [f32; 4], textures: [[f32; 4]; 4], fragment: Option<([f32; 4], [f32; 4])>) -> [f32; 4] {
        let mut previous = primary;
        // the buffer lags a stage behind, the first stage reads zero, the
        // second the configured buffer color, and a stage's update is only
        // visible to the stage after the next one.
        let mut buffer = [0.0; 4];
        let mut next_buffer = self.buffer_color;

        for (index, stage) in self.stages.iter().enumerate() {
            if !self.passthrough[index] {
                let pick = |source: Source| -> [f32; 4] {
                    match source {
                        Source::PrimaryColor => primary,
                        Source::PrimaryFragmentColor => fragment.map_or(primary, |(diffuse, _)| diffuse),
                        // without fragment lighting there is no specular term
                        Source::SecondaryFragmentColor => fragment.map_or([0.0, 0.0, 0.0, 1.0], |(_, specular)| specular),
                        Source::Texture(unit) => textures[unit.min(3)],
                        Source::PreviousBuffer => buffer,
                        Source::Constant => stage.constant,
                        Source::Previous => previous,
                    }
                };

                let rgb: [[f32; 3]; 3] = std::array::from_fn(|i| {
                    color_operand(pick(stage.color_sources[i]), stage.color_operands[i])
                });
                let alpha: [f32; 3] = std::array::from_fn(|i| {
                    alpha_operand(pick(stage.alpha_sources[i]), stage.alpha_operands[i])
                });

                let combined_rgb = combine_rgb(stage.color_op, rgb);
                let combined_alpha = match stage.color_op {
                    // this operation writes the dot product to alpha as
                    // well, ignoring the alpha combiner entirely.
                    Operation::Dot3Rgba => combined_rgb[0],
                    _ => combine_alpha(stage.alpha_op, alpha),
                };

                previous = [
                    (combined_rgb[0] * stage.color_scale).clamp(0.0, 1.0),
                    (combined_rgb[1] * stage.color_scale).clamp(0.0, 1.0),
                    (combined_rgb[2] * stage.color_scale).clamp(0.0, 1.0),
                    (combined_alpha * stage.alpha_scale).clamp(0.0, 1.0),
                ];
            }

            buffer = next_buffer;
            if index < 4 {
                if self.update_color[index] {
                    next_buffer[..3].copy_from_slice(&previous[..3]);
                }
                if self.update_alpha[index] {
                    next_buffer[3] = previous[3];
                }
            }
        }

        previous
    }
}

/// the texture environment the last draw decoded, and the registers it came
/// from. most draws keep the combiners of the one before, and comparing the
/// registers costs far less than decoding them again.
#[derive(Default)]
pub(crate) struct LastTexEnv {
    read: [u32; REGISTERS_READ],
    tex_env: Option<TexEnv>,
}

impl LastTexEnv {
    /// the texture environment the registers set up.
    pub(crate) fn read(&mut self, registers: &[u32]) -> TexEnv {
        let read = TexEnv::registers_read(registers);
        match self.tex_env {
            Some(tex_env) if self.read == read => tex_env,
            _ => {
                let tex_env = TexEnv::decode(&read);
                *self = LastTexEnv { read, tex_env: Some(tex_env) };
                tex_env
            }
        }
    }
}

/// splits a packed RGBA register value into 0..1 components.
fn unpack(value: u32) -> [f32; 4] {
    [
        (value & 0xFF) as f32 / 255.0,
        ((value >> 8) & 0xFF) as f32 / 255.0,
        ((value >> 16) & 0xFF) as f32 / 255.0,
        ((value >> 24) & 0xFF) as f32 / 255.0,
    ]
}

/// a stage's output multiplier, the raw field is a shift, not a factor.
fn scale_factor(raw: u32) -> f32 {
    match raw & 0x3 {
        1 => 2.0,
        2 => 4.0,
        _ => 1.0,
    }
}

fn color_operand(source: [f32; 4], operand: u32) -> [f32; 3] {
    match operand {
        0x0 => [source[0], source[1], source[2]],
        0x1 => [1.0 - source[0], 1.0 - source[1], 1.0 - source[2]],
        0x2 => [source[3]; 3],
        0x3 => [1.0 - source[3]; 3],
        0x4 => [source[0]; 3],
        0x5 => [1.0 - source[0]; 3],
        0x8 => [source[1]; 3],
        0x9 => [1.0 - source[1]; 3],
        0xC => [source[2]; 3],
        0xD => [1.0 - source[2]; 3],
        _ => [source[0], source[1], source[2]],
    }
}

fn alpha_operand(source: [f32; 4], operand: u32) -> f32 {
    match operand {
        0x0 => source[3],
        0x1 => 1.0 - source[3],
        0x2 => source[0],
        0x3 => 1.0 - source[0],
        0x4 => source[1],
        0x5 => 1.0 - source[1],
        0x6 => source[2],
        _ => 1.0 - source[2],
    }
}

fn combine_rgb(op: Operation, v: [[f32; 3]; 3]) -> [f32; 3] {
    match op {
        Operation::Replace => v[0],
        Operation::Modulate => std::array::from_fn(|i| v[0][i] * v[1][i]),
        Operation::Add => std::array::from_fn(|i| (v[0][i] + v[1][i]).min(1.0)),
        Operation::AddSigned => {
            std::array::from_fn(|i| (v[0][i] + v[1][i] - 0.5).clamp(0.0, 1.0))
        }
        Operation::Lerp => {
            std::array::from_fn(|i| v[0][i] * v[2][i] + v[1][i] * (1.0 - v[2][i]))
        }
        Operation::Subtract => std::array::from_fn(|i| (v[0][i] - v[1][i]).max(0.0)),
        Operation::Dot3Rgb | Operation::Dot3Rgba => {
            // both inputs are signed values packed into 0..1.
            let dot = (0..3)
                .map(|i| (v[0][i] * 2.0 - 1.0) * (v[1][i] * 2.0 - 1.0))
                .sum::<f32>()
                .clamp(0.0, 1.0);
            [dot; 3]
        }
        Operation::MultiplyAdd => {
            std::array::from_fn(|i| (v[0][i] * v[1][i] + v[2][i]).min(1.0))
        }
        Operation::AddMultiply => {
            std::array::from_fn(|i| ((v[0][i] + v[1][i]).min(1.0)) * v[2][i])
        }
    }
}

fn combine_alpha(op: Operation, v: [f32; 3]) -> f32 {
    match op {
        Operation::Replace => v[0],
        Operation::Modulate => v[0] * v[1],
        Operation::Add => (v[0] + v[1]).min(1.0),
        Operation::AddSigned => (v[0] + v[1] - 0.5).clamp(0.0, 1.0),
        Operation::Lerp => v[0] * v[2] + v[1] * (1.0 - v[2]),
        Operation::Subtract => (v[0] - v[1]).max(0.0),
        Operation::Dot3Rgb | Operation::Dot3Rgba => v[0],
        Operation::MultiplyAdd => (v[0] * v[1] + v[2]).min(1.0),
        Operation::AddMultiply => (v[0] + v[1]).min(1.0) * v[2],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// registers holding a single stage that just modulates texture 0 with
    /// the vertex color, the configuration the power-on default describes
    /// and what the rasterizer used to assume unconditionally.
    fn modulate_registers() -> Vec<u32> {
        let mut registers = vec![0u32; 0x300];
        for (index, base) in STAGE_REGISTERS.iter().enumerate() {
            // sources, primary color and texture 0 for the first stage,
            // previous for the rest, so later stages pass the value along.
            registers[*base] = if index == 0 { 0x0003_0003 } else { 0x000F_000F };
            registers[base + 2] = if index == 0 { 0x0001_0001 } else { 0x0000_0000 };
        }
        registers
    }

    #[test]
    fn a_modulate_stage_multiplies_its_inputs() {
        let env = TexEnv::read(&modulate_registers());
        let out = env.apply([1.0, 0.5, 0.0, 1.0], [[0.5, 0.5, 0.5, 0.5]; 4], None);
        assert!((out[0] - 0.5).abs() < 1e-5, "red was {}", out[0]);
        assert!((out[1] - 0.25).abs() < 1e-5, "green was {}", out[1]);
        assert!((out[2] - 0.0).abs() < 1e-5, "blue was {}", out[2]);
        assert!((out[3] - 0.5).abs() < 1e-5, "alpha was {}", out[3]);
    }

    /// a stage set to Replace from the vertex color has to leave it
    /// untouched, that is what an untextured draw configures, and folding
    /// in a texture there would tint geometry that should be flat.
    #[test]
    fn a_replace_stage_passes_the_primary_color_through() {
        let mut registers = vec![0u32; 0x300];
        for base in STAGE_REGISTERS {
            registers[base] = 0x0000_0000; // primary color everywhere
            registers[base + 2] = 0x0000_0000; // replace
        }
        let env = TexEnv::read(&registers);
        let primary = [0.25, 0.5, 0.75, 1.0];
        let out = env.apply(primary, [[1.0, 0.0, 0.0, 1.0]; 4], None);
        for channel in 0..4 {
            assert!(
                (out[channel] - primary[channel]).abs() < 1e-5,
                "channel {channel} was {} not {}",
                out[channel],
                primary[channel]
            );
        }
    }

    #[test]
    fn the_output_scale_is_a_shift_not_a_factor() {
        assert_eq!(scale_factor(0), 1.0);
        assert_eq!(scale_factor(1), 2.0);
        assert_eq!(scale_factor(2), 4.0);
        assert_eq!(scale_factor(3), 1.0);
    }

    /// registers with every stage passing the previous result along.
    fn passthrough_registers() -> Vec<u32> {
        let mut registers = vec![0u32; 0x300];
        for base in STAGE_REGISTERS {
            registers[base] = 0x000F_000F;
        }
        registers
    }

    /// a lighting color only counts in an input the operation reads.
    #[test]
    fn lighting_is_read_only_through_used_inputs() {
        let mut registers = passthrough_registers();
        // texture 0, with the primary fragment color in the second input
        registers[STAGE_REGISTERS[0]] = 0x0003_0013;
        assert!(!TexEnv::read(&registers).reads_lighting(), "replace reads one input");
        registers[STAGE_REGISTERS[0] + 2] = 1;
        assert!(TexEnv::read(&registers).reads_lighting(), "modulate reads two");
    }

    #[test]
    fn pass_through_stages_leave_the_color_alone() {
        let env = TexEnv::read(&passthrough_registers());
        assert!(env.passthrough.iter().all(|&p| p));
        let color = [0.25, 0.5, 0.75, 1.0];
        assert_eq!(env.apply(color, [[0.0; 4]; 4], None), color);
    }

    /// stage 0 writes red into the buffer.
    #[test]
    fn the_buffer_reaches_the_stage_after_next() {
        let mut registers = passthrough_registers();
        let [stage0, stage1, stage2, ..] = STAGE_REGISTERS;
        registers[stage0] = 0x000E_000E; // constant
        registers[stage0 + 3] = 0xFF00_00FF; // opaque red
        registers[REG_UPDATE_BUFFER] = 0x1100; // stage 0, color and alpha
        registers[REG_BUFFER_COLOR] = 0xFF00_FF00; // opaque green
        registers[stage1] = 0x000D_000D; // previous buffer

        let green = [0.0, 1.0, 0.0, 1.0];
        let red = [1.0, 0.0, 0.0, 1.0];
        let env = TexEnv::read(&registers);
        assert_eq!(env.apply([0.0; 4], [[0.0; 4]; 4], None), green);

        registers[stage2] = 0x000D_000D;
        let env = TexEnv::read(&registers);
        assert_eq!(env.apply([0.0; 4], [[0.0; 4]; 4], None), red);
    }

    /// TexEnv::read through registers_read and decode, and the kept one,
    /// decode what reading the registers straight did before, over random
    /// words in every register it reads and random registers elsewhere.
    #[test]
    fn reading_through_the_words_read_decodes_as_before() {
        fn old_read(registers: &[u32]) -> TexEnv {
            let stages: [Stage; 6] = std::array::from_fn(|i| {
                let base = STAGE_REGISTERS[i];
                let source = registers[base];
                let operand = registers[base + 1];
                let combiner = registers[base + 2];
                let constant = registers[base + 3];
                let scale = registers[base + 4];
                Stage {
                    color_sources: [Source::from_raw(source), Source::from_raw(source >> 4), Source::from_raw(source >> 8)],
                    alpha_sources: [Source::from_raw(source >> 16), Source::from_raw(source >> 20), Source::from_raw(source >> 24)],
                    color_operands: [operand & 0xF, (operand >> 4) & 0xF, (operand >> 8) & 0xF],
                    alpha_operands: [(operand >> 12) & 0x7, (operand >> 16) & 0x7, (operand >> 20) & 0x7],
                    color_op: Operation::from_raw(combiner),
                    alpha_op: Operation::from_raw(combiner >> 16),
                    constant: unpack(constant),
                    color_scale: scale_factor(scale),
                    alpha_scale: scale_factor(scale >> 16),
                }
            });
            let update = registers[REG_UPDATE_BUFFER];
            TexEnv {
                stages,
                passthrough: stages.map(|stage| stage.is_passthrough()),
                update_color: std::array::from_fn(|i| update & (0x100 << i) != 0),
                update_alpha: std::array::from_fn(|i| update & (0x1000 << i) != 0),
                buffer_color: unpack(registers[REG_BUFFER_COLOR]),
            }
        }
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut random = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 16) as u32
        };
        let read: Vec<usize> = STAGE_REGISTERS.iter().flat_map(|&base| base..base + 5).chain([REG_UPDATE_BUFFER, REG_BUFFER_COLOR]).collect();
        let mut registers = vec![0u32; 0x300];
        let mut kept = LastTexEnv::default();
        for step in 0..50_000 {
            if step % 500 == 0 {
                registers.iter_mut().for_each(|r| *r = random());
            }
            for _ in 0..1 + random() % 3 {
                let register = match random() % 4 {
                    0 => (random() % 0x300) as usize,
                    _ => read[(random() as usize) % read.len()],
                };
                // small values now and then, the fields are a few bits each
                registers[register] = if random() % 2 == 0 { random() } else { random() & 0x00FF_00FF };
            }
            let old = old_read(&registers);
            assert_eq!(TexEnv::read(&registers), old, "step {step}");
            assert_eq!(kept.read(&registers), old, "step {step}, kept");
            assert_eq!(kept.read(&registers).reads_lighting(), old.reads_lighting(), "step {step}");
        }
    }

    /// the environment kept from the last draw is the one the registers
    /// set up, through changes to any register it is decoded from and to
    /// ones it is not.
    #[test]
    fn a_kept_environment_follows_the_registers() {
        let mut seed = 7u32;
        // the high bits, the low ones of the sequence repeat soon
        let mut random = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed >> 8
        };
        let mut registers = vec![0u32; 0x300];
        let mut kept = LastTexEnv::default();
        let read: Vec<usize> = STAGE_REGISTERS.iter().flat_map(|&base| base..base + 5).chain([REG_UPDATE_BUFFER, REG_BUFFER_COLOR]).collect();
        for step in 0..2000 {
            // mostly a register the environment reads, a bit at a time as
            // titles change them, or now and then one it does not
            let register = match random() % 8 {
                0 => (random() % 0x300) as usize,
                _ => read[(random() as usize) % read.len()],
            };
            registers[register] ^= 1 << (random() % 32);
            assert_eq!(kept.read(&registers), TexEnv::read(&registers), "step {step}, register {register:#X}");
            if step % 3 == 0 {
                assert_eq!(kept.read(&registers), TexEnv::read(&registers), "step {step} again");
            }
        }
    }

    /// color and alpha have separate update bits.
    #[test]
    fn color_and_alpha_update_the_buffer_separately() {
        let mut registers = passthrough_registers();
        let [stage0, _, stage2, ..] = STAGE_REGISTERS;
        registers[stage0] = 0x000E_000E;
        registers[stage0 + 3] = 0x8000_00FF; // red, half alpha
        registers[REG_UPDATE_BUFFER] = 0x0100; // stage 0, color only
        registers[REG_BUFFER_COLOR] = 0xFF00_FF00;
        registers[stage2] = 0x000D_000D;

        let out = TexEnv::read(&registers).apply([0.0; 4], [[0.0; 4]; 4], None);
        assert_eq!(&out[..3], &[1.0, 0.0, 0.0]);
        assert_eq!(out[3], 1.0, "alpha must still be the buffer color's");
    }
}
