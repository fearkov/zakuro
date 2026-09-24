//! the PICA200's output merger, how a fragment combines with what is already in
//! the color buffer.

use crate::registers::*;

/// register holding the constant color the Constant* factors read.
const REG_BLEND_COLOR: usize = 0x0103;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Equation {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

impl Equation {
    fn from_raw(value: u32) -> Equation {
        match value & 0x7 {
            1 => Equation::Subtract,
            2 => Equation::ReverseSubtract,
            3 => Equation::Min,
            4 => Equation::Max,
            // 0, and the undocumented 5-7, which hardware treats as add.
            _ => Equation::Add,
        }
    }

    fn apply(self, source: f32, source_factor: f32, dest: f32, dest_factor: f32) -> f32 {
        let s = source * source_factor;
        let d = dest * dest_factor;
        match self {
            Equation::Add => s + d,
            Equation::Subtract => s - d,
            Equation::ReverseSubtract => d - s,
            // min and max ignore the factors, as they do in OpenGL.
            Equation::Min => source.min(dest),
            Equation::Max => source.max(dest),
        }
        .clamp(0.0, 1.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Factor {
    Zero,
    One,
    SourceColor,
    OneMinusSourceColor,
    DestColor,
    OneMinusDestColor,
    SourceAlpha,
    OneMinusSourceAlpha,
    DestAlpha,
    OneMinusDestAlpha,
    ConstantColor,
    OneMinusConstantColor,
    ConstantAlpha,
    OneMinusConstantAlpha,
    SourceAlphaSaturate,
}

impl Factor {
    fn from_raw(value: u32) -> Factor {
        match value & 0xF {
            0 => Factor::Zero,
            1 => Factor::One,
            2 => Factor::SourceColor,
            3 => Factor::OneMinusSourceColor,
            4 => Factor::DestColor,
            5 => Factor::OneMinusDestColor,
            6 => Factor::SourceAlpha,
            7 => Factor::OneMinusSourceAlpha,
            8 => Factor::DestAlpha,
            9 => Factor::OneMinusDestAlpha,
            10 => Factor::ConstantColor,
            11 => Factor::OneMinusConstantColor,
            12 => Factor::ConstantAlpha,
            13 => Factor::OneMinusConstantAlpha,
            14 => Factor::SourceAlphaSaturate,
            // 15 is undocumented, Citra treats it as one.
            _ => Factor::One,
        }
    }

    /// the factor's value for one channel.
    fn value(
        self,
        channel: usize,
        source: [f32; 4],
        dest: [f32; 4],
        constant: [f32; 4],
    ) -> f32 {
        match self {
            Factor::Zero => 0.0,
            Factor::One => 1.0,
            Factor::SourceColor => source[channel],
            Factor::OneMinusSourceColor => 1.0 - source[channel],
            Factor::DestColor => dest[channel],
            Factor::OneMinusDestColor => 1.0 - dest[channel],
            Factor::SourceAlpha => source[3],
            Factor::OneMinusSourceAlpha => 1.0 - source[3],
            Factor::DestAlpha => dest[3],
            Factor::OneMinusDestAlpha => 1.0 - dest[3],
            Factor::ConstantColor => constant[channel],
            Factor::OneMinusConstantColor => 1.0 - constant[channel],
            Factor::ConstantAlpha => constant[3],
            Factor::OneMinusConstantAlpha => 1.0 - constant[3],
            Factor::SourceAlphaSaturate => {
                if channel == 3 {
                    1.0
                } else {
                    source[3].min(1.0 - dest[3])
                }
            }
        }
    }
}

/// a decoded blend configuration.
#[derive(Debug, Clone, Copy)]
pub struct Blend {
    color_equation: Equation,
    alpha_equation: Equation,
    color_source: Factor,
    color_dest: Factor,
    alpha_source: Factor,
    alpha_dest: Factor,
    constant: [f32; 4],
}

impl Blend {
    /// reads the blend configuration, or None when the output merger is
    /// in logic-op mode rather than blending.
    pub fn read(registers: &[u32]) -> Option<Blend> {
        // bit 8 of the color operation register selects blending over the
        // logic op.
        if registers[REG_COLOR_OPERATION] & 0x100 == 0 {
            return None;
        }
        let config = registers[REG_BLEND_FUNC];
        let constant = registers[REG_BLEND_COLOR];
        Some(Blend {
            color_equation: Equation::from_raw(config),
            alpha_equation: Equation::from_raw(config >> 8),
            color_source: Factor::from_raw(config >> 16),
            color_dest: Factor::from_raw(config >> 20),
            alpha_source: Factor::from_raw(config >> 24),
            alpha_dest: Factor::from_raw(config >> 28),
            constant: [
                (constant & 0xFF) as f32 / 255.0,
                ((constant >> 8) & 0xFF) as f32 / 255.0,
                ((constant >> 16) & 0xFF) as f32 / 255.0,
                ((constant >> 24) & 0xFF) as f32 / 255.0,
            ],
        })
    }

    /// combines a fragment with the pixel already in the buffer.
    pub fn apply(&self, source: [f32; 4], dest: [f32; 4]) -> [f32; 4] {
        let mut out = [0.0; 4];
        for (channel, value) in out.iter_mut().enumerate().take(3) {
            *value = self.color_equation.apply(
                source[channel],
                self.color_source.value(channel, source, dest, self.constant),
                dest[channel],
                self.color_dest.value(channel, source, dest, self.constant),
            );
        }
        out[3] = self.alpha_equation.apply(
            source[3],
            self.alpha_source.value(3, source, dest, self.constant),
            dest[3],
            self.alpha_dest.value(3, source, dest, self.constant),
        );
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registers(blend_func: u32) -> Vec<u32> {
        let mut registers = vec![0u32; 0x300];
        registers[REG_COLOR_OPERATION] = 0x100;
        registers[REG_BLEND_FUNC] = blend_func;
        registers
    }

    /// ordinary transparency, add, source alpha, one minus source alpha,
    /// what almost every UI draw configures.
    #[test]
    fn standard_transparency_mixes_by_source_alpha() {
        // equation add (0) for both, src = SrcAlpha (6), dst = 1-SrcAlpha (7)
        let blend = Blend::read(&registers(0x7676_0000)).unwrap();
        let out = blend.apply([1.0, 0.0, 0.0, 0.25], [0.0, 0.0, 1.0, 1.0]);
        assert!((out[0] - 0.25).abs() < 1e-5, "red {}", out[0]);
        assert!((out[2] - 0.75).abs() < 1e-5, "blue {}", out[2]);
    }

    /// additive, source and destination both at factor one.
    #[test]
    fn additive_blending_brightens() {
        let blend = Blend::read(&registers(0x1111_0000)).unwrap();
        let out = blend.apply([0.25, 0.25, 0.25, 1.0], [0.5, 0.5, 0.5, 1.0]);
        assert!((out[0] - 0.75).abs() < 1e-5, "red {}", out[0]);
    }

    #[test]
    fn logic_op_mode_is_not_blending() {
        let mut registers = registers(0);
        registers[REG_COLOR_OPERATION] = 0;
        assert!(Blend::read(&registers).is_none());
    }
}
