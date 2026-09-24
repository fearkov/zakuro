//! decoding of the PICA200 shader instruction set.

/// the subset of the instruction set a vertex shader actually uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpCode {
    Add,
    Dp3,
    Dp4,
    Dph,
    Dst,
    Ex2,
    Lg2,
    LitP,
    Mul,
    Sge,
    Slt,
    Flr,
    Max,
    Min,
    Rcp,
    Rsq,
    Mova,
    Mov,
    DphI,
    DstI,
    SgeI,
    SltI,
    Break,
    Nop,
    End,
    BreakC,
    Call,
    CallC,
    CallU,
    IfU,
    IfC,
    Loop,
    Emit,
    SetEmit,
    JmpC,
    JmpU,
    Cmp,
    /// multiply-add whose *third* source is the wide one.
    MadI,
    Mad,
    /// anything we do not model.
    Unknown(u8),
}

impl OpCode {
    /// whether this is one of the inverted encodings, whose wide operand
    /// (the one that can be a uniform) is the second source.
    pub fn is_inverted(self) -> bool {
        matches!(self, OpCode::DphI | OpCode::DstI | OpCode::SgeI | OpCode::SltI)
    }

    fn from_raw(value: u8) -> OpCode {
        match value {
            0x00 => OpCode::Add,
            0x01 => OpCode::Dp3,
            0x02 => OpCode::Dp4,
            0x03 => OpCode::Dph,
            0x04 => OpCode::Dst,
            0x05 => OpCode::Ex2,
            0x06 => OpCode::Lg2,
            0x07 => OpCode::LitP,
            0x08 => OpCode::Mul,
            0x09 => OpCode::Sge,
            0x0A => OpCode::Slt,
            0x0B => OpCode::Flr,
            0x0C => OpCode::Max,
            0x0D => OpCode::Min,
            0x0E => OpCode::Rcp,
            0x0F => OpCode::Rsq,
            0x12 => OpCode::Mova,
            0x13 => OpCode::Mov,
            0x18 => OpCode::DphI,
            0x19 => OpCode::DstI,
            0x1A => OpCode::SgeI,
            0x1B => OpCode::SltI,
            0x20 => OpCode::Break,
            0x21 => OpCode::Nop,
            0x22 => OpCode::End,
            0x23 => OpCode::BreakC,
            0x24 => OpCode::Call,
            0x25 => OpCode::CallC,
            0x26 => OpCode::CallU,
            0x27 => OpCode::IfU,
            0x28 => OpCode::IfC,
            0x29 => OpCode::Loop,
            0x2A => OpCode::Emit,
            0x2B => OpCode::SetEmit,
            0x2C => OpCode::JmpC,
            0x2D => OpCode::JmpU,
            // the low opcode bit belongs to the first comparison mode.
            0x2E..=0x2F => OpCode::Cmp,
            // multiply-add only uses the top three opcode bits, the rest
            // of the field holds the destination register.
            0x30..=0x37 => OpCode::MadI,
            0x38..=0x3F => OpCode::Mad,
            other => OpCode::Unknown(other),
        }
    }
}

/// a raw instruction word.
#[derive(Debug, Clone, Copy)]
pub struct Instruction(pub u32);

impl Instruction {
    pub fn opcode(self) -> OpCode {
        OpCode::from_raw((self.0 >> 26) as u8)
    }

    /// index into the operand descriptor table for an arithmetic instruction.
    pub fn descriptor_index(self) -> u32 {
        self.0 & 0x7F
    }

    /// (src1, src2) register numbers.
    pub fn sources(self) -> (u32, u32) {
        if self.opcode().is_inverted() {
            ((self.0 >> 14) & 0x1F, (self.0 >> 7) & 0x7F)
        } else {
            ((self.0 >> 12) & 0x7F, (self.0 >> 7) & 0x1F)
        }
    }

    /// which address register indexes the wide source operand, src1
    /// normally, src2 for the inverted forms, or zero for none.
    pub fn address_register_index(self) -> u32 {
        (self.0 >> 19) & 0x3
    }

    pub fn destination(self) -> u32 {
        (self.0 >> 21) & 0x1F
    }

    // -- multiply-add, which packs its fields differently ------------------

    pub fn mad_descriptor_index(self) -> u32 {
        self.0 & 0x1F
    }

    /// (src1, src2, src3) register numbers.
    pub fn mad_sources(self) -> (u32, u32, u32) {
        let src1 = (self.0 >> 17) & 0x1F;
        if self.opcode() == OpCode::MadI {
            (src1, (self.0 >> 12) & 0x1F, (self.0 >> 5) & 0x7F)
        } else {
            (src1, (self.0 >> 10) & 0x7F, (self.0 >> 5) & 0x1F)
        }
    }

    /// which address register indexes the wide source of a multiply-add.
    pub fn mad_address_register_index(self) -> u32 {
        (self.0 >> 22) & 0x3
    }

    pub fn mad_destination(self) -> u32 {
        (self.0 >> 24) & 0x1F
    }

    // -- geometry shader output ---------------------------------------------

    /// setemit, which of the three vertex slots the next emit fills.
    pub fn emit_vertex(self) -> usize {
        ((self.0 >> 24) & 0x3) as usize
    }

    /// setemit, whether the next emit completes a triangle.
    pub fn emit_primitive(self) -> bool {
        (self.0 >> 23) & 1 != 0
    }

    // -- flow control ------------------------------------------------------

    /// (destination, instruction count).
    pub fn flow_target(self) -> (u32, u32) {
        (((self.0 >> 10) & 0xFFF), self.0 & 0xFF)
    }

    pub fn bool_index(self) -> u32 {
        (self.0 >> 22) & 0xF
    }

    pub fn integer_index(self) -> u32 {
        (self.0 >> 22) & 0x3
    }

    pub fn raw_condition(self) -> u32 {
        (self.0 >> 22) & 0x3
    }

    /// which of x and y the condition compares, and how they combine.
    pub fn condition_op(self) -> u32 {
        (self.0 >> 22) & 0x3
    }

    /// the values the condition registers are compared against.
    pub fn condition_reference(self) -> [bool; 2] {
        [(self.0 >> 25) & 1 != 0, (self.0 >> 24) & 1 != 0]
    }

    /// the two comparison modes a cmp applies to x and y.
    pub fn compare_modes(self) -> [u32; 2] {
        [(self.0 >> 24) & 0x7, (self.0 >> 21) & 0x7]
    }
}

/// supplies swizzles, negation and the write mask for an instruction.
#[derive(Debug, Clone, Copy)]
pub struct OperandDescriptor(pub u32);

impl OperandDescriptor {
    /// which destination components the instruction writes.
    pub fn destination_mask(self) -> u32 {
        self.0 & 0xF
    }

    pub fn apply_source1(self, value: [f32; 4]) -> [f32; 4] {
        swizzle(value, (self.0 >> 5) & 0xFF, (self.0 >> 4) & 1 != 0)
    }

    pub fn apply_source2(self, value: [f32; 4]) -> [f32; 4] {
        swizzle(value, (self.0 >> 14) & 0xFF, (self.0 >> 13) & 1 != 0)
    }

    pub fn apply_source3(self, value: [f32; 4]) -> [f32; 4] {
        swizzle(value, (self.0 >> 23) & 0xFF, (self.0 >> 22) & 1 != 0)
    }
}

/// applies a swizzle pattern and optional negation.
fn swizzle(value: [f32; 4], pattern: u32, negate: bool) -> [f32; 4] {
    let pick = |component: u32| value[((pattern >> (6 - component * 2)) & 3) as usize];
    let mut out = [pick(0), pick(1), pick(2), pick(3)];
    if negate {
        for component in &mut out {
            *component = -*component;
        }
    }
    out
}

/// decodes the 24-bit float format the shader uses for its uniforms.
pub fn decode_float24(raw: u32) -> f32 {
    let raw = raw & 0x00FF_FFFF;
    if raw == 0 {
        return 0.0;
    }
    let sign = (raw >> 23) & 1;
    let exponent = (raw >> 16) & 0x7F;
    let mantissa = raw & 0xFFFF;

    if exponent == 0 {
        // subnormal, the shader treats these as zero.
        return if sign != 0 { -0.0 } else { 0.0 };
    }
    if exponent == 0x7F {
        return if sign != 0 {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
    }

    let bits = (sign << 31) | ((exponent + 64) << 23) | (mantissa << 7);
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float24_decodes_simple_values() {
        // 1.0 is exponent 63 (the bias) with a zero mantissa.
        assert_eq!(decode_float24(63 << 16), 1.0);
        // 2.0 raises the exponent by one.
        assert_eq!(decode_float24(64 << 16), 2.0);
        // 0.5 lowers it by one.
        assert_eq!(decode_float24(62 << 16), 0.5);
        // the sign bit is the top bit of the 24.
        assert_eq!(decode_float24((1 << 23) | (63 << 16)), -1.0);
        assert_eq!(decode_float24(0), 0.0);
    }

    #[test]
    fn swizzle_takes_x_from_the_top_pair() {
        let value = [1.0, 2.0, 3.0, 4.0];
        // 0b00_01_10_11 selects x, y, z, w in order.
        assert_eq!(swizzle(value, 0b00_01_10_11, false), [1.0, 2.0, 3.0, 4.0]);
        // broadcasting x to every component.
        assert_eq!(swizzle(value, 0b00_00_00_00, false), [1.0; 4]);
        // reversing.
        assert_eq!(swizzle(value, 0b11_10_01_00, false), [4.0, 3.0, 2.0, 1.0]);
        // negation applies after the swizzle.
        assert_eq!(swizzle(value, 0b00_00_00_00, true), [-1.0; 4]);
    }

    #[test]
    fn decodes_an_instruction_the_way_hardware_does() {
        // mov o0, v0 as the assembler emits it, opcode 0x13, destination 0,
        // source 0, descriptor 0.
        let instruction = Instruction(0x13 << 26);
        assert_eq!(instruction.opcode(), OpCode::Mov);
        assert_eq!(instruction.destination(), 0);
        assert_eq!(instruction.sources().0, 0);
    }
}
