//! the ARM barrel shifter, shared by data processing and register-offset
//! addressing modes.

use zakuro_common::bits::bits;

use crate::{Cpu, PC};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShiftType {
    Lsl,
    Lsr,
    Asr,
    Ror,
}

impl ShiftType {
    #[inline(always)]
    pub fn from_bits(value: u32) -> ShiftType {
        match value & 3 {
            0 => ShiftType::Lsl,
            1 => ShiftType::Lsr,
            2 => ShiftType::Asr,
            _ => ShiftType::Ror,
        }
    }
}

/// applies a shift whose amount came from an immediate field.
#[inline(always)]
pub fn shift_by_immediate(ty: ShiftType, value: u32, amount: u32, carry_in: bool) -> (u32, bool) {
    match ty {
        ShiftType::Lsl => {
            if amount == 0 {
                (value, carry_in)
            } else {
                (value << amount, (value >> (32 - amount)) & 1 != 0)
            }
        }
        ShiftType::Lsr => {
            // LSR #0 encodes LSR #32.
            if amount == 0 {
                (0, value >> 31 != 0)
            } else {
                (value >> amount, (value >> (amount - 1)) & 1 != 0)
            }
        }
        ShiftType::Asr => {
            // ASR #0 encodes ASR #32.
            if amount == 0 {
                let sign = (value as i32) >> 31;
                (sign as u32, value >> 31 != 0)
            } else {
                (
                    ((value as i32) >> amount) as u32,
                    ((value as i32) >> (amount - 1)) & 1 != 0,
                )
            }
        }
        ShiftType::Ror => {
            // ROR #0 encodes RRX, a 33-bit rotate through the carry flag.
            if amount == 0 {
                let result = ((carry_in as u32) << 31) | (value >> 1);
                (result, value & 1 != 0)
            } else {
                (value.rotate_right(amount), (value >> (amount - 1)) & 1 != 0)
            }
        }
    }
}

/// applies a shift whose amount came from the bottom byte of a register.
#[inline(always)]
pub fn shift_by_register(ty: ShiftType, value: u32, amount: u32, carry_in: bool) -> (u32, bool) {
    let amount = amount & 0xFF;
    if amount == 0 {
        return (value, carry_in);
    }
    match ty {
        ShiftType::Lsl => match amount {
            1..=31 => (value << amount, (value >> (32 - amount)) & 1 != 0),
            32 => (0, value & 1 != 0),
            _ => (0, false),
        },
        ShiftType::Lsr => match amount {
            1..=31 => (value >> amount, (value >> (amount - 1)) & 1 != 0),
            32 => (0, value >> 31 != 0),
            _ => (0, false),
        },
        ShiftType::Asr => {
            if amount >= 32 {
                let sign = (value as i32) >> 31;
                (sign as u32, value >> 31 != 0)
            } else {
                (
                    ((value as i32) >> amount) as u32,
                    ((value as i32) >> (amount - 1)) & 1 != 0,
                )
            }
        }
        ShiftType::Ror => {
            let rotate = amount & 31;
            if rotate == 0 {
                // a multiple of 32, the value is unchanged but carry comes
                // from the top bit.
                (value, value >> 31 != 0)
            } else {
                (value.rotate_right(rotate), (value >> (rotate - 1)) & 1 != 0)
            }
        }
    }
}

/// decodes the "shifter operand" of a data-processing instruction, returning
/// the operand and the carry it produces.
#[inline(always)]
pub fn decode_operand(cpu: &Cpu, op: u32) -> (u32, bool) {
    let carry_in = cpu.cpsr.c;

    if op & (1 << 25) != 0 {
        // immediate, an 8-bit value rotated right by twice a 4-bit field.
        let imm = op & 0xFF;
        let rotate = bits(op, 8, 11) * 2;
        if rotate == 0 {
            (imm, carry_in)
        } else {
            let value = imm.rotate_right(rotate);
            (value, value >> 31 != 0)
        }
    } else {
        let rm = (op & 0xF) as usize;
        let ty = ShiftType::from_bits(bits(op, 5, 6));

        if op & (1 << 4) == 0 {
            let amount = bits(op, 7, 11);
            shift_by_immediate(ty, cpu.regs[rm], amount, carry_in)
        } else {
            // register-specified shift.
            let rs = bits(op, 8, 11) as usize;
            let value = if rm == PC {
                cpu.regs[PC].wrapping_add(4)
            } else {
                cpu.regs[rm]
            };
            let amount = if rs == PC {
                cpu.regs[PC].wrapping_add(4)
            } else {
                cpu.regs[rs]
            };
            shift_by_register(ty, value, amount, carry_in)
        }
    }
}
