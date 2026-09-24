//! data-processing instructions, the AND/SUB/ADD/MOV family.

use zakuro_common::bits::bits;

use super::alu::*;
use super::shifter;
use crate::{Bus, Cpu, Exit, PC};

/// reads Rn, accounting for the extra pipeline step a register-specified
/// shift introduces.
#[inline(always)]
fn read_rn(cpu: &Cpu, op: u32) -> u32 {
    let rn = bits(op, 16, 19) as usize;
    if rn == PC && op & (1 << 25) == 0 && op & (1 << 4) != 0 {
        cpu.regs[PC].wrapping_add(4)
    } else {
        cpu.regs[rn]
    }
}

pub fn execute<B: Bus>(cpu: &mut Cpu, _bus: &mut B, op: u32) -> Option<Exit> {
    let opcode = bits(op, 21, 24);
    let set_flags = op & (1 << 20) != 0;
    let rd = bits(op, 12, 15) as usize;

    let (operand, shifter_carry) = shifter::decode_operand(cpu, op);
    let a = read_rn(cpu, op);
    let carry_in = cpu.cpsr.c;

    // comparison opcodes (TST/TEQ/CMP/CMN) never write a register.
    let is_compare = (0b1000..=0b1011).contains(&opcode);

    let (result, carry, overflow, logical) = match opcode {
        0b0000 => (a & operand, shifter_carry, false, true),          // AND
        0b0001 => (a ^ operand, shifter_carry, false, true),          // EOR
        0b0010 => {
            let (r, c, v) = sub_with_flags(a, operand);
            (r, c, v, false)
        } // SUB
        0b0011 => {
            let (r, c, v) = sub_with_flags(operand, a);
            (r, c, v, false)
        } // RSB
        0b0100 => {
            let (r, c, v) = add_with_flags(a, operand);
            (r, c, v, false)
        } // ADD
        0b0101 => {
            let (r, c, v) = adc_with_flags(a, operand, carry_in);
            (r, c, v, false)
        } // ADC
        0b0110 => {
            let (r, c, v) = sbc_with_flags(a, operand, carry_in);
            (r, c, v, false)
        } // SBC
        0b0111 => {
            let (r, c, v) = sbc_with_flags(operand, a, carry_in);
            (r, c, v, false)
        } // RSC
        0b1000 => (a & operand, shifter_carry, false, true),          // TST
        0b1001 => (a ^ operand, shifter_carry, false, true),          // TEQ
        0b1010 => {
            let (r, c, v) = sub_with_flags(a, operand);
            (r, c, v, false)
        } // CMP
        0b1011 => {
            let (r, c, v) = add_with_flags(a, operand);
            (r, c, v, false)
        } // CMN
        0b1100 => (a | operand, shifter_carry, false, true),          // ORR
        0b1101 => (operand, shifter_carry, false, true),              // MOV
        0b1110 => (a & !operand, shifter_carry, false, true),         // BIC
        _ => (!operand, shifter_carry, false, true),                  // MVN
    };

    if set_flags {
        if rd == PC && !is_compare {
            // <op>S pc, ... is an exception return, CPSR comes from SPSR.
            cpu.restore_cpsr();
            cpu.set_reg(PC, result);
            return None;
        }
        cpu.set_nz(result);
        cpu.cpsr.c = carry;
        if !logical {
            cpu.cpsr.v = overflow;
        }
    }

    if !is_compare {
        cpu.set_reg(rd, result);
    }
    None
}
