//! multiplies, the plain 32- and 64-bit forms, the ARMv5TE DSP halfword
//! multiplies, and the saturating add/subtract family that shares their
//! encoding space.

use zakuro_common::bits::{bit, bits};

use super::alu::saturate_i32;
use crate::{Bus, Cpu, Exit};

/// MUL / MLA, note that Rd and Rn swap places relative to data processing.
pub fn short<B: Bus>(cpu: &mut Cpu, _bus: &mut B, op: u32) -> Option<Exit> {
    let rd = bits(op, 16, 19) as usize;
    let rn = bits(op, 12, 15) as usize;
    let rs = bits(op, 8, 11) as usize;
    let rm = (op & 0xF) as usize;
    let accumulate = bit(op, 21);
    let set_flags = bit(op, 20);

    let mut result = cpu.regs[rm].wrapping_mul(cpu.regs[rs]);
    if accumulate {
        result = result.wrapping_add(cpu.regs[rn]);
    }
    cpu.regs[rd] = result;

    if set_flags {
        cpu.set_nz(result);
        // the carry flag is architecturally UNPREDICTABLE here, leaving it
        // alone is what ARM11 does.
    }
    None
}

/// UMULL / UMLAL / SMULL / SMLAL / UMAAL.
pub fn long<B: Bus>(cpu: &mut Cpu, _bus: &mut B, op: u32) -> Option<Exit> {
    let rd_hi = bits(op, 16, 19) as usize;
    let rd_lo = bits(op, 12, 15) as usize;
    let rs = bits(op, 8, 11) as usize;
    let rm = (op & 0xF) as usize;
    let signed = bit(op, 22);
    let accumulate = bit(op, 21);
    let set_flags = bit(op, 20);

    // UMAAL is encoded as the otherwise-invalid signed=0, accumulate=0,
    // set_flags=0 slot one bit above the normal long multiplies.
    if bits(op, 21, 23) == 0b010 {
        let product = cpu.regs[rm] as u64 * cpu.regs[rs] as u64;
        let sum = product + cpu.regs[rd_lo] as u64 + cpu.regs[rd_hi] as u64;
        cpu.regs[rd_lo] = sum as u32;
        cpu.regs[rd_hi] = (sum >> 32) as u32;
        return None;
    }

    let result: u64 = if signed {
        let product = (cpu.regs[rm] as i32 as i64) * (cpu.regs[rs] as i32 as i64);
        let acc = if accumulate {
            ((cpu.regs[rd_hi] as u64) << 32 | cpu.regs[rd_lo] as u64) as i64
        } else {
            0
        };
        product.wrapping_add(acc) as u64
    } else {
        let product = cpu.regs[rm] as u64 * cpu.regs[rs] as u64;
        let acc = if accumulate {
            (cpu.regs[rd_hi] as u64) << 32 | cpu.regs[rd_lo] as u64
        } else {
            0
        };
        product.wrapping_add(acc)
    };

    cpu.regs[rd_lo] = result as u32;
    cpu.regs[rd_hi] = (result >> 32) as u32;

    if set_flags {
        cpu.cpsr.n = result >> 63 != 0;
        cpu.cpsr.z = result == 0;
    }
    None
}

/// the ARMv5TE halfword multiplies, SMLAxy, SMLAWy, SMULWy, SMLALxy
/// and SMULxy. x and y pick the bottom or top halfword of each operand.
pub fn halfword<B: Bus>(cpu: &mut Cpu, _bus: &mut B, op: u32) -> Option<Exit> {
    let rd = bits(op, 16, 19) as usize;
    let rn = bits(op, 12, 15) as usize;
    let rs = bits(op, 8, 11) as usize;
    let rm = (op & 0xF) as usize;
    let x = bit(op, 5);
    let y = bit(op, 6);

    let half = |value: u32, top: bool| -> i32 {
        if top {
            (value >> 16) as i16 as i32
        } else {
            value as i16 as i32
        }
    };

    match bits(op, 21, 22) {
        // SMLA<x><y>, 16x16 + 32. the sum wraps, an overflow only sets Q.
        0b00 => {
            let product = half(cpu.regs[rm], x) as i64 * half(cpu.regs[rs], y) as i64;
            let sum = product + cpu.regs[rn] as i32 as i64;
            cpu.regs[rd] = sum as u32;
            cpu.cpsr.q |= sum != sum as i32 as i64;
        }
        // SMLAW<y> / SMULW<y>, 32x16 keeping the top 32 bits of the 48-bit
        // product.
        0b01 => {
            let product = (cpu.regs[rm] as i32 as i64 * half(cpu.regs[rs], y) as i64) >> 16;
            if x {
                // SMULW<y>, no accumulate.
                cpu.regs[rd] = product as u32;
            } else {
                let sum = product + cpu.regs[rn] as i32 as i64;
                cpu.regs[rd] = sum as u32;
                cpu.cpsr.q |= sum != sum as i32 as i64;
            }
        }
        // SMLAL<x><y>, 16x16 accumulated into a 64-bit pair, no saturation.
        0b10 => {
            let product = half(cpu.regs[rm], x) as i64 * half(cpu.regs[rs], y) as i64;
            let acc = ((cpu.regs[rd] as u64) << 32 | cpu.regs[rn] as u64) as i64;
            let sum = acc.wrapping_add(product) as u64;
            cpu.regs[rn] = sum as u32;
            cpu.regs[rd] = (sum >> 32) as u32;
        }
        // SMUL<x><y>.
        _ => {
            let product = half(cpu.regs[rm], x) as i64 * half(cpu.regs[rs], y) as i64;
            cpu.regs[rd] = product as u32;
        }
    }
    None
}

/// QADD / QSUB / QDADD / QDSUB.
pub fn saturating<B: Bus>(cpu: &mut Cpu, _bus: &mut B, op: u32) -> Option<Exit> {
    let rd = bits(op, 12, 15) as usize;
    let rn = bits(op, 16, 19) as usize;
    let rm = (op & 0xF) as usize;
    let a = cpu.regs[rm] as i32 as i64;
    let b = cpu.regs[rn] as i32 as i64;

    let (value, saturated) = match bits(op, 21, 22) {
        0b00 => saturate_i32(a + b),
        0b01 => saturate_i32(a - b),
        0b10 => {
            // QDADD doubles Rn first, saturating that step too.
            let (doubled, sat1) = saturate_i32(b * 2);
            let (result, sat2) = saturate_i32(a + doubled as i32 as i64);
            (result, sat1 || sat2)
        }
        _ => {
            let (doubled, sat1) = saturate_i32(b * 2);
            let (result, sat2) = saturate_i32(a - doubled as i32 as i64);
            (result, sat1 || sat2)
        }
    };

    cpu.regs[rd] = value;
    cpu.cpsr.q |= saturated;
    None
}
