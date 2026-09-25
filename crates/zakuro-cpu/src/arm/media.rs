//! ARMv6 media instructions, parallel arithmetic, packing, saturation,
//! extension, byte reversal and the dual-multiply family.

use zakuro_common::bits::{bit, bits};

use super::alu::{signed_saturate, unsigned_saturate};
use crate::{Bus, Cpu, Exit};

pub fn execute<B: Bus>(cpu: &mut Cpu, _bus: &mut B, op: u32) -> Option<Exit> {
    let op1 = bits(op, 20, 24);
    let op2 = bits(op, 5, 7);

    match op1 >> 3 {
        // 000xx signed and 001xx unsigned parallel add/subtract.
        0b00 => parallel(cpu, op, op1, op2),
        // 01xxx packing, saturation, extension and reversal.
        0b01 => pack_sat_extend(cpu, op, op1, op2),
        // 10xxx signed multiplies.
        0b10 => dual_multiply(cpu, op, op1),
        _ => {
            if op1 == 0b11000 && op2 == 0b000 {
                usad8(cpu, op)
            } else {
                undefined(cpu, op)
            }
        }
    }
}

fn undefined(cpu: &Cpu, op: u32) -> Option<Exit> {
    Some(Exit::Undefined {
        pc: cpu.current_pc(),
        opcode: op,
    })
}

// ---------------------------------------------------------------------------
// Parallel add/subtract
// ---------------------------------------------------------------------------

/// signed and unsigned halfword/byte lanes, in plain, saturating (Q/UQ)
/// and halving (SH/UH) flavours.
fn parallel(cpu: &mut Cpu, op: u32, op1: u32, op2: u32) -> Option<Exit> {
    let rd = bits(op, 12, 15) as usize;
    let rn = bits(op, 16, 19) as usize;
    let rm = (op & 0xF) as usize;
    let a = cpu.regs[rn];
    let b = cpu.regs[rm];

    let signed = op1 & 0b100 == 0;
    let (saturating, halving) = match op1 & 0b11 {
        0b01 => (false, false),
        0b10 => (true, false),
        0b11 => (false, true),
        _ => return undefined(cpu, op),
    };

    let mut ge = 0u8;
    let result = match op2 {
        // halfword lanes. ASX and SAX cross the operand halves.
        0b000..=0b011 => {
            let (a_lo, a_hi) = (a as u16, (a >> 16) as u16);
            let (b_lo, b_hi) = (b as u16, (b >> 16) as u16);
            let (b_for_lo, b_for_hi) = match op2 {
                0b001 | 0b010 => (b_hi, b_lo),
                _ => (b_lo, b_hi),
            };
            // ADD16, add both.
            let (sub_lo, sub_hi) = match op2 {
                0b000 => (false, false),
                0b001 => (true, false),
                0b010 => (false, true),
                _ => (true, true),
            };

            let (lo, lo_ge) = lane16(a_lo, b_for_lo, sub_lo, signed, saturating, halving);
            let (hi, hi_ge) = lane16(a_hi, b_for_hi, sub_hi, signed, saturating, halving);
            if lo_ge {
                ge |= 0b0011;
            }
            if hi_ge {
                ge |= 0b1100;
            }
            (lo as u32) | ((hi as u32) << 16)
        }
        // byte lanes, only ADD8 and SUB8 exist.
        0b100 | 0b111 => {
            let sub = op2 == 0b111;
            let mut out = 0u32;
            for i in 0..4 {
                let av = (a >> (i * 8)) as u8;
                let bv = (b >> (i * 8)) as u8;
                let (value, ge_bit) = lane8(av, bv, sub, signed, saturating, halving);
                out |= (value as u32) << (i * 8);
                if ge_bit {
                    ge |= 1 << i;
                }
            }
            out
        }
        _ => return undefined(cpu, op),
    };

    cpu.regs[rd] = result;
    // only the plain forms update the GE flags, Q and H variants leave them.
    if !saturating && !halving {
        cpu.cpsr.ge = ge;
    }
    None
}

fn lane16(a: u16, b: u16, sub: bool, signed: bool, saturating: bool, halving: bool) -> (u16, bool) {
    let (wide, ge) = if signed {
        let r = if sub {
            a as i16 as i32 - b as i16 as i32
        } else {
            a as i16 as i32 + b as i16 as i32
        };
        (r, r >= 0)
    } else {
        let r = if sub {
            a as i32 - b as i32
        } else {
            a as i32 + b as i32
        };
        // unsigned GE means "no borrow" for a subtract and "carry out" for an
        // add.
        let ge = if sub { r >= 0 } else { r > 0xFFFF };
        (r, ge)
    };

    let value = if halving {
        (wide >> 1) as u16
    } else if saturating {
        if signed {
            signed_saturate(wide as i64, 16).0 as u16
        } else {
            unsigned_saturate(wide as i64, 16).0 as u16
        }
    } else {
        wide as u16
    };
    (value, ge)
}

fn lane8(a: u8, b: u8, sub: bool, signed: bool, saturating: bool, halving: bool) -> (u8, bool) {
    let (wide, ge) = if signed {
        let r = if sub {
            a as i8 as i32 - b as i8 as i32
        } else {
            a as i8 as i32 + b as i8 as i32
        };
        (r, r >= 0)
    } else {
        let r = if sub {
            a as i32 - b as i32
        } else {
            a as i32 + b as i32
        };
        let ge = if sub { r >= 0 } else { r > 0xFF };
        (r, ge)
    };

    let value = if halving {
        (wide >> 1) as u8
    } else if saturating {
        if signed {
            signed_saturate(wide as i64, 8).0 as u8
        } else {
            unsigned_saturate(wide as i64, 8).0 as u8
        }
    } else {
        wide as u8
    };
    (value, ge)
}

// ---------------------------------------------------------------------------
// Pack, saturate, extend, reverse, select
// ---------------------------------------------------------------------------

fn pack_sat_extend(cpu: &mut Cpu, op: u32, op1: u32, op2: u32) -> Option<Exit> {
    match (op1 & 0b111, op2) {
        (0b000, 0b000) | (0b000, 0b010) | (0b000, 0b100) | (0b000, 0b110) if op1 == 0b01000 => {
            pkh(cpu, op)
        }
        (_, 0b011) => extend(cpu, op, op1),
        (0b000, 0b101) if op1 == 0b01000 => select(cpu, op),
        (0b011, 0b001) if op1 == 0b01011 => {
            let (rd, rm) = dest_source(cpu, op);
            cpu.regs[rd] = rm.swap_bytes();
            None
        }
        (0b011, 0b101) if op1 == 0b01011 => {
            let (rd, rm) = dest_source(cpu, op);
            cpu.regs[rd] = ((rm & 0x00FF_00FF) << 8) | ((rm & 0xFF00_FF00) >> 8);
            None
        }
        // RBIT arrived with ARMv6T2, but implementing it costs nothing and
        // keeps hand-written assembly working.
        (0b111, 0b001) if op1 == 0b01111 => {
            let (rd, rm) = dest_source(cpu, op);
            cpu.regs[rd] = rm.reverse_bits();
            None
        }
        (0b111, 0b101) if op1 == 0b01111 => {
            let (rd, rm) = dest_source(cpu, op);
            let swapped = ((rm & 0xFF) << 8) | ((rm >> 8) & 0xFF);
            cpu.regs[rd] = swapped as u16 as i16 as i32 as u32;
            None
        }
        // SSAT16 / USAT16.
        (0b010, 0b001) | (0b110, 0b001) => saturate16(cpu, op, op1 & 0b100 != 0),
        // SSAT (op1 = 010xx) and USAT (op1 = 011xx), both with op2 = xx0.
        (_, o) if o & 1 == 0 && matches!(op1 >> 2 & 0b111, 0b010 | 0b011) => {
            saturate32(cpu, op, op1 & 0b00100 != 0)
        }
        _ => undefined(cpu, op),
    }
}

fn dest_source(cpu: &Cpu, op: u32) -> (usize, u32) {
    let rd = bits(op, 12, 15) as usize;
    let rm = cpu.regs[(op & 0xF) as usize];
    (rd, rm)
}

fn pkh(cpu: &mut Cpu, op: u32) -> Option<Exit> {
    let rd = bits(op, 12, 15) as usize;
    let a = cpu.regs[bits(op, 16, 19) as usize];
    let b = cpu.regs[(op & 0xF) as usize];
    let shift = bits(op, 7, 11);

    cpu.regs[rd] = if bit(op, 6) {
        // PKHTB, bottom half comes from Rm shifted right arithmetically.
        let amount = if shift == 0 { 32 } else { shift };
        let low = if amount >= 32 {
            ((b as i32) >> 31) as u32
        } else {
            ((b as i32) >> amount) as u32
        };
        (low & 0xFFFF) | (a & 0xFFFF_0000)
    } else {
        // PKHBT, top half comes from Rm shifted left.
        (a & 0xFFFF) | ((b << shift) & 0xFFFF_0000)
    };
    None
}

fn select(cpu: &mut Cpu, op: u32) -> Option<Exit> {
    let rd = bits(op, 12, 15) as usize;
    let a = cpu.regs[bits(op, 16, 19) as usize];
    let b = cpu.regs[(op & 0xF) as usize];
    let ge = cpu.cpsr.ge;

    let mut out = 0u32;
    for i in 0..4 {
        let byte = if ge & (1 << i) != 0 {
            (a >> (i * 8)) & 0xFF
        } else {
            (b >> (i * 8)) & 0xFF
        };
        out |= byte << (i * 8);
    }
    cpu.regs[rd] = out;
    None
}

/// the sign- and zero-extension family.
fn extend(cpu: &mut Cpu, op: u32, op1: u32) -> Option<Exit> {
    let rd = bits(op, 12, 15) as usize;
    let rn = bits(op, 16, 19) as usize;
    let rotated = cpu.regs[(op & 0xF) as usize].rotate_right(bits(op, 10, 11) * 8);
    let accumulate = rn != 15;
    let acc = cpu.regs[rn];

    // two of these operate on packed halfword pairs and so accumulate
    // lane-wise rather than as a single 32-bit add.
    let (value, packed) = match op1 & 0b111 {
        0b000 => {
            // SXTAB16 / SXTB16
            let lo = (rotated as u8 as i8 as i32) as u32 & 0xFFFF;
            let hi = ((rotated >> 16) as u8 as i8 as i32) as u32 & 0xFFFF;
            (lo | (hi << 16), true)
        }
        0b010 => (rotated as u8 as i8 as i32 as u32, false), // SXTAB / SXTB
        0b011 => (rotated as u16 as i16 as i32 as u32, false), // SXTAH / SXTH
        0b100 => (rotated & 0x00FF_00FF, true),             // UXTAB16 / UXTB16
        0b110 => (rotated & 0xFF, false),                   // UXTAB / UXTB
        0b111 => (rotated & 0xFFFF, false),                 // UXTAH / UXTH
        _ => return undefined(cpu, op),
    };

    cpu.regs[rd] = if !accumulate {
        value
    } else if packed {
        let lo = (acc as u16).wrapping_add(value as u16);
        let hi = ((acc >> 16) as u16).wrapping_add((value >> 16) as u16);
        lo as u32 | ((hi as u32) << 16)
    } else {
        acc.wrapping_add(value)
    };
    None
}

fn saturate32(cpu: &mut Cpu, op: u32, unsigned: bool) -> Option<Exit> {
    let rd = bits(op, 12, 15) as usize;
    let b = cpu.regs[(op & 0xF) as usize];
    // USAT saturates to sat_imm bits, SSAT to sat_imm + 1.
    let sat_bits = bits(op, 16, 20) + if unsigned { 0 } else { 1 };
    let shift_amount = bits(op, 7, 11);

    let shifted = if bit(op, 6) {
        // ASR, where an encoded zero means 32.
        let amount = if shift_amount == 0 { 32 } else { shift_amount };
        if amount >= 32 {
            ((b as i32) >> 31) as i64
        } else {
            ((b as i32) >> amount) as i64
        }
    } else {
        ((b << shift_amount) as i32) as i64
    };

    let (value, saturated) = if unsigned {
        unsigned_saturate(shifted, sat_bits)
    } else {
        signed_saturate(shifted, sat_bits)
    };
    cpu.regs[rd] = value;
    cpu.cpsr.q |= saturated;
    None
}

fn saturate16(cpu: &mut Cpu, op: u32, unsigned: bool) -> Option<Exit> {
    let rd = bits(op, 12, 15) as usize;
    let b = cpu.regs[(op & 0xF) as usize];
    let sat_bits = bits(op, 16, 19) + if unsigned { 0 } else { 1 };

    let mut out = 0u32;
    let mut saturated = false;
    for i in 0..2 {
        let lane = ((b >> (i * 16)) as u16) as i16 as i64;
        let (value, sat) = if unsigned {
            unsigned_saturate(lane, sat_bits)
        } else {
            signed_saturate(lane, sat_bits)
        };
        out |= (value & 0xFFFF) << (i * 16);
        saturated |= sat;
    }
    cpu.regs[rd] = out;
    cpu.cpsr.q |= saturated;
    None
}

// ---------------------------------------------------------------------------
// Signed dual multiplies
// ---------------------------------------------------------------------------

fn dual_multiply(cpu: &mut Cpu, op: u32, op1: u32) -> Option<Exit> {
    let rd = bits(op, 16, 19) as usize;
    let ra = bits(op, 12, 15) as usize;
    let rs = bits(op, 8, 11) as usize;
    let rm = (op & 0xF) as usize;

    let a = cpu.regs[rm];
    // the M bit swaps the halves of the second operand.
    let b = if bit(op, 5) {
        cpu.regs[rs].rotate_right(16)
    } else {
        cpu.regs[rs]
    };

    let a_lo = a as i16 as i64;
    let a_hi = (a >> 16) as i16 as i64;
    let b_lo = b as i16 as i64;
    let b_hi = (b >> 16) as i16 as i64;
    // bit 6 selects the subtracting variant (SMLSD / SMLSLD).
    let subtract = bit(op, 6);
    let dual = if subtract {
        a_lo * b_lo - a_hi * b_hi
    } else {
        a_lo * b_lo + a_hi * b_hi
    };

    match op1 {
        // SMLAD / SMUAD / SMLSD / SMUSD. the sum wraps, an overflow only
        // sets Q.
        0b10000 => {
            let sum = if ra == 15 {
                dual
            } else {
                dual + cpu.regs[ra] as i32 as i64
            };
            cpu.regs[rd] = sum as u32;
            cpu.cpsr.q |= sum != sum as i32 as i64;
        }
        // SMLALD / SMLSLD accumulate into the RdHi:RdLo pair.
        0b10100 => {
            let acc = ((cpu.regs[rd] as u64) << 32 | cpu.regs[ra] as u64) as i64;
            let sum = acc.wrapping_add(dual) as u64;
            cpu.regs[ra] = sum as u32;
            cpu.regs[rd] = (sum >> 32) as u32;
        }
        // SMMUL / SMMLA / SMMLS keep the top 32 bits of a full 64-bit product.
        0b10101 => {
            // bit 5 is the rounding bit here, and bit 6 selects SMMLS.
            let round = bit(op, 5);
            let product = (a as i32 as i64) * (cpu.regs[rs] as i32 as i64);
            let acc = if ra == 15 {
                0i64
            } else {
                (cpu.regs[ra] as i32 as i64) << 32
            };
            let mut total = if subtract {
                acc - product
            } else {
                acc + product
            };
            if round {
                total += 0x8000_0000;
            }
            cpu.regs[rd] = (total >> 32) as u32;
        }
        _ => return undefined(cpu, op),
    }
    None
}

/// USAD8 / USADA8, sum of absolute byte differences.
fn usad8(cpu: &mut Cpu, op: u32) -> Option<Exit> {
    let rd = bits(op, 16, 19) as usize;
    let ra = bits(op, 12, 15) as usize;
    let a = cpu.regs[(op & 0xF) as usize];
    let b = cpu.regs[bits(op, 8, 11) as usize];

    let mut sum = 0u32;
    for i in 0..4 {
        let x = ((a >> (i * 8)) & 0xFF) as i32;
        let y = ((b >> (i * 8)) & 0xFF) as i32;
        sum = sum.wrapping_add((x - y).unsigned_abs());
    }
    if ra != 15 {
        sum = sum.wrapping_add(cpu.regs[ra]);
    }
    cpu.regs[rd] = sum;
    None
}
