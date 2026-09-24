//! thumb-state instruction decoding.

use zakuro_common::bits::{bit, bits, sign_extend};

use crate::arm::alu::*;
use crate::arm::shifter::{shift_by_immediate, shift_by_register, ShiftType};
use crate::{Bus, Cpu, Exit, LR, PC, SP};

pub fn execute<B: Bus>(cpu: &mut Cpu, bus: &mut B, op: u32) -> Option<Exit> {
    match bits(op, 13, 15) {
        0b000 => {
            if bits(op, 11, 12) == 0b11 {
                add_subtract(cpu, op)
            } else {
                shift_immediate(cpu, op)
            }
        }
        0b001 => immediate_ops(cpu, op),
        0b010 => {
            if bit(op, 12) {
                load_store_register(cpu, bus, op)
            } else if bit(op, 11) {
                pc_relative_load(cpu, bus, op)
            } else if bit(op, 10) {
                high_register(cpu, bus, op)
            } else {
                alu_operations(cpu, op)
            }
        }
        0b011 => load_store_immediate(cpu, bus, op),
        0b100 => {
            if bit(op, 12) {
                sp_relative(cpu, bus, op)
            } else {
                load_store_halfword(cpu, bus, op)
            }
        }
        0b101 => {
            if bit(op, 12) {
                miscellaneous(cpu, bus, op)
            } else {
                load_address(cpu, op)
            }
        }
        0b110 => {
            if bit(op, 12) {
                conditional_branch(cpu, op)
            } else {
                block_transfer(cpu, bus, op)
            }
        }
        _ => long_branch(cpu, op),
    }
}

// -- format 1, shift by immediate -------------------------------------------

fn shift_immediate(cpu: &mut Cpu, op: u32) -> Option<Exit> {
    let ty = ShiftType::from_bits(bits(op, 11, 12));
    let amount = bits(op, 6, 10);
    let rs = bits(op, 3, 5) as usize;
    let rd = (op & 7) as usize;

    let (result, carry) = shift_by_immediate(ty, cpu.regs[rs], amount, cpu.cpsr.c);
    cpu.regs[rd] = result;
    cpu.set_nz(result);
    cpu.cpsr.c = carry;
    None
}

// -- format 2, add/subtract register or 3-bit immediate ----------------------

fn add_subtract(cpu: &mut Cpu, op: u32) -> Option<Exit> {
    let immediate = bit(op, 10);
    let subtract = bit(op, 9);
    let operand = if immediate {
        bits(op, 6, 8)
    } else {
        cpu.regs[bits(op, 6, 8) as usize]
    };
    let rs = bits(op, 3, 5) as usize;
    let rd = (op & 7) as usize;
    let a = cpu.regs[rs];

    let (result, carry, overflow) = if subtract {
        sub_with_flags(a, operand)
    } else {
        add_with_flags(a, operand)
    };
    cpu.regs[rd] = result;
    cpu.set_nz(result);
    cpu.cpsr.c = carry;
    cpu.cpsr.v = overflow;
    None
}

// -- format 3, MOV/CMP/ADD/SUB with an 8-bit immediate -----------------------

fn immediate_ops(cpu: &mut Cpu, op: u32) -> Option<Exit> {
    let rd = bits(op, 8, 10) as usize;
    let imm = op & 0xFF;
    let a = cpu.regs[rd];

    match bits(op, 11, 12) {
        0b00 => {
            // MOV
            cpu.regs[rd] = imm;
            cpu.set_nz(imm);
        }
        0b01 => {
            // CMP
            let (result, carry, overflow) = sub_with_flags(a, imm);
            cpu.set_nz(result);
            cpu.cpsr.c = carry;
            cpu.cpsr.v = overflow;
        }
        0b10 => {
            let (result, carry, overflow) = add_with_flags(a, imm);
            cpu.regs[rd] = result;
            cpu.set_nz(result);
            cpu.cpsr.c = carry;
            cpu.cpsr.v = overflow;
        }
        _ => {
            let (result, carry, overflow) = sub_with_flags(a, imm);
            cpu.regs[rd] = result;
            cpu.set_nz(result);
            cpu.cpsr.c = carry;
            cpu.cpsr.v = overflow;
        }
    }
    None
}

// -- format 4, the ALU operations --------------------------------------------

fn alu_operations(cpu: &mut Cpu, op: u32) -> Option<Exit> {
    let rs = bits(op, 3, 5) as usize;
    let rd = (op & 7) as usize;
    let a = cpu.regs[rd];
    let b = cpu.regs[rs];
    let carry_in = cpu.cpsr.c;

    match bits(op, 6, 9) {
        0x0 => {
            let r = a & b;
            cpu.regs[rd] = r;
            cpu.set_nz(r);
        }
        0x1 => {
            let r = a ^ b;
            cpu.regs[rd] = r;
            cpu.set_nz(r);
        }
        0x2 | 0x3 | 0x4 | 0x7 => {
            let ty = match bits(op, 6, 9) {
                0x2 => ShiftType::Lsl,
                0x3 => ShiftType::Lsr,
                0x4 => ShiftType::Asr,
                _ => ShiftType::Ror,
            };
            let (r, carry) = shift_by_register(ty, a, b, carry_in);
            cpu.regs[rd] = r;
            cpu.set_nz(r);
            cpu.cpsr.c = carry;
        }
        0x5 => {
            let (r, carry, overflow) = adc_with_flags(a, b, carry_in);
            cpu.regs[rd] = r;
            cpu.set_nz(r);
            cpu.cpsr.c = carry;
            cpu.cpsr.v = overflow;
        }
        0x6 => {
            let (r, carry, overflow) = sbc_with_flags(a, b, carry_in);
            cpu.regs[rd] = r;
            cpu.set_nz(r);
            cpu.cpsr.c = carry;
            cpu.cpsr.v = overflow;
        }
        0x8 => {
            // TST
            let r = a & b;
            cpu.set_nz(r);
        }
        0x9 => {
            // NEG, encoded as RSB #0.
            let (r, carry, overflow) = sub_with_flags(0, b);
            cpu.regs[rd] = r;
            cpu.set_nz(r);
            cpu.cpsr.c = carry;
            cpu.cpsr.v = overflow;
        }
        0xA => {
            let (r, carry, overflow) = sub_with_flags(a, b);
            cpu.set_nz(r);
            cpu.cpsr.c = carry;
            cpu.cpsr.v = overflow;
        }
        0xB => {
            let (r, carry, overflow) = add_with_flags(a, b);
            cpu.set_nz(r);
            cpu.cpsr.c = carry;
            cpu.cpsr.v = overflow;
        }
        0xC => {
            let r = a | b;
            cpu.regs[rd] = r;
            cpu.set_nz(r);
        }
        0xD => {
            let r = a.wrapping_mul(b);
            cpu.regs[rd] = r;
            cpu.set_nz(r);
        }
        0xE => {
            let r = a & !b;
            cpu.regs[rd] = r;
            cpu.set_nz(r);
        }
        _ => {
            let r = !b;
            cpu.regs[rd] = r;
            cpu.set_nz(r);
        }
    }
    None
}

// -- format 5, high registers and BX/BLX -------------------------------------

fn high_register<B: Bus>(cpu: &mut Cpu, _bus: &mut B, op: u32) -> Option<Exit> {
    let rd = ((op & 7) | ((op >> 4) & 8)) as usize;
    let rs = bits(op, 3, 6) as usize;

    // reading r15 in Thumb gives the instruction address plus 4, which is what
    // regs[PC] already holds, but with bit 1 cleared for the ALU forms.
    let source = cpu.regs[rs];

    match bits(op, 8, 9) {
        0b00 => {
            let result = cpu.regs[rd].wrapping_add(source);
            cpu.set_reg(rd, result);
        }
        0b01 => {
            let (result, carry, overflow) = sub_with_flags(cpu.regs[rd], source);
            cpu.set_nz(result);
            cpu.cpsr.c = carry;
            cpu.cpsr.v = overflow;
        }
        0b10 => cpu.set_reg(rd, source),
        _ => {
            // BX, or BLX when bit 7 is set.
            if bit(op, 7) {
                cpu.regs[LR] = cpu.regs[PC].wrapping_sub(2) | 1;
            }
            cpu.branch_exchange(source);
        }
    }
    None
}

// -- format 6, PC-relative load ----------------------------------------------

fn pc_relative_load<B: Bus>(cpu: &mut Cpu, bus: &mut B, op: u32) -> Option<Exit> {
    let rd = bits(op, 8, 10) as usize;
    let offset = (op & 0xFF) * 4;
    // the literal pool is addressed from the word-aligned PC.
    let addr = (cpu.regs[PC] & !3).wrapping_add(offset);
    cpu.regs[rd] = bus.read32(addr);
    None
}

// -- formats 7 and 8, register-offset loads and stores -----------------------

fn load_store_register<B: Bus>(cpu: &mut Cpu, bus: &mut B, op: u32) -> Option<Exit> {
    let ro = bits(op, 6, 8) as usize;
    let rb = bits(op, 3, 5) as usize;
    let rd = (op & 7) as usize;
    let addr = cpu.regs[rb].wrapping_add(cpu.regs[ro]);

    if bit(op, 9) {
        // format 8, sign-extended byte and halfword.
        match bits(op, 10, 11) {
            0b00 => bus.write16(addr, cpu.regs[rd] as u16),
            0b01 => cpu.regs[rd] = bus.read8(addr) as i8 as i32 as u32,
            0b10 => cpu.regs[rd] = bus.read16(addr) as u32,
            _ => cpu.regs[rd] = bus.read16(addr) as i16 as i32 as u32,
        }
    } else {
        match bits(op, 10, 11) {
            0b00 => bus.write32(addr, cpu.regs[rd]),
            0b01 => bus.write8(addr, cpu.regs[rd] as u8),
            0b10 => cpu.regs[rd] = bus.read32(addr),
            _ => cpu.regs[rd] = bus.read8(addr) as u32,
        }
    }
    None
}

// -- format 9, immediate-offset loads and stores -----------------------------

fn load_store_immediate<B: Bus>(cpu: &mut Cpu, bus: &mut B, op: u32) -> Option<Exit> {
    let byte = bit(op, 12);
    let load = bit(op, 11);
    let offset = bits(op, 6, 10);
    let rb = bits(op, 3, 5) as usize;
    let rd = (op & 7) as usize;
    // word accesses scale the offset by four, byte accesses not at all.
    let addr = cpu.regs[rb].wrapping_add(if byte { offset } else { offset * 4 });

    match (load, byte) {
        (true, false) => cpu.regs[rd] = bus.read32(addr),
        (true, true) => cpu.regs[rd] = bus.read8(addr) as u32,
        (false, false) => bus.write32(addr, cpu.regs[rd]),
        (false, true) => bus.write8(addr, cpu.regs[rd] as u8),
    }
    None
}

// -- format 10, halfword loads and stores ------------------------------------

fn load_store_halfword<B: Bus>(cpu: &mut Cpu, bus: &mut B, op: u32) -> Option<Exit> {
    let load = bit(op, 11);
    let offset = bits(op, 6, 10) * 2;
    let rb = bits(op, 3, 5) as usize;
    let rd = (op & 7) as usize;
    let addr = cpu.regs[rb].wrapping_add(offset);

    if load {
        cpu.regs[rd] = bus.read16(addr) as u32;
    } else {
        bus.write16(addr, cpu.regs[rd] as u16);
    }
    None
}

// -- format 11, SP-relative loads and stores ---------------------------------

fn sp_relative<B: Bus>(cpu: &mut Cpu, bus: &mut B, op: u32) -> Option<Exit> {
    let load = bit(op, 11);
    let rd = bits(op, 8, 10) as usize;
    let addr = cpu.regs[SP].wrapping_add((op & 0xFF) * 4);

    if load {
        cpu.regs[rd] = bus.read32(addr);
    } else {
        bus.write32(addr, cpu.regs[rd]);
    }
    None
}

// -- format 12, ADD Rd, PC/SP, #imm ------------------------------------------

fn load_address(cpu: &mut Cpu, op: u32) -> Option<Exit> {
    let rd = bits(op, 8, 10) as usize;
    let offset = (op & 0xFF) * 4;
    let base = if bit(op, 11) {
        cpu.regs[SP]
    } else {
        cpu.regs[PC] & !3
    };
    cpu.regs[rd] = base.wrapping_add(offset);
    None
}

// -- format 13/14 and the ARMv6 additions ------------------------------------

fn miscellaneous<B: Bus>(cpu: &mut Cpu, bus: &mut B, op: u32) -> Option<Exit> {
    match bits(op, 8, 11) {
        // ADD/SUB immediate to SP.
        0b0000 => {
            let offset = (op & 0x7F) * 4;
            cpu.regs[SP] = if bit(op, 7) {
                cpu.regs[SP].wrapping_sub(offset)
            } else {
                cpu.regs[SP].wrapping_add(offset)
            };
            None
        }
        // SXTH / SXTB / UXTH / UXTB.
        0b0010 => {
            let rd = (op & 7) as usize;
            let rm = bits(op, 3, 5) as usize;
            let value = cpu.regs[rm];
            cpu.regs[rd] = match bits(op, 6, 7) {
                0b00 => value as u16 as i16 as i32 as u32,
                0b01 => value as u8 as i8 as i32 as u32,
                0b10 => value & 0xFFFF,
                _ => value & 0xFF,
            };
            None
        }
        // REV / REV16 / REVSH.
        0b1010 => {
            let rd = (op & 7) as usize;
            let rm = bits(op, 3, 5) as usize;
            let value = cpu.regs[rm];
            cpu.regs[rd] = match bits(op, 6, 7) {
                0b00 => value.swap_bytes(),
                0b01 => ((value & 0x00FF_00FF) << 8) | ((value & 0xFF00_FF00) >> 8),
                0b11 => {
                    let swapped = ((value & 0xFF) << 8) | ((value >> 8) & 0xFF);
                    swapped as u16 as i16 as i32 as u32
                }
                _ => {
                    return Some(Exit::Undefined {
                        pc: cpu.current_pc(),
                        opcode: op,
                    })
                }
            };
            None
        }
        // CPS and SETEND, privileged or irrelevant, so nothing to do.
        0b0110 => None,
        // PUSH / POP.
        0b0100 | 0b0101 | 0b1100 | 0b1101 => push_pop(cpu, bus, op),
        // BKPT.
        0b1110 => Some(Exit::Breakpoint {
            pc: cpu.current_pc(),
            imm: op & 0xFF,
        }),
        // NOP and the ARMv6K hints.
        0b1111 => None,
        _ => Some(Exit::Undefined {
            pc: cpu.current_pc(),
            opcode: op,
        }),
    }
}

fn push_pop<B: Bus>(cpu: &mut Cpu, bus: &mut B, op: u32) -> Option<Exit> {
    let load = bit(op, 11);
    let extra = bit(op, 8);
    let list = op & 0xFF;
    let count = list.count_ones() + extra as u32;

    if load {
        // POP, read upward from SP.
        let mut addr = cpu.regs[SP];
        for i in 0..8 {
            if list & (1 << i) != 0 {
                cpu.regs[i] = bus.read32(addr);
                addr = addr.wrapping_add(4);
            }
        }
        cpu.regs[SP] = cpu.regs[SP].wrapping_add(count * 4);
        if extra {
            // POP {..., pc} interworks on ARMv5 and later.
            let target = bus.read32(addr);
            cpu.branch_exchange(target);
        }
    } else {
        // PUSH, the lowest register ends up at the lowest address.
        let start = cpu.regs[SP].wrapping_sub(count * 4);
        let mut addr = start;
        for i in 0..8 {
            if list & (1 << i) != 0 {
                bus.write32(addr, cpu.regs[i]);
                addr = addr.wrapping_add(4);
            }
        }
        if extra {
            bus.write32(addr, cpu.regs[LR]);
        }
        cpu.regs[SP] = start;
    }
    None
}

// -- format 15, LDMIA / STMIA ------------------------------------------------

fn block_transfer<B: Bus>(cpu: &mut Cpu, bus: &mut B, op: u32) -> Option<Exit> {
    let load = bit(op, 11);
    let rb = bits(op, 8, 10) as usize;
    let list = op & 0xFF;

    if list == 0 {
        // an empty list transfers r15 and moves the base by 0x40 on ARMv4.
        cpu.regs[rb] = cpu.regs[rb].wrapping_add(0x40);
        return None;
    }

    let mut addr = cpu.regs[rb];
    let writeback = !(load && list & (1 << rb) != 0);
    let final_base = addr.wrapping_add(list.count_ones() * 4);

    for i in 0..8 {
        if list & (1 << i) == 0 {
            continue;
        }
        if load {
            cpu.regs[i] = bus.read32(addr);
        } else {
            bus.write32(addr, cpu.regs[i]);
        }
        addr = addr.wrapping_add(4);
    }

    if writeback {
        cpu.regs[rb] = final_base;
    }
    None
}

// -- formats 16/17, conditional branch and SVC -------------------------------

fn conditional_branch(cpu: &mut Cpu, op: u32) -> Option<Exit> {
    let cond = bits(op, 8, 11);
    if cond == 0b1111 {
        // SWI / SVC.
        let next = cpu.regs[PC].wrapping_sub(2);
        cpu.branch(next);
        return Some(Exit::Supervisor(op & 0xFF));
    }
    if cond == 0b1110 {
        return Some(Exit::Undefined {
            pc: cpu.current_pc(),
            opcode: op,
        });
    }
    if crate::psr::check_condition(cond, &cpu.cpsr) {
        let offset = sign_extend(op & 0xFF, 8) << 1;
        let target = cpu.regs[PC].wrapping_add(offset as u32);
        cpu.branch(target);
    }
    None
}

// -- formats 18/19, unconditional and long branches --------------------------

fn long_branch(cpu: &mut Cpu, op: u32) -> Option<Exit> {
    match bits(op, 11, 12) {
        // b <label>
        0b00 => {
            let offset = sign_extend(op & 0x7FF, 11) << 1;
            let target = cpu.regs[PC].wrapping_add(offset as u32);
            cpu.branch(target);
        }
        // BL/BLX suffix with the low half of the offset.
        0b01 | 0b11 => {
            let target = cpu.regs[LR].wrapping_add((op & 0x7FF) << 1);
            let return_addr = cpu.regs[PC].wrapping_sub(2) | 1;
            cpu.regs[LR] = return_addr;
            if bits(op, 11, 12) == 0b01 {
                // BLX, switch to ARM state and force word alignment.
                cpu.cpsr.thumb = false;
                cpu.branch(target & !3);
            } else {
                cpu.branch(target);
            }
        }
        // BL/BLX prefix, stash the high half of the offset in LR.
        _ => {
            let offset = sign_extend(op & 0x7FF, 11) << 12;
            cpu.regs[LR] = cpu.regs[PC].wrapping_add(offset as u32);
        }
    }
    None
}
