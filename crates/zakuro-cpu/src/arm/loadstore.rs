//! loads and stores, the single-transfer family, the "extra" halfword and
//! doubleword forms, block transfers and the exclusive/swap primitives.

use zakuro_common::bits::{bit, bits};

use super::shifter::{self, ShiftType};
use crate::{Bus, Cpu, Exit, PC};

// ---------------------------------------------------------------------------
// Single data transfer, LDR / STR / LDRB / STRB
// ---------------------------------------------------------------------------

pub fn single<B: Bus>(cpu: &mut Cpu, bus: &mut B, op: u32) -> Option<Exit> {
    let rn = bits(op, 16, 19) as usize;
    let rd = bits(op, 12, 15) as usize;
    let pre_index = bit(op, 24);
    let add = bit(op, 23);
    let byte = bit(op, 22);
    let writeback = bit(op, 21);
    let load = bit(op, 20);

    let offset = if bit(op, 25) {
        // register offset with an immediate shift.
        let rm = (op & 0xF) as usize;
        let ty = ShiftType::from_bits(bits(op, 5, 6));
        let amount = bits(op, 7, 11);
        shifter::shift_by_immediate(ty, cpu.regs[rm], amount, cpu.cpsr.c).0
    } else {
        op & 0xFFF
    };

    let base = cpu.regs[rn];
    let offset_addr = if add {
        base.wrapping_add(offset)
    } else {
        base.wrapping_sub(offset)
    };
    let addr = if pre_index { offset_addr } else { base };

    if load {
        let value = if byte {
            bus.read8(addr) as u32
        } else {
            bus.read32(addr)
        };
        // write the base back before the load so that ldr rN, [rN], #imm
        // ends up with the loaded value, matching hardware.
        if (!pre_index || writeback) && rn != rd {
            cpu.regs[rn] = offset_addr;
        }
        if rd == PC {
            // from ARMv5 onwards a load into the PC interworks, bit 0 of the
            // loaded value selects Thumb.
            cpu.branch_exchange(value);
        } else {
            cpu.set_reg(rd, value);
        }
    } else {
        // storing r15 writes the instruction's address plus 12 on ARM11.
        let value = if rd == PC {
            cpu.regs[PC].wrapping_add(4)
        } else {
            cpu.regs[rd]
        };
        if byte {
            bus.write8(addr, value as u8);
        } else {
            bus.write32(addr, value);
        }
        if !pre_index || writeback {
            cpu.regs[rn] = offset_addr;
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Extra load/store, LDRH / STRH / LDRSB / LDRSH / LDRD / STRD
// ---------------------------------------------------------------------------

pub fn extra<B: Bus>(cpu: &mut Cpu, bus: &mut B, op: u32) -> Option<Exit> {
    let rn = bits(op, 16, 19) as usize;
    let rd = bits(op, 12, 15) as usize;
    let pre_index = bit(op, 24);
    let add = bit(op, 23);
    let immediate = bit(op, 22);
    let writeback = bit(op, 21);
    let load = bit(op, 20);
    let sh = bits(op, 5, 6);

    let offset = if immediate {
        (bits(op, 8, 11) << 4) | (op & 0xF)
    } else {
        cpu.regs[(op & 0xF) as usize]
    };

    let base = cpu.regs[rn];
    let offset_addr = if add {
        base.wrapping_add(offset)
    } else {
        base.wrapping_sub(offset)
    };
    let addr = if pre_index { offset_addr } else { base };

    let mut wrote_base = false;
    match (sh, load) {
        // LDRH / STRH
        (0b01, true) => {
            let value = bus.read16(addr) as u32;
            if !pre_index || writeback {
                cpu.regs[rn] = offset_addr;
                wrote_base = true;
            }
            cpu.set_reg(rd, value);
        }
        (0b01, false) => bus.write16(addr, cpu.regs[rd] as u16),

        // LDRSB
        (0b10, true) => {
            let value = bus.read8(addr) as i8 as i32 as u32;
            if !pre_index || writeback {
                cpu.regs[rn] = offset_addr;
                wrote_base = true;
            }
            cpu.set_reg(rd, value);
        }
        // LDRD, loads an even/odd register pair.
        (0b10, false) => {
            let lo = bus.read32(addr);
            let hi = bus.read32(addr.wrapping_add(4));
            if !pre_index || writeback {
                cpu.regs[rn] = offset_addr;
                wrote_base = true;
            }
            cpu.set_reg(rd, lo);
            cpu.set_reg(rd + 1, hi);
        }

        // LDRSH
        (0b11, true) => {
            let value = bus.read16(addr) as i16 as i32 as u32;
            if !pre_index || writeback {
                cpu.regs[rn] = offset_addr;
                wrote_base = true;
            }
            cpu.set_reg(rd, value);
        }
        // STRD
        (0b11, false) => {
            bus.write32(addr, cpu.regs[rd]);
            bus.write32(addr.wrapping_add(4), cpu.regs[rd + 1]);
        }

        _ => return Some(Exit::Undefined { pc: cpu.current_pc(), opcode: op }),
    }

    if !wrote_base && (!pre_index || writeback) {
        cpu.regs[rn] = offset_addr;
    }
    None
}

// ---------------------------------------------------------------------------
// Block transfer, LDM / STM
// ---------------------------------------------------------------------------

pub fn block<B: Bus>(cpu: &mut Cpu, bus: &mut B, op: u32) -> Option<Exit> {
    let rn = bits(op, 16, 19) as usize;
    let pre_index = bit(op, 24);
    let add = bit(op, 23);
    let user_bank = bit(op, 22);
    let writeback = bit(op, 21);
    let load = bit(op, 20);
    let mut list = op & 0xFFFF;

    let base = cpu.regs[rn];

    // an empty list transfers r15 alone and moves the base by 0x40.
    let empty = list == 0;
    if empty {
        list = 1 << 15;
    }
    let count = list.count_ones();
    let span = if empty { 0x40 } else { count * 4 };

    let lowest = if add {
        if pre_index {
            base.wrapping_add(4)
        } else {
            base
        }
    } else if pre_index {
        base.wrapping_sub(span)
    } else {
        base.wrapping_sub(span).wrapping_add(4)
    };

    let final_base = if add {
        base.wrapping_add(span)
    } else {
        base.wrapping_sub(span)
    };

    // S with r15 loaded means "restore CPSR too", without r15 it means the
    // transfer uses the User-mode bank.
    let transfer_user_bank = user_bank && !(load && bit(op, 15));

    let mut addr = lowest;
    if load {
        // the base register must not be written back if it was loaded.
        let base_in_list = list & (1 << rn) != 0;
        if writeback && !base_in_list {
            cpu.regs[rn] = final_base;
        }
        for i in 0..16 {
            if list & (1 << i) == 0 {
                continue;
            }
            let value = bus.read32(addr);
            addr = addr.wrapping_add(4);
            if transfer_user_bank {
                cpu.write_user_reg(i, value);
            } else if i == 15 {
                if user_bank {
                    cpu.restore_cpsr();
                }
                cpu.branch_exchange(value);
            } else {
                cpu.regs[i] = value;
            }
        }
        if writeback && base_in_list {
            // hardware leaves the loaded value in place.
        }
    } else {
        for i in 0..16 {
            if list & (1 << i) == 0 {
                continue;
            }
            let value = if transfer_user_bank {
                cpu.read_user_reg(i)
            } else if i == 15 {
                cpu.regs[PC].wrapping_add(4)
            } else if i == rn && writeback && (list & ((1 << rn) - 1)) != 0 {
                // storing the base when it is not the lowest register in the
                // list stores the written-back value on ARM11.
                final_base
            } else {
                cpu.regs[i]
            };
            bus.write32(addr, value);
            addr = addr.wrapping_add(4);
        }
        if writeback {
            cpu.regs[rn] = final_base;
        }
    }
    None
}

// ---------------------------------------------------------------------------
// SWP / SWPB
// ---------------------------------------------------------------------------

pub fn swap<B: Bus>(cpu: &mut Cpu, bus: &mut B, op: u32) -> Option<Exit> {
    let rn = bits(op, 16, 19) as usize;
    let rd = bits(op, 12, 15) as usize;
    let rm = (op & 0xF) as usize;
    let byte = bit(op, 22);
    let addr = cpu.regs[rn];
    let source = cpu.regs[rm];

    if byte {
        let old = bus.read8(addr) as u32;
        bus.write8(addr, source as u8);
        cpu.set_reg(rd, old);
    } else {
        let old = bus.read32(addr);
        bus.write32(addr, source);
        cpu.set_reg(rd, old);
    }
    None
}

// ---------------------------------------------------------------------------
// LDREX / STREX (ARMv6, with the B/H/D variants from ARMv6K)
// ---------------------------------------------------------------------------

pub fn exclusive<B: Bus>(cpu: &mut Cpu, bus: &mut B, op: u32) -> Option<Exit> {
    let load = bit(op, 20);
    let size = bits(op, 21, 22);
    let rn = bits(op, 16, 19) as usize;
    let addr = cpu.regs[rn];

    if load {
        let rd = bits(op, 12, 15) as usize;
        cpu.exclusive_addr = Some(addr);
        let value = match size {
            0b00 => bus.read32(addr),
            0b01 => {
                // LDREXD
                let lo = bus.read32(addr);
                let hi = bus.read32(addr.wrapping_add(4));
                cpu.regs[rd] = lo;
                cpu.set_reg(rd + 1, hi);
                return None;
            }
            0b10 => bus.read8(addr) as u32,
            _ => bus.read16(addr) as u32,
        };
        cpu.set_reg(rd, value);
    } else {
        let rd = bits(op, 12, 15) as usize;
        let rm = (op & 0xF) as usize;
        let value = cpu.regs[rm];

        // with a single emulated core the reservation can only be lost to a
        // context switch, which the kernel signals by clearing it.
        let succeeded = cpu.exclusive_addr == Some(addr);
        if succeeded {
            match size {
                0b00 => bus.write32(addr, value),
                0b01 => {
                    bus.write32(addr, cpu.regs[rm]);
                    bus.write32(addr.wrapping_add(4), cpu.regs[rm + 1]);
                }
                0b10 => bus.write8(addr, value as u8),
                _ => bus.write16(addr, value as u16),
            }
            cpu.exclusive_addr = None;
        }
        cpu.set_reg(rd, !succeeded as u32);
    }
    None
}
