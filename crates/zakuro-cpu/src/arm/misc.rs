//! branches, status register transfers and coprocessor access.

use zakuro_common::bits::{bit, bits, sign_extend};

use crate::psr::{Mode, Psr};
use crate::{Bus, Cpu, Exit, LR, PC};

pub fn branch<B: Bus>(cpu: &mut Cpu, _bus: &mut B, op: u32) -> Option<Exit> {
    let offset = sign_extend(op & 0x00FF_FFFF, 24) << 2;
    let target = cpu.regs[PC].wrapping_add(offset as u32);
    if bit(op, 24) {
        cpu.regs[LR] = cpu.regs[PC].wrapping_sub(4);
    }
    cpu.branch(target);
    None
}

/// BX and BLX with a register operand.
pub fn branch_exchange<B: Bus>(cpu: &mut Cpu, _bus: &mut B, op: u32) -> Option<Exit> {
    let rm = (op & 0xF) as usize;
    let target = cpu.regs[rm];
    // bits 7:4 are 0b0011 for BLX, 0b0001 for BX.
    if bits(op, 4, 7) == 0b0011 {
        cpu.regs[LR] = cpu.regs[PC].wrapping_sub(4);
    }
    cpu.branch_exchange(target);
    None
}

/// BLX label, the unconditional form that always switches to Thumb.
pub fn branch_link_exchange_immediate<B: Bus>(cpu: &mut Cpu, _bus: &mut B, op: u32) -> Option<Exit> {
    let offset = (sign_extend(op & 0x00FF_FFFF, 24) << 2) | ((bit(op, 24) as i32) << 1);
    let target = cpu.regs[PC].wrapping_add(offset as u32);
    cpu.regs[LR] = cpu.regs[PC].wrapping_sub(4);
    cpu.cpsr.thumb = true;
    cpu.branch(target);
    None
}

pub fn clz<B: Bus>(cpu: &mut Cpu, _bus: &mut B, op: u32) -> Option<Exit> {
    let rd = bits(op, 12, 15) as usize;
    let rm = (op & 0xF) as usize;
    cpu.regs[rd] = cpu.regs[rm].leading_zeros();
    None
}

pub fn mrs<B: Bus>(cpu: &mut Cpu, _bus: &mut B, op: u32) -> Option<Exit> {
    let rd = bits(op, 12, 15) as usize;
    let value = if bit(op, 22) {
        cpu.spsr.to_bits()
    } else {
        cpu.cpsr.to_bits()
    };
    cpu.regs[rd] = value;
    None
}

pub fn msr<B: Bus>(cpu: &mut Cpu, _bus: &mut B, op: u32) -> Option<Exit> {
    let field_mask = bits(op, 16, 19);
    let spsr = bit(op, 22);

    let value = if bit(op, 25) {
        let imm = op & 0xFF;
        imm.rotate_right(bits(op, 8, 11) * 2)
    } else {
        cpu.regs[(op & 0xF) as usize]
    };

    if spsr {
        let mut psr = cpu.spsr;
        psr.write_masked(value, field_mask, true);
        cpu.spsr = psr;
    } else {
        let privileged = cpu.cpsr.mode.is_privileged();
        // a mode change has to go through switch_mode so the banked
        // registers follow.
        let mut psr = cpu.cpsr;
        psr.write_masked(value, field_mask, privileged);
        if privileged && field_mask & 0b0001 != 0 && psr.mode != cpu.cpsr.mode {
            let new_mode = psr.mode;
            cpu.switch_mode(new_mode);
        }
        let mode = cpu.cpsr.mode;
        cpu.cpsr = Psr { mode, ..psr };
    }
    None
}

/// MRC / MCR. Only CP15 and the VFP coprocessors exist on this core.
pub fn coprocessor_register<B: Bus>(cpu: &mut Cpu, _bus: &mut B, op: u32) -> Option<Exit> {
    let cp = bits(op, 8, 11);
    let opc1 = bits(op, 21, 23);
    let crn = bits(op, 16, 19);
    let rd = bits(op, 12, 15) as usize;
    let crm = op & 0xF;
    let opc2 = bits(op, 5, 7);
    let load = bit(op, 20);

    match cp {
        15 => {
            if load {
                let value = cpu.cp15.read(opc1, crn, crm, opc2);
                if rd == 15 {
                    // MRC with Rd as r15 writes the condition flags rather
                    // than branching.
                    cpu.cpsr.n = bit(value, 31);
                    cpu.cpsr.z = bit(value, 30);
                    cpu.cpsr.c = bit(value, 29);
                    cpu.cpsr.v = bit(value, 28);
                } else {
                    cpu.regs[rd] = value;
                }
            } else {
                let value = cpu.regs[rd];
                cpu.cp15.write(opc1, crn, crm, opc2, value);
            }
            None
        }
        10 | 11 => crate::vfp::coprocessor_register(cpu, op),
        _ => {
            log::warn!(
                "access to unimplemented coprocessor p{cp} at 0x{:08X}",
                cpu.current_pc()
            );
            None
        }
    }
}

pub fn supervisor_call<B: Bus>(cpu: &mut Cpu, _bus: &mut B, op: u32) -> Option<Exit> {
    // the PC has already been advanced past the instruction by the fetch loop,
    // but it still carries the pipeline offset, rewind it so the kernel
    // resumes at the next instruction.
    let next = cpu.regs[PC].wrapping_sub(4);
    cpu.regs[PC] = next;
    cpu.branch(next);
    Some(Exit::Supervisor(op & 0x00FF_FFFF))
}

pub fn breakpoint<B: Bus>(cpu: &mut Cpu, _bus: &mut B, op: u32) -> Option<Exit> {
    let imm = (bits(op, 8, 19) << 4) | (op & 0xF);
    Some(Exit::Breakpoint {
        pc: cpu.current_pc(),
        imm,
    })
}

/// CPS, SETEND and the ARMv6K hint instructions.
pub fn hint_or_cps<B: Bus>(cpu: &mut Cpu, _bus: &mut B, op: u32) -> Option<Exit> {
    if bits(op, 20, 27) == 0b0001_0000 {
        // CPS. Only meaningful in a privileged mode.
        if cpu.cpsr.mode.is_privileged() {
            let imod = bits(op, 18, 19);
            let mmod = bit(op, 17);
            if imod & 0b10 != 0 {
                let disable = imod & 0b01 != 0;
                if bit(op, 8) {
                    cpu.cpsr.a = disable;
                }
                if bit(op, 7) {
                    cpu.cpsr.i = disable;
                }
                if bit(op, 6) {
                    cpu.cpsr.f = disable;
                }
            }
            if mmod {
                cpu.switch_mode(Mode::from_bits(op & 0x1F));
            }
        }
        return None;
    }
    // SETEND and the hints, nothing to do.
    None
}
