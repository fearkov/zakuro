//! ARM-state instruction decoding.

pub mod alu;
pub mod dataproc;
pub mod loadstore;
pub mod media;
pub mod misc;
pub mod multiply;
pub mod shifter;

use zakuro_common::bits::{bit, bits};

use crate::psr::check_condition;
use crate::{Bus, Cpu, Exit};

pub fn execute<B: Bus>(cpu: &mut Cpu, bus: &mut B, op: u32) -> Option<Exit> {
    let cond = op >> 28;
    if cond == 0xF {
        return unconditional(cpu, bus, op);
    }
    if !check_condition(cond, &cpu.cpsr) {
        return None;
    }

    match bits(op, 25, 27) {
        0b000 => decode_dataproc_register(cpu, bus, op),
        0b001 => decode_dataproc_immediate(cpu, bus, op),
        0b010 => loadstore::single(cpu, bus, op),
        0b011 => {
            if bit(op, 4) {
                media::execute(cpu, bus, op)
            } else {
                loadstore::single(cpu, bus, op)
            }
        }
        0b100 => loadstore::block(cpu, bus, op),
        0b101 => misc::branch(cpu, bus, op),
        0b110 => crate::vfp::coprocessor_load_store(cpu, bus, op),
        _ => {
            if bit(op, 24) {
                misc::supervisor_call(cpu, bus, op)
            } else if bit(op, 4) {
                misc::coprocessor_register(cpu, bus, op)
            } else {
                crate::vfp::data_processing(cpu, op)
            }
        }
    }
}

/// the 0b000 space, register data processing, plus multiplies, synchronization
/// primitives, extra load/stores and the status-register transfers hiding in
/// its unused encodings.
fn decode_dataproc_register<B: Bus>(cpu: &mut Cpu, bus: &mut B, op: u32) -> Option<Exit> {
    // bits 7 and 4 both set means we are outside the data-processing encoding.
    if op & 0x0000_0090 == 0x0000_0090 {
        if bits(op, 5, 6) == 0 {
            // multiplies and synchronization.
            if bit(op, 24) {
                return if bit(op, 23) {
                    loadstore::exclusive(cpu, bus, op)
                } else {
                    loadstore::swap(cpu, bus, op)
                };
            }
            // bits 23:21 pick the multiply.
            return match bits(op, 21, 23) {
                0b000 | 0b001 => multiply::short(cpu, bus, op),
                0b010 | 0b100 | 0b101 | 0b110 | 0b111 => multiply::long(cpu, bus, op),
                // 0b011 is MLS, which arrived in ARMv6T2 and does not exist
                // on the MP11 cores.
                _ => Some(Exit::Undefined {
                    pc: cpu.current_pc(),
                    opcode: op,
                }),
            };
        }
        return loadstore::extra(cpu, bus, op);
    }

    // the "miscellaneous" hole, opcode field 0b10xx with S clear.
    if bits(op, 23, 24) == 0b10 && !bit(op, 20) {
        return decode_misc(cpu, bus, op);
    }

    dataproc::execute(cpu, bus, op)
}

fn decode_misc<B: Bus>(cpu: &mut Cpu, bus: &mut B, op: u32) -> Option<Exit> {
    match bits(op, 4, 7) {
        0b0000 => {
            if bit(op, 21) {
                misc::msr(cpu, bus, op)
            } else {
                misc::mrs(cpu, bus, op)
            }
        }
        0b0001 => {
            if bits(op, 21, 22) == 0b11 {
                misc::clz(cpu, bus, op)
            } else {
                misc::branch_exchange(cpu, bus, op)
            }
        }
        0b0011 => misc::branch_exchange(cpu, bus, op),
        0b0101 => multiply::saturating(cpu, bus, op),
        0b0111 => misc::breakpoint(cpu, bus, op),
        // 1yx0, the ARMv5TE halfword multiplies.
        n if n & 0b1001 == 0b1000 => multiply::halfword(cpu, bus, op),
        _ => Some(Exit::Undefined {
            pc: cpu.current_pc(),
            opcode: op,
        }),
    }
}

/// the 0b001 space, immediate data processing and MSR with an immediate.
fn decode_dataproc_immediate<B: Bus>(cpu: &mut Cpu, bus: &mut B, op: u32) -> Option<Exit> {
    if bits(op, 23, 24) == 0b10 && !bit(op, 20) {
        if bit(op, 21) {
            // MSR immediate.
            return misc::msr(cpu, bus, op);
        }
        return Some(Exit::Undefined {
            pc: cpu.current_pc(),
            opcode: op,
        });
    }
    dataproc::execute(cpu, bus, op)
}

/// instructions encoded with cond == 0b1111, which on ARMv5 and later means
/// "unconditional" rather than "never".
fn unconditional<B: Bus>(cpu: &mut Cpu, bus: &mut B, op: u32) -> Option<Exit> {
    if op & 0xFE00_0000 == 0xFA00_0000 {
        return misc::branch_link_exchange_immediate(cpu, bus, op);
    }
    if op & 0xFFF0_00F0 == 0xF570_0010 {
        // CLREX, drop any outstanding LDREX reservation.
        cpu.exclusive_addr = None;
        return None;
    }

    match bits(op, 25, 27) {
        // CPS and SETEND.
        0b000 => misc::hint_or_cps(cpu, bus, op),
        // PLD and the cache preload hints, architecturally free to ignore.
        0b010 | 0b011 => None,
        // the "2" variants of the coprocessor instructions behave identically
        // for the coprocessors this core has.
        0b110 => crate::vfp::coprocessor_load_store(cpu, bus, op),
        0b111 => {
            if bit(op, 4) {
                misc::coprocessor_register(cpu, bus, op)
            } else {
                crate::vfp::data_processing(cpu, op)
            }
        }
        _ => Some(Exit::Undefined {
            pc: cpu.current_pc(),
            opcode: op,
        }),
    }
}
