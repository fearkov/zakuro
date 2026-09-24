//! ARM-state instruction tests.

mod common;

use common::{assert_flags, assert_reg, run, run_with, CODE_BASE, DATA_BASE};

// ---------------------------------------------------------------------------
// Data processing
// ---------------------------------------------------------------------------

#[test]
fn sub_borrow_clears_carry() {
    let Some(r) = run("mov r0, #0\n subs r1, r0, #1") else {
        return;
    };
    assert_reg!(r, 1, 0xFFFF_FFFF);
    assert_flags!(r, "N");
}

#[test]
fn sub_without_borrow_sets_carry() {
    let Some(r) = run("mov r0, #5\n subs r1, r0, #5") else {
        return;
    };
    assert_reg!(r, 1, 0);
    assert_flags!(r, "-ZC");
}

#[test]
fn signed_overflow_sets_v() {
    let Some(r) = run_with("adds r2, r0, r1", |cpu, _| {
        cpu.regs[0] = 0x7FFF_FFFF;
        cpu.regs[1] = 1;
    }) else {
        return;
    };
    assert_reg!(r, 2, 0x8000_0000);
    assert_flags!(r, "N--V");
}

#[test]
fn adc_carries_in() {
    let Some(r) = run_with("adcs r2, r0, r1", |cpu, _| {
        cpu.regs[0] = 0xFFFF_FFFF;
        cpu.regs[1] = 0;
        cpu.cpsr.c = true;
    }) else {
        return;
    };
    assert_reg!(r, 2, 0);
    assert_flags!(r, "-ZC");
}

#[test]
fn sbc_borrows_in() {
    // with carry clear, SBC subtracts an extra one.
    let Some(r) = run_with("sbcs r2, r0, r1", |cpu, _| {
        cpu.regs[0] = 10;
        cpu.regs[1] = 3;
        cpu.cpsr.c = false;
    }) else {
        return;
    };
    assert_reg!(r, 2, 6);
    assert_flags!(r, "--C");
}

#[test]
fn rsc_reverses_the_operands() {
    let Some(r) = run_with("rsc r2, r0, r1", |cpu, _| {
        cpu.regs[0] = 3;
        cpu.regs[1] = 10;
        cpu.cpsr.c = true;
    }) else {
        return;
    };
    assert_reg!(r, 2, 7);
}

/// the immediate rotate only sets the carry when the rotation is non-zero.
#[test]
fn immediate_rotate_updates_carry() {
    let Some(r) = run("movs r0, #0x80000000") else {
        return;
    };
    assert_reg!(r, 0, 0x8000_0000);
    assert_flags!(r, "N-C");
}

#[test]
fn immediate_without_rotate_preserves_carry() {
    let Some(r) = run_with("movs r0, #1", |cpu, _| cpu.cpsr.c = true) else {
        return;
    };
    assert_flags!(r, "--C");
}

// ---------------------------------------------------------------------------
// The barrel shifter's special cases
// ---------------------------------------------------------------------------

/// LSR #0 encodes LSR #32, the result is zero and carry is the old bit 31.
#[test]
fn lsr_zero_means_thirty_two() {
    let Some(r) = run_with("movs r1, r0, lsr #32", |cpu, _| {
        cpu.regs[0] = 0x8000_0000;
    }) else {
        return;
    };
    assert_reg!(r, 1, 0);
    assert_flags!(r, "-ZC");
}

/// ASR #0 encodes ASR #32, every bit becomes the sign bit.
#[test]
fn asr_zero_means_thirty_two() {
    let Some(r) = run_with("movs r1, r0, asr #32", |cpu, _| {
        cpu.regs[0] = 0x8000_0000;
    }) else {
        return;
    };
    assert_reg!(r, 1, 0xFFFF_FFFF);
    assert_flags!(r, "N-C");
}

/// ROR #0 encodes RRX, a 33-bit rotate through carry.
#[test]
fn rrx_rotates_through_carry() {
    let Some(r) = run_with("movs r1, r0, rrx", |cpu, _| {
        cpu.regs[0] = 0x0000_0003;
        cpu.cpsr.c = true;
    }) else {
        return;
    };
    assert_reg!(r, 1, 0x8000_0001);
    assert_flags!(r, "N-C");
}

/// a register-specified shift of 32 or more is well defined and differs from
/// the immediate encodings.
#[test]
fn register_shift_of_thirty_two() {
    let Some(r) = run_with("movs r2, r0, lsl r1", |cpu, _| {
        cpu.regs[0] = 0x0000_0001;
        cpu.regs[1] = 32;
    }) else {
        return;
    };
    assert_reg!(r, 2, 0);
    assert_flags!(r, "-ZC");
}

#[test]
fn register_shift_of_zero_preserves_carry() {
    let Some(r) = run_with("movs r2, r0, lsl r1", |cpu, _| {
        cpu.regs[0] = 0x1234;
        cpu.regs[1] = 0;
        cpu.cpsr.c = true;
    }) else {
        return;
    };
    assert_reg!(r, 2, 0x1234);
    assert_flags!(r, "--C");
}

/// reading r15 as an operand gives the instruction address plus 8, and plus 12
/// when the shift amount comes from a register.
#[test]
fn pc_reads_include_the_pipeline_offset() {
    let Some(r) = run("mov r0, pc\n mov r1, #0\n mov r2, pc, lsl r1") else {
        return;
    };
    // first instruction is at CODE_BASE.
    assert_reg!(r, 0, CODE_BASE + 8);
    // third instruction is at CODE_BASE + 8, read as +12.
    assert_reg!(r, 2, CODE_BASE + 8 + 12);
}

// ---------------------------------------------------------------------------
// Multiplies
// ---------------------------------------------------------------------------

#[test]
fn umull_and_smull_differ_in_sign() {
    let Some(r) = run_with("umull r0, r1, r4, r5\n smull r2, r3, r4, r5", |cpu, _| {
        cpu.regs[4] = 0xFFFF_FFFF; // -1
        cpu.regs[5] = 2;
    }) else {
        return;
    };
    // unsigned, 0xFFFFFFFF * 2 = 0x1_FFFFFFFE
    assert_reg!(r, 0, 0xFFFF_FFFE);
    assert_reg!(r, 1, 0x0000_0001);
    // signed, -1 * 2 = -2
    assert_reg!(r, 2, 0xFFFF_FFFE);
    assert_reg!(r, 3, 0xFFFF_FFFF);
}

#[test]
fn umaal_accumulates_both_halves() {
    let Some(r) = run_with("umaal r0, r1, r2, r3", |cpu, _| {
        cpu.regs[0] = 10;
        cpu.regs[1] = 20;
        cpu.regs[2] = 0x1000;
        cpu.regs[3] = 0x1000;
    }) else {
        return;
    };
    // 0x1000 * 0x1000 + 10 + 20 = 0x100_001E
    assert_reg!(r, 0, 0x0100_001E);
    assert_reg!(r, 1, 0);
}

#[test]
fn smlabb_uses_the_bottom_halves() {
    let Some(r) = run_with("smlabb r0, r1, r2, r3", |cpu, _| {
        cpu.regs[1] = 0xDEAD_0003;
        cpu.regs[2] = 0xBEEF_0005;
        cpu.regs[3] = 7;
    }) else {
        return;
    };
    assert_reg!(r, 0, 3 * 5 + 7);
}

#[test]
fn smultt_uses_the_top_halves() {
    let Some(r) = run_with("smultt r0, r1, r2", |cpu, _| {
        cpu.regs[1] = 0x0003_DEAD;
        cpu.regs[2] = 0x0005_BEEF;
    }) else {
        return;
    };
    assert_reg!(r, 0, 15);
}

#[test]
fn smulwb_keeps_the_high_word_of_a_48_bit_product() {
    let Some(r) = run_with("smulwb r0, r1, r2", |cpu, _| {
        cpu.regs[1] = 0x0001_0000;
        cpu.regs[2] = 0x0000_0004;
    }) else {
        return;
    };
    // (0x10000 * 4) >> 16 = 4
    assert_reg!(r, 0, 4);
}

#[test]
fn qadd_saturates_and_sets_q() {
    let Some(r) = run_with("qadd r0, r1, r2", |cpu, _| {
        cpu.regs[1] = 0x7FFF_FFFF;
        cpu.regs[2] = 0x0000_0001;
    }) else {
        return;
    };
    assert_reg!(r, 0, 0x7FFF_FFFF);
    assert_flags!(r, "----Q");
}

#[test]
fn qdadd_doubles_before_adding() {
    let Some(r) = run_with("qdadd r0, r1, r2", |cpu, _| {
        cpu.regs[1] = 5; // added
        cpu.regs[2] = 10; // doubled
    }) else {
        return;
    };
    assert_reg!(r, 0, 25);
}

#[test]
fn clz_counts_leading_zeros() {
    let Some(r) = run_with("clz r0, r1\n clz r2, r3", |cpu, _| {
        cpu.regs[1] = 0x0000_FFFF;
        cpu.regs[3] = 0;
    }) else {
        return;
    };
    assert_reg!(r, 0, 16);
    assert_reg!(r, 2, 32);
}

// ---------------------------------------------------------------------------
// ARMv6 media
// ---------------------------------------------------------------------------

/// RBIT is deliberately absent, it arrived with ARMv6T2 and the MP11 cores
/// do not have it, so the assembler rejects it for this target.
#[test]
fn byte_reversals() {
    let Some(r) = run_with("rev r0, r4\n rev16 r1, r4\n revsh r2, r4", |cpu, _| {
        cpu.regs[4] = 0x1122_33F0;
    }) else {
        return;
    };
    assert_reg!(r, 0, 0xF033_2211);
    assert_reg!(r, 1, 0x2211_F033);
    // REVSH swaps the bottom halfword and sign extends, 0x33F0 -> 0xF033
    assert_reg!(r, 2, 0xFFFF_F033);
}

#[test]
fn extends_with_rotation_and_accumulate() {
    let Some(r) = run_with(
        "sxtb r0, r4\n uxtb r1, r4\n sxth r2, r4\n uxtah r3, r5, r4, ror #16",
        |cpu, _| {
            cpu.regs[4] = 0x1234_80F0;
            cpu.regs[5] = 1;
        },
    ) else {
        return;
    };
    assert_reg!(r, 0, 0xFFFF_FFF0);
    assert_reg!(r, 1, 0x0000_00F0);
    assert_reg!(r, 2, 0xFFFF_80F0);
    // ror #16 gives 0x80F0_1234, low halfword 0x1234, plus 1.
    assert_reg!(r, 3, 0x1235);
}

#[test]
fn pkhbt_and_pkhtb_pick_opposite_halves() {
    let Some(r) = run_with("pkhbt r0, r4, r5\n pkhtb r1, r4, r5", |cpu, _| {
        cpu.regs[4] = 0xAAAA_BBBB;
        cpu.regs[5] = 0xCCCC_DDDD;
    }) else {
        return;
    };
    // PKHBT, bottom of Rn, top of Rm.
    assert_reg!(r, 0, 0xCCCC_BBBB);
    // PKHTB, top of Rn, bottom of Rm.
    assert_reg!(r, 1, 0xAAAA_DDDD);
}

#[test]
fn ssat_and_usat_clamp() {
    let Some(r) = run_with("ssat r0, #8, r4\n usat r1, #8, r5", |cpu, _| {
        cpu.regs[4] = 1000; // clamps to 127
        cpu.regs[5] = 0xFFFF_FFFF; // negative, clamps to 0
    }) else {
        return;
    };
    assert_reg!(r, 0, 127);
    assert_reg!(r, 1, 0);
    assert_flags!(r, "----Q");
}

#[test]
fn sel_uses_the_ge_flags() {
    // SADD8 sets GE per byte lane, then SEL picks from Rn where GE is set.
    let Some(r) = run_with("uadd8 r0, r4, r5\n sel r1, r6, r7", |cpu, _| {
        cpu.regs[4] = 0x00FF_00FF;
        cpu.regs[5] = 0x0001_0001;
        cpu.regs[6] = 0xAABB_CCDD;
        cpu.regs[7] = 0x1122_3344;
    }) else {
        return;
    };
    // lanes 0 and 2 overflow (0xFF + 1), so GE = 0b0101.
    assert_reg!(r, 0, 0x0000_0000);
    assert_reg!(r, 1, 0x1122_3344 & 0xFF00_FF00 | (0xAABB_CCDD & 0x00FF_00FF));
}

#[test]
fn parallel_halfword_add_and_saturating_variants() {
    let Some(r) = run_with("sadd16 r0, r4, r5\n qadd16 r1, r6, r7", |cpu, _| {
        cpu.regs[4] = 0x0001_0002;
        cpu.regs[5] = 0x0003_0004;
        cpu.regs[6] = 0x7FFF_8000;
        cpu.regs[7] = 0x0001_8000;
    }) else {
        return;
    };
    assert_reg!(r, 0, 0x0004_0006);
    // top lane saturates high, bottom lane saturates low.
    assert_reg!(r, 1, 0x7FFF_8000);
}

#[test]
fn asx_and_sax_cross_the_halves() {
    let Some(r) = run_with("sasx r0, r4, r5\n ssax r1, r4, r5", |cpu, _| {
        cpu.regs[4] = 0x000A_0014; // hi = 10, lo = 20
        cpu.regs[5] = 0x0003_0002; // hi = 3,  lo = 2
    }) else {
        return;
    };
    // SASX, low = lo(Rn) - hi(Rm) = 20 - 3 = 17, high = hi(Rn) + lo(Rm) = 12.
    assert_reg!(r, 0, 0x000C_0011);
    // SSAX, low = lo(Rn) + hi(Rm) = 23, high = hi(Rn) - lo(Rm) = 8.
    assert_reg!(r, 1, 0x0008_0017);
}

#[test]
fn smuad_and_smmul() {
    let Some(r) = run_with("smuad r0, r4, r5\n smmul r1, r6, r7", |cpu, _| {
        cpu.regs[4] = 0x0002_0003;
        cpu.regs[5] = 0x0004_0005;
        cpu.regs[6] = 0x4000_0000;
        cpu.regs[7] = 0x0000_0004;
    }) else {
        return;
    };
    // 3*5 + 2*4 = 23
    assert_reg!(r, 0, 23);
    // (0x40000000 * 4) >> 32 = 1
    assert_reg!(r, 1, 1);
}

#[test]
fn usad8_sums_absolute_differences() {
    let Some(r) = run_with("usad8 r0, r4, r5", |cpu, _| {
        cpu.regs[4] = 0x0A_14_1E_28;
        cpu.regs[5] = 0x01_02_03_04;
    }) else {
        return;
    };
    assert_reg!(r, 0, (0x0A - 1) + (0x14 - 2) + (0x1E - 3) + (0x28 - 4));
}

// ---------------------------------------------------------------------------
// Loads and stores
// ---------------------------------------------------------------------------

#[test]
fn post_indexed_load_writes_the_base_back() {
    let Some(r) = run_with("ldr r0, [r1], #4", |cpu, bus| {
        cpu.regs[1] = DATA_BASE;
        bus.write_bytes(DATA_BASE, &0xDEAD_BEEFu32.to_le_bytes());
    }) else {
        return;
    };
    assert_reg!(r, 0, 0xDEAD_BEEF);
    assert_reg!(r, 1, DATA_BASE + 4);
}

/// LDM where the base is also in the register list must keep the loaded
/// value rather than the written-back base.
#[test]
fn ldm_into_its_own_base_keeps_the_loaded_value() {
    let Some(r) = run_with("ldmia r1!, {r1, r2}", |cpu, bus| {
        cpu.regs[1] = DATA_BASE;
        bus.write_bytes(DATA_BASE, &0x1234_5678u32.to_le_bytes());
        bus.write_bytes(DATA_BASE + 4, &0x9ABC_DEF0u32.to_le_bytes());
    }) else {
        return;
    };
    assert_reg!(r, 1, 0x1234_5678);
    assert_reg!(r, 2, 0x9ABC_DEF0);
}

#[test]
fn scaled_register_offset() {
    let Some(r) = run_with("ldr r0, [r1, r2, lsl #2]", |cpu, bus| {
        cpu.regs[1] = DATA_BASE;
        cpu.regs[2] = 3;
        bus.write_bytes(DATA_BASE + 12, &0xCAFE_0000u32.to_le_bytes());
    }) else {
        return;
    };
    assert_reg!(r, 0, 0xCAFE_0000);
}

#[test]
fn halfword_and_signed_loads() {
    let Some(r) = run_with(
        "ldrh r0, [r4]\n ldrsh r1, [r4]\n ldrsb r2, [r4]\n ldrb r3, [r4]",
        |cpu, bus| {
            cpu.regs[4] = DATA_BASE;
            bus.write_bytes(DATA_BASE, &0x0000_80F0u32.to_le_bytes());
        },
    ) else {
        return;
    };
    assert_reg!(r, 0, 0x80F0);
    assert_reg!(r, 1, 0xFFFF_80F0);
    assert_reg!(r, 2, 0xFFFF_FFF0);
    assert_reg!(r, 3, 0xF0);
}

#[test]
fn doubleword_round_trip() {
    let Some(r) = run_with("strd r0, r1, [r4]\n ldrd r2, r3, [r4]", |cpu, _| {
        cpu.regs[0] = 0x1111_1111;
        cpu.regs[1] = 0x2222_2222;
        cpu.regs[4] = DATA_BASE;
    }) else {
        return;
    };
    assert_reg!(r, 2, 0x1111_1111);
    assert_reg!(r, 3, 0x2222_2222);
}

#[test]
fn block_transfer_addressing_modes() {
    // all four modes write the same three words, only the base ends up
    // different, and only DB/IB shift where the first word lands.
    let Some(r) = run_with(
        "stmia r4!, {r0, r1, r2}\n
         ldmdb r4!, {r5, r6, r7}",
        |cpu, _| {
            cpu.regs[0] = 0xA;
            cpu.regs[1] = 0xB;
            cpu.regs[2] = 0xC;
            cpu.regs[4] = DATA_BASE;
        },
    ) else {
        return;
    };
    assert_reg!(r, 5, 0xA);
    assert_reg!(r, 6, 0xB);
    assert_reg!(r, 7, 0xC);
    assert_reg!(r, 4, DATA_BASE);
}

#[test]
fn push_and_pop_round_trip() {
    let Some(r) = run_with("push {r0, r1, r2}\n pop {r3, r4, r5}", |cpu, _| {
        cpu.regs[0] = 1;
        cpu.regs[1] = 2;
        cpu.regs[2] = 3;
    }) else {
        return;
    };
    assert_reg!(r, 3, 1);
    assert_reg!(r, 4, 2);
    assert_reg!(r, 5, 3);
    assert_reg!(r, 13, common::STACK_TOP);
}

#[test]
fn swap_exchanges_memory_and_register() {
    let Some(r) = run_with("swp r0, r1, [r4]", |cpu, bus| {
        cpu.regs[1] = 0x5555_5555;
        cpu.regs[4] = DATA_BASE;
        bus.write_bytes(DATA_BASE, &0xAAAA_AAAAu32.to_le_bytes());
    }) else {
        return;
    };
    assert_reg!(r, 0, 0xAAAA_AAAA);
    assert_eq!(r.bus.read_u32(DATA_BASE), 0x5555_5555);
}

#[test]
fn strex_succeeds_only_after_ldrex() {
    let Some(r) = run_with(
        "strex r0, r5, [r4]\n
         ldrex r1, [r4]\n
         strex r2, r6, [r4]",
        |cpu, bus| {
            cpu.regs[4] = DATA_BASE;
            cpu.regs[5] = 0x1111_1111;
            cpu.regs[6] = 0x2222_2222;
            bus.write_bytes(DATA_BASE, &0u32.to_le_bytes());
        },
    ) else {
        return;
    };
    // the first store has no reservation and must fail.
    assert_reg!(r, 0, 1);
    assert_reg!(r, 1, 0);
    assert_reg!(r, 2, 0);
    assert_eq!(r.bus.read_u32(DATA_BASE), 0x2222_2222);
}

// ---------------------------------------------------------------------------
// Branches
// ---------------------------------------------------------------------------

#[test]
fn bl_sets_the_link_register() {
    let Some(r) = run("
        bl target
        b end
    target:
        mov r0, #1
    end:
        nop") else {
        return;
    };
    assert_reg!(r, 0, 1);
    // LR points just past the BL, which is the second instruction.
    assert_reg!(r, 14, CODE_BASE + 4);
}

#[test]
fn bx_switches_to_thumb() {
    let Some(r) = run("
        adr r0, thumb_code
        add r0, r0, #1
        bx r0
        .thumb
    thumb_code:
        movs r1, #42") else {
        return;
    };
    assert_reg!(r, 1, 42);
    assert!(r.cpu.cpsr.thumb, "should have switched to Thumb state");
}

#[test]
fn conditional_execution_skips() {
    let Some(r) = run("
        mov r0, #0
        cmp r0, #0
        movne r1, #1
        moveq r2, #2") else {
        return;
    };
    assert_reg!(r, 1, 0);
    assert_reg!(r, 2, 2);
}

#[test]
fn svc_reports_its_immediate() {
    let Some(r) = run("svc #0x32") else { return };
    assert_eq!(r.exit, zakuro_cpu::Exit::Supervisor(0x32));
    // the PC must be left pointing at the instruction after the SVC so the
    // kernel can simply resume.
    assert_eq!(r.cpu.regs[15], CODE_BASE + 4);
}

/// from ARMv5 onwards a load into the PC interworks.
#[test]
fn ldr_into_pc_interworks() {
    // the Thumb instruction sits immediately after the load, and the address
    // is planted by the test rather than by a relocation, so this exercises
    // the load itself and not the assembler's Thumb-bit handling.
    let thumb_target = CODE_BASE + 4;
    let Some(r) = run_with(
        "ldr pc, [r0]
         .thumb
         movs r1, #77",
        |cpu, bus| {
            cpu.regs[0] = DATA_BASE;
            bus.write_bytes(DATA_BASE, &(thumb_target | 1).to_le_bytes());
        },
    ) else {
        return;
    };
    assert!(r.cpu.cpsr.thumb, "should have switched to Thumb state");
    assert_reg!(r, 1, 77);
}

/// the same for LDM, which titles use to return from a function and switch
/// state in one instruction.
#[test]
fn ldm_into_pc_interworks() {
    let Some(r) = run("
         adr r0, thumb_target
         add r0, r0, #1
         push {r0}
         ldmia sp!, {pc}
         .thumb
        thumb_target:
         movs r1, #9") else {
        return;
    };
    assert!(r.cpu.cpsr.thumb, "should have switched to Thumb state");
    assert_reg!(r, 1, 9);
}
