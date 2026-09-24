//! thumb-state instruction tests.

mod common;

use common::{assert_flags, assert_reg, run_thumb, run_thumb_with, CODE_BASE, DATA_BASE};

#[test]
fn shift_by_immediate_sets_carry() {
    let Some(r) = run_thumb_with("lsls r1, r0, #1", |cpu, _| {
        cpu.regs[0] = 0x8000_0000;
    }) else {
        return;
    };
    assert_reg!(r, 1, 0);
    assert_flags!(r, "-ZC");
}

#[test]
fn add_and_subtract_three_bit_forms() {
    let Some(r) = run_thumb("movs r0, #10\n adds r1, r0, #3\n subs r2, r1, r0") else {
        return;
    };
    assert_reg!(r, 1, 13);
    assert_reg!(r, 2, 3);
}

#[test]
fn move_compare_add_subtract_immediate() {
    let Some(r) = run_thumb("movs r0, #200\n adds r0, #55\n subs r0, #5\n cmp r0, #250") else {
        return;
    };
    assert_reg!(r, 0, 250);
    assert_flags!(r, "-ZC");
}

#[test]
fn alu_operations() {
    let Some(r) = run_thumb_with(
        "ands r0, r4\n eors r1, r4\n orrs r2, r4\n bics r3, r4\n mvns r5, r4",
        |cpu, _| {
            cpu.regs[0] = 0xFF00_FF00;
            cpu.regs[1] = 0xFF00_FF00;
            cpu.regs[2] = 0x0000_00FF;
            cpu.regs[3] = 0xFFFF_FFFF;
            cpu.regs[4] = 0x0F0F_0F0F;
        },
    ) else {
        return;
    };
    assert_reg!(r, 0, 0x0F00_0F00);
    assert_reg!(r, 1, 0xF00F_F00F);
    assert_reg!(r, 2, 0x0F0F_0FFF);
    assert_reg!(r, 3, 0xF0F0_F0F0);
    assert_reg!(r, 5, 0xF0F0_F0F0);
}

#[test]
fn neg_is_reverse_subtract_from_zero() {
    let Some(r) = run_thumb_with("rsbs r0, r1, #0", |cpu, _| cpu.regs[1] = 5) else {
        return;
    };
    assert_reg!(r, 0, 0xFFFF_FFFB);
}

#[test]
fn multiply_sets_nz_only() {
    let Some(r) = run_thumb_with("muls r0, r1, r0", |cpu, _| {
        cpu.regs[0] = 0xFFFF_FFFF;
        cpu.regs[1] = 2;
        cpu.cpsr.v = true;
    }) else {
        return;
    };
    assert_reg!(r, 0, 0xFFFF_FFFE);
    assert_flags!(r, "N--V");
}

#[test]
fn high_register_operations() {
    let Some(r) = run_thumb_with("mov r8, r0\n add r9, r8\n mov r1, r9", |cpu, _| {
        cpu.regs[0] = 100;
        cpu.regs[9] = 5;
    }) else {
        return;
    };
    assert_reg!(r, 1, 105);
}

#[test]
fn pc_relative_load_uses_the_aligned_pc() {
    // the literal pool entry sits after the code, ldr rN, =value makes the
    // assembler build it for us.
    let Some(r) = run_thumb(
        "ldr r0, .Lpool
         b .Lend
         .align 2
        .Lpool:
         .word 0xCAFEBABE
        .Lend:
         nop",
    ) else {
        return;
    };
    assert_reg!(r, 0, 0xCAFE_BABE);
}

#[test]
fn register_offset_loads_and_stores() {
    let Some(r) = run_thumb_with(
        "str r0, [r4, r5]\n ldr r1, [r4, r5]\n strb r0, [r4, r5]\n ldrb r2, [r4, r5]",
        |cpu, _| {
            cpu.regs[0] = 0x1234_5678;
            cpu.regs[4] = DATA_BASE;
            cpu.regs[5] = 8;
        },
    ) else {
        return;
    };
    assert_reg!(r, 1, 0x1234_5678);
    assert_reg!(r, 2, 0x78);
}

#[test]
fn sign_extended_loads() {
    let Some(r) = run_thumb_with(
        "ldrsb r0, [r4, r5]\n ldrsh r1, [r4, r5]\n ldrh r2, [r4, r5]",
        |cpu, bus| {
            cpu.regs[4] = DATA_BASE;
            cpu.regs[5] = 0;
            bus.write_bytes(DATA_BASE, &0xFFFF_F0F0u32.to_le_bytes());
        },
    ) else {
        return;
    };
    assert_reg!(r, 0, 0xFFFF_FFF0);
    assert_reg!(r, 1, 0xFFFF_F0F0);
    assert_reg!(r, 2, 0x0000_F0F0);
}

#[test]
fn immediate_offsets_scale_by_access_size() {
    let Some(r) = run_thumb_with(
        "str r0, [r4, #4]\n ldr r1, [r4, #4]\n strh r0, [r4, #2]\n ldrh r2, [r4, #2]",
        |cpu, _| {
            cpu.regs[0] = 0xAABB_CCDD;
            cpu.regs[4] = DATA_BASE;
        },
    ) else {
        return;
    };
    assert_reg!(r, 1, 0xAABB_CCDD);
    assert_reg!(r, 2, 0xCCDD);
}

#[test]
fn sp_relative_and_load_address() {
    // adr assembles to the Thumb "add Rd, pc, #imm" form, which reads the
    // word-aligned PC rather than the raw one.
    let Some(r) = run_thumb(
        "add r0, sp, #16
         adr r1, .Ltarget
         str r0, [sp, #4]
         ldr r2, [sp, #4]
         b .Ltarget
         .align 2
        .Ltarget:
         nop",
    ) else {
        return;
    };
    assert_reg!(r, 0, common::STACK_TOP + 16);
    assert_reg!(r, 2, common::STACK_TOP + 16);
    // five 2-byte instructions precede the label, and .align 2 pads the
    // odd halfword, so the target sits at CODE_BASE + 12.
    assert_reg!(r, 1, CODE_BASE + 12);
}

#[test]
fn sp_adjustment() {
    let Some(r) = run_thumb("sub sp, #32\n add sp, #16") else {
        return;
    };
    assert_reg!(r, 13, common::STACK_TOP - 16);
}

#[test]
fn push_pop_with_lr_and_pc() {
    let Some(r) = run_thumb(
        "movs r0, #1
         bl subroutine
         b end
        subroutine:
         push {r0, lr}
         movs r0, #7
         pop {r1, pc}
        end:
         nop",
    ) else {
        return;
    };
    // r1 got the pushed r0 (1), r0 kept the value set inside the subroutine.
    assert_reg!(r, 0, 7);
    assert_reg!(r, 1, 1);
    assert!(r.cpu.cpsr.thumb, "POP {{pc}} must stay in Thumb state");
}

#[test]
fn block_transfer() {
    let Some(r) = run_thumb_with("stmia r4!, {r0, r1}\n subs r4, #8\n ldmia r4!, {r2, r3}", |cpu, _| {
        cpu.regs[0] = 0xAAAA;
        cpu.regs[1] = 0xBBBB;
        cpu.regs[4] = DATA_BASE;
    }) else {
        return;
    };
    assert_reg!(r, 2, 0xAAAA);
    assert_reg!(r, 3, 0xBBBB);
    assert_reg!(r, 4, DATA_BASE + 8);
}

#[test]
fn conditional_and_unconditional_branches() {
    let Some(r) = run_thumb(
        "movs r0, #0
         cmp r0, #0
         bne wrong
         movs r1, #1
         b done
        wrong:
         movs r1, #2
        done:
         nop",
    ) else {
        return;
    };
    assert_reg!(r, 1, 1);
}

#[test]
fn long_branch_with_link() {
    let Some(r) = run_thumb(
        "bl far
         b end
        far:
         movs r0, #99
         bx lr
        end:
         nop",
    ) else {
        return;
    };
    assert_reg!(r, 0, 99);
}

#[test]
fn blx_switches_to_arm() {
    let Some(r) = run_thumb(
        "blx arm_code
         b end
         .align 2
         .arm
         .type arm_code, %function
        arm_code:
         mov r0, #55
         bx lr
         .thumb
        end:
         nop",
    ) else {
        return;
    };
    assert_reg!(r, 0, 55);
    assert!(r.cpu.cpsr.thumb, "should have returned to Thumb state");
}

#[test]
fn armv6_extends_and_reversals() {
    let Some(r) = run_thumb_with(
        "sxtb r0, r4\n uxtb r1, r4\n sxth r2, r4\n rev r3, r4\n rev16 r5, r4\n revsh r6, r4",
        |cpu, _| cpu.regs[4] = 0x1122_33F0,
    ) else {
        return;
    };
    assert_reg!(r, 0, 0xFFFF_FFF0);
    assert_reg!(r, 1, 0xF0);
    assert_reg!(r, 2, 0x0000_33F0);
    assert_reg!(r, 3, 0xF033_2211);
    assert_reg!(r, 5, 0x2211_F033);
    assert_reg!(r, 6, 0xFFFF_F033);
}

#[test]
fn svc_from_thumb() {
    let Some(r) = run_thumb("svc #0x2A") else { return };
    assert_eq!(r.exit, zakuro_cpu::Exit::Supervisor(0x2A));
    assert_eq!(r.cpu.regs[15], CODE_BASE + 2);
}

/// a hand-encoded BLX halfword pair.
#[test]
fn blx_halfword_pair() {
    let Some(r) = run_thumb(
        "nop
         nop
        .short 0xF000
        .short 0xE804
         nop
         nop
         nop
         nop
         .arm
         .align 2
        target:
         mov r0, #55",
    ) else {
        return;
    };
    assert!(!r.cpu.cpsr.thumb, "BLX must land in ARM state");
    assert_reg!(r, 0, 55);
    // LR points just past the pair, with the Thumb bit set so a bx lr
    // returns to Thumb state.
    assert_reg!(r, 14, (CODE_BASE + 8) | 1);
}
