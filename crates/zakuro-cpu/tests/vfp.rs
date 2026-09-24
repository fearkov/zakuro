//! VFPv2 tests.

mod common;

use common::{assert_flags, assert_reg, run_with, DATA_BASE};

/// seeds single-precision registers before the program runs.
fn with_singles(source: &str, values: &[(usize, f32)]) -> Option<common::Run> {
    let values = values.to_vec();
    run_with(source, move |cpu, _| {
        for (index, value) in values {
            cpu.vfp.set_f32(index, value);
        }
    })
}

#[test]
fn single_precision_arithmetic() {
    let Some(r) = with_singles(
        "vadd.f32 s4, s0, s1
         vsub.f32 s5, s0, s1
         vmul.f32 s6, s0, s1
         vdiv.f32 s7, s0, s1",
        &[(0, 10.0), (1, 4.0)],
    ) else {
        return;
    };
    assert_eq!(r.f32_reg(4), 14.0);
    assert_eq!(r.f32_reg(5), 6.0);
    assert_eq!(r.f32_reg(6), 40.0);
    assert_eq!(r.f32_reg(7), 2.5);
}

#[test]
fn double_precision_arithmetic() {
    let Some(r) = run_with(
        "vadd.f64 d4, d0, d1
         vdiv.f64 d5, d0, d1",
        |cpu, _| {
            cpu.vfp.set_f64(0, 1.5);
            cpu.vfp.set_f64(1, 0.5);
        },
    ) else {
        return;
    };
    assert_eq!(r.f64_reg(4), 2.0);
    assert_eq!(r.f64_reg(5), 3.0);
}

#[test]
fn multiply_accumulate_variants() {
    let Some(r) = with_singles(
        "vmla.f32 s4, s0, s1
         vmls.f32 s5, s0, s1
         vnmul.f32 s6, s0, s1",
        &[(0, 3.0), (1, 4.0), (4, 100.0), (5, 100.0)],
    ) else {
        return;
    };
    assert_eq!(r.f32_reg(4), 112.0);
    assert_eq!(r.f32_reg(5), 88.0);
    assert_eq!(r.f32_reg(6), -12.0);
}

#[test]
fn unary_operations() {
    let Some(r) = with_singles(
        "vabs.f32 s4, s0
         vneg.f32 s5, s0
         vsqrt.f32 s6, s1
         vmov.f32 s7, s1",
        &[(0, -3.5), (1, 16.0)],
    ) else {
        return;
    };
    assert_eq!(r.f32_reg(4), 3.5);
    assert_eq!(r.f32_reg(5), 3.5);
    assert_eq!(r.f32_reg(6), 4.0);
    assert_eq!(r.f32_reg(7), 16.0);
}

/// VCMP writes FPSCR, and VMRS APSR_nzcv copies those bits into the ARM
/// condition flags so ordinary conditional instructions can branch on them.
#[test]
fn compare_then_transfer_flags_equal() {
    let Some(r) = with_singles(
        "vcmp.f32 s0, s1
         vmrs APSR_nzcv, fpscr",
        &[(0, 1.0), (1, 1.0)],
    ) else {
        return;
    };
    assert_flags!(r, "-ZC");
}

#[test]
fn compare_then_transfer_flags_less_than() {
    let Some(r) = with_singles(
        "vcmp.f32 s0, s1
         vmrs APSR_nzcv, fpscr",
        &[(0, 1.0), (1, 2.0)],
    ) else {
        return;
    };
    assert_flags!(r, "N");
}

#[test]
fn compare_then_transfer_flags_greater_than() {
    let Some(r) = with_singles(
        "vcmp.f32 s0, s1
         vmrs APSR_nzcv, fpscr",
        &[(0, 3.0), (1, 2.0)],
    ) else {
        return;
    };
    assert_flags!(r, "--C");
}

#[test]
fn compare_with_nan_is_unordered() {
    let Some(r) = with_singles(
        "vcmp.f32 s0, s1
         vmrs APSR_nzcv, fpscr",
        &[(0, f32::NAN), (1, 1.0)],
    ) else {
        return;
    };
    assert_flags!(r, "--CV");
}

#[test]
fn compare_against_zero() {
    let Some(r) = with_singles(
        "vcmp.f32 s0, #0
         vmrs APSR_nzcv, fpscr",
        &[(0, 0.0)],
    ) else {
        return;
    };
    assert_flags!(r, "-ZC");
}

#[test]
fn conversions_between_integers_and_floats() {
    let Some(r) = with_singles(
        "vcvt.s32.f32 s4, s0
         vcvt.u32.f32 s5, s1
         vcvt.f32.s32 s6, s2",
        // s2 holds the bit pattern of the integer -7.
        &[(0, -2.75), (1, 300.9)],
    )
    .and_then(|_| {
        run_with(
            "vcvt.s32.f32 s4, s0
             vcvt.u32.f32 s5, s1
             vcvt.f32.s32 s6, s2",
            |cpu, _| {
                cpu.vfp.set_f32(0, -2.75);
                cpu.vfp.set_f32(1, 300.9);
                cpu.vfp.regs[2] = (-7i32) as u32;
            },
        )
    }) else {
        return;
    };
    // VCVT to integer rounds towards zero.
    assert_eq!(r.cpu.vfp.regs[4] as i32, -2);
    assert_eq!(r.cpu.vfp.regs[5], 300);
    assert_eq!(r.f32_reg(6), -7.0);
}

#[test]
fn conversions_between_single_and_double() {
    // s12 deliberately does not overlap d2 (which aliases s4 and s5).
    let Some(r) = run_with(
        "vcvt.f64.f32 d2, s0
         vcvt.f32.f64 s12, d3",
        |cpu, _| {
            cpu.vfp.set_f32(0, 1.5);
            cpu.vfp.set_f64(3, -0.25);
        },
    ) else {
        return;
    };
    assert_eq!(r.f64_reg(2), 1.5);
    assert_eq!(r.f32_reg(12), -0.25);
}

#[test]
fn load_and_store_single() {
    let Some(r) = run_with("vstr s0, [r4]\n vldr s1, [r4]", |cpu, _| {
        cpu.vfp.set_f32(0, 12.25);
        cpu.regs[4] = DATA_BASE;
    }) else {
        return;
    };
    assert_eq!(r.f32_reg(1), 12.25);
    assert_eq!(r.bus.read_u32(DATA_BASE), 12.25f32.to_bits());
}

#[test]
fn load_and_store_double_with_offset() {
    let Some(r) = run_with("vstr d0, [r4, #8]\n vldr d1, [r4, #8]", |cpu, _| {
        cpu.vfp.set_f64(0, -1.0e10);
        cpu.regs[4] = DATA_BASE;
    }) else {
        return;
    };
    assert_eq!(r.f64_reg(1), -1.0e10);
}

#[test]
fn block_transfers_round_trip() {
    let Some(r) = run_with(
        "vstmia r4!, {s0-s3}
         sub r4, r4, #16
         vldmia r4!, {s8-s11}",
        |cpu, _| {
            cpu.regs[4] = DATA_BASE;
            for i in 0..4 {
                cpu.vfp.set_f32(i, (i as f32) + 0.5);
            }
        },
    ) else {
        return;
    };
    for i in 0..4 {
        assert_eq!(r.f32_reg(8 + i), (i as f32) + 0.5, "s{}", 8 + i);
    }
    assert_reg!(r, 4, DATA_BASE + 16);
}

#[test]
fn push_and_pop_doubles() {
    let Some(r) = run_with("vpush {d0-d1}\n vpop {d4-d5}", |cpu, _| {
        cpu.vfp.set_f64(0, 3.25);
        cpu.vfp.set_f64(1, -7.5);
    }) else {
        return;
    };
    assert_eq!(r.f64_reg(4), 3.25);
    assert_eq!(r.f64_reg(5), -7.5);
    assert_reg!(r, 13, common::STACK_TOP);
}

#[test]
fn move_between_core_and_vfp_registers() {
    let Some(r) = run_with(
        "vmov s0, r0
         vmov r1, s0
         vmov d2, r2, r3
         vmov r4, r5, d2",
        |cpu, _| {
            cpu.regs[0] = 0x4048_0000; // 3.125f
            cpu.regs[2] = 0x1111_1111;
            cpu.regs[3] = 0x2222_2222;
        },
    ) else {
        return;
    };
    assert_eq!(r.f32_reg(0), 3.125);
    assert_reg!(r, 1, 0x4048_0000);
    assert_reg!(r, 4, 0x1111_1111);
    assert_reg!(r, 5, 0x2222_2222);
}

/// the single and double register files alias, d1 covers s2 and s3.
#[test]
fn single_and_double_registers_alias() {
    let Some(r) = run_with("vmov s2, r0\n vmov s3, r1", |cpu, _| {
        cpu.regs[0] = 0x0000_0000;
        cpu.regs[1] = 0x3FF0_0000; // 1.0 as the high word of a double
    }) else {
        return;
    };
    assert_eq!(r.f64_reg(1), 1.0);
}

#[test]
fn fpscr_round_trips_through_vmsr() {
    let Some(r) = run_with("vmsr fpscr, r0\n vmrs r1, fpscr", |cpu, _| {
        cpu.regs[0] = 0x00C0_0000; // round towards zero
    }) else {
        return;
    };
    assert_reg!(r, 1, 0x00C0_0000);
}

/// short vectors, with FPSCR.LEN = 3 (four elements) a single vadd whose
/// destination is outside bank 0 adds four register pairs.
#[test]
fn short_vector_add_covers_four_registers() {
    let Some(r) = with_singles(
        "vmrs r0, fpscr
         orr r0, r0, #0x30000
         vmsr fpscr, r0
         vadd.f32 s8, s16, s24",
        &[
            (16, 1.0), (17, 2.0), (18, 3.0), (19, 4.0),
            (24, 10.0), (25, 20.0), (26, 30.0), (27, 40.0),
        ],
    ) else {
        return;
    };
    assert_eq!(
        [r.f32_reg(8), r.f32_reg(9), r.f32_reg(10), r.f32_reg(11)],
        [11.0, 22.0, 33.0, 44.0]
    );
}

/// a second operand in bank 0 is a scalar applied to every element, and a
/// destination in bank 0 makes the whole operation scalar.
#[test]
fn short_vectors_mix_with_scalars() {
    let Some(r) = with_singles(
        "vmrs r0, fpscr
         orr r0, r0, #0x30000
         vmsr fpscr, r0
         vmul.f32 s8, s16, s0
         vmul.f32 s4, s16, s0",
        &[(0, 2.0), (16, 1.0), (17, 2.0), (18, 3.0), (19, 4.0), (5, -1.0)],
    ) else {
        return;
    };
    assert_eq!(
        [r.f32_reg(8), r.f32_reg(9), r.f32_reg(10), r.f32_reg(11)],
        [2.0, 4.0, 6.0, 8.0]
    );
    assert_eq!(r.f32_reg(4), 2.0);
    assert_eq!(r.f32_reg(5), -1.0, "a bank 0 destination is scalar");
}

/// elements wrap around within their bank, and a stride of two skips
/// every other register.
#[test]
fn short_vectors_wrap_within_their_bank() {
    let Some(r) = with_singles(
        "vmrs r0, fpscr
         orr r0, r0, #0x310000
         vmsr fpscr, r0
         vmov.f32 s14, s22",
        &[(22, 1.0), (16, 2.0)],
    ) else {
        return;
    };
    // length 2, stride 2, s14 <- s22, then s8 (14 + 2 wraps to 8) <- s16.
    assert_eq!(r.f32_reg(14), 1.0);
    assert_eq!(r.f32_reg(8), 2.0);
}
