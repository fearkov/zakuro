//! arithmetic primitives shared by the ARM and Thumb decoders.

#[inline(always)]
pub fn add_with_flags(a: u32, b: u32) -> (u32, bool, bool) {
    let (result, carry) = a.overflowing_add(b);
    let overflow = ((a ^ result) & (b ^ result)) >> 31 != 0;
    (result, carry, overflow)
}

#[inline(always)]
pub fn adc_with_flags(a: u32, b: u32, carry_in: bool) -> (u32, bool, bool) {
    let sum = a as u64 + b as u64 + carry_in as u64;
    let result = sum as u32;
    let carry = sum > 0xFFFF_FFFF;
    let overflow = ((a ^ result) & (b ^ result)) >> 31 != 0;
    (result, carry, overflow)
}

/// ARM subtraction, the carry flag means "no borrow".
#[inline(always)]
pub fn sub_with_flags(a: u32, b: u32) -> (u32, bool, bool) {
    let result = a.wrapping_sub(b);
    let carry = a >= b;
    let overflow = ((a ^ b) & (a ^ result)) >> 31 != 0;
    (result, carry, overflow)
}

#[inline(always)]
pub fn sbc_with_flags(a: u32, b: u32, carry_in: bool) -> (u32, bool, bool) {
    let borrow = !carry_in as u64;
    let diff = (a as u64).wrapping_sub(b as u64).wrapping_sub(borrow);
    let result = diff as u32;
    let carry = (a as u64) >= (b as u64 + borrow);
    let overflow = ((a ^ b) & (a ^ result)) >> 31 != 0;
    (result, carry, overflow)
}

/// signed saturation to 32 bits, used by QADD and friends.
#[inline(always)]
pub fn saturate_i32(value: i64) -> (u32, bool) {
    if value > i32::MAX as i64 {
        (i32::MAX as u32, true)
    } else if value < i32::MIN as i64 {
        (i32::MIN as u32, true)
    } else {
        (value as u32, false)
    }
}

/// signed saturation to bits bits (SSAT).
#[inline(always)]
pub fn signed_saturate(value: i64, bits: u32) -> (u32, bool) {
    let max = (1i64 << (bits - 1)) - 1;
    let min = -(1i64 << (bits - 1));
    if value > max {
        (max as u32, true)
    } else if value < min {
        (min as u32, true)
    } else {
        (value as u32, false)
    }
}

/// unsigned saturation to bits bits (USAT).
#[inline(always)]
pub fn unsigned_saturate(value: i64, bits: u32) -> (u32, bool) {
    let max = if bits >= 32 {
        u32::MAX as i64
    } else {
        (1i64 << bits) - 1
    };
    if value > max {
        (max as u32, true)
    } else if value < 0 {
        (0, true)
    } else {
        (value as u32, false)
    }
}
