//! bitfield helpers

/// bits hi..=lo of value, inclusive
#[inline(always)]
pub const fn bits(value: u32, lo: u32, hi: u32) -> u32 {
    (value >> lo) & ((1u32 << (hi - lo + 1)).wrapping_sub(1))
}

/// one bit as a bool
#[inline(always)]
pub const fn bit(value: u32, n: u32) -> bool {
    (value >> n) & 1 != 0
}

/// sign extends the low n bits of value to a full i32
#[inline(always)]
pub const fn sign_extend(value: u32, n: u32) -> i32 {
    let shift = 32 - n;
    ((value << shift) as i32) >> shift
}

/// rotate right, ARM style for a rotate of 0
#[inline(always)]
pub const fn ror32(value: u32, amount: u32) -> u32 {
    value.rotate_right(amount & 31)
}

/// rounds value up to a multiple of align (power of two)
#[inline(always)]
pub const fn align_up(value: u32, align: u32) -> u32 {
    (value + align - 1) & !(align - 1)
}

/// rounds value down to a multiple of align (power of two)
#[inline(always)]
pub const fn align_down(value: u32, align: u32) -> u32 {
    value & !(align - 1)
}
