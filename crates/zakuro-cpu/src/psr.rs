//! program status register.

use zakuro_common::bits::bit;

/// ARM processor modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Mode {
    User = 0b10000,
    Fiq = 0b10001,
    Irq = 0b10010,
    Supervisor = 0b10011,
    Abort = 0b10111,
    Undefined = 0b11011,
    System = 0b11111,
}

impl Mode {
    pub fn from_bits(bits: u32) -> Mode {
        match bits & 0x1F {
            0b10001 => Mode::Fiq,
            0b10010 => Mode::Irq,
            0b10011 => Mode::Supervisor,
            0b10111 => Mode::Abort,
            0b11011 => Mode::Undefined,
            0b11111 => Mode::System,
            _ => Mode::User,
        }
    }

    pub fn is_privileged(self) -> bool {
        self != Mode::User
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Psr {
    pub n: bool,
    pub z: bool,
    pub c: bool,
    pub v: bool,
    /// sticky saturation flag, set by the DSP and saturating-arithmetic ops.
    pub q: bool,
    /// ARMv6 SIMD "greater than or equal" flags, one per byte lane.
    pub ge: u8,
    pub e: bool,
    pub a: bool,
    pub i: bool,
    pub f: bool,
    pub thumb: bool,
    pub mode: Mode,
}

impl Default for Psr {
    fn default() -> Self {
        Psr {
            n: false,
            z: false,
            c: false,
            v: false,
            q: false,
            ge: 0,
            e: false,
            a: false,
            i: false,
            f: false,
            thumb: false,
            mode: Mode::System,
        }
    }
}

impl Psr {
    pub fn to_bits(self) -> u32 {
        (self.n as u32) << 31
            | (self.z as u32) << 30
            | (self.c as u32) << 29
            | (self.v as u32) << 28
            | (self.q as u32) << 27
            | (self.ge as u32 & 0xF) << 16
            | (self.e as u32) << 9
            | (self.a as u32) << 8
            | (self.i as u32) << 7
            | (self.f as u32) << 6
            | (self.thumb as u32) << 5
            | self.mode as u32
    }

    pub fn from_bits(value: u32) -> Psr {
        Psr {
            n: bit(value, 31),
            z: bit(value, 30),
            c: bit(value, 29),
            v: bit(value, 28),
            q: bit(value, 27),
            ge: ((value >> 16) & 0xF) as u8,
            e: bit(value, 9),
            a: bit(value, 8),
            i: bit(value, 7),
            f: bit(value, 6),
            thumb: bit(value, 5),
            mode: Mode::from_bits(value),
        }
    }

    /// applies a write from MSR, honouring the field mask and dropping
    /// privileged fields when in User mode.
    pub fn write_masked(&mut self, value: u32, field_mask: u32, privileged: bool) {
        // bit 3 of the mask, flags (31:24), bit 2, status (23:16),
        // bit 1, extension (15:8), bit 0, control (7:0).
        if field_mask & 0b1000 != 0 {
            self.n = bit(value, 31);
            self.z = bit(value, 30);
            self.c = bit(value, 29);
            self.v = bit(value, 28);
            self.q = bit(value, 27);
        }
        if field_mask & 0b0100 != 0 && privileged {
            self.ge = ((value >> 16) & 0xF) as u8;
        }
        if field_mask & 0b0010 != 0 && privileged {
            self.e = bit(value, 9);
            self.a = bit(value, 8);
        }
        if field_mask & 0b0001 != 0 && privileged {
            self.i = bit(value, 7);
            self.f = bit(value, 6);
            // the T bit is never written directly by MSR.
            self.mode = Mode::from_bits(value);
        }
    }
}

/// the four condition-code bits an instruction's condition field is tested
/// against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Condition {
    Eq,
    Ne,
    Cs,
    Cc,
    Mi,
    Pl,
    Vs,
    Vc,
    Hi,
    Ls,
    Ge,
    Lt,
    Gt,
    Le,
    Al,
    /// 0b1111, on ARMv5 and later this encodes unconditional instructions
    /// rather than "never".
    Nv,
}

impl Condition {
    #[inline(always)]
    pub fn from_bits(bits: u32) -> Condition {
        match bits & 0xF {
            0x0 => Condition::Eq,
            0x1 => Condition::Ne,
            0x2 => Condition::Cs,
            0x3 => Condition::Cc,
            0x4 => Condition::Mi,
            0x5 => Condition::Pl,
            0x6 => Condition::Vs,
            0x7 => Condition::Vc,
            0x8 => Condition::Hi,
            0x9 => Condition::Ls,
            0xA => Condition::Ge,
            0xB => Condition::Lt,
            0xC => Condition::Gt,
            0xD => Condition::Le,
            0xE => Condition::Al,
            _ => Condition::Nv,
        }
    }
}

#[inline(always)]
pub fn check_condition(cond: u32, psr: &Psr) -> bool {
    match cond & 0xF {
        0x0 => psr.z,
        0x1 => !psr.z,
        0x2 => psr.c,
        0x3 => !psr.c,
        0x4 => psr.n,
        0x5 => !psr.n,
        0x6 => psr.v,
        0x7 => !psr.v,
        0x8 => psr.c && !psr.z,
        0x9 => !psr.c || psr.z,
        0xA => psr.n == psr.v,
        0xB => psr.n != psr.v,
        0xC => !psr.z && (psr.n == psr.v),
        0xD => psr.z || (psr.n != psr.v),
        _ => true,
    }
}
