//! ARM11 MPCore (ARMv6K) CPU emulation

pub mod arm;
pub mod cp15;
pub mod psr;
pub mod thumb;
pub mod vfp;

pub use cp15::Cp15;
pub use psr::{check_condition, Mode, Psr};
pub use vfp::Vfp;

/// everything the CPU can reach outside itself
pub trait Bus {
    fn read8(&mut self, addr: u32) -> u8;
    fn read16(&mut self, addr: u32) -> u16;
    fn read32(&mut self, addr: u32) -> u32;

    fn write8(&mut self, addr: u32, value: u8);
    fn write16(&mut self, addr: u32, value: u16);
    fn write32(&mut self, addr: u32, value: u32);

    /// instruction fetch
    #[inline(always)]
    fn fetch32(&mut self, addr: u32) -> u32 {
        self.read32(addr)
    }

    #[inline(always)]
    fn fetch16(&mut self, addr: u32) -> u16 {
        self.read16(addr)
    }
}

/// why [Cpu::run] returned
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exit {
    /// the cycle budget ran out, nothing special happened
    Timeout,
    /// an svc #imm was executed
    Supervisor(u32),
    /// an instruction the CPU doesn't implement
    Undefined { pc: u32, opcode: u32 },
    /// bkpt executed
    Breakpoint { pc: u32, imm: u32 },
    /// [Cpu::halt] was called from inside an instruction handler
    Halted,
}

pub const PC: usize = 15;
pub const LR: usize = 14;
pub const SP: usize = 13;

/// all of it kept in a save state
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Cpu {
    /// r0 to r15 for the current mode
    pub regs: [u32; 16],
    pub cpsr: Psr,
    pub spsr: Psr,

    /// banked r13/r14 for the privileged modes, plus the FIQ r8 to r12 bank
    banked_sp_lr: [[u32; 2]; 6],
    banked_spsr: [Psr; 6],
    fiq_r8_r12: [u32; 5],
    /// user mode r8 to r12, parked here while the core's in FIQ mode
    user_r8_r12: [u32; 5],

    pub vfp: Vfp,
    pub cp15: Cp15,

    /// addr reserved by LDREX, or None
    pub exclusive_addr: Option<u32>,

    /// set by a handler when it writes r15, so the fetch loop knows not to advance the PC itself
    branched: bool,
    halted: bool,

    /// free running instruction counter, used for scheduling
    pub cycles: u64,
}

impl Default for Cpu {
    fn default() -> Self {
        Self::new()
    }
}

/// what running code can see of the core, every register in every bank, the flags, VFP and the exclusive monitor,
/// w/o the cycle count, to tell if running some code left it as it was
#[derive(Clone, PartialEq, Eq)]
pub struct Snapshot {
    regs: [u32; 16],
    psrs: [Psr; 8],
    banked: [[u32; 2]; 6],
    fiq_user: [[u32; 5]; 2],
    vfp: [u32; 34],
    cp15: [u32; 4],
    exclusive_addr: Option<u32>,
    halted: bool,
}

impl Cpu {
    pub fn new() -> Cpu {
        Cpu {
            regs: [0; 16],
            cpsr: Psr::default(),
            spsr: Psr::default(),
            banked_sp_lr: [[0; 2]; 6],
            banked_spsr: [Psr::default(); 6],
            fiq_r8_r12: [0; 5],
            user_r8_r12: [0; 5],
            vfp: Vfp::new(),
            cp15: Cp15::new(),
            exclusive_addr: None,
            branched: false,
            halted: false,
            cycles: 0,
        }
    }

    /// the core as running code sees it, see Snapshot
    pub fn snapshot(&self) -> Snapshot {
        let [b0, b1, b2, b3, b4, b5] = self.banked_spsr;
        let mut vfp = [0u32; 34];
        vfp[..32].copy_from_slice(&self.vfp.regs);
        vfp[32] = self.vfp.fpscr;
        vfp[33] = self.vfp.fpexc;
        Snapshot {
            regs: self.regs,
            psrs: [self.cpsr, self.spsr, b0, b1, b2, b3, b4, b5],
            banked: self.banked_sp_lr,
            fiq_user: [self.fiq_r8_r12, self.user_r8_r12],
            vfp,
            cp15: [self.cp15.thread_id_rw, self.cp15.thread_id_ro, self.cp15.thread_id_priv, self.cp15.control],
            exclusive_addr: self.exclusive_addr,
            halted: self.halted,
        }
    }

    // register access

    #[inline(always)]
    pub fn reg(&self, index: usize) -> u32 {
        self.regs[index]
    }

    /// writes a register, routing r15 thru the branch path so every instruction that can write the PC gets it right for free
    #[inline(always)]
    pub fn set_reg(&mut self, index: usize, value: u32) {
        if index == PC {
            self.branch(value);
        } else {
            self.regs[index] = value;
        }
    }

    /// branch to addr, keeping the current instruction set
    #[inline(always)]
    pub fn branch(&mut self, addr: u32) {
        self.regs[PC] = if self.cpsr.thumb { addr & !1 } else { addr & !3 };
        self.branched = true;
    }

    /// interworking branch, bit 0 of addr picks Thumb
    #[inline(always)]
    pub fn branch_exchange(&mut self, addr: u32) {
        self.cpsr.thumb = addr & 1 != 0;
        self.regs[PC] = if self.cpsr.thumb { addr & !1 } else { addr & !3 };
        self.branched = true;
    }

    /// addr of the instruction running now
    #[inline(always)]
    pub fn current_pc(&self) -> u32 {
        self.regs[PC].wrapping_sub(if self.cpsr.thumb { 4 } else { 8 })
    }

    pub fn halt(&mut self) {
        self.halted = true;
    }

    pub fn is_halted(&self) -> bool {
        self.halted
    }

    pub fn resume(&mut self) {
        self.halted = false;
    }

    /// sets up the CPU to start running at entry w/ stack sp, like the kernel does when it makes a thread
    pub fn reset_to(&mut self, entry: u32, sp: u32) {
        self.regs = [0; 16];
        self.regs[SP] = sp;
        self.cpsr = Psr::default();
        self.cpsr.thumb = entry & 1 != 0;
        self.regs[PC] = entry & !1;
        self.exclusive_addr = None;
        self.branched = false;
        self.halted = false;
    }

    // mode switching

    fn bank_index(mode: Mode) -> Option<usize> {
        match mode {
            Mode::Fiq => Some(0),
            Mode::Irq => Some(1),
            Mode::Supervisor => Some(2),
            Mode::Abort => Some(3),
            Mode::Undefined => Some(4),
            Mode::User | Mode::System => None,
        }
    }

    /// swaps the banked registers when the mode field of CPSR changes
    pub fn switch_mode(&mut self, new_mode: Mode) {
        let old_mode = self.cpsr.mode;
        if old_mode == new_mode {
            return;
        }

        // save the outgoing bank
        if let Some(i) = Self::bank_index(old_mode) {
            self.banked_sp_lr[i] = [self.regs[SP], self.regs[LR]];
            self.banked_spsr[i] = self.spsr;
        } else {
            self.banked_sp_lr[5] = [self.regs[SP], self.regs[LR]];
        }
        if old_mode == Mode::Fiq {
            self.fiq_r8_r12.copy_from_slice(&self.regs[8..13]);
        } else if new_mode == Mode::Fiq {
            self.user_r8_r12.copy_from_slice(&self.regs[8..13]);
        }

        // load the incoming one
        if let Some(i) = Self::bank_index(new_mode) {
            self.regs[SP] = self.banked_sp_lr[i][0];
            self.regs[LR] = self.banked_sp_lr[i][1];
            self.spsr = self.banked_spsr[i];
        } else {
            self.regs[SP] = self.banked_sp_lr[5][0];
            self.regs[LR] = self.banked_sp_lr[5][1];
        }
        if new_mode == Mode::Fiq {
            self.regs[8..13].copy_from_slice(&self.fiq_r8_r12);
        } else if old_mode == Mode::Fiq {
            self.regs[8..13].copy_from_slice(&self.user_r8_r12);
        }

        self.cpsr.mode = new_mode;
    }

    /// reads a register from the User mode bank whatever the current mode, for LDM/STM w/ the S bit
    pub fn read_user_reg(&self, index: usize) -> u32 {
        match index {
            8..=12 if self.cpsr.mode == Mode::Fiq => self.user_r8_r12[index - 8],
            13 | 14 if !matches!(self.cpsr.mode, Mode::User | Mode::System) => {
                self.banked_sp_lr[5][index - 13]
            }
            PC => self.regs[PC].wrapping_add(4),
            _ => self.regs[index],
        }
    }

    /// writes a register in the User mode bank whatever the current mode
    pub fn write_user_reg(&mut self, index: usize, value: u32) {
        match index {
            8..=12 if self.cpsr.mode == Mode::Fiq => self.user_r8_r12[index - 8] = value,
            13 | 14 if !matches!(self.cpsr.mode, Mode::User | Mode::System) => {
                self.banked_sp_lr[5][index - 13] = value
            }
            PC => self.branch_exchange(value),
            _ => self.regs[index] = value,
        }
    }

    /// restores CPSR from SPSR, like MOVS pc, lr and friends do
    pub fn restore_cpsr(&mut self) {
        let spsr = self.spsr;
        self.switch_mode(spsr.mode);
        self.cpsr = spsr;
    }

    // execution

    /// runs up to budget instructions, returning early on anything the caller has to deal w/
    pub fn run<B: Bus>(&mut self, bus: &mut B, budget: u64) -> Exit {
        let end = self.cycles.wrapping_add(budget);
        while self.cycles < end {
            if self.halted {
                return Exit::Halted;
            }
            if let Some(exit) = self.step(bus) {
                return exit;
            }
        }
        Exit::Timeout
    }

    /// runs exactly one instruction
    /// always inlined, a call for every instruction the interpreter runs costs it about a tenth,
    /// and the compiler's choice flips w/ unrelated changes to its callers
    #[inline(always)]
    pub fn step<B: Bus>(&mut self, bus: &mut B) -> Option<Exit> {
        self.cycles += 1;
        if self.cpsr.thumb {
            let addr = self.regs[PC];
            let opcode = bus.fetch16(addr) as u32;
            self.regs[PC] = addr.wrapping_add(4);
            let exit = thumb::execute(self, bus, opcode);
            if self.branched {
                self.branched = false;
            } else {
                self.regs[PC] = addr.wrapping_add(2);
            }
            exit
        } else {
            let addr = self.regs[PC];
            let opcode = bus.fetch32(addr);
            self.regs[PC] = addr.wrapping_add(8);
            let exit = arm::execute(self, bus, opcode);
            if self.branched {
                self.branched = false;
            } else {
                self.regs[PC] = addr.wrapping_add(4);
            }
            exit
        }
    }

    // flag helpers

    #[inline(always)]
    pub fn set_nz(&mut self, value: u32) {
        self.cpsr.n = (value as i32) < 0;
        self.cpsr.z = value == 0;
    }
}

impl std::fmt::Debug for Cpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "CPU state (pc=0x{:08X}):", self.current_pc())?;
        for row in 0..4 {
            for col in 0..4 {
                let i = row * 4 + col;
                write!(f, "  r{i:<2}=0x{:08X}", self.regs[i])?;
            }
            writeln!(f)?;
        }
        write!(
            f,
            "  cpsr=0x{:08X} [{}{}{}{}{}] mode={:?} {}",
            self.cpsr.to_bits(),
            if self.cpsr.n { 'N' } else { '-' },
            if self.cpsr.z { 'Z' } else { '-' },
            if self.cpsr.c { 'C' } else { '-' },
            if self.cpsr.v { 'V' } else { '-' },
            if self.cpsr.q { 'Q' } else { '-' },
            self.cpsr.mode,
            if self.cpsr.thumb { "Thumb" } else { "ARM" },
        )
    }
}
