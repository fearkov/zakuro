//! system control coprocessor.

#[derive(Debug, Clone, Default)]
pub struct Cp15 {
    /// c13,c0,2, user read/write thread ID.
    pub thread_id_rw: u32,
    /// c13,c0,3, user read-only thread ID.
    pub thread_id_ro: u32,
    /// c13,c0,4, privileged-only thread ID.
    pub thread_id_priv: u32,
    /// c1,c0,0, control register. We keep the value so reads round-trip.
    pub control: u32,

    /// set when the guest asked for an instruction-cache invalidate since the
    /// last time the JIT looked. Range-based invalidation is tracked too.
    pub icache_invalidated: bool,
    pub invalidate_ranges: Vec<(u32, u32)>,
}

impl Cp15 {
    pub fn new() -> Cp15 {
        Cp15 {
            // ARM1136 default, enable I/D cache and branch prediction bits the
            // firmware would have set before the game starts.
            control: 0x0005_2078,
            ..Default::default()
        }
    }

    /// MRC p15, opc1, Rd, CRn, CRm, opc2
    pub fn read(&self, opc1: u32, crn: u32, crm: u32, opc2: u32) -> u32 {
        match (opc1, crn, crm, opc2) {
            // main ID register, ARM1176JZF-S style, which is what the 3DS
            // reports and what SDK code sniffs for.
            (0, 0, 0, 0) => 0x410F_B024,
            // cache type register.
            (0, 0, 0, 1) => 0x0F0D_2112,
            (0, 1, 0, 0) => self.control,
            (0, 13, 0, 2) => self.thread_id_rw,
            (0, 13, 0, 3) => self.thread_id_ro,
            (0, 13, 0, 4) => self.thread_id_priv,
            _ => {
                log::debug!("CP15 read of unimplemented c{crn},c{crm},{opc1},{opc2}");
                0
            }
        }
    }

    /// MCR p15, opc1, Rd, CRn, CRm, opc2
    pub fn write(&mut self, opc1: u32, crn: u32, crm: u32, opc2: u32, value: u32) {
        match (opc1, crn, crm, opc2) {
            (0, 1, 0, 0) => self.control = value,
            (0, 13, 0, 2) => self.thread_id_rw = value,
            (0, 13, 0, 3) => self.thread_id_ro = value,
            (0, 13, 0, 4) => self.thread_id_priv = value,

            // c7 is cache maintenance.
            (0, 7, 5, 0) | (0, 7, 5, 4) | (0, 7, 7, 0) => self.icache_invalidated = true,
            (0, 7, 5, 1) | (0, 7, 13, 1) => {
                self.invalidate_ranges.push((value & !31, 32));
            }
            // data cache clean/invalidate, no-ops for us, memory is coherent.
            (0, 7, ..) => {}

            _ => log::debug!(
                "CP15 write of unimplemented c{crn},c{crm},{opc1},{opc2} = 0x{value:08X}"
            ),
        }
    }

    /// drains the pending invalidation requests.
    pub fn take_invalidations(&mut self) -> (bool, Vec<(u32, u32)>) {
        let all = std::mem::take(&mut self.icache_invalidated);
        let ranges = std::mem::take(&mut self.invalidate_ranges);
        (all, ranges)
    }
}
