//! the fog the PICA200 mixes into a fragment after the combiners, by its
//! depth, through a table of 128 entries.

/// the fog's mode, in the bits under the combiner buffer's update, and
/// whether the depth goes into the table from the far end, bit 16.
pub const REG_MODE: usize = 0x0E0;
/// the fog's color, a byte a channel from red up.
pub const REG_COLOR: usize = 0x0E1;
/// the entry the next write to the table goes to.
pub const REG_TABLE_INDEX: usize = 0x0E6;
pub const REG_TABLE_DATA: usize = 0x0E8;
pub const REG_TABLE_DATA_END: usize = 0x0EF;

/// entries in the table.
pub const ENTRIES: usize = 128;
/// the mode that mixes the fog in, the other one, 7, is gas.
const MODE_FOG: u32 = 5;

/// the table, each entry the fog's factor at it and the step to the next.
pub struct Table {
    entries: Box<[[f32; 2]; ENTRIES]>,
    /// goes up with every write that changes an entry, so a copy elsewhere
    /// knows it is stale.
    generation: u64,
}

impl Default for Table {
    fn default() -> Self {
        Table { entries: Box::new([[0.0; 2]; ENTRIES]), generation: 0 }
    }
}

impl Table {
    /// takes a write to the table data registers, into the entry the index
    /// register names, which then moves on to the next.
    pub fn write(&mut self, registers: &mut [u32], value: u32) {
        let index = registers[REG_TABLE_INDEX];
        // the factor in 0.11 above the step, 13 bits with the sign on top
        let step = ((value & 0x1FFF) as i32) << 19 >> 19;
        let entry = [((value >> 13) & 0x7FF) as f32 / 2048.0, step as f32 / 2048.0];
        let slot = &mut self.entries[index as usize % ENTRIES];
        self.generation += (slot.map(f32::to_bits) != entry.map(f32::to_bits)) as u64;
        *slot = entry;
        registers[REG_TABLE_INDEX] = index.wrapping_add(1);
    }

    pub(crate) fn entries(&self) -> &[[f32; 2]; ENTRIES] {
        &self.entries
    }

    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }
}

/// a draw's fog, when its mode mixes it in.
pub struct Fog<'a> {
    color: [f32; 3],
    flip: bool,
    table: &'a Table,
}

impl<'a> Fog<'a> {
    pub fn read(registers: &[u32], table: &'a Table) -> Option<Fog<'a>> {
        let mode = registers[REG_MODE];
        let [r, g, b, _] = registers[REG_COLOR].to_le_bytes();
        (mode & 7 == MODE_FOG).then_some(Fog { color: [r, g, b].map(f32::from), flip: mode & (1 << 16) != 0, table })
    }

    /// mixes the fog into a fragment's color at a depth from 0 to 1, the
    /// table's factor of the color and the rest of the fog's.
    pub fn apply(&self, rgba: &mut [u8; 4], depth: f32) {
        let index = if self.flip { 1.0 - depth } else { depth } * ENTRIES as f32;
        let entry = index.floor().clamp(0.0, (ENTRIES - 1) as f32);
        let [factor, step] = self.table.entries[entry as usize];
        let factor = (factor + step * (index - entry)).clamp(0.0, 1.0);
        for (channel, fog) in rgba[..3].iter_mut().zip(self.color) {
            *channel = (factor * *channel as f32 + (1.0 - factor) * fog) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registers() -> Vec<u32> {
        vec![0; 0x300]
    }

    /// entries arrive as a factor and a signed step to the next one, into
    /// consecutive entries from the index.
    #[test]
    fn the_table_takes_factors_and_steps() {
        let mut registers = registers();
        let mut table = Table::default();
        registers[REG_TABLE_INDEX] = 5;
        // a half, then a step down of a quarter
        table.write(&mut registers, (1024 << 13) | (0x2000 - 512));
        table.write(&mut registers, 2047 << 13 | 1);
        assert_eq!(table.entries()[5], [0.5, -0.25]);
        assert_eq!(table.entries()[6], [2047.0 / 2048.0, 1.0 / 2048.0]);
        assert_eq!(registers[REG_TABLE_INDEX], 7);
        let generation = table.generation();
        registers[REG_TABLE_INDEX] = 5;
        table.write(&mut registers, (1024 << 13) | (0x2000 - 512));
        assert_eq!(table.generation(), generation, "the same entry again changes nothing");
    }

    /// the fog comes in as the factor falls, between entries too, and from
    /// the far end of the table when flipped. other modes leave the color.
    #[test]
    fn fog_mixes_by_depth() {
        let mut registers = registers();
        let mut table = Table::default();
        // a factor of 1 falling to 0 across the table
        for entry in 0..ENTRIES as u32 {
            let factor = 2048 - entry * 16;
            table.write(&mut registers, (factor.min(2047) << 13) | (0x2000 - 16));
        }
        registers[REG_COLOR] = 0x00_80_40_20;
        assert!(Fog::read(&registers, &table).is_none());
        registers[REG_MODE] = 7;
        assert!(Fog::read(&registers, &table).is_none(), "gas");

        registers[REG_MODE] = 5;
        let fog = Fog::read(&registers, &table).unwrap();
        let mut near = [200, 200, 200, 77];
        fog.apply(&mut near, 0.0);
        assert_eq!(near, [199, 199, 199, 77], "the alpha stays");
        let mut halfway = [200, 200, 200, 255];
        fog.apply(&mut halfway, 0.5);
        assert_eq!(halfway, [116, 132, 164, 255]);

        registers[REG_MODE] = 5 | 1 << 16;
        let fog = Fog::read(&registers, &table).unwrap();
        let mut far = [200, 200, 200, 255];
        fog.apply(&mut far, 1.0);
        assert_eq!(far, [199, 199, 199, 255], "flipped, the far end reads the start");
    }
}
