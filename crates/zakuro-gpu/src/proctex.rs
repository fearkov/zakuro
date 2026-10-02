//! the procedural texture, texture 3. rather than reading memory, the PICA
//! makes it up per fragment from its coordinates, a noise and lookup
//! tables. how each step comes out follows what Citra worked out against
//! the hardware.

pub const REG_CONFIG: usize = 0x0A8;
const REG_NOISE_U: usize = 0x0A9;
const REG_NOISE_V: usize = 0x0AA;
const REG_NOISE_FREQUENCY: usize = 0x0AB;
const REG_LUT: usize = 0x0AC;
const REG_LUT_OFFSET: usize = 0x0AD;
pub const REG_TABLE_INDEX: usize = 0x0AF;
pub const REG_TABLE_DATA: usize = 0x0B0;
pub const REG_TABLE_DATA_END: usize = 0x0B7;

/// the tables, by the number the index register gives them.
const NOISE: usize = 0;
const COLOR_MAP: usize = 2;
const ALPHA_MAP: usize = 3;
const COLOR: usize = 4;
const COLOR_DIFF: usize = 5;

/// entries in the noise and map tables, and in the color tables.
pub const MAP_ENTRIES: usize = 128;
pub const COLOR_ENTRIES: usize = 256;

/// the lookup tables, each entry decoded as it arrives.
pub struct Tables {
    /// the noise, color map and alpha map, a value and the step to the
    /// next entry each.
    maps: Box<[[[f32; 2]; MAP_ENTRIES]; 3]>,
    /// the colors, 0 to 255 a channel, and the steps to the next ones.
    colors: Box<[[f32; 4]; COLOR_ENTRIES]>,
    steps: Box<[[f32; 4]; COLOR_ENTRIES]>,
    /// goes up with every write that changes an entry, so a copy elsewhere
    /// knows it is stale. titles send the same tables over and over.
    generation: u64,
}

impl Default for Tables {
    fn default() -> Self {
        Tables {
            maps: Box::new([[[0.0; 2]; MAP_ENTRIES]; 3]),
            colors: Box::new([[0.0; 4]; COLOR_ENTRIES]),
            steps: Box::new([[0.0; 4]; COLOR_ENTRIES]),
            generation: 0,
        }
    }
}

/// an entry set to a new value, and whether that changed it, to the bit.
fn replace<const N: usize>(entry: &mut [f32; N], value: [f32; N]) -> bool {
    let changed = entry.map(f32::to_bits) != value.map(f32::to_bits);
    *entry = value;
    changed
}

impl Tables {
    /// takes a write to the table data registers, into the table and entry
    /// the index register names, which then moves on to the next entry.
    pub fn write(&mut self, registers: &mut [u32], value: u32) {
        let index = registers[REG_TABLE_INDEX];
        let entry = (index & 0xFF) as usize;
        let changed = match ((index >> 8) & 0xF) as usize {
            table @ (NOISE | COLOR_MAP | ALPHA_MAP) if entry < MAP_ENTRIES => {
                // a 0.12 value and the step to the next one, 12 bits with
                // the sign on top
                let step = (((value >> 12) & 0xFFF) as i32) << 20 >> 20;
                let map = match table {
                    NOISE => 0,
                    COLOR_MAP => 1,
                    _ => 2,
                };
                replace(&mut self.maps[map][entry], [(value & 0xFFF) as f32 / 4095.0, step as f32 / 4095.0])
            }
            COLOR => replace(&mut self.colors[entry], value.to_le_bytes().map(|c| c as f32)),
            // the steps come halved, signed bytes
            COLOR_DIFF => replace(&mut self.steps[entry], value.to_le_bytes().map(|c| c as i8 as f32 * 2.0)),
            _ => false,
        };
        self.generation += changed as u64;
        registers[REG_TABLE_INDEX] = (index & !0xFF) | ((entry as u32 + 1) & 0xFF);
    }

    /// the noise, color map and alpha map tables.
    pub(crate) fn maps(&self) -> &[[[f32; 2]; MAP_ENTRIES]; 3] {
        &self.maps
    }

    /// the colors and the steps between them.
    pub(crate) fn colors(&self) -> (&[[f32; 4]; COLOR_ENTRIES], &[[f32; 4]; COLOR_ENTRIES]) {
        (&self.colors, &self.steps)
    }

    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    /// a map table read at a coordinate from 0 to 1, between its entries.
    fn lookup(&self, map: usize, coordinate: f32) -> f32 {
        let at = coordinate * MAP_ENTRIES as f32;
        let entry = (at.max(0.0) as usize).min(MAP_ENTRIES - 1);
        let [value, step] = self.maps[map][entry];
        value + (at - entry as f32) * step
    }
}

/// the procedural texture as a draw sets it up.
#[derive(Clone, Copy)]
pub struct ProcTex<'a> {
    config: u32,
    noise: [u32; 2],
    frequency: u32,
    lut: u32,
    offset: u32,
    /// which texture coordinates it reads.
    pub coordinates: usize,
    tables: &'a Tables,
}

impl<'a> ProcTex<'a> {
    /// the procedural texture, when the texture configuration turns it on.
    pub fn read(registers: &[u32], tables: &'a Tables) -> Option<ProcTex<'a>> {
        let units = registers[crate::registers::REG_TEXTURE_CONFIG];
        if units & (1 << 10) == 0 {
            return None;
        }
        Some(ProcTex {
            config: registers[REG_CONFIG],
            noise: [registers[REG_NOISE_U], registers[REG_NOISE_V]],
            frequency: registers[REG_NOISE_FREQUENCY],
            lut: registers[REG_LUT],
            offset: registers[REG_LUT_OFFSET],
            coordinates: (((units >> 8) & 3) as usize).min(2),
            tables,
        })
    }

    /// the color at a pair of coordinates, 0 to 1 a channel.
    pub fn sample(&self, u: f32, v: f32) -> [f32; 4] {
        let config = self.config;
        let clamps = [config & 7, (config >> 3) & 7];
        let (mut u, mut v) = (u.abs(), v.abs());
        // the shifts come from the coordinates before the noise moves them
        let shifts = [shift(v, (config >> 16) & 3, clamps[0]), shift(u, (config >> 18) & 3, clamps[1])];
        if config & (1 << 15) != 0 {
            let noise = self.noise(u, v);
            let amplitude = |word: u32| (word as u16 as i16) as f32 / 4095.0;
            u = (u + noise * amplitude(self.noise[0])).abs();
            v = (v + noise * amplitude(self.noise[1])).abs();
        }
        let u = clamp(u + shifts[0], clamps[0]);
        let v = clamp(v + shifts[1], clamps[1]);

        let coordinate = self.tables.lookup(1, combine(u, v, (config >> 6) & 0xF));
        let width = ((self.lut >> 11) & 0xFF) as f32;
        let at = (self.offset & 0xFF) as f32 + coordinate * (width - 1.0).max(0.0);
        let (colors, steps) = self.tables.colors();
        let mut color = match self.lut & 7 {
            // linear, and linear with either kind of mipmap
            1 | 3 | 5 => {
                let entry = (at.max(0.0) as usize).min(COLOR_ENTRIES - 1);
                let part = at - entry as f32;
                std::array::from_fn(|c| colors[entry][c] + part * steps[entry][c])
            }
            _ => colors[(at.round().max(0.0) as usize).min(COLOR_ENTRIES - 1)],
        }
        .map(|c| (c / 255.0).clamp(0.0, 1.0));
        // alpha of its own skips the color table, the map gives it
        if config & (1 << 14) != 0 {
            color[3] = self.tables.lookup(2, combine(u, v, (config >> 10) & 0xF)).clamp(0.0, 1.0);
        }
        color
    }

    /// how much the noise moves a pair of coordinates, -1 to 1.
    fn noise(&self, u: f32, v: f32) -> f32 {
        let phase = |word: u32| (word >> 16) as f32 / 4096.0;
        let frequency = |half: u32| half_to_f32(half as u16);
        let x = 9.0 * frequency(self.frequency) * (u + phase(self.noise[0])).abs();
        let y = 9.0 * frequency(self.frequency >> 16) * (v + phase(self.noise[1])).abs();
        let (xi, yi) = (x as u32, y as u32);
        let (xf, yf) = (x - xi as f32, y - yi as f32);
        let g0 = random(xi, yi) * (xf + yf);
        let g1 = random(xi + 1, yi) * (xf + yf - 1.0);
        let g2 = random(xi, yi + 1) * (xf + yf - 1.0);
        let g3 = random(xi + 1, yi + 1) * (xf + yf - 2.0);
        let s = self.tables.lookup(0, xf);
        let t = self.tables.lookup(0, yf);
        let bottom = g0 * (1.0 - s) + g1 * s;
        let top = g2 * (1.0 - s) + g3 * s;
        bottom * (1.0 - t) + top * t
    }
}

/// a pseudo-random value from -1 to 1 for a point of the noise's grid.
fn random(x: u32, y: u32) -> f32 {
    const ROWS: [u32; 16] = [0, 4, 10, 8, 4, 9, 7, 12, 5, 15, 13, 14, 11, 15, 2, 11];
    const MIX: [u32; 16] = [10, 2, 15, 8, 0, 7, 4, 5, 5, 13, 2, 6, 13, 9, 3, 14];
    let one = |v: u32| (((v % 9 + 2) * 3) & 0xF) ^ ROWS[((v / 9) & 0xF) as usize];
    let u = one(x);
    let mut v = one(y);
    if u & 3 == 1 {
        v += 4;
    }
    v ^= (u & 1) * 6;
    v += 10 + u;
    v &= 0xF;
    v ^= MIX[u as usize];
    -1.0 + v as f32 * 2.0 / 15.0
}

/// how far the shift moves one coordinate, by the other.
fn shift(other: f32, mode: u32, clamp: u32) -> f32 {
    let amount = if clamp == 3 { 1.0 } else { 0.5 };
    let other = other as i32;
    match mode {
        // odd and even rows
        1 => amount * ((other / 2) % 2) as f32,
        2 => amount * (((other + 1) / 2) % 2) as f32,
        _ => 0.0,
    }
}

/// brings a coordinate back to 0 to 1.
fn clamp(c: f32, mode: u32) -> f32 {
    match mode {
        // to zero
        0 => {
            if c > 1.0 {
                0.0
            } else {
                c
            }
        }
        // to the edge
        1 => c.min(1.0),
        // repeat
        2 => c - c.floor(),
        // mirrored repeat
        3 => {
            let whole = c as i32;
            let part = c - whole as f32;
            if whole % 2 == 0 {
                part
            } else {
                1.0 - part
            }
        }
        // pulse
        4 => {
            if c <= 0.5 {
                0.0
            } else {
                1.0
            }
        }
        _ => c.clamp(0.0, 1.0),
    }
}

/// the two coordinates made into one.
fn combine(u: f32, v: f32, function: u32) -> f32 {
    let length = (u * u + v * v).sqrt();
    match function {
        0 => u,
        1 => u * u,
        2 => v,
        3 => v * v,
        4 => (u + v) * 0.5,
        5 => (u * u + v * v) * 0.5,
        6 => length.min(1.0),
        7 => u.min(v),
        8 => u.max(v),
        9 => (((u + v) * 0.5 + length) * 0.5).min(1.0),
        _ => 0.0,
    }
}

/// a 16-bit float, a sign, 5 bits of exponent and 10 of mantissa.
fn half_to_f32(half: u16) -> f32 {
    let sign = if half & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = ((half >> 10) & 0x1F) as i32;
    let mantissa = (half & 0x3FF) as f32 / 1024.0;
    match exponent {
        0 => sign * mantissa * 2f32.powi(-14),
        _ => sign * (1.0 + mantissa) * 2f32.powi(exponent - 15),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// the same tables sent again leave what the GPU has of them as it is,
    /// a changed entry of any of them makes it stale.
    #[test]
    fn only_a_changed_entry_makes_the_tables_stale() {
        let mut registers = vec![0u32; 0x300];
        let mut tables = Tables::default();
        let send = |tables: &mut Tables, registers: &mut Vec<u32>, color: u32| {
            for (table, value) in [(NOISE, 100 | (5 << 12)), (COLOR, color), (COLOR_DIFF, 0x01FF_0102)] {
                registers[REG_TABLE_INDEX] = (table as u32) << 8;
                tables.write(registers, value);
            }
        };
        send(&mut tables, &mut registers, 0xFF80_8080);
        let sent = tables.generation();
        send(&mut tables, &mut registers, 0xFF80_8080);
        assert_eq!(tables.generation(), sent);
        send(&mut tables, &mut registers, 0xFF80_8081);
        assert_ne!(tables.generation(), sent);
    }

    /// a flat grey table and a map straight through, the texture comes out
    /// the grey wherever it is read.
    #[test]
    fn a_flat_table_gives_its_color_everywhere() {
        let mut registers = vec![0u32; 0x300];
        let mut tables = Tables::default();
        registers[REG_TABLE_INDEX] = (COLOR_MAP as u32) << 8;
        for entry in 0..MAP_ENTRIES as u32 {
            // the value rises by a step an entry
            tables.write(&mut registers, (entry * 32) | (32 << 12));
        }
        registers[REG_TABLE_INDEX] = (COLOR as u32) << 8;
        for _ in 0..COLOR_ENTRIES {
            tables.write(&mut registers, 0xFF80_8080);
        }
        registers[crate::registers::REG_TEXTURE_CONFIG] = 1 << 10;
        // repeat both ways, u as the coordinate, a 256 entry linear table
        registers[REG_CONFIG] = 2 | 2 << 3;
        registers[REG_LUT] = 1 | 255 << 11;
        let proctex = ProcTex::read(&registers, &tables).expect("turned on");
        for (u, v) in [(0.0, 0.0), (0.3, 0.7), (1.6, 2.2)] {
            let color = proctex.sample(u, v);
            assert!((color[0] - 128.0 / 255.0).abs() < 1e-3, "{color:?} at {u}, {v}");
            assert!((color[3] - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn the_noise_stays_between_minus_one_and_one() {
        for x in 0..64 {
            for y in 0..64 {
                let value = random(x, y);
                assert!((-1.0..=1.0).contains(&value));
            }
        }
    }

    #[test]
    fn halves_read_as_floats() {
        assert_eq!(half_to_f32(0x3C00), 1.0);
        assert_eq!(half_to_f32(0xC000), -2.0);
        assert_eq!(half_to_f32(0x3800), 0.5);
    }
}
