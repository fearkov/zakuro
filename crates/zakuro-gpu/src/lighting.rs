//! fragment lighting. per pixel, the lights set up in the lighting registers
//! shade the surface whose orientation the vertex shader hands over as a
//! quaternion, and the combiners read the result as the primary (diffuse)
//! and secondary (specular) fragment colors. how each term comes out of the
//! lookup tables follows what Citra worked out.

pub const REG_ENABLE: usize = 0x08F;
const REG_LIGHTS: usize = 0x140;
const REG_GLOBAL_AMBIENT: usize = 0x1C0;
const REG_LIGHT_COUNT: usize = 0x1C2;
const REG_CONFIG0: usize = 0x1C3;
const REG_CONFIG1: usize = 0x1C4;
pub const REG_TABLE_INDEX: usize = 0x1C5;
const REG_DISABLE: usize = 0x1C6;
pub const REG_TABLE_DATA: usize = 0x1C8;
pub const REG_TABLE_DATA_END: usize = 0x1CF;
const REG_TABLE_ABSOLUTE: usize = 0x1D0;
const REG_TABLE_INPUT: usize = 0x1D1;
const REG_TABLE_SCALE: usize = 0x1D2;
const REG_LIGHT_SLOTS: usize = 0x1D9;

/// the tables, the specular distributions, fresnel, the reflections and a
/// spotlight and a distance attenuation for each light.
const DISTRIBUTION0: usize = 0;
const DISTRIBUTION1: usize = 1;
const FRESNEL: usize = 3;
const REFLECT_BLUE: usize = 4;
const REFLECT_GREEN: usize = 5;
const REFLECT_RED: usize = 6;
const SPOTLIGHT: usize = 8;
const DISTANCE: usize = 16;
const TABLES: usize = 24;

/// the lookup tables, each entry decoded as it arrives into its value and
/// the step to the next entry, which interpolates between them.
pub struct Tables(Box<[[[f32; 2]; 256]; TABLES]>);

impl Default for Tables {
    fn default() -> Self {
        Tables(Box::new([[[0.0; 2]; 256]; TABLES]))
    }
}

impl Tables {
    /// takes a write to the table data registers, into the table and entry
    /// the index register names, which then moves on to the next entry.
    pub fn write(&mut self, registers: &mut [u32], value: u32) {
        let index = registers[REG_TABLE_INDEX];
        let entry = (index & 0xFF) as usize;
        if let Some(table) = self.0.get_mut(((index >> 8) & 0x1F) as usize) {
            // a 0.12 value and the step to the next one, an 11-bit magnitude
            // with the sign above it
            let step = ((value >> 12) & 0x7FF) as f32 / 2047.0;
            let step = if value & (1 << 23) != 0 { -step } else { step };
            table[entry] = [(value & 0xFFF) as f32 / 4095.0, step];
        }
        registers[REG_TABLE_INDEX] = (index & !0xFF) | ((entry as u32 + 1) & 0xFF);
    }

    fn lookup(&self, table: usize, entry: u8, delta: f32) -> f32 {
        let [value, step] = self.0[table][entry as usize];
        value + step * delta
    }
}

/// a PICA float with the given mantissa and exponent widths, where a zero
/// exponent is not denormal but just the smallest one.
fn pica_float(bits: u32, mantissa: u32, exponent: u32) -> f32 {
    let width = mantissa + exponent + 1;
    let sign = (bits >> (mantissa + exponent)) << 31;
    if bits & ((1 << (width - 1)) - 1) == 0 {
        return f32::from_bits(sign);
    }
    let mut biased = (bits >> mantissa) & ((1 << exponent) - 1);
    biased = if biased == (1 << exponent) - 1 { 255 } else { biased + 128 - (1 << (exponent - 1)) };
    f32::from_bits(sign | ((bits & ((1 << mantissa) - 1)) << (23 - mantissa)) | (biased << 23))
}

/// a light color, three 10-bit fields where 255 stands for one.
fn color(value: u32) -> [f32; 3] {
    [(value >> 20) & 0x3FF, (value >> 10) & 0x3FF, value & 0x3FF].map(|c| c as f32 / 255.0)
}

/// a component of a spotlight direction, a signed 1.11 fixed point number.
fn fixed(value: u32) -> f32 {
    (((value & 0x1FFF) as i32) << 19 >> 19) as f32 / 2047.0
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn normalized(v: [f32; 3]) -> [f32; 3] {
    let length = dot(v, v).sqrt();
    v.map(|c| c / length)
}

/// v turned by the unit quaternion q.
fn rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let axis = [q[0], q[1], q[2]];
    let inner = cross(axis, v);
    let turn = cross(axis, std::array::from_fn(|i| inner[i] + v[i] * q[3]));
    std::array::from_fn(|i| v[i] + 2.0 * turn[i])
}

/// which tables a lighting configuration has room for.
fn supported(config: u32, table: usize) -> bool {
    match table {
        DISTRIBUTION0 => config != 1,
        DISTRIBUTION1 => !matches!(config, 0 | 1 | 5),
        SPOTLIGHT => !matches!(config, 2 | 3),
        FRESNEL => !matches!(config, 0 | 2 | 4),
        REFLECT_RED => config != 3,
        _ => matches!(config, 4 | 5 | 8),
    }
}

/// how a table is read, the dot product it takes, whether that is taken as
/// its absolute value, and the scale on the result.
#[derive(Debug, Clone, Copy)]
struct Lookup {
    input: u32,
    absolute: bool,
    scale: f32,
}

#[derive(Debug, Clone, Copy)]
struct Light {
    specular0: [f32; 3],
    specular1: [f32; 3],
    diffuse: [f32; 3],
    ambient: [f32; 3],
    position: [f32; 3],
    /// the unit vector toward a directional light.
    direction: [f32; 3],
    /// the inverse spotlight direction.
    spot: [f32; 3],
    directional: bool,
    two_sided: bool,
    geometric0: bool,
    geometric1: bool,
    /// bias and scale into the light's distance table, when it has one.
    distance: Option<(f32, f32)>,
    spotlight: bool,
    shadowed: bool,
    /// which light it is, which picks its spotlight and distance tables.
    number: usize,
}

#[derive(Debug, Clone, Copy)]
enum Bump {
    None,
    /// the texture unit holding the normals, and whether to rebuild z.
    Normal(usize, bool),
    Tangent(usize),
}

#[derive(Debug, Clone, Copy)]
struct Shadow {
    unit: usize,
    invert: bool,
    primary: bool,
    secondary: bool,
    alpha: bool,
}

/// the lighting of a draw, decoded from the registers once.
#[derive(Debug, Clone)]
pub struct Lighting {
    lights: Vec<Light>,
    global_ambient: [f32; 3],
    config: u32,
    distribution0: Option<Lookup>,
    distribution1: Option<Lookup>,
    fresnel: Option<Lookup>,
    reflect: [Option<Lookup>; 3],
    spotlight: Lookup,
    /// whether fresnel replaces the primary and the secondary alpha.
    fresnel_primary: bool,
    fresnel_secondary: bool,
    clamp_highlights: bool,
    /// whether anything reads the half vector, or the unit vector toward
    /// the viewer, most draws need neither and skip normalizing them.
    needs_half: bool,
    needs_view: bool,
    bump: Bump,
    shadow: Option<Shadow>,
}

impl Lighting {
    /// the lighting set up in the registers, or None when it is off.
    pub fn read(registers: &[u32]) -> Option<Lighting> {
        if registers[REG_ENABLE] & 1 == 0 || registers[REG_DISABLE] & 1 != 0 {
            return None;
        }
        let config0 = registers[REG_CONFIG0];
        let config1 = registers[REG_CONFIG1];
        let config = (config0 >> 4) & 0xF;
        let absolute = registers[REG_TABLE_ABSOLUTE];
        let inputs = registers[REG_TABLE_INPUT];
        let scales = registers[REG_TABLE_SCALE];
        // each table's input, abs and scale sit four bits apart, abs being
        // a disable bit one above
        let lookup = |field: u32| Lookup {
            input: (inputs >> (field * 4)) & 7,
            absolute: (absolute >> (field * 4 + 1)) & 1 == 0,
            scale: match (scales >> (field * 4)) & 7 {
                0 => 1.0,
                1 => 2.0,
                2 => 4.0,
                3 => 8.0,
                6 => 0.25,
                7 => 0.5,
                _ => 0.0,
            },
        };
        let table = |table: usize, disabled: u32, field: u32| {
            (config1 & (1 << disabled) == 0 && supported(config, table)).then(|| lookup(field))
        };

        let count = (registers[REG_LIGHT_COUNT] & 7) + 1;
        let lights: Vec<Light> = (0..count)
            .map(|slot| {
                let number = ((registers[REG_LIGHT_SLOTS] >> (slot * 4)) & 7) as usize;
                let block = &registers[REG_LIGHTS + number * 0x10..REG_LIGHTS + number * 0x10 + 0x10];
                let half = |value: u32| pica_float(value & 0xFFFF, 10, 5);
                let position = [half(block[4]), half(block[4] >> 16), half(block[5])];
                Light {
                    specular0: color(block[0]),
                    specular1: color(block[1]),
                    diffuse: color(block[2]),
                    ambient: color(block[3]),
                    position,
                    direction: normalized(position),
                    spot: [fixed(block[6]), fixed(block[6] >> 16), fixed(block[7])],
                    directional: block[9] & 1 != 0,
                    two_sided: block[9] & 2 != 0,
                    geometric0: block[9] & 4 != 0,
                    geometric1: block[9] & 8 != 0,
                    distance: (config1 & (1 << (24 + number)) == 0)
                        .then(|| (pica_float(block[10] & 0xF_FFFF, 12, 7), pica_float(block[11] & 0xF_FFFF, 12, 7))),
                    spotlight: config1 & (1 << (8 + number)) == 0 && supported(config, SPOTLIGHT),
                    shadowed: config1 & (1 << number) == 0,
                    number,
                }
            })
            .collect();

        let bump = match (config0 >> 28) & 3 {
            1 => Bump::Normal(((config0 >> 22) & 3) as usize, config0 & (1 << 30) == 0),
            2 => Bump::Tangent(((config0 >> 22) & 3) as usize),
            _ => Bump::None,
        };
        let shadow = (config0 & 1 != 0).then_some(Shadow {
            unit: ((config0 >> 24) & 3) as usize,
            invert: config0 & (1 << 18) != 0,
            primary: config0 & (1 << 16) != 0,
            secondary: config0 & (1 << 17) != 0,
            alpha: config0 & (1 << 19) != 0,
        });

        let distribution0 = table(DISTRIBUTION0, 16, 0);
        let distribution1 = table(DISTRIBUTION1, 17, 1);
        let fresnel = table(FRESNEL, 19, 3);
        let reflect = [table(REFLECT_RED, 20, 6), table(REFLECT_GREEN, 21, 5), table(REFLECT_BLUE, 22, 4)];
        let spotlight = lookup(2);

        // the dot products the tables that get read take
        let inputs: Vec<u32> = [distribution0, distribution1, fresnel]
            .into_iter()
            .chain(reflect)
            .flatten()
            .chain(lights.iter().any(|light| light.spotlight).then_some(spotlight))
            .map(|lookup| lookup.input)
            .collect();
        let needs_half = lights.iter().any(|light| light.geometric0 || light.geometric1)
            || inputs.iter().any(|&input| input == 0 || input == 1 || (input == 5 && config == 8));
        let needs_view = needs_half || inputs.iter().any(|&input| input == 1 || input == 2);

        Some(Lighting {
            lights,
            global_ambient: color(registers[REG_GLOBAL_AMBIENT]),
            config,
            distribution0,
            distribution1,
            fresnel,
            reflect,
            spotlight,
            fresnel_primary: config0 & (1 << 2) != 0,
            fresnel_secondary: config0 & (1 << 3) != 0,
            clamp_highlights: config0 & (1 << 27) != 0,
            needs_half,
            needs_view,
            bump,
            shadow,
        })
    }

    /// the primary and secondary fragment colors of a fragment whose
    /// surface quaternion and view vector the vertices interpolated,
    /// textures being what the texture units sampled there.
    pub fn shade(&self, tables: &Tables, quaternion: [f32; 4], view: [f32; 3], textures: &[[f32; 4]; 4]) -> ([f32; 4], [f32; 4]) {
        let length = quaternion.iter().map(|c| c * c).sum::<f32>().sqrt();
        let quaternion = if length > 0.0 { quaternion.map(|c| c / length) } else { [0.0, 0.0, 0.0, 1.0] };

        let shadow = self.shadow.map_or([1.0; 4], |shadow| {
            let texel = textures[shadow.unit];
            if shadow.invert { texel.map(|c| 1.0 - c) } else { texel }
        });

        let (surface_normal, surface_tangent) = match self.bump {
            Bump::None => ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
            Bump::Normal(unit, rebuild) => {
                let mut normal: [f32; 3] = std::array::from_fn(|i| textures[unit][i] * 2.0 - 1.0);
                if rebuild {
                    normal[2] = (1.0 - (normal[0] * normal[0] + normal[1] * normal[1])).max(0.0).sqrt();
                }
                (normal, [1.0, 0.0, 0.0])
            }
            Bump::Tangent(unit) => ([0.0, 0.0, 1.0], std::array::from_fn(|i| textures[unit][i] * 2.0 - 1.0)),
        };
        let normal = rotate(quaternion, surface_normal);
        // only the last configuration has a table that reads the tangent
        let tangent = if self.config == 8 { rotate(quaternion, surface_tangent) } else { surface_tangent };
        let norm_view = if self.needs_view { normalized(view) } else { [0.0; 3] };

        let mut diffuse_sum = [0.0, 0.0, 0.0, 1.0];
        let mut specular_sum = [0.0, 0.0, 0.0, 1.0];
        for (slot, light) in self.lights.iter().enumerate() {
            let light_vector = if light.directional {
                light.direction
            } else {
                normalized(std::array::from_fn(|i| light.position[i] + view[i]))
            };
            let (half, half_unit) = if self.needs_half {
                let half = std::array::from_fn(|i| norm_view[i] + light_vector[i]);
                (half, normalized(half))
            } else {
                ([0.0; 3], [0.0; 3])
            };

            let distance = light.distance.map_or(1.0, |(bias, scale)| {
                let offset: [f32; 3] = std::array::from_fn(|i| -view[i] - light.position[i]);
                let place = (scale * dot(offset, offset).sqrt() + bias).clamp(0.0, 1.0);
                let entry = (place * 256.0).floor().clamp(0.0, 255.0);
                tables.lookup(DISTANCE + light.number, entry as u8, place * 256.0 - entry)
            });

            let value = |lookup: Lookup, table: usize| {
                let result = match lookup.input {
                    0 => dot(normal, half_unit),
                    1 => dot(norm_view, half_unit),
                    2 => dot(normal, norm_view),
                    3 => dot(light_vector, normal),
                    4 => dot(light_vector, light.spot),
                    5 if self.config == 8 => {
                        let along = dot(normal, half_unit);
                        dot(std::array::from_fn(|i| half_unit[i] - normal[i] * along), tangent)
                    }
                    _ => 0.0,
                };
                let (entry, delta) = if lookup.absolute {
                    let result = if light.two_sided { result.abs() } else { result.max(0.0) };
                    let entry = (result * 256.0).floor().clamp(0.0, 255.0);
                    (entry as u8, result * 256.0 - entry)
                } else {
                    // the signed index wraps into the upper half of the table
                    let entry = (result * 128.0).floor().clamp(-128.0, 127.0);
                    (entry as i8 as u8, result * 128.0 - entry)
                };
                lookup.scale * tables.lookup(table, entry, delta)
            };

            let spot = if light.spotlight { value(self.spotlight, SPOTLIGHT + light.number) } else { 1.0 };
            let distribution0 = self.distribution0.map_or(1.0, |lookup| value(lookup, DISTRIBUTION0));
            let red = self.reflect[0].map_or(1.0, |lookup| value(lookup, REFLECT_RED));
            let reflect = [
                red,
                self.reflect[1].map_or(red, |lookup| value(lookup, REFLECT_GREEN)),
                self.reflect[2].map_or(red, |lookup| value(lookup, REFLECT_BLUE)),
            ];
            let distribution1 = self.distribution1.map_or(1.0, |lookup| value(lookup, DISTRIBUTION1));
            let mut specular0 = light.specular0.map(|c| c * distribution0);
            let mut specular1: [f32; 3] = std::array::from_fn(|i| distribution1 * reflect[i] * light.specular1[i]);

            // only the last light applies fresnel
            if slot == self.lights.len() - 1 {
                if let Some(lookup) = self.fresnel {
                    let fresnel = value(lookup, FRESNEL);
                    if self.fresnel_primary {
                        diffuse_sum[3] = fresnel;
                    }
                    if self.fresnel_secondary {
                        specular_sum[3] = fresnel;
                    }
                }
            }

            let facing = dot(light_vector, normal);
            let facing = if light.two_sided { facing.abs() } else { facing.max(0.0) };
            let highlights = if self.clamp_highlights && facing == 0.0 { 0.0 } else { 1.0 };
            if light.geometric0 || light.geometric1 {
                let length = dot(half, half);
                let factor = if length == 0.0 { 0.0 } else { (facing / length).min(1.0) };
                if light.geometric0 {
                    specular0 = specular0.map(|c| c * factor);
                }
                if light.geometric1 {
                    specular1 = specular1.map(|c| c * factor);
                }
            }

            let shadowed = |enabled: bool| {
                if enabled && light.shadowed { [shadow[0], shadow[1], shadow[2]] } else { [1.0; 3] }
            };
            let shadow_primary = shadowed(self.shadow.is_some_and(|s| s.primary));
            let shadow_secondary = shadowed(self.shadow.is_some_and(|s| s.secondary));
            for i in 0..3 {
                let diffuse = (light.diffuse[i] * facing + light.ambient[i]) * distance * spot;
                let specular = (specular0[i] + specular1[i]) * highlights * distance * spot;
                diffuse_sum[i] += diffuse * shadow_primary[i];
                specular_sum[i] += specular * shadow_secondary[i];
            }
        }

        if self.shadow.is_some_and(|s| s.alpha) {
            if self.fresnel_primary {
                diffuse_sum[3] *= shadow[3];
            }
            if self.fresnel_secondary {
                specular_sum[3] *= shadow[3];
            }
        }
        for (sum, ambient) in diffuse_sum.iter_mut().zip(self.global_ambient) {
            *sum += ambient;
        }
        // the unit hands the combiners 8-bit colors
        let quantize = |c: f32| ((c.clamp(0.0, 1.0) * 255.0) as u8) as f32 / 255.0;
        (diffuse_sum.map(quantize), specular_sum.map(quantize))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_floats_decode() {
        assert_eq!(pica_float(0x3C00, 10, 5), 1.0);
        assert_eq!(pica_float(0xC000, 10, 5), -2.0);
        assert_eq!(pica_float(0x0000, 10, 5), 0.0);
    }

    #[test]
    fn table_writes_advance_the_index() {
        let mut registers = vec![0u32; 0x200];
        registers[REG_TABLE_INDEX] = 3 << 8;
        let mut tables = Tables::default();
        tables.write(&mut registers, 4095);
        // half of the way down, the sign sits above the step
        tables.write(&mut registers, 2048 | ((0x800 | 0x400) << 12));
        assert_eq!(tables.0[3][0], [1.0, 0.0]);
        assert_eq!(tables.0[3][1][1], -1024.0 / 2047.0);
        assert_eq!(registers[REG_TABLE_INDEX], (3 << 8) | 2);
    }

    #[test]
    fn a_quaternion_turns_a_vector() {
        // a quarter turn about z takes x to y
        let half = std::f32::consts::FRAC_1_SQRT_2;
        let turned = rotate([0.0, 0.0, half, half], [1.0, 0.0, 0.0]);
        assert!((turned[0]).abs() < 1e-6 && (turned[1] - 1.0).abs() < 1e-6);
    }
}
