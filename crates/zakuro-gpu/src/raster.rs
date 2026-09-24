//! software rasterization of PICA200 draw calls.

use crate::registers::*;
use crate::shader::{self, ShaderState, ShaderUnit, Vec4};
use crate::{format::ColorFormat, GpuMemory};

/// reads a float24-encoded viewport register.
fn read_float24(registers: &[u32], index: usize) -> f32 {
    crate::shader::isa::decode_float24(registers[index])
}

/// decodes a GPUREG_*_LOC register into a physical address.
fn loc_register(registers: &[u32], index: usize) -> u32 {
    (registers[index] & 0x0FFF_FFFF) << 3
}

struct AttributeLoader {
    offset: u32,
    /// for each of the loader's up to twelve components, the attribute
    /// (0-11) it holds, or 12-15 for four to sixteen bytes of padding.
    components: u64,
    stride: u32,
    component_count: u32,
}

fn read_loaders(registers: &[u32]) -> [AttributeLoader; REG_ATTRIBUTE_LOADER_COUNT] {
    std::array::from_fn(|i| {
        let base = REG_ATTRIBUTE_LOADER + i * REG_ATTRIBUTE_LOADER_STRIDE;
        let word1 = registers[base + 1];
        let word2 = registers[base + 2];
        // the third word holds components 8-11 in its low half, the bytes
        // per vertex in bits [23:16] and the component count in [31:28].
        AttributeLoader {
            offset: registers[base] & 0x0FFF_FFFF,
            components: word1 as u64 | (((word2 & 0xFFFF) as u64) << 32),
            // bits 16-23, not 24-31. wasted hours on this shit
            stride: (word2 >> 16) & 0xFF,
            component_count: (word2 >> 28) & 0xF,
        }
    })
}

/// (type, component_count) for attribute slot, from the combined 64-bit
/// format register.
fn attribute_format(registers: &[u32], slot: u32) -> (u32, u32) {
    let combined =
        registers[REG_ATTRIBUTE_FORMAT_LOW] as u64 | ((registers[REG_ATTRIBUTE_FORMAT_HIGH] as u64) << 32);
    let nibble = ((combined >> (slot * 4)) & 0xF) as u32;
    (nibble & 0x3, (nibble >> 2) + 1)
}

/// byte size of one component of the given attribute type (0=byte, 1=ubyte,
/// 2=short, 3=float).
fn component_size(ty: u32) -> u32 {
    match ty {
        0 | 1 => 1,
        2 => 2,
        _ => 4,
    }
}

fn fetch_component<M: GpuMemory>(memory: &mut M, addr: u32, ty: u32) -> f32 {
    match ty {
        0 => {
            let mut b = [0u8; 1];
            memory.read(addr, &mut b);
            b[0] as i8 as f32
        }
        1 => {
            let mut b = [0u8; 1];
            memory.read(addr, &mut b);
            b[0] as f32
        }
        2 => {
            let mut b = [0u8; 2];
            memory.read(addr, &mut b);
            i16::from_le_bytes(b) as f32
        }
        _ => {
            let mut b = [0u8; 4];
            memory.read(addr, &mut b);
            f32::from_bits(u32::from_le_bytes(b))
        }
    }
}

/// fetches one vertex's attributes and lays them out as shader input
/// registers, following the loader and input-register-map configuration.
fn fetch_vertex<M: GpuMemory>(
    registers: &[u32],
    memory: &mut M,
    // physical address of the attribute arrays.
    base: u32,
    loaders: &[AttributeLoader; REG_ATTRIBUTE_LOADER_COUNT],
    fixed: &[Vec4; 16],
    vertex_index: u32,
) -> [Vec4; shader::INPUT_REGISTERS] {
    // components an array does not provide read as (0, 0, 0, 1), a
    // two-component texture coordinate arrives as (u, v, 0, 1).
    let mut attributes = [[0.0, 0.0, 0.0, 1.0]; 12];
    let mut loaded = [false; 12];

    for loader in loaders {
        // base is physical, and a loader's offset can carry it from one
        // region into another, titles point the base at the start of VRAM and
        // reach vertex data in FCRAM through the offset.
        let vertex = memory.translate(base + loader.offset + vertex_index * loader.stride);
        let mut offset = 0u32;
        for slot in 0..loader.component_count.min(12) {
            let id = ((loader.components >> (slot * 4)) & 0xF) as usize;
            if id >= 12 {
                // padding, aligned to a word, then 4, 8, 12 or 16 bytes.
                offset = offset.next_multiple_of(4) + (id as u32 - 11) * 4;
                continue;
            }
            let (ty, count) = attribute_format(registers, id as u32);
            let size = component_size(ty);
            // each attribute starts aligned to its component size.
            offset = offset.next_multiple_of(size);
            let mut value = [0.0f32, 0.0, 0.0, 1.0];
            for (component, slot) in value.iter_mut().enumerate().take(count as usize) {
                *slot = fetch_component(memory, vertex + offset + component as u32 * size, ty);
            }
            offset += size * count;
            attributes[id] = value;
            loaded[id] = true;
        }
    }

    // attributes flagged as fixed take the value the command list set,
    // unless an array feeds them after all.
    let fixed_mask = (registers[REG_ATTRIBUTE_FORMAT_HIGH] >> 16) & 0xFFF;
    for (id, attribute) in attributes.iter_mut().enumerate() {
        if !loaded[id] && fixed_mask & (1 << id) != 0 {
            *attribute = fixed[id];
        }
    }

    let count = ((registers[REG_VS_NUM_INPUT_ATTRIBUTES] & 0xF) + 1) as usize;
    map_inputs(registers, REG_VS_BLOCK, &attributes[..count.min(12)])
}

/// places attributes in a shader unit's input registers (v0-v15), as the
/// unit's attribute permutation (in the block at block) assigns them.
fn map_inputs(registers: &[u32], block: usize, attributes: &[Vec4]) -> [Vec4; shader::INPUT_REGISTERS] {
    let map = registers[block + SHADER_INPUT_MAP_LOW] as u64
        | ((registers[block + SHADER_INPUT_MAP_HIGH] as u64) << 32);
    let mut input = [shader::ZERO; shader::INPUT_REGISTERS];
    for (id, value) in attributes.iter().enumerate() {
        let register = ((map >> (id * 4)) & 0xF) as usize;
        input[register] = *value;
    }
    input
}

/// one vertex after shading, clip-space position plus varyings.
#[derive(Clone, Copy)]
struct Vertex {
    clip: Vec4,
    color: Vec4,
    /// texture coordinate sets 0-2, each (u, v).
    texcoords: [[f32; 2]; 3],
}

impl Vertex {
    /// the vertex a fraction t of the way from self to other, which is
    /// how clipping makes new vertices where an edge crosses a plane.
    fn lerp(&self, other: &Vertex, t: f32) -> Vertex {
        let mix = |a: f32, b: f32| a + (b - a) * t;
        Vertex {
            clip: std::array::from_fn(|i| mix(self.clip[i], other.clip[i])),
            color: std::array::from_fn(|i| mix(self.color[i], other.color[i])),
            texcoords: std::array::from_fn(|set| {
                std::array::from_fn(|i| mix(self.texcoords[set][i], other.texcoords[set][i]))
            }),
        }
    }
}

/// where each varying the rasterizer needs lives in the shader's output
/// registers, as GPUREG_SH_OUTMAP_O* describes it.
#[derive(Debug, Clone, Copy)]
struct OutputMap {
    /// (register, component) for each of x, y, z, w.
    position: [(usize, usize); 4],
    color: [Option<(usize, usize)>; 4],
    /// (u, v) of texture coordinate sets 0, 1 and 2.
    texcoords: [[Option<(usize, usize)>; 2]; 3],
}

impl Default for OutputMap {
    fn default() -> Self {
        // the layout picasso/nihstro assign when a shader does not say
        // otherwise, used only when the registers describe nothing at all.
        OutputMap {
            position: [(0, 0), (0, 1), (0, 2), (0, 3)],
            color: [
                Some((2, 0)),
                Some((2, 1)),
                Some((2, 2)),
                Some((2, 3)),
            ],
            texcoords: [[Some((3, 0)), Some((3, 1))], [None; 2], [None; 2]],
        }
    }
}

fn read_output_map(registers: &[u32]) -> OutputMap {
    let total = (registers[REG_SHADER_OUTPUT_TOTAL] & 0x7).min(7) as usize;
    if total == 0 {
        return OutputMap::default();
    }

    let mut map = OutputMap {
        position: [(0, 0), (0, 1), (0, 2), (0, 3)],
        color: [None; 4],
        texcoords: [[None; 2]; 3],
    };
    let mut saw_position = false;

    for register in 0..total {
        let word = registers[REG_SHADER_OUTPUT_MAP + register];
        for component in 0..4 {
            // each byte names the semantic that output component carries.
            let semantic = ((word >> (component * 8)) & 0x1F) as usize;
            let slot = (register, component);
            match semantic {
                0..=3 => {
                    map.position[semantic] = slot;
                    saw_position = true;
                }
                8..=11 => map.color[semantic - 8] = Some(slot),
                12..=13 => map.texcoords[0][semantic - 12] = Some(slot),
                14..=15 => map.texcoords[1][semantic - 14] = Some(slot),
                22..=23 => map.texcoords[2][semantic - 22] = Some(slot),
                _ => {}
            }
        }
    }

    if !saw_position {
        return OutputMap::default();
    }
    map
}

/// runs the vertex shader over one vertex's inputs and returns the
/// attributes it outputs, packed as the rest of the pipeline sees them.
fn run_vertex_shader(unit: &ShaderUnit, registers: &[u32], input: [Vec4; shader::INPUT_REGISTERS]) -> [Vec4; 16] {
    let mut state = ShaderState::new();
    state.input = input;
    shader::run(unit, &mut state);
    pack_outputs(&state.output, registers[REG_VS_OUTPUT_MASK])
}

/// a stage's output attributes are the output registers its mask enables, in
/// order, attribute 2 is the third enabled register, which is not necessarily
/// o2.
fn pack_outputs(outputs: &[Vec4; shader::OUTPUT_REGISTERS], mask: u32) -> [Vec4; 16] {
    // a mask of zero is a stage nobody configured, take the registers as
    // they are rather than dropping everything.
    if mask & 0xFFFF == 0 {
        return *outputs;
    }
    let mut packed = [shader::ZERO; 16];
    let enabled = (0..shader::OUTPUT_REGISTERS).filter(|register| mask & (1 << register) != 0);
    for (slot, register) in enabled.enumerate() {
        packed[slot] = outputs[register];
    }
    packed
}

/// picks the rasterizer's varyings out of a vertex's output attributes.
fn to_vertex(map: &OutputMap, attributes: &[Vec4; 16]) -> Vertex {
    let get = |slot: (usize, usize)| attributes[slot.0][slot.1];
    // a shader that writes no color means "use white", not "use whatever
    // happened to be in that register", an untextured draw should come out
    // lit rather than black, and a textured one unmodulated.
    let color_or = |slot: Option<(usize, usize)>| slot.map_or(1.0, get);

    Vertex {
        clip: [
            get(map.position[0]),
            get(map.position[1]),
            get(map.position[2]),
            get(map.position[3]),
        ],
        color: [
            color_or(map.color[0]),
            color_or(map.color[1]),
            color_or(map.color[2]),
            color_or(map.color[3]),
        ],
        texcoords: map.texcoords.map(|set| [set[0].map_or(0.0, get), set[1].map_or(0.0, get)]),
    }
}

/// takes shader inputs through the vertex shader and, when one is enabled,
/// the geometry shader, producing the vertices the rasterizer assembles.
fn process_vertices(
    registers: &[u32],
    vertex_shader: &ShaderUnit,
    geometry_shader: &ShaderUnit,
    inputs: impl Iterator<Item = [Vec4; shader::INPUT_REGISTERS]>,
) -> Vec<Vertex> {
    let map = read_output_map(registers);
    let mut first = true;
    let outputs = inputs.map(|input| {
        if first && log::log_enabled!(log::Level::Trace) {
            first = false;
            let used_uniforms = vertex_shader.float_uniforms.iter().filter(|u| **u != shader::ZERO).count();
            log::trace!(
                "vertex shader: entry {}, {used_uniforms} non-zero uniforms, output mask 0x{:X}, \
                 first vertex inputs {:?}",
                vertex_shader.entry_point,
                registers[REG_VS_OUTPUT_MASK],
                &input[..8],
            );
        }
        run_vertex_shader(vertex_shader, registers, input)
    });
    if registers[REG_GEOSTAGE_CONFIG] & 0x3 == 2 {
        geometry_stage(registers, geometry_shader, &map, outputs)
    } else {
        outputs.map(|attributes| to_vertex(&map, &attributes)).collect()
    }
}

/// runs the geometry shader, in the mode particles and sprites use, every
/// vertex's outputs (or a fixed run of vertices') become one invocation's
/// inputs, and the triangles it emits are drawn as a list.
fn geometry_stage(
    registers: &[u32],
    unit: &ShaderUnit,
    map: &OutputMap,
    vertex_outputs: impl Iterator<Item = [Vec4; 16]>,
) -> Vec<Vertex> {
    let mode = registers[REG_GS_CONFIG] & 0xFF;
    if mode != 0 {
        log::warn!("geometry shader mode {mode} is not implemented");
        return Vec::new();
    }
    let per_vertex = ((registers[REG_VS_OUTPUT_TOTAL] & 0xF) + 1) as usize;
    let per_invocation = ((registers[REG_GS_BLOCK + SHADER_INPUT_CONFIG] & 0xF) + 1) as usize;
    let output_mask = registers[REG_GS_BLOCK + SHADER_OUTPUT_MASK];

    let mut pending: Vec<Vec4> = Vec::with_capacity(16);
    let mut vertices = Vec::new();
    for outputs in vertex_outputs {
        pending.extend_from_slice(&outputs[..per_vertex]);
        if pending.len() < per_invocation {
            continue;
        }
        let mut state = ShaderState::new();
        state.input = map_inputs(registers, REG_GS_BLOCK, &pending[..per_invocation]);
        pending.clear();

        let mut emitter = shader::Emitter::default();
        let input = state.input;
        shader::run_geometry(unit, &mut state, &mut emitter);
        if vertices.is_empty() && log::log_enabled!(log::Level::Trace) {
            let used_uniforms = unit.float_uniforms.iter().filter(|u| **u != shader::ZERO).count();
            log::trace!(
                "geometry shader: entry {} com mode 0x{:X}, {per_vertex} per vertex, \
                 {per_invocation} per invocation, input map 0x{:08X}, output mask 0x{:X}, \
                 {used_uniforms} non-zero uniforms, bools 0x{:X}, ints {:?}, inputs {:?}, \
                 first output {:?}, {} triangles",
                unit.entry_point,
                registers[REG_VS_COM_MODE],
                registers[REG_GS_BLOCK + SHADER_INPUT_MAP_LOW],
                output_mask,
                unit.bool_uniforms,
                unit.int_uniforms,
                &input[..per_invocation.min(4)],
                emitter.triangles.first().map(|t| &t[0][..4]),
                emitter.triangles.len(),
            );
        }
        for triangle in &emitter.triangles {
            for outputs in triangle {
                vertices.push(to_vertex(map, &pack_outputs(outputs, output_mask)));
            }
        }
    }
    vertices
}

/// screen-space vertex, with every varying pre-divided by w so the
/// rasterizer only has to interpolate linearly and divide once per pixel.
#[derive(Clone, Copy, Debug)]
struct Screen {
    x: f32,
    y: f32,
    /// z/w, which is linear in screen space, the depth map turns it into
    /// the value the depth buffer holds.
    z: f32,
    inv_w: f32,
    color_over_w: Vec4,
    texcoords_over_w: [[f32; 2]; 3],
}

fn to_screen(vertex: Vertex, viewport: (f32, f32, f32, f32)) -> Option<Screen> {
    let (vx, vy, vw, vh) = viewport;
    let w = vertex.clip[3];
    if w.abs() < 1e-8 {
        return None;
    }
    let inv_w = 1.0 / w;
    let ndc_x = vertex.clip[0] * inv_w;
    let ndc_y = vertex.clip[1] * inv_w;

    // window coordinates, y points up, from the bottom of the buffer, the
    // way the PICA's viewport is defined.
    Some(Screen {
        x: vx + (ndc_x * 0.5 + 0.5) * vw,
        y: vy + (ndc_y * 0.5 + 0.5) * vh,
        z: vertex.clip[2] * inv_w,
        inv_w,
        color_over_w: vertex.color.map(|c| c * inv_w),
        texcoords_over_w: vertex.texcoords.map(|[u, v]| [u * inv_w, v * inv_w]),
    })
}

/// clips a triangle to the volume the PICA draws, w positive, and -w <= z <= 0.
fn clip_triangle(triangle: [Vertex; 3]) -> Vec<Vertex> {
    const EPSILON: f32 = 1e-5;
    // signed distances to each plane, a vertex is inside when all are >= 0.
    let planes: [fn(&Vertex) -> f32; 3] = [
        |v| v.clip[3] - EPSILON,
        |v| -v.clip[2],
        |v| v.clip[2] + v.clip[3],
    ];
    if triangle.iter().all(|v| planes.iter().all(|plane| plane(v) >= 0.0)) {
        return triangle.to_vec();
    }

    let mut polygon = triangle.to_vec();
    for plane in planes {
        let mut clipped = Vec::with_capacity(polygon.len() + 1);
        for (i, current) in polygon.iter().enumerate() {
            let next = &polygon[(i + 1) % polygon.len()];
            let (d0, d1) = (plane(current), plane(next));
            if d0 >= 0.0 {
                clipped.push(*current);
            }
            if (d0 >= 0.0) != (d1 >= 0.0) {
                clipped.push(current.lerp(next, d0 / (d0 - d1)));
            }
        }
        polygon = clipped;
        if polygon.len() < 3 {
            return Vec::new();
        }
    }
    polygon
}

/// texture unit 0's configuration and backing data for one draw call.
struct BoundTexture {
    data: Vec<u8>,
    /// bilinear rather than point sampling when magnifying.
    linear: bool,
    wrap_s: Wrap,
    wrap_t: Wrap,
    format: crate::texture::TextureFormat,
    width: u32,
    height: u32,
}

/// how a texture coordinate outside 0..1 is brought back inside.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Wrap {
    ClampToEdge,
    /// the PICA's border color is not modelled, clamping to the edge is
    /// what a title sees wherever its border matches the edge texel, which
    /// is by far the common case.
    ClampToBorder,
    Repeat,
    MirroredRepeat,
}

impl Wrap {
    fn from_raw(value: u32) -> Wrap {
        // three bits, of which only the low two are documented, the upper
        // four encodings behave like these, as Citra established.
        match value & 0x7 {
            0 | 4 => Wrap::ClampToEdge,
            1 | 5 => Wrap::ClampToBorder,
            3 => Wrap::MirroredRepeat,
            _ => Wrap::Repeat,
        }
    }

    /// maps an integer texel coordinate into 0..size.
    fn apply(self, coordinate: i32, size: u32) -> u32 {
        let size = size as i32;
        let wrapped = match self {
            Wrap::ClampToEdge | Wrap::ClampToBorder => coordinate.clamp(0, size - 1),
            Wrap::Repeat => coordinate.rem_euclid(size),
            Wrap::MirroredRepeat => {
                let period = coordinate.rem_euclid(size * 2);
                if period < size {
                    period
                } else {
                    size * 2 - 1 - period
                }
            }
        };
        wrapped as u32
    }
}

impl BoundTexture {
    fn texel(&self, x: i32, y: i32) -> [f32; 4] {
        let x = self.wrap_s.apply(x, self.width);
        let y = self.wrap_t.apply(y, self.height);
        let t = crate::texture::sample_texel(&self.data, self.format, self.width, x, y);
        t.map(|c| c as f32 / 255.0)
    }

    /// samples at (u, v), filtering the way the unit is configured.
    fn sample(&self, u: f32, v: f32) -> [f32; 4] {
        // PICA texture coordinates have v=0 at the bottom row.
        let fx = u * self.width as f32;
        let fy = (1.0 - v) * self.height as f32;

        if !self.linear {
            return self.texel(fx.floor() as i32, fy.floor() as i32);
        }

        // texel centers sit at half-integer positions.
        let fx = fx - 0.5;
        let fy = fy - 0.5;
        let x0 = fx.floor();
        let y0 = fy.floor();
        let tx = fx - x0;
        let ty = fy - y0;
        let (x0, y0) = (x0 as i32, y0 as i32);

        let a = self.texel(x0, y0);
        let b = self.texel(x0 + 1, y0);
        let c = self.texel(x0, y0 + 1);
        let d = self.texel(x0 + 1, y0 + 1);
        std::array::from_fn(|i| {
            let top = a[i] + (b[i] - a[i]) * tx;
            let bottom = c[i] + (d[i] - c[i]) * tx;
            top + (bottom - top) * ty
        })
    }
}

/// first register of each texture unit's block.
const TEXTURE_UNIT_BASES: [usize; 3] = [0x081, 0x091, 0x099];

/// reads the configuration of one texture unit and copies its image out of
/// guest memory, or None when the unit is disabled or unusable.
fn bind_texture<M: GpuMemory>(
    registers: &[u32],
    memory: &mut M,
    unit: usize,
) -> Option<BoundTexture> {
    // one bit per unit in GPUREG_TEXUNIT_CONFIG.
    if registers[REG_TEXTURE_CONFIG] & (1 << unit) == 0 {
        return None;
    }

    let base = TEXTURE_UNIT_BASES[unit];
    let dimensions = registers[base + 1];
    let height = dimensions & 0x7FF;
    let width = (dimensions >> 16) & 0x7FF;
    if width == 0 || height == 0 {
        return None;
    }

    let format_register = if unit == 0 { base + 13 } else { base + 5 };
    let format = crate::texture::TextureFormat::from_raw(registers[format_register]);
    let address = loc_register(registers, base + 4);
    if address == 0 {
        // a unit pointed at nothing.
        return None;
    }

    let addr = memory.translate(address);
    log::trace!("texture unit {unit}: 0x{address:08X} -> 0x{addr:08X}, {width}x{height} {format:?}");
    let bits = (width as u64) * (height as u64) * format.bits_per_pixel() as u64;
    let size = bits.div_ceil(8) as usize;

    let mut data = vec![0u8; size];
    memory.read(addr, &mut data);

    // filter mode in bit 1 (magnification) and 2 (minification), the wrap
    // modes for T and S in bits 8-10 and 12-14.
    let config = registers[base + 2];
    Some(BoundTexture {
        data,
        linear: config & 0x2 != 0,
        wrap_t: Wrap::from_raw(config >> 8),
        wrap_s: Wrap::from_raw(config >> 12),
        format,
        width,
        height,
    })
}

/// where and how depth testing reads/writes, or None when disabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compare {
    Never,
    Always,
    Equal,
    NotEqual,
    Less,
    LessOrEqual,
    Greater,
    GreaterOrEqual,
}

impl Compare {
    fn from_raw(value: u32) -> Compare {
        match value & 0x7 {
            0 => Compare::Never,
            1 => Compare::Always,
            2 => Compare::Equal,
            3 => Compare::NotEqual,
            4 => Compare::Less,
            5 => Compare::LessOrEqual,
            6 => Compare::Greater,
            _ => Compare::GreaterOrEqual,
        }
    }

    fn passes<T: PartialOrd>(self, value: T, reference: T) -> bool {
        match self {
            Compare::Never => false,
            Compare::Always => true,
            Compare::Equal => value == reference,
            Compare::NotEqual => value != reference,
            Compare::Less => value < reference,
            Compare::LessOrEqual => value <= reference,
            Compare::Greater => value > reference,
            Compare::GreaterOrEqual => value >= reference,
        }
    }
}

/// discards fragments whose alpha fails a comparison.
#[derive(Debug, Clone, Copy)]
struct AlphaTest {
    function: Compare,
    reference: u8,
}

impl AlphaTest {
    fn read(registers: &[u32]) -> Option<AlphaTest> {
        let config = registers[REG_ALPHA_TEST];
        (config & 1 != 0).then(|| AlphaTest {
            function: Compare::from_raw(config >> 4),
            reference: ((config >> 8) & 0xFF) as u8,
        })
    }

    fn passes(&self, alpha: u8) -> bool {
        self.function.passes(alpha, self.reference)
    }
}

/// what a stencil update does to the stored value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StencilOp {
    Keep,
    Zero,
    Replace,
    IncrementSaturate,
    DecrementSaturate,
    Invert,
    IncrementWrap,
    DecrementWrap,
}

impl StencilOp {
    fn from_raw(value: u32) -> StencilOp {
        match value & 0x7 {
            0 => StencilOp::Keep,
            1 => StencilOp::Zero,
            2 => StencilOp::Replace,
            3 => StencilOp::IncrementSaturate,
            4 => StencilOp::DecrementSaturate,
            5 => StencilOp::Invert,
            6 => StencilOp::IncrementWrap,
            _ => StencilOp::DecrementWrap,
        }
    }

    fn apply(self, value: u8, reference: u8) -> u8 {
        match self {
            StencilOp::Keep => value,
            StencilOp::Zero => 0,
            StencilOp::Replace => reference,
            StencilOp::IncrementSaturate => value.saturating_add(1),
            StencilOp::DecrementSaturate => value.saturating_sub(1),
            StencilOp::Invert => !value,
            StencilOp::IncrementWrap => value.wrapping_add(1),
            StencilOp::DecrementWrap => value.wrapping_sub(1),
        }
    }
}

/// GPUREG_STENCIL_TEST and GPUREG_STENCIL_OP.
#[derive(Debug, Clone, Copy)]
struct Stencil {
    function: Compare,
    reference: u8,
    /// bits the comparison looks at.
    compare_mask: u8,
    /// bits an update may change.
    write_mask: u8,
    /// applied when the stencil test fails, when it passes but the depth
    /// test fails, and when both pass.
    fail: StencilOp,
    depth_fail: StencilOp,
    pass: StencilOp,
}

impl Stencil {
    fn read(registers: &[u32]) -> Option<Stencil> {
        let test = registers[REG_STENCIL_TEST];
        if test & 1 == 0 {
            return None;
        }
        let op = registers[REG_STENCIL_OP];
        Some(Stencil {
            function: Compare::from_raw(test >> 4),
            write_mask: (test >> 8) as u8,
            reference: (test >> 16) as u8,
            compare_mask: (test >> 24) as u8,
            fail: StencilOp::from_raw(op),
            depth_fail: StencilOp::from_raw(op >> 4),
            pass: StencilOp::from_raw(op >> 8),
        })
    }

    /// the reference, not the stored value, is on the left of the comparison.
    fn passes(&self, stored: u8) -> bool {
        self.function
            .passes(self.reference & self.compare_mask, stored & self.compare_mask)
    }

    fn update(&self, stored: u8, op: StencilOp) -> u8 {
        let new = op.apply(stored, self.reference);
        (stored & !self.write_mask) | (new & self.write_mask)
    }
}

/// the depth/stencil buffer, and what a draw does with it.
struct DepthStencil {
    addr: u32,
    /// bytes per sample, 2 for D16, 3 for D24, 4 for D24S8, which keeps
    /// the stencil in the top byte.
    bytes: u32,
    test: Option<Compare>,
    write_depth: bool,
    stencil: Option<Stencil>,
    write_stencil: bool,
}

impl DepthStencil {
    fn read<M: GpuMemory>(registers: &[u32], memory: &M) -> Option<DepthStencil> {
        // GPUREG_DEPTH_COLOR_MASK, bit 0 enables the depth test, bits 4-6
        // pick the comparison, bit 12 enables writing depth back.
        let mask = registers[REG_DEPTH_COLOR_MASK];
        let writable = registers[REG_DEPTH_STENCIL_WRITE] != 0;
        let format = registers[REG_DEPTH_BUFFER_FORMAT] & 0x3;
        let test = (mask & 1 != 0).then(|| Compare::from_raw(mask >> 4));
        let write_depth = writable && mask & (1 << 12) != 0;
        let stencil = if format == 3 { Stencil::read(registers) } else { None };
        if test.is_none() && !write_depth && stencil.is_none() {
            return None;
        }
        Some(DepthStencil {
            addr: memory.translate(loc_register(registers, REG_DEPTH_BUFFER_ADDRESS)),
            bytes: match format {
                0 => 2,
                2 => 3,
                _ => 4,
            },
            test,
            write_depth,
            stencil,
            write_stencil: writable,
        })
    }

    /// the stored depth, as 0..1, and stencil.
    fn read_sample<M: GpuMemory>(&self, memory: &mut M, index: u32) -> (f32, u8) {
        let mut raw = [0u8; 4];
        memory.read(self.addr + index * self.bytes, &mut raw[..self.bytes as usize]);
        let value = u32::from_le_bytes(raw);
        match self.bytes {
            2 => (value as f32 / 65535.0, 0),
            _ => ((value & 0x00FF_FFFF) as f32 / 16_777_215.0, (value >> 24) as u8),
        }
    }

    fn write_depth<M: GpuMemory>(&self, memory: &mut M, index: u32, depth: f32) {
        let address = self.addr + index * self.bytes;
        if self.bytes == 2 {
            memory.write(address, &((depth * 65535.0) as u16).to_le_bytes());
        } else {
            // only the low three bytes, the stencil shares the word.
            memory.write(address, &((depth * 16_777_215.0) as u32).to_le_bytes()[..3]);
        }
    }

    fn write_stencil<M: GpuMemory>(&self, memory: &mut M, index: u32, value: u8) {
        if self.write_stencil && self.bytes == 4 {
            memory.write(self.addr + index * 4 + 3, &[value]);
        }
    }
}

/// how clip-space depth becomes the value in the depth buffer, z/w scaled and
/// offset by GPUREG_DEPTHMAP_SCALE and _OFFSET.
#[derive(Debug, Clone, Copy)]
struct DepthMap {
    scale: f32,
    offset: f32,
    /// a w-buffer multiplies the mapped value by w.
    w_buffer: bool,
}

impl DepthMap {
    fn read(registers: &[u32]) -> DepthMap {
        DepthMap {
            scale: read_float24(registers, REG_VIEWPORT_DEPTH_RANGE),
            offset: read_float24(registers, REG_VIEWPORT_DEPTH_NEAR),
            w_buffer: registers[REG_DEPTHMAP_ENABLE] & 1 == 0,
        }
    }
}

/// the color buffer a draw renders into.
struct ColorTarget {
    addr: u32,
    /// the pixels a draw may touch, in window coordinates, the viewport,
    /// cut to the buffer.
    left: i32,
    bottom: i32,
    right: i32,
    top: i32,
    /// the buffer's size.
    buffer_width: u32,
    buffer_height: u32,
    format: ColorFormat,
    /// which of red, green, blue and alpha the draw may change.
    write: [bool; 4],
}

/// what happened to the fragments of a draw, for tracing why a draw that
/// covers the screen leaves nothing on it.
#[derive(Debug, Default, Clone, Copy)]
struct FillStats {
    written: u32,
    depth_failed: u32,
    stencil_failed: u32,
    alpha_failed: u32,
}

/// everything about a draw that stays the same from triangle to triangle.
struct DrawState<'a> {
    target: ColorTarget,
    depth_map: DepthMap,
    depth_stencil: Option<DepthStencil>,
    tex_env: crate::tev::TexEnv,
    alpha_test: Option<AlphaTest>,
    blend: Option<crate::blend::Blend>,
    textures: &'a [Option<BoundTexture>; 3],
    /// texture unit 2 can read coordinate set 1 instead of its own.
    texture2_uses_coord1: bool,
}

/// rasterizes one triangle, texturing and the combiners, then the alpha,
/// stencil and depth tests in the order the hardware runs them, then
/// blending and the color write.
fn fill_triangle<M: GpuMemory>(
    memory: &mut M,
    [a, b, c]: [Screen; 3],
    state: &DrawState,
    stats: &mut FillStats,
) {
    let target = &state.target;
    let min_x = (a.x.min(b.x).min(c.x).floor() as i32).max(target.left);
    let max_x = (a.x.max(b.x).max(c.x).ceil() as i32).min(target.right);
    let min_y = (a.y.min(b.y).min(c.y).floor() as i32).max(target.bottom);
    let max_y = (a.y.max(b.y).max(c.y).ceil() as i32).min(target.top);
    if min_x >= max_x || min_y >= max_y {
        return;
    }

    // wind every triangle the same way, so that "inside" is the positive
    // side of all three edges.
    let mut area = Edge::new(&a, &b).at(c.x, c.y);
    let (b, c) = if area < 0.0 {
        area = -area;
        (c, b)
    } else {
        (b, c)
    };
    // zero area means the triangle is degenerate.
    if area < 1e-6 {
        return;
    }
    let inverse_area = 1.0 / area;
    let edges = [Edge::new(&b, &c), Edge::new(&c, &a), Edge::new(&a, &b)];

    let bpp = target.format.bytes_per_pixel();
    let writes_color = target.write.iter().any(|&w| w);
    let partial_write = writes_color && !target.write.iter().all(|&w| w);
    let any_texture = state.textures.iter().any(Option::is_some);
    let mut pixel = [0u8; 4];

    for y in min_y..max_y {
        for x in min_x..max_x {
            let px = x as f32 + 0.5;
            let py = y as f32 + 0.5;

            // barycentric weights via the edge functions.
            let values = edges.map(|edge| edge.at(px, py));
            if !edges.iter().zip(values).all(|(edge, value)| edge.covers(value)) {
                continue;
            }
            let [w0, w1, w2] = values.map(|value| value * inverse_area);

            // perspective-correct interpolation, interpolate attribute/w and
            // 1/w linearly in screen space, then divide.
            let inv_w = w0 * a.inv_w + w1 * b.inv_w + w2 * c.inv_w;
            if inv_w.abs() < 1e-8 {
                continue;
            }
            let w = 1.0 / inv_w;

            let mut depth = (w0 * a.z + w1 * b.z + w2 * c.z) * state.depth_map.scale
                + state.depth_map.offset;
            if state.depth_map.w_buffer {
                depth *= w;
            }
            let depth = depth.clamp(0.0, 1.0);

            let row = target.buffer_height - 1 - y as u32;
            let index = crate::format::morton_offset(x as u32, row, target.buffer_width, 1);

            // with no stencil to update, a fragment that fails the depth
            // test is gone whatever its color, so skip the shading.
            let stored = state
                .depth_stencil
                .as_ref()
                .map(|buffer| buffer.read_sample(memory, index));
            if let (Some(buffer), Some((stored_depth, _))) = (&state.depth_stencil, stored) {
                if buffer.stencil.is_none()
                    && buffer.test.is_some_and(|test| !test.passes(depth, stored_depth))
                {
                    stats.depth_failed += 1;
                    continue;
                }
            }

            let interpolate = |values: [f32; 3]| (w0 * values[0] + w1 * values[1] + w2 * values[2]) * w;
            let color: Vec4 = std::array::from_fn(|i| {
                interpolate([a.color_over_w[i], b.color_over_w[i], c.color_over_w[i]]).clamp(0.0, 1.0)
            });

            // sample the bound textures, then let the texture environment
            // decide what the fragment's color actually is, the samples and
            // the vertex color are only its inputs.
            let mut samples = [[0.0f32, 0.0, 0.0, 1.0]; 4];
            if any_texture {
                for (unit, texture) in state.textures.iter().enumerate() {
                    let Some(texture) = texture else { continue };
                    let set = match unit {
                        2 if state.texture2_uses_coord1 => 1,
                        unit => unit,
                    };
                    let u = interpolate([a.texcoords_over_w[set][0], b.texcoords_over_w[set][0], c.texcoords_over_w[set][0]]);
                    let v = interpolate([a.texcoords_over_w[set][1], b.texcoords_over_w[set][1], c.texcoords_over_w[set][1]]);
                    samples[unit] = texture.sample(u, v);
                }
            }

            let combined = state.tex_env.apply(color, samples);
            let mut rgba = combined.map(|c| (c * 255.0) as u8);

            if let Some(test) = state.alpha_test {
                if !test.passes(rgba[3]) {
                    stats.alpha_failed += 1;
                    continue;
                }
            }

            if let (Some(buffer), Some((stored_depth, stored_stencil))) = (&state.depth_stencil, stored) {
                if let Some(stencil) = &buffer.stencil {
                    if !stencil.passes(stored_stencil) {
                        buffer.write_stencil(memory, index, stencil.update(stored_stencil, stencil.fail));
                        stats.stencil_failed += 1;
                        continue;
                    }
                }
                let depth_passes = buffer.test.is_none_or(|test| test.passes(depth, stored_depth));
                if let Some(stencil) = &buffer.stencil {
                    let op = if depth_passes { stencil.pass } else { stencil.depth_fail };
                    buffer.write_stencil(memory, index, stencil.update(stored_stencil, op));
                }
                if !depth_passes {
                    stats.depth_failed += 1;
                    continue;
                }
                if buffer.write_depth {
                    buffer.write_depth(memory, index, depth);
                }
            }

            if !writes_color {
                continue;
            }

            // combine with what is already in the buffer, the way the output
            // merger is configured to, and keep the channels the draw may not
            // change.
            let address = target.addr + index * bpp as u32;
            if state.blend.is_some() || partial_write {
                memory.read(address, &mut pixel[..bpp]);
                let existing = target.format.decode(&pixel[..bpp]);
                if let Some(blend) = &state.blend {
                    let blended = blend.apply(
                        rgba.map(|c| c as f32 / 255.0),
                        existing.map(|c| c as f32 / 255.0),
                    );
                    rgba = blended.map(|c| (c * 255.0) as u8);
                }
                for channel in 0..4 {
                    if !target.write[channel] {
                        rgba[channel] = existing[channel];
                    }
                }
            }

            target.format.encode(rgba, &mut pixel[..bpp]);
            memory.write(address, &pixel[..bpp]);
            stats.written += 1;
        }
    }
}

/// one edge of a triangle, as a function that is zero on the edge and positive
/// on the triangle's side of it.
#[derive(Debug, Clone, Copy)]
struct Edge {
    x: f32,
    y: f32,
    dx: f32,
    dy: f32,
    sign: f32,
    owns_ties: bool,
}

impl Edge {
    fn new(from: &Screen, to: &Screen) -> Edge {
        let (start, end, sign) = if (from.x, from.y) <= (to.x, to.y) {
            (from, to, 1.0)
        } else {
            (to, from, -1.0)
        };
        let (dx, dy) = (to.x - from.x, to.y - from.y);
        Edge {
            x: start.x,
            y: start.y,
            dx: end.x - start.x,
            dy: end.y - start.y,
            sign,
            owns_ties: dy > 0.0 || (dy == 0.0 && dx < 0.0),
        }
    }

    #[inline]
    fn at(&self, px: f32, py: f32) -> f32 {
        self.sign * (self.dx * (py - self.y) - self.dy * (px - self.x))
    }

    #[inline]
    fn covers(&self, value: f32) -> bool {
        value > 0.0 || (value == 0.0 && self.owns_ties)
    }
}

/// executes a draw call, fetches vertices, shades them, assembles triangles
/// and fills them. Returns the number of vertices processed.
pub fn draw<M: GpuMemory>(
    registers: &[u32],
    vertex_shader: &ShaderUnit,
    geometry_shader: &ShaderUnit,
    fixed_attributes: &[Vec4; 16],
    memory: &mut M,
    indexed: bool,
) -> u32 {
    let vertex_count = registers[REG_VERTEX_COUNT];
    let first_vertex = registers[REG_VERTEX_OFFSET];
    if vertex_count == 0 || vertex_count > 0x1_0000 {
        return 0;
    }

    let attribute_base = loc_register(registers, REG_ATTRIBUTE_BASE);
    let loaders = read_loaders(registers);

    // resolve each of the vertex_count draw indices to an actual vertex
    // array index, sequential for DrawArrays, looked up in the index buffer
    // for DrawElements.
    let index_config = registers[REG_INDEX_ARRAY];
    let index_short = index_config & 0x8000_0000 != 0;
    let index_base = memory.translate(attribute_base + (index_config & 0x0FFF_FFFF));

    let resolve_index = |memory: &mut M, i: u32| -> u32 {
        if !indexed {
            return first_vertex + i;
        }
        if index_short {
            let mut b = [0u8; 2];
            memory.read(index_base + i * 2, &mut b);
            u16::from_le_bytes(b) as u32
        } else {
            let mut b = [0u8; 1];
            memory.read(index_base + i, &mut b);
            b[0] as u32
        }
    };

    // shade every vertex once.
    log::trace!(
        "array draw: {vertex_count} vertices from 0x{attribute_base:08X}, indexed {indexed}, formats \
         0x{:08X}{:08X}, loaders {:?}",
        registers[REG_ATTRIBUTE_FORMAT_HIGH],
        registers[REG_ATTRIBUTE_FORMAT_LOW],
        loaders
            .iter()
            .filter(|l| l.component_count > 0)
            .map(|l| (l.offset, l.components, l.stride, l.component_count))
            .collect::<Vec<_>>(),
    );
    let inputs: Vec<_> = (0..vertex_count)
        .map(|i| {
            let vertex_index = resolve_index(memory, i);
            fetch_vertex(registers, memory, attribute_base, &loaders, fixed_attributes, vertex_index)
        })
        .collect();
    let shaded = process_vertices(registers, vertex_shader, geometry_shader, inputs.into_iter());

    rasterize(registers, memory, &shaded);
    vertex_count
}

/// draws vertices a title sent one attribute at a time through the fixed
/// attribute registers ("immediate mode") instead of from arrays in memory.
pub fn draw_immediate<M: GpuMemory>(
    registers: &[u32],
    vertex_shader: &ShaderUnit,
    geometry_shader: &ShaderUnit,
    memory: &mut M,
    vertices: &[[Vec4; 16]],
) -> u32 {
    // immediate mode sizes its vertices by GPUREG_VSH_NUM_ATTR.
    let count = ((registers[REG_VS_ATTRIBUTE_COUNT] & 0xF) + 1) as usize;
    log::trace!("immediate draw: {} vertices of {count} attributes", vertices.len());
    let inputs = vertices
        .iter()
        .map(|attributes| map_inputs(registers, REG_VS_BLOCK, &attributes[..count]));
    let shaded = process_vertices(registers, vertex_shader, geometry_shader, inputs);
    rasterize(registers, memory, &shaded)
}

/// assembles shaded vertices into triangles the way the primitive configuration
/// says, and fills them with the current back-end state.
fn rasterize<M: GpuMemory>(registers: &[u32], memory: &mut M, shaded: &[Vertex]) -> u32 {
    let vertex_count = shaded.len();
    // the offset is two signed 10-bit fields.
    let signed10 = |value: u32| (((value & 0x3FF) << 22) as i32 >> 22) as f32;
    let viewport_x = signed10(registers[REG_VIEWPORT_XY]);
    let viewport_y = signed10(registers[REG_VIEWPORT_XY] >> 16);
    // the viewport registers hold *half* the extent, because what the hardware
    // actually wants is the scale factor that maps clip space (-1..1) onto the
    // target.
    let viewport_width = read_float24(registers, REG_VIEWPORT_WIDTH) * 2.0;
    let viewport_height = read_float24(registers, REG_VIEWPORT_HEIGHT) * 2.0;
    if viewport_width <= 0.0 || viewport_height <= 0.0 {
        return 0;
    }
    let viewport = (viewport_x, viewport_y, viewport_width, viewport_height);

    let color_buffer_raw = loc_register(registers, REG_COLOR_BUFFER_ADDRESS);
    let target_addr = memory.translate(color_buffer_raw);
    // the format sits in bits 16-18 of the register, not the low bits, the low
    // half holds how many bytes a pixel takes.
    let target_format = ColorFormat::from_raw((registers[REG_COLOR_BUFFER_FORMAT] >> 16) & 7);
    // the buffer's real size, which the tile addressing uses, it is often
    // padded past the viewport, and tiling at the viewport's width instead
    // reads back as the image repeating down the screen.
    let dimensions = registers[REG_FRAMEBUFFER_DIMENSIONS];
    let viewport_right = (viewport_x + viewport_width).round() as i32;
    let viewport_top = (viewport_y + viewport_height).round() as i32;
    let buffer_width = (dimensions & 0x7FF).max(viewport_right.max(1) as u32);
    let buffer_height = (((dimensions >> 12) & 0x3FF) + 1).max(viewport_top.max(1) as u32);
    let (target_width, target_height) = (buffer_width, buffer_height);

    // draws are far more frequent than fills or transfers (thousands per
    // second versus dozens), so this stays at trace level to keep debug
    // usable for spotting the rarer GX commands.
    if log::log_enabled!(log::Level::Trace) {
        log::trace!(
            "draw: {vertex_count} vertices, topology reg 0x{:X}, viewport \
             {viewport_width}x{viewport_height} @({viewport_x},{viewport_y}), color buffer raw \
             0x{color_buffer_raw:08X} -> 0x{target_addr:08X} {target_format:?}, \
             texture0_addr=0x{:08X} texture0_config=0x{:08X} texture0_format=0x{:X}",
            registers[REG_PRIMITIVE_CONFIG],
            registers[REG_TEXTURE0_ADDRESS],
            registers[REG_TEXTURE_CONFIG],
            registers[REG_TEXTURE0_FORMAT],
        );
    }

    // color writes need the buffer to allow them at all, then each
    // channel's bit in GPUREG_DEPTH_COLOR_MASK.
    let color_writable = registers[REG_COLOR_BUFFER_WRITE] != 0;
    let color_mask = registers[REG_DEPTH_COLOR_MASK] >> 8;
    let textures: [Option<BoundTexture>; 3] =
        std::array::from_fn(|unit| bind_texture(registers, memory, unit));
    let state = DrawState {
        target: ColorTarget {
            addr: target_addr,
            // drawing stays inside the viewport.
            left: (viewport_x.round() as i32).max(0),
            bottom: (viewport_y.round() as i32).max(0),
            right: viewport_right.min(buffer_width as i32),
            top: viewport_top.min(buffer_height as i32),
            buffer_width,
            buffer_height,
            format: target_format,
            write: std::array::from_fn(|channel| color_writable && color_mask & (1 << channel) != 0),
        },
        depth_map: DepthMap::read(registers),
        depth_stencil: DepthStencil::read(registers, memory),
        tex_env: crate::tev::TexEnv::read(registers),
        alpha_test: AlphaTest::read(registers),
        blend: crate::blend::Blend::read(registers),
        textures: &textures,
        texture2_uses_coord1: registers[REG_TEXTURE_CONFIG] & (1 << 13) != 0,
    };

    // GPUREG_PRIMITIVE_CONFIG bits [9:8], 0 = triangle list, 1 = strip,
    // 2 = fan, 3 = whatever the geometry shader emitted, which is a list.
    let topology = (registers[REG_PRIMITIVE_CONFIG] >> 8) & 0x3;

    let triangle_indices: Vec<(usize, usize, usize)> = match topology {
        1 => (2..shaded.len())
            .map(|i| {
                if i % 2 == 0 {
                    (i - 2, i - 1, i)
                } else {
                    (i - 1, i - 2, i)
                }
            })
            .collect(),
        2 => (2..shaded.len()).map(|i| (0, i - 1, i)).collect(),
        _ => (0..shaded.len() / 3).map(|t| (t * 3, t * 3 + 1, t * 3 + 2)).collect(),
    };

    // GPUREG_FACECULLING_CONFIG, 0 keeps everything, 1 keeps triangles
    // wound clockwise and 2 counter-clockwise, as seen with y pointing up.
    let cull_mode = registers[REG_FACE_CULLING] & 0x3;

    let triangle_count = triangle_indices.len();
    let mut clipped_away = 0u32;
    let mut culled = 0u32;
    let mut stats = FillStats::default();
    for (ia, ib, ic) in triangle_indices {
        let polygon = clip_triangle([shaded[ia], shaded[ib], shaded[ic]]);
        let screen: Vec<Screen> = polygon.iter().filter_map(|v| to_screen(*v, viewport)).collect();
        if screen.len() < 3 || screen.len() != polygon.len() {
            clipped_away += 1;
            continue;
        }
        if cull_mode != 0 {
            let [p, q, r] = [screen[0], screen[1], screen[2]];
            let counter_clockwise = (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x) > 0.0;
            if counter_clockwise == (cull_mode == 1) {
                culled += 1;
                continue;
            }
        }
        for i in 1..screen.len() - 1 {
            fill_triangle(memory, [screen[0], screen[i], screen[i + 1]], &state, &mut stats);
        }
    }

    if log::log_enabled!(log::Level::Trace) {
        let first_screen = shaded.first().and_then(|v| to_screen(*v, viewport));
        log::trace!(
            "draw result: {vertex_count} verts, {triangle_count} tris ({clipped_away} clipped \
             away, {culled} culled), {} pixels written ({} failed depth, {} failed stencil, {} failed alpha), \
             target 0x{target_addr:08X} {target_width}x{target_height} {target_format:?}, \
             texture={} first vertex clip={:?} screen(x,y,inv_w)={first_screen:?}, depth/color \
             mask 0x{:08X}, color op 0x{:08X}, blend 0x{:08X}, depth map {:?}, tev0 {:08X?}, \
             tev1 {:08X?}, tev buffer 0x{:08X}",
            stats.written,
            stats.depth_failed,
            stats.stencil_failed,
            stats.alpha_failed,
            textures[0].is_some(),
            shaded.first().map(|v| v.clip),
            registers[REG_DEPTH_COLOR_MASK],
            registers[REG_COLOR_OPERATION],
            registers[REG_BLEND_FUNC],
            state.depth_map,
            &registers[0x0C0..0x0C5],
            &registers[0x0C8..0x0CD],
            registers[0x0E0],
        );
    }

    stats.written
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// guest memory seen the way the GPU sees it, physical VRAM and FCRAM
    /// translate to different virtual windows.
    #[derive(Default)]
    struct ConsoleMemory(HashMap<u32, u8>);

    impl GpuMemory for ConsoleMemory {
        fn read(&mut self, addr: u32, out: &mut [u8]) {
            for (i, byte) in out.iter_mut().enumerate() {
                *byte = self.0.get(&(addr + i as u32)).copied().unwrap_or(0);
            }
        }

        fn write(&mut self, addr: u32, data: &[u8]) {
            for (i, byte) in data.iter().enumerate() {
                self.0.insert(addr + i as u32, *byte);
            }
        }

        fn translate(&self, paddr: u32) -> u32 {
            match paddr {
                0x2000_0000.. => 0x1400_0000 + (paddr - 0x2000_0000),
                0x1800_0000..=0x185F_FFFF => 0x1F00_0000 + (paddr - 0x1800_0000),
                _ => paddr,
            }
        }
    }

    /// titles point the attribute base at the start of VRAM and reach their
    /// vertex data in FCRAM through each loader's offset.
    #[test]
    fn a_loader_offset_can_reach_another_memory_region() {
        let mut memory = ConsoleMemory::default();
        // vertex data at physical 0x20000100, virtual 0x14000100.
        for (i, value) in [1.5f32, -2.0, 0.25, 1.0].iter().enumerate() {
            memory.write(0x1400_0100 + i as u32 * 4, &value.to_le_bytes());
        }

        let mut registers = vec![0u32; 0x300];
        registers[REG_ATTRIBUTE_FORMAT_LOW] = 0xF; // attribute 0, four floats
        registers[REG_ATTRIBUTE_LOADER] = 0x0800_0100;
        registers[REG_ATTRIBUTE_LOADER + 2] = (1 << 28) | (16 << 16);
        let loaders = read_loaders(&registers);

        let input = fetch_vertex(&registers, &mut memory, 0x1800_0000, &loaders, &[shader::ZERO; 16], 0);
        assert_eq!(input[0], [1.5, -2.0, 0.25, 1.0]);
    }

    const COLOR: u32 = 0x1000;
    const DEPTH: u32 = 0x3000;

    /// the inverse of decode_float24, for normal values.
    fn float24(value: f32) -> u32 {
        if value == 0.0 {
            return 0;
        }
        let bits = value.to_bits();
        let exponent = ((bits >> 23) & 0xFF) - 127 + 63;
        ((bits >> 31) << 23) | (exponent << 16) | ((bits >> 7) & 0xFFFF)
    }

    /// an 8x8 RGBA8 target with color writes on and a D24S8 buffer beside it.
    fn target_registers() -> Vec<u32> {
        let mut registers = vec![0u32; 0x300];
        registers[REG_VIEWPORT_WIDTH] = float24(4.0);
        registers[REG_VIEWPORT_HEIGHT] = float24(4.0);
        registers[REG_COLOR_BUFFER_ADDRESS] = COLOR >> 3;
        registers[REG_DEPTH_BUFFER_ADDRESS] = DEPTH >> 3;
        registers[REG_DEPTH_BUFFER_FORMAT] = 3;
        registers[REG_FRAMEBUFFER_DIMENSIONS] = 8 | (7 << 12);
        registers[REG_COLOR_BUFFER_WRITE] = 0xF;
        registers[REG_DEPTH_STENCIL_WRITE] = 0x3;
        registers[REG_DEPTH_COLOR_MASK] = 0xF << 8;
        registers
    }

    /// a triangle covering the whole target at clip-space depth z.
    fn cover(z: f32, color: Vec4) -> Vec<Vertex> {
        [[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]]
            .into_iter()
            .map(|[x, y]| Vertex { clip: [x, y, z, 1.0], color, texcoords: [[0.0; 2]; 3] })
            .collect()
    }

    fn pixels(memory: &mut ConsoleMemory) -> Vec<[u8; 4]> {
        (0..64)
            .map(|i| {
                let mut raw = [0u8; 4];
                memory.read(COLOR + i * 4, &mut raw);
                ColorFormat::Rgba8.decode(&raw)
            })
            .collect()
    }

    const RED: Vec4 = [1.0, 0.0, 0.0, 1.0];
    const GREEN: Vec4 = [0.0, 1.0, 0.0, 1.0];
    const BLUE: Vec4 = [0.0, 0.0, 1.0, 1.0];

    /// titles map depth with a scale of -1, so near is 1, and keep the
    /// nearer surface with a GREATER test against a buffer cleared to 0.
    #[test]
    fn reversed_depth_keeps_the_nearer_surface() {
        let mut registers = target_registers();
        registers[REG_VIEWPORT_DEPTH_RANGE] = float24(-1.0);
        registers[REG_DEPTHMAP_ENABLE] = 1;
        // test enabled, GREATER, with writes.
        registers[REG_DEPTH_COLOR_MASK] |= 1 | (6 << 4) | (1 << 12);

        let mut memory = ConsoleMemory::default();
        rasterize(&registers, &mut memory, &cover(-0.2, RED));
        rasterize(&registers, &mut memory, &cover(-0.8, GREEN));
        rasterize(&registers, &mut memory, &cover(-0.3, BLUE));
        assert!(pixels(&mut memory).iter().all(|&p| p == [0, 255, 0, 255]));
    }

    #[test]
    fn the_stencil_test_masks_pixels() {
        let mut registers = target_registers();
        // stencil, enabled, EQUAL, write mask 0xFF, reference 1, compare 0xFF.
        registers[REG_STENCIL_TEST] = 1 | (2 << 4) | (0xFF << 8) | (1 << 16) | (0xFF << 24);
        let mut memory = ConsoleMemory::default();
        // half the samples hold a stencil of 1.
        for sample in 0..32 {
            memory.write(DEPTH + sample * 4 + 3, &[1]);
        }
        let written = rasterize(&registers, &mut memory, &cover(-0.5, RED));
        assert_eq!(written, 32);
    }

    /// a floor-like triangle with its far corner behind the camera, the part in
    /// front of the camera covers everything above its near edge.
    #[test]
    fn a_triangle_crossing_the_camera_is_clipped_not_dropped() {
        let registers = target_registers();
        let mut memory = ConsoleMemory::default();
        let vertex = |clip: Vec4| Vertex { clip, color: RED, texcoords: [[0.0; 2]; 3] };
        let triangle = [
            vertex([-1.0, -1.0, -0.5, 1.0]),
            vertex([1.0, -1.0, -0.5, 1.0]),
            vertex([0.0, 2.0, 0.5, -1.0]),
        ];
        let written = rasterize(&registers, &mut memory, &triangle);
        assert_eq!(written, 64);
    }

    #[test]
    fn the_color_mask_keeps_channels_it_does_not_write() {
        let mut registers = target_registers();
        registers[REG_DEPTH_COLOR_MASK] = 1 << 8; // red only
        let mut memory = ConsoleMemory::default();
        for i in 0..64 {
            let mut raw = [0u8; 4];
            ColorFormat::Rgba8.encode([10, 20, 30, 40], &mut raw);
            memory.write(COLOR + i * 4, &raw);
        }
        rasterize(&registers, &mut memory, &cover(-0.5, [1.0, 1.0, 1.0, 1.0]));
        assert!(pixels(&mut memory).iter().all(|&p| p == [255, 20, 30, 40]));
    }

    /// the two triangles of a quad cover each pixel exactly once, including
    /// along the diagonal they share.
    #[test]
    fn a_quad_covers_every_pixel_once() {
        let registers = target_registers();
        let mut memory = ConsoleMemory::default();
        let vertex = |x: f32, y: f32| Vertex { clip: [x, y, -0.5, 1.0], color: RED, texcoords: [[0.0; 2]; 3] };
        let quad = [
            vertex(-1.0, -1.0),
            vertex(1.0, -1.0),
            vertex(1.0, 1.0),
            vertex(-1.0, -1.0),
            vertex(1.0, 1.0),
            vertex(-1.0, 1.0),
        ];
        assert_eq!(rasterize(&registers, &mut memory, &quad), 64);
    }

    /// culling keeps the winding the register asks for and drops the other.
    #[test]
    fn face_culling_keeps_one_winding() {
        let vertex = |x: f32, y: f32| Vertex { clip: [x, y, -0.5, 1.0], color: RED, texcoords: [[0.0; 2]; 3] };
        // counter-clockwise with y up.
        let triangle = [vertex(-1.0, -1.0), vertex(3.0, -1.0), vertex(-1.0, 3.0)];
        for (mode, expected) in [(0, 64), (1, 0), (2, 64)] {
            let mut registers = target_registers();
            registers[REG_FACE_CULLING] = mode;
            let mut memory = ConsoleMemory::default();
            assert_eq!(rasterize(&registers, &mut memory, &triangle), expected, "mode {mode}");
        }
    }

    /// window y counts up from the bottom of the buffer, whose rows are
    /// stored top first, a viewport over the upper half of an 8x8 buffer
    /// fills its first four rows in memory and leaves the rest alone.
    #[test]
    fn a_viewport_offset_places_the_image_from_the_bottom() {
        let mut registers = target_registers();
        registers[REG_VIEWPORT_HEIGHT] = float24(2.0);
        registers[REG_VIEWPORT_XY] = 4 << 16;
        let mut memory = ConsoleMemory::default();
        assert_eq!(rasterize(&registers, &mut memory, &cover(-0.5, RED)), 32);
        for row in 0..8 {
            for x in 0..8 {
                let index = crate::format::morton_offset(x, row, 8, 1);
                let mut raw = [0u8; 4];
                memory.read(COLOR + index * 4, &mut raw);
                let red = ColorFormat::Rgba8.decode(&raw) == [255, 0, 0, 255];
                assert_eq!(red, row < 4, "row {row}, column {x}");
            }
        }
    }
}
