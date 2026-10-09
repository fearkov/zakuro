//! software rasterization of PICA200 draw calls.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use rayon::prelude::*;

use crate::registers::*;
use crate::lighting::{Lighting, Tables};

#[cfg(feature = "vulkan")]
pub(crate) mod hardware;
use crate::texture::TextureFormat;
use crate::shader::{self, ShaderUnit, Vec4};
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

/// one component of an attribute, from its bytes.
fn component(bytes: &[u8], ty: u32) -> f32 {
    match ty {
        0 => bytes[0] as i8 as f32,
        1 => bytes[0] as f32,
        2 => i16::from_le_bytes([bytes[0], bytes[1]]) as f32,
        _ => f32::from_bits(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])),
    }
}

/// where an attribute sits in a loader's vertex.
struct Field {
    id: usize,
    offset: u32,
    ty: u32,
    count: u32,
}

/// one attribute array, as a vertex of it is laid out.
struct LoaderLayout {
    /// the loader, 0 to 11.
    index: usize,
    offset: u32,
    stride: u32,
    /// the bytes of a vertex the fields take.
    size: u32,
    fields: Vec<Field>,
}

/// how the attribute arrays lay out a vertex, worked out once for a draw
/// rather than for every vertex.
struct VertexLayout {
    loaders: Vec<LoaderLayout>,
}

impl VertexLayout {
    fn read(registers: &[u32]) -> VertexLayout {
        let loaders = read_loaders(registers)
            .iter()
            .enumerate()
            .filter(|(_, loader)| loader.component_count > 0)
            .map(|(index, loader)| {
                let mut offset = 0u32;
                let mut fields = Vec::new();
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
                    fields.push(Field { id, offset, ty, count });
                    offset += size * count;
                }
                LoaderLayout { index, offset: loader.offset, stride: loader.stride, size: offset, fields }
            })
            .collect();
        VertexLayout { loaders }
    }
}

/// a draw's vertices as the shader takes them, worked out once for the
/// draw: what each input register holds when no array feeds it, and the
/// fields the arrays do give, each going straight to its register.
struct InputPlan {
    template: [Vec4; shader::INPUT_REGISTERS],
    /// for each input register, the fixed attribute it takes, if one does.
    fixed: [Option<usize>; shader::INPUT_REGISTERS],
    loaders: Vec<LoaderPlan>,
}

/// the fields one array gives, and where a vertex of it is.
struct LoaderPlan {
    /// the loader, 0 to 11, whose register holds the array's offset.
    index: usize,
    offset: u32,
    stride: u32,
    size: u32,
    fields: Vec<PlacedField>,
}

/// a field of an array's vertex, and the input register it lands in.
struct PlacedField {
    offset: u32,
    ty: u32,
    count: u32,
    register: usize,
}

impl InputPlan {
    /// what fetch_vertex works out for every vertex, the same way: an
    /// attribute an array gives is read, one flagged fixed takes the value
    /// the command list set, the rest read (0, 0, 0, 1), and the unit's
    /// permutation places them, a later attribute over an earlier one in
    /// the same register.
    fn new(registers: &[u32], layout: &VertexLayout, fixed: &[Vec4; 16]) -> InputPlan {
        // the array field that gives each attribute, the last when several do
        let mut source = [None; 12];
        for (l, loader) in layout.loaders.iter().enumerate() {
            for (f, field) in loader.fields.iter().enumerate() {
                source[field.id] = Some((l, f));
            }
        }
        let count = (((registers[REG_VS_NUM_INPUT_ATTRIBUTES] & 0xF) + 1) as usize).min(12);
        let map = registers[REG_VS_BLOCK + SHADER_INPUT_MAP_LOW] as u64
            | ((registers[REG_VS_BLOCK + SHADER_INPUT_MAP_HIGH] as u64) << 32);
        let mut owner = [None; shader::INPUT_REGISTERS];
        for id in 0..count {
            owner[((map >> (id * 4)) & 0xF) as usize] = Some(id);
        }
        let fixed_mask = (registers[REG_ATTRIBUTE_FORMAT_HIGH] >> 16) & 0xFFF;
        let mut template = [shader::ZERO; shader::INPUT_REGISTERS];
        let mut fixed_registers = [None; shader::INPUT_REGISTERS];
        let mut loaders: Vec<LoaderPlan> = layout
            .loaders
            .iter()
            .map(|loader| LoaderPlan {
                index: loader.index,
                offset: loader.offset,
                stride: loader.stride,
                size: loader.size,
                fields: Vec::new(),
            })
            .collect();
        for (register, id) in owner.iter().enumerate() {
            let Some(id) = *id else { continue };
            match source[id] {
                Some((l, f)) => {
                    let field = &layout.loaders[l].fields[f];
                    loaders[l].fields.push(PlacedField { offset: field.offset, ty: field.ty, count: field.count, register });
                }
                None if fixed_mask & (1 << id) != 0 => fixed_registers[register] = Some(id),
                None => template[register] = [0.0, 0.0, 0.0, 1.0],
            }
        }
        loaders.retain(|loader| !loader.fields.is_empty());
        let mut plan = InputPlan { template, fixed: fixed_registers, loaders };
        plan.refresh(registers, fixed);
        plan
    }

    /// what a draw changes without changing how its vertices are laid
    /// out, where its arrays start and the values of its fixed attributes.
    fn refresh(&mut self, registers: &[u32], fixed: &[Vec4; 16]) {
        for loader in &mut self.loaders {
            loader.offset = registers[REG_ATTRIBUTE_LOADER + loader.index * REG_ATTRIBUTE_LOADER_STRIDE] & 0x0FFF_FFFF;
        }
        for (register, id) in self.fixed.iter().enumerate() {
            if let Some(id) = *id {
                self.template[register] = fixed[id];
            }
        }
    }

    /// one vertex's input registers.
    fn fetch<M: GpuMemory>(&self, memory: &mut M, base: u32, vertex_index: u32) -> [Vec4; shader::INPUT_REGISTERS] {
        let mut input = self.template;
        for loader in &self.loaders {
            // base is physical, and a loader's offset can carry it from one
            // region into another, titles point the base at the start of VRAM
            // and reach vertex data in FCRAM through the offset.
            let vertex = memory.translate(base + loader.offset + vertex_index * loader.stride);
            let size = loader.size as usize;
            match memory.slice(vertex, size) {
                Some(bytes) => loader.place(bytes, &mut input),
                None => {
                    // twelve fields of sixteen bytes, and their alignment, fit
                    let mut bytes = [0u8; 256];
                    memory.read(vertex, &mut bytes[..size]);
                    loader.place(&bytes[..size], &mut input);
                }
            }
        }
        input
    }
}

impl LoaderPlan {
    /// puts the fields of a vertex's bytes in their registers. components
    /// an array does not provide read as (0, 0, 0, 1), a two-component
    /// texture coordinate arrives as (u, v, 0, 1).
    fn place(&self, bytes: &[u8], input: &mut [Vec4; shader::INPUT_REGISTERS]) {
        for field in &self.fields {
            let size = component_size(field.ty) as usize;
            let mut value = [0.0f32, 0.0, 0.0, 1.0];
            for (i, slot) in value.iter_mut().enumerate().take(field.count as usize) {
                let at = field.offset as usize + i * size;
                *slot = component(&bytes[at..at + size], field.ty);
            }
            input[field.register] = value;
        }
    }
}

/// the words a draw's vertex layout comes from: the attribute formats,
/// each loader's components, stride and count, and how the vertex shader
/// takes its inputs. the loaders' offsets are not among them, titles go
/// from one mesh to the next by changing those alone.
const PLAN_KEY: usize = 2 + REG_ATTRIBUTE_LOADER_COUNT * 2 + 3;
/// the vertex layouts kept, more than a busy frame draws with.
const PLANS: usize = 32;

/// the input plans of the vertex layouts draws used last. working one out
/// takes longer than the rest of a small draw.
#[derive(Default)]
struct Plans {
    keys: Vec<[u32; PLAN_KEY]>,
    plans: Vec<InputPlan>,
    /// the plan the last draw used, and the one a new layout replaces next.
    current: usize,
    next: usize,
}

impl Plans {
    /// the plan for the layout the registers set, brought up to the draw.
    fn update(&mut self, registers: &[u32], fixed: &[Vec4; 16]) -> &InputPlan {
        let mut key = [0; PLAN_KEY];
        key[0] = registers[REG_ATTRIBUTE_FORMAT_LOW];
        key[1] = registers[REG_ATTRIBUTE_FORMAT_HIGH];
        for i in 0..REG_ATTRIBUTE_LOADER_COUNT {
            let base = REG_ATTRIBUTE_LOADER + i * REG_ATTRIBUTE_LOADER_STRIDE;
            key[2 + i * 2] = registers[base + 1];
            key[3 + i * 2] = registers[base + 2];
        }
        key[PLAN_KEY - 3] = registers[REG_VS_NUM_INPUT_ATTRIBUTES];
        key[PLAN_KEY - 2] = registers[REG_VS_BLOCK + SHADER_INPUT_MAP_LOW];
        key[PLAN_KEY - 1] = registers[REG_VS_BLOCK + SHADER_INPUT_MAP_HIGH];
        let found = if self.keys.get(self.current) == Some(&key) {
            Some(self.current)
        } else {
            self.keys.iter().position(|kept| *kept == key)
        };
        self.current = match found {
            Some(i) => {
                self.plans[i].refresh(registers, fixed);
                i
            }
            None => {
                let plan = InputPlan::new(registers, &VertexLayout::read(registers), fixed);
                if self.keys.len() < PLANS {
                    self.keys.push(key);
                    self.plans.push(plan);
                    self.keys.len() - 1
                } else {
                    let i = self.next;
                    self.next = (i + 1) % PLANS;
                    self.keys[i] = key;
                    self.plans[i] = plan;
                    i
                }
            }
        };
        &self.plans[self.current]
    }

    /// the plan update last gave.
    fn current(&self) -> &InputPlan {
        &self.plans[self.current]
    }
}

/// fetches one vertex's attributes and lays them out as shader input
#[cfg(test)]
fn fetch_vertex<M: GpuMemory>(
    registers: &[u32],
    memory: &mut M,
    // physical address of the attribute arrays.
    base: u32,
    layout: &VertexLayout,
    fixed: &[Vec4; 16],
    vertex_index: u32,
) -> [Vec4; shader::INPUT_REGISTERS] {
    InputPlan::new(registers, layout, fixed).fetch(memory, base, vertex_index)
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
    /// the surface's orientation and the vector to the viewer, which
    /// fragment lighting works from.
    quaternion: Vec4,
    view: [f32; 3],
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
            quaternion: std::array::from_fn(|i| mix(self.quaternion[i], other.quaternion[i])),
            view: std::array::from_fn(|i| mix(self.view[i], other.view[i])),
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
    quaternion: [Option<(usize, usize)>; 4],
    view: [Option<(usize, usize)>; 3],
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
            quaternion: [None; 4],
            view: [None; 3],
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
        quaternion: [None; 4],
        view: [None; 3],
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
                4..=7 => map.quaternion[semantic - 4] = Some(slot),
                8..=11 => map.color[semantic - 8] = Some(slot),
                12..=13 => map.texcoords[0][semantic - 12] = Some(slot),
                14..=15 => map.texcoords[1][semantic - 14] = Some(slot),
                18..=20 => map.view[semantic - 18] = Some(slot),
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
        quaternion: map.quaternion.map(|slot| slot.map_or(0.0, get)),
        view: map.view.map(|slot| slot.map_or(0.0, get)),
    }
}

/// takes shader inputs through the vertex shader, over threads when there
/// are enough of them, and the geometry shader when one is enabled,
/// producing the vertices the rasterizer assembles. order says which input
/// each vertex comes from when an index buffer repeats them, and comes back
/// to say which of the vertices made each one is, none for the geometry
/// shader's, which come in order.
fn process_vertices<'a>(
    registers: &[u32],
    vertex_shader: &ShaderUnit,
    geometry_shader: &ShaderUnit,
    inputs: &[[Vec4; shader::INPUT_REGISTERS]],
    order: Option<&'a [usize]>,
) -> (Vec<Vertex>, Option<&'a [usize]>) {
    if log::log_enabled!(log::Level::Trace) {
        if let Some(input) = inputs.first() {
            let used_uniforms = vertex_shader.float_uniforms.iter().filter(|u| **u != shader::ZERO).count();
            log::trace!(
                "vertex shader: entry {}, {used_uniforms} non-zero uniforms, output mask 0x{:X}, \
                 first vertex inputs {:?}",
                vertex_shader.entry_point,
                registers[REG_VS_OUTPUT_MASK],
                &input[..8],
            );
        }
    }
    let map = read_output_map(registers);
    if registers[REG_GEOSTAGE_CONFIG] & 0x3 == 2 {
        // the results in draw order, which repeats vertices an index buffer
        // names more than once
        let shaded = shade(registers, vertex_shader, inputs);
        let outputs: Box<dyn Iterator<Item = [Vec4; 16]>> = match order {
            Some(order) => Box::new(order.iter().map(|&i| shaded[i])),
            None => Box::new(shaded.iter().copied()),
        };
        return (geometry_stage(registers, geometry_shader, &map, outputs), None);
    }
    // each vertex made once, by the threads that shade it
    let mask = registers[REG_VS_OUTPUT_MASK];
    let vertices = shade_with(vertex_shader, inputs, |outputs| to_vertex(&map, &pack_outputs(&outputs, mask)));
    (vertices, order)
}

/// a draw's vertices, as shader inputs or already through the shaders.
#[derive(Clone, Copy)]
enum Vertices<'a> {
    Unshaded {
        vertex_shader: &'a ShaderUnit,
        geometry_shader: &'a ShaderUnit,
        inputs: &'a [[Vec4; shader::INPUT_REGISTERS]],
        /// which input each vertex is, when an index buffer repeats them.
        order: Option<&'a [usize]>,
    },
    /// vertex arrays for the GPU to decode and shade.
    #[cfg(feature = "vulkan")]
    Raw {
        vertex_shader: &'a ShaderUnit,
        raw: &'a hardware::RawInputs,
        /// each vertex of the draw, counted from the first of the arrays.
        vertices: &'a [u32],
    },
    Shaded(&'a [Vertex]),
}

impl Vertices<'_> {
    fn len(&self) -> usize {
        match self {
            Vertices::Unshaded { inputs, order, .. } => order.map_or(inputs.len(), <[usize]>::len),
            #[cfg(feature = "vulkan")]
            Vertices::Raw { vertices, .. } => vertices.len(),
            Vertices::Shaded(shaded) => shaded.len(),
        }
    }
}

/// where the GPU's vertex shader finds each varying, per semantic
/// component of position, color, texture coordinates 0, 1 and 2, the
/// quaternion and the view vector, the output register times four plus
/// the component, the way to_vertex reads them.
#[cfg(feature = "vulkan")]
fn output_semantics(registers: &[u32]) -> [u32; 24] {
    let map = read_output_map(registers);
    let mask = registers[REG_VS_OUTPUT_MASK] & 0xFFFF;
    // the output register behind each attribute, as pack_outputs packs them
    let mut enabled = [0; shader::OUTPUT_REGISTERS];
    let mut count = 0;
    for register in (0..shader::OUTPUT_REGISTERS).filter(|r| mask == 0 || mask & (1 << r) != 0) {
        enabled[count] = register;
        count += 1;
    }
    let enabled = &enabled[..count];
    let slot = |slot: Option<(usize, usize)>| match slot {
        None => hardware::MISSING,
        Some((attribute, component)) => {
            enabled.get(attribute).map_or(hardware::ZERO, |&register| (register * 4 + component) as u32)
        }
    };
    let [t0, t1, t2] = map.texcoords;
    let slots = map
        .position
        .map(Some)
        .into_iter()
        .chain(map.color)
        .chain(t0)
        .chain(t1)
        .chain(t2)
        .chain(map.quaternion)
        .chain(map.view);
    let mut semantics = [hardware::MISSING; 24];
    for (semantic, found) in semantics.iter_mut().zip(slots) {
        *semantic = slot(found);
    }
    semantics
}

/// the semantics output_semantics worked out for the last draw, and the
/// registers they came from, the output map and mask, which draws mostly
/// keep.
#[cfg(feature = "vulkan")]
#[derive(Default)]
struct LastSemantics {
    read: [u32; 9],
    semantics: Option<[u32; 24]>,
}

#[cfg(feature = "vulkan")]
impl LastSemantics {
    fn get(&mut self, registers: &[u32]) -> [u32; 24] {
        let mut read = [0; 9];
        read[0] = registers[REG_SHADER_OUTPUT_TOTAL];
        read[1..8].copy_from_slice(&registers[REG_SHADER_OUTPUT_MAP..=REG_SHADER_OUTPUT_MAP_END]);
        read[8] = registers[REG_VS_OUTPUT_MASK];
        match self.semantics {
            Some(semantics) if self.read == read => semantics,
            _ => {
                let semantics = output_semantics(registers);
                *self = LastSemantics { read, semantics: Some(semantics) };
                semantics
            }
        }
    }
}

/// vertices a draw needs before shading them is worth splitting over
/// threads, fewer cost more in handing them out than the threads save, on
/// a laptop's four cores most of all.
const PARALLEL_VERTICES: usize = 512;
/// the fewest vertices each thread takes at a time, whole batches of the
/// shader's.
const PARALLEL_CHUNK: usize = 64;

fn shade(registers: &[u32], unit: &ShaderUnit, inputs: &[[Vec4; shader::INPUT_REGISTERS]]) -> Vec<[Vec4; 16]> {
    let mask = registers[REG_VS_OUTPUT_MASK];
    shade_with(unit, inputs, |outputs| pack_outputs(&outputs, mask))
}

/// runs the vertex shader over inputs, over threads when there are enough
/// of them, and what to do with each one's outputs on the thread that made
/// them, in the inputs' order.
fn shade_with<T: Send>(
    unit: &ShaderUnit,
    inputs: &[[Vec4; shader::INPUT_REGISTERS]],
    then: impl Fn([Vec4; shader::OUTPUT_REGISTERS]) -> T + Sync,
) -> Vec<T> {
    if inputs.len() < PARALLEL_VERTICES {
        return shader::run_vertices(unit, inputs).into_iter().map(then).collect();
    }
    // a few chunks a thread, handing out many small ones costs more than
    // the threads waiting on the last
    let chunk = (inputs.len() / (rayon::current_num_threads() * 4)).next_multiple_of(8).max(PARALLEL_CHUNK);
    inputs.par_chunks(chunk).flat_map_iter(|chunk| shader::run_vertices(unit, chunk).into_iter().map(&then)).collect()
}

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

    // every invocation's inputs, then all of them through the shader in
    // batches
    let mut pending: Vec<Vec4> = Vec::with_capacity(16);
    let mut invocations = Vec::new();
    for outputs in vertex_outputs {
        pending.extend_from_slice(&outputs[..per_vertex]);
        if pending.len() < per_invocation {
            continue;
        }
        invocations.push(map_inputs(registers, REG_GS_BLOCK, &pending[..per_invocation]));
        pending.clear();
    }
    // many primitives go over threads, like many vertices do
    let emitted: Vec<Vec<_>> = if invocations.len() < PARALLEL_VERTICES / 4 {
        shader::run_geometry_many(unit, &invocations)
    } else {
        invocations.par_chunks(PARALLEL_CHUNK / 4).flat_map_iter(|chunk| shader::run_geometry_many(unit, chunk)).collect()
    };

    let mut vertices = Vec::new();
    for (input, triangles) in invocations.iter().zip(&emitted) {
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
                triangles.first().map(|t| &t[0][..4]),
                triangles.len(),
            );
        }
        for triangle in triangles {
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
    quaternion_over_w: Vec4,
    view_over_w: [f32; 3],
}

fn to_screen(vertex: &Vertex, viewport: (f32, f32, f32, f32)) -> Option<Screen> {
    let (vx, vy, vw, vh) = viewport;
    let w = vertex.clip[3];
    if w.abs() < 1e-8 {
        return None;
    }
    let inv_w = 1.0 / w;
    let ndc_x = vertex.clip[0] * inv_w;
    let ndc_y = vertex.clip[1] * inv_w;

    // window coordinates, y points up, from the bottom of the buffer, the
    // way the PICA's viewport is defined. it places vertices on a sixteenth
    // of a pixel.
    let snap = |c: f32| (c * 16.0).round() / 16.0;
    Some(Screen {
        x: snap(vx + (ndc_x * 0.5 + 0.5) * vw),
        y: snap(vy + (ndc_y * 0.5 + 0.5) * vh),
        z: vertex.clip[2] * inv_w,
        inv_w,
        color_over_w: vertex.color.map(|c| c * inv_w),
        texcoords_over_w: vertex.texcoords.map(|[u, v]| [u * inv_w, v * inv_w]),
        quaternion_over_w: vertex.quaternion.map(|c| c * inv_w),
        view_over_w: vertex.view.map(|c| c * inv_w),
    })
}

/// signed distances to the planes of the volume the PICA draws, w
/// positive, and -w <= z <= 0, a vertex is inside when all are >= 0.
const CLIP_PLANES: [fn(&Vertex) -> f32; 3] = [|v| v.clip[3] - 1e-5, |v| -v.clip[2], |v| v.clip[2] + v.clip[3]];

/// whether a vertex is inside the volume the PICA draws.
fn inside(vertex: &Vertex) -> bool {
    CLIP_PLANES.iter().all(|plane| plane(vertex) >= 0.0)
}

/// a vertex with z on an end of the range the PICA draws when it misses
/// one only by rounding. the PICA works positions out in 24-bit floats,
/// which land on 0 and -w, singles can miss them by a little, and 2D drawn
/// on the near or far plane, a title's video say, lost triangles to it.
fn held_in_range(mut vertex: Vertex) -> Vertex {
    let [_, _, z, w] = vertex.clip;
    let z_over_w = z / w;
    if z_over_w > 0.0 && z_over_w < 1e-8 {
        vertex.clip[2] = 0.0;
    } else if z_over_w < -1.0 && z_over_w > -1.00001 {
        vertex.clip[2] = -w;
    }
    vertex
}

/// clips a triangle to the volume the PICA draws.
fn clip_triangle(triangle: [Vertex; 3]) -> Vec<Vertex> {
    let triangle = triangle.map(held_in_range);
    let planes = CLIP_PLANES;
    if triangle.iter().all(inside) {
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

/// a texture that is rows of a buffer the GPU drew and still holds, which
/// the GPU copies instead of guest memory getting the buffer back.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct DrawnTexture {
    pub(crate) addr: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) format: ColorFormat,
}

/// texture unit 0's configuration and backing data for one draw call.
struct BoundTexture {
    /// the texture decoded, row by row from the top, none when it is drawn
    /// unless it reaches past the rows drawn.
    texels: Arc<[[u8; 4]]>,
    drawn: Option<DrawnTexture>,
    /// bilinear rather than point sampling when magnifying.
    linear: bool,
    wrap_s: Wrap,
    wrap_t: Wrap,
    width: u32,
    height: u32,
    border: [f32; 4],
    /// the texture pack's picture for it, which the host's GPU draws with
    /// in its place once it has been read.
    replacement: Option<Arc<crate::pack::Material>>,
}

/// what draws keep from one to the next.
#[derive(Default)]
pub struct Resources {
    pub textures: TextureCache,
    pub light_tables: Tables,
    pub proctex_tables: crate::proctex::Tables,
    pub fog_table: crate::fog::Table,
    scratch: Scratch,
    /// the bytes of the last memory fill, the next one's buffer.
    pub(crate) fill: Vec<u8>,
    /// the words of the last command list, the next one's buffer.
    pub(crate) words: Vec<u32>,
    /// the last draw's triangles as the GPU takes them, the next one's
    /// buffer.
    triangles: Vec<u32>,
    /// the last draw's texture environment and lighting, which the next
    /// mostly keeps.
    tex_env: crate::tev::LastTexEnv,
    lighting: crate::lighting::LastLighting,
    /// where the last draw's varyings were in the output registers.
    #[cfg(feature = "vulkan")]
    semantics: LastSemantics,
    /// the host GPU, when draws go to it rather than to the software path.
    #[cfg(feature = "vulkan")]
    pub(crate) hardware: Option<hardware::Hardware>,
}

/// the buffers a draw fills, kept from one draw to the next. allocating
/// them for each draw costs more than filling them, far more on Windows,
/// which hands a large block out fresh from the system every time, its
/// pages faulting in again as they are first written.
#[derive(Default)]
struct Scratch {
    bytes: Vec<u8>,
    indices: Vec<u32>,
    /// for each vertex index, the stamp of the last draw that named it and
    /// its place among that draw's vertices. a draw stamps the entries it
    /// uses instead of clearing a table as long as its largest index.
    stamps: Vec<u32>,
    places: Vec<u32>,
    stamp: u32,
    unique: Vec<u32>,
    order: Vec<usize>,
    inputs: Vec<[Vec4; shader::INPUT_REGISTERS]>,
    /// each vertex of a draw counted from the first one it uses.
    relative: Vec<u32>,
    plans: Plans,
    /// the arrays a draw hands the GPU to decode.
    #[cfg(feature = "vulkan")]
    raw: hardware::RawInputs,
}

impl Scratch {
    /// the vertices the indices name, each once, in the order they first
    /// come, and each index's place among them.
    fn unique(&mut self) {
        let last = self.indices.iter().copied().max().unwrap_or(0) as usize;
        if self.stamps.len() <= last {
            self.stamps.resize(last + 1, 0);
            self.places.resize(last + 1, 0);
        }
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            // the stamps came around, none of the old ones may match
            self.stamps.fill(0);
            self.stamp = 1;
        }
        self.unique.clear();
        self.order.clear();
        for &index in &self.indices {
            let i = index as usize;
            if self.stamps[i] != self.stamp {
                self.stamps[i] = self.stamp;
                self.places[i] = self.unique.len() as u32;
                self.unique.push(index);
            }
            self.order.push(self.places[i] as usize);
        }
    }
}

/// textures decoded to RGBA, kept across draws for as long as the bytes
/// they came from stay the same. decoding a texel for every sample, four
/// of them when filtering, costs far more than looking one up.
#[derive(Default)]
pub struct TextureCache {
    /// by address, format and size.
    entries: HashMap<TextureKey, Decoded>,
    texels: usize,
    /// counts command lists. nothing but drawing changes memory while one
    /// runs, so a texture checked during it is not checked again unless a
    /// draw writes over it.
    list: u64,
    /// where the draws of this list wrote, as address and length.
    written: Vec<(u32, u32)>,
    /// the texture pack whose pictures replace textures, and where a
    /// texture is laid out to hash it as Citra did.
    pack: Option<Arc<crate::pack::Pack>>,
    rows: Vec<u8>,
}

type TextureKey = (u32, TextureFormat, u32, u32);

/// a texture's texels, and the texture pack's picture replacing it.
type Found = (Arc<[[u8; 4]]>, Option<Arc<crate::pack::Material>>);

struct Decoded {
    /// the hash of the bytes it was decoded from, and the bytes, so that a
    /// change decodes again only the tiles it touched.
    hash: u64,
    bytes: Vec<u8>,
    texels: Arc<[[u8; 4]]>,
    /// the list it was last checked in, and how many writes that list had
    /// made by then.
    checked: (u64, usize),
    replacement: Option<Arc<crate::pack::Material>>,
}

/// how many texels the cache holds before it starts over, 256 MiB of them.
const CACHED_TEXELS: usize = 64 * 1024 * 1024;

impl TextureCache {
    /// replaces textures with the pack's pictures from now on, or with none.
    pub fn set_pack(&mut self, pack: Option<Arc<crate::pack::Pack>>) {
        self.pack = pack;
        self.entries.clear();
        self.texels = 0;
    }

    /// a new command list starts, and memory may have changed since the last.
    pub fn begin_list(&mut self) {
        self.list += 1;
        self.written.clear();
    }

    /// a draw writes to size bytes at addr.
    fn wrote(&mut self, addr: u32, size: u32) {
        if self.written.last() != Some(&(addr, size)) {
            self.written.push((addr, size));
        }
    }

    /// the texture, when it was checked during this list and nothing drew
    /// over it since.
    fn checked(&self, key: TextureKey, size: u32) -> Option<Found> {
        let decoded = self.entries.get(&key)?;
        let (list, writes) = decoded.checked;
        let addr = key.0;
        let overwritten = self.written[writes.min(self.written.len())..]
            .iter()
            .any(|&(start, length)| start < addr.saturating_add(size) && addr < start.saturating_add(length));
        (list == self.list && !overwritten).then(|| (decoded.texels.clone(), decoded.replacement.clone()))
    }

    fn decoded(&mut self, key: TextureKey, data: &[u8]) -> Found {
        let (_, format, width, height) = key;
        let hash = fingerprint(data);
        let checked = (self.list, self.written.len());
        if let Some(decoded) = self.entries.get_mut(&key) {
            if decoded.hash != hash && decoded.bytes.len() == data.len() && width % 8 == 0 && height % 8 == 0 {
                // a title changing a few sprites of an atlas rewrites a
                // handful of its 8x8 tiles, those alone are decoded again,
                // into a copy, as the old texels may still be uploading
                let mut copy: Arc<[[u8; 4]]> = Arc::from(&decoded.texels[..]);
                let texels = Arc::get_mut(&mut copy).expect("a copy no one else holds");
                let tile = data.len() / (width * height / 64) as usize;
                let tiles = decoded.bytes.chunks(tile).zip(data.chunks(tile)).enumerate();
                for (index, _) in tiles.filter(|(_, (old, new))| old != new) {
                    let (tile_x, tile_y) = (index as u32 % (width / 8) * 8, index as u32 / (width / 8) * 8);
                    for y in tile_y..tile_y + 8 {
                        for x in tile_x..tile_x + 8 {
                            texels[(y * width + x) as usize] = crate::texture::sample_texel(data, format, width, x, y);
                        }
                    }
                }
                decoded.bytes.copy_from_slice(data);
                decoded.texels = copy;
                decoded.hash = hash;
                decoded.replacement = picture_for(self.pack.as_deref(), data, &decoded.texels, key, &mut self.rows);
            }
            if decoded.hash == hash {
                decoded.checked = checked;
                return (decoded.texels.clone(), decoded.replacement.clone());
            }
        }
        let count = (width * height) as usize;
        if self.texels + count > CACHED_TEXELS {
            self.entries.clear();
            self.texels = 0;
        }
        let texels: Arc<[[u8; 4]]> = (0..height)
            .flat_map(|y| (0..width).map(move |x| (x, y)))
            .map(|(x, y)| crate::texture::sample_texel(data, format, width, x, y))
            .collect();
        let replacement = picture_for(self.pack.as_deref(), data, &texels, key, &mut self.rows);
        let decoded = Decoded { hash, bytes: data.to_vec(), texels: texels.clone(), checked, replacement: replacement.clone() };
        if let Some(old) = self.entries.insert(key, decoded) {
            self.texels -= old.texels.len();
        }
        self.texels += count;
        (texels, replacement)
    }
}

/// the pack's picture for a texture of these bytes, whenever they change.
fn picture_for(
    pack: Option<&crate::pack::Pack>,
    data: &[u8],
    texels: &[[u8; 4]],
    key: TextureKey,
    rows: &mut Vec<u8>,
) -> Option<Arc<crate::pack::Material>> {
    let (_, format, width, height) = key;
    if crate::pack::recording() {
        crate::pack::record(data, texels, format, width, height);
    }
    pack?.find(data, format, width, height, rows)
}

/// a quick hash of a texture's bytes, to notice when they change. four
/// running hashes side by side, which the CPU works on at once.
fn fingerprint(bytes: &[u8]) -> u64 {
    const K: u64 = 0x517C_C1B7_2722_0A95;
    let step = |hash: u64, word: u64| (hash.rotate_left(5) ^ word).wrapping_mul(K);
    let mut lanes = [bytes.len() as u64, 1, 2, 3];
    let (blocks, rest) = bytes.as_chunks::<32>();
    for block in blocks {
        for (lane, word) in lanes.iter_mut().zip(block.as_chunks::<8>().0) {
            *lane = step(*lane, u64::from_le_bytes(*word));
        }
    }
    let hash = lanes.into_iter().fold(0, step);
    rest.iter().fold(hash, |hash, &byte| step(hash, byte as u64))
}

/// how a texture coordinate outside 0..1 is brought back inside.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Wrap {
    ClampToEdge,
    /// outside the texture, the unit's border color.
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
        let outside = |wrap: Wrap, coordinate: i32, size: u32| {
            wrap == Wrap::ClampToBorder && !(0..size as i32).contains(&coordinate)
        };
        if outside(self.wrap_s, x, self.width) || outside(self.wrap_t, y, self.height) {
            return self.border;
        }
        let x = self.wrap_s.apply(x, self.width);
        let y = self.wrap_t.apply(y, self.height);
        self.texels[(y * self.width + x) as usize].map(|c| c as f32 / 255.0)
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
    cache: &mut TextureCache,
    unit: usize,
    drawn: Option<DrawnTexture>,
    past: bool,
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

    let key = (addr, format, width, height);
    let (texels, replacement) = match (drawn, cache.checked(key, size as u32)) {
        (Some(_), _) if !past => (Arc::default(), None),
        (_, Some(found)) => found,
        (_, None) => match memory.slice(addr, size) {
            Some(data) => cache.decoded(key, data),
            None => {
                let mut data = vec![0u8; size];
                memory.read(addr, &mut data);
                cache.decoded(key, &data)
            }
        },
    };

    // filter mode in bit 1 (magnification) and 2 (minification), the wrap
    // modes for T and S in bits 8-10 and 12-14.
    let config = registers[base + 2];
    Some(BoundTexture {
        texels,
        drawn,
        linear: config & 0x2 != 0,
        wrap_t: Wrap::from_raw(config >> 8),
        wrap_s: Wrap::from_raw(config >> 12),
        width,
        height,
        // the border color register comes first in each unit's block, RGBA8
        border: registers[base].to_le_bytes().map(|c| c as f32 / 255.0),
        // a buffer drawn into is the GPU's, not a pack's picture
        replacement: replacement.filter(|_| drawn.is_none()),
    })
}

/// a texture unit's texture as rows of a color buffer, when it is on and in
/// a format a color buffer has.
#[cfg(feature = "vulkan")]
fn drawn_texture<M: GpuMemory>(registers: &[u32], memory: &M, unit: usize) -> Option<DrawnTexture> {
    use crate::texture::TextureFormat;
    if registers[REG_TEXTURE_CONFIG] & (1 << unit) == 0 {
        return None;
    }
    let base = TEXTURE_UNIT_BASES[unit];
    let dimensions = registers[base + 1];
    let (height, width) = (dimensions & 0x7FF, (dimensions >> 16) & 0x7FF);
    let format_register = if unit == 0 { base + 13 } else { base + 5 };
    let format = match TextureFormat::from_raw(registers[format_register]) {
        TextureFormat::Rgba8 => ColorFormat::Rgba8,
        TextureFormat::Rgb8 => ColorFormat::Rgb8,
        TextureFormat::Rgba5551 => ColorFormat::Rgb5A1,
        TextureFormat::Rgb565 => ColorFormat::Rgb565,
        TextureFormat::Rgba4 => ColorFormat::Rgba4,
        _ => return None,
    };
    let address = loc_register(registers, base + 4);
    (width != 0 && height != 0 && address != 0).then(|| DrawnTexture { addr: memory.translate(address), width, height, format })
}

/// the guest memory a texture unit reads, when it is on.
#[cfg(feature = "vulkan")]
fn texture_range<M: GpuMemory>(registers: &[u32], memory: &M, unit: usize) -> Option<(u32, u32)> {
    if registers[REG_TEXTURE_CONFIG] & (1 << unit) == 0 {
        return None;
    }
    let base = TEXTURE_UNIT_BASES[unit];
    let dimensions = registers[base + 1];
    let (height, width) = (dimensions & 0x7FF, (dimensions >> 16) & 0x7FF);
    let format_register = if unit == 0 { base + 13 } else { base + 5 };
    let format = crate::texture::TextureFormat::from_raw(registers[format_register]);
    let address = loc_register(registers, base + 4);
    if width == 0 || height == 0 || address == 0 {
        return None;
    }
    let bits = width as u64 * height as u64 * format.bits_per_pixel() as u64;
    Some((memory.translate(address), bits.div_ceil(8) as u32))
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
    fn read_sample(&self, surface: &mut Band, index: u32) -> (f32, u8) {
        let mut raw = [0u8; 4];
        raw[..self.bytes as usize].copy_from_slice(surface.at(index));
        let value = u32::from_le_bytes(raw);
        match self.bytes {
            2 => (value as f32 / 65535.0, 0),
            _ => ((value & 0x00FF_FFFF) as f32 / 16_777_215.0, (value >> 24) as u8),
        }
    }

    fn write_depth(&self, surface: &mut Band, index: u32, depth: f32) {
        let sample = surface.at(index);
        if self.bytes == 2 {
            sample.copy_from_slice(&((depth * 65535.0) as u16).to_le_bytes());
        } else {
            // only the low three bytes, the stencil shares the word.
            sample[..3].copy_from_slice(&((depth * 16_777_215.0) as u32).to_le_bytes()[..3]);
        }
    }

    fn write_stencil(&self, surface: &mut Band, index: u32, value: u8) {
        if self.write_stencil && self.bytes == 4 {
            surface.at(index)[3] = value;
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
/// a copy of the rows of a buffer a draw can reach, so that its pixels do
/// not each go through guest memory. it goes back when the draw is done.
struct Surface {
    base: u32,
    bytes: u32,
    /// where the copy starts inside the buffer, in bytes.
    first: u32,
    data: Vec<u8>,
}

impl Surface {
    /// copies buffer rows rows of a tiled buffer, rounded out to whole
    /// rows of tiles, which is what keeps the copy one piece of memory.
    fn load<M: GpuMemory>(memory: &mut M, base: u32, bytes: u32, width: u32, height: u32, rows: Range<u32>) -> Surface {
        let tile_row = width / 8 * 64 * bytes;
        let total = height.div_ceil(8) * tile_row;
        // a width that is not a whole number of tiles spills into the
        // next row of tiles.
        let spill = if width.is_multiple_of(8) { 0 } else { 64 * bytes };
        let first = rows.start / 8 * tile_row;
        let end = (rows.end.div_ceil(8) * tile_row + spill).min(total).max(first);
        let mut data = vec![0u8; (end - first) as usize];
        memory.read(base + first, &mut data);
        Surface { base, bytes, first, data }
    }

    fn store<M: GpuMemory>(&self, memory: &mut M) {
        memory.write(self.base + self.first, &self.data);
    }

    fn whole(&mut self) -> Band<'_> {
        Band { bytes: self.bytes, first: self.first, data: &mut self.data }
    }

    /// the copy cut into one band per row of tiles, top of the buffer first.
    fn tile_rows(&mut self, width: u32) -> Vec<Band<'_>> {
        let tile_row = (width / 8 * 64 * self.bytes) as usize;
        let (bytes, first) = (self.bytes, self.first);
        self.data
            .chunks_mut(tile_row)
            .enumerate()
            .map(|(i, data)| Band { bytes, first: first + (i * tile_row) as u32, data })
            .collect()
    }
}

/// a stretch of a surface's rows, the part one thread fills.
struct Band<'a> {
    bytes: u32,
    /// where the stretch starts inside the buffer, in bytes.
    first: u32,
    data: &'a mut [u8],
}

impl Band<'_> {
    /// the bytes of the pixel at a tiled index.
    fn at(&mut self, index: u32) -> &mut [u8] {
        let offset = (index * self.bytes - self.first) as usize;
        &mut self.data[offset..offset + self.bytes as usize]
    }
}

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

impl FillStats {
    fn add(&mut self, other: FillStats) {
        self.written += other.written;
        self.depth_failed += other.depth_failed;
        self.stencil_failed += other.stencil_failed;
        self.alpha_failed += other.alpha_failed;
    }
}

/// everything about a draw that stays the same from triangle to triangle.
struct DrawState<'a> {
    target: ColorTarget,
    depth_map: DepthMap,
    depth_stencil: Option<DepthStencil>,
    tex_env: crate::tev::TexEnv,
    alpha_test: Option<AlphaTest>,
    blend: Option<crate::blend::Blend>,
    logic_op: Option<crate::blend::LogicOp>,
    textures: &'a [Option<BoundTexture>; 3],
    /// texture unit 2 can read coordinate set 1 instead of its own.
    texture2_uses_coord1: bool,
    lighting: Option<Lighting>,
    tables: &'a Tables,
    /// texture 3, made up from its coordinates, and its tables.
    proctex: Option<crate::proctex::ProcTex<'a>>,
    proctex_tables: &'a crate::proctex::Tables,
    fog: Option<crate::fog::Fog<'a>>,
    fog_table: &'a crate::fog::Table,
}

#[cfg(feature = "vulkan")]
impl DrawState<'_> {
    /// the draw as the GPU takes it.
    fn hardware<'a>(&'a self, registers: &'a [u32], geometry: hardware::Geometry<'a>) -> hardware::Draw<'a> {
        let target = &self.target;
        hardware::Draw {
            registers,
            target: target.addr,
            format: target.format,
            width: target.buffer_width,
            height: target.buffer_height,
            scissor: [target.left, target.bottom, target.right, target.top],
            depth: self.depth_stencil.as_ref().map(|d| (d.addr, d.bytes)),
            depth_map: self.depth_map,
            geometry,
            textures: self.textures,
            lighting: self.lighting.as_ref(),
            tables: self.tables,
            proctex: self.proctex.is_some(),
            proctex_tables: self.proctex_tables,
            fog: self.fog.is_some(),
            fog_table: self.fog_table,
        }
    }
}

/// rasterizes one triangle, texturing and the combiners, then the alpha,
/// stencil and depth tests in the order the hardware runs them, then
/// blending and the color write.
fn fill_triangle(
    color_surface: Option<&mut Band>,
    mut depth_surface: Option<&mut Band>,
    rows: Range<i32>,
    [a, b, c]: [Screen; 3],
    state: &DrawState,
    stats: &mut FillStats,
) {
    let target = &state.target;
    let min_x = (a.x.min(b.x).min(c.x).floor() as i32).max(target.left);
    let max_x = (a.x.max(b.x).max(c.x).ceil() as i32).min(target.right);
    let min_y = (a.y.min(b.y).min(c.y).floor() as i32).max(target.bottom).max(rows.start);
    let max_y = (a.y.max(b.y).max(c.y).ceil() as i32).min(target.top).min(rows.end);
    if min_x >= max_x || min_y >= max_y {
        return;
    }

    // q and -q are the same orientation, but halfway between them is not,
    // so the quaternions all go to the side of the first one.
    let flip = |mut vertex: Screen| {
        let q = vertex.quaternion_over_w;
        if (0..4).map(|i| q[i] * a.quaternion_over_w[i]).sum::<f32>() < 0.0 {
            vertex.quaternion_over_w = q.map(|c| -c);
        }
        vertex
    };
    let (b, c) = (flip(b), flip(c));

    // wind every triangle the same way, so that "inside" is the positive
    // side of all three edges.
    let mut area = Edge::new(&a, &b).at(fixed(c.x), fixed(c.y));
    let (b, c) = if area < 0 {
        area = -area;
        (c, b)
    } else {
        (b, c)
    };
    // zero area means the triangle is degenerate.
    if area == 0 {
        return;
    }
    let inverse_area = 1.0 / area as f32;
    let edges = [Edge::new(&b, &c), Edge::new(&c, &a), Edge::new(&a, &b)];

    let bpp = target.format.bytes_per_pixel();
    let mut color_surface = color_surface;
    let writes_color = color_surface.is_some();
    let partial_write = writes_color && !target.write.iter().all(|&w| w);
    let any_texture = state.textures.iter().any(Option::is_some) || state.proctex.is_some();
    let mut pixel = [0u8; 4];

    for y in min_y..max_y {
        for x in min_x..max_x {
            // the pixel's center, in sixteenths
            let (px, py) = (x as i64 * 16 + 8, y as i64 * 16 + 8);

            // barycentric weights via the edge functions.
            let values = edges.map(|edge| edge.at(px, py));
            if !edges.iter().zip(values).all(|(edge, value)| edge.covers(value)) {
                continue;
            }
            let [w0, w1, w2] = values.map(|value| value as f32 * inverse_area);

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
                .zip(depth_surface.as_deref_mut())
                .map(|(buffer, surface)| buffer.read_sample(surface, index));
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
                if let Some(proctex) = &state.proctex {
                    let set = proctex.coordinates;
                    let u = interpolate([a.texcoords_over_w[set][0], b.texcoords_over_w[set][0], c.texcoords_over_w[set][0]]);
                    let v = interpolate([a.texcoords_over_w[set][1], b.texcoords_over_w[set][1], c.texcoords_over_w[set][1]]);
                    samples[3] = proctex.sample(u, v);
                }
            }

            let fragment = state.lighting.as_ref().map(|lighting| {
                let quaternion = std::array::from_fn(|i| {
                    interpolate([a.quaternion_over_w[i], b.quaternion_over_w[i], c.quaternion_over_w[i]])
                });
                let view = std::array::from_fn(|i| interpolate([a.view_over_w[i], b.view_over_w[i], c.view_over_w[i]]));
                lighting.shade(state.tables, quaternion, view, &samples)
            });
            let combined = state.tex_env.apply(color, samples, fragment);
            let mut rgba = combined.map(|c| (c * 255.0) as u8);

            if let Some(test) = state.alpha_test {
                if !test.passes(rgba[3]) {
                    stats.alpha_failed += 1;
                    continue;
                }
            }
            if let Some(fog) = &state.fog {
                fog.apply(&mut rgba, depth);
            }

            if let (Some(buffer), Some((stored_depth, stored_stencil)), Some(surface)) =
                (&state.depth_stencil, stored, depth_surface.as_deref_mut())
            {
                if let Some(stencil) = &buffer.stencil {
                    if !stencil.passes(stored_stencil) {
                        buffer.write_stencil(surface, index, stencil.update(stored_stencil, stencil.fail));
                        stats.stencil_failed += 1;
                        continue;
                    }
                }
                let depth_passes = buffer.test.is_none_or(|test| test.passes(depth, stored_depth));
                if let Some(stencil) = &buffer.stencil {
                    let op = if depth_passes { stencil.pass } else { stencil.depth_fail };
                    buffer.write_stencil(surface, index, stencil.update(stored_stencil, op));
                }
                if !depth_passes {
                    stats.depth_failed += 1;
                    continue;
                }
                if buffer.write_depth {
                    buffer.write_depth(surface, index, depth);
                }
            }

            let Some(surface) = color_surface.as_deref_mut() else { continue };

            // combine with what is already in the buffer, the way the output
            // merger is configured to, and keep the channels the draw may not
            // change.
            if state.blend.is_some() || state.logic_op.is_some() || partial_write {
                pixel[..bpp].copy_from_slice(surface.at(index));
                let existing = target.format.decode(&pixel[..bpp]);
                if let Some(blend) = &state.blend {
                    let blended = blend.apply(
                        rgba.map(|c| c as f32 / 255.0),
                        existing.map(|c| c as f32 / 255.0),
                    );
                    rgba = blended.map(|c| (c * 255.0) as u8);
                } else if let Some(op) = state.logic_op {
                    rgba = std::array::from_fn(|channel| op.apply(rgba[channel], existing[channel]));
                }
                for channel in 0..4 {
                    if !target.write[channel] {
                        rgba[channel] = existing[channel];
                    }
                }
            }

            target.format.encode(rgba, &mut pixel[..bpp]);
            surface.at(index).copy_from_slice(&pixel[..bpp]);
            stats.written += 1;
        }
    }
}

/// a row of tiles, its color and depth and the window rows it holds.
type Piece<'a> = (Option<Band<'a>>, Option<Band<'a>>, Range<i32>);

/// pixels a draw has to cover before it is worth splitting over threads.
const PARALLEL_PIXELS: f32 = 4096.0;

/// fills the triangles on the copies of the buffers, rows being the window
/// rows they can reach. a big draw is cut into rows of tiles that threads
/// take on, which never touch the same pixel.
fn fill(
    color: &mut Option<Surface>,
    depth: &mut Option<Surface>,
    triangles: &[[Screen; 3]],
    rows: Range<i32>,
    state: &DrawState,
    stats: &mut FillStats,
) {
    let target = &state.target;
    let area: f32 = triangles
        .iter()
        .map(|t| {
            let (x, y) = (t.map(|v| v.x), t.map(|v| v.y));
            let span = |c: [f32; 3]| c.iter().copied().fold(f32::MIN, f32::max) - c.iter().copied().fold(f32::MAX, f32::min);
            span(x) * span(y)
        })
        .sum();
    // a width that is not a whole number of tiles spills across rows of
    // tiles, so those stay on one thread.
    if area < PARALLEL_PIXELS || !target.buffer_width.is_multiple_of(8) {
        let mut color_band = color.as_mut().map(Surface::whole);
        let mut depth_band = depth.as_mut().map(Surface::whole);
        for &triangle in triangles {
            fill_triangle(color_band.as_mut(), depth_band.as_mut(), rows.clone(), triangle, state, stats);
        }
        return;
    }

    let width = target.buffer_width;
    let height = target.buffer_height as i32;
    let first_tile_row = color.as_ref().or(depth.as_ref()).map_or(0, |s| s.first / (width / 8 * 64 * s.bytes)) as i32;
    let colors = color.as_mut().map(|surface| surface.tile_rows(width));
    let depths = depth.as_mut().map(|surface| surface.tile_rows(width));
    let count = colors.as_ref().or(depths.as_ref()).map_or(0, Vec::len);
    let mut colors = colors.map(|bands| bands.into_iter().map(Some).collect::<Vec<_>>());
    let mut depths = depths.map(|bands| bands.into_iter().map(Some).collect::<Vec<_>>());
    let pieces: Vec<Piece> = (0..count)
        .map(|i| {
            // tile row t holds buffer rows 8t to 8t+7, window rows counting
            // from the other end
            let t = first_tile_row + i as i32;
            let band_rows = (height - 8 * t - 8).max(rows.start)..(height - 8 * t).min(rows.end);
            let color_band = colors.as_mut().and_then(|bands| bands[i].take());
            let depth_band = depths.as_mut().and_then(|bands| bands[i].take());
            (color_band, depth_band, band_rows)
        })
        .filter(|piece| !piece.2.is_empty())
        .collect();
    let total = pieces
        .into_par_iter()
        .map(|(mut color_band, mut depth_band, band_rows)| {
            let mut stats = FillStats::default();
            for &triangle in triangles {
                fill_triangle(color_band.as_mut(), depth_band.as_mut(), band_rows.clone(), triangle, state, &mut stats);
            }
            stats
        })
        .reduce(FillStats::default, |mut all, stats| {
            all.add(stats);
            all
        });
    stats.add(total);
}

/// a screen coordinate in sixteenths of a pixel, the steps the PICA places
/// vertices on.
fn fixed(c: f32) -> i64 {
    (c * 16.0).round() as i64
}

/// one edge of a triangle, as a function that is zero on the edge and positive
/// on the triangle's side of it, in whole sixteenths so that a pixel exactly on
/// the edge is known to be.
#[derive(Debug, Clone, Copy)]
struct Edge {
    x: i64,
    y: i64,
    dx: i64,
    dy: i64,
    owns_ties: bool,
}

impl Edge {
    fn new(from: &Screen, to: &Screen) -> Edge {
        let (x, y) = (fixed(from.x), fixed(from.y));
        let (dx, dy) = (fixed(to.x) - x, fixed(to.y) - y);
        // a pixel centered on an edge goes to the triangle on its right, or
        // above it when the edge is flat, as on the PICA
        Edge { x, y, dx, dy, owns_ties: dy < 0 || (dy == 0 && dx > 0) }
    }

    /// the function at a point in sixteenths.
    #[inline]
    fn at(&self, px: i64, py: i64) -> i64 {
        self.dx * (py - self.y) - self.dy * (px - self.x)
    }

    #[inline]
    fn covers(&self, value: i64) -> bool {
        value > 0 || (value == 0 && self.owns_ties)
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
    resources: &mut Resources,
    indexed: bool,
) -> u32 {
    let vertex_count = registers[REG_VERTEX_COUNT];
    let first_vertex = registers[REG_VERTEX_OFFSET];
    if vertex_count == 0 || vertex_count > 0x1_0000 {
        return 0;
    }

    let attribute_base = loc_register(registers, REG_ATTRIBUTE_BASE);

    // resolve each of the vertex_count draw indices to an actual vertex
    // array index, sequential for DrawArrays, looked up in the index buffer
    // for DrawElements.
    let index_config = registers[REG_INDEX_ARRAY];
    let index_short = index_config & 0x8000_0000 != 0;
    let index_base = memory.translate(attribute_base + (index_config & 0x0FFF_FFFF));


    // an index buffer names most vertices several times, fetch and shade
    // each of them once.
    let mut scratch = std::mem::take(&mut resources.scratch);
    if indexed {
        // the whole index buffer at once, a read per index costs far more
        let size = if index_short { 2 } else { 1 };
        let len = (vertex_count * size) as usize;
        let bytes = &mut scratch.bytes;
        bytes.resize(len, 0);
        match memory.slice(index_base, len) {
            Some(slice) => bytes.copy_from_slice(slice),
            None => memory.read(index_base, bytes),
        }
    }
    let plan = scratch.plans.update(registers, fixed_attributes);
    log::trace!(
        "array draw: {vertex_count} vertices from 0x{attribute_base:08X}, indexed {indexed}, formats \
         0x{:08X}{:08X}, arrays {:?}",
        registers[REG_ATTRIBUTE_FORMAT_HIGH],
        registers[REG_ATTRIBUTE_FORMAT_LOW],
        plan.loaders.iter().map(|l| (l.offset, l.stride, l.size)).collect::<Vec<_>>(),
    );
    // the GPU decodes the arrays itself when it shades the draw, from the
    // first vertex the draw uses to the last
    #[cfg(feature = "vulkan")]
    if resources.hardware.as_ref().is_some_and(|hardware| hardware.shades()) && registers[REG_GEOSTAGE_CONFIG] & 0x3 != 2 {
        let (first, span) = if indexed {
            // over the index buffer's own bytes or halfwords, which the
            // vector units compare many at a time, they have no unsigned
            // comparison of whole words
            let (low, high) = if index_short {
                let halves = scratch.bytes.as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b));
                let (low, high) = halves.fold((u16::MAX, 0), |(low, high), index| (low.min(index), high.max(index)));
                (low as u32, high as u32)
            } else {
                let (low, high) = scratch.bytes.iter().fold((u8::MAX, 0), |(low, high), &index| (low.min(index), high.max(index)));
                (low as u32, high as u32)
            };
            (low, high - low + 1)
        } else {
            (first_vertex, vertex_count)
        };
        if raw_inputs(plan, memory, attribute_base, first, span, &mut scratch.raw) {
            scratch.relative.clear();
            if indexed {
                // straight from the index buffer, the draw needs its indices
                // as words only when the CPU decodes its arrays
                if index_short {
                    let halves = scratch.bytes.as_chunks::<2>().0.iter();
                    scratch.relative.extend(halves.map(|b| u16::from_le_bytes(*b) as u32 - first));
                } else {
                    scratch.relative.extend(scratch.bytes.iter().map(|&b| b as u32 - first));
                }
            } else {
                scratch.relative.extend(0..vertex_count);
            }
            let vertices = Vertices::Raw { vertex_shader, raw: &scratch.raw, vertices: &scratch.relative };
            if rasterize(registers, memory, resources, vertices).is_some() {
                resources.scratch = scratch;
                return vertex_count;
            }
        }
    }
    if indexed {
        scratch.indices.clear();
        if index_short {
            scratch.indices.extend(scratch.bytes.as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b) as u32));
        } else {
            scratch.indices.extend(scratch.bytes.iter().map(|&b| b as u32));
        }
        scratch.unique();
    } else {
        scratch.unique.clear();
        scratch.unique.extend((0..vertex_count).map(|i| first_vertex + i));
    }
    let plan = scratch.plans.current();
    scratch.inputs.clear();
    scratch.inputs.extend(scratch.unique.iter().map(|&vertex_index| plan.fetch(memory, attribute_base, vertex_index)));
    let order = indexed.then_some(&scratch.order[..]);
    let vertices = Vertices::Unshaded { vertex_shader, geometry_shader, inputs: &scratch.inputs, order };
    rasterize(registers, memory, resources, vertices);
    resources.scratch = scratch;
    vertex_count
}

/// draws vertices a title sent one attribute at a time through the fixed
/// attribute registers ("immediate mode") instead of from arrays in memory.
pub fn draw_immediate<M: GpuMemory>(
    registers: &[u32],
    vertex_shader: &ShaderUnit,
    geometry_shader: &ShaderUnit,
    memory: &mut M,
    resources: &mut Resources,
    vertices: &[[Vec4; 16]],
) -> u32 {
    // immediate mode sizes its vertices by GPUREG_VSH_NUM_ATTR.
    let count = ((registers[REG_VS_ATTRIBUTE_COUNT] & 0xF) + 1) as usize;
    log::trace!("immediate draw: {} vertices of {count} attributes", vertices.len());
    let inputs: Vec<_> = vertices
        .iter()
        .map(|attributes| map_inputs(registers, REG_VS_BLOCK, &attributes[..count]))
        .collect();
    let vertices = Vertices::Unshaded { vertex_shader, geometry_shader, inputs: &inputs, order: None };
    rasterize(registers, memory, resources, vertices).unwrap_or(0)
}

/// puts a draw's arrays for the GPU to decode in raw, from the first
/// vertex the draw uses on. false when they cost more to hand over than to
/// decode on the CPU or do not lie in one piece of memory.
#[cfg(feature = "vulkan")]
fn raw_inputs<M: GpuMemory>(
    plan: &InputPlan,
    memory: &mut M,
    base: u32,
    first: u32,
    span: u32,
    raw: &mut hardware::RawInputs,
) -> bool {
    // past this the vertices between the ones a draw uses cost too much
    const LIMIT: u64 = 4 << 20;
    raw.arrays.clear();
    raw.fields = [None; shader::INPUT_REGISTERS];
    raw.defaults = plan.template;
    let mut total = 0u64;
    for loader in &plan.loaders {
        let start = base.wrapping_add(loader.offset).wrapping_add(first.wrapping_mul(loader.stride));
        let len = (span - 1) * loader.stride + loader.size;
        total += len as u64;
        if total > LIMIT {
            return false;
        }
        // a vertex's address is translated on its own on the CPU, which
        // comes to the same where the array is in one piece
        let addr = memory.translate(start);
        if memory.slice(addr, len as usize).is_none() {
            return false;
        }
        for field in &loader.fields {
            // a field fills its components over (0, 0, 0, 1)
            raw.defaults[field.register] = [0.0, 0.0, 0.0, 1.0];
            raw.fields[field.register] = Some(hardware::RawField {
                array: raw.arrays.len(),
                stride: loader.stride,
                offset: field.offset,
                ty: field.ty,
                count: field.count,
            });
        }
        raw.arrays.push((addr, len));
    }
    true
}

/// the vertices of each triangle out of count of them, as assemble puts
/// them, each the vertex it names, into indices.
fn assemble_into(topology: u32, count: usize, vertex: impl Fn(usize) -> u32, indices: &mut Vec<u32>) {
    indices.clear();
    // a strip or a fan a triangle at a time, a flattened iterator has no
    // length to reserve by and pushes a vertex at a time
    if matches!(topology, 1 | 2) {
        indices.reserve(count.saturating_sub(2) * 3);
    }
    match topology {
        1 => {
            for i in 2..count {
                let (a, b) = if i % 2 == 0 { (i - 2, i - 1) } else { (i - 1, i - 2) };
                indices.extend([vertex(a), vertex(b), vertex(i)]);
            }
        }
        2 => {
            for i in 2..count {
                indices.extend([vertex(0), vertex(i - 1), vertex(i)]);
            }
        }
        _ => indices.extend((0..count / 3 * 3).map(vertex)),
    }
}

/// the triangles of a draw of count vertices as the GPU takes them, three
/// vertices each, numbered as the raw arrays number them, or by their place
/// among the vertices order gives, or by their place in the draw. a list of
/// raw vertices is its own triangles, which go as they are rather than
/// copied.
#[cfg(feature = "vulkan")]
fn triangle_indices<'a>(
    topology: u32,
    count: usize,
    raw: Option<&'a [u32]>,
    order: Option<&[usize]>,
    assembled: &'a mut Vec<u32>,
) -> &'a [u32] {
    match raw {
        Some(vertices) if !matches!(topology, 1 | 2) => &vertices[..count / 3 * 3],
        Some(vertices) => {
            assemble_into(topology, count, |i| vertices[i], assembled);
            assembled
        }
        None => {
            assemble_into(topology, count, |i| order.map_or(i, |order| order[i]) as u32, assembled);
            assembled
        }
    }
}

/// the vertices of each triangle out of count of them, for a list, a strip
/// (1) or a fan (2).
fn assemble(topology: u32, count: usize) -> Vec<(usize, usize, usize)> {
    match topology {
        1 => (2..count).map(|i| if i % 2 == 0 { (i - 2, i - 1, i) } else { (i - 1, i - 2, i) }).collect(),
        2 => (2..count).map(|i| (0, i - 1, i)).collect(),
        _ => (0..count / 3).map(|t| (t * 3, t * 3 + 1, t * 3 + 2)).collect(),
    }
}

/// assembles shaded vertices into triangles the way the primitive configuration
/// says, and fills them with the current back-end state.
/// the triangles drawn, none for arrays the GPU could not decode, which the
/// CPU decodes then.
fn rasterize<M: GpuMemory>(registers: &[u32], memory: &mut M, resources: &mut Resources, vertices: Vertices) -> Option<u32> {
    let vertex_count = vertices.len();
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
        return Some(0);
    }
    let viewport = (viewport_x, viewport_y, viewport_width, viewport_height);

    let color_buffer_raw = loc_register(registers, REG_COLOR_BUFFER_ADDRESS);
    let target_addr = memory.translate(color_buffer_raw);
    // the format sits in bits 16-18 of the register, not the low bits, the low
    // half holds how many bytes a pixel takes.
    let target_format = ColorFormat::from_color_buffer((registers[REG_COLOR_BUFFER_FORMAT] >> 16) & 7);
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
    // a texture can be a buffer the GPU drew into, guest memory has to have
    // it before it is read
    // one the GPU still holds it copies there
    let mut drawn: [Option<DrawnTexture>; 3] = [None; 3];
    // the rest of one reaching past the rows drawn is read as memory holds
    // it, as of what was written back last
    let mut past = [false; 3];
    #[cfg(feature = "vulkan")]
    if let Some(hardware) = resources.hardware.as_mut() {
        for (unit, drawn) in drawn.iter_mut().enumerate() {
            *drawn = drawn_texture(registers, memory, unit).filter(|texture| hardware.holds(texture));
            if let Some(texture) = drawn {
                past[unit] = !hardware.covers(texture);
                continue;
            }
            if let Some((addr, len)) = texture_range(registers, memory, unit) {
                if let Err(error) = hardware.prepare_read(memory, addr, len) {
                    log::error!("the GPU could not write back a buffer, {error}");
                }
            }
        }
    }
    let textures: [Option<BoundTexture>; 3] =
        std::array::from_fn(|unit| bind_texture(registers, memory, &mut resources.textures, unit, drawn[unit], past[unit]));
    let tex_env = resources.tex_env.read(registers);
    // the lighting only matters to a draw whose combiners read it.
    let lighting = if tex_env.reads_lighting() { resources.lighting.read(registers) } else { None };
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
        tex_env,
        alpha_test: AlphaTest::read(registers),
        blend: crate::blend::Blend::read(registers),
        logic_op: crate::blend::LogicOp::read(registers),
        textures: &textures,
        texture2_uses_coord1: registers[REG_TEXTURE_CONFIG] & (1 << 13) != 0,
        lighting,
        tables: &resources.light_tables,
        proctex: crate::proctex::ProcTex::read(registers, &resources.proctex_tables),
        proctex_tables: &resources.proctex_tables,
        fog: crate::fog::Fog::read(registers, &resources.fog_table),
        fog_table: &resources.fog_table,
    };
    // what the draw writes, which the textures of later draws in the list
    // may be
    let pixels = buffer_width * buffer_height;
    resources.textures.wrote(target_addr, pixels * target_format.bytes_per_pixel() as u32);
    if let Some(depth) = &state.depth_stencil {
        resources.textures.wrote(depth.addr, pixels * depth.bytes);
    }

    // GPUREG_PRIMITIVE_CONFIG bits [9:8], 0 = triangle list, 1 = strip,
    // 2 = fan, 3 = whatever the geometry shader emitted, which is a list.
    let topology = (registers[REG_PRIMITIVE_CONFIG] >> 8) & 0x3;

    // GPUREG_FACECULLING_CONFIG, 0 keeps everything, 1 keeps triangles
    // wound clockwise and 2 counter-clockwise, as seen with y pointing up.
    let cull_mode = registers[REG_FACE_CULLING] & 0x3;

    // the GPU runs the vertex shader itself, unless a geometry shader
    // comes after it
    #[cfg(feature = "vulkan")]
    if let Some(hardware) = resources.hardware.as_mut() {
        let target = &state.target;
        let whole = target.buffer_width.is_multiple_of(8) && target.buffer_height.is_multiple_of(8);
        let unshaded = match vertices {
            Vertices::Unshaded { vertex_shader, inputs, order, .. } => {
                Some((vertex_shader, hardware::Inputs::Decoded(inputs), order, None))
            }
            Vertices::Raw { vertex_shader, raw, vertices } => Some((vertex_shader, hardware::Inputs::Raw(raw), None, Some(vertices))),
            Vertices::Shaded(_) => None,
        };
        if let Some((unit, inputs, order, raw)) = unshaded.filter(|_| hardware.shades() && whole && registers[REG_GEOSTAGE_CONFIG] & 0x3 != 2) {
            let mut assembled = std::mem::take(&mut resources.triangles);
            let indices = triangle_indices(topology, vertex_count, raw, order, &mut assembled);
            let shading = hardware::Shading {
                unit,
                inputs,
                indices,
                semantics: resources.semantics.get(registers),
                viewport,
                cull: cull_mode,
            };
            let drawn = hardware.draw(memory, &state.hardware(registers, hardware::Geometry::Shaded(&shading)));
            let triangles = (indices.len() / 3) as u32;
            resources.triangles = assembled;
            match drawn {
                Ok(()) => return Some(triangles),
                Err(error) => log::error!("the GPU could not shade, {error}, shading on the CPU"),
            }
        }
    }
    let processed;
    let (shaded, order) = match vertices {
        Vertices::Shaded(shaded) => (shaded, None),
        Vertices::Unshaded { vertex_shader, geometry_shader, inputs, order } => {
            processed = process_vertices(registers, vertex_shader, geometry_shader, inputs, order);
            (&processed.0[..], processed.1)
        }
        // arrays the GPU did not draw, which the CPU decodes first
        #[cfg(feature = "vulkan")]
        Vertices::Raw { .. } => return None,
    };
    // which of the vertices made each vertex of the draw is
    let at = |i: usize| order.map_or(i, |order| order[i]);
    let triangle_indices = assemble(topology, order.map_or(shaded.len(), <[usize]>::len));

    let triangle_count = triangle_indices.len();
    let mut clipped_away = 0u32;
    let mut culled = 0u32;
    let mut stats = FillStats::default();
    // where each vertex inside the volume lands, worked out once however
    // many triangles share it, and the triangles as indices into those, a
    // triangle of such vertices needs no clipping. what clipping makes goes
    // after them
    const OUTSIDE: u32 = u32::MAX;
    let mut placed: Vec<Screen> = Vec::with_capacity(shaded.len());
    let slots: Vec<u32> = shaded
        .iter()
        .map(|vertex| match inside(vertex).then(|| to_screen(vertex, viewport)).flatten() {
            Some(screen) => {
                placed.push(screen);
                placed.len() as u32 - 1
            }
            None => OUTSIDE,
        })
        .collect();
    let mut indices: Vec<u32> = Vec::with_capacity(triangle_count * 3);
    let culls = |p: &Screen, q: &Screen, r: &Screen| {
        let counter_clockwise = (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x) > 0.0;
        cull_mode != 0 && counter_clockwise == (cull_mode == 1)
    };
    for (ia, ib, ic) in triangle_indices {
        let (a, b, c) = (at(ia), at(ib), at(ic));
        let [sa, sb, sc] = [slots[a], slots[b], slots[c]];
        if sa != OUTSIDE && sb != OUTSIDE && sc != OUTSIDE {
            if culls(&placed[sa as usize], &placed[sb as usize], &placed[sc as usize]) {
                culled += 1;
            } else {
                indices.extend([sa, sb, sc]);
            }
            continue;
        }
        let polygon = clip_triangle([shaded[a], shaded[b], shaded[c]]);
        let screen: Vec<Screen> = polygon.iter().filter_map(|v| to_screen(v, viewport)).collect();
        if screen.len() < 3 || screen.len() != polygon.len() {
            clipped_away += 1;
            continue;
        }
        if culls(&screen[0], &screen[1], &screen[2]) {
            culled += 1;
            continue;
        }
        let first = placed.len() as u32;
        placed.extend_from_slice(&screen);
        for i in 1..screen.len() as u32 - 1 {
            indices.extend([first, first + i, first + i + 1]);
        }
    }

    #[cfg(feature = "vulkan")]
    if let Some(hardware) = resources.hardware.as_mut() {
        let target = &state.target;
        // tiles are whole in any buffer a title really draws into
        if target.buffer_width.is_multiple_of(8) && target.buffer_height.is_multiple_of(8) {
            let geometry = hardware::Geometry::Placed { vertices: &placed, indices: &indices };
            match hardware.draw(memory, &state.hardware(registers, geometry)) {
                Ok(()) => return Some(triangle_count as u32),
                Err(error) => log::error!("the GPU could not draw, {error}, drawing in software"),
            }
        }
        // the software path works on guest memory, which has to hold what
        // the GPU drew
        if let Err(error) = hardware.flush(memory) {
            log::error!("the GPU could not write back its buffers, {error}");
        }
        // and on textures read from it, which the ones the GPU was to copy
        // were not
        if drawn.iter().any(Option::is_some) {
            let hardware = resources.hardware.take();
            let in_order: Vec<Vertex> = (0..order.map_or(shaded.len(), <[usize]>::len)).map(|i| shaded[at(i)]).collect();
            let drawn = rasterize(registers, memory, resources, Vertices::Shaded(&in_order));
            resources.hardware = hardware;
            return drawn;
        }
    }

    let triangles: Vec<[Screen; 3]> =
        indices.as_chunks::<3>().0.iter().map(|&[a, b, c]| [placed[a as usize], placed[b as usize], placed[c as usize]]).collect();
    // the rows the triangles can reach, as window rows and then as rows of
    // the buffer, which counts from the other end.
    let target = &state.target;
    let low = triangles.iter().map(|t| t.iter().map(|v| v.y).fold(f32::MAX, f32::min).floor() as i32).min();
    let high = triangles.iter().map(|t| t.iter().map(|v| v.y).fold(f32::MIN, f32::max).ceil() as i32).max();
    if let (Some(low), Some(high)) = (low, high) {
        let (low, high) = (low.max(target.bottom), high.min(target.top));
        if low < high {
            let rows = (target.buffer_height - high as u32)..(target.buffer_height - low as u32);
            let (width, height) = (target.buffer_width, target.buffer_height);
            let bpp = target.format.bytes_per_pixel() as u32;
            let mut color = target
                .write
                .iter()
                .any(|&w| w)
                .then(|| Surface::load(memory, target.addr, bpp, width, height, rows.clone()));
            let mut depth = state
                .depth_stencil
                .as_ref()
                .map(|buffer| Surface::load(memory, buffer.addr, buffer.bytes, width, height, rows.clone()));
            fill(&mut color, &mut depth, &triangles, low..high, &state, &mut stats);
            if let Some(color) = color.as_ref().filter(|_| stats.written > 0) {
                color.store(memory);
            }
            if let (Some(depth), Some(buffer)) = (&depth, &state.depth_stencil) {
                if buffer.write_depth || buffer.write_stencil {
                    depth.store(memory);
                }
            }
        }
    }

    if log::log_enabled!(log::Level::Trace) {
        let first_screen = shaded.first().and_then(|v| to_screen(v, viewport));
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

    Some(stats.written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// guest memory seen the way the GPU sees it, physical VRAM and FCRAM
    /// translate to different virtual windows. its first megabytes lie in
    /// one piece, as the linear heap and VRAM do, unless it is scattered.
    #[derive(Clone)]
    struct ConsoleMemory {
        flat: Vec<u8>,
        rest: HashMap<u32, u8>,
        /// whether slice hands out the flat bytes.
        whole: bool,
    }

    impl Default for ConsoleMemory {
        fn default() -> ConsoleMemory {
            ConsoleMemory { flat: vec![0; 4 << 20], rest: HashMap::new(), whole: true }
        }
    }

    impl ConsoleMemory {
        /// memory nothing is in one piece of, for the paths that copy.
        fn scattered() -> ConsoleMemory {
            ConsoleMemory { whole: false, ..ConsoleMemory::default() }
        }
    }

    impl GpuMemory for ConsoleMemory {
        fn read(&mut self, addr: u32, out: &mut [u8]) {
            for (i, byte) in out.iter_mut().enumerate() {
                let at = addr + i as u32;
                *byte = self.flat.get(at as usize).copied().unwrap_or_else(|| self.rest.get(&at).copied().unwrap_or(0));
            }
        }

        fn write(&mut self, addr: u32, data: &[u8]) {
            for (i, &byte) in data.iter().enumerate() {
                let at = addr + i as u32;
                match self.flat.get_mut(at as usize) {
                    Some(flat) => *flat = byte,
                    None => {
                        self.rest.insert(at, byte);
                    }
                }
            }
        }

        fn slice(&mut self, addr: u32, len: usize) -> Option<&[u8]> {
            self.slice_mut(addr, len).map(|slice| &*slice)
        }

        fn slice_mut(&mut self, addr: u32, len: usize) -> Option<&mut [u8]> {
            let at = addr as usize;
            if !self.whole || at + len > self.flat.len() {
                return None;
            }
            Some(&mut self.flat[at..at + len])
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
        let layout = VertexLayout::read(&registers);

        let input = fetch_vertex(&registers, &mut memory, 0x1800_0000, &layout, &[shader::ZERO; 16], 0);
        assert_eq!(input[0], [1.5, -2.0, 0.25, 1.0]);
    }

    /// a draw that keeps the last one's layout but moves its array or
    /// changes a fixed attribute reads its own vertices and values.
    #[test]
    fn a_kept_input_plan_follows_the_draw() {
        let mut memory = ConsoleMemory::default();
        for (i, value) in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].iter().enumerate() {
            memory.write(0x1400_0100 + i as u32 * 4, &value.to_le_bytes());
        }
        let mut registers = vec![0u32; 0x300];
        registers[REG_ATTRIBUTE_FORMAT_LOW] = 0xF; // attribute 0, four floats
        registers[REG_ATTRIBUTE_FORMAT_HIGH] = 1 << 17; // attribute 1 fixed
        registers[REG_ATTRIBUTE_LOADER] = 0x100;
        registers[REG_ATTRIBUTE_LOADER + 2] = (1 << 28) | (16 << 16);
        registers[REG_VS_NUM_INPUT_ATTRIBUTES] = 1;
        registers[REG_VS_BLOCK + SHADER_INPUT_MAP_LOW] = 0x10; // attribute 1 in v1
        let mut fixed = [shader::ZERO; 16];
        fixed[1] = [0.5; 4];

        let mut plans = Plans::default();
        let plan = plans.update(&registers, &fixed);
        assert_eq!(plan.fetch(&mut memory, 0x2000_0000, 0)[..2], [[1.0, 2.0, 3.0, 4.0], [0.5; 4]]);

        registers[REG_ATTRIBUTE_LOADER] = 0x110;
        fixed[1] = [0.25; 4];
        let plan = plans.update(&registers, &fixed);
        assert_eq!(plan.fetch(&mut memory, 0x2000_0000, 0)[..2], [[5.0, 6.0, 7.0, 8.0], [0.25; 4]]);
        assert_eq!(plans.keys.len(), 1);

        // two floats a vertex is another layout
        registers[REG_ATTRIBUTE_FORMAT_LOW] = 0x7;
        let plan = plans.update(&registers, &fixed);
        assert_eq!(plan.fetch(&mut memory, 0x2000_0000, 0)[0], [5.0, 6.0, 0.0, 1.0]);
        assert_eq!(plans.keys.len(), 2);
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
    fn rasterize_shaded<M: GpuMemory>(registers: &[u32], memory: &mut M, resources: &mut Resources, shaded: &[Vertex]) -> u32 {
        rasterize(registers, memory, resources, Vertices::Shaded(shaded)).unwrap_or(0)
    }

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
        registers[REG_LOGIC_OP] = 3;
        registers
    }

    /// a triangle covering the whole target at clip-space depth z.
    fn cover(z: f32, color: Vec4) -> Vec<Vertex> {
        [[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]]
            .into_iter()
            .map(|[x, y]| Vertex { clip: [x, y, z, 1.0], color, texcoords: [[0.0; 2]; 3], quaternion: [0.0, 0.0, 0.0, 1.0], view: [0.0; 3] })
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
        rasterize_shaded(&registers, &mut memory, &mut Resources::default(), &cover(-0.2, RED));
        rasterize_shaded(&registers, &mut memory, &mut Resources::default(), &cover(-0.8, GREEN));
        rasterize_shaded(&registers, &mut memory, &mut Resources::default(), &cover(-0.3, BLUE));
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
        let written = rasterize_shaded(&registers, &mut memory, &mut Resources::default(), &cover(-0.5, RED));
        assert_eq!(written, 32);
    }

    /// a floor-like triangle with its far corner behind the camera, the part in
    /// front of the camera covers everything above its near edge.
    #[test]
    fn a_triangle_crossing_the_camera_is_clipped_not_dropped() {
        let registers = target_registers();
        let mut memory = ConsoleMemory::default();
        let vertex = |clip: Vec4| Vertex { clip, color: RED, texcoords: [[0.0; 2]; 3], quaternion: [0.0, 0.0, 0.0, 1.0], view: [0.0; 3] };
        let triangle = [
            vertex([-1.0, -1.0, -0.5, 1.0]),
            vertex([1.0, -1.0, -0.5, 1.0]),
            vertex([0.0, 2.0, 0.5, -1.0]),
        ];
        let written = rasterize_shaded(&registers, &mut memory, &mut Resources::default(), &triangle);
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
        rasterize_shaded(&registers, &mut memory, &mut Resources::default(), &cover(-0.5, [1.0, 1.0, 1.0, 1.0]));
        assert!(pixels(&mut memory).iter().all(|&p| p == [255, 20, 30, 40]));
    }

    /// the two triangles of a quad cover each pixel exactly once, including
    /// along the diagonal they share.
    #[test]
    fn a_quad_covers_every_pixel_once() {
        let registers = target_registers();
        let mut memory = ConsoleMemory::default();
        let vertex = |x: f32, y: f32| Vertex { clip: [x, y, -0.5, 1.0], color: RED, texcoords: [[0.0; 2]; 3], quaternion: [0.0, 0.0, 0.0, 1.0], view: [0.0; 3] };
        let quad = [
            vertex(-1.0, -1.0),
            vertex(1.0, -1.0),
            vertex(1.0, 1.0),
            vertex(-1.0, -1.0),
            vertex(1.0, 1.0),
            vertex(-1.0, 1.0),
        ];
        assert_eq!(rasterize_shaded(&registers, &mut memory, &mut Resources::default(), &quad), 64);
    }

    /// a draw big enough to be split over threads covers the same pixels,
    /// each once, that it would on one.
    #[test]
    fn a_big_draw_split_over_threads_covers_every_pixel_once() {
        const BIG_COLOR: u32 = 0x10_0000;
        const BIG_DEPTH: u32 = 0x20_0000;
        let mut registers = target_registers();
        registers[REG_VIEWPORT_WIDTH] = float24(128.0);
        registers[REG_VIEWPORT_HEIGHT] = float24(128.0);
        registers[REG_COLOR_BUFFER_ADDRESS] = BIG_COLOR >> 3;
        registers[REG_DEPTH_BUFFER_ADDRESS] = BIG_DEPTH >> 3;
        registers[REG_FRAMEBUFFER_DIMENSIONS] = 256 | (255 << 12);
        let mut memory = ConsoleMemory::default();
        let vertex = |x: f32, y: f32| Vertex { clip: [x, y, -0.5, 1.0], color: RED, texcoords: [[0.0; 2]; 3], quaternion: [0.0, 0.0, 0.0, 1.0], view: [0.0; 3] };
        let quad = [
            vertex(-1.0, -1.0),
            vertex(1.0, -1.0),
            vertex(1.0, 1.0),
            vertex(-1.0, -1.0),
            vertex(1.0, 1.0),
            vertex(-1.0, 1.0),
        ];
        assert_eq!(rasterize_shaded(&registers, &mut memory, &mut Resources::default(), &quad), 256 * 256);
        for i in 0..256 * 256 {
            let mut raw = [0u8; 4];
            memory.read(BIG_COLOR + i * 4, &mut raw);
            assert_eq!(ColorFormat::Rgba8.decode(&raw), [255, 0, 0, 255], "pixel {i}");
        }
    }

    /// culling keeps the winding the register asks for and drops the other.
    #[test]
    fn face_culling_keeps_one_winding() {
        let vertex = |x: f32, y: f32| Vertex { clip: [x, y, -0.5, 1.0], color: RED, texcoords: [[0.0; 2]; 3], quaternion: [0.0, 0.0, 0.0, 1.0], view: [0.0; 3] };
        // counter-clockwise with y up.
        let triangle = [vertex(-1.0, -1.0), vertex(3.0, -1.0), vertex(-1.0, 3.0)];
        for (mode, expected) in [(0, 64), (1, 0), (2, 64)] {
            let mut registers = target_registers();
            registers[REG_FACE_CULLING] = mode;
            let mut memory = ConsoleMemory::default();
            assert_eq!(rasterize_shaded(&registers, &mut memory, &mut Resources::default(), &triangle), expected, "mode {mode}");
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
        assert_eq!(rasterize_shaded(&registers, &mut memory, &mut Resources::default(), &cover(-0.5, RED)), 32);
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

    #[test]
    fn clamp_to_border_reads_the_border_color() {
        let texture = |wrap: Wrap| BoundTexture {
            drawn: None,
            texels: vec![[0xFF; 4]; 8 * 8].into(),
            linear: false,
            wrap_s: wrap,
            wrap_t: wrap,
            width: 8,
            height: 8,
            border: RED,
            replacement: None,
        };
        let white = [1.0; 4];
        assert_eq!(texture(Wrap::ClampToBorder).texel(-1, 3), RED);
        assert_eq!(texture(Wrap::ClampToBorder).texel(3, 8), RED);
        assert_eq!(texture(Wrap::ClampToBorder).texel(3, 3), white);
        assert_eq!(texture(Wrap::ClampToEdge).texel(-1, 3), white);
    }

    /// within a command list memory changes only where draws write, so a
    /// texture is checked once, and again after a draw over it or in the
    /// next list.
    #[test]
    fn textures_are_checked_once_a_list_unless_drawn_over() {
        let mut cache = TextureCache::default();
        let key = (0x1000, TextureFormat::Rgba8, 8, 8);
        cache.begin_list();
        assert!(cache.checked(key, 256).is_none(), "never seen");
        cache.decoded(key, &[0x11; 256]);
        assert!(cache.checked(key, 256).is_some());
        cache.wrote(0x5000, 0x100);
        assert!(cache.checked(key, 256).is_some(), "a draw somewhere else");
        cache.wrote(0x1080, 0x10);
        assert!(cache.checked(key, 256).is_none(), "a draw over it");
        cache.decoded(key, &[0x11; 256]);
        assert!(cache.checked(key, 256).is_some(), "checked again after the draw");
        cache.begin_list();
        assert!(cache.checked(key, 256).is_none(), "a new list");
    }

    /// a few tiles of a texture rewritten, as a title animating sprites in
    /// an atlas, come out the same as the whole texture decoded again.
    #[test]
    fn a_partly_changed_texture_decodes_as_a_whole_one_would() {
        for format in [TextureFormat::Etc1, TextureFormat::Etc1A4, TextureFormat::Rgba8] {
            let (width, height) = (32, 16);
            let size = (width * height * format.bits_per_pixel() / 8) as usize;
            let before: Vec<u8> = (0..size).map(|i| (i * 7 % 251) as u8).collect();
            let mut after = before.clone();
            // the third tile of the second row
            let tile = size / 8;
            after[6 * tile..7 * tile].iter_mut().for_each(|byte| *byte = byte.wrapping_mul(3).wrapping_add(1));
            let key = (0x1000, format, width, height);
            let mut cache = TextureCache::default();
            let old = cache.decoded(key, &before).0;
            let updated = cache.decoded(key, &after).0;
            let whole = TextureCache::default().decoded(key, &after).0;
            assert_eq!(updated, whole, "{format:?}");
            let changed = old.iter().zip(updated.iter()).enumerate().filter(|(_, (a, b))| a != b);
            assert!(changed.clone().count() > 0, "{format:?} changed nothing");
            for (index, _) in changed {
                let (x, y) = (index as u32 % width, index as u32 / width);
                assert_eq!((x / 8, y / 8), (2, 1), "{format:?} changed outside the tile, at {x}, {y}");
            }
        }
    }

    #[test]
    fn the_texture_cache_notices_changed_bytes() {
        let mut cache = TextureCache::default();
        let key = (0x1000, TextureFormat::Rgba8, 8, 8);
        let first = cache.decoded(key, &[0x11; 256]).0;
        let again = cache.decoded(key, &[0x11; 256]).0;
        assert!(Arc::ptr_eq(&first, &again));
        let changed = cache.decoded(key, &[0x22; 256]).0;
        assert_ne!(first[0], changed[0]);
    }

    /// a pixel centered on an edge goes to the triangle on its right, or
    /// above a flat edge, and a corner a hair off a center counts as on it.
    #[test]
    fn edges_through_pixel_centers_follow_the_pica() {
        let registers = target_registers();
        let corner = |x: f32, y: f32| Vertex {
            clip: [x / 4.0 - 1.0, y / 4.0 - 1.0, -0.5, 1.0],
            color: RED,
            texcoords: [[0.0; 2]; 3],
            quaternion: [0.0, 0.0, 0.0, 1.0],
            view: [0.0; 3],
        };
        for (left, right) in [(1.5, 3.5), (1.4999967, 3.499992)] {
            let quad = [(left, 1.5), (right, 1.5), (right, 3.5), (left, 1.5), (right, 3.5), (left, 3.5)];
            let mut memory = ConsoleMemory::default();
            rasterize_shaded(&registers, &mut memory, &mut Resources::default(), &quad.map(|(x, y)| corner(x, y)));
            // the rows as the window counts them, from the bottom
            let covered: Vec<(u32, u32)> = (0..8)
                .flat_map(|y| (0..8).map(move |x| (x, y)))
                .filter(|&(x, y)| {
                    let mut raw = [0u8; 4];
                    memory.read(COLOR + crate::format::morton_offset(x, 7 - y, 8, 4), &mut raw);
                    raw != [0; 4]
                })
                .collect();
            assert_eq!(covered, [(1, 1), (2, 1), (1, 2), (2, 2)]);
        }
    }

    /// a pixel whose center sits on an edge, or just beside a steep one,
    /// goes to the same triangle on the GPU as in software.
    #[cfg(feature = "vulkan")]
    #[test]
    fn the_gpu_covers_the_same_pixels() {
        let Ok(hardware) = hardware::Hardware::new() else { return };
        const SIZE: u32 = 32;
        let mut registers = target_registers();
        registers[REG_VIEWPORT_WIDTH] = float24(SIZE as f32 / 2.0);
        registers[REG_VIEWPORT_HEIGHT] = float24(SIZE as f32 / 2.0);
        registers[REG_FRAMEBUFFER_DIMENSIONS] = SIZE | ((SIZE - 1) << 12);

        let mut seed = 1u32;
        let mut random = move |range: u32| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) % range
        };
        // corners on a sixteenth of a pixel, which every GPU places exactly,
        // half of them on pixel centers
        let mut coordinate = || {
            let sixteenths = random(SIZE * 16 + 1);
            let sixteenths = if random(2) == 0 { sixteenths / 8 * 8 + 8 } else { sixteenths };
            (sixteenths as f32 / 16.0) / (SIZE as f32 / 2.0) - 1.0
        };
        let mut software = ConsoleMemory::default();
        let mut gpu = ConsoleMemory::default();
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        let mut differing = 0;
        // one at a time over a cleared buffer, so that every edge shows
        for _ in 0..200 {
            let triangle: Vec<Vertex> = (0..3)
                .map(|_| Vertex {
                    clip: [coordinate(), coordinate(), -0.5, 1.0],
                    color: RED,
                    texcoords: [[0.0; 2]; 3],
                    quaternion: [0.0, 0.0, 0.0, 1.0],
                    view: [0.0; 3],
                })
                .collect();
            let cleared = vec![0u8; (SIZE * SIZE * 4) as usize];
            software.write(COLOR, &cleared);
            gpu.write(COLOR, &cleared);
            rasterize_shaded(&registers, &mut software, &mut Resources::default(), &triangle);
            rasterize_shaded(&registers, &mut gpu, &mut resources, &triangle);
            resources.hardware.as_mut().unwrap().flush(&mut gpu).unwrap();
            differing += (0..SIZE * SIZE)
                .filter(|i| {
                    let (mut a, mut b) = ([0u8; 4], [0u8; 4]);
                    software.read(COLOR + i * 4, &mut a);
                    gpu.read(COLOR + i * 4, &mut b);
                    a != b
                })
                .count();
        }
        assert_eq!(differing, 0);
    }

    /// blending, color masks and logic ops come out on the GPU as in
    /// software, over draws one after another that each do something else,
    /// as draws sharing a pipeline do when the GPU sets that as it goes.
    /// each draw covers a row of its own, so none blends over another's.
    #[cfg(feature = "vulkan")]
    #[test]
    fn the_gpu_blends_and_masks_like_the_cpu() {
        const REG_BLEND_COLOR: usize = 0x103;
        let Ok(hardware) = hardware::Hardware::new() else { return };
        // a device without logic ops draws without them
        let logic_ops = hardware.logic_ops();
        let mut seed = 7u32;
        let mut random = move |range: u32| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) % range
        };
        let row = |k: u32, color: Vec4| -> Vec<Vertex> {
            let (bottom, top) = (k as f32 / 4.0 - 1.0, (k + 1) as f32 / 4.0 - 1.0);
            let corner = |x: f32, y: f32| Vertex {
                clip: [x, y, -0.5, 1.0],
                color,
                texcoords: [[0.0; 2]; 3],
                quaternion: [0.0, 0.0, 0.0, 1.0],
                view: [0.0; 3],
            };
            vec![corner(-1.0, bottom), corner(1.0, bottom), corner(1.0, top), corner(-1.0, bottom), corner(1.0, top), corner(-1.0, top)]
        };
        let mut software = ConsoleMemory::default();
        let mut gpu = ConsoleMemory::default();
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        for round in 0..40 {
            let start: Vec<u8> = (0..64 * 4).map(|_| random(256) as u8).collect();
            software.write(COLOR, &start);
            gpu.write(COLOR, &start);
            // software blends in floats and truncates, GPUs round, so a
            // blended row can be a step off, a logic op's not at all
            let mut steps = [0u8; 8];
            for (k, steps) in steps.iter_mut().enumerate() {
                let mut registers = target_registers();
                if !logic_ops || random(2) == 0 {
                    let (color, alpha) = (random(6), random(6));
                    let factors = [random(16), random(16), random(16), random(16)];
                    registers[REG_COLOR_OPERATION] = 0x100;
                    registers[REG_BLEND_FUNC] =
                        color | alpha << 8 | factors[0] << 16 | factors[1] << 20 | factors[2] << 24 | factors[3] << 28;
                    registers[REG_BLEND_COLOR] = random(1 << 16) << 16 | random(1 << 16);
                    *steps = 1;
                } else {
                    registers[REG_COLOR_OPERATION] = 0;
                    registers[REG_LOGIC_OP] = random(16);
                }
                registers[REG_DEPTH_COLOR_MASK] = random(16) << 8;
                // a quarter above a byte, which software truncates and the
                // GPU rounds to the same byte
                let color = std::array::from_fn(|_| (random(255) as f32 + 0.25) / 255.0);
                rasterize_shaded(&registers, &mut software, &mut Resources::default(), &row(k as u32, color));
                rasterize_shaded(&registers, &mut gpu, &mut resources, &row(k as u32, color));
            }
            resources.hardware.as_mut().unwrap().flush(&mut gpu).unwrap();
            for (y, &steps) in steps.iter().enumerate() {
                for x in 0..8 {
                    let at = COLOR + crate::format::morton_offset(x, 7 - y as u32, 8, 4);
                    let (mut a, mut b) = ([0u8; 4], [0u8; 4]);
                    software.read(at, &mut a);
                    gpu.read(at, &mut b);
                    let off = a.iter().zip(&b).map(|(p, q)| p.abs_diff(*q)).max().unwrap_or(0);
                    assert!(off <= steps, "round {round}, row {y} is off by {off}, {a:?} against {b:?}");
                }
            }
        }
    }

    /// draws in a row in one batch read their own uniforms, whether they
    /// are the last draw's or not, the ones the combiners take when the CPU
    /// placed the vertices and the vertex shader's when the GPU shades them,
    /// and so does a draw like the last one in the batch before.
    #[cfg(feature = "vulkan")]
    #[test]
    fn draws_in_a_row_read_their_own_uniforms() {
        let Ok(hardware) = hardware::Hardware::new() else { return };
        // the position from v0 and the color from c0
        const IDENTITY: u32 = 0x1B << 5 | 0x1B << 14 | 0x1B << 23;
        let mut unit = ShaderUnit::new();
        unit.descriptors[0] = 0xF | IDENTITY;
        let mov = |destination: u32, source: u32| 0x13 << 26 | destination << 21 | source << 12;
        unit.program[..3].copy_from_slice(&[mov(0, 0x00), mov(1, 0x20), 0x22 << 26]);
        unit.prepare();
        let mut registers = target_registers();
        registers[REG_SHADER_OUTPUT_TOTAL] = 2;
        registers[REG_SHADER_OUTPUT_MAP] = 0x0302_0100;
        registers[REG_SHADER_OUTPUT_MAP + 1] = 0x0B0A_0908;
        registers[REG_VS_OUTPUT_MASK] = 0b11;
        // the first combiner stage takes the vertex color or its constant,
        // the others hand it on
        for base in [0x0C8, 0x0D0, 0x0D8, 0x0F0, 0x0F8] {
            registers[base] = 0x000F_000F;
        }
        let colors = [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]];
        // a row each, with the last row's color now and then, the last
        // being the first
        let picks = [1, 0, 1, 0, 2, 2, 1, 1];
        let corners = |k: usize| {
            let (bottom, top) = (k as f32 / 4.0 - 1.0, (k + 1) as f32 / 4.0 - 1.0);
            [(-1.0, bottom), (1.0, bottom), (1.0, top), (-1.0, bottom), (1.0, top), (-1.0, top)]
        };
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        for shaded in [false, false, true, true] {
            let mut software = ConsoleMemory::default();
            let mut gpu = ConsoleMemory::default();
            for memory in [&mut software, &mut gpu] {
                memory.write(COLOR, &[0; 64 * 4]);
            }
            for (k, &pick) in picks.iter().enumerate() {
                let color = colors[pick];
                if shaded {
                    registers[0x0C0] = 0;
                    unit.float_uniforms[0] = color;
                    let inputs: Vec<_> = corners(k)
                        .iter()
                        .map(|&(x, y)| {
                            let mut input = [shader::ZERO; shader::INPUT_REGISTERS];
                            input[0] = [x, y, -0.5, 1.0];
                            input
                        })
                        .collect();
                    let vertices = Vertices::Unshaded { vertex_shader: &unit, geometry_shader: &unit, inputs: &inputs, order: None };
                    rasterize(&registers, &mut software, &mut Resources::default(), vertices);
                    rasterize(&registers, &mut gpu, &mut resources, vertices);
                } else {
                    registers[0x0C0] = 0x000E_000E;
                    registers[0x0C3] = u32::from_le_bytes(color.map(|c| (c * 255.0) as u8));
                    let vertices: Vec<Vertex> = corners(k)
                        .iter()
                        .map(|&(x, y)| Vertex {
                            clip: [x, y, -0.5, 1.0],
                            color: [1.0; 4],
                            texcoords: [[0.0; 2]; 3],
                            quaternion: [0.0, 0.0, 0.0, 1.0],
                            view: [0.0; 3],
                        })
                        .collect();
                    rasterize_shaded(&registers, &mut software, &mut Resources::default(), &vertices);
                    rasterize_shaded(&registers, &mut gpu, &mut resources, &vertices);
                }
            }
            resources.hardware.as_mut().unwrap().flush(&mut gpu).unwrap();
            let (mut a, mut b) = ([0u8; 64 * 4], [0u8; 64 * 4]);
            software.read(COLOR, &mut a);
            gpu.read(COLOR, &mut b);
            assert!(a.chunks(4).all(|pixel| pixel != [0; 4]), "every pixel drawn, shaded {shaded}");
            assert!(a == b, "shaded {shaded}, {a:?} against {b:?}");
        }
    }

    /// draws big enough that the ring passes RING_FLUSH, so the batch is
    /// handed over at the start of a draw whose uniform and shading words
    /// equal the draw before's, still read their own words.
    #[cfg(feature = "vulkan")]
    #[test]
    fn uniforms_survive_a_ring_flush_between_draws() {
        let Ok(hardware) = hardware::Hardware::new() else { return };
        const IDENTITY: u32 = 0x1B << 5 | 0x1B << 14 | 0x1B << 23;
        let mut unit = ShaderUnit::new();
        unit.descriptors[0] = 0xF | IDENTITY;
        let mov = |destination: u32, source: u32| 0x13 << 26 | destination << 21 | source << 12;
        unit.program[..3].copy_from_slice(&[mov(0, 0x00), mov(1, 0x20), 0x22 << 26]);
        // left unprepared, so all 16 input registers are staged, 256 bytes
        // a vertex, and 65536 vertices take 16 MB of the ring each draw
        let mut registers = target_registers();
        registers[REG_SHADER_OUTPUT_TOTAL] = 2;
        registers[REG_SHADER_OUTPUT_MAP] = 0x0302_0100;
        registers[REG_SHADER_OUTPUT_MAP + 1] = 0x0B0A_0908;
        registers[REG_VS_OUTPUT_MASK] = 0b11;
        for base in [0x0C8, 0x0D0, 0x0D8, 0x0F0, 0x0F8] {
            registers[base] = 0x000F_000F;
        }
        // the vertex color times a constant, which only the uniform block
        // carries, so a block read from the wrong place shows
        registers[0x0C0] = 0x00E0_00E0;
        registers[0x0C2] = 0x0001_0001;
        registers[0x0C3] = 0xFF80_40C0;
        let colors = [[1.0, 0.5, 0.25, 1.0], [0.5, 1.0, 0.75, 1.0], [0.25, 0.5, 1.0, 1.0]];
        // runs of the same color, so a draw after a flush has the words of
        // the one before it
        let picks = [1, 1, 1, 0, 0, 0, 2, 2];
        let corners = |k: usize| {
            let (bottom, top) = (k as f32 / 4.0 - 1.0, (k + 1) as f32 / 4.0 - 1.0);
            [(-1.0, bottom), (1.0, bottom), (1.0, top), (-1.0, bottom), (1.0, top), (-1.0, top)]
        };
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        let mut software = ConsoleMemory::default();
        let mut gpu = ConsoleMemory::default();
        for memory in [&mut software, &mut gpu] {
            memory.write(COLOR, &[0; 64 * 4]);
        }
        let mut inputs = vec![[shader::ZERO; shader::INPUT_REGISTERS]; 65536];
        let order: Vec<usize> = (0..6).collect();
        for (k, &pick) in picks.iter().enumerate() {
            unit.float_uniforms[0] = colors[pick];
            for (input, &(x, y)) in inputs.iter_mut().zip(&corners(k)) {
                input[0] = [x, y, -0.5, 1.0];
            }
            let vertices = Vertices::Unshaded { vertex_shader: &unit, geometry_shader: &unit, inputs: &inputs, order: Some(&order) };
            rasterize(&registers, &mut software, &mut Resources::default(), vertices);
            assert!(rasterize(&registers, &mut gpu, &mut resources, vertices).is_some());
        }
        resources.hardware.as_mut().unwrap().flush(&mut gpu).unwrap();
        let (mut a, mut b) = ([0u8; 64 * 4], [0u8; 64 * 4]);
        software.read(COLOR, &mut a);
        gpu.read(COLOR, &mut b);
        assert!(a.chunks(4).all(|pixel| pixel != [0; 4]), "every pixel drawn");
        assert!(a == b, "{a:?} against {b:?}");
    }

    /// the GPU has the CPU's reads of the depth it draws ask for it first,
    /// once until guest memory gets it, and again after. the color buffer
    /// the same draws go to is read as memory holds it. writes over either
    /// ask the same way.
    #[cfg(feature = "vulkan")]
    #[test]
    fn the_gpu_guards_what_it_draws() {
        struct Guarded<'a>(&'a mut ConsoleMemory, Vec<(u32, u32)>, Vec<(u32, u32)>);
        impl GpuMemory for Guarded<'_> {
            fn read(&mut self, addr: u32, out: &mut [u8]) {
                self.0.read(addr, out)
            }
            fn write(&mut self, addr: u32, data: &[u8]) {
                self.0.write(addr, data)
            }
            fn guard(&mut self, addr: u32, len: u32) {
                self.1.push((addr, len))
            }
            fn guard_writes(&mut self, addr: u32, len: u32) {
                self.2.push((addr, len))
            }
        }
        let Ok(hardware) = hardware::Hardware::new() else { return };
        let mut registers = target_registers();
        // depth written, with no test
        registers[REG_DEPTH_COLOR_MASK] |= 1 << 12;
        let mut memory = ConsoleMemory::default();
        let mut guarded = Guarded(&mut memory, Vec::new(), Vec::new());
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        rasterize_shaded(&registers, &mut guarded, &mut resources, &cover(-0.5, RED));
        // the whole 8 by 8 depth buffer, of four bytes a pixel
        assert_eq!(guarded.1, [(DEPTH, 8 * 8 * 4)]);
        assert_eq!(guarded.2, [(COLOR, 8 * 8 * 4), (DEPTH, 8 * 8 * 4)]);
        rasterize_shaded(&registers, &mut guarded, &mut resources, &cover(-0.5, RED));
        assert_eq!((guarded.1.len(), guarded.2.len()), (1, 2));
        resources.hardware.as_mut().unwrap().flush(&mut guarded).unwrap();
        // depth memory the next draw writes over
        guarded.0.write(DEPTH, &[0xFF; 8 * 8 * 4]);
        rasterize_shaded(&registers, &mut guarded, &mut resources, &cover(-0.5, GREEN));
        assert_eq!((guarded.1.len(), guarded.2.len()), (2, 4));

        // what the CPU's read brings back is the depth, the color stays as
        // memory had it, red from the flush
        let hardware = resources.hardware.as_mut().unwrap();
        hardware.sync_depth(&mut memory, COLOR, DEPTH + 8 * 8 * 4 - COLOR).unwrap();
        assert!(pixels(&mut memory).iter().all(|&p| p == [255, 0, 0, 255]));
        let mut depth = [0u8; 8 * 8 * 4];
        memory.read(DEPTH, &mut depth);
        // the stencil byte stays, nothing wrote it
        assert!(depth.as_chunks::<4>().0.iter().all(|sample| sample[..3] != [0xFF; 3]), "the depth drawn came back");
    }

    /// fog mixes into what the combiners made by the fragment's depth, on
    /// the GPU as the software path does it.
    #[test]
    fn fog_mixes_in_by_depth() {
        let mut registers = target_registers();
        // a z-buffer, depth 0.5 at clip z -0.5
        registers[REG_DEPTHMAP_ENABLE] = 1;
        registers[REG_VIEWPORT_DEPTH_RANGE] = float24(-1.0);
        registers[REG_VIEWPORT_DEPTH_NEAR] = float24(0.0);
        registers[crate::fog::REG_MODE] |= 5;
        registers[crate::fog::REG_COLOR] = 0x00_80_40_20;
        let mut resources = Resources::default();
        // a factor falling from 1 to 0 across the table, a half at 0.5
        for entry in 0..crate::fog::ENTRIES as u32 {
            let factor = (2048 - entry * 16).min(2047);
            resources.fog_table.write(&mut registers, (factor << 13) | (0x2000 - 16));
        }
        let mut memory = ConsoleMemory::default();
        rasterize_shaded(&registers, &mut memory, &mut resources, &cover(-0.5, RED));
        assert!(pixels(&mut memory).iter().all(|&p| p == [143, 32, 64, 255]), "{:?}", pixels(&mut memory)[0]);

        #[cfg(feature = "vulkan")]
        if let Ok(hardware) = hardware::Hardware::new() {
            let mut gpu = ConsoleMemory::default();
            resources.hardware = Some(hardware);
            rasterize_shaded(&registers, &mut gpu, &mut resources, &cover(-0.5, RED));
            resources.hardware.as_mut().unwrap().flush(&mut gpu).unwrap();
            assert!(pixels(&mut gpu).iter().all(|&p| p == [143, 32, 64, 255]), "{:?}", pixels(&mut gpu)[0]);
        }
    }

    /// the target at another size, the viewport over all of it.
    fn sized(registers: &mut [u32], width: u32, height: u32) {
        registers[REG_VIEWPORT_WIDTH] = float24(width as f32 / 2.0);
        registers[REG_VIEWPORT_HEIGHT] = float24(height as f32 / 2.0);
        registers[REG_FRAMEBUFFER_DIMENSIONS] = width | ((height - 1) << 12);
    }

    /// a fill over every row the GPU drew leaves the buffer as memory holds
    /// it, and the next draw there has the CPU's writes ask first again.
    #[cfg(feature = "vulkan")]
    #[test]
    fn a_fill_over_all_that_was_drawn_guards_the_next_draw_again() {
        struct Guarded<'a>(&'a mut ConsoleMemory, Vec<(u32, u32)>);
        impl GpuMemory for Guarded<'_> {
            fn read(&mut self, addr: u32, out: &mut [u8]) {
                self.0.read(addr, out)
            }
            fn write(&mut self, addr: u32, data: &[u8]) {
                self.0.write(addr, data)
            }
            fn guard_writes(&mut self, addr: u32, len: u32) {
                self.1.push((addr, len))
            }
        }
        let Ok(hardware) = hardware::Hardware::new() else { return };
        let mut registers = target_registers();
        sized(&mut registers, 8, 16);
        // the window's top half, the first rows of memory
        registers[REG_VIEWPORT_XY] = 8 << 16;
        registers[REG_VIEWPORT_HEIGHT] = float24(4.0);
        let mut memory = ConsoleMemory::default();
        let mut guarded = Guarded(&mut memory, Vec::new());
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        rasterize_shaded(&registers, &mut guarded, &mut resources, &cover(-0.5, RED));
        assert_eq!(guarded.1, [(COLOR, 256)]);
        let fill = [0u8; 256];
        guarded.0.write(COLOR, &fill);
        resources.hardware.as_mut().unwrap().filled(COLOR, &fill).unwrap();
        rasterize_shaded(&registers, &mut guarded, &mut resources, &cover(-0.5, GREEN));
        assert_eq!(guarded.1, [(COLOR, 256), (COLOR, 256)]);
    }

    /// a buffer drawn over rows a fill left of a taller one, which still has
    /// other rows drawn, is the one a transfer of those rows reads.
    #[cfg(feature = "vulkan")]
    #[test]
    fn rows_drawn_after_a_fill_come_from_the_buffer_that_drew_them() {
        const OUTPUT: u32 = 0x10_0000;
        let Ok(hardware) = hardware::Hardware::new() else { return };
        let mut tall = target_registers();
        sized(&mut tall, 8, 16);
        tall[REG_DEPTH_COLOR_MASK] |= 1 << 12;
        let mut short = target_registers();
        // a depth buffer of its own keeps it apart from the taller one
        short[REG_DEPTH_COLOR_MASK] |= 1 << 12;
        short[REG_DEPTH_BUFFER_ADDRESS] = 0x5000 >> 3;
        let mut memory = ConsoleMemory::default();
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        rasterize_shaded(&tall, &mut memory, &mut resources, &cover(-0.5, RED));
        let fill = [0u8; 256];
        memory.write(COLOR, &fill);
        resources.hardware.as_mut().unwrap().filled(COLOR, &fill).unwrap();
        rasterize_shaded(&short, &mut memory, &mut resources, &cover(-0.5, GREEN));
        let hardware = resources.hardware.as_mut().unwrap();
        // a texture of those rows is copied from the shorter buffer on the
        // GPU, one over both comes from memory
        let texture = |height| DrawnTexture { addr: COLOR, width: 8, height, format: ColorFormat::Rgba8 };
        assert!(hardware.holds(&texture(8)));
        assert!(!hardware.holds(&texture(16)));
        let transfer = hardware::Transfer {
            input: COLOR,
            output: OUTPUT,
            input_width: 8,
            input_height: 8,
            output_width: 8,
            output_height: 8,
            copy: (8, 8),
            scale: (1, 1),
            flip: false,
            input_linear: false,
            output_tiled: false,
            input_format: ColorFormat::Rgba8,
            output_format: ColorFormat::Rgba8,
        };
        assert!(hardware.display_transfer(&mut memory, &transfer).unwrap());
        hardware.flush(&mut memory).unwrap();
        let mut out = [0u8; 256];
        memory.read(OUTPUT, &mut out);
        assert!(out.as_chunks::<4>().0.iter().all(|p| ColorFormat::Rgba8.decode(p) == [0, 255, 0, 255]));
    }

    /// memory written over rows a fill cleared of a buffer still drawn
    /// elsewhere, a DMA after the GPU was synced, is what a texture of those
    /// rows shows, and what the buffer holds once used again.
    #[cfg(feature = "vulkan")]
    #[test]
    fn rows_written_after_a_fill_reach_textures_and_the_buffer() {
        const OUTPUT: u32 = 0x10_0000;
        for mut memory in [ConsoleMemory::default(), ConsoleMemory::scattered()] {
            let Ok(hardware) = hardware::Hardware::new() else { return };
            let mut registers = target_registers();
            sized(&mut registers, 8, 16);
            let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
            rasterize_shaded(&registers, &mut memory, &mut resources, &cover(-0.5, RED));
            let fill = [0u8; 256];
            memory.write(COLOR, &fill);
            let hardware = resources.hardware.as_mut().unwrap();
            hardware.filled(COLOR, &fill).unwrap();
            hardware.sync(&mut memory, COLOR, 256).unwrap();
            let mut green = [0u8; 4];
            ColorFormat::Rgba8.encode([0, 255, 0, 255], &mut green);
            memory.write(COLOR, &green.repeat(64));
            let texture = DrawnTexture { addr: COLOR, width: 8, height: 8, format: ColorFormat::Rgba8 };
            assert!(!hardware.holds(&texture), "the texture comes from memory");
            // the rest of the drawing comes down, and the buffer is used again
            hardware.sync(&mut memory, COLOR + 256, 256).unwrap();
            let transfer = hardware::Transfer {
                input: COLOR,
                output: OUTPUT,
                input_width: 8,
                input_height: 16,
                output_width: 8,
                output_height: 16,
                copy: (8, 16),
                scale: (1, 1),
                flip: false,
                input_linear: false,
                output_tiled: false,
                input_format: ColorFormat::Rgba8,
                output_format: ColorFormat::Rgba8,
            };
            assert!(hardware.display_transfer(&mut memory, &transfer).unwrap());
            hardware.flush(&mut memory).unwrap();
            let mut out = [0u8; 512];
            memory.read(OUTPUT, &mut out);
            let pixels: Vec<[u8; 4]> = out.as_chunks::<4>().0.iter().map(|p| ColorFormat::Rgba8.decode(p)).collect();
            assert_eq!(pixels.iter().filter(|&&p| p == [0, 255, 0, 255]).count(), 64, "the rows written are green");
            assert_eq!(pixels.iter().filter(|&&p| p == [255, 0, 0, 255]).count(), 64, "the rows drawn stay red");
        }
    }

    /// a narrower buffer drawn over rows a fill left of a wider one does not
    /// cost the wider one the rows it drew elsewhere, which memory lacks,
    /// when it is used again.
    #[cfg(feature = "vulkan")]
    #[test]
    fn a_buffer_used_again_keeps_the_rows_it_drew() {
        const OUTPUT: u32 = 0x10_0000;
        let Ok(hardware) = hardware::Hardware::new() else { return };
        let mut wide = target_registers();
        sized(&mut wide, 16, 16);
        let narrow = target_registers();
        let mut memory = ConsoleMemory::default();
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        rasterize_shaded(&wide, &mut memory, &mut resources, &cover(-0.5, RED));
        // its first row of tiles, 8 rows of 16 pixels
        let fill = [0u8; 512];
        memory.write(COLOR, &fill);
        resources.hardware.as_mut().unwrap().filled(COLOR, &fill).unwrap();
        rasterize_shaded(&narrow, &mut memory, &mut resources, &cover(-0.5, GREEN));
        let hardware = resources.hardware.as_mut().unwrap();
        let transfer = hardware::Transfer {
            input: COLOR,
            output: OUTPUT,
            input_width: 16,
            input_height: 16,
            output_width: 16,
            output_height: 16,
            copy: (16, 16),
            scale: (1, 1),
            flip: false,
            input_linear: false,
            output_tiled: false,
            input_format: ColorFormat::Rgba8,
            output_format: ColorFormat::Rgba8,
        };
        assert!(hardware.display_transfer(&mut memory, &transfer).unwrap());
        hardware.sync(&mut memory, COLOR + 512, 512).unwrap();
        let mut rest = [0u8; 512];
        memory.read(COLOR + 512, &mut rest);
        assert!(rest.as_chunks::<4>().0.iter().all(|p| ColorFormat::Rgba8.decode(p) == [255, 0, 0, 255]), "the rows it drew stay red");
    }

    /// a fill over the first rows of a buffer the GPU drew leaves only the
    /// rest of the drawing to come down, so a buffer of another shape over
    /// the filled rows does not have it written back first.
    #[cfg(feature = "vulkan")]
    #[test]
    fn a_fill_leaves_only_the_rest_of_the_drawing() {
        let Ok(hardware) = hardware::Hardware::new() else { return };
        let mut registers = target_registers();
        // eight by sixteen, two rows of tiles
        registers[REG_VIEWPORT_HEIGHT] = float24(8.0);
        registers[REG_FRAMEBUFFER_DIMENSIONS] = 8 | (15 << 12);
        let mut memory = ConsoleMemory::default();
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        rasterize_shaded(&registers, &mut memory, &mut resources, &cover(-0.5, RED));
        let fill = [0x11u8, 0x22, 0x33, 0x44].repeat(64);
        memory.write(COLOR, &fill);
        let hardware = resources.hardware.as_mut().unwrap();
        hardware.filled(COLOR, &fill).unwrap();
        let mut rest = [0u8; 256];
        hardware.sync(&mut memory, COLOR, 256).unwrap();
        memory.read(COLOR + 256, &mut rest);
        assert_eq!(rest, [0; 256], "nothing over the filled rows had to come down");
        hardware.sync(&mut memory, COLOR + 256, 256).unwrap();
        memory.read(COLOR + 256, &mut rest);
        assert!(rest.as_chunks::<4>().0.iter().all(|p| ColorFormat::Rgba8.decode(p) == [255, 0, 0, 255]), "the rest of the drawing does");
        let mut first = [0u8; 256];
        memory.read(COLOR, &mut first);
        assert_eq!(first[..], fill[..], "and the fill stays");
    }

    /// a fill over the start of a buffer the GPU drew, in a pattern its
    /// pixels can't take, does not have the drawing come down then, which
    /// waits for the GPU. once it does come down, the bytes the fill wrote
    /// stay, even ones the fill left as memory held them.
    #[cfg(feature = "vulkan")]
    #[test]
    fn a_fill_over_part_of_a_drawing_leaves_the_rest_on_the_gpu() {
        let Ok(hardware) = hardware::Hardware::new() else { return };
        let mut memory = ConsoleMemory::default();
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        rasterize_shaded(&target_registers(), &mut memory, &mut resources, &cover(-0.5, RED));
        // a three byte pattern of zeros, as memory held them, over 100 bytes
        let hardware = resources.hardware.as_mut().unwrap();
        hardware.before_fill(&mut memory, COLOR, 100, 3).unwrap();
        memory.write(COLOR, &[0; 100]);
        hardware.filled(COLOR, &[0; 100]).unwrap();
        let mut all = [0u8; 256];
        memory.read(COLOR, &mut all);
        assert_eq!(all, [0; 256], "the drawing came down for the fill");
        hardware.sync(&mut memory, COLOR, 256).unwrap();
        memory.read(COLOR, &mut all);
        assert_eq!(all[..100], [0; 100], "the fill stays");
        assert!(all[100..].as_chunks::<4>().0.iter().all(|p| ColorFormat::Rgba8.decode(p) == [255, 0, 0, 255]), "the rest of the drawing comes down");
    }

    /// a texture reaching past the rows of the buffer drawn into it, as its
    /// sides are powers of two, samples the same from a copy the GPU makes,
    /// without the buffer coming down first, as from memory once it has the
    /// buffer, the rows past the buffer as memory holds them either way.
    #[cfg(feature = "vulkan")]
    #[test]
    fn a_texture_reaching_past_a_drawing_samples_like_memory() {
        let (Ok(first), Ok(second)) = (hardware::Hardware::new(), hardware::Hardware::new()) else { return };
        const SIZE: u32 = 32;
        const TARGET: u32 = 0x10_0000;
        // a buffer half as tall as the texture drawn into
        let mut drawing = target_registers();
        drawing[REG_VIEWPORT_WIDTH] = float24(SIZE as f32 / 2.0);
        drawing[REG_VIEWPORT_HEIGHT] = float24(SIZE as f32 / 4.0);
        drawing[REG_FRAMEBUFFER_DIMENSIONS] = SIZE | ((SIZE / 2 - 1) << 12);
        let mut sampling = target_registers();
        sampling[REG_COLOR_BUFFER_ADDRESS] = TARGET >> 3;
        sampling[REG_VIEWPORT_WIDTH] = float24(SIZE as f32 / 2.0);
        sampling[REG_VIEWPORT_HEIGHT] = float24(SIZE as f32 / 2.0);
        sampling[REG_FRAMEBUFFER_DIMENSIONS] = SIZE | ((SIZE - 1) << 12);
        sampling[REG_TEXTURE_CONFIG] = 1;
        sampling[REG_TEXTURE0_DIMENSIONS] = SIZE | (SIZE << 16);
        sampling[REG_TEXTURE0_ADDRESS] = COLOR >> 3;
        sampling[0x0C0] = 0x3 | (0x3 << 16);
        for stage in [0x0C8, 0x0D0, 0x0D8, 0x0F0, 0x0F8] {
            sampling[stage] = 0xF | (0xF << 16);
        }

        let mut seed = 7u32;
        let mut random = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1 << 24) as f32
        };
        let triangles: Vec<Vec<Vertex>> = (0..30)
            .map(|_| {
                let color = [random(), random(), random(), 1.0];
                (0..3)
                    .map(|_| Vertex {
                        clip: [random() * 2.0 - 1.0, random() * 2.0 - 1.0, -random(), 1.0],
                        color,
                        texcoords: [[0.0; 2]; 3],
                        quaternion: [0.0, 0.0, 0.0, 1.0],
                        view: [0.0; 3],
                    })
                    .collect()
            })
            .collect();
        // the rows past the buffer, whatever memory holds there
        let past: Vec<u8> = (0..SIZE * SIZE / 2 * 4).map(|_| (random() * 256.0) as u8).collect();
        let quad: Vec<Vertex> = [[-1.0, -1.0, 0.0, 0.0], [3.0, -1.0, 2.0, 0.0], [-1.0, 3.0, 0.0, 2.0]]
            .into_iter()
            .map(|[x, y, u, v]| Vertex {
                clip: [x, y, -0.5, 1.0],
                color: [1.0; 4],
                texcoords: [[u, v], [0.0; 2], [0.0; 2]],
                quaternion: [0.0, 0.0, 0.0, 1.0],
                view: [0.0; 3],
            })
            .collect();

        let mut results = Vec::new();
        for (hardware, back_first) in [(first, false), (second, true)] {
            let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
            let mut memory = ConsoleMemory::default();
            memory.write(COLOR + SIZE * SIZE / 2 * 4, &past);
            for triangle in &triangles {
                rasterize_shaded(&drawing, &mut memory, &mut resources, triangle);
            }
            if back_first {
                resources.hardware.as_mut().unwrap().flush(&mut memory).unwrap();
            }
            rasterize_shaded(&sampling, &mut memory, &mut resources, &quad);
            if !back_first {
                let mut drawn = vec![0u8; (SIZE * SIZE / 2 * 4) as usize];
                memory.read(COLOR, &mut drawn);
                assert!(drawn.iter().all(|&b| b == 0), "the drawing came down for the texture");
            }
            resources.hardware.as_mut().unwrap().flush(&mut memory).unwrap();
            let mut sampled = vec![0u8; (SIZE * SIZE * 4) as usize];
            memory.read(TARGET, &mut sampled);
            results.push(sampled);
        }
        assert!(results[0] == results[1], "the copy samples as memory does");
        let colors: std::collections::HashSet<&[u8]> = results[0].chunks(4).collect();
        assert!(colors.len() > 8, "only {} colors", colors.len());
    }

    /// a transfer out of the last rows of a buffer the GPU drew stays on
    /// the GPU when its output lies over the buffer's first rows, which it
    /// does not read, as Monster Hunter 3 Ultimate's bottom screen does.
    /// over rows it reads, the CPU does it in order.
    #[cfg(feature = "vulkan")]
    #[test]
    fn a_transfer_over_rows_it_does_not_read_stays_on_the_gpu() {
        let Ok(hardware) = hardware::Hardware::new() else { return };
        let mut registers = target_registers();
        // eight by sixteen, two rows of tiles
        registers[REG_VIEWPORT_HEIGHT] = float24(8.0);
        registers[REG_FRAMEBUFFER_DIMENSIONS] = 8 | (15 << 12);
        let mut memory = ConsoleMemory::default();
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        rasterize_shaded(&registers, &mut memory, &mut resources, &cover(-0.5, RED));
        let hardware = resources.hardware.as_mut().unwrap();
        let transfer = |input: u32, output: u32| hardware::Transfer {
            input,
            output,
            input_width: 8,
            input_height: 8,
            output_width: 8,
            output_height: 8,
            copy: (8, 8),
            scale: (1, 1),
            flip: false,
            input_linear: false,
            output_tiled: false,
            input_format: ColorFormat::Rgba8,
            output_format: ColorFormat::Rgba8,
        };
        // the second row of tiles to a plain buffer over the first
        assert!(hardware.display_transfer(&mut memory, &transfer(COLOR + 256, COLOR)).unwrap(), "the CPU did it");
        hardware.flush(&mut memory).unwrap();
        let mut out = [0u8; 256];
        memory.read(COLOR, &mut out);
        assert!(out.as_chunks::<4>().0.iter().all(|p| ColorFormat::Rgba8.decode(p) == [255, 0, 0, 255]));
        // over the rows it reads
        assert!(!hardware.display_transfer(&mut memory, &transfer(COLOR + 256, COLOR + 384)).unwrap());
    }

    /// a transfer on the GPU of an 8 pixel wide buffer at COLOR, of rows
    /// pixels, to a plain one at OUTPUT, and the pixels it leaves there.
    #[cfg(feature = "vulkan")]
    fn transferred(memory: &mut ConsoleMemory, resources: &mut Resources, rows: u32) -> Vec<[u8; 4]> {
        const OUTPUT: u32 = 0x10_0000;
        let hardware = resources.hardware.as_mut().unwrap();
        let transfer = hardware::Transfer {
            input: COLOR,
            output: OUTPUT,
            input_width: 8,
            input_height: rows,
            output_width: 8,
            output_height: rows,
            copy: (8, rows),
            scale: (1, 1),
            flip: false,
            input_linear: false,
            output_tiled: false,
            input_format: ColorFormat::Rgba8,
            output_format: ColorFormat::Rgba8,
        };
        assert!(hardware.display_transfer(memory, &transfer).unwrap());
        hardware.flush(memory).unwrap();
        let mut out = vec![0u8; 8 * rows as usize * 4];
        memory.read(OUTPUT, &mut out);
        out.as_chunks::<4>().0.iter().map(|p| ColorFormat::Rgba8.decode(p)).collect()
    }

    /// a pixel written over a buffer a fill cleared, as the CPU writes one,
    /// is in the buffer the next time it is used, the fill's pixel standing
    /// in for its shadow notices it as the bytes would.
    #[cfg(feature = "vulkan")]
    #[test]
    fn a_pixel_written_after_a_fill_reaches_the_buffer() {
        for mut memory in [ConsoleMemory::default(), ConsoleMemory::scattered()] {
            let Ok(hardware) = hardware::Hardware::new() else { return };
            let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
            rasterize_shaded(&target_registers(), &mut memory, &mut resources, &cover(-0.5, RED));
            let (mut blue, mut green) = ([0u8; 4], [0u8; 4]);
            ColorFormat::Rgba8.encode([0, 0, 255, 255], &mut blue);
            ColorFormat::Rgba8.encode([0, 255, 0, 255], &mut green);
            let fill = blue.repeat(64);
            memory.write(COLOR, &fill);
            resources.hardware.as_mut().unwrap().filled(COLOR, &fill).unwrap();
            memory.write(COLOR + 4 * 37, &green);
            let pixels = transferred(&mut memory, &mut resources, 8);
            assert_eq!(pixels.iter().filter(|&&p| p == [0, 255, 0, 255]).count(), 1, "the pixel written");
            assert_eq!(pixels.iter().filter(|&&p| p == [0, 0, 255, 255]).count(), 63, "and the fill");
        }
    }

    /// memory of one pixel but for one past the first few kilobytes is not
    /// kept as that pixel, the drawing over it reaches memory when written
    /// back, rather than looking like a write made after it.
    #[cfg(feature = "vulkan")]
    #[test]
    fn a_drawing_over_a_pixel_past_the_first_group_reaches_memory() {
        for mut memory in [ConsoleMemory::default(), ConsoleMemory::scattered()] {
            let Ok(hardware) = hardware::Hardware::new() else { return };
            let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
            let mut registers = target_registers();
            sized(&mut registers, 64, 64);
            registers[REG_DEPTH_BUFFER_ADDRESS] = 0x8_0000 >> 3;
            let (mut gray, mut green) = ([0u8; 4], [0u8; 4]);
            ColorFormat::Rgba8.encode([128, 128, 128, 255], &mut gray);
            ColorFormat::Rgba8.encode([0, 255, 0, 255], &mut green);
            let size = 64 * 64 * 4;
            memory.write(COLOR, &gray.repeat(64 * 64));
            let at = size as u32 - 4 * 3;
            memory.write(COLOR + at, &green);
            rasterize_shaded(&registers, &mut memory, &mut resources, &cover(-0.5, RED));
            resources.hardware.as_mut().unwrap().flush(&mut memory).unwrap();
            let mut all = vec![0u8; size];
            memory.read(COLOR, &mut all);
            let red = all.as_chunks::<4>().0.iter().filter(|p| ColorFormat::Rgba8.decode(&p[..]) == [255, 0, 0, 255]).count();
            assert_eq!(red, 64 * 64, "the drawing reached all of memory, the pixel past the first group too");
        }
    }

    /// a buffer over memory all of one pixel, which its shadow keeps as that
    /// pixel, has a pixel changed there the next time it is used, with the
    /// rows it drew.
    #[cfg(feature = "vulkan")]
    #[test]
    fn a_pixel_changed_in_memory_of_one_pixel_reaches_the_buffer() {
        for mut memory in [ConsoleMemory::default(), ConsoleMemory::scattered()] {
            let Ok(hardware) = hardware::Hardware::new() else { return };
            let mut registers = target_registers();
            sized(&mut registers, 8, 16);
            // the window's top half, the first rows of memory
            registers[REG_VIEWPORT_XY] = 8 << 16;
            registers[REG_VIEWPORT_HEIGHT] = float24(4.0);
            let (mut gray, mut green) = ([0u8; 4], [0u8; 4]);
            ColorFormat::Rgba8.encode([128, 128, 128, 255], &mut gray);
            ColorFormat::Rgba8.encode([0, 255, 0, 255], &mut green);
            memory.write(COLOR, &gray.repeat(128));
            let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
            rasterize_shaded(&registers, &mut memory, &mut resources, &cover(-0.5, RED));
            // a sync of rows it did not draw, which has it look at memory again
            resources.hardware.as_mut().unwrap().sync(&mut memory, COLOR + 256, 256).unwrap();
            memory.write(COLOR + 256 + 4 * 5, &green);
            let pixels = transferred(&mut memory, &mut resources, 16);
            assert_eq!(pixels.iter().filter(|&&p| p == [255, 0, 0, 255]).count(), 64, "the rows drawn");
            assert_eq!(pixels.iter().filter(|&&p| p == [0, 255, 0, 255]).count(), 1, "the pixel written");
            assert_eq!(pixels.iter().filter(|&&p| p == [128, 128, 128, 255]).count(), 63, "and what memory held");
        }
    }

    /// memory written after the GPU drew over it by something that does not
    /// wait for the drawing stays when the drawing is written back.
    #[cfg(feature = "vulkan")]
    #[test]
    fn writes_after_drawing_survive_the_write_back() {
        for mut memory in [ConsoleMemory::default(), ConsoleMemory::scattered()] {
            let Ok(hardware) = hardware::Hardware::new() else { return };
            let registers = target_registers();
            let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
            rasterize_shaded(&registers, &mut memory, &mut resources, &cover(-0.5, RED));
            memory.write(COLOR, &[0x5A; 16]);
            resources.hardware.as_mut().unwrap().sync(&mut memory, COLOR, 8 * 8 * 4).unwrap();
            let mut start = [0u8; 16];
            memory.read(COLOR, &mut start);
            assert_eq!(start, [0x5A; 16], "what was written after the drawing stays");
            assert!(pixels(&mut memory)[4..].iter().all(|&p| p == [255, 0, 0, 255]), "and the drawing is the rest");
        }
    }

    /// a service that has the drawing written back before it writes, as a
    /// file read does, keeps every byte it wrote, those equal to what memory
    /// held under the drawing too, when a write of the CPU later asks for
    /// the drawing again.
    #[cfg(feature = "vulkan")]
    #[test]
    fn writes_after_the_drawing_came_down_stay() {
        let Ok(hardware) = hardware::Hardware::new() else { return };
        let registers = target_registers();
        let mut memory = ConsoleMemory::default();
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        rasterize_shaded(&registers, &mut memory, &mut resources, &cover(-0.5, RED));
        let hardware = resources.hardware.as_mut().unwrap();
        hardware.sync(&mut memory, COLOR, 16).unwrap();
        // zeros, as memory held before the drawing
        memory.write(COLOR, &[0; 16]);
        hardware.sync(&mut memory, COLOR, 8 * 8 * 4).unwrap();
        let mut start = [0xFFu8; 16];
        memory.read(COLOR, &mut start);
        assert_eq!(start, [0; 16], "what the service wrote stays");
        assert!(pixels(&mut memory)[4..].iter().all(|&p| p == [255, 0, 0, 255]), "and the drawing is the rest");
    }

    /// vertices the GPU shades come out as the CPU shades them, clipped and
    /// culled the same, through a program with a loop, a call, both kinds
    /// of if, indexed uniforms and an output past a gap in the mask.
    #[cfg(feature = "vulkan")]
    #[test]
    fn the_gpu_shades_like_the_cpu() {
        let Ok(hardware) = hardware::Hardware::new() else { return };
        const SIZE: u32 = 32;
        const IDENTITY: u32 = 0x1B << 5 | 0x1B << 14 | 0x1B << 23;
        let mut unit = ShaderUnit::new();
        let descriptors = [
            0xF | IDENTITY,
            0x8 | IDENTITY,
            0x4 | IDENTITY,
            0x2 | IDENTITY,
            0x1 | IDENTITY,
            // w, a.w times b.x plus c.y
            0x1 | 0x1B << 5 | 0x55 << 23,
            // x, the first source's z
            0x8 | 0xAA << 5,
            0xE | IDENTITY,
            // the second source's x everywhere
            0xF | 0x1B << 5,
        ];
        unit.descriptors[..descriptors.len()].copy_from_slice(&descriptors);
        let op = |opcode: u32, destination: u32, src1: u32, src2: u32, index: u32, descriptor: u32| {
            opcode << 26 | destination << 21 | index << 19 | src1 << 12 | src2 << 7 | descriptor
        };
        let flow = |opcode: u32, condition: u32, destination: u32, count: u32| {
            opcode << 26 | condition << 22 | destination << 10 | count
        };
        let program = [
            op(0x12, 0, 0x02, 0, 0, 1),
            // the matrix at c1 through a0.x, which v2 sets to one
            op(0x02, 0, 0x20, 0x00, 1, 1),
            op(0x02, 0, 0x21, 0x00, 1, 2),
            op(0x02, 0, 0x22, 0x00, 1, 3),
            op(0x02, 0, 0x23, 0x00, 1, 4),
            op(0x13, 0x10, 0x2A, 0, 0, 0),
            // r0 is c5 + c6 + c7
            flow(0x29, 0, 7, 0),
            op(0x00, 0x10, 0x25, 0x10, 3, 0),
            flow(0x27, 0, 10, 1),
            op(0x08, 2, 0x01, 0x10, 0, 0),
            op(0x13, 2, 0x01, 0, 0, 0),
            // c8.x < v1.x and c8.y > v1.y
            0x2E << 26 | 2 << 24 | 4 << 21 | 0x28 << 12 | 0x01 << 7,
            flow(0x28, 0b1010, 14, 0),
            0x38 << 26 | 2 << 24 | 0x01 << 17 | 0x29 << 10 | 0x10 << 5 | 5,
            flow(0x24, 0, 20, 3),
            0x22 << 26,
        ];
        unit.program[..program.len()].copy_from_slice(&program);
        unit.program[20] = op(0x0E, 0x11, 0x29, 0, 0, 6);
        unit.program[21] = op(0x08, 0x12, 0x01, 0x11, 0, 8);
        unit.program[22] = op(0x0D, 2, 0x12, 0x10, 0, 7);
        let uniforms = [
            [0.0; 4],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.3, 1.0],
            [0.5, 0.1, 0.2, 0.3],
            [0.2, 0.3, 0.1, 0.3],
            [0.1, 0.2, 0.4, 0.4],
            [0.5, 0.5, 0.0, 0.0],
            [0.3, 0.0, 2.0, 0.0],
            [0.0; 4],
        ];
        unit.float_uniforms[..uniforms.len()].copy_from_slice(&uniforms);
        unit.int_uniforms[0] = [2, 0, 1, 0];
        unit.bool_uniforms = 1;
        unit.prepare();

        let mut registers = target_registers();
        registers[REG_VIEWPORT_XY] = 4 | 6 << 16;
        registers[REG_VIEWPORT_WIDTH] = float24(12.0);
        registers[REG_VIEWPORT_HEIGHT] = float24(12.0);
        registers[REG_FRAMEBUFFER_DIMENSIONS] = SIZE | ((SIZE - 1) << 12);
        registers[REG_DEPTH_COLOR_MASK] = 0xF << 8 | 1 | 4 << 4 | 1 << 12;
        registers[REG_VIEWPORT_DEPTH_RANGE] = float24(-1.0);
        registers[REG_DEPTHMAP_ENABLE] = 1;
        registers[REG_SHADER_OUTPUT_TOTAL] = 2;
        registers[REG_SHADER_OUTPUT_MAP] = 0x0302_0100;
        registers[REG_SHADER_OUTPUT_MAP + 1] = 0x0B0A_0908;
        registers[REG_VS_OUTPUT_MASK] = 0b101;

        let mut seed = 7u32;
        let mut random = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1 << 24) as f32
        };
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        let (mut differing, mut drawn, mut deepest) = (0, 0, 0);
        for case in 0..36 {
            let (topology, cull) = (case % 3, case / 3 % 3);
            // half the cases reach past the near and far planes, where the
            // CPU places the vertices clipping makes on a sixteenth of a
            // pixel and the GPU the ones it clipped, which moves depth a bit
            let clipped = case >= 18;
            let (near, depth_range) = if clipped { (-1.3, 1.6) } else { (-0.7, 0.6) };
            registers[REG_PRIMITIVE_CONFIG] = topology << 8;
            registers[REG_FACE_CULLING] = cull;
            let mut inputs = vec![[shader::ZERO; shader::INPUT_REGISTERS]; 24];
            for input in &mut inputs {
                input[0] = [random() * 2.6 - 1.3, random() * 2.6 - 1.3, random() * depth_range + near, 1.0];
                input[1] = [random(), random(), random(), random()];
                input[2] = [1.0, 0.0, 0.0, 0.0];
            }
            let order: Vec<usize> = (0..12).map(|_| (random() * 24.0) as usize).collect();
            let vertices = Vertices::Unshaded { vertex_shader: &unit, geometry_shader: &unit, inputs: &inputs, order: Some(&order) };

            let mut software = ConsoleMemory::default();
            let mut gpu = ConsoleMemory::default();
            for memory in [&mut software, &mut gpu] {
                memory.write(COLOR, &vec![0u8; (SIZE * SIZE * 4) as usize]);
                memory.write(DEPTH, &vec![0xFFu8; (SIZE * SIZE * 4) as usize]);
            }
            rasterize(&registers, &mut software, &mut Resources::default(), vertices);
            rasterize(&registers, &mut gpu, &mut resources, vertices);
            resources.hardware.as_mut().unwrap().flush(&mut gpu).unwrap();
            for i in 0..SIZE * SIZE {
                let (mut a, mut b, mut da, mut db) = ([0u8; 4], [0u8; 4], [0u8; 4], [0u8; 4]);
                software.read(COLOR + i * 4, &mut a);
                gpu.read(COLOR + i * 4, &mut b);
                software.read(DEPTH + i * 4, &mut da);
                gpu.read(DEPTH + i * 4, &mut db);
                drawn += (a != [0; 4]) as u32;
                if a.iter().zip(&b).any(|(a, b)| a.abs_diff(*b) > 3) {
                    differing += 1;
                } else if !clipped {
                    let depth = |d: [u8; 4]| u32::from_le_bytes(d) & 0xFF_FFFF;
                    deepest = deepest.max(depth(da).abs_diff(depth(db)));
                }
            }
        }
        eprintln!("{differing} of {drawn} drawn pixels differ, depth by up to {deepest}");
        assert!(drawn > 2000);
        assert!(differing * 100 < drawn, "{differing} of {drawn} pixels differ");
        assert!(deepest < 1 << 8, "depth differs by {deepest}");
    }

    /// guest memory with a stretch of it in one piece, from the linear
    /// heap's start, which slice hands out the way a title's vertex arrays
    /// lie, so draws from it go to the GPU as they are.
    #[cfg(feature = "vulkan")]
    struct ArrayMemory {
        rest: ConsoleMemory,
        arrays: Vec<u8>,
        /// how long each slice handed out was, a whole array's for the GPU
        /// among them, a vertex's when the CPU decodes them.
        sliced: Vec<usize>,
    }

    #[cfg(feature = "vulkan")]
    impl ArrayMemory {
        const START: u32 = 0x1400_0000;

        fn new() -> ArrayMemory {
            ArrayMemory { rest: ConsoleMemory::default(), arrays: vec![0; 0x1_0000], sliced: Vec::new() }
        }

        fn at(&self, addr: u32) -> Option<usize> {
            addr.checked_sub(Self::START).map(|at| at as usize).filter(|&at| at < self.arrays.len())
        }
    }

    #[cfg(feature = "vulkan")]
    impl GpuMemory for ArrayMemory {
        fn read(&mut self, addr: u32, out: &mut [u8]) {
            for (i, byte) in out.iter_mut().enumerate() {
                let addr = addr + i as u32;
                *byte = match self.at(addr) {
                    Some(at) => self.arrays[at],
                    None => {
                        let mut byte = [0];
                        self.rest.read(addr, &mut byte);
                        byte[0]
                    }
                };
            }
        }

        fn write(&mut self, addr: u32, data: &[u8]) {
            for (i, &byte) in data.iter().enumerate() {
                let addr = addr + i as u32;
                match self.at(addr) {
                    Some(at) => self.arrays[at] = byte,
                    None => self.rest.write(addr, &[byte]),
                }
            }
        }

        fn translate(&self, paddr: u32) -> u32 {
            self.rest.translate(paddr)
        }

        fn slice(&mut self, addr: u32, len: usize) -> Option<&[u8]> {
            let at = self.at(addr)?;
            let end = at.checked_add(len).filter(|&end| end <= self.arrays.len())?;
            self.sliced.push(len);
            Some(&self.arrays[at..end])
        }
    }

    /// draws from vertex arrays the GPU decodes itself come out as those
    /// whose vertices the CPU decodes, for each type of component, one to
    /// four of them, a fixed attribute, padding, one array or several,
    /// strides and offsets off a word, u8 and u16 indices and none, and
    /// lists, strips and fans.
    #[cfg(feature = "vulkan")]
    #[test]
    fn the_gpu_decodes_vertex_arrays_like_the_cpu() {
        for translates in [true, false] {
            let Ok(mut hardware) = hardware::Hardware::new() else { return };
            hardware.set_translates(translates, true);
            gpu_decodes_vertex_arrays_like_the_cpu(hardware);
        }
    }

    /// the arrays drawn through one GPU, translating programs or not.
    #[cfg(feature = "vulkan")]
    fn gpu_decodes_vertex_arrays_like_the_cpu(hardware: hardware::Hardware) {
        const SIZE: u32 = 32;
        // the arrays' base, physical FCRAM, at the linear heap's start
        const BASE: u32 = 0x2000_0000;
        const INDICES: u32 = 0x6000;
        const VERTICES: u32 = 24;
        const IDENTITY: u32 = 0x1B << 5 | 0x1B << 14 | 0x1B << 23;
        let mut unit = ShaderUnit::new();
        unit.descriptors[0] = 0xF | IDENTITY;
        let op = |opcode: u32, destination: u32, src1: u32, src2: u32| opcode << 26 | destination << 21 | src1 << 12 | src2 << 7;
        // o0 = c0 * v0, o1 = c1 * v1 + c2 + v2, v2 the fixed attribute
        let program = [op(0x08, 0x00, 0x20, 0x00), op(0x08, 0x10, 0x21, 0x01), op(0x00, 0x10, 0x22, 0x10), op(0x00, 0x01, 0x10, 0x02), 0x22 << 26];
        unit.program[..program.len()].copy_from_slice(&program);
        unit.prepare();
        let mut fixed = [shader::ZERO; 16];
        fixed[2] = [0.1, -0.05, 0.02, 0.0];

        let mut registers = target_registers();
        registers[REG_VIEWPORT_WIDTH] = float24(SIZE as f32 / 2.0);
        registers[REG_VIEWPORT_HEIGHT] = float24(SIZE as f32 / 2.0);
        registers[REG_FRAMEBUFFER_DIMENSIONS] = SIZE | ((SIZE - 1) << 12);
        registers[REG_SHADER_OUTPUT_TOTAL] = 2;
        registers[REG_SHADER_OUTPUT_MAP] = 0x0302_0100;
        registers[REG_SHADER_OUTPUT_MAP + 1] = 0x0B0A_0908;
        registers[REG_VS_OUTPUT_MASK] = 0b11;
        registers[REG_ATTRIBUTE_BASE] = BASE >> 3;
        // three attributes, the third fixed, each to the register of its
        // number
        registers[REG_VS_NUM_INPUT_ATTRIBUTES] = 2;
        registers[REG_VS_BLOCK + SHADER_INPUT_MAP_LOW] = 0x210;

        struct Case {
            /// each array's offset from the base, stride, and attribute ids
            /// or padding in order.
            loaders: Vec<(u32, u32, Vec<u32>)>,
            /// type and count of the position and of the color.
            position: (u32, u32),
            color: (u32, u32),
            /// u16 indices, u8 or none, and the first vertex used.
            indices: Option<bool>,
            first: u32,
        }
        let cases = [
            Case { loaders: vec![(0x100, 8, vec![0]), (0x800, 4, vec![1])], position: (3, 2), color: (1, 4), indices: Some(true), first: 0 },
            // a short at odd addresses, and a vertex longer than its stride
            Case { loaders: vec![(0x1001, 9, vec![0, 1, 12])], position: (2, 2), color: (0, 3), indices: Some(false), first: 0 },
            Case { loaders: vec![(0x2003, 3, vec![0]), (0x3002, 10, vec![1])], position: (0, 2), color: (2, 4), indices: None, first: 3 },
            // floats off a word, and one component of color
            Case { loaders: vec![(0x4001, 13, vec![0]), (0x5002, 6, vec![1])], position: (3, 2), color: (3, 1), indices: Some(true), first: 5 },
            // byte indices from well into the arrays, as a strip
            Case { loaders: vec![(0x1001, 9, vec![0, 1, 12])], position: (2, 2), color: (0, 3), indices: Some(false), first: 6 },
        ];
        let mut seed = 11u32;
        let mut random = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed >> 8
        };
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        for (number, case) in cases.iter().enumerate() {
            let nibble = |(ty, count): (u32, u32)| (count - 1) << 2 | ty;
            registers[REG_ATTRIBUTE_FORMAT_LOW] = nibble(case.position) | nibble(case.color) << 4;
            registers[REG_ATTRIBUTE_FORMAT_HIGH] = 1 << 18 | 2 << 28;
            for loader in 0..12 {
                let at = REG_ATTRIBUTE_LOADER + loader * 3;
                registers[at..at + 3].fill(0);
            }
            for (i, (offset, stride, entries)) in case.loaders.iter().enumerate() {
                let at = REG_ATTRIBUTE_LOADER + i * 3;
                let packed = entries.iter().enumerate().fold(0u64, |packed, (k, &id)| packed | (id as u64) << (k * 4));
                registers[at] = *offset;
                registers[at + 1] = packed as u32;
                registers[at + 2] = (packed >> 32) as u32 | stride << 16 | (entries.len() as u32) << 28;
            }
            registers[REG_PRIMITIVE_CONFIG] = (number as u32 % 3) << 8;
            registers[REG_VERTEX_COUNT] = VERTICES;
            registers[REG_VERTEX_OFFSET] = case.first;
            registers[REG_INDEX_ARRAY] = INDICES | if case.indices == Some(true) { 1 << 31 } else { 0 };
            // scales that bring each type to the screen and to colors
            let scale = |ty: u32| match ty {
                0 => 1.0 / 128.0,
                1 => 1.0 / 255.0,
                2 => 1.0 / 32768.0,
                _ => 1.0,
            };
            let ps = scale(case.position.0) * 0.9;
            unit.float_uniforms[0] = [ps, ps, 1.0, 1.0];
            let cs = scale(case.color.0) * if matches!(case.color.0, 0 | 2) { 0.5 } else { 1.0 };
            let offset = if matches!(case.color.0, 0 | 2) { 0.5 } else { 0.0 };
            unit.float_uniforms[1] = [cs; 4];
            unit.float_uniforms[2] = [offset; 4];

            // the arrays' bytes, floats kept to the screen and to colors,
            // and how long each array's vertex is
            let mut bytes = vec![0u8; 0x6000];
            let mut sizes = Vec::new();
            for (offset, stride, entries) in &case.loaders {
                let mut field = 0u32;
                for &id in entries {
                    if id >= 12 {
                        field = field.next_multiple_of(4) + (id - 11) * 4;
                        continue;
                    }
                    let (ty, count) = if id == 0 { case.position } else { case.color };
                    let size: u32 = [1, 1, 2, 4][ty as usize];
                    field = field.next_multiple_of(size);
                    for vertex in 0..case.first + VERTICES + 8 {
                        for component in 0..count {
                            let at = (offset + vertex * stride + field + component * size) as usize;
                            let value = random();
                            let value = if ty == 3 {
                                let float = (value & 0xFFFF) as f32 / 65536.0;
                                (if id == 0 { float * 1.8 - 0.9 } else { float }).to_bits()
                            } else {
                                value
                            };
                            bytes[at..at + size as usize].copy_from_slice(&value.to_le_bytes()[..size as usize]);
                        }
                    }
                    field += size * count;
                }
                sizes.push(field);
            }
            let indices: Vec<u32> = (0..VERTICES).map(|_| case.first + random() % 20).collect();
            let index_bytes: Vec<u8> = match case.indices {
                Some(true) => indices.iter().flat_map(|&index| (index as u16).to_le_bytes()).collect(),
                _ => indices.iter().map(|&index| index as u8).collect(),
            };

            let mut decoded = ConsoleMemory::default();
            let mut raw = ArrayMemory::new();
            decoded.write(0x1400_0000, &bytes);
            decoded.write(0x1400_0000 + INDICES, &index_bytes);
            raw.write(0x1400_0000, &bytes);
            raw.write(0x1400_0000 + INDICES, &index_bytes);
            decoded.write(COLOR, &vec![0u8; (SIZE * SIZE * 4) as usize]);
            raw.write(COLOR, &vec![0u8; (SIZE * SIZE * 4) as usize]);
            let indexed = case.indices.is_some();
            draw(&registers, &unit, &unit, &fixed, &mut decoded, &mut resources, indexed);
            resources.hardware.as_mut().unwrap().flush(&mut decoded).unwrap();
            draw(&registers, &unit, &unit, &fixed, &mut raw, &mut resources, indexed);
            resources.hardware.as_mut().unwrap().flush(&mut raw).unwrap();
            // the first array whole, from the first vertex used to the last
            let span = match indexed {
                true => indices.iter().max().unwrap() - indices.iter().min().unwrap() + 1,
                false => VERTICES,
            };
            let whole = ((span - 1) * case.loaders[0].1 + sizes[0]) as usize;
            assert!(raw.sliced.contains(&whole), "case {number}, the arrays went to the GPU as they are");
            let (mut a, mut b) = (vec![0u8; (SIZE * SIZE * 4) as usize], vec![0u8; (SIZE * SIZE * 4) as usize]);
            decoded.read(COLOR, &mut a);
            raw.read(COLOR, &mut b);
            let drawn = a.chunks(4).filter(|pixel| *pixel != [0; 4]).count();
            assert!(drawn > 50, "case {number}, only {drawn} pixels drawn");
            assert!(a == b, "case {number}, the arrays decode differently");
        }
    }

    /// a random number generator for the translation tests, the same
    /// numbers every run.
    #[cfg(feature = "vulkan")]
    struct Random(u64);

    #[cfg(feature = "vulkan")]
    impl Random {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }

        /// an arithmetic instruction of any kind, writing a temporary or
        /// the color, o2.
        fn arithmetic(&mut self) -> u32 {
            let (wide, narrow, index, descriptor) = (self.next() % 0x80, self.next() % 0x20, self.next() % 4, 1 + self.next() % 127);
            let destination = if self.next().is_multiple_of(4) { 2 } else { 0x10 + self.next() % 16 };
            let ops = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x12, 0x13];
            match self.next() % 12 {
                0..=7 => ops[(self.next() % ops.len() as u32) as usize] << 26 | destination << 21 | index << 19 | wide << 12 | narrow << 7 | descriptor,
                8 => [0x18, 0x19, 0x1A, 0x1B][(self.next() % 4) as usize] << 26 | destination << 21 | index << 19 | narrow << 14 | wide << 7 | descriptor,
                9 => 0x2E << 26 | (self.next() % 8) << 24 | (self.next() % 8) << 21 | index << 19 | wide << 12 | narrow << 7 | descriptor,
                10 => 0b111 << 29 | destination << 24 | index << 22 | (self.next() % 0x20) << 17 | wide << 10 | narrow << 5 | descriptor & 0x1F,
                _ => 0b110 << 29 | destination << 24 | index << 22 | (self.next() % 0x20) << 17 | narrow << 12 | wide << 5 | descriptor & 0x1F,
            }
        }

        /// a condition of any kind for a flow instruction, in its bits.
        fn condition(&mut self) -> u32 {
            (self.next() % 16) << 22
        }
    }

    /// the side of the square the translation tests draw into.
    #[cfg(feature = "vulkan")]
    const TRANSLATION_SIZE: u32 = 32;

    /// what the translation tests draw with, the position from o0 and the
    /// color from o2.
    #[cfg(feature = "vulkan")]
    fn translation_registers() -> Vec<u32> {
        let mut registers = target_registers();
        registers[REG_VIEWPORT_XY] = 4 | 6 << 16;
        registers[REG_VIEWPORT_WIDTH] = float24(12.0);
        registers[REG_VIEWPORT_HEIGHT] = float24(12.0);
        registers[REG_FRAMEBUFFER_DIMENSIONS] = TRANSLATION_SIZE | ((TRANSLATION_SIZE - 1) << 12);
        registers[REG_DEPTH_COLOR_MASK] = 0xF << 8 | 1 | 4 << 4 | 1 << 12;
        registers[REG_VIEWPORT_DEPTH_RANGE] = float24(-1.0);
        registers[REG_DEPTHMAP_ENABLE] = 1;
        registers[REG_SHADER_OUTPUT_TOTAL] = 2;
        registers[REG_SHADER_OUTPUT_MAP] = 0x0302_0100;
        registers[REG_SHADER_OUTPUT_MAP + 1] = 0x0B0A_0908;
        registers[REG_VS_OUTPUT_MASK] = 0b101;
        registers[REG_PRIMITIVE_CONFIG] = 0;
        registers
    }

    /// a unit running the program made some random words in, its flow moved
    /// along with it so the entry point matters, with random descriptors and
    /// uniforms, prepared, and all its words.
    #[cfg(feature = "vulkan")]
    fn random_unit(random: &mut Random, program: impl FnOnce(&mut Random) -> Vec<u32>) -> (ShaderUnit, Vec<u32>) {
        const IDENTITY: u32 = 0x1B << 5 | 0x1B << 14 | 0x1B << 23;
        let mut unit = ShaderUnit::new();
        unit.descriptors[0] = 0xF | IDENTITY;
        for descriptor in &mut unit.descriptors[1..] {
            *descriptor = random.next() & 0x7FFF_FFFF;
        }
        for uniform in unit.float_uniforms.iter_mut() {
            *uniform = std::array::from_fn(|_| match random.next() % 30 {
                0 => f32::INFINITY,
                1 => -0.0,
                _ => (random.next() % 2000) as f32 / 1000.0 - 1.0,
            });
        }
        for integer in &mut unit.int_uniforms {
            *integer = [(random.next() % 3) as u8, (random.next() % 8) as u8, (random.next() % 3) as u8, 0];
        }
        unit.bool_uniforms = random.next() as u16;
        // past instructions never run
        let skip = random.next() % 8;
        let mut words: Vec<u32> = (0..skip).map(|_| random.arithmetic()).collect();
        words.extend(program(random).into_iter().map(|word| match word >> 26 {
            0x24..=0x29 | 0x2C | 0x2D => word + (skip << 10),
            _ => word,
        }));
        unit.program[..words.len()].copy_from_slice(&words);
        unit.entry_point = skip;
        unit.prepare();
        (unit, words)
    }

    /// random inputs for a translation test's draw.
    #[cfg(feature = "vulkan")]
    fn random_inputs(random: &mut Random) -> Vec<[shader::Vec4; shader::INPUT_REGISTERS]> {
        let mut inputs = vec![[shader::ZERO; shader::INPUT_REGISTERS]; 24];
        for input in &mut inputs {
            let mut float = || (random.next() % 2000) as f32 / 1000.0 - 1.0;
            input[0] = [float() * 1.3, float() * 1.3, float() * 0.6 - 0.1, 1.0];
            for register in &mut input[1..] {
                *register = [float(), float(), float(), float()];
            }
        }
        inputs
    }

    /// the color and then the depth a draw leaves on the GPU.
    #[cfg(feature = "vulkan")]
    fn draw_on_gpu(registers: &[u32], resources: &mut Resources, unit: &ShaderUnit, inputs: &[[shader::Vec4; shader::INPUT_REGISTERS]]) -> Vec<u8> {
        let bytes = (TRANSLATION_SIZE * TRANSLATION_SIZE * 4) as usize;
        let mut memory = ConsoleMemory::default();
        memory.write(COLOR, &vec![0u8; bytes]);
        memory.write(DEPTH, &vec![0xFFu8; bytes]);
        let vertices = Vertices::Unshaded { vertex_shader: unit, geometry_shader: unit, inputs, order: None };
        rasterize(registers, &mut memory, resources, vertices);
        resources.hardware.as_mut().unwrap().flush(&mut memory).unwrap();
        let mut out = vec![0u8; bytes * 2];
        memory.read(COLOR, &mut out[..bytes]);
        memory.read(DEPTH, &mut out[bytes..]);
        out
    }

    /// draws programs made by program with the GPU interpreting them and
    /// with them translated, and checks color and depth come out the same
    /// to the bit, at least so many programs translated, and any other one
    /// only because it can run on past its end.
    #[cfg(feature = "vulkan")]
    fn draws_like_the_interpreter(cases: u32, least: u32, seed: u64, program: impl Fn(&mut Random) -> Vec<u32>) {
        let (Ok(mut translating), Ok(mut interpreting)) = (hardware::Hardware::new(), hardware::Hardware::new()) else {
            return;
        };
        translating.set_translates(true, true);
        interpreting.set_translates(false, true);
        let mut random = Random(seed);
        let registers = translation_registers();
        let mut translators = Resources { hardware: Some(translating), ..Default::default() };
        let mut interpreters = Resources { hardware: Some(interpreting), ..Default::default() };
        let mut drawn = 0;
        for case in 0..cases {
            let (unit, program) = random_unit(&mut random, &program);
            // the only program left to the interpreter is one that can run
            // on into the words after it
            if let Err(error) = unit.translate(&output_semantics(&registers)) {
                assert!(error.contains("reachable instructions"), "case {case} not translated, {error}, program {program:08X?}");
            }
            let inputs = random_inputs(&mut random);
            let translated = draw_on_gpu(&registers, &mut translators, &unit, &inputs);
            let interpreted = draw_on_gpu(&registers, &mut interpreters, &unit, &inputs);
            assert!(translated == interpreted, "case {case} draws differently translated, program {program:08X?}");
            let bytes = (TRANSLATION_SIZE * TRANSLATION_SIZE * 4) as usize;
            drawn += translated[..bytes].chunks(4).filter(|pixel| *pixel != [0; 4]).count();
        }
        let translations = translators.hardware.as_ref().unwrap().translations();
        eprintln!("{cases} programs, {translations} translated, {drawn} pixels drawn");
        assert!(drawn as u32 > cases * 40, "only {drawn} pixels drawn");
        assert!(translations as u32 >= least, "only {translations} of {cases} programs translated");
        assert_eq!(interpreters.hardware.as_ref().unwrap().translations(), 0);
    }

    /// 2D drawn on the near or far plane is drawn when z misses the plane
    /// only by rounding, as it would on the console, and clipped when it is
    /// really outside, the same on the CPU, on the GPU interpreting the
    /// program and with it translated.
    #[cfg(feature = "vulkan")]
    #[test]
    fn geometry_on_the_ends_of_the_z_range_is_drawn() {
        let mut registers = translation_registers();
        // color only, nothing but the clipping keeps a pixel out
        registers[REG_DEPTH_COLOR_MASK] = 0xF << 8;
        const IDENTITY: u32 = 0x1B << 5 | 0x1B << 14 | 0x1B << 23;
        let mut unit = ShaderUnit::new();
        unit.descriptors[0] = 0xF | IDENTITY;
        let mov = |destination: u32, source: u32| 0x13 << 26 | destination << 21 | source << 12;
        unit.program[..3].copy_from_slice(&[mov(0, 0), mov(2, 1), 0x22 << 26]);
        unit.prepare();
        let hardware = |translates: bool| {
            hardware::Hardware::new().ok().map(|mut hardware| {
                hardware.set_translates(translates, true);
                Resources { hardware: Some(hardware), ..Default::default() }
            })
        };
        let (mut interpreting, mut translating) = (hardware(false), hardware(true));
        let bytes = (TRANSLATION_SIZE * TRANSLATION_SIZE * 4) as usize;
        for (z, drawn) in [(-0.5, true), (-1.000005, true), (5e-9, true), (-1.001, false), (1e-6, false)] {
            let inputs: Vec<[shader::Vec4; shader::INPUT_REGISTERS]> = [[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]]
                .into_iter()
                .map(|[x, y]| {
                    let mut input = [shader::ZERO; shader::INPUT_REGISTERS];
                    input[0] = [x, y, z, 1.0];
                    input[1] = [1.0, 0.0, 0.0, 1.0];
                    input
                })
                .collect();
            let mut memory = ConsoleMemory::default();
            memory.write(COLOR, &vec![0u8; bytes]);
            let vertices = Vertices::Unshaded { vertex_shader: &unit, geometry_shader: &unit, inputs: &inputs, order: None };
            rasterize(&registers, &mut memory, &mut Resources::default(), vertices);
            let mut software = vec![0u8; bytes];
            memory.read(COLOR, &mut software);
            let red = software.chunks(4).filter(|pixel| ColorFormat::Rgba8.decode(pixel) == [255, 0, 0, 255]).count();
            assert_eq!(red > 0, drawn, "z {z} on the CPU, {red} pixels");
            for resources in [&mut interpreting, &mut translating].into_iter().flatten() {
                let gpu = draw_on_gpu(&registers, resources, &unit, &inputs);
                assert!(gpu[..bytes] == software[..], "z {z} on the GPU draws what the CPU does");
            }
        }
    }

    /// a program is interpreted while the compiler thread translates it and
    /// compiles its pipeline, then runs translated, drawing the same to the
    /// bit throughout, and the compiler stops with the GPU while it works.
    #[cfg(feature = "vulkan")]
    #[test]
    fn programs_translate_while_drawing_goes_on() {
        let (Ok(mut translating), Ok(mut interpreting)) = (hardware::Hardware::new(), hardware::Hardware::new()) else {
            return;
        };
        translating.set_translates(true, false);
        interpreting.set_translates(false, true);
        let mut random = Random(24);
        let registers = translation_registers();
        let mut translators = Resources { hardware: Some(translating), ..Default::default() };
        let mut interpreters = Resources { hardware: Some(interpreting), ..Default::default() };
        // a loop around a call to after the end, then an if
        let (unit, _) = random_unit(&mut random, |random| {
            let mut arithmetic = || random.arithmetic();
            vec![
                0x13 << 26,
                0x29 << 26 | 4 << 10,
                0x24 << 26 | 8 << 10 | 2,
                arithmetic(),
                arithmetic(),
                0x27 << 26 | 1 << 22 | 7 << 10,
                arithmetic(),
                0x22 << 26,
                arithmetic(),
                arithmetic(),
            ]
        });
        let inputs = random_inputs(&mut random);
        let started = std::time::Instant::now();
        let mut draws = 0;
        loop {
            // a pipeline made before the draw is the one it uses
            let translated = translators.hardware.as_ref().unwrap().translated_pipelines() > 0;
            assert!(draws > 0 || !translated, "the first draw waited for its program to be translated");
            let a = draw_on_gpu(&registers, &mut translators, &unit, &inputs);
            let b = draw_on_gpu(&registers, &mut interpreters, &unit, &inputs);
            assert!(a == b, "draw {draws} differs, translated {translated}");
            draws += 1;
            if translated {
                break;
            }
            assert!(started.elapsed().as_secs() < 60, "the program was never translated");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        eprintln!("translated after {draws} draws, {:?}", started.elapsed());
        // and one more the compiler is still busy with when everything goes
        let (unit, _) = random_unit(&mut random, |random| {
            std::iter::once(0x13 << 26).chain((0..60).map(|_| random.arithmetic())).chain([0x22 << 26]).collect()
        });
        draw_on_gpu(&registers, &mut translators, &unit, &inputs);
    }

    /// random fragment stages, every combiner source, operand, operation
    /// and scale, buffer updates, the three texture units over the texels at
    /// textures, procedural textures, lighting, the alpha test and both depth
    /// modes.
    #[cfg(feature = "vulkan")]
    fn random_fragment_stages(random: &mut Random, registers: &mut [u32], textures: u32) {
        for base in [0x0C0, 0x0C8, 0x0D0, 0x0D8, 0x0F0, 0x0F8] {
            registers[base] = random.next() & 0x0FFF_0FFF;
            registers[base + 1] = random.next() & 0x0077_7FFF;
            registers[base + 2] = (random.next() % 10) | ((random.next() % 10) << 16);
            registers[base + 3] = random.next();
            registers[base + 4] = random.next() & 0x0003_0003;
        }
        // the buffer updates and the buffer's first color
        registers[0x0E0] = random.next() & 0xFF00;
        registers[0x0FD] = random.next();
        // any of the units, unit 2 reading either coordinates, now and then a
        // procedural texture
        registers[REG_TEXTURE_CONFIG] = random.next() & 0x2307 | (random.next().is_multiple_of(4) as u32) << 10;
        for (unit, base) in TEXTURE_UNIT_BASES.into_iter().enumerate() {
            registers[base] = random.next();
            registers[base + 1] = 8 | 8 << 16;
            // the filters and the wrap modes
            registers[base + 2] = (random.next() % 4) << 1 | (random.next() % 4) << 8 | (random.next() % 4) << 12;
            registers[base + 4] = (textures + unit as u32 * 0x100) >> 3;
            registers[if unit == 0 { base + 13 } else { base + 5 }] = 0;
        }
        registers[crate::lighting::REG_ENABLE] = random.next().is_multiple_of(3) as u32;
        // on or off, a function and a reference
        registers[REG_ALPHA_TEST] = random.next() & 0xFF71;
        registers[REG_DEPTHMAP_ENABLE] = random.next() & 1;
    }

    /// the generic fragment shader, which draws while the one made for a
    /// combination compiles, draws what that one draws, to the bit, over
    /// random fragment stages, with the triangles placed by the CPU and with
    /// the vertices shaded on the GPU.
    #[cfg(feature = "vulkan")]
    #[test]
    fn the_generic_fragment_shader_draws_like_the_specialized_ones() {
        let (Ok(mut generic), Ok(specialized)) = (hardware::Hardware::new(), hardware::Hardware::new()) else {
            return;
        };
        generic.set_generic(true);
        const TEXTURES: u32 = 0x8_0000;
        let mut random = Random(25);
        let mut generics = Resources { hardware: Some(generic), ..Default::default() };
        let mut specializeds = Resources { hardware: Some(specialized), ..Default::default() };
        let bytes = (TRANSLATION_SIZE * TRANSLATION_SIZE * 4) as usize;
        let (mut drawn, mut differing) = (0, 0);
        for case in 0..180 {
            let mut registers = translation_registers();
            random_fragment_stages(&mut random, &mut registers, TEXTURES);
            let texels: Vec<u8> = (0..0x300).map(|_| random.next() as u8).collect();
            let mut results = Vec::new();
            if case < 120 {
                let triangles: Vec<Vec<Vertex>> = (0..4)
                    .map(|_| {
                        (0..3)
                            .map(|_| {
                                let mut float = || (random.next() % 2000) as f32 / 1000.0 - 1.0;
                                let w = 1.5 + float();
                                Vertex {
                                    clip: [float() * w, float() * w, (float() * 0.5 - 0.5) * w, w],
                                    color: [float().abs(), float().abs(), float().abs(), float().abs()],
                                    texcoords: [[float() * 2.0, float() * 2.0], [float() * 2.0, float() * 2.0], [float() * 2.0, float() * 2.0]],
                                    quaternion: [float(), float(), float(), float()],
                                    view: [float(), float(), float()],
                                }
                            })
                            .collect()
                    })
                    .collect();
                for resources in [&mut generics, &mut specializeds] {
                    let mut memory = ConsoleMemory::default();
                    memory.write(COLOR, &vec![0u8; bytes]);
                    memory.write(DEPTH, &vec![0xFFu8; bytes]);
                    memory.write(TEXTURES, &texels);
                    for triangle in &triangles {
                        rasterize_shaded(&registers, &mut memory, resources, triangle);
                    }
                    resources.hardware.as_mut().unwrap().flush(&mut memory).unwrap();
                    let mut out = vec![0u8; bytes * 2];
                    memory.read(COLOR, &mut out[..bytes]);
                    memory.read(DEPTH, &mut out[bytes..]);
                    results.push(out);
                }
            } else {
                // all that lighting and the units read, each output register
                // moved from the input register of its number, the position,
                // the quaternion, the color, the view and the coordinates
                registers[REG_SHADER_OUTPUT_TOTAL] = 6;
                let map = [0x0302_0100, 0x0706_0504, 0x0B0A_0908, 0x1F14_1312, 0x0F0E_0D0C, 0x1F1F_1716];
                registers[REG_SHADER_OUTPUT_MAP..REG_SHADER_OUTPUT_MAP + 6].copy_from_slice(&map);
                registers[REG_VS_OUTPUT_MASK] = 0b11_1111;
                let moves = (0..6).map(|register| 0x13 << 26 | register << 21 | register << 12);
                let (unit, _) = random_unit(&mut random, |_| moves.chain([0x22 << 26]).collect());
                let inputs = random_inputs(&mut random);
                for resources in [&mut generics, &mut specializeds] {
                    let vertices = Vertices::Unshaded { vertex_shader: &unit, geometry_shader: &unit, inputs: &inputs, order: None };
                    let mut memory = ConsoleMemory::default();
                    memory.write(COLOR, &vec![0u8; bytes]);
                    memory.write(DEPTH, &vec![0xFFu8; bytes]);
                    memory.write(TEXTURES, &texels);
                    rasterize(&registers, &mut memory, resources, vertices);
                    resources.hardware.as_mut().unwrap().flush(&mut memory).unwrap();
                    let mut out = vec![0u8; bytes * 2];
                    memory.read(COLOR, &mut out[..bytes]);
                    memory.read(DEPTH, &mut out[bytes..]);
                    results.push(out);
                }
            }
            // the coordinates the units and the procedural texture read are
            // interpolated as the driver sees fit for each shader, which now
            // and then puts a texel's weight a hair apart and a channel a
            // step off. everything else comes out the same to the bit, the
            // combiners, lighting, the tests and depth
            let coordinates = registers[REG_TEXTURE_CONFIG] & 0x407 != 0;
            let stages: Vec<u32> = [0x0C0, 0x0C1, 0x0C2, 0x0C4, 0x0E0, REG_TEXTURE_CONFIG, REG_ALPHA_TEST].map(|r| registers[r]).to_vec();
            for (i, (a, b)) in results[0].chunks(4).zip(results[1].chunks(4)).enumerate() {
                let apart = a.iter().zip(b).map(|(a, b)| a.abs_diff(*b)).max().unwrap_or(0);
                let allowed = if coordinates && i < bytes / 4 { 1 } else { 0 };
                assert!(apart <= allowed, "case {case}, pixel {i} is {a:?} through the generic shader, {b:?} through its own, {stages:08X?}");
                differing += (apart > 0) as usize;
            }
            drawn += results[0][..bytes].chunks(4).filter(|pixel| *pixel != [0; 4]).count();
        }
        eprintln!("{drawn} pixels drawn, {differing} a step off");
        assert!(drawn > 180 * 40, "only {drawn} pixels drawn");
        assert!(differing * 100 < drawn, "{differing} of {drawn} pixels a step off");
    }

    /// programs of every instruction, with flow of every kind going
    /// forward, draw the same translated as interpreted, to the bit.
    #[cfg(feature = "vulkan")]
    #[test]
    fn translated_shaders_draw_like_the_interpreter() {
        draws_like_the_interpreter(200, 200, 21, |random| {
            // the position as it comes, then instructions of every kind,
            // flow only going forward so every program ends, and blocks
            // ending within the program, past it are words of zero that
            // run to the end of the program memory
            let length = 4 + random.next() % 28;
            let mut program = vec![0x13 << 26];
            for at in 1..length {
                let forward = at + 1 + random.next() % (length - at);
                let (count, condition) = ((random.next() % 4).min(length - forward), random.condition());
                program.push(match random.next() % 16 {
                    0..=11 => random.arithmetic(),
                    12 => [0x24, 0x25, 0x26, 0x27, 0x28][(random.next() % 5) as usize] << 26 | condition | forward << 10 | count,
                    13 => [0x2C, 0x2D][(random.next() % 2) as usize] << 26 | condition | forward << 10 | count,
                    14 => 0x29 << 26 | (random.next() % 4) << 22 | (at + 1 + random.next() % 3).min(length - 1) << 10,
                    _ => [0x20 << 26, 0x23 << 26 | condition][(random.next() % 2) as usize],
                });
            }
            program.push(0x22 << 26);
            program
        });
    }

    /// programs that run on until the interpreter gives up on them, a body
    /// writing the color repeated by a jump back, stop at the same
    /// instruction translated, however long the body, so the color comes
    /// out the same to the bit.
    #[cfg(feature = "vulkan")]
    #[test]
    fn programs_that_never_end_draw_like_the_interpreter() {
        draws_like_the_interpreter(60, 60, 23, |random| {
            let body = 1 + random.next() % 9;
            let mut program = vec![0x13 << 26];
            program.extend((0..body).map(|_| random.arithmetic()));
            // back to the body's start when its boolean says so, which half
            // the time is always
            program.push((0x2D << 26) | ((random.next() % 16) << 22) | (1 << 10) | (random.next() % 2));
            program.push(0x22 << 26);
            program
        });
    }

    /// programs laid out the way the SDK's compiler lays them out, a main
    /// part calling subroutines after its end, each running into the next
    /// one where its call ends it, with ifs, loops, breaks and calls of
    /// their own, draw the same translated as interpreted, to the bit. a
    /// jump out of a loop leaves the loop's block over the call's, which
    /// then never ends, and execution runs on to the end of the program
    /// memory, so some of them stay interpreted, as they should.
    #[cfg(feature = "vulkan")]
    #[test]
    fn subroutines_draw_like_the_interpreter() {
        draws_like_the_interpreter(150, 100, 22, |random| {
            let routines = 1 + random.next() % 5;
            let main = 3 + random.next() % 8;
            let lengths: Vec<u32> = (0..routines).map(|_| 2 + random.next() % 10).collect();
            let starts: Vec<u32> = lengths
                .iter()
                .scan(main + 1, |at, &length| {
                    let start = *at;
                    *at += length;
                    Some(start)
                })
                .collect();
            let call = |random: &mut Random, after: u32| -> Option<u32> {
                // only routines further on, so nothing calls itself
                let first = starts.iter().position(|&start| start > after)?;
                let routine = first + (random.next() as usize % (starts.len() - first));
                let opcode = [0x24, 0x25, 0x26][(random.next() % 3) as usize];
                Some(opcode << 26 | random.condition() | starts[routine] << 10 | lengths[routine])
            };
            // the position as it comes, then the main part
            let mut program = vec![0x13 << 26];
            for at in 1..main {
                program.push(match random.next() % 4 {
                    0 => call(random, at).unwrap_or_else(|| random.arithmetic()),
                    _ => random.arithmetic(),
                });
            }
            program.push(0x22 << 26);
            // the subroutines, ifs and jumps forward within themselves, and
            // breaks within their loops, a break with no loop open would
            // close the call too and run on into the next subroutine
            for (&start, &length) in starts.iter().zip(&lengths) {
                let end = start + length;
                let mut looped = 0;
                for at in start..end {
                    let forward = at + 1 + random.next() % (end - at);
                    let count = (random.next() % 3).min(end - forward);
                    let word = match random.next() % 10 {
                        0 if at + 1 < end => {
                            let opcode = [0x27, 0x28][(random.next() % 2) as usize];
                            opcode << 26 | random.condition() | forward << 10 | count
                        }
                        1 if at + 1 < end => [0x2C, 0x2D][(random.next() % 2) as usize] << 26 | random.condition() | forward << 10,
                        2 => call(random, end - 1).unwrap_or_else(|| random.arithmetic()),
                        3 if at + 2 < end && at >= looped => {
                            let last = (at + 1 + random.next() % (end - at - 1)).min(end - 1);
                            looped = last + 1;
                            0x29 << 26 | (random.next() % 4) << 22 | last << 10
                        }
                        4 if at < looped => 0x23 << 26 | random.condition(),
                        _ => random.arithmetic(),
                    };
                    program.push(word);
                }
            }
            program
        });
    }

    /// a texture that is rows of a buffer the GPU just drew samples the
    /// same from a copy the GPU makes as from memory once it has the buffer,
    /// whether the buffer holds colors or depth and stencil.
    #[cfg(feature = "vulkan")]
    #[test]
    fn textures_drawn_on_the_gpu_sample_like_memory() {
        for depth in [false, true] {
            textures_drawn_on_the_gpu_sample_like_memory_from(depth);
        }
    }

    #[cfg(feature = "vulkan")]
    fn textures_drawn_on_the_gpu_sample_like_memory_from(depth: bool) {
        let (Ok(first), Ok(second)) = (hardware::Hardware::new(), hardware::Hardware::new()) else { return };
        const SIZE: u32 = 32;
        const TARGET: u32 = 0x10_0000;
        let mut drawing = target_registers();
        drawing[REG_VIEWPORT_WIDTH] = float24(SIZE as f32 / 2.0);
        drawing[REG_VIEWPORT_HEIGHT] = float24(SIZE as f32 / 2.0);
        drawing[REG_FRAMEBUFFER_DIMENSIONS] = SIZE | ((SIZE - 1) << 12);
        if depth {
            // depth written always, the stencil as it comes, the depth
            // being -z as titles map it
            drawing[REG_DEPTH_COLOR_MASK] |= 1 << 12;
            drawing[REG_VIEWPORT_DEPTH_RANGE] = float24(-1.0);
            drawing[REG_DEPTHMAP_ENABLE] = 1;
        }
        // the second draw fills a buffer half as tall with rows 8 to 23 of
        // the first, or of its depth buffer read as colors, as its texture,
        // straight from the combiners
        let mut sampling = target_registers();
        sampling[REG_COLOR_BUFFER_ADDRESS] = TARGET >> 3;
        sampling[REG_VIEWPORT_WIDTH] = float24(SIZE as f32 / 2.0);
        sampling[REG_VIEWPORT_HEIGHT] = float24(SIZE as f32 / 4.0);
        sampling[REG_FRAMEBUFFER_DIMENSIONS] = SIZE | ((SIZE / 2 - 1) << 12);
        sampling[REG_TEXTURE_CONFIG] = 1;
        sampling[REG_TEXTURE0_DIMENSIONS] = (SIZE / 2) | (SIZE << 16);
        let source = if depth { DEPTH } else { COLOR };
        sampling[REG_TEXTURE0_ADDRESS] = (source + 8 * SIZE * 4) >> 3;
        sampling[0x0C0] = 0x3 | (0x3 << 16);
        for stage in [0x0C8, 0x0D0, 0x0D8, 0x0F0, 0x0F8] {
            sampling[stage] = 0xF | (0xF << 16);
        }

        let mut seed = 5u32;
        let mut random = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1 << 24) as f32
        };
        let triangles: Vec<Vec<Vertex>> = (0..30)
            .map(|_| {
                let color = [random(), random(), random(), random()];
                (0..3)
                    .map(|_| Vertex {
                        clip: [random() * 2.0 - 1.0, random() * 2.0 - 1.0, -random(), 1.0],
                        color,
                        texcoords: [[0.0; 2]; 3],
                        quaternion: [0.0, 0.0, 0.0, 1.0],
                        view: [0.0; 3],
                    })
                    .collect()
            })
            .collect();
        // a triangle over the whole target, the texture spread across it
        let quad: Vec<Vertex> = [[-1.0, -1.0, 0.0, 0.0], [3.0, -1.0, 2.0, 0.0], [-1.0, 3.0, 0.0, 2.0]]
            .into_iter()
            .map(|[x, y, u, v]| Vertex {
                clip: [x, y, -0.5, 1.0],
                color: [1.0; 4],
                texcoords: [[u, v], [0.0; 2], [0.0; 2]],
                quaternion: [0.0, 0.0, 0.0, 1.0],
                view: [0.0; 3],
            })
            .collect();

        // once with the texture copied on the GPU, once with memory given
        // the drawing back first
        let mut results = Vec::new();
        for (hardware, back_first) in [(first, false), (second, true)] {
            let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
            let mut memory = ConsoleMemory::default();
            for triangle in &triangles {
                rasterize_shaded(&drawing, &mut memory, &mut resources, triangle);
            }
            if back_first {
                resources.hardware.as_mut().unwrap().flush(&mut memory).unwrap();
            }
            rasterize_shaded(&sampling, &mut memory, &mut resources, &quad);
            resources.hardware.as_mut().unwrap().flush(&mut memory).unwrap();
            let mut sampled = vec![0u8; (SIZE * SIZE / 2 * 4) as usize];
            memory.read(TARGET, &mut sampled);
            results.push(sampled);
        }
        assert!(results[0] == results[1], "reading {}", if depth { "depth" } else { "colors" });
        let colors: std::collections::HashSet<&[u8]> = results[0].chunks(4).collect();
        assert!(colors.len() > 8, "only {} colors", colors.len());
    }

    /// a display transfer out of a buffer drawn on the GPU leaves the same
    /// bytes as the CPU's, whatever the formats, flip, downscale and layout,
    /// and from rows into the buffer as well as its start.
    #[cfg(feature = "vulkan")]
    #[test]
    fn transfers_on_the_gpu_match_the_cpu() {
        use crate::format::ColorFormat::*;
        let Ok(hardware) = hardware::Hardware::new() else { return };
        const SIZE: u32 = 32;
        const OUTPUT: u32 = 0x10_0000;
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        let mut seed = 7u32;
        let mut random = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1 << 24) as f32
        };
        for input_format in [Rgba8, Rgb565] {
            let mut registers = target_registers();
            registers[REG_VIEWPORT_WIDTH] = float24(SIZE as f32 / 2.0);
            registers[REG_VIEWPORT_HEIGHT] = float24(SIZE as f32 / 2.0);
            registers[REG_FRAMEBUFFER_DIMENSIONS] = SIZE | ((SIZE - 1) << 12);
            registers[REG_COLOR_BUFFER_FORMAT] = input_format.color_buffer_raw() << 16;
            // triangles in all sorts of colors over each other
            let mut drawn = ConsoleMemory::default();
            for _ in 0..40 {
                let color = [random(), random(), random(), random()];
                let triangle: Vec<Vertex> = (0..3)
                    .map(|_| Vertex {
                        clip: [random() * 2.0 - 1.0, random() * 2.0 - 1.0, -0.5, 1.0],
                        color,
                        texcoords: [[0.0; 2]; 3],
                        quaternion: [0.0, 0.0, 0.0, 1.0],
                        view: [0.0; 3],
                    })
                    .collect();
                rasterize_shaded(&registers, &mut drawn, &mut resources, &triangle);
            }
            // memory gets what the GPU drew, which stays on the GPU too
            resources.hardware.as_mut().unwrap().flush(&mut drawn).unwrap();
            for output_format in [Rgba8, Rgb8, Rgb565, Rgb5A1, Rgba4] {
                let cases = [(false, 0, false, 0), (true, 0, true, 0), (false, 1, false, 0), (true, 2, false, 0), (false, 0, false, 1), (true, 2, false, 2)];
                for (flip, downscale, tiled, tile_rows) in cases {
                    let hardware = resources.hardware.as_mut().unwrap();
                    let (mut gpu, mut cpu) = (drawn.clone(), drawn.clone());
                    let (scale_x, scale_y) = [(1, 1), (2, 1), (2, 2)][downscale];
                    let input = COLOR + tile_rows * 8 * SIZE * input_format.bytes_per_pixel() as u32;
                    let input_height = SIZE - tile_rows * 8;
                    let (width, height) = (SIZE / scale_x, input_height / scale_y);
                    let transfer = hardware::Transfer {
                        input,
                        output: OUTPUT,
                        input_width: SIZE,
                        input_height,
                        output_width: width,
                        output_height: height,
                        copy: (width, height),
                        scale: (scale_x, scale_y),
                        flip,
                        input_linear: false,
                        output_tiled: tiled,
                        input_format,
                        output_format,
                    };
                    assert!(hardware.display_transfer(&mut gpu, &transfer).unwrap());
                    hardware.flush(&mut gpu).unwrap();
                    let flags = flip as u32
                        | (tiled as u32) << 5
                        | (hardware::format_index(input_format) as u32) << 8
                        | (hardware::format_index(output_format) as u32) << 12
                        | (downscale as u32) << 24;
                    // the output's size in the register counts input pixels
                    let output_size = (width * scale_x) | ((height * scale_y) << 16);
                    let input_size = SIZE | input_height << 16;
                    crate::Gpu::new().display_transfer(&mut cpu, input, OUTPUT, input_size, output_size, flags);
                    let len = (width * height) as usize * output_format.bytes_per_pixel();
                    let (mut a, mut b) = (vec![0u8; len], vec![0u8; len]);
                    cpu.read(OUTPUT, &mut a);
                    gpu.read(OUTPUT, &mut b);
                    assert!(a == b, "{input_format:?} to {output_format:?}, flip {flip}, downscale {downscale}, rows {tile_rows}");
                    // something worth comparing
                    let colors: std::collections::HashSet<&[u8]> = a.chunks(output_format.bytes_per_pixel()).collect();
                    assert!(colors.len() > 8);
                }
            }
        }
    }

    /// the triangles assemble_into puts down are assemble's, for a list,
    /// a strip and a fan of any length, each through the vertex it names.
    #[test]
    fn both_assemblers_make_the_same_triangles() {
        for topology in 0..4 {
            for count in 0..12 {
                let mut indices = vec![99];
                assemble_into(topology, count, |i| 100 + i as u32, &mut indices);
                let triangles: Vec<u32> =
                    assemble(topology, count).into_iter().flat_map(|(a, b, c)| [a, b, c].map(|i| 100 + i as u32)).collect();
                assert_eq!(indices, triangles, "topology {topology}, {count} vertices");
            }
        }
    }

    /// the triangles the GPU takes are the ones assemble_into puts down, for
    /// raw arrays as they number the vertices, a list's going as they are,
    /// for vertices in an order and for vertices in a row.
    #[cfg(feature = "vulkan")]
    #[test]
    fn the_gpu_takes_the_triangles_the_assembler_makes() {
        for topology in 0..4 {
            for count in 0..14 {
                let raw: Vec<u32> = (0..count as u32).map(|i| 1000 + i * 7 % 5).collect();
                let order: Vec<usize> = (0..count).map(|i| (i * 3) % 4).collect();
                let mut assembled = vec![99];
                let mut expected = Vec::new();
                assemble_into(topology, count, |i| raw[i], &mut expected);
                assert_eq!(triangle_indices(topology, count, Some(&raw), None, &mut assembled), expected, "raw, topology {topology}, {count} vertices");
                assemble_into(topology, count, |i| order[i] as u32, &mut expected);
                assert_eq!(triangle_indices(topology, count, None, Some(&order), &mut assembled), expected, "ordered, topology {topology}, {count} vertices");
                assemble_into(topology, count, |i| i as u32, &mut expected);
                assert_eq!(triangle_indices(topology, count, None, None, &mut assembled), expected, "in a row, topology {topology}, {count} vertices");
            }
        }
    }

    /// the semantics kept from the last draw are the ones the output map
    /// and mask give, through changes to those registers and to others.
    #[cfg(feature = "vulkan")]
    #[test]
    fn kept_semantics_follow_the_output_registers() {
        let mut seed = 11u32;
        // the high bits, the low ones of the sequence repeat soon
        let mut random = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed >> 8
        };
        let mut registers = vec![0u32; 0x300];
        let read: Vec<usize> =
            [REG_SHADER_OUTPUT_TOTAL, REG_VS_OUTPUT_MASK].into_iter().chain(REG_SHADER_OUTPUT_MAP..=REG_SHADER_OUTPUT_MAP_END).collect();
        // a map naming every semantic, which the changes below take apart
        registers[REG_SHADER_OUTPUT_TOTAL] = 7;
        for (i, register) in (REG_SHADER_OUTPUT_MAP..=REG_SHADER_OUTPUT_MAP_END).enumerate() {
            registers[register] = u32::from_le_bytes(std::array::from_fn(|c| (i * 4 + c) as u8));
        }
        registers[REG_VS_OUTPUT_MASK] = 0x7F;
        let mut kept = LastSemantics::default();
        for step in 0..2000 {
            let register = match random() % 8 {
                0 => (random() % 0x300) as usize,
                _ => read[(random() as usize) % read.len()],
            };
            // the low bits most of the time, which are what the map reads
            registers[register] ^= 1 << (random() % if random() % 4 == 0 { 32 } else { 5 });
            assert_eq!(kept.get(&registers), output_semantics(&registers), "step {step}, register {register:#X}");
        }
    }

    /// a drawing a transfer turned into a linear buffer, and another
    /// transfer turned back into one to sample, as titles do with a frame
    /// for their effects, stays on the GPU the whole way and leaves the bytes
    /// the CPU's two transfers do. a linear buffer only memory has the CPU
    /// transfers.
    #[cfg(feature = "vulkan")]
    #[test]
    fn transfers_out_of_linear_buffers_the_gpu_holds_match_the_cpu() {
        use crate::format::ColorFormat::*;
        let Ok(hardware) = hardware::Hardware::new() else { return };
        const SIZE: u32 = 32;
        const LINEAR: u32 = 0x10_0000;
        const OUTPUT: u32 = 0x18_0000;
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        let mut seed = 13u32;
        let mut random = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1 << 24) as f32
        };
        let mut registers = target_registers();
        registers[REG_VIEWPORT_WIDTH] = float24(SIZE as f32 / 2.0);
        registers[REG_VIEWPORT_HEIGHT] = float24(SIZE as f32 / 2.0);
        registers[REG_FRAMEBUFFER_DIMENSIONS] = SIZE | ((SIZE - 1) << 12);
        let mut drawn = ConsoleMemory::default();
        for _ in 0..40 {
            let color = [random(), random(), random(), random()];
            let triangle: Vec<Vertex> = (0..3)
                .map(|_| Vertex {
                    clip: [random() * 2.0 - 1.0, random() * 2.0 - 1.0, -0.5, 1.0],
                    color,
                    texcoords: [[0.0; 2]; 3],
                    quaternion: [0.0, 0.0, 0.0, 1.0],
                    view: [0.0; 3],
                })
                .collect();
            rasterize_shaded(&registers, &mut drawn, &mut resources, &triangle);
        }
        resources.hardware.as_mut().unwrap().flush(&mut drawn).unwrap();
        let flags = |flip: bool, input_linear: bool, output_tiled: bool, input: ColorFormat, output: ColorFormat, downscale: u32| {
            flip as u32
                | (input_linear as u32) << 1
                | ((input_linear != output_tiled) as u32) << 5
                | (hardware::format_index(input) as u32) << 8
                | (hardware::format_index(output) as u32) << 12
                | downscale << 24
        };
        for linear_format in [Rgba8, Rgb565] {
            for output_format in [Rgba8, Rgb565, Rgba4] {
                for (flip, downscale, tiled) in [(false, 0, true), (true, 0, true), (false, 2, true), (true, 1, false)] {
                    let hardware = resources.hardware.as_mut().unwrap();
                    let (mut gpu, mut cpu) = (drawn.clone(), drawn.clone());
                    let (scale_x, scale_y) = [(1, 1), (2, 1), (2, 2)][downscale as usize];
                    let (width, height) = (SIZE / scale_x, SIZE / scale_y);
                    let to_linear = hardware::Transfer {
                        input: COLOR,
                        output: LINEAR,
                        input_width: SIZE,
                        input_height: SIZE,
                        output_width: SIZE,
                        output_height: SIZE,
                        copy: (SIZE, SIZE),
                        scale: (1, 1),
                        flip: false,
                        input_linear: false,
                        output_tiled: false,
                        input_format: Rgba8,
                        output_format: linear_format,
                    };
                    let back = hardware::Transfer {
                        input: LINEAR,
                        output: OUTPUT,
                        output_width: width,
                        output_height: height,
                        copy: (width, height),
                        scale: (scale_x, scale_y),
                        flip,
                        input_linear: true,
                        output_tiled: tiled,
                        input_format: linear_format,
                        output_format,
                        ..to_linear
                    };
                    assert!(hardware.display_transfer(&mut gpu, &to_linear).unwrap());
                    assert!(hardware.display_transfer(&mut gpu, &back).unwrap(), "out of the linear buffer on the GPU");
                    hardware.flush(&mut gpu).unwrap();
                    let mut software = crate::Gpu::new();
                    let size = SIZE | SIZE << 16;
                    software.display_transfer(&mut cpu, COLOR, LINEAR, size, size, flags(false, false, false, Rgba8, linear_format, 0));
                    let output_size = (width * scale_x) | ((height * scale_y) << 16);
                    let back_flags = flags(flip, true, tiled, linear_format, output_format, downscale);
                    software.display_transfer(&mut cpu, LINEAR, OUTPUT, size, output_size, back_flags);
                    let len = (width * height) as usize * output_format.bytes_per_pixel();
                    let (mut a, mut b) = (vec![0u8; len], vec![0u8; len]);
                    cpu.read(OUTPUT, &mut a);
                    gpu.read(OUTPUT, &mut b);
                    assert!(a == b, "{linear_format:?} to {output_format:?}, flip {flip}, downscale {downscale}, tiled {tiled}");
                    let colors: std::collections::HashSet<&[u8]> = a.chunks(output_format.bytes_per_pixel()).collect();
                    assert!(colors.len() > 8);
                }
            }
        }
        // nothing the GPU holds there
        let hardware = resources.hardware.as_mut().unwrap();
        let elsewhere = hardware::Transfer {
            input: 0x0C_0000,
            output: OUTPUT,
            input_width: SIZE,
            input_height: SIZE,
            output_width: SIZE,
            output_height: SIZE,
            copy: (SIZE, SIZE),
            scale: (1, 1),
            flip: false,
            input_linear: true,
            output_tiled: true,
            input_format: Rgb565,
            output_format: Rgb565,
        };
        assert!(!hardware.display_transfer(&mut drawn.clone(), &elsewhere).unwrap());
        // an output of part of a tile is the CPU's too
        let partial = hardware::Transfer { input: LINEAR, output_height: SIZE - 4, copy: (SIZE, SIZE - 4), ..elsewhere };
        assert!(!hardware.display_transfer(&mut drawn.clone(), &partial).unwrap());
    }

    /// a fill over a whole linear buffer the GPU wrote, with a pattern that
    /// does not line up with its pixels, is what a transfer out of it reads,
    /// also when memory had that pattern already.
    #[cfg(feature = "vulkan")]
    #[test]
    fn a_transfer_reads_the_fill_over_a_linear_buffer_the_gpu_wrote() {
        use crate::format::ColorFormat::*;
        let Ok(hardware) = hardware::Hardware::new() else { return };
        const SIZE: u32 = 32;
        const LINEAR: u32 = 0x10_0000;
        const OUTPUT: u32 = 0x18_0000;
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        let mut registers = target_registers();
        registers[REG_VIEWPORT_WIDTH] = float24(SIZE as f32 / 2.0);
        registers[REG_VIEWPORT_HEIGHT] = float24(SIZE as f32 / 2.0);
        registers[REG_FRAMEBUFFER_DIMENSIONS] = SIZE | ((SIZE - 1) << 12);
        let mut gpu = ConsoleMemory::default();
        let triangle: Vec<Vertex> = [[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]]
            .into_iter()
            .map(|[x, y]| Vertex {
                clip: [x, y, -0.5, 1.0],
                color: [0.2, 0.6, 0.9, 1.0],
                texcoords: [[0.0; 2]; 3],
                quaternion: [0.0, 0.0, 0.0, 1.0],
                view: [0.0; 3],
            })
            .collect();
        rasterize_shaded(&registers, &mut gpu, &mut resources, &triangle);
        let hardware = resources.hardware.as_mut().unwrap();
        hardware.flush(&mut gpu).unwrap();
        let mut cpu = gpu.clone();
        // halves of a pixel that differ
        let pattern: Vec<u8> = [0xFF, 0xFF, 0x00, 0x00].repeat((SIZE * SIZE / 2) as usize);
        let fill = |hardware: &mut hardware::Hardware, memory: &mut ConsoleMemory| {
            hardware.before_fill(memory, LINEAR, pattern.len() as u32, 4).unwrap();
            memory.write(LINEAR, &pattern);
            hardware.filled(LINEAR, &pattern).unwrap();
        };
        let to_linear = hardware::Transfer {
            input: COLOR,
            output: LINEAR,
            input_width: SIZE,
            input_height: SIZE,
            output_width: SIZE,
            output_height: SIZE,
            copy: (SIZE, SIZE),
            scale: (1, 1),
            flip: false,
            input_linear: false,
            output_tiled: false,
            input_format: Rgba8,
            output_format: Rgb565,
        };
        let back = hardware::Transfer { input: LINEAR, output: OUTPUT, input_linear: true, output_tiled: true, input_format: Rgb565, ..to_linear };
        fill(hardware, &mut gpu);
        assert!(hardware.display_transfer(&mut gpu, &to_linear).unwrap());
        fill(hardware, &mut gpu);
        assert!(hardware.display_transfer(&mut gpu, &back).unwrap());
        hardware.flush(&mut gpu).unwrap();
        // the CPU's, the drawing gone under the fill
        let mut software = crate::Gpu::new();
        cpu.write(LINEAR, &pattern);
        let size = SIZE | SIZE << 16;
        software.display_transfer(&mut cpu, LINEAR, OUTPUT, size, size, 1 << 1 | (hardware::format_index(Rgb565) as u32) << 8 | (hardware::format_index(Rgb565) as u32) << 12);
        let len = (SIZE * SIZE * 2) as usize;
        let (mut a, mut b) = (vec![0u8; len], vec![0u8; len]);
        cpu.read(OUTPUT, &mut a);
        gpu.read(OUTPUT, &mut b);
        assert!(a == b, "the transfer read what the GPU drew under the fill");
    }

    /// a screen shown straight from the GPU's image is the picture the host
    /// gets a copy of otherwise, where the image says the screen lies in it,
    /// for a screen some rows into a buffer with longer rows than it shows.
    #[cfg(feature = "vulkan")]
    #[test]
    fn screens_shown_from_the_gpu_match_the_copies() {
        use crate::format::ColorFormat::Rgba8;
        const SIZE: u32 = 32;
        const OUTPUT: u32 = 0x10_0000;
        for scale in [1, 3] {
            let Ok(copied) = hardware::Hardware::new() else { return };
            let device = hardware::own_device().unwrap();
            let direct = hardware::Hardware::with_device(std::sync::Arc::new(device), true).unwrap();
            let mut pictures = Vec::new();
            for mut hardware in [copied, direct] {
                let scale = hardware.set_scale(scale);
                let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
                let mut registers = target_registers();
                registers[REG_VIEWPORT_WIDTH] = float24(SIZE as f32 / 2.0);
                registers[REG_VIEWPORT_HEIGHT] = float24(SIZE as f32 / 2.0);
                registers[REG_FRAMEBUFFER_DIMENSIONS] = SIZE | ((SIZE - 1) << 12);
                let mut seed = 11u32;
                let mut random = move || {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (seed >> 8) as f32 / (1 << 24) as f32
                };
                let mut memory = ConsoleMemory::default();
                for _ in 0..40 {
                    let color = [random(), random(), random(), 1.0];
                    let triangle: Vec<Vertex> = (0..3)
                        .map(|_| Vertex {
                            clip: [random() * 2.0 - 1.0, random() * 2.0 - 1.0, -0.5, 1.0],
                            color,
                            texcoords: [[0.0; 2]; 3],
                            quaternion: [0.0, 0.0, 0.0, 1.0],
                            view: [0.0; 3],
                        })
                        .collect();
                    rasterize_shaded(&registers, &mut memory, &mut resources, &triangle);
                }
                let hardware = resources.hardware.as_mut().unwrap();
                let transfer = hardware::Transfer {
                    input: COLOR,
                    output: OUTPUT,
                    input_width: SIZE,
                    input_height: SIZE,
                    output_width: SIZE,
                    output_height: SIZE,
                    copy: (SIZE, SIZE),
                    scale: (1, 1),
                    flip: false,
                    input_linear: false,
                    output_tiled: false,
                    input_format: Rgba8,
                    output_format: Rgba8,
                };
                assert!(hardware.display_transfer(&mut memory, &transfer).unwrap());
                // 20 rows from the fourth, of 24 pixels each
                let screen = hardware.screen(OUTPUT + 4 * SIZE * 4, (24, 20), SIZE, Rgba8, &[]).expect("drawn on the GPU");
                let picture = match hardware.screen_image(screen).unwrap() {
                    Some(image) => {
                        let (pixels, width) = hardware.upright_pixels(screen).unwrap();
                        let height = pixels.len() as u32 / 4 / width;
                        let [x, y, w, h] = image.area.map(f64::from);
                        let [left, top] = [(x * width as f64).round() as usize, (y * height as f64).round() as usize];
                        let [right, bottom] = [((x + w) * width as f64).round() as usize, ((y + h) * height as f64).round() as usize];
                        // the bounds half a texel in from the area's edges
                        let [l, t, r, b] = image.bounds.map(f64::from);
                        let half = [0.5 / width as f64, 0.5 / height as f64];
                        assert!((l - x - half[0]).abs() < 1e-6 && (t - y - half[1]).abs() < 1e-6);
                        assert!((x + w - r - half[0]).abs() < 1e-6 && (y + h - b - half[1]).abs() < 1e-6);
                        let width = width as usize;
                        (top..bottom).flat_map(|row| pixels[(row * width + left) * 4..(row * width + right) * 4].to_vec()).collect()
                    }
                    None => hardware.picture(screen).unwrap().expect("copied").0.to_vec(),
                };
                assert_eq!(picture.len(), (20 * 24 * 4 * scale * scale) as usize);
                pictures.push(picture);
            }
            let colors: std::collections::HashSet<&[u8]> = pictures[0].chunks(4).collect();
            assert!(colors.len() > 8, "something worth comparing");
            assert!(pictures[0] == pictures[1], "at {scale}x");
        }
    }
}
