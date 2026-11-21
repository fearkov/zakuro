//! what the emulated GPU holds, kept in a save state, its registers, the shader units' programs and uniforms,
//! the lookup tables and the work it's in the middle of
//! what the host's GPU keeps comes from memory again

use serde::{Deserialize, Serialize};

use crate::shader::Vec4;
use crate::{FramebufferConfig, Gpu, GpuMemory};

/// a shader unit, as saved
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnitState {
    pub(crate) program: Vec<u32>,
    pub(crate) descriptors: Vec<u32>,
    pub(crate) float_uniforms: Vec<Vec4>,
    pub(crate) int_uniforms: [[u8; 4]; 4],
    pub(crate) bool_uniforms: u16,
    pub(crate) entry_point: u32,
    pub(crate) program_write_offset: u32,
    pub(crate) descriptor_write_offset: u32,
    pub(crate) float_uniform_index: u32,
    pub(crate) float_uniform_component: u32,
    pub(crate) float_uniform_wide: bool,
    pub(crate) float_uniform_staging: [u32; 4],
}

/// the GPU, as saved
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuState {
    external: Vec<u32>,
    internal: Vec<u32>,
    vertex_shader: UnitState,
    geometry_shader: UnitState,
    fixed_attributes: Vec<Vec4>,
    fixed_attribute_staging: [u32; 3],
    fixed_attribute_words: u32,
    immediate_attributes: Vec<Vec4>,
    immediate_next: u32,
    immediate_vertices: Vec<Vec<Vec4>>,
    pending_jump: Option<(u32, u32)>,
    framebuffers: [FramebufferConfig; 2],
    counters: [u64; 5],
    light_tables: Vec<[f32; 2]>,
    proctex_maps: Vec<[f32; 2]>,
    proctex_colors: Vec<[f32; 4]>,
    proctex_steps: Vec<[f32; 4]>,
    fog: Vec<[f32; 2]>,
}

impl Gpu {
    /// what a save state keeps of the GPU
    pub fn save_state(&self) -> GpuState {
        let (proctex_maps, proctex_colors, proctex_steps) = self.resources.proctex_tables.saved();
        GpuState {
            external: self.external.to_vec(),
            internal: self.internal.to_vec(),
            vertex_shader: self.vertex_shader.saved(),
            geometry_shader: self.geometry_shader.saved(),
            fixed_attributes: self.fixed_attributes.to_vec(),
            fixed_attribute_staging: self.fixed_attribute_staging,
            fixed_attribute_words: self.fixed_attribute_words as u32,
            immediate_attributes: self.immediate.attributes.to_vec(),
            immediate_next: self.immediate.next_attribute as u32,
            immediate_vertices: self.immediate.vertices.iter().map(|vertex| vertex.to_vec()).collect(),
            pending_jump: self.pending_jump,
            framebuffers: self.framebuffers,
            counters: [self.command_lists, self.draw_calls, self.fills, self.transfers, self.vertices_drawn],
            light_tables: self.resources.light_tables.saved(),
            proctex_maps,
            proctex_colors,
            proctex_steps,
            fog: self.resources.fog_table.saved(),
        }
    }

    /// puts the GPU back like a save state had it, after memory, and drops everything the host's GPU drew, that comes from memory again
    pub fn load_state(&mut self, state: &GpuState) -> Result<(), String> {
        let fits = state.external.len() == self.external.len()
            && state.internal.len() == self.internal.len()
            && state.fixed_attributes.len() == self.fixed_attributes.len()
            && state.immediate_attributes.len() == self.immediate.attributes.len()
            && state.immediate_vertices.iter().all(|vertex| vertex.len() == 16);
        if !fits {
            return Err("the GPU's registers do not fit".into());
        }
        self.external.copy_from_slice(&state.external);
        self.internal.copy_from_slice(&state.internal);
        if !self.vertex_shader.restore(&state.vertex_shader) || !self.geometry_shader.restore(&state.geometry_shader) {
            return Err("the shader units do not fit".into());
        }
        self.fixed_attributes.copy_from_slice(&state.fixed_attributes);
        self.fixed_attribute_staging = state.fixed_attribute_staging;
        self.fixed_attribute_words = state.fixed_attribute_words as usize;
        self.immediate.attributes.copy_from_slice(&state.immediate_attributes);
        self.immediate.next_attribute = state.immediate_next as usize;
        self.immediate.vertices = state
            .immediate_vertices
            .iter()
            .map(|vertex| std::array::from_fn(|i| vertex[i]))
            .collect();
        self.pending_jump = state.pending_jump;
        self.framebuffers = state.framebuffers;
        [self.command_lists, self.draw_calls, self.fills, self.transfers, self.vertices_drawn] = state.counters;
        let resources = &mut self.resources;
        let tables = resources.light_tables.restore(&state.light_tables)
            && resources.proctex_tables.restore(&state.proctex_maps, &state.proctex_colors, &state.proctex_steps)
            && resources.fog_table.restore(&state.fog);
        if !tables {
            return Err("the lookup tables do not fit".into());
        }
        self.forget_drawing();
        Ok(())
    }

    /// writes back everything the host's GPU drew that memory doesn't have yet, so a save state takes memory as it stands
    pub fn write_back_all<M: GpuMemory>(&mut self, memory: &mut M) {
        self.sync_memory(memory, 0, u32::MAX);
    }

    /// drops what the host's GPU drew and decoded (memory having been put back like a save state had it) and what
    /// the last draws left to the next
    pub fn forget_drawing(&mut self) {
        self.resources.forget();
        if self.interpolation.is_some() {
            self.interpolation = Some(Default::default());
        }
    }
}
