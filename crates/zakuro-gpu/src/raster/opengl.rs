//! a small OpenGL rasterizer for untextured, vertex-colored triangles

use glow::HasContext;
use std::collections::HashMap;

use super::{ColorTarget, DepthMap, DepthStencil, Screen};
use crate::{GpuMemory, format::morton_offset};

const VERTEX_SHADER: &str = r#"#version 330 core
layout(location = 0) in vec2 position_over_w;
layout(location = 1) in float inverse_w;
layout(location = 2) in vec4 color;
layout(location = 3) in float depth_z;
out vec4 vertex_color;
 noperspective out float pica_depth;
 noperspective out float pica_inverse_w;
void main() {
    gl_Position = vec4(position_over_w, 0.0, inverse_w);
    vertex_color = color;
    pica_depth = depth_z;
    pica_inverse_w = inverse_w;
}
"#;

const FRAGMENT_SHADER: &str = r#"#version 330 core
in vec4 vertex_color;
 noperspective in float pica_depth;
 noperspective in float pica_inverse_w;
uniform float depth_scale;
uniform float depth_offset;
uniform bool w_buffer;
out vec4 color;
void main() {
    float depth = pica_depth * depth_scale + depth_offset;
    if (w_buffer) depth /= max(pica_inverse_w, 1e-20);
    gl_FragDepth = clamp(depth, 0.0, 1.0);
    color = floor(clamp(vertex_color, 0.0, 1.0) * 255.0) / 255.0;
}
"#;

/// OpenGL resources are created and used on the thread where the context is current.
pub(crate) struct OpenGlRasterizer {
    gl: std::sync::Arc<glow::Context>,
    program: glow::Program,
    vertex_array: glow::VertexArray,
    vertices: glow::Buffer,
    surfaces: HashMap<SurfaceKey, Surface>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct SurfaceKey {
    addr: u32,
    format: u32,
    width: u32,
    height: u32,
    depth_addr: u32,
    depth_bytes: u32,
}

struct Surface {
    framebuffer: glow::Framebuffer,
    color: glow::Texture,
    depth: Option<glow::Texture>,
    initialized: bool,
    dirty: bool,
    depth_dirty: bool,
}

impl OpenGlRasterizer {
    pub(crate) unsafe fn new(context: crate::SharedOpenGlContext) -> Result<Self, String> {
        let gl = context.0;
        let version = unsafe { gl.get_parameter_string(glow::VERSION) };
        if !version_at_least(&version, 4, 3) {
            return Err(format!(
                "OpenGL 4.3 is required for hardware rasterization, found {version}"
            ));
        }

        unsafe {
            let program = link_program(&gl)?;
            let vertex_array = gl
                .create_vertex_array()
                .map_err(|error| error.to_string())?;
            let vertices = gl.create_buffer().map_err(|error| error.to_string())?;

            gl.bind_vertex_array(Some(vertex_array));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vertices));
            let stride = (8 * size_of::<f32>()) as i32;
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, stride, 0);
            gl.enable_vertex_attrib_array(1);
            gl.vertex_attrib_pointer_f32(
                1,
                1,
                glow::FLOAT,
                false,
                stride,
                2 * size_of::<f32>() as i32,
            );
            gl.enable_vertex_attrib_array(2);
            gl.vertex_attrib_pointer_f32(
                2,
                4,
                glow::FLOAT,
                false,
                stride,
                3 * size_of::<f32>() as i32,
            );
            gl.enable_vertex_attrib_array(3);
            gl.vertex_attrib_pointer_f32(
                3,
                1,
                glow::FLOAT,
                false,
                stride,
                7 * size_of::<f32>() as i32,
            );
            gl.bind_vertex_array(None);

            Ok(Self {
                gl,
                program,
                vertex_array,
                vertices,
                surfaces: HashMap::new(),
            })
        }
    }

    pub(super) fn draw<M: GpuMemory>(
        &mut self,
        memory: &mut M,
        target: &ColorTarget,
        depth: Option<&DepthStencil>,
        depth_map: DepthMap,
        triangles: &[[Screen; 3]],
    ) -> Result<(), String> {
        if triangles.is_empty() {
            return Ok(());
        }
        let (width, height) = (target.buffer_width, target.buffer_height);
        let pixels = width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(4))
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or("OpenGL target dimensions overflow")?;
        let key = SurfaceKey {
            addr: target.addr,
            format: target.format as u32,
            width,
            height,
            depth_addr: depth.map_or(0, |d| d.addr),
            depth_bytes: depth.map_or(0, |d| d.bytes),
        };
        let cached = self.surfaces.contains_key(&key);
        if cached {
            self.sync_except(memory, target.addr, pixels as u32, true, Some(key));
            if let Some(depth) = depth {
                self.sync_except(
                    memory,
                    depth.addr,
                    width.saturating_mul(height).saturating_mul(depth.bytes),
                    true,
                    Some(key),
                );
            }
        } else {
            self.sync(memory, target.addr, pixels as u32, true);
            if let Some(depth) = depth {
                self.sync(
                    memory,
                    depth.addr,
                    width.saturating_mul(height).saturating_mul(depth.bytes),
                    true,
                );
            }
        }
        if !self.surfaces.contains_key(&key) {
            let (framebuffer, color, depth_texture) = unsafe {
                let framebuffer = self.gl.create_framebuffer().map_err(|e| e.to_string())?;
                let color = self.gl.create_texture().map_err(|e| e.to_string())?;
                self.gl.bind_texture(glow::TEXTURE_2D, Some(color));
                self.gl.tex_parameter_i32(
                    glow::TEXTURE_2D,
                    glow::TEXTURE_MIN_FILTER,
                    glow::NEAREST as i32,
                );
                self.gl.tex_parameter_i32(
                    glow::TEXTURE_2D,
                    glow::TEXTURE_MAG_FILTER,
                    glow::NEAREST as i32,
                );
                self.gl.tex_image_2d(
                    glow::TEXTURE_2D,
                    0,
                    glow::RGBA8 as i32,
                    width as i32,
                    height as i32,
                    0,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelUnpackData::Slice(None),
                );
                self.gl
                    .bind_framebuffer(glow::FRAMEBUFFER, Some(framebuffer));
                self.gl.framebuffer_texture_2d(
                    glow::FRAMEBUFFER,
                    glow::COLOR_ATTACHMENT0,
                    glow::TEXTURE_2D,
                    Some(color),
                    0,
                );
                let depth_texture = if key.depth_bytes != 0 {
                    let texture = self.gl.create_texture().map_err(|e| e.to_string())?;
                    self.gl.bind_texture(glow::TEXTURE_2D, Some(texture));
                    self.gl.tex_parameter_i32(
                        glow::TEXTURE_2D,
                        glow::TEXTURE_MIN_FILTER,
                        glow::NEAREST as i32,
                    );
                    self.gl.tex_parameter_i32(
                        glow::TEXTURE_2D,
                        glow::TEXTURE_MAG_FILTER,
                        glow::NEAREST as i32,
                    );
                    let internal = if key.depth_bytes == 2 {
                        glow::DEPTH_COMPONENT16
                    } else if key.depth_bytes == 4 {
                        glow::DEPTH24_STENCIL8
                    } else {
                        glow::DEPTH_COMPONENT24
                    };
                    self.gl.tex_image_2d(
                        glow::TEXTURE_2D,
                        0,
                        internal as i32,
                        width as i32,
                        height as i32,
                        0,
                        if key.depth_bytes == 4 {
                            glow::DEPTH_STENCIL
                        } else {
                            glow::DEPTH_COMPONENT
                        },
                        if key.depth_bytes == 2 {
                            glow::UNSIGNED_SHORT
                        } else if key.depth_bytes == 4 {
                            glow::UNSIGNED_INT_24_8
                        } else {
                            glow::FLOAT
                        },
                        glow::PixelUnpackData::Slice(None),
                    );
                    self.gl.framebuffer_texture_2d(
                        glow::FRAMEBUFFER,
                        if key.depth_bytes == 4 {
                            glow::DEPTH_STENCIL_ATTACHMENT
                        } else {
                            glow::DEPTH_ATTACHMENT
                        },
                        glow::TEXTURE_2D,
                        Some(texture),
                        0,
                    );
                    Some(texture)
                } else {
                    None
                };
                self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                (framebuffer, color, depth_texture)
            };
            self.surfaces.insert(
                key,
                Surface {
                    framebuffer,
                    color,
                    depth: depth_texture,
                    initialized: false,
                    dirty: false,
                    depth_dirty: false,
                },
            );
        }
        let surface = self.surfaces.get_mut(&key).unwrap();
        let mut tiled = vec![0; pixels];
        let linear = if surface.initialized {
            Vec::new()
        } else {
            memory.read(target.addr, &mut tiled);
            tiled_to_linear(&tiled, width, height)
        };
        let depth_linear = if !surface.initialized && key.depth_bytes != 0 {
            let mut depth_tiled = vec![0; (width * height * key.depth_bytes) as usize];
            memory.read(key.depth_addr, &mut depth_tiled);
            Some(depth_to_linear(
                &depth_tiled,
                width,
                height,
                key.depth_bytes,
            ))
        } else {
            None
        };

        let mut vertices = Vec::with_capacity(triangles.len() * 3 * 8);
        for triangle in triangles {
            for vertex in triangle {
                let inverse_w = vertex.inv_w;
                vertices.extend([
                    (2.0 * vertex.x / width as f32 - 1.0) * inverse_w,
                    (2.0 * vertex.y / height as f32 - 1.0) * inverse_w,
                    inverse_w,
                ]);
                vertices.extend(vertex.color_over_w.map(|component| component / inverse_w));
                vertices.push(vertex.z);
            }
        }
        let vertex_bytes = bytemuck::cast_slice(&vertices);

        unsafe {
            let gl = &self.gl;
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            gl.pixel_store_i32(glow::PACK_ALIGNMENT, 1);
            gl.bind_texture(glow::TEXTURE_2D, Some(surface.color));
            if !surface.initialized {
                gl.tex_sub_image_2d(
                    glow::TEXTURE_2D,
                    0,
                    0,
                    0,
                    width as i32,
                    height as i32,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelUnpackData::Slice(Some(&linear)),
                );
                if let (Some(depth_texture), Some(depth_data)) =
                    (surface.depth, depth_linear.as_ref())
                {
                    gl.bind_texture(glow::TEXTURE_2D, Some(depth_texture));
                    gl.tex_sub_image_2d(
                        glow::TEXTURE_2D,
                        0,
                        0,
                        0,
                        width as i32,
                        height as i32,
                        if key.depth_bytes == 4 {
                            glow::DEPTH_STENCIL
                        } else {
                            glow::DEPTH_COMPONENT
                        },
                        if key.depth_bytes == 2 {
                            glow::UNSIGNED_SHORT
                        } else if key.depth_bytes == 4 {
                            glow::UNSIGNED_INT_24_8
                        } else {
                            glow::FLOAT
                        },
                        glow::PixelUnpackData::Slice(Some(depth_data)),
                    );
                }
            }

            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(surface.framebuffer));
            if gl.check_framebuffer_status(glow::FRAMEBUFFER) != glow::FRAMEBUFFER_COMPLETE {
                gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                return Err("OpenGL framebuffer is incomplete".to_owned());
            }
            gl.viewport(0, 0, width as i32, height as i32);
            gl.enable(glow::SCISSOR_TEST);
            gl.scissor(
                target.left,
                target.bottom,
                target.right.saturating_sub(target.left),
                target.top.saturating_sub(target.bottom),
            );
            gl.disable(glow::BLEND);
            if let Some(depth_state) = depth {
                gl.enable(glow::DEPTH_TEST);
                gl.depth_func(depth_state.test.map_or(glow::ALWAYS, compare));
                gl.depth_mask(depth_state.write_depth);
                if let Some(stencil) = depth_state.stencil {
                    gl.enable(glow::STENCIL_TEST);
                    gl.stencil_func(
                        compare(stencil.function),
                        stencil.reference as i32,
                        stencil.compare_mask as u32,
                    );
                    gl.stencil_mask(if depth_state.write_stencil {
                        stencil.write_mask as u32
                    } else {
                        0
                    });
                    gl.stencil_op(
                        stencil_op(stencil.fail),
                        stencil_op(stencil.depth_fail),
                        stencil_op(stencil.pass),
                    );
                } else {
                    gl.disable(glow::STENCIL_TEST);
                }
            } else {
                gl.disable(glow::DEPTH_TEST);
                gl.disable(glow::STENCIL_TEST);
            }
            gl.disable(glow::CULL_FACE);
            gl.disable(glow::DITHER);
            gl.disable(glow::FRAMEBUFFER_SRGB);
            gl.color_mask(true, true, true, true);
            gl.use_program(Some(self.program));
            gl.bind_vertex_array(Some(self.vertex_array));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vertices));
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, vertex_bytes, glow::STREAM_DRAW);
            if let Some(location) = gl.get_uniform_location(self.program, "depth_scale") {
                gl.uniform_1_f32(Some(&location), depth_map.scale);
            }
            if let Some(location) = gl.get_uniform_location(self.program, "depth_offset") {
                gl.uniform_1_f32(Some(&location), depth_map.offset);
            }
            if let Some(location) = gl.get_uniform_location(self.program, "w_buffer") {
                gl.uniform_1_i32(Some(&location), depth_map.w_buffer as i32);
            }
            gl.draw_arrays(glow::TRIANGLES, 0, (vertices.len() / 8) as i32);
            gl.bind_vertex_array(None);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.disable(glow::SCISSOR_TEST);
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::STENCIL_TEST);
            gl.disable(glow::BLEND);
            gl.depth_mask(true);
            gl.stencil_mask(0xFF);
            gl.color_mask(true, true, true, true);
        }
        surface.initialized = true;
        surface.dirty = true;
        surface.depth_dirty |= depth.is_some_and(|d| d.write_depth || d.write_stencil);
        memory.guard(target.addr, (width * height * 4) as u32);
        memory.guard_writes(target.addr, (width * height * 4) as u32);
        if let Some(depth) = depth.filter(|d| d.write_depth || d.write_stencil) {
            let bytes = width.saturating_mul(height).saturating_mul(depth.bytes);
            memory.guard(depth.addr, bytes);
            memory.guard_writes(depth.addr, bytes);
        }
        Ok(())
    }

    /// Resolve GPU-new pixels only when guest memory is about to be observed or overwritten.
    pub(crate) fn sync<M: GpuMemory>(
        &mut self,
        memory: &mut M,
        addr: u32,
        len: u32,
        invalidate: bool,
    ) {
        self.sync_except(memory, addr, len, invalidate, None);
    }

    fn sync_except<M: GpuMemory>(
        &mut self,
        memory: &mut M,
        addr: u32,
        len: u32,
        invalidate: bool,
        except: Option<SurfaceKey>,
    ) {
        let end = addr.saturating_add(len);
        let keys: Vec<_> = self
            .surfaces
            .iter()
            .filter_map(|(key, surface)| {
                if Some(*key) == except {
                    return None;
                }
                let bytes = key.width.saturating_mul(key.height).saturating_mul(4);
                let color_overlap = addr < key.addr.saturating_add(bytes) && key.addr < end;
                let color_hit = surface.dirty && color_overlap;
                let depth_size = key
                    .width
                    .saturating_mul(key.height)
                    .saturating_mul(key.depth_bytes);
                let depth_overlap = key.depth_bytes != 0
                    && addr < key.depth_addr.saturating_add(depth_size)
                    && key.depth_addr < end;
                let depth_hit = surface.depth_dirty && depth_overlap;
                (color_hit || depth_hit || (invalidate && (color_overlap || depth_overlap)))
                    .then_some(*key)
            })
            .collect();
        for key in keys {
            let surface = self.surfaces.get_mut(&key).unwrap();
            let (width, height) = (key.width, key.height);
            if surface.dirty {
                let mut linear = vec![0u8; (width * height * 4) as usize];
                let mut tiled = vec![0u8; linear.len()];
                unsafe {
                    self.gl
                        .bind_framebuffer(glow::FRAMEBUFFER, Some(surface.framebuffer));
                    self.gl.pixel_store_i32(glow::PACK_ALIGNMENT, 1);
                    self.gl.read_pixels(
                        0,
                        0,
                        width as i32,
                        height as i32,
                        glow::RGBA,
                        glow::UNSIGNED_BYTE,
                        glow::PixelPackData::Slice(Some(&mut linear)),
                    );
                    self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                }
                linear_to_tiled(&linear, &mut tiled, width, height);
                memory.write(key.addr, &tiled);
                surface.dirty = false;
            }
            if surface.depth_dirty && key.depth_bytes != 0 {
                let mut depth_linear = vec![0u8; (width * height * key.depth_bytes) as usize];
                unsafe {
                    self.gl
                        .bind_framebuffer(glow::FRAMEBUFFER, Some(surface.framebuffer));
                    if key.depth_bytes == 4 {
                        self.gl.read_pixels(
                            0,
                            0,
                            width as i32,
                            height as i32,
                            glow::DEPTH_STENCIL,
                            glow::UNSIGNED_INT_24_8,
                            glow::PixelPackData::Slice(Some(bytemuck::cast_slice_mut(
                                &mut depth_linear,
                            ))),
                        );
                    } else if key.depth_bytes == 2 {
                        let mut values = vec![0u16; (width * height) as usize];
                        self.gl.read_pixels(
                            0,
                            0,
                            width as i32,
                            height as i32,
                            glow::DEPTH_COMPONENT,
                            glow::UNSIGNED_SHORT,
                            glow::PixelPackData::Slice(Some(bytemuck::cast_slice_mut(&mut values))),
                        );
                        for (i, value) in values.into_iter().enumerate() {
                            depth_linear[i * 2..i * 2 + 2].copy_from_slice(&value.to_le_bytes());
                        }
                    } else {
                        let mut values = vec![0f32; (width * height) as usize];
                        self.gl.read_pixels(
                            0,
                            0,
                            width as i32,
                            height as i32,
                            glow::DEPTH_COMPONENT,
                            glow::FLOAT,
                            glow::PixelPackData::Slice(Some(bytemuck::cast_slice_mut(&mut values))),
                        );
                        for (i, value) in values.into_iter().enumerate() {
                            let raw = (value.clamp(0.0, 1.0) * 16_777_215.0 + 0.5) as u32;
                            depth_linear[i * 3..i * 3 + 3].copy_from_slice(&raw.to_le_bytes()[..3]);
                        }
                    }
                    self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                }
                let mut depth_tiled = vec![0; depth_linear.len()];
                linear_to_tiled_bytes(
                    &depth_linear,
                    &mut depth_tiled,
                    width,
                    height,
                    key.depth_bytes,
                );
                memory.write(key.depth_addr, &depth_tiled);
                surface.depth_dirty = false;
            }
            surface.dirty = false;
            if invalidate {
                surface.initialized = false;
            }
        }
    }
}

unsafe fn link_program(gl: &glow::Context) -> Result<glow::Program, String> {
    unsafe {
        let program = gl.create_program().map_err(|error| error.to_string())?;
        let mut shaders = Vec::new();
        for (kind, source) in [
            (glow::VERTEX_SHADER, VERTEX_SHADER),
            (glow::FRAGMENT_SHADER, FRAGMENT_SHADER),
        ] {
            let shader = gl.create_shader(kind).map_err(|error| error.to_string())?;
            gl.shader_source(shader, source);
            gl.compile_shader(shader);
            if !gl.get_shader_compile_status(shader) {
                return Err(format!(
                    "OpenGL shader failed to compile: {}",
                    gl.get_shader_info_log(shader)
                ));
            }
            gl.attach_shader(program, shader);
            shaders.push(shader);
        }
        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            return Err(format!(
                "OpenGL program failed to link: {}",
                gl.get_program_info_log(program)
            ));
        }
        for shader in shaders {
            gl.detach_shader(program, shader);
            gl.delete_shader(shader);
        }
        Ok(program)
    }
}

fn compare(value: super::Compare) -> u32 {
    match value {
        super::Compare::Never => glow::NEVER,
        super::Compare::Always => glow::ALWAYS,
        super::Compare::Equal => glow::EQUAL,
        super::Compare::NotEqual => glow::NOTEQUAL,
        super::Compare::Less => glow::LESS,
        super::Compare::LessOrEqual => glow::LEQUAL,
        super::Compare::Greater => glow::GREATER,
        super::Compare::GreaterOrEqual => glow::GEQUAL,
    }
}

fn stencil_op(value: super::StencilOp) -> u32 {
    match value {
        super::StencilOp::Keep => glow::KEEP,
        super::StencilOp::Zero => glow::ZERO,
        super::StencilOp::Replace => glow::REPLACE,
        super::StencilOp::IncrementSaturate => glow::INCR,
        super::StencilOp::DecrementSaturate => glow::DECR,
        super::StencilOp::Invert => glow::INVERT,
        super::StencilOp::IncrementWrap => glow::INCR_WRAP,
        super::StencilOp::DecrementWrap => glow::DECR_WRAP,
    }
}

/// Convert swizzled PICA depth pixels to linear GL upload bytes. D24S8 needs the
/// packed GL order (24 depth bits followed by 8 stencil bits).
fn depth_to_linear(tiled: &[u8], width: u32, height: u32, bytes: u32) -> Vec<u8> {
    let gl_bytes = if bytes == 2 { 2 } else { 4 };
    let mut linear = vec![0; (width * height * gl_bytes) as usize];
    for y in 0..height {
        for x in 0..width {
            let from = morton_offset(x, height - 1 - y, width, bytes) as usize;
            let to = ((y * width + x) * gl_bytes) as usize;
            match bytes {
                2 => {
                    let value = u16::from_le_bytes(tiled[from..from + 2].try_into().unwrap());
                    linear[to..to + 2].copy_from_slice(&value.to_ne_bytes());
                }
                3 => {
                    let value =
                        u32::from_le_bytes([tiled[from], tiled[from + 1], tiled[from + 2], 0]);
                    linear[to..to + 4]
                        .copy_from_slice(&((value as f32 / 16_777_215.0).to_ne_bytes()));
                }
                _ => {
                    let value = u32::from_le_bytes(tiled[from..from + 4].try_into().unwrap());
                    let packed = ((value & 0x00FF_FFFF) << 8) | (value >> 24);
                    linear[to..to + 4].copy_from_slice(&packed.to_ne_bytes());
                }
            }
        }
    }
    linear
}

fn linear_to_tiled_bytes(linear: &[u8], tiled: &mut [u8], width: u32, height: u32, bytes: u32) {
    for y in 0..height {
        for x in 0..width {
            let from = ((y * width + x) * bytes) as usize;
            let to = morton_offset(x, height - 1 - y, width, bytes) as usize;
            if bytes == 4 {
                let packed = u32::from_ne_bytes(linear[from..from + 4].try_into().unwrap());
                let guest = (packed >> 8) | ((packed & 0xFF) << 24);
                tiled[to..to + 4].copy_from_slice(&guest.to_le_bytes());
            } else {
                tiled[to..to + bytes as usize]
                    .copy_from_slice(&linear[from..from + bytes as usize]);
            }
        }
    }
}

fn tiled_to_linear(tiled: &[u8], width: u32, height: u32) -> Vec<u8> {
    let mut linear = vec![0; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let source = morton_offset(x, height - 1 - y, width, 4) as usize;
            let color = crate::format::ColorFormat::Rgba8.decode(&tiled[source..source + 4]);
            let destination = ((y * width + x) * 4) as usize;
            linear[destination..destination + 4].copy_from_slice(&color);
        }
    }
    linear
}

fn linear_to_tiled(linear: &[u8], tiled: &mut [u8], width: u32, height: u32) {
    for y in 0..height {
        for x in 0..width {
            let source = ((y * width + x) * 4) as usize;
            let mut encoded = [0; 4];
            crate::format::ColorFormat::Rgba8.encode(
                linear[source..source + 4].try_into().expect("RGBA pixel"),
                &mut encoded,
            );
            let destination = morton_offset(x, height - 1 - y, width, 4) as usize;
            tiled[destination..destination + 4].copy_from_slice(&encoded);
        }
    }
}

fn version_at_least(version: &str, major: u32, minor: u32) -> bool {
    let mut parts = version.split('.');
    let Some(found_major) = parts.next().and_then(|part| part.parse::<u32>().ok()) else {
        return false;
    };
    let Some(found_minor) = parts.next().and_then(|part| {
        part.split(|character: char| !character.is_ascii_digit())
            .next()
            .and_then(|part| part.parse::<u32>().ok())
    }) else {
        return false;
    };
    (found_major, found_minor) >= (major, minor)
}

#[cfg(test)]
mod tests {
    use super::{
        depth_to_linear, linear_to_tiled, linear_to_tiled_bytes, tiled_to_linear, version_at_least,
    };

    #[test]
    fn recognizes_required_opengl_versions() {
        assert!(version_at_least("4.3.0 Intel", 4, 3));
        assert!(version_at_least("4.6 (Core Profile)", 4, 3));
        assert!(!version_at_least("4.2.0", 4, 3));
        assert!(!version_at_least("OpenGL ES 3.2", 4, 3));
    }

    #[test]
    fn tiled_rgba8_round_trips_through_linear_gl_rows() {
        for (width, height) in [(8, 8), (16, 8), (16, 16)] {
            let length = (width * height * 4) as usize;
            let tiled: Vec<u8> = (0..length).map(|i| (i * 37) as u8).collect();
            let linear = tiled_to_linear(&tiled, width, height);
            let mut restored = vec![0; length];
            linear_to_tiled(&linear, &mut restored, width, height);
            assert_eq!(restored, tiled);
        }
    }

    #[test]
    fn tiled_depth16_round_trips() {
        let (width, height) = (16, 8);
        let mut tiled = vec![0; (width * height * 2) as usize];
        for i in 0..width * height {
            tiled[(i * 2) as usize..(i * 2 + 2) as usize]
                .copy_from_slice(&((i * 997) as u16).to_le_bytes());
        }
        let linear = depth_to_linear(&tiled, width, height, 2);
        let mut restored = vec![0; tiled.len()];
        linear_to_tiled_bytes(&linear, &mut restored, width, height, 2);
        assert_eq!(restored, tiled);
    }

    #[test]
    fn tiled_depth24_stencil_round_trips() {
        let (width, height) = (16, 8);
        let mut tiled = vec![0; (width * height * 4) as usize];
        for i in 0..width * height {
            let value: u32 = ((i * 1237) & 0x00FF_FFFF) | ((i * 17 & 0xFF) << 24);
            tiled[(i * 4) as usize..(i * 4 + 4) as usize].copy_from_slice(&value.to_le_bytes());
        }
        let linear = depth_to_linear(&tiled, width, height, 4);
        let mut restored = vec![0; tiled.len()];
        linear_to_tiled_bytes(&linear, &mut restored, width, height, 4);
        assert_eq!(restored, tiled);
    }
}
