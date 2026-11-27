//! a small OpenGL rasterizer for untextured, vertex-colored triangles

use glow::HasContext;

use super::{ColorTarget, Screen};
use crate::{format::morton_offset, GpuMemory};

const VERTEX_SHADER: &str = r#"#version 330 core
layout(location = 0) in vec2 position_over_w;
layout(location = 1) in float inverse_w;
layout(location = 2) in vec4 color;
out vec4 vertex_color;
void main() {
    gl_Position = vec4(position_over_w, 0.0, inverse_w);
    vertex_color = color;
}
"#;

const FRAGMENT_SHADER: &str = r#"#version 330 core
in vec4 vertex_color;
out vec4 color;
void main() {
    color = floor(clamp(vertex_color, 0.0, 1.0) * 255.0) / 255.0;
}
"#;

/// OpenGL resources are created and used on the thread where the context is current.
pub(crate) struct OpenGlRasterizer {
    gl: std::sync::Arc<glow::Context>,
    program: glow::Program,
    vertex_array: glow::VertexArray,
    vertices: glow::Buffer,
    framebuffer: glow::Framebuffer,
    target: glow::Texture,
    size: (u32, u32),
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
            let framebuffer = gl.create_framebuffer().map_err(|error| error.to_string())?;
            let target = gl.create_texture().map_err(|error| error.to_string())?;

            gl.bind_vertex_array(Some(vertex_array));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vertices));
            let stride = (7 * size_of::<f32>()) as i32;
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
            gl.bind_vertex_array(None);

            gl.bind_texture(glow::TEXTURE_2D, Some(target));
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::NEAREST as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::NEAREST as i32,
            );
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(framebuffer));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(target),
                0,
            );
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);

            Ok(Self {
                gl,
                program,
                vertex_array,
                vertices,
                framebuffer,
                target,
                size: (0, 0),
            })
        }
    }

    pub(super) fn draw<M: GpuMemory>(
        &mut self,
        memory: &mut M,
        target: &ColorTarget,
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
        let mut tiled = vec![0; pixels];
        memory.read(target.addr, &mut tiled);
        let mut linear = tiled_to_linear(&tiled, width, height);

        let mut vertices = Vec::with_capacity(triangles.len() * 3 * 7);
        for triangle in triangles {
            for vertex in triangle {
                let inverse_w = vertex.inv_w;
                vertices.extend([
                    (2.0 * vertex.x / width as f32 - 1.0) * inverse_w,
                    (2.0 * vertex.y / height as f32 - 1.0) * inverse_w,
                    inverse_w,
                ]);
                vertices.extend(vertex.color_over_w.map(|component| component / inverse_w));
            }
        }
        let vertex_bytes = bytemuck::cast_slice(&vertices);

        unsafe {
            let gl = &self.gl;
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            gl.pixel_store_i32(glow::PACK_ALIGNMENT, 1);
            gl.bind_texture(glow::TEXTURE_2D, Some(self.target));
            if self.size != (width, height) {
                gl.tex_image_2d(
                    glow::TEXTURE_2D,
                    0,
                    glow::RGBA8 as i32,
                    width as i32,
                    height as i32,
                    0,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelUnpackData::Slice(Some(&linear)),
                );
                self.size = (width, height);
            } else {
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
            }

            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.framebuffer));
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
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::CULL_FACE);
            gl.disable(glow::DITHER);
            gl.color_mask(true, true, true, true);
            gl.use_program(Some(self.program));
            gl.bind_vertex_array(Some(self.vertex_array));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vertices));
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, vertex_bytes, glow::STREAM_DRAW);
            gl.draw_arrays(glow::TRIANGLES, 0, (vertices.len() / 7) as i32);
            gl.read_pixels(
                0,
                0,
                width as i32,
                height as i32,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(&mut linear)),
            );
            gl.bind_vertex_array(None);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.disable(glow::SCISSOR_TEST);
        }

        linear_to_tiled(&linear, &mut tiled, width, height);
        memory.write(target.addr, &tiled);
        Ok(())
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
    use super::{linear_to_tiled, tiled_to_linear, version_at_least};

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
}
