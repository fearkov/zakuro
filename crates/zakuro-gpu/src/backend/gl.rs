//! OpenGL presentation backend.

use glow::HasContext;

use super::{layout, PresentError, Presenter, ScreenImage, Viewport};

const VERTEX_SHADER: &str = r#"#version 330 core
// a single oversized triangle covers the viewport with no vertex buffer.
out vec2 uv;
void main() {
    vec2 position = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    uv = vec2(position.x, 1.0 - position.y);
    gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
}
"#;

const FRAGMENT_SHADER: &str = r#"#version 330 core
in vec2 uv;
out vec4 color;
uniform sampler2D screen;
void main() {
    color = vec4(texture(screen, uv).rgb, 1.0);
}
"#;

pub struct GlPresenter {
    gl: glow::Context,
    program: glow::Program,
    vertex_array: glow::VertexArray,
    /// index 0 is the top screen, 1 the bottom.
    textures: [glow::Texture; 2],
    /// dimensions each texture was last allocated at, so uploads can use
    /// tex_sub_image_2d when nothing changed.
    sizes: [(u32, u32); 2],
    window: (u32, u32),
}

impl GlPresenter {
    /// loader resolves OpenGL function names, as the windowing library
    /// provides.
    ///
    /// # Safety
    ///
    /// A current OpenGL 3.3 context must be bound on this thread.
    pub unsafe fn new(
        loader: impl FnMut(&str) -> *const std::ffi::c_void,
        window: (u32, u32),
    ) -> Result<GlPresenter, PresentError> {
        let gl = unsafe { glow::Context::from_loader_function(loader) };

        let version = unsafe { gl.get_parameter_string(glow::VERSION) };
        let renderer = unsafe { gl.get_parameter_string(glow::RENDERER) };
        log::info!("OpenGL {version} on {renderer}");

        let program = unsafe { link_program(&gl, VERTEX_SHADER, FRAGMENT_SHADER)? };
        let vertex_array = unsafe { gl.create_vertex_array() }
            .map_err(PresentError::Backend)?;

        let mut textures = Vec::with_capacity(2);
        for _ in 0..2 {
            let texture = unsafe { gl.create_texture() }.map_err(PresentError::Backend)?;
            unsafe {
                gl.bind_texture(glow::TEXTURE_2D, Some(texture));
                // nearest keeps the 3DS's pixels crisp when scaled up.
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
                gl.tex_parameter_i32(
                    glow::TEXTURE_2D,
                    glow::TEXTURE_WRAP_S,
                    glow::CLAMP_TO_EDGE as i32,
                );
                gl.tex_parameter_i32(
                    glow::TEXTURE_2D,
                    glow::TEXTURE_WRAP_T,
                    glow::CLAMP_TO_EDGE as i32,
                );
            }
            textures.push(texture);
        }

        Ok(GlPresenter {
            gl,
            program,
            vertex_array,
            textures: [textures[0], textures[1]],
            sizes: [(0, 0); 2],
            window,
        })
    }

    fn upload(&mut self, index: usize, image: &ScreenImage<'_>) {
        let gl = &self.gl;
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(self.textures[index]));
            if self.sizes[index] != (image.width, image.height) {
                gl.tex_image_2d(
                    glow::TEXTURE_2D,
                    0,
                    glow::RGBA8 as i32,
                    image.width as i32,
                    image.height as i32,
                    0,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelUnpackData::Slice(Some(image.pixels)),
                );
                self.sizes[index] = (image.width, image.height);
            } else {
                gl.tex_sub_image_2d(
                    glow::TEXTURE_2D,
                    0,
                    0,
                    0,
                    image.width as i32,
                    image.height as i32,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelUnpackData::Slice(Some(image.pixels)),
                );
            }
        }
    }

    fn draw_screen(&self, index: usize, viewport: Viewport) {
        let gl = &self.gl;
        unsafe {
            // OpenGL's origin is the bottom left, so the y coordinate flips.
            let y = self.window.1 as f32 - viewport.y - viewport.height;
            gl.viewport(
                viewport.x as i32,
                y as i32,
                viewport.width as i32,
                viewport.height as i32,
            );
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(self.textures[index]));
            gl.draw_arrays(glow::TRIANGLES, 0, 3);
        }
    }
}

impl Presenter for GlPresenter {
    fn name(&self) -> &'static str {
        "opengl"
    }

    fn present(
        &mut self,
        top: ScreenImage<'_>,
        bottom: ScreenImage<'_>,
    ) -> Result<(), PresentError> {
        if !top.is_empty() {
            self.upload(0, &top);
        }
        if !bottom.is_empty() {
            self.upload(1, &bottom);
        }

        let gl = &self.gl;
        unsafe {
            gl.viewport(0, 0, self.window.0 as i32, self.window.1 as i32);
            gl.clear_color(0.0, 0.0, 0.0, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
            gl.use_program(Some(self.program));
            gl.bind_vertex_array(Some(self.vertex_array));
            if let Some(location) = gl.get_uniform_location(self.program, "screen") {
                gl.uniform_1_i32(Some(&location), 0);
            }
        }

        let (top_viewport, bottom_viewport) = layout(self.window.0, self.window.1);
        if !top.is_empty() {
            self.draw_screen(0, top_viewport);
        }
        if !bottom.is_empty() {
            self.draw_screen(1, bottom_viewport);
        }

        unsafe {
            self.gl.bind_vertex_array(None);
        }
        Ok(())
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.window = (width.max(1), height.max(1));
    }
}

impl Drop for GlPresenter {
    fn drop(&mut self) {
        unsafe {
            self.gl.delete_program(self.program);
            self.gl.delete_vertex_array(self.vertex_array);
            for texture in self.textures {
                self.gl.delete_texture(texture);
            }
        }
    }
}

unsafe fn link_program(
    gl: &glow::Context,
    vertex: &str,
    fragment: &str,
) -> Result<glow::Program, PresentError> {
    let program = unsafe { gl.create_program() }.map_err(PresentError::Backend)?;

    let mut shaders = Vec::new();
    for (kind, source) in [
        (glow::VERTEX_SHADER, vertex),
        (glow::FRAGMENT_SHADER, fragment),
    ] {
        let shader = unsafe { gl.create_shader(kind) }.map_err(PresentError::Backend)?;
        unsafe {
            gl.shader_source(shader, source);
            gl.compile_shader(shader);
            if !gl.get_shader_compile_status(shader) {
                let log = gl.get_shader_info_log(shader);
                return Err(PresentError::Backend(format!(
                    "shader failed to compile: {log}"
                )));
            }
            gl.attach_shader(program, shader);
        }
        shaders.push(shader);
    }

    unsafe {
        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            let log = gl.get_program_info_log(program);
            return Err(PresentError::Backend(format!(
                "program failed to link: {log}"
            )));
        }
        for shader in shaders {
            gl.detach_shader(program, shader);
            gl.delete_shader(shader);
        }
    }
    Ok(program)
}
