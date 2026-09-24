//! the rendering backend interface.

/// everything a draw needs, handed over without copying the register file.
pub struct DrawCall<'a> {
    pub indexed: bool,
    pub registers: &'a [u32],
}

/// which backend a frontend asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RendererKind {
    /// no 3D rasterization, the framebuffers are presented as the fill and
    /// transfer engines leave them.
    #[default]
    Software,
    OpenGl,
    Vulkan,
}

impl RendererKind {
    pub fn parse(name: &str) -> Option<RendererKind> {
        match name.to_ascii_lowercase().as_str() {
            "software" | "none" | "null" => Some(RendererKind::Software),
            "opengl" | "gl" => Some(RendererKind::OpenGl),
            "vulkan" | "vk" => Some(RendererKind::Vulkan),
            _ => None,
        }
    }
}

pub trait Renderer {
    /// name shown in logs and in the window title.
    fn name(&self) -> &'static str;

    /// executes a draw with the current register state.
    fn draw(&mut self, call: DrawCall<'_>);

    /// one word of vertex shader program upload.
    fn upload_shader_code(&mut self, word: u32) {
        let _ = word;
    }

    /// one word of vertex shader operand descriptor upload.
    fn upload_shader_operand_descriptor(&mut self, word: u32) {
        let _ = word;
    }

    /// called once per emulated frame, after both screens have been
    /// transferred, so a deferred backend can flush.
    fn end_frame(&mut self) {}
}

/// the do-nothing backend.
#[derive(Default)]
pub struct SoftwareRenderer {
    pub draws: u64,
}

impl Renderer for SoftwareRenderer {
    fn name(&self) -> &'static str {
        "software"
    }

    fn draw(&mut self, _call: DrawCall<'_>) {
        self.draws += 1;
    }
}
