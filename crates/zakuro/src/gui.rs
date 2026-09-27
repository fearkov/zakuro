//! egui, fed the window's events and turned into an overlay the presenters
//! draw over the screens.

use egui::epaint::Primitive;
use egui::{TextureId, ViewportId};
use winit::event::WindowEvent;
use winit::window::Window;

use zakuro_gpu::{Overlay, OverlayMesh, OverlayTexture, OverlayVertex};

pub struct Gui {
    pub ctx: egui::Context,
    state: egui_winit::State,
}

impl Gui {
    pub fn new(window: &Window) -> Gui {
        let ctx = egui::Context::default();
        let state = egui_winit::State::new(
            ctx.clone(),
            ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            None,
            Some(8192),
        );
        Gui { ctx, state }
    }

    /// hands egui an event, true when it wants it for itself.
    pub fn event(&mut self, window: &Window, event: &WindowEvent) -> bool {
        self.state.on_window_event(window, event).consumed
    }

    /// whether egui is using the keyboard or the pointer, in which case the
    /// game should not see them.
    pub fn wants_keyboard(&self) -> bool {
        self.ctx.egui_wants_keyboard_input()
    }

    pub fn wants_pointer(&self) -> bool {
        self.ctx.egui_wants_pointer_input()
    }

    /// runs the interface for a frame and returns what to draw.
    pub fn frame(&mut self, window: &Window, ui: impl FnMut(&mut egui::Ui)) -> Overlay {
        let input = self.state.take_egui_input(window);
        let mut output = self.ctx.run_ui(input, ui);
        self.state.handle_platform_output(window, output.platform_output);
        let pixels_per_point = output.pixels_per_point;
        let size = window.inner_size();

        let textures = std::mem::take(&mut output.textures_delta.set)
            .into_iter()
            .flat_map(|(id, deltas)| {
                deltas.into_iter().map(move |delta| {
                    let egui::ImageData::Color(image) = &delta.image;
                    OverlayTexture {
                        id: texture(id),
                        offset: delta.pos.map(|[x, y]| [x as u32, y as u32]),
                        size: [image.width() as u32, image.height() as u32],
                        pixels: image.pixels.iter().flat_map(|color| color.to_array()).collect(),
                        linear: delta.options.magnification == egui::TextureFilter::Linear,
                    }
                })
            })
            .collect();
        let meshes = self
            .ctx
            .tessellate(output.shapes, pixels_per_point)
            .into_iter()
            .filter_map(|clipped| {
                let Primitive::Mesh(mesh) = clipped.primitive else { return None };
                let rect = clipped.clip_rect;
                let pixel = |value: f32, most: u32| ((value * pixels_per_point).round().max(0.0) as u32).min(most);
                Some(OverlayMesh {
                    texture: texture(mesh.texture_id),
                    clip: [
                        pixel(rect.min.x, size.width),
                        pixel(rect.min.y, size.height),
                        pixel(rect.max.x, size.width),
                        pixel(rect.max.y, size.height),
                    ],
                    vertices: mesh
                        .vertices
                        .iter()
                        .map(|vertex| OverlayVertex {
                            position: [vertex.pos.x * pixels_per_point, vertex.pos.y * pixels_per_point],
                            uv: [vertex.uv.x, vertex.uv.y],
                            color: vertex.color.to_array(),
                        })
                        .collect(),
                    indices: mesh.indices,
                })
            })
            .collect();
        let free = std::mem::take(&mut output.textures_delta.free).into_iter().map(texture).collect();
        Overlay { textures, meshes, free }
    }
}

/// egui's two kinds of texture in one space of ids.
fn texture(id: TextureId) -> u64 {
    match id {
        TextureId::Managed(n) => n * 2,
        TextureId::User(n) => n * 2 + 1,
    }
}
