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
        ctx.set_fonts(fonts(system_fonts()));
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

/// egui's own fonts, then extra ones to fall back on for what they lack.
/// game names come in Japanese, Korean and Chinese, whose characters
/// egui's fonts have none of and drew as boxes.
fn fonts(extra: Vec<Vec<u8>>) -> egui::FontDefinitions {
    let mut fonts = egui::FontDefinitions::default();
    for (i, font) in extra.into_iter().enumerate() {
        let name = format!("system {i}");
        fonts.font_data.insert(name.clone(), std::sync::Arc::new(egui::FontData::from_owned(font)));
        for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
            fonts.families.entry(family).or_default().push(name.clone());
        }
    }
    fonts
}

/// the system's fonts for Japanese, Korean and Chinese, the first one
/// there of each kind.
fn system_fonts() -> Vec<Vec<u8>> {
    font_candidates().iter().filter_map(|paths| paths.iter().find_map(|path| std::fs::read(path).ok())).collect()
}

/// where Windows keeps fonts for Japanese, Korean and Chinese, each kind
/// the ones to try in turn.
#[cfg(windows)]
fn font_candidates() -> Vec<Vec<std::path::PathBuf>> {
    let folder = std::path::PathBuf::from(std::env::var_os("WINDIR").unwrap_or_else(|| "C:\\Windows".into())).join("Fonts");
    let kinds: [&[&str]; 3] = [&["YuGothM.ttc", "meiryo.ttc", "msgothic.ttc"], &["malgun.ttf"], &["msyh.ttc", "simsun.ttc"]];
    kinds.iter().map(|names| names.iter().map(|name| folder.join(name)).collect()).collect()
}

/// where macOS keeps fonts for Japanese, Korean and Chinese.
#[cfg(target_os = "macos")]
fn font_candidates() -> Vec<Vec<std::path::PathBuf>> {
    let kinds: [&[&str]; 3] = [
        &["/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc"],
        &["/System/Library/Fonts/AppleSDGothicNeo.ttc"],
        &["/System/Library/Fonts/PingFang.ttc"],
    ];
    kinds.iter().map(|paths| paths.iter().map(std::path::PathBuf::from).collect()).collect()
}

/// where Linux distributions put Noto's collection, which has Japanese,
/// Korean and Chinese all in one, or else the font fontconfig picks for
/// Japanese.
#[cfg(all(unix, not(target_os = "macos")))]
fn font_candidates() -> Vec<Vec<std::path::PathBuf>> {
    let mut paths: Vec<std::path::PathBuf> = [
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    .iter()
    .map(std::path::PathBuf::from)
    .collect();
    if !paths.iter().any(|path| path.exists()) {
        let matched = std::process::Command::new("fc-match").args(["--format=%{file}", ":lang=ja"]).output();
        if let Some(output) = matched.ok().filter(|output| output.status.success()) {
            paths.push(String::from_utf8_lossy(&output.stdout).into_owned().into());
        }
    }
    vec![paths]
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::epaint::text::{Fonts, TextOptions};

    /// Japanese names have every character to draw with the system's
    /// fonts, where the system has them, and none with egui's alone. only
    /// their Japanese characters are asked about, epaint says no for any
    /// character the font with its replacement glyph has, Hack's letters
    /// for monospace among them.
    #[test]
    fn japanese_names_are_drawn_with_the_system_fonts() {
        let name = "イナズマイレブン・クロノストーン";
        let mut plain = Fonts::new(TextOptions::default(), fonts(Vec::new()));
        assert!(!plain.has_glyphs(&egui::FontId::proportional(14.0), name), "egui's own fonts have no Japanese");
        let found = system_fonts();
        if found.is_empty() {
            eprintln!("no font for Japanese on this system");
            return;
        }
        let mut with = Fonts::new(TextOptions::default(), fonts(found));
        for font in [egui::FontId::proportional(14.0), egui::FontId::monospace(14.0)] {
            assert!(with.has_glyphs(&font, name), "{font:?}");
        }
    }
}
