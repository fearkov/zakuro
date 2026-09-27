//! the library, the menu over a game and the settings, drawn with egui.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};

use egui::{Align, Color32, Layout, RichText, Vec2};
use winit::keyboard::KeyCode;

use crate::library::{Library, ICON_SIZE};
use crate::recompile::{Job, Stage};
use crate::settings::{Keys, Renderer, Settings};

/// what the user asked for.
#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    Play(PathBuf),
    /// recompile the library's game at this index.
    Recompile(usize),
    CancelRecompile(u64),
    ChooseFolder,
    ChooseRecompiler,
    ChooseBackground,
    Rescan,
    Resume,
    Reset,
    Library,
    Fullscreen,
    Quit,
    /// the settings changed, to save and apply.
    Settings,
}

#[derive(Default)]
pub struct Menus {
    /// the menu over a running game.
    pub menu_open: bool,
    pub settings_open: bool,
    /// the binding waiting for a key, by its place in Keys::all_mut.
    pub rebinding: Option<usize>,
    /// something to tell the user, until they close it.
    pub message: Option<String>,
    icons: HashMap<PathBuf, egui::TextureHandle>,
    background: Background,
}

/// the picture behind the library, decoded away from the interface.
#[derive(Default)]
struct Background {
    /// the file shown or being read.
    path: Option<PathBuf>,
    texture: Option<egui::TextureHandle>,
    loading: Option<Receiver<Result<egui::ColorImage, String>>>,
}

/// the longest side a background is kept at, bigger ones are scaled down.
const BACKGROUND_SIDE: u32 = 2560;

impl Background {
    /// the texture for path, starting to read it when it changed, none until
    /// it is ready or when there is no picture.
    fn get(&mut self, ctx: &egui::Context, path: Option<&Path>) -> Result<Option<&egui::TextureHandle>, String> {
        if self.path.as_deref() != path {
            self.path = path.map(Path::to_owned);
            self.texture = None;
            self.loading = path.map(|path| {
                let (send, receive) = channel();
                let path = path.to_owned();
                std::thread::spawn(move || {
                    let _ = send.send(decode(&path));
                });
                receive
            });
        }
        if let Some(receive) = &self.loading {
            match receive.try_recv() {
                Ok(image) => {
                    self.loading = None;
                    self.texture = Some(ctx.load_texture("library background", image?, egui::TextureOptions::LINEAR));
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => ctx.request_repaint(),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => self.loading = None,
            }
        }
        Ok(self.texture.as_ref())
    }
}

fn decode(path: &Path) -> Result<egui::ColorImage, String> {
    let image = image::open(path).map_err(|error| format!("could not open {}, {error}", path.display()))?;
    let image = if image.width().max(image.height()) > BACKGROUND_SIDE {
        image.resize(BACKGROUND_SIDE, BACKGROUND_SIDE, image::imageops::FilterType::Triangle)
    } else {
        image
    };
    let rgba = image.to_rgba8();
    Ok(egui::ColorImage::from_rgba_unmultiplied([rgba.width() as usize, rgba.height() as usize], rgba.as_raw()))
}

/// the part of a picture that fills an area without stretching it, cutting
/// off what sticks out on either side.
fn cover(area: Vec2, picture: [usize; 2]) -> egui::Rect {
    let (width, height) = (picture[0].max(1) as f32, picture[1].max(1) as f32);
    let area_ratio = area.x / area.y.max(1.0);
    let picture_ratio = width / height;
    if area_ratio > picture_ratio {
        // wider area, the picture's top and bottom are cut
        let shown = picture_ratio / area_ratio;
        egui::Rect::from_min_max(egui::pos2(0.0, (1.0 - shown) / 2.0), egui::pos2(1.0, (1.0 + shown) / 2.0))
    } else {
        let shown = area_ratio / picture_ratio;
        egui::Rect::from_min_max(egui::pos2((1.0 - shown) / 2.0, 0.0), egui::pos2((1.0 + shown) / 2.0, 1.0))
    }
}

impl Menus {
    /// the library, filling the window.
    pub fn library(&mut self, ui: &mut egui::Ui, library: &Library, settings: &Settings, jobs: &[Job]) -> Vec<Action> {
        let mut actions = Vec::new();
        if library.changed {
            self.icons.clear();
        }
        egui::Panel::top("library bar").show(ui, |ui| {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.heading("Zakuro");
                ui.separator();
                match &settings.games {
                    Some(folder) => ui.label(folder.display().to_string()),
                    None => ui.label(RichText::new("no games folder yet").weak()),
                };
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui.button("Settings").clicked() {
                        self.settings_open = true;
                    }
                    if ui.add_enabled(settings.games.is_some(), egui::Button::new("Refresh")).clicked() {
                        actions.push(Action::Rescan);
                    }
                    if ui.button("Choose folder…").clicked() {
                        actions.push(Action::ChooseFolder);
                    }
                });
            });
            ui.add_space(6.0);
        });
        let background = match self.background.get(ui.ctx(), settings.background.as_deref()) {
            Ok(texture) => texture.map(|texture| (texture.id(), texture.size())),
            Err(error) => {
                self.message = Some(error);
                None
            }
        };
        egui::CentralPanel::default().show(ui, |ui| {
            if let Some((texture, size)) = background {
                let area = ui.clip_rect();
                let alpha = (settings.background_opacity.clamp(0.0, 1.0) * 255.0).round() as u8;
                ui.painter().image(texture, area, cover(area.size(), size), Color32::from_white_alpha(alpha));
            }
            if settings.games.is_none() {
                ui.vertical_centered(|ui| {
                    ui.add_space(ui.available_height() / 3.0);
                    ui.label("Pick the folder your games are in to see them here.");
                    if ui.button("Choose folder…").clicked() {
                        actions.push(Action::ChooseFolder);
                    }
                });
                return;
            }
            if library.is_scanning() {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("Looking for games…");
                });
                return;
            }
            if library.games.is_empty() {
                ui.label("No .3ds, .cci or .cxi files in that folder.");
                return;
            }
            egui::ScrollArea::vertical().show(ui, |ui| {
                for (index, game) in library.games.iter().enumerate() {
                    let job = jobs.iter().find(|job| job.program_id == game.program_id);
                    // over a picture, each game gets a backing to keep it
                    // readable
                    let mut frame = egui::Frame::group(ui.style());
                    if background.is_some() {
                        frame = frame.fill(ui.visuals().panel_fill.gamma_multiply(0.85));
                    }
                    frame.show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.horizontal(|ui| {
                            let icon = game.icon.as_ref().map(|pixels| {
                                self.icons.entry(game.path.clone()).or_insert_with(|| {
                                    let image = egui::ColorImage::from_rgba_unmultiplied([ICON_SIZE, ICON_SIZE], pixels);
                                    ui.ctx().load_texture(game.path.display().to_string(), image, egui::TextureOptions::LINEAR)
                                })
                            });
                            match icon {
                                Some(texture) => {
                                    ui.add(egui::Image::new(&*texture).fit_to_exact_size(Vec2::splat(ICON_SIZE as f32)));
                                }
                                None => {
                                    ui.allocate_space(Vec2::splat(ICON_SIZE as f32));
                                }
                            }
                            ui.vertical(|ui| {
                                ui.label(RichText::new(&game.name).strong().size(16.0));
                                if !game.publisher.is_empty() {
                                    ui.label(&game.publisher);
                                }
                                ui.horizontal(|ui| {
                                    ui.label(RichText::new(format!("{:016X}", game.program_id)).weak().monospace());
                                    if game.recompiled {
                                        ui.label(RichText::new("recompiled").color(Color32::from_rgb(110, 200, 120)));
                                    }
                                });
                            });
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                if ui.button(RichText::new("▶ Play").size(15.0)).clicked() {
                                    actions.push(Action::Play(game.path.clone()));
                                }
                                match job {
                                    Some(job) if !job.stage().finished() => {
                                        if ui.button("Cancel").clicked() {
                                            actions.push(Action::CancelRecompile(job.program_id));
                                        }
                                        progress(ui, job);
                                    }
                                    _ => {
                                        let label = if game.recompiled { "Recompile again" } else { "Recompile" };
                                        if ui.button(label).on_hover_text("Turn the game's code into native code with 3dsrecomp, which makes it run faster. It takes around ten minutes.").clicked() {
                                            actions.push(Action::Recompile(index));
                                        }
                                        if let Some(Stage::Failed(error)) = job.map(Job::stage) {
                                            ui.label(RichText::new(format!("failed, {error}")).color(Color32::LIGHT_RED));
                                        }
                                    }
                                }
                            });
                        });
                    });
                }
            });
        });
        actions
    }

    /// what goes over a running game, the menu when it is open and the
    /// frame rate when asked for.
    pub fn game(&mut self, ui: &mut egui::Ui, name: &str, fps: Option<f32>, jobs: &[Job]) -> Vec<Action> {
        let mut actions = Vec::new();
        let ctx = ui.ctx().clone();
        if let Some(fps) = fps {
            egui::Area::new(egui::Id::new("fps")).fixed_pos(egui::pos2(8.0, 8.0)).show(&ctx, |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.label(RichText::new(format!("{fps:.0} fps")).monospace());
                });
            });
        }
        if !self.menu_open {
            return actions;
        }
        egui::Window::new("Paused")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
            .show(&ctx, |ui| {
                ui.label(RichText::new(name).strong());
                ui.add_space(6.0);
                let wide = |ui: &mut egui::Ui, text: &str| ui.add_sized([200.0, 28.0], egui::Button::new(text)).clicked();
                if wide(ui, "Resume") {
                    actions.push(Action::Resume);
                }
                if wide(ui, "Reset") {
                    actions.push(Action::Reset);
                }
                if wide(ui, "Fullscreen") {
                    actions.push(Action::Fullscreen);
                }
                if wide(ui, "Settings") {
                    self.settings_open = true;
                }
                if wide(ui, "Back to the library") {
                    actions.push(Action::Library);
                }
                if wide(ui, "Quit") {
                    actions.push(Action::Quit);
                }
                for job in jobs {
                    ui.separator();
                    ui.label(&job.name);
                    progress(ui, job);
                }
            });
        actions
    }

    /// the settings window, when open.
    pub fn settings(&mut self, ctx: &egui::Context, settings: &mut Settings, recompiler: Option<&std::path::Path>) -> Vec<Action> {
        let mut actions = Vec::new();
        let before = settings.clone();
        let mut open = self.settings_open;
        egui::Window::new("Settings").open(&mut open).resizable(false).show(ctx, |ui| {
            ui.heading("Games");
            ui.horizontal(|ui| {
                match &settings.games {
                    Some(folder) => ui.label(folder.display().to_string()),
                    None => ui.label(RichText::new("none").weak()),
                };
                if ui.button("Choose…").clicked() {
                    actions.push(Action::ChooseFolder);
                }
            });

            ui.separator();
            ui.heading("Graphics");
            egui::ComboBox::from_label("Presented with")
                .selected_text(match settings.renderer {
                    Renderer::Vulkan => "Vulkan",
                    Renderer::OpenGl => "OpenGL",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut settings.renderer, Renderer::Vulkan, "Vulkan");
                    ui.selectable_value(&mut settings.renderer, Renderer::OpenGl, "OpenGL");
                });
            ui.checkbox(&mut settings.hardware_rasterizer, "Draw the 3D on the GPU");
            ui.add(egui::Slider::new(&mut settings.scale, 1..=6).text("Window scale"));
            ui.label(RichText::new("The presenter changes the next time Zakuro starts, the 3D with the next game.").weak());

            ui.separator();
            ui.heading("Sound");
            ui.add(egui::Slider::new(&mut settings.volume, 0.0..=1.0).text("Volume").show_value(false));
            ui.checkbox(&mut settings.mute, "Mute");

            ui.separator();
            ui.heading("Interface");
            ui.checkbox(&mut settings.show_fps, "Show the frame rate");
            ui.horizontal(|ui| {
                ui.label("Library background");
                match &settings.background {
                    Some(path) => ui.label(path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default()),
                    None => ui.label(RichText::new("none").weak()),
                };
                if ui.button("Choose…").clicked() {
                    actions.push(Action::ChooseBackground);
                }
                if settings.background.is_some() && ui.button("Remove").clicked() {
                    settings.background = None;
                }
            });
            ui.add_enabled(
                settings.background.is_some(),
                egui::Slider::new(&mut settings.background_opacity, 0.0..=1.0)
                    .text("Opacity")
                    .custom_formatter(|value, _| format!("{:.0}%", value * 100.0)),
            );

            ui.separator();
            ui.heading("3dsrecomp");
            ui.horizontal(|ui| {
                match recompiler {
                    Some(path) => ui.label(path.display().to_string()),
                    None => ui.label(RichText::new("not found, choose where it is").color(Color32::LIGHT_RED)),
                };
                if ui.button("Choose…").clicked() {
                    actions.push(Action::ChooseRecompiler);
                }
            });

            ui.separator();
            ui.heading("Controls");
            egui::Grid::new("controls").num_columns(4).spacing([12.0, 4.0]).show(ui, |ui| {
                for (i, (name, key)) in settings.keys.all_mut().into_iter().enumerate() {
                    ui.label(name);
                    let text = if self.rebinding == Some(i) { "press a key…".to_owned() } else { key_name(*key) };
                    if ui.add_sized([110.0, 20.0], egui::Button::new(text)).clicked() {
                        self.rebinding = Some(i);
                    }
                    if i % 2 == 1 {
                        ui.end_row();
                    }
                }
            });
            if ui.button("Default controls").clicked() {
                settings.keys = Keys::default();
                self.rebinding = None;
            }
        });
        self.settings_open = open;
        if !open {
            self.rebinding = None;
        }
        if *settings != before {
            actions.push(Action::Settings);
        }
        actions
    }

    /// a message box, when there is something to tell.
    pub fn message(&mut self, ctx: &egui::Context) {
        let Some(text) = self.message.clone() else { return };
        egui::Window::new("Zakuro")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
            .show(ctx, |ui| {
                ui.label(text);
                if ui.button("OK").clicked() {
                    self.message = None;
                }
            });
    }
}

/// a recompile's progress bar with where it is.
fn progress(ui: &mut egui::Ui, job: &Job) {
    let stage = job.stage();
    let text = match &stage {
        Stage::Generating => "finding the code".to_owned(),
        Stage::Compiling { done, total } => format!("compiling {done} of {total}"),
        Stage::Installing => "installing".to_owned(),
        Stage::Done => "done".to_owned(),
        Stage::Failed(error) => format!("failed, {error}"),
    };
    let minutes = job.elapsed().as_secs() / 60;
    let seconds = job.elapsed().as_secs() % 60;
    ui.add(egui::ProgressBar::new(stage.fraction()).desired_width(220.0).text(format!("{text}, {minutes}:{seconds:02}")));
}

/// a key as a person would call it.
pub fn key_name(key: KeyCode) -> String {
    let name = format!("{key:?}");
    for prefix in ["Key", "Digit", "Arrow"] {
        if let Some(rest) = name.strip_prefix(prefix) {
            return rest.to_owned();
        }
    }
    name
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_background_fills_the_area_without_stretching() {
        // a square picture in a wide area loses its top and bottom
        let wide = cover(Vec2::new(200.0, 100.0), [100, 100]);
        assert_eq!((wide.min.x, wide.max.x), (0.0, 1.0));
        assert_eq!((wide.min.y, wide.max.y), (0.25, 0.75));
        // and in a tall one its sides
        let tall = cover(Vec2::new(100.0, 200.0), [100, 100]);
        assert_eq!((tall.min.x, tall.max.x), (0.25, 0.75));
        assert_eq!((tall.min.y, tall.max.y), (0.0, 1.0));
    }

    #[test]
    fn big_pictures_are_scaled_down_keeping_their_shape() {
        let path = std::env::temp_dir().join(format!("zakuro-background-{}.png", std::process::id()));
        image::RgbaImage::from_pixel(BACKGROUND_SIDE * 2, BACKGROUND_SIDE, image::Rgba([10, 20, 30, 255])).save(&path).unwrap();
        let decoded = decode(&path);
        std::fs::remove_file(&path).ok();
        let decoded = decoded.unwrap();
        assert_eq!(decoded.size, [BACKGROUND_SIDE as usize, BACKGROUND_SIDE as usize / 2]);
        assert_eq!(decoded.pixels[0].to_array(), [10, 20, 30, 255]);
        assert!(decode(Path::new("/nowhere/at/all.png")).is_err());
    }

    #[test]
    fn keys_have_short_names() {
        assert_eq!(key_name(KeyCode::KeyX), "X");
        assert_eq!(key_name(KeyCode::ArrowUp), "Up");
        assert_eq!(key_name(KeyCode::Digit1), "1");
        assert_eq!(key_name(KeyCode::Enter), "Enter");
    }
}
