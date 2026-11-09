//! the library, the menu over a game and the settings, drawn with egui.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};

use egui::{Align, Color32, Layout, RichText, Vec2};
use winit::keyboard::KeyCode;
use zakuro_core::services::keyboard::Request;

use crate::library::{Game, Library, ICON_SIZE};
use crate::recompile::{Job, Stage, Wanted};
use crate::gamepad::button_name;
use crate::settings::{
    game_key, CameraSource, Filter, GameSettings, Keys, Language, MicrophoneSource, Orientation, PadButtons, Renderer, Screens, Settings, Stereo3d, TopShape,
};

/// what the user asked for.
#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    Play(PathBuf),
    /// recompile these games of the library, one after the other.
    Recompile(Vec<Wanted>),
    /// download a compiler, then recompile these games of the library with
    /// it.
    DownloadCompiler(Vec<Wanted>),
    /// show the mods folder of the game with this program id.
    Mods(u64),
    /// stop recompiling the game with this id, or waiting to.
    CancelRecompile(u64),
    /// leave only the game recompiling now, none of those waiting.
    CancelWaiting,
    ChooseFolder,
    ChooseBackground,
    /// pick the picture the cameras see.
    ChooseCameraPicture,
    /// save the game's state to the slot chosen, or load it from there.
    SaveState,
    LoadState,
    /// open a room and join it, join another's, or leave the one joined.
    HostRoom,
    JoinRoom,
    LeaveRoom,
    /// take part in the Download Play offer at this place in the list, or
    /// leave the one taken part in.
    DlpJoin(usize),
    DlpLeave,
    Rescan,
    Resume,
    Reset,
    Library,
    Fullscreen,
    /// copy what a report on the game wants.
    CopyReport,
    Quit,
    /// the settings changed, to save and apply.
    Settings,
    /// close the game's keyboard with this text and the button pressed.
    Keyboard(String, usize),
    /// turn the running game's cheat at this index on or off.
    Cheat(usize, bool),
    /// add a cheat, its name and its code, to the running game.
    AddCheat(String, String),
    /// read the running game's cheats from their file again.
    ReloadCheats,
    /// show the cheats folder.
    CheatsFolder,
}

#[derive(Default)]
pub struct Menus {
    /// the menu over a running game.
    pub menu_open: bool,
    /// the cheats window, and the cheat being written in it.
    pub cheats_open: bool,
    cheat_name: String,
    cheat_code: String,
    pub settings_open: bool,
    /// the game whose own settings are open, its id and its name.
    pub game_settings: Option<(u64, String)>,
    /// the binding waiting for a key, by its place in Keys::all_mut.
    pub rebinding: Option<usize>,
    /// the controller binding waiting for a button, by its place in
    /// PadButtons::all_mut.
    pub rebinding_pad: Option<usize>,
    /// something to tell the user, until they close it.
    pub message: Option<String>,
    /// something to tell the user for a moment, over the game, and since
    /// when.
    pub toast: Option<(String, std::time::Instant)>,
    /// the slot states are saved to and loaded from, 1 to 9.
    pub slot: u8,
    /// the multiplayer window, and the password typed in it.
    pub multiplayer_open: bool,
    pub room_password: String,
    /// the icons of the games offered for Download Play.
    dlp_icons: HashMap<u64, egui::TextureHandle>,
    /// a report on the game, to copy from the message it goes with, and
    /// from no other.
    pub report: Option<(String, String)>,
    /// what is being typed into the game's keyboard, and what it asked for.
    typing: Option<(Request, String)>,
    /// the library's games to recompile once the player agrees to download
    /// a compiler for them.
    pub compiler_offer: Option<Vec<Wanted>>,
    icons: HashMap<PathBuf, egui::TextureHandle>,
    background: Background,
}

/// the room the player is in, as the multiplayer window shows it.
pub struct RoomView {
    pub name: String,
    /// each member's nickname and the game they play.
    pub members: Vec<(String, String)>,
    /// the port of the room this program holds open, when it is its own.
    pub port: Option<u16>,
    /// Download Play, while no game runs.
    pub download_play: Option<DlpView>,
}

/// the Download Play offers in the room, and where the guest is.
pub struct DlpView {
    pub offers: Vec<OfferView>,
    /// what the guest is doing, and how far along, while it takes part.
    pub doing: Option<(String, Option<f32>)>,
}

pub struct OfferView {
    pub child: u64,
    pub title: String,
    pub host: String,
    /// RGBA, ICON_SIZE square.
    pub icon: Option<Vec<u8>>,
    pub guests: u8,
    pub most: u8,
    /// why the player can't take part, when they can't.
    pub blocked: Option<String>,
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
    /// whether something is open that stops the game while it is.
    pub fn over_game(&self) -> bool {
        self.menu_open || self.settings_open || self.game_settings.is_some()
    }

    /// the library, filling the window. waiting are the games waiting for
    /// their turn to recompile.
    pub fn library(&mut self, ui: &mut egui::Ui, library: &Library, settings: &Settings, jobs: &[Job], waiting: &[Wanted]) -> Vec<Action> {
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
                    if ui.button("Multiplayer").clicked() {
                        self.multiplayer_open = true;
                    }
                    if ui.add_enabled(settings.games.is_some(), egui::Button::new("Refresh")).clicked() {
                        actions.push(Action::Rescan);
                    }
                    if ui.button("Choose folder…").clicked() {
                        actions.push(Action::ChooseFolder);
                    }
                    let outdated = idle(library.games.iter().filter(|game| game.outdated).map(Game::wanted), jobs, waiting);
                    if !outdated.is_empty() {
                        let recompile = ui.button(format!("Recompile outdated ({})", outdated.len())).on_hover_text(
                            "These games were recompiled by an older version of Zakuro. Recompiling them again gives them its improvements. They go one at a time, around ten minutes each, and games can be played meanwhile.",
                        );
                        if recompile.clicked() {
                            actions.push(Action::Recompile(outdated));
                        }
                    }
                    if !waiting.is_empty() && ui.button(format!("Cancel the {} waiting", waiting.len())).on_hover_text("The game recompiling now carries on.").clicked() {
                        actions.push(Action::CancelWaiting);
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
                ui.label("No .3ds, .cci, .cxi or .cia files in that folder.");
                return;
            }
            egui::ScrollArea::vertical().show(ui, |ui| {
                for game in &library.games {
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
                                if let Some(problem) = &game.problem {
                                    ui.label(RichText::new(problem).color(Color32::from_rgb(230, 160, 80)));
                                    return;
                                }
                                ui.horizontal(|ui| {
                                    ui.label(RichText::new(format!("{:016X}", game.program_id)).weak().monospace());
                                    if game.outdated {
                                        ui.label(RichText::new("recompiled by an older version").color(Color32::from_rgb(220, 180, 100)));
                                    } else if game.recompiled {
                                        ui.label(RichText::new("recompiled").color(Color32::from_rgb(110, 200, 120)));
                                    }
                                    if game.modded {
                                        ui.label(RichText::new("mods").color(Color32::from_rgb(120, 170, 230)));
                                    }
                                    if settings.has_own(game.program_id) {
                                        ui.label(RichText::new("own settings").color(Color32::from_rgb(120, 170, 230)));
                                    }
                                    if let Some(update) = &game.update {
                                        let version = zakuro_fs::version_name(update.version);
                                        ui.label(RichText::new(format!("update {version}")).color(Color32::from_rgb(220, 180, 100)));
                                    }
                                    if !game.dlc.is_empty() {
                                        ui.label(RichText::new("DLC").color(Color32::from_rgb(220, 180, 100)));
                                    }
                                });
                            });
                            if game.problem.is_some() {
                                return;
                            }
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
                                    _ if waiting.iter().any(|wanted| wanted.program_id == game.program_id) => {
                                        if ui.button("Cancel").clicked() {
                                            actions.push(Action::CancelRecompile(game.program_id));
                                        }
                                        ui.label(RichText::new("waiting to recompile").weak());
                                    }
                                    _ => {
                                        let label = if game.recompiled { "Recompile again" } else { "Recompile" };
                                        if ui.button(label).on_hover_text("Turn the game's code into native code with 3dsrecomp, which makes it run faster. It takes around ten minutes and needs a C compiler, gcc or clang, or Zig, which Zakuro offers to download when the computer has neither.").clicked() {
                                            actions.push(Action::Recompile(vec![game.wanted()]));
                                        }
                                        if let Some(Stage::Failed(error)) = job.map(Job::stage) {
                                            ui.label(RichText::new(format!("failed, {error}")).color(Color32::LIGHT_RED));
                                        }
                                    }
                                }
                                if ui.button("Mods").on_hover_text("Open the game's mods folder. Files put in its romfs folder replace the game's, the way Luma3DS and Citra take mods, and a texture pack made for Citra or Azahar goes in its textures folder.").clicked() {
                                    actions.push(Action::Mods(game.program_id));
                                }
                                if ui.button("Settings").on_hover_text("What this game has of its own, over the general settings.").clicked() {
                                    self.game_settings = Some((game.program_id, game.name.clone()));
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
    #[allow(clippy::too_many_arguments)]
    pub fn game(
        &mut self,
        ui: &mut egui::Ui,
        name: &str,
        program_id: Option<u64>,
        fps: Option<f32>,
        recompiled: bool,
        fast: bool,
        jobs: &[Job],
        waiting: usize,
        saved: Option<&str>,
    ) -> Vec<Action> {
        let mut actions = Vec::new();
        let ctx = ui.ctx().clone();
        if let Some((text, since)) = &self.toast {
            if since.elapsed() < std::time::Duration::from_millis(2500) {
                egui::Area::new(egui::Id::new("toast"))
                    .anchor(egui::Align2::CENTER_TOP, Vec2::new(0.0, 8.0))
                    .interactable(false)
                    .show(&ctx, |ui| {
                        egui::Frame::popup(ui.style()).show(ui, |ui| {
                            ui.label(text);
                        });
                    });
                ctx.request_repaint();
            } else {
                self.toast = None;
            }
        }
        if fast {
            egui::Area::new(egui::Id::new("fast forward"))
                .anchor(egui::Align2::RIGHT_TOP, Vec2::new(-8.0, 8.0))
                .interactable(false)
                .show(&ctx, |ui| {
                    egui::Frame::popup(ui.style()).show(ui, |ui| {
                        ui.label(RichText::new("» fast forward").monospace());
                    });
                });
        }
        if let Some(fps) = fps {
            // clicks go through to the screen under it
            egui::Area::new(egui::Id::new("fps")).fixed_pos(egui::pos2(8.0, 8.0)).interactable(false).show(&ctx, |ui| {
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
                ui.label(RichText::new(if recompiled { "running its recompiled code" } else { "interpreted" }).weak());
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
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut self.slot).range(1..=9).prefix("Slot "));
                    ui.label(RichText::new(saved.unwrap_or("empty")).weak());
                });
                ui.horizontal(|ui| {
                    if ui.add_sized([98.0, 28.0], egui::Button::new("Save state")).on_hover_text("F5 saves to this slot while playing.").clicked() {
                        actions.push(Action::SaveState);
                    }
                    if ui
                        .add_enabled(saved.is_some(), egui::Button::new("Load state").min_size(Vec2::new(98.0, 28.0)))
                        .on_hover_text("F8 loads this slot while playing.")
                        .clicked()
                    {
                        actions.push(Action::LoadState);
                    }
                });
                ui.add_space(4.0);
                if wide(ui, "Settings") {
                    self.settings_open = true;
                }
                if let Some(program_id) = program_id.filter(|_| wide(ui, "Settings for this game")) {
                    self.game_settings = Some((program_id, name.to_owned()));
                }
                if wide(ui, "Cheats") {
                    self.cheats_open = true;
                }
                if wide(ui, "Multiplayer") {
                    self.multiplayer_open = true;
                }
                if ui
                    .add_sized([200.0, 28.0], egui::Button::new("Show the camera a picture"))
                    .on_hover_text("Pick a picture for the console's cameras to see, a QR code's say, in place of the computer's camera.")
                    .clicked()
                {
                    actions.push(Action::ChooseCameraPicture);
                }
                if wide(ui, "Copy info for a report") {
                    actions.push(Action::CopyReport);
                }
                if wide(ui, "Back to the library") {
                    actions.push(Action::Library);
                }
                if wide(ui, "Quit") {
                    actions.push(Action::Quit);
                }
                for job in jobs.iter().filter(|job| !job.stage().finished()) {
                    ui.separator();
                    ui.label(&job.name);
                    progress(ui, job);
                }
                if waiting > 0 {
                    ui.label(RichText::new(format!("{waiting} more waiting to recompile")).weak());
                }
            });
        actions
    }

    /// the cheats window over a game, when open, its cheats by name, whether
    /// each is on, and their notes.
    pub fn cheats(&mut self, ctx: &egui::Context, cheats: &[(String, bool, String, bool)]) -> Vec<Action> {
        let mut actions = Vec::new();
        let mut open = self.cheats_open;
        egui::Window::new("Cheats").open(&mut open).default_width(380.0).show(ctx, |ui| {
            let mut checkbox = |ui: &mut egui::Ui, index: usize, name: &str, enabled: bool, notes: &str| {
                let mut on = enabled;
                let mut response = ui.checkbox(&mut on, name);
                if !notes.is_empty() {
                    response = response.on_hover_text(notes);
                }
                if response.changed() {
                    actions.push(Action::Cheat(index, on));
                }
            };
            // Zakuro's own, tested on this build of the game
            if cheats.iter().any(|cheat| cheat.3) {
                ui.label(RichText::new("Enhancements").strong());
                for (index, (name, enabled, notes, _)) in cheats.iter().enumerate().filter(|(_, cheat)| cheat.3) {
                    checkbox(ui, index, name, *enabled, "");
                    // what it does to the game and who made it, in sight
                    // before it is turned on
                    ui.indent(index, |ui| ui.label(RichText::new(notes).weak().small()));
                }
                ui.separator();
            }
            if !cheats.iter().any(|cheat| !cheat.3) {
                ui.label(RichText::new("No cheats yet. Add one below, or put the cheat file Citra or Azahar has for this game in the cheats folder.").weak());
            }
            egui::ScrollArea::vertical().max_height(320.0).show(ui, |ui| {
                for (index, (name, enabled, notes, _)) in cheats.iter().enumerate().filter(|(_, cheat)| !cheat.3) {
                    checkbox(ui, index, name, *enabled, notes);
                }
            });
            ui.separator();
            ui.label("Add a cheat");
            ui.add(egui::TextEdit::singleline(&mut self.cheat_name).hint_text("Name"));
            ui.add(egui::TextEdit::multiline(&mut self.cheat_code).hint_text("00000000 00000000").desired_rows(4).font(egui::TextStyle::Monospace));
            ui.horizontal(|ui| {
                if ui.button("Add").clicked() && !self.cheat_code.trim().is_empty() {
                    actions.push(Action::AddCheat(std::mem::take(&mut self.cheat_name), std::mem::take(&mut self.cheat_code)));
                }
                if ui.button("Reload").on_hover_text("Read the cheats again from their file, after changing it.").clicked() {
                    actions.push(Action::ReloadCheats);
                }
                if ui.button("Open the cheats folder").clicked() {
                    actions.push(Action::CheatsFolder);
                }
            });
        });
        self.cheats_open = open;
        actions
    }

    /// the multiplayer window, when open: the room the player is in, or
    /// joining one and opening one.
    pub fn multiplayer(&mut self, ctx: &egui::Context, settings: &mut Settings, room: Option<&RoomView>) -> Vec<Action> {
        let mut actions = Vec::new();
        let before = settings.clone();
        let mut open = self.multiplayer_open;
        egui::Window::new("Multiplayer").open(&mut open).default_width(380.0).show(ctx, |ui| {
            ui.label(
                RichText::new("Local wireless play with others goes through a room, which one of you opens and the others join over the network.")
                    .weak(),
            );
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label("Nickname");
                ui.add(egui::TextEdit::singleline(&mut settings.nickname).char_limit(10).hint_text("Player"))
                    .on_hover_text("The name others see, and the console's own, which games show. It changes with the next game started.");
            });
            ui.separator();
            match room {
                Some(room) => {
                    ui.label(RichText::new(format!("In the room {}", room.name)).strong());
                    if let Some(port) = room.port {
                        ui.label(format!("Open on port {port}. Others join with this computer's address and that port. Over the internet the router has to let that port through."));
                    }
                    ui.add_space(4.0);
                    for (nickname, game) in &room.members {
                        match game.is_empty() {
                            true => ui.label(nickname),
                            false => ui.label(format!("{nickname}, playing {game}")),
                        };
                    }
                    ui.add_space(4.0);
                    let leave = if room.port.is_some() { "Close the room" } else { "Leave the room" };
                    if ui.button(leave).clicked() {
                        actions.push(Action::LeaveRoom);
                    }
                    if let Some(download_play) = &room.download_play {
                        ui.separator();
                        ui.label(RichText::new("Download Play").strong());
                        match &download_play.doing {
                            Some((doing, progress)) => {
                                ui.label(doing);
                                if let Some(progress) = progress {
                                    ui.add(egui::ProgressBar::new(*progress).show_percentage());
                                }
                                if ui.button("Leave").clicked() {
                                    actions.push(Action::DlpLeave);
                                }
                            }
                            None if download_play.offers.is_empty() => {
                                ui.label(RichText::new("No one in the room offers a game to download. A game offers one from its own Download Play menu.").weak());
                            }
                            None => {
                                for (index, offer) in download_play.offers.iter().enumerate() {
                                    ui.horizontal(|ui| {
                                        if let Some(pixels) = &offer.icon {
                                            let texture = self.dlp_icons.entry(offer.child).or_insert_with(|| {
                                                let image = egui::ColorImage::from_rgba_unmultiplied([ICON_SIZE, ICON_SIZE], pixels);
                                                ui.ctx().load_texture(format!("download play {:016X}", offer.child), image, egui::TextureOptions::LINEAR)
                                            });
                                            ui.add(egui::Image::new(&*texture).fit_to_exact_size(Vec2::splat(32.0)));
                                        }
                                        ui.vertical(|ui| {
                                            ui.label(RichText::new(&offer.title).strong());
                                            ui.label(RichText::new(format!("from {}, {} of {} joined", offer.host, offer.guests, offer.most)).weak());
                                        });
                                        if ui.add_enabled(offer.blocked.is_none(), egui::Button::new("Join")).clicked() {
                                            actions.push(Action::DlpJoin(index));
                                        }
                                    });
                                    if let Some(why) = &offer.blocked {
                                        ui.label(RichText::new(why).weak());
                                    }
                                }
                            }
                        }
                    }
                }
                None => {
                    ui.label(RichText::new("Join a room").strong());
                    ui.horizontal(|ui| {
                        ui.label("Address");
                        ui.add(egui::TextEdit::singleline(&mut settings.room_address).hint_text("192.168.0.10:24873"));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Password");
                        ui.add(egui::TextEdit::singleline(&mut self.room_password).password(true).hint_text("none"));
                    });
                    if ui.add_enabled(!settings.room_address.trim().is_empty(), egui::Button::new("Join")).clicked() {
                        actions.push(Action::JoinRoom);
                    }
                    ui.separator();
                    ui.label(RichText::new("Open a room").strong());
                    ui.horizontal(|ui| {
                        ui.label("Name");
                        ui.add(egui::TextEdit::singleline(&mut settings.room_name).hint_text("Zakuro"));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Port");
                        ui.add(egui::DragValue::new(&mut settings.room_port).range(1024..=65535));
                    });
                    ui.label(RichText::new("The password above, when there is one, is the room's.").weak());
                    if ui.button("Open").clicked() {
                        actions.push(Action::HostRoom);
                    }
                }
            }
        });
        self.multiplayer_open = open;
        if *settings != before {
            actions.push(Action::Settings);
        }
        actions
    }

    /// the settings window, when open. own says the running game has
    /// settings of its own, which go over these.
    pub fn settings(&mut self, ctx: &egui::Context, settings: &mut Settings, own: bool) -> Vec<Action> {
        let mut actions = Vec::new();
        let before = settings.clone();
        let mut open = self.settings_open;
        // taller than a screen has room for, so it scrolls, and it can be
        // dragged to any size that fits
        let room = ctx.content_rect().height() - 16.0;
        egui::Window::new("Settings")
            .id(egui::Id::new("settings"))
            .open(&mut open)
            .vscroll(true)
            .default_size([460.0, room])
            .max_height(room)
            .show(ctx, |ui| {
                if own {
                    ui.label(RichText::new("This game has settings of its own, which go over these. They are in Settings for this game, in the menu.").weak());
                }
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
                rasterizer(ui, &mut settings.hardware_rasterizer);
                ui.add_enabled_ui(settings.hardware_rasterizer, |ui| {
                    resolution(ui, &mut settings.resolution);
                    texture_packs(ui, &mut settings.texture_packs);
                });
                let interpolates = settings.hardware_rasterizer && settings.renderer == Renderer::Vulkan;
                ui.add_enabled_ui(interpolates, |ui| interpolation(ui, &mut settings.frame_interpolation));
                window_scale(ui, &mut settings.scale);
                screens(ui, &mut settings.layout);
                rotation(ui, &mut settings.orientation);
                filter(ui, &mut settings.filter);
                integer_scale(ui, &mut settings.integer_scale);
                top_shape(ui, &mut settings.top_shape);
                stereo_3d(ui, &mut settings.stereo);
                let stereo = settings.stereo != Stereo3d::Off;
                ui.add_enabled_ui(stereo, |ui| depth_3d(ui, &mut settings.depth_3d));
                // on whatever the 3D is here, a game can have its own
                ui.checkbox(&mut settings.swap_eyes, "Swap eyes")
                    .on_hover_text("Each eye gets the other's picture, for viewing cross-eyed, glasses worn the other way round, or a monitor whose rows start with the other eye.");
                ui.label(
                    RichText::new("F9 switches them while playing, F10 between the top and the bottom screen alone.").weak(),
                );
                ui.label(RichText::new("The presenter changes the next time Zakuro starts, the 3D with the next game.").weak());

                ui.separator();
                ui.heading("Emulation");
                language(ui, &mut settings.language);
                egui::ComboBox::from_label("Cameras").selected_text(settings.camera.name()).show_ui(ui, |ui| {
                    for source in CameraSource::ALL {
                        ui.selectable_value(&mut settings.camera, source, source.name());
                    }
                })
                .response
                .on_hover_text("What the console's cameras see. The computer's camera turns on only while a game takes pictures.");
                if settings.camera == CameraSource::Picture {
                    ui.horizontal(|ui| {
                        ui.label("Picture");
                        match &settings.camera_picture {
                            Some(path) => ui.label(path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default()),
                            None => ui.label(RichText::new("none").weak()),
                        };
                        if ui.button("Choose…").clicked() {
                            actions.push(Action::ChooseCameraPicture);
                        }
                    });
                }
                egui::ComboBox::from_label("Microphone").selected_text(settings.microphone.name()).show_ui(ui, |ui| {
                    for source in MicrophoneSource::ALL {
                        ui.selectable_value(&mut settings.microphone, source, source.name());
                    }
                })
                .response
                .on_hover_text("What the console's microphone hears. The computer's microphone is open only while a game listens, which some do all the time, Super Mario 3D Land among them. Holding M blows into it either way, for games that ask you to.");
                code(ui, &mut settings.recompiled);
                ui.label(RichText::new("It changes with the next game started, or Reset.").weak());

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
                ui.heading("Controls");
                egui::Grid::new("controls").num_columns(4).spacing([12.0, 4.0]).show(ui, |ui| {
                    for (i, (name, key)) in settings.keys.all_mut().into_iter().enumerate() {
                        ui.label(name);
                        let text = if self.rebinding == Some(i) { "press a key…".to_owned() } else { key_name(*key) };
                        if ui.add_sized([110.0, 20.0], egui::Button::new(text)).clicked() {
                            self.rebinding = Some(i);
                            self.rebinding_pad = None;
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
                ui.label(RichText::new("Drag with the right mouse button to tilt the console, for games that use the motion sensors.").weak());

                ui.separator();
                ui.heading("Controller");
                egui::Grid::new("controller").num_columns(4).spacing([12.0, 4.0]).show(ui, |ui| {
                    for (i, (name, button)) in settings.pad.all_mut().into_iter().enumerate() {
                        ui.label(name);
                        let text = if self.rebinding_pad == Some(i) { "press a button…" } else { button_name(*button) };
                        if ui.add_sized([110.0, 20.0], egui::Button::new(text)).clicked() {
                            self.rebinding_pad = Some(i);
                            self.rebinding = None;
                        }
                        if i % 2 == 1 {
                            ui.end_row();
                        }
                    }
                });
                ui.label(RichText::new("The circle pad is the left stick, and Home opens the menu over the game.").weak());
                if ui.button("Default controller").clicked() {
                    settings.pad = PadButtons::default();
                    self.rebinding_pad = None;
                }
            });
        self.settings_open = open;
        if !open {
            self.rebinding = None;
            self.rebinding_pad = None;
        }
        if *settings != before {
            actions.push(Action::Settings);
        }
        actions
    }

    /// the window of what a game has of its own, when open.
    pub fn game_settings(&mut self, ctx: &egui::Context, settings: &mut Settings) -> Vec<Action> {
        let mut actions = Vec::new();
        let Some((program_id, name)) = self.game_settings.clone() else { return actions };
        let key = game_key(program_id);
        let mut game = settings.per_game.get(&key).cloned().unwrap_or_default();
        let before = game.clone();
        let general: &Settings = settings;
        let mut open = true;
        let room = ctx.content_rect().height() - 16.0;
        egui::Window::new(format!("Settings for {name}"))
            .id(egui::Id::new("game settings"))
            .open(&mut open)
            .vscroll(true)
            .default_size([420.0, room.min(540.0)])
            .max_height(room)
            .show(ctx, |ui| {
                ui.label(RichText::new("Tick what this game is to have of its own. The rest follows the general settings.").weak());
                ui.separator();
                ui.heading("Graphics");
                own(ui, &mut game.hardware_rasterizer, &general.hardware_rasterizer, rasterizer);
                own(ui, &mut game.resolution, &general.resolution, resolution);
                own(ui, &mut game.texture_packs, &general.texture_packs, texture_packs);
                own(ui, &mut game.frame_interpolation, &general.frame_interpolation, interpolation);
                own(ui, &mut game.scale, &general.scale, window_scale);
                own(ui, &mut game.layout, &general.layout, screens);
                own(ui, &mut game.orientation, &general.orientation, rotation);
                own(ui, &mut game.filter, &general.filter, filter);
                own(ui, &mut game.integer_scale, &general.integer_scale, integer_scale);
                own(ui, &mut game.top_shape, &general.top_shape, top_shape);
                own(ui, &mut game.stereo, &general.stereo, stereo_3d);
                own(ui, &mut game.depth_3d, &general.depth_3d, depth_3d);
                ui.separator();
                ui.heading("Emulation");
                own(ui, &mut game.language, &general.language, language);
                own(ui, &mut game.recompiled, &general.recompiled, code);
                ui.separator();
                ui.label(RichText::new("Where the 3D is drawn, its resolution, the texture packs, the language and what runs the code change the next time the game starts, or with Reset.").weak());
                if ui.add_enabled(game != GameSettings::default(), egui::Button::new("Follow the general settings in everything")).clicked() {
                    game = GameSettings::default();
                }
            });
        if !open {
            self.game_settings = None;
        }
        if game != before {
            match game == GameSettings::default() {
                true => drop(settings.per_game.remove(&key)),
                false => drop(settings.per_game.insert(key, game)),
            }
            actions.push(Action::Settings);
        }
        actions
    }

    /// a message box, when there is something to tell.
    /// the keyboard a game opened to have text typed in, it waits until a
    /// button closes it.
    pub fn keyboard(&mut self, ctx: &egui::Context, request: &Request) -> Option<Action> {
        let fresh = self.typing.as_ref().is_none_or(|(asked, _)| asked != request);
        if fresh {
            self.typing = Some((request.clone(), request.text.clone()));
        }
        let (_, text) = self.typing.as_mut()?;
        let confirm = request.buttons.len() - 1;
        let mut action = None;
        egui::Window::new("Keyboard")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
            .show(ctx, |ui| {
                let field = ui.add(
                    egui::TextEdit::singleline(text)
                        .hint_text(&request.hint)
                        .char_limit(request.max_length)
                        .desired_width(260.0),
                );
                if fresh {
                    field.request_focus();
                }
                let valid = request.accepts(text);
                ui.horizontal(|ui| {
                    for (button, label) in request.buttons.iter().enumerate() {
                        if ui.add_enabled(button != confirm || valid, egui::Button::new(label)).clicked() {
                            action = Some(Action::Keyboard(text.clone(), button));
                        }
                    }
                });
                // enter confirms, when the text will do
                if valid && field.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter)) {
                    action = Some(Action::Keyboard(text.clone(), confirm));
                }
            });
        if action.is_some() {
            self.typing = None;
        }
        action
    }

    /// asks whether to download a compiler, there being none to recompile
    /// with.
    pub fn compiler_offer(&mut self, ctx: &egui::Context) -> Option<Action> {
        let games = self.compiler_offer.clone()?;
        let size = crate::zig::download_size().map_or_else(String::new, |size| format!(", {} MB to download and around 400 MB once unpacked", size / 1_000_000));
        let mut action = None;
        egui::Window::new("Recompile")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
            .show(ctx, |ui| {
                ui.label(format!(
                    "Recompiling needs a C compiler, and Zakuro didn't find one on this computer. It can download Zig, a compiler that needs no installing, into its own folder{size}."
                ));
                ui.horizontal(|ui| {
                    if ui.button("Download it and recompile").clicked() {
                        action = Some(Action::DownloadCompiler(games));
                        self.compiler_offer = None;
                    }
                    if ui.button("Not now").clicked() {
                        self.compiler_offer = None;
                    }
                });
            });
        action
    }

    pub fn message(&mut self, ctx: &egui::Context) {
        let Some(text) = self.message.clone() else { return };
        // a line of recompiles that all fail makes it long, and it scrolls
        // to keep the OK that closes it on the screen
        let room = (ctx.content_rect().height() - 120.0).max(60.0);
        egui::Window::new("Zakuro")
            .id(egui::Id::new("message"))
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().max_height(room).show(ui, |ui| ui.label(text.as_str()));
                ui.horizontal(|ui| {
                    if let Some((_, report)) = self.report.as_ref().filter(|(about, _)| *about == text) {
                        if ui.button("Copy info for a report").clicked() {
                            ui.ctx().copy_text(report.clone());
                        }
                    }
                    if ui.button("OK").clicked() {
                        self.message = None;
                        self.report = None;
                    }
                });
            });
    }
}

/// a row of a game's own settings. ticked, the game has a value of its own,
/// starting from the general one. unticked, it follows the general one,
/// which shows greyed out.
fn own<T: Clone>(ui: &mut egui::Ui, own: &mut Option<T>, general: &T, row: impl FnOnce(&mut egui::Ui, &mut T)) {
    ui.horizontal(|ui| {
        let mut ticked = own.is_some();
        if ui.checkbox(&mut ticked, "").on_hover_text("This game's own, rather than the general setting.").changed() {
            *own = ticked.then(|| general.clone());
        }
        let mut value = own.clone().unwrap_or_else(|| general.clone());
        ui.add_enabled_ui(ticked, |ui| row(ui, &mut value));
        if ticked {
            *own = Some(value);
        }
    });
}

fn rasterizer(ui: &mut egui::Ui, value: &mut bool) {
    ui.checkbox(value, "Draw the 3D on the GPU");
}

fn resolution(ui: &mut egui::Ui, value: &mut u32) {
    ui.add(egui::Slider::new(value, 1..=8).text("Resolution").suffix("x"))
        .on_hover_text("How many times the console's resolution the 3D is drawn at. Past what the window shows, the extra pixels smooth the edges, at a cost to the GPU that grows fast.");
}

fn texture_packs(ui: &mut egui::Ui, value: &mut bool) {
    ui.checkbox(value, "Texture packs")
        .on_hover_text("Draw a game's texture pack, made for Citra or Azahar, put in the textures folder of the game's mods folder.");
}

fn interpolation(ui: &mut egui::Ui, value: &mut bool) {
    ui.checkbox(value, "Frame interpolation")
        .on_hover_text("A game at 30 frames a second gets a picture drawn in between each two of its own, with everything halfway between where it was in both, so it moves at 60. The screens show a 60th of a second late. It takes the 3D drawn on the GPU and the Vulkan presenter.");
}

fn window_scale(ui: &mut egui::Ui, value: &mut u32) {
    ui.add(egui::Slider::new(value, 1..=6).text("Window scale"));
}

fn screens(ui: &mut egui::Ui, value: &mut Screens) {
    egui::ComboBox::from_label("Screens").selected_text(value.name()).show_ui(ui, |ui| {
        for layout in Screens::ALL {
            ui.selectable_value(value, layout, layout.name());
        }
    });
}

fn rotation(ui: &mut egui::Ui, value: &mut Orientation) {
    egui::ComboBox::from_label("Rotation").selected_text(value.name()).show_ui(ui, |ui| {
        for orientation in Orientation::ALL {
            ui.selectable_value(value, orientation, orientation.name());
        }
    });
}

fn filter(ui: &mut egui::Ui, value: &mut Filter) {
    egui::ComboBox::from_label("Filter")
        .selected_text(value.name())
        .show_ui(ui, |ui| {
            for filter in Filter::ALL {
                ui.selectable_value(value, filter, filter.name());
            }
        })
        .response
        .on_hover_text("How the screens' pixels look scaled up: smooth, sharp with soft edges, or plain squares.");
}

fn integer_scale(ui: &mut egui::Ui, value: &mut bool) {
    ui.checkbox(value, "Scale by whole numbers only").on_hover_text("Every pixel gets the same size, with a border around the screens.");
}

fn top_shape(ui: &mut egui::Ui, value: &mut TopShape) {
    egui::ComboBox::from_label("Top screen shape")
        .selected_text(value.name())
        .show_ui(ui, |ui| {
            for shape in TopShape::ALL {
                ui.selectable_value(value, shape, shape.name());
            }
        })
        .response
        .on_hover_text("Shows the top screen wider than the console's. Games still draw it at the console's shape, so only turn this on with a widescreen code for the game, which draws a wider scene to match.");
}

fn stereo_3d(ui: &mut egui::Ui, value: &mut Stereo3d) {
    egui::ComboBox::from_label("3D")
        .selected_text(value.name())
        .show_ui(ui, |ui| {
            for stereo in Stereo3d::ALL {
                ui.selectable_value(value, stereo, stereo.name());
            }
        })
        .response
        .on_hover_text("The top screen in 3D, for games that draw it: red and cyan for glasses of those colors, side by side for 3D TVs and headsets, alternate rows for passive 3D monitors. Off, games draw flat, as with the console's 3D slider down.");
}

fn depth_3d(ui: &mut egui::Ui, value: &mut u32) {
    ui.add(egui::Slider::new(value, 0..=100).text("Depth").suffix("%")).on_hover_text("The console's 3D slider, how deep games draw the 3D.");
}

fn language(ui: &mut egui::Ui, value: &mut Language) {
    egui::ComboBox::from_label("Language")
        .selected_text(value.name())
        .show_ui(ui, |ui| {
            for language in Language::ALL {
                ui.selectable_value(value, language, language.name());
            }
        })
        .response
        .on_hover_text("The console's system language, which titles pick their localised text and files by. It changes with the next game started, or Reset.");
}

/// what runs the game's code.
fn code(ui: &mut egui::Ui, recompiled: &mut bool) {
    ui.radio_value(recompiled, true, "Recompiled code when there is some");
    ui.radio_value(recompiled, false, "Interpreter only");
}

/// a line a GameOverlay keeps over the game, in the bottom right corner.
pub fn corner(ctx: &egui::Context, text: &str) {
    egui::Area::new(egui::Id::new("overlay corner"))
        .anchor(egui::Align2::RIGHT_BOTTOM, Vec2::new(-8.0, -8.0))
        .interactable(false)
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.label(text);
            });
        });
}

/// the games that aren't recompiling or waiting to, each once, the same
/// game being in the folder twice as a .3ds and a .cci say.
fn idle(games: impl Iterator<Item = Wanted>, jobs: &[Job], waiting: &[Wanted]) -> Vec<Wanted> {
    let busy = |id: u64| waiting.iter().any(|wanted| wanted.program_id == id) || jobs.iter().any(|job| job.program_id == id && !job.stage().finished());
    let mut idle: Vec<Wanted> = Vec::new();
    for game in games {
        if !busy(game.program_id) && !idle.iter().any(|wanted| wanted.program_id == game.program_id) {
            idle.push(game);
        }
    }
    idle
}

/// a recompile's progress bar with where it is.
fn progress(ui: &mut egui::Ui, job: &Job) {
    let stage = job.stage();
    let text = match &stage {
        Stage::Downloading(arrived) => format!("downloading the compiler, {:.0}%", arrived * 100.0),
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

    /// the settings stay inside a short screen, scrolling in it instead of
    /// running past its bottom.
    #[test]
    fn the_settings_fit_a_short_screen() {
        let ctx = egui::Context::default();
        let mut menus = Menus { settings_open: true, ..Menus::default() };
        let mut settings = Settings::default();
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, Vec2::new(800.0, 360.0));
        for _ in 0..3 {
            let input = egui::RawInput { screen_rect: Some(screen), ..Default::default() };
            ctx.run_ui(input, |ui| {
                menus.settings(ui.ctx(), &mut settings, false);
            })
            .drop_without_applying_deltas();
        }
        let window = ctx.memory(|memory| memory.area_rect(egui::Id::new("settings"))).unwrap();
        assert!(screen.contains_rect(window), "{window:?} in {screen:?}");
    }

    /// runs a frame of a panel with one row of a game's settings, clicking
    /// where the row starts when asked to.
    fn frame_of_a_row(ctx: &egui::Context, click: bool, mut row: impl FnMut(&mut egui::Ui)) {
        let at = egui::pos2(16.0, 16.0);
        let mut events = vec![egui::Event::PointerMoved(at)];
        if click {
            let press = |pressed| egui::Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE };
            events.extend([press(true), press(false)]);
        }
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, Vec2::new(400.0, 200.0));
        let input = egui::RawInput { screen_rect: Some(screen), events, ..Default::default() };
        let panel = egui::Frame::NONE.inner_margin(8.0);
        ctx.run_ui(input, |ui| drop(panel.show(ui, |ui| row(ui)))).drop_without_applying_deltas();
    }

    /// ticking a row gives the game the general value to start from, and
    /// unticking it has the game follow the general setting again.
    #[test]
    fn ticking_a_row_gives_the_game_a_value_of_its_own() {
        let ctx = egui::Context::default();
        let mut value = None;
        for expected in [Some(3), None] {
            frame_of_a_row(&ctx, false, |ui| own(ui, &mut value, &3u32, resolution));
            frame_of_a_row(&ctx, true, |ui| own(ui, &mut value, &3u32, resolution));
            assert_eq!(value, expected);
        }
    }

    /// a game's settings go in the file only while it has some of its own,
    /// and fit a short screen as the general ones do.
    #[test]
    fn a_games_settings_fit_a_short_screen() {
        let ctx = egui::Context::default();
        let mut menus = Menus { game_settings: Some((0x0004_0000_0012_3400, "A game".to_owned())), ..Menus::default() };
        let mut settings = Settings::default();
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, Vec2::new(800.0, 360.0));
        for _ in 0..3 {
            let input = egui::RawInput { screen_rect: Some(screen), ..Default::default() };
            ctx.run_ui(input, |ui| {
                menus.game_settings(ui.ctx(), &mut settings);
            })
            .drop_without_applying_deltas();
        }
        let window = ctx.memory(|memory| memory.area_rect(egui::Id::new("game settings"))).unwrap();
        assert!(screen.contains_rect(window), "{window:?} in {screen:?}");
        assert!(settings.per_game.is_empty(), "opening the window gives the game nothing");
    }

    /// however long a message gets, its OK stays on the screen.
    #[test]
    fn a_long_message_can_still_be_closed() {
        let ctx = egui::Context::default();
        let mut menus = Menus { message: Some("Recompiling a game failed, it did.\n\n".repeat(200)), ..Menus::default() };
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, Vec2::new(800.0, 480.0));
        for _ in 0..3 {
            let input = egui::RawInput { screen_rect: Some(screen), ..Default::default() };
            ctx.run_ui(input, |ui| menus.message(ui.ctx())).drop_without_applying_deltas();
        }
        let window = ctx.memory(|memory| memory.area_rect(egui::Id::new("message"))).unwrap();
        assert!(screen.contains_rect(window), "{window:?} in {screen:?}");
    }

    /// a game in line, or having its turn, isn't offered again.
    #[test]
    fn games_already_in_line_are_left_out() {
        let wanted = |program_id, path: &str| Wanted { program_id, path: path.into() };
        let games = || [wanted(1, "a"), wanted(2, "b"), wanted(3, "c")].into_iter();
        let ids = |games: Vec<Wanted>| games.into_iter().map(|game| game.program_id).collect::<Vec<_>>();
        assert_eq!(ids(idle(games(), &[], &[])), [1, 2, 3]);
        let twice = idle([wanted(1, "a.3ds"), wanted(2, "b"), wanted(1, "a.cci")].into_iter(), &[], &[]);
        assert_eq!(twice, [wanted(1, "a.3ds"), wanted(2, "b")], "the same game twice in the folder");
        assert_eq!(ids(idle(games(), &[], &[wanted(2, "b")])), [1, 3]);
        let job = Job::at_stage(1, Stage::Compiling { done: 1, total: 2 });
        assert_eq!(ids(idle(games(), std::slice::from_ref(&job), &[wanted(2, "b")])), [3]);
        let job = Job::at_stage(1, Stage::Failed("no".to_owned()));
        assert_eq!(ids(idle(games(), std::slice::from_ref(&job), &[])), [1, 2, 3]);
    }

    #[test]
    fn keys_have_short_names() {
        assert_eq!(key_name(KeyCode::KeyX), "X");
        assert_eq!(key_name(KeyCode::ArrowUp), "Up");
        assert_eq!(key_name(KeyCode::Digit1), "1");
        assert_eq!(key_name(KeyCode::Enter), "Enter");
    }
}
