//! what the user chose, kept between runs in the system's place for
//! settings.

use std::path::{Path, PathBuf};

use gilrs::Button;
use serde::{Deserialize, Serialize};
use winit::keyboard::KeyCode;
use zakuro_core::services::hid::PadState;
use zakuro_gpu::ScreenLayout;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Renderer {
    Vulkan,
    OpenGl,
}

/// how the screens are arranged in the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Screens {
    #[default]
    Stacked,
    SideBySide,
    TopOnly,
    BottomOnly,
}

impl Screens {
    pub const ALL: [Screens; 4] = [Screens::Stacked, Screens::SideBySide, Screens::TopOnly, Screens::BottomOnly];

    pub fn name(self) -> &'static str {
        match self {
            Screens::Stacked => "Top over bottom",
            Screens::SideBySide => "Side by side",
            Screens::TopOnly => "Top screen only",
            Screens::BottomOnly => "Bottom screen only",
        }
    }

    pub fn screens(self) -> ScreenLayout {
        match self {
            Screens::Stacked => ScreenLayout::Stacked,
            Screens::SideBySide => ScreenLayout::SideBySide,
            Screens::TopOnly => ScreenLayout::TopOnly,
            Screens::BottomOnly => ScreenLayout::BottomOnly,
        }
    }

    /// the one after it, going round.
    pub fn next(self) -> Screens {
        let at = Screens::ALL.iter().position(|&layout| layout == self).unwrap_or(0);
        Screens::ALL[(at + 1) % Screens::ALL.len()]
    }

    /// with one screen showing, the other one alone, else the same.
    pub fn other_screen(self) -> Screens {
        match self {
            Screens::TopOnly => Screens::BottomOnly,
            Screens::BottomOnly => Screens::TopOnly,
            both => both,
        }
    }
}

/// the keys the console's buttons and circle pad are on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Keys {
    pub a: KeyCode,
    pub b: KeyCode,
    pub x: KeyCode,
    pub y: KeyCode,
    pub l: KeyCode,
    pub r: KeyCode,
    pub start: KeyCode,
    pub select: KeyCode,
    pub up: KeyCode,
    pub down: KeyCode,
    pub left: KeyCode,
    pub right: KeyCode,
    pub circle_up: KeyCode,
    pub circle_down: KeyCode,
    pub circle_left: KeyCode,
    pub circle_right: KeyCode,
}

impl Default for Keys {
    fn default() -> Self {
        Keys {
            a: KeyCode::KeyX,
            b: KeyCode::KeyZ,
            x: KeyCode::KeyS,
            y: KeyCode::KeyA,
            l: KeyCode::KeyQ,
            r: KeyCode::KeyW,
            start: KeyCode::Enter,
            select: KeyCode::Backspace,
            up: KeyCode::ArrowUp,
            down: KeyCode::ArrowDown,
            left: KeyCode::ArrowLeft,
            right: KeyCode::ArrowRight,
            circle_up: KeyCode::KeyI,
            circle_down: KeyCode::KeyK,
            circle_left: KeyCode::KeyJ,
            circle_right: KeyCode::KeyL,
        }
    }
}

impl Keys {
    /// every binding with the name it goes by, to show and change them.
    pub fn all_mut(&mut self) -> [(&'static str, &mut KeyCode); 16] {
        [
            ("A", &mut self.a),
            ("B", &mut self.b),
            ("X", &mut self.x),
            ("Y", &mut self.y),
            ("L", &mut self.l),
            ("R", &mut self.r),
            ("Start", &mut self.start),
            ("Select", &mut self.select),
            ("D-pad up", &mut self.up),
            ("D-pad down", &mut self.down),
            ("D-pad left", &mut self.left),
            ("D-pad right", &mut self.right),
            ("Circle pad up", &mut self.circle_up),
            ("Circle pad down", &mut self.circle_down),
            ("Circle pad left", &mut self.circle_left),
            ("Circle pad right", &mut self.circle_right),
        ]
    }
}

/// the controller buttons the console's buttons are on, the face buttons
/// where the 3DS has them rather than by their letters. the circle pad is
/// the left stick.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PadButtons {
    pub a: Button,
    pub b: Button,
    pub x: Button,
    pub y: Button,
    pub l: Button,
    pub r: Button,
    pub start: Button,
    pub select: Button,
    pub up: Button,
    pub down: Button,
    pub left: Button,
    pub right: Button,
}

impl Default for PadButtons {
    fn default() -> Self {
        PadButtons {
            a: Button::East,
            b: Button::South,
            x: Button::North,
            y: Button::West,
            l: Button::LeftTrigger,
            r: Button::RightTrigger,
            start: Button::Start,
            select: Button::Select,
            up: Button::DPadUp,
            down: Button::DPadDown,
            left: Button::DPadLeft,
            right: Button::DPadRight,
        }
    }
}

impl PadButtons {
    /// every binding with the name it goes by, to show and change them.
    pub fn all_mut(&mut self) -> [(&'static str, &mut Button); 12] {
        [
            ("A", &mut self.a),
            ("B", &mut self.b),
            ("X", &mut self.x),
            ("Y", &mut self.y),
            ("L", &mut self.l),
            ("R", &mut self.r),
            ("Start", &mut self.start),
            ("Select", &mut self.select),
            ("D-pad up", &mut self.up),
            ("D-pad down", &mut self.down),
            ("D-pad left", &mut self.left),
            ("D-pad right", &mut self.right),
        ]
    }

    /// each of the console's buttons and the controller button it is on.
    pub fn map(&self) -> [(PadState, Button); 12] {
        [
            (PadState::A, self.a),
            (PadState::B, self.b),
            (PadState::X, self.x),
            (PadState::Y, self.y),
            (PadState::L, self.l),
            (PadState::R, self.r),
            (PadState::START, self.start),
            (PadState::SELECT, self.select),
            (PadState::UP, self.up),
            (PadState::DOWN, self.down),
            (PadState::LEFT, self.left),
            (PadState::RIGHT, self.right),
        ]
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// the folder the library lists games from.
    pub games: Option<PathBuf>,
    pub renderer: Renderer,
    /// draw the 3D on the GPU rather than in software.
    pub hardware_rasterizer: bool,
    /// how many times the console's resolution the GPU draws the 3D at.
    pub resolution: u32,
    /// run a game's recompiled code when it has some, rather than
    /// interpreting everything.
    pub recompiled: bool,
    /// window size, times the console's.
    pub scale: u32,
    pub layout: Screens,
    /// 0 to 1.
    pub volume: f32,
    pub mute: bool,
    pub show_fps: bool,
    /// a picture behind the library.
    pub background: Option<PathBuf>,
    /// how much of it shows, 0 to 1.
    pub background_opacity: f32,
    pub keys: Keys,
    pub pad: PadButtons,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            games: None,
            renderer: Renderer::Vulkan,
            hardware_rasterizer: true,
            resolution: 3,
            recompiled: true,
            scale: 2,
            layout: Screens::Stacked,
            volume: 1.0,
            mute: false,
            show_fps: false,
            background: None,
            background_opacity: 0.35,
            keys: Keys::default(),
            pad: PadButtons::default(),
        }
    }
}

impl Settings {
    /// the saved settings, or the defaults when there are none.
    pub fn load() -> Settings {
        let Some(path) = path() else { return Settings::default() };
        match std::fs::read_to_string(&path) {
            Ok(text) => toml::from_str(&text).unwrap_or_else(|error| {
                log::warn!("could not read {}, {error}, using the defaults", path.display());
                Settings::default()
            }),
            Err(_) => Settings::default(),
        }
    }

    pub fn save(&self) {
        let Some(path) = path() else { return };
        let written = toml::to_string_pretty(self)
            .map_err(|error| error.to_string())
            .and_then(|text| write(&path, &text).map_err(|error| error.to_string()));
        if let Err(error) = written {
            log::warn!("could not save the settings to {}, {error}", path.display());
        }
    }
}

fn write(path: &Path, text: &str) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, text)
}

/// the system's place for a program's settings.
fn path() -> Option<PathBuf> {
    let var = |name: &str| std::env::var_os(name).filter(|value| !value.is_empty()).map(PathBuf::from);
    let dir = if cfg!(windows) {
        var("APPDATA")
    } else if cfg!(target_os = "macos") {
        var("HOME").map(|home| home.join("Library/Application Support"))
    } else {
        var("XDG_CONFIG_HOME").or_else(|| var("HOME").map(|home| home.join(".config")))
    };
    dir.map(|dir| dir.join("zakuro").join("settings.toml"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_come_back_as_they_were_saved() {
        let mut settings =
            Settings { games: Some("/games".into()), show_fps: true, layout: Screens::TopOnly, ..Settings::default() };
        settings.keys.a = KeyCode::KeyK;
        settings.pad.a = Button::South;
        let text = toml::to_string_pretty(&settings).unwrap();
        assert_eq!(toml::from_str::<Settings>(&text).unwrap(), settings);
    }

    #[test]
    fn missing_fields_take_their_defaults() {
        let settings: Settings = toml::from_str("show_fps = true\n[keys]\na = \"KeyK\"\n").unwrap();
        assert!(settings.show_fps);
        assert_eq!(settings.keys.a, KeyCode::KeyK);
        assert_eq!(settings.keys.b, Keys::default().b);
        assert_eq!(settings.scale, 2);
        assert_eq!(settings.layout, Screens::Stacked);
    }

    #[test]
    fn layouts_go_round() {
        let mut layout = Screens::Stacked;
        for _ in 0..Screens::ALL.len() {
            layout = layout.next();
        }
        assert_eq!(layout, Screens::Stacked);
    }
}
