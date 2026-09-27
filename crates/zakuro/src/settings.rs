//! what the user chose, kept between runs in the system's place for
//! settings.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use winit::keyboard::KeyCode;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Renderer {
    Vulkan,
    OpenGl,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// the folder the library lists games from.
    pub games: Option<PathBuf>,
    pub renderer: Renderer,
    /// draw the 3D on the GPU rather than in software.
    pub hardware_rasterizer: bool,
    /// window size, times the console's.
    pub scale: u32,
    /// 0 to 1.
    pub volume: f32,
    pub mute: bool,
    pub show_fps: bool,
    /// a picture behind the library.
    pub background: Option<PathBuf>,
    /// how much of it shows, 0 to 1.
    pub background_opacity: f32,
    /// 3dsrecomp, to recompile games from the library, its program or a
    /// folder holding it, found on the path when not set.
    pub recompiler: Option<PathBuf>,
    pub keys: Keys,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            games: None,
            renderer: Renderer::Vulkan,
            hardware_rasterizer: true,
            scale: 2,
            volume: 1.0,
            mute: false,
            show_fps: false,
            background: None,
            background_opacity: 0.35,
            recompiler: None,
            keys: Keys::default(),
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
        let mut settings = Settings { games: Some("/games".into()), show_fps: true, ..Settings::default() };
        settings.keys.a = KeyCode::KeyK;
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
    }
}
