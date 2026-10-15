//! what the user chose, kept between runs in the system's place for
//! settings.

use std::path::{Path, PathBuf};

use gilrs::Button;
use serde::{Deserialize, Serialize};
use winit::keyboard::KeyCode;
use zakuro_core::services::hid::PadState;
use zakuro_gpu::ScreenLayout;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Renderer {
    #[default]
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
    LargeTop,
}

impl Screens {
    pub const ALL: [Screens; 5] = [Screens::Stacked, Screens::SideBySide, Screens::LargeTop, Screens::TopOnly, Screens::BottomOnly];

    pub fn name(self) -> &'static str {
        match self {
            Screens::Stacked => "Top over bottom",
            Screens::SideBySide => "Screens side by side",
            Screens::TopOnly => "Top screen only",
            Screens::BottomOnly => "Bottom screen only",
            Screens::LargeTop => "Big top screen, small bottom one",
        }
    }

    pub fn screens(self) -> ScreenLayout {
        match self {
            Screens::Stacked => ScreenLayout::Stacked,
            Screens::SideBySide => ScreenLayout::SideBySide,
            Screens::TopOnly => ScreenLayout::TopOnly,
            Screens::BottomOnly => ScreenLayout::BottomOnly,
            Screens::LargeTop => ScreenLayout::LargeTop,
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

/// how the screens' pixels look scaled up to the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Filter {
    #[default]
    Smooth,
    Sharp,
    Pixels,
}

impl Filter {
    pub const ALL: [Filter; 3] = [Filter::Smooth, Filter::Sharp, Filter::Pixels];

    pub fn name(self) -> &'static str {
        match self {
            Filter::Smooth => "Smooth",
            Filter::Sharp => "Sharp",
            Filter::Pixels => "Pixels",
        }
    }

    pub fn screens(self) -> zakuro_gpu::ScreenFilter {
        match self {
            Filter::Smooth => zakuro_gpu::ScreenFilter::Smooth,
            Filter::Sharp => zakuro_gpu::ScreenFilter::Sharp,
            Filter::Pixels => zakuro_gpu::ScreenFilter::Pixels,
        }
    }
}

/// the top screen's shape. a game draws into 400x240 whatever this is, so
/// anything but the console's own only looks right with a widescreen code
/// on, which makes the game draw a wider scene into those pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TopShape {
    #[default]
    Console,
    Wide16x9,
    Wide21x9,
}

impl TopShape {
    pub const ALL: [TopShape; 3] = [TopShape::Console, TopShape::Wide16x9, TopShape::Wide21x9];

    pub fn name(self) -> &'static str {
        match self {
            TopShape::Console => "The console's, 5:3",
            TopShape::Wide16x9 => "16:9",
            TopShape::Wide21x9 => "21:9",
        }
    }

    /// the top screen's width at its height of 240.
    pub fn width(self) -> f32 {
        match self {
            TopShape::Console => 400.0,
            TopShape::Wide16x9 => 240.0 * 16.0 / 9.0,
            TopShape::Wide21x9 => 560.0,
        }
    }
}

/// how the top screen's 3D shows, for games that draw it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stereo3d {
    /// games draw flat, as with the console's 3D slider down.
    #[default]
    Off,
    Anaglyph,
    SideBySide,
    HalfSideBySide,
    Interlaced,
}

impl Stereo3d {
    pub const ALL: [Stereo3d; 5] = [Stereo3d::Off, Stereo3d::Anaglyph, Stereo3d::SideBySide, Stereo3d::HalfSideBySide, Stereo3d::Interlaced];

    pub fn name(self) -> &'static str {
        match self {
            Stereo3d::Off => "Off",
            Stereo3d::Anaglyph => "Red and cyan glasses",
            Stereo3d::SideBySide => "Side by side",
            Stereo3d::HalfSideBySide => "Side by side, half width",
            Stereo3d::Interlaced => "Alternate rows",
        }
    }

    pub fn mode(self) -> zakuro_gpu::StereoMode {
        match self {
            Stereo3d::Off => zakuro_gpu::StereoMode::Off,
            Stereo3d::Anaglyph => zakuro_gpu::StereoMode::Anaglyph,
            Stereo3d::SideBySide => zakuro_gpu::StereoMode::SideBySide,
            Stereo3d::HalfSideBySide => zakuro_gpu::StereoMode::HalfSideBySide,
            Stereo3d::Interlaced => zakuro_gpu::StereoMode::Interlaced,
        }
    }
}

/// the console's system language, which a title reads from cfg:u and picks
/// its localised text and files by, the codes the console numbers them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Language {
    Japanese,
    #[default]
    English,
    French,
    German,
    Italian,
    Spanish,
    ChineseSimplified,
    Korean,
    Dutch,
    Portuguese,
    Russian,
    ChineseTraditional,
}

impl Language {
    pub const ALL: [Language; 12] = [
        Language::Japanese,
        Language::English,
        Language::French,
        Language::German,
        Language::Italian,
        Language::Spanish,
        Language::ChineseSimplified,
        Language::Korean,
        Language::Dutch,
        Language::Portuguese,
        Language::Russian,
        Language::ChineseTraditional,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Language::Japanese => "Japanese",
            Language::English => "English",
            Language::French => "French",
            Language::German => "German",
            Language::Italian => "Italian",
            Language::Spanish => "Spanish",
            Language::ChineseSimplified => "Chinese (Simplified)",
            Language::Korean => "Korean",
            Language::Dutch => "Dutch",
            Language::Portuguese => "Portuguese",
            Language::Russian => "Russian",
            Language::ChineseTraditional => "Chinese (Traditional)",
        }
    }

    /// the number the console's blocks give the language.
    pub fn code(self) -> u8 {
        match self {
            Language::Japanese => 0,
            Language::English => 1,
            Language::French => 2,
            Language::German => 3,
            Language::Italian => 4,
            Language::Spanish => 5,
            Language::ChineseSimplified => 6,
            Language::Korean => 7,
            Language::Dutch => 8,
            Language::Portuguese => 9,
            Language::Russian => 10,
            Language::ChineseTraditional => 11,
        }
    }
}

/// what the console's cameras see.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CameraSource {
    /// the computer's camera, on only while a game takes pictures.
    #[default]
    Webcam,
    /// a picture from a file, a QR code's say.
    Picture,
    /// nothing, black.
    Nothing,
}

impl CameraSource {
    pub const ALL: [CameraSource; 3] = [CameraSource::Webcam, CameraSource::Picture, CameraSource::Nothing];

    pub fn name(self) -> &'static str {
        match self {
            CameraSource::Webcam => "The computer's camera",
            CameraSource::Picture => "A picture",
            CameraSource::Nothing => "Nothing",
        }
    }
}

/// the keys the console's buttons, circle pad and C-stick are on.
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
    pub zl: KeyCode,
    pub zr: KeyCode,
    pub c_up: KeyCode,
    pub c_down: KeyCode,
    pub c_left: KeyCode,
    pub c_right: KeyCode,
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
            zl: KeyCode::Digit1,
            zr: KeyCode::Digit2,
            c_up: KeyCode::KeyT,
            c_down: KeyCode::KeyG,
            c_left: KeyCode::KeyF,
            c_right: KeyCode::KeyH,
        }
    }
}

impl Keys {
    /// whether a console button is on key.
    pub fn binds(&self, key: KeyCode) -> bool {
        self.clone().all_mut().into_iter().any(|(_, bound)| *bound == key)
    }

    /// every binding with the name it goes by, to show and change them.
    pub fn all_mut(&mut self) -> [(&'static str, &mut KeyCode); 22] {
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
            ("ZL", &mut self.zl),
            ("ZR", &mut self.zr),
            ("C-stick up", &mut self.c_up),
            ("C-stick down", &mut self.c_down),
            ("C-stick left", &mut self.c_left),
            ("C-stick right", &mut self.c_right),
        ]
    }
}

/// the controller buttons the console's buttons are on, the face buttons
/// where the 3DS has them rather than by their letters. the circle pad is
/// the left stick and the C-stick the right one.
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
    pub zl: Button,
    pub zr: Button,
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
            zl: Button::LeftTrigger2,
            zr: Button::RightTrigger2,
        }
    }
}

impl PadButtons {
    /// every binding with the name it goes by, to show and change them.
    pub fn all_mut(&mut self) -> [(&'static str, &mut Button); 14] {
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
            ("ZL", &mut self.zl),
            ("ZR", &mut self.zr),
        ]
    }

    /// each of the console's buttons and the controller button it is on.
    pub fn map(&self) -> [(PadState, Button); 14] {
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
            (PadState::ZL, self.zl),
            (PadState::ZR, self.zr),
        ]
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// the folder the library lists games from.
    pub games: Option<PathBuf>,
    /// the console's system language, which titles pick their localised
    /// text and files by.
    #[serde(deserialize_with = "or_default")]
    pub language: Language,
    #[serde(deserialize_with = "or_default")]
    pub renderer: Renderer,
    /// draw the 3D on the GPU rather than in software.
    pub hardware_rasterizer: bool,
    /// how many times the console's resolution the GPU draws the 3D at.
    pub resolution: u32,
    /// draw the texture packs put in games' mods folders.
    pub texture_packs: bool,
    /// draw a picture in between each two of a game at 30 frames a second.
    pub frame_interpolation: bool,
    /// run a game's recompiled code when it has some, rather than
    /// interpreting everything.
    pub recompiled: bool,
    /// window size, times the console's.
    pub scale: u32,
    #[serde(deserialize_with = "or_default")]
    pub layout: Screens,
    /// how the screens' pixels look scaled up.
    #[serde(deserialize_with = "or_default")]
    pub filter: Filter,
    /// scale the screens by whole numbers only, leaving a border.
    pub integer_scale: bool,
    /// how wide the top screen shows.
    #[serde(deserialize_with = "or_default")]
    pub top_shape: TopShape,
    /// how the 3D shows.
    #[serde(deserialize_with = "or_default")]
    pub stereo: Stereo3d,
    /// how deep games draw the 3D, the console's slider, 0 to 100.
    pub depth_3d: u32,
    /// each eye gets the other's picture.
    pub swap_eyes: bool,
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
    /// the enhancements turned on, by name, for each game by its title id.
    pub enhancements: std::collections::BTreeMap<String, Vec<String>>,
    /// what the console's cameras see, and the picture when it is one.
    #[serde(deserialize_with = "or_default")]
    pub camera: CameraSource,
    pub camera_picture: Option<PathBuf>,
    /// the name the player goes by in rooms, the console's own too.
    pub nickname: String,
    /// the number that tells this console apart over local wireless, made
    /// once.
    pub console_id: u64,
    /// the room last joined, and the one opened, its name and port.
    pub room_address: String,
    pub room_name: String,
    pub room_port: u16,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            games: None,
            language: Language::English,
            renderer: Renderer::Vulkan,
            hardware_rasterizer: true,
            resolution: 3,
            texture_packs: true,
            frame_interpolation: false,
            recompiled: true,
            scale: 2,
            layout: Screens::Stacked,
            filter: Filter::Smooth,
            integer_scale: false,
            top_shape: TopShape::Console,
            stereo: Stereo3d::Off,
            depth_3d: 50,
            swap_eyes: false,
            volume: 1.0,
            mute: false,
            show_fps: false,
            background: None,
            background_opacity: 0.35,
            keys: Keys::default(),
            pad: PadButtons::default(),
            enhancements: std::collections::BTreeMap::new(),
            camera: CameraSource::Webcam,
            camera_picture: None,
            nickname: String::new(),
            console_id: 0,
            room_address: String::new(),
            room_name: String::new(),
            room_port: zakuro_core::room::DEFAULT_PORT,
        }
    }
}

/// a setting's value, or its default when the file holds one this build
/// does not know, a newer Zakuro's say, so that one value does not lose all
/// the others.
fn or_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de> + Default,
{
    let value = toml::Value::deserialize(deserializer)?;
    Ok(T::deserialize(value).unwrap_or_default())
}

impl Settings {
    /// where the console's 3D slider is, down while the 3D is off, so games
    /// do not draw a picture nobody sees.
    pub fn slider_3d(&self) -> f32 {
        if self.stereo == Stereo3d::Off {
            0.0
        } else {
            self.depth_3d.min(100) as f32 / 100.0
        }
    }

    /// the 3D the presenter shows.
    pub fn stereo(&self) -> zakuro_gpu::Stereo {
        zakuro_gpu::Stereo { mode: self.stereo.mode(), swap: self.swap_eyes }
    }

    /// the saved settings, or the defaults when there are none.
    pub fn load() -> Settings {
        let Some(path) = path() else { return Settings::default() };
        match std::fs::read_to_string(&path) {
            Ok(text) => toml::from_str(&text).unwrap_or_else(|error| {
                // kept aside, the next save would write over it
                let aside = path.with_extension("toml.bad");
                let _ = std::fs::copy(&path, &aside);
                log::warn!("could not read {}, {error}, using the defaults, it is kept as {}", path.display(), aside.display());
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
            Settings { games: Some("/games".into()), show_fps: true, layout: Screens::TopOnly, language: Language::German, ..Settings::default() };
        settings.keys.a = KeyCode::KeyK;
        settings.pad.a = Button::South;
        let text = toml::to_string_pretty(&settings).unwrap();
        assert_eq!(toml::from_str::<Settings>(&text).unwrap(), settings);
    }

    #[test]
    fn a_value_this_build_does_not_know_keeps_the_others() {
        let settings: Settings = toml::from_str("show_fps = true\nlayout = \"three_screens\"\nfilter = \"crt\"\n").unwrap();
        assert!(settings.show_fps);
        assert_eq!(settings.layout, Screens::default());
        assert_eq!(settings.filter, Filter::default());
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
