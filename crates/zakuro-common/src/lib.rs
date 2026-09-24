//! types and constants shared by every Zakuro crate.

pub mod bits;
pub mod memory_map;
pub mod result;

pub use result::{ResultCode, ResultValue};

/// a virtual address inside an emulated process.
pub type VAddr = u32;
/// a physical address on the 3DS bus.
pub type PAddr = u32;

/// both 3DS screens, as the GPU and the presentation layer see them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Screen {
    Top,
    Bottom,
}

impl Screen {
    pub const fn width(self) -> u32 {
        match self {
            // the framebuffers are stored rotated 90 degrees, the "width" the
            // GPU writes is the physical height of the panel.
            Screen::Top => 400,
            Screen::Bottom => 320,
        }
    }

    pub const fn height(self) -> u32 {
        240
    }
}

/// which console we pretend to be.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConsoleModel {
    #[default]
    Old3ds,
    New3ds,
}

impl ConsoleModel {
    pub const fn is_new3ds(self) -> bool {
        matches!(self, ConsoleModel::New3ds)
    }
}
