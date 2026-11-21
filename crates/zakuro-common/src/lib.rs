//! types and consts every zakuro crate shares

pub mod bits;
pub mod hash;
pub mod memory_map;
pub mod result;

pub use result::{ResultCode, ResultValue};

/// a virtual addr inside an emulated process
pub type VAddr = u32;
/// a physical addr on the 3DS bus
pub type PAddr = u32;

/// both 3DS screens, as the GPU and the presenter see them
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Screen {
    Top,
    Bottom,
}

impl Screen {
    pub const fn width(self) -> u32 {
        match self {
            // framebuffers are stored rotated 90 degrees, the "width" the GPU writes is the panel's physical height
            Screen::Top => 400,
            Screen::Bottom => 320,
        }
    }

    pub const fn height(self) -> u32 {
        240
    }
}

/// which console we pretend to be
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
