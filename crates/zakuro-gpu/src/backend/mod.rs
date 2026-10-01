//! presentation backends.

#[cfg(feature = "opengl")]
pub mod gl;
#[cfg(feature = "vulkan")]
pub mod vulkan;
#[cfg(feature = "vulkan")]
mod vulkan_overlay;

/// one screen's pixels, already converted to straight RGBA8.
pub struct ScreenImage<'a> {
    pub width: u32,
    pub height: u32,
    pub pixels: &'a [u8],
}

impl ScreenImage<'_> {
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0 || self.pixels.is_empty()
    }
}

/// what is drawn over the screens, a user interface, as textured triangles
/// in window pixels.
#[derive(Default)]
pub struct Overlay {
    /// textures to make or change before drawing.
    pub textures: Vec<OverlayTexture>,
    pub meshes: Vec<OverlayMesh>,
    /// textures that go away once this frame is drawn.
    pub free: Vec<u64>,
}

/// pixels for a texture of the overlay.
pub struct OverlayTexture {
    pub id: u64,
    /// where the pixels go in a texture that exists, none to make it anew at
    /// this size.
    pub offset: Option<[u32; 2]>,
    pub size: [u32; 2],
    /// RGBA, premultiplied by alpha, in sRGB.
    pub pixels: Vec<u8>,
    /// filtered when scaled, rather than nearest.
    pub linear: bool,
}

/// one corner of an overlay triangle.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct OverlayVertex {
    /// window pixels from the top left.
    pub position: [f32; 2],
    pub uv: [f32; 2],
    /// premultiplied sRGB.
    pub color: [u8; 4],
}

/// triangles of the overlay that share a texture and a clip.
pub struct OverlayMesh {
    pub texture: u64,
    /// the part of the window they may draw in, left, top, right, bottom.
    pub clip: [u32; 4],
    pub vertices: Vec<OverlayVertex>,
    pub indices: Vec<u32>,
}

/// where a screen goes in the window, in pixels from the top left.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// how the screens are arranged in the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScreenLayout {
    /// the top screen over the bottom one, as on the console.
    #[default]
    Stacked,
    /// next to each other, the top screen on the left.
    SideBySide,
    /// the top screen alone.
    TopOnly,
}

impl ScreenLayout {
    /// the size the screens take up together at the console's resolution.
    pub fn size(self) -> (u32, u32) {
        match self {
            ScreenLayout::Stacked => (400, 480),
            ScreenLayout::SideBySide => (720, 240),
            ScreenLayout::TopOnly => (400, 240),
        }
    }
}

/// works out where the screens go inside a window, keeping the 3DS's
/// aspect ratio and centring them. the bottom screen has no place when only
/// the top one shows.
pub fn layout(window_width: u32, window_height: u32, arrangement: ScreenLayout) -> (Viewport, Option<Viewport>) {
    let (total_width, total_height) = arrangement.size();
    let (total_width, total_height) = (total_width as f32, total_height as f32);

    let scale = (window_width as f32 / total_width).min(window_height as f32 / total_height);
    let offset_x = (window_width as f32 - total_width * scale) / 2.0;
    let offset_y = (window_height as f32 - total_height * scale) / 2.0;

    let top = Viewport {
        x: offset_x,
        y: offset_y,
        width: 400.0 * scale,
        height: 240.0 * scale,
    };
    let bottom = match arrangement {
        // the bottom screen is narrower, so it is centered under the top one.
        ScreenLayout::Stacked => Some(Viewport {
            x: offset_x + 40.0 * scale,
            y: offset_y + 240.0 * scale,
            width: 320.0 * scale,
            height: 240.0 * scale,
        }),
        ScreenLayout::SideBySide => Some(Viewport {
            x: offset_x + 400.0 * scale,
            y: offset_y,
            width: 320.0 * scale,
            height: 240.0 * scale,
        }),
        ScreenLayout::TopOnly => None,
    };
    (top, bottom)
}

#[derive(Debug, thiserror::Error)]
pub enum PresentError {
    #[error("{0}")]
    Backend(String),
    #[error("the swapchain is out of date and was recreated")]
    OutOfDate,
}

pub trait Presenter {
    fn name(&self) -> &'static str;

    /// draws both screens into the window, and the overlay over them.
    fn present(
        &mut self,
        top: ScreenImage<'_>,
        bottom: ScreenImage<'_>,
        overlay: &Overlay,
    ) -> Result<(), PresentError>;

    /// the window changed size.
    fn resize(&mut self, width: u32, height: u32);

    /// how the screens are arranged from the next frame on.
    fn set_layout(&mut self, arrangement: ScreenLayout);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_keeps_the_aspect_ratio_and_centers() {
        // a window exactly 400x480 needs no scaling or offset.
        let (top, bottom) = layout(400, 480, ScreenLayout::Stacked);
        let bottom = bottom.unwrap();
        assert_eq!(top.x, 0.0);
        assert_eq!(top.y, 0.0);
        assert_eq!(top.width, 400.0);
        assert_eq!(bottom.y, 240.0);
        assert_eq!(bottom.x, 40.0);
        assert_eq!(bottom.width, 320.0);

        // doubling both dimensions doubles the scale.
        let (top, _) = layout(800, 960, ScreenLayout::Stacked);
        assert_eq!(top.width, 800.0);
        assert_eq!(top.height, 480.0);

        // a window that is too wide letterboxes horizontally.
        let (top, _) = layout(1000, 480, ScreenLayout::Stacked);
        assert_eq!(top.width, 400.0);
        assert_eq!(top.x, 300.0);
    }

    #[test]
    fn the_other_layouts_place_the_screens_their_way() {
        // side by side, the bottom screen starts where the top one ends
        let (top, bottom) = layout(1440, 480, ScreenLayout::SideBySide);
        let bottom = bottom.unwrap();
        assert_eq!((top.x, top.y, top.width), (0.0, 0.0, 800.0));
        assert_eq!((bottom.x, bottom.y, bottom.width, bottom.height), (800.0, 0.0, 640.0, 480.0));

        // the top screen alone fills a window of its shape, and a taller
        // one centers it
        let (top, bottom) = layout(800, 960, ScreenLayout::TopOnly);
        assert!(bottom.is_none());
        assert_eq!((top.x, top.y, top.width, top.height), (0.0, 240.0, 800.0, 480.0));
    }
}
